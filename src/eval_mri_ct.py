#!/usr/bin/env python3
"""Evaluate the MRI <-> CT flow (step 4) on the test split.

For each MRI/CT target direction this script reports:
  - oracle_recon: D_target(E(target))
  - aux_decode:   D_target(E(known))
  - flow_<name>:  D_target(Flow(E(known)))

It also logs latent distances and flow displacement alignment with the true
known->target latent displacement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brats_dataset import BraTS2023AllDataset, get_transforms  # noqa: E402
from eval import compute_generation_metrics  # noqa: E402
from stage1 import RAE  # noqa: E402
from stage2.transport import Sampler, create_transport  # noqa: E402
from utils.dist_utils import cleanup_distributed, setup_distributed  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402
from utils.train_utils import get_autocast_scaler, parse_configs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MRI/CT RAE + flow latent diagnostic.")
    parser.add_argument("--config", type=str, required=True, help="Step-4 flow config.")
    parser.add_argument("--data-path", type=str, required=True, help="MRI/CT JSON path.")
    parser.add_argument("--rae-path", type=str, required=True, help="Exported Step-3 RAE checkpoint.")
    parser.add_argument(
        "--flow-ckpt",
        action="append",
        default=[],
        help="Flow checkpoint to evaluate. Format: name:path or path. Can be repeated.",
    )
    parser.add_argument("--flow-state-key", type=str, default="ema", choices=["ema", "model"],
                        help="Checkpoint state key used for flow weights.")
    parser.add_argument("--include-untrained-flow", action="store_true",
                        help="Also evaluate a freshly initialized DiT/flow model.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--phase", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-modalities", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU eval batch size.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--metric-batch-size", type=int, default=128)
    parser.add_argument("--metrics", type=str, default="rfid,psnr,ssim,is")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="If >0, evaluate only the first N samples.")
    parser.add_argument("--sampler-num-steps", type=str, default="",
                        help="Optional comma-separated sampler num_steps sweep, e.g. '2,4,8,16,25,50'.")
    parser.add_argument("--keep-npzs", action="store_true",
                        help="Keep temporary image shards instead of deleting them.")
    return parser.parse_args()


class RAE4Dec(torch.nn.Module):
    def __init__(self, base_rae: RAE, num_modalities: int):
        super().__init__()
        self.encoder = base_rae.encoder
        self.register_buffer("encoder_mean", base_rae.encoder_mean.clone())
        self.register_buffer("encoder_std", base_rae.encoder_std.clone())
        self.encoder_input_size = base_rae.encoder_input_size
        self.encode = base_rae.encode
        self.reshape_to_2d = getattr(base_rae, "reshape_to_2d", True)
        self.do_normalization = getattr(base_rae, "do_normalization", False)
        self.eps = getattr(base_rae, "eps", 1e-5)
        if self.do_normalization:
            self.register_buffer("latent_mean", base_rae.latent_mean.clone())
            self.register_buffer("latent_var", base_rae.latent_var.clone())
        else:
            self.register_buffer("latent_mean", None, persistent=False)
            self.register_buffer("latent_var", None, persistent=False)
        self.num_modalities = num_modalities
        self.decoders = torch.nn.ModuleList([deepcopy(base_rae.decoder) for _ in range(num_modalities)])

    def decode(self, z: torch.Tensor, m: int) -> torch.Tensor:
        decoder = self.decoders[m]
        if self.do_normalization:
            z = z * torch.sqrt(self.latent_var + self.eps) + self.latent_mean
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            z = z.view(b, c, h * w).transpose(1, 2)
        output = decoder(z, drop_cls_token=False).logits
        x_rec = decoder.unpatchify(output)
        x_rec = x_rec * self.encoder_std + self.encoder_mean
        return x_rec


class ScalarStats:
    def __init__(self, device: torch.device):
        self.device = device
        self.data: Dict[str, Dict[str, torch.Tensor]] = {}

    def add(self, name: str, values: torch.Tensor) -> None:
        values = values.detach().float().flatten()
        if values.numel() == 0:
            return
        current = {
            "count": torch.tensor(float(values.numel()), device=self.device),
            "sum": values.sum().to(self.device),
            "sumsq": values.square().sum().to(self.device),
            "min": values.min().to(self.device),
            "max": values.max().to(self.device),
        }
        if name not in self.data:
            self.data[name] = current
            return
        dst = self.data[name]
        dst["count"] += current["count"]
        dst["sum"] += current["sum"]
        dst["sumsq"] += current["sumsq"]
        dst["min"] = torch.minimum(dst["min"], current["min"])
        dst["max"] = torch.maximum(dst["max"], current["max"])

    def reduce(self) -> None:
        if not dist.is_initialized():
            return
        for stat in self.data.values():
            for key in ("count", "sum", "sumsq"):
                dist.all_reduce(stat[key], op=dist.ReduceOp.SUM)
            dist.all_reduce(stat["min"], op=dist.ReduceOp.MIN)
            dist.all_reduce(stat["max"], op=dist.ReduceOp.MAX)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name, stat in self.data.items():
            count = max(float(stat["count"].item()), 1.0)
            mean = float((stat["sum"] / count).item())
            var = max(float((stat["sumsq"] / count).item()) - mean * mean, 0.0)
            out[name] = {
                "mean": mean,
                "std": math.sqrt(var),
                "min": float(stat["min"].item()),
                "max": float(stat["max"].item()),
                "count": int(count),
            }
        return out


def cfg_to_dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if cfg is not None else {}


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_named_ckpt(spec: str, fallback_idx: int) -> Tuple[str, str]:
    if ":" in spec and not spec.startswith("/"):
        name, path = spec.split(":", 1)
    else:
        path = spec
        name = Path(path).stem if path else f"ckpt{fallback_idx}"
    name = name.strip().replace("/", "_")
    return name, path.strip()


def to_uint8_nhwc(x: torch.Tensor) -> np.ndarray:
    return x.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


def latent_flat(z: torch.Tensor) -> torch.Tensor:
    return z.float().flatten(1)


def add_latent_stats(stats: ScalarStats, prefix: str, z_target: torch.Tensor, z_known: torch.Tensor,
                     z_flow: Optional[torch.Tensor] = None) -> None:
    target_f = latent_flat(z_target)
    known_f = latent_flat(z_known)
    target_delta = target_f - known_f
    target_delta_norm = torch.linalg.vector_norm(target_delta, dim=1)

    stats.add(f"{prefix}/known_target_cos", F.cosine_similarity(known_f, target_f, dim=1, eps=1e-6))
    stats.add(f"{prefix}/known_target_mse", (known_f - target_f).square().mean(dim=1))
    stats.add(f"{prefix}/known_to_target_norm", target_delta_norm)

    if z_flow is None:
        return

    flow_f = latent_flat(z_flow)
    flow_to_target = flow_f - target_f
    flow_delta = flow_f - known_f
    flow_delta_norm = torch.linalg.vector_norm(flow_delta, dim=1)
    flow_to_target_norm = torch.linalg.vector_norm(flow_to_target, dim=1)

    stats.add(f"{prefix}/flow_target_cos", F.cosine_similarity(flow_f, target_f, dim=1, eps=1e-6))
    stats.add(f"{prefix}/flow_target_mse", flow_to_target.square().mean(dim=1))
    stats.add(f"{prefix}/flow_to_target_norm", flow_to_target_norm)
    stats.add(
        f"{prefix}/flow_to_target_norm_over_known_to_target",
        flow_to_target_norm / (target_delta_norm + 1e-6),
    )
    stats.add(f"{prefix}/flow_delta_norm", flow_delta_norm)
    stats.add(
        f"{prefix}/flow_delta_norm_over_target_delta",
        flow_delta_norm / (target_delta_norm + 1e-6),
    )
    stats.add(
        f"{prefix}/flow_delta_target_delta_cos",
        F.cosine_similarity(flow_delta, target_delta, dim=1, eps=1e-6),
    )
    target_norm_sq = target_delta.square().sum(dim=1).clamp_min(1e-6)
    alpha = (flow_delta * target_delta).sum(dim=1) / target_norm_sq
    parallel = alpha[:, None] * target_delta
    residual = flow_delta - parallel
    residual_norm = torch.linalg.vector_norm(residual, dim=1)
    target_norm = torch.sqrt(target_norm_sq)
    stats.add(f"{prefix}/flow_parallel_alpha", alpha)
    stats.add(
        f"{prefix}/flow_parallel_norm_over_target_delta",
        torch.linalg.vector_norm(parallel, dim=1) / (target_norm + 1e-6),
    )
    stats.add(
        f"{prefix}/flow_residual_norm_over_target_delta",
        residual_norm / (target_norm + 1e-6),
    )
    stats.add(
        f"{prefix}/flow_residual_norm_over_flow_delta",
        residual_norm / (flow_delta_norm + 1e-6),
    )


def write_mode_shards(
    out_dir: Path,
    target_idx: int,
    mode_to_gens: Dict[str, List[np.ndarray]],
    refs: List[np.ndarray],
    rank: int,
) -> None:
    ref_arr = np.concatenate(refs, axis=0)
    for mode, chunks in mode_to_gens.items():
        mode_dir = out_dir / "npz_shards" / f"target_{target_idx}" / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        gen_arr = np.concatenate(chunks, axis=0)
        np.savez_compressed(mode_dir / f"rank_{rank:02d}.npz", gen=gen_arr, ref=ref_arr)


def compute_mode_metrics(
    out_dir: Path,
    target_idx: int,
    mode: str,
    world_size: int,
    max_samples: int,
    metrics: List[str],
    metric_batch_size: int,
    device: torch.device,
    keep_npzs: bool,
) -> Dict[str, float]:
    mode_dir = out_dir / "npz_shards" / f"target_{target_idx}" / mode
    gens, refs = [], []
    for rank in range(world_size):
        shard = np.load(mode_dir / f"rank_{rank:02d}.npz")
        gens.append(shard["gen"])
        refs.append(shard["ref"])
    gen_arr = np.concatenate(gens, axis=0)
    ref_arr = np.concatenate(refs, axis=0)
    if max_samples > 0:
        gen_arr = gen_arr[:max_samples]
        ref_arr = ref_arr[:max_samples]
    result = compute_generation_metrics(
        ref_arr,
        gen_arr,
        device,
        batch_size=metric_batch_size,
        metrics_to_compute=metrics,
    )
    if not keep_npzs:
        shutil.rmtree(mode_dir, ignore_errors=True)
    return {k: float(v) for k, v in result.items()}


@torch.no_grad()
def eval_direct_modes(
    args: argparse.Namespace,
    dataset,
    target_idx: int,
    known_idxs: List[int],
    rae: RAE4Dec,
    rank: int,
    world_size: int,
    device: torch.device,
    autocast_kwargs: dict,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    stats = ScalarStats(device)
    refs: List[np.ndarray] = []
    mode_to_gens: Dict[str, List[np.ndarray]] = {"oracle_recon": [], "aux_decode": []}

    indices = make_rank_indices(len(dataset), rank, world_size, args.max_samples)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    iterator: Iterable = tqdm(loader, desc=f"target {target_idx} direct", disable=(rank != 0), file=sys.stdout)
    for batch in iterator:
        imgs = [x.to(device, non_blocking=True) for x in batch]
        x_target = imgs[target_idx]
        with autocast(**autocast_kwargs):
            z_target = rae.encode(x_target)
            z_known = torch.stack([rae.encode(imgs[k]) for k in known_idxs], dim=0).mean(dim=0)
            oracle = rae.decode(z_target, target_idx).clamp(0, 1)
            aux = rae.decode(z_known, target_idx).clamp(0, 1)

        refs.append(to_uint8_nhwc(x_target))
        mode_to_gens["oracle_recon"].append(to_uint8_nhwc(oracle))
        mode_to_gens["aux_decode"].append(to_uint8_nhwc(aux))
        add_latent_stats(stats, f"target_{target_idx}/aux", z_target, z_known)

    write_mode_shards(Path(args.output_dir), target_idx, mode_to_gens, refs, rank)
    stats.reduce()
    if dist.is_initialized():
        dist.barrier()
    return {"latent": stats.to_dict()}


@torch.no_grad()
def eval_flow_mode(
    args: argparse.Namespace,
    dataset,
    target_idx: int,
    known_idxs: List[int],
    mode_name: str,
    flow_model: torch.nn.Module,
    sample_fn,
    rae: RAE4Dec,
    use_classes: bool,
    rank: int,
    world_size: int,
    device: torch.device,
    autocast_kwargs: dict,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    stats = ScalarStats(device)
    refs: List[np.ndarray] = []
    mode_to_gens: Dict[str, List[np.ndarray]] = {mode_name: []}
    cond_id = target_idx * 16 + sum(2 ** i for i in known_idxs)

    indices = make_rank_indices(len(dataset), rank, world_size, args.max_samples)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    iterator: Iterable = tqdm(loader, desc=f"target {target_idx} {mode_name}", disable=(rank != 0), file=sys.stdout)
    for batch in iterator:
        imgs = [x.to(device, non_blocking=True) for x in batch]
        x_target = imgs[target_idx]
        with autocast(**autocast_kwargs):
            z_target = rae.encode(x_target)
            z_known = torch.stack([rae.encode(imgs[k]) for k in known_idxs], dim=0).mean(dim=0)
            model_kwargs = {}
            if use_classes:
                model_kwargs["y"] = torch.full((z_known.size(0),), cond_id, dtype=torch.long, device=device)
            z_flow = sample_fn(z_known, flow_model.forward, **model_kwargs)[-1]
            decoded = rae.decode(z_flow, target_idx).clamp(0, 1)

        refs.append(to_uint8_nhwc(x_target))
        mode_to_gens[mode_name].append(to_uint8_nhwc(decoded))
        add_latent_stats(stats, f"target_{target_idx}/{mode_name}", z_target, z_known, z_flow)

    write_mode_shards(Path(args.output_dir), target_idx, mode_to_gens, refs, rank)
    stats.reduce()
    if dist.is_initialized():
        dist.barrier()
    return {"latent": stats.to_dict()}


def make_rank_indices(dataset_len: int, rank: int, world_size: int, max_samples: int) -> List[int]:
    n = min(dataset_len, max_samples) if max_samples > 0 else dataset_len
    chunk = n // world_size
    start = rank * chunk
    end = (rank + 1) * chunk if rank < world_size - 1 else n
    return list(range(start, end))


def load_rae(rae_config, rae_path: str, num_modalities: int, device: torch.device) -> RAE4Dec:
    if "ckpt" in rae_config:
        del rae_config["ckpt"]
    base_rae: RAE = instantiate_from_config(rae_config).to(device)
    rae = RAE4Dec(base_rae, num_modalities=num_modalities).to(device)
    ckpt = torch.load(rae_path, map_location="cpu")
    sd = ckpt.get("model", ckpt.get("ema", ckpt))
    sd = dict(sd)
    for key in ("decoder_pred.weight", "decoder_pred.bias"):
        sd.pop(key, None)
    rae.load_state_dict(sd, strict=True)
    rae.eval().requires_grad_(False)
    return rae


def make_flow_model(model_config, ckpt_path: Optional[str], state_key: str, device: torch.device) -> torch.nn.Module:
    model = instantiate_from_config(model_config).to(device).eval()
    model.requires_grad_(False)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if state_key not in ckpt:
            raise KeyError(f"Checkpoint {ckpt_path} does not contain key {state_key!r}.")
        model.load_state_dict(ckpt[state_key], strict=True)
    return model


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic is intended to run on CUDA GPUs.")

    rank, world_size, device = setup_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "npz_shards").mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    full_cfg = OmegaConf.load(args.config)
    rae_config, model_config, transport_config, sampler_config, _, misc_config, _, _ = parse_configs(full_cfg)
    if rae_config is None or model_config is None or transport_config is None:
        raise ValueError("Config must contain stage_1, stage_2, and transport sections.")
    if "ckpt" in full_cfg.get("stage_1", {}):
        del full_cfg["stage_1"]["ckpt"]

    _, autocast_kwargs = get_autocast_scaler(args)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    misc = cfg_to_dict(misc_config)
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    transport = create_transport(**cfg_to_dict(transport_config).get("params", {}), time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)
    sampler_cfg = cfg_to_dict(sampler_config)
    sampler_mode = str(sampler_cfg.get("mode", "ODE")).upper()
    sampler_params = dict(sampler_cfg.get("params", {}))
    sampler_steps = parse_int_list(args.sampler_num_steps)
    sampler_variants = []
    if sampler_steps:
        for num_steps in sampler_steps:
            params = dict(sampler_params)
            params["num_steps"] = num_steps
            if sampler_mode == "ODE":
                sampler_variants.append((f"s{num_steps}", sampler.sample_ode(**params)))
            elif sampler_mode == "SDE":
                sampler_variants.append((f"s{num_steps}", sampler.sample_sde(**params)))
            else:
                raise NotImplementedError(f"Unsupported sampler mode: {sampler_mode}")
    else:
        if sampler_mode == "ODE":
            sampler_variants.append(("", sampler.sample_ode(**sampler_params)))
        elif sampler_mode == "SDE":
            sampler_variants.append(("", sampler.sample_sde(**sampler_params)))
        else:
            raise NotImplementedError(f"Unsupported sampler mode: {sampler_mode}")

    use_classes = model_config.get("params", {}).get("num_classes", None) is not None
    rae = load_rae(rae_config, args.rae_path, args.num_modalities, device)
    dataset = BraTS2023AllDataset(args.data_path, args.phase, get_transforms("val"), num_modalities=args.num_modalities)
    if rank == 0:
        print(f"[diagnose] dataset={args.data_path} phase={args.phase} size={len(dataset)} world_size={world_size}")
        print(f"[diagnose] RAE={args.rae_path}")

    flow_specs: List[Tuple[str, Optional[str]]] = []
    if args.include_untrained_flow:
        flow_specs.append(("flow_untrained", None))
    for i, spec in enumerate(args.flow_ckpt):
        name, path = parse_named_ckpt(spec, i)
        flow_specs.append((f"flow_{name}", path))

    summary: Dict[str, object] = {
        "config": args.config,
        "data_path": args.data_path,
        "rae_path": args.rae_path,
        "phase": args.phase,
        "max_samples": args.max_samples,
        "world_size": world_size,
        "sampler_mode": sampler_mode,
        "sampler_num_steps": sampler_steps or [sampler_params.get("num_steps", None)],
        "targets": {},
    }

    for target_idx in range(args.num_modalities):
        known_idxs = [i for i in range(args.num_modalities) if i != target_idx]
        target_key = f"target_{target_idx}"
        if rank == 0:
            print(f"\n[diagnose] target={target_idx}, known={known_idxs}")
        direct_stats = eval_direct_modes(
            args, dataset, target_idx, known_idxs, rae, rank, world_size, device, autocast_kwargs
        )
        target_summary: Dict[str, object] = {
            "known_idxs": known_idxs,
            "image_metrics": {},
            "latent_stats": direct_stats["latent"],
        }
        if rank == 0:
            for mode in ("oracle_recon", "aux_decode"):
                print(f"[metrics] target={target_idx} mode={mode}")
                target_summary["image_metrics"][mode] = compute_mode_metrics(
                    output_dir, target_idx, mode, world_size, args.max_samples,
                    metrics, args.metric_batch_size, device, args.keep_npzs,
                )

        for mode_name, ckpt_path in flow_specs:
            if rank == 0:
                print(f"[diagnose] loading {mode_name}: {ckpt_path or 'fresh init'}")
            flow_model = make_flow_model(model_config, ckpt_path, args.flow_state_key, device)
            for sampler_tag, sample_fn in sampler_variants:
                eval_mode_name = f"{mode_name}_{sampler_tag}" if sampler_tag else mode_name
                flow_stats = eval_flow_mode(
                    args, dataset, target_idx, known_idxs, eval_mode_name, flow_model, sample_fn,
                    rae, use_classes, rank, world_size, device, autocast_kwargs,
                )
                if rank == 0:
                    target_summary["latent_stats"].update(flow_stats["latent"])
                    print(f"[metrics] target={target_idx} mode={eval_mode_name}")
                    target_summary["image_metrics"][eval_mode_name] = compute_mode_metrics(
                        output_dir, target_idx, eval_mode_name, world_size, args.max_samples,
                        metrics, args.metric_batch_size, device, args.keep_npzs,
                    )
                if dist.is_initialized():
                    dist.barrier()
            del flow_model
            torch.cuda.empty_cache()
            if dist.is_initialized():
                dist.barrier()

        if rank == 0:
            summary["targets"][target_key] = target_summary
            with (output_dir / "diagnostic_summary.json").open("w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

    if rank == 0:
        # Add mean metrics over targets for quick reading.
        targets = summary["targets"]
        mean_metrics: Dict[str, Dict[str, float]] = {}
        modes = sorted({
            mode
            for target in targets.values()
            for mode in target["image_metrics"].keys()
        })
        for mode in modes:
            mean_metrics[mode] = {}
            for metric in metrics:
                vals = [
                    target["image_metrics"][mode][metric]
                    for target in targets.values()
                    if mode in target["image_metrics"] and metric in target["image_metrics"][mode]
                ]
                if vals:
                    mean_metrics[mode][metric] = float(sum(vals) / len(vals))
        summary["mean_image_metrics"] = mean_metrics
        with (output_dir / "diagnostic_summary.json").open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        if not args.keep_npzs:
            shutil.rmtree(output_dir / "npz_shards", ignore_errors=True)
        print(f"\n[diagnose] wrote {output_dir / 'diagnostic_summary.json'}")

    if dist.is_initialized():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()

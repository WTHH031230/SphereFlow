import argparse
import logging
import math
import os
import random
from time import time
from pathlib import Path
from copy import deepcopy
import itertools
from glob import glob
import json as json_module
import sys

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset as TorchSubset
from torch.cuda.amp import autocast
from omegaconf import OmegaConf

##### model imports
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler, ModelType

##### general utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.dist_utils import setup_distributed, cleanup_distributed

##### Eval utils
from eval import compute_generation_metrics

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-2 flow model all modality combinations.")
    parser.add_argument("--config", type=str, required=True, help="YAML config.")
    parser.add_argument("--data-path", type=Path, required=True, help="BraTS JSON path.")
    parser.add_argument("--rae-path", type=str, required=True, help="Path to trained RAE4Dec checkpoint.")
    parser.add_argument("--flow-ckpt", type=str, required=True, help="Path to trained flow model checkpoint.")
    parser.add_argument("--results-dir", type=str, default="eval_results")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--sampler-mode", type=str, choices=["ODE", "SDE"], default=None,
                        help="Override sampler mode from config (ODE or SDE).")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples drawn per input for best-of-N selection by SSIM. "
                             "Only meaningful for SDE (stochastic); ODE is deterministic so >1 has no effect.")
    parser.add_argument("--max-eval-samples", type=int, default=None,
                        help="Cap test set to the first N samples (for multi-GPU smoke / quick validation).")
    parser.add_argument("--only-combo", type=str, default=None,
                        help='Evaluate only one modality combo: "target-known1-known2-..." e.g. "0-1-2-3".')
    parser.add_argument("--sde-diffusion-norm", type=float, default=None,
                        help="SDE only: set diffusion_norm (>0 for stochastic; omit to use transport default).")
    return parser.parse_args()


class RAE4Dec(torch.nn.Module):
    def __init__(self, base_rae: RAE):
        super().__init__()
        self.encoder = base_rae.encoder
        self.encoder_mean = base_rae.encoder_mean
        self.encoder_std = base_rae.encoder_std
        self.encoder_input_size = base_rae.encoder_input_size
        self.encode = base_rae.encode # shares noise/scale logic
        
        self.reshape_to_2d = getattr(base_rae, "reshape_to_2d", True)
        self.do_normalization = getattr(base_rae, "do_normalization", False)

        self.decoders = torch.nn.ModuleList([deepcopy(base_rae.decoder) for _ in range(4)])

    def decode(self, z, m):
        decoder = self.decoders[m]
        if self.do_normalization:
            if hasattr(self, 'latent_mean') and hasattr(self, 'latent_var'):
                z = z * torch.sqrt(self.latent_var + 1e-5) + self.latent_mean
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
        output = decoder(z, drop_cls_token=False).logits
        x_rec = decoder.unpatchify(output)
        x_rec = x_rec * self.encoder_std + self.encoder_mean
        return x_rec


class EvalWrapper(torch.utils.data.Dataset):
    def __init__(self, ds, target, known, cond_id_val, full_rae, device):
        self.ds = ds
        self.target = target
        self.known = known
        self.cond_id_val = cond_id_val
        self.full_rae = full_rae
        self.device = device
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        imgs = self.ds[idx]
        x_target = imgs[self.target]
        x_known_list = [imgs[i] for i in self.known]
        x_known_stack = torch.stack(x_known_list, dim=0)
        return x_target, x_known_stack, torch.tensor(self.cond_id_val, dtype=torch.long)


@torch.no_grad()
def evaluate_flow_all_comb(
    model_fn, sample_fn, rae_model, eval_ds, target_idx, known_idxs,
    cond_id_val, latent_sz, micro_bs, ac_kwargs, exp_dir,
    metrics_list, rank, world_size, device, ref_npz_path=None, num_samples=1
):
    import numpy as np
    from torch.utils.data import Subset
    from tqdm import tqdm

    temp_dir = os.path.join(exp_dir, "eval_npzs")
    if rank == 0: os.makedirs(temp_dir, exist_ok=True)
    if world_size > 1: dist.barrier()

    N = len(eval_ds)
    chunk = N // world_size
    start = rank * chunk
    end = (rank + 1) * chunk if rank < world_size - 1 else N

    subset = Subset(eval_ds, list(range(start, end)))
    eval_loader = DataLoader(
        subset, batch_size=micro_bs, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False
    )

    generations = []
    references = []
    iterator = tqdm(eval_loader, desc=f"[Rank {rank}] Sampling target={target_idx} known={known_idxs}", file=sys.stdout) if rank == 0 else eval_loader

    if num_samples > 1:
        from torchmetrics.functional.image.ssim import structural_similarity_index_measure as _ssim_fn

    with torch.inference_mode():
        for batch in iterator:
            x_target = batch[0].to(device, non_blocking=True)
            x_known_stack = batch[1].to(device, non_blocking=True)
            cond_ids = batch[2].to(device, non_blocking=True)

            B, num_known = x_known_stack.shape[0], x_known_stack.shape[1]
            z_known_list = []
            for ki in range(num_known):
                z_known_list.append(rae_model.encode(x_known_stack[:, ki]))
            z_known = torch.stack(z_known_list, dim=0).mean(dim=0)

            if num_samples > 1:
                # Best-of-N selection by per-image SSIM against the reference
                x_ref_f = x_target.clamp(0, 1).float()
                best_decoded = None
                best_ssim = None
                for _ in range(num_samples):
                    with autocast(**ac_kwargs):
                        s = sample_fn(z_known, model_fn, y=cond_ids)[-1]
                    cand = rae_model.decode(s, target_idx).clamp(0, 1).float()
                    scores = _ssim_fn(cand, x_ref_f, data_range=1.0, reduction="none")  # [B]
                    if best_decoded is None:
                        best_decoded, best_ssim = cand, scores
                    else:
                        improved = scores > best_ssim          # [B] bool
                        sel = improved.view(B, 1, 1, 1).expand_as(cand)
                        best_decoded = torch.where(sel, cand, best_decoded)
                        best_ssim = torch.where(improved, scores, best_ssim)
                decoded = best_decoded
            else:
                with autocast(**ac_kwargs):
                    samples = sample_fn(z_known, model_fn, y=cond_ids)[-1]
                    decoded = rae_model.decode(samples, target_idx).clamp(0, 1)

            gen_np = decoded.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            ref_np = x_target.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            for img in gen_np: generations.append(img)
            for img in ref_np: references.append(img)

    generations = np.stack(generations) if len(generations)>0 else np.zeros((0,))
    references = np.stack(references) if len(references)>0 else np.zeros((0,))
    
    shard_path = os.path.join(temp_dir, f"gen_target{target_idx}_known{'-'.join(map(str,known_idxs))}_{rank:02d}.npz")
    if len(generations) > 0:
        np.savez(shard_path, arr_0=generations, ref=references)
    
    if world_size > 1: dist.barrier()

    metrics = None
    if rank == 0:
        all_gens, all_refs = [], []
        for r in range(world_size):
            shard_file = os.path.join(temp_dir, f"gen_target{target_idx}_known{'-'.join(map(str,known_idxs))}_{r:02d}.npz")
            if os.path.exists(shard_file):
                shard_data = np.load(shard_file)
                all_gens.append(shard_data["arr_0"])
                all_refs.append(shard_data["ref"])

        combined_gens = np.concatenate(all_gens, axis=0)[:N]
        combined_refs = np.concatenate(all_refs, axis=0)[:N]

        metrics = compute_generation_metrics(
            combined_refs, combined_gens, device,
            batch_size=128, metrics_to_compute=metrics_list
        )
        for r in range(world_size):
            shard_file = os.path.join(temp_dir, f"gen_target{target_idx}_known{'-'.join(map(str,known_idxs))}_{r:02d}.npz")
            if os.path.exists(shard_file): os.remove(shard_file)

    if world_size > 1: dist.barrier()
    return metrics


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires at least one GPU.")
    
    rank, world_size, device = setup_distributed()
    full_cfg = OmegaConf.load(args.config)
    
    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
        eval_config
    ) = parse_configs(full_cfg)

    # Clear predefined ckpt so instantiate_from_config doesn't auto-load
    if "ckpt" in rae_config: del rae_config["ckpt"]

    def to_dict(cfg): return OmegaConf.to_container(cfg, resolve=True) if cfg else {}
    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    training_cfg = to_dict(training_config)

    os.makedirs(args.results_dir, exist_ok=True)
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # -------------------------------------------------------------------------
    # Training Parameters Alignment
    # -------------------------------------------------------------------------
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    
    global_batch_size = training_cfg.get("global_batch_size", None)
    if global_batch_size is not None:
        micro_batch_size = int(global_batch_size) // (world_size * grad_accum_steps)
    else:
        micro_batch_size = int(training_cfg.get("batch_size", 16))

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scaler, autocast_kwargs = get_autocast_scaler(args)

    # -------------------------------------------------------------------------
    # Model Init
    # -------------------------------------------------------------------------
    base_rae = instantiate_from_config(rae_config).to(device)
    rae = RAE4Dec(base_rae).to(device)
    
    if rank == 0: logger.info(f"Loading Stage-2 RAE weights from {args.rae_path}")
    sd = torch.load(args.rae_path, map_location="cpu")
    load_result = rae.load_state_dict(sd.get("model", sd.get("ema", sd)), strict=False)
    if rank == 0 and load_result.unexpected_keys:
        logger.info(f"Ignored unexpected keys (likely decoder_pred alias): {load_result.unexpected_keys}")
    del sd  # free checkpoint memory
    torch.cuda.empty_cache()
    rae.eval().requires_grad_(False)

    ema_model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    if rank == 0: logger.info(f"Loading Flow Checkpoint from {args.flow_ckpt}")
    ema_model.load_state_dict(torch.load(args.flow_ckpt, map_location="cpu")["ema"])
    ema_model.eval().requires_grad_(False)
    
    # -------------------------------------------------------------------------
    # Data & Transport
    # -------------------------------------------------------------------------
    import sys
    if os.getcwd() not in sys.path: sys.path.insert(0, os.getcwd())
    from brats_dataset import BraTS2023AllDataset, get_transforms

    # Evaluation Config
    eval_metrics = ["rfid", "psnr", "ssim", "is"]
    reference_npz_path = None
    if eval_config:
        eval_metrics = list(eval_config.get("metrics", ["rfid", "psnr", "ssim", "is"]))
        reference_npz_path = eval_config.get("reference_npz_path", None)

    if rank == 0: logger.info(f"Loading test dataset from {args.data_path}")
    test_dataset = BraTS2023AllDataset(str(args.data_path), 'test', get_transforms('val'))

    if args.max_eval_samples is not None:
        cap = min(args.max_eval_samples, len(test_dataset))
        if rank == 0:
            logger.info(f"Capping test set to {cap} samples (--max-eval-samples).")
        test_dataset = TorchSubset(test_dataset, list(range(cap)))

    only_target, only_known = None, None
    if args.only_combo is not None:
        parts = [int(x) for x in args.only_combo.replace(",", "-").split("-") if x.strip() != ""]
        if len(parts) < 2:
            raise ValueError('--only-combo must look like "0-1-2-3" (target + at least one known modality).')
        only_target, only_known = parts[0], sorted(parts[1:])
        if rank == 0:
            logger.info(f"Only evaluating combo target={only_target}, known={only_known} (--only-combo).")

    # Transport Init
    transport = create_transport(**to_dict(transport_config.get("params", {})), time_dist_shift=time_dist_shift)
    transport_sampler = Sampler(transport)
    
    config_mode = sampler_cfg.get("mode", "ODE").upper() if sampler_cfg else "ODE"
    sampler_mode = args.sampler_mode.upper() if args.sampler_mode else config_mode
    # Only use config params when the mode matches; otherwise fall back to function defaults
    # to avoid passing ODE-specific params (atol/rtol/reverse) to sample_sde() or vice versa.
    config_params = dict(sampler_cfg.get("params", {})) if sampler_cfg else {}
    if sampler_mode == config_mode:
        sampler_params = dict(config_params)
    else:
        # Mode overridden: only pass num_steps (shared param); skip ODE-only keys like atol/rtol/reverse
        sampler_params = {k: v for k, v in config_params.items() if k == "num_steps"}
        if sampler_mode == "SDE":
            # sample_sde defaults: last_step=Mean + last_step_size=0.04 shorten the grid vs ODE (t1=1-last_step_size),
            # which tanks metrics when comparing to ODE. Full-interval SDE matches ODE's endpoint unless YAML sets these.
            sampler_params.setdefault("last_step", None)

    if sampler_mode == "SDE" and args.sde_diffusion_norm is not None:
        sampler_params["diffusion_norm"] = args.sde_diffusion_norm

    if sampler_mode == "SDE" and transport.model_type == ModelType.VELOCITY:
        dn = sampler_params.get("diffusion_norm", None)
        if dn is None:
            dn = 0.0
        else:
            dn = float(dn)
        if dn > 1e-8:
            ns = int(sampler_params.get("num_steps", 50))
            sampler_params["num_steps"] = max(ns, 250)
    if rank == 0: logger.info(f"Sampler mode: {sampler_mode} (config default: {config_mode}), params: {sampler_params}")
    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    # -------------------------------------------------------------------------
    # Final Test - All Combinations
    # -------------------------------------------------------------------------
    if rank == 0: logger.info("Starting all-combinations evaluation...")
    
    # Load existing metrics if available (for resuming)
    metrics_json_path = os.path.join(args.results_dir, "all_combinations_metrics.json")
    all_test_metrics = {}
    if rank == 0 and os.path.exists(metrics_json_path):
        try:
            with open(metrics_json_path, "r") as f:
                all_test_metrics = json_module.load(f)
            logger.info(f"Loaded {len(all_test_metrics)} existing combinations from {metrics_json_path}")
        except Exception as e:
            logger.warning(f"Failed to load existing metrics: {e}, starting fresh")
            all_test_metrics = {}
    
    # Broadcast existing keys to all ranks
    if world_size > 1:
        existing_keys = list(all_test_metrics.keys()) if rank == 0 else None
        existing_keys = [existing_keys]
        dist.broadcast_object_list(existing_keys, src=0)
        existing_keys = set(existing_keys[0]) if existing_keys[0] else set()
    else:
        existing_keys = set(all_test_metrics.keys())
    
    # For every target modality
    for t_idx in range(4):
        if only_target is not None and t_idx != only_target:
            continue
        avail_mod = [i for i in range(4) if i != t_idx]
        
        # Subsets of {avail_mod} of length 1, 2, 3
        for L in range(1, len(avail_mod) + 1):
            for subset in itertools.combinations(avail_mod, L):
                k_idxs = list(subset)
                k_idxs.sort()
                if only_known is not None and k_idxs != only_known:
                    continue
                
                # Check if this combination already has metrics
                dict_key = f"target_{t_idx}_from_{'_'.join(map(str, k_idxs))}"
                if dict_key in existing_keys:
                    if rank == 0: logger.info(f"Skipping: {dict_key} (already evaluated)")
                    continue
                
                eval_cond_id = t_idx * 16 + sum([2**i for i in k_idxs])
                if rank == 0: logger.info(f"Evaluating: Target {t_idx} from Known {k_idxs} (Cond ID: {eval_cond_id})")
                
                test_wrap = EvalWrapper(test_dataset, t_idx, k_idxs, eval_cond_id, rae, device)
                
                test_metrics = evaluate_flow_all_comb(
                    ema_model.forward, eval_sampler, rae,
                    test_wrap, t_idx, k_idxs, eval_cond_id,
                    latent_size, micro_batch_size, autocast_kwargs,
                    args.results_dir, eval_metrics, rank, world_size, device,
                    ref_npz_path=reference_npz_path,
                    num_samples=args.num_samples,
                )
                
                if rank == 0 and test_metrics is not None:
                    dict_key = f"target_{t_idx}_from_{'_'.join(map(str, k_idxs))}"
                    all_test_metrics[dict_key] = test_metrics
                    with open(os.path.join(args.results_dir, "all_combinations_metrics.json"), "w") as f:
                        json_module.dump(all_test_metrics, f, indent=4)

    if rank == 0:
        logger.info(f"All evaluation combinations completed. Results saved to {os.path.join(args.results_dir, 'all_combinations_metrics.json')}")

    if world_size > 1:
        dist.barrier()
    cleanup_distributed()

if __name__ == "__main__": 
    main()

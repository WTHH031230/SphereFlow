"""SphereFlow completion + 3D-UNet segmentation evaluation worker.

Runs in the ``sphereflow`` conda env. Spawned as a subprocess by
``eval_completion.py`` (which runs in ``sphereflow_baseline``).

We load:
    * RAE (DINOv2-B encoder + 4 fine-tuned decoders, "RAE4Dec" wrapper)
    * Flow-matching DiT-XXL transformer (EMA weights)
    * The trained 3D-UNet (same checkpoint as the main script)

For each test case × missing-modality pattern, we:
    1. For each missing target modality m_target:
        * Encode all available (known) modality slices via DINO -> latents
        * Average the latents across known modalities
        * Run the flow sampler conditioning on cond_id = m_target*16 + sum(2^i for i in known)
        * Decode the resulting latent with decoder[m_target]
        * Average the 3-channel decoded output along the channel dim -> single-channel slice
       -> Fill the missing slot of the completed volume.
    2. Z-score (per-channel non-zero) normalise the completed volume.
    3. Run 3D-UNet, compute per-region Dice (WT, TC, ET).

DDP-friendly: launch with ``torchrun --standalone --nproc_per_node=N``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from itertools import combinations

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

# -------- locate project root and 3D-UNet code --------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get(
    'RAE_PROJECT_ROOT',
    os.path.abspath(os.path.join(HERE, '..', '..')),
)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
# SphereFlow imports live under src/
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# These imports are from the 3D-UNet code (work in either env)
import config as C
from dataset import BraTS3DDataset
from losses import (
    per_channel_dice,
    per_channel_hd95,
    per_channel_iou,
    per_channel_sensitivity,
)
from transforms import NormalizeNonzeroPerChannel
from unet3d import UNet3D


METRIC_NAMES = ['dice', 'iou', 'sensitivity', 'hd95']


# ---------------------------------------------------------------------------
#  DDP utilities
# ---------------------------------------------------------------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return get_rank() == 0


def setup_ddp() -> int:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            device_id=torch.device(f'cuda:{local_rank}'),
        )
        return local_rank
    return 0


def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True, help='3D-UNet checkpoint')
    p.add_argument('--json-path', type=str, default=C.JSON_PATH)
    p.add_argument('--phase', type=str, default='test',
                   choices=['train', 'val', 'test'])
    p.add_argument('--patterns', type=str, nargs='+', required=True,
                   help='4-char binary mask strings, e.g. "1110".')
    p.add_argument('--num-cases', type=int, default=-1)
    p.add_argument('--slice-batch', type=int, default=8)
    p.add_argument('--unet-batch', type=int, default=1,
                   help='How many completed volumes to stack into a single 3D-UNet '
                        'forward (segmentation step). 1 = current behaviour.')
    p.add_argument('--chunk-depth', type=int, default=0)
    p.add_argument('--output-json', type=str, required=True)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--amp-dtype', type=str, default=C.AMP_DTYPE,
                   choices=['bf16', 'fp16'])

    p.add_argument('--rae-config', type=str, required=True)
    p.add_argument('--rae-path',   type=str, required=True)
    p.add_argument('--rae-flow',   type=str, required=True)
    p.add_argument('--sampler-num-steps', type=int, default=0,
                   help='Override sampler.params.num_steps from --rae-config when >0.')
    p.add_argument('--no-brain-mask', action='store_true',
                   help='Disable brain-mask post-processing. By default, '
                        'generated channels are multiplied by the brain mask '
                        'derived from raw available channels. See '
                        'eval_completion.py for the rationale.')
    p.add_argument('--no-normalize', action='store_true',
                   help='Feed raw [0,1] volumes to the UNet instead of '
                        'z-score normalising them. Auto-detected from '
                        'ckpt["config"]["no_normalize"] if present.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# RAE4Dec wrapper: shared encoder + one decoder per modality
# ---------------------------------------------------------------------------
class RAE4Dec(nn.Module):
    """Wraps a base RAE so it has 4 separate decoders (one per modality)."""

    def __init__(self, base):
        super().__init__()
        self.encoder = base.encoder
        self.encoder_mean = base.encoder_mean
        self.encoder_std = base.encoder_std
        self.encoder_input_size = base.encoder_input_size
        self.encode = base.encode
        self.reshape_to_2d = getattr(base, 'reshape_to_2d', True)
        self.do_normalization = getattr(base, 'do_normalization', False)
        self.decoders = nn.ModuleList(
            [deepcopy(base.decoder) for _ in range(4)]
        )

    def decode(self, z, m):
        dec = self.decoders[m]
        if self.do_normalization and hasattr(self, 'latent_mean'):
            z = z * torch.sqrt(self.latent_var + 1e-5) + self.latent_mean
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            z = z.view(b, c, h * w).transpose(1, 2)
        out = dec(z, drop_cls_token=False).logits
        x_rec = dec.unpatchify(out)
        return x_rec * self.encoder_std + self.encoder_mean


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location='cpu', mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location='cpu')


def _pick_state_dict(ckpt):
    if isinstance(ckpt, dict):
        return ckpt.get('ema', ckpt.get('model', ckpt))
    return ckpt


def _force_install_rae_utils():
    """Override the top-level ``utils`` package to point at ``src/utils``.

    Plain ``from utils.<name> import ...`` is unreliable here: after
    ``stage2.transport`` is imported, torch initialises CUDA, which lazily
    inserts ``nvidia_cutlass_dsl`` paths and ``import utils`` resolves to
    ``cutlass/base_dsl/utils`` instead of RAE's namespace package. We
    install RAE's ``utils`` directly into ``sys.modules`` and rebind the
    finder to it, so subsequent relative imports inside RAE utils submodules
    (e.g. ``from .dist_utils import ...``) work.
    """
    import types
    utils_dir = os.path.join(PROJECT_ROOT, 'src', 'utils')
    mod = types.ModuleType('utils')
    mod.__path__ = [utils_dir]
    sys.modules['utils'] = mod
    # Drop any cached cutlass-utils submodules that may already have been
    # populated; they're harmless but a clean slate prevents confusion.
    for k in [k for k in sys.modules if k.startswith('utils.')]:
        del sys.modules[k]


def build_rae_and_flow(args, device):
    """Returns (rae, ema_flow_model, sampler, latent_size)."""
    from omegaconf import OmegaConf
    from stage2.transport import create_transport, Sampler
    _force_install_rae_utils()
    from utils.model_utils import instantiate_from_config
    from utils.train_utils import parse_configs

    cfg = OmegaConf.load(args.rae_config)
    (rae_cfg, model_cfg, transport_cfg, sampler_cfg,
     _, misc_cfg, _, _) = parse_configs(cfg)
    if 'ckpt' in rae_cfg:
        del rae_cfg['ckpt']

    def _d(c):
        return OmegaConf.to_container(c, resolve=True) if c else {}

    misc = _d(misc_cfg)
    latent_size = tuple(int(d) for d in misc.get('latent_size', (768, 16, 16)))
    shift_dim = misc.get('time_dist_shift_dim', math.prod(latent_size))
    shift_base = misc.get('time_dist_shift_base', 4096)
    time_shift = math.sqrt(shift_dim / shift_base)

    # RAE configs use paths like ``configs/decoder/ViTG`` that are resolved
    # relative to the current working directory. Temporarily chdir into the
    # repository root so ``AutoConfig.from_pretrained`` can find them.
    rae_root = PROJECT_ROOT
    prev_cwd = os.getcwd()
    try:
        os.chdir(rae_root)
        base_rae = instantiate_from_config(rae_cfg).to(device)
    finally:
        os.chdir(prev_cwd)

    rae = RAE4Dec(base_rae).to(device)
    if hasattr(base_rae, 'decoder'):
        del base_rae.decoder
    torch.cuda.empty_cache()

    rae_ckpt = _torch_load_cpu(args.rae_path)
    info = rae.load_state_dict(_pick_state_dict(rae_ckpt), strict=False)
    if is_main():
        print(f"  [RAE] loaded {args.rae_path}; "
              f"missing={len(info.missing_keys)}, "
              f"unexpected={len(info.unexpected_keys)}",
              flush=True)
    del rae_ckpt
    torch.cuda.empty_cache()
    rae.eval().requires_grad_(False)

    flow_model = instantiate_from_config(model_cfg).to(device)
    ema = deepcopy(flow_model).to(device).eval().requires_grad_(False)
    flow_ckpt = _torch_load_cpu(args.rae_flow)
    ema_sd = flow_ckpt.get('ema', flow_ckpt.get('model', flow_ckpt))
    info_flow = ema.load_state_dict(ema_sd, strict=False)
    if is_main():
        print(f"  [Flow-XXL] loaded EMA from {args.rae_flow}; "
              f"missing={len(info_flow.missing_keys)}, "
              f"unexpected={len(info_flow.unexpected_keys)}",
              flush=True)
    del flow_model, flow_ckpt
    torch.cuda.empty_cache()

    transport = create_transport(
        **_d(transport_cfg.get('params', {})),
        time_dist_shift=time_shift,
    )
    s_cfg = _d(sampler_cfg) if sampler_cfg else {}
    s_params = dict(s_cfg.get('params', {}))
    if getattr(args, 'sampler_num_steps', 0) and args.sampler_num_steps > 0:
        s_params['num_steps'] = int(args.sampler_num_steps)
    if is_main():
        print(f"  [Sampler] mode=ODE params={s_params}", flush=True)
    sampler = Sampler(transport).sample_ode(**s_params)
    return rae, ema, sampler, latent_size


# ---------------------------------------------------------------------------
#  Volume completion
# ---------------------------------------------------------------------------
def _slice_iter(D: int, batch: int):
    for s in range(0, D, batch):
        yield s, min(s + batch, D)


def compute_brain_mask(raw_vol: torch.Tensor, available_idxs) -> torch.Tensor:
    m = torch.zeros(raw_vol.shape[1:], dtype=torch.bool)
    for i in available_idxs:
        m = m | (raw_vol[i] > 0)
    return m


def apply_brain_mask(completed: torch.Tensor, raw_vol: torch.Tensor,
                     available_idxs, missing_idxs) -> torch.Tensor:
    if not missing_idxs:
        return completed
    bm = compute_brain_mask(raw_vol, available_idxs)
    if bm.device != completed.device:
        bm = bm.to(completed.device)
    for mi in missing_idxs:
        completed[mi] = completed[mi] * bm.to(completed.dtype)
    return completed


@torch.no_grad()
def rae_complete_volume(volume: torch.Tensor, mask_str: str,
                        rae, ema_forward, sampler, slice_batch: int,
                        device, autocast_ctx) -> torch.Tensor:
    """
    Args:
        volume: (4, D, H, W) raw [0, 1] tensor (CPU). Missing channels are zero.
        mask_str: 4-char binary string, '1' = available, '0' = missing.
    Returns:
        completed (4, D, H, W) tensor in [0, 1] (CPU).
    """
    out = volume.clone()
    avail = [i for i, c in enumerate(mask_str) if c == '1']
    miss = [i for i, c in enumerate(mask_str) if c == '0']
    if not miss:
        return out

    D = volume.shape[1]
    known_bitmask = sum(2 ** i for i in avail)

    for m_target in miss:
        cond_id_val = m_target * 16 + known_bitmask

        for s, e in _slice_iter(D, slice_batch):
            # Encode each known modality (B, 1, H, W) -> repeat to (B, 3, H, W)
            z_list = []
            for ki in avail:
                x_k = volume[ki:ki + 1, s:e]                       # (1, B, H, W)
                x_k = x_k.permute(1, 0, 2, 3).repeat(1, 3, 1, 1)   # (B, 3, H, W)
                x_k = x_k.to(device, non_blocking=True)
                z_list.append(rae.encode(x_k))
            z_known = torch.stack(z_list, dim=0).mean(dim=0)        # (B, C, h, w)

            B = z_known.shape[0]
            cond_ids = torch.full((B,), cond_id_val,
                                  dtype=torch.long, device=device)

            with autocast_ctx:
                samples = sampler(z_known, ema_forward, y=cond_ids)[-1]
                decoded = rae.decode(samples, m_target).clamp(0, 1)  # (B, 3, H, W)

            # Average 3 channels back to single-channel grayscale
            decoded_gray = decoded.mean(dim=1)                       # (B, H, W)
            out[m_target, s:e] = decoded_gray.cpu().float()
    return out


# ---------------------------------------------------------------------------
#  Segmentation
# ---------------------------------------------------------------------------
def _normalise_volume(image_np: np.ndarray) -> torch.Tensor:
    norm = NormalizeNonzeroPerChannel()(
        {'image': image_np, 'label': None})['image']
    return torch.from_numpy(norm).float()


def build_amp_context(use_amp: bool, dtype: str):
    if not use_amp:
        return nullcontext()
    amp_dtype = torch.bfloat16 if dtype == 'bf16' else torch.float16
    return torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True)


@torch.no_grad()
def _unet_forward_maybe_chunked(unet, img: torch.Tensor, chunk_depth: int,
                                autocast_ctx) -> torch.Tensor:
    if chunk_depth <= 0 or chunk_depth >= img.shape[2]:
        with autocast_ctx:
            return unet(img)
    assert chunk_depth % 8 == 0, (
        f"chunk_depth must be a multiple of 8, got {chunk_depth}.")
    D = img.shape[2]
    out_chunks = []
    for s in range(0, D, chunk_depth):
        e = min(s + chunk_depth, D)
        if (e - s) % 8 != 0:
            pad = ((e - s + 7) // 8) * 8 - (e - s)
            chunk = torch.nn.functional.pad(
                img[:, :, s:e], (0, 0, 0, 0, 0, pad))
        else:
            chunk = img[:, :, s:e].contiguous()
        with autocast_ctx:
            out = unet(chunk)
        out = out[:, :, :e - s]
        out_chunks.append(out)
    return torch.cat(out_chunks, dim=2)


@torch.no_grad()
def segment_and_metrics(unet, completed_vols_01, labels, device, autocast_ctx,
                        chunk_depth: int = 0, normalize: bool = True):
    """See eval_completion.segment_and_metrics; returns per-region tensors
    on CPU for each of the 4 metrics, shape (B, 3) each."""
    imgs = []
    for v in completed_vols_01:
        if normalize:
            imgs.append(_normalise_volume(v.detach().cpu().numpy()))
        else:
            imgs.append(v.detach().cpu().float())
    img = torch.stack(imgs, dim=0).to(device, non_blocking=True)
    lbl = torch.stack([l.float() for l in labels], dim=0).to(
        device, non_blocking=True)
    logits = _unet_forward_maybe_chunked(unet, img, chunk_depth, autocast_ctx)
    prob = torch.sigmoid(logits.float())
    return {
        'dice':        per_channel_dice(prob, lbl).detach().cpu(),
        'iou':         per_channel_iou(prob, lbl).detach().cpu(),
        'sensitivity': per_channel_sensitivity(prob, lbl).detach().cpu(),
        'hd95':        per_channel_hd95(prob, lbl).detach().cpu(),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    local_rank = setup_ddp()
    rank, world_size = get_rank(), get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    torch.backends.cudnn.benchmark = False  # keep memory down (see eval_completion.py)

    if is_main():
        print(f"=== eval_completion_sphereflow ===")
        print(f"World size: {world_size}, rank: {rank}")
        print(f"Args:\n{json.dumps(vars(args), indent=2, default=str)}", flush=True)

    # -------- Patterns --------
    patterns = []
    for ms in args.patterns:
        avail = [i for i, c in enumerate(ms) if c == '1']
        miss = [i for i, c in enumerate(ms) if c == '0']
        patterns.append((ms, avail, miss))

    # -------- Dataset --------
    dataset = BraTS3DDataset(
        json_path=args.json_path,
        phase=args.phase,
        transform=None,
        expected_depth=C.EXPECTED_DEPTH,
    )
    n_total_cases = len(dataset)
    if args.num_cases > 0:
        n_total_cases = min(n_total_cases, args.num_cases)
    all_idxs = list(range(n_total_cases))
    case_idxs = all_idxs[rank::world_size]
    if is_main():
        print(f"Dataset: phase={args.phase}, n_cases_total={len(dataset)}, "
              f"using={n_total_cases}; this rank handles {len(case_idxs)}",
              flush=True)

    # -------- 3D-UNet --------
    unet = UNet3D(in_channels=C.IN_CHANNELS, num_classes=C.NUM_CLASSES).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    info = unet.load_state_dict(state, strict=False)
    ckpt_no_norm = bool(ckpt.get('config', {}).get('no_normalize', False))
    no_normalize = args.no_normalize or ckpt_no_norm
    if is_main():
        print(f"  [UNet] loaded {args.ckpt}; "
              f"missing={len(info[0])}, unexpected={len(info[1])}", flush=True)
        print(f"  [normalize] {'DISABLED (raw [0,1])' if no_normalize else 'ENABLED (z-score)'} "
              f"(from ckpt config={ckpt_no_norm}, cli={args.no_normalize})",
              flush=True)
    unet.eval()

    # -------- RAE + Flow --------
    rae, ema, sampler, latent_size = build_rae_and_flow(args, device)
    autocast_ctx = build_amp_context(use_amp=not args.no_amp, dtype=args.amp_dtype)

    # -------- Loop --------
    results = {
        ms: {
            **{f'{m}_sum': torch.zeros(3, device=device) for m in METRIC_NAMES},
            'count': 0,
        }
        for ms, _, _ in patterns
    }
    unet_batch = max(1, int(args.unet_batch))

    def _flush(buf_vols, buf_lbls, buf_keys):
        if not buf_vols:
            return
        metrics = segment_and_metrics(
            unet, buf_vols, buf_lbls, device, autocast_ctx,
            chunk_depth=args.chunk_depth, normalize=not no_normalize,
        )
        for i, ms in enumerate(buf_keys):
            for m in METRIC_NAMES:
                results[ms][f'{m}_sum'] += metrics[m][i].to(device)
            results[ms]['count'] += 1
            if is_main():
                d = metrics['dice'][i].tolist()
                print(f"      seg-batch ms={ms} dice="
                      f"[{d[0]:.3f},{d[1]:.3f},{d[2]:.3f}]",
                      flush=True)

    n_cases = len(case_idxs)
    if is_main():
        print(f"\n  Processing {n_cases} cases x {len(patterns)} patterns ...",
              flush=True)
    t_start = time.time()

    apply_bmask = not args.no_brain_mask
    if is_main():
        print(f"  brain_mask_postprocess = {apply_bmask}", flush=True)
    for ci, idx in enumerate(case_idxs):
        sample = dataset[idx]
        image_raw = torch.from_numpy(sample['image']).float()    # (4, D, H, W)
        label = torch.from_numpy(sample['label']).float()        # (3, D, H, W)
        case_name = sample['case_name']

        buf_vols, buf_lbls, buf_keys = [], [], []
        for mask_str, avail, miss in patterns:
            t0 = time.time()
            completed = rae_complete_volume(
                image_raw, mask_str, rae, ema.forward, sampler,
                args.slice_batch, device, autocast_ctx,
            )
            if apply_bmask:
                completed = apply_brain_mask(completed, image_raw, avail, miss)
            buf_vols.append(completed)
            buf_lbls.append(label)
            buf_keys.append(mask_str)
            if is_main():
                elapsed = time.time() - t0
                print(f"    case {ci+1}/{n_cases} ({case_name}) "
                      f"pattern={mask_str} completion ({elapsed:.1f}s)",
                      flush=True)
            if len(buf_vols) >= unet_batch:
                _flush(buf_vols, buf_lbls, buf_keys)
                buf_vols, buf_lbls, buf_keys = [], [], []
        _flush(buf_vols, buf_lbls, buf_keys)

        if is_main():
            mins = (time.time() - t_start) / 60
            est_total = mins / (ci + 1) * n_cases
            print(f"  >> case {ci+1}/{n_cases} done; "
                  f"elapsed={mins:.1f} min; est total={est_total:.1f} min",
                  flush=True)

    # -------- All-reduce --------
    if is_dist():
        for ms, r in results.items():
            n = torch.tensor([r['count']], device=device, dtype=torch.float32)
            for m in METRIC_NAMES:
                dist.all_reduce(r[f'{m}_sum'], op=dist.ReduceOp.SUM)
            dist.all_reduce(n, op=dist.ReduceOp.SUM)
            r['count'] = int(n.item())

    # -------- Save --------
    if is_main():
        agg = {}
        for ms, r in results.items():
            n = max(r['count'], 1)
            entry = {'count': r['count']}
            for m in METRIC_NAMES:
                vals = (r[f'{m}_sum'] / n).cpu().tolist()
                entry[m] = {
                    'WT': vals[0], 'TC': vals[1], 'ET': vals[2],
                    'mean': float(np.mean(vals)),
                }
            d = entry['dice']
            entry['WT'] = d['WT']; entry['TC'] = d['TC']
            entry['ET'] = d['ET']; entry['mean'] = d['mean']
            agg[ms] = entry
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump({
                'method': 'SphereFlow',
                'metrics': METRIC_NAMES,
                'results': agg,
            }, f, indent=2)
        print(f"\n  [SphereFlow] saved -> {args.output_json}", flush=True)

    cleanup_ddp()


if __name__ == '__main__':
    main()

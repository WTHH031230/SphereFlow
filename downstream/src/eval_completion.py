"""Evaluate trained 3D-UNet on BraTS test set under various missing-modality
**completion** methods.

For each missing-modality pattern (14 of the 15 non-empty subsets — we skip
the trivial ``1111`` "no-missing" case unless you opt in via ``--include-full``),
we:

    1. Take the raw 4-mod volume in ``[0, 1]``.
    2. Apply a completion method (or ``Zero`` baseline) to fill in the missing
       modalities.
    3. Apply per-channel non-zero z-score normalisation (same as training).
    4. Run the trained 3D-UNet to get sigmoid probabilities.
    5. Compute per-region (WT / TC / ET) Dice against the GT label.

Methods supported:

    * ``Zero``      (no model — naive zero-fill baseline)
    * ``MMT``
    * ``M2DN``
    * ``ModImp``
    * ``Zhang``
    * ``MMGAN``
    * ``MMSyn``

NOTE: ``SphereFlow`` is **NOT** runnable from here. It lives in a different
conda env (``sphereflow``) and must be launched separately via
``eval_completion_sphereflow.py`` to avoid an NCCL deadlock (rank 0 would
otherwise block on the subprocess while other ranks time out at a barrier).

This script runs in the ``sphereflow_baseline`` conda env.

Each (method) writes ``partial_<method>.json`` into ``--output-dir``. The
aggregate ``results.json`` / ``results.csv`` is produced by a separate
``aggregate_results.py`` step that scans the partial files (so that we can
combine inline methods + the SphereFlow run that happens out-of-process).

DDP-friendly: launch with ``torchrun --standalone --nproc_per_node=N`` to
shard cases across N GPUs (the same model is loaded on each rank).

Example:

    conda run -n sphereflow_baseline torchrun --standalone --nproc_per_node=4 \
        eval_completion.py \
        --ckpt runs/brats_unet3d/checkpoints/best.pth \
        --output-dir runs/brats_unet3d/completion_eval \
        --methods Zhang MMSyn
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from itertools import combinations

import numpy as np
import torch
import torch.distributed as dist

import config as C
from completion_methods import (
    INLINE_METHOD_REGISTRY,
    DEFAULT_CKPTS,
)

INLINE_METHODS = list(INLINE_METHOD_REGISTRY.keys())
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
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
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
    p.add_argument('--ckpt', type=str, required=True,
                   help='Trained 3D-UNet checkpoint (.pth)')
    p.add_argument('--json-path', type=str, default=C.JSON_PATH)
    p.add_argument('--phase', type=str, default='test',
                   choices=['train', 'val', 'test'])
    p.add_argument('--methods', type=str, nargs='+', default=INLINE_METHODS,
                   choices=INLINE_METHODS,
                   help='Which completion methods to evaluate. SphereFlow is NOT '
                        'allowed here; run eval_completion_sphereflow.py '
                        'separately in the `sphereflow` conda env.')
    p.add_argument('--patterns', type=str, nargs='+', default=None,
                   help='Restrict to specific patterns, e.g. "1110 1010". '
                        'Default: all 14 patterns with at least one missing modality.')
    p.add_argument('--include-full', action='store_true',
                   help='Also evaluate the "1111" no-missing pattern (skipped by default).')
    p.add_argument('--num-cases', type=int, default=-1,
                   help='Limit to first N test cases (for fast iteration). '
                        '-1 = all cases.')
    p.add_argument('--slice-batch', type=int, default=8,
                   help='Number of slices per forward pass for 2D methods.')
    p.add_argument('--unet-batch', type=int, default=1,
                   help='How many completed volumes to stack into a single 3D-UNet '
                        'forward (segmentation step). 1 = current behaviour. '
                        'Higher values use more GPU memory but cut per-case '
                        'overhead. Patterns of the same case are grouped first; '
                        'leftovers smaller than --unet-batch run as a smaller '
                        'final batch.')
    p.add_argument('--no-amp', action='store_true',
                   help='Disable AMP for the segmentation forward pass.')
    p.add_argument('--amp-dtype', type=str, default=C.AMP_DTYPE,
                   choices=['bf16', 'fp16'])
    p.add_argument('--chunk-depth', type=int, default=0,
                   help='If >0, split the (4,D,H,W) input along the depth axis '
                        'into chunks of this size for the 3D-UNet forward pass '
                        '(saves peak GPU memory). 0 = whole-volume forward.')
    p.add_argument('--no-brain-mask', action='store_true',
                   help='Disable the brain-mask post-processing. By default, '
                        'generated (missing) channels are multiplied by the '
                        'brain mask derived from raw available channels '
                        '(any available modality > 0). This undoes background '
                        'bleeding from generative models (CodeBrain / Zhang / '
                        'M2DN) which otherwise poisons the per-channel '
                        'non-zero z-score normalisation in the segmentation '
                        'head. Strongly recommended with z-score UNets — '
                        'for a raw [0,1] UNet this is usually unnecessary.')
    p.add_argument('--no-normalize', action='store_true',
                   help='Feed raw [0,1] volumes to the UNet instead of '
                        'z-score normalising them. Auto-detected from '
                        'the UNet checkpoint (ckpt["config"]["no_normalize"]) '
                        'if the ckpt was trained that way — this flag only '
                        'needs to be set manually for ckpts without that '
                        'field.')
    p.add_argument('--output-dir', type=str, default='',
                   help='Where to save results. Default: <ckpt_dir>/../completion_eval')

    # checkpoint overrides
    p.add_argument('--mmt-ckpt',       type=str, default=None)
    p.add_argument('--m2dn-ckpt',      type=str, default=None)
    p.add_argument('--modimp-ckpt',    type=str, default=None)
    p.add_argument('--zhang-ckpt',     type=str, default=None)
    p.add_argument('--mmgan-ckpt',     type=str, default=None)
    p.add_argument('--mmsyn-ckpt',     type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
#  Patterns
# ---------------------------------------------------------------------------
def enumerate_missing_patterns(include_full: bool = False, restrict=None):
    """Return list of (mask_str, available_idxs, missing_idxs).

    mask_str is a 4-char binary string in modality order [T1, T1ce, T2, FLAIR],
    e.g. ``"1110"`` means [T1, T1ce, T2 available; FLAIR missing].
    """
    out = []
    for k in range(1, 4 + 1):
        for sub in combinations(range(4), k):
            mask_str = ''.join('1' if i in sub else '0' for i in range(4))
            avail = list(sub)
            miss = [i for i in range(4) if i not in sub]
            if not include_full and not miss:
                continue
            if restrict is not None and mask_str not in restrict:
                continue
            out.append((mask_str, avail, miss))
    return out


# ---------------------------------------------------------------------------
#  Segmentation evaluation per case for a (method, pattern) pair
# ---------------------------------------------------------------------------
def _normalise_volume(image_np: np.ndarray) -> torch.Tensor:
    """Apply z-score-per-channel-non-zero, return float tensor (4, D, H, W)."""
    norm = NormalizeNonzeroPerChannel()(
        {'image': image_np, 'label': None})['image']
    return torch.from_numpy(norm).float()


@torch.no_grad()
def _unet_forward_maybe_chunked(unet, img: torch.Tensor, chunk_depth: int,
                                autocast_ctx) -> torch.Tensor:
    """Run UNet on (1, 4, D, H, W). If ``chunk_depth > 0``, split along D and
    concatenate outputs (saves peak workspace memory). Note that boundary slices
    of each chunk see zero-padded "context"; for smoke-test memory savings this
    is acceptable, for production prefer whole-volume forward (``chunk_depth=0``).
    The 3D-UNet has 3 levels of stride-2 max-pool, so each chunk size must be
    a multiple of 8 to keep encoder/decoder spatial sizes aligned.
    """
    if chunk_depth <= 0 or chunk_depth >= img.shape[2]:
        with autocast_ctx:
            return unet(img)
    assert chunk_depth % 8 == 0, (
        f"chunk_depth must be a multiple of 8 (3 stride-2 pool levels), "
        f"got {chunk_depth}.")
    D = img.shape[2]
    out_chunks = []
    for s in range(0, D, chunk_depth):
        e = min(s + chunk_depth, D)
        if (e - s) % 8 != 0:
            # last chunk may be < chunk_depth; pad to next multiple of 8
            pad = ((e - s + 7) // 8) * 8 - (e - s)
            chunk = torch.nn.functional.pad(
                img[:, :, s:e], (0, 0, 0, 0, 0, pad))
        else:
            chunk = img[:, :, s:e].contiguous()
        with autocast_ctx:
            out = unet(chunk)
        out = out[:, :, :e - s]   # remove padding from output
        out_chunks.append(out)
    return torch.cat(out_chunks, dim=2)


@torch.no_grad()
def segment_and_metrics(unet, completed_vols_01, labels, device, autocast_ctx,
                        chunk_depth: int = 0, normalize: bool = True):
    """Run the UNet on a list of completed volumes (one batch) and return
    per-region metrics for each volume.

    Args:
        completed_vols_01: list of (4, D, H, W) float tensors in [0, 1].
        labels: list of (3, D, H, W) float tensors in {0, 1}; same length.
        normalize: if True, apply NormalizeNonzeroPerChannel z-score before
                   UNet forward; if False, feed raw [0, 1].

    Returns:
        dict {metric_name -> (B, 3) tensor on CPU}.
    """
    imgs = []
    for v in completed_vols_01:
        if normalize:
            imgs.append(_normalise_volume(v.detach().cpu().numpy()))
        else:
            imgs.append(v.detach().cpu().float())
    img = torch.stack(imgs, dim=0).to(device, non_blocking=True)     # (B,4,D,H,W)
    lbl = torch.stack([l.float() for l in labels], dim=0).to(
        device, non_blocking=True)                                    # (B,3,D,H,W)
    logits = _unet_forward_maybe_chunked(unet, img, chunk_depth, autocast_ctx)
    prob = torch.sigmoid(logits.float())

    return {
        'dice':        per_channel_dice(prob, lbl).detach().cpu(),
        'iou':         per_channel_iou(prob, lbl).detach().cpu(),
        'sensitivity': per_channel_sensitivity(prob, lbl).detach().cpu(),
        'hd95':        per_channel_hd95(prob, lbl).detach().cpu(),
    }


def build_amp_context(use_amp: bool, dtype: str):
    if not use_amp:
        return nullcontext()
    amp_dtype = torch.bfloat16 if dtype == 'bf16' else torch.float16
    return torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True)


def compute_brain_mask(raw_vol: torch.Tensor, available_idxs) -> torch.Tensor:
    """Brain-support mask (D, H, W) bool from raw available modalities.

    We take the union of ``raw_vol[i] > 0`` for each available channel. This
    mirrors how the original preprocessed BraTS volumes encode the brain
    (non-zero) vs zero-padded background. When a generative model (CodeBrain,
    Zhang, M2DN...) leaks non-zero values outside the brain, multiplying its
    output by this mask removes the leak and restores the channel support the
    UNet was trained on.
    """
    m = torch.zeros(raw_vol.shape[1:], dtype=torch.bool)
    for i in available_idxs:
        m = m | (raw_vol[i] > 0)
    return m


def apply_brain_mask(completed: torch.Tensor, raw_vol: torch.Tensor,
                     available_idxs, missing_idxs) -> torch.Tensor:
    """In-place set ``completed[missing_idxs, bg] = 0`` using brain mask.

    Returns ``completed`` for chaining.
    """
    if not missing_idxs:
        return completed
    bm = compute_brain_mask(raw_vol, available_idxs)            # (D, H, W)
    if bm.device != completed.device:
        bm = bm.to(completed.device)
    for mi in missing_idxs:
        completed[mi] = completed[mi] * bm.to(completed.dtype)
    return completed


# ---------------------------------------------------------------------------
#  Inline evaluation loop for one method (handles all patterns & cases)
# ---------------------------------------------------------------------------
def _empty_metric_buffers(patterns, device):
    return {
        ms: {
            **{f'{m}_sum': torch.zeros(3, device=device) for m in METRIC_NAMES},
            'count': 0,
        }
        for ms, _, _ in patterns
    }


def _flush_unet_batch(unet, batch_volumes, batch_labels, batch_keys, results,
                     device, autocast_ctx, chunk_depth, normalize):
    """Run a UNet forward over the queued (volume, label, mask_str) triples
    and add per-metric per-region values to ``results``."""
    if not batch_volumes:
        return
    metrics = segment_and_metrics(
        unet, batch_volumes, batch_labels, device, autocast_ctx,
        chunk_depth=chunk_depth, normalize=normalize,
    )
    for i, ms in enumerate(batch_keys):
        for m in METRIC_NAMES:
            results[ms][f'{m}_sum'] += metrics[m][i].to(device)
        results[ms]['count'] += 1


def evaluate_method_inline(method, dataset, case_idxs, patterns, unet, device,
                           autocast_ctx, log_prefix='', chunk_depth=0,
                           apply_bmask=True, normalize=True, unet_batch=1):
    """
    Runs one method over case_idxs × patterns and returns a dict:
        results[mask_str]: {
            'dice_sum': tensor(3,), 'iou_sum': tensor(3,),
            'f1_sum':   tensor(3,), 'hd95_sum': tensor(3,),
            'count':    int,
        }
    """
    results = _empty_metric_buffers(patterns, device)

    n_cases = len(case_idxs)
    for ci, idx in enumerate(case_idxs):
        sample = dataset[idx]
        image_raw = torch.from_numpy(sample['image']).float()    # (4, D, H, W) in [0,1]
        label = torch.from_numpy(sample['label']).float()        # (3, D, H, W)

        # Queue completed volumes for batched UNet forward.
        batch_vols, batch_lbls, batch_keys = [], [], []
        for mask_str, avail, miss in patterns:
            mask_t = torch.tensor([float(c) for c in mask_str], dtype=torch.float32)
            completed = method.complete(image_raw, mask_t)           # (4, D, H, W) in [0,1]
            if apply_bmask:
                completed = apply_brain_mask(completed, image_raw, avail, miss)
            batch_vols.append(completed)
            batch_lbls.append(label)
            batch_keys.append(mask_str)
            if len(batch_vols) >= unet_batch:
                _flush_unet_batch(
                    unet, batch_vols, batch_lbls, batch_keys, results,
                    device, autocast_ctx, chunk_depth, normalize,
                )
                batch_vols, batch_lbls, batch_keys = [], [], []

        # Flush leftovers.
        _flush_unet_batch(
            unet, batch_vols, batch_lbls, batch_keys, results,
            device, autocast_ctx, chunk_depth, normalize,
        )

        if is_main() and (ci + 1) % 5 == 0:
            print(f"    {log_prefix}case {ci+1}/{n_cases} done", flush=True)

    return results


def reduce_results_across_ranks(results, device):
    """All-reduce (sum) every <metric>_sum tensor and the integer counts."""
    if not is_dist():
        return results
    out = {}
    for ms, r in results.items():
        n = torch.tensor([r['count']], device=device, dtype=torch.float32)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        entry = {'count': int(n.item())}
        for m in METRIC_NAMES:
            t = r[f'{m}_sum'].clone()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            entry[f'{m}_sum'] = t
        out[ms] = entry
    return out


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------
def _cell(d, region, metric):
    """Pull (region, metric) from a partial entry; supports new nested
    schema and the legacy flat-Dice schema."""
    if d is None:
        return None
    if metric in d and isinstance(d[metric], dict):
        return d[metric].get(region)
    if metric == 'dice':
        return d.get(region)
    return None


def format_table(all_results, patterns, methods, metric='dice'):
    """Build a pretty per-(method, pattern) table for one metric."""
    lines = [f"Metric: {metric.upper()}"]
    header = (f"{'pattern':>8} {'avail':>14}  | " +
              '  '.join(f"{m:>10}" for m in methods))
    lines.append('=' * len(header))
    lines.append(header)
    lines.append('-' * len(header))
    region_full = ['WT', 'TC', 'ET', 'mean']

    last_k = -1
    for mask_str, avail, miss in patterns:
        k = len(avail)
        if k != last_k:
            if last_k != -1:
                lines.append('-' * len(header))
            last_k = k
        avail_str = '+'.join(C.MODALITY_NAMES[i] for i in avail) or '<none>'
        for ri, region in enumerate(region_full):
            row_label = f"{mask_str:>8} {avail_str:>14}" if ri == 0 else ' ' * 23
            row = f"{row_label}  | "
            cells = []
            for m in methods:
                v = _cell(all_results.get(m, {}).get(mask_str), region, metric)
                cells.append(f"{'-':>10}" if v is None else f"{v:>10.4f}")
            lines.append(row + '  '.join(cells) + f"  ({region})")
    lines.append('=' * len(header))
    return '\n'.join(lines)


def format_summary_table(all_results, patterns, methods, metric='dice'):
    """Summary by k_known (# of known modalities) for one metric."""
    lines = [f"Metric: {metric.upper()} -- average across patterns with same "
             f"number of known modalities:"]
    header = (f"{'k_known':>8}  | " +
              '  '.join(f"{m:>10}" for m in methods))
    lines.append('=' * len(header))
    lines.append(header)
    lines.append('-' * len(header))

    by_k = {}
    for mask_str, avail, _ in patterns:
        by_k.setdefault(len(avail), []).append(mask_str)
    for region in ['WT', 'TC', 'ET', 'mean']:
        for k in sorted(by_k):
            row = f"{k:>8}  | "
            cells = []
            for m in methods:
                vals = []
                for ms in by_k[k]:
                    v = _cell(all_results.get(m, {}).get(ms), region, metric)
                    if v is not None:
                        vals.append(v)
                cells.append(f"{'-':>10}" if not vals
                             else f"{np.mean(vals):>10.4f}")
            lines.append(row + '  '.join(cells) + f"  ({region})")
        lines.append('-' * len(header))
    lines.append('=' * len(header))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    local_rank = setup_ddp()
    rank, world_size = get_rank(), get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    # In inference we disable cudnn benchmark/heuristic memory hogs: a single
    # 3D-UNet forward at (1,4,80,256,256) can otherwise spike workspace memory
    # to 50 GB+, which OOMs when sharing GPUs with the training process.
    torch.backends.cudnn.benchmark = False

    if is_main():
        print(f"=== eval_completion ===")
        print(f"World size: {world_size}, rank: {rank}")
        print(f"Methods to run: {args.methods}")
        print(f"brain_mask_postprocess = {not args.no_brain_mask}")
        print(f"Cmd args:\n{json.dumps(vars(args), indent=2, default=str)}")
        sys.stdout.flush()

    # -------------- output dir --------------
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt))),
        'completion_eval',
    )
    if is_main():
        os.makedirs(out_dir, exist_ok=True)

    # -------------- Patterns --------------
    restrict = set(args.patterns) if args.patterns else None
    patterns = enumerate_missing_patterns(include_full=args.include_full,
                                          restrict=restrict)
    if is_main():
        print(f"Patterns to evaluate ({len(patterns)}):")
        for ms, av, mi in patterns:
            avail_str = '+'.join(C.MODALITY_NAMES[i] for i in av) or '<none>'
            print(f"    {ms}  avail=[{avail_str}]  miss="
                  f"[{','.join(C.MODALITY_NAMES[i] for i in mi)}]")

    # -------------- Dataset --------------
    dataset = BraTS3DDataset(
        json_path=args.json_path,
        phase=args.phase,
        transform=None,                       # we apply normalisation manually
        expected_depth=C.EXPECTED_DEPTH,
    )
    n_total_cases = len(dataset)
    if args.num_cases > 0:
        n_total_cases = min(n_total_cases, args.num_cases)

    # shard cases across ranks
    all_idxs = list(range(n_total_cases))
    case_idxs = all_idxs[rank::world_size]
    if is_main():
        print(f"Dataset: phase={args.phase}, n_cases_total={len(dataset)}, "
              f"using={n_total_cases}, this rank handles {len(case_idxs)}")

    # -------------- 3D-UNet --------------
    unet = UNet3D(in_channels=C.IN_CHANNELS, num_classes=C.NUM_CLASSES).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = unet.load_state_dict(state, strict=False)
    ckpt_no_norm = bool(ckpt.get('config', {}).get('no_normalize', False))
    no_normalize = args.no_normalize or ckpt_no_norm
    if is_main():
        print(f"[UNet] loaded {args.ckpt}; "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"[normalize] {'DISABLED (raw [0,1] input)' if no_normalize else 'ENABLED (z-score)'} "
              f"(from ckpt config={ckpt_no_norm}, cli={args.no_normalize})")
    unet.eval()

    autocast_ctx = build_amp_context(use_amp=not args.no_amp, dtype=args.amp_dtype)

    # -------------- Run methods --------------
    all_results = {}    # method -> {mask_str: {'WT','TC','ET','mean'}}

    for m_name in args.methods:
        if is_main():
            print(f"\n{'=' * 60}")
            print(f"  Method: {m_name}")
            print(f"{'=' * 60}")
        t0 = time.time()
        cls = INLINE_METHOD_REGISTRY[m_name]
        # build with overridden ckpts when applicable
        if m_name == 'MMT':
            method = cls(ckpt=args.mmt_ckpt, slice_batch=args.slice_batch)
        elif m_name == 'M2DN':
            method = cls(ckpt=args.m2dn_ckpt, slice_batch=args.slice_batch)
        elif m_name == 'ModImp':
            method = cls(ckpt=args.modimp_ckpt, slice_batch=args.slice_batch)
        elif m_name == 'Zhang':
            method = cls(ckpt=args.zhang_ckpt, slice_batch=args.slice_batch)
        elif m_name == 'MMGAN':
            method = cls(ckpt=args.mmgan_ckpt, slice_batch=args.slice_batch)
        elif m_name == 'MMSyn':
            method = cls(ckpt=args.mmsyn_ckpt, slice_batch=args.slice_batch)
        else:
            method = cls()

        method.load(device)

        method_results = evaluate_method_inline(
            method, dataset, case_idxs, patterns, unet, device,
            autocast_ctx, log_prefix=f'[{m_name}] ',
            chunk_depth=args.chunk_depth,
            apply_bmask=not args.no_brain_mask,
            normalize=not no_normalize,
            unet_batch=max(1, int(args.unet_batch)),
        )
        method.unload()

        method_results = reduce_results_across_ranks(method_results, device)

        if is_main():
            agg = {}
            for ms, r in method_results.items():
                n = max(r['count'], 1)
                entry = {'count': r['count']}
                for m in METRIC_NAMES:
                    vals = (r[f'{m}_sum'] / n).cpu().tolist()
                    entry[m] = {
                        'WT': vals[0], 'TC': vals[1], 'ET': vals[2],
                        'mean': float(np.mean(vals)),
                    }
                # Top-level WT/TC/ET/mean are the Dice scores.
                d = entry['dice']
                entry['WT'] = d['WT']; entry['TC'] = d['TC']
                entry['ET'] = d['ET']; entry['mean'] = d['mean']
                agg[ms] = entry
            all_results[m_name] = agg
            elapsed = time.time() - t0
            print(f"  [{m_name}] done in {elapsed/60:.1f} min "
                  f"(n_cases_per_pattern={list(agg.values())[0]['count']})")
            # Save partial after each method
            partial_json = os.path.join(out_dir, f'partial_{m_name}.json')
            with open(partial_json, 'w') as f:
                json.dump({
                    'method': m_name,
                    'metrics': METRIC_NAMES,
                    'results': agg,
                }, f, indent=2)

    # -------------- Reporting (this run's methods only) --------------
    if is_main():
        ordered_methods = [m for m in args.methods]
        for metric in METRIC_NAMES:
            print('\n')
            print(format_table(all_results, patterns, ordered_methods, metric))
        for metric in METRIC_NAMES:
            print('\n')
            print(format_summary_table(all_results, patterns, ordered_methods, metric))
        print('\nPartial JSONs were written to:')
        for m in ordered_methods:
            print(f"  - {os.path.join(out_dir, f'partial_{m}.json')}")
        print('\nRun ``aggregate_results.py --partials-dir <out>`` to merge '
              'all partials (including any partial_SphereFlow.json from the '
              'separate worker) into results.json / results.csv.')

    cleanup_ddp()


if __name__ == '__main__':
    main()

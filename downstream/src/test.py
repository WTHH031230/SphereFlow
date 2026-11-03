"""Evaluate the trained 3D U-Net on the BraTS test set under all 15 modality
combinations (missing modalities are filled with zeros).

Modalities (channel order, see ``data_preprocess/preprocess_brats.py``):
    0 = T1   (t1n)
    1 = T1ce (t1c)
    2 = T2   (t2w)
    3 = FLAIR (t2f)

The 15 combinations are enumerated by all non-empty subsets of {0,1,2,3}.
For each subset, modalities not in the subset are zeroed out, then we run
the model and compute per-region Dice (WT/TC/ET).

Launch with torchrun for multi-GPU evaluation:

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        test.py --ckpt runs/brats_unet3d/checkpoints/best.pth
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from contextlib import nullcontext
from itertools import combinations

import numpy as np
import torch
import torch.distributed as dist

import config as C
from dataset import get_dataloader
from losses import per_channel_dice
from transforms import val_transform, val_transform_raw01
from unet3d import UNet3D


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return get_rank() == 0


def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True,
                   help='Path to the trained checkpoint (.pth)')
    p.add_argument('--json-path', type=str, default=C.JSON_PATH)
    p.add_argument('--phase', type=str, default='test',
                   choices=['train', 'val', 'test'])
    p.add_argument('--batch-size', type=int, default=C.TEST_BATCH_SIZE)
    p.add_argument('--num-workers', type=int, default=C.NUM_WORKERS)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--amp-dtype', type=str, default=C.AMP_DTYPE,
                   choices=['bf16', 'fp16'])
    p.add_argument('--output-dir', type=str, default='',
                   help='Directory to save the results CSV/JSON. '
                        'Default: <ckpt_dir>/../eval_results')
    p.add_argument('--no-normalize', action='store_true',
                   help='Feed raw [0,1] volumes to the UNet instead of '
                        'z-score-normalised inputs. Auto-detected from '
                        'the checkpoint config if present; this flag '
                        'forces it on from the CLI.')
    return p.parse_args()


def build_amp_context(use_amp: bool, dtype: str):
    if not use_amp:
        return nullcontext()
    amp_dtype = torch.bfloat16 if dtype == 'bf16' else torch.float16
    return torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True)


def enumerate_subsets(n_modalities: int = 4):
    """Return a list of (subset_tuple, name_string) for non-empty subsets."""
    out = []
    for k in range(1, n_modalities + 1):
        for sub in combinations(range(n_modalities), k):
            mask_str = ''.join('1' if i in sub else '0' for i in range(n_modalities))
            name = '+'.join(C.MODALITY_NAMES[i] for i in sub)
            out.append((sub, mask_str, name))
    return out


def main():
    args = parse_args()
    local_rank = setup_ddp()
    rank, world_size = get_rank(), get_world_size()

    device = torch.device(f'cuda:{local_rank}')
    torch.backends.cudnn.benchmark = True

    if is_main():
        print(f"[rank {rank}/{world_size}] device={device}")
        print(f"Args: {vars(args)}")

    # --------------------- model ---------------------
    model = UNet3D(in_channels=C.IN_CHANNELS, num_classes=C.NUM_CLASSES).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    ckpt_no_norm = bool(ckpt.get('config', {}).get('no_normalize', False))
    no_normalize = args.no_normalize or ckpt_no_norm
    if is_main():
        print(f"Loaded {args.ckpt}; missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"Missing keys: {missing[:8]}...")
        if unexpected:
            print(f"Unexpected keys: {unexpected[:8]}...")
        print(f"[normalize] {'DISABLED (raw [0,1])' if no_normalize else 'ENABLED (z-score)'} "
              f"(from ckpt config={ckpt_no_norm}, cli={args.no_normalize})")
    model.eval()

    # --------------------- data ---------------------
    t_val = val_transform_raw01 if no_normalize else val_transform
    loader, _, _ = get_dataloader(
        json_path=args.json_path,
        phase=args.phase,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=t_val,
        distributed=is_dist(),
        expected_depth=C.EXPECTED_DEPTH,
    )
    if is_main():
        print(f"Eval cases (phase={args.phase}): {len(loader.dataset)}")

    autocast_ctx = build_amp_context(use_amp=not args.no_amp, dtype=args.amp_dtype)

    subsets = enumerate_subsets(n_modalities=C.IN_CHANNELS)

    # accumulator: dict[mask_str] -> tensor (NUM_CLASSES,)
    n_classes = C.NUM_CLASSES
    dice_sum = {ms: torch.zeros(n_classes, device=device) for _, ms, _ in subsets}
    n_count = torch.zeros(1, device=device)

    n_total_cases = len(loader)
    with torch.no_grad():
        for it, batch in enumerate(loader):
            image_full = batch['image'].to(device, non_blocking=True)  # (B, 4, D, H, W)
            label = batch['label'].to(device, non_blocking=True)        # (B, 3, D, H, W)
            B = image_full.size(0)

            for sub, mask_str, _ in subsets:
                mask = torch.zeros(C.IN_CHANNELS, device=device)
                for c in sub:
                    mask[c] = 1.0
                # broadcast to (1, C, 1, 1, 1)
                mask = mask.view(1, -1, 1, 1, 1)
                image_masked = image_full * mask

                with autocast_ctx:
                    logits = model(image_masked)
                prob = torch.sigmoid(logits.float())
                dice = per_channel_dice(prob, label)  # (B, C)
                dice_sum[mask_str] += dice.sum(dim=0)

            n_count += B
            if is_main() and (it % 5 == 0):
                print(f"[{it+1}/{n_total_cases}] processed")

    # all-reduce across ranks
    if is_dist():
        dist.all_reduce(n_count, op=dist.ReduceOp.SUM)
        for v in dice_sum.values():
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
    n_total = max(n_count.item(), 1.0)

    # --------------------- aggregate + report ---------------------
    if is_main():
        out_dir = args.output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt))),
            'eval_results')
        os.makedirs(out_dir, exist_ok=True)

        results = []
        # group by number of available modalities for prettier reporting
        for sub, mask_str, name in subsets:
            d = (dice_sum[mask_str] / n_total).cpu().tolist()
            row = {
                'mask': mask_str,
                'modalities': name,
                'num_modalities': len(sub),
            }
            for i, region in enumerate(C.REGION_NAMES):
                row[f'dice_{region}'] = d[i]
            row['dice_mean'] = float(np.mean(d))
            results.append(row)

        # ----- print pretty table -----
        header = (f"{'#':>2}  {'mask':>6}  {'modalities':<25} "
                  f"{'WT':>8} {'TC':>8} {'ET':>8} {'mean':>8}")
        print('\n' + '=' * len(header))
        print(f"BraTS test results (n_cases={int(n_total)}, phase={args.phase})")
        print('=' * len(header))
        print(header)
        print('-' * len(header))
        # group by num modalities
        last_k = 0
        for row in sorted(results, key=lambda r: (r['num_modalities'], r['mask'])):
            if row['num_modalities'] != last_k:
                print('-' * len(header))
                last_k = row['num_modalities']
            print(f"{row['num_modalities']:>2}  {row['mask']:>6}  {row['modalities']:<25} "
                  f"{row['dice_WT']:>8.4f} {row['dice_TC']:>8.4f} "
                  f"{row['dice_ET']:>8.4f} {row['dice_mean']:>8.4f}")
        print('-' * len(header))

        # ----- per-k averages -----
        per_k = {}
        for row in results:
            per_k.setdefault(row['num_modalities'], []).append(row)
        print("\nAverages grouped by number of available modalities:")
        print(f"{'k':>2}  {'WT':>8} {'TC':>8} {'ET':>8} {'mean':>8}")
        for k in sorted(per_k):
            rows = per_k[k]
            wt = float(np.mean([r['dice_WT'] for r in rows]))
            tc = float(np.mean([r['dice_TC'] for r in rows]))
            et = float(np.mean([r['dice_ET'] for r in rows]))
            mean = float(np.mean([r['dice_mean'] for r in rows]))
            print(f"{k:>2}  {wt:>8.4f} {tc:>8.4f} {et:>8.4f} {mean:>8.4f}")

        # ----- overall average across all 15 combinations -----
        overall = {
            'WT': float(np.mean([r['dice_WT'] for r in results])),
            'TC': float(np.mean([r['dice_TC'] for r in results])),
            'ET': float(np.mean([r['dice_ET'] for r in results])),
        }
        overall['mean'] = float(np.mean(list(overall.values())))
        print("\nOverall average across all 15 combinations:")
        print(f"  WT={overall['WT']:.4f}  TC={overall['TC']:.4f}  "
              f"ET={overall['ET']:.4f}  mean={overall['mean']:.4f}")

        # ----- save -----
        json_out = os.path.join(out_dir, 'modality_combinations.json')
        with open(json_out, 'w') as f:
            json.dump({'phase': args.phase,
                       'ckpt': args.ckpt,
                       'n_cases': int(n_total),
                       'modality_order': C.MODALITY_NAMES,
                       'region_order': C.REGION_NAMES,
                       'results': results,
                       'overall': overall}, f, indent=2)
        csv_out = os.path.join(out_dir, 'modality_combinations.csv')
        with open(csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {json_out}\n       {csv_out}")

    cleanup_ddp()


if __name__ == '__main__':
    main()

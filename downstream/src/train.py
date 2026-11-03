"""DDP training for 3D U-Net on BraTS 2023.

Launch with torchrun, e.g.:

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

import config as C
from dataset import get_dataloader
from losses import DiceBCELoss, per_channel_dice
from transforms import (
    train_transform, val_transform,
    train_transform_raw01, val_transform_raw01,
)
from unet3d import UNet3D


# ------------------------------------------------------------------
# DDP / utility helpers
# ------------------------------------------------------------------

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


def set_seed(seed: int, rank: int = 0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return tensor
    out = tensor.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= get_world_size()
    return out


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int,
              min_lr_ratio: float = 0.01) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=C.TRAINING_EPOCH)
    p.add_argument('--batch-size', type=int, default=C.TRAIN_BATCH_SIZE,
                   help='per-GPU training batch size')
    p.add_argument('--val-batch-size', type=int, default=C.VAL_BATCH_SIZE)
    p.add_argument('--lr', type=float, default=C.LEARNING_RATE)
    p.add_argument('--weight-decay', type=float, default=C.WEIGHT_DECAY)
    p.add_argument('--warmup-epochs', type=int, default=C.WARMUP_EPOCHS)
    p.add_argument('--num-workers', type=int, default=C.NUM_WORKERS)
    p.add_argument('--output-dir', type=str, default=C.OUTPUT_DIR)
    p.add_argument('--json-path', type=str, default=C.JSON_PATH)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--amp-dtype', type=str, default=C.AMP_DTYPE,
                   choices=['bf16', 'fp16'])
    p.add_argument('--swanlab', action='store_true', default=True)
    p.add_argument('--no-swanlab', dest='swanlab', action='store_false')
    p.add_argument('--swanlab-project', type=str, default=C.SWANLAB_PROJECT)
    p.add_argument('--swanlab-experiment', type=str, default=C.SWANLAB_EXPERIMENT)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--no-normalize', action='store_true',
                   help='Skip z-score normalisation; feed raw [0,1] volumes '
                        'to the UNet. Intended for robustness against '
                        'generative-model background leakage at eval time.')
    return p.parse_args()


def build_amp_context(use_amp: bool, dtype: str):
    if not use_amp:
        return nullcontext(), None
    amp_dtype = torch.bfloat16 if dtype == 'bf16' else torch.float16
    autocast = torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True)
    # GradScaler is only needed for fp16; bf16 does not need it.
    scaler = torch.amp.GradScaler('cuda') if amp_dtype is torch.float16 else None
    return autocast, scaler


def main():
    args = parse_args()
    local_rank = setup_ddp()
    rank, world_size = get_rank(), get_world_size()
    set_seed(C.SEED, rank)

    device = torch.device(f'cuda:{local_rank}')
    torch.backends.cudnn.benchmark = True

    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
        print(f"[rank {rank}/{world_size}] device={device}")
        print(f"Args: {vars(args)}")

    # --------------------- swanlab ---------------------
    swan_run = None
    if args.swanlab and is_main():
        try:
            import swanlab
            swan_run = swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_experiment,
                config={**vars(args),
                        'in_channels': C.IN_CHANNELS,
                        'num_classes': C.NUM_CLASSES,
                        'input_size': C.INPUT_SIZE,
                        'world_size': world_size},
                logdir=os.path.join(args.output_dir, 'swanlab'),
            )
        except Exception as e:
            print(f"swanlab init failed: {e}; continuing without swanlab.")
            import traceback
            traceback.print_exc()
            swan_run = None

    # --------------------- data ---------------------
    if args.no_normalize:
        t_train, t_val = train_transform_raw01, val_transform_raw01
        if is_main():
            print("[normalize] DISABLED — using raw [0,1] inputs "
                  "(train_transform_raw01 / val_transform_raw01)")
    else:
        t_train, t_val = train_transform, val_transform
        if is_main():
            print("[normalize] ENABLED — NormalizeNonzeroPerChannel (z-score)")

    train_loader, train_sampler, _ = get_dataloader(
        json_path=args.json_path,
        phase='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=t_train,
        distributed=is_dist(),
        expected_depth=C.EXPECTED_DEPTH,
    )
    val_loader, val_sampler, _ = get_dataloader(
        json_path=args.json_path,
        phase='val',
        batch_size=args.val_batch_size,
        num_workers=max(1, args.num_workers // 2),
        transform=t_val,
        distributed=is_dist(),
        expected_depth=C.EXPECTED_DEPTH,
    )

    if is_main():
        print(f"train cases: {len(train_loader.dataset)}, "
              f"val cases: {len(val_loader.dataset)}")

    # --------------------- model ---------------------
    model = UNet3D(in_channels=C.IN_CHANNELS, num_classes=C.NUM_CLASSES).to(device)
    if is_dist():
        # SyncBN: aggregate BN stats across all DDP ranks so the effective batch
        # size for BatchNorm is `world_size * per_gpu_batch` (8 for 4xGPU x batch=2).
        # Without this the per-rank batch of 2 produces extremely noisy running
        # stats and validation Dice swings wildly between epochs.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    criterion = DiceBCELoss(dice_weight=C.DICE_WEIGHT, bce_weight=C.BCE_WEIGHT)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    autocast_ctx, scaler = build_amp_context(use_amp=not args.no_amp,
                                             dtype=args.amp_dtype)

    iters_per_epoch = len(train_loader)
    total_iters = iters_per_epoch * args.epochs
    warmup_iters = iters_per_epoch * args.warmup_epochs

    # --------------------- resume ---------------------
    start_epoch = 0
    best_val_dice = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        net = model.module if is_dist() else model
        net.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_dice = ckpt.get('best_val_dice', 0.0)
        if is_main():
            print(f"Resumed from {args.resume}, starting at epoch {start_epoch}, "
                  f"best_val_dice={best_val_dice:.4f}")

    # --------------------- training loop ---------------------
    global_step = start_epoch * iters_per_epoch

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        t0 = time.time()
        train_loss_sum, train_dice_sum, train_bce_sum, n_batches = 0.0, 0.0, 0.0, 0

        for it, batch in enumerate(train_loader):
            image = batch['image'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)

            lr = cosine_lr(global_step, total_iters, args.lr, warmup_iters)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(image)
                loss, dice_loss, bce_loss = criterion(logits, label)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            train_loss_sum += loss.item()
            train_dice_sum += dice_loss.item()
            train_bce_sum += bce_loss.item()
            n_batches += 1
            global_step += 1

            if is_main() and (it % C.LOG_INTERVAL == 0):
                elapsed = time.time() - t0
                ips = (it + 1) * image.size(0) * world_size / max(elapsed, 1e-6)
                print(f"[Epoch {epoch+1}/{args.epochs}] iter {it+1}/{iters_per_epoch} "
                      f"lr={lr:.2e} loss={loss.item():.4f} "
                      f"dice_loss={dice_loss.item():.4f} "
                      f"bce_loss={bce_loss.item():.4f} "
                      f"({ips:.1f} samples/s)  [lower is better]")
                if swan_run is not None:
                    swan_run.log({
                        'train/loss_step': loss.item(),
                        'train/dice_loss_step': dice_loss.item(),
                        'train/bce_loss_step': bce_loss.item(),
                        'train/lr': lr,
                    }, step=global_step)

        # epoch-level training metrics (averaged across ranks)
        loss_t = torch.tensor([train_loss_sum, train_dice_sum, train_bce_sum,
                               float(n_batches)], device=device)
        if is_dist():
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        n_total = loss_t[3].item()
        train_loss = loss_t[0].item() / max(n_total, 1)
        train_dice = loss_t[1].item() / max(n_total, 1)
        train_bce = loss_t[2].item() / max(n_total, 1)

        # --------------------- validation ---------------------
        val_dice_per_region = None
        val_loss = None
        if (epoch + 1) % C.VAL_INTERVAL == 0 or (epoch + 1) == args.epochs:
            val_dice_per_region, val_loss = validate(
                model, val_loader, criterion, device, autocast_ctx)

        if is_main():
            log = {
                'train/loss': train_loss,
                'train/dice_loss': train_dice,
                'train/bce_loss': train_bce,
                'epoch': epoch + 1,
            }
            msg = (f"Epoch {epoch+1:03d}/{args.epochs} "
                   f"train_loss={train_loss:.4f} "
                   f"(dice_loss={train_dice:.4f} bce_loss={train_bce:.4f})")
            if val_dice_per_region is not None:
                for name, d in zip(C.REGION_NAMES, val_dice_per_region):
                    log[f'val/dice_{name}'] = d
                mean_dice = float(np.mean(val_dice_per_region))
                log['val/dice_mean'] = mean_dice
                log['val/loss'] = val_loss
                msg += (f" | val_loss={val_loss:.4f}  "
                        f"DICE  WT={val_dice_per_region[0]:.4f} "
                        f"TC={val_dice_per_region[1]:.4f} "
                        f"ET={val_dice_per_region[2]:.4f} "
                        f"mean={mean_dice:.4f}  [higher is better]")
            print(msg)
            if swan_run is not None:
                swan_run.log(log, step=global_step)

            # --------------------- checkpointing ---------------------
            net = model.module if is_dist() else model
            ckpt = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_dice': best_val_dice,
                'config': {
                    'in_channels': C.IN_CHANNELS,
                    'num_classes': C.NUM_CLASSES,
                    'input_size': C.INPUT_SIZE,
                    'no_normalize': bool(args.no_normalize),
                },
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'checkpoints', 'last.pth'))
            if (epoch + 1) % C.SAVE_INTERVAL == 0:
                torch.save(ckpt, os.path.join(args.output_dir, 'checkpoints',
                                              f'epoch_{epoch+1:03d}.pth'))
            if val_dice_per_region is not None:
                mean_dice = float(np.mean(val_dice_per_region))
                if mean_dice > best_val_dice:
                    best_val_dice = mean_dice
                    ckpt['best_val_dice'] = best_val_dice
                    torch.save(ckpt, os.path.join(args.output_dir,
                                                  'checkpoints', 'best.pth'))
                    print(f"  ** new best val dice = {best_val_dice:.4f}, "
                          f"saved best.pth")

        if is_dist():
            dist.barrier()

    if swan_run is not None:
        swan_run.finish()
    cleanup_ddp()


@torch.no_grad()
def validate(model, val_loader, criterion, device, autocast_ctx):
    model.eval()
    n = 0
    dice_sum = torch.zeros(C.NUM_CLASSES, device=device)
    loss_sum = torch.tensor(0.0, device=device)
    for batch in val_loader:
        image = batch['image'].to(device, non_blocking=True)
        label = batch['label'].to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(image)
            loss, _, _ = criterion(logits, label)
        prob = torch.sigmoid(logits.float())
        dice = per_channel_dice(prob, label)  # (B, C)
        dice_sum += dice.sum(dim=0)
        loss_sum += loss * image.size(0)
        n += image.size(0)

    n_t = torch.tensor(float(n), device=device)
    if is_dist():
        dist.all_reduce(dice_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_t, op=dist.ReduceOp.SUM)
    n_total = max(n_t.item(), 1.0)
    return (dice_sum / n_total).cpu().tolist(), (loss_sum / n_total).item()


if __name__ == '__main__':
    main()

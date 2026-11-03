# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
Stage-1 RAE training script with ALL-modality auxiliary support.

When training modality M, all other modalities can serve as auxiliaries. Each step
randomly samples a non-empty subset of auxiliary modalities, averages their latent z
vectors, then applies the w-weighted interpolation: z_c = w * z_a + (1-w) * mean(z_aux).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import shutil
from glob import glob
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from eval import evaluate_reconstruction_distributed
from torchmetrics.functional.image import structural_similarity_index_measure
from disc import (
    DiffAug,
    LPIPS,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)
from stage1 import RAE

##### general utils
from utils import wandb_utils, swanlab_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import *
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *
from PIL import Image
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 RAE with GAN and LPIPS losses.")
    parser.add_argument("--config", type=str, default="configs/stage1/brats.yaml", help="YAML config containing a stage_1 section.")
    parser.add_argument("--data-path", type=str, default="brats2023_distribution.json", help="Directory with ImageFolder structure or path to BraTS JSON.")
    parser.add_argument("--dataset", type=str, default="brats", choices=["imagefolder", "brats", "mri_ct"], help="Dataset type.")
    parser.add_argument("--modality", type=int, default=None, choices=[0, 1, 2, 3], help="Modality index for BraTS dataset.")
    parser.add_argument("--use-all", action="store_true", help="Use all-modality auxiliary: randomly sample a non-empty subset of remaining modalities, average their z, interpolate with w.")
    parser.add_argument("--results-dir", type=str, default="results/stage1", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, default=256, help="Image resolution (assumes square images).")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--global-batch-size", type=int, default=None, help="Override training.global_batch_size from the config.")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop training after this many steps (useful for memory testing).")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")    
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging if set.')
    parser.add_argument('--swanlab', action='store_true', help='Use SwanLab for logging if set.')
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for rae.encode, rae.forward).")
    parser.add_argument("--pretrained-decoder-path", type=str, default=None, help="Path to pre-trained decoder weights.")
    parser.add_argument("--psnr-weight", type=float, default=0.0, help="Weight for PSNR loss.")
    parser.add_argument("--ssim-weight", type=float, default=0.0, help="Weight for SSIM loss.")
    parser.add_argument("--w-sampling-mode", type=str, default="uniform", choices=["uniform", "power"], help="Mode for sampling interpolation weight w.")
    parser.add_argument("--w-sampling-exponent", type=float, default=2.0, help="Exponent for power-based w sampling. If mode is 'power', w = 1 - (1 - u)^exp, biasing towards 1.")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping validation per epoch based on rFID, PSNR, SSIM.")
    parser.add_argument("--noise-in-encode", action="store_true", help="Add noise inside encode() (before interpolation) instead of before decode(). Restores original noising behavior.")
    parser.add_argument("--adaptive-weight-mode", type=str, default="gradient", choices=["gradient", "loss_ratio", "fixed"], help="How to compute adaptive weight: gradient (original, high VRAM), loss_ratio (loss-based approx, zero overhead), fixed (always 1.0).")
    parser.add_argument("--pretrained-disc-path", type=str, default=None, help="Path to checkpoint containing pretrained discriminator weights (loaded from 'disc' key).")
    return parser.parse_args()


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


def select_gan_losses(disc_kind: str, gen_kind: str):
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")

    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, gen_loss_fn


@torch.no_grad()
def generate_reference_npz(dataset, path, batch_size=128, num_workers=4):
    from tqdm import tqdm
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    all_images = []
    for images, _ in tqdm(loader, desc="Generating reference NPZ"):
        # images is [B, C, H, W], convert to [B, H, W, C] uint8
        img_np = images.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        all_images.append(img_np)
    all_images = np.concatenate(all_images, axis=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, arr_0=all_images)


def get_base_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": get_base_model(model).state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "disc": get_base_model(disc).state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    get_base_model(model).load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    get_base_model(disc).load_state_dict(checkpoint["disc"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)

def main():
    args = parse_args()
    #### Dist Init
    rank, world_size, device = setup_distributed()
    
    #### Config init
    full_cfg = OmegaConf.load(args.config)
    (rae_config, *_) = parse_configs(full_cfg)
    training_section = full_cfg.get("training", None)
    training_cfg = OmegaConf.to_container(training_section, resolve=True) if training_section is not None else {}
    training_cfg = dict(training_cfg) if isinstance(training_cfg, dict) else {}

    gan_section = full_cfg.get("gan", None)
    gan_cfg = OmegaConf.to_container(gan_section, resolve=True) if gan_section is not None else {}
    if not gan_cfg:
        raise ValueError("Config must define a top-level 'gan' section for stage-1 training.")
    disc_cfg = gan_cfg.get("disc", {})
    if not disc_cfg:
        raise ValueError("gan.disc configuration is required for stage-1 training.")
    loss_cfg = gan_cfg.get("loss", {})
    perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
    disc_weight = float(loss_cfg.get("disc_weight", 0.0))
    gan_start_epoch = int(loss_cfg.get("disc_start", 0))
    disc_update_epoch = int(loss_cfg.get("disc_upd_start", gan_start_epoch))
    lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))
    
    disc_updates = int(loss_cfg.get("disc_updates", 1))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    disc_loss_type = loss_cfg.get("disc_loss", "hinge")
    gen_loss_type = loss_cfg.get("gen_loss", "vanilla")
    batch_size = int(training_cfg.get("batch_size", 16))
    global_batch_size = args.global_batch_size if args.global_batch_size is not None else training_cfg.get("global_batch_size", None)
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
        batch_size = global_batch_size // world_size
    else:
        global_batch_size = batch_size * world_size
    num_workers = int(training_cfg.get("num_workers", 4))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 1250)) 
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4)) # ckpt interval is epoch based
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 200))
    default_seed = int(training_cfg.get("global_seed", 0))
    eval_section = full_cfg.get("eval", None)
    
    if eval_section:
        do_eval = True
        eval_interval = int(eval_section.get("eval_interval", 5000))
        eval_model = eval_section.get("eval_model", False) # by default eval ema. This decides whether to **additionally** eval the non-ema model.
        eval_metrics = list(eval_section.get("metrics", ["rfid", "psnr", "ssim", "is"])) # by default eval all
        eval_data = eval_section.get("data_path", None)
        reference_npz_path = eval_section.get("reference_npz_path", None)
        assert eval_data, "eval.data_path must be specified to enable evaluation."
        
        assert len(eval_metrics) > 0, "eval.metrics must contain at least one metric to compute."
    else:
        do_eval = False
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    # update args as a dict to full_cfg
    full_cfg.cmd_args = vars(args)
    full_cfg.experiment_dir = experiment_dir
    full_cfg.checkpoint_dir = checkpoint_dir
    
    if args.pretrained_decoder_path is not None:
        rae_config.ckpt = args.pretrained_decoder_path
        logger.info(f"Overriding pretrained_decoder_path with: {args.pretrained_decoder_path}")
    
    #### Model init
    rae: RAE = instantiate_from_config(rae_config).to(device)
    if args.compile:
        rae.encode = torch.compile(rae.encode)
        rae.forward = torch.compile(rae.forward)
    rae.encoder.eval()
    rae.decoder.train()
    ema_model = deepcopy(rae).to(device).eval()
    ema_model.requires_grad_(False)
    # only train decoder
    rae.encoder.requires_grad_(False)
    rae.decoder.requires_grad_(True)
    if world_size > 1:
        ddp_model = DDP(rae, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=False, find_unused_parameters=True)  # type: ignore[arg-type]
    else:
        ddp_model = rae
    rae = get_base_model(ddp_model)
    decoder = rae.decoder
    discriminator, disc_aug = build_discriminator(disc_cfg, device)
    if args.pretrained_disc_path is not None:
        disc_ckpt = torch.load(args.pretrained_disc_path, map_location="cpu")
        disc_sd = disc_ckpt.get("disc", disc_ckpt)
        discriminator.load_state_dict(disc_sd, strict=False)
        logger.info(f"Loaded pretrained discriminator from: {args.pretrained_disc_path}")
        del disc_ckpt, disc_sd
    if world_size > 1:
        ddp_disc = DDP(discriminator, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=False, find_unused_parameters=True)  # type: ignore[arg-type]
    else:
        ddp_disc = discriminator
    discriminator = ddp_disc
    disc_scheduler: LambdaLR | None = None
    disc_sched_msg: Optional[str] = None

    discriminator.train()
    disc_loss_fn, gen_loss_fn = select_gan_losses(disc_loss_type, gen_loss_type)
    
    lpips = LPIPS().to(device)
    lpips.eval()
    
    #### Opt, Schedl init
    optimizer, optim_msg = build_optimizer(decoder.parameters(), training_cfg)
    disc_params = [p for p in discriminator.parameters() if p.requires_grad]
    disc_optimizer, disc_optim_msg = build_optimizer(disc_params, disc_cfg)
    
    #### AMP init
    scaler, autocast_kwargs = get_autocast_scaler(args)
    
    
    #### Data init
    use_all = getattr(args, "use_all", False) and args.dataset in ["brats", "mri_ct"]
    all_modalities = [0, 1] if args.dataset == "mri_ct" else [0, 1, 2, 3]
    if use_all:
        aux_modalities = [m for m in all_modalities if m != args.modality]
        # Pre-compute all non-empty subsets of auxiliary modalities
        aux_subsets: List[List[int]] = []
        for r in range(1, len(aux_modalities) + 1):
            for subset in combinations(aux_modalities, r):
                aux_subsets.append(list(subset))
        logger.info(f"All-modality mode: main={args.modality}, aux subsets={aux_subsets}")

    if args.dataset in ["brats", "mri_ct"]:
        import sys
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())
        from brats_dataset import BraTS2023Dataset, BraTS2023AllDataset, get_transforms

        train_transform = get_transforms('train')
        val_transform = get_transforms('val')

        if use_all:
            num_mods = 2 if args.dataset == "mri_ct" else 4
            loader_dataset = BraTS2023AllDataset(
                json_path=args.data_path,
                phase='train',
                transform=train_transform,
                num_modalities=num_mods
            )
            logger.info(f"{args.dataset.upper()} all-modality: main={args.modality}, all {num_mods} modalities loaded.")
        else:
            loader_dataset = BraTS2023Dataset(
                json_path=args.data_path,
                phase='train',
                transform=train_transform,
                modality_index=args.modality
            )
        sampler = DistributedSampler(loader_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(
            loader_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if do_eval:
            eval_dataset = BraTS2023Dataset(
                json_path=args.data_path,
                phase='test',
                transform=val_transform,
                modality_index=args.modality
            )
            logger.info(f"{args.dataset.upper()} Evaluation dataset loaded from {args.data_path}, modality {args.modality}, containing {len(eval_dataset)} images.")
    else:
        first_crop_size = 384 if args.image_size == 256 else int(args.image_size * 1.5)
        stage1_transform = transforms.Compose(
            [
                transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(args.image_size),
                transforms.ToTensor(),
            ]
        )    
        loader, sampler = prepare_dataloader(
            Path(args.data_path), batch_size, num_workers, rank, world_size, transform=stage1_transform
        )
        if do_eval:
            eval_dataset = ImageFolder(
                str(eval_data),
                transform=transforms.Compose([
                    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                    transforms.ToTensor(),
                ])
            )
            logger.info(f"Evaluation dataset loaded from {eval_data}, containing {len(eval_dataset)} images.")

        # Automatic generation of reference_npz_path if it doesn't exist
        if do_eval and reference_npz_path is not None and not os.path.exists(reference_npz_path):
            if rank == 0:
                logger.info(f"reference_npz_path {reference_npz_path} does not exist. Generating it from eval_dataset...")
                generate_reference_npz(eval_dataset, reference_npz_path, batch_size=batch_size, num_workers=num_workers)
                logger.info("Reference NPZ generation complete.")
            if world_size > 1:
                dist.barrier()
    
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches. Check dataset and batch size settings.")
    
    # Schedl init after knowing dataset length
    scheduler: LambdaLR | None = None
    sched_msg: Optional[str] = None
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    if disc_cfg.get("scheduler"):
        disc_scheduler, disc_sched_msg = build_scheduler(disc_optimizer, steps_per_epoch, disc_cfg)
    
    ### Resuming and checkpointing
    start_epoch = 0
    global_step = 0
    maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    if maybe_resume_ckpt_path is not None:
        logger.info(f"Experiment resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming...")
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
                discriminator,
                disc_optimizer,
                disc_scheduler,
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # starting from fresh, save worktree and configs
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")
    
    ### Logging experiment details
    if rank == 0:
        num_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        logger.info(f"Stage-1 RAE trainable parameters: {num_params/1e6:.2f}M")
        logger.info(f"Discriminator architecture:\n{discriminator}")
        num_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        logger.info(f"Discriminator trainable parameters: {num_params/1e6:.2f}M")
        logger.info(f"Using {disc_loss_type} discriminator loss and {gen_loss_type} generator loss.")
        logger.info(f"Perceptual (LPIPS) weight: {perceptual_weight:.6f}, GAN weight: {disc_weight:.6f}")
        logger.info(f"GAN training starts at epoch {gan_start_epoch}, discriminator updates start at epoch {disc_update_epoch}, LPIPS loss starts at epoch {lpips_start_epoch}.")
        if disc_aug is not None:
            logger.info(f"Using DiffAug with policies: {disc_aug}")
        else:
            logger.info("Not using DiffAug.")
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        # print optim and schel
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler for generator.")
        logger.info(disc_optim_msg)
        print(disc_sched_msg if disc_sched_msg else "No LR scheduler for discriminator.")
        logger.info(f"Training for {num_epochs} epochs, batch size {batch_size} per GPU.")
        if getattr(args, "use_all", False):
            bias_str = " (biased towards 1)" if args.w_sampling_mode == 'power' else ""
            logger.info(f"All-modality sampling mode: {args.w_sampling_mode}{bias_str} (exponent: {args.w_sampling_exponent if args.w_sampling_mode == 'power' else 'N/A'})")
        logger.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")


    last_layer = decoder.decoder_pred.weight
    gan_start_step = gan_start_epoch * steps_per_epoch
    disc_update_step = disc_update_epoch * steps_per_epoch
    lpips_start_step = lpips_start_epoch * steps_per_epoch
    
    # Decide where to apply noise: inside encode() or before decode()
    _orig_noise_tau = rae.noise_tau
    if not args.noise_in_encode:
        rae.noise_tau = 0
        ema_model.noise_tau = 0
        if rank == 0:
            logger.info(f"Disabled encode noising (orig noise_tau={_orig_noise_tau}), will apply noise before decode instead.")
    else:
        if rank == 0:
            logger.info(f"Keeping noise inside encode() (noise_tau={_orig_noise_tau}), noise applied before interpolation.")
    
    if world_size > 1:
        dist.barrier()
        
    base_metrics = None
    if getattr(args, "early_stop", False) and do_eval:
        if rank == 0:
            logger.info("Early stop enabled. Doing initial evaluation to determine baseline metrics...")
        eval_mod = ema_model if not eval_model else get_base_model(ddp_model)
        eval_stats = evaluate_reconstruction_distributed(
            eval_mod,
            eval_dataset,
            len(eval_dataset),
            rank=rank,
            world_size=world_size,
            device=device,
            batch_size=batch_size,
            metrics_to_compute=eval_metrics,
            experiment_dir=experiment_dir,
            global_step=global_step,
            autocast_kwargs=autocast_kwargs,
            reference_npz_path=reference_npz_path,
            train_dataset=loader_dataset if 'ifid' in eval_metrics else None,
        )
        if rank == 0 and eval_stats is not None:
            base_metrics = eval_stats
            logger.info(f"Baseline metrics: {base_metrics}")
        if world_size > 1:
            dist.barrier()
            
    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        rae.encoder.eval() # encoder always in eval mode
        sampler.set_epoch(epoch)
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        if use_all:
            epoch_metrics["inter_recon"] = torch.zeros(1, device=device)
            epoch_metrics["inter_lpips"] = torch.zeros(1, device=device)
            epoch_metrics["inter_gan"] = torch.zeros(1, device=device)
            epoch_metrics["inter_psnr"] = torch.zeros(1, device=device)
            epoch_metrics["inter_ssim"] = torch.zeros(1, device=device)
        num_batches = 0
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0  and rank == 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt" 
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
                discriminator,
                disc_optimizer,
                disc_scheduler,
            )
            # Only keep the latest two checkpoints
            checkpoints = sorted(glob(f"{checkpoint_dir}/ep-[0-9]*.pt"))
            if len(checkpoints) > 1:
                for old_ckpt in checkpoints[:-2]:
                    os.remove(old_ckpt)
        if world_size > 1:
            dist.barrier()
        for step, batch in enumerate(loader):
            if use_all:
                # batch is a tuple of all modality images
                num_modalities = 2 if args.dataset == "mri_ct" else 4
                all_imgs = [batch[m].to(device, non_blocking=True) for m in range(num_modalities)]
                x_a = all_imgs[args.modality]
            else:
                x_a = batch[0].to(device, non_blocking=True)
                all_imgs = None
            images = x_a
            B = images.size(0)
            rae_base = get_base_model(ddp_model)

            use_gan = global_step >= gan_start_step and disc_weight > 0.0
            train_disc = global_step >= disc_update_step and disc_weight > 0.0
            use_lpips = global_step >= lpips_start_step and perceptual_weight > 0.0
            real_normed = images * 2.0 - 1.0
            optimizer.zero_grad(set_to_none=True)
            discriminator.eval()
            with autocast(**autocast_kwargs):
                # Main branch: x_a -> z_a -> x_rec (use rae_base to avoid DDP module issues on multi-GPU)
                z_a = rae_base.encode(images)  # noise_tau=0 when noise_in_encode=False, else noising already applied
                if not args.noise_in_encode:
                    z_a_noised = rae_base.noising(z_a) if (rae_base.training and _orig_noise_tau > 0) else z_a
                else:
                    z_a_noised = z_a
                recon = rae_base.decode(z_a_noised)
                recon_normed = recon * 2.0 - 1.0
                rec_loss = (recon - images).abs().mean() # L1
                # PSNR and SSIM loss
                mse = F.mse_loss(recon, images)
                psnr_loss = -10 * torch.log10(1.0 / (mse + 1e-8))
                ssim_loss = 1.0 - structural_similarity_index_measure(recon, images, data_range=1.0)

                if use_lpips:
                    lpips_loss = lpips(real_normed, recon_normed)
                else:
                    lpips_loss = rec_loss.new_zeros(())
                recon_total = rec_loss + perceptual_weight * lpips_loss + args.psnr_weight * psnr_loss + args.ssim_weight * ssim_loss
                if use_gan:
                    fake_aug = disc_aug.aug(recon_normed)
                    logits_fake, _ = ddp_disc(fake_aug, None)
                    gan_loss = gen_loss_fn(logits_fake)
                else:
                    gan_loss = torch.zeros_like(recon_total)

                # All-modality: randomly sample auxiliary subset, average their z,
                # z_c = w*z_a + (1-w)*z_aux_avg, x_rec_c; add w-weighted L1/LPIPS/GAN for x_rec_c
                inter_recon = rec_loss.new_zeros(())
                inter_lpips = rec_loss.new_zeros(())
                inter_gan_raw = rec_loss.new_zeros(())
                if use_all and all_imgs is not None:
                    # Randomly pick one non-empty subset of auxiliary modalities
                    chosen_subset = random.choice(aux_subsets)
                    # Encode each auxiliary modality and average z (no noise)
                    z_aux_list = []
                    for aux_m in chosen_subset:
                        z_aux_list.append(rae_base.encode(all_imgs[aux_m]))
                    z_aux_avg = torch.stack(z_aux_list, dim=0).mean(dim=0)
                    w = torch.rand(B, device=x_a.device, dtype=x_a.dtype)
                    if getattr(args, "w_sampling_mode", "uniform") == "power":
                        # Bias towards 1: w = 1 - (1 - u)^exp
                        w = 1.0 - (1.0 - w).pow(args.w_sampling_exponent)
                    w_z = w.view(B, *([1] * (z_a.dim() - 1)))
                    z_c = w_z * z_a + (1.0 - w_z) * z_aux_avg
                    if not args.noise_in_encode:
                        # z's are clean, add noise after interpolation
                        z_c_noised = rae_base.noising(z_c) if (rae_base.training and _orig_noise_tau > 0) else z_c
                    else:
                        # z's already noisy from encode(), no extra noise
                        z_c_noised = z_c
                    x_rec_c = rae_base.decode(z_c_noised)
                    target_c = images
                    rec_c_per_sample = (x_rec_c - target_c).abs().mean(dim=(1, 2, 3))
                    mse_c_per_sample = (x_rec_c - target_c).pow(2).mean(dim=(1, 2, 3))
                    psnr_c_per_sample = -10 * torch.log10(1.0 / (mse_c_per_sample + 1e-8))
                    ssim_c_per_sample = 1.0 - structural_similarity_index_measure(x_rec_c, target_c, data_range=1.0, reduction="none")

                    if use_lpips:
                        real_c_normed = target_c * 2.0 - 1.0
                        rec_c_normed = x_rec_c * 2.0 - 1.0
                        lpips_c_per_sample = lpips(real_c_normed, rec_c_normed, reduction="none").squeeze()
                    else:
                        lpips_c_per_sample = rec_c_per_sample.new_zeros(B, device=rec_c_per_sample.device)
                    inter_recon = (w * rec_c_per_sample).mean()
                    inter_lpips = (w * perceptual_weight * lpips_c_per_sample).mean()
                    inter_psnr = (w * args.psnr_weight * psnr_c_per_sample).mean()
                    inter_ssim = (w * args.ssim_weight * ssim_c_per_sample).mean()
                    if use_gan:
                        rec_c_normed = x_rec_c * 2.0 - 1.0
                        fake_c_aug = disc_aug.aug(rec_c_normed)
                        logits_fake_c, _ = ddp_disc(fake_c_aug, None)
                        w_expanded = w.view(B, 1).expand_as(logits_fake_c)
                        inter_gan_raw = (w_expanded * (-logits_fake_c)).mean()
            # Calculate adaptive weight outside autocast (autograd operation, not forward pass)
            if use_gan:
                if args.adaptive_weight_mode == "gradient":
                    adaptive_weight = calculate_adaptive_weight(
                        recon_total, gan_loss, last_layer, max_d_weight
                    )
                elif args.adaptive_weight_mode == "loss_ratio":
                    with torch.no_grad():
                        adaptive_weight = (recon_total.detach() / (gan_loss.detach().abs() + 1e-6)).clamp(0.0, max_d_weight)
                else:
                    adaptive_weight = torch.ones(1, device=recon_total.device)
                total_loss = recon_total + disc_weight * adaptive_weight * gan_loss
            else:
                adaptive_weight = torch.zeros_like(recon_total)
                total_loss = recon_total
            if use_all and all_imgs is not None:
                total_loss = total_loss + inter_recon + inter_lpips + inter_psnr + inter_ssim + disc_weight * adaptive_weight * inter_gan_raw
            total_loss.float()
            if scaler:
                scaler.scale(total_loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            update_ema(ema_model, get_base_model(ddp_model), ema_decay)

            disc_metrics: Dict[str, torch.Tensor] = {}
            if train_disc:
                # Set model to eval mode and get fresh reconstruction with updated weights
                ddp_model.eval()
                ddp_disc.train()
                for _ in range(disc_updates):
                    disc_optimizer.zero_grad(set_to_none=True)
                    with autocast(**autocast_kwargs):
                        # Fresh forward pass with updated model weights (no gradient)
                        with torch.no_grad():
                            recon_disc = ddp_model(images)
                            recon_disc_normed = recon_disc * 2.0 - 1.0
                        # discretize
                        fake_detached = recon_disc_normed.clamp(-1.0, 1.0)
                        fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                        fake_input = disc_aug.aug(fake_detached)
                        real_input = disc_aug.aug(real_normed)
                        logits_fake, logits_real = discriminator(fake_input, real_input)
                        d_loss = disc_loss_fn(logits_real, logits_fake)
                        accuracy = (logits_real > logits_fake).float().mean() # accuracy of the discriminator
                    d_loss.float()
                    if scaler:
                        scaler.scale(d_loss).backward()
                        scaler.step(disc_optimizer)
                        scaler.update()
                    else:
                        d_loss.backward()
                        disc_optimizer.step()
                    disc_metrics = {
                        "disc_loss": d_loss.detach(),
                        "logits_real": logits_real.detach().mean(),
                        "logits_fake": logits_fake.detach().mean(),
                        "disc_accuracy": accuracy.detach(),
                    }
                    epoch_metrics["disc_loss"] += d_loss.detach()
                    epoch_metrics["disc_accuracy"] += accuracy.detach()
                    if disc_scheduler is not None:
                        disc_scheduler.step()
                ddp_disc.eval()
                # Set model back to train mode
                ddp_model.train()
                rae.encoder.eval() # encoder always in eval mode

            epoch_metrics["recon"] += rec_loss.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            epoch_metrics["psnr"] += psnr_loss.detach()
            epoch_metrics["ssim"] += ssim_loss.detach()
            epoch_metrics["gan"] += gan_loss.detach()
            epoch_metrics["total"] += total_loss.detach()
            if use_all and all_imgs is not None:
                epoch_metrics["inter_recon"] += inter_recon.detach()
                epoch_metrics["inter_lpips"] += inter_lpips.detach()
                epoch_metrics["inter_gan"] += inter_gan_raw.detach()
                epoch_metrics["inter_psnr"] += inter_psnr.detach()
                epoch_metrics["inter_ssim"] += inter_ssim.detach()
            num_batches += 1

            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                stats = {
                    "loss/total": total_loss.detach().item(),
                    "loss/recon": rec_loss.detach().item(),
                    "loss/lpips": lpips_loss.detach().item(),
                    "loss/psnr": psnr_loss.detach().item(),
                    "loss/ssim": ssim_loss.detach().item(),
                    "loss/gan": gan_loss.detach().item(),
                    "lr/generator": optimizer.param_groups[0]["lr"],
                }
                if use_all and all_imgs is not None:    
                    stats["loss/inter_recon"] = inter_recon.detach().item()
                    stats["loss/inter_lpips"] = inter_lpips.detach().item()
                    stats["loss/inter_psnr"] = inter_psnr.detach().item()
                    stats["loss/inter_ssim"] = inter_ssim.detach().item()
                    stats["loss/inter_gan"] = inter_gan_raw.detach().item()
                if disc_metrics:
                    stats.update(
                        {
                            "loss/disc": disc_metrics["disc_loss"].item(),
                            "disc/logits_real": disc_metrics["logits_real"].item(),
                            "disc/logits_fake": disc_metrics["logits_fake"].item(),
                            "lr/discriminator": disc_optimizer.param_groups[0]["lr"],
                            "disc/accuracy": disc_metrics["disc_accuracy"].item(),
                            "disc/weight": adaptive_weight.item(),
                        }
                    )
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(stats, step=global_step)
                if args.swanlab:
                    swanlab_utils.log(stats, step=global_step)
            if global_step % sample_every == 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    if use_all and all_imgs is not None:
                        # n_vis: number of samples per modality to show
                        n_vis = 2
                        rows = []
                        
                        # Row 1 & 2: Reconstructions for all modalities
                        for m_idx in range(num_modalities):
                            samp = all_imgs[m_idx][:n_vis]
                            recs = ema_model.decode(ema_model.encode(samp))
                            # Combine GT and Rec for this modality
                            rows.append(torch.cat([samp, recs], dim=0)) # [4, C, H, W]
                        
                        # Row 3+: Cross-modality generation (Interpolation with w=0)
                        if num_modalities >= 2:
                            for target_m in range(num_modalities):
                                aux_indices = [i for i in range(num_modalities) if i != target_m]
                                z_aux_list = [ema_model.encode(all_imgs[i][:n_vis]) for i in aux_indices]
                                z_aux_avg = torch.stack(z_aux_list, dim=0).mean(dim=0)
                                # Each run trains a single decoder, so decode() takes no modality index
                                gen_target = ema_model.decode(z_aux_avg)
                                target_gt = all_imgs[target_m][:n_vis]
                                rows.append(torch.cat([target_gt, gen_target], dim=0))
                        
                        comparison = torch.cat(rows, dim=0).cpu().float()
                        grid = make_grid(comparison, nrow=n_vis)
                    else:
                        sample_images = images[:4]
                        samples = ema_model.decode(ema_model.encode(sample_images))
                        comparison = torch.cat([sample_images, samples], dim=0).cpu().float()
                        grid = make_grid(comparison, nrow=sample_images.size(0))
                    
                    if args.wandb:
                        wandb_utils.log_image(grid, step=global_step)
                    if args.swanlab:
                        swanlab_utils.log_image(grid, step=global_step)
                logger.info("Generating EMA samples done.")
            if do_eval and (eval_interval > 0 and global_step % eval_interval == 0):
                logger.info("Starting evaluation...")
                eval_models = [(ema_model, "ema")]
                if eval_model:
                    eval_models.append((get_base_model(ddp_model), "model"))
                for eval_mod, mod_name in eval_models:
                    eval_stats = evaluate_reconstruction_distributed(
                        eval_mod,
                        eval_dataset,
                        len(eval_dataset),
                        rank = rank,
                        world_size = world_size,
                        device = device,
                        batch_size = batch_size,
                        metrics_to_compute = eval_metrics,
                        experiment_dir = experiment_dir,
                        global_step = global_step,
                        autocast_kwargs = autocast_kwargs,
                        reference_npz_path = reference_npz_path,
                        train_dataset = loader_dataset if 'ifid' in eval_metrics else None,
                    )
                    # log with prefix
                    eval_stats = {f"eval_{mod_name}/{k}": v for k, v in eval_stats.items()} if eval_stats is not None else {}
                    if args.wandb:
                        wandb_utils.log(eval_stats, step=global_step)
                    if args.swanlab:
                        swanlab_utils.log(eval_stats, step=global_step)
                logger.info("Evaluation done.")
                if rank == 0:
                    eval_npz_dir = os.path.join(experiment_dir, "eval_npzs")
                    if os.path.exists(eval_npz_dir):
                        shutil.rmtree(eval_npz_dir)
                        logger.info(f"Deleted validation artifacts at {eval_npz_dir}")
            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                logger.info(f"Max steps {args.max_steps} reached. Stopping training.")
                break
        else:
            if rank == 0 and num_batches > 0:
                avg_recon = (epoch_metrics["recon"] / num_batches).item()
                avg_lpips = (epoch_metrics["lpips"] / num_batches).item()
                avg_psnr = (epoch_metrics["psnr"] / num_batches).item()
                avg_ssim = (epoch_metrics["ssim"] / num_batches).item()
                avg_gan = (epoch_metrics["gan"] / num_batches).item()
                avg_total = (epoch_metrics["total"] / num_batches).item()
                epoch_stats = {
                    "epoch/loss_total": avg_total,
                    "epoch/loss_recon": avg_recon,
                    "epoch/loss_lpips": avg_lpips,
                    "epoch/loss_psnr": avg_psnr,
                    "epoch/loss_ssim": avg_ssim,
                    "epoch/loss_gan": avg_gan,
                }
                if use_all and "inter_recon" in epoch_metrics and num_batches > 0:
                    epoch_stats["epoch/loss_inter_recon"] = (epoch_metrics["inter_recon"] / num_batches).item()
                    epoch_stats["epoch/loss_inter_lpips"] = (epoch_metrics["inter_lpips"] / num_batches).item()
                    epoch_stats["epoch/loss_inter_psnr"] = (epoch_metrics["inter_psnr"] / num_batches).item()
                    epoch_stats["epoch/loss_inter_ssim"] = (epoch_metrics["inter_ssim"] / num_batches).item()
                    epoch_stats["epoch/loss_inter_gan"] = (epoch_metrics["inter_gan"] / num_batches).item()
                if disc_metrics:
                    epoch_stats.update(
                        {
                            "epoch/loss_disc": (epoch_metrics["disc_loss"] / num_batches).item(),
                            "epoch/disc_logits_real": disc_metrics["logits_real"].item(),
                            "epoch/disc_logits_fake": disc_metrics["logits_fake"].item(),
                            "epoch/disc_accuracy": (epoch_metrics["disc_accuracy"] / num_batches).item(),
                        }
                    )
                logger.info(
                    f"[Epoch {epoch}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
                )
                if args.wandb:
                    wandb_utils.log(epoch_stats, step=global_step)
                if args.swanlab:
                    swanlab_utils.log(epoch_stats, step=global_step)
            
            if getattr(args, "early_stop", False) and do_eval:
                if rank == 0:
                    logger.info(f"Early stop check: Evaluating at end of epoch {epoch}...")
                eval_mod = ema_model if not eval_model else get_base_model(ddp_model)
                eval_stats = evaluate_reconstruction_distributed(
                    eval_mod,
                    eval_dataset,
                    len(eval_dataset),
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    batch_size=batch_size,
                    metrics_to_compute=eval_metrics,
                    experiment_dir=experiment_dir,
                    global_step=global_step,
                    autocast_kwargs=autocast_kwargs,
                    reference_npz_path=reference_npz_path
                )
                
                stop_training = torch.tensor(0, dtype=torch.int32, device=device)
                if rank == 0 and eval_stats is not None and base_metrics is not None:
                    eval_stats_log = {f"eval_early_stop/{k}": v for k, v in eval_stats.items()}
                    if args.wandb:
                        wandb_utils.log(eval_stats_log, step=global_step)
                    if args.swanlab:
                        swanlab_utils.log(eval_stats_log, step=global_step)
                        
                    curr_rfid = eval_stats.get("rfid", float('inf'))
                    curr_psnr = eval_stats.get("psnr", 0.0)
                    curr_ssim = eval_stats.get("ssim", 0.0)
                    base_rfid = base_metrics.get("rfid", 0.0)
                    base_psnr = base_metrics.get("psnr", 0.0)
                    base_ssim = base_metrics.get("ssim", 0.0)
                    
                    if curr_rfid < base_rfid and curr_psnr > base_psnr + 2.0 and curr_ssim > base_ssim + 0.02:
                        logger.info(f"Early Stop condition met! rFID: {curr_rfid:.4f} < {base_rfid:.4f}, PSNR: {curr_psnr:.4f} > {base_psnr:.4f} + 2.0, SSIM: {curr_ssim:.4f} > {base_ssim:.4f} + 0.02")
                        stop_training.fill_(1)
                    else:
                        logger.info(f"Early Stop condition not met. Cur rFID: {curr_rfid:.4f}, PSNR: {curr_psnr:.4f}, SSIM: {curr_ssim:.4f}")
                        logger.info(f"Target: rFID < {base_rfid:.4f}, PSNR > {base_psnr+2.0:.4f}, SSIM > {base_ssim+0.02:.4f}")
                        
                    eval_npz_dir = os.path.join(experiment_dir, "eval_npzs")
                    if os.path.exists(eval_npz_dir):
                        shutil.rmtree(eval_npz_dir)
                
                if world_size > 1:
                    dist.all_reduce(stop_training, op=dist.ReduceOp.MAX)
                
                if stop_training.item() == 1:
                    break
            
            continue
        break
    # save the final ckpt
    ckpt_path = f"{checkpoint_dir}/ep-last.pt"
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
            discriminator,
            disc_optimizer,
            disc_scheduler,
        )
        # Only keep the latest two checkpoints
        checkpoints = sorted(glob(f"{checkpoint_dir}/ep-*.pt"))
        if len(checkpoints) > 2:
            for old_ckpt in checkpoints[:-2]:
                os.remove(old_ckpt)
    
    if world_size > 1:
        dist.barrier()

    # Final Test evaluation
    if do_eval and args.dataset in ["brats", "mri_ct"]:
        logger.info("Starting final test evaluation...")
        from brats_dataset import BraTS2023Dataset, get_transforms
        test_dataset = BraTS2023Dataset(
            json_path=args.data_path,
            phase='test', 
            transform=get_transforms('test'),
            modality_index=args.modality
        )
        logger.info(f"{args.dataset.upper()} Test dataset loaded from {args.data_path}, modality {args.modality}, containing {len(test_dataset)} images.")
        
        test_dir = os.path.join(experiment_dir, "test")
        if rank == 0:
            os.makedirs(test_dir, exist_ok=True)
        if world_size > 1:
            dist.barrier()
            
        test_metrics = evaluate_reconstruction_distributed(
            ema_model,
            test_dataset,
            len(test_dataset),
            rank=rank,
            world_size=world_size,
            device=device,
            batch_size=batch_size,
            metrics_to_compute=("psnr", "ssim", "lpips", "rfid", "is", "ifid"),
            experiment_dir=test_dir,
            global_step=global_step,
            autocast_kwargs=autocast_kwargs,
            reference_npz_path=None,
            train_dataset=loader_dataset,
        )
        
        if rank == 0 and test_metrics is not None:
            import json
            with open(os.path.join(test_dir, "test_metrics.json"), "w") as f:
                json.dump(test_metrics, f, indent=4)
            logger.info(f"Final test metrics saved to {os.path.join(test_dir, 'test_metrics.json')}")

    if world_size > 1:
        dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
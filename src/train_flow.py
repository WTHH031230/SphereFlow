# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A training script for Flow Matching (Transport) with all modalities using DDT.
The model is conditioned ONLY on the modality combination ID (target + known mask).
"""
import argparse
import logging
import math
import os
import random
from collections import defaultdict, OrderedDict
from time import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from copy import deepcopy
from glob import glob

import torch
# Enable TF32 matmuls
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional.image import structural_similarity_index_measure
from omegaconf import OmegaConf

##### model imports
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler
from stage2.transport.utils import mean_flat

##### general utils
from utils import wandb_utils, swanlab_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import setup_distributed, cleanup_distributed

##### Eval utils
from eval import evaluate_generation_distributed

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)

def get_base_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP so encode/decode etc. avoid .module issues on multi-GPU."""
    if hasattr(model, "module"):
        return model.module
    return model

def weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    values = values.float().flatten()
    weights = weights.to(values).float().flatten()
    return (values * weights).sum() / weights.sum().clamp_min(eps)

def endpoint_from_velocity(xt: torch.Tensor, t: torch.Tensor, v_pred: torch.Tensor) -> torch.Tensor:
    t_view = t.view(t.size(0), *([1] * (xt.ndim - 1)))
    return xt - t_view * v_pred

def orthogonal_residual_loss(
    z_known: torch.Tensor,
    z_target: torch.Tensor,
    z_pred: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_delta = (z_target - z_known).float().flatten(1)
    pred_delta = (z_pred - z_known).float().flatten(1)
    target_norm_sq = target_delta.square().sum(dim=1).clamp_min(eps)
    alpha = (pred_delta * target_delta).sum(dim=1) / target_norm_sq
    residual = pred_delta - alpha[:, None] * target_delta
    residual_norm_sq = residual.square().sum(dim=1)
    loss_per_sample = residual_norm_sq / target_norm_sq
    residual_ratio = torch.sqrt(residual_norm_sq / target_norm_sq)
    return weighted_mean(loss_per_sample, weights), residual_ratio.detach().mean(), alpha.detach().mean()

def decoded_endpoint_image_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = pred.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    l1_per = (pred - target).abs().flatten(1).mean(dim=1)
    mse_per = (pred - target).square().flatten(1).mean(dim=1)
    psnr_per = -10.0 * torch.log10(mse_per.clamp_min(1e-8))
    ssim_per = 1.0 - structural_similarity_index_measure(
        pred,
        target,
        data_range=1.0,
        reduction="none",
    )
    if ssim_per.ndim == 0:
        ssim_loss = ssim_per
    else:
        ssim_loss = weighted_mean(ssim_per.flatten(), weights)
    return weighted_mean(l1_per, weights), ssim_loss, weighted_mean(psnr_per, weights)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 flow model on all RAE latents.")
    parser.add_argument("--config", type=str, required=True, help="YAML config.")
    parser.add_argument("--data-path", type=Path, required=True, help="BraTS JSON path.")
    parser.add_argument("--dataset", type=str, default="brats", choices=["brats", "mri_ct"], help="Dataset type.")
    parser.add_argument("--rae-path", type=str, required=True, help="Path to trained RAE4Dec checkpoint.")
    parser.add_argument("--results-dir", type=str, default="ckpts")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--global-seed", type=int, default=None)
    parser.add_argument("--force-known", type=int, default=None, help="Force known modality index.")
    parser.add_argument("--force-target", type=int, default=None, help="Force target modality index.")
    parser.add_argument("--test-only", action="store_true", help="Skip training and only run final test evaluation on the resumed checkpoint.")
    parser.add_argument("--no-save", action="store_true", help="Skip worktree and checkpoint saving for quick probes.")
    parser.add_argument("--latest-checkpoint-interval", type=int, default=0,
                        help="If > 0, update checkpoints/ep-last.pt every N completed epochs.")
    parser.add_argument("--save-best-rfid", action="store_true",
                        help="Save a checkpoint whenever periodic EMA eval reaches a new best mean rFID.")
    parser.add_argument("--best-rfid-dir", type=str, default="best_checkpoints",
                        help="Subdirectory under the experiment dir used for the best-rFID checkpoint.")
    parser.add_argument("--flow-weight", type=float, default=1.0,
                        help="Weight for the standard flow matching loss.")
    parser.add_argument("--image-l1-weight", type=float, default=0.0,
                        help="Weight for decoded endpoint L1 image loss.")
    parser.add_argument("--image-ssim-weight", type=float, default=0.0,
                        help="Weight for decoded endpoint SSIM image loss.")
    parser.add_argument("--orth-weight", type=float, default=0.0,
                        help="Weight for endpoint orthogonal residual loss.")
    parser.add_argument("--image-min-t", type=float, default=0.0,
                        help="Minimum sampled t for decoded endpoint image losses.")
    parser.add_argument("--orth-min-t", type=float, default=0.0,
                        help="Minimum sampled t for orthogonal residual loss.")
    parser.add_argument("--no-endpoint-t-weight", action="store_true",
                        help="Disable t weighting for endpoint image/orthogonal losses.")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")
    
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

    # Clear predefined ckpt so instantiate_from_config doesn't auto-load, we load manually later
    if "ckpt" in rae_config:
        del rae_config["ckpt"]
    if "ckpt" in full_cfg.get("stage_1", {}):
        del full_cfg["stage_1"]["ckpt"]

    def to_dict(cfg): return OmegaConf.to_container(cfg, resolve=True) if cfg else {}
    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    training_cfg = to_dict(training_config)

    # -------------------------------------------------------------------------
    # Training Parameters Alignment
    # -------------------------------------------------------------------------
    _num_classes = model_config.get("params", {}).get("num_classes", None)
    use_classes = _num_classes is not None and int(_num_classes) > 0
    num_classes = int(_num_classes) if use_classes else 0
    if rank == 0:
        logging.info(f"use_classes={use_classes}, num_classes={num_classes}")
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    num_epochs = int(training_cfg.get("epochs", 400))
    endpoint_image_enabled = args.image_l1_weight > 0 or args.image_ssim_weight > 0
    endpoint_orth_enabled = args.orth_weight > 0
    endpoint_aux_enabled = endpoint_image_enabled or endpoint_orth_enabled
    
    global_batch_size = training_cfg.get("global_batch_size", None)
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % (world_size * grad_accum_steps) == 0, f"global_batch_size {global_batch_size} must be divisible by world_size * grad_accum_steps ({world_size * grad_accum_steps})"
        micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    else:
        micro_batch_size = int(training_cfg.get("batch_size", 16))
        global_batch_size = micro_batch_size * world_size * grad_accum_steps

    num_workers = int(training_cfg.get("num_workers", 8))
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 2500)) 
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4))
    
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Precision setup via get_autocast_scaler
    scaler, autocast_kwargs = get_autocast_scaler(args)

    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    
    # -------------------------------------------------------------------------
    # Model Init
    # -------------------------------------------------------------------------
    
    class RAE4Dec(torch.nn.Module):
        def __init__(self, base_rae: RAE, num_modalities=4):
            super().__init__()
            self.encoder = base_rae.encoder
            self.register_buffer('encoder_mean', base_rae.encoder_mean.clone())
            self.register_buffer('encoder_std', base_rae.encoder_std.clone())
            self.encoder_input_size = base_rae.encoder_input_size
            self.encode = base_rae.encode

            self.reshape_to_2d = getattr(base_rae, "reshape_to_2d", True)
            self.do_normalization = getattr(base_rae, "do_normalization", False)
            self.num_modalities = num_modalities

            self.decoders = torch.nn.ModuleList([deepcopy(base_rae.decoder) for _ in range(num_modalities)])

        def decode(self, z, m):
            pass # overwritten below

    num_mods = 2 if args.dataset == "mri_ct" else 4
    base_rae = instantiate_from_config(rae_config).to(device)
    rae = RAE4Dec(base_rae, num_modalities=num_mods).to(device)
    
    # Define dynamic decode logic to match train_shared_encoder.py
    def decode_with_modality(self_rae, z, m=None):
        if m is None:
            m = args.modality
        decoder = self_rae.decoders[m]
        
        if self_rae.do_normalization:
             z = z * torch.sqrt(base_rae.latent_var + base_rae.eps) + base_rae.latent_mean
            
        if self_rae.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
        output = decoder(z, drop_cls_token=False).logits
        x_rec = decoder.unpatchify(output)
        x_rec = x_rec * self_rae.encoder_std + self_rae.encoder_mean
        return x_rec

    import types
    rae.decode = types.MethodType(decode_with_modality, rae)

    logger.info(f"Loading Stage-2 RAE weights from {args.rae_path}")
    sd = torch.load(args.rae_path, map_location="cpu")
    rae_state_dict = sd.get("model", sd.get("ema", sd))
    for key in ("decoder_pred.weight", "decoder_pred.bias"):
        rae_state_dict.pop(key, None)
    rae.load_state_dict(rae_state_dict, strict=True)
    del sd, rae_state_dict # free checkpoint memory
    torch.cuda.empty_cache()
    rae.eval().requires_grad_(False)

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device) 
    
    if args.compile:
        try:
            model.forward = torch.compile(model.forward)
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")
        
    ema_model = deepcopy(model).to(device).eval().requires_grad_(False)
    model.requires_grad_(True)
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    model = ddp_model.module
    ddp_model.train()
    
    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")

    # -------------------------------------------------------------------------
    # Optimizer & Data & Transport
    # -------------------------------------------------------------------------
    
    optimizer, optim_msg = build_optimizer([p for p in model.parameters() if p.requires_grad], training_cfg)

    # Data Init
    import sys
    if os.getcwd() not in sys.path: sys.path.insert(0, os.getcwd())
    from brats_dataset import BraTS2023AllDataset, get_transforms
    train_dataset = BraTS2023AllDataset(str(args.data_path), 'train', get_transforms('train'), num_modalities=num_mods)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        train_dataset, batch_size=micro_batch_size, 
        sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    # Evaluation Dataset Init
    do_eval = eval_config is not None
    eval_metrics = ["rfid", "psnr", "ssim", "is"]  # defaults
    reference_npz_path = None
    if do_eval:
        eval_interval = int(eval_config.get("eval_interval", 5000))
        eval_model = eval_config.get("eval_model", False)
        eval_metrics = list(eval_config.get("metrics", ["rfid", "psnr", "ssim", "is"]))
        eval_data_json = str(args.data_path)
        reference_npz_path = eval_config.get("reference_npz_path", None)
        eval_dataset = BraTS2023AllDataset(eval_data_json, 'test', get_transforms('val'), num_modalities=num_mods)
        logger.info(f"Evaluation dataset loaded from {eval_data_json}, size: {len(eval_dataset)}")
        logger.info(f"Evaluation metrics: {eval_metrics}")

    # Scheduler Init
    loader_batches = len(loader)
    steps_per_epoch = loader_batches // grad_accum_steps if grad_accum_steps > 1 else loader_batches
    scheduler = None
    sched_msg = ""
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)

    # Transport Init
    transport = create_transport(**to_dict(transport_config.get("params", {})), time_dist_shift=time_dist_shift)
    transport_sampler = Sampler(transport)
    
    sampler_mode = sampler_cfg.get("mode", "ODE").upper() if sampler_cfg else "ODE"
    sampler_params = dict(sampler_cfg.get("params", {})) if sampler_cfg else {}
    
    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    # -------------------------------------------------------------------------
    # Eval Helpers — EvalWrapper returns (x_target, z_known_latent, cond_id)
    # which maps exactly to batch[0]=x_a, batch[1]=x_b (latent), batch[2]=cond
    # evaluate_generation_distributed will use x_b directly as latent if
    # the shape check passes, or re-encode otherwise.
    # z_known is pre-encoded in __getitem__.
    # -------------------------------------------------------------------------
    class EvalWrapper(torch.utils.data.Dataset):
        """Wraps BraTS2023AllDataset for evaluation with a fixed (target, known) split.
        
        Returns (x_target_image, z_known_latent, cond_id_scalar) so that
        evaluate_generation_distributed sees:
            batch[0] = x_a = target image (for reference)
            batch[1] = x_b = pre-encoded known-modality latent
            batch[2] = cond_id (unused by eval fn directly, passed via model_kwargs)
        """
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
            # Return the target and the stacked known images; they are encoded in the eval loop
            x_target = imgs[self.target]
            x_known_list = [imgs[i] for i in self.known]
            # Stack known modality images along a new dim so we can encode them later
            x_known_stack = torch.stack(x_known_list, dim=0)  # [num_known, C, H, W]
            return x_target, x_known_stack, torch.tensor(self.cond_id_val, dtype=torch.long)

    class VAEEvalWrapper:
        """Wraps RAE4Dec so evaluate_generation_distributed can call
        rae.encode(x) and rae.decode(z) with a single argument."""
        def __init__(self, full_rae, target_idx):
            self.full_rae = full_rae
            self.encoder = full_rae.encoder
            self.target_idx = target_idx
        def encode(self, x):
            return self.full_rae.encode(x)
        def decode(self, z):
            return self.full_rae.decode(z, self.target_idx)

    # -------------------------------------------------------------------------
    # Resuming
    # -------------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_rfid = float("inf")
    best_rfid_step = None
    best_rfid_dir = os.path.join(experiment_dir, args.best_rfid_dir)
    best_rfid_ckpt_path = os.path.join(best_rfid_dir, "ep-best-rfid.pt")
    best_rfid_metrics_path = os.path.join(best_rfid_dir, "best_rfid_metrics.json")
    if rank == 0 and args.save_best_rfid and os.path.exists(best_rfid_metrics_path):
        try:
            import json as json_module
            with open(best_rfid_metrics_path) as f:
                best_meta = json_module.load(f)
            best_rfid = float(best_meta.get("best_rfid", best_rfid))
            best_rfid_step = best_meta.get("step", None)
            logger.info(f"Loaded previous best rFID={best_rfid:.6f} at step={best_rfid_step}.")
        except Exception as exc:
            logger.warning(f"Could not load previous best-rFID metadata: {exc}")
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
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # Starting from fresh, save worktree and configs
        if rank == 0 and not args.no_save:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")
        elif rank == 0:
            logger.info("--no-save enabled: skipping worktree snapshot.")

    # Log parameters
    if rank == 0:
        logger.info(f"Stage-2 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
        logger.info(optim_msg)
        if sched_msg: logger.info(sched_msg)
        logger.info(f"Training for {num_epochs} epochs, micro batch size {micro_batch_size}, global batch size {global_batch_size}.")
        logger.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")
        if endpoint_aux_enabled:
            logger.info(
                "Endpoint auxiliary losses enabled: flow_weight=%.4g, image_l1=%.4g, image_ssim=%.4g, "
                "orth=%.4g, image_min_t=%.4g, orth_min_t=%.4g, t_weight=%s",
                args.flow_weight,
                args.image_l1_weight,
                args.image_ssim_weight,
                args.orth_weight,
                args.image_min_t,
                args.orth_min_t,
                not args.no_endpoint_t_weight,
            )

    # -------------------------------------------------------------------------
    # Custom evaluation function for all-modality flow
    # (replaces evaluate_generation_distributed for this special dataset format)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_flow_all(
        model_fn,
        sample_fn,
        rae_model,
        eval_ds,
        target_idx,
        known_idxs,
        cond_id_val,
        latent_sz,
        micro_bs,
        ac_kwargs,
        exp_dir,
        g_step,
        metrics_list,
        ref_npz_path=None,
    ):
        """Evaluate flow model on all-modality dataset.
        
        This replaces evaluate_generation_distributed because the standard
        eval function expects (x_a, x_b) format, but our EvalWrapper returns
        (x_target, x_known_stack, cond_id) with pre-stacked known images.
        """
        import numpy as np
        from torch.utils.data import Subset
        from tqdm import tqdm

        temp_dir = os.path.join(exp_dir, "eval_npzs")
        if rank == 0:
            print(f"\n[Eval] Starting flow-all evaluation at step {g_step}")
            os.makedirs(temp_dir, exist_ok=True)
        if world_size > 1:
            dist.barrier()

        N = len(eval_ds)
        chunk = N // world_size
        if rank < world_size - 1:
            start = rank * chunk
            end = (rank + 1) * chunk
        else:
            start = rank * chunk
            end = N

        rank_indices = list(range(start, end))
        subset = Subset(eval_ds, rank_indices)
        eval_loader = DataLoader(
            subset, batch_size=micro_bs, shuffle=False,
            num_workers=0, pin_memory=False, drop_last=False
        )

        generations = []
        references = []
        iterator = tqdm(eval_loader, desc=f"[Rank {rank}] Sampling", file=sys.stdout) if rank == 0 else eval_loader

        with torch.inference_mode():
            for batch in iterator:
                x_target = batch[0].to(device, non_blocking=True)
                x_known_stack = batch[1].to(device, non_blocking=True)  # [B, num_known, C, H, W]
                cond_ids = batch[2].to(device, non_blocking=True)  # [B]

                # Encode each known modality and average
                # x_known_stack shape: [B, num_known, C, H, W]
                B_eval, num_known = x_known_stack.shape[0], x_known_stack.shape[1]
                z_known_list = []
                for ki in range(num_known):
                    z_ki = rae_model.encode(x_known_stack[:, ki])
                    z_known_list.append(z_ki)
                z_known = torch.stack(z_known_list, dim=0).mean(dim=0)

                model_kwargs = {}
                if use_classes:
                    model_kwargs["y"] = cond_ids

                with autocast(**ac_kwargs):
                    samples = sample_fn(z_known, model_fn, **model_kwargs)[-1]
                    decoded = rae_model.decode(samples, target_idx).clamp(0, 1)

                gen_np = decoded.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                ref_np = x_target.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                for img in gen_np:
                    generations.append(img)
                for img in ref_np:
                    references.append(img)

        generations = np.stack(generations)
        references = np.stack(references)
        shard_path = os.path.join(temp_dir, f"gen_{g_step:07d}_{rank:02d}.npz")
        np.savez(shard_path, arr_0=generations, ref=references)

        if rank == 0:
            print(f"[Rank {rank}] Saved {len(generations)} generation to {shard_path}")

        if world_size > 1:
            dist.barrier()

        metrics = None
        if rank == 0:
            from eval import compute_generation_metrics
            all_gens = []
            all_refs = []
            for r in range(world_size):
                shard_file = os.path.join(temp_dir, f"gen_{g_step:07d}_{r:02d}.npz")
                shard_data = np.load(shard_file)
                all_gens.append(shard_data["arr_0"])
                all_refs.append(shard_data["ref"])

            combined_gens = np.concatenate(all_gens, axis=0)[:N]
            combined_refs = np.concatenate(all_refs, axis=0)[:N]
            print(f"[Eval] Combined generation NPZ shape: {combined_gens.shape}")
            print(f"[Eval] Combined reference NPZ shape: {combined_refs.shape}")

            print(f"[Eval] Computing metrics ({', '.join(metrics_list)})...")
            metrics = compute_generation_metrics(
                combined_refs, combined_gens, device,
                batch_size=128, metrics_to_compute=metrics_list
            )
            print(f"[Eval] Step {g_step} Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")

            for r in range(world_size):
                shard_file = os.path.join(temp_dir, f"gen_{g_step:07d}_{r:02d}.npz")
                if os.path.exists(shard_file):
                    os.remove(shard_file)

        if world_size > 1:
            dist.barrier()
        return metrics

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    
    import sys
    log_steps = 0
    running_loss = 0.0
    running_aux: Dict[str, float] = defaultdict(float)
    running_aux_count = 0
    start_time = time()
    
    dist.barrier()
    for epoch in range(start_epoch, num_epochs if not args.test_only else start_epoch):
        ddp_model.train()
        sampler.set_epoch(epoch)
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        optimizer.zero_grad()
        
        # Checkpoint at epoch start
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0 and rank == 0 and not args.no_save:
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
            )
            # Only keep the latest two epoch checkpoints
            checkpoints = sorted(glob(f"{checkpoint_dir}/ep-[0-9]*.pt"))
            if len(checkpoints) > 2:
                for old_ckpt in checkpoints[:-2]:
                    os.remove(old_ckpt)
        
        for step, batch in enumerate(loader):
            all_imgs = [img.to(device, non_blocking=True) for img in batch]
            with torch.no_grad(): 
                z_list = [rae.encode(img) for img in all_imgs]
            
            # Randomly select target modality and known subset for training
            if args.force_known is not None and args.force_target is not None:
                m_target = args.force_target
                m_known = [args.force_known]
            else:
                m_target = random.randint(0, num_mods - 1)
                m_known = random.sample([i for i in range(num_mods) if i != m_target], random.randint(1, num_mods - 1))
            
            z_aux = torch.stack([z_list[i] for i in m_known], dim=0).mean(dim=0)
            z_main = z_list[m_target]
            
            # Condition ID = target*16 + mask (only for conditional models)
            model_kwargs = {"x0": z_aux}
            if use_classes:
                cond_id = torch.full((z_main.size(0),), m_target * 16 + sum([2**i for i in m_known]), dtype=torch.long, device=device)
                model_kwargs["y"] = cond_id
            
            aux_log_terms: Dict[str, torch.Tensor] = {}
            with autocast(**autocast_kwargs):
                if endpoint_aux_enabled:
                    t, x0, x1 = transport.sample(z_main, x0=z_aux)
                    t, xt, ut = transport.path_sampler.plan(t, x0, x1)
                    endpoint_kwargs = {}
                    if use_classes:
                        endpoint_kwargs["y"] = cond_id
                    v_pred = ddp_model(xt, t, **endpoint_kwargs)
                    flow_loss = mean_flat((v_pred - ut) ** 2).mean()
                    z_pred = endpoint_from_velocity(xt, t, v_pred)

                    base_weights = (
                        torch.ones_like(t, dtype=torch.float32)
                        if args.no_endpoint_t_weight
                        else t.detach().float()
                    )
                    image_weights = base_weights * (t.detach().float() >= args.image_min_t).float()
                    orth_weights = base_weights * (t.detach().float() >= args.orth_min_t).float()

                    image_l1 = torch.zeros((), device=device)
                    image_ssim = torch.zeros((), device=device)
                    image_psnr = torch.zeros((), device=device)
                    if endpoint_image_enabled:
                        decoded = rae.decode(z_pred, m_target)
                        image_l1, image_ssim, image_psnr = decoded_endpoint_image_losses(
                            decoded,
                            all_imgs[m_target],
                            image_weights,
                        )

                    orth_loss = torch.zeros((), device=device)
                    orth_ratio = torch.zeros((), device=device)
                    orth_alpha = torch.zeros((), device=device)
                    if endpoint_orth_enabled:
                        orth_loss, orth_ratio, orth_alpha = orthogonal_residual_loss(
                            z_aux,
                            z_main,
                            z_pred,
                            orth_weights,
                        )

                    loss = (
                        args.flow_weight * flow_loss
                        + args.image_l1_weight * image_l1
                        + args.image_ssim_weight * image_ssim
                        + args.orth_weight * orth_loss
                    )
                    aux_log_terms = {
                        "flow": flow_loss.detach(),
                        "image_l1": image_l1.detach(),
                        "image_ssim_loss": image_ssim.detach(),
                        "image_psnr": image_psnr.detach(),
                        "orth": orth_loss.detach(),
                        "orth_ratio": orth_ratio.detach(),
                        "orth_alpha": orth_alpha.detach(),
                        "t_mean": t.detach().float().mean(),
                        "image_active": (image_weights > 0).float().mean(),
                    }
                else:
                    loss_terms = transport.training_losses(ddp_model, z_main, model_kwargs)
                    loss = loss_terms["loss"].mean()
            
            if scaler:
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                (loss / grad_accum_steps).backward()
            
            # Optimizer step at grad_accum boundary
            if (step + 1) % grad_accum_steps == 0:
                if clip_grad:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                if scheduler: scheduler.step()
                update_ema(ema_model, model, decay=ema_decay)
                optimizer.zero_grad(set_to_none=True)
                
            running_loss += loss.item()
            epoch_metrics['loss'] += loss.detach()
            if aux_log_terms:
                for key, value in aux_log_terms.items():
                    running_aux[key] += float(value.item())
                    epoch_metrics[key] += value.detach()
                running_aux_count += 1
            
            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                avg_loss = running_loss / (log_interval if global_step > 0 else 1)
                stats = {"train/loss": avg_loss, "train/lr": optimizer.param_groups[0]["lr"]}
                if running_aux_count > 0:
                    stats.update({
                        f"train/{key}": value / running_aux_count
                        for key, value in sorted(running_aux.items())
                    })
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb: wandb_utils.log(stats, step=global_step)
                if args.swanlab: swanlab_utils.log(stats, step=global_step)
                running_loss = 0.0
                running_aux.clear()
                running_aux_count = 0

            # Sampling
            if global_step % sample_every == 0 and rank == 0:
                ddp_model.eval()
                with torch.no_grad():
                    n_vis = min(4, all_imgs[0].size(0))
                    rows = []
                    
                    if args.force_known is not None and args.force_target is not None:
                        # Focused Mode: Input -> GT -> Gen
                        m_in, m_out = args.force_known, args.force_target
                        
                        x_in = all_imgs[m_in][:n_vis]
                        x_gt = all_imgs[m_out][:n_vis]
                        
                        # Generate from Flow
                        z_in = rae.encode(x_in)
                        sample_kwargs = {}
                        if use_classes:
                            c_id = torch.full((n_vis,), m_out * 16 + (2**m_in), dtype=torch.long, device=device)
                            sample_kwargs["y"] = c_id
                        with autocast(**autocast_kwargs):
                            samples = eval_sampler(z_in, ema_model.forward, **sample_kwargs)[-1]
                            gen_out = rae.decode(samples, m_out).clamp(0, 1)
                        
                        rows.append(x_in)    # Row 1: MRI GT
                        rows.append(x_gt)    # Row 2: CT GT
                        rows.append(gen_out) # Row 3: CT Gen
                    else:
                        # General mode: show actual flow generations for each target.
                        for m_out in range(num_mods):
                            k_idxs = [idx for idx in range(num_mods) if idx != m_out]
                            if not k_idxs:
                                continue
                            z_in = torch.stack([z_list[idx][:n_vis] for idx in k_idxs], dim=0).mean(dim=0)
                            x_in = all_imgs[k_idxs[0]][:n_vis]
                            x_gt = all_imgs[m_out][:n_vis]
                            sample_kwargs = {}
                            if use_classes:
                                mask = sum(2 ** idx for idx in k_idxs)
                                c_id = torch.full((n_vis,), m_out * 16 + mask, dtype=torch.long, device=device)
                                sample_kwargs["y"] = c_id
                            with autocast(**autocast_kwargs):
                                samples = eval_sampler(z_in, ema_model.forward, **sample_kwargs)[-1]
                                gen_out = rae.decode(samples, m_out).clamp(0, 1)
                            rows.append(x_in)
                            rows.append(x_gt)
                            rows.append(gen_out)
                    
                    out = make_grid(torch.cat(rows, dim=0).cpu().float(), nrow=n_vis)
                    if args.wandb: wandb_utils.log_image(out, step=global_step)
                    if args.swanlab: swanlab_utils.log_image(out, step=global_step)
                ddp_model.train()

            # Periodic Evaluation
            if do_eval and (eval_interval > 0 and global_step % eval_interval == 0):
                logger.info(f"Starting evaluation at step {global_step}...")
                ddp_model.eval()
                ema_eval_rfid_values = []
                ema_eval_metrics_by_target = {}
                
                # If force-target/known are set, only evaluate that specific path
                t_eval_indices = [args.force_target] if (args.force_target is not None and args.force_known is not None) else range(num_mods)
                
                for t_idx in t_eval_indices:
                    if args.force_known is not None and args.force_target is not None:
                        k_idxs = [args.force_known]
                    else:
                        k_idxs = [i for i in range(num_mods) if i != t_idx]
                    
                    if not k_idxs: continue
                    
                    mask_eval = sum([2**i for i in k_idxs])
                    eval_cond_id = t_idx * 16 + mask_eval
                    
                    eval_models = [(ema_model.forward, "ema")]
                    if eval_model: eval_models.append((model.forward, "model"))
                    
                    eval_wrap = EvalWrapper(eval_dataset, t_idx, k_idxs, eval_cond_id, rae, device)
                    
                    for fn, mod_name in eval_models:
                        eval_stats = evaluate_flow_all(
                            fn, eval_sampler, rae,
                            eval_wrap, t_idx, k_idxs, eval_cond_id,
                            latent_size, micro_batch_size, autocast_kwargs,
                            experiment_dir, global_step, eval_metrics,
                            ref_npz_path=reference_npz_path,
                        )
                        if rank == 0 and eval_stats:
                            if mod_name == "ema" and "rfid" in eval_stats:
                                ema_eval_rfid_values.append(float(eval_stats["rfid"]))
                                ema_eval_metrics_by_target[f"target_{t_idx}"] = {
                                    k: float(v) for k, v in eval_stats.items()
                                }
                            eval_stats = {f"eval_{mod_name}_mod{t_idx}/{k}": v for k, v in eval_stats.items()}
                            if args.wandb: wandb_utils.log(eval_stats, step=global_step)
                            if args.swanlab: swanlab_utils.log(eval_stats, step=global_step)

                if rank == 0 and args.save_best_rfid and ema_eval_rfid_values:
                    mean_rfid = sum(ema_eval_rfid_values) / len(ema_eval_rfid_values)
                    logger.info(f"Periodic EMA eval mean rFID at step {global_step}: {mean_rfid:.6f}")
                    if mean_rfid < best_rfid:
                        best_rfid = mean_rfid
                        best_rfid_step = global_step
                        logger.info(
                            f"New best EMA mean rFID {best_rfid:.6f} at step {global_step}; "
                            f"saving {best_rfid_ckpt_path}"
                        )
                        save_checkpoint(
                            best_rfid_ckpt_path,
                            global_step,
                            epoch,
                            ddp_model,
                            ema_model,
                            optimizer,
                            scheduler,
                        )
                        import json as json_module
                        os.makedirs(best_rfid_dir, exist_ok=True)
                        with open(best_rfid_metrics_path, "w") as f:
                            json_module.dump(
                                {
                                    "best_rfid": best_rfid,
                                    "step": global_step,
                                    "epoch": epoch,
                                    "targets": ema_eval_metrics_by_target,
                                },
                                f,
                                indent=4,
                            )
                
                ddp_model.train()
                logger.info("Evaluation done.")

            global_step += 1
            num_batches += 1

        # Epoch-level logging
        if rank == 0 and num_batches > 0:
            avg_loss = epoch_metrics['loss'].item() / num_batches
            epoch_stats = {"epoch/loss": avg_loss}
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if args.wandb: wandb_utils.log(epoch_stats, step=global_step)
            if args.swanlab: swanlab_utils.log(epoch_stats, step=global_step)

        if (
            args.latest_checkpoint_interval > 0
            and (epoch + 1) % args.latest_checkpoint_interval == 0
            and rank == 0
            and not args.no_save
        ):
            latest_ckpt_path = f"{checkpoint_dir}/ep-last.pt"
            logger.info(f"Updating latest checkpoint after epoch {epoch + 1}: {latest_ckpt_path}")
            save_checkpoint(
                latest_ckpt_path,
                global_step,
                epoch + 1,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )

    # Save final checkpoint
    if rank == 0 and not args.no_save:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt"
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
        )
        # Only keep the latest two checkpoints
        checkpoints = sorted(glob(f"{checkpoint_dir}/ep-*.pt"))
        if len(checkpoints) > 2:
            for old_ckpt in checkpoints[:-2]:
                os.remove(old_ckpt)
    elif rank == 0:
        logger.info("--no-save enabled: skipping final checkpoint.")

    if world_size > 1:
        dist.barrier()

    # -------------------------------------------------------------------------
    # Final Test
    # -------------------------------------------------------------------------
    if do_eval:
        logger.info(f"Starting final test evaluation across all {num_mods} modalities...")
        test_dataset = BraTS2023AllDataset(str(args.data_path), 'test', get_transforms('val'), num_modalities=num_mods)
        test_dir = os.path.join(experiment_dir, "test")
        if rank == 0: os.makedirs(test_dir, exist_ok=True)
        dist.barrier()
        
        import json as json_module
        all_test_metrics = {}
        
        # If force-target/known are set, only test that specific path
        t_test_indices = [args.force_target] if (args.force_target is not None and args.force_known is not None) else range(num_mods)
        
        for t_idx in t_test_indices:
            if args.force_known is not None and args.force_target is not None:
                k_idxs = [args.force_known]
            else:
                k_idxs = [i for i in range(num_mods) if i != t_idx]
            
            if not k_idxs: continue
            
            eval_cond_id = t_idx * 16 + sum([2**i for i in k_idxs])
            
            test_wrap = EvalWrapper(test_dataset, t_idx, k_idxs, eval_cond_id, rae, device)
            
            test_metrics = evaluate_flow_all(
                ema_model.forward, eval_sampler, rae,
                test_wrap, t_idx, k_idxs, eval_cond_id,
                latent_size, micro_batch_size, autocast_kwargs,
                test_dir, 0, eval_metrics,
                ref_npz_path=reference_npz_path,
            )
            if rank == 0 and test_metrics is not None:
                all_test_metrics[f"target_{t_idx}"] = test_metrics
            
        if rank == 0:
            with open(os.path.join(test_dir, "test_metrics.json"), "w") as f:
                json_module.dump(all_test_metrics, f, indent=4)
            logger.info(f"Final test metrics saved to {os.path.join(test_dir, 'test_metrics.json')}")

    if world_size > 1:
        dist.barrier()
    logger.info("Done!")
    cleanup_distributed()

if __name__ == "__main__": 
    main()

# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""SphereFlow Stage 1, step 3: shared-encoder fine-tuning.

The encoder (fully, or through LoRA adapters) is trained jointly with one decoder per modality.
For every modality m of a batch:
  L_rec    = L1 + LPIPS (+ PSNR / SSIM terms) of D_m(E(x_m))
  L_align  = cos_w * (1 - cos(E(x_m), E_0(x_m))) + mse_w * ||GAP(E(x_m)) - GAP(E_0(x_m))||^2
  L_expand = w * [L1 + LPIPS (+ PSNR / SSIM terms)] of D_m(w * E(x_m) + (1 - w) * mean_k E(x_k))
with E_0 a frozen copy of the initial encoder and k a random non-empty subset of the other
modalities. The encoder receives gradients from every term, including both ends of the
interpolation; decoder m only from its own terms. MRI <-> CT adds paired geometry losses on the
encoder (token anchor, cosine / norm of the pooled MRI - CT displacement).

Two memory modes compute the same update:
  default             all decoders stay on the GPU
  --offload-decoders  decoders, their AdamW state and EMA live in CPU memory and are moved to the
                      GPU one at a time
"""

from __future__ import annotations

import argparse
import math
import os
import random
from copy import deepcopy
from glob import glob
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.functional.image import structural_similarity_index_measure

from disc import LPIPS
from stage1 import RAE
from utils import swanlab_utils, wandb_utils
from utils.dist_utils import cleanup_distributed, setup_distributed
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import configure_experiment_dirs, find_resume_checkpoint, save_worktree
from utils.train_utils import get_autocast_scaler, parse_configs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SphereFlow step 3: shared-encoder fine-tuning.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--dataset", type=str, default="brats", choices=["brats", "mri_ct"])
    for m in range(4):
        p.add_argument(f"--stage1-path-{m}", type=str, default=None, help=f"Step-2 checkpoint of modality {m}.")
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--global-batch-size", type=int, default=None)
    p.add_argument("--global-seed", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many steps.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--swanlab", action="store_true")
    p.add_argument("--offload-decoders", action="store_true",
                   help="Keep decoders in CPU memory and train them on the GPU one at a time.")

    p.add_argument("--w-sampling-mode", choices=["uniform", "power"], default="uniform")
    p.add_argument("--w-sampling-exponent", type=float, default=2.0, help="power mode: w = 1 - (1 - u)^exponent.")

    p.add_argument("--cos-weight", type=float, default=0.1)
    p.add_argument("--mse-weight", type=float, default=0.1)
    p.add_argument("--psnr-weight", type=float, default=0.0)
    p.add_argument("--ssim-weight", type=float, default=0.0)
    p.add_argument("--lambda-token-anchor", type=float, default=0.0, help="MRI <-> CT: token cosine to the frozen encoder.")
    p.add_argument("--lambda-vel-cos", type=float, default=0.0, help="MRI <-> CT: cosine of the pooled displacement.")
    p.add_argument("--lambda-vnorm", type=float, default=0.0, help="MRI <-> CT: norm ratio of the pooled displacement.")

    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false",
                   help="Disable gradient checkpointing on the encoder.")
    p.add_argument("--encoder-lora-rank", type=int, default=0, help="If > 0, train LoRA adapters instead of the full encoder.")
    p.add_argument("--encoder-lora-alpha", type=float, default=16.0)
    p.add_argument("--encoder-lora-dropout", type=float, default=0.0)
    p.add_argument("--encoder-lora-targets", type=str, default="query,key,value,dense,fc1,fc2,qkv,proj",
                   help="Comma-separated substrings of encoder Linear names to wrap with LoRA; '*' for all.")
    p.add_argument("--encoder-lora-last-n", type=int, default=0, help="If > 0, apply LoRA only in the last N blocks.")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Encoder utilities (gradient checkpointing, LoRA)
# ---------------------------------------------------------------------------
def _find_transformer_blocks(encoder):
    """Best-effort lookup for ViT/DINO transformer blocks across wrappers."""
    for attr in ('blocks', 'layer', 'layers'):
        candidate = getattr(encoder, attr, None)
        if candidate is not None and isinstance(candidate, torch.nn.ModuleList):
            return candidate
    inner = getattr(encoder, 'encoder', None)
    if inner is not None:
        for attr in ('layer', 'layers', 'blocks'):
            candidate = getattr(inner, attr, None)
            if candidate is not None and isinstance(candidate, torch.nn.ModuleList):
                return candidate
        inner_encoder = getattr(inner, 'encoder', None)
        if inner_encoder is not None:
            for attr in ('layer', 'layers', 'blocks'):
                candidate = getattr(inner_encoder, attr, None)
                if candidate is not None and isinstance(candidate, torch.nn.ModuleList):
                    return candidate
    return None


def _enable_gradient_checkpointing(encoder) -> bool:
    if hasattr(encoder, 'gradient_checkpointing_enable'):
        encoder.gradient_checkpointing_enable()
        return True
    blocks = _find_transformer_blocks(encoder)
    if blocks is None:
        return False
    from torch.utils.checkpoint import checkpoint as torch_ckpt

    for block in blocks:
        def make_ckpt_forward(orig_fn):
            def ckpt_forward(*args, **kwargs):
                return torch_ckpt(orig_fn, *args, use_reentrant=False, **kwargs)
            return ckpt_forward
        block.forward = make_ckpt_forward(block.forward)
    return True


class LoRALinear(torch.nn.Module):
    """Small LoRA wrapper for an existing Linear layer."""

    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.lora_A = torch.nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scale
        return self.base(x) + lora_out


def _replace_linears_with_lora(module, prefix, patterns, rank, alpha, dropout) -> int:
    wrapped = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, torch.nn.Linear):
            if "*" in patterns or any(pattern in full_name for pattern in patterns):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                wrapped += 1
            continue
        wrapped += _replace_linears_with_lora(child, full_name, patterns, rank, alpha, dropout)
    return wrapped


def apply_encoder_lora(encoder, rank, alpha, dropout, targets, last_n) -> str:
    encoder.requires_grad_(False)
    patterns = [part.strip() for part in targets.split(",") if part.strip()]
    blocks = _find_transformer_blocks(encoder) if last_n > 0 else None
    if blocks:
        n = min(last_n, len(blocks))
        wrapped = sum(
            _replace_linears_with_lora(block, f"blocks.{idx}", patterns, rank, alpha, dropout)
            for idx, block in enumerate(blocks[-n:], start=len(blocks) - n)
        )
        scope = f"last {n}/{len(blocks)} blocks"
    else:
        wrapped = _replace_linears_with_lora(encoder, "", patterns, rank, alpha, dropout)
        scope = "all blocks"
    if wrapped == 0:
        raise RuntimeError(f"No encoder Linear layers matched LoRA targets {patterns}.")
    for module in encoder.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
    return f"LoRA on {wrapped} Linear layers ({scope}, rank={rank}, alpha={alpha})"


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------
class RAE4Dec(torch.nn.Module):
    """Shared encoder + one decoder per modality, in the checkpoint layout read by step 4 and evaluation."""

    def __init__(self, base: RAE, num_modalities: int):
        super().__init__()
        self.encoder = base.encoder
        self.register_buffer("encoder_mean", base.encoder_mean.clone())
        self.register_buffer("encoder_std", base.encoder_std.clone())
        self.decoders = torch.nn.ModuleList([deepcopy(base.decoder) for _ in range(num_modalities)])
        self.decoder_pred = self.decoders[0].decoder_pred


def decode(base: RAE, decoder: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    if base.do_normalization:
        z = z * torch.sqrt(base.latent_var + base.eps) + base.latent_mean
    if base.reshape_to_2d:
        b, c, h, w = z.shape
        z = z.view(b, c, h * w).transpose(1, 2)
    x = decoder.unpatchify(decoder(z, drop_cls_token=False).logits)
    return x * base.encoder_std + base.encoder_mean


def to_tokens(z: torch.Tensor) -> torch.Tensor:
    if z.dim() == 4:
        b, c, h, w = z.shape
        return z.view(b, c, h * w).transpose(1, 2)
    return z


@torch.no_grad()
def frozen_tokens(base: RAE, frozen_encoder: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    size = base.encoder_input_size
    x = F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False)
    x = (x - base.encoder_mean) / base.encoder_std
    return to_tokens(frozen_encoder(x))


@torch.no_grad()
def ema_update(ema_module: torch.nn.Module, module: torch.nn.Module, decay: float) -> None:
    for e, p in zip(ema_module.parameters(), module.parameters()):
        e.mul_(decay).add_(p.detach(), alpha=1 - decay)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def allreduce_mean(tensors: List[torch.Tensor], world_size: int, bucket_bytes: int = 256 << 20) -> None:
    if world_size == 1 or not tensors:
        return
    bucket, size = [], 0
    for t in tensors + [None]:
        if t is not None:
            bucket.append(t)
            size += t.numel() * t.element_size()
        if bucket and (t is None or size >= bucket_bytes):
            flat = torch._utils._flatten_dense_tensors(bucket)
            dist.all_reduce(flat)
            flat.div_(world_size)
            for dst, src in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
                dst.copy_(src)
            bucket, size = [], 0


def grads_of(module: torch.nn.Module) -> List[torch.Tensor]:
    return [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]


# ---------------------------------------------------------------------------
#  Losses
# ---------------------------------------------------------------------------
class Step3Losses:
    def __init__(self, args, base: RAE, frozen_encoder, lpips, perceptual_weight: float, noise_tau: float):
        self.args = args
        self.base = base
        self.frozen_encoder = frozen_encoder
        self.lpips = lpips
        self.perceptual_weight = perceptual_weight
        self.noise_tau = noise_tau

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.encode(x)

    def plan(self, batch_size: int, num_modalities: int, latent_shape, device) -> List[Dict]:
        """Draw every random quantity of a step up front, so both memory modes see the same ones."""
        plans = []
        for m in range(num_modalities):
            others = [k for k in range(num_modalities) if k != m]
            subset = random.sample(others, random.randint(1, num_modalities - 1))
            w = torch.rand(batch_size, device=device)
            if self.args.w_sampling_mode == "power":
                w = 1.0 - (1.0 - w).pow(self.args.w_sampling_exponent)
            noise = []
            for _ in range(2):
                sigma = self.noise_tau * torch.rand((batch_size, 1, 1, 1), device=device)
                noise.append(sigma * torch.randn((batch_size, *latent_shape), device=device))
            plans.append({"subset": subset, "w": w, "noise_rec": noise[0], "noise_exp": noise[1]})
        return plans

    def rec(self, decoder, z, x, noise, lpips_on: bool) -> torch.Tensor:
        x_rec = decode(self.base, decoder, z + noise)
        loss = (x_rec - x).abs().mean()
        if lpips_on:
            loss = loss + self.perceptual_weight * self.lpips(x * 2 - 1, x_rec * 2 - 1)
        if self.args.psnr_weight:
            loss = loss + self.args.psnr_weight * (-10 * torch.log10(1.0 / (F.mse_loss(x_rec, x) + 1e-8)))
        if self.args.ssim_weight:
            loss = loss + self.args.ssim_weight * (1.0 - structural_similarity_index_measure(x_rec, x, data_range=1.0))
        return loss

    def expand(self, decoder, z_main, z_aux, x, plan, lpips_on: bool) -> torch.Tensor:
        w = plan["w"]
        w_z = w.view(-1, *([1] * (z_main.dim() - 1))).to(z_main.dtype)
        x_rec = decode(self.base, decoder, w_z * z_main + (1 - w_z) * z_aux + plan["noise_exp"])
        per_sample = (x_rec - x).abs().mean(dim=(1, 2, 3))
        if lpips_on:
            per_sample = per_sample + self.perceptual_weight * self.lpips(x * 2 - 1, x_rec * 2 - 1, reduction="none").view(-1)
        if self.args.psnr_weight:
            mse = (x_rec - x).pow(2).mean(dim=(1, 2, 3))
            per_sample = per_sample + self.args.psnr_weight * (-10 * torch.log10(1.0 / (mse + 1e-8)))
        if self.args.ssim_weight:
            ssim = structural_similarity_index_measure(x_rec, x, data_range=1.0, reduction="none")
            per_sample = per_sample + self.args.ssim_weight * (1.0 - ssim)
        return (w * per_sample).mean()

    def align(self, z, frozen) -> torch.Tensor:
        if not (self.args.cos_weight or self.args.mse_weight):
            return z.new_zeros(())
        tokens = to_tokens(z)
        cos = 1 - F.cosine_similarity(tokens, frozen, dim=-1).mean()
        mse = F.mse_loss(tokens.mean(dim=1), frozen.mean(dim=1))
        return self.args.cos_weight * cos + self.args.mse_weight * mse

    def geometry(self, z_mr, z_ct, frozen_mr, frozen_ct) -> torch.Tensor:
        z_mr, z_ct = to_tokens(z_mr).float(), to_tokens(z_ct).float()
        frozen_mr, frozen_ct = frozen_mr.float(), frozen_ct.float()
        anchor = 0.5 * (
            (1 - F.cosine_similarity(z_mr, frozen_mr, dim=-1, eps=1e-6)).mean()
            + (1 - F.cosine_similarity(z_ct, frozen_ct, dim=-1, eps=1e-6)).mean()
        )
        v = z_mr.mean(dim=1) - z_ct.mean(dim=1)
        v0 = frozen_mr.mean(dim=1) - frozen_ct.mean(dim=1)
        vel_cos = (1 - F.cosine_similarity(v, v0, dim=-1, eps=1e-6)).mean()
        ratio = torch.linalg.vector_norm(v, dim=-1) / (torch.linalg.vector_norm(v0, dim=-1) + 1e-6)
        vnorm = ((ratio - 1) ** 2).mean()
        return (self.args.lambda_token_anchor * anchor + self.args.lambda_vel_cos * vel_cos
                + self.args.lambda_vnorm * vnorm)

    @property
    def use_geometry(self) -> bool:
        return self.args.dataset == "mri_ct" and bool(
            self.args.lambda_token_anchor or self.args.lambda_vel_cos or self.args.lambda_vnorm
        )


# ---------------------------------------------------------------------------
#  One training step
# ---------------------------------------------------------------------------
class Step3Trainer:
    def __init__(self, model, ema, losses: Step3Losses, enc_opt, dec_opts, schedulers, args,
                 device, world_size: int, ema_decay: float, clip_grad: Optional[float], autocast_kwargs):
        self.model = model
        self.ema = ema
        self.losses = losses
        self.enc_opt = enc_opt
        self.dec_opts = dec_opts
        self.schedulers = schedulers
        self.offload = args.offload_decoders
        self.device = device
        self.world_size = world_size
        self.ema_decay = ema_decay
        self.clip_grad = clip_grad
        self.autocast_kwargs = autocast_kwargs

    def autocast(self):
        return torch.autocast(device_type=self.device.type, **self.autocast_kwargs)

    def step(self, imgs: List[torch.Tensor], lpips_on: bool) -> Dict[str, float]:
        M = len(imgs)
        losses = self.losses
        with torch.no_grad(), self.autocast():
            latent_shape = losses.encode(imgs[0][:1]).shape[1:]
        frozen = [frozen_tokens(losses.base, losses.frozen_encoder, x) for x in imgs]
        plans = losses.plan(imgs[0].size(0), M, latent_shape, self.device)
        self.enc_opt.zero_grad(set_to_none=True)
        if self.offload:
            stats, dec_sq = self._forward_backward_offload(imgs, frozen, plans, lpips_on)
        else:
            for opt in self.dec_opts:
                opt.zero_grad(set_to_none=True)
            stats = self._forward_backward_resident(imgs, frozen, plans, lpips_on)
            dec_sq = 0.0
            for decoder in self.model.decoders:
                grads = grads_of(decoder)
                allreduce_mean(grads, self.world_size)
                dec_sq += sum(g.float().pow(2).sum().item() for g in grads)
        enc_grads = grads_of(self.model.encoder)
        allreduce_mean(enc_grads, self.world_size)

        coef = 1.0
        if self.clip_grad is not None:
            total = math.sqrt(dec_sq + sum(g.float().pow(2).sum().item() for g in enc_grads))
            coef = min(1.0, self.clip_grad / (total + 1e-6))
            stats["grad_norm"] = total

        for g in enc_grads:
            g.mul_(coef)
        self.enc_opt.step()
        ema_update(self.ema.encoder, self.model.encoder, self.ema_decay)
        for m, (decoder, ema_decoder, opt) in enumerate(zip(self.model.decoders, self.ema.decoders, self.dec_opts)):
            if self.offload:
                decoder.to(self.device)
                ema_decoder.to(self.device)
                move_optimizer_state(opt, self.device)
            for g in grads_of(decoder):
                g.mul_(coef)
            opt.step()
            opt.zero_grad(set_to_none=True)
            ema_update(ema_decoder, decoder, self.ema_decay)
            if self.offload:
                decoder.to("cpu")
                ema_decoder.to("cpu")
                move_optimizer_state(opt, "cpu")
        for scheduler in self.schedulers:
            scheduler.step()
        return stats

    def _forward_backward_resident(self, imgs, frozen, plans, lpips_on) -> Dict[str, float]:
        """All decoders on the GPU: rec + align for every modality in one graph, then expand per modality."""
        M, losses, decoders = len(imgs), self.losses, self.model.decoders
        stats = {"rec": 0.0, "align": 0.0, "expand": 0.0, "geometry": 0.0}
        with self.autocast():
            total = 0.0
            z_list = []
            for m in range(M):
                z = losses.encode(imgs[m])
                rec = losses.rec(decoders[m], z, imgs[m], plans[m]["noise_rec"], lpips_on)
                align = losses.align(z, frozen[m])
                total = total + (rec + align) / M
                z_list.append(z)
                stats["rec"] += rec.item() / M
                stats["align"] += align.item() / M
            if losses.use_geometry:
                geometry = losses.geometry(z_list[0], z_list[1], frozen[0], frozen[1])
                total = total + geometry
                stats["geometry"] = geometry.item()
        total.backward()
        del total, z_list
        for m in range(M):
            with self.autocast():
                z_main = losses.encode(imgs[m])
                z_aux = torch.stack([losses.encode(imgs[k]) for k in plans[m]["subset"]]).mean(dim=0)
                expand = losses.expand(decoders[m], z_main, z_aux, imgs[m], plans[m], lpips_on) / M
            expand.backward()
            stats["expand"] += expand.item()
        return stats

    def _forward_backward_offload(self, imgs, frozen, plans, lpips_on) -> Tuple[Dict[str, float], float]:
        """One decoder on the GPU at a time; its gradients are reduced and parked in CPU memory."""
        M, losses, decoders = len(imgs), self.losses, self.model.decoders
        stats = {"rec": 0.0, "align": 0.0, "expand": 0.0, "geometry": 0.0}
        dec_sq = 0.0
        if losses.use_geometry:
            with self.autocast():
                geometry = losses.geometry(losses.encode(imgs[0]), losses.encode(imgs[1]), frozen[0], frozen[1])
            geometry.backward()
            stats["geometry"] = geometry.item()
        for m in range(M):
            decoder = decoders[m]
            decoder.to(self.device)
            with self.autocast():
                z = losses.encode(imgs[m])
                rec = losses.rec(decoder, z, imgs[m], plans[m]["noise_rec"], lpips_on)
                align = losses.align(z, frozen[m])
                z_aux = torch.stack([losses.encode(imgs[k]) for k in plans[m]["subset"]]).mean(dim=0)
                expand = losses.expand(decoder, z, z_aux, imgs[m], plans[m], lpips_on)
                loss = (rec + align + expand) / M
            loss.backward()
            grads = grads_of(decoder)
            allreduce_mean(grads, self.world_size)
            dec_sq += sum(g.float().pow(2).sum().item() for g in grads)
            decoder.to("cpu")
            stats["rec"] += rec.item() / M
            stats["align"] += align.item() / M
            stats["expand"] += expand.item() / M
        return stats, dec_sq


# ---------------------------------------------------------------------------
#  Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(path, step, epoch, model, ema, enc_opt, dec_opts, schedulers) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": {"encoder": enc_opt.state_dict(), "decoders": [opt.state_dict() for opt in dec_opts]},
        "scheduler": [s.state_dict() for s in schedulers],
    }, path)


def load_checkpoint(path, model, ema, enc_opt, dec_opts, schedulers) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    enc_opt.load_state_dict(ckpt["optimizer"]["encoder"])
    for opt, state in zip(dec_opts, ckpt["optimizer"]["decoders"]):
        opt.load_state_dict(state)
    for scheduler, state in zip(schedulers, ckpt["scheduler"]):
        scheduler.load_state_dict(state)
    return ckpt.get("epoch", 0), ckpt.get("step", 0)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()

    full_cfg = OmegaConf.load(args.config)
    rae_config = parse_configs(full_cfg)[0]
    training_cfg = dict(OmegaConf.to_container(full_cfg.get("training", {}), resolve=True))
    loss_cfg = dict(OmegaConf.to_container(full_cfg.get("gan", {}), resolve=True)).get("loss", {})
    perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
    lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))

    global_batch_size = args.global_batch_size or int(training_cfg.get("global_batch_size", 16 * world_size))
    assert global_batch_size % world_size == 0, "global batch size must be divisible by the world size"
    batch_size = global_batch_size // world_size
    num_workers = int(training_cfg.get("num_workers", 4))
    clip_grad = float(training_cfg.get("clip_grad") or 0.0) or None
    log_interval = int(training_cfg.get("log_interval", 100))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 1))
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 50))
    global_seed = args.global_seed if args.global_seed is not None else int(training_cfg.get("global_seed", 0))
    seed = global_seed * world_size + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    if world_size > 1:
        dist.barrier()

    #### Model
    num_modalities = 2 if args.dataset == "mri_ct" else 4
    stage1_paths = [getattr(args, f"stage1_path_{m}") for m in range(num_modalities)]
    assert all(stage1_paths), f"--stage1-path-0..{num_modalities - 1} are required"
    rae_config = OmegaConf.create(OmegaConf.to_container(rae_config, resolve=True))
    rae_config.pop("ckpt", None)
    base: RAE = instantiate_from_config(rae_config)
    base.load_state_dict(torch.load(stage1_paths[0], map_location="cpu")["ema"], strict=False)
    model = RAE4Dec(base, num_modalities)
    for m in range(1, num_modalities):
        ema_sd = torch.load(stage1_paths[m], map_location="cpu")["ema"]
        model.decoders[m].load_state_dict({k[len("decoder."):]: v for k, v in ema_sd.items() if k.startswith("decoder.")})
        del ema_sd
    del base.decoder
    noise_tau = float(base.noise_tau)
    base.noise_tau = 0
    base.to(device)
    frozen_encoder = deepcopy(model.encoder).to(device).eval().requires_grad_(False)

    if args.encoder_lora_rank > 0:
        scope = apply_encoder_lora(model.encoder, args.encoder_lora_rank, args.encoder_lora_alpha,
                                   args.encoder_lora_dropout, args.encoder_lora_targets, args.encoder_lora_last_n)
        model.encoder.to(device)
    else:
        model.encoder.requires_grad_(True)
        scope = "full encoder"
    model.train()
    ema = deepcopy(model).eval().requires_grad_(False)
    ema.encoder.to(device)
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(model.encoder)
    if not args.offload_decoders:
        model.decoders.to(device)
        ema.decoders.to(device)

    #### Optimizers (encoder + one per decoder)
    enc_opt, optim_msg = build_optimizer([p for p in model.encoder.parameters() if p.requires_grad], training_cfg)
    dec_opts = []
    for decoder in model.decoders:
        decoder.to(device)
        dec_opts.append(build_optimizer([p for p in decoder.parameters() if p.requires_grad], training_cfg)[0])
        if args.offload_decoders:
            decoder.to("cpu")

    #### Data
    from brats_dataset import BraTS2023AllDataset, get_transforms
    dataset = BraTS2023AllDataset(args.data_path, phase="train", transform=get_transforms("train"),
                                  num_modalities=num_modalities)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                        pin_memory=True, drop_last=True)
    steps_per_epoch = len(loader)
    schedulers = [build_scheduler(opt, steps_per_epoch, training_cfg)[0] for opt in [enc_opt, *dec_opts]] \
        if training_cfg.get("scheduler") else []

    lpips = LPIPS().to(device).eval().requires_grad_(False)
    losses = Step3Losses(args, base, frozen_encoder, lpips, perceptual_weight, noise_tau)
    _, autocast_kwargs = get_autocast_scaler(args)
    trainer = Step3Trainer(model, ema, losses, enc_opt, dec_opts, schedulers, args, device, world_size,
                           ema_decay, clip_grad, autocast_kwargs)

    #### Resume
    start_epoch, global_step = 0, 0
    resume_path = find_resume_checkpoint(experiment_dir)
    if resume_path is not None:
        start_epoch, global_step = load_checkpoint(resume_path, model, ema, enc_opt, dec_opts, schedulers)
        logger.info(f"Resumed from {resume_path} (epoch={start_epoch}, step={global_step}).")
    elif rank == 0:
        save_worktree(experiment_dir, full_cfg)

    if rank == 0:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Encoder: {scope}. Trainable parameters: {n_train / 1e6:.1f}M. {optim_msg}")
        logger.info(f"Decoders {'offloaded to CPU' if args.offload_decoders else 'resident on GPU'}; "
                    f"gradient clipping: {clip_grad}; w sampling: {args.w_sampling_mode}.")
        logger.info(f"{len(dataset)} samples, {steps_per_epoch} steps/epoch, batch {batch_size}/GPU, "
                    f"epochs {start_epoch} -> {num_epochs}.")

    #### Train
    running: Dict[str, float] = {}
    last_epoch = start_epoch
    for epoch in range(start_epoch, num_epochs):
        last_epoch = epoch + 1
        sampler.set_epoch(epoch)
        for batch in loader:
            imgs = [x.to(device, non_blocking=True) for x in batch]
            stats = trainer.step(imgs, lpips_on=epoch >= lpips_start_epoch and perceptual_weight > 0)
            global_step += 1
            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v
            if global_step % log_interval == 0 and rank == 0:
                logs = {f"train/{k}": v / log_interval for k, v in running.items()}
                logs["train/lr"] = enc_opt.param_groups[0]["lr"]
                logger.info(f"(step={global_step:07d} epoch={epoch}) " + ", ".join(f"{k[6:]}={v:.4f}" for k, v in logs.items()))
                if args.wandb:
                    wandb_utils.log(logs, step=global_step)
                if args.swanlab:
                    swanlab_utils.log(logs, step=global_step)
            if global_step % log_interval == 0:
                running = {}
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if rank == 0 and checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(f"{checkpoint_dir}/ep-{epoch + 1:07d}.pt", global_step, epoch + 1,
                            model, ema, enc_opt, dec_opts, schedulers)
            for old in sorted(glob(f"{checkpoint_dir}/ep-[0-9]*.pt"))[:-2]:
                os.remove(old)
        if world_size > 1:
            dist.barrier()
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    if rank == 0:
        save_checkpoint(f"{checkpoint_dir}/ep-last.pt", global_step, last_epoch,
                        model, ema, enc_opt, dec_opts, schedulers)
        logger.info("Done.")
    cleanup_distributed()


if __name__ == "__main__":
    main()

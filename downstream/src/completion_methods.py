"""Modality completion methods for BraTS missing-modality segmentation.

Each method implements the ``CompletionMethod`` interface:

    method = SomeMethod(ckpt='...', slice_batch=8)
    method.load(device)
    completed = method.complete(volume, mask)   # (4, D, H, W) -> (4, D, H, W)
    method.unload()

Conventions:
    * Volumes: ``torch.float32`` in ``[0, 1]``, shape ``(4, D, H, W)`` with the
      BraTS channel order ``[T1, T1ce, T2, FLAIR]``. Missing modalities are
      zero-filled on input.
    * Masks: ``torch.float32`` in ``{0, 1}``, shape ``(4,)``. ``1=available``,
      ``0=missing``.
    * Output: a *complete* 4-modality volume in ``[0, 1]``. Available
      modalities are preserved untouched; missing ones are filled by the model.

The main script (``eval_completion.py``) handles the per-pattern, per-case
loop and the segmentation/Dice evaluation. SphereFlow is *not* in this file - it
runs in a separate ``sphereflow`` conda env via a worker subprocess.
"""
from __future__ import annotations

import os
import sys
import argparse
from typing import List

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BASELINE_SRC = os.path.join(PROJECT_ROOT, 'baseline', 'src')
if BASELINE_SRC not in sys.path:
    sys.path.insert(0, BASELINE_SRC)


# ---------------------------------------------------------------------------
#  Default checkpoint locations (override via CLI args in eval_completion.py)
# ---------------------------------------------------------------------------
# Checkpoints are looked up under $CKPT_DIR (default: <repo>/checkpoints); see README.md
_CKPT_DIR = os.environ.get('CKPT_DIR', os.path.join(PROJECT_ROOT, 'checkpoints'))
DEFAULT_CKPTS = {
    'MMT':       os.path.join(_CKPT_DIR, 'baselines/brats/MMT.pt'),
    'M2DN':      os.path.join(_CKPT_DIR, 'baselines/brats/M2DN.pt'),
    'ModImp':    os.path.join(_CKPT_DIR, 'baselines/brats/ModImp.pt'),
    'Zhang':     os.path.join(_CKPT_DIR, 'baselines/brats/Zhang.pt'),
    'MMGAN':     os.path.join(_CKPT_DIR, 'baselines/brats/MMGAN.pt'),
    'MMSyn':     os.path.join(_CKPT_DIR, 'baselines/brats/MMSyn.pt'),
}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _load_state_dict(model: nn.Module, ckpt_path: str, key_aliases=('state_dict', 'model', 'ema')):
    """Load a state-dict into model, stripping DDP / torch.compile prefixes
    and skipping keys whose shapes don't match (with a warning).
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        sd = None
        for k in key_aliases:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                break
        if sd is None:
            sd = ckpt
    else:
        sd = ckpt

    cleaned = {}
    for k, v in sd.items():
        kk = k.replace('module.', '').replace('_orig_mod.', '')
        cleaned[kk] = v

    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in cleaned.items():
        if k in model_sd and hasattr(v, 'shape') and model_sd[k].shape == v.shape:
            filtered[k] = v
        elif k in model_sd:
            skipped.append((k, list(v.shape) if hasattr(v, 'shape') else None,
                            list(model_sd[k].shape)))
    info = model.load_state_dict(filtered, strict=False)
    return {
        'loaded': len(filtered),
        'missing': len(info.missing_keys),
        'unexpected': len(info.unexpected_keys),
        'skipped_shape_mismatch': skipped[:5],
    }


def _slice_iter(D: int, batch: int):
    """Yield (start, end) for slice-batches over depth D."""
    for s in range(0, D, batch):
        yield s, min(s + batch, D)


# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------
class CompletionMethod:
    name = 'Base'

    def __init__(self, ckpt: str = None, slice_batch: int = 8):
        self.ckpt = ckpt
        self.slice_batch = slice_batch
        self.device = None
        self.model = None

    def load(self, device):
        raise NotImplementedError

    @torch.no_grad()
    def complete(self, volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unload(self):
        for attr in ('model', 'rec_model', 'grad_model'):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
#  Zero-fill baseline (no model)
# ---------------------------------------------------------------------------
class ZeroFill(CompletionMethod):
    name = 'Zero'

    def __init__(self):
        super().__init__(ckpt=None, slice_batch=0)

    def load(self, device):
        self.device = device

    @torch.no_grad()
    def complete(self, volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.view(4, 1, 1, 1).to(volume.device)
        return volume * m


# ---------------------------------------------------------------------------
#  MMT
# ---------------------------------------------------------------------------
class MMTMethod(CompletionMethod):
    name = 'MMT'

    def __init__(self, ckpt: str = None, slice_batch: int = 8):
        super().__init__(ckpt or DEFAULT_CKPTS['MMT'], slice_batch)

    def load(self, device):
        from network.MMT.mmt import MMT
        self.device = device
        self.model = MMT(patch_size=4, dim=6, depths=[2, 2, 2, 2],
                         num_heads=[3, 6, 12, 24]).to(device)
        info = _load_state_dict(self.model, self.ckpt)
        print(f"  [MMT] loaded {info['loaded']} keys "
              f"(missing={info['missing']}, unexpected={info['unexpected']})")
        self.model.eval()

    @torch.no_grad()
    def complete(self, volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # MMT eval returns ONLY the missing modalities, in the order of
        # missing_indices = [i for i, v in enumerate(mask) if v == 0]
        out = volume.clone()
        D = volume.shape[1]
        missing = [i for i, v in enumerate(mask.tolist()) if v == 0]
        if not missing:
            return out

        m = mask.unsqueeze(0).to(self.device).float()  # (1, 4)
        for s, e in _slice_iter(D, self.slice_batch):
            x = volume[:, s:e].permute(1, 0, 2, 3).unsqueeze(2).to(self.device)  # (B, 4, 1, H, W)
            mb = m.expand(x.size(0), -1)
            y_gen = self.model(x, mb)            # (B, N_miss, 1, H, W)
            for j, mi in enumerate(missing):
                out[mi, s:e] = y_gen[:, j, 0].clamp(0, 1).cpu()
        return out


# ---------------------------------------------------------------------------
#  M2DN
# ---------------------------------------------------------------------------
class M2DNMethod(CompletionMethod):
    name = 'M2DN'

    def __init__(self, ckpt: str = None, slice_batch: int = 4):
        super().__init__(ckpt or DEFAULT_CKPTS['M2DN'], slice_batch)

    def load(self, device):
        from network.M2DN.m2dn import M2DN
        self.device = device
        self.model = M2DN().to(device)
        info = _load_state_dict(self.model, self.ckpt)
        print(f"  [M2DN] loaded {info['loaded']} keys "
              f"(missing={info['missing']}, unexpected={info['unexpected']})")
        self.model.eval()

    @torch.no_grad()
    def complete(self, volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # M2DN returns full 4-mod tensor (B, 4, 1, H, W) - DDIM sampling.
        out = volume.clone()
        D = volume.shape[1]
        missing = [i for i, v in enumerate(mask.tolist()) if v == 0]
        if not missing:
            return out

        m = mask.unsqueeze(0).to(self.device).float()  # (1, 4)
        for s, e in _slice_iter(D, self.slice_batch):
            x = volume[:, s:e].permute(1, 0, 2, 3).unsqueeze(2).to(self.device)  # (B, 4, 1, H, W)
            mb = m.expand(x.size(0), -1)
            pred = self.model(x, mask_indices=mb, mode='inference')  # (B, 4, 1, H, W)
            for mi in missing:
                out[mi, s:e] = pred[:, mi, 0].clamp(0, 1).cpu()
        return out


# ---------------------------------------------------------------------------
#  ModImp
# ---------------------------------------------------------------------------
class ModImpMethod(CompletionMethod):
    name = 'ModImp'

    def __init__(self, ckpt: str = None, slice_batch: int = 8):
        super().__init__(ckpt or DEFAULT_CKPTS['ModImp'], slice_batch)

    def load(self, device):
        from network.ModImp.modimp import ImputationModel
        self.device = device
        self.model = ImputationModel(
            img_size=256, patch_size=2, embed_dim=24,
            depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
            con_dim=512, mask_ratio_spa=0.5,
        ).to(device)
        info = _load_state_dict(self.model, self.ckpt)
        print(f"  [ModImp] loaded {info['loaded']} keys "
              f"(missing={info['missing']}, unexpected={info['unexpected']})")
        self.model.eval()

    @torch.no_grad()
    def complete(self, volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # ModImp returns logits of shape (B, 4, H, W) - no extra C dim.
        # Every row in a batch must have the same number of kept modalities,
        # which is satisfied here because we batch by-pattern (same mask everywhere).
        out = volume.clone()
        D = volume.shape[1]
        missing = [i for i, v in enumerate(mask.tolist()) if v == 0]
        if not missing:
            return out

        m = mask.unsqueeze(0).to(self.device).float()  # (1, 4)
        for s, e in _slice_iter(D, self.slice_batch):
            x = volume[:, s:e].permute(1, 0, 2, 3).unsqueeze(2).to(self.device)  # (B, 4, 1, H, W)
            mb = m.expand(x.size(0), -1)
            logits, _, _ = self.model(x, mb)     # (B, 4, H, W)
            for mi in missing:
                out[mi, s:e] = logits[:, mi].clamp(0, 1).cpu()
        return out


# ---------------------------------------------------------------------------
#  Zhang
# ---------------------------------------------------------------------------
class ZhangMethod(CompletionMethod):
    name = 'Zhang'

    def __init__(self, ckpt: str = None, slice_batch: int = 8):
        super().__init__(ckpt or DEFAULT_CKPTS['Zhang'], slice_batch)

    def load(self, device):
        from network.Zhang.zhang import ZhangModel
        self.device = device
        self.model = ZhangModel(num_mods=4).to(device)
        info = _load_state_dict(self.model, self.ckpt)
        print(f"  [Zhang] loaded {info['loaded']} keys "
              f"(missing={info['missing']}, unexpected={info['unexpected']})")
        self.model.eval()

    @torch.no_grad()
    def complete(self, volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Zhang expects [-1, 1] input; output is in [-1, 1] via tanh.
        from network.Zhang.zhang import zhang_to_model_range, zhang_to_image_range
        out = volume.clone()
        D = volume.shape[1]
        missing = [i for i, v in enumerate(mask.tolist()) if v == 0]
        if not missing:
            return out

        m = mask.unsqueeze(0).to(self.device).float()  # (1, 4)
        for s, e in _slice_iter(D, self.slice_batch):
            x = volume[:, s:e].permute(1, 0, 2, 3).unsqueeze(2).to(self.device)  # (B, 4, 1, H, W)
            mb = m.expand(x.size(0), -1)
            x_neg11 = zhang_to_model_range(x) * mb.view(-1, 4, 1, 1, 1)
            pred = self.model(x_neg11, mb)        # (B, 4, 1, H, W) in [-1, 1]
            pred01 = zhang_to_image_range(pred).clamp(0, 1)
            for mi in missing:
                out[mi, s:e] = pred01[:, mi, 0].cpu()
        return out


# ---------------------------------------------------------------------------
#  MMGAN  (Sharma & Hamarneh, IEEE TMI 2020 — multi-modal pix2pix-style GAN)
# ---------------------------------------------------------------------------
class MMGANMethod(CompletionMethod):
    name = 'MMGAN'

    def __init__(self, ckpt: str = None, slice_batch: int = 8,
                 z_type: str = 'noise', use_ic: bool = True,
                 with_tanh: bool = False):
        super().__init__(ckpt or DEFAULT_CKPTS['MMGAN'], slice_batch)
        self.z_type = z_type
        self.use_ic = use_ic
        self.with_tanh = with_tanh

    def load(self, device):
        from network.MMGAN.mmgan import MMGAN
        self.device = device
        self.model = MMGAN(
            num_mods=4,
            z_type=self.z_type,
            use_ic=self.use_ic,
            with_tanh=self.with_tanh,
            with_relu=not self.with_tanh,
        ).to(device)
        info = _load_state_dict(self.model, self.ckpt)
        print(f"  [MMGAN] loaded {info['loaded']} keys "
              f"(missing={info['missing']}, unexpected={info['unexpected']})")
        self.model.eval()

    @torch.no_grad()
    def complete(self, volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # MMGAN.forward accepts [B, N, H, W] or [B, N, 1, H, W] and returns
        # [B, N, H, W] (4-D, IC already overwrites available channels).
        out = volume.clone()
        D = volume.shape[1]
        missing = [i for i, v in enumerate(mask.tolist()) if v == 0]
        if not missing:
            return out

        m = mask.unsqueeze(0).to(self.device).float()  # (1, 4)
        # Deterministic noise impute across runs / patterns / cases.
        torch.manual_seed(0)
        for s, e in _slice_iter(D, self.slice_batch):
            x = volume[:, s:e].permute(1, 0, 2, 3).unsqueeze(2).to(self.device)  # (B, 4, 1, H, W)
            mb = m.expand(x.size(0), -1)
            fake = self.model(x, mb)             # (B, 4, H, W)
            for mi in missing:
                out[mi, s:e] = fake[:, mi].clamp(0, 1).cpu()
        return out


# ---------------------------------------------------------------------------
# MMSyn (Chartsias et al., IEEE TMI 2018)
# ---------------------------------------------------------------------------
class MMSynMethod(CompletionMethod):
    name = 'MMSyn'

    def __init__(self, ckpt: str = None, slice_batch: int = 4,
                 latent_dim: int = 16, fusion: str = 'max'):
        super().__init__(ckpt or DEFAULT_CKPTS['MMSyn'], slice_batch)
        self.latent_dim = latent_dim
        self.fusion = fusion

    def load(self, device):
        from network.MMSyn.mmsyn import MMSyn
        self.device = device
        self.model = MMSyn(
            num_mods=4, latent_dim=self.latent_dim,
            in_channels=1, out_channels=1, fusion=self.fusion,
        ).to(device)
        info = _load_state_dict(self.model, self.ckpt)
        print(f"  [MMSyn] loaded {info['loaded']} keys "
              f"(missing={info['missing']}, unexpected={info['unexpected']})")
        self.model.eval()

    @torch.no_grad()
    def complete(self, volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # MMSyn.forward returns dict; 'fused' is decoder_j(fused_latent) of
        # shape (B, 4, 1, H, W). Linear-head output may overshoot [0,1] so clamp.
        out = volume.clone()
        D = volume.shape[1]
        missing = [i for i, v in enumerate(mask.tolist()) if v == 0]
        if not missing:
            return out

        m = mask.unsqueeze(0).to(self.device).float()  # (1, 4)
        for s, e in _slice_iter(D, self.slice_batch):
            x = volume[:, s:e].permute(1, 0, 2, 3).unsqueeze(2).to(self.device)  # (B, 4, 1, H, W)
            mb = m.expand(x.size(0), -1)
            pred = self.model(x, mb)['fused']    # (B, 4, 1, H, W)
            for mi in missing:
                out[mi, s:e] = pred[:, mi, 0].clamp(0, 1).cpu()
        return out


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------
INLINE_METHOD_REGISTRY = {
    'Zero':      ZeroFill,
    'MMT':       MMTMethod,
    'M2DN':      M2DNMethod,
    'ModImp':    ModImpMethod,
    'Zhang':     ZhangMethod,
    'MMGAN':     MMGANMethod,
    'MMSyn':     MMSynMethod,
}

ALL_METHODS = ['Zero', 'MMT', 'M2DN', 'ModImp', 'Zhang', 'MMGAN', 'MMSyn', 'SphereFlow']

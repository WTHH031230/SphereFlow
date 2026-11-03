"""
PyTorch port of Chartsias et al., "Multimodal MR Synthesis via Modality-Invariant
Latent Representation" (IEEE TMI 2018) and Joyce et al., "Robust Multi-Modal MR
Image Synthesis" (MICCAI 2017).

Original Keras implementation: https://github.com/agis85/multimodal_brain_synthesis

Architecture:
  Per-modality EncoderUNet  : [B, 1, H, W] → [B, ld_enc, H, W]
  Fusion  (max / ave / hemis): N latents + mask → [B, ld_dec, H, W]
  Per-modality Decoder      : latent → [B, 1, H, W]
                              applied to *each individual* latent and the fused one,
                              so the loss can enforce a modality-invariant latent.

Mask convention (shared with the rest of this project):
  mask[b, i] = 1.0  → modality i is available for sample b
  mask[b, i] = 0.0  → modality i is missing
"""

from typing import Dict

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _conv_lrelu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv 3×3 (same) + LeakyReLU(0.01) — Keras default 'lrelu'."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.01, inplace=True),
    )


def _conv_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Encoder (per-modality, U-Net-style with two down/up levels)
# ─────────────────────────────────────────────────────────────────────────────

class EncoderUNet(nn.Module):
    """
    Mirrors `Multimodel.encoder_maker` from the Keras source: 32→64→128 channels,
    two MaxPool downsamples + two NN-upsamples with concat skips, and a final
    `latent_dim` 3×3 conv with LeakyReLU.

    Input  : [B, in_channels, H, W]
    Output : [B, latent_dim, H, W]   (spatial size preserved; H,W must be div by 4)
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 16):
        super().__init__()

        # ── Down path
        self.l0 = nn.Sequential(_conv_lrelu(in_channels, 32), _conv_lrelu(32, 32))
        self.pool1 = nn.MaxPool2d(2)
        self.l1 = nn.Sequential(_conv_lrelu(32, 64), _conv_lrelu(64, 64))
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(_conv_lrelu(64, 128), _conv_lrelu(128, 128))

        # ── Up path
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1_proj = _conv_lrelu(128, 64)
        self.l1_dec = nn.Sequential(_conv_lrelu(128, 64), _conv_lrelu(64, 64))
        self.up0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up0_proj = _conv_lrelu(64, 32)
        self.l0_dec = nn.Sequential(_conv_lrelu(64, 32), _conv_lrelu(32, 32))

        # ── Latent head (LeakyReLU on the latent, as in the Keras code)
        self.latent_head = nn.Sequential(
            nn.Conv2d(32, latent_dim, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.l0(x)                                  # [B, 32, H, W]
        s1 = self.l1(self.pool1(s0))                     # [B, 64, H/2, W/2]
        b = self.bottleneck(self.pool2(s1))              # [B, 128, H/4, W/4]

        u1 = self.up1_proj(self.up1(b))                  # [B, 64, H/2, W/2]
        u1 = self.l1_dec(torch.cat([s1, u1], dim=1))     # [B, 64, H/2, W/2]
        u0 = self.up0_proj(self.up0(u1))                 # [B, 32, H, W]
        u0 = self.l0_dec(torch.cat([s0, u0], dim=1))     # [B, 32, H, W]
        return self.latent_head(u0)                      # [B, latent_dim, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Decoder (per-modality, residual-concat ResNet-lite)
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Mirrors `Multimodel.decoder_maker` (two CB blocks with residual concat skips,
    then a 1×1 conv + ReLU head).

    Input  : [B, latent_dim, H, W]
    Output : [B, out_channels, H, W]
    """

    def __init__(self, latent_dim: int = 16, out_channels: int = 1, relu_head: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.b1 = nn.Sequential(_conv_relu(latent_dim, 32), _conv_relu(32, 32))
        self.b2 = nn.Sequential(_conv_relu(latent_dim + 32, 32), _conv_relu(32, 32))
        # Head sees the cumulative skip stack: latent + b1 + b2.
        # Linear head by default; outputs are clamped to [0, 1] at evaluation.
        self.head = nn.Conv2d(latent_dim + 64, out_channels, 1)
        if relu_head:
            self.head = nn.Sequential(self.head, nn.ReLU(inplace=True))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        c1 = self.b1(z)
        s1 = torch.cat([z, c1], dim=1)
        c2 = self.b2(s1)
        s2 = torch.cat([s1, c2], dim=1)
        return self.head(s2)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level model
# ─────────────────────────────────────────────────────────────────────────────

class MMSyn(nn.Module):
    """
    Multimodal Brain Synthesis with modality-invariant latent representation.

    Args
    ----
    num_mods    : number of modalities N (4 for BraTS, 2 for MRI/CT).
    latent_dim  : latent channels seen by the *decoder* (paper default 16).
    in_channels : per-modality input channels (1 for grey-scale).
    out_channels: per-modality output channels (1).
    fusion      : 'max' | 'ave' | 'hemis'  — how to merge per-modality latents.
    relu_head   : end each decoder with ReLU (paper) instead of a linear head.

    forward(img, mask) returns a dict with:
      'fused' : [B, N, 1, H, W]      decoder_j(fused_latent) for each j
      'ind'   : [B, N, N, 1, H, W]   ind[:, i, j] = decoder_j(latent_i)
      'emb'   : [B, N, ld_enc, H, W] per-modality individual latents
      'fused_emb' : [B, ld_dec, H, W] fused latent before decoding
      'mask'  : [B, N]               echoed for downstream loss masking

    DDP-safety: all N encoders and N decoders are exercised every forward pass
    (we feed zero-imputed inputs to the missing-modality encoders so their
    parameters always receive gradients).
    """

    FUSION_MODES = ('max', 'ave', 'hemis')

    def __init__(self,
                 num_mods: int = 4,
                 latent_dim: int = 16,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 fusion: str = 'max',
                 relu_head: bool = False):
        super().__init__()
        if fusion not in self.FUSION_MODES:
            raise ValueError(f"fusion must be one of {self.FUSION_MODES}, got {fusion!r}")
        if fusion == 'hemis' and latent_dim % 2 != 0:
            raise ValueError("hemis fusion requires an even latent_dim")

        self.num_mods = num_mods
        self.fusion = fusion
        self.latent_dim = latent_dim                      # what the decoder sees
        self.enc_latent_dim = latent_dim // 2 if fusion == 'hemis' else latent_dim

        self.encoders = nn.ModuleList([
            EncoderUNet(in_channels=in_channels, latent_dim=self.enc_latent_dim)
            for _ in range(num_mods)
        ])
        self.decoders = nn.ModuleList([
            Decoder(latent_dim=latent_dim, out_channels=out_channels, relu_head=relu_head)
            for _ in range(num_mods)
        ])

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_5d(img: torch.Tensor) -> torch.Tensor:
        """Accept [B, N, H, W] or [B, N, 1, H, W] and return [B, N, 1, H, W]."""
        if img.dim() == 5:
            return img
        if img.dim() == 4:
            return img.unsqueeze(2)
        raise ValueError(f"Expected 4-D or 5-D image tensor, got {tuple(img.shape)}")

    def _fuse(self, embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        embs : [B, N, ld_enc, H, W]
        mask : [B, N]
        returns fused latent [B, ld_dec, H, W]
        """
        B, N, _, H, W = embs.shape
        m5 = mask.to(dtype=embs.dtype, device=embs.device).view(B, N, 1, 1, 1)

        if self.fusion == 'max':
            very_neg = torch.finfo(embs.dtype).min / 4.0  # avoid overflow when subtracted
            masked = torch.where(m5 > 0.5, embs, embs.new_full((), very_neg))
            # If a sample has no available modality, fall back to zero rather than -inf.
            no_avail = (mask.sum(dim=1, keepdim=True) <= 0)  # [B, 1]
            fused = masked.max(dim=1).values                  # [B, ld_enc, H, W]
            fused = torch.where(no_avail.view(B, 1, 1, 1).expand_as(fused),
                                fused.new_zeros(()), fused)
            return fused

        if self.fusion == 'ave':
            denom = mask.to(embs.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
            weighted = (embs * m5).sum(dim=1)                                    # [B, ld_enc, H, W]
            return weighted / denom.view(B, 1, 1, 1)

        # ── hemis: concat(mean over avail, var over avail), each ld_enc → ld_dec
        denom = mask.to(embs.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)      # [B, 1]
        mean = (embs * m5).sum(dim=1) / denom.view(B, 1, 1, 1)                   # [B, ld_enc, H, W]
        diff = embs - mean.unsqueeze(1)
        var = (diff * diff * m5).sum(dim=1) / denom.view(B, 1, 1, 1)             # [B, ld_enc, H, W]
        return torch.cat([mean, var], dim=1)                                     # [B, 2*ld_enc=ld_dec, H, W]

    def _to_decoder_dim(self, emb: torch.Tensor) -> torch.Tensor:
        """
        For HeMIS: per-modality embedding has ld_enc channels; decoders expect ld_dec
        (= 2*ld_enc). Pad with zeros — matches Keras `zeros_for_var`.
        """
        if self.fusion != 'hemis':
            return emb
        return torch.cat([emb, torch.zeros_like(emb)], dim=1)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        img5 = self._ensure_5d(img)                           # [B, N, 1, H, W]
        B, N, _, H, W = img5.shape
        if N != self.num_mods:
            raise ValueError(f"Got {N} modalities, model was built for {self.num_mods}")

        # 1. Encode every modality (zero-impute missing ones first so all encoder
        #    params get gradients regardless of the mask — required for DDP).
        m5 = mask.to(dtype=img5.dtype, device=img5.device).view(B, N, 1, 1, 1)
        img_in = img5 * m5                                    # zero-out missing channels

        embs = torch.stack([self.encoders[i](img_in[:, i]) for i in range(N)], dim=1)
        # embs: [B, N, ld_enc, H, W]

        # 2. Fuse the latents, taking the mask into account.
        fused_emb = self._fuse(embs, mask)                    # [B, ld_dec, H, W]

        # 3. Decode each output modality from the fused latent and from each
        #    individual latent (so the loss can pull all latents to the same point).
        fused_outs = []
        for j in range(N):
            fused_outs.append(self.decoders[j](fused_emb))    # [B, 1, H, W]
        fused_pred = torch.stack(fused_outs, dim=1)           # [B, N, 1, H, W]

        ind_pred = []
        for i in range(N):
            emb_i = self._to_decoder_dim(embs[:, i])          # [B, ld_dec, H, W]
            row = []
            for j in range(N):
                row.append(self.decoders[j](emb_i))           # [B, 1, H, W]
            ind_pred.append(torch.stack(row, dim=1))          # [B, N, 1, H, W]
        ind_pred = torch.stack(ind_pred, dim=1)               # [B, N, N, 1, H, W]

        return {
            'fused': fused_pred,
            'ind': ind_pred,
            'emb': embs,
            'fused_emb': fused_emb,
            'mask': mask.to(device=img5.device),
        }

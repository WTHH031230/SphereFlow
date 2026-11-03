"""
M2DN: Multi-modal Modality-masked Diffusion Network

Paper: "Multi-Modal Modality-Masked Diffusion Network for Brain MRI Synthesis
        With Random Modality Missing" (TMI 2024)

Architecture:
  - Modality-specific encoders (ConvBlock in Fig. 3(c))
  - U-shaped multi-modal fusion block:
      · 3-stage encoding (ResBlock x2 + bilinear downsampling)
      · Channel attention on the deepest encoder / decoder stages
      · Binary modality mask injection by channel concatenation at the bottleneck
      · 3-stage decoding (ResBlock x2 + upsampling)
  - Modality-specific decoders (DeconvBlock in Fig. 3(d), keeping Sigmoid output per user request)
  - DDPM forward process (cosine schedule, noise only on missing modalities)
  - x0-prediction training objective
  - DDIM inference with configurable steps
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Helpers ────────────────────────────────────────────────────────────────

def _num_groups(ch: int, preferred: int = 8) -> int:
    """Return the largest valid GroupNorm group count ≤ preferred."""
    for g in [preferred, preferred // 2, 4, 2, 1]:
        if g > 0 and ch % g == 0:
            return g
    return 1


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding.  t: [B] (float/int) → [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) *
        torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # [B, half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # [B, dim]


# ─── Building Blocks ─────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding + 2-layer MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.dim))


class ResBlock(nn.Module):
    """
    Residual block: GroupNorm → ReLU → Conv × 2, with optional timestep injection.
    Paper: "GroupNorm+ReLU+3×3 Conv" operations with a skip connection.
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int = 0):
        super().__init__()
        g_in  = _num_groups(in_ch)
        g_out = _num_groups(out_ch)
        self.norm1 = nn.GroupNorm(g_in,  in_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # Out-of-place: the timestep embedding is shared across blocks.
        self.act   = nn.ReLU(inplace=False)
        self.t_proj = (
            nn.Sequential(
                nn.Linear(t_dim, out_ch),
                nn.SiLU(),
            ) if t_dim > 0 else None
        )
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor,
                t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        if self.t_proj is not None and t_emb is not None:
            h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention (paper cites [46])."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        return x * self.fc(x).view(B, C, 1, 1)


class ModalityEncoder(nn.Module):
    """
    Per-modality encoder: img [B,C,H,W] → feature [B,out_ch,H/2,W/2].
    Keep the encoder laid out like the paper's ConvBlock panel:
      Conv2D -> (GroupNorm + ReLU + Conv2D) -> (GroupNorm + ReLU + Conv2D/DownSample)
    The final downsampling is implemented by the last stride-2 convolution,
    which matches the paper text while keeping the block visually aligned
    with the network diagram.
    """

    def __init__(self, img_ch: int, out_ch: int):
        super().__init__()
        g = _num_groups(out_ch)
        self.conv_in = nn.Conv2d(img_ch, out_ch, 3, padding=1)
        self.norm_mid = nn.GroupNorm(g, out_ch)
        self.conv_mid = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm_down = nn.GroupNorm(g, out_ch)
        self.conv_down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)  # ↓ H/2
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.conv_mid(self.act(self.norm_mid(x)))
        x = self.conv_down(self.act(self.norm_down(x)))
        return x


class ModalityDecoder(nn.Module):
    """
    Per-modality decoder: feature [B,in_ch,H/2,W/2] → img [B,img_ch,H,W].
    Align with Fig. 3(d) DeconvBlock:
      (GroupNorm + ReLU + Conv2D) -> UpSample -> (GroupNorm + ReLU + Conv2D)
    The paper figure ends with Tanh; we keep Sigmoid here per user request.
    """

    def __init__(self, in_ch: int, img_ch: int):
        super().__init__()
        g = _num_groups(in_ch)
        self.norm1 = nn.GroupNorm(g, in_ch)
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, in_ch)
        self.conv2 = nn.Conv2d(in_ch, img_ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor,
                target_hw: Tuple[int, int]) -> torch.Tensor:
        x = self.conv1(self.act(self.norm1(x)))
        x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        x = self.conv2(self.act(self.norm2(x)))
        return torch.sigmoid(x)


class MultiModalFusionBlock(nn.Module):
    """
    U-shaped multi-modal fusion block (paper Section IV-A).

    Input:  [B, M*enc_ch, H/2, W/2]  (channel concat from modality-specific encoders)
    Output: [B, M*enc_ch, H/2, W/2]  (channel-detached into M partitions by caller)

    The figure shows a symmetric 3-stage encoder/decoder without long U-Net skip
    concatenations. Channel attention appears after the third encoder downsample
    and on the decoder side after the first upsampling stage.
    """

    def __init__(self, in_ch: int, num_modalities: int, t_dim: int):
        super().__init__()
        low_ch = max(1, in_ch // num_modalities)
        mid_ch = max(low_ch, in_ch // 2)

        self.enc1_a = ResBlock(in_ch,  in_ch,  t_dim)
        self.enc1_b = ResBlock(in_ch,  in_ch,  t_dim)
        self.enc2_a = ResBlock(in_ch,  mid_ch, t_dim)
        self.enc2_b = ResBlock(mid_ch, mid_ch, t_dim)
        self.enc3_a = ResBlock(mid_ch, low_ch, t_dim)
        self.enc3_b = ResBlock(low_ch, low_ch, t_dim)
        self.enc_attn = ChannelAttention(low_ch)

        self.dec3_a = ResBlock(low_ch + num_modalities, low_ch, t_dim)
        self.dec3_b = ResBlock(low_ch,                  low_ch, t_dim)
        self.dec_attn = ChannelAttention(low_ch)
        self.dec2_a = ResBlock(low_ch,  mid_ch, t_dim)
        self.dec2_b = ResBlock(mid_ch,  mid_ch, t_dim)
        self.dec1_a = ResBlock(mid_ch,  in_ch,  t_dim)
        self.dec1_b = ResBlock(in_ch,   in_ch,  t_dim)

    @staticmethod
    def _bilinear_downsample(x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        target_hw = (max(1, h // 2), max(1, w // 2))
        return F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)

    def forward(self,
                x: torch.Tensor,        # [B, in_ch, H, W]  (H = H_img/2)
                t_emb: torch.Tensor,    # [B, t_dim]
                mask_v: torch.Tensor,   # [B, M]  availability mask
                ) -> torch.Tensor:

        # ── Encoding ──────────────────────────────────────────────────────
        x1 = self.enc1_b(self.enc1_a(x, t_emb), t_emb)                    # [B, M*c,   H/2, W/2]
        x2 = self._bilinear_downsample(x1)                                # [B, M*c,   H/4, W/4]
        x2 = self.enc2_b(self.enc2_a(x2, t_emb), t_emb)                   # [B, M*c/2, H/4, W/4]
        x3 = self._bilinear_downsample(x2)                                # [B, M*c/2, H/8, W/8]
        x3 = self.enc3_b(self.enc3_a(x3, t_emb), t_emb)                   # [B, c,     H/8, W/8]
        bot = self._bilinear_downsample(x3)                               # [B, c,     H/16,W/16]
        bot = self.enc_attn(bot)

        # ── Modality-mask injection ─────────────────────────────────────
        Hb, Wb = bot.shape[2], bot.shape[3]
        # Paper convention for the bottleneck condition: 1 = missing, 0 = available.
        mask_sp = (1.0 - mask_v)[:, :, None, None].expand(-1, -1, Hb, Wb)  # [B,M,Hb,Wb]
        x = torch.cat([bot, mask_sp], dim=1)                           # [B, c+M, H/16, W/16]

        # ── Decoding ──────────────────────────────────────────────────────
        x = self.dec3_b(self.dec3_a(x, t_emb), t_emb)                     # [B, c,   H/16, W/16]
        x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)  # [B, c,   H/8,  W/8]
        x = self.dec_attn(x)
        x = self.dec2_b(self.dec2_a(x, t_emb), t_emb)                     # [B, M*c/2, H/8,  W/8]
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)  # [B, M*c/2, H/4, W/4]
        x = self.dec1_b(self.dec1_a(x, t_emb), t_emb)                     # [B, M*c,   H/4,  W/4]
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)  # [B, M*c,   H/2, W/2]
        return x


# ─── Main Model ──────────────────────────────────────────────────────────────

class M2DN(nn.Module):
    """
    Multi-modal Modality-masked Diffusion Network (M2DN).

    Args:
        num_modalities: M, number of imaging modalities (4 for BraTS, 2 for MRI-CT)
        img_channels:   C, channels per modality image (1 for grayscale MRI/CT)
        enc_channels:   output channels of each modality encoder (default 32)
        t_emb_dim:      timestep embedding dimension (default 256)
        T:              total diffusion timesteps for training (default 1000)
        n_inf_steps:    DDIM steps for inference (paper uses 18)

    Forward interface:
        x:            [B, M, 1, H, W]  (or [B, M, H, W] — auto-expanded)
        mask_indices: [B, M]  with 1=available, 0=missing
        mode:         'train' → returns (pred_x0 [B,M,1,H,W], mse_loss)
                      'inference' → returns pred_x0 [B,M,1,H,W]
    """

    def __init__(
        self,
        num_modalities: int = 4,
        img_channels:   int = 1,
        enc_channels:   int = 32,
        t_emb_dim:      int = 256,
        T:              int = 1000,
        n_inf_steps:    int = 18,
    ):
        super().__init__()
        self.M          = num_modalities
        self.C          = img_channels
        self.T          = T
        self.n_inf_steps = n_inf_steps
        self.enc_ch     = enc_channels

        # Timestep embedding
        self.t_emb = TimestepEmbedding(t_emb_dim)

        # Per-modality encoders / decoders
        self.encoders = nn.ModuleList([
            ModalityEncoder(img_channels, enc_channels)
            for _ in range(num_modalities)
        ])
        self.decoders = nn.ModuleList([
            ModalityDecoder(enc_channels, img_channels)
            for _ in range(num_modalities)
        ])

        # U-shaped fusion block
        self.fusion = MultiModalFusionBlock(
            in_ch=num_modalities * enc_channels,
            num_modalities=num_modalities,
            t_dim=t_emb_dim,
        )

        # Cosine noise schedule buffers
        self._register_schedule()

    # ── Noise Schedule ────────────────────────────────────────────────────────

    def _register_schedule(self):
        """Pre-compute cosine schedule buffers (alpha_bar, etc.)."""
        s = 0.008
        t_all = torch.arange(self.T + 1, dtype=torch.float64)
        f = torch.cos(((t_all / self.T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = (f / f[0]).float()                              # [T+1]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0, 0.999)
        alpha_bar_t = torch.cumprod(1 - betas, dim=0)              # [T]  (t=1..T)

        self.register_buffer('alpha_bar',             alpha_bar_t)
        self.register_buffer('sqrt_alpha_bar',        alpha_bar_t.sqrt())
        self.register_buffer('sqrt_one_minus_alpha_bar', (1 - alpha_bar_t).sqrt())

    # ── Core Denoising Network ────────────────────────────────────────────────

    def _denoise_net(self,
                     x_t:    torch.Tensor,   # [B, M, 1, H, W]
                     t:      torch.Tensor,   # [B]  1-indexed timestep
                     mask_v: torch.Tensor,   # [B, M]  1=avail, 0=missing
                     ) -> torch.Tensor:      # [B, M, 1, H, W]  predicted x0
        B, M, C, H, W = x_t.shape

        # Timestep embedding
        t_emb = self.t_emb((t - 1).float())    # 0-indexed for embedding

        # Per-modality encoding
        feats = [enc(x_t[:, i]) for i, enc in enumerate(self.encoders)]
        feats_cat = torch.cat(feats, dim=1)     # [B, M*enc_ch, H/2, W/2]

        # U-shaped fusion with modality-mask conditioning
        fused = self.fusion(feats_cat, t_emb, mask_v)   # [B, M*enc_ch, H/2, W/2]

        # Split into M partitions and decode each back to image resolution
        partitions = torch.chunk(fused, M, dim=1)        # M × [B, enc_ch, H/2, W/2]
        outputs = [dec(p, target_hw=(H, W))
                   for dec, p in zip(self.decoders, partitions)]   # M × [B,1,H,W]

        return torch.stack(outputs, dim=1)               # [B, M, 1, H, W]

    # ── Forward Diffusion ─────────────────────────────────────────────────────

    def q_sample(self,
                 x0:     torch.Tensor,   # [B, M, 1, H, W]
                 t:      torch.Tensor,   # [B]  1-indexed
                 mask_v: torch.Tensor,   # [B, M]  1=avail, 0=missing
                 noise:  Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:
        """
        Forward process: add noise only to missing modalities.
        Available modalities are left at their clean values.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        t_idx = (t - 1).long()
        sqab  = self.sqrt_alpha_bar[t_idx][:, None, None, None, None]
        sq1ab = self.sqrt_one_minus_alpha_bar[t_idx][:, None, None, None, None]
        x_noisy = sqab * x0 + sq1ab * noise

        # Keep available modalities clean
        avail = mask_v[:, :, None, None, None].bool().expand_as(x0)
        return torch.where(avail, x0, x_noisy)

    # ── Training Step ─────────────────────────────────────────────────────────

    def forward_train(self,
                      x:      torch.Tensor,   # [B, M, 1, H, W]
                      mask_v: torch.Tensor,   # [B, M]
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample random t, add noise to missing modalities, predict x0, compute MSE.
        Returns (pred_x0, loss).
        """
        B = x.size(0)
        t = torch.randint(1, self.T + 1, (B,), device=x.device)
        x_t = self.q_sample(x, t, mask_v)
        pred_x0 = self._denoise_net(x_t, t, mask_v)

        # MSE on ALL modalities (synthesis for missing + self-recon for available)
        loss = F.mse_loss(pred_x0, x)
        return pred_x0, loss

    # ── DDIM Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward_inference(self,
                          x:      torch.Tensor,   # [B, M, 1, H, W]
                          mask_v: torch.Tensor,   # [B, M]
                          ) -> torch.Tensor:      # [B, M, 1, H, W]
        """
        DDIM sampling with n_inf_steps steps.
        Available modalities are re-injected at every step to maintain fidelity.
        """
        B, M, C, H, W = x.shape
        dev = x.device

        # Available modality indicator
        avail = mask_v[:, :, None, None, None].bool().expand_as(x)

        # Initialise: available → clean x, missing → pure Gaussian noise
        x_t = torch.where(avail, x, torch.randn_like(x))

        # DDIM timestep sequence  T → 1  (n_inf_steps intervals)
        ts = torch.linspace(self.T, 1, self.n_inf_steps + 1).long().to(dev)

        for i in range(self.n_inf_steps):
            t_curr = ts[i]
            t_prev = ts[i + 1]

            t_batch = t_curr.expand(B)
            x0_pred = self._denoise_net(x_t, t_batch, mask_v).clamp(0.0, 1.0)

            if t_prev > 0:
                alpha_t      = self.alpha_bar[t_curr - 1]
                alpha_t_prev = self.alpha_bar[t_prev - 1]
                eps = (x_t - alpha_t.sqrt() * x0_pred) / (1 - alpha_t).sqrt().clamp(min=1e-8)
                x_t_prev = alpha_t_prev.sqrt() * x0_pred + (1 - alpha_t_prev).sqrt() * eps
            else:
                x_t_prev = x0_pred

            # Re-inject clean available modalities at every step
            x_t = torch.where(avail, x, x_t_prev)

        return x_t

    # ── Unified Forward ───────────────────────────────────────────────────────

    def forward(self,
                x,
                mask=None,
                mask_indices=None,
                mode: str = 'inference'):
        """
        Unified interface.

        Args:
            x:            [B, M, 1, H, W] or [B, M, H, W]
            mask:         [B, M]  1=available, 0=missing  (positional)
            mask_indices: same as mask, as a keyword argument
            mode:         'train' → (pred_x0, mse_loss)
                          'inference' → pred_x0
        """
        m = mask if mask is not None else mask_indices
        if m is None:
            raise ValueError("Provide mask or mask_indices.")
        if x.dim() == 4:          # [B, M, H, W] → [B, M, 1, H, W]
            x = x.unsqueeze(2)

        if mode == 'train':
            return self.forward_train(x, m)
        else:
            return self.forward_inference(x, m)

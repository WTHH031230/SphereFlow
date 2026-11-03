"""
Loss functions for MM-GAN (Sharma & Hamarneh, IEEE TMI 2020).

Generator total loss (paper Eq. 4, with lam = 0.9):
    L_G = (1 - lam) * L_GAN + lam * L_pixel
      L_GAN   : MSE(D(fake, real_x), 1)         (PatchGAN, LSGAN-style)
      L_pixel : L1(fake_missing, real_missing)  if `ic`, else L1(fake, real_x)
      L_synth : MSE(fake_missing, real_missing) — reported only, not back-propped.

Discriminator loss:
    L_D = 0.5 * (MSE(D(real_x, real_x), 1) + MSE(D(fake.detach(), real_x), label))
      label has 1 in the available patch-channels and 0 in the missing ones, so
      the available channels of `fake` (which equal the real ones under IC)
      should still be classified as real.

Conventions:
  fake_x / real_x : [B, N, H, W]    (model-native shape from MMGAN)
  mask            : [B, N]   1 = available / 0 = missing
  Discriminator output (PatchGAN): [B, N, H/16, W/16]
"""

import torch
import torch.nn as nn


class MMGANLoss(nn.Module):
    def __init__(self,
                 lam: float = 0.9,
                 ic: bool = True):
        super().__init__()
        self.lam = lam
        self.ic = ic

        # PatchGAN uses LSGAN (MSE); pixel loss is L1.
        self.criterion_gan = nn.MSELoss()
        self.criterion_pixel = nn.L1Loss()
        self.criterion_synth = nn.MSELoss()

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _real_label(pred: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(pred)

    @staticmethod
    def _patch_label_from_mask(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Build the per-patch ground-truth label tensor used for D's "fake" pass.

        pred : [B, N, h, w]  (PatchGAN logits)
        mask : [B, N]   1 = available (real) / 0 = missing (fake)
        returns : same shape as pred, broadcast from mask along (h, w).
        """
        B, N, h, w = pred.shape
        m = mask.to(device=pred.device, dtype=pred.dtype).view(B, N, 1, 1)
        return m.expand(B, N, h, w).contiguous()

    def _missing_indices(self, mask: torch.Tensor):
        """Indices where mask == 0, returned as (b_idx, n_idx) tensors."""
        return (mask < 0.5).nonzero(as_tuple=True)

    # ── Generator loss ───────────────────────────────────────────────────────

    def generator_loss(self,
                       fake_x: torch.Tensor,
                       real_x: torch.Tensor,
                       mask: torch.Tensor,
                       discriminator: nn.Module) -> dict:
        """
        fake_x        : [B, N, H, W]   generator output (with IC already applied)
        real_x        : [B, N, H, W]   ground-truth images
        mask          : [B, N]
        discriminator : conditional PatchGAN

        Returns dict {total, gan, pixel, synth}.
        """
        pred_fake = discriminator(fake_x, real_x)
        loss_gan = self.criterion_gan(pred_fake, self._real_label(pred_fake))

        if self.ic:
            b_idx, n_idx = self._missing_indices(mask)
            if b_idx.numel() == 0:
                # All modalities present: degrade gracefully to whole-image L1.
                loss_pixel = self.criterion_pixel(fake_x, real_x)
                loss_synth = self.criterion_synth(fake_x, real_x)
            else:
                fake_missing = fake_x[b_idx, n_idx]
                real_missing = real_x[b_idx, n_idx]
                loss_pixel = self.criterion_pixel(fake_missing, real_missing)
                loss_synth = self.criterion_synth(fake_missing, real_missing)
        else:
            loss_pixel = self.criterion_pixel(fake_x, real_x)
            loss_synth = self.criterion_synth(fake_x, real_x)

        total = (1.0 - self.lam) * loss_gan + self.lam * loss_pixel
        return {"total": total, "gan": loss_gan, "pixel": loss_pixel, "synth": loss_synth}

    # ── Discriminator loss ───────────────────────────────────────────────────

    def discriminator_loss(self,
                           fake_x: torch.Tensor,
                           real_x: torch.Tensor,
                           mask: torch.Tensor,
                           discriminator: nn.Module) -> dict:
        """
        fake_x : [B, N, H, W]   generator output, **detached by caller**
        real_x : [B, N, H, W]   ground-truth
        mask   : [B, N]
        """
        pred_real = discriminator(real_x, real_x)
        loss_real = self.criterion_gan(pred_real, self._real_label(pred_real))

        pred_fake = discriminator(fake_x, real_x)
        label_fake = self._patch_label_from_mask(pred_fake, mask)
        loss_fake = self.criterion_gan(pred_fake, label_fake)

        total = 0.5 * (loss_real + loss_fake)
        return {"total": total, "real": loss_real, "fake": loss_fake}

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "Call generator_loss() or discriminator_loss() directly."
        )

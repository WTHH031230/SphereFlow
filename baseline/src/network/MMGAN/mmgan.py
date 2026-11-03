"""
Sharma & Hamarneh, "Missing MRI Pulse Sequence Synthesis using Multi-Modal
Generative Adversarial Network" (MM-GAN), IEEE TMI 2020.

Ported from https://github.com/trane293/mm-gan (train_mmgan_brats2018.py, modules/advanced_gans/models.py).

Architecture:
  Generator
  └── U-Net (8 down / 7 up, InstanceNorm + LeakyReLU/ReLU, optional Tanh head).
      Input  : [B, N, H, W]   missing modalities are imputed (noise/zeros/avg).
      Output : [B, N, H, W]   reconstructions for all N modalities.
  Discriminator
  └── Conditional PatchGAN (concat(fake|real, real_x) → 4-channel patch logits).
      One discriminator instance handles all N modalities jointly.

Implicit Conditioning (IC, paper Section IV-B):
  After G produces ŷ, the available modalities in ŷ are overwritten with the real
  ones. Pixel/synthesis losses are then computed only on the missing channels.

Convention shared with the rest of this project:
  img  : [B, N, 1, H, W]   (or [B, N, H, W] — both accepted by MMGAN.forward)
  mask : [B, N]   float32, 1 = available, 0 = missing
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Weight init (mirrors mm-gan modules/advanced_gans/models.py)
# ─────────────────────────────────────────────────────────────────────────────

def weights_init_normal(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        if getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net building blocks
# ─────────────────────────────────────────────────────────────────────────────

class _UNetDown(nn.Module):
    def __init__(self, in_size: int, out_size: int, normalize: bool = True, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _UNetUp(nn.Module):
    def __init__(self, in_size: int, out_size: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


class GeneratorUNet(nn.Module):
    """U-Net generator from MM-GAN (8 downs / 7 ups + final upsample)."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 with_tanh: bool = False, with_relu: bool = True):
        super().__init__()

        self.down1 = _UNetDown(in_channels, 64, normalize=False)
        self.down2 = _UNetDown(64, 128)
        self.down3 = _UNetDown(128, 256)
        self.down4 = _UNetDown(256, 512, dropout=0.2)
        self.down5 = _UNetDown(512, 512, dropout=0.2)
        self.down6 = _UNetDown(512, 512, dropout=0.2)
        self.down7 = _UNetDown(512, 512, dropout=0.2)
        self.down8 = _UNetDown(512, 512, normalize=False, dropout=0.2)

        self.up1 = _UNetUp(512, 512, dropout=0.2)
        self.up2 = _UNetUp(1024, 512, dropout=0.2)
        self.up3 = _UNetUp(1024, 512, dropout=0.2)
        self.up4 = _UNetUp(1024, 512, dropout=0.2)
        self.up5 = _UNetUp(1024, 256)
        self.up6 = _UNetUp(512, 128)
        self.up7 = _UNetUp(256, 64)

        head_layers = [
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
        ]
        if with_tanh:
            head_layers.append(nn.Tanh())
        elif with_relu:
            head_layers.append(nn.ReLU())
        self.final = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


# ─────────────────────────────────────────────────────────────────────────────
# Conditional PatchGAN discriminator
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """Conditional PatchGAN: input is concat(img_A, img_B) along channels."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, dataset: str = "BRATS2018"):
        super().__init__()

        if "BRATS" in dataset:
            def block(in_filters, out_filters, normalization=True):
                layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
                if normalization:
                    layers.append(nn.InstanceNorm2d(out_filters))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(in_channels * 2, 64, normalization=False),
                *block(64, 128),
                *block(128, 256),
                *block(256, 512),
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(512, out_channels, 4, padding=1, bias=False),
            )
        else:
            # Lighter variant retained for parity with the upstream code (ISLES2015).
            def block(in_filters, out_filters, normalization=True):
                layers = [nn.Conv2d(in_filters, out_filters, 4, stride=4, padding=1)]
                if normalization:
                    layers.append(nn.InstanceNorm2d(out_filters))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(in_channels * 2, 8, normalization=False),
                *block(8, 8),
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(8, out_channels, 4, padding=1, bias=False),
            )

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat((img_A, img_B), dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Imputation helpers (paper Section IV-A: noise / zeros / average)
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    """Accept [B, N, H, W] or [B, N, 1, H, W] and return [B, N, H, W]."""
    if x.dim() == 5 and x.size(2) == 1:
        return x.squeeze(2)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected [B, N, H, W] or [B, N, 1, H, W], got {tuple(x.shape)}")


def impute_input(img: torch.Tensor, mask: torch.Tensor, z_type: str = "noise") -> torch.Tensor:
    """
    Build G's input by replacing missing-modality channels with an imputation tensor.

    img   : [B, N, H, W] (or [B, N, 1, H, W])
    mask  : [B, N]  1 = available / 0 = missing
    z_type: 'noise' | 'zeros' | 'average'   (matches train_mmgan_brats2018.py)
    """
    img4 = _ensure_4d(img)
    B, N, H, W = img4.shape
    out = img4.clone()
    mask_b = mask.to(device=img4.device, dtype=img4.dtype)  # [B, N]

    for b in range(B):
        avail_idx = (mask_b[b] >= 0.5).nonzero(as_tuple=True)[0]
        miss_idx = (mask_b[b] < 0.5).nonzero(as_tuple=True)[0]
        if miss_idx.numel() == 0:
            continue

        if z_type == "noise":
            impute = torch.randn((H, W), device=img4.device, dtype=img4.dtype)
        elif z_type == "zeros":
            impute = torch.zeros((H, W), device=img4.device, dtype=img4.dtype)
        elif z_type == "average":
            if avail_idx.numel() == 0:
                impute = torch.zeros((H, W), device=img4.device, dtype=img4.dtype)
            else:
                impute = img4[b, avail_idx].mean(dim=0)
        else:
            raise ValueError(f"Unknown z_type: {z_type!r}")

        out[b, miss_idx] = impute.unsqueeze(0).expand(miss_idx.numel(), -1, -1)

    return out


def impute_reals_into_fake(x_z: torch.Tensor, fake_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Implicit Conditioning (paper Section IV-B): overwrite fake outputs at the
    available channels with the corresponding real inputs.

    x_z    : G input  [B, N, H, W] — its available channels still hold the real images
    fake_x : G output [B, N, H, W]
    mask   : [B, N]   1 = available / 0 = missing
    """
    out = fake_x.clone()
    mask_b = mask.to(device=fake_x.device)
    avail = mask_b >= 0.5  # [B, N]

    if not avail.any():
        return out

    # Per-batch overwrite — broadcasted indexing keeps gradients elsewhere intact.
    # Under autocast `fake_x` may be fp16 while `x_z` stays fp32; align dtype/device
    # explicitly so the index-put doesn't crash with "Half vs Float" mismatch.
    idx_b, idx_n = avail.nonzero(as_tuple=True)
    src = x_z[idx_b, idx_n]
    if src.dtype != out.dtype or src.device != out.device:
        src = src.to(dtype=out.dtype, device=out.device)
    out[idx_b, idx_n] = src
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Top-level model
# ─────────────────────────────────────────────────────────────────────────────

class MMGAN(nn.Module):
    """
    Wraps the U-Net generator with imputation + (optional) implicit conditioning.

    Args
    ----
    num_mods   : number of modalities N (4 for BraTS, 2 for MRI/CT).
    z_type     : how to impute missing channels of G's input ('noise' | 'zeros' | 'average').
    use_ic     : enable implicit conditioning (overwrite fake's available channels with real).
    with_tanh  : final activation Tanh (set True when training in [-1, 1]).
    with_relu  : final activation ReLU (set True when training in [0, 1] / non-negative).

    Forward returns a 4-D tensor [B, N, H, W] (model-native shape). The trainer
    wraps it back to [B, N, 1, H, W] when the rest of the pipeline expects that.
    """

    def __init__(self,
                 num_mods: int = 4,
                 z_type: str = "noise",
                 use_ic: bool = True,
                 with_tanh: bool = False,
                 with_relu: bool = True):
        super().__init__()
        self.num_mods = num_mods
        self.z_type = z_type
        self.use_ic = use_ic
        self.generator = GeneratorUNet(
            in_channels=num_mods,
            out_channels=num_mods,
            with_tanh=with_tanh,
            with_relu=with_relu,
        )

    def impute(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return impute_input(img, mask, z_type=self.z_type)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        img  : [B, N, H, W] or [B, N, 1, H, W]
        mask : [B, N]
        returns : [B, N, H, W]
        """
        x_z = self.impute(img, mask)
        fake = self.generator(x_z)
        if self.use_ic:
            fake = impute_reals_into_fake(x_z, fake, mask)
        return fake

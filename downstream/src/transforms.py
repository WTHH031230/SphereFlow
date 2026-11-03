"""3D transforms for BraTS volumes.

Note: the preprocessed BraTS data already has each channel scaled to [0, 1]
(``clip(0, 4095) / 4095``). However, that is a uniform global rescaling and
does not equalise the intensity distributions across modalities or cases.

Following the original 3D-UNet recipe (and the standard BraTS pipeline,
e.g. nnU-Net), we additionally apply per-channel z-score normalisation
computed over the **non-zero** voxels (so that the brain mask, not the
zero-padded background, drives the statistics). This gives the network
a more balanced input distribution and dramatically improves convergence,
especially for the largest tumour region (WT)."""
from __future__ import annotations

import numpy as np
import torch


def _to_tensor(sample):
    """Convert ndarray volumes to float tensors."""
    out = {}
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(np.ascontiguousarray(v)).float()
        else:
            out[k] = v
    return out


class NormalizeNonzeroPerChannel:
    """Per-channel z-score normalisation using only non-zero voxels.

    Equivalent to MONAI's ``NormalizeIntensityd(nonzero=True,
    channel_wise=True)`` but implemented directly on numpy arrays so we don't
    pull a heavy MONAI transform pipeline.
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def __call__(self, sample):
        image = sample['image']  # (C, D, H, W)
        out = np.empty_like(image, dtype=np.float32)
        for c in range(image.shape[0]):
            x = image[c]
            mask = x > 0
            if mask.any():
                vals = x[mask]
                mean = vals.mean()
                std = vals.std()
                norm = (x - mean) / (std + self.eps)
                # keep background (originally zero) at 0 to preserve the brain mask
                norm[~mask] = 0.0
                out[c] = norm.astype(np.float32)
            else:
                out[c] = x.astype(np.float32)
        sample = {**sample, 'image': out}
        return sample


class RandFlip3D:
    """Random flip along given spatial axes (axis ids in (D, H, W) space)."""

    def __init__(self, prob: float = 0.5, axes=(1, 2)):
        self.prob = prob
        self.axes = axes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        for ax in self.axes:
            if np.random.rand() < self.prob:
                # image/label are (C, D, H, W); spatial axis offset by 1
                image = np.flip(image, axis=ax + 1)
                label = np.flip(label, axis=ax + 1)
        sample = {**sample, 'image': image.copy(), 'label': label.copy()}
        return sample


class RandIntensityShiftScale:
    """Per-channel random intensity shift + scale (image only).

    Applied to the *normalised* (z-score) volume; only the non-zero (brain)
    region is perturbed so that background stays at 0.
    """

    def __init__(self, shift: float = 0.1, scale: float = 0.1, prob: float = 0.5):
        self.shift = shift
        self.scale = scale
        self.prob = prob

    def __call__(self, sample):
        image = sample['image']
        if np.random.rand() < self.prob:
            c = image.shape[0]
            scale = (1.0 + np.random.uniform(-self.scale, self.scale, size=(c, 1, 1, 1))).astype(np.float32)
            shift = np.random.uniform(-self.shift, self.shift, size=(c, 1, 1, 1)).astype(np.float32)
            mask = (image != 0).astype(np.float32)  # preserve brain mask
            image = image * scale + shift
            image = image * mask
        sample = {**sample, 'image': image}
        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


train_transform = Compose([
    NormalizeNonzeroPerChannel(),
    RandFlip3D(prob=0.5, axes=(1, 2)),  # flip H, W (skip D since it's the depth axis)
    RandIntensityShiftScale(shift=0.1, scale=0.1, prob=0.5),
    _to_tensor,
])

val_transform = Compose([
    NormalizeNonzeroPerChannel(),
    _to_tensor,
])


# ---------------------------------------------------------------------------
# Raw [0, 1] pipelines: no z-score normalisation, spatial flips only.
# ---------------------------------------------------------------------------
train_transform_raw01 = Compose([
    RandFlip3D(prob=0.5, axes=(1, 2)),
    _to_tensor,
])

val_transform_raw01 = Compose([
    _to_tensor,
])

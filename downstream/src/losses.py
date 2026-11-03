"""Loss functions and evaluation metrics for multi-channel binary 3D
segmentation (BraTS WT/TC/ET)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt


class SoftDiceLoss(nn.Module):
    """Soft Dice over each output channel, averaged across channels and batch."""

    def __init__(self, smooth: float = 1e-5, sigmoid: bool = True):
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits) if self.sigmoid else logits
        # logits/target: (B, C, D, H, W)
        dims = tuple(range(2, prob.dim()))
        intersect = (prob * target).sum(dim=dims)
        denom = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersect + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combined soft-Dice + BCE-with-logits loss."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0,
                 smooth: float = 1e-5):
        super().__init__()
        self.dice = SoftDiceLoss(smooth=smooth, sigmoid=True)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        dice_loss = self.dice(logits, target)
        bce_loss = self.bce(logits, target)
        total = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return total, dice_loss.detach(), bce_loss.detach()


@torch.no_grad()
def per_channel_dice(prob: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5, smooth: float = 1e-5) -> torch.Tensor:
    """Hard binary Dice score per channel; returns shape (B, C)."""
    pred = (prob > threshold).float()
    dims = tuple(range(2, prob.dim()))
    intersect = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    # When both pred and target are empty, Dice is defined as 1.
    has_target = (target.sum(dim=dims) > 0).float()
    has_pred = (pred.sum(dim=dims) > 0).float()
    dice = (2.0 * intersect + smooth) / (denom + smooth)
    empty_both = (1.0 - has_target) * (1.0 - has_pred)
    dice = dice * (1.0 - empty_both) + 1.0 * empty_both
    return dice


@torch.no_grad()
def per_channel_iou(prob: torch.Tensor, target: torch.Tensor,
                    threshold: float = 0.5, smooth: float = 1e-5) -> torch.Tensor:
    """Hard binary IoU (Jaccard) score per channel; returns shape (B, C).

    When both pred and target are empty, IoU is defined as 1.
    """
    pred = (prob > threshold).float()
    dims = tuple(range(2, prob.dim()))
    intersect = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - intersect
    has_target = (target.sum(dim=dims) > 0).float()
    has_pred = (pred.sum(dim=dims) > 0).float()
    iou = (intersect + smooth) / (union + smooth)
    empty_both = (1.0 - has_target) * (1.0 - has_pred)
    iou = iou * (1.0 - empty_both) + 1.0 * empty_both
    return iou


@torch.no_grad()
def per_channel_sensitivity(prob: torch.Tensor, target: torch.Tensor,
                            threshold: float = 0.5,
                            smooth: float = 1e-5) -> torch.Tensor:
    """Voxel-wise Sensitivity (Recall, TPR) = TP / (TP + FN) per channel;
    shape (B, C).

    Tells what fraction of the ground-truth tumor voxels were captured —
    orthogonal to Dice/IoU which couple recall and precision. By convention,
    when the ground-truth region is empty (no positives), sensitivity is
    defined as 1 (no positives to miss).
    """
    pred = (prob > threshold).float()
    dims = tuple(range(2, prob.dim()))
    tp = (pred * target).sum(dim=dims)
    fn = ((1.0 - pred) * target).sum(dim=dims)
    has_target = (target.sum(dim=dims) > 0).float()
    sens = (tp + smooth) / (tp + fn + smooth)
    sens = sens * has_target + 1.0 * (1.0 - has_target)
    return sens


# ---------------------------------------------------------------------------
#  HD95 (Hausdorff 95-percentile) — runs on CPU/numpy.
# ---------------------------------------------------------------------------
def _hd95_one(pred: np.ndarray, target: np.ndarray,
              empty_penalty: float) -> float:
    """HD95 between two binary 3D masks (D, H, W).

    Cropped to the union bbox for speed, then a Euclidean distance transform
    of each complement is queried at the surface voxels of the other.
    """
    p_any = bool(pred.any())
    t_any = bool(target.any())
    if not p_any and not t_any:
        return 0.0
    if not p_any or not t_any:
        return float(empty_penalty)

    union = pred | target
    coords = np.argwhere(union)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    pad = 2
    mins = np.maximum(mins - pad, 0)
    maxs = np.minimum(maxs + pad, np.array(pred.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))
    p = pred[sl]
    t = target[sl]

    # Surface voxels = mask & ~erosion(mask). For volumes that are entirely
    # filled (rare), erosion may be empty; fall back to the mask itself.
    p_surf = p & ~binary_erosion(p)
    t_surf = t & ~binary_erosion(t)
    if not p_surf.any():
        p_surf = p
    if not t_surf.any():
        t_surf = t

    dt_to_p = distance_transform_edt(~p)
    dt_to_t = distance_transform_edt(~t)
    d_pt = dt_to_t[p_surf]
    d_tp = dt_to_p[t_surf]
    return float(max(np.percentile(d_pt, 95), np.percentile(d_tp, 95)))


@torch.no_grad()
def per_channel_hd95(prob: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5,
                     empty_penalty: float | None = None) -> torch.Tensor:
    """Per-channel HD95 (in voxel units); returns shape (B, C).

    Args:
        prob: (B, C, D, H, W) sigmoid probabilities.
        target: (B, C, D, H, W) binary {0, 1}.
        empty_penalty: value to assign when exactly one of pred/target is
            empty (the other is non-empty). Defaults to the volume diagonal
            so the metric remains finite & well-defined per BraTS conventions.
    """
    pred = (prob > threshold).cpu().numpy().astype(bool)
    tgt = (target > 0.5).cpu().numpy().astype(bool)
    B, C = pred.shape[:2]
    if empty_penalty is None:
        D, H, W = pred.shape[-3:]
        empty_penalty = math.sqrt(D * D + H * H + W * W)
    out = torch.zeros((B, C), dtype=torch.float32)
    for b in range(B):
        for c in range(C):
            out[b, c] = _hd95_one(pred[b, c], tgt[b, c], empty_penalty)
    return out

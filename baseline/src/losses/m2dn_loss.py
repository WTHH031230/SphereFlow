"""
M2DN Loss

The M2DN model already computes its MSE training loss internally inside
forward_train().  This module exposes a standalone M2DNLoss class for
interface consistency with the rest of the codebase and for any future
extensions (e.g. perceptual terms).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class M2DNLoss(nn.Module):
    """
    MSE reconstruction loss for M2DN.

    The M2DN paper (Eq. 9) defines:
        L_M2DN = || M - f_θ(M_t, t, v) ||²_2

    where M is the ground truth for ALL modalities (both missing and
    available), and f_θ is the denoising network output (x0 prediction).

    This class is kept thin because the main diffusion sampling logic
    (timestep sampling, forward-process noise addition) lives inside
    M2DN.forward_train().  The trainer can call:

        pred_x0, loss = model(x, mask_indices=mask, mode='train')

    and use `loss` directly, or it can call:

        loss = loss_fn(pred_x0, x_gt)

    for logging / additional terms.
    """

    def __init__(self, lambda_recon: float = 1.0):
        super().__init__()
        self.lambda_recon = lambda_recon

    def forward(self,
                pred_x0: torch.Tensor,   # [B, M, 1, H, W]  model prediction
                target:  torch.Tensor,   # [B, M, 1, H, W]  ground truth
                ) -> torch.Tensor:
        return self.lambda_recon * F.mse_loss(pred_x0, target)

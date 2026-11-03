"""
Loss for the Chartsias / Joyce multimodal-brain-synthesis model.

Two parts (matching the Keras reference):

1. Reconstruction (MAE per (input emb, output decoder) pair):
     L_recon = sum_j w_j * (
         (fuse_outs ?  L1(decoder_j(fused_emb), gt_j))           : 0)
       + (ind_outs  ? mean_i_avail L1(decoder_j(emb_i), gt_j))    : 0)
     )
   - Individual-emb terms only count modalities that are *available* for that
     sample (the Keras code always assumes every modality is available, but our
     pipeline trains with random missing-modality masks).

2. Embedding-distance ("em_concat" in the original):
     L_concat = mean over batch and pixels of Var_i(emb_i)
   computed only over the available embeddings of each sample. Encourages a
   modality-invariant latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMSynLoss(nn.Module):
    def __init__(self,
                 lambda_recon: float = 1.0,
                 lambda_concat: float = 1.0,
                 ind_outs: bool = True,
                 fuse_outs: bool = True):
        super().__init__()
        if not (ind_outs or fuse_outs):
            raise ValueError("At least one of ind_outs / fuse_outs must be True.")
        self.lambda_recon = lambda_recon
        self.lambda_concat = lambda_concat
        self.ind_outs = ind_outs
        self.fuse_outs = fuse_outs

    @staticmethod
    def _l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).abs().mean()

    def _ind_recon(self,
                   ind_pred: torch.Tensor,
                   gt: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """
        ind_pred : [B, N, N, 1, H, W]   ind[:, i, j] = decoder_j(emb_i)
        gt       : [B, N, 1, H, W]
        mask     : [B, N]               1 = available
        """
        B, N = mask.shape
        m = mask.to(dtype=ind_pred.dtype, device=ind_pred.device)             # [B, N]
        # Per-pixel absolute error for every (i, j) combo.
        # Broadcast gt over the source-emb axis i so we can subtract directly.
        err = (ind_pred - gt.unsqueeze(1)).abs()                               # [B, N, N, 1, H, W]
        # Reduce over spatial + channel → [B, N(i), N(j)]
        per_ij = err.mean(dim=(-1, -2, -3))                                    # [B, N, N]
        # Mask out source embeddings that weren't available for that sample.
        per_ij = per_ij * m.unsqueeze(-1)                                      # [B, N(i), N(j)]
        # Average over available source embeddings per sample.
        denom = m.sum(dim=1).clamp_min(1.0).view(B, 1)                         # [B, 1]
        per_j = per_ij.sum(dim=1) / denom                                      # [B, N(j)]
        return per_j.mean()                                                    # scalar

    def _fused_recon(self,
                     fused_pred: torch.Tensor,
                     gt: torch.Tensor) -> torch.Tensor:
        """
        fused_pred : [B, N, 1, H, W]   decoder_j(fused_emb)
        gt         : [B, N, 1, H, W]
        Average MAE over all modalities (each output slot is a target).
        """
        return (fused_pred - gt).abs().mean()

    def _concat_var(self,
                    emb: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """
        emb  : [B, N, C, H, W]   per-modality latents
        mask : [B, N]            1 = available

        Variance across the available embeddings. Returns scalar mean over batch
        and spatial positions.
        """
        B, N, C, H, W = emb.shape
        m = mask.to(dtype=emb.dtype, device=emb.device).view(B, N, 1, 1, 1)
        denom = mask.to(emb.dtype).sum(dim=1).clamp_min(1.0).view(B, 1, 1, 1)  # [B, 1, 1, 1]
        mean = (emb * m).sum(dim=1) / denom                                    # [B, C, H, W]
        diff = emb - mean.unsqueeze(1)
        var = (diff * diff * m).sum(dim=1) / denom                             # [B, C, H, W]

        # Only contribute where >= 2 modalities are available (variance is
        # undefined / trivially zero with a single sample).
        valid = (mask.to(emb.dtype).sum(dim=1) >= 2.0).view(B, 1, 1, 1).to(emb.dtype)
        if valid.sum() == 0:
            return var.new_zeros(())
        # Weighted mean: only over samples with >= 2 modalities.
        weight = valid.expand_as(var)
        return (var * weight).sum() / weight.sum().clamp_min(1.0)

    def forward(self,
                pred_dict: dict,
                gt: torch.Tensor,
                mask: torch.Tensor) -> dict:
        """
        pred_dict : output of MMSyn.forward
        gt        : [B, N, 1, H, W]   ground-truth, all modalities supplied
        mask      : [B, N]            1 = available, 0 = missing
        Returns dict {total, recon_fused, recon_ind, concat}.
        """
        if gt.dim() == 4:
            gt = gt.unsqueeze(2)

        recon_fused = pred_dict['fused'].new_zeros(())
        recon_ind = pred_dict['fused'].new_zeros(())

        if self.fuse_outs:
            recon_fused = self._fused_recon(pred_dict['fused'], gt)
        if self.ind_outs:
            recon_ind = self._ind_recon(pred_dict['ind'], gt, mask)

        recon = recon_fused + recon_ind
        concat = self._concat_var(pred_dict['emb'], mask)

        total = self.lambda_recon * recon + self.lambda_concat * concat

        return {
            "total": total,
            "recon_fused": recon_fused,
            "recon_ind": recon_ind,
            "concat": concat,
        }

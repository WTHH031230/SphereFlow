"""
Loss functions for Zhang et al. (IEEE TMI 2025).

Generator loss (eq. 10):
    L_Gen = λ1·L_syn + λ2·L_rec + λ3·L_adv
      L_syn : L1 on *missing* modality outputs  (λ1 = 100)
      L_rec : L1 on *available* modality outputs (λ2 =  30)
      L_adv : least-squares GAN loss on missing  (λ3 =   1)

Discriminator loss (eq. 11):
    L_Dis = Σ_{missing i} [ L2(Dis_i(Ŷ_i)) + L2(Dis_i(Y_i) − 1) ]
"""

import torch
import torch.nn as nn


class ZhangLoss(nn.Module):
    """
    Computes generator and discriminator losses for ZhangModel.

    Args:
        lambda_syn : weight for synthesis (missing modality L1) loss  (paper: 100)
        lambda_rec : weight for reconstruction (available modality L1) (paper:  30)
        lambda_adv : weight for adversarial loss                       (paper:   1)
    """

    def __init__(self,
                 lambda_syn: float = 100.0,
                 lambda_rec: float = 30.0,
                 lambda_adv: float = 1.0):
        super().__init__()
        self.lambda_syn = lambda_syn
        self.lambda_rec = lambda_rec
        self.lambda_adv = lambda_adv

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Mean over active entries only.

        value : arbitrary shape starting with batch dimension
        mask  : [B] or [B, ...] broadcastable to value
        """
        mask = mask.to(dtype=value.dtype, device=value.device)
        while mask.dim() < value.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(value)
        denom = mask.sum().clamp_min(1.0)
        return (value * mask).sum() / denom

    @staticmethod
    def _disc_bank(discriminators: nn.Module) -> nn.Module:
        """
        Unwrap a DDP-wrapped discriminator bank so per-modality indexing works in both
        single-GPU and multi-GPU training.
        """
        return discriminators.module if hasattr(discriminators, "module") else discriminators

    # ── Generator loss ────────────────────────────────────────────────────────

    def generator_loss(self,
                       pred:  torch.Tensor,
                       gt:    torch.Tensor,
                       ac:    torch.Tensor,
                       discriminators: nn.Module) -> dict:
        """
        pred           : [B, N, 1, H, W]  generator output
        gt             : [B, N, 1, H, W]  ground-truth images
        ac             : [B, N]  float32, 1=available / 0=missing
        discriminators : nn.ModuleList of N PatchDiscriminators

        Returns dict with keys: 'total', 'syn', 'rec', 'adv'
        """
        B, N = ac.shape
        disc_bank = self._disc_bank(discriminators)
        l_syn = pred.new_zeros(())
        l_rec = pred.new_zeros(())
        l_adv = pred.new_zeros(())

        for i in range(N):
            ac_i = ac[:, i]                             # [B]
            pred_i = pred[:, i]                         # [B, 1, H, W]
            gt_i   = gt[:, i]                           # [B, 1, H, W]
            missing_i = 1.0 - ac_i

            # L1 synthesis loss on missing modalities (eq. 7)
            l_syn = l_syn + self._masked_mean((pred_i - gt_i).abs(), missing_i)

            # L1 reconstruction loss on available modalities (eq. 8)
            l_rec = l_rec + self._masked_mean((pred_i - gt_i).abs(), ac_i)

            # Adversarial loss for missing modalities (eq. 9)
            disc_fake = disc_bank[i](pred_i)
            disc_real = disc_bank[i](gt_i)
            adv_term = (disc_fake - 1.0).pow(2) + disc_real.pow(2)
            l_adv = l_adv + self._masked_mean(adv_term, missing_i)

        total = (self.lambda_syn * l_syn
                 + self.lambda_rec * l_rec
                 + self.lambda_adv * l_adv)

        return {'total': total, 'syn': l_syn, 'rec': l_rec, 'adv': l_adv}

    # ── Discriminator loss ────────────────────────────────────────────────────

    def discriminator_loss(self,
                           pred:  torch.Tensor,
                           gt:    torch.Tensor,
                           ac:    torch.Tensor,
                           discriminators: nn.Module) -> torch.Tensor:
        """
        Least-squares discriminator loss (eq. 11) — only for missing modalities.

        pred           : [B, N, 1, H, W]  generator output (detached by caller)
        gt             : [B, N, 1, H, W]  ground-truth
        ac             : [B, N]  float32
        discriminators : nn.ModuleList of N PatchDiscriminators

        Returns scalar loss tensor.
        """
        B, N = ac.shape
        disc_bank = self._disc_bank(discriminators)
        l_dis = pred.new_zeros(())

        for i in range(N):
            ac_i = ac[:, i]                            # [B]
            pred_i = pred[:, i]                     # [B, 1, H, W]  (detached)
            gt_i   = gt[:, i]
            missing_i = 1.0 - ac_i

            # L_Dis += (1-ac_i) * [ L2(Dis(Ŷ_i)) + L2(Dis(Y_i) − 1) ]
            disc_fake = disc_bank[i](pred_i)
            disc_real = disc_bank[i](gt_i)
            dis_term = disc_fake.pow(2) + (disc_real - 1.0).pow(2)
            l_dis = l_dis + self._masked_mean(dis_term, missing_i)

        return l_dis

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "Call generator_loss() or discriminator_loss() directly."
        )

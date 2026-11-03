import torch
import torch.nn as nn
import torch.nn.functional as F


class ImputationLoss(nn.Module):
    def __init__(self, 
                 alpha_lmse, 
                 alpha_lcon, 
                 alpha_lconsis,
                 temp_con=0.8,      
                 temp_consis=0.9,
                 batch_size=32,
                 embed_dim=512,
                 device='cpu'):
        super().__init__()
        
        self.l_mse = nn.MSELoss()
        self.l_con = IntraModalityContrastLoss(batch_size, temp_con, device)
        self.l_consis = ModalityConsisLoss(batch_size, temp_consis, embed_dim, device)
        
        self.alpha_lmse = alpha_lmse
        self.alpha_lcon = alpha_lcon
        self.alpha_lconsis = alpha_lconsis
        
    def forward(self, x_rec, y_rec, x_con, y_con):
        l_mse = self.alpha_lmse * self.l_mse(x_rec, y_rec)
        l_consis = self.alpha_lconsis * self.l_consis(x_con, y_con)
        # Paper Eq(3): sim(r_i^spa, r_j^seq) -> anchor=spa, candidates=seq
        l_con = self.alpha_lcon * self.l_con(y_con, x_con)
        return l_mse, l_consis, l_con
            
        
class IntraModalityContrastLoss(torch.nn.Module):
    def __init__(self, batch_size=1, temperature=0.5, device='cpu'):
        super().__init__()
        # batch_size is kept in the signature for the callers; the contrast follows the paper's Eq. (3) within each sample.
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(device))

    def forward(self, x_i, x_j):
        # x_i, x_j: [B, M, D]
        b, m, d = x_i.size()

        # Paper Eq(3):
        # L = -log( exp(sim(r_i^spa, r_i^seq)/t) / sum_{j!=i} exp(sim(r_i^spa, r_j^seq)/t) )
        # Here: first arg is treated as anchor (spa), second arg as candidates (seq).
        anchor = F.normalize(x_i.float(), dim=-1)  # [B,M,D]
        cand = F.normalize(x_j.float(), dim=-1)    # [B,M,D]
        sim = torch.einsum("bmd,bnd->bmn", anchor, cand) / self.temp  # [B,M,M]
        sim_pos = torch.diagonal(sim, dim1=1, dim2=2)  # [B,M]
        num = torch.exp(sim_pos)
        eye = torch.eye(m, device=sim.device).unsqueeze(0)  # [1,M,M]
        # The denominator includes the positive term, so the probability stays <= 1.
        denom_off = torch.sum(torch.exp(sim) * (1.0 - eye), dim=2)  # [B,M]
        denom = denom_off + num  # [B,M]
        ratio = num / (denom + 1e-8)
        loss = -torch.log(ratio + 1e-8)

        return loss.mean()


class ModalityConsisLoss(nn.Module):
    def __init__(self, batch_size=1, temperature=0.5, embed_dim=512, device='cpu'):
        super().__init__()
        # Paper: use an MLP on concatenated pooled features to derive v_ij^{seq}, v_ij^{spa} in R^512
        # Use the SAME MLP for both branches (shared weights).
        self.projector = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        ).to(device)

        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(device))

    def _paper_contrast_combinations(self, anchor_bkd: torch.Tensor, cand_bkd: torch.Tensor):
        """
        Paper L_consis follows same definition as L_con but with temperature=0.9.
        Positive: same combination across branches; Negative: different combinations (within-sample).
        anchor_bkd/cand_bkd: [B, K, D], here K=3.
        Returns: loss_mean, ratio tensor [B,K]
        """
        b, k, d = anchor_bkd.size()
        anchor = F.normalize(anchor_bkd.float(), dim=-1)
        cand = F.normalize(cand_bkd.float(), dim=-1)
        sim = torch.einsum("bkd,bjd->bkj", anchor, cand) / self.temp  # [B,K,K]
        sim_pos = torch.diagonal(sim, dim1=1, dim2=2)  # [B,K]
        num = torch.exp(sim_pos)  # [B,K]
        eye = torch.eye(k, device=sim.device).unsqueeze(0)  # [1,K,K]
        denom_off = torch.sum(torch.exp(sim) * (1.0 - eye), dim=2)  # [B,K]
        denom = denom_off + num
        ratio = num / (denom + 1e-8)
        loss = -torch.log(ratio + 1e-8)  # [B,K] >= 0
        return loss.mean(), ratio, denom_off
    
    def forward(self, f_seq, f_spa):
        # f_seq, f_spa: [B, M, D]
        b, m, d = f_seq.size()
        
        if m < 4:
            # With fewer than 4 modalities (MRI <-> CT) the paper's combinations do not exist, so L_consis is 0.
            return torch.tensor(0.0, device=f_seq.device)

        # input modality: [t1c, t1, t2, t2f]
        # Paper combinations: (T1c,T2), (T1,T2), (Tf,T2)
        v_spa_t1c_t2 = self.projector(torch.cat([f_spa[:, 0], f_spa[:, 2]], dim=-1))  # [B,D]
        v_seq_t1c_t2 = self.projector(torch.cat([f_seq[:, 0], f_seq[:, 2]], dim=-1))  # [B,D]
        
        v_spa_t1_t2 = self.projector(torch.cat([f_spa[:, 1], f_spa[:, 2]], dim=-1))    # [B,D]
        v_seq_t1_t2 = self.projector(torch.cat([f_seq[:, 1], f_seq[:, 2]], dim=-1))    # [B,D]
        
        v_spa_t2_t2f = self.projector(torch.cat([f_spa[:, 3], f_spa[:, 2]], dim=-1))   # [B,D]
        v_seq_t2_t2f = self.projector(torch.cat([f_seq[:, 3], f_seq[:, 2]], dim=-1))   # [B,D]

        # Stack to [B,3,D] to match paper's "different modality combinations" negatives (within sample)
        v_spa = torch.stack([v_spa_t1c_t2, v_spa_t1_t2, v_spa_t2_t2f], dim=1)  # [B,3,D]
        v_seq = torch.stack([v_seq_t1c_t2, v_seq_t1_t2, v_seq_t2_t2f], dim=1)  # [B,3,D]

        loss_mean, ratio, denom_off = self._paper_contrast_combinations(v_spa, v_seq)

        return loss_mean
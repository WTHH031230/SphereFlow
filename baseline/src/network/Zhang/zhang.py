"""
Zhang et al., "Unified Multi-Modal Image Synthesis for Missing Modality Imputation"
IEEE Transactions on Medical Imaging, Vol. 44, No. 1, January 2025.
DOI: 10.1109/TMI.2024.3424785

Architecture:
  Generator
  ├── CDS-Encoder  (Commonality- & Discrepancy-Sensitive Encoder)
  │     ├── ESi  : 4 modality-specific encoding streams (individual weights for
  │     │          scales 0-2; *shared* weights across modalities for scales 3-4)
  │     ├── ESC  : 1 common encoding stream (shared across all modalities)
  │     └── Fuse : concat(ESi^s, ESC^s) → 1×1 Conv → F_i^(s)
  ├── DFUM         (Dynamic Feature Unification Module, 5 instances, one per scale)
  │     ├── Hard Integration : max over available modalities (eq. 1)
  │     └── Soft Integration : attention-weighted sum w/ softmax (eq. 2-3)
  └── Decoder      (4 modality-specific U-Net decoders with DFUM skip connections;
                    deeper blocks share weights across decoders)
  PatchDiscriminator  (70×70 PatchGAN, one per modality)

Input / output convention (matches the rest of this project):
  x   : [B, num_mods, 1, H, W]   missing modalities are zero-imputed by caller
  ac  : [B, num_mods]  float32,   1 = available, 0 = missing
  out : [B, num_mods, 1, H, W]   reconstructed / synthesized images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def zhang_to_model_range(x: torch.Tensor) -> torch.Tensor:
    """Map image intensities from [0, 1] to [-1, 1] for Zhang."""
    return x * 2.0 - 1.0


def zhang_to_image_range(x: torch.Tensor) -> torch.Tensor:
    """Map Zhang outputs from [-1, 1] back to [0, 1]."""
    return (x + 1.0) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Basic building block
# ─────────────────────────────────────────────────────────────────────────────

def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """3×3 Conv + InstanceNorm + ReLU  (called "CB" in the paper)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Attention Block  (Fig. 3c)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionBlock(nn.Module):
    """
    Paper-style multi-scale spatial attention using fixed 32-channel
    3×3 / 5×5 / 7×7 branches followed by a 1×1 projection to 1 channel.
    Input  : [B, C, H, W]
    Output : [B, 1, H, W]  sigmoid spatial attention map
    """

    MID_CH = 32

    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = nn.Conv2d(channels, self.MID_CH, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels, self.MID_CH, 5, 1, 2)
        self.conv7 = nn.Conv2d(channels, self.MID_CH, 7, 1, 3)
        self.fuse = nn.Conv2d(self.MID_CH * 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        return torch.sigmoid(self.fuse(f))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dynamic Feature Unification Module  (Fig. 3, Section III-B2)
# ─────────────────────────────────────────────────────────────────────────────

class DFUM(nn.Module):
    """
    Hard Integration  : element-wise max over available modalities (eq. 1)
    Soft Integration  : attention-weighted sum with softmax across modalities (eq. 2-3)
    Output            : Conv1×1( concat(F_hard, F_soft) )  (eq. 4)

    Degenerates gracefully to a self-attention module when only one modality
    is available (eqs. 5-6).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.attn = AttentionBlock(channels)
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, features: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
        """
        features : [B, N, C, H, W]
        ac       : [B, N]  float32
        returns  : [B, C, H, W]
        """
        B, N, C, H, W = features.shape
        outputs = []

        for b in range(B):
            available = ac[b] > 0.5
            avail_feats = features[b, available]  # [M, C, H, W]

            if avail_feats.shape[0] == 0:
                # The paper always keeps at least one modality available.
                # This zero fallback prevents crashes for accidental all-missing masks.
                f_hard = features.new_zeros(1, C, H, W)
                f_soft = features.new_zeros(1, C, H, W)
            elif avail_feats.shape[0] == 1:
                # Eq. (5)-(6): DFUM degenerates to identity + self-attention.
                f_hard = avail_feats
                attn_map = self.attn(avail_feats)     # [1, 1, H, W]
                f_soft = attn_map * avail_feats       # [1, C, H, W]
            else:
                # Eq. (1): hard integration over available modalities only.
                f_hard = avail_feats.max(dim=0, keepdim=True)[0]

                # Eq. (2)-(3): spatial attention weights are normalized across modalities.
                attn_map = self.attn(avail_feats)     # [M, 1, H, W]
                weights = F.softmax(attn_map, dim=0)  # [M, 1, H, W]
                f_soft = (weights * avail_feats).sum(dim=0, keepdim=True)

            fused = self.fuse(torch.cat([f_hard, f_soft], dim=1))  # [1, C, H, W]
            outputs.append(fused.squeeze(0))

        return torch.stack(outputs, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CDS-Encoder  (Fig. 2, Section III-B1)
# ─────────────────────────────────────────────────────────────────────────────

class CDSEncoder(nn.Module):
    """
    Fig. 2 style CDS-Encoder:
      • ESC has 4 stages: [64, 128, 256, 512]
      • ESi has 4 modality-specific stages with two fusion nodes per stage
      • the deepest 1024 stage is shared across modalities

    The first four outputs F_i^(1..4) are produced by the second fusion node of
    each stage. The deepest output F_i^(5) is produced by the shared 1024-stage
    after max-pooling F_i^(4), matching the paper figure.
    """

    STAGE_CH = (64, 128, 256, 512)
    BOTTLENECK_CH = 1024
    FUSE_CH = (64, 128, 256, 512, 1024)   # output channels of F_i^(s)

    def __init__(self, num_mods: int = 4):
        super().__init__()
        self.num_mods = num_mods
        self.pool = nn.MaxPool2d(2)

        # ── Common stream (ESC): 4 shared stages, each with two CB blocks ───
        self.esc_stages = nn.ModuleList()
        esc_in = 1
        for out_ch in self.STAGE_CH:
            self.esc_stages.append(nn.ModuleList([
                conv_block(esc_in, out_ch),
                conv_block(out_ch, out_ch),
            ]))
            esc_in = out_ch

        # ── Modality-specific ESi stages with per-stage double fusion ───────
        self.esi_stages = nn.ModuleList()
        self.fuse1 = nn.ModuleList()
        self.fuse2 = nn.ModuleList()
        for _ in range(num_mods):
            mod_stages = nn.ModuleList()
            mod_fuse1 = nn.ModuleList()
            mod_fuse2 = nn.ModuleList()
            esi_in = 1
            for out_ch in self.STAGE_CH:
                mod_stages.append(nn.ModuleList([
                    conv_block(esi_in, out_ch),
                    conv_block(out_ch, out_ch),
                ]))
                mod_fuse1.append(nn.Conv2d(out_ch * 2, out_ch, 1))
                mod_fuse2.append(nn.Conv2d(out_ch * 2, out_ch, 1))
                esi_in = out_ch
            self.esi_stages.append(mod_stages)
            self.fuse1.append(mod_fuse1)
            self.fuse2.append(mod_fuse2)

        # ── Deep shared bottleneck (red dotted box in Fig. 2) ───────────────
        self.shared_bottleneck = nn.Sequential(
            conv_block(self.STAGE_CH[-1], self.BOTTLENECK_CH),
            conv_block(self.BOTTLENECK_CH, self.BOTTLENECK_CH),
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fuse(fuse_layer: nn.Module, esi_feat: torch.Tensor, esc_feat: torch.Tensor) -> torch.Tensor:
        return fuse_layer(torch.cat([esi_feat, esc_feat], dim=1))

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> list:
        """
        x       : [B, num_mods, 1, H, W]  (missing modalities already zero-imputed)
        returns : list of 5 tensors, each [B, num_mods, C_s, H_s, W_s]
                  index 0 = finest (H×W), index 4 = coarsest (H/16 × W/16)
        """
        all_fused = [[] for _ in range(5)]

        for i in range(self.num_mods):
            xi = x[:, i]                        # [B, 1, H, W]
            esc_x = xi
            esi_x = xi

            # First four scales follow the paper figure: CB -> f -> CB -> f.
            for s, out_ch in enumerate(self.STAGE_CH):
                esc_stage = self.esc_stages[s]
                esi_stage = self.esi_stages[i][s]

                esc_1 = esc_stage[0](esc_x)
                esi_1 = esi_stage[0](esi_x)
                esi_mid = self._fuse(self.fuse1[i][s], esi_1, esc_1)

                esc_2 = esc_stage[1](esc_1)
                esi_2 = esi_stage[1](esi_mid)
                fused = self._fuse(self.fuse2[i][s], esi_2, esc_2)
                all_fused[s].append(fused)

                if s < len(self.STAGE_CH) - 1:
                    esc_x = self.pool(esc_2)
                    esi_x = self.pool(fused)
                else:
                    esi_x = self.pool(fused)

            # Deepest shared 1024 stage takes pooled F_i^(4) and has no ESC fusion.
            all_fused[4].append(self.shared_bottleneck(esi_x))

        # Stack into [B, num_mods, C_s, H_s, W_s]
        return [torch.stack(all_fused[s], dim=1) for s in range(5)]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Decoder stream  (DSi, Fig. 2)
# ─────────────────────────────────────────────────────────────────────────────

class _DecoderUpBlock(nn.Module):
    """Upsample → concat DFUM skip → CB → CB."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.body = nn.Sequential(
            conv_block(in_ch + skip_ch, out_ch),
            conv_block(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.body(torch.cat([self.up(x), skip], dim=1))


class DecoderStream(nn.Module):
    """
    Single modality decoding stream (DSi).

    Receives the 5 DFUM feature maps (finest-first) and decodes them to a
    single-channel image via U-Net-style upsampling + skip connections.

    Decoder channels (input → output per block):
      Block 0: 1024 + 512 = 1536 → 512   (scales 4 → 3)
      Block 1:  512 + 256 =  768 → 256   (scales 3 → 2)
      Block 2:  256 + 128 =  384 → 128   (scales 2 → 1)
      Block 3:  128 +  64 =  192 →  64   (scales 1 → 0)
      Head  :   64               →   1   (1×1 conv + Tanh)

    Parameter sharing (red dotted box for DSi in Fig. 2):
      Only the first high-level decoding block is shared across all streams.
      The remaining blocks and the head are modality-specific.
    """

    # DFUM output channels per scale (index 0 = finest)
    DFUM_CH = CDSEncoder.FUSE_CH     # (64, 128, 256, 512, 1024)

    def __init__(self,
                 shared_block: nn.Module,
                 own_blocks: nn.ModuleList,
                 head: nn.Module):
        """
        shared_block  : shared up-block across all decoder streams  (scale 4→3)
        own_blocks    : per-modality up-blocks                      (scales 3→2, 2→1, 1→0)
        head          : 1×1 conv + Tanh
        """
        super().__init__()
        self.shared = shared_block
        self.own = own_blocks
        self.head = head

    def forward(self, dfum_feats: list) -> torch.Tensor:
        """
        dfum_feats : list[5]  each [B, C_s, H_s, W_s], index 0 = finest
        returns    : [B, 1, H, W]
        """
        x = dfum_feats[4]                                 # coarsest [B, 1024, H/16, W/16]
        x = self.shared(x, dfum_feats[3])                 # → [B, 512, H/8,  W/8]
        x = self.own[0](x, dfum_feats[2])                 # → [B, 256, H/4,  W/4]
        x = self.own[1](x, dfum_feats[1])                 # → [B, 128, H/2,  W/2]
        x = self.own[2](x, dfum_feats[0])                 # → [B,  64, H,    W]
        return self.head(x)                               # → [B,   1, H,    W]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Generator
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):

    def __init__(self, num_mods: int = 4):
        super().__init__()
        self.num_mods = num_mods
        CH = CDSEncoder.FUSE_CH      # (64, 128, 256, 512, 1024)

        self.encoder = CDSEncoder(num_mods)
        self.dfums   = nn.ModuleList([DFUM(CH[s]) for s in range(5)])

        # ── Decoder: share the first 512-channel block as in Fig. 2 ───────
        shared_up = _DecoderUpBlock(CH[4], CH[3], CH[3])  # 1024+512 → 512
        self.decoders = nn.ModuleList()
        for _ in range(num_mods):
            own_up = nn.ModuleList([
                _DecoderUpBlock(CH[3], CH[2], CH[2]),    # 512+256 → 256
                _DecoderUpBlock(CH[2], CH[1], CH[1]),    # 256+128 → 128
                _DecoderUpBlock(CH[1], CH[0], CH[0]),    # 128+ 64 →  64
            ])
            head = nn.Sequential(
                nn.Conv2d(CH[0], 1, 1),
                nn.Tanh(),
            )
            self.decoders.append(DecoderStream(shared_up, own_up, head))

        # The shared_up blocks are used by all decoders; register them once so
        # they appear in self.parameters() and are moved with .to(device).
        self.shared_decoder_up = shared_up

    def forward(self, x: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
        """
        x  : [B, num_mods, 1, H, W]  (missing modalities zero-imputed)
        ac : [B, num_mods]  float32,  1=available / 0=missing
        returns : [B, num_mods, 1, H, W]
        """
        # ── Encode ────────────────────────────────────────────────────────
        enc_feats = self.encoder(x)          # list[5], each [B, N, C_s, H_s, W_s]

        # ── DFUM at each scale ────────────────────────────────────────────
        dfum_feats = [self.dfums[s](enc_feats[s], ac) for s in range(5)]
        # each element: [B, C_s, H_s, W_s]  (finest-first, index 0 = H×W)

        # ── Decode each modality ──────────────────────────────────────────
        outputs = [self.decoders[i](dfum_feats) for i in range(self.num_mods)]
        return torch.stack(outputs, dim=1)   # [B, N, 1, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PatchGAN Discriminator  (Section III-C)
# ─────────────────────────────────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator (Isola et al., Pix2Pix CVPR 2017).
    One instance per modality, optimised with a least-squares (L2) loss.
    """

    def __init__(self, in_ch: int = 1, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = ndf
        for _ in range(1, n_layers):
            layers += [
                nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(ch * 2, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch *= 2
        layers += [
            nn.Conv2d(ch, ch * 2, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ch * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 2, 1, 4, 1, 1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Top-level model  (generator + discriminator bank)
# ─────────────────────────────────────────────────────────────────────────────

class ZhangModel(nn.Module):
    """
    Wraps the generator and the bank of N discriminators.

    Generator forward is exposed directly; discriminators are accessed via
    self.discriminators[i] for training the adversarial objective.
    """

    def __init__(self, num_mods: int = 4):
        super().__init__()
        self.num_mods = num_mods
        self.generator      = Generator(num_mods)
        self.discriminators = nn.ModuleList([PatchDiscriminator() for _ in range(num_mods)])

    def forward(self, x: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
        """
        x  : [B, num_mods, 1, H, W]   (missing modalities zero-imputed by caller)
        ac : [B, num_mods]  float32
        returns : [B, num_mods, 1, H, W]
        """
        return self.generator(x, ac)

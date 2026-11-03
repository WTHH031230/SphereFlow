from monai.networks.nets.swin_unetr import get_window_size, SwinTransformerBlock, BasicLayer, compute_mask
from collections.abc import Sequence
from typing import Optional, Tuple
from einops import rearrange

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm


class MMT_Components(nn.Module):
    def __init__(self, num_contrasts=4, embed_dim=96):
        """
        Args:
            num_contrasts: Total number of unique MRI contrasts (e.g., T1, T2 -> 4).
            embed_dim: The feature dimension C used in the Transformer blocks.
        """
        super().__init__()
        
        # --- Definition of Contrast Encodings ---
        # Shape: (4, C)
        self.contrast_encodings = nn.Parameter(torch.randn(num_contrasts, embed_dim))

        # --- Definition of Contrast Queries ---
        # Shape: (4, C)
        self.contrast_queries = nn.Parameter(torch.randn(num_contrasts, embed_dim))

    def get_encoding(self, contrast_index):
        """
        Retrieve encoding for PRESENT modalities.
        Args:
            contrast_index: Binary mask (B, 4) where 1 means present.
        Returns:
            Tensor of shape (B, M_valid, C)
        """
        # Get batch size
        B = contrast_index.shape[0]
        
        # 1. Expand parameters to match batch size: (4, C) -> (B, 4, C)
        expanded_encodings = self.contrast_encodings.unsqueeze(0).expand(B, -1, -1)
        
        # 2. Create boolean mask for selection
        mask = (contrast_index == 1) # (B, 4)
        
        # 3. Select features. 
        # Note: tensor[mask] returns a flattened tensor of shape (Total_Valid, C)
        selected_tokens = expanded_encodings[mask]
        
        # 4. Reshape back to (B, M_valid, C)
        return selected_tokens.view(B, -1, self.contrast_encodings.shape[-1])

    def get_query(self, contrast_index):
        """
        Retrieve query for MISSING modalities (Target).
        Args:
            contrast_index: Binary mask (B, 4) where 0 means missing (target).
        Returns:
            Tensor of shape (B, M_missing, C)
        """
        B = contrast_index.shape[0]
        
        # 1. Expand parameters: (4, C) -> (B, 4, C)
        expanded_queries = self.contrast_queries.unsqueeze(0).expand(B, -1, -1)
        
        # 2. Create boolean mask for selection (Select where index is 0)
        mask = (contrast_index == 0) # (B, 4)
        
        # 3. Select features
        selected_tokens = expanded_queries[mask]
        
        # 4. Reshape back to (B, M_missing, C)
        return selected_tokens.view(B, -1, self.contrast_queries.shape[-1])


class MMTWindowAttention(nn.Module):
    """
    M-Swin Window Attention Block (Supports both Self and Cross Attention).
    Handles dynamic sequence lengths (variable M contrasts) and external identity injection.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],  # Spatial window size only (Wh, Ww)
        num_heads: int,
        is_cross_attention: bool = False, # Flag to switch between Self/Cross mode
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.is_cross_attention = is_cross_attention

        # --- Standard Swin Relative Position Bias Setup (Spatial only) ---
        # We define bias for spatial dims (Wh * Ww, Wh * Ww).
        # It will be tiled dynamically in forward to match (M * Wh * Ww).
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Generate relative coords (Standard Swin logic)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) 
        coords_flatten = torch.flatten(coords, 1) # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1) # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)

        # --- Project Layers Definitions ---
        if self.is_cross_attention:
            # For Cross-Attn: Query comes from Target, K/V come from Encoder
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            # For Self-Attn: Q/K/V all come from Input
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        x_encoder: Optional[torch.Tensor] = None,
        q_bias: Optional[torch.Tensor] = None, 
        k_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (B*num_windows, N, C). N = M * Wh * Ww.
            mask: Attention mask (num_windows, Wh*Ww, Wh*Ww) or None.
            x_encoder: Encoder output features (for Cross-Attn only).
            q_bias: Bias vector for Query injection (B*nW, N, C) or broadcastable.
            k_bias: Bias vector for Key injection (B*nW, N, C) or broadcastable.
        """
        B_, N, C = x.shape
        
        if self.is_cross_attention:
            # Cross-Attn: x is Query (Target), x_encoder is Key/Value (Source)
            assert x_encoder is not None, "x_encoder required for cross attention"
            q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            
            N_enc = x_encoder.shape[1]
            kv = self.kv(x_encoder).reshape(B_, N_enc, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            # Self-Attn: x is source for Q, K, V
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        
        if q_bias is not None:        
            q_bias_spatial = q_bias.unsqueeze(2).expand(-1, -1, self.window_size[0] * self.window_size[1], -1)
            q_bias_spatial = q_bias_spatial.reshape(q_bias.shape[0], -1, C) 

            B_bias = q_bias_spatial.shape[0]
            if B_ != B_bias:
                num_windows = B_ // B_bias
                q_bias_expanded = torch.repeat_interleave(q_bias_spatial, num_windows, dim=0)
            else:
                q_bias_expanded = q_bias_spatial

            q_bias_final = q_bias_expanded.view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = q + q_bias_final
            
        if k_bias is not None:
            k_bias_spatial = k_bias.unsqueeze(2).expand(-1, -1, self.window_size[0] * self.window_size[1], -1)
            k_bias_spatial = k_bias_spatial.reshape(k_bias.shape[0], -1, C)
            
            B_bias = k_bias_spatial.shape[0]
            if B_ != B_bias:
                num_windows = B_ // B_bias
                k_bias_expanded = torch.repeat_interleave(k_bias_spatial, num_windows, dim=0)
            else:
                k_bias_expanded = k_bias_spatial

            k_bias_final = k_bias_expanded.view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = k + k_bias_final

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B_, heads, N_q, N_k)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )  # (Wh*Ww, Wh*Ww, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, Wh*Ww, Wh*Ww)
        
        M_q = N // (self.window_size[0] * self.window_size[1])
        M_k = k.shape[2] // (self.window_size[0] * self.window_size[1])
        
        relative_position_bias = relative_position_bias.repeat(1, M_q, M_k) 
        
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, k.shape[2]) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, k.shape[2])

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MMTSwinTransformerBlock(SwinTransformerBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
        is_cross_attention: bool = False,
    ) -> None:

        super().__init__(
            dim, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, drop, attn_drop, drop_path,
            act_layer, norm_layer, use_checkpoint
        )
        self.attn = MMTWindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        if is_cross_attention:
            self.norm3 = norm_layer(dim)
            self.attn2 = MMTWindowAttention(
                dim,
                window_size=self.window_size,
                num_heads=num_heads,
                is_cross_attention=is_cross_attention,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            self.attn2 = None

    def window_partition(self, x, window_size):
        """window partition operation based on: "Liu et al.,
        Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
        <https://arxiv.org/abs/2103.14030>"
        https://github.com/microsoft/Swin-Transformer

        Args:
            x: input tensor.
            window_size: local window size.
        """
        b, m, h, w, c = x.shape
        x = x.view(b, m, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, m * window_size[0] * window_size[1], c)

        return windows

    def window_reverse(self, windows, window_size, dims):
        """window reverse operation based on: "Liu et al.,
        Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
        <https://arxiv.org/abs/2103.14030>"
        https://github.com/microsoft/Swin-Transformer

        Args:
            windows: windows tensor.
            window_size: local window size.
            dims: dimension values.
        """
        m = windows.shape[1]
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], m, window_size[0], window_size[1], -1)
        x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(b, m, h, w, -1)
        return x

    def forward_part1(self, x, mask_matrix, q_bias, k_bias):
        x = self.norm1(x)
        b, m, h, w, c = x.shape
        window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
        pad_l = pad_t = 0
        pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, _, hp, wp, _ = x.shape
        dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
            if mask_matrix is not None:
                attn_mask = mask_matrix.repeat(1, m, m)
            else:
                attn_mask = None
        else:
            shifted_x = x
            attn_mask = None
        x_windows = self.window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask, q_bias=q_bias, k_bias=k_bias)
        attn_windows = attn_windows.view(-1, *((m,) + window_size + (c,)))
        shifted_x = self.window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :h, :w, :].contiguous()

        return x

    def forward_part3(self, x, mask_matrix, x_encoder, q_bias, k_bias):
        x = self.norm1(x)
        b, m, h, w, c = x.shape
        n = x_encoder.shape[1]
        window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
        pad_l = pad_t = 0
        pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x_encoder = F.pad(x_encoder, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, _, hp, wp, _ = x.shape
        dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
            shifted_x_encoder = torch.roll(x_encoder, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
            if mask_matrix is not None:
                attn_mask = mask_matrix.repeat(1, m, n)
            else:
                attn_mask = None
        else:
            shifted_x = x
            shifted_x_encoder = x_encoder
            attn_mask = None
        x_windows = self.window_partition(shifted_x, window_size)
        x_encoder_windows = self.window_partition(shifted_x_encoder, window_size)
        attn_windows = self.attn2(x_windows, mask=attn_mask, x_encoder=x_encoder_windows, q_bias=q_bias, k_bias=k_bias)
        attn_windows = attn_windows.view(-1, *((m,) + window_size + (c,)))
        shifted_x = self.window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :h, :w, :].contiguous()

        return x

    def forward(self, x, mask_matrix, x_encoder=None, contrast_encodings=None, contrast_query=None):
        shortcut = x # B M H W C
        if self.attn2 is not None:
            p1_q_bias = contrast_query
            p1_k_bias = contrast_query
            p2_q_bias = contrast_query
            p2_k_bias = contrast_encodings
        else:
            p1_q_bias = contrast_encodings
            p1_k_bias = contrast_encodings
            p2_q_bias = None
            p2_k_bias = None
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, p1_q_bias, p1_k_bias, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix, p1_q_bias, p1_k_bias)
        x = shortcut + self.drop_path(x)
        if self.attn2 is not None:
            if self.use_checkpoint:
                x = x + checkpoint.checkpoint(self.forward_part3, x, mask_matrix, x_encoder, p2_q_bias, p2_k_bias, use_reentrant=False)
            else:
                x = x + self.forward_part3(x, mask_matrix, x_encoder, p2_q_bias, p2_k_bias)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x


class MMTBasicLayer(BasicLayer):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
        is_cross_attention: bool = False,
        num_mods: int = 4,
    ) -> None:

        super().__init__(
            dim, depth, num_heads, window_size, drop_path, mlp_ratio, qkv_bias, drop, attn_drop,
            norm_layer, downsample, use_checkpoint
        )
        self.is_cross_attention = is_cross_attention
        self.mmt_components = MMT_Components(embed_dim=dim, num_contrasts=num_mods)
        self.blocks = nn.ModuleList(
            [
                MMTSwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    is_cross_attention=is_cross_attention,
                )
                for i in range(depth)
            ]
        )
    
    def forward(self, x, mask_indices, x_encoder=None, reconstruction=False):
        contrast_encodings = self.mmt_components.get_encoding(mask_indices) # B, M, C
        if reconstruction:
            contrast_query = self.mmt_components.get_query(1 - mask_indices) # B, M, C
        else:
            contrast_query = self.mmt_components.get_query(mask_indices) # B, N, C
        x_shape = x.size()
        b, m, h, w, c = x_shape
        window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
        hp = int(np.ceil(h / window_size[0])) * window_size[0]
        wp = int(np.ceil(w / window_size[1])) * window_size[1]
        attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask, x_encoder, contrast_encodings, contrast_query)
        x = x.view(b, m, h, w, -1)
        return x
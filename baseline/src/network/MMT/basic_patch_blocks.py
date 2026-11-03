import torch
import torch.nn as nn

class PatchPartition(nn.Module):
    def __init__(self, in_channels=1, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Output dim is patch_size * patch_size * in_channels = 16 * C
        self.out_dim = patch_size * patch_size * in_channels

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, M, C, H, W)
        Returns:
            x: Tensor of shape (B, M, H/P, W/P, P*P*C)
        """
        # B: Batch, M: Modality, C: Channel, H: Height, W: Width
        B, M, C, H, W = x.shape
        
        # Check if H and W are divisible by patch_size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image size ({H}x{W}) must be divisible by patch size ({self.patch_size})."

        # Reshape to create patches
        # 1. View as (B, M, C, H//P, P, W//P, P)
        x = x.view(B, M, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        
        # 2. Permute to (B, M, H//P, W//P, P, P, C) to get spatial dims first
        x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
        
        # 3. Flatten the patch and channel dimensions: (B, M, H//P, W//P, P*P*C)
        x = x.view(B, M, H // self.patch_size, W // self.patch_size, -1)
        
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Input dim is C, concatenated 2x2 neighbors -> 4*C
        # Linear projection reduces 4*C to 2*C
        self.reduction = nn.Linear(4 * input_dim, 2 * input_dim, bias=False)
        self.layer_norm = nn.LayerNorm(4 * input_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, M, H, W, C)
        Returns:
            x: Tensor of shape (B, M, H/2, W/2, 2*C)
        """
        B, M, H, W, C = x.shape
        
        # Ensure spatial dimensions are even
        # (Padding logic omitted for brevity, assuming H, W are even)
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even."

        # Sampling 2x2 patches
        x0 = x[:, :, 0::2, 0::2, :]  # Top-Left
        x1 = x[:, :, 1::2, 0::2, :]  # Bottom-Left
        x2 = x[:, :, 0::2, 1::2, :]  # Top-Right
        x3 = x[:, :, 1::2, 1::2, :]  # Bottom-Right

        # Concatenate along the channel dimension -> (B, M, H/2, W/2, 4*C)
        x = torch.cat([x0, x1, x2, x3], -1)
        
        # Apply LayerNorm
        x = self.layer_norm(x)
        
        # Linear projection to reduce dimension -> (B, M, H/2, W/2, 2*C)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Linear layer doubles the dimension: C -> 2*C
        self.expand = nn.Linear(input_dim, 2 * input_dim, bias=False)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, M, H, W, C)
        Returns:
            x: Tensor of shape (B, M, H*2, W*2, C/2)
        """
        B, M, H, W, C = x.shape

        # 1. Expand feature dimension: (B, M, H, W, 2*C)
        x = self.expand(x)
        
        # 2. Reshape to split the channel dimension for spatial upscaling
        x = x.view(B, M, H, W, 2, 2, C // 2)
        
        # 3. Permute to place spatial blocks correctly: (B, M, H, 2, W, 2, C/2)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        
        # 4. Merge spatial dimensions: (B, M, H*2, W*2, C/2)
        x = x.view(B, M, H * 2, W * 2, C // 2)

        return x
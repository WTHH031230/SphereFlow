import torch
import torch.nn as nn
from einops import rearrange

def process_modality(x, mask_indices, mode='pad', num_mods=4):
    """
    Args:
        x: Input tensor. 
           - If mode='pad': (B, M_valid, C, H, W)
           - If mode='unpad'/'zero': (B, num_mods, C, H, W)
        mask_indices: (B, num_mods), values are 0 (missing) or 1 (present)
        mode: 'pad', 'unpad', or 'zero'
    """
    mask_bool = (mask_indices == 1) 
    
    b = x.shape[0]
    
    if mode == 'pad':
        # Input x: (B, M_valid, C, H, W) -> Compact form
        # Target: (B, num_mods, C, H, W) -> Sparse form
        _, m_valid, c, h, w = x.shape
        x_full = torch.zeros(b, num_mods, c, h, w).to(x.device, dtype=x.dtype)
        x_full[mask_bool] = x.reshape(-1, c, h, w)
        return x_full

    elif mode == 'unpad':
        # Input x: (B, num_mods, C, H, W)
        # Target: (B, M_valid, C, H, W)
        _, _, c, h, w = x.shape
        features = x[mask_bool] 
        m_valid = features.shape[0] // b
        return features.reshape(b, m_valid, c, h, w)

    elif mode == 'zero':
        mask_expand = mask_indices.view(b, num_mods, 1, 1, 1)
        return x * mask_expand.to(dtype=x.dtype)
        
    return x


class ResBlock(nn.Module):
    def __init__(self, dim, num_mods=4):
        super().__init__()
        self.num_mods = num_mods
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(num_mods * dim, num_mods * dim, (5, 5), (1, 1), (2, 2), groups=num_mods),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
            nn.Conv2d(num_mods * dim, num_mods * dim, (5, 5), (1, 1), (2, 2), groups=num_mods)
        )

    def forward(self, x, mask_indices):
        assert len(x.shape) == 5 # B, num_mods, C, H, W
        x = process_modality(x, mask_indices, 'zero', self.num_mods)
        b = x.shape[0]
        x = rearrange(x, 'b m c h w -> b (m c) h w') 
        x = self.conv5x5(x) + x
        x = rearrange(x, 'b (m c) h w -> b m c h w', b=b, m=self.num_mods)
        x = process_modality(x, mask_indices, 'zero', self.num_mods)
        return x


class ImageEncodingBlock(nn.Module):
    def __init__(self, dim, num_mods=4):
        super().__init__()
        self.num_mods = num_mods
        self.conv7x7 = nn.Conv2d(num_mods, num_mods * dim, (7, 7), (1, 1), (3, 3), groups=num_mods)
        self.res_conv5x5 = nn.ModuleList([
            ResBlock(dim, num_mods),
            ResBlock(dim, num_mods)
        ])
    
    def forward(self, x, mask_indices):
        assert len(x.shape) == 5 # B, M, C, H, W
        b = x.shape[0]
        x = process_modality(x, mask_indices, 'pad', self.num_mods) # B, M, C, H, W -> B, num_mods, C, H, W
        x = rearrange(x, 'b m c h w -> b (m c) h w') 
        x = self.conv7x7(x)
        x = rearrange(x, 'b (m c) h w -> b m c h w', b=b, m=self.num_mods)
        for block in self.res_conv5x5:
            x = block(x, mask_indices)
        x = process_modality(x, mask_indices, 'unpad', self.num_mods)
        return x


class ImageDecodingBlock(nn.Module):
    def __init__(self, dim, num_mods=4):
        super().__init__()
        self.num_mods = num_mods
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(num_mods * dim, num_mods, (7, 7), (1, 1), (3, 3), groups=num_mods),
            # nn.LeakyReLU(0.2, inplace=True)
            nn.ReLU()
        )
        self.res_conv5x5 = nn.ModuleList([
            ResBlock(dim, num_mods),
            ResBlock(dim, num_mods)
        ])
    
    def forward(self, x, mask_indices):
        assert len(x.shape) == 5 # B, M, C, H, W
        b = x.shape[0]
        x = process_modality(x, mask_indices, 'pad', self.num_mods) # B, M, C, H, W -> B, num_mods, C, H, W
        for block in self.res_conv5x5:
            x = block(x, mask_indices)
        x = rearrange(x, 'b m c h w -> b (m c) h w') 
        x = self.conv7x7(x)
        x = rearrange(x, 'b (m c) h w -> b m c h w', b=b, m=self.num_mods)
        x = process_modality(x, mask_indices, 'unpad', self.num_mods)
        return x



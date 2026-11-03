from network.MMT.basic_cnn_blocks import ImageDecodingBlock
from network.MMT.basic_swin_blocks import MMTBasicLayer
from network.MMT.basic_patch_blocks import PatchExpanding
import torch.nn as nn
from network.MMT.basic_swin_blocks import MMT_Components
import torch


class MMTDecoder(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        dim=6, 
        depths=[2, 2, 2, 2], 
        num_heads=[3, 6, 12, 24], 
        window_size=(8, 8),
        num_mods=4
    ):
        """
        MMT Decoder: Decodes features using Contrast Queries and Cross-Attention. [cite: 313]
        """
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = 16 * dim
        self.stem_dim = dim
        self.patch_size = patch_size

        # Initialize queries with deepest dimension
        deepest_dim = int(self.embed_dim * (2 ** (self.num_layers - 1)))
        self.mmt_components = MMT_Components(embed_dim=deepest_dim, num_contrasts=num_mods)

        self.layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # Build Layers (Deep -> Shallow): Stage 3 -> 0
        for i_layer in range(self.num_layers - 1, -1, -1):
            dim = int(self.embed_dim * (2 ** i_layer))
            
            layer = MMTBasicLayer(
                dim=dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                drop_path=[0.0] * depths[i_layer],
                is_cross_attention=True, # Cross-Attend to Encoder
                num_mods=num_mods
            )
            self.layers.append(layer)

            if i_layer > 0:
                self.upsamples.append(PatchExpanding(input_dim=dim))
        
        patch_dim = (patch_size ** 2) * self.stem_dim
        self.final_projection = nn.Linear(self.embed_dim, patch_dim)
        self.img_decode = ImageDecodingBlock(self.stem_dim, num_mods=num_mods)

    def forward(self, encoder_features, mask_indices, reconstruction=False):
        """
        encoder_features: [Stage0, Stage1, Stage2, Stage3] (from Encoder Up-path)
        """
        # Get Queries for MISSING modalities
        if reconstruction:
            contrast_queries = self.mmt_components.get_query(1 - mask_indices)
        else:
            contrast_queries = self.mmt_components.get_query(mask_indices)
        
        # Expand to deepest resolution
        bottleneck_feat = encoder_features[0]
        H_deep, W_deep = bottleneck_feat.shape[2], bottleneck_feat.shape[3]
        x = contrast_queries.unsqueeze(2).unsqueeze(3).expand(-1, -1, H_deep, W_deep, -1)
        
        # Decode
        for i, layer in enumerate(self.layers):
            # Match resolution: Layer 0 (Deep) uses encoder_features[3]
            enc_feat = encoder_features[i]
            
            x = layer(x, mask_indices, x_encoder=enc_feat, reconstruction=reconstruction)
            
            if i < len(self.upsamples):
                x = self.upsamples[i](x)

        x = self.final_projection(x)
        B, M_miss, H_p, W_p, _ = x.shape
        x = x.view(B, M_miss, H_p, W_p, self.patch_size, self.patch_size, self.stem_dim)
        x = x.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
        x = x.reshape(B, M_miss, self.stem_dim, H_p * self.patch_size, W_p * self.patch_size)
        if reconstruction:
            x = self.img_decode(x, mask_indices)
        else:
            x = self.img_decode(x, 1 - mask_indices)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing MMTDecoder on device: {device}")

    img_size = (256, 256)
    patch_size = 4
    dim = 6 
    depths = [2, 2, 2, 2]
    num_heads = [3, 6, 12, 24]
    window_size = (8, 8)

    decoder = MMTDecoder(
        patch_size=patch_size,
        dim=dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size
    ).to(device)

    total_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Decoder Total Parameters: {total_params / 1e6:.2f}M")

    B = 2
    encoder_features = []
    resolutions = [8, 16, 32, 64] # 1/32, 1/16, 1/8, 1/4
    
    embed_dim_base = 16 * dim # 96
    
    for i in range(4):
        # Stage index: 3, 2, 1, 0
        stage_idx = 3 - i
        current_dim = int(embed_dim_base * (2 ** stage_idx))
        res = resolutions[i]
        
        M_valid = 3
        feat = torch.randn(B, M_valid, res, res, current_dim).to(device)
        encoder_features.append(feat)
        print(f"Dummy Encoder Feature [{i}] shape: {feat.shape}")

    mask_indices = torch.tensor([
        [1, 1, 1, 0], 
        [1, 1, 0, 1]
    ], dtype=torch.float32).to(device)

    print("\n--- Running Decoder Forward Pass ---")
    try:
        
        output = decoder(encoder_features, mask_indices)
        
        print(f"Output Shape: {output.shape}")
        
        expected_shape = (B, 1, 1, 256, 256)
        
        if output.shape == expected_shape:
            print("Verification Passed: Output shape matches expected dimensions.")
        else:
            print(f"Verification Failed: Expected {expected_shape}, got {output.shape}")
            
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
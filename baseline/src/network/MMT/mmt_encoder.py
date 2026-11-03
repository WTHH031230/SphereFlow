import torch
import torch.nn as nn
from network.MMT.basic_swin_blocks import MMTBasicLayer
from network.MMT.basic_patch_blocks import PatchPartition, PatchMerging, PatchExpanding
from network.MMT.basic_cnn_blocks import ImageEncodingBlock

class MMTEncoder(nn.Module):
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
        MMT Encoder composed of CNN Stem, Patch Partition, and Swin Transformer stages.
        
        Args:
            img_size: Input image resolution (H, W).
            patch_size: Patch size for partition (P).
            in_chans: Number of input channels per modality (usually 1).
            dim: Feature dimension output by the CNN Stem.
            depths: Number of blocks in each Swin stage.
            num_heads: Number of attention heads in each Swin stage.
            window_size: Window size for local attention.
        """
        super().__init__()
        self.num_layers = len(depths)
        embed_dim = 16 * dim
        self.embed_dim = embed_dim
        
        # --- 1. CNN Stem ---
        # Extracts initial features from raw images
        self.stem = ImageEncodingBlock(dim=dim, num_mods=num_mods)
        
        # --- 2. Patch Partition & Linear Embedding ---
        # Converts spatial features (H, W, stem_dim) -> tokens (H/P, W/P, embed_dim)
        self.patch_partition = PatchPartition(in_channels=dim, patch_size=patch_size)
        
        # Project patch features to transformer embedding dimension
        patch_dim = (patch_size ** 2) * dim
        self.linear_embedding = nn.Linear(patch_dim, embed_dim)
        self.pos_drop = nn.Dropout(p=0.0)

        # --- 3. Swin Transformer Stages (Encoder Layers) ---
        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        self.layers2 = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        self.concat_linears = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            # MMT Basic Layer handles the swin blocks and contrast encoding injection
            layer = MMTBasicLayer(
                dim=int(embed_dim * (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                drop_path=[0.0] * depths[i_layer], # Simplified drop path
                is_cross_attention=False, # Encoder uses Self-Attention
                num_mods=num_mods
            )
            self.layers.append(layer)

            j_layer = self.num_layers - i_layer - 1
            layer2 = MMTBasicLayer(
                dim=int(embed_dim * (2 ** j_layer)),
                depth=depths[j_layer],
                num_heads=num_heads[j_layer],
                window_size=window_size,
                drop_path=[0.0] * depths[j_layer], # Simplified drop path
                is_cross_attention=False, # Encoder uses Self-Attention
                num_mods=num_mods
            )
            self.layers2.append(layer2)
            
            # Add Patch Merging for all stages except the last one
            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(input_dim=int(embed_dim * (2 ** i_layer)))
                self.downsamples.append(downsample)

                upsample = PatchExpanding(input_dim=int(embed_dim * (2 ** j_layer)))
                self.upsamples.append(upsample)

                concat_linear = nn.Linear(2 * int(embed_dim * (2 ** (j_layer - 1))), int(embed_dim * (2 ** (j_layer - 1))))
                self.concat_linears.append(concat_linear)

    def forward(self, x, mask_indices):
        """
        Args:
            x: Input tensor (B, M, C, H, W)
            mask_indices: Modality availability mask (B, 4)
        Returns:
            features: List of tensors from each stage [stage1, stage2, stage3, stage4]
                      Used for skip connections in the decoder.
        """
        # 1. Forward through CNN Stem
        # Output shape: (B, M, stem_dim, H, W)
        x = self.stem(x, mask_indices)
        
        # 2. Patch Partition
        # Output shape: (B, M, H/4, W/4, 16*stem_dim)
        x = self.patch_partition(x)
        
        # 3. Linear Embedding
        # Output shape: (B, M, H/4, W/4, embed_dim)
        x = self.linear_embedding(x)
        x = self.pos_drop(x)

        features = []

        out_features = []
        
        # 4. Forward through Swin Stages
        for i, layer in enumerate(self.layers):
            # Apply Transformer Blocks (MMTBasicLayer handles mask_indices internally)
            # x shape: (B, M, H_i, W_i, C_i)
            x = layer(x, mask_indices)
            
            # Store features for Skip Connections
            features.append(x)
            
            # Apply Downsampling (PatchMerging) if not the last stage
            if i < self.num_layers - 1:
                x = self.downsamples[i](x)

        features = features[::-1]

        # x: B, M, H // 32, W // 32, 128C
        for i, layer in enumerate(self.layers2):
            if i:
                x = torch.cat([x, features[i]], dim=-1)
                x = self.concat_linears[i - 1](x)
            x = layer(x, mask_indices)
            out_features.append(x)
            if i < self.num_layers - 1:
                x = self.upsamples[i](x)
        return out_features


if __name__ == "__main__":
    # --- Configuration for Testing ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing MMTEncoder on device: {device}")

    # Define model hyperparameters
    img_size = (256, 256)
    patch_size = 4
    depths = [2, 2, 2, 2]
    num_heads = [3, 6, 12, 24]

    # --- Instantiate the Encoder ---
    model = MMTEncoder(
        patch_size=patch_size,
        dim=6,
        depths=depths,
        num_heads=num_heads,
        window_size=(8, 8)
    ).to(device)

    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params / 1e6:.2f}M")

    # --- Generate Dummy Input Data ---
    # Shape: (Batch, Modality, Channel, Height, Width)
    B, M, C, H, W = 2, 3, 1, 256, 256
    x = torch.randn(B, M, C, H, W).to(device)

    # Generate Dummy Mask (Batch, 4)
    # 1 indicates present, 0 indicates missing
    mask_indices = torch.tensor([
        [0, 1, 1, 1],  # Sample 1: All modalities present
        [1, 1, 1, 0]   # Sample 2: Modalities 2 and 4 missing
    ], dtype=torch.float32).to(device)

    # --- Run Forward Pass ---
    print("\n--- Running Forward Pass ---")
    try:
        features = model(x, mask_indices)
        
        # --- Verification ---
        print(f"Input Shape: {x.shape}")
        print(f"Output Features List Length: {len(features)}")
        
        # Expected downsampling factors for each stage relative to input H, W
        # Note: Swin outputs shape (B, M, H_new, W_new, C_new)
        expected_resolutions = [32, 16, 8, 4] 
        
        for i, feat in enumerate(features):
            scale = expected_resolutions[i]
            expected_dim = 6 * 16 * (2 ** (len(features) - i - 1))
            curr_h, curr_w = H // scale, W // scale
            
            print(f"Stage {i+1} Output Shape: {feat.shape}")
            
            # Check dimensions
            assert feat.shape[0] == B, f"Batch size mismatch at stage {i}"
            assert feat.shape[1] == M, f"Modality count mismatch at stage {i}"
            assert feat.shape[2] == curr_h and feat.shape[3] == curr_w, f"Spatial resolution mismatch at stage {i}"
            assert feat.shape[4] == expected_dim, f"Channel dimension mismatch at stage {i}"

        print("\n✅ MMTEncoder structure verification passed successfully.")

    except Exception as e:
        print(f"\nError during verification: {e}")
        import traceback
        traceback.print_exc()
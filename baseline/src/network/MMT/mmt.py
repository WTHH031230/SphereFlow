from network.MMT.mmt_encoder import MMTEncoder
from network.MMT.mmt_decoder import MMTDecoder
import torch
import torch.nn as nn
from network.MMT.basic_cnn_blocks import process_modality


class MMT(nn.Module):
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
        MMT (Multi-modal Multi-scale Transformer) for Missing Modality Synthesis.
        Combines MMTEncoder and MMTDecoder into a unified architecture.
        
        Args:
            patch_size: Patch size for partitioning.
            dim: Base feature dimension (C).
            depths: Depth of Swin Transformer layers.
            num_heads: Number of attention heads.
            window_size: Window size for local attention.
        """
        super().__init__()
        
        # --- Encoder: Extract multi-scale features from available modalities ---
        self.encoder = MMTEncoder(
            patch_size=patch_size,
            dim=dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            num_mods=num_mods
        )

        # --- Decoder: Synthesize missing modalities using encoder features ---
        self.decoder = MMTDecoder(
            patch_size=patch_size,
            dim=dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            num_mods=num_mods
        )
        self.num_mods = num_mods

    def forward(self, x, mask_indices):
        """
        Args:
            x: Ground Truth Images (B, 4, C, H, W).
            mask_indices: Binary mask indicating availability (B, 4). 
                          1 for available, 0 for missing.
        
        Returns:
            y_gen: Synthesized images for the MISSING modalities.
                     Shape depends on decoder implementation, usually corresponds to missing slots.
        """
        x = process_modality(x, mask_indices, 'unpad', self.num_mods)
        encoder_features = self.encoder(x, mask_indices)
        y_gen = self.decoder(encoder_features, mask_indices, False)
        if self.training:
            x_recon = self.decoder(encoder_features, mask_indices, True)
            return y_gen, x_recon
        else:
            return y_gen

if __name__ == "__main__":
    # --- Configuration for Testing ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing MMT Full Model on device: {device}")

    # Hyperparameters
    config = {
        "patch_size": 4,
        "dim": 6,
        "depths": [2, 2, 2, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": (8, 8)
    }

    # Initialize Model
    model = MMT(**config).to(device)
    
    # Parameter Count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total MMT Parameters: {total_params / 1e6:.2f}M")

    # --- Dummy Data ---
    # Batch size 2, 4 Modalities (e.g., T1, T1c, T2, FLAIR)
    B, M, C, H, W = 2, 4, 1, 256, 256
    x = torch.randn(B, M, C, H, W).to(device)
    
    # Mask: Sample 1 has mod 0,1,2; Sample 2 has mod 0,3
    mask_indices = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32).to(device)

    # --- Forward Pass ---
    print("\n--- Running MMT Forward ---")
    try:
        output, x_recon = model(x, mask_indices)
        
        # Verification
        print(f"Input Shape: {x.shape}")
        print(f"Output Shape: {output.shape}")
        print(f"Recon Shape: {x_recon.shape}")       
        print("\nMMT Full Model forward pass successful.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
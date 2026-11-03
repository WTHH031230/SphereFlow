import torch
import torch.nn as nn
from .decoders import GeneralDecoder
from .encoders import ARCHS
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional
from math import sqrt
from typing import Protocol

class Stage1Protocal(Protocol):
    # must have patch size attribute
    patch_size: int
    hidden_size: int 
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

class RAE(nn.Module):
    def __init__(self, 
        # ---- encoder configs ----
        encoder_cls: str = 'Dinov2withNorm',
        encoder_config_path: str = 'facebook/dinov2-base',
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        # ---- decoder configs ----
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- noising, reshaping and normalization-----
        noise_tau: float = 0.8,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)

        print(f"encoder_config_path: {encoder_config_path}")
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.register_buffer('encoder_mean', torch.tensor(proc.image_mean).view(1, 3, 1, 1))
        self.register_buffer('encoder_std', torch.tensor(proc.image_std).view(1, 3, 1, 1))
        
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 # number of patches of the latent
        
        # decoder
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim # set the hidden size of the decoder to be the same as the encoder's output
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches)) 
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
        # load pretrained decoder weights
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            checkpoint = torch.load(pretrained_decoder_path, map_location='cpu')
            if "ema" in checkpoint:
                state_dict = checkpoint["ema"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            
            # extract decoder weights and strip prefix if needed
            decoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("decoder."):
                    decoder_state_dict[k[len("decoder."):]] = v
                elif "encoder." not in k: # if it doesn't have decoder. but also isn't encoder., assume it's pure decoder
                    decoder_state_dict[k] = v
                    
            keys = self.decoder.load_state_dict(decoder_state_dict, strict=True)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")
            if len(keys.unexpected_keys) > 0:
                print(f"Unexpected keys when loading pretrained decoder: {keys.unexpected_keys}")
        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            self.register_buffer('latent_mean', stats.get('mean', None))
            self.register_buffer('latent_var', stats.get('var', None))
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False
            self.register_buffer('latent_mean', None, persistent=False)
            self.register_buffer('latent_var', None, persistent=False)
    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # normalize input
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
        x = (x - self.encoder_mean) / self.encoder_std
        z = self.encoder(x)
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            z = (z - self.latent_mean) / torch.sqrt(self.latent_var + self.eps)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_with(z, self.decoder)

    def decode_with(self, z: torch.Tensor, decoder: torch.nn.Module) -> torch.Tensor:
        """Decode latent z with a given decoder (e.g. for dual-modality training)."""
        if self.do_normalization:
            z = z * torch.sqrt(self.latent_var + self.eps) + self.latent_mean
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
        output = decoder(z, drop_cls_token=False).logits
        x_rec = decoder.unpatchify(output)
        x_rec = x_rec * self.encoder_std + self.encoder_mean
        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
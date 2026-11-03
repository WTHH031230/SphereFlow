import torch
import torch.nn as nn
import functools
import numpy as np
import os
import sys

# -------------------------------------------------------------------------
# 0. Helpers for multi-GPU wrapping (DDP / torch.compile / Accelerate)
# -------------------------------------------------------------------------

def _unwrap_module(m):
    """
    Unwrap common wrappers:
    - DDP: .module
    - torch.compile: ._orig_mod (may also appear in state_dict keys)
    """
    if m is None:
        return None
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


class DiscriminatorBank(nn.Module):
    """
    Wrap a list of discriminators in a single nn.Module so it can be wrapped by DDP.
    Access discriminators by modality index.
    """
    def __init__(self, discriminators):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)

    def get(self, idx: int) -> nn.Module:
        return self.discriminators[idx]

    def forward(self, x, idx: int):
        """
        Supports two call styles:
        - idx: int -> run discriminator[idx] on the whole batch x
        - idx: Tensor[int64] shape [N] -> route each sample in x (shape [N, C, H, W])
          to its modality discriminator, and return a dict: {mod_idx: pred}
          where pred keeps the MultiscaleDiscriminator output structure.
        """
        if isinstance(idx, int):
            return self.discriminators[idx](x)

        if not torch.is_tensor(idx):
            raise TypeError(f"idx must be int or torch.Tensor, got {type(idx)}")
        if idx.dim() != 1:
            raise ValueError(f"idx must be 1D tensor, got shape {tuple(idx.shape)}")
        if x.dim() != 4:
            raise ValueError(f"x must be 4D (N,C,H,W) when idx is tensor, got shape {tuple(x.shape)}")

        preds = {}
        # Keep order within each modality as it appears in x (stable boolean mask)
        for m in range(len(self.discriminators)):
            sel = (idx == m)
            if torch.any(sel):
                preds[int(m)] = self.discriminators[m](x[sel])
        return preds


def _get_discriminator(discriminators, mod_idx: int) -> nn.Module:
    """
    Supports:
    - DiscriminatorBank (preferred for multi-GPU)
    - nn.ModuleList / list of discriminators
    - Wrapped versions (DDP / torch.compile)
    """
    disc = _unwrap_module(discriminators)
    if disc is None:
        raise ValueError("discriminators is None")

    if hasattr(disc, "get") and callable(getattr(disc, "get")):
        return disc.get(mod_idx)

    # ModuleList / list
    return disc[mod_idx]


def _call_discriminator(discriminators, x, mod_idx: int):
    """
    IMPORTANT: if `discriminators` is DDP-wrapped, we should call it via the wrapper
    (so DDP can run its per-forward bookkeeping). For non-callable containers
    (ModuleList/list), we fall back to indexing.
    """
    try:
        return discriminators(x, mod_idx)
    except TypeError:
        netD = _get_discriminator(discriminators, mod_idx)
        return netD(x)

# -------------------------------------------------------------------------
# 1. Discriminator & GANLoss Definitions (From Reference)
# -------------------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
      
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = real_tensor.to(input.device)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = fake_tensor.to(input.device)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

# -------------------------------------------------------------------------
# 2. MMTLoss (Modified to use GANLoss)
# -------------------------------------------------------------------------

class MMTLoss(nn.Module):
    def __init__(self, lambda_r=5.0, lambda_s=20.0, lambda_adv=0.1):
        """
        MMT Loss Function adapted for 5D Tensor outputs and MultiscaleDiscriminator.
        """
        super().__init__()
        self.lambda_r = lambda_r
        self.lambda_s = lambda_s
        self.lambda_adv = lambda_adv

        self.l1_loss = nn.L1Loss()
        # Initializing GANLoss to handle multiscale outputs automatically
        self.criterionGAN = GANLoss(use_lsgan=True)

    def extract_gt_targets(self, x_gt, mask):
        """
        Slices the Ground Truth x based on the mask to match model outputs.
        """
        B, M, C, H, W = x_gt.shape
        
        gt_recon_list = []
        gt_synth_list = []
        
        for b in range(B):
            avail_indices = (mask[b] == 1).nonzero(as_tuple=True)[0]
            missing_indices = (mask[b] == 0).nonzero(as_tuple=True)[0]
            
            gt_recon_list.append(x_gt[b, avail_indices])
            gt_synth_list.append(x_gt[b, missing_indices])
            
        gt_recon = torch.stack(gt_recon_list, dim=0) 
        gt_synth = torch.stack(gt_synth_list, dim=0)
        
        return gt_recon, gt_synth

    @staticmethod
    def _slice_multiscale_pred(pred, slc: slice):
        """
        pred: MultiscaleDiscriminator output, usually List[List[Tensor]].
        Returns same nested structure but with batch dimension sliced.
        """
        # MultiscaleDiscriminator returns: result[list over scales] -> each is [tensor] if getIntermFeat=False
        sliced = []
        for scale_out in pred:
            sliced_scale = []
            for t in scale_out:
                sliced_scale.append(t[slc])
            sliced.append(sliced_scale)
        return sliced

    def _pack_by_modality(self, tensors_by_mod):
        """
        tensors_by_mod: dict[int -> Tensor[Nm,1,H,W]]
        Return:
          x_all: Tensor[N,1,H,W] concatenated by modality increasing
          idx_all: LongTensor[N] modality indices aligned with x_all
          ranges: dict[int -> (start, end)]
        """
        xs = []
        idxs = []
        ranges = {}
        start = 0
        for m in sorted(tensors_by_mod.keys()):
            x_m = tensors_by_mod[m]
            n = int(x_m.shape[0])
            if n == 0:
                continue
            xs.append(x_m)
            idxs.append(torch.full((n,), int(m), device=x_m.device, dtype=torch.long))
            ranges[int(m)] = (start, start + n)
            start += n
        if len(xs) == 0:
            # Return empty placeholders on CPU (won't be used)
            return None, None, {}
        x_all = torch.cat(xs, dim=0)
        idx_all = torch.cat(idxs, dim=0)
        return x_all, idx_all, ranges

    def compute_d_loss_vectorized(self, x_gt, mask, y_gen_detach, discriminators):
        """
        Compute discriminator loss with a SINGLE DDP forward on `discriminators`.
        """
        _, target_synth = self.extract_gt_targets(x_gt, mask)

        # Build per-modality real/fake batches (real then fake) so we can slice later.
        B = x_gt.shape[0]
        per_mod = {}
        split_counts = {}  # m -> (n_real, n_fake)

        for b in range(B):
            missing_indices = (mask[b] == 0).nonzero(as_tuple=True)[0]
            for i, mod_idx in enumerate(missing_indices):
                m = int(mod_idx.item())
                # each is (1,H,W) -> (1,1,H,W)
                real_img = target_synth[b, i].unsqueeze(0)
                fake_img = y_gen_detach[b, i].unsqueeze(0)
                if m not in per_mod:
                    per_mod[m] = {"real": [], "fake": []}
                per_mod[m]["real"].append(real_img)
                per_mod[m]["fake"].append(fake_img)

        tensors_by_mod = {}
        for m, d in per_mod.items():
            real_m = torch.cat(d["real"], dim=0) if len(d["real"]) else None
            fake_m = torch.cat(d["fake"], dim=0) if len(d["fake"]) else None
            n_real = int(real_m.shape[0]) if real_m is not None else 0
            n_fake = int(fake_m.shape[0]) if fake_m is not None else 0
            if n_real + n_fake == 0:
                continue
            split_counts[m] = (n_real, n_fake)
            if n_real == 0:
                x_m = fake_m
            elif n_fake == 0:
                x_m = real_m
            else:
                x_m = torch.cat([real_m, fake_m], dim=0)
            tensors_by_mod[m] = x_m

        x_all, idx_all, ranges = self._pack_by_modality(tensors_by_mod)
        if x_all is None:
            # No missing -> no discriminator update
            return torch.zeros((), device=x_gt.device, dtype=x_gt.dtype)

        preds = discriminators(x_all, idx_all)  # dict: m -> pred

        loss_D = 0.0
        count_d = 0
        for m, (n_real, n_fake) in split_counts.items():
            if m not in preds:
                continue
            pred_m = preds[m]
            # Slice within this modality's local batch (real first, fake second)
            if n_real > 0:
                pred_real = self._slice_multiscale_pred(pred_m, slice(0, n_real))
                loss_d_real = self.criterionGAN(pred_real, True)
            else:
                loss_d_real = 0.0
            if n_fake > 0:
                pred_fake = self._slice_multiscale_pred(pred_m, slice(n_real, n_real + n_fake))
                loss_d_fake = self.criterionGAN(pred_fake, False)
            else:
                loss_d_fake = 0.0
            loss_D = loss_D + (loss_d_real + loss_d_fake) * 0.5
            count_d += 1

        if count_d > 0:
            loss_D = loss_D / count_d
        else:
            loss_D = torch.zeros((), device=x_gt.device, dtype=x_gt.dtype)
        return loss_D

    def compute_g_adv_vectorized(self, mask, y_gen, discriminators):
        """
        Compute generator adversarial loss with a SINGLE DDP forward on `discriminators`,
        while freezing discriminator params (grads only flow to y_gen).
        """
        B = y_gen.shape[0]
        per_mod_fake = {}
        for b in range(B):
            missing_indices = (mask[b] == 0).nonzero(as_tuple=True)[0]
            for i, mod_idx in enumerate(missing_indices):
                m = int(mod_idx.item())
                fake_img = y_gen[b, i].unsqueeze(0)  # (1,1,H,W)
                if m not in per_mod_fake:
                    per_mod_fake[m] = []
                per_mod_fake[m].append(fake_img)

        tensors_by_mod = {}
        for m, lst in per_mod_fake.items():
            if len(lst):
                tensors_by_mod[m] = torch.cat(lst, dim=0)

        x_all, idx_all, ranges = self._pack_by_modality(tensors_by_mod)
        if x_all is None:
            return torch.zeros((), device=y_gen.device, dtype=y_gen.dtype)

        # Freeze D params for G update
        req_flags = []
        for p in discriminators.parameters():
            req_flags.append(p.requires_grad)
            p.requires_grad_(False)
        try:
            preds = discriminators(x_all, idx_all)  # dict m -> pred
        finally:
            for p, f in zip(discriminators.parameters(), req_flags):
                p.requires_grad_(f)

        loss_adv = 0.0
        cnt = 0
        for m, pred_m in preds.items():
            loss_adv = loss_adv + self.criterionGAN(pred_m, True)
            cnt += 1
        if cnt > 0:
            loss_adv = loss_adv / cnt
        else:
            loss_adv = torch.zeros((), device=y_gen.device, dtype=y_gen.dtype)
        return loss_adv

    def forward(self, x_gt, mask, y_gen, x_recon, discriminators):
        """
        Returns:
            loss_G: Total Generator Loss
            loss_D: Total Discriminator Loss
            log_vars: Dict for logging
        """
        
        target_recon, target_synth = self.extract_gt_targets(x_gt, mask)
        
        # -------------------------
        # Generator Losses
        # -------------------------
        
        # 1.1 Synthesis Loss (L1)
        loss_s = self.l1_loss(y_gen, target_synth)
        
        # 1.2 Reconstruction Loss (L1)
        loss_r = self.l1_loss(x_recon, target_recon)
        
        # The adversarial term of G is added by the trainer (compute_g_adv_vectorized).
        loss_adv = torch.zeros((), device=x_gt.device, dtype=x_gt.dtype)
        loss_G = (self.lambda_r * loss_r) + (self.lambda_s * loss_s)

        # -------------------------
        # Discriminator Losses
        # -------------------------
        loss_D = 0.0
        
        # D loss: vectorized single-forward version for DDP safety
        loss_D = self.compute_d_loss_vectorized(x_gt, mask, y_gen.detach(), discriminators)

        log_vars = {
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
            "loss_r": loss_r.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv if isinstance(loss_adv, float) else loss_adv.item()
        }
        
        return loss_G, loss_D, log_vars

# -------------------------------------------------------------------------
# 3. Main Execution
# -------------------------------------------------------------------------

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)
    
    # Import MMT from the network module
    from network.MMT.mmt import MMT

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

    # Initialize MMT Model
    model = MMT(**config).to(device)
    
    # Parameter Count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total MMT Parameters: {total_params / 1e6:.2f}M")

    # --- Dummy Data ---
    # Batch size 2, 4 Modalities (e.g., T1, T1c, T2, FLAIR)
    B, M, C, H, W = 2, 4, 1, 256, 256
    x = torch.randn(B, M, C, H, W).to(device)
    
    # Mask: Sample 1 has mod 0,1,2 (missing 3); Sample 2 has mod 0,3 (missing 1,2)
    mask_indices = torch.tensor([
        [1, 1, 0, 0],
        [1, 0, 0, 1]
    ], dtype=torch.float32).to(device)

    # --- Initialize Discriminators (One per modality) ---
    # Using define_D with MultiscaleDiscriminator
    gpu_ids = [0] if torch.cuda.is_available() else []
    discriminators = []
    for i in range(4):
        netD = define_D(
            input_nc=1, 
            ndf=64, 
            n_layers_D=3, 
            norm='instance', 
            use_sigmoid=False, 
            num_D=2, # Use 2 scales for testing speed
            gpu_ids=gpu_ids
        )
        discriminators.append(netD)
    
    # --- Initialize Loss ---
    criterion = MMTLoss().to(device)
    
    # --- Forward Pass ---
    print("\n--- Running MMT Forward ---")
    y_gen, x_recon = model(x, mask_indices)
    
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape (Synthesis): {y_gen.shape}")
    print(f"Recon Shape: {x_recon.shape}")
    
    # --- Loss Calculation ---
    loss_G, loss_D, logs = criterion(x, mask_indices, y_gen, x_recon, discriminators)
    
    print("\n--- Loss Calculation ---")
    print(f"Generator Loss: {loss_G.item():.4f}")
    print(f"Discriminator Loss: {loss_D.item():.4f}")
    print(f"Details: {logs}")
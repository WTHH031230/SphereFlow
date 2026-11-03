import os
import time
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from tqdm import tqdm
import swanlab
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs, broadcast_object_list

try:
    from src.brats_dataset import get_loader
    from network.MMT.basic_cnn_blocks import process_modality
    from utils.evaluation_metrics import compute_measure
    
    from losses.mmt_loss import MMTLoss, define_D as define_D_mmt, DiscriminatorBank
    from network.MMT.mmt import MMT
    
    from src.disc.lpips import LPIPS
    
    from losses.modimp_loss import ImputationLoss as ModImpLoss 
    from network.ModImp.modimp import ImputationModel as ModImpModel

    from network.Zhang.zhang import ZhangModel, zhang_to_image_range, zhang_to_model_range
    from losses.zhang_loss import ZhangLoss

    from network.M2DN.m2dn import M2DN
    from losses.m2dn_loss import M2DNLoss

    from network.MMGAN.mmgan import (
        MMGAN,
        Discriminator as MMGANDiscriminator,
        weights_init_normal as mmgan_weights_init,
    )
    from losses.mmgan_loss import MMGANLoss

    from network.MMSyn.mmsyn import MMSyn
    from losses.mmsyn_loss import MMSynLoss
    
    from src.disc.lpips import LPIPS
    from src.eval import evaluate_reconstruction_distributed, calculate_rfid, calculate_is
    from src.brats_dataset import BraTS2023Dataset as RAEBraTSDataset, get_transforms as get_rae_transforms
    from torchvision.utils import make_grid
    
except ImportError as e:
    print(f"Import Error: {e}. Please ensure your file structure matches the imports.")

class UnifiedEvalWrapper(nn.Module):
    def __init__(self, model, model_type, target_modality=0):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.target_modality = target_modality

    def forward(self, x):
        # x is [B, 2, H, W] from self.eval_dataset_all
        B = x.size(0)
        # Reshape to [B, 2, 1, H, W] for models that expect it
        full_x = x.unsqueeze(2) 
        
        # --- Generative Evaluation (Synthesis from others) ---
        # Task: Predict target_modality from the available modality
        mask = torch.ones((B, 2), device=x.device)
        mask[:, self.target_modality] = 0.0
        
        if self.model_type == 'ModImp':
            recon_vol, _, _ = self.model(full_x, mask)
            recon = recon_vol[:, self.target_modality] 
            if recon.dim() == 3: recon = recon.unsqueeze(1)
            
        elif self.model_type == 'MMT':
            # Since only 1 modality is missing (target_modality), y_gen is [B, 1, 1, H, W]
            y_gen = self.model(full_x, mask)
            recon = y_gen[:, 0, 0:1]
        elif self.model_type == 'Zhang':
            img_masked = zhang_to_model_range(full_x) * mask.view(B, 2, 1, 1, 1)
            pred = self.model(img_masked, mask)
            recon = zhang_to_image_range(pred[:, self.target_modality, 0:1]).clamp(0.0, 1.0)
        elif self.model_type == 'M2DN':
            recon_vol = self.model(full_x, mask_indices=mask, mode='inference')
            recon = recon_vol[:, self.target_modality, 0:1]
        elif self.model_type == 'MMGAN':
            fake = self.model(full_x.squeeze(2), mask)               # [B, N, H, W]
            recon = fake[:, self.target_modality:self.target_modality + 1]  # [B, 1, H, W]
        elif self.model_type == 'MMSyn':
            pred_dict = self.model(full_x, mask)
            recon = pred_dict['fused'][:, self.target_modality]      # [B, 1, H, W]
        else:
            recon = full_x[:, self.target_modality, 0:1]
        
        # Align channels with target: model returns [B, 1, H, W], target is [B, 3, H, W]
        return recon.repeat(1, 3, 1, 1) # [B, 3, H, W]

@torch.no_grad()
def generate_reference_npz(dataset, path, batch_size=128, num_workers=4):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    all_images = []
    for images, _ in tqdm(loader, desc="Generating reference NPZ"):
        # images is [B, C, H, W], convert to [B, H, W, C] uint8
        img_np = images.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        all_images.append(img_np)
    all_images = np.concatenate(all_images, axis=0)
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, arr_0=all_images)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

class UnifiedTrainer:
    @staticmethod
    def _set_requires_grad(module, flag: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = flag

    @staticmethod
    def _zhang_prepare_input(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return zhang_to_model_range(img) * mask.view(img.size(0), img.size(1), 1, 1, 1)

    @staticmethod
    def _zhang_to_image_range(pred: torch.Tensor) -> torch.Tensor:
        return zhang_to_image_range(pred).clamp(0.0, 1.0)

    def __init__(self, args) -> None:
        self.args = args
        
        # These models have mask-dependent branches, so DDP needs unused-parameter detection.
        kwargs_handlers = []
        if getattr(args, "model_type", None) in ("MMT", "ModImp", "Zhang", "MMGAN") or getattr(args, "find_unused_parameters", False):
            kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))

        self.accelerator = Accelerator(
            mixed_precision="fp16" if args.amp else "no",
            log_with="swanlab" if args.logdir and swanlab is not None else None,
            project_dir=args.logdir,
            kwargs_handlers=kwargs_handlers,
        )
        self.device = self.accelerator.device
        
        # Same RNG on every rank, so all ranks draw the same missing-modality masks.
        set_seed(42, device_specific=False)
            
        self.trainer_initialized = False
        self.start_epoch = self.args.start_epoch
        self.global_step = 0
        # Initialize best_metric: PSNR/SSIM are higher-better, others (like Loss) are lower-better.
        # Most tasks here use 'avg_psnr' for checkpointing.
        self.best_metric = -1.0 

        self._init_curriculum_lists()
        
        # Initialize LPIPS for evaluation (aligned with train_single_decoder.py)
        try:
            self.lpips_metric = LPIPS().to(self.device).eval()
            self.accelerator.print("LPIPS metric initialized for evaluation.")
        except Exception as e:
            self.lpips_metric = None
            self.accelerator.print(f"LPIPS initialization failed: {e}. Skipping perceptual metrics.")

    def _compute_single_image_metrics(self, gt_2d: torch.Tensor, pred_2d: torch.Tensor):
        # gt_2d/pred_2d: [H, W]
        gt_4d = gt_2d.unsqueeze(0).unsqueeze(0).contiguous()
        pred_4d = pred_2d.unsqueeze(0).unsqueeze(0).contiguous()
        data_range = (gt_4d.max() - gt_4d.min()).clamp(min=1e-8).item()
        psnr, ssim, rmse = compute_measure(gt_4d, pred_4d, data_range)
        
        lpips_v = 0.0
        if self.lpips_metric is not None:
            # LPIPS expects [-1, 1] and 3 channels
            # MRI / CT slices are 1-channel, repeat to 3
            gt_3c = gt_4d.repeat(1, 3, 1, 1).to(self.device) * 2.0 - 1.0
            pred_3c = pred_4d.repeat(1, 3, 1, 1).to(self.device) * 2.0 - 1.0
            with torch.no_grad():
                lpips_v = self.lpips_metric(gt_3c, pred_3c).item()
                
        return float(psnr), float(ssim), float(rmse), float(lpips_v)

    def _init_curriculum_lists(self):
        # Full: 2 inputs, 0 missing
        self.full_modalities = [[1, 1]]
        # Bidirectional single-missing training:
        # [1, 0] => MRI -> CT, [0, 1] => CT -> MRI
        self.single_missing = [
            [1, 0],
            [0, 1],
        ]
        # Only support single missing since there are only 2 modalities
        self.double_missing = self.single_missing
        self.tri_missing = self.single_missing

    def initial_trainer(self):
        self.trainer_initialized = False
        self.config_dataset()
        self.init_model_and_optimizer() 
        self.config_losses()       
        self.config_writer()
        
        # Models are prepared before optimizers.
        models_to_prepare = [self.model]

        # The ModImp loss has trainable projectors, so it is prepared as well.
        prepare_loss = (
            hasattr(self, "loss_fn")
            and isinstance(self.loss_fn, nn.Module)
            and any(p.requires_grad for p in self.loss_fn.parameters())
        )
        if prepare_loss:
            models_to_prepare.append(self.loss_fn)

        if hasattr(self, 'discriminator') and self.discriminator is not None:
            models_to_prepare.append(self.discriminator)
        if hasattr(self, 'discriminators') and self.discriminators is not None:
            models_to_prepare.append(self.discriminators)

        objects_to_prepare = [
            *models_to_prepare,
            self.optimizer,
        ]
        if hasattr(self, 'optimizer_D') and self.optimizer_D is not None:
            objects_to_prepare.append(self.optimizer_D)
        objects_to_prepare.extend([self.train_loader, self.val_loader, self.scheduler])
        if hasattr(self, 'scheduler_D') and self.scheduler_D is not None:
            objects_to_prepare.append(self.scheduler_D)

        prepared_objects = self.accelerator.prepare(*objects_to_prepare)

        idx = 0
        self.model = prepared_objects[idx]; idx += 1
        if prepare_loss:
            self.loss_fn = prepared_objects[idx]; idx += 1
        if hasattr(self, 'discriminator') and self.discriminator is not None:
            self.discriminator = prepared_objects[idx]; idx += 1
        if hasattr(self, 'discriminators') and self.discriminators is not None:
            self.discriminators = prepared_objects[idx]; idx += 1

        self.optimizer = prepared_objects[idx]; idx += 1
        if hasattr(self, 'optimizer_D') and self.optimizer_D is not None:
            self.optimizer_D = prepared_objects[idx]; idx += 1

        self.train_loader = prepared_objects[idx]; idx += 1
        self.val_loader = prepared_objects[idx]; idx += 1
        self.scheduler = prepared_objects[idx]; idx += 1
        if hasattr(self, 'scheduler_D') and self.scheduler_D is not None:
            self.scheduler_D = prepared_objects[idx]; idx += 1

        # Resume logic: Auto-detect if resume_path is not provided
        resume_path = self.args.resume_path
        if not resume_path:
            # 1. Try to find 'last_state' directory (Accelerate save_state)
            potential_state = os.path.join(self.args.logdir, "last_state")
            if os.path.exists(potential_state):
                resume_path = potential_state
            else:
                # 2. Try to find the latest .pt checkpoint
                potential_pt = os.path.join(self.args.logdir, f"ckpt_{self.args.model_type}_latest.pt")
                if os.path.exists(potential_pt):
                    resume_path = potential_pt

        if resume_path:
            if os.path.isdir(resume_path):
                self.accelerator.print(f"Auto-detected state directory: {resume_path}. Resuming state...")
                self.accelerator.load_state(resume_path)
                # Load metadata to recover epoch and global_step
                res_cfg = os.path.join(resume_path, "resume_config.json")
                if os.path.exists(res_cfg):
                    with open(res_cfg, "r") as f:
                        cfg = json.load(f)
                        self.args.start_epoch = cfg.get("epoch", 0) + 1
                        self.global_step = cfg.get("global_step", 0)
                        self.best_metric = cfg.get("best_metric", self.best_metric)
                        self.accelerator.print(f"Resume Config: Epoch={self.args.start_epoch}, Step={self.global_step}, BestMetric={self.best_metric}")
                else:
                    self.accelerator.print("Warning: resume_config.json not found in last_state. Epoch will start from 0.")
            
            elif os.path.isfile(resume_path) or resume_path.endswith('.pt'):
                self.accelerator.print(f"Loading weights from checkpoint: {resume_path}")
                checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    unwrapped_model.load_state_dict(checkpoint['state_dict'])
                else:
                    unwrapped_model.load_state_dict(checkpoint)
                
                if not self.args.test and isinstance(checkpoint, dict):
                    if 'optimizer' in checkpoint and hasattr(self, 'optimizer'):
                        try: self.optimizer.load_state_dict(checkpoint['optimizer'])
                        except Exception as e: self.accelerator.print(f"Failed to load G optimizer: {e}")
                    if 'optimizer_D' in checkpoint and hasattr(self, 'optimizer_D'):
                        try: self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
                        except Exception as e: self.accelerator.print(f"Failed to load D optimizer: {e}")
                    if 'scheduler' in checkpoint and hasattr(self, 'scheduler'):
                        try: self.scheduler.load_state_dict(checkpoint['scheduler'])
                        except: pass
                    if 'scheduler_D' in checkpoint and hasattr(self, 'scheduler_D'):
                        try: self.scheduler_D.load_state_dict(checkpoint['scheduler_D'])
                        except: pass
                        
                    if 'epoch' in checkpoint:
                        self.args.start_epoch = checkpoint['epoch'] + 1
                        self.global_step = checkpoint.get('global_step', 0)
                        self.best_metric = checkpoint.get('best_metric', self.best_metric)
                        self.accelerator.print(f"Resuming from Epoch {self.args.start_epoch}, Step {self.global_step}")
        else:
            self.accelerator.print("No checkpoint found. Starting from scratch.")
            
        # Prepare Reference NPZs for Validation (Rank 0 only)
        if self.accelerator.is_main_process:
            for m in range(2):
                ref_path = os.path.join(self.args.logdir, f"ref_val_m{m}.npz")
                if not os.path.exists(ref_path):
                    self.accelerator.print(f"Generating reference NPZ for validation modality {m}...")
                    generate_reference_npz(self.eval_datasets_3ch[m], ref_path, batch_size=self.args.batch_size, num_workers=self.args.workers)
        self.accelerator.wait_for_everyone()

        self.trainer_initialized = True
        
    def config_dataset(self):
        self.train_loader = get_loader(
            json_path=self.args.data_json,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            phase='train',
            drop_last=True,
        )
        self.val_loader = get_loader(
            json_path=self.args.data_json,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            phase='val',
        )
        self.accelerator.print(f'Train size: {len(self.train_loader)*self.args.batch_size} | Val size: {len(self.val_loader)*self.args.batch_size}')
        
        # Eval datasets for evaluate_reconstruction_distributed
        # 1. Dataset returning both modalities (used as input for generative evaluation)
        self.eval_dataset_all = RAEBraTSDataset(
            json_path=self.args.data_json,
            phase='val',
            transform=get_rae_transforms('val'),
            modality_index=None
        )

        # 2. Datasets returning 3-channel single modality (used for generating reference NPZs)
        self.eval_datasets_3ch = []
        for m in range(2):
            m_dataset = RAEBraTSDataset(
                json_path=self.args.data_json,
                phase='val',
                transform=get_rae_transforms('val'),
                modality_index=m
            )
            self.eval_datasets_3ch.append(m_dataset)
        
        self.accelerator.print(f'Eval dataset loaded. Full size: {len(self.eval_dataset_all)}')

    def init_model_and_optimizer(self):
        self.accelerator.print(f">> Initializing Model: {self.args.model_type}")
        
        # ==================== ModImp ====================
        if self.args.model_type == 'ModImp':
            self.model = ModImpModel(
                img_size=self.args.img_size, patch_size=self.args.patch_size,
                embed_dim=self.args.embed_dim, depths=self.args.depths,
                num_heads=self.args.num_heads, con_dim=self.args.con_dim,
                mask_ratio_spa=self.args.mask_ratio_spa,
                num_modality=2
            )
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.optim_lr, betas=self.args.betas, weight_decay=self.args.decay)

        # ==================== MMT ====================
        elif self.args.model_type == 'MMT':
            self.model = MMT(patch_size=4, dim=6, depths=[2,2,2,2], num_heads=[3,6,12,24], num_mods=2)
            # Wrap discriminator list into a single Module so it can be wrapped by DDP/Accelerate.
            disc_list = [define_D_mmt(input_nc=1, ndf=64, n_layers_D=3, gpu_ids=[]) for _ in range(2)]
            self.discriminators = DiscriminatorBank(disc_list)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.optim_lr, betas=self.args.betas)
            self.optimizer_D = torch.optim.AdamW(self.discriminators.parameters(), lr=self.args.d_lr, betas=(0.5, 0.999))

        # ==================== M2DN ====================
        elif self.args.model_type == 'M2DN':
            self.model = M2DN(
                num_modalities=2,
                img_channels=1,
                enc_channels=getattr(self.args, 'm2dn_enc_ch', 32),
                t_emb_dim=getattr(self.args, 'm2dn_t_emb_dim', 256),
                T=getattr(self.args, 'm2dn_T', 1000),
                n_inf_steps=getattr(self.args, 'm2dn_n_inf_steps', 18),
            )
            if getattr(self.args, 'm2dn_setting', 'paper') == 'paper':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.args.optim_lr,
                    betas=(0.9, 0.999),
                )
            else:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.args.optim_lr,
                    betas=self.args.betas,
                    weight_decay=self.args.decay,
                )
            self.optimizer_D  = None
            self.discriminator = None

        # ==================== Zhang ====================
        elif self.args.model_type == 'Zhang':
            self.model = ZhangModel(num_mods=2)
            disc_bank = self.model.discriminators
            self.model.discriminators = nn.ModuleList()
            self.discriminators = disc_bank
            self.optimizer = torch.optim.SGD(
                self.model.generator.parameters(),
                lr=self.args.optim_lr,
                momentum=0.9,
                weight_decay=self.args.decay,
            )
            self.optimizer_D = torch.optim.SGD(
                self.discriminators.parameters(),
                lr=self.args.d_lr,
                momentum=0.9,
                weight_decay=self.args.decay,
            )

        # ==================== MMSyn ====================
        elif self.args.model_type == 'MMSyn':
            self.model = MMSyn(
                num_mods=2,
                latent_dim=getattr(self.args, 'mmsyn_latent_dim', 16),
                in_channels=1,
                out_channels=1,
                fusion=getattr(self.args, 'mmsyn_fusion', 'max'),
                relu_head=True,
            )
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.optim_lr,
                betas=(0.9, 0.999),
            )

        # ==================== MMGAN ====================
        elif self.args.model_type == 'MMGAN':
            with_tanh = getattr(self.args, 'mmgan_use_tanh', False)
            self.model = MMGAN(
                num_mods=2,
                z_type=getattr(self.args, 'mmgan_z_type', 'noise'),
                use_ic=bool(getattr(self.args, 'mmgan_ic', 1)),
                with_tanh=with_tanh,
                with_relu=not with_tanh,
            )
            self.model.apply(mmgan_weights_init)

            self.discriminator = MMGANDiscriminator(in_channels=2, out_channels=2, dataset='BRATS2018')
            self.discriminator.apply(mmgan_weights_init)

            b1 = getattr(self.args, 'mmgan_b1', 0.5)
            b2 = getattr(self.args, 'mmgan_b2', 0.999)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.optim_lr, betas=(b1, b2))
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.d_lr, betas=(b1, b2))

        # ModImp adds its loss parameters to the optimizer in config_losses, so its schedulers are built there.
        if self.args.model_type != 'ModImp':
            self._init_schedulers()
        else:
            self.scheduler = None
            self.scheduler_D = None

    def config_losses(self):
        self.accelerator.print(f">> Configuring Loss for {self.args.model_type}")
        
        if self.args.model_type == 'ModImp':
            self.loss_fn = ModImpLoss(
                alpha_lmse=self.args.alpha_lmse, alpha_lcon=self.args.alpha_lcon, 
                alpha_lconsis=self.args.alpha_lconsis, batch_size=self.args.batch_size, embed_dim=self.args.con_dim, device=self.device
            )
            # Add the trainable projectors of the loss to the optimizer.
            try:
                loss_params_all = [p for p in self.loss_fn.parameters() if p.requires_grad]
                if len(loss_params_all) > 0:
                    opt_param_ids = set(id(p) for g in self.optimizer.param_groups for p in g.get("params", []))
                    new_loss_params = [p for p in loss_params_all if id(p) not in opt_param_ids]
                    if len(new_loss_params) > 0:
                        self.optimizer.add_param_group({"params": new_loss_params})
            except Exception:
                pass
            self._init_schedulers()
            
        elif self.args.model_type == 'MMT':
            self.loss_fn = MMTLoss(lambda_r=5.0, lambda_s=20.0, lambda_adv=0.1).to(self.device)

        elif self.args.model_type == 'Zhang':
            self.loss_fn = ZhangLoss(lambda_syn=100.0, lambda_rec=30.0, lambda_adv=1.0)

        elif self.args.model_type == 'M2DN':
            self.loss_fn = M2DNLoss(lambda_recon=1.0)

        elif self.args.model_type == 'MMGAN':
            self.loss_fn = MMGANLoss(
                lam=getattr(self.args, 'mmgan_lam', 0.9),
                ic=bool(getattr(self.args, 'mmgan_ic', 1)),
            )

        elif self.args.model_type == 'MMSyn':
            self.loss_fn = MMSynLoss(
                lambda_recon=getattr(self.args, 'mmsyn_lambda_recon', 1.0),
                lambda_concat=getattr(self.args, 'mmsyn_lambda_concat', 1.0),
                ind_outs=bool(getattr(self.args, 'mmsyn_ind_outs', 1)),
                fuse_outs=bool(getattr(self.args, 'mmsyn_fuse_outs', 1)),
            )

    def _init_schedulers(self):
        steps_per_epoch = len(self.train_loader)
        total_steps = (self.args.max_epochs - self.args.start_epoch) * steps_per_epoch

        if self.args.model_type == 'Zhang':
            hold_epochs = 50.0 * float(max(self.args.max_epochs, 1)) / 200.0
            total_epochs = float(max(self.args.max_epochs, 1))

            def lr_lambda(step):
                epoch_pos = float(step) / max(steps_per_epoch, 1)
                if total_epochs <= hold_epochs or epoch_pos < hold_epochs:
                    return 1.0
                progress = (epoch_pos - hold_epochs) / max(total_epochs - hold_epochs, 1.0)
                return max(0.0, 1.0 - progress)

            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            if hasattr(self, 'optimizer_D') and self.optimizer_D is not None:
                self.scheduler_D = LambdaLR(self.optimizer_D, lr_lambda=lr_lambda)
            return

        if self.args.model_type == 'M2DN' and getattr(self.args, 'm2dn_setting', 'paper') == 'paper':
            self.scheduler = None
            self.scheduler_D = None
            return

        final_ratio = self.args.eta_min / self.args.optim_lr

        # Cosine decay from 1.0 to final_ratio, no restarts
        def lr_lambda(step):
            if step >= total_steps:
                return final_ratio
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
            return final_ratio + (1.0 - final_ratio) * cosine_val

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        
        if hasattr(self, 'optimizer_D') and self.optimizer_D is not None:
            final_ratio_D = self.args.eta_min / self.args.d_lr
            def lr_lambda_D(step):
                if step >= total_steps:
                    return final_ratio_D
                cosine_val = 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
                return final_ratio_D + (1.0 - final_ratio_D) * cosine_val
            self.scheduler_D = LambdaLR(self.optimizer_D, lr_lambda=lr_lambda_D)
            
    def config_writer(self):
        self.use_swanlab = False
        if self.args.logdir is not None and self.accelerator.is_main_process:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            exp_name = f"{self.args.model_type}_{timestamp}"

            init_kwargs = {
                "swanlab": {
                    "experiment_name": exp_name,
                    "config": vars(self.args),
                    "logdir": self.args.logdir
                }
            }
            self.accelerator.init_trackers(
                project_name="BraTS-Unified-Imputation",
                init_kwargs=init_kwargs
            )
            self.use_swanlab = True
            self.accelerator.print(f"SwanLab initialized via Accelerator: {exp_name}")

    def get_mask_indice(self, batch_size, epoch):
        if self.args.model_type == 'ModImp':
            if epoch < int(self.args.max_epochs // 5):
                group = "single"
            elif int(self.args.max_epochs // 5) <= epoch < int(3 * self.args.max_epochs // 5):
                group = random.choice(["double", "single"])
            else:
                group = random.choice(["tri", "double", "single"])
        else:
            group = random.choice(["tri", "double", "single"])

        if group == "full":
            mask_pool = self.full_modalities
        elif group == "tri":
            mask_pool = self.tri_missing
        elif group == "double":
            mask_pool = self.double_missing
        else:
            mask_pool = self.single_missing

        # All ranks use the same mask, so DDP runs the same branch everywhere.
        use_dist = (
            getattr(self.accelerator, "num_processes", 1) > 1
            and dist.is_available()
            and dist.is_initialized()
        )

        if use_dist:
            # Create on rank0, broadcast to all ranks.
            if self.accelerator.is_main_process:
                if batch_size <= len(mask_pool):
                    chosen = random.sample(mask_pool, k=batch_size)
                else:
                    chosen = [random.choice(mask_pool) for _ in range(batch_size)]
                mask = torch.tensor(chosen, dtype=torch.float32, device=self.device)  # [B,2]
            else:
                mask = torch.empty((batch_size, 2), dtype=torch.float32, device=self.device)

            dist.broadcast(mask, src=0)
            return mask

        # single process fallback
        if batch_size <= len(mask_pool):
            chosen = random.sample(mask_pool, k=batch_size)
        else:
            chosen = [random.choice(mask_pool) for _ in range(batch_size)]

        mask = torch.tensor(chosen, dtype=torch.float32, device=self.device)  # [B, 2]
        return mask

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_meters = {} 
        
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch} [{self.args.model_type}]", disable=not self.accelerator.is_local_main_process)
        
        for idx, batch_data in enumerate(pbar):
            if self.args.max_steps > 0 and idx >= self.args.max_steps:
                self.accelerator.print(f"Reached max_steps ({self.args.max_steps}), ending epoch early.")
                break
            
            self.global_step += 1
            images, _ = batch_data
            img = images.unsqueeze(2)  # [B, N, 1, H, W]
            B = img.size(0)

            if self.args.model_type == 'ModImp':
                mask_indice = self.get_mask_indice(B, epoch)
                with self.accelerator.autocast():
                    x_rec, x_con, y_con = self.model(img, mask_indice)
                    l_mse, l_diff, l_con = self.loss_fn(x_rec, img.squeeze(2), x_con, y_con)
                    total_loss = l_mse + l_diff + l_con
                
                self.optimizer.zero_grad()
                self.accelerator.backward(total_loss)
                self.optimizer.step()
                log_dict = {'mse': l_mse.item(), 'diff': l_diff.item(), 'con': l_con.item(), 'total': total_loss.item()}

                if self.global_step % getattr(self.args, 'sample_every', 500) == 0 and self.accelerator.is_main_process:
                    with torch.no_grad():
                        n_show = min(4, B)
                        # Prepare MRI GT and CT GT
                        mri_gt = img[:n_show, 0]
                        ct_gt = img[:n_show, 1]
                        if mri_gt.dim() == 3: mri_gt = mri_gt.unsqueeze(1)
                        if ct_gt.dim() == 3: ct_gt = ct_gt.unsqueeze(1)
                        
                        # x_rec already contains both modalities from model output
                        ct_gen = x_rec[:n_show, 1]
                        if ct_gen.dim() == 3: ct_gen = ct_gen.unsqueeze(1)

                        vis_mri_gt = mri_gt.repeat(1, 3, 1, 1)
                        vis_ct_gt = ct_gt.repeat(1, 3, 1, 1)
                        vis_ct_gen = ct_gen.repeat(1, 3, 1, 1)
                        
                        comparison = torch.cat([vis_mri_gt, vis_ct_gt, vis_ct_gen], dim=0).cpu().float()
                        grid = make_grid(comparison, nrow=n_show, normalize=True, value_range=(0, 1))
                        grid_np = grid.clamp(0, 1).mul(255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        if self.use_swanlab:
                            self.accelerator.log({"samples": swanlab.Image(grid_np)}, step=self.global_step)
                        self.accelerator.print(f"Logged visualization samples (MRI GT, CT GT, CT Gen) for ModImp at step {self.global_step}")

            elif self.args.model_type == 'MMT':
                mask_indice = self.get_mask_indice(B, epoch)
                with self.accelerator.autocast():
                    y_gen, x_recon = self.model(img, mask_indice)
                    loss_G, loss_D, log_dict_mmt = self.loss_fn(img, mask_indice, y_gen, x_recon, self.discriminators)

                # Update D
                self.optimizer_D.zero_grad()
                self.accelerator.backward(loss_D)
                self.optimizer_D.step()
                # Update G
                self.optimizer.zero_grad()
                # Adversarial term for G: one packed D forward with D frozen.
                if hasattr(self.loss_fn, "compute_g_adv_vectorized"):
                    with self.accelerator.autocast():
                        loss_adv_g = self.loss_fn.compute_g_adv_vectorized(mask_indice, y_gen, self.discriminators)
                    loss_G = loss_G + (self.loss_fn.lambda_adv * loss_adv_g)
                    # Keep log dict consistent
                    try:
                        log_dict_mmt["loss_adv"] = float(loss_adv_g.detach().float().item())
                        log_dict_mmt["loss_G"] = float(loss_G.detach().float().item())
                    except Exception:
                        pass
                self.accelerator.backward(loss_G)
                self.optimizer.step()
                log_dict = log_dict_mmt

                if self.global_step % getattr(self.args, 'sample_every', 500) == 0 and self.accelerator.is_main_process:
                    with torch.no_grad():
                        n_show = min(4, B)
                        # Prepare MRI GT and CT GT
                        mri_gt = img[:n_show, 0]
                        ct_gt = img[:n_show, 1]
                        if mri_gt.dim() == 3: mri_gt = mri_gt.unsqueeze(1)
                        if ct_gt.dim() == 3: ct_gt = ct_gt.unsqueeze(1)
                        
                        # Robustly extract CT (modality 1) for each sample in the batch
                        ct_gen_list = []
                        for b in range(n_show):
                            if mask_indice[b, 1] == 0:  # CT is missing for this sample
                                miss_indices = (mask_indice[b] == 0).nonzero(as_tuple=True)[0]
                                ct_idx = (miss_indices == 1).nonzero(as_tuple=True)[0].item()
                                ct_gen_list.append(y_gen[b, ct_idx])
                            else:  # CT is present for this sample
                                avail_indices = (mask_indice[b] == 1).nonzero(as_tuple=True)[0]
                                ct_idx = (avail_indices == 1).nonzero(as_tuple=True)[0].item()
                                ct_gen_list.append(x_recon[b, ct_idx])
                        ct_gen = torch.stack(ct_gen_list, dim=0)

                        if ct_gen.dim() == 4: ct_gen = ct_gen[:, 0:1] # Ensure [N, 1, H, W]

                        vis_mri_gt = mri_gt.repeat(1, 3, 1, 1)
                        vis_ct_gt = ct_gt.repeat(1, 3, 1, 1)
                        vis_ct_gen = ct_gen.repeat(1, 3, 1, 1)
                        
                        comparison = torch.cat([vis_mri_gt, vis_ct_gt, vis_ct_gen], dim=0).cpu().float()
                        grid = make_grid(comparison, nrow=n_show, normalize=True, value_range=(0, 1))
                        grid_np = grid.clamp(0, 1).mul(255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        if self.use_swanlab:
                            self.accelerator.log({"samples": swanlab.Image(grid_np)}, step=self.global_step)
                        self.accelerator.print(f"Logged visualization samples (MRI GT, CT GT, CT Gen) for MMT at step {self.global_step}")

            elif self.args.model_type == 'Zhang':
                mask_indice = self.get_mask_indice(B, epoch)
                img_target = zhang_to_model_range(img)
                img_input = self._zhang_prepare_input(img, mask_indice)

                with self.accelerator.autocast():
                    pred = self.model(img_input, mask_indice)

                self._set_requires_grad(self.discriminators, True)
                self.optimizer_D.zero_grad()
                disc_loss = self.loss_fn.discriminator_loss(
                    pred.detach(), img_target, mask_indice, self.discriminators
                )
                self.accelerator.backward(disc_loss)
                self.optimizer_D.step()

                self._set_requires_grad(self.discriminators, False)
                self.optimizer.zero_grad()
                with self.accelerator.autocast():
                    pred = self.model(img_input, mask_indice)
                g_losses = self.loss_fn.generator_loss(
                    pred, img_target, mask_indice, self.discriminators
                )
                self.accelerator.backward(g_losses['total'])
                self.optimizer.step()
                self._set_requires_grad(self.discriminators, True)

                log_dict = {
                    'loss_G': g_losses['total'].item(),
                    'syn': g_losses['syn'].item(),
                    'rec': g_losses['rec'].item(),
                    'adv_G': g_losses['adv'].item(),
                    'loss_D': disc_loss.item(),
                }

                if self.global_step % getattr(self.args, 'sample_every', 500) == 0 and self.accelerator.is_main_process:
                    with torch.no_grad():
                        n_show = min(4, B)
                        vis_inp = img[:n_show, 0, 0].unsqueeze(1).repeat(1, 3, 1, 1)
                        vis_out = self._zhang_to_image_range(pred[:n_show, 0, 0]).unsqueeze(1).repeat(1, 3, 1, 1)
                        comparison = torch.cat([vis_inp, vis_out], dim=0).cpu().float()
                        grid = make_grid(comparison, nrow=n_show, normalize=True, value_range=(0, 1))
                        grid_np = grid.clamp(0, 1).mul(255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        if self.use_swanlab:
                            self.accelerator.log({"samples": swanlab.Image(grid_np)}, step=self.global_step)
                        self.accelerator.print(f"Logged visualization samples for Zhang at step {self.global_step}")

            elif self.args.model_type == 'MMSyn':
                mask_indice = self.get_mask_indice(B, epoch)
                with self.accelerator.autocast():
                    pred_dict = self.model(img, mask_indice)
                    losses = self.loss_fn(pred_dict, img, mask_indice)

                self.optimizer.zero_grad()
                self.accelerator.backward(losses['total'])
                self.optimizer.step()

                log_dict = {
                    'total': losses['total'].item(),
                    'recon_fused': losses['recon_fused'].item(),
                    'recon_ind': losses['recon_ind'].item(),
                    'concat': losses['concat'].item(),
                }

                # Visualization: MRI GT / CT GT / CT predicted from fused latent.
                if self.global_step % getattr(self.args, 'sample_every', 500) == 0 and self.accelerator.is_main_process:
                    with torch.no_grad():
                        n_show = min(4, B)
                        mri_gt = img[:n_show, 0, 0].unsqueeze(1).repeat(1, 3, 1, 1)
                        ct_gt = img[:n_show, 1, 0].unsqueeze(1).repeat(1, 3, 1, 1)
                        ct_gen = pred_dict['fused'][:n_show, 1, 0].unsqueeze(1).clamp(0, 1).repeat(1, 3, 1, 1)
                        comparison = torch.cat([mri_gt, ct_gt, ct_gen], dim=0).cpu().float()
                        grid = make_grid(comparison, nrow=n_show, normalize=True, value_range=(0, 1))
                        grid_np = grid.clamp(0, 1).mul(255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        if self.use_swanlab:
                            self.accelerator.log({"samples": swanlab.Image(grid_np)}, step=self.global_step)
                        self.accelerator.print(f"Logged MMSyn samples (MRI GT, CT GT, CT Gen) at step {self.global_step}")

            elif self.args.model_type == 'MMGAN':
                mask_indice = self.get_mask_indice(B, epoch)
                img_4d = img.squeeze(2)  # [B, 2, H, W]

                # ── Update Discriminator ───────────────────────────────────
                self._set_requires_grad(self.discriminator, True)
                with self.accelerator.autocast():
                    fake_x = self.model(img_4d, mask_indice)
                self.optimizer_D.zero_grad()
                with self.accelerator.autocast():
                    d_losses = self.loss_fn.discriminator_loss(
                        fake_x.detach(), img_4d, mask_indice, self.discriminator
                    )
                self.accelerator.backward(d_losses['total'])
                self.optimizer_D.step()

                # ── Update Generator ───────────────────────────────────────
                self._set_requires_grad(self.discriminator, False)
                self.optimizer.zero_grad()
                with self.accelerator.autocast():
                    fake_x = self.model(img_4d, mask_indice)
                    g_losses = self.loss_fn.generator_loss(
                        fake_x, img_4d, mask_indice, self.discriminator
                    )
                self.accelerator.backward(g_losses['total'])
                self.optimizer.step()
                self._set_requires_grad(self.discriminator, True)

                log_dict = {
                    'loss_G': g_losses['total'].item(),
                    'gan': g_losses['gan'].item(),
                    'pixel': g_losses['pixel'].item(),
                    'synth': g_losses['synth'].item(),
                    'loss_D': d_losses['total'].item(),
                    'D_real': d_losses['real'].item(),
                    'D_fake': d_losses['fake'].item(),
                }

                # Visualization: MRI GT / CT GT / CT Gen
                if self.global_step % getattr(self.args, 'sample_every', 500) == 0 and self.accelerator.is_main_process:
                    with torch.no_grad():
                        n_show = min(4, B)
                        mri_gt = img_4d[:n_show, 0].unsqueeze(1).repeat(1, 3, 1, 1)
                        ct_gt = img_4d[:n_show, 1].unsqueeze(1).repeat(1, 3, 1, 1)
                        ct_gen = fake_x[:n_show, 1].unsqueeze(1).clamp(0, 1).repeat(1, 3, 1, 1)
                        comparison = torch.cat([mri_gt, ct_gt, ct_gen], dim=0).cpu().float()
                        grid = make_grid(comparison, nrow=n_show, normalize=True, value_range=(0, 1))
                        grid_np = grid.clamp(0, 1).mul(255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        if self.use_swanlab:
                            self.accelerator.log({"samples": swanlab.Image(grid_np)}, step=self.global_step)
                        self.accelerator.print(f"Logged MMGAN samples (MRI GT, CT GT, CT Gen) at step {self.global_step}")

            elif self.args.model_type == 'M2DN':
                mask_indice = self.get_mask_indice(B, epoch)
                with self.accelerator.autocast():
                    pred_x0, loss = self.model(img, mask_indices=mask_indice, mode='train')

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()
                log_dict = {'mse_loss': loss.item(), 'total_loss': loss.item()}

                # Visualization: show MRI GT / CT GT / CT predicted
                if self.global_step % getattr(self.args, 'sample_every', 500) == 0 and self.accelerator.is_main_process:
                    with torch.no_grad():
                        n_show = min(4, B)
                        mri_gt = img[:n_show, 0]
                        ct_gt  = img[:n_show, 1]
                        if mri_gt.dim() == 4: mri_gt = mri_gt[:, 0]
                        if ct_gt.dim()  == 4: ct_gt  = ct_gt[:, 0]
                        ct_pred = pred_x0[:n_show, 1, 0]
                        vis = torch.cat([mri_gt, ct_gt, ct_pred], dim=0)  # [3n, H, W]
                        vis = vis.unsqueeze(1).repeat(1, 3, 1, 1).cpu().float()
                        grid = make_grid(vis, nrow=n_show, normalize=True, value_range=(0, 1))
                        grid_np = grid.clamp(0, 1).mul(255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        if self.use_swanlab:
                            self.accelerator.log({"samples": swanlab.Image(grid_np)}, step=self.global_step)
                        self.accelerator.print(f"Logged M2DN samples (MRI GT / CT GT / CT Pred) at step {self.global_step}")

            if self.scheduler is not None:
                self.scheduler.step()
            if hasattr(self, 'scheduler_D') and self.scheduler_D is not None:
                self.scheduler_D.step()

            log_dict['lr'] = self.optimizer.param_groups[0]['lr']
            for k, v in log_dict.items():
                if k not in loss_meters: loss_meters[k] = AverageMeter()
                loss_meters[k].update(v, B)
            
            pbar.set_postfix(**{k: (f"{v.avg:.5f}" if k == "lr" else f"{v.avg:.4f}") for k, v in loss_meters.items()})

            if self.use_swanlab and self.accelerator.is_main_process:
                self.accelerator.log({f"train/iter_{k}": v for k, v in log_dict.items()}, step=epoch * len(self.train_loader) + idx)

        return {k: v.avg for k, v in loss_meters.items()}

    def validate(self, epoch, max_batches: Optional[int] = None):
        self.model.eval()
        self.accelerator.print(f"\n>>> Starting evaluation at epoch {epoch}")
        
        limit = self.args.max_steps if self.args.max_steps > 0 else 1e9
        
        final_metrics = {}
        
        # All models are evaluated with evaluate_reconstruction_distributed (PSNR / SSIM / rFID / IS)
        all_mod_metrics = {}
        autocast_kwargs = {"enabled": self.args.amp, "dtype": torch.float16 if self.args.amp else torch.float32}
        
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        
        for m in range(2):
            mod_name = "CT" if m == 1 else "MRI"
            self.accelerator.print(f"--- Validating Modality {mod_name} (m{m}) (Generation from others) ---")
            eval_dataset = self.eval_dataset_all
            eval_model = UnifiedEvalWrapper(
                self.accelerator.unwrap_model(self.model), 
                model_type=self.args.model_type, 
                target_modality=m, 
            )
            
            ref_path = os.path.join(self.args.logdir, f"ref_val_m{m}.npz")
            
            num_samples = len(eval_dataset)
            if self.args.max_steps > 0:
                num_samples = min(num_samples, self.args.max_steps * self.args.batch_size)

            m_metrics = evaluate_reconstruction_distributed(
                eval_model,
                eval_dataset,
                num_samples=num_samples,
                batch_size=self.args.batch_size,
                rank=rank,
                world_size=world_size,
                device=self.device,
                experiment_dir=self.args.logdir,
                global_step=self.global_step,
                autocast_kwargs=autocast_kwargs,
                reference_npz_path=ref_path,
                metrics_to_compute=("psnr", "ssim", "rfid", "is")
            )
            
            if self.accelerator.is_main_process and m_metrics is not None:
                for k, v in m_metrics.items():
                    all_mod_metrics[f"m{m}_{k}"] = v
        
        if self.accelerator.is_main_process:
            avg_metrics = {}
            for metric_name in ["psnr", "ssim", "rfid", "is"]:
                vals = [all_mod_metrics[f"m{m}_{metric_name}"] for m in range(2) if f"m{m}_{metric_name}" in all_mod_metrics]
                if vals:
                    avg_metrics[f"avg_{metric_name}"] = sum(vals) / len(vals)
            final_metrics = {**all_mod_metrics, **avg_metrics}
        
        # Broadcast final_metrics from rank 0 to all other ranks to ensure consistent decisions (e.g., save checkpoint)
        if self.accelerator.num_processes > 1:
            obj_list = [final_metrics]
            broadcast_object_list(obj_list, from_process=0)
            final_metrics = obj_list[0]
        
        return final_metrics

    def save_checkpoint(self, file_name, epoch, best_metric):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            state = {
                "epoch": epoch, 
                "best_metric": best_metric,
                "global_step": self.global_step,
                "state_dict": unwrapped_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            if hasattr(self, 'optimizer_D') and self.optimizer_D is not None:
                state["optimizer_D"] = self.optimizer_D.state_dict()
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                state["scheduler"] = self.scheduler.state_dict()
            if hasattr(self, 'scheduler_D') and self.scheduler_D is not None:
                state["scheduler_D"] = self.scheduler_D.state_dict()
                
            path = os.path.join(self.args.logdir, file_name)
            torch.save(state, path)
            print(f"Saved checkpoint: {path}")

    def save_state(self, path, epoch, best_metric):
        """
        Using accelerator.save_state to save all modules, optimizers, schedulers, and random states.
        Plus a custom metadata file for epoch and progress tracking.
        """
        self.accelerator.save_state(path)
        if self.accelerator.is_main_process:
            with open(os.path.join(path, "resume_config.json"), "w") as f:
                json.dump({
                    "epoch": epoch, 
                    "global_step": self.global_step, 
                    "best_metric": best_metric
                }, f)
            print(f"Saved accelerator state and resume_config to: {path}")

    def train(self):
        if not self.trainer_initialized:
            self.initial_trainer()

        if not self.args.test:
            for epoch in range(self.args.start_epoch, self.args.max_epochs):
                self.accelerator.print(f"\n{time.ctime()} Epoch: {epoch}")
                
                train_metrics = self.train_one_epoch(epoch)
                self.accelerator.print(f"Train Stats: {train_metrics}")
                
                if self.use_swanlab and self.accelerator.is_main_process:
                    self.accelerator.log({f"train/epoch_{k}": v for k, v in train_metrics.items()}, step=epoch)

                if (epoch + 1) % self.args.val_every == 0:
                    val_metrics = self.validate(epoch)
                    self.accelerator.print(f"Val Stats: {val_metrics}")
                    
                    # Model selection uses avg_psnr from the generative evaluation
                    metric_key = 'avg_psnr'
                    current_metric = val_metrics.get(metric_key, -1.0 if 'psnr' in metric_key else 1e9)
                    is_higher_better = 'psnr' in metric_key or 'ssim' in metric_key

                    if self.use_swanlab and self.accelerator.is_main_process:
                        self.accelerator.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)

                    improvement = (current_metric > self.best_metric) if is_higher_better else (current_metric < self.best_metric)
                    if improvement:
                        self.accelerator.print(f"New Best ({self.best_metric:.6f} --> {current_metric:.6f})")
                        self.best_metric = current_metric
                        self.save_checkpoint(f"ckpt_{self.args.model_type}_best.pt", epoch, self.best_metric)

                    self.save_checkpoint(f"ckpt_{self.args.model_type}_latest.pt", epoch, self.best_metric)
                    
                    self.save_state(os.path.join(self.args.logdir, "last_state"), epoch, self.best_metric)
        else:
            self.accelerator.print(">>> Test-only mode enabled. Skipping training loop.")

        self.test_scenarios()

        self.accelerator.end_training()

    def test_scenarios(self):
        self.accelerator.print("\n>>> Starting Final Scenarios Test (Comprehensive Evaluation)")
        self.model.eval()
        
        # Bidirectional MRI/CT evaluation scenarios
        scenarios = {
            "MRI_to_CT": [1, 0],
            "CT_to_MRI": [0, 1],
        }
        
        test_dataset = RAEBraTSDataset(
            json_path=self.args.data_json,
            phase='test', 
            transform=get_rae_transforms('test'),
            modality_index=None
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.args.batch_size, shuffle=False, 
            num_workers=self.args.workers, pin_memory=True
        )
        test_loader = self.accelerator.prepare(test_loader)
        
        all_results = {}
        autocast_kwargs = {"enabled": self.args.amp, "dtype": torch.float16 if self.args.amp else torch.float32}
        
        # Temp directory for distributed evaluation
        temp_dir = os.path.join(self.args.logdir, "test_scenarios_eval")
        if self.accelerator.is_main_process:
            os.makedirs(temp_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        for name, mask_template in scenarios.items():
            self.accelerator.print(f"\n--- Scenario: {name} {mask_template} ---")
            missing_indices = [i for i, v in enumerate(mask_template) if v == 0]
            
            # 1. Inference and Shard Saving
            recons = {m: [] for m in missing_indices}
            refs = {m: [] for m in missing_indices}
            
            for idx, (images, _) in enumerate(tqdm(test_loader, desc=f"Inference {name}", disable=not self.accelerator.is_local_main_process)):
                if self.args.max_steps > 0 and idx >= self.args.max_steps:
                    break
                B = images.size(0)
                mask = torch.tensor(mask_template).float().to(self.device).unsqueeze(0).repeat(B, 1)
                
                with torch.inference_mode():
                    with self.accelerator.autocast():
                        img_input = images.unsqueeze(2) if images.dim() == 4 else images
                        if self.args.model_type == 'ModImp':
                            pred, _, _ = self.model(img_input, mask)
                        elif self.args.model_type == 'MMT':
                            pred = self.model(img_input, mask)
                        elif self.args.model_type == 'Zhang':
                            pred = self._zhang_to_image_range(self.model(self._zhang_prepare_input(img_input, mask), mask))
                        elif self.args.model_type == 'M2DN':
                            pred = self.model(img_input, mask_indices=mask, mode='inference')
                        elif self.args.model_type == 'MMGAN':
                            # MMGAN forward returns [B, N, H, W].
                            pred = self.model(img_input.squeeze(2), mask)
                        elif self.args.model_type == 'MMSyn':
                            pred = self.model(img_input, mask)['fused']
                        else: continue

                for m in missing_indices:
                    if self.args.model_type == 'MMT':
                        # MMT generated order follows missing order
                        m_idx_in_pred = missing_indices.index(m)
                        pred_m = pred[:, m_idx_in_pred, 0]
                    elif self.args.model_type in ('Zhang', 'M2DN', 'MMSyn'):
                        pred_m = pred[:, m, 0]
                    elif self.args.model_type == 'MMGAN':
                        pred_m = pred[:, m]
                    else: # ModImp
                        pred_m = pred[:, m]
                    
                    gt_m = images[:, m]
                    
                    # Convert to uint8 [H, W, 3] for metrics
                    pred_np = pred_m.clamp(0, 1).mul(255).unsqueeze(-1).repeat(1, 1, 1, 3).to("cpu", dtype=torch.uint8).numpy()
                    gt_np = gt_m.clamp(0, 1).mul(255).unsqueeze(-1).repeat(1, 1, 1, 3).to("cpu", dtype=torch.uint8).numpy()
                    
                    for i in range(B):
                        recons[m].append(pred_np[i])
                        refs[m].append(gt_np[i])

            for m in missing_indices:
                shard_path = os.path.join(temp_dir, f"{name}_m{m}_rank{self.accelerator.process_index}.npz")
                np.savez(shard_path, recon=np.stack(recons[m]), ref=np.stack(refs[m]))
            
            self.accelerator.wait_for_everyone()

            # 2. Aggregate and Compute Metrics (Main Process)
            if self.accelerator.is_main_process:
                scenario_res = {"per_modality": {}, "overall": {}}
                all_psnrs, all_ssims, all_rfids, all_is = [], [], [], []
                
                for m in missing_indices:
                    combined_recons = []
                    combined_refs = []
                    for r in range(self.accelerator.num_processes):
                        shard_path = os.path.join(temp_dir, f"{name}_m{m}_rank{r}.npz")
                        with np.load(shard_path) as data:
                            combined_recons.append(data["recon"])
                            combined_refs.append(data["ref"])
                        try:
                            os.remove(shard_path)
                        except Exception:
                            pass
                    
                    combined_recons = np.concatenate(combined_recons, axis=0)
                    combined_refs = np.concatenate(combined_refs, axis=0)
                    
                    # Compute Metrics (PSNR, SSIM, rFID, IS)
                    from src.eval.ref_iqa import calculate_psnr, calculate_ssim
                    m_psnr = calculate_psnr(combined_refs, combined_recons, bs=64, device=str(self.device.type), disable_bar=True)
                    m_ssim = calculate_ssim(combined_refs, combined_recons, bs=64, device=str(self.device.type), disable_bar=True)
                    m_rfid = calculate_rfid(combined_refs, combined_recons, bs=64, device=str(self.device.type))
                    
                    try:
                        m_is = calculate_is(combined_recons, bs=64, device=str(self.device.type))
                    except Exception as e:
                        print(f"Error computing IS for m{m}: {e}")
                        m_is = 0.0
                    
                    mod_name = "CT" if m == 1 else "MRI"
                    scenario_res["per_modality"][mod_name] = {
                        "psnr": float(m_psnr), "ssim": float(m_ssim), "rfid": float(m_rfid), "is": float(m_is)
                    }
                    all_psnrs.append(m_psnr)
                    all_ssims.append(m_ssim)
                    all_rfids.append(m_rfid)
                    all_is.append(m_is)
                    self.accelerator.print(f"  Modality {mod_name}: PSNR={m_psnr:.4f}, SSIM={m_ssim:.4f}, rFID={m_rfid:.4f}, IS={m_is:.4f}")

                scenario_res["overall"] = {
                    "psnr": float(np.mean(all_psnrs)),
                    "ssim": float(np.mean(all_ssims)),
                    "rfid": float(np.mean(all_rfids)),
                    "is": float(np.mean(all_is))
                }
                all_results[name] = scenario_res
                self.accelerator.print(f"  Scenario {name} Avg: PSNR={scenario_res['overall']['psnr']:.4f}, rFID={scenario_res['overall']['rfid']:.4f}, IS={scenario_res['overall']['is']:.4f}")
            
            self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            save_path = os.path.join(self.args.logdir, f"test_results_{self.args.model_type}.json")
            with open(save_path, 'w') as f:
                json.dump(all_results, f, indent=4)
            print(f"\nFinal test results saved to {save_path}")
            
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if os.path.exists(temp_dir):
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Failed to cleanup temp_dir {temp_dir}: {e}")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type', type=str, required=True, choices=['MMT', 'M2DN', 'ModImp', 'Zhang', 'MMGAN', 'MMSyn'])
    
    parser.add_argument('--data_json', type=str, default='brats2023_distribution.json')
    parser.add_argument('--logdir', type=str, default=os.environ.get("OUTPUT_DIR", "./outputs"))
    parser.add_argument('--resume_path', '--checkpoint_path', dest='resume_path', type=str, default=None)
    
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--optim_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=1e-4) # For MMT D
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--decay', type=float, default=1e-5)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--sample_every', type=int, default=500, help='Visualization interval (steps)')
    parser.add_argument('--reference_npz_path', type=str, default=None, help='Reference NPZ for rFID calculation')
    parser.add_argument('--test', action='store_true', help='Test only mode')
    
    parser.add_argument('--T_0', type=int, default=20)
    parser.add_argument('--eta_min', type=float, default=1e-6)
    
    parser.add_argument('--max_steps', type=int, default=-1, help='Limit steps per epoch for debugging')
    
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=24)
    parser.add_argument('--depths', nargs='+', type=int, default=[2, 2, 2, 2])
    parser.add_argument('--num_heads', nargs='+', type=int, default=[3, 6, 12, 24])
    parser.add_argument('--con_dim', type=int, default=512)
    parser.add_argument('--mask_ratio_spa', type=float, default=0.5)
    parser.add_argument('--alpha_lmse', type=float, default=1.0)
    parser.add_argument('--alpha_lcon', type=float, default=0.3)
    parser.add_argument('--alpha_lconsis', type=float, default=0.5)

    # M2DN specific args
    parser.add_argument('--m2dn_enc_ch',      type=int, default=32,   help='M2DN: encoder output channels per modality')
    parser.add_argument('--m2dn_t_emb_dim',   type=int, default=256,  help='M2DN: timestep embedding dimension')
    parser.add_argument('--m2dn_T',           type=int, default=1000, help='M2DN: total diffusion timesteps')
    parser.add_argument('--m2dn_n_inf_steps', type=int, default=18,   help='M2DN: DDIM inference steps')
    parser.add_argument('--m2dn_setting', type=str, default='paper', choices=['paper', 'default'],
                        help='M2DN preset. `paper` uses Adam(betas=0.9/0.999), zero weight decay, and constant LR.')

    # MMGAN specific args
    parser.add_argument('--mmgan_z_type', type=str, default='noise', choices=['noise', 'zeros', 'average'],
                        help='MMGAN: how to impute the missing-modality input channels for G.')
    parser.add_argument('--mmgan_ic', type=int, default=1, choices=[0, 1],
                        help='MMGAN: 1 enables Implicit Conditioning (paper Section IV-B).')
    parser.add_argument('--mmgan_lam', type=float, default=0.9,
                        help='MMGAN: weight on L_pixel in (1-lam)*L_GAN + lam*L_pixel.')
    parser.add_argument('--mmgan_b1', type=float, default=0.5, help='MMGAN: Adam beta1.')
    parser.add_argument('--mmgan_b2', type=float, default=0.999, help='MMGAN: Adam beta2.')
    parser.add_argument('--mmgan_use_tanh', action='store_true',
                        help='MMGAN: use Tanh head (set when training in [-1, 1]).')

    # MMSyn specific args
    parser.add_argument('--mmsyn_latent_dim', type=int, default=16)
    parser.add_argument('--mmsyn_fusion', type=str, default='max', choices=['max', 'ave', 'hemis'])
    parser.add_argument('--mmsyn_lambda_recon', type=float, default=1.0)
    parser.add_argument('--mmsyn_lambda_concat', type=float, default=1.0)
    parser.add_argument('--mmsyn_ind_outs', type=int, default=1, choices=[0, 1])
    parser.add_argument('--mmsyn_fuse_outs', type=int, default=1, choices=[0, 1])

    args = parser.parse_args(argv)

    if args.model_type == 'M2DN' and args.m2dn_setting == 'paper':
        args.betas = (0.9, 0.999)
        args.decay = 0.0

    if args.model_type == 'MMGAN':
        # Paper defaults: Adam(lr=2e-4, betas=(0.5, 0.999)).
        if args.optim_lr == 1e-4:
            args.optim_lr = 2e-4
        if args.d_lr == 1e-4:
            args.d_lr = 2e-4

    args.logdir = os.path.join(args.logdir, args.model_type)
        
    os.makedirs(args.logdir, exist_ok=True)
    
    trainer = UnifiedTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

try:
    from src.brats_dataset import get_loader, BraTS2023Dataset, get_transforms
    
    from network.ModImp.modimp import ImputationModel
    from network.MMT.mmt import MMT
    from network.M2DN.m2dn import M2DN
    from network.Zhang.zhang import ZhangModel, zhang_to_image_range, zhang_to_model_range
    from network.MMGAN.mmgan import MMGAN
    from network.MMSyn.mmsyn import MMSyn
except ImportError as e:
    print(f"Import Error: {e}")

try:
    from src.eval.fid import calculate_is
    from src.eval.ref_iqa import calculate_psnr, calculate_ssim
    from src.eval import calculate_rfid
except ImportError as e:
    print(f"Import Error for Eval Metrics: {e}")
    calculate_is = None
    calculate_psnr = None
    calculate_ssim = None
    calculate_rfid = None

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed, DistributedDataParallelKwargs
except ImportError:
    Accelerator = None
    set_seed = None
    DistributedDataParallelKwargs = None


class ShardSampler(Sampler[int]):
    """
    Distributed sampler without padding (PyTorch's DistributedSampler pads with duplicates when drop_last=False).
    - indices: [rank, rank+world_size, rank+2*world_size, ...]
    - for validation / test (no shuffle)
    """
    def __init__(self, dataset_len: int, num_replicas: int, rank: int):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset_len = int(dataset_len)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, self.dataset_len, self.num_replicas))

    def __len__(self):
        if self.dataset_len <= self.rank:
            return 0
        return int(math.ceil((self.dataset_len - self.rank) / self.num_replicas))


class UnifiedTester:
    def __init__(self, args, accelerator):
        self.args = args
        self.accelerator = accelerator
        self.device = self.accelerator.device
        
        self._init_test_scenarios()
        
        self.init_model()
        self.load_checkpoint()
        
        self.test_loader, self.test_dataset_len = self._build_test_loader()
        self.accelerator.print(f"Test dataset size: {self.test_dataset_len} | num_processes={self.accelerator.num_processes}")

        self.model = self.accelerator.prepare(self.model)
        self.model.eval()

    def _build_test_loader(self):
        """
        Single process: get_loader.
        Multiple processes (accelerate launch): ShardSampler, no duplicated samples.
        """
        if self.accelerator.num_processes <= 1:
            loader = get_loader(
                json_path=self.args.data_json,
                batch_size=self.args.batch_size,
                num_workers=self.args.workers,
                phase='test'
            )
            try:
                dataset_len = len(loader.dataset)
            except Exception:
                dataset_len = len(loader) * self.args.batch_size
            return loader, dataset_len

        transforms = get_transforms("test")
        dataset = BraTS2023Dataset(json_path=self.args.data_json, phase="test", transform=transforms)
        sampler = ShardSampler(
            dataset_len=len(dataset),
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            sampler=sampler,
            num_workers=self.args.workers,
            pin_memory=True,
        )
        return loader, len(dataset)

    def _init_test_scenarios(self):
        # Order: T1, T1c, T2, T2f
        self.scenarios = {
            # 1 Modality Available
            "T1_only":        [1, 0, 0, 0],
            "T1c_only":       [0, 1, 0, 0],
            "T2_only":        [0, 0, 1, 0],
            "T2f_only":       [0, 0, 0, 1],
            
            # 2 Modalities Available
            "T1_T1c":         [1, 1, 0, 0],
            "T1_T2":          [1, 0, 1, 0],
            "T1_T2f":         [1, 0, 0, 1],
            "T1c_T2":         [0, 1, 1, 0],
            "T1c_T2f":        [0, 1, 0, 1],
            "T2_T2f":         [0, 0, 1, 1],
            
            # 3 Modalities Available
            "T1_T1c_T2":      [1, 1, 1, 0],
            "T1_T1c_T2f":     [1, 1, 0, 1],
            "T1_T2_T2f":      [1, 0, 1, 1],
            "T1c_T2_T2f":     [0, 1, 1, 1],
            
            # All Modalities Available
        }

    def init_model(self):
        self.accelerator.print(f">> Initializing Model: {self.args.model_type}")
        
        if self.args.model_type == 'ModImp':
            self.model = ImputationModel(
                img_size=self.args.img_size, patch_size=self.args.patch_size,
                embed_dim=self.args.embed_dim, depths=self.args.depths,
                num_heads=self.args.num_heads, con_dim=self.args.con_dim,
                mask_ratio_spa=self.args.mask_ratio_spa
            ).to(self.device)
            
        elif self.args.model_type == 'MMT':
            self.model = MMT(patch_size=4, dim=6, depths=[2,2,2,2], num_heads=[3,6,12,24]).to(self.device)
            
        elif self.args.model_type == 'Zhang':
            self.model = ZhangModel(num_mods=4).to(self.device)

        elif self.args.model_type == 'MMGAN':
            with_tanh = getattr(self.args, 'mmgan_use_tanh', False)
            self.model = MMGAN(
                num_mods=4,
                z_type=getattr(self.args, 'mmgan_z_type', 'noise'),
                use_ic=bool(getattr(self.args, 'mmgan_ic', 1)),
                with_tanh=with_tanh,
                with_relu=not with_tanh,
            ).to(self.device)

        elif self.args.model_type == 'MMSyn':
            self.model = MMSyn(
                num_mods=4,
                latent_dim=getattr(self.args, 'mmsyn_latent_dim', 16),
                in_channels=1,
                out_channels=1,
                fusion=getattr(self.args, 'mmsyn_fusion', 'max'),
            ).to(self.device)

        elif self.args.model_type == 'M2DN':
            self.model = M2DN(
                num_modalities=4,
                img_channels=1,
                enc_channels=getattr(self.args, 'm2dn_enc_ch', 32),
                t_emb_dim=getattr(self.args, 'm2dn_t_emb_dim', 256),
                T=getattr(self.args, 'm2dn_T', 1000),
                n_inf_steps=getattr(self.args, 'm2dn_n_inf_steps', 18),
            ).to(self.device)

    def load_checkpoint(self):
        if not self.args.checkpoint_path:
            raise ValueError("Please provide --checkpoint_path")
            
        self.accelerator.print(f"Loading checkpoint from {self.args.checkpoint_path}")
        # Load weights on CPU first (every process loads its own copy)
        checkpoint = torch.load(self.args.checkpoint_path, map_location="cpu", weights_only=False)
        
        # Handle state_dict key mismatch (e.g. from DDP)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '').replace('_orig_mod.', '')
            new_state_dict[new_key] = v
            
        # strict=False allows loading even if some keys are missing/extra.
        self.model.load_state_dict(new_state_dict, strict=False)

    def get_fixed_mask(self, batch_size, mask_template):
        mask = torch.tensor(mask_template, dtype=torch.float32, device=self.device)
        mask = mask.unsqueeze(0).repeat(batch_size, 1)
        return mask
    
    def run_inference(self, img, mask):
        if self.args.model_type == 'ModImp':
            x_rec, _, _ = self.model(img, mask)
            return x_rec
            
        elif self.args.model_type == 'MMT':
            y_gen = self.model(img, mask)
            return y_gen
        
        elif self.args.model_type == 'M2DN':
            x_pred = self.model(img, mask)
            return x_pred

        elif self.args.model_type == 'Zhang':
            img_masked = zhang_to_model_range(img) * mask.view(img.size(0), 4, 1, 1, 1)
            pred = self.model(img_masked, mask)
            return zhang_to_image_range(pred).clamp(0.0, 1.0)

        elif self.args.model_type == 'MMGAN':
            # MMGAN works in 4D: [B, N, H, W] in / out.
            img_4d = img.squeeze(2) if img.dim() == 5 else img
            return self.model(img_4d, mask)

        elif self.args.model_type == 'MMSyn':
            return self.model(img, mask)['fused']

        return None

    def _reduce_stats_sum(self, stats: torch.Tensor) -> torch.Tensor:
        """
        stats: shape [20] = [psnr_sum, ssim_sum, rmse_sum, cnt] + 4*(psnr/ssim/rmse/cnt)
        """
        if self.accelerator.num_processes > 1:
            # accelerate.reduce is missing in some versions; fall back to torch.distributed
            if hasattr(self.accelerator, "reduce"):
                stats = self.accelerator.reduce(stats, reduction="sum")
            else:
                import torch.distributed as dist
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats

    def evaluate_scenario(self, scenario_name, target_mask):
        pbar = tqdm(
            self.test_loader,
            desc=f"Testing {scenario_name} {target_mask}",
            disable=not self.accelerator.is_main_process,
        )

        missing_indices = [i for i, v in enumerate(target_mask) if v == 0]
        
        local_gen_imgs_m = [[] for _ in range(4)]
        local_gt_imgs_m = [[] for _ in range(4)]
        
        for batch_data in pbar:
            images, _ = batch_data
            img = images.unsqueeze(2).to(self.device)
            B = img.size(0)
            
            mask = self.get_fixed_mask(B, target_mask)
            
            with torch.no_grad():
                pred = self.run_inference(img, mask)
            
            if torch.all(mask == 1):
                continue 

            for m in missing_indices:
                if self.args.model_type in ['Zhang', 'M2DN', 'MMSyn']:
                    pred_m = pred[:, m, 0]
                elif self.args.model_type in ['ModImp']:
                    pred_m = pred[:, m]
                elif self.args.model_type in ['MMT']:
                    # For MMT, y_gen is generated in the order of missing indices
                    m_idx_in_pred = missing_indices.index(m)
                    pred_m = pred[:, m_idx_in_pred, 0]
                elif self.args.model_type == 'MMGAN':
                    pred_m = pred[:, m]
                else:
                    continue
                
                gt_m = img[:, m, 0] if img.dim() == 5 else img[:, m]
                
                pred_np = pred_m.clamp(0, 1).mul(255).unsqueeze(-1).repeat(1, 1, 1, 3).to("cpu", dtype=torch.uint8).numpy()
                gt_np = gt_m.clamp(0, 1).mul(255).unsqueeze(-1).repeat(1, 1, 1, 3).to("cpu", dtype=torch.uint8).numpy()
                
                for i in range(B):
                    local_gen_imgs_m[m].append(pred_np[i])
                    local_gt_imgs_m[m].append(gt_np[i])

        img_size = self.args.img_size if hasattr(self.args, 'img_size') else 256
        
        temp_dir = os.path.join(self.args.output_dir, "eval_npzs")
        if self.accelerator.is_main_process:
            os.makedirs(temp_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes
        shard_path = os.path.join(temp_dir, f"{scenario_name}_rank{rank:02d}.npz")
        
        save_dict = {}
        for m in missing_indices:
            save_dict[f'gen_m{m}'] = np.stack(local_gen_imgs_m[m]) if len(local_gen_imgs_m[m]) > 0 else np.zeros((0, img_size, img_size, 3), dtype=np.uint8)
            save_dict[f'gt_m{m}'] = np.stack(local_gt_imgs_m[m]) if len(local_gt_imgs_m[m]) > 0 else np.zeros((0, img_size, img_size, 3), dtype=np.uint8)
            
        np.savez(shard_path, **save_dict)
        self.accelerator.wait_for_everyone()

        scenario_res = {"per_modality": {}, "overall": {}}
        all_psnrs, all_ssims, all_rfids, all_is = [], [], [], []

        if self.accelerator.is_main_process:
            self.accelerator.print(f"Computing Metrics for {scenario_name}...")
            
            for m in missing_indices:
                combined_recons = []
                combined_refs = []
                for r in range(world_size):
                    sp = os.path.join(temp_dir, f"{scenario_name}_rank{r:02d}.npz")
                    if os.path.exists(sp):
                        data = np.load(sp)
                        if len(data[f'gen_m{m}']) > 0: combined_recons.append(data[f'gen_m{m}'])
                        if len(data[f'gt_m{m}']) > 0: combined_refs.append(data[f'gt_m{m}'])

                if len(combined_recons) > 0:
                    combined_recons = np.concatenate(combined_recons, axis=0)
                    combined_refs = np.concatenate(combined_refs, axis=0)
                    
                    m_psnr, m_ssim, m_rfid, m_is = 0.0, 0.0, 0.0, 0.0
                    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    if calculate_psnr is not None:
                        m_psnr = float(calculate_psnr(combined_refs, combined_recons, bs=self.args.batch_size, device=compute_device, disable_bar=True))
                        m_ssim = float(calculate_ssim(combined_refs, combined_recons, bs=self.args.batch_size, device=compute_device, disable_bar=True))
                        m_rfid = float(calculate_rfid(combined_refs, combined_recons, bs=self.args.batch_size, device=compute_device))
                    
                    if calculate_is is not None:
                        try:
                            m_is = float(calculate_is(combined_recons, bs=self.args.batch_size, device=compute_device))
                        except Exception as e:
                            print(f"Error computing IS for m{m}: {e}")
                    
                    scenario_res["per_modality"][f"m{m}"] = {
                        "PSNR": m_psnr, "SSIM": m_ssim, "rFID": m_rfid, "IS": m_is, "count": len(combined_recons)
                    }
                    all_psnrs.append(m_psnr)
                    all_ssims.append(m_ssim)
                    all_rfids.append(m_rfid)
                    all_is.append(m_is)

            for r in range(world_size):
                sp = os.path.join(temp_dir, f"{scenario_name}_rank{r:02d}.npz")
                if os.path.exists(sp):
                    os.remove(sp)
            
            if len(missing_indices) > 0:
                scenario_res["overall"] = {
                    "PSNR": float(np.mean(all_psnrs)) if all_psnrs else 0.0,
                    "SSIM": float(np.mean(all_ssims)) if all_ssims else 0.0,
                    "rFID": float(np.mean(all_rfids)) if all_rfids else 0.0,
                    "IS": float(np.mean(all_is)) if all_is else 0.0,
                }
            else:
                scenario_res["overall"] = {"PSNR": 0.0, "SSIM": 0.0, "rFID": 0.0, "IS": 0.0}
                
        self.accelerator.wait_for_everyone()
        
        try:
            from accelerate.utils import broadcast_object_list
            obj_list = [scenario_res]
            broadcast_object_list(obj_list, from_process=0)
            scenario_res = obj_list[0]
        except Exception:
            pass
            
        return scenario_res

    def run(self):
        self.accelerator.print("\n=== Starting Full Modality Combination Evaluation ===")
        self.accelerator.print(f"Model: {self.args.model_type}")
        results = {}
        
        for name, mask_template in self.scenarios.items():
            self.accelerator.print(f"\nEvaluating: {name} | Mask: {mask_template}")
            metrics = self.evaluate_scenario(name, mask_template)
            results[name] = {
                "mask": mask_template,
                "metrics": metrics
            }
            o = metrics.get("overall", {})
            self.accelerator.print(f"Result [{name}]: PSNR={o.get('PSNR', 0.0):.4f}, SSIM={o.get('SSIM', 0.0):.4f}, rFID={o.get('rFID', 0.0):.4f}, IS={o.get('IS', 0.0):.4f}")
            self.save_results(results, final=False)
            
        self.save_results(results, final=True)

    def save_results(self, results, final=True):
        if not self.accelerator.is_main_process:
            return
        save_path = os.path.join(self.args.output_dir, f"full_test_results_{self.args.model_type}.json")
        tmp_path = f"{save_path}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(results, f, indent=4)
        os.replace(tmp_path, save_path)
        prefix = "Results saved" if final else "Partial results updated"
        print(f"\n{prefix} to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type', type=str, required=True, choices=['MMT', 'M2DN', 'ModImp', 'Zhang', 'MMGAN', 'MMSyn'])
    parser.add_argument('--data_json', type=str, default='brats2023_distribution.json')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained checkpoint')
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.environ.get("OUTPUT_DIR", "./outputs"), "eval"))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action='store_true', help='Enable fp16 autocast (accelerate mixed_precision=fp16)')
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--workers', type=int, default=4)
    
    # ModImp Params
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=24)
    parser.add_argument('--depths', nargs='+', type=int, default=[2, 2, 2, 2])
    parser.add_argument('--num_heads', nargs='+', type=int, default=[3, 6, 12, 24])
    parser.add_argument('--con_dim', type=int, default=512)
    parser.add_argument('--mask_ratio_spa', type=float, default=0.5)

    # M2DN specific
    parser.add_argument('--m2dn_enc_ch', type=int, default=32)
    parser.add_argument('--m2dn_t_emb_dim', type=int, default=256)
    parser.add_argument('--m2dn_T', type=int, default=1000)
    parser.add_argument('--m2dn_n_inf_steps', type=int, default=18)

    # MMGAN specific
    parser.add_argument('--mmgan_z_type', type=str, default='noise', choices=['noise', 'zeros', 'average'])
    parser.add_argument('--mmgan_ic', type=int, default=1, choices=[0, 1])
    parser.add_argument('--mmgan_use_tanh', action='store_true')

    # MMSyn specific
    parser.add_argument('--mmsyn_latent_dim', type=int, default=16)
    parser.add_argument('--mmsyn_fusion', type=str, default='max', choices=['max', 'ave', 'hemis'])

    args = parser.parse_args()

    if Accelerator is None:
        raise ImportError("accelerate is not installed: `pip install accelerate`, then run with `accelerate launch baseline/src/test_brats.py ...`.")

    kwargs_handlers = []
    if args.model_type in ("MMT", "MMGAN") and DistributedDataParallelKwargs is not None:
        kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))

    accelerator = Accelerator(
        mixed_precision="fp16" if args.amp else "no",
        kwargs_handlers=kwargs_handlers,
    )
    if set_seed is not None:
        set_seed(args.seed, device_specific=False)
    else:
        torch.manual_seed(args.seed)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    tester = UnifiedTester(args, accelerator)
    tester.run()

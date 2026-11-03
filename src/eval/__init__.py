from .ref_iqa import calculate_psnr, calculate_lpips, calculate_ssim
from .fid import calculate_rfid, calculate_gfid, calculate_is, get_inception_moments, calculate_ifid
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Dict, Optional
import os
import sys
def compute_reconstruction_metrics(
    ref_arr: np.ndarray,
    rec_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    metrics_to_compute = ("psnr", "ssim", "rfid", "lpips"),
    disable_bar: bool = True
) -> Dict[str, float]:
    """
    Compute reconstruction metrics between reference and reconstructed images.

    Args:
        ref_arr: Reference images [N, H, W, C] uint8
        rec_arr: Reconstructed images [N, H, W, C] uint8
        device: Device for computation
        batch_size: Batch size for metric computation

    Returns:
        Dictionary with metrics: eval/psnr, eval/ssim, eval/rfid, eval/lpips
    """
    device_str = "cuda" if device.type == "cuda" else "cpu"
    results_dict = {}
    if 'psnr' in metrics_to_compute:
        psnr = calculate_psnr(ref_arr, rec_arr, batch_size, device_str, disable_bar=disable_bar)
        results_dict["psnr"] = psnr
    if 'ssim' in metrics_to_compute:
        ssim = calculate_ssim(ref_arr, rec_arr, batch_size, device_str, disable_bar=disable_bar)
        results_dict["ssim"] = ssim
    if 'rfid' in metrics_to_compute:
        rfid = calculate_rfid(ref_arr, rec_arr, batch_size, device_str)
        results_dict["rfid"] = rfid
    if 'lpips' in metrics_to_compute:
        lpips = calculate_lpips(ref_arr, rec_arr, batch_size, device_str, disable_bar=disable_bar)
        results_dict["lpips"] = lpips
    if 'is' in metrics_to_compute:
        is_score = calculate_is(rec_arr, batch_size, device_str)
        results_dict["is"] = is_score
    assert len(results_dict) > 0, "No metrics were computed."
    return results_dict
def compute_generation_metrics(
    ref_arr,
    rec_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    metrics_to_compute = ("psnr", "ssim", "rfid", "is")
):
    """PSNR, SSIM, rFID and IS, as in Stage 1. ref_arr: reference images [N,H,W,C] uint8, or a dict with mu/sigma (gFID + IS only)."""
    device_str = "cuda" if device.type == "cuda" else "cpu"
    metrics = {}
    
    # Handle the case where ref_arr is a dict (gFID + IS case)
    if isinstance(ref_arr, dict) and "mu" in ref_arr and "sigma" in ref_arr:
        if 'gfid' in metrics_to_compute:
            metrics["gfid"] = calculate_gfid(rec_arr, ref_arr, batch_size, device_str)
        if 'is' in metrics_to_compute:
            metrics["is"] = calculate_is(rec_arr, batch_size, device_str)
        return metrics

    ref_images = np.asarray(ref_arr)
    # If no valid reference images provided, only compute Inception Score
    if ref_images.ndim != 4 or ref_images.shape[0] != rec_arr.shape[0]:
        if 'is' in metrics_to_compute:
            metrics["is"] = calculate_is(rec_arr, batch_size, device_str)
        return metrics

    if 'psnr' in metrics_to_compute:
        metrics["psnr"] = calculate_psnr(ref_images, rec_arr, batch_size, device_str, disable_bar=True)
    if 'ssim' in metrics_to_compute:
        metrics["ssim"] = calculate_ssim(ref_images, rec_arr, batch_size, device_str, disable_bar=True)
    if 'rfid' in metrics_to_compute:
        metrics["rfid"] = calculate_rfid(ref_images, rec_arr, batch_size, device_str)
    if 'is' in metrics_to_compute:
        metrics["is"] = calculate_is(rec_arr, batch_size, device_str)
    if 'lpips' in metrics_to_compute:
        metrics["lpips"] = calculate_lpips(ref_images, rec_arr, batch_size, device_str, disable_bar=True)
        
    return metrics
@torch.no_grad()
def evaluate_generation_distributed(
    model_fn,
    sample_fn,
    latent_size, # for noise 
    additional_model_kwargs,
    use_guidance: bool,
    rae, 
    val_dataset,
    num_samples: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    experiment_dir: str,
    global_step: int,
    autocast_kwargs: dict,
    metric_batch_size: int = 128,
    reference_npz_path: Optional[str] = None,
    use_auxiliary_init: bool = False,
    null_label: int = 0,
    image_size: Optional[int] = None,
    metrics_to_compute = ("psnr", "ssim", "rfid", "is"),
    num_workers: int = 4
) -> Optional[Dict[str, float]]:
    """
    Evaluate reconstruction metrics using all GPUs in a distributed manner.

    Args:
        val_dataset: Validation dataset
        batch_size: Batch size per GPU for reconstruction
        rank: Current GPU rank
        world_size: Total number of GPUs
        device: Device to use
        experiment_dir: Experiment directory
        global_step: Current training step
        autocast_kwargs: Autocast configuration
        metric_batch_size: Batch size for metric computation (on rank 0)
        reference_npz_path: Optional path to existing reference NPZ file

    Returns:
        Dictionary of metrics (only on rank 0, None on other ranks)
    """
    # model.eval()
    # Save shard NPZ
    temp_dir = os.path.join(experiment_dir, "eval_npzs")
    if rank == 0:
        print(f"\n[Eval] Starting distributed sampling evaluation at step {global_step}")
        os.makedirs(temp_dir, exist_ok=True)

    # Wait for rank 0 to create the directory before other ranks try to save
    if world_size > 1:
        dist.barrier()
    # print(f"[Rank {rank}] Starting sampling...")
    # Each rank processes its shard
    N = min(len(val_dataset), num_samples)
    chunk = N // world_size

    if rank < world_size - 1:
        start = rank * chunk
        end   = (rank + 1) * chunk
    else:
        # Last rank takes the remainder (and handles N < world_size gracefully)
        start = rank * chunk
        end   = N

    rank_indices = list(range(start, end))
    subset = Subset(val_dataset, rank_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        drop_last=False,
    )

    # Reconstruct images on this rank
    generations = []
    references = [] if use_auxiliary_init else None
    iterator = tqdm(loader, desc=f"[Rank {rank}] Sampling", file=sys.stdout) if rank == 0 else loader

    with torch.inference_mode():
        if use_auxiliary_init:
            for batch in iterator:
                # Handle both (x_a, x_b) and (x_a, x_b, extra)
                x_a, x_b = batch[0], batch[1]
                x_a = x_a.to(device, non_blocking=True)
                x_b = x_b.to(device, non_blocking=True)
                if image_size is not None and (x_a.shape[-1] != image_size or x_a.shape[-2] != image_size):
                    x_a = torch.nn.functional.interpolate(
                        x_a, size=(image_size, image_size), mode="bilinear", align_corners=False
                    )
                    x_b = torch.nn.functional.interpolate(
                        x_b, size=(image_size, image_size), mode="bilinear", align_corners=False
                    )
                
                # If x_b is already a latent (encoded in dataset), don't re-encode
                if x_b.ndim == 4 and x_b.shape[1] == rae.encoder.patch_embed.proj.out_channels if hasattr(rae, 'encoder') else False:
                    z_b = x_b
                else:
                    z_b = rae.encode(x_b)
                
                n = z_b.size(0)
                model_kwargs = dict(**additional_model_kwargs)
                if "y" not in model_kwargs:
                    model_kwargs["y"] = torch.zeros(n, dtype=torch.long, device=device)
                
                with autocast(**autocast_kwargs):
                    samples = sample_fn(z_b, model_fn, **model_kwargs)[-1]
                    samples = rae.decode(samples).clamp(0, 1)
                gen_np = samples.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                ref_np = x_a.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                for img in gen_np:
                    generations.append(img)
                for img in ref_np:
                    references.append(img)
        else:
            for _, label in iterator:
                n = label.size(0)
                z = torch.randn(n, *latent_size, device=device)
                y = label.to(device)
                if use_guidance:
                    z = torch.cat([z, z], dim=0)
                    y_null = torch.full((n,), null_label, device=device)
                    y = torch.cat([y, y_null], dim=0)
                model_kwargs = dict(y=y, **additional_model_kwargs)
                with autocast(**autocast_kwargs):
                    samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                    if use_guidance:
                        samples = samples.chunk(2, dim=0)
                    samples = rae.decode(samples).clamp(0, 1)
                gen_np = samples.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                for img in gen_np:
                    generations.append(img)

    generations = np.stack(generations)
    shard_path = os.path.join(temp_dir, f"gen_{global_step:07d}_{rank:02d}.npz")
    if use_auxiliary_init:
        references = np.stack(references)
        np.savez(shard_path, arr_0=generations, ref=references)
    else:
        np.savez(shard_path, arr_0=generations)

    if rank == 0:
        print(f"[Rank {rank}] Saved {len(generations)} generation to {shard_path}")

    # Wait for all ranks to finish generation
    if world_size > 1:
        dist.barrier()

    # Rank 0 computes metrics
    metrics = None
    if rank == 0:
        ref_stats = None
        # Combine all generation shards
        all_gens = []
        for r in range(world_size):
            shard_file = os.path.join(temp_dir, f"gen_{global_step:07d}_{r:02d}.npz")
            shard_data = np.load(shard_file)["arr_0"]
            all_gens.append(shard_data)

        combined_recons = np.concatenate(all_gens, axis=0)

        # Load reference: from dataset (use_auxiliary_init) or from NPZ
        device_str = "cuda" if device.type == "cuda" else "cpu"
        if use_auxiliary_init:
            all_refs = []
            for r in range(world_size):
                shard_file = os.path.join(temp_dir, f"gen_{global_step:07d}_{r:02d}.npz")
                shard_data = np.load(shard_file)
                all_refs.append(shard_data["ref"])
            ref_images = np.concatenate(all_refs, axis=0)
            print(f"[Eval] Using reference from auxiliary-init dataset, shape: {ref_images.shape}")
        else:
            if not os.path.exists(reference_npz_path):
                raise FileNotFoundError(f"Reference NPZ not found at {reference_npz_path}")
            ref_stats = np.load(reference_npz_path)
            print(f"[Eval] Loaded reference NPZ from {reference_npz_path}")
            ref_images = ref_stats["arr_0"] if "arr_0" in ref_stats.files else None
        
        if ref_images is not None:
            final_n = min(len(ref_images), len(combined_recons), num_samples)
            ref_images = ref_images[:final_n]
            combined_recons = combined_recons[:final_n]
            print(f"[Eval] Final size for metrics: {final_n}")

        print(f"[Eval] Computing metrics ({', '.join(metrics_to_compute)})...")
        if ref_images is not None and isinstance(ref_images, np.ndarray) and ref_images.ndim == 4:
            metrics = compute_generation_metrics(
                ref_images,
                combined_recons,
                device,
                metric_batch_size,
                metrics_to_compute=metrics_to_compute
            )
        else:
            has_ref_moments = ref_stats is not None and hasattr(ref_stats, "files") and "mu" in ref_stats.files and "sigma" in ref_stats.files
            if has_ref_moments:
                metrics = {}
                if 'gfid' in metrics_to_compute:
                    metrics["gfid"] = calculate_gfid(combined_recons, ref_stats, metric_batch_size, device_str)
                if 'is' in metrics_to_compute:
                    metrics["is"] = calculate_is(combined_recons, metric_batch_size, device_str)
            else:
                metrics = {}
                if 'is' in metrics_to_compute:
                    metrics["is"] = calculate_is(combined_recons, metric_batch_size, device_str)

        # Print results
        print(f"[Eval] Step {global_step} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # Cleanup reconstruction shards
        for r in range(world_size):
            shard_file = os.path.join(temp_dir, f"gen_{global_step:07d}_{r:02d}.npz")
            if os.path.exists(shard_file):
                os.remove(shard_file)

    if world_size > 1:
        dist.barrier()
    return metrics
@torch.no_grad()
def evaluate_reconstruction_distributed(
    model,
    val_dataset,
    num_samples: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    experiment_dir: str,
    global_step: int,
    autocast_kwargs: dict,
    metric_batch_size: int = 128,
    reference_npz_path: Optional[str] = None,
    metrics_to_compute: Optional[list] = ("psnr", "ssim", "rfid"),
    train_dataset=None,
    max_train_samples: int = 5000,
) -> Optional[Dict[str, float]]:
    """
    Evaluate reconstruction metrics using all GPUs in a distributed manner.

    Args:
        model: Model to evaluate (should be in eval mode)
        val_dataset: Validation dataset
        batch_size: Batch size per GPU for reconstruction
        rank: Current GPU rank
        world_size: Total number of GPUs
        device: Device to use
        experiment_dir: Experiment directory
        global_step: Current training step
        autocast_kwargs: Autocast configuration
        metric_batch_size: Batch size for metric computation (on rank 0)
        reference_npz_path: Optional path to existing reference NPZ file

    Returns:
        Dictionary of metrics (only on rank 0, None on other ranks)
    """
    # model.eval()
    # Save shard NPZ
    temp_dir = os.path.join(experiment_dir, "eval_npzs")
    if rank == 0:
        print(f"\n[Eval] Starting distributed reconstruction evaluation at step {global_step}")
        os.makedirs(temp_dir, exist_ok=True)

    # Wait for rank 0 to create the directory before other ranks try to save
    if world_size > 1:
        dist.barrier()
    # print(f"[Rank {rank}] Starting reconstruction...")
    # Each rank processes its shard
    N = min(len(val_dataset), num_samples)
    chunk = N // world_size

    if rank < world_size - 1:
        start = rank * chunk
        end   = (rank + 1) * chunk
    else:
        # Last rank takes the remainder (and handles N < world_size gracefully)
        start = rank * chunk
        end   = N

    rank_indices = list(range(start, end))
    subset = Subset(val_dataset, rank_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Reconstruct images on this rank
    reconstructions = []
    references = []
    iterator = tqdm(loader, desc=f"[Rank {rank}] Reconstructing", file=sys.stdout) if rank == 0 else loader

    with torch.inference_mode():
        for batch in iterator:
            images = batch[0]
            labels = batch[1]
            
            if reference_npz_path is None:
                # Use label as reference if it's a tensor, otherwise use input images
                ref_to_save = labels if (isinstance(labels, torch.Tensor) and labels.ndim >= 3) else images
                ref_np = ref_to_save.clamp(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                for img in ref_np:
                    references.append(img)
            
            images = images.to(device, non_blocking=True)
            with autocast(**autocast_kwargs):
                recon = model(images)

            # Convert to numpy uint8 [H, W, C]
            recon = recon.clamp(0, 1)
            recon_np = recon.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for img in recon_np:
                reconstructions.append(img)

    reconstructions = np.stack(reconstructions)
    shard_path = os.path.join(temp_dir, f"recon_{global_step:07d}_{rank:02d}.npz")
    if reference_npz_path is None:
        references = np.stack(references)
        np.savez(shard_path, arr_0=reconstructions, ref=references)
    else:
        np.savez(shard_path, arr_0=reconstructions)

    if rank == 0:
        print(f"[Rank {rank}] Saved {len(reconstructions)} reconstructions to {shard_path}")

    # Wait for all ranks to finish reconstruction
    if world_size > 1:
        dist.barrier()

        # Rank 0 computes metrics
    metrics = None
    if rank == 0:
        # Combine all reconstruction shards
        all_recons = []
        all_refs = []
        for r in range(world_size):
            shard_file = os.path.join(temp_dir, f"recon_{global_step:07d}_{r:02d}.npz")
            with np.load(shard_file) as shard_data:
                all_recons.append(shard_data["arr_0"])
                if reference_npz_path is None:
                    all_refs.append(shard_data["ref"])

        combined_recons = np.concatenate(all_recons, axis=0)
        
        if reference_npz_path is not None:
            # Load reference NPZ
            if not os.path.exists(reference_npz_path):
                raise FileNotFoundError(f"Reference NPZ not found at {reference_npz_path}")
            with np.load(reference_npz_path) as ref_data:
                ref_images = ref_data["arr_0"]
            print(f"[Eval] Loaded reference NPZ from {reference_npz_path}, shape: {ref_images.shape}")
        else:
            ref_images = np.concatenate(all_refs, axis=0)
            print(f"[Eval] Combined reference from dataset shape: {ref_images.shape}")

        final_n = min(len(ref_images), len(combined_recons), num_samples)
        ref_images = ref_images[:final_n]
        combined_recons = combined_recons[:final_n]
        print(f"[Eval] Final size for metrics: {final_n}")

        # Compute metrics
        print("[Eval] Computing metrics...")
        metrics = compute_reconstruction_metrics(
            ref_images,
            combined_recons,
            device,
            metric_batch_size,
            metrics_to_compute=metrics_to_compute,
            disable_bar= True, # by default no bar
        )

        # Compute iFID if requested: FID(val, decode(slerp(z_val, z_nn_train, α)))
        if 'ifid' in metrics_to_compute and train_dataset is not None:
            print("[Eval] Computing iFID (interpolation FID)...")
            device_str = "cuda" if device.type == "cuda" else "cpu"
            n_train = min(len(train_dataset), max_train_samples)
            train_subset = Subset(train_dataset, list(range(n_train)))
            train_loader = DataLoader(
                train_subset, batch_size=metric_batch_size,
                shuffle=False, num_workers=4, pin_memory=True, drop_last=False,
            )
            train_images_list = []
            for tbatch in train_loader:
                timg = tbatch[0].clamp(0, 1).mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                train_images_list.append(timg)
            train_images = np.concatenate(train_images_list, axis=0)

            if hasattr(model, 'encode') and hasattr(model, 'decode'):
                metrics["ifid"] = calculate_ifid(
                    ref_images, train_images, model,
                    batch_size=metric_batch_size, device=device_str,
                )
            else:
                print("[Eval] Model lacks encode/decode, skipping iFID.")

        # Print results
        print(f"[Eval] Step {global_step} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # Cleanup reconstruction shards
        for r in range(world_size):
            shard_file = os.path.join(temp_dir, f"recon_{global_step:07d}_{r:02d}.npz")
            if os.path.exists(shard_file):
                os.remove(shard_file)

    if world_size > 1:
        dist.barrier()
    # model.train()

    return metrics
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-img", type=str, default="samples/imagenet-256-val.npz")
    parser.add_argument("--rec-img", type=str, default="samples/sdvae-ft-mse-f8d4.npz")
    parser.add_argument("--bs", type=int, default=128)
    args = parser.parse_args()

    # Load images
    device = "cuda"
    ref_img = np.load(args.ref_img)["arr_0"]
    rec_img = np.load(args.rec_img)["arr_0"]
    print(f"Loaded images: ref: {ref_img.shape}, rec: {rec_img.shape}")
    
    psnr = calculate_psnr(ref_img, rec_img, args.bs, device)
    print(f"PSNR: {psnr:.6f}")
    lpips = calculate_lpips(ref_img, rec_img, args.bs, device)
    print(f"LPIPS: {lpips:.6f}")
    ssim_val = calculate_ssim(ref_img, rec_img, args.bs, device)
    print(f"SSIM: {ssim_val:.6f}")
    rfid = calculate_rfid(ref_img, rec_img, args.bs, device)
    print(f"rFID: {rfid:.6f}")

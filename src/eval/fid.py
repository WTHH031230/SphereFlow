import argparse
from pathlib import Path
import numpy as np

from torch_fidelity import calculate_metrics
from .utils import ImgArrDataset

from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3  # original torch-fidelity :contentReference[oaicite:2]{index=2}
import scipy.linalg
import torch
import torch.nn.functional as F

def _fid_from_moments(mu1, sigma1, mu2, sigma2) -> float:
    # FID formula :contentReference[oaicite:3]{index=3}

    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):  # numerical noise
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(max(fid, 0.0))


@torch.no_grad()
def _compute_inception_moments_from_arr(arr: np.ndarray, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Uses torch-fidelity's InceptionV3 feature extractor to get 2048-d pool features.
    Assumes arr is [N,H,W,C] or [N,C,H,W], uint8 (0..255) or float (0..1 or 0..255).
    """
    # Convert to torch in NCHW uint8 as the safest default.
    x = arr
    if x.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape {x.shape}")

    if x.shape[-1] == 3:  # NHWC -> NCHW
        x = np.transpose(x, (0, 3, 1, 2))

    if x.dtype != np.uint8:
        # If float in [0,1], scale up; otherwise assume already 0..255-ish
        x_f = x.astype(np.float32)
        if x_f.max() <= 1.5:
            x_f = x_f * 255.0
        x = np.clip(x_f, 0, 255).astype(np.uint8)

    xt = torch.from_numpy(x).to(device=device, dtype=torch.uint8)

    fe = FeatureExtractorInceptionV3(name="inception-v3-compat", features_list=['2048']).to(device).eval()  # preregistered extractor name :contentReference[oaicite:4]{index=4}

    feats = []
    for i in range(0, xt.shape[0], batch_size):
        batch = xt[i : i + batch_size]
        f = fe(batch)[0]              # (B, 2048)
        feats.append(f.detach().cpu())

    feats = torch.cat(feats, dim=0).double().numpy()  # (N, 2048) float64
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def get_inception_moments(
    arr: np.ndarray,
    batch_size: int = 64,
    device: str = "cuda",
) -> dict:
    """Inception feature moments (mu, sigma) of an image array, used as the gFID reference distribution."""
    mu, sigma = _compute_inception_moments_from_arr(arr, batch_size=batch_size, device=device)
    return {"mu": mu, "sigma": sigma}


def calculate_gfid(
    arr1: np.ndarray,
    ref_arr: dict,
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    mu_ref, sigma_ref = ref_arr['mu'], ref_arr['sigma']
    mu_gen, sigma_gen = _compute_inception_moments_from_arr(arr1, batch_size=batch_size, device=device)
    return _fid_from_moments(mu_gen, sigma_gen, mu_ref, sigma_ref)

def calculate_rfid(
    arr1,
    arr2=None,
    bs=64,
    device="cuda",
    fid_statistics_file=None,
):
    arr1_ds = ImgArrDataset(arr1)

    if fid_statistics_file is not None:
        metrics_kwargs = dict(
            input1=arr1_ds,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            batch_size=bs,
            fid=True,
            cuda=(device == "cuda"),
        )
    else:
        if arr2 is None:
            raise ValueError("Either arr2 or fid_statistics_file must be provided.")
        arr2_ds = ImgArrDataset(arr2)
        metrics_kwargs = dict(
            input1=arr1_ds,
            input2=arr2_ds,
            batch_size=bs,
            fid=True,
            cuda=(device == "cuda"),
        )

    metrics = calculate_metrics(**metrics_kwargs)
    return metrics["frechet_inception_distance"]


def calculate_is(
    arr,
    bs=64,
    device="cuda",
):
    arr_ds = ImgArrDataset(arr)
    metrics = calculate_metrics(
        input1=arr_ds,
        batch_size=bs,
        isc=True,
        cuda=(device == "cuda"),
        verbose=False,
    )
    return metrics["inception_score_mean"]


def _slerp(z1: torch.Tensor, z2: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
    """Spherical linear interpolation between two latent tensors.

    Adapted from evalvae.py in Making-rFID-Predictive-of-Diffusion-gFID.
    For nearly parallel vectors (sin_theta < 1e-6), falls back to linear interpolation.
    """
    B = z1.shape[0]
    z1f = z1.reshape(B, -1).float()
    z2f = z2.reshape(B, -1).float()

    z1n = z1f / (z1f.norm(dim=1, keepdim=True) + eps)
    z2n = z2f / (z2f.norm(dim=1, keepdim=True) + eps)

    dot = (z1n * z2n).sum(dim=1).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    w1 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    w2 = torch.sin(t * theta) / (sin_theta + eps)
    zt = w1.unsqueeze(1) * z1f + w2.unsqueeze(1) * z2f

    # Fall back to linear for nearly parallel vectors
    small = sin_theta < 1e-6
    if small.any():
        zt[small] = (1.0 - t) * z1f[small] + t * z2f[small]

    return zt.reshape_as(z1)


@torch.no_grad()
def calculate_ifid(
    val_arr: np.ndarray,
    train_arr: np.ndarray,
    model,
    batch_size: int = 64,
    device: str = "cuda",
    alpha: float = 0.5,
) -> float:
    """Compute iFID (interpolation FID).

    iFID = FID(val_images, decode(slerp(z_val, z_nn, alpha)))
    where z_nn is the nearest neighbor of z_val in the training latent set
    (found by cosine similarity).

    Reference: 'Making rFID Predictive of Diffusion gFID', evalvae.py.

    Args:
        val_arr:    Validation images [N, H, W, C] uint8.
        train_arr:  Training images   [M, H, W, C] uint8.
        model:      Autoencoder with .encode(x) and .decode(z) methods.
        batch_size: Batch size for encode/decode passes.
        device:     Compute device string ("cuda" or "cpu").
        alpha:      Interpolation weight toward the NN (0 = val image, 1 = NN).

    Returns:
        iFID score (float).
    """
    def _to_nchw(arr: np.ndarray) -> torch.Tensor:
        x = arr
        if x.ndim != 4:
            raise ValueError(f"Expected 4-D array, got shape {x.shape}")
        if x.shape[-1] in (1, 3, 4):          # NHWC -> NCHW
            x = np.transpose(x, (0, 3, 1, 2))
        x = x.astype(np.float32)
        if x.max() > 1.5:                      # uint8 range
            x = x / 255.0
        return torch.from_numpy(x).to(device)

    was_training = model.training
    model.eval()

    val_t   = _to_nchw(val_arr)    # (N, C, H, W)
    train_t = _to_nchw(train_arr)  # (M, C, H, W)

    # ── 1. Encode training images ──────────────────────────────────────────
    z_train_list = []
    for i in range(0, train_t.shape[0], batch_size):
        z = model.encode(train_t[i : i + batch_size])
        z_train_list.append(z.cpu())
    z_train      = torch.cat(z_train_list, dim=0)          # (M, ...)
    z_train_flat = z_train.reshape(z_train.shape[0], -1)   # (M, D)

    # ── 2. Encode val, find NN, slerp, decode ─────────────────────────────
    interp_list = []
    for i in range(0, val_t.shape[0], batch_size):
        vb    = val_t[i : i + batch_size]
        z_val = model.encode(vb)                            # (B, ...)
        z_val_flat = z_val.reshape(z_val.shape[0], -1)     # (B, D)

        # Cosine-similarity nearest-neighbor lookup
        z_val_n   = F.normalize(z_val_flat.float(), dim=1)
        z_train_n = F.normalize(z_train_flat.float().to(device), dim=1)
        sim    = z_val_n @ z_train_n.T                     # (B, M)
        nn_idx = sim.argmax(dim=1)                         # (B,)

        z_nn   = z_train[nn_idx.cpu()].to(device)          # (B, ...)
        z_intp = _slerp(z_val, z_nn, alpha)

        x_intp = model.decode(z_intp).clamp(0.0, 1.0)
        interp_np = (
            x_intp.mul(255.0).permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8).numpy()
        )
        interp_list.append(interp_np)

    if was_training:
        model.train()

    interp_arr = np.concatenate(interp_list, axis=0)       # (N, H, W, C)
    return calculate_rfid(val_arr, interp_arr, bs=batch_size, device=device)


@torch.no_grad()
def _extract_inception_features(arr: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    """Extract InceptionV3 2048-d pool features. Returns (N, 2048) float64 numpy."""
    x = arr
    if x.shape[-1] == 3:
        x = np.transpose(x, (0, 3, 1, 2))
    if x.dtype != np.uint8:
        x_f = x.astype(np.float32)
        if x_f.max() <= 1.5:
            x_f = x_f * 255.0
        x = np.clip(x_f, 0, 255).astype(np.uint8)
    xt = torch.from_numpy(x).to(device=device, dtype=torch.uint8)
    fe = FeatureExtractorInceptionV3(
        name="inception-v3-compat", features_list=["2048"]
    ).to(device).eval()
    feats = []
    for i in range(0, xt.shape[0], batch_size):
        feats.append(fe(xt[i : i + batch_size])[0].detach().cpu())
    del fe
    return torch.cat(feats, dim=0).double().numpy()   # (N, 2048)


@torch.no_grad()
def calculate_ifid_distributed(
    val_arr: np.ndarray,
    train_arr: np.ndarray,
    model,
    batch_size: int = 64,
    device: str = "cuda",
    alpha: float = 0.5,
    rank: int = 0,
    world_size: int = 1,
    temp_dir: str = "/tmp/ifid_shards",
):
    """Distributed version of calculate_ifid.

    Each rank encodes its val shard, finds NNs in train latents, slerps and
    decodes; results are saved as NPZ shards.  Rank 0 loads all shards and
    returns FID(val_arr, interp_arr).  Other ranks return None.

    Requires torch.distributed to be initialised before calling.
    """
    import torch.distributed as dist
    import os

    os.makedirs(temp_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    def _to_nchw(arr: np.ndarray) -> torch.Tensor:
        x = arr
        if x.ndim != 4:
            raise ValueError(f"Expected 4-D array, got shape {x.shape}")
        if x.shape[-1] in (1, 3, 4):
            x = np.transpose(x, (0, 3, 1, 2))
        x = x.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        return torch.from_numpy(x).to(device)

    was_training = model.training
    model.eval()

    train_t = _to_nchw(train_arr)   # (M, C, H, W) — same on every rank

    # ── 1. Each rank encodes the full training set (small, ~5000 imgs) ─────
    z_train_list = []
    for i in range(0, train_t.shape[0], batch_size):
        z = model.encode(train_t[i : i + batch_size])
        z_train_list.append(z.cpu())
    z_train      = torch.cat(z_train_list, dim=0)
    z_train_flat = z_train.reshape(z_train.shape[0], -1).float()

    # ── 2. Distribute val images across ranks ─────────────────────────────
    N = val_arr.shape[0]
    chunk = N // world_size
    start = rank * chunk
    end   = N if rank == world_size - 1 else (rank + 1) * chunk
    val_shard = _to_nchw(val_arr[start:end])   # (local_N, C, H, W)

    # ── 3. Encode val shard, NN lookup, slerp, decode ─────────────────────
    z_train_n = F.normalize(z_train_flat.to(device), dim=1)  # (M, D) on GPU

    interp_list = []
    for i in range(0, val_shard.shape[0], batch_size):
        vb         = val_shard[i : i + batch_size]
        z_val      = model.encode(vb)
        z_val_flat = z_val.reshape(z_val.shape[0], -1).float()
        z_val_n    = F.normalize(z_val_flat, dim=1)
        sim        = z_val_n @ z_train_n.T                   # (B, M)
        nn_idx     = sim.argmax(dim=1)
        z_nn       = z_train[nn_idx.cpu()].to(device)
        z_intp     = _slerp(z_val, z_nn, alpha)
        x_intp     = model.decode(z_intp).clamp(0.0, 1.0)
        interp_np  = (
            x_intp.mul(255.0).permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8).numpy()
        )
        interp_list.append(interp_np)

    local_interp = np.concatenate(interp_list, axis=0) if interp_list else \
                   np.zeros((0, *val_arr.shape[1:]), dtype=np.uint8)

    if was_training:
        model.train()

    # ── 4. Each rank extracts Inception features for its shards in parallel ─
    # Free RAE latent tensors before loading InceptionV3 to save GPU memory.
    del z_train_n, z_train_flat, z_train, train_t, val_shard
    torch.cuda.empty_cache()

    val_shard_np = val_arr[start:end]                  # already uint8 numpy
    feat_val    = _extract_inception_features(val_shard_np, batch_size, device)   # (local_N, 2048)
    feat_interp = _extract_inception_features(local_interp,  batch_size, device)  # (local_N, 2048)

    # ── 5. Gather features to rank 0 and compute FID ─────────────────────
    if world_size > 1:
        all_feat_val    = [None] * world_size if rank == 0 else None
        all_feat_interp = [None] * world_size if rank == 0 else None
        dist.gather_object(feat_val,    all_feat_val,    dst=0)
        dist.gather_object(feat_interp, all_feat_interp, dst=0)
    else:
        all_feat_val    = [feat_val]
        all_feat_interp = [feat_interp]

    if rank != 0:
        return None

    feats_val    = np.concatenate(all_feat_val,    axis=0)   # (N, 2048)
    feats_interp = np.concatenate(all_feat_interp, axis=0)   # (N, 2048)
    mu1, sigma1  = feats_val.mean(0),    np.cov(feats_val,    rowvar=False)
    mu2, sigma2  = feats_interp.mean(0), np.cov(feats_interp, rowvar=False)
    return _fid_from_moments(mu1, sigma1, mu2, sigma2)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description="Compute FID using original torch-fidelity")
    parser.add_argument("--arr1", type=str, required=True,
                        help="Path to generated images array (.npy or .npz). Array should be N x H x W x 3 or N x 3 x H x W.")
    parser.add_argument("--arr2", type=str, default=None,
                        help="Optional path to reference images array (.npy or .npz). If set, uses torch_fidelity.calculate_metrics.")

    # reference stats path (npz with mu/sigma keys)
    parser.add_argument("--ref", "--fid-statistics-file", dest="ref", type=str, default=None,
                        help="Path to reference stats (.npz with mu/sigma, mu_s/sigma_s, mu_clip/sigma_clip, etc.). "
                             "If set, computes moments for arr1 using torch-fidelity InceptionV3 and FID vs these stats.")

    # which keys in the ref npz to use
    parser.add_argument("--ref-mu-key", type=str, default="mu",
                        help="Key in --ref .npz for reference mean (default: mu).")
    parser.add_argument("--ref-sigma-key", type=str, default="sigma",
                        help="Key in --ref .npz for reference covariance (default: sigma).")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-scipy", action="store_true",
                        help="Do not use scipy.linalg.sqrtm; use torch eig fallback (slower/less identical).")

    args = parser.parse_args()

    # exactly one of arr2 or ref must be provided
    if (args.arr2 is None) == (args.ref is None):
        parser.error("Specify exactly one of --arr2 or --ref/--fid-statistics-file")

    def load_array(path: str) -> np.ndarray:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix == ".npy":
            return np.load(p)
        elif p.suffix == ".npz":
            z = np.load(p)
            # common conventions: 'arr_0' or 'images'
            if "arr_0" in z.files:
                return z["arr_0"]
            if "images" in z.files:
                return z["images"]
            raise KeyError(f"{p} is .npz but has no 'arr_0' or 'images'. Keys={z.files}")
        else:
            raise ValueError(f"Unsupported array file type: {p.suffix} (expected .npy or .npz)")

    arr1 = load_array(args.arr1)
    print("[INFO] arr1:", arr1.shape, arr1.dtype)

    if args.arr2 is not None:
        arr2 = load_array(args.arr2)
        print("[INFO] arr2:", arr2.shape, arr2.dtype)

        fid = calculate_rfid(
            arr1=arr1,
            arr2=arr2,
            bs=args.batch_size,
            device=args.device,
            fid_statistics_file=None,   # unused in upstream path
        )
        print(f"[RESULT] FID: {fid:.6f}")
        raise SystemExit(0)

    # stats mode: arr1 vs ref npz moments
    ref_path = Path(args.ref)
    if not ref_path.exists():
        raise FileNotFoundError(f"Ref stats not found: {ref_path}")

    if ref_path.suffix != ".npz":
        raise ValueError(
            f"--ref must be a .npz containing mu/sigma. Got: {ref_path.suffix}. "
            "Original torch-fidelity cannot use .pt stats without input2."
        )

    stats = np.load(ref_path)
    if args.ref_mu_key not in stats.files or args.ref_sigma_key not in stats.files:
        raise KeyError(
            f"Missing '{args.ref_mu_key}'/'{args.ref_sigma_key}' in {ref_path}. "
            f"Available keys: {list(stats.files)}"
        )
    print(f"[INFO] ref stats: {ref_path}")
    # If you want to respect --no-scipy, thread it through:
    # (edit your fid_arr_vs_npzstats / _fid_from_moments accordingly)
    fid = calculate_gfid(
        arr1=arr1,
        ref_arr=stats,
        batch_size=args.batch_size,
        device=args.device,
        # optionally add: use_scipy=(not args.no_scipy)
    )
    print(f"[RESULT] FID: {fid:.6f}")
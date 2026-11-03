#!/usr/bin/env python3
"""Split a multi-decoder RAE checkpoint into per-modality single-RAE state dicts.

Input:  Step-3 checkpoint with ``encoder.*`` and ``decoders.{m}.*`` keys.
Output: ``<out-dir>/init_m{m}.pt`` = shared encoder + ``decoder.*`` of modality m,
        used as ``--pretrained-decoder-path`` for the optional decoder fine-tuning.

Called by ``scripts/train/brats/finetune_decoder.sh`` with ``--state-key model``.
"""
import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-modalities", type=int, default=4)
    parser.add_argument("--state-key", choices=["model", "ema"], default="model",
                        help="Which weights to extract (falls back to the raw dict if the key is absent).")
    parser.add_argument("--with-latent-stats", action="store_true",
                        help="Also copy encoder_* / latent_mean / latent_var keys.")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if args.state_key == "ema":
        sd = ckpt.get("ema", ckpt.get("model", ckpt))
    else:
        sd = ckpt["model"] if "model" in ckpt else ckpt
    os.makedirs(args.out_dir, exist_ok=True)

    if args.with_latent_stats:
        enc_sd = {k: v for k, v in sd.items() if k.startswith("encoder.") or k.startswith("encoder_")}
        enc_sd.update({k: v for k, v in sd.items() if k in ("latent_mean", "latent_var")})
    else:
        enc_sd = {k: v for k, v in sd.items() if k.startswith("encoder.")}

    for m in range(args.num_modalities):
        out_sd = dict(enc_sd)
        prefix = f"decoders.{m}."
        found = False
        for k, v in sd.items():
            if k.startswith(prefix):
                out_sd[k.replace(prefix, "decoder.")] = v
                found = True
        if not found:
            print(f"Warning: no decoder weights found for modality {m}")
            continue
        out_path = os.path.join(args.out_dir, f"init_m{m}.pt")
        torch.save(out_sd, out_path)
        print(f"Modality {m}: wrote {out_path}")


if __name__ == "__main__":
    main()

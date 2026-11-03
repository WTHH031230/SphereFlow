#!/usr/bin/env python3
"""Build the RAE checkpoint used by the SphereFlow flow model (BraTS).

Takes the step-3 shared-encoder RAE (``encoder.*`` + ``decoders.{m}.*``) and, if given, replaces each
``decoders.{m}.*`` with the ``decoder.*`` weights of the fine-tuned single-decoder checkpoint of
modality ``m`` (finetune_decoder.sh). The top-level ``decoder_pred.*`` alias is re-synced from
``decoders.0.decoder_pred.*``.
"""
import argparse
import os

import torch


def _state(ckpt):
    if "ema" in ckpt:
        return ckpt["ema"]
    if "model" in ckpt:
        return ckpt["model"]
    return ckpt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Step-3 RAE checkpoint (shared encoder + all decoders).")
    parser.add_argument("--ft-decoders", nargs="*", default=[],
                        help="Optional fine-tuned decoder checkpoints, one per modality, in modality order.")
    parser.add_argument("--out", required=True, help="Output path of the patched RAE checkpoint.")
    args = parser.parse_args()

    if not os.path.exists(args.base):
        raise SystemExit(f"Base RAE checkpoint not found: {args.base}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    patched = dict(_state(torch.load(args.base, map_location="cpu")))
    for i, ft_path in enumerate(args.ft_decoders):
        if not os.path.exists(ft_path):
            print(f"Warning: FT checkpoint for modality {i} not found at {ft_path}")
            continue
        print(f"Patching modality {i} from {ft_path}")
        for k, v in _state(torch.load(ft_path, map_location="cpu")).items():
            if k.startswith("decoder."):
                patched[k.replace("decoder.", f"decoders.{i}.")] = v

    for k in [k for k in patched if k.startswith("decoder_pred.")]:
        src_key = k.replace("decoder_pred.", "decoders.0.decoder_pred.")
        if src_key in patched:
            patched[k] = patched[src_key]
            print(f"Synced {k} <- {src_key}")
        else:
            del patched[k]
            print(f"Removed orphan key {k}")

    torch.save({"model": patched}, args.out)
    print(f"Saved patched RAE to {args.out}")


if __name__ == "__main__":
    main()

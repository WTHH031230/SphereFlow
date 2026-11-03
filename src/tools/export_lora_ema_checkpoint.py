#!/usr/bin/env python3
"""Export a Step-3 LoRA EMA checkpoint as a plain RAE state dict."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, MutableMapping

import torch


def merge_lora_state_dict(
    state_dict: MutableMapping[str, torch.Tensor],
    lora_alpha: float,
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    lora_a_keys = sorted(k for k in state_dict if k.endswith(".lora_A"))
    lora_prefixes = {k[: -len(".lora_A")] for k in lora_a_keys}

    for key, value in state_dict.items():
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            continue
        if key.endswith(".base.weight") or key.endswith(".base.bias"):
            continue
        merged[key] = value.detach().cpu().clone() if torch.is_tensor(value) else value

    for prefix in sorted(lora_prefixes):
        a_key = f"{prefix}.lora_A"
        b_key = f"{prefix}.lora_B"
        w_key = f"{prefix}.base.weight"
        bias_key = f"{prefix}.base.bias"
        if a_key not in state_dict or b_key not in state_dict or w_key not in state_dict:
            raise KeyError(f"Incomplete LoRA keys for prefix '{prefix}'")

        lora_a = state_dict[a_key].detach().cpu()
        lora_b = state_dict[b_key].detach().cpu()
        base_w = state_dict[w_key].detach().cpu()
        rank = max(int(lora_a.shape[0]), 1)
        scale = float(lora_alpha) / rank
        merged_w = base_w.float().add(lora_b.float().matmul(lora_a.float()), alpha=scale)
        merged[f"{prefix}.weight"] = merged_w.to(dtype=base_w.dtype)

        if bias_key in state_dict:
            merged[f"{prefix}.bias"] = state_dict[bias_key].detach().cpu().clone()

    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--state-key", default="ema", help="Checkpoint key to export, usually 'ema'.")
    parser.add_argument("--output-key", default="ema", help="State key used in the exported checkpoint.")
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    if isinstance(checkpoint, dict) and args.state_key in checkpoint:
        source_state = checkpoint[args.state_key]
    elif isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        source_state = checkpoint
    else:
        available = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)
        raise KeyError(f"Could not find state key '{args.state_key}'. Available: {available}")

    exported_state = merge_lora_state_dict(source_state, args.lora_alpha)
    exported = {
        "step": checkpoint.get("step", 0) if isinstance(checkpoint, dict) else 0,
        "epoch": checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0,
        args.output_key: exported_state,
        "source_checkpoint": str(args.input),
        "merged_lora_alpha": float(args.lora_alpha),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, args.output)
    print(
        f"Exported {len(exported_state)} tensors to {args.output} "
        f"from {args.input} ({len(source_state) - len(exported_state)} LoRA/base tensors folded)."
    )


if __name__ == "__main__":
    main()

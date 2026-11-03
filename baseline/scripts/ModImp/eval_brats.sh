#!/bin/bash
# ModImp (Qi et al., MICCAI 2025): evaluate a BraTS checkpoint on all 14 missing-modality masks
# (PSNR / SSIM / rFID / IS per target).
# env: sphereflow_baseline
# in : CHECKPOINT_PATH (default: $CKPT_DIR/baselines/brats/ModImp.pt)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/env.sh"
cd "$REPO_DIR"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CKPT_DIR/baselines/brats/ModImp.pt}"
EVAL_DIR="${EVAL_DIR:-$OUTPUT_DIR/eval/baselines_brats/ModImp}"
require_file "$BRATS_JSON" "$CHECKPOINT_PATH"
mkdir -p "$EVAL_DIR"
accelerate launch --multi_gpu --num_processes "$NPROC" --gpu_ids "$GPUS" --main_process_port "${MASTER_PORT:-29521}" \
    baseline/src/test_brats.py \
    --model_type ModImp \
    --data_json "$BRATS_JSON" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_dir "$EVAL_DIR" \
    --batch_size 64 \
    --workers 4 \
    --amp

#!/bin/bash
# MMSyn (Chartsias et al., TMI 2018): evaluate a BraTS checkpoint on all 14 missing-modality masks
# (PSNR / SSIM / rFID / IS per target).
# env: sphereflow_baseline
# in : CHECKPOINT_PATH (default: $CKPT_DIR/baselines/brats/MMSyn.pt)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/env.sh"
cd "$REPO_DIR"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CKPT_DIR/baselines/brats/MMSyn.pt}"
EVAL_DIR="${EVAL_DIR:-$OUTPUT_DIR/eval/baselines_brats/MMSyn}"
require_file "$BRATS_JSON" "$CHECKPOINT_PATH"
mkdir -p "$EVAL_DIR"
accelerate launch --multi_gpu --num_processes "$NPROC" --gpu_ids "$GPUS" --main_process_port "${MASTER_PORT:-29571}" \
    baseline/src/test_brats.py \
    --model_type MMSyn \
    --data_json "$BRATS_JSON" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_dir "$EVAL_DIR" \
    --batch_size 8 \
    --workers 4 \
    --amp \
    --mmsyn_latent_dim 16 --mmsyn_fusion max

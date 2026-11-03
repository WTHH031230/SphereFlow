#!/bin/bash
# SphereFlow on MRI<->CT: PSNR / SSIM / rFID / IS on the test split with a
# 4-step Euler ODE (target_0 = CT->MRI, target_1 = MRI->CT).
# env: sphereflow
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$REPO_DIR"
CONFIG="${CONFIG:-configs/stage2/mri_ct.yaml}"
RAE_PATH="${RAE_PATH:-$CKPT_DIR/sphereflow/mri_ct/rae.pt}"
FLOW_CKPT="${FLOW_CKPT:-$CKPT_DIR/sphereflow/mri_ct/flow.pt}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_DIR/eval/sphereflow_mri_ct}"
STEPS="${STEPS:-4}"
require_file "$MRICT_JSON" "$RAE_PATH" "$FLOW_CKPT"

CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC \
    src/eval_mri_ct.py \
    --config "$CONFIG" \
    --data-path "$MRICT_JSON" \
    --rae-path "$RAE_PATH" \
    --flow-ckpt "final:${FLOW_CKPT}" \
    --output-dir "$RESULTS_DIR" \
    --phase test \
    --precision bf16 \
    --batch-size 64 \
    --metric-batch-size 128 \
    --sampler-num-steps "$STEPS"

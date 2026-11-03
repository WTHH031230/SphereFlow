#!/bin/bash
# SphereFlow Stage 1, step (i): single-modality warm-up.
# Frozen DINOv2-B/14 (registers) encoder + one ViT-G decoder per modality, L1 + LPIPS + GAN.
# env: sphereflow
# out: $OUTPUT_DIR/sphereflow/brats/step1_single/mod{0..3}/checkpoints/ep-last.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$REPO_DIR"
CONFIG="${CONFIG:-configs/stage1/brats.yaml}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_DIR/sphereflow/brats/step1_single}"
GLOBAL_BS="${GLOBAL_BS:-256}"
PRECISION="${PRECISION:-bf16}"
require_file "$BRATS_JSON"

for MODALITY in ${MODALITIES:-0 1 2 3}; do
    echo "[Step 1] modality ${MODALITY}"
    CUDA_VISIBLE_DEVICES=$GPUS EXPERIMENT_NAME="mod${MODALITY}" torchrun \
        --standalone --nproc_per_node=$NPROC \
        src/train_single_decoder.py \
        --config "$CONFIG" \
        --data-path "$BRATS_JSON" \
        --dataset brats \
        --modality "$MODALITY" \
        --results-dir "$RESULTS_DIR" \
        --precision "$PRECISION" \
        --global-batch-size "$GLOBAL_BS" \
        ${LOGGER_ARGS[@]+"${LOGGER_ARGS[@]}"}
done

#!/bin/bash
# SphereFlow Stage 1, step (ii): manifold expansion. Each decoder is fine-tuned on cross-modal
# interpolants z_tau = tau*z_t + (1-tau)*z_s with the time-weighted reconstruction loss
# (--use-all, power-biased tau schedule). Warm-started from step 1 (decoder + discriminator).
# env: sphereflow
# out: $OUTPUT_DIR/sphereflow/brats/step2_interp/mod{0..3}/checkpoints/ep-last.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$REPO_DIR"
CONFIG="${CONFIG:-configs/stage1/brats_interp.yaml}"
STEP1_DIR="${STEP1_DIR:-$OUTPUT_DIR/sphereflow/brats/step1_single}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_DIR/sphereflow/brats/step2_interp}"
GLOBAL_BS="${GLOBAL_BS:-256}"
PRECISION="${PRECISION:-bf16}"

for MODALITY in ${MODALITIES:-0 1 2 3}; do
    STEP1_CKPT="${STEP1_DIR}/mod${MODALITY}/checkpoints/ep-last.pt"
    require_file "$STEP1_CKPT"
    if [[ -f "${RESULTS_DIR}/mod${MODALITY}/checkpoints/ep-last.pt" ]]; then
        echo "[Step 2] modality ${MODALITY} already done, skipping"; continue
    fi
    echo "[Step 2] modality ${MODALITY} (from ${STEP1_CKPT})"
    CUDA_VISIBLE_DEVICES=$GPUS EXPERIMENT_NAME="mod${MODALITY}" torchrun \
        --standalone --nproc_per_node=$NPROC \
        src/train_interp_decoder.py \
        --config "$CONFIG" \
        --data-path "$BRATS_JSON" \
        --dataset brats \
        --modality "$MODALITY" \
        --use-all \
        --results-dir "$RESULTS_DIR" \
        --precision "$PRECISION" \
        --global-batch-size "$GLOBAL_BS" \
        --pretrained-decoder-path "$STEP1_CKPT" \
        --pretrained-disc-path "$STEP1_CKPT" \
        --w-sampling-mode power \
        ${LOGGER_ARGS[@]+"${LOGGER_ARGS[@]}"}
done

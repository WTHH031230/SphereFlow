#!/bin/bash
# SphereFlow (MRI<->CT) Stage 1, step (i): single-modality decoders (0 = MRI, 1 = CT).
# env: sphereflow
# out: $OUTPUT_DIR/sphereflow/mri_ct/step1_single/{mri,ct}/checkpoints/ep-last.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$REPO_DIR"
CONFIG="${CONFIG:-configs/stage1/mri_ct.yaml}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_DIR/sphereflow/mri_ct/step1_single}"
GLOBAL_BS="${GLOBAL_BS:-256}"
PRECISION="${PRECISION:-bf16}"
require_file "$MRICT_JSON"

for MODALITY in 0 1; do
    MOD_NAME=$([ $MODALITY -eq 0 ] && echo "mri" || echo "ct")
    CUDA_VISIBLE_DEVICES=$GPUS EXPERIMENT_NAME="${MOD_NAME}" torchrun \
        --standalone --nproc_per_node=$NPROC \
        src/train_single_decoder.py \
        --config "$CONFIG" \
        --data-path "$MRICT_JSON" \
        --dataset mri_ct \
        --modality "$MODALITY" \
        --results-dir "$RESULTS_DIR" \
        --precision "$PRECISION" \
        --global-batch-size "$GLOBAL_BS" \
        ${LOGGER_ARGS[@]+"${LOGGER_ARGS[@]}"}
done

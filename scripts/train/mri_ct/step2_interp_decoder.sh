#!/bin/bash
# SphereFlow (MRI<->CT) Stage 1, step (ii): manifold expansion with cross-modal interpolation.
# env: sphereflow
# out: $OUTPUT_DIR/sphereflow/mri_ct/step2_interp/{mri,ct}/checkpoints/ep-last.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$REPO_DIR"
CONFIG="${CONFIG:-configs/stage1/mri_ct_interp.yaml}"
STEP1_DIR="${STEP1_DIR:-$OUTPUT_DIR/sphereflow/mri_ct/step1_single}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_DIR/sphereflow/mri_ct/step2_interp}"
GLOBAL_BS="${GLOBAL_BS:-128}"
PRECISION="${PRECISION:-bf16}"

for MODALITY in 0 1; do
    MOD_NAME=$([ $MODALITY -eq 0 ] && echo "mri" || echo "ct")
    STEP1_CKPT="${STEP1_DIR}/${MOD_NAME}/checkpoints/ep-last.pt"
    require_file "$STEP1_CKPT"
    if [[ -f "${RESULTS_DIR}/${MOD_NAME}/checkpoints/ep-last.pt" ]]; then
        echo "[Step 2] ${MOD_NAME} already done, skipping"; continue
    fi
    CUDA_VISIBLE_DEVICES=$GPUS EXPERIMENT_NAME="${MOD_NAME}" torchrun \
        --standalone --nproc_per_node=$NPROC \
        src/train_interp_decoder.py \
        --config "$CONFIG" \
        --data-path "$MRICT_JSON" \
        --dataset mri_ct \
        --modality "$MODALITY" \
        --use-all \
        --results-dir "$RESULTS_DIR" \
        --precision "$PRECISION" \
        --global-batch-size "$GLOBAL_BS" \
        --pretrained-decoder-path "$STEP1_CKPT" \
        --w-sampling-mode power \
        ${LOGGER_ARGS[@]+"${LOGGER_ARGS[@]}"}
done

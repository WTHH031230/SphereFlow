#!/bin/bash
# SphereFlow Stage 1, step (iii): joint fine-tuning of ONE shared encoder with all four decoders
# (L1 + LPIPS + cosine / pooled-MSE latent alignment, no GAN).
# env: sphereflow
# out: $OUTPUT_DIR/sphereflow/brats/step3_shared/all/checkpoints/ep-last.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$REPO_DIR"
CONFIG="${CONFIG:-configs/stage1/brats.yaml}"
STEP2_DIR="${STEP2_DIR:-$OUTPUT_DIR/sphereflow/brats/step2_interp}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_DIR/sphereflow/brats/step3_shared}"
GLOBAL_BS="${GLOBAL_BS:-32}"
PRECISION="${PRECISION:-bf16}"
COS_WEIGHT="${COS_WEIGHT:-0.1}"
MSE_WEIGHT="${MSE_WEIGHT:-0.1}"
OFFLOAD_ARGS=()
[[ "${OFFLOAD_DECODERS:-0}" == "1" ]] && OFFLOAD_ARGS=(--offload-decoders)

CKPTS=()
for m in 0 1 2 3; do
    CKPTS+=("${STEP2_DIR}/mod${m}/checkpoints/ep-last.pt")
done
require_file "${CKPTS[@]}"

CUDA_VISIBLE_DEVICES=$GPUS EXPERIMENT_NAME="all" torchrun \
    --standalone --nproc_per_node=$NPROC \
    src/train_shared_encoder.py \
    --config "$CONFIG" \
    --stage1-path-0 "${CKPTS[0]}" \
    --stage1-path-1 "${CKPTS[1]}" \
    --stage1-path-2 "${CKPTS[2]}" \
    --stage1-path-3 "${CKPTS[3]}" \
    --cos-weight "$COS_WEIGHT" \
    --mse-weight "$MSE_WEIGHT" \
    --data-path "$BRATS_JSON" \
    --dataset brats \
    --results-dir "$RESULTS_DIR" \
    --precision "$PRECISION" \
    --global-batch-size "$GLOBAL_BS" \
    --w-sampling-mode power \
    --psnr-weight 0.0 \
    --ssim-weight 0.0 \
    ${OFFLOAD_ARGS[@]+"${OFFLOAD_ARGS[@]}"} \
    ${LOGGER_ARGS[@]+"${LOGGER_ARGS[@]}"}

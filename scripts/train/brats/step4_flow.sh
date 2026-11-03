#!/bin/bash
# SphereFlow Stage 2: cross-modal flow matching (data-as-source, x0 = mean of observed latents,
# x1 = target latent) with a DiT^DH-XXL conditioned on the (target, mask) class token.
# Uses the step-3 RAE; if all four decoders from the optional finetune_decoder.sh exist, they replace
# the step-3 decoders.
# env: sphereflow
# in : STEP3_CKPT (default: step-3 output), FT_DIR (default: finetune_decoder.sh output)
# out: $OUTPUT_DIR/sphereflow/brats/step4_flow/rae.pt
#      $OUTPUT_DIR/sphereflow/brats/step4_flow/all/checkpoints/ep-last.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$REPO_DIR"
CONFIG="${CONFIG:-configs/stage2/brats.yaml}"
OUT="$OUTPUT_DIR/sphereflow/brats"
STEP3_CKPT="${STEP3_CKPT:-$OUT/step3_shared/all/checkpoints/ep-last.pt}"
FT_DIR="${FT_DIR:-$OUT/finetune_decoder}"
RESULTS_DIR="${RESULTS_DIR:-$OUT/step4_flow}"
RAE_PATH="${RAE_PATH:-$RESULTS_DIR/rae.pt}"
PRECISION="${PRECISION:-bf16}"
require_file "$STEP3_CKPT"

FT_ARGS=()
FT_DECODERS=("$FT_DIR"/mod{0,1,2,3}/checkpoints/ep-last.pt)
if ls "${FT_DECODERS[@]}" >/dev/null 2>&1; then
    FT_ARGS=(--ft-decoders "${FT_DECODERS[@]}")
fi
python src/tools/patch_rae_decoders.py --base "$STEP3_CKPT" ${FT_ARGS[@]+"${FT_ARGS[@]}"} --out "$RAE_PATH"

CUDA_VISIBLE_DEVICES=$GPUS EXPERIMENT_NAME="all" torchrun \
    --standalone --nproc_per_node=$NPROC \
    src/train_flow.py \
    --config "$CONFIG" \
    --data-path "$BRATS_JSON" \
    --rae-path "$RAE_PATH" \
    --results-dir "$RESULTS_DIR" \
    --image-size 256 \
    --precision "$PRECISION" \
    --global-seed 42 \
    ${LOGGER_ARGS[@]+"${LOGGER_ARGS[@]}"}

#!/bin/bash
# Optional: decoder fine-tuning after step 3.
# The shared encoder + per-modality decoder are extracted from the Step-3 checkpoint, then each
# decoder is fine-tuned (encoder frozen) with interpolation training + PSNR/SSIM losses, early stop.
# env: sphereflow
# in : STEP3_CKPT (default: step-3 output)
# out: $OUTPUT_DIR/sphereflow/brats/finetune_decoder/mod{m}/checkpoints/ep-last.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$REPO_DIR"
CONFIG="${CONFIG:-configs/stage1/brats_finetune.yaml}"
STEP3_CKPT="${STEP3_CKPT:-$OUTPUT_DIR/sphereflow/brats/step3_shared/all/checkpoints/ep-last.pt}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_DIR/sphereflow/brats/finetune_decoder}"
GLOBAL_BS="${GLOBAL_BS:-128}"
PRECISION="${PRECISION:-bf16}"
require_file "$STEP3_CKPT"

INIT_DIR="$RESULTS_DIR/init"
python src/tools/extract_decoders.py --ckpt "$STEP3_CKPT" --out-dir "$INIT_DIR" --num-modalities 4 --state-key model

for m in ${MODALITIES:-0 1 2 3}; do
    echo "[Fine-tune] decoder of modality $m"
    CUDA_VISIBLE_DEVICES=$GPUS EXPERIMENT_NAME="mod${m}" torchrun \
        --standalone --nproc_per_node=$NPROC \
        src/train_interp_decoder.py \
        --config "$CONFIG" \
        --data-path "$BRATS_JSON" \
        --dataset brats \
        --modality "$m" \
        --use-all \
        --results-dir "$RESULTS_DIR" \
        --precision "$PRECISION" \
        --global-batch-size "$GLOBAL_BS" \
        --pretrained-decoder-path "$INIT_DIR/init_m${m}.pt" \
        --psnr-weight 0.01 \
        --ssim-weight 0.01 \
        --early-stop \
        --w-sampling-mode power \
        --noise-in-encode \
        ${LOGGER_ARGS[@]+"${LOGGER_ARGS[@]}"}
done

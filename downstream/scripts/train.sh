#!/bin/bash
# Train the downstream 3D U-Net segmenter (WT/TC/ET) on full-modality BraTS 2023 with raw [0, 1]
# inputs (no z-score normalisation), then run the 15-mask zero-fill test (test.sh).
# env: sphereflow_baseline
# out: $UNET_OUTPUT_DIR/checkpoints/best.pth
#   BATCH_SIZE=4 EPOCHS=120 bash downstream/scripts/train.sh
#   SKIP_TRAIN=1 bash downstream/scripts/train.sh     # only run the test
#   SKIP_TEST=1 bash downstream/scripts/train.sh      # skip the test
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../scripts/env.sh"
export BRATS_JSON
cd "$SCRIPT_DIR/../src"

if [[ "${CONDA_DEFAULT_ENV:-}" != "sphereflow_baseline" ]]; then
    echo "WARNING: conda env 'sphereflow_baseline' is not active (CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-none})." >&2
fi

BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-2e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
UNET_OUTPUT_DIR="${UNET_OUTPUT_DIR:-$OUTPUT_DIR/unet3d/brats_unet3d_raw01}"
MASTER_PORT="${MASTER_PORT:-29503}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_TEST="${SKIP_TEST:-0}"

mkdir -p "${UNET_OUTPUT_DIR}"

echo "==========================================================="
echo "Train raw [0,1] 3D U-Net on BraTS 2023"
echo "  GPUs:           ${GPUS} (nproc=${NPROC})"
echo "  Batch (per-GPU): ${BATCH_SIZE} (total=$((BATCH_SIZE * NPROC)))"
echo "  Epochs:         ${EPOCHS}"
echo "  LR:             ${LR}"
echo "  Output dir:     ${UNET_OUTPUT_DIR}"
echo "==========================================================="

if [ "${SKIP_TRAIN}" != "1" ]; then
    echo ""
    echo "===== STEP 1/2: training ====="
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    torchrun --standalone --nproc_per_node="${NPROC}" \
             --master_port="${MASTER_PORT}" \
        train.py \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --lr "${LR}" \
            --num-workers "${NUM_WORKERS}" \
            --output-dir "${UNET_OUTPUT_DIR}" \
            --no-normalize \
            --swanlab-experiment "brats_unet3d_raw01" \
            --swanlab 2>&1 | tee -a "${UNET_OUTPUT_DIR}/train.log"
else
    echo "(SKIP_TRAIN=1; skipping step 1)"
fi

CKPT="${UNET_OUTPUT_DIR}/checkpoints/best.pth"
if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: best checkpoint not found after training: ${CKPT}" >&2
    exit 1
fi
echo ""
echo "Best checkpoint: ${CKPT}"

if [ "${SKIP_TEST}" != "1" ]; then
    echo ""
    echo "===== STEP 2/2: 15-mask test (zero-fill) ====="
    bash "$SCRIPT_DIR/test.sh" "${CKPT}" test 2>&1 \
        | tee -a "${UNET_OUTPUT_DIR}/test.log"
else
    echo "(SKIP_TEST=1; skipping step 2)"
fi

echo ""
echo "==========================================================="
echo "All done: $(date)"
echo "  Training output: ${UNET_OUTPUT_DIR}"
echo "  Best ckpt:       ${CKPT}"
echo "==========================================================="

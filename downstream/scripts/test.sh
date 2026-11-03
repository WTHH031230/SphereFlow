#!/bin/bash
# Evaluate the 3D U-Net under all 15 modality combinations (missing modalities zero-filled).
#   bash downstream/scripts/test.sh [ckpt_path] [phase]
# env: sphereflow_baseline
# in : ckpt_path (default: $UNET_CKPT, else $CKPT_DIR/unet3d/brats.pt), phase (default: test)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../scripts/env.sh"
export BRATS_JSON
cd "$SCRIPT_DIR/../src"

if [[ "${CONDA_DEFAULT_ENV:-}" != "sphereflow_baseline" ]]; then
    echo "WARNING: conda env 'sphereflow_baseline' is not active (CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-none})." >&2
fi

CKPT="${1:-${UNET_CKPT:-$CKPT_DIR/unet3d/brats.pt}}"
PHASE="${2:-test}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT="${MASTER_PORT:-29502}"
require_file "$CKPT"

echo "========================================================="
echo "Evaluating 3D U-Net under 15 modality combinations"
echo "  Checkpoint: ${CKPT}"
echo "  Phase:      ${PHASE}"
echo "  GPUs:       ${GPUS} (nproc=${NPROC})"
echo "========================================================="

CUDA_VISIBLE_DEVICES="${GPUS}" \
torchrun --standalone --nproc_per_node="${NPROC}" \
         --master_port="${MASTER_PORT}" \
    test.py \
        --ckpt "${CKPT}" \
        --phase "${PHASE}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}"

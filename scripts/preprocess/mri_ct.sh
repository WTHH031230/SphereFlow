#!/bin/bash
# SynthRAD2023 (brain) paired MRI/CT 2D slices -> [0,1], pad to 256x256, 70/10/20 split (seed 42).
# env: sphereflow
#   MRICT_RAW_DIR : directory with A_norm/ (MRI), B_norm/ (CT), C_norm/ (mask) 2D .npy slices in [-1, 1]
#   DATA_ROOT     : output root; slices go to $DATA_ROOT/mri_ct/{train,val,test}/
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$REPO_DIR"
: "${MRICT_RAW_DIR:?set MRICT_RAW_DIR}" "${DATA_ROOT:?set DATA_ROOT}"
export MRICT_RAW_DIR DATA_ROOT MRICT_JSON
python src/preprocess/preprocess_mri_ct.py

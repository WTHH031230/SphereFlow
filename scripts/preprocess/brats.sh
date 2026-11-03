#!/bin/bash
# BraTS 2023 GLI -> 2D slices (clip [0,4095]/4095, pad/crop to 155x256x256, middle 80 axial slices,
# case-level 70/10/20 split with seed 42) and the split JSON.
# env: sphereflow
#   BRATS_RAW_DIR : ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData (one sub-folder per case)
#   DATA_ROOT     : output root; slices go to $DATA_ROOT/brats/{train,val,test}/
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$REPO_DIR"
: "${BRATS_RAW_DIR:?set BRATS_RAW_DIR}" "${DATA_ROOT:?set DATA_ROOT}"
export BRATS_RAW_DIR DATA_ROOT
python src/preprocess/preprocess_brats.py
# Re-index the produced slices (alternative to scripts/preprocess/make_splits.sh).
SPLIT_OUT_DIR="$DATA_DIR" python src/preprocess/index_brats.py

#!/bin/bash
# Write data/brats2023_distribution.json and data/brain_mri_ct_distribution.json from the released
# splits in data/splits/ (paths relative to DATA_ROOT).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
: "${DATA_ROOT:?set DATA_ROOT (directory that contains brats/ and mri_ct/)}"
python "$REPO_DIR/src/preprocess/make_splits.py" --data-root "$DATA_ROOT" --out-dir "$DATA_DIR" "$@"

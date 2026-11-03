#!/bin/bash
# Zhang et al. (TMI 2024), re-implemented: train on MRI<->CT (2 modalities, both directions); the bidirectional
# test runs automatically at the end (test_results_Zhang.json).
# env: sphereflow_baseline
# out: $OUTPUT_DIR/baselines/mri_ct/Zhang/
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/env.sh"
cd "$REPO_DIR"
LOG_BASE="${LOG_BASE:-$OUTPUT_DIR/baselines/mri_ct}"
require_file "$MRICT_JSON"
mkdir -p "$LOG_BASE"
RESUME_ARGS=()
[[ -n "${RESUME_PATH:-}" ]] && RESUME_ARGS=(--resume_path "$RESUME_PATH")
accelerate launch --multi_gpu --num_processes "$NPROC" --gpu_ids "$GPUS" --main_process_port "${MASTER_PORT:-29520}" \
    baseline/src/train_mri_ct.py \
    --model_type Zhang \
    --data_json "$MRICT_JSON" \
    --logdir "$LOG_BASE" \
    --batch_size 32 --workers 4 --max_epochs 150 \
    --optim_lr 2e-4 --d_lr 2e-4 --decay 1e-4 --eta_min 1e-6 --val_every 5 --sample_every 500 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

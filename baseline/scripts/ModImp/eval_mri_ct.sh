#!/bin/bash
# ModImp (Qi et al., MICCAI 2025): evaluate an MRI<->CT checkpoint (CT->MRI and MRI->CT) with
# test_mri_ct.py; writes test_results_ModImp.json to LOG_BASE/ModImp.
# env: sphereflow_baseline
# in : CHECKPOINT_PATH (default: $CKPT_DIR/baselines/mri_ct/ModImp.pt)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/env.sh"
cd "$REPO_DIR"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CKPT_DIR/baselines/mri_ct/ModImp.pt}"
LOG_BASE="${LOG_BASE:-$OUTPUT_DIR/eval/baselines_mri_ct}"
require_file "$MRICT_JSON" "$CHECKPOINT_PATH"
mkdir -p "$LOG_BASE"
accelerate launch --multi_gpu --num_processes "$NPROC" --gpu_ids "$GPUS" --main_process_port "${MASTER_PORT:-29522}" \
    baseline/src/test_mri_ct.py \
    --model_type ModImp \
    --data_json "$MRICT_JSON" \
    --logdir "$LOG_BASE" \
    --batch_size 64 --workers 4 --max_epochs 150 \
    --optim_lr 1e-4 --d_lr 1e-4 --decay 1e-5 --eta_min 1e-6 --val_every 5 --sample_every 500 \
    --checkpoint_path "$CHECKPOINT_PATH"

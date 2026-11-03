#!/bin/bash
# MM-GAN (Sharma & Hamarneh, TMI 2019): evaluate an MRI<->CT checkpoint (CT->MRI and MRI->CT) with
# test_mri_ct.py; writes test_results_MMGAN.json to LOG_BASE/MMGAN.
# env: sphereflow_baseline
# in : CHECKPOINT_PATH (default: $CKPT_DIR/baselines/mri_ct/MMGAN.pt)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/env.sh"
cd "$REPO_DIR"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CKPT_DIR/baselines/mri_ct/MMGAN.pt}"
LOG_BASE="${LOG_BASE:-$OUTPUT_DIR/eval/baselines_mri_ct}"
require_file "$MRICT_JSON" "$CHECKPOINT_PATH"
mkdir -p "$LOG_BASE"
accelerate launch --multi_gpu --num_processes "$NPROC" --gpu_ids "$GPUS" --main_process_port "${MASTER_PORT:-29552}" \
    baseline/src/test_mri_ct.py \
    --model_type MMGAN \
    --data_json "$MRICT_JSON" \
    --logdir "$LOG_BASE" \
    --max_epochs 150 --batch_size 32 --workers 4 \
    --optim_lr 2e-4 --d_lr 2e-4 --decay 0 --eta_min 1e-6 --val_every 5 --sample_every 500 --amp \
    --mmgan_z_type noise --mmgan_ic 1 --mmgan_lam 0.9 --mmgan_b1 0.5 --mmgan_b2 0.999 \
    --checkpoint_path "$CHECKPOINT_PATH"

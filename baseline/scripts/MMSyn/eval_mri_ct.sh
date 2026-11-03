#!/bin/bash
# MMSyn (Chartsias et al., TMI 2018): evaluate an MRI<->CT checkpoint (CT->MRI and MRI->CT) with
# test_mri_ct.py; writes test_results_MMSyn.json to LOG_BASE/MMSyn.
# env: sphereflow_baseline
# in : CHECKPOINT_PATH (default: $CKPT_DIR/baselines/mri_ct/MMSyn.pt)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/env.sh"
cd "$REPO_DIR"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CKPT_DIR/baselines/mri_ct/MMSyn.pt}"
LOG_BASE="${LOG_BASE:-$OUTPUT_DIR/eval/baselines_mri_ct}"
require_file "$MRICT_JSON" "$CHECKPOINT_PATH"
mkdir -p "$LOG_BASE"
accelerate launch --multi_gpu --num_processes "$NPROC" --gpu_ids "$GPUS" --main_process_port "${MASTER_PORT:-29572}" \
    baseline/src/test_mri_ct.py \
    --model_type MMSyn \
    --data_json "$MRICT_JSON" \
    --logdir "$LOG_BASE" \
    --max_epochs 150 --batch_size 32 --workers 4 \
    --optim_lr 1e-4 --decay 0 --eta_min 1e-6 --val_every 5 --sample_every 500 --amp \
    --mmsyn_latent_dim 16 --mmsyn_fusion max --mmsyn_lambda_recon 1.0 --mmsyn_lambda_concat 1.0 \
    --mmsyn_ind_outs 1 --mmsyn_fuse_outs 1 \
    --checkpoint_path "$CHECKPOINT_PATH"

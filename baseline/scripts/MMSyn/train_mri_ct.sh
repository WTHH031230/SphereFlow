#!/bin/bash
# MMSyn (Chartsias et al., TMI 2018): train on MRI<->CT (2 modalities, both directions); the bidirectional
# test runs automatically at the end (test_results_MMSyn.json).
# env: sphereflow_baseline
# out: $OUTPUT_DIR/baselines/mri_ct/MMSyn/
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/env.sh"
cd "$REPO_DIR"
LOG_BASE="${LOG_BASE:-$OUTPUT_DIR/baselines/mri_ct}"
require_file "$MRICT_JSON"
mkdir -p "$LOG_BASE"
RESUME_ARGS=()
[[ -n "${RESUME_PATH:-}" ]] && RESUME_ARGS=(--resume_path "$RESUME_PATH")
accelerate launch --multi_gpu --num_processes "$NPROC" --gpu_ids "$GPUS" --main_process_port "${MASTER_PORT:-29562}" \
    baseline/src/train_mri_ct.py \
    --model_type MMSyn \
    --data_json "$MRICT_JSON" \
    --logdir "$LOG_BASE" \
    --max_epochs 150 --batch_size 32 --workers 4 \
    --optim_lr 1e-4 --decay 0 --eta_min 1e-6 --val_every 5 --sample_every 500 --amp \
    --mmsyn_latent_dim 16 --mmsyn_fusion max --mmsyn_lambda_recon 1.0 --mmsyn_lambda_concat 1.0 \
    --mmsyn_ind_outs 1 --mmsyn_fuse_outs 1 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

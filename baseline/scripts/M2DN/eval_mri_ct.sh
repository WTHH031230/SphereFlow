#!/bin/bash
# M2DN (Meng et al., TMI 2024), re-implemented: evaluate an MRI<->CT checkpoint (CT->MRI and MRI->CT) with
# test_mri_ct.py; writes test_results_M2DN.json to LOG_BASE/M2DN.
# env: sphereflow_baseline
# in : CHECKPOINT_PATH (default: $CKPT_DIR/baselines/mri_ct/M2DN.pt)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/env.sh"
cd "$REPO_DIR"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CKPT_DIR/baselines/mri_ct/M2DN.pt}"
LOG_BASE="${LOG_BASE:-$OUTPUT_DIR/eval/baselines_mri_ct}"
require_file "$MRICT_JSON" "$CHECKPOINT_PATH"
mkdir -p "$LOG_BASE"
accelerate launch --multi_gpu --num_processes "$NPROC" --gpu_ids "$GPUS" --main_process_port "${MASTER_PORT:-29541}" \
    baseline/src/test_mri_ct.py \
    --model_type M2DN \
    --data_json "$MRICT_JSON" \
    --logdir "$LOG_BASE" \
    --max_epochs 300 --batch_size 64 --workers 4 \
    --optim_lr 5e-4 --eta_min 1e-6 --decay 1e-5 --val_every 5 --sample_every 500 --amp \
    --m2dn_enc_ch 32 --m2dn_t_emb_dim 256 --m2dn_T 1000 --m2dn_n_inf_steps 18 --m2dn_setting paper \
    --checkpoint_path "$CHECKPOINT_PATH"

#!/bin/bash
# ModImp (Qi et al., MICCAI 2025): train on BraTS (4 modalities, random missing masks); a full
# modality-combination test runs automatically at the end (test_results_ModImp.json).
# env: sphereflow_baseline
# out: $OUTPUT_DIR/baselines/brats/ModImp/ckpt_ModImp_{best,latest}.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/env.sh"
cd "$REPO_DIR"
LOG_BASE="${LOG_BASE:-$OUTPUT_DIR/baselines/brats}"
require_file "$BRATS_JSON"
mkdir -p "$LOG_BASE"
RESUME_ARGS=()
[[ -n "${RESUME_PATH:-}" ]] && RESUME_ARGS=(--resume_path "$RESUME_PATH")
accelerate launch --multi_gpu --num_processes "$NPROC" --gpu_ids "$GPUS" --main_process_port "${MASTER_PORT:-29511}" \
    baseline/src/train_brats.py \
    --model_type ModImp \
    --data_json "$BRATS_JSON" \
    --logdir "$LOG_BASE" \
    --batch_size 64 --max_epochs 50 --val_every 5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

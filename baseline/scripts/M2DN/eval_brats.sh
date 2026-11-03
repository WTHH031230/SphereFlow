#!/bin/bash
# M2DN (Meng et al., TMI 2024), re-implemented: evaluate a BraTS checkpoint on all 14 missing-modality masks
# (PSNR / SSIM / rFID / IS per target).
# env: sphereflow_baseline
# in : CHECKPOINT_PATH (default: $CKPT_DIR/baselines/brats/M2DN.pt)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/env.sh"
cd "$REPO_DIR"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CKPT_DIR/baselines/brats/M2DN.pt}"
EVAL_DIR="${EVAL_DIR:-$OUTPUT_DIR/eval/baselines_brats/M2DN}"
require_file "$BRATS_JSON" "$CHECKPOINT_PATH"
mkdir -p "$EVAL_DIR"
accelerate launch --multi_gpu --num_processes "$NPROC" --gpu_ids "$GPUS" --main_process_port "${MASTER_PORT:-29540}" \
    baseline/src/test_brats.py \
    --model_type M2DN \
    --data_json "$BRATS_JSON" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_dir "$EVAL_DIR" \
    --batch_size 32 \
    --workers 4 \
    --amp \
    --m2dn_enc_ch 32 --m2dn_t_emb_dim 256 --m2dn_T 1000 --m2dn_n_inf_steps 18

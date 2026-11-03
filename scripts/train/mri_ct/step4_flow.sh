#!/bin/bash
# SphereFlow (MRI<->CT) Stage 2: bidirectional DiT^DH-XL flow on the frozen Step-3 RAE,
# flow loss + decoded-endpoint image loss (L1/SSIM 0.02) + orthogonal-residual loss (0.5).
# env: sphereflow
# in : RAE_PATH (default: step-3 output)
# out: $OUTPUT_DIR/sphereflow/mri_ct/step4_flow/all/checkpoints/ep-last.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$REPO_DIR"
FLOW_CONFIG="${FLOW_CONFIG:-configs/stage2/mri_ct.yaml}"
OUT="$OUTPUT_DIR/sphereflow/mri_ct"
RAE_PATH="${RAE_PATH:-$OUT/step3_shared/rae_step3.pt}"
RESULTS_DIR="${RESULTS_DIR:-$OUT/step4_flow}"
EXPERIMENT_NAME="all"
TMP_DIR="$RESULTS_DIR/configs"
FLOW_EPOCHS="${FLOW_EPOCHS:-400}"
FLOW_GLOBAL_BS="${FLOW_GLOBAL_BS:-256}"
FLOW_EVAL_INTERVAL="${FLOW_EVAL_INTERVAL:-1000}"
FLOW_CHECKPOINT_INTERVAL="${FLOW_CHECKPOINT_INTERVAL:-4}"
PRECISION="${PRECISION:-bf16}"
require_file "$FLOW_CONFIG" "$MRICT_JSON" "$RAE_PATH"
mkdir -p "$TMP_DIR"

FLOW_TRAIN_CONFIG="$TMP_DIR/flow.yaml"
python - "$FLOW_CONFIG" "$FLOW_TRAIN_CONFIG" "$FLOW_EPOCHS" "$FLOW_GLOBAL_BS" \
    "$FLOW_CHECKPOINT_INTERVAL" "$FLOW_EVAL_INTERVAL" <<'PY'
import sys, yaml
src, out, epochs, bs, ckpt_int, eval_int = sys.argv[1:]
cfg = yaml.safe_load(open(src))
cfg.setdefault("training", {})
cfg["training"].update(epochs=int(epochs), global_batch_size=int(bs), checkpoint_interval=int(ckpt_int),
                       sample_every=500, log_interval=100)
cfg["training"].setdefault("scheduler", {})["decay_end_epoch"] = int(epochs)
cfg.setdefault("eval", {}).update(eval_interval=int(eval_int), metrics=["rfid", "psnr", "ssim", "is"])
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
PY

CUDA_VISIBLE_DEVICES=$GPUS EXPERIMENT_NAME=$EXPERIMENT_NAME torchrun \
    --standalone --nproc_per_node=$NPROC \
    src/train_flow.py \
    --config "$FLOW_TRAIN_CONFIG" \
    --data-path "$MRICT_JSON" \
    --dataset mri_ct \
    --rae-path "$RAE_PATH" \
    --results-dir "$RESULTS_DIR" \
    --image-size 256 \
    --precision "$PRECISION" \
    --global-seed 42 \
    --latest-checkpoint-interval 1 \
    --save-best-rfid \
    --flow-weight 1.0 \
    --image-l1-weight 0.02 \
    --image-ssim-weight 0.02 \
    --orth-weight 0.5 \
    --image-min-t 0.0 \
    --orth-min-t 0.0 \
    ${LOGGER_ARGS[@]+"${LOGGER_ARGS[@]}"}

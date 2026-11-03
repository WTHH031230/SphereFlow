#!/bin/bash
# SphereFlow on BraTS:
#   1) image metrics (PSNR / SSIM / rFID / IS) for every (target, observed-set) combination on the
#      full test set, 2-step Euler ODE;
#   2) Dice of the 3D U-Net segmenter on the imputed volumes for all 15 availability masks;
#   3) per-mask summary (suite_summary.json): 1-to-1 = masks with one observed modality,
#      N-to-N = all 14 incomplete masks.
# env: sphereflow
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$REPO_DIR"

BASE_CONFIG="${BASE_CONFIG:-configs/stage2/brats.yaml}"
RAE_PATH="${RAE_PATH:-$CKPT_DIR/sphereflow/brats/rae.pt}"
RAE_FLOW="${RAE_FLOW:-$CKPT_DIR/sphereflow/brats/flow.pt}"
UNET_CKPT="${UNET_CKPT:-$CKPT_DIR/unet3d/brats.pt}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_DIR/eval/sphereflow_brats}"
STEP="${STEP:-2}"                         # Euler ODE steps
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-20080}"   # full BraTS test split
NUM_CASES="${NUM_CASES:--1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
SLICE_BATCH="${SLICE_BATCH:-8}"
CHUNK_DEPTH="${CHUNK_DEPTH:-0}"
SKIP_DICE="${SKIP_DICE:-0}"
PATTERNS=(0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111)
require_file "$BRATS_JSON" "$RAE_PATH" "$RAE_FLOW"

CONFIG_DIR="$RESULTS_DIR/configs"; IMAGE_DIR="$RESULTS_DIR/image_metrics"
DICE_JSON="$RESULTS_DIR/dice_all_patterns.json"
mkdir -p "$CONFIG_DIR" "$IMAGE_DIR"

STEP_CONFIG="$CONFIG_DIR/$(basename "$BASE_CONFIG" .yaml)_steps${STEP}.yaml"
python - "$BASE_CONFIG" "$STEP_CONFIG" "$STEP" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True))
cfg.sampler.params.num_steps = int(sys.argv[3])
OmegaConf.save(cfg, sys.argv[2])
PY

echo "[1/3] image metrics"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC \
    src/eval_brats.py \
    --config "$STEP_CONFIG" \
    --data-path "$BRATS_JSON" \
    --rae-path "$RAE_PATH" \
    --flow-ckpt "$RAE_FLOW" \
    --results-dir "$IMAGE_DIR" \
    --image-size 256 \
    --precision "$AMP_DTYPE" \
    --sampler-mode ODE \
    --num-samples 1 \
    --max-eval-samples "$MAX_EVAL_SAMPLES" \
    --global-seed 42

if [[ "$SKIP_DICE" == "1" ]]; then exit 0; fi
require_file "$UNET_CKPT"
echo "[2/3] Dice with the 3D U-Net (all 15 masks)"
CUDA_VISIBLE_DEVICES=$GPUS RAE_PROJECT_ROOT="$REPO_DIR" torchrun --standalone --nproc_per_node=$NPROC \
    "$REPO_DIR/downstream/src/eval_completion_sphereflow.py" \
    --ckpt "$UNET_CKPT" \
    --json-path "$BRATS_JSON" \
    --phase test \
    --patterns "${PATTERNS[@]}" \
    --num-cases "$NUM_CASES" \
    --slice-batch "$SLICE_BATCH" \
    --chunk-depth "$CHUNK_DEPTH" \
    --output-json "$DICE_JSON" \
    --amp-dtype "$AMP_DTYPE" \
    --rae-config "$STEP_CONFIG" \
    --rae-path "$RAE_PATH" \
    --rae-flow "$RAE_FLOW" \
    --sampler-num-steps "$STEP"

echo "[3/3] per-mask summary"
python - "$IMAGE_DIR/all_combinations_metrics.json" "$DICE_JSON" "$RESULTS_DIR/suite_summary.json" "$STEP" <<'PY'
import json, sys
image = json.load(open(sys.argv[1])); dice = json.load(open(sys.argv[2]))["results"]
patterns = ["0001", "0010", "0011", "0100", "0101", "0110", "0111",
            "1000", "1001", "1010", "1011", "1100", "1101", "1110", "1111"]
records = []
for p in patterns:
    known = [i for i, c in enumerate(p) if c == "1"]
    missing = [i for i, c in enumerate(p) if c == "0"]
    keys = [f"target_{t}_from_{'_'.join(map(str, known))}" for t in missing]
    rec = {"pattern": p, "sample_steps": int(sys.argv[4]), "known_modalities": known,
           "missing_modalities": missing, "image_target_keys": keys,
           "dice_count": int(dice[p]["count"]), "dice_wt": float(dice[p]["WT"]),
           "dice_tc": float(dice[p]["TC"]), "dice_et": float(dice[p]["ET"]),
           "dice_mean": float(dice[p]["mean"])}
    for name in ["psnr", "ssim", "rfid", "is"]:
        vals = [float(image[k][name]) for k in keys]
        rec[name] = sum(vals) / len(vals) if vals else None
    records.append(rec)
json.dump(records, open(sys.argv[3], "w"), indent=2, sort_keys=True)
print(f"wrote {sys.argv[3]}")
PY

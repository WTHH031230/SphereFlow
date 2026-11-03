#!/bin/bash
# Downstream segmentation table: impute the missing modalities of every BraTS test case with each
# method, segment with the frozen 3D U-Net and report Dice / HD95 / IoU / Sensitivity per mask.
#   step 1 (sphereflow_baseline env): eval_completion.py            -- Zero-fill + baselines
#   step 2 (sphereflow env)       : eval_completion_sphereflow.py -- SphereFlow
#   step 3 (sphereflow_baseline env): aggregate_results.py          -- partial_*.json -> results.json / results.csv
# Both conda envs must exist; CONDA_BASE defaults to `conda info --base`.
# Baseline checkpoints are read from $CKPT_DIR/baselines/brats/... (see checkpoints/README.md).
# Skip steps with SKIP_INLINE=1, SKIP_SPHEREFLOW=1, SKIP_AGGREGATE=1; choose baselines with e.g.
#   INLINE_METHODS="Zhang MMSyn" bash downstream/scripts/eval.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../scripts/env.sh"
export BRATS_JSON
cd "$SCRIPT_DIR/../src"

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: cannot locate conda; set CONDA_BASE explicitly." >&2
    exit 1
fi

BASELINE_TORCHRUN="${BASELINE_TORCHRUN:-$CONDA_BASE/envs/sphereflow_baseline/bin/torchrun}"
BASELINE_PYTHON="${BASELINE_PYTHON:-$CONDA_BASE/envs/sphereflow_baseline/bin/python}"
SPHEREFLOW_TORCHRUN="${SPHEREFLOW_TORCHRUN:-$CONDA_BASE/envs/sphereflow/bin/torchrun}"
for bin in "$BASELINE_TORCHRUN" "$BASELINE_PYTHON" "$SPHEREFLOW_TORCHRUN"; do
    if [ ! -x "$bin" ]; then
        echo "ERROR: missing executable $bin" >&2
        exit 1
    fi
done

CKPT="${UNET_CKPT:-$CKPT_DIR/unet3d/brats.pt}"
EVAL_DIR="${EVAL_DIR:-$OUTPUT_DIR/eval/segmentation}"
SPHEREFLOW_NPROC="${SPHEREFLOW_NPROC:-$NPROC}"
INLINE_METHODS="${INLINE_METHODS:-Zero MMT M2DN ModImp Zhang MMGAN MMSyn}"
CHUNK_DEPTH="${CHUNK_DEPTH:-0}"
# Per-rank slice batch for the 2D completion methods; lower it if GPU memory is short.
SLICE_BATCH="${SLICE_BATCH:-64}"
# Per-rank UNet (segmentation) batch. We complete one volume at a time but
# stack patterns of the same case for the UNet forward to amortise overhead.
UNET_BATCH="${UNET_BATCH:-4}"
SPHEREFLOW_UNET_BATCH="${SPHEREFLOW_UNET_BATCH:-2}"
# SphereFlow ODE sampler steps. 0 = use the value in the YAML config.
SAMPLER_NUM_STEPS="${SAMPLER_NUM_STEPS:-2}"

RAE_CONFIG="${RAE_CONFIG:-$REPO_DIR/configs/stage2/brats.yaml}"
# SphereFlow (BraTS) weights: RAE (step-3 shared encoder + fine-tuned decoders), DiTDH-XXL flow
RAE_PATH="${RAE_PATH:-$CKPT_DIR/sphereflow/brats/rae.pt}"
RAE_FLOW="${RAE_FLOW:-$CKPT_DIR/sphereflow/brats/flow.pt}"
require_file "$CKPT"

ALL_PATTERNS=(1000 0100 0010 0001 1100 1010 1001 0110 0101 0011 1110 1101 1011 0111)

export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$EVAL_DIR"
LOG_FILE="$EVAL_DIR/eval.log"

echo "============================================================" | tee "$LOG_FILE"
echo "Chained evaluation start: $(date)"        | tee -a "$LOG_FILE"
echo "  CKPT            = $CKPT"                | tee -a "$LOG_FILE"
echo "  EVAL_DIR        = $EVAL_DIR"          | tee -a "$LOG_FILE"
echo "  NPROC (inline)  = $NPROC"               | tee -a "$LOG_FILE"
echo "  SPHEREFLOW_NPROC = $SPHEREFLOW_NPROC"           | tee -a "$LOG_FILE"
echo "  GPUS            = $GPUS"                | tee -a "$LOG_FILE"
echo "  INLINE_METHODS  = $INLINE_METHODS"      | tee -a "$LOG_FILE"
echo "  CHUNK_DEPTH     = $CHUNK_DEPTH"         | tee -a "$LOG_FILE"
echo "  SLICE_BATCH     = $SLICE_BATCH"         | tee -a "$LOG_FILE"
echo "  UNET_BATCH      = $UNET_BATCH"          | tee -a "$LOG_FILE"
echo "  SPHEREFLOW_UNET_BATCH = $SPHEREFLOW_UNET_BATCH"      | tee -a "$LOG_FILE"
echo "  SKIP_INLINE     = ${SKIP_INLINE:-0}"    | tee -a "$LOG_FILE"
echo "  SKIP_SPHEREFLOW = ${SKIP_SPHEREFLOW:-0}"       | tee -a "$LOG_FILE"
echo "  SKIP_AGGREGATE  = ${SKIP_AGGREGATE:-0}" | tee -a "$LOG_FILE"
echo "  EXTRA passthru  = $*"                   | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"


# ----------------------------------------------------------------------------
# Step 1: inline methods (sphereflow_baseline env)
# ----------------------------------------------------------------------------
if [ "${SKIP_INLINE:-0}" != "1" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "===== STEP 1/3: inline methods ($INLINE_METHODS) =====" | tee -a "$LOG_FILE"
    "$BASELINE_TORCHRUN" --standalone --nproc_per_node=$NPROC \
        eval_completion.py \
        --ckpt          "$CKPT" \
        --output-dir    "$EVAL_DIR" \
        --methods       $INLINE_METHODS \
        --chunk-depth   "$CHUNK_DEPTH" \
        --slice-batch   "$SLICE_BATCH" \
        --unet-batch    "$UNET_BATCH" \
        "$@" 2>&1 | tee -a "$LOG_FILE"
else
    echo "(SKIP_INLINE=1; skipping step 1)" | tee -a "$LOG_FILE"
fi


# ----------------------------------------------------------------------------
# Step 2: SphereFlow worker (sphereflow env, separate DDP world to avoid NCCL deadlock)
# ----------------------------------------------------------------------------
if [ "${SKIP_SPHEREFLOW:-0}" != "1" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "===== STEP 2/3: SphereFlow worker (conda env: sphereflow) =====" | tee -a "$LOG_FILE"
    env -u PYTHONPATH \
    "$SPHEREFLOW_TORCHRUN" --standalone --nproc_per_node=$SPHEREFLOW_NPROC \
        eval_completion_sphereflow.py \
        --ckpt          "$CKPT" \
        --output-json   "$EVAL_DIR/partial_SphereFlow.json" \
        --patterns      "${ALL_PATTERNS[@]}" \
        --rae-config    "$RAE_CONFIG" \
        --rae-path      "$RAE_PATH" \
        --rae-flow      "$RAE_FLOW" \
        --chunk-depth   "$CHUNK_DEPTH" \
        --slice-batch   "$SLICE_BATCH" \
        --unet-batch    "$SPHEREFLOW_UNET_BATCH" \
        --sampler-num-steps "$SAMPLER_NUM_STEPS" \
        2>&1 | tee -a "$LOG_FILE"
else
    echo "(SKIP_SPHEREFLOW=1; skipping step 2)" | tee -a "$LOG_FILE"
fi


# ----------------------------------------------------------------------------
# Step 3: aggregate all partials -> results.json / results.csv
# ----------------------------------------------------------------------------
if [ "${SKIP_AGGREGATE:-0}" != "1" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "===== STEP 3/3: aggregate partials =====" | tee -a "$LOG_FILE"
    "$BASELINE_PYTHON" aggregate_results.py --partials-dir "$EVAL_DIR" 2>&1 | tee -a "$LOG_FILE"
else
    echo "(SKIP_AGGREGATE=1; skipping step 3)" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Chained evaluation done: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

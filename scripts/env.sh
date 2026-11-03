# Shared defaults for every script in scripts/, baseline/scripts/ and downstream/scripts/. Source it, do not execute it.
# Every variable can be overridden from the calling environment, e.g.
#   GPUS=0,1 NPROC=2 CKPT_DIR=/path/to/ckpts bash scripts/eval/brats.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR

# Materialized split JSONs (see scripts/preprocess/make_splits.sh).
export DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
export BRATS_JSON="${BRATS_JSON:-$DATA_DIR/brats2023_distribution.json}"
export MRICT_JSON="${MRICT_JSON:-$DATA_DIR/brain_mri_ct_distribution.json}"

# Pretrained / released checkpoints (layout: checkpoints/README.md) and run outputs.
export CKPT_DIR="${CKPT_DIR:-$REPO_DIR/checkpoints}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs}"

# GPUs.
export GPUS="${GPUS:-0,1,2,3}"
export NPROC="${NPROC:-4}"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUTPUT_DIR/.mplconfig}"
mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"

# Experiment tracking. The baseline trainer always logs through SwanLab (via accelerate);
# default to offline mode so no account is needed. SphereFlow scripts only log to
# SwanLab when USE_SWANLAB=1.
export SWANLAB_MODE="${SWANLAB_MODE:-offline}"
LOGGER_ARGS=()
if [[ "${USE_SWANLAB:-0}" == "1" ]]; then
    LOGGER_ARGS=(--swanlab)
fi

require_file() {
    for f in "$@"; do
        if [[ ! -f "$f" ]]; then
            echo "ERROR: required file not found: $f" >&2
            exit 1
        fi
    done
}

#!/bin/bash
# SphereFlow (MRI<->CT) Stage 1, step (iii): shared-encoder fine-tuning with LoRA (rank 8) on the
# DINOv2 encoder + both decoders, with geometry losses (token anchor / velocity-cosine /
# velocity-norm) and PSNR/SSIM terms; the EMA weights are then merged into a plain RAE checkpoint.
# env: sphereflow
# in : step-2 checkpoints
# out: $OUTPUT_DIR/sphereflow/mri_ct/step3_shared/rae_step3.pt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$REPO_DIR"

STEP3_CONFIG="${STEP3_CONFIG:-configs/stage1/mri_ct_shared.yaml}"
STEP2_DIR="${STEP2_DIR:-$OUTPUT_DIR/sphereflow/mri_ct/step2_interp}"
STEP2_CKPT_0="${STEP2_CKPT_0:-$STEP2_DIR/mri/checkpoints/ep-last.pt}"
STEP2_CKPT_1="${STEP2_CKPT_1:-$STEP2_DIR/ct/checkpoints/ep-last.pt}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_DIR/sphereflow/mri_ct/step3_shared}"
EXPERIMENT_NAME="all"
TMP_DIR="$RESULTS_DIR/configs"

STEP3_EPOCHS="${STEP3_EPOCHS:-150}"
STEP3_CHECKPOINT_INTERVAL="${STEP3_CHECKPOINT_INTERVAL:-10}"
STEP3_GLOBAL_BS="${STEP3_GLOBAL_BS:-64}"
PRECISION="${PRECISION:-bf16}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_LAST_N="${LORA_LAST_N:-0}"
LORA_TARGETS="${LORA_TARGETS:-query,key,value,dense,fc1,fc2,qkv,proj}"
LAMBDA_TOKEN_ANCHOR="${LAMBDA_TOKEN_ANCHOR:-2.0}"
LAMBDA_VEL_COS="${LAMBDA_VEL_COS:-0.5}"
LAMBDA_VNORM="${LAMBDA_VNORM:-0.1}"
STEP3_PSNR_WEIGHT="${STEP3_PSNR_WEIGHT:-0.005}"
STEP3_SSIM_WEIGHT="${STEP3_SSIM_WEIGHT:-0.05}"
OFFLOAD_ARGS=()
[[ "${OFFLOAD_DECODERS:-0}" == "1" ]] && OFFLOAD_ARGS=(--offload-decoders)
require_file "$STEP2_CKPT_0" "$STEP2_CKPT_1" "$MRICT_JSON"
mkdir -p "$TMP_DIR"

STEP3_RUN_CONFIG="$TMP_DIR/step3.yaml"
python - "$STEP3_CONFIG" "$STEP3_RUN_CONFIG" "$STEP3_EPOCHS" "$STEP3_GLOBAL_BS" "$STEP3_CHECKPOINT_INTERVAL" <<'PY'
import sys, yaml
src, out, epochs, bs, ckpt_int = sys.argv[1:]
cfg = yaml.safe_load(open(src))
cfg.setdefault("training", {})
cfg["training"].update(epochs=int(epochs), global_batch_size=int(bs), checkpoint_interval=int(ckpt_int),
                       log_interval=100)
cfg["training"].setdefault("scheduler", {})["decay_end_epoch"] = int(epochs)
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
PY

CUDA_VISIBLE_DEVICES=$GPUS EXPERIMENT_NAME=$EXPERIMENT_NAME torchrun \
    --standalone --nproc_per_node=$NPROC \
    src/train_shared_encoder.py \
    --config "$STEP3_RUN_CONFIG" \
    --stage1-path-0 "$STEP2_CKPT_0" \
    --stage1-path-1 "$STEP2_CKPT_1" \
    --cos-weight 0.0 \
    --mse-weight 0.0 \
    --lambda-token-anchor "$LAMBDA_TOKEN_ANCHOR" \
    --lambda-vel-cos "$LAMBDA_VEL_COS" \
    --lambda-vnorm "$LAMBDA_VNORM" \
    --encoder-lora-rank "$LORA_RANK" \
    --encoder-lora-alpha "$LORA_ALPHA" \
    --encoder-lora-dropout "$LORA_DROPOUT" \
    --encoder-lora-last-n "$LORA_LAST_N" \
    --encoder-lora-targets "$LORA_TARGETS" \
    --data-path "$MRICT_JSON" \
    --dataset mri_ct \
    --results-dir "$RESULTS_DIR" \
    --precision "$PRECISION" \
    --global-batch-size "$STEP3_GLOBAL_BS" \
    --w-sampling-mode power \
    --no-gradient-checkpointing \
    --psnr-weight "$STEP3_PSNR_WEIGHT" \
    --ssim-weight "$STEP3_SSIM_WEIGHT" \
    ${OFFLOAD_ARGS[@]+"${OFFLOAD_ARGS[@]}"} \
    ${LOGGER_ARGS[@]+"${LOGGER_ARGS[@]}"}

python src/tools/export_lora_ema_checkpoint.py \
    --input "$RESULTS_DIR/$EXPERIMENT_NAME/checkpoints/ep-last.pt" \
    --output "$RESULTS_DIR/rae_step3.pt" \
    --state-key ema \
    --lora-alpha "$LORA_ALPHA"

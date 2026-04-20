#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train/run_stage.sh <stage1|stage2|stage3> [config_path]"
  exit 1
fi

STAGE_NAME="$1"
CONFIG_PATH="${2:-}"

case "$STAGE_NAME" in
  stage1)
    DEFAULT_CONFIG="configs/stages/stage1_qwen2_vl_7b.json"
    ;;
  stage2)
    DEFAULT_CONFIG="configs/stages/stage2_qwen2_vl_7b.json"
    ;;
  stage3)
    DEFAULT_CONFIG="configs/stages/stage3_qwen2_vl_7b.json"
    ;;
  *)
    echo "Unknown stage: $STAGE_NAME"
    echo "Supported stages: stage1, stage2, stage3"
    exit 1
    ;;
esac

if [[ -z "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$DEFAULT_CONFIG"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Stage config does not exist: $CONFIG_PATH"
  exit 1
fi

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-24001}"
REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"

if [[ "$REPORT_TO" == "wandb" && "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "[WARN] WANDB_API_KEY is not set; fallback to offline mode."
  export WANDB_MODE=offline
fi

echo "[INFO] stage=${STAGE_NAME} config=${CONFIG_PATH}"
echo "[INFO] gpu=${GPU_LIST} master_port=${MASTER_PORT} report_to=${REPORT_TO} wandb_mode=${WANDB_MODE}"

deepspeed --include "localhost:${GPU_LIST}" --master_port "$MASTER_PORT" \
  jarvisvla/train/train.py \
  --stage_name "$STAGE_NAME" \
  --stage_config_path "$CONFIG_PATH" \
  --report_to "$REPORT_TO"

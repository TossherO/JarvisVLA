#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train/run_stage.sh <config_path>"
  exit 1
fi

CONFIG_PATH="$1"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Stage config does not exist: $CONFIG_PATH"
  exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_LIST="$CUDA_VISIBLE_DEVICES"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    GPU_LIST="0"
  fi
fi
MASTER_PORT="${MASTER_PORT:-24001}"
export WANDB_MODE="${WANDB_MODE:-offline}"

deepspeed --include "localhost:${GPU_LIST}" --master_port "$MASTER_PORT" \
  jarvisvla/train/train.py \
  --stage_config_path "$CONFIG_PATH"

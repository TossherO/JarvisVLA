#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train/run_stage_multinode.sh <config_path> [hostfile_path]"
  exit 1
fi

CONFIG_PATH="$1"
HOSTFILE_PATH="${2:-scripts/train/hostfile}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Stage config does not exist: $CONFIG_PATH"
  exit 1
fi

if [[ ! -f "$HOSTFILE_PATH" ]]; then
  echo "Hostfile does not exist: $HOSTFILE_PATH"
  exit 1
fi

MASTER_PORT="${MASTER_PORT:-24001}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# 可选：通过 DEEPSPEED_INCLUDE 指定节点与 GPU 映射。
# 例如：DEEPSPEED_INCLUDE="hgx1:0,1,2,3@hgx2:0,1,2,3"
if [[ -n "${DEEPSPEED_INCLUDE:-}" ]]; then
  deepspeed --hostfile "$HOSTFILE_PATH" --include "$DEEPSPEED_INCLUDE" --master_port "$MASTER_PORT" \
    jarvisvla/train/train.py \
    --stage_config_path "$CONFIG_PATH"
else
  deepspeed --hostfile "$HOSTFILE_PATH" --master_port "$MASTER_PORT" \
    jarvisvla/train/train.py \
    --stage_config_path "$CONFIG_PATH"
fi

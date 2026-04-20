#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/stages/stage3_qwen2_vl_7b.json}"
bash scripts/train/run_stage.sh stage3 "$CONFIG_PATH"

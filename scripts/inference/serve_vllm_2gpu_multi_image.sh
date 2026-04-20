#!/bin/bash

# Two-GPU vLLM launch profile for JarvisVLA Qwen2-VL 7B.
# This profile keeps memory usage conservative while enabling multi-image requests.

set -euo pipefail

cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-0,1}
# model_name_or_path=${MODEL_PATH:-/share/public_datasets/VLA/nitrogen/jarvisvla_models/JarvisVLA-Qwen2-VL-7B}
model_name_or_path=${MODEL_PATH:-/share/public_datasets/VLA/nitrogen/jarvisvla_models/qwen2-vl-stage3-test1-c1-e1-b8-a1}
port=${PORT:-8000}
tp_size=${TP_SIZE:-2}
max_model_len=${MAX_MODEL_LEN:-8448}
max_num_seqs=${MAX_NUM_SEQS:-10}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8448}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.8}
image_limit=${IMAGE_LIMIT:-6}
served_model_name=${SERVED_MODEL_NAME:-jarvisvla}

echo "[vLLM] CUDA_VISIBLE_DEVICES=${cuda_visible_devices}"
echo "[vLLM] TP_SIZE=${tp_size}, MAX_MODEL_LEN=${max_model_len}, MAX_NUM_SEQS=${max_num_seqs}, MAX_NUM_BATCHED_TOKENS=${max_num_batched_tokens}"
echo "[vLLM] GPU_MEMORY_UTILIZATION=${gpu_memory_utilization}, IMAGE_LIMIT=${image_limit}"

CUDA_VISIBLE_DEVICES=${cuda_visible_devices} vllm serve "${model_name_or_path}" \
  --port "${port}" \
  --tensor-parallel-size "${tp_size}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --max-model-len "${max_model_len}" \
  --max-num-seqs "${max_num_seqs}" \
  --max-num-batched-tokens "${max_num_batched_tokens}" \
  --trust-remote-code \
  --served-model-name "${served_model_name}" \
  --limit-mm-per-prompt "image=${image_limit}"

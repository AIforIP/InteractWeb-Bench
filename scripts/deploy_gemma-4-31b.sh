#!/bin/bash

# Automatically read and parse the .env file
if [ -f ../.env ]; then
  set -a; source ../.env; set +a
elif [ -f .env ]; then
  set -a; source .env; set +a
fi

# This script launches Node 5 and reads the port from NODE_5 configuration
TARGET_PORT=${LOCAL_NODE_5_PORT:-8028}

echo "Starting vLLM Node 5 on port: $TARGET_PORT"

# Kill any process currently using the target port to avoid conflicts
fuser -k ${TARGET_PORT}/tcp >/dev/null 2>&1

MODEL_PATH="/data/shared/users/wangqiyao/models/gemma-4-31B-it"

# Launch a vLLM inference server for Gemma-4-31B with multi-GPU support
CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "gemma-4-31B-it" \
    --port $TARGET_PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --max-model-len 128000 \
    --reasoning-parser gemma4 \
    --limit-mm-per-prompt '{"image": 5, "video": 0}' \
    --gpu-memory-utilization 0.85 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --tensor-parallel-size 2
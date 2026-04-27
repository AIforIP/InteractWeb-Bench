#!/bin/bash

# Automatically read and parse the .env file
if [ -f ../.env ]; then
  set -a; source ../.env; set +a
elif [ -f .env ]; then
  set -a; source .env; set +a
fi

# This script launches Node 2 and reads the port from NODE_2 configuration
TARGET_PORT=${LOCAL_NODE_2_PORT:-8025}

echo "Starting vLLM Node 2 on port: $TARGET_PORT"

# Kill any process currently using the target port to avoid conflicts
fuser -k ${TARGET_PORT}/tcp >/dev/null 2>&1

MODEL_PATH="/data/shared/users/wangqiyao/models/Qwen3.5-9B"

# Launch a vLLM inference server for Qwen3.5-9B
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "Qwen3.5-9B" \
    --port $TARGET_PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --max-model-len 128000 \
    --limit-mm-per-prompt '{"image": 5, "video": 0}' \
    --gpu-memory-utilization 0.85 \
    --enforce-eager
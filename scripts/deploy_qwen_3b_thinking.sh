#!/bin/bash

# 自动读取并解析 .env 文件
if [ -f ../.env ]; then
  set -a; source ../.env; set +a
elif [ -f .env ]; then
  set -a; source .env; set +a
fi

# 该脚本作为节点 1 启动，读取 NODE_1 的端口
TARGET_PORT=${LOCAL_NODE_1_PORT:-8024}

echo "Starting vLLM Node 1 on port: $TARGET_PORT"

fuser -k ${TARGET_PORT}/tcp >/dev/null 2>&1

MODEL_PATH="/home/hhr/home/models/Qwen3-VL-2B-Thinking"

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "Qwen3-VL-2B-Thinking" \
    --port $TARGET_PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 128000 \
    --limit-mm-per-prompt '{"image": 5}' \
    --gpu-memory-utilization 0.7 \
    --enforce-eager
#!/bin/bash

# 自动读取并解析 .env 文件
if [ -f ../.env ]; then
  set -a; source ../.env; set +a
elif [ -f .env ]; then
  set -a; source .env; set +a
fi

# 该脚本作为节点 1 启动，读取 NODE_1 的端口
TARGET_PORT=${LOCAL_NODE_4_PORT:-8027}

echo "Starting vLLM Node 4 on port: $TARGET_PORT"

fuser -k ${TARGET_PORT}/tcp >/dev/null 2>&1

MODEL_PATH="/data/shared/users/wangqiyao/models/Qwen3.5-27B"

CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "Qwen3.5-27B" \
    --port $TARGET_PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --max-model-len 128000 \
    --limit-mm-per-prompt '{"image": 5, "video": 0}' \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --enforce-eager
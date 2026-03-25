#!/bin/bash
# 自动清理 8024 端口，防止启动冲突
fuser -k 8024/tcp >/dev/null 2>&1

MODEL_PATH="/data/shared/users/wangqiyao/models/Qwen3.5-9B"

# 使用 Python 模块的绝对路径启动
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "Qwen3.5-9B" \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --max-model-len 16384 \
    --limit-mm-per-prompt '{"image": 5}' \
    --gpu-memory-utilization 0.7 \
    --tensor-parallel-size 1 \
    --enforce-eager
#!/bin/bash
# 自动清理 8024 端口，防止启动冲突
fuser -k 8024/tcp >/dev/null 2>&1

MODEL_PATH="/home/hhr/home/hhr/models/Qwen3-VL-2B-Thinking"

# 使用 Python 模块的绝对路径启动
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "Qwen/Qwen3-VL-2B-Thinking" \
    --port 8024 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --limit-mm-per-prompt '{"image": 5}' \
    --gpu-memory-utilization 0.7 \
    --enforce-eager
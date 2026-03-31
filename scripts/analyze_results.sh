#!/bin/bash
# Usage: bash scripts/analyze_results.sh <LOG_DIR>

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

python3 /app/src/experiment/result_analyze.py \
    --dir "/app/experiment_results/kimi-k2.5/logs" \
    --data "/app/data"

#!/bin/bash
# Usage: bash scripts/remove_error_tasks.sh <EXPERIMENT_DIR>
# Example: bash scripts/remove_error_tasks.sh experiment_results/Qwen3.5-9B
#
# Deletes from logs/ and workspaces/ any task folders that exist in error_logs/.

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <EXPERIMENT_DIR>"
    echo "Example: $0 experiment_results/Qwen3.5-9B"
    exit 1
fi

EXPERIMENT_DIR="$1"
ERROR_LOGS_DIR="$EXPERIMENT_DIR/error_logs"
LOGS_DIR="$EXPERIMENT_DIR/logs"
WORKSPACES_DIR="$EXPERIMENT_DIR/workspaces"

if [ ! -d "$ERROR_LOGS_DIR" ]; then
    echo "❌ error_logs directory not found: $ERROR_LOGS_DIR"
    exit 1
fi

echo "🔍 Scanning error_logs: $ERROR_LOGS_DIR"
echo "   → will delete matching entries from: $LOGS_DIR"
echo "   → will delete matching entries from: $WORKSPACES_DIR"
echo ""

deleted_logs=0
deleted_ws=0
skipped=0

for task_dir in "$ERROR_LOGS_DIR"/*/; do
    task_id=$(basename "$task_dir")

    log_target="$LOGS_DIR/$task_id"
    ws_target="$WORKSPACES_DIR/$task_id"

    if [ -d "$log_target" ]; then
        rm -rf "$log_target"
        echo "  🗑️  logs/$task_id"
        deleted_logs=$((deleted_logs + 1))
    else
        skipped=$((skipped + 1))
    fi

    if [ -d "$ws_target" ]; then
        rm -rf "$ws_target"
        echo "  🗑️  workspaces/$task_id"
        deleted_ws=$((deleted_ws + 1))
    fi
done

echo ""
echo "✅ Done."
echo "   Deleted from logs/:      $deleted_logs"
echo "   Deleted from workspaces: $deleted_ws"
echo "   Not found in logs/:      $skipped"

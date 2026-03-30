#!/bin/bash
# QuantRabbit — Claudeアカウント移行スクリプト
# 最新のメモリ・タスク定義を復元（上書き）

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== QuantRabbit Migration ==="
echo ""

# 1. ユーザーメモリ — 全上書き
MEMORY_DEST="$HOME/.claude/projects/-Users-tossaki-App-QuantRabbit/memory"
mkdir -p "$MEMORY_DEST"
cp -f "$SCRIPT_DIR/user_memory/"* "$MEMORY_DEST/"
echo "[OK] ユーザーメモリ復元 ($(ls "$SCRIPT_DIR/user_memory/" | wc -l | tr -d ' ')ファイル)"

# 2. Scheduled Tasks — 全上書き
for task_dir in "$SCRIPT_DIR"/scheduled_tasks_*/; do
    task_name=$(basename "$task_dir" | sed 's/scheduled_tasks_//')
    dest="$HOME/.claude/scheduled-tasks/$task_name"
    mkdir -p "$dest"
    cp -f "$task_dir"* "$dest/"
    echo "[OK] Scheduled Task復元: $task_name"
done

echo ""
echo "=== 復元完了 ==="
echo ""
echo "次のステップ:"
echo "1. Claude Codeで /Users/tossaki/App/QuantRabbit を開く"
echo "2. Claudeに「traderタスクを有効化して」と言う"
echo "3. 「トレード開始」で即動く"

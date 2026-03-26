#!/bin/bash
# Trade scheduled tasks setup script (v5)
# Usage: bash tools/setup_scheduled_tasks.sh
#
# v5 architecture:
#   Claude Code: trader only (Opus, 2-3min)
#   Cowork:      analyst (Sonnet, 10min) + secretary (11min) + news (15min)
#
# This script only sets up the Claude Code trader task.
# Cowork tasks are managed via Cowork's scheduled task UI.

set -e
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TASKS_DIR="$HOME/.claude/scheduled-tasks"

echo "=== QuantRabbit v5 — Claude Code Tasks Setup ==="
echo "Repo: $REPO_DIR"
echo "Tasks dir: $TASKS_DIR"
echo ""
echo "Architecture:"
echo "  Claude Code: trader (Opus, */3)"
echo "  Cowork:      quantrabbit-analyst (*/10)"
echo "  Cowork:      quantrabbit-secretary (*/11)"
echo "  Cowork:      quantrabbit-news (*/15)"
echo ""

# --- trader (the only Claude Code task) ---
mkdir -p "$TASKS_DIR/trader"
cat > "$TASKS_DIR/trader/SKILL.md" << 'SKILL_EOF'
---
name: trader
description: 凄腕プロトレーダー — 市場を読み裁量でトレードする (2-3分間隔)
---

docs/TRADER_PROMPT.md を読み、その指示に従って実行せよ。

あなたは凄腕プロトレーダーだ。ボットじゃない。市場を自分で読み、自分の頭で判断し、最適な一手を打て。

作業ディレクトリは __REPO_DIR__。

重要:
- docs/TRADER_PROMPT.md に全てが書いてある。まず読め
- shared_state.json を毎サイクル読め（analyst/secretary/newsがCoworkから書いてくれている）
- quality_alertがtriggeredなら全てに優先して対応
- 注文は logs/live_trade_log.txt に記録
- 常駐スクリプト(while True + sleep)は絶対に書かない
- 1サイクル2-3分で完結。次のサイクルは新しいコンテキスト
SKILL_EOF
sed -i '' "s|__REPO_DIR__|$REPO_DIR|g" "$TASKS_DIR/trader/SKILL.md"
echo "[OK] trader"

# --- Disable old v3/v4 tasks ---
for old_task in scalp-trader market-radar macro-intel secretary analyst swing-trader scalp-fast; do
  if [ -d "$TASKS_DIR/$old_task" ]; then
    echo "[DISABLED] $old_task (v3/v4 legacy — moved to Cowork or consolidated)"
    # Don't delete, just mark as disabled
    echo "# DISABLED — v5 architecture. This task moved to Cowork." > "$TASKS_DIR/$old_task/DISABLED"
  fi
done

echo ""
echo "=== Setup complete ==="
echo ""
echo "Claude Code tasks:"
echo "  trader: */3 * * * * (Opus)"
echo ""
echo "Cowork tasks (manage via Cowork desktop app):"
echo "  quantrabbit-analyst:   */10 * * * *"
echo "  quantrabbit-secretary: */11 * * * *"
echo "  quantrabbit-news:      */15 * * * *"
echo ""
echo "Next steps:"
echo "  1. config/env.toml に OANDA API キーを設定"
echo "  2. Claude Code で trader タスクを有効化 (*/3)"
echo "  3. Cowork で analyst/secretary/news を 'Run now' で初回承認"

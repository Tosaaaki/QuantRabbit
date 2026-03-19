#!/bin/bash
# Trade scheduled tasks setup script (v4 — 3 agent architecture)
# Usage: bash scripts/trader_tools/setup_scheduled_tasks.sh
#
# Registers trader, analyst, secretary tasks.
# Old v3 tasks (scalp-fast, swing-trader, market-radar, macro-intel) are archived in docs/archive/.

set -e
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TASKS_DIR="$HOME/.claude/scheduled-tasks"

echo "=== Trading Scheduled Tasks Setup (v4) ==="
echo "Repo: $REPO_DIR"
echo "Tasks dir: $TASKS_DIR"
echo ""

# --- trader ---
mkdir -p "$TASKS_DIR/trader"
cat > "$TASKS_DIR/trader/SKILL.md" << SKILL_EOF
---
name: trader
description: Discretionary trader — read the market, think, act (2-3min interval)
---

Read docs/TRADER_PROMPT.md and follow its instructions.

You are a professional discretionary trader. Not a score-following bot.
Read the market. Form your own view. If you have conviction, act. If not, wait.

Working directory: $REPO_DIR

Essentials:
- Read logs/live_monitor_summary.json first (compact, raw data, updated every 30s)
- Full data at logs/live_monitor.json if you need deeper analysis
- Manage existing positions first, then look for new entries
- You can scalp (2-5pip) OR swing (10-50pip) — read what the market is offering
- OANDA API direct (urllib). Config: config/env.toml
- API: https://api-fxtrade.oanda.com
- Log trades to logs/live_trade_log.txt with TRADE: prefix
- Register all trades to logs/trade_registry.json
- Max 1500 units per trade. No exceptions.
- No Agent subprocesses (they timeout)
- No global lock needed
SKILL_EOF
echo "[OK] trader"

# --- analyst ---
mkdir -p "$TASKS_DIR/analyst"
cat > "$TASKS_DIR/analyst/SKILL.md" << SKILL_EOF
---
name: analyst
description: Trader's right hand — macro analysis + performance + tools (10min interval)
---

Read docs/ANALYST_PROMPT.md and follow its instructions.

You are the dedicated analyst for Claude the pro trader.
Research macro, analyze cross-pair flows, track performance, provide actionable insights.

Working directory: $REPO_DIR

Important:
- **Task lock (execute BEFORE anything else. Never skip this):**
  \`\`\`bash
  cd $REPO_DIR && python3 scripts/trader_tools/task_lock.py acquire global_agent 5 --pid \$PPID --caller analyst
  \`\`\`
  → If output starts with \`SKIP\` or \`YIELD\` → **exit immediately.**
  → If output is \`ACQUIRED\` → proceed.
  **On task completion (success or error), always execute:**
  \`\`\`bash
  cd $REPO_DIR && python3 scripts/trader_tools/task_lock.py release global_agent --caller analyst
  \`\`\`
- Never place orders (analysis and information only)
- Use WebSearch for news and macro information
- Run \`scripts/trader_tools/trade_performance.py\` for performance stats
- Run \`scripts/trader_tools/refresh_factor_cache.py --all --quiet\` to update H1/H4 data
- Update logs/shared_state.json with macro_bias
- Take ONE action every cycle — update bias, write alert, build tool, or improve prompt
- May edit docs/TRADER_PROMPT.md to improve trader behavior
- May create tools in \`scripts/trader_tools/\`
SKILL_EOF
echo "[OK] analyst"

# --- secretary ---
mkdir -p "$TASKS_DIR/secretary"
cat > "$TASKS_DIR/secretary/SKILL.md" << SKILL_EOF
---
name: secretary
description: Trading secretary — health check + critical alerts (every 11 min)
---

Read docs/SECRETARY_PROMPT.md and follow its instructions.

You are the secretary for Claude the professional trader. Keep things running smoothly.
Monitor trader and analyst agents, detect anomalies, report only what matters.

Working directory is $REPO_DIR.

Important:
- **NO global lock required** — secretary is read-only (no orders)
- Never place orders (monitoring and reporting only)
- Complete quickly (target: under 30 seconds)
- Write results to logs/secretary_report.json
- Only alert on critical issues: margin >90%, 3+ consecutive losses, stale positions, tasks not running
- Clean up old alerts in shared_state.json (>1 hour old)
- **Timestamps: ALWAYS use \`date -u +%Y-%m-%dT%H:%M:%SZ\` via Bash. NEVER write timestamps by hand.**
- **All output MUST be in English.**
SKILL_EOF
echo "[OK] secretary"

# --- Clean up old v3 tasks if they exist ---
for old_task in scalp-fast swing-trader market-radar macro-intel scalp-trader; do
    if [ -d "$TASKS_DIR/$old_task" ]; then
        rm -rf "$TASKS_DIR/$old_task"
        echo "[REMOVED] $old_task (v3 legacy)"
    fi
done

echo ""
echo "=== Setup complete (v4: 3-agent architecture) ==="
echo ""
echo "Active tasks:"
echo "  - trader:    2-3min interval (Opus)"
echo "  - analyst:   10min interval (Sonnet)"
echo "  - secretary: 11min interval (Sonnet)"
echo ""
echo "Next steps:"
echo "  1. config/env.toml に OANDA API キーを設定"
echo "  2. Claude Code で「トレード開始」と言えば起動"
echo "  3. 旧体制に戻す場合: docs/archive/README.md を参照"

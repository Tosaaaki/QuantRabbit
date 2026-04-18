# QuantRabbit — Codex Discretionary FX Trading System

## Live Architecture

- `trader`: recurring discretionary trader task
- `daily-review`: distills lessons into `collab_trade/strategy_memory.md`
- `daily-slack-summary`: posts the morning summary to `#qr-daily`
- `quality-audit`: independent market analyst prompt retained for challenge and chart reading
- `qr-news-digest` / `qr-news-flow-append`: hourly news context pipeline

## Host Differences

- Codex currently runs `trader` every 20 minutes
- Claude compatibility currently runs `trader` every 10 minutes
- Codex `quality-audit` is currently paused
- Claude compatibility `quality-audit` currently runs every 45 minutes
- Shared runtime is lock-based: 10-minute minimum session, 15-minute stale-lock threshold, 17-minute watchdog

## Source Of Truth

- `AGENTS.md`: architecture, rules, document map
- `docs/SKILL_trader.md`: canonical recurring trader playbook
- `docs/SKILL_daily-review.md`: canonical daily-review playbook
- `docs/SKILL_quality-audit.md`: canonical quality-audit playbook
- `docs/codex_automations/*.md`: thin Codex wrappers pointing at canonical prompts
- `collab_trade/CLAUDE.md`: manual collaborative-trading guide only

## Key Runtime Files

- `collab_trade/state.md`: cross-session handoff
- `collab_trade/strategy_memory.md`: distilled long-term lessons
- `collab_trade/summary.md`: day summary and performance context
- `logs/live_trade_log.txt`: execution log
- `logs/news_digest.md`: hourly news summary
- `logs/news_flow_log.md`: 48-hour narrative flow log
- `logs/quality_audit.md`: latest auditor memo

## Useful Commands

```bash
python3 tools/check_task_sync.py
python3 tools/task_runtime.py trader preflight
python3 tools/session_data.py
python3 tools/profit_check.py --all
python3 tools/protection_check.py
```

## Legacy

- Historical bot-layer tools and docs remain in the repo for rollback or forensic reference
- They are not part of the live recurring trader path

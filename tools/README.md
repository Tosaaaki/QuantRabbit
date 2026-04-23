# QuantRabbit Runtime Tools

This directory contains one-shot analysis, execution, validation, and runtime
handoff scripts used by the discretionary Codex trader. The live path is driven
by `docs/SKILL_trader.md` and the scheduled Codex automation; tools are helpers,
not autonomous workers.

## Rules

- Scripts must be runnable as one-shot commands from the repo root.
- Write operational receipts to the appropriate runtime file, especially
  `logs/live_trade_log.txt` for any order action.
- Do not add persistent `while True` loops or background bot processes.
- Prefer shared helpers already in `tools/` and `collab_trade/memory/`.
- Keep prompts and operator guides in English unless the output is an intentional
  Slack/user-facing Japanese message.

## Core Routes

- `task_runtime.py`: lock, cadence, stale-lock, and watchdog coordination.
- `session_data.py`: session-start market, news, memory, and action-board fetch.
- `place_trader_order.py`: exact pretrade guard plus direct OANDA order send.
- `validate_trader_state.py`: SESSION_END handoff and live receipt validation.
- `session_end.py`: validation, hot updates, S-hunt ledger, outcome sync, memory ingest, and runtime git sync.
- `quality_audit.py`: independent chart/read-through audit.
- `daily_review.py`: daily evidence review and memory evolution.

## Checks

```bash
python3 tools/check_task_sync.py
python3 tools/validate_trader_state.py
python3 -m pytest archive/tests/test_validate_trader_state.py
```

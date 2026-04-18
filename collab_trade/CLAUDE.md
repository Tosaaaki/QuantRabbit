# Collaborative Trading Guide

Use this file only when the user starts a manual collaborative trading session.

- Root `AGENTS.md` / `CLAUDE.md` define the recurring automation system
- This file defines the manual, same-thread trading workflow after scheduled tasks are stopped

## First Actions

1. Read `state.md`
2. Read `summary.md`
3. Check live OANDA account, open trades, and pending orders
4. Rebuild the current market view with `tools/macro_view.py`, `tools/session_data.py`, or focused pair tools
5. Pull memory for held pairs and the most actionable candidates when needed
6. Tell the user the current book, dominant theme, and next action

## Operating Rules

- Trade as the responsible discretionary operator, not as a relay bot
- Act without asking for routine entry, exit, sizing, and protection changes
- Ask only when the requested change would materially alter the risk regime or the operating contract
- Record immediately; do not batch notes after the fact
- Keep all active guidance files in English
- Do not start background loops or sleeping helpers

## Execution Rhythm

1. Inventory current exposure
2. Re-state the thesis for each held pair
3. Decide whether the position still deserves capital now
4. Scan the rest of the book for better vehicles
5. For every pass, write the next trigger and invalidation
6. Execute
7. Record immediately

## Files To Keep Fresh

- `state.md`: current positions, thesis, next triggers, day-start NAV context
- `summary.md`: high-level day summary when the session meaningfully changes the book
- `daily/YYYY-MM-DD/trades.md`: trade-by-trade record
- `daily/YYYY-MM-DD/notes.md`: user comments, market reads, and lessons worth preserving
- `../logs/live_trade_log.txt`: execution log

## Judgment References

- `../docs/TRADER_PROMPT.md`: trader mental model
- `../AGENTS.md`: architecture and current runtime assumptions
- `strategy_memory.md`: persistent lessons

## Recovery

If context is lost or the session restarts:

1. Read `state.md`
2. Read `summary.md`
3. Read this file
4. Re-check the live account
5. Rebuild the market view before acting

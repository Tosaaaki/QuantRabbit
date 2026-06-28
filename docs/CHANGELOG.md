# Changelog

## 2026-06-28

- Added `tools/position_sizing.py` and dry-run-only `tools/place_trader_order.py` for target-aware, risk-based order sizing.
- Added the `10% EXTENSION GATE` checklist and RiskEngine target-path metadata guards.
- Added `docs/SKILL_daily-review.md` Daily Target Review with +5%/+10% miss classification.
- Added the required FULL_TRADER `5% PATH BOARD` and `ATTACK STACK` session contract.
- Added read-only trader-state/task-sync validators so B/C churn cannot be documented as a +5% path.
- Added `tools/daily_target.py` as a read-only UTC 00:00 daily target engine.
- Added `tools/session_data.py` to print the session-start daily target block with +5% base target and +10% extension target.
- Added `tools/intraday_pl_update.py` so intraday P/L uses UTC trading-day boundaries while display surfaces may still show JST.
- Documented that +10% is extension-only and does not change live ordering behavior in this change.

# Changelog

## 2026-06-29

- Added MARKET_READ_FIRST receipt discipline: GPT trader receipts must include naked read, 30m/2h prediction, and forced-trade read before blockers; verifier rejects blocker-first, LIVE_READY=0-only, and NEGATIVE_EXPECTANCY-only reasoning; market-read predictions are recorded/scored separately from trade P/L.
- Added USER_ALPHA / OPERATOR_ALPHA continuation handling: daily-review publishes profitable user-led winners separately, GPT trader receipts must answer continuation or exact blocker, and stale pending replacement sends retain ignored pending ids through final gateway validation.
- Added controlled target-path live routing behind `QR_TARGET_PATH_LIVE_ENABLED=1`, with LiveOrderGateway receipts and LIVE-LEARNING daily-review classification.

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

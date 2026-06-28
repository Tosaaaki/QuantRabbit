# Daily Target Report

- Generated at UTC: `2026-06-26T06:38:30.445890+00:00`
- Status: `PURSUE_TARGET`
- Start equity: `173945 JPY`
- Campaign day (JST9): `2026-06-26`
- Target: `17394 JPY` (`10.0%`)
- Minimum daily floor: `8697 JPY` (`5.0%`)
- Realized PnL: `0 JPY`
- Trader unrealized PnL: `0 JPY`
- Progress: `0 JPY` (`0.0%` of target)
- Account unrealized PnL: `0 JPY` (includes manual/tagless exposure)
- Account progress: `0 JPY` (`0.0%` of target, broker NAV view)
- Minimum-floor progress: `0.0%`; remaining floor `8697 JPY`
- Remaining target: `17394 JPY`
- Open risk: `0 JPY`
- Remaining risk budget: `17394 JPY`
- Target trades per day: `30` (`ai_test_bot_target_band_6pct_required_trades_capped_floored_by_min_per_trade_pct`)
- Target trade pace basis: `6.0%`
- Per-trade risk cap: `1739 JPY`
- Current equity estimate: `173945 JPY`

## Blockers

- remaining target 17394 JPY still needs live-ready campaign coverage

## Open Positions

- none

## Target Contract

- The 10% daily target is tracked as a product KPI and execution objective, not a guaranteed return.
- The 5% daily floor is tracked as the minimum same-day progress line; reaching it does not stop the 10% campaign by itself.
- Unprotected trader-owned or external exposure makes remaining risk budget unavailable; operator-managed manual/tagless exposure is TP-managed only and does not block fresh entries.
- Trader progress excludes operator-managed manual/tagless P/L for risk gating, while account progress shows broker NAV including that exposure.
- Reaching the target switches the system toward protection-first behavior before any new risk is added.

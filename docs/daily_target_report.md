# Daily Target Report

- Generated at UTC: `2026-07-02T10:12:33.217801+00:00`
- Status: `PURSUE_TARGET`
- Start equity: `289574 JPY`
- Campaign day (JST9): `2026-07-02`
- Target: `28957 JPY` (`10.0%`)
- Minimum daily floor: `14479 JPY` (`5.0%`)
- Realized PnL: `-2524 JPY`
- Trader unrealized PnL: `0 JPY`
- Progress: `-2524 JPY` (`-8.7%` of target)
- Account unrealized PnL: `0 JPY` (includes manual/tagless exposure)
- Account progress: `-2524 JPY` (`-8.7%` of target, broker NAV view)

## Rolling 30D 4X Policy

- Policy: `ROLLING_30D_4X`
- Window: `2026-06-30T00:22:43.760529+00:00` → `2026-07-30T00:22:43.760529+00:00`
- Rolling 30d start equity: `175305 JPY`
- current_equity_raw: `287050 JPY`
- capital_flows_30d: `100000 JPY`
- funding_adjusted_equity: `187050 JPY`
- rolling_30d_multiplier_raw: `1.6374x`
- rolling_30d_multiplier_funding_adjusted: `1.0670x`
- remaining_to_4x_raw: `414170 JPY`
- remaining_to_4x_funding_adjusted: `514170 JPY`
- Current 30d multiplier: `1.0670x` (funding-adjusted)
- Remaining to 4x: `514170 JPY` (funding-adjusted)
- required_calendar_daily_return_raw: `3.2902%`
- required_active_day_return_raw: `4.5133%`
- required_calendar_daily_return_funding_adjusted: `4.9061%`
- required_active_day_return_funding_adjusted: `6.7492%`
- Required calendar daily return: `4.9061%`
- Required active-day return: `6.7492%`
- performance_basis: `funding_adjusted`
- sizing_basis: `raw_nav`
- Pace state: `BEHIND`

## Daily Pace Marker

- Minimum-floor progress: `-17.4%`; remaining floor `17002 JPY`
- Remaining target: `31481 JPY`
- Open risk: `0 JPY`
- Remaining risk budget: `28957 JPY`
- Target trades per day: `30` (`ai_test_bot_target_band_6pct_required_trades_capped_floored_by_min_per_trade_pct`)
- Target trade pace basis: `6.0%`
- Per-trade risk cap: `2896 JPY`
- Current equity estimate: `287050 JPY`

## Blockers

- remaining target 31481 JPY still needs live-ready campaign coverage

## Open Positions

- none

## Target Contract

- The top KPI is rolling 30-calendar-day 4x equity growth.
- Rolling 30d performance uses funding_adjusted_equity; current_equity_raw remains the broker NAV basis for risk, margin, and sizing.
- Capital flows are recorded as deposits/withdrawals, not trading P/L, and are excluded from funding-adjusted return.
- Backward-compatible required_calendar_daily_return and required_active_day_return are funding-adjusted primary values.
- The +5% daily line is a pace marker, review trigger, and protection milestone; it must not force B/C churn on no-edge days.
- The 10% daily target is extension-only behind the favorable-market gate, not a guaranteed return.
- Unprotected trader-owned or external exposure makes remaining risk budget unavailable; operator-managed manual/tagless exposure is TP-managed only and does not block fresh entries.
- Trader progress excludes operator-managed manual/tagless P/L for risk gating, while account progress shows broker NAV including that exposure.
- Reaching the target switches the system toward protection-first behavior before any new risk is added.

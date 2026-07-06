# Daily Target Report

- Generated at UTC: `2026-07-06T01:57:27.837390+00:00`
- Status: `PURSUE_TARGET`
- Start equity: `287186 JPY`
- Campaign day (JST9): `2026-07-06`
- Target: `28719 JPY` (`10.0%`)
- Minimum daily floor: `14359 JPY` (`5.0%`)
- Realized PnL: `0 JPY`
- Trader unrealized PnL: `0 JPY`
- Progress: `0 JPY` (`0.0%` of target)
- Account unrealized PnL: `-12644 JPY` (includes manual/tagless exposure)
- Account progress: `-12644 JPY` (`-44.0%` of target, broker NAV view)

## Rolling 30D 4X Policy

- Policy: `ROLLING_30D_4X`
- Window: `2026-07-02T03:57:37.461200+00:00` → `2026-08-01T03:57:37.461200+00:00`
- Rolling 30d start equity: `171435 JPY`
- current_equity_raw: `274542 JPY`
- capital_flows_30d: `100000 JPY`
- funding_adjusted_equity: `174542 JPY`
- rolling_30d_multiplier_raw: `1.6014x`
- rolling_30d_multiplier_funding_adjusted: `1.0181x`
- remaining_to_4x_raw: `411198 JPY`
- remaining_to_4x_funding_adjusted: `511198 JPY`
- Current 30d multiplier: `1.0181x` (funding-adjusted)
- Remaining to 4x: `511198 JPY` (funding-adjusted)
- required_calendar_daily_return_raw: `3.5718%`
- required_active_day_return_raw: `4.9020%`
- required_calendar_daily_return_funding_adjusted: `5.3860%`
- required_active_day_return_funding_adjusted: `7.4157%`
- Required calendar daily return: `5.3860%`
- Required active-day return: `7.4157%`
- performance_basis: `funding_adjusted`
- sizing_basis: `raw_nav`
- Pace state: `BEHIND`

## Daily Pace Marker

- Minimum-floor progress: `0.0%`; remaining floor `14359 JPY`
- Remaining target: `28719 JPY`
- Open risk: `0 JPY`
- Remaining risk budget: `28719 JPY`
- Target trades per day: `30` (`ai_test_bot_target_band_6pct_required_trades_capped_floored_by_min_per_trade_pct`)
- Target trade pace basis: `6.0%`
- Per-trade risk cap: `2872 JPY`
- Current equity estimate: `274542 JPY`

## Blockers

- none

## Open Positions

- `472987` `EUR_USD SHORT` owner=`operator_manual` units=`30000` upl=`-12644` risk=`unknown` missing=`SL`

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

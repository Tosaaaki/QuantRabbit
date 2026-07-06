# AUD_JPY SHORT BREAKOUT_FAILURE LIMIT Proof Pack

- Generated: `2026-07-06T01:59:31Z`
- Lane: `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT`
- Classification: `PORTFOLIO_COMPONENT_REPAIR_REQUIRED`
- Standalone 4x: `False`
- Portfolio component possible after repair: `True`
- Can create live permission: `False`

## Economics

- Expected JPY/trade: `189.9205`
- Estimated trades/day: `34.8974`
- Expected active-day contribution: `6627.7317` JPY
- Expected daily return on funding-adjusted equity: `3.7972`%
- Required calendar daily return: `5.386027`%

## Geometry / Margin

- Entry / TP / SL: `112.169` / `111.832` / `112.294`
- Reward/risk: `2.6960000000000264`; reward/loss pips `33.70000000000033` / `12.5`
- Units / risk / margin: `3000` / `375.0` / `13460.28`

## Failed Checks

- `s5_bidask_spread_included_replay`
- `sample_count_floor`
- `daily_stability_floor`
- `forecast_executability`
- `risk_engine_pass`
- `live_order_gateway_pass`
- `gpt_verifier_pass`
- `guardian_operator_review_clear`

## Current Blockers

- `RANGE_FORECAST_REQUIRES_RANGE_ROTATION`
- `SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH`
- `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`

# AUD_JPY SHORT BREAKOUT_FAILURE LIMIT Proof Pack

- Generated: `2026-07-06T09:42:39Z`
- Lane: `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT`
- Classification: `REPAIR_REQUIRED`
- Standalone 4x: `False`
- Portfolio component possible after repair: `True`
- Can create live permission: `False`

## Economics

- Expected JPY/trade: `189.9205`
- Estimated trades/day: `34.8974`
- Expected active-day contribution: `6627.7317` JPY
- Expected daily return on funding-adjusted equity: `3.7891`%
- Required calendar daily return: `5.412477`%

## Geometry / Margin

- Entry / TP / SL: `112.502` / `112.165` / `112.627`
- Reward/risk: `2.6959999999999127`; reward/loss pips `33.69999999999891` / `12.5`
- Units / risk / margin: `3000` / `375.0` / `13500.24`

## Failed Checks

- `s5_bidask_spread_included_replay`
- `sample_count_floor`
- `daily_stability_floor`
- `forecast_executability`
- `risk_engine_pass`
- `live_order_gateway_pass`
- `gpt_verifier_pass`
- `guardian_operator_review_clear`

## Required Proof Matrix

| proof | status |
|---|---|
| `S5 samples` | `MISSING` |
| `active days` | `MISSING` |
| `forecast executable proof` | `MISSING` |
| `geometry proof` | `PRESENT_BUT_NOT_PERMISSION` |
| `attached TP proof` | `PRESENT_BUT_NOT_PERMISSION` |
| `RiskEngine` | `MISSING` |
| `Gateway` | `MISSING` |
| `GPT verifier` | `MISSING` |
| `guardian/operator review` | `MISSING` |

## Current Blockers

- `RANGE_FORECAST_REQUIRES_RANGE_ROTATION`
- `SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH`
- `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`

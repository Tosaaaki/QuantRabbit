# A/S Proof-Pack Queue

- Generated: `2026-07-06T15:11:03Z`
- Queue count: `2`
- PROOF_READY: `0`
- Can create live permission: `0`

| lane | class | daily % | distance | can enter proof pack | blockers |
|---|---|---:|---:|---|---|
| `trend_trader:USD_CHF:LONG:TREND_CONTINUATION` | `HISTORICAL_ONLY` | 34.3376 | 7 | `False` | RANGE_FORECAST_REQUIRES_RANGE_ROTATION, REWARD_RISK_TOO_LOW, LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT, MATRIX_REPAIR_REJECT_CONTEXT, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, EXHAUSTION_RANGE_CHASE, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY, STRATEGY_NOT_ELIGIBLE |
| `range_trader:GBP_USD:LONG:RANGE_ROTATION` | `HISTORICAL_ONLY` | 7.9116 | 7 | `False` | REWARD_RISK_TOO_LOW, RANGE_COUNTERTREND_RR_TOO_LOW, LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT, MATRIX_REPAIR_REJECT_CONTEXT, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, RANGE_ROTATION_BROADER_LOCATION_CHASE, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, FORECAST_WATCH_ONLY, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY |

## Missing Proof

- `trend_trader:USD_CHF:LONG:TREND_CONTINUATION`: fresh_744h_replay, s5_bidask_spread_included_replay, geometry_proof, risk_engine_pass, live_order_gateway_pass, gpt_verifier_pass, no_guardian_operator_review_blocker
- `range_trader:GBP_USD:LONG:RANGE_ROTATION`: fresh_744h_replay, s5_bidask_spread_included_replay, geometry_proof, risk_engine_pass, live_order_gateway_pass, gpt_verifier_pass, no_guardian_operator_review_blocker

# A/S Proof-Pack Queue

- Generated: `2026-07-06T08:51:35Z`
- Queue count: `4`
- PROOF_READY: `0`
- Can create live permission: `0`

| lane | class | daily % | distance | can enter proof pack | blockers |
|---|---|---:|---:|---|---|
| `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT` | `REPAIR_REQUIRED` | 3.7891 | 5 | `True` | RANGE_FORECAST_REQUIRES_RANGE_ROTATION, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED |
| `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE` | `REPAIR_REQUIRED` | 3.7891 | 6 | `True` | RANGE_FORECAST_REQUIRES_RANGE_ROTATION, PATTERN_REVERSAL_CHASE, EXHAUSTION_RANGE_CHASE, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED |
| `range_trader:GBP_USD:LONG:RANGE_ROTATION` | `HISTORICAL_ONLY` | 8.1576 | 7 | `False` | LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, RANGE_ROTATION_BROADER_LOCATION_CHASE, EXHAUSTION_RANGE_CHASE, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, STRATEGY_NOT_ELIGIBLE |
| `trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION` | `HISTORICAL_ONLY` | 7.741 | 7 | `False` | RANGE_FORECAST_REQUIRES_RANGE_ROTATION, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, PATTERN_REVERSAL_CHASE, EXHAUSTION_RANGE_CHASE, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, STRATEGY_NOT_ELIGIBLE |

## Missing Proof

- `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT`: s5_bidask_spread_included_replay, risk_engine_pass, live_order_gateway_pass, gpt_verifier_pass, no_guardian_operator_review_blocker
- `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE`: s5_bidask_spread_included_replay, geometry_proof, risk_engine_pass, live_order_gateway_pass, gpt_verifier_pass, no_guardian_operator_review_blocker
- `range_trader:GBP_USD:LONG:RANGE_ROTATION`: fresh_744h_replay, s5_bidask_spread_included_replay, geometry_proof, risk_engine_pass, live_order_gateway_pass, gpt_verifier_pass, no_guardian_operator_review_blocker
- `trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION`: fresh_744h_replay, s5_bidask_spread_included_replay, geometry_proof, risk_engine_pass, live_order_gateway_pass, gpt_verifier_pass, no_guardian_operator_review_blocker

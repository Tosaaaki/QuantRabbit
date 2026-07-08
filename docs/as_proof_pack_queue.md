# A/S Proof-Pack Queue

- Generated: `2026-07-08T07:07:00Z`
- Queue count: `2`
- Rejected candidates: `4`
- PROOF_READY: `0`
- Can create live permission: `0`

| lane | class | daily % | distance | can enter proof pack | blockers |
|---|---|---:|---:|---|---|
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` | `EVIDENCE_GAP` | None | 7 | `True` | RANGE_FORECAST_REQUIRES_RANGE_ROTATION, OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT` | `EVIDENCE_GAP` | None | 7 | `True` | RANGE_FORECAST_REQUIRES_RANGE_ROTATION, OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED |

## Missing Proof

- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`: s5_bidask_spread_included_replay, sample_count_floor, active_day_floor, risk_engine_pass, live_order_gateway_pass, gpt_verifier_pass, no_guardian_operator_review_blocker
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT`: s5_bidask_spread_included_replay, sample_count_floor, active_day_floor, risk_engine_pass, live_order_gateway_pass, gpt_verifier_pass, no_guardian_operator_review_blocker

## Rejected Candidates

- `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE`: spread_included_bidask_replay_negative_for_exact_lane, packaged_bidask_rule_live_block_negative_expectancy; blockers=RANGE_FORECAST_REQUIRES_RANGE_ROTATION, PATTERN_REVERSAL_CHASE, EXHAUSTION_RANGE_CHASE, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED
- `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT`: spread_included_bidask_replay_negative_for_exact_lane, packaged_bidask_rule_live_block_negative_expectancy; blockers=RANGE_FORECAST_REQUIRES_RANGE_ROTATION, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, S5_DAILY_SAMPLE_CONCENTRATED, S5_POSITIVE_DAY_RATE_LOW
- `range_trader:GBP_USD:LONG:RANGE_ROTATION`: spread_included_bidask_replay_negative_for_exact_lane, packaged_bidask_rule_live_block_negative_expectancy; blockers=STALE_QUOTE, LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, RANGE_ROTATION_BROADER_LOCATION_CHASE, EXHAUSTION_RANGE_CHASE, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, STRATEGY_NOT_ELIGIBLE
- `trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION`: spread_included_bidask_replay_negative_for_exact_lane, packaged_bidask_rule_live_block_negative_expectancy; blockers=RANGE_FORECAST_REQUIRES_RANGE_ROTATION, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, PATTERN_REVERSAL_CHASE, EXHAUSTION_RANGE_CHASE, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, STRATEGY_NOT_ELIGIBLE

# Portfolio 4x Path Planner

- Generated: `2026-07-06T15:11:03Z`
- Status: `NO_LIVE_READY_PORTFOLIO`
- Can reach 4x now: `False`
- Non-hard-excluded candidates: `68`
- Standalone math candidates: `2`
- Fastest mathematical basket reaches required return: `True`; live eligible `False`

## Top Ranked Repair Work

| rank | lane | class | daily % | score | distance | units | blockers |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `trend_trader:USD_CHF:LONG:TREND_CONTINUATION` | `HISTORICAL_ONLY` | 34.3376 | 251.7008 | 7 | 0 | 10 |
| 2 | `range_trader:GBP_USD:LONG:RANGE_ROTATION` | `HISTORICAL_ONLY` | 7.9116 | 40.2928 | 7 | 0 | 10 |
| 3 | `failure_trader:GBP_JPY:LONG:BREAKOUT_FAILURE:LIMIT` | `HISTORICAL_ONLY` | 5.9441 | 31.5528 | 6 | 0 | 8 |
| 4 | `failure_trader:GBP_JPY:LONG:BREAKOUT_FAILURE` | `HISTORICAL_ONLY` | 5.9441 | 30.0528 | 6 | 0 | 9 |
| 5 | `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE:LIMIT` | `HISTORICAL_ONLY` | 3.05 | 21.9 | 6 | 1000 | 7 |
| 6 | `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` | `HISTORICAL_ONLY` | 3.05 | 18.9 | 6 | 1000 | 9 |
| 7 | `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` | `HISTORICAL_ONLY` | 3.7359 | 13.8872 | 6 | 0 | 8 |
| 8 | `trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET` | `HISTORICAL_ONLY` | 3.7359 | 12.3872 | 6 | 0 | 9 |
| 9 | `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT` | `EVIDENCE_GAP` | None | 10.0 | 7 | 1000 | 4 |
| 10 | `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` | `EVIDENCE_GAP` | None | 5.5 | 7 | 1000 | 7 |
| 11 | `range_trader:AUD_JPY:LONG:RANGE_ROTATION:MARKET` | `HISTORICAL_ONLY` | 1.0566 | 4.4528 | 6 | 1000 | 8 |
| 12 | `range_trader:AUD_JPY:SHORT:RANGE_ROTATION` | `HISTORICAL_ONLY` | None | -1.0 | 6 | 1000 | 6 |
| 13 | `range_trader:AUD_JPY:LONG:RANGE_ROTATION` | `HISTORICAL_ONLY` | 1.0566 | -2.0472 | 6 | 7000 | 7 |
| 14 | `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` | `HISTORICAL_ONLY` | None | -4.0 | 6 | 1000 | 8 |
| 15 | `range_trader:GBP_JPY:SHORT:RANGE_ROTATION` | `HISTORICAL_ONLY` | None | -6.0 | 6 | 4000 | 4 |

## Global Blockers

- `{'profitability_acceptance_status': 'PROFITABILITY_ACCEPTANCE_BLOCKED', 'profitability_acceptance_blockers': ['SELF_IMPROVEMENT_P0_PRESENT: self-improvement audit still has 2 P0 finding(s)', 'NEGATIVE_EXPECTANCY_ACTIVE: capture economics is still NEGATIVE_EXPECTANCY', 'MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE: 1 TP-proven segment(s) are still net-damaged by MARKET_ORDER_TRADE_CLOSE leakage', 'MARKET_CLOSE_LEAK_FAMILY_BLOCKED: EUR_USD LONG BREAKOUT_FAILURE system-gateway MARKET_ORDER_TRADE_CLOSE loss family remains blocked from fresh-entry and repair-exit live routing until the exact exception proof stack exists.', 'MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE: 30-day OANDA candle replay says the current TP-progress repair improves loss-side closes, but the replayed loss-close P/L is still net negative'], 'support_status': 'SUPPORT_BLOCKED', 'support_blockers': [{'code': 'GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING', 'message': 'guardian receipt consumption status does not allow normal new-entry routing; status=GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED classifications=NEEDS_OPERATOR_REVIEW,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,STALE_ACKNOWLEDGED,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,STALE_ACKNOWLEDGED,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY', 'severity': 'P0'}, {'code': 'GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER', 'message': 'receipt_lifecycle=EXPIRED while consumed_by_trader=false; sources=2', 'severity': 'WARN'}, {'code': 'GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW', 'message': 'Receipt event 832d2908eeb84b2f REDUCE requires operator review before normal new-entry routing; operator_review_status=OPERATOR_REVIEW_STALE, reason=operator review row is expired.', 'severity': 'P0'}, {'code': 'SELF_IMPROVEMENT_P0_PRESENT', 'message': 'self-improvement audit has 2 P0 finding(s)', 'severity': 'P0'}, {'code': 'NO_LIVE_READY_LANES', 'message': 'daily target is open but no lane is LIVE_READY', 'severity': 'P1'}, {'code': 'PROFITABILITY_ACCEPTANCE_BLOCKED', 'message': 'profitability acceptance status is PROFITABILITY_ACCEPTANCE_BLOCKED', 'severity': 'P0'}], 'memory_status': 'MEMORY_HEALTH_BLOCKED'}`

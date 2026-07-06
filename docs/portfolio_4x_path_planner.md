# Portfolio 4x Path Planner

- Generated: `2026-07-06T01:59:31Z`
- Status: `NO_LIVE_READY_PORTFOLIO`
- Can reach 4x now: `False`
- Non-hard-excluded candidates: `53`
- Standalone math candidates: `2`
- Fastest mathematical basket reaches required return: `True`; live eligible `False`

## Top Ranked Repair Work

| rank | lane | class | daily % | score | distance | units | blockers |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT` | `REPAIR_REQUIRED` | 3.7972 | 63.3776 | 5 | 3000 | 4 |
| 2 | `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE` | `REPAIR_REQUIRED` | 3.7972 | 56.3776 | 6 | 3000 | 6 |
| 3 | `trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION` | `HISTORICAL_ONLY` | 7.7576 | 46.0608 | 7 | 3000 | 8 |
| 4 | `range_trader:GBP_USD:LONG:RANGE_ROTATION` | `HISTORICAL_ONLY` | 8.175 | 45.4 | 7 | 0 | 8 |
| 5 | `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` | `HISTORICAL_ONLY` | 3.8603 | 10.3824 | 6 | 0 | 11 |
| 6 | `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` | `EVIDENCE_GAP` | None | 10.0 | 7 | 4000 | 4 |
| 7 | `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT` | `EVIDENCE_GAP` | None | 10.0 | 7 | 4000 | 4 |
| 8 | `range_trader:NZD_CHF:SHORT:RANGE_ROTATION` | `HISTORICAL_ONLY` | None | 0.5 | 6 | 1000 | 5 |
| 9 | `range_trader:AUD_CHF:LONG:RANGE_ROTATION` | `HISTORICAL_ONLY` | None | -1.0 | 6 | 1000 | 6 |
| 10 | `range_trader:AUD_CHF:LONG:RANGE_ROTATION:MARKET` | `HISTORICAL_ONLY` | None | -2.5 | 6 | 1000 | 7 |
| 11 | `range_trader:NZD_CHF:SHORT:RANGE_ROTATION:MARKET` | `HISTORICAL_ONLY` | None | -4.0 | 6 | 1000 | 8 |
| 12 | `range_trader:GBP_JPY:SHORT:RANGE_ROTATION` | `HISTORICAL_ONLY` | None | -7.5 | 6 | 1000 | 5 |
| 13 | `range_trader:USD_JPY:LONG:RANGE_ROTATION` | `HISTORICAL_ONLY` | None | -7.5 | 6 | 4000 | 5 |
| 14 | `range_trader:AUD_JPY:LONG:RANGE_ROTATION` | `HISTORICAL_ONLY` | 1.0918 | -8.7656 | 6 | 0 | 9 |
| 15 | `range_trader:AUD_JPY:LONG:RANGE_ROTATION:MARKET` | `HISTORICAL_ONLY` | 1.0918 | -8.7656 | 6 | 0 | 9 |

## Global Blockers

- `{'profitability_acceptance_status': 'PROFITABILITY_ACCEPTANCE_BLOCKED', 'profitability_acceptance_blockers': ['SELF_IMPROVEMENT_P0_PRESENT: self-improvement audit still has 1 P0 finding(s)', 'NEGATIVE_EXPECTANCY_ACTIVE: capture economics is still NEGATIVE_EXPECTANCY', 'MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE: 1 TP-proven segment(s) are still net-damaged by MARKET_ORDER_TRADE_CLOSE leakage', 'MARKET_CLOSE_LEAK_FAMILY_BLOCKED: EUR_USD LONG BREAKOUT_FAILURE system-gateway MARKET_ORDER_TRADE_CLOSE loss family remains blocked from fresh-entry and repair-exit live routing until the exact exception proof stack exists.', 'MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE: 30-day OANDA candle replay says the current TP-progress repair improves loss-side closes, but the replayed loss-close P/L is still net negative'], 'support_status': 'SUPPORT_BLOCKED', 'support_blockers': [{'code': 'GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING', 'message': 'guardian receipt consumption status does not allow normal new-entry routing; status=GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED classifications=NEEDS_OPERATOR_REVIEW,NEEDS_OPERATOR_REVIEW', 'severity': 'P0'}, {'code': 'GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING', 'message': 'guardian receipt operator review does not allow normal new-entry routing; status=GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED_CURRENT_P0_BLOCKS_ROUTING decisions=OPERATOR_ACKNOWLEDGED_HISTORICAL', 'severity': 'P0'}, {'code': 'SELF_IMPROVEMENT_P0_PRESENT', 'message': 'self-improvement audit has 1 P0 finding(s)', 'severity': 'P0'}, {'code': 'NO_LIVE_READY_LANES', 'message': 'daily target is open but no lane is LIVE_READY', 'severity': 'P1'}, {'code': 'PROFITABILITY_ACCEPTANCE_BLOCKED', 'message': 'profitability acceptance status is PROFITABILITY_ACCEPTANCE_BLOCKED', 'severity': 'P0'}], 'memory_status': 'MEMORY_HEALTH_PASS'}`

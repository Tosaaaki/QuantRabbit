# Portfolio 4x Path Planner

- Generated: `2026-07-08T07:07:05Z`
- Status: `NO_LIVE_READY_PORTFOLIO`
- Can reach 4x now: `False`
- Non-hard-excluded candidates: `51`
- Standalone math candidates: `0`
- Fastest mathematical basket reaches required return: `False`; live eligible `False`

## Top Ranked Repair Work

| rank | lane | class | daily % | score | distance | units | blockers |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` | `EVIDENCE_GAP` | None | 10.0 | 7 | 3000 | 4 |
| 2 | `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT` | `EVIDENCE_GAP` | None | 10.0 | 7 | 3000 | 4 |
| 3 | `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:MARKET` | `HISTORICAL_ONLY` | None | -15.5 | 8 | 3000 | 5 |
| 4 | `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` | `HISTORICAL_ONLY` | None | -17.0 | 8 | 3000 | 6 |
| 5 | `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET` | `HISTORICAL_ONLY` | None | -18.5 | 8 | 3000 | 7 |
| 6 | `trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION` | `REJECTED` | 7.9407 | -68.4744 | 6 | 3000 | 8 |
| 7 | `range_trader:GBP_USD:LONG:RANGE_ROTATION` | `REJECTED` | 8.368 | -76.056 | 7 | 0 | 10 |
| 8 | `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT` | `REJECTED` | 3.8869 | -94.9048 | 6 | 3000 | 4 |
| 9 | `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE` | `REJECTED` | 3.8869 | -97.9048 | 6 | 3000 | 6 |
| 10 | `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` | `REJECTED` | 3.9514 | -108.8888 | 6 | 0 | 11 |
| 11 | `range_trader:AUD_CHF:SHORT:RANGE_ROTATION` | `REJECTED` | None | -126.5 | 7 | 1000 | 7 |
| 12 | `range_trader:AUD_JPY:LONG:RANGE_ROTATION` | `REJECTED` | 1.1176 | -127.0592 | 6 | 0 | 8 |
| 13 | `range_trader:AUD_JPY:LONG:RANGE_ROTATION:MARKET` | `REJECTED` | 1.1176 | -127.0592 | 6 | 0 | 8 |
| 14 | `range_trader:AUD_JPY:SHORT:RANGE_ROTATION` | `REJECTED` | None | -127.5 | 6 | 3000 | 5 |
| 15 | `range_trader:GBP_JPY:SHORT:RANGE_ROTATION` | `REJECTED` | None | -127.5 | 6 | 1000 | 5 |

## Global Blockers

- `{'profitability_acceptance_status': 'PROFITABILITY_ACCEPTANCE_BLOCKED', 'profitability_acceptance_blockers': ['SELF_IMPROVEMENT_P0_PRESENT: self-improvement audit still has 1 P0 finding(s)', 'NEGATIVE_EXPECTANCY_ACTIVE: capture economics is still NEGATIVE_EXPECTANCY', 'MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE: 1 TP-proven segment(s) are still net-damaged by MARKET_ORDER_TRADE_CLOSE leakage', 'MARKET_CLOSE_LEAK_FAMILY_BLOCKED: EUR_USD LONG BREAKOUT_FAILURE system-gateway MARKET_ORDER_TRADE_CLOSE loss family remains blocked from fresh-entry and repair-exit live routing until the exact exception proof stack exists.', 'MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE: 30-day OANDA candle replay says the current TP-progress repair improves loss-side closes, but the replayed loss-close P/L is still net negative'], 'support_status': 'SUPPORT_BLOCKED', 'support_blockers': [{'code': 'GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING', 'message': 'guardian receipt consumption status does not allow normal new-entry routing; status=GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED classifications=NEEDS_OPERATOR_REVIEW,NEEDS_OPERATOR_REVIEW', 'severity': 'P0'}, {'code': 'GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING', 'message': 'guardian receipt operator review does not allow normal new-entry routing; status=GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED_CURRENT_P0_BLOCKS_ROUTING decisions=OPERATOR_ACKNOWLEDGED_HISTORICAL', 'severity': 'P0'}, {'code': 'SELF_IMPROVEMENT_P0_PRESENT', 'message': 'self-improvement audit has 1 P0 finding(s)', 'severity': 'P0'}, {'code': 'NO_LIVE_READY_LANES', 'message': 'daily target is open but no lane is LIVE_READY', 'severity': 'P1'}, {'code': 'PROFITABILITY_ACCEPTANCE_BLOCKED', 'message': 'profitability acceptance status is PROFITABILITY_ACCEPTANCE_BLOCKED', 'severity': 'P0'}], 'memory_status': 'MEMORY_HEALTH_PASS'}`

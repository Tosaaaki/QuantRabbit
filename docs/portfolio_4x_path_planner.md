# Portfolio 4x Path Planner

- Generated: `2026-07-05T18:15:59Z`
- Status: `NO_LIVE_READY_PORTFOLIO`
- Can reach 4x now: `False`
- Non-hard-excluded candidates: `44`
- Standalone math candidates: `2`
- Fastest mathematical basket reaches required return: `True`; live eligible `False`

## Top Ranked Repair Work

| rank | lane | class | daily % | score | distance | units | blockers |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT` | `REPAIR_REQUIRED` | 3.8818 | 55.5544 | 8 | 1000 | 7 |
| 2 | `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE` | `REPAIR_REQUIRED` | 3.8818 | 51.0544 | 8 | 1000 | 10 |
| 3 | `range_trader:GBP_USD:LONG:RANGE_ROTATION` | `HISTORICAL_ONLY` | 8.357 | 39.856 | 8 | 0 | 10 |
| 4 | `trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION` | `HISTORICAL_ONLY` | 7.9303 | 34.9424 | 8 | 0 | 11 |
| 5 | `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT` | `REPAIR_REQUIRED` | None | 14.0 | 9 | 1000 | 6 |
| 6 | `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` | `REPAIR_REQUIRED` | None | 12.5 | 9 | 1000 | 7 |
| 7 | `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` | `HISTORICAL_ONLY` | 3.9462 | 1.5696 | 8 | 0 | 12 |
| 8 | `range_trader:AUD_JPY:LONG:RANGE_ROTATION` | `HISTORICAL_ONLY` | 1.1161 | -12.5712 | 8 | 2000 | 9 |
| 9 | `range_trader:AUD_JPY:LONG:RANGE_ROTATION:MARKET` | `HISTORICAL_ONLY` | 1.1161 | -18.0712 | 8 | 0 | 10 |
| 10 | `range_trader:GBP_JPY:SHORT:RANGE_ROTATION` | `HISTORICAL_ONLY` | None | -24.0 | 8 | 0 | 8 |
| 11 | `range_trader:USD_JPY:LONG:RANGE_ROTATION` | `HISTORICAL_ONLY` | None | -24.0 | 8 | 0 | 8 |
| 12 | `range_trader:USD_CAD:SHORT:RANGE_ROTATION` | `HISTORICAL_ONLY` | None | -25.5 | 8 | 0 | 9 |
| 13 | `trend_trader:GBP_JPY:LONG:TREND_CONTINUATION` | `HISTORICAL_ONLY` | None | -25.5 | 8 | 0 | 9 |
| 14 | `trend_trader:USD_JPY:LONG:TREND_CONTINUATION` | `HISTORICAL_ONLY` | None | -25.5 | 8 | 0 | 9 |
| 15 | `range_trader:AUD_JPY:SHORT:RANGE_ROTATION` | `HISTORICAL_ONLY` | None | -27.0 | 8 | 0 | 10 |

## Global Blockers

- `{'profitability_acceptance_status': 'PROFITABILITY_ACCEPTANCE_BLOCKED', 'profitability_acceptance_blockers': ['SELF_IMPROVEMENT_P0_PRESENT: self-improvement audit still has 1 P0 finding(s)', 'NEGATIVE_EXPECTANCY_ACTIVE: capture economics is still NEGATIVE_EXPECTANCY', 'MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE: 1 TP-proven segment(s) are still net-damaged by MARKET_ORDER_TRADE_CLOSE leakage', 'MARKET_CLOSE_LEAK_FAMILY_BLOCKED: EUR_USD LONG BREAKOUT_FAILURE system-gateway MARKET_ORDER_TRADE_CLOSE loss family remains blocked from fresh-entry and repair-exit live routing until the exact exception proof stack exists.'], 'support_status': 'SUPPORT_BLOCKED', 'support_blockers': [{'code': 'GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING', 'message': 'guardian receipt consumption status does not allow normal new-entry routing; status=GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED classifications=NEEDS_OPERATOR_REVIEW,NEEDS_OPERATOR_REVIEW', 'severity': 'P0'}, {'code': 'GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING', 'message': 'guardian receipt operator review does not allow normal new-entry routing; status=GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED_CURRENT_P0_BLOCKS_ROUTING decisions=OPERATOR_ACKNOWLEDGED_HISTORICAL', 'severity': 'P0'}, {'code': 'SELF_IMPROVEMENT_P0_PRESENT', 'message': 'self-improvement audit has 1 P0 finding(s)', 'severity': 'P0'}, {'code': 'NO_LIVE_READY_LANES', 'message': 'daily target is open but no lane is LIVE_READY', 'severity': 'P1'}, {'code': 'PROFITABILITY_ACCEPTANCE_BLOCKED', 'message': 'profitability acceptance status is PROFITABILITY_ACCEPTANCE_BLOCKED', 'severity': 'P0'}], 'memory_status': 'MEMORY_HEALTH_BLOCKED'}`

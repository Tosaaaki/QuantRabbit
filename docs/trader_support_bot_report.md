# Trader Support Bot Report

- Generated at UTC: `2026-07-05T18:09:33.484126+00:00`
- Status: `SUPPORT_BLOCKED`
- Read only: `True`
- Live side effects: `0`

## Support Gates

| Gate | Value |
|---|---|
| Fresh entry send allowed | `False` |
| Repair basket send allowed | `False` |
| Guardian active | `True` source=`launchd+heartbeat` |
| Guardian heartbeat fresh | `True` age=`8.985`s |
| qr-trader scheduled-run watchdog | `UNAVAILABLE` severity=`INFO` minutes_since=`None` |
| Guardian receipt consumption | `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` normal_routing_allowed=`False` unresolved=`2` |
| LIVE_READY lanes | `0` / `73` |
| Near-ready diagnostic lanes | `73` |
| Order intents freshness | `ORDER_INTENTS_ALIGNED_WITH_BROKER_SNAPSHOT` staleness=`-2.831`s |
| Profitability acceptance freshness | `PROFITABILITY_ACCEPTANCE_ALIGNED_WITH_INPUTS` stale_inputs=`[]` |
| Repair LIVE_READY lanes | `0` |
| Repair lanes after guardian recovery | `0` |
| Repair frontier lanes | `3` |
| RANGE forecast superseded repair lanes | `4` |
| OANDA audit-only local TP proof required | `0` |
| RANGE forecast missing counterpart lanes | `0` |
| Repair frontier clear after support | `0` |
| Repair frontier blocked after support | `3` |
| Global unlock frontier lanes | `0` |
| Profit-capture misses | `14` gap=`5485.238` JPY counterfactual_delta=`18729.334` JPY repair_replay_triggered=`13` repair_delta=`18771.737` JPY |
| Current profit-capture positions | bankable=`0` watch=`0` blocked=`0` |
| Open positions | `1` unknown_owner=`0` operator_manual=`1` |
| Operator manual JPY add guard | `False` code=`None` |
| Open trader positions | `0` upl=`0.0` JPY |
| Directional inversion 5% counterfactuals | `1` |
| Target remaining | `28718.6` JPY |
| Rolling 30d equity | raw=`270740.498` funding_adjusted=`170740.498` capital_flows_30d=`100000.0` JPY |
| Rolling 30d multiplier | raw=`1.57926` funding_adjusted=`0.995949` |
| Remaining to 4x | raw=`414999.723` funding_adjusted=`514999.723` JPY |
| Required return | calendar_funding_adjusted=`5.405806` active_funding_adjusted=`7.443193` |
| Target basis | performance=`funding_adjusted` sizing=`raw_nav` |
| Firepower 5% audit estimate | `True` best=`evidence_queue` |
| Firepower 5% operational reachable | `False` blockers=`['GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING', 'GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING', 'SELF_IMPROVEMENT_P0_PRESENT', 'NO_LIVE_READY_LANES', 'PROFITABILITY_ACCEPTANCE_BLOCKED', 'FRESH_ENTRY_SEND_NOT_ALLOWED']` |

## Blockers

- `P0` `GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING`: guardian receipt consumption status does not allow normal new-entry routing; status=GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED classifications=NEEDS_OPERATOR_REVIEW,NEEDS_OPERATOR_REVIEW
- `P0` `GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING`: guardian receipt operator review does not allow normal new-entry routing; status=GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED_CURRENT_P0_BLOCKS_ROUTING decisions=OPERATOR_ACKNOWLEDGED_HISTORICAL
- `P0` `SELF_IMPROVEMENT_P0_PRESENT`: self-improvement audit has 1 P0 finding(s)
- `P1` `NO_LIVE_READY_LANES`: daily target is open but no lane is LIVE_READY
- `P0` `PROFITABILITY_ACCEPTANCE_BLOCKED`: profitability acceptance status is PROFITABILITY_ACCEPTANCE_BLOCKED

## Near-Ready Lanes

- Shortest path: `range_trader:GBP_USD:SHORT:RANGE_ROTATION` status=`BLOCKED_NEAR_READY_LANE` next=`fresh broker snapshot, forecasts, projections, and order_intents from the same evidence packet`
- Shortest path remains non-executable: live_permission=`False` ordinary_fresh_entries_must_remain_blocked=`True`

| Lane | Pair | Side | Method | Status | Blockers | Evidence needed |
|---|---|---|---|---|---|---|
| `range_trader:GBP_USD:SHORT:RANGE_ROTATION` | `GBP_USD` | `SHORT` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `STALE_QUOTE, SPREAD_TOO_WIDE, LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | fresh broker snapshot, forecasts, projections, and order_intents from the same evidence packet; clear global support/profitability P0s before ordinary fresh entries; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread; raw-NAV/margin capacity sufficient for the broker 1000-unit production floor and risk budget |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED, STALE_QUOTE, SPREAD_TOO_WIDE, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | fresh broker snapshot, forecasts, projections, and order_intents from the same evidence packet; fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread; explicit operator approval or changed broker truth before overlapping manual/operator exposure |
| `range_trader:EUR_JPY:SHORT:RANGE_ROTATION` | `EUR_JPY` | `SHORT` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `STALE_QUOTE, SPREAD_TOO_WIDE, LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | fresh broker snapshot, forecasts, projections, and order_intents from the same evidence packet; clear global support/profitability P0s before ordinary fresh entries; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread; raw-NAV/margin capacity sufficient for the broker 1000-unit production floor and risk budget |
| `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT` | `AUD_JPY` | `SHORT` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, STALE_QUOTE, SPREAD_TOO_WIDE, HARVEST_TP_STRUCTURE_MISSING, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | fresh broker snapshot, forecasts, projections, and order_intents from the same evidence packet; fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread |
| `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `STALE_QUOTE, SPREAD_TOO_WIDE, REWARD_RISK_TOO_LOW, EXHAUSTION_RANGE_CHASE, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | fresh broker snapshot, forecasts, projections, and order_intents from the same evidence packet; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED, STALE_QUOTE, SPREAD_TOO_WIDE, REWARD_RISK_TOO_LOW, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | fresh broker snapshot, forecasts, projections, and order_intents from the same evidence packet; fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread; explicit operator approval or changed broker truth before overlapping manual/operator exposure |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:MARKET` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED, STALE_QUOTE, SPREAD_TOO_WIDE, REWARD_RISK_TOO_LOW, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | fresh broker snapshot, forecasts, projections, and order_intents from the same evidence packet; clear global support/profitability P0s before ordinary fresh entries; fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread; explicit operator approval or changed broker truth before overlapping manual/operator exposure |
| `range_trader:GBP_JPY:SHORT:RANGE_ROTATION` | `GBP_JPY` | `SHORT` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `STALE_QUOTE, SPREAD_TOO_WIDE, RANGE_COUNTERTREND_RR_TOO_LOW, LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | fresh broker snapshot, forecasts, projections, and order_intents from the same evidence packet; clear global support/profitability P0s before ordinary fresh entries; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread; raw-NAV/margin capacity sufficient for the broker 1000-unit production floor and risk budget |
| `range_trader:USD_JPY:LONG:RANGE_ROTATION` | `USD_JPY` | `LONG` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `STALE_QUOTE, SPREAD_TOO_WIDE, LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, RANGE_ROTATION_BROADER_LOCATION_CHASE, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | fresh broker snapshot, forecasts, projections, and order_intents from the same evidence packet; clear global support/profitability P0s before ordinary fresh entries; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread; raw-NAV/margin capacity sufficient for the broker 1000-unit production floor and risk budget |
| `trend_trader:EUR_JPY:SHORT:TREND_CONTINUATION` | `EUR_JPY` | `SHORT` | `TREND_CONTINUATION` | `DRY_RUN_BLOCKED` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, STALE_QUOTE, SPREAD_TOO_WIDE, LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | fresh broker snapshot, forecasts, projections, and order_intents from the same evidence packet; clear global support/profitability P0s before ordinary fresh entries; fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread; raw-NAV/margin capacity sufficient for the broker 1000-unit production floor and risk budget |

- Ordinary fresh entries must remain blocked for every near-ready diagnostic row until its blockers clear in refreshed broker/forecast/replay evidence.

## qr-trader Scheduled Run Watchdog

- Status: `UNAVAILABLE` severity=`INFO`
- Generated: `None`
- Last run evidence: `None`
- Last run source: `None` path=`None`
- Minutes since last run: `None`
- Missed expected window: `None`
- Suspected cause: none

## Guardian Receipt Consumption

- Status: `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`
- Generated: `2026-07-03T20:46:22.414090+00:00`
- Normal routing allowed: `False`
- Unresolved issue count: `2`
- Recommended next action: No unresolved guardian receipt issue is currently blocking normal routing.
- `NEEDS_OPERATOR_REVIEW` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`832d2908eeb84b2f` action=`REDUCE` lifecycle=`EXPIRED` normal_routing_allowed=`False`
- `NEEDS_OPERATOR_REVIEW` issue=`CURRENT_GUARDIAN_P0_UNKNOWN_EXPOSURE` event=`aafaf3622a11c9c7` action=`REDUCE` lifecycle=`CURRENT_GUARDIAN_EVENT` normal_routing_allowed=`False`

## Guardian Receipt Issues

- Guardian receipt issues: none

## Operator Manual Positions

| Pair | Side | Units | Avg Entry | UPL JPY | Pip Value | Thesis | State | Margin | Harvest Zone | Invalidation Evidence |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| `EUR_USD` | `SHORT` | `30000` | `1.14048` | `-16445.4764` | `485.1173` | operator-confirmed manual EUR_USD short; keep open and monitor read-only | `ALIVE` | `OK` | operator review required before any TP modification or profit action | no major-figure invalidation configured; red P/L alone is ignored |

- Management rule: observe, TP-assist, and report only; no SL, loss-side close, or averaging unless the operator explicitly asks.

## Repair Requests

| Code | Status | Source | Verify |
|---|---|---|---|
| `REPAIR_DIRECTIONAL_INVERSION_COUNTERFACTUAL` | `WAITING_FOR_DIRECTIONAL_INVERSION_REPLAY_EVIDENCE` | `BROKER_TRUTH_OPPOSITE_SIDE_WOULD_CLEAR_MINIMUM_5PCT`, `DIRECTIONAL_INVERSION_REPLAY_EVIDENCE_MISSING`, `STALE_QUOTE`, `TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE`, `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`, `SPREAD_TOO_WIDE`, `HARVEST_TP_STRUCTURE_MISSING`, `RANGE_COUNTERTREND_RR_TOO_LOW`, `STOP_TOO_THIN_FOR_SPREAD`, `TARGET_TOO_THIN_FOR_SPREAD`, `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`, `EXHAUSTION_RANGE_CHASE`, `PATTERN_REVERSAL_CHASE`, `LOSS_ASYMMETRY_GUARD_EXCEEDED` | `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot`, `PYTHONPATH=src python3 scripts/oanda_history_replay_validate.py --forecast-history data/forecast_history.jsonl --pairs EUR_USD --granularity S5 --auto-history-min-days 30` |
| `COLLECT_BIDASK_REPLAY_EVIDENCE` | `BIDASK_REPLAY_WAITING_FOR_FORECAST_SAMPLE_COVERAGE` | `BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN` | `python3 scripts/oanda_history_replay_validate.py --forecast-history data/forecast_history.jsonl --granularity S5 --history-dir logs/replay/oanda_history/20260703T072439Z --history-dir logs/replay/oanda_history/20260703T080331Z --history-dir logs/replay/oanda_history/20260703T120929Z --history-dir logs/replay/oanda_history/20260703T123013Z --history-dir logs/replay/oanda_history/20260703T134642Z --history-dir logs/replay/oanda_history/20260703T135559Z --history-dir logs/replay/oanda_history/20260703T142126Z --history-dir logs/replay/oanda_history/20260703T142653Z --history-dir logs/replay/oanda_history/20260703T143956Z --auto-history-min-days 30 --stable-min-active-days 3 --stable-max-daily-sample-share 0.7 --stable-min-positive-day-rate 0.6666666667` |
| `REPAIR_FRONTIER_LANE_BLOCKER` | `FRONTIER_WAITING_FOR_FRESH_QUOTE` | `STALE_QUOTE` | `PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json`, `PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts` |

- Automation contract: Codex may edit code/tests/docs, run tests, commit, and sync live; orders, cancels, closes, and launchd changes require explicit approval or the existing gateway path.

## Bid/Ask Replay Coverage

- Status: `UNDER_SAMPLED` replay=`BIDASK_REPLAY_WAITING_FOR_FORECAST_SAMPLE_COVERAGE` price_truth=`PRICE_TRUTH_OK` fetch_required=`False`
- Rule counts: edge=`0` daily_stable_edge=`0` support=`17` daily_stable_support=`12` rank_only_support=`5` negative=`52`
- Daily stability requirement: active_days>=`3` max_daily_share<=`0.7` positive_day_rate>=`0.6666666666666666`
- Under-sampled pair-directions: `53` missing_evaluated_samples=`0`

| Pair | Direction | Samples | Active days | Gap reasons |
|---|---|---:|---:|---|
| `AUD_CAD` | `DOWN` | `620` | `None` | none |
| `AUD_CAD` | `UP` | `890` | `None` | none |
| `AUD_CHF` | `DOWN` | `756` | `None` | none |
| `AUD_CHF` | `UP` | `529` | `None` | none |
| `AUD_JPY` | `DOWN` | `1361` | `None` | none |
| `AUD_JPY` | `UP` | `1246` | `None` | none |
| `AUD_NZD` | `DOWN` | `488` | `None` | none |
| `AUD_NZD` | `UP` | `857` | `None` | none |
- First under-sampled directions: `AUD_CAD:DOWN`, `AUD_CAD:UP`, `AUD_CHF:DOWN`, `AUD_CHF:UP`, `AUD_JPY:DOWN`, `AUD_JPY:UP`, `AUD_NZD:DOWN`, `AUD_NZD:UP`, `AUD_USD:DOWN`, `AUD_USD:UP`, `CAD_CHF:UP`, `CAD_JPY:DOWN`
- Current-intent local BA history: `LOCAL_HISTORY_GAP` complete=`False` diagnostic_only=`True` missing=S5:AUD_CHF,AUD_JPY,AUD_USD,CHF_JPY,EUR_AUD,EUR_JPY,EUR_USD,GBP_AUD,GBP_JPY,GBP_USD,NZD_USD,USD_CAD,USD_JPY; M5:AUD_CHF,AUD_JPY,AUD_USD,CHF_JPY,EUR_AUD,EUR_JPY,EUR_USD,GBP_AUD,GBP_JPY,GBP_USD,NZD_USD,USD_CAD,USD_JPY
- Diagnostic history fetch command: `PYTHONPATH=src python3 scripts/oanda_history_fetch.py --pairs AUD_CHF,AUD_JPY,AUD_USD,CHF_JPY,EUR_AUD,EUR_JPY,EUR_USD,GBP_AUD,GBP_JPY,GBP_USD,NZD_USD,USD_CAD,USD_JPY --granularities S5,M5 --price BA --days 120 --output-dir logs/replay/oanda_history`
- Live permission: remains blocked until forecast sample coverage graduates from `UNDER_SAMPLED` and replay rules become daily-stable/live-grade.

## Profit Capture Repair

- Status: `HISTORICAL_DIAGNOSTIC_ONLY`
- Actual loss-close PL JPY: `-39275.343`
- Counterfactual profit-capture PL JPY: `-20546.009`
- Counterfactual profit-capture delta JPY: `18729.334`
- Production-gate replay triggered: `13`
- Production-gate replay delta JPY: `18771.737`
- Clearance condition: post-repair production-gate replay remains clean; historical pre-repair misses stay diagnostic and must not be used as the clearance condition
- Verify: `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80`

| Repair Trade | Pair | Side | Exit | Trigger UTC | Repair pips | Noise floor | Repair JPY | Delta JPY |
|---|---|---|---|---|---:|---:|---:|---:|
| `471240` | `EUR_USD` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `2026-05-14T21:36:00+00:00` | `3.7` | `3.0` | `897.467` | `6165.013` |
| `471232` | `EUR_USD` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `2026-05-14T17:39:00+00:00` | `3.0` | `1.7` | `339.582` | `3647.004` |
| `471414` | `EUR_USD` | `SHORT` | `MARKET_ORDER_TRADE_CLOSE` | `2026-05-21T15:08:00+00:00` | `4.1` | `3.55` | `198.898` | `2841.004` |
| `471979` | `AUD_CHF` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `2026-06-04T06:54:00+00:00` | `5.5` | `1.9` | `398.528` | `1613.933` |
| `472222` | `GBP_CHF` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `2026-06-11T17:34:00+00:00` | `7.5` | `2.8` | `298.896` | `1280.69` |

| Trade | Pair | Side | Exit | MFE JPY | TP progress | Counterfactual JPY | Delta JPY | Realized JPY |
|---|---|---|---|---:|---:|---:|---:|---:|
| `471240` | `EUR_USD` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `897.467` | `0.4066` | `662.185` | `5929.731` | `-5267.546` |
| `471232` | `EUR_USD` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `373.54` | `0.3626` | `309.02` | `3616.442` | `-3307.422` |
| `471414` | `EUR_USD` | `SHORT` | `MARKET_ORDER_TRADE_CLOSE` | `198.898` | `0.5125` | `116.428` | `2758.534` | `-2642.106` |
| `471979` | `AUD_CHF` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `456.496` | `0.3913` | `349.98` | `1565.385` | `-1215.405` |
| `472222` | `GBP_CHF` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `298.896` | `0.3138` | `285.745` | `1267.539` | `-981.794` |

## Guardian

- Required: `True`
- Label: `com.quantrabbit.position-guardian`
- Plist: `/Users/tossaki/Library/LaunchAgents/com.quantrabbit.position-guardian.plist` exists=`True`
- Launchd loaded: `True`
- Heartbeat path: `/Users/tossaki/App/QuantRabbit-live/data/position_guardian.json`
- Heartbeat generated: `2026-07-05T18:09:24.499319+00:00`

## Directional Inversion Counterfactuals

| Trade | Owner | Pair | Actual | Opposite | Actual UPL JPY | Opposite gross JPY | Clears 5% minimum | Replay status | Evidence status |
|---|---|---|---|---|---:|---:|---|---|---|
| `472987` | `operator_manual` | `EUR_USD` | `SHORT` | `LONG` | `-16445.476` | `16445.476` | `True` | `CONTRARIAN_REPLAY_NO_PAIR_SUPPORT` | `MISSING_REPEATED_SPREAD_INCLUDED_EVIDENCE` |

- Counterfactuals are gross sign-flips of current broker-truth unrealized P/L; they are repair evidence, not live inversion permission.

## Top Intent Blockers

- `STALE_QUOTE`: `73`
- `SPREAD_TOO_WIDE`: `73`
- `TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE`: `73`
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`: `73`
- `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION`: `64`
- `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`: `64`
- `LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT`: `60`
- `EXHAUSTION_RANGE_CHASE`: `33`
- `RANGE_FORECAST_REQUIRES_RANGE_ROTATION`: `26`
- `STRATEGY_NOT_ELIGIBLE`: `22`
- `RANGE_ROTATION_BROADER_LOCATION_CHASE`: `21`
- `HARVEST_TP_STRUCTURE_MISSING`: `17`

## Guardian Recovery Candidates

- none

## Repair Frontier Blockers After Support

| Blocker | Lanes | Reward JPY | Examples |
|---|---:|---:|---|
| `STALE_QUOTE` | `3` | `8801.635` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`, `range_trader:AUD_JPY:LONG:RANGE_ROTATION`, `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` |
| `TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE` | `3` | `8801.635` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`, `range_trader:AUD_JPY:LONG:RANGE_ROTATION`, `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` |
| `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | `3` | `8801.635` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`, `range_trader:AUD_JPY:LONG:RANGE_ROTATION`, `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` |
| `SPREAD_TOO_WIDE` | `3` | `8801.635` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`, `range_trader:AUD_JPY:LONG:RANGE_ROTATION`, `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` |
| `HARVEST_TP_STRUCTURE_MISSING` | `1` | `5979.879` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` |
| `RANGE_COUNTERTREND_RR_TOO_LOW` | `1` | `2314.0` | `range_trader:AUD_JPY:LONG:RANGE_ROTATION` |
| `STOP_TOO_THIN_FOR_SPREAD` | `1` | `2314.0` | `range_trader:AUD_JPY:LONG:RANGE_ROTATION` |
| `TARGET_TOO_THIN_FOR_SPREAD` | `1` | `2314.0` | `range_trader:AUD_JPY:LONG:RANGE_ROTATION` |

## Repair Frontier

| Lane | Pair | Side | Method | Reward JPY | Remaining blockers after support |
|---|---|---|---|---:|---|
| `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` | `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `5979.879` | `STALE_QUOTE, SPREAD_TOO_WIDE, PATTERN_REVERSAL_CHASE, HARVEST_TP_STRUCTURE_MISSING, EXHAUSTION_RANGE_CHASE, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` |
| `range_trader:AUD_JPY:LONG:RANGE_ROTATION` | `AUD_JPY` | `LONG` | `RANGE_ROTATION` | `2314.0` | `STALE_QUOTE, SPREAD_TOO_WIDE, LOSS_ASYMMETRY_GUARD_EXCEEDED, RANGE_COUNTERTREND_RR_TOO_LOW, TARGET_TOO_THIN_FOR_SPREAD, STOP_TOO_THIN_FOR_SPREAD, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` |
| `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `507.756` | `STALE_QUOTE, SPREAD_TOO_WIDE, REWARD_RISK_TOO_LOW, EXHAUSTION_RANGE_CHASE, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` |

## OANDA Audit-Only Local TP Proof Required

- none

## RANGE Forecast Superseded Repair Lanes

| Lane | Pair | Side | Method | Reward JPY | Range counterpart |
|---|---|---|---|---:|---|
| `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT` | `AUD_JPY` | `SHORT` | `BREAKOUT_FAILURE` | `5054.0` | `range_trader:AUD_JPY:SHORT:RANGE_ROTATION` |
| `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE` | `AUD_JPY` | `SHORT` | `BREAKOUT_FAILURE` | `5054.0` | `range_trader:AUD_JPY:SHORT:RANGE_ROTATION` |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `291.07` | `range_trader:EUR_USD:SHORT:RANGE_ROTATION` |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `211.835` | `range_trader:EUR_USD:SHORT:RANGE_ROTATION` |

## RANGE Forecast Missing Counterpart Repair Lanes

- none

## Global Unlock Frontier

- none

## Target Firepower Evidence

- Status: `VERIFIED_TARGET_10_ROUTE_ESTIMATED`
- Best bucket: `evidence_queue`
- 5% estimated reachable: `True`
- 10% estimated reachable: `True`
- Audit only, no live permission grant: `true`
- `high_precision` return/day=`59.015709` trades_needed_5pct=`8.0` vehicles=`['AUD_USD|SHORT|range_reversion|tp1.25_sl1', 'GBP_USD|SHORT|range_reversion|tp1.25_sl1', 'AUD_USD|SHORT|range_reversion|tp1.25_sl1']`
- `evidence_queue` return/day=`61.172817` trades_needed_5pct=`9.0` vehicles=`['EUR_JPY|SHORT|trend_continuation|tp1_sl0.75', 'EUR_JPY|SHORT|trend_continuation|tp1.25_sl1', 'USD_JPY|SHORT|range_reversion|tp1_sl0.75']`

## Acceptance Repair Plan

- Loop breaker: Rerunning profitability-acceptance alone cannot clear these P0s; the listed proof condition must change first. Evidence-collection items also require fresh replay/mining output before they can graduate into live-grade turnover.

| Code | Clearance condition | Verify |
|---|---|---|
| `SELF_IMPROVEMENT_P0_PRESENT` | self_improvement_audit has zero P0 findings, or the only remaining discipline finding has been demoted by verified clean gateway close recovery | `PYTHONPATH=src python3 -m quant_rabbit.cli self-improvement-audit` |
| `NEGATIVE_EXPECTANCY_ACTIVE` | capture_economics.status is no longer NEGATIVE_EXPECTANCY, or entries are limited to exact TP-proven repair/harvest shapes with positive expectancy evidence | `PYTHONPATH=src python3 -m quant_rabbit.cli capture-economics` |
| `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE` | no TP-proven segment remains net-damaged by MARKET_ORDER_TRADE_CLOSE leakage; preserve broker TP and guardian capture instead of scaling market-close loss paths | `PYTHONPATH=src python3 -m quant_rabbit.cli capture-economics` |
| `MARKET_CLOSE_LEAK_FAMILY_BLOCKED` | the named acceptance P0 finding disappears from profitability_acceptance after its evidence metric changes | `PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance` |

## Acceptance Evidence Collection

| Code | Clearance condition | Verify |
|---|---|---|
| `BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN` | OANDA bid/ask price truth is complete for loaded samples; collect more forecast_history samples across the under-sampled pair-directions, then rerun replay validation and require global all-currency sample coverage to graduate from UNDER_SAMPLED before claiming all-currency high-turn readiness | `python3 scripts/oanda_history_replay_validate.py --forecast-history data/forecast_history.jsonl --granularity S5 --history-dir logs/replay/oanda_history/20260703T072439Z --history-dir logs/replay/oanda_history/20260703T080331Z --history-dir logs/replay/oanda_history/20260703T120929Z --history-dir logs/replay/oanda_history/20260703T123013Z --history-dir logs/replay/oanda_history/20260703T134642Z --history-dir logs/replay/oanda_history/20260703T135559Z --history-dir logs/replay/oanda_history/20260703T142126Z --history-dir logs/replay/oanda_history/20260703T142653Z --history-dir logs/replay/oanda_history/20260703T143956Z --auto-history-min-days 30 --stable-min-active-days 3 --stable-max-daily-sample-share 0.7 --stable-min-positive-day-rate 0.6666666667` |

## Current Profit Capture

- none

## Open Trader Positions

- none

## Operator Actions

- `REFRESH_SUPPORT_PANEL`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — read-only status refresh for trader operations
- `REVIEW_GUARDIAN_RECEIPT_OPERATOR_REVIEW`: `sed -n '1,200p' docs/guardian_receipt_operator_review_report.md` — operator review artifact exists but does not clear normal routing; inspect the decision row before any fresh-entry routing
- `MONITOR_POST_REPAIR_TP_PROGRESS_EVIDENCE`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — post-repair production-gate replay is clean; keep monitoring without treating pre-repair misses as a live blocker
- `REFRESH_EVIDENCE_PACKET`: `PYTHONPATH=src python3 -m quant_rabbit.cli cycle-refresh --daily-risk-pct 10` — refresh forecasts, intents, sidecar audits, and route from current broker truth
- `READ_ACCEPTANCE_BLOCKERS`: `sed -n '1,220p' docs/profitability_acceptance_report.md` — inspect red/green acceptance invariants before increasing turnover
- `FOLLOW_ACCEPTANCE_REPAIR_PLAN`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — use profitability_acceptance.repair_plan clearance conditions; rerunning acceptance alone will loop until those proof metrics change
- `WORK_REPAIR_FRONTIER_REMAINING_BLOCKERS`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — guardian/global repairs are not enough; top remaining repair-frontier blocker is STALE_QUOTE across 3 lane(s)
- `WORK_TARGET_FIREPOWER_BLOCKERS`: `PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance` — firepower audit estimates enough turnover for the 5% floor, but live permission still depends on clearing acceptance, guardian, and lane blockers

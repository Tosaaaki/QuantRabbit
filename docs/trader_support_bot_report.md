# Trader Support Bot Report

- Generated at UTC: `2026-07-06T15:10:53.773668+00:00`
- Status: `SUPPORT_BLOCKED`
- Read only: `True`
- Live side effects: `0`

## Support Gates

| Gate | Value |
|---|---|
| Fresh entry send allowed | `False` |
| Repair basket send allowed | `False` |
| Guardian active | `True` source=`launchd+heartbeat` |
| Guardian heartbeat fresh | `True` age=`24.647`s |
| qr-trader scheduled-run watchdog | `BLOCKED` severity=`P0` minutes_since=`54.556` |
| Guardian receipt consumption | `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` normal_routing_allowed=`False` unresolved=`1` |
| LIVE_READY lanes | `0` / `101` |
| Near-ready diagnostic lanes | `101` |
| Order intents freshness | `ORDER_INTENTS_ALIGNED_WITH_BROKER_SNAPSHOT` staleness=`-116.617`s |
| Profitability acceptance freshness | `PROFITABILITY_ACCEPTANCE_ALIGNED_WITH_INPUTS` stale_inputs=`[]` |
| Repair LIVE_READY lanes | `0` |
| Repair lanes after guardian recovery | `0` |
| Repair frontier lanes | `4` |
| RANGE forecast superseded repair lanes | `2` |
| OANDA audit-only local TP proof required | `0` |
| RANGE forecast missing counterpart lanes | `0` |
| Repair frontier clear after support | `0` |
| Repair frontier blocked after support | `4` |
| Global unlock frontier lanes | `0` |
| Profit-capture misses | `14` gap=`5505.702` JPY counterfactual_delta=`18741.932` JPY repair_replay_triggered=`13` repair_delta=`18787.162` JPY |
| Current profit-capture positions | bankable=`0` watch=`0` blocked=`0` |
| Open positions | `1` unknown_owner=`0` operator_manual=`1` |
| Operator manual JPY add guard | `False` code=`None` |
| Open trader positions | `0` upl=`0.0` JPY |
| Directional inversion 5% counterfactuals | `0` |
| Target remaining | `28718.6` JPY |
| Rolling 30d equity | raw=`280353.469` funding_adjusted=`180353.469` capital_flows_30d=`100000.0` JPY |
| Rolling 30d multiplier | raw=`1.599232` funding_adjusted=`1.028798` |
| Remaining to 4x | raw=`420866.752` funding_adjusted=`520866.752` JPY |
| Required return | calendar_funding_adjusted=`5.978975` active_funding_adjusted=`8.240683` |
| Target basis | performance=`funding_adjusted` sizing=`raw_nav` |
| Firepower 5% audit estimate | `True` best=`evidence_queue` |
| Firepower 5% operational reachable | `False` blockers=`['GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING', 'GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER', 'GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW', 'SELF_IMPROVEMENT_P0_PRESENT', 'NO_LIVE_READY_LANES', 'PROFITABILITY_ACCEPTANCE_BLOCKED', 'FRESH_ENTRY_SEND_NOT_ALLOWED']` |

## Blockers

- `P0` `GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING`: guardian receipt consumption status does not allow normal new-entry routing; status=GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED classifications=NEEDS_OPERATOR_REVIEW,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,STALE_ACKNOWLEDGED,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,STALE_ACKNOWLEDGED,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY,HISTORICAL_ONLY
- `WARN` `GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER`: receipt_lifecycle=EXPIRED while consumed_by_trader=false; sources=2
- `P0` `GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW`: Receipt event 832d2908eeb84b2f REDUCE requires operator review before normal new-entry routing; operator_review_status=OPERATOR_REVIEW_STALE, reason=operator review row is expired.
- `P0` `SELF_IMPROVEMENT_P0_PRESENT`: self-improvement audit has 2 P0 finding(s)
- `P1` `NO_LIVE_READY_LANES`: daily target is open but no lane is LIVE_READY
- `P0` `PROFITABILITY_ACCEPTANCE_BLOCKED`: profitability acceptance status is PROFITABILITY_ACCEPTANCE_BLOCKED

## Near-Ready Lanes

- Shortest path: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` status=`BLOCKED_NEAR_READY_LANE` next=`spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method`
- Shortest path remains non-executable: live_permission=`False` ordinary_fresh_entries_must_remain_blocked=`True`

| Lane | Pair | Side | Method | Status | Blockers | Evidence needed |
|---|---|---|---|---|---|---|
| `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` | spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method |
| `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` | `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` | spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread |
| `range_trader:GBP_JPY:SHORT:RANGE_ROTATION` | `GBP_JPY` | `SHORT` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `LOSS_ASYMMETRY_GUARD_EXCEEDED, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` | spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` | fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; explicit operator approval or changed broker truth before overlapping manual/operator exposure |
| `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE:LIMIT` | `EUR_JPY` | `LONG` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` | clear global support/profitability P0s before ordinary fresh entries; fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method |
| `range_trader:USD_CAD:LONG:RANGE_ROTATION` | `USD_CAD` | `LONG` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `RANGE_COUNTERTREND_RR_TOO_LOW, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, RANGE_ROTATION_BROADER_LOCATION_CHASE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` | clear global support/profitability P0s before ordinary fresh entries; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread |
| `range_trader:EUR_JPY:SHORT:RANGE_ROTATION` | `EUR_JPY` | `SHORT` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `RANGE_COUNTERTREND_RR_TOO_LOW, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` | clear global support/profitability P0s before ordinary fresh entries; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread |
| `range_trader:CAD_JPY:SHORT:RANGE_ROTATION` | `CAD_JPY` | `SHORT` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `RANGE_COUNTERTREND_RR_TOO_LOW, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, RANGE_FORMING_HTF_TREND_CONFLICT, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` | clear global support/profitability P0s before ordinary fresh entries; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread |
| `range_trader:GBP_USD:SHORT:RANGE_ROTATION` | `GBP_USD` | `SHORT` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `RANGE_COUNTERTREND_RR_TOO_LOW, LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` | clear global support/profitability P0s before ordinary fresh entries; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread; raw-NAV/margin capacity sufficient for the broker 1000-unit production floor and risk budget |
| `trend_trader:EUR_JPY:SHORT:TREND_CONTINUATION` | `EUR_JPY` | `SHORT` | `TREND_CONTINUATION` | `DRY_RUN_BLOCKED` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, CHART_DIRECTION_CONFLICT, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` | clear global support/profitability P0s before ordinary fresh entries; fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread |

- Ordinary fresh entries must remain blocked for every near-ready diagnostic row until its blockers clear in refreshed broker/forecast/replay evidence.

## qr-trader Scheduled Run Watchdog

- Status: `BLOCKED` severity=`P0`
- Generated: `2026-07-06T15:07:36.376459+00:00`
- Last run evidence: `2026-07-06T14:13:03.019205+00:00`
- Last run source: `trader_journal.ts` path=`/Users/tossaki/App/QuantRabbit-live/logs/trader_journal.jsonl`
- Minutes since last run: `54.556`
- Missed expected window: `False`
- Suspected cause: Guardian receipt consumption issues require the next trader cycle to resolve or classify them before ordinary entries.

## Guardian Receipt Consumption

- Status: `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`
- Generated: `2026-07-06T14:15:46.357972+00:00`
- Normal routing allowed: `False`
- Unresolved issue count: `1`
- Recommended next action: Run the qr-trader draft/preflight path to write a durable guardian_receipt_consumption classification for the current watchdog receipt issue before any ordinary fresh entry. Current unclassified event(s): d8a0bfd2a3bbf851.
- `NEEDS_OPERATOR_REVIEW` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`832d2908eeb84b2f` action=`REDUCE` lifecycle=`EXPIRED` normal_routing_allowed=`False`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`eb0e4061dc56c2f2` action=`NO_ACTION` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`93556cccbc851078` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`76513b64ad02a1cc` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `STALE_ACKNOWLEDGED` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`76513b64ad02a1cc` action=`HOLD` lifecycle=`ACTIVE` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`39af8884172fe20b` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`4946323e3e8dd9b5` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`aaa294fa0ecb9c30` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`ea2d3ebd4caec51b` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `STALE_ACKNOWLEDGED` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`ea2d3ebd4caec51b` action=`HOLD` lifecycle=`ACTIVE` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`123116a3d2a9573a` action=`NO_ACTION` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`44a3051351c608a5` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`84d38ff4eb2494d8` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`789a3de4be61bbe5` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`6ea3f90c57c9d22b` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`592abf5880f5c11c` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`d9b3bf5e56609732` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`62e493868408ffb2` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`4eb3e8f18ae39816` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`
- `HISTORICAL_ONLY` issue=`GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` event=`78b5716824c5d935` action=`HOLD` lifecycle=`EXPIRED` normal_routing_allowed=`True`

## Guardian Receipt Issues

- Guardian receipt issues:
  - `WARN` `GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER`: receipt_lifecycle=EXPIRED while consumed_by_trader=false; sources=2
  - `P0` `GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW`: Receipt event 832d2908eeb84b2f REDUCE requires operator review before normal new-entry routing; operator_review_status=OPERATOR_REVIEW_STALE, reason=operator review row is expired.

## Operator Manual Positions

| Pair | Side | Units | Avg Entry | UPL JPY | Pip Value | Thesis | State | Margin | Harvest Zone | Invalidation Evidence |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| `EUR_USD` | `SHORT` | `30000` | `1.14048` | `-6832.5057` | `488.0361` | operator manual thesis | `ALIVE` | `OK` |  | no major-figure invalidation configured; red P/L alone is ignored |

- Management rule: observe, TP-assist, and report only; no SL, loss-side close, or averaging unless the operator explicitly asks.

## Repair Requests

| Code | Status | Source | Verify |
|---|---|---|---|
| `REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY` | `RESIDUAL_GROUPS_ALREADY_BLOCKED_WAITING_FOR_REPLAY` | `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE` | `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80` |
| `COLLECT_BIDASK_REPLAY_EVIDENCE` | `BIDASK_REPLAY_WAITING_FOR_FORECAST_SAMPLE_COVERAGE` | `BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN` | `python3 scripts/oanda_history_replay_validate.py --forecast-history data/forecast_history.jsonl --granularity S5 --history-dir logs/replay/oanda_history/20260703T072439Z --history-dir logs/replay/oanda_history/20260703T080331Z --history-dir logs/replay/oanda_history/20260703T120929Z --history-dir logs/replay/oanda_history/20260703T123013Z --history-dir logs/replay/oanda_history/20260703T134642Z --history-dir logs/replay/oanda_history/20260703T135559Z --history-dir logs/replay/oanda_history/20260703T142126Z --history-dir logs/replay/oanda_history/20260703T142653Z --history-dir logs/replay/oanda_history/20260703T143956Z --auto-history-min-days 30 --stable-min-active-days 3 --stable-max-daily-sample-share 0.7 --stable-min-positive-day-rate 0.6666666667` |
| `REPAIR_FRONTIER_LANE_BLOCKER` | `READY_FOR_CODE_OR_EVIDENCE_REPAIR` | `GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` | `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot`, `PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts` |

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
- Current-intent local BA history: `LOCAL_HISTORY_COMPLETE` complete=`True` diagnostic_only=`True` missing=S5:none; M5:none
- Live permission: remains blocked until forecast sample coverage graduates from `UNDER_SAMPLED` and replay rules become daily-stable/live-grade.

## Profit Capture Repair

- Status: `HISTORICAL_DIAGNOSTIC_ONLY`
- Actual loss-close PL JPY: `-39275.343`
- Counterfactual profit-capture PL JPY: `-20533.411`
- Counterfactual profit-capture delta JPY: `18741.932`
- Production-gate replay triggered: `13`
- Production-gate replay delta JPY: `18787.162`
- Clearance condition: post-repair production-gate replay remains clean; historical pre-repair misses stay diagnostic and must not be used as the clearance condition
- Verify: `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80`

| Repair Trade | Pair | Side | Exit | Trigger UTC | Repair pips | Noise floor | Repair JPY | Delta JPY |
|---|---|---|---|---|---:|---:|---:|---:|
| `471240` | `EUR_USD` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `2026-05-14T21:36:00+00:00` | `3.7` | `3.0` | `902.544` | `6170.09` |
| `471232` | `EUR_USD` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `2026-05-14T17:39:00+00:00` | `3.0` | `1.7` | `341.503` | `3648.925` |
| `471414` | `EUR_USD` | `SHORT` | `MARKET_ORDER_TRADE_CLOSE` | `2026-05-21T15:08:00+00:00` | `4.1` | `3.55` | `200.023` | `2842.129` |
| `471979` | `AUD_CHF` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `2026-06-04T06:54:00+00:00` | `5.5` | `1.9` | `399.189` | `1614.593` |
| `472222` | `GBP_CHF` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `2026-06-11T17:34:00+00:00` | `7.5` | `2.8` | `299.392` | `1281.186` |

| Trade | Pair | Side | Exit | MFE JPY | TP progress | Counterfactual JPY | Delta JPY | Realized JPY |
|---|---|---|---|---:|---:|---:|---:|---:|
| `471240` | `EUR_USD` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `902.544` | `0.4066` | `665.931` | `5933.477` | `-5267.546` |
| `471232` | `EUR_USD` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `375.654` | `0.3626` | `310.768` | `3618.19` | `-3307.422` |
| `471414` | `EUR_USD` | `SHORT` | `MARKET_ORDER_TRADE_CLOSE` | `200.023` | `0.5125` | `117.087` | `2759.193` | `-2642.106` |
| `471979` | `AUD_CHF` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `457.252` | `0.3913` | `350.56` | `1565.965` | `-1215.405` |
| `472222` | `GBP_CHF` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `299.392` | `0.3138` | `286.218` | `1268.012` | `-981.794` |

## Guardian

- Required: `True`
- Label: `com.quantrabbit.position-guardian`
- Plist: `/Users/tossaki/Library/LaunchAgents/com.quantrabbit.position-guardian.plist` exists=`True`
- Launchd loaded: `True`
- Heartbeat path: `/Users/tossaki/App/QuantRabbit-live/data/position_guardian.json`
- Heartbeat generated: `2026-07-06T15:10:29.126516+00:00`

## Directional Inversion Counterfactuals

| Trade | Owner | Pair | Actual | Opposite | Actual UPL JPY | Opposite gross JPY | Clears 5% minimum | Replay status | Evidence status |
|---|---|---|---|---|---:|---:|---|---|---|
| `472987` | `operator_manual` | `EUR_USD` | `SHORT` | `LONG` | `-6832.506` | `6832.506` | `False` | `CONTRARIAN_REPLAY_REJECTED` | `MISSING_REPEATED_SPREAD_INCLUDED_EVIDENCE` |

- Counterfactuals are gross sign-flips of current broker-truth unrealized P/L; they are repair evidence, not live inversion permission.

## Top Intent Blockers

- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`: `101`
- `GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY`: `101`
- `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION`: `92`
- `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`: `88`
- `LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT`: `60`
- `EXHAUSTION_RANGE_CHASE`: `44`
- `RANGE_COUNTERTREND_RR_TOO_LOW`: `39`
- `STRATEGY_NOT_ELIGIBLE`: `39`
- `RANGE_FORECAST_REQUIRES_RANGE_ROTATION`: `36`
- `FORECAST_WATCH_ONLY`: `30`
- `RANGE_ROTATION_BROADER_LOCATION_CHASE`: `30`
- `REWARD_RISK_TOO_LOW`: `29`

## Guardian Recovery Candidates

- none

## Repair Frontier Blockers After Support

| Blocker | Lanes | Reward JPY | Examples |
|---|---:|---:|---|
| `GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` | `4` | `8382.618` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT`, `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`, `range_trader:AUD_JPY:LONG:RANGE_ROTATION` |
| `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | `4` | `8382.618` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT`, `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`, `range_trader:AUD_JPY:LONG:RANGE_ROTATION` |
| `STRATEGY_NOT_ELIGIBLE` | `1` | `1652.0` | `range_trader:AUD_JPY:LONG:RANGE_ROTATION` |
| `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE` | `4` | `8382.618` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT`, `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`, `range_trader:AUD_JPY:LONG:RANGE_ROTATION` |
| `LOSS_ASYMMETRY_GUARD_EXCEEDED` | `2` | `3268.0` | `range_trader:AUD_JPY:LONG:RANGE_ROTATION`, `range_trader:GBP_JPY:SHORT:RANGE_ROTATION` |
| `BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE` | `1` | `2557.309` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` |
| `EXHAUSTION_RANGE_CHASE` | `1` | `1652.0` | `range_trader:AUD_JPY:LONG:RANGE_ROTATION` |
| `RANGE_ROTATION_BROADER_LOCATION_CHASE` | `1` | `1652.0` | `range_trader:AUD_JPY:LONG:RANGE_ROTATION` |

## Repair Frontier

| Lane | Pair | Side | Method | Reward JPY | Remaining blockers after support |
|---|---|---|---|---:|---|
| `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `2557.309` | `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` |
| `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` | `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `2557.309` | `BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` |
| `range_trader:AUD_JPY:LONG:RANGE_ROTATION` | `AUD_JPY` | `LONG` | `RANGE_ROTATION` | `1652.0` | `LOSS_ASYMMETRY_GUARD_EXCEEDED, RANGE_ROTATION_BROADER_LOCATION_CHASE, EXHAUSTION_RANGE_CHASE, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY, STRATEGY_NOT_ELIGIBLE` |
| `range_trader:GBP_JPY:SHORT:RANGE_ROTATION` | `GBP_JPY` | `SHORT` | `RANGE_ROTATION` | `1616.0` | `LOSS_ASYMMETRY_GUARD_EXCEEDED, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY` |

## OANDA Audit-Only Local TP Proof Required

- none

## RANGE Forecast Superseded Repair Lanes

| Lane | Pair | Side | Method | Reward JPY | Range counterpart |
|---|---|---|---|---:|---|
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `427.845` | `range_trader:EUR_USD:SHORT:RANGE_ROTATION` |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `426.218` | `range_trader:EUR_USD:SHORT:RANGE_ROTATION` |

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
| `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE` | month-scale production-gate replay is non-negative, or the top residual pair/side/method groups are removed by close-gate, TP-capture, or entry-selection changes before turnover is scaled | `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80` |

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
- `REFRESH_QR_TRADER_RUN_WATCHDOG`: `PYTHONPATH=src python3 tools/qr_trader_run_watchdog.py` — read-only scheduled-run watchdog refresh; does not run qr-trader, Codex wake, or broker calls by default
- `REVIEW_GUARDIAN_RECEIPT_CONSUMPTION`: `sed -n '1,160p' docs/guardian_receipt_consumption_report.md` — Run the qr-trader draft/preflight path to write a durable guardian_receipt_consumption classification for the current watchdog receipt issue before any ordinary fresh entry. Current unclassified event(s): d8a0bfd2a3bbf851.
- `QR_TRADER_WATCHDOG_RESOLVE_GUARDIAN_RECEIPT_IN_NEXT_TRADER_CYCLE`: `read data/guardian_action_receipt.json and docs/guardian_action_review.md before normal entries` — guardian receipt is unconsumed or expired without trader resolution
- `MONITOR_POST_REPAIR_TP_PROGRESS_EVIDENCE`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — post-repair production-gate replay is clean; keep monitoring without treating pre-repair misses as a live blocker
- `REFRESH_EVIDENCE_PACKET`: `PYTHONPATH=src python3 -m quant_rabbit.cli cycle-refresh --daily-risk-pct 10` — refresh forecasts, intents, sidecar audits, and route from current broker truth
- `READ_ACCEPTANCE_BLOCKERS`: `sed -n '1,220p' docs/profitability_acceptance_report.md` — inspect red/green acceptance invariants before increasing turnover
- `FOLLOW_ACCEPTANCE_REPAIR_PLAN`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — use profitability_acceptance.repair_plan clearance conditions; rerunning acceptance alone will loop until those proof metrics change
- `VERIFY_TP_PROGRESS_REPLAY_REPAIR`: `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80` — prove the OANDA candle replay TP-progress miss has cleared at the required coverage before treating high-turnover profit capture as repaired
- `WORK_REPAIR_FRONTIER_REMAINING_BLOCKERS`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — guardian/global repairs are not enough; top remaining repair-frontier blocker is GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY across 4 lane(s)
- `WORK_TARGET_FIREPOWER_BLOCKERS`: `PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance` — firepower audit estimates enough turnover for the 5% floor, but live permission still depends on clearing acceptance, guardian, and lane blockers

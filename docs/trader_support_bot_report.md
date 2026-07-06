# Trader Support Bot Report

- Generated at UTC: `2026-07-06T01:59:17.654938+00:00`
- Status: `SUPPORT_BLOCKED`
- Read only: `True`
- Live side effects: `0`

## Support Gates

| Gate | Value |
|---|---|
| Fresh entry send allowed | `False` |
| Repair basket send allowed | `False` |
| Guardian active | `True` source=`launchd+heartbeat` |
| Guardian heartbeat fresh | `True` age=`22.603`s |
| qr-trader scheduled-run watchdog | `UNAVAILABLE` severity=`INFO` minutes_since=`None` |
| Guardian receipt consumption | `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` normal_routing_allowed=`False` unresolved=`2` |
| LIVE_READY lanes | `0` / `82` |
| Near-ready diagnostic lanes | `82` |
| Order intents freshness | `ORDER_INTENTS_ALIGNED_WITH_BROKER_SNAPSHOT` staleness=`-16.357`s |
| Profitability acceptance freshness | `PROFITABILITY_ACCEPTANCE_ALIGNED_WITH_INPUTS` stale_inputs=`[]` |
| Repair LIVE_READY lanes | `0` |
| Repair lanes after guardian recovery | `0` |
| Repair frontier lanes | `2` |
| RANGE forecast superseded repair lanes | `4` |
| OANDA audit-only local TP proof required | `3` |
| RANGE forecast missing counterpart lanes | `0` |
| Repair frontier clear after support | `0` |
| Repair frontier blocked after support | `2` |
| Global unlock frontier lanes | `0` |
| Profit-capture misses | `1` gap=`214.2` JPY counterfactual_delta=`446.04` JPY repair_replay_triggered=`1` repair_delta=`466.2` JPY |
| Current profit-capture positions | bankable=`0` watch=`0` blocked=`0` |
| Open positions | `1` unknown_owner=`0` operator_manual=`1` |
| Operator manual JPY add guard | `False` code=`None` |
| Open trader positions | `0` upl=`0.0` JPY |
| Directional inversion 5% counterfactuals | `0` |
| Target remaining | `28718.6` JPY |
| Rolling 30d equity | raw=`274541.897` funding_adjusted=`174541.897` capital_flows_30d=`100000.0` JPY |
| Rolling 30d multiplier | raw=`1.601434` funding_adjusted=`1.018123` |
| Remaining to 4x | raw=`411198.324` funding_adjusted=`511198.324` JPY |
| Required return | calendar_funding_adjusted=`5.386027` active_funding_adjusted=`7.415701` |
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

- Shortest path: `range_trader:GBP_USD:SHORT:RANGE_ROTATION` status=`BLOCKED_NEAR_READY_LANE` next=`clear global support/profitability P0s before ordinary fresh entries`
- Shortest path remains non-executable: live_permission=`False` ordinary_fresh_entries_must_remain_blocked=`True`

| Lane | Pair | Side | Method | Status | Blockers | Evidence needed |
|---|---|---|---|---|---|---|
| `range_trader:GBP_USD:SHORT:RANGE_ROTATION` | `GBP_USD` | `SHORT` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | clear global support/profitability P0s before ordinary fresh entries; fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side |
| `range_trader:EUR_JPY:SHORT:RANGE_ROTATION` | `EUR_JPY` | `SHORT` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | clear global support/profitability P0s before ordinary fresh entries; fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method |
| `range_trader:USD_JPY:SHORT:RANGE_ROTATION` | `USD_JPY` | `SHORT` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | clear global support/profitability P0s before ordinary fresh entries; fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method |
| `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT` | `AUD_JPY` | `SHORT` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method |
| `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `REWARD_RISK_TOO_LOW, EXHAUSTION_RANGE_CHASE, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; explicit operator approval or changed broker truth before overlapping manual/operator exposure |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; explicit operator approval or changed broker truth before overlapping manual/operator exposure |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:MARKET` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `DRY_RUN_BLOCKED` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | clear global support/profitability P0s before ordinary fresh entries; fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; explicit operator approval or changed broker truth before overlapping manual/operator exposure |
| `range_trader:USD_JPY:LONG:RANGE_ROTATION` | `USD_JPY` | `LONG` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, RANGE_ROTATION_BROADER_LOCATION_CHASE, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | clear global support/profitability P0s before ordinary fresh entries; fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread |
| `range_trader:NZD_CHF:SHORT:RANGE_ROTATION` | `NZD_CHF` | `SHORT` | `RANGE_ROTATION` | `DRY_RUN_BLOCKED` | `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION, RANGE_ROTATION_BROADER_LOCATION_CHASE, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | clear global support/profitability P0s before ordinary fresh entries; fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side; spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method; current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread |

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
| `EUR_USD` | `SHORT` | `30000` | `1.14048` | `-12061.6976` | `486.3588` | operator-confirmed manual EUR_USD short; keep open and monitor read-only | `ALIVE` | `OK` | operator review required before any TP modification or profit action | no major-figure invalidation configured; red P/L alone is ignored |

- Management rule: observe, TP-assist, and report only; no SL, loss-side close, or averaging unless the operator explicitly asks.

## Repair Requests

| Code | Status | Source | Verify |
|---|---|---|---|
| `REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY` | `RESIDUAL_GROUPS_ALREADY_BLOCKED_WAITING_FOR_REPLAY` | `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE` | `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80` |
| `PROVE_OANDA_AUDIT_ONLY_LOCAL_TP_EDGE` | `READY_FOR_READ_ONLY_EVIDENCE_COLLECTION` | `OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED`, `RANGE_FORECAST_REQUIRES_RANGE_ROTATION`, `OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED`, `CHART_DIRECTION_CONFLICT`, `SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH`, `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`, `REWARD_RISK_TOO_LOW`, `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`, `RANGE_COUNTERTREND_RR_TOO_LOW` | `PYTHONPATH=src python3 scripts/oanda_history_fetch.py --pairs EUR_JPY,EUR_USD,GBP_JPY --granularities S5,M5 --price BA --days 120 --output-dir logs/replay/oanda_history`, `PYTHONPATH=src python3 scripts/oanda_history_replay_validate.py --history-dir logs/replay/oanda_history --granularity S5` |
| `COLLECT_BIDASK_REPLAY_EVIDENCE` | `BIDASK_REPLAY_WAITING_FOR_FORECAST_SAMPLE_COVERAGE` | `BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN` | `python3 scripts/oanda_history_replay_validate.py --forecast-history data/forecast_history.jsonl --granularity S5 --history-dir logs/replay/oanda_history/20260703T072439Z --history-dir logs/replay/oanda_history/20260703T080331Z --history-dir logs/replay/oanda_history/20260703T120929Z --history-dir logs/replay/oanda_history/20260703T123013Z --history-dir logs/replay/oanda_history/20260703T134642Z --history-dir logs/replay/oanda_history/20260703T135559Z --history-dir logs/replay/oanda_history/20260703T142126Z --history-dir logs/replay/oanda_history/20260703T142653Z --history-dir logs/replay/oanda_history/20260703T143956Z --auto-history-min-days 30 --stable-min-active-days 3 --stable-max-daily-sample-share 0.7 --stable-min-positive-day-rate 0.6666666667` |
| `REPAIR_FRONTIER_LANE_BLOCKER` | `READY_FOR_CODE_OR_EVIDENCE_REPAIR` | `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot`, `PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts` |

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
- Current-intent local BA history: `LOCAL_HISTORY_GAP` complete=`False` diagnostic_only=`True` missing=S5:AUD_CHF,AUD_JPY,AUD_USD,CAD_JPY,CHF_JPY,EUR_AUD,EUR_JPY,EUR_USD,GBP_AUD,GBP_JPY,GBP_NZD,GBP_USD,NZD_CHF,NZD_USD,USD_JPY; M5:AUD_CHF,AUD_JPY,AUD_USD,CAD_JPY,CHF_JPY,EUR_AUD,EUR_JPY,EUR_USD,GBP_AUD,GBP_JPY,GBP_NZD,GBP_USD,NZD_CHF,NZD_USD,USD_JPY
- Diagnostic history fetch command: `PYTHONPATH=src python3 scripts/oanda_history_fetch.py --pairs AUD_CHF,AUD_JPY,AUD_USD,CAD_JPY,CHF_JPY,EUR_AUD,EUR_JPY,EUR_USD,GBP_AUD,GBP_JPY,GBP_NZD,GBP_USD,NZD_CHF,NZD_USD,USD_JPY --granularities S5,M5 --price BA --days 120 --output-dir logs/replay/oanda_history`
- Live permission: remains blocked until forecast sample coverage graduates from `UNDER_SAMPLED` and replay rules become daily-stable/live-grade.

## Profit Capture Repair

- Status: `HISTORICAL_DIAGNOSTIC_ONLY`
- Actual loss-close PL JPY: `-8677.383`
- Counterfactual profit-capture PL JPY: `-8231.343`
- Counterfactual profit-capture delta JPY: `446.04`
- Production-gate replay triggered: `1`
- Production-gate replay delta JPY: `466.2`
- Clearance condition: post-repair production-gate replay remains clean; historical pre-repair misses stay diagnostic and must not be used as the clearance condition
- Verify: `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80`

| Repair Trade | Pair | Side | Exit | Trigger UTC | Repair pips | Noise floor | Repair JPY | Delta JPY |
|---|---|---|---|---|---:|---:|---:|---:|
| `472792` | `USD_JPY` | `SHORT` | `STOP_LOSS_ORDER` | `2026-06-22T06:45:00+00:00` | `2.0` | `1.6` | `126.0` | `466.2` |

| Trade | Pair | Side | Exit | MFE JPY | TP progress | Counterfactual JPY | Delta JPY | Realized JPY |
|---|---|---|---|---:|---:|---:|---:|---:|
| `472792` | `USD_JPY` | `SHORT` | `STOP_LOSS_ORDER` | `214.2` | `0.6071` | `105.84` | `446.04` | `-340.2` |

## Guardian

- Required: `True`
- Label: `com.quantrabbit.position-guardian`
- Plist: `/Users/tossaki/Library/LaunchAgents/com.quantrabbit.position-guardian.plist` exists=`True`
- Launchd loaded: `True`
- Heartbeat path: `/Users/tossaki/App/QuantRabbit-live/data/position_guardian.json`
- Heartbeat generated: `2026-07-06T01:58:55.051605+00:00`

## Directional Inversion Counterfactuals

| Trade | Owner | Pair | Actual | Opposite | Actual UPL JPY | Opposite gross JPY | Clears 5% minimum | Replay status | Evidence status |
|---|---|---|---|---|---:|---:|---|---|---|
| `472987` | `operator_manual` | `EUR_USD` | `SHORT` | `LONG` | `-12061.698` | `12061.698` | `False` | `CONTRARIAN_REPLAY_NO_PAIR_SUPPORT` | `MISSING_REPEATED_SPREAD_INCLUDED_EVIDENCE` |

- Counterfactuals are gross sign-flips of current broker-truth unrealized P/L; they are repair evidence, not live inversion permission.

## Top Intent Blockers

- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`: `82`
- `SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH`: `80`
- `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`: `73`
- `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION`: `71`
- `LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT`: `38`
- `EXHAUSTION_RANGE_CHASE`: `37`
- `RANGE_ROTATION_BROADER_LOCATION_CHASE`: `27`
- `SPREAD_TOO_WIDE`: `27`
- `RANGE_FORECAST_REQUIRES_RANGE_ROTATION`: `26`
- `STRATEGY_NOT_ELIGIBLE`: `23`
- `STRATEGY_PROFILE_MISSING`: `23`
- `FORECAST_WATCH_ONLY`: `21`

## Guardian Recovery Candidates

- none

## Repair Frontier Blockers After Support

| Blocker | Lanes | Reward JPY | Examples |
|---|---:|---:|---|
| `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` | `2` | `1780.073` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT`, `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` |
| `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE` | `2` | `1780.073` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT`, `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` |
| `EXHAUSTION_RANGE_CHASE` | `2` | `1780.073` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT`, `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` |
| `REWARD_RISK_TOO_LOW` | `2` | `1780.073` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT`, `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` |
| `PATTERN_REVERSAL_CHASE` | `1` | `792.765` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` |

## Repair Frontier

| Lane | Pair | Side | Method | Reward JPY | Remaining blockers after support |
|---|---|---|---|---:|---|
| `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `987.308` | `REWARD_RISK_TOO_LOW, EXHAUSTION_RANGE_CHASE, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` |
| `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` | `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `792.765` | `REWARD_RISK_TOO_LOW, PATTERN_REVERSAL_CHASE, EXHAUSTION_RANGE_CHASE, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` |

## OANDA Audit-Only Local TP Proof Required

- Local OANDA history coverage: `PRICE_TRUTH_FETCH_REQUIRED`; missing S5:EUR_JPY,EUR_USD,GBP_JPY; M5:EUR_JPY,EUR_USD,GBP_JPY.
- These candidates still lack live-grade replay or local TP proof. Escape condition: exact pair/side/method TAKE_PROFIT_ORDER proof with positive expectancy, zero TP losses, and positive Wilson-stressed expectancy, or exact OANDA HARVEST vehicle promotion through current-risk / normal-cap 5% firepower scaling.

| Lane | Pair | Side | Method | Scope | Vehicle | Replay evidence | Historical clears local proof | Remaining blockers after guardian |
|---|---|---|---|---|---|---|---|---|
| `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` | `EUR_USD` | `SHORT` | `TREND_CONTINUATION` | `PAIR_SIDE_METHOD` | `EUR_USD|SHORT|pullback_continuation|tp1.25_sl1` | `status=HIGH_PRECISION_VALIDATED n=13 win=0.846154 pf=5.282459 day%=1.068182 5%trades=6` | `False` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED, OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED, CHART_DIRECTION_CONFLICT, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` |
| `trend_trader:EUR_JPY:SHORT:TREND_CONTINUATION` | `EUR_JPY` | `SHORT` | `TREND_CONTINUATION` | `MISSING_METHOD_SCOPE` | `EUR_JPY|SHORT|trend_continuation|tp1_sl1` | `status=HIGH_PRECISION_VALIDATED n=14 win=0.785714 pf=2.946562 day%=1.342024 5%trades=9` | `False` | `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, REWARD_RISK_TOO_LOW, OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` |
| `range_trader:GBP_JPY:SHORT:RANGE_ROTATION` | `GBP_JPY` | `SHORT` | `RANGE_ROTATION` | `MISSING_SCOPED` | `GBP_JPY|SHORT|range_reversion|tp1_sl1` | `status=HIGH_PRECISION_VALIDATED n=21 win=0.857143 pf=6.22137 day%=0.990431 5%trades=10` | `False` | `RANGE_COUNTERTREND_RR_TOO_LOW, OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` |

## RANGE Forecast Superseded Repair Lanes

| Lane | Pair | Side | Method | Reward JPY | Range counterpart |
|---|---|---|---|---:|---|
| `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT` | `AUD_JPY` | `SHORT` | `BREAKOUT_FAILURE` | `1011.0` | `range_trader:AUD_JPY:SHORT:RANGE_ROTATION` |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `810.598` | `range_trader:EUR_USD:SHORT:RANGE_ROTATION` |
| `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE` | `AUD_JPY` | `SHORT` | `BREAKOUT_FAILURE` | `678.0` | `range_trader:AUD_JPY:SHORT:RANGE_ROTATION` |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `551.207` | `range_trader:EUR_USD:SHORT:RANGE_ROTATION` |

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
- `REVIEW_GUARDIAN_RECEIPT_OPERATOR_REVIEW`: `sed -n '1,200p' docs/guardian_receipt_operator_review_report.md` — operator review artifact exists but does not clear normal routing; inspect the decision row before any fresh-entry routing
- `MONITOR_POST_REPAIR_TP_PROGRESS_EVIDENCE`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — post-repair production-gate replay is clean; keep monitoring without treating pre-repair misses as a live blocker
- `REFRESH_EVIDENCE_PACKET`: `PYTHONPATH=src python3 -m quant_rabbit.cli cycle-refresh --daily-risk-pct 10` — refresh forecasts, intents, sidecar audits, and route from current broker truth
- `READ_ACCEPTANCE_BLOCKERS`: `sed -n '1,220p' docs/profitability_acceptance_report.md` — inspect red/green acceptance invariants before increasing turnover
- `FOLLOW_ACCEPTANCE_REPAIR_PLAN`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — use profitability_acceptance.repair_plan clearance conditions; rerunning acceptance alone will loop until those proof metrics change
- `VERIFY_TP_PROGRESS_REPLAY_REPAIR`: `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80` — prove the OANDA candle replay TP-progress miss has cleared at the required coverage before treating high-turnover profit capture as repaired
- `WORK_REPAIR_FRONTIER_REMAINING_BLOCKERS`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — guardian/global repairs are not enough; top remaining repair-frontier blocker is GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED across 2 lane(s)
- `MINE_LOCAL_TP_PROOF_FOR_OANDA_AUDIT_ONLY`: `PYTHONPATH=src python3 scripts/oanda_history_fetch.py --pairs EUR_JPY,EUR_USD,GBP_JPY --granularities S5,M5 --price BA --days 120 --output-dir logs/replay/oanda_history` — OANDA campaign firepower is audit-only for these lanes; fetch only missing spread-included candle truth before treating the candidate as improved evidence
- `VALIDATE_OANDA_AUDIT_ONLY_BIDASK_REPLAY`: `PYTHONPATH=src python3 scripts/oanda_history_replay_validate.py --history-dir logs/replay/oanda_history --granularity S5` — score forecast_history against local OANDA S5 bid/ask candles with spread included; this proves whether the prediction side actually made money on historical candles
- `MINE_OANDA_AUDIT_ONLY_CAMPAIGN_FIREPOWER`: `PYTHONPATH=src python3 scripts/oanda_universal_rotation_miner.py --history-root logs/replay/oanda_history --history-glob '*_M5_BA_*.jsonl' --pairs EUR_JPY,EUR_USD,GBP_JPY` — rerun the multi-month M5 bid/ask candle miner for the audit-only pairs so improved range/failed-break/pullback vehicles are tested before reinsertion
- `PACKAGE_OANDA_AUDIT_ONLY_FIREPOWER_RULES_AFTER_REVIEW`: `PYTHONPATH=src python3 scripts/package_oanda_universal_rotation_rules.py` — packages mined OANDA replay rows into the tracked runtime rule artifact; Codex may run this after reviewing mining evidence, then test, commit, and sync before live runtime uses the new evidence
- `RERUN_INTENTS_AFTER_OANDA_AUDIT_ONLY_REPLAY`: `PYTHONPATH=src python3 -m quant_rabbit.cli cycle-refresh --daily-risk-pct 10` — rerun the full evidence/intents/acceptance loop after replay mining so improved evidence is revalidated instead of leaving the old blocked packet in place
- `WORK_TARGET_FIREPOWER_BLOCKERS`: `PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance` — firepower audit estimates enough turnover for the 5% floor, but live permission still depends on clearing acceptance, guardian, and lane blockers

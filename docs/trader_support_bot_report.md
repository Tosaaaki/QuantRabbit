# Trader Support Bot Report

- Generated at UTC: `2026-06-26T06:42:36.632482+00:00`
- Status: `SUPPORT_BLOCKED`
- Read only: `True`
- Live side effects: `0`

## Support Gates

| Gate | Value |
|---|---|
| Fresh entry send allowed | `False` |
| Repair basket send allowed | `False` |
| Guardian active | `True` source=`launchd+heartbeat` |
| Guardian heartbeat fresh | `True` age=`119.067`s |
| LIVE_READY lanes | `0` / `82` |
| Order intents freshness | `ORDER_INTENTS_ALIGNED_WITH_BROKER_SNAPSHOT` staleness=`-9.631`s |
| Repair LIVE_READY lanes | `0` |
| Repair lanes after guardian recovery | `0` |
| Repair frontier lanes | `2` |
| RANGE forecast superseded repair lanes | `4` |
| OANDA audit-only local TP proof required | `0` |
| RANGE forecast missing counterpart lanes | `0` |
| Repair frontier clear after support | `0` |
| Repair frontier blocked after support | `2` |
| Global unlock frontier lanes | `0` |
| Profit-capture misses | `14` gap=`5481.904` JPY counterfactual_delta=`18725.905` JPY repair_replay_triggered=`13` repair_delta=`18772.146` JPY |
| Current profit-capture positions | bankable=`0` watch=`0` blocked=`0` |
| Open positions | `0` unknown_owner=`0` |
| Open trader positions | `0` upl=`0.0` JPY |
| Directional inversion 5% counterfactuals | `0` |
| Target remaining | `17394.49` JPY |
| Firepower 5% audit estimate | `True` best=`evidence_queue` |
| Firepower 5% operational reachable | `False` blockers=`['SELF_IMPROVEMENT_P0_PRESENT', 'NO_LIVE_READY_LANES', 'PROFITABILITY_ACCEPTANCE_BLOCKED', 'FRESH_ENTRY_SEND_NOT_ALLOWED']` |

## Blockers

- `P0` `SELF_IMPROVEMENT_P0_PRESENT`: self-improvement audit has 1 P0 finding(s)
- `P1` `NO_LIVE_READY_LANES`: daily target is open but no lane is LIVE_READY
- `P0` `PROFITABILITY_ACCEPTANCE_BLOCKED`: profitability acceptance status is PROFITABILITY_ACCEPTANCE_BLOCKED

## Repair Requests

| Code | Status | Source | Verify |
|---|---|---|---|
| `REVIEW_CLOSE_GATE_EVIDENCE_FAILURES` | `HISTORICAL_ACCEPTANCE_WINDOW_ACTIVE` | `LOSS_CLOSE_GATE_EVIDENCE_MISSING`, `RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK` | `PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit` |
| `REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY` | `RESIDUAL_GROUPS_ALREADY_BLOCKED_WAITING_FOR_REPLAY` | `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE` | `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80` |
| `REPAIR_FRONTIER_LANE_BLOCKER` | `FRONTIER_PROTECTIVE_GUARDRAIL_ACTIVE` | `BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE` | `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot`, `PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts` |

- Automation contract: Codex may edit code/tests/docs, run tests, commit, and sync live; orders, cancels, closes, and launchd changes require explicit approval or the existing gateway path.

## Profit Capture Repair

- Status: `HISTORICAL_DIAGNOSTIC_ONLY`
- Actual loss-close PL JPY: `-35276.446`
- Counterfactual profit-capture PL JPY: `-16550.541`
- Counterfactual profit-capture delta JPY: `18725.905`
- Production-gate replay triggered: `13`
- Production-gate replay delta JPY: `18772.146`
- Clearance condition: post-repair production-gate replay remains clean; historical pre-repair misses stay diagnostic and must not be used as the clearance condition
- Verify: `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80`

| Repair Trade | Pair | Side | Exit | Trigger UTC | Repair pips | Noise floor | Repair JPY | Delta JPY |
|---|---|---|---|---|---:|---:|---:|---:|
| `471240` | `EUR_USD` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `2026-05-14T21:36:00+00:00` | `3.7` | `3.0` | `899.163` | `6166.709` |
| `471232` | `EUR_USD` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `2026-05-14T17:39:00+00:00` | `3.0` | `1.7` | `340.224` | `3647.646` |
| `471414` | `EUR_USD` | `SHORT` | `MARKET_ORDER_TRADE_CLOSE` | `2026-05-21T15:08:00+00:00` | `4.1` | `3.55` | `199.274` | `2841.38` |
| `471979` | `AUD_CHF` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `2026-06-04T06:54:00+00:00` | `5.5` | `1.9` | `396.681` | `1612.085` |
| `472222` | `GBP_CHF` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `2026-06-11T17:34:00+00:00` | `7.5` | `2.8` | `297.511` | `1279.305` |

| Trade | Pair | Side | Exit | MFE JPY | TP progress | Counterfactual JPY | Delta JPY | Realized JPY |
|---|---|---|---|---:|---:|---:|---:|---:|
| `471240` | `EUR_USD` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `899.163` | `0.4066` | `663.437` | `5930.983` | `-5267.546` |
| `471232` | `EUR_USD` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `374.246` | `0.3626` | `309.604` | `3617.026` | `-3307.422` |
| `471414` | `EUR_USD` | `SHORT` | `MARKET_ORDER_TRADE_CLOSE` | `199.274` | `0.5125` | `116.648` | `2758.754` | `-2642.106` |
| `471979` | `AUD_CHF` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `454.38` | `0.3913` | `348.358` | `1563.763` | `-1215.405` |
| `472222` | `GBP_CHF` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `297.511` | `0.3138` | `284.42` | `1266.214` | `-981.794` |

## Guardian

- Required: `True`
- Label: `com.quantrabbit.position-guardian`
- Plist: `/Users/tossaki/Library/LaunchAgents/com.quantrabbit.position-guardian.plist` exists=`True`
- Launchd loaded: `True`
- Heartbeat path: `/Users/tossaki/App/QuantRabbit-live/data/position_guardian.json`
- Heartbeat generated: `2026-06-26T06:40:37.565273+00:00`

## Directional Inversion Counterfactuals

- none

## Top Intent Blockers

- `SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH`: `80`
- `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION`: `74`
- `RANGE_ROTATION_BROADER_LOCATION_CHASE`: `40`
- `FORECAST_WATCH_ONLY`: `31`
- `EXHAUSTION_RANGE_CHASE`: `30`
- `SPREAD_TOO_WIDE`: `27`
- `STRATEGY_PROFILE_MISSING`: `24`
- `RANGE_FORECAST_REQUIRES_RANGE_ROTATION`: `24`
- `STRATEGY_NOT_ELIGIBLE`: `22`
- `FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE`: `18`
- `RANGE_COUNTERTREND_RR_TOO_LOW`: `15`
- `LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT`: `14`

## Guardian Recovery Candidates

- none

## Repair Frontier Blockers After Support

| Blocker | Lanes | Reward JPY | Examples |
|---|---:|---:|---|
| `BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE` | `1` | `2041.444` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` |
| `EXHAUSTION_RANGE_CHASE` | `1` | `2041.444` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` |
| `PATTERN_REVERSAL_CHASE` | `1` | `2041.444` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` |
| `REWARD_RISK_TOO_LOW` | `1` | `476.337` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` |

## Repair Frontier

| Lane | Pair | Side | Method | Reward JPY | Remaining blockers after support |
|---|---|---|---|---:|---|
| `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` | `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `2041.444` | `BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE, PATTERN_REVERSAL_CHASE, EXHAUSTION_RANGE_CHASE` |
| `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `476.337` | `REWARD_RISK_TOO_LOW` |

## OANDA Audit-Only Local TP Proof Required

- none

## RANGE Forecast Superseded Repair Lanes

| Lane | Pair | Side | Method | Reward JPY | Range counterpart |
|---|---|---|---|---:|---|
| `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT` | `AUD_JPY` | `SHORT` | `BREAKOUT_FAILURE` | `1451.0` | `range_trader:AUD_JPY:SHORT:RANGE_ROTATION` |
| `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE` | `AUD_JPY` | `SHORT` | `BREAKOUT_FAILURE` | `1011.0` | `range_trader:AUD_JPY:SHORT:RANGE_ROTATION` |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `686.962` | `range_trader:EUR_USD:SHORT:RANGE_ROTATION` |
| `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` | `EUR_USD` | `SHORT` | `BREAKOUT_FAILURE` | `427.731` | `range_trader:EUR_USD:SHORT:RANGE_ROTATION` |

## RANGE Forecast Missing Counterpart Repair Lanes

- none

## Global Unlock Frontier

- none

## Target Firepower Evidence

- Status: `VERIFIED_MINIMUM_5_ROUTE_ESTIMATED`
- Best bucket: `evidence_queue`
- 5% estimated reachable: `True`
- 10% estimated reachable: `True`
- Audit only, no live permission grant: `true`
- `high_precision` return/day=`5.299476` trades_needed_5pct=`12.0` vehicles=`['EUR_JPY|SHORT|trend_continuation|tp1.25_sl1', 'EUR_JPY|SHORT|trend_continuation|tp1_sl1', 'EUR_JPY|SHORT|range_reversion|tp1.25_sl1']`
- `evidence_queue` return/day=`11.156301` trades_needed_5pct=`8.0` vehicles=`['EUR_JPY|SHORT|range_reversion|tp1.25_sl1', 'EUR_JPY|SHORT|range_reversion|tp1_sl0.75', 'EUR_JPY|SHORT|range_reversion|tp0.75_sl0.75']`

## Acceptance Repair Plan

- Loop breaker: Rerunning profitability-acceptance alone cannot clear these P0s; the listed proof condition must change first. Evidence-collection items also require fresh replay/mining output before they can graduate into live-grade turnover.

| Code | Clearance condition | Verify |
|---|---|---|
| `SELF_IMPROVEMENT_P0_PRESENT` | self_improvement_audit has zero P0 findings, or the only remaining discipline finding has been demoted by verified clean gateway close recovery | `PYTHONPATH=src python3 -m quant_rabbit.cli self-improvement-audit` |
| `NEGATIVE_EXPECTANCY_ACTIVE` | capture_economics.status is no longer NEGATIVE_EXPECTANCY, or entries are limited to exact TP-proven repair/harvest shapes with positive expectancy evidence | `PYTHONPATH=src python3 -m quant_rabbit.cli capture-economics` |
| `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE` | no TP-proven segment remains net-damaged by MARKET_ORDER_TRADE_CLOSE leakage; preserve broker TP and guardian capture instead of scaling market-close loss paths | `PYTHONPATH=src python3 -m quant_rabbit.cli capture-economics` |
| `RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK` | recent_leak_loss_closes is zero inside the 7-day acceptance window, or each loss-side market close has contained-risk timing plus durable gateway/GPT close proof | `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80` |
| `LOSS_CLOSE_GATE_EVIDENCE_MISSING` | every recent GPT loss-side market close has durable close_gate_evidence in verification_observations, or the missing-evidence closes age out of the 7-day acceptance window without new leaks | `PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit` |
| `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE` | month-scale production-gate replay is non-negative, or the top residual pair/side/method groups are removed by close-gate, TP-capture, or entry-selection changes before turnover is scaled | `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80` |

## Acceptance Evidence Collection

- none

## Current Profit Capture

- none

## Open Trader Positions

- none

## Operator Actions

- `REFRESH_SUPPORT_PANEL`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — read-only status refresh for trader operations
- `MONITOR_POST_REPAIR_TP_PROGRESS_EVIDENCE`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — post-repair production-gate replay is clean; keep monitoring without treating pre-repair misses as a live blocker
- `REFRESH_EVIDENCE_PACKET`: `PYTHONPATH=src python3 -m quant_rabbit.cli cycle-refresh --daily-risk-pct 10` — refresh forecasts, intents, sidecar audits, and route from current broker truth
- `READ_ACCEPTANCE_BLOCKERS`: `sed -n '1,220p' docs/profitability_acceptance_report.md` — inspect red/green acceptance invariants before increasing turnover
- `FOLLOW_ACCEPTANCE_REPAIR_PLAN`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — use profitability_acceptance.repair_plan clearance conditions; rerunning acceptance alone will loop until those proof metrics change
- `VERIFY_CLOSE_GATE_EVIDENCE`: `PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit` — confirm future GPT loss-side closes have durable PASS close_gate_evidence
- `RECHECK_LOSS_CLOSE_LEAK_WINDOW`: `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80` — verify the 7-day loss-close leak window is shrinking before adding turnover
- `VERIFY_TP_PROGRESS_REPLAY_REPAIR`: `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80` — prove the OANDA candle replay TP-progress miss has cleared at the required coverage before treating high-turnover profit capture as repaired
- `WORK_REPAIR_FRONTIER_REMAINING_BLOCKERS`: `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot` — guardian/global repairs are not enough; top remaining repair-frontier blocker is BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE across 1 lane(s)
- `WORK_TARGET_FIREPOWER_BLOCKERS`: `PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance` — firepower audit estimates enough turnover for the 5% floor, but live permission still depends on clearing acceptance, guardian, and lane blockers

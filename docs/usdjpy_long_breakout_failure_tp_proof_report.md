# USD_JPY LONG BREAKOUT_FAILURE TP Proof

Generated: `2026-07-04T12:44:28Z`

Mode: read-only evidence. No orders, cancels, closes, SL/TP changes, or execution flag changes.

## Verdict

`PARTIAL_REPLAY_PROOF_NOT_LIVE_READY`

The S5 bid/ask replay aggregate for `failure_trader:USD_JPY:LONG:BREAKOUT_FAILURE:LIMIT` is positive for the exact TP10/SL7 LIMIT harvest shape, but it is not enough to repair the live strategy profile or mark the lane A/S `LIVE_READY`.

Why it stops:

- Current `order_intents.json` generated `2026-07-04T12:43:40.432860+00:00` does not emit `failure_trader:USD_JPY:LONG:BREAKOUT_FAILURE:LIMIT`.
- `capture_economics.json` has no `USD_JPY|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER` method scope.
- The replay rule references `logs/reports/forecast_improvement/oanda_history_replay_validate_20260703T145218Z.json`, but that raw audit report is not present locally.
- The tracked rule artifact reports average MAE and worst daily realized P/L, but not exact per-sample max adverse or equity-curve drawdown.
- `profitability_acceptance` is fresh against current inputs, but still `PROFITABILITY_ACCEPTANCE_BLOCKED`.

## Exact Shape

| Field | Value |
| --- | --- |
| lane_id | `failure_trader:USD_JPY:LONG:BREAKOUT_FAILURE:LIMIT` |
| pair | `USD_JPY` |
| side | `LONG` |
| strategy | `BREAKOUT_FAILURE` |
| entry_type | `LIMIT` |
| exit_method | `TAKE_PROFIT_ORDER` |
| TP/SL | TP10 / SL7 |
| TP mode | `ATTACHED_TECHNICAL_TP` |
| TP intent | `HARVEST` |
| chase permission | with-move chase not allowed |

## Replay Evidence

Source: `src/quant_rabbit/bidask_replay_precision_rules.json`, `daily_stable_contrarian_edge_rules[0]`.

Rule: `USD_JPY_DOWN_H61_240m_CLT0p50_FADE_TO_UP_S5_BIDASK_CONTRARIAN_HARVEST_TP10_SL7`

| Metric | Value |
| --- | ---: |
| adoption_status | `LIVE_GRADE_DAILY_STABLE` |
| samples | 91 |
| active_days | 8 |
| first_day / last_day | 2026-06-05 / 2026-07-03 |
| optimized_avg_realized_pips | 4.5363 |
| optimized_profit_factor | 3.2194 |
| optimized_win_rate | 0.6703 |
| directional_hit_rate | 0.7473 |
| positive_day_rate | 0.75 |
| optimized TP / SL | 10.0 / 7.0 pips |
| avg_MFE / avg_MAE | 18.3385 / 13.0088 pips |
| worst_daily_realized_pips | -22.0 |
| max_daily_sample_share | 0.5824 |

This is spread-included at the rule level because the supporting artifact is the S5 bid/ask replay precision rule set. It is not a mid-price-only proof.

## Current Fresh Inputs

| Artifact | Current state |
| --- | --- |
| broker snapshot | fetched `2026-07-04T12:43:37.253681+00:00` |
| capture_economics | generated `2026-07-04T12:43:27.668063+00:00`, `NEGATIVE_EXPECTANCY` |
| order_intents | generated `2026-07-04T12:43:40.432860+00:00`, 73 results, 0 `LIVE_READY` |
| target lane in current order_intents | absent |
| profitability_acceptance | generated `2026-07-04T12:43:58.880706+00:00`, fresh but blocked |
| trader_support_bot | `PROFITABILITY_ACCEPTANCE_ALIGNED_WITH_INPUTS`, `live_ready_lanes=0` |

The old A/S candidate board was generated `2026-07-04T11:53:12Z` from live-runtime order intents generated `2026-07-03T20:16:34.043087+00:00`; it is evidence history, not a current routing input.

## Strategy Profile Decision

No strategy profile repair was made.

Reason: the proof does not satisfy the live repair contract for exact local `USD_JPY|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER` capture scope, does not include the raw replay audit file, and the fresh order-intent rebuild no longer emits the target lane. Updating `USD_JPY LONG` or `BREAKOUT_FAILURE` here would broaden a stale candidate into live eligibility without the required evidence.

## A/S Readiness

| Check | Result |
| --- | --- |
| A/S grade | No |
| LIVE_READY | No |
| RiskEngine dry-run | Not run for target lane because current order intents do not contain the lane |
| LiveOrderGateway eligibility | Not eligible; no current intent and hard blockers remain |
| Fresh GPT TRADE/ADD receipt | Missing |
| Guardian/operator review | Blocks normal routing |
| Profitability acceptance | Fresh but blocked |
| Negative expectancy override | Not allowed |
| EUR_USD manual conflict | Manual EUR_USD remains excluded from system P/L and occupancy |

## Remaining Clearing Conditions

- Missing evidence: produce exact `USD_JPY|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER` local capture proof with positive expectancy and required Wilson-stressed support.
- Missing evidence: restore or regenerate the raw S5 replay audit with per-sample path, max adverse, and drawdown series.
- Stale artifact: rebuild current order intents and require this exact lane to appear with TP10/SL7 LIMIT harvest geometry.
- Strategy profile blocker: repair only a method/shape-scoped profile after the exact evidence above exists; do not unlock broad `USD_JPY LONG` or broad `BREAKOUT_FAILURE`.
- Expectancy blocker: clear current `PROFITABILITY_ACCEPTANCE_BLOCKED` causes without weakening gates.
- Guardian/operator-review blocker: normal routing must be allowed before any fresh entry path can be live.
- True safety blocker: fresh GPT-5.5 TRADE/ADD receipt, RiskEngine pass, and LiveOrderGateway pass are still mandatory.

## Safety

No broker-side action was performed. EUR_USD trade `472987` remains `OPERATOR_MANUAL` / `KEEP`; no auto close, SL attach, TP modification, same-theme add, system P/L counting, or system occupancy counting was authorized.

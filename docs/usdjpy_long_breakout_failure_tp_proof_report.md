# USD_JPY LONG BREAKOUT_FAILURE TP Proof

Generated: `2026-07-05T08:23:46Z`

Mode: read-only evidence. No orders, cancels, closes, SL/TP changes, execution flag changes, or broker-state modifications.

## Verdict

`TP_PROOF_REJECTED_CURRENT_REPLAY_NEGATIVE_UNDERSAMPLED`

The tracked packaged rule is positive, but its declared raw audit report is absent and the fresh USD_JPY-only bid/ask replay from current forecast_history does not confirm the exact TP10/SL7 DOWN->UP vehicle: it is under-sampled, has negative average realized pips, PF below breakeven, and positive-day rate below the daily-stability floor. The fresh order-intent rebuild also does not emit the lane.

The candidate did not become A/S or `LIVE_READY`, and no strategy-profile repair was applied.

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

## Current Inputs

| Artifact | Current state |
| --- | --- |
| broker snapshot | fetched `2026-07-05T08:21:41.843707+00:00` |
| capture_economics | generated `2026-07-05T08:21:32.133082+00:00`, `NEGATIVE_EXPECTANCY` |
| order_intents | generated `2026-07-05T08:21:44.548031+00:00`, 73 results, 0 `LIVE_READY` |
| target lane in current order_intents | `False` |
| profitability_acceptance | generated `2026-07-05T08:22:01.555344+00:00`, `PROFITABILITY_ACCEPTANCE_BLOCKED` |
| profitability freshness | `PROFITABILITY_ACCEPTANCE_ALIGNED_WITH_INPUTS`, stale=`False` |
| trader_support_bot | generated `2026-07-05T08:22:20.107962+00:00`, blockers remain |

The old A/S candidate board was generated `2026-07-04T11:53:12Z` from order intents `2026-07-03T20:16:34.043087+00:00`; it is stale candidate history, not current routing input.

## Stale Candidate Snapshot

| Metric | Value |
| --- | ---: |
| entry / TP / SL | 161.355 / 161.455 / 161.285 |
| units | 4000 |
| expected RR | 1.4286 |
| risk JPY | 280.0 |
| reward JPY | 400.0 |
| spread pips | 0.8 |
| forecast direction / confidence | DOWN / 0.1714 |
| market support ok | False |

## Replay Evidence

Packaged historical rule:

| Metric | Value |
| --- | ---: |
| rule | `USD_JPY_DOWN_H61_240m_CLT0p50_FADE_TO_UP_S5_BIDASK_CONTRARIAN_HARVEST_TP10_SL7` |
| samples | 91 |
| active_days | 8 |
| optimized_avg_realized_pips | 4.5363 |
| optimized_profit_factor | 3.2194 |
| directional_hit_rate | 0.7473 |
| positive_day_rate | 0.75 |
| avg_MFE / avg_MAE | 18.3385 / 13.0088 |
| worst_daily_realized_pips | -22.0 |
| raw audit present | False |

Fresh USD_JPY-only current replay:

| Metric | Broad DOWN->UP TP10/SL7 | Strict current bucket |
| --- | ---: | ---: |
| n | 9 | 7 |
| active_days | 2 | 2 |
| avg_realized_pips | -0.0111111111 | -0.4428571429 |
| profit_factor | 0.9971428571 | 0.8892857143 |
| win_rate | 0.4444444444 | 0.4285714286 |
| hit_rate | 0.5555555556 | 0.4285714286 |
| positive_day_rate | 0.5 | 0.5 |
| avg_MFE / avg_MAE | 7.8111111111 / 10.6444444444 | 5.9714285714 / 11.4142857143 |
| worst_daily_realized_pips | -1.0 | -9.1 |

Spread is included in the fresh replay because `oanda_history_replay_validate.py` uses S5 bid/ask candles: UP enters at ask and exits at bid.

## Strategy Profile Decision

No strategy profile repair was made. USD_JPY LONG remains `BLOCK_UNTIL_NEW_EVIDENCE`.

Reason: not applied because exact TP proof is invalid against current replay, local capture scope is missing, and the fresh target lane is absent. Broad `USD_JPY LONG` and broad `BREAKOUT_FAILURE` remain locked; no with-move chase was enabled.

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

## Remaining Blockers

- `CURRENT_REPLAY_UNDER_SAMPLED`
- `CURRENT_REPLAY_ACTIVE_DAYS_THIN`
- `CURRENT_REPLAY_NEGATIVE_EXPECTANCY`
- `CURRENT_REPLAY_PF_BELOW_BREAKEVEN`
- `CURRENT_REPLAY_POSITIVE_DAY_RATE_LOW`
- `MISSING_LOCAL_TP_SCOPE`
- `FRESH_TARGET_LANE_ABSENT`
- `STRATEGY_PROFILE_BLOCK_UNTIL_NEW_EVIDENCE`
- `SELF_IMPROVEMENT_P0_PRESENT`
- `NEGATIVE_EXPECTANCY_ACTIVE`
- `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE`
- `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`
- `GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING`
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING`
- `NO_LIVE_READY_LANES`
- `NO_FRESH_GPT_TRADE_ADD_RECEIPT`

See `remaining_clearing_conditions` in the JSON proof for clearing condition, file/module, and evidence requirement per blocker.

## Capital And Manual Protection

- current_equity_raw equals broker NAV: `270740.4982` / `270740.4982`.
- funding_adjusted_equity excludes the 100,000 JPY deposit: `170740.4982`.
- rolling_30d_multiplier_funding_adjusted is authoritative: `0.995949`; raw multiplier `1.57926` is context only.
- EUR_USD `472987` present=`True`, classification=`OPERATOR_MANUAL`, management_intent=`KEEP`.
- manual EUR_USD system_pl_counted=`False`, same_theme_auto_add_allowed=`False`, auto_sl_attach_allowed=`False`, auto_tp_modify_allowed=`False`.

## Safety

No broker-side action was performed. No order was placed, cancelled, or modified; no position was closed; no SL/TP was attached or changed; no execution flag was enabled.

# EUR_USD SHORT BREAKOUT_FAILURE LIMIT S5 Bid/Ask Replay

Status: `LIMIT_S5_BIDASK_REPLAY_PASSED_STILL_BLOCKED`

This is read-only evidence. It is not live permission, SCOUT permission, gateway permission, or operator clearance. No order, cancel, close, TP/SL change, launchd change, or broker-state mutation was performed.

## Target

- Shape: `EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST`
- Vehicle: `LIMIT_ORDER` only
- TP type: `ATTACHED_TECHNICAL_TP` / `TAKE_PROFIT_ORDER`
- Replay: local OANDA `S5` bid/ask candles
- SHORT side model: entry checked on `bid`, TP exit checked on `ask`
- Source: `/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history/20260703T120929Z/EUR_USD/EUR_USD_S5_BA_20260305T120929Z_20260703T120929Z.jsonl.gz`

## Result

The exact LIMIT-only replay passed on the 4 observed LIMIT samples, with a timestamp-alignment caveat:

| Metric | Value |
|---|---:|
| Replay sample count | 4 |
| Replay wins | 4 |
| Replay losses | 0 |
| Observed LIMIT net JPY | 3255.0938 |
| Net expectancy after bid/ask | 813.7734 JPY/trade |
| S5 spread min/max | 1.4 / 10.0 pips |

The broad 20/0 TAKE_PROFIT_ORDER proof packet is not used as LIMIT proof because it includes MARKET and STOP vehicles. The 4/0 LIMIT result is positive, but it remains under-sampled and blocked from live-grade promotion.

## Sample Replay

| Trade | Source | Entry order | TP order | S5 entry touch | S5 TP touch | Net JPY |
|---|---|---:|---:|---|---|---:|
| `472732` | execution ledger | 1.14486 | 1.14406 | 2026-06-19 08:01:45Z | 2026-06-22 13:01:30Z | 813.7706 |
| `469278` | legacy history | 1.17656 | 1.17590 | 2026-04-21 08:33:10Z | 2026-04-21 09:09:00Z | 314.6844 |
| `469427` | legacy history | 1.17325 | 1.17240 | 2026-04-22 14:39:05Z | 2026-04-22 14:53:20Z | 270.2989 |
| `469898` | legacy history | 1.16790 | 1.16645 | 2026-04-29 23:32:10Z | 2026-04-30 02:45:15Z | 1856.3399 |

The exact recorded entry/exit timestamp candle does not touch the order price in the local S5 aggregation. The replay therefore records the first S5 touch after the LIMIT arming/reference time and the first S5 ask touch of the TP after entry. That timing gap is retained in the JSON and should not be hidden when this proof is consumed. For `472732`, OANDA transaction `fullPrice` still confirms bid/ask cost-inclusive entry and TP fill, but generated proof import should preserve the S5 lag instead of claiming exact same-candle reconstruction.

## Exclusions

- MARKET_ORDER samples are excluded from LIMIT proof.
- STOP_ORDER samples are excluded from LIMIT proof.
- MARKET_ORDER_TRADE_CLOSE losses are excluded from HARVEST proof.
- The excluded market-close leak remains visible: `payoff_shape_diagnosis` reports 10 EUR_USD SHORT BREAKOUT_FAILURE market-close losses for -7636.3 JPY.

## Still Blocked

- `LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY`: only 4 exact LIMIT samples.
- `S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION`: touch-order replay passes, but strict same-candle timestamp reconstruction does not.
- `NOT_IN_PROOF_QUEUE`: A/S proof queue remains empty.
- `NEGATIVE_EXPECTANCY_ACTIVE`: global capture economics remains negative.
- `MARKET_CLOSE_LEAK_PRESENT_EXCLUDED`: loss-side market-close leakage is excluded, not repaired.
- `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`: global month-scale residual remains visible.
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`: normal routing still blocked.
- `NO_LIVE_READY_PORTFOLIO` and `NO_FRESH_GATEWAY_PERMISSION`: this replay is not a gateway packet.

## Next Read-Only Actions

1. Canonically reconcile/import legacy LIMIT rows `469278`, `469427`, and `469898` without broker mutation.
2. Mine more exact LIMIT / ATTACHED_TP / HARVEST samples without MARKET or STOP rows.
3. Regenerate payoff, harvest, A/S proof queue, portfolio planner, and goal-loop artifacts read-only.
4. Rerun timing/profitability acceptance so negative expectancy and month-scale residuals stay visible.
5. If a refreshed exact LIMIT replay turns non-positive, classify the vehicle as `NO_SCOUT` / `NO_TRADE`.

## Safety

This artifact does not authorize live order entry, SCOUT collection, gateway routing, cancellation, close, TP/SL modification, launchd load/reload, gate relaxation, 4x deficit lot backsolve, or operator-decision inference.

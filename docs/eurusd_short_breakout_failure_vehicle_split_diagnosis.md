# EUR_USD SHORT BREAKOUT_FAILURE Vehicle Split Diagnosis

Generated: 2026-07-08T01:59:41Z

Status: `LIMIT_VEHICLE_REPLAY_REQUIRED`

This diagnosis is read-only. It created no orders, cancels, closes, TP/SL changes, launchd changes, gateway actions, or live permission.

## Result

The 20 observed attached-TP samples remain positive as a broad broker-fill packet:

| Bucket | Samples | Wins | Losses | Net JPY | Expectancy JPY/trade |
|---|---:|---:|---:|---:|---:|
| All observed TP fills | 20 | 20 | 0 | 12865.8326 | 643.2916 |
| LIMIT_ORDER only | 4 | 4 | 0 | 3255.0938 | 813.7734 |
| MARKET_ORDER only | 9 | 9 | 0 | 3303.5031 | 367.0559 |
| STOP_ORDER only | 7 | 7 | 0 | 6307.2357 | 901.0337 |
| MARKET_ORDER + STOP_ORDER | 16 | 16 | 0 | 9610.7388 | 600.6712 |

The split does not support live promotion. The LIMIT vehicle is positive but has only 4 samples and no exact independent S5 bid/ask replay. The MARKET/STOP vehicle is positive in observed fills, but it is not one vehicle: MARKET rows lack requested entry price/invalidation reconstruction, and STOP rows have separate trigger and breakout timing semantics.

## Direct Answers

1. LIMIT 4 only positive?
   Yes. The LIMIT-only subset is 4/4 positive, net `3255.0938 JPY`, expectancy `813.7734 JPY/trade`.

2. MARKET/STOP 16 separate vehicle has consistent winning shape?
   It has a positive observed outcome, 16/16 and net `9610.7388 JPY`, but not a consistent single vehicle. MARKET and STOP must be split before proof or schema work.

3. LIMIT proof does not mix MARKET/STOP samples?
   Correct. The LIMIT subset contains only trade IDs `472732`, `469278`, `469427`, and `469898`. The broad 20/0 packet is not used as LIMIT proof.

4. If MARKET/STOP is separate, can entry risk/slippage/invalidation be defined now?
   Not from this packet. MARKET rows have `entry_order_price=null`, so requested-price slippage and pre-entry invalidation are not reconstructed. STOP rows need trigger-side replay, forecast/timing support, and invalidation rules.

5. Which vehicle is closer to the 4x HARVEST proof path?
   LIMIT is closer. Current 4x artifacts rank `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` and `:LIMIT` as EVIDENCE_GAP candidates; `:MARKET` is HISTORICAL_ONLY. The contract's narrow repair exception is non-market attached-TP HARVEST and exact-shape replay is LIMIT-only here.

6. Does either path create live permission?
   No. `live_permission_allowed=false`; proof queue and live-ready counts remain zero, and global blockers remain active.

## Why Status Is Replay Required

`VEHICLE_SPLIT_SUPPORTS_LIMIT_PROMOTION` would be too strong because LIMIT has only 4 exact vehicle samples and all samples lack independent exact S5 bid/ask replay.

`VEHICLE_SPLIT_SUPPORTS_MARKET_STOP_REDEFINITION` would also be too strong because MARKET and STOP are not one contract. A new vehicle definition would need separate proof packets and gateway contracts for each.

Therefore the correct next state is `LIMIT_VEHICLE_REPLAY_REQUIRED`, while also keeping `market_stop_replay_required=true` for separate future diagnostics.

## Remaining Blockers

- `LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY`: only 4/20 positive samples are exact LIMIT_ORDER entries.
- `EXACT_LIMIT_S5_BIDASK_REPLAY_MISSING`: every sample still lacks independent exact S5 replay.
- `MARKET_STOP_NOT_SINGLE_VEHICLE`: 16/16 positive is not one execution contract.
- `GLOBAL_NEGATIVE_EXPECTANCY_ACTIVE`: capture economics remains negative.
- `MARKET_CLOSE_LEAK_PRESENT`: target-shape market-close losses remain visible and excluded from TP proof.
- `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`: month-scale residual repair remains a live blocker.
- `NO_LIVE_READY_OR_PROOF_QUEUE_PERMISSION`: current 4x/proof artifacts do not create permission.
- `GUARDIAN_OPERATOR_REVIEW_REQUIRED`: normal routing remains blocked.

## Next Read-Only Actions

1. Attach exact EUR/USD S5 bid/ask replay for `LIMIT/ATTACHED_TECHNICAL_TP/HARVEST`.
2. Mine more LIMIT-only target-shape TAKE_PROFIT_ORDER rows without mixing MARKET or STOP.
3. Split MARKET_ORDER and STOP_ORDER into separate diagnostic packets.
4. Define MARKET requested-entry, max slippage, invalidation, and gateway contract before any replay.
5. Define STOP trigger replay, timing support, invalidation, and stop-chase compliance before any replay.
6. Rerun profitability acceptance, proof queue, and 4x planner read-only after exact replay exists.

## Notion

Notion was searched and fetched read-only. The page `quant-rabbit-profitability-evidence-repair-2026-07-03` corroborates the active blockers: negative expectancy, market-close leakage, month-scale replay residuals, no-live-ready target coverage, and repair-frontier blockers. No Notion page was created, moved, edited, or commented on.

See `data/eurusd_short_breakout_failure_vehicle_split_diagnosis.json` for per-sample vehicle evidence.

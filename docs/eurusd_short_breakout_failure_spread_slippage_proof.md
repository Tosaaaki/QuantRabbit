# EUR_USD SHORT BREAKOUT_FAILURE Spread/Slippage Proof

Generated: 2026-07-08T01:45:52Z

Status: `SPREAD_SLIPPAGE_PROOF_INCOMPLETE`

This packet is read-only. It produced no orders, cancels, closes, TP/SL changes, launchd changes, gateway actions, or live permission.

## Result

The duplicate-free TP proof floor is reached at `20` wins / `0` losses, and all 20 checked samples are positive after observed broker fills or legacy spread-tagged fills. The observed cost-inclusive net is `12865.8326 JPY`, or `643.2916 JPY/trade`.

That is not enough to pass the requested exact vehicle proof. The requested vehicle is `LIMIT/ATTACHED_TECHNICAL_TP/HARVEST`, but only `4/20` checked samples are `LIMIT_ORDER` entries. The other `16/20` are `MARKET_ORDER` or `STOP_ORDER` entries. Also, no exact independent `EUR_USD SHORT BREAKOUT_FAILURE LIMIT HARVEST` S5 bid/ask replay artifact is attached in this workspace.

Therefore the spread/slippage proof remains incomplete, not failed on observed broker P/L.

## Checks

| Check | Result |
|---|---:|
| Duplicate-free samples | `20/20` |
| TP wins / losses | `20 / 0` |
| Observed cost-inclusive passing samples | `20` |
| Observed cost-inclusive failing samples | `0` |
| Exact LIMIT vehicle samples | `4` |
| Non-LIMIT entry samples | `16` |
| Independent exact S5 bid/ask replay attached | `false` |
| Market-close losses mixed into TP proof | `false` |
| Live permission created | `false` |

Observed entry spreads were `0.8 pip` on the execution-ledger samples and `Sp=0.8pip` on the legacy samples. Observed TP-fill exit spreads were `0.8-1.0 pip`. TP fill-vs-TP-price slippage for execution-ledger samples was `0.0` to `+0.3 pip` in the SHORT-favorable direction. Legacy samples carry timestamp, price, units, realized P/L, exit reason, and `Sp=0.8pip`, but not full bid/ask candle replay.

## Why It Is Not Live Grade

- `BIDASK_REPLAY_MISSING_FOR_EXACT_VEHICLE`: no exact EUR/USD S5 BA replay was attached. `logs/replay/oanda_history/latest_summary.json` currently lists `AUD_JPY` and `GBP_USD`, not `EUR_USD`.
- `EXACT_LIMIT_VEHICLE_SAMPLE_MIXED`: 20/0 is broker-TP proof, but not 20 exact LIMIT samples.
- `MARKET_CLOSE_LEAK_PRESENT`: target-shape market-close losses remain excluded from TP proof and still block live-grade promotion.
- `NEGATIVE_EXPECTANCY_ACTIVE`: global capture economics remains negative (`-177.4 JPY/trade`, net `-40616.9 JPY`).
- `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`: month-scale residual remains visible in current profitability/harvest artifacts.
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`: operator review remains blocked with normal routing false.
- `NOT_IN_PROOF_QUEUE`, `NO_LIVE_READY_PORTFOLIO`, and `NO_FRESH_GATEWAY_PERMISSION` remain true.

## Notion

Notion was searched and fetched read-only. The closest page, `quant-rabbit-profitability-evidence-repair-2026-07-03`, corroborates no broker-side actions, negative expectancy, market-close leakage, month-scale blockers, no-live-ready gates, and operator-review blockers. No Notion page was created, moved, edited, or commented on.

## Next Read-Only Actions

1. Attach or fetch EUR/USD S5 BA history for the target window without broker mutation.
2. Replay the exact `LIMIT/ATTACHED_TECHNICAL_TP/HARVEST` geometry only.
3. Keep exact LIMIT samples separate from broader MARKET/STOP attached-TP samples.
4. Regenerate payoff/harvest/A-S proof artifacts only after exact replay exists.
5. Keep negative expectancy, market-close leakage, and month-scale residuals visible.

See `data/eurusd_short_breakout_failure_spread_slippage_proof.json` for the per-sample evidence.

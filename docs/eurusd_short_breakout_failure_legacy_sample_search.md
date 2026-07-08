# EUR_USD SHORT BREAKOUT_FAILURE Legacy Sample Search

Run date: 2026-07-08 JST

## Verdict

Status: `LEGACY_SAMPLES_FOUND_PROOF_FLOOR_REACHED`

Read-only search found 3 duplicate-free legacy broker TP samples for `EUR_USD|SHORT|BREAKOUT_FAILURE` or a clearly equivalent failed-shelf/failed-retest family. Current proof was 17 wins / 0 losses with proof floor 20. Adding the 3 accepted legacy rows gives 20 wins / 0 losses for the TP proof-floor count.

This does not create live permission. The target still carries market-close leakage and the system-level capture economics are still negative.

## Safety

- Read-only only.
- No live orders.
- No cancels, closes, TP/SL edits, launchd changes, broker-state mutations, gate loosening, proof-floor lowering, 4x lot backsolve, or secret reads/prints.
- Notion was searched/fetched read-only. No Notion pages were created, moved, edited, or commented on.

## Accepted Samples

| Trade | Exit TX | Time UTC | Units | Entry | Exit | P/L JPY | Exit reason | Why accepted |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `469278` | `469281` | 2026-04-21 09:08:35 | 3000 | 1.17656 | 1.17590 | 314.6844 | `TAKE_PROFIT_ORDER` | `direct_usd_shelf_retest_short`; archive says failed retest of broken shelf and pre-armed TP. |
| `469427` | `469430` | 2026-04-22 14:53:12 | 2000 | 1.17325 | 1.17240 | 270.2989 | `TAKE_PROFIT_ORDER` | Entry thesis says failed shelf retest / lower-half churn; strategy memory says late EUR_USD SHORT retrace LIMIT -> TP. |
| `469898` | `469912` | 2026-04-30 02:45:12 | 8000 | 1.16790 | 1.16645 | 1856.3399 | `TAKE_PROFIT_ORDER` | Entry thesis says `eurusd_failed_shelf_retest`; state.md labels the live seat medium-term breakout/failure / upper-box fade. |

All three have timestamp, price, units, attached TP, spread field, exit reason, realized P/L, transaction ID, and a current `execution_ledger.db` duplicate count of 0.

## Main Sources

- `data/legacy_history.db live_trade_events`
  - `469278`: entry line `10433`, close line `10434`
  - `469427`: entry line `10519`, close line `10520`
  - `469898`: entry line `10889`, protection update line `10890`, close line `10893`
- `data/legacy_history.db legacy_records`
  - `469278`: chunks `86206/86207`, pretrade `983`, seat outcome `14735`
  - `469427`: pretrade `1110`, seat outcome `22804`
  - `469898`: seat outcome `36888`, trade row `58785`
- Archive markdown:
  - `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z/collab_trade/daily/2026-04-21/trades.md`
  - `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z/collab_trade/daily/2026-04-22/trades.md`
  - `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z/collab_trade/daily/2026-04-29/state.md`
  - `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z/collab_trade/strategy_memory.md`
- `data/execution_ledger.db`
  - Accepted legacy IDs `469278`, `469427`, and `469898` were not present in current `execution_events` proof rows.
- Notion:
  - Search query: `QuantRabbit EUR_USD SHORT BREAKOUT_FAILURE TAKE_PROFIT_ORDER 469278 469427 469898`
  - Search query: `QuantRabbit EUR_USD SHORT proof floor 20 remaining 3 market close leak negative expectancy`
  - Fetched page: `quant-rabbit-profitability-evidence-repair-2026-07-03`
  - Result: corroborated no broker-side actions, negative expectancy, market-close leakage, no-live-ready gates, and operator review blockers.

## Rejected Or Held Out

| Trade | Reason |
| --- | --- |
| `469523` | TP close exists, but inspected thesis did not unambiguously map to BREAKOUT_FAILURE or failed-shelf/retest. |
| `469530` | TP close exists, but inspected thesis reads as direct USD support / lower-shelf continuation, not clear BREAKOUT_FAILURE. |
| `469548` | TP close exists, but inspected thesis reads as live direct-USD / inventory lead rather than failed-breakout proof. |
| `469404` | STOP_LOSS_ORDER loss. Not TP/HARVEST proof. |
| `469357` | False-break reclaim loss, not TAKE_PROFIT_ORDER proof. |
| `469641`, `469715`, `469741`, `469855`, `469880` | TP close rows visible, but the joined source did not expose enough entry/strategy evidence for exact-shape proof. Not needed after the three stronger samples reached the floor. |

## Still Blocking Live Permission

- `NEGATIVE_EXPECTANCY_ACTIVE`: `data/capture_economics.json` overall remains 229 trades, win rate 0.5983, avg win 418.0 JPY, avg loss 1063.9 JPY, expectancy -177.4 JPY/trade, net -40616.9 JPY.
- `MARKET_CLOSE_LEAK_PRESENT`: target shape still has 10 market-close losses for -7636.3 JPY. These were not counted as proof.
- `SPREAD_SLIPPAGE_PROOF_MISSING`: accepted rows include spread fields, but the target still needs the separate positive spread/slippage proof artifact.
- `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`: remains a global blocker from the acquisition plan.
- Guardian/operator review and current gateway blockers remain. Proof-floor recovery is evidence, not execution permission.

## Next Action

Reconcile/import the three accepted legacy TP rows into the exact proof artifact and rebuild capture-economics, payoff, and harvest reports read-only. After that, rerun spread/slippage and month-scale checks. Keep live permission false unless all current broker-truth, forecast, risk, verifier, guardian/operator, and gateway checks pass.

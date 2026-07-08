# EUR_USD SHORT BREAKOUT_FAILURE Proof Floor Update

Run date: 2026-07-08 JST

## Verdict

Status: `PROOF_FLOOR_REACHED_STILL_BLOCKED`

The legacy search samples `469278`, `469427`, and `469898` can be accepted as duplicate-free `EUR_USD SHORT` failed-shelf / failed-retest `TAKE_PROFIT_ORDER` proof evidence. This moves the target shape from 17 wins / 0 losses / gap 3 to 20 wins / 0 losses / gap 0 for the proof-floor count.

This is evidence integration only. It does not create live permission, does not rewrite the current generated `harvest_live_grade_path.json`, and does not clear gateway, guardian, market-close, spread/slippage, or month-scale blockers.

## Proof Delta

| State | Wins | Losses | Proof floor | Remaining | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Before legacy update | 17 | 0 | 20 | 3 | Sample gap present |
| After accepted legacy update | 20 | 0 | 20 | 0 | Proof floor reached |

Accepted samples:

| Trade | Exit TX | Entry UTC | Exit UTC | Units | Entry | Exit | P/L JPY | Exit reason | Accepted as |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `469278` | `469281` | 2026-04-21 08:32:39 | 2026-04-21 09:08:35 | 3000 | 1.17656 | 1.17590 | 314.6844 | `TAKE_PROFIT_ORDER` | broken-shelf / failed-retest LIMIT with pre-armed TP |
| `469427` | `469430` | 2026-04-22 14:33:24 | 2026-04-22 14:53:12 | 2000 | 1.17325 | 1.17240 | 270.2989 | `TAKE_PROFIT_ORDER` | failed shelf retest / lower-half churn LIMIT with attached TP |
| `469898` | `469912` | 2026-04-29 23:16:58 | 2026-04-30 02:45:12 | 8000 | 1.16790 | 1.16645 | 1856.3399 | `TAKE_PROFIT_ORDER` | failed shelf retest / medium-term breakout-failure LIMIT with attached TP |

Duplicate check: `execution_ledger.db` returned `duplicate_count=0` for the accepted trade IDs and exit transaction IDs.

## Canonical Integration

`data/eurusd_short_breakout_failure_proof_floor_update.json` is the canonical proof-floor update packet for this task. It records the accepted legacy rows as evidence and marks the sample gap cleared.

Existing generated artifacts are intentionally not rewritten in this read-only pass:

- `data/harvest_live_grade_path.json` still reports 17 TP trades / 0 losses / proof gap 3.
- `data/payoff_shape_diagnosis.json` still reports the pre-update target-shape proof state.
- `data/as_proof_pack_queue.json` still has `queue_count=0`.
- `data/portfolio_4x_path_planner.json` still has `NO_LIVE_READY_PORTFOLIO`.

That separation is deliberate: proof evidence is now ready to import/reconcile, but generated live-grade artifacts must be rebuilt by the normal read-only evidence pipeline.

## Live-Grade Re-Evaluation

The former sample-gap blocker is cleared by this update packet. The target is still not live-grade:

- `SPREAD_SLIPPAGE_PROOF_MISSING`: legacy rows include timestamp, price, units, and `Sp=0.8pip`, but there is still no positive spread-included replay/proof attached for the exact LIMIT attached-TP HARVEST vehicle.
- `MARKET_CLOSE_LEAK_PRESENT`: target shape still has 10 `MARKET_ORDER_TRADE_CLOSE` losses for -7636.3 JPY. These were not counted as proof.
- `NEGATIVE_EXPECTANCY_ACTIVE`: global capture economics remains 229 trades, win rate 0.5983, avg win 418.0 JPY, avg loss 1063.9 JPY, expectancy -177.4 JPY/trade, net -40616.9 JPY.
- `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`: global residual remains -20504.5826 JPY. Current harvest packet shows no direct target-shape month-scale blocker, but the global blocker is still visible.
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`: operator review remains blocked; receipt `832d2908eeb84b2f` is expired/stale for routing and `normal_routing_allowed=false`.
- `NOT_IN_PROOF_QUEUE`: A/S proof queue is still empty and was not rewritten.
- `NO_FRESH_GATEWAY_PERMISSION`: `live_order_request.status=NO_ACTION`, `send_requested=false`, and broker snapshot is stale for live permission.

Current best intent remains `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT`, `DRY_RUN_BLOCKED`, attached technical TP / HARVEST, with blockers including range forecast mismatch, operator-manual same-theme overlap, self-improvement adverse path, and guardian/operator review.

## Notion Check

Notion was searched and fetched read-only. The closest page, `quant-rabbit-profitability-evidence-repair-2026-07-03`, corroborates no broker-side actions, negative expectancy, market-close leakage, month-scale replay blockers, no-live-ready gates, and operator-review blockers. No Notion updates were made.

## Next Read-Only Actions

1. Import or reconcile accepted legacy TP rows `469278`, `469427`, and `469898` into the canonical proof/capture source path.
2. Rebuild `capture-economics`, payoff/harvest diagnostics, A/S proof queue, and portfolio planner read-only so generated artifacts move from 17/0 gap 3 to 20/0 gap 0.
3. Attach positive spread/slippage proof for the exact LIMIT attached-TP HARVEST geometry.
4. Rerun 744h timing/profitability acceptance to confirm month-scale blockers.
5. Keep guardian/operator review, current broker truth, forecast/telemetry, strategy, risk, verifier, and gateway checks as hard live-permission requirements.

## Safety

No live order, cancel, close, TP/SL edit, launchd change, broker mutation, gate relaxation, proof-floor lowering, 4x lot backsolve, secret exposure, or invented operator decision was performed or authorized.

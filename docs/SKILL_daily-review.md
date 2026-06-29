# Daily Review Skill

## Daily Target Review

- Reached +5%? `YES` / `NO`
- Reached +10%? `YES` / `NO`
- 10% Extension Gate: `YES` / `NO`
- Max progress: JPY and `%` from day-start NAV
- Final progress: JPY and `%` from day-start NAV
- Best target-path trade: pair / side / vehicle / grade / contribution
- Worst target-path drag: pair / side / vehicle / grade / risk used

## Miss Classification

If +5% was missed, classify the primary miss as exactly one:

- `discovery`: no valid A/S path was found early enough.
- `deployment`: valid path existed but was not converted into an executable receipt/order.
- `sizing`: valid path was under-sized, over-capped, or could not fund a production lot.
- `vehicle`: thesis was plausible but order type, TP/SL geometry, or pair expression was wrong.
- `management`: plus P/L or TP progress was not protected, harvested, or reloaded correctly.
- `bad session`: spread, whipsaw, news, or market structure made the day unsuitable.

If +10% was missed after +5% was reached, classify the extension miss with the same labels and state whether the 10% Extension Gate was actually `YES`.

## LIVE-LEARNING Target-Path Classification

For every target-path trade actually sent by `LiveOrderGateway`, classify the trade as exactly one:

- `discovery failure`: path role, attack-stack slot, grade, or thesis mapping was missing or wrong.
- `deployment failure`: live send was recorded but no resolved broker P/L is available yet.
- `sizing failure`: suggested/final units, production-lot fit, risk, or +5% contribution was invalid or too small.
- `vehicle failure`: the thesis was plausible, but pair/side/order type/TP/SL expression lost within planned risk.
- `management failure`: loss exceeded planned risk or exit path showed poor protection/harvest/close management.
- `good execution`: broker outcome was non-negative and the receipt evidence was complete.

`daily-review` writes these rows under `target_path_live_reviews` in `data/trader_overrides.json` and summarizes counts in `_diagnostics.target_path_live_review_counts`. This is feedback for the next cycle, not a live-order path.

## USER_ALPHA / OPERATOR_ALPHA Review

Profitable manual/operator-discovered outcomes are classified separately from system-discovered bot edge.

- `user discovered / system managed`: user/operator found the entry and broker-side TP or system TP management realized the win.
- `system discovered / system managed`: normal trader/gateway-attributed winner; this remains bot performance.
- `user discovered / system failed to continue`: user/operator winner closed green, but the next trader cycle did not evaluate RELOAD / SECOND_SHOT / +5% continuation.
- `system ignored user-alpha continuation`: active `user_alpha_continuation` existed and the receipt neither continued the pair/side nor named an exact blocker.
- `stale pending blocked continuation`: stale pending occupancy or duplicate parent-lane geometry prevented cancel/replace of the pending id being replaced.

`daily-review` writes profitable user-led outcomes under `user_alpha_trades` and the active obligation under `user_alpha_continuation` in `data/trader_overrides.json`. These rows must not be folded into system direction bias, same-day loss streaks, or bot expectancy. They are continuation evidence for the next trader receipt.

## Market Read Review

For every row in `data/market_read_predictions.jsonl`, score the market read separately from trade P/L:

- 30m prediction accuracy.
- 2h prediction accuracy.
- Full-read verdict: `CORRECT`, `WRONG`, `MIXED`, or `INVALIDATED_FIRST`; unresolved rows remain `PENDING`.
- Whether blockers prevented a correct read from being traded.
- Whether a wrong read still passed filters and was traded.
- Best-trade-if-forced outcome count, separate from actual execution.
- Codex read vs operator manual trade comparison when operator-manual comparison evidence exists; otherwise mark unknown.

This is discovery/execution separation. A blocked but correct read is discovery success / execution miss. A wrong read that passes filters is market-read failure. Do not turn negative expectancy, `LIVE_READY=0`, or blocker codes into a substitute for current tape prediction.

## Required Evidence

- `data/daily_target_state.json`
- `docs/daily_target_report.md`
- `data/order_intents.json`
- `docs/gpt_trader_decision_report.md`
- `data/market_read_predictions.jsonl`
- `docs/market_read_score_report.md`
- `docs/autotrade_cycle_report.md`
- `docs/execution_ledger_report.md`
- `tools/position_sizing.py` or `tools/place_trader_order.py` dry-run output for fresh target-path orders
- `target_path_receipt` rows recorded by `LiveOrderGateway` / `execution-ledger-sync` for any LIVE-LEARNING target-path send
- `user_alpha_trades` / `user_alpha_continuation` in `data/trader_overrides.json` for profitable user-led winners

## Output Rule

- Do not place live orders from daily review.
- Feed repair findings into trader docs, strategy memory, tests, or the next trader override review before the next cycle.

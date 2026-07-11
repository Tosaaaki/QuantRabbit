# Daily Review Skill

## Rolling 30D Target Review

- Rolling policy: `ROLLING_30D_4X`
- Rolling 30d start equity
- Current equity raw
- Capital flows 30d
- Funding-adjusted equity
- Rolling 30d multiplier raw
- Rolling 30d multiplier funding-adjusted
- Remaining to 4x raw
- Remaining to 4x funding-adjusted
- Required calendar daily return raw
- Required active-day return raw
- Required calendar daily return funding-adjusted
- Required active-day return funding-adjusted
- Required calendar daily return (legacy alias of funding-adjusted)
- Required active-day return (legacy alias of funding-adjusted)
- Performance basis
- Sizing basis
- Pace state: `AHEAD` / `ON_PACE` / `BEHIND` / `DANGER`
- Performance and 30d 4x pace use funding-adjusted equity; risk, margin, and sizing use raw broker NAV.

## Daily Pace Review

- +5% pace marker met? `YES` / `NO`
- Reached +10%? `YES` / `NO`
- 10% Extension Gate: `YES` / `NO`
- Max progress: JPY and `%` from day-start NAV
- Final progress: JPY and `%` from day-start NAV
- Acceptable no-edge day? `YES` / `NO`
- Best target-path trade: pair / side / vehicle / grade / contribution
- Worst target-path drag: pair / side / vehicle / grade / risk used

## Miss Classification

If +5% pace was missed, classify the primary miss as exactly one:

- `discovery`: no valid A/S path was found early enough.
- `deployment`: valid path existed but was not converted into an executable receipt/order.
- `sizing`: valid path was under-sized, over-capped, or could not fund a production lot.
- `vehicle`: thesis was plausible but order type, TP/SL geometry, or pair expression was wrong.
- `management`: plus P/L or TP progress was not protected, harvested, or reloaded correctly.
- `bad session`: spread, whipsaw, news, or market structure made the day unsuitable.
- `acceptable no-edge`: no A/S setup and no +10% extension gate existed; do not mark this as forced execution failure.

If +10% was missed after +5% was reached, classify the extension miss with the same labels and state whether the 10% Extension Gate was actually `YES`.

Also score:

- Missed +10 extension when gate was `YES`.
- Trade-shape match across all candidate pairs, not USD_JPY-only replay.
- Thesis-state exit quality: `ALIVE` / `WOUNDED` / `INVALIDATED` / `EMERGENCY`.
- SL/loss-cut failure: stop inside noise/battle/event/theme zone, or loss-side close without invalidation.
- Margin/carry failure: margin rescue add, margin closeout tolerance, or unattended carry.

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

For every row in `data/market_read_predictions.jsonl`, score the market read separately from trade P/L. Keep schema-v1 rows as immutable legacy target/full evidence; its stored `CORRECT`/`WRONG`/`MIXED`/`INVALIDATED_FIRST` labels are not direction accuracy.

For schema v2 report, separately for 30m and 2h:

- Endpoint direction accuracy (`CORRECT` / `WRONG`).
- Target completion (`TOUCHED` / `NOT_TOUCHED`).
- Invalidation touch and M5-derived first-touch order; same-candle TP/invalidations are ambiguous.
- Full-read completion, without folding a correct direction but incomplete target into `WRONG`.
- Missing/incomplete M5 truth as `UNRESOLVED`, never an endpoint-derived `*_FIRST` label.
- Exact duplicate/coalesced and same-source conflicting/ineligible counts.
- Direct-origin counts: top-level `originating_decision_receipt_id`, `direct_execution_attribution`, and `direct_realized_outcome`. Join only the same row's exact `(gptd, mr2)` ids to explicit gateway order/fill/trade ids and then exact trade ids to realized P/L.
- Prior-prediction reaction counts: `reaction_chain.first_subsequent_decision`, reaction execution attribution, and reaction realized outcome. This answers what the next decision did after the prior prediction; it is not the originating prediction's own execution or P/L.
- Both lineage paths must report `pair_or_time_inference_used=false`. Missing broker IDs stay `UNATTRIBUTED`; missing outcomes stay `UNRESOLVED`.

The v2 truth source is `MID_CANDLE_DIAGNOSTIC`, read-only, and `live_permission=false`. Its bounded feedback may inform the next GPT market read but cannot change verifier, gateway, risk, sizing, or execution permission.

This is discovery/execution separation. A blocked but correct read is discovery success / execution miss. A wrong read that passes filters is market-read failure. Do not turn negative expectancy, `LIVE_READY=0`, or blocker codes into a substitute for current tape prediction.

Rolling 30d 4x is a product KPI, not a proved capability. Never call a point estimate, replay result, raw-JPY comparison, or small recent winning sample proof of positive expectancy, improvement, or 4x reachability. Report the strict lifetime/recent split and the remaining evidence gap.

## Required Evidence

- `data/daily_target_state.json`
- `docs/daily_target_report.md`
- Rolling 30d fields in `data/daily_target_state.json`: `rolling_30d_start_equity`, `current_equity_raw`, `capital_flows_30d`, `funding_adjusted_equity`, `rolling_30d_multiplier_raw`, `rolling_30d_multiplier_funding_adjusted`, `remaining_to_4x_raw`, `remaining_to_4x_funding_adjusted`, `required_calendar_daily_return_raw`, `required_active_day_return_raw`, `required_calendar_daily_return_funding_adjusted`, `required_active_day_return_funding_adjusted`, `required_calendar_daily_return`, `required_active_day_return`, `performance_basis`, `sizing_basis`, `pace_state`.
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

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

## Required Evidence

- `data/daily_target_state.json`
- `docs/daily_target_report.md`
- `data/order_intents.json`
- `docs/gpt_trader_decision_report.md`
- `docs/autotrade_cycle_report.md`
- `docs/execution_ledger_report.md`
- `tools/position_sizing.py` or `tools/place_trader_order.py` dry-run output for fresh target-path orders

## Output Rule

- Do not place live orders from daily review.
- Feed repair findings into trader docs, strategy memory, or tests before the next cycle.

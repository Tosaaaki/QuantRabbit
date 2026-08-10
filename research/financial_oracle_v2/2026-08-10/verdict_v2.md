# Financial oracle and executable path verdict

## Financial result

`TRADE_CASHFLOW_FINANCIAL_ORACLE_V2` is **PASS** for all 251 frozen trades.
The previous last-close-only label omitted 59 DAILY_FINANCING transactions
(164 cohort components affecting 58 trades, -9,278.5941 JPY) and two partial
reductions (-3,792.3 JPY).

The corrected 64-day VALIDATION baseline is 101 trades, +11,706.0523 JPY,
PF 1.44693294, expectancy +115.9015 JPY/trade, episode-terminal DD
9,876.0991 JPY and event-time DD 10,170.1845 JPY. The prior +15,144.4802 JPY
was overstated by 3,438.4279 JPY. The baseline remains positive; this does not
prove daily opportunity count, scaling, or an exit improvement.

Independent verification reproduced 251/251 trade labels, 567/567 account
balance transitions, all fixed-window metrics, and all financing residuals
(20/20 checks). The bounded financial tests pass 8/8.

## Body/wick and path result

OANDA S5 bid/ask exists for 146/251 trades (AUD_JPY, EUR_JPY, EUR_USD).
Observed executable-side MFE/MAE lower bounds exist for 145 trades. Only 5/251
have a complete interior S5 endpoint grid under the strict no-fill rule; the
remaining 246 are `UNRESOLVED`, including 105 trades whose pair has no OANDA
S5 source. Fifteen interior bars touch an active SL, but within-S5 order remains
unresolved and is not treated as a proven counterfactual stop fill.

Window/split strict path coverage is:

| Window | TRAIN | VALIDATION |
|---|---:|---:|
| 16d | 2/13 | 0/12 |
| 32d | 0/43 | 2/31 |
| 64d | 3/145 | 2/101 |

The independent path oracle passes 306/306 checks, including 251 raw entry
receipts, 12 independently rescanned paths, side-correct wick extrema, unit
conservation, and gross concurrent-margin arithmetic. Synthetic path tests
pass 8/8. Output regeneration is deterministic.

The peak gross trade-level margin proxy is 362,877.1112 JPY with nine cohort
trades open. This is deliberately not called account margin: broker netting,
available margin, and external/manual inventory have zero evidence coverage.

## Exit and multidimensional gate

Baseline, fixed BE, all-cost BE, partial TP+BE, pure ATR trail, SMA
deterioration trail, structure-break exit, and time exit are **not yet
admissible for comparative VALIDATION**. Strict path coverage is below the
preregistered minimum and decision-time fee/financing schedule, account margin,
partial-fill depth, and executable unwind evidence remain missing. Existing
SMA early-exit and sizing shadows were calculated against the superseded V1
financial label and cannot be adopted without V2 relabelling.

The frozen sparse-cube order remains: single-axis causal ablation, promising
two-axis interactions only, connected stable TRAIN plateau/Pareto, then one
frozen VALIDATION replay. Missing values remain missing. Hedge ratios 0.25,
0.35, and full lock remain independent diagnostic arms with unwind; none is a
profit assumption. The 10% and 30% targets remain reporting metrics, not sweep
objectives. Holdout, live, Paper, broker orders, and deploy were not used.

# Historical learning admission verdict v1

Decision: **REJECT for policy adoption; retain as executed-only diagnostic.** The failure is not “charts cannot be learned.” The current ledger does not yet preserve the full causal decision/label contract needed to prove that a learned trade/skip policy makes money.

## What was reconstructed

- 549 pre-holdout entry episodes: 251 executed with actual after-cost labels, 167 canceled unfilled, 107 accepted unresolved, 23 rejected, and 1 executed open/invalid.
- Full-entry label coverage is 45.72%. An append-only skip-decision ledger is absent, so true skip episodes are zero rather than fabricated.
- Causal prior-forecast coverage among labeled executions is 54.98%; entry-thesis coverage is 14.34%.
- Future forecast joins: 0. Label timestamps at/before feature time: 0. Forbidden future/outcome features: 0.

Actual broker P/L already includes bid/ask spread and fill slippage; financing is added explicitly and fee is zero under the account contract. Decision-time opportunity cost, cross-currency margin conversion and unfilled/skip counterfactuals are missing and fail closed.

## Walk-forward result

The initial 16-day window had only 13 purged TRAIN and 12 VALIDATION events, so models were not fit. The 32-day window had 43/31; the 64-day window had 145/101.

On 64-day VALIDATION, the existing all-executed cohort was +15,144.48 JPY with PF 1.585. The preregistered gate models did not improve it:

| Model | Selected / available | Net JPY | PF | Paired LCB JPY | Result |
|---|---:|---:|---:|---:|---|
| HistGradientBoosting | 39 / 101 | +1,676.30 | 1.144 | -338.10 | REJECT |
| Logistic | 13 / 101 | -3,712.13 | 0.294 | -416.00 | REJECT |
| Ridge | 7 / 101 | -1,756.20 | 0.309 | -397.04 | REJECT |
| Frozen forecast rule | 0 / 101 | 0 | n/a | -384.13 | REJECT |

The 32-day models show some positive absolute subsets, but every paired LCB is negative and the sign does not persist to 64 days. Accuracy is therefore not used as a rescue metric.

A separate standard-library oracle recomputed selected count, Net, PF, expectancy and DD from selected episode IDs for 12 window/model combinations: 12 passed, 0 failed.

## Root cause and safe next contract

1. Record every candidate decision before execution in one append-only schema, including `TRADE`, `SKIP`, `UNFILLED`, `REJECTED`, feature timestamp, exact intended bid/ask order geometry, regime and inventory/margin snapshot.
2. Attach future-only BA/S5 counterfactual labels after the horizon for every episode, including spread, slippage stress, financing, opportunity cost and margin feasibility. Never backfill with M1 or interpolation.
3. Re-run the same purged 16/32/64 contract. Only after all-entry coverage, multi-regime stability, positive Net/PF/paired LCB and DD/margin gates pass should Optuna or external boosting be considered.

No Paper, live, broker order, deploy, production configuration or holdout was touched.

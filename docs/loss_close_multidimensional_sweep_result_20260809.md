# Loss-close multidimensional sweep — pre-holdout checkpoint (2026-08-09)

## Outcome

The bounded multidimensional machinery is implemented, but the real paired
economic sweep is not yet admissible. Local canonical evidence contains 12
STOP_LOSS events in pairs with some S5 bid/ask history. Seven events have
causal feature context, but zero have a complete S5 sequence from original
entry through the fixed 60-minute unwind. Missing S5 rows are not filled
forward because that would manufacture fill ordering.

Status: `BLOCKED_INSUFFICIENT_STRICT_S5_COHORT`.

## What was swept

Stage 1 is a bounded 27-cell grid:

- frame set: M1, M5, or M1+M5;
- coupled structure/regime windows: 6/18, 12/24, or 24/48 bars;
- breakout/acceptance windows: 4/2, 8/2, or 12/3 bars;
- rail-attack tolerance frozen at 0.08.

This is not a full Cartesian search. Stage 2 may change one local structure
axis around a TRAIN plateau. Stage 3 may change only tolerance around a
surviving TRAIN centre. A plateau needs at least three connected cells and a
centre with at least two neighbours. VALIDATION can only confirm or reject the
frozen TRAIN region. A single best cell can never be adopted. The one-hour
embargo equals the maximum unwind horizon. TEST/HOLDOUT is rejected.

## Real cohort audit

| Item | Count |
|---|---:|
| Ledger STOP_LOSS events in the local-S5 pair family | 12 |
| Causal price-action context calculated | 7 |
| TRAIN context events after embargo | 4 |
| VALIDATION context events | 3 |
| Strict entry-to-unwind S5 paths | 0 |
| Minimum required per split | 30 |

For the seven context-ready events, missing five-second intervals range from 2
to 608. Three more events cannot establish an unambiguous first SL touch from
the local S5 sequence, and two have no local S5 coverage at the event time.

The strongest context-only cell was M1, structure/regime 12/24,
breakout/acceptance 4/2, tolerance 0.08. It produced a named multi-bar setup and
an against-inventory direction on all seven context events; the one/two-candle
control was against inventory on three. This is not an expectancy result. The
cohort is selected at stop-loss events, so adverse structure at that point is
partly built into the sampling condition. No cell was selected and no Stage-2
refinement was opened.

## Economic contract not relaxed

- Spread remains intrinsic in executable bid/ask.
- Fee, slippage, and financing must be charged once and cannot be inferred
  away because a feature looks useful.
- Hypotheses A (0.25/0.35 reverse STOP) and B (equal loss lock at entry or SL)
  still require resolved fill order, longest-leg margin, trend continuation,
  maximum drawdown, deterministic ruin floor, and complete unwind.
- The existing S5 hedge scorer treats every missing interval as fatal. The
  context layer's no-quote-gap warning exception does not flow into it.
- No Paper/live/broker/order/deploy/holdout permission exists.

## Evidence

- Machine-readable scan:
  `research/loss_close_price_action_sweep/preholdout_stage1_report.json`
- Scanner:
  `research/loss_close_price_action_sweep/run_preholdout_stage1.py`
- Plateau selector:
  `src/quant_rabbit/loss_close_multidimensional_sweep.py`
- Tests:
  `tests/test_loss_close_multidimensional_sweep.py`

The next economic run requires an immutable, gapless tick/S5 cohort with the
same original entries and frozen TP/SL. Substituting M1 or synthesising missing
S5 bars would answer a different question and is not accepted here.

# Contextual candidate engine — closed, 2026-08-19

Corpus: `research/manual_method_contextual_scorecard/2026-08-13/artifacts/live/evaluation_rows.jsonl`
(9,464 rows, 8,060 price-true), produced by the 20 `manual_method_*` processes
running since 2026-08-13.

## Verdict

**No edge, gross or net.** The restriction to arithmetically feasible cells did
not rescue it, and the one apparently significant cluster is a single day of one
currency's drift with the engine parked on one side.

## 1. The engine has no gross edge

Executable returns decompose exactly — `move = (long - short)/2`,
`cost = -(long + short)/2` — so gross directional capture is `selected + cost`:

| horizon | mean gross | share > 0 | mean cost | mean net |
|---|---|---|---|---|
| 5 min | **-0.031** | **49.2%** | 3.322 | -3.353 |
| 30 min | -0.241 | 50.1% | 3.321 | -3.562 |
| 60 min | -0.488 | 50.9% | 3.247 | -3.735 |

A coin. The measured -3.35 net was the toll with nothing underneath it. This
closes the hypothesis that a narrower pair set would flip the sign: there was no
gross edge for a tighter spread to protect.

## 2. Restricting to feasible cells does not change it

Admission was outcome-blind (`tools/feasibility_gate.py`): a cell is feasible
when a perfect predictor clears zero, `mean(|move|) - mean(cost) > 0`. 33 of 84
cells admitted; 51 arithmetically closed.

Within the 33 admitted cells (1,596 rows, 170 clock blocks):

| | |
|---|---|
| mean gross | **-0.274 pips** |
| share > 0 | 51.8% |
| clock-block bootstrap 95% CI | **[-0.803, +0.225]** |
| P(mean gross > 0) | 14.2% |
| cells with positive gross | 13 of 33 (chance gives ~16) |

Bootstrap blocks are clocks, not rows: 28 pairs sampled at one clock share the
same shocks, so resampling rows would fake independence the corpus lacks.

## 3. The one significant-looking cluster is a stuck side

Top cells by gross were all CHF-quote, with naive t up to +3.70. They are not
four findings:

| cell | n | side mix | mean move | mean gross |
|---|---|---|---|---|
| CAD_CHF 60min | 49 | **LONG 48 / SHORT 1** | +2.304 | +1.992 |
| EUR_CHF 60min | 55 | LONG 51 / SHORT 4 | +1.970 | +1.792 |
| NZD_CHF 60min | 43 | LONG 38 / SHORT 5 | +1.608 | +1.366 |
| GBP_CHF 60min | 43 | LONG 41 / SHORT 2 | +1.403 | +1.483 |

The engine was not choosing a direction — it held one. `gross ≈ mean move` in
every row of that table, which is the signature of a constant-side position
rather than a prediction. CHF weakened across the window and a parked long
`XXX_CHF` collected it.

177 of those 190 rows fall on **2026-08-13 alone**. The four cells share one
currency leg, so the cluster is closer to one observation than to four, on one
day. This is the failure mode already recorded in
`project_regime_verdict_20260805` — concentration mistaken for sample size.

## 4. The corpus is 10.5 hours, not 6 days

| | |
|---|---|
| first entry | 2026-08-13T14:22:00Z |
| last entry | 2026-08-14T00:45:22Z |
| entry days | 08-13: 8,820 rows / 08-14: 644 rows |
| `observed_at` on 08-18 | 4,004 rows |

Candidate generation ran for about 10.5 hours and stopped. Everything since has
been the evaluators re-scoring a frozen window — 4,004 rows re-observed on 08-18
against entries that all predate 08-14T00:45Z. The health file's
`SERVICE_NOT_RUNNING` and `LATEST_CONTEXT_CLOCK_STALE` are that.

Generation continued for nine hours past the 2026-08-13T15:00:00Z spread
calibration expiry, so the expiry is **not** the cause of the stop; the recorded
cause is `SERVICE_NOT_RUNNING` / `SERVICE_STDERR_NONZERO`.

## What this rules out and what it does not

Ruled out: this candidate engine, on this feature set, at 5/30/60 minutes. Also
ruled out is pair selection as a rescue for it — cost reduction cannot help a
gross of zero.

Not ruled out: anything about longer horizons, non-price inputs, or the operator's
own method, none of which this corpus tested. The 2026-08-07 conclusion stands
unchanged — the remaining doors are cost reduction and non-price inputs — and
this result removes one candidate from behind the first door without touching
the second.

## Consequence

The 20 `manual_method_*` processes in `~/.codex/worktrees/203e/QuantRabbit`
(five at ~70% CPU, one at 458 CPU-minutes) are re-scoring a 10.5-hour corpus
whose signal is a coin. They are producing nothing further. That worktree is
also at detached HEAD with 7.8 GB of untracked output, so a cleanup erases it.

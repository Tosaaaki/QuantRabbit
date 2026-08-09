# SYSTEM_UTILIZATION_RCA / FUSED_DECISION_V1 verdict

## Decision

**HOLD_EVIDENCE_PIPELINE_BLOCKED.** The programs are not useless, but most of
their outputs cannot reach one reconstructable decision-time record. Connecting
more model parameters before fixing that evidence join would optimize missingness.

## What changed

- Preregistered `FULL_INFERENCE_ENSEMBLE_V1` and `FUSED_DECISION_V1` before evaluation.
- Aligned 21 systems across the same 251 episodes: 5,271 inference rows plus a
  separate 251-row actual-after-cost outcome boundary.
- Emitted exactly one research answer per episode: `WAIT` 196, `SKIP` 55,
  `TRADE` 0, `MANAGE` 0.
- Preserved system-family disagreement, missing inputs, source hashes, and
  output hashes. `ALL_TRADES` remains a comparator and never becomes fallback.

## Utilization RCA

| State | Systems |
|---|---:|
| Used | 8 |
| Generated only | 4 |
| Disconnected | 4 |
| Nullified by fallback | 1 |
| Insufficient evidence | 4 |

Episode-aligned causal coverage is the binding problem: forecast 138/251,
gapless price action 136/251, both 55/251, decision-time entry thesis 18/251.
Only one episode has forecast + price action + thesis + margin evidence.
64-day VALIDATION margin evidence is 15/101 (14.85%); decision-time fillability
and unwind evidence is 0/101. The market-read ledger has 5 predictions but its
execution-link artifact is absent in this snapshot.

The prior xarray/SALib/pymoo/MAPIE 0 JPY increment is therefore explained:
they were consumed by research verifiers, not by a final decision consumer.
xarray organized rows; SALib produced an unstable TRAIN rank; pymoo's constrained
front became empty under margin; MAPIE measured intervals without changing an
action. None could contribute realized incremental P/L in that wiring.

## Paired financial result

The 64-day VALIDATION ALL_TRADES comparator is +15,144.4802 JPY. The strict fused
decision selects no admissible trade, so its research shadow is 0 JPY,
incremental -15,144.4802 JPY with a negative paired LCB. This is opportunity
cost, not a live loss: no live/Paper decision was changed.

The single statistical family selected four validation trades but produced
-1,082.5907 JPY, incremental -16,227.0709 JPY, PF 0.174, and negative paired
LCB. Technical-only and two-family uncertainty-gated models selected zero.
The 16/32-day two-family sets were too small for the preregistered OOF rule.
No candidate meets Net > baseline and paired LCB > 0.

## Adoption boundary

Do not tune fusion weights or add another library yet. The next independent
fix is a research-only, append-only point-in-time evidence ledger keyed by
`decision_id` that persists bid/ask fillability, spread/slippage/financing,
margin/exposure snapshot, and exit/unwind validity before the outcome exists.
After that, regenerate the same 16/32/64 contract and only then reopen stacking,
regime mixture, Pareto + MAPIE abstention, and drop-one-family contribution.

Holdout remained unread. Live, Paper, broker mutation, orders, deploy, runtime
configuration, active DB/WAL, and existing live dirty files were untouched.

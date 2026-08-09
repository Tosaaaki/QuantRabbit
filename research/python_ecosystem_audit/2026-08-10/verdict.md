# Audit verdict

## Decision

The capacity blocker is resolved. Six candidates were installed one at a time
in candidate-specific, ignored Python 3.12 arm64 environments from SHA-256
recorded wheels. The baseline research lock and runtime dependencies were not
changed.

Adopt as research-only adapters: xarray 2026.7.0, SALib 1.5.2, pymoo 0.6.2,
and MAPIE 1.5.0. Hold DoWhy 0.14 as an isolated causal diagnostic and hold
River 0.25.0 until a real chronological drift signal is available. None of
these decisions admits a strategy, changes fill/PnL truth, or establishes
after-cost profitability.

## Same-fixture evidence

- 252 episode records and 567 canonical long-table rows were used by every
  adapter. TRAIN/VALIDATION remained separate and holdout was unread.
- The QR after-cost/LCB digest was identical before and after every adapter.
- xarray: populated value max error 0; after-cost and LCB sum error 0; known
  missing coordinate remained NaN.
- SALib: fixed-seed Sobol repeat was exact; QR factorial lookup error 0. The
  bounded fixture was dominated by regime sensitivity, with a smaller
  regime×method interaction. This is fixture evidence only.
- pymoo: the constrained VALIDATION Pareto front exactly matched the self-owned
  oracle (one nondominated fixture candidate among 15 feasible candidates).
- DoWhy: estimated effect matched independent OLS within 1e-12; the seeded
  placebo estimate was near zero. Causal identification assumptions remain
  assumptions, so this is not promotion evidence.
- MAPIE: 64 fit / 64 conformal / 123 complete VALIDATION rows, one missing row
  excluded rather than zero-filled; manual bound error 0 and observed fixture
  coverage 95.12%.
- River: chronological online mean error 0 and repeat exact; ADWIN emitted no
  change point, so no drift conclusion is admitted.
- Eleven tests passed, plus Python compilation. Raw time and Python allocation
  are recorded without claiming cross-semantic speedups.

## Capacity and isolation

The deterministic preflight reported 73,430,388,736 free bytes and no removal.
Final evidence generation reported 71,050,883,072 free bytes. Candidate venvs
and wheelhouse used about 2.19 GiB together, below the 5 GiB run-owned cap.
No candidate crossed the 8 GiB free-space pause, 5 GiB hard stop, or 1 GiB per
install decrease threshold. The research-path-excluded Git status SHA remained
`e7a32f2f...78fd4` for every install. Open DB/WAL handles were in another
worktree and were not touched.

## Boundaries and next evidence

External libraries remain adapters behind QuantRabbit's decision-time,
bid/ask, spread/slippage/financing/opportunity-cost, fill, margin and unwind
contract. Their default fills, PnL, CV, missing-value behavior and scalar
objectives are never truth.

The next admissible step is shadow use on the preregistered real TRAIN and
VALIDATION cohorts. xarray/SALib/pymoo/MAPIE may be retained only if the real
cohort preserves numeric equality, missingness, determinism, margin/DD
constraints and after-cost LCB. DoWhy needs an explicit causal graph and
refutation contract; River needs a real chronological drift signal. Holdout,
live, Paper, broker, order and deploy remain prohibited.

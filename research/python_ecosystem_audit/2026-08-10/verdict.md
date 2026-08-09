# Audit verdict

## Decision

**No new dependency is adopted in this cycle.** Existing pinned NumPy,
SciPy, scikit-learn, Hypothesis and joblib remain the research baseline. The
new multidimensional reporting layer is a small, dependency-free adapter with
an xarray-compatible coordinate model; xarray, SALib, pymoo, DoWhy, MAPIE and
River are `HOLD_ADAPTER_READY` until an isolated install and same-cohort
shadow run can be afforded safely. The host had only about 717 MiB free, so
installing a stack or compiling binaries would violate the bounded-resource
contract.

## Bounded proof

- 252 synthetic episode records, 567 canonical long-table metric rows.
- Long-table → sparse labelled cube is deterministic; one deliberate missing
  `VALIDATION/RANGE/cube_shadow/stress_plus_1pip/margin_cap_70/TIMEOUT` cell is
  absent, not zero.
- Ten two-factor interaction contrasts are retained. `analysis.json` keeps
  after-cost net, LCB, PF, drawdown, margin, turnover, fill/unwind validity and
  coverage as separate metrics; it does not scalarise them into PnL.
- Validation Pareto filtering returned one feasible nondominated fixture
  candidate under the preregistered coverage/margin/fill/unwind constraints.
  This is a fixture smoke result, not a trading admission.
- Side-aware bid/ask arithmetic and explicit fee/financing/slippage/opportunity
  cost pass an independent hand-check. Fallback placebo, split-conformal and
  chronological drift adapters are deterministic and marked
  `EXECUTED_FALLBACK`, never represented as external-package evidence.
- Six tests pass, including reproducibility, missing-cell preservation,
  interaction retention, no-holdout use, no-live side effect, and the
  bid/ask oracle.

## Adoption ledger

`library_ledger.json` is the machine-readable decision record. The important
boundaries are:

- `ADOPT_KEEP`: NumPy, SciPy, scikit-learn, Hypothesis, joblib — already pinned
  and used for numeric/statistical/model/property-test work.
- `HOLD`: Polars, PyArrow, DuckDB, Numba, Optuna, EconML, PyMC, ruptures,
  SymPy, TA-Lib parity — potentially useful, but no measured need or isolated
  proof yet.
- `HOLD_ADAPTER_READY`: xarray, pymoo, SALib, DoWhy, MAPIE, River — bounded
  fallback adapters are ready, external imports are absent, and installation is
  deferred until disk and exact lock conditions are safe.
- `REJECT`: generic backtesting/reporting engines and hmmlearn/pandas-ta for
  this contract. Their generic OHLC/fill/report defaults, limited maintenance,
  or lack of an auditability gain cannot replace QR's S5 bid/ask, financing,
  margin, partial-fill and unwind truth.

## Next safe step

When at least 2 GiB free and an isolated Python 3.12 arm64 environment can be
created without touching the runtime, install one candidate at a time (first
xarray, then SALib/pymoo, then DoWhy/MAPIE/River). Re-run this exact fixture
and then a TRAIN-only real-cohort shadow. Adopt only if numeric equality,
memory/latency, seed stability, leakage tests and after-cost LCB are all
read back. A candidate that changes the sign of the QR result is a diagnostic,
not a promotion.


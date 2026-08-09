# QuantRabbit Python ecosystem audit (2026-08-10)

This is a research-only, non-invasive audit. It inventories the existing
QuantRabbit implementation and compares external OSS candidates against the
same causal, bid/ask and after-cost contract. It does not change `src/`, the
live tree, Paper/broker/order/deploy configuration, or holdout data.

## Contract

The canonical artifact is a long table. Every row carries the required fields:

`episode_id`, `source_sha`, `decision_time`, `pair`, `timeframe`, `regime`,
`strategy`, `parameter_set`, `cost_scenario`, `exposure_state`, `exit_policy`,
`viewpoint`, `metric`, `value`, `uncertainty`, `sample_count`, and
`admission_status`.

The cube adapter derives the comparison axes
`split × timeframe × pair × regime × method × cost × risk × exit` from the
long table. Missing cells remain `null`/absent; they are never converted to
zero. Metrics remain separate (`after_cost_net_jpy`, `lcb_jpy`, `profit_factor`,
`max_drawdown_jpy`, `margin_coverage`, `turnover`, `fill_validity`,
`unwind_validity`, and `sample_coverage`).

All fill arithmetic is self-owned. A long entry is filled at ask and a short
entry at bid; exits use the opposite side. Spread, observed slippage,
financing, opportunity cost, margin feasibility, partial-fill and unwind flags
are data fields. External backtest defaults are never the oracle.

`TRAIN`, `VALIDATION`, and `holdout` are coordinates, not pooled rows. Any
candidate that cannot preserve decision-time causality, source lineage, and
the same cost model is `REJECT` or `HOLD`, never an adopted truth source.

## Reproduce

The baseline audit remains in the existing admission environment. Optional
packages are isolated from it in one ignored venv per candidate. Every wheel,
including transitives, is recorded in `adapter_wheel_manifest.json`; the full
installed distribution/license inventory is in `adapter_sbom.json`.

```bash
research/historical_learning_admission/.venv/bin/python \
  research/python_ecosystem_audit/2026-08-10/build_audit.py
research/historical_learning_admission/.venv/bin/python \
  -m unittest discover -s research/python_ecosystem_audit/2026-08-10 -p 'test_*.py'
python3 research/python_ecosystem_audit/2026-08-10/run_external_adapters.py
```

The last command is verify-only by default: it re-executes all probes and
fails if the wheelhouse/SBOM differs, without rewriting captured benchmarks.
Use `--capture` only when intentionally refreshing the dated evidence.

`build_audit.py` and `run_external_adapters.py` write only this research
directory. Rollback for one candidate is removal of only
`.adapter_envs/<candidate>` and `.wheelhouse/<candidate>`; rollback of the
tracked evidence is reverting the dedicated research commit. No runtime path
imports these adapters.

## External adapter result

- xarray 2026.7.0: adopted for research cube representation. All 567 populated
  values matched exactly, after-cost/LCB sums were unchanged, and the known
  absent cell stayed NaN.
- SALib 1.5.2: adopted for TRAIN-only sensitivity. The 32-cell QR lookup was
  reproduced with zero error; the missing outcome was excluded, not imputed.
- pymoo 0.6.2: adopted for research Pareto filtering. Its nondominated front
  exactly matched the QR oracle under the same five constraints.
- MAPIE 1.5.0: adopted for research uncertainty intervals. Chronological
  fit/conformal/validation partitions were preserved and manual finite-sample
  quantile bounds matched exactly.
- DoWhy 0.14: held as an isolated diagnostic. It matched manual OLS and the
  placebo refuter behaved as expected, but causal assumptions are unproven and
  its isolated dependency set is large and conflicts with the baseline SciPy
  pin.
- River 0.25.0: held until a real drift signal exists. Ordered online mean
  matched exactly, but ADWIN detected no change in this bounded fixture.

Raw adapter latency and peak Python allocation are recorded in
`external_adapter_report.json`. They are not presented as speedups because the
external operations do not all have semantically identical custom baselines.

## Intake policy

OSS candidates are checked from official documentation/repository/release
metadata, not popularity. The ledger records license, Python/Apple-Silicon
compatibility, maintenance/CI evidence, deterministic behavior, dependencies,
security/operational concerns, and the exact failure mode at QuantRabbit's
truth boundary. No GitHub code is copied.

## Real-cohort continuation

`verdict_real_shadow.md` applies only xarray, SALib, pymoo and MAPIE to the
frozen 16/32/64-day TRAIN/embargoed-VALIDATION cohort. The derived canonical
table adds the `window` coordinate and retains the named split, timeframe,
pair, regime, method, cost, risk and exit axes. DoWhy and River remain held
and are not executed by `run_real_shadow.py`.

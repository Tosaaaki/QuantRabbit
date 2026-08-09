# Historical learning library audit

Audited 2026-08-09 against official project/PyPI release metadata. The repository's `.venv` was not changed. Installation is isolated at `research/historical_learning_admission/.venv` and excluded from Git.

## Selected and pinned

| Package | Version | License | Python / Apple Silicon readback | Purpose |
|---|---:|---|---|---|
| NumPy | 2.5.1 | BSD-family | CPython 3.12 arm64 wheel imported | arrays and deterministic numeric oracle |
| SciPy | 1.18.0 | BSD-3-Clause | CPython 3.12 arm64 wheel imported | statistical primitives; no hidden backtest defaults |
| scikit-learn | 1.9.0 | BSD-3-Clause | CPython 3.12 arm64 wheel imported | logistic, Ridge and histogram tree baseline |
| Hypothesis | 6.165.2 | MPL-2.0 | CPython 3.12 arm64 wheel imported | property/leakage invariants |

All transitive packages are pinned in `requirements.lock`: joblib 1.5.3, narwhals 2.24.0, sortedcontainers 2.4.0 and threadpoolctl 3.6.0. Runtime readback: Python 3.12.8, `arm64`. Installed environment size: 204 MiB.

Official metadata: [NumPy](https://pypi.org/project/numpy/), [SciPy](https://pypi.org/project/scipy/), [scikit-learn](https://pypi.org/project/scikit-learn/), [Hypothesis](https://pypi.org/project/hypothesis/).

## Deferred or rejected for this admission run

- pandas, Polars and PyArrow: unnecessary for 549 JSONL episodes; standard-library streaming keeps the schema and timestamps explicit.
- statsmodels and arch: no preregistered econometric test needs them yet.
- Numba: no measured hotspot justifies another compiler/runtime dependency.
- vectorbt, Backtrader and backtesting.py: generic OHLC/mid-price and commission defaults do not implement this bid/ask, financing, partial-fill, margin and dual-unwind contract. Their framework defaults would require more auditing than the research-local replay.
- QuantStats and empyrical: metrics are independently recomputed from per-episode JPY values; empyrical is stale for this purpose.
- Optuna: explicitly forbidden outside TRAIN and unnecessary because no hyperparameter search is preregistered.
- LightGBM, XGBoost and CatBoost: deferred until simple baselines show stable after-cost improvement. LightGBM also adds a macOS OpenMP dependency. No deep-learning library was introduced.

Fresh metadata compared: [pandas](https://pypi.org/project/pandas/), [Polars](https://pypi.org/project/polars/), [PyArrow](https://pypi.org/project/pyarrow/), [statsmodels](https://pypi.org/project/statsmodels/), [arch](https://pypi.org/project/arch/), [Numba](https://pypi.org/project/numba/), [vectorbt](https://pypi.org/project/vectorbt/), [Backtrader](https://pypi.org/project/backtrader/), [backtesting.py](https://pypi.org/project/backtesting/), [Optuna](https://pypi.org/project/optuna/), [LightGBM](https://pypi.org/project/lightgbm/), [XGBoost](https://pypi.org/project/xgboost/), [CatBoost](https://pypi.org/project/catboost/).

## Default and leakage audit

- No close, exit reason, realized P/L, financing, projection resolution or future candle is present in the feature vector.
- The forecast join is same-pair and `forecast_timestamp <= ORDER_ACCEPTED timestamp`; direct readback found zero future joins.
- Actual fill P/L is the label, not a mid-price backtest. Spread and fill slippage are implicit in broker-realized P/L; financing is explicit.
- Margin and opportunity cost are not silently defaulted. Missing decision-time cross-currency conversion and skip counterfactuals fail the adoption gate.
- Library output is checked by hand-calculable fixtures, property tests, and an independent per-event arithmetic readback. No library result can override the acceptance contract.

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

Use the already pinned admission environment. No package installation is
required or performed by this audit (the host has less than 1 GiB free space).

```bash
research/historical_learning_admission/.venv/bin/python \
  research/python_ecosystem_audit/2026-08-10/build_audit.py
research/historical_learning_admission/.venv/bin/python \
  -m unittest discover -s research/python_ecosystem_audit/2026-08-10 -p 'test_*.py'
```

`build_audit.py` writes only this directory's generated JSON/JSONL artifacts.
The optional adapters record `not_installed` rather than silently falling back
to an external default. Rollback is deleting this research path or reverting
its dedicated commit; no runtime path imports it.

## Intake policy

OSS candidates are checked from official documentation/repository/release
metadata, not popularity. The ledger records license, Python/Apple-Silicon
compatibility, maintenance/CI evidence, deterministic behavior, dependencies,
security/operational concerns, and the exact failure mode at QuantRabbit's
truth boundary. No GitHub code is copied.


# Monthly 3x direct proof

This research path treats 200,000 JPY → 600,000 JPY after costs as the only success condition. It contains the frozen V1 multidimensional scan, the single-axis V2 completed-bar quality refinement, tests, and an independent oracle.

```bash
python3 research/monthly_3x_direct_proof/2026-08-10/run_direct_proof.py
python3 research/monthly_3x_direct_proof/2026-08-10/run_signal_quality_refine.py
python3 research/monthly_3x_direct_proof/2026-08-10/verify_independent_oracle.py
python3 -m pytest -q research/monthly_3x_direct_proof/2026-08-10/test_direct_proof.py
```

Research only: no holdout, live, Paper, broker order, or deploy access.

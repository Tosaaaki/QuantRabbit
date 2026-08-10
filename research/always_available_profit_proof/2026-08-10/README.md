# Always-available conditional profit proof

`ALWAYS_AVAILABLE_CONDITIONAL_PROFIT_V1` separates three claims that had been mixed together:

1. the exact profitable vehicle exists;
2. the engine always returns one answer (`TRADE` or `WAIT`);
3. a profitable trade is **not** available at every time.

The forward engine consumes only decision-time evidence. Realized outcomes are used only by the frozen historical existence certificate and never flow into a forward decision. `WAIT` is the executable answer whenever evidence, fillability, financing, margin, unwind, stability, or sample coverage is missing.

Run:

```bash
python3 research/always_available_profit_proof/2026-08-10/run_proof.py
python3 research/always_available_profit_proof/2026-08-10/verify_independent_oracle.py
python3 -m pytest -q research/always_available_profit_proof/2026-08-10/test_proof.py
```

This path is research-only. It does not read the sealed holdout or touch live, Paper, broker order, or deploy paths.

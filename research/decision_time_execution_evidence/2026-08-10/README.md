# DECISION_TIME_EXECUTION_EVIDENCE_LEDGER_V1

Research-only reconstruction of point-in-time execution evidence for the frozen
251-episode cohort. It reruns the frozen `FUSED_DECISION_V1` rule without
loosening any execution, cost, margin, or unwind constraint.

## Result

- Exact `ORDER_ACCEPTED` intent: 251/251.
- Causal OANDA S5 bid/ask pricing: 154/251.
- Causal pre-fill depth/fillability evidence: 153/251.
- Observed fills and closes: 251/251, isolated as evaluation-only.
- Complete decision-time fee/financing schedule: 0/251.
- Complete decision-time margin available/used/rate: 0/251.
- Executable decision-time unwind validity: 0/251.
- Strict eligible: 0/251; therefore the same-eligible-cohort financial comparison
  is `NOT_EVALUABLE`, not zero profit.
- Frozen 64-day weighted fusion: 22 predictions, all lower confidence bounds
  non-positive; this is edge insufficiency only on those modeled rows.

The evidence pipeline deficiency and the modeled-edge deficiency are separate.
The all-family fusion method itself remains unevaluated under a complete
execution-evidence cohort.

## Reproduce

```bash
python3 research/decision_time_execution_evidence/2026-08-10/build_ledger.py
python3 research/decision_time_execution_evidence/2026-08-10/verify_ledger.py
python3 research/decision_time_execution_evidence/2026-08-10/verify_financial.py
python3 -m unittest -v research/decision_time_execution_evidence/2026-08-10/test_ledger.py
```

No holdout, live, Paper, broker mutation, order, deployment, or dependency
installation is used. The forward acquisition contract records what future
decision-time receipts must exist instead of retroactively inventing evidence.

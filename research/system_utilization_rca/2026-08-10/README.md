# SYSTEM_UTILIZATION_RCA / FULL_INFERENCE_ENSEMBLE_V1

Research-only audit and fusion proof over the frozen 251 actual-after-cost
episodes. It does not import or call live, Paper, broker, order, or deploy paths.

Run:

```bash
python3 research/system_utilization_rca/2026-08-10/run_fusion.py
python3 -m unittest research/system_utilization_rca/2026-08-10/test_fusion.py -v
python3 research/system_utilization_rca/2026-08-10/verify_fusion.py
```

The inference and outcome tables are deliberately separate. Missing values are
null/absent sparse cells, never zero. `ALL_TRADES` is only the paired comparator;
the fused decision engine never silently falls back to it.

The current result is `HOLD_EVIDENCE_PIPELINE_BLOCKED`: a fused answer is emitted
for every episode, but no answer is an admissible `TRADE`, because historical
decision-time fillability/unwind evidence is absent and margin evidence covers
only 15/101 rows in 64-day VALIDATION. See `verdict_v1.md` for the decision.

# CAPITAL_PRESERVATION_GATE_V1

Research-only fail-closed permission layer for new exposure. It consumes only
decision-time inputs and emits a deterministic `TRADE`, `WAIT`, `SKIP`, or
`MANAGE` receipt. It never sends an order or grants live permission.

The policy permits new exposure only when all six execution-evidence stages are
complete, a TRAIN-fixed after-cost lower confidence bound is positive, the
candidate worst-case loss is known and within an equity-derived cap, the
non-refillable daily gross-loss budget remains available, and the drawdown lock
has not fired. Missing inputs remain missing; they are never treated as zero.

Run:

```bash
python3 research/capital_preservation_gate/2026-08-10/test_capital_preservation.py
python3 research/capital_preservation_gate/2026-08-10/run_replay.py
python3 research/capital_preservation_gate/2026-08-10/verify_replay.py
```

Outputs are dry-run receipts only. `receipts_v1.jsonl` is intentionally retained
as the audit path. The frozen historical cohort cannot demonstrate the valid
`TRADE` route because causal fee/financing, margin, unwind, equity, and bounded
loss evidence was not archived. The positive-route unit fixture proves the
policy is not an unconditional permanent ban.

Live-risk failure mode: wiring this research component directly into execution
without fresh broker equity, non-refillable loss-spend, cost schedule, margin,
and unwind evidence would either block everything or—if a caller substituted
defaults—silently widen risk. Runtime wiring is therefore explicitly out of
scope and forbidden by this contract.

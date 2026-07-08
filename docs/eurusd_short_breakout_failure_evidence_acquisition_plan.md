# EUR_USD SHORT BREAKOUT_FAILURE Evidence Acquisition Plan

Generated UTC: 2026-07-08T07:07:23Z

Status: `EVIDENCE_PLAN_READY`

This is a read-only evidence acquisition plan for `EUR_USD|SHORT|BREAKOUT_FAILURE`. It is not live permission. It sends no order, cancels no order, closes no position, modifies no TP/SL, and does not touch launchd.

## Current State

- Broad broker TP proof: `20` TAKE_PROFIT_ORDER wins, `0` TP losses after read-only reconciliation of legacy LIMIT rows `469278`, `469427`, and `469898`.
- Proof floor: `20`; broad TP remaining sample gap: `0`.
- Exact LIMIT/HARVEST S5 bid/ask replay: `4` wins, `0` losses, net expectancy `813.7734 JPY/trade`.
- Exact LIMIT/HARVEST sample floor remains underdone: `4/20`; remaining LIMIT-only sample gap: `16`.
- SCOUT recommendation remains evidence-acquisition only until exact LIMIT sample floor, S5 touch-lag reconciliation, operator review, risk, verifier, gateway, and negative-expectancy blockers clear.
- Current SCOUT status is `SCOUT_BLOCKED_OPERATOR_REVIEW`.
- `proof_queue_count=2`, `proof_ready_count=0`, and `can_create_live_permission_count=0`.
- Global capture economics is still `NEGATIVE_EXPECTANCY`.
- Month-scale TP-progress replay remains globally negative.
- Market-close leakage is present for this target shape.
- Positive exact LIMIT S5 bid/ask replay exists, but it is under-sampled and has touch-lag reconciliation blockers.

## Required Answers

1. Existing read-only history for the old broad TP sample gap:

   The old broad `17/0` gap is now reconciled. Existing duplicate-free legacy broker rows `469278`, `469427`, and `469898` were accepted as `TAKE_PROFIT_ORDER` evidence and `payoff_shape_diagnosis` now reports `20/0`, net `12865.8232 JPY`, expectancy `643.2912 JPY/trade`. This is broad broker TP proof only; it is not exact LIMIT-only live-grade proof.

2. If exact LIMIT history is insufficient, can live-SCOUTless work increase samples?

   Not as true broker LIMIT/HARVEST proof. Replay, paper, and dry-run evidence can support diagnosis, but they cannot count as new broker samples. Without uncounted existing exact LIMIT rows, new true samples require future fills, which this plan does not authorize.

3. Exact LIMIT S5 bid/ask replay:

   `data/eurusd_short_breakout_failure_limit_s5_bidask_replay.json` is positive for the exact `LIMIT` / `ATTACHED_TECHNICAL_TP` / `HARVEST` vehicle, but only on 4 samples. It also passes by touch order, not strict same-candle fill reconstruction, so S5 touch lag must remain visible.

4. Market-close leak split:

   Split the target shape into attached broker TP/HARVEST outcomes versus `MARKET_ORDER_TRADE_CLOSE` outcomes. The current target has `10` market-close losses for `-7636.3 JPY`. SCOUT proof must exclude those closes and must not rely on a market close, runner, cancel, or close path.

5. Month-scale negative:

   There is limited room because `harvest_live_grade_path` currently shows no direct target-shape month-scale blocker. That is not clearance. The 744h replay must become non-blocking, or a refreshed acceptance artifact must prove the exact target HARVEST exception is not blocked while keeping global negative evidence visible.

6. Transition after broad proof reconciliation:

   Broad TP proof completion moves the old sample gap to reconciled evidence. The active path still stays `EVIDENCE_ACQUISITION` because exact LIMIT-only sample floor, S5 touch-lag reconciliation, risk/verifier/gateway, guardian/operator review, global negative expectancy, market-close leak, and month-scale blockers remain. This is not live permission.

## Read-Only Command Plan

```bash
python3 -m json.tool data/eurusd_short_breakout_failure_evidence_acquisition_plan.json >/dev/null
PYTHONPATH=src python3 -m unittest tests.test_eurusd_short_breakout_failure_evidence_acquisition_plan -v
```

Check for uncounted existing exact LIMIT/HARVEST history:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli capture-economics
PYTHONPATH=src python3 -m quant_rabbit.cli payoff-shape-diagnosis --output data/payoff_shape_diagnosis.json --report docs/payoff_shape_diagnosis_report.md
```

Refresh month-scale and market-close leak evidence:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80
PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance
```

Refresh exact replay proof without changing runtime gates:

```bash
python3 -m json.tool data/eurusd_short_breakout_failure_limit_s5_bidask_replay.json >/dev/null
```

Rebuild read-only proof state after evidence refresh:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop
PYTHONPATH=src python3 -m quant_rabbit.cli as-4x-proof-path
PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator
PYTHONPATH=src python3 -m quant_rabbit.cli trader-goal-loop-orchestrator
PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract
```

## Success Conditions

- `take_profit_trades >= 20`.
- `take_profit_losses == 0`.
- Exact TP expectancy remains positive.
- Exact LIMIT/HARVEST spread-included replay remains positive.
- Exact LIMIT/HARVEST sample floor is not falsely marked complete while it remains `4/20`.
- Month-scale replay no longer blocks fresh entries or the target exception.
- Market-close leak is not used as proof and no longer blocks the target.
- Negative expectancy remains visible.
- Proof floor remains at least `20`.
- Operator/guardian clearance is explicit and not invented.
- `live_permission_allowed=false` remains true for this plan.

## Safety Boundary

Do not send, stage, cancel, close, modify TP/SL, change launchd, loosen gates, lower proof floor below 20, hide negative expectancy, hide month-scale negatives, hide market-close leakage, back-solve lots from 4x deficit, expose secrets, or invent operator decisions.

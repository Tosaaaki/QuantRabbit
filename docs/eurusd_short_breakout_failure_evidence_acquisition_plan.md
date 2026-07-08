# EUR_USD SHORT BREAKOUT_FAILURE Evidence Acquisition Plan

Generated UTC: 2026-07-08T00:20:25Z

Status: `EVIDENCE_PLAN_READY`

This is a read-only evidence acquisition plan for `EUR_USD|SHORT|BREAKOUT_FAILURE`. It is not live permission. It sends no order, cancels no order, closes no position, modifies no TP/SL, and does not touch launchd.

## Current State

- Exact TP proof: `17` TAKE_PROFIT_ORDER wins, `0` TP losses.
- Proof floor: `20`; remaining sample gap: `3`.
- SCOUT recommendation is still `SCOUT_REQUIRES_MORE_EVIDENCE`.
- Current SCOUT status is `SCOUT_BLOCKED_OPERATOR_REVIEW`.
- `proof_queue_count=0`, `proof_ready_count=0`, and `can_create_live_permission_count=0`.
- Global capture economics is still `NEGATIVE_EXPECTANCY`.
- Month-scale TP-progress replay remains globally negative.
- Market-close leakage is present for this target shape.
- Positive spread/slippage proof for the exact current vehicle is missing.

## Required Answers

1. Existing read-only history for the remaining 3 samples:

   Current artifacts do not prove that the remaining 3 exact TP samples already exist. The authoritative current count is `17/0`. The only read-only way to close this gap is to find already-existing unreconciled broker/legacy history via ledger sync/import and recompute capture/payoff diagnosis.

2. If existing history is insufficient, can live-SCOUTless work increase samples?

   Not as true broker TP proof. Replay, paper, and dry-run evidence can support spread proof or month-scale diagnosis, but they cannot count as new `TAKE_PROFIT_ORDER` proof-floor samples. Without uncounted existing history, new true samples require future live fills, which this plan does not authorize.

3. Spread proof missing:

   Run EUR_USD S5 bid/ask replay with spread included and attach only a positive, daily-stable result for the exact `LIMIT` / `ATTACHED_TECHNICAL_TP` / `HARVEST` vehicle. Do not overwrite runtime bidask rules from a candidate replay artifact.

4. Market-close leak split:

   Split the target shape into attached broker TP/HARVEST outcomes versus `MARKET_ORDER_TRADE_CLOSE` outcomes. The current target has `10` market-close losses for `-7636.3 JPY`. SCOUT proof must exclude those closes and must not rely on a market close, runner, cancel, or close path.

5. Month-scale negative:

   There is limited room because `harvest_live_grade_path` currently shows no direct target-shape month-scale blocker. That is not clearance. The 744h replay must become non-blocking, or a refreshed acceptance artifact must prove the exact target HARVEST exception is not blocked while keeping global negative evidence visible.

6. Transition after evidence completes:

   Evidence-only completion should move the operator-review classification from `SCOUT_REQUIRES_MORE_EVIDENCE` to `SCOUT_EVIDENCE_COMPLETE_OPERATOR_REVIEW_REQUIRED`. If an explicit operator/guardian clearance also exists and all evidence gates pass, the existing operator-review vocabulary can become `SCOUT_APPROVE_RECOMMENDED`. Neither state is live permission.

## Read-Only Command Plan

```bash
python3 -m json.tool data/eurusd_short_breakout_failure_evidence_acquisition_plan.json >/dev/null
PYTHONPATH=src python3 -m unittest tests.test_eurusd_short_breakout_failure_evidence_acquisition_plan -v
```

Check for uncounted existing TP history:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli execution-ledger-sync
PYTHONPATH=src python3 -m quant_rabbit.cli capture-economics
PYTHONPATH=src python3 -m quant_rabbit.cli payoff-shape-diagnosis --output data/payoff_shape_diagnosis.json --report docs/payoff_shape_diagnosis_report.md
```

Refresh month-scale and market-close leak evidence:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80
PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance
```

Acquire spread-included proof candidate without changing runtime gates:

```bash
PYTHONPATH=src python3 scripts/oanda_history_fetch.py --pairs EUR_USD --granularities S5 --price BA --days 45 --output-dir logs/replay/oanda_history
PYTHONPATH=src python3 scripts/oanda_history_replay_validate.py --pairs EUR_USD --granularity S5 --history-dir logs/replay/oanda_history --output-dir logs/reports/forecast_improvement/eurusd_short_breakout_failure_spread_probe
```

Rebuild read-only proof state after evidence refresh:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop
PYTHONPATH=src python3 -m quant_rabbit.cli as-4x-proof-path
PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator
PYTHONPATH=src python3 -m quant_rabbit.cli trader-goal-loop-orchestrator
```

## Success Conditions

- `take_profit_trades >= 20`.
- `take_profit_losses == 0`.
- Exact TP expectancy remains positive.
- Spread-included replay for the exact current vehicle is non-negative.
- Month-scale replay no longer blocks fresh entries or the target exception.
- Market-close leak is not used as proof and no longer blocks the target.
- Negative expectancy remains visible.
- Proof floor remains at least `20`.
- Operator/guardian clearance is explicit and not invented.
- `live_permission_allowed=false` remains true for this plan.

## Safety Boundary

Do not send, stage, cancel, close, modify TP/SL, change launchd, loosen gates, lower proof floor below 20, hide negative expectancy, hide month-scale negatives, hide market-close leakage, back-solve lots from 4x deficit, expose secrets, or invent operator decisions.

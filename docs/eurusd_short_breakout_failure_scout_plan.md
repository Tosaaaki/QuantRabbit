# EUR_USD SHORT BREAKOUT_FAILURE SCOUT Plan

Generated: 2026-07-07T12:59:09Z

## Verdict

Final judgement: `SCOUT_BLOCKED_OPERATOR_REVIEW`

`EUR_USD|SHORT|BREAKOUT_FAILURE` is the closest HARVEST proof path, but it is not in `proof_queue` and is not live-permission capable. The right design is a future `HARVEST_PROOF_COLLECTION_SCOUT`, not a live promotion. No order, cancel, close, launchd change, or gateway action is authorized by this plan.

## Direct Reason It Is Not In Proof Queue

- `data/as_proof_pack_queue.json`: `queue_count=0`, `proof_ready_count=0`, `can_create_live_permission_count=0`.
- `data/harvest_live_grade_path.json`: this shape has `actual_proof_queue_member=false`, `planner_can_enter_proof_pack=false`, `can_create_live_permission=false`, and `live_promotion_allowed=false`.
- Exact TP proof is positive but thin: `TAKE_PROFIT_ORDER` is `17` trades / `0` TP losses, with proof floor `20`, so the direct sample gap is `3`.
- Current matching intents are `DRY_RUN_BLOCKED`, including:
  - `RANGE_FORECAST_REQUIRES_RANGE_ROTATION`
  - `OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED`
  - `SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH`
  - `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`
- `proof_queue_count=0` is therefore a blocker/evidence state, not live permission.

## Missing Evidence

Sample size:
`17/0` exact TP samples are strong but below the `20`-trade proof floor. Need at least `3` more exact attached-TP HARVEST wins without adding TP losses.

Expectancy:
The exact TP vehicle is positive (`take_profit_expectancy_jpy=613.2`, pessimistic/positive-rotation `304.0708`), but global `capture_economics` is still `NEGATIVE_EXPECTANCY` (`expectancy=-177.4 JPY/trade`, `net=-40616.9 JPY`). This cannot be hidden.

Replay:
No direct month-scale blocker is shown for this exact shape in `harvest_live_grade_path`, but `profitability_acceptance` still has `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE` globally. If the next refresh creates a direct residual blocker for this shape, SCOUT becomes invalid.

Market-close leak:
The shape has `10` `MARKET_ORDER_TRADE_CLOSE` losses for `-7636.3 JPY`. Any proof path must avoid market-close leakage and use attached TP HARVEST only.

Guardian/operator:
`guardian_receipt_consumption.normal_routing_allowed=false` and `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` remains active. Operator review is required before any gateway path can be considered.

Fresh gateway packet:
`broker_snapshot` is from `2026-07-06T18:29:43Z`, `live_order_request` is `NO_ACTION`, and the last GPT decision is an old `CANCEL_PENDING` receipt. Fresh broker truth and a fresh GPT proof receipt are missing.

## SCOUT Design Value

Design value exists, but current execution permission does not.

The reason to keep the design is narrow: this is the shortest current HARVEST proof path (`17/0`, proof gap `3`), min-lot is numerically feasible, and no direct target month-scale blocker is shown. The reason not to execute is stronger right now: guardian/operator review, current forecast and self-improvement blockers, market-close leakage, stale/freshness gaps, and missing spread/slippage proof all remain.

## Proposed Contract If Unblocked

- Mode: `HARVEST_PROOF_COLLECTION_SCOUT`
- Vehicle: `LIMIT` only, exact `EUR_USD SHORT BREAKOUT_FAILURE`
- TP: `ATTACHED_TECHNICAL_TP`
- TP intent: `HARVEST`
- No `MARKET` scout.
- No failed-side `STOP` chase.
- No runner permission.
- `max_loss_jpy_cap`: `418.0`
- Normal live risk reference in the current intent packet: `2871.8597`; scout cap is deliberately smaller.
- Units: current feasible proof size `3000`, only if still above the `1000` unit production floor and still valid under fresh broker truth.
- Sizing must not be derived from `remaining_to_4x`.

This cap comes from the current loss-asymmetry guard / observed average-winner cap, not from the 4x deficit.

## Invalid Conditions

- Any live order send, cancel, close, position modification, or launchd mutation.
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` remains active.
- `normal_routing_allowed=false`.
- Fresh broker truth or a fresh GPT proof receipt is missing.
- Current blockers remain: `RANGE_FORECAST_REQUIRES_RANGE_ROTATION`, `SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH`, `OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED`, or unresolved chart-direction conflict.
- Vehicle is `MARKET`, failed-side `STOP`, runner, or not exact attached-TP HARVEST.
- `max_loss_jpy > 418.0`, units below `1000`, or lot derived from 4x shortfall.
- Spread/slippage or bid/ask replay is negative or missing.
- Direct month-scale residual blocker appears for this shape.
- Any planned proof depends on market close leakage.
- Any TP loss appears before proof_queue admission.
- Partial TP + runner is treated as permission rather than diagnostic replay.

## Withdrawal Conditions

- Stop pursuing SCOUT if the next refresh cannot reduce the exact TP sample gap, or if any exact TP loss appears before the proof floor.
- Stop pursuing SCOUT if spread-included replay stays missing or turns negative for the exact `LIMIT` attached-TP HARVEST vehicle.
- Stop pursuing SCOUT if market-close leakage cannot be contained without cancel/close/market-exit behavior.
- Stop pursuing SCOUT if a direct month-scale residual blocker appears for this target, or if global month-scale replay remains negative with no target-specific containment path.
- Stop pursuing SCOUT if guardian/operator review cannot set `normal_routing_allowed=true`.
- Stop pursuing SCOUT if fresh broker truth reprices the vehicle below `1000` units, above `418.0 JPY` max loss, or into adverse spread/slippage.
- Stop pursuing SCOUT if current market evidence keeps `RANGE` forecast without valid HARVEST geometry, `LONG` chart bias, or `SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH` as hard blockers.
- Stop pursuing SCOUT if proof collection would require deriving lot from `remaining_to_4x` or relaxing any gate.

## Evidence Success Conditions

- Exact TP samples: `take_profit_trades >= 20`.
- TP losses: `take_profit_losses == 0`.
- Exact TP expectancy remains positive.
- Wilson/pessimistic expectancy remains positive.
- Spread-included replay for the exact vehicle is non-negative.
- Market-close leak blocker is false or explicitly contained.
- No direct month-scale residual blocker exists for the target shape.
- Guardian/operator review clears with `normal_routing_allowed=true`.
- Fresh broker truth and fresh GPT proof receipt exist.
- The target becomes a real proof queue member.

## Next Read-Only Actions

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80
PYTHONPATH=src python3 -m quant_rabbit.cli capture-economics
PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance
PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop
PYTHONPATH=src python3 -m quant_rabbit.cli as-4x-proof-path
PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator
```

These are evidence-refresh actions only. They do not authorize live order routing.

## Partial TP + Runner

`partial TP + runner` remains diagnostic only. Current evidence has one runner-tail case (`trade_id=472819`) where a profit close left later runner upside, but `runner_candidates=0` and market-close leak is still active. This is not live permission.

## Safety Boundary

- `live_side_effects=[]`
- No live order.
- No cancel.
- No close.
- No launchd change.
- No gate relaxation.
- Negative expectancy remains visible.
- Month-scale replay blockers remain visible.
- `proof_queue_count=0` is not live permission.
- `live_promotion_allowed=false` means no order.
- Lot is not reverse-calculated from 4x shortfall.
- Secrets were not read or printed.

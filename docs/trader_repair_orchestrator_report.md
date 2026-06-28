# Trader Repair Orchestrator Report

- Generated at UTC: `2026-06-26T06:42:36.753218+00:00`
- Status: `ORCHESTRATOR_BLOCKED`
- Trader request: ``
- Selected request: `None`
- Actionable requests: `0`
- Approval-required requests: `0`
- Waiting requests: `3`
- Repair request source: `top_level_repair_requests`
- Recovered from embedded support: `False`
- Read only: `True`
- Live side effects: `0`

## Execution Contract

- Codex may execute: `read_artifacts, edit_code, edit_tests, edit_runtime_contract_docs, run_unit_tests, commit, sync_live_runtime`
- Approval required for: `order_send, order_cancel, position_close, launchd_load, launchd_reload`
- Forbidden direct actions: `direct_oanda_order_write, direct_oanda_trade_close, direct_launchd_mutation, model_api_call_from_quantrabbit_code`
- Commit and live sync required: `True`
- QuantRabbit code may call model API: `False`
- Policy: Order send, order cancel, position close, and launchd load/reload must go through explicit operator approval or the existing gateway path. This orchestrator is never live permission.

## Codex Work Order

- Status: `NO_ACTIONABLE_CODEX_WORK`
- Selection reason: None
- Objective: None
- Evidence summary keys: ``
- Deliverables: ``
- Final verification: ``
- Commit and live sync required: `None`
- Read-only until gateway or approval: `True`
- Approval required for: `order_send, order_cancel, position_close, launchd_load, launchd_reload`
- Existing gateway paths: `{"launchd_load_or_reload": "operator_explicit_approval_after_preflight", "order_cancel": "LiveOrderGateway", "order_send": "LiveOrderGateway", "position_close": "PositionProtectionGateway"}`

## Loop Engineering Prompt

- Version: `loop_engineering_prompt_v1`
- Objective: Drive QuantRabbit toward the daily 5% minimum from starting equity by repeatedly selecting the highest-causal blocker, taking only approved code/evidence actions, and verifying that operational reachability improves without bypassing broker truth.
- Hypothesis: The next blocker is evidence-window work: REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY (RESIDUAL_GROUPS_ALREADY_BLOCKED_WAITING_FOR_REPLAY). Do not edit gates until the clearance evidence changes.
- Operational 5pct reachable: `False`
- Audit 5pct estimated reachable: `True`
- Live-ready lanes: `0`
- Guardian active: `True`
- Artifact contradictions: ``
- Actionable: ``
- Approval required: ``
- Waiting: `REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY, REVIEW_CLOSE_GATE_EVIDENCE_FAILURES, REPAIR_FRONTIER_LANE_BLOCKER`
- Next loop: `Treat REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY as waiting for evidence, not implementation.; Run only the listed read-only verification/evidence commands and compare the new artifact against the clearance condition.; If the same blocker repeats unchanged, move to the next actionable request or report the live-evidence wait instead of rewriting the same guard.`
- Verification: `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80, PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit, PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot, PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts, PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator`

```text
QuantRabbit loop engineering prompt:

Goal: pursue the 5% daily minimum as an audit/repair obligation, not as a promise of market returns.
Trader request: (none)
State: orchestrator=ORCHESTRATOR_BLOCKED, target=PURSUE_TARGET, live_ready=0, guardian_active=True, guardian_lock=True, operational_5pct=False, audit_5pct=True.
Queue: selected=None, waiting_p0=REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY, REVIEW_CLOSE_GATE_EVIDENCE_FAILURES, support_blockers=SELF_IMPROVEMENT_P0_PRESENT, NO_LIVE_READY_LANES, PROFITABILITY_ACCEPTANCE_BLOCKED.
Execution frontier: repair_top=failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE blocked_by=BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE, PATTERN_REVERSAL_CHASE, EXHAUSTION_RANGE_CHASE proof=TP_PROVEN_HARVEST/tp_trades=20; frontier_blockers=BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE(1): failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE co_blocked_by=EXHAUSTION_RANGE_CHASE, PATTERN_REVERSAL_CHASE | EXHAUSTION_RANGE_CHASE(1): failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE co_blocked_by=BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE, PATTERN_REVERSAL_CHASE | PATTERN_REVERSAL_CHASE(1): failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE co_blocked_by=BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE, EXHAUSTION_RANGE_CHASE.
Profitability RCA: capture=NEGATIVE_EXPECTANCY; overall_exp_jpy=-164.6; overall_net_jpy=-37031.0; tp_exp_jpy=508.4; tp_net_jpy=48804.8; market_close_exp_jpy=-768.7; market_close_net_jpy=-74564.8; tp_market_close_leak_segments=1; acceptance_blockers=SELF_IMPROVEMENT_P0_PRESENT,NEGATIVE_EXPECTANCY_ACTIVE,MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE,RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK; operational_blockers=SELF_IMPROVEMENT_P0_PRESENT,NO_LIVE_READY_LANES,PROFITABILITY_ACCEPTANCE_BLOCKED,FRESH_ENTRY_SEND_NOT_ALLOWED.
Approval required details: (none).
Artifact contradictions: (none).
Hypothesis: The next blocker is evidence-window work: REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY (RESIDUAL_GROUPS_ALREADY_BLOCKED_WAITING_FOR_REPLAY). Do not edit gates until the clearance evidence changes.

Next loop:
- Treat REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY as waiting for evidence, not implementation.
- Run only the listed read-only verification/evidence commands and compare the new artifact against the clearance condition.
- If the same blocker repeats unchanged, move to the next actionable request or report the live-evidence wait instead of rewriting the same guard.

Self-review questions:
- What single blocker currently prevents operational 5% reachability, and is it causal rather than merely frequent?
- Does the next action change broker state, launchd state, order state, or position state? If yes, which approved gateway or explicit operator approval covers it?
- Is any blocker contradicted by fresher support state, and did I refresh the stale artifact before treating it as causal?
- Am I trying to recover profit by increasing churn while capture economics is negative, or by preserving a TP-proven HARVEST shape?
- What evidence would prove this loop iteration worked, and which command will refresh that evidence?
- Is a TP-proven lane blocked only by margin or broker exposure, and would clearing it require broker-state change outside Codex?
- What is the strongest counterargument that the current blocker is actually a correct protective guardrail?

Anti-loop rules:
- Do not treat audit-only 5% firepower as operational reachability unless operational_minimum_5pct_reachable is true.
- If the selected actionable request is lower priority than a waiting P0 blocker, treat it as auxiliary evidence work, not as clearing the P0 or proving operational 5%.
- If fresher support state contradicts intent blocker counts, classify that blocker as artifact-stale and refresh the evidence packet before selecting repair work from it.
- Do not rerun profitability-acceptance as the fix unless an input artifact, gateway proof, or live evidence window changed first.
- If OANDA audit-only S5/M5 history is complete and replay cannot clear local TP proof, do not rerun validate/mine/package; wait for new local TP receipts, new forecast/candle evidence, or exact HARVEST live-grade promotion.
- Do not lower MIN_PRODUCTION_LOT_UNITS, bypass MARGIN_TOO_THIN_FOR_MIN_LOT or LOSS_AND_MARGIN_TOO_THIN_FOR_MIN_LOT, synthesize PASS close evidence, or loosen protective market-structure guards without a failing regression and a positive-path test.
- Do not send orders, cancel orders, close positions, mutate launchd, or call model APIs from QuantRabbit code outside the existing gateway or explicit operator approval boundary.
- If the top item is waiting for live evidence, collect or wait for the named evidence; do not reimplement the same already-blocking guard.

Verification commands:
- PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80
- PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit
- PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot
- PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts
- PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator
```

## Queue

| Code | Priority | Automation | Match | Dependency | Verify |
|---|---|---|---:|---:|---|
| `REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY` | `P0` | `WAITING_FOR_LIVE_EVIDENCE_WINDOW` | `0` | `4` | `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80` |
| `REVIEW_CLOSE_GATE_EVIDENCE_FAILURES` | `P0` | `WAITING_FOR_LIVE_EVIDENCE_WINDOW` | `0` | `90` | `PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit` |
| `REPAIR_FRONTIER_LANE_BLOCKER` | `P1` | `WAITING_FOR_LIVE_EVIDENCE_WINDOW` | `0` | `5` | `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot`, `PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts` |

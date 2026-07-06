# Trader Repair Orchestrator Report

- Generated at UTC: `2026-07-06T15:10:58.506441+00:00`
- Status: `READY_FOR_CODEX_REPAIR`
- Trader request: ``
- Selected request: `REPAIR_FRONTIER_LANE_BLOCKER`
- Actionable requests: `1`
- Approval-required requests: `0`
- Waiting requests: `2`
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

- Status: `READY_FOR_CODEX_IMPLEMENTATION`
- Selection reason: Repair queue position is based on automation status, priority, dependency rank, request match, and code.
- Objective: After global support gates are removed, repair-frontier lanes still have lane-local blockers.
- Evidence summary keys: `co_blocker_codes, code, count, example_lane_ids, reward_jpy`
- Deliverables: `code_patch_or_documented_no_code_change, regression_tests_for_the_named_failure, positive_path_tests_for_the_allowed_shape, updated_runtime_contract_docs_when_behavior_changes, passing_targeted_tests, passing_required_verification_commands, passing_full_unittest_discover, git_commit_with_codex_attribution, verified_live_runtime_sync`
- Final verification: `PYTHONPATH=src python3 -m unittest tests.test_intent_generator -v, PYTHONPATH=src python3 -m unittest tests.test_trader_support_bot -v, PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot, PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts, PYTHONPATH=src python3 -m unittest discover -s tests -v, git status --short, scripts/sync-live-runtime.sh`
- Commit and live sync required: `True`
- Read-only until gateway or approval: `True`
- Approval required for: `order_send, order_cancel, position_close, launchd_load, launchd_reload`
- Existing gateway paths: `{"launchd_load_or_reload": "operator_explicit_approval_after_preflight", "order_cancel": "LiveOrderGateway", "order_send": "LiveOrderGateway", "position_close": "PositionProtectionGateway"}`

## Loop Engineering Prompt

- Version: `loop_engineering_prompt_v1`
- Objective: Drive QuantRabbit toward the daily 5% minimum from starting equity by repeatedly selecting the highest-causal blocker, taking only approved code/evidence actions, and verifying that operational reachability improves without bypassing broker truth.
- Hypothesis: The causal P0 blocker remains REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY waiting on evidence; the selected actionable loop is auxiliary work on REPAIR_FRONTIER_LANE_BLOCKER: After global support gates are removed, repair-frontier lanes still have lane-local blockers. Do not treat this as operational 5% proof until the waiting P0 clearance evidence changes.
- Operational 5pct reachable: `False`
- Audit 5pct estimated reachable: `True`
- Live-ready lanes: `0`
- Guardian active: `True`
- Artifact contradictions: ``
- Actionable: `REPAIR_FRONTIER_LANE_BLOCKER`
- Approval required: ``
- Waiting: `REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY, COLLECT_BIDASK_REPLAY_EVIDENCE`
- Next loop: `Keep the waiting P0 blockers in scope: REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY. The selected work is auxiliary until their clearance evidence changes.; Read the selected request evidence and suggested files before editing.; Implement the smallest code/test/doc change that directly clears the selected clearance condition.; Run targeted tests and listed verification commands; do not count report reruns as repair unless the underlying evidence changed.; Commit with Codex attribution, sync live runtime, and verify live HEAD matches the promoted commit.`
- Verification: `PYTHONPATH=src python3 -m unittest tests.test_intent_generator -v, PYTHONPATH=src python3 -m unittest tests.test_trader_support_bot -v, PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot, PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts, PYTHONPATH=src python3 -m unittest discover -s tests -v, git status --short, scripts/sync-live-runtime.sh, PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator`

```text
QuantRabbit loop engineering prompt:

Goal: pursue the 5% daily minimum as an audit/repair obligation, not as a promise of market returns.
Trader request: (none)
State: orchestrator=READY_FOR_CODEX_REPAIR, target=PURSUE_TARGET, live_ready=0, guardian_active=True, guardian_lock=False, operational_5pct=False, audit_5pct=True.
Queue: selected=REPAIR_FRONTIER_LANE_BLOCKER, waiting_p0=REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY, support_blockers=GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER, GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW, SELF_IMPROVEMENT_P0_PRESENT, NO_LIVE_READY_LANES, PROFITABILITY_ACCEPTANCE_BLOCKED.
Execution frontier: repair_top=failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT blocked_by=BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY proof=TP_PROVEN_HARVEST/tp_trades=20; frontier_blockers=GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY(4): failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT, failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE co_blocked_by=BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, LOSS_ASYMMETRY_GUARD_EXCEEDED | GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED(4): failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT, failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE co_blocked_by=BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY, LOSS_ASYMMETRY_GUARD_EXCEEDED | STRATEGY_NOT_ELIGIBLE(1): range_trader:AUD_JPY:LONG:RANGE_ROTATION co_blocked_by=BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, EXHAUSTION_RANGE_CHASE, GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY.
Profitability RCA: capture=NEGATIVE_EXPECTANCY; overall_exp_jpy=-176.2; overall_net_jpy=-41225.9; tp_exp_jpy=508.4; tp_net_jpy=48804.8; market_close_exp_jpy=-756.7; market_close_net_jpy=-74151.8; tp_market_close_leak_segments=1; acceptance_blockers=SELF_IMPROVEMENT_P0_PRESENT,NEGATIVE_EXPECTANCY_ACTIVE,MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE,MARKET_CLOSE_LEAK_FAMILY_BLOCKED; operational_blockers=GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING,GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER,GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW,SELF_IMPROVEMENT_P0_PRESENT.
Approval required details: (none).
Artifact contradictions: (none).
Hypothesis: The causal P0 blocker remains REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY waiting on evidence; the selected actionable loop is auxiliary work on REPAIR_FRONTIER_LANE_BLOCKER: After global support gates are removed, repair-frontier lanes still have lane-local blockers. Do not treat this as operational 5% proof until the waiting P0 clearance evidence changes.

Next loop:
- Keep the waiting P0 blockers in scope: REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY. The selected work is auxiliary until their clearance evidence changes.
- Read the selected request evidence and suggested files before editing.
- Implement the smallest code/test/doc change that directly clears the selected clearance condition.
- Run targeted tests and listed verification commands; do not count report reruns as repair unless the underlying evidence changed.
- Commit with Codex attribution, sync live runtime, and verify live HEAD matches the promoted commit.

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
- PYTHONPATH=src python3 -m unittest tests.test_intent_generator -v
- PYTHONPATH=src python3 -m unittest tests.test_trader_support_bot -v
- PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot
- PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts
- PYTHONPATH=src python3 -m unittest discover -s tests -v
- git status --short
- scripts/sync-live-runtime.sh
- PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator
```

## Queue

| Code | Priority | Repair status | Automation | Match | Dependency | Clearance | Verify |
|---|---|---|---|---:|---:|---|---|
| `REPAIR_FRONTIER_LANE_BLOCKER` | `P1` | `READY_FOR_CODE_OR_EVIDENCE_REPAIR` | `READY_FOR_CODEX_IMPLEMENTATION` | `0` | `5` | GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY disappears from repair_frontier_remaining_blockers or gains a tested downgrade path. | `PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot`, `PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts` |
| `REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY` | `P0` | `RESIDUAL_GROUPS_ALREADY_BLOCKED_WAITING_FOR_REPLAY` | `WAITING_FOR_LIVE_EVIDENCE_WINDOW` | `0` | `4` | month-scale production-gate replay is non-negative, or the top residual pair/side/method groups are removed by close-gate, TP-capture, or entry-selection changes before turnover is scaled | `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80` |
| `COLLECT_BIDASK_REPLAY_EVIDENCE` | `P1` | `BIDASK_REPLAY_WAITING_FOR_FORECAST_SAMPLE_COVERAGE` | `WAITING_FOR_LIVE_EVIDENCE_WINDOW` | `0` | `7` | OANDA bid/ask price truth is complete for loaded samples; collect more forecast_history samples across the under-sampled pair-directions, then rerun replay validation and require global all-currency sample coverage to graduate from UNDER_SAMPLED before claiming all-currency high-turn readiness | `python3 scripts/oanda_history_replay_validate.py --forecast-history data/forecast_history.jsonl --granularity S5 --history-dir logs/replay/oanda_history/20260703T072439Z --history-dir logs/replay/oanda_history/20260703T080331Z --history-dir logs/replay/oanda_history/20260703T120929Z --history-dir logs/replay/oanda_history/20260703T123013Z --history-dir logs/replay/oanda_history/20260703T134642Z --history-dir logs/replay/oanda_history/20260703T135559Z --history-dir logs/replay/oanda_history/20260703T142126Z --history-dir logs/replay/oanda_history/20260703T142653Z --history-dir logs/replay/oanda_history/20260703T143956Z --auto-history-min-days 30 --stable-min-active-days 3 --stable-max-daily-sample-share 0.7 --stable-min-positive-day-rate 0.6666666667` |

## Selected Request

- Code: `REPAIR_FRONTIER_LANE_BLOCKER`
- Dependency rank: `5`
- Selection reason: Repair queue position is based on automation status, priority, dependency rank, request match, and code.
- Problem: After global support gates are removed, repair-frontier lanes still have lane-local blockers.
- Why now: This is the next non-guardian blocker that would keep high-turn repair baskets from becoming executable.
- Suggested files: `src/quant_rabbit/strategy/intent_generator.py, scripts/oanda_history_replay_validate.py, tests/test_intent_generator.py, tests/test_trader_support_bot.py`
- Targeted tests: `PYTHONPATH=src python3 -m unittest tests.test_intent_generator -v, PYTHONPATH=src python3 -m unittest tests.test_trader_support_bot -v`
- Final verification: `PYTHONPATH=src python3 -m unittest tests.test_intent_generator -v, PYTHONPATH=src python3 -m unittest tests.test_trader_support_bot -v, PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot, PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts, PYTHONPATH=src python3 -m unittest discover -s tests -v, git status --short, scripts/sync-live-runtime.sh`

# A/S Lane Candidate Board

- Generated: `2026-07-08T07:07:00Z`
- Total lanes: `80`
- LIVE_READY lanes: `0`
- A/S LIVE_READY path exists: `False`
- Normal routing: `BLOCKED`

## 30D 4X Target Math

- Rolling 30d start equity: `171435.0552`
- Current raw / broker NAV: `270516.4198`
- Capital flows 30d: `100000.0`
- Funding-adjusted equity: `170516.4198`
- Funding-adjusted multiplier: `0.994641`
- Target from rolling start 4x: `685740.2208`
- Prompt-style current funding-adjusted 4x target: `682065.6792`
- Remaining to 4x funding-adjusted: `515223.801`
- Required calendar daily return: `5.633087`%

## Firepower Summary

- `total_order_intent_rows`: `80`
- `candidate_rows_after_hard_exclusions`: `51`
- `hard_excluded_rows`: `29`
- `rows_meeting_required_daily_return_prefilter`: `2`
- `can_create_live_permission_rows`: `0`
- `can_enter_proof_pack_rows`: `4`
- `as_live_ready_path_exists`: `False`
- `normal_routing_status`: `BLOCKED`
- `p0_dependency_count`: `4`

## Remaining P0 Dependency Graph

- `MEMORY_HEALTH_BLOCKED`: `ACTIVE_BLOCKER`; can create permission `False`
- `NEGATIVE_EXPECTANCY_ACTIVE`: `NEGATIVE_EXPECTANCY_REALIZED`; can create permission `False`
- `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE`: `MARKET_CLOSE_LEAK_FAMILY`; can create permission `False`
- `MARKET_CLOSE_LEAK_FAMILY_BLOCKED`: `MARKET_CLOSE_LEAK_FAMILY`; can create permission `False`

## Closest Candidate

- Lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Classification: `EVIDENCE_GAP`
- Proof distance: `7`
- Can create live permission: `False`

## Exact Blocker Preventing LIVE_READY

- Primary: `PROFITABILITY_ACCEPTANCE_BLOCKED`
- P0 rows: `MEMORY_HEALTH_BLOCKED, NEGATIVE_EXPECTANCY_ACTIVE, MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE, MARKET_CLOSE_LEAK_FAMILY_BLOCKED`
- Global blockers: `NEGATIVE_EXPECTANCY_ACTIVE, MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE, MARKET_CLOSE_LEAK_FAMILY_BLOCKED, SELF_IMPROVEMENT_P0_PRESENT, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE`
- A/S LIVE_READY stays zero: `True`

## New Evidence Loop Artifacts

- `data/remaining_profitability_p0_decomposition.json`
- `docs/remaining_profitability_p0_decomposition.md`
- `data/post_gate_capture_economics_decomposition.json`
- `docs/post_gate_capture_economics_decomposition.md`
- `data/post_gate_gap_family_repair_table.json`
- `docs/post_gate_gap_family_repair_table.md`
- `data/rolling_30d_4x_firepower_board.json`
- `docs/rolling_30d_4x_firepower_board.md`
- `data/as_proof_pack_queue.json`
- `docs/as_proof_pack_queue.md`
- `data/post_gate_expectancy_gap_trace.json`
- `docs/post_gate_expectancy_gap_trace.md`
- `data/historical_only_to_fresh_proof_replay.json`
- `docs/historical_only_to_fresh_proof_replay.md`
- `data/audjpy_short_breakout_failure_repair_proof.json`
- `docs/audjpy_short_breakout_failure_repair_proof.md`
- `data/audjpy_short_breakout_failure_limit_proof_pack.json`
- `docs/audjpy_short_breakout_failure_limit_proof_pack.md`
- `data/audjpy_limit_fresh_s5_bidask_replay.json`
- `docs/audjpy_limit_fresh_s5_bidask_replay.md`
- `data/manual_eurusd_tp_replacement_provenance.json`
- `docs/manual_eurusd_tp_replacement_provenance.md`
- `data/profitability_acceptance_blocker_reconciliation.json`
- `docs/profitability_acceptance_blocker_reconciliation.md`
- `data/portfolio_4x_path_planner.json`
- `docs/portfolio_4x_path_planner.md`

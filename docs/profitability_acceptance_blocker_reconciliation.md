# Profitability Acceptance Blocker Reconciliation

- Generated: `2026-07-06T09:04:10Z`
- Status: `PROFITABILITY_ACCEPTANCE_BLOCKED`
- Normal routing: `BLOCKED`
- A/S LIVE_READY path exists: `False`
- Can create live permission: `False`

| code | classification | current blocker | blocks fresh entries | clearance |
|---|---|---|---|---|
| `OPERATOR_MANUAL_TP_OPT_OUT_BYPASS` | `FIXED_NEEDS_CLEAN_WINDOW` | `False` | `True` | Keep a clean proof window where tp_rebalancer, PositionManager, and PositionProtectionGateway all preserve operator-manual packets with auto_tp_modify_allowed=false, and broker transaction IDs do not advance from unauthorized TP/SL/close writes. |
| `SELF_IMPROVEMENT_P0_PRESENT` | `ACTIVE_BLOCKER` | `True` | `True` | Regenerate memory-health and self-improvement-audit with zero P0 findings. |
| `NEGATIVE_EXPECTANCY_ACTIVE` | `ACTIVE_BLOCKER` | `True` | `True` | Accepted realized capture economics must become non-negative without promoting blocked historical families. |
| `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE` | `ACTIVE_BLOCKER` | `True` | `True` | TP-proven segments must no longer be net-damaged by unproven MARKET_ORDER_TRADE_CLOSE leakage. |
| `MARKET_CLOSE_LEAK_FAMILY_BLOCKED` | `TAXONOMY_DUPLICATE` | `True` | `True` | Clear only with exact close-gate proof, contained-risk timing evidence, and TP-proven exception evidence for the family. |
| `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE` | `CONTAINED_NOT_CLEARED` | `True` | `True` | Fresh 30-day TP-progress repair replay must be non-negative after the accepted filters and cannot rely on optimism. |
| `PROJECTION_HEADLINE_PRECISION_ECONOMIC_GAP` | `EVIDENCE_GAP` | `False` | `True` | Projection buckets used for live support must clear economic precision, not just headline hit-rate precision. |
| `BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN` | `EVIDENCE_GAP` | `False` | `True` | All-currency bid/ask replay coverage must be thick enough for the live-grade proof target. |
| `NO_LIVE_READY_TARGET_COVERAGE` | `EVIDENCE_GAP` | `False` | `True` | At least one lane must regenerate LIVE_READY with risk, gateway, GPT, guardian, telemetry, and acceptance proof. |
| `REPAIR_FRONTIER_BLOCKED` | `EVIDENCE_GAP` | `False` | `True` | Closest repair lanes must fill exact proof gaps; repair ranking does not create permission. |
| `EXECUTION_LEDGER_STALE` | `STALE_SUPERSEDED` | `False` | `False` | Keep execution ledger synchronized with broker snapshot before regenerating acceptance. |
| `RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK` | `STALE_SUPERSEDED` | `False` | `False` | If it reappears, reconcile the exact market-close receipt and keep the family blocked until proven. |
| `HISTORICAL_PROFIT_CAPTURE_MISSED` | `CONTAINED_NOT_CLEARED` | `False` | `True` | Treat historical capture misses as repair inputs only until post-repair replay and fresh proof are non-negative. |

## Boundary

No row in this reconciliation creates live permission. ACTIVE, fixed-needs-clean-window, contained, duplicate, and evidence-gap rows all keep normal routing blocked until acceptance, lane proof, RiskEngine, LiveOrderGateway, GPT verifier, and guardian gates pass together.

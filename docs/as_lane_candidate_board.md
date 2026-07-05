# A/S Lane Candidate Board

- Generated: `2026-07-05T16:24:00Z`
- Order intents generated: `2026-07-05T16:23:00.721047+00:00`
- Total lanes: `73`
- LIVE_READY lanes: `0`
- A/S LIVE_READY path exists: `False`
- Normal routing: `BLOCKED`

## USD_JPY

- Target lane present in current order_intents: `False`
- Present only in stale board: `True`
- A/S allowed: `False`
- LIVE_READY allowed: `False`
- Stale packaged rule excluded: `True`
- Blockers: `CURRENT_REPLAY_UNDER_SAMPLED, CURRENT_REPLAY_ACTIVE_DAYS_THIN, CURRENT_REPLAY_NEGATIVE_EXPECTANCY, CURRENT_REPLAY_PF_BELOW_BREAKEVEN, CURRENT_REPLAY_POSITIVE_DAY_RATE_LOW, MISSING_LOCAL_TP_SCOPE, FRESH_TARGET_LANE_ABSENT, STRATEGY_PROFILE_BLOCK_UNTIL_NEW_EVIDENCE, SELF_IMPROVEMENT_P0_PRESENT, NEGATIVE_EXPECTANCY_ACTIVE, MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE, MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE, GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING, GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING, NO_LIVE_READY_LANES, NO_FRESH_GPT_TRADE_ADD_RECEIPT`

## EUR_JPY SHORT

- Classification: `REJECTED_NEGATIVE_EXPECTANCY`
- A/S candidate: `False`
- LIVE_READY allowed: `False`
- Reason: `Broad EUR_JPY SHORT S5 and pair-shape evidence is negative; narrow positive confluences are post-hoc, audit-only, and below live-grade sample/Wilson/active-day requirements.`

## AUD_USD

- Classification: `REJECTED`
- A/S candidate: `False`
- LIVE_READY allowed: `False`
- Reason: `All current AUD_USD lanes are DRY_RUN_BLOCKED with negative-expectancy, month-scale residual, stale quote, spread, loss-budget, bid/ask replay, and guardian review blockers.`

## Month-Scale Residual Family Gate

- Family count: `23`
- Priority families: `7`
- Tail families: `16`
- Before filters improved P/L JPY: `-20863.5316`
- After filters residual P/L JPY: `2984.1927`
- Month-scale replay clears after filters: `True`
- All negative families can create live permission: `False`

| family | residual P/L | cause | blocker | action | A/S now | can ever A/S |
|---|---:|---|---|---|---|---|
| GBP_USD LONG BREAKOUT_FAILURE | -2981.8961 | `BAD_EXIT` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_LOCAL_TP_PROOF` | `False` | `True` |
| AUD_USD LONG RANGE_ROTATION | -2690.6967 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` |
| EUR_USD LONG RANGE_ROTATION | -2333.8215 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` |
| EUR_USD SHORT RANGE_ROTATION | -2181.1565 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` |
| NZD_CAD SHORT RANGE_ROTATION | -2044.4543 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` |
| AUD_USD SHORT RANGE_ROTATION | -1705.6738 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` |
| EUR_JPY LONG RANGE_ROTATION | -1071.9 | `RANGE_CHASE` | `MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED` | `REQUIRE_GEOMETRY_REPAIR` | `False` | `True` |

## Repair Analysis Artifacts

- `data/market_close_leak_trade_table.json`
- `docs/market_close_leak_trade_table.md`
- `data/month_scale_tp_replay_residuals.json`
- `docs/month_scale_tp_replay_residuals.md`
- `data/month_scale_residual_family_table.json`
- `docs/month_scale_residual_family_table.md`
- `data/historical_profit_capture_missed_table.json`
- `docs/historical_profit_capture_missed_table.md`

## Profitability Blockers

| blocker | attribution | clearing condition | fresh entries blocked |
|---|---|---|---|
| `NEGATIVE_EXPECTANCY_ACTIVE` | SYSTEM_LEDGER_AFTER_OPERATOR_MANUAL_EXCLUSION | capture_economics must be non-negative, or fresh entries must be restricted to exact attached-TP HARVEST repair lanes with local proof. | `True` |
| `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE` | SYSTEM_GATEWAY_ATTRIBUTED_ONLY; operator manual rows excluded | No TP-proven segment remains net-damaged by MARKET_ORDER_TRADE_CLOSE leakage. | `True` |
| `RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK` | CURRENT_ACCEPTANCE_ABSENT; stale mentions are not permission evidence | If re-raised, recent loss-side gateway market closes must be zero or have contained-risk timing plus durable close-gate proof. | `True` |
| `HISTORICAL_PROFIT_CAPTURE_MISSED` | SYSTEM_EXIT_MANAGEMENT_HISTORY; operator manual EUR_USD is excluded | Post-repair live evidence stays clean and the 744h replay residual becomes non-negative or ages out. | `True` |
| `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE` | SYSTEM_REPLAY_RESIDUALS; matching pair/side/method groups only are executable blockers | Rerun 744h execution-timing-audit and profitability-acceptance until replay is non-negative or matching residual groups disappear. | `True` |
| `BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN` | REPLAY_EVIDENCE_COVERAGE; not operator/manual | All-currency or target-lane S5 bid/ask evidence must meet sample, active-day, and daily-stability floors for the exact shape. | `True` |

## Next Best Candidates

- `profitability_acceptance_repair_frontier`: `blocked` - REPAIR_FRONTIER_BLOCKED remains active
- `EUR_JPY SHORT local TP proof`: `REJECTED_NEGATIVE_EXPECTANCY` - Broad EUR_JPY SHORT S5 and pair-shape evidence is negative; narrow positive confluences are post-hoc, audit-only, and below live-grade sample/Wilson/active-day requirements.
- `USD_JPY LONG BREAKOUT_FAILURE LIMIT`: `rejected_current_replay_negative_undersampled` - stale packaged rule excluded; fresh exact TP10/SL7 proof failed

## Shortest Path

- Status: `blocked_no_as_live_ready_lane`

Blocker hierarchy:
- No order_intents row is LIVE_READY.
- Normal routing remains BLOCKED.
- Profitability acceptance no longer raises MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE; residual-family gates still block every negative historical family from fresh A/S permission.
- Residual family table blocks every negative pair/side/method group from fresh A/S permission until exact proof clears the matching family.
- USD_JPY LONG BREAKOUT_FAILURE LIMIT remains rejected until fresh exact positive proof exists.
- EUR_JPY SHORT remains rejected/evidence-gap until spread-included non-negative proof exists.
- AUD_USD remains rejected by current DRY_RUN_BLOCKED lanes and month-scale residual blockers.
- Guardian/operator review and fresh GPT TRADE/ADD receipt requirements remain unfulfilled.

Required sequence:
- Keep USD_JPY stale packaged rule excluded; do not promote USD_JPY LONG without fresh exact TP proof.
- Repair EUR_USD LONG BREAKOUT_FAILURE market-close leakage or keep that lane out of fresh entry routing.
- Clear the 744h month-scale TP-progress replay residuals for matching pair/side/method groups.
- Find a pair/side/strategy with spread-included local TP proof, sufficient samples/days, and non-negative replay.
- Regenerate order_intents and require LIVE_READY plus RiskEngine, LiveOrderGateway, guardian/operator review, and GPT-5.5 TRADE/ADD receipt.

## Funding / Manual Safety

- Funding-adjusted 30d multiplier: `0.995949`
- Current equity raw equals broker NAV: `True`
- Funding-adjusted multiplier is KPI: `True`
- 100,000 JPY deposit excluded from performance: `True`
- EUR_USD `472987` classification: `OPERATOR_MANUAL`
- EUR_USD `472987` intent: `KEEP`
- System P/L counted: `False`
- System occupancy counted: `False`
- Auto close allowed: `False`
- Auto SL attach allowed: `False`
- Auto TP modify allowed: `False`


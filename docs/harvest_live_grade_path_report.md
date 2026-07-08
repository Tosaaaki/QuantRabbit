# HARVEST Live-Grade Path

Generated: `2026-07-08T07:19:30Z`

## Verdict

- Status: `DIAGNOSIS_COMPLETE_BLOCKED_NO_LIVE_PERMISSION`
- Live permission allowed: `False`
- Live promotion allowed: `False`
- Runner creation decision: `HARVEST_PROOF_FIRST`
- Closest HARVEST: `EUR_USD|SHORT|BREAKOUT_FAILURE`
- Proof queue: queue=`2` proof_ready=`0` can_create_live_permission=`0`
- Portfolio: `NO_LIVE_READY_PORTFOLIO` live_ready_lanes=`0`

This is a read-only diagnosis. It did not send orders, cancel orders, close positions, change launchd, relax gates, or derive lot size from the 4x deficit.

## Closest Candidate

- Shape: `EUR_USD|SHORT|BREAKOUT_FAILURE`
- Class: `HARVEST_PROOF_FLOOR_REACHED_EVIDENCE_ONLY`
- Proof queue member: `True`
- Planner can enter proof pack: `True`
- Can create live permission: `False`
- Rank reason: TP proof 20/20 gap 0; proof_queue_member=True; planner_can_enter_proof_pack=True; LIMIT S5 bid/ask replay 4/4 net=813.7734; overall_net_jpy=3705.2

## TP Proof

- Source: `data/eurusd_short_breakout_failure_proof_floor_update.json`
- TAKE_PROFIT_ORDER wins/trades: `20`
- TAKE_PROFIT_ORDER losses: `0`
- Proof floor: `20`
- Remaining broad TP gap: `0`
- Legacy samples accepted: `469278, 469427, 469898`
- Canonical integration: `CANONICAL_PROOF_UPDATE_READY_AS_EVIDENCE_ONLY`

## Exact LIMIT S5 Bid/Ask Replay

- Status: `LIMIT_S5_BIDASK_REPLAY_PASSED_STILL_BLOCKED`
- Samples: `4`
- Wins/Losses: `4` / `0`
- Net expectancy after bid/ask: `813.7734`
- Live-grade candidate: `False`

## Promotion Blockers

- `RANGE_FORECAST_REQUIRES_RANGE_ROTATION`: BLOCKING_LIVE_GRADE
- `OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED`: BLOCKING_LIVE_GRADE
- `SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH`: BLOCKING_LIVE_GRADE
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`: BLOCKING_LIVE_GRADE
- `MARKET_CLOSE_LEAK_PRESENT`: BLOCKING_LIVE_GRADE
- `LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY`: BLOCKING_LIVE_GRADE
- `ACTIVE_DAY_FLOOR_NOT_MET`: BLOCKING_LIVE_GRADE
- `RISK_ENGINE_PASS_MISSING`: BLOCKING_LIVE_GRADE
- `LIVE_ORDER_GATEWAY_PREFLIGHT_MISSING`: BLOCKING_LIVE_GRADE
- `FRESH_GPT_VERIFIER_TRADE_RECEIPT_MISSING`: BLOCKING_LIVE_GRADE
- `PROOF_QUEUE_MEMBER_BUT_NOT_PROOF_READY`: BLOCKING_LIVE_GRADE
- `S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION`: BLOCKING_LIVE_GRADE
- `NEGATIVE_EXPECTANCY_ACTIVE`: BLOCKING_LIVE_GRADE
- `MARKET_CLOSE_LEAK_PRESENT_EXCLUDED`: BLOCKING_LIVE_GRADE
- `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`: BLOCKING_LIVE_GRADE
- `NO_LIVE_READY_PORTFOLIO`: BLOCKING_LIVE_GRADE
- `NO_FRESH_GATEWAY_PERMISSION`: BLOCKING_LIVE_GRADE
- `SELF_IMPROVEMENT_P0_PRESENT`: BLOCKING_LIVE_GRADE
- `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE`: BLOCKING_LIVE_GRADE
- `MARKET_CLOSE_LEAK_FAMILY_BLOCKED`: BLOCKING_LIVE_GRADE
- `PROFITABILITY_ACCEPTANCE_BLOCKED`: BLOCKING_LIVE_GRADE
- `GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING`: BLOCKING_LIVE_GRADE
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING`: BLOCKING_LIVE_GRADE
- `NO_LIVE_READY_LANES`: BLOCKING_LIVE_GRADE
- `PORTFOLIO_PLANNER_CANNOT_CREATE_LIVE_PERMISSION`: BLOCKING_LIVE_GRADE
- `PROOF_QUEUE_NOT_PROOF_READY`: BLOCKING_LIVE_GRADE
- `PROOF_QUEUE_CANNOT_CREATE_LIVE_PERMISSION`: BLOCKING_LIVE_GRADE

## Ranked HARVEST Candidates

| Rank | Shape | Class | TP proof | Queue | Planner | Live |
|---:|---|---|---:|---|---|---|
| 1 | `EUR_USD|SHORT|BREAKOUT_FAILURE` | `HARVEST_PROOF_FLOOR_REACHED_EVIDENCE_ONLY` | 20/20 gap 0 | `True` | `True` | `False` |
| 2 | `EUR_USD|SHORT|TREND_CONTINUATION` | `HARVEST_POSITIVE_THIN_SAMPLE` | 5/20 gap 15 | `False` | `False` | `False` |
| 3 | `EUR_USD|LONG|BREAKOUT_FAILURE` | `HARVEST_POSITIVE_TP_PROVEN` | 20/20 gap 0 | `False` | `False` | `False` |
| 4 | `AUD_CAD|LONG|BREAKOUT_FAILURE` | `HARVEST_POSITIVE_THIN_SAMPLE` | 2/20 gap 18 | `False` | `False` | `False` |
| 5 | `GBP_USD|SHORT|BREAKOUT_FAILURE` | `HARVEST_POSITIVE_THIN_SAMPLE` | 2/20 gap 18 | `False` | `False` | `False` |
| 6 | `EUR_USD|SHORT|RANGE_ROTATION` | `HARVEST_POSITIVE_THIN_SAMPLE` | 10/20 gap 10 | `False` | `False` | `False` |
| 7 | `AUD_JPY|SHORT|BREAKOUT_FAILURE` | `HARVEST_POSITIVE_THIN_SAMPLE` | 6/20 gap 14 | `False` | `True` | `False` |
| 8 | `GBP_USD|LONG|RANGE_ROTATION` | `HARVEST_POSITIVE_THIN_SAMPLE` | 3/20 gap 17 | `False` | `False` | `False` |
| 9 | `EUR_USD|LONG|TREND_CONTINUATION` | `HARVEST_POSITIVE_THIN_SAMPLE` | 2/20 gap 18 | `False` | `False` | `False` |
| 10 | `AUD_JPY|LONG|BREAKOUT_FAILURE` | `HARVEST_POSITIVE_THIN_SAMPLE` | 2/20 gap 18 | `False` | `False` | `False` |
| 11 | `GBP_USD|LONG|BREAKOUT_FAILURE` | `HARVEST_POSITIVE_THIN_SAMPLE` | 10/20 gap 10 | `False` | `False` | `False` |
| 12 | `EUR_USD|LONG|RANGE_ROTATION` | `HARVEST_POSITIVE_THIN_SAMPLE` | 3/20 gap 17 | `False` | `False` | `False` |

## Next Read-Only Actions

- `canonicalize_limit_s5_bidask_replay`: target `EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST`; live_side_effects=[]
- `mine_additional_exact_limit_harvest_samples`: target `EUR_USD|SHORT|BREAKOUT_FAILURE`; live_side_effects=[]
- `refresh_profitability_and_goal_chain`: target `portfolio`; live_side_effects=[]

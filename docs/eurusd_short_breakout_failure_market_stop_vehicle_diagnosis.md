# EUR_USD SHORT BREAKOUT_FAILURE MARKET/STOP Vehicle Diagnosis

Generated: `2026-07-08T08:19:44Z`

## Verdict

- Status: `MARKET_STOP_VEHICLE_PROMISING_STILL_BLOCKED`
- Target shape: `EUR_USD|SHORT|BREAKOUT_FAILURE`
- Recommended active path: `EVIDENCE_ACQUISITION:STOP_HARVEST_EXACT_REPLAY_BEFORE_SCOUT_OR_OPERATOR_REVIEW`
- Live permission allowed: `False`
- Read-only: `True`

## Vehicle Split

- LIMIT baseline: `4` samples, remaining exact LIMIT gap `16`.
- MARKET_HARVEST: `9/0` net `3303.5031` JPY, status `MARKET_HARVEST_REPLAY_REQUIRED`.
- STOP_HARVEST: `7/0` net `6307.2357` JPY, status `STOP_HARVEST_REPLAY_REQUIRED`.
- Closer active path candidate: `STOP_HARVEST`.

MARKET_ORDER and STOP_ORDER samples are not mixed into LIMIT proof and are not merged with each other.

## Risk Boundary

- MARKET requested entry price available: `False`.
- MARKET risk defined for scout: `False`.
- STOP trigger price available: `True`.
- STOP risk defined for scout: `False`.

## Remaining Blockers

- `MARKET_HARVEST_EXACT_REPLAY_REQUIRED`: BLOCKING_MARKET_VEHICLE_PROOF
- `STOP_HARVEST_EXACT_REPLAY_REQUIRED`: BLOCKING_STOP_VEHICLE_PROOF
- `MARKET_ENTRY_RISK_SLIPPAGE_INVALIDATION_UNDEFINED`: BLOCKING_MARKET_SCOUT
- `STOP_TRIGGER_SLIPPAGE_INVALIDATION_UNDEFINED`: BLOCKING_STOP_SCOUT
- `LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY`: VISIBLE_LIMIT_BASELINE_BLOCKER
- `MARKET_STOP_NOT_ALLOWED_IN_LIMIT_PROOF`: PROOF_BOUNDARY
- `NO_LIVE_PERMISSION_CREATED`: SAFETY_BOUNDARY
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`: VISIBLE_EXISTING_BLOCKER
- `MARKET_CLOSE_LEAK_PRESENT`: VISIBLE_EXISTING_BLOCKER
- `PROOF_QUEUE_MEMBER_BUT_NOT_PROOF_READY`: VISIBLE_EXISTING_BLOCKER
- `NEGATIVE_EXPECTANCY_ACTIVE`: VISIBLE_EXISTING_BLOCKER
- `MARKET_CLOSE_LEAK_PRESENT_EXCLUDED`: VISIBLE_EXISTING_BLOCKER
- `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`: VISIBLE_EXISTING_BLOCKER
- `NO_LIVE_READY_PORTFOLIO`: VISIBLE_EXISTING_BLOCKER
- `NO_FRESH_GATEWAY_PERMISSION`: VISIBLE_EXISTING_BLOCKER
- `PORTFOLIO_PLANNER_CANNOT_CREATE_LIVE_PERMISSION`: VISIBLE_EXISTING_BLOCKER

## Next Read-Only Actions

- Build exact STOP_HARVEST S5 bid/ask trigger and TP replay from the 7 STOP_ORDER samples.
- Define STOP_HARVEST invalidation, max trigger slippage, and stop-chase rejection rules before scout review.
- Build a MARKET_HARVEST packet with requested entry quote, max slippage, invalidation, and TP/SL geometry before replay.
- Keep LIMIT_HARVEST proof isolated at 4/20; do not import MARKET or STOP samples into the LIMIT floor.
- After exact replay packets exist, rerun proof queue, 4x planner, active contract, and operator-review material read-only.

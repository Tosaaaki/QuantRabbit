# Post-Trade Learning Report

- Generated at UTC: `2026-05-01T02:41:34.237949+00:00`
- Status: `READY_FOR_REVIEW`
- Candidates: `1`
- Profile update candidates: `0`

## Blockers

- none

## Candidates

- `live_order` lane=`failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` pair=`EUR_USD LONG` pl=`None` recommendation=`NO_PROFILE_CHANGE`
  - reason: entry receipt exists but no close/fill outcome was supplied
  - refs: lane:failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE, live_order_request, position_execution, trader_decision

## Learning Contract

- Learning memory is advisory and cannot force entries, suppress exits, or resize trades.
- Profile changes are candidates only until backed by receipts and validated by live risk gates.
- Losses beyond the current cap become blockers, not prompt-only lessons.

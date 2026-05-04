# Coverage Optimization Report

- Generated at UTC: `2026-05-04T16:25:29.944632+00:00`
- Status: `COVERAGE_GAP`
- Remaining target: `19196 JPY`
- Live-ready reward: `0 JPY` (`0.0%`)
- Sequential ladder reward: `0 JPY` (`0.0%`, steps=`0`)
- Potential reward after promotions: `0 JPY` (`0.0%`)
- Remaining risk budget: `4202 JPY`

## Blockers

- live-ready reward misses remaining target by 19196 JPY
- even promoted dry-run reward misses remaining target by 19196 JPY
- no LIVE_READY lanes exist
- replay evidence covers target on 3/50 days

## Action Items

- build at least 5 additional live-ready trigger receipts
- expand lane generation across timing windows or pairs; current repaired ladder cannot cover target
- repair blockers for: AUD_JPY, EUR_JPY, EUR_USD, GBP_USD
- rerun replay/backtest after coverage changes and keep gap reasons as product blockers

## Lanes

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`7680` risk=`960` rr=`8.00` live_ready=`False` promotion_candidate=`False`
  - blocker: AUD_JPY quote is stale: 48.4s > 20s
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`1899` risk=`972` rr=`1.95` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY quote is stale: 48.4s > 20s
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`4928` risk=`1038` rr=`4.75` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_USD quote is stale: 49.5s > 20s
  - blocker: USD_JPY conversion quote is stale: 48.4s > 20s
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`6225` risk=`1038` rr=`6.00` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_USD quote is stale: 49.5s > 20s
  - blocker: USD_JPY conversion quote is stale: 48.4s > 20s
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`2069` risk=`935` rr=`2.21` live_ready=`False` promotion_candidate=`False`
  - blocker: GBP_USD quote is stale: 48.6s > 20s
  - blocker: USD_JPY conversion quote is stale: 48.4s > 20s
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`7680` risk=`960` rr=`8.00` live_ready=`False` promotion_candidate=`False`
  - blocker: AUD_JPY quote is stale: 48.4s > 20s
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`1899` risk=`972` rr=`1.95` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY quote is stale: 48.4s > 20s
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`4928` risk=`1038` rr=`4.75` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_USD quote is stale: 49.5s > 20s
  - blocker: USD_JPY conversion quote is stale: 48.4s > 20s
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`6225` risk=`1038` rr=`6.00` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_USD quote is stale: 49.5s > 20s
  - blocker: USD_JPY conversion quote is stale: 48.4s > 20s
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`2069` risk=`935` rr=`2.21` live_ready=`False` promotion_candidate=`False`
  - blocker: GBP_USD quote is stale: 48.6s > 20s
  - blocker: USD_JPY conversion quote is stale: 48.4s > 20s
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`7680` risk=`960` rr=`8.00` live_ready=`False` promotion_candidate=`False`
  - blocker: AUD_JPY quote is stale: 48.4s > 20s
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`1899` risk=`972` rr=`1.95` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY quote is stale: 48.4s > 20s
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`4928` risk=`1038` rr=`4.75` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_USD quote is stale: 49.5s > 20s
  - blocker: USD_JPY conversion quote is stale: 48.4s > 20s
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`6225` risk=`1038` rr=`6.00` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_USD quote is stale: 49.5s > 20s
  - blocker: USD_JPY conversion quote is stale: 48.4s > 20s
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`2069` risk=`935` rr=`2.21` live_ready=`False` promotion_candidate=`False`
  - blocker: GBP_USD quote is stale: 48.6s > 20s
  - blocker: USD_JPY conversion quote is stale: 48.4s > 20s

## Coverage Contract

- Coverage is executable reward from current receipts, not a profit guarantee.
- `DRY_RUN_PASSED` lanes count only as potential coverage until strategy blockers are promoted by receipts.
- A target gap remains a product blocker until it is closed by live-ready, risk-valid lanes or a no-market gap receipt.

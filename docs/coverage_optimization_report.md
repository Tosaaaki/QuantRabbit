# Coverage Optimization Report

- Generated at UTC: `2026-05-04T14:19:20.881981+00:00`
- Status: `COVERAGE_GAP`
- Remaining target: `19012 JPY`
- Live-ready reward: `0 JPY` (`0.0%`)
- Sequential ladder reward: `0 JPY` (`0.0%`, steps=`0`)
- Potential reward after promotions: `0 JPY` (`0.0%`)
- Remaining risk budget: `3202 JPY`

## Blockers

- live-ready reward misses remaining target by 19012 JPY
- even promoted dry-run reward misses remaining target by 19012 JPY
- no LIVE_READY lanes exist
- replay evidence covers target on 3/50 days

## Action Items

- build at least 5 additional live-ready trigger receipts
- expand lane generation across timing windows or pairs; current repaired ladder cannot cover target
- repair blockers for: AUD_JPY, EUR_JPY, EUR_USD, GBP_USD
- rerun replay/backtest after coverage changes and keep gap reasons as product blockers

## Lanes

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`7680` risk=`960` rr=`8.00` live_ready=`False` promotion_candidate=`False`
  - blocker: AUD_JPY quote is stale: 40.2s > 20s
  - blocker: open risk 1001 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`1872` risk=`960` rr=`1.95` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY quote is stale: 40.3s > 20s
  - blocker: EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
  - blocker: open risk 1001 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`4656` risk=`980` rr=`4.75` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_USD quote is stale: 40.5s > 20s
  - blocker: USD_JPY conversion quote is stale: 40.6s > 20s
  - blocker: open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`5881` risk=`980` rr=`6.00` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_USD quote is stale: 40.5s > 20s
  - blocker: USD_JPY conversion quote is stale: 40.6s > 20s
  - blocker: open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`2174` risk=`980` rr=`2.22` live_ready=`False` promotion_candidate=`False`
  - blocker: GBP_USD quote is stale: 40.3s > 20s
  - blocker: USD_JPY conversion quote is stale: 40.6s > 20s
  - blocker: open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`7680` risk=`960` rr=`8.00` live_ready=`False` promotion_candidate=`False`
  - blocker: AUD_JPY quote is stale: 40.2s > 20s
  - blocker: open risk 1001 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`1872` risk=`960` rr=`1.95` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY quote is stale: 40.3s > 20s
  - blocker: EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
  - blocker: open risk 1001 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`4656` risk=`980` rr=`4.75` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_USD quote is stale: 40.5s > 20s
  - blocker: USD_JPY conversion quote is stale: 40.6s > 20s
  - blocker: open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`5881` risk=`980` rr=`6.00` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_USD quote is stale: 40.5s > 20s
  - blocker: USD_JPY conversion quote is stale: 40.6s > 20s
  - blocker: open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`2174` risk=`980` rr=`2.22` live_ready=`False` promotion_candidate=`False`
  - blocker: GBP_USD quote is stale: 40.3s > 20s
  - blocker: USD_JPY conversion quote is stale: 40.6s > 20s
  - blocker: open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`7680` risk=`960` rr=`8.00` live_ready=`False` promotion_candidate=`False`
  - blocker: AUD_JPY quote is stale: 40.2s > 20s
  - blocker: open risk 1001 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`1872` risk=`960` rr=`1.95` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY quote is stale: 40.3s > 20s
  - blocker: EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
  - blocker: open risk 1001 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`4656` risk=`980` rr=`4.75` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_USD quote is stale: 40.5s > 20s
  - blocker: USD_JPY conversion quote is stale: 40.6s > 20s
  - blocker: open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`5881` risk=`980` rr=`6.00` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_USD quote is stale: 40.5s > 20s
  - blocker: USD_JPY conversion quote is stale: 40.6s > 20s
  - blocker: open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`2174` risk=`980` rr=`2.22` live_ready=`False` promotion_candidate=`False`
  - blocker: GBP_USD quote is stale: 40.3s > 20s
  - blocker: USD_JPY conversion quote is stale: 40.6s > 20s
  - blocker: open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY

## Coverage Contract

- Coverage is executable reward from current receipts, not a profit guarantee.
- `DRY_RUN_PASSED` lanes count only as potential coverage until strategy blockers are promoted by receipts.
- A target gap remains a product blocker until it is closed by live-ready, risk-valid lanes or a no-market gap receipt.

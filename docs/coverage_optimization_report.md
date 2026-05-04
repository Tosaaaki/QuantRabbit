# Coverage Optimization Report

- Generated at UTC: `2026-05-04T08:27:20.937998+00:00`
- Status: `COVERAGE_GAP`
- Remaining target: `21011 JPY`
- Live-ready reward: `20568 JPY` (`97.9%`)
- Sequential ladder reward: `20568 JPY` (`97.9%`, steps=`12`)
- Potential reward after promotions: `20568 JPY` (`97.9%`)
- Remaining risk budget: `4202 JPY`

## Blockers

- live-ready reward misses remaining target by 443 JPY
- even promoted dry-run reward misses remaining target by 443 JPY
- live-ready ladder risk exceeds remaining daily risk budget and sequential coverage still misses target
- replay evidence covers target on 3/50 days

## Action Items

- build at least 1 additional live-ready trigger receipts
- expand lane generation across timing windows or pairs; current repaired ladder cannot cover target
- repair blockers for: AUD_JPY
- rerun replay/backtest after coverage changes and keep gap reasons as product blockers

## Lanes

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`3024` risk=`378` rr=`8.00` live_ready=`False` promotion_candidate=`False`
  - blocker: AUD_JPY spread 2.1pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`2147` risk=`452` rr=`4.75` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`2712` risk=`452` rr=`6.00` live_ready=`True` promotion_candidate=`False`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`3024` risk=`378` rr=`8.00` live_ready=`False` promotion_candidate=`False`
  - blocker: AUD_JPY spread 2.1pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`2147` risk=`452` rr=`4.75` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY` reward=`2712` risk=`452` rr=`6.00` live_ready=`True` promotion_candidate=`False`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`3024` risk=`378` rr=`8.00` live_ready=`False` promotion_candidate=`False`
  - blocker: AUD_JPY spread 2.1pip exceeds 2.5x normal 0.8pip
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`2147` risk=`452` rr=`4.75` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY` reward=`2712` risk=`452` rr=`6.00` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`912` risk=`468` rr=`1.95` live_ready=`True` promotion_candidate=`False`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`1086` risk=`490` rr=`2.22` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`912` risk=`468` rr=`1.95` live_ready=`True` promotion_candidate=`False`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`1086` risk=`490` rr=`2.22` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`912` risk=`468` rr=`1.95` live_ready=`True` promotion_candidate=`False`
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`1086` risk=`490` rr=`2.22` live_ready=`True` promotion_candidate=`False`

## Coverage Contract

- Coverage is executable reward from current receipts, not a profit guarantee.
- `DRY_RUN_PASSED` lanes count only as potential coverage until strategy blockers are promoted by receipts.
- A target gap remains a product blocker until it is closed by live-ready, risk-valid lanes or a no-market gap receipt.

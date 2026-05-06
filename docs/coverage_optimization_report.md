# Coverage Optimization Report

- Generated at UTC: `2026-05-06T12:34:17.186859+00:00`
- Status: `COVERAGE_GAP`
- Remaining target: `20895 JPY`
- Live-ready reward: `565 JPY` (`2.7%`)
- Unique live-ready lanes: `14` (duplicates removed=`8`)
- Sequential ladder reward: `565 JPY` (`2.7%`, steps=`14`)
- Potential reward after promotions: `565 JPY` (`2.7%`)
- Remaining risk budget: `4202 JPY`

## Blockers

- live-ready reward misses remaining target by 20329 JPY
- even promoted dry-run reward misses remaining target by 20329 JPY
- replay evidence covers target on 4/50 days

## Action Items

- dedupe same entry/tp/sl receipts before considering multi-entry execution
- build at least 473 additional live-ready trigger receipts
- expand lane generation across timing windows or pairs; current repaired ladder cannot cover target
- repair blockers for: EUR_JPY, EUR_USD, GBP_USD
- rerun replay/backtest after coverage changes and keep gap reasons as product blockers

## Lanes

- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`52` risk=`18` rr=`2.84` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:MARKET` status=`LIVE_READY` reward=`52` risk=`18` rr=`2.84` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY` reward=`36` risk=`18` rr=`1.96` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION:MARKET` status=`LIVE_READY` reward=`31` risk=`18` rr=`1.68` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY` reward=`52` risk=`18` rr=`2.84` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET` status=`LIVE_READY` reward=`52` risk=`18` rr=`2.84` live_ready=`True` promotion_candidate=`False`
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`70` risk=`18` rr=`3.83` live_ready=`True` promotion_candidate=`False`
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE:MARKET` status=`LIVE_READY` reward=`70` risk=`18` rr=`3.83` live_ready=`True` promotion_candidate=`False`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`31` risk=`18` rr=`1.68` live_ready=`True` promotion_candidate=`False`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION:MARKET` status=`LIVE_READY` reward=`25` risk=`18` rr=`1.34` live_ready=`True` promotion_candidate=`False`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`70` risk=`18` rr=`3.83` live_ready=`True` promotion_candidate=`False`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION:MARKET` status=`LIVE_READY` reward=`70` risk=`18` rr=`3.83` live_ready=`True` promotion_candidate=`False`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`28` risk=`18` rr=`1.51` live_ready=`True` promotion_candidate=`False`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:MARKET` status=`LIVE_READY` reward=`28` risk=`18` rr=`1.51` live_ready=`True` promotion_candidate=`False`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`27` risk=`18` rr=`1.50` live_ready=`True` promotion_candidate=`False`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION:MARKET` status=`DRY_RUN_BLOCKED` reward=`28` risk=`18` rr=`1.51` live_ready=`False` promotion_candidate=`False`
  - blocker: GBP_USD range MARKET lane is not inside the rail zone; keep the pending LIMIT rail order instead of chasing the box interior.
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`28` risk=`18` rr=`1.51` live_ready=`True` promotion_candidate=`False`
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION:MARKET` status=`LIVE_READY` reward=`28` risk=`18` rr=`1.51` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`27` risk=`18` rr=`1.50` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE:MARKET` status=`DRY_RUN_BLOCKED` reward=`27` risk=`18` rr=`1.50` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`27` risk=`18` rr=`1.50` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION:MARKET` status=`DRY_RUN_BLOCKED` reward=`27` risk=`18` rr=`1.50` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
  - blocker: EUR_JPY range MARKET lane is not inside the rail zone; keep the pending LIMIT rail order instead of chasing the box interior.
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`27` risk=`18` rr=`1.50` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION:MARKET` status=`DRY_RUN_BLOCKED` reward=`27` risk=`18` rr=`1.50` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`41` risk=`18` rr=`2.27` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET` status=`LIVE_READY` reward=`42` risk=`18` rr=`2.27` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`34` risk=`18` rr=`1.88` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION:MARKET` status=`DRY_RUN_BLOCKED` reward=`42` risk=`18` rr=`2.27` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_USD range MARKET lane is not inside the rail zone; keep the pending LIMIT rail order instead of chasing the box interior.
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`41` risk=`18` rr=`2.27` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET` status=`LIVE_READY` reward=`42` risk=`18` rr=`2.27` live_ready=`True` promotion_candidate=`False`

## Coverage Contract

- Coverage is executable reward from current receipts, not a profit guarantee.
- `DRY_RUN_PASSED` lanes count only as potential coverage until strategy blockers are promoted by receipts.
- A target gap remains a product blocker until it is closed by live-ready, risk-valid lanes or a no-market gap receipt.

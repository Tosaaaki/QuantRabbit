# Coverage Optimization Report

- Generated at UTC: `2026-05-06T03:22:32.043428+00:00`
- Status: `COVERAGE_REQUIRES_REPLAY_EVIDENCE`
- Remaining target: `20895 JPY`
- Live-ready reward: `25808 JPY` (`123.5%`)
- Sequential ladder reward: `21864 JPY` (`104.6%`, steps=`9`)
- Potential reward after promotions: `25808 JPY` (`123.5%`)
- Remaining risk budget: `4202 JPY`

## Blockers

- replay evidence covers target on 4/50 days

## Action Items

- execute coverage as a sequential ladder; do not deploy all live-ready lanes as simultaneous exposure
- repair blockers for: EUR_JPY
- rerun replay/backtest after coverage changes and keep gap reasons as product blockers

## Lanes

- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`2817` risk=`987` rr=`2.85` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY` reward=`1398` risk=`987` rr=`1.42` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY` reward=`2817` risk=`987` rr=`2.85` live_ready=`True` promotion_candidate=`False`
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`3680` risk=`960` rr=`3.83` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`1440` risk=`960` rr=`1.50` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
  - blocker: EUR_JPY LONG requires trigger/pending-entry receipts before live use: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`2221` risk=`987` rr=`2.25` live_ready=`True` promotion_candidate=`False`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`1480` risk=`987` rr=`1.50` live_ready=`True` promotion_candidate=`False`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`1230` risk=`960` rr=`1.28` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`808` risk=`960` rr=`0.84` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
  - blocker: planned reward/risk 0.84x is below 1.20x
  - blocker: EUR_JPY LONG requires trigger/pending-entry receipts before live use: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`1316` risk=`987` rr=`1.33` live_ready=`True` promotion_candidate=`False`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`1468` risk=`987` rr=`1.49` live_ready=`True` promotion_candidate=`False`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`3680` risk=`960` rr=`3.83` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`1440` risk=`960` rr=`1.50` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
  - blocker: EUR_JPY LONG requires trigger/pending-entry receipts before live use: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`2221` risk=`987` rr=`2.25` live_ready=`True` promotion_candidate=`False`
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`1480` risk=`987` rr=`1.50` live_ready=`True` promotion_candidate=`False`

## Coverage Contract

- Coverage is executable reward from current receipts, not a profit guarantee.
- `DRY_RUN_PASSED` lanes count only as potential coverage until strategy blockers are promoted by receipts.
- A target gap remains a product blocker until it is closed by live-ready, risk-valid lanes or a no-market gap receipt.

# Coverage Optimization Report

- Generated at UTC: `2026-05-04T18:05:31.882909+00:00`
- Status: `COVERAGE_REQUIRES_REPLAY_EVIDENCE`
- Remaining target: `19196 JPY`
- Live-ready reward: `61207 JPY` (`318.9%`)
- Sequential ladder reward: `23040 JPY` (`120.0%`, steps=`3`)
- Potential reward after promotions: `61207 JPY` (`318.9%`)
- Remaining risk budget: `4202 JPY`

## Blockers

- replay evidence covers target on 3/50 days

## Action Items

- execute coverage as a sequential ladder; do not deploy all live-ready lanes as simultaneous exposure
- repair blockers for: EUR_JPY
- rerun replay/backtest after coverage changes and keep gap reasons as product blockers

## Lanes

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`7680` risk=`960` rr=`8.00` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`1883` risk=`966` rr=`1.95` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY spread 2.3pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`4660` risk=`981` rr=`4.75` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`5886` risk=`981` rr=`6.00` live_ready=`True` promotion_candidate=`False`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`2176` risk=`981` rr=`2.22` live_ready=`True` promotion_candidate=`False`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`7680` risk=`960` rr=`8.00` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`1883` risk=`966` rr=`1.95` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY spread 2.3pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`4660` risk=`981` rr=`4.75` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY` reward=`5886` risk=`981` rr=`6.00` live_ready=`True` promotion_candidate=`False`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`2176` risk=`981` rr=`2.22` live_ready=`True` promotion_candidate=`False`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`7680` risk=`960` rr=`8.00` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`1883` risk=`966` rr=`1.95` live_ready=`False` promotion_candidate=`False`
  - blocker: EUR_JPY spread 2.3pip exceeds 2.5x normal 0.8pip
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`4660` risk=`981` rr=`4.75` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY` reward=`5886` risk=`981` rr=`6.00` live_ready=`True` promotion_candidate=`False`
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`2176` risk=`981` rr=`2.22` live_ready=`True` promotion_candidate=`False`

## Coverage Contract

- Coverage is executable reward from current receipts, not a profit guarantee.
- `DRY_RUN_PASSED` lanes count only as potential coverage until strategy blockers are promoted by receipts.
- A target gap remains a product blocker until it is closed by live-ready, risk-valid lanes or a no-market gap receipt.

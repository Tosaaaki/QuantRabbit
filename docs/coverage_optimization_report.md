# Coverage Optimization Report

- Generated at UTC: `2026-05-04T07:29:29.263482+00:00`
- Status: `COVERAGE_REQUIRES_REPLAY_EVIDENCE`
- Remaining target: `21011 JPY`
- Live-ready reward: `31741 JPY` (`151.1%`)
- Sequential ladder reward: `21802 JPY` (`103.8%`, steps=`7`)
- Potential reward after promotions: `31741 JPY` (`151.1%`)
- Remaining risk budget: `4202 JPY`

## Blockers

- replay evidence covers target on 3/50 days

## Action Items

- execute coverage as a sequential ladder; do not deploy all live-ready lanes as simultaneous exposure
- rerun replay/backtest after coverage changes and keep gap reasons as product blockers

## Lanes

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`3840` risk=`480` rr=`8.00` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`2147` risk=`452` rr=`4.75` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`2712` risk=`452` rr=`6.00` live_ready=`True` promotion_candidate=`False`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`3840` risk=`480` rr=`8.00` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`2147` risk=`452` rr=`4.75` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY` reward=`2712` risk=`452` rr=`6.00` live_ready=`True` promotion_candidate=`False`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`3840` risk=`480` rr=`8.00` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`2147` risk=`452` rr=`4.75` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY` reward=`2712` risk=`452` rr=`6.00` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`796` risk=`408` rr=`1.95` live_ready=`True` promotion_candidate=`False`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`1086` risk=`490` rr=`2.22` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`796` risk=`408` rr=`1.95` live_ready=`True` promotion_candidate=`False`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`1086` risk=`490` rr=`2.22` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`796` risk=`408` rr=`1.95` live_ready=`True` promotion_candidate=`False`
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`1086` risk=`490` rr=`2.22` live_ready=`True` promotion_candidate=`False`

## Coverage Contract

- Coverage is executable reward from current receipts, not a profit guarantee.
- `DRY_RUN_PASSED` lanes count only as potential coverage until strategy blockers are promoted by receipts.
- A target gap remains a product blocker until it is closed by live-ready, risk-valid lanes or a no-market gap receipt.

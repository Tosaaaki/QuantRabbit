# Coverage Optimization Report

- Generated at UTC: `2026-05-06T05:31:43.599437+00:00`
- Status: `COVERAGE_REQUIRES_REPLAY_EVIDENCE`
- Remaining target: `11120 JPY`
- Live-ready reward: `33405 JPY` (`300.4%`)
- Sequential ladder reward: `13806 JPY` (`124.2%`, steps=`4`)
- Potential reward after promotions: `33405 JPY` (`300.4%`)
- Remaining risk budget: `4202 JPY`

## Blockers

- replay evidence covers target on 4/50 days

## Action Items

- execute coverage as a sequential ladder; do not deploy all live-ready lanes as simultaneous exposure
- rerun replay/backtest after coverage changes and keep gap reasons as product blockers

## Lanes

- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`2790` risk=`978` rr=`2.85` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY` reward=`1976` risk=`978` rr=`2.02` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY` reward=`2790` risk=`978` rr=`2.85` live_ready=`True` promotion_candidate=`False`
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`3672` risk=`960` rr=`3.83` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`1428` risk=`952` rr=`1.50` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`2200` risk=`978` rr=`2.25` live_ready=`True` promotion_candidate=`False`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`1466` risk=`978` rr=`1.50` live_ready=`True` promotion_candidate=`False`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`3672` risk=`960` rr=`3.83` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`1428` risk=`948` rr=`1.51` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`1976` risk=`978` rr=`2.02` live_ready=`True` promotion_candidate=`False`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`1241` risk=`978` rr=`1.27` live_ready=`True` promotion_candidate=`False`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`3672` risk=`960` rr=`3.83` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`1428` risk=`952` rr=`1.50` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`2200` risk=`978` rr=`2.25` live_ready=`True` promotion_candidate=`False`
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`1466` risk=`978` rr=`1.50` live_ready=`True` promotion_candidate=`False`

## Coverage Contract

- Coverage is executable reward from current receipts, not a profit guarantee.
- `DRY_RUN_PASSED` lanes count only as potential coverage until strategy blockers are promoted by receipts.
- A target gap remains a product blocker until it is closed by live-ready, risk-valid lanes or a no-market gap receipt.

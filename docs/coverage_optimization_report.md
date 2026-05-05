# Coverage Optimization Report

- Generated at UTC: `2026-05-05T03:22:14.759559+00:00`
- Status: `COVERAGE_REQUIRES_REPLAY_EVIDENCE`
- Remaining target: `19344 JPY`
- Live-ready reward: `23881 JPY` (`123.5%`)
- Sequential ladder reward: `19457 JPY` (`100.6%`, steps=`6`)
- Potential reward after promotions: `28255 JPY` (`146.1%`)
- Remaining risk budget: `4202 JPY`

## Blockers

- replay evidence covers target on 4/50 days

## Action Items

- promote 3 dry-run receipts only after their strategy blockers clear
- execute coverage as a sequential ladder; do not deploy all live-ready lanes as simultaneous exposure
- repair blockers for: EUR_JPY, EUR_USD
- rerun replay/backtest after coverage changes and keep gap reasons as product blockers

## Lanes

- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`2806` risk=`983` rr=`2.85` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY` reward=`2806` risk=`983` rr=`2.85` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY` reward=`2806` risk=`983` rr=`2.85` live_ready=`True` promotion_candidate=`False`
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`3680` risk=`960` rr=`3.83` live_ready=`True` promotion_candidate=`False`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_PASSED` reward=`1458` risk=`972` rr=`1.50` live_ready=`False` promotion_candidate=`True`
  - blocker: EUR_JPY LONG requires trigger/pending-entry receipts before live use: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`2212` risk=`983` rr=`2.25` live_ready=`False` promotion_candidate=`False`
  - blocker: fresh EUR_USD LONG entry opposes protected EUR_USD SHORT id=470188; use position management instead of a new entry order
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY` reward=`1475` risk=`983` rr=`1.50` live_ready=`True` promotion_candidate=`False`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`3680` risk=`960` rr=`3.83` live_ready=`True` promotion_candidate=`False`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_PASSED` reward=`1458` risk=`972` rr=`1.50` live_ready=`False` promotion_candidate=`True`
  - blocker: EUR_JPY LONG requires trigger/pending-entry receipts before live use: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`2212` risk=`983` rr=`2.25` live_ready=`False` promotion_candidate=`False`
  - blocker: fresh EUR_USD LONG entry opposes protected EUR_USD SHORT id=470188; use position management instead of a new entry order
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY` reward=`1475` risk=`983` rr=`1.50` live_ready=`True` promotion_candidate=`False`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`3680` risk=`960` rr=`3.83` live_ready=`True` promotion_candidate=`False`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_PASSED` reward=`1458` risk=`972` rr=`1.50` live_ready=`False` promotion_candidate=`True`
  - blocker: EUR_JPY LONG requires trigger/pending-entry receipts before live use: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`2212` risk=`983` rr=`2.25` live_ready=`False` promotion_candidate=`False`
  - blocker: fresh EUR_USD LONG entry opposes protected EUR_USD SHORT id=470188; use position management instead of a new entry order
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY` reward=`1475` risk=`983` rr=`1.50` live_ready=`True` promotion_candidate=`False`

## Coverage Contract

- Coverage is executable reward from current receipts, not a profit guarantee.
- `DRY_RUN_PASSED` lanes count only as potential coverage until strategy blockers are promoted by receipts.
- A target gap remains a product blocker until it is closed by live-ready, risk-valid lanes or a no-market gap receipt.

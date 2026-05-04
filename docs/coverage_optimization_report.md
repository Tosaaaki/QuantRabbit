# Coverage Optimization Report

- Generated at UTC: `2026-05-04T06:08:31.892384+00:00`
- Status: `COVERAGE_GAP`
- Remaining target: `19884 JPY`
- Live-ready reward: `0 JPY` (`0.0%`)
- Sequential ladder reward: `0 JPY` (`0.0%`, steps=`0`)
- Potential reward after promotions: `0 JPY` (`0.0%`)
- Remaining risk budget: `0 JPY`

## Blockers

- daily target ledger requires protection repair before fresh risk
- remaining risk budget is unavailable
- live-ready reward misses remaining target by 19884 JPY
- even promoted dry-run reward misses remaining target by 19884 JPY
- no LIVE_READY lanes exist
- replay evidence covers target on 3/50 days

## Action Items

- build at least 11 additional live-ready trigger receipts
- expand lane generation across timing windows or pairs; current repaired ladder cannot cover target
- repair blockers for: AUD_JPY, EUR_JPY, EUR_USD, GBP_USD
- rerun replay/backtest after coverage changes and keep gap reasons as product blockers

## Lanes

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`3024` risk=`378` rr=`8.00` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`2145` risk=`452` rr=`4.75` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`2709` risk=`452` rr=`6.00` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`3024` risk=`378` rr=`8.00` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`2145` risk=`452` rr=`4.75` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`2709` risk=`452` rr=`6.00` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`3024` risk=`378` rr=`8.00` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`2145` risk=`452` rr=`4.75` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`2709` risk=`452` rr=`6.00` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`738` risk=`378` rr=`1.95` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED` reward=`1085` risk=`489` rr=`2.22` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`738` risk=`378` rr=`1.95` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED` reward=`1085` risk=`489` rr=`2.22` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`738` risk=`378` rr=`1.95` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED` reward=`1085` risk=`489` rr=`2.22` live_ready=`False` promotion_candidate=`False`
  - blocker: only protected trader-owned positions can be layered; EUR_USD LONG id=470130 is not eligible
  - blocker: external/manual risk is open: EUR_USD LONG id=470130 20000u; adopt or close before new entries
  - blocker: open position lacks TP/SL: EUR_USD LONG id=470130 20000u

## Coverage Contract

- Coverage is executable reward from current receipts, not a profit guarantee.
- `DRY_RUN_PASSED` lanes count only as potential coverage until strategy blockers are promoted by receipts.
- A target gap remains a product blocker until it is closed by live-ready, risk-valid lanes or a no-market gap receipt.

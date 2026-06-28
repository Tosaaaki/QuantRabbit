# Completion Status Report

- Generated at UTC: `2026-06-02T01:02:24.439653+00:00`
- Status: `BLOCKED`
- Open positions: `6`
- Pending entry orders: `0`
- Remaining target: `30727 JPY`
- Live-ready lanes: `5`
- Close recommendations: `0`
- Close Gate B authorized: `False`
- Coverage status: `COVERAGE_GAP`

## Blockers

- `COVERAGE_BLOCKER` live-ready reward misses remaining target by 28436 JPY
- `COVERAGE_BLOCKER` even promoted dry-run reward misses remaining target by 28111 JPY
- `COVERAGE_BLOCKER` replay evidence covers target on 4/50 days
- `REPLAY_COVERAGE_GAP` legacy replay covers target on 4/50 days
- `EXECUTION_REPLAY_MISSING` execution replay receipt is missing
- `CERTIFICATION_STALE` dry-run certification is older than current coverage, order-intents, execution replay, or live-order artifacts
- `CERTIFICATION_BLOCKER` coverage optimization still has blockers
- `CERTIFICATION_BLOCKER` execution replay receipt is missing
- `CERTIFICATION_BLOCKER` latest GPT trader decision was rejected

## Next Actions

- `COVERAGE_ACTION` promote 2 dry-run receipts only after their strategy blockers clear
- `COVERAGE_ACTION` build at least 63 additional live-ready trigger receipts
- `COVERAGE_ACTION` expand lane generation across timing windows or pairs; current repaired ladder cannot cover target
- `COVERAGE_ACTION` repair blockers for: AUD_JPY, EUR_CAD, EUR_JPY, EUR_USD, GBP_AUD, GBP_USD, NZD_CAD, USD_CHF
- `COVERAGE_ACTION` rerun replay/backtest after coverage changes and keep gap reasons as product blockers
- `RUN_EXECUTION_REPLAY` run replay-execution with a quote path after at least one LIVE_READY intent exists
- `RERUN_CERTIFICATION` rerun certify-dry-run after coverage, replay, learning, and no-send artifacts pass

## Completion Contract

- Completion requires broker truth, live-ready coverage, execution replay, learning receipts, and dry-run certification to pass together.
- Only unprotected trader-owned, external, or over-budget exposure blocks fresh entries; manual/tagless operator exposure is TP-managed only.
- Protected trader-owned exposure and trader-owned pending entries may add only through basket portfolio risk validation.
- The 10% daily target remains a risk-bounded product KPI, not permission to force trades.

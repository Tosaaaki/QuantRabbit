# DOJO strategy-worker first wave

This is a worn-TRAIN study, not forward proof and not live permission.

- Window: `2025-06-01T00:00:00Z` to `2025-07-01T00:00:00Z`
- Initial balance: JPY 200,000
- Pairs: AUD_USD, EUR_USD, GBP_USD, NZD_USD, USD_JPY
- Workers: compression break, daily break pullback, range-fade limit, spike fade
- Fixed cells: OHLC/OLHC x BASE/STRESS
- True LOPO: main plus five leave-one-pair-out replays per cell
- Fixed denominator: 4 candidates x 4 cells x 6 replay scopes = 96 replays
- Study SHA-256: `29cca3bfc02d35d5380f33b9332221e8f72065a27a8fdedbb12e4ee2d9b9fd12`

The June 2025 five-pair source passes the existing 98% full-day, 80% short-day,
and 900-second causal-gap preflight. No threshold was relaxed to admit it.

Attempts 2 and 3 are intentionally disabled. They require an append-only
cumulative lineage registry, independently verified prior TRAIN results, and a
no-repeat candidate registry. Repeated ad-hoc calls must not turn a nominal
14-candidate budget into an unbounded search.

Historical monthly aggregation uses the exact 2024-01 through 2026-06
denominator. It can report whether all 30 pessimistic STRESS months reached
3x, but it always keeps promotion, live permission, and order authority off.

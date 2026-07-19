# DOJO strategy-worker first wave

This is a worn-TRAIN study, not forward proof and not live permission.

- Window: `2025-06-01T00:00:00Z` to `2025-07-01T00:00:00Z`
- Initial balance: JPY 200,000
- Pairs: AUD_USD, EUR_USD, GBP_USD, NZD_USD, USD_JPY
- Workers: compression break, daily break pullback, range-fade limit, spike fade
- Fixed cells: OHLC/OLHC x BASE/STRESS
- True LOPO: main plus five leave-one-pair-out replays per cell
- Fixed denominator: 4 candidates x 4 cells x 6 replay scopes = 96 replays
- Study SHA-256: `bacf5389b53cf1c1077aa867d8ae8c145df74101b3b94c4f171f65806304f1d7`

The June 2025 five-pair source passes the existing 98% full-day, 80% short-day,
and 900-second causal-gap preflight. No threshold was relaxed to admit it.

This factory emits attempt 1 only. The generic trainer does not yet enforce a
cumulative lineage across separate studies, so attempts 2 and 3 are not
admissible evidence until an append-only registry binds independently verified
prior TRAIN results and rejects repeated candidates. Repeated ad-hoc studies
must not turn a nominal 14-candidate budget into an unbounded search.

The monthly scorer is currently an arithmetic-only scaffold: it fixes the
2024-01 through 2026-06 denominator but is not yet bridged to verified trainer
outputs or family-LOPO replays. It cannot establish research evidence and
always keeps promotion, live permission, and order authority off.

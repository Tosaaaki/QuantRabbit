# Active Opportunity Board

- Status: `BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY`
- Read-only: `True`
- Live permission allowed: `False`
- Total lanes: `93`
- Pairs scanned: `18`
- Vehicles scanned: `LIMIT, MARKET, STOP`

## Top Lane

- Lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT`
- Pair / direction / family / vehicle: `EUR_USD` / `SHORT` / `BREAKOUT_FAILURE` / `LIMIT`
- Status: `EVIDENCE_ACQUISITION`
- Expected edge JPY: `613.2`
- Proof / replay: `EVIDENCE_GAP;proof_gap=0.0` / `SPREAD_SLIPPAGE_PROOF_INCOMPLETE`
- Next action: Acquire or canonicalize proof/replay evidence for EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT; do not send or mix vehicles.

## Coverage

- LIVE_READY diagnostic count: `0`
- HARVEST_READY count: `0`
- SCOUT_READY count: `0`
- EVIDENCE_ACQUISITION count: `17`
- NO_TRADE count: `76`

## Active Path

EVIDENCE_ACQUISITION: failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT is the closest read-only path; active_contract=EVIDENCE_ACQUISITION.

Scanned 18 pairs and 93 pair/direction/family/vehicle lanes. Current closest path is failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT (EVIDENCE_ACQUISITION), but live permission remains false. EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST can move the 4x loop forward only as read-only HARVEST evidence: broad TP proof material shows 20/0, exact LIMIT S5 replay shows 4/0 with expectancy 813.7734 JPY/trade, while proof queue, guardian, negative expectancy, and live gateway blockers remain visible.

- Root improvement target: Parallelize evidence acquisition across comparable lanes, starting with failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT; Make the EUR_USD SHORT BREAKOUT_FAILURE LIMIT HARVEST vehicle live-grade evidence by canonical replay/proof import, not by mixing MARKET/STOP samples or relaxing gates.
- Expected edge improvement: Top lane expected_edge_jpy=613.2; improvement is evidence quality/ranking coverage, not live permission. Expected improvement is evidence quality, not live permission: exact LIMIT replay and proof-floor reconciliation can separate TP-positive HARVEST (20/0 material; harvest artifact currently 20/0) from market-close leakage.

## Top No-Trade Causes

- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`: 66
- `SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH`: 66
- `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`: 63
- `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION`: 63
- `BAD_UNITS`: 39
- `LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT`: 39
- `STRATEGY_NOT_ELIGIBLE`: 35
- `EXHAUSTION_RANGE_CHASE`: 31
- `SPREAD_TOO_WIDE`: 27
- `RANGE_ROTATION_BROADER_LOCATION_CHASE`: 24
- `CHART_DIRECTION_CONFLICT`: 23
- `STRATEGY_PROFILE_MISSING`: 23

## Safety

This board does not authorize live order entry, SCOUT execution, gateway permission, cancellation, close, broker-state mutation, launchd load/reload, gate relaxation, 4x deficit lot backsolve, secret disclosure, or inferred operator approval.

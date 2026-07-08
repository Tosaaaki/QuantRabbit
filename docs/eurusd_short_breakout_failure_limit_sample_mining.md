# EUR_USD SHORT BREAKOUT_FAILURE LIMIT Sample Mining

Generated: `2026-07-08T07:50:36Z`

## Verdict

- Status: `LOCAL_LIMIT_SAMPLE_COVERAGE_EXHAUSTED_STILL_UNDERSAMPLED`
- Target shape: `EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST`
- Read-only: `True`
- Live permission allowed: `False`
- Current replayed exact LIMIT samples: `4`
- Additional acceptable local samples found: `0`
- Remaining exact LIMIT samples: `16`

This artifact is evidence coverage only. It did not send, stage, cancel, close, mutate broker state, change launchd, or relax gates.

## Coverage Summary

- Execution ledger: `{'ACCEPTED_CURRENT_REPLAY': 1, 'REJECTED_NON_TARGET_METHOD': 8, 'REJECTED_NON_TARGET_OR_NOT_LIMIT': 4, 'REJECTED_NOT_LIMIT_ORDER': 7}`
- Legacy history: `{'ACCEPTED_CURRENT_REPLAY': 3, 'REJECTED_CURATED_STRATEGY_NOT_EXACT': 1, 'REJECTED_NOT_LIMIT_ORDER': 7}`

## Additional Acceptable Samples

- None.

## Rejection Boundary

- MARKET_ORDER and STOP_ORDER wins remain outside LIMIT proof.
- RANGE_ROTATION and TREND_CONTINUATION wins remain outside BREAKOUT_FAILURE proof.
- Ambiguous direct-USD or continuation legacy rows remain rejected unless a stronger source maps them to failed-break/retest.
- Partial-close or duplicate close rows are not counted as fresh exact attached-TP samples.

## Next Read-Only Actions

- Do not reclassify MARKET/STOP or range/trend wins into the LIMIT proof floor.
- If more samples are needed, import or locate additional broker/legacy rows with exact LIMIT_ORDER entry, attached TAKE_PROFIT_ORDER exit, and BREAKOUT_FAILURE-equivalent strategy evidence.
- If exact LIMIT history remains exhausted, split MARKET and STOP vehicles into separate read-only proof contracts instead of mixing them into LIMIT.
- Keep live_permission_allowed=false until risk, guardian/operator review, verifier, gateway, and fresh broker truth all pass.

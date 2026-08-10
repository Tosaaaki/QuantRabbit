# MONTHLY_2X_DIRECT_PROOF_V1 — contract and inherited-evidence checkpoint

Status: `TARGET_PATH_NOT_YET_PROVEN`

This checkpoint changes only the terminal target from 3.0x to 2.0x. It does not change the cohort, chronology, cost model, margin cap, drawdown cap, missingness rules, holdout boundary, or acceptance gates. It does not rerun prior simulations.

## Frozen contract

- Start: 200,000 JPY
- Required rolling 30-calendar-day equity: at least 400,000 JPY after all costs
- Gross margin: at most 150,000 JPY
- Maximum drawdown: at most 80,000 JPY
- Completed bars, bid/ask side-aware execution, adverse half-spread slippage, fees and financing
- TRAIN, one-hour embargo, VALIDATION at 16/32/64 days; holdout unread
- Live, Paper, broker, order and deploy are prohibited

Contract SHA-256: `c2d2211fe90305f4935cb30d3bb4dcb6f4a2cbbd320278ad942c96658b4d6724`

## Independent gap calculation

- Inherited best rolling-30d multiple: `1.0476891546051672`
- Linear gap to 2.0x: `0.9523108453948328`
- Equity gap from 200,000 JPY: `190,462.16907896656 JPY`
- Required factor from the inherited best: `1.9089631606940909`
- Required additional gain from the inherited best equity: `90.89631606940908%`
- Log-growth gap: `0.6465602468591838`

The exact-rational linear oracle and independent logarithmic ratio oracle both report a positive shortfall.

## Mechanical 2x reclassification

| Existing family | Status | Decisive readback |
| --- | --- | --- |
| Multidimensional sweeps | `TARGET_PATH_NOT_YET_PROVEN` | 5,670 rows; stable multiwindow candidates 0 |
| Technical fusion / MTF | `TARGET_PATH_NOT_YET_PROVEN` | TRAIN plateau 0; stable 32d/64d 0; X-MTF eligible validation 0 |
| Trail / BE / partial TP | `NOT_EVALUABLE` | strict path 5; fee/financing, margin/netting, partial-fill/unwind missing |
| Hedge arms | `TARGET_PATH_NOT_YET_PROVEN` | all four preregistered arms `REJECT` |
| Dynamic lot / inventory / exposure | `TARGET_PATH_NOT_YET_PROVEN` | cap-compliant 2x rows 0; best `1.0310861778747464`; paired LCB `-10.300239585396039 JPY` |
| X-derived methods | `NOT_EVALUABLE` | validation eligible 0; two insufficient-evidence and one disconnected route |
| OSS adapters | `TARGET_PATH_NOT_YET_PROVEN` | financial parity retained; profitability increment `0 JPY` |
| 28-pair rotation | `TARGET_PATH_NOT_YET_PROVEN` | fast and slow variants both TRAIN plateau 0 and stable 32d/64d 0 |
| Conditional positive vehicle | `TARGET_PATH_NOT_YET_PROVEN` | 4 trades / `+3,255.0938 JPY`; 16 more independent proofs required by its frozen gate |
| Capital-preservation gate | `NOT_EVALUABLE` as a profit source | 251 WAIT / 0 TRADE; profit generation remains unproved |

No row is `PROVEN`. Missing evidence is not converted to zero or to a loss.

## Dominant blocker

No preregistered family has both a positive TRAIN LCB plateau and stable 32d/64d validation. Separately, strict decision-time executable coverage is zero: cost/financing, margin/exposure, and exit/unwind coverage are each 0 of 251 decisions.

## Next independent hypothesis

`EVENT_DRIVEN_CROSS_ASSET_DISLOCATION_V1`: preregister a new event-driven opportunity source using timestamped macro-release surprise, synchronized cross-asset reaction, and side-aware FX bid/ask execution. This is a new mechanism and data source, not a retune of observed technical, MTF, hedge, sizing, or rotation thresholds. It is not admitted for evaluation until source lineage, event chronology, after-cost fill, margin and unwind evidence are fixed while the holdout remains unread.

## Verification

- All 15 inherited report files match their preregistered SHA-256 values and committed HEAD bytes.
- Reclassification tests: 8/8 PASS.
- Holdout read: false.
- Live/Paper/broker/order/deploy touched: false.

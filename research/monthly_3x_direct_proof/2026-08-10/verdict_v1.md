# Monthly 3x direct proof verdict

The acceptance target was fixed at **200,000 JPY → 600,000 JPY in 30 days**, after bid/ask and adverse slippage, with margin capped at 150,000 JPY and realized drawdown capped at 80,000 JPY.

## Search completed

- V1: 5,670 rows across AUDJPY/EURJPY/EURUSD, five price-action families, three lookbacks, three sessions, and seven exit policies.
- V2: 2,430 rows after freezing the strongest exit (`TIME_24`) and adding only completed-bar body, SMA-slope, and ATR-expansion quality filters.
- Every signal used a completed M5 bar; entry was the next side-correct OANDA open. Wick first-touch, conservative STOP-first, spread, additional adverse half-spread slippage, and financing-boundary exclusion were fixed.
- TRAIN → one-hour embargo → VALIDATION was applied to 16/32/64-day windows. The sealed holdout was not read.

## Result

No TRAIN point had a positive bootstrap lower bound, so there was no connected TRAIN plateau and no frozen point eligible for 32/64-day multi-window validation. Consequently, **monthly 3x is not proved** under this contract.

The closest apparent result explains why validation-only selection is forbidden: EURJPY SMA-pullback, lookback 12, 07–16 UTC, body ≥0.5 ATR produced 32-day VALIDATION expectancy +146.20 JPY/1000u and LCB +60.00, but its matching TRAIN expectancy was -48.08 and its 64-day VALIDATION LCB was -71.16. This is a regime reversal, not a repeatable edge.

BE, fixed ATR brackets, ATR trail, SMA deterioration, and structure-break exits did not create a stable TRAIN plateau. Increasing size cannot repair a negative lower bound and therefore cannot prove 3x.

Status: `MONTHLY_3X_NOT_PROVED`. This is not task completion; it is the direct rejection of this bounded strategy family. The next independent hypothesis must change the opportunity source rather than retune these thresholds.

# SL alternative paired robustness verdict v2

Decision: **REJECT all four alternatives for adoption.** This does not claim that a future, different contract can never profit. It says the preregistered 16/32/64-day contract does not establish after-cost profitability.

## Frozen comparison

The control is the current SL close. Alternatives are kept separate: 0.25 and 0.35 reverse STOP at the SL point, equal opposite hedge at initial entry, and equal opposite hedge at the SL point. Entry, TP/SL cohort, S5 bid/ask sources, cost and risk gates are otherwise paired. Holdout remained sealed.

After bounded read-only OANDA retrieval, the 32-day cohort reached 7/7 calculable events. The 64-day cohort remained 7/14 calculable because actual S5 gaps, non-reproducible frozen TP/SL first touches, or missing same-candle ordering remain. No interpolation or M1 substitution was used.

## 64-day VALIDATION readback

| Arm | Trades / cohort | Net JPY | PF | Expectancy JPY | Max DD JPY | Paired LCB JPY | Peak gross margin JPY | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Current SL | 6 / 14 | -7,015.78 | 0 | -1,169.30 | 7,015.78 | 0 | 62,899.82 | control fails absolute gates |
| Reverse STOP 0.25 | 6 / 14 | -8,091.88 | 0 | -1,348.65 | 8,091.88 | -364.78 | 78,506.21 | REJECT |
| Reverse STOP 0.35 | 6 / 14 | -8,522.33 | 0 | -1,420.39 | 8,522.33 | -510.69 | 84,786.70 | REJECT |
| Equal hedge at entry | 6 / 14 | -1,986.78 | 0 | -331.13 | 1,986.78 | +360.27 | 125,799.65 | REJECT |
| Equal hedge at SL | 6 / 14 | -8,329.29 | 0 | -1,388.22 | 8,329.29 | -330.78 | 125,799.65 | REJECT |

The initial equal hedge improves relative to an especially poor SL cohort, but still loses money absolutely. Its paired LCB therefore cannot satisfy Net > 0 and PF > 1. Equal opposite inventory cancels directional exposure; spread, slippage, financing uncertainty, double margin and unwind risk remain.

## RCA and refinement disposition

1. **Structural cost lock:** equal hedges lock price exposure, not profit. The 64-day initial-hedge spread estimate is 2,068.91 JPY and slippage stress 215.82 JPY; net remains negative.
2. **Trend/mean-reversion failure:** reverse 0.25/0.35 loses in the observed validation and has negative paired LCB. A trend continuing after SL is not guaranteed, while mean reversion is not guaranteed for the equal hedge unwind.
3. **Admission data floor:** the complete 64-day cohort has only 14 STOP episodes and 7 exact diagnostic paths, below the preregistered 30 events per split. Same-S5 fill ordering, financing across rollover, dual unwind, partial fills and complete DD are not proved.

The two preregistered refinements were not outcome-tuned or opened as adoption candidates. R1 price-action gating can only reduce the already insufficient 14-event cohort. R2 shortening unwind from 3600s to 1800s cannot raise the complete cohort above the fixed 30-per-split gate. The independent historical-learning hypothesis was opened instead.

Independent arithmetic readback recomputed trades, net, PF, expectancy, DD, cost totals and margin peaks for 30 arm/split combinations: 30 passed, 0 failed. Eight property/leakage tests also passed.

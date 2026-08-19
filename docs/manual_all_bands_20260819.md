# Manual trading, all history, all timeframe bands

Swept the complete OANDA transaction id space (1 → 473,334, 100% coverage):
**70,948 order fills, 2025-05-30 → 2026-08-19**. Separating by client
extension tag gives 32,779 bot-opened trades and **1,361 untagged (manual)**,
of which 1,358 closed.

An earlier cut in this session used only 17 trades because it read the
transaction stream from 2026-07-16. That was a window, not a sample.

## Manual trades carrying a take-profit, by TP distance

Maximum adverse excursion measured per trade from M1/M5 bid-ask bars, then
candidate stops applied to winners and losers alike.

| band | n | TP median | hold median | no stop | best stop | verdict |
|---|---|---|---|---|---|---|
| scalp ≤15p | 184 | 8p | 0.5 h | **−0.04** | −0.27 (−16p) | negative |
| **15–50p (H1)** | **227** | 26p | 1.9 h | **−1.81** | **−0.82** (−52p) | **negative** |
| 50–150p | 54 | 68p | 2.5 h | −16.68 | −11.04 | strongly negative |
| >150p | 10 | 255p | 36.8 h | +77.18 | +77.18 | n=10, not evidence |

Every band with a usable sample is negative at every stop distance tested
(1×, 2×, 3× the median take-profit). The >150p band shows +77.18 pips/trade
with a one-sided lower bound of +0.28, but n = 10 against sd 147.8 — the same
few-large-winners shape that was rejected three times earlier today — and its
36.8-hour holds are neither the scalp nor the H1 style in question.

## The scalp result in detail, and why the recent window misled

| | n | mean pips | win |
|---|---|---|---|
| 2026-08 only (the earlier cut) | 17 | **+2.32** | 82% |
| **everything before it** | **166** | **−0.59** | **48%** |
| all history | 183 | −0.32 | 51% |

Month by month with the −10 stop, the two largest samples are both negative:

```
2025-06  n= 22  +2.18      2026-03  n= 73  -1.26
2025-12  n=  6  +4.97      2026-04  n= 45  -0.45
2026-02  n=  3  +4.70      2026-06  n=  8  -6.11
2026-08  n= 17  +2.32      2026-07  n=  7  -1.19
```

Every positive month has n ≤ 22. The two months with real sample size are
negative.

## Consequence

The 45-trade collection plan built earlier today is withdrawn. It was
designed to answer a question that 183 existing trades already answer, and
collecting 45 more would not change the answer — it would reproduce the same
window-sized sample that produced the false positive.

What the pre-registration machinery is still worth: the rule freezing, the
discipline check, the paper loop and the decision recorder all remain the
correct instruments for testing a *new* method. They were built to stop a
fitted parameter from being reported as evidence, and that is exactly what
just happened to the +2.32.

## Limits

"Manual" here means "no client extension tag". A bot order placed without a
tag would be misclassified, and 893 manual trades carrying no take-profit are
outside this analysis entirely — those are the purely discretionary exits and
they are not measured here.

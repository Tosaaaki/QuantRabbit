# L031 — is detection power a design variable? (axis ⑥)

L029 left the wall as a sample-size problem, not an edge problem:
`n = (1.645·sd/μ)² = (1.645·100.7/5.56)² ≈ 888 days ≈ 3.5 years`.
μ is exhausted. sd was assumed to be dominated by common currency factors — remove
them and the required n collapses without finding any new edge.

**Tested. The hypothesis is wrong.**

| window | series | n | mean | sd | LB | required n | years |
|---|---|---|---|---|---|---|---|
| 2020-2023 | raw portfolio | 889 | 5.562 | 100.74 | +0.004 | 888 | 3.5 |
| 2020-2023 | **factor-neutral** | 889 | 5.667 | **100.18** | +0.140 | **846** | 3.4 |
| 2024-2026 | raw portfolio | 677 | 10.036 | 170.78 | **−0.761** | 784 | 3.1 |
| 2024-2026 | **factor-neutral** | 677 | 9.280 | **169.86** | −1.459 | **907** | 3.6 |

**Common currency factors explain 1.1% of the variance** in both windows. sd moves
100.7 → 100.2 and 170.8 → 169.9. The required n does not move (and gets *worse*
in 2024-2026). Currency-strength factors were rebuilt from price only, per day,
from the pairs actually present (`r_XY = s_X − s_Y`, sum-zero), with no strategy
information — so this is not a weak-instrument artifact of the strategy.

The variance is **idiosyncratic**, not factor beta. A 48-hour position opened at
an arbitrary minute is not a daily-return exposure; its dispersion comes from the
−50-pip disaster stop and the fat tail of two-day moves. Hedging currency beta
removes nothing because there was almost nothing there to remove.

## The correction this turned up

Building the factor model exposed an asymmetry in the corpus that changes how
L026 should be read:

| coverage | pairs |
|---|---|
| 2020-01-01 → 2026-07-09 | AUD_USD, EUR_USD, GBP_USD, NZD_USD, USD_JPY |
| **2024-01-01** → 2026-07-09 | AUD_JPY, CAD_JPY, CHF_JPY, EUR_JPY, GBP_JPY, NZD_JPY |

L026's final proof — "the quiet gate survives genuinely unseen years 2020-2023" —
therefore ran on **5 pairs, not 11**, and every one of its 1,686 gated pair-days
had **USD on one leg**:

```
USD 1686 / 1686 legs   GBP 350   AUD 337   EUR 334   JPY 333   NZD 332
```

In the pair basis that reads as five instruments. In the currency basis it is
close to **one bet on the dollar**, sampled 1.9 ways a day. That is consistent
with the measured Σvar_i / var_portfolio of 0.87 (below 1 ⇒ positively
correlated, i.e. no diversification), and it is a second reason — independent of
the pair×day pooling error — that the 889 "days" were never 889 observations.

The 2024-2026 window inverts the concentration rather than fixing it: 3.75
pairs/day but 1,608 JPY legs against 1,146 USD, and the variance ratio falls to
0.66.

## Alpha is unstable year to year

| 2020 | 2021 | 2022 | 2023 | | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| +9.38 | +6.71 | +11.06 | **−4.25** | | +11.94 | +13.61 | **−0.69** |

(factor-neutral residual + alpha; 2024-2026 is TRAIN/TEST and contaminated, and
its raw portfolio lower bound is already negative at −0.761)

## What is closed and what this redirects to

**Closed:** variance reduction by currency-factor hedging. The wall does not move
on this axis.

**Not closed, and now better specified:** if the variance is idiosyncratic, then
independent bets *do* reduce it as 1/√k — the binding quantity is the number of
genuinely uncorrelated opportunities, not the factor exposure. The strategy takes
**1.9 per day** because the frame allows one entry per pair per 48-hour hold
across 11 correlated pairs. At k=20 genuinely independent bets, sd/√k would cut
required n roughly tenfold, to ~85 days — a testable quarter instead of 3.5
years.

FX majors cannot supply that k: the measured ratios (0.87, 0.66) say they are one
or two bets wearing eleven names. Getting k requires instruments that do not
share a currency leg — which points at axis ① (a different market; the
`codex/crypto-bitbank-*` line already exists and was never concluded), not at
more FX pairs.

## Reproduce

```
research/paper/cache_daily_closes.py     # daily closes from the M1 corpus
research/paper/factor_power.py           # QR_WIN=2024,2025,2026 to switch window
```

# L032 — how many independent bets can be bought? (axis ①)

L031 showed the variance is idiosyncratic, so `sd ∝ 1/√k` and required
`n = (1.645·sd/μ)²` falls as `1/k`. At the frame's k = 1.9/day the wall is 888
days. k = 20 would put it at ~85 days — one quarter. So: **which market supplies
k?**

Measured identically in both markets: daily log returns 2025-2026, effective
independent count `N_eff = (Σλ)²/Σλ²` from the correlation matrix, and median
|daily move| against round-trip cost.

| market | instruments | mean pairwise ρ | **N_eff** | sd_avg → sd_portfolio |
|---|---|---|---|---|
| FX majors | 11 | +0.371 | **2.85** | 0.49% → 0.32% (0.66; independent would be 0.30) |
| bitbank JPY | 16 | **+0.694** | **1.87** | 3.51% → 3.01% (0.86; independent would be 0.25) |

**Sixteen crypto pairs are fewer than two independent bets.** They are one
risk factor wearing sixteen names — worse than FX's eleven names carrying 2.85.

Move-to-cost is also worse, not better:

| | best | worst |
|---|---|---|
| FX (real fill spread) | USD_JPY **54.4** | NZD_JPY 10.5 |
| bitbank (taker 0.24% round trip, **assumed** published rate) | gala 10.8 | trx 3.1 |

USD_JPY's daily move buys 54× its cost; the best crypto pair buys 10.8×. Crypto
loses on both axes at taker fees.

Crypto is only interesting as a **maker** (−0.02%/side ⇒ negative round-trip
cost). But passive execution is already closed in this project: L027 tested
short-hold passive limits across 48 configurations and all failed — adverse
selection exceeded the half-spread saved. Crypto flow is not less toxic than FX
flow, and at k = 1.87 even a zero cost leaves required n near 900 days.

## What k would have to be, and what the universe supplies

| k source | k | required n | |
|---|---|---|---|
| current frame | 1.90 | 888 d | 3.5 y |
| all 11 FX simultaneously | 2.85 | 592 d | 2.4 y |
| all 16 crypto simultaneously | 1.87 | 904 d | 3.6 y |
| FX + crypto, assuming ρ=0 between them (generous) | 4.71 | 358 d | 1.4 y |
| **needed for a one-quarter test** | **~20** | **85 d** | — |

The retail-accessible liquid universe measured here supplies **2 to 5**
independent streams, not 20. Adding instruments does not add bets; it adds
names on the same two or three factors (USD, JPY/risk, crypto-beta).

(The n conversion assumes only `sd ∝ 1/√k` and that μ survives widening. μ
surviving is not tested and is the optimistic direction — spreading across more
instruments usually thins the average edge.)

## The structural reading

Three axes have now been measured and all say the same thing from different
angles:

- **μ** (edge) — exhausted across 60,000 swept cells; the one survivor was a
  pooling artifact (L029)
- **sd via factor hedging** — 1.1% of variance is factor beta; nothing to remove
  (L031)
- **sd via diversification** — the universe holds 2–5 independent bets, not 20
  (L032, here)

`n = (1.645·sd/μ)² ≈ 850–900 days` is therefore not an artifact of one strategy.
It is a property of retail-cost directional trading in this asset universe:
**a directional edge here cannot be verified in under about three years.**

That is the same statement as "these markets are efficient at the retail
cost-and-information level," arrived at from the measurement side rather than
assumed.

## What this leaves

If a directional edge cannot be *verified* before the capital is spent, then a
mechanism that earns cannot rest on one. It has to rest on a return that is
**mechanically known ex ante** rather than statistically discovered. This
project has already found three such things and measured all of them:

| mechanical return | status |
|---|---|
| carry / swap | real, but decaying (USD_JPY 2025 +2.8%) and the swap table was 2.6× overstated |
| triangular arbitrage | **100% out-of-sample convergence** — and 108× too small to matter |
| maker rebate | eaten by adverse selection (L027, 48/48 configurations) |

The pattern across all three is identical: **real, verifiable without statistics,
and sub-scale for retail size.** That, not a missing signal, is the shape of the
problem.

## Reproduce

```
research/paper/independent_bets.py   # fetches bitbank public candles, caches locally
```

Fee rates for bitbank are the published maker/taker values entered as constants,
not measured fills. Any conclusion that turns on them needs verification.

# Synthesis — everything measured, reduced to one equation

2026-08-19. Every axis tested today and in the 15-month log collapses into a
single quantity. This document is the complete state, for a decision.

## 1. The master law

For a one-sided 95% test that a return series is positive:

```
n_required = (1.645 / S_daily)²  =  682 / S_annual²      [trading days]
```

Verified against the measured survivor: μ = 5.562, sd = 100.74 →
S_daily = 0.0552, S_annual = **0.877** → n = **888 days**. Identical to the
figure L029 derived independently.

Every axis is therefore the same question: **how much does it raise Sharpe?**

| S_annual | verification time |
|---|---|
| 0.5 | 10.8 years |
| **0.88 (best ever measured, in-sample)** | **3.5 years** |
| 1.0 | 2.7 years |
| 1.5 | 1.2 years |
| 2.0 | 0.7 years |
| **3.0** | **0.3 years (one quarter)** |

## 2. Every axis, in the same units

| axis | intervention | S_annual | verification | verdict |
|---|---|---|---|---|
| — | current frame (k=1.9, retail cost) | 0.88 | 3.5 y | baseline |
| ② cost | execution cost **halved** | 1.05 | 2.5 y | insufficient |
| ② cost | execution cost **zero** (impossible) | 1.22 | 1.8 y | **insufficient even at the limit** |
| ① k | all 11 FX simultaneously | 1.07 | 2.3 y | insufficient |
| ① k | all 16 crypto simultaneously | 0.87 | 3.6 y | **worse than FX** |
| ① k | FX + crypto assuming ρ=0 (generous) | 1.38 | 1.4 y | insufficient |
| ⑥ power | currency-factor neutralisation | 0.90 | 3.4 y | **1.1% of variance was beta** |
| ①+② | **every favourable assumption stacked** | **1.92** | **0.7 y** | still not a quarter |
| — | **needed for a one-quarter test** | **2.83** | 0.3 y | — |
| ④ capital | more capital / prop funding | **unchanged** | unchanged | n is scale-invariant |

Two results deserve emphasis because they were the last hopes:

- **Free execution is not enough.** Even at zero cost the series needs 456 days.
  Cost reduction raises μ by 2.20 pips/day at most (1.90 trades × 1.16 pips mean
  spread), which moves S from 0.88 to 1.22. The variance is simply too large.
- **More capital changes nothing statistically.** n depends on the return
  *series*, not its size. Prop funding relocates who bears the loss; it does not
  make the edge knowable, and typical 5–10% drawdown rules are tighter than this
  distribution (43% win days, median −18 pips, top 5 of 889 days = 53% of profit)
  can survive.

## 3. The untested inputs, priced against the same bar

The 2026-08-07 log left "non-price inputs" as one of two remaining doors: COT,
event calendars, news, order-book depth. The law prices them exactly:

| target window | required μ | multiple of current |
|---|---|---|
| 2 years | 7.38 pips/day | **1.33×** |
| 1 year | 10.44 pips/day | **1.88×** |
| 1 quarter | 17.97 pips/day | **3.23×** |

So "should we test COT?" becomes "does COT nearly double the daily edge?" COT is
a weekly, lagged, aggregated positioning report; nothing in its published effect
sizes approaches 1.88×. News-event windows are real but the retail-tradeable
residue after spread-widening and latency is small. Order-book depth genuinely
can produce high Sharpe — that is what HFT is — but it requires colocation and
the S5 corpus contains no depth data at all.

None of the three clears the one-year bar, let alone the quarter.

## 4. The edge probably is not there in the first place

All of section 2 assumes S = 0.877 is real. It is not a clean estimate:

- it is **in-sample**, selected from ~60,000 swept cells
- the correct one-row-per-day lower bound is **+0.0044** — n > 888 required,
  sample 889, passed by one day (L029)
- **every individual year's bound is negative**; 2023's mean is −4.25
- top 5 of 889 days carry 53% of profit; without them the bound is −2.52
- the 28-day forward window ran **−18.83 pips/day** (not significant alone at
  9.2 percentile, but not supportive either)
- the "unseen years" proof ran on **5 pairs with USD on 1,686 of 1,686 legs** —
  one dollar bet sampled 1.9 ways (L031)

Honest point estimate of the true directional edge in this universe: **zero,
possibly negative.** The 3.5-year verification figure describes how long it
would take to confirm something that the evidence says is not there.

## 5. Where k = 20 actually lives

The k question has a truthful answer, and it is not FX or crypto. Independent
bets at that count exist in **equities**: thousands of names whose residuals,
after removing market and sector factors, are genuinely close to independent.
That is the structural reason equity stat-arb funds exist and FX stat-arb funds
mostly do not — FX has eight currencies, i.e. seven factors, and the measured
N_eff of 2.85 across eleven pairs is that arithmetic showing through.

It is named here for completeness, not as a recommendation: retail equity
execution costs on the small names where residual edge lives, plus borrow for
the short leg, plus the fact that this is the most competed space in finance,
make it a worse retail proposition than what was already closed.

## 6. The three mechanical returns share one shape

Returns that need no statistical discovery — knowable ex ante — were all found
and all measured:

| return | verifiable without statistics? | size |
|---|---|---|
| carry / swap | yes | real, decaying (USD_JPY 2025 +2.8%/yr), swap table was 2.6× overstated |
| triangular arbitrage | yes — **100% out-of-sample convergence** | **108× too small** |
| maker rebate | yes | eaten by adverse selection, 48/48 configurations |

**Real, verifiable, sub-scale.** Three independent confirmations of the same
thing: at retail size, in this universe, the mechanical edges exist and do not
pay. The problem was never a missing signal.

## 7. What the objective can honestly be

At S = 0.877 — the most generous number available — annual return ≈ S × vol,
with max drawdown of roughly the same order as vol:

| vol target | expected return | rough max DD | on 246,269 JPY |
|---|---|---|---|
| 10% | 8.8%/yr | ~10% | +21,700/yr |
| 20% | 17.6%/yr | ~20% | +43,300/yr |

Monthly 2× requires 26×/year. At S = 0.877 that needs roughly 2,950% annualised
volatility — ruin on the first adverse week. This is the same conclusion as the
leverage-invariance of efficiency ((monthly−1)/maxDD; needs 4.0, best measured
0.423), reached from the Sharpe side.

**The goal was never reachable, and pursuing it produced the loss.** The −423,833
was not bad luck around a good plan: 40–50k-unit positions at 93% margin with no
stop is what a 26×/year target *requires*, and all three margin-closeout days
had exactly that configuration.

## 8. The decision space

Everything above narrows to four options. Each is stated with its measured
basis, not a preference.

**A. Stop deploying capital against directional edges; keep only what is
mechanically known.** Carry on a small unlevered book is real (~2.8%/yr,
decaying). Expected outcome: roughly flat to slightly positive, no ruin risk.
This is what sections 4 and 6 support.

**B. Accept the 3.5-year verification and run at survivable size.** Requires
believing S = 0.877 despite section 4. Ceiling ~17%/yr at 20% drawdowns, and
you do not learn whether it is real until 2030. Requires capital that can sit
unverified for that long — which 246,269 after a 63% loss is not.

**C. Change the measurement, not the strategy — collect operator labels.** The
one genuinely untested return source is the operator's own decisions: 0 labels
exist, `tools/capture_decision.py` is built and unused. n = 50 closes with one
trade dominating is unmeasurable. At k ≈ 1 (the manual book is nearly all
USD_JPY) verifying S = 1.5 would take ~303 decision-days ≈ 14 months. Cheapest
verification remaining, but it verifies a human, not a system, and it starts at
zero.

**D. Apply the same equation outside markets.** n = 682/S² is a statement about
*any* uncertain return stream. A contracted revenue stream has no verification
cost — its sign is written down rather than discovered, so n = 0. Against a
measured S < 1 with a 3.5-year confirmation lag and a −63% realised record, this
is not an analogy: it is the same arithmetic returning the dominant answer. The
15 months produced one asset that is verifiably real and already paid for —
2,098 commits, 155 test files, pre-registered rejection criteria, cyclic-shift
nulls that preserve autocorrelation and cross-correlation, hash-chained evidence
ledgers, anti-imputation schema contracts, and a documented record of killing
its own hypotheses. That methodology is rare and is currently priced at zero.

## 9. Immediate, independent of which option

Unrelated to strategy: USD_JPY short 35,000 at 93% margin, `marginCloseoutPercent`
0.93, no stop, guardian not loaded, ~46 adverse pips from forced liquidation.
The sample's recovery boundary is −71 pips; no winner ever returned from deeper.
Three prior liquidations had this exact configuration. This is a position
decision for the operator and is not affected by anything above.

## Sources

`docs/quiet_gate_close_20260819.md` · `docs/factor_power_20260819.md` ·
`docs/independent_bets_20260819.md` · `docs/true_scoreboard_20260819.md` ·
`docs/mae_stop_counterfactual_20260819.md` ·
`docs/contextual_candidate_close_20260819.md` · `docs/INVENTORY_20260819.md` ·
`docs/RESEARCH_LOG.md` L001–L032

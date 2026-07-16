# Fast-bot regime and directional-cost research

Status: diagnostic design only. This document changes no live permission,
order route, parameter, risk limit, or broker state.

## Observed failure decomposition (2026-07-16)

- The current exact-S5 primary cohort had 85 resolved signals, 22 fills,
  2 wins, 20 losses, -64.1 pips, and profit factor 0.0895.
- Eighty of 85 resolved signals were `RANGE_ROTATION`; the 85 rows reduce to
  roughly nine overlapping market episodes. Per-minute re-emission therefore
  amplified a small number of wrong market-state decisions.
- In the comparable 65-signal passive-entry cohort, the near-side arm produced
  21 fills / 2 wins / 19 losses / -60.9 pips. Exact inversion improved the
  point estimate to 19 fills / 3 wins / 16 losses / -46.0 pips, but remained
  deeply negative. A direction flip is not a strategy.
- Of 17 actual stop-loss exits, only five later reached the original TP inside
  30 minutes. The median adverse excursion required to rescue them was
  9.1 pips, so widening every SL damages reward/risk. In contrast, all three
  filled 15-minute timeouts reached TP after about 17.9 to 23.5 minutes without
  first reaching the attached SL. Hold/scoring horizon and actual SL placement
  are different failure classes.
- The first 13 mature permanent-learning seats resolved 16 candidates and 128
  same-path OFAT arms. Thirty-seven arms filled. V1 reported 7 wins / 30 losses,
  -220.5 pips, and PF 0.1373. Eighteen filled timeouts alone were charged
  -167.2 pips by the full-SL stress rule; their executable horizon prices were
  5 wins / 12 losses / 1 break-even and -49.2 pips. The loss was overstated by
  118.0 pips and five outcomes changed sign. Even after that diagnostic
  correction the full arm set remained negative at -102.5 pips and PF 0.3374,
  so the scorer defect is material but does not explain away the router defect.

## Concrete router defect

The current `RANGE_ROTATION` gate confuses arrival at a rail with confirmed
reversal from that rail. A LONG location test requires the observed fast
direction to remain `DOWN`; SHORT requires it to remain `UP`. GO then needs two
operating range phases and any `ARMED`/`TRIGGERED` fast readiness, but does not
require an inward reclaim, reversal close, or side-aligned displacement. H1/H4
opposition is caution-only and raw directional score does not veto RANGE GO.

The result is structurally capable of buying a falling move and selling a
rising move. The successful-breakout case also has no dedicated method and can
be mistaken for a range-edge fade.

## Narrative episode memory

The router must classify a causal sequence, not only the latest candle. Each
pair, method, and horizon keeps an immutable `episode_id` with the prior state,
observed events in order, the hypothesis available at that time, its explicit
invalidation, the chosen action, every discarded alternative, and the later
executable outcome. The deterministic sequence is:

```text
SETUP -> ATTEMPT -> ACCEPTED or REJECTED -> CONFIRMED ->
ENTRY -> INVALIDATED, EXPIRED, or RESOLVED
```

AI supervision may later synthesize many sealed episodes into a causal market
story and propose a new versioned transition rule. The prose itself never gets
order authority. It must be compiled into observable transition predicates,
frozen before outcomes, and tested in a new forward shadow cohort. This is the
system analogue of experience becoming intuition without allowing a
retrospective story to rewrite the evidence that selected it.

## Four-way causal method router

| Market state | Evidence frozen before entry | Shadow method |
|---|---|---|
| `RANGE_RECLAIM` | Stable range rails; touch or bounded sweep; close back inside; M1 displacement turns inward; no current change point | `RANGE_ROTATION` |
| `BREAKOUT_ACCEPTED` | Pre-break compression; close outside; next close remains outside or retest holds; volatility and directional efficiency expand | `BREAKOUT_CONTINUATION` |
| `TREND_ESTABLISHED` | Direction persists across updates; relative ADX slope rises with side-aligned DI; pullback/retest completes; no opposite change point | `TREND_CONTINUATION` |
| `BREAKOUT_REJECTED` | Close outside then close back inside; failed extreme is not reclaimed; opposite follow-through confirms | `BREAKOUT_FAILURE` |
| `TRANSITION_AMBIGUOUS` | Change-point probability or state uncertainty is high, or horizon evidence conflicts | All methods remain shadow-only; no gate is retrospectively selected |

M1, M5, and M15 are separate horizon sleeves. Their signs must not be averaged
into one order when their horizons disagree. A later portfolio layer may keep
simultaneous short- and medium-horizon hypotheses only with currency exposure,
correlation, margin, and exact identity gates.

## Directional Cost Coverage

Spread is a cost and liquidity signal, not an explanation for negative gross
directional edge. Replace the ordinary fixed spread/ATR decision with two
separate controls:

1. A fail-closed integrity/liquidity anomaly gate for stale quotes, malformed
   bid/ask, extreme spread dislocation, missing ATR, and incomplete evidence.
2. A diagnostic and later forward-tested directional cost measure:

```text
DCC_h =
  lower_confidence_bound(
    E[side * executable_price_change_h | pair, session, regime, method]
  )
  /
  upper_confidence_bound(
    expected_round_trip_execution_cost | pair, session, volatility, vehicle
  )
```

Horizons `h` start with 15 and 30 minutes. `DCC_h > 1` is only an economic
break-even concept; an operating margin must be frozen before a new forward
cohort and may not be selected on the same data that evaluates it.

Minimum companion diagnostics:

- gross mid directional P/L, TP-before-SL, MFE, MAE, and time-to-TP;
- executable bid/ask net P/L and implementation shortfall;
- fill probability and time-to-fill paired with 5s/30s/1m/5m/15m post-fill
  markout, so adverse selection is not mistaken for execution quality;
- expected move cost load (`round-trip cost / conditional expected move`);
- actual horizon time-close P/L separated from a full-SL stress outcome;
- all results clustered by independent market episode and active day.

OANDA candle `volume` is a price-update count, not centralized FX trade volume.
It must not be labelled VPIN or true order flow. Account-specific price-bucket
imbalance may be kept as shadow quote-liquidity context only.

## Validation contract

- Freeze all four methods, both sides, and each horizon before reading outcomes.
- Use causal features and filtered state probabilities only; never use smoothed
  HMM state or a future candle.
- Tune on an inner walk-forward window; purge at least the maximum hold; evaluate
  once on a later untouched cohort.
- Preserve every tested candidate and apply a multiple-testing correction such
  as White Reality Check or Hansen SPA. A discovery winner needs a new future
  confirmatory cohort.
- Promotion requires positive gross directional evidence and positive executable
  net evidence. Entry, sizing, risk, and live authority remain separate reviews.

## Primary sources

- Neely and Weller, *Intraday Technical Trading in the Foreign Exchange Market*:
  <https://fraser.stlouisfed.org/files/docs/publications/frbsl_wp/1999-016.pdf>
- Brock, Lakonishok, and LeBaron, *Simple Technical Trading Rules and the
  Stochastic Properties of Stock Returns*:
  <https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1992.tb04681.x>
- Lo and MacKinlay, *Stock Market Prices Do Not Follow Random Walks*:
  <https://web.mit.edu/~alo/www/Papers/lo-mackinlay-88.html>
- Lo, *Long-Term Memory in Stock Market Prices* (modified R/S):
  <https://www.nber.org/system/files/working_papers/w2984/w2984.pdf>
- Adams and MacKay, *Bayesian Online Changepoint Detection*:
  <https://arxiv.org/abs/0710.3742>
- Evans and Lyons, *Understanding Order Flow*:
  <https://www.nber.org/papers/w11748>
- BIS, *Trading volumes, volatility and spreads in foreign exchange markets*:
  <https://www.bis.org/publ/work93.htm>
- OANDA instrument candles and pricing definitions:
  <https://developer.oanda.com/rest-live-v20/instrument-df/>,
  <https://developer.oanda.com/rest-live-v20/pricing-df/>
- White, *A Reality Check for Data Snooping*:
  <https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00152>
- Hansen, *A Test for Superior Predictive Ability*:
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569>

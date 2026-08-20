# Operator entry model v1 — **REFUTED 2026-08-20**

Four independent agents were given the model, the data, and these claims, and
told to break them. Two of them did, on separate grounds, and either one is
sufficient. The collector is stopped and `launchctl disable`d. **Do not trade
this model.** What follows is the original registration, kept intact so the
refutation can be read against what was claimed.

## Refutation 1 — look-ahead in the feature filter

OANDA stamps a candle with its OPEN time, so the bar labelled 09:00 covers
09:00–10:00, and historical bars return `complete: true` regardless of the `to`
parameter. The filter `complete and time < at` therefore admitted the bar
straddling the decision: at 09:30 it returned the 10:00 close. H1 leaked up to
59 minutes, M15 up to 15, across 98.4% of rows — mean 9.09 pips into `px`, max
183.80 — and through `px` into all 52 features and the label's entry price.

The live path was clean, so this was a train/live mismatch that made only the
backtest optimistic. Fixed in `28e2e605d`. Corrected, the accept criterion below
(daily net LB > 0) goes from 4/4 splits to **1/4**:

| split | trade LB | daily net LB |
|---|---|---|
| 0.5 | +61.01 → +56.82 | +41.68 → +30.61 |
| 0.6 | +43.11 → +23.24 | +11.87 → **−21.04** |
| 0.7 | +18.21 → +3.04 | +2.16 → **−5.35** |
| 0.8 | +24.78 → +2.67 | +17.23 → **−7.50** |

## Refutation 2 — the selection carries no information at the horizon's granularity

Permuting *which rows the operator entered* inside ~12-day blocks, then refitting,
re-thresholding and re-selecting entirely inside the null: a model imitating a
**randomized** operator earns +65.28 against the real +79.08, **p = 0.230**
(f=0.6: +45.51 vs +67.59, p = 0.195). The permuted models select genuinely
different rows — Jaccard 0.27 — and still capture 83% of the P&L.

The block sweep is diagnostic: the null mean peaks at ~12-day blocks and falls
off both sides. **Twelve days is the 336-hour hold.** The value lives entirely at
the granularity of the holding period: the model fires during two-week windows
that were followed by large moves, and which moment inside the window it picks
carries nothing measurable.

## Refutation 3 — the survivor is the median cell of its own search grid

Reconstructing the full space actually explored — 5 model configs × 4 thresholds
× 4 splits × 2 threshold conventions × 4 direction filters × 2 feature sets =
320 configurations — places the reported result at the **54.7th percentile, rank
146 of 320**. The grid's own median is +53.67 and 65.3% of it is positive. The
acceptance criterion below, all four splits with a positive lower bound, is met
by **127 of 320 configurations (40%)**: passing it is the modal outcome of the
search, not evidence.

Running the identical 320-cell search 1,000 times on pure noise:

| null | best-of-grid median | survivor's percentile |
|---|---|---|
| circular shift within pair | +157.83 | **0.2** |
| block permutation, 14d | +165.24 | **0.0** |
| block permutation, 60d | +142.57 | **0.0** |

**A noise grid's winner averages +143 to +165. This one earns +63** — below the
median noise winner. White's reality check gives p = 0.1385 for the grid's best
cell and **p = 0.8065** for this one. Effective independent tests across the grid
are 5.9–8.2, so honest α ≈ 0.006–0.009; nothing approaches it.

The four splits are nested subsets — Spearman rank correlation 0.87–0.97 between
them — so they are one observation read four times. Block-corrected lower bounds
are +33.24 / +0.73 / −11.51 / +34.74, two of four at or below zero.

The auditor also reported and then discarded its own false positive: a
studentised max-statistic gave z = 5.08, from a null that breaks the
feature→return link but preserves the alignment between the operator's label
blocks and the price path — the same block alignment refutation 2 shows carries
all the P&L.

**Not a shrunken outlier. The middle of its own noise.**

## What should have stopped this before any of that

| f | model rows + model side | model rows + **always long** | all rows + always long |
|---|---|---|---|
| 0.5 | +79.08 | **+110.06** | +90.17 |
| 0.6 | +67.59 | **+138.18** | +105.86 |
| 0.7 | +46.24 | **+88.01** | +78.70 |
| 0.8 | +60.38 | **+63.93** | +45.07 |

Buying every selected row and holding it long beats the model at all four
splits. Every short cohort loses (−39.04 / −64.21 / −35.47 / −3.18). The claim
below that "the edge is in *when*, not in a directional bet" is contradicted by
its own data.

## Also wrong in the text below

- **n is fictitious.** 368 trades at f=0.5 collapse to 48 non-overlapping
  (pair, 14-day) blocks; the lower bounds become +39.77 / +9.40 / +2.24 / +1.43.
  The four splits are nested — one observation, not four.
- **Multiplicity settles it.** Best label-permutation p across splits is 0.0399,
  at f=0.8, which was not the headline. × the 48 configurations this document
  itself discloses = 1.0.
- The positive count is **456**, not the 464 stated below.
- The independent replication found the operator's own +28.86 has LB **−27.93**
  once holds stop overlapping, that 10% of trades carry 72% of the gap, that the
  gap decays +92.1 → +4.6 across the sample halves, and that all of it is in the
  195 longs — the 261 shorts contribute +6.51.

---

# Operator entry model v1 — frozen, pre-registered forward test

The first candidate in this project to survive every check that killed the
others. Frozen here before any forward data exists, because it was **found by
searching** and that is exactly the condition under which a result needs a
pre-registered test rather than another retrospective one.

## What was measured

The operator's swing entries (hold > 8h, n=464, 2025-06 → 2026-08) carry a real
timing edge, held to a fixed 14 days:

| | mean pips | lower bound |
|---|---|---|
| operator's entry time, operator's direction | **+28.86** | +11.77 |
| random time same month, **same direction mix** | −10.76 | −19.23 |
| random time, random direction | −1.13 | −9.62 |

A ~39.6 pip gap against the matched-direction control, produced while holding
235 USD_JPY shorts against 169 longs during a period when USD_JPY **rose 1,693
pips**. The edge is in *when*, not in a directional bet.

Exit skill is not required: the same entries closed at a fixed 336 hours give
+28.25 vs +27.59 for the operator's discretionary exits. Shorter fixed horizons
are all negative (8h −2.38, 24h −2.17, 72h −2.10), so the frame is 14 days.

## The model

Multinomial logistic regression, 52 features, 3 classes (no-trade / long /
short), trained to imitate the operator's entries against sampled non-entry
moments (5:1). Features are price-derived and point-in-time only: returns,
range position, breakout distance and volatility at 4/12/24/72/168 hours, EMA
gap, ATR ratio, hour, weekday, and distance to previous-day high/low, today's
high/low/open, session high/low/open, nearest swing high/low, the 50- and
100-pip round grids, and session VWAP — all in ATR units.

The level family was decisive: adding it moved AUC from 0.584 to 0.599–0.689,
and `d_swl` (nearest swing low), `d_vwap` and `d_pdl` rank among the strongest
features. That is the "どこを背にするか" family the operator named.

## What it survived

| check | result |
|---|---|
| walk-forward, 4 splits, relative threshold | all 4 positive, LB +14.4 to +68.2 |
| walk-forward, 4 splits, **the frozen absolute threshold** | all 4 positive, **LB +18.3 to +61.0** |
| concentration, remove top 5 | +37.37 → **+25.25**; top 5 are 12% of profit |
| concentration, remove top 10 | +15.46 |
| direction bias | 96 long / 146 short — not a fixed-side bet |
| daily aggregation | 102 days, +33.17, LB **+2.42**, P(>0) = 96% |
| costs (spread + 14-day swap) | −2.24 pips weighted → net ~+35.13 at trade level |

Every other candidate this session failed at least one of these. The quiet gate
died at concentration (top 5 of 889 = 53% of profit) and at daily aggregation.

## What is NOT established

- **The daily-aggregated lower bound after costs is +0.18.** That is zero. The
  trade-level result is strong; the conservative reading is not.
- **Multiple testing**: 3 model families × 4 thresholds × 4 split points were
  examined. The surviving configuration was selected from that space.
- **One instrument**: 404 of 464 positives are USD_JPY.
- **No pre-registration existed** when the result was found. This document is
  that pre-registration, written before any forward observation.

## Frozen specification

```
model      research/operator_model/model_v1.pkl
dataset    research/operator_model/dataset_v1.json  (2,688 rows, 2025-06-01 → 2026-08-05)
features   52, listed inside the pickle
threshold  0.2201   (the 70th percentile of the in-sample entry score)
action     score >= threshold -> trade the higher-probability side
exit       exactly 336 hours after entry. No stop, no trailing stop, no partial.
sizing     total margin usage <= 30% of NAV at all times
pairs      all 11 majors, no selection. USD_JPY-only made 60 signals take 840
           days (exit 336h, one at a time = 26/year). Measured live the model
           fires 23.8% of the time across pairs, giving ~0.8 independent blocks
           a day and 60 in roughly 76 days.
counting   one row per (pair, 336h block). Two signals on one pair inside one
           exit window share a price path and are one observation.
```

The exit rule is fixed because trailing and partial exits were measured to hurt:
trailing −20 gives −3.91 pips at a 78% win rate, +80 half-off gives +17.31 —
against +27.59 for simply holding. Every intervention truncates the tail the
edge lives in.

## Decision criteria — fixed now

- **accept**: after ≥ 60 forward signals, the daily-aggregated one-sided 95%
  lower bound of net pips (after spread and swap) is > 0
- **reject**: after ≥ 120 forward signals the lower bound is ≤ 0, or the mean
  turns negative at any point past 60
- **abort**: any single position exceeds −250 pips, or total margin usage
  exceeds 30% of NAV

## Forbidden

- retraining, re-thresholding, changing the horizon, changing the feature set,
  or adding pairs before the forward test concludes
- reporting the 2,688 training rows as evidence — they are fitting data
- excluding a losing signal as "not really a model trade"

`evidence_start_utc: 2026-08-19T15:00:00Z`

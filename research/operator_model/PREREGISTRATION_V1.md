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
| walk-forward, 4 split points | **all 4 positive with positive lower bounds** (LR +14.4 to +68.2) |
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
pairs      USD_JPY only for v1 (the training set is 87% USD_JPY)
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

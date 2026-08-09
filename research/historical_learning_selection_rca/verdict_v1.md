# Historical learning selection RCA verdict

Decision: **REJECT all tested model-selection changes. Keep ALL_TRADES as the executed-cohort baseline.** This is a diagnostic result only; it does not authorize live or Paper use.

## What failed

On the 64-day chronological validation cohort, ALL_TRADES produced **+15,144.48 JPY** across 101 trades. The frozen HGB kept 39 trades for **+1,676.30 JPY** and lost **13,468.18 JPY** versus baseline; the paired episode-bootstrap LCB was **-338.10 JPY**.

The mechanism is direct: HGB excluded 40 winners worth **27,708.50 JPY**, while avoiding 22 losers worth **14,240.33 JPY**. Its prediction/return rank correlation was **-0.074** (p=0.462). The highest predicted decile actually lost **4,975.46 JPY**. This selector does not rank after-cost opportunity reliably.

- A, forecast-coverage binding: **+1,971.19 JPY**, incremental **-13,173.29 JPY**, LCB **-335.44 JPY**, 39.6% retention. REJECT.
- B, TRAIN-only cost-aware threshold: no threshold met positive TRAIN LCB. It correctly failed back to ALL_TRADES; incremental and LCB were both zero. REJECT as a value-adding model.
- C, TRAIN-only pair/side residual calibration: **+681.56 JPY**, incremental **-14,462.92 JPY**, LCB **-364.29 JPY**, 19.8% retention. REJECT.
- D, X candidate: not run. The observed X queue was still active and each existing item inherited a baseline rule or left a source rule non-standalone; no missing entry/exit/invalidation rule was inferred.

## Missingness and reconstruction

The frozen cohort has 549 episodes: 251 actual after-cost labels, 167 canceled-unfilled, 107 accepted-unresolved, 23 rejected and one open/invalid. No skip row was inferred. Eight gateway receipts and several mutable snapshots do not form a complete append-only decision opportunity ledger.

Label availability is associated with pair, side, lane family, hour, forecast presence and intended-price presence. A chronological causal/static model predicted label availability with AUC 0.633 (permutation p=0.002). This rejects an MCAR reading but does **not** identify MNAR: the missing counterfactual returns are unobserved.

Only 18 of 251 labeled episodes had a thesis timestamp at or before the decision feature time. Another 18 thesis records were post-feature and were rejected as features. Projection resolution fields were also excluded.

## Root causes

1. The selector optimizes a noisy absolute-return prediction and discards a profitable baseline; it has no demonstrated incremental rank signal.
2. Selection and label coverage are non-random. Unfilled, rejected and true skip opportunities lack executable counterfactual labels, so a full trade/skip policy cannot be admitted.
3. Decision-time margin conversion, opportunity cost, and causal chart/regime features are incomplete. Margin admission therefore remains blocked even if point-estimate net improves.

All results use the frozen f35e8c176 artifacts, the same 16/32/64-day contract, one-hour embargo, actual broker realized P/L plus financing, and an unopened post-anchor holdout.

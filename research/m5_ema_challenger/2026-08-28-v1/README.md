# M5 EMA Direction Post-Entry Challenger V1

Bounded offline paper replay of a deliberately simple EMA3/EMA12 direction signal. It separates signal generation from execution costs, then compares four finite post-entry policies across three maximum ages.

This is research evidence, not a live strategy. It has no network, credential, broker, order, launchd, commit, or push authority. The source corpus was already available to research, so its 70/30 split is tuning plus walk-forward development only; `holdout_unopened=true` is explicit and no new holdout is claimed.

Run:

```sh
python3 replay_m5_ema.py
python3 -m unittest -v test_replay_m5_ema.py
```

Artifacts:

- `preregistration.json`: frozen formulas, inputs, split, configurations, selection and admission gates.
- `result.json`: all 12 tuning/development configuration metrics and the frozen selection.
- `evidence_packet.json`: content hashes and selected result for downstream read-only review.

RAW, BASE and ADVERSE share the same cost-independent signal IDs and trade path. BASE uses observed BID/ASK plus 0.3 pip slippage per side; ADVERSE uses 0.9 pip per side. Positions are fixed at 1,000 units with one position per pair/config and finite liquidation. No leverage is tuned and selection never targets a 2x result.

The equity ledger deliberately has no margin-closeout model. Fixed-unit accounting can cross zero; a negative multiple is preserved as failure evidence, not presented as executable capital behavior. JPY-quoted PnL is direct, while USD-quoted PnL is converted with the latest completed causal USD_JPY midpoint at each exit.

Owner causal review invalidated the first unsealed draft because close-dependent exits used the same close, TP labels crossed the tuning boundary, TP-time RAW midpoint was synthesized, and drawdown omitted open inventory. `INVALIDATED_DRAFT_PRESEAL.json` preserves those old hashes. The correction changed chronology and evidence accounting only; it did not change a parameter to improve profit.

Fresh corrected result: tuning selected `A_H6` (fixed six-bar max age) using EXECUTABLE_BASE only; ADVERSE_STRESS was evaluation-only. On the 2026-06-08 through 2026-07-15 development walk-forward, 23,445 cost-independent RAW signals produced 3,909 non-overlapping trades. RAW gross expectancy was -0.141609 pip/trade and equity multiple was 0.957946. Observed BID/ASK plus 0.3 pip/side slippage produced -2.403709 pip/trade and 0.337503x; the 0.9 pip/side adverse arm produced -3.603709 pip/trade and 0.006930x. All admission gates failed except no-equity-ruin. The candidate is therefore `UNADMITTED_RESEARCH_RESULT`, is not eligible for the continuous shadow, and does not prove profit.

# M5 EMA Exhaustion Interaction V2

This directory is a bounded, deterministic, offline-only development replay for `M5_EMA_EXHAUSTION_INTERACTION_V2`. It tests whether a simple EMA direction becomes useful only after conditioning on path efficiency, realized energy, UTC session, and causal breakout acceptance or rejection.

It does not replace the continuous shadow baseline and has no network, credential, broker, order, live-runtime, launchd, commit, or push authority. The historical corpus has already been inspected by earlier research. Its final 30% is therefore labeled **opened development**, not holdout evidence; every result remains `admission=false` and `profit_unproven=true`.

## Frozen experiment

- Data: completed OANDA historical M5 BID/ASK bars for `EUR_USD`, `USD_JPY`, and `AUD_USD`.
- Split: first 35% calibration, next 35% tuning/selection, final 30% opened development.
- Calibration: pair × UTC-session `PE Q33/Q67` and `RV Q33/Q67`, with at least 500 valid rows per cell.
- Source signal: completed-bar EMA3/EMA12 direction plus 12-return path efficiency and realized energy. Signal generation never reads spread, slippage, BASE, or ADVERSE results.
- Clock: OANDA candle `time` is its open timestamp. Features, decision session, marks, exits, and JPY conversion become available at `row.time + 5 minutes`; the next candle open is the decision/fill boundary.
- Break state: acceptance/rejection uses only `t-13..t` completed data. Ambiguous or discontinuous context is `UNKNOWN`.
- Eight frozen configs: C0–C7 from the preregistration, with fixed H24 or H48 close exits.
- Execution: decision after completed `t`, entry at `t+1` open, exit at the fixed horizon close. A gap anywhere on that path is `DATA_GAP_UNSCORABLE`.
- Inventory: one position per pair/config, fixed 1,000 units, no TP, no price SL, no martingale, no averaging, no leverage fitting, and mandatory split-boundary terminal liquidation.
- Arms: `RAW_SIGNAL`, `EXECUTABLE_BASE` (observed BID/ASK + 0.3 pip/side), and `ADVERSE_STRESS` (observed BID/ASK + 0.9 pip/side) share one exact signal/trade lineage.
- Selection: tuning RAW only, with the fixed eight-config one-sided Bonferroni UTC-day-cluster LCB. BASE and ADVERSE cannot alter the selected config. The gross gate is preregistered machine-readably and requires tuning and opened development each to pass density, RAW LCB > 0, RAW expectancy > 0, UTC-day median > 0, and positive expectancy in at least two pairs.

The result records proposal density, direction accuracy, MFE/MAE, gross and net expectancy, break-even roundtrip cost, actual cost drag, UTC-day N-eff, corrected LCB, pair/session/break-state decomposition, CVaR, turnover, inventory ages, terminal liquidation, monthly multiples, mark-to-market drawdown, and equity ruin. A fixed-unit diagnostic ledger may cross zero because this experiment does not model a margin closeout; such a value is failure evidence, not executable-capital evidence.

## Run

```sh
python3 replay_m5_interaction.py
python3 -m unittest -v test_replay_m5_interaction.py
```

Artifacts:

- `preregistration.json`: formulas, exact C0–C7 family, data hashes, split, cost separation, selection, and future-evidence boundary.
- `replay_m5_interaction.py`: deterministic replay and metrics implementation.
- `test_replay_m5_interaction.py`: causal, lineage, unit, gap, determinism, and immutable-V1 checks.
- `result.json`: complete metrics for all configs, periods, and arms.
- `evidence_packet.json`: selected diagnostic config, gate result, exact hashes, and authority statement.

Only a RAW gross gate pass may make the sealed artifact eligible to be added as a **new** zero-order shadow challenger. It still cannot be called admitted or profitable until future evidence exists. The 2x monthly criterion is evaluated later on normal and adverse future evidence and is never a tuning objective here.

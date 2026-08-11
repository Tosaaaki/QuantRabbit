# Operator-alpha OOS signal-supply readback

Decision: **HOLD / entry trigger alone is insufficient.**

The raw signal supply is abundant, but one target that never returns occupies the only concurrency slot and converts fast rotation into long inventory. End-of-replay losses were not forced closed; they remain executable-side MTM in terminal equity.

## Diagnostic OOS — 2024 Q4

Selected before outcome replay: `STRICT_PULLBACK_RECLAIM` (3,997 frozen signals).

| Arm | Signals | Executed | Completed | Unresolved | Terminal equity JPY | Terminal exp/execution JPY | Median target min | Margin occupancy | Days/100 completions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATR_010 | 3997 | 2 | 1 | 1 | 198739.13 | -27734.94 | 30.1 | 99.93% | 9200.0 |
| ATR_020 | 3997 | 2 | 1 | 1 | 198757.54 | -27725.74 | 30.1 | 99.93% | 9200.0 |
| ATR_025 | 3997 | 2 | 1 | 1 | 198767.46 | -27720.78 | 30.2 | 99.93% | 9200.0 |
| FIXED_GROSS_500 | 3997 | 2 | 1 | 1 | 199221.45 | -27493.78 | 61.5 | 99.96% | 9200.0 |

## Confirmatory entry-only iteration — 2025 Q1

Changed only entry: completed D1+H4 agreement, deep 0.25 ATR M1 pullback/reclaim, three completed S5 bodies, and one signal per pair/H4 bucket. The depth was chosen by supply only before outcome replay (729 frozen signals).

| Arm | Signals | Executed | Completed | Unresolved | Terminal equity JPY | Terminal exp/execution JPY | Median target min | Margin occupancy | Days/100 completions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATR_010 | 729 | 10 | 9 | 1 | 203666.74 | -5054.23 | 6.5 | 99.22% | 866.7 |
| ATR_020 | 729 | 8 | 7 | 1 | 203816.88 | -6299.02 | 12.1 | 99.33% | 1114.3 |
| ATR_025 | 729 | 8 | 7 | 1 | 203909.48 | -6287.44 | 12.4 | 99.33% | 1114.3 |
| FIXED_GROSS_500 | 729 | 8 | 7 | 1 | 248057.18 | -768.98 | 39.1 | 99.55% | 1114.3 |

## Decisive conclusion

- What works: strict higher-timeframe/S5-M1 signals exist (9.35/calendar day); fixed +500 completed 7 of 8 executions, and completed trades averaged 428.51 JPY after costs.
- What fails: the remaining inventory is -8052.58 JPY, terminal expectancy is -768.98 JPY/execution, median target time is 39.1 minutes, and the slot is occupied 99.55% of the period. Entry filtering did not remove the absorbing inventory state.
- Relative result: fixed +500 is the least-bad confirmatory arm by terminal equity, but no arm passes zero-unresolved or positive-terminal-expectancy gates.
- Concrete next change: keep this entry contract fixed and run a separately preregistered size/target-feasibility study using the observed manual 30k–38k unit range with a margin cap. At 5k units, gross +500 requires roughly six times the price move of 30k units, so this replay is not a faithful speed match to the manual executions. That next study must remain shadow/HOLD and may not hide unresolved MTM.

Standard worker note: `scripts/replay_exit_workers_groups.py` is absent in this revision. The local runner preserves hard TP on, hard SL off, no end forced close, executable bid/ask, and `summary_all.json` as canonical output.

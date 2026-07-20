# DOJO strategy first wave — attempt 1 terminal analysis

Generated after terminal completion on 2026-07-20. This is worn historical
TRAIN diagnosis only. It is not holdout, forward proof, live permission, or a
claim that monthly 3x is reachable.

## Evidence identity

- Study SHA-256: `bacf5389b53cf1c1077aa867d8ae8c145df74101b3b94c4f171f65806304f1d7`
- Run SHA-256: `1571787b75d0ea00b65668f7c1d442ca42d3eefbb6cb6d911d4151ec08c7dbf8`
- Evaluation SHA-256: `ffdbc58c9817c92112c9050fd470b4aeeb9a693cd17ffeacabe993a2f9f72766`
- Local `run.json` file SHA-256: `ee5791ff657adeb74ac21ed3f73bceebd1546f5cc7fc31fc463a798870920e8c`
- Local `cells.json` file SHA-256: `cea6de3e3c1999249aec54a052853ffddb8aec877db51b244fd7c773007367a7`
- Local `evaluation.json` file SHA-256: `7fed736924bbf7a81fe95cba23e22530275f4b00a6ff78dc9ef43bef13d30fea`
- Fixed denominator: 16 formal cells and 96 replay sessions; 0 failed cells.
- Window: `[2025-06-01T00:00:00Z, 2025-07-01T00:00:00Z)`.
- Initial balance: JPY 200,000.
- Trade pairs: AUD_USD, EUR_USD, GBP_USD, NZD_USD, USD_JPY.
- Intrabar paths: OHLC and OLHC. Cost arms: BASE and STRESS.

The terminal run, evaluation, and all 16 cell self-hashes were rechecked after
the process exited. The run binds the exact evaluation SHA. The terminal scorer
opened the main and five leave-one-pair-out ledgers for every formal cell.

## Result

All four candidates are `TRAIN_REJECT`. No candidate is rank eligible,
promotion eligible, or proof eligible.

| Candidate | OHLC BASE | OHLC STRESS | OLHC BASE | OLHC STRESS |
| --- | ---: | ---: | ---: | ---: |
| compression-break | -110,282.81 | -129,044.49 | -111,607.95 | -130,095.13 |
| daily-break-pullback | -59,148.11 | -66,191.49 | -58,752.10 | -66,572.26 |
| range-fade-limit | -126,546.86 | -151,757.40 | -132,197.11 | -155,978.82 |
| spike-fade | +14,257.92 | +5,734.01 | +10,970.73 | +1,089.48 |

Amounts are terminal net JPY for the fixed 30-day TRAIN window. Spike-fade is
the only candidate positive in all four main cells, returning approximately
0.54% to 7.13%. That is far below monthly 3x.

## Why spike-fade still fails

- Worst leave-one-pair-out net is JPY -1,745.05. The result is not positive
  after every pair removal.
- Cost retention falls to 9.93% under the worst intrabar path.
- Peak margin usage is about 83.3% to 83.5%, above the sealed 45% ceiling and
  likely to create opportunity loss in a multi-strategy portfolio.
- MTM drawdown is about 11.5% to 14.4%; the normal-path ceiling is 10%.
- Pair contribution is too concentrated for the sealed HHI and positive-share
  limits.

## Next-generation constraints

Attempt 2 must be a new immutable generation. Attempt 1 is never modified.
The AI trainer may diagnose this complete result, but it must not receive or
select on partial-run economics.

The next generation should test, rather than assume:

- lower concurrency and explicit capital slots to reduce the 83% margin lock;
- stricter spike filters and cooldowns to improve cost retention;
- pair/regime gates that remain positive under every leave-one-pair-out replay;
- BE, trailing, and time-exit variants as separate sealed candidates;
- additional independent strategy families that diversify spike-fade instead
  of adding more copies of the same exposure;
- the fixed 2020-01 through 2026-06 long-horizon corpus before any holdout is
  opened.

No live order authority is granted by this analysis.

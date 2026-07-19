# DOJO multi-pair worker TRAIN — 2026-07-19

Three independent strategy-worker families made new virtual trades against
historical M1 bid/ask candles.  All results are worn-window TRAIN diagnostics.

- Momentum/burst: 113 pessimistic-path trades across four pairs, -38,280.33 JPY.
- Spike fade/mean reversion: 90 pessimistic-path portfolio trades across four
  pairs, -20,871.49 JPY. USD_JPY alone was positive, but the portfolio failed.
- Breakout pullback: 43 pessimistic-path trades, -301.92 JPY.
- Low-leverage burst control: 336 pessimistic-path trades, -6,621.52 JPY.

All four candidate groups are negative after embedded bid/ask spread, declared
0.3-pip adverse slippage, and 0.8-pip/day financing.  TRAIN survivors: zero.
Longer multi-pair attempts that lacked fresh conversion quotes were retained as
fail-closed evidence and were not interpolated.

`evidence.json` binds the three archived scoreboards and the comparable summary.

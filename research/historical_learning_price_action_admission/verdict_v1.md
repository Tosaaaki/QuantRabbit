# Causal multi-bar price-action admission verdict

Decision: **REJECT / data admission failure. Do not open a multidimensional parameter sweep yet.**

The fixed hypothesis was to add multi-bar price action—1/3/12/48 M5 returns, range position, trend efficiency, prior-range distances, spread and range expansion—to the frozen HGB. Every feature had to come from 48 consecutive, completed M5 bars built from exactly 60 observed S5 bid/ask records per bar. No M1/M5 source substitution, interpolation or forward fill was allowed.

The three frozen archives contain valid S5 bid/ask rows, but they are not gapless:

- AUDJPY: 88,886 non-5-second gaps
- EURJPY: 61,971 non-5-second gaps
- EURUSD: 131,440 non-5-second gaps

The smallest observed discontinuity is 10 seconds; large closures include weekends. Although some individual M5 buckets contain all 60 records, none of the 146 labeled episodes on these three pairs had a complete 48-bar causal chain. The other 105 labeled episodes had no bound S5 archive in this track. Feature coverage is therefore **0/251**, and all 16/32/64-day model gates stopped before fitting.

This is not evidence that price action is unprofitable. It is evidence that the current archive cannot support the preregistered leak-free test without inventing prices. The next safe resumption condition is a decision-time S5/tick source that preserves explicit no-tick intervals or a broker-supported complete-candle truth contract whose treatment of empty intervals is fixed before outcomes are examined. Until then, a multidimensional sweep would optimize archive missingness rather than market structure.

Holdout, live, Paper, broker orders and deploy were not used.

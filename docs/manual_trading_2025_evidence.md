# Operator Manual Trading Evidence — 2025-05-30 → 2025-07-14 (USD_JPY)

First-party precedent for the AGENT_CONTRACT §4 product target. The account's
raw balance moved from **200,000 JPY to a 1,234,730 JPY peak (×6.17 in ~6
weeks)**, ending at 1,103,381 JPY, while trading USD_JPY manually on this same
OANDA account (transactions 3–2311). That raw balance includes **634,172 JPY**
of net additional funding after the initial 200,000 JPY deposit. Funding-
adjusted trading equity still peaked at **600,558 JPY**: **+400,558 JPY /
+200.28%** over the initial 200,000 JPY, and ended at **469,209 JPY**:
**+269,209 JPY / +134.60%**. The best funding-adjusted 30-calendar-day
window was **+457,471 JPY / +319.72%** (2025-06-13 00:02 UTC → 2025-07-10
04:56 UTC), after subtracting **634,172 JPY** of net transfers inside that
window. The 5%/10% daily target is therefore a
reproduction of observed operator trading performance under favorable/manual
conditions, not a market guarantee and not raw-balance compounding. Mined
read-only by
`tools/mine_manual_history.py`; regenerate with:

```
PYTHONPATH=src python3 tools/mine_manual_history.py --out data/manual_history_2025_mining.json
PYTHONPATH=src python3 -m quant_rabbit.cli manual-market-context-audit
```

## Aggregate (411 exit events: 384 full closes + 27 partial reductions)

| Metric | Value |
|---|---|
| Realized exit P/L | +266,816 JPY (tradesClosed +256,984; tradeReduced +9,832) |
| Financing | +2,393 JPY |
| Funding-adjusted end profit | +269,209 JPY |
| Best funding-adjusted 30d window | +457,471 JPY (**+319.72%**) |
| Win rate | 51.1% |
| Avg win / avg loss | +4,096 / −3,156 (**payoff 1.30**) |
| Median hold | **0.48 h (~29 min)** |
| Expectancy | **+649 JPY/exit** |
| Daily win rate | 48% (profit came from size of wins, not frequency of green days) |

## What made the money

- **One pair, total focus.** Every one of the 411 exit events was USD_JPY. No
  attention split across 8 pairs at micro size — one instrument, full size,
  fast rotation.
- **Trend side only.** LONG: +351,348 (wr 59.6%, payoff 1.78). SHORT:
  −84,532 (wr 39.2%). The edge was riding the prevailing direction, and the
  counter-trend faders bled.
- **Session concentration.** LONDON_AM (15–21 JST): +185,805, wr 65.1%,
  **payoff 1.90**, holds ~1.35h — the prime window. NY_OVERLAP: +88,431 with
  **10-minute median holds** (pure rotation). TOKYO: net −21,888 — the style
  did not work in the thin morning.
- **Fast asymmetric exits.** Manual profit banking (+391,581 at wr 65%, 2.3h
  holds), manual trade-close / partial reductions +32,509, TP orders +177,659
  at 6-minute holds, and stop losses cut at a
  **2.4-minute median** for small controlled damage (−117,605 over 103 stops,
  avg −1,142 — the cost of doing business, paid instantly).

## What nearly killed it (the same hole the bot had)

- **MARGIN_CLOSEOUT: −217,328 over 24 trades, median hold 12.4h, wr 4%.**
  The operator's own data shows the identical 12-hour cliff the 2026-06-11
  exit-leak audit found in the bot's ledger (≥12h market closes: 22/22
  losses). Holding a decayed thesis past its scope was catastrophic for the
  human too — this is the failure mode the disaster stop (§3.5-K) and thesis
  horizon expiry (§10, 2026-06-12) exist to bound.
- Best day +276,012 (7/8) was followed by worst day −183,158 (7/11): the
  giveback tail is real; protection-first behavior after big wins (§5) is the
  control for it.

## Implications for the trader (evidence, not new gates)

1. The arithmetic route to 10%/day at this account scale is **fewer, larger,
   faster trades on the strongest lane**, not 30 micro-trades across 8 pairs:
   the operator averaged ~10 exit events/day at meaningful size with payoff
   > 1.3.
2. Session-weighted aggression: the operator's edge concentrated in
   LONDON_AM/NY_OVERLAP (15–24 JST); Tokyo rotation was negative for this
   style. Session weighting already exists in SITUATION_WEIGHTS — this is
   per-account empirical support for leaning it harder.
3. Capture tempo: winners were banked in minutes-to-hours, never days. The
   thesis-horizon machinery now encodes this structurally.

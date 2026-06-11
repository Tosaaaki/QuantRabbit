# Operator Manual Trading Evidence — 2025-05-30 → 2025-07-14 (USD_JPY)

First-party precedent for the AGENT_CONTRACT §4 product target. The operator
turned **200,000 JPY into a 1,234,730 JPY peak (×6.17 in ~6 weeks)**, ending at
1,103,381 JPY, trading USD_JPY manually on this same OANDA account
(transactions 3–2311). The 5%/10% daily target is a reproduction of observed
operator performance, not aspiration. Mined read-only by
`tools/mine_manual_history.py`; regenerate with:

```
PYTHONPATH=src python3 tools/mine_manual_history.py --out data/manual_history_2025_mining.json
```

## Aggregate (384 closed trades)

| Metric | Value |
|---|---|
| Net | +256,983 JPY (window 5/15–7/15; balance curve shows +903k peak-to-start) |
| Win rate | 48.7% |
| Avg win / avg loss | +4,518 / −3,195 (**payoff 1.41**) |
| Median hold | **0.32 h (~19 min)** |
| Expectancy | **+669 JPY/trade** |
| Daily win rate | 44% (profit came from size of wins, not frequency of green days) |

## What made the money

- **One pair, total focus.** Every one of the 384 trades was USD_JPY. No
  attention split across 8 pairs at micro size — one instrument, full size,
  fast rotation.
- **Trend side only.** LONG: +336,428 (wr 55.6%, payoff 2.04). SHORT:
  −79,444 (wr 39.9%). The edge was riding the prevailing direction, and the
  counter-trend faders bled.
- **Session concentration.** LONDON_AM (15–21 JST): +180,193, wr 58.6%,
  **payoff 2.46**, holds ~1.3h — the prime window. NY_OVERLAP: +84,210 with
  **7-minute median holds** (pure rotation). TOKYO: net −21,888 — the style
  did not work in the thin morning.
- **Fast asymmetric exits.** Manual profit banking (+391,581 at wr 65%, 2.3h
  holds), TP orders +177,659 at 6-minute holds, and stop losses cut at a
  **2.4-minute median** for small controlled damage (−117,605 over 103 stops,
  avg −1,142 — the cost of doing business, paid instantly).

## What nearly killed it (the same hole the bot had)

- **MARGIN_CLOSEOUT: −217,328 over 24 trades, median hold 12.4h, wr 4%.**
  The operator's own data shows the identical 12-hour cliff the 2026-06-11
  exit-leak audit found in the bot's ledger (≥12h market closes: 22/22
  losses). Holding a decayed thesis past its scope was catastrophic for the
  human too — this is the failure mode the disaster stop (§3.5-K) and thesis
  horizon expiry (§10, 2026-06-12) exist to bound.
- Best day +276,012 (7/8) was followed by worst day −182,448 (7/11): the
  giveback tail is real; protection-first behavior after big wins (§5) is the
  control for it.

## Implications for the trader (evidence, not new gates)

1. The arithmetic route to 10%/day at this account scale is **fewer, larger,
   faster trades on the strongest lane**, not 30 micro-trades across 8 pairs:
   the operator averaged ~9 trades/day at meaningful size with payoff > 1.4.
2. Session-weighted aggression: the operator's edge concentrated in
   LONDON_AM/NY_OVERLAP (15–24 JST); Tokyo rotation was negative for this
   style. Session weighting already exists in SITUATION_WEIGHTS — this is
   per-account empirical support for leaning it harder.
3. Capture tempo: winners were banked in minutes-to-hours, never days. The
   thesis-horizon machinery now encodes this structurally.

# MAE-based stop counterfactual — manual USD_JPY, 2026-07-16 → 08-18

Fifty closed manual USD_JPY trades. For every one, the maximum adverse
excursion (MAE) was measured from M5 candles between open and close, so a
candidate stop can be applied honestly: **to winners as well as losers**. The
earlier losers-only counterfactual in this session was selection-biased and is
superseded by this document.

## The shape of the book

| side | n | median result | median MAE | deepest MAE among them |
|---|---|---|---|---|
| winners | 29 | +7 pips, hold 0.8 h | **−4 pips** | −30 (the +111,650), −57 (+4,515), −71 (+26,640) |
| losers | 21 | −90 pips adverse, hold 27.6 h | deep | −159 |

Winners are shallow and fast; losers are deep and slow. **No winner in this
sample recovered from deeper than −71 pips**; every trade that went beyond −80
died. That separation — not any signal — is the exploitable structure.

## Honest counterfactual: stop applied to all 50 trades

Stop fires when MAE ≤ −X; the trade then loses exactly `units × X × pip`.
Otherwise the actual result stands. Actual net: **−39,788 JPY**.

| stop | net | delta | winners killed |
|---|---|---|---|
| −20 | −24,874 | +14,914 | 5 |
| **−30** | **−55,679** | **−15,891** | 3 — including the +111,650, whose MAE was exactly −30 |
| −40 | +36,489 | +76,277 | 2 |
| **−57 (approved rule)** | **+12,862** | **+52,650** | 1 (a +4,515) |
| −60 | +5,658 | +45,446 | 1 |
| **−80** | **+17,156** | **+56,944** | **0** |
| −100 | −14,048 | +25,740 | 0 |

Three things this table says:

1. **A tight stop is harmful here too.** −30 makes the book *worse* by killing
   the biggest winner at its exact MAE. This is consistent with the 08-09
   finding that 5–9-pip stops cost money — tight stops harvest noise.
2. **A deep stop flips the sign of the whole book.** At −57 or −80 the net goes
   from −39,788 to positive. The improvement on the loss side is spread across
   four independent large losers (+23.9k, +20.4k, +9.7k, …), not one lucky row.
3. **Do not tune the threshold on this sample.** n=50, one trade carries the
   win side, and adjacent thresholds swing the net by ±90k. The −40 "optimum"
   is a curve-fit. The defensible choices are the two that need no fitting:
   the **already-approved rule** (2026-06-11, H4 ATR×2.5 — ≈57 pips at today's
   ATR of 23), or **−80**, the boundary deeper than every winner's recovery.

## The rule already exists and is not running

The disaster stop was approved on 2026-06-11 (commit 61328d5): broker SL at
H4 ATR×2.5×session on every new entry, sizing-independent, no trailing. Had it
been active over this window, the manual book ends **+12,862 instead of
−39,788** — and the four margin-closeout days lose their teeth.

It is not running anywhere:

- `launchctl` shows **no QR service loaded** (guardian plist exists on disk,
  service absent);
- the trader runtime that attaches the SL to entries has been dead since 07-22;
- manual orders placed by hand carry no SL, and nothing repairs that.

So the finding is not "invent a rule". It is: **the approved rule was never
attached to the manual flow, and the broker's margin call substituted for it at
−90 to −160 pips.**

## Margin coupling made it worse

A margin closeout is a **portfolio** event: it liquidates positions
collectively. Eleven of the seventeen closeout losses were 1,000–2,000-unit
trades (−90 to −162 pips of adverse at close) that were dragged down as
collateral when the 40–50k positions exhausted the margin at ~93% utilisation.
Whether those small trades would have recovered is unknowable — their paths are
censored at the closeout — but they were not killed by their own risk.

At the current state (93% margin, `marginCloseoutPercent` 0.93) the account is
one adverse hour away from repeating this coupling.

## What the corrected ledger says about the manual method

With the approved stop applied, the measured manual line over these five weeks
is ≈ **+343 JPY per trade, ≈ +12.9k on ~250k NAV ≈ +5%/month** — one window,
one instrument, n=50, dominated by one +111,650 trade. That is the first
positive measured line in the whole system's history, and it is an order of
magnitude away from monthly 2×. Scaling it is a question of labels (the capsule
recorder) and of surviving long enough to collect them — which is exactly what
the stop buys.

## Superseded claim

The claim earlier in this session that "a −30 stop would have saved +157,886"
compared stops against losers only. With winners included it reverses to
−15,891. Corrected here.

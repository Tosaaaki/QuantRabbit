# True scoreboard — account 001-009-13679149-002, lifetime

Every funding transfer on the account (42 events, swept from transaction id 1
to 473332, read-only) against current NAV. This is the number every research
artifact, shadow scorecard and monthly target must reconcile to.

## Lifetime

| | JPY |
|---|---|
| net deposited (2025-05-30 → 2026-07-01) | **670,102** |
| NAV (2026-08-19) | **246,269** |
| **lifetime P&L** | **−423,833 (−63.2%)** |

Attribution by era (2026 pieces measured; 2025 by remainder):

| era | engine | P&L |
|---|---|---|
| 2025-05 → 2025-12 | legacy bots (pre-vNext, archived) | ≈ **−355k** |
| 2026-03 → 04 | Claude discretionary trader | −21.6k |
| 2026-05 → 07 | vNext bot lanes | −33.9k |
| 2026-05 → 08 | manual | −13.2k (+11.4k then −24.6k) |

Three generations of automation and one discretionary era. All negative.

## Nothing in the record is provably positive — including manual

This session's earlier claim of a "first positive measured line (+5%/month with
the approved stop)" does not survive leave-one-out and is **retracted**:

| manual since 07-16 | with the +111,650 trade | without it |
|---|---|---|
| actual | −24,590 | −136,240 |
| with approved −57 stop | +12,862 | **−98,788** |

One trade decides the sign of every cut. The same knife that closed the CHF
cluster (one day, one currency, one observation) closes this.

Other splits fail the same way. Size (small +12.0k / large −36.6k) and hold
time (≤12h −83.3k / >12h +58.7k) each invert or collapse when one to four
trades are removed — the hold-time split is further confounded because margin
coupling, not time, killed the fast losers. **At n=50 with this variance, no
statistical property of the manual method is measurable.**

What does survive correction, because it is mechanical rather than
statistical:

1. The four large margin-closeout losses are reduced independently by any deep
   stop (−57: +52.7k total; −80: +56.9k, zero winners killed).
2. No winner in the sample recovered from deeper than −71 pips.
3. The account's forced liquidations happened at −90 to −160 pips because no
   stop existed and margin sat at ~93%.

## The goal is the risk factor

Monthly 2× needs efficiency ((monthly−1)/maxDD) ≥ 4.0. Best ever measured:
0.423; sustained: 0.02. Efficiency is leverage-invariant, so no position size
reaches it. What chasing the multiple *did* produce is measurable: 40–50k-unit
positions at 93% margin with no stop — the exact configuration of all three
closeout days. **Fifteen months and −423,833 JPY are the experimental record of
the goal itself.** An edge, if one exists here, scales linearly with capital
and not at all with leverage; the multiple was never on the table.

## Present exposure (2026-08-19, read-only)

USD_JPY short 35,000 @ 158.932: **−45 pips adverse**, no SL, margin available
16,145 → **~46 more adverse pips to forced liquidation (~159.84)**. The sample's
recovery boundary is −71 pips: no winner has ever come back from deeper. The
position is inside the zone where holding has no precedent of success and
28 pips from the zone where it has precedent only of liquidation.

Decision is the operator's. The numbers above are what the ledger says about
positions that have stood here before.

## What "稼げる仕組み" means given this record

Not a trading system. In order:

1. **Survival mechanics, broker-side.** Fund a sub-account with only the
   capital at risk; the main balance becomes unreachable by margin coupling.
   Code-free, cannot die when software dies — the approved 6/11 disaster stop
   never touched the manual flow precisely because it lived in software that
   stopped on 07-22.
2. **Measurement before construction.** 0 operator decision labels exist; the
   capsule recorder is built and unused. Until labels accumulate, every claim
   about the manual method — positive or negative — is unfalsifiable, as the
   retraction above demonstrates.
3. **Months of survival** to reach an n where anything is provable, then scale
   by capital. −423,833 happened because building always preceded measuring;
   the sequence must invert.

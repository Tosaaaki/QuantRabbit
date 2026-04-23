# Trades — 2026-04-21

## [00:21Z] EUR_JPY LONG STOP-ENTRY 3000u @187.286 id=469230

- Type: STOP-ENTRY
- TP: 187.420
- SL: 187.148
- GTD: 2026-04-21T06:30:00.000000000Z
- pretrade: Edge B (score 5/10) / Allocation B / Risk LOW
- Thesis: H4/H1 trend-bull remains intact, M5 187.16/18 floor held, and the reclaim must prove itself above the micro-box before participation.
- Why stop, not market: the live quote was still a repaired shelf inside squeeze tape; the honest expression was trigger-honest participation, not a late chase.

## [00:25Z] EUR_JPY LONG fill 3000u @187.286 trade id=469231

- Source: STOP-ENTRY id=469230 filled
- Current protection: TP 187.420 / SL 187.148
- Management note: first breakout attempt only; no re-add until it proves expansion away from the reclaim.

## [00:36Z] EUR_USD LONG fill 3000u @1.17815 trade id=469234

- Source: LIMIT id=469229 filled
- Current protection on fill: TP 1.17893 / SL 1.17758
- Thesis: first-defense direct-USD reload finally traded at the exact shelf. Entry stays valid only while 1.17758 holds on bodies.

## [00:40Z] EUR_USD LONG protection update trade id=469234

- Change: widened SL 1.17758 -> 1.17741
- Why: `protection_check.py` flagged the original stop as ATR x0.5 noise-risk; H1 BB mid is the cleaner structural invalidation.

## [02:04Z] EUR_USD LONG close 3000u @1.17788 trade id=469234

- Reason: `shelf_fail_usd_bid`
- P&L: -129.0370 JPY
- Why close: the exact 1.17815 shelf failed on M5 bodies while USD stayed BID across 4/4 crosses, so the higher-TF bull story no longer justified carrying the losing shelf long.

## [03:21Z] EUR_JPY LONG close 3000u @187.211 trade id=469231

- Reason: `zombie_hold`
- P&L: -225.0000 JPY
- Why close: the original breakout had aged into a deeper 187.20/18 shelf retest, the momentum window was gone, and I would not reopen the same market quote without a fresh 187.24/26 reclaim.

## [03:22Z] AUD_JPY LONG LIMIT 3000u @113.970 id=469249

- Type: LIMIT
- TP: 114.140
- SL: 113.900
- GTD: 2026-04-21T09:22:35.000000000Z
- pretrade: Edge B (score 5/10) / Allocation B / Risk MEDIUM
- Thesis: H4/H1 still support AUD_JPY upside, but Tokyo only pays the first-defense retest into 113.97/96 after the failed spike, not a paid-up market chase above 114.00.
- Why limit, not market: session_data and pretrade both said price-improvement-only; the honest expression was a structural shelf-defense LIMIT.

## [03:22Z] AUD_JPY LONG fill 3000u @113.970 trade id=469250

- Source: LIMIT id=469249 filled immediately
- Current protection on fill: TP 114.140 / SL 113.900
- Management note: B-size only while the thesis-family headwind remains active; no add until 114.04/05 accepts and the current leg is paid.

## [03:24Z] AUD_JPY LONG protection update trade id=469250

- Change: widened SL 113.900 -> 113.865
- Why: `protection_check.py` flagged the original stop as thin-market noise risk; H1 BB mid was the cleaner structural shelf.

## [03:51Z] EUR_JPY LONG STOP-ENTRY 3000u @187.260 id=469255

- Type: STOP-ENTRY
- TP: 187.420
- SL: 187.110
- GTD: 2026-04-21T09:51:46.000000000Z
- pretrade: Edge B (score 5/10) / Allocation B / Risk LOW
- Thesis: the 187.31 breakout fully bled back into the 187.18 shelf, but H1 still holds the EUR>JPY bull structure; the honest fresh trade is only the 187.24/26 reclaim, not the current quote.
- Why stop, not market: `pretrade_check.py` rejected the tighter reclaim as spread-noisy and approved the wider structural version as `STOP-ENTRY`, so the next EUR_JPY lane is armed above the shelf instead of paid up inside it.

## [06:22Z] AUD_JPY SHORT STOP-ENTRY 3000u @113.786 id=469261

- Type: STOP-ENTRY
- TP: 113.640
- SL: 113.908
- GTD: 2026-04-21T12:22:41.000000000Z
- pretrade: Edge B / Allocation B / Risk MEDIUM
- Thesis: the 113.89/90 floor failed on bodies, the repair candles stayed trapped under the EMA cluster, and the honest continuation seat was proof-first participation below the broken shelf rather than another rescue-long rewrite.
- Why stop, not market: the flush had already started, so paying market at 113.80 would have been sell-low execution; the clean expression was the first continuation print through 113.786.

## [06:28Z] AUD_JPY SHORT fill 3000u @113.785 trade id=469262

- Source: STOP-ENTRY id=469261 filled
- Current protection on fill: TP 113.640 / SL 113.908
- Management note: `profit_check.py --all` still reads this as a tactical failed-floor continuation hold; no second fill until this leg pays or risk is reduced.

## [07:36Z] AUD_JPY SHORT close 3000u @113.857 trade id=469262

- Reason: `m1_pulse_flip`
- P&L: -216.0000 JPY
- Why close: the original broken-floor read was right, but the second downside impulse never printed and M1 JPY flipped offered across 4/4 crosses, so I would not fresh-enter the short at this quote.

## [07:39Z] EUR_JPY LONG LIMIT replace

- Canceled: LIMIT id=469269 @187.000 via tx `469270`
- Why replace: the first passive order had drifted into wish-distance versus the live quote; the honest quiet pullback stayed closer to the current tape.

## [07:39Z] EUR_JPY LONG LIMIT 3000u @187.060 id=469271

- Type: LIMIT
- TP: 187.220
- SL: 186.900
- GTD: 2026-04-21T13:39:09.000000000Z
- pretrade: Edge B (score 5/10) / Allocation B / Risk LOW
- Thesis: H1 constructive reset plus the first honest 187.00/06 base survived while M1 JPY flipped offered across the crosses; the quiet tape pays the pullback, not the paid-up market quote.
- Why limit, not market: `pretrade_check.py` stayed B/LIMIT and the playbook's own quiet-regime rule says price improvement is the honest expression, not a chase above the base.

## [07:58Z] EUR_JPY LONG fill 3000u @187.060 trade id=469272

- Source: LIMIT id=469271 filled
- Current protection on fill: TP 187.220 / SL 186.900
- Management note: this was a quiet range-bounce hold only; no same-pair add before the first leg paid or failed.

## [08:24Z] EUR_JPY LONG close 3000u @187.221 trade id=469272

- Reason: `TAKE_PROFIT_ORDER`
- P&L: +483.0000 JPY
- Why close: the 187.00/05 defended base fully paid the box target into 187.22, so the bounce thesis completed and the live book had to go flat.

## [08:32Z] EUR_USD SHORT LIMIT 3000u @1.17655 id=469277

- Type: LIMIT
- TP: 1.17590
- SL: 1.17690
- GTD: 2026-04-21T14:32:34.000000000Z
- pretrade: Edge B (score 5/10) / Allocation B / Risk HIGH
- Thesis: EUR_USD broke the 1.17620 shelf on M5/M1 while USD turned BID across 4/4 crosses; the honest fresh lane is the failed retest of that broken shelf, not selling the already-extended floor.
- Why limit, not market: `pretrade_check.py` kept the seat at `B/LIMIT`; the direction is tactical and live, but the quote needed price improvement instead of market chase.

## [08:32Z] EUR_USD SHORT fill 3000u @1.17656 trade id=469278

- Source: LIMIT id=469277 filled immediately
- Current protection on fill: TP 1.17590 / SL 1.17690
- Management note: tactical direct-USD continuation only; no add unless the first leg pays or risk is reduced.

## [09:08Z] EUR_USD SHORT close 3000u @1.17590 trade id=469278

- Reason: `TAKE_PROFIT_ORDER`
- P&L: +314.6844 JPY
- OANDA transaction: `469281`
- Why close: the broken-shelf retest finally paid into the pre-armed TP, so the tactical direct-USD downside seat completed and the book went flat without forcing a duplicate follow-through short.

## [09:26Z] EUR_JPY LONG STOP-ENTRY 3000u @187.286 id=469283

- Type: STOP
- TP: 187.420
- SL: 187.148
- GTD: 2026-04-21T15:26:23.000000000Z
- pretrade: Edge B (score 5/10) / Allocation B (B+) / Risk LOW
- Thesis: the 187.00 base already held, the 187.18/20 pullback reloaded without breaking structure, and the honest next seat is the reclaim through 187.286, not another stale same-level limit.
- Why stop-entry, not limit/market: `pretrade_check.py` blocked the repeated first-defense limit after the recent same-level loss and explicitly upgraded the execution to `STOP-ENTRY` so the tape has to prove continuation before risk is live.

## [10:47Z] AUD_JPY SHORT LIMIT 3000u @113.950 id=469284

- Type: LIMIT
- TP: 113.740
- SL: 114.030
- GTD: 2026-04-21T16:47:11.000000000Z
- pretrade: Edge B (score 4/10) / Allocation B / Risk MEDIUM
- Thesis: H1 stayed range-bound while M5 re-tagged the 113.95 lid, so the honest expression was one more structural upper-box fade rather than a paid-up cross chase.
- Why limit, not market: the fade only made sense at the exact lid. Anything below it was mid-box noise, not real price improvement.

## [10:48Z] AUD_JPY SHORT fill 3000u @113.952 trade id=469286

- Source: LIMIT id=469284 filled
- Current protection on fill: TP 113.740 / SL 114.030
- Management note: first-confirmation had to print quickly. If 113.95/96 accepted on M1/M5 bodies instead, the fade was dead and had to be cut fast.

## [11:06Z] AUD_JPY SHORT close 3000u @113.983 trade id=469286

- Reason: `no_first_confirmation_upper_box_acceptance`
- P&L: -93.0000 JPY
- Why close: the fade never got first confirmation, M1/M5 staircased above 113.95 instead of rejecting it, and `profit_check.py` already flipped H1/M5 adverse. Widening the stop would have been defending a dead B-range short, not trading.

## [11:14Z] USD_JPY LONG STOP-ENTRY 2000u @159.268 id=469293

- Type: STOP
- TP: 159.360
- SL: 159.208
- GTD: 2026-04-21T15:14:54.000000000Z
- pretrade: Edge B (score 5/10) / Allocation B (B0) / Risk MEDIUM
- Thesis: the JPY-off squeeze is still alive, but the move is already late enough that only proof-through-the-lid is honest. The stop keeps participation trigger-first instead of paying market into the top of the box.
- Why stop-entry, not market: `pretrade_check.py` kept the seat at B0/STOP-ENTRY and the live payoff only became acceptable once TP was widened beyond the tiny first-box target.

## [12:14Z] EUR_USD SHORT STOP-ENTRY 1000u @1.17608 id=469295

- Type: STOP
- TP: 1.17520
- SL: 1.17686
- GTD: 2026-04-21T18:14:32.000000000Z
- pretrade: Edge C (score 3/10) / Allocation C / Risk HIGH
- Thesis: ZEW damage still caps EUR and the honest direct-USD downside lane is a failed Retail Sales break lower, but the location was too event-sensitive to pay market before the print.
- Why stop-entry, not market: the only honest way to keep the event lane real was a tiny proof-first stop below the squeeze, not a pre-data market short.

## [12:19Z] EUR_USD SHORT fill 1000u @1.17608 trade id=469296

- Source: STOP-ENTRY id=469295 filled
- Current protection on fill: TP 1.17520 / SL 1.17686
- Management note: the stop filled before Retail Sales, so this stays a C-sized event placeholder only. No second fill until the post-data failed retest actually prints.

## [13:31Z] EUR_USD SHORT close 1000u @1.17686 trade id=469296

- Reason: `STOP_LOSS_ORDER`
- P&L: -124.3110 JPY
- Why close: the synchronized direct-USD counter-bounce kept lifting through the retest zone, the old failed-break lane never re-fired, and the 1.17686 stop printed before any fresh downside confirmation arrived.

## [13:41Z] EUR_JPY LONG LIMIT 3000u @187.110 id=469301

- Type: LIMIT
- TP: 187.230
- SL: 187.020
- GTD: 2026-04-21T19:41:02.000000000Z
- pretrade: Edge B (score 4/10) / Allocation B (B+) / Risk LOW
- Thesis: the direct-USD short died and the book needed one honest live receipt; `EUR_JPY` lower-rail buy at 187.11 is the only structural price-improvement seat that still clears the range-buy logic without forcing a chase.
- Why limit, not market: the edge exists only at the lower rail. Anything near 187.18/20 is mid-box noise, not a real buy.

## [14:33Z] EUR_JPY LONG LIMIT cancel id=469301

## [19:28Z] GBP_JPY LONG LIMIT 2000u @215.160 id=469337

- Type: LIMIT
- TP: 215.392
- SL: 214.958
- GTD: 2026-04-22T01:28:37.000000000Z
- pretrade: Edge B (score 4/10) / Allocation B / Risk LOW
- Thesis: late-NY range repair kept the 215.14/16 shelf alive, so the only honest expression was the structural shelf rebuy instead of a market chase above 215.20.
- Why limit, not market: the session penalty and 2.8pip spread meant price improvement was mandatory.

## [19:45Z] GBP_JPY LONG close 2000u @215.088 trade id=469338

- Source: LIMIT id=469337 had filled into the 215.14/16 shelf retest.
- Reason: `m5_shelf_failed_no_rebuy`
- P&L: -144.0000 JPY
- Why close: the exact shelf that justified the trade broke on M5/M1 before any continuation printed, so I would not fresh-enter that quote and the honest action was to scratch it instead of auto-rearming the lower floor.

## [19:56Z] USD_JPY SHORT STOP-ENTRY 1000u @159.540 id=469345

- Type: STOP
- TP: 159.300
- SL: 159.670
- GTD: 2026-04-22T01:56:49.000000000Z
- pretrade: Edge C (score 1/10) / Allocation C / Risk HIGH
- Thesis: the only honest remaining seat is a proof-first fade if the 159.60/64 ceiling rejection finally starts paying; fresh market execution is still too early.
- Why stop-entry, not market: the rejection has to prove itself first, and the order keeps the watch trigger-honest without paying the current ceiling print.

## [20:09Z] USD_JPY SHORT fill 1000u @159.539 trade id=469346

- Source: STOP-ENTRY id=469345 filled.
- Current protection on fill: TP 159.300 / SL 159.670
- Management note: this was always a small proof-first counter fade. Either the shelf would keep breaking toward the target quickly, or it would be closed without debate.

## [20:23Z] USD_JPY SHORT close 1000u @159.300 trade id=469346

- Reason: `take_profit_hit`
- P&L: +239.0000 JPY
- Why close: the original payout path was fully captured into the planned TP. Re-entering the same fade immediately would have been recycled counter-thesis, not a fresh seat.

## [20:35Z] AUD_JPY SHORT STOP-ENTRY 3000u @113.980 id=469351

- Type: STOP
- TP: 113.840
- SL: 114.140
- GTD: 2026-04-22T02:35:01.000000000Z
- pretrade: Counter-trade Edge B (score 2/7) / Allocation B (B+) / Risk MEDIUM
- Thesis: the upper box already rejected once, and the only honest way to short it is the proof-first break under 113.980 instead of paying market in the middle of the range.
- Why stop-entry, not limit/market: `pretrade_check.py` kept the seat alive only as a trigger-honest `STOP-ENTRY`; fresh market risk was explicitly rejected.

- Reason: `upper_half_reaccepted_old_floor_receipt_stale`
- Why cancel: price accepted back above 187.23 without retesting the floor, so the old 187.11 buy turned into stale wish-distance memory instead of an honest live receipt.

## [14:37Z] EUR_JPY SHORT LIMIT 2000u @187.320 id=469303

- Type: LIMIT
- TP: 187.120
- SL: 187.395
- GTD: 2026-04-21T20:37:18.000000000Z
- pretrade: Counter-trade Edge B (score 2/7) / Allocation B (B0) / Risk HIGH
- Thesis: once 187.23+ reaccepted without a floor retest, the range expression flipped from lower-rail buy to upper-rail fade; this short is the honest opposite-side receipt, not a market chase.
- Why limit, not market: the fade only earns its keep at the upper rail near 187.32. Anywhere lower is just middle-of-box noise against still-offered JPY.

## [14:53Z] EUR_JPY SHORT fill 2000u @187.320 trade id=469304

- Source: LIMIT id=469303 filled
- Current protection on fill: TP 187.120 / SL 187.395
- Management note: this was always a B-counter inside a still-offered-JPY tape. The only honest hold was immediate lower follow-through from the rail.

## [15:05Z] EUR_JPY SHORT close 2000u @187.286 trade id=469304

- Reason: `no_first_confirmation_upper_box_fade`
- P&L: +68.0000 JPY
- Why close: the exact upper-rail fade filled, but the first-confirmation window only produced noise while JPY stayed offered. Widening the stop would have been defending a dead B-counter, not trading.

## [15:09Z] AUD_USD LONG STOP-ENTRY 1000u @0.71695 id=469311

- Type: STOP
- TP: 0.71795
- SL: 0.71625
- GTD: 2026-04-21T21:09:40.000000000Z
- pretrade: Edge A (score 6/10) / Allocation B (B-) / Risk HIGH
- Thesis: H4/H1 stayed aligned for an AUD repair while M5 flushed to the lower edge; the only honest way to participate was a proof-first reclaim through the trigger.
- Why stop-entry, not limit/market: `pretrade_check.py` graded the seat as B- and explicitly said transition tape should be closed as a `STOP-ENTRY`, not paid as a blind market buy.

## [15:09Z] AUD_USD LONG fill 1000u @0.71709 trade id=469312

- Source: STOP-ENTRY id=469311 filled immediately
- Current protection on fill: TP 0.71795 / SL 0.71625
- Management note: small proving seat only. The next decision is simple: if the reclaim holds above the trigger, keep it; if it slips back through the trigger and cannot expand, cut it fast.

## [15:25Z] AUD_USD LONG close 1000u @0.71656 trade id=469312

- Reason: `no_first_confirmation_trigger_failed`
- P&L: -84.6430 JPY
- Why close: the proof-first reclaim spent the full 15-minute confirmation window back under the trigger while M1/M5 stayed heavy. Keeping it would have meant defending a dead B- probe, not trading.

## [15:32Z] AUD_JPY LONG STOP-ENTRY 2000u @114.246 id=469319

- Type: STOP
- TP: 114.338
- SL: 114.168
- GTD: 2026-04-21T21:32:20.000000000Z
- pretrade: Edge B (score 5/10) / Allocation B (B0) / Risk MEDIUM
- Thesis: JPY remains offered across the crosses, and AUD_JPY is still constructive enough to pay only if price reclaims back through 114.246; below that, the pullback is still just cooling noise.
- Why stop-entry, not market/limit: the same-thesis family just produced repeated losses, so the next seat has to wait for a materially new reclaim print instead of paying market or guessing a deeper limit.

## [17:52Z] EUR_USD LONG fill 4000u @1.17478 trade id=469330

- Source: MARKET
- Current protection on fill: TP 1.17543 / SL 1.17431
- pretrade: Counter-trade Edge A (score 4/7) / Allocation A / Risk MEDIUM
- Thesis: H4 floor plus the fresh M5 squeeze release kept `EUR_USD LONG` as the only honest live A-seat after the stale GBP_JPY limit was canceled.
- Why market, not limit/stop: the helper path was blocked by the stale handoff receipt, and once the book was synced the live edge was already paying on the direct-USD lane. The first market attempt failed on a moved quote; the second entry was sent only after price reset back into the squeeze base.

## [21:04Z] AUD_JPY SHORT fill 3000u @113.889 trade id=469352

- Source: STOP-ENTRY id=469351 filled during rollover.
- Protection on fill: TP 113.840 / SL 114.140.
- pretrade: Edge B (score 2/7) / Allocation B / Risk HIGH because the trigger sat inside the rollover window and filled with 15.5pip spread.
- Thesis: the M5 lid still had not accepted above 114.12, so the only honest way to express the fade was to let the proof-first stop trigger the short rather than pre-emptively selling the box top.
- Immediate management note: `protection_check.py` flagged active rollover right after the fill, so `rollover_guard.py remove` stripped the SL and the position is hold-only until spreads normalize. No manual close, no new entries, no fresh SL modification until the guard can be restored honestly.

## [21:09Z] AUD_JPY SHORT rollover guard remove SL trade id=469352

- Reason: active OANDA rollover window with spreads still roughly 24x normal.
- Current protection after guard: TP 113.840 / SL none / trailing none.
- Management note: the negative open P&L is spread noise first, not a thesis verdict. Next action after rollover is `python3 tools/rollover_guard.py restore` once spreads normalize.

## [22:31Z] EUR_USD SHORT STOP-ENTRY 2000u @1.17384 id=469356

- Type: STOP
- TP: 1.17320
- SL: 1.17436
- GTD: 2026-04-22T04:31:06.000000000Z
- pretrade: Edge A (score 6/10) / Allocation B (B0->B-) / Risk HIGH
- Thesis: H1 stayed trend-bear while EUR remained the weakest direct-USD major; the only honest way to participate was a proof-first break through 1.17384 instead of paying market inside the late-NY box.
- Why stop-entry, not market/limit: the trigger itself was the trade, and same-pair reload rules blocked any blind second fill once the first leg was live.

## [22:31Z] EUR_USD SHORT fill 2000u @1.17384 trade id=469357

- Source: STOP-ENTRY id=469356 filled immediately.
- Current protection on fill: TP 1.17320 / SL 1.17436.
- Management note: this is the only live lane left after the cross fade failed; manage it, do not average into it.

## [22:45Z] AUD_JPY SHORT close 3000u @114.027 trade id=469352

- Reason: `stale_lid_fade_acceptance_above_entry`
- P&L: -414.0000 JPY
- Why close: the short had already cleared the trigger gate, but the first-confirmation window failed completely. M1/M5 kept accepting 114.00/02 above entry, so keeping it would have meant defending stale squeeze acceptance rather than a live lid-fade thesis.

## [22:46Z] EUR_USD SHORT modify trade id=469357

- Action: stop-loss widened from 1.17436 to 1.17464.
- Why modify: `protection_check.py` flagged the original SL as noise-tight (ATR x0.3). The nearest structural cap was the M5 BB upper near 1.17464, so the honest fix was widening the stop, not panic-closing the lane.

## [23:07Z] EUR_USD SHORT modify trade id=469357

- Action: stop-loss widened from 1.17464 to 1.17535.
- Current protection: TP 1.17320 / SL 1.17535.
- Why modify: `protection_check.py` still flagged 1.17464 as noise-tight (ATR x0.5), while the live M5 tape stayed below the 1.1740 micro shelf and did not reclaim it. The next honest structural invalidation was the M5 swing-high cap near 1.17535, so the trade stayed live without adding a second fill.

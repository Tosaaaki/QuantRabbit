
## 22:05Z — GBP_USD CLOSE (tape stall) -162 JPY ✗
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Direction | CLOSE SHORT |
| Units | 3000u |
| Entry | 1.35321 |
| Close | 1.35355 |
| Loss | -3.4pip |
| P/L | -162.54 JPY |
| id | 468386 |
| Reason | Late-NY lower-half fill lost its fresh-short basis. M5 stalled, GBP M1 flipped bid, and the position failed the "would I enter this now?" test. |

## 22:08Z — GBP_USD STOP LONG refreshed
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Direction | LONG |
| Units | 3000u |
| Entry | STOP @1.35535 |
| TP | 1.35620 (+8.5pip) |
| SL | 1.35470 (-6.5pip) |
| GTD | 2026-04-17 04:30Z |
| Order ID | 468394 |
| Reason | Keep only the failure-break path after the stale short was cut; late-NY fresh-short reload graded C, so the trader map stays conditional instead of forcing a lower-half re-entry. |

## 02:42Z — GBP_USD CLOSE (TP auto-fire) +513 JPY ✓
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Direction | CLOSE LONG |
| Units | 2000u |
| Entry | 1.35619 (LIMIT fill) |
| Close | 1.35781 (TP hit) |
| Gain | +16.2pip |
| P/L | +513.23 JPY |
| id | 468117 |

## 02:54Z — AUD_JPY ENTRY LONG (EMA20 dip-buy)
| Field | Value |
|-------|-------|
| Pair | AUD_JPY |
| Direction | LONG |
| Units | 2000u |
| Entry | MARKET @114.036 |
| TP | 114.250 (+21.4pip) |
| SL | 113.870 (-16.6pip) |
| Trade ID | 468130 |
| Spread | 1.6pip |
| Thesis | Band walk (113.75→114.13) pulled back 10pip to EMA20 @114.02. H4 MID BULL (ADX=45 StRSI=0.76, room). H1 FLOOR (StRSI=0.08 = fresh H1 impulse starting). AUD strongest currency H4 (+37). JPY M1 offered (-13). M5 div pullback absorbed at EMA20 = continuation signal. |
| pretrade | A (MEDIUM, score=7). Warned H4 CCI=88/RSI=75 overbought. |
| Conviction | A → 2000u (conservative for H4 overbought warning + employment MISS) |
| Max loss | 2000u × 16.6pip × 0.01 JPY/pip/unit = -332 JPY |
| Zombie | 05:00Z |

## 02:47Z — GBP_USD LIMIT LONG (rotation)
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Direction | LONG |
| Units | 3000u |
| Entry | LIMIT @1.35700 |
| TP | 1.35920 (+22pip) |
| SL | 1.35560 (-14pip) |
| GTD | 06:30Z |
| Order ID | 468128 |
| Spread | 1.3pip |
| Thesis | Post-TP rotation. Theme CONFIRMED (first TP hit today). BB mid pullback @EMA20 area. H4 StRSI=0.16 (floor, 84% room). UK GDP 06:00Z BEAT scenario = 1.360+. M5 StRSI=1.0 = pullback from 1.35838 to 1.35700 likely (13.8pip). |
| pretrade | A (MEDIUM, 62%WR) |
| Conviction | A → 3000u (theme confirmed: A→S sizing, conservative for pre-GDP) |
| R:R | 1.6:1 (expandable on GDP BEAT) |

## 01:44Z — USD_JPY LIMIT SHORT
| Field | Value |
|-------|-------|
| Pair | USD_JPY |
| Direction | SHORT |
| Units | 2000u |
| Entry | LIMIT @158.850 |
| TP | 158.470 (+38pip) |
| SL | 158.970 (-12pip) |
| GTD | 2026-04-16T06:00Z |
| Order ID | 468120 |
| Spread | 0.8pip |
| Thesis | H1 BEAR cascade 159.00→158.70. M5 StRSI=1.00 bounce at EMA20 resistance. H1 Fib corrective wave at 90% (near top). H4 mid-BEAR (ADX=30). Sell rally. |
| pretrade | A score=6 (B-size cap: WR=33%, noisy scanner recipe) |
| Conviction | B → 2000u |
| R:R | 3.2:1 |

## EUR_JPY LIMIT LONG 2000u id=468121 — 02:10Z
| Field | Value |
|-------|-------|
| Entry type | Counter-S |
| LIMIT | 187.350 |
| TP | 187.470 (+12pip) |
| SL | 187.280 (-7pip) |
| GTD | 06:10Z |
| Spread | 1.9pip |
| Pretrade | LOW (B, 66% WR) |
| Thesis | H4+H1 StRSI=0.00 structural floor. AUD MISS spike (187.30 low) absorbed. M5 bounce confirmed (StRSI=1.00). EUR H4=BID(+25). Entry on M5 pullback toward BB lower @187.35 |
| Against | JPY M15=BID(+20) headwind. EUR M15=offered(-15). Both short-term headwinds |
| If wrong | Close below 187.28 (spike low from AUD MISS = thesis broken if violated) |
| Zombie | 02:40Z |

## AUD_JPY LIMIT LONG 2000u id=468122 — 02:15Z
| Field | Value |
|-------|-------|
| Entry type | Momentum dip-buy in H4 trend |
| LIMIT | 113.900 |
| TP | 114.080 (+18pip) |
| SL | 113.740 (-16pip) |
| GTD | 06:15Z |
| Spread | 1.6pip |
| Pretrade | MEDIUM (A conv, 60% WR) |
| Thesis | H4 BULL (ADX=45, StRSI=0.76 mid, room) + H1 at floor (StRSI=0.08) = classic dip-buy in trend. AUD Employment MISS spike hit 113.75 and fully recovered = buyers in control post-data. AUD H4 strongest currency (+37). LIMIT at M5 pullback zone (below recovery bounce). |
| Against | H4 RSI=75 (overbought). Employment MISS (fresh AUD structural bear). JPY M15=BID(+20) headwind |
| If wrong | Price breaks below 113.74 (spike low violated = MISS more impactful than recovery) |
| Zombie | 04:15Z |

### EUR_JPY LONG 2000u id=468141 — 2026-04-16 03:10Z
| Field | Value |
|-------|-------|
| Entry | 187.190 (market, Sp=7pip spike-wide) |
| TP | 187.400 (former range mid, structural) |
| SL | 187.000 (below H4 floor) |
| Type | Counter-S |
| Conviction | B |
| Pretrade | LOW (B 4/8) |
| Thesis | JPY spike 02:35Z took EUR_JPY from 187.47 to 187.12 (35pip). H4+H1 StRSI=0.00 floor. M5 bullish div. Counter-S proven 4/5. JPY M1=BID(+62) = spike not structural reversal. |
| R:R | 21:19 = 1.1:1 (wide spread degraded entry) |
| Max loss | -380 JPY |

### EUR_USD LIMIT LONG 2000u id=468144 — pending
| Field | Value |
|-------|-------|
| LIMIT | @1.18060 (former resistance = new support after band walk breakout) |
| TP | 1.18200 (14pip net) |
| SL | 1.17980 (8pip below) |
| GTD | 09:10Z |
| Type | Momentum dip-buy |
| Conviction | B |
| Thesis | EUR_USD band walk breakout to 1.1822. Former resistance 1.18030 = new support. H4 ADX=52 BULL. EUR strongest H4 +25. Dip-buy on pullback to structural level. |

## 03:25Z — LIMIT changes

| Action | Pair | Dir | Units | Price | TP | SL | GTD | id | Reason |
|--------|------|-----|-------|-------|----|----|-----|-----|--------|
| CANCEL | GBP_USD | LONG | 3000 | 1.35700 | — | — | — | 468128 | GDP binary + 23pip away → unreliable |
| LIMIT | AUD_JPY | LONG | 2500 | 113.930 | 114.100 | 113.800 | 06:00Z | 468146 | H4+H1 BULL, AUD strongest, M5 Fib61.8% dip |

pretrade: AUD_JPY B (H4 overbought flag noted, 2500u conservative)

### 04:26Z — EUR_JPY LIMIT CANCELLED (id=468150 @187.190)
Reason: Slot swap. JPY spike at 03:05Z already absorbed. M1 JPY offered(-14) = JPY spike unlikely to repeat quickly before GDP. EUR_JPY 27pip away = low fill probability. AUD_JPY better vehicle for dominant theme.

### 04:26Z — AUD_JPY LONG LIMIT 2000u @113.930 id=468152
| Field | Value |
|-------|-------|
| Entry | 113.930 (LIMIT) |
| TP | 114.220 |
| SL | 113.730 |
| GTD | 08:30Z |
| Spread | 1.6pip |
| Thesis | TREND-BULL dip-buy. AUD strongest (H4+M15+M1 all bid). JPY offered. EMA20 dip zone at 113.93. |
| Pretrade | B (59% WR, +3,729 JPY historical) |
| Conviction | B | Size: 2000u (5% NAV) |
| Margin | +7.2% → total ~19% worst case both LIMITs |

### EUR_USD LONG 2000u — 04:50Z entry
| Field | Value |
|-------|-------|
| Entry time | 2026-04-16 04:50 UTC |
| Pair | EUR_USD |
| Side | LONG |
| Units | 2000u |
| Entry price | 1.18090 |
| TP | 1.18180 (+9.0pip) |
| SL | 1.17962 (-12.8pip) |
| TradeID | 468155 |
| Spread | 0.8pip |
| Conviction | B |
| pretrade | MEDIUM (57% WR, recent all WIN) |
| Thesis | M5 RANGE scalp: BB lower (RANGE=11.8pip), StRSI=0.0, lower wicks absorbing. H1 BULL (ADX=26 DI+=27). H4 MID BULL (StRSI=0.41, room). EUR H4 BID(+25). Post-spike (03:05Z) flag consolidation bounce. |
| GDP risk | UK 06:00Z (+75min). EUR/GBP correlated. Pre-GDP TP target: BB mid ~1.18150 if stalls. |

## 05:33Z — EUR_USD CLOSE (pre-GDP zombie) -140 JPY ✗
| Field | Value |
|-------|-------|
| Pair | EUR_USD |
| Direction | CLOSE LONG |
| Units | 2000u |
| Entry | 1.18090 |
| Close | 1.18046 |
| Loss | -4.4pip |
| P/L | -140.01 JPY |
| id | 468155 |
| Reason | Pre-GDP zombie close. M5 EMAs declining (EMA12<EMA20 both falling). Price making lower lows post-03:05Z spike. Thesis (M5 RANGE bounce) dead. GDP risk in 27 min. Zombie deadline was 05:50Z — closed early on M5 confirmation. |
| Lesson | Pre-event SQUEEZE with declining M5 EMAs = skip or LIMIT only. Market entry into pre-GDP squeeze = negative EV even with H4 MID BULL. |

## 05:35Z — AUD_USD SHORT LIMIT CANCELLED (id=468158)
Reason: 11.4pip from fill price, GTD 05:55Z (22min). Price at 0.71836 vs LIMIT @0.71950. Cancelled early to free LIMIT slot for EUR_JPY pre-GDP coverage.

## 05:35Z — EUR_JPY LONG LIMIT 2000u @187.180 id=468164
| Field | Value |
|-------|-------|
| Entry | 187.180 (LIMIT) |
| TP | 187.530 (+35pip) |
| SL | 186.950 (-23pip) |
| GTD | 09:00Z |
| Spread | 2.3pip |
| Type | Counter-S (H4 floor) |
| Conviction | B |
| Pretrade | LOW (B, 4/8 counter-grade). Historical 65% WR, +107 JPY avg. |
| R:R | 35:23 = 1.5:1 |
| Thesis | H4 Counter-S floor (StRSI=0.01, proven recipe 4/5 = same as this morning's EUR_JPY +420). H1 RANGE bottom 187.16-187.20 (structural spike low). Pre-GDP: GDP miss → JPY spike → fills at range bottom (excellent structural entry). GDP beat → no fill → no harm. |
| Max loss | 2000 × 23 × 0.01 = -460 JPY |
| Margin | +11.9% NAV → worst case both LIMITs: 19.1% (well within 85%) |

## 06:33Z — EUR_USD ENTRY LONG (LIMIT fill — BB lower + H1/M15 StRSI=0.00)
| Field | Value |
|-------|-------|
| Pair | EUR_USD |
| Direction | LONG |
| Units | 2000u |
| Entry | 1.17970 (LIMIT fill id=468167) |
| TP | 1.18180 |
| SL | 1.17850 |
| Pretrade | A (H4 BID+21, H1 StRSI=0.00, M15 StRSI=0.00, system best edge) |
| Context | USD M1 bounce within H4 bear. M5 fib -250% over-extension = reversal imminent |

## Session 07:05-07:15Z
### Actions
| Time | Action | Pair | Side | Units | Price | TP | SL | id | P&L | Notes |
|------|--------|------|------|-------|-------|----|-----|-----|-----|-------|
| 07:09Z | CANCEL | EUR_JPY | LONG | 2000 | 187.180 | - | - | 468164 | - | 25pip away 2h GTD, freed slot |
| 07:09Z | LIMIT | GBP_USD | LONG | 2000 | 1.35640 | 1.35780 | 1.35530 | 468172 | - | RANGE BB lower. GDP beat. London open. B-conviction |

### Position: EUR_USD LONG 2000u @1.17970 id=468167
- UPL at session start: +98 JPY (+5.6pip)
- profit_check: TAKE_PROFIT (M5 StRSI=0.90, H1 DI- slight dominance)
- Decision: C HOLD — H1 TREND-BULL at EMA12 dip-buy zone. London open 51min. H4 MID BULL.
- TP=1.18180 (H1 BB upper). SL=1.17850. Zombie: 09:00Z.

## 13:47Z — GBP_JPY CLOSE (dirty cross expression) -384 JPY ✗
| Field | Value |

## 16:45Z — GBP_USD LIMIT SHORT (reload restored after worker path disappeared)
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Direction | SHORT |
| Units | 3000u |
| Entry | LIMIT @1.35445 |
| TP | 1.35220 |
| SL | 1.35530 |
| GTD | 20:55Z |
| Order ID | 468292 |
| Spread | 1.3pip |
| Thesis | The selloff impulse is still intact, but the live tape is compressing in the lower half of the box. The correct reload is the upper-half bounce back into 1.3544-1.3546, not a market chase at 1.3530. GDP-beat reversal is still fully faded and cable remains the cleanest USD-firmness seat. |
| pretrade | Edge B / Allocation B (score=5). Historical GBP_USD shorts are poor, so this stays a structural 3000u reload only. |
| Conviction | B -> 3000u (minimum viable size, hero pair concentration, late-session restraint) |
| Margin | With 468288 also live, worst-case pending margin remains only ~0.26% NAV. |
| Why now | `state.md` still referenced vanished worker order 468285, so the trader had to restore the missing reload path directly before session end. |
|-------|-------|
| Pair | GBP_JPY |
| Direction | CLOSE SHORT |
| Units | 3000u |
| Entry | 215.442 |
| Close | 215.570 |
| Loss | -12.8pip |
| P/L | -384.00 JPY |
| id | 468245 |
| Reason | JPY stopped acting like a clean weak leg, so GBP_JPY became the dirty version of the same late GBP short thesis. Kept GBP_USD as the cleaner expression and freed margin from 52% to 26%. |
| Notes | Third same-side short loss of the session sequence. No new SHORT entries this session; keep only the best remaining expression. |

## 16:24Z — EUR_USD worker LIMIT CANCELLED (id=468286)
| Field | Value |
|-------|-------|
| Pair | EUR_USD |
| Direction | CANCEL SHORT LIMIT |
| Units | 5000u |
| Entry | 1.17856 |
| TP | 1.17786 |
| SL | 1.17884 |
| tag | range_bot |
| Reason | Redundant USD-firmness expression. GBP_USD remained the cleaner hero pair and carrying both worker shorts would have blocked any valid second path on cable under the session margin cap. |

## 16:25Z — GBP_USD STOP LONG (failure-side path)
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Direction | LONG |
| Units | 3000u |
| Entry | STOP @1.35535 |
| TP | 1.35620 (+8.5pip) |
| SL | 1.35470 (-6.5pip) |
| GTD | 2026-04-16T20:25:27Z |
| Order ID | 468288 |
| Spread | 1.3pip |
| Thesis | GBP_USD is still the clean hero short, but the book cannot be one-price-and-done. If price reclaims 1.35535, the bearish staircase is broken and the next trade belongs to the upside failure break rather than another trapped short. |
| pretrade | B-equivalent session judgment: late theme, normal spread, failure level clearly defined. |
| Conviction | B → 3000u |
| Margin | With the surviving GBP_USD worker short 468285, worst-case fill stays ~69.5% NAV. |

## 16:31Z — EUR_USD worker LIMIT CANCELLED again (id=468289)
| Field | Value |
|-------|-------|
| Pair | EUR_USD |
| Direction | CANCEL SHORT LIMIT |
| Units | 5000u |
| Entry | 1.17860 |
| tag | range_bot |
| Reason | The local worker recreated the old EUR_USD fade before the new policy propagated. Removed immediately so the live book matches `PAUSE/CANCEL` and keeps GBP_USD as the only USD-firmness expression. |

---

## 07:47Z — GBP_JPY ENTRY LONG 2000u @215.467 id=468179

| Field | Value |
|-------|-------|
| Pair | GBP_JPY |
| Side | LONG |
| Units | 2000u |
| Entry | 215.467 (ask) |
| Spread | 2.8pip |
| TP | 215.580 (BB mid, +11.3pip net) |
| SL | 215.200 (-26.7pip structural, below range) |
| pretrade | C(1) → override to B (chart evidence) |
| Conviction | B (cold streak max) |
| Type | Range scalp |
| Zombie | 09:00Z |

**Thesis**: Range floor bounce. Price AT BB lower 215.44 (touched 215.40 with lower wick). M5/M15 StRSI=0.00 extreme oversold + H4 StRSI=0.02 floor. GDP beat = GBP structural bid. Multi-TF extreme pattern (proven 4/5 per recipe tracker).

**pretrade override**: C(1) rated caution on macro (GBP=0.00, JPY=+0.09). But range scalp buys the floor regardless of macro direction — floor is structural, not trend-chasing. Chart evidence > macro score here.

**FOR**: ① M5/M15 extreme oversold (Timing) ② H4 floor (Structure) ③ GDP beat GBP bid (Macro)
**Different lens**: H4 RSI_div bear — at H4 floor, bearish div = LAST LEG completing, not new downleg. Neutral for this setup.
**AGAINST**: JPY slightly bid (+0.09). M5 making lower lows but wick defense forming.
**If wrong**: M5 close below 215.40 with no wick defense = range broken → SL 215.200 catches.

---

## 07:51Z — GBP_USD LIMIT FILL LONG 2000u @1.35639 id=468175

| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Side | LONG |
| Units | 2000u |
| Entry | 1.35639 (LIMIT fill) |
| Spread | 1.3pip |
| TP | 1.35780 (BB upper, +14pip) |
| SL | 1.35530 (below range, structural) |
| pretrade | B (placed earlier this session) |

## 10:29Z — EUR_USD CLOSE (timing dead) -459.20 JPY
| Field | Value |
|-------|-------|
| Pair | EUR_USD |
| Direction | CLOSE LONG |
| Units | 3000u |
| Entry | 1.17826 |
| Close | 1.17730 |
| Loss | -9.6pip |
| P/L | -459.20 JPY |
| id | 468201 |
| Reason | Closed after preclose check. M5 extended through the prior low, M15 remained adverse, and the trade was already beyond its intended counter-bounce window. H4 floor alone was not enough to keep it. |

## 10:29Z — GBP_JPY LIMIT SHORT 3000u @215.440 id=468223
| Field | Value |
|-------|-------|
| Pair | GBP_JPY |
| Direction | SHORT (LIMIT) |
| Units | 3000u |
| Entry | 215.440 |
| TP | 215.220 (+22pip) |
| SL | 215.610 (-17pip) |
| GTD | 18:30Z |
| Spread | 2.8pip |
| pretrade | B (score=5, mid wave) |
| Conviction | B-size 3000u (theme late) |
| Thesis | UK GDP beat was fully rejected and GBP_JPY remains the cleanest M5+H1 trend-bear chart. Sell a structural bounce into EMA20 / BB mid rather than chase the low. |
| Margin after fill | ~20.8% of NAV |
| Type | Range scalp |
| Zombie | 09:00Z |

**Thesis**: LIMIT placed at BB lower of range 1.3565-1.3578. GDP +0.5% beat structural GBP bid. H4 StRSI=0.18 (bottom bull, massive room). London open catalyst expected 08:00Z. Filled on pre-London dip.

---

## 08:08Z — EUR_USD SL CLOSE -2000u @1.17850 id=468167 PL=-382 JPY

| Field | Value |
|-------|-------|
| Pair | EUR_USD |
| Side | LONG exit (SL fired) |
| Units | 2000u |
| Exit | 1.17850 (SL triggered) |
| Entry | 1.17970 (LIMIT fill 06:28Z) |
| P&L | -382 JPY |
| Spread at close | 0.8pip |
| Hold time | 1h40m |
| Exit reason | SL fired at London open — classic SL hunt at 08:08Z. Price bounced to 1.17881 immediately after. |

**Post-mortem**: H1 DI- turned adverse after entry (confirmed by profit_check 08:05Z session). London open SL sweep pattern — price dipped to exactly 1.17850 (our SL) then bounced. The H4 bull thesis remains intact (H4 StRSI=0.42 MID BULL) but H1 structure had deteriorated. SL placement at 1.17850 was structural (below M5 wave low at entry) but thin margin in squeeze = vulnerable to London hunt. Anti-churn rule prevents re-entry (new price worse than close price). Lesson: London open first 15min = high SL hunt risk. Either widen SL to ATR×2.0 or enter AFTER 08:15Z when London direction is clear.


| 08:55Z | LIMIT | EUR_USD | LONG | 3000u | @1.17750 | TP=1.18050 | SL=1.17600 | id=468195 | GTD=16:00Z | B-conv | pretrade=C(override) | Anti-churn=pass | H1 StRSI=0.00 + USD M1 flip offered + H4 MID BULL |
| 09:00Z | LIMIT | GBP_JPY | LONG | 3000u | @215.380 | TP=215.680 | SL=215.200 | id=468197 | GTD=14:00Z | B-conv | H4_StRSI=0.02_extreme+GDP_beat+Fib83%_retrace | Anti-churn=pass(13.4pip_better) | Replaced_GBP_USD_468194 |

## EUR_USD LONG 3000u — 09:08 UTC
| Field | Value |
|-------|-------|
| Entry | 1.17826 |
| Units | 3000u |
| id | 468201 |
| TP | 1.18100 (+27.4pip) |
| SL | 1.17700 (-12.6pip) |
| Spread | 0.8pip |
| pretrade | C (override) |
| Conviction | B (Counter-S S-recipe, pretrade=C+macro opposing = conservative) |
| Thesis | Counter-S: H4+H1 StRSI=0.00 (extreme floor within H4 ADX=50 DI+=31 BULL) + M5 StRSI=1.0 (local bounce exhausted). H4 structural bull + H1/H4 extreme oversold = reversal expected. |
| Type | Momentum |
| Expected hold | 30-120min |
| Zombie at | 11:08Z |

| 09:30Z | CLOSE | AUD_JPY | LONG | 2000u | @113.987 | PL=-126 JPY | reason=H4_StRSI=0.85_ceiling+noisy_scanner+C4_fails+zombie_46min |
| 09:31Z | CANCEL | GBP_USD | LIMIT | 3000u | @1.35350 id=468209 | — | reason=USD_M1_bid_momentum+avoid_4th_long |
| 11:11Z | CANCEL | USD_JPY | LIMIT | 3000u | @159.074 id=468226 | — | reason=stale_range_bot_order_outside_current_GBP_weakness_book |
| 11:11Z | LIMIT | GBP_USD | SHORT | 3000u | @1.35620 | TP=1.35270 | SL=1.35820 | id=468228 | GTD=18:30Z | B-conv pretrade=B(4) | GDP_spike_rejected+M5H1_trend_bear+bounce_sell_lower_high |

## 11:11Z — GBP_USD LIMIT SHORT 3000u @1.35620 id=468228
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Side | SHORT |
| Units | 3000u |
| Entry | 1.35620 LIMIT |
| TP | 1.35270 |
| SL | 1.35820 |
| GTD | 18:30Z |
| Spread at placement | 1.3pip |
| pretrade | B (score=4) |
| Thesis | UK GDP beat was fully rejected, GBP remains the weakest execution currency, and both M5/H1 still lean lower. Selling the lower-high bounce keeps the book aligned with the active London theme. |

## 11:25Z — GBP_USD ENTRY SHORT 3000u @1.35454 id=468230
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Side | SHORT |
| Units | 3000u |
| Entry | 1.35454 (MARKET order id=468229) |
| TP | 1.35270 |
| SL | 1.35580 |
| Spread | 1.3pip |
| pretrade | C (score=3) -> override to B |
| Conviction | B-size 3000u (theme late, but flat-book blocker + clean live continuation) |
| Type | Momentum |
| Expected hold | 30-120min |
| First confirmation by | 11:40Z |
| Zombie | 13:25Z |
| Margin after | 25.9% live / ~63.0% worst-case with both pending LIMITs |

**Thesis**: GDP beat was fully rejected and GBP_USD kept printing one-way M5 continuation instead of giving the planned 1.35620 bounce. H1 is rolling lower again under EMA20, so I sold the active move rather than staying flat into the ECB window.

**FOR**: ① M5 bearish continuation with no lower-wick defense ② H1 trend-bear structure reasserting ③ GBP remains the weakest execution currency on both M15 and M1
**AGAINST**: H4 StRSI is still at the floor, so this is an intraday continuation short, not a swing hold. The move is late, so size stays B.
**If wrong**: ECB or a GBP squeeze reclaims 1.35580 and turns this into a failed continuation. That invalidates the short.

## 12:35Z — GBP_JPY ENTRY SHORT 3000u @215.442 id=468245
| Field | Value |
|-------|-------|
| Pair | GBP_JPY |
| Side | SHORT |
| Units | 3000u |
| Entry | 215.442 (LIMIT fill from order id=468223) |
| TP | 215.220 |
| SL | 215.610 |
| Spread at fill | 3.0pip |
| Conviction | B-size 3000u (theme late) |
| Type | Momentum reload into trend-bear |
| Margin after fill | ~87% live book once the worker GBP_USD short was also on |
| Thesis | The planned GBP_JPY bounce-sell filled exactly at the EMA20 / BB-mid pullback while the broader M5 and H1 channels still leaned bear. This is still a GBP-liquidation trade, not a new theme. |

## 12:44Z — GBP_USD LIMIT SHORT CANCEL id=468228
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Side | SHORT LIMIT cancel |
| Units | 3000u |
| Price | 1.35620 |
| id | 468228 |
| Reason | Live margin was already ~87% after GBP_JPY 468245 filled and `trend_bot_market` added GBP_USD 468242, so the reload no longer fit the worst-case book even though the thesis still lived. |

| 14:22Z | CLOSE | GBP_USD | SHORT | 3000u | @1.35270 | PL=+882 JPY | reason=TP_auto_fire |
| 14:29Z | LIMIT | GBP_USD | SHORT | 3000u | @1.35305 | TP=1.35140 | SL=1.35420 | id=468261 | GTD=18:29Z | B-conv rotation reload | Late GBP weakness still alive, but only on a bounce back into broken structure |
| 14:29Z | LIMIT | AUD_JPY | LONG | 3000u | @114.000 | TP=114.180 | SL=113.920 | id=468262 | GTD=18:29Z | B-conv opposite path | Floor-defense long only if 114.000 fills and the squeeze base holds |

## 14:22Z — GBP_USD TP AUTO-FIRE SHORT CLOSE 3000u @1.35270 id=468230
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Side | SHORT exit |
| Units | 3000u |
| Exit | 1.35270 (TP auto-fire) |
| P&L | +882 JPY |
| Hold time | ~2h57m |
| Reason | The late GBP weakness theme still paid through the original structural TP even after the book slowed down. |

## 14:29Z — GBP_USD LIMIT SHORT 3000u @1.35305 id=468261
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Side | SHORT |
| Units | 3000u |
| Entry | 1.35305 LIMIT |
| TP | 1.35140 |
| SL | 1.35420 |
| GTD | 18:29Z |
| Spread at placement | 1.3pip |
| pretrade | B (score=4) |
| Thesis | Rotation reload after the TP: the theme still leans GBP-bearish, but the move is stretched below the prior target, so the re-entry belongs only on a bounce back into broken structure. |

## 14:29Z — AUD_JPY LIMIT LONG 3000u @114.000 id=468262
| Field | Value |
|-------|-------|
| Pair | AUD_JPY |
| Side | LONG |
| Units | 3000u |
| Entry | 114.000 LIMIT |
| TP | 114.180 |
| SL | 113.920 |
| GTD | 18:29Z |
| Spread at placement | 1.6pip |
| pretrade | B (score=4) |
| Thesis | Opposite-side path while flat: AUD remains the strongest structure, but the long only belongs if 114.000 holds as floor defense inside the current squeeze. |

## 15:03Z — EUR_JPY CLOSE (worker orphan cleanup) +6 JPY ✓
| Field | Value |
|-------|-------|
| Pair | EUR_JPY |
| Direction | CLOSE SHORT |
| Units | 3000u |
| Entry | 187.495 |
| Close | 187.493 |
| Gain | +0.2pip |
| P/L | +6.00 JPY |
| id | 468265 |
| Reason | Worker/manual orphan cleanup. The M5 box was only ~10pip wide while spread stayed 1.6-1.9pip, profit_check already favored taking profit, and book quality improved by returning to a flat book covered by GBP_USD and EUR_USD short reloads plus the AUD_JPY opposite-side defense order. |

## 15:09Z — GBP_USD ENTRY SHORT (reload LIMIT fill)
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Direction | SHORT |
| Units | 3000u |
| Entry | 1.35305 (LIMIT fill from id=468261) |
| TP | 1.35140 |
| SL | 1.35420 |
| Trade ID | 468273 |
| Spread | 1.3pip |
| Thesis | Rotation reload after the earlier TP: the bounce back into broken structure finally filled, and the lower-high staircase under EMA20 is still the cleanest live GBP weakness seat. |
| pretrade | B (score=4, from the original placement) |
| Conviction | B |
| Expected hold | 30m-2h |

## 17:09Z — EUR_USD CLOSE (unexpected worker short cleanup) -76.57 JPY ✗
| Field | Value |
|-------|-------|
| Pair | EUR_USD |
| Direction | CLOSE SHORT |
| Units | 4000u |
| Entry | 1.17722 (trend_bot_market fill id=468294) |
| Close | 1.17734 |
| Loss | -1.2pip |
| P/L | -76.57 JPY |
| Reason | Worker inventory cleanup. The refreshed policy had EUR_USD paused, the short filled in the lower half of the box with M5 already oversold, and I would not open that short fresh there. Keeping it would have been process drift, not conviction. |

## 17:11Z — EUR_USD CLOSE (repeat worker short cleanup) +69.93 JPY ✓
| Field | Value |
|-------|-------|
| Pair | EUR_USD |
| Direction | CLOSE SHORT |
| Units | 4000u |
| Entry | 1.17728 (trend_bot_market fill id=468302) |
| Close | 1.17717 |
| Gain | +1.1pip |
| P/L | +69.93 JPY |
| Reason | Repeat worker inventory cleanup. The same off-plan EUR_USD short reappeared immediately after the first cleanup, which proved the worker layer was not honoring the intended pair freeze; trader flattened it and escalated the worker policy to PAUSE_ALL. |

## 19:25Z — USD_JPY worker LIMIT CANCEL id=468375
| Field | Value |
|-------|-------|
| Pair | USD_JPY |
| Side | SHORT LIMIT cancel |
| Units | 3000u |
| Price | 159.37194 |
| id | 468375 |
| Reason | Deterministic guard reopened an off-plan repair short into an upper-half squeeze near intervention air. Trader canceled it and removed USD_JPY from the worker repair map. |

## 19:27Z — EUR_USD worker LIMIT CANCEL id=468377
| Field | Value |
|-------|-------|
| Pair | EUR_USD |
| Side | SHORT LIMIT cancel |
| Units | 3000u |
| Price | 1.17835 |
| id | 468377 |
| Reason | Guard immediately recycled the next off-plan worker repair seat into EUR_USD, duplicating the trader-owned cable short theme. Trader canceled it and stepped the worker layer down to REDUCE_ONLY. |

## 19:27Z — EUR_USD worker LIMIT CANCEL id=468379
| Field | Value |
|-------|-------|
| Pair | EUR_USD |
| Side | SHORT LIMIT cancel |
| Units | 3000u |
| Price | 1.17835 |
| id | 468379 |
| Reason | One more stale EUR_USD worker short remained live after the policy downgrade. Trader canceled it so the worker book finished flat under REDUCE_ONLY. |

## 20:26Z — GBP_USD STOP LONG replacement id=468383
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Direction | LONG STOP |
| Units | 3000u |
| Entry | 1.35535 |
| TP | 1.35620 |
| SL | 1.35470 |
| GTD | 2026-04-17T00:30:00Z |
| Order ID | 468383 |
| Spread | 1.3pip |
| Thesis | Failure-flip continuation map stays valid while cable keeps compressing under 1.3533/35. If price accepts above 1.35535, the squeeze has reclaimed the cap and the other side of the hero-pair map must stay live. |
| Reason | Prior stop order 468288 expired at 20:25Z while the box was still unresolved, so the failure-side path was re-posted with a real multi-hour GTD to keep both structural paths deployed. |

## 20:46Z — GBP_USD LIMIT SHORT CANCEL id=468292
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Side | SHORT LIMIT cancel |
| Units | 3000u |
| Price | 1.35445 |
| id | 468292 |
| Reason | The 1.35445 sell-high was no longer the live shelf after cable slipped into a lower-band bleed. Trader canceled the stale price instead of letting the map decay into an effectively dead reload. |

## 20:46Z — GBP_USD LIMIT SHORT replacement id=468385
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Direction | SHORT LIMIT |
| Units | 3000u |
| Entry | 1.35305 |
| TP | 1.35220 |
| SL | 1.35385 |
| GTD | 2026-04-17T00:30:00Z |
| Order ID | 468385 |
| Spread | 1.3pip |
| pretrade | B (score=4, refreshed live) |
| Conviction | B |
| Thesis | Cable is still the cleanest live GBP-weakness vehicle, but the active sell shelf has shifted down to 1.3530/10 as the M5 squeeze bleeds along the lower band. The short map stays valid only if the reload matches that live shelf. |
| Reason | Replaced the stale 1.35445 sell-high with a closer structural reload so the hero-pair two-way map remains real through the next session instead of expiring into one-path deployment. |

## 22:57Z — AUD_JPY LIMIT LONG placed (Tokyo bounce play)
| Field | Value |
|-------|-------|
| Pair | AUD_JPY |
| Direction | LONG (LIMIT) |
| Units | 3000u |
| Entry | LIMIT @113.884 |
| TP | 114.020 (+13.6pip) |
| SL | 113.800 (-8.4pip) |
| R:R | 1.62 |
| GTD | 2026-04-17T06:54Z |
| Order ID | 468471 |
| Pretrade | C (setup quality weak: ADX=16, macro opposing AUD=-0.08 JPY=+0.05) |
| EV | +228 JPY/trade historical (positive despite C) |
| Thesis | H1 StRSI=0.00 floor exhaustion + H4 MID BULL (StRSI=0.31, room) + Fib M5 N=BULL q=2.45 (excellent) + H1 Fib at Fib17% (near end of H1 bear wave). Structural bounce play for Tokyo session open. |
| Why LIMIT | Worst trading hour (22:00-23:00 UTC) + cold streak B-max. LIMIT waits for Tokyo session for actual fill. Avoids worst-hour market entry. |
| Max loss | 252 JPY (3000u × 8.4pip × 0.01) |

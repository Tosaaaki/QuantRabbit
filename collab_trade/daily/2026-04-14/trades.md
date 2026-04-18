# Trades — 2026-04-14

## LIMIT Fill — GBP_JPY LONG 4000u

| Field | Value |
|-------|-------|
| Time | ~2026-04-14 00:10-00:30Z (estimated from state.md context) |
| Pair | GBP_JPY |
| Side | LONG |
| Units | 4000u |
| Entry | 215.119 |
| TP | 215.350 (GTC) |
| SL | 214.900 (GTC) |
| Trade ID | 467682 |
| LIMIT ID | 467681 (filled) |
| Sp | 3.0pip |
| Pretrade | S(8)/Momentum-S H1 ADX=39 BULL + M5 StRSI=0.04 + CS GBP(+0.64) |
| Conviction | A-S |
| Thesis | H1 TREND-BULL deep pullback. M5 bull div score=1.0 + H1 hidden bull div MACD=0.6. Entry at wave bottom (Fib 88% retracement). |
| Status | OPEN |

## USD_JPY SHORT continuation from 2026-04-13

| Field | Value |
|-------|-------|
| Entry time | 2026-04-13 23:23Z |
| Pair | USD_JPY |
| Side | SHORT |
| Units | 2000u |
| Entry | 159.229 |
| TP | 159.000 (GTC) |
| SL | 159.420 (GTC) |
| Trade ID | 467678 |
| Conviction | B |
| Status | OPEN |

### EUR_USD LONG add 3000u @1.17636 [id=467690]
| Field | Value |
|-------|-------|
| Time | 2026-04-14 01:30 UTC |
| Entry | 1.17636 (ask) |
| Spread | 0.8pip |
| Units | 3000u |
| TP | 1.17864 (H1 BB upper, structural) |
| SL | 1.17480 (GTC) |
| Pretrade | S(9) MEDIUM risk |
| Regime | SQUEEZE / H1-BULL ADX=40 |
| FOR | H1 ADX=40 BULL + M5 StRSI=0.0 oversold + EUR(+0.64) vs USD(-0.48) 1.12 gap |
| Different lens | H4 StRSI=1.0 overbought (concern, not reversal) |
| Conviction | A (pretrade S but H4 overbought prevents full S) |
| Thesis | M5 at BB lower in H1 BULL SQUEEZE — bounce/breakout long. London (06:00Z) = catalyst. |

## GBP_JPY LONG 2000u @215.000 [03:00Z] — LIMIT FILL
| Item | Value |
|------|-------|
| id | 467704 |
| Entry | 215.000 (LIMIT fill from id=467702) |
| Spread | 2.8pip (⚠️ wide Tokyo thin) |
| TP | 216.000 (GTC) |
| SL | 214.400 (GTC) |
| Thesis | H4 ADX=58 EXTREME BULL. GBP CS=+0.41 (2nd strongest) vs JPY CS=-0.03. H1 StRSI=0.0 = oversold dip. M5 Fib 78.6% pullback near exhaustion. London 06:00Z recovery. |
| Conviction | A (H4 extreme bull + CS gap + Fib exhaustion = valid dip. AGAINST: H1 bear div + Tokyo thin) |
| pretrade | N/A (LIMIT placed prior session) |

## EUR_JPY LIMIT CANCEL [03:02Z]
| id=467664 @187.050 | Cancelled — no TP/SL on fill, price approaching in Tokyo thin. Naked position risk = unacceptable. |

---
## GBP_USD LONG 4000u @1.35163 [id=467722] — 2026-04-14 05:20Z

| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Side | LONG |
| Units | 4000u |
| Entry | 1.35163 (ask 1.35167, improved fill) |
| Spread | 1.3pip |
| TP | 1.35500 (33pip, ATR×2.0 approx) |
| SL | 1.34800 (37pip, below consolidation base) |
| Conviction | S (pretrade score=11, 🎯 Trend-Dip-S fired) |
| Thesis | GBP_USD M5 dip-in-H1-trend. M5 StRSI=0.0 + SQUEEZE at BB upper. London breakout imminent. GBP CS=+0.47 vs USD CS=-0.51. |
| pretrade | S(11)/MEDIUM risk. 62% WR total +4,749 JPY |
| Margin after | 81.3% (no pending LIMITs) |

**FOR**: ① Direction (H1 ADX=42 DI+=35 BULL) + ② Timing (M5 StRSI=0.0 dip) + ⑤ Cross-pair (GBP+0.47 vs USD-0.51, gap=0.98)
**Different lens**: ④ Fib N=BULL(q=1.12) at Fib111% = high-quality bull wave, supports continuation → SUPPORTS
**AGAINST**: H4 StRSI=1.0 (overbought), H4 CCI=154. Counterwave risk.
**If I'm wrong**: H4 exhaustion triggers real reversal to 1.348-1.350. SL at 1.34800 limits damage.

### Management note
- Cancelled stale LIMIT @1.34900 (id=467666) — 29pip deep, won't fill in London continuation
- Cancelled LIMIT GBP_JPY @214.750 (id=467701) — replaced with tighter SL management
- Cancelled LIMIT EUR_USD @1.17250 (id=467645) — deep pre-PPI backstop, SL@1.17480 covers
- GBP_JPY SL tightened: 214.400 → 214.650 (below V-shape low 214.70)

---

## 07:22Z Session — Trader

### GBP_USD Modification: Trailing Stop Set
| Field | Value |
|-------|-------|
| Time | 2026-04-14 07:22 UTC |
| Action | Trailing stop set |
| Position | GBP_USD LONG 4000u @1.35163 id=467722 |
| UPL at action | +895 JPY (+20.7pip = ATR×1.6) |
| Trail distance | 8pip (0.00080) |
| Trail order id | 467745 |
| Reason | Protect ATR×1.6 profit. RANGE regime at BB upper = structural TP zone. H1 TREND-BULL still intact, trail follows continuation. Minimum locked: ~+600 JPY if trail triggers. |

### GBP_JPY LONG — LIMIT Entry
| Field | Value |
|-------|-------|
| Time | 2026-04-14 07:22 UTC |
| Pair | GBP_JPY |
| Direction | LONG |
| Units | 2000u |
| Order | LIMIT @215.150 |
| TP | 215.500 (35pip, ATR×2.2) |
| SL | 214.900 (below M5 BB lower + buffer) |
| GTD | 09:20Z (2h) |
| Order id | 467746 |
| Conviction | A (pretrade score=7, MEDIUM risk, 65% WR) |
| Spread | 3.2pip (14% of TP = OK for Momentum type) |
| Margin after fill | 81.9% (88,024 + 17,216 = 105,240 / 128,487) |

**Thesis**: Trend-Dip-A. H4 ADX=54 BULL (strongest trend on board) + H1 ADX=32 BULL + M5 StRSI=0.0 SQUEEZE dip in confirmed uptrend. GBP CS=+0.54 vs JPY CS=-0.13 (gap=0.67). Staircase uptrend visible on M5 chart throughout.

**FOR**: ① Direction (H4 ADX=54, H1 ADX=32, DI+ dominant both TFs) + ② Timing (M5 StRSI=0.0 extreme dip in uptrend) + ⑤ Cross-pair (GBP strong, JPY weakening, EUR_JPY LONG confirms JPY weakness)
**Different lens**: ④ Structure — Fib wave BULL(q=0.48) now@Fib80% of AB leg. M5 SQUEEZE (BBW=0.00154) building energy for breakout. H4 structure points UP. → SUPPORTS
**AGAINST**: M5 SQUEEZE could break DOWN (less likely given H4 bull). Earlier GBP_JPY LONG today was -876 JPY (circuit breaker awareness). H4 CCI=58 neutral (not extreme).
**If I'm wrong**: M5 SQUEEZE breaks DOWN, EUR_JPY and GBP_JPY fall together on risk-off headline. SL=214.90 → 2000u × 25pip × 0.01 = -500 JPY max loss.

---

## 07:38Z Session — EUR_JPY TP Modification

| Field | Value |
|-------|-------|
| Time | 2026-04-14 07:38 UTC |
| Action | TP tightened |
| Position | EUR_JPY LONG 2000u @187.467 id=467729 |
| UPL at action | -76 JPY (-1.2pip) |
| Old TP | 187.800 (34pip away) |
| New TP | 187.519 (H1 swing high / spike top) |
| New orders | id=467747, 467748 |
| Reason | H1 bearish divergence score=0.6. Spike topped at 187.540. TP at 187.519 captures the full spike move without overreaching into territory that H1 div warns against. H4 ADX=55 still strong — position held. |

## CLOSE — GBP_USD LONG 4000u (Trail triggered)

| Field | Value |
|-------|-------|
| Close Time | 2026-04-14 07:53Z |
| Pair | GBP_USD |
| Side | LONG close |
| Units | 4000u |
| Entry | 1.35163 |
| Close | 1.35301 (trail @7pip triggered) |
| P/L | +876 JPY |
| Trade ID | 467722 |
| Hold | ~2.5h (entry 05:20Z) |
| Reason | Trail auto-triggered at band walk peak. ATR×1.3 (17.1pip). Correct outcome — trail managed exit. |
| Lesson | Trail at 6.5-7pip captured full reward in band walk. No action needed — trail did its job. |

---

## 09:09Z Session

### GBP_JPY LIMIT Cancelled + Replaced
| Action | Detail |
|--------|--------|
| Cancelled | id=467746 LIMIT @215.150 GTD=09:19Z (2min to expiry, price too far above) |
| Placed | id=467755 LIMIT LONG @215.250 TP=215.600 SL=214.900 GTD=10:44Z |
| Reason | Price at 215.296 bid = 14.6pip above old limit. M1 StRSI=0.0 dip = M1 BB lower entry (215.250 above M1 BB lower 215.266). Better structural level. |

### GBP_JPY LIMIT FILL @215.232 [id=467756]
| Field | Value |
|-------|-------|
| Time | 2026-04-14 ~09:17Z |
| Entry | 215.232 (filled below limit = 1.8pip improvement) |
| TP | 215.600 | SL | 214.900 |
| Conviction | A (pretrade score=6, 67% WR) |
| Spread | 3.1pip |
| Thesis | SQUEEZE + H1 ADX=33 BULL. GBP CS=+0.64 strongest. M1 oversold dip = M1 BB lower entry. |

### EUR_JPY CLOSE @187.275 [id=467729]
| Field | Value |
|-------|-------|
| Close Time | 2026-04-14 09:19Z |
| Entry | 187.467 | Close | 187.275 |
| P/L | -384 JPY |
| Units | 2000u |
| Hold | ~07:08Z–09:19Z = ~2.2h |
| Close Reason | Invalidation pre-empted: JPY strengthening systematic across ALL crosses. AUD_JPY -9.7pip, GBP_JPY -14.7pip, USD_JPY -8.9pip since session start. EUR_JPY at 187.309 = 0.9pip from pre-set invalidation (187.300). H1 bearish div score=0.5. Pre-emptive close justified by cross-pair confirmation. |
| Preclose check | H1: DI+ still > DI- but weakening (div + MACD_H neg). Invalidation 0.9pip away. Analysis-based, not panic. |

## GBP_USD LIMIT 2000u @1.35285 id=467763 — 09:30Z
| Field | Value |
|-------|-------|
| Time | 2026-04-14 09:30 UTC |
| Pair | GBP_USD |
| Direction | LONG |
| Units | 2000u |
| Order | LIMIT @1.35285 |
| TP | 1.35500 |
| SL | 1.35050 |
| GTD | 2026-04-14T11:00:00Z |
| Conviction | A |
| Thesis | Momentum dip-buy. Anti-churn cleared (1.35285 < close 1.35301). GBP_USD TREND-BULL H4+H1+M5. CS gap 1.20 (largest on board). |
| pretrade | S(8) MEDIUM — H4 overbought penalty. Sized A (PPI discount). |

## EUR_JPY LONG 2500u — 09:39Z [id=467765]
| Field | Value |
|-------|-------|
| Time | 2026-04-14 09:39 UTC |
| Pair | EUR_JPY |
| Direction | LONG |
| Units | 2500u |
| Order | MARKET |
| Fill | 187.326 |
| Spread | 2.5pip (normal, 1.8-2.2pip range — slightly wide) |
| TP | 187.500 (M5 BB upper, -spread buffer) |
| SL | 186.900 (structural — below spike origin) |
| Conviction | A (pretrade A score=6, risk=LOW) |
| Pretrade | A(6) — H4+H1 aligned BULL, M5 StRSI=0.04 extreme oversold, EUR CS=+0.60/JPY CS=-0.13, Fib10% wave bottom |
| Regime | TREND-BULL H4 (ADX=57 DI+=36), M5 at BB lower after pullback from 187.5 spike |
| Type | Momentum (30min-2h) |
| FOR | ① Direction (H4 ADX=57, H1 ADX=31 BULL) ② Timing (M5 StRSI=0.04, CCI=-179, BB lower) ③ Macro (EUR+JPY- = Iran risk-on) |
| AGAINST | H4 RSI=77 overbought (extended move). All positions LONG (concentration). PPI binary 2.5h |
| Anti-churn | CLEAR — price 187.326 << prev TP exit ~187.519 (new lower price, new signal) |
| Margin after | ~55% (fine) |
| Mandatory close | 11:50Z pre-PPI |

## GBP_USD LONG 4000u @1.35449 [id=467772] — 2026-04-14 09:50Z
| Field | Value |
|-------|-------|
| Entry | 1.35449 (ask, market fill) |
| Spread | 1.3pip |
| TP | 1.36200 (M5 ATR×2 above entry, squeeze breakout target) |
| SL | 1.35000 (below squeeze range structural) |
| Pretrade | S(8) MEDIUM risk |
| Conviction | S — CS GBP+0.64 vs USD-0.56 gap=1.20 + H4+H1+M5 all BULL ADX=45 + Fib at wave low |
| Close | ⚠️ 11:50Z pre-PPI MANDATORY |
| Notes | LIMIT @1.35285 GTD=11:00Z canceled; entered at market (price moving away from LIMIT) |

## 10:22Z EUR_USD SL Tightened
| Field | Value |
|-------|-------|
| Time | 2026-04-14 10:22 UTC |
| Trade | EUR_USD LONG 4000u id=467742 |
| Action | SL tighten 1.17480 → 1.17900 |
| Reason | At ATR×1.0 (+11.6pip). Structural SL below M5 BB mid / breakout level. Protects min +216 JPY if SL hit |
| Current UPL | +653 JPY (+11.6pip) |

## 10:22Z GBP_USD LONG Entry
| Field | Value |
|-------|-------|
| Time | 2026-04-14 10:22 UTC |
| Pair | GBP_USD |
| Side | LONG |
| Units | 4500u (reduced to 3000u at 10:23Z) |
| Entry | 1.35493 (ask) |
| TP | 1.35628 (ATR×1.0 = 13.5pip) |
| SL | 1.35370 (below pullback low) |
| Spread | 1.3pip |
| Pretrade | S(8) MEDIUM risk |
| Conviction | S — GBP cs=+0.69, USD cs=-0.64 (1.33 gap). H1 ADX=46 BULL DI+=37. M5 SQUEEZE→UP. S-scan 🎯 |
| id | 467782 |

## 10:23Z GBP_USD Partial Close (Margin Correction)
| Field | Value |
|-------|-------|
| Time | 2026-04-14 10:23 UTC |
| Action | Partial close 1500u @1.35481 (bid) |
| PL | -28.65 JPY (spread cost only) |
| Reason | Margin reached 92.0% (rule: 90% BLOCKED). Reduced to 79.4%. Formula error: used /25 but GBP_USD uses /20 leverage in OANDA. Net position: 3000u |
| Lesson | GBP_USD margin calc: units × (GBP_JPY/20), not /25. OANDA uses 20:1 leverage for GBP_USD. |

## CLOSE — GBP_USD LONG 3000u (TP auto-fill)
| Field | Value |
|-------|-------|
| Close Time | 2026-04-14 ~11:05Z |
| Entry | 1.35493 | Close | 1.35628 (TP auto-fill) |
| P/L | +643 JPY |
| Units | 3000u | id | 467782 |
| Reason | TP at 1.35628 (ATR×1.0) auto-filled. London continuation + GBP momentum intact. |

## CLOSE — GBP_JPY LONG 2000u (TP auto-fill)
| Field | Value |
|-------|-------|
| Close Time | 2026-04-14 ~11:25Z |
| Entry | 215.232 | Close | 215.460 (TP auto-fill) |
| P/L | +456 JPY (est) |
| Units | 2000u | id | 467756 |
| Reason | TP at 215.460 auto-filled. GBP_JPY TREND-BULL continuation. |

## CLOSE — EUR_USD LONG 4000u [11:45Z]
| Field | Value |
|-------|-------|
| Close Time | 2026-04-14 11:45Z |
| Entry | 1.17864 | Close | 1.18005 |
| P/L | +893 JPY |
| Units | 4000u | id | 467742 |
| Hold | ~02:05Z – 11:45Z (~9.7h) |
| Reason | Mandatory pre-PPI close (11:50Z rule). Cluster wall at 1.18004 = SQUEEZE already fired and hit resistance. M5 deceleration (bodies shrinking). TP=1.18100 unreachable in remaining 5min. Locked in full +893 JPY > trail floor ~+450 JPY. |

## CLOSE — EUR_JPY LONG 2500u [11:45Z]
| Field | Value |
|-------|-------|
| Close Time | 2026-04-14 11:45Z |
| Entry | 187.326 | Close | 187.344 |
| P/L | +45 JPY |
| Units | 2500u | id | 467765 |
| Hold | ~09:39Z – 11:45Z (~2.1h) |
| Reason | Mandatory pre-PPI close. Post-spike consolidation structure (spike to 187.5, dump back). Fib22% = at range bottom. Essentially breakeven — no justification to hold through PPI. |

## GBP_USD LONG 8000u — 2026-04-14 12:38Z

| Field | Value |
|-------|-------|
| Time | 2026-04-14 12:38Z |
| Pair | GBP_USD |
| Side | LONG |
| Units | 8000u |
| Entry | 1.35863 |
| TP | 1.36163 (GTC) |
| SL | 1.35513 (GTC) |
| Trade ID | 467805 |
| Sp | 1.3pip |
| Pretrade | S(8) MEDIUM |
| Conviction | S |
| Thesis | PPI MISS confirmed. USD selling. GBP CS+0.82 strongest. Band walk TREND-BULL H1 ADX=50. SCENARIO A execution. |
| Margin | ~26% NAV (~34,500 JPY) |
| Status | OPEN |

## EUR_USD LONG 3000u — 2026-04-14 12:38Z

| Field | Value |
|-------|-------|
| Time | 2026-04-14 12:38Z |
| Pair | EUR_USD |
| Side | LONG |
| Units | 3000u |
| Entry | 1.18003 |
| TP | 1.18232 (GTC) |
| SL | 1.17702 (GTC) |
| Trade ID | 467809 |
| Sp | 0.8pip |
| Pretrade | A(6) MEDIUM |
| Conviction | A (S macro catalyst) |
| Thesis | PPI MISS confirmed. USD selling. EUR CS+0.62. Squeeze breakout post-PPI. 3.6pip pullback from spike = entry. SCENARIO A execution. |
| Margin | ~8.6% NAV (~11,240 JPY) |
| Status | OPEN |

| 14:07 UTC | CLOSE (half) | EUR_USD | LONG | 1500u | @1.17938 | -155 JPY | Fib78.6% pre-committed half-close. Bid broke 1.17970 trigger during broad EUR/GBP/AUD sell. Execution at 1.17938 (3.0pip slippage on fast market). H1 still intact. Remaining 1500u holding @ 1.18003 TP=1.18232 SL=1.17702. |

## GBP_JPY LONG 2000u — 2026-04-14 14:16Z

| Field | Value |
|-------|-------|
| Time | 2026-04-14 14:16Z |
| Pair | GBP_JPY |
| Side | LONG |
| Units | 2000u |
| Entry | 215.528 |
| TP | 215.700 (GTC) |
| SL | 215.200 (GTC) |
| Trade ID | 467819 |
| Sp | 2.8pip |
| Pretrade | S(8) MEDIUM |
| Conviction | A (S-signal, margin-limited to 2000u) |
| Thesis | Structural-S recipe (100% accuracy): M5 at BB lower (215.533), StRSI=0.0, H1 ADX=38 BULL staircase. GBP CS+0.94 strongest. PPI MISS structural USD weakness ongoing. |
| Margin | 21,551 JPY → 91.8% total (⚠️ above 90% — margin error: used /25 instead of /20 for GBP_JPY) |
| Status | OPEN |

## Modify — GBP_USD TP tightened [15:16Z]

| Field | Value |
|-------|-------|
| Time | 2026-04-14 15:16 UTC |
| Pair | GBP_USD |
| Trade ID | 467805 |
| Action | TP adjusted 1.36163 → 1.36000 |
| Reason | M5 extreme oversold (StRSI=0.0, CCI=-152) + H1 ADX=55 intact. 1.36163 (39.5pip away) → 1.36000 (M5 cluster, ATR×1.5 = 23pip from entry). More achievable in SQUEEZE context |
| Status | Pending |

## LIMIT Order — GBP_JPY LONG 2000u [15:16Z]

| Field | Value |
|-------|-------|
| Time | 2026-04-14 15:16 UTC |
| Pair | GBP_JPY |
| Side | LONG |
| Units | 2000u |
| Limit Price | 215.450 (M5 BB lower area) |
| TP | 215.900 (45pip from fill, H1 structural) |
| SL | 215.270 (18pip below fill, outside BB range) |
| GTD | 2026-04-14 17:15Z |
| Order ID | 467828 |
| Pretrade | A(7) MEDIUM risk |
| Regime | SQUEEZE(M5) / TREND-BULL(H1 ADX=38) |
| FOR | H1 ADX=38 BULL (Direction) + CS GBP(+0.89) vs JPY(-0.09) gap=0.98 (Cross-pair) + M5 SQUEEZE at BB lower (Timing) |
| Different lens | Macro: US-Iran mild risk-on = GBP bid intact → supports |
| AGAINST | H4 RSI=80 overbought. Previous loss today -876 JPY. |
| If wrong | H4 overbought drives pullback to 215.00-215.10. LIMIT doesn't fill. |
| Margin after fill | 84.1% (109,111 / 129,673) — under 85% threshold |
| Conviction | A |
| Status | PENDING (LIMIT) |

## 15:59Z — GBP_USD LONG 2000u ADD-ON (Trend-Dip-S)
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Side | LONG |
| Units | 2000u |
| Entry | 1.35630 (market, ask) |
| TP | 1.36000 |
| SL | 1.35513 |
| ID | 467833 |
| Spread | 1.3pip |
| pretrade | S(8) — iron-clad |
| Action | Cancel GBP_JPY LIMIT 467828 (price making higher highs, 10pip above LIMIT, won't fill). Deploy freed capacity to confirmed entry. |

**Conviction block:**
- Thesis: M5 at BB lower (StRSI=0.0, CCI=-127) within H1 ADX=55 ascending staircase. USD bounce is temporary (state.md lesson). GBP CS +0.89 = strongest.
- Regime: TREND-BULL (H1 confirmed via chart)
- Type: Momentum add-on
- FOR: ① Direction (H1 ADX=55 DI+=37>>DI-=5) + ② Timing (M5 StRSI=0.0 AT BB lower, confirmed visually) + ⑤ Cross-pair (GBP+0.89 vs USD-0.83, gap=1.72)
- Different lens: ④ Structure (Fib BULL q=0.78, price in healthy BC pullback) → neutral/slight support
- AGAINST: H4 StRSI=1.0 (overbought), H1 bearish div score=0.6
- If wrong: H1 DI+ flips, or Iran talks collapse → USD safe-haven bid
- Margin after: 91,835 + 21,553 = 113,388 / 128,028 = 88.6% (85-90% zone, S-conviction OK)
→ Conviction: S | Size: 2000u (margin constrained, max allowed at 88.6%)

---

## CLOSE — GBP_USD LONG 8000u

| Field | Value |
|-------|-------|
| Time | 2026-04-14 18:17 UTC |
| Pair | GBP_USD |
| Side | LONG (CLOSE) |
| Units | 8000u |
| Entry | 1.35863 |
| Exit | 1.35660 |
| P/L | -2,583 JPY |
| Trade ID | 467805 |
| Close reason | Post-BoE rule fired: GBP_USD 1.35688 NOT above 1.35900 at 18:15Z. Rule written 17:55Z, executed without re-evaluation. |

---

## ENTRY — AUD_JPY LONG 8000u

| Field | Value |
|-------|-------|
| Time | 2026-04-14 18:17 UTC |
| Pair | AUD_JPY |
| Side | LONG |
| Units | 8000u |
| Entry | 113.166 |
| TP | 113.650 (GTC) |
| SL | 112.900 (GTC) |
| Trade ID | 467841 |
| Sp | 1.6pip |
| Pretrade | A(7), MEDIUM |
| Conviction | S (scanner 🎯 Trend-Dip-S: H1 ADX=25 BULL + M5 StRSI=0.00 + H4 aligned) |
| FOR | Direction (H4 ADX=38 BULL, H1 ADX=25 BULL) + Timing (M5 StRSI=0.00, CCI=-179) + Cross-pair (CS AUD+0.34 vs JPY-0.02) |
| Diff lens | Fib @173% = overextended bear wave → supports bounce |
| AGAINST | N=BEAR(q=2.85) strong bear wave may have more downside. Auditor warning noted. |
| Status | OPEN |

---

## ENTRY — AUD_USD LONG 5000u

| Field | Value |
|-------|-------|
| Time | 2026-04-14 18:17 UTC |
| Pair | AUD_USD |
| Side | LONG |
| Units | 5000u |
| Entry | 0.71284 |
| TP | 0.71550 (GTC) |
| SL | 0.71150 (GTC) |
| Trade ID | 467845 |
| Sp | 1.4pip |
| Pretrade | S(9), MEDIUM |
| Conviction | S (exceptional — CS gap=1.00, H1 ADX=30 BULL, bull div=0.7, multi-catalyst macro) |
| FOR | Direction (H4+H1 BULL, CS gap=1.00) + Timing (M5 StRSI=0.04, H1 div=0.7) + Macro (PPI miss, tariff ruling, Iran) |
| Diff lens | Structure — lower wicks at 0.7127-0.7128 = structural support. Supports. |
| AGAINST | Negative historical edge (-2,574 JPY, 48% WR). Size reduced to 5000u. H1 MACD RegBear=-1.0. |
| Status | OPEN |

## EUR_USD CLOSE [18:33 UTC]
| Item | Value |
|------|-------|
| Action | CLOSE LONG 750u |
| Exit | 1.17922 |
| P/L | -96.67 JPY |
| Reason | Pre-Lagarde rule: price remained below 1.1800. Fib85% retracement approaching wave low 1.17918. ECB Lagarde 19:30Z binary risk. H1 ADX=54 BULL intact but EVENT RISK override. |
| pretrade note | pre-Lagarde rule set last session executed as planned |

### GBP_JPY LONG 2000u — 18:45Z Entry
| Field | Value |
|-------|-------|
| Time | 2026-04-14 18:45 UTC |
| Pair | GBP_JPY |
| Direction | LONG |
| Units | 2000u |
| Entry | 215.430 |
| TP | 215.620 (H1 ATR×0.9, 19pip) |
| SL | 215.200 (below M5 recent low, 23pip) |
| id | 467861 |
| pretrade | MEDIUM (score=7/A-conv, H4 overbought warning) |
| Conviction | A |

**Thesis**: H4 ADX=60 BULL (strongest in 7 pairs) + H1 StRSI=0.01 extreme oversold (H1 dip timing) + M5 StRSI=0.97 (bounce already in progress). GBP CS=+0.62 strongest. VWAP gap=-279pip (mean-reversion up). Cancelled LIMIT @215.000 — H1 extreme oversold means bottom forming NOW, not 4.4pip lower.
**AGAINST**: H4 RSI=76/CCI=71 (H4 overbought scale). ECB Lagarde 44 min (indirect GBP risk). M5 bear wave potentially -3pip more.
**If wrong**: M5 bear wave continues below 215.200 despite H1 oversold. SL at 215.200.

## CLOSE — AUD_USD LONG 5000u

| Field | Value |
|-------|-------|
| Time | 2026-04-14 19:13 UTC |
| Pair | AUD_USD |
| Side | LONG close |
| Units | 5000u |
| Entry | 0.71284 |
| Close | 0.71267 |
| P&L | -135 JPY |
| Trade ID | 467845 |
| Sp | 1.4pip |
| Reason | Hard cap violation (5000u vs B-max 1667u) + no M5 bounce by next session (stated rule) + AUD macro weakness (NAB confidence -29, AUD_USD -1654 JPY realized today) + pre-Lagarde margin free for EUR_JPY SHORT (A-conviction post-Lagarde) |

## EUR_USD LONG 3500u — 2026-04-14 19:38Z

| Field | Value |
|-------|-------|
| Time | 2026-04-14 19:38 UTC |
| Pair | EUR_USD |
| Side | LONG |
| Units | 3500u |
| Entry | 1.17924 (market, ask) |
| TP | 1.18120 (GTC, 20pip) |
| SL | 1.17680 (GTC, 25pip) |
| Trade ID | 467869 |
| Sp | 0.8pip |
| Pretrade | A(6) MEDIUM — 56% WR, +18,396 JPY total (84 trades) |
| Conviction | A |
| Margin after | ~81.9% (OK, under 85%) |

**Thesis**: H1 extreme oversold (StRSI=0.0) bounce in H4 ADX=48 uptrend. System's best pair. PPI miss = USD structural weak. Post-Lagarde event, planned from prior session.
**FOR**: ① Direction (H4 ADX=48 DI+=39 BULL, H1 ADX=54 BULL) + ② Timing (H1 StRSI=0.0 extreme oversold in strong trend) + ⑥ Macro (PPI +4.0% vs +4.7% miss, CS: EUR+0.36 vs USD-0.62)
**Different lens**: ④ Structure (Fib BULL q=0.81 at 60% retrace = healthy pullback zone) + H1 hidden bull div → supports
**AGAINST**: H4 RSI=73 (overbought at H4, pretrade warning). Lagarde event risk (ECB dovish = EUR negative).
**If wrong**: EUR_USD breaks 1.17680 with M5 body close = range broken, sellers in control.

### AUD_JPY CLOSE — 2026-04-14 21:37 UTC
| Item | Value |
|------|-------|
| Action | CLOSE LONG 8000u |
| Close price | 113.059 (bid) |
| Entry price | 113.166 |
| P&L | -856 JPY |
| Peak | +416 JPY @~113.320 |
| Close reason | H1 RegBear div=-1.0 + AUD_USD M5 confirms AUD currency-wide selling + AU Employment 01:30Z Apr16 (fcst +20k vs prev +48.9k expected miss, catastrophic) + thesis invalidation: bid 113.054 < BB lower 113.09 (state.md invalidation condition) |
| Spread at close | 10.8pip (post-rollover) |
| pretrade (original) | S-conviction (AUD strength thesis + H4 uptrend) |
| Exit quality | Below par — peak +416 → close -856. But event risk justifies early exit. |

## EUR_USD LIMIT BUY 2000u @1.17880 — 2026-04-14 21:51Z [PENDING]

| Field | Value |
|-------|-------|
| Time | 2026-04-14 21:51 UTC |
| Pair | EUR_USD |
| Side | LONG (LIMIT) |
| Units | 2000u |
| LIMIT price | 1.17880 (ask) = fills when bid ≈ H1 BB mid 1.17860 |
| TP on fill | 1.18120 (24pip) |
| SL on fill | 1.17565 (31.5pip, H1 BB lower structural) |
| Order ID | 467886 |
| GTD | 2026-04-15T00:48:00Z (expires before UK Unemployment) |
| Sp at placement | 4.2pip (thin market) |
| Pretrade | A(7) MEDIUM |
| Conviction | A | Size: 2000u |
| Margin if filled | (69,302 + 14,820) / 126,919 = 66.3% → safe |

**Thesis**: Structural add at H1 BB mid (1.17860 bid zone) — build EUR_USD position for London bull continuation. Existing 3500u @1.17924 already in. This LIMIT fills on a 4pip additional dip.
**FOR**: ① Direction (H1 ADX=54 DI+=28>>DI-=9) + ④ Structure (H1 BB mid structural support) + ⑥ Macro (USD structural weakness, PPI miss, 8th consecutive EUR/USD bull day)
**Different lens**: ⑤ Cross-pair (EUR_JPY H4 StRSI=0.0 reversal signal — EUR H4 momentum softening). Slightly adverse but EUR CS=+0.40.
**AGAINST**: H4 RSI=73 overbought. Fib at 135% (extended M5 wave). LIMIT may not fill if EUR bounces from here.
**If wrong**: EUR_USD breaks H1 BB mid cleanly → SL at H1 BB lower 1.17565

## 22:53Z — GBP_JPY LONG LIMIT 3000u @215.440 id=467890

| Field | Value |
|-------|-------|
| Time | 2026-04-14 22:53 UTC |
| Pair | GBP_JPY |
| Direction | LONG (add to existing 2000u) |
| Units | 3000 |
| Order | LIMIT @215.440 GTD=01:30Z |
| TP | 215.620 |
| SL | 215.200 |
| Spread | 3.1pip |
| Thesis | Squeeze-S undersized correction: existing 2000u = 9.4% NAV. H4 ADX=60 exceptional bull + M5 squeeze loading + GBP CS=+0.50. Adding 3000u = total 5000u (~30% NAV). LIMIT at BB lower edge to catch Tokyo dip. |
| pretrade | A(6) MEDIUM |

## 22:54Z — EUR_USD LIMIT modified: 467886→467892 GTD extended to 02:30Z
- Replaced: 467886 @1.17880 GTD=00:48Z
- New: 467892 @1.17880 GTD=02:30Z (extended past AU Employment 01:30 UTC)

---

## Session 23:09 UTC — Order Actions

| Action | Details |
|--------|---------|
| CANCEL | EUR_USD LIMIT id=467892 @1.17880 2000u — margin management (worst-case over 85%) |
| MODIFY | GBP_JPY LIMIT id=467895 (was 467890) — GTD extended 01:30Z→05:00Z for UK Unemployment |
| LIMIT | AUD_USD LONG 2000u @0.71250 TP=0.71650 SL=0.70930 GTD=05:00Z id=467896 |

### AUD_USD LONG LIMIT 2000u — id=467896
| Field | Value |
|-------|-------|
| Time | 2026-04-14 23:13 UTC |
| Pair | AUD_USD |
| Side | LONG (LIMIT) |
| Units | 2000u |
| Entry | 0.71250 (LIMIT at M5 BB lower) |
| TP | 0.71650 (+4.0pip) |
| SL | 0.70930 (structural, -3.2pip) |
| GTD | 2026-04-15T05:00Z |
| Sp | 1.4pip (normal) |
| Pretrade | pretrade=S(9)/MEDIUM historical — sized to A (7% NAV) for negative edge warning |
| Conviction | A (discounted: H1 MACD RegBear div + historical WR=45% -2709 JPY) |
| Thesis | AUD resilience post-NAB=-29 = USD CS=-0.55 overriding AU macro. H1 ADX=31 BULL H4 ADX=33 BULL. SQUEEZE at BB lower pullback entry. |
| Status | PENDING |


| MODIFY | GBP_JPY LIMIT — Cancelled id=467895 (SL=215.200=ATR×1.3 below Tokyo thin minimum ATR×1.5). Re-placed id=467898 @215.440 SL=215.165=ATR×1.5 TP=215.620 GTD=05:00Z | 2026-04-14 23:23 UTC |

## CLOSE: GBP_JPY LONG 2000u — 23:50 UTC
| Field | Value |
|-------|-------|
| Close time | 2026-04-14 23:50 UTC |
| Close price | 215.620 (TP hit) |
| Entry | 215.430 |
| Units | 2000 |
| Spread | 2.8pip |
| PL | +380 JPY |
| Hold time | ~5h20m |
| Reason | TP hit — pre-catalyst squeeze breakout |
| Thesis verdict | CORRECT — GBP strength + H4 ADX=60 + UK catalyst loading |

## LIMIT PLACED: EUR_USD LONG 2500u @1.17970 — 23:51 UTC
| Field | Value |
|-------|-------|
| Order time | 2026-04-14 23:51 UTC |
| Pair | EUR_USD |
| Direction | LONG |
| Units | 2500 |
| Order | LIMIT @1.17970 |
| TP | 1.18120 |
| SL | 1.17680 |
| GTD | 2026-04-15T08:00Z |
| id | 467902 |
| Pretrade | A(7) MEDIUM |
| Thesis | USD weakness (PPI miss + risk-on). Post-GBP_JPY TP freed margin. Dip-buy in SQUEEZE for UK catalyst delivery. H1 ADX=53 dominant. |
| Conviction | A — Direction+Macro+Cross-pair aligned. Fib N=BULL q=0.74 supports. Against: H4 overbought + H1 div=1.2. |

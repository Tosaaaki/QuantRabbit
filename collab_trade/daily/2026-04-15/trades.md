# Trades — 2026-04-15

## Open Positions (inherited from 2026-04-14)

| Time | Pair | Dir | Units | Entry | TP | SL | UPL | Status |
|------|------|-----|-------|-------|-----|-----|-----|--------|
| ~14/19:37Z | GBP_USD | LONG | 2000 | 1.35630 | 1.35822 | 1.35380 | +219 JPY | OPEN id=467833 |
| ~14/19:39Z | EUR_USD | LONG | 3500 | 1.17924 | 1.18120 | 1.17680 | ~+0 JPY | OPEN id=467869 |
| ~15/00:01Z | EUR_USD | LONG | 2500 | 1.17970 | 1.18120 | 1.17680 | -183 JPY | OPEN id=467903 (LIMIT fill) |

## Actions This Session (00:12 UTC)

| Time | Action | Details |
|------|--------|---------|
| 00:12 UTC | CANCEL | GBP_JPY LIMIT id=467898 @215.440 — naked LIMIT, Tokyo thin |
| 00:12 UTC | LIMIT | GBP_USD LONG 2000u @1.35640 TP=1.35820 SL=1.35380 GTD=05:00Z id=467907 |

## EUR_USD LIMIT Fill (retroactive log)
- LIMIT id=467902 placed last session @23:51 UTC at 1.17970
- Filled ~00:01 UTC as EUR_USD ask reached 1.17970
- New trade id=467903 | pretrade=A (conviction recorded in state.md 2026-04-14)

## Conviction Block — GBP_USD LONG 2000u @1.35640 id=467907
```
Thesis: M5 BB lower bounce within H1 TREND-BULL; USD weakness dip-buy pre-UK catalyst (06:00Z)
Regime: TREND-BULL — buy dips at BB lower
Type: Momentum (30min-2h; UK data resolves by 06:30Z)
Pair edge: 63% WR, +4,007 JPY total → supports
FOR: Direction(H1 ADX=51 DI+=31>>DI-=10) + Timing(M5 StRSI=0.0 BB lower) + Macro(GBP CS=+0.64 strongest)
Different lens: Fib N=BULL q=1.31 @Fib65% = wave 35% remaining → SUPPORTS
              Cross-pair: GBP_JPY H4 ADX=60 BULL = GBP-wide → SUPPORTS
AGAINST: H4 overbought (CCI=108 RSI=73) = extended. H1 MACD div=0.6 = deceleration.
If I'm wrong: UK unemployment ≥5.2%, GBP sells → SL 1.35380 (-16.0pip = -2,142 JPY on 2000u)
If nothing by: LIMIT expires 05:00Z — no fill = no cost
Margin after: 52.1% + ~15.9% → 68.0% total ✓ below 85%
→ Conviction: A | Size: 2000u (~15% NAV) | pretrade=S(8) but A-sized due to H4 overbought
```

## 01:52Z — LIMIT AUD_JPY LONG 1400u @113.14 id=467909
| Item | Value |
|------|-------|
| Time | 2026-04-15 01:52 UTC |
| Entry | 113.140 (LIMIT) |
| Spread | 1.6pip |
| TP | 113.280 |
| SL | 113.000 |
| GTD | 2026-04-15 05:00Z |
| Type | Range-Mean-Revert |
| Conviction | B (pretrade score=5, MEDIUM) |
| Thesis | Range-buy at BB lower 113.14 — 7 tested lower touches, H1 TREND-BULL (EMA slope +10pip), fib N=BEAR near exhaustion at Fib68%, Tokyo compression |
| FOR | Structure (7 BB lower touches) + Direction (H1 TREND-BULL) + Macro (China imports surge, Iran peace = risk-on AUD bid) |
| Different lens | fib BEAR micro-wave Fib68% → near exhaustion = timing aligns for bounce |
| AGAINST | H1 MACD RegBear div=-1.0 (developing); AUD business confidence -29 |
| Size | 1400u (5% NAV) |

---
## 02:36Z Session Actions

| Time | Action | Pair | Side | Units | Price | Note |
|------|--------|------|------|-------|-------|------|
| 02:36Z | CANCEL LIMIT | AUD_JPY | LONG | 1400 | 113.140 | id=467909 — upper band (113.27), 13pip away, TTL=2h40m. Zero fill probability. Margin cleanup. |

### Position Review (02:36Z)
- GBP_USD LONG 2000u @1.35630: UPL ~+6 JPY. H1 TREND-BULL intact. UK data 06:00Z = catalyst. HOLD.
- EUR_USD LONG 3500u @1.17924: UPL ~-468 JPY. H1 StRSI=0.00 + ADX=52 BULL dip. M5 SQUEEZE pending UP. HOLD. SL=1.17680 structural.
- EUR_USD LONG 2500u @1.17970: UPL ~-518 JPY. Same thesis. H1 chart: steep ascending EMA, current dip at H1 BB lower. HOLD.

### Pending LIMITs (active)
- GBP_USD LONG 2000u @1.35640 id=467907 GTD=05:00Z — 3.8pip below current
- GBP_JPY LONG 2000u @215.520 id=467908 GTD=05:00Z — 6pip below current

### Next Session
- 05:00Z: LIMITs expire → margin frees → USD_JPY SHORT + EUR_JPY RANGE BUY
- 06:00Z: UK Unemployment HIGH IMPACT — binary for GBP positions

---
## 03:48Z Session Actions

| Time | Action | Pair | Side | Units | Price | Note |
|------|--------|------|------|-------|-------|------|
| 03:48Z | CANCEL LIMIT | EUR_JPY | LONG | 1667 | 187.280 | id=467918 — needs 9pip drop in thin Tokyo market; fill probability low in 2h. Swapping for AUD_JPY. |
| 03:48Z | LIMIT | AUD_JPY | LONG | 1667 | 113.280 | id=467921 TP=113.550 SL=113.180 GTD=07:00Z — EMA20 micro-pullback in H1/M5 TREND-BULL ascending staircase. Pretrade A(7) 61%WR. |

### Conviction Block — AUD_JPY LONG LIMIT 1667u @113.28 id=467921
```
Thesis: AUD_JPY micro-pullback to M5 EMA20 (~113.28) in H1/M5 TREND-BULL ascending staircase. AUD CS=+0.18, JPY CS=-0.16.
Regime: TREND-BULL (H1 ADX=20, M5 ascending staircase visually confirmed) — dip buy at EMA20
Type: Scalp/Momentum (30min-2h)
Pair edge: 61% WR, +4,169 JPY total → SUPPORTS. Pretrade: A(7) MEDIUM.
FOR: Direction (H4 ADX=36 DI+=31>>18, M5 ascending green bodies EMA12>EMA20 rising) + Timing (M1 StRSI=0.02 extreme oversold, M5 at SQUEEZE base = entry zone) + Cross-pair (AUD CS +0.18 pos vs JPY -0.16 = divergence, Iran risk-on)
Different lens: Structure (Fib N=BULL q=1.39 = next wave high-quality, H1 ascending staircase Fib at pull-back zone) → SUPPORTS
AGAINST: Thin market (03:00Z). H1 ADX=20 (borderline range/trend).
If I'm wrong: 113.20 body close = EMA20 violated, micro-bull wave breaks. SL=113.180 structural.
If nothing by: GTD=07:00Z auto-cancels. London open movement may trigger.
Margin after fill: 75.4% (no pending LIMITs). Worst-case unchanged.
→ Conviction: B (thin market discount from pretrade A) | Size: 1,667u (5% NAV)
```

---
## ~02:27Z LIMIT Fill (detected at 02:36Z session)

| Time | Action | Pair | Side | Units | Price | TP | SL | id | Note |
|------|--------|------|------|-------|-------|----|----|-----|------|
---
## 04:09Z Session Actions

| Time | Action | Pair | Side | Units | Price | Note |
|------|--------|------|------|-------|-------|------|
| 04:09Z | LIMIT | EUR_JPY | LONG | 1667 | 187.350 | id=467924 TP=187.580 SL=187.180 GTD=05:30Z — H4 ADX=55 BULL StRSI=0.00 dip-buy. Pretrade B(4) Risk:LOW. |

### Conviction Block — EUR_JPY LONG LIMIT 1667u @187.350 id=467924
```
Thesis: H4 EUR_JPY ADX=55 DI+=30>>11 BULL with StRSI=0.00 reset = textbook dip-buy in strong trend. Iran ceasefire = risk-on = JPY weak. LIMIT 7pip below current for structural entry.
Regime: H4 STRONG BULL / M5 RANGE/SQUEEZE
Type: Momentum (30min-1h)
Pair edge: 67% WR, +3,245 JPY total (18W 9L) → supports
FOR: Direction (H4 ADX=55 DI+=30>>11 STRONGLY BULL) + Macro (Iran ceasefire risk-on = JPY weak, JPY CS -0.17) + Cross-pair (AUD_JPY H4 BULL = JPY weakness currency-wide)
Different lens: Structure (Fib103% of recent BEAR micro-wave = bear wave exhausted, price at M5 BB mid support 187.36) → SUPPORTS
AGAINST: H1 weak (ADX=13, no H1 confirmation). H4 RSI=73 overbought = extended from H4 perspective.
If I'm wrong: UK data surprise risk-off → JPY bid → below 187.18 structural invalidation.
If nothing by: GTD=05:30Z auto-cancels (before UK data).
Margin after fill: 69.3% + AUD_JPY(5.9%) + EUR_JPY(9.8%) = 85.0% worst case (at 85% boundary for B-LIMIT)
→ Conviction: B | Size: 1,667u (5% NAV) | pretrade=B(4) Risk:LOW
```

---
## ~02:27Z LIMIT Fill (detected at 02:36Z session)

| Time | Action | Pair | Side | Units | Price | TP | SL | id | Note |
|------|--------|------|------|-------|-------|----|----|-----|------|
| ~02:27Z | ENTRY (LIMIT FILL) | GBP_USD | LONG | 2000 | 1.35639 | 1.35820 | 1.35380 | 467911 | Dip-buy LIMIT @1.35640 filled. H1 BULL thesis. UK data 06:00Z. |

---
## 04:36Z Session Actions

| Time | Action | Pair | Side | Units | Price | Note |
|------|--------|------|------|-------|-------|------|
| 04:36Z | CANCEL LIMIT | EUR_JPY | LONG | 1667 | 187.350 | id=467924 — EUR_JPY making higher highs, fill impossible at 05:30Z expiry. Swapping for USD_JPY SHORT opportunity. |
| 04:36Z | LIMIT | USD_JPY | SHORT | 1667 | 159.050 | id=467926 TP=158.780 SL=159.220 GTD=05:45Z — H1 TREND-BEAR at 159.00 resistance ceiling, SQUEEZE, upper wicks rejected multiple times. pretrade=A(6) Risk:HIGH, sized B. |

### Conviction Block — USD_JPY SHORT LIMIT 1667u @159.050 id=467926
```
Thesis: USD_JPY at 159.00-159.07 H1 resistance ceiling, M5 SQUEEZE resolving DOWN. H1 TREND-BEAR. USD structurally weak (-0.23 G10). UK data 06:00Z is USD-negative catalyst. Multiple upper wicks rejected — can't close above ceiling. LIMIT at ceiling captures any re-test.
Regime: SQUEEZE (M5) / TREND-BEAR (H1)
Type: Momentum (30min-2h)
Pair edge: 33% WR, -21,841 JPY total — WARNING. Historical all-session avg. Regime filter not applied in history.
FOR: ① Direction: H1 TREND-BEAR (ADX=33 DI-=19>DI+=17) + ② Timing: M5 SQUEEZE with multiple upper wicks at 159.00-159.05 ceiling (bodies flat, can't close above) + ⑥ Macro: USD=-0.23 weakest G10, PPI miss, UK data USD-negative direction
Different lens: ④ Structure: M5 Fib BULL @Fib51% (micro-bounce within squeeze) — neutral. Not at extreme. Doesn't override H1 BEAR. Neutral.
AGAINST: Historical WR=33% (-21,841 JPY total). M5 making higher highs within squeeze (not confirmed down yet). Thin Tokyo pre-London. LIMIT may not fill if squeeze resolves down naturally without testing 159.05.
If I'm wrong: 159.12+ body close = ceiling broken bullishly. SL=159.220 fires. Loss ≈ 1667u × 17pip × 1/150 × 150 ≈ ~424 JPY = within 2% NAV.
If nothing by: GTD=05:45Z auto-cancel (15min before UK data). Zero cost if not filled.
Margin after: 69.4% + 8.3% → 77.7% | Worst case (AUD_JPY LIMIT fills too): 83.7% ✓
→ Conviction: B | Size: 1,667u (5% NAV) | LIMIT SHORT @159.050 TP=158.780 SL=159.220 GTD=05:45Z
```

### State after fill
- GBP_USD: NOW 4000u total (467833@1.35630 + 467911@1.35639), avg=1.35635
- EUR_USD: 6000u total (3500u@1.17924 + 2500u@1.17970)
- Margin: 69.6%
- Fib analysis: EUR_USD Fib184% N=BEAR (deep M5 correction), GBP_USD Fib220% N=BEAR (same), USD_JPY Fib94% N=BULL (bounce imminent — don't SHORT now), AUD_USD N=BULL good.

## AUD_USD LONG LIMIT 1667u — id=467927 [04:52Z]
| Field | Value |
|-------|-------|
| Time | 2026-04-15 04:52 UTC |
| Pair | AUD_USD |
| Side | LONG (LIMIT) |
| Units | 1667u |
| LIMIT price | 0.71285 |
| TP | 0.71445 |
| SL | 0.71185 |
| GTD | 2026-04-15 05:45Z |
| Spread | 1.4pip |
| Pretrade | B(7)/MEDIUM |
| Thesis | AUD ascending staircase SQUEEZE breakout. AUD_JPY also ascending (currency-wide AUD bid). USD CS=-0.23 structural weak (PPI miss). LIMIT @BB mid = catch pre-London dip. |
| Conviction | B |
| Regime | SQUEEZE(M5) / TREND-BULL(H4 ADX=35 DI+=35>>17) |
| FOR | H4 ADX=35 DI+ dominant + AUD_JPY aligned (cross-pair) + M5 ascending accumulation |
| AGAINST | 45% WR historical / UK binary risk 74min / AUD_JPY HidBear H1 div |
| IF WRONG | UK miss → risk-off → below Fib 0.71230. SL structural at 0.71185 |
| IF NOTHING | GTD 05:45Z expires. Zero cost. |

---

## GBP_JPY LONG LIMIT 2000u — id=467933 [06:14Z]
| Field | Value |
|-------|-------|
| Time | 2026-04-15 06:14 UTC |
| Pair | GBP_JPY |
| Side | LONG (LIMIT) |
| Units | 2000u |
| LIMIT price | 215.600 |
| TP | 216.450 |
| SL | 214.900 |
| GTD | 2026-04-15 10:05Z |
| Spread | 3.1pip (normal) |
| Pretrade | S(8)/MEDIUM |
| Thesis | M5 pullback to BB lower (215.60) in H4 MONSTER bull (ADX=61 DI+=39). GBP CS=+0.63 strongest, JPY CS=-0.20. London open (08:00Z) = catalyst. |
| Conviction | A (pretrade S but H4 RSI=77 overbought concern + margin constraint → A-size) |
| Regime | H4 TREND-BULL (ADX=61) / M5 RANGE-SQUEEZE pullback |
| FOR | H4 ADX=61 DI+=39>>10 MONSTER + H1 ADX=40 DI+=25>>9  / CS GBP(+0.63) vs JPY(-0.20) gap=0.83 / Risk-on Iran deal + UK data neutral = GBP intact |
| AGAINST | H4 RSI=77 overbought. H1 MACD_H fading. Yesterday SL hit at 214.90. |
| IF WRONG | BOJ hawkish or Iran collapse → JPY spike. Below 215.00 = H4 change. |
| IF NOTHING | 08:30Z → cancel if price never dipped OR above 216.00 (missed opportunity) |
| Margin after | 82.3% (within 85%) — will expand to full A/S size after EUR_USD cut clause fires |

---

## 06:19Z — EUR_USD CLOSE ×2 (Cut Clause Triggered)

| Time | Pair | Side | Units | Entry | Close | PL | id | Reason |
|------|------|------|-------|-------|-------|-----|-----|--------|
| 06:19Z | EUR_USD | LONG | 2500 | 1.17970 | 1.17890 | -318.54 JPY | 467903 | 06:30Z cut clause: bid=1.17910 < 1.1795. Session kill at 06:29Z = cannot wait. Executing 11min early. |
| 06:19Z | EUR_USD | LONG | 3500 | 1.17924 | 1.17890 | -189.53 JPY | 467869 | Same cut clause. NY orphan 11h. H4 BULL intact but discipline over hope. |

**Total EUR_USD P&L today: -508.07 JPY**
**Lesson: Session timing constraint (kill at 06:29Z) forced early execution of 06:30Z cut clause. Both conditions valid: bid MET threshold, preclose confirmed H4 intact but NY orphan discipline applies.**

---

## 06:22Z — GBP_JPY LONG LIMIT 3000u @215.550 id=467945 (Range-bounce add-on)

| Field | Value |
|-------|-------|
| Time | 2026-04-15 06:22 UTC |
| Pair | GBP_JPY |
| Side | LONG (LIMIT) |
| Units | 3000u |
| LIMIT price | 215.550 |
| TP | 216.450 |
| SL | 214.900 |
| GTD | 2026-04-15 10:05Z |
| Conviction | A (add-on to id=467933 @215.60. Combined 5000u = A-size after EUR_USD margin freed) |
| Thesis | Range-bounce add-on after EUR_USD cut clause freed ~42k JPY margin. H4 ADX=61 MONSTER BULL. GBP CS=+0.63. 8 upper band touches vs 3 lower. M5 buyers leaning, making higher highs. |
| Margin after if both fill | ~49% NAV — well within limits |

## 06:43Z Actions

| Time | Action | Details |
|------|--------|---------|
| 06:43Z | MODIFY | GBP_JPY id=467934 TP 216.450→215.770 (M5 BB upper structural, ATR×1.0 from entry) |
| 06:43Z | LIMIT | EUR_USD LONG 2000u @1.17870 TP=1.17960 SL=1.17830 GTD=08:40Z id=467952 |

## Conviction Block — EUR_USD LONG LIMIT 2000u @1.17870 id=467952
```
Thesis: A-conv LIMIT: M5 extreme oversold (StRSI=0.0) at BB lower in H4 BULL ADX=50. Fib 98% pullback = re-entry at wave base. Spring coiled.
Regime: SQUEEZE/TREND-BULL — buy at BB lower on M5 oversold
Type: Scalp/Momentum (30min-2h via London open)
Pair edge: 56% WR (49W/36L), +18,077 JPY total → supports
FOR: Direction(H4 ADX=50 DI+=37, H1 ADX=43 DI+=26) + Timing(M5 StRSI=0.0 extreme oversold, BB lower) + Macro(USD weak -0.37, Iran deal optimism, soft PPI)
Different lens: Structure(Fib 98% pullback → re-entry at M5 bull wave base, H=1.17993 L=1.17893 → entry @1.17870 = below wave base = deeper re-entry) → SUPPORTS
AGAINST: H4 RSI=71 overbought (extended rally). M5 sellers still leaning bearish (bodies growing). All positions LONG (concentration).
If I'm wrong: USD bounces on Iran talks collapse → below 1.17830 SL (-4pip = -536 JPY on 2000u)
If nothing by: LIMIT expires 08:40Z — re-evaluate at London open. If not filled, cancel.
Margin after: 51% + 10% EUR_USD + 20% GBP_JPY LIMIT worst case = 81% ✓
→ Conviction: A | Size: 2000u (10% NAV margin) | pretrade=A(6) MEDIUM
```

## 06:45Z — EUR_USD LIMIT FILLED

| Field | Value |
|-------|-------|
| Time | 2026-04-15 06:45 UTC |
| Order id | 467952 → Trade id 467953 |
| Entry | 1.17869 (LIMIT was 1.17870, filled 0.1pip better) |
| Direction | LONG 2000u |
| TP | 1.17960 (9.1pip) |
| SL | 1.17830 (3.9pip) |
| Spread | 0.8pip |
| Conviction | A (pretrade score=6 MEDIUM) |

## 07:27Z — EUR_JPY LONG 2500u ENTRY + GBP_USD 467833 CLOSE

### GBP_USD CLOSE — id=467833 (zombie 15h18m)
| Field | Value |
|-------|-------|
| Close | 1.35624 |
| P/L | -19 JPY |
| Reason | Zombie (15h18m), closed to free margin for EUR_JPY S-entry |
| pretrade | B(4)LOW (original entry) |

### EUR_JPY LONG 2500u — id=467968
| Field | Value |
|-------|-------|
| Time | 2026-04-15 07:27Z |
| Entry | 187.344 (market) |
| TP | 187.470 (H1 BB upper -0.4pip spread, +12.6pip) |
| SL | 187.220 (below H1 BB lower 187.250, -12.4pip) |
| Spread | 1.9pip |
| pretrade | B(4) LOW |
| S-scanner | Structural-S [proven 3/3] fired |
| Conviction | A |
| Size | 2500u (15% NAV) |
| Thesis | M5 StRSI=0.00 CCI=-201 at BB lower bounce within H4 ADX=55 DI+=30 bull. H1 SQUEEZE at support. |
| Invalidation | Close below H1 BB lower 187.250 (SL handles) |
| If nothing by | 07:57Z (30min) — close if no bounce to 187.40+ |

## EUR_JPY CLOSE id=467968 — 07:34Z
| Field | Value |
|-------|-------|
| Close | 187.350 |
| P/L | +15 JPY |
| Reason | M5 fib BEAR q=1.95 at Fib52% — bear continuation expected, not bouncing as thesis required |
| Note | 07:57Z check condition confirmed early by fib wave. Freed margin for AUD_JPY LIMIT |

## AUD_JPY LIMIT PLACED id=467979 — 07:35Z
| Field | Value |
|-------|-------|
| Entry | LIMIT @113.420 |
| TP | 113.600 |
| SL | 113.260 |
| GTD | 11:30Z |
| Units | 3000u |
| Conviction | A |
| pretrade | B(4) MEDIUM |
| Thesis | Ascending staircase dip-buy. Cleanest chart on board. Fib@20% pullback zone. Iran risk-on AUD bid. |

---
## 08:13Z — GBP_JPY SL TRIGGERED (both positions)

| Time | Pair | Side | Units | Entry | Close | PL | id | Reason |
|------|------|------|-------|-------|-------|-----|-----|--------|
| ~08:13Z | GBP_JPY | LONG | 3000 | 215.550 | ~215.320 | ~-620 JPY | 467956 | SL auto-triggered. Auditor correct: M5 TREND-BEAR, not bounce |
| ~08:13Z | GBP_JPY | LONG | 2000 | 215.600 | ~215.320 | ~-620 JPY | 467934 | SL auto-triggered. Same thesis, same result |

**GBP_JPY total today: -1,240 JPY (2 trades)**
**Lesson: Lower wicks in M5 TREND-BEAR staircase ≠ reversal signal. Audit was right. State.md was wrong.**

---
## 08:13Z — AUD_JPY LIMIT FILLED id=467984

| Field | Value |
|-------|-------|
| Time | ~08:13 UTC |
| Order | LIMIT id=467979 filled → Trade id=467984 |
| Pair | AUD_JPY |
| Side | LONG |
| Units | 3000u |
| Entry | 113.420 |
| TP | 113.600 |
| SL | 113.260 |
| Spread | 1.6pip |
| Conviction | A (Structural-S scanner proven 3/3) |
| Thesis | M5 at BB lower + StRSI=0.00 + H4 ADX=37 BULL. AUD strongest, JPY weak. Risk-on Iran deal. |

---
## 08:23Z — USD_JPY SHORT LIMIT 1000u @158.980 id=467987

```
Thesis: USD_JPY RANGE 158.83-159.02. LIMIT SHORT at range upper for direction balance and Iran-collapse hedge.
Regime: RANGE (M5 ADX=16, mixed bodies mid-range)
Type: Counter/Range-Mean-Revert
FOR: Structure (BB upper 159.00-159.02 = range ceiling, multiple rejections) + Macro (USD=-0.29 weakest = USD_JPY structurally capped) + Direction balance (all other positions LONG)
Different lens: USD_JPY SHORT profits in BOTH risk-off (JPY safe-haven bid) AND ongoing USD weakness. Dual-scenario hedge.
AGAINST: Current price 158.80 — needs 18pip rally to fill. Might not fill before GTD=12:00Z.
If wrong: 159.12 body close = range broken up. SL fires. Loss ≈ -234 JPY.
Conviction: B | Size: 1000u (~5% NAV) | TP=158.830 (+15pip) SL=159.120 (-14pip)
```

## CLOSE — AUD_JPY LONG 3000u id=467984 [08:47Z]
| Field | Value |
|-------|-------|
| Time | 2026-04-15 08:47 UTC |
| Pair | AUD_JPY |
| Direction | LONG |
| Units | 3000u |
| Entry | 113.420 |
| Exit | 113.384 |
| P/L | -108 JPY |
| Reason | M5 N-wave BULL invalidated (bid 113.386 < A point 113.408). H1 StRSI=1.0 extreme overbought headwind. Bounce from M5 BB lower failed — price fell further after entry. C(3) test: would not enter LONG at current level with H1 overbought. |
| Lesson | H1 StRSI=1.0 overbought + M5 bounce attempt = correction-within-overbought pattern. Recipe fires correctly at entry but H1 overbought overrides M5 oversold signal. |

## AUD_USD LONG 4000u — id=467999 [09:01Z]
| Field | Value |
|-------|-------|
| Time | 2026-04-15 09:01 UTC |
| Pair | AUD_USD |
| Side | LONG |
| Units | 4000u |
| Entry | 0.71466 (market, ask) |
| Spread | 1.4pip |
| TP | 0.71650 (+18.4pip) |
| SL | 0.71370 (-9.6pip) |
| Conviction | A (Rule4: AUD_USD neg edge caps at A) |
| pretrade | S(8) MEDIUM |
| Thesis | Structural-S dip buy: M5 BB lower StRSI=0.0 within H4 ADX=35 DI+=34 BULL + H1 ADX=31 BULL. EU Ind.Prod beat -> USD weakness confirmed. AUD CS=+0.38 strongest. Risk-on Iran optimism. |
| FOR | Direction(H4+H1 ADX bull) + Timing(M5 StRSI=0.0 at BB lower) + Macro(AUD strongest+USD weakest+EU beat) |
| Different lens | Structure: Fib range L=0.71404, price above + Ichi M5 0pip above cloud → SUPPORTS |
| AGAINST | H1 StRSI=1.0 overbought + MACD div bear(0.6). AUD_USD 45% WR historically. |
| If wrong | H1 distributes, M5 breaks range low 0.71404 → 0.7110-0.7120 |
| If nothing by | 09:30Z → close at market |

## 09:59Z — EUR_USD LONG 2000u @1.17820 id=468024

| Item | Value |
|------|-------|
| Time | 2026-04-15 09:59 UTC |
| Entry | 1.17820 (MARKET) |
| Spread | 0.8pip |
| TP | 1.17950 |
| SL | 1.17700 |
| Units | 2000 |
| Type | Range-Mean-Revert |
| Conviction | A (pretrade score=6) |
| Thesis | H4 ADX=50 DI+=35 BULL + M5 BB lower range bounce. H1 SQUEEZE breakout UP expected. AUD/GBP strongest. |

```
FOR: Direction(H4 ADX=50 DI+=35 BULL) + Structure(M5 BB lower = range support, 10+ touches) + Macro(USD CS=-0.38 weakest, Iran peace risk-on)
Different lens: Momentum(M5 sellers still dominant) → slight AGAINST but lower wicks = buyers defending
AGAINST: M5 sellers still present, could push slightly lower
If I'm wrong: Break below 1.177 = BB lower breakdown → SL at 1.17700
Theme confidence: proving (today 4 losses, no TP yet) → B-size 2000u
Margin after: 12% (EUR_USD 2000u) + 5% (USD_JPY LIMIT) + 7% (AUD_JPY LIMIT) = 24% worst case ✓
→ Conviction: A | Size: 2000u (B-size cap due to theme=proving)
```

## 09:59Z — CLOSE EUR_JPY SHORT 2000u @187.210 id=468007

| Item | Value |
|------|-------|
| Close time | 2026-04-15 09:59 UTC |
| Close price | 187.210 |
| Entry | 187.125 |
| P/L | -170 JPY |
| Reason | Counter-S H4+H1 StRSI=0.00 oversold recovery (proven 4/5). M5 momentum shifted from staircase to contested/buyers. "If entered NOW?" check: would NOT re-enter SHORT given H4 oversold bull context. |

## LIMIT GBP_USD LONG — id=468034 [10:32Z]
| Item | Value |
|------|-------|
| Type | LIMIT |
| Pair | GBP_USD |
| Side | LONG |
| Units | 1200u |
| Entry | 1.35150 |
| TP | 1.35665 (+51.5pip) |
| SL | 1.34900 (-25pip) |
| GTD | 2026-04-15T16:30Z |
| Spread | 1.3pip |
| Pretrade | S (score=9, MEDIUM risk) |

Thesis: H4 ADX=50 DI+=38 mega bull + H1 StRSI=0.00 extreme oversold + dip to H1 BB lower (~1.355). 
  H1 chart shows clear ascending uptrend from 1.272→1.360, pullback to H1 BB lower = structural dip-buy.
  Entry @1.3515 = wick fill below H1 BB lower. Theme: USD weakness (CS=-0.21).
FOR: ① H4+H1 direction aligned (ADX=50/40, DI+ dominant) ② H1 StRSI=0.00 extreme oversold (Timing) ⑤ USD CS=-0.21 weakest (cross-pair)
Different lens: Fib N=BEAR(q=0.53) on M5 → weak bear wave signal. Contradicts but low quality (0.53).
AGAINST: M5 staircase still intact (no lower wicks defending yet → LIMIT not market), Fib M5 N=BEAR, churn flag
If wrong: Breaks below 1.350, SQUEEZE resolves downward on GBP weakness
Zombie: 14:30Z
Conviction: A (pretrade=S, sized B-max 1200u due to theme=proving + max_loss_rule 476 JPY)
Max loss: 1200u × 25pip × (0.0001 × 158.98) = ~476 JPY

## 12:35Z — Cancel LIMITs (soft catalyst)
- CANCEL AUD_JPY LIMIT id=468033 @113.330 | reason: soft catalyst = risk-on, AUD_JPY going up not dipping to 113.33
- CANCEL GBP_USD LIMIT id=468048 @1.35400 | reason: soft catalyst = GBP_USD rallying not dipping to LIMIT

## 12:37Z — ENTRY GBP_JPY LONG
| Field | Value |
|-------|-------|
| Pair | GBP_JPY |
| Side | LONG |
| Units | 1500u |
| Entry | 215.637 |
| TP | 215.900 (+26.3pip) |
| SL | 215.330 (structural swing low, -30.7pip) |
| Trade ID | 468052 |
| Spread | 3.1pip |
| pretrade | A (MEDIUM, score=7) |
| Conviction | A |

Thesis: Import Prices catalyst confirmed soft (EUR_USD +5pip post-data, USD_JPY down). GBP strongest currency (H4=+35, M1=+4). JPY offered across all TFs. H4 ADX=59 DI+=35 StRSI=0.09 = MEGA BULL reset at H4 floor = fresh bull wave starting.
Type: Momentum (hold 30min-2h, TP=ATR×2.0)
FOR: Direction (H4 ADX=59 DI+=35 mega bull) + Cross-pair (GBP strongest H4/M15/M1, JPY offered all TFs) + Macro (soft catalyst confirmed, risk-on, Iran diplomacy)
Different lens: Structure (H4 StRSI=0.09 = floor reset = early wave, room to run) → SUPPORTS
AGAINST: M15 StRSI=1.00 (overbought short-term) | Spread 3.1pip (above normal 2.5-3.2pip range)
If wrong: Price breaks below 215.33 swing low = buyers gave up. SL fires at -461 JPY (1500u × 30.7pip × 0.01)
Margin: ~82% after entry (no pending LIMITs) ← below 85% ✓

## 12:53Z — EUR_USD 10000u CLOSE

| Item | Value |
|------|-------|
| Time | 2026-04-15 12:53 UTC |
| Pair | EUR_USD |
| Side | LONG |
| Units | -10000u |
| Close price | 1.17852 |
| P/L | +539.8 JPY |
| Reason | H1 DI- dominant (THESIS ADVERSE). M5 reversed from band-walk to sellers accelerating. USD M1 BID 4/4. Preclose confirmed structure change. |

## 12:53Z — LIMIT PLACED: EUR_JPY LONG 2000u @187.20

| Item | Value |
|------|-------|
| Order id | 468059 |
| Type | LIMIT GTD 8h |
| Entry | 187.200 |
| TP | 187.550 |
| SL | 186.950 |
| Conviction | A |
| Thesis | H4 ADX=54 StRSI=0.00 extreme floor reset in H4 bull. H1 Fib@89% bear exhausting near H1 BB lower (187.187). |

## 12:53Z — LIMIT PLACED: GBP_USD LONG 2000u @1.354

| Item | Value |
|------|-------|
| Order id | 468060 |
| Type | LIMIT GTD 8h |
| Entry | 1.35400 |
| TP | 1.35800 |
| SL | 1.35000 |
| Conviction | B |
| Thesis | RANGE lower bound play. M5 BB lower structural + chart lower wicks (buyers absorbing). GBP H4 strongest. |

---
## 13:32Z — GBP_JPY LONG 2000u (Add-on to 1500u position)

| Field | Value |
|-------|-------|
| Time | 2026-04-15 13:32 UTC |
| Pair | GBP_JPY |
| Direction | LONG |
| Units | 2000u (add-on to id=468052 1500u @215.637) |
| Entry | @215.678 (ask) |
| Trade ID | 468065 |
| Spread | 2.8pip |
| TP | 215.900 |
| SL | 215.500 |
| pretrade | A(7) / MEDIUM |
| Conviction | B (theme=proving, no TP confirmed today) |
| Thesis | TREND-BULL add-on: M1 StRSI=0.0 micro-bottom in M5 band walk. H4 StRSI=0.25 BOTTOM BULL (fresh wave). GBP strongest (+30), JPY weakest (-16). M5 bodies growing, accelerating. |
| AGAINST | M15 StRSI=1.0 overbought, H4 MACD_H declining (H4 deceleration) |
| If wrong | Price fails to reach 215.90, reverses below 215.50 (M5 BB lower) |

## 13:32Z — EUR_JPY LIMIT PLACED

| Field | Value |
|-------|-------|
| Order ID | 468068 |
| Pair | EUR_JPY |
| Direction | LONG |
| Units | 2000u |
| LIMIT Price | 187.350 |
| GTD | 2026-04-15T21:30Z |
| TP | 187.600 |
| SL | 187.150 |
| Reason | H1 StRSI=1.0 (above H1 BB upper) + H1 ADX=13 (weak) = pullback to M5 BB lower (187.357) / H1 BB mid (187.341) confluence. Replaced stale LIMIT @187.20 (34pip too far) |

## 13:?Z — EUR_JPY LIMIT CANCELLED (468059 @187.200)
- Cancelled because: 34pip below current price in TREND-BULL, H1 overbought + H1 ADX=13 = pair trending away but volatile, not structural at that level. Replaced with @187.350.

---
## 13:28Z — EUR_USD LONG CLOSED (TP auto-fire)

| Field | Value |
|-------|-------|
| Time | 2026-04-15 13:28 UTC |
| Pair | EUR_USD |
| Direction | CLOSE LONG |
| Units | 2000u |
| Entry | @1.17820 (13:28Z - 3h30m held) |
| Exit | @1.17951 (TP auto-fire) |
| P&L | **+415.76 JPY** |
| Held | ~3h30m |
| Reason | TP hit structural resistance 1.17950. ATR×1.0 target achieved. |
| Note | Theme CONFIRMED — first TP today. Theme confidence upgrading from proving → confirmed. |

| 13:54 | ENTRY | EUR_USD | LONG | 3000u | @1.17858 | TP=1.18000 | SL=1.17700 | id=468070 | pretrade=A(7)/MEDIUM | M5 StRSI=0.0 dip in confirmed H4 TREND-BULL, USD resuming sell |

---
## 18:45Z — GBP_JPY LONG CLOSED (18:50Z trigger executed 5min early)

| Field | Value |
|-------|-------|
| Time | 2026-04-15 18:45 UTC |
| Pair | GBP_JPY |
| Direction | CLOSE LONG |
| Units | 750u (reduced from original 2000u via 13:?Z SL hit and partial closes) |
| Entry | 215.637 |
| Exit | 215.829 |
| P&L | **+144 JPY** |
| Held | ~6h08m |
| Reason | 18:50Z pre-Taylor trigger executed 5min early. AT swing high resistance (0.7pip per profit_check). M5 bodies compressing. Taylor @19:00Z in 15min = binary event risk. EV(close now +143) > EV(wait 5min ~+124 expected). |
| pretrade | B (inherited from session earlier) |
| Note | TP=215.900 was 6.2pip away. Sound decision to lock +144 JPY rather than risk Taylor event |

## EUR_JPY LONG entry — 2026-04-15 19:47Z
| Field | Value |
|-------|-------|
| Trade ID | 468092 |
| Pair | EUR_JPY |
| Side | LONG |
| Units | 4,000 |
| Entry | 187.619 (market ask) |
| TP | 187.720 |
| SL | 187.510 |
| Spread | 2.1pip |
| Pretrade | A conviction, LOW risk (score=6, 68% WR, avg+116 JPY) |
| Conviction | A (theme confirmed, H4 early wave) |
| Type | Scalp |
| Zombie | 20:17Z |
| Thesis | M5 at BB lower (7.0pip) with StRSI=0.0. Post-Lagarde EUR bid structural support. H4 StRSI=0.16 = early wave, ADX=54 BULL = room to run. Structural-S proven recipe. |
| Risk | (187.619-187.510) × 4000 = 436 JPY |
| Reward | (187.720-187.619) × 4000 = 404 JPY |


## 2026-04-15 ~20:05-20:17 UTC Session

### EUR_JPY LONG 4000u @187.619 → CLOSE @187.564 -220 JPY
- Entry: 19:48Z (prev session). Structural-S recipe. H4 ADX=54 StRSI=0.16 early bull + M5 BB lower StRSI=0.0.
- Close: 20:11Z. Zombie rule (scalp 24min vs 15-30min expected). C(4)=NO (wouldn't enter at 187.579 ask with EMA12 declining, EUR M1=offered-28). Price broke below post-Lagarde range low 187.59.
- Reason: zombie_scalp_past_expected_hold
- pretrade: Structural-S [proven 3/3]

### EUR_USD LIMIT modified
- Cancelled: id=468083 GTD=21:00Z
- Re-placed: id=468096 @1.17900 2000u TP=1.18150 SL=1.17750 GTD=22:00Z
- Reason: extend GTD. Structural level = Fib50% H1 correction.

### GBP_USD LIMIT placed
- id=468097 @1.35630 1667u TP=1.35720 SL=1.35550 GTD=23:00Z
- B-conviction: RANGE lower + H1 Fib38.2-50% correction zone. H4 ADX=53 MID BULL.

---
## ~20:30Z — GBP_USD LIMIT FILL (logged retroactively 21:05Z)

| Time | Action | Pair | Side | Units | Price | TP | SL | id | Note |
|------|--------|------|------|-------|-------|----|----|-----|------|
| ~20:30Z | LIMIT_FILL | GBP_USD | LONG | 1667 | 1.35629 | 1.35720 | 1.35550(rollover_removed) | 468102 | LIMIT id=468097 @1.35630 filled. SL removed by rollover_guard prior to 21:00Z maintenance |

### Context
- LIMIT placed at 20:11Z session: id=468097 @1.35630 1667u GTD=23:00Z
- Fill: ~20:30Z @1.35629 (1pip improvement) → trade id=468102
- Rollover guard removed SL=1.35550 before 21:00Z spread spike (saved for restore)
- Post-rollover: restore SL=1.35550, HOLD for TP=1.35720 (range bounce thesis)
- Pretrade: B-conviction (range lower bounce, Fib38%, M15 StRSI=0.00)

## Actions 21:25-21:35 UTC (Late NY rollover session)

| Time | Action | Details |
|------|--------|---------|
| 21:29 UTC | CANCEL | EUR_USD LIMIT id=468096 @1.17900 — below range floor (1.1793), wrong structural level |
| 21:29 UTC | LIMIT | EUR_USD LONG 2000u @1.17940 TP=1.18030 SL=1.17890 GTD=23:00Z id=468108 — range floor 7x tested, B-size Tokyo thin |

### GBP_USD Position (held through rollover)
- id=468102 LONG 1667u @1.35629 | UPL=-111 JPY | TP=1.35720 | SL=REMOVED (rollover_guard active, saved=1.35550)
- Rollover window: spreads still 23x normal at 21:29Z (GBP_USD 10.2pip). No action taken.
- Zombie: 22:30Z → close if TP not hit

## 22:25Z — GBP_USD CLOSE (zombie)

| Item | Value |
|------|-------|
| Time | 2026-04-15 22:25 UTC |
| Close | @1.35633 |
| Units | 1667u |
| P/L | +10.57 JPY |
| Reason | Zombie 22:30Z — held 1h57m (expected <1h, ratio 2.0×). +8 JPY breakeven, no catalyst for TP=1.35720 in Late NY thin market. M5 SQUEEZE unresolved, H1 N=BEAR(q=1.58) structural warning. Dead money. |
| Spread | 1.3pip (normalized post-rollover) |
| Pretrade | B (range bounce) |
| Conviction | C → A (hold justified during rollover constraint only) |

Close reason: zombie — expected hold exceeded, M5 SQUEEZE not breaking toward TP, H1 next N-wave bearish. Locking +10.57 JPY vs risking breakeven deterioration in thin Late NY.

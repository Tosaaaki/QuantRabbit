# Trades — 2026-04-10

## Closed

| Time(UTC) | Pair | Dir | Units | Entry | Exit | P&L | Notes |
|-----------|------|-----|-------|-------|------|-----|-------|
| ~14:14 | AUD_JPY | LONG | 7000u | 112.634 | 112.793 (TP) | +1,120 JPY | S-Momentum |
| ~14:24 | AUD_JPY | LONG | 3000u | 112.578 | 112.800 (TP) | +666 JPY | S-Momentum |

**Today confirmed**: +1,786 JPY

## Open

| Open | Pair | Dir | Units | Entry | TP | SL | UPL | pretrade |
|------|------|-----|-------|-------|-----|-----|-----|----------|
| ~14:00? | GBP_JPY | LONG | 2000u | 214.263 | 215.000 | 213.800 | +142 JPY | S |
| ~14:20? | GBP_JPY | LONG | 3000u | 214.393 | 215.000 | 213.800 | -174 JPY | S |

## Pending LIMITs

| Placed | Pair | Dir | Units | LIMIT | TP | SL | GTD | id |
|--------|------|-----|-------|-------|-----|-----|-----|----|
| 14:23 | EUR_JPY | LONG | 3000u | 186.500 | 187.100 | 185.900 | 18:23Z | 467410 |
| 14:41 | EUR_USD | LONG | 2000u | 1.17200 | 1.17550 | 1.17050 | 17:30Z | 467411 |

## 14:52Z — GBP_USD LONG 2500u

| Field | Value |
|-------|-------|
| Time | 2026-04-10 14:52Z |
| Pair | GBP_USD |
| Side | LONG |
| Units | 2500u |
| Entry | 1.34601 |
| TP | 1.34770 |
| SL | 1.34530 |
| id | 467414 |
| pretrade | A(7) MEDIUM |
| Spread | 1.3pip |

**Thesis**: M5 StRSI=0.0 extreme oversold at M5 BB lower (1.346). H4 ADX=42 strong BULL. GBP CS+0.78 (strongest) vs USD CS-0.24. Currency strength gap=1.02. Dip buy in H4 uptrend.

**FOR**: H4 ADX=42 BULL DI+=35 (Direction) + M5 StRSI=0.0 BB lower (Timing) + CS gap=1.02 (Cross-pair)
**Different lens**: Structure — Fib at 87% retracement = near wave low structural support → supports
**AGAINST**: H1 RSI+MACD dual bear div (0.6). USD_JPY SQUEEZE could break UP → USD bid.
**If wrong**: USD_JPY squeezes UP → GBP_USD falls through 1.345 → SL=1.34530 triggers, -7.9pip = -314 JPY

**EUR_JPY LIMIT 467410 CANCELLED**: H4+H1 StRSI=1.0 overbought multi-TF, 17pip below market, stale.

---

## GBP_USD TP FILL — 2026-04-10 ~15:22Z

| Field | Value |
|-------|-------|
| Close | TP hit @1.34770 |
| Entry | 1.34601 |
| Units | 2500u |
| P&L | ~+670 JPY |
| Reason | TP_fill (+16.9pip) |
| id | 467414 |

---

## EUR_USD LIMIT MODIFY — 2026-04-10 15:29Z

| Field | Value |
|-------|-------|
| Action | Cancelled 467411 (2000u GTD17:30), placed 467426 (3000u GTD19:00) |
| Price | @1.17200 |
| TP | 1.17550 (+35pip) |
| SL | 1.17050 (-15pip) |
| pretrade | B(5) MEDIUM (at market price; LIMIT at wave low = structural) |
| Reason | Size increase: EUR_USD best pair (+9k total), 5 consec wins at MEDIUM, structural entry at M5 BB lower/wave low |


## AUD_JPY LIMIT LONG 4000u — 2026-04-10 16:43 UTC

| Item | Value |
|------|-------|
| Type | LIMIT |
| id | 467434 |
| Price | 112.560 |
| TP | 113.100 (+54pip) |
| SL | 112.250 (-31pip) |
| GTD | 19:00 UTC |
| Spread | 1.6pip |
| Conviction | A |
| pretrade | S(8) MEDIUM (H4 overbought flag) |
| Thesis | H1 ADX=41 BULL + M5 StRSI=0.0 oversold SQUEEZE + Fib N=BULL q=0.97. Pullback from 113.7 to 112.56 = 1 H1 ATR correction. Next wave BULL. |
| AGAINST | H1 MACD div=0.6 (bear = trend fatigue). Iran risk (AUD tail). |
| Status | **CANCELLED 17:27Z** — AUD_USD TREND-BEAR + Friday weekend risk + Iran tail risk |

---

## EUR_USD LIMIT FILL — 2026-04-10 ~15:31-16:39Z (retroactive record)

| Field | Value |
|-------|-------|
| Action | LIMIT_FILL (id=467426 → trade id=467427) |
| Pair | EUR_USD |
| Side | LONG |
| Units | 3000u |
| Entry | 1.17200 |
| TP | 1.17346 (modified 16:39Z from 1.17550) |
| SL | 1.17050 |
| Spread | 0.8pip |
| Status | **OPEN** |
| Note | Auto-filled LIMIT. TP reduced 16:39Z from 1.17550 → 1.17346 (H1 swing high structural, ATR×1.3) |

**Thesis**: EUR strongest (CS+0.44) vs USD weakest (UMich miss confirmed USD structural weakness). H4 ADX=44 BULL. LIMIT at M5 wave low / BB mid. Cross-pair: EUR_JPY also H4 ADX=58 BULL. AUD_USD TREND-BEAR = not broad USD strength = EUR/GBP carry intact.

---

## Open Positions — 17:42 UTC

| Open | Pair | Dir | Units | Entry | TP | SL | UPL | id |
|------|------|-----|-------|-------|-----|-----|-----|-----|
| 13:11Z | GBP_JPY | LONG | 2000u | 214.263 | 214.466 | 213.800 | +126 JPY | 467385 |
| 14:13Z | GBP_JPY | LONG | 3000u | 214.393 | 214.466 | 213.800 | -201 JPY | 467403 |
| ~15:31Z | EUR_USD | LONG | 3000u | 1.17200 | 1.17346 | 1.17050 | +67 JPY | 467427 |

**Today confirmed P&L**: +1,985 JPY (OANDA)
**Closed today**: AUD_JPY ×2 (+1,786 JPY), GBP_USD ×1 (+670 JPY) — adjusted via OANDA


## AUD_JPY LIMIT LONG 3000u — 2026-04-10 18:22 UTC

| Item | Value |
|------|-------|
| Type | LIMIT |
| id | 467436 |
| Price | 112.600 |
| TP | 112.720 (+12pip) |
| SL | 112.480 (-12pip) |
| GTD | 21:00 UTC (expire at NY close — no weekend carry) |
| Spread | 1.6pip |
| Conviction | A |
| pretrade | S(8)/MEDIUM (H4 RSI=77 overbought noted — downgraded A for range regime) |

**Thesis**: AUD_JPY confirmed RANGE 112.58-112.72 (M5 chart — clear bounces at both extremes). LIMIT at range bottom (BB lower ~112.58). CS AUD(+0.35) vs JPY(-0.59) = gap 0.94 = macro supports buyers at range low. Not chasing (current price 112.694 = near range top). GTD=21:00 UTC protects against weekend gap.

**FOR**: CS gap=0.94 AUD>JPY (Cross-pair) + RANGE confirmed lower BB 112.58 (Structure) + H4/H1 ADX=56/41 BULL (Direction)
**Different lens**: Fib (Structure) — AUD_JPY BULL range q=1.29, Fib@10% = at/near range bottom = structural bounce zone → supports
**AGAINST**: H1 MACD bear div (fatigue signal). Friday thin market. H4 RSI=77 (overbought — don't chase). 3rd AUD_JPY entry today (churn flag).
**If wrong**: Range breaks below 112.55 with body → AUD sellers dominate → SL 112.48 triggers = -120 JPY (3000u × 12pip × ~0.333)

## Session 18:35-18:40 UTC

### GBP_JPY TPs Confirmed (18:26 UTC)
| Field | id=467385 | id=467403 |
|-------|-----------|-----------|
| Entry | 2000u @214.263 | 3000u @214.393 |
| Exit | @214.466 (TP) | @214.466 (TP) |
| P&L | +406 JPY | +219 JPY |

### LIMIT GBP_JPY LONG 3000u @214.350 (id=467443)
| Item | Value |
|------|-------|
| Time | 2026-04-10 18:40 UTC |
| Type | LIMIT GTD=21:00UTC |
| Entry | 214.350 |
| TP | 214.600 |
| SL | 214.050 |
| Spread | 3.2pip |
| Conviction | A |
| pretrade | A(6)/MEDIUM |
| Thesis | H1 ADX=62 band walk continues. M5 SQUEEZE breakout. CS GBP(+0.69) vs JPY(-0.59)=1.28 gap. LIMIT below close @214.466 (anti-churn compliant). |

---

## AUD_JPY LIMIT ADJUST — 2026-04-10 18:57Z

| Field | Value |
|-------|-------|
| Action | Cancelled 467436 (3000u @112.600), placed 467446 (5000u @112.640) |
| TP | 112.760 (from 112.720) |
| SL | 112.480 (unchanged) |
| GTD | 21:00 UTC (unchanged) |
| Reason | Chart reading: EMA20/BB mid confirmed at 112.640 (multiple bounces visible). 112.600 was below visible support. Sized up: pretrade S(8) supports 5000u. |


---

## AUD_JPY MARKET LONG 8000u — 2026-04-10 19:41 UTC

| Item | Value |
|------|-------|
| Type | MARKET |
| id | 467449 |
| Entry | 112.742 |
| TP | 112.930 (+18.8pip) |
| SL | 112.550 (-19.2pip) |
| Spread | 1.6pip |
| Conviction | S |
| pretrade | S(9) MEDIUM — iron-clad, size up |
| Risk | 8000u × 19pip × 0.01 = 1,520 JPY max loss (1.3% NAV) ✓ |
| Margin | 36,080 JPY (30% NAV). Worst case with EUR_USD+GBP_JPY LIMITs: 67% ✓ |

**Action**: Cancelled LIMIT 467446 (@112.640) — dip happening NOW (M5 StRSI=0.0), capturing via market entry at better overall timing.

**Thesis**: AUD_JPY M5 extreme oversold dip-buy within H4 ADX=56/H1 ADX=42 strong BULL trend. Lower wicks expanding = buyers returning. CS gap 1.08 (AUD+0.42/JPY-0.66).

**FOR**: Direction (H4 ADX=56 BULL + H1 ADX=42 BULL) + Timing (M5 StRSI=0.00 extreme) + Cross-pair (CS gap=1.08)
**Different lens**: Structure — Fib BULL now@65%, next wave BULL q=1.45 (healthy wave quality) → supports
**AGAINST**: H4 RSI=77 moderate overbought (not extreme like EUR_JPY RSI=85). Bodies shrinking M5. Friday thin.
**If wrong**: EMA20 at 112.640 broken → SL 112.550 → -1,520 JPY


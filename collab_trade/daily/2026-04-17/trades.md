# Trades — 2026-04-17

## Day-start NAV: 123,741.69 JPY

---

## Modifications

### 00:35 UTC — AUD_USD id=468595 TP adjusted
- Action: TP 0.71800 → 0.71693 (H1 BB mid, ATR×0.9 from entry 0.71609)
- Reason: protection_check flagged TP too wide (ATR×2.2). Range regime confirmed by chart. 0.71693 = achievable in Tokyo-London transition.
- txn: 468609, 468610

---

## Open Orders Placed

### 00:35 UTC — EUR_USD LIMIT LONG id=468613
- Pair: EUR_USD | Side: LONG | Units: 3,000u
- Limit Price: 1.17800 (M5 BB lower / H4 BOTTOM BULL structural floor)
- TP: 1.17862 (M5 BB upper - spread buffer)
- SL: 1.17730 (below H1 structural support)
- GTD: 2026-04-17T08:30:00Z (London open window)
- Pretrade conviction: B (H4 structural BULL, range floor, but M15 DI- dominant = limit only)
- Thesis: H4 ADX=42 BULL, StRSI=0.15 (floor). EUR M1 BID(+23). Chart shows RANGE at 1.1778-1.1788. Range buy at structural floor for London continuation of USD-weakness theme.
- Spread: 0.8pip | Margin: +~1,818 JPY if filled (1.1%)

---

## Entries

### 00:56 UTC — EUR_USD LONG 3000u @1.17800 FILLED id=468635
- Pair: EUR_USD | Side: LONG | Units: 3,000u
- Entry: 1.17800 (LIMIT filled — price dipped to ask=1.17800)
- TP: 1.17862 (H1 swing high - spread, already set on fill)
- SL: 1.17730 (H1 structural support, already set on fill)
- Pretrade conviction: MEDIUM | Type: Momentum | Hold: 30min-2h | Zombie: 02:56Z
- Thesis: H4 ADX=42 BULL StRSI=0.15 floor. SQUEEZE M5. H1 BEAR wave 71% complete = reversal. EUR M1 BID(+12) strongest. LIMIT at BB lower structural.
- Spread: 0.8pip | Different lens: H4 MACD hist decelerating = pullback momentum fading = thesis entry is correct timing.
- Session: Tokyo. Entry hour: 01:00 UTC.

---

## Positions Carried (from prior sessions)

| id | Pair | Side | Units | Entry | TP | SL | UPL | Status |
|----|------|------|-------|-------|----|----|-----|--------|
| 468595 | AUD_USD | LONG | 3,000 | 0.71609 | 0.71693 | none | -43→tbd | C-HOLD |
| 468471 | AUD_JPY | LIMIT | 3,000 | @113.884 | pending | — | — | pending |
| 468600 | AUD_USD | LIMIT(bot) | 3,690 | @0.71498 | pending | — | — | pending |
| 468613 | EUR_USD | LIMIT | 3,000 | @1.17800 | 1.17862 | 1.17730 | — | pending |

---

## P&L Summary
- Realized today (OANDA): TBD — bot closure losses from prior sub-session (~-1,250 JPY est.)
- Discretionary P&L: 0 entries, 0 losses
- Unrealized: AUD_USD ~-43 JPY

## 02:57Z - Market Entries (Band Walk)

| # | Pair | Dir | Units | Entry | TP | SL | id | Reason |
|---|------|-----|-------|-------|-----|-----|-----|--------|
| 1 | AUD_JPY | LONG | 3000 | 114.158 | 114.350 | 114.000 | 468853 | Band walk M5 TREND-BULL. AUD H4=+30, JPY offered 4/4. No retrace to LIMIT levels. |
| 2 | EUR_JPY | LONG | 3000 | 187.720 | 188.000 | none | 468857 | Band walk M5 TREND-BULL. EUR H4=BID+13, M15 ADX=39. LIMIT @187.55 reload. |

Pretrade: AUD_JPY=positive EV +223 JPY/trade (57% WR). EUR_JPY=positive EV +136 JPY/trade (62% WR).
Cold streak → B-size (3000u each). Margin after: ~40% projected.

### Close AUD_JPY — 2026-04-17 07:25Z
| Field | Value |
|-------|-------|
| Pair | AUD_JPY |
| Side | LONG |
| Units | 3000u |
| Close price | 114.148 |
| P/L | -288 JPY |
| Reason | M5 SQUEEZE resolved DOWN; M15 FRESH BEAR ADX=36; TP unreachable (15pip away); entry TF failed |

### Entry GBP_USD LONG — 2026-04-17 07:26Z
| Field | Value |
|-------|-------|
| Pair | GBP_USD |
| Side | LONG |
| Units | 3000u |
| Entry | 1.35132 |
| TP | 1.35250 |
| SL | 1.34990 |
| Trade ID | 469010 |
| Type | Counter-S |
| pretrade | MEDIUM/B |
| Thesis | Counter-S: H4 StRSI=0.02 + H1 StRSI=0.00 + H1 div @1.35158. London profit-taking flush 1.3524→1.3505 complete. USD weakness macro intact (ceasefire + tariff floor). M5 StRSI=1.0 confirms floor bounce. |

### 07:54 UTC — AUD_JPY LIMIT LONG id=469014
- Pair: AUD_JPY | Side: LONG | Units: 3,000u
- Limit Price: 114.080 (H1 EMA20 dip-buy zone, post-London flush)
- TP: 114.300 | SL: 113.950 | GTD: 09:30Z
- Conviction: B | pretrade: C/LOW(2) — H4 RSI=77 warning, but AUD H4 strongest (+33), JPY H4 weakest (-17), strategy_memory override applies
- Context: M5 TREND-BEAR correction within H4 ADX=47 BULL. AUD M1=BID(+11) starting recovery. LIMIT only (JPY M15 still bid).
- R:R: 22pip TP / 13pip SL = 1.7:1 | Max loss: ~390 JPY

### 17:35 UTC — EUR_USD LIMIT LONG id=469047
- Pair: EUR_USD | Side: LONG | Units: 3,000u
- Limit Price: 1.17900 (H1 EMA20 floor / post-flush structural bid)
- TP: 1.18120 | SL: 1.17820
- GTD: 2026-04-17T21:35:29Z
- Pretrade conviction: B / Allocation B (pre-event squeeze = LIMIT only)
- Thesis: H1 TREND-BULL correction into support. M5 and refreshed M1 both showed squeeze after a second flush to 1.1786-1.1790, so the honest action was structural bid only, not a market chase before Waller.
- Spread: 0.8pip | Tag: trader

### 17:36 UTC — EUR_USD LONG 3000u @1.17900 FILLED id=469048
- Pair: EUR_USD | Side: LONG | Units: 3,000u
- Entry: 1.17900 (LIMIT @1.17900 filled exactly)
- TP: 1.18120 (+22pip) | SL: 1.17820 (-8pip)
- Spread at fill: 0.8pip
- Margin: ~22,391 JPY (18.5% NAV at fill)
- Pretrade: B / Allocation B
- Thesis: H1 bull correction into support. The floor fill was worth one shot, but only if it converted into a 1.1802 reclaim before Waller.
- Conviction: B
- Zombie at: 18:36Z

---

## Entries

### 09:39 UTC — EUR_JPY LONG id=469026 [LIMIT FILL]
- Pair: EUR_JPY | Side: LONG | Units: 3,000u
- Entry: 187.616 (LIMIT @187.620 filled at better price)
- TP: 187.900 (+28pip) | SL: 187.450 (-16.6pip)
- Spread at fill: 1.3pip
- Margin: ~22,515 JPY (18.8% NAV)
- Pretrade: checked prior session (B, 60% WR, EV +163/trade)
- Thesis: M5 RANGE lower boundary 187.59-187.62 (3× tested). H4 StRSI=0.07 floor with acceleration. EUR H4+M15+M1 all BID. H1 BULL mid-wave (Fib43%, room to target). JPY M15 bid = short-term headwind absorbed by structural floor fill.
- Conviction: B→A (different lens: Fib H1 Fib43% mid-wave + EUR currency-wide bid = supports)
- Zombie at: 11:39Z

### Close EUR_JPY — 2026-04-17 11:07Z
| Field | Value |
|-------|-------|
| Pair | EUR_JPY |
| Side | LONG |
| Units | 3000u |
| Close price | 187.740 |
| P/L | +372 JPY |
| Reason | TAKE_PROFIT_ORDER at M5 range-upper cluster after lower-boundary fill thesis paid |
| OANDA txn | 469037 |
| Rotation | No reload placed. Fresh pretrade at the paid range-top came back Edge C / Allocation C, so the clean decision was to stay paid and wait for a better reset. |

### Close AUD_JPY — 2026-04-17 12:50Z
| Field | Value |
|-------|-------|
| Pair | AUD_JPY |
| Side | LONG |
| Units | 5000u |
| Entry | 114.080 |
| Close price | 114.304 |
| P/L | +1120 JPY |
| Reason | TAKE_PROFIT_ORDER filled after delayed post-event breakout from the defended 114.08 H1 support seat |
| OANDA txn | 469039 |
| Trade ID | 469029 |
| Note | No discretionary half-take needed. The original TP at 114.300 did the work once the squeeze finally resolved upward. |

### Close EUR_USD — 2026-04-17 17:44Z
| Field | Value |
|-------|-------|
| Pair | EUR_USD |
| Side | LONG |
| Units | 3000u |
| Entry | 1.17900 |
| Close price | 1.17895 |
| P/L | -24 JPY |
| Reason | pre_event_no_reclaim: the floor LIMIT filled, but M1/M5 never reclaimed and holding into Waller would have been inertia, not a fresh trade |
| OANDA txn | 469052 |
| Trade ID | 469048 |
| Note | Small scratch loss accepted. The fill itself was not the trigger; the missing 1.1802 reclaim was. |

### Entry GBP_JPY — 2026-04-17 12:57Z [LIMIT FILL]
| Field | Value |
|-------|-------|
| Pair | GBP_JPY |
| Side | LONG |
| Units | 3000u |
| Entry | 215.181 |
| TP | 215.400 |
| SL | 215.100 |
| Trade ID | 469041 |
| Source | Backup LIMIT @215.190 filled during the post-breakout fade |
| Thesis | Old H4 floor + H1 range-lower backup order stayed armed after the regime rotated. Fill was passive, not a fresh market-order judgment. |

### Close GBP_JPY — 2026-04-17 13:03Z
| Field | Value |
|-------|-------|
| Pair | GBP_JPY |
| Side | LONG |
| Units | 3000u |
| Close price | 215.099 |
| P/L | -246 JPY |
| Reason | STOP_LOSS_ORDER hit almost immediately after the backup LIMIT filled; the post-breakout fade kept accelerating and the lower half of the range did not hold |
| OANDA txn | 469044 |
| Trade ID | 469041 |
| Lesson | Once the active expression rotates away from a wide-spread JPY cross, clear the backup order instead of leaving accidental exposure armed. |

### Entry Order AUD_JPY — 2026-04-17 18:06Z [PENDING LIMIT]
| Field | Value |
|-------|-------|
| Pair | AUD_JPY |
| Side | SHORT |
| Units | 3000u |
| Entry | LIMIT 113.680 |
| TP | 113.560 |
| SL | 113.750 |
| Order ID | 469055 |
| GTD | 2026-04-17 23:00Z |
| Pretrade | Edge A / Allocation A / Risk MEDIUM, sized down to B (3000u) because Waller is still binary |
| Thesis | JPY remains BID across H4/M15/M1 while AUD_JPY M5/M1 is squeezed under 113.66-113.68. Positive pair expectancy on AUD_JPY SHORT + scanner Structural-S level makes the upper-box fade the only live structural receipt before the event. |

### 19:11 UTC — AUD_JPY SHORT 3000u @113.680 FILLED id=469056
| Field | Value |
|-------|-------|
| Pair | AUD_JPY |
| Side | SHORT |
| Units | 3000u |
| Entry | 113.680 |
| Filled from | LIMIT order id=469055 |
| Open time | 2026-04-17T19:11:52.869905693Z |
| TP | 113.560 (order id=469057) |
| SL | 113.750 (order id=469058) |
| Trade ID | 469056 |
| Tag | trader |
| Thesis | The upper-box fade actually triggered: AUD_JPY stayed capped under 113.68, H1 remained corrective bear, and the only honest handoff state became live short inventory rather than a flat book with a pending order. |

### Entry GBP_JPY — 2026-04-17 20:27Z [STOP FILL]
| Field | Value |
|-------|-------|
| Pair | GBP_JPY |
| Side | SHORT |
| Units | 3000u |
| Entry | 214.375 |
| Trigger | STOP 214.388 broke and filled immediately (order id=469061) |
| TP | 214.210 (id=469063) |
| SL | 214.490 (id=469064) |
| Trade ID | 469062 |
| Type | Momentum |
| Pretrade | Edge A / Allocation A from pretrade_check, manually capped to B-size for late session and weak pair history |
| Thesis | GBP stayed the weakest live seller, JPY remained the strongest H4 currency, and GBP_JPY was the only cross that converted a squeeze rollover into a real downside trigger. The honest trade was the breakdown receipt, not another prose-only flat handoff. |
| Spread at fill | 3.5pip |
| Margin | ~32,159 JPY (~26.6% NAV at fill) |
| Session | NY late / pre-close |

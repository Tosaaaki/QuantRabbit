# Trades 2026-04-13

## GBP_JPY LONG 3000u @214.942 [17:08Z]
| Item | Value |
|------|-------|
| Direction | LONG (MARKET) |
| Size | 3,000u |
| Entry | 214.942 |
| TP | 215.370 |
| SL | 214.400 |
| Trade ID | 467638 |
| Spread | 2.8pip |
| pretrade | A(7) MEDIUM |
| Thesis | M5 BB lower StRSI=0.02 (extreme oversold) in H1 ADX=34 BULL trend. Squeeze break expected. GBP(+0.46) strongest vs JPY(-0.26) weakest. H4 ADX=57 BULL. |
| AGAINST | H4 RSI=78 + H4+H1 MACD bear div = overextension. |
| Conviction | A |

## LIMIT AUD_JPY LONG 2000u @112.800 [17:08Z]
| Item | Value |
|------|-------|
| Direction | LONG (LIMIT) |
| Size | 2,000u |
| Entry | 112.800 |
| TP | 113.400 |
| SL | 112.450 |
| Order ID | 467641 |
| GTD | 2026-04-13T21:00Z |
| Thesis | H4 ADX=43 DI+=31 BULL. Chart TREND-BULL band walk. Currently at H1 BB upper (112.951) — wait for pullback to H1 mid area. |
| Conviction | A |

## 18:46Z — Order Management + EUR_JPY LIMIT

### CANCELLED: AUD_USD LIMIT id=467629
- Was: 2000u @0.70540 GTD 20:00Z
- Reason: 38pip away from current (0.70921), no fill expected. Cancel to free margin for EUR_JPY.

### CANCELLED: AUD_JPY LIMIT id=467641
- Was: 2000u @112.800 GTD 21:00Z (rollover)
- Reason: Expires at rollover, re-enter at Fib 38.2% (112.910) after AUD data 00:30Z/01:30Z.

### NEW: EUR_JPY LONG LIMIT id=467648
| Field | Value |
|-------|-------|
| Pair | EUR_JPY |
| Dir | LONG |
| Units | 2000u |
| Entry | @187.050 (LIMIT) |
| TP | 187.350 (GTC, +30pip) |
| SL | 186.900 (GTC, -15pip) |
| GTD | 2026-04-13T22:45Z |
| Spread | 1.9pip |
| pretrade | B(score=5) |
| Conviction | A (margin-constrained to 2000u) |
| Thesis | EUR_JPY SQUEEZE broke UP (spike to 187.229, Fib257%). LIMIT at Fib 61.8% pullback (187.050). H4 ADX=56 BULL + EUR strongest +0.38 + JPY weakest -0.24. Not chasing spike — buying dip. |
| Invalidation | Spike reversal through wave base 187.002 → SL 186.900 (-300 JPY max loss) |

## USD_JPY SHORT 2000u @159.400 LIMIT [19:12Z] — PENDING
| Item | Value |
|------|-------|
| Direction | SHORT (LIMIT) |
| Size | 2,000u |
| Entry | 159.400 |
| TP | 158.900 |
| SL | 159.650 |
| GTD | 2026-04-13T21:00Z |
| id | 467649 |
| Conviction | B |
| Thesis | H1 StRSI=0.0 bounce Fib69% complete → N=BEAR q=0.26. LIMIT @159.40 rally-sell before bear resumes. |
| Pretrade | A(score=6) Risk=HIGH WR-WARN:33% → conviction downgraded to B |
| FOR | Direction(H1+M5 BEAR) + Timing(M5 StRSI=0.98 bounce peak) + Macro(USD weakest) |
| Different lens | Fib N=BEAR q=0.26 LOW quality → conviction DOWN |
| AGAINST | Historical WR=33% (-22k), H4 neutral, Fib low quality |

| Time | Pair | Side | Units | Entry | TP | SL | Type | Thesis | Conv | Pretrade | Status |
|------|------|------|-------|-------|----|----|------|--------|------|----------|--------|
| 21:24Z | GBP_USD | LONG | 2000u | LIMIT 1.34900 | 1.35450 | 1.34400 | Momentum | M5 BB lower + StRSI=0.0 + H1 BULL ADX=33 + GBP(+0.69) strongest. GTD 04/14 01:00Z | A | A(7)/MEDIUM | PENDING id=467662 |

## EUR_JPY LIMIT GTD EXTENSION [21:48Z]
| Action | Cancel id=467648 (exp=22:45Z) → Re-place id=467664 (exp=04:00Z) |
|--------|------------------------------------------------------------------|
| Pair | EUR_JPY |
| Dir | LONG LIMIT |
| Units | 2000u |
| Price | 187.050 |
| TP | 187.350 |
| SL | 186.900 |
| GTD | 2026-04-14T04:00Z (was 22:45Z) |
| Reason | Fib@99% BULL wave exhausted. N=BEAR q=2.02. M5 large bearish pullback. 30pip needed for fill, 55min was insufficient. Extended for Asia session (6.25h window). |

## USD_JPY SHORT 2000u @159.402 — CLOSED TP [23:11Z]
| Item | Value |
|------|-------|
| Direction | SHORT |
| Size | 2,000u |
| Entry | 159.402 |
| Close | 159.314 (TP fill) |
| P&L | +176 JPY (+8.8pip) |
| Trade ID | 467652 |
| Hold time | ~4h |
| Reason | TP GTC fill. M5 SQUEEZE broke down through 159.314. USD weak confirmed. |
| pretrade | A(6) MEDIUM |

## EUR_USD LONG LIMIT 3000u @1.17500 [23:14Z] — PENDING id=467673
| Item | Value |
|------|-------|
| Direction | LONG (LIMIT) |
| Size | 3,000u |
| Entry | 1.17500 |
| TP | 1.17750 |
| SL | 1.17350 |
| GTD | 2026-04-14T04:00Z |
| Trade ID | 467673 |
| Spread | 0.8pip |
| pretrade | A(6) MEDIUM |
| Conviction | A |
| Thesis | H4 ADX=42 BULL + EUR strongest +0.69 vs USD weakest -0.44. LIMIT at first H4 dip level (~14pip below current) = M5 BB lower edge zone. TP=1.17750 is within H4 continuation target. |
| FOR | Direction (H4+H1 ADX=42/36 BULL) + Macro (CS gap=1.13) + Structure (1.17500 = M5 BB lower in squeeze, pullback zone) |
| Different lens | Fib (M5 micro BEAR wave q=0.61 showing pullback — 1.17500 is near pullback completion zone) → supports |
| AGAINST | H4 RSI=71 overbought. AUD binary 00:30Z may spike USD briefly through 1.17500. GTD=04:00Z limits max exposure. |
| If wrong | Squeeze breaks DOWN hard through 1.17350 without bouncing at 1.17500. SL protects. |

## 23:21Z — GBP_JPY LONG CLOSE (Trail triggered)
| Item | Value |
|------|-------|
| Direction | LONG close (trail trigger) |
| Units | 3000u |
| Entry | 214.942 |
| Close | ~215.243 (trail trigger) |
| P&L | +903 JPY |
| Hold | ~10h |
| Reason | Trail 10pip triggered as price failed to reach TP=215.370. H1 TREND-BULL intact. |

## 23:24Z — USD_JPY SHORT entry
| Item | Value |
|------|-------|
| Direction | SHORT |
| Units | 2000u |
| Entry | 159.229 |
| TP | 159.000 |
| SL | 159.420 |
| Spread | 0.8pip |
| Pretrade | B(5) HIGH risk |
| Conviction | B — USD weak CS=-0.44, SQUEEZE broke DOWN, JPY bid (JGB 29y high) |
| FOR | Direction(H1 DI-dom, M5 DI-=34) + Macro(BOJ hike repricing) + Cross-pair(EUR/GBP USD pairs UP) |
| Diff lens | Structure: Fib N=BEAR, below BULL wave low 159.328 → confirms breakdown |
| AGAINST | H4 neutral, H1 ADX=21 weak, WR=33% historical |
| If wrong | Bounces above 159.42. SL there. |
| Margin | 10.1% → ~42% worst case (pending LIMITs) ✓ |

### [23:39Z] LIMIT GBP_JPY LONG 4000u @215.120 [id=467681]
| Field | Value |
|-------|-------|
| Type | A-S Trend-Dip |
| Entry | LIMIT @215.120 |
| TP | 215.350 (+23pip net) |
| SL | 214.900 (-22pip) |
| GTD | 2026-04-14T01:35Z |
| Pretrade | S(8) MEDIUM-risk, WR=68%, +3,670 JPY total |
| FOR | Direction(H1 ADX=39 BULL) + Timing(M5 StRSI=0.04 extreme) + Cross-pair(GBP+0.64/JPY-0.35) |
| Different lens | Fib@78% pullback within bull wave → bounce zone ✓ |
| AGAINST | H4 StRSI=1.0 overbought — H4 extended but H1 still driving |
| If wrong | Price breaks below 215.146 (Fib 100%) = bull wave dead → SL at 214.90 |
| Margin after | ~37.5% live, ~70% worst-case with pending LIMITs |

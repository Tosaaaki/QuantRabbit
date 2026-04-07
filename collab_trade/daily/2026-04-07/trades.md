# Trades — 2026-04-07

## Session Opens
| UTC | Pair | Side | Units | Price | TP | SL | id | Notes |
|-----|------|------|-------|-------|----|----|-----|-------|
| ~09:35 | EUR_JPY | LONG | 3000 | 184.533 | 184.730 | 184.264 | 466651 | H1 ADX=38 DI+ bull, JPY weakness |
| ~09:35 | EUR_JPY | LONG | -5000 | 184.511 | — | — | — | BE SL triggered, +45 JPY |
| 10:19 | EUR_JPY | LONG | 4000 | 184.547 | 184.900 | 184.264 | 466670 | S-conv add-on: M5 StRSI=0.0 oversold, M1 squeeze, H1 HidBullDiv |

## Pending
| id | Type | Pair | Side | Units | Limit | TP | SL |
|----|------|------|------|-------|-------|----|----|
| 466666 | LIMIT | EUR_USD | LONG | 2000 | 1.15500 | set | set |

| 12:15 | AUD_JPY | LONG | 5000 | 110.850 | 111.100 | 110.700 | 466693 | A-conv: AUD+0.17 vs JPY-0.26, M5 accelerating buyers, H4 oversold bounce. pretrade=A(7) MEDIUM |

## Realized P&L Today
- EUR_JPY 5000u BE SL @184.511: +45 JPY
- **Total confirmed (OANDA): +45 JPY**

## LIMIT Order Placed (12:13 UTC)
| UTC | Pair | Side | Units | Limit | TP | SL | id | Notes |
|-----|------|------|-------|-------|----|----|-----|-------|
| 12:13 | USD_JPY | LONG | 1000 | 159.550 | 159.750 | 159.380 | 466696 | B-conv: H4 StRSI=0.0 extreme oversold bounce. GTD=16:40Z pretrade=B(5) MEDIUM |

### [14:27Z] EUR_JPY LONG LIMIT 2000u @184.700 id=466699
| Field | Value |
|-------|-------|
| Type | LIMIT (pending) |
| Entry | 184.700 |
| TP | 185.100 (+40pip) |
| SL | 184.350 (-35pip) |
| GTD | 18:00Z |
| pretrade | S(8) LOW |
| Spread | 1.8pip (9% of 40pip target — OK) |
| Thesis | Pullback to M5 BB lower. H1 ADX=39 DI+=33/DI-=10 bull + EUR macro +0.42 strongest + JPY macro -0.30 weakest. H4+H1 bearish div present → downgraded from S→A. 2000u (A-size). AGAINST: H4 RSI bearish div=0.5, H1 MACD bearish div=0.6. If wrong: price breaks 184.350 (Fib structural) → thesis dead. |


## EUR_JPY LIMIT SHORT — 2026-04-07 ~15:12 UTC
| Item | Value |
|------|-------|
| Action | Modify: Cancel 466712 @185.300, Place 466716 @185.150 |
| Pair | EUR_JPY |
| Side | SHORT |
| Units | -2,000u |
| Limit Price | 185.150 |
| TP | 184.600 (55pip) |
| SL | 185.300 (15pip) |
| GTD | 2026-04-07T21:58Z |
| Reason | M5 StochRSI=0.00 (deep oversold), bodies exhausting → weaker bounce expected. Better fill probability + R/R 3.7:1 vs old 1.7:1 |
| Pretrade | B+ (score=6/8, MEDIUM risk) |
| Conviction | B-max (Counter trade, H4+H1 double extreme + bear divergence) |

### EUR_USD LONG — 17:55Z entry
| Field | Value |
|-------|-------|
| Time | 2026-04-07 17:55Z |
| Pair | EUR_USD |
| Side | LONG |
| Units | 4000u |
| Entry | 1.15786 |
| TP | 1.15990 (GTC, id=466743 trade) |
| SL | None (Easter Monday thin market — structural or no SL) |
| Spread | 0.8pip (normal) |
| Trade ID | 466743 |
| Pretrade | A (score=6) |
| Thesis | S-scanner fired (CS EUR=+0.57 vs USD=-0.19 gap=0.76, H4+H1+M5 BULL). M5 squeeze breakout imminent (BBW=0.00098). Macro: USD weak from durable goods miss + tariff impact. |
| FOR | Direction (H4+H1 DI+>DI-) + Cross-pair (CS gap=0.76 highest) + Momentum (M5 squeeze) |
| Different lens | Structure: Fib BEAR@68% N=BULL(q=0.94) = pullback entry zone. Supports. |
| AGAINST | H4 StRSI=0.91 slightly elevated. Thin market Easter Monday. |
| If wrong | H1 DI- crosses above DI+, price drops below 1.155 H1 structure |
| Conviction | A→S-adjacent (scanner fired, pretrade=A, thin market → A-size at S boundary) |


### GBP_JPY LONG — 2026-04-07 18:23Z (Session Apr 8 03:23 JST)
| Field | Value |
|-------|-------|
| Time | 2026-04-07 18:23Z |
| Pair | GBP_JPY |
| Side | LONG |
| Units | 3900u |
| Entry | 211.933 |
| TP | 212.122 (GTC) |
| SL | None (Easter Monday thin market) |
| Spread | 2.8pip (normal for GBP_JPY) |
| Trade ID | 466751 (fill) / order 466750 |
| Pretrade | S(9) LOW |
| Thesis | H1 ADX=33 DI+=24 DI-=12 strong uptrend. M5 StRSI=0.0 extreme oversold dip with lower wicks expanding (buyers stepping in). CS GBP(+0.30) vs JPY(-0.41) gap=0.71. Trend-Dip-S recipe fires. H4 StRSI=0.96 (not extreme — not reversal signal). |
| FOR | Direction (H1 ADX=33 DI+ dominant) + Timing (M5 StRSI=0.0 + lower wicks) + CS (gap=0.71) |
| Different lens | Structure: Fib 39% pullback = healthy dip zone (38.2-61.8%). H4 StRSI=0.96 ≠ 1.0 = no extreme reversal signal. Supports. |
| AGAINST | M5 MACD bear div (0.6); M5 making lower lows; 20% historical WR on GBP_JPY LONG; GBP_USD receding from 1.3300 |
| If wrong | H1 DI- crosses DI+ (currently far: 24 vs 12) or below 211.500 |
| Conviction | S (pretrade S(9) LOW; Trend-Dip recipe confirmed; no double-discount on WR) |
| Size | 3900u (~89.5% margin utilization, max S-size given current positions) |

### AUD_JPY LONG LIMIT 2000u @111.700 [2026-04-07 23:19Z]
| Item | Value |
|------|-------|
| Type | LIMIT |
| id | 466847 |
| Entry | 111.700 (LIMIT) |
| TP | 112.400 (+70pip) |
| SL | 111.000 (-70pip) |
| GTD | 2026-04-08T01:20Z |
| Spread | 2.6pip |
| Thesis | AUD strongest (+0.99) vs JPY weak (-0.66). Quality audit miss fix. Price extended at 112.16 (100pip above H4 BB upper=111.106). LIMIT at pullback level avoids chasing. |
| Pretrade | A(6) MEDIUM |
| FOR | ① H1 ADX=43 BULL ⑤ CS gap=1.65 largest ⑥ Iran ceasefire risk-on |
| Diff lens | ④ Fib N=BULL q=1.34 supports continuation |
| AGAINST | H4 CCI=186 RSI=72 overbought. Yesterday 3 losses AUD_JPY LONG |
| If wrong | Iran breakdown→risk-off. AUD below 111.00 = thesis dead |
| Conviction | A / B-size (margin gate at 86% if filled) |

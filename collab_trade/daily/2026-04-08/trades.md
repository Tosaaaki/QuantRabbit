# Trades — 2026-04-08

## EUR_JPY LONG (id=466719)

| Field | Value |
|-------|-------|
| Entry time | 2026-04-07 15:16Z |
| Entry price | 185.045 (ask) |
| Units | 4500 |
| TP | 185.381 (structural: H1 momentum target ATR×2.0) |
| SL | 184.800 (below 78.6% Fib = 184.792, structural) |
| Spread | 1.9pip |
| Pretrade | S(9) LOW risk |
| Type | Momentum (H1 trend dip) |
| Thesis | EUR_JPY H1 ADX=43 strong uptrend, M5 pulled back to BB lower + StRSI=0.0 |
| FOR | Direction (H1 ADX=43 DI+=37) + Timing (M5 StRSI=0.0 BB lower) + Macro (EUR+0.60 vs JPY-0.45 gap=1.05) |
| AGAINST | H4+H1 bearish div, H1 RSI=75 extreme, VWAP 82pip extended, Iran JPY risk |
| If wrong | H1 DI+ flips or Iran JPY spike below 184.792 |

## AUD_JPY SHORT LIMIT (id=466722)
| Field | Value |
|-------|-------|
| Order time | 2026-04-07 15:16Z |
| Limit price | 111.080 |
| Units | -3000 |
| TP on fill | 110.700 |
| SL on fill | 111.250 |
| GTD | 2026-04-07 20:07Z |
| Type | Counter (H4+H1+M5 triple overbought) |
| Thesis | H4 StRSI=1.0 RSI_div=0.5 bear, H1 StRSI=0.99 RSI=73 extreme, M5 StRSI=1.0 CCI=134 — expecting pullback |

## Pending (from previous session)
- EUR_USD LONG LIMIT @1.15550 id=466714 GTD=2026-04-07T22:00Z — buy the dip on USD weakness

---

## EUR_USD LONG (id=466735) — 02:18 JST Apr 8 (17:18 UTC Apr 7)

| Field | Value |
|-------|-------|
| Entry time | 2026-04-07 17:18Z |
| Entry price | 1.15694 (ask) |
| Units | 4000 |
| TP | 1.15789 (structural: wave H=1.15797 - spread) |
| SL | None (Easter Monday thin market — discretionary) |
| Spread | 0.8pip |
| Pretrade | A(7) MEDIUM |
| Type | Momentum-S |
| Thesis | M5 StRSI=0.0 at BB lower in H1 uptrend. EUR(+0.57) vs USD(-0.19) CS gap=0.76. |
| FOR | Timing (M5 StRSI=0.0 + BB lower) + Cross-pair (CS gap=0.76) + Direction (H1 ADX=24 DI+=31 BULL) |
| Different lens | Structure (Fib 82% retracement = near wave bottom) → supports |
| AGAINST | H4 StRSI=0.91 extended. Easter Monday EU closed thin market. |
| If wrong | USD safe-haven bid from Iran → breaks below 1.1555 (H1 Ichimoku cloud) |

## Session: 2026-04-07 17:27Z

### LIMIT Orders Placed

| Time | Action | Pair | Dir | Units | Price | TP | SL | GTD | ID |
|------|--------|------|-----|-------|-------|----|----|-----|-----|
| 17:27Z | LIMIT entry | GBP_JPY | LONG | 2000 | 211.750 | 212.000 | None | 19:27Z | 466737 |
| 17:27Z | LIMIT entry | USD_JPY | SHORT | 1000 | 159.990 | 159.500 | None | 17:57Z | 466738 |

**GBP_JPY conviction block:**
```
Thesis: H1 ADX=33 BULL. GBP(+0.25) vs JPY(-0.42) CS gap=0.67. M1 StRSI=0.06 dip entry.
Type: Momentum
FOR: ① Direction (H1 ADX=33 DI+=23 BULL) ② Cross-pair (CS gap=0.67) ③ Timing (M1 StRSI=0.06 + Fib @28%)
Different lens: H4 StRSI=0.96 near extreme (not 1.0). CCI=128, RSI=63 → not at extreme threshold.
AGAINST: H4 near extreme, Iran risk-off, Easter Monday thin market, WR history 20% (1/5)
If wrong: JPY bounces hard, H1 DI+ flips. Discretionary close below 211.5.
→ Conviction: A | Size: 2000u (15% NAV)
pretrade: S(8) LOW risk
```

**USD_JPY SHORT conviction block:**
```
Thesis: BoJ intervention zone at 160. Entering SHORT counter-trade for direction balance.
Type: Counter
FOR: ② Timing (M5 StRSI=1.0 extreme overbought) ③ Macro (BoJ 160 intervention zone confirmed) ④ Structure (160 = hard ceiling)
Different lens: H4 ADX=12 range, H4 StRSI=0.83 = not extreme. H1 also range. Trend is weak.
AGAINST: H1 DI+=25 slight bull bias, M1 squeeze could break up, not confirmed intervention
If wrong: USD_JPY breaks 160 without intervention. GTD=30min limits exposure.
→ Conviction: B | Size: 1000u
```
# Trades — 2026-04-08

## EUR_USD LONG LIMIT (id=466740)
| Field | Value |
|-------|-------|
| Time | 2026-04-07 17:46Z |
| Action | LIMIT_ENTRY |
| Pair | EUR_USD |
| Side | LONG |
| Units | 2000u |
| Entry | 1.1563 |
| TP | 1.1590 (+27pip) |
| SL | None (Easter Monday) |
| GTD | 2026-04-07T22:00Z |
| pretrade | A(6) MEDIUM |
| Spread | 0.8pip |
| Thesis | EUR(+0.57) USD(-0.19) CS-gap=0.76. H4+H1 BULL. M5 StRSI=1.0 peaked → LIMIT dip buy |


## Session 2026-04-08 ~18:04Z

### LIMITS Placed

| Time | Pair | Dir | Units | Price | TP | Type | id |
|------|------|-----|-------|-------|-----|------|-----|
| 18:04Z | GBP_JPY | LONG | 2000 | 211.750 | 212.100 | LIMIT GTD+4h | 466747 |
| 18:04Z | AUD_JPY | SHORT | 1200 | 111.180 | 111.020 | Counter LIMIT GTD+2h | 466748 |

### Held Positions
- EUR_USD LONG 4000u @1.15786, TP @1.15990 (id=466743) — Hold (H1 ADX=24 BULL, CS gap=0.76)
- EUR_JPY LONG 4500u @185.045, TP @185.143 (id=466719) — Hold (H1 ADX=44 BULL strong)
- Expired LIMIT 466737 (GBP_JPY @211.750) cancelled. Re-placed as 466747.

### Quality Audit Addressed
- AUD_USD: Valid pass — M5 upper wicks + H4 StRSI=0.95 + EUR_USD covers theme
- GBP_JPY: Re-placed LIMIT @211.750 (2000u, A-size, H4 near extreme caps to A not S)
- AUD_JPY: Counter SHORT LIMIT placed (H4+H1 both StRSI=1.00 extreme)
- EUR_JPY undersized (4500u): Addressed — existing position HELD, not re-entered

## Session 18:25Z — Margin Crisis Response

### CLOSE: EUR_JPY LONG 4500u (id=466719)
| Field | Value |
|-------|-------|
| Close time | 2026-04-07 18:30Z |
| Close price | 184.974 |
| Entry | 185.045 |
| P&L | -319.5 JPY |
| Reason | Margin crisis 97% (rule: 95%+ = force close). H4 StRSI=1.00 + RSI bear div. EUR theme covered by EUR_USD. |

### MODIFY (HALF_TP): EUR_USD LONG 2000u @1.15812 (id=466743)
| Field | Value |
|-------|-------|
| Time | 2026-04-07 18:30Z |
| Close price | 1.15812 |
| Entry | 1.15786 |
| P&L | +82.9 JPY |
| Remaining | 2000u LONG |
| Reason | profit_check HALF_TP (ATR 0.3x, M5 StRSI=1.00 overbought). Margin relief. |

### LIMIT PLACED: AUD_USD LONG 7000u @0.69480 (id=466758)
| Field | Value |
|-------|-------|
| Time | 2026-04-07 18:30Z |
| Entry (pending) | 0.69480 |
| TP | 0.69780 |
| SL | None (Easter Monday thin market) |
| GTD | 2026-04-07T22:30Z |
| Reason | S-candidate quality audit miss fix. CS AUD(+0.45) vs USD(-0.21) gap=0.66. M5 pullback to BB lower level. |
| pretrade | S-candidate (quality audit flagged as missed) |


## Session 2026-04-08 20:07Z

### CLOSED: AUD_JPY SHORT 1200u
| Field | Value |
|-------|-------|
| Close time | 2026-04-07 20:07Z |
| Close price | 111.250 |
| Entry | 111.184 |
| PL | -79.2 JPY |
| Reason | H1 ADX=37 DI+=32 BULL — counter-trade thesis (H4/H1 StRSI=1.0 reversal) invalidated by strong trend. S-conviction scanner fires LONG on AUD_JPY. Exit wrong-direction position. |

### ENTRY: EUR_JPY LONG 2000u
| Field | Value |
|-------|-------|
| Entry time | 2026-04-07 20:07Z |
| Entry price | 185.038 |
| Units | 2000 |
| TP | 185.400 |
| SL | 184.850 |
| Conviction | A (pretrade score=6, LOW risk) |
| Thesis | H1 ADX=45 strong BULL + CS EUR(+0.64)/JPY(-0.48) gap=1.12. M5 BB squeeze breakout pending. AUD_USD trailing stop added (9pip). |
| Pretrade | A(6) LOW |

## GBP_JPY LONG [20:21Z]
| Field | Value |
|-------|-------|
| Entry | 212.150 (ask) |
| Units | 4,000u |
| TP | 213.200 |
| SL | 211.750 |
| Spread | 2.8pip |
| Pretrade | S(8) LOW |
| Conviction | S — CS gap=0.93 + H1 ADX=34 + M5 squeeze + Iran ceasefire risk-on |
| AGAINST | H1 MACD RegBear div (-1.0) |
| Trade ID | 466781 |
| Margin | ~33,940 JPY (30% NAV) |

## AUD_JPY SHORT counter LIMIT [20:22Z]
| Field | Value |
|-------|-------|
| Type | LIMIT SHORT (counter-trade) |
| Entry | @111.350 |
| Units | 1,000u |
| TP | 111.016 |
| SL | 111.700 |
| GTD | 2026-04-08 00:00Z |
| Reason | H4+H1+M5 all StRSI=1.0 extreme overbought. Spike fade play. |
| Order ID | 466784 |

## EUR_USD LONG (id=466791)

| Field | Value |
|-------|-------|
| Entry time | 2026-04-07 20:52Z |
| Entry price | 1.15956 (ask) |
| Units | 4000 |
| TP | 1.16240 (structural momentum target) |
| SL | 1.15720 (below Fib 78.6% area) |
| Spread | 0.8pip |
| Pretrade | S(8) MEDIUM risk |
| Type | Momentum (Trend Dip) |
| Thesis | EUR strongest (+0.64) vs USD weakest (-0.35), M5 pulled back to StRSI=0.07 extreme oversold within H1 ADX=28 bull trend; Iran ceasefire risk-on |
| FOR | Direction (H4+H1+M5 DI+ dominant, H1 ADX=28) + Timing (M5 StRSI=0.07 extreme oversold) + Macro (EUR-USD CS gap=0.99, Iran risk-on) |
| Different lens | Structure (Fib BULL wave at 25% = room to H=1.16052+) → supports |
| AGAINST | H1 StRSI=1.00 extreme overbought (pullback may extend); Iran binary deadline tonight |
| If wrong | Iran escalates → risk-off USD bid → drops to 1.1572 SL, H1 structure breaks |
| Quality audit | Addressing missed S-candidate from previous session |

## CLOSE: AUD_JPY SHORT 1000u (id=466787) — Session 4 [21:26Z]

| Field | Value |
|-------|-------|
| Close time | 2026-04-07 21:26 UTC |
| Close price | 111.476 (ask) |
| Entry | 111.355 |
| PL | -121 JPY |
| Reason | H1 ADX=38 DI+ dominant = thesis invalid. Counter-trade (H4+H1 StRSI=1.0 reversal) overridden by strong H1 DI+ trend. S-scanner fires LONG on AUD_JPY. Exit before loss grows. |
| Spread at close | 10.6pip (Iran thin market) |


## EUR_USD LONG (id=466834) — session Apr 8 07:58 JST

| Field | Value |
|-------|-------|
| Entry time | 2026-04-08 22:58Z |
| Entry price | 1.16664 (ask) |
| Units | 5000 |
| TP | 1.16940 (+27pip) |
| Trail | 20pip |
| Spread | 0.8pip |
| Pretrade | S(8) MEDIUM risk |
| Type | Momentum-S |
| Thesis | EUR(+0.71) vs USD(-0.43) CS=1.14. ECB hawkish pivot + Fed cut = USD structurally weak. H1 ADX=31 DI+=32 intact. M5 buyers dominant accelerating. |
| FOR | Direction(H1ADX31+H4+H1+M5 BULL) + Macro(ECB hawkish, USD weak) + CrossPair(all EUR BULL aligned) |
| Different lens | Structure: Fib191% extended, H1 CCI=89 (not extreme, room remains) → neutral |
| AGAINST | H4+H1 StRSI=1.0 overbought + M5 RSI+MACD divergence (trend fatigue) |
| If wrong | H1 DI- reversal or ECB reversal → drop to 1.1580-1.1600 |
| Conviction | S |

## GBP_USD LONG (id=466836) — session Apr 8 07:58 JST

| Field | Value |
|-------|-------|
| Entry time | 2026-04-08 22:58Z |
| Entry price | 1.33725 (ask) |
| Units | 3000 |
| TP | 1.34100 (+38pip) |
| Trail | 20pip |
| Spread | 1.3pip |
| Pretrade | A(7) LOW risk |
| Type | Momentum-A |
| Thesis | GBP(+0.57) vs USD(-0.43). STRONGEST M5 momentum (ADX=50, higher highs). USD weak theme. |
| FOR | Direction(H1ADX29+H4+H1+M5 BULL) + CrossPair(CS gap=1.00) + Momentum(M5 ADX=50 strongest) |
| Different lens | Timing: M5 RSI=80 overbought + M5 divergence (RSI_div=0.8) → contradicts. Conviction A not S. |
| AGAINST | H4 StRSI=1.0 + M5 divergence + recent GBP_USD LONG losses Apr 7 |
| If wrong | M5 stalls, divergence confirms → reversal to 1.3280-1.3300 |
| Conviction | A |

## AUD_USD SHORT LIMIT (id=466841) — counter-trade

| Field | Value |
|-------|-------|
| Order time | 2026-04-08 22:58Z |
| Limit price | 0.71000 (SHORT) |
| Units | -2000 |
| TP on fill | 0.70600 |
| SL on fill | 0.71200 |
| GTD | 2026-04-08T02:58Z |
| Type | Counter (H4+H1 StRSI=1.0 fade) |
| Thesis | AUD_USD Fib208% extension + H4+H1 StRSI=1.0 = most overextended. Fade at 0.7100 if it runs. |

## AUD_USD LONG (id=466844) — session Apr 8 08:10 JST (23:10 UTC)

| Field | Value |
|-------|-------|
| Entry time | 2026-04-08 23:10Z |
| Entry price | 0.70564 (ask) |
| Units | 5000 |
| TP | 0.71200 (ceiling, ATR×3.3 — trailing stop manages exit) |
| Trail | 20pip |
| Spread | 1.4pip (normal) |
| Pretrade | S(8) MEDIUM risk |
| Type | Momentum-A |
| Thesis | AUD(+0.99) strongest vs USD(-0.70) weakest. Gap=1.69 largest CS spread across all pairs. M5 buyers dominant, bodies growing. Not yet in AUD portfolio. |
| FOR | Direction(H1ADX33 DI+=47 strong BULL) + CrossPair(CS gap=1.69 AUD strongest) + Momentum(M5 EMA slope +, ROC5=0.52) |
| Different lens | Structure: Fib@309% but N=BULL (new wave started, not exhaustion) + H4 range ADX=17 (range breakout, not trend exhaustion) → neutral/supports |
| AGAINST | H4+H1+M5 StRSI=1.0 all overbought + M5 RSI+MACD divergence + Iran event risk 26h |
| If wrong | Iran escalates → risk-off AUD dump → below 0.700 invalidation |
| Conviction | A (pretrade S but H4 overbought + M5 divergence downgrades to A) |
| Quality audit | Addressing missed AUD_USD S-candidate |

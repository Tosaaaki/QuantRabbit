# Trader State — 2026-04-08
**Last Updated**: 2026-04-07 23:15 UTC

## Market Reading (M5 Price Action)
1. Stronger: **Buyers (EUR/GBP/AUD vs USD)** — Iran ceasefire 2-week suspension confirmed. AUD(+0.99) EUR(+0.94) GBP(+0.78) vs USD(-0.70) JPY(-0.66). Risk-on.
2. Phase: Late-stage acceleration. H4+H1+M5 StRSI=1.0 + M5 divergences across all USD pairs = exhaustion signals present. Extended above H4 BB uppers.
3. Positions: 3 LONG (EUR_USD, GBP_USD, AUD_USD) + 1 LIMIT LONG (AUD_JPY pending at 111.70).

## Positions (Current)

### EUR_USD LONG (id=466834) — 5000u @1.16664
- **Thesis**: EUR(+0.94) vs USD(-0.70) CS=1.64. Iran ceasefire + ECB hawkish + USD structurally weak.
- **Basis**: H4+H1+M5 BULL MTF aligned. H1 ADX=34 DI+=48. M5 divergences (fatigue) present.
- **Invalidation**: H1 DI- reversal or Iran escalation below 1.1600.
- **TP**: 1.16940 (+15pip) | **Trail**: 20pip
- **3-option**: C — Hold as-is. ATR 0.7×, MACD_H+ slope up, 15pip to TP.
- **UPL**: ~+1,314 JPY (+12.4pip). **Peak**: 12.4pip.

### GBP_USD LONG (id=466836) — 3000u @1.33725
- **Thesis**: GBP(+0.78) vs USD(-0.70). H1 ADX=32 strong trend. USD weak theme.
- **Basis**: H4+H1+M5 BULL. M5 ADX=49 = strong momentum. CS gap=1.48.
- **Invalidation**: M5 momentum stalls, trail stop hit.
- **TP**: 1.34100 (+19pip) | **Trail**: 20pip
- **3-option**: C — Hold as-is. ATR 0.8×, 19pip to TP, MACD_H+ running.
- **UPL**: ~+1,401 JPY (+18.4pip). **Peak**: 18.4pip.

### AUD_USD LONG (id=466844) — 5000u @0.70564
- **Thesis**: AUD(+0.99) strongest, USD(-0.70) weakest. CS gap=1.69 = largest.
- **Basis**: H1 ADX=33 DI+=47 BULL. Iran ceasefire = risk-on = AUD bid.
- **Invalidation**: Iran escalation risk-off → AUD dump below 0.700.
- **TP**: 0.71200 (wide aspirational) | **Trail**: 20pip (real protection)
- **3-option**: C — Hold as-is. ATR 0.6×, Fib wave at start of new N-wave (AUD_USD Fib-3%).
- **UPL**: ~+1,330 JPY (+11.6pip). **Peak**: 11.6pip.

## Pending Orders
NONE — AUD_JPY LIMIT cancelled (id=466847). Price 112.23 = 53pip above LIMIT 111.700 with bullish momentum; if it filled, it would require a 53pip collapse = thesis broken context.

## Directional Mix
**3 LONG / 0 SHORT ⚠️ One-sided**
- All positions: USD-weak + risk-on theme. Justified by Iran ceasefire macro.
- Counter-trade assessed: H4 StRSI: USD_JPY=0.41, all others=1.0. USD_JPY LONG (bounce): H4 ADX=12 range = B only, margin 76.7% prevents adding. GBP_JPY/EUR_JPY SHORT: spreads 4.2pip/2.6pip too wide.
- Conclusion: no valid counter-trade opportunity today.

## Quality Audit Issues — Status
- ✅ AUD_USD LONG: Entered (id=466844). Fixed.
- ✅ AUD_JPY LIMIT: Placed then cancelled — price moved 53pip past LIMIT. H4 range + blow-off territory. Pass documented.
- **EUR_JPY LONG**: PASS. Spread 2.6pip (above normal 2.2pip max). Margin 76.7% limits new entries to B-size only. Enter if: spread ≤2.2pip AND margin <70%.
- **GBP_JPY LONG**: PASS. Spread 4.2pip (too wide vs normal 3.2pip max).

## 7-Pair Scan

### Tier 1 (held)
EUR_USD, GBP_USD, AUD_USD — see Positions above.

### Tier 2 Quick Scan
- **USD_JPY**: H1 StRSI=0.0 CCI=-298 extreme oversold. Sellers dominant M5 (making lower lows). H4 ADX=12 range. Bounce exists but H4 range = B conviction only. Enter LONG if: H4 ADX>20 AND H1 DI+ turn. Not now.
- **EUR_JPY**: H4+H1+M5 BULL. Spread 2.6pip (slightly wide). I would enter LONG if: spread ≤2.2pip AND pullback to 184.80 (H1 EMA20). H1 MACD div=0.6(bear) = fatigue signal, caution.
- **AUD_JPY**: H1 ADX=43 strong BULL. AUD strongest (+0.99). But price 112.23 = 112pip above H4 BB upper (111.106) = blow-off. LIMIT at 111.100 (H4 BB upper) if pullback comes. Spread 2.3pip normal.
- **GBP_JPY**: H1 ADX=38 BULL. Spread 4.2pip too wide (normal max 3.2pip). Pass.

## Capital Deployment
```
Current: marginUsed=91,415 / NAV=119,175 = 76.7%
Available to 85% cap: ~9,884 JPY
No new entries: all pairs 100pip+ above H4 BB upper. EUR_JPY spread 2.6pip still slightly wide.
Idle LIMIT deployment: AUD_JPY @111.100 if meaningful pullback (currently 112pip away — not placing now)
```

## Action Tracking
- Last action: 2026-04-07 23:15 UTC — AUD_JPY LIMIT id=466847 cancelled (53pip from LIMIT, unlikely to fill)
- Today's confirmed P&L: 0 JPY (no closed trades per OANDA)
- Next action trigger: EUR_USD TP@1.16940 OR GBP_USD TP@1.34100 OR trail hit → redeploy into EUR_JPY/AUD_JPY on spread normalization

## Lessons (Recent)
- [4/8] BE SL trap: AUD_JPY +1,200 JPY → BE SL → +40 JPY. HALF TP always beats BE SL.
- [4/8] Don't enter when price is 100pip above H4 BB upper. That's blow-off territory. LIMIT at pullback.
- [4/8] Quality audit "S-candidate missed" requires entry OR documented pass reason. Both satisfy the audit.
- [4/8] LIMIT at pullback: only makes sense if pullback TO LIMIT doesn't invalidate the thesis. AUD_JPY 53pip drop to 111.70 = thesis broken context. Cancel the LIMIT.
- [4/8] profit_check.py bug: regex @([\d.]+) captured trailing period from state.md line. Fixed to @([\d]+\.[\d]+).

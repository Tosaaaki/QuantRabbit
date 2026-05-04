# Professional Intermarket & Macro Context for FX Trading (2024-2026)
*Catalog for QuantRabbit — pairs: EUR_USD, AUD_JPY, EUR_JPY, GBP_USD, GBP_JPY*

## 1. Pair-by-pair intermarket map (what still works, what broke)

### EUR_USD
| Variable | Direction | Lead/lag | Signal strength 2024-26 |
|---|---|---|---|
| US10Y – Bund 10Y spread (DGS10 − DE10Y) | Spread ↑ → EURUSD ↓ | Coincident, 20-day rolling β very high through 2024 | Near-perfect through Q1 2025; **broke after April 2 2025** when tariff narrative caused USD to weaken despite widening US-Bund spread. Treat as "default ON, switch OFF when the move is USD-credibility driven." |
| US 2Y – Schatz 2Y spread | Same sign | Leads 10Y spread by ~1 day on policy days | Cleaner signal for short-horizon EUR/USD |
| DXY | Inverse, mechanical (~57% EUR weight) | Coincident | Always |
| ECB-Fed OIS implied terminal | Widens for USD → EURUSD ↓ | Leads spot 2-5 days around CB meetings | Strong in 2025-26 |
| Italian BTP-Bund spread (sovereign stress) | Widening → EURUSD ↓ (EUR risk premium) | Coincident | Activates only in stress |

**Regime-break flag**: when the US move is "USD-sell-America" (tariff/credibility/fiscal), yield-spread model inverts. Diagnose by checking whether DXY moves with or against the US-Bund spread.

### GBP_USD
| Variable | Direction | Notes |
|---|---|---|
| UK 10Y Gilt − US10Y spread | Spread ↑ → GBPUSD ↑ | BoE research: ~20bp gilt rise Jan-Sep 2025 was driven by global term premium + UK fiscal anxiety |
| SONIA OIS curve / BoE-implied easing | Less easing priced → GBP ↑ | Tight correlation in 2025 |
| UK fiscal headlines / DMO supply | Risk premium → uncontrolled gilt sell-off → uncontrolled GBP sell-off | Tail-risk regime |
| FTSE 100 / global equities | Risk-on positive (FTSE is also commodity proxy) | Weak |
| DXY | Mechanical (~12% weight) | Always |

### AUD_JPY (the cleanest risk barometer)
| Variable | Direction | Notes |
|---|---|---|
| S&P 500 / global equities | Same sign | Forex.com calls AUDJPY "essentially a proxy for everything"; rolling 20D corr typically +0.6 to +0.9 in stress |
| Iron ore / copper / LME base | AUD side: same sign | AUDUSD-iron ore ρ ≈ 0.88 historically |
| US10Y / DM rates | JPY side: rates ↑ → JPY ↓ → AUDJPY ↑ | The "carry" leg |
| China credit / CNY | AUD side: same sign as CNH | Important gating variable |
| VIX | Inverse (VIX ↑ → AUDJPY ↓) | Cleanest single-variable check |

**AUD_JPY is the single most informative cross we trade.** If you only watch one risk gauge, this is it.

### EUR_JPY
| Variable | Direction | Notes |
|---|---|---|
| Bund 10Y − JGB 10Y spread | Spread ↑ → EURJPY ↑ | JGB at 2.51% (May 2026) vs Bund 2.44% — historic compression |
| Risk appetite (S&P 500) | Same sign as AUDJPY but lower beta | Carry/risk leg |
| BoJ rate path (now at +1.0% projected end-2026) | JPY ↑ on hike repricing | The dominant 2024-26 driver |
| ECB terminal rate | EUR side | Secondary |

### GBP_JPY
Same drivers as EUR_JPY but louder. UK 10Y around 4.8% peak Sep 2025 → fattest carry in the JPY-cross complex → biggest unwind risk on risk-off. **Highest realized vol of our 5 pairs.**

### Universal regime-break flags
1. **JPY decoupling from yields** (mid-2024, Aug 2024 unwind, H2 2025): yen trades risk/carry instead of rate differential.
2. **USD decoupling from yields** (post-April 2 2025 tariff regime): dollar weakens on rising US yields when narrative is fiscal/credibility.
3. **AUD decoupling from China commodities** (when domestic RBA path dominates).

---

## 2. Rate differentials & central bank stance

**Front of curve (2Y) for policy, belly (5Y) for cycle, long end (10Y/30Y) for term premium / fiscal.**

- **2Y-2Y spread** = pure expected-policy differential. Best for FX over 1d-1w horizon.
- **5Y-5Y forward spread** = "where will rates be in 5y, 5y from now?" — best long-horizon valuation anchor.
- **OIS-implied terminal rate**: CME FedWatch (Fed), Reuters/Eikon, or scrape SOFR/SONIA/ESTR/TONA futures.
- **Real yields (TIPS-based)**: DGS10 − T10YIE on FRED. Real yield diff often dominates nominal in unstable inflation regime.

**Current Fed–ECB gap is ~160bp** (Fed 3.50-3.75, ECB 2.15) per March 2026. **BoJ at +1.0% projected end-2026** — the biggest policy delta.

**Retail-feasible feeds**:
- FRED API: DGS2, DGS5, DGS10, DGS30, T10Y2Y, T10YIE, DFEDTARU, DFF (free)
- ECB Data Portal: yield_curve datasets (free)
- BoE Statistical: yield curve data (free)
- JGB: MoF Japan publishes JGB curve daily (CSV, free)
- OANDA CFD only covers US10Y_USD and a few European curves

---

## 3. Risk-on / risk-off composite

**The "Macro Risk Trinity"**:
- **Equity vol**: VIX (FRED `VIXCLS`)
- **Rates vol**: MOVE Index (CBOE; or use TLT IV / 30Y futures realized vol as proxy)
- **Credit**: ICE BofA HY OAS (FRED `BAMLH0A0HYM2`)

**FX-specific additions**:
- Gold/Copper ratio (FRED `GOLDPMGBD228NLBM` / `PCOPPUSDM`) — at 50-year extreme in 2025 = structural risk-off
- USDJPY (yes, itself) — increasingly a fear gauge
- AUDJPY — clean cross-asset proxy
- HYG/LQD ratio (Yahoo Finance) — credit risk-on/off
- USDKRW or USDZAR — EM stress

**Simple composite (free data)**:
```
risk_score = z(VIX, 60d)·-1 + z(HYG/LQD, 60d) + z(SPX_60d_return)
           + z(AUDJPY_60d_return) + z(Copper/Gold, 60d) + z(US2Y_change, 60d)·-1
```
Each component normalized to 60-day z-score, equal-weighted, clipped to ±3. Score > +1 = risk-on extreme (fade JPY/CHF longs), < −1 = risk-off (size up JPY/CHF longs, cut AUD/GBP).

The MOVE/VIX divergence is the highest-signal warning: high MOVE + calm VIX = "bond market is pricing a Fed/inflation problem the equity market hasn't noticed yet."

---

## 4. CFTC COT — actually useful signals

**Reports that matter**: TFF (Traders in Financial Futures) Combined, weekly. Released Friday 3:30pm ET, reflecting Tuesday positioning.

**For each currency future (6E, 6B, 6J, 6A) compute**:
1. **Leveraged Funds net %** of OI = (long − short) / total OI
2. **Z-score / percentile rank** over 1Y and 3Y rolling window
3. **Week-on-week change** (delta) — the "flow" signal
4. **Commercial / Dealer net** — contrarian to leveraged

**How professionals use it (honest take)**:
- **Extreme percentile (>90 or <10)** = setup, not trigger. Wait for **a) extreme + b) price reversal + c) week's delta turning** for signal.
- **Time horizon**: 2-8 weeks — too slow for scalp, useful as **regime/bias overlay** for swing.
- **Week-over-week delta** more actionable than absolute level for shorter horizon.
- **Commercials** are usually contrarian to specs.

**Honest critique**: COT is futures-only (~10% of FX volume; spot/forward dominates). It misses corporate hedging, EM CB flows, big asset-manager spot trades. Best treated as **one of three sentiment inputs** alongside retail SSI and risk-reversal options skew.

**Free API**:
- Socrata: `https://publicreporting.cftc.gov/resource/gpe5-46if.json` (TFF Futures Only)
- `https://publicreporting.cftc.gov/resource/jun7-fc8e.json` (TFF Combined)
- Python lib: `cot_reports` (NDelventhal/cot_reports)

---

## 5. Retail sentiment (OANDA / IG / FXSSI / MyFXBook)

**The honest answer**: retail positioning is a **moderately useful contrarian** signal in **trending** markets, **noise** in chop.

- **IG Client Sentiment (IGCS)**: free; works as contrarian
- **OANDA Open Orders / Positions**: shows price-level distribution — best used to spot **stop clusters** for intraday liquidity raids
- **MyFXBook Community Outlook**: similar contrarian signal

**Usage rule of thumb**: only fade retail when (a) extreme (≥70/30 split), (b) price already moving against retail, (c) confirmed by COT or risk-reversal.

---

## 6. Calendar — beyond `in_window`

**Tier scoring**:

| Tier | Events | Window before | Window after | Size impact |
|---|---|---|---|---|
| S (paramount) | FOMC decision, NFP, US CPI, BoJ + MoF intervention windows | 30-60 min | 60-180 min | Skip or 25% size |
| A (high) | ECB, BoE, BoJ rate decisions; US Core PCE; UK CPI; AU CPI; Powell/Lagarde testimony | 15-30 min | 60 min | 50% size |
| B (medium) | PMIs (S&P Global flash), Retail Sales, ADP, German IFO/ZEW, JOLTS, Tankan | 10 min | 30 min | 75% size |
| C (low) | Trade balance, housing, minor surveys | 5 min | 10 min | Normal |

**Pre-event drift**: empirically tiny. Don't position into the print on that signal.

**Post-event volatility pattern**: vol compresses 30-90 min before, explodes at release, takes 2-4 hours to settle. **First spike often retraces** (the "head fake"). Wait 5-15 min for direction to confirm.

**"Buy rumor, sell fact"**: works on rate-decision days when path is fully priced. Diagnose by checking OIS-implied probability — if ≥85% priced going in, expect mean-reversion on the print.

---

## 7. JST holidays — actual impact

| Period | Dates | Effect |
|---|---|---|
| Golden Week | Apr 29 – May 5/6 | Tokyo closed. JPY crosses: thin liquidity, **wider spreads (often 2-3x normal)**, amplified moves. MoF intervention statistically more likely. EURUSD/GBPUSD: minor spillover unless intervention |
| Obon | ~Aug 13-16 | Same pattern, milder. JP institutional flow halved |
| Year-end / New Year | Dec 28 – Jan 3 | Worst FX liquidity of year. Skip JPY crosses Dec 30 – Jan 2 |
| Children's Day (May 5) | Single day | Tail end of GW; combined with UK Bank Holiday in 2026 → double-thin window |

**What practitioners do**:
1. **Skip JPY pairs** Apr 29 – May 6 except for explicit intervention plays.
2. **Wait for London open** (~16:00 JST/8:00 BST) when Tokyo closed.
3. **Reduce size 50-70%** even on EUR_USD during these windows.
4. **No tight stops** — codified in memory `feedback_no_tight_sl_thin_market`.

---

## 8. Currency strength index — methodology

**Three methods**:

1. **Pair-averaged returns** (simplest): for each currency C, average its return vs the other 7 G8 pairs. Easy to game by single-pair noise.
2. **Index-weighted** (DXY-style): trade-weighted basket weights. More stable.
3. **PCA-based** (institutional): take all G10 returns, extract first principal component. Less gamed, but requires rolling estimation.

**Recommendation**: **median pair return over 7 majors** (median, not mean — robust to one-pair shocks) on rolling 1h, 4h, 1d windows.

**What strength rank actually predicts**:
- **1h horizon**: noise dominates, low predictive power.
- **1d horizon**: weak positive autocorrelation (~0.1-0.2) — modest momentum signal.
- **1w horizon**: best edge — the rank tends to persist 3-7 days.
- **Cross signal**: trade strongest vs weakest is the canonical setup; expected edge ~5-15 pips/day on G10 majors after costs.

---

## 9. Free / cheap data sources

| Source | What | Endpoint |
|---|---|---|
| FRED | Yields, real rates, vol indices, credit spreads | `https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key=KEY&file_type=json` |
| CFTC Socrata | COT all formats | `https://publicreporting.cftc.gov/resource/gpe5-46if.json` |
| Yahoo Finance (yfinance) | Equity indices, ETFs (HYG/LQD/TLT), futures | yfinance Python lib |
| ECB Data Portal | EUR yield curve, ECB rates | `https://data-api.ecb.europa.eu/service/data/...` |
| BoE Statistical | Gilt curve, BoE rate | Statistical Interactive DB |
| MoF Japan | JGB curve | mof.go.jp daily CSV |
| OANDA v20 | Spot FX, CFDs (US10Y_USD, DE10YB_EUR, SPX500, JP225, BCO/USD, WTICO/USD, XAU/XAG, NATGAS) | Existing API |
| TradingEconomics | Calendar, indicators | API (paid for full) |
| CME FedWatch | Fed rate probabilities | cmegroup.com |

---

## 10. Concrete confluence examples

**EUR_USD SHORT high conviction**: DXY breaking 1Y range top + US10Y–Bund10Y spread widened +12bp in 5d + S&P -1.2% with USDJPY UP + EU flash PMI miss + Leveraged Funds COT still net-long EUR at 87th %ile.

**EUR_USD LONG contrarian**: US10Y–Bund10Y narrowed −20bp in 10d + DXY rejected at 1Y high + ECB hawkish surprise + Leveraged Funds COT net-short EUR at 12th %ile + retail (IGCS) net-long.

**USD_JPY SHORT (proxy for our JPY crosses)**: US10Y rolled −15bp + JGB10Y broke 2.50% upside + MoF verbal intervention escalation + Golden Week thin liquidity + leveraged funds COT short JPY at 5th %ile + risk score < −1. **High intervention-asymmetry**.

**AUD_JPY SHORT, classical risk-off**: VIX > 22 (60d 90th %ile) + S&P 500 −1.5% intraday + HY OAS widened +20bp + copper −2% + USDJPY DOWN. **Cleanest expression**.

**AUD_JPY LONG, risk-on continuation**: Copper at $11k+ LME + S&P new high + VIX <14 + China stimulus surprise + AUD COT specs net-short.

**GBP_USD LONG, BoE-driven**: SONIA OIS pricing OUT 25bp of cuts in past 5d + UK CPI surprise hot + DXY soft + UK 10Y gilt − US10Y narrowed only modestly.

**GBP_JPY SHORT, asymmetric carry unwind**: UK gilts selling off on fiscal headlines + S&P -1% + USDJPY breaking down + GBPJPY at 1Y high. **Highest beta JPY cross**.

**EUR_JPY SHORT, BoJ catalyst**: BoJ minutes hawkish + JGB10Y > Bund10Y for first time in cycle + EU PMIs softening + ECB OIS pricing more cuts.

**EUR_USD SHORT, "tariff regime" check**: DXY breakout + US-Bund spread widening — BUT **check first**: is USD moving WITH yields (yes → trade the breakout) or AGAINST yields (no → April-2-2025-style USD-credibility regime, fade the breakout)?

---

## 11. Operational recommendations

1. **Add 8 FRED series to nightly pull**: DGS2, DGS10, T10Y2Y, T10YIE, BAMLH0A0HYM2, VIXCLS, DEXUSEU, GOLDPMGBD228NLBM.
2. **Add CFTC TFF Combined weekly pull**; compute z-score & percentile per currency.
3. **Build composite risk score** — single scalar fed to trader prompt.
4. **OANDA CFDs to actively monitor**: `US10Y_USD`, `DE10YB_EUR`, `SPX500_USD`, `JP225_USD`, `WTICO_USD`, `XAU_USD`.
5. **Rate-differential dashboards**: pre-compute (US2Y−DE2Y), (US10Y−DE10Y), (US10Y−JGB10Y), (UK10Y−US10Y).
6. **JST-holiday calendar overlay**: Golden Week Apr 29-May 6, Obon Aug 13-16, year-end Dec 28-Jan 3.
7. **Pair-level intermarket scorecard**: top 3 drivers + their current z-score + the regime-break check.
8. **Tier-S/A event calendar**: replace boolean `in_window` with `event_tier ∈ {S,A,B,C,none}` and `minutes_to_event`.

The single biggest information gap today is the **USD-credibility regime check** — knowing when the yield-spread model has flipped sign. That's a one-line diagnostic (sign of 5d-rolling β between DXY change and US10Y change).

---

## Sources

- [USD/JPY H2 2025 Forecast - FOREX.com](https://www.forex.com/en-us/news-and-analysis/usd-jpy-h2-2025-forecast-correlation-breakdown-political-risks-a/)
- [US/Germany 10Y Yield Spread vs EUR/USD - MacroMicro](https://en.macromicro.me/collections/2204/euro-dollar/17798/euro-us-germany-10y-treasury-note-spread)
- [AUDJPY The FX Stock Market Correlation Play - FOREX.com](https://www.forex.com/en-uk/news-and-analysis/audjpy-the-fx-stock-market-correlation-play/)
- [AUD/JPY: The Ultimate Risk Barometer - Investing.com](https://www.investing.com/analysis/aud-jpy---the-ultimate-risk-barometer-200176289)
- [Macro Risk Trinity OAS|VIX|MOVE - TradingView](https://www.tradingview.com/script/rDsRnbFT-Macro-Risk-Trinity-OAS-VIX-MOVE/)
- [VIX FRED Data Series VIXCLS](https://fred.stlouisfed.org/series/VIXCLS)
- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/fred/)
- [Commitments of Traders - CFTC](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)
- [TFF Futures Only Socrata API](https://dev.socrata.com/foundry/publicreporting.cftc.gov/gpe5-46if/embed)
- [cot_reports Python library](https://github.com/NDelventhal/cot_reports)
- [Currency strength index - Wikipedia](https://en.wikipedia.org/wiki/Currency_strength_index)
- [Forex Factory: JN Bank Holiday calendar](https://www.forexfactory.com/calendar/393-jn-bank-holiday)
- [Yen Carry Trade Unwinding](https://discoveryalert.com.au/yen-carry-trade-unwinding-risks-2025/)
- [What were the drivers of UK long-term interest rates - Bank of England](https://www.bankofengland.co.uk/bank-insights/2026/what-were-the-drivers-of-uk-long-term-interest-rates-in-2025)
- [Copper-to-Gold Ratio Economic Indicator](https://gold-standard.org/insights/the-copper-to-gold-ratio-a-powerful)

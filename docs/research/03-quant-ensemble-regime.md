# Quantitative Ensemble Methods for FX Regime Classification & Trade Signals — Research Brief (2024–2026)

Audience: a Claude-operated discretionary FX trader currently consuming ~15 indicators per pair/timeframe naively. Goal: replace "list values, vibe-check" with structured ensemble logic.

---

## 1. Regime classification — what works on OHLC, what fails on FX

| Method | OHLC-only? | FX viability | What it tells you | Notes |
|---|---|---|---|---|
| Hurst exponent (R/S, DFA, Generalized Hurst) | Yes | High on H1+, weak intraday | H>0.55 trending, H<0.45 mean-rev, ~0.5 random walk | R/S has upward small-sample bias — use 0.55/0.45 not 0.50/0.50 |
| Variance Ratio test (Lo-MacKinlay) | Yes | High | Confirms whether VR(q)≠1 is statistically significant | Use as a stat sig guard around Hurst |
| ADF / KPSS | Yes | Moderate | Stationarity of the *level* series | FX prices are I(1); apply to spread/residual series, not raw price |
| Choppiness Index (14) | Yes | High | <38.2 trending, >61.8 chop, 38.2–61.8 transitional | Cheap, robust |
| ADX (14) | Yes | High | >25 trend, <20 range; rising vs falling matters more than level | Best treated as a regime gate, not a direction signal |
| HMM (Gaussian / non-homogeneous) | Yes | Moderate–high | Probabilistic 2–4 state regime | Non-Homogeneous HMM (time-varying transitions) outperforms vanilla HMM on FX |
| Gaussian Mixture Model on (return, vol) | Yes | Moderate | Soft-cluster current bar into 2–3 regimes using BIC | Lighter than HMM; no temporal smoothing |
| BOCPD (Bayesian Online Change-Point) | Yes | High | Posterior probability that a regime break just occurred | Outperforms GLR/KS in 2024–2025 work. Hazard λ≈100 reasonable on daily; λ≈300–500 on M5 |
| CUSUM / Page-Hinkley | Yes | High | Cheap "something changed" alarm on returns or vol | Use as fast confirm |
| GARCH(1,1) / Realized GARCH on RV | Needs HF | High | Conditional vol forecast and persistence | Realized-GARCH (HAR-RV style) dominates plain GARCH at intraday |
| HV percentile rank (rolling 1y) | Yes | High | "Is current ATR/RV in 0–20 / 20–80 / 80–100 percentile?" | The single highest-leverage cheap regime feature |

**Practical recipe for QuantRabbit**: **Hurst(rolling 200 bars, threshold 0.45/0.55) + Choppiness(14) + ADX(14) + HV percentile (ATR/close, 1y rolling)** is enough to define a 4-state regime (TREND_STRONG, TREND_WEAK, RANGE, BREAKOUT_PENDING) without an HMM. Add HMM only if you want soft probabilities and have ≥2y of bar data per pair.

Recent work:
- "Incorporating improved directional change and regime change detection in FX" (Physica A 2023)
- Macrosynergy — "Detecting trends and mean reversion with the Hurst exponent"
- "Bayesian Online Changepoint Detection for Financial Time Series" (2025, ICCSMT)
- Two Sigma — "A Machine Learning Approach to Regime Modeling"

---

## 2. Indicator confluence scoring — mapping 15 indicators to scores without overfitting

The naive failure: treating 15 indicators as 15 votes. RSI, Stoch, CCI, Williams %R, MFI, ROC are *the same momentum factor*. **Empirically, the rank correlation between RSI(14), Stoch(14,3,3), and CCI(20) on FX M5 is typically 0.75–0.92.**

**Step 1 — group by what they actually measure:**

| Family | Indicators | What it answers |
|---|---|---|
| Trend direction | EMAs, MACD signal, Aroon, SuperTrend | Up / down / neutral |
| Trend strength | ADX, Choppiness, Aroon spread | Trend present / absent |
| Momentum | RSI, Stoch, CCI, ROC, Williams %R, MFI, MACD histogram | Overbought / oversold / accelerating |
| Volatility | ATR, BB width, Donchian width | Expanding / contracting, percentile |
| Location / structure | BB %B, Donchian position, VWAP distance, Ichimoku Kumo position, BOS/CHOCH | Where in the range, broken structure? |

Every score should average *within a family first*, then combine across families. Otherwise the 6-momentum-indicators bloc dominates.

**Step 2 — rank/Z-score normalize, don't use raw values.** RSI 70 means different things on EUR_USD M5 vs AUD_JPY H1. Compute a rolling Z-score (or percentile rank over 500 bars) per indicator per pair per timeframe. **This is the single biggest win for an LLM trader citing absolute values.**

**Step 3 — three composite scores:**

```
TrendScore = mean( z(price - EMA50)/ATR, z(MACD_hist), sign(SuperTrend), z(Aroon_up - Aroon_down) )
MeanRevScore = mean( z(BB %B - 0.5), z(RSI - 50), z(Stoch - 50), z(VWAP_dist) ) * (-1 if extended)
BreakoutScore = mean( z(BB_width pct rank), z(ATR pct rank), 1{Donchian_break}, 1{BOS} )
```

Direction = sign(TrendScore). Conviction = |TrendScore| if regime=TREND, else |MeanRevScore| if regime=RANGE, else |BreakoutScore| if regime=BREAKOUT_PENDING. **The regime gate decides which score is read.** This is the single most underused structure in retail systems.

**Step 4 — agreement-disagreement metric.** Compute σ across the family-scores. Low σ + strong sign = clean setup. High σ = mixed market = stand aside. Concretely: if TrendScore > 0.7 and MeanRevScore < -0.7 → no-trade, not "average them to zero."

Tree-based feature importance (XGBoost / LightGBM with SHAP) consistently identifies ATR-percentile, MACD histogram, and EMA-distance as the durable winners; RSI/Stoch/CCI are highly redundant.

---

## 3. Statistical filters on the price stream itself

| Filter | Recipe | What it tells you |
|---|---|---|
| Lag-1 return autocorr (rolling 100) | `corr(r_t, r_{t-1})` | >0.05 → momentum/trend; <-0.05 → mean-revert; |·|<0.02 → random walk |
| ACF of \|r\| at lags 1..20 | Should be slow-decay positive | If absent → low vol clustering, breakouts less likely |
| Ljung-Box on r (lag 10) | p<0.05 → autocorrelation present | Sanity-check for trend regimes |
| ARCH-LM | p<0.05 → vol clustering | Default true on FX; alarm if it disappears |
| Variance Ratio VR(q) for q∈{2,4,8,16} | If VR(q)>1 trend; <1 mean-rev | Multi-horizon view |
| Rolling skew/kurt (window 200) | Excess kurt >5, |skew|>1 → fat-tail/jump regime | Cuts size in tail-heavy windows |
| **Lee-Mykland jump test** | `L_t = r_t / σ̂_t` where σ̂ from bipower variation; threshold ~ Gumbel | Flags news-spike bars; do not enter for k bars after |
| BNS bipower variation (jump share) | (RV − BV)/RV | If >0.3 recently → jumpy regime, widen stops |
| Flat-spot detector | count of bars with |r| < ε in last N | High flat count + tight BB = squeeze pre-break |
| Microstructure noise (signature plot) | RV at 1m vs 5m vs 15m | Inflation at 1m → noisy quotes; trade slower |
| Hurst on returns (rolling 200) | DFA exponent | Independent of price level, FX-robust |

**For QuantRabbit, cheapest valuable additions:** lag-1 autocorr, |r| ACF decay, rolling kurtosis, Lee-Mykland on the last 50 bars. These four catch most "market is broken right now" states that the 15-indicator panel misses.

Lee-Mykland reference: Lee & Mykland 2008 *Review of Financial Studies* — gold standard for nonparametric jump detection at intraday frequencies.

---

## 4. Multi-timeframe ensembling for FX intraday

The mature pattern is hierarchical, not voting:

1. **Bias** (D1 / H4): direction permission + regime tag.
2. **Framework** (H1): location and structure — VWAP, prior day H/L, daily-pivot levels, swing structure (BOS/CHOCH).
3. **Trigger** (M15 / M5): entry timing — momentum cross, BB %B reversal, Donchian break.
4. **Risk** (M1 / tick): execution and SL placement using ATR(M5) × k.

Voting is the wrong frame: the higher TF is a *gate*, not a peer. If H1 disagrees with M5, you do not trade — period — even if 4/5 lower-TF indicators agree.

For confluence inside that hierarchy, a defensible rule is:

```
trade_allowed = (H1_regime == M15_regime)            # regime alignment
            and sign(H1_TrendScore) == sign(M5_TriggerScore)  # direction alignment
            and (H1_ADX > 20 or M15_BBwidth_pct > 60)         # tradable structure
            and not Lee_Mykland_jump_in_last_5_bars            # no noise spike
            and conviction(M5) >= threshold                    # trigger strength
```

Key principle: **disagreement → no-trade**, never "average it out." Dropping disagree-no-trade is the #1 cause of regime brittleness in retail confluence systems.

---

## 5. Feature engineering for FX trader inputs

| Feature | Recipe | Why |
|---|---|---|
| Multi-horizon returns | r_5, r_15, r_60, r_240, r_1440 (minutes) | Scale-aware momentum |
| Vol-normalized returns | r_h / ATR_h | Comparable across pairs |
| Rolling cumulative return | sum(r) over last N | Trend strength independent of indicator parameters |
| BB squeeze/expansion | BB_width / BB_width.rolling(100).mean() | <0.7 = squeeze; >1.3 = expansion |
| ATR percentile (1y) | rank(ATR_14, lookback=252×bars/day) | Single best vol-regime number |
| Realized correlation | corr(r_pair_A, r_pair_B, window=100) | Cross-pair structure |
| USD strength index | mean of normalized log-returns of USD vs other 7 majors; or DXY (EUR 57.6%, JPY 13.6%) | Disambiguates EUR_USD move = EUR strong vs USD weak |
| JPY strength index | same vs JPY | Critical for EUR_JPY / AUD_JPY decomposition |
| Risk-on/off proxy | corr(JPY_strength, AUD_strength) flipped sign; or AUD_JPY vs S&P futures | Known FX driver |
| DXY momentum | DXY EMA-distance and ROC | Cross-asset confirmation for any USD pair |
| Cross-pair divergence | EUR_USD vs GBP_USD residual after rolling beta | Idiosyncratic vs USD-driven moves |

For AUD_JPY and EUR_JPY specifically, decomposing into (JPY strength) × (AUD or EUR strength) tells you whether to size up: when *both* sides are pushing the same way, the cross moves cleanly.

---

## 6. Win rate / expectancy estimation

The honest answer: a single backtest Sharpe is mostly noise. Bailey & López de Prado's program is now standard:

- **Walk-forward** is the *minimum* bar.
- **Combinatorially Symmetric Cross-Validation (CSCV)** → **Probability of Backtest Overfitting (PBO)** → **Deflated Sharpe Ratio (DSR)** is the rigorous stack.
- **Combinatorial Purged CV (CPCV)** beats walk-forward (López de Prado, *Advances in Financial ML*, 2018).
- **Bootstrap of trade-level returns** is fine for per-setup expectancy *when you have ≥100 trades*. With <30 trades, expectancy is a guess.

Quoted rule of thumb (Bailey): backtest Sharpe ≥3 is almost certainly fake; live Sharpe >1.2 is rare; >1.5 is usually overfit. For per-setup metrics on QuantRabbit, report (trade count, win rate, mean R, median R, deflated mean R) — never just win rate.

For an LLM trader, the practical version is: maintain a journal of (setup tag → outcome in R-multiples). Bucket by regime tag. Report rolling 30-trade win rate AND require ≥30 samples before quoting it. Below 30, say "insufficient data."

---

## 7. The honest "more indicators = better" critique

1. **Collinearity inflates fake-confidence.** Six momentum indicators agreeing is one observation, not six. VIFs on FX indicator panels routinely exceed 10.
2. **Marginal information after ~5–7 indicators is near zero.**
3. **Regime brittleness.** Rules tuned on 2018–2022 EUR_USD ranges break in 2023–2025.
4. **Selection bias on the indicator set itself.**
5. **Live Sharpe almost always disappoints backtest Sharpe by ~50%.**

The 2024 practitioner consensus: a small panel (5–8 features), grouped by family, normalized, with a regime gate, beats a 15-indicator panel reasoned about narratively.

---

## 8. Concrete confluence rules

**A. ADX + MA trend-following:** ADX(14) > 25 AND ADX rising; price above EMA(14) for long; entry on first pullback to EMA after ADX cross of 25; Stop: 1× ATR(14); target 2× ATR.

**B. EMA-9/21 + RSI + ADX triple-confirm:** LONG: EMA9 cross above EMA21, RSI > 55, ADX > 20; SHORT: EMA9 cross below EMA21, RSI < 45, ADX > 20.

**C. EMA-8/34 + MACD + RSI band:** LONG: EMA8 > EMA34 AND MACD > MACD_signal AND 45 ≤ RSI ≤ 70; SL = entry − 1.5×ATR(14), TP = entry + 2.0×ATR(14).

**D. Choppiness + Bollinger squeeze breakout:** Choppiness(14) crosses below 38.2; BB(20,2) width in bottom 20% percentile of last 100 bars; trigger: close outside BB after squeeze.

**E. Volatility-Regime Classifier composite (TradingView QuantRegime):** Combines Hurst, ADX, Choppiness into 8-state regime. Trade trend in TREND_STRONG (Hurst>0.55, ADX>25, Chop<38.2). Trade mean-reversion in RANGE_LOW_VOL (Hurst<0.45, ADX<20, Chop>61.8, ATR-pct<40). Stand aside in TRANSITION.

**F. Macrosynergy rolling Hurst rule:** Rolling H over 252-day DFA; long trend strategies when H > 0.55 for ≥10 days; long mean-rev when H < 0.45 for ≥10 days.

**G. Multi-TF hierarchical:** D1 HH/HL or LH/LL → bias; H4 supply/demand zone in bias direction; H1 VWAP touch or pivot tag inside zone; M15 candle structure + RSI cross 50; M5 entry on micro-BOS.

---

## Suggested implementation order for QuantRabbit

1. **Add normalization layer** — convert every indicator value to a rolling Z-score and percentile rank per pair per TF. **This alone fixes most "RSI 70 means…" naive citation.**
2. **Define regime gate** — Hurst(200) + ADX(14) + Choppiness(14) + ATR-percentile (1y) → 4-state tag per TF.
3. **Three family scores** — Trend / MeanRev / Breakout, each averaged within family first.
4. **Hierarchical disagree-no-trade** — H1 regime must equal M15 regime; H1 TrendScore sign must equal M5 trigger sign.
5. **Add price-stream filters** — lag-1 autocorr, ATR-pct, |r| ACF decay, Lee-Mykland jump flag.
6. **Add cross-currency context** — USD strength, JPY strength, AUD-JPY vs equity-risk proxy.
7. **Journal with R-multiples** — bucket by regime tag, require 30 trades before quoting per-setup win rate.
8. **Skip HMM/PCA/GMM until #1–7 are live.**

---

## Sources

- [QuantConnect: Intraday Application of Hidden Markov Models](https://www.quantconnect.com/research/17900/intraday-application-of-hidden-markov-models/)
- [Macrosynergy — Detecting trends and mean reversion with the Hurst exponent](https://macrosynergy.com/research/detecting-trends-and-mean-reversion-with-the-hurst-exponent/)
- [Two Sigma — A Machine Learning Approach to Regime Modeling](https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/)
- [BOCPD — Gundersen blog tutorial](https://gregorygundersen.com/blog/2019/08/13/bocd/)
- [Bayesian AR online change-point with time-varying parameters (arXiv 2407.16376)](https://arxiv.org/html/2407.16376v1)
- [Lee & Mykland — Jumps in Financial Markets (RFS)](https://galton.uchicago.edu/~mykland/paperlinks/LeeMykland-2535.pdf)
- [QTMRL: Multi-Indicator RL Trading Agent (arXiv 2508.20467)](https://arxiv.org/html/2508.20467v1)
- [Bailey & López de Prado — Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- [Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- [Multi Timeframe Trading — MindMathMoney](https://www.mindmathmoney.com/articles/multi-timeframe-analysis-trading-strategy-the-complete-guide-to-trading-multiple-timeframes)
- [Choppiness Index — LuxAlgo](https://www.luxalgo.com/blog/choppiness-index-quantifying-consolidation/)
- [Volatility Regime Classifier (TradingView)](https://www.tradingview.com/script/zagpmoKH-Volatility-Regime-Classifier-QuantRegime/)
- [Currency Strength Meter — Babypips MarketMilk](https://marketmilk.babypips.com/currency-strength)
- [DXY composition (TradingView)](https://www.tradingview.com/symbols/TVC-DXY/)

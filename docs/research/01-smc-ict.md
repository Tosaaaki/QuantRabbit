# SMC / ICT Reading Layer — Catalog for QuantRabbit Trader

Audience: a discretionary FX trader operating EUR_USD, AUD_JPY, EUR_JPY, GBP_USD, GBP_JPY on OANDA, who already has the indicator stack but lacks the price-action narrative. Goal: define the primitives precisely enough to (a) extract them from OHLC in Python and (b) embed them as a "what to read" checklist in `SKILL_trader.md`.

## 1. The SMC/ICT primitive set actually used in 2025-2026

ICT/SMC is essentially price action with a vocabulary of liquidity. Strip the marketing and what remains is a small, surprisingly disciplined grammar: structure (BOS/CHOCH), supply/demand zones (OB/Breaker/Mitigation), inefficiencies (FVG), liquidity targets (EQH/EQL, session H/L, PDH/PDL), and time windows (killzones). All of them are derivable from OHLC; none of them require tape data.

### 1.1 Structure primitives

| Primitive | Definition (OHLC-detectable) | Native TF | Confluence | Common misuse | Strength |
|---|---|---|---|---|---|
| **Swing High/Low** | Pivot of length N (typically 3–5 bars on either side) where high is local max / low is local min. | All | The atomic unit; everything else is derived | Using N too small → noisy structure | n/a |
| **BOS (Break of Structure)** | After a confirmed swing in trend direction, price closes beyond the most recent swing extreme in same direction. `close_break=True` is the rigorous form. | All; confirmed on H1+ for bias | Trend continuation; only valid AFTER a CHOCH or in established trend | Calling BOS on a wick-only break; ignoring HTF context | Medium alone, high with displacement |
| **CHOCH (Change of Character)** | First break of a counter-trend swing. In an uptrend: first close below the most recent Higher Low. Reverses bias. | M15+ for intraday, H4+ for swing | Marks the regime flip; the very first signal of a reversal | Treating every CHOCH as a reversal entry — most mean-revert | High signal; low edge alone |
| **Internal vs External Structure** | External = HTF swings (H4/D1). Internal = LTF swings between two external swings. | External H4/D, Internal M15/H1 | "Trade with the external trend, against internal pullbacks" | Confusing internal CHOCH with external trend reversal | High when used as filter |

### 1.2 Supply/demand zones

| Primitive | Definition (OHLC-detectable) | TF | Confluence | Misuse | Strength |
|---|---|---|---|---|---|
| **Order Block (OB)** | Last opposing candle before a displacement that causes a BOS. Bullish OB = last down-candle before strong up-move that breaks structure. Zone = candle's open→low (bullish) / open→high (bearish). | All; HTF OBs (H4/D) are highest probability | Must have (a) displacement out, (b) break of recent structure, (c) ideally an FVG attached | Marking every red candle as an OB; using OBs that have already been mitigated | Medium; high with FVG+sweep |
| **Breaker Block** | An OB that failed: price swept through the original OB, then the structure reversed. Defining trait: a liquidity sweep occurred on the OB's TF. | All | Liquidity sweep + structure shift | Confused with mitigation block | High — built on a confirmed stop-hunt |
| **Mitigation Block** | Like a breaker but no liquidity was swept before the reversal. Lower probability. | All | Counter-trend continuation zones | Treating identically to breakers | Low–Medium |
| **Propulsion Block** | A re-entry zone formed when price returns to a prior OB and respects it again with displacement. | M15+ | Trend continuation | Over-fitting; many "propulsions" are just retests | Medium |

### 1.3 Inefficiency / imbalance

| Primitive | Definition | TF | Confluence | Misuse | Strength |
|---|---|---|---|---|---|
| **Fair Value Gap (FVG)** | Three-candle pattern where candle 1's high < candle 3's low (bullish) or candle 1's low > candle 3's high (bearish). The gap = the unfilled zone. Middle candle must be the displacement. | All — FVGs on H1+ are tradeable, M1/M5 FVGs decorate not direct | "Internal range liquidity" — price is drawn back to fill them | Trading every FVG; ignoring that ~30–45% never produce tradeable reactions per recent academic work (Kondapally 2026) | Medium individually; high as confluence inside an OB |
| **Inversion FVG (iFVG)** | An FVG that was filled and is now respected as opposing S/R. | M5–H1 | Like a breaker block but for inefficiency | Calling unfilled FVGs "inverted" | Medium |
| **Volume Imbalance / Gap** | Single-candle gap between body of candle N-1 and body of candle N (open ≠ prior close). Less common in 24h FX except at weekly open. | All | Confluence to FVG | Rarely usable in spot FX | Low |

### 1.4 Liquidity primitives

| Primitive | Definition | TF | Confluence | Misuse | Strength |
|---|---|---|---|---|---|
| **Equal Highs / Equal Lows (EQH/EQL)** | Two or more swing highs/lows within ATR×0.1 of each other. Stops cluster just beyond. | M15+ | Magnet for sweeps | Confusing range top with EQH; tolerance too tight or too loose | High as target |
| **Liquidity Sweep / Stop Hunt** | Wick beyond a prior swing extreme/EQH/EQL/session high/PDH that closes back inside the prior range. Programmatic: `high > prior_swing_high AND close < prior_swing_high` (bearish sweep). | All | The single most repeatable ICT pattern. Combine with CHOCH on LTF | Trading the sweep without LTF confirmation; sweeps in trending markets often continue | High |
| **External Range Liquidity (ERL)** | Liquidity above/below the dealing range — prior swing highs/lows of the HTF. | H4/D | Targets after BOS | n/a | High |
| **Internal Range Liquidity (IRL)** | FVGs and minor swings inside the dealing range. | M15/H1 | Targets in consolidation | n/a | Medium |
| **PDH / PDL / PWH / PWL** | Previous Day/Week High and Low. Always-watched draws on liquidity. | D/W | Universal magnets | Treating as static S/R rather than liquidity targets | High |
| **Session H/L** | High and low of Asian / London / NY sessions. | M5–H1 | Sweep targets in subsequent sessions | n/a | High |

### 1.5 Direction / framing primitives

| Primitive | Definition | TF | Notes |
|---|---|---|---|
| **Dealing Range** | Most recent significant HTF swing-high to swing-low. Defined by N-bar pivots on H4/D. | H4/D | Required to compute premium/discount |
| **Equilibrium** | 50% of dealing range. | derived | Sells in premium (above 50%), buys in discount (below 50%) |
| **Premium / Discount Zones** | Above 50% = premium (sell territory), below 50% = discount (buy territory). | derived | Bias filter, not a signal |
| **Optimal Trade Entry (OTE)** | 0.62–0.79 Fib retracement of last impulse leg, sweet-spot 0.705. | M5–H1 entry | Sniper entry zone within the discount/premium |
| **Displacement** | Impulse candle(s) with body ≥ k × ATR(N) that creates an FVG and breaks structure. | All | Confirms institutional participation |

## 2. Session microstructure (this is where ICT is most operational)

ICT thinks in NY-time windows. Times below are NY local; DST shifts the GMT mapping by an hour twice yearly.

| Window | NY time | Purpose | Read |
|---|---|---|---|
| **Asian Range** | 19:00 (prev) – 00:00 | Accumulation. Range is the H/L between these hours. | Mark Asian H and Asian L; both are liquidity. |
| **NY Midnight Open ("True Day Open")** | 00:00 NY | The reference price for the trading day. | Bias relative to this line. Above = bullish day premise, below = bearish. |
| **London Killzone** | 02:00–05:00 | London open expansion / Judas swing. | Look for sweep of Asian H or L, then CHOCH back the other way. |
| **Judas Swing window** | 00:00–05:00 | The fake move before the real move. | If price runs Asian high then reverses through midnight open with displacement → real move is short. Symmetric for longs. |
| **NY AM Killzone** | 07:00–10:00 (FX) / 08:30–11:00 (indices) | Primary distribution. | London-NY overlap = deepest liquidity. |
| **Silver Bullet** | 10:00–11:00 NY | A 1-hour FVG sniper window inside NY AM. | Trade only FVGs formed in this hour, in the direction of HTF bias. |
| **NY PM Killzone** | 13:30–16:00 | Secondary move; reversal-prone after lunch. | Lower conviction; useful for fade plays at PDH/PDL. |
| **ITH/ITL** | Pre 09:30 NY | "Intermediate-term" highs/lows on M15 formed before NY equity open. | Often the day's external liquidity targets. |

**JP holiday relevance.** USD/JPY, EUR/JPY, GBP/JPY, AUD/JPY are materially affected when Tokyo banks are closed (Golden Week, Obon, Year-end). Empirically: thin order book → moves on small flow are larger; intervention risk is non-trivial; the Tokyo "fixing" (09:55 JST) anomaly is documented in NBER work. Concrete trader rules: (a) no tight SL on JPY pairs during Golden Week (already a hard rule per memory `feedback_no_tight_sl_thin_market`), (b) Asian range is unreliable as a liquidity reference — London may not sweep it, (c) reduce size by 30–50% on JPY crosses, (d) flag any JN bank holiday from Forex Factory calendar.

## 3. Multi-timeframe top-down workflow

| TF | What you READ | What you DECIDE | Output |
|---|---|---|---|
| **W** | Macro structure: weekly trend, last weekly BOS/CHOCH, PWH/PWL. | Macro bias (long/short/neutral). | One-word bias. |
| **D1** | Daily dealing range, last D1 OB/FVG, PDH/PDL, where price sits in premium/discount. | Whether today's premise aligns with W. | "We are in D1 discount, looking for longs into D1 OB at 1.0850." |
| **H4** | H4 OBs and FVGs that have not been mitigated. H4 internal structure between the D1 swings. | Which H4 zone is the POI (point of interest) for the session. | A specific zone. |
| **H1** | Confirms H4 zone is being approached. Tracks H1 BOS/CHOCH. | Whether to arm an entry. | Arm/disarm. |
| **M15** | First TF where ICT entries actually live. Looks for liquidity sweep + CHOCH at the H4 POI. | Trigger setup detected: yes/no. | Setup defined. |
| **M5** | Refinement: the M5 FVG or OB inside the M15 setup. | Entry candle selection. | Entry zone. |
| **M1** | Execution only. Final displacement off the zone. | Pull the trigger. | Fill. |

**Narrowing rule**: bias only narrows downward; it never widens. If H1 contradicts D1, do not trade — wait for D1 to confirm or H1 to reset.

## 4. Confluence / "high-conviction" stacks

| Stack name | Recipe | Conviction |
|---|---|---|
| **OB-FVG sniper** | M15 BOS into H4 OB inside D1 discount + M5 FVG forms on the BOS leg + entry at OB-FVG overlap | S |
| **Judas reversal** | London hour sweeps Asian high, closes back below midnight open, M5 CHOCH down, FVG forms on displacement | S (intraday only) |
| **Silver Bullet** | HTF bias bullish, 10:00–11:00 NY, M5 FVG forms in the direction of bias, entry on first revisit | A |
| **Breaker + sweep** | Prior OB failed, price swept its low (or high), broke structure opposite, returns to the breaker | A |
| **EQH/EQL run** | M15 EQH detected, price sweeps it during NY AM with displacement, M1 CHOCH down inside H4 premium | A |
| **PDH/PDL fade** | Price tags PDH during NY PM with no HTF FVG above; M5 CHOCH down | B |

Anti-stacks (DO NOT trade): a single FVG with no structure context; an OB without displacement out; a CHOCH against an unbroken HTF trend; a sweep in a strongly trending HTF (sweeps in trends usually continue).

## 5. What's computable from OHLC vs what isn't

| Primitive | Computable from OHLC? | Notes |
|---|---|---|
| Swing pivots, BOS, CHOCH | Yes | Standard pivot detection. |
| OB / Breaker / Mitigation | Yes | Need swing detection + displacement filter (ATR-based). |
| FVG / iFVG | Yes | Pure 3-candle geometry. |
| EQH / EQL | Yes | Pivots within tolerance × ATR. |
| Liquidity sweeps | Yes | wick-beyond + close-back-inside. |
| Premium/discount + OTE | Yes | Dealing range + Fib. |
| Killzones, Asian range, NY midnight, Silver Bullet | Yes | Pure timestamp logic (with DST handling). |
| PDH/PDL/PWH/PWL | Yes | Daily/weekly OHLC. |
| Displacement | Yes | Body ≥ k × ATR + creates FVG. |
| **Order flow / DOM** | **No** | OANDA retail FX has no L2 book. Skip. |
| **Volume profile (true)** | **No** in spot FX (no consolidated tape). Tick-volume proxy only. | Use cautiously. |
| **CME COT / institutional positioning** | Indirect | Weekly CFTC report for JPY/EUR/GBP futures — useful as bias confirm. |
| **Tokyo fix flow** | **No** but the time-of-day effect is documented (NBER) and exploitable as a calendar feature. | Encode 09:55 JST as a feature window. |

## 6. Honest critique — what the quant evidence actually says

**What appears to have edge (defensible):**
- Liquidity sweeps + immediate reversal at session H/L, PDH/PDL — well-supported by stop-clustering literature and consistent with the documented Tokyo fix anomaly.
- Time-of-day effects: the London open expansion and NY 10am window have measurable autocorrelation/volatility signatures. ICT killzones essentially encode this.
- Mean reversion to FVG/imbalance regions. The 2026 Kondapally study (n=32,202) found roughly 55–70% of FVGs produce a tradeable reaction — modest but real edge if combined with HTF context.
- Premium/discount as a position-sizing filter (only fade in premium, only buy in discount within an HTF range) maps onto well-known mean-reversion-within-range literature.

**What is folklore / unsupported:**
- "Smart money is hunting your stops specifically." Market makers create liquidity; retail stops are a footprint, not a target. The pattern is real, the causal story is wrong.
- OBs have no published edge independent of structure context.
- The sub-decimal mysticism (0.705 OTE specifically vs 0.618 / 0.786) has no statistical support beyond the surrounding zone.
- Most public SMC "backtests" suffer from forward-looking bias.

**Practical posture for QuantRabbit:** treat ICT/SMC as a structured *vocabulary for narrative*, not a trading system. Use the OHLC-derivable primitives as features. Use the time windows as conditioning variables. Do not adopt the metaphysics.

## 7. Translation hooks for implementation

Suggested feature extractors, all OHLC-only:

1. `swings(df, n=3)` → swing high/low arrays.
2. `structure(df, swings)` → BOS/CHOCH events with timestamp, level, direction, displacement_atr.
3. `order_blocks(df, structure)` → list of (side, top, bottom, mitigated_at, has_fvg, has_sweep).
4. `fvgs(df)` → list of (side, top, bottom, created_at, filled_at).
5. `liquidity_pools(df)` → EQH/EQL clusters with tolerance = 0.1×ATR.
6. `sweeps(df, levels)` → for each prior swing/EQH/PDH, detect wick-beyond + close-back-inside events.
7. `dealing_range(df, tf='H4')` → most recent significant swing-H to swing-L; compute premium/discount/OTE.
8. `sessions(df)` → tag each bar with asian/london/ny_am/ny_pm/silver_bullet/judas_window in NY-anchored time, DST-aware.
9. `key_levels(df)` → PDH, PDL, PWH, PWL, NY_midnight_open, Asian_H, Asian_L per day.
10. `jp_holiday_flag(date)` → from Forex Factory or `jpholiday` Python package; gates JPY-pair sizing.

Reference Python implementation to study (not adopt as-is): [`joshyattridge/smart-money-concepts`](https://github.com/joshyattridge/smart-money-concepts).

For the `SKILL_trader.md` checklist, the minimum "what to read every cycle" list:

1. W/D bias word; D1 dealing range; D1 premium or discount.
2. PDH, PDL, PWH, PWL, NY midnight open. (5 numbers, always quoted.)
3. Asian H, Asian L, current session.
4. Last unmitigated H4 OB above and below price.
5. Open FVGs on H1.
6. Most recent BOS/CHOCH on H1 and M15 with timestamps.
7. Equal highs/lows within current dealing range.
8. JP holiday flag for next 3 days.
9. Stack name if a known confluence is present (e.g. "Judas reversal armed").

## Sources

- [GitHub: joshyattridge/smart-money-concepts](https://github.com/joshyattridge/smart-money-concepts)
- [LuxAlgo: ICT Concepts — Order Blocks](https://www.luxalgo.com/blog/ict-trader-concepts-order-blocks-unpacked/)
- [InnerCircleTrader.net tutorials (Killzones, OTE, IRL/ERL, Power of 3)](https://innercircletrader.net/)
- [TradingFinder ICT articles](https://tradingfinder.com/)
- [NBER w22820: Puzzles in the FOREX Tokyo Fixing](https://www.nber.org/system/files/working_papers/w22820/w22820.pdf)
- [Springer: A Deep Learning Approach to Identify FVGs in Forex (Kondapally 2026)](https://link.springer.com/chapter/10.1007/978-3-032-10670-4_36)
- [TradingRush: Debunking SMC](https://tradingrush.net/debunking-smart-money-concepts-trading-strategies/)
- [AlgoStorm: ICT & SMC Realistic Evidence-Based Review](https://algostorm.com/ict-smc-realistic-overview/)
- [ScienceDirect: Liquidity in the Global Currency Market](https://www.sciencedirect.com/science/article/pii/S0304405X22001891)
- [Forex Factory: JN Bank Holiday calendar](https://www.forexfactory.com/calendar/393-jn-bank-holiday)

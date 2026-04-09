# Risk Management

## Most Important Principle: Default is Take Profit. If you're holding, justify it.

**2026-03-27 lesson: GBP unrealized profit +3,000 yen → couldn't take profit due to HOLD bias → -4,796 yen.**
**The old rule "patience without cutting generates profit" was distorting judgment. Reversing it now.**

| Old (problematic) | New (default reversed) |
|---|---|
| Hold unless you can explain why to exit | **Exit unless you can explain why to hold** |
| Default = Hold | **Default = Take Profit** |
| Don't cut below ATR 50% | **Every session, every position: justify holding** |

**This applies at ALL profit levels, not just ATR×1.0.** ATR×1.0 triggers profit_check for data, but the principle is unconditional: if you have unrealized profit and can't write a specific reason to hold, close.

- **Every session, every position in profit**: Write the "Close or Hold" block (see format below). If "I'm not closing because" is blank or vague, close.
- **ATR×1.0 reached = profit_check trigger.** Run `python3 tools/profit_check.py` for additional 6-axis data
- **If profit_check outputs TAKE_PROFIT/HALF_TP, you have 30 seconds to justify holding.** If you can't, take profit
- **Avoid junk profits too**: Below ATR×0.5 and not worth the spread = too early. Exception: momentum reversal detected

## Directional Bias Check (lesson from 4/1 wipeout)

**All positions in the same direction = not diversification, it's a bet.**

2026-04-01: GBP_JPY/AUD_JPY/EUR_JPY all SHORT (concentrated JPY crosses) → all SL hit on bounce. With directional diversification, half would have been profitable.

**Checklist (every session):**
- All positions SHORT or all LONG → **explain why there's not a single position in the opposite direction**. Can't explain = bias
- More than 3 positions on the same pair → **sign of averaging-down hell. Go make money on other pairs**
- All positions are JPY crosses → **all-in on JPY. Full wipeout risk if JPY reverses**

**Countermeasures:**
- Main position in thesis direction + small position for bounce = normal
- Holding both LONG and SHORT is normal. Only one side is abnormal
- Don't bet all capital on the same theme (e.g., JPY strength)

## Thin Market / Holiday SL Rule (4/3 Good Friday lesson: -984 JPY)

**When liquidity is thin (holidays, Good Friday, year-end, early Asian session), DO NOT set tight SLs.**

| Condition | SL Rule |
|---|---|
| Spread > 2× normal | **No SL. Discretionary management only.** |
| Good Friday / bank holidays | **No SL or ATR×2.5+ minimum.** ATR×1.2 gets hunted. |
| Tokyo session (00:00-06:00Z) holding overnight | **No trailing stop. Fixed SL only if any.** |
| User says "SLいらない" | **Remove SL. Do not re-add. Do not close on your own judgment.** |
| **Daily rollover (5 PM ET)** | **Remove ALL SL/Trailing 20 min before. Restore after spread normalizes.** |

**What happened 4/3**: EUR_USD trail=11pip, GBP_USD trail=15pip, AUD_USD SL=0.69015 — ALL hunted on Good Friday thin liquidity. Total -984 JPY. Every thesis was correct. Every loss was from noise stops.

**When the user explicitly removes SL, that is a direct order.** The trader task must not re-add protection or close the position on its own judgment. The user is managing risk manually.

### Daily Rollover SL Guard (4/10 preventive measure)

**OANDA daily maintenance at 5 PM ET causes spread spikes every day.** Summer: 21:00 UTC (06:00 JST). Winter: 22:00 UTC (07:00 JST). Any SL/Trailing set at normal spread levels WILL get hunted during this window.

**`protection_check.py` detects rollover approach automatically.** When it outputs `ROLLOVER WINDOW`:
1. Run `python3 tools/rollover_guard.py remove` — removes all SL/Trailing, saves state
2. Wait for rollover to pass (~15 min)
3. Run `python3 tools/rollover_guard.py restore` — re-applies saved SL/Trailing

**The trader task runs protection_check at every session start.** If rollover is approaching, the trader MUST run rollover_guard.py remove before doing anything else. If rollover has passed and saved state exists, run restore first.

## Conviction-Based Sizing — Determined by Depth of Analysis, Not Indicator Count

**High conviction → size up. Low conviction → size down.**

**Size = margin allocation per entry, as % of NAV. Check `NAV`, `marginUsed`, `marginAvailable` from session_data.py before every entry.**

### Conviction is NOT "how many indicators agree." It's "how deeply have you looked, and does the whole picture cohere?"

**Past data (3/20-4/3): 7 conviction-S opportunities found, 5 were undersized by 70% avg. 6,740-13,140 JPY thrown away. Root cause: trader checked 2-3 familiar indicators, rated B, and stopped looking. Deeper analysis would have revealed S.**

### 6 Indicator Categories (reference — what to check)

| # | Category | Indicators | What it tells you |
|---|----------|-----------|-------------------|
| ① | **Direction** | ADX/DI, EMA slope, MACD direction | Where is the market going? |
| ② | **Timing** | StochRSI, RSI extreme, CCI, BB position | Is now the right moment? |
| ③ | **Momentum** | MACD hist change, ROC, EMA cross | Is the move accelerating or dying? |
| ④ | **Structure** | Fib, cluster, swing distance, Ichimoku | Where are the walls? |
| ⑤ | **Cross-pair** | Correlated pairs, currency strength | Is this currency-wide or pair-specific? |
| ⑥ | **Macro** | News, events, flow | What's the bigger story? |

### Before every entry — write this block (required)

```
Thesis: [1 sentence — what trade and why NOW]
Type: [Scalp / Momentum / Swing]
FOR:  ___ (category) + ___ (category) + ___ (category)
Different lens: [check 1+ indicator from a category NOT in FOR] → supports / contradicts / neutral
AGAINST: ___ [specific. "nothing" only if you actually checked]
If I'm wrong: ___ [the scenario where this trade loses, and at what price]
→ Conviction: [S/A/B/C] | Size: ___u (___% NAV)
```

**"Different lens" is the key line.** It forces you to look at the trade from an angle you wouldn't normally check. It moves conviction in BOTH directions:

| Different lens result | Effect |
|----------------------|--------|
| Supports FOR | Conviction UP — B can become A or S |
| Neutral | No change — but you know you checked |
| Contradicts FOR | Conviction DOWN — move to AGAINST, adjust size |
| Overturns thesis | Abort entry or reverse direction |

**Example — conviction upgrade (B → S):**
```
First impression: M5 StochRSI=0.0, H1 unclear → "probably B"
Different lens: Fib → price at H1 38.2% pullback. Ichimoku → above cloud. Cluster → 5× tested support
→ Actually S-Scalp. Everything aligns. Size up to 10,000u
```

**Example — conviction downgrade (S → C):**
```
First impression: ADX=50 BEAR, M5 falling, macro JPY strong → "S-Swing SHORT"
Different lens: CCI → -274 (extreme). Fib → 78.6% reached (exhaustion zone)
→ Actually C. Next move is a bounce. Abort SHORT
```

### Conviction = story coherence, not category count

| Conviction | What it means | FOR | Different lens | AGAINST | If I'm wrong |
|------------|--------------|-----|----------------|---------|-------------|
| **S** | "Everything points the same way" | 3+ categories, strong | Supports or confirms | Checked, nothing credible | Specific single scenario, not happening now |
| **A** | "Good setup, one manageable risk" | 2-3 categories | Mostly supports | Specific concern, but has mitigation | Specific, plausible but has a plan |
| **B** | "See something, but picture incomplete" | 1-2 categories | Mixed or unchecked | Real contradictions or mostly unchecked | Multiple scenarios or vague |
| **C** | "Don't fully understand this trade" | Thin | Contradicts or unchecked | Strong contradictions | Can't articulate clearly |

### S-Type determines hold time and TP (after conviction is determined)

| S Type | MTF | Hold Time | Target |
|--------|-----|-----------|--------|
| **Scalp** | M1→M5→H1 | 5-30 min | ATR×0.5-1.0 |
| **Momentum** | M5→M15→H1 | 30min-2h | ATR×1.0-2.0 |
| **Swing** | H1→H4→macro | 2h-1day | ATR×2.0+ |

### Sizing table

| Conviction | Margin for this entry | At NAV 200k, USD_JPY @150 |
|------------|----------------------|---------------------------|
| **S (any type)** | **~30% of NAV** | margin 60k = **10,000u** |
| **A** | **~15% of NAV** | margin 30k = **5,000u** |
| **B** | **~5% of NAV** | margin 10k = **1,667u** |
| **C** | **~2% of NAV** | margin 4k = **667u** |

Units = margin_budget / (price / 25). AUD_JPY @97 → S = 60k/(97/25) = **15,500u**.

**Before every entry: marginUsed + new margin must stay below NAV × 0.90.** `marginAvailable` from OANDA tells you directly.

**One conviction-S trade beats ten conviction-B trades. Don't grind for volume.**
**If Different lens reveals S conditions and you still enter at B-size, you are throwing away money.**

## add-on — Be flexible. But one absolute rule.

**Pyramiding and averaging down are both fine. Depends on the situation.**

However: **"Can you give a different reason than last time?"** If not, don't add.
- Same StochRSI=1.0 for the third time → not new information. That's "strong trend," not "ceiling"
- New signal on a different TF, news, support touch, etc. → new basis. Adding OK

## Take Profit — 2-stage structure: ATR triggers, multi-scan decides

### STEP 1: Triggers (when to evaluate)

| Trigger | Condition | Action |
|----------|------|-----------|
| **ATR×1.0 reached** | Unrealized profit reached ATR or more | Run `profit_check.py` → make TP decision |
| **Session start** | Beginning of every session | `profit_check.py --all` to check all positions |
| **Sudden move on another position** | Stop hit or spike on another pair | Immediately check all profitable positions |
| **M5 momentum reversal** | MACD hist reversal + StochRSI overheated | Consider taking profit on that position |

### STEP 2: Decision (how to decide) — profit_check.py 6 axes

profit_check.py evaluates the following simultaneously and outputs a recommendation:

1. **ATR ratio** — unrealized profit / ATR. 1.0x = trigger, above 1.5x = sufficient
2. **M5 momentum** — MACD hist direction, StochRSI position, EMA slope
3. **H1 structure** — ADX, DI direction, alignment with thesis, divergence
4. **7-pair correlation** — directional agreement/divergence across correlated pairs
5. **S/R distance** — distance to next resistance/support
6. **Peak comparison** — drawdown from peak recorded in state.md

### STEP 3: Default is Take Profit

- **TAKE_PROFIT recommended** → justify holding in 30 seconds. If you can't, full exit
- **HALF_TP recommended** → half profit-take is the default. Need a strong reason to hold full size
- **HOLD recommended** → OK. But add the justification to state.md (include peak record)
- **Example of valid basis**: "H1 ADX=30 DI+ accelerating, 15 pip left to TP target, correlated pairs also in same direction"
- **Invalid basis**: "Still looks like it'll move" / "Thesis is alive (no specifics)"

### Preventing Junk Profits

Below ATR×0.5 and not worth the spread = too early. **However, if momentum reversal is detected, taking profit before ATR target is OK.**

### BE SL is NOT profit-taking (4/8 AUD_JPY lesson: +1,200 JPY → +40 JPY)

**2026-04-08 lesson: AUD_JPY peaked at +1,200 JPY. Trader moved SL to breakeven (entry+1pip) instead of taking profit. Price reversed. BE SL hit. Result: +40 JPY. Lost 1,160 JPY of unrealized profit.**

**BE SL is the 3/27 Default HOLD trap wearing a different mask.** It lets the trader say "I'm managing risk" while actually holding and hoping. "Zero capital risk" sounds safe but means "I chose +0 over +1,200."

**When ATR×1.0 is reached, these are the ONLY valid actions:**

| Action | What it does | When to use |
|--------|-------------|-------------|
| **A. HALF TP** | Close half at market + trailing stop on remainder | Default. Locks in profit while keeping upside |
| **B. FULL TP** | Close all at market | M5 momentum reversal + H1 weakening |
| **C. HOLD + trailing** | Keep full size but trailing stop at ≥50% of unrealized profit | H1 ADX>30 strong trend + M5 still making new highs |

**BE SL (fixed SL at entry price) is NOT on this list.** It gives back 100% of unrealized profit if triggered. That's not risk management — it's a lottery ticket.

**If you write "SL moved to BE" in the log, you must also write:**
```
BE SL chosen. Current UPL: +___ JPY (+___pip).
If SL hit: I keep +___ JPY. I am giving back ___ JPY.
I prefer this over HALF TP at market because: ___
```

**TP spread buffer**: When setting a fixed TP near a structural level, subtract the current spread from the target price. A TP that misses by 0.4 pip because of spread is the same as no TP. `TP = structural_level - spread` for LONGs, `+ spread` for SHORTs.

## Stop Loss — Structural placement. No ATR-only. No monetary triggers.

**There are zero monetary-based stop loss rules. Whether it's -500 yen or -1000 yen, do not use the amount as a decision factor.**

### SL placement must be structural (4/3 lesson: -984 JPY from ATR×0.6 mechanical SLs)

Every SL must answer: **"What market structure is at this price?"**

```
❌ SL at ATR×1.2 → "because that's the formula" → bot. Gets hunted by noise
✅ SL at 110.060 → "NFP spike low 110.082 minus spread buffer" → structural
✅ SL at 1.3180 → "below Fib 78.6% + Ichimoku cloud base convergence" → structural
✅ No SL → "Good Friday thin market, discretionary management" → pro judgment
```

ATR is for **sizing** (how much risk am I taking), not **placement** (where should SL go). Use swing lows, Fib levels, DI reversal points, cluster support, Ichimoku cloud — levels where the market actually reacted.

### Position management — always 3 options, never 2

When conditions change after entry (event approaching, thin market, timeframe shift), evaluate all 3:

| Option | When to choose |
|--------|---------------|
| **A. Hold + adjust** | Timeframe changed → widen SL to structural level, extend TP |
| **B. Cut and re-enter** | Better setup available at a different level/time. In profit + event risk = strong candidate |
| **C. Hold as-is** | Current protection matches current conditions |

**4/3 example**: EUR_USD/GBP_USD in profit, Good Friday, NFP 10h away. Only option considered was "trail at ATR×0.6." Option B (take profit now, re-enter post-NFP with confirmed direction) was the correct play but was never evaluated.

### Loss management decision flow

Decision flow when you're concerned about an unrealized loss (this is all. Don't look at the amount):

1. **Has the H1 structure changed?**
   - Did DI+/DI- flip? Did ADX change direction?
   - → NO → **Don't cut. The thesis is alive**
   - → YES → Go to step 2

2. **Has the basis for the thesis disappeared?**
   - Is the entry basis (Div, macro, flow, etc.) still valid?
   - → Basis still there → **Cut size in half and hold** (don't fully exit)
   - → Basis also gone → **Full exit OK**

3. **Is a signal appearing in the opposite direction?**
   - Clear reversal Div on H1 + momentum reversal confirmed on M5
   - → Both YES → Consider flipping (reversing the position)
   - → Only one → Half profit-take and watch

**Never do these:**
- Cut based solely on the loss amount (-500 yen because, -1000 yen because)
- Panic market stop on a spike → cutting at the top/bottom is the worst outcome. Wait for the retracement
- Set your own monetary thresholds (e.g., "pain rule", "stop line") and follow them mechanically

**You are a professional trader. Don't fear the amount. Read the market.**

## Margin Management (capital efficiency is everything)

**Make money with money. Don't leave margin sitting idle. But calculate BEFORE you spend.**

### Pre-entry margin check (mandatory — before conviction block)

**Every entry must include this calculation in the conviction block:**
```
Current margin: ___% | + this entry: ___JPY | + pending LIMITs: ___JPY | → worst case: ___%
```

| After-entry margin (incl. pending LIMITs) | Rule |
|------------------------------------------|------|
| **Below 85%** | OK |
| **85-90%** | S-conviction only, no pending LIMITs that could push over 90% |
| **Above 90%** | **BLOCKED. Free margin first.** |

**4/8 lesson: 3 positions stacked without margin calculation → 97% → forced EUR_JPY close at -319 JPY. Pending AUD_USD LIMIT (31k) was ignored in calculation. Always include pending LIMIT margin in worst-case projection.**

### Margin utilization targets

- **Below 60% = ask yourself if you're missing good setups**: But margin itself is not a reason to enter. Enter as a result of reading the market
- **Below 80% = still have room**: Actively look for additional entries
- **Above 90% = no new positions**: Hedges only (margin = 0)
- **Above 95% = forced half profit-take**: Immediately half-close the position with the largest unrealized loss
- **Wait 30 minutes after closeout**: No re-entry on the same thesis immediately

## Failure Patterns (don't repeat)

| Pattern | Countermeasure |
|----------|------|
| Profits too small (+40 yen × 13 trades) | Don't exit below ATR 50%. Ask if it's worth the spread + effort |
| Delayed profit-take (+2,833 → closeout) | Lock in when unrealized profit is meaningful. Both too early and too late are wrong |
| Panic stop → price reverses (GBP -237, EUR -246) | Check H1 structure + thesis basis. If both alive, don't cut |
| add-on × 7 times on same basis | Don't add without a new reason |
| Chasing (jumping in after 20 consecutive bullish candles) | Overheating = counter-trend opportunity |
| Panic stop on spike (-3,832 yen) | Wait for the retracement. Don't market-order at top/bottom |
| H1 paralysis (+353 → +86) | Detect momentum change on MTF and take profit → rotate |
| **Panic-dumped all positions → price came back** | If H1 structure hasn't changed, don't cut. Keep at least half at worst |
| **Consecutive losses on one pair** | Re-analyze that pair's H1 structure. If you can't read it, step away (judge by "can you read it," not the amount) |
| **Junk profit spree** | Don't exit below ATR 50%. Ask if it's worth the spread + effort |
| **Bad R/R ratio** | Letting winners run comes first. Don't tighten stops — delay taking profit |
| **Default HOLD trap (GBP 3/27)** | +3,000 yen → -4,796 yen. Held too long on "thesis is alive." **Default is now Take Profit** |
| **Missed profit-take while distracted** | GBP unrealized profit gone while handling AUD. On sudden moves, immediately check all positions' unrealized profit |
| **All positions same direction → wipeout (4/1)** | GBP_JPY/AUD_JPY/EUR_JPY all SHORT → all SL hit on bounce. **Diversify direction. One-way concentration is gambling** |
| **Transcribing indicators = mistaking it for analysis (4/1)** | "ADX=50 MONSTER BEAR" repeated 30 sessions, same conclusion every time. **Indicators are the past. Look at the shape of the chart** |
| **Left unrealized profit to die (4/1)** | EUR_USD +536 yen, GBP_JPY +60 yen → HOLD → SL hit. **Take what the market gives you** |
| **Adding in the same direction after the move is exhausted (4/1)** | New SHORT at H4 CCI=-274 RSI=29 = selling after a 200-pip drop. **Next move is a bounce** |
| **Tight SL on thin market = free money for market makers (4/3)** | Good Friday: trail 11-15pip + SL 10pip → ALL hunted. -984 JPY total. Thesis was right on every trade. **Holiday/thin market = NO SL or ATR×2.5+** |
| **Closing after user said "hold, no SL" (4/3)** | User removed SL at 13:04. Claude closed AUD_JPY at 13:44 anyway. -338 JPY. User had to re-enter. **User instruction to remove SL = hold. Don't override.** |
| **Panic close → panic re-entry = double loss (4/3)** | Closed AUD_JPY @110.077 (-338円), re-entered 7min later @110.118 (Sp1.8pip, pretrade=C(1)). Held=損失ゼロ。**再エントリー前に「クローズ時より良い価格か？新しい根拠はあるか？」両方Noならスプレッド払って同じ物を買い戻してるだけ** |
| **ATR×N mechanical SL = bot (4/3)** | EUR_USD trail=ATR×0.6(11pip), GBP_USD trail=ATR×0.7(15pip) on Good Friday → all hunted. -984 JPY. **SL must be at structural levels (swing low, Fib, cluster), not ATR multipliers. If you can't name the structure, don't set the SL** |
| **Binary position management (4/3)** | Only considered "trail or hold." Never evaluated "take profit now and re-enter post-NFP." In profit + event risk + thin market = cut and re-enter was the best play. **Always evaluate 3 options: adjust / cut-and-re-enter / hold-as-is** |
| **BE SL as stealth HOLD (4/8)** | AUD_JPY +1,200 JPY peak → SL moved to BE (entry+1pip) → price reversed → +40 JPY. Skipped profit_check at ATR×1.0. "Zero capital risk" framing hid the real cost: giving back 1,160 JPY. **BE SL is not profit-taking. At ATR×1.0: HALF TP / FULL TP / trailing at ≥50% profit. No BE SL.** |
| **TP missed by spread (4/8)** | AUD_JPY TP=111.100, bid peaked 111.096 (0.4pip short). Spread=2.4pip ate the fill. **Subtract spread from structural TP level. TP = level - spread for LONGs.** |
| **Margin overflow → forced close = self-inflicted loss (4/8)** | EUR_JPY+EUR_USD+GBP_JPY stacked without margin calculation → 97% → EUR_JPY forced close at -319 JPY. Thesis was alive. **Calculate margin BEFORE entering. Include pending LIMITs in worst-case. Above 85% = stop adding.** |
| **Market order in thin liquidity = giving away pips (4/8)** | Easter Monday: EUR_USD 4000u + GBP_JPY 3900u entered via market order in thin liquidity. Spread wider, slippage risk. **Thin market (holiday/early Asian) = LIMIT at M5 support, even for S-conviction.** |

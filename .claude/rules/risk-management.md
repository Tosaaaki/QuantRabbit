# Risk Management

## Most Important Principle: Default is Take Profit. If you're holding, justify it.

**2026-03-27 lesson: GBP unrealized profit +3,000 yen → couldn't take profit due to HOLD bias → -4,796 yen.**
**The old rule "patience without cutting generates profit" was distorting judgment. Reversing it now.**

| Old (problematic) | New (default reversed) |
|---|---|
| Hold unless you can explain why to exit | **Exit unless you can explain why to hold** |
| Default = Hold | **Default = Take Profit** |
| Don't cut below ATR 50% | **ATR×1.0 reached = trigger profit evaluation** |

- **ATR×1.0 reached = profit_check trigger.** Run `python3 tools/profit_check.py` for 6-axis evaluation
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

## Conviction-Based Sizing

**High conviction → size up. Low conviction → size down.**

**Size = % of current NAV. Check session_data.py output for live NAV before sizing.**

| Conviction | pretrade score | Conditions | Size (% of NAV) | Example |
|------|------|------|--------|-----|
| **S (ironclad)** | 8+ | H1+H4+macro all aligned, Div confirmed | **~5% NAV** | H4 ADX>30 + H1 same direction + M5 pullback |
| **A (high)** | 6-7 | H1 direction aligned + M5 timing confirmed | **~3% NAV** | H1 bullish + M5 StochRSI=0.0 |
| **B (normal)** | 4-5 | Signal from 1 TF only | **~1% NAV** | M5 Div only, H1 unclear |
| **C (probe)** | 0-3 | Thin basis | **~0.5% NAV** | Counter-trend within range |

NAV 200k example: S≈10000u, A≈6000u, B≈2000u, C≈1000u. If NAV drops to 150k: S≈7500u, A≈4500u, B≈1500u, C≈750u.

**One conviction-S trade beats ten conviction-B trades. Don't grind for volume.**

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

## Stop Loss — No monetary triggers. Judge by market conditions.

**There are zero monetary-based stop loss rules. Whether it's -500 yen or -1000 yen, do not use the amount as a decision factor.**

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

**Make money with money. Don't leave margin sitting idle.**

- **Below 60% = ask yourself if you're missing good setups**: If you're below 60% while being able to scan 7 pairs × all TFs simultaneously, look again more carefully. **But margin itself is not a reason to enter.** Enter as a result of reading the market
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

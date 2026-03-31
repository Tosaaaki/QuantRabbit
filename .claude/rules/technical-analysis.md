# Technical Analysis Rules

## MTF Hierarchy (Most Important)
- **H1 = Big picture (direction), M15 = Momentum shift detection, M5 = Entry confirmation, M1 = Timing**
- Don't freeze up just because H1 is bearish and hold short. Read momentum changes across MTF and rotate.

## Every Cycle: 7-Pair Cross Scan

| What to look for | Meaning |
|----------|------|
| StochRSI extreme (0.0 or 1.0) | Reversal opportunity |
| Divergence (score>0) | Reversal signal |
| CCI ±200+ | Extreme overheating, mean-reversion trade |
| BB squeeze (bands close together) | Breakout pending |

## Inter-Pair Relationships (Your Strongest Weapon)
- USD_JPY M1 top (StochRSI=1.0) → Reversal opportunity in USD pairs (EUR/GBP/AUD)
- EUR_USD and GBP_USD both M5 oversold simultaneously → USD-wide strength reversal signal
- AUD_USD and AUD_JPY moving in same direction → AUD currency itself is moving. High conviction.

## How to Use Indicators — Choose by Situation, Not Fixed Set

### Base (Always Check)
ADX/DI, EMA12/20, RSI — these are mandatory every time. Use these to judge "what state is the market in right now."

### Situational Add-ons (Pick from the 84-weapon arsenal as needed)

| Situation | Add these indicators | Why they work |
|------|-------------|-----------|
| Strong trend (ADX>30) | EMA slope (5/10/20), MACD hist, ROC5/10 | Momentum changes = TP timing |
| Suspected trend exhaustion | Divergence (RSI+MACD), upper/lower wick avg (wick_avg_pips) | Wick expansion + Div = reversal near |
| Range / stagnation | BB width (BBW), CCI, VWAP deviation, Keltner width (kc_width) | BB vs KC comparison improves squeeze accuracy |
| Breakout pending | Donchian width, BB squeeze, Chaikin Vol | Depth of vol compression = explosive breakout power |
| Support/resistance detection | Cluster (cluster_high/low_gap), swing distance, Ichimoku cloud | Where are the price walls |
| Reversal entry | StochRSI extreme + CCI±200, wick pattern (wick_avg), hit count (high/low_hits) | How many times tested = wall strength |
| Overheating check | Combined RSI + CCI + VWAP deviation + BB position | Single indicator gives false overheating reads. Use composite for accuracy |
| Volatility spike | ATR, vol_5m, Chaikin Vol, BB width change rate | Vol change = basis for SL width and size adjustment |
| Post-TP / rotation plan | Fib retrace/ext (fib_wave.py), N-wave structure | Quantify re-entry zones and TP targets |
| Pullback / retracement judgment | Fib 38.2-61.8% + BB mid + EMA20 + cluster | Multiple level confluence increases reliability |

### Important: Test Combinations and Record Results

- **Always record results when using a new combination**: add to the "Indicator Combination Learnings" section in strategy_memory.md
- **Combinations that worked**: record situation, pair, TF, combination, and result as a set
- **Combinations that didn't work**: also record why it failed (don't repeat the same mistakes)
- **All 84 are weapons**: an indicator you haven't used isn't "useless" — it's "untested." Try them aggressively.

## Proven Patterns (How to Win)
- **Immediate re-entry after TP**: If the thesis is still alive, get back in immediately
- **Base confirmation → full add-on**: Confirm with 3 consecutive M1 bullish candles → add
- **Flip**: The moment you notice flow reversing, switch in 1 second
- **Hidden Div → immediate entry**: Don't ignore it when a Div score appears
- **Pair diversification**: Same thesis across different pairs (e.g., EUR+GBP for USD weakness theme)

## N-Wave Recognition and Fibonacci — Map of "Where Are You in the Wave"

### Why You Need It
Oscillators (RSI/StochRSI/CCI) are the "current temperature." N-wave and Fib are the map of "where are you now and where are you going."
A thermometer alone can't build a rotation plan. Carry the map.

### N-Wave Structure
```
Bullish:  A(low) → B(high) → C(pullback) → D(new high)
Bearish:  A(high) → B(low) → C(pullback) → D(new low)
```
- AB leg = first impulse. Defines direction and size
- BC leg = pullback. Fib 38.2-61.8% is healthy. Above 78.6% = wave breakdown
- CD leg = second impulse. CD ≥ AB×0.6 is healthy
- quality = min(AB,CD) / BC. Ideal is 1.0 or above

### Fibonacci Usage

| Scenario | Level | Action |
|------|--------|-----------|
| Waiting for pullback (re-entry) | 38.2%, 50%, 61.8% retrace | Pullback/retracement zone |
| TP target | 127.2%, 161.8% extension | Project AB from C |
| Thesis invalidation | 78.6%, 100% retrace | Wave breakdown judgment |
| Confluence | Fib + BB mid + Ichimoku cloud + cluster | Multiple overlaps = strong S/R |

### Execution
```bash
python3 tools/fib_wave.py {PAIR} {TF} {BARS}   # single pair
python3 tools/fib_wave.py --all                  # all pairs
```

### How to Use Every Cycle
1. Run `fib_wave.py --all` at session start to grasp wave structure across all pairs
2. Reference Fib levels for TP/entry decisions
3. Record current wave position, next entry price, and TP price in the "Wave Plan" section of state.md
4. After TP → plan re-entry at Fib pullback levels (rotation)

## Exhaust the 84-Weapon Arsenal

adaptive_technicals.py selects indicators by situation for you, but relying on it alone is not enough.
Read the situation yourself and choose the indicators you need.

### Underused Weapons (Try Them Aggressively)
- **Keltner Channel width (kc_width)**: Compare with BB. BB > KC = breakout confirmation
- **Chaikin Volatility**: Vol change rate. Rapid expansion = breakout imminent
- **Donchian width**: Range breadth. Narrowing = breakout signal
- **VWAP deviation**: Distance from fair value. Extreme deviation = mean-reversion trade
- **Wick average (wick_avg)**: Length of upper/lower wicks. Expansion = reversal pressure
- **High/low touch count (hits)**: How many times tested. More = easier to break
- **Ichimoku cloud position**: Above/below cloud = strength/weakness. Cloud thickness = support strength
- **Price cluster (cluster_gap)**: Price concentration level = strong S/R

### Combination Principles
- Don't stack indicators from the same category (RSI + StochRSI + CCI = all oscillators)
- Combine different categories: Trend (ADX) + Momentum (MACD) + Structure (Fib) + Volatility (ATR)
- Record results in strategy_memory.md whenever you try a new combination

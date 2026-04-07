# Trader Lessons — Historical Failure Patterns & Learnings

**This file is referenced by daily-review for distillation into strategy_memory.md.**
**The trader task does NOT read this file every session — lessons are distilled into strategy_memory.md.**

---

## Sizing & Conviction

- **[3/20-4/3] Conviction-S undersizing = biggest silent profit killer**: 7 trades met conviction-S conditions. 5 of 7 were entered at B-size (1,000-2,000u) instead of S-size (8,000-10,000u). Actual P&L: +2,360 JPY. At S-sizing: +9,100-15,500 JPY. **6,740-13,140 JPY thrown away.** Root cause: trader checked 2-3 familiar indicators, rated B, and stopped looking.
- **[4/1] Backward sizing**: Winning trades at 2000u for +300 JPY, losing trades at 10500u for -2,253 JPY. Go big when winning, go small when losing — not the reverse.

## Stop Loss & Protection

- **[4/3] Good Friday tight SL wipeout (-984 JPY)**: EUR_USD trail=11pip (ATR×0.69), GBP_USD trail=15pip (ATR×0.7), AUD_USD SL=10pip — ALL hunted on Good Friday thin liquidity. Total -984 JPY. Every thesis was correct. Every loss was from noise stops.
- **[4/3] Tokyo session trailing stop trap**: EUR_USD trail=ATR×0.7, GBP_USD trail=ATR×0.7 → both clipped at 02:22Z by Tokyo noise. Trail was set as "NFP protection" the night before but ATR×0.7 is too tight for 00:00-06:00Z.
- **[4/3] Pre-NFP trail = double trap**: Trail added at 00:37Z. Tokyo noise clipped BOTH at 02:22Z — 10 hours BEFORE NFP fired. -316 JPY from trails alone.
- **[4/3] ATR×N mechanical SL = bot**: SL at ATR×0.6 = noise width on thin market. SL at swing low / Fib 78.6% / DI reversal = meaningful. If you can't name the market structure at the SL price, the SL is bot-like.

## Position Management

- **[4/3] Binary position management = missed best option (-984 JPY)**: Opus only considered "trail or hold" — never "take profit now and re-enter post-NFP." On Good Friday thin market with NFP 10h away, cutting in profit + waiting for direction was the correct play.
- **[3/27] Default HOLD trap (-4,796 JPY)**: GBP unrealized profit +3,000 JPY → couldn't take profit due to HOLD bias → -4,796 JPY. Option B (take profit at +3,000) = 7,796 JPY thrown away.
- **[4/1] All-SHORT wipeout (-4,438 JPY)**: GBP_JPY/AUD_JPY/EUR_JPY all SHORT + all JPY crosses. A bounce came and wiped everything. With directional diversification, half would have been profitable.
- **[4/1] Left unrealized profit to die**: EUR_USD +536 JPY, GBP_JPY +60 JPY → HOLD → SL hit. Take what the market gives you.
- **[4/1] Adding in same direction after move exhausted**: New SHORT at H4 CCI=-274 RSI=29 = selling after a 200-pip drop. Next move is a bounce.

## Entry Timing & Analysis

- **[3/31-4/1] Random timing on correct thesis (WR 43%)**: EUR_USD LONG entered 8× on "USD weak + H1 bull." Winners had specific timing (M5 StRSI=0.0 at Fib 38.2%). Losers were "it dipped, so I bought." Thesis must say what happened NOW, not what's been true for days.
- **[4/1] Indicator transcription ≠ analysis**: "H1 ADX=50 DI-=31 MONSTER BEAR → SHORT" repeated 30 sessions with same conclusion. Market was bouncing but trusted indicators over chart. Indicators describe the past, charts describe the present.
- **[2026-03-30] USD_JPY momentum death undetected**: Reached +20pip → failed to update the high, started making lower lows gradually → cut at -9pip 4 hours later. Trusted StRSI=0.0 "bounce incoming." The chart was saying "momentum is gone."
- **[4/3] Scan had "Skip pre-NFP" × 5 pairs for 4+ hours**: That's not analysis. That's copy-paste. A pro trader scanning 7 pairs always finds something interesting.

## User Instruction Handling

- **[4/3] Closed after user said "hold, no SL" (-338 JPY)**: User removed SL at 13:04. Claude closed AUD_JPY at 13:44 anyway. -338 JPY. User had to re-enter.
- **[4/3] Panic close → panic re-entry = double loss**: Closed AUD_JPY @110.077 (-338 JPY), re-entered 7min later @110.118 (Sp 1.8pip, pretrade=C(1)). If held = loss zero. Before re-entering: "Is the price better than where I closed? Is there a new reason?" If both NO, you're paying spread to buy back what you threw away.

## Directional Bias

- **[4/1] All positions same direction = not diversification**: GBP_JPY/AUD_JPY/EUR_JPY all SHORT (concentrated JPY crosses) → all SL hit on bounce. "MONSTER BEAR" bot-thinking repeated endlessly. Chart was showing a bounce but trusted indicators.
- **[4/7] Counter-trade column discovery**: All 7 pairs had LONG-only plans while AUD_JPY H4 StRSI=1.0 with bearish MACD div, GBP_JPY H4 MACD div=-1.0. Short-term SHORTs were available but invisible because macro was pointing LONG.

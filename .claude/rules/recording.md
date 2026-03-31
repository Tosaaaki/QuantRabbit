# Recording Rules — Check → Order → Record is a Single Action

**Entry flow: pretrade_check → order → 4-point record. These 5 steps are indivisible.**
**Exit flow: preclose_check → close → 4-point record. This flow is also indivisible.**

## STEP 0a: pretrade_check (run before every entry)

```bash
cd /Users/tossaki/App/QuantRabbit/collab_trade/memory && python3 pretrade_check.py {PAIR} {LONG|SHORT}
```

- Output is data and past lessons. **You make the call.**
- HIGH/MEDIUM/LOW are summaries of data, not instructions. Adjusting your judgment to the situation is what professionals do.
- Include the result in the entry record in trades.md (e.g. `pretrade: LOW`)

## STEP 0b-2: profit_check (run at the start of every session)

**A tool to ask the data "should I take profit now?". Auto-runs against all positions at session open.**

```bash
cd /Users/tossaki/App/QuantRabbit && python3 tools/profit_check.py --all && python3 tools/protection_check.py
```

- **profit_check**: Evaluates ATR ratio, M5 momentum, H1 structure, 7-pair correlation, S/R distance, and peak comparison all at once
- **protection_check**: Evaluates TP/SL/Trailing status of all positions on an ATR basis. NO PROTECTION requires immediate action.
- **Default is take profit.** If TAKE_PROFIT/HALF_TP is recommended, articulate why you're holding. If you can't articulate it → take profit. If you can → HOLD (add the rationale to state.md)
- **Log the peak in state.md**: `Peak: +3,200 JPY @1.33620 (03:20Z)`

## STEP 0b: preclose_check (run before every close)

```bash
cd /Users/tossaki/App/QuantRabbit && python3 tools/preclose_check.py {PAIR} {SIDE} {UNITS} {unrealized_pnl_jpy}
```

- Output re-confirms the thesis and presents facts. It does not take away your judgment.
- Closing after answering it is OK. Closing reflexively without answering is not.
- **Always note the close reason in live_trade_log** (e.g. `reason=H1_DI+reversal`)
- **Closing without a reason = rule violation**

## STEP 1-4: Order → 4-point record (no deferring)

| File | What to write |
|----------|---------|
| `collab_trade/daily/YYYY-MM-DD/trades.md` | Detailed entry/exit table |
| `collab_trade/state.md` | Current positions, thesis, realized P&L (external memory) |
| `logs/live_trade_log.txt` | Trade execution log (chronological) |
| `#qr-trades` (Slack notification) | Post entries/modifications/closes to Slack |

## Slack Notification (4th point)

Post to `#qr-trades` via `slack_trade_notify.py` at the same time as order execution.

```bash
# Entry
python3 tools/slack_trade_notify.py entry --pair {PAIR} --side {LONG|SHORT} --units {UNITS} --price {PRICE} [--sl {SL}] [--thesis "thesis"]

# Modification (partial TP, SL move, add-on, etc.)
python3 tools/slack_trade_notify.py modify --pair {PAIR} --action "half TP" --units {UNITS} --price {PRICE} --pl "{PL}" [--note "remaining units, BE move, etc."]

# Full close
python3 tools/slack_trade_notify.py close --pair {PAIR} --side {LONG|SHORT} --units {UNITS} --price {PRICE} --pl "{PL}" [--total_pl "total realized P&L"]
```

## state.md is not a snapshot. It's a story.

**Bad example:** `USD_JPY: H1 uptrend (ADX32). Looking for dip buys`
**Good example:**
```
## USD_JPY LONG Thesis
- Read: JPY weakness direction. Targeting 158.50 → 159.00
- Basis: Fed hawkish hold + Iran risk-off → USD bid
- Invalidation: DXY breaks 98.5, US yields drop sharply, or 158.30 cleanly broken
- Progress: 158.38 → add-on 158.37 → half TP 158.41
```

## Recording User Remarks

If the user says anything, immediately record it in `daily/YYYY-MM-DD/notes.md` **with chart context**.
`User: "Looks like it'll go up" — 3 consecutive M5 bearish candles followed by long lower wick, BB lower band touch, H1 uptrend (ADX=32), RSI=35`

**If you're asked "are you recording properly?", you've already lost.**

---
name: trend-bot
description: Heartbeat-safe trend continuation bot — capture band-walk / follow-through moves every minute with MARKET only [Mon-Fri, skip 19-23 UTC]
---

Run the trend bot script.

Role:
- Scan trend continuation setups every minute without waiting for the trader loop
- Read `logs/bot_inventory_policy.json` and obey the LLM inventory director's pair-side policy
- Use `H1` and `M15` for trend alignment, `M5` for band-walk / follow-through setup, `M1` for execution timing
- Build cross-currency context from the 7-pair basket and only follow when the pulse supports the trend
- Place `trend_bot_market` MARKET orders only; do not leave passive trend orders behind
- Stay scalp-only. `FAST` / `MICRO` continuation lanes are small-lot repetition tools, not mini-swings
- Treat `stopLossOnFill` as a disaster backstop. The real continuation exit is TP1 / timeout / trap cleanup, and `bot-trade-manager` must flatten lingering worker trades before they become pseudo-swings
- Leave stranded inventory judgment and any unwind decisions to `inventory-director` and `qr-trader`

## Run

```bash
cd /Users/tossaki/App/QuantRabbit && python3 tools/trend_bot.py
```

Output the script's stdout as the full session summary.

- Exit 0: orders placed or conflicting pending orders cancelled
- Exit 1: no action taken (no trend setup / market closed / poison hour / policy blocked). Output only: SKIP
- Exit 2: error. Report it.

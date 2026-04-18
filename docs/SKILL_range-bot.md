---
name: range-bot
description: Heartbeat-safe range entry bot — keep/place range orders every minute, MARKET only for live S/A edge [Mon-Fri, late NY / pre-Tokyo = LIMIT only]
---

Run the range bot script.

Role:
- Scan range setups every minute without churning fresh pending bot orders
- Read `logs/bot_inventory_policy.json` and obey the LLM inventory director's pair-side policy
- Use `M5` as the setup frame, `M15` as the band-walk / fast-break veto frame, `H1` as the breakout-risk / range-confirmation frame, and `M1` as the micro entry-timing frame
- Build cross-currency context from the 7-pair basket and do not fade when the currency pulse points to a real breakout
- Place or keep `range_bot` LIMIT orders at BB extremes
- Cap passive LIMIT distance from live price so the worker does not park unrealistic far-away reloads that never fill
- Upgrade only live `S/A` edge setups with aligned `M1` micro timing to reduced-size `range_bot_market` MARKET entries, and allow strong `B` setups only in explicit `MICRO` harvest mode
- Do not fade band-walks just because price is sitting on a Bollinger edge
- Stay scalp-only. `FAST` / `MICRO` are high-turnover harvest lanes by design: small lot, repeated shots, fast cleanup if the bite fails
- Treat `stopLossOnFill` as a disaster backstop, not the main exit. If a worker trade cannot resolve quickly, `bot-trade-manager` should flatten it instead of letting it age into a swing hold
- Leave closeout-brake work to `bot-trade-manager`
- Leave stranded inventory judgment and pair-mode control to `inventory-director`

## Run

```bash
cd /Users/tossaki/App/QuantRabbit && python3 tools/range_bot.py
```

Output the script's stdout as the full session summary.

- Exit 0: orders placed, kept, or cancelled (normal operation)
- Exit 1: no action taken (no ranges / market closed / nothing to do). Output only: SKIP
- Exit 2: error. Report it.

Late NY / pre-Tokyo (`19:00-23:59 UTC`) is not a full stop. In that window the worker stays passive-only: LIMIT orders may remain or be placed, but MARKET chase is disabled.

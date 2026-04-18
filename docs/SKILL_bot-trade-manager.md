---
name: bot-trade-manager
description: Local scalp timeout + emergency brake for bot-tagged orders and trades [Mon-Fri, every 1 min]
---

Run the bot trade manager script.

Role:
- Review only bot-tagged pending orders and open trades (`range_bot`, `range_bot_market`, `trend_bot_market`)
- Keep the worker layer scalp-only by flattening stale bot scalps before they drift into trader-owned swing behavior
- Cancel dangerous pending orders when policy says pause/cancel, rollover starts, or panic margin is close
- Reduce or close bot trades when scalp timeout hits, or when closeout risk or deadlock pressure is already elevated
- Leave normal stranded-inventory judgment to `inventory-director`

## Run

```bash
cd /Users/tossaki/App/QuantRabbit && python3 tools/bot_trade_manager.py
```

Output the script's stdout as the full session summary.

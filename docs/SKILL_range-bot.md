---
name: range-bot
description: Range scalp LIMIT bot — detect ranges, place LIMITs at BB extremes [Mon-Fri, skip 19-23 UTC]
---

Run the range bot script. It handles everything autonomously:
range detection, margin check, order placement, Slack notification, and cleanup.

## Run

```bash
cd /Users/tossaki/App/QuantRabbit && python3 tools/range_bot.py
```

Output the script's stdout as the full session summary.

- Exit 0: orders placed or evaluated (normal operation)
- Exit 1: no action taken (no ranges / market closed / poison hour / margin full). Output only: SKIP
- Exit 2: error. Report it.

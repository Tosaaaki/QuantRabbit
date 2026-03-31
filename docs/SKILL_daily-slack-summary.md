---
name: daily-slack-summary
description: Auto-post daily trade summary to Slack #qr-daily every morning at 7:00 JST
---

Post the daily trade summary to the Slack #qr-daily channel.

Steps:
1. Run the following script:
```bash
cd /Users/tossaki/App/QuantRabbit && python3 tools/slack_daily_summary.py
```
2. If successful, done. If an error occurs, inspect the output and attempt a fix.

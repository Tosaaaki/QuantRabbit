---
name: intraday-pl-update
description: Post today's realized P&L to #qr-daily every 3 hours (weekdays 9:00-24:00 JST)
---

Post today's intraday P&L update to Slack #qr-daily using the dedicated script.

## Steps

1. Run the script:
```bash
cd /Users/tossaki/App/QuantRabbit && python3 tools/intraday_pl_update.py
```

2. If successful, done. If an error occurs, inspect the output and attempt a fix.

## Notes
- Weekdays only (Mon-Fri). FX market is closed on weekends
- The script fetches directly from OANDA transactions API (not log file parsing)
- "Today" is defined as the current JST date (00:00-23:59 JST)
- The daily-performance-report task sends a full-period report at 10:30 JST — this task is just a lightweight intraday update
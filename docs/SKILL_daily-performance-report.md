---
name: daily-performance-report
description: Aggregate realized P&L from OANDA at 10:30 JST daily and post a performance report to #qr-daily
---

Run the performance report script and confirm success.

## Steps

### 1. Run the report script

```bash
cd /Users/tossaki/App/QuantRabbit && python3 tools/daily_performance_report.py
```

This script:
- Fetches all ORDER_FILL transactions from OANDA since 2026-03-18 (handles pagination)
- Aggregates realized P&L by today / this week / all time
- Gets current account summary (Balance, NAV, Unrealized P&L)
- Posts formatted report to #qr-daily via Slack API

### 2. Confirm success

The script prints "Posted performance report to Slack" on success, or an error message on failure. If it fails, read the error and diagnose.

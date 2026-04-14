---
name: daily-performance-report
description: Aggregate realized P&L from OANDA at 10:30 JST daily and post a performance report to #qr-daily
---

Aggregate realized P&L from OANDA trade history and post a performance report to the Slack #qr-daily channel.

## Steps

### 1. Fetch realized P&L from OANDA API

Project directory: /Users/tossaki/App/QuantRabbit

Read credentials from config/env.toml:
```python
token = [l.split('=')[1].strip().strip('"') for l in open('config/env.toml').read().split('\n') if l.startswith('oanda_token')][0]
acct = [l.split('=')[1].strip().strip('"') for l in open('config/env.toml').read().split('\n') if l.startswith('oanda_account_id')][0]
base = 'https://api-fxtrade.oanda.com'
```

Retrieve via OANDA REST API v3:
- GET /v3/accounts/{acct}/summary → Balance, NAV, unrealizedPL
- GET /v3/accounts/{acct}/transactions?from=2026-03-18T00:00:00Z&to={today+1}T00:00:00Z → paginate all pages
- Aggregate non-zero `pl` fields by day

### 2. Aggregate

Calculate realized P&L for the following 4 periods:
- **Today**: realized P&L and close count for the current day
- **This week**: daily realized P&L from Monday through today, with total
- **This month**: daily realized P&L for the current month with cumulative trend
- **All time**: cumulative total since 2026-03-18 (system start date)

### 3. Post to Slack

Read from config/env.toml:
- slack_bot_token
- slack_channel_daily (= channel ID for #qr-daily)

POST to https://slack.com/api/chat.postMessage.

Format example:
```
📊 *QuantRabbit Performance Report* ({date} {time} JST)

*[Today {date}]* {pl} JPY ({N} closes)

*[This week {Monday}~{today}]* {total} JPY
├ {date}: {pl} JPY ({N} closes)
├ ...
└ {date}: {pl} JPY ({N} closes)

*[This month / All time since 3/18]*
{date}: {pl} JPY → ...
*Cumulative: {total} JPY* {✅ if positive, ⚠️ if negative}

*[Account status]*
Balance: {balance} JPY | NAV: {nav} JPY | Unrealized P&L: {upl} JPY
```

### 4. Confirm success

Check that the Slack API response contains ok: true. Display error details if it fails.

## Notes
- FX market is weekdays only (Mon-Fri)
- All-time start date is 2026-03-18 (QuantRabbit system launch date)
- OANDA transactions API is paginated — iterate all pages
- Round amounts down to the nearest JPY (no decimals)
---
name: daily-slack-summary
description: Auto-post daily trade summary to Slack #qr-daily every morning at 7:00 JST
---

Post the previous JST day's trade summary to the Slack `#qr-daily` channel.

Steps:
1. Run the script:
```bash
cd /Users/tossaki/App/QuantRabbit && PYTHONPATH=src python3 tools/slack_daily_summary.py
```
2. If it prints `Posted daily summary for YYYY-MM-DD to channel ...`, done.
3. If it prints `Already posted for YYYY-MM-DD, skipping`, done — duplicate guard.
4. On error, inspect the output and fix; rerun without arguments.

Manual / verification (does not post to Slack):
```bash
cd /Users/tossaki/App/QuantRabbit && PYTHONPATH=src python3 tools/slack_daily_summary.py --date YYYY-MM-DD --dry-run
```

Notes:
- Realized P&L source: OANDA `/v3/accounts/{id}/transactions` filtered to `ORDER_FILL` within the JST day window. Day boundary = JST 00:00–23:59 (= UTC 15:00 prev day → UTC 14:59 same day).
- Daily-realized-percent is `realized_pl / day_start_balance × 100`, where `day_start_balance` is reconstructed from the first fill's `accountBalance − pl`.
- The `Account Status` block is the **current** account state at report time — it is not an end-of-day historical NAV snapshot.
- Manual `--date` runs bypass the dedup lock at `logs/daily_summary_last.txt`. Auto-runs (cron) write the lock so the same date is never double-posted.
- Credentials live in `.env.local` (vNext §9). Required keys: `QR_OANDA_TOKEN`, `QR_OANDA_ACCOUNT_ID`, `QR_OANDA_BASE_URL`, `QR_SLACK_BOT_TOKEN`, `QR_SLACK_CHANNEL_DAILY`. Manual checks must use `--dry-run` unless the intent is to post a real Slack report.

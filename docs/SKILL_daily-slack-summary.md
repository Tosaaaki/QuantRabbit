---
name: daily-slack-summary
description: Build the daily trade summary without posting to Slack
---

Build the previous JST day's trade summary. Do not post it to Slack unless the
operator explicitly reverses the 2026-05-30 directive: 「Slackに送らないで」.

Steps:
1. Run the script in dry-run mode:
```bash
cd /Users/tossaki/App/QuantRabbit && PYTHONPATH=src python3 tools/slack_daily_summary.py --dry-run
```
2. Review the rendered summary in stdout.
3. On error, inspect the output and fix; rerun with `--dry-run`.

Manual / verification (does not post to Slack):
```bash
cd /Users/tossaki/App/QuantRabbit && PYTHONPATH=src python3 tools/slack_daily_summary.py --date YYYY-MM-DD --dry-run
```

Notes:
- Realized P&L source: OANDA `/v3/accounts/{id}/transactions` filtered to `ORDER_FILL` within the JST day window. Day boundary = JST 00:00–23:59 (= UTC 15:00 prev day → UTC 14:59 same day).
- Daily-realized-percent is `realized_pl / day_start_balance × 100`, where `day_start_balance` is reconstructed from the first fill's `accountBalance − pl`.
- The `Account Status` block is the **current** account state at report time — it is not an end-of-day historical NAV snapshot.
- Manual `--date` runs bypass the dedup lock at `logs/daily_summary_last.txt`. Auto-runs (cron) write the lock so the same date is never double-posted.
- Credentials live in `.env.local` (vNext §9). Required OANDA keys: `QR_OANDA_TOKEN`, `QR_OANDA_ACCOUNT_ID`, `QR_OANDA_BASE_URL`. Slack credentials are not needed for dry-run review.

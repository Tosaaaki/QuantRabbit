#!/usr/bin/env python3
"""Post daily summary to #qr-daily.

Uses OANDA transactions API for realized P&L (not log file parsing).
Day boundary = JST 00:00-23:59 (= UTC 15:00 prev day to UTC 14:59 same day).

Usage:
    python3 tools/slack_daily_summary.py [--date YYYY-MM-DD]
        --date: JST date to summarize (default: previous JST day)
"""
import urllib.request, json, sys, os, argparse
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
JST = timezone(timedelta(hours=9))


def load_config():
    cfg = {}
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'env.toml')
    for line in open(path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def oanda_get(path, token):
    url = f"https://api-fxtrade.oanda.com{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    return json.loads(urllib.request.urlopen(req, timeout=15).read())


def post(text, channel_id, token):
    payload = {"channel": channel_id, "text": text}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    if not resp.get('ok'):
        print(f"ERROR: {resp.get('error', 'unknown')}", file=sys.stderr)
        sys.exit(1)
    return resp


def get_account_summary(cfg):
    token = cfg['oanda_token']
    acct = cfg['oanda_account_id']
    data = oanda_get(f"/v3/accounts/{acct}/summary", token)['account']
    return {
        'balance': float(data['balance']),
        'nav': float(data['NAV']),
        'unrealized_pl': float(data['unrealizedPL']),
        'open_positions': int(data['openPositionCount']),
        'open_trades': int(data['openTradeCount']),
    }


def jst_day_to_utc_range(date_str):
    """Convert a JST date (YYYY-MM-DD) to UTC time range.

    JST 00:00 = UTC 15:00 previous day.
    Returns (from_utc, to_utc) strings for OANDA API.
    """
    target = datetime.strptime(date_str, "%Y-%m-%d")
    start = target - timedelta(hours=9)  # JST 00:00 in UTC
    end = target + timedelta(days=1) - timedelta(hours=9)  # next JST 00:00 in UTC
    return (
        start.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        end.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
    )


def get_realized_pl_for_date(token, acct, date_str):
    """Fetch realized P&L for a JST date from OANDA transactions API.

    date_str: YYYY-MM-DD (JST date).
    Day boundary: JST 00:00-23:59 (= UTC 15:00 prev day to UTC 14:59 same day).

    Returns (total_pl, entry_count, close_count, day_start_balance).
    day_start_balance is the account balance right before the first fill of the day.
    """
    from_utc, to_utc = jst_day_to_utc_range(date_str)

    total_pl = 0.0
    close_count = 0
    entry_count = 0
    day_start_balance = None

    path = f"/v3/accounts/{acct}/transactions?from={from_utc}&to={to_utc}&type=ORDER_FILL"
    try:
        data = oanda_get(path, token)
    except Exception as e:
        print(f"WARN: transactions API error: {e}", file=sys.stderr)
        return total_pl, entry_count, close_count, day_start_balance

    pages = data.get("pages", [])
    for page_url in pages:
        try:
            req = urllib.request.Request(page_url, headers={"Authorization": f"Bearer {token}"})
            page_data = json.loads(urllib.request.urlopen(req, timeout=15).read())
        except Exception as e:
            print(f"WARN: page fetch error: {e}", file=sys.stderr)
            continue

        for tx in page_data.get("transactions", []):
            if tx.get("type") != "ORDER_FILL":
                continue
            pl = float(tx.get("pl", "0"))
            # Capture day-start balance from the first transaction
            if day_start_balance is None:
                acct_bal = float(tx.get("accountBalance", "0"))
                day_start_balance = acct_bal - pl
            if pl != 0:
                total_pl += pl
                close_count += 1
            else:
                entry_count += 1

    return total_pl, entry_count, close_count, day_start_balance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None, help='JST date YYYY-MM-DD (default: previous JST day)')
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        # Previous JST day
        target_date = (datetime.now(JST) - timedelta(days=1)).strftime('%Y-%m-%d')

    # Dedup guard: skip if already posted for this date
    lock_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    lock_file = os.path.join(lock_dir, 'daily_summary_last.txt')
    if not args.date:  # only guard auto-runs, not manual --date runs
        try:
            if os.path.exists(lock_file) and open(lock_file).read().strip() == target_date:
                print(f"Already posted for {target_date}, skipping")
                return
        except Exception:
            pass

    cfg = load_config()
    acct_summary = get_account_summary(cfg)
    realized_pl, entry_count, close_count, day_start_balance = get_realized_pl_for_date(
        cfg['oanda_token'], cfg['oanda_account_id'], target_date
    )

    # Build summary
    lines = []
    lines.append(f"\U0001f4ca *Daily Summary: {target_date}*")
    lines.append("")

    # Daily P&L with % change (use day-start balance for accurate %)
    pl_icon = "\U0001f7e2" if realized_pl >= 0 else "\U0001f534"
    if day_start_balance and day_start_balance > 0:
        pct_change = realized_pl / day_start_balance * 100
    else:
        pct_change = 0
    lines.append(f"{pl_icon} *Daily Realized P&L: {realized_pl:+,.0f}JPY ({pct_change:+.2f}%)*")

    # Trade count
    if entry_count > 0 or close_count > 0:
        lines.append(f"\U0001f4dd Entries: {entry_count} | Closes: {close_count}")
    else:
        lines.append(f"\U0001f4dd No trade records")

    lines.append("")

    # Account status
    lines.append("*Account Status:*")
    lines.append(f"  Balance: {acct_summary['balance']:,.0f}JPY")
    lines.append(f"  NAV: {acct_summary['nav']:,.0f}JPY")
    lines.append(f"  Unrealized P&L: {acct_summary['unrealized_pl']:+,.0f}JPY")
    lines.append(f"  Open: {acct_summary['open_trades']} trades ({acct_summary['open_positions']} pairs)")

    message = "\n".join(lines)
    channel = cfg.get('slack_channel_daily', cfg.get('slack_channel_id', ''))

    post(message, channel, cfg['slack_bot_token'])
    print(f"Posted daily summary for {target_date} to #{channel}")

    # Record posted date to prevent duplicates
    if not args.date:
        try:
            with open(lock_file, 'w') as f:
                f.write(target_date)
        except Exception:
            pass


if __name__ == '__main__':
    main()

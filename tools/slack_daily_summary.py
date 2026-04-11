#!/usr/bin/env python3
"""Post daily summary to #qr-daily.

Uses OANDA transactions API for realized P&L (not log file parsing).
Day boundary = UTC 00:00-23:59 (= JST 09:00-08:59), matching trader task.

Usage:
    python3 tools/slack_daily_summary.py [--date YYYY-MM-DD]
        --date: UTC date to summarize (default: previous UTC day)
"""
import urllib.request, json, sys, os, argparse
from datetime import datetime, timedelta, timezone

UTC = timezone.utc


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


def get_realized_pl_for_date(token, acct, date_str):
    """Fetch realized P&L for a UTC date from OANDA transactions API.

    date_str: YYYY-MM-DD (UTC date).
    Day boundary: UTC 00:00:00 to 23:59:59.
    """
    from_utc = f"{date_str}T00:00:00.000000000Z"
    next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    to_utc = f"{next_day}T00:00:00.000000000Z"

    total_pl = 0.0
    close_count = 0
    entry_count = 0

    path = f"/v3/accounts/{acct}/transactions?from={from_utc}&to={to_utc}&type=ORDER_FILL"
    try:
        data = oanda_get(path, token)
    except Exception as e:
        print(f"WARN: transactions API error: {e}", file=sys.stderr)
        return total_pl, entry_count, close_count

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
            if pl != 0:
                total_pl += pl
                close_count += 1
            else:
                entry_count += 1

    return total_pl, entry_count, close_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None, help='UTC date YYYY-MM-DD (default: previous UTC day)')
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        # Previous UTC day (this script runs at 07:00 JST = 22:00 UTC previous day)
        target_date = (datetime.now(UTC) - timedelta(days=1)).strftime('%Y-%m-%d')

    cfg = load_config()
    acct_summary = get_account_summary(cfg)
    realized_pl, entry_count, close_count = get_realized_pl_for_date(
        cfg['oanda_token'], cfg['oanda_account_id'], target_date
    )

    # Build summary
    lines = []
    lines.append(f"\U0001f4ca *Daily Summary: {target_date}*")
    lines.append("")

    # Daily P&L
    pl_icon = "\U0001f7e2" if realized_pl >= 0 else "\U0001f534"
    lines.append(f"{pl_icon} *Daily Realized P&L: {realized_pl:+,.0f}JPY*")

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


if __name__ == '__main__':
    main()

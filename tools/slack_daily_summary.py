#!/usr/bin/env python3
"""Post daily summary to #qr-daily
Usage: python3 tools/slack_daily_summary.py [--date YYYY-MM-DD]
"""
import urllib.request, json, sys, os, argparse, glob
from datetime import datetime, timedelta


def load_config():
    cfg = {}
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'env.toml')
    for line in open(path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


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
    req = urllib.request.Request(
        f"https://api-fxtrade.oanda.com/v3/accounts/{acct}/summary",
        headers={"Authorization": f"Bearer {token}"}
    )
    data = json.loads(urllib.request.urlopen(req).read())['account']
    return {
        'balance': float(data['balance']),
        'nav': float(data['NAV']),
        'unrealized_pl': float(data['unrealizedPL']),
        'margin_used': float(data['marginUsed']),
        'open_positions': int(data['openPositionCount']),
        'open_trades': int(data['openTradeCount']),
    }


def get_daily_trades(date_str):
    """Aggregate entries/closes for the day from trades.md"""
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'collab_trade', 'daily', date_str)
    trades_path = os.path.join(base, 'trades.md')
    if not os.path.exists(trades_path):
        return None

    content = open(trades_path).read()
    entries = content.count('ENTRY') + content.count('entry') + content.count('Entry')
    closes = content.count('CLOSE') + content.count('close') + content.count('Close') + content.count('決済')
    return {'entries': entries, 'closes': closes, 'has_data': True}


def get_daily_pl_from_log(date_str):
    """Aggregate realized P&L for the day from live_trade_log.txt"""
    log_path = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'live_trade_log.txt')
    if not os.path.exists(log_path):
        return 0.0

    total_pl = 0.0
    for line in open(log_path):
        if date_str not in line:
            continue
        # Extract P/L: PL=+123.45 or P/L=-67.89 pattern
        if 'PL=' in line or 'P/L=' in line:
            try:
                sep = 'P/L=' if 'P/L=' in line else 'PL='
                pl_part = line.split(sep)[1].split()[0].replace('円', '').replace(',', '').rstrip('JPY').rstrip('J')
                total_pl += float(pl_part)
            except (ValueError, IndexError):
                pass
        # realized_pl pattern
        elif 'realized_pl' in line:
            try:
                import re
                m = re.search(r'realized_pl["\s:=]+([+-]?[\d.]+)', line)
                if m:
                    total_pl += float(m.group(1))
            except (ValueError, AttributeError):
                pass
    return total_pl


def get_performance_summary():
    """Get win rate etc. from trade_performance.py results"""
    import subprocess
    perf_path = os.path.join(os.path.dirname(__file__), 'trade_performance.py')
    try:
        result = subprocess.run(
            [sys.executable, perf_path],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.join(os.path.dirname(__file__), '..', '..')
        )
        output = result.stdout
        # Find the Overall line
        for line in output.split('\n'):
            if 'Overall' in line and 'WR=' in line:
                return line.strip()
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None, help='YYYY-MM-DD (default: yesterday)')
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    cfg = load_config()
    acct = get_account_summary(cfg)
    daily_pl = get_daily_pl_from_log(target_date)
    trades_info = get_daily_trades(target_date)
    perf_line = get_performance_summary()

    # Build summary
    lines = []
    lines.append(f"\U0001f4ca *Daily Summary: {target_date}*")
    lines.append("")

    # Daily P&L
    pl_icon = "\U0001f7e2" if daily_pl >= 0 else "\U0001f534"
    lines.append(f"{pl_icon} *Daily Realized P&L: {daily_pl:+,.0f}JPY*")

    # Trade count
    if trades_info and trades_info['has_data']:
        lines.append(f"\U0001f4dd Entries: {trades_info['entries']} | Closes: {trades_info['closes']}")
    else:
        lines.append(f"\U0001f4dd No trade records")

    lines.append("")

    # Account status
    lines.append("*Account Status:*")
    lines.append(f"  Balance: {acct['balance']:,.0f}JPY")
    lines.append(f"  NAV: {acct['nav']:,.0f}JPY")
    lines.append(f"  Unrealized P&L: {acct['unrealized_pl']:+,.0f}JPY")
    lines.append(f"  Open: {acct['open_trades']} trades ({acct['open_positions']} pairs)")

    # Performance
    if perf_line:
        lines.append("")
        lines.append(f"*Overall:* {perf_line}")

    message = "\n".join(lines)
    channel = cfg.get('slack_channel_daily', cfg.get('slack_channel_id', ''))

    post(message, channel, cfg['slack_bot_token'])
    print(f"Posted daily summary for {target_date} to #{channel}")


if __name__ == '__main__':
    main()

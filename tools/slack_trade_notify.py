#!/usr/bin/env python3
"""Post trade notifications to #qr-trades
Usage:
  python3 tools/slack_trade_notify.py entry --pair USD_JPY --side LONG --units 5000 --price 158.42 [--sl 158.00] [--thesis "Fed hawkish"]
  python3 tools/slack_trade_notify.py modify --pair USD_JPY --action "TP half-close" --units 2500 --price 158.55 --pl "+13pip, +1,625JPY" [--note "remaining 2500, move to BE"]
  python3 tools/slack_trade_notify.py close --pair USD_JPY --side LONG --units 5000 --price 158.60 --pl "+18pip, +2,250JPY" [--total_pl "+3,875JPY"]
"""
import urllib.request, json, sys, os, argparse
from datetime import datetime


def load_config():
    cfg = {}
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'env.toml')
    for line in open(path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def post(text, channel_id, thread_ts=None):
    cfg = load_config()
    token = cfg['slack_bot_token']
    payload = {"channel": channel_id, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
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


def format_entry(args):
    now = datetime.now().strftime('%H:%M')
    icon = "\U0001f7e2" if args.side == "LONG" else "\U0001f534"
    sl_text = f"SL: {args.sl}" if args.sl else "SL: none (discretionary)"
    thesis_text = f"\n  Thesis: {args.thesis}" if args.thesis else ""
    return f"{icon} {args.side} {args.pair} {args.units}units @{args.price}  [{now}]\n  {sl_text}{thesis_text}"


def format_modify(args):
    now = datetime.now().strftime('%H:%M')
    note_text = f"\n  {args.note}" if args.note else ""
    return f"\U0001f504 {args.pair} {args.action} {args.units}units @{args.price} ({args.pl})  [{now}]{note_text}"


def get_today_realized_pl():
    """Fetch today's realized P&L from OANDA Transaction API (ground truth)."""
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'env.toml')
        cfg_lines = open(cfg_path).read().split('\n')
        token = [l.split('=')[1].strip().strip('"') for l in cfg_lines if l.startswith('oanda_token')][0]
        acct = [l.split('=')[1].strip().strip('"') for l in cfg_lines if l.startswith('oanda_account_id')][0]

        # OANDA day boundary: 5 PM ET (21:00 UTC summer, 22:00 UTC winter)
        from datetime import timezone, timedelta
        now_utc = datetime.now(timezone.utc)
        # Use 21:00 UTC as boundary (summer time). If before 21:00, start from yesterday 21:00
        boundary = now_utc.replace(hour=21, minute=0, second=0, microsecond=0)
        if now_utc.hour < 21:
            boundary -= timedelta(days=1)
        from_time = boundary.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')

        url = f'https://api-fxtrade.oanda.com/v3/accounts/{acct}/transactions?from={from_time}'
        req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())

        total_pl = 0.0
        for page_url in data.get('pages', []):
            req2 = urllib.request.Request(page_url, headers={'Authorization': f'Bearer {token}'})
            txns = json.loads(urllib.request.urlopen(req2, timeout=10).read()).get('transactions', [])
            for t in txns:
                if t.get('type') == 'ORDER_FILL':
                    total_pl += float(t.get('pl', '0'))
        return total_pl
    except Exception:
        return None


def format_close(args):
    now = datetime.now().strftime('%H:%M')
    # Auto-fetch today's realized P&L from OANDA (ignore --total_pl from caller)
    today_pl = get_today_realized_pl()
    if today_pl is not None:
        total = f"\n  Total realized P&L: Today: {today_pl:+.1f} JPY realized"
    elif args.total_pl:
        total = f"\n  Total realized P&L: {args.total_pl}"
    else:
        total = ""
    return f"\u2B1B {args.pair} {args.side} FULL CLOSE {args.units}units @{args.price} ({args.pl})  [{now}]{total}"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    entry = sub.add_parser('entry')
    entry.add_argument('--pair', required=True)
    entry.add_argument('--side', required=True)
    entry.add_argument('--units', required=True)
    entry.add_argument('--price', required=True)
    entry.add_argument('--sl', default=None)
    entry.add_argument('--thesis', default=None)
    entry.add_argument('--thread', default=None)

    modify = sub.add_parser('modify')
    modify.add_argument('--pair', required=True)
    modify.add_argument('--action', required=True)
    modify.add_argument('--units', required=True)
    modify.add_argument('--price', required=True)
    modify.add_argument('--pl', required=True)
    modify.add_argument('--note', default=None)
    modify.add_argument('--thread', default=None)

    close = sub.add_parser('close')
    close.add_argument('--pair', required=True)
    close.add_argument('--side', required=True)
    close.add_argument('--units', required=True)
    close.add_argument('--price', required=True)
    close.add_argument('--pl', required=True)
    close.add_argument('--total_pl', default=None)
    close.add_argument('--thread', default=None)

    args = parser.parse_args()
    cfg = load_config()
    channel = cfg['slack_channel_trades']

    if args.cmd == 'entry':
        text = format_entry(args)
    elif args.cmd == 'modify':
        text = format_modify(args)
    elif args.cmd == 'close':
        text = format_close(args)
    else:
        parser.print_help()
        sys.exit(1)

    resp = post(text, channel, thread_ts=args.thread)
    ts = resp.get('ts', '')
    print(f"OK: posted (ts={ts})")


if __name__ == '__main__':
    main()

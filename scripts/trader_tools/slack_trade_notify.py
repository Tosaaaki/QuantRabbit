#!/usr/bin/env python3
"""トレード通知を #qr-trades に投稿する
Usage:
  python3 scripts/trader_tools/slack_trade_notify.py entry --pair USD_JPY --side LONG --units 5000 --price 158.42 [--sl 158.00] [--thesis "Fed hawkish"]
  python3 scripts/trader_tools/slack_trade_notify.py modify --pair USD_JPY --action "TP半利確" --units 2500 --price 158.55 --pl "+13pip, +1,625円" [--note "残2500, BE移動"]
  python3 scripts/trader_tools/slack_trade_notify.py close --pair USD_JPY --side LONG --units 5000 --price 158.60 --pl "+18pip, +2,250円" [--total_pl "+3,875円"]
"""
import urllib.request, json, sys, os, argparse
from datetime import datetime


def load_config():
    cfg = {}
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'env.toml')
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
    sl_text = f"SL: {args.sl}" if args.sl else "SL: なし(裁量)"
    thesis_text = f"\n  テーゼ: {args.thesis}" if args.thesis else ""
    return f"{icon} {args.side} {args.pair} {args.units}units @{args.price}  [{now}]\n  {sl_text}{thesis_text}"


def format_modify(args):
    now = datetime.now().strftime('%H:%M')
    note_text = f"\n  {args.note}" if args.note else ""
    return f"\U0001f504 {args.pair} {args.action} {args.units}units @{args.price} ({args.pl})  [{now}]{note_text}"


def format_close(args):
    now = datetime.now().strftime('%H:%M')
    total = f"\n  確定益合計: {args.total_pl}" if args.total_pl else ""
    return f"\u2B1B {args.pair} {args.side} 全決済 {args.units}units @{args.price} ({args.pl})  [{now}]{total}"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action')

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

    if args.action == 'entry':
        text = format_entry(args)
    elif args.action == 'modify':
        text = format_modify(args)
    elif args.action == 'close':
        text = format_close(args)
    else:
        parser.print_help()
        sys.exit(1)

    resp = post(text, channel, thread_ts=args.thread)
    ts = resp.get('ts', '')
    print(f"OK: posted (ts={ts})")


if __name__ == '__main__':
    main()

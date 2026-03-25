#!/usr/bin/env python3
"""Slack チャンネルの最新メッセージを読む（urllib版）
Usage: python3 scripts/trader_tools/slack_read.py [--limit N] [--channel CHANNEL_ID]
"""
import urllib.request, urllib.parse, json, sys, os

def load_config():
    cfg = {}
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'env.toml')
    for line in open(path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg

def read_messages(channel_id=None, limit=10):
    cfg = load_config()
    token = cfg['slack_bot_token']
    channel = channel_id or cfg.get('slack_channel_id', '')

    params = urllib.parse.urlencode({'channel': channel, 'limit': limit})
    url = f"https://slack.com/api/conversations.history?{params}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    })

    resp = json.loads(urllib.request.urlopen(req).read())

    if not resp.get('ok'):
        print(f"ERROR: {resp.get('error', 'unknown')}", file=sys.stderr)
        sys.exit(1)

    return resp.get('messages', [])

def format_messages(messages):
    from datetime import datetime
    lines = []
    for msg in reversed(messages):
        ts = datetime.fromtimestamp(float(msg['ts']))
        user = msg.get('user', msg.get('bot_id', 'unknown'))
        text = msg.get('text', '')
        lines.append(f"[{ts:%H:%M:%S}] {user}: {text}")
    return '\n'.join(lines)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--channel', type=str, default=None)
    parser.add_argument('--json', action='store_true', help='Output raw JSON')
    args = parser.parse_args()

    messages = read_messages(channel_id=args.channel, limit=args.limit)

    if args.json:
        print(json.dumps(messages, ensure_ascii=False, indent=2))
    else:
        print(format_messages(messages))

#!/usr/bin/env python3
"""Read latest messages from a Slack channel (urllib version)
Usage: python3 tools/slack_read.py [--limit N] [--channel CHANNEL_ID] [--user-only] [--after TS] [--json]
"""
import urllib.request, urllib.parse, json, sys, os

BOT_USER_ID = "U0AP9UF8XL0"

def load_config():
    cfg = {}
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'env.toml')
    for line in open(path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg

def read_messages(channel_id=None, limit=10, after=None):
    cfg = load_config()
    token = cfg['slack_bot_token']
    channel = channel_id or cfg.get('slack_channel_id', '')

    params = {'channel': channel, 'limit': limit}
    if after:
        params['oldest'] = after
    url = f"https://slack.com/api/conversations.history?{urllib.parse.urlencode(params)}"
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
    parser.add_argument('--user-only', action='store_true',
                        help='Exclude bot posts (U0AP9UF8XL0) and show only human posts')
    parser.add_argument('--after', type=str, default=None,
                        help='Only fetch messages after this timestamp (Slack ts format)')
    args = parser.parse_args()

    messages = read_messages(channel_id=args.channel, limit=args.limit, after=args.after)

    if args.user_only:
        messages = [m for m in messages
                    if m.get('user') != BOT_USER_ID
                    and 'bot_id' not in m
                    and m.get('subtype') != 'bot_message']

    if args.json:
        print(json.dumps(messages, ensure_ascii=False, indent=2))
    else:
        print(format_messages(messages))

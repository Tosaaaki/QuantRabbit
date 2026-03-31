#!/usr/bin/env python3
"""Post a message to a Slack channel (urllib version)
Usage: python3 tools/slack_post.py "message" [--channel CHANNEL_ID] [--thread TS]
"""
import urllib.request, json, sys, os

def load_config():
    cfg = {}
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'env.toml')
    for line in open(path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg

def post_message(text, channel_id=None, thread_ts=None):
    cfg = load_config()
    token = cfg['slack_bot_token']
    channel = channel_id or cfg.get('slack_channel_id', '')

    payload = {
        "channel": channel,
        "text": text,
    }
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('message', type=str, help='Message to post')
    parser.add_argument('--channel', type=str, default=None)
    parser.add_argument('--thread', type=str, default=None, help='Thread ts')
    args = parser.parse_args()

    result = post_message(args.message, channel_id=args.channel, thread_ts=args.thread)
    ts = result.get('ts', '')
    print(f"OK: posted (ts={ts})")

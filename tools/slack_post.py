#!/usr/bin/env python3
"""Post a message to a Slack channel.

Reads Slack credentials from `.env.local` (vNext convention; AGENT_CONTRACT §9
forbids `config/env.toml`). Override the env file with `QR_SLACK_ENV_FILE` or
share `QR_OANDA_ENV_FILE` (the loader falls back to it).

Slack posting is disabled by default. Set `QR_SLACK_SEND_ENABLE=1` in the
process environment to allow a real `chat.postMessage` call.

Usage:
    python3 tools/slack_post.py "message" [--channel CHANNEL_ID] [--thread TS]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ALLOWED_KEYS = {
    "QR_SLACK_BOT_TOKEN",
    "QR_SLACK_CHANNEL_ID",
    "QR_SLACK_CHANNEL_TRADES",
    "QR_SLACK_CHANNEL_DAILY",
    "QR_SLACK_CHANNEL_COMMANDS",
}

TRUE_VALUES = {"1", "true", "TRUE", "yes", "YES"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _env_file() -> Path:
    override = os.environ.get("QR_SLACK_ENV_FILE") or os.environ.get("QR_OANDA_ENV_FILE")
    if override:
        return Path(override)
    return _repo_root() / ".env.local"


def _clean(value: str) -> str:
    text = value.strip()
    if "#" in text and not (text.startswith('"') or text.startswith("'")):
        text = text.split("#", 1)[0].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text


def load_slack_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    path = _env_file()
    if path.exists():
        for raw in path.read_text(errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in ALLOWED_KEYS:
                cfg[key] = _clean(value)
    for key in ALLOWED_KEYS:
        if key in os.environ:
            cfg[key] = os.environ[key]
    return cfg


def slack_send_enabled() -> bool:
    return os.environ.get("QR_SLACK_SEND_ENABLE", "").strip() in TRUE_VALUES


def post_message(text: str, channel_id: str, token: str, thread_ts: str | None = None) -> dict:
    if not slack_send_enabled():
        print("SKIP: Slack posting disabled (set QR_SLACK_SEND_ENABLE=1 to send)")
        return {"ok": True, "skipped": True, "reason": "slack_disabled"}

    payload: dict[str, object] = {"channel": channel_id, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"ERROR: HTTP {exc.code}: {exc.read().decode(errors='ignore')}", file=sys.stderr)
        sys.exit(1)
    if not body.get("ok"):
        print(f"ERROR: {body.get('error', 'unknown')}", file=sys.stderr)
        sys.exit(1)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("message", help="Message to post")
    parser.add_argument("--channel", default=None, help="Channel id (default: QR_SLACK_CHANNEL_ID)")
    parser.add_argument("--thread", default=None, help="Thread ts to reply into")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be sent")
    args = parser.parse_args()

    cfg = load_slack_config()
    token = cfg.get("QR_SLACK_BOT_TOKEN")
    channel = args.channel or cfg.get("QR_SLACK_CHANNEL_ID")

    if args.dry_run:
        print(f"[dry-run] channel={channel} thread={args.thread}")
        print(args.message)
        return

    if not slack_send_enabled():
        post_message(args.message, channel or "", token or "", thread_ts=args.thread)
        return

    if not token:
        print("ERROR: QR_SLACK_BOT_TOKEN missing in .env.local", file=sys.stderr)
        sys.exit(2)
    if not channel:
        print("ERROR: --channel or QR_SLACK_CHANNEL_ID required", file=sys.stderr)
        sys.exit(2)

    resp = post_message(args.message, channel, token, thread_ts=args.thread)
    print(f"OK: posted (ts={resp.get('ts', '')})")


if __name__ == "__main__":
    main()

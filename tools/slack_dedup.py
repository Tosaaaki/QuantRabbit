#!/usr/bin/env python3
"""Slack reply deduplication — file-based lock to prevent duplicate replies.

Mechanism:
  - Each replied-to message ts is recorded in logs/.slack_replied_ts
  - Before replying, check if the ts is already in the file
  - After replying, atomically append the ts

Usage as library:
    from slack_dedup import is_already_replied, mark_replied

Usage as CLI (for Bash integration):
    python3 tools/slack_dedup.py check 1712345678.123456   # exit 0 = not replied, exit 1 = already replied
    python3 tools/slack_dedup.py mark  1712345678.123456   # record that we replied
    python3 tools/slack_dedup.py list                      # show all replied ts
    python3 tools/slack_dedup.py clean                     # remove entries older than 48h
"""
import os
import sys
import time
import fcntl
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DEDUP_FILE = ROOT / "logs" / ".slack_replied_ts"
MAX_AGE_SECONDS = 48 * 3600  # 48 hours


def _read_entries():
    """Read all (ts, epoch) entries from dedup file."""
    if not DEDUP_FILE.exists():
        return []
    entries = []
    for line in DEDUP_FILE.read_text().strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        ts = parts[0]
        epoch = float(parts[1]) if len(parts) > 1 else 0
        entries.append((ts, epoch))
    return entries


def is_already_replied(message_ts: str) -> bool:
    """Check if we already replied to this message ts."""
    entries = _read_entries()
    replied_set = {e[0] for e in entries}
    return message_ts in replied_set


def mark_replied(message_ts: str) -> None:
    """Record that we replied to this message ts. Atomic with file lock."""
    if is_already_replied(message_ts):
        return
    DEDUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DEDUP_FILE, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(f"{message_ts} {time.time():.0f}\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


## --- Layer 2: Recent post cooldown (content-based) ---

POST_LOG_FILE = ROOT / "logs" / ".slack_post_log"
COOLDOWN_SECONDS = 30 * 60  # 30 minutes


def _normalize(text: str) -> str:
    """Normalize message for comparison: lowercase, collapse whitespace, first 30 chars."""
    return text.strip().lower().replace("\n", " ").replace("\\n", " ")[:30]


def _is_similar(a: str, b: str) -> bool:
    """Check if two messages are similar enough to be duplicates.
    Uses prefix matching: if one starts with the other (after normalization), it's a dup."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    # At least 10 chars must match (avoid matching very short strings like "OK")
    min_len = min(len(na), len(nb))
    if min_len < 10:
        return na == nb
    return na.startswith(nb) or nb.startswith(na)


def check_recent_post_cooldown(message: str, channel: str) -> Optional[str]:
    """Check if a similar message was posted recently. Returns time-ago string or None."""
    if not POST_LOG_FILE.exists():
        return None
    now = time.time()
    for line in POST_LOG_FILE.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ep, ch, content = float(parts[0]), parts[1], parts[2]
        age = now - ep
        if age < COOLDOWN_SECONDS and ch == channel and _is_similar(message, content):
            mins = int(age // 60)
            return f"{mins}m ago" if mins > 0 else f"{int(age)}s ago"
    return None


def log_post(message: str, channel: str) -> None:
    """Log a posted message for cooldown checking."""
    POST_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POST_LOG_FILE, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            # Keep first 100 chars for matching
            content = message.replace("\n", "\\n").replace("\t", " ")[:100]
            f.write(f"{time.time():.0f}\t{channel}\t{content}\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    # Prune old entries (keep last 48h)
    _prune_post_log()


def _prune_post_log():
    """Remove post log entries older than 48h."""
    if not POST_LOG_FILE.exists():
        return
    now = time.time()
    kept = []
    for line in POST_LOG_FILE.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 1:
            try:
                if now - float(parts[0]) < MAX_AGE_SECONDS:
                    kept.append(line)
            except ValueError:
                pass
    POST_LOG_FILE.write_text("\n".join(kept) + "\n" if kept else "")


def clean_old_entries() -> int:
    """Remove entries older than MAX_AGE_SECONDS. Returns count removed."""
    entries = _read_entries()
    now = time.time()
    kept = [(ts, ep) for ts, ep in entries if now - ep < MAX_AGE_SECONDS]
    removed = len(entries) - len(kept)
    if removed > 0:
        with open(DEDUP_FILE, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                for ts, ep in kept:
                    f.write(f"{ts} {ep:.0f}\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    return removed


def main():
    if len(sys.argv) < 2:
        print("Usage: slack_dedup.py {check|mark|list|clean} [message_ts]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "check":
        if len(sys.argv) < 3:
            print("Usage: slack_dedup.py check <message_ts>", file=sys.stderr)
            sys.exit(2)
        ts = sys.argv[2]
        if is_already_replied(ts):
            print(f"ALREADY_REPLIED ts={ts}")
            sys.exit(1)
        else:
            print(f"NOT_REPLIED ts={ts}")
            sys.exit(0)

    elif cmd == "mark":
        if len(sys.argv) < 3:
            print("Usage: slack_dedup.py mark <message_ts>", file=sys.stderr)
            sys.exit(2)
        ts = sys.argv[2]
        mark_replied(ts)
        print(f"MARKED ts={ts}")

    elif cmd == "list":
        entries = _read_entries()
        for ts, ep in entries:
            print(f"{ts}  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ep))}")
        print(f"Total: {len(entries)} entries")

    elif cmd == "clean":
        removed = clean_old_entries()
        entries = _read_entries()
        print(f"Cleaned {removed} old entries. {len(entries)} remaining.")

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

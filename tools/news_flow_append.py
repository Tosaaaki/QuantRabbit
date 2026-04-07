#!/usr/bin/env python3
"""
news_flow_append.py — Append a compact hourly snapshot to logs/news_flow_log.md.

Run after qr-news-digest writes news_digest.md.
Keeps 48h of hourly snapshots so the trader (and daily-review) can read
how the macro narrative evolved over the day.

Usage:
    python3 tools/news_flow_append.py
"""

import re
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))
BASE = Path(__file__).parent.parent
DIGEST = BASE / "logs" / "news_digest.md"
FLOW_LOG = BASE / "logs" / "news_flow_log.md"
MAX_ENTRIES = 48  # 48 hours of hourly snapshots


def extract_snapshot(digest_text: str) -> dict:
    """Pull key fields from news_digest.md."""
    # Timestamp from header
    ts_match = re.search(r"# FX News Digest — (.+)", digest_text)
    ts = ts_match.group(1).strip() if ts_match else datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")

    # Top High Impact item (first bold title in the 🔴 section)
    hi_section = re.search(r"## 🔴 High Impact.*?(?=## 🟡|## 📅|$)", digest_text, re.DOTALL)
    hot = "none"
    theme = "no high impact"
    if hi_section:
        items = re.findall(r"\*\*([^*\n]+?)\*\*", hi_section.group(0))
        if items:
            hot = items[0][:80]
        # Theme = first parenthetical after bold title (e.g. "NFP BEAT (April 4…)")
        theme_match = re.search(r"\*\*[^*]+\*\*[^\n]*\n+[-—]\s*(.+)", hi_section.group(0))
        if theme_match:
            theme = theme_match.group(1).strip()[:120]
        else:
            theme = hot

    # Top Watch item
    wl_section = re.search(r"## 🟡 Watch List.*?(?=## 📅|## 🔴|$)", digest_text, re.DOTALL)
    watch = "none"
    if wl_section:
        w_items = re.findall(r"\*\*([^*\n]+?)\*\*", wl_section.group(0))
        if w_items:
            watch = w_items[0][:60]

    return {"ts": ts, "hot": hot, "theme": theme, "watch": watch}


def already_logged(flow_text: str, ts: str) -> bool:
    """Avoid duplicate entries for the same timestamp."""
    return f"## {ts}" in flow_text


def append_snapshot(snap: dict) -> None:
    """Append compact entry to news_flow_log.md."""
    entry = (
        f"## {snap['ts']}\n"
        f"HOT: {snap['hot']}\n"
        f"THEME: {snap['theme']}\n"
        f"WATCH: {snap['watch']}\n\n"
    )

    existing = FLOW_LOG.read_text(encoding="utf-8") if FLOW_LOG.exists() else ""

    if already_logged(existing, snap["ts"]):
        print(f"SKIP — already logged: {snap['ts']}")
        return

    updated = existing + entry
    FLOW_LOG.write_text(updated, encoding="utf-8")
    print(f"APPENDED — {snap['ts']}: {snap['hot'][:50]}")


def trim_log() -> None:
    """Keep only the most recent MAX_ENTRIES entries."""
    if not FLOW_LOG.exists():
        return

    text = FLOW_LOG.read_text(encoding="utf-8")
    # Split on "## YYYY" pattern
    parts = re.split(r"(?=^## \d{4}-)", text, flags=re.MULTILINE)
    parts = [p for p in parts if p.strip()]

    if len(parts) > MAX_ENTRIES:
        trimmed = parts[-MAX_ENTRIES:]
        FLOW_LOG.write_text("".join(trimmed), encoding="utf-8")
        print(f"TRIMMED — kept {MAX_ENTRIES} entries (dropped {len(parts) - MAX_ENTRIES})")
    else:
        print(f"OK — {len(parts)} entries in log")


def main():
    if not DIGEST.exists():
        print("ERROR — news_digest.md not found. Run qr-news-digest first.")
        return 1

    digest_text = DIGEST.read_text(encoding="utf-8")
    snap = extract_snapshot(digest_text)
    append_snapshot(snap)
    trim_log()
    return 0


if __name__ == "__main__":
    exit(main())

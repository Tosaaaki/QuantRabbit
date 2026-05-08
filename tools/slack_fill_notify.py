#!/usr/bin/env python3
"""Post per-trade fill / close / cancel events to Slack `#qr-trades`.

Reads `data/execution_ledger.db` for `execution_events` newer than the
last-seen `inserted_at_utc` marker stored in `logs/.slack_fill_last_seen`,
then posts one Slack message per ORDER_FILLED / TRADE_CLOSED /
ORDER_CANCELED event. PROTECTION_CREATED is folded into the matching
ORDER_FILLED line so a single fill produces a single notification with
its TP attached.

Designed to be called from `scripts/run-autotrade-live.sh` after
`autotrade-cycle` completes. Idempotent: re-running with no new events
posts nothing and exits 0. Manual `--since YYYY-MM-DDTHH:MM:SS+00:00`
overrides the marker.

Usage:
    python3 tools/slack_fill_notify.py [--since ISO8601] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slack_post import load_slack_config, post_message  # noqa: E402


JST = timezone(timedelta(hours=9))


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ledger_path() -> Path:
    return _repo_root() / "data" / "execution_ledger.db"


def _marker_path() -> Path:
    return _repo_root() / "logs" / ".slack_fill_last_seen"


def _read_marker() -> str:
    path = _marker_path()
    if not path.exists():
        # Default: only post events from the last hour on first run.
        return (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    return path.read_text().strip() or (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()


def _write_marker(value: str) -> None:
    path = _marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _fetch_events(db: Path, since_iso: str) -> list[sqlite3.Row]:
    if not db.exists():
        return []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """
            SELECT event_uid, inserted_at_utc, ts_utc, event_type, lane_id, pair, side,
                   units, price, tp, sl, realized_pl_jpy, exit_reason, trade_id, raw_json
            FROM execution_events
            WHERE inserted_at_utc > ?
              AND event_type IN ('ORDER_FILLED','TRADE_CLOSED','ORDER_CANCELED','PROTECTION_CREATED')
            ORDER BY inserted_at_utc ASC, rowid ASC
            """,
            (since_iso,),
        )
        return list(cur.fetchall())
    finally:
        conn.close()


def _utc_to_jst_short(iso: str) -> str:
    try:
        # Trim sub-microsecond precision OANDA emits.
        text = iso.replace("Z", "+00:00")
        if "." in text:
            head, _, tail = text.partition(".")
            digits = ""
            rest = ""
            for ch in tail:
                if ch.isdigit() and len(digits) < 6:
                    digits += ch
                elif not ch.isdigit():
                    rest = tail[len(digits) :]
                    break
            text = f"{head}.{digits}{rest}" if digits else head + rest
        dt = datetime.fromisoformat(text)
        return dt.astimezone(JST).strftime("%H:%M")
    except (ValueError, TypeError):
        return iso[:16]


def _fold_protection(events: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """Merge PROTECTION_CREATED into the immediately-preceding ORDER_FILLED."""
    folded: list[dict[str, Any]] = []
    for row in events:
        d = dict(row)
        if d["event_type"] == "PROTECTION_CREATED" and folded:
            prev = folded[-1]
            if (
                prev["event_type"] == "ORDER_FILLED"
                and prev["ts_utc"] == d["ts_utc"]
                and not prev.get("tp")
            ):
                prev["tp"] = d.get("price")
                continue
        folded.append(d)
    return folded


def _format_event(event: dict[str, Any]) -> str | None:
    etype = event["event_type"]
    when = _utc_to_jst_short(event.get("ts_utc") or event.get("inserted_at_utc") or "")
    pair = event.get("pair") or "?"
    side = (event.get("side") or "").upper()
    units = event.get("units")
    price = event.get("price")
    tp = event.get("tp")
    pl = event.get("realized_pl_jpy")
    reason = event.get("exit_reason") or ""

    if etype == "ORDER_FILLED":
        icon = "\U0001f7e2" if units and units > 0 else "\U0001f534"
        units_abs = abs(int(units)) if units else 0
        tp_text = f"  TP: {tp}" if tp else "  (no TP yet)"
        return f"{icon} ENTRY {side} {pair} {units_abs}u @{price}  [{when} JST]{tp_text}"

    if etype == "TRADE_CLOSED":
        # OANDA records `side` here as the closing-leg direction (opposite of
        # the position side). Display without side to avoid misleading the
        # operator; the pair + signed P&L is unambiguous, and the operator
        # can correlate to their open positions.
        icon = "⬛"
        units_abs = abs(int(units)) if units else 0
        pl_part = f" ({pl:+,.0f} JPY)" if pl is not None else ""
        reason_part = f"  reason: {reason.lower().replace('_', ' ')}" if reason else ""
        return f"{icon} CLOSE {pair} {units_abs}u @{price}{pl_part}  [{when} JST]{reason_part}"

    if etype == "ORDER_CANCELED":
        reason_part = f"  reason: {reason.lower().replace('_', ' ')}" if reason else ""
        return f"⚪ CANCEL {pair or ''} pending order  [{when} JST]{reason_part}"

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default=None, help="ISO8601 lower bound on inserted_at_utc")
    parser.add_argument("--dry-run", action="store_true", help="Print messages instead of posting")
    parser.add_argument("--limit", type=int, default=20, help="Cap messages posted in one run (default 20)")
    args = parser.parse_args()

    since = args.since or _read_marker()
    events = _fetch_events(_ledger_path(), since)
    if not events:
        if args.dry_run:
            print(f"[dry-run] no new events since {since}")
        return

    folded = _fold_protection(events)
    interesting = [e for e in folded if e["event_type"] != "PROTECTION_CREATED"]
    if args.limit and len(interesting) > args.limit:
        interesting = interesting[-args.limit :]

    slack = load_slack_config()
    token = slack.get("QR_SLACK_BOT_TOKEN")
    channel = slack.get("QR_SLACK_CHANNEL_TRADES") or slack.get("QR_SLACK_CHANNEL_ID")
    if not args.dry_run:
        if not token or not channel:
            print("ERROR: QR_SLACK_BOT_TOKEN and QR_SLACK_CHANNEL_TRADES required", file=sys.stderr)
            sys.exit(2)

    posted = 0
    for ev in interesting:
        text = _format_event(ev)
        if not text:
            continue
        if args.dry_run:
            print(text)
        else:
            try:
                post_message(text, channel, token)
                posted += 1
            except SystemExit:
                # post_message exits on Slack API failure; let it propagate.
                raise

    # Advance the marker even on dry-run? No — only after real posts.
    if not args.dry_run and folded:
        latest = max(e["inserted_at_utc"] for e in folded)
        _write_marker(latest)
        print(f"OK: posted {posted} fills, marker -> {latest}")
    elif args.dry_run:
        print(f"[dry-run] would post {sum(1 for e in interesting if _format_event(e))} events")


if __name__ == "__main__":
    main()

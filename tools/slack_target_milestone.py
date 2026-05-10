#!/usr/bin/env python3
"""Post daily target milestones to Slack `#qr-trades`.

Reads `data/daily_target_state.json` after each `autotrade-cycle` and posts
when `progress_pct` crosses a configured threshold for the campaign day.
Tracked thresholds: 25%, 50%, 75%, 100% on the upside; -3%, -5%, -8% on
the downside (drawdown alerts). Each (campaign_day, threshold) pair is
posted at most once via the marker file `logs/.slack_target_milestones`.

Designed to be called from `scripts/run-autotrade-live.sh` after the
cycle completes. Idempotent.

Usage:
    python3 tools/slack_target_milestone.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slack_post import load_slack_config, post_message  # noqa: E402

UP_MILESTONES = [25.0, 50.0, 75.0, 100.0]
DOWN_MILESTONES = [-3.0, -5.0, -8.0]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _state_path() -> Path:
    return _repo_root() / "data" / "daily_target_state.json"


def _marker_path() -> Path:
    return _repo_root() / "logs" / ".slack_target_milestones"


def _load_marker() -> dict[str, list[float]]:
    path = _marker_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_marker(marker: dict[str, list[float]]) -> None:
    path = _marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker, indent=2))


def _due_milestones(progress_pct: float, already_posted: list[float]) -> list[float]:
    posted_set = set(already_posted)
    due: list[float] = []
    for level in UP_MILESTONES:
        if progress_pct >= level and level not in posted_set:
            due.append(level)
    for level in DOWN_MILESTONES:
        if progress_pct <= level and level not in posted_set:
            due.append(level)
    return due


def _format(level: float, state: dict) -> str:
    pct = state.get("progress_pct", 0.0)
    remaining = state.get("remaining_target_jpy")
    equity = state.get("current_equity_jpy")
    start = state.get("start_balance_jpy")
    realized = None
    if equity is not None and start is not None:
        realized = float(equity) - float(start)
    day = state.get("campaign_day_jst", "?")

    # Lead with the actual progress; mention the crossed threshold as suffix.
    # Reader was misparsing `-3% (-4.45%)` as two metrics; flip the order so
    # the live number is what they see first.
    if level >= 100.0:
        icon = "🎯"
        label = f"DAILY TARGET HIT  {pct:+.2f}%"
        protect = "  → switch to protection-first; cancel trader-owned pending entries"
    elif level > 0:
        icon = "✅" if level >= 75 else "📈"
        label = f"Target progress {pct:+.2f}%  (crossed {level:.0f}% level)"
        protect = ""
    elif level <= -8.0:
        icon = "🚨"
        label = f"DRAWDOWN ALERT {pct:+.2f}%  (crossed {level:.0f}% level)"
        protect = "  → consider stopping fresh entries until structure recovers"
    elif level <= -5.0:
        icon = "⚠️"
        label = f"Drawdown {pct:+.2f}%  (crossed {level:.0f}% level)"
        protect = ""
    else:
        icon = "🟡"
        label = f"Drawdown {pct:+.2f}%  (crossed {level:.0f}% level)"
        protect = ""

    realized_part = f"\n  Realized: {realized:+,.0f} JPY" if realized is not None else ""
    remaining_part = f"\n  Remaining to target: {float(remaining):+,.0f} JPY" if remaining is not None else ""
    equity_part = f"\n  Equity: {float(equity):,.0f} JPY (start {float(start):,.0f})" if equity is not None and start is not None else ""

    return f"{icon} *{label}*  [{day}]{realized_part}{remaining_part}{equity_part}{protect}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print messages instead of posting")
    args = parser.parse_args()

    state_path = _state_path()
    if not state_path.exists():
        if args.dry_run:
            print(f"[dry-run] no state file at {state_path}")
        return
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {state_path}: {exc}", file=sys.stderr)
        sys.exit(2)

    progress_pct = state.get("progress_pct")
    day = state.get("campaign_day_jst")
    if progress_pct is None or not day:
        if args.dry_run:
            print("[dry-run] state missing progress_pct or campaign_day_jst")
        return

    marker = _load_marker()
    already = marker.get(day, [])
    due = _due_milestones(float(progress_pct), already)
    if not due:
        if args.dry_run:
            print(f"[dry-run] no milestones due (progress_pct={progress_pct:+.2f}%, posted={already})")
        return

    slack = load_slack_config()
    token = slack.get("QR_SLACK_BOT_TOKEN")
    channel = slack.get("QR_SLACK_CHANNEL_TRADES") or slack.get("QR_SLACK_CHANNEL_ID")
    if not args.dry_run:
        if not token or not channel:
            print("ERROR: QR_SLACK_BOT_TOKEN and QR_SLACK_CHANNEL_TRADES required", file=sys.stderr)
            sys.exit(2)

    posted: list[float] = []
    for level in due:
        text = _format(level, state)
        if args.dry_run:
            print(text)
            posted.append(level)
        else:
            post_message(text, channel, token)
            posted.append(level)

    if not args.dry_run and posted:
        marker[day] = sorted(set(already + posted))
        _save_marker(marker)
        print(f"OK: posted {len(posted)} milestone(s) for {day}: {posted}")
    elif args.dry_run:
        print(f"[dry-run] would post {len(posted)} milestone(s)")


if __name__ == "__main__":
    main()

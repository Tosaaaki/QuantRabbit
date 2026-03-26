#!/usr/bin/env python3
"""Prune stale alerts from shared_state.json.

Removes alerts older than max_age_minutes (default: 60).
Keeps DIRECTIVE and FIX alerts regardless of age.

Usage:
    python tools/prune_alerts.py [--max-age 60] [--dry-run]
"""

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SHARED_STATE_PATH = BASE_DIR / "logs" / "shared_state.json"

# Patterns that extract timestamps from alert strings
TS_PATTERNS = [
    re.compile(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?)\]"),
    re.compile(r"\[(\d{2}:\d{2}Z)\]"),
]

# Alert types that should never be pruned automatically
KEEP_PREFIXES = ("DIRECTIVE", "FIX", "INVESTIGATE")


def extract_alert_time(alert: str, today: datetime) -> datetime | None:
    """Try to extract a UTC datetime from an alert string."""
    for pat in TS_PATTERNS:
        m = pat.search(alert)
        if m:
            ts_str = m.group(1)
            try:
                if len(ts_str) <= 6:  # "HH:MMZ" format
                    h, rest = ts_str.split(":")
                    minute = rest.rstrip("Z")
                    return today.replace(hour=int(h), minute=int(minute), second=0, microsecond=0)
                return datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                continue
    return None


def should_keep(alert: str, now: datetime, max_age: timedelta, today: datetime) -> bool:
    """Determine if an alert should be kept."""
    # Always keep certain types
    stripped = alert.lstrip()
    for prefix in KEEP_PREFIXES:
        if stripped.startswith(prefix) or stripped.startswith(f"\u2605{prefix}"):
            return True

    alert_time = extract_alert_time(alert, today)
    if alert_time is None:
        # No timestamp — keep it (can't determine age)
        return True

    if alert_time.tzinfo is None:
        alert_time = alert_time.replace(tzinfo=timezone.utc)

    return (now - alert_time) < max_age


def prune(max_age_minutes: int = 60, dry_run: bool = False) -> dict:
    """Prune old alerts from shared_state.json."""
    try:
        state = json.loads(SHARED_STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"error": str(e)}

    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    max_age = timedelta(minutes=max_age_minutes)

    # Prune both alert fields
    result = {}
    for field in ("alerts", "secretary_alerts"):
        old_alerts = state.get(field, [])
        new_alerts = [a for a in old_alerts if should_keep(a, now, max_age, today)]
        removed = len(old_alerts) - len(new_alerts)
        result[field] = {"before": len(old_alerts), "after": len(new_alerts), "removed": removed}
        if not dry_run:
            state[field] = new_alerts

    if not dry_run and any(r["removed"] > 0 for r in result.values()):
        SHARED_STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    result["dry_run"] = dry_run
    return result


def main():
    max_age = 60
    dry_run = False
    args = sys.argv[1:]
    if "--max-age" in args:
        idx = args.index("--max-age")
        if idx + 1 < len(args):
            max_age = int(args[idx + 1])
    if "--dry-run" in args:
        dry_run = True

    result = prune(max_age_minutes=max_age, dry_run=dry_run)

    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)

    mode = "DRY RUN" if dry_run else "PRUNED"
    for field, stats in result.items():
        if field == "dry_run":
            continue
        print(f"  {field}: {stats['before']} → {stats['after']} ({stats['removed']} removed)")
    print(f"  Mode: {mode} | Max age: {max_age}min")


if __name__ == "__main__":
    main()

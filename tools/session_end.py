#!/usr/bin/env python3
"""
Session end gate — enforces minimum session duration.
The model MUST use this script to end a session. Direct lock cleanup is prohibited.

Usage: python3 tools/session_end.py
Exit codes:
  0 = SESSION_END completed (lock released, ingest done)
  1 = TOO_EARLY (keep trading)
  2 = ERROR
"""
import os, sys, time, subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "logs" / ".trader_lock"
START_FILE = ROOT / "logs" / ".trader_start"

MIN_ELAPSED = 480   # 8 min minimum — below this, SESSION_END is blocked
SESSION_END_AT = 540  # 9 min — normal SESSION_END threshold

def main():
    now = int(time.time())

    # Read start time
    if not START_FILE.exists():
        print("ERROR: no .trader_start file — session not properly initialized")
        sys.exit(2)

    try:
        start = int(START_FILE.read_text().strip())
    except ValueError:
        print("ERROR: .trader_start contains invalid data")
        sys.exit(2)

    elapsed = now - start
    start_utc = datetime.fromtimestamp(start, tz=timezone.utc).strftime("%H:%M")
    now_utc = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%H:%M")

    # Time gate
    if elapsed < MIN_ELAPSED:
        remaining = MIN_ELAPSED - elapsed
        print(f"TOO_EARLY elapsed={elapsed}s ({start_utc}→{now_utc} UTC)")
        print(f"Minimum {MIN_ELAPSED}s required. {remaining}s remaining.")
        print("Go deeper: fib_wave --all, Different lens, Tier 2 M5 chart reading, LIMIT placement.")
        print("Run mid_session_check.py and keep trading.")
        sys.exit(1)

    # Check state.md freshness
    state_md = ROOT / "collab_trade" / "state.md"
    if state_md.exists():
        state_age = now - int(state_md.stat().st_mtime)
        if state_age > 3600:
            print(f"⚠️ STATE.MD STALE ({state_age}s old) — UPDATE IT before session_end")
            sys.exit(1)

    # === SESSION_END ===
    print(f"SESSION_END elapsed={elapsed}s ({start_utc}→{now_utc} UTC)")

    # trade_performance
    try:
        result = subprocess.run(
            ["python3", str(ROOT / "tools" / "trade_performance.py"), "--days", "1"],
            capture_output=True, text=True, timeout=15, cwd=str(ROOT)
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n")[:25]:
                print(line)
    except Exception as e:
        print(f"trade_performance warning: {e}")

    # ingest (with timeout)
    try:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        subprocess.run(
            ["python3", str(ROOT / "collab_trade" / "memory" / "ingest.py"), date_str, "--force"],
            capture_output=True, text=True, timeout=30, cwd=str(ROOT)
        )
    except Exception as e:
        print(f"ingest warning: {e}")

    # Release lock
    try:
        LOCK.unlink(missing_ok=True)
        START_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    print("LOCK_RELEASED")
    print(f"--- Use these exact times in your summary: {start_utc}→{now_utc} UTC ({elapsed}s) ---")

if __name__ == "__main__":
    main()

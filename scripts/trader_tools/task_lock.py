#!/usr/bin/env python3
"""Task lock manager using individual lock files (not shared_state.json).

Usage:
  python3 scripts/trader_tools/task_lock.py acquire <task_name> <timeout_minutes>
  python3 scripts/trader_tools/task_lock.py release <task_name>
  python3 scripts/trader_tools/task_lock.py status

acquire: Exit 0 if lock acquired, exit 1 if another instance is running (skip).
release: Always exit 0.
status:  Print all locks.

Lock files: logs/locks/<task_name>.lock (JSON with pid + started_at)
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LOCK_DIR = Path(__file__).resolve().parents[2] / "logs" / "locks"


def _lock_file(task_name: str) -> Path:
    return LOCK_DIR / f"{task_name}.lock"


def acquire(task_name: str, timeout_min: int) -> bool:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lf = _lock_file(task_name)

    if lf.exists():
        try:
            lock = json.loads(lf.read_text())
            if lock.get("running"):
                started = lock.get("started_at", "")
                if started:
                    started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                    elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
                    if elapsed < timeout_min * 60:
                        print(
                            f"SKIP: {task_name} is already running (started {int(elapsed)}s ago, pid={lock.get('pid','')})"
                        )
                        return False
                    else:
                        print(
                            f"STALE: {task_name} lock expired ({int(elapsed)}s > {timeout_min*60}s), taking over"
                        )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass  # corrupt lock file, take over

    lock_data = {
        "running": True,
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pid": os.getpid(),
    }
    lf.write_text(json.dumps(lock_data))
    print(f"ACQUIRED: {task_name}")
    return True


def release(task_name: str) -> None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lf = _lock_file(task_name)
    lock_data = {
        "running": False,
        "released_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    lf.write_text(json.dumps(lock_data))
    print(f"RELEASED: {task_name}")


def status() -> None:
    if not LOCK_DIR.exists():
        print("No locks directory")
        return
    found = False
    for lf in sorted(LOCK_DIR.glob("*.lock")):
        found = True
        try:
            lock = json.loads(lf.read_text())
            running = lock.get("running", False)
            started = lock.get("started_at", lock.get("released_at", "?"))
            pid = lock.get("pid", "")
            print(
                f"  {lf.stem}: {'RUNNING' if running else 'idle'} (at: {started}, pid: {pid})"
            )
        except (json.JSONDecodeError, ValueError):
            print(f"  {lf.stem}: CORRUPT")
    if not found:
        print("No task locks found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "acquire":
        task = sys.argv[2]
        timeout = int(sys.argv[3])
        ok = acquire(task, timeout)
        sys.exit(0 if ok else 1)
    elif cmd == "release":
        release(sys.argv[2])
    elif cmd == "status":
        status()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

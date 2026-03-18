#!/usr/bin/env python3
"""Task lock manager using atomic mkdir (race-condition free).

Usage:
  python3 scripts/trader_tools/task_lock.py acquire <task_name> <timeout_minutes>
  python3 scripts/trader_tools/task_lock.py release <task_name>
  python3 scripts/trader_tools/task_lock.py status

acquire: Exit 0 + prints ACQUIRED if lock taken. Exit 1 + prints SKIP if busy.
release: Always exit 0.
status:  Print all locks.

Uses mkdir atomicity: os.mkdir() is atomic on POSIX — only one process succeeds.
Lock dir: logs/locks/<task_name>.d/ (directory as lock token)
Metadata: logs/locks/<task_name>.d/meta.json
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LOCK_DIR = Path(__file__).resolve().parents[2] / "logs" / "locks"


def _lock_dir(task_name: str) -> Path:
    return LOCK_DIR / f"{task_name}.d"


def acquire(task_name: str, timeout_min: int) -> bool:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    ld = _lock_dir(task_name)

    try:
        os.mkdir(ld)  # atomic — raises FileExistsError if already exists
    except FileExistsError:
        # Lock exists — check if stale
        meta_file = ld / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                started = meta.get("started_at", "")
                if started:
                    started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                    elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
                    if elapsed > timeout_min * 60:
                        print(f"STALE: {task_name} lock expired ({int(elapsed)}s > {timeout_min*60}s), taking over")
                        # Force release and retry
                        _force_remove(ld)
                        return acquire(task_name, timeout_min)
                    print(f"SKIP: {task_name} is already running (started {int(elapsed)}s ago, pid={meta.get('pid', '')})")
                    return False
            except (json.JSONDecodeError, ValueError, KeyError):
                # Corrupt metadata — force release
                print(f"CORRUPT: {task_name} lock metadata corrupt, taking over")
                _force_remove(ld)
                return acquire(task_name, timeout_min)
        else:
            # Dir exists but no metadata — stale, take over
            print(f"STALE: {task_name} lock dir exists but no metadata, taking over")
            _force_remove(ld)
            return acquire(task_name, timeout_min)

    # mkdir succeeded — we own the lock. Write metadata.
    meta = {
        "running": True,
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pid": os.getpid(),
    }
    (ld / "meta.json").write_text(json.dumps(meta))
    print(f"ACQUIRED: {task_name}")
    return True


def release(task_name: str) -> None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    ld = _lock_dir(task_name)
    _force_remove(ld)
    print(f"RELEASED: {task_name}")


def _force_remove(ld: Path) -> None:
    """Remove lock directory and its contents."""
    if ld.exists():
        for f in ld.iterdir():
            f.unlink(missing_ok=True)
        try:
            ld.rmdir()
        except OSError:
            pass


def status() -> None:
    if not LOCK_DIR.exists():
        print("No locks directory")
        return
    found = False
    for item in sorted(LOCK_DIR.iterdir()):
        if item.is_dir() and item.name.endswith(".d"):
            found = True
            task_name = item.name[:-2]
            meta_file = item / "meta.json"
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    running = meta.get("running", False)
                    started = meta.get("started_at", "?")
                    pid = meta.get("pid", "")
                    elapsed = ""
                    if started != "?":
                        try:
                            started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                            elapsed = f", {int((datetime.now(timezone.utc) - started_dt).total_seconds())}s ago"
                        except ValueError:
                            pass
                    print(f"  {task_name}: RUNNING (pid: {pid}{elapsed})")
                except (json.JSONDecodeError, ValueError):
                    print(f"  {task_name}: CORRUPT")
            else:
                print(f"  {task_name}: STALE (no metadata)")
    # Also check for old-style .lock files
    for lf in sorted(LOCK_DIR.glob("*.lock")):
        found = True
        try:
            lock = json.loads(lf.read_text())
            print(f"  {lf.stem}: OLD_FORMAT ({'RUNNING' if lock.get('running') else 'idle'})")
        except (json.JSONDecodeError, ValueError):
            print(f"  {lf.stem}: OLD_FORMAT (corrupt)")
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

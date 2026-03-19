#!/usr/bin/env python3
"""Task lock manager using atomic mkdir (race-condition free).

Usage:
  python3 scripts/trader_tools/task_lock.py acquire <task_name> <timeout_minutes> [--pid <PID>]
  python3 scripts/trader_tools/task_lock.py release <task_name>
  python3 scripts/trader_tools/task_lock.py status

acquire: Exit 0 + prints ACQUIRED if lock taken. Exit 1 + prints SKIP if busy.
         --pid: PID of the long-lived parent process (e.g. $PPID from shell).
         Used for stale detection — if the PID is dead, lock is stale.
release: Always exit 0.
status:  Print all locks with PID liveness.

Stale detection priority:
  1. PID provided & dead → stale (take over)
  2. PID provided & alive → SKIP (no matter how long)
  3. No PID → fall back to timeout
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LOCK_DIR = Path(__file__).resolve().parents[2] / "logs" / "locks"


def _lock_dir(task_name: str) -> Path:
    return LOCK_DIR / f"{task_name}.d"


def _is_pid_alive(pid: int) -> bool:
    """Check if a process is still running (macOS/Linux)."""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # signal 0: just check existence
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user


def acquire(task_name: str, timeout_min: int, owner_pid: int = 0) -> bool:
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
                pid = meta.get("pid", 0)
                started = meta.get("started_at", "")

                elapsed = 0
                if started:
                    try:
                        started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                        elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
                    except ValueError:
                        pass

                # Primary: PID-based check
                if pid and _is_pid_alive(pid):
                    print(f"SKIP: {task_name} is running (pid={pid} alive, {int(elapsed)}s elapsed)")
                    return False

                if pid and not _is_pid_alive(pid):
                    print(f"STALE: {task_name} pid={pid} is dead ({int(elapsed)}s elapsed), taking over")
                    _force_remove(ld)
                    return acquire(task_name, timeout_min, owner_pid)

                # Fallback: no PID → use timeout
                if elapsed > timeout_min * 60:
                    print(f"STALE: {task_name} no pid, expired ({int(elapsed)}s > {timeout_min*60}s), taking over")
                    _force_remove(ld)
                    return acquire(task_name, timeout_min, owner_pid)

                print(f"SKIP: {task_name} is running (no pid, {int(elapsed)}s elapsed)")
                return False

            except (json.JSONDecodeError, ValueError, KeyError):
                print(f"CORRUPT: {task_name} lock metadata corrupt, taking over")
                _force_remove(ld)
                return acquire(task_name, timeout_min, owner_pid)
        else:
            print(f"STALE: {task_name} lock dir exists but no metadata, taking over")
            _force_remove(ld)
            return acquire(task_name, timeout_min, owner_pid)

    # mkdir succeeded — we own the lock. Write metadata.
    # Use owner_pid (long-lived parent) if provided, else fall back to our own PID
    meta = {
        "running": True,
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pid": owner_pid if owner_pid else os.getpid(),
    }
    (ld / "meta.json").write_text(json.dumps(meta))
    print(f"ACQUIRED: {task_name} (tracking pid={meta['pid']})")
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
                    started = meta.get("started_at", "?")
                    pid = meta.get("pid", "")
                    elapsed = ""
                    if started != "?":
                        try:
                            started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                            elapsed = f", {int((datetime.now(timezone.utc) - started_dt).total_seconds())}s ago"
                        except ValueError:
                            pass
                    alive = _is_pid_alive(pid) if pid else False
                    state = "RUNNING" if alive else "STALE(pid dead)"
                    print(f"  {task_name}: {state} (pid={pid}{elapsed})")
                except (json.JSONDecodeError, ValueError):
                    print(f"  {task_name}: CORRUPT")
            else:
                print(f"  {task_name}: STALE (no metadata)")
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
        # Parse optional --pid
        pid = 0
        if "--pid" in sys.argv:
            idx = sys.argv.index("--pid")
            if idx + 1 < len(sys.argv):
                pid = int(sys.argv[idx + 1])
        ok = acquire(task, timeout, pid)
        sys.exit(0 if ok else 1)
    elif cmd == "release":
        release(sys.argv[2])
    elif cmd == "status":
        status()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

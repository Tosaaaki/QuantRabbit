#!/usr/bin/env python3
"""Task lock manager using atomic mkdir (race-condition free).

Usage:
  python3 scripts/trader_tools/task_lock.py acquire <lock_name> <timeout_minutes> [--pid <PID>] [--caller <caller_name>]
  python3 scripts/trader_tools/task_lock.py release <lock_name> [--caller <caller_name>]
  python3 scripts/trader_tools/task_lock.py status

acquire: Exit 0 + prints ACQUIRED if lock taken. Exit 1 + prints SKIP if busy.
         --pid: PID of the long-lived parent process (e.g. $PPID from shell).
         --caller: Name of the calling task (for rotation fairness).
release: Always exit 0.
status:  Print all locks with PID liveness.

Rotation fairness (--caller):
  When using a shared lock (e.g. global_agent), the same caller cannot
  acquire twice in a row. After A finishes, A must yield to B/C/D.
  If no other caller arrives within the grace period, A can acquire again.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LOCK_DIR = Path(__file__).resolve().parents[2] / "logs" / "locks"
ROTATION_FILE = LOCK_DIR / "last_runner.json"
ROTATION_GRACE_SEC = 10  # after this many seconds, same caller can go again


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


def _check_rotation(lock_name: str, caller: str) -> bool:
    """Return True if caller is allowed to run (rotation check passed)."""
    if not caller:
        return True
    try:
        if ROTATION_FILE.exists():
            data = json.loads(ROTATION_FILE.read_text())
            entry = data.get(lock_name, {})
            last_caller = entry.get("caller", "")
            finished_at = entry.get("finished_at", "")

            if last_caller == caller and finished_at:
                try:
                    fin_dt = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
                    elapsed = (datetime.now(timezone.utc) - fin_dt).total_seconds()
                    if elapsed < ROTATION_GRACE_SEC:
                        return False  # too soon, yield to others
                except ValueError:
                    pass
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return True


def _record_rotation(lock_name: str, caller: str) -> None:
    """Record that caller finished using the lock."""
    if not caller:
        return
    try:
        data = {}
        if ROTATION_FILE.exists():
            data = json.loads(ROTATION_FILE.read_text())
        data[lock_name] = {
            "caller": caller,
            "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        ROTATION_FILE.write_text(json.dumps(data))
    except (json.JSONDecodeError, ValueError):
        # Corrupt file, overwrite
        data = {lock_name: {
            "caller": caller,
            "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }}
        ROTATION_FILE.write_text(json.dumps(data))


def acquire(task_name: str, timeout_min: int, owner_pid: int = 0, caller: str = "") -> bool:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    ld = _lock_dir(task_name)

    # Rotation check — same caller must yield for ROTATION_GRACE_SEC
    if not _check_rotation(task_name, caller):
        print(f"YIELD: {task_name} — {caller} just ran, yielding to others ({ROTATION_GRACE_SEC}s grace)")
        return False

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

                # Timeout always wins — scheduled task parent PIDs
                # can outlive the actual task execution
                timeout_sec = timeout_min * 60
                if elapsed > timeout_sec:
                    reason = f"pid={'alive' if pid and _is_pid_alive(pid) else 'dead/none'}"
                    print(f"STALE: {task_name} expired ({int(elapsed)}s > {timeout_sec}s, {reason}), taking over")
                    _force_remove(ld)
                    return acquire(task_name, timeout_min, owner_pid, caller)

                # Within timeout: PID-based check
                if pid and not _is_pid_alive(pid):
                    print(f"STALE: {task_name} pid={pid} is dead ({int(elapsed)}s elapsed), taking over")
                    _force_remove(ld)
                    return acquire(task_name, timeout_min, owner_pid, caller)

                print(f"SKIP: {task_name} is running (pid={pid}, {int(elapsed)}s elapsed)")
                return False

            except (json.JSONDecodeError, ValueError, KeyError):
                print(f"CORRUPT: {task_name} lock metadata corrupt, taking over")
                _force_remove(ld)
                return acquire(task_name, timeout_min, owner_pid, caller)
        else:
            print(f"STALE: {task_name} lock dir exists but no metadata, taking over")
            _force_remove(ld)
            return acquire(task_name, timeout_min, owner_pid, caller)

    # mkdir succeeded — we own the lock. Write metadata.
    meta = {
        "running": True,
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pid": owner_pid if owner_pid else os.getpid(),
        "caller": caller,
    }
    (ld / "meta.json").write_text(json.dumps(meta))
    print(f"ACQUIRED: {task_name} (caller={caller or '?'}, pid={meta['pid']})")
    return True


def release(task_name: str, caller: str = "") -> None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    ld = _lock_dir(task_name)

    # Read caller from meta if not provided
    if not caller:
        meta_file = ld / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                caller = meta.get("caller", "")
            except (json.JSONDecodeError, ValueError):
                pass

    _force_remove(ld)
    _record_rotation(task_name, caller)
    print(f"RELEASED: {task_name} (caller={caller or '?'})")


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
                    caller = meta.get("caller", "?")
                    elapsed = ""
                    if started != "?":
                        try:
                            started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                            elapsed = f", {int((datetime.now(timezone.utc) - started_dt).total_seconds())}s ago"
                        except ValueError:
                            pass
                    alive = _is_pid_alive(pid) if pid else False
                    state = "RUNNING" if alive else "STALE(pid dead)"
                    print(f"  {task_name}: {state} (caller={caller}, pid={pid}{elapsed})")
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

    # Show rotation info
    if ROTATION_FILE.exists():
        try:
            data = json.loads(ROTATION_FILE.read_text())
            for lock_name, entry in data.items():
                caller = entry.get("caller", "?")
                fin = entry.get("finished_at", "?")
                print(f"  [rotation] {lock_name}: last={caller} at {fin}")
        except (json.JSONDecodeError, ValueError):
            pass

    if not found and not ROTATION_FILE.exists():
        print("No task locks found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    # Parse optional flags
    pid = 0
    caller = ""
    if "--pid" in sys.argv:
        idx = sys.argv.index("--pid")
        if idx + 1 < len(sys.argv):
            pid = int(sys.argv[idx + 1])
    if "--caller" in sys.argv:
        idx = sys.argv.index("--caller")
        if idx + 1 < len(sys.argv):
            caller = sys.argv[idx + 1]

    if cmd == "acquire":
        task = sys.argv[2]
        timeout = int(sys.argv[3])
        ok = acquire(task, timeout, pid, caller)
        sys.exit(0 if ok else 1)
    elif cmd == "release":
        release(sys.argv[2], caller)
    elif cmd == "status":
        status()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

#!/usr/bin/env python3
"""Host-neutral task runtime helpers shared by Claude/Codex playbooks."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9 fallback
    ZoneInfo = None


ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
TRADER_LOCK = LOGS / ".trader_lock"
TRADER_START = LOGS / ".trader_start"
TRADER_WATCHDOG = LOGS / ".trader_watchdog"
TASK_LOCK = ROOT / "tools" / "task_lock.py"
JST = ZoneInfo("Asia/Tokyo") if ZoneInfo is not None else None


def _now_jst() -> datetime:
    if JST is None:
        return datetime.now().astimezone()
    return datetime.now(JST)


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_trader_lock() -> tuple[int, int]:
    try:
        parts = TRADER_LOCK.read_text().strip().split()
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except (FileNotFoundError, ValueError):
        pass
    return 0, 0


def _write_trader_lock(ts: int, owner_pid: int) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    TRADER_LOCK.write_text(f"{ts} {owner_pid}\n")


def _read_trader_start() -> int:
    try:
        return int(TRADER_START.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _read_watchdog() -> tuple[int, int]:
    try:
        parts = TRADER_WATCHDOG.read_text().strip().split()
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except (FileNotFoundError, ValueError):
        pass
    return 0, 0


def _write_watchdog(pid: int, session_start: int) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    TRADER_WATCHDOG.write_text(f"{pid} {session_start}\n")


def _clear_watchdog_file(expected_session_start: int | None = None) -> None:
    if expected_session_start is not None:
        _, session_start = _read_watchdog()
        if session_start and session_start != expected_session_start:
            return
    TRADER_WATCHDOG.unlink(missing_ok=True)


def _stop_watchdog(expected_session_start: int | None = None) -> None:
    pid, session_start = _read_watchdog()
    if not pid:
        TRADER_WATCHDOG.unlink(missing_ok=True)
        return
    if expected_session_start is not None and session_start and session_start != expected_session_start:
        return
    if pid != os.getpid() and _is_pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    _clear_watchdog_file(expected_session_start if expected_session_start is not None else session_start or None)


def _run_ingest_for_previous_session(lock_ts: int | None = None) -> None:
    session_ts = lock_ts
    try:
        session_ts = _read_trader_start() or session_ts
    except Exception:
        pass
    if not session_ts:
        session_ts = int(time.time())
    date_str = datetime.fromtimestamp(session_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "record_s_hunt_ledger.py")],
            cwd=str(ROOT),
            timeout=10,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    print(f"STALE_CLEANUP: running ingest for previous session date={date_str}")
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "collab_trade" / "memory" / "ingest.py"), date_str],
            cwd=str(ROOT),
            timeout=30,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("STALE_INGEST_DONE")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"STALE_INGEST_WARN: {exc}")


def trader_preflight(timeout_sec: int) -> int:
    now_jst = _now_jst()
    dow = now_jst.isoweekday()
    hour = now_jst.hour
    if (dow == 6 and hour >= 6) or dow == 7 or (dow == 1 and hour < 7):
        print(f"WEEKEND_HALT dow={dow} hour={hour}")
        return 1

    if not TRADER_LOCK.exists():
        _stop_watchdog()
        print("NO_LOCK — 新規セッション開始")
        return 0

    lock_time, old_pid = _read_trader_lock()
    now = int(time.time())
    age = max(0, now - lock_time) if lock_time else timeout_sec + 1
    if age < timeout_sec and _is_pid_alive(old_pid):
        print(f"ALREADY_RUNNING age={age}s pid={old_pid}")
        return 1

    print(f"STALE_LOCK age={age}s — 引き継ぎ開始")
    _stop_watchdog(_read_trader_start() or None)
    if old_pid and _is_pid_alive(old_pid):
        try:
            os.kill(old_pid, signal.SIGTERM)
            print(f"KILLED_STALE pid={old_pid}")
        except OSError:
            pass
    _run_ingest_for_previous_session(lock_time or None)
    return 0


def trader_watchdog(owner_pid: int, watchdog_sec: int, session_start: int) -> int:
    time.sleep(watchdog_sec)
    current_start = _read_trader_start()
    if current_start != session_start or not TRADER_LOCK.exists():
        _clear_watchdog_file(session_start)
        return 0
    lock_time, lock_pid = _read_trader_lock()
    if lock_pid != owner_pid:
        _clear_watchdog_file(session_start)
        return 0
    try:
        if _is_pid_alive(owner_pid):
            os.kill(owner_pid, signal.SIGTERM)
    except OSError:
        pass
    TRADER_LOCK.unlink(missing_ok=True)
    TRADER_START.unlink(missing_ok=True)
    _clear_watchdog_file(session_start)
    return 0


def trader_start(owner_pid: int, watchdog_sec: int) -> int:
    _stop_watchdog()
    now = int(time.time())
    _write_trader_lock(now, owner_pid)
    TRADER_START.write_text(f"{now}\n")
    watchdog = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "trader",
            "watchdog",
            "--owner-pid",
            str(owner_pid),
            "--watchdog-sec",
            str(watchdog_sec),
            "--session-start",
            str(now),
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _write_watchdog(watchdog.pid, now)
    print(f"LOCK_ACQUIRED pid={owner_pid} start={now} watchdog_pid={watchdog.pid}")
    return 0


def trader_cycle(owner_pid: int) -> int:
    now = int(time.time())
    _write_trader_lock(now, owner_pid)
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "session_end.py")], cwd=str(ROOT))
    if result.returncode == 0:
        return 0
    if result.returncode != 1:
        return result.returncode
    mid = subprocess.run([sys.executable, str(ROOT / "tools" / "mid_session_check.py")], cwd=str(ROOT))
    return mid.returncode


def trader_cleanup_watchdog(session_start: int | None = None) -> int:
    _stop_watchdog(session_start)
    return 0


def quality_audit_preflight(owner_pid: int, timeout_min: int) -> int:
    now_jst = _now_jst()
    dow = now_jst.isoweekday()
    hour = now_jst.hour
    if (dow == 6 and hour >= 6) or dow == 7 or (dow == 1 and hour < 7):
        print(f"SKIP: quality_audit market closed dow={dow} hour={hour}")
        return 1

    result = subprocess.run(
        [
            sys.executable,
            str(TASK_LOCK),
            "acquire",
            "quality_audit",
            str(timeout_min),
            "--pid",
            str(owner_pid),
            "--caller",
            "quality-audit",
        ],
        cwd=str(ROOT),
    )
    return result.returncode


def quality_audit_release() -> int:
    result = subprocess.run(
        [
            sys.executable,
            str(TASK_LOCK),
            "release",
            "quality_audit",
            "--caller",
            "quality-audit",
        ],
        cwd=str(ROOT),
    )
    return result.returncode


def inventory_director_preflight(owner_pid: int, timeout_min: int) -> int:
    now_jst = _now_jst()
    dow = now_jst.isoweekday()
    hour = now_jst.hour
    if (dow == 6 and hour >= 6) or dow == 7 or (dow == 1 and hour < 7):
        print(f"SKIP: inventory_director market closed dow={dow} hour={hour}")
        return 1

    lock_time, trader_pid = _read_trader_lock()
    if TRADER_LOCK.exists() and trader_pid and _is_pid_alive(trader_pid):
        age = max(0, int(time.time()) - lock_time) if lock_time else 0
        print(f"SKIP: trader active pid={trader_pid} age={age}s")
        return 1

    result = subprocess.run(
        [
            sys.executable,
            str(TASK_LOCK),
            "acquire",
            "inventory_director",
            str(timeout_min),
            "--pid",
            str(owner_pid),
            "--caller",
            "inventory-director",
        ],
        cwd=str(ROOT),
    )
    return result.returncode


def inventory_director_release() -> int:
    result = subprocess.run(
        [
            sys.executable,
            str(TASK_LOCK),
            "release",
            "inventory_director",
            "--caller",
            "inventory-director",
        ],
        cwd=str(ROOT),
    )
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)

    trader = subparsers.add_parser("trader")
    trader_sub = trader.add_subparsers(dest="action", required=True)

    trader_pre = trader_sub.add_parser("preflight")
    trader_pre.add_argument("--timeout-sec", type=int, default=900)

    trader_start_parser = trader_sub.add_parser("start")
    trader_start_parser.add_argument("--owner-pid", type=int, default=os.getppid())
    trader_start_parser.add_argument("--watchdog-sec", type=int, default=1020)

    trader_cycle_parser = trader_sub.add_parser("cycle")
    trader_cycle_parser.add_argument("--owner-pid", type=int, default=os.getppid())

    trader_watchdog_parser = trader_sub.add_parser("watchdog")
    trader_watchdog_parser.add_argument("--owner-pid", type=int, required=True)
    trader_watchdog_parser.add_argument("--watchdog-sec", type=int, default=1020)
    trader_watchdog_parser.add_argument("--session-start", type=int, required=True)

    trader_cleanup_parser = trader_sub.add_parser("cleanup-watchdog")
    trader_cleanup_parser.add_argument("--session-start", type=int)

    audit = subparsers.add_parser("quality-audit")
    audit_sub = audit.add_subparsers(dest="action", required=True)

    audit_pre = audit_sub.add_parser("preflight")
    audit_pre.add_argument("--owner-pid", type=int, default=os.getppid())
    audit_pre.add_argument("--timeout-min", type=int, default=25)

    audit_sub.add_parser("release")

    inventory = subparsers.add_parser("inventory-director")
    inventory_sub = inventory.add_subparsers(dest="action", required=True)

    inventory_pre = inventory_sub.add_parser("preflight")
    inventory_pre.add_argument("--owner-pid", type=int, default=os.getppid())
    inventory_pre.add_argument("--timeout-min", type=int, default=15)

    inventory_sub.add_parser("release")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.task == "trader":
        if args.action == "preflight":
            return trader_preflight(args.timeout_sec)
        if args.action == "start":
            return trader_start(args.owner_pid, args.watchdog_sec)
        if args.action == "cycle":
            return trader_cycle(args.owner_pid)
        if args.action == "watchdog":
            return trader_watchdog(args.owner_pid, args.watchdog_sec, args.session_start)
        if args.action == "cleanup-watchdog":
            return trader_cleanup_watchdog(args.session_start)
    if args.task == "quality-audit":
        if args.action == "preflight":
            return quality_audit_preflight(args.owner_pid, args.timeout_min)
        if args.action == "release":
            return quality_audit_release()
    if args.task == "inventory-director":
        if args.action == "preflight":
            return inventory_director_preflight(args.owner_pid, args.timeout_min)
        if args.action == "release":
            return inventory_director_release()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

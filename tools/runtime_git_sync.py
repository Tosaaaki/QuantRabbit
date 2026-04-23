#!/usr/bin/env python3
"""Safely auto-sync tracked runtime files back to main.

This helper exists to keep the live QuantRabbit runtime from leaving `main`
dirty just because tracked handoff files changed. It is intentionally strict:

- Only approved runtime paths are ever staged.
- Trader-session sync requires a clean baseline snapshot from session start.
- Any unrelated dirtiness causes a skip, not a partial commit.
- Non-main branches are never auto-committed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
TRADER_BASELINE = LOGS / ".trader_git_baseline.json"

TRADER_ALLOWED_PATTERNS = (
    "collab_trade/state.md",
    "collab_trade/strategy_memory.md",
    "collab_trade/memory/lesson_registry.json",
    "collab_trade/daily/*/state.md",
    "collab_trade/daily/*/trades.md",
)

DAILY_REVIEW_ALLOWED_PATTERNS = (
    "collab_trade/strategy_memory.md",
    "collab_trade/memory/lesson_registry.json",
)


def _run_git(args: list[str], *, check: bool = True, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    return subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=check,
    )


def _branch_name() -> str:
    result = _run_git(["branch", "--show-current"], timeout=5)
    return result.stdout.strip()


def _git_status() -> dict[str, str]:
    result = _run_git(["status", "--porcelain=v1", "--untracked-files=all"], timeout=10)
    rows: dict[str, str] = {}
    for raw_line in result.stdout.splitlines():
        if not raw_line:
            continue
        status = raw_line[:2]
        path = raw_line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        rows[path] = status
    return rows


def _is_allowed(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch(path, pattern) for pattern in patterns)


def _write_baseline(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def _load_baseline(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def snapshot_trader_baseline(session_start: int) -> int:
    try:
        status = _git_status()
        branch = _branch_name()
    except subprocess.CalledProcessError as exc:
        print(f"RUNTIME_GIT_BASELINE_WARN git failed: {exc.stderr.strip() or exc}")
        return 0

    payload = {
        "mode": "trader",
        "session_start": session_start,
        "captured_at": int(time.time()),
        "branch": branch,
        "dirty_paths": sorted(status),
    }
    _write_baseline(TRADER_BASELINE, payload)
    dirty_text = ",".join(payload["dirty_paths"][:5]) if payload["dirty_paths"] else "clean"
    extra = max(len(payload["dirty_paths"]) - 5, 0)
    suffix = f" +{extra} more" if extra else ""
    print(f"RUNTIME_GIT_BASELINE_OK branch={branch or 'detached'} dirty={dirty_text}{suffix}")
    return 0


def _validate_state_if_needed(paths: list[str]) -> None:
    if "collab_trade/state.md" not in paths:
        return
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "validate_trader_state.py"), str(ROOT / "collab_trade" / "state.md")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stdout or result.stderr or "validation failed").strip()
        raise RuntimeError(f"state validation failed before git sync: {detail}")


def _compose_trader_message() -> str:
    state_path = ROOT / "collab_trade" / "state.md"
    try:
        text = state_path.read_text()
    except FileNotFoundError:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return f"state: sync trader runtime {stamp}"
    match = re.search(r"^\*\*Last Updated\*\*:\s*(.+)$", text, re.M)
    stamp = match.group(1).strip() if match else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"state: sync trader runtime {stamp}"


def _compose_daily_review_message(review_date: str) -> str:
    return f"state: sync daily review memory {review_date}"


def _stage_commit_push(paths: list[str], message: str) -> int:
    if not paths:
        print("RUNTIME_GIT_SYNC_SKIP no eligible tracked runtime changes")
        return 0

    _validate_state_if_needed(paths)
    _run_git(["add", "--", *paths], timeout=15)

    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *paths],
        cwd=str(ROOT),
        timeout=10,
        check=False,
    )
    if diff.returncode == 0:
        print("RUNTIME_GIT_SYNC_SKIP nothing staged after filtering")
        return 0

    commit = _run_git(["commit", "-m", message], timeout=30)
    commit_hash = commit.stdout.split()[1].strip("[]") if commit.stdout.startswith("[") else "unknown"
    print(f"RUNTIME_GIT_COMMIT_OK commit={commit_hash} files={len(paths)}")

    push = _run_git(["push", "origin", "main"], timeout=60)
    if push.stdout.strip():
        print(push.stdout.strip())
    if push.stderr.strip():
        print(push.stderr.strip())
    print(f"RUNTIME_GIT_PUSH_OK branch=main commit={commit_hash}")
    return 0


def sync_trader(session_start: int) -> int:
    baseline = _load_baseline(TRADER_BASELINE)
    if not baseline:
        print("RUNTIME_GIT_SYNC_SKIP missing trader baseline")
        return 0
    if baseline.get("session_start") != session_start:
        print(
            "RUNTIME_GIT_SYNC_SKIP baseline session mismatch "
            f"expected={session_start} actual={baseline.get('session_start')}"
        )
        return 0

    branch = _branch_name()
    if branch != "main" or baseline.get("branch") != "main":
        print(f"RUNTIME_GIT_SYNC_SKIP branch={branch or 'detached'} baseline={baseline.get('branch')}")
        return 0

    current = _git_status()
    current_paths = set(current)
    baseline_paths = set(baseline.get("dirty_paths") or [])
    if baseline_paths:
        preview = ",".join(sorted(baseline_paths)[:5])
        extra = max(len(baseline_paths) - 5, 0)
        suffix = f" +{extra} more" if extra else ""
        print(f"RUNTIME_GIT_SYNC_SKIP repo was already dirty at session start: {preview}{suffix}")
        return 0

    disallowed = sorted(path for path in current_paths if not _is_allowed(path, TRADER_ALLOWED_PATTERNS))
    if disallowed:
        preview = ",".join(disallowed[:5])
        extra = max(len(disallowed) - 5, 0)
        suffix = f" +{extra} more" if extra else ""
        print(f"RUNTIME_GIT_SYNC_SKIP unrelated dirty paths: {preview}{suffix}")
        return 0

    changed = sorted(path for path in current_paths if path not in baseline_paths)
    return _stage_commit_push(changed, _compose_trader_message())


def sync_daily_review(review_date: str) -> int:
    branch = _branch_name()
    if branch != "main":
        print(f"RUNTIME_GIT_SYNC_SKIP branch={branch or 'detached'}")
        return 0

    current = _git_status()
    if not current:
        print("RUNTIME_GIT_SYNC_SKIP no dirty files")
        return 0

    disallowed = sorted(path for path in current if not _is_allowed(path, DAILY_REVIEW_ALLOWED_PATTERNS))
    if disallowed:
        preview = ",".join(disallowed[:5])
        extra = max(len(disallowed) - 5, 0)
        suffix = f" +{extra} more" if extra else ""
        print(f"RUNTIME_GIT_SYNC_SKIP unrelated dirty paths: {preview}{suffix}")
        return 0

    changed = sorted(current)
    return _stage_commit_push(changed, _compose_daily_review_message(review_date))


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely auto-sync tracked runtime files to main")
    sub = parser.add_subparsers(dest="cmd", required=True)

    snap = sub.add_parser("snapshot-trader-baseline", help="Record the trader session's starting git dirtiness")
    snap.add_argument("--session-start", required=True, type=int)

    trader = sub.add_parser("sync-trader", help="Commit/push trader runtime files if the repo stayed runtime-only dirty")
    trader.add_argument("--session-start", required=True, type=int)

    review = sub.add_parser("sync-daily-review", help="Commit/push daily-review memory files if no unrelated dirt exists")
    review.add_argument("--date", required=True, help="Reviewed UTC date, used in the commit message")

    args = parser.parse_args()
    try:
        if args.cmd == "snapshot-trader-baseline":
            return snapshot_trader_baseline(args.session_start)
        if args.cmd == "sync-trader":
            return sync_trader(args.session_start)
        if args.cmd == "sync-daily-review":
            return sync_daily_review(args.date)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        print(f"RUNTIME_GIT_SYNC_ERROR {detail}")
        return 1
    except Exception as exc:
        print(f"RUNTIME_GIT_SYNC_ERROR {exc}")
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

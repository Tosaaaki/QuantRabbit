"""Pause and restore QuantRabbit scheduled tasks across the weekend.

The live trader contract allows exactly one trader scheduler at a time.  This
module snapshots the current scheduler state before pausing weekend tasks, then
restores only that snapshot on Monday so a disabled Claude trader is not
accidentally enabled.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.analysis.market_status import compute_market_status

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - kept for older local Python.
    tomllib = None  # type: ignore[assignment]


CODEX_ACTIVE = "ACTIVE"
CODEX_PAUSED = "PAUSED"
DEFAULT_CODEX_TASK_IDS = ("qr-trader", "qr-news-digest", "qr-hole-audit", "qr-self-improvement-watch")
DEFAULT_CLAUDE_TASK_IDS = ("trader", "trader_v2")
DEFAULT_DECABOT_LAUNCHD_LABELS = ("com.decabot.ai", "com.decabot.monitor", "com.decabot.review")
CODEX_TASK_EXCLUDED_PREFIXES = ("qr-weekend-",)
CODEX_RESTORE_FORCED_PAUSED_TASK_IDS = frozenset({"qr-self-improvement-watch"})
WEEKDAY_CODES = frozenset({"MO", "TU", "WE", "TH", "FR"})
QUANT_RABBIT_PROJECT_BASENAMES = frozenset({"QuantRabbit", "QuantRabbit-live"})
TRADER_TASK_KEYS = frozenset({"codex:qr-trader", "claude:trader", "claude:trader_v2"})


class TaskSwitchError(RuntimeError):
    """Raised when restoring would create an unsafe scheduler state."""


@dataclass(frozen=True)
class TaskSpec:
    key: str
    kind: str
    task_id: str
    path: Path


@dataclass(frozen=True)
class Change:
    key: str
    path: str
    field: str
    before: str | bool | None
    after: str | bool | None
    changed: bool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("pause", "restore", "ack-codex-scheduler-refresh"),
    )
    parser.add_argument(
        "--operation-id",
        help="Exact pending Codex scheduler refresh id returned by pause/restore.",
    )
    parser.add_argument(
        "--updated-task",
        action="append",
        default=[],
        help="Caller attestation in task_id=STATUS form; repeat in supplied row order.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-market-open",
        action="store_true",
        help=(
            "For scheduled restore only: keep the weekend snapshot paused until "
            "the DST-aware New York weekly market boundary is open."
        ),
    )
    parser.add_argument(
        "--require-market-closed",
        action="store_true",
        help=(
            "For scheduled pause only: keep tasks active until the DST-aware "
            "New York weekly market boundary is closed."
        ),
    )
    args = parser.parse_args(argv)
    if args.require_market_open and args.action != "restore":
        parser.error("--require-market-open is valid only with restore")
    if args.require_market_closed and args.action != "pause":
        parser.error("--require-market-closed is valid only with pause")
    if args.require_market_open and args.require_market_closed:
        parser.error("market-open and market-closed requirements are mutually exclusive")
    if args.action == "ack-codex-scheduler-refresh" and not args.operation_id:
        parser.error("--operation-id is required for ack-codex-scheduler-refresh")
    if args.action == "ack-codex-scheduler-refresh" and not args.updated_task:
        parser.error("at least one --updated-task is required for ack-codex-scheduler-refresh")
    if args.action != "ack-codex-scheduler-refresh" and args.operation_id:
        parser.error("--operation-id is valid only with ack-codex-scheduler-refresh")
    if args.action != "ack-codex-scheduler-refresh" and args.updated_task:
        parser.error("--updated-task is valid only with ack-codex-scheduler-refresh")
    try:
        result = switch_tasks(
            args.action,
            dry_run=args.dry_run,
            require_market_open=args.require_market_open,
            require_market_closed=args.require_market_closed,
            operation_id=args.operation_id,
            updated_tasks=tuple(args.updated_task),
        )
    except TaskSwitchError as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def switch_tasks(
    action: str,
    *,
    dry_run: bool = False,
    require_market_open: bool = False,
    require_market_closed: bool = False,
    operation_id: str | None = None,
    updated_tasks: tuple[str, ...] = (),
    now: datetime | None = None,
) -> dict[str, Any]:
    state_file = _state_file()
    clock = now or datetime.now(timezone.utc)
    timestamp = clock.replace(microsecond=0).isoformat()
    if action == "ack-codex-scheduler-refresh":
        if dry_run:
            raise TaskSwitchError("scheduler refresh acknowledgement cannot be dry-run")
        if not operation_id:
            raise TaskSwitchError("scheduler refresh acknowledgement requires operation_id")
        with _state_lock(state_file):
            return _ack_codex_scheduler_refresh(
                state_file=state_file,
                operation_id=operation_id,
                updated_tasks=updated_tasks,
                timestamp=timestamp,
            )
    specs = _task_specs()
    with _state_lock(state_file):
        if action == "pause":
            if require_market_closed:
                waiting = _scheduled_pause_wait(
                    specs=specs,
                    state_file=state_file,
                    dry_run=dry_run,
                    now=clock,
                )
                if waiting is not None:
                    return waiting
            return _pause(specs, state_file=state_file, dry_run=dry_run, timestamp=timestamp)
        if action == "restore":
            if require_market_open:
                waiting = _scheduled_restore_wait(
                    state_file=state_file,
                    dry_run=dry_run,
                    now=clock,
                )
                if waiting is not None:
                    return waiting
            return _restore(specs, state_file=state_file, dry_run=dry_run, timestamp=timestamp)
        raise TaskSwitchError(f"unsupported action: {action}")


def _scheduled_pause_wait(
    *,
    specs: list[TaskSpec],
    state_file: Path,
    dry_run: bool,
    now: datetime,
) -> dict[str, Any] | None:
    """Keep the last winter-market hour active without creating a snapshot."""

    market_status = compute_market_status(now)
    if not market_status.is_fx_open:
        return None
    current_tasks = {spec.key: _read_task(spec) for spec in specs}
    return {
        "status": "WAITING_FOR_MARKET_CLOSE",
        "action": "pause",
        "dry_run": dry_run,
        "state_file": str(state_file),
        "baseline_active_trader_tasks": _active_trader_keys(current_tasks),
        "changed_count": 0,
        "changes": [],
        "warnings": ["weekly market is still open; tasks and snapshot left unchanged"],
        "market_status": market_status.to_dict(),
    }


def _scheduled_restore_wait(
    *,
    state_file: Path,
    dry_run: bool,
    now: datetime,
) -> dict[str, Any] | None:
    """Leave the Saturday snapshot untouched until the real weekly open.

    The automation runs at both 06:00 and 07:00 JST.  New York summer time
    opens at the first run; winter time opens at the second.  Manual restore
    retains its explicit operator semantics because this check is opt-in.
    """

    state = _load_state(state_file)
    mode = state.get("mode")
    if mode not in {"paused", "restored"}:
        raise TaskSwitchError(f"state file is not a restorable snapshot: mode={mode!r}")
    baseline_tasks = state.get("tasks")
    if not isinstance(baseline_tasks, dict):
        raise TaskSwitchError("state file is missing task snapshot")
    active_traders = _active_trader_keys(baseline_tasks)
    if len(active_traders) > 1:
        joined = ", ".join(active_traders)
        raise TaskSwitchError(f"refusing to restore multiple trader schedulers: {joined}")

    market_status = compute_market_status(now)
    if market_status.is_fx_open:
        return None
    return {
        "status": "WAITING_FOR_MARKET_OPEN",
        "action": "restore",
        "dry_run": dry_run,
        "state_file": str(state_file),
        "baseline_active_trader_tasks": active_traders,
        "changed_count": 0,
        "changes": [],
        "warnings": ["weekly market is still closed; paused snapshot preserved"],
        "market_status": market_status.to_dict(),
    }


def _pause(
    specs: list[TaskSpec],
    *,
    state_file: Path,
    dry_run: bool,
    timestamp: str,
) -> dict[str, Any]:
    existing = _load_state_if_present(state_file)
    warnings: list[str] = []
    existing_pending = existing.get("codex_scheduler_refresh_pending")
    if isinstance(existing_pending, dict) and existing_pending.get("action") != "pause":
        raise TaskSwitchError(
            "refusing to replace an unacknowledged Codex scheduler refresh with pause"
        )
    if existing.get("mode") == "paused":
        state = existing
        state["last_pause_reapplied_at_utc"] = timestamp
        warnings.append("existing paused snapshot reused")
    else:
        tasks = {spec.key: _read_task(spec) for spec in specs}
        state = {
            "schema_version": 1,
            "mode": "paused",
            "created_at_utc": timestamp,
            "tasks": tasks,
            "managed_task_keys": [spec.key for spec in specs],
        }
    if not dry_run:
        _write_state(state_file, state)

    desired = {spec.key: _paused_task_state(spec, _read_task(spec)) for spec in specs}
    if not dry_run:
        _assert_pending_refresh_compatible(state, action="pause", specs=specs, desired=desired)
    changes = _apply_states(specs, desired, action="pause", dry_run=dry_run)
    refresh_rows = _codex_scheduler_refresh_rows(changes, dry_run=dry_run)
    operation_id = None
    if not dry_run:
        state["pause_applied_at_utc"] = timestamp
        state["last_changes"] = [_change_dict(change) for change in changes]
        operation_id = _stage_codex_scheduler_refresh(
            state,
            action="pause",
            timestamp=timestamp,
            rows=refresh_rows,
        )
        _write_state(state_file, state)
    return _summary(
        action="pause",
        dry_run=dry_run,
        state_file=state_file,
        baseline_tasks=state.get("tasks", {}),
        changes=changes,
        refresh_rows=refresh_rows,
        operation_id=operation_id,
        warnings=warnings,
    )


def _restore(
    specs: list[TaskSpec],
    *,
    state_file: Path,
    dry_run: bool,
    timestamp: str,
) -> dict[str, Any]:
    state = _load_state(state_file)
    mode = state.get("mode")
    if mode not in {"paused", "restored"}:
        raise TaskSwitchError(f"state file is not a restorable snapshot: mode={mode!r}")

    baseline_tasks = state.get("tasks")
    if not isinstance(baseline_tasks, dict):
        raise TaskSwitchError("state file is missing task snapshot")
    active_traders = _active_trader_keys(baseline_tasks)
    if len(active_traders) > 1:
        joined = ", ".join(active_traders)
        raise TaskSwitchError(f"refusing to restore multiple trader schedulers: {joined}")

    desired = {
        spec.key: _restore_task_state(spec, baseline_tasks[spec.key])
        for spec in specs
        if spec.key in baseline_tasks
    }
    if not dry_run:
        _assert_pending_refresh_compatible(state, action="restore", specs=specs, desired=desired)
    changes = _apply_states(specs, desired, action="restore", dry_run=dry_run)
    refresh_rows = _codex_scheduler_refresh_rows(changes, dry_run=dry_run)
    operation_id = None
    warnings: list[str] = []
    if mode == "restored":
        warnings.append("weekend snapshot already restored")
        if any(change.changed for change in changes):
            warnings.append("restored snapshot drift reconciled")
    if not dry_run:
        if mode == "paused":
            state["restore_requested_at_utc"] = timestamp
        elif any(change.changed for change in changes):
            state["last_restore_reconciliation_requested_at_utc"] = timestamp
        state["last_changes"] = [_change_dict(change) for change in changes]
        operation_id = _stage_codex_scheduler_refresh(
            state,
            action="restore",
            timestamp=timestamp,
            rows=refresh_rows,
        )
        if not refresh_rows:
            _finalize_scheduler_refresh_state(state, action="restore", timestamp=timestamp)
        _write_state(state_file, state)
    return _summary(
        action="restore",
        dry_run=dry_run,
        state_file=state_file,
        baseline_tasks=baseline_tasks,
        changes=changes,
        refresh_rows=refresh_rows,
        operation_id=operation_id,
        warnings=warnings,
    )


def _task_specs() -> list[TaskSpec]:
    codex_root = Path(os.environ.get("QR_WEEKEND_CODEX_AUTOMATION_ROOT", "~/.codex/automations")).expanduser()
    claude_root = Path(os.environ.get("QR_WEEKEND_CLAUDE_TASK_ROOT", "~/.claude/scheduled-tasks")).expanduser()
    specs: list[TaskSpec] = []
    for task_id in _codex_task_ids(codex_root):
        specs.append(
            TaskSpec(
                key=f"codex:{task_id}",
                kind="codex",
                task_id=task_id,
                path=codex_root / task_id / "automation.toml",
            )
        )
    for task_id in _claude_task_ids(claude_root):
        specs.append(
            TaskSpec(
                key=f"claude:{task_id}",
                kind="claude",
                task_id=task_id,
                path=claude_root / task_id / "schedule.json",
            )
        )
    specs.extend(_decabot_launchd_specs())
    return specs


def _codex_task_ids(root: Path) -> tuple[str, ...]:
    configured = _env_list("QR_WEEKEND_CODEX_TASKS", ())
    if configured:
        return configured
    discovered = _discover_codex_weekday_task_ids(root)
    return discovered or DEFAULT_CODEX_TASK_IDS


def _claude_task_ids(root: Path) -> tuple[str, ...]:
    configured = _env_list("QR_WEEKEND_CLAUDE_TASKS", ())
    if configured:
        return configured
    discovered = _discover_claude_weekday_task_ids(root)
    return discovered or DEFAULT_CLAUDE_TASK_IDS


def _decabot_launchd_specs() -> list[TaskSpec]:
    if os.environ.get("QR_WEEKEND_DECABOT_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
        return []
    labels = _env_list("QR_WEEKEND_DECABOT_LABELS", DEFAULT_DECABOT_LAUNCHD_LABELS)
    root = Path(os.environ.get("QR_WEEKEND_DECABOT_LAUNCH_AGENT_ROOT", "~/Library/LaunchAgents")).expanduser()
    specs: list[TaskSpec] = []
    for label in labels:
        specs.append(
            TaskSpec(
                key=f"decabot:{label}",
                kind="decabot_launchd",
                task_id=label,
                path=root / f"{label}.plist",
            )
        )
    return specs


def _discover_codex_weekday_task_ids(root: Path) -> tuple[str, ...]:
    if not root.exists():
        return ()
    task_ids: list[str] = []
    for path in sorted(root.glob("*/automation.toml")):
        payload = _read_toml_payload(path)
        task_id = str(payload.get("id") or path.parent.name)
        if not task_id.startswith("qr-"):
            continue
        if _codex_task_excluded(task_id):
            continue
        if not _schedule_touches_weekdays(str(payload.get("rrule") or "")):
            continue
        task_ids.append(task_id)
    return tuple(dict.fromkeys(task_ids))


def _discover_claude_weekday_task_ids(root: Path) -> tuple[str, ...]:
    if not root.exists():
        return ()
    task_ids: list[str] = []
    for path in sorted(root.glob("*/schedule.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise TaskSwitchError(f"invalid Claude schedule JSON: {path}: {exc}") from exc
        if not isinstance(payload, dict):
            continue
        project = str(payload.get("project") or "")
        if Path(project).name not in QUANT_RABBIT_PROJECT_BASENAMES:
            continue
        if not _schedule_touches_weekdays(str(payload.get("cronExpression") or "")):
            continue
        task_id = str(payload.get("taskId") or path.parent.name)
        task_ids.append(task_id)
    return tuple(dict.fromkeys(task_ids))


def _codex_task_excluded(task_id: str) -> bool:
    return any(task_id.startswith(prefix) for prefix in CODEX_TASK_EXCLUDED_PREFIXES)


def _schedule_touches_weekdays(schedule: str) -> bool:
    if not schedule.strip():
        return False
    rrule_match = re.search(r"(?:^|;)BYDAY=([^;]+)", schedule)
    if rrule_match:
        days = {day.strip().upper() for day in rrule_match.group(1).split(",") if day.strip()}
        return bool(days & WEEKDAY_CODES)
    cron_parts = schedule.split()
    if len(cron_parts) == 5:
        return _cron_day_of_week_touches_weekdays(cron_parts[4])
    return True


def _cron_day_of_week_touches_weekdays(field: str) -> bool:
    normalized = field.strip().upper()
    if not normalized or normalized in {"*", "?", "MON-FRI"}:
        return True
    if normalized in {"SAT", "SUN", "SA", "SU", "6", "0", "7", "6,0", "6,7"}:
        return False
    weekday_names = {
        "MON": 1,
        "MO": 1,
        "TUE": 2,
        "TU": 2,
        "WED": 3,
        "WE": 3,
        "THU": 4,
        "TH": 4,
        "FRI": 5,
        "FR": 5,
        "SAT": 6,
        "SA": 6,
        "SUN": 0,
        "SU": 0,
    }
    for part in normalized.split(","):
        token = part.split("/", 1)[0]
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = weekday_names.get(start_raw, _parse_cron_day_int(start_raw))
            end = weekday_names.get(end_raw, _parse_cron_day_int(end_raw))
            if start is None or end is None:
                return True
            days = _cron_day_range(start, end)
        else:
            day = weekday_names.get(token, _parse_cron_day_int(token))
            if day is None:
                return True
            days = {day}
        if days & {1, 2, 3, 4, 5}:
            return True
    return False


def _parse_cron_day_int(value: str) -> int | None:
    if not value.isdigit():
        return None
    day = int(value)
    if day == 7:
        return 0
    if 0 <= day <= 6:
        return day
    return None


def _cron_day_range(start: int, end: int) -> set[int]:
    if start <= end:
        return set(range(start, end + 1))
    return set(range(start, 7)) | set(range(0, end + 1))


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if not value:
        return default
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or default


def _state_file() -> Path:
    default = Path("~/.codex/quant_rabbit_weekend_task_state.json").expanduser()
    return Path(os.environ.get("QR_WEEKEND_TASK_STATE_FILE", str(default))).expanduser()


@contextmanager
def _state_lock(state_file: Path):
    """Serialize snapshot/pending/ack read-modify-write operations."""

    lock_path = state_file.with_name(state_file.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_task(spec: TaskSpec) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": spec.kind,
        "task_id": spec.task_id,
        "path": str(spec.path),
        "exists": spec.path.exists(),
    }
    if not spec.path.exists():
        return base
    if spec.kind == "codex":
        base["status"] = _read_codex_status(spec.path)
    elif spec.kind == "claude":
        base["enabled"] = _read_claude_enabled(spec.path)
    elif spec.kind == "decabot_launchd":
        base["loaded"] = _read_launchd_loaded(spec.task_id)
        base["domain"] = _launchd_domain()
    else:  # pragma: no cover - defensive guard for future task kinds.
        raise TaskSwitchError(f"unsupported task kind: {spec.kind}")
    return base


def _paused_task_state(spec: TaskSpec, current: dict[str, Any]) -> dict[str, Any]:
    desired = dict(current)
    if spec.kind == "codex":
        desired["status"] = CODEX_PAUSED
    elif spec.kind == "claude":
        desired["enabled"] = False
    elif spec.kind == "decabot_launchd":
        desired["loaded"] = False
    return desired


def _restore_task_state(spec: TaskSpec, snapshot: dict[str, Any]) -> dict[str, Any]:
    desired = dict(snapshot)
    if spec.kind == "codex" and spec.task_id in CODEX_RESTORE_FORCED_PAUSED_TASK_IDS:
        desired["status"] = CODEX_PAUSED
    return desired


def _apply_states(
    specs: list[TaskSpec],
    desired_by_key: dict[str, dict[str, Any]],
    *,
    action: str,
    dry_run: bool,
) -> list[Change]:
    changes: list[Change] = []
    specs_by_key = {spec.key: spec for spec in specs}
    ordered_keys = sorted(
        desired_by_key,
        key=lambda key: _task_apply_priority(specs_by_key.get(key), action=action),
    )
    for key in ordered_keys:
        desired = desired_by_key[key]
        spec = specs_by_key.get(key)
        if spec is None:
            continue
        if not desired.get("exists", True):
            continue
        if not spec.path.exists():
            raise TaskSwitchError(f"task existed in snapshot but is missing now: {spec.path}")
        before = _read_task(spec)
        if spec.kind == "codex":
            after_status = str(desired.get("status", CODEX_PAUSED))
            before_status = before.get("status")
            changed = before_status != after_status
            if changed and not dry_run:
                _write_codex_status(spec.path, after_status)
            changes.append(
                Change(
                    key=key,
                    path=str(spec.path),
                    field="status",
                    before=before_status,
                    after=after_status,
                    changed=changed,
                )
            )
        elif spec.kind == "claude":
            after_enabled = bool(desired.get("enabled", False))
            before_enabled = bool(before.get("enabled", False))
            changed = before_enabled != after_enabled
            if changed and not dry_run:
                _write_claude_enabled(spec.path, after_enabled)
            changes.append(
                Change(
                    key=key,
                    path=str(spec.path),
                    field="enabled",
                    before=before_enabled,
                    after=after_enabled,
                    changed=changed,
                )
            )
        elif spec.kind == "decabot_launchd":
            after_loaded = bool(desired.get("loaded", False))
            before_loaded = bool(before.get("loaded", False))
            changed = before_loaded != after_loaded
            if changed and not dry_run:
                _write_launchd_loaded(spec, after_loaded)
            changes.append(
                Change(
                    key=key,
                    path=str(spec.path),
                    field="loaded",
                    before=before_loaded,
                    after=after_loaded,
                    changed=changed,
                )
            )
    return changes


def _task_apply_priority(spec: TaskSpec | None, *, action: str) -> tuple[int, str]:
    key = spec.key if spec is not None else ""
    is_trader = key in TRADER_TASK_KEYS
    if action == "pause":
        return (0 if is_trader else 1, key)
    if action == "restore":
        return (1 if is_trader else 0, key)
    raise TaskSwitchError(f"unsupported apply action: {action}")


def _read_codex_status(path: Path) -> str:
    text = path.read_text()
    payload = _read_toml_payload(path, text=text)
    status = payload.get("status")
    if isinstance(status, str):
        return status
    match = re.search(r'(?m)^status = "([^"]+)"$', text)
    if not match:
        raise TaskSwitchError(f"missing Codex automation status: {path}")
    return match.group(1)


def _read_toml_payload(path: Path, *, text: str | None = None) -> dict[str, Any]:
    source = path.read_text() if text is None else text
    if tomllib is None:
        return {}
    try:
        payload = tomllib.loads(source)
    except tomllib.TOMLDecodeError as exc:
        raise TaskSwitchError(f"invalid TOML: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_codex_status(path: Path, status: str) -> None:
    text = path.read_text()
    new_text, count = re.subn(r'(?m)^status = "([^"]+)"$', f'status = "{status}"', text, count=1)
    if count != 1:
        raise TaskSwitchError(f"missing Codex automation status: {path}")
    path.write_text(new_text)


def _read_claude_enabled(path: Path) -> bool:
    payload = json.loads(path.read_text())
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise TaskSwitchError(f"missing Claude schedule enabled boolean: {path}")
    return enabled


def _write_claude_enabled(path: Path, enabled: bool) -> None:
    payload = json.loads(path.read_text())
    if not isinstance(payload.get("enabled"), bool):
        raise TaskSwitchError(f"missing Claude schedule enabled boolean: {path}")
    payload["enabled"] = enabled
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _launchd_domain() -> str:
    return os.environ.get("QR_WEEKEND_LAUNCHD_DOMAIN", f"gui/{os.getuid()}")


def _run_launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["launchctl", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise TaskSwitchError("launchctl is required for DecaBot weekend launchd management") from exc


def _read_launchd_loaded(label: str) -> bool:
    proc = _run_launchctl("print", f"{_launchd_domain()}/{label}")
    if proc.returncode == 0:
        return True
    stderr = proc.stderr or ""
    if "Could not find service" in stderr or "No such process" in stderr:
        return False
    raise TaskSwitchError(f"failed to inspect launchd service {label}: {stderr.strip() or proc.stdout.strip()}")


def _write_launchd_loaded(spec: TaskSpec, loaded: bool) -> None:
    domain = _launchd_domain()
    if loaded:
        _checked_launchctl("bootstrap", domain, str(spec.path), label=spec.task_id)
        _checked_launchctl("enable", f"{domain}/{spec.task_id}", label=spec.task_id)
    else:
        proc = _run_launchctl("bootout", domain, str(spec.path))
        if proc.returncode == 0:
            return
        stderr = proc.stderr or ""
        if "Could not find service" in stderr or "No such process" in stderr:
            return
        raise TaskSwitchError(f"failed to stop launchd service {spec.task_id}: {stderr.strip() or proc.stdout.strip()}")


def _checked_launchctl(*args: str, label: str) -> None:
    proc = _run_launchctl(*args)
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip()
        raise TaskSwitchError(f"failed to update launchd service {label}: {message}")


def _load_state_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_state(path)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise TaskSwitchError(f"weekend task state file is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TaskSwitchError(f"weekend task state is not an object: {path}")
    return payload


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        tmp.write(rendered)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _active_trader_keys(tasks: dict[str, Any]) -> list[str]:
    active: list[str] = []
    for key in sorted(TRADER_TASK_KEYS):
        state = tasks.get(key)
        if not isinstance(state, dict) or not state.get("exists", False):
            continue
        if key.startswith("codex:") and state.get("status") == CODEX_ACTIVE:
            active.append(key)
        if key.startswith("claude:") and state.get("enabled") is True:
            active.append(key)
    return active


def _codex_scheduler_refresh_rows(
    changes: list[Change],
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    if dry_run:
        return []
    return [
        {
            "task_id": change.key.split(":", 1)[1],
            "status": change.after,
            "automation_path": change.path,
            "config_file_changed": change.changed,
        }
        for change in changes
        if change.key.startswith("codex:") and change.field == "status"
    ]


def _stage_codex_scheduler_refresh(
    state: dict[str, Any],
    *,
    action: str,
    timestamp: str,
    rows: list[dict[str, Any]],
) -> str | None:
    existing = state.get("codex_scheduler_refresh_pending")
    if not rows:
        if isinstance(existing, dict):
            raise TaskSwitchError("cannot discard an unacknowledged Codex scheduler refresh")
        return None
    identity_rows = _scheduler_refresh_identity_rows(rows)
    if isinstance(existing, dict):
        existing_action = str(existing.get("action") or "")
        existing_rows = existing.get("rows")
        existing_identity = (
            _scheduler_refresh_identity_rows(existing_rows)
            if isinstance(existing_rows, list)
            else []
        )
        if existing_action == action and existing_identity == identity_rows:
            existing_id = existing.get("operation_id")
            if not isinstance(existing_id, str) or not existing_id:
                raise TaskSwitchError("pending Codex scheduler refresh has no operation id")
            return existing_id
        raise TaskSwitchError(
            "refusing to replace an unacknowledged Codex scheduler refresh: "
            f"pending_action={existing_action!r} requested_action={action!r}"
        )
    seed = json.dumps(
        {
            "action": action,
            "snapshot_created_at_utc": state.get("created_at_utc"),
            "rows": identity_rows,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    operation_id = "codex-refresh-" + hashlib.sha256(seed.encode()).hexdigest()[:24]
    state["codex_scheduler_refresh_pending"] = {
        "operation_id": operation_id,
        "action": action,
        "requested_at_utc": timestamp,
        "rows": rows,
    }
    return operation_id


def _assert_pending_refresh_compatible(
    state: dict[str, Any],
    *,
    action: str,
    specs: list[TaskSpec],
    desired: dict[str, dict[str, Any]],
) -> None:
    existing = state.get("codex_scheduler_refresh_pending")
    if not isinstance(existing, dict):
        return
    planned_rows: list[dict[str, Any]] = []
    specs_by_key = {spec.key: spec for spec in specs}
    ordered_keys = sorted(
        desired,
        key=lambda key: _task_apply_priority(specs_by_key.get(key), action=action),
    )
    for key in ordered_keys:
        spec = specs_by_key.get(key)
        target = desired[key]
        if (
            spec is None
            or spec.kind != "codex"
            or not target.get("exists", True)
            or not spec.path.exists()
        ):
            continue
        planned_rows.append(
            {
                "task_id": spec.task_id,
                "status": str(target.get("status", CODEX_PAUSED)),
                "automation_path": str(spec.path),
            }
        )
    existing_rows = existing.get("rows")
    if (
        existing.get("action") != action
        or not isinstance(existing_rows, list)
        or _scheduler_refresh_identity_rows(existing_rows)
        != _scheduler_refresh_identity_rows(planned_rows)
    ):
        raise TaskSwitchError(
            "refusing task mutation while a different Codex scheduler refresh is unacknowledged"
        )


def _scheduler_refresh_identity_rows(rows: list[Any]) -> list[dict[str, str]]:
    identity: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TaskSwitchError("scheduler refresh row must be an object")
        task_id = str(row.get("task_id") or "")
        status = str(row.get("status") or "")
        automation_path = str(row.get("automation_path") or "")
        if not task_id or status not in {CODEX_ACTIVE, CODEX_PAUSED} or not automation_path:
            raise TaskSwitchError("scheduler refresh row identity is incomplete")
        identity.append(
            {
                "task_id": task_id,
                "status": status,
                "automation_path": automation_path,
            }
        )
    return identity


def _ack_codex_scheduler_refresh(
    *,
    state_file: Path,
    operation_id: str,
    updated_tasks: tuple[str, ...],
    timestamp: str,
) -> dict[str, Any]:
    state = _load_state(state_file)
    pending = state.get("codex_scheduler_refresh_pending")
    if not isinstance(pending, dict):
        acknowledged = state.get("codex_scheduler_refresh_acknowledged")
        if isinstance(acknowledged, dict) and acknowledged.get("operation_id") == operation_id:
            expected_tasks = tuple(str(item) for item in acknowledged.get("updated_tasks") or [])
            if updated_tasks != expected_tasks:
                raise TaskSwitchError("scheduler refresh acknowledgement attestation mismatch")
            return {
                "status": "OK",
                "action": "ack-codex-scheduler-refresh",
                "operation_id": operation_id,
                "acknowledged_action": acknowledged.get("action"),
                "acknowledged_task_count": len(expected_tasks),
                "caller_attestation": list(expected_tasks),
                "idempotent_replay": True,
                "state_file": str(state_file),
            }
        raise TaskSwitchError("no Codex scheduler refresh is pending")
    expected = pending.get("operation_id")
    if operation_id != expected:
        raise TaskSwitchError(
            f"scheduler refresh operation mismatch: expected={expected!r} actual={operation_id!r}"
        )
    action = str(pending.get("action") or "")
    if action not in {"pause", "restore"}:
        raise TaskSwitchError(f"invalid pending scheduler refresh action: {action!r}")
    rows = pending.get("rows")
    if not isinstance(rows, list) or not rows:
        raise TaskSwitchError("pending scheduler refresh has no task rows")
    expected_tasks = tuple(
        f"{row['task_id']}={row['status']}"
        for row in _scheduler_refresh_identity_rows(rows)
    )
    if updated_tasks != expected_tasks:
        raise TaskSwitchError(
            "scheduler refresh caller attestation mismatch: "
            f"expected={list(expected_tasks)!r} actual={list(updated_tasks)!r}"
        )

    _finalize_scheduler_refresh_state(state, action=action, timestamp=timestamp)
    state["codex_scheduler_refresh_acknowledged"] = {
        "operation_id": operation_id,
        "action": action,
        "acknowledged_at_utc": timestamp,
        "task_ids": [str(row.get("task_id") or "") for row in rows if isinstance(row, dict)],
        "updated_tasks": list(expected_tasks),
    }
    state.pop("codex_scheduler_refresh_pending", None)
    _write_state(state_file, state)
    return {
        "status": "OK",
        "action": "ack-codex-scheduler-refresh",
        "operation_id": operation_id,
        "acknowledged_action": action,
        "acknowledged_task_count": len(rows),
        "caller_attestation": list(expected_tasks),
        "idempotent_replay": False,
        "state_file": str(state_file),
    }


def _finalize_scheduler_refresh_state(
    state: dict[str, Any],
    *,
    action: str,
    timestamp: str,
) -> None:
    if action == "restore":
        was_restored = state.get("mode") == "restored"
        state["mode"] = "restored"
        if was_restored:
            state["last_restore_reconciled_at_utc"] = timestamp
        else:
            state["restored_at_utc"] = timestamp
    elif action == "pause":
        state["mode"] = "paused"
        state["pause_scheduler_acknowledged_at_utc"] = timestamp
    else:  # pragma: no cover - internal callers are constrained.
        raise TaskSwitchError(f"unsupported scheduler refresh action: {action}")


def _summary(
    *,
    action: str,
    dry_run: bool,
    state_file: Path,
    baseline_tasks: dict[str, Any],
    changes: list[Change],
    refresh_rows: list[dict[str, Any]],
    operation_id: str | None,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "status": "PENDING_CODEX_SCHEDULER_REFRESH" if refresh_rows else "OK",
        "action": action,
        "dry_run": dry_run,
        "state_file": str(state_file),
        "baseline_active_trader_tasks": _active_trader_keys(baseline_tasks),
        "changed_count": sum(1 for change in changes if change.changed),
        "changes": [_change_dict(change) for change in changes],
        # Editing automation.toml is durable state, but Codex Desktop owns the
        # in-memory scheduler.  The weekend automation must resubmit every row
        # through automation_update, including unchanged rows on its retry,
        # before it can claim that pause/restore reached the scheduler.
        "codex_scheduler_refresh_operation_id": operation_id,
        "codex_scheduler_refresh_required": refresh_rows,
        "warnings": warnings,
    }


def _change_dict(change: Change) -> dict[str, Any]:
    return {
        "key": change.key,
        "path": change.path,
        "field": change.field,
        "before": change.before,
        "after": change.after,
        "changed": change.changed,
    }


if __name__ == "__main__":
    raise SystemExit(main())

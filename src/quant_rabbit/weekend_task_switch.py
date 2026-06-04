"""Pause and restore QuantRabbit scheduled tasks across the weekend.

The live trader contract allows exactly one trader scheduler at a time.  This
module snapshots the current scheduler state before pausing weekend tasks, then
restores only that snapshot on Monday so a disabled Claude trader is not
accidentally enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - kept for older local Python.
    tomllib = None  # type: ignore[assignment]


CODEX_ACTIVE = "ACTIVE"
CODEX_PAUSED = "PAUSED"
DEFAULT_CODEX_TASK_IDS = ("qr-trader", "qr-news-digest")
DEFAULT_CLAUDE_TASK_IDS = ("trader", "trader_v2")
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
    parser.add_argument("action", choices=("pause", "restore"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = switch_tasks(args.action, dry_run=args.dry_run)
    except TaskSwitchError as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def switch_tasks(action: str, *, dry_run: bool = False, now: datetime | None = None) -> dict[str, Any]:
    specs = _task_specs()
    state_file = _state_file()
    timestamp = (now or datetime.now(UTC)).replace(microsecond=0).isoformat()
    if action == "pause":
        return _pause(specs, state_file=state_file, dry_run=dry_run, timestamp=timestamp)
    if action == "restore":
        return _restore(specs, state_file=state_file, dry_run=dry_run, timestamp=timestamp)
    raise TaskSwitchError(f"unsupported action: {action}")


def _pause(
    specs: list[TaskSpec],
    *,
    state_file: Path,
    dry_run: bool,
    timestamp: str,
) -> dict[str, Any]:
    existing = _load_state_if_present(state_file)
    warnings: list[str] = []
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
    changes = _apply_states(specs, desired, dry_run=dry_run)
    if not dry_run:
        state["pause_applied_at_utc"] = timestamp
        state["last_changes"] = [_change_dict(change) for change in changes]
        _write_state(state_file, state)
    return _summary(
        action="pause",
        dry_run=dry_run,
        state_file=state_file,
        baseline_tasks=state.get("tasks", {}),
        changes=changes,
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
    if mode == "restored":
        return _summary(
            action="restore",
            dry_run=dry_run,
            state_file=state_file,
            baseline_tasks=state.get("tasks", {}),
            changes=[],
            warnings=["weekend snapshot already restored"],
        )
    if mode != "paused":
        raise TaskSwitchError(f"state file is not a paused snapshot: mode={mode!r}")

    baseline_tasks = state.get("tasks")
    if not isinstance(baseline_tasks, dict):
        raise TaskSwitchError("state file is missing task snapshot")
    active_traders = _active_trader_keys(baseline_tasks)
    if len(active_traders) > 1:
        joined = ", ".join(active_traders)
        raise TaskSwitchError(f"refusing to restore multiple trader schedulers: {joined}")

    desired = {spec.key: baseline_tasks[spec.key] for spec in specs if spec.key in baseline_tasks}
    changes = _apply_states(specs, desired, dry_run=dry_run)
    if not dry_run:
        state["mode"] = "restored"
        state["restored_at_utc"] = timestamp
        state["last_changes"] = [_change_dict(change) for change in changes]
        _write_state(state_file, state)
    return _summary(
        action="restore",
        dry_run=dry_run,
        state_file=state_file,
        baseline_tasks=baseline_tasks,
        changes=changes,
        warnings=[],
    )


def _task_specs() -> list[TaskSpec]:
    codex_root = Path(os.environ.get("QR_WEEKEND_CODEX_AUTOMATION_ROOT", "~/.codex/automations")).expanduser()
    claude_root = Path(os.environ.get("QR_WEEKEND_CLAUDE_TASK_ROOT", "~/.claude/scheduled-tasks")).expanduser()
    specs: list[TaskSpec] = []
    for task_id in _env_list("QR_WEEKEND_CODEX_TASKS", DEFAULT_CODEX_TASK_IDS):
        specs.append(
            TaskSpec(
                key=f"codex:{task_id}",
                kind="codex",
                task_id=task_id,
                path=codex_root / task_id / "automation.toml",
            )
        )
    for task_id in _env_list("QR_WEEKEND_CLAUDE_TASKS", DEFAULT_CLAUDE_TASK_IDS):
        specs.append(
            TaskSpec(
                key=f"claude:{task_id}",
                kind="claude",
                task_id=task_id,
                path=claude_root / task_id / "schedule.json",
            )
        )
    return specs


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if not value:
        return default
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or default


def _state_file() -> Path:
    default = Path("~/.codex/quant_rabbit_weekend_task_state.json").expanduser()
    return Path(os.environ.get("QR_WEEKEND_TASK_STATE_FILE", str(default))).expanduser()


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
    else:  # pragma: no cover - defensive guard for future task kinds.
        raise TaskSwitchError(f"unsupported task kind: {spec.kind}")
    return base


def _paused_task_state(spec: TaskSpec, current: dict[str, Any]) -> dict[str, Any]:
    desired = dict(current)
    if spec.kind == "codex":
        desired["status"] = CODEX_PAUSED
    elif spec.kind == "claude":
        desired["enabled"] = False
    return desired


def _apply_states(
    specs: list[TaskSpec],
    desired_by_key: dict[str, dict[str, Any]],
    *,
    dry_run: bool,
) -> list[Change]:
    changes: list[Change] = []
    specs_by_key = {spec.key: spec for spec in specs}
    for key, desired in desired_by_key.items():
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
    return changes


def _read_codex_status(path: Path) -> str:
    text = path.read_text()
    if tomllib is not None:
        try:
            payload = tomllib.loads(text)
            status = payload.get("status")
            if isinstance(status, str):
                return status
        except tomllib.TOMLDecodeError as exc:
            raise TaskSwitchError(f"invalid TOML: {path}: {exc}") from exc
    match = re.search(r'(?m)^status = "([^"]+)"$', text)
    if not match:
        raise TaskSwitchError(f"missing Codex automation status: {path}")
    return match.group(1)


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


def _summary(
    *,
    action: str,
    dry_run: bool,
    state_file: Path,
    baseline_tasks: dict[str, Any],
    changes: list[Change],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "status": "OK",
        "action": action,
        "dry_run": dry_run,
        "state_file": str(state_file),
        "baseline_active_trader_tasks": _active_trader_keys(baseline_tasks),
        "changed_count": sum(1 for change in changes if change.changed),
        "changes": [_change_dict(change) for change in changes],
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

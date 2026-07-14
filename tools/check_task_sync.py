#!/usr/bin/env python3
"""Check that the current task contract is synchronized across docs/tools."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    tomllib = None  # type: ignore[assignment]

import validate_trader_state


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_PATH = REPO_ROOT / "AGENTS.md"
CHANGELOG_PATH = REPO_ROOT / "docs" / "CHANGELOG.md"
STRATEGY_MEMORY_PATH = REPO_ROOT / "collab_trade" / "strategy_memory.md"
QR_TRADER_AUTOMATION_PATH = Path.home() / ".codex" / "automations" / "qr-trader" / "automation.toml"
QR_WEEKEND_MARKET_OFF_AUTOMATION_PATH = (
    Path.home() / ".codex" / "automations" / "qr-weekend-market-off" / "automation.toml"
)
QR_WEEKEND_MARKET_ON_AUTOMATION_PATH = (
    Path.home() / ".codex" / "automations" / "qr-weekend-market-on" / "automation.toml"
)
QR_WEEKEND_TASK_STATE_PATH = Path.home() / ".codex" / "quant_rabbit_weekend_task_state.json"
EXPECTED_QR_TRADER_RRULE = "FREQ=MINUTELY;INTERVAL=60;BYDAY=SU,MO,TU,WE,TH,FR,SA"
EXPECTED_QR_TRADER_MODEL = "gpt-5.5"
EXPECTED_QR_TRADER_REASONING = "high"
EXPECTED_QR_TRADER_CWD = "/Users/tossaki/App/QuantRabbit-live"
EXPECTED_QR_TRADER_GUARDIAN_STARTUP_READS = (
    "docs/guardian_event_report.md",
    "data/guardian_events.json",
    "data/guardian_escalation.json",
    "docs/guardian_action_review.md",
    "data/guardian_action_receipt.json",
    "data/guardian_action_cycle_result.json",
    "data/guardian_trigger_contract.json",
    "docs/guardian_trigger_contract_report.md",
    "data/guardian_receipt_consumption.json",
    "docs/guardian_receipt_consumption_report.md",
    "data/guardian_receipt_operator_review.json",
    "docs/guardian_receipt_operator_review_report.md",
    "data/qr_trader_run_watchdog.json",
    "docs/qr_trader_run_watchdog_report.md",
)
EXPECTED_QR_TRADER_RUNTIME_DRIFT_PROMPT_PHRASES = (
    "data/guardian_receipt_consumption.json",
    "data/guardian_receipt_operator_review.json",
    "named proof/acceptance evidence",
    "data/trader_goal_loop_orchestrator.json",
    "runtime drift and **do not** block the run",
)
EXPECTED_WEEKEND_SCHEDULER_REFRESH_PROMPT_PHRASES = (
    "PENDING_CODEX_SCHEDULER_REFRESH",
    "codex_scheduler_refresh_required",
    "codex_scheduler_refresh_operation_id",
    "config_file_changed=false",
    "automation_update",
    "ack-codex-scheduler-refresh --operation-id",
    "--updated-task",
    "editing automation.toml alone is not proof",
    "never turn a requested PAUSED task ACTIVE",
)
EXPECTED_WEEKEND_AUTOMATIONS = {
    "qr-weekend-market-off": {
        "name": "QR weekend market off",
        "rrule": "FREQ=WEEKLY;BYDAY=SA;BYHOUR=6,7;BYMINUTE=0",
        "command": "quant_rabbit.weekend_task_switch pause --require-market-closed",
        "order": "pause order with qr-trader first",
    },
    "qr-weekend-market-on": {
        "name": "QR weekend market on",
        "rrule": "FREQ=WEEKLY;BYDAY=MO;BYHOUR=6,7;BYMINUTE=0",
        "command": "quant_rabbit.weekend_task_switch restore --require-market-open",
        "order": "restore order with qr-trader last",
    },
}
EXPECTED_WEEKEND_CWD = "/Users/tossaki/App/QuantRabbit"
EXPECTED_WEEKEND_MODEL = "gpt-5-codex"
EXPECTED_WEEKEND_REASONING = "medium"
SOURCE_DIRT_EXPLANATION_REQUIRED_PATHS = (
    "src/quant_rabbit/automation.py",
    "src/quant_rabbit/broker/execution.py",
    "src/quant_rabbit/risk.py",
)


def _require_text(path: Path, needles: tuple[str, ...], issues: list[str]) -> None:
    text = path.read_text()
    for needle in needles:
        if needle not in text:
            issues.append(f"{path.relative_to(REPO_ROOT)} missing: {needle}")


def _validate_qr_trader_automation(issues: list[str], *, now_utc: datetime | None = None) -> None:
    if not QR_TRADER_AUTOMATION_PATH.exists():
        issues.append(f"qr-trader automation missing: {QR_TRADER_AUTOMATION_PATH}")
        return
    payload = _load_toml_payload(QR_TRADER_AUTOMATION_PATH.read_text())
    checks = {
        "rrule": EXPECTED_QR_TRADER_RRULE,
        "model": EXPECTED_QR_TRADER_MODEL,
        "reasoning_effort": EXPECTED_QR_TRADER_REASONING,
    }
    for key, expected in checks.items():
        actual = payload.get(key)
        if actual != expected:
            issues.append(f"qr-trader automation {key} expected {expected!r}, got {actual!r}")
    status = payload.get("status")
    if status != "ACTIVE" and not _qr_trader_weekend_pause_active(status, now_utc=now_utc):
        issues.append(f"qr-trader automation status expected 'ACTIVE', got {status!r}")
    cwds = payload.get("cwds")
    if cwds != [EXPECTED_QR_TRADER_CWD]:
        issues.append(f"qr-trader automation cwds expected {[EXPECTED_QR_TRADER_CWD]!r}, got {cwds!r}")
    prompt = str(payload.get("prompt") or "")
    for required_path in EXPECTED_QR_TRADER_GUARDIAN_STARTUP_READS:
        if required_path not in prompt:
            issues.append(f"qr-trader automation prompt missing guardian startup read: {required_path}")
    for required_phrase in EXPECTED_QR_TRADER_RUNTIME_DRIFT_PROMPT_PHRASES:
        if required_phrase not in prompt:
            issues.append(f"qr-trader automation prompt missing runtime drift allowance: {required_phrase}")


def _validate_weekend_scheduler_automation(
    path: Path,
    *,
    label: str,
    issues: list[str],
) -> None:
    if not path.exists():
        issues.append(f"{label} automation missing: {path}")
        return
    payload = _load_toml_payload(path.read_text())
    expected = EXPECTED_WEEKEND_AUTOMATIONS.get(label)
    if expected is None:
        issues.append(f"unsupported weekend automation label: {label}")
        return
    exact_checks = {
        "id": label,
        "kind": "cron",
        "name": expected["name"],
        "status": "ACTIVE",
        "rrule": expected["rrule"],
        "model": EXPECTED_WEEKEND_MODEL,
        "reasoning_effort": EXPECTED_WEEKEND_REASONING,
        "execution_environment": "local",
    }
    for key, wanted in exact_checks.items():
        actual = payload.get(key)
        if actual != wanted:
            issues.append(f"{label} automation {key} expected {wanted!r}, got {actual!r}")
    if payload.get("cwds") != [EXPECTED_WEEKEND_CWD]:
        issues.append(
            f"{label} automation cwds expected {[EXPECTED_WEEKEND_CWD]!r}, "
            f"got {payload.get('cwds')!r}"
        )
    target = payload.get("target")
    if not isinstance(target, dict) or target.get("project_id") != EXPECTED_WEEKEND_CWD:
        issues.append(f"{label} automation target project must be {EXPECTED_WEEKEND_CWD!r}")
    prompt = str(payload.get("prompt") or "")
    for phrase in (str(expected["command"]), str(expected["order"])):
        if phrase not in prompt:
            issues.append(f"{label} automation prompt missing safety contract: {phrase}")
    for phrase in EXPECTED_WEEKEND_SCHEDULER_REFRESH_PROMPT_PHRASES:
        if phrase not in prompt:
            issues.append(f"{label} automation prompt missing scheduler refresh contract: {phrase}")


def _load_toml_payload(text: str) -> dict[str, Any]:
    if tomllib is not None:
        return tomllib.loads(text)
    payload: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if value.startswith("[") and value.endswith("]"):
            payload[key] = [item.strip().strip('"') for item in value[1:-1].split(",") if item.strip()]
        elif value.startswith("{") and value.endswith("}"):
            # Python 3.10 has no stdlib tomllib.  Preserve the inline-table
            # shape used by Codex automation targets instead of degrading it
            # to an opaque string and falsely reporting a project mismatch.
            table: dict[str, str] = {}
            for item in value[1:-1].split(","):
                if "=" not in item:
                    continue
                table_key, table_value = item.split("=", 1)
                table[table_key.strip()] = table_value.strip().strip('"')
            payload[key] = table
        elif value.startswith('"') and value.endswith('"'):
            payload[key] = value[1:-1]
        else:
            payload[key] = value
    return payload


def _qr_trader_weekend_pause_active(status: Any, *, now_utc: datetime | None = None) -> bool:
    if str(status or "").upper() != "PAUSED":
        return False
    state = _load_json_payload(QR_WEEKEND_TASK_STATE_PATH)
    if state.get("mode") != "paused":
        return False
    if not _state_manages_qr_trader(state):
        return False
    clock = now_utc or datetime.now(timezone.utc)
    return _is_weekend_pause_window(clock.astimezone(timezone.utc))


def _load_json_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _state_manages_qr_trader(state: dict[str, Any]) -> bool:
    managed = state.get("managed_task_keys")
    if isinstance(managed, list) and "codex:qr-trader" in managed:
        return True
    tasks = state.get("tasks")
    if isinstance(tasks, dict) and "codex:qr-trader" in tasks:
        return True
    changes = state.get("last_changes")
    if isinstance(changes, list):
        return any(isinstance(item, dict) and item.get("key") == "codex:qr-trader" for item in changes)
    return False


def _is_weekend_pause_window(now_utc: datetime) -> bool:
    jst = now_utc.astimezone(timezone(timedelta(hours=9), "JST"))
    weekday = jst.weekday()
    local_minutes = jst.hour * 60 + jst.minute
    if weekday == 5:
        return local_minutes >= 6 * 60
    if weekday == 6:
        return True
    if weekday == 0:
        return local_minutes < 7 * 60
    return False


def unexplained_source_dirt(
    status_lines: tuple[str, ...] | list[str],
    explanations: dict[str, str],
) -> list[str]:
    dirty_paths = {_git_status_path(line) for line in status_lines}
    required = set(SOURCE_DIRT_EXPLANATION_REQUIRED_PATHS)
    return sorted(
        path
        for path in dirty_paths
        if path in required and not str(explanations.get(path) or "").strip()
    )


def _git_status_path(line: str) -> str:
    # `git status --short` is two status columns, a space, then the path. Rename
    # output keeps the destination after ` -> `.
    raw = str(line or "")[3:].strip()
    if " -> " in raw:
        raw = raw.rsplit(" -> ", 1)[-1]
    return raw


def main() -> int:
    issues = validate_trader_state.validate_contract()
    _require_text(
        AGENTS_PATH,
        (
            "docs/AGENT_CONTRACT.md",
            "Do not edit the stubs",
        ),
        issues,
    )
    _require_text(
        CHANGELOG_PATH,
        (
            "5% PACE BOARD",
            "ATTACK STACK",
            "10% EXTENSION GATE",
            "position_sizing.py",
            "guardian-action-cycle",
            "gpt-5.5 high every 60 minutes",
        ),
        issues,
    )
    _require_text(
        STRATEGY_MEMORY_PATH,
        (
            "5% PACE BOARD",
            "ATTACK STACK",
            "B/C",
            "10% Extension Gate",
            "position_sizing.py",
        ),
        issues,
    )
    _validate_qr_trader_automation(issues)
    _validate_weekend_scheduler_automation(
        QR_WEEKEND_MARKET_OFF_AUTOMATION_PATH,
        label="qr-weekend-market-off",
        issues=issues,
    )
    _validate_weekend_scheduler_automation(
        QR_WEEKEND_MARKET_ON_AUTOMATION_PATH,
        label="qr-weekend-market-on",
        issues=issues,
    )

    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        return 1
    print("task sync OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

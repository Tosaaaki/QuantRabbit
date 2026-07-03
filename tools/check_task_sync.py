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
QR_WEEKEND_TASK_STATE_PATH = Path.home() / ".codex" / "quant_rabbit_weekend_task_state.json"
EXPECTED_QR_TRADER_RRULE = "RRULE:FREQ=MINUTELY;INTERVAL=60;BYDAY=MO,TU,WE,TH,FR,SA"
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

    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        return 1
    print("task sync OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

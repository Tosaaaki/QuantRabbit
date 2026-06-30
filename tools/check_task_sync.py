#!/usr/bin/env python3
"""Check that the current task contract is synchronized across docs/tools."""

from __future__ import annotations

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
)


def _require_text(path: Path, needles: tuple[str, ...], issues: list[str]) -> None:
    text = path.read_text()
    for needle in needles:
        if needle not in text:
            issues.append(f"{path.relative_to(REPO_ROOT)} missing: {needle}")


def _validate_qr_trader_automation(issues: list[str]) -> None:
    if not QR_TRADER_AUTOMATION_PATH.exists():
        issues.append(f"qr-trader automation missing: {QR_TRADER_AUTOMATION_PATH}")
        return
    payload = _load_toml_payload(QR_TRADER_AUTOMATION_PATH.read_text())
    checks = {
        "rrule": EXPECTED_QR_TRADER_RRULE,
        "model": EXPECTED_QR_TRADER_MODEL,
        "reasoning_effort": EXPECTED_QR_TRADER_REASONING,
        "status": "ACTIVE",
    }
    for key, expected in checks.items():
        actual = payload.get(key)
        if actual != expected:
            issues.append(f"qr-trader automation {key} expected {expected!r}, got {actual!r}")
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

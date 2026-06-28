#!/usr/bin/env python3
"""Validate the FULL_TRADER session-state contract.

This checker is read-only. It verifies that the session-start tool and trader
runtime prompt both carry the required +5% path board and attack stack, so a
future trader cycle cannot silently drift back to B/C churn as a target path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import session_data


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_TRADER_PATH = REPO_ROOT / "docs" / "SKILL_trader.md"

REQUIRED_PATH_BOARD_LINES = (
    "## 5% PATH BOARD",
    "Remaining to +5%:",
    "Path A / HERO:",
    "Pair / side / vehicle:",
    "Expected pips:",
    "Suggested units:",
    "Expected contribution:",
    "Entry:",
    "TP:",
    "SL:",
    "Status: live / armable / blocked",
    "Exact blocker if blocked:",
    "Path B / SECOND SHOT:",
    "Status:",
    "Path C / NO HONEST PATH:",
    "Exact blocker:",
    "Next trigger:",
    "Shelf-life:",
)

REQUIRED_ATTACK_STACK_LINES = (
    "## ATTACK STACK",
    "Hero thesis:",
    "Why this thesis can still reach +5% today:",
    "NOW:",
    "Pair / side / vehicle:",
    "Entry:",
    "TP:",
    "SL:",
    "Units:",
    "Why now:",
    "RELOAD:",
    "Why this is better price, not hesitation:",
    "SECOND SHOT:",
    "Why this is same theme, different expression:",
    "If any slot is empty:",
    "Exact blocker:",
    "Next trigger:",
    "Shelf-life:",
)

REQUIRED_RULE_LINES = (
    "Under +5%, trader must name an A/S path or exact blocker.",
    "B/C trades cannot be the +5% target path.",
    "One distant pending order is not enough.",
    '"Trigger not printed yet" is an arm condition for LIMIT/STOP, not a dead thesis.',
    "The path must map to ATTACK STACK.",
)

REQUIRED_EXTENSION_GATE_LINES = (
    "## 10% EXTENSION GATE",
    "Default: NO",
    "YES only if:",
    "Progress is strong, ideally +3.5%+, or protected S/A winner can carry past +5%.",
    "Hero thesis still paying.",
    "3+ pairs confirm same currency theme, or hero pair has clean trend/band-walk.",
    "Spread stable.",
    "No major whipsaw event in next 30m.",
    "Last A/S trade green, protected, or structurally alive.",
    "Real reload/second-shot level exists, not chase.",
    "EXTEND mode requires A/S grade risk.",
    "After +5%, Extension Gate NO blocks fresh B risk.",
    "tools/position_sizing.py",
    "tools/place_trader_order.py",
)


def missing_lines(text: str, required: Iterable[str]) -> list[str]:
    return [line for line in required if line not in text]


def validate_contract() -> list[str]:
    issues: list[str] = []
    skill_text = SKILL_TRADER_PATH.read_text()
    session_text = "\n\n".join(
        (
            session_data.FIVE_PCT_PATH_BOARD_TEMPLATE,
            session_data.ATTACK_STACK_TEMPLATE,
            session_data.FIVE_PCT_PATH_RULES,
            session_data.EXTENSION_GATE_TEMPLATE,
        )
    )

    checks = (
        ("session_data path board", session_text, REQUIRED_PATH_BOARD_LINES),
        ("session_data attack stack", session_text, REQUIRED_ATTACK_STACK_LINES),
        ("session_data path rules", session_text, REQUIRED_RULE_LINES),
        ("session_data extension gate", session_text, REQUIRED_EXTENSION_GATE_LINES),
        ("docs/SKILL_trader.md path board", skill_text, REQUIRED_PATH_BOARD_LINES),
        ("docs/SKILL_trader.md attack stack", skill_text, REQUIRED_ATTACK_STACK_LINES),
        ("docs/SKILL_trader.md path rules", skill_text, REQUIRED_RULE_LINES),
        ("docs/SKILL_trader.md extension gate", skill_text, REQUIRED_EXTENSION_GATE_LINES),
    )
    for label, text, required in checks:
        missing = missing_lines(text, required)
        if missing:
            issues.append(f"{label} missing: {', '.join(missing)}")

    if "B/C churn" not in skill_text:
        issues.append("docs/SKILL_trader.md must explicitly reject B/C churn as target-path proof")
    if "remaining_to_5pct is above zero" not in skill_text:
        issues.append("docs/SKILL_trader.md must tie the board to under-+5% sessions")

    return issues


def main() -> int:
    issues = validate_contract()
    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        return 1
    print("trader state contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

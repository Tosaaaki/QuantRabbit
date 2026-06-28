#!/usr/bin/env python3
"""Check that the current task contract is synchronized across docs/tools."""

from __future__ import annotations

from pathlib import Path

import validate_trader_state


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_PATH = REPO_ROOT / "AGENTS.md"
CHANGELOG_PATH = REPO_ROOT / "docs" / "CHANGELOG.md"
STRATEGY_MEMORY_PATH = REPO_ROOT / "collab_trade" / "strategy_memory.md"


def _require_text(path: Path, needles: tuple[str, ...], issues: list[str]) -> None:
    text = path.read_text()
    for needle in needles:
        if needle not in text:
            issues.append(f"{path.relative_to(REPO_ROOT)} missing: {needle}")


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
            "5% PATH BOARD",
            "ATTACK STACK",
            "10% EXTENSION GATE",
            "position_sizing.py",
        ),
        issues,
    )
    _require_text(
        STRATEGY_MEMORY_PATH,
        (
            "5% PATH BOARD",
            "ATTACK STACK",
            "B/C",
            "10% Extension Gate",
            "position_sizing.py",
        ),
        issues,
    )

    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        return 1
    print("task sync OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

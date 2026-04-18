#!/usr/bin/env python3
"""Verify canonical prompts, Claude task assets, and Codex wrappers stay aligned."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()

TASKS = {
    "trader": "docs/SKILL_trader.md",
    "daily-review": "docs/SKILL_daily-review.md",
    "quality-audit": "docs/SKILL_quality-audit.md",
    "inventory-director": "docs/SKILL_inventory-director.md",
    "range-bot": "docs/SKILL_range-bot.md",
    "bot-trade-manager": "docs/SKILL_bot-trade-manager.md",
    "daily-performance-report": "docs/SKILL_daily-performance-report.md",
    "daily-slack-summary": "docs/SKILL_daily-slack-summary.md",
    "intraday-pl-update": "docs/SKILL_intraday-pl-update.md",
}


def find_claude_task_dir(task: str) -> Path | None:
    base = HOME / ".claude" / "scheduled-tasks"
    candidates = [
        base / task,
        base / f"{task}.DISABLED",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_frontmatter_description(path: Path) -> str | None:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        if stripped.startswith("description:"):
            return stripped.split(":", 1)[1].strip().strip('"')
    return None


def main() -> int:
    errors: list[str] = []
    for task, rel_path in TASKS.items():
        canonical = (ROOT / rel_path).resolve()
        claude_task_dir = find_claude_task_dir(task)
        claude_link = claude_task_dir / "SKILL.md" if claude_task_dir is not None else None
        codex_wrapper = ROOT / "docs" / "codex_automations" / f"{task}.md"
        canonical_description = read_frontmatter_description(canonical)

        if not canonical.exists():
            errors.append(f"missing canonical prompt: {canonical}")
            continue

        if claude_link is None or not claude_link.exists():
            errors.append(f"missing Claude compatibility link: {HOME / '.claude' / 'scheduled-tasks' / task / 'SKILL.md'}")
        elif claude_link.resolve() != canonical:
            errors.append(f"Claude link mismatch: {claude_link} -> {claude_link.resolve()} (expected {canonical})")
        else:
            legacy_ja = claude_task_dir / "SKILL_ja.md"
            if legacy_ja.exists():
                errors.append(f"remove deprecated Japanese prompt copy: {legacy_ja}")
            schedule_json = claude_task_dir / "schedule.json"
            if schedule_json.exists() and canonical_description is not None:
                try:
                    schedule = json.loads(schedule_json.read_text())
                except json.JSONDecodeError as exc:
                    errors.append(f"invalid Claude schedule metadata: {schedule_json} ({exc})")
                else:
                    actual_description = schedule.get("description")
                    if actual_description != canonical_description:
                        errors.append(
                            "Claude schedule description mismatch: "
                            f"{schedule_json} -> {actual_description!r} (expected {canonical_description!r})"
                        )

        if not codex_wrapper.exists():
            errors.append(f"missing Codex wrapper: {codex_wrapper}")
        else:
            text = codex_wrapper.read_text()
            if str(canonical) not in text:
                errors.append(f"Codex wrapper does not reference canonical prompt: {codex_wrapper}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("OK: task prompt sync points are aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

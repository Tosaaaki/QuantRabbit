#!/usr/bin/env python3
"""Auto-generate carry-forward hot updates from the current state handoff."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from record_s_hunt_ledger import STATE_PATH, _extract_section, build_entry
from state_hot_update import add_hot_update


def _clip(text: str, limit: int = 140) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _position_block(state_text: str, pair: str, direction: str) -> str:
    section = _extract_section(state_text, "Positions (Current)")
    if not section:
        return ""
    pattern = rf"^###\s+{re.escape(pair)}\s+{re.escape(direction)}\b([\s\S]*?)(?=^###\s+|\Z)"
    match = re.search(pattern, section, re.M)
    return match.group(1).strip() if match else ""


def _trigger_text(horizon: dict) -> str:
    trigger = " ".join(str(horizon.get("trigger") or "").split())
    if not trigger:
        if_not_live = " ".join(str(horizon.get("if_not_live") or "").split())
        match = re.search(r"exact trigger\s+(.+?)(?:\s*\|\s*invalidation|$)", if_not_live, re.I)
        if match:
            trigger = match.group(1).strip()
    trigger = re.sub(r"^\s*was\s+", "", trigger, flags=re.I)
    return trigger.strip(" .")


def _clean_reason(deployment_result: str) -> str:
    text = " ".join((deployment_result or "").split()).strip()
    match = re.search(r"dead thesis because\s+(.+)$", text, re.I)
    reason = match.group(1) if match else text
    reason = reason.strip(" .")
    reason = reason.replace("the ", "")
    reason = reason.replace("still has not", "did not")
    reason = reason.replace("before the close", "before close")
    reason = reason.replace("before rollover", "before rollover")
    reason = reason.replace("no market order is allowed now", "fresh execution was blocked")
    return _clip(reason, 110)


def _live_note(state_text: str, horizon: dict) -> str:
    pair = str(horizon.get("pair") or "").strip()
    direction = str(horizon.get("direction") or "").strip()
    block = _position_block(state_text, pair, direction).lower()
    trigger = _trigger_text(horizon)

    if "rollover" in block and (
        "hold-only" in block
        or "management-only" in block
        or "no fresh" in block
        or "rollover-only" in block
    ):
        return (
            f"{pair} {direction} | Live seat shifted into rollover hold-only management. "
            "Next seat: no fresh market order before spreads normalize."
        )

    if "would i entered now? no" in block or "would i? no" in block:
        return (
            f"{pair} {direction} | Filled seat is now management-only, not fresh execution. "
            "Next seat: add only on a new clean trigger, not at the current price."
        )

    if trigger:
        return (
            f"{pair} {direction} | Trigger converted into real deployment. "
            f"Next seat: respect {trigger} as the only honest add path."
        )

    return (
        f"{pair} {direction} | Trigger converted into real deployment. "
        "Next seat: add only if a new clean trigger prints."
    )


def _dead_note(horizon: dict) -> str:
    pair = str(horizon.get("pair") or "").strip()
    direction = str(horizon.get("direction") or "").strip()
    trigger = _trigger_text(horizon)
    reason = _clean_reason(str(horizon.get("deployment_result") or ""))
    trigger_lower = trigger.lower()

    if "reclaim" in trigger_lower:
        return (
            f"{pair} {direction} watch | Reclaim idea stayed trigger-only. "
            f"Next seat: require {trigger}, not a rewritten trigger."
        )
    if "rejection" in trigger_lower or "reject" in trigger_lower:
        return (
            f"{pair} {direction} watch | Rejection never printed cleanly. "
            f"Next seat: require {trigger}, not a pre-emptive fade."
        )
    if "break" in trigger_lower:
        return (
            f"{pair} {direction} watch | Break idea stayed trigger-only. "
            f"Next seat: require {trigger}, not a chase."
        )
    if trigger:
        return (
            f"{pair} {direction} watch | {reason}. "
            f"Next seat: require {trigger}, not a rewrite."
        )
    return (
        f"{pair} {direction} watch | {reason}. "
        "Next seat: do not upgrade it without a fresh trigger."
    )


def generate_hot_updates(state_path: Path = STATE_PATH) -> list[tuple[str, str]]:
    entry = build_entry(state_path)
    if entry is None:
        return []

    state_text = state_path.read_text()
    live_notes: list[tuple[str, str]] = []
    watch_notes: list[tuple[str, str]] = []

    for horizon in entry.get("horizons", []):
        pair = str(horizon.get("pair") or "").strip()
        direction = str(horizon.get("direction") or "").strip()
        if not pair or not direction:
            continue
        prefix = f"{pair} {direction}"
        deployment = str(horizon.get("deployment_result") or "").lower()
        if "entered id=" in deployment:
            live_notes.append((prefix, _live_note(state_text, horizon)))
            continue
        if "dead thesis because" in deployment:
            watch_notes.append((prefix, _dead_note(horizon)))

    ordered = live_notes[:1] + watch_notes[:2]

    unique: list[tuple[str, str]] = []
    seen = set()
    for prefix, note in ordered:
        key = (prefix, note)
        if key in seen:
            continue
        seen.add(key)
        unique.append((prefix, note))
    return unique


def apply_hot_updates(
    state_path: Path = STATE_PATH,
    *,
    limit: int = 5,
    dry_run: bool = False,
) -> list[str]:
    generated = generate_hot_updates(state_path)
    notes = [note for _, note in generated]
    if dry_run:
        return notes

    # add_hot_update prepends, so apply in reverse to preserve display order.
    for prefix, note in reversed(generated):
        add_hot_update(
            note,
            limit=limit,
            state_path=state_path,
            replace_prefix=prefix,
        )
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-generate Hot Updates from state.md receipts")
    parser.add_argument("--state", type=Path, default=STATE_PATH, help="Alternate state.md path")
    parser.add_argument("--limit", type=int, default=5, help="Keep only the latest N hot updates")
    parser.add_argument("--dry-run", action="store_true", help="Print generated notes without editing the file")
    args = parser.parse_args()

    state_path = Path(args.state).expanduser().resolve()
    notes = apply_hot_updates(state_path, limit=args.limit, dry_run=args.dry_run)
    if not notes:
        print("AUTO_HOT_UPDATES_SKIP no qualifying notes")
        return 0

    for note in notes:
        print(note)
    if not args.dry_run:
        print(f"AUTO_HOT_UPDATES_OK count={len(notes)} state={state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

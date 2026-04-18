#!/usr/bin/env python3
"""Append or refresh intraday hot updates inside collab_trade/state.md."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "collab_trade" / "state.md"
SECTION_HEADER = "## Hot Updates"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _find_hot_updates(lines: list[str]) -> tuple[int | None, int | None]:
    start = None
    end = None
    for idx, line in enumerate(lines):
        if line.strip() == SECTION_HEADER:
            start = idx
            continue
        if start is not None and idx > start and line.startswith("## "):
            end = idx
            break
    if start is not None and end is None:
        end = len(lines)
    return start, end


def _normalize_note(note: str) -> str:
    compact = " ".join(note.strip().split())
    return compact


def add_hot_update(
    note: str,
    limit: int = 5,
    *,
    state_path: Path = STATE_PATH,
    replace_prefix: str | None = None,
) -> int:
    if not state_path.exists():
        raise FileNotFoundError(f"{state_path} not found")

    lines = state_path.read_text().splitlines()
    start, end = _find_hot_updates(lines)
    bullet = f"- {_utc_stamp()} | {_normalize_note(note)}"

    if start is None:
        insert_at = next(
            (idx for idx, line in enumerate(lines) if line.strip().startswith("## Positions")),
            len(lines),
        )
        block = [SECTION_HEADER, bullet, ""]
        lines[insert_at:insert_at] = block
        state_path.write_text("\n".join(lines) + "\n")
        return 1

    existing = lines[start + 1:end]
    bullets = [line for line in existing if line.strip().startswith("- ")]
    non_bullets = [line for line in existing if not line.strip().startswith("- ")]

    note_key = note.strip()
    filtered = []
    for line in bullets:
        if replace_prefix and replace_prefix in line:
            continue
        if note_key in line:
            continue
        filtered.append(line)
    updated = [bullet] + filtered
    if limit > 0:
        updated = updated[:limit]

    replacement = updated
    if non_bullets or (end is not None and end < len(lines)):
        replacement.append("")
    lines[start + 1:end] = replacement
    state_path.write_text("\n".join(lines) + "\n")
    return len(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a hot update to collab_trade/state.md")
    sub = parser.add_subparsers(dest="cmd", required=True)

    add = sub.add_parser("add", help="Add one hot update bullet")
    add.add_argument("--note", required=True, help="Compressed next-seat correction")
    add.add_argument("--limit", type=int, default=5, help="Keep only the latest N bullets")
    add.add_argument("--state", type=Path, default=STATE_PATH, help="Alternate state.md path")
    add.add_argument(
        "--replace-prefix",
        default=None,
        help="Remove existing bullets containing this prefix before inserting the new one",
    )

    args = parser.parse_args()

    if args.cmd == "add":
        state_path = Path(args.state).expanduser().resolve()
        count = add_hot_update(
            args.note,
            limit=args.limit,
            state_path=state_path,
            replace_prefix=args.replace_prefix,
        )
        print(f"updated {state_path} hot updates ({count} kept)")


if __name__ == "__main__":
    main()

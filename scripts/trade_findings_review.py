#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINDINGS_PATH = REPO_ROOT / "docs" / "TRADE_FINDINGS.md"
ENTRY_HEADING_RE = re.compile(r"^#{2,3}\s+(\d{4}-\d{2}-\d{2}\b.*)$")
FIELD_RE = re.compile(r"^- ([^:]+):\s*(.*)$")


@dataclass
class Entry:
    heading: str
    fields: dict[str, str]
    raw_text: str
    order: int


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review TRADE_FINDINGS entries before profitability/risk changes."
    )
    parser.add_argument("--path", default=str(DEFAULT_FINDINGS_PATH), help="Path to TRADE_FINDINGS.md")
    parser.add_argument(
        "--query",
        default="",
        help="Comma or space separated strategy/hypothesis/close_reason terms to search.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of entries to print.")
    parser.add_argument("--chars", type=int, default=220, help="Maximum characters per printed field.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    return parser.parse_args(argv)


def _compact(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _tokenize_query(raw: str) -> list[str]:
    parts = [part.strip().lower() for part in raw.replace(",", " ").split()]
    return [part for part in parts if part]


def _parse_findings(path: Path) -> list[Entry]:
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: list[Entry] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    current_fields: dict[str, list[str]] = {}
    current_field: str | None = None

    def flush() -> None:
        nonlocal current_heading, current_lines, current_fields, current_field
        if current_heading is None:
            return
        rendered_fields = {
            key: "\n".join(value).strip()
            for key, value in current_fields.items()
            if "\n".join(value).strip()
        }
        entries.append(
            Entry(
                heading=current_heading,
                fields=rendered_fields,
                raw_text="\n".join(current_lines).strip(),
                order=len(entries),
            )
        )
        current_heading = None
        current_lines = []
        current_fields = {}
        current_field = None

    for line in lines:
        heading_match = ENTRY_HEADING_RE.match(line)
        if heading_match:
            flush()
            current_heading = heading_match.group(1).strip()
            current_lines = [line]
            current_fields = {}
            current_field = None
            continue
        if current_heading is None:
            continue
        current_lines.append(line)
        field_match = FIELD_RE.match(line)
        if field_match:
            current_field = field_match.group(1).strip()
            current_fields.setdefault(current_field, [])
            remainder = field_match.group(2).rstrip()
            if remainder:
                current_fields[current_field].append(remainder)
            continue
        if current_field is not None:
            current_fields[current_field].append(line.rstrip())

    flush()
    return entries


def _entry_field(entry: Entry, *names: str) -> str:
    for name in names:
        value = entry.fields.get(name)
        if value:
            return value
    return ""


def _score_entry(entry: Entry, tokens: list[str]) -> int:
    if not tokens:
        return 1
    haystack = f"{entry.heading}\n{entry.raw_text}".lower()
    return sum(1 for token in tokens if token in haystack)


def _select_entries(entries: list[Entry], tokens: list[str], limit: int) -> list[Entry]:
    scored: list[tuple[int, Entry]] = []
    for entry in entries:
        score = _score_entry(entry, tokens)
        if score <= 0:
            continue
        scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], item[1].order))
    return [entry for _, entry in scored[: max(0, limit)]]


def _build_output(source: Path, entries: list[Entry], query: str, max_chars: int) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for entry in entries:
        items.append(
            {
                "heading": entry.heading,
                "verdict": _compact(_entry_field(entry, "Verdict"), max_chars),
                "status": _compact(_entry_field(entry, "Status"), max_chars),
                "hypothesis_key": _compact(_entry_field(entry, "Hypothesis Key"), max_chars),
                "primary_loss_driver": _compact(_entry_field(entry, "Primary Loss Driver"), max_chars),
                "mechanism_fired": _compact(_entry_field(entry, "Mechanism Fired"), max_chars),
                "do_not_repeat_unless": _compact(
                    _entry_field(entry, "Do Not Repeat Unless"),
                    max_chars,
                ),
                "why": _compact(_entry_field(entry, "Why", "Why/Hypothesis"), max_chars),
                "hypothesis": _compact(_entry_field(entry, "Hypothesis", "Why/Hypothesis"), max_chars),
                "observed": _compact(
                    _entry_field(entry, "Observed/Fact", "Observed", "Fact"),
                    max_chars,
                ),
                "next_action": _compact(_entry_field(entry, "Next Action"), max_chars),
            }
        )
    return {
        "source": str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source),
        "query": query,
        "checklist": [
            "Confirm the same hypothesis key was not already tried with the same primary loss driver.",
            "Confirm whether the previous mechanism actually fired; fired=0 means it was not the dominant cause.",
            "If the dominant loss driver is unchanged, explain what is different before repeating the change.",
            "Write Hypothesis Key / Primary Loss Driver / Mechanism Fired / Do Not Repeat Unless in the next TRADE_FINDINGS entry.",
        ],
        "entries": items,
    }


def _print_human(payload: dict[str, Any]) -> None:
    print("TRADE_FINDINGS preflight")
    print(f"source: {payload['source']}")
    print(f"query: {payload['query'] or '(recent entries)'}")
    print("")
    print("Checklist:")
    for line in payload["checklist"]:
        print(f"- {line}")
    print("")
    print("Matches:")
    entries = payload.get("entries") or []
    if not entries:
        print("- no matching entries")
        return
    for idx, entry in enumerate(entries, start=1):
        print(f"{idx}. {entry['heading']}")
        for key in (
            "verdict",
            "status",
            "hypothesis_key",
            "primary_loss_driver",
            "mechanism_fired",
            "do_not_repeat_unless",
            "why",
            "hypothesis",
            "observed",
            "next_action",
        ):
            value = entry.get(key) or ""
            if value:
                print(f"   {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path = Path(args.path).resolve()
    if not path.exists():
        print(f"missing findings file: {path}", file=sys.stderr)
        return 1
    entries = _parse_findings(path)
    tokens = _tokenize_query(args.query)
    selected = _select_entries(entries, tokens, args.limit)
    payload = _build_output(path, selected, args.query, args.chars)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

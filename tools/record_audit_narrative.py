#!/usr/bin/env python3
"""Append narrative audit opportunities from quality_audit.md to audit_history.jsonl."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_MD = ROOT / "logs" / "quality_audit.md"
DEFAULT_HISTORY = ROOT / "logs" / "audit_history.jsonl"
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]

NEW_CONVICTION_RE = re.compile(
    r"Edge:\s*([SABC])\s*\|\s*Allocation:\s*([SABC])\s*\|\s*"
    r"(LONG|SHORT|WAIT)(?:\s*@([\d.]+))?(?:\s*TP=([\d.]+))?",
)
OLD_CONVICTION_RE = re.compile(
    r"Conviction:\s*([SABC])\s*\|\s*"
    r"(LONG|SHORT|WAIT)(?:\s*@([\d.]+))?(?:\s*TP=([\d.]+))?",
)
NARRATIVE_PICK_RE = re.compile(
    r"^-\s*(\w+_\w+)\s+(LONG|SHORT)\s*\|\s*Edge\s*([SABC])\s*"
    r"\|\s*Allocation\s*([SABC])\s*\|\s*Entry\s*@?([\d.]+)"
    r"(?:\s*\|\s*TP=?([\d.]+))?(?:\s*\|\s*Why:\s*(.+))?$",
    re.M,
)


def _rotate_history(path: Path) -> None:
    max_lines = 5000
    if path.exists() and path.stat().st_size > 2_000_000:
        lines = path.read_text().strip().splitlines()
        if len(lines) > max_lines:
            path.write_text("\n".join(lines[-max_lines:]) + "\n")


def _extract_section(text: str, heading: str) -> str:
    match = re.search(rf"^### {re.escape(heading)}\s*$", text, re.M)
    if not match:
        return ""
    remainder = text[match.end():]
    next_heading = re.search(r"^### ", remainder, re.M)
    end = match.end() + next_heading.start() if next_heading else len(text)
    return text[match.end():end].strip()


def _parse_timestamp(text: str) -> str:
    for pattern in (
        r"^## Auditor's View — ([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{5} UTC)",
        r"^# Quality Audit — ([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{5} UTC)",
    ):
        match = re.search(pattern, text, re.M)
        if match:
            return match.group(1)
    raise ValueError("Could not find audit timestamp in quality_audit.md")


def _parse_convictions(text: str) -> list[dict]:
    block = _extract_section(text, "7-Pair Conviction Map")
    if not block:
        return []

    sections: list[tuple[str, str]] = []
    starts = list(re.finditer(rf"^({'|'.join(PAIRS)}):\s*$", block, re.M))
    for idx, match in enumerate(starts):
        start = match.end()
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(block)
        sections.append((match.group(1), block[start:end]))

    convictions = []
    for pair, body in sections:
        match = NEW_CONVICTION_RE.search(body)
        if match:
            edge, allocation, direction, entry_price, tp_price = match.groups()
        else:
            legacy = OLD_CONVICTION_RE.search(body)
            if not legacy:
                continue
            edge, direction, entry_price, tp_price = legacy.groups()
            allocation = edge
        convictions.append(
            {
                "pair": pair,
                "direction": direction,
                "edge": edge,
                "allocation": allocation,
                "entry_price": float(entry_price) if entry_price else None,
                "tp_price": float(tp_price) if tp_price else None,
            }
        )
    return convictions


def _parse_narrative_picks(text: str) -> list[dict]:
    block = _extract_section(text, "Narrative Opportunities (not held by trader)")
    picks = []
    if block:
        for match in NARRATIVE_PICK_RE.finditer(block):
            pair, direction, edge, allocation, entry_price, tp_price, why = match.groups()
            picks.append(
                {
                    "pair": pair,
                    "direction": direction,
                    "edge": edge,
                    "allocation": allocation,
                    "entry_price": float(entry_price) if entry_price else None,
                    "tp_price": float(tp_price) if tp_price else None,
                    "why": (why or "").strip(),
                    "held_status": "NOT_HELD",
                }
            )
    return picks


def _parse_strongest_unheld(text: str) -> dict | None:
    patterns = (
        re.compile(
            r"Strongest NOT held by trader:\s*(\w+_\w+)\s+(LONG|SHORT)\s+—\s*"
            r"Edge\s*([SABC])\s*/\s*Allocation\s*([SABC])\s*because\s*(.+)",
        ),
        re.compile(
            r"Strongest NOT held by trader:\s*(\w+_\w+)\s+(LONG|SHORT)\s+—\s*"
            r"\[([SABC])(?:/([SABC]))?\]\s*because\s*(.+)",
        ),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        pair, direction, edge, allocation, why = match.groups()
        return {
            "pair": pair,
            "direction": direction,
            "edge": edge,
            "allocation": allocation or edge,
            "why": why.strip(),
            "held_status": "NOT_HELD",
        }
    return None


def build_entry(text: str) -> dict:
    convictions = _parse_convictions(text)
    narrative_picks = _parse_narrative_picks(text)
    strongest_unheld = _parse_strongest_unheld(text)

    if strongest_unheld:
        key = (strongest_unheld["pair"], strongest_unheld["direction"])
        if not any((pick["pair"], pick["direction"]) == key for pick in narrative_picks):
            narrative_picks.append(
                {
                    **strongest_unheld,
                    "entry_price": None,
                    "tp_price": None,
                }
            )

    return {
        "timestamp": _parse_timestamp(text),
        "source": "narrative",
        "convictions": convictions,
        "narrative_picks": narrative_picks,
        "strongest_unheld": strongest_unheld,
    }


def _already_recorded(path: Path, timestamp: str) -> bool:
    if not path.exists():
        return False
    for line in reversed(path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("source") != "narrative":
            continue
        return entry.get("timestamp") == timestamp
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-md", default=str(DEFAULT_AUDIT_MD))
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY))
    args = parser.parse_args()

    audit_md = Path(args.audit_md)
    history = Path(args.history_path)

    if not audit_md.exists():
        print("quality_audit.md not found", file=sys.stderr)
        return 1

    text = audit_md.read_text()
    try:
        entry = build_entry(text)
    except Exception as exc:
        print(f"PARSE_ERROR: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(entry, indent=2, ensure_ascii=False))
        return 0

    if not entry["convictions"] and not entry["narrative_picks"] and not entry["strongest_unheld"]:
        print("NO_NARRATIVE_OPPORTUNITIES_FOUND")
        return 0

    _rotate_history(history)
    if _already_recorded(history, entry["timestamp"]):
        print(f"NARRATIVE_ALREADY_RECORDED timestamp={entry['timestamp']}")
        return 0

    history.parent.mkdir(parents=True, exist_ok=True)
    with open(history, "a") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"RECORDED timestamp={entry['timestamp']} "
        f"convictions={len(entry['convictions'])} "
        f"narrative_picks={len(entry['narrative_picks'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

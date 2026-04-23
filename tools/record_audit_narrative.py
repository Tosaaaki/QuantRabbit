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
RANGE_PICK_RE = re.compile(
    r"^Best RANGE-(BUY|SELL):\s*(\w+_\w+)\s*@\s*([\d.]+)\s*->\s*TP\s*([\d.]+)\s*\(([^)]*)\)",
    re.M,
)
INVENTORY_LEAD_RE = re.compile(
    r"Inventory lead:\s*(\w+_\w+)\s+(LONG|SHORT)\s*\|\s*"
    r"Edge\s*([SABC])\s*\|\s*Allocation\s*([SABC])"
    r"(?:\s*\|\s*Entry\s*@?([\d.]+))?"
    r"(?:\s*\|\s*TP=?([\d.]+))?"
    r"(?:\s*\|\s*(?:Why|would upgrade if):\s*(.+))?",
    re.I,
)


def _rotate_history(path: Path) -> None:
    max_lines = 5000
    if path.exists() and path.stat().st_size > 2_000_000:
        lines = path.read_text().strip().splitlines()
        if len(lines) > max_lines:
            path.write_text("\n".join(lines[-max_lines:]) + "\n")


def _extract_section(text: str, heading: str) -> str:
    matches = list(re.finditer(rf"^### {re.escape(heading)}\s*$", text, re.M))
    if not matches:
        return ""
    match = matches[-1]
    remainder = text[match.end():]
    next_heading = re.search(r"^### ", remainder, re.M)
    end = match.end() + next_heading.start() if next_heading else len(text)
    return text[match.end():end].strip()


def _extract_first_section(text: str, headings: tuple[str, ...]) -> str:
    for heading in headings:
        block = _extract_section(text, heading)
        if block:
            return block
    return ""


def _parse_timestamp(text: str) -> str:
    for pattern in (
        r"^## Auditor's View — ([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{5} UTC)",
        r"^# Quality Audit — ([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{5} UTC)",
    ):
        matches = list(re.finditer(pattern, text, re.M))
        if matches:
            return matches[-1].group(1)
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
    block = _extract_first_section(
        text,
        (
            "Narrative Opportunities (not held by trader)",
            "Deployment Inventory (not held by trader)",
            "Narrative Opportunities / Deployment Inventory (not held by trader)",
        ),
    )
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


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("JPY") else 10000


def _parse_range_opportunities(text: str) -> list[dict]:
    block = _extract_first_section(
        text,
        (
            "Range Opportunities (actionable - trader reads this)",
            "Range Opportunities (actionable — trader reads this)",
        ),
    )
    if not block:
        return []

    lines = block.splitlines()
    picks: list[dict] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        match = RANGE_PICK_RE.match(line)
        if not match:
            idx += 1
            continue

        side, pair, entry_price, tp_price, note = match.groups()
        direction = "LONG" if side.upper() == "BUY" else "SHORT"
        visual = ""
        risk = ""
        idx += 1
        while idx < len(lines):
            detail = lines[idx].strip()
            if not detail:
                idx += 1
                continue
            if RANGE_PICK_RE.match(detail) or detail.startswith("No range trades:") or detail.startswith("### "):
                break
            if detail.lower().startswith("visual:"):
                visual = detail.split(":", 1)[1].strip()
            elif detail.lower().startswith("risk:"):
                risk = detail.split(":", 1)[1].strip()
            idx += 1

        entry = float(entry_price)
        tp = float(tp_price)
        target_pips = abs(tp - entry) * _pip_factor(pair)
        spread_multiple = None
        spread_match = re.search(r"([\d.]+)x spread", note, re.IGNORECASE)
        if spread_match:
            spread_multiple = float(spread_match.group(1))

        picks.append(
            {
                "pair": pair,
                "direction": direction,
                "entry_price": entry,
                "tp_price": tp,
                "target_pips": target_pips,
                "spread_multiple": spread_multiple,
                "range_side": side.upper(),
                "note": note.strip(),
                "visual": visual,
                "risk": risk,
            }
        )

    opposite_by_pair: dict[str, dict[str, float]] = {}
    for pick in picks:
        opposite_by_pair.setdefault(str(pick["pair"]), {})[str(pick["direction"])] = float(pick["entry_price"])
    for pick in picks:
        pair = str(pick["pair"])
        direction = str(pick["direction"])
        opposite_direction = "SHORT" if direction == "LONG" else "LONG"
        pick["opposite_entry_price"] = opposite_by_pair.get(pair, {}).get(opposite_direction)

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


def _parse_inventory_lead(text: str) -> dict | None:
    match = INVENTORY_LEAD_RE.search(text)
    if not match:
        return None
    pair, direction, edge, allocation, entry_price, tp_price, why = match.groups()
    return {
        "pair": pair,
        "direction": direction.upper(),
        "edge": edge.upper(),
        "allocation": allocation.upper(),
        "entry_price": float(entry_price) if entry_price else None,
        "tp_price": float(tp_price) if tp_price else None,
        "why": (why or "").strip(),
        "held_status": "NOT_HELD",
    }


def build_entry(text: str) -> dict:
    convictions = _parse_convictions(text)
    narrative_picks = _parse_narrative_picks(text)
    strongest_unheld = _parse_strongest_unheld(text)
    inventory_lead = _parse_inventory_lead(text)
    range_opportunities = _parse_range_opportunities(text)

    if not strongest_unheld and narrative_picks:
        rank = {"S": 4, "A": 3, "B": 2, "C": 1}
        strongest_unheld = max(
            narrative_picks,
            key=lambda item: (
                rank.get(str(item.get("edge") or ""), 0),
                rank.get(str(item.get("allocation") or ""), 0),
                float(item.get("entry_price") or 0.0),
            ),
        )

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

    if inventory_lead:
        key = (inventory_lead["pair"], inventory_lead["direction"])
        if not any((pick["pair"], pick["direction"]) == key for pick in narrative_picks):
            narrative_picks.append(inventory_lead)

    return {
        "timestamp": _parse_timestamp(text),
        "source": "narrative",
        "convictions": convictions,
        "narrative_picks": narrative_picks,
        "strongest_unheld": strongest_unheld,
        "inventory_lead": inventory_lead,
        "range_opportunities": range_opportunities,
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

    if (
        not entry["convictions"]
        and not entry["narrative_picks"]
        and not entry["strongest_unheld"]
        and not entry["inventory_lead"]
        and not entry["range_opportunities"]
    ):
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
        f"narrative_picks={len(entry['narrative_picks'])} "
        f"range_opportunities={len(entry['range_opportunities'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

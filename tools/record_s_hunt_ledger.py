#!/usr/bin/env python3
"""
Record the trader's S Hunt handoff into an append-only JSONL ledger.

This turns each session's short/medium/long horizon read into a machine-readable
artifact so daily_review can score:
  - what was deployed
  - what stayed armed but unfilled
  - what was passed
  - whether the pass was directionally correct

Usage:
  python3 tools/record_s_hunt_ledger.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "collab_trade" / "state.md"
LEDGER_PATH = ROOT / "logs" / "s_hunt_ledger.jsonl"
PAIR_RE = re.compile(r"\b([A-Z]{3}_[A-Z]{3})\s+(LONG|SHORT)\b")
STATE_UPDATED_RE = re.compile(r"^\*\*Last Updated\*\*:\s+(.+)$", re.M)
SECTION_RE = r"^## {name}\n(.*?)(?=^## |\Z)"
HORIZON_HEADERS = (
    "Short-term S",
    "Medium-term S",
    "Long-term S",
)
PODIUM_RE = re.compile(r"^Podium #(\d+):\s*(.+)$")


def _normalize_horizon_key(value: str | None) -> str | None:
    if not value:
        return None
    return value.replace(" S", "").strip()


def _extract_section(text: str, name: str) -> str:
    match = re.search(SECTION_RE.format(name=re.escape(name)), text, re.S | re.M)
    return match.group(1).strip() if match else ""


def _parse_state_updated(text: str) -> str | None:
    match = STATE_UPDATED_RE.search(text)
    return match.group(1).strip() if match else None


def _load_reference_price(pair: str | None) -> dict:
    if not pair:
        return {}

    technicals_path = ROOT / "logs" / f"technicals_{pair}.json"
    if not technicals_path.exists():
        return {}

    try:
        data = json.loads(technicals_path.read_text())
    except Exception:
        return {}

    timeframes = data.get("timeframes", {})
    ref = {}
    for tf in ("M1", "M5", "H1"):
        tf_data = timeframes.get(tf, {})
        close = tf_data.get("close")
        if close is not None:
            ref[f"{tf.lower()}_close"] = close
            ref[f"{tf.lower()}_timestamp"] = tf_data.get("timestamp")
    if "m1_close" in ref:
        ref["reference_price"] = ref["m1_close"]
    elif "m5_close" in ref:
        ref["reference_price"] = ref["m5_close"]
    return ref


def _split_horizon_blocks(section: str) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    current_name = None
    current_lines: list[str] = []
    for raw in section.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if any(stripped.startswith(f"{header} ") or stripped.startswith(f"{header}:") for header in HORIZON_HEADERS):
            if current_name:
                blocks.append((current_name, current_lines))
            current_name = next(header for header in HORIZON_HEADERS if stripped.startswith(header))
            current_lines = [stripped]
        elif current_name:
            current_lines.append(stripped)
    if current_name:
        blocks.append((current_name, current_lines))
    return blocks


def _extract_field(text: str, label: str, stop_labels: tuple[str, ...]) -> str | None:
    if stop_labels:
        stop_expr = "|".join(re.escape(x) for x in stop_labels)
        pattern = rf"{re.escape(label)}\s*(.*?)(?=(?:{stop_expr})|$)"
    else:
        pattern = rf"{re.escape(label)}\s*(.*)$"
    match = re.search(pattern, text)
    return match.group(1).strip(" .") if match else None


def _parse_horizon(name: str, lines: list[str]) -> dict:
    first = lines[0]
    rest = " ".join(lines).strip()
    after_colon = first.split(":", 1)[1].strip() if ":" in first else ""
    pair_match = PAIR_RE.search(rest)
    pair = pair_match.group(1) if pair_match else None
    direction = pair_match.group(2) if pair_match else None

    type_value = None
    if pair_match:
        tail = after_colon[pair_match.end() - (len(first.split(":", 1)[0]) + 1):].strip()
        tail = re.split(r"\.\s+| Why this is S| MTF chain:| Payout path:| Orderability:", tail)[0].strip(" .")
        if tail:
            type_value = tail

    text = rest
    why = _extract_field(
        text,
        "Why this is S on this horizon:",
        ("MTF chain:", "Payout path:", "Orderability:", "If not live:", "Deployment result:"),
    ) or _extract_field(
        text,
        "Why S here:",
        ("MTF chain:", "Payout path:", "Orderability:", "If not live:", "Deployment result:"),
    ) or _extract_field(
        text,
        "Why S:",
        ("MTF chain:", "Payout path:", "Orderability:", "If not live:", "Exact trigger:", "Invalidation:", "Deployment result:"),
    )
    mtf_chain = _extract_field(
        text,
        "MTF chain:",
        ("Payout path:", "Orderability:", "If not live:", "Deployment result:"),
    )
    payout_path = _extract_field(
        text,
        "Payout path:",
        ("Orderability:", "If not live:", "Deployment result:"),
    )
    orderability = _extract_field(
        text,
        "Orderability:",
        ("If not live:", "Exact trigger:", "Invalidation:", "Deployment result:"),
    )
    if_not_live = _extract_field(
        text,
        "If not live:",
        ("Deployment result:",),
    )
    if not if_not_live:
        exact_trigger = _extract_field(
            text,
            "Exact trigger:",
            ("Invalidation:", "Deployment result:"),
        )
        exact_invalidation = _extract_field(
            text,
            "Invalidation:",
            ("Deployment result:",),
        )
        if exact_trigger:
            if_not_live = f"exact trigger {exact_trigger}"
            if exact_invalidation:
                if_not_live += f" | invalidation {exact_invalidation}"
    deployment_result = _extract_field(text, "Deployment result:", tuple())

    trigger = None
    invalidation = None
    trigger_match = re.search(r"Trigger\s+(.+?)(?:\s*\|\s*invalidation\s+(.+)|\s*;\s*invalidation\s+(.+)|\s+invalidation\s+(.+)|$)", text)
    if trigger_match:
        trigger = trigger_match.group(1).strip(" .")
        invalidation = next((g.strip(" .") for g in trigger_match.groups()[1:] if g), None)
    elif if_not_live:
        trigger_match = re.search(r"exact trigger\s+(.+?)(?:\s*\|\s*invalidation\s+(.+)|$)", if_not_live)
        if trigger_match:
            trigger = trigger_match.group(1).strip(" .")
            if trigger_match.group(2):
                invalidation = trigger_match.group(2).strip(" .")
    if not trigger:
        trigger = _extract_field(
            text,
            "Exact trigger:",
            ("Invalidation:", "Deployment result:"),
        )
    if not invalidation:
        invalidation = _extract_field(
            text,
            "Invalidation:",
            ("Deployment result:",),
        )

    result = {
        "horizon": name,
        "pair": pair,
        "direction": direction,
        "type": type_value,
        "why": why,
        "mtf_chain": mtf_chain,
        "payout_path": payout_path,
        "orderability": orderability,
        "if_not_live": if_not_live,
        "trigger": trigger,
        "invalidation": invalidation,
        "deployment_result": deployment_result,
        "raw": text,
    }
    result.update(_load_reference_price(pair))
    return result


def _parse_horizon_deployment(section: str) -> dict:
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    mapping = {}
    for line in lines:
        if not line.startswith("Horizon deployment"):
            continue
        payload = line.split(":", 1)[1].strip()
        # Supports both one-line and future multiline receipts.
        for label in ("Short-term", "Medium-term", "Long-term"):
            match = re.search(rf"{label}\s*[: ]\s*(.*?)(?=(?:Short-term|Medium-term|Long-term)\s*[: ]|$)", payload)
            if match:
                mapping[label] = match.group(1).strip(" /")
    for line in lines:
        for label in ("Short-term", "Medium-term", "Long-term"):
            if line.startswith(f"{label}:"):
                mapping[label] = line.split(":", 1)[1].strip()
    return mapping


def _parse_simple_line(section: str, label: str) -> str | None:
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith(label):
            return stripped.split(":", 1)[1].strip() if ":" in stripped else stripped
    return None


def _clean_labeled_value(part: str, label: str) -> str | None:
    lowered = part.lower()
    prefix = label.lower()
    if lowered.startswith(prefix):
        return part[len(label):].strip(" :")
    return None


def _parse_s_excavation(section: str) -> dict:
    pair_rows: list[dict] = []
    podium_rows: list[dict] = []

    for raw in section.splitlines():
        line = raw.strip()
        if not line:
            continue

        podium_match = PODIUM_RE.match(line)
        if podium_match:
            rank = int(podium_match.group(1))
            payload = podium_match.group(2).strip()
            pair_match = PAIR_RE.search(payload)
            pair = pair_match.group(1) if pair_match else None
            direction = pair_match.group(2) if pair_match else None
            parts = [part.strip() for part in payload.split("|")]
            row = {
                "rank": rank,
                "pair": pair,
                "direction": direction,
                "closest_to_s_because": next((_clean_labeled_value(part, "Closest-to-S because") for part in parts if "Closest-to-S because" in part), None),
                "still_blocked_by": next((_clean_labeled_value(part, "Still blocked by") for part in parts if "Still blocked by" in part), None),
                "upgrade_action": next((_clean_labeled_value(part, "If it upgrades") for part in parts if "If it upgrades" in part), None),
                "raw": line,
            }
            row.update(_load_reference_price(pair))
            podium_rows.append(row)
            continue

        pair_match = re.match(r"^([A-Z]{3}_[A-Z]{3}):\s*(.+)$", line)
        if not pair_match:
            continue

        pair = pair_match.group(1)
        payload = pair_match.group(2).strip()
        parts = [part.strip() for part in payload.split("|")]
        row = {
            "pair": pair,
            "best_expression": next((_clean_labeled_value(part, "Best expression") for part in parts if "Best expression" in part), None),
            "why_not_s_now": next((_clean_labeled_value(part, "Why not S now") for part in parts if "Why not S now" in part), None),
            "upgrade_to_s_only_if": next((_clean_labeled_value(part, "Upgrade to S only if") for part in parts if "Upgrade to S only if" in part), None),
            "dead_if": next((_clean_labeled_value(part, "Dead if") for part in parts if "Dead if" in part), None),
            "raw": line,
        }
        row.update(_load_reference_price(pair))
        pair_rows.append(row)

    pair_rows.sort(key=lambda item: item.get("pair") or "")
    podium_rows.sort(key=lambda item: int(item.get("rank") or 99))
    return {"pairs": pair_rows, "podium": podium_rows}


def _load_existing_keys() -> set[str]:
    if not LEDGER_PATH.exists():
        return set()
    keys = set()
    for line in LEDGER_PATH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("state_last_updated"):
            keys.add(row["state_last_updated"])
    return keys


def build_entry(state_path: Path = STATE_PATH) -> dict | None:
    if not state_path.exists():
        return None

    text = state_path.read_text()
    state_last_updated = _parse_state_updated(text)
    if not state_last_updated:
        return None

    s_hunt_section = _extract_section(text, "S Hunt")
    capital_section = _extract_section(text, "Capital Deployment")
    action_section = _extract_section(text, "Action Tracking")
    market_section = _extract_section(text, "Market Narrative")
    deepening_section = _extract_section(text, "Deepening Pass")
    excavation_section = _extract_section(text, "S Excavation Matrix")

    horizons = [_parse_horizon(name, lines) for name, lines in _split_horizon_blocks(s_hunt_section)]
    horizon_deployment = _parse_horizon_deployment(capital_section)
    excavation = _parse_s_excavation(excavation_section)
    for horizon in horizons:
        deployment = horizon_deployment.get(_normalize_horizon_key(horizon["horizon"]))
        if deployment and not horizon.get("deployment_result"):
            horizon["deployment_result"] = deployment

    session_date = state_last_updated.split()[0]
    created_at = datetime.now(timezone.utc).isoformat()
    entry = {
        "created_at": created_at,
        "session_date": session_date,
        "state_last_updated": state_last_updated,
        "best_expression_now": _parse_simple_line(market_section, "Best expression NOW"),
        "primary_vehicle": _parse_simple_line(market_section, "Primary vehicle"),
        "best_direct_usd_seat": _parse_simple_line(market_section, "Best direct-USD seat NOW"),
        "next_fresh_risk_allowed": _parse_simple_line(market_section, "Next fresh risk allowed NOW"),
        "s_excavation_pairs": excavation.get("pairs", []),
        "s_excavation_podium": excavation.get("podium", []),
        "horizons": horizons,
        "horizon_deployment": horizon_deployment,
        "live_now": _parse_simple_line(capital_section, "LIVE NOW"),
        "reload": _parse_simple_line(capital_section, "RELOAD"),
        "second_shot": _parse_simple_line(capital_section, "SECOND SHOT / OTHER SIDE"),
        "flat_book_status": _parse_simple_line(capital_section, "Flat-book status"),
        "next_action_trigger": _parse_simple_line(action_section, "- Next action trigger"),
        "deepening_pass": " ".join(line.strip() for line in deepening_section.splitlines() if line.strip()),
    }
    return entry


def main() -> int:
    entry = build_entry()
    if entry is None:
        print("S_HUNT_LEDGER_SKIP no parseable state")
        return 0

    existing = _load_existing_keys()
    if entry["state_last_updated"] in existing:
        print(f"S_HUNT_LEDGER_SKIP duplicate state={entry['state_last_updated']}")
        return 0

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a") as fh:
        fh.write(json.dumps(entry, ensure_ascii=True) + "\n")
    print(
        f"S_HUNT_LEDGER_LOGGED state={entry['state_last_updated']} "
        f"horizons={len(entry['horizons'])} file={LEDGER_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

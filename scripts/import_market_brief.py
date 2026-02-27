#!/usr/bin/env python3
"""Import a human-written market brief into playbook input JSON files.

This parser targets short Japanese/English briefs that include:
- spot levels (USD/JPY, EUR/USD, AUD/JPY, EUR/JPY)
- DXY
- rates (US10Y / JP10Y)
- event table with JST times
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

JST = timezone(timedelta(hours=9))
UTC = timezone.utc


@dataclass(frozen=True)
class ParsedEvent:
    name: str
    when_jst: datetime
    impact: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_first_number(text: str, patterns: list[str]) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            return float(m.group(1))
        except Exception:
            continue
    return None


def _extract_brief_date_jst(text: str, fallback: Optional[str]) -> date:
    if fallback:
        return datetime.strptime(fallback, "%Y-%m-%d").date()
    m = re.search(r"(20\d{2})\s*[/-]\s*(\d{1,2})\s*[/-]\s*(\d{1,2})", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return datetime.now(JST).date()


def _clean_cell(cell: str) -> str:
    out = re.sub(r"\*+", "", cell)
    out = re.sub(r"\[[^\]]+\]\([^\)]+\)", "", out)
    return out.strip()


def _parse_jst_cell_to_datetime(cell: str, base_date: date) -> Optional[datetime]:
    raw = _clean_cell(cell)
    m_date = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})", raw)
    use_date = base_date
    if m_date:
        use_date = date(base_date.year, int(m_date.group(1)), int(m_date.group(2)))
    elif "翌" in raw:
        use_date = base_date + timedelta(days=1)

    m_time = re.search(r"(\d{1,2})\s*:\s*(\d{2})", raw)
    if not m_time:
        return None
    hh = int(m_time.group(1))
    mm = int(m_time.group(2))
    return datetime(use_date.year, use_date.month, use_date.day, hh, mm, tzinfo=JST)


def _guess_impact(name: str) -> str:
    token = name.lower()
    if any(k in token for k in ("ppi", "cpi", "nfp", "fomc", "pce", "boj", "ecb", "fed")):
        return "high"
    if any(k in token for k in ("pmi", "ism", "gdp", "employment")):
        return "medium"
    return "medium"


def _extract_events_from_table(text: str, base_date: date) -> list[ParsedEvent]:
    events: list[ParsedEvent] = []
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        if "イベント" in cells[0] and "東京時間" in line:
            continue
        if set(cells[0]) <= {":", "-"}:
            continue

        name = _clean_cell(cells[0])
        jst_cell = cells[2]
        when = _parse_jst_cell_to_datetime(jst_cell, base_date)
        if not when:
            continue
        if not name:
            continue
        events.append(ParsedEvent(name=name, when_jst=when, impact=_guess_impact(name)))
    return events


def _extract_events_from_lines(text: str, base_date: date) -> list[ParsedEvent]:
    events: list[ParsedEvent] = []
    for line in text.splitlines():
        if "JST" not in line and "東京時間" not in line:
            continue
        if ":" not in line:
            continue
        when = _parse_jst_cell_to_datetime(line, base_date)
        if not when:
            continue
        cleaned = _clean_cell(line)
        cleaned = re.sub(r"\d{1,2}:\d{2}", "", cleaned).strip(" -:|")
        if len(cleaned) < 3:
            cleaned = "event"
        events.append(ParsedEvent(name=cleaned, when_jst=when, impact=_guess_impact(cleaned)))
    return events


def _dedupe_events(events: list[ParsedEvent]) -> list[ParsedEvent]:
    seen: set[tuple[str, str]] = set()
    out: list[ParsedEvent] = []
    for ev in sorted(events, key=lambda x: x.when_jst):
        key = (ev.name.lower(), ev.when_jst.isoformat())
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def _build_external_snapshot(text: str) -> dict:
    usdjpy = _find_first_number(text, [r"USD\s*/\s*JPY\s*[：:]\s*([0-9]+(?:\.[0-9]+)?)"])
    eurusd = _find_first_number(text, [r"EUR\s*/\s*USD\s*[：:]\s*([0-9]+(?:\.[0-9]+)?)"])
    audjpy = _find_first_number(text, [r"AUD\s*/\s*JPY\s*[：:]\s*([0-9]+(?:\.[0-9]+)?)"])
    eurjpy = _find_first_number(text, [r"EUR\s*/\s*JPY\s*[：:]\s*([0-9]+(?:\.[0-9]+)?)"])
    dxy = _find_first_number(text, [r"DXY\s*[：:]\s*([0-9]+(?:\.[0-9]+)?)", r"ドル指数\s*DXY\s*[：:]\s*([0-9]+(?:\.[0-9]+)?)"])
    us10y = _find_first_number(text, [r"米10年\s*([0-9]+(?:\.[0-9]+)?)"])
    jp10y = _find_first_number(text, [r"日10年\s*([0-9]+(?:\.[0-9]+)?)"])

    pairs: dict[str, dict[str, float]] = {}
    if eurusd is not None:
        pairs["EUR_USD"] = {"price": round(eurusd, 6)}
    if audjpy is not None:
        pairs["AUD_JPY"] = {"price": round(audjpy, 6)}
    if eurjpy is not None:
        pairs["EUR_JPY"] = {"price": round(eurjpy, 6)}
    if usdjpy is not None:
        pairs["USD_JPY"] = {"price": round(usdjpy, 6)}

    out: dict = {"pairs": pairs}
    if dxy is not None:
        out["dxy"] = round(dxy, 6)
    rates: dict[str, float] = {}
    if us10y is not None:
        rates["US10Y"] = round(us10y, 6)
    if jp10y is not None:
        rates["JP10Y"] = round(jp10y, 6)
    if rates:
        out["rates"] = rates
    return out


def _build_events_payload(events: list[ParsedEvent]) -> dict:
    rows = []
    for ev in events:
        rows.append(
            {
                "name": ev.name,
                "impact": ev.impact,
                "time_jst": ev.when_jst.strftime("%Y-%m-%d %H:%M"),
                "time_utc": ev.when_jst.astimezone(UTC).replace(microsecond=0).isoformat(),
            }
        )
    return {"events": rows}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Import market brief markdown/text to playbook input JSON")
    ap.add_argument("--input", required=True, help="Path to market brief text/markdown")
    ap.add_argument(
        "--external-output",
        default="logs/market_external_snapshot.json",
        help="Output JSON path for market external snapshot",
    )
    ap.add_argument(
        "--events-output",
        default="logs/market_events.json",
        help="Output JSON path for events",
    )
    ap.add_argument(
        "--date-jst",
        default=None,
        help="Override base date in JST (YYYY-MM-DD). If omitted, infer from text.",
    )
    args = ap.parse_args()

    text = _read_text(Path(args.input))
    base_date = _extract_brief_date_jst(text, args.date_jst)

    snapshot = _build_external_snapshot(text)
    snapshot.setdefault("notes", {})
    snapshot["notes"].update(
        {
            "source": "market_brief_import",
            "ingested_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "base_date_jst": base_date.isoformat(),
        }
    )

    events = _extract_events_from_table(text, base_date)
    if not events:
        events = _extract_events_from_lines(text, base_date)
    events = _dedupe_events(events)
    events_payload = _build_events_payload(events)
    events_payload["notes"] = {
        "source": "market_brief_import",
        "ingested_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "base_date_jst": base_date.isoformat(),
        "event_count": len(events),
    }

    _write_json(Path(args.external_output), snapshot)
    _write_json(Path(args.events_output), events_payload)

    print(f"external_snapshot: {args.external_output}")
    print(f"events: {args.events_output}")
    print(f"event_count: {len(events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

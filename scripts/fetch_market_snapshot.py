#!/usr/bin/env python3
"""Fetch market snapshot (prices/rates/events) for deterministic ops playbook.

Outputs:
- external snapshot JSON (pairs, dxy, rates)
- events JSON (UTC/JST timestamps + impact)

Data sources:
- Stooq CSV endpoint (FX + DXY futures)
- TradingEconomics pages (US10Y/JP10Y + calendar HTML)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup

UTC = timezone.utc
JST = timezone(timedelta(hours=9))

STOOQ_CSV = "https://stooq.com/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv"
TE_US10Y_URL = "https://tradingeconomics.com/united-states/government-bond-yield"
TE_JP10Y_URL = "https://tradingeconomics.com/japan/government-bond-yield"
TE_CALENDAR_URL = "https://tradingeconomics.com/calendar"

SYMBOL_MAP = {
    "USD_JPY": "usdjpy",
    "EUR_USD": "eurusd",
    "AUD_JPY": "audjpy",
    "EUR_JPY": "eurjpy",
    "DXY": "dx.f",
}

COUNTRY_ALLOWLIST_DEFAULT = {
    "united states",
    "japan",
    "euro area",
    "germany",
    "united kingdom",
    "australia",
    "china",
}


@dataclass(frozen=True)
class EventRow:
    name: str
    impact: str
    time_utc: datetime


class FetchError(RuntimeError):
    pass


def _fetch_text(url: str, *, timeout: float = 15.0) -> str:
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (QuantRabbit/ops-playbook)",
                "Accept": "text/html,application/json,text/csv,*/*",
            },
        )
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        raise FetchError(f"fetch failed url={url} err={exc}") from exc


def _parse_stooq_row(csv_text: str) -> dict[str, Any]:
    reader = csv.DictReader(StringIO(csv_text))
    rows = list(reader)
    if not rows:
        raise FetchError("empty stooq csv")
    row = rows[0]

    def f(key: str) -> Optional[float]:
        raw = (row.get(key) or "").strip()
        if raw in {"", "N/D"}:
            return None
        try:
            return float(raw)
        except Exception:
            return None

    out = {
        "symbol": (row.get("Symbol") or "").strip(),
        "date": (row.get("Date") or "").strip() or None,
        "time": (row.get("Time") or "").strip() or None,
        "open": f("Open"),
        "high": f("High"),
        "low": f("Low"),
        "close": f("Close"),
    }
    return out


def _fetch_stooq_symbol(symbol: str) -> dict[str, Any]:
    csv_text = _fetch_text(STOOQ_CSV.format(symbol=symbol), timeout=12.0)
    return _parse_stooq_row(csv_text)


def _calc_change_pct(open_price: Optional[float], close_price: Optional[float]) -> Optional[float]:
    if open_price is None or close_price is None:
        return None
    if abs(open_price) < 1e-12:
        return None
    return (close_price / open_price - 1.0) * 100.0


def _extract_te_bond_value(html: str) -> Optional[float]:
    # Prefer TEChartsMeta numeric value.
    m = re.search(r'TEChartsMeta\s*=\s*\[\{[^\}]*"value":([0-9.]+)', html)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    # Fallback to meta description pattern.
    m2 = re.search(r"yield[^%]*to\s+([0-9]+(?:\.[0-9]+)?)%", html, flags=re.IGNORECASE)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            pass
    return None


def _parse_event_time_utc(day_text: str, time_text: str) -> Optional[datetime]:
    day_text = day_text.strip()
    time_text = " ".join(time_text.strip().split())
    if not day_text or not time_text:
        return None
    try:
        dt = datetime.strptime(f"{day_text} {time_text}", "%Y-%m-%d %I:%M %p")
    except ValueError:
        return None
    return dt.replace(tzinfo=UTC)


def _impact_from_span_classes(classes: list[str]) -> str:
    joined = " ".join(classes)
    if "calendar-date-3" in joined:
        return "high"
    if "calendar-date-2" in joined:
        return "medium"
    return "low"


def _extract_calendar_events(
    html: str,
    *,
    countries: set[str],
    now_utc: datetime,
    before_min: int,
    after_min: int,
) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr[data-url]")
    out: list[EventRow] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        country = str(row.get("data-country") or "").strip().lower()
        if countries and country not in countries:
            continue

        tds = row.find_all("td")
        if not tds:
            continue

        # First td contains date-class + time span.
        t0 = tds[0]
        day_text = None
        for cls in t0.get("class") or []:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(cls).strip()):
                day_text = str(cls).strip()
                break
        if not day_text:
            continue

        span = t0.select_one("span[class*=calendar-date]")
        if span is None:
            continue
        time_text = span.get_text(" ", strip=True)
        dt_utc = _parse_event_time_utc(day_text, time_text)
        if dt_utc is None:
            continue

        minutes = int(round((dt_utc - now_utc).total_seconds() / 60.0))
        if minutes < -abs(before_min) or minutes > abs(after_min):
            continue

        anchor = row.select_one("a.calendar-event")
        if anchor is None:
            continue
        name = " ".join(anchor.get_text(" ", strip=True).split())
        if not name:
            continue

        impact = _impact_from_span_classes(list(span.get("class") or []))
        key = (name.lower(), dt_utc.isoformat())
        if key in seen:
            continue
        seen.add(key)
        out.append(EventRow(name=name, impact=impact, time_utc=dt_utc))

    out.sort(key=lambda r: r.time_utc)
    return [
        {
            "name": row.name,
            "impact": row.impact,
            "time_utc": row.time_utc.replace(microsecond=0).isoformat(),
            "time_jst": row.time_utc.astimezone(JST).strftime("%Y-%m-%d %H:%M JST"),
            "minutes_to_event": int(round((row.time_utc - now_utc).total_seconds() / 60.0)),
        }
        for row in out
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_snapshot(now_utc: datetime) -> tuple[dict[str, Any], dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for pair, symbol in SYMBOL_MAP.items():
        values[pair] = _fetch_stooq_symbol(symbol)

    us10y_html = _fetch_text(TE_US10Y_URL, timeout=15.0)
    jp10y_html = _fetch_text(TE_JP10Y_URL, timeout=15.0)
    us10y = _extract_te_bond_value(us10y_html)
    jp10y = _extract_te_bond_value(jp10y_html)

    pairs_payload: dict[str, dict[str, float]] = {}
    for pair in ("USD_JPY", "EUR_USD", "AUD_JPY", "EUR_JPY"):
        row = values.get(pair) or {}
        close = row.get("close")
        open_price = row.get("open")
        if isinstance(close, (int, float)):
            item: dict[str, float] = {"price": round(float(close), 6)}
            change = _calc_change_pct(
                float(open_price) if isinstance(open_price, (int, float)) else None,
                float(close),
            )
            if change is not None:
                item["change_pct_24h"] = round(change, 6)
            pairs_payload[pair] = item

    dxy_close = values.get("DXY", {}).get("close")
    dxy_open = values.get("DXY", {}).get("open")
    dxy_change = _calc_change_pct(
        float(dxy_open) if isinstance(dxy_open, (int, float)) else None,
        float(dxy_close) if isinstance(dxy_close, (int, float)) else None,
    )

    external: dict[str, Any] = {
        "pairs": pairs_payload,
        "notes": {
            "source": "stooq+tradingeconomics",
            "generated_at": now_utc.replace(microsecond=0).isoformat(),
            "sources": {
                "stooq": "https://stooq.com",
                "tradingeconomics_us10y": TE_US10Y_URL,
                "tradingeconomics_jp10y": TE_JP10Y_URL,
            },
        },
    }
    if isinstance(dxy_close, (int, float)):
        external["dxy"] = round(float(dxy_close), 6)
    if dxy_change is not None:
        external["dxy_change_pct_24h"] = round(dxy_change, 6)
    rates: dict[str, float] = {}
    if isinstance(us10y, (int, float)):
        rates["US10Y"] = round(float(us10y), 6)
    if isinstance(jp10y, (int, float)):
        rates["JP10Y"] = round(float(jp10y), 6)
    if rates:
        external["rates"] = rates

    calendar_html = _fetch_text(TE_CALENDAR_URL, timeout=20.0)
    events = _extract_calendar_events(
        calendar_html,
        countries=COUNTRY_ALLOWLIST_DEFAULT,
        now_utc=now_utc,
        before_min=60,
        after_min=24 * 60,
    )
    events_payload = {
        "events": events,
        "notes": {
            "source": TE_CALENDAR_URL,
            "timezone": "UTC",
            "generated_at": now_utc.replace(microsecond=0).isoformat(),
            "country_allowlist": sorted(COUNTRY_ALLOWLIST_DEFAULT),
            "window_minutes": {"before": 60, "after": 1440},
        },
    }
    return external, events_payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch market snapshot for ops playbook")
    ap.add_argument(
        "--external-output",
        default="logs/market_external_snapshot.json",
        help="Path to write external snapshot JSON",
    )
    ap.add_argument(
        "--events-output",
        default="logs/market_events.json",
        help="Path to write events JSON",
    )
    args = ap.parse_args()

    now_utc = datetime.now(UTC)
    external, events = build_snapshot(now_utc)
    _write_json(Path(args.external_output), external)
    _write_json(Path(args.events_output), events)

    print(f"external_snapshot: {args.external_output}")
    print(f"events: {args.events_output}")
    print(f"events_count: {len(events.get('events') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

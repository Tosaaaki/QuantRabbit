#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

DEFAULT_FEEDS = (
    ("fed_press_all", "https://www.federalreserve.gov/feeds/press_all.xml", "fed"),
    (
        "fed_press_monetary",
        "https://www.federalreserve.gov/feeds/press_monetary.xml",
        "fed",
    ),
    ("boj_whatsnew", "https://www.boj.or.jp/en/rss/whatsnew.xml", "boj"),
    ("boj_statistics", "https://www.boj.or.jp/en/rss/statistics.xml", "boj"),
)
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (QuantRabbit; +https://github.com/Tosaaaki/QuantRabbit)",
    "Accept": "application/rss+xml, application/xml, text/xml; q=0.9, */*; q=0.8",
    "Accept-Language": "en-US,en; q=0.9",
    "Cache-Control": "no-cache",
}

CRITICAL_KEYWORDS = (
    "statement",
    "minutes",
    "monetary policy",
    "policy board",
    "interest rate",
    "rate decision",
    "outlook report",
    "fomc",
)
MEDIUM_KEYWORDS = (
    "inflation",
    "employment",
    "cpi",
    "ppi",
    "gdp",
    "speech",
    "press release",
    "statistics",
)
USD_HAWKISH = ("raise", "inflation", "tighten", "higher for longer", "balance sheet")
USD_DOVISH = ("cut", "easing", "stimulus", "accommodative", "slowdown")
JPY_HAWKISH = ("raise", "tighten", "normalization", "yield", "hike")
JPY_DOVISH = ("easing", "accommodative", "stimulus", "purchase", "support")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        tmp_path = Path(fh.name)
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def _build_request(url: str) -> urllib.request.Request:
    headers = dict(DEFAULT_HEADERS)
    lowered = url.lower()
    if "federalreserve.gov" in lowered:
        headers["Referer"] = (
            "https://www.federalreserve.gov/newsevents/pressreleases.htm"
        )
    elif "boj.or.jp" in lowered:
        headers["Referer"] = "https://www.boj.or.jp/en/"
    return urllib.request.Request(url, headers=headers)


def _fetch_bytes(
    url: str,
    *,
    timeout_sec: float,
    retries: int = 2,
    retry_delay_sec: float = 0.4,
) -> bytes:
    attempts = max(1, int(retries) + 1)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(
                _build_request(url), timeout=timeout_sec
            ) as resp:
                return resp.read()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(max(0.0, float(retry_delay_sec)) * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"failed to fetch {url}")


def _first_text(node: ET.Element, *paths: str) -> str:
    for path in paths:
        found = node.find(path)
        if found is not None and found.text and found.text.strip():
            return found.text.strip()
    return ""


def _parse_published_at(node: ET.Element) -> str:
    raw = _first_text(
        node, "pubDate", "updated", "published", "{http://www.w3.org/2005/Atom}updated"
    )
    if not raw:
        return ""
    try:
        parsed = parsedate_to_datetime(raw)
    except Exception:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _source_keywords(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for token in CRITICAL_KEYWORDS + MEDIUM_KEYWORDS:
        if token in lowered and token not in found:
            found.append(token)
    return found


def _headline_severity(text: str, *, age_hours: float) -> tuple[int, str]:
    lowered = text.lower()
    if any(token in lowered for token in CRITICAL_KEYWORDS) and age_hours <= 48.0:
        return 3, "high"
    if any(token in lowered for token in MEDIUM_KEYWORDS) and age_hours <= 36.0:
        return 2, "medium"
    if age_hours <= 24.0:
        return 1, "low"
    return 0, "none"


def _bias_score(source_kind: str, text: str) -> int:
    lowered = text.lower()
    score = 0
    if source_kind == "fed":
        if any(token in lowered for token in USD_HAWKISH):
            score += 1
        if any(token in lowered for token in USD_DOVISH):
            score -= 1
    elif source_kind == "boj":
        if any(token in lowered for token in JPY_HAWKISH):
            score -= 1
        if any(token in lowered for token in JPY_DOVISH):
            score += 1
    return score


def _iter_items(xml_bytes: bytes) -> list[ET.Element]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    items = root.findall(".//item")
    if items:
        return items
    return root.findall(".//{http://www.w3.org/2005/Atom}entry")


def build_report(
    *,
    lookback_hours: int,
    timeout_sec: float,
    per_feed_limit: int,
) -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=max(1, int(lookback_hours)))
    source_rows: list[dict[str, Any]] = []
    headlines: list[dict[str, Any]] = []
    bias_score = 0
    severity_value = 0

    for source_name, url, source_kind in DEFAULT_FEEDS:
        source_row = {
            "name": source_name,
            "url": url,
            "kind": source_kind,
            "status": "ok",
            "item_count": 0,
        }
        try:
            payload = _fetch_bytes(url, timeout_sec=float(timeout_sec))
            items = _iter_items(payload)
        except Exception as exc:
            source_row["status"] = "error"
            source_row["error"] = str(exc)
            source_rows.append(source_row)
            continue

        used = 0
        for item in items:
            published_at = _parse_published_at(item)
            if published_at:
                published_dt = datetime.fromisoformat(
                    published_at.replace("Z", "+00:00")
                ).astimezone(timezone.utc)
                if published_dt < cutoff:
                    continue
                age_hours = max(0.0, (now_utc - published_dt).total_seconds() / 3600.0)
            else:
                age_hours = float(lookback_hours)
            title = _first_text(item, "title", "{http://www.w3.org/2005/Atom}title")
            link = _first_text(item, "link", "{http://www.w3.org/2005/Atom}link")
            if not title:
                continue
            sev_value, sev_label = _headline_severity(title, age_hours=age_hours)
            bias_score += _bias_score(source_kind, title)
            severity_value = max(severity_value, sev_value)
            headlines.append(
                {
                    "source": source_name,
                    "kind": source_kind,
                    "title": title,
                    "link": link,
                    "published_at": published_at,
                    "age_hours": round(age_hours, 2),
                    "severity": sev_label,
                    "keywords": _source_keywords(title),
                }
            )
            used += 1
            if used >= int(per_feed_limit):
                break
        source_row["item_count"] = used
        source_rows.append(source_row)

    headlines.sort(
        key=lambda item: (
            {"high": 3, "medium": 2, "low": 1, "none": 0}.get(
                str(item.get("severity") or "none"), 0
            ),
            -float(item.get("age_hours") or 9999.0),
            str(item.get("title") or ""),
        ),
        reverse=True,
    )
    severity_label = {0: "none", 1: "low", 2: "medium", 3: "high"}.get(
        severity_value, "none"
    )
    caution_window_active = bool(severity_value >= 2 and headlines)
    if bias_score >= 1:
        usd_jpy_bias = "up"
    elif bias_score <= -1:
        usd_jpy_bias = "down"
    else:
        usd_jpy_bias = "neutral"
    return {
        "generated_at": now_utc.isoformat(),
        "lookback_hours": int(lookback_hours),
        "feed_count": len(DEFAULT_FEEDS),
        "source_error_count": sum(
            1 for row in source_rows if row.get("status") == "error"
        ),
        "event_severity": severity_label,
        "caution_window_active": caution_window_active,
        "usd_jpy_bias": usd_jpy_bias,
        "bias_score": int(bias_score),
        "sources": source_rows,
        "headlines": headlines[:12],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fetch slow macro/news context from official feeds"
    )
    ap.add_argument("--output", default="logs/macro_news_context.json")
    ap.add_argument("--history", default="logs/macro_news_context_history.jsonl")
    ap.add_argument("--lookback-hours", type=int, default=72)
    ap.add_argument("--timeout-sec", type=float, default=5.0)
    ap.add_argument("--per-feed-limit", type=int, default=3)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_report(
        lookback_hours=max(1, int(args.lookback_hours)),
        timeout_sec=max(1.0, float(args.timeout_sec)),
        per_feed_limit=max(1, int(args.per_feed_limit)),
    )
    _write_json_atomic(Path(args.output).resolve(), payload)
    _append_jsonl(Path(args.history).resolve(), payload)
    print(
        f"[macro-news-context-worker] wrote {args.output} "
        f"severity={payload['event_severity']} headlines={len(payload['headlines'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

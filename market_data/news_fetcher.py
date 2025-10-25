"""
market_data.news_fetcher
~~~~~~~~~~~~~~~~~~~~~~~~
OANDA Labs の経済指標カレンダー (約 5 分キャッシュ) からイベントを取得し、
Cloud Storage `fx-news/raw/*.json` へアップロードする。OANDA から取得できない
場合は従来の RSS をフォールバックとして利用する。
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import pathlib
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional

import feedparser
import requests
import toml
from google.cloud import storage

# --- 設定・定数 ---

FETCH_LIMIT = int(os.environ.get("NEWS_FETCH_MAX", "10"))
FETCH_INTERVAL_SECONDS = int(os.environ.get("NEWS_FETCH_INTERVAL_SEC", "60"))
OANDA_TIMEOUT = float(os.environ.get("OANDA_NEWS_TIMEOUT_SEC", "8.0"))
ENABLE_RSS_FALLBACK = os.environ.get("NEWS_ENABLE_RSS_FALLBACK", "false").lower() == "true"

# 環境変数や config から設定を読み込む
try:
    cfg = None
    cfg_path = pathlib.Path("config/env.local.toml")
    if cfg_path.exists():
        cfg = toml.load(cfg_path.open("r"))
    else:
        toml_path = pathlib.Path("config/env.toml")
        if toml_path.exists():
            cfg = toml.load(toml_path.open("r"))
except Exception:
    cfg = None

def _cfg_lookup(*keys: str) -> Optional[str]:
    if not cfg:
        return None
    cur = cfg
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur  # type: ignore[return-value]

BUCKET = _cfg_lookup("gcp", "bucket_news") or os.environ.get("BUCKET") or os.environ.get("BUCKET_NEWS") or "fx-news"
OANDA_TOKEN = (
    _cfg_lookup("oanda_labs_token")
    or _cfg_lookup("oanda_token")
    or os.environ.get("OANDA_LABS_TOKEN")
    or os.environ.get("OANDA_TOKEN")
)
practice_raw = _cfg_lookup("oanda_practice") or os.environ.get("OANDA_PRACTICE", "true")
OANDA_PRACTICE = str(practice_raw).lower() != "false"
OANDA_CALENDAR_URL = "https://api-fxpractice.oanda.com/v1/calendar" if OANDA_PRACTICE else "https://api-fxtrade.oanda.com/v1/calendar"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "QuantRabbitNewsFetcher/2025.10",
        "Accept": "application/json",
        "X-Accept-Datetime-Format": "RFC3339",
    }
)

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
FF_NEXT = "https://nfs.faireconomy.media/ff_calendar_nextweek.xml"
DAILYFX = "https://www.dailyfx.com/feeds/most-recent-stories"

HIGH_WORDS = re.compile("|".join(["BoJ", "FOMC", "Non-Farm", "CPI", "政策金利", "雇用統計"]))

IMPACT_MAP = {"high": 3, "medium": 2, "med": 2, "low": 1}

# --- ヘルパー関数 ---

def _impact_from_label(value: Optional[str]) -> int:
    if not value:
        return 1
    return IMPACT_MAP.get(value.strip().lower(), 1)

def _impact(item) -> int:
    impact_str = "low"
    if "impact" in item:
        impact_str = item["impact"].strip().lower()
    elif "title" in item:
        title = item["title"]
        if "high impact" in title.lower():
            impact_str = "high"
        elif "medium impact" in title.lower():
            impact_str = "medium"

    if impact_str == "high":
        return 3
    if impact_str == "medium":
        return 2
    return 1

def _event_time(item) -> str | None:
    if hasattr(item, "published_parsed"):
        return datetime.datetime(*item.published_parsed[:6]).isoformat()
    if hasattr(item, "updated_parsed"):
        return datetime.datetime(*item.updated_parsed[:6]).isoformat()
    return None

def _push(blob_name: str, data: dict):
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data), content_type="application/json")

def _normalize_oanda_event(event: Dict[str, Any]) -> Optional[dict]:
    timestamp = event.get("timestamp") or event.get("timestamp_ms") or event.get("timestamp_millis")
    if timestamp:
        # OANDA returns seconds; guard for accidental millis
        if isinstance(timestamp, (float, int)) and timestamp > 4_000_000_000:
            timestamp = float(timestamp) / 1000.0
        try:
            ts_dt = datetime.datetime.utcfromtimestamp(float(timestamp)).isoformat()
        except (TypeError, ValueError):
            ts_dt = None
    else:
        ts_dt = None

    impact_val = _impact_from_label(event.get("impact") or event.get("importance"))
    if impact_val < 2:
        return None

    uid_components: List[str] = []
    for key in ("id", "timestamp", "title", "currency"):
        val = event.get(key)
        if val is not None:
            uid_components.append(str(val))
    uid = "oanda-" + "-".join(uid_components) if uid_components else f"oanda-{uuid.uuid4()}"

    body_parts: List[str] = []
    for label, key in (
        ("Actual", "actual"),
        ("Forecast", "forecast"),
        ("Previous", "previous"),
        ("Unit", "unit"),
        ("Region", "region"),
        ("Comment", "comment"),
    ):
        val = event.get(key)
        if val not in (None, "", []):
            body_parts.append(f"{label}: {val}")
    notes = event.get("notes") or event.get("detail")
    if notes:
        body_parts.append(str(notes))

    data = {
        "uid": uid,
        "src": "OANDA Calendar",
        "impact": impact_val,
        "event_time": ts_dt,
        "title": event.get("title", "Unknown event"),
        "currency": event.get("currency", ""),
        "time_utc": ts_dt,
        "body": " | ".join(body_parts),
    }
    return data

def _oanda_events(limit: int) -> List[dict]:
    if not OANDA_TOKEN:
        logging.warning("OANDA token missing; skip OANDA calendar fetch.")
        return []

    headers = {"Authorization": f"Bearer {OANDA_TOKEN}"}
    try:
        resp = session.get(
            OANDA_CALENDAR_URL,
            params={"instrument": "USD_JPY", "period": 86400},
            headers=headers,
            timeout=OANDA_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logging.exception("Failed to fetch OANDA calendar: %s", exc)
        return []

    events = payload.get("events") or []
    normalized: List[dict] = []
    for event in events:
        norm = _normalize_oanda_event(event)
        if not norm:
            continue
        normalized.append(norm)
        if len(normalized) >= limit:
            break
    return normalized

def _publish(items: Iterable[dict]) -> int:
    count = 0
    for entry in items:
        uid = entry.get("uid") or f"oanda-{uuid.uuid4()}"
        blob_name = f"raw/{uid}.json"
        blob = bucket.blob(blob_name)
        if blob.exists():
            continue
        _push(blob_name, entry)
        count += 1
    return count

# --- メイン処理 ---

async def fetch_loop():
    while True:
        await _rss_once()
        await asyncio.sleep(FETCH_INTERVAL_SECONDS)

async def _rss_once():
    fetched_count = 0
    fetched_count += _publish(_oanda_events(FETCH_LIMIT))
    remaining = FETCH_LIMIT - fetched_count
    if remaining <= 0 or not ENABLE_RSS_FALLBACK:
        return

    for url in (FF_URL, FF_NEXT, DAILYFX):
        if fetched_count >= FETCH_LIMIT:
            break
        parsed = feedparser.parse(url)
        feed_title = getattr(parsed.feed, "title", "RSS")
        for entry in parsed.entries:
            if fetched_count >= FETCH_LIMIT:
                break

            uid = entry.get("id") or str(uuid.uuid4())
            blob_name = f"raw/{uid}.json"
            blob = bucket.blob(blob_name)
            if blob.exists():
                continue

            impact_val = _impact(entry)
            if impact_val < 2:
                continue

            event_time_val = _event_time(entry)
            data = {
                "uid": uid,
                "src": feed_title,
                "impact": impact_val,
                "event_time": event_time_val,
                "title": entry.title,
                "currency": entry.get("currency", ""),
                "time_utc": entry.get("date", ""),
                "body": entry.get("summary", ""),
            }
            _push(blob_name, data)
            fetched_count += 1

if __name__ == "__main__":
    print("Start news fetcher…  Ctrl‑C to stop")
    try:
        asyncio.run(_rss_once())
    except KeyboardInterrupt:
        pass

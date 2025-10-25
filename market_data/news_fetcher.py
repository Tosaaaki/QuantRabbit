"""
market_data.news_fetcher
~~~~~~~~~~~~~~~~~~~~~~~~
USD/JPY 関連のニュース/指標フィードを 60 秒おきに巡回し、高インパクト情報を
Cloud Storage `fx-news/raw/*.json` へアップロードする。

2025-10: OANDA Labs API が 403 で利用できないため、FairEconomy / DailyFX /
Google News (USDJPY, BoJ キーワード) などの公開 RSS を利用する。
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import os
import pathlib
import re
import uuid
import xml.etree.ElementTree as ET
from typing import Iterable, List

import feedparser
import requests
import toml
from google.cloud import storage

# --- 設定・定数 -------------------------------------------------------------

FETCH_LIMIT = int(os.environ.get("NEWS_FETCH_MAX", "20"))
FETCH_INTERVAL_SECONDS = int(os.environ.get("NEWS_FETCH_INTERVAL_SEC", "300"))
MIN_IMPACT = int(os.environ.get("NEWS_MIN_IMPACT", "2"))  # 2=medium, 3=high

# env.toml 優先で GCS バケットを決定
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

if cfg:
    BUCKET = cfg.get("gcp", {}).get("bucket_news")
else:
    BUCKET = None

if not BUCKET:
    BUCKET = os.environ.get("BUCKET") or os.environ.get("BUCKET_NEWS") or "fx-news"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)

DEFAULT_FEEDS = [
    "https://www.dailyfx.com/feeds/most-recent-stories",
    "https://news.google.com/rss/search?q=USDJPY&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=BoJ+OR+%22Bank+of+Japan%22&hl=en-US&gl=US&ceid=US:en",
]

FAIRECONOMY_XML_FEEDS = [
    ("https://nfs.faireconomy.media/ff_calendar_thisweek.xml", "FairEconomy ThisWeek"),
]

extra_feeds = os.environ.get("NEWS_EXTRA_FEEDS")
if extra_feeds:
    DEFAULT_FEEDS.extend([url.strip() for url in extra_feeds.split(",") if url.strip()])

HIGH_WORDS = re.compile("|".join(["BoJ", "FOMC", "Non-Farm", "CPI", "政策金利", "雇用統計"]), re.IGNORECASE)

COUNTRY_TO_CURRENCY = {
    "Japan": "JPY",
    "United States": "USD",
    "Euro-Zone": "EUR",
    "Germany": "EUR",
    "France": "EUR",
    "Italy": "EUR",
    "Spain": "EUR",
    "United Kingdom": "GBP",
    "Australia": "AUD",
    "New Zealand": "NZD",
    "China": "CNY",
    "Canada": "CAD",
    "Switzerland": "CHF",
}

# --- ユーティリティ ---------------------------------------------------------

def _impact(entry) -> int:
    impact = entry.get("impact") or entry.get("importance")
    if isinstance(impact, str):
        impact = impact.lower()
        if impact == "high":
            return 3
        if impact in ("medium", "med"):
            return 2
        return 1
    title = entry.get("title", "")
    summary = entry.get("summary", "")
    if HIGH_WORDS.search(title) or HIGH_WORDS.search(summary):
        return 3
    return 2

def _event_time(entry) -> str | None:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime.datetime(*entry.published_parsed[:6]).isoformat()
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        return datetime.datetime(*entry.updated_parsed[:6]).isoformat()
    if entry.get("published"):
        return entry["published"]
    if entry.get("updated"):
        return entry["updated"]
    return None

def _currency(entry) -> str:
    currency = entry.get("currency")
    if currency:
        return currency
    text = f"{entry.get('title','')} {entry.get('summary','')}".upper()
    if "USD" in text and "JPY" in text:
        return "USD_JPY"
    if "JPY" in text:
        return "JPY"
    if "USD" in text:
        return "USD"
    return ""

def _uid(entry) -> str:
    for key in ("id", "guid", "link"):
        val = entry.get(key)
        if val:
            return str(val)
    joined = f"{entry.get('title','')}{entry.get('summary','')}{entry.get('published','')}"
    return hashlib.sha256(joined.encode("utf-8", errors="ignore")).hexdigest()

def _push(blob_name: str, data: dict):
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data), content_type="application/json")

def _country_currency(country: str | None) -> str:
    if not country:
        return ""
    return COUNTRY_TO_CURRENCY.get(country.strip(), "")

def _parse_faireconomy_datetime(date_str: str | None, time_str: str | None) -> str | None:
    if not date_str:
        return None
    date_str = date_str.strip()
    time_candidate = (time_str or "").strip()
    if time_candidate.lower() in {"", "all day", "tentative"}:
        time_candidate = ""
    candidates: List[str] = []
    if time_candidate:
        candidates.extend(
            [
                f"{date_str} {time_candidate}",
                f"{date_str} {time_candidate.upper()}",
            ]
        )
    else:
        candidates.append(date_str)
    formats = ["%m-%d-%Y %I:%M%p", "%m-%d-%Y %I%p", "%m-%d-%Y %H:%M", "%m-%d-%Y"]
    for value in candidates:
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(value, fmt)
                return dt.isoformat()
            except ValueError:
                continue
    return None

def _store_entry(data: dict) -> bool:
    uid = data.get("uid") or str(uuid.uuid4())
    data["uid"] = uid
    blob_name = f"raw/{uid}.json"
    blob = bucket.blob(blob_name)
    if blob.exists():
        return False
    _push(blob_name, data)
    return True

def _fetch_faireconomy_events(limit: int) -> List[dict]:
    events: List[dict] = []
    if limit <= 0:
        return events
    for url, source in FAIRECONOMY_XML_FEEDS:
        if len(events) >= limit:
            break
        try:
            resp = requests.get(url, timeout=8.0)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as exc:
            logging.warning("[NEWS] failed to fetch FairEconomy url=%s err=%s", url, exc)
            continue

        for node in root.findall(".//event"):
            if len(events) >= limit:
                break
            title = (node.findtext("title") or "").strip()
            if not title:
                continue
            country = (node.findtext("country") or "").strip()
            date_str = node.findtext("date")
            time_str = node.findtext("time")
            event_time = _parse_faireconomy_datetime(date_str, time_str)
            impact_label = (node.findtext("impact") or "").strip().lower()
            impact_val = 3 if impact_label == "high" else 2 if impact_label == "medium" else 1
            if impact_val < MIN_IMPACT:
                continue

            uid_seed = f"{source}|{title}|{date_str}|{time_str}"
            uid = hashlib.sha256(uid_seed.encode("utf-8", errors="ignore")).hexdigest()
            body_parts = []
            for key, label in (
                ("country", "Country"),
                ("forecast", "Forecast"),
                ("previous", "Previous"),
                ("date", "Date"),
                ("time", "Time"),
            ):
                val = (node.findtext(key) or "").strip()
                if val:
                    body_parts.append(f"{label}: {val}")
            body = " | ".join(body_parts)
            data = {
                "uid": uid,
                "src": source,
                "impact": impact_val,
                "event_time": event_time,
                "title": title,
                "currency": _country_currency(country),
                "time_utc": event_time or "",
                "body": body,
            }
            events.append(data)
    return events

# --- メイン処理 -------------------------------------------------------------

async def fetch_loop():
    logging.info("[NEWS] fetch loop started interval=%ss limit=%s", FETCH_INTERVAL_SECONDS, FETCH_LIMIT)
    while True:
        await _rss_once()
        await asyncio.sleep(FETCH_INTERVAL_SECONDS)

async def _rss_once():
    fetched_count = 0
    for entry in _fetch_faireconomy_events(limit=FETCH_LIMIT):
        if fetched_count >= FETCH_LIMIT:
            break
        if _store_entry(entry):
            fetched_count += 1

    for url in DEFAULT_FEEDS:
        if fetched_count >= FETCH_LIMIT:
            break
        try:
            parsed = feedparser.parse(url)
        except Exception as exc:
            logging.exception("[NEWS] failed to parse feed url=%s err=%s", url, exc)
            continue
        feed_title = getattr(parsed.feed, "title", url)
        for entry in parsed.entries:
            if fetched_count >= FETCH_LIMIT:
                break

            impact_val = _impact(entry)
            if impact_val < MIN_IMPACT:
                continue

            event_time_val = _event_time(entry)
            data = {
                "uid": _uid(entry),
                "src": feed_title,
                "impact": impact_val,
                "event_time": event_time_val,
                "title": entry.get("title", ""),
                "currency": _currency(entry),
                "time_utc": entry.get("published", ""),
                "body": entry.get("summary", ""),
            }
            if _store_entry(data):
                fetched_count += 1

    logging.info("[NEWS] cycle completed fetched=%s", fetched_count)

def publish_from_iter(items: Iterable[dict]) -> int:
    count = 0
    for entry in items:
        if _store_entry(entry):
            count += 1
    return count

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("Start RSS fetcher…  Ctrl‑C to stop")
    try:
        asyncio.run(_rss_once())
    except KeyboardInterrupt:
        pass

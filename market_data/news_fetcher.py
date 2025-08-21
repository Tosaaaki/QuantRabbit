"""
market_data.news_fetcher
~~~~~~~~~~~~~~~~~~~~~~~~
60 秒おきに RSS を取得し、高インパクトのイベントを
Cloud Storage `fx-news/raw/*.json` へアップロードする。
"""

from __future__ import annotations
import asyncio
import datetime
import json
import toml
import uuid
import feedparser
import re
from google.cloud import storage

import os
import pathlib

# --- 設定・定数 ---

# 環境変数や config からバケット名を読み込む
try:
    cfg = None
    cfg_path = pathlib.Path("config/env.local.toml")
    if cfg_path.exists():
        cfg = toml.load(cfg_path.open("r"))
    else:
        toml_path = pathlib.Path("config/env.toml")
        if toml_path.exists():
            cfg = toml.load(toml_path.open("r"))
    if cfg:
        BUCKET = cfg.get("gcp", {}).get("bucket_news")
    else:
        BUCKET = None
except Exception:
    BUCKET = None

if not BUCKET:
    BUCKET = os.environ.get("BUCKET") or os.environ.get("BUCKET_NEWS") or "fx-news"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
FF_NEXT = "https://nfs.faireconomy.media/ff_calendar_nextweek.xml"
DAILYFX = "https://www.dailyfx.com/feeds/most-recent-stories"

HIGH_WORDS = re.compile(
    "|".join(["BoJ", "FOMC", "Non-Farm", "CPI", "政策金利", "雇用統計"])
)

# --- ヘルパー関数 ---

def _impact(item) -> int:
    # ... (変更なし)
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
    # ... (変更なし)
    if hasattr(item, "published_parsed"):
        return datetime.datetime(*item.published_parsed[:6]).isoformat()
    if hasattr(item, "updated_parsed"):
        return datetime.datetime(*item.updated_parsed[:6]).isoformat()
    return None

def _push(blob_name: str, data: dict):
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data), content_type="application/json")

# --- メイン処理 ---

async def fetch_loop():
    while True:
        await _rss_once()
        await asyncio.sleep(3600)  # 1時間待機

async def _rss_once():
    now = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    fetched_count = 0

    for url in (FF_URL, FF_NEXT, DAILYFX):
        if fetched_count >= 10:
            break

        parsed = feedparser.parse(url)
        for entry in parsed.entries:
            if fetched_count >= 10:
                break

            uid = entry.get("id") or str(uuid.uuid4())
            
            # GCSにすでに同じUIDのファイルが存在するかチェック
            blob_name = f"raw/{uid}.json"
            blob = bucket.blob(blob_name)
            if blob.exists():
                continue

            impact_val = _impact(entry)
            if impact_val < 2:  # medium (2) or high (3) のみ
                continue

            event_time_val = _event_time(entry)
            data = {
                "uid": uid,
                "src": parsed.feed.title,
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
    print("Start RSS fetcher…  Ctrl‑C to stop")
    try:
        # For local testing, you might want to run it once
        asyncio.run(_rss_once())
    except KeyboardInterrupt:
        pass
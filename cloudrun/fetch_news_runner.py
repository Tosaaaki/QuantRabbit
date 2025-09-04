import logging

logging.basicConfig(level=logging.INFO)

import os
import json
import feedparser
import httpx
import re
import datetime
import uuid
import pathlib
import toml  # deprecated here; will avoid if not installed
from google.cloud import storage
from flask import Flask

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# --- 設定・定数 ---

# 環境変数や config からバケット名を読み込む（統一デフォルト: quantrabbit-fx-news）
BUCKET = os.environ.get("BUCKET") or os.environ.get("BUCKET_NEWS")
if not BUCKET:
    try:
        cfg = None
        for path in (pathlib.Path("config/env.local.toml"), pathlib.Path("config/env.toml")):
            if path.exists():
                cfg = toml.load(path.open("r"))
                break
        if cfg:
            BUCKET = cfg.get("news_bucket_name") or (
                cfg.get("gcp", {}).get("bucket_news") if isinstance(cfg.get("gcp"), dict) else None
            )
    except Exception:
        BUCKET = None
if not BUCKET:
    BUCKET = "quantrabbit-fx-news"
IMPACT_MIN = int(os.environ.get("IMPACT_MIN", "1"))  # relax filter to allow all by default

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
FF_NEXT = "https://nfs.faireconomy.media/ff_calendar_nextweek.xml"
DAILYFX = "https://www.dailyfx.com/feeds/most-recent-stories"
GOOGLE_NEWS = "https://news.google.com/rss/search?q=USD%2FJPY+OR+USDJPY&hl=en-US&gl=US&ceid=US:en"

HIGH_WORDS = re.compile("|".join(["BoJ", "FOMC", "Non-Farm", "CPI", "政策金利", "雇用統計"]))

NEWS_FEEDS = (FF_URL, FF_NEXT, DAILYFX, GOOGLE_NEWS)

# --- ヘルパー関数 ---

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


def _hash_uid(*parts: str) -> str:
    import hashlib
    base = "|".join([p or "" for p in parts])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _fetch_feed(url: str) -> feedparser.FeedParserDict:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True, headers=headers) as client:
            r = client.get(url)
            r.raise_for_status()
            return feedparser.parse(r.text)
    except Exception:
        try:
            return feedparser.parse(url)
        except Exception:
            return feedparser.parse("")

# --- メイン処理 ---

@app.route("/", methods=["GET"])
def run():
    fetched_count = 0

    for url in NEWS_FEEDS:
        if fetched_count >= 10:  # 最大10件に制限
            break
        parsed = _fetch_feed(url)
        try:
            feed_title = parsed.feed.title
        except Exception:
            feed_title = url
        logging.info(f"feed={url} entries={len(parsed.entries) if hasattr(parsed,'entries') else 0}")
        for entry in parsed.entries:
            if fetched_count >= 10:  # 最大10件に制限
                break
            uid = entry.get("id") or entry.get("link") or _hash_uid(entry.get("title",""), entry.get("published",""))
            uid_safe = _hash_uid(uid)

            # GCSにすでに同じUIDのファイルが存在するかチェック
            blob_name = f"raw/{uid_safe}.json"
            blob = bucket.blob(blob_name)
            if blob.exists():
                continue

            impact_val = _impact(entry)
            if impact_val < IMPACT_MIN:
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

    logging.info(f"Fetched {fetched_count} new news items.")
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

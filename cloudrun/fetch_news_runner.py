import logging

logging.basicConfig(level=logging.INFO)

import os
import json
import feedparser
try:
    import httpx
except ImportError:  # Cloud Run イメージ側の依存欠如を吸収
    httpx = None  # type: ignore
import re
import datetime
import time
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
# Prefer fresh items in the last few hours for Google News
GOOGLE_NEWS_RECENT = (
    "https://news.google.com/rss/search?"
    "q=(USD%2FJPY%20OR%20USDJPY)%20when%3A3h&hl=en-US&gl=US&ceid=US:en"
)
GOOGLE_NEWS_DAY = (
    "https://news.google.com/rss/search?"
    "q=(USD%2FJPY%20OR%20USDJPY)%20when%3A24h&hl=en-US&gl=US&ceid=US:en"
)

HIGH_WORDS = re.compile("|".join(["BoJ", "FOMC", "Non-Farm", "CPI", "政策金利", "雇用統計"]))

def _should_fetch_ff_next(today: datetime.date | None = None) -> bool:
    d = today or datetime.datetime.utcnow().date()
    # 木(3)〜日(6)のみ nextweek を参照（月〜水は404になりやすい）
    return d.weekday() in (3, 4, 5, 6)


def _news_feeds() -> tuple[str, ...]:
    feeds: list[str] = [FF_URL, DAILYFX, GOOGLE_NEWS_RECENT, GOOGLE_NEWS_DAY]
    if _should_fetch_ff_next():
        feeds.insert(1, FF_NEXT)
    return tuple(feeds)

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
    if httpx is not None:
        # 軽量の指数バックオフ（429/5xx/タイムアウト時のみ再試行）
        backoff = 0.5
        tries = 3
        try:
            with httpx.Client(timeout=8.0, follow_redirects=True, headers=headers) as client:
                for attempt in range(tries):
                    try:
                        resp = client.get(url)
                        sc = resp.status_code
                        if sc == 429 or 500 <= sc:
                            logging.warning("httpx %s for %s; retry %d/%d", sc, url, attempt + 1, tries)
                            if attempt < tries - 1:
                                time.sleep(backoff)
                                backoff *= 2
                                continue
                        if sc >= 400:
                            logging.warning("httpx status=%s for %s; fallback to feedparser", sc, url)
                            break
                        return feedparser.parse(resp.text)
                    except (Exception,) as e:
                        # ReadTimeout/ConnectError等
                        logging.warning("httpx exception for %s: %s (retry %d/%d)", url, str(e), attempt + 1, tries)
                        if attempt < tries - 1:
                            time.sleep(backoff)
                            backoff *= 2
                            continue
                        break
        except Exception as e:
            logging.warning("httpx client setup failed for %s; fallback: %s", url, str(e))
    # ネットワーク不調時に feedparser の直接取得でハングしないよう、
    # フォールバックは空パースに限定
    return feedparser.parse("")

# --- メイン処理 ---

@app.route("/", methods=["GET"])
def run():
    fetched_count = 0

    for url in _news_feeds():
        if fetched_count >= 25:  # 最大件数を引き上げ
            break
        parsed = _fetch_feed(url)
        try:
            feed_title = parsed.feed.title
        except Exception:
            feed_title = url
        logging.info(f"feed={url} entries={len(parsed.entries) if hasattr(parsed,'entries') else 0}")
        for entry in parsed.entries:
            if fetched_count >= 25:
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

from __future__ import annotations

import asyncio
import json
import sqlite3
import pathlib
import datetime
from google.cloud import storage
from utils.secrets import get_secret

# --- config ---
# env.toml から設定を取得
PROJECT_ID = get_secret("gcp_project_id")
BUCKET = get_secret("news_bucket_name")

# DB 初期化 -------------------------------------------------
_DB_PATH = pathlib.Path("logs/news.db")
_DB_PATH.parent.mkdir(exist_ok=True)

conn = sqlite3.connect(_DB_PATH)
cur = conn.cursor()
cur.execute(
    """
CREATE TABLE IF NOT EXISTS news(
    uid TEXT PRIMARY KEY,
    ts_utc TEXT,
    event_time TEXT, -- ISO format UTC
    horizon TEXT,
    summary TEXT,
    sentiment INTEGER,
    impact INTEGER, -- 1(low), 2(mid), 3(high)
    pair_bias TEXT
);
"""
)
conn.commit()


# GCS クライアント -----------------------------------------
storage_client = storage.Client(project=PROJECT_ID)  # 修正
bucket = storage_client.bucket(BUCKET)

SUMMARY_PREFIX = "summary/"
PROCESSED_PREFIX = "processed/"

# -----------------------------------------------------------


async def ingest_loop(interval_sec: int = 30):
    """30 秒ごとに summary/ をチェックして DB に挿入"""
    while True:
        await _run_once()
        await asyncio.sleep(interval_sec)


async def _run_once():
    blobs = bucket.list_blobs(prefix=SUMMARY_PREFIX)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        try:
            text = blob.download_as_text()
            data = json.loads(text)
            _upsert(data)
            # 移動: summary/ → processed/
            new_name = blob.name.replace(SUMMARY_PREFIX, PROCESSED_PREFIX, 1)
            bucket.rename_blob(blob, new_name)
        except Exception as e:
            print("summary_ingestor error:", e)


def _upsert(d: dict):
    cur.execute(
        """
    INSERT INTO news(uid, ts_utc, event_time, horizon, summary, sentiment, impact, pair_bias)
    VALUES(?,?,?,?,?,?,?,?)
    ON CONFLICT(uid) DO UPDATE SET
        event_time=excluded.event_time,
        horizon=excluded.horizon,
        summary=excluded.summary,
        sentiment=excluded.sentiment,
        impact=excluded.impact,
        pair_bias=excluded.pair_bias
    """,
        (
            d["uid"],
            datetime.datetime.utcnow().isoformat(timespec="seconds"),
            d.get("event_time"),
            d.get("horizon", "short"),
            d.get("summary", ""),
            d.get("sentiment", 0),
            d.get("impact", 1),
            d.get("pair_bias", ""),
        ),
    )
    conn.commit()


def get_latest_news(limit_short: int = 3, limit_long: int = 5) -> dict:
    """DBから最新のニュースを取得して返す"""
    cur.execute(
        """
        SELECT summary, ts_utc, horizon, sentiment, impact, event_time, pair_bias
        FROM news
        ORDER BY ts_utc DESC
        """
    )
    rows = cur.fetchall()

    news_short = []
    news_long = []

    for summary, ts, horizon, sentiment, impact, event_time, pair_bias in rows:
        item = {
            "summary": summary,
            "ts": ts,
            "horizon": horizon,
            "sentiment": sentiment,
            "impact": impact,
            "event_time": event_time,
            "pair_bias": pair_bias,
        }
        if horizon == "short" and len(news_short) < limit_short:
            news_short.append(item)
        elif horizon == "long" and len(news_long) < limit_long:
            news_long.append(item)

        if len(news_short) >= limit_short and len(news_long) >= limit_long:
            break

    return {"short": news_short, "long": news_long}


def check_event_soon(within_minutes: int = 30, min_impact: int = 3) -> bool:
    """指定された時間内に指定されたインパクト以上の経済指標があるかチェック"""
    now = datetime.datetime.utcnow()
    future_limit = now + datetime.timedelta(minutes=within_minutes)

    cur.execute(
        """SELECT event_time FROM news 
           WHERE impact >= ? AND event_time BETWEEN ? AND ?""",
        (min_impact, now.isoformat(), future_limit.isoformat()),
    )
    return cur.fetchone() is not None


# --------------- CLI self-test -----------------------------
if __name__ == "__main__":
    # # ダミーデータでテスト
    # _upsert({
    #     "uid": "test-123", "horizon": "short", "summary": "円安が進行中。",
    #     "sentiment": 1, "pair_bias": "USD_JPY_UP"
    # })
    # print(get_latest_news())

    import asyncio

    print("Start ingest loop (Ctrl-C to stop)...")
    try:
        asyncio.run(ingest_loop(10))
    except KeyboardInterrupt:
        pass

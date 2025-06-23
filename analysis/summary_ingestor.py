"""
analysis.summary_ingestor
~~~~~~~~~~~~~~~~~~~~~~~~~
Cloud Storage bucket `fx-news/summary/` に保存された
ニュース要約 JSON を定期ポーリングし、SQLite `logs/news.db`
に UPSERT するユーティリティ。
"""

from __future__ import annotations

import asyncio, json, sqlite3, tomllib, pathlib, datetime
from google.cloud import storage

CONF = tomllib.loads(open("config/env.toml", "rb").read())
BUCKET = CONF["gcp"]["bucket_news"]

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
    horizon TEXT,
    summary TEXT,
    sentiment INTEGER,
    pair_bias TEXT
);
"""
)
conn.commit()


# GCS クライアント -----------------------------------------
storage_client = storage.Client()
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
    INSERT INTO news(uid, ts_utc, horizon, summary, sentiment, pair_bias)
    VALUES(?,?,?,?,?,?)
    ON CONFLICT(uid) DO UPDATE SET
        horizon=excluded.horizon,
        summary=excluded.summary,
        sentiment=excluded.sentiment,
        pair_bias=excluded.pair_bias
    """,
        (
            d["uid"],
            datetime.datetime.utcnow().isoformat(timespec="seconds"),
            d.get("horizon", "short"),
            d.get("summary", ""),
            d.get("sentiment", 0),
            d.get("pair_bias", ""),
        ),
    )
    conn.commit()


# --------------- CLI self‑test -----------------------------
if __name__ == "__main__":
    import asyncio
    print("Start ingest loop (Ctrl‑C to stop)…")
    try:
        asyncio.run(ingest_loop(10))
    except KeyboardInterrupt:
        pass
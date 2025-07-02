# QuantRabbit/cloudrun/fetch_news_runner.py
"""
Cloud Run – Fetch News
----------------------
• 1 回呼ばれると Google ニュース RSS から 3 本だけ取って
  GCS (fx-news/raw/) に JSON を置く
• 失敗しても例外を握りつぶさず 500 を返す（Pub/Sub/Scheduler が再試行できるように）
"""

import os, json, uuid, datetime, feedparser
from google.cloud import storage
from flask import Flask, request

app = Flask(__name__)

BUCKET = os.environ["BUCKET"]
RSS    = os.environ.get("RSS_FEED", "https://news.google.com/rss/search?q=usd+jpy&hl=ja&gl=JP&ceid=JP:ja")

bucket = storage.Client().bucket(BUCKET)

def fetch_and_upload():
    """
    One-shot job:
    * Parse RSS
    * Upload up to 3 items to gs://<BUCKET>/raw/
    * Print how many were pushed; exceptions will surface to Cloud Run Job and mark failure.
    """
    feed = feedparser.parse(RSS)
    if not feed.entries:
        raise RuntimeError("no RSS entries")

    uploaded = 0
    for e in feed.entries[:3]:
        uid = str(uuid.uuid4())
        blob = bucket.blob(f"raw/{uid}.json")
        raw = {
            "uid": uid,
            "title": e.title,
            "body": getattr(e, "summary", ""),
            "link": e.link,
            "fetched_at": datetime.datetime.utcnow().isoformat(timespec="seconds")
        }
        blob.upload_from_string(
            json.dumps(raw, ensure_ascii=False),
            content_type="application/json"
        )
        uploaded += 1
    print(f"✅ uploaded {uploaded} article(s)")

@app.route("/", methods=["POST"])
def index():
    envelope = request.get_json()
    if not envelope:
        return "no Pub/Sub message received", 400
    
    if not isinstance(envelope, dict) or "message" not in envelope:
        return "invalid Pub/Sub message format", 400

    pubsub_message = envelope["message"]

    # Pub/Subメッセージのデータはbase64エンコードされている場合がある
    # 今回はデータ自体は使用しないが、形式として受け付ける
    if "data" in pubsub_message:
        # data = base64.b64decode(pubsub_message["data"]).decode("utf-8").strip()
        pass # データは使用しない

    try:
        fetch_and_upload()
        return "success", 200
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}", 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

import os
import json
import logging
from datetime import datetime

from flask import Flask
from google.cloud import storage
from google.cloud import firestore


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BUCKET = os.environ.get("BUCKET") or os.environ.get("BUCKET_NEWS") or "quantrabbit-fx-news"
SUMMARY_PREFIX = "summary/"
PROCESSED_PREFIX = "processed/"

app = Flask(__name__)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)
fs = firestore.Client()


@app.route("/", methods=["GET"])  # 1リクエスト=1回分の取り込み
def run_once():
    try:
        processed = 0
        blobs = bucket.list_blobs(prefix=SUMMARY_PREFIX)
        for blob in blobs:
            name = blob.name
            if not name or name.endswith("/"):
                continue
            try:
                payload = json.loads(blob.download_as_text())
                uid = payload.get("uid") or name
                doc = {
                    "uid": uid,
                    "ts_utc": payload.get("ts_utc") or datetime.utcnow().isoformat(timespec="seconds"),
                    "event_time": payload.get("event_time"),
                    "horizon": payload.get("horizon", "short"),
                    "summary": payload.get("summary", ""),
                    "sentiment": int(payload.get("sentiment", 0) or 0),
                    "impact": int(payload.get("impact", 1) or 1),
                    "pair_bias": payload.get("pair_bias", ""),
                }
                fs.collection("news").document(uid).set(doc)
                # move to processed/
                new_name = name.replace(SUMMARY_PREFIX, PROCESSED_PREFIX, 1)
                bucket.rename_blob(blob, new_name)
                processed += 1
            except Exception as e:
                logging.error(f"ingestor error for blob={name}: {e}")
        logging.info(f"news ingested: {processed}")
        return "OK", 200
    except Exception as e:
        logging.error(f"Unhandled ingestor error: {e}")
        return "ERROR", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


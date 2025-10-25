import os
import json
import base64
import logging
import datetime
from flask import Flask, request
from google.cloud import storage
import openai

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

openai.api_key = os.environ.get("OPENAI_API_KEY")
BUCKET = os.environ.get("BUCKET")
SUMMARIZER_MODEL = (
    os.environ.get("OPENAI_SUMMARIZER_MODEL")
    or os.environ.get("OPENAI_MODEL")
    or "gpt-4o-mini"
)
logging.info("Configured OpenAI summarizer model: %s", SUMMARIZER_MODEL)

if not BUCKET:
    logging.error("BUCKET environment variable is not set.")
    # You might want to exit or handle this case appropriately
    # For now, let's try to default it, but this is not ideal
    BUCKET = "fx-news"

app = Flask(__name__)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)


def _normalize_result(res: object) -> dict:
    """Return dict with keys 'summary' (str) and 'sentiment' (int)."""
    if isinstance(res, str):
        return {"summary": res.strip(), "sentiment": 0}
    if not isinstance(res, dict):
        return {"summary": str(res).strip(), "sentiment": 0}

    summary = res.get("summary") or res.get("要約") or res.get("title") or ""
    sentiment = res.get("sentiment") or res.get("impact") or 0
    try:
        sentiment = int(sentiment)
    except (ValueError, TypeError):
        sentiment = 0
    return {"summary": summary.strip(), "sentiment": sentiment}


def summarize(text: str) -> dict:
    """OpenAI summarizer → return dict {summary, sentiment}"""
    if not openai.api_key:
        logging.error("OPENAI_API_KEY is not set. Cannot summarize.")
        return {"summary": "Error: API key not configured.", "sentiment": 0}

    prompt = (
        "以下のニュース本文を最大120字で日本語要約し、"
        "USD/JPYへのインパクトを -2〜+2 の整数で付与し "
        "次の JSON 形式で返してください。\n"
        '{"summary":"...", "sentiment":-2〜+2}\n'
        "### 原文\n" + text[:1500]
    )
    base_kwargs = {
        "model": SUMMARIZER_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "あなたは金融ニュースの要約アシスタントです。",
            },
            {"role": "user", "content": prompt},
        ],
    }
    if "gpt-5" not in SUMMARIZER_MODEL:
        base_kwargs["temperature"] = 0.3
    token_kwargs_order = [
        {"max_completion_tokens": 120},
        {"max_tokens": 120},
    ]

    try:
        resp = None
        logging.info("Sending request to OpenAI API.")
        last_error = None
        for token_kwargs in token_kwargs_order:
            try:
                resp = openai.chat.completions.create(**base_kwargs, **token_kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                param_name = next(iter(token_kwargs))
                if "Unsupported parameter" in msg and param_name in msg:
                    last_error = exc
                    continue
                raise
        else:
            raise RuntimeError(
                str(last_error) if last_error else "Unable to call OpenAI"
            )
        logging.info("Received response from OpenAI API.")
        content = resp.choices[0].message.content
        logging.info(f"OpenAI response content: {content}")
        data = json.loads(content)
        return _normalize_result(data)
    except Exception as e:
        logging.error(f"An error occurred with OpenAI API: {e}")
        return {"summary": f"Error during summarization: {e}", "sentiment": 0}


@app.route("/", methods=["POST"])
def ingest():
    logging.info(f"Request headers: {request.headers}")
    logging.info(f"Request data: {request.data}")
    try:
        envelope = request.get_json()
        if not envelope or "message" not in envelope:
            logging.error(f"Invalid Pub/Sub message format: {request.data}")
            return "Invalid message format", 400

        msg = envelope["message"]
        attrs = msg.get("attributes", {})
        object_name = attrs.get("objectId")

        if not object_name and "data" in msg:
            try:
                decoded_data = base64.b64decode(msg["data"]).decode("utf-8")
                logging.info(f"Decoded data: {decoded_data}")
                data_json = json.loads(decoded_data)
                object_name = data_json.get("name")
            except Exception as e:
                logging.error(f"Error decoding message data: {e}")
                object_name = None

        logging.info(f"Extracted object name: {object_name}")

        if not object_name or not object_name.startswith("raw/"):
            logging.info(f"Skipping non-raw or missing object_name: {object_name}")
            return "Skipping", 200

        logging.info(f"Processing object: {object_name}")
        blob = bucket.blob(object_name)
        try:
            raw_text = blob.download_as_text()
            logging.info(f"Downloaded raw text for {object_name}: {raw_text[:200]}...")
            raw = json.loads(raw_text)
        except Exception as e:
            logging.error(f"Failed to download or parse blob {object_name}: {e}")
            return "Error processing blob", 500

        text_to_summarize = raw.get("title", "") + "\n" + raw.get("body", "")
        logging.info(f"Text to summarize: {text_to_summarize[:200]}...")
        result = summarize(text_to_summarize)
        logging.info(f"Summarization result: {result}")

        summary_blob_name = object_name.replace("raw/", "summary/", 1)
        payload = {
            "uid": raw.get("uid", object_name),
            "ts_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "event_time": raw.get("event_time"),
            "impact": raw.get("impact", 1),
            "summary": result["summary"],
            "sentiment": int(result.get("sentiment", 0) or 0),
            "horizon": "short",
            "pair_bias": raw.get("currency", "USD/JPY"),
        }
        logging.info(f"Uploading payload to {summary_blob_name}: {payload}")
        summary_blob = bucket.blob(summary_blob_name)
        summary_blob.upload_from_string(
            json.dumps(payload),
            content_type="application/json",
        )
        logging.info(f"Successfully summarized and uploaded {object_name}")
        return "OK", 200

    except Exception as e:
        logging.error(f"Unhandled error in ingest: {e}", exc_info=True)
        return "Internal Server Error", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

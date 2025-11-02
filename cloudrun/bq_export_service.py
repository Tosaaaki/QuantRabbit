from __future__ import annotations

import os
import traceback
from flask import Flask, jsonify, request

from analytics.bq_exporter import BigQueryExporter, _DB_DEFAULT, _STATE_DEFAULT

app = Flask(__name__)


def _export(limit: int | None = None):
    exporter = BigQueryExporter(
        sqlite_path=os.getenv("SQLITE_PATH", _DB_DEFAULT),
        state_path=os.getenv("BQ_SYNC_STATE", _STATE_DEFAULT),
        project_id=os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT"),
        dataset_id=os.getenv("BQ_DATASET", "quantrabbit"),
        trades_table_id=os.getenv("BQ_TRADES_TABLE", "trades_raw"),
        feature_table_id=os.getenv("BQ_FEATURE_TABLE", "trades_daily_features"),
    )
    stats = exporter.export(limit=limit or int(os.getenv("BQ_MAX_EXPORT", "5000")))
    return {"exported": stats.exported, "last_updated_at": stats.last_updated_at}


@app.route("/export", methods=["GET"])  # idempotent; safe for Scheduler
def export_handler():
    try:
        limit_param = request.args.get("limit")
        limit = int(limit_param) if limit_param else None
    except Exception:
        limit = None
    try:
        result = _export(limit)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return (
            jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}),
            500,
        )


@app.route("/")
def health():
    return jsonify({"ok": True, "service": "bq_exporter"})


from __future__ import annotations

import os
import traceback
from flask import Flask, jsonify, request

from analytics.risk_model import RiskModelPipeline

app = Flask(__name__)


def _pipeline() -> RiskModelPipeline:
    return RiskModelPipeline(
        project_id=os.getenv("RISK_MODEL_PROJECT") or os.getenv("BQ_PROJECT"),
        dataset_id=os.getenv("RISK_MODEL_DATASET") or os.getenv("BQ_DATASET"),
        feature_table=os.getenv("RISK_MODEL_FEATURE_TABLE") or os.getenv("BQ_FEATURE_TABLE"),
        model_id=os.getenv("RISK_MODEL_ID"),
        pubsub_topic=os.getenv("RISK_PUBSUB_TOPIC"),
        min_trades=int(os.getenv("RISK_MODEL_MIN_TRADES", "5")),
        lookback_days=int(os.getenv("RISK_MODEL_LOOKBACK_DAYS", "120")),
        state_path=os.getenv("RISK_MODEL_STATE", "logs/risk_scores.json"),
    )


@app.route("/run", methods=["POST", "GET"])
def run_pipeline():
    train_flag = request.args.get("train", "true").lower() != "false"
    publish_flag = request.args.get("publish", "true").lower() != "false"
    try:
        pipeline = _pipeline()
        scores = pipeline.run(train=train_flag, publish=publish_flag)
        return jsonify(
            {
                "ok": True,
                "train": train_flag,
                "publish": publish_flag,
                "count": len(scores),
                "scores": [score.__dict__ for score in scores],
            }
        )
    except Exception as exc:
        return (
            jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}),
            500,
        )


@app.route("/")
def health() -> tuple[str, int]:
    return jsonify({"ok": True, "service": "risk-model"}), 200

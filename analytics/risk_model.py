from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from google.api_core.exceptions import BadRequest, NotFound
from google.cloud import bigquery, pubsub_v1


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, (int, Decimal)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


@dataclass
class RiskScore:
    pocket: str
    strategy: str
    trade_date: str
    trades: int
    win_rate: Optional[float]
    avg_pips: Optional[float]
    pf: Optional[float]
    predicted_pf: Optional[float]
    score: float
    multiplier: float


class RiskModelPipeline:
    """Train a BigQuery ML model and publish risk multipliers via Pub/Sub."""

    def __init__(
        self,
        *,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        feature_table: Optional[str] = None,
        model_id: Optional[str] = None,
        pubsub_topic: Optional[str] = None,
        min_trades: int = 5,
        lookback_days: int = 120,
        state_path: Optional[str] = None,
    ) -> None:
        project_env = (
            project_id
            or os.getenv("RISK_MODEL_PROJECT")
            or os.getenv("BQ_PROJECT")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
        )
        if not project_env:
            raise RuntimeError("BQ project is not configured (set RISK_MODEL_PROJECT or BQ_PROJECT)")

        self.project_id = project_env
        self.dataset_id = dataset_id or os.getenv("RISK_MODEL_DATASET") or os.getenv("BQ_DATASET") or "quantrabbit"
        self.feature_table = feature_table or os.getenv("RISK_MODEL_FEATURE_TABLE") or os.getenv("BQ_FEATURE_TABLE") or "trades_daily_features"
        self.model_id = model_id or os.getenv("RISK_MODEL_ID") or "strategy_risk_model"
        self.min_trades = max(1, int(os.getenv("RISK_MODEL_MIN_TRADES", str(min_trades))))
        self.lookback_days = max(7, int(os.getenv("RISK_MODEL_LOOKBACK_DAYS", str(lookback_days))))
        self.topic = pubsub_topic or os.getenv("RISK_PUBSUB_TOPIC")
        self.client = bigquery.Client(project=self.project_id)
        self.publisher: Optional[pubsub_v1.PublisherClient] = None
        if self.topic:
            self.publisher = pubsub_v1.PublisherClient()
        state_default = state_path or os.getenv("RISK_MODEL_STATE", "logs/risk_scores.json")
        self.state_path = Path(state_default)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, *, train: bool = True, publish: bool = True) -> List[RiskScore]:
        if train:
            self.train_model()
        scores = self.predict_scores()
        payload = {
            "type": "risk_scores",
            "source": os.getenv("RISK_MODEL_SOURCE", "risk-model"),
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": f"{self.project_id}.{self.dataset_id}.{self.model_id}",
            "entries": [asdict(score) for score in scores],
        }
        self._write_state(payload)
        if publish and self.publisher and self.topic:
            topic_path = self._topic_path(self.topic)
            future = self.publisher.publish(
                topic_path,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                origin="risk-model",
            )
            try:
                message_id = future.result(timeout=30)
                logging.info("[RISK_MODEL] published message_id=%s topic=%s", message_id, topic_path)
            except Exception as exc:
                logging.warning("[RISK_MODEL] publish failed: %s", exc)
        elif publish and not self.topic:
            logging.warning("[RISK_MODEL] publish requested but RISK_PUBSUB_TOPIC is unset")
        return scores

    def train_model(self) -> None:
        query = f"""
        CREATE OR REPLACE MODEL `{self.project_id}.{self.dataset_id}.{self.model_id}`
        OPTIONS(
          model_type='linear_reg',
          input_label_cols=['target_pf'],
          data_split_method='AUTO_SPLIT'
        ) AS
        WITH base AS (
          SELECT
            trade_date,
            pocket,
            strategy,
            trades,
            win_rate,
            avg_pips,
            pf,
            LEAD(pf) OVER (PARTITION BY pocket, strategy ORDER BY trade_date) AS next_pf
          FROM `{self.project_id}.{self.dataset_id}.{self.feature_table}`
          WHERE trades >= @min_trades
            AND trade_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @lookback_days DAY)
        )
        SELECT
          pocket,
          strategy,
          trades,
          win_rate,
          avg_pips,
          pf,
          next_pf AS target_pf
        FROM base
        WHERE next_pf IS NOT NULL
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("min_trades", "INT64", self.min_trades),
                bigquery.ScalarQueryParameter("lookback_days", "INT64", self.lookback_days),
            ]
        )
        logging.info(
            "[RISK_MODEL] training model %s.%s.%s (lookback=%sd, min_trades=%s)",
            self.project_id,
            self.dataset_id,
            self.model_id,
            self.lookback_days,
            self.min_trades,
        )
        job = self.client.query(query, job_config=job_config)
        try:
            job.result()
        except BadRequest as exc:
            message = str(exc)
            if "Input data doesn't contain any rows" in message:
                logging.info("[RISK_MODEL] skip training: no rows in feature table (min_trades=%s)", self.min_trades)
                return
            raise

    def predict_scores(self) -> List[RiskScore]:
        try:
            self.client.get_model(f"{self.project_id}.{self.dataset_id}.{self.model_id}")
        except NotFound:
            logging.info(
                "[RISK_MODEL] model %s.%s.%s does not exist yet; skip prediction",
                self.project_id,
                self.dataset_id,
                self.model_id,
            )
            return []
        query = f"""
        WITH latest AS (
          SELECT
            trade_date,
            pocket,
            strategy,
            trades,
            win_rate,
            avg_pips,
            pf,
            ROW_NUMBER() OVER (PARTITION BY pocket, strategy ORDER BY trade_date DESC) AS rn
          FROM `{self.project_id}.{self.dataset_id}.{self.feature_table}`
          WHERE trades >= @min_trades
        ),
        scored AS (
          SELECT
            trade_date,
            pocket,
            strategy,
            trades,
            win_rate,
            avg_pips,
            pf
          FROM latest
          WHERE rn = 1
        )
        SELECT * FROM ML.PREDICT(
          MODEL `{self.project_id}.{self.dataset_id}.{self.model_id}`,
          TABLE scored
        )
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("min_trades", "INT64", self.min_trades)]
        )
        rows = list(self.client.query(query, job_config=job_config).result())
        if not rows:
            logging.info("[RISK_MODEL] no rows available for prediction (min_trades=%s)", self.min_trades)
            return []
        scores: List[RiskScore] = []
        for row in rows:
            predicted_pf = _to_float(row.get("predicted_target_pf"))
            trade_date = row.get("trade_date")
            trades = int(row.get("trades") or 0)
            win_rate = _to_float(row.get("win_rate"))
            avg_pips = _to_float(row.get("avg_pips"))
            pf = _to_float(row.get("pf"))
            pocket = str(row.get("pocket") or "").lower()
            strategy = str(row.get("strategy") or "")
            score_val = self._score_value(predicted_pf, pf, win_rate, trades)
            multiplier = self._multiplier(score_val, predicted_pf, trades)
            scores.append(
                RiskScore(
                    pocket=pocket,
                    strategy=strategy,
                    trade_date=str(trade_date),
                    trades=trades,
                    win_rate=win_rate,
                    avg_pips=avg_pips,
                    pf=pf,
                    predicted_pf=predicted_pf,
                    score=round(score_val, 4),
                    multiplier=round(multiplier, 3),
                )
            )
        scores.sort(key=lambda item: (item.pocket, item.strategy))
        return scores

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _score_value(
        self,
        predicted_pf: Optional[float],
        current_pf: Optional[float],
        win_rate: Optional[float],
        trades: int,
    ) -> float:
        pf_term = ((predicted_pf or 1.0) - 1.0) * 0.6
        base_term = ((current_pf or 1.0) - 1.0) * 0.25
        win_term = ((win_rate or 0.5) - 0.5) * 0.5
        confidence = min(max(trades, 0) / 25.0, 1.0)
        return (pf_term + base_term + win_term) * confidence

    def _multiplier(self, score: float, predicted_pf: Optional[float], trades: int) -> float:
        pf = predicted_pf or 1.0
        strong_conf = trades >= 15
        if score >= 0.28 or (pf >= 1.85 and strong_conf):
            return 1.4
        if score >= 0.18 or pf >= 1.55:
            return 1.25
        if score >= 0.08 or pf >= 1.25:
            return 1.1
        if score <= -0.3 or pf <= 0.7:
            return 0.5
        if score <= -0.18 or pf <= 0.85:
            return 0.7
        if score <= -0.08 or pf <= 0.95:
            return 0.85
        return 1.0

    def _topic_path(self, topic: str) -> str:
        if topic.startswith("projects/"):
            return topic
        if not self.publisher:
            raise RuntimeError("Pub/Sub publisher is not initialized")
        return self.publisher.topic_path(self.project_id, topic)

    def _write_state(self, payload: Dict[str, Any]) -> None:
        try:
            self.state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception as exc:
            logging.warning("[RISK_MODEL] failed to persist state: %s", exc)


def load_scores_from_state(state_path: str | Path = "logs/risk_scores.json") -> Iterable[RiskScore]:
    path = Path(state_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        entries = data.get("entries", [])
        out: List[RiskScore] = []
        for item in entries:
            out.append(
                RiskScore(
                    pocket=item.get("pocket", ""),
                    strategy=item.get("strategy", ""),
                    trade_date=item.get("trade_date", ""),
                    trades=int(item.get("trades") or 0),
                    win_rate=_to_float(item.get("win_rate")),
                    avg_pips=_to_float(item.get("avg_pips")),
                    pf=_to_float(item.get("pf")),
                    predicted_pf=_to_float(item.get("predicted_pf")),
                    score=float(item.get("score") or 0.0),
                    multiplier=float(item.get("multiplier") or 1.0),
                )
            )
        return out
    except Exception:
        return []

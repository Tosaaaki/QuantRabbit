from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from google.api_core import exceptions as gexc
from google.cloud import bigquery, storage

from utils.secrets import get_secret


def _optional_secret(key: str) -> Optional[str]:
    try:
        return get_secret(key)
    except KeyError:
        return None


def _utcnow_str() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LotInsight:
    pocket: str
    side: str
    trade_count: int
    win_rate: float
    profit_factor: float
    avg_pips: float
    std_pips: float
    total_pips: float
    avg_hold_minutes: float | None
    score: float
    risk_multiplier: float
    confidence: str
    generated_at: str


class LotPatternAnalyzer:
    """BigQuery trades テーブルを解析し、Pocket ごとのロット調整ヒントを生成する。"""

    def __init__(
        self,
        *,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        trades_table: Optional[str] = None,
        insights_table: Optional[str] = None,
        lookback_days: int = 14,
        min_trades: int = 15,
        gcs_bucket: Optional[str] = None,
        gcs_object: str = "analytics/lot_insights.json",
    ) -> None:
        secret_project = _optional_secret("gcp_project_id") or _optional_secret("BQ_PROJECT")
        self.project_id = project_id or secret_project
        self.dataset_id = dataset_id or _optional_secret("BQ_DATASET") or "quantrabbit"
        self.trades_table = trades_table or _optional_secret("BQ_TRADES_TABLE") or "trades_raw"
        self.insights_table = insights_table or "lot_insights"
        self.lookback_days = lookback_days
        self.min_trades = min_trades
        self.client = bigquery.Client(project=self.project_id) if self.project_id else bigquery.Client()

        # GCS 出力は任意
        bucket = gcs_bucket
        if bucket is None:
            bucket = _optional_secret("analytics_bucket_name") or _optional_secret("ui_bucket_name")
        self._gcs_bucket_name = bucket
        self._gcs_object = gcs_object
        self._storage_client: Optional[storage.Client] = None
        self._storage_bucket = None
        if bucket:
            try:
                self._storage_client = storage.Client(project=self.project_id)
                self._storage_bucket = self._storage_client.bucket(bucket)
            except Exception as exc:  # noqa: BLE001
                logging.warning("[LotAnalyzer] GCS 初期化に失敗: %s", exc)
                self._storage_client = None
                self._gcs_bucket_name = None
                self._storage_bucket = None

        self._ensure_dataset()
        self._ensure_insights_table()

    # -------------------- public API --------------------

    def run(self) -> List[LotInsight]:
        rows = self._query_metrics()
        if not rows:
            logging.info("[LotAnalyzer] 対象期間内のトレードがありません。")
            return []
        insights = [self._build_insight(row) for row in rows]
        insights = [ins for ins in insights if ins is not None]  # type: ignore[list-item]
        insights = [ins for ins in insights if math.isfinite(ins.risk_multiplier)]
        if not insights:
            logging.info("[LotAnalyzer] 有効なインサイトを生成できませんでした。")
            return []
        self._store_bigquery(insights)
        self._write_gcs(insights)
        return insights

    # -------------------- internal helpers --------------------

    def _query_metrics(self) -> Iterable[bigquery.table.Row]:
        table_ref = f"{self.client.project}.{self.dataset_id}.{self.trades_table}"
        sql = f"""
WITH raw AS (
  SELECT
    pocket,
    CASE WHEN SAFE_CAST(units AS INT64) >= 0 THEN 'LONG' ELSE 'SHORT' END AS side,
    SAFE_CAST(pl_pips AS FLOAT64) AS pl_pips,
    SAFE_CAST(entry_time AS STRING) AS entry_time,
    SAFE_CAST(close_time AS STRING) AS close_time,
    SAFE_CAST(units AS INT64) AS units
  FROM `{table_ref}`
  WHERE state = 'CLOSED'
    AND close_time IS NOT NULL
    AND updated_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY)
),
base AS (
  SELECT
    pocket,
    side,
    pl_pips,
    ABS(units) AS abs_units,
    CASE
      WHEN close_time IS NOT NULL AND entry_time IS NOT NULL THEN
        TIMESTAMP_DIFF(
          SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S%Ez', close_time),
          SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S%Ez', entry_time),
          MINUTE
        )
    END AS hold_minutes
  FROM raw
  WHERE pl_pips IS NOT NULL
),
per_side AS (
  SELECT
    pocket,
    side,
    COUNT(*) AS trade_count,
    SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS winning_trades,
    AVG(pl_pips) AS avg_pips,
    SUM(pl_pips) AS total_pips,
    STDDEV_SAMP(pl_pips) AS std_pips,
    SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS gross_profit,
    ABS(SUM(CASE WHEN pl_pips < 0 THEN pl_pips ELSE 0 END)) AS gross_loss,
    AVG(hold_minutes) AS avg_hold_minutes
  FROM base
  GROUP BY pocket, side
),
per_pocket AS (
  SELECT
    pocket,
    'ALL' AS side,
    COUNT(*) AS trade_count,
    SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS winning_trades,
    AVG(pl_pips) AS avg_pips,
    SUM(pl_pips) AS total_pips,
    STDDEV_SAMP(pl_pips) AS std_pips,
    SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS gross_profit,
    ABS(SUM(CASE WHEN pl_pips < 0 THEN pl_pips ELSE 0 END)) AS gross_loss,
    AVG(hold_minutes) AS avg_hold_minutes
  FROM base
  GROUP BY pocket
)
SELECT * FROM per_side
UNION ALL
SELECT * FROM per_pocket
ORDER BY pocket, side
"""
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("lookback_days", "INT64", self.lookback_days),
            ]
        )
        try:
            job = self.client.query(sql, job_config=job_config)
            return list(job.result())
        except gexc.NotFound as exc:
            raise RuntimeError(f"LotAnalyzer trades table missing: {table_ref}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LotAnalyzer query failed: {exc}") from exc

    def _build_insight(self, row: bigquery.table.Row) -> Optional[LotInsight]:
        pocket = row.get("pocket")
        side = row.get("side")
        trade_count = int(row.get("trade_count") or 0)
        if not pocket or trade_count == 0:
            return None

        winning_trades = float(row.get("winning_trades") or 0.0)
        win_rate = winning_trades / trade_count if trade_count else 0.0
        avg_pips = float(row.get("avg_pips") or 0.0)
        total_pips = float(row.get("total_pips") or 0.0)
        std_pips = float(row.get("std_pips") or 0.0)
        gross_profit = float(row.get("gross_profit") or 0.0)
        gross_loss = float(row.get("gross_loss") or 0.0)
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
        avg_hold = row.get("avg_hold_minutes")
        avg_hold_minutes: Optional[float] = None
        if avg_hold is not None:
            try:
                avg_hold_minutes = float(avg_hold)
            except (TypeError, ValueError):
                avg_hold_minutes = None

        score = self._compute_score(trade_count, win_rate, pf, avg_pips, std_pips)
        multiplier = self._score_to_multiplier(score, trade_count)
        confidence = self._confidence_label(trade_count, pf, win_rate)

        return LotInsight(
            pocket=pocket,
            side=side,
            trade_count=trade_count,
            win_rate=round(win_rate, 4),
            profit_factor=round(pf, 3) if math.isfinite(pf) else float("inf"),
            avg_pips=round(avg_pips, 3),
            std_pips=round(std_pips, 3) if std_pips is not None else 0.0,
            total_pips=round(total_pips, 3),
            avg_hold_minutes=round(avg_hold_minutes, 2) if avg_hold_minutes is not None else None,
            score=round(score, 3),
            risk_multiplier=round(multiplier, 3),
            confidence=confidence,
            generated_at=_utcnow_str(),
        )

    def _compute_score(
        self,
        trade_count: int,
        win_rate: float,
        profit_factor: float,
        avg_pips: float,
        std_pips: float,
    ) -> float:
        if trade_count < self.min_trades:
            return 0.0

        wr_component = max(min((win_rate - 0.5) * 1.6, 0.5), -0.5)
        pf_component = 0.0
        if math.isfinite(profit_factor):
            pf_component = max(min((profit_factor - 1.0) * 0.6, 0.5), -0.5)

        stability_component = 0.0
        if std_pips and abs(std_pips) > 1e-6:
            stability_component = max(min((avg_pips / std_pips) * 0.3, 0.3), -0.3)

        score = wr_component + pf_component + stability_component

        volume_bonus = 0.0
        if trade_count >= 100:
            volume_bonus = 0.1
        elif trade_count >= 60:
            volume_bonus = 0.05

        return max(min(score + volume_bonus, 0.6), -0.6)

    def _score_to_multiplier(self, score: float, trade_count: int) -> float:
        base = 1.0
        if trade_count < self.min_trades:
            return base
        multiplier = base + score
        return max(0.5, min(1.6, multiplier))

    def _confidence_label(self, trade_count: int, pf: float, win_rate: float) -> str:
        if trade_count < self.min_trades or win_rate <= 0.5:
            return "low"
        if trade_count >= 100 and pf >= 1.25 and win_rate >= 0.56:
            return "high"
        if trade_count >= 45 and pf >= 1.1 and win_rate >= 0.53:
            return "medium"
        return "low"

    def _store_bigquery(self, insights: Iterable[LotInsight]) -> None:
        table_ref = f"{self.client.project}.{self.dataset_id}.{self.insights_table}"
        payload = [self._insight_to_row(ins) for ins in insights]
        if not payload:
            return
        errors = self.client.insert_rows_json(table_ref, payload)
        if errors:
            raise RuntimeError(f"LotAnalyzer insert failed: {errors}")
        logging.info(
            "[LotAnalyzer] insights inserted rows=%d table=%s", len(payload), table_ref
        )

    def _write_gcs(self, insights: Iterable[LotInsight]) -> None:
        if not self._gcs_bucket_name or not self._storage_client or not self._storage_bucket:
            return
        data = [asdict(ins) for ins in insights]
        serialized = json.dumps(
            {"generated_at": _utcnow_str(), "insights": data},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            blob = self._storage_bucket.blob(self._gcs_object)
            blob.cache_control = "no-cache"
            blob.upload_from_string(serialized, content_type="application/json")
            logging.info(
                "[LotAnalyzer] insights uploaded to gs://%s/%s",
                self._gcs_bucket_name,
                self._gcs_object,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("[LotAnalyzer] GCS アップロードに失敗: %s", exc)

    def _ensure_insights_table(self) -> None:
        table_ref = f"{self.client.project}.{self.dataset_id}.{self.insights_table}"
        schema = [
            bigquery.SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("pocket", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("side", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("trade_count", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("win_rate", "FLOAT64"),
            bigquery.SchemaField("profit_factor", "FLOAT64"),
            bigquery.SchemaField("avg_pips", "FLOAT64"),
            bigquery.SchemaField("std_pips", "FLOAT64"),
            bigquery.SchemaField("total_pips", "FLOAT64"),
            bigquery.SchemaField("avg_hold_minutes", "FLOAT64"),
            bigquery.SchemaField("score", "FLOAT64"),
            bigquery.SchemaField("risk_multiplier", "FLOAT64"),
            bigquery.SchemaField("confidence", "STRING"),
        ]
        try:
            self.client.get_table(table_ref)
        except gexc.NotFound:
            logging.info("[LotAnalyzer] insights table %s を作成します。", table_ref)
            table = bigquery.Table(table_ref, schema=schema)
            table.time_partitioning = bigquery.TimePartitioning(
                field="generated_at", expiration_ms=None
            )
            table.clustering_fields = ["pocket", "side"]
            self.client.create_table(table)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LotAnalyzer insights table check failed: {exc}") from exc

    def _ensure_dataset(self) -> None:
        dataset_ref = bigquery.DatasetReference(self.client.project, self.dataset_id)
        try:
            self.client.get_dataset(dataset_ref)
        except gexc.NotFound:
            logging.info("[LotAnalyzer] dataset %s を作成します。", dataset_ref.dataset_id)
            dataset = bigquery.Dataset(dataset_ref)
            location = _optional_secret("BQ_LOCATION") or "US"
            dataset.location = location
            try:
                self.client.create_dataset(dataset, timeout=30)
            except gexc.Conflict:
                pass
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"LotAnalyzer dataset create failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LotAnalyzer dataset check failed: {exc}") from exc

    @staticmethod
    def _insight_to_row(insight: LotInsight) -> Dict[str, Any]:
        return {
            "generated_at": insight.generated_at,
            "pocket": insight.pocket,
            "side": insight.side,
            "trade_count": insight.trade_count,
            "win_rate": insight.win_rate,
            "profit_factor": insight.profit_factor,
            "avg_pips": insight.avg_pips,
            "std_pips": insight.std_pips,
            "total_pips": insight.total_pips,
            "avg_hold_minutes": insight.avg_hold_minutes,
            "score": insight.score,
            "risk_multiplier": insight.risk_multiplier,
            "confidence": insight.confidence,
        }

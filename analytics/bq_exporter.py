from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from google.cloud import bigquery

_DB_DEFAULT = "logs/trades.db"
_STATE_DEFAULT = "logs/bq_sync_state.json"
_BQ_PROJECT = os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
_BQ_DATASET = os.getenv("BQ_DATASET", "quantrabbit")
_BQ_TRADES_TABLE = os.getenv("BQ_TRADES_TABLE", "trades_raw")
_BQ_FEATURE_TABLE = os.getenv("BQ_FEATURE_TABLE", "trades_daily_features")
_BQ_MAX_EXPORT = int(os.getenv("BQ_MAX_EXPORT", "5000"))


@dataclass
class ExportStats:
    exported: int = 0
    last_updated_at: Optional[str] = None


class BigQueryExporter:
    """Export trades from SQLite to BigQuery and derive daily features."""

    def __init__(
        self,
        sqlite_path: str = _DB_DEFAULT,
        state_path: str = _STATE_DEFAULT,
        project_id: Optional[str] = _BQ_PROJECT,
        dataset_id: str = _BQ_DATASET,
        trades_table_id: str = _BQ_TRADES_TABLE,
        feature_table_id: str = _BQ_FEATURE_TABLE,
    ) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.state_path = Path(state_path)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.trades_table_id = trades_table_id
        self.feature_table_id = feature_table_id
        self.client = bigquery.Client(project=self.project_id) if self.project_id else bigquery.Client()
        self._ensure_dataset()
        self._ensure_trades_table()
        self._ensure_feature_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def export(self, limit: int = _BQ_MAX_EXPORT) -> ExportStats:
        """Export new trades into BigQuery; returns stats."""

        rows = self._fetch_trades(limit=limit)
        if not rows:
            logging.info("[BQ] no new trades to export")
            return ExportStats(exported=0, last_updated_at=self._get_last_cursor())

        table_ref = f"{self.dataset_id}.{self.trades_table_id}"
        logging.info("[BQ] exporting %s rows to %s", len(rows), table_ref)
        errors = self.client.insert_rows_json(table_ref, rows)
        if errors:
            raise RuntimeError(f"BigQuery insert failed: {errors}")

        last_updated_at = rows[-1]["updated_at"]
        self._set_last_cursor(last_updated_at)
        self._export_daily_features()
        logging.info("[BQ] export completed up to %s", last_updated_at)
        return ExportStats(exported=len(rows), last_updated_at=last_updated_at)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_dataset(self) -> None:
        dataset_ref = bigquery.Dataset(f"{self.client.project}.{self.dataset_id}")
        try:
            self.client.get_dataset(dataset_ref)
        except Exception:
            logging.info("[BQ] creating dataset %s", dataset_ref.dataset_id)
            dataset_ref.location = os.getenv("BQ_LOCATION", "US")
            self.client.create_dataset(dataset_ref, timeout=30)

    def _ensure_trades_table(self) -> None:
        table_ref = f"{self.client.project}.{self.dataset_id}.{self.trades_table_id}"
        schema = [
            bigquery.SchemaField("ticket_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("entry_time", "TIMESTAMP"),
            bigquery.SchemaField("close_time", "TIMESTAMP"),
            bigquery.SchemaField("pocket", "STRING"),
            bigquery.SchemaField("instrument", "STRING"),
            bigquery.SchemaField("units", "INTEGER"),
            bigquery.SchemaField("entry_price", "FLOAT"),
            bigquery.SchemaField("close_price", "FLOAT"),
            bigquery.SchemaField("pl_pips", "FLOAT"),
            bigquery.SchemaField("realized_pl", "FLOAT"),
            bigquery.SchemaField("state", "STRING"),
            bigquery.SchemaField("strategy", "STRING"),
            bigquery.SchemaField("macro_regime", "STRING"),
            bigquery.SchemaField("micro_regime", "STRING"),
            bigquery.SchemaField("close_reason", "STRING"),
            bigquery.SchemaField("version", "STRING"),
        ]
        try:
            self.client.get_table(table_ref)
        except Exception:
            logging.info("[BQ] creating trades table %s", table_ref)
            table = bigquery.Table(table_ref, schema=schema)
            table.time_partitioning = bigquery.TimePartitioning(field="close_time")
            table.clustering_fields = ["pocket", "strategy"]
            self.client.create_table(table)

    def _ensure_feature_table(self) -> None:
        table_ref = f"{self.client.project}.{self.dataset_id}.{self.feature_table_id}"
        schema = [
            bigquery.SchemaField("trade_date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("pocket", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("strategy", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("trades", "INTEGER"),
            bigquery.SchemaField("win_rate", "FLOAT"),
            bigquery.SchemaField("avg_pips", "FLOAT"),
            bigquery.SchemaField("pf", "FLOAT"),
            bigquery.SchemaField("updated_at", "TIMESTAMP"),
        ]
        try:
            self.client.get_table(table_ref)
        except Exception:
            logging.info("[BQ] creating feature table %s", table_ref)
            table = bigquery.Table(table_ref, schema=schema)
            self.client.create_table(table)

    def _fetch_trades(self, limit: int) -> List[Dict[str, Any]]:
        if not self.sqlite_path.exists():
            logging.warning("[BQ] sqlite file %s not found", self.sqlite_path)
            return []
        last_cursor = self._get_last_cursor()
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        sql = (
            "SELECT ticket_id, pocket, instrument, units, entry_price, close_price, pl_pips, "
            "realized_pl, entry_time, close_time, strategy, macro_regime, micro_regime, close_reason, "
            "state, version, updated_at "
            "FROM trades WHERE updated_at IS NOT NULL {cursor_clause} "
            "ORDER BY datetime(updated_at) ASC LIMIT ?"
        )
        cursor_clause = ""
        params: List[Any] = []
        if last_cursor:
            cursor_clause = "AND datetime(updated_at) > datetime(?)"
            params.append(last_cursor)
        sql = sql.format(cursor_clause=cursor_clause)
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        out: List[Dict[str, Any]] = []
        for row in rows:
            updated_at = _as_timestamp(row["updated_at"]) or datetime.now(timezone.utc)
            payload = {
                "ticket_id": row["ticket_id"],
                "updated_at": updated_at.isoformat(),
                "entry_time": _iso_or_none(row["entry_time"]),
                "close_time": _iso_or_none(row["close_time"]),
                "pocket": row["pocket"],
                "instrument": row["instrument"],
                "units": row["units"],
                "entry_price": row["entry_price"],
                "close_price": row["close_price"],
                "pl_pips": row["pl_pips"],
                "realized_pl": row["realized_pl"],
                "state": row["state"],
                "strategy": row["strategy"],
                "macro_regime": row["macro_regime"],
                "micro_regime": row["micro_regime"],
                "close_reason": row["close_reason"],
                "version": row["version"],
            }
            out.append(payload)
        return out

    def _export_daily_features(self) -> None:
        query = f"""
        INSERT INTO `{self.client.project}.{self.dataset_id}.{self.feature_table_id}`
        (trade_date, pocket, strategy, trades, win_rate, avg_pips, pf, updated_at)
        SELECT
          DATE(close_time) AS trade_date,
          pocket,
          strategy,
          COUNTIF(state = 'CLOSED') AS trades,
          SAFE_DIVIDE(SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END), COUNTIF(state = 'CLOSED')) AS win_rate,
          SAFE_DIVIDE(SUM(pl_pips), NULLIF(COUNTIF(state = 'CLOSED'), 0)) AS avg_pips,
          SAFE_DIVIDE(SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END),
                      NULLIF(ABS(SUM(CASE WHEN pl_pips < 0 THEN pl_pips ELSE 0 END)), 0)) AS pf,
          CURRENT_TIMESTAMP() AS updated_at
        FROM `{self.client.project}.{self.dataset_id}.{self.trades_table_id}`
        WHERE close_time IS NOT NULL AND close_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        GROUP BY trade_date, pocket, strategy
        """
        job = self.client.query(query)
        job.result()

    def _get_last_cursor(self) -> Optional[str]:
        if not self.state_path.exists():
            return None
        try:
            state = json.loads(self.state_path.read_text())
            return state.get("last_updated_at")
        except Exception:
            return None

    def _set_last_cursor(self, value: str) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({"last_updated_at": value}, indent=2))


def _iso_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    try:
        text = str(value)
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def _as_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    try:
        text = str(value)
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

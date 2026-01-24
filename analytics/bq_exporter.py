from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.api_core import exceptions as gexc
from google.cloud import bigquery

_DB_DEFAULT = "logs/trades.db"
_STATE_DEFAULT = "logs/bq_sync_state.json"
_BQ_PROJECT = os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
_BQ_DATASET = os.getenv("BQ_DATASET", "quantrabbit")
_BQ_TABLE = os.getenv("BQ_TRADES_TABLE", "trades_raw")
_BQ_MAX_EXPORT = int(os.getenv("BQ_MAX_EXPORT", "5000"))


@dataclass
class ExportStats:
    exported: int = 0
    last_updated_at: Optional[str] = None


class BigQueryExporter:
    """SQLite trades テーブルを BigQuery に同期する。"""

    def __init__(
        self,
        sqlite_path: str = _DB_DEFAULT,
        state_path: str = _STATE_DEFAULT,
        project_id: Optional[str] = _BQ_PROJECT,
        dataset_id: str = _BQ_DATASET,
        table_id: str = _BQ_TABLE,
    ) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.state_path = Path(state_path)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = (
            bigquery.Client(project=self.project_id)
            if self.project_id
            else bigquery.Client()
        )
        self._ensure_dataset()
        self._ensure_table()

    def export(self, limit: int = _BQ_MAX_EXPORT) -> ExportStats:
        rows = self._fetch_trades(limit)
        if not rows:
            logging.info("[BQ] new rowsなし")
            return ExportStats(exported=0, last_updated_at=self._get_last_cursor())

        table_ref = f"{self.client.project}.{self.dataset_id}.{self.table_id}"
        logging.info("[BQ] %s 件を %s へ送信中...", len(rows), table_ref)
        errors = self.client.insert_rows_json(table_ref, rows)
        if errors:
            raise RuntimeError(f"BigQuery insert failed: {errors}")

        last_updated = rows[-1]["updated_at"]
        self._set_last_cursor(last_updated)
        logging.info("[BQ] export 完了 (last_updated=%s)", last_updated)
        return ExportStats(exported=len(rows), last_updated_at=last_updated)

    # -------------------- internal helpers --------------------

    def _ensure_dataset(self) -> None:
        dataset_ref = bigquery.DatasetReference(self.client.project, self.dataset_id)
        try:
            self.client.get_dataset(dataset_ref)
        except gexc.NotFound:
            logging.info("[BQ] dataset %s を作成します", dataset_ref.dataset_id)
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = os.getenv("BQ_LOCATION", "US")
            try:
                self.client.create_dataset(dataset, timeout=30)
            except gexc.Conflict:
                logging.info("[BQ] dataset %s は既に存在します", dataset_ref.dataset_id)

    def _ensure_table(self) -> None:
        table_ref = f"{self.client.project}.{self.dataset_id}.{self.table_id}"
        schema = [
            bigquery.SchemaField("ticket_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("transaction_id", "INTEGER"),
            bigquery.SchemaField("entry_time", "TIMESTAMP"),
            bigquery.SchemaField("open_time", "TIMESTAMP"),
            bigquery.SchemaField("close_time", "TIMESTAMP"),
            bigquery.SchemaField("pocket", "STRING"),
            bigquery.SchemaField("instrument", "STRING"),
            bigquery.SchemaField("units", "INTEGER"),
            bigquery.SchemaField("closed_units", "INTEGER"),
            bigquery.SchemaField("entry_price", "FLOAT"),
            bigquery.SchemaField("close_price", "FLOAT"),
            bigquery.SchemaField("fill_price", "FLOAT"),
            bigquery.SchemaField("pl_pips", "FLOAT"),
            bigquery.SchemaField("realized_pl", "FLOAT"),
            bigquery.SchemaField("commission", "FLOAT"),
            bigquery.SchemaField("financing", "FLOAT"),
            bigquery.SchemaField("state", "STRING"),
            bigquery.SchemaField("close_reason", "STRING"),
            bigquery.SchemaField("version", "STRING"),
            bigquery.SchemaField("unrealized_pl", "FLOAT"),
            bigquery.SchemaField("strategy", "STRING"),
            bigquery.SchemaField("strategy_tag", "STRING"),
            bigquery.SchemaField("client_order_id", "STRING"),
            bigquery.SchemaField("entry_thesis", "STRING"),
            bigquery.SchemaField("macro_regime", "STRING"),
            bigquery.SchemaField("micro_regime", "STRING"),
        ]
        try:
            table = self.client.get_table(table_ref)
            existing = {field.name for field in table.schema}
            missing = [field for field in schema if field.name not in existing]
            if missing:
                table.schema = table.schema + missing
                self.client.update_table(table, ["schema"])
                logging.info(
                    "[BQ] table %s schema updated (+%s)",
                    table_ref,
                    ", ".join(field.name for field in missing),
                )
        except gexc.NotFound:
            logging.info("[BQ] table %s を作成します", table_ref)
            table = bigquery.Table(table_ref, schema=schema)
            table.time_partitioning = bigquery.TimePartitioning(field="close_time")
            table.clustering_fields = ["pocket"]
            self.client.create_table(table)

    def _fetch_trades(self, limit: int) -> List[Dict[str, Any]]:
        if not self.sqlite_path.exists():
            logging.warning("[BQ] SQLite %s が存在しません", self.sqlite_path)
            return []
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        last_cursor = self._get_last_cursor()

        where_clause = "WHERE updated_at IS NOT NULL"
        params: List[Any] = []
        if last_cursor:
            where_clause += " AND datetime(updated_at) > datetime(?)"
            params.append(last_cursor)

        sql = f"""
SELECT transaction_id, ticket_id, updated_at, entry_time, open_time, close_time,
       pocket, instrument, units, closed_units, entry_price, close_price, fill_price,
       pl_pips, realized_pl, commission, financing, state, close_reason, version,
       unrealized_pl, strategy, strategy_tag, client_order_id, entry_thesis,
       macro_regime, micro_regime
FROM trades
{where_clause}
ORDER BY datetime(updated_at) ASC
LIMIT ?
"""
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        payload: List[Dict[str, Any]] = []
        for row in rows:
            ticket_id = row["ticket_id"]
            if not ticket_id:
                continue
            updated_at = self._as_timestamp(row["updated_at"])
            payload.append(
                {
                    "ticket_id": ticket_id,
                    "updated_at": updated_at.isoformat() if updated_at else None,
                    "transaction_id": row["transaction_id"],
                    "entry_time": self._iso(row["entry_time"]),
                    "open_time": self._iso(row["open_time"]),
                    "close_time": self._iso(row["close_time"]),
                    "pocket": row["pocket"],
                    "instrument": row["instrument"],
                    "units": row["units"],
                    "closed_units": row["closed_units"],
                    "entry_price": row["entry_price"],
                    "close_price": row["close_price"],
                    "fill_price": row["fill_price"],
                    "pl_pips": row["pl_pips"],
                    "realized_pl": row["realized_pl"],
                    "commission": row["commission"],
                    "financing": row["financing"],
                    "state": row["state"],
                    "close_reason": row["close_reason"],
                    "version": row["version"],
                    "unrealized_pl": row["unrealized_pl"],
                    "strategy": row["strategy"],
                    "strategy_tag": row["strategy_tag"],
                    "client_order_id": row["client_order_id"],
                    "entry_thesis": row["entry_thesis"],
                    "macro_regime": row["macro_regime"],
                    "micro_regime": row["micro_regime"],
                }
            )
        return [r for r in payload if r["updated_at"] is not None]

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

    @staticmethod
    def _iso(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            if isinstance(value, datetime):
                dt = value
            else:
                text = str(value)
                if not text:
                    return None
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            return None

    @staticmethod
    def _as_timestamp(value: Any) -> Optional[datetime]:
        iso = BigQueryExporter._iso(value)
        if not iso:
            return None
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))

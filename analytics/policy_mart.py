from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from google.api_core import exceptions as gexc
from google.cloud import bigquery

DEFAULT_DATASET = os.getenv("BQ_DATASET", "quantrabbit")
DEFAULT_PROJECT = os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
DEFAULT_TABLE = os.getenv("BQ_TRADES_TABLE", "trades_raw")

DEFAULT_TIMEZONE = os.getenv("POLICY_TIMEZONE", "Asia/Tokyo")
_VOL_BUCKETS_RAW = os.getenv("POLICY_VOL_BUCKETS", "2,4,6")


def _parse_thresholds(raw: str) -> List[float]:
    out: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue
    return sorted(out)


def _build_vol_bucket_expr(atr_expr: str, thresholds: Sequence[float]) -> str:
    atr_expr = atr_expr.strip()
    if atr_expr.upper() == "NULL":
        return "'unknown'"
    if not thresholds:
        return f"CASE WHEN {atr_expr} IS NULL THEN 'unknown' ELSE 'all' END"
    parts = [f"WHEN {atr_expr} IS NULL THEN 'unknown'"]
    prev = 0.0
    for idx, th in enumerate(thresholds):
        label = f"{int(prev)}-{int(th)}"
        parts.append(f"WHEN {atr_expr} < {th} THEN '{label}'")
        prev = th
    label = f"{int(prev)}+"
    parts.append(f"ELSE '{label}'")
    return "CASE " + " ".join(parts) + " END"


class PolicyMartClient:
    def __init__(
        self,
        *,
        project_id: Optional[str] = DEFAULT_PROJECT,
        dataset_id: str = DEFAULT_DATASET,
        trades_table: str = DEFAULT_TABLE,
        timezone: str = DEFAULT_TIMEZONE,
        vol_thresholds: Optional[Sequence[float]] = None,
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.trades_table = trades_table
        self.timezone = timezone or DEFAULT_TIMEZONE
        self.vol_thresholds = list(vol_thresholds or _parse_thresholds(_VOL_BUCKETS_RAW))
        self.client = bigquery.Client(project=self.project_id) if self.project_id else bigquery.Client()
        self._table_schema = self._get_table_schema()

    def _get_table_schema(self) -> Dict[str, bigquery.SchemaField]:
        table_ref = f"{self.client.project}.{self.dataset_id}.{self.trades_table}"
        try:
            table = self.client.get_table(table_ref)
            return {field.name: field for field in table.schema}
        except Exception as exc:
            logging.warning("[POLICY_MART] schema fetch failed: %s", exc)
            return {}

    def _has_column(self, name: str) -> bool:
        return name in self._table_schema

    def _strategy_expr(self) -> str:
        if self._has_column("strategy") and self._has_column("strategy_tag"):
            return "COALESCE(SAFE_CAST(strategy AS STRING), SAFE_CAST(strategy_tag AS STRING), 'unknown')"
        if self._has_column("strategy"):
            return "COALESCE(SAFE_CAST(strategy AS STRING), 'unknown')"
        if self._has_column("strategy_tag"):
            return "COALESCE(SAFE_CAST(strategy_tag AS STRING), 'unknown')"
        return "'unknown'"

    def _regime_expr(self) -> str:
        if self._has_column("regime"):
            return "COALESCE(SAFE_CAST(regime AS STRING), 'unknown')"
        if self._has_column("entry_thesis"):
            return (
                "COALESCE(JSON_EXTRACT_SCALAR(CAST(entry_thesis AS STRING), '$.regime'), 'unknown')"
            )
        return "'unknown'"

    def _atr_expr(self) -> str:
        if self._has_column("atr_pips"):
            return "SAFE_CAST(atr_pips AS FLOAT64)"
        if self._has_column("entry_thesis"):
            return (
                "SAFE_CAST(COALESCE("
                "JSON_EXTRACT_SCALAR(CAST(entry_thesis AS STRING), '$.atr_pips'),"
                "JSON_EXTRACT_SCALAR(CAST(entry_thesis AS STRING), '$.atr_entry'),"
                "JSON_EXTRACT_SCALAR(CAST(entry_thesis AS STRING), '$.atr'),"
                "JSON_EXTRACT_SCALAR(CAST(entry_thesis AS STRING), '$.atr_m1')"
                ") AS FLOAT64)"
            )
        return "SAFE_CAST(NULL AS FLOAT64)"

    def _time_band_expr(self) -> str:
        time_src = "COALESCE(close_time, entry_time, updated_at)"
        hour_expr = f"EXTRACT(HOUR FROM DATETIME({time_src}, '{self.timezone}'))"
        band_id = f"CAST(FLOOR({hour_expr} / 4) AS INT64)"
        start_hour = f"({band_id} * 4)"
        end_hour = f"({band_id} * 4 + 3)"
        return (
            "CONCAT("
            f"LPAD(CAST({start_hour} AS STRING), 2, '0'),"
            "'-',"
            f"LPAD(CAST({end_hour} AS STRING), 2, '0')"
            ")"
        )

    def build_query(
        self,
        *,
        lookback_days: int = 14,
        min_trades: int = 0,
        inline_params: bool = False,
    ) -> tuple[str, List[bigquery.ScalarQueryParameter]]:
        table_ref = f"{self.client.project}.{self.dataset_id}.{self.trades_table}"
        strategy_expr = self._strategy_expr()
        regime_expr = self._regime_expr()
        atr_expr = self._atr_expr()
        time_band_expr = self._time_band_expr()
        vol_bucket_expr = _build_vol_bucket_expr(atr_expr, self.vol_thresholds)
        pocket_expr = "COALESCE(SAFE_CAST(pocket AS STRING), 'unknown')"

        lookback_expr = f"{int(lookback_days)}"
        lookback_param = "@lookback_days"
        if inline_params:
            lookback_param = lookback_expr

        sql = f"""
WITH base AS (
  SELECT
    {pocket_expr} AS pocket,
    {strategy_expr} AS strategy,
    {regime_expr} AS regime,
    {time_band_expr} AS time_band,
    {vol_bucket_expr} AS vol_bucket,
    SAFE_CAST(pl_pips AS FLOAT64) AS pl_pips,
    entry_time,
    close_time
  FROM `{table_ref}`
  WHERE state = 'CLOSED'
    AND close_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_param} DAY)
)
SELECT
  pocket,
  strategy,
  regime,
  time_band,
  vol_bucket,
  COUNT(*) AS trade_count,
  SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS wins,
  SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS gross_profit,
  SUM(CASE WHEN pl_pips < 0 THEN ABS(pl_pips) ELSE 0 END) AS gross_loss,
  SUM(pl_pips) AS total_pips,
  AVG(pl_pips) AS avg_pips,
  AVG(
    CASE
      WHEN entry_time IS NOT NULL AND close_time IS NOT NULL THEN
        TIMESTAMP_DIFF(close_time, entry_time, MINUTE)
    END
  ) AS avg_hold_minutes
FROM base
GROUP BY pocket, strategy, regime, time_band, vol_bucket
"""
        if min_trades > 0:
            if inline_params:
                sql += f"\nHAVING trade_count >= {int(min_trades)}\n"
            else:
                sql += "\nHAVING trade_count >= @min_trades\n"
        sql += "ORDER BY pocket, strategy, regime, time_band, vol_bucket\n"

        params: List[bigquery.ScalarQueryParameter] = []
        if not inline_params:
            params.append(bigquery.ScalarQueryParameter("lookback_days", "INT64", int(lookback_days)))
            if min_trades > 0:
                params.append(bigquery.ScalarQueryParameter("min_trades", "INT64", int(min_trades)))
        return sql, params

    def fetch_rows(self, *, lookback_days: int = 14, min_trades: int = 0) -> List[Dict[str, Any]]:
        sql, params = self.build_query(lookback_days=lookback_days, min_trades=min_trades)
        try:
            job = self.client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
            return [dict(row) for row in job.result()]
        except gexc.NotFound:
            logging.warning("[POLICY_MART] trades table not found.")
            return []
        except Exception as exc:
            logging.warning("[POLICY_MART] query failed: %s", exc)
            return []

    def create_view(self, *, view_name: str = "policy_mart_view", lookback_days: int = 14, min_trades: int = 0) -> None:
        sql, _ = self.build_query(lookback_days=lookback_days, min_trades=min_trades, inline_params=True)
        view_ref = f"{self.client.project}.{self.dataset_id}.{view_name}"
        query = f"CREATE OR REPLACE VIEW `{view_ref}` AS {sql}"
        try:
            self.client.query(query).result()
            logging.info("[POLICY_MART] view created: %s", view_ref)
        except Exception as exc:
            logging.warning("[POLICY_MART] view create failed: %s", exc)

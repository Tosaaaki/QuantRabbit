"""Aggregate entry_thesis factors into a daily BigQuery table."""
from __future__ import annotations

import argparse
import logging
import os

from google.api_core import exceptions as gexc
from google.cloud import bigquery


DEFAULT_PROJECT = os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
DEFAULT_DATASET = os.getenv("BQ_DATASET", "quantrabbit")
DEFAULT_TABLE = os.getenv("BQ_ENTRY_THESIS_DAILY_TABLE", "entry_thesis_daily_causes")
DEFAULT_VIEW = os.getenv("BQ_ENTRY_THESIS_STRUCT_VIEW", "entry_thesis_struct_view")
DEFAULT_TIMEZONE = os.getenv("ENTRY_THESIS_TIMEZONE", "Asia/Tokyo")
DEFAULT_LOOKBACK_DAYS = int(os.getenv("ENTRY_THESIS_DAILY_LOOKBACK_DAYS", "1"))
DEFAULT_MIN_TRADES = int(os.getenv("ENTRY_THESIS_DAILY_MIN_TRADES", "5"))


def _ensure_table(client: bigquery.Client, dataset_id: str, table_id: str) -> None:
    dataset_ref = bigquery.DatasetReference(client.project, dataset_id)
    try:
        client.get_dataset(dataset_ref)
    except gexc.NotFound:
        logging.info("[BQ] dataset %s missing. Creating...", dataset_id)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = os.getenv("BQ_LOCATION", "US")
        client.create_dataset(dataset, exists_ok=True)

    table_ref = dataset_ref.table(table_id)
    schema = [
        bigquery.SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("day_jst", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("pocket", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("strategy_key", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("cause_dim", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("cause_value", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("trade_count", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("win_rate", "FLOAT64"),
        bigquery.SchemaField("profit_factor", "FLOAT64"),
        bigquery.SchemaField("total_pips", "FLOAT64"),
        bigquery.SchemaField("avg_pips", "FLOAT64"),
        bigquery.SchemaField("gross_profit", "FLOAT64"),
        bigquery.SchemaField("gross_loss", "FLOAT64"),
        bigquery.SchemaField("avg_hold_minutes", "FLOAT64"),
    ]
    try:
        client.get_table(table_ref)
    except gexc.NotFound:
        logging.info("[BQ] creating table %s", table_ref.path)
        table = bigquery.Table(table_ref, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(field="day_jst")
        client.create_table(table)


def _ensure_view(client: bigquery.Client, dataset_id: str, view_name: str) -> None:
    view_ref = f"{client.project}.{dataset_id}.{view_name}"
    try:
        client.get_table(view_ref)
    except gexc.NotFound as exc:
        raise RuntimeError(f"BigQuery view missing: {view_ref}") from exc


def _run_job(
    client: bigquery.Client,
    dataset_id: str,
    view_name: str,
    table_id: str,
    timezone: str,
    lookback_days: int,
    min_trades: int,
) -> None:
    view_ref = f"`{client.project}.{dataset_id}.{view_name}`"
    table_ref = f"`{client.project}.{dataset_id}.{table_id}`"

    query = f"""
DECLARE end_date DATE DEFAULT CURRENT_DATE('{timezone}');
DECLARE start_date DATE DEFAULT DATE_SUB(end_date, INTERVAL @lookback_days DAY);

DELETE FROM {table_ref}
WHERE day_jst >= start_date AND day_jst < end_date;

INSERT INTO {table_ref} (
  generated_at,
  day_jst,
  pocket,
  strategy_key,
  cause_dim,
  cause_value,
  trade_count,
  win_rate,
  profit_factor,
  total_pips,
  avg_pips,
  gross_profit,
  gross_loss,
  avg_hold_minutes
)
WITH base AS (
  SELECT
    DATE(close_time, '{timezone}') AS day_jst,
    COALESCE(pocket, 'unknown') AS pocket,
    COALESCE(strategy_key, 'unknown') AS strategy_key,
    SAFE_CAST(pl_pips AS FLOAT64) AS pl_pips,
    entry_time,
    close_time,
    entry_hour_jst,
    entry_range_bucket,
    atr_bucket,
    trend_bias,
    pattern_tag,
    flags
  FROM {view_ref}
  WHERE close_time >= TIMESTAMP(start_date, '{timezone}')
    AND close_time < TIMESTAMP(end_date, '{timezone}')
    AND state = 'CLOSED'
),
labeled AS (
  SELECT
    day_jst,
    pocket,
    strategy_key,
    'entry_range_bucket' AS cause_dim,
    COALESCE(entry_range_bucket, 'unknown') AS cause_value,
    pl_pips,
    entry_time,
    close_time
  FROM base
  UNION ALL
  SELECT
    day_jst,
    pocket,
    strategy_key,
    'entry_hour' AS cause_dim,
    COALESCE(LPAD(CAST(entry_hour_jst AS STRING), 2, '0'), 'unknown') AS cause_value,
    pl_pips,
    entry_time,
    close_time
  FROM base
  UNION ALL
  SELECT
    day_jst,
    pocket,
    strategy_key,
    'atr_bucket' AS cause_dim,
    COALESCE(atr_bucket, 'unknown') AS cause_value,
    pl_pips,
    entry_time,
    close_time
  FROM base
  UNION ALL
  SELECT
    day_jst,
    pocket,
    strategy_key,
    'pattern_tag' AS cause_dim,
    COALESCE(NULLIF(pattern_tag, ''), 'unknown') AS cause_value,
    pl_pips,
    entry_time,
    close_time
  FROM base
  UNION ALL
  SELECT
    day_jst,
    pocket,
    strategy_key,
    'trend_bias' AS cause_dim,
    CAST(IFNULL(trend_bias, FALSE) AS STRING) AS cause_value,
    pl_pips,
    entry_time,
    close_time
  FROM base
  UNION ALL
  SELECT
    day_jst,
    pocket,
    strategy_key,
    'flag' AS cause_dim,
    COALESCE(NULLIF(flag, ''), 'unknown') AS cause_value,
    pl_pips,
    entry_time,
    close_time
  FROM base,
  UNNEST(IFNULL(flags, [])) AS flag
)
SELECT
  CURRENT_TIMESTAMP() AS generated_at,
  day_jst,
  pocket,
  strategy_key,
  cause_dim,
  cause_value,
  COUNT(*) AS trade_count,
  AVG(IF(pl_pips > 0, 1.0, 0.0)) AS win_rate,
  SAFE_DIVIDE(
    SUM(IF(pl_pips > 0, pl_pips, 0.0)),
    NULLIF(SUM(IF(pl_pips < 0, ABS(pl_pips), 0.0)), 0.0)
  ) AS profit_factor,
  SUM(pl_pips) AS total_pips,
  AVG(pl_pips) AS avg_pips,
  SUM(IF(pl_pips > 0, pl_pips, 0.0)) AS gross_profit,
  SUM(IF(pl_pips < 0, ABS(pl_pips), 0.0)) AS gross_loss,
  AVG(TIMESTAMP_DIFF(close_time, entry_time, MINUTE)) AS avg_hold_minutes
FROM labeled
GROUP BY day_jst, pocket, strategy_key, cause_dim, cause_value
HAVING trade_count >= @min_trades
"""

    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("lookback_days", "INT64", lookback_days),
                bigquery.ScalarQueryParameter("min_trades", "INT64", min_trades),
            ]
        ),
    )
    job.result()


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily entry_thesis cause analysis job.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--view", default=DEFAULT_VIEW)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    client = bigquery.Client(project=DEFAULT_PROJECT) if DEFAULT_PROJECT else bigquery.Client()

    _ensure_table(client, args.dataset, args.table)
    _ensure_view(client, args.dataset, args.view)
    _run_job(
        client,
        args.dataset,
        args.view,
        args.table,
        args.timezone,
        max(1, args.lookback_days),
        max(1, args.min_trades),
    )
    logging.info("[entry_thesis_daily] completed")


if __name__ == "__main__":
    main()

"""Compute short-horizon trading KPIs and persist them to BigQuery.

This job is intended to run on a short cadence (5-15 minutes) via Cloud Scheduler
or any other orchestration layer. It reads recent executions from the
`trades_raw` table, calculates core metrics such as win rate, profit factor, and
drawdown and stores the result in the `realtime_metrics` table that the trading
bot can subscribe to.

The job keeps the metrics table lean by pruning entries that are older than the
retention horizon. All configuration knobs can be overridden through
environment variables so the same code can run in local development, Cloud Run
or Composer.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from typing import Iterable, List

from google.api_core import exceptions as gexc
from google.cloud import bigquery


DEFAULT_DATASET = os.getenv("BQ_DATASET", "quantrabbit")
DEFAULT_PROJECT = os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
DEFAULT_TABLE = os.getenv("BQ_REALTIME_METRICS_TABLE", "realtime_metrics")
DEFAULT_LOOKBACK_MIN = int(os.getenv("REALTIME_METRICS_LOOKBACK_MIN", "720"))
DEFAULT_RETENTION_HOURS = int(os.getenv("REALTIME_METRICS_RETENTION_H", "168"))


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
        bigquery.SchemaField("pocket", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("strategy", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("total_trades", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("win_rate", "FLOAT64"),
        bigquery.SchemaField("profit_factor", "FLOAT64"),
        bigquery.SchemaField("avg_hold_minutes", "FLOAT64"),
        bigquery.SchemaField("total_pips", "FLOAT64"),
        bigquery.SchemaField("max_drawdown_pips", "FLOAT64"),
        bigquery.SchemaField("losing_streak", "INT64"),
    ]
    try:
        client.get_table(table_ref)
    except gexc.NotFound:
        logging.info("[BQ] creating table %s", table_ref.path)
        client.create_table(bigquery.Table(table_ref, schema=schema))


def _fetch_metrics(
    client: bigquery.Client,
    dataset_id: str,
    lookback_minutes: int,
) -> List[bigquery.table.Row]:
    query = f"""
WITH recent AS (
  SELECT
    pocket,
    COALESCE(strategy, 'unknown') AS strategy,
    close_time,
    entry_time,
    pl_pips,
    CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END AS won,
    SUM(pl_pips) OVER (
      PARTITION BY pocket, COALESCE(strategy, 'unknown')
      ORDER BY close_time
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS equity_curve
  FROM `{client.project}.{dataset_id}.trades_raw`
  WHERE close_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback MINUTE)
    AND state = 'CLOSED'
)
, agg AS (
  SELECT
    pocket,
    strategy,
    COUNT(*) AS total_trades,
    AVG(won) AS win_rate,
    SAFE_DIVIDE(
      SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END),
      NULLIF(ABS(SUM(CASE WHEN pl_pips < 0 THEN pl_pips ELSE 0 END)), 0)
    ) AS profit_factor,
    AVG(TIMESTAMP_DIFF(close_time, entry_time, SECOND)) / 60.0 AS avg_hold_minutes,
    SUM(pl_pips) AS total_pips
  FROM recent
  GROUP BY pocket, strategy
)
, dd AS (
  SELECT
    pocket,
    strategy,
    MAX(drawdown) AS max_drawdown_pips
  FROM (
    SELECT
      pocket,
      strategy,
      MAX(equity_curve) OVER (
        PARTITION BY pocket, strategy
        ORDER BY close_time
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) - equity_curve AS drawdown
    FROM recent
  )
  GROUP BY pocket, strategy
)
, streak_base AS (
  SELECT
    pocket,
    strategy,
    close_time,
    won,
    SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) OVER (
      PARTITION BY pocket, strategy
      ORDER BY close_time
    ) AS win_group
  FROM recent
)
, streaks AS (
  SELECT
    pocket,
    strategy,
    MAX(loss_seq) AS losing_streak
  FROM (
    SELECT
      pocket,
      strategy,
      CASE
        WHEN won = 0 THEN
          ROW_NUMBER() OVER (PARTITION BY pocket, strategy ORDER BY close_time)
          - ROW_NUMBER() OVER (
              PARTITION BY pocket, strategy, win_group
              ORDER BY close_time
            ) + 1
        ELSE 0
      END AS loss_seq
    FROM streak_base
  )
  GROUP BY pocket, strategy
)
SELECT
  TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MINUTE) AS generated_at,
  agg.pocket,
  agg.strategy,
  agg.total_trades,
  agg.win_rate,
  agg.profit_factor,
  agg.avg_hold_minutes,
  agg.total_pips,
  COALESCE(dd.max_drawdown_pips, 0) AS max_drawdown_pips,
  COALESCE(streaks.losing_streak, 0) AS losing_streak
FROM agg
LEFT JOIN dd USING (pocket, strategy)
LEFT JOIN streaks USING (pocket, strategy)
ORDER BY pocket, strategy
"""

    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "lookback", "INT64", lookback_minutes
                )
            ]
        ),
    )
    return list(job.result())


def _insert_metrics(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    rows: Iterable[bigquery.table.Row],
) -> int:
    if not rows:
        return 0

    table_ref = client.dataset(dataset_id).table(table_id)
    payload = []
    for row in rows:
        payload.append(
            {
                "generated_at": row["generated_at"].isoformat(),
                "pocket": row["pocket"],
                "strategy": row["strategy"],
                "total_trades": row["total_trades"],
                "win_rate": row["win_rate"],
                "profit_factor": row["profit_factor"],
                "avg_hold_minutes": row["avg_hold_minutes"],
                "total_pips": row["total_pips"],
                "max_drawdown_pips": row["max_drawdown_pips"],
                "losing_streak": row["losing_streak"],
            }
        )

    errors = client.insert_rows_json(table_ref, payload)
    if errors:
        raise RuntimeError(f"BigQuery insert failed: {errors}")
    return len(payload)


def _prune_old_rows(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    retention_hours: int,
) -> None:
    query = f"""
DELETE FROM `{client.project}.{dataset_id}.{table_id}`
WHERE generated_at < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @retention HOUR)
"""
    client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "retention", "INT64", retention_hours
                )
            ]
        ),
    ).result()


def run(
    project: str | None = None,
    dataset: str = DEFAULT_DATASET,
    table: str = DEFAULT_TABLE,
    lookback_minutes: int = DEFAULT_LOOKBACK_MIN,
    retention_hours: int = DEFAULT_RETENTION_HOURS,
) -> None:
    logging.info(
        "[REALTIME] start (dataset=%s table=%s lookback=%s retention=%sh)",
        dataset,
        table,
        lookback_minutes,
        retention_hours,
    )
    client = bigquery.Client(project=project) if project else bigquery.Client()
    _ensure_table(client, dataset, table)
    rows = _fetch_metrics(client, dataset, lookback_minutes)
    inserted = _insert_metrics(client, dataset, table, rows)
    _prune_old_rows(client, dataset, table, retention_hours)
    logging.info("[REALTIME] completed rows=%s", inserted)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime KPI job")
    parser.add_argument("--project", help="GCP project override", default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_MIN)
    parser.add_argument("--retention", type=int, default=DEFAULT_RETENTION_HOURS)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(
        project=args.project,
        dataset=args.dataset,
        table=args.table,
        lookback_minutes=args.lookback,
        retention_hours=args.retention,
    )


if __name__ == "__main__":
    main()

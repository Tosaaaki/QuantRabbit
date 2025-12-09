#!/usr/bin/env python3
"""
BQ 上のローソク（例: candles_m1）を集計し、価格帯ごとの反転/到達傾向マップを JSON スナップショットとして出力する。
用途: 事前計算した水準マップを GCS/Firestore に置き、VM は TTL キャッシュで参照する前提。
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.cloud import bigquery, storage


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _run_query(
    client: bigquery.Client,
    *,
    dataset: str,
    table: str,
    lookback_days: int,
    bucket_pips: float,
    retrace_pips: float,
    min_hits: int,
) -> List[Dict[str, Any]]:
    bucket_size = bucket_pips * 0.01
    retrace = retrace_pips * 0.01
    table_ref = f"{client.project}.{dataset}.{table}"
    sql = f"""
DECLARE bucket_size FLOAT64 DEFAULT @bucket_size;
DECLARE retrace FLOAT64 DEFAULT @retrace;

WITH base AS (
  SELECT
    TIMESTAMP(ts) AS ts,
    SAFE_CAST(close AS FLOAT64) AS close,
    SAFE_CAST(high AS FLOAT64) AS high,
    SAFE_CAST(low AS FLOAT64) AS low,
    ROUND(SAFE_DIVIDE(SAFE_CAST(close AS FLOAT64), bucket_size)) * bucket_size AS bucket,
    LEAD(SAFE_CAST(close AS FLOAT64), 5) OVER (ORDER BY ts) AS close_5,
    LEAD(SAFE_CAST(close AS FLOAT64), 15) OVER (ORDER BY ts) AS close_15,
    LEAD(SAFE_CAST(close AS FLOAT64), 30) OVER (ORDER BY ts) AS close_30,
    MAX(SAFE_CAST(high AS FLOAT64)) OVER (ORDER BY ts ROWS BETWEEN CURRENT ROW AND 14 FOLLOWING) AS max_high_15,
    MIN(SAFE_CAST(low AS FLOAT64)) OVER (ORDER BY ts ROWS BETWEEN CURRENT ROW AND 14 FOLLOWING) AS min_low_15
  FROM `{table_ref}`
  WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY)
)

SELECT
  bucket,
  COUNT(*) AS hit_count,
  AVG(close_5 - close) AS mean_ret_5,
  AVG(close_15 - close) AS mean_ret_15,
  AVG(close_30 - close) AS mean_ret_30,
  AVG(CAST(close_5 > close AS INT64)) AS p_up_5,
  AVG(CAST(close_5 < close AS INT64)) AS p_down_5,
  AVG(CAST(close_15 > close AS INT64)) AS p_up_15,
  AVG(CAST(close_15 < close AS INT64)) AS p_down_15,
  AVG(max_high_15 - close) AS avg_up_range_15,
  AVG(close - min_low_15) AS avg_down_range_15,
  AVG(CAST((max_high_15 - close) >= retrace AS INT64)) AS p_up_retrace_15,
  AVG(CAST((close - min_low_15) >= retrace AS INT64)) AS p_down_retrace_15
FROM base
WHERE bucket IS NOT NULL
GROUP BY bucket
HAVING COUNT(*) >= @min_hits
ORDER BY bucket
"""
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("lookback_days", "INT64", lookback_days),
            bigquery.ScalarQueryParameter("bucket_size", "FLOAT64", bucket_size),
            bigquery.ScalarQueryParameter("retrace", "FLOAT64", retrace),
            bigquery.ScalarQueryParameter("min_hits", "INT64", min_hits),
        ]
    )
    job = client.query(sql, job_config=job_config)
    return [dict(row) for row in job.result()]


def _write_gcs(bucket: str, obj: str, text: str, project: Optional[str]) -> None:
    client = storage.Client(project=project)
    blob = client.bucket(bucket).blob(obj)
    blob.upload_from_string(text, content_type="application/json")
    logging.info("[GCS] uploaded to gs://%s/%s", bucket, obj)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="quantrabbit")
    parser.add_argument("--table", type=str, default="candles_m1")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--bucket-pips", type=float, default=5.0, help="バケット幅（pips単位, 1 pip=0.01)")
    parser.add_argument("--retrace-pips", type=float, default=10.0, help="到達判定のしきい値（pips）")
    parser.add_argument("--min-hits", type=int, default=30, help="この件数未満のバケットは除外")
    parser.add_argument("--json-out", type=Path, default=Path("/tmp/level_map.json"))
    parser.add_argument("--gcs-bucket", type=str, default=None)
    parser.add_argument("--gcs-object", type=str, default="analytics/level_map.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _setup_logging(args.verbose)

    client = bigquery.Client(project=args.project)
    try:
        rows = _run_query(
            client,
            dataset=args.dataset,
            table=args.table,
            lookback_days=args.lookback_days,
            bucket_pips=args.bucket_pips,
            retrace_pips=args.retrace_pips,
            min_hits=args.min_hits,
        )
    except Exception as exc:  # noqa: BLE001
        logging.error("[BQ] query failed: %s", exc)
        return 1

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "project": client.project,
        "dataset": args.dataset,
        "table": args.table,
        "lookback_days": args.lookback_days,
        "bucket_pips": args.bucket_pips,
        "retrace_pips": args.retrace_pips,
        "min_hits": args.min_hits,
        "levels": rows,
        "count": len(rows),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    logging.info("[WRITE] %d buckets -> %s", len(rows), args.json_out)

    if args.gcs_bucket:
        try:
            _write_gcs(args.gcs_bucket, args.gcs_object, args.json_out.read_text(), args.project)
        except Exception as exc:  # noqa: BLE001
            logging.error("[GCS] upload failed: %s", exc)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

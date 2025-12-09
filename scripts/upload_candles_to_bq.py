#!/usr/bin/env python3
"""
ローカルのローソク JSON（例: logs/oanda/candles_M1_latest.json）を BigQuery にロードするユーティリティ。
ストリーミングではなくバッチ前提。M1 だけでなく timeframe を指定して使える。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from google.cloud import bigquery


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _iter_rows(path: Path, timeframe: str, instrument: str) -> Iterable[Dict[str, Any]]:
    data = json.loads(path.read_text())
    candles = data.get("candles", data if isinstance(data, list) else [])
    fetched_at = data.get("fetched_at")
    for c in candles:
        ts = c.get("time") or c.get("timestamp") or c.get("ts")
        yield {
            "ts": ts,
            "open": c.get("open"),
            "high": c.get("high"),
            "low": c.get("low"),
            "close": c.get("close"),
            "volume": c.get("volume"),
            "timeframe": c.get("timeframe") or timeframe,
            "instrument": c.get("instrument") or instrument,
            "source": path.name,
            "fetched_at": fetched_at,
        }


def _ensure_table(client: bigquery.Client, dataset: str, table: str) -> bigquery.Table:
    table_ref = f"{client.project}.{dataset}.{table}"
    schema = [
        bigquery.SchemaField("ts", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("open", "FLOAT"),
        bigquery.SchemaField("high", "FLOAT"),
        bigquery.SchemaField("low", "FLOAT"),
        bigquery.SchemaField("close", "FLOAT"),
        bigquery.SchemaField("volume", "INTEGER"),
        bigquery.SchemaField("timeframe", "STRING"),
        bigquery.SchemaField("instrument", "STRING"),
        bigquery.SchemaField("source", "STRING"),
        bigquery.SchemaField("fetched_at", "TIMESTAMP"),
    ]
    try:
        table_obj = client.get_table(table_ref)
        return table_obj
    except Exception:
        pass

    dataset_ref = bigquery.DatasetReference(client.project, dataset)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        ds = bigquery.Dataset(dataset_ref)
        client.create_dataset(ds)
        logging.info("[BQ] created dataset %s", dataset_ref.dataset_id)

    table_obj = bigquery.Table(table_ref, schema=schema)
    table_obj.time_partitioning = bigquery.TimePartitioning(field="ts")
    table_obj.clustering_fields = ["timeframe"]
    table_obj = client.create_table(table_obj)
    logging.info("[BQ] created table %s", table_ref)
    return table_obj


def _load(
    client: bigquery.Client,
    rows: List[Dict[str, Any]],
    dataset: str,
    table: str,
    *,
    write_disposition: str,
) -> None:
    schema = [
        bigquery.SchemaField("ts", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("open", "FLOAT"),
        bigquery.SchemaField("high", "FLOAT"),
        bigquery.SchemaField("low", "FLOAT"),
        bigquery.SchemaField("close", "FLOAT"),
        bigquery.SchemaField("volume", "INTEGER"),
        bigquery.SchemaField("timeframe", "STRING"),
        bigquery.SchemaField("instrument", "STRING"),
        bigquery.SchemaField("source", "STRING"),
        bigquery.SchemaField("fetched_at", "TIMESTAMP"),
    ]
    if not rows:
        logging.info("[LOAD] no rows to upload")
        return
    table_ref = f"{client.project}.{dataset}.{table}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        schema=schema,
    )
    job = client.load_table_from_json(rows, table_ref, job_config=job_config)
    job.result()
    logging.info("[LOAD] uploaded %d rows into %s", len(rows), table_ref)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        default=["logs/oanda/candles_M1_latest.json"],
        help="入力ファイル (JSON or NDJSON)。デフォルトは最新の M1 ローソク。",
    )
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="quantrabbit")
    parser.add_argument("--table", type=str, default="candles_m1")
    parser.add_argument("--instrument", type=str, default="USD_JPY")
    parser.add_argument("--timeframe", type=str, default="M1")
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="テーブルを置き換えてからロードする（開発用）。既存データは消えるので注意。",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _setup_logging(args.verbose)

    client = bigquery.Client(project=args.project)
    _ensure_table(client, args.dataset, args.table)

    all_rows: List[Dict[str, Any]] = []
    for inp in args.inputs:
        path = Path(inp).expanduser()
        if not path.exists():
            logging.warning("[SKIP] not found: %s", path)
            continue
        rows = list(_iter_rows(path, args.timeframe, args.instrument))
        logging.info("[READ] %s -> %d rows", path, len(rows))
        all_rows.extend(rows)

    write_disp = bigquery.WriteDisposition.WRITE_TRUNCATE if args.truncate else bigquery.WriteDisposition.WRITE_APPEND
    try:
        _load(client, all_rows, args.dataset, args.table, write_disposition=write_disp)
    except Exception as exc:  # noqa: BLE001
        logging.error("[LOAD] failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

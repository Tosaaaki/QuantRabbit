"""Export recent candles to BigQuery for historical analysis/backtesting."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Iterable, List

from google.api_core import exceptions as gexc
from google.cloud import bigquery

from market_data.candle_fetcher import fetch_historical_candles


DEFAULT_PROJECT = os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
DEFAULT_DATASET = os.getenv("BQ_DATASET", "quantrabbit")
DEFAULT_TABLE = os.getenv("BQ_CANDLES_TABLE", "candles")


def _ensure_table(client: bigquery.Client, dataset_id: str, table_id: str) -> None:
    dataset_ref = bigquery.DatasetReference(client.project, dataset_id)
    try:
        client.get_dataset(dataset_ref)
    except gexc.NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = os.getenv("BQ_LOCATION", "US")
        client.create_dataset(dataset, exists_ok=True)

    schema = [
        bigquery.SchemaField("ts", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("instrument", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("timeframe", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("open", "FLOAT64"),
        bigquery.SchemaField("high", "FLOAT64"),
        bigquery.SchemaField("low", "FLOAT64"),
        bigquery.SchemaField("close", "FLOAT64"),
    ]
    table_ref = dataset_ref.table(table_id)
    try:
        client.get_table(table_ref)
    except gexc.NotFound:
        table = bigquery.Table(table_ref, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(field="ts")
        client.create_table(table)


OANDA_GRANULARITY = {
    "M1": "M1",
    "M5": "M5",
    "H1": "H1",
    "H4": "H4",
    "D1": "D",
}


def _fetch_and_transform(
    instrument: str,
    timeframe: str,
    count: int,
) -> List[dict]:
    api_tf = OANDA_GRANULARITY.get(timeframe, timeframe)
    candles = asyncio.run(fetch_historical_candles(instrument, api_tf, count))
    return [
        {
            "ts": c["time"].isoformat(),
            "instrument": instrument,
            "timeframe": timeframe,
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
        }
        for c in candles
    ]


def _insert(client: bigquery.Client, dataset_id: str, table_id: str, rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    table_ref = client.dataset(dataset_id).table(table_id)
    errors = client.insert_rows_json(table_ref, rows)
    if errors:
        raise RuntimeError(f"BigQuery insert failed: {errors}")
    return len(rows)


def run(
    instrument: str = "USD_JPY",
    timeframes: Iterable[str] = ("M1", "H4", "D1"),
    count: int = 500,
    project: str | None = None,
    dataset: str = DEFAULT_DATASET,
    table: str = DEFAULT_TABLE,
) -> None:
    client = bigquery.Client(project=project) if project else bigquery.Client()
    _ensure_table(client, dataset, table)

    total_inserted = 0
    for tf in timeframes:
        try:
            rows = _fetch_and_transform(instrument, tf, count)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("[CANDLE] fetch failed tf=%s error=%s", tf, exc)
            continue
        inserted = _insert(client, dataset, table, rows)
        total_inserted += inserted
        logging.info("[CANDLE] uploaded %s rows (tf=%s)", inserted, tf)

    logging.info("[CANDLE] done total=%s", total_inserted)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export candles to BigQuery")
    parser.add_argument("--instrument", default="USD_JPY")
    parser.add_argument("--timeframes", nargs="*", default=["M1", "H4"])
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--project", default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(
        instrument=args.instrument,
        timeframes=args.timeframes,
        count=args.count,
        project=args.project,
        dataset=args.dataset,
        table=args.table,
    )


if __name__ == "__main__":
    main()

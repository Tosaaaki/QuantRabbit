"""Helpers to fetch candle data from BigQuery."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

from google.cloud import bigquery

DEFAULT_PROJECT = os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
DEFAULT_DATASET = os.getenv("BQ_DATASET", "quantrabbit")
DEFAULT_TABLE = os.getenv("BQ_CANDLES_TABLE", "candles_m1")


def _parse_datetime(value: str | dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        ts = value
    else:
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        ts = dt.datetime.fromisoformat(raw)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def fetch_bq_candles(
    *,
    instrument: str,
    timeframe: str,
    start: str | dt.datetime,
    end: str | dt.datetime,
    project: Optional[str] = None,
    dataset: str = DEFAULT_DATASET,
    table: str = DEFAULT_TABLE,
    limit: Optional[int] = None,
    dedupe: bool = True,
) -> tuple[List[dict], str]:
    client = bigquery.Client(project=project) if project else bigquery.Client()
    table_fqn = f"{client.project}.{dataset}.{table}"

    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end)
    if dedupe:
        sql = (
            "SELECT ts, "
            "ANY_VALUE(open) AS open, "
            "ANY_VALUE(high) AS high, "
            "ANY_VALUE(low) AS low, "
            "ANY_VALUE(close) AS close, "
            "ANY_VALUE(volume) AS volume "
            f"FROM `{table_fqn}` "
            "WHERE instrument = @instrument "
            "AND timeframe = @timeframe "
            "AND ts >= @start AND ts < @end "
            "GROUP BY ts "
            "ORDER BY ts ASC"
        )
    else:
        sql = (
            "SELECT ts, open, high, low, close, volume "
            f"FROM `{table_fqn}` "
            "WHERE instrument = @instrument "
            "AND timeframe = @timeframe "
            "AND ts >= @start AND ts < @end "
            "ORDER BY ts ASC"
        )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("instrument", "STRING", instrument),
            bigquery.ScalarQueryParameter("timeframe", "STRING", timeframe),
            bigquery.ScalarQueryParameter("start", "TIMESTAMP", start_dt),
            bigquery.ScalarQueryParameter("end", "TIMESTAMP", end_dt),
        ]
    )
    rows = client.query(sql, job_config=job_config).result()
    candles: List[dict] = []
    for row in rows:
        ts = row.get("ts")
        if isinstance(ts, str):
            ts = _parse_datetime(ts)
        elif isinstance(ts, dt.datetime):
            ts = _parse_datetime(ts)
        else:
            continue
        open_v = row.get("open")
        high_v = row.get("high")
        low_v = row.get("low")
        close_v = row.get("close")
        if open_v is None or high_v is None or low_v is None or close_v is None:
            continue
        volume = row.get("volume")
        candles.append(
            {
                "time": ts.isoformat().replace("+00:00", "Z"),
                "mid": {
                    "o": float(open_v),
                    "h": float(high_v),
                    "l": float(low_v),
                    "c": float(close_v),
                },
                "volume": int(volume or 0),
                "complete": True,
            }
        )
    return candles, table_fqn


def write_candles_json(
    candles: Iterable[dict],
    *,
    instrument: str,
    timeframe: str,
    start: str | dt.datetime,
    end: str | dt.datetime,
    table_fqn: str,
    out_path: Path,
) -> Path:
    payload = {
        "fetched_at": dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat(),
        "instrument": instrument,
        "timeframe": timeframe,
        "source": "bigquery",
        "table": table_fqn,
        "range": {
            "start": _parse_datetime(start).isoformat().replace("+00:00", "Z"),
            "end": _parse_datetime(end).isoformat().replace("+00:00", "Z"),
        },
        "candles": list(candles),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path

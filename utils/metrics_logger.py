from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

_DB_PATH = Path("logs/metrics.db")
_LOCK = threading.Lock()
_AGG_METRICS = {
    m.strip()
    for m in os.getenv("METRICS_AGGREGATE", "onepip_maker_skip").split(",")
    if m.strip()
}
_AGG_WINDOW_SEC = max(1.0, float(os.getenv("METRICS_AGG_WINDOW_SEC", "10.0")))
_AGG_DROP_TAG_KEYS = {
    k.strip()
    for k in os.getenv("METRICS_AGG_DROP_TAG_KEYS", "latency_ms").split(",")
    if k.strip()
}
_SAMPLE_METRICS = {
    m.strip()
    for m in os.getenv("METRICS_SAMPLE", "").split(",")
    if m.strip()
}
_SAMPLE_EVERY_SEC = max(0.1, float(os.getenv("METRICS_SAMPLE_EVERY_SEC", "1.0")))


@dataclass
class _AggBucket:
    count: int = 0
    total: float = 0.0
    min_val: float | None = None
    max_val: float | None = None
    last_flush: float = 0.0
    tags: dict[str, object] | None = None


_AGG_BUCKETS: dict[tuple[str, str], _AggBucket] = {}
_LAST_SAMPLE_TS: dict[str, float] = {}


def _ensure_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            ts TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            tags TEXT
        )
        """
    )


def _sanitize_tag_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


def _normalize_tags(tags: Optional[Mapping[str, object]]) -> dict[str, object]:
    if not tags:
        return {}
    normalized: dict[str, object] = {}
    for key, val in tags.items():
        if key in _AGG_DROP_TAG_KEYS:
            continue
        normalized[key] = _sanitize_tag_value(val)
    return normalized


def _tags_key(tags: Mapping[str, object]) -> str:
    return json.dumps(tags, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _write_payload(payload: dict[str, object], metric: str) -> None:
    attempts = 0
    while attempts < 2:
        attempts += 1
        con = sqlite3.connect(_DB_PATH, timeout=1.5)
        try:
            con.execute("PRAGMA busy_timeout=1500;")
            con.execute("PRAGMA journal_mode=WAL;")
            _ensure_schema(con)
            con.execute(
                "INSERT INTO metrics(ts, metric, value, tags) VALUES (:ts, :metric, :value, :tags)",
                payload,
            )
            con.commit()
            return
        except sqlite3.OperationalError as exc:
            if attempts >= 2:
                logging.debug("[metrics] drop metric=%s due to lock: %s", metric, exc)
        except Exception as exc:  # pragma: no cover - defensive
            logging.debug("[metrics] drop metric=%s due to error: %s", metric, exc)
            return
        finally:
            con.close()


def log_metric(
    metric: str,
    value: float,
    *,
    tags: Optional[Mapping[str, object]] = None,
    ts: Optional[datetime] = None,
) -> None:
    now_mono = time.monotonic()
    if metric in _AGG_METRICS:
        norm_tags = _normalize_tags(tags)
        key = (metric, _tags_key(norm_tags))
        with _LOCK:
            bucket = _AGG_BUCKETS.get(key)
            if bucket is None:
                bucket = _AggBucket(tags=norm_tags)
                _AGG_BUCKETS[key] = bucket
            val = float(value)
            bucket.count += 1
            bucket.total += val
            bucket.min_val = val if bucket.min_val is None else min(bucket.min_val, val)
            bucket.max_val = val if bucket.max_val is None else max(bucket.max_val, val)
            if bucket.last_flush == 0.0:
                bucket.last_flush = now_mono
                return
            if now_mono - bucket.last_flush < _AGG_WINDOW_SEC:
                return
            flush_tags = dict(bucket.tags or {})
            flush_tags.update(
                {
                    "agg": "sum",
                    "count": bucket.count,
                    "min": bucket.min_val,
                    "max": bucket.max_val,
                    "window_sec": _AGG_WINDOW_SEC,
                }
            )
            payload = {
                "metric": metric,
                "value": float(bucket.total),
                "ts": datetime.utcnow().isoformat(),
                "tags": json.dumps(flush_tags, ensure_ascii=True),
            }
            bucket.count = 0
            bucket.total = 0.0
            bucket.min_val = None
            bucket.max_val = None
            bucket.last_flush = now_mono
            _write_payload(payload, metric)
            return

    if metric in _SAMPLE_METRICS:
        last = _LAST_SAMPLE_TS.get(metric)
        if last is not None and now_mono - last < _SAMPLE_EVERY_SEC:
            return
        _LAST_SAMPLE_TS[metric] = now_mono

    payload = {
        "metric": metric,
        "value": float(value),
        "ts": (ts or datetime.utcnow()).isoformat(),
        "tags": json.dumps(tags or {}, ensure_ascii=True),
    }
    with _LOCK:
        _write_payload(payload, metric)

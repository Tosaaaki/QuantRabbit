"""Export ops insights (execution quality, market state, exposure, health) to BigQuery."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from google.api_core import exceptions as gexc
from google.cloud import bigquery

from utils.oanda_account import get_account_snapshot, get_position_summary

DEFAULT_PROJECT = os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
DEFAULT_DATASET = os.getenv("BQ_DATASET", "quantrabbit")
DEFAULT_EXEC_TABLE = os.getenv("BQ_EXECUTION_QUALITY_TABLE", "execution_quality")
DEFAULT_MARKET_TABLE = os.getenv("BQ_MARKET_STATE_TABLE", "market_state")
DEFAULT_EXPOSURE_TABLE = os.getenv("BQ_EXPOSURE_TABLE", "exposure_snapshot")
DEFAULT_HEALTH_TABLE = os.getenv("BQ_OPS_HEALTH_TABLE", "ops_health")

DEFAULT_LOOKBACK_MIN = int(os.getenv("BQ_OPS_INSIGHTS_LOOKBACK_MIN", "120"))
DEFAULT_RETENTION_HOURS = int(os.getenv("BQ_OPS_INSIGHTS_RETENTION_H", "168"))
DEFAULT_RANGE_FALLBACK_MIN = int(os.getenv("OPS_RANGE_FALLBACK_MIN", "1440"))

ORDERS_DB = Path(os.getenv("BQ_ORDERS_DB", "logs/orders.db"))
METRICS_DB = Path(os.getenv("BQ_METRICS_DB", "logs/metrics.db"))
HEALTH_SNAPSHOT_PATH = Path(
    os.getenv("OPS_HEALTH_SNAPSHOT_PATH", "logs/health_snapshot.json")
)
REPLAY_DIR = Path(os.getenv("OPS_REPLAY_DIR", "logs/replay/USD_JPY"))
EXPOSURE_INSTRUMENT = os.getenv("OPS_EXPOSURE_INSTRUMENT", "USD_JPY")

EXEC_ENABLED = os.getenv("OPS_EXEC_QUALITY_ENABLED", "1") not in {"0", "false", "off"}
MARKET_ENABLED = os.getenv("OPS_MARKET_STATE_ENABLED", "1") not in {"0", "false", "off"}
EXPOSURE_ENABLED = os.getenv("OPS_EXPOSURE_ENABLED", "1") not in {"0", "false", "off"}
HEALTH_ENABLED = os.getenv("OPS_HEALTH_ENABLED", "1") not in {"0", "false", "off"}

FAIL_STATUSES = {"REJECTED", "FAILED", "ERROR", "CANCELLED"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    data = sorted(values)
    n = len(data)
    if n == 1:
        return float(data[0])
    k = (n - 1) * q
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return float(data[int(k)])
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return float(d0 + d1)


def _ensure_dataset(client: bigquery.Client, dataset_id: str) -> None:
    dataset_ref = bigquery.DatasetReference(client.project, dataset_id)
    try:
        client.get_dataset(dataset_ref)
    except gexc.NotFound:
        logging.info("[BQ] dataset %s missing. Creating...", dataset_id)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = os.getenv("BQ_LOCATION", "US")
        client.create_dataset(dataset, exists_ok=True)


def _ensure_table(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    schema: List[bigquery.SchemaField],
    *,
    partition_field: str = "generated_at",
) -> None:
    table_ref = bigquery.DatasetReference(client.project, dataset_id).table(table_id)
    try:
        table = client.get_table(table_ref)
        existing = {field.name for field in table.schema}
        missing = [field for field in schema if field.name not in existing]
        if missing:
            table.schema = list(table.schema) + missing
            client.update_table(table, ["schema"])
            logging.info("[BQ] table %s schema updated (+%d)", table_ref.path, len(missing))
        return
    except gexc.NotFound:
        pass
    logging.info("[BQ] creating table %s", table_ref.path)
    table = bigquery.Table(table_ref, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(field=partition_field)
    client.create_table(table)


def _metric_values(con: sqlite3.Connection, metric: str, since_ts: str) -> List[float]:
    rows = con.execute(
        "SELECT value FROM metrics WHERE metric = ? AND ts >= ?",
        (metric, since_ts),
    ).fetchall()
    out: List[float] = []
    for (val,) in rows:
        try:
            out.append(float(val))
        except Exception:
            continue
    return out


def _range_stats(con: sqlite3.Connection, since_ts: str) -> Dict[str, Any]:
    def _fetch_rows(ts: str) -> List[tuple]:
        return con.execute(
            "SELECT value, tags FROM metrics WHERE metric = ? AND ts >= ?",
            ("range_mode_active", ts),
        ).fetchall()

    rows = _fetch_rows(since_ts)
    values: List[float] = []
    scores: List[float] = []
    last_tags: Optional[Dict[str, Any]] = None
    for value, tags in rows:
        try:
            values.append(float(value))
        except Exception:
            continue
        if tags:
            try:
                parsed = json.loads(tags)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                last_tags = parsed
                score = parsed.get("score")
                if score is not None:
                    try:
                        scores.append(float(score))
                    except Exception:
                        pass
    if not values and DEFAULT_RANGE_FALLBACK_MIN > 0:
        fallback_ts = (_utcnow() - timedelta(minutes=DEFAULT_RANGE_FALLBACK_MIN)).isoformat()
        rows = _fetch_rows(fallback_ts)
        values = []
        scores = []
        last_tags = None
        for value, tags in rows:
            try:
                values.append(float(value))
            except Exception:
                continue
            if tags:
                try:
                    parsed = json.loads(tags)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    last_tags = parsed
                    score = parsed.get("score")
                    if score is not None:
                        try:
                            scores.append(float(score))
                        except Exception:
                            pass

    ratio = (sum(values) / len(values)) if values else None
    score_avg = (sum(scores) / len(scores)) if scores else None
    reason = None
    if last_tags:
        reason = last_tags.get("reason")
    return {"ratio": ratio, "score_avg": score_avg, "reason": reason}


def _orders_stats(since_ts: str) -> Dict[str, Dict[str, Any]]:
    if not ORDERS_DB.exists():
        return {}
    con = sqlite3.connect(str(ORDERS_DB))
    rows = con.execute(
        "SELECT pocket, status, error_code FROM orders WHERE ts >= ?",
        (since_ts,),
    ).fetchall()
    con.close()
    stats: Dict[str, Dict[str, Any]] = {}
    for pocket, status, error_code in rows:
        pocket_key = str(pocket or "unknown")
        data = stats.setdefault(
            pocket_key,
            {"orders_total": 0, "orders_failed": 0, "orders_reject_like": 0},
        )
        data["orders_total"] += 1
        status_text = str(status or "unknown")
        if error_code or status_text.upper() in FAIL_STATUSES:
            data["orders_failed"] += 1
        if "reject" in status_text.lower():
            data["orders_reject_like"] += 1
    for pocket_key, data in stats.items():
        total = data["orders_total"]
        failed = data["orders_failed"]
        reject_like = data["orders_reject_like"]
        data["reject_rate"] = round(failed / total, 4) if total else 0.0
        data["reject_like_rate"] = round(reject_like / total, 4) if total else 0.0
        data["pocket"] = pocket_key
    return stats


def _load_recent_candles(pattern: str, maxlen: int) -> List[Dict[str, Any]]:
    if not REPLAY_DIR.exists():
        return []
    files = sorted(REPLAY_DIR.glob(pattern))
    if not files:
        return []
    data: deque = deque(maxlen=maxlen)
    for path in files[-2:]:
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            continue
    return list(data)


def _trend_from_candles(candles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candles or len(candles) < 5:
        return None
    candles = sorted(candles, key=lambda r: r.get("ts") or "")
    closes = [c.get("close") for c in candles if isinstance(c.get("close"), (int, float))]
    highs = [c.get("high") for c in candles if isinstance(c.get("high"), (int, float))]
    lows = [c.get("low") for c in candles if isinstance(c.get("low"), (int, float))]
    if not closes or not highs or not lows:
        return None
    start = closes[0]
    end = closes[-1]
    net = end - start
    ranges = [h - l for h, l in zip(highs, lows)]
    avg_range = sum(ranges) / len(ranges) if ranges else 0.0
    strength = abs(net) / avg_range if avg_range else 0.0
    if strength < 1.0:
        direction = "flat"
    else:
        direction = "up" if net > 0 else "down"
    return {
        "direction": direction,
        "strength": round(strength, 3),
        "net": round(net, 4),
        "avg_range": round(avg_range, 4),
        "last_ts": candles[-1].get("ts"),
        "count": len(candles),
    }


def _trend_snapshot() -> Dict[str, Any]:
    return {
        "m1": _trend_from_candles(_load_recent_candles("USD_JPY_M1_*.jsonl", 120)),
        "h1": _trend_from_candles(_load_recent_candles("USD_JPY_H1_*.jsonl", 24)),
        "h4": _trend_from_candles(_load_recent_candles("USD_JPY_H4_*.jsonl", 30)),
    }


def _build_execution_quality_rows(
    since_ts: str,
    generated_at: str,
    window_min: int,
) -> List[Dict[str, Any]]:
    if not EXEC_ENABLED:
        return []
    rows: List[Dict[str, Any]] = []
    with sqlite3.connect(str(METRICS_DB)) as con:
        spread_p95 = _percentile(_metric_values(con, "decision_spread_pips", since_ts), 0.95)
        latency_p95 = _percentile(_metric_values(con, "decision_latency_ms", since_ts), 0.95)
        lag_p95 = _percentile(_metric_values(con, "data_lag_ms", since_ts), 0.95)

    pocket_stats = _orders_stats(since_ts)
    if not pocket_stats:
        pocket_stats = {"all": {"orders_total": 0, "orders_failed": 0, "orders_reject_like": 0, "reject_rate": 0.0, "reject_like_rate": 0.0, "pocket": "all"}}

    for pocket_key, data in pocket_stats.items():
        rows.append(
            {
                "generated_at": generated_at,
                "window_minutes": window_min,
                "pocket": pocket_key,
                "orders_total": data.get("orders_total", 0),
                "orders_failed": data.get("orders_failed", 0),
                "reject_rate": data.get("reject_rate"),
                "reject_like_rate": data.get("reject_like_rate"),
                "decision_spread_p95": spread_p95,
                "decision_latency_p95": latency_p95,
                "data_lag_p95": lag_p95,
            }
        )
    return rows


def _build_market_state_row(
    since_ts: str,
    generated_at: str,
    window_min: int,
) -> Optional[Dict[str, Any]]:
    if not MARKET_ENABLED:
        return None
    range_ratio = None
    range_score = None
    range_reason = None
    if METRICS_DB.exists():
        with sqlite3.connect(str(METRICS_DB)) as con:
            stats = _range_stats(con, since_ts)
            range_ratio = stats.get("ratio")
            range_score = stats.get("score_avg")
            range_reason = stats.get("reason")
    trend = _trend_snapshot()
    def _get_tf(tf: str) -> Dict[str, Any]:
        item = trend.get(tf) if isinstance(trend, dict) else None
        if not isinstance(item, dict):
            return {}
        return item

    m1 = _get_tf("m1")
    h1 = _get_tf("h1")
    h4 = _get_tf("h4")
    return {
        "generated_at": generated_at,
        "window_minutes": window_min,
        "range_active_ratio": range_ratio,
        "range_score_avg": range_score,
        "range_reason": range_reason,
        "trend_m1_direction": m1.get("direction"),
        "trend_m1_strength": m1.get("strength"),
        "trend_m1_net": m1.get("net"),
        "trend_m1_avg_range": m1.get("avg_range"),
        "trend_m1_last_ts": m1.get("last_ts"),
        "trend_h1_direction": h1.get("direction"),
        "trend_h1_strength": h1.get("strength"),
        "trend_h1_net": h1.get("net"),
        "trend_h1_avg_range": h1.get("avg_range"),
        "trend_h1_last_ts": h1.get("last_ts"),
        "trend_h4_direction": h4.get("direction"),
        "trend_h4_strength": h4.get("strength"),
        "trend_h4_net": h4.get("net"),
        "trend_h4_avg_range": h4.get("avg_range"),
        "trend_h4_last_ts": h4.get("last_ts"),
    }


def _build_exposure_row(
    generated_at: str,
) -> Optional[Dict[str, Any]]:
    if not EXPOSURE_ENABLED:
        return None
    try:
        snap = get_account_snapshot()
    except Exception as exc:
        logging.warning("[EXPOSURE] account snapshot failed: %s", exc)
        return None
    long_units, short_units = get_position_summary(EXPOSURE_INSTRUMENT)
    return {
        "generated_at": generated_at,
        "instrument": EXPOSURE_INSTRUMENT,
        "nav": snap.nav,
        "balance": snap.balance,
        "margin_available": snap.margin_available,
        "margin_used": snap.margin_used,
        "margin_rate": snap.margin_rate,
        "free_margin_ratio": snap.free_margin_ratio,
        "health_buffer": snap.health_buffer,
        "unrealized_pl": snap.unrealized_pl,
        "long_units": long_units,
        "short_units": short_units,
        "net_units": long_units - short_units,
    }


def _load_health_snapshot() -> Optional[Dict[str, Any]]:
    if not HEALTH_SNAPSHOT_PATH.exists():
        return None
    try:
        payload = json.loads(HEALTH_SNAPSHOT_PATH.read_text())
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _build_health_row(generated_at: str) -> Optional[Dict[str, Any]]:
    if not HEALTH_ENABLED:
        return None
    snap = _load_health_snapshot()
    if not snap:
        return None
    return {
        "generated_at": generated_at,
        "hostname": snap.get("hostname"),
        "git_rev": snap.get("git_rev"),
        "uptime_sec": snap.get("uptime_sec"),
        "trades_last_entry": snap.get("trades_last_entry"),
        "trades_last_close": snap.get("trades_last_close"),
        "trades_count_24h": snap.get("trades_count_24h"),
        "signals_last_ts": snap.get("signals_last_ts"),
        "orders_last_ts": snap.get("orders_last_ts"),
        "data_lag_ms": snap.get("data_lag_ms"),
        "decision_latency_ms": snap.get("decision_latency_ms"),
        "disk_used_pct": snap.get("disk_used_pct"),
        "disk_free_mb": snap.get("disk_free_mb"),
        "service_active_json": json.dumps(snap.get("service_active") or {}, ensure_ascii=True),
        "orders_status_1h_json": json.dumps(snap.get("orders_status_1h") or [], ensure_ascii=True),
        "db_mtime_json": json.dumps(snap.get("db_mtime") or {}, ensure_ascii=True),
        "db_size_bytes_json": json.dumps(snap.get("db_size_bytes") or {}, ensure_ascii=True),
        "load_avg_json": json.dumps(snap.get("load_avg") or [], ensure_ascii=True),
    }


def _insert_rows(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    rows: Iterable[Dict[str, Any]],
) -> int:
    rows = list(rows)
    if not rows:
        return 0
    table_ref = client.dataset(dataset_id).table(table_id)
    errors = client.insert_rows_json(table_ref, rows)
    if errors:
        raise RuntimeError(f"BigQuery insert failed: {errors}")
    return len(rows)


def _prune_table(
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
            query_parameters=[bigquery.ScalarQueryParameter("retention", "INT64", retention_hours)]
        ),
    ).result()


def run(
    *,
    project: Optional[str],
    dataset: str,
    lookback_min: int,
    retention_hours: int,
) -> None:
    now = _utcnow()
    since_ts = (now - timedelta(minutes=lookback_min)).isoformat()
    generated_at = now.isoformat()

    client = bigquery.Client(project=project) if project else bigquery.Client()
    _ensure_dataset(client, dataset)

    exec_schema = [
        bigquery.SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("window_minutes", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("pocket", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("orders_total", "INT64"),
        bigquery.SchemaField("orders_failed", "INT64"),
        bigquery.SchemaField("reject_rate", "FLOAT64"),
        bigquery.SchemaField("reject_like_rate", "FLOAT64"),
        bigquery.SchemaField("decision_spread_p95", "FLOAT64"),
        bigquery.SchemaField("decision_latency_p95", "FLOAT64"),
        bigquery.SchemaField("data_lag_p95", "FLOAT64"),
    ]
    market_schema = [
        bigquery.SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("window_minutes", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("range_active_ratio", "FLOAT64"),
        bigquery.SchemaField("range_score_avg", "FLOAT64"),
        bigquery.SchemaField("range_reason", "STRING"),
        bigquery.SchemaField("trend_m1_direction", "STRING"),
        bigquery.SchemaField("trend_m1_strength", "FLOAT64"),
        bigquery.SchemaField("trend_m1_net", "FLOAT64"),
        bigquery.SchemaField("trend_m1_avg_range", "FLOAT64"),
        bigquery.SchemaField("trend_m1_last_ts", "TIMESTAMP"),
        bigquery.SchemaField("trend_h1_direction", "STRING"),
        bigquery.SchemaField("trend_h1_strength", "FLOAT64"),
        bigquery.SchemaField("trend_h1_net", "FLOAT64"),
        bigquery.SchemaField("trend_h1_avg_range", "FLOAT64"),
        bigquery.SchemaField("trend_h1_last_ts", "TIMESTAMP"),
        bigquery.SchemaField("trend_h4_direction", "STRING"),
        bigquery.SchemaField("trend_h4_strength", "FLOAT64"),
        bigquery.SchemaField("trend_h4_net", "FLOAT64"),
        bigquery.SchemaField("trend_h4_avg_range", "FLOAT64"),
        bigquery.SchemaField("trend_h4_last_ts", "TIMESTAMP"),
    ]
    exposure_schema = [
        bigquery.SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("instrument", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("nav", "FLOAT64"),
        bigquery.SchemaField("balance", "FLOAT64"),
        bigquery.SchemaField("margin_available", "FLOAT64"),
        bigquery.SchemaField("margin_used", "FLOAT64"),
        bigquery.SchemaField("margin_rate", "FLOAT64"),
        bigquery.SchemaField("free_margin_ratio", "FLOAT64"),
        bigquery.SchemaField("health_buffer", "FLOAT64"),
        bigquery.SchemaField("unrealized_pl", "FLOAT64"),
        bigquery.SchemaField("long_units", "FLOAT64"),
        bigquery.SchemaField("short_units", "FLOAT64"),
        bigquery.SchemaField("net_units", "FLOAT64"),
    ]
    health_schema = [
        bigquery.SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("hostname", "STRING"),
        bigquery.SchemaField("git_rev", "STRING"),
        bigquery.SchemaField("uptime_sec", "FLOAT64"),
        bigquery.SchemaField("trades_last_entry", "TIMESTAMP"),
        bigquery.SchemaField("trades_last_close", "TIMESTAMP"),
        bigquery.SchemaField("trades_count_24h", "INT64"),
        bigquery.SchemaField("signals_last_ts", "INT64"),
        bigquery.SchemaField("orders_last_ts", "TIMESTAMP"),
        bigquery.SchemaField("data_lag_ms", "FLOAT64"),
        bigquery.SchemaField("decision_latency_ms", "FLOAT64"),
        bigquery.SchemaField("disk_used_pct", "FLOAT64"),
        bigquery.SchemaField("disk_free_mb", "INT64"),
        bigquery.SchemaField("service_active_json", "STRING"),
        bigquery.SchemaField("orders_status_1h_json", "STRING"),
        bigquery.SchemaField("db_mtime_json", "STRING"),
        bigquery.SchemaField("db_size_bytes_json", "STRING"),
        bigquery.SchemaField("load_avg_json", "STRING"),
    ]

    if EXEC_ENABLED:
        _ensure_table(client, dataset, DEFAULT_EXEC_TABLE, exec_schema)
    if MARKET_ENABLED:
        _ensure_table(client, dataset, DEFAULT_MARKET_TABLE, market_schema)
    if EXPOSURE_ENABLED:
        _ensure_table(client, dataset, DEFAULT_EXPOSURE_TABLE, exposure_schema)
    if HEALTH_ENABLED:
        _ensure_table(client, dataset, DEFAULT_HEALTH_TABLE, health_schema)

    exec_rows = _build_execution_quality_rows(since_ts, generated_at, lookback_min)
    market_row = _build_market_state_row(since_ts, generated_at, lookback_min)
    exposure_row = _build_exposure_row(generated_at)
    health_row = _build_health_row(generated_at)

    if exec_rows:
        inserted = _insert_rows(client, dataset, DEFAULT_EXEC_TABLE, exec_rows)
        logging.info("[BQ] execution_quality rows=%s", inserted)
        _prune_table(client, dataset, DEFAULT_EXEC_TABLE, retention_hours)
    if market_row:
        inserted = _insert_rows(client, dataset, DEFAULT_MARKET_TABLE, [market_row])
        logging.info("[BQ] market_state rows=%s", inserted)
        _prune_table(client, dataset, DEFAULT_MARKET_TABLE, retention_hours)
    if exposure_row:
        inserted = _insert_rows(client, dataset, DEFAULT_EXPOSURE_TABLE, [exposure_row])
        logging.info("[BQ] exposure_snapshot rows=%s", inserted)
        _prune_table(client, dataset, DEFAULT_EXPOSURE_TABLE, retention_hours)
    if health_row:
        inserted = _insert_rows(client, dataset, DEFAULT_HEALTH_TABLE, [health_row])
        logging.info("[BQ] ops_health rows=%s", inserted)
        _prune_table(client, dataset, DEFAULT_HEALTH_TABLE, retention_hours)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ops insights to BigQuery.")
    parser.add_argument("--project", default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
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
        lookback_min=args.lookback,
        retention_hours=args.retention,
    )


if __name__ == "__main__":
    main()

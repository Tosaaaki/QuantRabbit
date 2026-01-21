from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from google.cloud import storage

from utils.secrets import get_secret


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_query(db_path: Path, query: str) -> Optional[Any]:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute(query)
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        return None


def _load_bucket_name() -> Optional[str]:
    for key in ("ui_bucket_name", "GCS_BACKUP_BUCKET"):
        try:
            return get_secret(key)
        except KeyError:
            continue
    return None


def _build_snapshot() -> dict[str, Any]:
    hostname = socket.gethostname()
    logs_dir = Path("/home/tossaki/QuantRabbit/logs")
    trades_db = logs_dir / "trades.db"
    signals_db = logs_dir / "signals.db"
    orders_db = logs_dir / "orders.db"
    metrics_db = logs_dir / "metrics.db"

    deploy_id = None
    try:
        deploy_path = Path("/var/lib/quantrabbit/deploy_id")
        if deploy_path.exists():
            deploy_id = deploy_path.read_text().strip()
    except Exception:
        pass

    snapshot = {
        "generated_at": _utcnow_iso(),
        "hostname": hostname,
        "deploy_id": deploy_id,
        "trades_last_entry": _safe_query(
            trades_db, "select max(entry_time) from trades;"
        ),
        "trades_last_close": _safe_query(
            trades_db, "select max(close_time) from trades;"
        ),
        "trades_count_24h": _safe_query(
            trades_db,
            "select count(*) from trades where entry_time >= datetime('now','-1 day');",
        ),
        "signals_last_ts": _safe_query(signals_db, "select max(ts) from signals;"),
        "orders_last_ts": _safe_query(orders_db, "select max(ts) from orders;"),
        "data_lag_ms": _safe_query(
            metrics_db,
            "select value from metrics where name='data_lag_ms' order by ts desc limit 1;",
        ),
        "decision_latency_ms": _safe_query(
            metrics_db,
            "select value from metrics where name='decision_latency_ms' order by ts desc limit 1;",
        ),
    }
    return snapshot


def main() -> None:
    bucket_name = _load_bucket_name()
    if not bucket_name:
        logging.warning("[HEALTH] bucket not configured; skip upload")
        return
    try:
        project_id = get_secret("gcp_project_id")
    except KeyError:
        project_id = None

    object_path = os.getenv("HEALTH_OBJECT_PATH")
    if not object_path:
        object_path = f"realtime/health_{socket.gethostname()}.json"

    snapshot = _build_snapshot()
    try:
        client = storage.Client(project=project_id) if project_id else storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)
        blob.cache_control = "no-cache"
        blob.upload_from_string(
            json.dumps(snapshot, ensure_ascii=True, separators=(",", ":")),
            content_type="application/json",
        )
        logging.info(
            "[HEALTH] snapshot uploaded bucket=%s object=%s", bucket_name, object_path
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("[HEALTH] upload failed: %s", exc)


if __name__ == "__main__":
    main()

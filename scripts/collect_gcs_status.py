#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import re


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "command failed")
    return proc.stdout


def _gcs_cat(uri: str, project: Optional[str]) -> str:
    cmd = ["gcloud", "storage", "cat", uri]
    if project:
        cmd.append(f"--project={project}")
    return _run(cmd)


def _gcs_ls(uri: str, project: Optional[str]) -> list[str]:
    cmd = ["gcloud", "storage", "ls", uri]
    if project:
        cmd.append(f"--project={project}")
    out = _run(cmd)
    return [line.strip() for line in out.splitlines() if line.strip()]


def _gcs_cp(src: str, dest: Path, project: Optional[str]) -> None:
    cmd = ["gcloud", "storage", "cp", src, str(dest)]
    if project:
        cmd.append(f"--project={project}")
    _run(cmd)


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


def _safe_query_rows(db_path: Path, query: str) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(query)
            return [dict(row) for row in cur.fetchall()]
    except Exception:
        return []


def _parse_iso_ts(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _hours_lag(earlier: Optional[datetime], later: Optional[datetime]) -> Optional[float]:
    if earlier is None or later is None:
        return None
    return (later - earlier).total_seconds() / 3600.0


def _classify_staleness(hours: Optional[float], warn_hours: float, critical_hours: float) -> str | None:
    if hours is None:
        return None
    if hours < 0:
        return "future"
    if hours >= critical_hours:
        return "critical"
    if hours >= warn_hours:
        return "warning"
    return "ok"


def _load_recent_signals(db_path: Path, limit: int = 5) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "select ts_ms, payload from signals order by ts_ms desc limit ?;",
                (int(limit),),
            )
            rows: list[dict[str, Any]] = []
            for ts_ms, payload in cur.fetchall():
                item: dict[str, Any] = {"ts_ms": ts_ms}
                try:
                    data = json.loads(payload)
                except Exception:
                    data = None
                if isinstance(data, dict):
                    for key in (
                        "pocket",
                        "strategy",
                        "confidence",
                        "action",
                        "client_order_id",
                        "proposed_units",
                    ):
                        if key in data:
                            item[key] = data[key]
                rows.append(item)
            return rows
    except Exception:
        return []


def _extract_core_tar(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {"trades.db", "orders.db", "orders_snapshot_48h.db", "signals.db", "metrics.db"}
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            name = Path(member.name).name
            if name in wanted:
                tf.extract(member, path=out_dir)


def _find_latest_core(bucket: str, host: str, project: Optional[str]) -> Optional[str]:
    prefix = f"gs://{bucket}/qr-logs/{host}/core_*"
    objects = _gcs_ls(prefix, project)
    if not objects:
        return None
    core_objects = [o for o in objects if re.search(r"/core_\d{8}T\d{6}Z\.tar(\.gz)?$", o)]
    if not core_objects:
        return None
    return sorted(core_objects)[-1]


def _load_ui_state(uri: str, project: Optional[str]) -> dict[str, Any]:
    raw = _gcs_cat(uri, project)
    return json.loads(raw)


def collect_status(
    *,
    ui_bucket: str,
    backup_bucket: str,
    host: str,
    project: Optional[str],
    ui_object: str,
    out_dir: Path,
    orders_snapshot_warn_hours: float = 24.0,
    orders_snapshot_critical_hours: float = 120.0,
) -> dict[str, Any]:
    ui_state = _load_ui_state(f"gs://{ui_bucket}/{ui_object}", project)
    core_obj = _find_latest_core(backup_bucket, host, project)
    if not core_obj:
        return {"ui_state": ui_state, "core": None, "core_object": None}

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_tar = out_dir / Path(core_obj).name
    _gcs_cp(core_obj, tmp_tar, project)
    extract_dir = out_dir / "core_extract"
    _extract_core_tar(tmp_tar, extract_dir)

    trades_db = next(extract_dir.rglob("trades.db"), None)
    orders_db = next(extract_dir.rglob("orders.db"), None)
    signals_db = next(extract_dir.rglob("signals.db"), None)
    metrics_db = next(extract_dir.rglob("metrics.db"), None)
    orders_snapshot_db = next(extract_dir.rglob("orders_snapshot_48h.db"), None)
    orders_db_source = "none"
    if not orders_db and orders_snapshot_db:
        orders_db = orders_snapshot_db
        orders_db_source = "orders_snapshot_48h.db(fallback)"
    elif orders_db is not None:
        orders_db_source = "orders.db"
    snapshot_order = _safe_query(orders_snapshot_db, "select max(ts) from orders;") if orders_snapshot_db else None
    order_db_latest = _safe_query(orders_db, "select max(ts) from orders;") if orders_db else None
    trade_db_latest = _safe_query(trades_db, "select max(entry_time) from trades;") if trades_db else None
    if trade_db_latest is None:
        trade_db_latest = _safe_query(trades_db, "select max(close_time) from trades;") if trades_db else None
    orders_snapshot_age_vs_trades_h = _hours_lag(_parse_iso_ts(snapshot_order), _parse_iso_ts(trade_db_latest))

    core = {
        "core_object": core_obj,
        "trades_db_latest_ts": trade_db_latest,
        "orders_snapshot_latest_ts": snapshot_order,
        "orders_db_latest_ts": order_db_latest,
        "orders_db_source": orders_db_source,
        "orders_snapshot_age_vs_trades_h": orders_snapshot_age_vs_trades_h,
        "orders_snapshot_freshness": _classify_staleness(
            orders_snapshot_age_vs_trades_h,
            orders_snapshot_warn_hours,
            orders_snapshot_critical_hours,
        ),
        "trades_recent": _safe_query_rows(
            trades_db,
            "select ticket_id,pocket,client_order_id,units,entry_time,close_time,pl_pips,state "
            "from trades order by entry_time desc limit 5;",
        )
        if trades_db
        else [],
        "orders_last": _safe_query_rows(
            orders_db,
            "select ts,pocket,side,units,client_order_id,status from orders "
            "order by ts desc limit 5;",
        )
        if orders_db
        else [],
        "orders_status_1h": _safe_query_rows(
            orders_db,
            "select status,count(*) as count from orders "
            "where ts >= datetime('now','-1 hour') "
            "group by status order by count desc limit 8;",
        )
        if orders_db
        else [],
        "signals_last_ts_ms": _safe_query(
            signals_db, "select max(ts_ms) from signals;"
        )
        if signals_db
        else None,
        "signals_recent": _load_recent_signals(signals_db, limit=5) if signals_db else [],
        "data_lag_ms": _safe_query(
            metrics_db,
            "select value from metrics where metric='data_lag_ms' order by ts desc limit 1;",
        )
        if metrics_db
        else None,
        "decision_latency_ms": _safe_query(
            metrics_db,
            "select value from metrics where metric='decision_latency_ms' order by ts desc limit 1;",
        )
        if metrics_db
        else None,
        "healthbeat_ts": _safe_query(
            metrics_db,
            "select max(ts) from metrics where metric='healthbeat';",
        )
        if metrics_db
        else None,
    }

    return {"ui_state": ui_state, "core": core, "core_object": core_obj}


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect VM status from GCS snapshots.")
    parser.add_argument("--ui-bucket", required=True, help="UI bucket name")
    parser.add_argument("--backup-bucket", required=True, help="Backup bucket name")
    parser.add_argument("--host", required=True, help="VM hostname")
    parser.add_argument("--project", default=None, help="GCP project (optional)")
    parser.add_argument(
        "--ui-object",
        default="realtime/ui_state.json",
        help="UI snapshot object path",
    )
    parser.add_argument(
        "--out-dir",
        default="remote_logs_current",
        help="Directory to store downloaded core bundle",
    )
    parser.add_argument("--out", default=None, help="Write JSON to file")
    parser.add_argument(
        "--orders-snapshot-warn-hours",
        type=float,
        default=24.0,
        help="Threshold (hours) to mark orders_snapshot_48h as warning",
    )
    parser.add_argument(
        "--orders-snapshot-critical-hours",
        type=float,
        default=120.0,
        help="Threshold (hours) to mark orders_snapshot_48h as critical",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    status = collect_status(
        ui_bucket=args.ui_bucket,
        backup_bucket=args.backup_bucket,
        host=args.host,
        project=args.project,
        ui_object=args.ui_object,
        out_dir=out_dir,
        orders_snapshot_warn_hours=args.orders_snapshot_warn_hours,
        orders_snapshot_critical_hours=args.orders_snapshot_critical_hours,
    )
    payload = json.dumps(status, ensure_ascii=True, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

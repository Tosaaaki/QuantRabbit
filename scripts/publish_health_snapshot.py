from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
try:
    os.chdir(PROJECT_ROOT)
except Exception:
    pass

try:
    from google.cloud import storage
except Exception:  # pragma: no cover - fallback to CLI upload
    storage = None

from utils.secrets import get_secret
from utils.gcs_uploader import upload_json_via_metadata


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


def _safe_query_rows(db_path: Path, query: str) -> Optional[list[dict[str, Any]]]:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    except Exception:
        return None


def _load_recent_signals(db_path: Path, limit: int = 5) -> Optional[list[dict[str, Any]]]:
    if not db_path.exists():
        return None
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
        return None


def _run_cmd(args: list[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    out = proc.stdout.strip() if proc.stdout else ""
    if proc.returncode != 0 and not out:
        return None
    return out or None


def _systemd_is_active(unit: str) -> Optional[bool]:
    if not shutil.which("systemctl"):
        return None
    output = _run_cmd(["systemctl", "is-active", unit])
    if output is None:
        return None
    return output.strip() == "active"


def _port_listening(port: int) -> Optional[bool]:
    pattern = re.compile(rf":{port}\\b")
    commands = []
    if shutil.which("ss"):
        commands.append(["ss", "-ltn"])
    if shutil.which("netstat"):
        commands.append(["netstat", "-ltn"])
    if not commands:
        return None
    for cmd in commands:
        output = _run_cmd(cmd)
        if output is None:
            continue
        if any(pattern.search(line) for line in output.splitlines()):
            return True
    return False


def _disk_usage_pct(path: Path) -> Optional[float]:
    try:
        usage = shutil.disk_usage(path)
    except Exception:
        return None
    if usage.total <= 0:
        return None
    return round(usage.used / usage.total * 100.0, 2)


def _free_mb(path: Path) -> Optional[int]:
    try:
        usage = shutil.disk_usage(path)
    except Exception:
        return None
    return int(usage.free // (1024 * 1024))


def _uptime_sec() -> Optional[float]:
    try:
        text = Path("/proc/uptime").read_text().strip()
        return float(text.split()[0])
    except Exception:
        return None


def _mtime_iso(path: Path) -> Optional[str]:
    try:
        ts = path.stat().st_mtime
    except Exception:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _size_bytes(path: Path) -> Optional[int]:
    try:
        return int(path.stat().st_size)
    except Exception:
        return None


def _git_rev(repo_dir: Path) -> Optional[str]:
    if not (repo_dir / ".git").exists():
        return None
    output = _run_cmd(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"])
    return output.strip() if output else None


def _upload_via_cli(bucket: str, object_path: str, payload: str) -> bool:
    target = f"gs://{bucket}/{object_path}"
    for cmd in (["gcloud", "storage", "cp", "-", target], ["gsutil", "cp", "-", target]):
        if not shutil.which(cmd[0]):
            continue
        try:
            proc = subprocess.run(
                cmd,
                input=payload,
                text=True,
                capture_output=True,
                timeout=10.0,
                check=False,
            )
        except Exception:
            continue
        if proc.returncode == 0:
            logging.info("[HEALTH] snapshot uploaded via %s -> %s", cmd[0], target)
            return True
    return False


def _load_bucket_name() -> Optional[str]:
    for key in ("ui_bucket_name", "GCS_BACKUP_BUCKET"):
        try:
            return get_secret(key)
        except KeyError:
            continue
    return None


def _build_snapshot() -> dict[str, Any]:
    hostname = socket.gethostname()
    repo_dir = PROJECT_ROOT
    logs_dir = PROJECT_ROOT / "logs"
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
        "snapshot_version": 2,
        "generated_at": _utcnow_iso(),
        "hostname": hostname,
        "deploy_id": deploy_id,
        "git_rev": _git_rev(repo_dir),
        "uptime_sec": _uptime_sec(),
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
        "trades_recent": _safe_query_rows(
            trades_db,
            "select ticket_id,pocket,client_order_id,units,entry_time,close_time,pl_pips,state "
            "from trades order by entry_time desc limit 5;",
        ),
        "signals_last_ts": _safe_query(signals_db, "select max(ts_ms) from signals;"),
        "signals_recent": _load_recent_signals(signals_db, limit=5),
        "orders_last_ts": _safe_query(orders_db, "select max(ts) from orders;"),
        "orders_recent": _safe_query_rows(
            orders_db,
            "select ts,pocket,side,units,client_order_id,status from orders "
            "order by ts desc limit 5;",
        ),
        "orders_status_1h": _safe_query_rows(
            orders_db,
            "select status,count(*) as count from orders "
            "where ts >= datetime('now','-1 hour') "
            "group by status order by count desc limit 8;",
        ),
        "data_lag_ms": _safe_query(
            metrics_db,
            "select value from metrics where metric='data_lag_ms' order by ts desc limit 1;",
        ),
        "decision_latency_ms": _safe_query(
            metrics_db,
            "select value from metrics where metric='decision_latency_ms' order by ts desc limit 1;",
        ),
        "healthbeat_ts": _safe_query(
            metrics_db,
            "select max(ts) from metrics where metric='healthbeat';",
        ),
        "db_mtime": {
            "trades": _mtime_iso(trades_db),
            "orders": _mtime_iso(orders_db),
            "signals": _mtime_iso(signals_db),
            "metrics": _mtime_iso(metrics_db),
        },
        "db_size_bytes": {
            "trades": _size_bytes(trades_db),
            "orders": _size_bytes(orders_db),
            "signals": _size_bytes(signals_db),
            "metrics": _size_bytes(metrics_db),
        },
        "service_active": {
            "quantrabbit": _systemd_is_active("quantrabbit.service"),
            "quant_health_snapshot": _systemd_is_active("quant-health-snapshot.service"),
            "quant_health_timer": _systemd_is_active("quant-health-snapshot.timer"),
            "quant_ssh_watchdog": _systemd_is_active("quant-ssh-watchdog.service"),
            "quant_ssh_timer": _systemd_is_active("quant-ssh-watchdog.timer"),
            "quant_bq_sync": _systemd_is_active("quant-bq-sync.service"),
        },
        "ssh_active": _systemd_is_active("ssh"),
        "sshd_active": _systemd_is_active("sshd"),
        "guest_agent_active": _systemd_is_active("google-guest-agent"),
        "ssh_port_listening": _port_listening(22),
        "disk_used_pct": _disk_usage_pct(Path("/")),
        "disk_free_mb": _free_mb(Path("/")),
    }
    try:
        load1, load5, load15 = os.getloadavg()
        snapshot["load_avg"] = [round(load1, 3), round(load5, 3), round(load15, 3)]
    except Exception:
        snapshot["load_avg"] = None
    return snapshot


def _write_local(snapshot: dict[str, Any]) -> None:
    logs_dir = PROJECT_ROOT / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "health_snapshot.json").write_text(
            json.dumps(snapshot, ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )
    except Exception:
        pass


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
    _write_local(snapshot)
    payload = json.dumps(snapshot, ensure_ascii=True, separators=(",", ":"))
    try:
        if storage is None:
            raise RuntimeError("google-cloud-storage not available")
        client = storage.Client(project=project_id) if project_id else storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)
        blob.cache_control = "no-cache"
        blob.upload_from_string(payload, content_type="application/json")
        logging.info("[HEALTH] snapshot uploaded bucket=%s object=%s", bucket_name, object_path)
        return
    except Exception as exc:  # noqa: BLE001
        logging.warning("[HEALTH] upload failed: %s", exc)
    if _upload_via_cli(bucket_name, object_path, payload):
        return
    if upload_json_via_metadata(
        bucket_name, object_path, payload, cache_control="no-cache"
    ):
        logging.info("[HEALTH] snapshot uploaded via metadata bucket=%s", bucket_name)


if __name__ == "__main__":
    main()

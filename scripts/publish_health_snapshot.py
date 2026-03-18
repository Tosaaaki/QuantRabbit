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
import urllib.error
import urllib.request
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
from utils.strategy_tags import normalize_strategy_lookup_key
from utils.strategy_tags import resolve_strategy_tag


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


def _load_recent_signals(
    db_path: Path, limit: int = 5
) -> Optional[list[dict[str, Any]]]:
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


def _systemd_unit_info(unit: str) -> Optional[dict[str, Any]]:
    if not shutil.which("systemctl"):
        return None
    output = _run_cmd(
        [
            "systemctl",
            "show",
            unit,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "Result",
            "-p",
            "NRestarts",
            "-p",
            "ActiveEnterTimestamp",
            "-p",
            "ExecMainStartTimestamp",
            "-p",
            "ExecMainExitTimestamp",
            "-p",
            "ExecMainStatus",
        ]
    )
    if output is None:
        return None
    info: dict[str, Any] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key] = value
    if not info:
        return None
    info["unit"] = unit
    return info


def _port_listening(port: int) -> Optional[bool]:
    pattern = re.compile(rf":{port}\b")
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


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                data[key] = value
    except Exception:
        return {}
    return data


def _load_env_chain(paths: list[Path]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for path in paths:
        merged.update(_parse_env_file(path))
    return merged


def _coerce_int(raw: object, default: int) -> int:
    try:
        return int(str(raw))
    except Exception:
        return default


def _age_sec_from_iso(raw: Optional[str]) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    return max(
        0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()
    )


def _age_sec_from_mtime(path: Path) -> Optional[float]:
    try:
        return max(0.0, datetime.now(timezone.utc).timestamp() - path.stat().st_mtime)
    except Exception:
        return None


def _artifact_integrity(
    path: Path,
    *,
    timestamp_fields: tuple[str, ...] = (),
    max_age_sec: Optional[int] = None,
) -> dict[str, Any]:
    payload = _load_json(path)
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "mtime": _mtime_iso(path),
        "size_bytes": _size_bytes(path),
    }
    timestamp_value = None
    if isinstance(payload, dict):
        for key in timestamp_fields:
            raw = payload.get(key)
            if isinstance(raw, str) and raw.strip():
                timestamp_value = raw.strip()
                break
    age_sec = (
        _age_sec_from_iso(timestamp_value)
        if timestamp_value
        else _age_sec_from_mtime(path)
    )
    if timestamp_value:
        info["timestamp"] = timestamp_value
    if age_sec is not None:
        info["age_sec"] = round(age_sec, 1)
    if max_age_sec is not None:
        info["max_age_sec"] = int(max_age_sec)
        info["fresh"] = bool(age_sec is not None and age_sec <= max_age_sec)
    info["_payload"] = payload
    return info


def _sqlite_table_exists(db_path: Path, table_name: str) -> Optional[bool]:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "select 1 from sqlite_master where type='table' and name=? limit 1;",
                (table_name,),
            )
            return cur.fetchone() is not None
    except Exception:
        return None


def _http_json(url: str, *, timeout_sec: float = 1.5) -> Optional[dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _merge_strategy_records(*sources: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for src in sources:
        for key, rec in src.items():
            bucket = merged.setdefault(
                key,
                {
                    "entry_active": False,
                    "exit_active": False,
                    "active": False,
                    "enabled": None,
                    "sources": set(),
                },
            )
            bucket["entry_active"] = bucket["entry_active"] or bool(
                getattr(rec, "entry_active", False)
            )
            bucket["exit_active"] = bucket["exit_active"] or bool(
                getattr(rec, "exit_active", False)
            )
            bucket["active"] = bucket["active"] or bool(getattr(rec, "active", False))
            if getattr(rec, "enabled", None) is not None:
                bucket["enabled"] = bool(getattr(rec, "enabled"))
            bucket["sources"].update(getattr(rec, "sources", set()) or set())
    return merged


def _canonical_known_strategy_key(raw: str, known_keys: list[str]) -> str:
    resolved = resolve_strategy_tag(raw, known_keys=known_keys) or raw
    lookup = normalize_strategy_lookup_key(resolved)
    if not lookup:
        return resolved
    matches = [
        candidate
        for candidate in known_keys
        if normalize_strategy_lookup_key(candidate) == lookup
    ]
    if not matches:
        return resolved
    matches.sort(
        key=lambda candidate: (
            0 if any(ch.isupper() for ch in candidate) else 1,
            -len(candidate),
            candidate,
        )
    )
    return matches[0]


def _participation_feedback_boosts(
    payload: Optional[dict[str, Any]],
    *,
    known_keys: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    strategies = payload.get("strategies") if isinstance(payload, dict) else None
    if not isinstance(strategies, dict):
        return {}
    boosted: dict[str, dict[str, Any]] = {}
    for raw_key, item in strategies.items():
        if not isinstance(item, dict):
            continue
        if str(item.get("action") or "").strip().lower() != "boost_participation":
            continue
        attempts = _coerce_int(item.get("attempts") or item.get("preflights"), 0)
        fills = _coerce_int(item.get("fills") or item.get("filled"), 0)
        lot_multiplier = float(
            item.get("lot_multiplier") or item.get("units_multiplier") or 1.0
        )
        probability_boost = float(
            item.get("probability_boost") or item.get("probability_offset") or 0.0
        )
        cadence_floor = float(item.get("cadence_floor") or 1.0)
        if attempts < 1 or fills < 1:
            continue
        if (
            max(lot_multiplier, 1.0) <= 1.0
            and max(probability_boost, 0.0) <= 0.0
            and max(cadence_floor, 1.0) <= 1.0
        ):
            continue
        strategy_key = (
            resolve_strategy_tag(str(raw_key or "").strip())
            or str(raw_key or "").strip()
        )
        if known_keys:
            strategy_key = _canonical_known_strategy_key(strategy_key, known_keys)
        if not strategy_key:
            continue
        current = boosted.get(strategy_key)
        boosted[strategy_key] = {
            "attempts": attempts + _coerce_int((current or {}).get("attempts"), 0),
            "fills": fills + _coerce_int((current or {}).get("fills"), 0),
        }
    return boosted


def _strategy_feedback_integrity(
    *,
    project_root: Path,
    logs_dir: Path,
    trades_db: Path,
    participation_payload: Optional[dict[str, Any]] = None,
    participation_fresh: bool = False,
) -> dict[str, Any]:
    max_age_sec = _coerce_int(os.getenv("HEALTH_STRATEGY_FEEDBACK_MAX_AGE_SEC"), 1800)
    info = _artifact_integrity(
        logs_dir / "strategy_feedback.json",
        timestamp_fields=("updated_at",),
        max_age_sec=max_age_sec,
    )
    payload = info.pop("_payload", None)
    strategies = payload.get("strategies") if isinstance(payload, dict) else None
    strategy_names = sorted((strategies or {}).keys())
    info["strategies_count"] = len(strategy_names)

    env = _load_env_chain(
        [
            project_root / "ops" / "env" / "quant-v2-runtime.env",
            project_root / "ops" / "env" / "quant-strategy-feedback.env",
            project_root / "ops" / "env" / "local-v2-stack.env",
        ]
    )
    lookback_days = _coerce_int(env.get("STRATEGY_FEEDBACK_LOOKBACK_DAYS"), 14)
    min_trades = _coerce_int(env.get("STRATEGY_FEEDBACK_MIN_TRADES"), 12)
    info["lookback_days"] = lookback_days
    info["min_trades"] = min_trades
    boosted_probe_strategies: dict[str, dict[str, Any]] = {}
    info["boosted_low_sample_strategies"] = []

    try:
        from analysis import strategy_feedback_worker as feedback_worker

        local_pid_dir = project_root / "logs" / "local_v2_stack" / "pids"
        systemd_dir = project_root / "systemd"
        running_services = feedback_worker._systemctl_running_services()
        if not running_services:
            running_services = feedback_worker._local_stack_running_services(
                local_pid_dir
            )
        discovered = _merge_strategy_records(
            feedback_worker._discover_from_control(),
            feedback_worker._discover_from_systemd(
                systemd_dir, running_services, datetime.now(timezone.utc)
            ),
        )
        boosted_probe_strategies = (
            _participation_feedback_boosts(
                participation_payload,
                known_keys=list(discovered.keys()),
            )
            if participation_fresh
            else {}
        )
        info["boosted_low_sample_strategies"] = sorted(boosted_probe_strategies.keys())
        stats_by_tag, latest_by_tag = feedback_worker._discover_from_trades(
            trades_db, lookback_days
        )
        if discovered:
            stats_by_tag, latest_by_tag = feedback_worker._remap_stats_to_known_keys(
                stats_by_tag,
                latest_by_tag,
                list(discovered.keys()),
            )
        active_strategies: list[str] = []
        eligible_active: list[str] = []
        eligible_missing: list[str] = []
        for tag in sorted(discovered.keys()):
            rec = discovered[tag]
            if rec.get("enabled") is False or not rec.get("entry_active"):
                continue
            active_strategies.append(tag)
            stats = stats_by_tag.get(tag)
            if stats is None or int(getattr(stats, "trades", 0)) < min_trades:
                if tag not in boosted_probe_strategies:
                    continue
            if stats is None and tag not in boosted_probe_strategies:
                continue
            eligible_active.append(tag)
            if tag not in strategy_names:
                eligible_missing.append(tag)
        info["active_strategies"] = active_strategies
        info["eligible_active_strategies"] = eligible_active
        info["eligible_missing_strategies"] = eligible_missing
        info["coverage_ok"] = not eligible_missing
    except Exception as exc:  # noqa: BLE001
        info["coverage_ok"] = False
        info["coverage_error"] = str(exc)
    return info


def _build_mechanism_integrity(
    *,
    project_root: Path,
    logs_dir: Path,
    orders_db: Path,
    trades_db: Path,
) -> dict[str, Any]:
    dynamic_alloc = _artifact_integrity(
        project_root / "config" / "dynamic_alloc.json",
        timestamp_fields=("as_of",),
        max_age_sec=_coerce_int(os.getenv("HEALTH_DYNAMIC_ALLOC_MAX_AGE_SEC"), 1800),
    )
    dynamic_payload = dynamic_alloc.pop("_payload", None)
    dynamic_alloc["strategies_count"] = len(
        (dynamic_payload or {}).get("strategies") or {}
    )

    pattern_book = _artifact_integrity(
        project_root / "config" / "pattern_book.json",
        max_age_sec=_coerce_int(os.getenv("HEALTH_PATTERN_BOOK_MAX_AGE_SEC"), 1800),
    )
    pattern_payload = pattern_book.pop("_payload", None)
    pattern_book["has_content"] = (
        bool(pattern_payload)
        if pattern_payload is not None
        else bool(pattern_book["size_bytes"])
    )

    forecast_runtime = _artifact_integrity(
        logs_dir / "forecast_improvement_latest.json",
        timestamp_fields=("generated_at",),
        max_age_sec=_coerce_int(
            os.getenv("HEALTH_FORECAST_RUNTIME_MAX_AGE_SEC"), 21600
        ),
    )
    forecast_payload = forecast_runtime.pop("_payload", None)
    runtime_overrides = (
        (forecast_payload or {}).get("runtime_overrides")
        if isinstance(forecast_payload, dict)
        else {}
    )
    forecast_runtime["verdict"] = (
        (forecast_payload or {}).get("verdict")
        if isinstance(forecast_payload, dict)
        else None
    )
    forecast_runtime["runtime_overrides_enabled"] = (
        runtime_overrides.get("enabled")
        if isinstance(runtime_overrides, dict)
        else None
    )

    forecast_service = {
        "port": 8302,
        "listening": _port_listening(8302),
        "health": _http_json("http://127.0.0.1:8302/health"),
    }
    forecast_health = forecast_service.get("health")
    forecast_service["ok"] = bool(
        (isinstance(forecast_health, dict) and forecast_health.get("ok") is True)
        or forecast_service.get("listening") is True
    )

    blackboard = {
        "entry_intent_board_table": _sqlite_table_exists(
            orders_db, "entry_intent_board"
        ),
        "recent_rows_24h": _safe_query(
            orders_db,
            "select count(*) from entry_intent_board where ts_epoch >= strftime('%s','now') - 86400;",
        ),
    }

    entry_path_summary = _artifact_integrity(
        logs_dir / "entry_path_summary_latest.json",
        timestamp_fields=("generated_at",),
        max_age_sec=_coerce_int(
            os.getenv("HEALTH_ENTRY_PATH_SUMMARY_MAX_AGE_SEC"), 1800
        ),
    )
    entry_path_payload = entry_path_summary.pop("_payload", None)
    entry_path_summary["strategies_count"] = len(
        (entry_path_payload or {}).get("strategies") or {}
    )
    entry_path_summary["orders_considered"] = (entry_path_payload or {}).get(
        "orders_considered"
    )

    participation_alloc = _artifact_integrity(
        project_root / "config" / "participation_alloc.json",
        timestamp_fields=("as_of",),
        max_age_sec=_coerce_int(
            os.getenv("HEALTH_PARTICIPATION_ALLOC_MAX_AGE_SEC"), 1800
        ),
    )
    participation_payload = participation_alloc.pop("_payload", None)
    participation_alloc["strategies_count"] = len(
        (participation_payload or {}).get("strategies") or {}
    )

    loser_cluster = _artifact_integrity(
        logs_dir / "loser_cluster_latest.json",
        timestamp_fields=("generated_at",),
        max_age_sec=_coerce_int(os.getenv("HEALTH_LOSER_CLUSTER_MAX_AGE_SEC"), 3600),
    )
    loser_cluster_payload = loser_cluster.pop("_payload", None)
    loser_cluster["strategies_count"] = len(
        (loser_cluster_payload or {}).get("strategies") or {}
    )
    loser_cluster["top_clusters_count"] = len(
        (loser_cluster_payload or {}).get("top_clusters") or []
    )

    auto_canary = _artifact_integrity(
        project_root / "config" / "auto_canary_overrides.json",
        timestamp_fields=("generated_at",),
        max_age_sec=_coerce_int(os.getenv("HEALTH_AUTO_CANARY_MAX_AGE_SEC"), 3600),
    )
    auto_canary_payload = auto_canary.pop("_payload", None)
    auto_canary["strategies_count"] = len(
        (auto_canary_payload or {}).get("strategies") or {}
    )

    macro_news_context = _artifact_integrity(
        logs_dir / "macro_news_context.json",
        timestamp_fields=("generated_at",),
        max_age_sec=_coerce_int(
            os.getenv("HEALTH_MACRO_NEWS_CONTEXT_MAX_AGE_SEC"), 3600
        ),
    )
    macro_news_payload = macro_news_context.pop("_payload", None)
    macro_news_context["event_severity"] = (macro_news_payload or {}).get(
        "event_severity"
    )
    macro_news_context["caution_window_active"] = (macro_news_payload or {}).get(
        "caution_window_active"
    )
    macro_news_context["source_error_count"] = _coerce_int(
        (macro_news_payload or {}).get("source_error_count"),
        0,
    )

    strategy_feedback = _strategy_feedback_integrity(
        project_root=project_root,
        logs_dir=logs_dir,
        trades_db=trades_db,
        participation_payload=(
            participation_payload if participation_alloc.get("exists") else None
        ),
        participation_fresh=participation_alloc.get("fresh") is not False,
    )

    missing: list[str] = []
    if not strategy_feedback.get("exists"):
        missing.append("strategy_feedback_missing")
    elif strategy_feedback.get("fresh") is False:
        missing.append("strategy_feedback_stale")
    if strategy_feedback.get("coverage_ok") is False:
        missing.append("strategy_feedback_coverage_gap")

    if not dynamic_alloc.get("exists"):
        missing.append("dynamic_alloc_missing")
    elif dynamic_alloc.get("fresh") is False:
        missing.append("dynamic_alloc_stale")

    if not pattern_book.get("exists"):
        missing.append("pattern_book_missing")
    elif pattern_book.get("fresh") is False:
        missing.append("pattern_book_stale")

    if not entry_path_summary.get("exists"):
        missing.append("entry_path_summary_missing")
    elif entry_path_summary.get("fresh") is False:
        missing.append("entry_path_summary_stale")

    if not participation_alloc.get("exists"):
        missing.append("participation_alloc_missing")
    elif participation_alloc.get("fresh") is False:
        missing.append("participation_alloc_stale")

    if not forecast_runtime.get("exists"):
        missing.append("forecast_runtime_missing")
    elif forecast_runtime.get("fresh") is False:
        missing.append("forecast_runtime_stale")

    if forecast_service.get("ok") is False:
        missing.append("forecast_service_down")
    if blackboard.get("entry_intent_board_table") is not True:
        missing.append("entry_intent_board_missing")
    if not loser_cluster.get("exists"):
        missing.append("loser_cluster_missing")
    elif loser_cluster.get("fresh") is False:
        missing.append("loser_cluster_stale")
    if not auto_canary.get("exists"):
        missing.append("auto_canary_missing")
    elif auto_canary.get("fresh") is False:
        missing.append("auto_canary_stale")
    if not macro_news_context.get("exists"):
        missing.append("macro_news_context_missing")
    elif macro_news_context.get("fresh") is False:
        missing.append("macro_news_context_stale")
    elif macro_news_context.get("source_error_count", 0) > 0:
        missing.append("macro_news_context_source_errors")

    return {
        "ok": not missing,
        "missing_mechanisms": missing,
        "strategy_feedback": strategy_feedback,
        "dynamic_alloc": dynamic_alloc,
        "pattern_book": pattern_book,
        "entry_path_summary": entry_path_summary,
        "participation_alloc": participation_alloc,
        "forecast_runtime": forecast_runtime,
        "forecast_service": forecast_service,
        "blackboard": blackboard,
        "loser_cluster": loser_cluster,
        "auto_canary": auto_canary,
        "macro_news_context": macro_news_context,
    }


def _upload_via_cli(bucket: str, object_path: str, payload: str) -> bool:
    target = f"gs://{bucket}/{object_path}"
    for cmd in (
        ["gcloud", "storage", "cp", "-", target],
        ["gsutil", "cp", "-", target],
    ):
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

    service_units = [
        "quantrabbit.service",
        "quant-main.service",
        "quant-ui-snapshot.service",
        "quant-health-snapshot.service",
        "quant-autotune-ui.service",
        "quant-ssh-watchdog.service",
        "quant-bq-sync.service",
    ]
    service_info: dict[str, Any] = {}
    for unit in service_units:
        info = _systemd_unit_info(unit)
        if info:
            service_info[unit] = info

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
            "select count(*) from trades where entry_time >= strftime('%Y-%m-%dT%H:%M:%S','now','-1 day');",
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
            "where ts >= strftime('%Y-%m-%dT%H:%M:%S','now','-1 hour') "
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
            "quant_main": _systemd_is_active("quant-main.service"),
            "quant_health_snapshot": _systemd_is_active(
                "quant-health-snapshot.service"
            ),
            "quant_health_timer": _systemd_is_active("quant-health-snapshot.timer"),
            "quant_ssh_watchdog": _systemd_is_active("quant-ssh-watchdog.service"),
            "quant_ssh_timer": _systemd_is_active("quant-ssh-watchdog.timer"),
            "quant_bq_sync": _systemd_is_active("quant-bq-sync.service"),
        },
        "service_info": service_info,
        "ssh_active": _systemd_is_active("ssh"),
        "sshd_active": _systemd_is_active("sshd"),
        "guest_agent_active": _systemd_is_active("google-guest-agent"),
        "ssh_port_listening": _port_listening(22),
        "disk_used_pct": _disk_usage_pct(Path("/")),
        "disk_free_mb": _free_mb(Path("/")),
    }
    snapshot["mechanism_integrity"] = _build_mechanism_integrity(
        project_root=repo_dir,
        logs_dir=logs_dir,
        orders_db=orders_db,
        trades_db=trades_db,
    )
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
    snapshot = _build_snapshot()
    _write_local(snapshot)
    payload = json.dumps(snapshot, ensure_ascii=True, separators=(",", ":"))

    upload_disabled = os.getenv("HEALTH_UPLOAD_DISABLE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if upload_disabled:
        logging.info("[HEALTH] upload disabled by HEALTH_UPLOAD_DISABLE")
        return

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
    try:
        if storage is None:
            raise RuntimeError("google-cloud-storage not available")
        client = storage.Client(project=project_id) if project_id else storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)
        blob.cache_control = "no-cache"
        blob.upload_from_string(payload, content_type="application/json")
        logging.info(
            "[HEALTH] snapshot uploaded bucket=%s object=%s", bucket_name, object_path
        )
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

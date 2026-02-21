#!/usr/bin/env python3
"""VM 上でトレード同期と BigQuery 送信を連続実行するパイプライン。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Optional
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analytics.bq_exporter import BigQueryExporter, _BQ_MAX_EXPORT
from analytics.gcs_publisher import GCSRealtimePublisher
from analytics.lot_pattern_analyzer import LotPatternAnalyzer
from execution.position_manager import PositionManager

LOGS_DIR = Path("logs")
LOG_FILE = Path("logs/pipeline.log")
METRICS_DB = Path("logs/metrics.db")
ORDERS_DB = Path("logs/orders.db")
SIGNALS_DB = Path("logs/signals.db")
HEALTH_SNAPSHOT = LOGS_DIR / "health_snapshot.json"
ERROR_LOG_PATHS = [
    p.strip()
    for p in os.getenv(
        "UI_ERROR_LOG_PATHS",
        f"{LOG_FILE},{LOGS_DIR / 'autotune_ui.log'}",
    ).split(",")
    if p.strip()
]
ERROR_LOG_MAX_LINES = int(os.getenv("UI_ERROR_LOG_MAX_LINES", "10"))
ERROR_LOG_MAX_BYTES = int(os.getenv("UI_ERROR_LOG_MAX_BYTES", "180000"))


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


PIPELINE_DB_READ_TIMEOUT_SEC = max(0.5, _env_float("PIPELINE_DB_READ_TIMEOUT_SEC", 3.0))
BQ_FAILURE_BACKOFF_BASE_SEC = max(5.0, _env_float("BQ_FAILURE_BACKOFF_BASE_SEC", 120.0))
BQ_FAILURE_BACKOFF_MAX_SEC = max(
    BQ_FAILURE_BACKOFF_BASE_SEC,
    _env_float("BQ_FAILURE_BACKOFF_MAX_SEC", 1800.0),
)


def _load_latest_metric(metric: str) -> Optional[float]:
    if not METRICS_DB.exists():
        return None
    try:
        with sqlite3.connect(METRICS_DB, timeout=PIPELINE_DB_READ_TIMEOUT_SEC) as con:
            cur = con.execute(
                "SELECT value FROM metrics WHERE metric = ? ORDER BY ts DESC LIMIT 1",
                (metric,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return float(row[0])
    except Exception:
        return None


def _tail_text(path: Path, max_bytes: int) -> str:
    try:
        size = path.stat().st_size
    except Exception:
        return ""
    if size <= 0:
        return ""
    try:
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(-max_bytes, os.SEEK_END)
            data = fh.read()
    except Exception:
        return ""
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _parse_log_line(line: str) -> dict:
    parts = line.split(" - ", 3)
    if len(parts) >= 3:
        ts = parts[0].strip()
        level = parts[1].strip()
        msg = parts[3].strip() if len(parts) >= 4 else ""
        return {"ts": ts, "level": level, "message": msg or line.strip()}
    return {"ts": None, "level": None, "message": line.strip()}


def _load_recent_log_errors(limit: int = 10) -> list[dict]:
    results: list[dict] = []
    for raw_path in ERROR_LOG_PATHS:
        path = Path(raw_path)
        if not path.exists():
            continue
        text = _tail_text(path, ERROR_LOG_MAX_BYTES)
        if not text:
            continue
        lines = [line for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            if len(results) >= limit:
                return results
            upper = line.upper()
            if "ERROR" not in upper and "CRITICAL" not in upper and "TRACEBACK" not in upper:
                continue
            parsed = _parse_log_line(line)
            parsed["source"] = path.name
            results.append(parsed)
    return results


def _load_health_snapshot() -> Optional[dict]:
    if not HEALTH_SNAPSHOT.exists():
        return None
    try:
        data = json.loads(HEALTH_SNAPSHOT.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _extract_order_meta(payload: dict) -> dict:
    meta: dict = {}
    entry_thesis = payload.get("entry_thesis") or (payload.get("meta") or {}).get("entry_thesis")
    if isinstance(entry_thesis, dict):
        limited: dict = {}
        for key in (
            "strategy_tag",
            "strategy",
            "worker_id",
            "worker",
            "focus_tag",
            "macro_regime",
            "micro_regime",
        ):
            if key in entry_thesis and entry_thesis.get(key) is not None:
                limited[key] = entry_thesis.get(key)
        if limited:
            meta["entry_thesis"] = limited
        strategy = entry_thesis.get("strategy_tag") or entry_thesis.get("strategy")
        if strategy:
            meta["strategy"] = strategy
        worker = entry_thesis.get("worker_id") or entry_thesis.get("worker")
        if worker:
            meta["worker"] = worker
        focus_tag = entry_thesis.get("focus_tag") or entry_thesis.get("focus")
        if focus_tag:
            meta["focus_tag"] = focus_tag
    if "strategy" not in meta:
        strategy = (
            payload.get("strategy_tag")
            or payload.get("strategy")
            or (payload.get("meta") or {}).get("strategy_tag")
            or (payload.get("meta") or {}).get("strategy")
        )
        if strategy:
            meta["strategy"] = strategy
    return meta


def _parse_json_object(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_order_side(side: object, units: object) -> Optional[str]:
    side_str = str(side or "").strip().lower()
    if side_str in {"buy", "sell"}:
        return side_str
    try:
        units_val = float(units or 0)
    except Exception:
        return None
    if units_val > 0:
        return "buy"
    if units_val < 0:
        return "sell"
    return None


def _load_order_context_rows(con: sqlite3.Connection, rows: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    client_ids = sorted({str(row.get("client_order_id") or "").strip() for row in rows if row.get("client_order_id")})
    ticket_ids = sorted({str(row.get("ticket_id") or "").strip() for row in rows if row.get("ticket_id")})
    if not client_ids and not ticket_ids:
        return {}, {}

    clauses: list[str] = []
    params: list[str] = []
    if client_ids:
        placeholders = ",".join("?" for _ in client_ids)
        clauses.append(f"client_order_id IN ({placeholders})")
        params.extend(client_ids)
    if ticket_ids:
        placeholders = ",".join("?" for _ in ticket_ids)
        clauses.append(f"ticket_id IN ({placeholders})")
        params.extend(ticket_ids)

    if not clauses:
        return {}, {}

    cur = con.execute(
        "SELECT ts, status, client_order_id, ticket_id, pocket, side, units, request_json "
        "FROM orders "
        f"WHERE ({' OR '.join(clauses)}) "
        "AND status NOT LIKE 'close_%' "
        "ORDER BY "
        "CASE status "
        "WHEN 'submit_attempt' THEN 0 "
        "WHEN 'filled' THEN 1 "
        "WHEN 'accepted' THEN 2 "
        "WHEN 'preflight_start' THEN 3 "
        "ELSE 9 END, "
        "ts ASC"
        ,
        tuple(params),
    )
    refs = [dict(r) for r in cur.fetchall()]

    by_client: dict[str, dict] = {}
    by_ticket: dict[str, dict] = {}
    for ref in refs:
        client_id = str(ref.get("client_order_id") or "").strip()
        ticket_id = str(ref.get("ticket_id") or "").strip()
        if client_id and client_id not in by_client:
            by_client[client_id] = ref
        if ticket_id and ticket_id not in by_ticket:
            by_ticket[ticket_id] = ref
    return by_client, by_ticket


def _merge_order_meta(primary_payload: dict, fallback_payload: dict) -> dict:
    primary = _extract_order_meta(primary_payload)
    if not fallback_payload:
        return primary
    fallback = _extract_order_meta(fallback_payload)
    if not fallback:
        return primary
    merged = dict(fallback)
    merged.update(primary)
    if isinstance(fallback.get("entry_thesis"), dict) and isinstance(primary.get("entry_thesis"), dict):
        thesis = dict(fallback["entry_thesis"])
        thesis.update(primary["entry_thesis"])
        merged["entry_thesis"] = thesis
    return merged


def _enrich_order_rows(rows: list[dict], con: sqlite3.Connection) -> list[dict]:
    by_client, by_ticket = _load_order_context_rows(con, rows)
    enriched: list[dict] = []
    for row in rows:
        client_id = str(row.get("client_order_id") or "").strip()
        ticket_id = str(row.get("ticket_id") or "").strip()
        ref = by_client.get(client_id) or by_ticket.get(ticket_id) or {}

        if not str(row.get("pocket") or "").strip():
            pocket_ref = str(ref.get("pocket") or "").strip()
            if pocket_ref:
                row["pocket"] = pocket_ref

        side = _normalize_order_side(row.get("side"), row.get("units"))
        if not side:
            side = _normalize_order_side(ref.get("side"), row.get("units"))
        if side:
            row["side"] = side

        payload = _parse_json_object(row.get("request_json"))
        ref_payload = _parse_json_object(ref.get("request_json"))
        meta = _merge_order_meta(payload, ref_payload)
        if meta:
            row.update(meta)

        row.pop("request_json", None)
        enriched.append(row)
    return enriched


def _load_last_orders(limit: int = 5) -> list[dict]:
    if not ORDERS_DB.exists():
        return []
    try:
        with sqlite3.connect(ORDERS_DB, timeout=PIPELINE_DB_READ_TIMEOUT_SEC) as con:
            con.row_factory = sqlite3.Row
            cur = con.execute(
                "SELECT ts, pocket, side, units, status, client_order_id, "
                "ticket_id, error_code, error_message, request_json "
                "FROM orders ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            rows = [dict(r) for r in cur.fetchall()]
            results = _enrich_order_rows(rows, con)
        return results
    except Exception:
        return []


def _load_recent_order_errors(limit: int = 8, hours: int = 24) -> list[dict]:
    if not ORDERS_DB.exists():
        return []
    try:
        with sqlite3.connect(ORDERS_DB, timeout=PIPELINE_DB_READ_TIMEOUT_SEC) as con:
            con.row_factory = sqlite3.Row
            cur = con.execute(
                "SELECT ts, pocket, side, units, status, client_order_id, "
                "ticket_id, error_code, error_message, request_json "
                "FROM orders "
                "WHERE ts >= strftime('%Y-%m-%dT%H:%M:%S', 'now', ?) "
                "AND (error_code IS NOT NULL AND error_code != '' "
                "OR status LIKE 'error%' OR status LIKE 'reject%') "
                "ORDER BY ts DESC LIMIT ?",
                (f"-{int(hours)} hour", int(limit)),
            )
            rows = [dict(r) for r in cur.fetchall()]
            results = _enrich_order_rows(rows, con)
        return results
    except Exception:
        return []


def _load_order_status_counts(limit: int = 8, hours: int = 1) -> list[dict]:
    if not ORDERS_DB.exists():
        return []
    try:
        with sqlite3.connect(ORDERS_DB, timeout=PIPELINE_DB_READ_TIMEOUT_SEC) as con:
            con.row_factory = sqlite3.Row
            cur = con.execute(
                "SELECT status, count(*) AS count FROM orders "
                "WHERE ts >= strftime('%Y-%m-%dT%H:%M:%S', 'now', ?) "
                "GROUP BY status ORDER BY count DESC LIMIT ?",
                (f"-{int(hours)} hour", int(limit)),
            )
            rows = [dict(r) for r in cur.fetchall()]
        return rows
    except Exception:
        return []


def _load_last_signal_ts_ms() -> Optional[int]:
    if not SIGNALS_DB.exists():
        return None
    try:
        with sqlite3.connect(SIGNALS_DB, timeout=PIPELINE_DB_READ_TIMEOUT_SEC) as con:
            cur = con.execute("SELECT max(ts_ms) FROM signals")
            row = cur.fetchone()
        if not row:
            return None
        val = row[0]
        return int(val) if val is not None else None
    except Exception:
        return None


def _load_recent_signals(limit: int = 5) -> list[dict]:
    if not SIGNALS_DB.exists():
        return []
    try:
        with sqlite3.connect(SIGNALS_DB, timeout=PIPELINE_DB_READ_TIMEOUT_SEC) as con:
            cur = con.execute(
                "SELECT ts_ms, payload FROM signals ORDER BY ts_ms DESC LIMIT ?",
                (int(limit),),
            )
            rows: list[dict] = []
            for ts_ms, payload in cur.fetchall():
                item: dict = {"ts_ms": ts_ms}
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


def _load_last_metric_ts(metric: str) -> Optional[str]:
    if not METRICS_DB.exists():
        return None
    try:
        with sqlite3.connect(METRICS_DB, timeout=PIPELINE_DB_READ_TIMEOUT_SEC) as con:
            cur = con.execute(
                "SELECT ts FROM metrics WHERE metric = ? ORDER BY ts DESC LIMIT 1",
                (metric,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return str(row[0]) if row[0] is not None else None
    except Exception:
        return None


def _setup_logging(verbose: bool) -> None:
    LOG_FILE.parent.mkdir(exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def _run_cycle(
    pm: PositionManager, gcs_publisher: GCSRealtimePublisher | None, ui_recent: int
) -> list[dict]:
    logging.info("[PIPELINE] sync_trades start")
    new_trades = pm.sync_trades()

    if gcs_publisher and gcs_publisher.enabled:
        try:
            open_positions = pm.get_open_positions()
        except Exception as exc:  # noqa: BLE001
            logging.exception("[PIPELINE] open positions 取得に失敗: %s", exc)
            open_positions = {}
        recent_trades = pm.fetch_recent_trades(limit=ui_recent)
        metrics = pm.get_performance_summary()
        if isinstance(metrics, dict):
            data_lag_ms = _load_latest_metric("data_lag_ms")
            decision_latency_ms = _load_latest_metric("decision_latency_ms")
            if data_lag_ms is not None:
                metrics["data_lag_ms"] = data_lag_ms
            if decision_latency_ms is not None:
                metrics["decision_latency_ms"] = decision_latency_ms
            for key in (
                "account.nav",
                "account.balance",
                "account.free_margin_ratio",
                "account.margin_usage_ratio",
                "account.health_buffer",
            ):
                value = _load_latest_metric(key)
                if value is not None:
                    metrics[key] = value
            health_snapshot = _load_health_snapshot()
            if health_snapshot:
                metrics["health_snapshot"] = health_snapshot
            last_orders = _load_last_orders()
            metrics["orders_last"] = last_orders
            status_counts = _load_order_status_counts()
            metrics["orders_status_1h"] = status_counts
            metrics["orders_errors_recent"] = _load_recent_order_errors()
            metrics["log_errors_recent"] = _load_recent_log_errors(
                limit=ERROR_LOG_MAX_LINES
            )
            last_signal_ts = _load_last_signal_ts_ms()
            if last_signal_ts is not None:
                metrics["signals_last_ts_ms"] = last_signal_ts
            recent_signals = _load_recent_signals()
            metrics["signals_recent"] = recent_signals
            healthbeat_ts = _load_last_metric_ts("healthbeat")
            metrics["healthbeat_ts"] = healthbeat_ts
        try:
            gcs_publisher.publish_snapshot(
                new_trades=new_trades,
                recent_trades=recent_trades,
                open_positions=open_positions,
                metrics=metrics,
                snapshot_mode="full",
                snapshot_source="gcs",
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception("[PIPELINE] GCS スナップショット更新に失敗: %s", exc)

    return new_trades


def _sync_remote_logs(dest_dir: Path) -> None:
    """Copy key SQLite/log snapshots to a remote_logs directory."""

    target = dest_dir.expanduser()
    target.mkdir(parents=True, exist_ok=True)
    files = [
        Path("logs/trades.db"),
        Path("logs/orders.db"),
        Path("logs/metrics.db"),
        Path("logs/oanda/candles_M1_latest.json"),
        Path("logs/oanda/candles_H1_latest.json"),
        Path("logs/oanda/candles_H4_latest.json"),
    ]
    copied = 0
    for src in files:
        if src.exists():
            shutil.copy2(src, target / src.name)
            copied += 1
    oanda_dir = Path("logs/oanda")
    if oanda_dir.exists():
        txns = sorted(oanda_dir.glob("transactions_*.jsonl"))
        if txns:
            latest = txns[-1]
            shutil.copy2(latest, target / latest.name)
            copied += 1
            meta = latest.with_name(f"{latest.stem}_meta.json")
            if meta.exists():
                shutil.copy2(meta, target / meta.name)
                copied += 1
    logging.info("[REMOTE] synced %d files to %s", copied, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval",
        type=float,
        default=120.0,
        help="ループ間隔（秒）。--once 指定時は無視される。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="1 回の BigQuery export 上限行数。未指定なら環境変数に従う。",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="1 サイクルのみ実行して終了。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG ログを有効化。",
    )
    parser.add_argument(
        "--bq-interval",
        type=float,
        default=600.0,
        help="BigQuery へのエクスポート実行間隔（秒）。0 以下で新規トレード発生時のみ。",
    )
    parser.add_argument(
        "--no-bq-on-new",
        action="store_true",
        help="新規トレードがあっても即時 BigQuery export を行わない。",
    )
    parser.add_argument(
        "--ui-recent",
        type=int,
        default=50,
        help="UI スナップショットに含める直近トレード件数。",
    )
    parser.add_argument(
        "--disable-gcs",
        action="store_true",
        help="GCS へのリアルタイム出力を抑止する。",
    )
    parser.add_argument(
        "--disable-lot-insights",
        action="store_true",
        help="BigQuery 解析によるロットインサイト生成を無効化する。",
    )
    parser.add_argument(
        "--disable-bq",
        action="store_true",
        help="BigQuery エクスポートを完全に無効化する。Lot insights もスキップ。",
    )
    parser.add_argument(
        "--remote-dir",
        type=Path,
        default=None,
        help="Sync trades/candles snapshots to the given directory (e.g. remote_logs_current).",
    )
    parser.add_argument(
        "--remote-sync-interval",
        type=float,
        default=300.0,
        help="Minimum seconds between remote log syncs (default: 300).",
    )
    args = parser.parse_args()
    _setup_logging(args.verbose)
    remote_dir = args.remote_dir.expanduser() if args.remote_dir else None

    if args.disable_bq:
        logging.error("[PIPELINE] --disable-bq is not supported (BigQuery required).")
        return 2
    if args.disable_lot_insights:
        logging.error("[PIPELINE] --disable-lot-insights is not supported (BigQuery required).")
        return 2

    try:
        exporter = BigQueryExporter()
    except Exception as exc:  # pragma: no cover - defensive
        logging.error("[PIPELINE] BigQuery exporter init failed: %s", exc)
        return 2
    gcs_publisher = None if args.disable_gcs else GCSRealtimePublisher()
    try:
        analyzer = LotPatternAnalyzer()
    except Exception as exc:  # pragma: no cover - defensive
        logging.error("[PIPELINE] Lot insights init failed: %s", exc)
        return 2
    pm = PositionManager()
    stop_requested = False
    last_bq_export = 0.0
    bq_failure_count = 0
    bq_pause_until = 0.0
    bq_interval = args.bq_interval
    bq_on_new = not args.no_bq_on_new
    last_remote_sync = 0.0

    def _handle_signal(signum: int, _frame: Optional[object]) -> None:
        nonlocal stop_requested
        logging.info("[PIPELINE] signal %s を受信。終了準備...", signum)
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    try:
        while True:
            try:
                new_trades = _run_cycle(pm, gcs_publisher, args.ui_recent)

                now = time.monotonic()
                if remote_dir and args.remote_sync_interval > 0:
                    if now - last_remote_sync >= args.remote_sync_interval:
                        try:
                            _sync_remote_logs(remote_dir)
                            last_remote_sync = now
                        except Exception as exc:  # noqa: BLE001
                            logging.exception("[REMOTE] sync failed: %s", exc)
                run_bq = False
                if args.once:
                    run_bq = True
                elif bq_interval <= 0:
                    run_bq = bool(new_trades) if bq_on_new else False
                else:
                    if now - last_bq_export >= bq_interval:
                        run_bq = True
                    elif bq_on_new and new_trades and now - last_bq_export >= 5.0:
                        # 新規トレードがあり、最後の export から短時間経過した場合でも水平分散
                        run_bq = True

                if run_bq and exporter:
                    if now < bq_pause_until:
                        remaining = max(0.0, bq_pause_until - now)
                        logging.warning(
                            "[PIPELINE] BQ export cooldown active; skip for %.1fs (failures=%d)",
                            remaining,
                            bq_failure_count,
                        )
                    else:
                        try:
                            stats = exporter.export(limit=args.limit or _BQ_MAX_EXPORT)
                            bq_failure_count = 0
                            bq_pause_until = 0.0
                            logging.info(
                                "[PIPELINE] export done rows=%s last_updated=%s",
                                stats.exported,
                                stats.last_updated_at,
                            )
                            try:
                                insights = analyzer.run()
                                logging.info(
                                    "[PIPELINE] lot insights generated=%d",
                                    len(insights),
                                )
                            except Exception as exc:  # noqa: BLE001
                                logging.exception("[PIPELINE] lot insights 生成に失敗: %s", exc)
                            last_bq_export = now
                        except Exception as exc:  # noqa: BLE001
                            bq_failure_count += 1
                            power = min(10, max(0, bq_failure_count - 1))
                            cooldown = min(
                                BQ_FAILURE_BACKOFF_MAX_SEC,
                                BQ_FAILURE_BACKOFF_BASE_SEC * (2 ** power),
                            )
                            bq_pause_until = time.monotonic() + cooldown
                            last_bq_export = now
                            logging.exception(
                                "[PIPELINE] BQ export failed (failures=%d cooldown=%.1fs): %s",
                                bq_failure_count,
                                cooldown,
                                exc,
                            )
            except Exception as exc:  # noqa: BLE001
                logging.exception("[PIPELINE] サイクル失敗: %s", exc)
            if args.once or stop_requested:
                break
            sleep_for = max(args.interval, 5.0)
            logging.debug("[PIPELINE] sleeping %.1fs", sleep_for)
            for _ in range(int(sleep_for)):
                if stop_requested:
                    break
                time.sleep(1)
            if stop_requested:
                break
            residual = sleep_for - int(sleep_for)
            if residual > 0 and not stop_requested:
                time.sleep(residual)
    finally:
        pm.close()

    logging.info("[PIPELINE] shutdown complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

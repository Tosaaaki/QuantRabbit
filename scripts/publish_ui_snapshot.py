#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
import sqlite3
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
try:
    os.chdir(PROJECT_ROOT)
except Exception:
    pass

LOGS_DIR = PROJECT_ROOT / "logs"
try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

from analytics.gcs_publisher import GCSRealtimePublisher

METRICS_DB = LOGS_DIR / "metrics.db"
ORDERS_DB = LOGS_DIR / "orders.db"
SIGNALS_DB = LOGS_DIR / "signals.db"
TRADES_DB = LOGS_DIR / "trades.db"
HEALTH_SNAPSHOT = LOGS_DIR / "health_snapshot.json"
PIPELINE_LOG = LOGS_DIR / "pipeline.log"
DB_READ_TIMEOUT_SEC = float(os.getenv("UI_DB_READ_TIMEOUT_SEC", "0.2"))
SYNC_TRADES_ENABLED = os.getenv("UI_SNAPSHOT_SYNC_TRADES", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
SYNC_TRADES_LITE_ENABLED = os.getenv(
    "UI_SNAPSHOT_SYNC_TRADES_LITE", "0"
).strip().lower() in {
    "1",
    "true",
    "yes",
}
SYNC_TRADES_TTL_SEC = float(os.getenv("UI_SNAPSHOT_SYNC_TTL_SEC", "60"))
SYNC_TRADES_MARKER = Path(
    os.getenv("UI_SNAPSHOT_SYNC_MARKER", "logs/ui_snapshot_sync.json")
)
LITE_SNAPSHOT_FAST = (
    os.getenv("UI_SNAPSHOT_LITE_MODE", "full").strip().lower()
    in {"fast", "minimal"}
)
INCLUDE_POSITIONS = os.getenv("UI_SNAPSHOT_INCLUDE_POSITIONS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
ERROR_LOG_PATHS = [
    p.strip()
    for p in os.getenv(
        "UI_ERROR_LOG_PATHS",
        f"{PIPELINE_LOG},{LOGS_DIR / 'autotune_ui.log'}",
    ).split(",")
    if p.strip()
]
ERROR_LOG_MAX_LINES = int(os.getenv("UI_ERROR_LOG_MAX_LINES", "10"))
ERROR_LOG_MAX_BYTES = int(os.getenv("UI_ERROR_LOG_MAX_BYTES", "180000"))
SKIP_OANDA = os.getenv("UI_SNAPSHOT_SKIP_OANDA", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


def _load_latest_metric(metric: str) -> Optional[float]:
    if not METRICS_DB.exists():
        return None
    try:
        con = sqlite3.connect(METRICS_DB, timeout=DB_READ_TIMEOUT_SEC)
        cur = con.execute(
            "SELECT value FROM metrics WHERE metric = ? ORDER BY ts DESC LIMIT 1",
            (metric,),
        )
        row = cur.fetchone()
        con.close()
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


def _load_last_metric_ts(metric: str) -> Optional[str]:
    if not METRICS_DB.exists():
        return None
    try:
        con = sqlite3.connect(METRICS_DB, timeout=DB_READ_TIMEOUT_SEC)
        cur = con.execute(
            "SELECT ts FROM metrics WHERE metric = ? ORDER BY ts DESC LIMIT 1",
            (metric,),
        )
        row = cur.fetchone()
        con.close()
        if not row:
            return None
        return str(row[0]) if row[0] is not None else None
    except Exception:
        return None


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
        con = sqlite3.connect(ORDERS_DB, timeout=DB_READ_TIMEOUT_SEC)
        con.row_factory = sqlite3.Row
        cur = con.execute(
            "SELECT ts, pocket, side, units, status, client_order_id, "
            "ticket_id, error_code, error_message, request_json "
            "FROM orders ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        )
        rows = [dict(r) for r in cur.fetchall()]
        results = _enrich_order_rows(rows, con)
        con.close()
        return results
    except Exception:
        return []


def _load_recent_order_errors(limit: int = 8, hours: int = 24) -> list[dict]:
    if not ORDERS_DB.exists():
        return []
    try:
        con = sqlite3.connect(ORDERS_DB, timeout=DB_READ_TIMEOUT_SEC)
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
        con.close()
        return results
    except Exception:
        return []


def _load_order_status_counts(limit: int = 8, hours: int = 1) -> list[dict]:
    if not ORDERS_DB.exists():
        return []
    try:
        con = sqlite3.connect(ORDERS_DB, timeout=DB_READ_TIMEOUT_SEC)
        con.row_factory = sqlite3.Row
        cur = con.execute(
            "SELECT status, count(*) AS count FROM orders "
            "WHERE ts >= strftime('%Y-%m-%dT%H:%M:%S', 'now', ?) "
            "GROUP BY status ORDER BY count DESC LIMIT ?",
            (f"-{int(hours)} hour", int(limit)),
        )
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows
    except Exception:
        return []


def _load_last_signal_ts_ms() -> Optional[int]:
    if not SIGNALS_DB.exists():
        return _load_last_order_ts_ms()
    try:
        con = sqlite3.connect(SIGNALS_DB, timeout=DB_READ_TIMEOUT_SEC)
        cur = con.execute("SELECT max(ts_ms) FROM signals")
        row = cur.fetchone()
        con.close()
        if not row:
            return _load_last_order_ts_ms()
        val = row[0]
        sig_ts = int(val) if val is not None else None
    except Exception:
        sig_ts = None
    order_ts = _load_last_order_ts_ms()
    if order_ts is None:
        return sig_ts
    if sig_ts is None:
        return order_ts
    if order_ts > sig_ts + 30 * 60 * 1000:
        return order_ts
    return sig_ts


def _parse_ts_ms(ts: object) -> Optional[int]:
    if ts is None:
        return None
    try:
        dt = datetime.fromisoformat(str(ts))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _load_last_order_ts_ms() -> Optional[int]:
    if not ORDERS_DB.exists():
        return None
    try:
        con = sqlite3.connect(ORDERS_DB, timeout=DB_READ_TIMEOUT_SEC)
        cur = con.execute("SELECT max(ts) FROM orders WHERE request_json IS NOT NULL")
        row = cur.fetchone()
        con.close()
        if not row:
            return None
        return _parse_ts_ms(row[0])
    except Exception:
        return None


def _load_recent_order_signals(limit: int = 5) -> list[dict]:
    if not ORDERS_DB.exists():
        return []
    try:
        con = sqlite3.connect(ORDERS_DB, timeout=DB_READ_TIMEOUT_SEC)
        con.row_factory = sqlite3.Row
        cur = con.execute(
            "SELECT ts, pocket, side, units, status, client_order_id, request_json "
            "FROM orders WHERE request_json IS NOT NULL "
            "AND status NOT LIKE 'close_%' "
            "ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        )
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
    except Exception:
        return []

    results: list[dict] = []
    for row in rows:
        ts_ms = _parse_ts_ms(row.get("ts"))
        if ts_ms is None:
            continue
        try:
            payload = json.loads(row.get("request_json") or "{}")
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        entry_thesis = payload.get("entry_thesis") or {}
        if not isinstance(entry_thesis, dict):
            entry_thesis = {}
        strategy = (
            entry_thesis.get("strategy_tag")
            or payload.get("strategy_tag")
            or payload.get("strategy")
            or (payload.get("meta") or {}).get("strategy_tag")
        )
        if not strategy:
            continue
        confidence = entry_thesis.get("confidence") or payload.get("confidence")
        side = (row.get("side") or "").lower()
        action = "OPEN_LONG" if side == "buy" else "OPEN_SHORT" if side == "sell" else None
        results.append(
            {
                "ts_ms": ts_ms,
                "pocket": row.get("pocket"),
                "strategy": strategy,
                "confidence": confidence,
                "action": action,
                "client_order_id": row.get("client_order_id"),
                "proposed_units": row.get("units"),
            }
        )
    return results


def _load_recent_signals(limit: int = 5) -> list[dict]:
    if not SIGNALS_DB.exists():
        return _load_recent_order_signals(limit=limit)
    try:
        con = sqlite3.connect(SIGNALS_DB, timeout=DB_READ_TIMEOUT_SEC)
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
        con.close()
        if rows:
            order_signals = _load_recent_order_signals(limit=limit)
            if order_signals:
                sig_ts = rows[0].get("ts_ms")
                ord_ts = order_signals[0].get("ts_ms")
                if isinstance(sig_ts, int) and isinstance(ord_ts, int):
                    if ord_ts > sig_ts + 30 * 60 * 1000:
                        return order_signals
            return rows
        return _load_recent_order_signals(limit=limit)
    except Exception:
        return _load_recent_order_signals(limit=limit)


def _load_recent_trades(limit: int = 50) -> list[dict]:
    if not TRADES_DB.exists():
        return []
    try:
        con = sqlite3.connect(TRADES_DB, timeout=DB_READ_TIMEOUT_SEC)
        con.row_factory = sqlite3.Row
        cur = con.execute(
            """
            SELECT ticket_id, pocket, instrument, units, closed_units, entry_price, close_price,
                   fill_price, pl_pips, realized_pl, commission, financing,
                   entry_time, close_time, close_reason,
                   state, updated_at,
                   strategy_tag, strategy, client_order_id, entry_thesis
            FROM trades
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows
    except Exception:
        return []


def _should_sync_trades() -> bool:
    if not SYNC_TRADES_ENABLED:
        return False
    if SYNC_TRADES_TTL_SEC <= 0:
        return True
    if not SYNC_TRADES_MARKER.exists():
        return True
    try:
        data = json.loads(SYNC_TRADES_MARKER.read_text(encoding="utf-8"))
    except Exception:
        return True
    last_ts = float(data.get("ts") or 0.0)
    return (time.time() - last_ts) >= SYNC_TRADES_TTL_SEC


def _mark_sync_trades(count: int) -> None:
    try:
        SYNC_TRADES_MARKER.parent.mkdir(parents=True, exist_ok=True)
        SYNC_TRADES_MARKER.write_text(
            json.dumps({"ts": time.time(), "count": int(count)}),
            encoding="utf-8",
        )
    except Exception:
        pass


def _init_position_manager():
    if SKIP_OANDA:
        logging.info("[UI] UI_SNAPSHOT_SKIP_OANDA=1 -> OANDA 呼び出しをスキップします。")
        return None
    try:
        from execution.position_manager import PositionManager
    except Exception as exc:  # noqa: BLE001
        logging.warning("[UI] PositionManager import failed: %s", exc)
        return None
    try:
        return PositionManager()
    except Exception as exc:  # noqa: BLE001
        logging.warning("[UI] PositionManager init failed: %s", exc)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish UI snapshot to GCS.")
    parser.add_argument(
        "--recent",
        type=int,
        default=int(os.getenv("UI_RECENT_TRADES_LIMIT", "80")),
        help="Number of recent trades to include.",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Skip OANDA calls and publish a lightweight snapshot.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    gcs = GCSRealtimePublisher()
    if not gcs.enabled:
        logging.warning("[UI] GCS publisher is disabled")
        return 1

    metrics: dict = {}
    new_trades: list = []
    recent_trades = _load_recent_trades(limit=int(args.recent))
    open_positions: dict = {}
    pm = _init_position_manager()
    if pm is not None:
        if args.lite:
            try:
                if SYNC_TRADES_ENABLED and SYNC_TRADES_LITE_ENABLED and _should_sync_trades():
                    try:
                        synced = pm.sync_trades()
                        _mark_sync_trades(len(synced or []))
                    except Exception as exc:  # noqa: BLE001
                        logging.warning("[UI] sync_trades(lite) failed: %s", exc)
                recent_trades = pm.fetch_recent_trades(limit=int(args.recent))
                metrics = pm.get_performance_summary()
            except Exception as exc:  # noqa: BLE001
                logging.warning("[UI] get_performance_summary failed: %s", exc)
                metrics = {}
            if (not LITE_SNAPSHOT_FAST) or INCLUDE_POSITIONS:
                try:
                    open_positions = pm.get_open_positions(include_unknown=True)
                except Exception as exc:  # noqa: BLE001
                    logging.warning("[UI] get_open_positions failed: %s", exc)
                    open_positions = {}
        else:
            try:
                new_trades = pm.sync_trades()
            except Exception as exc:  # noqa: BLE001
                logging.warning("[UI] sync_trades failed: %s", exc)
                new_trades = []
            try:
                open_positions = pm.get_open_positions(include_unknown=True)
            except Exception as exc:  # noqa: BLE001
                logging.warning("[UI] get_open_positions failed: %s", exc)
                open_positions = {}
            try:
                recent_trades = pm.fetch_recent_trades(limit=int(args.recent))
            except Exception as exc:  # noqa: BLE001
                logging.warning("[UI] fetch_recent_trades failed: %s", exc)
                recent_trades = []
            try:
                metrics = pm.get_performance_summary()
            except Exception as exc:  # noqa: BLE001
                logging.warning("[UI] get_performance_summary failed: %s", exc)
                metrics = {}
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
        healthbeat_ts = _load_last_metric_ts("healthbeat")
        metrics["healthbeat_ts"] = healthbeat_ts
        if not (args.lite and LITE_SNAPSHOT_FAST):
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

    snapshot_mode = (
        "lite-fast"
        if args.lite and LITE_SNAPSHOT_FAST
        else ("lite" if args.lite else "full")
    )

    try:
        gcs.publish_snapshot(
            new_trades=new_trades,
            recent_trades=recent_trades,
            open_positions=open_positions,
            metrics=metrics,
            snapshot_mode=snapshot_mode,
            snapshot_source="gcs",
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("[UI] publish_snapshot failed: %s", exc)
        return 2
    finally:
        if pm is not None:
            try:
                pm.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

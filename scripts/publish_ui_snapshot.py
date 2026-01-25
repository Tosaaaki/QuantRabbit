#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analytics.gcs_publisher import GCSRealtimePublisher
from execution.position_manager import PositionManager

METRICS_DB = Path("logs/metrics.db")
ORDERS_DB = Path("logs/orders.db")
SIGNALS_DB = Path("logs/signals.db")
TRADES_DB = Path("logs/trades.db")
DB_READ_TIMEOUT_SEC = float(os.getenv("UI_DB_READ_TIMEOUT_SEC", "0.2"))
LITE_SNAPSHOT_FAST = (
    os.getenv("UI_SNAPSHOT_LITE_MODE", "full").strip().lower()
    in {"fast", "minimal"}
)


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


def _load_last_orders(limit: int = 5) -> list[dict]:
    if not ORDERS_DB.exists():
        return []
    try:
        con = sqlite3.connect(ORDERS_DB, timeout=DB_READ_TIMEOUT_SEC)
        con.row_factory = sqlite3.Row
        cur = con.execute(
            "SELECT ts, pocket, side, units, status, client_order_id "
            "FROM orders ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        )
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows
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
            "WHERE ts >= datetime('now', ?) "
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
        return None
    try:
        con = sqlite3.connect(SIGNALS_DB, timeout=DB_READ_TIMEOUT_SEC)
        cur = con.execute("SELECT max(ts_ms) FROM signals")
        row = cur.fetchone()
        con.close()
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
        return rows
    except Exception:
        return []


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
                   state, updated_at
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish UI snapshot to GCS.")
    parser.add_argument(
        "--recent",
        type=int,
        default=50,
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
    pm = None
    if args.lite:
        new_trades = []
        if LITE_SNAPSHOT_FAST:
            open_positions = {}
        else:
            try:
                pm = PositionManager()
                open_positions = pm.get_open_positions()
            except Exception as exc:  # noqa: BLE001
                logging.warning("[UI] get_open_positions failed: %s", exc)
                open_positions = {}
        recent_trades = _load_recent_trades(limit=int(args.recent))
    else:
        pm = PositionManager()
        try:
            new_trades = pm.sync_trades()
        except Exception as exc:  # noqa: BLE001
            logging.warning("[UI] sync_trades failed: %s", exc)
            new_trades = []
        try:
            open_positions = pm.get_open_positions()
        except Exception as exc:  # noqa: BLE001
            logging.warning("[UI] get_open_positions failed: %s", exc)
            open_positions = {}
        try:
            recent_trades = pm.fetch_recent_trades(limit=int(args.recent))
        except Exception as exc:  # noqa: BLE001
            logging.warning("[UI] fetch_recent_trades failed: %s", exc)
            recent_trades = []
        metrics = pm.get_performance_summary()
    if isinstance(metrics, dict):
        data_lag_ms = _load_latest_metric("data_lag_ms")
        decision_latency_ms = _load_latest_metric("decision_latency_ms")
        if data_lag_ms is not None:
            metrics["data_lag_ms"] = data_lag_ms
        if decision_latency_ms is not None:
            metrics["decision_latency_ms"] = decision_latency_ms
        healthbeat_ts = _load_last_metric_ts("healthbeat")
        metrics["healthbeat_ts"] = healthbeat_ts
        if not (args.lite and LITE_SNAPSHOT_FAST):
            last_orders = _load_last_orders()
            metrics["orders_last"] = last_orders
            status_counts = _load_order_status_counts()
            metrics["orders_status_1h"] = status_counts
            last_signal_ts = _load_last_signal_ts_ms()
            if last_signal_ts is not None:
                metrics["signals_last_ts_ms"] = last_signal_ts
            recent_signals = _load_recent_signals()
            metrics["signals_recent"] = recent_signals

    try:
        gcs.publish_snapshot(
            new_trades=new_trades,
            recent_trades=recent_trades,
            open_positions=open_positions,
            metrics=metrics,
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

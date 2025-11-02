"""
Utilities for writing trade snapshots into the local SQLite ledger.

This is used by the trading services to guarantee that every order attempt
is recorded locally even when Firestore or remote sync is unavailable.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

TRADE_LOGGER_DB = Path(os.getenv("TRADE_LOGGER_DB", "logs/trades.db"))

_LOCK = threading.Lock()
_CONN: Optional[sqlite3.Connection] = None


def _ensure_connection() -> Optional[sqlite3.Connection]:
    global _CONN
    with _LOCK:
        if _CONN is not None:
            return _CONN
        try:
            TRADE_LOGGER_DB.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(TRADE_LOGGER_DB, check_same_thread=False)
            conn.row_factory = sqlite3.Row
        except Exception:
            return None
        try:
            _ensure_schema(conn)
        except Exception:
            conn.close()
            return None
        _CONN = conn
        return _CONN


def _ensure_schema(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ticket_id TEXT UNIQUE,
          pocket TEXT,
          instrument TEXT,
          units INTEGER,
          entry_price REAL,
          close_price REAL,
          pl_pips REAL,
          entry_time TEXT,
          close_time TEXT,
          strategy TEXT,
          macro_regime TEXT,
          micro_regime TEXT,
          close_reason TEXT,
          realized_pl REAL,
          unrealized_pl REAL,
          state TEXT,
          updated_at TEXT,
          version TEXT
        )
        """
    )
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_ticket_logger ON trades(ticket_id)"
    )
    conn.commit()


def _normalize_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_units(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_time(value: Any) -> Optional[str]:
    if not value:
        return None
    try:
        text = str(value)
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    except Exception:
        return None


def _resolve_state(doc: Dict[str, Any]) -> str:
    state = str(doc.get("state") or "").strip().upper()
    if state:
        return state
    if doc.get("order_error"):
        return "ERROR"
    if doc.get("close_time"):
        return "CLOSED"
    if doc.get("trade_id") and not doc.get("order_error"):
        return "OPEN"
    return "PLACED"


def _prepare_row(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ticket = str(doc.get("trade_id") or doc.get("ticket_id") or "").strip()
    if not ticket or ticket.upper() == "FAIL":
        return None

    entry_time = _normalize_time(doc.get("entry_time") or doc.get("ts"))
    close_time = _normalize_time(doc.get("close_time"))
    close_reason = doc.get("close_reason")
    if close_reason is not None:
        close_reason = str(close_reason)

    row = {
        "ticket_id": ticket,
        "pocket": doc.get("pocket"),
        "instrument": doc.get("instrument") or "USD_JPY",
        "units": _normalize_units(doc.get("units")),
        "entry_price": _normalize_number(doc.get("entry_price") or doc.get("price")),
        "close_price": _normalize_number(doc.get("close_price")),
        "pl_pips": _normalize_number(doc.get("pl_pips")),
        "entry_time": entry_time,
        "close_time": close_time,
        "strategy": doc.get("strategy"),
        "macro_regime": doc.get("macro_regime"),
        "micro_regime": doc.get("micro_regime"),
        "close_reason": close_reason,
        "realized_pl": _normalize_number(doc.get("realized_pl")),
        "unrealized_pl": _normalize_number(doc.get("unrealized_pl")),
        "state": _resolve_state(doc),
        "updated_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec="seconds"),
        "version": doc.get("version") or doc.get("trade_version"),
    }
    return row


def log_trade_snapshot(doc: Dict[str, Any]) -> None:
    """Persist a trade snapshot into the local SQLite ledger."""

    row = _prepare_row(doc)
    if not row:
        return

    conn = _ensure_connection()
    if conn is None:
        return

    sql = (
        "INSERT INTO trades("
        "ticket_id,pocket,instrument,units,entry_price,close_price,pl_pips,"
        "entry_time,close_time,strategy,macro_regime,micro_regime,close_reason,"
        "realized_pl,unrealized_pl,state,updated_at,version"
        ") VALUES (:ticket_id,:pocket,:instrument,:units,:entry_price,:close_price,:pl_pips,"
        ":entry_time,:close_time,:strategy,:macro_regime,:micro_regime,:close_reason,"
        ":realized_pl,:unrealized_pl,:state,:updated_at,:version)"
        " ON CONFLICT(ticket_id) DO UPDATE SET "
        "pocket=excluded.pocket,"
        "instrument=excluded.instrument,"
        "units=excluded.units,"
        "entry_price=excluded.entry_price,"
        "close_price=excluded.close_price,"
        "pl_pips=excluded.pl_pips,"
        "entry_time=excluded.entry_time,"
        "close_time=excluded.close_time,"
        "strategy=excluded.strategy,"
        "macro_regime=excluded.macro_regime,"
        "micro_regime=excluded.micro_regime,"
        "close_reason=excluded.close_reason,"
        "realized_pl=excluded.realized_pl,"
        "unrealized_pl=excluded.unrealized_pl,"
        "state=excluded.state,"
        "updated_at=excluded.updated_at,"
        "version=excluded.version"
    )
    try:
        with _LOCK:
            conn.execute(sql, row)
            conn.commit()
    except Exception:
        # Avoid crashing trading flows; sync daemon will reconcile.
        pass


def _reset_connection_for_tests() -> None:
    """Testing hook to clear cached connection."""

    global _CONN
    with _LOCK:
        if _CONN is not None:
            try:
                _CONN.close()
            except Exception:
                pass
        _CONN = None

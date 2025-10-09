"""
utils.exit_events
~~~~~~~~~~~~~~~~~~
Utility helpers to persist GPT exit advisor events for later analysis.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import sqlite3
import threading
from typing import Any, Dict, Optional

_DB_PATH = pathlib.Path("logs/trades.db")
_DB_PATH.parent.mkdir(exist_ok=True)
_LOCK = threading.Lock()


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(_DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS exit_advice_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          recorded_at TEXT NOT NULL,
          trade_id TEXT,
          strategy TEXT,
          pocket TEXT,
          version TEXT,
          event_type TEXT,
          action TEXT,
          confidence REAL,
          target_tp_pips REAL,
          target_sl_pips REAL,
          note TEXT,
          price REAL,
          move_pips REAL,
          payload_json TEXT
        )
        """
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_exit_events_trade ON exit_advice_events(trade_id)"
    )
    return con


def log_exit_event(
    *,
    trade_id: Optional[str],
    strategy: Optional[str],
    pocket: Optional[str],
    version: Optional[str],
    event_type: str,
    action: Optional[str] = None,
    confidence: Optional[float] = None,
    target_tp_pips: Optional[float] = None,
    target_sl_pips: Optional[float] = None,
    note: Optional[str] = None,
    price: Optional[float] = None,
    move_pips: Optional[float] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a GPT exit advisor event in logs/trades.db."""

    recorded_at = _dt.datetime.utcnow().isoformat(timespec="seconds")
    payload_json = json.dumps(payload, ensure_ascii=False) if payload else None

    with _LOCK:
        con = _connect()
        try:
            con.execute(
                """
                INSERT INTO exit_advice_events(
                  recorded_at, trade_id, strategy, pocket, version,
                  event_type, action, confidence, target_tp_pips, target_sl_pips,
                  note, price, move_pips, payload_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    recorded_at,
                    trade_id,
                    strategy,
                    pocket,
                    version,
                    event_type,
                    action,
                    confidence,
                    target_tp_pips,
                    target_sl_pips,
                    note,
                    price,
                    move_pips,
                    payload_json,
                ),
            )
            con.commit()
        finally:
            con.close()

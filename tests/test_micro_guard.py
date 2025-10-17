import datetime
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution import micro_guard


def _setup_connection(tmp_path):
    conn = sqlite3.connect(tmp_path / "micro_guard.db")
    conn.execute(
        """
        CREATE TABLE trades (
            pocket TEXT,
            state TEXT,
            pl_pips REAL,
            close_time TEXT
        )
        """
    )
    return conn


def _reset_cache():
    micro_guard._CACHE["recent"].update({"ts": 0.0, "value": False})
    micro_guard._CACHE["cooldown"].update({"ts": 0.0, "value": False})


@pytest.fixture(autouse=True)
def restore_guard(tmp_path):
    conn = _setup_connection(tmp_path)
    original_conn = micro_guard._CON
    micro_guard._CON = conn
    _reset_cache()
    try:
        yield conn
    finally:
        micro_guard._CON = original_conn
        _reset_cache()


def test_cooldown_triggers_for_large_loss(restore_guard):
    now = datetime.datetime.now(datetime.timezone.utc)
    restore_guard.execute(
        "INSERT INTO trades VALUES (?,?,?,?)",
        ("micro", "CLOSED", -9.5, now.isoformat()),
    )
    restore_guard.commit()

    assert micro_guard.micro_loss_cooldown_active() is True
    # cache makes the second call inexpensive but still True
    assert micro_guard.micro_loss_cooldown_active() is True


def test_recent_loss_guard_accumulates_drawdown(restore_guard):
    base = datetime.datetime.now(datetime.timezone.utc)
    points = (-3.5, -4.0, -5.5)
    for idx, pl in enumerate(points):
        ts = (base - datetime.timedelta(minutes=idx * 5)).isoformat()
        restore_guard.execute(
            "INSERT INTO trades VALUES (?,?,?,?)",
            ("micro", "CLOSED", pl, ts),
        )
    restore_guard.commit()

    assert micro_guard.micro_recent_loss_guard() is True

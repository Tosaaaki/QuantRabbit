import datetime
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

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


class _FakeTradeRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def _rows(self):
        cursor = self._conn.execute("SELECT pocket, state, pl_pips, close_time FROM trades")
        entries = []
        for pocket, state, pl_pips, close_time in cursor.fetchall():
            ts = datetime.datetime.fromisoformat(close_time) if close_time else None
            entries.append(
                SimpleNamespace(
                    pocket=pocket,
                    state=state,
                    pl_pips=pl_pips,
                    close_time=ts,
                )
            )
        entries.sort(
            key=lambda item: item.close_time.timestamp() if item.close_time else float("-inf"),
            reverse=True,
        )
        return entries

    def recent_trades(self, pocket, *, limit=200, closed_only=True, since=None):
        since_ts = since.timestamp() if since else None
        rows = []
        for item in self._rows():
            if pocket and item.pocket != pocket:
                continue
            if closed_only and (item.state or "").upper() != "CLOSED":
                continue
            if since_ts is not None and item.close_time and item.close_time.timestamp() < since_ts:
                continue
            rows.append(item)
            if len(rows) >= limit:
                break
        return rows

    def last_closed_trade(self, pocket):
        result = self.recent_trades(pocket, limit=1, closed_only=True)
        return result[0] if result else None


@pytest.fixture(autouse=True)
def restore_guard(tmp_path):
    conn = _setup_connection(tmp_path)
    repo = _FakeTradeRepo(conn)
    original_repo = micro_guard._TRADE_REPO
    micro_guard._TRADE_REPO = repo
    _reset_cache()
    try:
        yield conn
    finally:
        micro_guard._TRADE_REPO = original_repo
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

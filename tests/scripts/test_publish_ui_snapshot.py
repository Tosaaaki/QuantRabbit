from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from scripts import publish_ui_snapshot


def _create_trades_db(path, rows: list[tuple[str, str, float, float]]) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            CREATE TABLE trades (
                close_time TEXT,
                pocket TEXT,
                pl_pips REAL,
                realized_pl REAL
            )
            """
        )
        con.executemany(
            "INSERT INTO trades(close_time, pocket, pl_pips, realized_pl) VALUES (?, ?, ?, ?)",
            rows,
        )
        con.commit()
    finally:
        con.close()


def test_build_hourly_trades_from_db_populates_full_lookback(tmp_path, monkeypatch) -> None:
    now = datetime(2026, 2, 27, 6, 30, tzinfo=timezone.utc)
    db_path = tmp_path / "trades.db"
    rows = [
        ((now - timedelta(hours=1)).isoformat(), "scalp_fast", 1.2, 120.0),
        ((now - timedelta(hours=4)).isoformat(), "micro", -0.5, -50.0),
        ((now - timedelta(hours=3)).isoformat(), "manual", 9.0, 900.0),
        ((now - timedelta(hours=9)).isoformat(), "scalp_fast", 4.0, 400.0),
    ]
    _create_trades_db(db_path, rows)

    monkeypatch.setattr(publish_ui_snapshot, "TRADES_DB", db_path)
    monkeypatch.setattr(publish_ui_snapshot, "HOURLY_LOOKBACK_HOURS", 6)

    payload = publish_ui_snapshot._build_hourly_trades_from_db(now)

    assert payload is not None
    assert payload["timezone"] == "JST"
    assert payload["lookback_hours"] == 6
    assert len(payload["hours"]) == 6
    assert sum(int(row["trades"]) for row in payload["hours"]) == 2
    assert round(sum(float(row["pips"]) for row in payload["hours"]), 2) == 0.7


def test_build_hourly_trades_from_db_returns_none_when_db_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(publish_ui_snapshot, "TRADES_DB", tmp_path / "missing.db")

    payload = publish_ui_snapshot._build_hourly_trades_from_db(
        datetime(2026, 2, 27, 6, 30, tzinfo=timezone.utc)
    )

    assert payload is None


def test_build_hourly_trades_from_recent_trades_fallback_works_without_db(monkeypatch) -> None:
    now = datetime(2026, 2, 27, 6, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(publish_ui_snapshot, "HOURLY_LOOKBACK_HOURS", 6)
    payload = publish_ui_snapshot._build_hourly_trades_from_recent_trades(
        [
            {
                "close_time": (now - timedelta(hours=1)).isoformat(),
                "pocket": "scalp_fast",
                "pl_pips": 1.2,
                "realized_pl": 120.0,
            },
            {
                "close_time": (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "pocket": "micro",
                "pl_pips": -0.4,
                "realized_pl": -40.0,
            },
            {
                "close_time": (now - timedelta(hours=3)).isoformat(),
                "pocket": "manual",
                "pl_pips": 9.0,
                "realized_pl": 900.0,
            },
        ],
        now=now,
    )

    assert payload["lookback_hours"] == 6
    assert len(payload["hours"]) == 6
    assert sum(int(row["trades"]) for row in payload["hours"]) == 2

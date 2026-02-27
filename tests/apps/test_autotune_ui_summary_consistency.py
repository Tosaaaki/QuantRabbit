from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import apps.autotune_ui as ui


def _strategy_control_stub() -> dict:
    return {
        "global": {"entry_enabled": True, "exit_enabled": True, "global_lock": False},
        "strategies": [],
        "error": None,
        "discovered_count": 0,
    }


def _trade_row(*, ticket: str, hours_ago: float, pl_pips: float) -> dict:
    close_dt = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    stamp = close_dt.isoformat()
    return {
        "ticket_id": ticket,
        "pocket": "scalp",
        "units": 100.0,
        "pl_pips": pl_pips,
        "realized_pl": pl_pips * 100.0,
        "entry_time": stamp,
        "close_time": stamp,
        "updated_at": stamp,
        "state": "closed",
        "close_reason": "tp_hit" if pl_pips > 0 else "sl_hit",
        "strategy_tag": "scalp_ping_5s_b",
        "strategy": "scalp_ping_5s_b",
    }


def _hourly_rows(lookback: int, *, reference_now: datetime, trades: int = 1) -> list[dict]:
    anchor = reference_now.astimezone(ui._JST).replace(minute=0, second=0, microsecond=0)
    rows: list[dict] = []
    for i in range(lookback):
        hour = anchor - timedelta(hours=i)
        rows.append(
            {
                "key": hour.isoformat(),
                "label": hour.strftime("%m/%d %H:%M"),
                "pips": 0.0,
                "jpy": 0.0,
                "trades": trades,
                "wins": trades,
                "losses": 0,
                "win_rate": 1.0 if trades else 0.0,
            }
        )
    return rows


def test_summarise_snapshot_prefers_db_rollup_for_summary_cards(monkeypatch):
    monkeypatch.setattr(ui, "_load_strategy_control_state", _strategy_control_stub)
    monkeypatch.setattr(
        ui,
        "_load_trade_rollup_jst",
        lambda _now: {
            "daily_pips": -35.3,
            "daily_jpy": -297.0,
            "daily_trades": 50,
            "yesterday_pips": 0.0,
            "yesterday_jpy": 0.0,
            "yesterday_trades": 0,
            "weekly_pips": -40.0,
            "weekly_jpy": -320.0,
            "weekly_trades": 50,
            "wins_7d": 20,
            "losses_7d": 30,
        },
    )

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recent_trades": [
            _trade_row(ticket="1", hours_ago=0.5, pl_pips=-10.0),
            _trade_row(ticket="2", hours_ago=1.5, pl_pips=8.0),
        ],
        "open_positions": {},
        "metrics": {
            "daily": {"pips": 0.0, "jpy": 0.0, "trades": 0},
            "weekly": {"pips": 0.0, "jpy": 0.0, "trades": 0},
            "total": {"pips": 0.0, "jpy": 0.0, "wins": 0, "losses": 0, "win_rate": 0.0, "trades": 0},
        },
    }

    result = ui._summarise_snapshot(snapshot)
    perf = result["performance"]

    assert perf["daily_pl_pips"] == -35.3
    assert perf["daily_pl_jpy"] == -297.0
    assert perf["weekly_pl_pips"] == -40.0
    assert perf["recent_closed"] == 50
    assert perf["wins"] == 20
    assert perf["losses"] == 30
    assert perf["win_rate_percent"] == 40.0


def test_summarise_snapshot_repairs_zero_win_loss_when_rollup_unavailable(monkeypatch):
    monkeypatch.setattr(ui, "_load_strategy_control_state", _strategy_control_stub)
    monkeypatch.setattr(ui, "_load_trade_rollup_jst", lambda _now: None)

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recent_trades": [
            _trade_row(ticket="1", hours_ago=0.5, pl_pips=-4.0),
            _trade_row(ticket="2", hours_ago=1.0, pl_pips=6.0),
            _trade_row(ticket="3", hours_ago=2.0, pl_pips=0.0),
        ],
        "open_positions": {},
        "metrics": {
            "daily": {"pips": 0.0, "jpy": 0.0, "trades": 3},
            "total": {"pips": 0.0, "jpy": 0.0, "wins": 0, "losses": 0, "win_rate": 0.0, "trades": 3},
        },
    }

    result = ui._summarise_snapshot(snapshot)
    perf = result["performance"]

    assert perf["recent_closed"] >= 2
    assert perf["wins"] == 1
    assert perf["losses"] == 1
    assert perf["win_rate_percent"] == 50.0


def test_summarise_snapshot_repairs_daily_zero_metrics_when_recent_trades_exist(monkeypatch):
    monkeypatch.setattr(ui, "_load_strategy_control_state", _strategy_control_stub)
    monkeypatch.setattr(ui, "_load_trade_rollup_jst", lambda _now: None)

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recent_trades": [
            _trade_row(ticket="1", hours_ago=0.5, pl_pips=-10.0),
            _trade_row(ticket="2", hours_ago=1.5, pl_pips=8.0),
        ],
        "open_positions": {},
        "metrics": {
            "daily": {"pips": 0.0, "jpy": 0.0, "trades": 2},
            "weekly": {"pips": 0.0, "jpy": 0.0, "trades": 2},
            "total": {"pips": 0.0, "jpy": 0.0, "wins": 0, "losses": 0, "win_rate": 0.0, "trades": 2},
            "daily_change": {"pips": 0.0, "jpy": 0.0, "jpy_pct": 0.0, "equity_nav": 50000.0},
        },
    }

    result = ui._summarise_snapshot(snapshot)
    perf = result["performance"]

    assert perf["daily_pl_pips"] == -2.0
    assert perf["daily_pl_jpy"] == -200.0
    assert perf["weekly_pl_pips"] == -2.0
    assert perf["wins"] == 1
    assert perf["losses"] == 1
    assert perf["win_rate_percent"] == 50.0
    assert perf["daily_change_pips"] == -2.0
    assert perf["daily_change_jpy"] == -200.0


def test_summarise_snapshot_reconciles_nonzero_metrics_from_recent_trades_without_db(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(ui, "_load_strategy_control_state", _strategy_control_stub)
    monkeypatch.setattr(ui, "_load_trade_rollup_jst", lambda _now: None)
    monkeypatch.setattr(ui, "TRADES_DB", tmp_path / "missing-trades.db")

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recent_trades": [
            _trade_row(ticket="1", hours_ago=0.5, pl_pips=-3.0),
            _trade_row(ticket="2", hours_ago=1.0, pl_pips=1.0),
        ],
        "open_positions": {},
        "metrics": {
            "daily": {"pips": 5.0, "jpy": 500.0, "trades": 5},
            "yesterday": {"pips": 2.0, "jpy": 200.0, "trades": 2},
            "weekly": {"pips": 5.0, "jpy": 500.0, "trades": 5},
            "total": {"pips": 5.0, "jpy": 500.0, "wins": 5, "losses": 0, "win_rate": 1.0, "trades": 5},
            "daily_change": {"pips": 3.0, "jpy": 300.0, "jpy_pct": 1.0, "equity_nav": 50000.0},
        },
    }

    result = ui._summarise_snapshot(snapshot)
    perf = result["performance"]

    assert perf["daily_pl_pips"] == -2.0
    assert perf["daily_pl_jpy"] == -200.0
    assert perf["yesterday_pl_pips"] == 0.0
    assert perf["yesterday_pl_jpy"] == 0.0
    assert perf["weekly_pl_pips"] == -2.0
    assert perf["weekly_pl_jpy"] == -200.0
    assert perf["recent_closed"] == 2
    assert perf["wins"] == 1
    assert perf["losses"] == 1
    assert perf["win_rate_percent"] == 50.0
    assert perf["daily_change_pips"] == -2.0
    assert perf["daily_change_jpy"] == -200.0


def test_summarise_snapshot_keeps_snapshot_metrics_when_hourly_snapshot_is_usable(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(ui, "_load_strategy_control_state", _strategy_control_stub)
    monkeypatch.setattr(ui, "_load_trade_rollup_jst", lambda _now: None)
    monkeypatch.setattr(ui, "TRADES_DB", tmp_path / "missing-trades.db")

    now = datetime.now(timezone.utc)
    snapshot = {
        "generated_at": now.isoformat(),
        "recent_trades": [
            _trade_row(ticket="1", hours_ago=0.5, pl_pips=-3.0),
            _trade_row(ticket="2", hours_ago=1.0, pl_pips=1.0),
        ],
        "open_positions": {},
        "metrics": {
            "daily": {"pips": 5.0, "jpy": 500.0, "trades": 5},
            "yesterday": {"pips": 2.0, "jpy": 200.0, "trades": 2},
            "weekly": {"pips": 5.0, "jpy": 500.0, "trades": 5},
            "total": {"pips": 5.0, "jpy": 500.0, "wins": 5, "losses": 0, "win_rate": 1.0, "trades": 5},
            "daily_change": {"pips": 3.0, "jpy": 300.0, "jpy_pct": 1.0, "equity_nav": 50000.0},
            "hourly_trades": {
                "timezone": "JST",
                "lookback_hours": 24,
                "exclude_manual": True,
                "hours": _hourly_rows(24, reference_now=now, trades=1),
            },
        },
    }

    result = ui._summarise_snapshot(snapshot)
    perf = result["performance"]

    assert perf["daily_pl_pips"] == 5.0
    assert perf["daily_pl_jpy"] == 500.0
    assert perf["weekly_pl_pips"] == 5.0
    assert perf["weekly_pl_jpy"] == 500.0
    assert perf["wins"] == 5
    assert perf["losses"] == 0
    assert perf["win_rate_percent"] == 100.0


def test_summarise_snapshot_repairs_summary_when_hourly_is_usable_but_rollups_missing(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(ui, "_load_strategy_control_state", _strategy_control_stub)
    monkeypatch.setattr(ui, "_load_trade_rollup_jst", lambda _now: None)
    monkeypatch.setattr(ui, "TRADES_DB", tmp_path / "missing-trades.db")

    now = datetime.now(timezone.utc)
    snapshot = {
        "generated_at": now.isoformat(),
        "recent_trades": [
            _trade_row(ticket="1", hours_ago=0.5, pl_pips=-3.0),
            _trade_row(ticket="2", hours_ago=1.0, pl_pips=1.0),
        ],
        "open_positions": {},
        "metrics": {
            "hourly_trades": {
                "timezone": "JST",
                "lookback_hours": 24,
                "exclude_manual": True,
                "hours": _hourly_rows(24, reference_now=now, trades=1),
            },
        },
    }

    result = ui._summarise_snapshot(snapshot)
    perf = result["performance"]

    assert perf["daily_pl_pips"] == -2.0
    assert perf["daily_pl_jpy"] == -200.0
    assert perf["weekly_pl_pips"] == -2.0
    assert perf["weekly_pl_jpy"] == -200.0


def test_load_trade_rollup_jst_aggregates_windows(tmp_path, monkeypatch):
    db_path = tmp_path / "trades.db"
    con = sqlite3.connect(db_path)
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
        now = datetime(2026, 2, 27, 4, 0, tzinfo=timezone.utc)
        rows = [
            # today(JST): 2 trades -> pips +1.0
            ((now - timedelta(hours=1)).isoformat(), "scalp", 2.0, 200.0),
            ((now - timedelta(hours=2)).isoformat(), "micro", -1.0, -100.0),
            # yesterday(JST): 1 trade
            ((now - timedelta(hours=20)).isoformat(), "scalp", 3.0, 300.0),
            # within 7d but before yesterday
            ((now - timedelta(days=3)).isoformat(), "scalp", -2.0, -200.0),
            # manual pocket should be excluded
            ((now - timedelta(hours=1)).isoformat(), "manual", 9.0, 900.0),
            # older than 7d
            ((now - timedelta(days=10)).isoformat(), "scalp", 5.0, 500.0),
        ]
        con.executemany(
            "INSERT INTO trades(close_time, pocket, pl_pips, realized_pl) VALUES (?, ?, ?, ?)",
            rows,
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(ui, "TRADES_DB", db_path)
    rollup = ui._load_trade_rollup_jst(now)

    assert rollup is not None
    assert rollup["daily_pips"] == 1.0
    assert rollup["daily_jpy"] == 100.0
    assert rollup["daily_trades"] == 2
    assert rollup["yesterday_pips"] == 3.0
    assert rollup["yesterday_jpy"] == 300.0
    assert rollup["weekly_pips"] == 2.0
    assert rollup["weekly_jpy"] == 200.0
    assert rollup["weekly_trades"] == 4
    assert rollup["wins_7d"] == 2
    assert rollup["losses_7d"] == 2

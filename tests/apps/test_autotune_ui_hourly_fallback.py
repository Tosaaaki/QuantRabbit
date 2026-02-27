from __future__ import annotations

from datetime import datetime, timedelta, timezone

import apps.autotune_ui as ui


def _trade(hours_ago: float, *, pocket: str = "scalp", pl_pips: float = 1.0) -> dict:
    close_time = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return {
        "close_time": close_time,
        "pocket": pocket,
        "pl_pips": pl_pips,
        "realized_pl": pl_pips * 100.0,
    }


def _total_trades(payload: dict) -> int:
    return sum(int(row.get("trades", 0)) for row in payload.get("hours", []))


def _hourly_rows_for_reference(*, lookback: int, reference_now: datetime) -> list[dict]:
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
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
            }
        )
    return rows


def test_build_hourly_fallback_prefers_db_window_over_limited_snapshot(monkeypatch):
    monkeypatch.setattr(ui, "_HOURLY_TRADES_LOOKBACK", 6)
    monkeypatch.setattr(ui, "_load_hourly_fallback_aggregates", lambda _start: None)
    monkeypatch.setattr(
        ui,
        "_load_hourly_fallback_trades",
        lambda: [_trade(4.5), _trade(1.0)],
    )

    result = ui._build_hourly_fallback([_trade(1.0)])

    assert _total_trades(result) == 2


def test_build_hourly_fallback_uses_snapshot_trades_when_db_unavailable(monkeypatch):
    monkeypatch.setattr(ui, "_HOURLY_TRADES_LOOKBACK", 6)
    monkeypatch.setattr(ui, "_load_hourly_fallback_aggregates", lambda _start: None)
    monkeypatch.setattr(ui, "_load_hourly_fallback_trades", lambda: [])

    result = ui._build_hourly_fallback(
        [
            _trade(1.0, pl_pips=2.0),
            _trade(2.0, pocket="manual", pl_pips=5.0),
            _trade(5.0, pl_pips=-1.0),
            _trade(8.0, pl_pips=3.0),
        ]
    )

    assert _total_trades(result) == 2


def test_build_hourly_fallback_uses_aggregate_query_when_available(monkeypatch):
    monkeypatch.setattr(ui, "_HOURLY_TRADES_LOOKBACK", 6)
    monkeypatch.setattr(
        ui,
        "_load_hourly_fallback_aggregates",
        lambda _start: {
            "2099-01-01 01:00:00": {
                "pips": 3.0,
                "jpy": 300.0,
                "trades": 2,
                "wins": 2,
                "losses": 0,
            }
        },
    )
    monkeypatch.setattr(ui, "_load_hourly_fallback_trades", lambda: [_trade(1.0)])

    result = ui._build_hourly_fallback([])

    assert _total_trades(result) == 0


def test_build_hourly_fallback_uses_snapshot_trades_when_aggregate_is_empty(monkeypatch):
    monkeypatch.setattr(ui, "_HOURLY_TRADES_LOOKBACK", 6)
    monkeypatch.setattr(ui, "_load_hourly_fallback_aggregates", lambda _start: {})
    monkeypatch.setattr(ui, "_load_hourly_fallback_trades", lambda: [])

    result = ui._build_hourly_fallback(
        [
            _trade(1.0, pl_pips=1.2),
            _trade(2.0, pl_pips=-0.7),
        ]
    )

    assert _total_trades(result) == 2


def test_hourly_trades_usable_requires_full_lookback(monkeypatch):
    monkeypatch.setattr(ui, "_HOURLY_TRADES_LOOKBACK", 6)
    reference_now = datetime(2026, 2, 27, 3, 30, tzinfo=timezone.utc)
    full_rows = _hourly_rows_for_reference(lookback=6, reference_now=reference_now)

    assert (
        ui._hourly_trades_is_usable(
            {"timezone": "JST", "lookback_hours": 6, "hours": full_rows},
            reference_now=reference_now,
        )
        is True
    )
    assert (
        ui._hourly_trades_is_usable(
            {"timezone": "JST", "lookback_hours": 6, "hours": full_rows[1:]},
            reference_now=reference_now,
        )
        is False
    )
    assert (
        ui._hourly_trades_is_usable(
            {"timezone": "UTC", "lookback_hours": 6, "hours": full_rows},
            reference_now=reference_now,
        )
        is False
    )

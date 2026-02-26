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


def test_build_hourly_fallback_prefers_db_window_over_limited_snapshot(monkeypatch):
    monkeypatch.setattr(ui, "_HOURLY_TRADES_LOOKBACK", 6)
    monkeypatch.setattr(
        ui,
        "_load_hourly_fallback_trades",
        lambda: [_trade(4.5), _trade(1.0)],
    )

    result = ui._build_hourly_fallback([_trade(1.0)])

    assert _total_trades(result) == 2


def test_build_hourly_fallback_uses_snapshot_trades_when_db_unavailable(monkeypatch):
    monkeypatch.setattr(ui, "_HOURLY_TRADES_LOOKBACK", 6)
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

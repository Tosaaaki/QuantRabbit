from __future__ import annotations

from datetime import datetime, timezone

import apps.autotune_ui as ui


def _trade_row() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "ticket_id": "1",
        "pocket": "scalp",
        "units": 100.0,
        "pl_pips": 1.0,
        "realized_pl": 100.0,
        "entry_time": now,
        "close_time": now,
        "updated_at": now,
        "state": "closed",
        "close_reason": "tp_hit",
        "strategy_tag": "scalp_ping_5s_b",
        "strategy": "scalp_ping_5s_b",
    }


def _strategy_control_stub() -> dict:
    return {
        "global": {"entry_enabled": True, "exit_enabled": True, "global_lock": False},
        "strategies": [],
        "error": None,
        "discovered_count": 0,
    }


def test_summarise_snapshot_falls_back_when_hourly_trades_is_incomplete(monkeypatch):
    monkeypatch.setattr(ui, "_load_strategy_control_state", _strategy_control_stub)
    monkeypatch.setattr(
        ui,
        "_build_hourly_fallback",
        lambda _trades: {
            "timezone": "JST",
            "lookback_hours": 24,
            "exclude_manual": True,
            "hours": [{"label": "fallback", "trades": 9}],
        },
    )

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recent_trades": [_trade_row()],
        "open_positions": {},
        "metrics": {
            "hourly_trades": {
                "timezone": "JST",
                "lookback_hours": 12,
                "exclude_manual": True,
                "hours": [{"label": "partial", "trades": 1}] * 12,
            }
        },
    }

    result = ui._summarise_snapshot(snapshot)

    assert result["hourly_trades"]["hours"][0]["label"] == "fallback"


def test_summarise_snapshot_keeps_hourly_trades_when_complete(monkeypatch):
    monkeypatch.setattr(ui, "_load_strategy_control_state", _strategy_control_stub)
    monkeypatch.setattr(
        ui,
        "_build_hourly_fallback",
        lambda _trades: {
            "timezone": "JST",
            "lookback_hours": 24,
            "exclude_manual": True,
            "hours": [{"label": "fallback", "trades": 9}],
        },
    )

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recent_trades": [_trade_row()],
        "open_positions": {},
        "metrics": {
            "hourly_trades": {
                "timezone": "JST",
                "lookback_hours": 24,
                "exclude_manual": True,
                "hours": [{"label": "snapshot", "trades": 1}] * 24,
            }
        },
    }

    result = ui._summarise_snapshot(snapshot)

    assert result["hourly_trades"]["hours"][0]["label"] == "snapshot"


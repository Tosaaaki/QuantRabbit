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


def test_load_dashboard_data_uses_local_fallback_when_snapshots_unavailable(monkeypatch):
    monkeypatch.setattr(
        ui,
        "_collect_snapshot_candidates",
        lambda: (
            [],
            [
                {"source": "remote(ui_snapshot_lite_url)", "status": "skip", "error": "timeout"},
                {"source": "gcs", "status": "skip", "error": "missing"},
            ],
        ),
    )
    monkeypatch.setattr(ui, "_pick_snapshot_by_preference", lambda _candidates: None)
    monkeypatch.setattr(ui, "_load_recent_trades", lambda limit=50: [_trade_row()])
    monkeypatch.setattr(
        ui,
        "_build_hourly_fallback",
        lambda _trades: {
            "timezone": "JST",
            "lookback_hours": 24,
            "exclude_manual": True,
            "hours": [{"label": "fallback-hour", "trades": 2}],
        },
    )
    monkeypatch.setattr(ui, "_load_strategy_control_state", _strategy_control_stub)

    result = ui._load_dashboard_data()

    assert result["available"] is True
    assert result["snapshot"]["source"] == "local-fallback"
    assert result["snapshot"]["mode"] == "local-fallback"
    assert "timeout" in (result["snapshot"]["fetch_error"] or "")
    assert result["hourly_trades"]["hours"][0]["label"] == "fallback-hour"
    assert len(result["recent_trades"]) == 1

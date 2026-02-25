from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from workers.scalp_ping_5s_c import exit_worker as c_exit_worker


def test_c_exit_direction_flip_short_side_overrides_apply(monkeypatch):
    worker = c_exit_worker.RangeFaderExitWorker()
    worker.exit_policy_start_ts = 0.0
    worker.new_policy_start_ts = 0.0

    monkeypatch.setattr(
        c_exit_worker,
        "_exit_profile_for_tag",
        lambda *_args, **_kwargs: {
            "min_hold_sec": 1.0,
            "loss_cut_enabled": False,
            "direction_flip": {
                "enabled": True,
                "min_hold_sec": 90.0,
                "min_adverse_pips": 2.0,
                "score_threshold": 0.70,
                "release_threshold": 0.50,
                "confirm_hits": 3,
                "confirm_window_sec": 30.0,
                "cooldown_sec": 0.0,
                "forecast_weight": 0.35,
                "de_risk_enabled": False,
                "short_min_hold_sec": 30.0,
                "short_min_adverse_pips": 0.8,
                "short_score_threshold": 0.55,
                "short_release_threshold": 0.42,
                "short_confirm_hits": 1,
                "short_confirm_window_sec": 20.0,
                "reason": "m1_structure_break",
            },
        },
    )
    monkeypatch.setattr(
        worker,
        "_direction_flip_score",
        lambda **_kwargs: (0.60, {"score": 0.60}),
    )
    monkeypatch.setattr(
        c_exit_worker,
        "all_factors",
        lambda: {
            "M1": {"rsi": 54.0, "adx": 20.0, "atr_pips": 1.2, "ma10": 150.0, "ma20": 150.0},
            "H4": {},
        },
    )

    async def _fake_pro_stop(*_args, **_kwargs):
        return False

    monkeypatch.setattr(c_exit_worker, "maybe_close_pro_stop", _fake_pro_stop)

    closed_reasons: list[str] = []

    async def _fake_close(trade_id, units, reason, pnl, client_order_id, allow_negative=False):
        closed_reasons.append(str(reason))

    monkeypatch.setattr(worker, "_close", _fake_close)

    now = datetime.now(timezone.utc)
    trade = {
        "trade_id": "C-short-override",
        "units": -1400,
        "price": 150.100,
        "unrealized_pl_pips": -1.1,
        "open_time": (now - timedelta(seconds=40)).isoformat(),
        "client_order_id": "cid-c-short-override",
        "entry_thesis": {"strategy_tag": "scalp_ping_5s_c_live"},
    }

    asyncio.run(worker._review_trade(trade, now, mid=150.030, range_active=False))

    assert closed_reasons == ["m1_structure_break"]


def test_c_exit_non_range_max_hold_short_override_applies(monkeypatch):
    worker = c_exit_worker.RangeFaderExitWorker()
    worker.exit_policy_start_ts = 0.0
    worker.new_policy_start_ts = 0.0

    monkeypatch.setattr(
        c_exit_worker,
        "_exit_profile_for_tag",
        lambda *_args, **_kwargs: {
            "min_hold_sec": 1.0,
            "loss_cut_enabled": False,
            "loss_cut_reason_time": "time_stop_side",
            "non_range_max_hold_sec": 900.0,
            "non_range_max_hold_sec_short": 60.0,
            "direction_flip": {"enabled": False},
        },
    )
    monkeypatch.setattr(
        c_exit_worker,
        "all_factors",
        lambda: {"M1": {"rsi": 50.0, "adx": 15.0, "atr_pips": 1.0}, "H4": {}},
    )

    async def _fake_pro_stop(*_args, **_kwargs):
        return False

    monkeypatch.setattr(c_exit_worker, "maybe_close_pro_stop", _fake_pro_stop)

    closed: list[tuple[str, str]] = []

    async def _fake_close(trade_id, units, reason, pnl, client_order_id, allow_negative=False):
        closed.append((str(trade_id), str(reason)))

    monkeypatch.setattr(worker, "_close", _fake_close)

    now = datetime.now(timezone.utc)
    short_trade = {
        "trade_id": "C-short-hold",
        "units": -1200,
        "price": 150.100,
        "unrealized_pl_pips": -0.8,
        "open_time": (now - timedelta(seconds=85)).isoformat(),
        "client_order_id": "cid-c-short-hold",
        "entry_thesis": {"strategy_tag": "scalp_ping_5s_c_live"},
    }
    long_trade = {
        "trade_id": "C-long-hold",
        "units": 1200,
        "price": 150.100,
        "unrealized_pl_pips": -0.8,
        "open_time": (now - timedelta(seconds=85)).isoformat(),
        "client_order_id": "cid-c-long-hold",
        "entry_thesis": {"strategy_tag": "scalp_ping_5s_c_live"},
    }

    asyncio.run(worker._review_trade(short_trade, now, mid=150.030, range_active=False))
    asyncio.run(worker._review_trade(long_trade, now, mid=150.030, range_active=False))

    assert ("C-short-hold", "time_stop_side") in closed
    assert all(trade_id != "C-long-hold" for trade_id, _ in closed)


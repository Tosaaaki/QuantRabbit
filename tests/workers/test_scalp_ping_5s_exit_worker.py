from __future__ import annotations

import time
import asyncio
from datetime import datetime, timedelta, timezone

from workers.scalp_ping_5s import exit_worker


def test_direction_flip_score_high_on_clear_opposite_trend(monkeypatch):
    worker = exit_worker.RangeFaderExitWorker()
    monkeypatch.setattr(
        exit_worker,
        "_forecast_bias_from_factors",
        lambda *_args, **_kwargs: {
            "p_up": 0.22,
            "edge": 0.72,
            "projection_score": -0.66,
        },
    )
    score, diag = worker._direction_flip_score(
        side="long",
        fac_m1={
            "atr_pips": 1.5,
            "rsi": 34.0,
            "adx": 28.0,
            "ma10": 150.00,
            "ma20": 150.08,
            "vwap_gap": -5.2,
            "ema_slope_10": -0.03,
        },
        fac_h4={},
        forecast_weight=0.35,
    )
    assert score >= 0.65
    assert diag["forecast_score"] >= 0.5


def test_direction_flip_score_low_when_trend_and_forecast_support_trade(monkeypatch):
    worker = exit_worker.RangeFaderExitWorker()
    monkeypatch.setattr(
        exit_worker,
        "_forecast_bias_from_factors",
        lambda *_args, **_kwargs: {
            "p_up": 0.82,
            "edge": 0.62,
            "projection_score": 0.44,
        },
    )
    score, diag = worker._direction_flip_score(
        side="long",
        fac_m1={
            "atr_pips": 1.7,
            "rsi": 63.0,
            "adx": 24.0,
            "ma10": 150.08,
            "ma20": 150.00,
            "vwap_gap": 4.8,
            "ema_slope_10": 0.04,
        },
        fac_h4={},
        forecast_weight=0.35,
    )
    assert score <= 0.3
    assert diag["forecast_score"] <= 0.2


def test_direction_flip_requires_hysteresis_hits(monkeypatch):
    worker = exit_worker.RangeFaderExitWorker()
    monkeypatch.setattr(
        worker,
        "_direction_flip_score",
        lambda **_kwargs: (0.9, {"score": 0.9}),
    )
    reason = None
    for _ in range(3):
        reason, _diag = worker._maybe_direction_flip_reason(
            trade_id="T1",
            side="short",
            hold_sec=180.0,
            adverse_pips=3.0,
            reason="m1_structure_break",
            enabled=True,
            min_hold_sec=60.0,
            min_adverse_pips=1.0,
            score_threshold=0.64,
            score_release=0.46,
            confirm_hits=3,
            confirm_window_sec=30.0,
            cooldown_sec=0.0,
            forecast_weight=0.35,
            fac_m1={},
            fac_h4={},
        )
    assert reason == "m1_structure_break"


def test_direction_flip_derisk_then_exit(monkeypatch):
    worker = exit_worker.RangeFaderExitWorker()
    scores = iter([0.58, 0.92])
    monkeypatch.setattr(
        worker,
        "_direction_flip_score",
        lambda **_kwargs: (next(scores), {"score": 0.0}),
    )

    reason, _diag = worker._maybe_direction_flip_reason(
        trade_id="T2",
        side="short",
        hold_sec=180.0,
        adverse_pips=3.0,
        reason="m1_structure_break",
        enabled=True,
        min_hold_sec=60.0,
        min_adverse_pips=1.0,
        score_threshold=0.64,
        score_release=0.46,
        confirm_hits=1,
        confirm_window_sec=30.0,
        cooldown_sec=0.0,
        forecast_weight=0.35,
        fac_m1={},
        fac_h4={},
        de_risk_enabled=True,
        de_risk_threshold=0.54,
        de_risk_cooldown_sec=0.0,
    )
    assert reason == "__de_risk__"

    st = worker._direction_flip_states.get("T2")
    assert st is not None
    st.de_risked = True

    reason, _diag = worker._maybe_direction_flip_reason(
        trade_id="T2",
        side="short",
        hold_sec=180.0,
        adverse_pips=3.0,
        reason="m1_structure_break",
        enabled=True,
        min_hold_sec=60.0,
        min_adverse_pips=1.0,
        score_threshold=0.64,
        score_release=0.46,
        confirm_hits=1,
        confirm_window_sec=30.0,
        cooldown_sec=0.0,
        forecast_weight=0.35,
        fac_m1={},
        fac_h4={},
        de_risk_enabled=True,
        de_risk_threshold=0.54,
        de_risk_cooldown_sec=0.0,
    )
    assert reason == "m1_structure_break"


def test_new_policy_gate_skips_existing_positions(monkeypatch):
    worker = exit_worker.RangeFaderExitWorker()
    worker.exit_policy_start_ts = 0.0
    worker.new_policy_start_ts = time.time() + 3600.0

    monkeypatch.setattr(
        exit_worker,
        "_exit_profile_for_tag",
        lambda *_args, **_kwargs: {
            "min_hold_sec": 1.0,
            "profit_pips": 50.0,
            "trail_start_pips": 60.0,
            "loss_cut_enabled": True,
            "loss_cut_require_sl": False,
            "loss_cut_soft_pips": 0.0,
            "loss_cut_hard_pips": 2.0,
            "loss_cut_max_hold_sec": 60.0,
            "non_range_max_hold_sec": 60.0,
            "direction_flip": {
                "enabled": True,
                "min_hold_sec": 10.0,
                "min_adverse_pips": 1.0,
                "score_threshold": 0.6,
                "release_threshold": 0.4,
                "confirm_hits": 1,
                "confirm_window_sec": 30.0,
                "cooldown_sec": 0.0,
                "forecast_weight": 0.35,
                "reason": "m1_structure_break",
            },
        },
    )
    monkeypatch.setattr(
        exit_worker,
        "all_factors",
        lambda: {
            "M1": {
                "rsi": 28.0,
                "adx": 30.0,
                "atr_pips": 1.4,
                "ma10": 150.00,
                "ma20": 150.08,
                "vwap_gap": -4.5,
                "ema_slope_10": -0.03,
            },
            "H4": {"adx": 24.0, "ma10": 150.0, "ma20": 150.1},
        },
    )
    monkeypatch.setattr(
        exit_worker,
        "_forecast_bias_from_factors",
        lambda *_args, **_kwargs: {
            "p_up": 0.2,
            "edge": 0.8,
            "projection_score": -0.7,
        },
    )

    async def _fake_pro_stop(*_args, **_kwargs):
        return False

    monkeypatch.setattr(exit_worker, "maybe_close_pro_stop", _fake_pro_stop)

    closed: list[str] = []

    async def _fake_close(*_args, **_kwargs):
        closed.append("x")

    monkeypatch.setattr(worker, "_close", _fake_close)

    now = datetime.now(timezone.utc)
    trade = {
        "trade_id": "T-new-gate",
        "units": 1000,
        "price": 150.100,
        "unrealized_pl_pips": -3.0,
        "open_time": (now - timedelta(minutes=15)).isoformat(),
        "client_order_id": "cid-1",
        "entry_thesis": {"strategy_tag": "scalp_ping_5s_live"},
    }

    asyncio.run(worker._review_trade(trade, now, mid=150.030, range_active=False))

    assert closed == []


def test_direction_flip_derisk_sentinel_falls_back_to_direction_reason(monkeypatch):
    worker = exit_worker.RangeFaderExitWorker()
    worker.exit_policy_start_ts = 0.0
    worker.new_policy_start_ts = 0.0

    monkeypatch.setattr(
        exit_worker,
        "_exit_profile_for_tag",
        lambda *_args, **_kwargs: {
            "min_hold_sec": 1.0,
            "loss_cut_enabled": False,
            "direction_flip": {
                "enabled": True,
                "min_hold_sec": 10.0,
                "min_adverse_pips": 1.0,
                "de_risk_enabled": True,
                "de_risk_threshold": 0.54,
                "de_risk_fraction": 0.40,
                "de_risk_min_units": 1000,
                "de_risk_min_remaining": 1000,
                "de_risk_cooldown_sec": 0.0,
                "de_risk_reason": "risk_reduce",
                "score_threshold": 0.64,
                "release_threshold": 0.46,
                "confirm_hits": 1,
                "confirm_window_sec": 25.0,
                "cooldown_sec": 0.0,
                "forecast_weight": 0.35,
                "reason": "m1_structure_break",
            },
        },
    )
    monkeypatch.setattr(
        worker,
        "_direction_flip_score",
        lambda **_kwargs: (0.91, {"score": 0.91}),
    )
    monkeypatch.setattr(
        exit_worker,
        "all_factors",
        lambda: {
            "M1": {
                "rsi": 30.0,
                "adx": 26.0,
                "atr_pips": 1.4,
                "ma10": 150.00,
                "ma20": 150.08,
                "vwap_gap": -4.2,
                "ema_slope_10": -0.03,
            },
            "H4": {"adx": 24.0, "ma10": 150.0, "ma20": 150.1},
        },
    )

    async def _fake_pro_stop(*_args, **_kwargs):
        return False

    monkeypatch.setattr(exit_worker, "maybe_close_pro_stop", _fake_pro_stop)

    async def _fake_close_trade(*_args, **_kwargs):
        return False

    monkeypatch.setattr(exit_worker, "close_trade", _fake_close_trade)

    closed_reasons: list[str] = []

    async def _fake_close(trade_id, units, reason, pnl, client_order_id, allow_negative=False):
        closed_reasons.append(str(reason))

    monkeypatch.setattr(worker, "_close", _fake_close)

    now = datetime.now(timezone.utc)
    trade = {
        "trade_id": "T-derisk-fallback",
        "units": 1800,
        "price": 150.100,
        "unrealized_pl_pips": -3.2,
        "open_time": (now - timedelta(minutes=12)).isoformat(),
        "client_order_id": "cid-derisk-fallback",
        "entry_thesis": {"strategy_tag": "scalp_ping_5s_b_live"},
    }

    asyncio.run(worker._review_trade(trade, now, mid=150.030, range_active=False))

    assert closed_reasons == ["m1_structure_break"]


def test_direction_flip_short_side_overrides_apply(monkeypatch):
    worker = exit_worker.RangeFaderExitWorker()
    worker.exit_policy_start_ts = 0.0
    worker.new_policy_start_ts = 0.0

    monkeypatch.setattr(
        exit_worker,
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
        exit_worker,
        "all_factors",
        lambda: {
            "M1": {"rsi": 54.0, "adx": 20.0, "atr_pips": 1.2, "ma10": 150.0, "ma20": 150.0},
            "H4": {},
        },
    )

    async def _fake_pro_stop(*_args, **_kwargs):
        return False

    monkeypatch.setattr(exit_worker, "maybe_close_pro_stop", _fake_pro_stop)

    closed_reasons: list[str] = []

    async def _fake_close(trade_id, units, reason, pnl, client_order_id, allow_negative=False):
        closed_reasons.append(str(reason))

    monkeypatch.setattr(worker, "_close", _fake_close)

    now = datetime.now(timezone.utc)
    trade = {
        "trade_id": "T-short-override",
        "units": -1400,
        "price": 150.100,
        "unrealized_pl_pips": -1.1,
        "open_time": (now - timedelta(seconds=40)).isoformat(),
        "client_order_id": "cid-short-override",
        "entry_thesis": {"strategy_tag": "scalp_ping_5s_b_live"},
    }

    asyncio.run(worker._review_trade(trade, now, mid=150.030, range_active=False))

    assert closed_reasons == ["m1_structure_break"]


def test_non_range_max_hold_short_override_only_applies_to_short(monkeypatch):
    worker = exit_worker.RangeFaderExitWorker()
    worker.exit_policy_start_ts = 0.0
    worker.new_policy_start_ts = 0.0

    monkeypatch.setattr(
        exit_worker,
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
        exit_worker,
        "all_factors",
        lambda: {"M1": {"rsi": 50.0, "adx": 15.0, "atr_pips": 1.0}, "H4": {}},
    )

    async def _fake_pro_stop(*_args, **_kwargs):
        return False

    monkeypatch.setattr(exit_worker, "maybe_close_pro_stop", _fake_pro_stop)

    closed: list[tuple[str, str]] = []

    async def _fake_close(trade_id, units, reason, pnl, client_order_id, allow_negative=False):
        closed.append((str(trade_id), str(reason)))

    monkeypatch.setattr(worker, "_close", _fake_close)

    now = datetime.now(timezone.utc)
    short_trade = {
        "trade_id": "T-short-hold",
        "units": -1200,
        "price": 150.100,
        "unrealized_pl_pips": -0.8,
        "open_time": (now - timedelta(seconds=85)).isoformat(),
        "client_order_id": "cid-short-hold",
        "entry_thesis": {"strategy_tag": "scalp_ping_5s_b_live"},
    }
    long_trade = {
        "trade_id": "T-long-hold",
        "units": 1200,
        "price": 150.100,
        "unrealized_pl_pips": -0.8,
        "open_time": (now - timedelta(seconds=85)).isoformat(),
        "client_order_id": "cid-long-hold",
        "entry_thesis": {"strategy_tag": "scalp_ping_5s_b_live"},
    }

    asyncio.run(worker._review_trade(short_trade, now, mid=150.030, range_active=False))
    asyncio.run(worker._review_trade(long_trade, now, mid=150.030, range_active=False))

    assert ("T-short-hold", "time_stop_side") in closed
    assert all(trade_id != "T-long-hold" for trade_id, _ in closed)

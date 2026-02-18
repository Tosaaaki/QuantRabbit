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

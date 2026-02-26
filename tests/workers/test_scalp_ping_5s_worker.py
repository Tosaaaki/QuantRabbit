from __future__ import annotations

import asyncio
import datetime
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")


def _tick(epoch: float, bid: float, ask: float, mid: float) -> dict[str, float]:
    return {"epoch": epoch, "bid": bid, "ask": ask, "mid": mid}


def _sample_signal(side: str, *, mode: str = "momentum"):
    from workers.scalp_ping_5s import worker

    return worker.TickSignal(
        side=side,
        mode=mode,
        mode_score=1.0,
        momentum_score=1.0,
        revert_score=0.0,
        confidence=80,
        momentum_pips=0.4,
        trigger_pips=0.2,
        imbalance=0.7,
        tick_rate=8.0,
        span_sec=1.2,
        tick_age_ms=10.0,
        spread_pips=0.2,
        bid=150.0,
        ask=150.02,
        mid=150.01,
        range_pips=0.6,
        instant_range_pips=0.5,
        signal_window_sec=1.2,
    )


def test_build_tick_signal_detects_long(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.time, "time", lambda: 101.0)
    rows = [
        _tick(100.0, 150.000, 150.002, 150.001),
        _tick(100.2, 150.002, 150.004, 150.003),
        _tick(100.4, 150.004, 150.006, 150.005),
        _tick(100.6, 150.006, 150.008, 150.007),
        _tick(100.8, 150.008, 150.010, 150.009),
        _tick(101.0, 150.010, 150.012, 150.011),
    ]

    monkeypatch.setattr(worker.config, "MIN_TICKS", 6)
    monkeypatch.setattr(worker.config, "MIN_SIGNAL_TICKS", 4)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_SEC", 1.2)
    monkeypatch.setattr(worker.config, "MAX_TICK_AGE_MS", 900.0)
    monkeypatch.setattr(worker.config, "MOMENTUM_TRIGGER_PIPS", 0.7)
    monkeypatch.setattr(worker.config, "MOMENTUM_SPREAD_MULT", 0.6)
    monkeypatch.setattr(worker.config, "IMBALANCE_MIN", 0.58)
    monkeypatch.setattr(worker.config, "MIN_TICK_RATE", 3.0)

    sig = worker._build_tick_signal(rows, spread_pips=0.2)
    assert sig is not None
    assert sig.side == "long"
    assert sig.momentum_pips > 0
    assert sig.confidence >= worker.config.CONFIDENCE_FLOOR


def test_is_spread_stale_block_detects_state_stale() -> None:
    from workers.scalp_ping_5s import worker

    assert worker._is_spread_stale_block(
        blocked=True,
        spread_state={"stale": True},
        spread_reason="",
    ) is True


def test_is_spread_stale_block_detects_reason_only() -> None:
    from workers.scalp_ping_5s import worker

    assert worker._is_spread_stale_block(
        blocked=True,
        spread_state={"stale": False},
        spread_reason="spread_stale age=9000ms > max=4000ms",
    ) is True


def test_is_spread_stale_block_skips_when_not_blocked() -> None:
    from workers.scalp_ping_5s import worker

    assert worker._is_spread_stale_block(
        blocked=False,
        spread_state={"stale": True},
        spread_reason="spread_stale",
    ) is False


def test_resolve_allow_hour_entry_policy_inside_allow_hours(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "ALLOW_HOURS_JST", (10, 11))
    monkeypatch.setattr(worker.config, "ALLOW_HOURS_SOFT_ENABLED", False)

    # 01:05 UTC -> 10:05 JST
    now_utc = datetime.datetime(2026, 2, 26, 1, 5, tzinfo=datetime.timezone.utc)
    policy = worker._resolve_allow_hour_entry_policy(now_utc)
    assert policy.outside_hour_jst is None
    assert policy.hard_block is False
    assert policy.soft_mode is False
    assert policy.units_mult == pytest.approx(1.0, abs=1e-9)


def test_resolve_allow_hour_entry_policy_hard_block_outside(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "ALLOW_HOURS_JST", (18, 19, 22))
    monkeypatch.setattr(worker.config, "ALLOW_HOURS_SOFT_ENABLED", False)

    # 01:05 UTC -> 10:05 JST (outside allow window)
    now_utc = datetime.datetime(2026, 2, 26, 1, 5, tzinfo=datetime.timezone.utc)
    policy = worker._resolve_allow_hour_entry_policy(now_utc)
    assert policy.outside_hour_jst == 10
    assert policy.hard_block is True
    assert policy.soft_mode is False
    assert policy.units_mult == pytest.approx(1.0, abs=1e-9)


def test_resolve_allow_hour_entry_policy_soft_mode_outside(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "ALLOW_HOURS_JST", (18, 19, 22))
    monkeypatch.setattr(worker.config, "ALLOW_HOURS_SOFT_ENABLED", True)
    monkeypatch.setattr(worker.config, "ALLOW_HOURS_OUTSIDE_UNITS_MULT", 0.58)
    monkeypatch.setattr(worker.config, "ALLOW_HOURS_OUTSIDE_MIN_CONFIDENCE", 74)
    monkeypatch.setattr(worker.config, "ALLOW_HOURS_OUTSIDE_MIN_ENTRY_PROBABILITY", 0.67)
    monkeypatch.setattr(worker.config, "CONFIDENCE_FLOOR", 58)
    monkeypatch.setattr(worker.config, "CONFIDENCE_CEIL", 92)

    # 01:05 UTC -> 10:05 JST (outside allow window)
    now_utc = datetime.datetime(2026, 2, 26, 1, 5, tzinfo=datetime.timezone.utc)
    policy = worker._resolve_allow_hour_entry_policy(now_utc)
    assert policy.outside_hour_jst == 10
    assert policy.hard_block is False
    assert policy.soft_mode is True
    assert policy.units_mult == pytest.approx(0.58, abs=1e-9)
    assert policy.min_confidence == 74
    assert policy.min_entry_probability == pytest.approx(0.67, abs=1e-9)


def test_resolve_final_signal_for_side_filter_keeps_aligned_signal(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "SIDE_FILTER", "short")
    routed = _sample_signal("short", mode="revert")
    anchor = _sample_signal("short", mode="momentum")

    final_signal, reason = worker._resolve_final_signal_for_side_filter(
        routed_signal=routed,
        anchor_signal=anchor,
    )

    assert final_signal is routed
    assert reason == "side_filter_aligned"


def test_resolve_final_signal_for_side_filter_restores_anchor_after_flip(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "SIDE_FILTER", "short")
    routed = _sample_signal("long", mode="momentum_fflip")
    anchor = _sample_signal("short", mode="momentum")

    final_signal, reason = worker._resolve_final_signal_for_side_filter(
        routed_signal=routed,
        anchor_signal=anchor,
    )

    assert final_signal is not None
    assert final_signal.side == "short"
    assert final_signal.mode == "momentum_sidefilter"
    assert reason == "side_filter_fallback:long->short"


def test_resolve_final_signal_for_side_filter_blocks_without_anchor(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "SIDE_FILTER", "short")
    routed = _sample_signal("long", mode="momentum_fflip")

    final_signal, reason = worker._resolve_final_signal_for_side_filter(
        routed_signal=routed,
        anchor_signal=None,
    )

    assert final_signal is None
    assert reason == "side_filter_final_block:long"


def test_maybe_rescue_min_units_applies_when_thresholds_met(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "MIN_UNITS", 1)
    monkeypatch.setattr(worker.config, "MIN_UNITS_RESCUE_ENABLED", True)
    monkeypatch.setattr(worker.config, "MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY", 0.58)
    monkeypatch.setattr(worker.config, "MIN_UNITS_RESCUE_MIN_CONFIDENCE", 70)

    units, reason = worker._maybe_rescue_min_units(
        units=0,
        base_units=180,
        units_risk=120,
        entry_probability=0.72,
        confidence=83,
    )

    assert units == 1
    assert reason == "rescued"


def test_maybe_rescue_min_units_skips_when_probability_is_low(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "MIN_UNITS", 1)
    monkeypatch.setattr(worker.config, "MIN_UNITS_RESCUE_ENABLED", True)
    monkeypatch.setattr(worker.config, "MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY", 0.60)
    monkeypatch.setattr(worker.config, "MIN_UNITS_RESCUE_MIN_CONFIDENCE", 70)

    units, reason = worker._maybe_rescue_min_units(
        units=0,
        base_units=180,
        units_risk=120,
        entry_probability=0.55,
        confidence=83,
    )

    assert units == 0
    assert reason == "probability_below_rescue_floor"


def test_build_tick_signal_rejects_chasing_long(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.time, "time", lambda: 202.0)
    rows = [
        _tick(200.0, 150.000, 150.002, 150.001),
        _tick(200.2, 150.004, 150.006, 150.005),
        _tick(200.4, 150.008, 150.010, 150.009),
        _tick(200.6, 150.010, 150.012, 150.011),
        _tick(200.8, 150.011, 150.013, 150.012),
        _tick(201.0, 150.011, 150.013, 150.012),
    ]

    monkeypatch.setattr(worker.config, "MIN_TICKS", 6)
    monkeypatch.setattr(worker.config, "MIN_SIGNAL_TICKS", 4)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_SEC", 1.2)
    monkeypatch.setattr(worker.config, "MAX_TICK_AGE_MS", 900.0)
    monkeypatch.setattr(worker.config, "MOMENTUM_TRIGGER_PIPS", 0.8)
    monkeypatch.setattr(worker.config, "MOMENTUM_SPREAD_MULT", 0.6)
    monkeypatch.setattr(worker.config, "IMBALANCE_MIN", 0.58)
    monkeypatch.setattr(worker.config, "MIN_TICK_RATE", 3.0)

    sig = worker._build_tick_signal(rows, spread_pips=0.2)
    assert sig is None


def test_build_tick_signal_rejects_low_imbalance(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.time, "time", lambda: 201.0)
    rows = [
        _tick(200.0, 150.000, 150.002, 150.001),
        _tick(200.2, 150.001, 150.003, 150.002),
        _tick(200.4, 150.000, 150.002, 150.001),
        _tick(200.6, 150.001, 150.003, 150.002),
        _tick(200.8, 150.000, 150.002, 150.001),
        _tick(201.0, 150.001, 150.003, 150.002),
    ]

    monkeypatch.setattr(worker.config, "MIN_TICKS", 6)
    monkeypatch.setattr(worker.config, "MIN_SIGNAL_TICKS", 4)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_SEC", 1.2)
    monkeypatch.setattr(worker.config, "MAX_TICK_AGE_MS", 900.0)
    monkeypatch.setattr(worker.config, "MOMENTUM_TRIGGER_PIPS", 0.4)
    monkeypatch.setattr(worker.config, "MOMENTUM_SPREAD_MULT", 0.4)
    monkeypatch.setattr(worker.config, "IMBALANCE_MIN", 0.85)
    monkeypatch.setattr(worker.config, "MIN_TICK_RATE", 2.0)

    sig = worker._build_tick_signal(rows, spread_pips=0.2)
    assert sig is None


def test_build_tick_signal_keeps_continuing_momentum(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.time, "time", lambda: 203.0)
    rows = [
        _tick(202.0, 150.001, 150.003, 150.002),
        _tick(202.2, 150.004, 150.006, 150.005),
        _tick(202.4, 150.007, 150.009, 150.008),
        _tick(202.6, 150.010, 150.012, 150.011),
        _tick(202.8, 150.013, 150.015, 150.014),
        _tick(203.0, 150.016, 150.018, 150.017),
    ]

    monkeypatch.setattr(worker.config, "MIN_TICKS", 6)
    monkeypatch.setattr(worker.config, "MIN_SIGNAL_TICKS", 4)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_SEC", 1.2)
    monkeypatch.setattr(worker.config, "MAX_TICK_AGE_MS", 900.0)
    monkeypatch.setattr(worker.config, "MOMENTUM_TRIGGER_PIPS", 0.8)
    monkeypatch.setattr(worker.config, "MOMENTUM_SPREAD_MULT", 0.6)
    monkeypatch.setattr(worker.config, "IMBALANCE_MIN", 0.58)
    monkeypatch.setattr(worker.config, "MIN_TICK_RATE", 3.0)

    sig = worker._build_tick_signal(rows, spread_pips=0.2)
    assert sig is not None
    assert sig.side == "long"
    assert sig.momentum_pips > 0


def test_compute_targets_is_spread_aware() -> None:
    from workers.scalp_ping_5s import worker

    tp_pips, sl_pips = worker._compute_targets(
        spread_pips=0.9,
        momentum_pips=2.4,
        tp_profile=worker.TpTimingProfile(),
    )
    assert tp_pips >= min(
        worker.config.TP_MAX_PIPS,
        max(worker.config.TP_BASE_PIPS, 0.9 + worker.config.TP_NET_MIN_PIPS),
    )
    assert tp_pips <= worker.config.TP_MAX_PIPS
    assert sl_pips >= (0.9 * worker.config.SL_SPREAD_MULT + worker.config.SL_SPREAD_BUFFER_PIPS)
    assert sl_pips <= worker.config.SL_MAX_PIPS


def test_compute_targets_uses_spread_plus_micro_edge(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "TP_BASE_PIPS", 0.2)
    monkeypatch.setattr(worker.config, "TP_NET_MIN_PIPS", 0.25)
    monkeypatch.setattr(worker.config, "TP_MOMENTUM_BONUS_MAX", 0.0)
    monkeypatch.setattr(worker.config, "TP_MAX_PIPS", 1.0)

    tp_pips, _ = worker._compute_targets(
        spread_pips=0.30,
        momentum_pips=0.1,
        tp_profile=worker.TpTimingProfile(),
    )
    assert tp_pips == pytest.approx(0.55, abs=1e-6)


def test_directional_bias_scale_downsizes_contra_flow(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    rows = [
        _tick(100.0, 153.000, 153.002, 153.001),
        _tick(101.0, 153.003, 153.005, 153.004),
        _tick(102.0, 153.005, 153.007, 153.006),
        _tick(103.0, 153.007, 153.009, 153.008),
        _tick(104.0, 153.009, 153.011, 153.010),
    ]

    monkeypatch.setattr(worker.config, "SIDE_BIAS_ENABLED", True)
    monkeypatch.setattr(worker.config, "SIDE_BIAS_WINDOW_SEC", 10.0)
    monkeypatch.setattr(worker.config, "SIDE_BIAS_MIN_TICKS", 4)
    monkeypatch.setattr(worker.config, "SIDE_BIAS_MIN_DRIFT_PIPS", 0.4)
    monkeypatch.setattr(worker.config, "SIDE_BIAS_SCALE_GAIN", 0.5)
    monkeypatch.setattr(worker.config, "SIDE_BIAS_SCALE_FLOOR", 0.3)
    monkeypatch.setattr(worker.config, "SIDE_BIAS_BLOCK_THRESHOLD", 0.0)

    short_scale, short_meta = worker._directional_bias_scale(rows, "short")
    long_scale, _ = worker._directional_bias_scale(rows, "long")

    assert short_scale == pytest.approx(0.75, abs=1e-9)
    assert short_meta["drift_pips"] > 0.0
    assert long_scale == pytest.approx(1.0, abs=1e-9)


def test_build_technical_trade_profile_routes_tighten(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "TECH_ROUTER_ENABLED", True)
    monkeypatch.setattr(worker.config, "TECH_ROUTER_COUNTER_TP_MULT", 0.84)
    monkeypatch.setattr(worker.config, "TECH_ROUTER_COUNTER_SL_MULT", 0.91)
    monkeypatch.setattr(worker.config, "TECH_ROUTER_COUNTER_HOLD_MULT", 0.66)
    monkeypatch.setattr(worker.config, "TECH_ROUTER_COUNTER_HARD_LOSS_MULT", 0.87)
    monkeypatch.setattr(worker.config, "TECH_ROUTER_EDGE_TP_BOOST_MAX", 0.0)
    monkeypatch.setattr(worker.config, "TECH_ROUTER_EDGE_HOLD_BOOST_MAX", 0.0)
    monkeypatch.setattr(worker.config, "TECH_ROUTER_EDGE_HARD_LOSS_BOOST_MAX", 0.0)

    profile = worker._build_technical_trade_profile(
        route_reasons=["mtf", "horizon"],
        lookahead_decision=SimpleNamespace(allow_entry=False, edge_pips=0.0),
    )

    assert profile.counter_pressure is True
    assert profile.route_reasons == ("mtf", "horizon")
    assert profile.tp_mult == pytest.approx(0.84, abs=1e-9)
    assert profile.sl_mult == pytest.approx(0.91, abs=1e-9)
    assert profile.hold_mult == pytest.approx(0.66, abs=1e-9)
    assert profile.hard_loss_mult == pytest.approx(0.87, abs=1e-9)


def test_build_technical_trade_profile_strong_edge_boosts(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "TECH_ROUTER_ENABLED", True)
    monkeypatch.setattr(worker.config, "LOOKAHEAD_EDGE_REF_PIPS", 0.6)
    monkeypatch.setattr(worker.config, "TECH_ROUTER_EDGE_TP_BOOST_MAX", 0.2)
    monkeypatch.setattr(worker.config, "TECH_ROUTER_EDGE_HOLD_BOOST_MAX", 0.3)
    monkeypatch.setattr(worker.config, "TECH_ROUTER_EDGE_HARD_LOSS_BOOST_MAX", 0.1)

    profile = worker._build_technical_trade_profile(
        route_reasons=[],
        lookahead_decision=SimpleNamespace(allow_entry=True, edge_pips=1.3),
    )

    assert profile.counter_pressure is False
    assert profile.route_reasons == ()
    assert profile.tp_mult > 1.0
    assert profile.hold_mult > 1.0
    assert profile.hard_loss_mult > 1.0
    assert profile.sl_mult == pytest.approx(1.0, abs=1e-9)


def test_scaled_force_exit_thresholds_apply_profile(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "TECH_ROUTER_HOLD_MIN_SEC", 25.0)
    monkeypatch.setattr(worker.config, "TECH_ROUTER_HOLD_MAX_MULT", 1.4)
    profile = worker.TechnicalTradeProfile(
        tp_mult=1.0,
        sl_mult=1.0,
        hold_mult=0.55,
        hard_loss_mult=0.8,
        counter_pressure=True,
        route_reasons=("mtf",),
    )

    hold_sec, hard_loss = worker._scaled_force_exit_thresholds(
        base_max_hold_sec=120.0,
        base_hard_loss_pips=3.0,
        profile=profile,
    )
    assert hold_sec == pytest.approx(66.0, abs=1e-9)
    assert hard_loss == pytest.approx(2.4, abs=1e-9)


def test_trade_force_exit_threshold_overrides_from_entry_thesis() -> None:
    from workers.scalp_ping_5s import worker

    trade = {
        "entry_thesis": {
            "force_exit_max_hold_sec": 45.0,
            "force_exit_max_floating_loss_pips": 1.8,
        }
    }
    assert worker._trade_force_exit_max_hold_sec(trade, 120.0) == pytest.approx(45.0, abs=1e-9)
    assert worker._trade_force_exit_hard_loss_pips(trade, 3.0) == pytest.approx(1.8, abs=1e-9)


def test_compute_trap_state_active_when_hedged_and_underwater(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "TRAP_MIN_LONG_UNITS", 8000)
    monkeypatch.setattr(worker.config, "TRAP_MIN_SHORT_UNITS", 8000)
    monkeypatch.setattr(worker.config, "TRAP_MAX_NET_RATIO", 0.45)
    monkeypatch.setattr(worker.config, "TRAP_MIN_COMBINED_DD_PIPS", 0.8)
    monkeypatch.setattr(worker.config, "TRAP_REQUIRE_NET_LOSS", True)

    positions = {
        "scalp_fast": {
            "long_units": 12000,
            "short_units": 11000,
            "long_avg_price": 150.05,
            "short_avg_price": 149.95,
            "unrealized_pl": -4200.0,
        },
        "manual": {
            "long_units": 2000,
            "short_units": 3000,
            "long_avg_price": 150.04,
            "short_avg_price": 149.96,
            "unrealized_pl": -600.0,
        },
        "__net__": {"units": 0},
    }

    state = worker._compute_trap_state(positions, mid_price=150.00)
    assert state.active is True
    assert state.long_units == pytest.approx(14000.0)
    assert state.short_units == pytest.approx(14000.0)
    assert state.net_ratio <= 0.01
    assert state.combined_dd_pips >= 0.8
    assert state.unrealized_pl < 0.0


def test_compute_trap_state_blocks_when_unrealized_not_negative(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "TRAP_MIN_LONG_UNITS", 8000)
    monkeypatch.setattr(worker.config, "TRAP_MIN_SHORT_UNITS", 8000)
    monkeypatch.setattr(worker.config, "TRAP_MAX_NET_RATIO", 0.45)
    monkeypatch.setattr(worker.config, "TRAP_MIN_COMBINED_DD_PIPS", 0.8)
    monkeypatch.setattr(worker.config, "TRAP_REQUIRE_NET_LOSS", True)

    positions = {
        "scalp_fast": {
            "long_units": 9000,
            "short_units": 9000,
            "long_avg_price": 150.03,
            "short_avg_price": 149.97,
            "unrealized_pl": 120.0,
        }
    }

    state = worker._compute_trap_state(positions, mid_price=150.00)
    assert state.combined_dd_pips >= 0.8
    assert state.unrealized_pl > 0.0
    assert state.active is False


def test_allow_signal_when_max_active_prefers_rebalance_side(monkeypatch) -> None:
    monkeypatch.setenv("oanda_token", "dummy")
    monkeypatch.setenv("oanda_account_id", "dummy")
    monkeypatch.setenv("oanda_practice", "true")
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "MAX_ACTIVE_TRADES", 10)
    monkeypatch.setattr(worker.config, "ALLOW_OPPOSITE_WHEN_MAX_ACTIVE", True)

    assert (
        worker._allow_signal_when_max_active(
            side="short",
            active_total=10,
            active_long=10,
            active_short=0,
            max_active_trades=10,
        )
        is True
    )
    assert (
        worker._allow_signal_when_max_active(
            side="long",
            active_total=10,
            active_long=10,
            active_short=0,
            max_active_trades=10,
        )
        is False
    )
    assert (
        worker._allow_signal_when_max_active(
            side="long",
            active_total=10,
            active_long=5,
            active_short=5,
            max_active_trades=10,
        )
        is False
    )
    assert (
        worker._allow_signal_when_max_active(
            side="long",
            active_total=9,
            active_long=9,
            active_short=0,
            max_active_trades=10,
        )
        is True
    )


def test_allow_signal_when_max_active_respects_disable_flag(monkeypatch) -> None:
    monkeypatch.setenv("oanda_token", "dummy")
    monkeypatch.setenv("oanda_account_id", "dummy")
    monkeypatch.setenv("oanda_practice", "true")
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "MAX_ACTIVE_TRADES", 10)
    monkeypatch.setattr(worker.config, "ALLOW_OPPOSITE_WHEN_MAX_ACTIVE", False)

    assert (
        worker._allow_signal_when_max_active(
            side="short",
            active_total=10,
            active_long=10,
            active_short=0,
            max_active_trades=10,
        )
        is False
    )


def test_resolve_active_caps_expands_when_margin_headroom_is_healthy(monkeypatch) -> None:
    monkeypatch.setenv("oanda_token", "dummy")
    monkeypatch.setenv("oanda_account_id", "dummy")
    monkeypatch.setenv("oanda_practice", "true")
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "MAX_ACTIVE_TRADES", 20)
    monkeypatch.setattr(worker.config, "MAX_PER_DIRECTION", 12)
    monkeypatch.setattr(worker.config, "ACTIVE_CAP_MARGIN_BYPASS_ENABLED", True)
    monkeypatch.setattr(worker.config, "ACTIVE_CAP_BYPASS_MIN_FREE_RATIO", 0.06)
    monkeypatch.setattr(worker.config, "ACTIVE_CAP_BYPASS_MIN_MARGIN_AVAILABLE_JPY", 8000.0)
    monkeypatch.setattr(worker.config, "ACTIVE_CAP_BYPASS_EXTRA_TOTAL", 12)
    monkeypatch.setattr(worker.config, "ACTIVE_CAP_BYPASS_EXTRA_PER_DIRECTION", 8)

    total_cap, side_cap, expanded = worker._resolve_active_caps(
        free_ratio=0.074,
        margin_available=11239.4,
    )

    assert expanded is True
    assert total_cap == 32
    assert side_cap == 20


def test_resolve_active_caps_keeps_base_when_margin_headroom_is_low(monkeypatch) -> None:
    monkeypatch.setenv("oanda_token", "dummy")
    monkeypatch.setenv("oanda_account_id", "dummy")
    monkeypatch.setenv("oanda_practice", "true")
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "MAX_ACTIVE_TRADES", 20)
    monkeypatch.setattr(worker.config, "MAX_PER_DIRECTION", 12)
    monkeypatch.setattr(worker.config, "ACTIVE_CAP_MARGIN_BYPASS_ENABLED", True)
    monkeypatch.setattr(worker.config, "ACTIVE_CAP_BYPASS_MIN_FREE_RATIO", 0.06)
    monkeypatch.setattr(worker.config, "ACTIVE_CAP_BYPASS_MIN_MARGIN_AVAILABLE_JPY", 8000.0)
    monkeypatch.setattr(worker.config, "ACTIVE_CAP_BYPASS_EXTRA_TOTAL", 12)
    monkeypatch.setattr(worker.config, "ACTIVE_CAP_BYPASS_EXTRA_PER_DIRECTION", 8)

    total_cap, side_cap, expanded = worker._resolve_active_caps(
        free_ratio=0.051,
        margin_available=11239.4,
    )
    assert expanded is False
    assert total_cap == 20
    assert side_cap == 12


def test_resolve_dynamic_direction_cap_tightens_to_min_on_adverse_cluster(monkeypatch) -> None:
    monkeypatch.setenv("oanda_token", "dummy")
    monkeypatch.setenv("oanda_account_id", "dummy")
    monkeypatch.setenv("oanda_practice", "true")
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "DYNAMIC_DIRECTION_CAP_ENABLED", True)
    monkeypatch.setattr(worker.config, "DYNAMIC_DIRECTION_CAP_MIN", 1)
    monkeypatch.setattr(worker.config, "DYNAMIC_DIRECTION_CAP_WEAK_CAP", 2)
    monkeypatch.setattr(worker.config, "DYNAMIC_DIRECTION_CAP_WEAK_BIAS_SCORE", 0.52)
    monkeypatch.setattr(worker.config, "DYNAMIC_DIRECTION_CAP_WEAK_HORIZON_SCORE", 0.56)
    monkeypatch.setattr(worker.config, "DYNAMIC_DIRECTION_CAP_ADVERSE_ACTIVE_START", 2)
    monkeypatch.setattr(worker.config, "DYNAMIC_DIRECTION_CAP_ADVERSE_DD_PIPS", 0.45)
    monkeypatch.setattr(worker.config, "DYNAMIC_DIRECTION_CAP_ADVERSE_CAP", 2)
    monkeypatch.setattr(worker.config, "DYNAMIC_DIRECTION_CAP_METRICS_ADVERSE_CAP", 1)

    eval_info = worker.SideAdverseStackEval(
        current_side="long",
        target_side="short",
        active_same_side=3,
        active_opposite_side=0,
        current_trades=12,
        target_trades=7,
        current_sl_rate=0.62,
        target_sl_rate=0.31,
        current_market_plus_rate=0.14,
        target_market_plus_rate=0.39,
        side_mult=0.72,
        dd_mult=0.64,
        units_mult=0.46,
        dd_pips=0.82,
        adverse=True,
        reason="metrics_adverse+dd_scale",
    )
    direction_bias = worker.DirectionBias(
        side="short",
        score=0.22,
        momentum_pips=-0.2,
        flow=-0.4,
        range_pips=0.6,
        vol_norm=0.3,
        tick_rate=0.9,
        span_sec=2.0,
    )
    horizon = worker.HorizonBias(
        long_side="short",
        long_score=0.24,
        mid_side="short",
        mid_score=0.20,
        short_side="short",
        short_score=0.18,
        micro_side="short",
        micro_score=0.16,
        composite_side="short",
        composite_score=0.22,
        agreement=1,
    )

    cap, reason = worker._resolve_dynamic_direction_cap(
        side="long",
        base_cap=4,
        side_adverse_eval=eval_info,
        direction_bias=direction_bias,
        horizon=horizon,
    )

    assert cap == 1
    assert "metrics_adverse" in reason


def test_resolve_dynamic_direction_cap_returns_base_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("oanda_token", "dummy")
    monkeypatch.setenv("oanda_account_id", "dummy")
    monkeypatch.setenv("oanda_practice", "true")
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "DYNAMIC_DIRECTION_CAP_ENABLED", False)
    monkeypatch.setattr(worker.config, "DYNAMIC_DIRECTION_CAP_MIN", 1)

    eval_info = worker.SideAdverseStackEval(
        current_side="long",
        target_side="short",
        active_same_side=4,
        active_opposite_side=0,
        current_trades=9,
        target_trades=7,
        current_sl_rate=0.60,
        target_sl_rate=0.28,
        current_market_plus_rate=0.12,
        target_market_plus_rate=0.35,
        side_mult=0.70,
        dd_mult=0.62,
        units_mult=0.44,
        dd_pips=0.90,
        adverse=True,
        reason="metrics_adverse+dd_scale",
    )

    cap, reason = worker._resolve_dynamic_direction_cap(
        side="long",
        base_cap=4,
        side_adverse_eval=eval_info,
        direction_bias=None,
        horizon=None,
    )

    assert cap == 4
    assert reason == "disabled"


def test_enforce_new_entry_time_stop_uses_entry_thesis_hold_override(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "STRATEGY_TAG", "scalp_ping_5s_live")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_HOLD_SEC", 900.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_FLOATING_LOSS_PIPS", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERY_WINDOW_SEC", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERABLE_LOSS_PIPS", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_ACTIONS", 3)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_REASON", "time_stop")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_FLOATING_LOSS_REASON", "max_floating_loss")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERY_REASON", "no_recovery")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_REQUIRE_POLICY_GENERATION", False)

    calls: list[str] = []

    async def _fake_close_trade(
        trade_id: str,
        units: int | None = None,
        client_order_id: str | None = None,
        allow_negative: bool = False,
        exit_reason: str | None = None,
        env_prefix: str | None = None,
    ) -> bool:
        calls.append(trade_id)
        return True

    monkeypatch.setattr(worker, "close_trade", _fake_close_trade)

    now_utc = datetime.datetime(2026, 2, 12, 22, 0, 0, tzinfo=datetime.timezone.utc)
    pocket_info = {
        "open_trades": [
            {
                "trade_id": "override-hit",
                "units": 1000,
                "open_time": "2026-02-12T21:58:00+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "entry_thesis": {
                    "strategy_tag": "scalp_ping_5s_live",
                    "force_exit_max_hold_sec": 60.0,
                },
            },
            {
                "trade_id": "override-safe",
                "units": 1000,
                "open_time": "2026-02-12T21:59:30+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "entry_thesis": {
                    "strategy_tag": "scalp_ping_5s_live",
                    "force_exit_max_hold_sec": 180.0,
                },
            },
        ]
    }

    closed = asyncio.run(
        worker._enforce_new_entry_time_stop(
            pocket_info=pocket_info,
            now_utc=now_utc,
            logger=worker.LOG,
        )
    )
    assert closed == 1
    assert calls == ["override-hit"]


@pytest.mark.asyncio
async def test_enforce_new_entry_time_stop_respects_policy_generation(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "STRATEGY_TAG", "scalp_ping_5s_live")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_HOLD_SEC", 900.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_FLOATING_LOSS_PIPS", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERY_WINDOW_SEC", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERABLE_LOSS_PIPS", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_ACTIONS", 3)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_REASON", "time_stop")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_FLOATING_LOSS_REASON", "max_floating_loss")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERY_REASON", "no_recovery")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_REQUIRE_POLICY_GENERATION", True)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_POLICY_GENERATION", "2026-02-11-losscap-v1")

    calls: list[tuple[str, int, str | None, bool, str | None, str | None]] = []

    async def _fake_close_trade(
        trade_id: str,
        units: int | None = None,
        client_order_id: str | None = None,
        allow_negative: bool = False,
        exit_reason: str | None = None,
        env_prefix: str | None = None,
    ) -> bool:
        calls.append((trade_id, int(units or 0), client_order_id, allow_negative, exit_reason, env_prefix))
        return True

    monkeypatch.setattr(worker, "close_trade", _fake_close_trade)

    now_utc = datetime.datetime(2026, 2, 11, 22, 0, 0, tzinfo=datetime.timezone.utc)
    pocket_info = {
        "open_trades": [
            {
                "trade_id": "old-no-generation",
                "units": 900,
                "open_time": "2026-02-11T20:00:00+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "entry_thesis": {"strategy_tag": "scalp_ping_5s_live"},
            },
            {
                "trade_id": "new-matching",
                "units": -1200,
                "open_time": "2026-02-11T21:40:00+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "client_id": "qr-new",
                "entry_thesis": {
                    "strategy_tag": "scalp_ping_5s_live",
                    "policy_generation": "2026-02-11-losscap-v1",
                },
            },
            {
                "trade_id": "new-mismatch",
                "units": 1200,
                "open_time": "2026-02-11T21:40:00+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "entry_thesis": {
                    "strategy_tag": "scalp_ping_5s_live",
                    "policy_generation": "old-generation",
                },
            },
            {
                "trade_id": "new-too-young",
                "units": 1200,
                "open_time": "2026-02-11T21:53:30+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "entry_thesis": {
                    "strategy_tag": "scalp_ping_5s_live",
                    "policy_generation": "2026-02-11-losscap-v1",
                },
            },
        ]
    }

    closed = await worker._enforce_new_entry_time_stop(
        pocket_info=pocket_info,
        now_utc=now_utc,
        logger=worker.LOG,
    )
    assert closed == 1
    assert len(calls) == 1
    assert calls[0][0] == "new-matching"
    assert calls[0][1] == 1200
    assert calls[0][2] == "qr-new"
    assert calls[0][3] is True
    assert calls[0][4] == "time_stop"


@pytest.mark.asyncio
async def test_enforce_new_entry_time_stop_skips_protected_existing_trades(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "STRATEGY_TAG", "scalp_ping_5s_live")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_HOLD_SEC", 300.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_FLOATING_LOSS_PIPS", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERY_WINDOW_SEC", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERABLE_LOSS_PIPS", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_ACTIONS", 3)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_REASON", "time_stop")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_REQUIRE_POLICY_GENERATION", True)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_POLICY_GENERATION", "2026-02-12-ping5-root-v1")

    calls: list[tuple[str, int, str | None, bool, str | None, str | None]] = []

    async def _fake_close_trade(
        trade_id: str,
        units: int | None = None,
        client_order_id: str | None = None,
        allow_negative: bool = False,
        exit_reason: str | None = None,
        env_prefix: str | None = None,
    ) -> bool:
        calls.append((trade_id, int(units or 0), client_order_id, allow_negative, exit_reason, env_prefix))
        return True

    monkeypatch.setattr(worker, "close_trade", _fake_close_trade)

    now_utc = datetime.datetime(2026, 2, 12, 3, 0, 0, tzinfo=datetime.timezone.utc)
    pocket_info = {
        "open_trades": [
            {
                "trade_id": "old-protected",
                "units": 1000,
                "open_time": "2026-02-12T00:50:00+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "entry_thesis": {
                    "strategy_tag": "scalp_ping_5s_live",
                    "policy_generation": "2026-02-12-ping5-root-v1",
                },
            },
            {
                "trade_id": "new-eligible",
                "units": 1000,
                "open_time": "2026-02-12T02:40:00+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "client_id": "qr-new-eligible",
                "entry_thesis": {
                    "strategy_tag": "scalp_ping_5s_live",
                    "policy_generation": "2026-02-12-ping5-root-v1",
                },
            },
        ]
    }

    closed = await worker._enforce_new_entry_time_stop(
        pocket_info=pocket_info,
        now_utc=now_utc,
        logger=worker.LOG,
        protected_trade_ids={"old-protected"},
    )
    assert closed == 1
    assert len(calls) == 1
    assert calls[0][0] == "new-eligible"
    assert calls[0][2] == "qr-new-eligible"
    assert calls[0][4] == "time_stop"


@pytest.mark.asyncio
async def test_enforce_new_entry_time_stop_closes_no_recovery_loss(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "STRATEGY_TAG", "scalp_ping_5s_live")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_HOLD_SEC", 900.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_FLOATING_LOSS_PIPS", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERY_WINDOW_SEC", 180.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERABLE_LOSS_PIPS", 1.5)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_ACTIONS", 3)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_REASON", "time_stop")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_FLOATING_LOSS_REASON", "max_floating_loss")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERY_REASON", "no_recovery")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_REQUIRE_POLICY_GENERATION", True)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_POLICY_GENERATION", "2026-02-12-hold600-v2")

    calls: list[tuple[str, int, str | None, bool, str | None, str | None]] = []

    async def _fake_close_trade(
        trade_id: str,
        units: int | None = None,
        client_order_id: str | None = None,
        allow_negative: bool = False,
        exit_reason: str | None = None,
        env_prefix: str | None = None,
    ) -> bool:
        calls.append((trade_id, int(units or 0), client_order_id, allow_negative, exit_reason, env_prefix))
        return True

    monkeypatch.setattr(worker, "close_trade", _fake_close_trade)

    now_utc = datetime.datetime(2026, 2, 12, 1, 0, 0, tzinfo=datetime.timezone.utc)
    pocket_info = {
        "open_trades": [
            {
                "trade_id": "recover-timeout",
                "units": 1200,
                "open_time": "2026-02-12T00:55:00+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "client_id": "qr-rec-timeout",
                "unrealized_pl_pips": -2.1,
                "entry_thesis": {
                    "strategy_tag": "scalp_ping_5s_live",
                    "policy_generation": "2026-02-12-hold600-v2",
                },
            },
            {
                "trade_id": "still-recovering",
                "units": 1200,
                "open_time": "2026-02-12T00:58:40+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "unrealized_pl_pips": -4.0,
                "entry_thesis": {
                    "strategy_tag": "scalp_ping_5s_live",
                    "policy_generation": "2026-02-12-hold600-v2",
                },
            },
            {
                "trade_id": "small-loss",
                "units": 1200,
                "open_time": "2026-02-12T00:55:00+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "unrealized_pl_pips": -0.9,
                "entry_thesis": {
                    "strategy_tag": "scalp_ping_5s_live",
                    "policy_generation": "2026-02-12-hold600-v2",
                },
            },
        ]
    }

    closed = await worker._enforce_new_entry_time_stop(
        pocket_info=pocket_info,
        now_utc=now_utc,
        logger=worker.LOG,
    )
    assert closed == 1
    assert len(calls) == 1
    assert calls[0][0] == "recover-timeout"
    assert calls[0][4] == "no_recovery"


@pytest.mark.asyncio
async def test_enforce_new_entry_time_stop_closes_max_floating_loss(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "STRATEGY_TAG", "scalp_ping_5s_live")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_HOLD_SEC", 900.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_FLOATING_LOSS_PIPS", 3.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERY_WINDOW_SEC", 180.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERABLE_LOSS_PIPS", 1.5)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_ACTIONS", 3)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_REASON", "time_stop")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_FLOATING_LOSS_REASON", "max_floating_loss")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERY_REASON", "no_recovery")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_REQUIRE_POLICY_GENERATION", True)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_POLICY_GENERATION", "2026-02-12-hold600-v2")

    calls: list[tuple[str, int, str | None, bool, str | None, str | None]] = []

    async def _fake_close_trade(
        trade_id: str,
        units: int | None = None,
        client_order_id: str | None = None,
        allow_negative: bool = False,
        exit_reason: str | None = None,
        env_prefix: str | None = None,
    ) -> bool:
        calls.append((trade_id, int(units or 0), client_order_id, allow_negative, exit_reason, env_prefix))
        return True

    monkeypatch.setattr(worker, "close_trade", _fake_close_trade)

    now_utc = datetime.datetime(2026, 2, 12, 1, 0, 0, tzinfo=datetime.timezone.utc)
    pocket_info = {
        "open_trades": [
            {
                "trade_id": "hard-loss",
                "units": 1000,
                "open_time": "2026-02-12T00:59:20+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "unrealized_pl_pips": -3.5,
                "entry_thesis": {
                    "strategy_tag": "scalp_ping_5s_live",
                    "policy_generation": "2026-02-12-hold600-v2",
                },
            }
        ]
    }

    closed = await worker._enforce_new_entry_time_stop(
        pocket_info=pocket_info,
        now_utc=now_utc,
        logger=worker.LOG,
    )
    assert closed == 1
    assert len(calls) == 1
    assert calls[0][0] == "hard-loss"
    assert calls[0][4] == "max_floating_loss"


@pytest.mark.asyncio
async def test_enforce_new_entry_time_stop_closes_giveback_lock(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    worker._TRADE_MFE_PIPS.clear()
    monkeypatch.setattr(worker.config, "STRATEGY_TAG", "scalp_ping_5s_live")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_HOLD_SEC", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_FLOATING_LOSS_PIPS", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERY_WINDOW_SEC", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_RECOVERABLE_LOSS_PIPS", 0.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_GIVEBACK_ENABLED", True)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_GIVEBACK_ARM_PIPS", 1.5)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_GIVEBACK_BACKOFF_PIPS", 1.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_GIVEBACK_MIN_HOLD_SEC", 15.0)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_GIVEBACK_PROTECT_PIPS", -0.1)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_GIVEBACK_REASON", "giveback_lock")
    monkeypatch.setattr(worker.config, "FORCE_EXIT_MAX_ACTIONS", 3)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_REQUIRE_POLICY_GENERATION", True)
    monkeypatch.setattr(worker.config, "FORCE_EXIT_POLICY_GENERATION", "2026-02-12-giveback-v1")

    calls: list[tuple[str, int, str | None, bool, str | None, str | None]] = []

    async def _fake_close_trade(
        trade_id: str,
        units: int | None = None,
        client_order_id: str | None = None,
        allow_negative: bool = False,
        exit_reason: str | None = None,
        env_prefix: str | None = None,
    ) -> bool:
        calls.append((trade_id, int(units or 0), client_order_id, allow_negative, exit_reason, env_prefix))
        return True

    monkeypatch.setattr(worker, "close_trade", _fake_close_trade)

    first_now = datetime.datetime(2026, 2, 12, 2, 0, 0, tzinfo=datetime.timezone.utc)
    pocket_info_peak = {
        "open_trades": [
            {
                "trade_id": "giveback-1",
                "units": 1100,
                "open_time": "2026-02-12T01:59:20+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "unrealized_pl_pips": 1.8,
                "entry_thesis": {
                    "strategy_tag": "scalp_ping_5s_live",
                    "policy_generation": "2026-02-12-giveback-v1",
                },
            }
        ]
    }
    closed_first = await worker._enforce_new_entry_time_stop(
        pocket_info=pocket_info_peak,
        now_utc=first_now,
        logger=worker.LOG,
    )
    assert closed_first == 0
    assert calls == []

    second_now = datetime.datetime(2026, 2, 12, 2, 0, 20, tzinfo=datetime.timezone.utc)
    pocket_info_reversal = {
        "open_trades": [
            {
                "trade_id": "giveback-1",
                "units": 1100,
                "open_time": "2026-02-12T01:59:20+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "client_id": "qr-giveback-1",
                "unrealized_pl_pips": -0.2,
                "entry_thesis": {
                    "strategy_tag": "scalp_ping_5s_live",
                    "policy_generation": "2026-02-12-giveback-v1",
                },
            }
        ]
    }
    closed_second = await worker._enforce_new_entry_time_stop(
        pocket_info=pocket_info_reversal,
        now_utc=second_now,
        logger=worker.LOG,
    )
    assert closed_second == 1
    assert len(calls) == 1
    assert calls[0][0] == "giveback-1"
    assert calls[0][4] == "giveback_lock"


@pytest.mark.asyncio
async def test_apply_profit_bank_release_respects_exclusions(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "PROFIT_BANK_ENABLED", True)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_MAX_ACTIONS", 1)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_COOLDOWN_SEC", 0.0)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_MIN_TARGET_LOSS_JPY", 100.0)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_MAX_TARGET_LOSS_JPY", 5000.0)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_TARGET_MIN_HOLD_SEC", 60.0)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_TARGET_ORDER", "largest_loss")
    monkeypatch.setattr(worker.config, "PROFIT_BANK_TARGET_REQUIRE_OPEN_BEFORE_START", False)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_EXCLUDE_TRADE_IDS", ("skip-trade",))
    monkeypatch.setattr(worker.config, "PROFIT_BANK_EXCLUDE_CLIENT_IDS", ("skip-client",))
    monkeypatch.setattr(worker.config, "PROFIT_BANK_REASON", "profit_bank_release")
    monkeypatch.setattr(worker.config, "STRATEGY_TAG", "scalp_ping_5s_live")
    monkeypatch.setattr(worker.config, "POCKET", "scalp_fast")
    monkeypatch.setattr(worker, "_PROFIT_BANK_LAST_CLOSE_MONO", 0.0)

    monkeypatch.setattr(
        worker,
        "_load_profit_bank_stats",
        lambda **_kwargs: (1200.0, 0.0, 1200.0),
    )
    monkeypatch.setattr(
        worker,
        "_profit_bank_available_budget_jpy",
        lambda **_kwargs: 400.0,
    )

    calls: list[tuple[str, int, str | None, bool, str | None, str | None]] = []

    async def _fake_close_trade(
        trade_id: str,
        units: int | None = None,
        client_order_id: str | None = None,
        allow_negative: bool = False,
        exit_reason: str | None = None,
        env_prefix: str | None = None,
    ) -> bool:
        calls.append((trade_id, int(units or 0), client_order_id, allow_negative, exit_reason, env_prefix))
        return True

    monkeypatch.setattr(worker, "close_trade", _fake_close_trade)

    now_utc = datetime.datetime(2026, 2, 12, 6, 0, 0, tzinfo=datetime.timezone.utc)
    pocket_info = {
        "open_trades": [
            {
                "trade_id": "skip-trade",
                "client_id": "qr-skip",
                "units": 1200,
                "open_time": "2026-02-12T05:40:00+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "unrealized_pl": -320.0,
            },
            {
                "trade_id": "skip-client-trade",
                "client_id": "skip-client",
                "units": 1000,
                "open_time": "2026-02-12T05:40:00+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "unrealized_pl": -220.0,
            },
            {
                "trade_id": "eligible-trade",
                "client_id": "qr-eligible",
                "units": 900,
                "open_time": "2026-02-12T05:40:00+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "unrealized_pl": -210.0,
            },
        ]
    }

    closed = await worker._apply_profit_bank_release(
        pocket_info=pocket_info,
        now_utc=now_utc,
        logger=worker.LOG,
        protected_trade_ids={"protected-trade"},
    )
    assert closed == 1
    assert len(calls) == 1
    assert calls[0][0] == "eligible-trade"
    assert calls[0][1] == -900
    assert calls[0][2] == "qr-eligible"
    assert calls[0][3] is True
    assert calls[0][4] == "profit_bank_release"


@pytest.mark.asyncio
async def test_apply_profit_bank_release_skips_when_budget_is_too_small(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "PROFIT_BANK_ENABLED", True)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_MAX_ACTIONS", 1)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_COOLDOWN_SEC", 0.0)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_MIN_TARGET_LOSS_JPY", 100.0)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_MAX_TARGET_LOSS_JPY", 5000.0)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_TARGET_MIN_HOLD_SEC", 60.0)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_TARGET_ORDER", "largest_loss")
    monkeypatch.setattr(worker.config, "PROFIT_BANK_TARGET_REQUIRE_OPEN_BEFORE_START", False)
    monkeypatch.setattr(worker.config, "PROFIT_BANK_EXCLUDE_TRADE_IDS", ())
    monkeypatch.setattr(worker.config, "PROFIT_BANK_EXCLUDE_CLIENT_IDS", ())
    monkeypatch.setattr(worker.config, "PROFIT_BANK_REASON", "profit_bank_release")
    monkeypatch.setattr(worker.config, "STRATEGY_TAG", "scalp_ping_5s_live")
    monkeypatch.setattr(worker.config, "POCKET", "scalp_fast")
    monkeypatch.setattr(worker, "_PROFIT_BANK_LAST_CLOSE_MONO", 0.0)

    monkeypatch.setattr(
        worker,
        "_load_profit_bank_stats",
        lambda **_kwargs: (700.0, 0.0, 700.0),
    )
    monkeypatch.setattr(
        worker,
        "_profit_bank_available_budget_jpy",
        lambda **_kwargs: 80.0,
    )

    calls: list[str] = []

    async def _fake_close_trade(
        trade_id: str,
        units: int | None = None,
        client_order_id: str | None = None,
        allow_negative: bool = False,
        exit_reason: str | None = None,
        env_prefix: str | None = None,
    ) -> bool:
        calls.append(trade_id)
        return True

    monkeypatch.setattr(worker, "close_trade", _fake_close_trade)

    now_utc = datetime.datetime(2026, 2, 12, 6, 5, 0, tzinfo=datetime.timezone.utc)
    pocket_info = {
        "open_trades": [
            {
                "trade_id": "loss-1",
                "client_id": "qr-loss-1",
                "units": 1100,
                "open_time": "2026-02-12T05:40:00+00:00",
                "strategy_tag": "scalp_ping_5s_live",
                "unrealized_pl": -240.0,
            }
        ]
    }

    closed = await worker._apply_profit_bank_release(
        pocket_info=pocket_info,
        now_utc=now_utc,
        logger=worker.LOG,
    )
    assert closed == 0
    assert calls == []


def test_tick_density_uses_window_cutoff() -> None:
    from workers.scalp_ping_5s import worker

    rows = [
        _tick(100.0, 150.000, 150.002, 150.001),
        _tick(101.0, 150.001, 150.003, 150.002),
        _tick(102.0, 150.002, 150.004, 150.003),
        _tick(103.0, 150.003, 150.005, 150.004),
    ]
    density = worker._tick_density(rows, 2.0)
    assert density == pytest.approx(1.5, abs=1e-9)


def test_tick_span_ratio_uses_first_and_last_epoch() -> None:
    from workers.scalp_ping_5s import worker

    rows = [
        _tick(100.0, 150.000, 150.002, 150.001),
        _tick(101.0, 150.001, 150.003, 150.002),
        _tick(103.0, 150.003, 150.005, 150.004),
    ]
    span_ratio = worker._tick_span_ratio(rows, 5.0)
    assert span_ratio == pytest.approx(0.6, abs=1e-9)


@pytest.mark.asyncio
async def test_maybe_keepalive_snapshot_fetches_when_stale(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "SNAPSHOT_FALLBACK_ENABLED", True)
    monkeypatch.setattr(worker.config, "SNAPSHOT_KEEPALIVE_ENABLED", True)
    monkeypatch.setattr(worker.config, "SNAPSHOT_KEEPALIVE_MIN_INTERVAL_SEC", 0.8)
    monkeypatch.setattr(worker.config, "SNAPSHOT_KEEPALIVE_MAX_AGE_MS", 2500.0)
    monkeypatch.setattr(worker.config, "SNAPSHOT_KEEPALIVE_MIN_DENSITY", 0.0)
    monkeypatch.setattr(worker.config, "SNAPSHOT_KEEPALIVE_MIN_SPAN_RATIO", 0.0)
    monkeypatch.setattr(worker.config, "ENTRY_QUALITY_WINDOW_SEC", 30.0)

    snapshot_calls: list[bool] = []

    async def _fake_snapshot(_logger) -> bool:
        snapshot_calls.append(True)
        return True

    monkeypatch.setattr(worker, "_fetch_price_snapshot", _fake_snapshot)

    last_fetch, stats = await worker._maybe_keepalive_snapshot(
        now_mono=10.0,
        last_snapshot_fetch=0.0,
        rows=[_tick(100.0, 150.000, 150.002, 150.001)],
        latest_tick_age_ms=5300.0,
        logger=worker.LOG,
    )

    assert snapshot_calls == [True]
    assert last_fetch == pytest.approx(10.0)
    assert stats is not None
    assert stats["reason"] == "stale"
    assert float(stats["age_ms"]) == pytest.approx(5300.0)


@pytest.mark.asyncio
async def test_maybe_keepalive_snapshot_skips_when_microstructure_is_healthy(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "SNAPSHOT_FALLBACK_ENABLED", True)
    monkeypatch.setattr(worker.config, "SNAPSHOT_KEEPALIVE_ENABLED", True)
    monkeypatch.setattr(worker.config, "SNAPSHOT_KEEPALIVE_MIN_INTERVAL_SEC", 0.8)
    monkeypatch.setattr(worker.config, "SNAPSHOT_KEEPALIVE_MAX_AGE_MS", 2500.0)
    monkeypatch.setattr(worker.config, "SNAPSHOT_KEEPALIVE_MIN_DENSITY", 0.8)
    monkeypatch.setattr(worker.config, "SNAPSHOT_KEEPALIVE_MIN_SPAN_RATIO", 0.6)
    monkeypatch.setattr(worker.config, "ENTRY_QUALITY_WINDOW_SEC", 30.0)

    snapshot_calls: list[bool] = []

    async def _fake_snapshot(_logger) -> bool:
        snapshot_calls.append(True)
        return True

    monkeypatch.setattr(worker, "_fetch_price_snapshot", _fake_snapshot)

    rows = [
        _tick(70.0, 150.000, 150.002, 150.001),
        _tick(80.0, 150.001, 150.003, 150.002),
        _tick(90.0, 150.002, 150.004, 150.003),
        _tick(100.0, 150.003, 150.005, 150.004),
    ]
    last_fetch, stats = await worker._maybe_keepalive_snapshot(
        now_mono=10.0,
        last_snapshot_fetch=0.0,
        rows=rows,
        latest_tick_age_ms=800.0,
        logger=worker.LOG,
    )

    assert snapshot_calls == []
    assert last_fetch == pytest.approx(0.0)
    assert stats is None


@pytest.mark.asyncio
async def test_maybe_topup_micro_density_fetches_snapshot_when_below_target(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "SNAPSHOT_FALLBACK_ENABLED", True)
    monkeypatch.setattr(worker.config, "SNAPSHOT_TOPUP_ENABLED", True)
    monkeypatch.setattr(worker.config, "SNAPSHOT_TOPUP_TARGET_DENSITY", 1.4)
    monkeypatch.setattr(worker.config, "SNAPSHOT_TOPUP_MIN_INTERVAL_SEC", 1.0)
    monkeypatch.setattr(worker.config, "ENTRY_QUALITY_WINDOW_SEC", 30.0)

    density_values = iter([0.6, 1.5])
    monkeypatch.setattr(worker, "_tick_density_over_window", lambda _sec: next(density_values))

    snapshot_calls: list[bool] = []

    async def _fake_snapshot(_logger) -> bool:
        snapshot_calls.append(True)
        return True

    monkeypatch.setattr(worker, "_fetch_price_snapshot", _fake_snapshot)

    last_fetch, stats = await worker._maybe_topup_micro_density(
        now_mono=10.0,
        last_snapshot_fetch=0.0,
        logger=worker.LOG,
    )

    assert snapshot_calls == [True]
    assert last_fetch == pytest.approx(10.0)
    assert stats is not None
    assert stats["before"] == pytest.approx(0.6)
    assert stats["after"] == pytest.approx(1.5)
    assert stats["target"] == pytest.approx(1.4)


@pytest.mark.asyncio
async def test_maybe_topup_micro_density_skips_when_density_is_enough(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "SNAPSHOT_FALLBACK_ENABLED", True)
    monkeypatch.setattr(worker.config, "SNAPSHOT_TOPUP_ENABLED", True)
    monkeypatch.setattr(worker.config, "SNAPSHOT_TOPUP_TARGET_DENSITY", 1.4)
    monkeypatch.setattr(worker.config, "SNAPSHOT_TOPUP_MIN_INTERVAL_SEC", 1.0)
    monkeypatch.setattr(worker.config, "ENTRY_QUALITY_WINDOW_SEC", 30.0)
    monkeypatch.setattr(worker, "_tick_density_over_window", lambda _sec: 1.6)

    snapshot_calls: list[bool] = []

    async def _fake_snapshot(_logger) -> bool:
        snapshot_calls.append(True)
        return True

    monkeypatch.setattr(worker, "_fetch_price_snapshot", _fake_snapshot)

    last_fetch, stats = await worker._maybe_topup_micro_density(
        now_mono=10.0,
        last_snapshot_fetch=0.0,
        logger=worker.LOG,
    )

    assert snapshot_calls == []
    assert last_fetch == pytest.approx(0.0)
    assert stats is None


def test_fetch_price_snapshot_normalizes_stale_quote_timestamp(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker, "_OANDA_TOKEN", "token")
    monkeypatch.setattr(worker, "_OANDA_ACCOUNT", "account")
    monkeypatch.setattr(worker.config, "MAX_TICK_AGE_MS", 2500.0)

    stale_quote_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=5)
    monkeypatch.setattr(worker, "_parse_time", lambda _raw: stale_quote_ts)

    class _DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "prices": [
                    {
                        "bids": [{"price": "150.000"}],
                        "asks": [{"price": "150.010"}],
                        "time": "2026-02-12T02:00:00Z",
                    }
                ]
            }

    class _DummyClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, *args, **kwargs):
            return _DummyResponse()

    monkeypatch.setattr(worker.httpx, "AsyncClient", _DummyClient)

    captured: dict[str, object] = {}
    monkeypatch.setattr(worker.spread_monitor, "update_from_tick", lambda tick: captured.setdefault("tick", tick))
    monkeypatch.setattr(worker.tick_window, "record", lambda tick: captured.setdefault("recorded", tick))

    ok = asyncio.run(worker._fetch_price_snapshot(worker.LOG))
    assert ok is True
    recorded = captured.get("recorded")
    assert recorded is not None
    recorded_time = getattr(recorded, "time")
    if recorded_time.tzinfo is None:
        recorded_time = recorded_time.replace(tzinfo=datetime.timezone.utc)
    age_sec = (datetime.datetime.now(datetime.timezone.utc) - recorded_time).total_seconds()
    assert age_sec < 1.5

from __future__ import annotations

import asyncio
import datetime
import os

import pytest

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")


def _tick(epoch: float, bid: float, ask: float, mid: float) -> dict[str, float]:
    return {"epoch": epoch, "bid": bid, "ask": ask, "mid": mid}


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
        )
        is True
    )
    assert (
        worker._allow_signal_when_max_active(
            side="long",
            active_total=10,
            active_long=10,
            active_short=0,
        )
        is False
    )
    assert (
        worker._allow_signal_when_max_active(
            side="long",
            active_total=10,
            active_long=5,
            active_short=5,
        )
        is False
    )
    assert (
        worker._allow_signal_when_max_active(
            side="long",
            active_total=9,
            active_long=9,
            active_short=0,
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
        )
        is False
    )


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

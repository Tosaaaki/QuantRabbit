from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import importlib
import sys
from types import SimpleNamespace

import os

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")
os.environ.setdefault("oanda_token", "dummy")
os.environ.setdefault("oanda_account_id", "dummy")
os.environ.setdefault("oanda_practice", "true")

from scripts import replay_exit_workers as replay


def _tick(ts: datetime, *, bid: float, ask: float) -> replay.TickRow:
    return replay.TickRow(
        ts=ts,
        epoch=ts.timestamp(),
        bid=bid,
        ask=ask,
        mid=(bid + ask) * 0.5,
    )


def test_ping5s_force_exit_timeout_uses_directional_hold() -> None:
    cfg = SimpleNamespace(
        FORCE_EXIT_ENABLED=True,
        FORCE_EXIT_MAX_ACTIONS=2,
        FORCE_EXIT_MAX_HOLD_SEC=95.0,
        SHORT_FORCE_EXIT_MAX_HOLD_SEC=70.0,
    )
    assert replay._ping5s_force_exit_timeout_sec(cfg, side="long") == 95.0
    assert replay._ping5s_force_exit_timeout_sec(cfg, side="short") == 70.0


def test_simbroker_timeout_closes_trade_with_time_stop() -> None:
    broker = replay.SimBroker(latency_ms=0.0, fill_mode="lko")
    opened = datetime(2026, 2, 22, 0, 0, 0, tzinfo=timezone.utc)
    first_tick = _tick(opened, bid=150.000, ask=150.002)
    broker.set_last_tick(first_tick)
    trade = broker.open_trade(
        pocket="scalp_fast",
        strategy_tag="scalp_ping_5s_c_live",
        direction="long",
        entry_price=first_tick.ask,
        entry_time=opened,
        tp_pips=0.0,
        sl_pips=0.0,
        timeout_sec=1.0,
        units=10000,
        source="scalp_replay",
        entry_thesis={"timeout_sec": 1.0, "force_exit_reason": "time_stop"},
    )
    assert trade.get("trade_id")

    timeout_tick = _tick(opened + timedelta(seconds=1.1), bid=150.001, ask=150.003)
    broker.set_last_tick(timeout_tick)
    broker.check_timeouts(timeout_tick)

    assert not broker.open_trades
    assert len(broker.closed_trades) == 1
    assert broker.closed_trades[0]["reason"] == "time_stop"


def test_simbroker_timeout_force_closes_when_profit_guard_rejects() -> None:
    broker = replay.SimBroker(latency_ms=0.0, fill_mode="lko")
    broker._close_guard_params = lambda _pocket, _tag: {  # type: ignore[assignment]
        "min_profit_pips": 0.5,
        "ratio": None,
        "ratio_reasons": set(),
        "tp_min": 0.0,
    }
    opened = datetime(2026, 2, 22, 0, 0, 0, tzinfo=timezone.utc)
    entry_tick = _tick(opened, bid=150.000, ask=150.002)
    broker.set_last_tick(entry_tick)
    broker.open_trade(
        pocket="scalp_fast",
        strategy_tag="scalp_ping_5s_c_live",
        direction="long",
        entry_price=entry_tick.ask,
        entry_time=opened,
        tp_pips=0.0,
        sl_pips=0.0,
        timeout_sec=1.0,
        units=10000,
        source="scalp_replay",
        entry_thesis={"timeout_sec": 1.0, "force_exit_reason": "time_stop"},
    )

    # +0.1 pip: guard would reject ordinary close, so timeout fallback must force close.
    timeout_tick = _tick(opened + timedelta(seconds=1.2), bid=150.003, ask=150.005)
    broker.set_last_tick(timeout_tick)
    broker.check_timeouts(timeout_tick)

    assert not broker.open_trades
    assert len(broker.closed_trades) == 1
    assert broker.closed_trades[0]["reason"] == "time_stop"


def test_ping5s_entry_units_intent_uses_config_bounds() -> None:
    worker = SimpleNamespace(_confidence_scale=lambda conf: 0.45)
    cfg = SimpleNamespace(BASE_ENTRY_UNITS=800, MIN_UNITS=100, MAX_UNITS=2200)
    assert replay._ping5s_entry_units_intent(worker, cfg, confidence=75) == 360

    worker_hi = SimpleNamespace(_confidence_scale=lambda conf: 5.0)
    assert replay._ping5s_entry_units_intent(worker_hi, cfg, confidence=99) == 2200


def test_signal_signed_units_prefers_entry_units_intent() -> None:
    signal = {"entry_units_intent": 850}
    assert replay._signal_signed_units(signal, direction="long") == 850
    assert replay._signal_signed_units(signal, direction="short") == -850

    fallback_signal: dict[str, float] = {}
    assert replay._signal_signed_units(fallback_signal, direction="long") == 10000
    assert replay._signal_signed_units(fallback_signal, direction="short") == -10000


def test_ping5s_variant_d_module_mapping(monkeypatch) -> None:
    monkeypatch.setenv("SCALP_REPLAY_PING_VARIANT", "D")
    reloaded = importlib.reload(replay)
    try:
        assert reloaded._PING5S_VARIANT == "D"
        assert reloaded._PING5S_MODE == "scalp_ping_5s_d"
        assert reloaded._PING5S_FALLBACK_TAG == "scalp_ping_5s_d_live"
        assert reloaded._PING5S_ALT_PREFIX == "SCALP_PING_5S_D"
        assert reloaded._PING5S_WORKER_MODULE == "workers.scalp_ping_5s_d.worker"
        assert reloaded._PING5S_EXIT_MODULE == "workers.scalp_ping_5s_d.exit_worker"
    finally:
        monkeypatch.delenv("SCALP_REPLAY_PING_VARIANT", raising=False)
        importlib.reload(reloaded)


def test_patch_exit_module_patches_delegate_module(monkeypatch) -> None:
    broker = replay.SimBroker(latency_ms=0.0, fill_mode="lko")

    delegate = SimpleNamespace(
        __name__="dummy_delegate_mod",
        PositionManager=None,
        close_trade=None,
        set_trade_protections=lambda *args, **kwargs: False,
    )

    class _DummyWorker:
        pass

    _DummyWorker.__module__ = "dummy_delegate_mod"
    wrapper = SimpleNamespace(
        __name__="dummy_wrapper_mod",
        PositionManager=None,
        close_trade=None,
        RangeFaderExitWorker=_DummyWorker,
    )
    monkeypatch.setitem(sys.modules, "dummy_delegate_mod", delegate)

    replay._patch_exit_module(wrapper, broker)

    assert callable(wrapper.close_trade)
    assert callable(delegate.close_trade)

    opened = datetime(2026, 2, 22, 0, 0, 0, tzinfo=timezone.utc)
    tick = _tick(opened, bid=150.000, ask=150.002)
    broker.set_last_tick(tick)
    trade_a = broker.open_trade(
        pocket="scalp_fast",
        strategy_tag="scalp_ping_5s_d_live",
        direction="long",
        entry_price=tick.ask,
        entry_time=opened,
        tp_pips=0.0,
        sl_pips=0.0,
        timeout_sec=None,
        units=1000,
        source="test",
    )
    trade_b = broker.open_trade(
        pocket="scalp_fast",
        strategy_tag="scalp_ping_5s_d_live",
        direction="long",
        entry_price=tick.ask,
        entry_time=opened,
        tp_pips=0.0,
        sl_pips=0.0,
        timeout_sec=None,
        units=1000,
        source="test",
    )

    assert asyncio.run(wrapper.close_trade(str(trade_a["trade_id"]), exit_reason="wrapper"))
    assert asyncio.run(delegate.close_trade(str(trade_b["trade_id"]), exit_reason="delegate"))


def test_patch_module_clock_uses_sim_clock() -> None:
    module = SimpleNamespace(
        time=SimpleNamespace(
            time=lambda: 1.0,
            monotonic=lambda: 2.0,
        )
    )
    sim_clock = replay.SimClock()
    sim_clock.now = 12345.67
    replay._patch_module_clock(module, sim_clock)
    assert module.time.time() == 12345.67
    assert module.time.monotonic() == 12345.67


def test_patch_ping_runtime_clock_patches_worker_and_exit(monkeypatch) -> None:
    worker = SimpleNamespace(
        time=SimpleNamespace(
            time=lambda: 1.0,
            monotonic=lambda: 2.0,
        )
    )
    exit_mod = SimpleNamespace(
        time=SimpleNamespace(
            time=lambda: 3.0,
            monotonic=lambda: 4.0,
        )
    )
    monkeypatch.setattr(
        replay,
        "_load_ping5s_runtime",
        lambda: (worker, SimpleNamespace(), exit_mod),
    )
    sim_clock = replay.SimClock()
    sim_clock.now = 9876.5
    replay._patch_ping_runtime_clock(sim_clock)
    assert worker.time.time() == 9876.5
    assert worker.time.monotonic() == 9876.5
    assert exit_mod.time.time() == 9876.5
    assert exit_mod.time.monotonic() == 9876.5


def test_replay_hour_filters_fallback_to_variant_env(monkeypatch) -> None:
    monkeypatch.setenv("SCALP_REPLAY_PING_VARIANT", "D")
    monkeypatch.delenv("SCALP_REPLAY_ALLOW_JST_HOURS", raising=False)
    monkeypatch.delenv("SCALP_REPLAY_BLOCK_JST_HOURS", raising=False)
    monkeypatch.setenv("SCALP_PING_5S_D_ALLOW_HOURS_JST", "1,10")
    monkeypatch.setenv("SCALP_PING_5S_D_BLOCK_HOURS_JST", "3,5")

    reloaded = importlib.reload(replay)
    try:
        assert reloaded._replay_or_strategy_hours(
            "SCALP_REPLAY_ALLOW_JST_HOURS",
            strategy_suffix="ALLOW_HOURS_JST",
        ) == {1, 10}
        assert reloaded._replay_or_strategy_hours(
            "SCALP_REPLAY_BLOCK_JST_HOURS",
            strategy_suffix="BLOCK_HOURS_JST",
        ) == {3, 5}
    finally:
        monkeypatch.delenv("SCALP_REPLAY_PING_VARIANT", raising=False)
        monkeypatch.delenv("SCALP_PING_5S_D_ALLOW_HOURS_JST", raising=False)
        monkeypatch.delenv("SCALP_PING_5S_D_BLOCK_HOURS_JST", raising=False)
        importlib.reload(reloaded)


def test_ping5s_signal_respects_post_regime_side_filter(monkeypatch) -> None:
    class _FakeWorker:
        def _build_tick_signal(self, rows, spread_pips):
            return (
                SimpleNamespace(
                    side="long",
                    confidence=80,
                    spread_pips=spread_pips,
                    momentum_pips=1.0,
                    range_pips=2.0,
                    instant_range_pips=1.0,
                    mode="momentum",
                ),
                "ok",
            )

        def _build_mtf_regime(self, _factors):
            return SimpleNamespace(mode="reversion", side="neutral", heat_score=0.2)

        def _apply_mtf_regime(self, signal, regime):
            _ = regime
            return (
                SimpleNamespace(
                    side="short",
                    confidence=signal.confidence,
                    spread_pips=signal.spread_pips,
                    momentum_pips=signal.momentum_pips,
                    range_pips=signal.range_pips,
                    instant_range_pips=signal.instant_range_pips,
                    mode=f"{signal.mode}_mtf_fade",
                ),
                1.0,
                "mtf_reversion_fade",
            )

        def _load_tp_timing_profile(self, *_args, **_kwargs):
            return SimpleNamespace()

        def _compute_targets(self, **_kwargs):
            return 1.0, 1.0

        def _confidence_scale(self, _conf):
            return 1.0

    fake_cfg = SimpleNamespace(
        WINDOW_SEC=5.0,
        TP_ENABLED=True,
        FORCE_EXIT_ENABLED=True,
        FORCE_EXIT_MAX_ACTIONS=2,
        FORCE_EXIT_MAX_HOLD_SEC=45.0,
        SHORT_FORCE_EXIT_MAX_HOLD_SEC=35.0,
        FORCE_EXIT_REASON="time_stop",
        BASE_ENTRY_UNITS=1000,
        MIN_UNITS=100,
        MAX_UNITS=3000,
        STRATEGY_TAG="scalp_ping_5s_c_live",
        POCKET="scalp",
        SIDE_FILTER="long",
    )

    monkeypatch.setattr(replay, "_load_ping5s_runtime", lambda: (_FakeWorker(), fake_cfg, SimpleNamespace()))
    monkeypatch.setattr(replay.tick_window, "recent_ticks", lambda _sec: [{"mid": 150.0}])
    monkeypatch.setattr(replay.spread_monitor, "is_blocked", lambda: (False, 0.2, None, None))

    out = replay._signal_scalp_ping_5s_b({}, {}, {}, None, datetime.now(timezone.utc))
    assert out is None


def test_candidate_regime_route_matrix() -> None:
    assert replay._candidate_regime_route("Trend", "Trend") == "trend"
    assert replay._candidate_regime_route("Breakout", "Trend") == "trend"
    assert replay._candidate_regime_route("Range", "Breakout") == "breakout"
    assert replay._candidate_regime_route("Mixed", "Range") == "range"
    assert replay._candidate_regime_route("Trend", "Mixed") == "mixed"
    assert replay._candidate_regime_route(None, None) == "unknown"


def test_close_trade_record_copies_regime_fields_from_entry_thesis() -> None:
    broker = replay.SimBroker(latency_ms=0.0, fill_mode="lko")
    opened = datetime(2026, 2, 24, 0, 0, 0, tzinfo=timezone.utc)
    tick_open = _tick(opened, bid=150.000, ask=150.002)
    broker.set_last_tick(tick_open)
    broker.open_trade(
        pocket="scalp_fast",
        strategy_tag="scalp_ping_5s_c_live",
        direction="long",
        entry_price=tick_open.ask,
        entry_time=opened,
        tp_pips=0.0,
        sl_pips=0.0,
        timeout_sec=1.0,
        units=1000,
        source="scalp_replay",
        entry_thesis={
            "macro_regime": "Trend",
            "micro_regime": "Range",
            "regime_route": "mixed",
        },
    )

    tick_close = _tick(opened + timedelta(seconds=2), bid=150.001, ask=150.003)
    broker.set_last_tick(tick_close)
    broker.check_timeouts(tick_close)
    assert len(broker.closed_trades) == 1
    closed = broker.closed_trades[0]
    assert closed["macro_regime"] == "trend"
    assert closed["micro_regime"] == "range"
    assert closed["regime_route"] == "mixed"

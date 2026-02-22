from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib
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

"""Unit tests for the scalp exit manager logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sys
import types

import pytest

# Basic stub helpers ---------------------------------------------------------


def _install_stub(module_name: str, module: types.ModuleType) -> None:
    """Register a stub module if the real one is not already loaded."""
    if module_name not in sys.modules:
        sys.modules[module_name] = module

# Provide lightweight stubs so importing workers.scalp_exit.worker does not require
# real OANDA dependencies (oandapyV20 / env secrets) during unit tests.
fake_order_mgr = types.ModuleType("execution.order_manager")

async def _fake_close_trade(trade_id: str) -> bool:  # pragma: no cover - shim
    return True

fake_order_mgr.close_trade = _fake_close_trade
_install_stub("execution.order_manager", fake_order_mgr)

fake_pos_mgr = types.ModuleType("execution.position_manager")

class _DummyPositionManager:  # pragma: no cover - shim
    def __init__(self):
        pass

    def get_open_positions(self):
        return {}

    def close(self):
        return None

fake_pos_mgr.PositionManager = _DummyPositionManager
_install_stub("execution.position_manager", fake_pos_mgr)

fake_factor_cache = types.ModuleType("indicators.factor_cache")

def _fake_all_factors():
    return {"M1": {}, "M5": {}, "H1": {}}

fake_factor_cache.all_factors = _fake_all_factors
_install_stub("indicators.factor_cache", fake_factor_cache)

fake_tick_window = types.ModuleType("market_data.tick_window")
fake_tick_window.recent_ticks = lambda seconds=1.0, limit=1: []  # pragma: no cover
_install_stub("market_data.tick_window", fake_tick_window)

fake_policy_bus = types.ModuleType("analysis.policy_bus")
fake_policy_bus.latest = lambda: None  # pragma: no cover
fake_policy_bus.publish = lambda data: None  # pragma: no cover
_install_stub("analysis.policy_bus", fake_policy_bus)

from workers.scalp_exit import config as exit_config
from workers.scalp_exit.worker import ScalpExitManager


def _build_trade(
    *,
    trade_id: str = "t1",
    units: int = 1_000,
    price: float = 153.10,
    opened_ago: timedelta = timedelta(seconds=5),
    strategy_tag: str = "pullback_scalp",
) -> dict:
    opened_at = datetime.now(timezone.utc) - opened_ago
    return {
        "trade_id": trade_id,
        "units": units,
        "price": price,
        "open_time": opened_at.isoformat().replace("+00:00", "Z"),
        "entry_thesis": {"strategy_tag": strategy_tag},
    }


def _default_factors() -> dict:
    return {
        "rsi": 50.0,
        "atr": 0.00012,  # 1.2 pips
        "candles": [{"close": 153.10} for _ in range(30)],
    }


@pytest.fixture(autouse=True)
def _tweak_config(monkeypatch):
    """Relax thresholds so tests can run quickly."""

    monkeypatch.setattr(exit_config, "HARD_STOP_PIPS", 5.0, raising=False)
    monkeypatch.setattr(exit_config, "MIN_NEGATIVE_HOLD_SEC", 30.0, raising=False)
    monkeypatch.setattr(exit_config, "NEGATIVE_HOLD_TIMEOUT_SEC", 90.0, raising=False)
    monkeypatch.setattr(exit_config, "ATR_MOMENTUM_MIN_PIPS", 0.5, raising=False)
    monkeypatch.setattr(exit_config, "ADX_STRONG_THRESHOLD", 25.0, raising=False)
    monkeypatch.setattr(exit_config, "BASE_PROFIT_PIPS", 2.0, raising=False)
    monkeypatch.setattr(exit_config, "LOCK_AT_PROFIT_PIPS", 2.4, raising=False)
    monkeypatch.setattr(exit_config, "LOCK_BUFFER_PIPS", 0.4, raising=False)
    monkeypatch.setattr(exit_config, "TRAIL_START_PIPS", 3.0, raising=False)
    monkeypatch.setattr(exit_config, "TRAIL_BACKOFF_PIPS", 0.6, raising=False)
    monkeypatch.setattr(exit_config, "HARD_PROFIT_PIPS", 6.0, raising=False)


def test_negative_trade_is_held_when_trend_and_vol_are_soft(monkeypatch):
    now_price = 153.08  # -2 pips for a long
    monkeypatch.setattr("workers.scalp_exit.worker._latest_mid", lambda: now_price)

    manager = ScalpExitManager()
    manager.update_policy({"exit_profile": {"allow_negative_exit": True}})

    trade = _build_trade(price=153.10, opened_ago=timedelta(seconds=10))
    m1 = _default_factors()
    context = {
        "M5": {"atr": 0.00012, "adx": 14.0},
        "H1": {"atr": 0.00020, "adx": 18.0},
    }

    reason = manager.evaluate(trade, m1_factors=m1, context_factors=context, now=datetime.now(timezone.utc))
    assert reason is None  # should keep the trade open


def test_negative_trade_times_out_when_strong_trend_and_long_hold(monkeypatch):
    now_price = 153.07  # -3 pips for a long trade
    monkeypatch.setattr("workers.scalp_exit.worker._latest_mid", lambda: now_price)

    manager = ScalpExitManager()
    manager.update_policy({"exit_profile": {"allow_negative_exit": True}})

    trade = _build_trade(price=153.10, opened_ago=timedelta(seconds=120))
    m1 = _default_factors()
    context = {
        "M5": {"atr": 0.00090, "adx": 32.0},  # elevated vol + strong trend
        "H1": {"atr": 0.00080, "adx": 30.0},
    }

    reason = manager.evaluate(trade, m1_factors=m1, context_factors=context, now=datetime.now(timezone.utc))
    assert reason == "scalp_time_stop"


def test_lock_release_triggers_when_profit_retraces(monkeypatch):
    entry = 153.10
    # First call: run price up so lock_floor is set
    monkeypatch.setattr("workers.scalp_exit.worker._latest_mid", lambda: entry + 0.03)  # +3 pips

    manager = ScalpExitManager()
    manager.update_policy(
        {
            "exit_profile": {"allow_negative_exit": True},
            "be_profile": {"trigger_pips": 2.5, "min_lock_pips": 0.5},
        }
    )

    trade = _build_trade(price=entry, opened_ago=timedelta(seconds=200))
    m1 = _default_factors()
    context = {
        "M5": {"atr": 0.00030, "adx": 20.0},
        "H1": {"atr": 0.00045, "adx": 22.0},
    }

    first = manager.evaluate(trade, m1_factors=m1, context_factors=context, now=datetime.now(timezone.utc))
    assert first is None

    # Now drop below lock floor and ensure release
    monkeypatch.setattr("workers.scalp_exit.worker._latest_mid", lambda: entry + 0.015)  # +1.5 pips
    second = manager.evaluate(trade, m1_factors=m1, context_factors=context, now=datetime.now(timezone.utc))
    assert second == "scalp_lock_release"

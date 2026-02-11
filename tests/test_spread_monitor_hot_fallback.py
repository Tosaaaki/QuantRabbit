from __future__ import annotations

import os
from datetime import datetime, timezone
from types import SimpleNamespace


os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")


def _reset_state(spread_monitor) -> None:
    spread_monitor._snapshot = None
    spread_monitor._history.clear()
    spread_monitor._baseline_history.clear()
    spread_monitor._hot_history.clear()
    spread_monitor._blocked_until = 0.0
    spread_monitor._blocked_reason = ""
    spread_monitor._last_logged_blocked = False
    spread_monitor._stale_since = None
    spread_monitor._fallback_state_cache = None
    spread_monitor._fallback_state_cached_at = 0.0


def test_is_blocked_uses_tick_cache_hot_spread(monkeypatch):
    from market_data import spread_monitor

    _reset_state(spread_monitor)
    monkeypatch.setattr(spread_monitor, "DISABLE_SPREAD_GUARD", False)
    monkeypatch.setattr(spread_monitor, "TICK_CACHE_FALLBACK_ENABLED", True)
    monkeypatch.setattr(spread_monitor, "HOT_TRIGGER_ENABLED", True)
    monkeypatch.setattr(spread_monitor, "HOT_TRIGGER_PIPS", 1.0)
    monkeypatch.setattr(spread_monitor, "HOT_MIN_SAMPLES", 2)
    monkeypatch.setattr(spread_monitor, "HOT_COOLDOWN_SECONDS", 3.0)
    monkeypatch.setattr(spread_monitor.time, "time", lambda: 100.0)
    monkeypatch.setattr(spread_monitor.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(
        spread_monitor.tick_window,
        "recent_ticks",
        lambda seconds=0.0, limit=None: [
            {"epoch": 99.6, "bid": 150.000, "ask": 150.011, "mid": 150.0055},
            {"epoch": 99.8, "bid": 150.001, "ask": 150.012, "mid": 150.0065},
            {"epoch": 100.0, "bid": 150.002, "ask": 150.013, "mid": 150.0075},
        ],
    )

    blocked, remain, state, reason = spread_monitor.is_blocked()
    assert blocked is True
    assert remain >= 2
    assert state is not None
    assert state.get("source") == "tick_cache"
    assert int(state.get("hot_samples") or 0) >= 2
    assert "hot_spread_now" in reason


def test_is_blocked_allows_tight_tick_cache_spread(monkeypatch):
    from market_data import spread_monitor

    _reset_state(spread_monitor)
    monkeypatch.setattr(spread_monitor, "DISABLE_SPREAD_GUARD", False)
    monkeypatch.setattr(spread_monitor, "TICK_CACHE_FALLBACK_ENABLED", True)
    monkeypatch.setattr(spread_monitor, "HOT_TRIGGER_ENABLED", True)
    monkeypatch.setattr(spread_monitor, "HOT_TRIGGER_PIPS", 1.0)
    monkeypatch.setattr(spread_monitor, "HOT_MIN_SAMPLES", 2)
    monkeypatch.setattr(spread_monitor.time, "time", lambda: 200.0)
    monkeypatch.setattr(spread_monitor.time, "monotonic", lambda: 20.0)
    monkeypatch.setattr(
        spread_monitor.tick_window,
        "recent_ticks",
        lambda seconds=0.0, limit=None: [
            {"epoch": 199.8, "bid": 151.000, "ask": 151.004, "mid": 151.002},
            {"epoch": 200.0, "bid": 151.001, "ask": 151.005, "mid": 151.003},
        ],
    )

    blocked, remain, state, reason = spread_monitor.is_blocked()
    assert blocked is False
    assert remain == 0
    assert state is not None
    assert state.get("source") == "tick_cache"
    assert reason == ""


def test_update_from_tick_triggers_hot_guard_fast(monkeypatch):
    from market_data import spread_monitor

    _reset_state(spread_monitor)
    monkeypatch.setattr(spread_monitor, "DISABLE_SPREAD_GUARD", False)
    monkeypatch.setattr(spread_monitor, "HOT_TRIGGER_ENABLED", True)
    monkeypatch.setattr(spread_monitor, "HOT_TRIGGER_PIPS", 1.0)
    monkeypatch.setattr(spread_monitor, "HOT_MIN_SAMPLES", 2)
    monkeypatch.setattr(spread_monitor, "HOT_COOLDOWN_SECONDS", 3.0)
    mono_values = iter([1.0, 1.2, 1.21, 1.21])
    monkeypatch.setattr(spread_monitor.time, "monotonic", lambda: next(mono_values))

    tick_ts = datetime(2026, 2, 11, 0, 0, tzinfo=timezone.utc)
    spread_monitor.update_from_tick(
        SimpleNamespace(bid=152.000, ask=152.011, time=tick_ts)
    )
    spread_monitor.update_from_tick(
        SimpleNamespace(bid=152.001, ask=152.012, time=tick_ts)
    )
    blocked, remain, state, reason = spread_monitor.is_blocked()
    assert blocked is True
    assert remain >= 2
    assert state is not None
    assert state.get("source") == "snapshot"
    assert "hot_spread" in reason

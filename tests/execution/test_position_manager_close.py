from __future__ import annotations

import threading
import time

from execution import position_manager


class _DummyCon:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _make_pm(con: _DummyCon | None) -> position_manager.PositionManager:
    pm = position_manager.PositionManager.__new__(position_manager.PositionManager)
    pm.con = con
    return pm


def test_close_service_mode_skips_remote_close_even_with_fallback(monkeypatch) -> None:
    pm = _make_pm(_DummyCon())
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        position_manager,
        "_position_manager_service_enabled",
        lambda: True,
        raising=False,
    )

    def _fake_service_request(path: str, payload: dict) -> dict:
        calls.append((path, payload))
        return {"ok": True}

    monkeypatch.setattr(
        position_manager,
        "_position_manager_service_request",
        _fake_service_request,
        raising=False,
    )

    pm.close()

    assert pm.con is None
    assert calls == []


def test_close_local_mode_keeps_remote_close_path(monkeypatch) -> None:
    pm = _make_pm(_DummyCon())
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        position_manager,
        "_position_manager_service_enabled",
        lambda: False,
        raising=False,
    )

    def _fake_service_request(path: str, payload: dict) -> None:
        calls.append((path, payload))
        return None

    monkeypatch.setattr(
        position_manager,
        "_position_manager_service_request",
        _fake_service_request,
        raising=False,
    )

    pm.close()

    assert pm.con is None
    assert calls == [("/position/close", {})]


def test_sync_trades_uses_recent_cache_without_fetch(monkeypatch) -> None:
    pm = position_manager.PositionManager.__new__(position_manager.PositionManager)
    pm._ensure_connection_open = lambda: None
    pm._sync_trades_lock = threading.Lock()
    pm._last_sync_cache = [{"ticket_id": "t1"}, {"ticket_id": "t2"}]
    pm._last_sync_cache_ts = time.monotonic()
    pm._last_sync_cache_window_sec = 5.0
    pm._last_sync_poll_ts = 0.0

    monkeypatch.setattr(
        position_manager,
        "_position_manager_service_request",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        pm,
        "_fetch_closed_trades",
        lambda: (_ for _ in ()).throw(AssertionError("fetch should not run")),
        raising=False,
    )

    result = position_manager.PositionManager.sync_trades(pm, max_fetch=1)
    assert result == [{"ticket_id": "t2"}]


def test_sync_trades_min_interval_skips_duplicate_poll(monkeypatch) -> None:
    pm = position_manager.PositionManager.__new__(position_manager.PositionManager)
    pm._ensure_connection_open = lambda: None
    pm._sync_trades_lock = threading.Lock()
    pm._last_sync_cache = [{"ticket_id": "cached"}]
    pm._last_sync_cache_ts = time.monotonic() - 10.0
    pm._last_sync_cache_window_sec = 0.01
    pm._last_sync_poll_ts = time.monotonic()

    monkeypatch.setattr(
        position_manager,
        "_position_manager_service_request",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        position_manager,
        "_POSITION_MANAGER_SYNC_MIN_INTERVAL_SEC",
        2.0,
        raising=False,
    )
    monkeypatch.setattr(
        pm,
        "_fetch_closed_trades",
        lambda: (_ for _ in ()).throw(AssertionError("fetch should not run")),
        raising=False,
    )

    result = position_manager.PositionManager.sync_trades(pm, max_fetch=10)
    assert result == [{"ticket_id": "cached"}]


def test_bounded_cache_put_evicts_oldest_entry() -> None:
    cache: dict[str, object] = {}
    for idx in range(4):
        position_manager._bounded_cache_put(
            cache,
            f"k{idx}",
            {"v": idx},
            max_entries=3,
        )
    assert list(cache.keys()) == ["k1", "k2", "k3"]

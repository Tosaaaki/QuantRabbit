from __future__ import annotations

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

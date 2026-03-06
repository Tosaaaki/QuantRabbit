from __future__ import annotations

import pathlib
import sys

from fastapi.testclient import TestClient

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.position_manager import worker


def test_service_port_defaults_to_8301(monkeypatch) -> None:
    monkeypatch.delenv("POSITION_MANAGER_SERVICE_PORT", raising=False)

    assert worker._service_port() == 8301


def test_service_port_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("POSITION_MANAGER_SERVICE_PORT", "9315")

    assert worker._service_port() == 9315


class _DummyPositionManager:
    def get_open_positions(self, include_unknown: bool = False) -> dict:
        return {}

    def close(self) -> None:
        return None


def test_sync_trades_uses_dedicated_lock(monkeypatch) -> None:
    monkeypatch.setattr(
        worker.position_manager,
        "PositionManager",
        lambda: _DummyPositionManager(),
    )

    async def _fake_call_manager_with_timeout(*_args, **_kwargs):
        return [{"ticket_id": "t1"}]

    monkeypatch.setattr(
        worker,
        "_call_manager_with_timeout",
        _fake_call_manager_with_timeout,
    )

    with TestClient(worker.app) as client:
        worker.app.state.position_manager_db_call_lock.acquire()
        try:
            response = client.post("/position/sync_trades", json={"max_fetch": 5})
        finally:
            worker.app.state.position_manager_db_call_lock.release()

    assert response.status_code == 200
    assert response.json() == {"ok": True, "result": [{"ticket_id": "t1"}]}


def test_sync_trades_busy_uses_sync_lock(monkeypatch) -> None:
    monkeypatch.setattr(
        worker.position_manager,
        "PositionManager",
        lambda: _DummyPositionManager(),
    )

    async def _fake_call_manager_with_timeout(*_args, **_kwargs):
        return [{"ticket_id": "t1"}]

    monkeypatch.setattr(
        worker,
        "_call_manager_with_timeout",
        _fake_call_manager_with_timeout,
    )

    with TestClient(worker.app) as client:
        worker.app.state.position_manager_sync_trades_call_lock.acquire()
        try:
            response = client.post("/position/sync_trades", json={"max_fetch": 5})
        finally:
            worker.app.state.position_manager_sync_trades_call_lock.release()

    assert response.status_code == 200
    assert response.json() == {"ok": False, "error": "position manager busy"}

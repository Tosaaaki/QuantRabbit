from __future__ import annotations

from fastapi.testclient import TestClient

import apps.autotune_ui as ui


def test_ops_strategy_control_rejects_missing_token(monkeypatch):
    monkeypatch.setattr(ui, "_ops_required_token", lambda: "ops-token")
    client = TestClient(ui.app)

    response = client.post(
        "/ops/strategy-control",
        data={
            "target": "global",
            "entry_enabled": "1",
            "exit_enabled": "1",
            "global_lock": "0",
            "note": "no token",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    location = response.headers.get("location", "")
    assert "strategy_control_error=Unauthorized" in location


def test_ops_strategy_control_accepts_form_token(monkeypatch):
    monkeypatch.setattr(ui, "_ops_required_token", lambda: "ops-token")

    called: dict[str, object] = {}

    def _set_global_flags(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(ui.strategy_control, "set_global_flags", _set_global_flags)
    client = TestClient(ui.app)

    response = client.post(
        "/ops/strategy-control",
        data={
            "target": "global",
            "entry_enabled": "1",
            "exit_enabled": "0",
            "global_lock": "0",
            "note": "with token",
            "ops_token": "ops-token",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    location = response.headers.get("location", "")
    assert "strategy_control_notice=" in location
    assert "strategy_control_error=" not in location
    assert called.get("entry") is True
    assert called.get("exit") is False
    assert called.get("lock") is False

from __future__ import annotations

import apps.autotune_ui as ui


def test_get_secret_optional_uses_ttl_cache(monkeypatch):
    calls = {"count": 0}

    def _secret_stub(key: str) -> str:
        calls["count"] += 1
        return "token-123"

    monkeypatch.setattr(ui, "get_secret", _secret_stub)
    monkeypatch.setattr(ui, "_SECRET_CACHE_TTL_SEC", 60.0)
    ui._secret_cache.clear()

    assert ui._get_secret_optional("ui_snapshot_url") == "token-123"
    assert ui._get_secret_optional("ui_snapshot_url") == "token-123"
    assert calls["count"] == 1


def test_get_secret_optional_caches_missing_key(monkeypatch):
    calls = {"count": 0}

    def _missing_secret_stub(key: str) -> str:
        calls["count"] += 1
        raise KeyError(key)

    monkeypatch.setattr(ui, "get_secret", _missing_secret_stub)
    monkeypatch.setattr(ui, "_SECRET_CACHE_TTL_SEC", 60.0)
    ui._secret_cache.clear()

    assert ui._get_secret_optional("ui_snapshot_lite_url") is None
    assert ui._get_secret_optional("ui_snapshot_lite_url") is None
    assert calls["count"] == 1


def test_load_strategy_control_state_uses_cache(monkeypatch):
    calls = {"count": 0}

    def _state_stub():
        calls["count"] += 1
        return {
            "global": {"entry_enabled": True, "exit_enabled": True, "global_lock": False},
            "strategies": [{"slug": "alpha", "entry_enabled": True, "exit_enabled": True, "global_lock": False}],
            "error": None,
            "discovered_count": 1,
        }

    monkeypatch.setattr(ui, "_compute_strategy_control_state", _state_stub)
    monkeypatch.setattr(ui, "_STRATEGY_CONTROL_CACHE_TTL_SEC", 60.0)
    ui._strategy_control_cache = None
    ui._strategy_control_cache_ts = 0.0

    first = ui._load_strategy_control_state()
    second = ui._load_strategy_control_state()

    assert calls["count"] == 1
    assert first is not second
    first["global"]["entry_enabled"] = False
    assert second["global"]["entry_enabled"] is True

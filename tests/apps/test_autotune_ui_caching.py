from __future__ import annotations

import time

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
    monkeypatch.setattr(ui, "_strategy_control_cache_signature", lambda: (1, 1, 1, 1))
    monkeypatch.setattr(ui, "_STRATEGY_CONTROL_CACHE_TTL_SEC", 60.0)
    ui._strategy_control_cache = None
    ui._strategy_control_cache_ts = 0.0
    ui._strategy_control_cache_sig = None

    first = ui._load_strategy_control_state()
    second = ui._load_strategy_control_state()

    assert calls["count"] == 1
    assert first is not second
    first["global"]["entry_enabled"] = False
    assert second["global"]["entry_enabled"] is True


def test_strategy_control_update_invalidates_cache_global(monkeypatch):
    monkeypatch.setattr(ui, "_ops_required_token", lambda: "ops-token")
    monkeypatch.setattr(ui.strategy_control, "set_global_flags", lambda **_: None)
    monkeypatch.setattr(ui, "_strategy_control_cache_signature", lambda: (1, 1, 1, 1))
    ui._strategy_control_cache = {"global": {"entry_enabled": False}}
    ui._strategy_control_cache_ts = time.monotonic()
    ui._strategy_control_cache_sig = (1, 1, 1, 1)

    result = ui._handle_strategy_control_action(
        target="global",
        strategy=None,
        entry_enabled="1",
        exit_enabled="1",
        lock_enabled="0",
        note="cache clear",
        token="ops-token",
        x_qr_token=None,
        authorization=None,
    )

    assert result["ok"] is True
    assert ui._strategy_control_cache is None
    assert ui._strategy_control_cache_ts == 0.0
    assert ui._strategy_control_cache_sig is None


def test_strategy_control_update_invalidates_cache_strategy(monkeypatch):
    monkeypatch.setattr(ui, "_ops_required_token", lambda: "ops-token")
    monkeypatch.setattr(ui.strategy_control, "set_strategy_flags", lambda *_, **__: None)
    monkeypatch.setattr(ui, "_strategy_control_cache_signature", lambda: (1, 1, 1, 1))
    ui._strategy_control_cache = {"global": {"entry_enabled": True}}
    ui._strategy_control_cache_ts = time.monotonic()
    ui._strategy_control_cache_sig = (1, 1, 1, 1)

    result = ui._handle_strategy_control_action(
        target="strategy",
        strategy="scalp_ping_5s_b",
        entry_enabled="0",
        exit_enabled="1",
        lock_enabled="0",
        note="cache clear",
        token="ops-token",
        x_qr_token=None,
        authorization=None,
    )

    assert result["ok"] is True
    assert ui._strategy_control_cache is None
    assert ui._strategy_control_cache_ts == 0.0
    assert ui._strategy_control_cache_sig is None


def test_load_strategy_control_state_reloads_when_signature_changes(monkeypatch):
    calls = {"count": 0}
    signature = {"value": (1, 1, 1, 1)}

    def _state_stub():
        calls["count"] += 1
        return {
            "global": {"entry_enabled": True, "exit_enabled": True, "global_lock": False},
            "strategies": [],
            "error": None,
            "discovered_count": calls["count"],
        }

    monkeypatch.setattr(ui, "_compute_strategy_control_state", _state_stub)
    monkeypatch.setattr(ui, "_strategy_control_cache_signature", lambda: signature["value"])
    monkeypatch.setattr(ui, "_STRATEGY_CONTROL_CACHE_TTL_SEC", 60.0)
    ui._strategy_control_cache = None
    ui._strategy_control_cache_ts = 0.0
    ui._strategy_control_cache_sig = None

    first = ui._load_strategy_control_state()
    signature["value"] = (2, 1, 1, 1)
    second = ui._load_strategy_control_state()

    assert calls["count"] == 2
    assert first["discovered_count"] == 1
    assert second["discovered_count"] == 2

from __future__ import annotations

from utils.env_utils import env_bool, env_float, env_get, env_int


def test_env_get_uses_prefixed_keys_first(monkeypatch) -> None:
    monkeypatch.setenv("BASE_UNITS_EQUITY_REF", "999999")
    monkeypatch.setenv("TRENDMA_BASE_UNITS_EQUITY_REF", "2000000")
    monkeypatch.setenv("TRENDMA_UNIT_BASE_UNITS_EQUITY_REF", "3000000")

    assert env_get("BASE_UNITS_EQUITY_REF", "1000000", prefix="TRENDMA") == "3000000"


def test_env_get_strict_prefix_does_not_fallback_to_global(monkeypatch) -> None:
    monkeypatch.setenv("PERF_GUARD_MODE", "warn")

    assert env_get("PERF_GUARD_MODE", "block", prefix="TRENDMA", allow_global_fallback=False) == "block"
    assert env_get("PERF_GUARD_MODE", "block", prefix="TRENDMA", allow_global_fallback=True) == "warn"


def test_env_typed_helpers_respect_strict_prefix_mode(monkeypatch) -> None:
    monkeypatch.setenv("PERF_GUARD_ENABLED", "0")
    monkeypatch.setenv("DYN_SIZE_BASE_RISK_PCT", "0.05")
    monkeypatch.setenv("PERF_GUARD_MIN_TRADES", "77")

    assert env_bool("PERF_GUARD_ENABLED", True, prefix="M1SCALP", allow_global_fallback=False) is True
    assert env_float("DYN_SIZE_BASE_RISK_PCT", 0.01, prefix="M1SCALP", allow_global_fallback=False) == 0.01
    assert env_int("PERF_GUARD_MIN_TRADES", 12, prefix="M1SCALP", allow_global_fallback=False) == 12

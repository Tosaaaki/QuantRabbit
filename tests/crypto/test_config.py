from __future__ import annotations

import pytest

from quant_rabbit.crypto.config import CryptoSafetyContract, SafetyInvariantError


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NO_EXECUTE", "false"),
        ("CRYPTO_LIVE_READY", "true"),
        ("WITHDRAWAL_ENABLED", "true"),
        ("CRYPTO_ORDER_AUTHORITY", "TRADER"),
    ],
)
def test_live_authority_environment_fails_closed(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(SafetyInvariantError):
        CryptoSafetyContract.from_env()


def test_safe_defaults_are_paper_only(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "NO_EXECUTE",
        "CRYPTO_LIVE_READY",
        "WITHDRAWAL_ENABLED",
        "CRYPTO_ORDER_AUTHORITY",
    ):
        monkeypatch.delenv(name, raising=False)
    assert CryptoSafetyContract.from_env().as_dict() == {
        "no_execute": True,
        "crypto_live_ready": False,
        "withdrawal_enabled": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }

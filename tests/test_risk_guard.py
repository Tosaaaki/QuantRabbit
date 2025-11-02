import importlib
from typing import Callable

import pytest

from execution import risk_guard

_ENV_KEYS = ("RISK_MAX_LOT", "MAX_LOT", "RISK_PCT")


@pytest.fixture
def reload_guard(monkeypatch) -> Callable[..., object]:
    def _reload(**overrides):
        for key in _ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        for key, value in overrides.items():
            monkeypatch.setenv(key, str(value))
        return importlib.reload(risk_guard)

    yield _reload

    importlib.reload(risk_guard)


def test_allowed_lot_respects_max_override(reload_guard):
    module = reload_guard(RISK_PCT=0.02, RISK_MAX_LOT=3.0)
    assert module.allowed_lot(10000, sl_pips=5) == 3.0
    reload_guard()


def test_allowed_lot_increases_with_risk_pct(reload_guard):
    module = reload_guard(RISK_PCT=0.05, RISK_MAX_LOT=2.5)
    assert module.allowed_lot(10000, sl_pips=20) == 2.5
    reload_guard()

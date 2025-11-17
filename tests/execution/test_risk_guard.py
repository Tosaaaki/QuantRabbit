from __future__ import annotations

from execution import risk_guard


def test_build_exposure_state_handles_manual_and_bot_units(monkeypatch):
    positions = {
        "manual": {"open_trades": [{"units": 15000}]},
        "macro": {"open_trades": [{"units": 25000}, {"units": -5000}]},
        "micro": {"units": 8000},  # fallback to net units
        "__net__": {"units": 0},
    }

    # Avoid noisy metric logging during unit tests.
    monkeypatch.setattr(
        risk_guard, "log_metric", lambda *args, **kwargs: None, raising=False
    )

    state = risk_guard.build_exposure_state(
        positions,
        equity=5_000_000.0,  # JPY
        price=150.0,  # USDJPY
        cap_ratio=0.9,
    )
    assert state is not None
    assert state.manual_units == 15000.0
    # bot units counts absolute trades (25000 + 5000 + 8000)
    assert state.bot_units == 38000.0
    assert state.limit_units() > 0
    assert state.available_units() > 0
    assert not state.would_exceed(1000)
    # Consuming almost all remaining capacity should trip the guard.
    assert state.would_exceed(int(state.available_units()) + 1000)


def test_build_exposure_state_returns_none_when_equity_missing(monkeypatch):
    monkeypatch.setattr(
        risk_guard, "log_metric", lambda *args, **kwargs: None, raising=False
    )
    state = risk_guard.build_exposure_state(
        {"macro": {"units": 10000}},
        equity=None,
        price=150.0,
    )
    assert state is None

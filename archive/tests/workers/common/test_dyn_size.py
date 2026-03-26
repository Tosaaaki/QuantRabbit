from __future__ import annotations

from types import SimpleNamespace

from workers.common import dyn_size


def _snapshot(*, free_margin_ratio: float = 0.25):
    return SimpleNamespace(
        nav=10000.0,
        balance=10000.0,
        margin_available=10000.0,
        margin_rate=0.04,
        free_margin_ratio=free_margin_ratio,
    )


def test_compute_units_keeps_soft_floor_when_conditions_are_healthy(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dyn_size, "get_account_snapshot", lambda: _snapshot())
    monkeypatch.setattr(
        dyn_size,
        "scale_base_units",
        lambda base_entry_units, **kwargs: base_entry_units,
    )
    monkeypatch.setattr(dyn_size, "allowed_lot", lambda *args, **kwargs: 0.008)
    monkeypatch.setattr(
        dyn_size.perf_guard,
        "perf_scale",
        lambda *args, **kwargs: SimpleNamespace(multiplier=1.0, reason="flat"),
    )

    res = dyn_size.compute_units(
        entry_price=157.8,
        sl_pips=3.0,
        base_entry_units=1000,
        min_units=100,
        max_margin_usage=0.9,
        spread_pips=0.6,
        spread_soft_cap=1.0,
        adx=22.0,
        signal_score=0.5,
        pocket="scalp",
        strategy_tag="Healthy",
    )

    assert res.units == 900
    assert res.factors["perf_mult"] == 1.0


def test_compute_units_preserves_reduction_below_base_when_perf_scale_reduces(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dyn_size, "get_account_snapshot", lambda: _snapshot())
    monkeypatch.setattr(
        dyn_size,
        "scale_base_units",
        lambda base_entry_units, **kwargs: base_entry_units,
    )
    monkeypatch.setattr(dyn_size, "allowed_lot", lambda *args, **kwargs: 0.01)
    monkeypatch.setattr(
        dyn_size.perf_guard,
        "perf_scale",
        lambda *args, **kwargs: SimpleNamespace(multiplier=0.8, reason="reduce"),
    )

    res = dyn_size.compute_units(
        entry_price=157.8,
        sl_pips=3.0,
        base_entry_units=1000,
        min_units=100,
        max_margin_usage=0.9,
        spread_pips=0.6,
        spread_soft_cap=1.0,
        adx=22.0,
        signal_score=0.5,
        pocket="scalp",
        strategy_tag="Weak",
    )

    assert res.units == 800
    assert res.factors["perf_mult"] == 0.8

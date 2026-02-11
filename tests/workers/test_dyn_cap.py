from __future__ import annotations

import pytest

from workers.common.dyn_cap import compute_cap


def test_dyn_cap_default_range_cap_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DYN_CAP_RANGE_CAP", raising=False)

    res = compute_cap(
        atr_pips=2.0,
        free_ratio=0.5,
        range_active=True,
        perf_pf=None,
        pos_bias=0.0,
        cap_min=0.25,
        cap_max=0.95,
    )
    assert res.cap == pytest.approx(0.55)


def test_dyn_cap_global_range_cap_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DYN_CAP_RANGE_CAP", "0.82")

    res = compute_cap(
        atr_pips=2.0,
        free_ratio=0.5,
        range_active=True,
        perf_pf=None,
        pos_bias=0.0,
        cap_min=0.25,
        cap_max=0.95,
    )
    assert res.cap == pytest.approx(0.82)


def test_dyn_cap_prefix_pf_cut_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DYN_CAP_PF_CUT_MAX", raising=False)
    monkeypatch.delenv("DYN_CAP_PF_CUT_MULT", raising=False)
    monkeypatch.setenv("M1SCALP_DYN_CAP_PF_CUT_MAX", "0.75")
    monkeypatch.setenv("M1SCALP_DYN_CAP_PF_CUT_MULT", "0.95")

    baseline = compute_cap(
        atr_pips=2.0,
        free_ratio=0.5,
        range_active=False,
        perf_pf=0.8,
        pos_bias=0.0,
        cap_min=0.25,
        cap_max=0.95,
    )
    tuned = compute_cap(
        atr_pips=2.0,
        free_ratio=0.5,
        range_active=False,
        perf_pf=0.8,
        pos_bias=0.0,
        cap_min=0.25,
        cap_max=0.95,
        env_prefix="M1SCALP",
    )

    assert tuned.cap > baseline.cap

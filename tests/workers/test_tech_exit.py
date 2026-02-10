from __future__ import annotations

import importlib

import pytest


def test_tech_exit_disabled_skips_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    import workers.common.tech_exit as te

    importlib.reload(te)
    te._LAST_TECH_EXIT_TS.clear()

    called = {"n": 0}

    def _fake_eval(**kwargs):  # type: ignore[no-untyped-def]
        called["n"] += 1
        return object()

    monkeypatch.setattr(te, "evaluate_exit_techniques", _fake_eval)

    ok, reason, allow_neg = te.maybe_tech_exit(
        trade={"trade_id": "T1"},
        side="long",
        pocket="scalp",
        pnl_pips=-5.0,
        hold_sec=999.0,
        current_price=150.0,
        strategy_tag="Foo",
        exit_profile={"tech_exit_enabled": False},
    )
    assert ok is False
    assert reason is None
    assert allow_neg is False
    assert called["n"] == 0


def test_tech_exit_uses_pocket_default_min_neg(monkeypatch: pytest.MonkeyPatch) -> None:
    import workers.common.tech_exit as te
    from analysis.technique_engine import TechniqueExitDecision

    importlib.reload(te)
    te._LAST_TECH_EXIT_TS.clear()

    called = {"n": 0}

    def _fake_eval(**kwargs):  # type: ignore[no-untyped-def]
        called["n"] += 1
        return TechniqueExitDecision(True, "tech_return_fail", True, {})

    monkeypatch.setattr(te, "evaluate_exit_techniques", _fake_eval)

    # pocket=micro => default min_neg_pips=3.5, so pnl=-3.4 should not evaluate.
    ok, reason, allow_neg = te.maybe_tech_exit(
        trade={"trade_id": "T2"},
        side="long",
        pocket="micro",
        pnl_pips=-3.4,
        hold_sec=999.0,
        current_price=150.0,
        strategy_tag="MicroRangeBreak",
        exit_profile={"tech_exit_enabled": True},
    )
    assert ok is False
    assert reason is None
    assert allow_neg is False
    assert called["n"] == 0


def test_tech_exit_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    import workers.common.tech_exit as te
    from analysis.technique_engine import TechniqueExitDecision

    importlib.reload(te)
    te._LAST_TECH_EXIT_TS.clear()

    calls = {"n": 0}

    def _fake_eval(**kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return TechniqueExitDecision(True, "tech_return_fail", True, {})

    times = {"t": 100.0}

    def _fake_mono() -> float:
        return float(times["t"])

    monkeypatch.setattr(te, "evaluate_exit_techniques", _fake_eval)
    monkeypatch.setattr(te.time, "monotonic", _fake_mono)

    prof = {
        "tech_exit_enabled": True,
        "tech_exit_min_neg_pips": 1.0,
        "tech_exit_cooldown_sec": 10.0,
    }

    ok1, reason1, allow1 = te.maybe_tech_exit(
        trade={"trade_id": "T3"},
        side="short",
        pocket="scalp",
        pnl_pips=-2.0,
        hold_sec=999.0,
        current_price=150.0,
        strategy_tag="fast_scalp",
        exit_profile=prof,
    )
    assert ok1 is True
    assert reason1 == "tech_return_fail"
    assert allow1 is True
    assert calls["n"] == 1

    times["t"] = 105.0  # within cooldown
    ok2, reason2, allow2 = te.maybe_tech_exit(
        trade={"trade_id": "T3"},
        side="short",
        pocket="scalp",
        pnl_pips=-2.0,
        hold_sec=999.0,
        current_price=150.0,
        strategy_tag="fast_scalp",
        exit_profile=prof,
    )
    assert ok2 is False
    assert reason2 is None
    assert allow2 is False
    assert calls["n"] == 1


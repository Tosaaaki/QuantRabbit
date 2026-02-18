from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _exit_forecast_modules() -> list[str]:
    root = Path(__file__).resolve().parents[2] / "workers"
    modules: list[str] = []
    for path in sorted(root.glob("*/exit_forecast.py")):
        if path.parent.name == "common":
            continue
        modules.append(f"workers.{path.parent.name}.exit_forecast")
    return modules


EXIT_FORECAST_MODULES = _exit_forecast_modules()


def _mod(module_path: str):
    return importlib.import_module(module_path)


@pytest.mark.parametrize("module_path", EXIT_FORECAST_MODULES)
def test_build_adjustment_disabled(module_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _mod(module_path)
    monkeypatch.setenv("EXIT_FORECAST_ENABLED", "0")
    adj = module.build_exit_forecast_adjustment(
        side="long",
        entry_thesis={"forecast": {"p_up": 0.2}},
    )
    assert adj.enabled is False
    assert adj.contra_score == 0.0
    assert adj.profit_take_mult == 1.0


@pytest.mark.parametrize("module_path", EXIT_FORECAST_MODULES)
def test_build_adjustment_contra_tightens_exit(module_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _mod(module_path)
    monkeypatch.delenv("EXIT_FORECAST_ENABLED", raising=False)
    monkeypatch.setenv("EXIT_FORECAST_MIN_AGAINST_PROB", "0.54")
    adj = module.build_exit_forecast_adjustment(
        side="long",
        entry_thesis={
            "forecast": {
                "p_up": 0.18,
                "edge": 0.77,
                "reason": "edge_block",
            }
        },
    )
    assert adj.enabled is True
    assert adj.against_prob is not None and adj.against_prob > 0.8
    assert adj.contra_score > 0.4
    assert adj.loss_cut_mult < 1.0
    assert adj.profit_take_mult < 1.0
    assert adj.lock_buffer_mult > 1.0


@pytest.mark.parametrize("module_path", EXIT_FORECAST_MODULES)
def test_apply_adjustments_for_targets_and_loss_cut(module_path: str) -> None:
    module = _mod(module_path)
    adj = module.build_exit_forecast_adjustment(
        side="short",
        entry_thesis={
            "forecast": {"p_up": 0.85, "edge": 0.8},
        },
    )
    profit_take, trail_start, trail_backoff, lock_buffer = module.apply_exit_forecast_to_targets(
        profit_take=3.0,
        trail_start=2.0,
        trail_backoff=1.0,
        lock_buffer=0.5,
        adjustment=adj,
    )
    assert profit_take <= 3.0
    assert trail_start <= 2.0
    assert trail_backoff <= 1.0
    assert lock_buffer >= 0.5

    soft, hard, hold = module.apply_exit_forecast_to_loss_cut(
        soft_pips=10.0,
        hard_pips=16.0,
        max_hold_sec=1200.0,
        adjustment=adj,
    )
    assert soft <= 10.0
    assert hard <= 16.0
    assert hard >= soft
    assert hold is not None and hold <= 1200.0


@pytest.mark.parametrize("module_path", EXIT_FORECAST_MODULES)
def test_target_reach_prob_influences_contra_score(
    module_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _mod(module_path)
    monkeypatch.delenv("EXIT_FORECAST_ENABLED", raising=False)
    monkeypatch.setenv("EXIT_FORECAST_PRICE_HINT_ENABLED", "0")
    monkeypatch.setenv("EXIT_FORECAST_TARGET_REACH_ENABLED", "1")
    monkeypatch.setenv("EXIT_FORECAST_TARGET_REACH_WEIGHT_MAX", "0.35")
    monkeypatch.setenv("EXIT_FORECAST_MIN_AGAINST_PROB", "0.70")

    low = module.build_exit_forecast_adjustment(
        side="long",
        entry_thesis={
            "forecast": {
                "p_up": 0.62,
                "edge": 0.72,
                "target_reach_prob": 0.20,
            }
        },
    )
    high = module.build_exit_forecast_adjustment(
        side="long",
        entry_thesis={
            "forecast": {
                "p_up": 0.62,
                "edge": 0.72,
                "target_reach_prob": 0.85,
            }
        },
    )

    assert low.target_reach_prob == pytest.approx(0.2)
    assert high.target_reach_prob == pytest.approx(0.85)
    assert low.contra_score > high.contra_score

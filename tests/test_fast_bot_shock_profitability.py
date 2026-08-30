from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "analyze_fast_bot_shock_profitability.py"
SPEC = importlib.util.spec_from_file_location("shock_profitability", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def metric(*, trades: int, pf: float, net: float, streak: int = 3) -> dict:
    return {
        "trades": trades,
        "risk_scaled_profit_factor": pf,
        "risk_scaled_net_pip_units": net,
        "maximum_loss_streak": streak,
    }


def test_selection_never_uses_holdout() -> None:
    cells = [
        {
            "candidate": "A",
            "target_r": 0.5,
            "train": metric(trades=150, pf=1.2, net=10),
            "validation": metric(trades=80, pf=1.1, net=5),
            "holdout": metric(trades=50, pf=0.1, net=-100),
        },
        {
            "candidate": "B",
            "target_r": 0.5,
            "train": metric(trades=150, pf=1.1, net=8),
            "validation": metric(trades=80, pf=1.05, net=4),
            "holdout": metric(trades=50, pf=5.0, net=500),
        },
    ]
    assert MODULE.select_without_holdout(cells)["candidate"] == "A"


def test_selection_rejects_negative_validation() -> None:
    cells = [
        {
            "candidate": "A",
            "target_r": 0.5,
            "train": metric(trades=150, pf=1.2, net=10),
            "validation": metric(trades=80, pf=0.99, net=-1),
        }
    ]
    assert MODULE.select_without_holdout(cells) is None


def test_candidate_predicates_are_side_relative() -> None:
    predicate = MODULE._candidates()["CONFIRM_H1_H4"]
    base = {
        "continuation_confirmed": True,
        "h1_return_pips": 4.0,
        "h4_return_pips": 8.0,
        "h1_efficiency": 0.3,
        "spread_ratio": 1.0,
        "spread_pips": 0.8,
    }
    assert predicate({**base, "direction": 1})
    assert predicate({**base, "direction": -1})
    assert not predicate({**base, "h4_return_pips": -0.1})


def test_holdout_requires_both_directions_and_cost_resilience() -> None:
    holdout = {
        "trades": 40,
        "risk_scaled_profit_factor": 1.2,
        "risk_scaled_net_pip_units": 4.0,
        "cost_stress_0_5": {"risk_scaled_profit_factor": 0.95},
    }
    sides = {"UP": {"trades": 20}, "DOWN": {"trades": 20}}
    assert MODULE._holdout_admissible(holdout, sides)
    assert not MODULE._holdout_admissible(holdout, {"UP": {"trades": 35}, "DOWN": {"trades": 5}})


def test_split_boundaries_are_chronological() -> None:
    assert MODULE._split(MODULE.TRAIN_END - 1) == "TRAIN_2020_2023"
    assert MODULE._split(MODULE.TRAIN_END) == "VALIDATION_2024_2025"
    assert MODULE._split(MODULE.VALIDATION_END) == "HOLDOUT_2026"

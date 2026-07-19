from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-blind-exit-scenarios.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_blind_exit_scenarios", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m5_context_excludes_decision_bar_high_low_and_close() -> None:
    module = _module()
    rows = [(index * 300, index, index, index, index) for index in range(50)]
    decision_bar = 40

    window = module._completed_m5_window(rows, decision_bar)

    assert len(window) == module.M5_BARS == 36
    assert window[0][0] == 4 * 300
    assert window[-1][0] == 39 * 300
    assert all(row[0] < rows[decision_bar][0] for row in window)
    assert rows[decision_bar] not in window


def test_m5_context_rejects_insufficient_completed_history() -> None:
    module = _module()
    rows = [(index * 300,) for index in range(40)]

    with pytest.raises(ValueError, match="insufficient completed M5 history"):
        module._completed_m5_window(rows, module.M5_BARS - 1)


@pytest.mark.parametrize(
    ("trend", "row", "entry", "tp", "pip"),
    [
        (
            "LONG",
            (0, 100.0, 103.0, 99.0, 101.0, 100.1, 103.1, 99.1, 101.1),
            100.0,
            103.0,
            1.0,
        ),
        (
            "SHORT",
            (0, 100.0, 101.0, 97.0, 99.0, 100.1, 101.1, 97.0, 99.1),
            100.0,
            97.0,
            1.0,
        ),
    ],
)
def test_hold_tp_can_fill_inside_the_decision_bar(
    trend: str,
    row: tuple[float, ...],
    entry: float,
    tp: float,
    pip: float,
) -> None:
    module = _module()

    assert (
        module._hold_branch_pips(
            [row],
            0,
            trend=trend,
            entry=entry,
            tp=tp,
            pip=pip,
        )
        == module.TP_PIPS
    )


@pytest.mark.parametrize(
    ("trend", "row", "tp"),
    [
        ("LONG", (0, 100.0, 103.0, 99.0, 101.0, 100.1, 103.1, 99.1, 101.1), 103.0),
        ("SHORT", (0, 100.0, 101.0, 97.0, 99.0, 100.1, 101.1, 97.0, 99.1), 97.0),
    ],
)
def test_entry_bar_tp_touch_excludes_scenario_before_decision(
    trend: str, row: tuple[float, ...], tp: float
) -> None:
    module = _module()

    assert module._tp_fills_before_decision(
        [row],
        0,
        300,
        trend=trend,
        tp=tp,
    )


def test_missing_bar_invalidates_predecision_and_hold_truth() -> None:
    module = _module()
    row = (0, 100.0, 101.0, 99.0, 100.0, 100.1, 101.1, 99.1, 100.1)
    after_gap = (
        600,
        100.0,
        101.0,
        99.0,
        100.0,
        100.1,
        101.1,
        99.1,
        100.1,
    )

    assert (
        module._tp_fills_before_decision(
            [row, after_gap],
            0,
            900,
            trend="LONG",
            tp=103.0,
        )
        is None
    )
    assert (
        module._hold_branch_pips(
            [row, after_gap],
            0,
            trend="LONG",
            entry=100.0,
            tp=103.0,
            pip=1.0,
        )
        is None
    )

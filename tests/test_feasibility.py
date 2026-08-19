"""Contract tests for the outcome-blind feasibility screen."""

from __future__ import annotations

import pytest

from quant_rabbit.feasibility import (
    SIGNAL_COLUMNS,
    Cell,
    FeasibilityError,
    Row,
    build_cells,
    decompose,
    parse_rows,
    screen,
)


def _record(pair: str, horizon: int, move: float, cost: float) -> dict:
    """Invert the identity: build a row that decomposes back to (move, cost)."""

    return {
        "pair": pair,
        "horizon_minutes": horizon,
        "long_executable_return_pips": move - cost,
        "short_executable_return_pips": -move - cost,
    }


def _corpus(pair: str, horizon: int, move: float, cost: float, n: int = 40) -> list[dict]:
    # Alternate the sign so |move| is stable while the signed mean is ~zero,
    # which is what a directionless market looks like.
    return [_record(pair, horizon, move if index % 2 else -move, cost) for index in range(n)]


# --------------------------------------------------------------------------- #
# The identity                                                                  #
# --------------------------------------------------------------------------- #


def test_decompose_recovers_move_and_cost() -> None:
    move, cost = decompose(*(4.0 - 1.5, -4.0 - 1.5))
    assert move == pytest.approx(4.0)
    assert cost == pytest.approx(1.5)


def test_oracle_return_equals_the_better_side() -> None:
    long_pips, short_pips = 4.0 - 1.5, -4.0 - 1.5
    row = parse_rows([_record("USD_JPY", 5, 4.0, 1.5)])[0]
    assert row.oracle_pips == pytest.approx(max(long_pips, short_pips))


def test_oracle_is_negative_when_cost_exceeds_the_move() -> None:
    assert Row("GBP_NZD", 5, 1.0, 8.3).oracle_pips == pytest.approx(-7.3)


# --------------------------------------------------------------------------- #
# Outcome blindness — the property that keeps selection bias out               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("column", sorted(SIGNAL_COLUMNS))
def test_any_signal_column_is_refused(column: str) -> None:
    record = _record("USD_JPY", 5, 2.0, 0.8) | {column: 1.23}
    with pytest.raises(FeasibilityError, match="outcome-blind"):
        parse_rows([record])


def test_a_null_signal_column_is_tolerated() -> None:
    """A column present but empty carries no outcome, so it is not a leak."""

    record = _record("USD_JPY", 5, 2.0, 0.8) | {"selected_direction": None}
    assert len(parse_rows([record])) == 1


def test_verdict_ignores_which_side_actually_won() -> None:
    """Flipping every move's sign leaves the ceiling untouched."""

    rows = [_record("USD_JPY", 60, 3.0, 0.8) for _ in range(40)]
    flipped = [_record("USD_JPY", 60, -3.0, 0.8) for _ in range(40)]
    assert build_cells(parse_rows(rows))[0].mean_oracle_pips == pytest.approx(
        build_cells(parse_rows(flipped))[0].mean_oracle_pips
    )


# --------------------------------------------------------------------------- #
# Verdicts                                                                      #
# --------------------------------------------------------------------------- #


def test_wide_spread_short_horizon_is_impossible() -> None:
    """GBP_NZD at 5 minutes: a perfect predictor still loses."""

    cell = build_cells(parse_rows(_corpus("GBP_NZD", 5, 2.4, 8.3)))[0]
    assert cell.verdict == "IMPOSSIBLE"
    assert cell.mean_oracle_pips < 0


def test_tight_spread_long_horizon_is_feasible() -> None:
    cell = build_cells(parse_rows(_corpus("USD_JPY", 60, 4.3, 0.8)))[0]
    assert cell.verdict == "FEASIBLE"
    assert cell.mean_oracle_pips == pytest.approx(3.5)


def test_break_even_cell_is_not_admitted() -> None:
    """Exactly zero is closed: a ceiling of zero pays nothing."""

    assert build_cells(parse_rows(_corpus("EUR_CHF", 30, 1.5, 1.5)))[0].verdict == "IMPOSSIBLE"


def test_min_ceiling_raises_the_bar() -> None:
    cells = build_cells(parse_rows(_corpus("USD_JPY", 60, 1.8, 0.8)))
    assert cells[0].admits(0.0)
    assert not cells[0].admits(2.0)


# --------------------------------------------------------------------------- #
# Aggregation hygiene                                                           #
# --------------------------------------------------------------------------- #


def test_thin_cells_are_dropped_not_admitted_on_noise() -> None:
    assert build_cells(parse_rows(_corpus("USD_JPY", 60, 5.0, 0.8, n=10))) == []


def test_unmeasured_bars_are_dropped_never_imputed() -> None:
    payload = _corpus("USD_JPY", 60, 3.0, 0.8, n=5) + [
        {"pair": "USD_JPY", "horizon_minutes": 60, "long_executable_return_pips": None,
         "short_executable_return_pips": None}
    ]
    assert len(parse_rows(payload)) == 5


def test_pairs_and_horizons_stay_separate_cells() -> None:
    payload = _corpus("USD_JPY", 5, 0.8, 0.8) + _corpus("USD_JPY", 60, 4.3, 0.8)
    cells = {(cell.pair, cell.horizon_minutes): cell for cell in build_cells(parse_rows(payload))}
    assert cells[("USD_JPY", 5)].verdict == "IMPOSSIBLE"
    assert cells[("USD_JPY", 60)].verdict == "FEASIBLE"


def test_screen_splits_admitted_from_closed() -> None:
    payload = _corpus("GBP_NZD", 5, 2.4, 8.3) + _corpus("USD_JPY", 60, 4.3, 0.8)
    report = screen(payload)
    assert (report["admitted"], report["closed"]) == (1, 1)
    assert report["outcome_blind"] is True
    assert report["cells_detail"][0]["pair"] == "USD_JPY"


def test_screen_reports_zero_admitted_rather_than_lowering_the_bar() -> None:
    report = screen(_corpus("GBP_NZD", 5, 2.4, 8.3))
    assert report["admitted"] == 0
    assert report["closed"] == 1


def test_stderr_shrinks_as_rows_accumulate() -> None:
    def stderr(n: int) -> float:
        payload = [_record("USD_JPY", 60, 3.0 + (index % 7) * 0.5, 0.8) for index in range(n)]
        return build_cells(parse_rows(payload))[0].oracle_stderr_pips

    assert stderr(400) < stderr(40)

"""Outcome-blind feasibility screen for (pair, horizon) cells.

What this answers
-----------------
Before asking whether a signal is any good, ask whether the cell it trades in
can pay for itself *at all*. A perfect predictor — one that picks the winning
side on every single bar — earns `|mid move| - cost` per trade. Where that is
negative, no signal, no threshold, no sizing and no stop design can produce
profit. The cell is arithmetically closed.

The identity this rests on
--------------------------
OANDA executable returns decompose exactly::

    long  = exit_bid - entry_ask =  move - cost
    short = entry_bid - exit_ask = -move - cost

    move = (long - short) / 2          # signed mid-to-mid move
    cost = -(long + short) / 2         # round-trip spread borne by one trade

so the perfect-oracle return of a single row is `max(long, short)`, which is
`|move| - cost`. No model, no fit, no free parameter.

Why it is outcome-blind
-----------------------
The screen reads only `long_executable_return_pips` and
`short_executable_return_pips`. It never reads the direction a strategy chose,
its confidence, or its realised result. A cell therefore cannot be admitted
because it happened to win — which is the selection bias the 2026-08-11
methodology audit graded `FAIL`. `parse_rows` refuses a payload that tries to
feed it a signal column.

What a positive ceiling does and does not mean
----------------------------------------------
`FEASIBLE` means profit is *not forbidden* here. It is an upper bound attained
only by a predictor that is never wrong. Measured predictors capture some
fraction of it, and on the 2026-08-13 contextual candidate corpus that fraction
was zero: gross directional capture was -0.03 pips at 5 minutes with 49.2% of
rows positive — a coin. Treating a positive ceiling as an expectation would
repeat exactly that error.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Columns that carry a strategy's opinion. Their presence in a feasibility
# payload means the screen is being handed outcome-conditioned data.
SIGNAL_COLUMNS = frozenset(
    {
        "selected_direction_executable_return_pips",
        "directional_executable_return_pips",
        "selected_direction",
        "relative_confidence",
        "cross_pair_rank",
        "directional_candidate",
    }
)


class FeasibilityError(ValueError):
    """Raised when a payload would let signal information into the screen."""


@dataclass(frozen=True)
class Row:
    """One evaluated bar, reduced to the only two facts the screen may use."""

    pair: str
    horizon_minutes: int
    move_pips: float
    cost_pips: float

    @property
    def oracle_pips(self) -> float:
        """What a never-wrong predictor earns on this bar."""

        return abs(self.move_pips) - self.cost_pips


@dataclass(frozen=True)
class Cell:
    pair: str
    horizon_minutes: int
    rows: int
    mean_abs_move_pips: float
    mean_cost_pips: float
    mean_oracle_pips: float
    median_oracle_pips: float
    oracle_stderr_pips: float

    @property
    def verdict(self) -> str:
        """`IMPOSSIBLE` when even a perfect predictor cannot break even."""

        return "FEASIBLE" if self.mean_oracle_pips > 0 else "IMPOSSIBLE"

    def admits(self, min_ceiling_pips: float) -> bool:
        return self.mean_oracle_pips >= min_ceiling_pips

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair": self.pair,
            "horizon_minutes": self.horizon_minutes,
            "rows": self.rows,
            "mean_abs_move_pips": round(self.mean_abs_move_pips, 4),
            "mean_cost_pips": round(self.mean_cost_pips, 4),
            "mean_oracle_pips": round(self.mean_oracle_pips, 4),
            "median_oracle_pips": round(self.median_oracle_pips, 4),
            "oracle_stderr_pips": round(self.oracle_stderr_pips, 4),
            "verdict": self.verdict,
        }


def decompose(long_pips: float, short_pips: float) -> tuple[float, float]:
    """Split a long/short executable pair into (signed move, round-trip cost)."""

    return (long_pips - short_pips) / 2.0, -(long_pips + short_pips) / 2.0


def parse_rows(payload: Iterable[dict[str, Any]], *, strict: bool = True) -> list[Row]:
    """Reduce evaluation records to `Row`s, dropping anything not price-true.

    Rows whose executable returns are missing are dropped rather than filled:
    a null outcome is an unmeasured bar, and imputing it would invent the very
    price truth the corpus says it lacks.
    """

    rows: list[Row] = []
    for record in payload:
        if strict:
            leaked = SIGNAL_COLUMNS.intersection(record)
            if leaked and any(record[column] is not None for column in leaked):
                raise FeasibilityError(
                    f"feasibility screen must stay outcome-blind; drop {sorted(leaked)} before calling"
                )
        long_pips = record.get("long_executable_return_pips")
        short_pips = record.get("short_executable_return_pips")
        pair = record.get("pair")
        horizon = record.get("horizon_minutes")
        if long_pips is None or short_pips is None or pair is None or horizon is None:
            continue
        try:
            move, cost = decompose(float(long_pips), float(short_pips))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(move) and math.isfinite(cost)):
            continue
        rows.append(Row(pair=str(pair), horizon_minutes=int(horizon), move_pips=move, cost_pips=cost))
    return rows


def build_cells(rows: Sequence[Row], *, min_rows: int = 30) -> list[Cell]:
    """Aggregate rows into (pair, horizon) cells.

    Cells thinner than `min_rows` are dropped: a ceiling estimated from a
    handful of bars is noise, and admitting on it reintroduces the small-sample
    selection the audit flagged.
    """

    grouped: dict[tuple[str, int], list[Row]] = {}
    for row in rows:
        grouped.setdefault((row.pair, row.horizon_minutes), []).append(row)

    cells: list[Cell] = []
    for (pair, horizon), group in grouped.items():
        if len(group) < min_rows:
            continue
        oracle = [row.oracle_pips for row in group]
        spread = statistics.stdev(oracle) if len(oracle) > 1 else 0.0
        cells.append(
            Cell(
                pair=pair,
                horizon_minutes=horizon,
                rows=len(group),
                mean_abs_move_pips=statistics.fmean(abs(row.move_pips) for row in group),
                mean_cost_pips=statistics.fmean(row.cost_pips for row in group),
                mean_oracle_pips=statistics.fmean(oracle),
                median_oracle_pips=statistics.median(oracle),
                oracle_stderr_pips=spread / math.sqrt(len(oracle)) if oracle else 0.0,
            )
        )
    return sorted(cells, key=lambda cell: (-cell.mean_oracle_pips, cell.pair, cell.horizon_minutes))


def screen(
    payload: Iterable[dict[str, Any]],
    *,
    min_ceiling_pips: float = 0.0,
    min_rows: int = 30,
    strict: bool = True,
) -> dict[str, Any]:
    """Full screen: decompose, aggregate, and split admitted from closed cells."""

    rows = parse_rows(payload, strict=strict)
    cells = build_cells(rows, min_rows=min_rows)
    admitted = [cell for cell in cells if cell.admits(min_ceiling_pips)]
    return {
        "schema": "QR_FEASIBILITY_SCREEN_V1",
        "method": "perfect-oracle ceiling = mean(max(long, short)) = mean(|move|) - mean(cost)",
        "outcome_blind": True,
        "min_ceiling_pips": min_ceiling_pips,
        "min_rows_per_cell": min_rows,
        "rows_used": len(rows),
        "cells": len(cells),
        "admitted": len(admitted),
        "closed": len(cells) - len(admitted),
        "interpretation": (
            "A ceiling is what a never-wrong predictor earns. FEASIBLE means profit is not "
            "forbidden, never that it is available. Cells marked IMPOSSIBLE cannot be rescued "
            "by any signal, threshold, sizing rule or stop design."
        ),
        "cells_detail": [cell.as_dict() for cell in cells],
    }

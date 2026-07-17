"""Add-on ladder shadow evaluator: pyramiding and bounded nanpin (W22).

Re-scores an already-filled trade under pre-declared add-on rules on exact
S5 executable opens: PYRAMID adds one unit each time price moves favorably
by the declared step; NANPIN adds one unit each adverse step.  Adds are
strictly bounded (max_adds sealed in the rule; unbounded martingale is
forbidden), every unit exits at the base trade's original exit price, and
fills happen only on real S5 opens crossing the level — no synthesis.
Outputs blended pips per unit and the peak exposure multiple so risk is
never hidden by averaging.
"""

from __future__ import annotations

import bisect
import math
from typing import Any

from quant_rabbit.adaptive_exact_s5_profit_engine import ExactS5Series, TradeOutcome

POLICY = "BOUNDED_ADDON_LADDER_S5_OPEN_FILLS_V1"
MAX_ADDS_HARD_CAP = 3


def resolve_addon_ladder(
    series: ExactS5Series,
    outcome: TradeOutcome,
    *,
    mode: str,
    step_pips: float,
    max_adds: int,
    pip_factor: float,
) -> dict[str, Any]:
    """Blend the base fill with bounded ladder adds on real S5 opens."""

    if mode not in {"PYRAMID", "NANPIN"}:
        raise ValueError("mode must be PYRAMID or NANPIN")
    if (
        isinstance(max_adds, bool)
        or not isinstance(max_adds, int)
        or not 0 < max_adds <= MAX_ADDS_HARD_CAP
    ):
        raise ValueError("max_adds must be within the sealed hard cap")
    if not isinstance(step_pips, (int, float)) or float(step_pips) <= 0:
        raise ValueError("step_pips must be positive")
    factor = float(pip_factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("pip_factor must be positive")
    step = float(step_pips) / factor
    side = outcome.side
    entry_epoch = int(outcome.entry_utc.timestamp())
    exit_epoch = int(outcome.exit_utc.timestamp())
    base_entry = outcome.entry_ask if side == "LONG" else outcome.entry_bid
    exit_price = outcome.exit_bid if side == "LONG" else outcome.exit_ask

    unit_entries = [base_entry]
    start = bisect.bisect_right(series.s5_epochs, entry_epoch)
    position = start
    while position < len(series.s5_epochs) and len(unit_entries) <= max_adds:
        epoch = int(series.s5_epochs[position])
        if epoch >= exit_epoch:
            break
        add_index = len(unit_entries)
        if side == "LONG":
            trigger = (
                base_entry + add_index * step
                if mode == "PYRAMID"
                else base_entry - add_index * step
            )
            executable = float(series.ask_opens[position])
            crossed = (
                executable >= trigger if mode == "PYRAMID" else executable <= trigger
            )
        else:
            trigger = (
                base_entry - add_index * step
                if mode == "PYRAMID"
                else base_entry + add_index * step
            )
            executable = float(series.bid_opens[position])
            crossed = (
                executable <= trigger if mode == "PYRAMID" else executable >= trigger
            )
        if crossed:
            unit_entries.append(executable)
        position += 1

    per_unit = [
        (exit_price - unit) * factor if side == "LONG" else (unit - exit_price) * factor
        for unit in unit_entries
    ]
    return {
        "policy": POLICY,
        "mode": mode,
        "step_pips": float(step_pips),
        "max_adds": max_adds,
        "units_filled": len(unit_entries),
        "peak_exposure_multiple": len(unit_entries),
        "base_realized_pips": round(outcome.realized_pips, 9),
        "blended_total_pips": round(sum(per_unit), 9),
        "blended_pips_per_unit": round(sum(per_unit) / len(per_unit), 9),
        "unbounded_martingale": False,
        "order_authority": "NONE",
    }

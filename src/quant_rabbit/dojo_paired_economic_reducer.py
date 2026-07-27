"""Terminal economic reducer for the 84-cell paired model queue.

The reducer consumes only replay results where one accepted, content-addressed
model response was applied to each coordinate/cadence cell.  It never opens a
future decision packet and has no broker, model, network, or live side effect.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from quant_rabbit.dojo_paired_inventory_counterfactual import (
    AUTHORITY,
    CADENCE_IDS,
    RESULT_CONTRACT,
)
from quant_rabbit.dojo_paired_model_queue import canonical_sha256
from quant_rabbit.dojo_portfolio_replay_reducer import canonical_portfolio_sha256


CONTRACT: Final = "QR_DOJO_PAIRED_MODEL_ECONOMIC_REDUCER_V1"
SCHEMA_VERSION: Final = 1
EXPECTED_COORDINATE_COUNT: Final = 12
EXPECTED_CELL_COUNT: Final = 84


class DojoPairedEconomicReducerError(ValueError):
    """The applied economic denominator or decomposition is invalid."""


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoPairedEconomicReducerError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise DojoPairedEconomicReducerError(f"{label} is outside bounds")
    return number


def reduce_paired_model_economics(
    result_values: Sequence[Mapping[str, Any]],
    *,
    ai_execution_cost_jpy: float | None,
) -> dict[str, Any]:
    """Aggregate the fixed denominator after all accepted actions are replayed."""

    if len(result_values) != EXPECTED_COORDINATE_COUNT:
        raise DojoPairedEconomicReducerError(
            "exactly 12 applied coordinate results are required"
        )
    cells: list[dict[str, Any]] = []
    coordinate_ids: set[str] = set()
    for raw_result in result_values:
        result = dict(raw_result)
        unsigned = {
            key: item for key, item in result.items() if key != "result_sha256"
        }
        coordinate_id = result.get("coordinate_id")
        cadence_rows = result.get("cadence_rows")
        if (
            result.get("contract") != RESULT_CONTRACT
            or result.get("result_sha256") != canonical_portfolio_sha256(unsigned)
            or result.get("economic_application_status")
            != "APPLIED_ACCEPTED_MODEL_RESPONSES"
            or result.get("applied_model_response_count") != len(CADENCE_IDS)
            or result.get("authority") != dict(AUTHORITY)
            or not isinstance(coordinate_id, str)
            or coordinate_id in coordinate_ids
            or not isinstance(cadence_rows, list)
            or len(cadence_rows) != len(CADENCE_IDS)
            or {row.get("cadence_id") for row in cadence_rows}
            != set(CADENCE_IDS)
        ):
            raise DojoPairedEconomicReducerError(
                "applied coordinate result boundary is invalid"
            )
        coordinate_ids.add(coordinate_id)
        for row in cadence_rows:
            decomposition = row.get("economic_decomposition")
            if (
                not isinstance(decomposition, Mapping)
                or row.get("provider_model_call_count") != 1
                or row.get(
                    "phase_b_actual_model_checkpoint_decisions_measured"
                )
                is not True
            ):
                raise DojoPairedEconomicReducerError(
                    "one applied provider decision per cell is required"
                )
            cell = {
                "coordinate_id": coordinate_id,
                "family_id": result["family_id"],
                "cost_scenario": result["cost_scenario"],
                "cadence_id": row["cadence_id"],
                "tp_gross_profit_jpy": _number(
                    decomposition.get("tp_gross_profit_jpy"),
                    "tp_gross_profit_jpy",
                    minimum=0.0,
                ),
                "other_exit_gross_profit_jpy": _number(
                    decomposition.get("other_exit_gross_profit_jpy"),
                    "other_exit_gross_profit_jpy",
                    minimum=0.0,
                ),
                "normal_loss_jpy": _number(
                    decomposition.get("normal_loss_jpy"),
                    "normal_loss_jpy",
                    minimum=0.0,
                ),
                "forced_margin_loss_jpy": _number(
                    decomposition.get("forced_margin_loss_jpy"),
                    "forced_margin_loss_jpy",
                    minimum=0.0,
                ),
                "missed_profit_jpy": _number(
                    decomposition.get("missed_profit_vs_bot_jpy_full_run"),
                    "missed_profit_jpy",
                    minimum=0.0,
                ),
                "execution_cost_jpy": _number(
                    decomposition.get("execution_cost_jpy_full_run"),
                    "execution_cost_jpy",
                    minimum=0.0,
                ),
                "financing_cost_jpy": _number(
                    decomposition.get("financing_cost_jpy_full_run"),
                    "financing_cost_jpy",
                    minimum=0.0,
                ),
                "winning_ai_cut_count": int(
                    decomposition.get("winning_ai_cut_count", -1)
                ),
                "bot_only_net_jpy": _number(
                    row.get("bot_only_full_month_net_jpy"),
                    "bot_only_net_jpy",
                ),
                "ai_managed_net_jpy": _number(
                    row.get("ai_managed_full_month_net_jpy"),
                    "ai_managed_net_jpy",
                ),
            }
            if cell["winning_ai_cut_count"] < 0:
                raise DojoPairedEconomicReducerError(
                    "winning_ai_cut_count is invalid"
                )
            cells.append(cell)
    if (
        len(cells) != EXPECTED_CELL_COUNT
        or len({(row["coordinate_id"], row["cadence_id"]) for row in cells})
        != EXPECTED_CELL_COUNT
    ):
        raise DojoPairedEconomicReducerError(
            "exact 12 x 7 economic denominator is required"
        )
    ai_cost = (
        None
        if ai_execution_cost_jpy is None
        else _number(ai_execution_cost_jpy, "ai_execution_cost_jpy", minimum=0.0)
    )
    cost_per_cell = None if ai_cost is None else ai_cost / EXPECTED_CELL_COUNT
    for cell in cells:
        objective_before_ai_cost = (
            cell["tp_gross_profit_jpy"]
            + cell["other_exit_gross_profit_jpy"]
            - cell["normal_loss_jpy"]
            - cell["forced_margin_loss_jpy"]
            - cell["missed_profit_jpy"]
            - cell["execution_cost_jpy"]
            - cell["financing_cost_jpy"]
        )
        cell["ai_execution_cost_jpy"] = cost_per_cell
        cell["objective_before_ai_cost_jpy"] = objective_before_ai_cost
        cell["objective_after_ai_cost_jpy"] = (
            None
            if cost_per_cell is None
            else objective_before_ai_cost - cost_per_cell
        )
        cell["additional_reduction_to_positive_jpy"] = (
            None
            if cell["objective_after_ai_cost_jpy"] is None
            else max(0.0, -cell["objective_after_ai_cost_jpy"])
        )
    totals = {
        key: sum(float(row[key]) for row in cells)
        for key in (
            "tp_gross_profit_jpy",
            "other_exit_gross_profit_jpy",
            "normal_loss_jpy",
            "forced_margin_loss_jpy",
            "missed_profit_jpy",
            "execution_cost_jpy",
            "financing_cost_jpy",
            "bot_only_net_jpy",
            "ai_managed_net_jpy",
            "objective_before_ai_cost_jpy",
        )
    }
    totals["ai_execution_cost_jpy"] = ai_cost
    totals["objective_after_ai_cost_jpy"] = (
        None
        if ai_cost is None
        else totals["objective_before_ai_cost_jpy"] - ai_cost
    )
    totals["additional_reduction_to_positive_jpy"] = (
        None
        if totals["objective_after_ai_cost_jpy"] is None
        else max(0.0, -totals["objective_after_ai_cost_jpy"])
    )
    totals["winning_ai_cut_count"] = sum(
        int(row["winning_ai_cut_count"]) for row in cells
    )
    body = {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "economic_application_status": "APPLIED_ACCEPTED_MODEL_RESPONSES",
        "fixed_coordinate_count": EXPECTED_COORDINATE_COUNT,
        "fixed_cadence_count": len(CADENCE_IDS),
        "fixed_cell_count": EXPECTED_CELL_COUNT,
        "formula": (
            "TP_GROSS_PROFIT + OTHER_EXIT_GROSS_PROFIT - NORMAL_LOSS - "
            "FORCED_MARGIN_LOSS - MISSED_PROFIT - EXECUTION_COST - "
            "FINANCING_COST - AI_EXECUTION_COST"
        ),
        "aggregation_scope": (
            "FIXED_DENOMINATOR_DIAGNOSTIC_SUM; INDEPENDENT ACCOUNTS ARE NOT "
            "ONE TRADEABLE CAPITAL ACCOUNT"
        ),
        "profitability_status": (
            "UNDETERMINED_AI_COST_MISSING"
            if ai_cost is None
            else "DETERMINED_RESEARCH_ONLY"
        ),
        "totals": totals,
        "cells": sorted(
            cells,
            key=lambda row: (row["coordinate_id"], row["cadence_id"]),
        ),
        "profit_guaranteed": False,
        "promotion_eligible": False,
        "authority": dict(AUTHORITY),
    }
    return {**body, "reducer_sha256": canonical_sha256(body)}


__all__ = [
    "CONTRACT",
    "DojoPairedEconomicReducerError",
    "reduce_paired_model_economics",
]

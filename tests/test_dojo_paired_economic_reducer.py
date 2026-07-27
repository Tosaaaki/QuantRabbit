from __future__ import annotations

import copy

import pytest

from quant_rabbit.dojo_paired_economic_reducer import (
    DojoPairedEconomicReducerError,
    reduce_paired_model_economics,
)
from quant_rabbit.dojo_paired_inventory_counterfactual import (
    AUTHORITY,
    CADENCE_IDS,
    RESULT_CONTRACT,
)
from quant_rabbit.dojo_portfolio_replay_reducer import canonical_portfolio_sha256


def _results() -> list[dict]:
    results = []
    for coordinate in range(12):
        rows = []
        for cadence in CADENCE_IDS:
            rows.append(
                {
                    "cadence_id": cadence,
                    "provider_model_call_count": 1,
                    "phase_b_actual_model_checkpoint_decisions_measured": True,
                    "bot_only_full_month_net_jpy": -100.0,
                    "ai_managed_full_month_net_jpy": 25.0,
                    "economic_decomposition": {
                        "tp_gross_profit_jpy": 200.0,
                        "other_exit_gross_profit_jpy": 10.0,
                        "normal_loss_jpy": 80.0,
                        "forced_margin_loss_jpy": 0.0,
                        "missed_profit_vs_bot_jpy_full_run": 5.0,
                        "execution_cost_jpy_full_run": 50.0,
                        "financing_cost_jpy_full_run": 5.0,
                        "winning_ai_cut_count": 1,
                    },
                }
            )
        body = {
            "contract": RESULT_CONTRACT,
            "schema_version": 1,
            "coordinate_id": f"{coordinate:064x}",
            "family_id": f"family-{coordinate}",
            "cost_scenario": "BASE" if coordinate % 2 == 0 else "STRESS",
            "economic_application_status": "APPLIED_ACCEPTED_MODEL_RESPONSES",
            "applied_model_response_count": 7,
            "cadence_rows": rows,
            "authority": dict(AUTHORITY),
        }
        results.append(
            {**body, "result_sha256": canonical_portfolio_sha256(body)}
        )
    return results


def test_reducer_applies_fixed_84_cell_formula() -> None:
    reduced = reduce_paired_model_economics(
        _results(),
        ai_execution_cost_jpy=84.0,
    )

    assert reduced["fixed_cell_count"] == 84
    assert reduced["economic_application_status"] == (
        "APPLIED_ACCEPTED_MODEL_RESPONSES"
    )
    assert reduced["profitability_status"] == "DETERMINED_RESEARCH_ONLY"
    assert reduced["totals"]["tp_gross_profit_jpy"] == 16_800.0
    assert reduced["totals"]["ai_execution_cost_jpy"] == 84.0
    assert reduced["totals"]["winning_ai_cut_count"] == 84
    assert reduced["totals"]["objective_after_ai_cost_jpy"] == 5_796.0
    assert reduced["totals"]["additional_reduction_to_positive_jpy"] == 0.0
    assert reduced["authority"]["order_authority"] == "NONE"


def test_reducer_marks_profitability_unknown_without_ai_cost() -> None:
    reduced = reduce_paired_model_economics(
        _results(),
        ai_execution_cost_jpy=None,
    )

    assert reduced["profitability_status"] == "UNDETERMINED_AI_COST_MISSING"
    assert reduced["totals"]["objective_after_ai_cost_jpy"] is None
    assert reduced["totals"]["additional_reduction_to_positive_jpy"] is None


def test_reducer_rejects_missing_or_unapplied_cell() -> None:
    results = _results()
    broken = copy.deepcopy(results)
    broken[0]["cadence_rows"][0]["provider_model_call_count"] = 0
    body = {
        key: value
        for key, value in broken[0].items()
        if key != "result_sha256"
    }
    broken[0]["result_sha256"] = canonical_portfolio_sha256(body)

    with pytest.raises(
        DojoPairedEconomicReducerError,
        match="one applied provider decision",
    ):
        reduce_paired_model_economics(
            broken,
            ai_execution_cost_jpy=1.0,
        )

from __future__ import annotations

import math
from pathlib import Path

import pytest

import derived_pair_audit_runner_v1 as derived
import profit_gate_cause_feasibility_audit_v1 as audit


ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def payload() -> dict:
    return audit.validate(ROOT)


def test_audit_is_read_only_paper_evidence_not_profit_admission(payload: dict) -> None:
    assert payload["classification"] == "NON_STRATEGY_READ_ONLY_CAUSE_FEASIBILITY_EVIDENCE"
    assert payload["holdout_state"] == "UNOPENED"
    assert payload["authority"] == audit.AUTHORITY
    assert payload["external_orders"] == 0
    assert payload["official_strategy_run_performed"] is False
    assert payload["strategy_adoption_authorized"] is False
    assert payload["profit_gate_pass_inferred"] is False
    assert payload["protected_strategy_artifact_hashes"]["unchanged"] is True


def test_thirteen_valid_seals_six_streams_and_invalid_v34_are_primary_pair_evidence(
    payload: dict,
) -> None:
    direct = payload["pair_direct_readback"]
    assert direct["valid_sealed_cycle_count"] == 13
    assert direct["unique_raw_signal_stream_count"] == 6
    assert direct["independent_pair_specialized_strategy_cycle_count"] == 0
    assert direct["pair_signal_id_dedupe"] == derived.EXPECTED_DEDUPE
    assert direct["invalid_v34"]["metrics_admissible"] is False
    assert direct["invalid_v34"]["official_seal_exists"] is False
    v25 = direct["v25_v27_direct_checks"]["V25"]
    assert v25["EUR_USD"]["direction_accuracy"] == 19 / 35
    assert v25["EUR_USD"]["jpy_contribution_legacy_sealed_convention"] \
        ["EXECUTABLE_BASE"] > 0
    assert v25["EUR_USD"]["jpy_contribution_legacy_sealed_convention"] \
        ["ADVERSE_STRESS"] < 0
    assert v25["USD_JPY"]["direction_accuracy"] == 16 / 29
    assert v25["USD_JPY"]["jpy_contribution_legacy_sealed_convention"] \
        ["EXECUTABLE_BASE"] < 0


def test_range_resource_exists_but_current_1x_oracle_capacity_is_insufficient(
    payload: dict,
) -> None:
    evidence = payload["completed_daily_range_and_oracle_feasibility"]
    envelope = payload["next_feasibility_envelope"]
    assert evidence["current_1x_required_capture_exceeds_full_daily_range"] is True
    assert envelope["current_1x_capacity_insufficient"] is True
    assert envelope["entry_or_exit_tuning_alone_can_close_current_1x_gap"] is False
    assert envelope["fixed_gross_cap_grid"] == [1.0, 4.0, 8.0, 12.0, 20.0]
    for month, expected in derived.EXPECTED_ORACLE_CAPTURE_PERCENT.items():
        solutions = envelope["normal_oracle_cap_grid_by_month"][month]
        for cap, percent in expected.items():
            assert math.isclose(
                solutions[cap]["required_daily_high_low_capture_percent"],
                percent,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            assert solutions[cap]["strategy_evidence"] is False
    assert envelope["perfect_full_range_normal_cost_cap20_ceiling_by_month"]["2026-05"] > 2
    assert envelope["perfect_full_range_normal_cost_cap20_ceiling_by_month"]["2026-06"] > 2
    assert envelope["adverse_or_realistic_lcb_result_inferred"] is False
    assert envelope["gross_cap_change_authorized"] is False
    assert envelope["current_execution_authorized"] is False


def test_graph_residual_and_dst_only_v42_are_no_go(payload: dict) -> None:
    graph = payload["graph_identifiability"]
    gate = payload["v42_go_no_go"]
    assert graph["cycle_space_dimension"] == 0
    assert graph["overidentifying_residual_degrees_of_freedom"] == 0
    assert graph["candidate_1_status"] == "REJECTED_STRUCTURALLY_WITHOUT_RETURN_OR_COST_DATA"
    assert graph["profit_or_cost_outcomes_used_for_rejection"] is False
    assert gate["dst_only_strategy_execution"] == "NO_GO"
    assert gate["dst_role"] == "CHRONOLOGY_FOUNDATION_ONLY_NOT_NEW_EDGE"
    assert gate["current_official_v42_execution_authorized"] is False
    assert gate["selected_family_id"] == (
        "EUR_USD_H4_REGIME_TO_M15_ENTRY_TIMING_HIERARCHICAL"
    )
    assert gate["pre_fixed_improvement_gates"] == {
        "maximum_adverse_cost_to_expected_move_ratio": 0.5,
        "minimum_gross_edge_density_product_multiplier": 4.0,
        "minimum_gross_edge_per_independent_bet_multiplier": 2.0,
        "minimum_independent_bet_density_multiplier": 2.0,
    }
    assert gate["small_positive_result_disposition"] == (
        "COMPONENT_CANDIDATE_ONLY_NOT_PROFIT_ADMISSION"
    )


def test_future_geometric_gate_is_formula_only_and_historical_gate_still_fails(
    payload: dict,
) -> None:
    future = payload["future_geometric_mean_gate"]
    assert math.isclose(future["example_computed_geometric_mean"], 2.0,
                        rel_tol=0.0, abs_tol=1e-15)
    assert future["executable"] is False
    assert future["may_register_or_execute_future_profit_gate"] is False
    assert set(future["blocking_unspecified_fields"]) == {
        "near_target_tolerance", "evaluation_month_count", "worst_month_floor",
        "maximum_drawdown_guard", "margin_guard", "ruin_guard",
    }
    for entity in payload["full_month_summaries"].values():
        assert entity["EXECUTABLE_BASE"]["all_full_months_at_least_2x"] is False
        assert entity["ADVERSE_STRESS"]["all_full_months_at_least_2x"] is False
    accounting = payload["legacy_accounting_read_only_diagnostic"]
    assert accounting["correction_reveals_hidden_2x"] is False
    assert all(item["corrected_both_below_2x"] for item in accounting["cycles"].values())


def test_v253_is_reproducible_but_does_not_identify_incremental_llm_edge(
    payload: dict,
) -> None:
    v253 = payload["legacy_actual_llm_read_only_evidence"][
        "v253_development_walk_inventory_policy"
    ]
    assert v253["same_policy_weights"] is True
    assert v253["bot_gross_cap"] == 1.0
    assert v253["actual_llm_gross_cap"] == 12.0
    assert v253["same_gross_cap"] is False
    assert v253["same_cap_bot_control_present"] is False
    assert v253["same_cap_mechanical_validation_control_present"] is False
    assert v253["incremental_llm_edge_identified"] is False
    assert v253["actual_llm"]["normal"]["total_multiple"] == 1.093389101612891
    assert v253["actual_llm"]["adverse"]["total_multiple"] == 1.0379005274152473
    assert v253["actual_llm"]["normal"]["months_at_or_above_2x"] == 0
    assert v253["actual_llm"]["adverse"]["months_at_or_above_2x"] == 0
    aud = v253["aud_usd_development_diagnostic"]
    assert aud["episodes"] == 8
    assert aud["direction_accuracy"] == 0.875
    assert aud["small_sample"] is True
    assert aud["llm_selected_individual_directions"] is False


def test_v15_v18_v13_and_legacy_jpy_profits_are_not_misattributed(payload: dict) -> None:
    evidence = payload["legacy_actual_llm_read_only_evidence"]
    v15 = evidence["v15_jpy_portfolio_veto"]
    assert v15["normal"]["net_pnl_jpy"] > 0
    assert v15["stress_3x"]["net_pnl_jpy"] < 0
    assert v15["aud_jpy_episode_count"] == 1
    assert v15["aud_jpy_exit_reason"] == "END_OF_DATA"
    assert v15["single_episode_may_support_edge_claim"] is False
    v18 = evidence["v18_aud_jpy_counterexample"]
    assert v18["normal_net_pnl_jpy"] < 0
    assert v18["stress_3x_net_pnl_jpy"] < 0
    v13 = evidence["v13_usd_major_inventory_policy"]
    assert v13["contained_walk_forward_signals"] == 370
    assert v13["contained_additive_returns"]["RAW_SIGNAL"] > 0
    assert v13["contained_additive_returns"]["EXECUTABLE_BASE"] > 0
    assert v13["contained_additive_returns"]["ADVERSE_STRESS"] < 0
    assert evidence["ai_test_bot_deterministic_not_actual_llm"][
        "actual_llm_attribution_proven"
    ] is False
    assert evidence["broker_cohort_64d_101_trades"][
        "actual_llm_or_codex_attribution_proven"
    ] is False
    assert evidence["current_v25_v41_actual_llm_status"][
        "machine_verified_actual_llm_strategy_arm_present"
    ] is False
    assert evidence["profit_or_adoption_claim_allowed"] is False


def test_llm_next_diagnostic_is_same_cap_and_forward_shadow_remains_disconnected(
    payload: dict,
) -> None:
    evidence = payload["legacy_actual_llm_read_only_evidence"]
    diagnostic = evidence["diagnostic_priority_after_bot_pair_repair_plumbing"][0]
    assert diagnostic["fixed_gross_cap"] == 12.0
    assert diagnostic["arms"] == [
        "BOT_FIXED_SAME_CAP",
        "MECHANICAL_VALIDATION_RULE_SAME_CAP",
        "FROZEN_ACTUAL_LLM_SAME_CAP",
    ]
    assert diagnostic["profit_search"] is False
    forward = payload["next_research_matrix"]["historical_and_forward_shadow"]
    assert forward["forward_shadow_start_authorized"] is False
    assert forward["forward_shadow"]["practice_market_data_token_accessed"] is False
    core = forward["credential_free_shadow_core_candidate"]
    assert core["network_connection_authorized"] is False
    assert "CREDENTIAL_LOADER" in core["forbidden_dependencies"]
    assert core["offline_drop_in_formats"] == ["CSV", "JSONL"]
    assert payload["v42_go_no_go"]["existing_v42_work_order_sha256"] == (
        "29d541646f57efffe543007d94ce0958a2fa3cc68180cf521101886a8b09b524"
    )

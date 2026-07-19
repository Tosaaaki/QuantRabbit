from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHA256_HEX_LENGTH = 64


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _assert_ai_exit_score_reaggregates(evidence: dict) -> None:
    cells = evidence["cells"]
    score = evidence["score"]
    assert len({cell["id"] for cell in cells}) == len(cells)
    assert len({cell["packet_canonical_sha256"] for cell in cells}) == len(cells)
    assert all(
        len(cell["packet_canonical_sha256"]) == SHA256_HEX_LENGTH for cell in cells
    )
    for cell in cells:
        expected = (
            cell["cut_pips"] if cell["decision"] == "CUT_NOW" else cell["hold_pips"]
        )
        assert cell["ai_pips"] == expected
        oracle = "CUT_NOW" if cell["cut_pips"] >= cell["hold_pips"] else "HOLD"
        assert cell["oracle_decision"] == oracle
    assert score["cell_count"] == len(cells)
    assert score["unique_pair_count"] == len(
        {cell["pair_revealed_after_decision"] for cell in cells}
    )
    assert score["oracle_choice_hits"] == sum(
        cell["decision"] == cell["oracle_decision"] for cell in cells
    )
    assert score["ai_total_pips"] == round(sum(cell["ai_pips"] for cell in cells), 1)
    assert score["always_cut_total_pips"] == round(
        sum(cell["cut_pips"] for cell in cells), 1
    )
    assert score["always_hold_total_pips"] == round(
        sum(cell["hold_pips"] for cell in cells), 1
    )
    assert score["delta_vs_always_cut_pips"] == round(
        score["ai_total_pips"] - score["always_cut_total_pips"], 1
    )
    assert score["delta_vs_always_hold_pips"] == round(
        score["ai_total_pips"] - score["always_hold_total_pips"], 1
    )


def test_invalidated_ai_exit_train_v1_is_preserved_but_not_comparable() -> None:
    evidence = _load("research/training/dojo-ai-exit-train-v1/evidence.json")
    cells = evidence["cells"]

    assert evidence["classification"] == "INVALIDATED_SAME_BAR_LOOKAHEAD"
    assert evidence["proof_eligible"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["live_permission"] is False
    assert evidence["order_authority"] == "NONE"
    assert evidence["fresh_context_contract"]["one_judgment_per_context"] is True
    assert evidence["fresh_context_contract"]["context_count"] == len(cells) == 8
    assert evidence["fresh_context_contract"]["lookahead_free"] is False
    assert evidence["score"]["valid_for_policy_comparison"] is False
    assert evidence["score"]["absolute_profit_positive"] is False
    _assert_ai_exit_score_reaggregates(evidence)


def test_ai_exit_train_v2_is_lookahead_free_and_score_reaggregates() -> None:
    evidence = _load("research/training/dojo-ai-exit-train-v2/evidence.json")
    assert evidence["classification"] == "INVALIDATED_SOURCE_GAP_CONTINUITY"
    assert evidence["proof_eligible"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["live_permission"] is False
    assert evidence["order_authority"] == "NONE"
    assert evidence["fresh_context_contract"]["one_judgment_per_context"] is True
    assert (
        evidence["fresh_context_contract"]["context_count"]
        == len(evidence["cells"])
        == 8
    )
    assert evidence["lookahead_contract"] == {
        "decision_price_basis": "DECISION_BAR_OPEN",
        "m5_context_cutoff": "STRICTLY_BEFORE_DECISION_BAR",
        "decision_bar_high_low_close_excluded": True,
        "completed_m5_bar_count": 36,
        "completed_h1_close_count": 120,
        "regression_test": "tests/test_build_blind_exit_scenarios.py",
    }
    assert evidence["score"]["absolute_profit_positive"] is False
    assert evidence["score"]["delta_vs_always_cut_pips"] > 0
    assert evidence["score"]["delta_vs_always_hold_pips"] < 0
    assert evidence["score"]["valid_for_policy_comparison"] is False
    generator = ROOT / evidence["source"]["generator_path"]
    assert (
        hashlib.sha256(generator.read_bytes()).hexdigest()
        == (evidence["source"]["current_generator_file_sha256"])
    )
    correction = evidence["source"]["current_generator_correction_check"]
    assert correction["original_40_cell_set_bytes_match"] is False
    assert correction["evaluated_first_4_scenario_bytes_match"] is True
    assert correction["evaluated_first_4_answer_bytes_match"] is True
    assert correction["first_direct_gap_invalidated_cell_id"] == "X05"
    assert correction["downstream_evaluated_ids_with_cohort_identity_shift"] == [
        "X06",
        "X07",
        "X08",
    ]
    assert correction["evaluated_decisions_and_scores_reuse_allowed"] is False
    assert correction["entry_bar_tp_touch_fixed"] is True
    assert correction["decision_bar_tp_touch_fixed"] is True
    _assert_ai_exit_score_reaggregates(evidence)


def test_worker_multipair_train_index_has_no_promoted_or_positive_group() -> None:
    evidence = _load(
        "research/training/dojo-worker-multipair-train-20260719/evidence.json"
    )
    results = evidence["results"]

    assert evidence["classification"] == "TRAIN_DIAGNOSTIC_ONLY"
    assert evidence["proof_eligible"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["live_permission"] is False
    assert evidence["order_authority"] == "NONE"
    assert evidence["candidate_group_count"] == len(results) == 4
    assert evidence["strategy_family_count"] == 3
    assert all(row["pessimistic_trade_count"] > 0 for row in results)
    assert all(row["pessimistic_pnl_jpy"] < 0 for row in results)
    assert all(row["positive_after_cost"] is False for row in results)
    assert all(row["survivor"] is False for row in results)
    assert all(len(row["artifact_file_sha256"]) == SHA256_HEX_LENGTH for row in results)
    aggregate = evidence["aggregate_classification"]
    assert aggregate["survivor_count"] == 0
    assert aggregate["all_candidate_groups_negative_after_declared_cost"] is True
    assert aggregate["monthly_3x_reached"] is False


def test_ai_supervised_worker_tuning_keeps_train_survivors_out_of_proof() -> None:
    evidence = _load("research/training/dojo-worker-ai-tuning-20260719/evidence.json")
    batches = {row["batch_id"]: row for row in evidence["batches"]}

    assert evidence["classification"] == "WORN_HISTORICAL_TRAIN_ONLY"
    assert evidence["proof_eligible"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["live_permission"] is False
    assert evidence["order_authority"] == "NONE"
    assert evidence["trainer_loop"]["candidate_replays_actual_historical_market"]
    assert evidence["trainer_loop"]["goal_backsolving_allowed"] is False

    tailguard = batches["SPIKE_FADE_TAILGUARD_V1"]
    assert tailguard["train_survivor_count"] == 2
    assert tailguard["fill_time_concurrency_audit"]["breach_count"] == 0
    assert all(
        candidate["classification"] == "TRAIN_SURVIVOR"
        for candidate in tailguard["candidates"]
    )
    pullback = batches["LOW_FREQUENCY_PULLBACK_A2"]
    assert pullback["train_survivor_count"] == 1
    assert pullback["pessimistic_calendar_30d_multiple"] < 1.001
    momentum = batches["MOMENTUM_SELECTIVITY_V1"]
    assert momentum["train_survivor_count"] == 0
    assert all(row["pessimistic_pnl_jpy"] < 0 for row in momentum["candidates"])
    diversity = batches["BOT_DIVERSITY_V1_PRE_CAP_FIX"]
    assert diversity["classification"] == (
        "INVALIDATED_FILL_TIME_CONCURRENCY_CAP_NOT_ENFORCED"
    )
    assert diversity["rerun_required"] is False
    diversity_fixed = batches["BOT_DIVERSITY_V1_CAPS_FIXED"]
    assert diversity_fixed["train_survivor_count"] == 0
    assert diversity_fixed["fill_time_concurrency_audit"]["breach_count"] == 0
    assert diversity_fixed["fill_time_concurrency_audit"]["pair_cap_rejections"] > 0
    assert all(row["pessimistic_pnl_jpy"] < 0 for row in diversity_fixed["candidates"])
    followup = batches["ROUND_NUMBER_FADE_TP3_FOLLOWUP"]
    assert followup["classification"] == "REJECTED_TRAIN_HYPOTHESIS"
    assert followup["parameter_diff"]["risk_increase"] is False
    assert followup["fill_time_concurrency_audit"]["breach_count"] == 0
    assert all(row["pnl_jpy"] < 0 for row in followup["intrabar_results"])
    capital = batches["CAPITAL_HOLD_OPPORTUNITY_V1"]
    policies = {row["candidate_id"]: row for row in capital["policies"]}
    assert capital["diagnostic_winner"] == "full_time_release"
    assert capital["audit_gates"]["owner_pair_or_global_cap_breaches"] == 0
    assert capital["audit_gates"]["margin_closeouts"] == 0
    assert policies["full_time_release"]["pnl_jpy"] > policies["full_hold"]["pnl_jpy"]
    assert (
        policies["full_time_release"]["realized_drawdown_fraction"]
        < (policies["full_hold"]["realized_drawdown_fraction"])
    )
    assert policies["split_reserve_hold"]["insufficient_margin_rejections"] == 0
    assert all(row["classification"] == "TRAIN_SURVIVOR" for row in policies.values())

    aggregate = evidence["aggregate"]
    assert aggregate["valid_batch_count"] == 6
    assert aggregate["train_survivor_count"] == 6
    assert aggregate["proof_survivor_count"] == 0
    assert aggregate["monthly_3x_reached"] is False
    assert aggregate["worn_train_monthly_3x_diagnostic_reached"] is True
    assert aggregate["monthly_3x_proof_reached"] is False
    assert all(
        len(row["artifact_file_sha256"]) == SHA256_HEX_LENGTH
        for row in evidence["batches"]
    )


def test_ai_exit_evidence_digests_match_goal_board_bindings() -> None:
    board_input = _load("research/registries/dojo_goal_board_input_20260719.json")
    lanes = {lane["lane_id"]: lane for lane in board_input["lanes"]}
    for version, expected_status, expected_lookahead in (
        ("v1", "INVALID", False),
        ("v2", "INVALID", True),
    ):
        path = ROOT / f"research/training/dojo-ai-exit-train-{version}/evidence.json"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        lane = lanes[f"ai_exit_train_{version}_diagnostic"]
        assert lane["status"] == expected_status
        assert lane["provenance"]["content_sha256"] == actual
        assert lane["provenance"]["lookahead_free"] is expected_lookahead
        assert lane["provenance"]["prospective"] is False


def test_ai_capital_recycle_separates_direction_gate_from_failed_allocation() -> None:
    evidence = _load("research/training/dojo-ai-capital-recycle-train-v1/evidence.json")
    score = evidence["score"]
    cells = evidence["cells"]

    assert evidence["classification"] == "WORN_TRAIN_REJECTED_DIAGNOSTIC"
    assert evidence["proof_eligible"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["live_permission"] is False
    assert evidence["causal_contract"]["one_judgment_per_fresh_context"] is True
    assert evidence["causal_contract"][
        "answer_key_generated_after_all_responses_sealed"
    ]
    assert len(cells) == score["cell_count"] == 6
    assert score["existing_direction_gate_hits"] == 6
    assert score["next_direction_hits"] == 3
    assert score["ai_total_capacity_pips"] == round(
        sum(row["ai_capacity_pips"] for row in cells), 3
    )
    assert score["always_hold_full_total_capacity_pips"] == round(
        sum(row["always_hold_capacity_pips"] for row in cells), 2
    )
    assert score["delta_vs_always_hold_capacity_pips"] < 0
    assert score["absolute_profit_positive"] is False
    assert evidence["posthoc_next_hypothesis"]["eligible_as_result"] is False

    board_input = _load("research/registries/dojo_goal_board_input_20260719.json")
    lanes = {lane["lane_id"]: lane for lane in board_input["lanes"]}
    path = ROOT / "research/training/dojo-ai-capital-recycle-train-v1/evidence.json"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    for lane_id in (
        "worker_ai_capital_recycle_substrate_train_v1",
        "ai_capital_recycle_train_v1",
    ):
        assert lanes[lane_id]["provenance"]["content_sha256"] == actual


def test_v3r1_supersession_stale_positive_is_explicitly_invalidated() -> None:
    correction = _load("research/corrections/dojo-ai-forward-v3r1-supersession-v1.json")
    invalidated = correction["invalidated_evidence"]
    supersession = _load(
        "research/forward/dojo-ai-forward-phase1-v3r1/supersession.json"
    )
    replacement_path = ROOT / invalidated["replacement_evidence_path"]
    replacement = json.loads(replacement_path.read_text(encoding="utf-8"))
    stale_rows = {row["path"]: row["result"] for row in supersession["evidence"]}
    assert (
        correction["applies_to_artifact_sha256"]
        == (supersession["supersession_sha256"])
    )
    assert stale_rows[invalidated["path"]] == invalidated["stale_result"]
    assert (
        hashlib.sha256(replacement_path.read_bytes()).hexdigest()
        == (invalidated["replacement_evidence_sha256"])
    )
    assert replacement["score"]["verdict"] == invalidated["replacement_result"]
    assert correction["successor_policy_selection_evidence_valid"] is False
    assert correction["supersession_state_remains_valid"] is True
    assert invalidated["status"] == "INVALIDATED_SAME_BAR_LOOKAHEAD"
    assert invalidated["replacement_result"] == (
        "PRESERVED_NUMBERS_INVALID_FOR_POLICY_COMPARISON"
    )
    assert correction["proof_eligible"] is False
    assert correction["live_permission"] is False

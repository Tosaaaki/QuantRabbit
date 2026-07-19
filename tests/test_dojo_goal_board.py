from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from quant_rabbit.dojo_goal_board import (
    INPUT_CONTRACT,
    DojoGoalBoardError,
    build_goal_board,
    canonical_sha256,
    load_goal_board_input,
    required_daily_return,
)


ROOT = Path(__file__).resolve().parents[1]


def _lane(
    lane_id: str,
    *,
    lane_type: str = "WORKER",
    status: str = "EDGE_PROVEN",
    digest_char: str = "a",
    cluster: str = "USDJPY_MEAN_REVERSION",
    parent_lane_ids: list[str] | None = None,
    parent_digests: dict[str, str] | None = None,
) -> dict:
    parents = parent_lane_ids or []
    return {
        "lane_id": lane_id,
        "lane_type": lane_type,
        "status": status,
        "parent_lane_ids": parents,
        "provenance": {
            "valid": status != "INVALID",
            "prospective": True,
            "lookahead_free": True,
            "content_sha256": digest_char * 64,
            "invalid_reasons": [] if status != "INVALID" else ["VOIDED_EVIDENCE"],
            "parent_digests": parent_digests or {},
        },
        "risk": {
            "mark_to_market": True,
            "bounded": True,
            "normal_mtm_max_drawdown_fraction": 0.08,
            "stressed_mtm_max_drawdown_fraction": 0.12,
        },
        "margin": {
            "peak_usage_fraction": 0.40,
            "cap_fraction": 0.45,
        },
        "correlation_cluster": cluster,
        "dependence": {
            "instrument": "USD_JPY",
            "strategy_family": "MEAN_REVERSION",
            "cohort_id": "USDJPY_FORWARD_COHORT_V1",
        },
        "distribution_30d": {
            "method": "PROSPECTIVE_FORWARD",
            "sample_months": 3,
            "active_days": 90 if lane_type == "AI" else 60,
            "stressed_median_multiple": 3.10,
            "post_cost_edge_lcb": 0.01,
            "probability_3x_lcb": 0.55,
            "probability_losing_month": 0.08,
            "probability_drawdown_20pct": 0.008,
            "probability_ruin_12m": 0.0008,
        },
        "sizing": {
            "observed_at_declared_size": True,
            "reverse_engineered_from_goal": False,
        },
    }


def _input(*lanes: dict) -> dict:
    return {"contract": INPUT_CONTRACT, "lanes": list(lanes)}


def _evaluation(board: dict, lane_id: str) -> dict:
    return next(row for row in board["lane_evaluations"] if row["lane_id"] == lane_id)


def test_required_daily_returns_match_closed_form() -> None:
    calendar = required_daily_return(days=30)
    trading = required_daily_return(days=22)

    assert calendar == pytest.approx(0.0372991973)
    assert trading == pytest.approx(0.0512047867)
    assert (1.0 + calendar) ** 30 == pytest.approx(3.0)
    assert (1.0 + trading) ** 22 == pytest.approx(3.0)


def test_current_voided_ai_and_diagnostic_worker_are_hypothesis_not_3x() -> None:
    worker = _lane("worker_current", status="HYPOTHESIS")
    worker["provenance"]["prospective"] = False
    worker["distribution_30d"].update(
        {
            "method": "BOOTSTRAP_DIAGNOSTIC",
            "sample_months": 0,
            "active_days": 54,
            "stressed_median_multiple": 1.036,
            "probability_3x_lcb": 0.0,
            "probability_losing_month": 0.28,
            "probability_drawdown_20pct": 0.01,
            "probability_ruin_12m": 0.0,
        }
    )
    ai = _lane(
        "ai_current_void",
        lane_type="AI",
        status="INVALID",
        digest_char="b",
        parent_lane_ids=["worker_current"],
        parent_digests={"worker_current": "a" * 64},
    )
    ai["provenance"]["invalid_reasons"] = ["PACKET_LOOKAHEAD_LEAK"]

    board = build_goal_board(_input(worker, ai))

    assert board["edge_status"] == "HYPOTHESIS"
    assert board["goal_status"] == "3X_NOT_REACHABLE"
    assert _evaluation(board, "worker_current")["edge_status"] == "HYPOTHESIS"
    assert _evaluation(board, "ai_current_void")["edge_status"] == "INVALID"
    assert board["guarantee"] is False
    assert board["live_permission"] is False
    assert board["order_authority"] == "NONE"


def test_untrusted_claim_cannot_reach_either_proof_axis() -> None:
    lane = _lane("worker_proven")

    board = build_goal_board(_input(lane))
    evaluation = _evaluation(board, "worker_proven")

    assert evaluation["edge_status"] == "HYPOTHESIS"
    assert evaluation["goal_status"] == "3X_NOT_REACHABLE"
    assert board["edge_status"] == "HYPOTHESIS"
    assert board["goal_status"] == "3X_NOT_REACHABLE"
    assert "TRUSTED_PROOF_ARTIFACT_REQUIRED" in evaluation["edge_blockers"]
    body = {key: value for key, value in board.items() if key != "board_sha256"}
    assert board["board_sha256"] == canonical_sha256(body)
    assert board["sizing_backsolve_allowed"] is False


@pytest.mark.parametrize(
    ("field", "value", "blocker"),
    [
        ("stressed_median_multiple", 2.99, "STRESSED_MEDIAN_BELOW_3X"),
        ("probability_3x_lcb", 0.49, "PROBABILITY_3X_LCB_BELOW_FLOOR"),
        ("probability_losing_month", 0.11, "LOSING_MONTH_PROBABILITY_TOO_HIGH"),
        (
            "probability_drawdown_20pct",
            0.011,
            "DRAWDOWN_20PCT_PROBABILITY_TOO_HIGH",
        ),
        ("probability_ruin_12m", 0.0011, "RUIN_12M_PROBABILITY_TOO_HIGH"),
    ],
)
def test_each_3x_threshold_separates_tail_from_goal(
    field: str, value: float, blocker: str
) -> None:
    lane = _lane("worker_tail")
    lane["distribution_30d"][field] = value

    evaluation = _evaluation(build_goal_board(_input(lane)), "worker_tail")

    assert evaluation["edge_status"] == "HYPOTHESIS"
    assert evaluation["goal_status"] == "3X_NOT_REACHABLE"
    assert blocker in evaluation["goal_blockers"]


def test_zero_probability_lower_bound_is_not_reachable() -> None:
    lane = _lane("worker_no_tail")
    lane["distribution_30d"]["probability_3x_lcb"] = 0.0

    evaluation = _evaluation(build_goal_board(_input(lane)), "worker_no_tail")

    assert evaluation["edge_status"] == "HYPOTHESIS"
    assert evaluation["goal_status"] == "3X_NOT_REACHABLE"


def test_reverse_engineered_sizing_can_never_make_goal_compatible() -> None:
    lane = _lane("worker_backsolved")
    lane["sizing"]["reverse_engineered_from_goal"] = True

    evaluation = _evaluation(build_goal_board(_input(lane)), "worker_backsolved")

    assert evaluation["edge_status"] == "HYPOTHESIS"
    assert evaluation["goal_status"] == "3X_NOT_REACHABLE"
    assert (
        "TARGET_REVERSE_ENGINEERED_SIZING_NOT_EVIDENCE" in evaluation["goal_blockers"]
    )


@pytest.mark.parametrize(
    ("section", "field", "value", "blocker"),
    [
        (
            "risk",
            "normal_mtm_max_drawdown_fraction",
            0.11,
            "NORMAL_MTM_DRAWDOWN_TOO_HIGH",
        ),
        (
            "risk",
            "stressed_mtm_max_drawdown_fraction",
            0.16,
            "STRESSED_MTM_DRAWDOWN_TOO_HIGH",
        ),
        ("margin", "peak_usage_fraction", 0.46, "MARGIN_CAP_BREACHED"),
        ("margin", "cap_fraction", 0.96, "MARGIN_CAP_EXCEEDS_ABSOLUTE_BOUND"),
    ],
)
def test_risk_and_margin_failure_downgrades_edge(
    section: str, field: str, value: float, blocker: str
) -> None:
    lane = _lane("worker_risk_fail")
    lane[section][field] = value
    if field == "normal_mtm_max_drawdown_fraction":
        lane["risk"]["stressed_mtm_max_drawdown_fraction"] = 0.16

    evaluation = _evaluation(build_goal_board(_input(lane)), "worker_risk_fail")

    assert evaluation["edge_status"] == "HYPOTHESIS"
    assert evaluation["goal_status"] == "3X_NOT_REACHABLE"
    assert blocker in evaluation["edge_blockers"]


def test_invalid_parent_fails_ai_and_board_closed() -> None:
    parent = _lane("worker_void", status="INVALID")
    ai = _lane(
        "ai_claim",
        lane_type="AI",
        status="EDGE_PROVEN",
        digest_char="b",
        parent_lane_ids=["worker_void"],
        parent_digests={"worker_void": "a" * 64},
    )

    board = build_goal_board(_input(parent, ai))
    ai_evaluation = _evaluation(board, "ai_claim")

    assert ai_evaluation["edge_status"] == "INVALID"
    assert ai_evaluation["goal_status"] == "3X_NOT_REACHABLE"
    assert "PARENT_INVALID:worker_void" in ai_evaluation["edge_blockers"]
    assert board["edge_status"] == "INVALID"
    assert board["goal_status"] == "3X_NOT_REACHABLE"


def test_ai_parent_digest_and_cluster_are_exact_bindings() -> None:
    parent = _lane("worker_parent")
    ai = _lane(
        "ai_overlay",
        lane_type="AI",
        digest_char="b",
        cluster="FOREIGN_CLUSTER",
        parent_lane_ids=["worker_parent"],
        parent_digests={"worker_parent": "f" * 64},
    )

    ai_evaluation = _evaluation(build_goal_board(_input(parent, ai)), "ai_overlay")

    assert ai_evaluation["edge_status"] == "INVALID"
    assert any(
        code.startswith("PARENT_PROVENANCE_DIGEST_MISMATCH:")
        for code in ai_evaluation["edge_blockers"]
    )
    assert any(
        code.startswith("PARENT_CORRELATION_CLUSTER_MISMATCH:")
        for code in ai_evaluation["edge_blockers"]
    )


def test_ai_requires_proven_parent_but_valid_binding_stays_hypothesis() -> None:
    parent = _lane("worker_parent", status="HYPOTHESIS")
    ai = _lane(
        "ai_overlay",
        lane_type="AI",
        digest_char="b",
        parent_lane_ids=["worker_parent"],
        parent_digests={"worker_parent": "a" * 64},
    )

    ai_evaluation = _evaluation(build_goal_board(_input(parent, ai)), "ai_overlay")

    assert ai_evaluation["edge_status"] == "HYPOTHESIS"
    assert ai_evaluation["goal_status"] == "3X_NOT_REACHABLE"
    assert "PARENT_EDGE_NOT_PROVEN:worker_parent" in ai_evaluation["edge_blockers"]


def test_same_usdjpy_cluster_is_never_double_added() -> None:
    worker_a = _lane("worker_a", digest_char="a")
    worker_b = _lane("worker_b", digest_char="b")
    for lane in (worker_a, worker_b):
        lane["distribution_30d"].update(
            {
                "stressed_median_multiple": 2.0,
                "probability_3x_lcb": 0.30,
            }
        )

    board = build_goal_board(_input(worker_a, worker_b))

    assert board["goal_status"] == "3X_NOT_REACHABLE"
    assert board["portfolio"]["independent_correlation_cluster_count"] == 0
    assert board["portfolio"]["included_lane_ids"] == []
    assert len(board["portfolio"]["suppressed_correlated_lane_ids"]) == 1
    assert board["portfolio"]["distribution_summed"] is False
    cluster = board["correlation_clusters"][0]
    assert cluster["independence_validated"] is False
    assert cluster["double_count_prevented"] is True
    assert cluster["return_summation_allowed"] is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda lane: lane.update({"unknown": True}),
        lambda lane: lane["distribution_30d"].update(
            {"probability_losing_month": True}
        ),
        lambda lane: lane["distribution_30d"].update(
            {"probability_ruin_12m": float("nan")}
        ),
        lambda lane: lane["risk"].update({"normal_mtm_max_drawdown_fraction": 0.20}),
    ],
)
def test_malformed_lane_input_is_rejected(mutator) -> None:
    lane = _lane("worker_strict")
    mutator(lane)
    if lane["risk"]["normal_mtm_max_drawdown_fraction"] == 0.20:
        lane["risk"]["stressed_mtm_max_drawdown_fraction"] = 0.10

    with pytest.raises(DojoGoalBoardError):
        build_goal_board(_input(lane))


def test_input_loader_rejects_duplicate_keys_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"contract":"x","contract":"y"}', encoding="utf-8")
    with pytest.raises(DojoGoalBoardError, match="duplicate JSON key"):
        load_goal_board_input(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(DojoGoalBoardError, match="non-finite"):
        load_goal_board_input(nonfinite)


def test_script_builds_atomic_machine_board(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    output = tmp_path / "board.json"
    source.write_text(json.dumps(_input(_lane("worker_cli"))), encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build-dojo-goal-board.py"),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    board = json.loads(output.read_text(encoding="utf-8"))
    assert board["edge_status"] == "HYPOTHESIS"
    assert board["goal_status"] == "3X_NOT_REACHABLE"
    assert board["live_permission"] is False
    assert not list(tmp_path.glob(".board.json.*.tmp"))


def test_duplicate_lane_ids_and_status_provenance_conflicts_are_rejected() -> None:
    lane = _lane("duplicate")
    with pytest.raises(DojoGoalBoardError, match="unique lane_id"):
        build_goal_board(_input(lane, deepcopy(lane)))

    conflict = _lane("conflict", status="HYPOTHESIS")
    conflict["provenance"]["valid"] = False
    conflict["provenance"]["invalid_reasons"] = ["CONFLICT"]
    with pytest.raises(DojoGoalBoardError, match="requires status INVALID"):
        build_goal_board(_input(conflict))


def test_self_asserted_json_can_never_promote_to_proof() -> None:
    lane = _lane("self_asserted")

    evaluation = _evaluation(build_goal_board(_input(lane)), "self_asserted")

    assert evaluation["edge_status"] == "HYPOTHESIS"
    assert evaluation["goal_status"] == "3X_NOT_REACHABLE"
    assert "TRUSTED_PROOF_ARTIFACT_REQUIRED" in evaluation["edge_blockers"]


def test_loss_only_distribution_can_never_be_edge_proven() -> None:
    lane = _lane("loss_only")
    lane["distribution_30d"].update(
        {
            "post_cost_edge_lcb": -0.01,
            "stressed_median_multiple": 0.01,
            "probability_3x_lcb": 0.0,
            "probability_losing_month": 1.0,
            "probability_drawdown_20pct": 1.0,
            "probability_ruin_12m": 1.0,
        }
    )

    evaluation = _evaluation(build_goal_board(_input(lane)), "loss_only")

    assert evaluation["edge_status"] == "HYPOTHESIS"
    assert "POST_COST_EDGE_LOWER_BOUND_NOT_POSITIVE" in evaluation["edge_blockers"]


def test_ai_cannot_copy_a_self_asserted_parent_proof() -> None:
    parent = _lane("parent", digest_char="a")
    ai = _lane(
        "ai_copy",
        lane_type="AI",
        digest_char="b",
        parent_lane_ids=["parent"],
        parent_digests={"parent": "a" * 64},
    )

    board = build_goal_board(_input(parent, ai))

    assert _evaluation(board, "parent")["edge_status"] == "HYPOTHESIS"
    assert _evaluation(board, "ai_copy")["edge_status"] == "HYPOTHESIS"
    assert (
        "PARENT_EDGE_NOT_PROVEN:parent"
        in _evaluation(board, "ai_copy")["edge_blockers"]
    )


def test_self_declared_cluster_names_cannot_create_independence() -> None:
    first = _lane("same_exposure_a", cluster="ATTACKER_CLUSTER_A")
    second = _lane("same_exposure_b", digest_char="b", cluster="ATTACKER_CLUSTER_B")

    board = build_goal_board(_input(first, second))

    assert board["portfolio"]["independent_correlation_cluster_count"] == 0
    assert board["portfolio"]["included_lane_ids"] == []


def test_cli_refuses_overwrite_and_input_output_alias(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    output = tmp_path / "board.json"
    original = json.dumps(_input(_lane("worker_cli_exclusive")))
    source.write_text(original, encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build-dojo-goal-board.py"),
        "--input",
        str(source),
        "--output",
        str(output),
    ]

    first = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    first_bytes = output.read_bytes()

    # A later or older input may not replace the immutable publication.  This
    # is the rollback/concurrent-writer boundary, not just idempotency.
    source.write_text(json.dumps(_input(_lane("different_revision"))), encoding="utf-8")
    second = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 2
    assert output.read_bytes() == first_bytes

    source.write_text(original, encoding="utf-8")
    alias = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build-dojo-goal-board.py"),
            "--input",
            str(source),
            "--output",
            str(source),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert alias.returncode == 2
    assert source.read_text(encoding="utf-8") == original


def test_actual_evidence_bytes_are_bound_but_unknown_contract_stays_untrusted(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "proof.json"
    artifact_bytes = json.dumps(
        {"contract": "QR_UNREVIEWED_DOJO_PROOF_V1", "claimed_edge": True},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact.write_bytes(artifact_bytes)
    lane = _lane("unreviewed_artifact")
    lane["provenance"].update(
        {
            "evidence_path": "proof.json",
            "evidence_contract": "QR_UNREVIEWED_DOJO_PROOF_V1",
            "content_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        }
    )

    evaluation = _evaluation(
        build_goal_board(_input(lane), project_root=tmp_path),
        "unreviewed_artifact",
    )

    assert evaluation["edge_status"] == "HYPOTHESIS"
    assert "TRUSTED_PROOF_CONTRACT_UNAVAILABLE" in evaluation["edge_blockers"]
    assert (
        evaluation["evidence_verification"]["actual_sha256"]
        == hashlib.sha256(artifact_bytes).hexdigest()
    )


def test_evidence_path_escape_and_digest_tamper_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"contract":"QR_UNREVIEWED_DOJO_PROOF_V1"}')
    lane = _lane("escaped_artifact")
    lane["provenance"].update(
        {
            "evidence_path": "../outside.json",
            "evidence_contract": "QR_UNREVIEWED_DOJO_PROOF_V1",
            "content_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        }
    )
    escaped = _evaluation(
        build_goal_board(_input(lane), project_root=root), "escaped_artifact"
    )
    assert "EVIDENCE_PATH_NOT_PROJECT_RELATIVE" in escaped["edge_blockers"]

    inside = root / "inside.json"
    inside.write_text('{"contract":"QR_UNREVIEWED_DOJO_PROOF_V1"}')
    lane["lane_id"] = "tampered_artifact"
    lane["provenance"]["evidence_path"] = "inside.json"
    lane["provenance"]["content_sha256"] = "f" * 64
    tampered = _evaluation(
        build_goal_board(_input(lane), project_root=root), "tampered_artifact"
    )
    assert "EVIDENCE_ARTIFACT_DIGEST_MISMATCH" in tampered["edge_blockers"]


def test_unknown_distribution_values_are_null_and_fail_closed_explicitly() -> None:
    lane = _lane("unmeasured_strategy_idea", status="HYPOTHESIS")
    lane["risk"].update(
        {
            "mark_to_market": False,
            "bounded": False,
            "normal_mtm_max_drawdown_fraction": None,
            "stressed_mtm_max_drawdown_fraction": None,
        }
    )
    lane["margin"]["peak_usage_fraction"] = None
    lane["distribution_30d"] = {
        "method": None,
        "sample_months": None,
        "active_days": None,
        "stressed_median_multiple": None,
        "post_cost_edge_lcb": None,
        "probability_3x_lcb": None,
        "probability_losing_month": None,
        "probability_drawdown_20pct": None,
        "probability_ruin_12m": None,
    }

    evaluation = _evaluation(build_goal_board(_input(lane)), "unmeasured_strategy_idea")

    assert evaluation["edge_status"] == "HYPOTHESIS"
    assert evaluation["goal_status"] == "3X_NOT_REACHABLE"
    assert evaluation["distribution_30d"]["stressed_median_multiple"] is None
    assert "DISTRIBUTION_METHOD_UNKNOWN" in evaluation["edge_blockers"]
    assert "PROSPECTIVE_MONTHS_UNKNOWN" in evaluation["edge_blockers"]
    assert "ACTIVE_DAYS_UNKNOWN" in evaluation["edge_blockers"]
    assert "NORMAL_MTM_DRAWDOWN_UNKNOWN" in evaluation["edge_blockers"]
    assert "STRESSED_MTM_DRAWDOWN_UNKNOWN" in evaluation["edge_blockers"]
    assert "MARGIN_PEAK_USAGE_UNKNOWN" in evaluation["edge_blockers"]
    assert "STRESSED_MEDIAN_MULTIPLE_UNKNOWN" in evaluation["goal_blockers"]
    assert "PROBABILITY_3X_LCB_UNKNOWN" in evaluation["goal_blockers"]


def test_unverified_dependence_is_not_an_independent_validated_cluster() -> None:
    board = build_goal_board(_input(_lane("idea_a"), _lane("idea_b", digest_char="b")))

    assert board["portfolio"]["independent_correlation_cluster_count"] == 0
    assert board["portfolio"]["included_lane_ids"] == []
    assert "VALIDATED" not in board["portfolio"]["aggregation_policy"]
    assert board["correlation_clusters"][0]["independence_validated"] is False
    assert board["proof_admission"] == {
        "trusted_proof_contracts": [],
        "promotion_possible": False,
        "self_asserted_json_can_promote": False,
    }


def test_strategy_idea_and_invalid_legacy_performance_are_separate() -> None:
    idea = _lane("strategy_idea", status="HYPOTHESIS")
    legacy = _lane("w46_w53_legacy_performance", status="INVALID", digest_char="b")
    legacy["provenance"]["invalid_reasons"] = [
        "LEGACY_PERFORMANCE_PROVENANCE_INVALIDATED"
    ]

    board = build_goal_board(_input(idea, legacy))

    assert _evaluation(board, "strategy_idea")["edge_status"] == "HYPOTHESIS"
    assert _evaluation(board, "w46_w53_legacy_performance")["edge_status"] == "INVALID"


def test_supersession_is_reciprocal_and_cannot_reactivate_old_evidence() -> None:
    prior = _lane("worker_forward_v1", status="EDGE_PROVEN")
    current = _lane("worker_forward_v2", status="HYPOTHESIS", digest_char="b")
    prior["lifecycle"] = {
        "state": "SUPERSEDED",
        "superseded_by_lane_id": "worker_forward_v2",
        "supersedes_lane_ids": [],
    }
    current["lifecycle"] = {
        "state": "CURRENT",
        "superseded_by_lane_id": None,
        "supersedes_lane_ids": ["worker_forward_v1"],
    }

    board = build_goal_board(_input(prior, current))
    prior_evaluation = _evaluation(board, "worker_forward_v1")

    assert prior_evaluation["edge_status"] == "HYPOTHESIS"
    assert prior_evaluation["goal_status"] == "3X_NOT_REACHABLE"
    assert prior_evaluation["lifecycle"]["state"] == "SUPERSEDED"
    assert (
        "LIFECYCLE_SUPERSEDED_BY:worker_forward_v2" in prior_evaluation["edge_blockers"]
    )

    current["lifecycle"]["supersedes_lane_ids"] = []
    with pytest.raises(DojoGoalBoardError, match="not reciprocal"):
        build_goal_board(_input(prior, current))


def test_implementation_only_lane_cannot_claim_run_results() -> None:
    lane = _lane("ai_v3_implementation", lane_type="AI", status="HYPOTHESIS")
    lane["parent_lane_ids"] = ["worker_parent"]
    lane["provenance"]["parent_digests"] = {"worker_parent": "b" * 64}
    lane["provenance"]["prospective"] = False
    lane["lifecycle"] = {
        "state": "IMPLEMENTATION_ONLY",
        "superseded_by_lane_id": None,
        "supersedes_lane_ids": [],
    }
    lane["distribution_30d"].update(
        {
            "method": "UNAVAILABLE",
            "sample_months": 0,
            "active_days": 0,
            "stressed_median_multiple": None,
            "post_cost_edge_lcb": None,
            "probability_3x_lcb": None,
            "probability_losing_month": None,
            "probability_drawdown_20pct": None,
            "probability_ruin_12m": None,
        }
    )
    lane["risk"].update(
        {
            "normal_mtm_max_drawdown_fraction": None,
            "stressed_mtm_max_drawdown_fraction": None,
        }
    )
    lane["margin"]["peak_usage_fraction"] = None
    lane["sizing"]["observed_at_declared_size"] = False
    parent = _lane("worker_parent", status="HYPOTHESIS", digest_char="b")

    board = build_goal_board(_input(parent, lane))
    evaluation = _evaluation(board, "ai_v3_implementation")
    assert evaluation["edge_status"] == "HYPOTHESIS"
    assert evaluation["goal_status"] == "3X_NOT_REACHABLE"
    assert "IMPLEMENTATION_ONLY_NO_RUN_ARTIFACT" in evaluation["edge_blockers"]

    lane["distribution_30d"]["stressed_median_multiple"] = 3.1
    with pytest.raises(DojoGoalBoardError, match="cannot claim measured results"):
        build_goal_board(_input(parent, lane))


def test_current_registry_keeps_ideas_separate_from_invalid_legacy_artifacts() -> None:
    source = load_goal_board_input(
        ROOT / "research/registries/dojo_goal_board_input_20260719.json"
    )
    lanes = {lane["lane_id"]: lane for lane in source["lanes"]}

    idea = lanes["usd_jpy_mean_reversion_strategy_ideas_v1"]
    forward_v1 = lanes["worker_forward_smoke_v1_started"]
    forward_v2 = lanes["worker_forward_smoke_v2_started"]
    worker_legacy = lanes["worker_w46_w53_legacy_performance_invalid"]
    ai_v1 = lanes["ai_prompt_phase_v1_registered_zero_cells"]
    ai_v2 = lanes["ai_prompt_phase_v2_started_superseded"]
    ai_v3_implementation = lanes["ai_prompt_phase_v3_implementation_no_run"]
    ai_v3_failed = lanes["ai_prompt_phase_v3_startup_incomplete"]
    ai_v3r1 = lanes["ai_prompt_phase_v3r1_started"]
    ai_exit_train_v1 = lanes["ai_exit_train_v1_diagnostic"]
    ai_exit_train_v2 = lanes["ai_exit_train_v2_diagnostic"]
    ai_exit_entry_policy = lanes["worker_ai_exit_entry_policy_train_v2"]
    capital_worker = lanes["worker_capital_time_release_train_v1"]
    ai_capital_parent = lanes["worker_ai_capital_recycle_substrate_train_v1"]
    ai_capital = lanes["ai_capital_recycle_train_v1"]
    tailguard = lanes["worker_spikefade_tailguard_sl25_train_v1"]
    tailguard_lowlev = lanes["worker_spikefade_tailguard_lowlev_train_v1"]
    pullback = lanes["worker_pullback_a2_train_v1"]
    separate_regime = lanes["worker_survivor_regime_replay_train_v1"]
    clean_legacy = lanes["ai_dayread_w54_clean_legacy_contract_invalid"]
    assert idea["status"] == "HYPOTHESIS"
    assert idea["distribution_30d"]["stressed_median_multiple"] is None
    assert forward_v1["lifecycle"]["state"] == "SUPERSEDED"
    assert forward_v2["status"] == "HYPOTHESIS"
    assert forward_v2["lifecycle"]["state"] == "CURRENT"
    assert forward_v2["provenance"]["prospective"] is True
    assert forward_v2["provenance"]["evidence_path"].endswith(
        "dojo-worker-forward-smoke-v2/precommit.json"
    )
    assert forward_v2["distribution_30d"]["active_days"] == 0
    assert ai_v1["lifecycle"]["state"] == "SUPERSEDED"
    assert ai_v2["lifecycle"]["state"] == "SUPERSEDED"
    assert ai_v3_implementation["lifecycle"]["state"] == "SUPERSEDED"
    assert ai_v3_failed["status"] == "INVALID"
    assert ai_v3_failed["lifecycle"]["state"] == "SUPERSEDED"
    assert (
        "VALIDITY_GENESIS_INITIALIZATION_FAILED"
        in (ai_v3_failed["provenance"]["invalid_reasons"])
    )
    assert ai_v3r1["status"] == "INVALID"
    assert ai_v3r1["lifecycle"]["state"] == "SUPERSEDED"
    assert ai_v3r1["provenance"]["prospective"] is True
    assert ai_v3r1["provenance"]["evidence_path"].endswith(
        "dojo-ai-forward-v3r1-supersession-v1.json"
    )
    assert ai_v3r1["distribution_30d"]["active_days"] == 0
    assert ai_exit_train_v1["status"] == "INVALID"
    assert ai_exit_train_v1["lifecycle"]["state"] == "SUPERSEDED"
    assert ai_exit_train_v1["provenance"]["lookahead_free"] is False
    assert ai_exit_train_v2["status"] == "INVALID"
    assert ai_exit_train_v2["lifecycle"]["state"] == "CURRENT"
    assert ai_exit_train_v2["provenance"]["lookahead_free"] is True
    assert ai_exit_train_v2["provenance"]["prospective"] is False
    assert ai_exit_train_v2["parent_lane_ids"] == [
        "worker_ai_exit_entry_policy_train_v2"
    ]
    assert ai_exit_train_v2.get("dependence") is None
    assert ai_exit_entry_policy.get("dependence") is None
    assert ai_exit_train_v2["provenance"]["evidence_path"].endswith(
        "dojo-ai-exit-train-v2/evidence.json"
    )
    assert capital_worker["provenance"]["prospective"] is False
    assert capital_worker["margin"]["peak_usage_fraction"] is None
    assert ai_capital["parent_lane_ids"] == [ai_capital_parent["lane_id"]]
    assert (
        ai_capital["provenance"]["content_sha256"]
        == (ai_capital_parent["provenance"]["content_sha256"])
    )
    assert ai_capital["provenance"]["prospective"] is False
    assert ai_exit_train_v1["provenance"]["evidence_path"].endswith(
        "dojo-ai-exit-train-v1/evidence.json"
    )
    for tuned_worker in (tailguard, tailguard_lowlev, pullback):
        assert tuned_worker["status"] == "HYPOTHESIS"
        assert tuned_worker["lifecycle"]["state"] == "SUPERSEDED"
        assert tuned_worker["lifecycle"]["superseded_by_lane_id"] == (
            "worker_survivor_regime_replay_train_v1"
        )
        assert tuned_worker["provenance"]["valid"] is True
        assert tuned_worker["provenance"]["prospective"] is False
        assert tuned_worker["provenance"]["evidence_path"].endswith(
            "dojo-worker-ai-tuning-20260719/evidence.json"
        )
    assert separate_regime["lifecycle"]["state"] == "CURRENT"
    assert set(separate_regime["lifecycle"]["supersedes_lane_ids"]) == {
        tailguard["lane_id"],
        tailguard_lowlev["lane_id"],
        pullback["lane_id"],
    }
    assert (
        tailguard["margin"]["peak_usage_fraction"] > tailguard["margin"]["cap_fraction"]
    )
    assert (
        tailguard_lowlev["margin"]["peak_usage_fraction"]
        < tailguard_lowlev["margin"]["cap_fraction"]
    )
    assert pullback["sizing"]["reverse_engineered_from_goal"] is False
    assert worker_legacy["status"] == "INVALID"
    assert (
        "LEGACY_PERFORMANCE_PROVENANCE_INVALIDATED"
        in worker_legacy["provenance"]["invalid_reasons"]
    )
    assert clean_legacy["status"] == "INVALID"
    assert clean_legacy["distribution_30d"]["probability_losing_month"] is None

    board = build_goal_board(source, project_root=ROOT)

    assert board["edge_status"] == "HYPOTHESIS"
    assert board["goal_status"] == "3X_NOT_REACHABLE"
    assert board["proof_admission"]["promotion_possible"] is False
    assert board["effective_independent_n"] == 0
    assert board["portfolio"]["independent_correlation_cluster_count"] == 0
    assert board["portfolio"]["included_lane_ids"] == []
    assert board["guarantee"] is False
    assert board["live_permission"] is False
    assert board["order_authority"] == "NONE"
    assert board["broker_mutation_allowed"] is False
    assert _evaluation(board, "worker_forward_smoke_v2_started")["edge_status"] == (
        "HYPOTHESIS"
    )
    ai_v3r1_evaluation = _evaluation(board, "ai_prompt_phase_v3r1_started")
    assert ai_v3r1_evaluation["edge_status"] == "INVALID"
    assert "REDUNDANT_CANDLE_ONLY_INPUT_CLASS" in (ai_v3r1_evaluation["edge_blockers"])
    ai_exit_v1_evaluation = _evaluation(board, "ai_exit_train_v1_diagnostic")
    assert ai_exit_v1_evaluation["edge_status"] == "INVALID"
    assert (
        "DECISION_BAR_HIGH_LOW_CLOSE_EXPOSED_AT_DECISION_BAR_OPEN"
        in (ai_exit_v1_evaluation["edge_blockers"])
    )
    ai_exit_evaluation = _evaluation(board, "ai_exit_train_v2_diagnostic")
    assert ai_exit_evaluation["edge_status"] == "INVALID"
    assert (
        "SOURCE_GAP_INVALIDATED_X05_AND_SHIFTED_X06_X08"
        in ai_exit_evaluation["edge_blockers"]
    )
    assert (
        "PARENT_INVALID:worker_ai_exit_entry_policy_train_v2"
        in ai_exit_evaluation["edge_blockers"]
    )
    assert not any(
        "MISMATCH" in blocker for blocker in ai_exit_evaluation["edge_blockers"]
    )
    tailguard_evaluation = _evaluation(
        board, "worker_spikefade_tailguard_sl25_train_v1"
    )
    assert tailguard_evaluation["edge_status"] == "HYPOTHESIS"
    assert "PROSPECTIVE_PROVENANCE_REQUIRED" in tailguard_evaluation["edge_blockers"]
    assert "MARGIN_CAP_BREACHED" in tailguard_evaluation["edge_blockers"]
    lowlev_evaluation = _evaluation(board, "worker_spikefade_tailguard_lowlev_train_v1")
    assert lowlev_evaluation["edge_status"] == "HYPOTHESIS"
    assert "MARGIN_CAP_BREACHED" not in lowlev_evaluation["edge_blockers"]
    pullback_evaluation = _evaluation(board, "worker_pullback_a2_train_v1")
    assert pullback_evaluation["edge_status"] == "HYPOTHESIS"
    assert "PROSPECTIVE_PROVENANCE_REQUIRED" in pullback_evaluation["edge_blockers"]
    capital_evaluation = _evaluation(board, "worker_capital_time_release_train_v1")
    assert capital_evaluation["edge_status"] == "HYPOTHESIS"
    assert "MARGIN_PEAK_USAGE_UNKNOWN" in capital_evaluation["edge_blockers"]
    ai_capital_evaluation = _evaluation(board, "ai_capital_recycle_train_v1")
    assert ai_capital_evaluation["edge_status"] == "HYPOTHESIS"
    assert (
        "PARENT_EDGE_NOT_PROVEN:worker_ai_capital_recycle_substrate_train_v1"
        in ai_capital_evaluation["edge_blockers"]
    )
    assert not any(
        "MISMATCH" in blocker for blocker in ai_capital_evaluation["edge_blockers"]
    )


def test_current_content_addressed_board_exactly_matches_registry_input() -> None:
    registry_dir = ROOT / "research/registries"
    board_paths = sorted(registry_dir.glob("dojo_goal_board_20260719_*.json"))
    assert len(board_paths) == 1

    source = load_goal_board_input(registry_dir / "dojo_goal_board_input_20260719.json")
    expected = build_goal_board(source, project_root=ROOT)
    actual = json.loads(board_paths[0].read_text(encoding="utf-8"))

    assert actual == expected
    assert board_paths[0].stem.endswith(expected["board_sha256"])

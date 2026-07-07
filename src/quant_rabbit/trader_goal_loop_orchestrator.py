from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
    DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR,
    DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR_REPORT,
    DEFAULT_TRADER_REPAIR_ORCHESTRATOR,
)


CONTRACT_VERSION = "trader_goal_loop_orchestrator_v1"
WORK_TYPES = {
    "NO_ACTION_WAIT",
    "READ_ONLY_EVIDENCE_REFRESH",
    "PAYOFF_SHAPE_DIAGNOSIS",
    "HARVEST_PROOF_PATH",
    "SCOUT_PLAN",
    "OPERATOR_REVIEW_REPORT",
    "CODE_REPAIR",
    "LIVE_PERMISSION_READY_CHECK",
    "EDGE_IMPROVEMENT_EXPERIMENT",
}
INPUT_ARTIFACT_NAMES = (
    "trader_repair_orchestrator",
    "payoff_shape_diagnosis",
    "harvest_live_grade_path",
    "eurusd_short_breakout_failure_scout_plan",
    "as_proof_pack_queue",
    "as_lane_candidate_board",
    "portfolio_4x_path_planner",
    "live_order_request",
    "broker_snapshot",
)
DEFAULT_HARVEST_LIVE_GRADE_PATH = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "harvest_live_grade_path.json"
DEFAULT_EURUSD_SHORT_BREAKOUT_FAILURE_SCOUT_PLAN = (
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "eurusd_short_breakout_failure_scout_plan.json"
)
DEFAULT_AS_PROOF_PACK_QUEUE = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "as_proof_pack_queue.json"
DEFAULT_AS_LANE_CANDIDATE_BOARD = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "as_lane_candidate_board.json"
DEFAULT_PORTFOLIO_4X_PATH_PLANNER = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "portfolio_4x_path_planner.json"
PAYOFF_STALE_AFTER_SECONDS = 24 * 60 * 60
BROKER_TRUTH_FRESH_SECONDS = 10 * 60
SUCCESS_CONDITION_SCHEMA_VERSION = "success_condition_v1"


@dataclass(frozen=True)
class TraderGoalLoopOrchestratorSummary:
    status: str
    output_path: Path
    report_path: Path
    selected_next_work_type: str
    current_phase: str
    live_permission_allowed: bool


class TraderGoalLoopOrchestrator:
    """Choose the next read-only Codex work type for the 4x improvement loop."""

    def __init__(
        self,
        *,
        trader_repair_orchestrator_path: Path = DEFAULT_TRADER_REPAIR_ORCHESTRATOR,
        payoff_shape_diagnosis_path: Path = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
        harvest_live_grade_path: Path = DEFAULT_HARVEST_LIVE_GRADE_PATH,
        scout_plan_path: Path = DEFAULT_EURUSD_SHORT_BREAKOUT_FAILURE_SCOUT_PLAN,
        as_proof_pack_queue_path: Path = DEFAULT_AS_PROOF_PACK_QUEUE,
        as_lane_candidate_board_path: Path = DEFAULT_AS_LANE_CANDIDATE_BOARD,
        portfolio_4x_path_planner_path: Path = DEFAULT_PORTFOLIO_4X_PATH_PLANNER,
        live_order_request_path: Path = DEFAULT_LIVE_ORDER_REQUEST,
        broker_snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        output_path: Path = DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR,
        report_path: Path = DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR_REPORT,
        now_utc: datetime | None = None,
    ) -> None:
        self.paths = {
            "trader_repair_orchestrator": trader_repair_orchestrator_path,
            "payoff_shape_diagnosis": payoff_shape_diagnosis_path,
            "harvest_live_grade_path": harvest_live_grade_path,
            "eurusd_short_breakout_failure_scout_plan": scout_plan_path,
            "as_proof_pack_queue": as_proof_pack_queue_path,
            "as_lane_candidate_board": as_lane_candidate_board_path,
            "portfolio_4x_path_planner": portfolio_4x_path_planner_path,
            "live_order_request": live_order_request_path,
            "broker_snapshot": broker_snapshot_path,
        }
        self.output_path = output_path
        self.report_path = report_path
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> TraderGoalLoopOrchestratorSummary:
        payload = self.build_payload()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload), encoding="utf-8")
        return TraderGoalLoopOrchestratorSummary(
            status=str(payload["status"]),
            output_path=self.output_path,
            report_path=self.report_path,
            selected_next_work_type=str(payload["selected_next_work_type"]),
            current_phase=str(payload["current_phase"]),
            live_permission_allowed=bool(payload["live_permission_allowed"]),
        )

    def build_payload(self) -> dict[str, Any]:
        artifacts = {name: _load_artifact(path) for name, path in self.paths.items()}
        artifact_index = _artifact_index(artifacts)
        proof_state = _proof_state(artifacts)
        payoff_state = _payoff_state(artifacts, self.now_utc)
        harvest_state = _harvest_state(artifacts)
        scout_state = _scout_state(artifacts)
        operator_review_state = _operator_review_state(artifacts, proof_state, scout_state)
        edge_improvement_state = _edge_improvement_state(
            proof_state=proof_state,
            payoff_state=payoff_state,
            harvest_state=harvest_state,
            scout_state=scout_state,
        )
        schema_state = _schema_state(artifacts)
        artifact_health = _artifact_health(artifacts)
        live_ready_state = _live_permission_ready_state(
            proof_state=proof_state,
            harvest_state=harvest_state,
            operator_review_state=operator_review_state,
            artifacts=artifacts,
            now_utc=self.now_utc,
        )
        candidate_work_type, selection_reason = _select_work_type(
            payoff_state=payoff_state,
            harvest_state=harvest_state,
            scout_state=scout_state,
            operator_review_state=operator_review_state,
            edge_improvement_state=edge_improvement_state,
            artifact_health=artifact_health,
            schema_state=schema_state,
            live_ready_state=live_ready_state,
        )
        key_blocker = _key_blocker(
            selected_next_work_type=candidate_work_type,
            proof_state=proof_state,
            payoff_state=payoff_state,
            harvest_state=harvest_state,
            scout_state=scout_state,
            operator_review_state=operator_review_state,
            artifact_health=artifact_health,
            schema_state=schema_state,
        )
        repeat_loop_guard = _repeat_loop_guard(
            output_path=self.output_path,
            artifact_index=artifact_index,
            selected_next_work_type=candidate_work_type,
            key_blocker=key_blocker,
        )
        if repeat_loop_guard["same_work_type_artifact_and_blocker"]:
            selection_reason += (
                " repeat_loop_guard also shows the same work type/artifact hash/key blocker as the "
                "previous goal-loop artifact; dispatch should wait for changed generated_at, artifact hash, "
                "or key blocker, but the artifact-derived classification is preserved."
            )
        selected_next_work_type = candidate_work_type
        current_phase = _current_phase(
            selected_next_work_type=selected_next_work_type,
            payoff_state=payoff_state,
            harvest_state=harvest_state,
            scout_state=scout_state,
            operator_review_state=operator_review_state,
            edge_improvement_state=edge_improvement_state,
        )
        success_condition = _success_condition(selected_next_work_type)
        current_state = _current_state_for_evaluation(
            proof_state=proof_state,
            payoff_state=payoff_state,
            harvest_state=harvest_state,
            scout_state=scout_state,
            operator_review_state=operator_review_state,
            edge_improvement_state=edge_improvement_state,
            artifact_health=artifact_health,
            schema_state=schema_state,
            live_ready_state=live_ready_state,
        )
        success_condition_evaluation = _evaluate_success_condition(success_condition, current_state)
        next_allowed_commands = _next_allowed_commands(selected_next_work_type)
        selected_next_prompt = _selected_next_prompt(
            selected_next_work_type=selected_next_work_type,
            current_phase=current_phase,
            selection_reason=selection_reason,
            four_x_progress_hypothesis=_four_x_progress_hypothesis(edge_improvement_state, scout_state),
            root_improvement_target=_root_improvement_target(edge_improvement_state, scout_state),
            expected_edge_improvement=_expected_edge_improvement(edge_improvement_state, scout_state),
            success_condition=success_condition,
            next_allowed_commands=next_allowed_commands,
        )
        payload = {
            "contract_version": CONTRACT_VERSION,
            "status": "NEXT_WORK_SELECTED",
            "generated_at_utc": self.now_utc.isoformat(),
            "read_only": True,
            "live_side_effects": [],
            "current_phase": current_phase,
            "selected_next_work_type": selected_next_work_type,
            "selected_next_prompt": selected_next_prompt,
            "selection_reason": selection_reason,
            "four_x_progress_hypothesis": _four_x_progress_hypothesis(edge_improvement_state, scout_state),
            "root_improvement_target": _root_improvement_target(edge_improvement_state, scout_state),
            "expected_edge_improvement": _expected_edge_improvement(edge_improvement_state, scout_state),
            "proof_state": proof_state,
            "payoff_state": payoff_state,
            "harvest_state": harvest_state,
            "scout_state": scout_state,
            "operator_review_state": operator_review_state,
            "edge_improvement_state": edge_improvement_state,
            "repeat_loop_guard": repeat_loop_guard,
            "success_condition": success_condition,
            "success_condition_evaluation": success_condition_evaluation,
            "next_allowed_commands": next_allowed_commands,
            "requires_operator_approval": selected_next_work_type == "OPERATOR_REVIEW_REPORT",
            "live_permission_allowed": False,
            "live_permission_ready_check_state": live_ready_state,
            "artifact_health": artifact_health,
            "schema_state": schema_state,
            "artifact_index": artifact_index,
            "safety_contract": _safety_contract(),
        }
        if selected_next_work_type not in WORK_TYPES:
            raise ValueError(f"unknown selected_next_work_type: {selected_next_work_type}")
        return payload


def _load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_artifact_status": "missing", "_path": str(path), "_sha256": None}
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    payload = dict(payload)
    payload["_artifact_status"] = "present"
    payload["_path"] = str(path)
    payload["_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def _artifact_index(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entries = {}
    for name in INPUT_ARTIFACT_NAMES:
        artifact = artifacts[name]
        entries[name] = {
            "path": artifact.get("_path"),
            "status": artifact.get("_artifact_status"),
            "sha256": artifact.get("_sha256"),
            "generated_at_utc": _artifact_generated_at(artifact),
        }
    combined = hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    generated_values = [
        value.get("generated_at_utc")
        for value in entries.values()
        if isinstance(value.get("generated_at_utc"), str)
    ]
    return {
        "artifacts": entries,
        "combined_sha256": combined,
        "latest_input_generated_at_utc": max(generated_values) if generated_values else None,
    }


def _artifact_generated_at(artifact: dict[str, Any]) -> str | None:
    for key in ("generated_at_utc", "fetched_at_utc"):
        value = artifact.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _payoff_state(artifacts: dict[str, dict[str, Any]], now_utc: datetime) -> dict[str, Any]:
    artifact = artifacts["payoff_shape_diagnosis"]
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "path": artifact.get("_path"),
            "stale": True,
            "verdict": None,
            "live_promotion_allowed": False,
            "runner_candidate_count": 0,
            "harvest_candidate_count": 0,
            "missing_source_artifacts": [],
            "negative_expectancy_visible": None,
            "month_scale_negative_visible": None,
        }
    verdict = artifact.get("overall_payoff_shape_verdict")
    verdict = verdict if isinstance(verdict, dict) else {}
    generated_at = _parse_dt(artifact.get("generated_at_utc"))
    age_seconds = (now_utc - generated_at).total_seconds() if generated_at else None
    stale = (
        generated_at is None
        or age_seconds is None
        or age_seconds > PAYOFF_STALE_AFTER_SECONDS
        or _explicit_artifact_issue(artifact) in {"STALE", "CONTRADICTED"}
    )
    live_blockers = _string_list(verdict.get("live_promotion_blockers"))
    return {
        "artifact_status": "present",
        "path": artifact.get("_path"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "age_seconds": _round(age_seconds),
        "stale": bool(stale),
        "status": artifact.get("status"),
        "verdict": verdict.get("classification"),
        "capture_economics_status": verdict.get("capture_economics_status"),
        "live_promotion_allowed": bool(verdict.get("live_promotion_allowed")),
        "live_promotion_blockers": live_blockers,
        "runner_candidate_count": len(artifact.get("runner_candidates") or []),
        "harvest_candidate_count": len(artifact.get("harvest_candidates") or []),
        "missing_source_artifacts": list(artifact.get("missing_source_artifacts") or []),
        "negative_expectancy_visible": (
            verdict.get("capture_economics_status") == "NEGATIVE_EXPECTANCY"
            or "NEGATIVE_EXPECTANCY" in live_blockers
        ),
        "month_scale_negative_visible": (
            verdict.get("month_scale_blocker") == "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"
            or "MONTH_SCALE_REPLAY_NEGATIVE" in live_blockers
        ),
    }


def _harvest_state(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    artifact = artifacts["harvest_live_grade_path"]
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "path": artifact.get("_path"),
            "closest_harvest_selected": False,
            "closest_harvest_candidate_id": None,
            "live_promotion_allowed": False,
            "promotion_blockers": [],
        }
    closest = artifact.get("closest_harvest_candidate")
    closest = closest if isinstance(closest, dict) else {}
    current_intent = closest.get("current_intent_best") if isinstance(closest.get("current_intent_best"), dict) else {}
    tp_proof = closest.get("tp_proof") if isinstance(closest.get("tp_proof"), dict) else {}
    return {
        "artifact_status": "present",
        "path": artifact.get("_path"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "status": artifact.get("status"),
        "closest_harvest_selected": bool(closest),
        "closest_harvest_candidate_id": closest.get("candidate_id") or closest.get("shape_key"),
        "closest_shape_key": closest.get("shape_key"),
        "pair": closest.get("pair"),
        "side": closest.get("side"),
        "method": closest.get("method"),
        "actual_proof_queue_member": bool(closest.get("actual_proof_queue_member")),
        "planner_can_enter_proof_pack": bool(closest.get("planner_can_enter_proof_pack")),
        "can_create_live_permission": bool(closest.get("can_create_live_permission")),
        "live_promotion_allowed": bool(artifact.get("live_promotion_allowed")) and bool(
            closest.get("live_promotion_allowed")
        ),
        "promotion_blockers": _string_list(closest.get("promotion_blockers"))
        + _codes_from_blockers(artifact.get("promotion_blockers")),
        "tp_proof": {
            "take_profit_trades": _int(tp_proof.get("take_profit_trades")),
            "take_profit_losses": _int(tp_proof.get("take_profit_losses")),
            "proof_floor_trades": _int(tp_proof.get("proof_floor_trades")),
            "proof_gap_trades": _int(tp_proof.get("proof_gap_trades")),
            "take_profit_expectancy_jpy": tp_proof.get("take_profit_expectancy_jpy"),
        },
        "current_intent_best": {
            "lane_id": current_intent.get("lane_id"),
            "status": current_intent.get("status"),
            "order_type": current_intent.get("order_type"),
            "risk_allowed": current_intent.get("risk_allowed"),
            "risk_jpy": current_intent.get("risk_jpy"),
            "units": current_intent.get("units"),
            "live_blocker_codes": _string_list(current_intent.get("live_blocker_codes")),
        },
    }


def _scout_state(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    artifact = artifacts["eurusd_short_breakout_failure_scout_plan"]
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "path": artifact.get("_path"),
            "diagnosed": False,
            "status": None,
            "scout_mode_allowed": False,
            "operator_approval_required": False,
            "blocker_codes": [],
        }
    blockers = _codes_from_blockers(artifact.get("proof_queue_entry_blockers"))
    return {
        "artifact_status": "present",
        "path": artifact.get("_path"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "diagnosed": bool(artifact.get("status")),
        "status": artifact.get("status"),
        "target_shape": artifact.get("target_shape"),
        "scout_mode_allowed": bool(artifact.get("scout_mode_allowed")),
        "scout_mode_reason": artifact.get("scout_mode_reason"),
        "operator_approval_required": bool(artifact.get("operator_approval_required")),
        "max_loss_jpy_cap": artifact.get("max_loss_jpy_cap"),
        "min_lot_feasibility": artifact.get("min_lot_feasibility") or {},
        "blocker_codes": blockers,
        "proof_queue_entry_blockers": artifact.get("proof_queue_entry_blockers") or [],
        "evidence_success_conditions": artifact.get("evidence_success_conditions") or [],
    }


def _proof_state(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    proof = artifacts["as_proof_pack_queue"]
    board = artifacts["as_lane_candidate_board"]
    portfolio = artifacts["portfolio_4x_path_planner"]
    repair = artifacts["trader_repair_orchestrator"]
    summary = proof.get("summary") if isinstance(proof.get("summary"), dict) else {}
    queue = proof.get("queue") if isinstance(proof.get("queue"), list) else []
    rejected = proof.get("rejected_candidates") if isinstance(proof.get("rejected_candidates"), list) else []
    board_blocker = (
        board.get("exact_blocker_preventing_live_ready")
        if isinstance(board.get("exact_blocker_preventing_live_ready"), dict)
        else {}
    )
    portfolio_summary = portfolio.get("summary") if isinstance(portfolio.get("summary"), dict) else {}
    repair_state = _repair_current_state(repair)
    proof_queue_count = _first_int(summary.get("queue_count"), len(queue), repair_state.get("proof_queue_count"))
    proof_ready_count = _first_int(
        summary.get("proof_ready_count"),
        sum(1 for row in queue if isinstance(row, dict) and row.get("proof_ready") is True),
        repair_state.get("proof_ready_count"),
    )
    can_create_count = _first_int(
        summary.get("can_create_live_permission_count"),
        sum(
            1
            for row in queue
            if isinstance(row, dict) and row.get("can_create_live_permission") is True
        ),
        repair_state.get("can_create_live_permission_count"),
    )
    return {
        "as_proof_pack_queue_artifact_status": proof.get("_artifact_status"),
        "as_lane_candidate_board_artifact_status": board.get("_artifact_status"),
        "portfolio_4x_path_planner_artifact_status": portfolio.get("_artifact_status"),
        "proof_queue_generated_at_utc": proof.get("generated_at_utc"),
        "as_board_generated_at_utc": board.get("generated_at_utc"),
        "portfolio_planner_generated_at_utc": portfolio.get("generated_at_utc"),
        "proof_queue_count": proof_queue_count,
        "proof_ready_count": proof_ready_count,
        "can_create_live_permission_count": can_create_count,
        "rejected_proof_candidate_count": _first_int(
            summary.get("rejected_candidate_count"),
            len(rejected),
            repair_state.get("rejected_proof_candidate_count"),
        ),
        "proof_queue_empty": proof_queue_count == 0,
        "proof_queue_count_is_live_permission": False,
        "as_live_ready_path_exists": bool(board.get("as_live_ready_path_exists") or summary.get("as_live_ready_path_exists")),
        "normal_routing_status": _first_str(
            board.get("normal_routing_status"),
            portfolio.get("normal_routing_status"),
            repair_state.get("proof_normal_routing_status"),
        ),
        "routing_allowed": _first_bool(
            board.get("routing_allowed"),
            repair_state.get("proof_routing_allowed"),
        ),
        "primary_blocker": _first_str(
            board_blocker.get("primary"),
            repair_state.get("proof_primary_blocker"),
        ),
        "global_blockers": _unique(
            _string_list(board_blocker.get("global_blockers"))
            + _string_list(repair_state.get("proof_global_blockers"))
        ),
        "p0_rows": _unique(_string_list(board_blocker.get("p0_rows")) + _string_list(repair_state.get("proof_p0_rows"))),
        "portfolio_status": portfolio.get("portfolio_status"),
        "portfolio_can_reach_4x_now": bool(portfolio.get("can_reach_4x_now")),
        "portfolio_can_create_live_permission": bool(portfolio_summary.get("can_create_live_permission")),
    }


def _operator_review_state(
    artifacts: dict[str, dict[str, Any]],
    proof_state: dict[str, Any],
    scout_state: dict[str, Any],
) -> dict[str, Any]:
    markers = _guardian_markers(artifacts)
    normal_routing_allowed = _first_bool(
        _normal_routing_from_scout(scout_state),
        proof_state.get("routing_allowed"),
    )
    blocker_codes = _unique(
        proof_state.get("global_blockers", [])
        + scout_state.get("blocker_codes", [])
        + [marker for marker in markers if "GUARDIAN" in marker or "OPERATOR_REVIEW" in marker]
    )
    operator_review_required = (
        normal_routing_allowed is False
        and any("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED" in code for code in blocker_codes)
    ) or scout_state.get("status") == "SCOUT_BLOCKED_OPERATOR_REVIEW"
    stale_or_expired = any("EXPIRED" in marker or "OPERATOR_REVIEW_STALE" in marker for marker in markers)
    guardian_clear = (
        normal_routing_allowed is True
        and not operator_review_required
        and not any("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED" in code for code in blocker_codes)
    )
    return {
        "normal_routing_allowed": normal_routing_allowed,
        "normal_routing_status": proof_state.get("normal_routing_status"),
        "operator_review_required": bool(operator_review_required),
        "guardian_clear": bool(guardian_clear),
        "guardian_markers": markers,
        "guardian_expired_or_operator_review_stale": bool(stale_or_expired),
        "blocker_codes": blocker_codes,
        "source": "input_artifact_summaries_only",
        "missing_raw_guardian_receipt_artifact": True,
    }


def _edge_improvement_state(
    *,
    proof_state: dict[str, Any],
    payoff_state: dict[str, Any],
    harvest_state: dict[str, Any],
    scout_state: dict[str, Any],
) -> dict[str, Any]:
    tp_proof = harvest_state.get("tp_proof") if isinstance(harvest_state.get("tp_proof"), dict) else {}
    proof_gap = _int(tp_proof.get("proof_gap_trades"))
    take_profit_trades = _int(tp_proof.get("take_profit_trades"))
    take_profit_losses = _int(tp_proof.get("take_profit_losses"))
    expectancy = tp_proof.get("take_profit_expectancy_jpy")
    blockers = _unique(
        _string_list(harvest_state.get("promotion_blockers"))
        + _string_list(proof_state.get("global_blockers"))
        + _string_list(scout_state.get("blocker_codes"))
    )
    candidate_available = bool(
        harvest_state.get("closest_harvest_selected")
        and payoff_state.get("verdict") == "MIXED_HARVEST_PRIMARY"
        and not harvest_state.get("live_promotion_allowed")
    )
    experiments: list[dict[str, Any]] = []
    if candidate_available:
        experiments.append(
            {
                "experiment_type": "sampling",
                "target": harvest_state.get("closest_harvest_candidate_id"),
                "hypothesis": "attached-TP HARVEST proof can move from thin positive sample to live-grade evidence after missing TP samples are collected without TP-loss leakage.",
                "read_only_design": "operator review can decide whether a proof-collection SCOUT is worth approving; this orchestrator only packages the evidence and never sends it.",
                "success_signal": {
                    "take_profit_trades": take_profit_trades,
                    "take_profit_losses": take_profit_losses,
                    "proof_gap_trades": proof_gap,
                    "take_profit_expectancy_jpy": expectancy,
                },
            }
        )
        experiments.append(
            {
                "experiment_type": "payoff",
                "target": harvest_state.get("closest_harvest_candidate_id"),
                "hypothesis": "entry/exit filtering should preserve the TP-positive HARVEST bucket while excluding market-close leak and NO_TRADE shapes.",
                "read_only_design": "compare current blockers, no-trade shapes, replay residuals, and TP proof before proposing code or scout work.",
                "success_signal": {
                    "negative_expectancy_visible": payoff_state.get("negative_expectancy_visible"),
                    "month_scale_negative_visible": payoff_state.get("month_scale_negative_visible"),
                    "proof_queue_count": proof_state.get("proof_queue_count"),
                },
            }
        )
    return {
        "candidate_available": candidate_available,
        "candidate_work_type": "EDGE_IMPROVEMENT_EXPERIMENT" if candidate_available else None,
        "target_shape": harvest_state.get("closest_harvest_candidate_id"),
        "proof_gap_trades": proof_gap,
        "take_profit_trades": take_profit_trades,
        "take_profit_losses": take_profit_losses,
        "take_profit_expectancy_jpy": expectancy,
        "min_lot_numerically_feasible": (
            scout_state.get("min_lot_feasibility", {}).get("feasible_if_all_non_lot_gates_clear")
            if isinstance(scout_state.get("min_lot_feasibility"), dict)
            else None
        ),
        "proof_queue_count": proof_state.get("proof_queue_count"),
        "proof_queue_count_is_live_permission": False,
        "live_promotion_allowed": bool(harvest_state.get("live_promotion_allowed")),
        "blockers_to_preserve": blockers,
        "experiments": experiments,
    }


def _schema_state(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing_fields: list[dict[str, str]] = []
    required_fields = {
        "payoff_shape_diagnosis": ("status", "overall_payoff_shape_verdict"),
        "harvest_live_grade_path": ("status", "closest_harvest_candidate", "live_promotion_allowed"),
        "eurusd_short_breakout_failure_scout_plan": ("status", "live_side_effects"),
        "as_proof_pack_queue": ("summary", "live_side_effects"),
        "as_lane_candidate_board": ("normal_routing_status", "live_side_effects"),
        "portfolio_4x_path_planner": ("portfolio_status", "live_side_effects"),
        "live_order_request": ("status", "generated_at_utc"),
        "broker_snapshot": ("fetched_at_utc",),
    }
    for name, fields in required_fields.items():
        artifact = artifacts[name]
        if artifact.get("_artifact_status") == "missing":
            continue
        for field in fields:
            if field not in artifact:
                missing_fields.append({"artifact": name, "field": field})
    success_condition_issues = _success_condition_schema_issues(artifacts["trader_repair_orchestrator"])
    return {
        "required_field_missing": bool(missing_fields),
        "missing_fields": missing_fields,
        "success_condition_schema_ok": not success_condition_issues,
        "success_condition_schema_issues": success_condition_issues,
        "code_repair_required": bool(missing_fields or success_condition_issues),
    }


def _artifact_health(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    explicit_issues: list[dict[str, str]] = []
    for name, artifact in artifacts.items():
        issue = _explicit_artifact_issue(artifact)
        if issue:
            explicit_issues.append({"artifact": name, "issue": issue})
    repair = artifacts["trader_repair_orchestrator"]
    repair_state = _repair_current_state(repair)
    contradiction_codes = _string_list(repair_state.get("artifact_contradiction_codes")) + _string_list(
        repair.get("artifact_contradiction_codes")
    )
    freshness_issues = _freshness_issues(repair)
    return {
        "explicit_issues": explicit_issues,
        "artifact_contradiction_codes": _unique(contradiction_codes),
        "freshness_issues": freshness_issues,
        "has_stale_or_contradicted_artifact": bool(
            explicit_issues or contradiction_codes or [row for row in freshness_issues if row.get("status") == "CONTRADICTED"]
        ),
    }


def _live_permission_ready_state(
    *,
    proof_state: dict[str, Any],
    harvest_state: dict[str, Any],
    operator_review_state: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    now_utc: datetime,
) -> dict[str, Any]:
    broker = artifacts["broker_snapshot"]
    fetched_at = _parse_dt(broker.get("fetched_at_utc"))
    broker_age_seconds = (now_utc - fetched_at).total_seconds() if fetched_at else None
    broker_truth_fresh = broker_age_seconds is not None and 0 <= broker_age_seconds <= BROKER_TRUTH_FRESH_SECONDS
    checks = {
        "proof_queue_count_gt_zero": proof_state.get("proof_queue_count", 0) > 0,
        "can_create_live_permission_count_gt_zero": proof_state.get("can_create_live_permission_count", 0) > 0,
        "live_promotion_allowed": bool(harvest_state.get("live_promotion_allowed")),
        "guardian_clear": bool(operator_review_state.get("guardian_clear")),
        "broker_truth_fresh": bool(broker_truth_fresh),
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "broker_snapshot_fetched_at_utc": broker.get("fetched_at_utc"),
        "broker_truth_age_seconds": _round(broker_age_seconds),
        "broker_truth_fresh_seconds": BROKER_TRUTH_FRESH_SECONDS,
        "live_permission_allowed": False,
        "reason": "This is readiness-check routing only; it never grants live permission.",
    }


def _select_work_type(
    *,
    payoff_state: dict[str, Any],
    harvest_state: dict[str, Any],
    scout_state: dict[str, Any],
    operator_review_state: dict[str, Any],
    edge_improvement_state: dict[str, Any],
    artifact_health: dict[str, Any],
    schema_state: dict[str, Any],
    live_ready_state: dict[str, Any],
) -> tuple[str, str]:
    if payoff_state.get("artifact_status") == "missing" or payoff_state.get("stale"):
        return (
            "PAYOFF_SHAPE_DIAGNOSIS",
            "payoff_shape_diagnosis is missing or stale, so the next loop must refresh payoff shape before selecting downstream work.",
        )
    if payoff_state.get("verdict") == "MIXED_HARVEST_PRIMARY" and not harvest_state.get(
        "closest_harvest_selected"
    ):
        return (
            "HARVEST_PROOF_PATH",
            "payoff verdict is MIXED_HARVEST_PRIMARY but no closest harvest candidate is selected.",
        )
    if harvest_state.get("closest_harvest_selected") and not scout_state.get("diagnosed"):
        return (
            "SCOUT_PLAN",
            "closest harvest candidate exists but the exact scout plan has not been diagnosed.",
        )
    if scout_state.get("status") == "SCOUT_BLOCKED_OPERATOR_REVIEW":
        return (
            "OPERATOR_REVIEW_REPORT",
            "scout status is SCOUT_BLOCKED_OPERATOR_REVIEW; the next artifact must package SCOUT approval/rejection evidence only, not live permission.",
        )
    if artifact_health.get("has_stale_or_contradicted_artifact"):
        return (
            "READ_ONLY_EVIDENCE_REFRESH",
            "one or more artifacts are explicitly STALE or CONTRADICTED, so the next step is read-only evidence refresh.",
        )
    if schema_state.get("code_repair_required"):
        return (
            "CODE_REPAIR",
            "required artifact fields or success_condition schema checks are missing; repair the orchestrator/schema before continuing.",
        )
    if edge_improvement_state.get("candidate_available"):
        return (
            "EDGE_IMPROVEMENT_EXPERIMENT",
            "operator review is not the immediate blocker and a HARVEST candidate still has read-only expectancy/sampling/payoff experiments that can move 4x evidence forward without live permission.",
        )
    if live_ready_state.get("all_checks_passed"):
        return (
            "LIVE_PERMISSION_READY_CHECK",
            "proof queue, live-promotion, guardian-clear, and broker-fresh readiness checks all pass; perform a read-only live-permission readiness check only.",
        )
    return (
        "NO_ACTION_WAIT",
        "no rule selected actionable work and live permission readiness is not met; wait for refreshed evidence or operator state change.",
    )


def _key_blocker(
    *,
    selected_next_work_type: str,
    proof_state: dict[str, Any],
    payoff_state: dict[str, Any],
    harvest_state: dict[str, Any],
    scout_state: dict[str, Any],
    operator_review_state: dict[str, Any],
    artifact_health: dict[str, Any],
    schema_state: dict[str, Any],
) -> str:
    if selected_next_work_type == "PAYOFF_SHAPE_DIAGNOSIS":
        return "PAYOFF_SHAPE_DIAGNOSIS_MISSING_OR_STALE"
    if selected_next_work_type == "HARVEST_PROOF_PATH":
        return "MIXED_HARVEST_PRIMARY_CLOSEST_HARVEST_UNSELECTED"
    if selected_next_work_type == "SCOUT_PLAN":
        return f"SCOUT_UNDIAGNOSED:{harvest_state.get('closest_harvest_candidate_id')}"
    if selected_next_work_type == "OPERATOR_REVIEW_REPORT":
        if operator_review_state.get("normal_routing_allowed") is not False:
            return f"SCOUT_BLOCKED_OPERATOR_REVIEW:NORMAL_ROUTING_{operator_review_state.get('normal_routing_allowed')}"
        return "SCOUT_BLOCKED_OPERATOR_REVIEW:NORMAL_ROUTING_FALSE"
    if selected_next_work_type == "READ_ONLY_EVIDENCE_REFRESH":
        return ",".join(
            sorted(
                [str(item.get("issue")) for item in artifact_health.get("explicit_issues", [])]
                + _string_list(artifact_health.get("artifact_contradiction_codes"))
            )
        )
    if selected_next_work_type == "CODE_REPAIR":
        return ",".join(
            sorted(
                [f"{item.get('artifact')}:{item.get('field')}" for item in schema_state.get("missing_fields", [])]
                + [str(item.get("issue")) for item in schema_state.get("success_condition_schema_issues", [])]
            )
        )
    if selected_next_work_type == "LIVE_PERMISSION_READY_CHECK":
        return "READY_CHECK_ONLY"
    if selected_next_work_type == "EDGE_IMPROVEMENT_EXPERIMENT":
        return f"EDGE_IMPROVEMENT:{harvest_state.get('closest_harvest_candidate_id')}:PROOF_QUEUE_{proof_state.get('proof_queue_count')}"
    return _first_str(
        proof_state.get("primary_blocker"),
        payoff_state.get("verdict"),
        scout_state.get("status"),
        ",".join(operator_review_state.get("blocker_codes") or []),
        "NO_PROGRESS",
    )


def _repeat_loop_guard(
    *,
    output_path: Path,
    artifact_index: dict[str, Any],
    selected_next_work_type: str,
    key_blocker: str,
) -> dict[str, Any]:
    previous = _read_previous_output(output_path)
    artifact_hash = artifact_index["combined_sha256"]
    current = {
        "work_type": selected_next_work_type,
        "artifact_hash": artifact_hash,
        "key_blocker": key_blocker,
        "latest_input_generated_at_utc": artifact_index.get("latest_input_generated_at_utc"),
    }
    previous_guard = (
        previous.get("repeat_loop_guard") if isinstance(previous.get("repeat_loop_guard"), dict) else {}
    )
    previous_fingerprint = (
        previous_guard.get("current_fingerprint")
        if isinstance(previous_guard.get("current_fingerprint"), dict)
        else {}
    )
    same = bool(previous_fingerprint) and all(
        previous_fingerprint.get(key) == value for key, value in current.items()
    )
    return {
        "policy": "Do not repeat the same work type for the same artifact hash and key blocker. Repeat only after input generated_at, artifact hash, or key blocker changes.",
        "current_fingerprint": current,
        "previous_fingerprint": previous_fingerprint or None,
        "same_work_type_artifact_and_blocker": same,
        "repeat_allowed": not same,
        "repeat_reason": (
            "new_or_changed_input_artifact"
            if not same
            else "same_work_type_artifact_hash_and_key_blocker"
        ),
    }


def _current_phase(
    *,
    selected_next_work_type: str,
    payoff_state: dict[str, Any],
    harvest_state: dict[str, Any],
    scout_state: dict[str, Any],
    operator_review_state: dict[str, Any],
    edge_improvement_state: dict[str, Any],
) -> str:
    if selected_next_work_type == "PAYOFF_SHAPE_DIAGNOSIS":
        return "PAYOFF_SHAPE_DIAGNOSIS_REFRESH_REQUIRED"
    if selected_next_work_type == "HARVEST_PROOF_PATH":
        return "HARVEST_PROOF_PATH_REQUIRED"
    if selected_next_work_type == "SCOUT_PLAN":
        return "SCOUT_PLAN_REQUIRED"
    if selected_next_work_type == "OPERATOR_REVIEW_REPORT":
        return "SCOUT_BLOCKED_OPERATOR_REVIEW"
    if selected_next_work_type == "READ_ONLY_EVIDENCE_REFRESH":
        return "ARTIFACT_EVIDENCE_REFRESH_REQUIRED"
    if selected_next_work_type == "CODE_REPAIR":
        return "GOAL_LOOP_SCHEMA_REPAIR_REQUIRED"
    if selected_next_work_type == "LIVE_PERMISSION_READY_CHECK":
        return "LIVE_PERMISSION_READY_CHECK_ONLY"
    if selected_next_work_type == "EDGE_IMPROVEMENT_EXPERIMENT":
        return "HARVEST_EDGE_IMPROVEMENT_EXPERIMENT"
    return _first_str(
        scout_state.get("status"),
        harvest_state.get("status"),
        payoff_state.get("verdict"),
        operator_review_state.get("normal_routing_status"),
        "NO_ACTION_WAIT",
    )


def _success_condition(work_type: str) -> dict[str, Any]:
    definitions: dict[str, tuple[str, str, list[dict[str, Any]]]] = {
        "PAYOFF_SHAPE_DIAGNOSIS": (
            "any",
            "payoff_shape_diagnosis is present, fresh, and has a machine-readable verdict.",
            [
                {"field": "payoff_artifact_status", "operator": "eq", "value": "present"},
                {"field": "payoff_stale", "operator": "eq", "value": False},
                {"field": "payoff_verdict", "operator": "in", "value": ["MIXED_HARVEST_PRIMARY", "RUNNER_PRIMARY", "NO_TRADE"]},
            ],
        ),
        "HARVEST_PROOF_PATH": (
            "all",
            "closest harvest candidate is selected without creating live permission.",
            [
                {"field": "closest_harvest_selected", "operator": "eq", "value": True},
                {"field": "harvest_live_promotion_allowed", "operator": "eq", "value": False},
                {"field": "live_permission_allowed", "operator": "eq", "value": False},
            ],
        ),
        "SCOUT_PLAN": (
            "all",
            "exact scout plan is diagnosed and remains read-only unless operator approval and proof gates exist.",
            [
                {"field": "scout_diagnosed", "operator": "eq", "value": True},
                {"field": "scout_live_side_effects_empty", "operator": "eq", "value": True},
                {"field": "live_permission_allowed", "operator": "eq", "value": False},
            ],
        ),
        "OPERATOR_REVIEW_REPORT": (
            "all",
            "operator-review report is limited to SCOUT approval/rejection evidence and never creates live permission.",
            [
                {"field": "scout_status", "operator": "eq", "value": "SCOUT_BLOCKED_OPERATOR_REVIEW"},
                {"field": "operator_review_material_only", "operator": "eq", "value": True},
                {"field": "live_permission_allowed", "operator": "eq", "value": False},
            ],
        ),
        "READ_ONLY_EVIDENCE_REFRESH": (
            "all",
            "stale or contradicted artifact markers disappear after read-only refresh.",
            [
                {"field": "has_stale_or_contradicted_artifact", "operator": "eq", "value": False},
                {"field": "live_permission_allowed", "operator": "eq", "value": False},
            ],
        ),
        "CODE_REPAIR": (
            "all",
            "schema and success_condition checks are repaired.",
            [
                {"field": "required_field_missing", "operator": "eq", "value": False},
                {"field": "success_condition_schema_ok", "operator": "eq", "value": True},
                {"field": "live_permission_allowed", "operator": "eq", "value": False},
            ],
        ),
        "LIVE_PERMISSION_READY_CHECK": (
            "all",
            "read-only readiness check confirms every prerequisite; it still does not grant live permission.",
            [
                {"field": "proof_queue_count", "operator": "gt", "value": 0},
                {"field": "can_create_live_permission_count", "operator": "gt", "value": 0},
                {"field": "harvest_live_promotion_allowed", "operator": "eq", "value": True},
                {"field": "guardian_clear", "operator": "eq", "value": True},
                {"field": "broker_truth_fresh", "operator": "eq", "value": True},
                {"field": "live_permission_allowed", "operator": "eq", "value": False},
            ],
        ),
        "EDGE_IMPROVEMENT_EXPERIMENT": (
            "all",
            "read-only entry/exit/payoff/sampling experiment is available for a HARVEST candidate without granting live permission.",
            [
                {"field": "edge_experiment_candidate", "operator": "eq", "value": True},
                {"field": "harvest_live_promotion_allowed", "operator": "eq", "value": False},
                {"field": "live_permission_allowed", "operator": "eq", "value": False},
            ],
        ),
        "NO_ACTION_WAIT": (
            "all",
            "no repeated work is requested until input artifacts or key blocker change.",
            [
                {"field": "repeat_allowed", "operator": "eq", "value": True},
                {"field": "live_permission_allowed", "operator": "eq", "value": False},
            ],
        ),
    }
    mode, description, checks = definitions[work_type]
    return {
        "schema_version": SUCCESS_CONDITION_SCHEMA_VERSION,
        "mode": mode,
        "description": description,
        "checks": checks,
        "verification_scope": "Evaluate against the next data/trader_goal_loop_orchestrator.json current_state fields after artifact refresh.",
    }


def _current_state_for_evaluation(
    *,
    proof_state: dict[str, Any],
    payoff_state: dict[str, Any],
    harvest_state: dict[str, Any],
    scout_state: dict[str, Any],
    operator_review_state: dict[str, Any],
    edge_improvement_state: dict[str, Any],
    artifact_health: dict[str, Any],
    schema_state: dict[str, Any],
    live_ready_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "payoff_artifact_status": payoff_state.get("artifact_status"),
        "payoff_stale": payoff_state.get("stale"),
        "payoff_verdict": payoff_state.get("verdict"),
        "closest_harvest_selected": harvest_state.get("closest_harvest_selected"),
        "harvest_live_promotion_allowed": harvest_state.get("live_promotion_allowed"),
        "scout_diagnosed": scout_state.get("diagnosed"),
        "scout_status": scout_state.get("status"),
        "scout_live_side_effects_empty": True,
        "normal_routing_allowed": operator_review_state.get("normal_routing_allowed"),
        "operator_review_material_only": True,
        "guardian_clear": operator_review_state.get("guardian_clear"),
        "edge_experiment_candidate": edge_improvement_state.get("candidate_available"),
        "has_stale_or_contradicted_artifact": artifact_health.get("has_stale_or_contradicted_artifact"),
        "required_field_missing": schema_state.get("required_field_missing"),
        "success_condition_schema_ok": schema_state.get("success_condition_schema_ok"),
        "proof_queue_count": proof_state.get("proof_queue_count"),
        "can_create_live_permission_count": proof_state.get("can_create_live_permission_count"),
        "broker_truth_fresh": (live_ready_state.get("checks") or {}).get("broker_truth_fresh"),
        "repeat_allowed": True,
        "live_permission_allowed": False,
    }


def _evaluate_success_condition(condition: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    mode = condition.get("mode") if condition.get("mode") in {"all", "any"} else "all"
    checks = []
    for check in condition.get("checks") or []:
        if not isinstance(check, dict):
            continue
        actual = state.get(str(check.get("field")))
        passed = _compare(actual, str(check.get("operator")), check.get("value"))
        checks.append(
            {
                "field": check.get("field"),
                "operator": check.get("operator"),
                "expected": check.get("value"),
                "actual": actual,
                "passed": passed,
            }
        )
    passed = all(item["passed"] for item in checks) if mode == "all" else any(item["passed"] for item in checks)
    return {
        "schema_version": "success_condition_evaluation_v1",
        "mode": mode,
        "checks": checks,
        "passed": bool(passed),
        "status": "MET" if passed else "NOT_MET",
    }


def _next_allowed_commands(work_type: str) -> list[str]:
    commands = {
        "PAYOFF_SHAPE_DIAGNOSIS": [
            "PYTHONPATH=src python3 -m quant_rabbit.cli payoff-shape-diagnosis",
            "python3 -m json.tool data/payoff_shape_diagnosis.json >/dev/null",
        ],
        "HARVEST_PROOF_PATH": [
            "python3 -m json.tool data/payoff_shape_diagnosis.json >/dev/null",
            "python3 -m json.tool data/harvest_live_grade_path.json >/dev/null",
        ],
        "SCOUT_PLAN": [
            "python3 -m json.tool data/eurusd_short_breakout_failure_scout_plan.json >/dev/null",
        ],
        "OPERATOR_REVIEW_REPORT": [
            "python3 -m json.tool data/trader_goal_loop_orchestrator.json >/dev/null",
            "PYTHONPATH=src python3 -m unittest tests.test_trader_goal_loop_orchestrator -v",
        ],
        "READ_ONLY_EVIDENCE_REFRESH": [
            "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
            "PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop",
            "PYTHONPATH=src python3 -m quant_rabbit.cli as-4x-proof-path",
            "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator",
            "PYTHONPATH=src python3 -m quant_rabbit.cli trader-goal-loop-orchestrator",
        ],
        "CODE_REPAIR": [
            "PYTHONPATH=src python3 -m unittest tests.test_trader_goal_loop_orchestrator -v",
            "PYTHONPATH=src python3 -m unittest tests.test_trader_repair_orchestrator -v",
        ],
        "LIVE_PERMISSION_READY_CHECK": [
            "python3 -m json.tool data/trader_goal_loop_orchestrator.json >/dev/null",
            "PYTHONPATH=src python3 -m unittest tests.test_trader_goal_loop_orchestrator -v",
        ],
        "EDGE_IMPROVEMENT_EXPERIMENT": [
            "python3 -m json.tool data/trader_goal_loop_orchestrator.json >/dev/null",
            "PYTHONPATH=src python3 -m unittest tests.test_trader_goal_loop_orchestrator -v",
        ],
        "NO_ACTION_WAIT": [
            "python3 -m json.tool data/trader_goal_loop_orchestrator.json >/dev/null",
        ],
    }
    return commands[work_type]


def _selected_next_prompt(
    *,
    selected_next_work_type: str,
    current_phase: str,
    selection_reason: str,
    four_x_progress_hypothesis: str,
    root_improvement_target: str,
    expected_edge_improvement: str,
    success_condition: dict[str, Any],
    next_allowed_commands: list[str],
) -> str:
    input_artifacts = "\n".join(f"- data/{_artifact_filename(name)}" for name in INPUT_ARTIFACT_NAMES)
    commands = "\n".join(f"- `{command}`" for command in next_allowed_commands)
    success = json.dumps(success_condition, ensure_ascii=False, indent=2, sort_keys=True)
    return f"""QuantRabbit の 4x改善ループで `{selected_next_work_type}` を実施してください。

目的:
現在の artifact 状態を読み、次の改善アクションを read-only で進める。これは live permission ではありません。発注許可、cancel/close 許可、launchd 操作許可を作らず、改善ループの次アクションだけを扱ってください。

現在フェーズ:
`{current_phase}`

選定理由:
{selection_reason}

4xに近づく根本改善は何か:
{root_improvement_target}

4x progress hypothesis:
{four_x_progress_hypothesis}

expected edge improvement:
{expected_edge_improvement}

入力 artifact:
{input_artifacts}

存在しない artifact は `missing` として扱い、推測で補完しないでください。

絶対禁止:
- live order / cancel / close / launchd load/reload を行わない
- gate を緩めない
- negative expectancy / month-scale replay negative を隠さない
- proof_queue_count=0 を live permission と誤認しない
- 4x未達額から lot を逆算しない
- secret / token を出さない
- QuantRabbit code から model API を呼ばない

出力:
- read-only の JSON または Markdown report を作る場合は `live_side_effects=[]` と `live_permission_allowed=false` を明記する
- 既存 artifact の blocker を消したように見せない
- operator の明示判断が必要な状態では、判断材料だけを整理し、operator decision 自体を捏造しない
- `OPERATOR_REVIEW_REPORT` の場合は、SCOUTを許可/却下するための判断材料だけを整理し、live permission とは明確に切り分ける
- `EDGE_IMPROVEMENT_EXPERIMENT` の場合は、HARVEST候補の entry / exit / payoff / sampling 実験設計だけを行い、発注や gate 緩和をしない

success condition:
```json
{success}
```

検証コマンド:
{commands}

注意:
この作業は live permission ready check であっても発注許可ではありません。最終送信は既存 gateway、fresh broker truth、risk/gpt verifier、guardian/operator review clearance、明示 live flag が別途そろうまで禁止です。
"""


def _four_x_progress_hypothesis(edge_state: dict[str, Any], scout_state: dict[str, Any]) -> str:
    target = edge_state.get("target_shape") or scout_state.get("target_shape") or "closest HARVEST candidate"
    proof_gap = edge_state.get("proof_gap_trades")
    if proof_gap is not None:
        return (
            f"{target} の attached-TP HARVEST 証拠を proof floor まで進め、market-close leak と "
            f"month-scale negative を隠さず除外/修復できれば、rolling 30d funding-adjusted equity 4x に近づく "
            f"正の期待値レーンを増やせる。現在の不足サンプルは {proof_gap}。"
        )
    return (
        f"{target} の entry/exit/payoff/sampling を read-only で分解し、NO_TRADE に落とすべき形と "
        "HARVEST live-grade 化すべき形を分けることで 4x に必要な正の期待値候補を増やす。"
    )


def _root_improvement_target(edge_state: dict[str, Any], scout_state: dict[str, Any]) -> str:
    target = edge_state.get("target_shape") or scout_state.get("target_shape") or "HARVEST candidate"
    return (
        f"{target} を、発注許可ではなく read-only の SCOUT 判断材料と期待値改善実験で live-grade HARVEST 候補へ近づける。"
    )


def _expected_edge_improvement(edge_state: dict[str, Any], scout_state: dict[str, Any]) -> str:
    wins = edge_state.get("take_profit_trades")
    losses = edge_state.get("take_profit_losses")
    expectancy = edge_state.get("take_profit_expectancy_jpy")
    proof_gap = edge_state.get("proof_gap_trades")
    cap = scout_state.get("max_loss_jpy_cap")
    return (
        f"TP proof {wins}勝 / {losses} TP負け、期待値 {expectancy} JPY、proof gap {proof_gap} sample、"
        f"max_loss_jpy_cap {cap} を起点に、追加証拠で HARVEST の薄い正期待値を補強し、"
        "market-close leak / negative expectancy / month-scale replay negative を隠さず NO_TRADE 除外へ回す。"
    )


def _render_report(payload: dict[str, Any]) -> str:
    proof = payload.get("proof_state", {})
    payoff = payload.get("payoff_state", {})
    harvest = payload.get("harvest_state", {})
    scout = payload.get("scout_state", {})
    operator = payload.get("operator_review_state", {})
    lines = [
        "# Trader Goal Loop Orchestrator",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Read only: `{payload.get('read_only')}`",
        f"- Live side effects: `{payload.get('live_side_effects')}`",
        f"- Live permission allowed: `{payload.get('live_permission_allowed')}`",
        f"- Current phase: `{payload.get('current_phase')}`",
        f"- Selected next work type: `{payload.get('selected_next_work_type')}`",
        f"- Selection reason: {payload.get('selection_reason')}",
        f"- Four x progress hypothesis: {payload.get('four_x_progress_hypothesis')}",
        f"- Root improvement target: {payload.get('root_improvement_target')}",
        f"- Expected edge improvement: {payload.get('expected_edge_improvement')}",
        "",
        "## Key State",
        "",
        f"- Payoff verdict: `{payoff.get('verdict')}` / stale=`{payoff.get('stale')}`",
        f"- HARVEST closest: `{harvest.get('closest_harvest_candidate_id')}` / live promotion allowed=`{harvest.get('live_promotion_allowed')}`",
        f"- Scout status: `{scout.get('status')}` / allowed=`{scout.get('scout_mode_allowed')}`",
        f"- Proof queue count: `{proof.get('proof_queue_count')}`",
        f"- Can create live permission count: `{proof.get('can_create_live_permission_count')}`",
        f"- Normal routing allowed: `{operator.get('normal_routing_allowed')}`",
        f"- Guardian clear: `{operator.get('guardian_clear')}`",
        "",
        "## Repeat Guard",
        "",
        f"- Repeat allowed: `{payload.get('repeat_loop_guard', {}).get('repeat_allowed')}`",
        f"- Key blocker: `{payload.get('repeat_loop_guard', {}).get('current_fingerprint', {}).get('key_blocker')}`",
        "",
        "## Success Condition Evaluation",
        "",
        f"- Status: `{payload.get('success_condition_evaluation', {}).get('status')}`",
        "",
        "## Next Allowed Commands",
        "",
    ]
    for command in payload.get("next_allowed_commands") or []:
        lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "## Safety Boundary",
            "",
            "- This artifact is not live permission.",
            "- Order send, order cancel, position close, and launchd load/reload remain prohibited here.",
            "- Negative expectancy, month-scale replay negatives, and proof queue emptiness are surfaced as blockers.",
        ]
    )
    return "\n".join(lines) + "\n"


def _safety_contract() -> dict[str, Any]:
    return {
        "read_only": True,
        "live_side_effects_allowed": [],
        "live_permission_allowed": False,
        "proof_queue_count_zero_is_not_permission": True,
        "no_4x_deficit_lot_backsolve": True,
        "quant_rabbit_code_may_call_model_api": False,
        "forbidden_actions": [
            "live_order",
            "cancel_order",
            "close_position",
            "launchd_load",
            "launchd_reload",
            "gate_relaxation",
            "secret_or_token_output",
        ],
    }


def _read_previous_output(output_path: Path) -> dict[str, Any]:
    try:
        if not output_path.exists():
            return {}
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _repair_current_state(repair: dict[str, Any]) -> dict[str, Any]:
    loop = repair.get("loop_engineering_prompt") if isinstance(repair.get("loop_engineering_prompt"), dict) else {}
    state = loop.get("current_state") if isinstance(loop.get("current_state"), dict) else {}
    return state


def _normal_routing_from_scout(scout_state: dict[str, Any]) -> bool | None:
    for blocker in scout_state.get("proof_queue_entry_blockers") or []:
        if not isinstance(blocker, dict):
            continue
        evidence = blocker.get("evidence") if isinstance(blocker.get("evidence"), dict) else {}
        value = evidence.get("normal_routing_allowed")
        if isinstance(value, bool):
            return value
    return None


def _guardian_markers(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    markers: list[str] = []
    for artifact in artifacts.values():
        _collect_guardian_strings(artifact, markers)
    return _unique(markers)


def _collect_guardian_strings(value: Any, markers: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_upper = str(key).upper()
            if isinstance(item, str):
                item_upper = item.upper()
                key_matches = any(
                    token in key_upper
                    for token in (
                        "GUARDIAN",
                        "OPERATOR_REVIEW",
                        "RECEIPT",
                        "LIFECYCLE",
                    )
                )
                item_matches = any(
                    token in item_upper
                    for token in (
                        "GUARDIAN_RECEIPT",
                        "GUARDIAN RECEIPT",
                        "OPERATOR_REVIEW",
                        "OPERATOR REVIEW",
                        "NEEDS_OPERATOR_REVIEW",
                    )
                )
                lifecycle_matches = "LIFECYCLE" in key_upper and item_upper in {
                    "ACTIVE",
                    "EXPIRED",
                    "STALE",
                    "CONSUMED",
                    "REJECTED",
                    "HISTORICAL_ONLY",
                }
                if len(item_upper) <= 240 and (key_matches or item_matches or lifecycle_matches):
                    markers.append(item_upper)
            else:
                _collect_guardian_strings(item, markers)
    elif isinstance(value, list):
        for item in value:
            _collect_guardian_strings(item, markers)


def _explicit_artifact_issue(artifact: dict[str, Any]) -> str | None:
    status = str(artifact.get("status") or "").upper()
    if "CONTRADICTED" in status:
        return "CONTRADICTED"
    if status == "STALE" or status.endswith("_STALE") or "ARTIFACT_STALE" in status:
        return "STALE"
    return None


def _freshness_issues(repair: dict[str, Any]) -> list[dict[str, Any]]:
    reason = repair.get("proof_queue_empty_reason") if isinstance(repair.get("proof_queue_empty_reason"), dict) else {}
    categories = reason.get("categories") if isinstance(reason.get("categories"), list) else []
    issues = []
    for row in categories:
        if not isinstance(row, dict):
            continue
        freshness = row.get("freshness") if isinstance(row.get("freshness"), dict) else {}
        status = str(freshness.get("status") or "").upper()
        if status in {"STALE", "CONTRADICTED"}:
            issues.append(
                {
                    "category": row.get("category"),
                    "status": status,
                    "source": freshness.get("source"),
                    "generated_at_utc": freshness.get("generated_at_utc"),
                    "reference_timestamp": freshness.get("freshness_reference_timestamp"),
                }
            )
    return issues


def _success_condition_schema_issues(repair: dict[str, Any]) -> list[dict[str, str]]:
    if repair.get("_artifact_status") == "missing":
        return []
    actions = []
    top_actions = repair.get("next_evidence_actions")
    if isinstance(top_actions, list):
        actions.extend(top_actions)
    state_actions = _repair_current_state(repair).get("next_evidence_actions")
    if isinstance(state_actions, list):
        actions.extend(state_actions)
    issues: list[dict[str, str]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            issues.append({"artifact": "trader_repair_orchestrator", "issue": f"next_evidence_actions[{index}] not object"})
            continue
        condition = action.get("success_condition")
        if not isinstance(condition, dict):
            issues.append({"artifact": "trader_repair_orchestrator", "issue": f"{action.get('action_id') or index}: missing success_condition"})
            continue
        checks = condition.get("checks")
        if condition.get("schema_version") != SUCCESS_CONDITION_SCHEMA_VERSION:
            issues.append({"artifact": "trader_repair_orchestrator", "issue": f"{action.get('action_id') or index}: success_condition schema_version missing"})
        if condition.get("mode") not in {"all", "any"}:
            issues.append({"artifact": "trader_repair_orchestrator", "issue": f"{action.get('action_id') or index}: success_condition mode invalid"})
        if not isinstance(checks, list) or not checks:
            issues.append({"artifact": "trader_repair_orchestrator", "issue": f"{action.get('action_id') or index}: success_condition checks missing"})
    return issues


def _codes_from_blockers(value: Any) -> list[str]:
    codes: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                code = item.get("code")
                if code:
                    codes.append(str(code))
            elif isinstance(item, str):
                codes.append(item.split(":", 1)[0])
    elif isinstance(value, dict):
        code = value.get("code")
        if code:
            codes.append(str(code))
    return _unique(codes)


def _artifact_filename(name: str) -> str:
    return {
        "trader_repair_orchestrator": "trader_repair_orchestrator.json",
        "payoff_shape_diagnosis": "payoff_shape_diagnosis.json",
        "harvest_live_grade_path": "harvest_live_grade_path.json",
        "eurusd_short_breakout_failure_scout_plan": "eurusd_short_breakout_failure_scout_plan.json",
        "as_proof_pack_queue": "as_proof_pack_queue.json",
        "as_lane_candidate_board": "as_lane_candidate_board.json",
        "portfolio_4x_path_planner": "portfolio_4x_path_planner.json",
        "live_order_request": "live_order_request.json",
        "broker_snapshot": "broker_snapshot.json",
    }[name]


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if operator in {"eq", "=="}:
        return actual == expected
    if operator in {"neq", "!="}:
        return actual != expected
    if operator == "gt":
        return _float(actual) is not None and _float(expected) is not None and _float(actual) > _float(expected)
    if operator == "gte" or operator == ">=":
        return _float(actual) is not None and _float(expected) is not None and _float(actual) >= _float(expected)
    if operator == "lt":
        return _float(actual) is not None and _float(expected) is not None and _float(actual) < _float(expected)
    if operator == "lte" or operator == "<=":
        return _float(actual) is not None and _float(expected) is not None and _float(actual) <= _float(expected)
    if operator == "in":
        return isinstance(expected, list) and actual in expected
    return False


def _first_int(*values: Any) -> int:
    for value in values:
        parsed = _int(value)
        if parsed is not None:
            return parsed
    return 0


def _first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _first_bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        return [value]
    return []


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)

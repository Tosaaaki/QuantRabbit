from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_ACTIVE_TRADER_CONTRACT,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
    DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
    DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR,
    DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR_REPORT,
    DEFAULT_TRADER_REPAIR_ORCHESTRATOR,
)
from quant_rabbit.trader_repair_orchestrator import (
    validated_evidence_action_material,
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
    "ACTIVE_TRADER_CONTRACT_EVIDENCE",
    "ACTIVE_LANE_EVIDENCE_DISPATCH",
}
INPUT_ARTIFACT_NAMES = (
    "trader_repair_orchestrator",
    "active_trader_contract",
    "payoff_shape_diagnosis",
    "harvest_live_grade_path",
    "eurusd_short_breakout_failure_scout_plan",
    "as_proof_pack_queue",
    "as_lane_candidate_board",
    "portfolio_4x_path_planner",
    "guardian_receipt_consumption",
    "guardian_receipt_operator_review",
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
REPAIR_MATERIAL_EVIDENCE_WAIT_STATUS = "WAITING_FOR_MATERIAL_EVIDENCE"
REPAIR_ACTION_MAPPING_REQUIRED_STATUS = "ACTIVE_LANE_ACTION_MAPPING_REQUIRED"
REPEAT_MATERIAL_HISTORY_LIMIT = 64


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
        active_trader_contract_path: Path = DEFAULT_ACTIVE_TRADER_CONTRACT,
        payoff_shape_diagnosis_path: Path = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
        harvest_live_grade_path: Path = DEFAULT_HARVEST_LIVE_GRADE_PATH,
        scout_plan_path: Path = DEFAULT_EURUSD_SHORT_BREAKOUT_FAILURE_SCOUT_PLAN,
        as_proof_pack_queue_path: Path = DEFAULT_AS_PROOF_PACK_QUEUE,
        as_lane_candidate_board_path: Path = DEFAULT_AS_LANE_CANDIDATE_BOARD,
        portfolio_4x_path_planner_path: Path = DEFAULT_PORTFOLIO_4X_PATH_PLANNER,
        guardian_receipt_consumption_path: Path = DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
        guardian_receipt_operator_review_path: Path = DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
        live_order_request_path: Path = DEFAULT_LIVE_ORDER_REQUEST,
        broker_snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        output_path: Path = DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR,
        report_path: Path = DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR_REPORT,
        now_utc: datetime | None = None,
    ) -> None:
        self.paths = {
            "trader_repair_orchestrator": trader_repair_orchestrator_path,
            "active_trader_contract": active_trader_contract_path,
            "payoff_shape_diagnosis": payoff_shape_diagnosis_path,
            "harvest_live_grade_path": harvest_live_grade_path,
            "eurusd_short_breakout_failure_scout_plan": scout_plan_path,
            "as_proof_pack_queue": as_proof_pack_queue_path,
            "as_lane_candidate_board": as_lane_candidate_board_path,
            "portfolio_4x_path_planner": portfolio_4x_path_planner_path,
            "guardian_receipt_consumption": guardian_receipt_consumption_path,
            "guardian_receipt_operator_review": guardian_receipt_operator_review_path,
            "live_order_request": live_order_request_path,
            "broker_snapshot": broker_snapshot_path,
        }
        self.output_path = output_path
        self.report_path = report_path
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> TraderGoalLoopOrchestratorSummary:
        # Serialize goal writers and hold the repair artifact's shared sibling
        # lock from read through publish. An ACK/refresh writer therefore cannot
        # change the authoritative repair state midway through this handoff,
        # while a newer queued goal run will rebuild after this one releases.
        with _artifact_state_lock(self.output_path, exclusive=True):
            with _artifact_state_lock(
                self.paths["trader_repair_orchestrator"],
                exclusive=False,
            ):
                payload = self.build_payload()
                _atomic_publish_goal_outputs(
                    output_path=self.output_path,
                    output_content=(
                        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                        + "\n"
                    ),
                    report_path=self.report_path,
                    report_content=_render_report(payload),
                )
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
        repair_loop_state = _repair_loop_state(artifacts)
        active_contract_state = _active_contract_state(artifacts, self.now_utc)
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
            active_contract_state=active_contract_state,
            repair_loop_state=repair_loop_state,
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
            active_contract_state=active_contract_state,
            repair_loop_state=repair_loop_state,
        )
        classified_success_condition = _success_condition(candidate_work_type)
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
            active_contract_state=active_contract_state,
            repair_loop_state=repair_loop_state,
        )
        repeat_loop_guard = _repeat_loop_guard(
            output_path=self.output_path,
            artifact_index=artifact_index,
            selected_next_work_type=candidate_work_type,
            key_blocker=key_blocker,
            material_state=_repeat_material_state(
                proof_state=proof_state,
                payoff_state=payoff_state,
                harvest_state=harvest_state,
                scout_state=scout_state,
                operator_review_state=operator_review_state,
                edge_improvement_state=edge_improvement_state,
                artifact_health=artifact_health,
                schema_state=schema_state,
                active_contract_state=active_contract_state,
                repair_loop_state=repair_loop_state,
                live_order_state=_live_order_repeat_state(
                    artifacts["live_order_request"]
                ),
            ),
        )
        if repeat_loop_guard["material_history_hit"]:
            selection_reason += (
                " repeat_loop_guard found this canonical material state in bounded history; "
                "generated_at, raw artifact hash, broker age, and timestamp-only classification churn "
                "are not progress, so the classified task is recorded but not redispatched."
            )
        work_dispatch_allowed = _work_dispatch_allowed(
            candidate_work_type,
            repeat_loop_guard=repeat_loop_guard,
            repair_loop_state=repair_loop_state,
        )
        repeat_suppressed = bool(
            candidate_work_type != "NO_ACTION_WAIT"
            and not work_dispatch_allowed
            and repeat_loop_guard.get("material_history_hit") is True
        )
        selected_next_work_type = (
            "NO_ACTION_WAIT" if repeat_suppressed else candidate_work_type
        )
        current_phase = (
            "WAITING_FOR_NEW_MATERIAL_STATE"
            if repeat_suppressed
            else _current_phase(
                selected_next_work_type=selected_next_work_type,
                payoff_state=payoff_state,
                harvest_state=harvest_state,
                scout_state=scout_state,
                operator_review_state=operator_review_state,
                edge_improvement_state=edge_improvement_state,
                active_contract_state=active_contract_state,
                repair_loop_state=repair_loop_state,
            )
        )
        success_condition = _success_condition(selected_next_work_type)
        current_state = {
            **current_state,
            "work_dispatch_allowed": work_dispatch_allowed,
            "repeat_suppressed": repeat_suppressed,
        }
        success_condition_evaluation = _evaluate_success_condition(
            success_condition,
            current_state,
        )
        next_allowed_commands = (
            _next_allowed_commands(
                selected_next_work_type,
                repair_loop_state=repair_loop_state,
            )
            if work_dispatch_allowed
            else []
        )
        four_x_progress_hypothesis = _four_x_progress_hypothesis(
            edge_improvement_state,
            scout_state,
            active_contract_state=active_contract_state,
        )
        root_improvement_target = _root_improvement_target(
            edge_improvement_state,
            scout_state,
            active_contract_state=active_contract_state,
        )
        expected_edge_improvement = _expected_edge_improvement(
            edge_improvement_state,
            scout_state,
            active_contract_state=active_contract_state,
        )
        selected_next_prompt = (
            _repeat_suppressed_prompt(
                classified_next_work_type=candidate_work_type,
                key_blocker=key_blocker,
                repeat_loop_guard=repeat_loop_guard,
            )
            if repeat_suppressed
            else _selected_next_prompt(
                selected_next_work_type=selected_next_work_type,
                current_phase=current_phase,
                selection_reason=selection_reason,
                four_x_progress_hypothesis=four_x_progress_hypothesis,
                root_improvement_target=root_improvement_target,
                expected_edge_improvement=expected_edge_improvement,
                success_condition=success_condition,
                next_allowed_commands=next_allowed_commands,
                active_contract_state=active_contract_state,
                repair_loop_state=repair_loop_state,
            )
        )
        payload = {
            "contract_version": CONTRACT_VERSION,
            "status": "NEXT_WORK_SELECTED",
            "generated_at_utc": self.now_utc.isoformat(),
            "read_only": True,
            "live_side_effects": [],
            "current_phase": current_phase,
            "selected_next_work_type": selected_next_work_type,
            "classified_next_work_type": candidate_work_type,
            "selected_next_prompt": selected_next_prompt,
            "selection_reason": selection_reason,
            "four_x_progress_hypothesis": four_x_progress_hypothesis,
            "root_improvement_target": root_improvement_target,
            "expected_edge_improvement": expected_edge_improvement,
            "proof_state": proof_state,
            "repair_loop_state": repair_loop_state,
            "payoff_state": payoff_state,
            "harvest_state": harvest_state,
            "scout_state": scout_state,
            "operator_review_state": operator_review_state,
            "edge_improvement_state": edge_improvement_state,
            "active_contract_state": active_contract_state,
            "repeat_loop_guard": repeat_loop_guard,
            "work_dispatch_allowed": work_dispatch_allowed,
            "repeat_suppressed": repeat_suppressed,
            "success_condition": success_condition,
            "success_condition_evaluation": success_condition_evaluation,
            "classified_success_condition": classified_success_condition,
            "next_allowed_commands": next_allowed_commands,
            "requires_operator_approval_for_this_report": False,
            "requires_operator_review_before_scout_or_routing": bool(
                operator_review_state.get("operator_review_required")
            ),
            "live_permission_allowed": False,
            "live_permission_ready_check_state": live_ready_state,
            "artifact_health": artifact_health,
            "schema_state": schema_state,
            "artifact_index": artifact_index,
            "safety_contract": _safety_contract(),
        }
        if selected_next_work_type == "ACTIVE_LANE_EVIDENCE_DISPATCH":
            payload["next_allowed_command_steps"] = list(
                repair_loop_state.get("active_lane_dispatch_command_steps") or []
            )
        if selected_next_work_type not in WORK_TYPES:
            raise ValueError(f"unknown selected_next_work_type: {selected_next_work_type}")
        return payload


def _prepare_atomic_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _replace_prepared_text(temp_path: Path, path: Path) -> None:
    os.replace(temp_path, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_publish_goal_outputs(
    *,
    output_path: Path,
    output_content: str,
    report_path: Path,
    report_content: str,
) -> None:
    """Stage both files, then commit derivative report and JSON state last."""

    output_temp = _prepare_atomic_text(output_path, output_content)
    try:
        report_temp = _prepare_atomic_text(report_path, report_content)
    except BaseException:
        output_temp.unlink(missing_ok=True)
        raise
    try:
        _replace_prepared_text(report_temp, report_path)
        _replace_prepared_text(output_temp, output_path)
    finally:
        report_temp.unlink(missing_ok=True)
        output_temp.unlink(missing_ok=True)


@contextmanager
def _artifact_state_lock(path: Path, *, exclusive: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(handle.fileno(), mode)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def _active_contract_state(artifacts: dict[str, dict[str, Any]], now_utc: datetime) -> dict[str, Any]:
    artifact = artifacts["active_trader_contract"]
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "path": artifact.get("_path"),
            "active_prompt_available": False,
            "selected_active_path": None,
            "next_prompt": None,
            "next_trade_enabling_action": None,
            "stale": True,
            "live_permission_allowed": False,
        }
    generated_at = _parse_dt(artifact.get("generated_at_utc"))
    age_seconds = (now_utc - generated_at).total_seconds() if generated_at else None
    current = artifact.get("current_state") if isinstance(artifact.get("current_state"), dict) else {}
    board = current.get("active_opportunity_board") if isinstance(current.get("active_opportunity_board"), dict) else {}
    board_top = board.get("top_lane") if isinstance(board.get("top_lane"), dict) else {}
    frontier = (
        current.get("non_eurusd_live_grade_frontier")
        if isinstance(current.get("non_eurusd_live_grade_frontier"), dict)
        else {}
    )
    frontier_lane = (
        frontier.get("next_evidence_lane")
        if isinstance(frontier.get("next_evidence_lane"), dict)
        else {}
    )
    selected_path = str(artifact.get("selected_active_path") or "")
    next_prompt = artifact.get("next_prompt") if isinstance(artifact.get("next_prompt"), str) else ""
    next_action = (
        artifact.get("next_trade_enabling_action")
        if isinstance(artifact.get("next_trade_enabling_action"), str)
        else ""
    )
    stale = (
        generated_at is None
        or age_seconds is None
        or age_seconds > PAYOFF_STALE_AFTER_SECONDS
        or _explicit_artifact_issue(artifact) in {"STALE", "CONTRADICTED"}
    )
    active_prompt_available = bool(
        not stale
        and selected_path
        and selected_path != "NO_TRADE_WITH_CAUSE"
        and next_prompt
        and next_action
        and artifact.get("live_permission_allowed") is False
        and artifact.get("live_side_effects", []) == []
    )
    return {
        "artifact_status": "present",
        "path": artifact.get("_path"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "age_seconds": _round(age_seconds),
        "stale": bool(stale),
        "status": artifact.get("status"),
        "selected_active_path": selected_path or None,
        "active_prompt_available": active_prompt_available,
        "next_prompt": next_prompt or None,
        "next_trade_enabling_action": next_action or None,
        "selected_active_path_reason": artifact.get("selected_active_path_reason"),
        "target_shape": artifact.get("target_shape"),
        "four_x_progress_hypothesis": artifact.get("four_x_progress_hypothesis"),
        "root_improvement_target": artifact.get("root_improvement_target"),
        "expected_edge_improvement": artifact.get("expected_edge_improvement"),
        "top_lane_id": board_top.get("lane_id"),
        "top_lane_status": board_top.get("status"),
        "top_lane_vehicle": board_top.get("vehicle"),
        "frontier_lane_id": frontier_lane.get("lane_id"),
        "frontier_lane_vehicle": frontier_lane.get("vehicle"),
        "remaining_blockers": [
            row.get("code")
            for row in artifact.get("remaining_blockers", [])
            if isinstance(row, dict) and row.get("code")
        ],
        "live_permission_allowed": False,
    }


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


def _repair_loop_state(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    repair = artifacts["trader_repair_orchestrator"]
    if repair.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "status": None,
            "actionable_request_count": 0,
            "approval_required_request_count": 0,
            "waiting_request_count": 0,
            "repair_request_count": 0,
            "selected_request_code": None,
            "next_evidence_action_count": 0,
            "waiting_for_evidence": False,
            "actionable_repair_selected": False,
        }
    actions = repair.get("next_evidence_actions") if isinstance(repair.get("next_evidence_actions"), list) else []
    progress_receipts = (
        repair.get("evidence_action_progress")
        if isinstance(repair.get("evidence_action_progress"), list)
        else []
    )
    repair_current_state = _repair_current_state(repair)
    validated_actions: list[dict[str, Any]] = []
    validated_progress: list[dict[str, Any]] = []
    invalid_watermarks: list[str] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            invalid_watermarks.append(f"next_action:{index}:NOT_OBJECT")
            continue
        material = validated_evidence_action_material(
            action,
            current_state=repair_current_state,
            source="next_action",
        )
        if material is None:
            invalid_watermarks.append(
                f"next_action:{action.get('action_id') or index}:INVALID_WATERMARK"
            )
            continue
        validated_actions.append(material)
    for index, receipt in enumerate(progress_receipts):
        if not isinstance(receipt, dict):
            invalid_watermarks.append(f"progress_receipt:{index}:NOT_OBJECT")
            continue
        material = validated_evidence_action_material(
            receipt,
            current_state=repair_current_state,
            source="progress_receipt",
        )
        if material is None:
            invalid_watermarks.append(
                f"progress_receipt:{receipt.get('action_id') or index}:INVALID_WATERMARK"
            )
            continue
        validated_progress.append(material)
    evidence_action_material = {
        "next_actions": validated_actions,
        "progress_receipts": validated_progress,
    }
    evidence_action_material_sha256 = (
        _stable_json_sha256(evidence_action_material)
        if validated_actions or validated_progress
        else None
    )
    waiting_count = _first_int(repair.get("waiting_request_count"))
    actionable_count = _first_int(repair.get("actionable_request_count"))
    approval_count = _first_int(repair.get("approval_required_request_count"))
    selected_code = repair.get("selected_request_code") if isinstance(repair.get("selected_request_code"), str) else None
    work_order = repair.get("codex_work_order") if isinstance(repair.get("codex_work_order"), dict) else {}
    material_evidence_wait = bool(
        work_order.get("status") == REPAIR_MATERIAL_EVIDENCE_WAIT_STATUS
        and work_order.get("dispatch_allowed") is False
    )
    action_mapping_required = bool(
        work_order.get("status") == REPAIR_ACTION_MAPPING_REQUIRED_STATUS
        and work_order.get("dispatch_allowed") is False
    )
    active_lane_execution_pending = bool(
        work_order.get("status") == "READ_ONLY_EVIDENCE_WORK"
        and work_order.get("execution_pending") is True
        and work_order.get("pending_dispatch_id")
    )
    actionable_repair_selected = bool(actionable_count > 0 and selected_code)
    queued_evidence_wait = bool(
        str(repair.get("status") or "") == "ORCHESTRATOR_BLOCKED"
        and actionable_count == 0
        and approval_count == 0
        and waiting_count > 0
        and not selected_code
        and actions
    )
    waiting_for_evidence = queued_evidence_wait or material_evidence_wait
    return {
        "artifact_status": "present",
        "status": repair.get("status"),
        "generated_at_utc": repair.get("generated_at_utc"),
        "actionable_request_count": actionable_count,
        "approval_required_request_count": approval_count,
        "waiting_request_count": waiting_count,
        "repair_request_count": _first_int(repair.get("repair_request_count")),
        "selected_request_code": selected_code,
        "next_evidence_action_count": len(actions),
        "waiting_for_evidence": waiting_for_evidence,
        "queued_evidence_wait": queued_evidence_wait,
        "material_evidence_wait": material_evidence_wait,
        "action_mapping_required": action_mapping_required,
        "actionable_repair_selected": actionable_repair_selected,
        "active_lane_execution_pending": active_lane_execution_pending,
        "codex_work_order_status": work_order.get("status"),
        "codex_work_order_objective": work_order.get("objective"),
        "codex_work_order_automation_prompt": work_order.get("automation_prompt"),
        "codex_work_order_suggested_files": _string_list(
            work_order.get("suggested_files")
        ),
        "codex_work_order_required_tests": _string_list(
            work_order.get("required_tests")
        ),
        "codex_work_order_targeted_test_commands": _string_list(
            work_order.get("targeted_test_commands")
        ),
        "codex_work_order_verification_commands": _string_list(
            work_order.get("verification_commands")
        ),
        "codex_work_order_final_verification_commands": _string_list(
            work_order.get("final_verification_commands")
        ),
        "codex_work_order_reason_code": work_order.get("reason_code"),
        "active_lane_action_code": work_order.get("action_code"),
        "active_lane_material_digest": work_order.get("material_digest"),
        "active_lane_pending_dispatch_id": work_order.get("pending_dispatch_id"),
        "active_lane_dispatch_commands": _string_list(work_order.get("suggested_commands")),
        "active_lane_dispatch_command_steps": [
            dict(step)
            for step in work_order.get("suggested_command_steps") or []
            if isinstance(step, dict)
        ],
        "waiting_request_codes": _string_list(repair.get("waiting_request_codes")),
        "next_evidence_action_ids": [
            str(action.get("action_id"))
            for action in actions
            if isinstance(action, dict) and action.get("action_id")
        ],
        "next_evidence_action_material_sha256": evidence_action_material_sha256,
        "validated_evidence_action_count": len(validated_actions),
        "validated_evidence_progress_count": len(validated_progress),
        "invalid_evidence_action_watermark_count": len(invalid_watermarks),
        "invalid_evidence_action_watermarks": invalid_watermarks,
    }


def _operator_review_state(
    artifacts: dict[str, dict[str, Any]],
    proof_state: dict[str, Any],
    scout_state: dict[str, Any],
) -> dict[str, Any]:
    markers = _guardian_markers(artifacts)
    raw_guardian = _raw_guardian_receipt_state(artifacts)
    inferred_normal_routing_allowed = _first_bool(
        _normal_routing_from_scout(scout_state),
        proof_state.get("routing_allowed"),
    )
    normal_routing_allowed = _first_bool(
        raw_guardian.get("normal_routing_allowed"),
        inferred_normal_routing_allowed,
    )
    raw_guardian_clear = raw_guardian.get("normal_routing_allowed") is True
    raw_blocked = raw_guardian.get("normal_routing_allowed") is False
    raw_blocker_codes = _string_list(raw_guardian.get("blocker_codes"))
    inferred_blocker_codes = _unique(
        proof_state.get("global_blockers", [])
        + scout_state.get("blocker_codes", [])
        + [marker for marker in markers if "GUARDIAN" in marker or "OPERATOR_REVIEW" in marker]
    )
    if raw_guardian_clear:
        stale_guardian_blocker_codes = [
            code for code in inferred_blocker_codes if "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED" in code
        ]
        blocker_codes = _unique(
            raw_blocker_codes
            + [
                code
                for code in inferred_blocker_codes
                if "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED" not in code
            ]
        )
    else:
        stale_guardian_blocker_codes = []
        blocker_codes = _unique(raw_blocker_codes + inferred_blocker_codes)
    has_operator_review_blocker = any(
        "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED" in code for code in blocker_codes
    )
    operator_review_required = (
        raw_blocked
        or (normal_routing_allowed is False and has_operator_review_blocker)
        or (scout_state.get("status") == "SCOUT_BLOCKED_OPERATOR_REVIEW" and not raw_guardian_clear)
    )
    stale_or_expired = any("EXPIRED" in marker or "OPERATOR_REVIEW_STALE" in marker for marker in markers)
    guardian_clear = (
        raw_guardian_clear
        or (
            normal_routing_allowed is True
            and not operator_review_required
            and not has_operator_review_blocker
        )
    )
    return {
        "normal_routing_allowed": normal_routing_allowed,
        "inferred_normal_routing_allowed": inferred_normal_routing_allowed,
        "normal_routing_status": proof_state.get("normal_routing_status"),
        "operator_review_required": bool(operator_review_required),
        "guardian_clear": bool(guardian_clear),
        "guardian_markers": markers,
        "guardian_expired_or_operator_review_stale": bool(stale_or_expired),
        "blocker_codes": blocker_codes,
        "stale_guardian_blocker_codes_suppressed": stale_guardian_blocker_codes,
        "raw_guardian_receipt_state": raw_guardian,
        "source": (
            "raw_guardian_receipt_artifacts"
            if raw_guardian.get("artifact_present")
            else "input_artifact_summaries_only"
        ),
        "missing_raw_guardian_receipt_artifact": not raw_guardian.get("artifact_present"),
    }


def _raw_guardian_receipt_state(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    consumption = artifacts.get("guardian_receipt_consumption", {})
    review = artifacts.get("guardian_receipt_operator_review", {})
    consumption_present = consumption.get("_artifact_status") == "present"
    review_present = review.get("_artifact_status") == "present"
    values = [
        value
        for value in (
            consumption.get("normal_routing_allowed") if consumption_present else None,
            review.get("normal_routing_allowed") if review_present else None,
        )
        if isinstance(value, bool)
    ]
    if any(value is False for value in values):
        normal_routing_allowed: bool | None = False
    elif values and all(value is True for value in values):
        normal_routing_allowed = True
    else:
        normal_routing_allowed = None
    blocker_codes: list[str] = []
    if consumption_present and consumption.get("normal_routing_allowed") is False:
        blocker_codes.append(str(consumption.get("status") or "GUARDIAN_RECEIPT_CONSUMPTION_BLOCKED"))
    if review_present and review.get("normal_routing_allowed") is False:
        blocker_codes.append(str(review.get("status") or "GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKED"))
    return {
        "artifact_present": consumption_present or review_present,
        "normal_routing_allowed": normal_routing_allowed,
        "consumption_status": consumption.get("status") if consumption_present else None,
        "consumption_normal_routing_allowed": (
            consumption.get("normal_routing_allowed") if consumption_present else None
        ),
        "consumption_unresolved_issue_count": (
            consumption.get("unresolved_issue_count") if consumption_present else None
        ),
        "operator_review_status": review.get("status") if review_present else None,
        "operator_review_normal_routing_allowed": (
            review.get("normal_routing_allowed") if review_present else None
        ),
        "operator_review_unresolved_review_count": (
            review.get("unresolved_review_count") if review_present else None
        ),
        "blocker_codes": _unique(blocker_codes),
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
    repair = artifacts["trader_repair_orchestrator"]
    success_condition_issues = _success_condition_schema_issues(repair)
    work_order = repair.get("codex_work_order") if isinstance(repair.get("codex_work_order"), dict) else {}
    active_lane_action_mapping_required = bool(
        work_order.get("status") == REPAIR_ACTION_MAPPING_REQUIRED_STATUS
    )
    invalid_evidence_action_watermark = any(
        "invalid evidence action" in str(issue.get("issue") or "")
        or "invalid progress receipt" in str(issue.get("issue") or "")
        for issue in success_condition_issues
    )
    return {
        "required_field_missing": bool(missing_fields),
        "missing_fields": missing_fields,
        "success_condition_schema_ok": not success_condition_issues,
        "success_condition_schema_issues": success_condition_issues,
        "invalid_evidence_action_watermark": invalid_evidence_action_watermark,
        "active_lane_action_mapping_required": active_lane_action_mapping_required,
        "code_repair_required": bool(
            missing_fields
            or success_condition_issues
            or active_lane_action_mapping_required
        ),
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
    active_contract_state: dict[str, Any],
    repair_loop_state: dict[str, Any],
) -> tuple[str, str]:
    if repair_loop_state.get("action_mapping_required"):
        return (
            "CODE_REPAIR",
            "trader_repair_orchestrator found an active-lane action with no approved command "
            "mapping; repair and test that explicit mapping before redispatching the active-contract prompt.",
        )
    if repair_loop_state.get("actionable_repair_selected"):
        return (
            "CODE_REPAIR",
            "trader_repair_orchestrator selected actionable repair "
            f"{repair_loop_state.get('selected_request_code')}; preserve its objective, files, tests, "
            "and verification contract ahead of active-lane or active-contract evidence work.",
        )
    if schema_state.get("invalid_evidence_action_watermark"):
        return (
            "CODE_REPAIR",
            "trader_repair_orchestrator evidence progress failed canonical origin/hash/evaluation "
            "validation; repair that schema boundary once instead of treating opaque SHA churn "
            "as evidence progress.",
        )
    if repair_loop_state.get("active_lane_execution_pending"):
        return (
            "ACTIVE_LANE_EVIDENCE_DISPATCH",
            "trader_repair_orchestrator has one stable active-lane evidence dispatch pending "
            "completion acknowledgement; preserve that dispatch id and command plan instead of "
            "replacing it with another active-contract prompt.",
        )
    if repair_loop_state.get("material_evidence_wait"):
        return (
            "NO_ACTION_WAIT",
            "trader_repair_orchestrator suppressed an unchanged active-lane evidence dispatch; "
            "wait until its material digest or current action stage changes instead of redispatching "
            "the same active_trader_contract prompt.",
        )
    if (
        repair_loop_state.get("waiting_for_evidence")
        and active_contract_state.get("active_prompt_available")
        and _active_contract_supersedes_repair_waiting(active_contract_state, repair_loop_state)
    ):
        return (
            "ACTIVE_TRADER_CONTRACT_EVIDENCE",
            "active_trader_contract was refreshed after trader_repair_orchestrator's waiting-evidence state and carries the concrete lane-specific next_prompt; dispatch that work instead of looping on already-satisfied generic artifact refresh.",
        )
    if repair_loop_state.get("waiting_for_evidence"):
        if artifact_health.get("has_stale_or_contradicted_artifact"):
            return (
                "READ_ONLY_EVIDENCE_REFRESH",
                "trader_repair_orchestrator is waiting for evidence and current artifacts are stale or contradicted; run one read-only evidence refresh before waiting for changed market/proof inputs.",
            )
        return (
            "NO_ACTION_WAIT",
            "trader_repair_orchestrator reports ORCHESTRATOR_BLOCKED with no actionable Codex repair and waiting evidence actions, and artifact health is already clear; do not rerun the same read-only refresh until market/proof inputs, guardian triggers, or blocker state change.",
        )
    if active_contract_state.get("active_prompt_available"):
        return (
            "ACTIVE_TRADER_CONTRACT_EVIDENCE",
            "active_trader_contract has the freshest concrete active path and next_prompt; dispatch that lane-specific read-only evidence work before generic payoff refresh.",
        )
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
    if (
        scout_state.get("status") == "SCOUT_BLOCKED_OPERATOR_REVIEW"
        and operator_review_state.get("operator_review_required")
    ):
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


def _active_contract_supersedes_repair_waiting(
    active_contract_state: dict[str, Any],
    repair_loop_state: dict[str, Any],
) -> bool:
    active_generated = _parse_dt(active_contract_state.get("generated_at_utc"))
    repair_generated = _parse_dt(repair_loop_state.get("generated_at_utc"))
    if active_generated is None or repair_generated is None:
        return False
    return active_generated > repair_generated


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
    active_contract_state: dict[str, Any],
    repair_loop_state: dict[str, Any],
) -> str:
    if selected_next_work_type == "ACTIVE_LANE_EVIDENCE_DISPATCH":
        return (
            "ACTIVE_LANE_EVIDENCE_DISPATCH:"
            f"{repair_loop_state.get('active_lane_pending_dispatch_id')}"
        )
    if selected_next_work_type == "CODE_REPAIR" and repair_loop_state.get(
        "action_mapping_required"
    ):
        return (
            "ACTIVE_LANE_ACTION_MAPPING_REQUIRED:"
            f"{repair_loop_state.get('active_lane_action_code')}"
        )
    if selected_next_work_type == "CODE_REPAIR" and repair_loop_state.get(
        "actionable_repair_selected"
    ):
        return f"TRADER_REPAIR_REQUEST:{repair_loop_state.get('selected_request_code')}"
    if selected_next_work_type == "NO_ACTION_WAIT" and repair_loop_state.get(
        "material_evidence_wait"
    ):
        return (
            "REPAIR_MATERIAL_EVIDENCE_WAIT:"
            f"{repair_loop_state.get('active_lane_action_code')}:"
            f"{repair_loop_state.get('active_lane_material_digest')}"
        )
    if selected_next_work_type == "READ_ONLY_EVIDENCE_REFRESH" and repair_loop_state.get("waiting_for_evidence"):
        ids = ",".join(_string_list(repair_loop_state.get("next_evidence_action_ids")))
        return f"REPAIR_ORCHESTRATOR_WAITING_FOR_EVIDENCE:{ids or 'NO_ACTION_IDS'}"
    if selected_next_work_type == "ACTIVE_TRADER_CONTRACT_EVIDENCE":
        return f"ACTIVE_CONTRACT:{active_contract_state.get('selected_active_path')}:{active_contract_state.get('top_lane_id')}:{active_contract_state.get('frontier_lane_id')}"
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
    material_state: dict[str, Any],
) -> dict[str, Any]:
    previous = _read_previous_output(output_path)
    artifact_hash = artifact_index["combined_sha256"]
    material_state_sha256 = _stable_json_sha256(material_state)
    current = {
        "work_type": selected_next_work_type,
        "artifact_hash": artifact_hash,
        "key_blocker": key_blocker,
        "latest_input_generated_at_utc": artifact_index.get("latest_input_generated_at_utc"),
        "material_state_sha256": material_state_sha256,
    }
    previous_guard = (
        previous.get("repeat_loop_guard") if isinstance(previous.get("repeat_loop_guard"), dict) else {}
    )
    previous_fingerprint = (
        previous_guard.get("current_fingerprint")
        if isinstance(previous_guard.get("current_fingerprint"), dict)
        else {}
    )
    previous_history = _string_list(previous_guard.get("material_history_sha256"))
    previous_material_sha = previous_fingerprint.get("material_state_sha256")
    if (
        isinstance(previous_material_sha, str)
        and len(previous_material_sha) == 64
        and previous_material_sha not in previous_history
    ):
        previous_history.append(previous_material_sha)
    previous_history = previous_history[-REPEAT_MATERIAL_HISTORY_LIMIT:]
    material_history_hit = material_state_sha256 in previous_history
    same_material_as_previous = bool(previous_fingerprint) and (
        previous_material_sha == material_state_sha256
    )
    same_work_type_material_and_blocker = bool(same_material_as_previous) and (
        previous_fingerprint.get("work_type") == selected_next_work_type
        and previous_fingerprint.get("key_blocker") == key_blocker
    )
    same_raw_artifact_hash = bool(previous_fingerprint) and (
        previous_fingerprint.get("artifact_hash") == artifact_hash
    )
    explicit_wait = selected_next_work_type == "NO_ACTION_WAIT"
    repeat_allowed = not material_history_hit and not explicit_wait
    material_history = [*previous_history]
    if material_state_sha256 not in material_history:
        material_history.append(material_state_sha256)
    material_history = material_history[-REPEAT_MATERIAL_HISTORY_LIMIT:]
    return {
        "policy": (
            "Do not redispatch any canonical material state already present in bounded history, "
            "even when timestamp-only churn changes the classified work type or key blocker. "
            "generated_at, fetched_at, path, age, and raw artifact hash churn never authorize "
            "another dispatch."
        ),
        "current_fingerprint": current,
        "previous_fingerprint": previous_fingerprint or None,
        "material_history_sha256": material_history,
        "material_history_limit": REPEAT_MATERIAL_HISTORY_LIMIT,
        "material_history_hit": material_history_hit,
        "same_material_as_previous": same_material_as_previous,
        "same_work_type_artifact_and_blocker": same_work_type_material_and_blocker,
        "same_work_type_material_and_blocker": same_work_type_material_and_blocker,
        "same_raw_artifact_hash": same_raw_artifact_hash,
        "repeat_allowed": repeat_allowed,
        "repeat_reason": (
            "explicit_no_action_wait"
            if explicit_wait
            else (
                "material_state_seen_in_bounded_history"
                if material_history_hit
                else (
                    "no_previous_material_fingerprint"
                    if not previous_history
                    else "new_material_state"
                )
            )
        ),
    }


def _repeat_material_state(
    *,
    proof_state: dict[str, Any],
    payoff_state: dict[str, Any],
    harvest_state: dict[str, Any],
    scout_state: dict[str, Any],
    operator_review_state: dict[str, Any],
    edge_improvement_state: dict[str, Any],
    artifact_health: dict[str, Any],
    schema_state: dict[str, Any],
    active_contract_state: dict[str, Any],
    repair_loop_state: dict[str, Any],
    live_order_state: dict[str, Any],
) -> dict[str, Any]:
    """Return only decision-relevant state; volatile refresh metadata is excluded."""

    return {
        "proof": {
            key: proof_state.get(key)
            for key in (
                "as_proof_pack_queue_artifact_status",
                "as_lane_candidate_board_artifact_status",
                "portfolio_4x_path_planner_artifact_status",
                "proof_queue_count",
                "proof_ready_count",
                "can_create_live_permission_count",
                "rejected_proof_candidate_count",
                "proof_queue_empty",
                "as_live_ready_path_exists",
                "normal_routing_status",
                "routing_allowed",
                "primary_blocker",
                "portfolio_status",
                "portfolio_can_reach_4x_now",
                "portfolio_can_create_live_permission",
            )
        }
        | {
            "global_blockers": sorted(_string_list(proof_state.get("global_blockers"))),
            "p0_rows": sorted(_string_list(proof_state.get("p0_rows"))),
        },
        "payoff": {
            key: payoff_state.get(key)
            for key in (
                "artifact_status",
                "status",
                "verdict",
                "capture_economics_status",
                "live_promotion_allowed",
                "runner_candidate_count",
                "harvest_candidate_count",
                "negative_expectancy_visible",
                "month_scale_negative_visible",
            )
        }
        | {
            "live_promotion_blockers": sorted(
                _string_list(payoff_state.get("live_promotion_blockers"))
            ),
        },
        "harvest": {
            key: harvest_state.get(key)
            for key in (
                "artifact_status",
                "status",
                "closest_harvest_candidate_id",
                "closest_harvest_selected",
                "live_promotion_allowed",
                "replay_status",
            )
        }
        | {
            "promotion_blockers": sorted(
                _string_list(harvest_state.get("promotion_blockers"))
            ),
            "tp_proof": harvest_state.get("tp_proof") or {},
        },
        "scout": {
            key: scout_state.get(key)
            for key in (
                "artifact_status",
                "diagnosed",
                "status",
                "target_shape",
                "scout_mode_allowed",
                "operator_approval_required",
                "max_loss_jpy_cap",
            )
        }
        | {
            "blocker_codes": sorted(
                _string_list(scout_state.get("blocker_codes"))
            ),
            "min_lot_feasibility": {
                key: (scout_state.get("min_lot_feasibility") or {}).get(key)
                for key in (
                    "status",
                    "min_production_lot_units",
                    "units_mode",
                    "loss_budget_min_lot_blocker_present",
                    "feasibility_proven_current",
                    "feasible_if_all_non_lot_gates_clear",
                    "no_4x_deficit_lot_backsolve",
                )
            },
        },
        "operator_review": {
            key: operator_review_state.get(key)
            for key in (
                "normal_routing_allowed",
                "inferred_normal_routing_allowed",
                "normal_routing_status",
                "operator_review_required",
                "guardian_clear",
                "guardian_expired_or_operator_review_stale",
                "missing_raw_guardian_receipt_artifact",
            )
        }
        | {
            "blocker_codes": sorted(
                _string_list(operator_review_state.get("blocker_codes"))
            ),
            "raw_guardian_receipt_state": operator_review_state.get(
                "raw_guardian_receipt_state"
            )
            or {},
        },
        "edge_improvement": {
            key: edge_improvement_state.get(key)
            for key in (
                "candidate_available",
                "candidate_work_type",
                "target_shape",
                "proof_gap_trades",
                "take_profit_trades",
                "take_profit_losses",
                "take_profit_expectancy_jpy",
                "min_lot_numerically_feasible",
                "proof_queue_count",
                "live_promotion_allowed",
            )
        }
        | {
            "blockers_to_preserve": sorted(
                _string_list(edge_improvement_state.get("blockers_to_preserve"))
            ),
        },
        "artifact_health": {
            "explicit_issues": sorted(
                (
                    str(row.get("artifact")),
                    str(row.get("issue")),
                )
                for row in artifact_health.get("explicit_issues") or []
                if isinstance(row, dict)
            ),
            "artifact_contradiction_codes": sorted(
                _string_list(artifact_health.get("artifact_contradiction_codes"))
            ),
        },
        "schema": {
            "required_field_missing": schema_state.get("required_field_missing"),
            "missing_fields": sorted(
                (
                    str(row.get("artifact")),
                    str(row.get("field")),
                )
                for row in schema_state.get("missing_fields") or []
                if isinstance(row, dict)
            ),
            "success_condition_schema_ok": schema_state.get(
                "success_condition_schema_ok"
            ),
            "success_condition_schema_issues": sorted(
                (
                    str(row.get("artifact")),
                    str(row.get("issue")),
                )
                for row in schema_state.get("success_condition_schema_issues") or []
                if isinstance(row, dict)
            ),
            "active_lane_action_mapping_required": schema_state.get(
                "active_lane_action_mapping_required"
            ),
            "code_repair_required": schema_state.get("code_repair_required"),
        },
        "active_contract": {
            key: active_contract_state.get(key)
            for key in (
                "artifact_status",
                "status",
                "selected_active_path",
                "target_shape",
                "top_lane_id",
                "top_lane_status",
                "top_lane_vehicle",
                "frontier_lane_id",
                "frontier_lane_vehicle",
                "next_trade_enabling_action",
                "live_permission_allowed",
            )
        }
        | {
            "remaining_blockers": sorted(
                _string_list(active_contract_state.get("remaining_blockers"))
            ),
        },
        "repair_loop": {
            key: repair_loop_state.get(key)
            for key in (
                "artifact_status",
                "status",
                "actionable_request_count",
                "approval_required_request_count",
                "waiting_request_count",
                "repair_request_count",
                "selected_request_code",
                "next_evidence_action_count",
                "waiting_for_evidence",
                "queued_evidence_wait",
                "material_evidence_wait",
                "action_mapping_required",
                "actionable_repair_selected",
                "active_lane_execution_pending",
                "codex_work_order_status",
                "codex_work_order_reason_code",
                "active_lane_action_code",
                "active_lane_material_digest",
                "active_lane_pending_dispatch_id",
                "next_evidence_action_material_sha256",
            )
        }
        | {
            "waiting_request_codes": sorted(
                _string_list(repair_loop_state.get("waiting_request_codes"))
            ),
            "next_evidence_action_ids": sorted(
                _string_list(repair_loop_state.get("next_evidence_action_ids"))
            ),
        },
        "live_order": live_order_state,
    }


def _live_order_repeat_state(artifact: dict[str, Any]) -> dict[str, Any]:
    """Normalize gateway transitions without refresh timestamps or prose churn."""

    request = (
        artifact.get("order_request")
        if isinstance(artifact.get("order_request"), dict)
        else {}
    )
    return {
        "artifact_status": artifact.get("_artifact_status"),
        "status": artifact.get("status"),
        "cycle_send_requested": artifact.get("cycle_send_requested"),
        "send_requested": artifact.get("send_requested"),
        "sent": artifact.get("sent"),
        "sent_count": artifact.get("sent_count"),
        "lane_id": artifact.get("lane_id"),
        "lane_ids": sorted(_string_list(artifact.get("lane_ids"))),
        "requested_units": artifact.get("requested_units"),
        "scaled_units": artifact.get("scaled_units"),
        "size_multiple": artifact.get("size_multiple"),
        "portfolio_position_cap": artifact.get("portfolio_position_cap"),
        "risk_issue_codes": _normalized_issue_codes(artifact.get("risk_issues")),
        "strategy_issue_codes": _normalized_issue_codes(
            artifact.get("strategy_issues")
        ),
        "order_request": {
            key: request.get(key)
            for key in (
                "status",
                "request_id",
                "order_request_id",
                "client_order_id",
                "decision_id",
                "lane_id",
                "send_requested",
                "sent",
            )
            if key in request
        },
    }


def _normalized_issue_codes(value: Any) -> list[str]:
    codes: list[str] = []
    for row in value if isinstance(value, list) else []:
        if isinstance(row, str):
            codes.append(row)
            continue
        if not isinstance(row, dict):
            continue
        code = _first_str(
            row.get("code"),
            row.get("reason_code"),
            row.get("status"),
        )
        if code:
            codes.append(code)
    return sorted(_unique(codes))


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _work_dispatch_allowed(
    selected_next_work_type: str,
    *,
    repeat_loop_guard: dict[str, Any],
    repair_loop_state: dict[str, Any],
) -> bool:
    if selected_next_work_type == "NO_ACTION_WAIT":
        return False
    if selected_next_work_type == "ACTIVE_LANE_EVIDENCE_DISPATCH":
        # Exact pending dispatch ids have an acknowledgement lifecycle and must
        # remain available until the repair orchestrator records the ACK.
        return True
    if (
        selected_next_work_type == "CODE_REPAIR"
        and repair_loop_state.get("actionable_repair_selected") is True
        and bool(repair_loop_state.get("selected_request_code"))
    ):
        # A selected repair request is durable. Generic schema repair has no
        # request id/ack lifecycle and is therefore subject to repeat history.
        return True
    return bool(repeat_loop_guard.get("repeat_allowed"))


def _repeat_suppressed_prompt(
    *,
    classified_next_work_type: str,
    key_blocker: str,
    repeat_loop_guard: dict[str, Any],
) -> str:
    material_sha = (
        repeat_loop_guard.get("current_fingerprint", {}).get(
            "material_state_sha256"
        )
    )
    return f"""QuantRabbit の同一改善作業は再実行しないでください。

分類候補: `{classified_next_work_type}`
実行選択: `NO_ACTION_WAIT`
key blocker: `{key_blocker}`
canonical material state: `{material_sha}`

前回から意味のある状態変化がありません。generated_at、fetched_at、age、path、raw artifact hash の更新だけは進捗ではありません。コード修正、同じ証拠更新コマンド、同じ検証、発注、cancel、close、launchd変更を実行せず、machine-readable success state、material digest、lane/blocker、または broker/gateway permission state が変わるまで待ってください。"""


def _current_phase(
    *,
    selected_next_work_type: str,
    payoff_state: dict[str, Any],
    harvest_state: dict[str, Any],
    scout_state: dict[str, Any],
    operator_review_state: dict[str, Any],
    edge_improvement_state: dict[str, Any],
    active_contract_state: dict[str, Any],
    repair_loop_state: dict[str, Any],
) -> str:
    if selected_next_work_type == "ACTIVE_LANE_EVIDENCE_DISPATCH":
        return "ACTIVE_LANE_EVIDENCE_DISPATCH_PENDING_ACK"
    if selected_next_work_type == "ACTIVE_TRADER_CONTRACT_EVIDENCE":
        return f"ACTIVE_CONTRACT_{active_contract_state.get('selected_active_path') or 'EVIDENCE'}"
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
        if repair_loop_state.get("actionable_repair_selected"):
            return (
                "TRADER_REPAIR_"
                f"{repair_loop_state.get('selected_request_code') or 'SELECTED'}_REQUIRED"
            )
        return "GOAL_LOOP_SCHEMA_REPAIR_REQUIRED"
    if selected_next_work_type == "LIVE_PERMISSION_READY_CHECK":
        return "LIVE_PERMISSION_READY_CHECK_ONLY"
    if selected_next_work_type == "EDGE_IMPROVEMENT_EXPERIMENT":
        return "HARVEST_EDGE_IMPROVEMENT_EXPERIMENT"
    if selected_next_work_type == "NO_ACTION_WAIT" and repair_loop_state.get("waiting_for_evidence"):
        return "WAITING_FOR_EVIDENCE_OR_MARKET_TRIGGER"
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
                {"field": "requires_operator_approval_for_this_report", "operator": "eq", "value": False},
                {"field": "requires_operator_review_before_scout_or_routing", "operator": "eq", "value": True},
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
            "schema, success_condition, and active-lane action mappings are repaired.",
            [
                {"field": "repair_orchestrator_actionable_request_count", "operator": "eq", "value": 0},
                {"field": "required_field_missing", "operator": "eq", "value": False},
                {"field": "success_condition_schema_ok", "operator": "eq", "value": True},
                {"field": "active_lane_action_mapping_required", "operator": "eq", "value": False},
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
        "ACTIVE_TRADER_CONTRACT_EVIDENCE": (
            "all",
            "active_trader_contract exposes a concrete lane-specific read-only evidence action.",
            [
                {"field": "active_contract_prompt_available", "operator": "eq", "value": True},
                {"field": "active_contract_stale", "operator": "eq", "value": False},
                {"field": "active_contract_live_permission_allowed", "operator": "eq", "value": False},
                {"field": "live_permission_allowed", "operator": "eq", "value": False},
            ],
        ),
        "ACTIVE_LANE_EVIDENCE_DISPATCH": (
            "all",
            "the stable active-lane dispatch is acknowledged or superseded by material evidence.",
            [
                {"field": "active_lane_execution_pending", "operator": "eq", "value": False},
                {"field": "live_permission_allowed", "operator": "eq", "value": False},
            ],
        ),
        "NO_ACTION_WAIT": (
            "all",
            "no work is dispatchable and live permission remains false until canonical material changes.",
            [
                {"field": "work_dispatch_allowed", "operator": "eq", "value": False},
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
    active_contract_state: dict[str, Any],
    repair_loop_state: dict[str, Any],
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
        "requires_operator_approval_for_this_report": False,
        "requires_operator_review_before_scout_or_routing": bool(
            operator_review_state.get("operator_review_required")
        ),
        "edge_experiment_candidate": edge_improvement_state.get("candidate_available"),
        "has_stale_or_contradicted_artifact": artifact_health.get("has_stale_or_contradicted_artifact"),
        "required_field_missing": schema_state.get("required_field_missing"),
        "success_condition_schema_ok": schema_state.get("success_condition_schema_ok"),
        "active_lane_action_mapping_required": schema_state.get(
            "active_lane_action_mapping_required"
        ),
        "proof_queue_count": proof_state.get("proof_queue_count"),
        "can_create_live_permission_count": proof_state.get("can_create_live_permission_count"),
        "broker_truth_fresh": (live_ready_state.get("checks") or {}).get("broker_truth_fresh"),
        "active_contract_prompt_available": active_contract_state.get("active_prompt_available"),
        "active_contract_stale": active_contract_state.get("stale"),
        "active_contract_live_permission_allowed": active_contract_state.get("live_permission_allowed"),
        "repair_orchestrator_waiting_for_evidence": repair_loop_state.get("waiting_for_evidence"),
        "repair_orchestrator_actionable_request_count": repair_loop_state.get("actionable_request_count"),
        "repair_orchestrator_waiting_request_count": repair_loop_state.get("waiting_request_count"),
        "repair_orchestrator_next_evidence_action_count": repair_loop_state.get("next_evidence_action_count"),
        "active_lane_execution_pending": repair_loop_state.get(
            "active_lane_execution_pending"
        ),
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


def _next_allowed_commands(
    work_type: str,
    *,
    repair_loop_state: dict[str, Any] | None = None,
) -> list[str]:
    if work_type == "ACTIVE_LANE_EVIDENCE_DISPATCH":
        return _string_list((repair_loop_state or {}).get("active_lane_dispatch_commands"))
    if work_type == "CODE_REPAIR" and (repair_loop_state or {}).get(
        "actionable_repair_selected"
    ):
        selected_commands = _unique(
            _string_list((repair_loop_state or {}).get("codex_work_order_targeted_test_commands"))
            + _string_list((repair_loop_state or {}).get("codex_work_order_verification_commands"))
            + _string_list((repair_loop_state or {}).get("codex_work_order_final_verification_commands"))
        )
        if selected_commands:
            return selected_commands
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
        "ACTIVE_TRADER_CONTRACT_EVIDENCE": [
            "python3 -m json.tool data/active_trader_contract.json >/dev/null",
            "python3 -m json.tool data/trader_goal_loop_orchestrator.json >/dev/null",
            "PYTHONPATH=src python3 -m unittest tests.test_trader_goal_loop_orchestrator tests.test_active_trader_contract -v",
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
    active_contract_state: dict[str, Any],
    repair_loop_state: dict[str, Any],
) -> str:
    input_artifacts = "\n".join(f"- data/{_artifact_filename(name)}" for name in INPUT_ARTIFACT_NAMES)
    commands = "\n".join(f"- `{command}`" for command in next_allowed_commands)
    success = json.dumps(success_condition, ensure_ascii=False, indent=2, sort_keys=True)
    if selected_next_work_type == "ACTIVE_LANE_EVIDENCE_DISPATCH":
        command_steps = json.dumps(
            repair_loop_state.get("active_lane_dispatch_command_steps") or [],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        return f"""QuantRabbit の active-lane evidence dispatch を完了してください。

dispatch id:
{repair_loop_state.get('active_lane_pending_dispatch_id')}

current action:
{repair_loop_state.get('active_lane_action_code')}

この同一 dispatch id を新しい作業として複製しないでください。以下の固定 read-only command plan を順番どおり実行し、末尾の exact-digest acknowledgement が成功した時だけ完了です。途中失敗時はackせず、同じpending dispatchを保持してください。

commands:
{commands or '- (missing command plan; stop and repair the contract)'}

machine-readable command steps (`ok_rcs` は成功扱いする終了コード):
```json
{command_steps}
```

success_condition:
```json
{success}
```

発注、cancel、close、launchd変更、gate緩和、live permission生成は禁止です。"""
    if selected_next_work_type == "CODE_REPAIR" and repair_loop_state.get(
        "actionable_repair_selected"
    ):
        suggested_files = "\n".join(
            f"- `{path}`"
            for path in _string_list(
                repair_loop_state.get("codex_work_order_suggested_files")
            )
        )
        required_tests = "\n".join(
            f"- {test}"
            for test in _string_list(
                repair_loop_state.get("codex_work_order_required_tests")
            )
        )
        return f"""QuantRabbit の選択済み修理案件を実施してください。

selected repair request:
{repair_loop_state.get('selected_request_code')}

objective:
{repair_loop_state.get('codex_work_order_objective') or '(missing; inspect trader_repair_orchestrator.json and fail closed)'}

automation contract:
{repair_loop_state.get('codex_work_order_automation_prompt') or '(missing)'}

suggested files:
{suggested_files or '- (none provided)'}

required tests:
{required_tests or '- (none provided)'}

この修理案件は terminal trader-support-bot → trader-repair-orchestrator で選ばれた現在の最優先作業です。active_trader_contract の通常evidence promptへ置き換えず、修理後に同じartifactを再評価してください。

success_condition:
```json
{success}
```

verification commands:
{commands or '- (none provided; inspect the selected work order)'}

発注、cancel、close、launchd変更、gate緩和、live permission生成は禁止です。"""
    if selected_next_work_type == "ACTIVE_TRADER_CONTRACT_EVIDENCE" and active_contract_state.get("next_prompt"):
        active_prompt = str(active_contract_state.get("next_prompt"))
        action = str(active_contract_state.get("next_trade_enabling_action") or "")
        blockers = ", ".join(_string_list(active_contract_state.get("remaining_blockers"))[:12])
        return f"""QuantRabbit の 4x改善ループで `ACTIVE_TRADER_CONTRACT_EVIDENCE` を実施してください。

目的:
active_trader_contract が選んだ現在の lane-specific evidence work を read-only で進める。これは live permission ではありません。発注許可、cancel/close 許可、launchd 操作許可を作らず、改善ループの次アクションだけを扱ってください。

active_trader_contract next_prompt:
{active_prompt}

active_trader_contract next_trade_enabling_action:
{action}

保持する blocker:
{blockers}

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
- active contract の board lane と frontier evidence lane を同じ unblock plan として扱い、どちらか片方だけに戻さない

success condition:
```json
{success}
```

検証コマンド:
{commands}

注意:
この作業は live permission ready check であっても発注許可ではありません。最終送信は既存 gateway、fresh broker truth、risk/gpt verifier、guardian/operator review clearance、明示 live flag が別途そろうまで禁止です。
"""
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


def _active_contract_text(active_contract_state: dict[str, Any] | None, key: str) -> str | None:
    state = active_contract_state if isinstance(active_contract_state, dict) else {}
    if not state.get("active_prompt_available"):
        return None
    value = state.get(key)
    return str(value).strip() if isinstance(value, str) and value.strip() else None


def _four_x_progress_hypothesis(
    edge_state: dict[str, Any],
    scout_state: dict[str, Any],
    *,
    active_contract_state: dict[str, Any] | None = None,
) -> str:
    active_text = _active_contract_text(active_contract_state, "four_x_progress_hypothesis")
    if active_text:
        return active_text
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


def _root_improvement_target(
    edge_state: dict[str, Any],
    scout_state: dict[str, Any],
    *,
    active_contract_state: dict[str, Any] | None = None,
) -> str:
    active_text = _active_contract_text(active_contract_state, "root_improvement_target")
    if active_text:
        return active_text
    target = edge_state.get("target_shape") or scout_state.get("target_shape") or "HARVEST candidate"
    return (
        f"{target} を、発注許可ではなく read-only の SCOUT 判断材料と期待値改善実験で live-grade HARVEST 候補へ近づける。"
    )


def _expected_edge_improvement(
    edge_state: dict[str, Any],
    scout_state: dict[str, Any],
    *,
    active_contract_state: dict[str, Any] | None = None,
) -> str:
    active_text = _active_contract_text(active_contract_state, "expected_edge_improvement")
    if active_text:
        return active_text
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
    active_contract = payload.get("active_contract_state", {})
    repair_loop = payload.get("repair_loop_state", {})
    review_required = bool(payload.get("requires_operator_review_before_scout_or_routing"))
    approval_boundary_note = (
        "- このreport生成自体は承認不要。ただしSCOUT/normal routing前にはoperator review必須。"
        if review_required
        else "- このreport生成自体は承認不要。現在のartifact状態ではSCOUT/normal routing前のoperator review必須状態ではありません。"
    )
    lines = [
        "# Trader Goal Loop Orchestrator",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Read only: `{payload.get('read_only')}`",
        f"- Live side effects: `{payload.get('live_side_effects')}`",
        f"- Live permission allowed: `{payload.get('live_permission_allowed')}`",
        f"- requires_operator_approval_for_this_report: `{payload.get('requires_operator_approval_for_this_report')}`",
        f"- requires_operator_review_before_scout_or_routing: `{payload.get('requires_operator_review_before_scout_or_routing')}`",
        f"- Current phase: `{payload.get('current_phase')}`",
        f"- Selected next work type: `{payload.get('selected_next_work_type')}`",
        f"- Classified next work type: `{payload.get('classified_next_work_type')}`",
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
        f"- Active contract selected path: `{active_contract.get('selected_active_path')}`",
        f"- Active contract top lane: `{active_contract.get('top_lane_id')}`",
        f"- Active contract frontier lane: `{active_contract.get('frontier_lane_id')}`",
        f"- Active contract prompt available: `{active_contract.get('active_prompt_available')}`",
        f"- Repair orchestrator status: `{repair_loop.get('status')}`",
        f"- Repair orchestrator actionable/waiting: `{repair_loop.get('actionable_request_count')}` / `{repair_loop.get('waiting_request_count')}`",
        f"- Repair orchestrator waiting for evidence: `{repair_loop.get('waiting_for_evidence')}`",
        "",
        "## Approval Boundary",
        "",
        approval_boundary_note,
        "- `requires_operator_approval_for_this_report` is report-generation approval only.",
        "- `requires_operator_review_before_scout_or_routing` is the separate SCOUT/normal-routing gate.",
        "",
        "## Repeat Guard",
        "",
        f"- Repeat allowed: `{payload.get('repeat_loop_guard', {}).get('repeat_allowed')}`",
        f"- Work dispatch allowed: `{payload.get('work_dispatch_allowed')}`",
        f"- Repeat suppressed: `{payload.get('repeat_suppressed')}`",
        f"- Material history hit: `{payload.get('repeat_loop_guard', {}).get('material_history_hit')}`",
        f"- Key blocker: `{payload.get('repeat_loop_guard', {}).get('current_fingerprint', {}).get('key_blocker')}`",
        f"- Canonical material state: `{payload.get('repeat_loop_guard', {}).get('current_fingerprint', {}).get('material_state_sha256')}`",
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
    if not payload.get("next_allowed_commands"):
        lines.append("- (none; no new material state to dispatch)")
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
    actions: list[Any] = []
    top_actions = repair.get("next_evidence_actions")
    if isinstance(top_actions, list):
        actions.extend(top_actions)
    state_actions = _repair_current_state(repair).get("next_evidence_actions")
    if isinstance(state_actions, list):
        actions.extend(state_actions)
    progress_receipts = repair.get("evidence_action_progress")
    if not isinstance(progress_receipts, list):
        progress_receipts = []
    current_state = _repair_current_state(repair)
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
        watermark_fields_present = any(
            key in action
            for key in (
                "progress_watermark_contract",
                "progress_watermark_origin",
                "progress_watermark_source",
                "success_condition_evaluation",
            )
        )
        if watermark_fields_present and validated_evidence_action_material(
            action,
            current_state=current_state,
            source="next_action",
        ) is None:
            issues.append(
                {
                    "artifact": "trader_repair_orchestrator",
                    "issue": (
                        f"{action.get('action_id') or index}: invalid evidence action "
                        "watermark/origin/evaluation"
                    ),
                }
            )
    for index, receipt in enumerate(progress_receipts):
        if not isinstance(receipt, dict) or validated_evidence_action_material(
            receipt,
            current_state=current_state,
            source="progress_receipt",
        ) is None:
            issues.append(
                {
                    "artifact": "trader_repair_orchestrator",
                    "issue": (
                        f"evidence_action_progress[{index}]: invalid progress receipt"
                    ),
                }
            )
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
        "active_trader_contract": "active_trader_contract.json",
        "payoff_shape_diagnosis": "payoff_shape_diagnosis.json",
        "harvest_live_grade_path": "harvest_live_grade_path.json",
        "eurusd_short_breakout_failure_scout_plan": "eurusd_short_breakout_failure_scout_plan.json",
        "as_proof_pack_queue": "as_proof_pack_queue.json",
        "as_lane_candidate_board": "as_lane_candidate_board.json",
        "portfolio_4x_path_planner": "portfolio_4x_path_planner.json",
        "guardian_receipt_consumption": "guardian_receipt_consumption.json",
        "guardian_receipt_operator_review": "guardian_receipt_operator_review.json",
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

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_ACTIVE_TRADER_CONTRACT,
    DEFAULT_ACTIVE_TRADER_CONTRACT_REPORT,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
    DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR,
)


CONTRACT_VERSION = "active_trader_contract_v1"
CONTRACT_GOAL = "monthly_funding_adjusted_equity_4x"
TARGET_SHAPE = "EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST"
ALLOWED_ACTIVE_PATHS = {
    "HARVEST_READY_CHECK",
    "SCOUT_READY_CHECK",
    "EVIDENCE_ACQUISITION",
    "EDGE_IMPROVEMENT_EXPERIMENT",
    "OPERATOR_REVIEW_REPORT",
    "LIVE_PERMISSION_READY_CHECK",
    "NO_TRADE_WITH_CAUSE",
}

DEFAULT_HARVEST_LIVE_GRADE_PATH = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "harvest_live_grade_path.json"
DEFAULT_EURUSD_SCOUT_PLAN = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "eurusd_short_breakout_failure_scout_plan.json"
DEFAULT_AS_PROOF_PACK_QUEUE = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "as_proof_pack_queue.json"
DEFAULT_AS_LANE_CANDIDATE_BOARD = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "as_lane_candidate_board.json"
DEFAULT_PORTFOLIO_4X_PATH_PLANNER = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "portfolio_4x_path_planner.json"
DEFAULT_EURUSD_PROOF_FLOOR_UPDATE = (
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "eurusd_short_breakout_failure_proof_floor_update.json"
)
DEFAULT_EURUSD_LIMIT_S5_BIDASK_REPLAY = (
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "eurusd_short_breakout_failure_limit_s5_bidask_replay.json"
)


@dataclass(frozen=True)
class ActiveTraderContractSummary:
    status: str
    output_path: Path
    report_path: Path
    selected_active_path: str
    live_permission_allowed: bool


class ActiveTraderContract:
    """Write the read-only 4x active-path contract for the trader loop."""

    def __init__(
        self,
        *,
        trader_goal_loop_path: Path = DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR,
        payoff_shape_diagnosis_path: Path = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
        harvest_live_grade_path: Path = DEFAULT_HARVEST_LIVE_GRADE_PATH,
        scout_plan_path: Path = DEFAULT_EURUSD_SCOUT_PLAN,
        proof_pack_queue_path: Path = DEFAULT_AS_PROOF_PACK_QUEUE,
        lane_candidate_board_path: Path = DEFAULT_AS_LANE_CANDIDATE_BOARD,
        portfolio_4x_path_planner_path: Path = DEFAULT_PORTFOLIO_4X_PATH_PLANNER,
        live_order_request_path: Path = DEFAULT_LIVE_ORDER_REQUEST,
        broker_snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        daily_target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        proof_floor_update_path: Path = DEFAULT_EURUSD_PROOF_FLOOR_UPDATE,
        limit_s5_bidask_replay_path: Path = DEFAULT_EURUSD_LIMIT_S5_BIDASK_REPLAY,
        output_path: Path = DEFAULT_ACTIVE_TRADER_CONTRACT,
        report_path: Path = DEFAULT_ACTIVE_TRADER_CONTRACT_REPORT,
        now_utc: datetime | None = None,
    ) -> None:
        self.paths = {
            "trader_goal_loop_orchestrator": trader_goal_loop_path,
            "payoff_shape_diagnosis": payoff_shape_diagnosis_path,
            "harvest_live_grade_path": harvest_live_grade_path,
            "eurusd_short_breakout_failure_scout_plan": scout_plan_path,
            "as_proof_pack_queue": proof_pack_queue_path,
            "as_lane_candidate_board": lane_candidate_board_path,
            "portfolio_4x_path_planner": portfolio_4x_path_planner_path,
            "live_order_request": live_order_request_path,
            "broker_snapshot": broker_snapshot_path,
            "daily_target_state": daily_target_state_path,
            "eurusd_short_breakout_failure_proof_floor_update": proof_floor_update_path,
            "eurusd_short_breakout_failure_limit_s5_bidask_replay": limit_s5_bidask_replay_path,
        }
        self.output_path = output_path
        self.report_path = report_path
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> ActiveTraderContractSummary:
        payload = self.build_payload()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload), encoding="utf-8")
        return ActiveTraderContractSummary(
            status=str(payload["status"]),
            output_path=self.output_path,
            report_path=self.report_path,
            selected_active_path=str(payload["selected_active_path"]),
            live_permission_allowed=bool(payload["live_permission_allowed"]),
        )

    def build_payload(self) -> dict[str, Any]:
        artifacts = {name: _load_artifact(path) for name, path in self.paths.items()}
        artifact_index = _artifact_index(artifacts)
        harvest = _harvest_contract_state(artifacts["harvest_live_grade_path"])
        scout = _scout_contract_state(artifacts["eurusd_short_breakout_failure_scout_plan"])
        proof = _proof_contract_state(artifacts["as_proof_pack_queue"])
        board = _board_contract_state(artifacts["as_lane_candidate_board"])
        portfolio = _portfolio_contract_state(artifacts["portfolio_4x_path_planner"])
        live_order = _live_order_contract_state(artifacts["live_order_request"])
        goal_loop = _goal_loop_contract_state(artifacts["trader_goal_loop_orchestrator"])
        proof_floor = _proof_floor_contract_state(
            artifacts["eurusd_short_breakout_failure_proof_floor_update"]
        )
        replay = _limit_replay_contract_state(
            artifacts["eurusd_short_breakout_failure_limit_s5_bidask_replay"],
            proof=proof,
        )
        scout = _normalize_stale_blocker_codes(scout, proof=proof, proof_floor=proof_floor, replay=replay)
        proof_floor = _normalize_stale_blocker_codes(
            proof_floor,
            proof=proof,
            proof_floor=proof_floor,
            replay=replay,
        )
        active_deployment_gap = _active_deployment_gap(
            harvest=harvest,
            scout=scout,
            proof=proof,
            board=board,
            portfolio=portfolio,
            replay=replay,
        )
        no_action = _no_action_contract(
            harvest=harvest,
            scout=scout,
            proof=proof,
            board=board,
            portfolio=portfolio,
            replay=replay,
            active_deployment_gap=active_deployment_gap,
        )
        selected_active_path, selection_reason = _select_active_path(
            no_action=no_action,
            replay=replay,
            proof_floor=proof_floor,
            scout=scout,
            goal_loop=goal_loop,
            harvest=harvest,
            proof=proof,
        )
        remaining_blockers = _remaining_blockers(
            harvest=harvest,
            scout=scout,
            proof=proof,
            board=board,
            portfolio=portfolio,
            live_order=live_order,
            replay=replay,
            proof_floor=proof_floor,
        )
        payload = {
            "contract_version": CONTRACT_VERSION,
            "status": _status(selected_active_path, replay, remaining_blockers),
            "generated_at_utc": self.now_utc.isoformat(),
            "read_only": True,
            "live_side_effects": [],
            "live_permission_allowed": False,
            "contract_goal": CONTRACT_GOAL,
            "active_path_required": True,
            "selected_active_path": selected_active_path,
            "selected_active_path_reason": selection_reason,
            "allowed_active_paths": sorted(ALLOWED_ACTIVE_PATHS),
            "target_shape": TARGET_SHAPE,
            "four_x_progress_hypothesis": _four_x_progress_hypothesis(replay, proof_floor, harvest),
            "root_improvement_target": _root_improvement_target(replay),
            "expected_edge_improvement": _expected_edge_improvement(replay, proof_floor, harvest),
            "no_action_allowed": bool(no_action["no_action_allowed"]),
            "no_action_contract": no_action,
            "active_deployment_gap": active_deployment_gap,
            "next_trade_enabling_action": _next_trade_enabling_action(selected_active_path, replay),
            "remaining_blockers": remaining_blockers,
            "current_state": {
                "harvest": harvest,
                "scout": scout,
                "proof": proof,
                "board": board,
                "portfolio": portfolio,
                "live_order": live_order,
                "goal_loop": goal_loop,
                "proof_floor": proof_floor,
                "limit_s5_bidask_replay": replay,
            },
            "safety_contract": _safety_contract(),
            "next_prompt": _next_prompt(selected_active_path, remaining_blockers),
            "artifact_index": artifact_index,
        }
        if selected_active_path not in ALLOWED_ACTIVE_PATHS:
            raise ValueError(f"unknown selected_active_path: {selected_active_path}")
        if payload["live_permission_allowed"]:
            raise ValueError("active trader contract must never grant live permission")
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
    for name, artifact in artifacts.items():
        entries[name] = {
            "path": artifact.get("_path"),
            "status": artifact.get("_artifact_status"),
            "sha256": artifact.get("_sha256"),
            "generated_at_utc": artifact.get("generated_at_utc") or artifact.get("fetched_at_utc"),
        }
    combined = hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"artifacts": entries, "combined_sha256": combined}


def _harvest_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "candidate_present": False,
            "live_promotion_allowed": False,
            "promotion_blockers": ["HARVEST_ARTIFACT_MISSING"],
        }
    closest = artifact.get("closest_harvest_candidate")
    closest = closest if isinstance(closest, dict) else {}
    current_intent = closest.get("current_intent_best")
    current_intent = current_intent if isinstance(current_intent, dict) else {}
    tp_proof = closest.get("tp_proof")
    tp_proof = tp_proof if isinstance(tp_proof, dict) else {}
    return {
        "artifact_status": "present",
        "status": artifact.get("status"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "candidate_present": bool(closest),
        "candidate_id": closest.get("candidate_id") or closest.get("shape_key"),
        "target_shape_without_vehicle": closest.get("shape_key"),
        "actual_proof_queue_member": bool(closest.get("actual_proof_queue_member")),
        "planner_can_enter_proof_pack": bool(closest.get("planner_can_enter_proof_pack")),
        "can_create_live_permission": bool(closest.get("can_create_live_permission")),
        "live_promotion_allowed": bool(artifact.get("live_promotion_allowed"))
        and bool(closest.get("live_promotion_allowed")),
        "promotion_blockers": _unique(
            _string_list(closest.get("promotion_blockers")) + _codes_from_blockers(artifact.get("promotion_blockers"))
        ),
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
            "risk_allowed": bool(current_intent.get("risk_allowed")),
            "risk_jpy": current_intent.get("risk_jpy"),
            "units": current_intent.get("units"),
            "live_blocker_codes": _string_list(current_intent.get("live_blocker_codes")),
            "tp_execution_mode": current_intent.get("tp_execution_mode"),
            "tp_target_intent": current_intent.get("tp_target_intent"),
        },
    }


def _scout_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "scout_candidate_present": False,
            "scout_mode_allowed": False,
            "blocker_codes": ["SCOUT_PLAN_MISSING"],
        }
    blockers = _codes_from_blockers(artifact.get("proof_queue_entry_blockers"))
    min_lot = artifact.get("min_lot_feasibility")
    min_lot = min_lot if isinstance(min_lot, dict) else {}
    return {
        "artifact_status": "present",
        "status": artifact.get("status"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "target_shape": artifact.get("target_shape"),
        "scout_candidate_present": bool(artifact.get("target_shape")),
        "scout_mode_allowed": bool(artifact.get("scout_mode_allowed")),
        "operator_approval_required": bool(artifact.get("operator_approval_required")),
        "max_loss_jpy_cap": artifact.get("max_loss_jpy_cap"),
        "min_lot_feasible": bool(min_lot.get("feasible_if_all_non_lot_gates_clear")),
        "min_lot_status": min_lot.get("status"),
        "blocker_codes": blockers,
        "evidence_success_conditions": artifact.get("evidence_success_conditions") or [],
    }


def _proof_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "proof_queue_count": 0,
            "proof_ready_count": 0,
            "can_create_live_permission_count": 0,
            "proof_queue_count_is_live_permission": False,
        }
    summary = artifact.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    queue = artifact.get("queue") if isinstance(artifact.get("queue"), list) else []
    return {
        "artifact_status": "present",
        "generated_at_utc": artifact.get("generated_at_utc"),
        "proof_queue_count": _first_int(summary.get("queue_count"), len(queue)),
        "proof_ready_count": _first_int(summary.get("proof_ready_count"), 0),
        "can_create_live_permission_count": _first_int(summary.get("can_create_live_permission_count"), 0),
        "rejected_candidate_count": _first_int(summary.get("rejected_candidate_count"), 0),
        "proof_queue_empty": _first_int(summary.get("queue_count"), len(queue)) == 0,
        "proof_queue_count_is_live_permission": False,
    }


def _board_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {"artifact_status": "missing", "routing_allowed": False, "blocker_codes": ["AS_BOARD_MISSING"]}
    blocker = artifact.get("exact_blocker_preventing_live_ready")
    blocker = blocker if isinstance(blocker, dict) else {}
    return {
        "artifact_status": "present",
        "generated_at_utc": artifact.get("generated_at_utc"),
        "normal_routing_status": artifact.get("normal_routing_status"),
        "routing_allowed": bool(artifact.get("routing_allowed")),
        "as_live_ready_path_exists": bool(artifact.get("as_live_ready_path_exists")),
        "primary_blocker": blocker.get("primary"),
        "blocker_codes": _unique(
            _string_list(blocker.get("global_blockers"))
            + _string_list(blocker.get("p0_rows"))
            + ([str(blocker.get("primary"))] if blocker.get("primary") else [])
        ),
    }


def _portfolio_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "portfolio_status": "MISSING",
            "can_reach_4x_now": False,
            "can_create_live_permission": False,
        }
    summary = artifact.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    return {
        "artifact_status": "present",
        "generated_at_utc": artifact.get("generated_at_utc"),
        "portfolio_status": artifact.get("portfolio_status"),
        "can_reach_4x_now": bool(artifact.get("can_reach_4x_now")),
        "can_create_live_permission": bool(summary.get("can_create_live_permission")),
        "live_ready_lanes": _first_int(summary.get("live_ready_lanes"), 0),
    }


def _live_order_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {"artifact_status": "missing", "status": "MISSING", "send_requested": False, "sent": False}
    return {
        "artifact_status": "present",
        "generated_at_utc": artifact.get("generated_at_utc"),
        "status": artifact.get("status"),
        "send_requested": bool(artifact.get("send_requested")),
        "sent": bool(artifact.get("sent")),
    }


def _goal_loop_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {"artifact_status": "missing", "selected_next_work_type": None}
    return {
        "artifact_status": "present",
        "generated_at_utc": artifact.get("generated_at_utc"),
        "status": artifact.get("status"),
        "selected_next_work_type": artifact.get("selected_next_work_type"),
        "four_x_progress_hypothesis": artifact.get("four_x_progress_hypothesis"),
        "root_improvement_target": artifact.get("root_improvement_target"),
        "expected_edge_improvement": artifact.get("expected_edge_improvement"),
        "live_permission_allowed": bool(artifact.get("live_permission_allowed")),
    }


def _proof_floor_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {"artifact_status": "missing", "proof_floor_reached": False, "blocker_codes": ["PROOF_FLOOR_UPDATE_MISSING"]}
    post = artifact.get("post_update_tp_proof")
    post = post if isinstance(post, dict) else {}
    pre = artifact.get("pre_update_tp_proof")
    pre = pre if isinstance(pre, dict) else {}
    return {
        "artifact_status": "present",
        "status": artifact.get("status"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "target_shape": artifact.get("target_shape"),
        "proof_floor_reached": bool(post.get("proof_floor_reached")),
        "wins": _first_int(post.get("wins"), pre.get("wins"), 0),
        "losses": _first_int(post.get("losses"), pre.get("losses"), 0),
        "proof_floor": _first_int(post.get("proof_floor"), pre.get("proof_floor"), 20),
        "remaining_samples": _first_int(post.get("remaining_samples"), pre.get("remaining_samples"), 0),
        "canonical_integration_status": artifact.get("canonical_integration_status"),
        "scout_status_after_update": artifact.get("scout_status_after_update"),
        "harvest_live_grade_status_after_update": artifact.get("harvest_live_grade_status_after_update"),
        "blocker_codes": _codes_from_blockers(artifact.get("remaining_blockers")),
    }


def _limit_replay_contract_state(artifact: dict[str, Any], *, proof: dict[str, Any] | None = None) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "status": "MISSING",
            "passed": False,
            "live_grade_candidate": False,
            "blocker_codes": ["EXACT_LIMIT_S5_BIDASK_REPLAY_MISSING"],
        }
    blockers = _codes_from_blockers(artifact.get("remaining_blockers")) + _codes_from_blockers(
        artifact.get("proof_queue_blockers_if_positive")
    )
    if proof and proof.get("proof_queue_count", 0) > 0:
        blockers = [
            code
            for code in blockers
            if code not in {"NOT_IN_PROOF_QUEUE", "PROOF_QUEUE_EMPTY_NO_LIVE_PERMISSION"}
        ]
    return {
        "artifact_status": "present",
        "status": artifact.get("status"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "target_shape": artifact.get("target_shape"),
        "passed": bool(artifact.get("s5_bidask_replay_status")) and int(artifact.get("replay_losses") or 0) == 0,
        "s5_bidask_replay_status": artifact.get("s5_bidask_replay_status"),
        "replay_sample_count": _first_int(artifact.get("replay_sample_count"), 0),
        "replay_wins": _first_int(artifact.get("replay_wins"), 0),
        "replay_losses": _first_int(artifact.get("replay_losses"), 0),
        "net_expectancy_after_bidask": artifact.get("net_expectancy_after_bidask"),
        "live_grade_candidate": bool(artifact.get("live_grade_candidate")),
        "market_stop_samples_excluded": bool(artifact.get("market_stop_samples_excluded")),
        "market_close_excluded": bool(artifact.get("market_close_excluded")),
        "next_read_only_actions": _string_list(artifact.get("next_read_only_actions")),
        "blocker_codes": _unique(blockers),
    }


def _active_deployment_gap(
    *,
    harvest: dict[str, Any],
    scout: dict[str, Any],
    proof: dict[str, Any],
    board: dict[str, Any],
    portfolio: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    triggers: list[str] = []
    if harvest.get("candidate_present"):
        triggers.append("HARVEST_CANDIDATE_PRESENT")
    if scout.get("scout_candidate_present"):
        triggers.append("SCOUT_CANDIDATE_PRESENT")
    if scout.get("max_loss_jpy_cap") is not None:
        triggers.append("MAX_LOSS_CAP_DEFINED")
    if scout.get("min_lot_feasible"):
        triggers.append("MIN_LOT_FEASIBLE")
    if proof.get("proof_queue_count", 0) > 0:
        triggers.append("PROOF_QUEUE_HAS_CANDIDATES")
    if replay.get("artifact_status") == "missing" or replay.get("passed") or not replay.get("live_grade_candidate"):
        triggers.append("REPLAY_OR_PROOF_ACTION_AVAILABLE")
    if board.get("as_live_ready_path_exists") or portfolio.get("can_reach_4x_now"):
        triggers.append("PORTFOLIO_PATH_REVIEW_AVAILABLE")
    status = "ACTIVE_PATH_REQUIRED" if triggers else "NO_ACTIVE_GAP_INPUTS_VISIBLE"
    return {
        "status": status,
        "active_path_triggers": triggers,
        "would_no_action_violate_contract": bool(triggers),
        "live_deployment_allowed": False,
        "reason": (
            "read-only evidence, HARVEST/SCOUT candidate, max-loss cap, min-lot feasibility, or replay work exists"
            if triggers
            else "no local artifact currently exposes HARVEST, SCOUT, proof, margin, or replay forward work"
        ),
    }


def _no_action_contract(
    *,
    harvest: dict[str, Any],
    scout: dict[str, Any],
    proof: dict[str, Any],
    board: dict[str, Any],
    portfolio: dict[str, Any],
    replay: dict[str, Any],
    active_deployment_gap: dict[str, Any],
) -> dict[str, Any]:
    blocked_by = list(active_deployment_gap.get("active_path_triggers") or [])
    no_action_allowed = not blocked_by
    return {
        "no_action_allowed": no_action_allowed,
        "reason": (
            "No HARVEST/SCOUT/proof/replay path is visible in local artifacts."
            if no_action_allowed
            else "NO_ACTION is forbidden because at least one profit, proof, or evidence-acquisition path is visible."
        ),
        "blocked_by": blocked_by,
        "next_unlock_action": _unlock_action(replay),
        "why_no_scout": _why_no_scout(scout, board),
        "why_no_harvest": _why_no_harvest(harvest, proof, replay),
        "why_no_evidence_action": (
            ""
            if replay.get("artifact_status") == "missing" or not replay.get("live_grade_candidate")
            else "No evidence-action blocker visible."
        ),
    }


def _select_active_path(
    *,
    no_action: dict[str, Any],
    replay: dict[str, Any],
    proof_floor: dict[str, Any],
    scout: dict[str, Any],
    goal_loop: dict[str, Any],
    harvest: dict[str, Any],
    proof: dict[str, Any],
) -> tuple[str, str]:
    if not no_action.get("no_action_allowed") and (
        replay.get("artifact_status") == "missing"
        or not replay.get("passed")
        or not replay.get("live_grade_candidate")
        or "S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION" in replay.get("blocker_codes", [])
    ):
        return (
            "EVIDENCE_ACQUISITION",
            "Exact LIMIT S5 bid/ask replay exists or is required, but it is not live-grade proof and still needs canonical reconciliation or more exact LIMIT samples.",
        )
    if scout.get("status") == "SCOUT_BLOCKED_OPERATOR_REVIEW":
        return (
            "OPERATOR_REVIEW_REPORT",
            "SCOUT is diagnosed but blocked on operator/guardian review; report only judgement material, not live permission.",
        )
    if harvest.get("candidate_present") and not harvest.get("live_promotion_allowed"):
        return (
            "HARVEST_READY_CHECK",
            "HARVEST candidate exists but promotion blockers remain; check proof queue, replay, and blocker visibility.",
        )
    if proof.get("proof_queue_count", 0) > 0:
        return (
            "LIVE_PERMISSION_READY_CHECK",
            "Proof queue has candidates, so readiness can be checked without granting permission.",
        )
    selected = str(goal_loop.get("selected_next_work_type") or "")
    if selected in {"EDGE_IMPROVEMENT_EXPERIMENT", "OPERATOR_REVIEW_REPORT"}:
        return selected, "Mirrors the trader goal-loop selected work type."
    if proof_floor.get("proof_floor_reached"):
        return (
            "EVIDENCE_ACQUISITION",
            "Proof floor is reached in evidence material, but canonical artifact integration and live-grade replay blockers remain.",
        )
    return "NO_TRADE_WITH_CAUSE", "No active proof path is visible; no-trade must carry machine-readable cause."


def _remaining_blockers(
    *,
    harvest: dict[str, Any],
    scout: dict[str, Any],
    proof: dict[str, Any],
    board: dict[str, Any],
    portfolio: dict[str, Any],
    live_order: dict[str, Any],
    replay: dict[str, Any],
    proof_floor: dict[str, Any],
) -> list[dict[str, Any]]:
    codes = _unique(
        _string_list(harvest.get("promotion_blockers"))
        + _string_list(scout.get("blocker_codes"))
        + _string_list(board.get("blocker_codes"))
        + _string_list(replay.get("blocker_codes"))
        + _string_list(proof_floor.get("blocker_codes"))
    )
    proof_queue_count = proof.get("proof_queue_count", 0)
    if proof_queue_count == 0:
        codes.append("PROOF_QUEUE_COUNT_ZERO_NOT_PERMISSION")
    else:
        stale_empty_queue_codes = {
            "NOT_IN_PROOF_QUEUE",
            "PROOF_QUEUE_EMPTY_NO_LIVE_PERMISSION",
            "PROOF_QUEUE_COUNT_ZERO_NOT_PERMISSION",
        }
        codes = [code for code in codes if code not in stale_empty_queue_codes]
    harvest_tp = harvest.get("tp_proof") if isinstance(harvest.get("tp_proof"), dict) else {}
    broad_tp_proof_reached = bool(proof_floor.get("proof_floor_reached")) or _first_int(
        harvest_tp.get("proof_gap_trades"),
        999,
    ) == 0
    if broad_tp_proof_reached:
        codes = [code for code in codes if code != "SAMPLE_GAP"]
    replay_expectancy = _first_float(replay.get("net_expectancy_after_bidask"))
    if replay.get("passed") and (replay_expectancy is None or replay_expectancy >= 0):
        stale_spread_codes = {
            "POSITIVE_SPREAD_SLIPPAGE_PROOF_MISSING",
            "S5_BIDASK_SPREAD_INCLUDED_REPLAY_MISSING",
            "SPREAD_SLIPPAGE_PROOF_MISSING",
        }
        codes = [code for code in codes if code not in stale_spread_codes]
    if not proof_floor.get("proof_floor_reached"):
        codes.append("PROOF_FLOOR_NOT_CANONICALLY_REACHED")
    if not replay.get("live_grade_candidate"):
        codes.append("LIMIT_S5_REPLAY_NOT_LIVE_GRADE")
    if not portfolio.get("can_create_live_permission"):
        codes.append("PORTFOLIO_PLANNER_CANNOT_CREATE_LIVE_PERMISSION")
    if live_order.get("send_requested") or live_order.get("sent"):
        codes.append("UNEXPECTED_LIVE_ORDER_REQUEST_STATE")
    else:
        codes.append("NO_LIVE_ORDER_REQUEST")
    rows = []
    for code in _unique(codes):
        rows.append(
            {
                "code": code,
                "status": _blocker_status(code),
                "blocks_live_permission": True,
            }
        )
    return rows


def _normalize_stale_blocker_codes(
    state: dict[str, Any],
    *,
    proof: dict[str, Any],
    proof_floor: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    codes = _string_list(state.get("blocker_codes"))
    if not codes:
        return state
    if proof.get("proof_queue_count", 0) > 0:
        codes = [
            code
            for code in codes
            if code not in {"NOT_IN_PROOF_QUEUE", "PROOF_QUEUE_EMPTY_NO_LIVE_PERMISSION"}
        ]
    if proof_floor.get("proof_floor_reached"):
        codes = [code for code in codes if code != "SAMPLE_GAP"]
    replay_expectancy = _first_float(replay.get("net_expectancy_after_bidask"))
    if replay.get("passed") and (replay_expectancy is None or replay_expectancy >= 0):
        stale_spread_codes = {
            "POSITIVE_SPREAD_SLIPPAGE_PROOF_MISSING",
            "S5_BIDASK_SPREAD_INCLUDED_REPLAY_MISSING",
            "SPREAD_SLIPPAGE_PROOF_MISSING",
        }
        codes = [code for code in codes if code not in stale_spread_codes]
    normalized = dict(state)
    normalized["blocker_codes"] = _unique(codes)
    return normalized


def _blocker_status(code: str) -> str:
    if code in {"NEGATIVE_EXPECTANCY_ACTIVE", "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"}:
        return "VISIBLE_PROFITABILITY_BLOCKER"
    if code in {"S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION", "LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY"}:
        return "BLOCKING_PROOF_QUEUE_PROMOTION"
    if "GUARDIAN" in code or "OPERATOR" in code:
        return "BLOCKING_ROUTING_OR_REVIEW"
    return "BLOCKING_LIVE_PERMISSION"


def _status(selected_active_path: str, replay: dict[str, Any], blockers: list[dict[str, Any]]) -> str:
    if selected_active_path == "EVIDENCE_ACQUISITION" and replay.get("passed"):
        return "ACTIVE_PATH_SELECTED_REPLAY_PASSED_STILL_BLOCKED"
    if selected_active_path == "EVIDENCE_ACQUISITION":
        return "ACTIVE_PATH_SELECTED_REPLAY_REQUIRED"
    if blockers:
        return "ACTIVE_PATH_SELECTED_STILL_BLOCKED"
    return "ACTIVE_PATH_SELECTED"


def _unlock_action(replay: dict[str, Any]) -> str:
    if replay.get("artifact_status") == "missing":
        return "Build data/eurusd_short_breakout_failure_limit_s5_bidask_replay.json from local S5 bid/ask candles."
    if "S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION" in replay.get("blocker_codes", []):
        return "Preserve S5 fill/touch lag and reconcile legacy LIMIT rows before proof import."
    if replay.get("replay_sample_count", 0) < 20:
        return "Mine more exact LIMIT / ATTACHED_TP / HARVEST samples without MARKET or STOP rows."
    return "Regenerate proof queue and readiness artifacts after replay/proof import."


def _why_no_scout(scout: dict[str, Any], board: dict[str, Any]) -> str:
    if scout.get("status") == "SCOUT_BLOCKED_OPERATOR_REVIEW":
        return "SCOUT is blocked by operator/guardian review; classify as SCOUT_BLOCKED_OPERATOR_REVIEW, not NO_ACTION."
    if not scout.get("scout_candidate_present"):
        return "No SCOUT candidate is visible."
    if not board.get("routing_allowed"):
        return "Normal routing is not allowed in the current A/S board."
    return "SCOUT requires fresh proof/replay evidence before readiness."


def _why_no_harvest(harvest: dict[str, Any], proof: dict[str, Any], replay: dict[str, Any]) -> str:
    if not harvest.get("candidate_present"):
        return "No HARVEST candidate is visible."
    if not replay.get("live_grade_candidate"):
        return "HARVEST candidate is not live-grade because exact LIMIT S5 replay/proof queue blockers remain."
    if proof.get("proof_queue_count", 0) == 0:
        return "HARVEST candidate is not in the proof queue."
    return "HARVEST readiness requires proof queue and live gateway checks."


def _four_x_progress_hypothesis(
    replay: dict[str, Any],
    proof_floor: dict[str, Any],
    harvest: dict[str, Any],
) -> str:
    wins = replay.get("replay_wins")
    losses = replay.get("replay_losses")
    proof_wins = proof_floor.get("wins")
    proof_losses = proof_floor.get("losses")
    expectancy = replay.get("net_expectancy_after_bidask")
    return (
        f"{TARGET_SHAPE} can move the 4x loop forward only as read-only HARVEST evidence: "
        f"broad TP proof material shows {proof_wins}/{proof_losses}, exact LIMIT S5 replay shows "
        f"{wins}/{losses} with expectancy {expectancy} JPY/trade, while proof queue, guardian, "
        "negative expectancy, and live gateway blockers remain visible."
    )


def _root_improvement_target(replay: dict[str, Any]) -> str:
    if replay.get("artifact_status") == "missing":
        return "Build the exact EUR_USD SHORT BREAKOUT_FAILURE LIMIT HARVEST S5 bid/ask replay."
    return (
        "Make the EUR_USD SHORT BREAKOUT_FAILURE LIMIT HARVEST vehicle live-grade evidence by "
        "canonical replay/proof import, not by mixing MARKET/STOP samples or relaxing gates."
    )


def _expected_edge_improvement(
    replay: dict[str, Any],
    proof_floor: dict[str, Any],
    harvest: dict[str, Any],
) -> str:
    tp = harvest.get("tp_proof") if isinstance(harvest.get("tp_proof"), dict) else {}
    return (
        "Expected improvement is evidence quality, not live permission: exact LIMIT replay and "
        f"proof-floor reconciliation can separate TP-positive HARVEST ({proof_floor.get('wins')}/"
        f"{proof_floor.get('losses')} material; harvest artifact currently {tp.get('take_profit_trades')}/"
        f"{tp.get('take_profit_losses')}) from market-close leakage."
    )


def _next_trade_enabling_action(selected_active_path: str, replay: dict[str, Any]) -> str:
    if selected_active_path == "EVIDENCE_ACQUISITION":
        if replay.get("artifact_status") == "missing":
            return "Generate exact EUR_USD SHORT BREAKOUT_FAILURE LIMIT HARVEST S5 bid/ask replay artifact."
        return "Canonicalize the exact LIMIT S5 bid/ask replay, reconcile legacy rows, and mine more exact LIMIT/HARVEST samples."
    if selected_active_path == "OPERATOR_REVIEW_REPORT":
        return "Package SCOUT approval/rejection evidence without creating live permission."
    if selected_active_path == "HARVEST_READY_CHECK":
        return "Check HARVEST proof queue admission blockers after exact replay/proof import."
    if selected_active_path == "LIVE_PERMISSION_READY_CHECK":
        return "Verify proof queue, risk, verifier, gateway, guardian, and current broker truth without sending orders."
    if selected_active_path == "EDGE_IMPROVEMENT_EXPERIMENT":
        return "Design read-only payoff/sampling experiment for HARVEST candidate."
    return "Emit NO_TRADE_WITH_CAUSE with machine-readable blockers and next unlock action."


def _safety_contract() -> dict[str, Any]:
    return {
        "live_order_send_allowed": False,
        "cancel_allowed": False,
        "close_allowed": False,
        "broker_state_mutation_allowed": False,
        "launchd_load_reload_allowed": False,
        "gate_relaxation_allowed": False,
        "proof_queue_count_zero_is_not_permission": True,
        "no_market_stop_samples_in_limit_proof": True,
        "no_market_close_loss_in_harvest_proof": True,
        "no_4x_deficit_lot_backsolve": True,
        "operator_decision_inference_allowed": False,
        "secret_print_allowed": False,
    }


def _next_prompt(selected_active_path: str, remaining_blockers: list[dict[str, Any]]) -> str:
    blocker_codes = ", ".join(row["code"] for row in remaining_blockers[:10])
    return (
        f"Implement {selected_active_path} for {TARGET_SHAPE} as read-only work. "
        "Do not send, cancel, close, mutate broker state, relax gates, or infer operator approval. "
        f"Keep blockers visible: {blocker_codes}."
    )


def _render_report(payload: dict[str, Any]) -> str:
    blockers = payload.get("remaining_blockers") or []
    current = payload.get("current_state") if isinstance(payload.get("current_state"), dict) else {}
    replay = current.get("limit_s5_bidask_replay") if isinstance(current.get("limit_s5_bidask_replay"), dict) else {}
    no_action = payload.get("no_action_contract") if isinstance(payload.get("no_action_contract"), dict) else {}
    lines = [
        "# Active Trader Contract",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Contract goal: `{payload.get('contract_goal')}`",
        f"- Target shape: `{payload.get('target_shape')}`",
        f"- Selected active path: `{payload.get('selected_active_path')}`",
        f"- Live permission allowed: `{payload.get('live_permission_allowed')}`",
        f"- Read-only: `{payload.get('read_only')}`",
        "",
        "## 4x Progress",
        "",
        payload.get("four_x_progress_hypothesis", ""),
        "",
        f"- Root improvement target: {payload.get('root_improvement_target')}",
        f"- Expected edge improvement: {payload.get('expected_edge_improvement')}",
        f"- Next trade-enabling action: {payload.get('next_trade_enabling_action')}",
        "",
        "## No Action Contract",
        "",
        f"- No action allowed: `{no_action.get('no_action_allowed')}`",
        f"- Reason: {no_action.get('reason')}",
        f"- Next unlock action: {no_action.get('next_unlock_action')}",
        f"- Why no scout: {no_action.get('why_no_scout')}",
        f"- Why no harvest: {no_action.get('why_no_harvest')}",
        "",
        "## Exact LIMIT S5 Replay",
        "",
        f"- Replay status: `{replay.get('status')}`",
        f"- S5 bid/ask status: `{replay.get('s5_bidask_replay_status')}`",
        f"- Samples: `{replay.get('replay_sample_count')}`",
        f"- Wins/losses: `{replay.get('replay_wins')}` / `{replay.get('replay_losses')}`",
        f"- Net expectancy after bid/ask: `{replay.get('net_expectancy_after_bidask')}`",
        f"- Live-grade candidate: `{replay.get('live_grade_candidate')}`",
        "",
        "## Remaining Blockers",
        "",
    ]
    if blockers:
        lines.extend(f"- `{row.get('code')}`: {row.get('status')}" for row in blockers)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "This artifact does not authorize live order entry, SCOUT execution, gateway routing, cancellation, close, TP/SL modification, launchd load/reload, gate relaxation, 4x deficit lot backsolve, or operator-decision inference.",
            "",
        ]
    )
    return "\n".join(lines)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _codes_from_blockers(value: Any) -> list[str]:
    if not isinstance(value, list):
        return _string_list(value)
    codes = []
    for item in value:
        if isinstance(item, dict):
            code = item.get("code") or item.get("row_code") or item.get("blocker")
            if code:
                codes.append(str(code))
        elif item:
            codes.append(str(item))
    return codes


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> int:
    for value in values:
        parsed = _int(value)
        if parsed is not None:
            return parsed
    return 0


def _first_float(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None

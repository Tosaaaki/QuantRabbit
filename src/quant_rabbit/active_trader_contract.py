from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
    DEFAULT_ACTIVE_TRADER_CONTRACT,
    DEFAULT_ACTIVE_TRADER_CONTRACT_REPORT,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_ENTRY_FREQUENCY_RECOVERY,
    DEFAULT_FORECAST_PATTERN_REFRESH,
    DEFAULT_GUARDIAN_EVENTS,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER,
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
    DEFAULT_RANGE_RAIL_GEOMETRY_REPAIR,
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
DEFAULT_EURUSD_LIMIT_SAMPLE_MINING = (
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "eurusd_short_breakout_failure_limit_sample_mining.json"
)
FAILED_EXACT_REPLAY_MARKERS = (
    "S5_TP_PATH_DOES_NOT_RECONSTRUCT_OBSERVED_TP_FILLS",
    "STOP_S5_TRIGGER_OR_TP_PATH_REPLAY_FAILED",
)
BIDASK_REPLAY_EVIDENCE_REFRESH_BLOCKER = "BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED"
TP_PROVEN_ROTATION_BLOCKER = "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"
TP_PROOF_ACQUISITION_ROUTE_UNREACHABLE_BLOCKER = "TP_PROOF_ACQUISITION_ROUTE_UNREACHABLE"
TP_PROOF_ACQUISITION_ROUTE_UNVERIFIED_BLOCKER = "TP_PROOF_ACQUISITION_ROUTE_UNVERIFIED"
NEGATIVE_BLOCKER_MARKERS = (
    "NEGATIVE_EXPECTANCY",
    "REPLAY_NEGATIVE",
    "BIDASK_REPLAY_NEGATIVE",
    "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
    "MARKET_CLOSE_LEAK_DOMINATES",
    "MARKET_CLOSE_LEAK_FAMILY_BLOCKED",
)
OPERATOR_REVIEW_MARKERS = (
    "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
    "GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW",
    "NEEDS_OPERATOR_REVIEW",
    "OPERATOR_REVIEW_REQUIRED",
    "SCOUT_BLOCKED_OPERATOR_REVIEW",
    "DRAFT_REQUIRES_OPERATOR_REVIEW",
    "REVIEW_UNKNOWN_OWNER_EXPOSURE",
    "OPERATOR_REQUESTS_KEEP_BLOCKED",
    "OPERATOR_REQUESTS_FRESH_REVIEW",
)
GUARDIAN_RECEIPT_OPERATOR_REVIEW_MARKERS = ("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",)
GUARDIAN_RECEIPT_ROUTING_CLEAR_STALE_CODES = {
    "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
    "GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING",
    "GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING",
    "GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW",
}
ACTIVE_BOARD_STALE_SOURCE_SUPPRESSED_CODES = GUARDIAN_RECEIPT_ROUTING_CLEAR_STALE_CODES | {
    "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
}


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
        limit_sample_mining_path: Path = DEFAULT_EURUSD_LIMIT_SAMPLE_MINING,
        active_opportunity_board_path: Path = DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
        non_eurusd_live_grade_frontier_path: Path | None = None,
        entry_frequency_recovery_path: Path | None = None,
        forecast_pattern_refresh_path: Path | None = None,
        range_rail_geometry_repair_path: Path | None = None,
        guardian_events_path: Path | None = None,
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
            "eurusd_short_breakout_failure_limit_sample_mining": limit_sample_mining_path,
            "active_opportunity_board": active_opportunity_board_path,
        }
        if non_eurusd_live_grade_frontier_path is not None:
            self.paths["non_eurusd_live_grade_frontier"] = non_eurusd_live_grade_frontier_path
        if entry_frequency_recovery_path is not None:
            self.paths["entry_frequency_recovery"] = entry_frequency_recovery_path
        if forecast_pattern_refresh_path is not None:
            self.paths["forecast_pattern_refresh"] = forecast_pattern_refresh_path
        if range_rail_geometry_repair_path is not None:
            self.paths["range_rail_geometry_repair"] = range_rail_geometry_repair_path
        if guardian_events_path is not None:
            self.paths["guardian_events"] = guardian_events_path
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
        daily_target = _daily_target_contract_state(artifacts["daily_target_state"])
        proof_floor = _proof_floor_contract_state(
            artifacts["eurusd_short_breakout_failure_proof_floor_update"]
        )
        replay = _limit_replay_contract_state(
            artifacts["eurusd_short_breakout_failure_limit_s5_bidask_replay"],
            proof=proof,
        )
        limit_sample_mining = _limit_sample_mining_contract_state(
            artifacts["eurusd_short_breakout_failure_limit_sample_mining"]
        )
        active_opportunity_board = _active_opportunity_board_contract_state(
            artifacts["active_opportunity_board"]
        )
        non_eurusd_frontier = _non_eurusd_frontier_contract_state(
            artifacts.get(
                "non_eurusd_live_grade_frontier",
                {
                    "_artifact_status": "missing",
                    "_path": str(DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER),
                    "_sha256": None,
                },
            )
        )
        entry_frequency_recovery = _entry_frequency_recovery_contract_state(
            artifacts.get(
                "entry_frequency_recovery",
                {
                    "_artifact_status": "missing",
                    "_path": str(DEFAULT_ENTRY_FREQUENCY_RECOVERY),
                    "_sha256": None,
                },
            )
        )
        forecast_pattern_refresh = _forecast_pattern_refresh_contract_state(
            artifacts.get(
                "forecast_pattern_refresh",
                {
                    "_artifact_status": "missing",
                    "_path": str(DEFAULT_FORECAST_PATTERN_REFRESH),
                    "_sha256": None,
                },
            )
        )
        range_rail_geometry_repair = _range_rail_geometry_repair_contract_state(
            artifacts.get(
                "range_rail_geometry_repair",
                {
                    "_artifact_status": "missing",
                    "_path": str(DEFAULT_RANGE_RAIL_GEOMETRY_REPAIR),
                    "_sha256": None,
                },
            )
        )
        guardian_events = _guardian_events_contract_state(
            artifacts.get(
                "guardian_events",
                {
                    "_artifact_status": "missing",
                    "_path": str(DEFAULT_GUARDIAN_EVENTS),
                    "_sha256": None,
                },
            )
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
            active_opportunity_board=active_opportunity_board,
            non_eurusd_frontier=non_eurusd_frontier,
        )
        no_action = _no_action_contract(
            harvest=harvest,
            scout=scout,
            proof=proof,
            board=board,
            portfolio=portfolio,
            replay=replay,
            active_opportunity_board=active_opportunity_board,
            non_eurusd_frontier=non_eurusd_frontier,
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
            active_opportunity_board=active_opportunity_board,
            non_eurusd_frontier=non_eurusd_frontier,
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
            limit_sample_mining=limit_sample_mining,
            active_opportunity_board=active_opportunity_board,
            non_eurusd_frontier=non_eurusd_frontier,
        )
        target_shape = _contract_target_shape(
            selected_active_path=selected_active_path,
            active_opportunity_board=active_opportunity_board,
            non_eurusd_frontier=non_eurusd_frontier,
            entry_frequency_recovery=entry_frequency_recovery,
            forecast_pattern_refresh=forecast_pattern_refresh,
            range_rail_geometry_repair=range_rail_geometry_repair,
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
            "target_shape": target_shape,
            "four_x_progress_hypothesis": _four_x_progress_hypothesis(
                replay,
                proof_floor,
                harvest,
                active_opportunity_board=active_opportunity_board,
                non_eurusd_frontier=non_eurusd_frontier,
                entry_frequency_recovery=entry_frequency_recovery,
                forecast_pattern_refresh=forecast_pattern_refresh,
                range_rail_geometry_repair=range_rail_geometry_repair,
                guardian_events=guardian_events,
            ),
            "root_improvement_target": _root_improvement_target(
                replay,
                active_opportunity_board=active_opportunity_board,
                non_eurusd_frontier=non_eurusd_frontier,
                entry_frequency_recovery=entry_frequency_recovery,
                forecast_pattern_refresh=forecast_pattern_refresh,
                range_rail_geometry_repair=range_rail_geometry_repair,
                guardian_events=guardian_events,
            ),
            "expected_edge_improvement": _expected_edge_improvement(
                replay,
                proof_floor,
                harvest,
                active_opportunity_board=active_opportunity_board,
                non_eurusd_frontier=non_eurusd_frontier,
                entry_frequency_recovery=entry_frequency_recovery,
                forecast_pattern_refresh=forecast_pattern_refresh,
                range_rail_geometry_repair=range_rail_geometry_repair,
                guardian_events=guardian_events,
            ),
            "no_action_allowed": bool(no_action["no_action_allowed"]),
            "no_action_contract": no_action,
            "active_deployment_gap": active_deployment_gap,
            "next_trade_enabling_action": _next_trade_enabling_action(
                selected_active_path,
                replay,
                limit_sample_mining,
                active_opportunity_board=active_opportunity_board,
                non_eurusd_frontier=non_eurusd_frontier,
                entry_frequency_recovery=entry_frequency_recovery,
                forecast_pattern_refresh=forecast_pattern_refresh,
                range_rail_geometry_repair=range_rail_geometry_repair,
                guardian_events=guardian_events,
            ),
            "remaining_blockers": remaining_blockers,
            "current_state": {
                "harvest": harvest,
                "scout": scout,
                "proof": proof,
                "board": board,
                "portfolio": portfolio,
                "live_order": live_order,
                "goal_loop": goal_loop,
                "daily_target": daily_target,
                "proof_floor": proof_floor,
                "limit_s5_bidask_replay": replay,
                "limit_sample_mining": limit_sample_mining,
                "active_opportunity_board": active_opportunity_board,
                "non_eurusd_live_grade_frontier": non_eurusd_frontier,
                "entry_frequency_recovery": entry_frequency_recovery,
                "forecast_pattern_refresh": forecast_pattern_refresh,
                "range_rail_geometry_repair": range_rail_geometry_repair,
                "guardian_events": guardian_events,
            },
            "safety_contract": _safety_contract(),
            "next_prompt": _next_prompt(
                selected_active_path,
                remaining_blockers,
                active_opportunity_board=active_opportunity_board,
                non_eurusd_frontier=non_eurusd_frontier,
                entry_frequency_recovery=entry_frequency_recovery,
                forecast_pattern_refresh=forecast_pattern_refresh,
                range_rail_geometry_repair=range_rail_geometry_repair,
                guardian_events=guardian_events,
            ),
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


def _daily_target_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    """Expose pace economics without letting them alter active-path routing."""

    if artifact.get("_artifact_status") == "missing":
        return {"artifact_status": "missing", "trade_pace_feasibility": None}
    return {
        "artifact_status": "present",
        "generated_at_utc": artifact.get("generated_at_utc"),
        "status": artifact.get("status"),
        "target_jpy": artifact.get("target_jpy"),
        "target_trades_per_day": artifact.get("target_trades_per_day"),
        "uncapped_required_trades_per_day": artifact.get(
            "uncapped_required_trades_per_day"
        ),
        "uncapped_required_trades_per_day_basis_return_pct": artifact.get(
            "uncapped_required_trades_per_day_basis_return_pct"
        ),
        "selected_basis_uncapped_required_trades_per_day": artifact.get(
            "selected_basis_uncapped_required_trades_per_day"
        ),
        "selected_basis_return_pct": artifact.get("selected_basis_return_pct"),
        "operating_pace_trades_per_day": artifact.get("operating_pace_trades_per_day"),
        "automated_operating_cap_trades_per_day": artifact.get(
            "automated_operating_cap_trades_per_day"
        ),
        "observed_trades_per_day": artifact.get("observed_trades_per_day"),
        "observed_expectancy_jpy_per_trade": artifact.get(
            "observed_expectancy_jpy_per_trade"
        ),
        "frequency_multiple_required": artifact.get("frequency_multiple_required"),
        "planned_reward_at_operating_pace_jpy": artifact.get(
            "planned_reward_at_operating_pace_jpy"
        ),
        "stretch_required_minus_operating_gap_trades_per_day": artifact.get(
            "stretch_required_minus_operating_gap_trades_per_day"
        ),
        "selected_required_minus_operating_gap_trades_per_day": artifact.get(
            "selected_required_minus_operating_gap_trades_per_day"
        ),
        "trade_pace_feasible_within_operating_pace": artifact.get(
            "trade_pace_feasible_within_operating_pace"
        ),
        "trade_pace_feasibility": artifact.get("trade_pace_feasibility"),
        "advisory_only": True,
    }


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
        return {
            "artifact_status": "missing",
            "selected_next_work_type": None,
            "classified_next_work_type": None,
            "work_dispatch_allowed": None,
            "repeat_suppressed": None,
        }
    return {
        "artifact_status": "present",
        "generated_at_utc": artifact.get("generated_at_utc"),
        "status": artifact.get("status"),
        "selected_next_work_type": artifact.get("selected_next_work_type"),
        "classified_next_work_type": artifact.get("classified_next_work_type"),
        "work_dispatch_allowed": artifact.get("work_dispatch_allowed"),
        "repeat_suppressed": artifact.get("repeat_suppressed"),
        "four_x_progress_hypothesis": artifact.get("four_x_progress_hypothesis"),
        "root_improvement_target": artifact.get("root_improvement_target"),
        "expected_edge_improvement": artifact.get("expected_edge_improvement"),
        "live_permission_allowed": bool(artifact.get("live_permission_allowed")),
    }


def _goal_loop_dispatch_allowed(goal_loop: dict[str, Any]) -> bool:
    """Accept legacy artifacts, but fail closed on explicit or malformed suppression state."""

    dispatch_allowed = goal_loop.get("work_dispatch_allowed")
    repeat_suppressed = goal_loop.get("repeat_suppressed")
    if dispatch_allowed is not None and not isinstance(dispatch_allowed, bool):
        return False
    if repeat_suppressed is not None and not isinstance(repeat_suppressed, bool):
        return False
    return dispatch_allowed is not False and repeat_suppressed is not True


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


def _limit_sample_mining_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "status": "MISSING",
            "floor_met": False,
            "blocker_codes": ["LIMIT_SAMPLE_MINING_ARTIFACT_MISSING"],
        }
    floor = artifact.get("sample_floor") if isinstance(artifact.get("sample_floor"), dict) else {}
    execution = (
        artifact.get("execution_ledger_coverage")
        if isinstance(artifact.get("execution_ledger_coverage"), dict)
        else {}
    )
    legacy = (
        artifact.get("legacy_history_coverage")
        if isinstance(artifact.get("legacy_history_coverage"), dict)
        else {}
    )
    return {
        "artifact_status": "present",
        "status": artifact.get("status"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "target_shape": artifact.get("target_shape"),
        "current_replayed_exact_limit_samples": _first_int(floor.get("current_replayed_exact_limit_samples"), 0),
        "additional_acceptable_local_samples_found": _first_int(
            floor.get("additional_acceptable_local_samples_found"),
            0,
        ),
        "remaining_exact_limit_samples": _first_int(floor.get("remaining_exact_limit_samples"), 0),
        "required_exact_limit_samples": _first_int(floor.get("required_exact_limit_samples"), 20),
        "floor_met": bool(floor.get("floor_met")),
        "execution_ledger_summary": execution.get("summary") or {},
        "legacy_history_summary": legacy.get("summary") or {},
        "blocker_codes": _codes_from_blockers(artifact.get("remaining_blockers")),
    }


def _active_opportunity_board_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "status": "MISSING",
            "total_lanes": 0,
            "top_lane": {},
            "failed_exact_replay_consumed_count": 0,
            "consumed_failed_replay_lanes": [],
            "consumed_failed_replay_blocker_codes": [],
        }
    summary = artifact.get("coverage_summary") if isinstance(artifact.get("coverage_summary"), dict) else {}
    global_safety = artifact.get("global_safety") if isinstance(artifact.get("global_safety"), dict) else {}
    top_lane = artifact.get("top_lane") if isinstance(artifact.get("top_lane"), dict) else {}
    guardian_routing_clear = global_safety.get("guardian_receipt_normal_routing_allowed") is True
    lanes = artifact.get("ranked_active_lanes") if isinstance(artifact.get("ranked_active_lanes"), list) else []
    top_lane_summary = _contract_lane_summary(
        top_lane,
        guardian_routing_clear=guardian_routing_clear,
    )
    status_counts = _normalized_active_board_status_counts(
        summary,
        lanes,
        top_lane=top_lane,
    )
    stale_source_reasons = artifact.get("stale_source_reasons") if isinstance(artifact.get("stale_source_reasons"), list) else []
    stale_source_blocker_codes = _unique(
        _codes_from_blockers(stale_source_reasons)
        + _string_list(top_lane.get("stale_source_blockers"))
        + [
            code
            for lane in lanes
            if isinstance(lane, dict)
            for code in _string_list(lane.get("stale_source_blockers"))
        ]
    )
    consumed: list[dict[str, Any]] = []
    blocker_codes: list[str] = []
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        blockers = _string_list(lane.get("blockers"))
        if not any(any(marker in blocker for marker in FAILED_EXACT_REPLAY_MARKERS) for blocker in blockers):
            continue
        failed_codes = [
            blocker
            for blocker in blockers
            if any(marker in blocker for marker in FAILED_EXACT_REPLAY_MARKERS)
        ]
        blocker_codes.extend(failed_codes)
        consumed.append(
            {
                "lane_id": lane.get("lane_id"),
                "pair": lane.get("pair"),
                "direction": lane.get("direction"),
                "strategy_family": lane.get("strategy_family"),
                "vehicle": lane.get("vehicle"),
                "status": lane.get("status"),
                "replay_status": lane.get("replay_status"),
                "scout_candidate_after_replay": False,
                "consumed_as": "FAILED_EXACT_REPLAY_NOT_SCOUT_READY",
                "failed_blocker_codes": _unique(failed_codes),
            }
        )
    return {
        "artifact_status": "present",
        "status": artifact.get("status"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "total_lanes": _first_int(summary.get("total_lanes"), len(lanes)),
        "live_ready_count": status_counts["LIVE_READY"],
        "harvest_ready_count": status_counts["HARVEST_READY"],
        "scout_ready_count": status_counts["SCOUT_READY"],
        "evidence_acquisition_count": status_counts["EVIDENCE_ACQUISITION"],
        "operator_review_required_count": status_counts["OPERATOR_REVIEW_REQUIRED"],
        "pairs_scanned_count": len(_string_list(summary.get("pairs_scanned"))),
        "vehicles_scanned": _string_list(summary.get("vehicles_scanned")),
        "next_active_path": artifact.get("next_active_path"),
        "top_lane": top_lane_summary,
        "stale_source_blocker_codes": stale_source_blocker_codes,
        "guardian_receipt_normal_routing_allowed": guardian_routing_clear,
        "failed_exact_replay_consumed_count": len(consumed),
        "consumed_failed_replay_lanes": consumed[:10],
        "consumed_failed_replay_blocker_codes": _unique(blocker_codes),
        "stop_harvest_failed_replay_consumed": any(
            row.get("pair") == "EUR_USD"
            and row.get("direction") == "SHORT"
            and row.get("strategy_family") == "BREAKOUT_FAILURE"
            and row.get("vehicle") == "STOP"
            for row in consumed
        ),
        "live_permission_allowed": False,
    }


def _normalized_active_board_status_counts(
    summary: dict[str, Any],
    lanes: list[Any],
    *,
    top_lane: dict[str, Any],
) -> dict[str, int]:
    summary_keys = {
        "LIVE_READY": "live_ready_count",
        "HARVEST_READY": "harvest_ready_count",
        "SCOUT_READY": "scout_ready_count",
        "EVIDENCE_ACQUISITION": "evidence_acquisition_count",
        "OPERATOR_REVIEW_REQUIRED": "operator_review_required_count",
        "NO_TRADE_WITH_CAUSE": "no_trade_count",
    }
    counts = {
        status: _first_int(summary.get(summary_key), 0)
        for status, summary_key in summary_keys.items()
    }
    total_lanes = _first_int(summary.get("total_lanes"), len(lanes))
    complete_ranked_lanes = total_lanes > 0 and len(lanes) == total_lanes
    route_rows = lanes if complete_ranked_lanes else [top_lane]
    for lane in route_rows:
        if not isinstance(lane, dict) or not _tp_proof_acquisition_route_unreachable(lane):
            continue
        raw_status = str(lane.get("status") or "")
        if raw_status == "NO_TRADE_WITH_CAUSE":
            continue
        if raw_status in counts:
            counts[raw_status] = max(0, counts[raw_status] - 1)
        counts["NO_TRADE_WITH_CAUSE"] += 1
    return counts


def _non_eurusd_frontier_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "status": "MISSING",
            "scanned_intents": 0,
            "scanned_pairs_count": 0,
            "top_lane": {},
            "top_non_eurusd_lane": {},
            "next_evidence_lane": {},
            "next_active_path": None,
            "non_eurusd_closer_than_eurusd": False,
            "spread_too_wide_not_ignored": False,
            "bidask_negative_not_ignored": False,
            "usd_cad_long_breakout_failure_count": 0,
            "live_permission_allowed": False,
        }
    checks = artifact.get("required_checks") if isinstance(artifact.get("required_checks"), dict) else {}
    next_lane = checks.get("next_evidence_lane") if isinstance(checks.get("next_evidence_lane"), dict) else {}
    top_non = artifact.get("top_non_eurusd_lane") if isinstance(artifact.get("top_non_eurusd_lane"), dict) else {}
    top_lane = artifact.get("top_lane") if isinstance(artifact.get("top_lane"), dict) else {}
    if not next_lane:
        next_lane = top_non or top_lane
    return {
        "artifact_status": "present",
        "status": artifact.get("status"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "read_only": artifact.get("read_only") is True,
        "scanned_intents": _first_int(artifact.get("scanned_intents"), 0),
        "scanned_pairs_count": len(_string_list(artifact.get("scanned_pairs"))),
        "top_lane": _frontier_lane_summary(top_lane),
        "top_non_eurusd_lane": _frontier_lane_summary(top_non),
        "next_evidence_lane": _frontier_lane_summary(next_lane),
        "next_active_path": artifact.get("next_active_path"),
        "non_eurusd_closer_than_eurusd": checks.get("non_eurusd_closer_than_eurusd") is True,
        "spread_too_wide_not_ignored": checks.get("spread_too_wide_not_ignored") is True,
        "bidask_negative_not_ignored": checks.get("bidask_negative_not_ignored") is True,
        "usd_cad_long_breakout_failure_count": len(
            checks.get("usd_cad_long_breakout_failure_blocker_breakdown")
            if isinstance(checks.get("usd_cad_long_breakout_failure_blocker_breakdown"), list)
            else []
        ),
        "live_permission_allowed": False,
    }


def _entry_frequency_recovery_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "status": "MISSING",
            "target_lane_count": 0,
            "top_lane": {},
            "tuning_queue": [],
            "next_contract_prompt": None,
            "live_permission_allowed": False,
        }
    top = artifact.get("top_lane") if isinstance(artifact.get("top_lane"), dict) else {}
    tuning_queue = artifact.get("forecast_pattern_tuning_queue")
    tuning_queue = tuning_queue if isinstance(tuning_queue, list) else []
    return {
        "artifact_status": "present",
        "status": artifact.get("status"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "target_lane_count": _first_int(artifact.get("target_lane_count"), len(artifact.get("target_lanes") or [])),
        "top_lane": {
            "lane_id": top.get("lane_id"),
            "pair": top.get("pair"),
            "direction": top.get("direction"),
            "strategy_family": top.get("strategy_family"),
            "vehicle": top.get("vehicle"),
            "status": top.get("status"),
            "forecast_status": (top.get("forecast_audit") or {}).get("status")
            if isinstance(top.get("forecast_audit"), dict)
            else None,
            "profile_status": (top.get("strategy_profile_audit") or {}).get("status")
            if isinstance(top.get("strategy_profile_audit"), dict)
            else None,
            "tp_proof_status": (top.get("tp_proof_audit") or {}).get("status")
            if isinstance(top.get("tp_proof_audit"), dict)
            else None,
        },
        "tuning_queue": [
            {
                "priority": row.get("priority"),
                "lane_id": row.get("lane_id"),
                "action_type": row.get("action_type"),
                "description": row.get("description"),
                "preserve_blockers": _string_list(row.get("preserve_blockers")),
            }
            for row in tuning_queue[:8]
            if isinstance(row, dict)
        ],
        "next_contract_prompt": artifact.get("next_contract_prompt"),
        "live_permission_allowed": False,
    }


def _forecast_pattern_refresh_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "status": "MISSING",
            "top_lane": {},
            "next_actions": [],
            "next_contract_prompt": None,
            "live_permission_allowed": False,
        }
    top = artifact.get("top_lane") if isinstance(artifact.get("top_lane"), dict) else {}
    counterpart = (
        top.get("range_rotation_counterpart")
        if isinstance(top.get("range_rotation_counterpart"), dict)
        else {}
    )
    forecast = top.get("forecast_range_box") if isinstance(top.get("forecast_range_box"), dict) else {}
    projection = (
        top.get("projection_trigger_audit")
        if isinstance(top.get("projection_trigger_audit"), dict)
        else {}
    )
    actions = artifact.get("next_actions") if isinstance(artifact.get("next_actions"), list) else []
    return {
        "artifact_status": "present",
        "status": artifact.get("status"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "top_lane": {
            "lane_id": top.get("lane_id"),
            "pair": top.get("pair"),
            "direction": top.get("direction"),
            "strategy_family": top.get("strategy_family"),
            "vehicle": top.get("vehicle"),
            "status": top.get("status"),
            "range_counterpart_status": counterpart.get("status"),
            "forecast_box_status": forecast.get("status"),
            "trigger_projection_status": projection.get("status"),
        },
        "next_actions": [
            {
                "priority": row.get("priority"),
                "lane_id": row.get("lane_id"),
                "action_type": row.get("action_type"),
                "description": row.get("description"),
                "preserve_blockers": _string_list(row.get("preserve_blockers")),
            }
            for row in actions[:8]
            if isinstance(row, dict)
        ],
        "next_contract_prompt": artifact.get("next_contract_prompt"),
        "live_permission_allowed": False,
    }


def _range_rail_geometry_repair_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "status": "MISSING",
            "top_lane": {},
            "next_actions": [],
            "next_contract_prompt": None,
            "live_permission_allowed": False,
        }
    top = artifact.get("top_lane") if isinstance(artifact.get("top_lane"), dict) else {}
    box = top.get("range_box") if isinstance(top.get("range_box"), dict) else {}
    geometry = top.get("counterpart_geometry") if isinstance(top.get("counterpart_geometry"), dict) else {}
    condition = (
        top.get("rail_success_condition")
        if isinstance(top.get("rail_success_condition"), dict)
        else {}
    )
    actions = artifact.get("next_actions") if isinstance(artifact.get("next_actions"), list) else []
    return {
        "artifact_status": "present",
        "status": artifact.get("status"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "top_lane": {
            "lane_id": top.get("lane_id"),
            "pair": top.get("pair"),
            "direction": top.get("direction"),
            "strategy_family": top.get("strategy_family"),
            "vehicle": top.get("vehicle"),
            "status": top.get("status"),
            "rail_status": box.get("rail_status"),
            "box_position": box.get("box_position"),
            "required_zone": box.get("required_zone"),
            "counterpart_geometry_status": geometry.get("status"),
            "counterpart_geometry_ready": bool(geometry.get("geometry_ready")),
            "rail_success_condition": condition,
        },
        "next_actions": [
            {
                "priority": row.get("priority"),
                "lane_id": row.get("lane_id"),
                "action_type": row.get("action_type"),
                "description": row.get("description"),
                "preserve_blockers": _string_list(row.get("preserve_blockers")),
            }
            for row in actions[:8]
            if isinstance(row, dict)
        ],
        "next_contract_prompt": artifact.get("next_contract_prompt"),
        "live_permission_allowed": False,
    }


def _guardian_events_contract_state(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("_artifact_status") == "missing":
        return {
            "artifact_status": "missing",
            "generated_at_utc": None,
            "range_rail_trigger_count": 0,
            "range_rail_triggers": [],
            "live_permission_allowed": False,
        }
    events = artifact.get("events") if isinstance(artifact.get("events"), list) else []
    range_rail_triggers: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if str(event.get("event_type") or "").upper() != "CONTRACT_ADD_TRIGGER":
            continue
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        trigger = details.get("contract_trigger") if isinstance(details.get("contract_trigger"), dict) else {}
        if str(trigger.get("kind") or "").upper() != "RANGE_RAIL_RECHECK":
            continue
        condition = trigger.get("condition") if isinstance(trigger.get("condition"), dict) else {}
        rail_condition = (
            trigger.get("rail_success_condition")
            if isinstance(trigger.get("rail_success_condition"), dict)
            else {}
        )
        range_rail_triggers.append(
            {
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "lane_id": trigger.get("lane_id"),
                "pair": event.get("pair") or trigger.get("pair"),
                "direction": event.get("direction") or trigger.get("side") or rail_condition.get("direction"),
                "action_hint": event.get("action_hint") or trigger.get("action_hint"),
                "severity": event.get("severity"),
                "dedupe_key": event.get("dedupe_key"),
                "price_zone": event.get("price_zone"),
                "condition": condition,
                "rail_success_condition": rail_condition,
                "preserve_blockers": _string_list(trigger.get("preserve_blockers")),
                "live_permission_allowed": trigger.get("live_permission_allowed") is True,
                "contract_triggers_do_not_execute": trigger.get("contract_triggers_do_not_execute") is True,
            }
        )
    return {
        "artifact_status": "present",
        "generated_at_utc": artifact.get("generated_at_utc"),
        "range_rail_trigger_count": len(range_rail_triggers),
        "range_rail_triggers": range_rail_triggers[:12],
        "live_permission_allowed": False,
    }


def _contract_lane_summary(lane: dict[str, Any], *, guardian_routing_clear: bool = False) -> dict[str, Any]:
    if not lane:
        return {}
    blockers = _string_list(lane.get("blockers"))
    stale_source_blockers = _string_list(lane.get("stale_source_blockers"))
    if guardian_routing_clear:
        stale_guardian_codes = [
            code
            for code in blockers
            if code in GUARDIAN_RECEIPT_ROUTING_CLEAR_STALE_CODES
        ]
        if stale_guardian_codes:
            blockers = [code for code in blockers if code not in stale_guardian_codes]
            stale_source_blockers = _unique(stale_source_blockers + stale_guardian_codes)
    route_unreachable = _tp_proof_acquisition_route_unreachable(lane)
    if route_unreachable:
        blockers = _unique([_tp_proof_acquisition_route_blocker(lane)] + blockers)
    return {
        "lane_id": lane.get("lane_id"),
        "pair": lane.get("pair"),
        "direction": lane.get("direction"),
        "strategy_family": lane.get("strategy_family"),
        "vehicle": lane.get("vehicle"),
        "status": _normalized_board_lane_status(
            lane.get("status"),
            blockers,
            edge_improvement_candidate=lane.get("edge_improvement_candidate") is True,
        ),
        "replay_status": lane.get("replay_status"),
        "proof_status": lane.get("proof_status"),
        "risk_status": lane.get("risk_status"),
        "guardian_status": lane.get("guardian_status"),
        "operator_review_status": lane.get("operator_review_status"),
        "expected_edge_jpy": lane.get("expected_edge_jpy"),
        "next_action": (
            _tp_proof_acquisition_route_rerank_action(lane)
            if route_unreachable
            else lane.get("next_action")
        ),
        "local_tp_proof": (
            lane.get("local_tp_proof")
            if isinstance(lane.get("local_tp_proof"), dict)
            else {}
        ),
        "tp_proof_acquisition_required": lane.get("tp_proof_acquisition_required") is True,
        "tp_proof_acquisition_route_reachable": lane.get(
            "tp_proof_acquisition_route_reachable"
        ),
        "tp_proof_acquisition_route_status": lane.get("tp_proof_acquisition_route_status"),
        "tp_proof_acquisition_route_reason": lane.get("tp_proof_acquisition_route_reason"),
        "positive_rotation_mode": lane.get("positive_rotation_mode"),
        "blockers": blockers[:24],
        "stale_source_blockers": stale_source_blockers[:12],
        "edge_improvement_candidate": lane.get("edge_improvement_candidate") is True,
        "edge_improvement_target": lane.get("edge_improvement_target"),
    }


def _frontier_lane_summary(lane: dict[str, Any]) -> dict[str, Any]:
    if not lane:
        return {}
    return {
        "lane_id": lane.get("lane_id"),
        "pair": lane.get("pair"),
        "direction": lane.get("direction"),
        "strategy_family": lane.get("strategy_family"),
        "vehicle": lane.get("vehicle"),
        "status": lane.get("status"),
        "distance_to_live_ready": lane.get("distance_to_live_ready"),
        "bidask_status": lane.get("bidask_status"),
        "spread_status": lane.get("spread_status"),
        "forecast_status": lane.get("forecast_status"),
        "loss_budget_status": lane.get("loss_budget_status"),
        "tp_proof_count": lane.get("tp_proof_count"),
        "tp_proof_floor": lane.get("tp_proof_floor"),
        "expected_edge_jpy": lane.get("expected_edge_jpy"),
        "next_action": lane.get("next_action"),
        "blockers": _string_list(lane.get("blockers"))[:24],
    }


def _normalized_board_lane_status(
    status: Any,
    blockers: list[str],
    *,
    edge_improvement_candidate: bool = False,
) -> str:
    raw = str(status or "NO_TRADE_WITH_CAUSE")
    if {
        TP_PROOF_ACQUISITION_ROUTE_UNREACHABLE_BLOCKER,
        TP_PROOF_ACQUISITION_ROUTE_UNVERIFIED_BLOCKER,
    }.intersection(blockers):
        return "NO_TRADE_WITH_CAUSE"
    if any(any(marker in blocker for marker in FAILED_EXACT_REPLAY_MARKERS) for blocker in blockers):
        return "NO_TRADE_WITH_CAUSE"
    if any(any(marker in blocker for marker in GUARDIAN_RECEIPT_OPERATOR_REVIEW_MARKERS) for blocker in blockers):
        return "OPERATOR_REVIEW_REQUIRED"
    if BIDASK_REPLAY_EVIDENCE_REFRESH_BLOCKER in blockers:
        return raw
    if raw == "EVIDENCE_ACQUISITION" and blockers == [TP_PROVEN_ROTATION_BLOCKER]:
        return raw
    if raw == "EVIDENCE_ACQUISITION" and "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR" in blockers:
        return raw
    if raw == "EVIDENCE_ACQUISITION" and edge_improvement_candidate:
        return raw
    if raw == "EVIDENCE_ACQUISITION" and "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH" in blockers:
        return raw
    if any(any(marker in blocker for marker in NEGATIVE_BLOCKER_MARKERS) for blocker in blockers):
        return "NO_TRADE_WITH_CAUSE"
    if any(any(marker in blocker for marker in OPERATOR_REVIEW_MARKERS) for blocker in blockers):
        return "OPERATOR_REVIEW_REQUIRED"
    return raw


def _tp_proof_acquisition_route_unreachable(value: dict[str, Any]) -> bool:
    status = str(value.get("tp_proof_acquisition_route_status") or "")
    explicitly_blocked = bool(
        value.get("tp_proof_acquisition_route_reachable") is False
    ) or bool(
        status
        in {
            TP_PROOF_ACQUISITION_ROUTE_UNREACHABLE_BLOCKER,
            TP_PROOF_ACQUISITION_ROUTE_UNVERIFIED_BLOCKER,
        }
    )
    return bool(
        explicitly_blocked
        or (
            value.get("tp_proof_acquisition_required") is True
            and value.get("tp_proof_acquisition_route_reachable") is not True
        )
    )


def _tp_proof_acquisition_route_blocker(value: dict[str, Any]) -> str:
    status = str(value.get("tp_proof_acquisition_route_status") or "")
    if status in {
        TP_PROOF_ACQUISITION_ROUTE_UNREACHABLE_BLOCKER,
        TP_PROOF_ACQUISITION_ROUTE_UNVERIFIED_BLOCKER,
    }:
        return status
    return TP_PROOF_ACQUISITION_ROUTE_UNREACHABLE_BLOCKER


def _tp_proof_acquisition_route_rerank_action(value: dict[str, Any]) -> str:
    lane_id = str(value.get("lane_id") or "current board lane")
    route_blocker = _tp_proof_acquisition_route_blocker(value)
    reason = str(value.get("tp_proof_acquisition_route_reason") or "").strip()
    reason_suffix = f" Reason: {reason}." if reason else ""
    return (
        f"No trade for {lane_id}; {route_blocker} means the current gates cannot create "
        f"the requested TP-proof receipts.{reason_suffix} Wait for a materially new exact "
        "reachable route or rerank another lane while preserving RR, chase, profitability, "
        "risk, verifier, and gateway blockers."
    )


def _active_deployment_gap(
    *,
    harvest: dict[str, Any],
    scout: dict[str, Any],
    proof: dict[str, Any],
    board: dict[str, Any],
    portfolio: dict[str, Any],
    replay: dict[str, Any],
    active_opportunity_board: dict[str, Any],
    non_eurusd_frontier: dict[str, Any],
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
    if active_opportunity_board.get("total_lanes", 0) > 0:
        triggers.append("ACTIVE_OPPORTUNITY_BOARD_RERANK_AVAILABLE")
    if active_opportunity_board.get("failed_exact_replay_consumed_count", 0) > 0:
        triggers.append("FAILED_EXACT_REPLAY_CONSUMED")
    if _frontier_evidence_action_available(non_eurusd_frontier):
        triggers.append("NON_EURUSD_LIVE_GRADE_FRONTIER_AVAILABLE")
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
    active_opportunity_board: dict[str, Any],
    non_eurusd_frontier: dict[str, Any],
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
        "why_no_board_rerank": _why_no_board_rerank(active_opportunity_board),
        "why_no_evidence_action": (
            ""
            if (
                replay.get("artifact_status") == "missing"
                or not replay.get("live_grade_candidate")
                or _frontier_evidence_action_available(non_eurusd_frontier)
            )
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
    active_opportunity_board: dict[str, Any],
    non_eurusd_frontier: dict[str, Any],
) -> tuple[str, str]:
    board_top = active_opportunity_board.get("top_lane")
    board_top = board_top if isinstance(board_top, dict) else {}
    board_status = str(board_top.get("status") or "")
    frontier_lane = _frontier_evidence_lane(non_eurusd_frontier)
    if _active_board_all_no_trade(active_opportunity_board, board_status=board_status):
        if frontier_lane:
            return (
                "EVIDENCE_ACQUISITION",
                "Latest active opportunity board reranked all lanes as NO_TRADE_WITH_CAUSE, but "
                "non_eurusd_live_grade_frontier exposes a current read-only evidence action for "
                f"{frontier_lane.get('lane_id')}; keep blockers visible and do not send.",
            )
        return (
            "NO_TRADE_WITH_CAUSE",
            "Latest active opportunity board reranked all lanes as NO_TRADE_WITH_CAUSE; do not fall back to stale single-lane evidence work.",
        )
    if active_opportunity_board.get("total_lanes", 0) > 0 and board_status in {
        "LIVE_READY",
        "HARVEST_READY",
        "SCOUT_READY",
        "EVIDENCE_ACQUISITION",
        "OPERATOR_REVIEW_REQUIRED",
    }:
        path_by_status = {
            "LIVE_READY": "LIVE_PERMISSION_READY_CHECK",
            "HARVEST_READY": "HARVEST_READY_CHECK",
            "SCOUT_READY": "SCOUT_READY_CHECK",
            "EVIDENCE_ACQUISITION": "EVIDENCE_ACQUISITION",
            "OPERATOR_REVIEW_REQUIRED": "OPERATOR_REVIEW_REPORT",
        }
        consumed = active_opportunity_board.get("failed_exact_replay_consumed_count", 0)
        consumed_note = (
            f" Failed exact replay lanes consumed={consumed}; do not repeat the same STOP replay."
            if consumed
            else ""
        )
        frontier_note = ""
        if _frontier_supplements_board_evidence(board_top, non_eurusd_frontier):
            frontier_note = (
                " Non-EUR frontier points to the same pair/side/family proof lane "
                f"{frontier_lane.get('lane_id')} ({frontier_lane.get('vehicle')}); "
                "consume it with the board lane as one read-only unblock plan."
            )
        elif _frontier_parallel_board_evidence(board_top, non_eurusd_frontier):
            frontier_note = (
                " Non-EUR frontier also exposes a distinct read-only evidence lane "
                f"{frontier_lane.get('lane_id')} ({frontier_lane.get('vehicle')}); "
                "queue it as secondary evidence work after the board lane so EUR/USD does not monopolize repair."
            )
        return (
            path_by_status[board_status],
            "Latest active opportunity board is available from the previous refresh and has already "
            f"ranked {active_opportunity_board.get('total_lanes')} lanes; top lane "
            f"{board_top.get('lane_id')} is {board_status}.{consumed_note}{frontier_note}",
        )
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
    if (
        _goal_loop_dispatch_allowed(goal_loop)
        and selected in {"EDGE_IMPROVEMENT_EXPERIMENT", "OPERATOR_REVIEW_REPORT"}
    ):
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
    limit_sample_mining: dict[str, Any],
    active_opportunity_board: dict[str, Any],
    non_eurusd_frontier: dict[str, Any],
) -> list[dict[str, Any]]:
    active_board_top = active_opportunity_board.get("top_lane")
    active_board_top = active_board_top if isinstance(active_board_top, dict) else {}
    frontier_lane = _frontier_evidence_lane(non_eurusd_frontier)
    use_frontier_lane = (
        _active_board_all_no_trade(active_opportunity_board)
        or _frontier_supplements_board_evidence(active_board_top, non_eurusd_frontier)
        or _frontier_parallel_board_evidence(active_board_top, non_eurusd_frontier)
    ) and frontier_lane
    frontier_codes = _string_list(frontier_lane.get("blockers")) if use_frontier_lane else []
    active_board_authoritative = _active_board_authoritative(active_opportunity_board)
    if active_board_authoritative:
        codes = _unique(
            _string_list(active_board_top.get("blockers"))
            + _string_list(active_opportunity_board.get("consumed_failed_replay_blocker_codes"))
            + frontier_codes
        )
    else:
        codes = _unique(
            _string_list(harvest.get("promotion_blockers"))
            + _string_list(scout.get("blocker_codes"))
            + _string_list(board.get("blocker_codes"))
            + _string_list(replay.get("blocker_codes"))
            + _string_list(proof_floor.get("blocker_codes"))
            + _string_list(limit_sample_mining.get("blocker_codes"))
            + _string_list(active_board_top.get("blockers"))
            + _string_list(active_opportunity_board.get("consumed_failed_replay_blocker_codes"))
            + frontier_codes
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
    if active_opportunity_board.get("guardian_receipt_normal_routing_allowed") is True:
        codes = [code for code in codes if code not in GUARDIAN_RECEIPT_ROUTING_CLEAR_STALE_CODES]
    active_board_stale_codes = set(
        _string_list(active_board_top.get("stale_source_blockers"))
        + _string_list(active_opportunity_board.get("stale_source_blocker_codes"))
    )
    if active_board_stale_codes:
        suppress_from_board_truth = active_board_stale_codes & ACTIVE_BOARD_STALE_SOURCE_SUPPRESSED_CODES
        codes = [code for code in codes if code not in suppress_from_board_truth]
    if not active_board_authoritative:
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
    elif not active_board_authoritative:
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


def _active_board_authoritative(active_opportunity_board: dict[str, Any], *, board_status: str | None = None) -> bool:
    board_top = active_opportunity_board.get("top_lane")
    board_top = board_top if isinstance(board_top, dict) else {}
    status = str(board_status if board_status is not None else board_top.get("status") or "")
    return active_opportunity_board.get("total_lanes", 0) > 0 and status in {
        "LIVE_READY",
        "HARVEST_READY",
        "SCOUT_READY",
        "EVIDENCE_ACQUISITION",
        "OPERATOR_REVIEW_REQUIRED",
        "NO_TRADE_WITH_CAUSE",
    }


def _active_board_all_no_trade(active_opportunity_board: dict[str, Any], *, board_status: str | None = None) -> bool:
    board_top = active_opportunity_board.get("top_lane")
    board_top = board_top if isinstance(board_top, dict) else {}
    status = str(board_status if board_status is not None else board_top.get("status") or "")
    if active_opportunity_board.get("total_lanes", 0) <= 0 or status != "NO_TRADE_WITH_CAUSE":
        return False
    active_counts = sum(
        int(active_opportunity_board.get(key) or 0)
        for key in (
            "live_ready_count",
            "harvest_ready_count",
            "scout_ready_count",
            "evidence_acquisition_count",
            "operator_review_required_count",
        )
    )
    return active_counts == 0


def _frontier_evidence_action_available(non_eurusd_frontier: dict[str, Any]) -> bool:
    if non_eurusd_frontier.get("artifact_status") != "present":
        return False
    if non_eurusd_frontier.get("live_permission_allowed") is True:
        return False
    lane = _frontier_evidence_lane(non_eurusd_frontier)
    if not lane:
        return False
    status = str(non_eurusd_frontier.get("status") or "")
    return status in {
        "NON_EURUSD_FRONTIER_FOUND",
        "ONLY_EURUSD_FRONTIER_FOUND",
        "ALL_FRONTIER_BLOCKED_BY_NEGATIVE_EXPECTANCY",
        "ALL_FRONTIER_BLOCKED_BY_SPREAD_OR_FORECAST",
    }


def _frontier_evidence_lane(non_eurusd_frontier: dict[str, Any]) -> dict[str, Any]:
    lane = non_eurusd_frontier.get("next_evidence_lane")
    lane = lane if isinstance(lane, dict) else {}
    if lane.get("lane_id"):
        return lane
    top_non = non_eurusd_frontier.get("top_non_eurusd_lane")
    top_non = top_non if isinstance(top_non, dict) else {}
    if top_non.get("lane_id"):
        return top_non
    top = non_eurusd_frontier.get("top_lane")
    top = top if isinstance(top, dict) else {}
    return top if top.get("lane_id") else {}


def _lane_target_shape(lane: dict[str, Any]) -> str | None:
    pair = str(lane.get("pair") or "").strip().upper()
    direction = str(lane.get("direction") or lane.get("side") or "").strip().upper()
    strategy = str(lane.get("strategy_family") or "").strip().upper()
    vehicle = str(lane.get("vehicle") or "").strip().upper()
    parts = [part for part in (pair, direction, strategy, vehicle) if part]
    return "|".join(parts) if len(parts) >= 4 else None


def _frontier_blocker_fragment(lane: dict[str, Any], *, label: str = "Frontier blockers") -> str:
    blockers = _unique(_string_list(lane.get("blockers")))[:10]
    if not blockers:
        return ""
    return f"{label}: {', '.join(blockers)}. "


def _frontier_evidence_prompt(
    non_eurusd_frontier: dict[str, Any],
    *,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
    guardian_events: dict[str, Any] | None = None,
) -> str:
    lane = _frontier_evidence_lane(non_eurusd_frontier)
    if not lane:
        return "Use non_eurusd_live_grade_frontier: acquire the next frontier-ranked evidence packet."
    shape = _lane_target_shape(lane)
    shape_text = f" ({shape})" if shape else ""
    next_action = _frontier_next_action_text(
        lane,
        non_eurusd_frontier,
        entry_frequency_recovery=entry_frequency_recovery,
        forecast_pattern_refresh=forecast_pattern_refresh,
        range_rail_geometry_repair=range_rail_geometry_repair,
        guardian_events=guardian_events,
    )
    return (
        "Use non_eurusd_live_grade_frontier: "
        f"next evidence lane {lane.get('lane_id')}{shape_text} "
        f"({lane.get('vehicle')}, {lane.get('distance_to_live_ready')}). "
        f"{next_action} "
        f"{_frontier_blocker_fragment(lane)}"
        "Keep negative expectancy, spread, bid/ask, forecast, and loss-budget blockers visible; do not send."
    )


def _frontier_artifact_prompt(
    frontier_lane: dict[str, Any],
    *,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
    guardian_events: dict[str, Any] | None = None,
) -> str | None:
    consumed = _frontier_artifact_consumption(
        frontier_lane,
        entry_frequency_recovery=entry_frequency_recovery,
        forecast_pattern_refresh=forecast_pattern_refresh,
        range_rail_geometry_repair=range_rail_geometry_repair,
        guardian_events=guardian_events,
    )
    return consumed[1] if consumed else None


def _frontier_artifact_consumption(
    frontier_lane: dict[str, Any],
    *,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
    guardian_events: dict[str, Any] | None = None,
) -> tuple[str, str] | None:
    guardian_prompt = _guardian_range_rail_trigger_prompt(guardian_events or {}, frontier_lane)
    if guardian_prompt:
        return ("guardian_events", guardian_prompt)
    for artifact_name, artifact in (
        ("range_rail_geometry_repair", range_rail_geometry_repair or {}),
        ("forecast_pattern_refresh", forecast_pattern_refresh or {}),
        ("entry_frequency_recovery", entry_frequency_recovery or {}),
    ):
        if artifact.get("artifact_status") != "present":
            continue
        top = artifact.get("top_lane") if isinstance(artifact.get("top_lane"), dict) else {}
        prompt = str(artifact.get("next_contract_prompt") or "").strip()
        if top and prompt and _lane_shape(top) == _lane_shape(frontier_lane):
            return artifact_name, prompt
    return None


def _frontier_next_action_text(
    frontier_lane: dict[str, Any],
    non_eurusd_frontier: dict[str, Any],
    *,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
    guardian_events: dict[str, Any] | None = None,
) -> str:
    return (
        _frontier_artifact_prompt(
            frontier_lane,
            entry_frequency_recovery=entry_frequency_recovery,
            forecast_pattern_refresh=forecast_pattern_refresh,
            range_rail_geometry_repair=range_rail_geometry_repair,
            guardian_events=guardian_events,
        )
        or frontier_lane.get("next_action")
        or non_eurusd_frontier.get("next_active_path")
        or "Acquire the frontier evidence packet."
    )


def _frontier_supplements_board_evidence(
    board_top: dict[str, Any],
    non_eurusd_frontier: dict[str, Any],
) -> bool:
    if not board_top or not _frontier_evidence_action_available(non_eurusd_frontier):
        return False
    if str(board_top.get("status") or "") != "EVIDENCE_ACQUISITION":
        return False
    frontier_lane = _frontier_evidence_lane(non_eurusd_frontier)
    if not frontier_lane or frontier_lane.get("lane_id") == board_top.get("lane_id"):
        return False
    board_shape = (
        str(board_top.get("pair") or "").upper(),
        str(board_top.get("direction") or board_top.get("side") or "").upper(),
        str(board_top.get("strategy_family") or "").upper(),
    )
    frontier_shape = (
        str(frontier_lane.get("pair") or "").upper(),
        str(frontier_lane.get("direction") or frontier_lane.get("side") or "").upper(),
        str(frontier_lane.get("strategy_family") or "").upper(),
    )
    if board_shape != frontier_shape or not all(board_shape):
        return False
    evidence_markers = {
        "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
        "LOCAL_TP_PROOF_ZERO_TRADES",
        "HARVEST_TP_STRUCTURE_MISSING",
        "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
        "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
    }
    blocker_codes = set(_string_list(board_top.get("blockers"))) | set(
        _string_list(frontier_lane.get("blockers"))
    )
    return bool(blocker_codes & evidence_markers)


def _frontier_parallel_board_evidence(
    board_top: dict[str, Any],
    non_eurusd_frontier: dict[str, Any],
) -> bool:
    if not board_top or not _frontier_evidence_action_available(non_eurusd_frontier):
        return False
    if str(non_eurusd_frontier.get("status") or "") != "NON_EURUSD_FRONTIER_FOUND":
        return False
    if str(board_top.get("status") or "") != "EVIDENCE_ACQUISITION":
        return False
    if str(board_top.get("pair") or "").upper() != "EUR_USD":
        return False
    frontier_lane = _frontier_evidence_lane(non_eurusd_frontier)
    if not frontier_lane or frontier_lane.get("lane_id") == board_top.get("lane_id"):
        return False
    if str(frontier_lane.get("pair") or "").upper() == "EUR_USD":
        return False
    if _frontier_supplements_board_evidence(board_top, non_eurusd_frontier):
        return False
    board_blockers = set(_string_list(board_top.get("blockers")))
    board_bidask_refresh_or_negative = {
        "BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED",
        "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
        "BIDASK_REPLAY_NEGATIVE",
    }
    if board_blockers & board_bidask_refresh_or_negative:
        return False
    return bool(frontier_lane.get("next_action") or non_eurusd_frontier.get("next_active_path"))


def _frontier_supplement_prompt(
    board_top: dict[str, Any],
    non_eurusd_frontier: dict[str, Any],
    *,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
    guardian_events: dict[str, Any] | None = None,
) -> str:
    if _frontier_supplements_board_evidence(board_top, non_eurusd_frontier):
        frontier_lane = _frontier_evidence_lane(non_eurusd_frontier)
        frontier_shape = _lane_target_shape(frontier_lane)
        frontier_ref = frontier_shape or str(frontier_lane.get("lane_id") or "frontier lane")
        next_action = _frontier_next_action_text(
            frontier_lane,
            non_eurusd_frontier,
            entry_frequency_recovery=entry_frequency_recovery,
            forecast_pattern_refresh=forecast_pattern_refresh,
            range_rail_geometry_repair=range_rail_geometry_repair,
            guardian_events=guardian_events,
        )
        return (
            f" Pair this with frontier evidence {frontier_ref}: "
            f"{next_action} "
            f"{_frontier_blocker_fragment(frontier_lane)}"
            "Keep both blocker sets visible; do not send."
        )
    if _frontier_parallel_board_evidence(board_top, non_eurusd_frontier):
        frontier_lane = _frontier_evidence_lane(non_eurusd_frontier)
        frontier_shape = _lane_target_shape(frontier_lane)
        frontier_ref = frontier_shape or str(frontier_lane.get("lane_id") or "frontier lane")
        next_action = _frontier_next_action_text(
            frontier_lane,
            non_eurusd_frontier,
            entry_frequency_recovery=entry_frequency_recovery,
            forecast_pattern_refresh=forecast_pattern_refresh,
            range_rail_geometry_repair=range_rail_geometry_repair,
            guardian_events=guardian_events,
        )
        return (
            f" Parallel non_eurusd_live_grade_frontier evidence {frontier_ref}: "
            f"{next_action} "
            f"{_frontier_blocker_fragment(frontier_lane, label='Non-EUR frontier blockers')}"
            "Keep the non-EUR blocker set visible; do not send."
        )
    return ""


def _frontier_action_suffix(
    board_top: dict[str, Any],
    non_eurusd_frontier: dict[str, Any],
    *,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
    guardian_events: dict[str, Any] | None = None,
) -> str:
    frontier_lane = _frontier_evidence_lane(non_eurusd_frontier)
    if not frontier_lane:
        return ""
    next_action = _frontier_next_action_text(
        frontier_lane,
        non_eurusd_frontier,
        entry_frequency_recovery=entry_frequency_recovery,
        forecast_pattern_refresh=forecast_pattern_refresh,
        range_rail_geometry_repair=range_rail_geometry_repair,
        guardian_events=guardian_events,
    )
    if _frontier_supplements_board_evidence(board_top, non_eurusd_frontier):
        return (
            " Pair this with non_eurusd_live_grade_frontier evidence lane "
            f"{frontier_lane.get('lane_id')} "
            f"({frontier_lane.get('vehicle')}, {frontier_lane.get('distance_to_live_ready')}). "
            f"{next_action} "
            f"{_frontier_blocker_fragment(frontier_lane)}"
            "Keep both blocker sets visible; do not send."
        )
    if _frontier_parallel_board_evidence(board_top, non_eurusd_frontier):
        return (
            " Parallel non_eurusd_live_grade_frontier evidence lane "
            f"{frontier_lane.get('lane_id')} "
            f"({frontier_lane.get('vehicle')}, {frontier_lane.get('distance_to_live_ready')}). "
            f"{next_action} "
            f"{_frontier_blocker_fragment(frontier_lane, label='Non-EUR frontier blockers')}"
            "Keep the non-EUR blocker set visible; do not send."
        )
    return ""


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
    if any(marker in code for marker in FAILED_EXACT_REPLAY_MARKERS):
        return "FAILED_REPLAY_CONSUMED_BLOCKS_SCOUT"
    if code == BIDASK_REPLAY_EVIDENCE_REFRESH_BLOCKER:
        return "BLOCKING_EVIDENCE_REFRESH"
    if code in {"NEGATIVE_EXPECTANCY_ACTIVE", "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"}:
        return "VISIBLE_PROFITABILITY_BLOCKER"
    if code in {
        "S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION",
        "LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY",
        "LOCAL_LIMIT_SAMPLE_COVERAGE_EXHAUSTED",
    }:
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


def _why_no_board_rerank(active_opportunity_board: dict[str, Any]) -> str:
    if active_opportunity_board.get("artifact_status") == "missing":
        return "No previous active opportunity board artifact is available yet."
    if active_opportunity_board.get("total_lanes", 0) <= 0:
        return "Previous active opportunity board has no lane candidates."
    failed = active_opportunity_board.get("failed_exact_replay_consumed_count", 0)
    if failed:
        return (
            f"Previous active opportunity board already consumed {failed} failed exact replay lane(s); "
            "rerank from board facts instead of repeating that replay."
        )
    return "Previous active opportunity board is available for multi-pair/multi-vehicle rerank context."


def _four_x_progress_hypothesis(
    replay: dict[str, Any],
    proof_floor: dict[str, Any],
    harvest: dict[str, Any],
    *,
    active_opportunity_board: dict[str, Any] | None = None,
    non_eurusd_frontier: dict[str, Any] | None = None,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
    guardian_events: dict[str, Any] | None = None,
) -> str:
    board = active_opportunity_board if isinstance(active_opportunity_board, dict) else {}
    board_top = board.get("top_lane") if isinstance(board.get("top_lane"), dict) else {}
    frontier = non_eurusd_frontier if isinstance(non_eurusd_frontier, dict) else {}
    active_shape = _active_board_target_shape(board)
    frontier_shape = _frontier_target_shape(frontier)
    consumed_prompt = _consumed_lane_prompt(
        board_top,
        entry_frequency_recovery=entry_frequency_recovery,
        forecast_pattern_refresh=forecast_pattern_refresh,
        range_rail_geometry_repair=range_rail_geometry_repair,
        guardian_events=guardian_events,
    )
    if active_shape and frontier_shape and _frontier_parallel_board_evidence(board_top, frontier):
        board_blockers = ", ".join(_string_list(board_top.get("blockers"))[:5])
        frontier_lane = _frontier_evidence_lane(frontier)
        frontier_blockers = ", ".join(_string_list(frontier_lane.get("blockers"))[:5])
        expected_edge = board_top.get("expected_edge_jpy")
        edge_text = f" board expected_edge_jpy={expected_edge};" if expected_edge is not None else ""
        return (
            f"{active_shape} is the active board path, and {frontier_shape} is the parallel "
            f"non-EUR/USD frontier evidence lane.{edge_text} The 4x loop moves forward only if both "
            "lanes keep their blockers visible and the next cycle consumes the non-EUR frontier action "
            f"instead of reverting to an EUR/USD-only loop. Board blockers: {board_blockers or 'none'}. "
            f"Frontier blockers: {frontier_blockers or 'none'}."
        )
    if active_shape and consumed_prompt:
        source, _prompt = consumed_prompt
        blockers = ", ".join(_string_list(board_top.get("blockers"))[:6])
        expected_edge = board_top.get("expected_edge_jpy")
        edge_text = f" expected_edge_jpy={expected_edge};" if expected_edge is not None else ""
        return (
            f"{active_shape} is the current multi-pair active path.{edge_text} It can move the 4x loop "
            f"forward only as read-only lane-local evidence by consuming {source} and then advancing "
            f"the next proof/reprice action. Remaining blockers stay visible: {blockers or 'none'}."
        )
    if active_shape:
        blockers = ", ".join(_string_list(board_top.get("blockers"))[:6])
        expected_edge = board_top.get("expected_edge_jpy")
        edge_text = f" expected_edge_jpy={expected_edge};" if expected_edge is not None else ""
        return (
            f"{active_shape} is the current multi-pair active path.{edge_text} It can move the 4x loop "
            f"forward only by preserving and reducing concrete lane blockers: {blockers or 'none'}."
        )
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


def _root_improvement_target(
    replay: dict[str, Any],
    *,
    active_opportunity_board: dict[str, Any] | None = None,
    non_eurusd_frontier: dict[str, Any] | None = None,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
    guardian_events: dict[str, Any] | None = None,
) -> str:
    board = active_opportunity_board if isinstance(active_opportunity_board, dict) else {}
    board_top = board.get("top_lane") if isinstance(board.get("top_lane"), dict) else {}
    frontier = non_eurusd_frontier if isinstance(non_eurusd_frontier, dict) else {}
    active_shape = _active_board_target_shape(board)
    frontier_shape = _frontier_target_shape(frontier)
    consumed_prompt = _consumed_lane_prompt(
        board_top,
        entry_frequency_recovery=entry_frequency_recovery,
        forecast_pattern_refresh=forecast_pattern_refresh,
        range_rail_geometry_repair=range_rail_geometry_repair,
        guardian_events=guardian_events,
    )
    if active_shape and frontier_shape and _frontier_parallel_board_evidence(board_top, frontier):
        return (
            f"Advance {active_shape} and the parallel non-EUR/USD frontier lane {frontier_shape} as "
            "one blocker-preserving evidence plan; do not let the EUR/USD board lane hide the "
            "frontier bid/ask, spread, forecast, TP-proof, or expectancy gaps."
        )
    if active_shape and consumed_prompt:
        return (
            f"Advance {active_shape} by consuming the latest lane-local evidence artifact "
            "instead of repeating older board work or the legacy EUR_USD loop."
        )
    if active_shape:
        return (
            f"Advance {active_shape} toward live-grade evidence from the current active board, "
            "while preserving forecast, spread, bid/ask, proof, and expectancy blockers."
        )
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
    *,
    active_opportunity_board: dict[str, Any] | None = None,
    non_eurusd_frontier: dict[str, Any] | None = None,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
    guardian_events: dict[str, Any] | None = None,
) -> str:
    board = active_opportunity_board if isinstance(active_opportunity_board, dict) else {}
    board_top = board.get("top_lane") if isinstance(board.get("top_lane"), dict) else {}
    frontier = non_eurusd_frontier if isinstance(non_eurusd_frontier, dict) else {}
    active_shape = _active_board_target_shape(board)
    frontier_shape = _frontier_target_shape(frontier)
    consumed_prompt = _consumed_lane_prompt(
        board_top,
        entry_frequency_recovery=entry_frequency_recovery,
        forecast_pattern_refresh=forecast_pattern_refresh,
        range_rail_geometry_repair=range_rail_geometry_repair,
        guardian_events=guardian_events,
    )
    if active_shape and frontier_shape and _frontier_parallel_board_evidence(board_top, frontier):
        expected_edge = board_top.get("expected_edge_jpy")
        edge_text = f" board expected_edge_jpy={expected_edge};" if expected_edge is not None else ""
        frontier_lane = _frontier_evidence_lane(frontier)
        frontier_edge = frontier_lane.get("expected_edge_jpy")
        frontier_edge_text = (
            f" frontier expected_edge_jpy={frontier_edge};" if frontier_edge is not None else ""
        )
        return (
            f"Expected improvement is broader executable evidence coverage: {active_shape} remains the "
            f"board lane,{edge_text} while {frontier_shape} supplies the non-EUR/USD repair surface,"
            f"{frontier_edge_text} with its blocker set carried into remaining_blockers. Live permission "
            "remains false until both lane-local gates and gateway checks pass."
        )
    if active_shape and consumed_prompt:
        expected_edge = board_top.get("expected_edge_jpy")
        edge_text = f" current board expected_edge_jpy={expected_edge};" if expected_edge is not None else ""
        return (
            f"Expected improvement is loop progress for {active_shape}:{edge_text} consume the "
            "latest lane-local action so the next cycle moves to repricing/proof collection "
            "instead of repeating drought or forecast refresh analysis. Live permission remains false."
        )
    if active_shape:
        expected_edge = board_top.get("expected_edge_jpy")
        edge_text = f" current board expected_edge_jpy={expected_edge};" if expected_edge is not None else ""
        return (
            f"Expected improvement is evidence quality for {active_shape}:{edge_text} choose the "
            "shortest blocker-preserving unlock path across pairs and vehicles without granting live permission."
        )
    tp = harvest.get("tp_proof") if isinstance(harvest.get("tp_proof"), dict) else {}
    return (
        "Expected improvement is evidence quality, not live permission: exact LIMIT replay and "
        f"proof-floor reconciliation can separate TP-positive HARVEST ({proof_floor.get('wins')}/"
        f"{proof_floor.get('losses')} material; harvest artifact currently {tp.get('take_profit_trades')}/"
        f"{tp.get('take_profit_losses')}) from market-close leakage."
    )


def _consumed_lane_prompt(
    board_top: dict[str, Any],
    *,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
    guardian_events: dict[str, Any] | None = None,
) -> tuple[str, str] | None:
    guardian_prompt = _guardian_range_rail_trigger_prompt(guardian_events or {}, board_top)
    if guardian_prompt:
        return ("guardian_events", guardian_prompt)
    if _range_rail_geometry_repair_matches_board(range_rail_geometry_repair or {}, board_top):
        prompt = str((range_rail_geometry_repair or {}).get("next_contract_prompt") or "").strip()
        if prompt:
            return ("range_rail_geometry_repair", prompt)
    if _forecast_pattern_refresh_matches_board(forecast_pattern_refresh or {}, board_top):
        prompt = str((forecast_pattern_refresh or {}).get("next_contract_prompt") or "").strip()
        if prompt:
            return ("forecast_pattern_refresh", prompt)
    if _entry_frequency_recovery_matches_board(entry_frequency_recovery or {}, board_top):
        prompt = str((entry_frequency_recovery or {}).get("next_contract_prompt") or "").strip()
        if prompt:
            return ("entry_frequency_recovery", prompt)
    return None


def _guardian_range_rail_trigger_prompt(
    guardian_events: dict[str, Any],
    board_top: dict[str, Any],
) -> str | None:
    trigger = _guardian_range_rail_trigger_for_board(guardian_events, board_top)
    if not trigger:
        return None
    lane_id = trigger.get("lane_id") or board_top.get("lane_id")
    price_zone = str(trigger.get("price_zone") or "range rail trigger fired").strip()
    blockers = ", ".join(_string_list(trigger.get("preserve_blockers"))[:8])
    return (
        "Consume data/guardian_events.json for "
        f"{lane_id}: CONTRACT_ADD_TRIGGER fired for the watched range rail ({price_zone}). "
        "Do not repeat WAIT_FOR_RANGE_RAIL_RECHECK; refresh broker truth and active board, "
        "then reprice the RANGE_ROTATION counterpart and continue exact TP-proof collection. "
        f"Preserve blockers: {blockers or 'current board blockers'}. "
        "This is a read-only wake/next-work transition, not live permission."
    )


def _guardian_range_rail_trigger_for_board(
    guardian_events: dict[str, Any],
    board_top: dict[str, Any],
) -> dict[str, Any]:
    if not guardian_events or not board_top:
        return {}
    if guardian_events.get("artifact_status") != "present":
        return {}
    triggers = guardian_events.get("range_rail_triggers")
    if not isinstance(triggers, list):
        return {}
    board_lane_id = str(board_top.get("lane_id") or "").strip()
    board_shape = _lane_shape(board_top)
    for trigger in triggers:
        if not isinstance(trigger, dict):
            continue
        lane_id = str(trigger.get("lane_id") or "").strip()
        if board_lane_id and lane_id == board_lane_id:
            return trigger
        if board_shape and _lane_shape(trigger) == board_shape:
            return trigger
    return {}


def _next_trade_enabling_action(
    selected_active_path: str,
    replay: dict[str, Any],
    limit_sample_mining: dict[str, Any] | None = None,
    *,
    active_opportunity_board: dict[str, Any] | None = None,
    non_eurusd_frontier: dict[str, Any] | None = None,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
    guardian_events: dict[str, Any] | None = None,
) -> str:
    active_opportunity_board = active_opportunity_board or {}
    non_eurusd_frontier = non_eurusd_frontier or {}
    entry_frequency_recovery = entry_frequency_recovery or {}
    forecast_pattern_refresh = forecast_pattern_refresh or {}
    range_rail_geometry_repair = range_rail_geometry_repair or {}
    guardian_events = guardian_events or {}
    board_top = active_opportunity_board.get("top_lane")
    board_top = board_top if isinstance(board_top, dict) else {}
    if selected_active_path == "NO_TRADE_WITH_CAUSE" and _tp_proof_acquisition_route_unreachable(
        board_top
    ):
        return _tp_proof_acquisition_route_rerank_action(board_top)
    if selected_active_path == "EVIDENCE_ACQUISITION":
        frontier_lane = _frontier_evidence_lane(non_eurusd_frontier)
        if (
            _active_board_all_no_trade(active_opportunity_board)
            and _frontier_evidence_action_available(non_eurusd_frontier)
            and frontier_lane
        ):
            return _frontier_evidence_prompt(
                non_eurusd_frontier,
                entry_frequency_recovery=entry_frequency_recovery,
                forecast_pattern_refresh=forecast_pattern_refresh,
                range_rail_geometry_repair=range_rail_geometry_repair,
                guardian_events=guardian_events,
            )
        if board_top:
            suffix = ""
            if active_opportunity_board.get("failed_exact_replay_consumed_count", 0) > 0:
                suffix = " STOP exact replay is already consumed as failed/not-SCOUT-ready; do not repeat it."
            frontier_suffix = ""
            if frontier_lane:
                frontier_suffix = _frontier_action_suffix(
                    board_top,
                    non_eurusd_frontier,
                    entry_frequency_recovery=entry_frequency_recovery,
                    forecast_pattern_refresh=forecast_pattern_refresh,
                    range_rail_geometry_repair=range_rail_geometry_repair,
                    guardian_events=guardian_events,
                )
            frontier_consumption = (
                _frontier_artifact_consumption(
                    frontier_lane,
                    entry_frequency_recovery=entry_frequency_recovery,
                    forecast_pattern_refresh=forecast_pattern_refresh,
                    range_rail_geometry_repair=range_rail_geometry_repair,
                    guardian_events=guardian_events,
                )
                if frontier_lane
                else None
            )
            if (
                _range_rail_latest_forecast_not_range(range_rail_geometry_repair, board_top)
                and _frontier_evidence_action_available(non_eurusd_frontier)
                and frontier_lane
            ):
                frontier_prompt = _frontier_evidence_prompt(
                    non_eurusd_frontier,
                    entry_frequency_recovery=entry_frequency_recovery,
                    forecast_pattern_refresh=forecast_pattern_refresh,
                    range_rail_geometry_repair=range_rail_geometry_repair,
                    guardian_events=guardian_events,
                )
                return (
                    "Use the latest active_opportunity_board rerank: "
                    f"top lane {board_top.get('lane_id')} ({board_top.get('vehicle')}, {board_top.get('status')}) "
                    "has consumed range_rail_geometry_repair, but the latest forecast is no longer RANGE. "
                    "Do not repeat range-box refresh for that invalidated range-rail path. "
                    f"{frontier_prompt}"
                    f"{suffix}"
                )
            consumed_prompt = _consumed_lane_prompt(
                board_top,
                entry_frequency_recovery=entry_frequency_recovery,
                forecast_pattern_refresh=forecast_pattern_refresh,
                range_rail_geometry_repair=range_rail_geometry_repair,
                guardian_events=guardian_events,
            )
            if consumed_prompt:
                artifact_name, artifact_prompt = consumed_prompt
                return (
                    "Use the latest active_opportunity_board rerank: "
                    f"top lane {board_top.get('lane_id')} ({board_top.get('vehicle')}, {board_top.get('status')}). "
                    f"Consume {artifact_name} artifact as the current next action: {artifact_prompt}"
                    f"{frontier_suffix}"
                    f"{suffix}"
                )
            if frontier_consumption and _frontier_parallel_board_evidence(board_top, non_eurusd_frontier):
                artifact_name, artifact_prompt = frontier_consumption
                board_blockers = ", ".join(_string_list(board_top.get("blockers"))[:8])
                return (
                    "Advance non_eurusd_live_grade_frontier as the current next action: "
                    f"frontier lane {frontier_lane.get('lane_id')} "
                    f"({frontier_lane.get('vehicle')}, {frontier_lane.get('distance_to_live_ready')}). "
                    f"Consume {artifact_name} artifact: {artifact_prompt} "
                    f"Keep active board lane {board_top.get('lane_id')} visible but do not let it hide the frontier artifact; "
                    f"board blockers: {board_blockers or 'none'}. "
                    f"{_frontier_blocker_fragment(frontier_lane, label='Non-EUR frontier blockers')}"
                    "Do not send, cancel, close, relax gates, or treat frontier artifact consumption as live permission."
                    f"{suffix}"
                )
            return (
                "Use the latest active_opportunity_board rerank: "
                f"top lane {board_top.get('lane_id')} ({board_top.get('vehicle')}, {board_top.get('status')}). "
                f"{board_top.get('next_action') or 'Acquire the next board-ranked evidence packet.'}"
                f"{frontier_suffix}"
                f"{suffix}"
            )
        if replay.get("artifact_status") == "missing":
            return "Generate exact EUR_USD SHORT BREAKOUT_FAILURE LIMIT HARVEST S5 bid/ask replay artifact."
        if (limit_sample_mining or {}).get("status") == "LOCAL_LIMIT_SAMPLE_COVERAGE_EXHAUSTED_STILL_UNDERSAMPLED":
            return (
                "Canonicalize the exact LIMIT S5 bid/ask replay and import or locate additional "
                "exact LIMIT/HARVEST rows; current local execution/legacy coverage found 0 new acceptable samples."
            )
        return "Canonicalize the exact LIMIT S5 bid/ask replay, reconcile legacy rows, and mine more exact LIMIT/HARVEST samples."
    if selected_active_path == "OPERATOR_REVIEW_REPORT":
        if board_top:
            suffix = ""
            if active_opportunity_board.get("failed_exact_replay_consumed_count", 0) > 0:
                suffix = " STOP exact replay is already consumed as failed/not-SCOUT-ready; do not repeat it."
            after_review = _operator_review_follow_up(board_top)
            return (
                "Package operator/guardian review evidence for the latest active_opportunity_board top lane "
                f"{board_top.get('lane_id')} ({board_top.get('vehicle')}, {board_top.get('status')}); "
                "keep live permission false until explicit operator review is recorded."
                f"{after_review}"
                f"{suffix}"
            )
        return "Package SCOUT approval/rejection evidence without creating live permission."
    if selected_active_path == "HARVEST_READY_CHECK":
        return "Check HARVEST proof queue admission blockers after exact replay/proof import."
    if selected_active_path == "LIVE_PERMISSION_READY_CHECK":
        return "Verify proof queue, risk, verifier, gateway, guardian, and current broker truth without sending orders."
    if selected_active_path == "EDGE_IMPROVEMENT_EXPERIMENT":
        return "Design read-only payoff/sampling experiment for HARVEST candidate."
    if selected_active_path == "NO_TRADE_WITH_CAUSE" and board_top:
        blockers = ", ".join(_string_list(board_top.get("blockers"))[:5])
        return (
            "Emit NO_TRADE_WITH_CAUSE from the latest active_opportunity_board: "
            f"top lane {board_top.get('lane_id')} ({board_top.get('vehicle')}) is blocked by {blockers}."
        )
    return "Emit NO_TRADE_WITH_CAUSE with machine-readable blockers and next unlock action."


def _operator_review_follow_up(board_top: dict[str, Any]) -> str:
    next_action = str(board_top.get("next_action") or "")
    marker = "After review clears,"
    if marker in next_action:
        return " " + next_action[next_action.index(marker) :]
    if board_top.get("edge_improvement_candidate") is True:
        target = board_top.get("edge_improvement_target") or board_top.get("lane_id") or "top lane"
        return (
            f" After review clears, run read-only EDGE_IMPROVEMENT_EXPERIMENT for {target}; "
            "preserve remaining blockers and do not send."
        )
    return ""


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


def _entry_frequency_recovery_matches_board(
    entry_frequency_recovery: dict[str, Any],
    board_top: dict[str, Any],
) -> bool:
    if not entry_frequency_recovery or not board_top:
        return False
    if entry_frequency_recovery.get("artifact_status") != "present":
        return False
    top = entry_frequency_recovery.get("top_lane")
    if not isinstance(top, dict):
        return False
    if not entry_frequency_recovery.get("next_contract_prompt"):
        return False
    if top.get("lane_id") and top.get("lane_id") == board_top.get("lane_id"):
        return True
    return (
        str(top.get("pair") or "").upper(),
        str(top.get("direction") or "").upper(),
        str(top.get("strategy_family") or "").upper(),
    ) == (
        str(board_top.get("pair") or "").upper(),
        str(board_top.get("direction") or "").upper(),
        str(board_top.get("strategy_family") or "").upper(),
    )


def _forecast_pattern_refresh_matches_board(
    forecast_pattern_refresh: dict[str, Any],
    board_top: dict[str, Any],
) -> bool:
    if not forecast_pattern_refresh or not board_top:
        return False
    if forecast_pattern_refresh.get("artifact_status") != "present":
        return False
    if _forecast_pattern_latest_forecast_not_range(forecast_pattern_refresh, board_top):
        return False
    top = forecast_pattern_refresh.get("top_lane")
    if not isinstance(top, dict):
        return False
    if not forecast_pattern_refresh.get("next_contract_prompt"):
        return False
    if top.get("lane_id") and top.get("lane_id") == board_top.get("lane_id"):
        return True
    return (
        str(top.get("pair") or "").upper(),
        str(top.get("direction") or "").upper(),
        str(top.get("strategy_family") or "").upper(),
    ) == (
        str(board_top.get("pair") or "").upper(),
        str(board_top.get("direction") or "").upper(),
        str(board_top.get("strategy_family") or "").upper(),
    )


def _range_rail_geometry_repair_matches_board(
    range_rail_geometry_repair: dict[str, Any],
    board_top: dict[str, Any],
) -> bool:
    if not range_rail_geometry_repair or not board_top:
        return False
    if range_rail_geometry_repair.get("artifact_status") != "present":
        return False
    if _range_rail_latest_forecast_not_range(range_rail_geometry_repair, board_top):
        return False
    top = range_rail_geometry_repair.get("top_lane")
    if not isinstance(top, dict):
        return False
    if not range_rail_geometry_repair.get("next_contract_prompt"):
        return False
    if top.get("lane_id") and top.get("lane_id") == board_top.get("lane_id"):
        return True
    return (
        str(top.get("pair") or "").upper(),
        str(top.get("direction") or "").upper(),
        str(top.get("strategy_family") or "").upper(),
    ) == (
        str(board_top.get("pair") or "").upper(),
        str(board_top.get("direction") or "").upper(),
        str(board_top.get("strategy_family") or "").upper(),
    )


def _artifact_top_matches_board(artifact: dict[str, Any], board_top: dict[str, Any]) -> bool:
    top = artifact.get("top_lane") if isinstance(artifact.get("top_lane"), dict) else {}
    if not top or not board_top:
        return False
    if top.get("lane_id") and top.get("lane_id") == board_top.get("lane_id"):
        return True
    return _lane_shape(top) == _lane_shape(board_top)


def _forecast_pattern_latest_forecast_not_range(
    forecast_pattern_refresh: dict[str, Any],
    board_top: dict[str, Any],
) -> bool:
    if not _artifact_top_matches_board(forecast_pattern_refresh, board_top):
        return False
    top = forecast_pattern_refresh.get("top_lane")
    top = top if isinstance(top, dict) else {}
    return str(top.get("forecast_box_status") or "").upper() == "LATEST_FORECAST_NOT_RANGE"


def _range_rail_latest_forecast_not_range(
    range_rail_geometry_repair: dict[str, Any],
    board_top: dict[str, Any],
) -> bool:
    if not _artifact_top_matches_board(range_rail_geometry_repair, board_top):
        return False
    top = range_rail_geometry_repair.get("top_lane")
    top = top if isinstance(top, dict) else {}
    return str(top.get("rail_status") or "").upper() == "LATEST_FORECAST_NOT_RANGE"


def _lane_shape(value: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(value.get("pair") or "").strip().upper(),
        str(value.get("direction") or value.get("side") or "").strip().upper(),
        str(value.get("strategy_family") or "").strip().upper(),
    )


def _next_prompt(
    selected_active_path: str,
    remaining_blockers: list[dict[str, Any]],
    *,
    active_opportunity_board: dict[str, Any] | None = None,
    non_eurusd_frontier: dict[str, Any] | None = None,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
    guardian_events: dict[str, Any] | None = None,
) -> str:
    blocker_codes = ", ".join(row["code"] for row in remaining_blockers[:10])
    active_board_shape = _active_board_target_shape(active_opportunity_board)
    frontier_shape = _frontier_target_shape(non_eurusd_frontier)
    board_top = (
        (active_opportunity_board or {}).get("top_lane")
        if isinstance((active_opportunity_board or {}).get("top_lane"), dict)
        else {}
    )
    if selected_active_path == "NO_TRADE_WITH_CAUSE" and _tp_proof_acquisition_route_unreachable(
        board_top
    ):
        next_action = _tp_proof_acquisition_route_rerank_action(board_top)
        return (
            f"{next_action} Keep blockers visible: {blocker_codes}. "
            "Do not request receipts the current gates cannot create, relax RR/chase gates, or infer live permission."
        )
    frontier_suffix = _frontier_supplement_prompt(
        board_top,
        non_eurusd_frontier or {},
        entry_frequency_recovery=entry_frequency_recovery,
        forecast_pattern_refresh=forecast_pattern_refresh,
        range_rail_geometry_repair=range_rail_geometry_repair,
        guardian_events=guardian_events,
    )
    frontier_lane = _frontier_evidence_lane(non_eurusd_frontier or {})
    frontier_consumption = (
        _frontier_artifact_consumption(
            frontier_lane,
            entry_frequency_recovery=entry_frequency_recovery,
            forecast_pattern_refresh=forecast_pattern_refresh,
            range_rail_geometry_repair=range_rail_geometry_repair,
            guardian_events=guardian_events,
        )
        if frontier_lane
        else None
    )
    if (
        selected_active_path == "EVIDENCE_ACQUISITION"
        and _active_board_all_no_trade(active_opportunity_board or {})
        and _frontier_evidence_action_available(non_eurusd_frontier or {})
    ):
        frontier_prompt = _frontier_evidence_prompt(
            non_eurusd_frontier or {},
            entry_frequency_recovery=entry_frequency_recovery,
            forecast_pattern_refresh=forecast_pattern_refresh,
            range_rail_geometry_repair=range_rail_geometry_repair,
            guardian_events=guardian_events,
        )
        return (
            f"{frontier_prompt} "
            f"Keep blockers visible: {blocker_codes}. "
            "This is read-only evidence/tuning work, not live permission."
        )
    guardian_prompt = _guardian_range_rail_trigger_prompt(guardian_events or {}, board_top)
    if selected_active_path == "EVIDENCE_ACQUISITION" and guardian_prompt:
        return (
            f"{guardian_prompt}{frontier_suffix} Keep blockers visible: {blocker_codes}. "
            "This is read-only evidence/tuning work, not live permission."
        )
    if (
        selected_active_path == "EVIDENCE_ACQUISITION"
        and _range_rail_latest_forecast_not_range(range_rail_geometry_repair or {}, board_top)
        and _frontier_evidence_action_available(non_eurusd_frontier or {})
    ):
        frontier_prompt = _frontier_evidence_prompt(
            non_eurusd_frontier or {},
            entry_frequency_recovery=entry_frequency_recovery,
            forecast_pattern_refresh=forecast_pattern_refresh,
            range_rail_geometry_repair=range_rail_geometry_repair,
            guardian_events=guardian_events,
        )
        return (
            "The board-selected range-rail path has been consumed, but the latest forecast is no longer RANGE; "
            "do not repeat range-box refresh for that invalidated lane. "
            f"{frontier_prompt} "
            f"Keep board blockers visible: {blocker_codes}. "
            "This is read-only evidence/tuning work, not live permission."
        )
    if (
        selected_active_path == "EVIDENCE_ACQUISITION"
        and frontier_consumption
        and _frontier_parallel_board_evidence(board_top, non_eurusd_frontier or {})
    ):
        artifact_name, artifact_prompt = frontier_consumption
        return (
            "Advance non_eurusd_live_grade_frontier evidence as the current next work. "
            f"Consume {artifact_name} artifact for {frontier_lane.get('lane_id')}: {artifact_prompt} "
            f"Keep active board lane {board_top.get('lane_id')} visible with blockers: {blocker_codes}. "
            f"{_frontier_blocker_fragment(frontier_lane, label='Non-EUR frontier blockers')}"
            "This is read-only evidence/tuning work, not live permission."
        )
    if (
        selected_active_path == "EVIDENCE_ACQUISITION"
        and _range_rail_geometry_repair_matches_board(range_rail_geometry_repair or {}, board_top)
    ):
        rail_prompt = str((range_rail_geometry_repair or {}).get("next_contract_prompt") or "").strip()
        if rail_prompt:
            return (
                f"{rail_prompt}{frontier_suffix} Keep blockers visible: {blocker_codes}. "
                "This is read-only evidence/tuning work, not live permission."
            )
    if (
        selected_active_path == "EVIDENCE_ACQUISITION"
        and _forecast_pattern_refresh_matches_board(forecast_pattern_refresh or {}, board_top)
    ):
        pattern_prompt = str((forecast_pattern_refresh or {}).get("next_contract_prompt") or "").strip()
        if pattern_prompt:
            return (
                f"{pattern_prompt}{frontier_suffix} Keep blockers visible: {blocker_codes}. "
                "This is read-only evidence/tuning work, not live permission."
            )
    if (
        selected_active_path == "EVIDENCE_ACQUISITION"
        and _entry_frequency_recovery_matches_board(entry_frequency_recovery or {}, board_top)
    ):
        recovery_prompt = str((entry_frequency_recovery or {}).get("next_contract_prompt") or "").strip()
        if recovery_prompt:
            return (
                f"{recovery_prompt}{frontier_suffix} Keep blockers visible: {blocker_codes}. "
                "This is read-only evidence/tuning work, not live permission."
            )
    if _active_board_all_no_trade(active_opportunity_board or {}):
        target_shape = frontier_shape or active_board_shape or TARGET_SHAPE
    elif (
        _frontier_supplements_board_evidence(board_top, non_eurusd_frontier or {})
        or _frontier_parallel_board_evidence(board_top, non_eurusd_frontier or {})
    ) and frontier_shape:
        target_shape = f"{active_board_shape or TARGET_SHAPE} plus frontier evidence {frontier_shape}"
    else:
        target_shape = active_board_shape or TARGET_SHAPE
    return (
        f"Implement {selected_active_path} for {target_shape} as read-only work. "
        f"{frontier_suffix}"
        "Do not send, cancel, close, mutate broker state, relax gates, or infer operator approval. "
        f"Keep blockers visible: {blocker_codes}."
    )


def _active_board_target_shape(active_opportunity_board: dict[str, Any] | None) -> str | None:
    board = active_opportunity_board if isinstance(active_opportunity_board, dict) else {}
    top = board.get("top_lane") if isinstance(board.get("top_lane"), dict) else {}
    pair = str(top.get("pair") or "").strip().upper()
    direction = str(top.get("direction") or top.get("side") or "").strip().upper()
    strategy = str(top.get("strategy_family") or "").strip().upper()
    vehicle = str(top.get("vehicle") or "").strip().upper()
    payoff = str(top.get("payoff_shape") or "").strip().upper()
    parts = [part for part in (pair, direction, strategy, vehicle, payoff) if part]
    return "|".join(parts) if len(parts) >= 4 else None


def _frontier_target_shape(non_eurusd_frontier: dict[str, Any] | None) -> str | None:
    frontier = non_eurusd_frontier if isinstance(non_eurusd_frontier, dict) else {}
    lane = _frontier_evidence_lane(frontier)
    return _lane_target_shape(lane)


def _contract_target_shape(
    *,
    selected_active_path: str | None = None,
    active_opportunity_board: dict[str, Any] | None = None,
    non_eurusd_frontier: dict[str, Any] | None = None,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
) -> str:
    frontier_shape = _frontier_target_shape(non_eurusd_frontier)
    if frontier_shape and _frontier_overrides_board_target_shape(
        selected_active_path=selected_active_path,
        active_opportunity_board=active_opportunity_board,
        non_eurusd_frontier=non_eurusd_frontier,
        entry_frequency_recovery=entry_frequency_recovery,
        forecast_pattern_refresh=forecast_pattern_refresh,
        range_rail_geometry_repair=range_rail_geometry_repair,
    ):
        return frontier_shape
    active_shape = _active_board_target_shape(active_opportunity_board)
    frontier = non_eurusd_frontier if isinstance(non_eurusd_frontier, dict) else {}
    board = active_opportunity_board if isinstance(active_opportunity_board, dict) else {}
    board_top = board.get("top_lane") if isinstance(board.get("top_lane"), dict) else {}
    if active_shape and frontier_shape and _frontier_parallel_board_evidence(board_top, frontier):
        return f"{active_shape} plus frontier evidence {frontier_shape}"
    return active_shape or frontier_shape or TARGET_SHAPE


def _frontier_overrides_board_target_shape(
    *,
    selected_active_path: str | None,
    active_opportunity_board: dict[str, Any] | None,
    non_eurusd_frontier: dict[str, Any] | None,
    entry_frequency_recovery: dict[str, Any] | None = None,
    forecast_pattern_refresh: dict[str, Any] | None = None,
    range_rail_geometry_repair: dict[str, Any] | None = None,
) -> bool:
    if selected_active_path != "EVIDENCE_ACQUISITION":
        return False
    frontier = non_eurusd_frontier if isinstance(non_eurusd_frontier, dict) else {}
    if not _frontier_evidence_action_available(frontier):
        return False
    board = active_opportunity_board if isinstance(active_opportunity_board, dict) else {}
    board_top = board.get("top_lane") if isinstance(board.get("top_lane"), dict) else {}
    if _active_board_all_no_trade(board):
        return True
    if _range_rail_latest_forecast_not_range(range_rail_geometry_repair or {}, board_top):
        return True
    frontier_lane = _frontier_evidence_lane(frontier)
    return bool(
        frontier_lane
        and _frontier_parallel_board_evidence(board_top, frontier)
        and _frontier_artifact_consumption(
            frontier_lane,
            entry_frequency_recovery=entry_frequency_recovery,
            forecast_pattern_refresh=forecast_pattern_refresh,
            range_rail_geometry_repair=range_rail_geometry_repair,
        )
    )


def _render_report(payload: dict[str, Any]) -> str:
    blockers = payload.get("remaining_blockers") or []
    current = payload.get("current_state") if isinstance(payload.get("current_state"), dict) else {}
    replay = current.get("limit_s5_bidask_replay") if isinstance(current.get("limit_s5_bidask_replay"), dict) else {}
    mining = current.get("limit_sample_mining") if isinstance(current.get("limit_sample_mining"), dict) else {}
    active_board = (
        current.get("active_opportunity_board")
        if isinstance(current.get("active_opportunity_board"), dict)
        else {}
    )
    non_eurusd_frontier = (
        current.get("non_eurusd_live_grade_frontier")
        if isinstance(current.get("non_eurusd_live_grade_frontier"), dict)
        else {}
    )
    board_top = active_board.get("top_lane") if isinstance(active_board.get("top_lane"), dict) else {}
    frontier_lane = _frontier_evidence_lane(non_eurusd_frontier)
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
        f"- Why no board rerank: {no_action.get('why_no_board_rerank')}",
        "",
        "## Active Opportunity Board",
        "",
        f"- Artifact status: `{active_board.get('artifact_status')}`",
        f"- Generated at: `{active_board.get('generated_at_utc')}`",
        f"- Lanes scanned: `{active_board.get('total_lanes')}`",
        f"- Top lane: `{board_top.get('lane_id')}` / `{board_top.get('status')}`",
        f"- Failed exact replay consumed count: `{active_board.get('failed_exact_replay_consumed_count')}`",
        f"- STOP HARVEST failed replay consumed: `{active_board.get('stop_harvest_failed_replay_consumed')}`",
        "",
        "## Non-EUR/USD Frontier",
        "",
        f"- Artifact status: `{non_eurusd_frontier.get('artifact_status')}`",
        f"- Status: `{non_eurusd_frontier.get('status')}`",
        f"- Scanned intents: `{non_eurusd_frontier.get('scanned_intents')}`",
        f"- Scanned pairs: `{non_eurusd_frontier.get('scanned_pairs_count')}`",
        f"- Next evidence lane: `{frontier_lane.get('lane_id')}`",
        f"- Next frontier path: {non_eurusd_frontier.get('next_active_path')}",
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
        "## Exact LIMIT Sample Mining",
        "",
        f"- Mining status: `{mining.get('status')}`",
        f"- Current samples: `{mining.get('current_replayed_exact_limit_samples')}`",
        f"- Additional acceptable local samples: `{mining.get('additional_acceptable_local_samples_found')}`",
        f"- Remaining exact LIMIT gap: `{mining.get('remaining_exact_limit_samples')}`",
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

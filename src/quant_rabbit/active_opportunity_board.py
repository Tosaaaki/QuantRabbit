from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
    DEFAULT_ACTIVE_OPPORTUNITY_BOARD_REPORT,
    DEFAULT_ACTIVE_TRADER_CONTRACT,
    DEFAULT_AS_LANE_CANDIDATE_BOARD,
    DEFAULT_AS_PROOF_PACK_QUEUE,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
    DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
    DEFAULT_HARVEST_LIVE_GRADE_PATH,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
    DEFAULT_PORTFOLIO_4X_PATH_PLANNER,
    DEFAULT_STRATEGY_PROFILE,
    DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR,
    DEFAULT_VERIFICATION_LEDGER,
    ROOT,
)


BOARD_VERSION = "active_opportunity_board_v1"
LANE_STATUSES = {
    "LIVE_READY",
    "HARVEST_READY",
    "SCOUT_READY",
    "EVIDENCE_ACQUISITION",
    "OPERATOR_REVIEW_REQUIRED",
    "NO_TRADE_WITH_CAUSE",
}

# Read-only repair-priority weights. These do not size trades, relax gates, or
# authorize execution; they only keep the board stable when multiple blocked
# lanes are compared in one artifact.
STATUS_PRIORITY = {
    "LIVE_READY": 600,
    "HARVEST_READY": 520,
    "EVIDENCE_ACQUISITION": 500,
    "SCOUT_READY": 450,
    "OPERATOR_REVIEW_REQUIRED": 360,
    "NO_TRADE_WITH_CAUSE": 0,
}

NEGATIVE_BLOCKER_MARKERS = (
    "NEGATIVE_EXPECTANCY",
    "REPLAY_NEGATIVE",
    "BIDASK_REPLAY_NEGATIVE",
    "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
    "MARKET_CLOSE_LEAK_DOMINATES",
    "MARKET_CLOSE_LEAK_FAMILY_BLOCKED",
)
EVIDENCE_BLOCKER_MARKERS = (
    "SAMPLE_GAP",
    "PROOF_QUEUE_MEMBER_BUT_NOT_PROOF_READY",
    "PROOF_FLOOR",
    "EVIDENCE_GAP",
    "LIMIT_SAMPLE_FLOOR",
    "S5_TOUCH_LAG",
    "REPLAY_MISSING",
    "SPREAD_SLIPPAGE_PROOF_MISSING",
    "LOCAL_LIMIT_SAMPLE_COVERAGE_EXHAUSTED",
)
SPREAD_BLOCKER_MARKERS = ("SPREAD_TOO_WIDE", "TARGET_TOO_THIN_FOR_SPREAD")
RISK_BLOCKER_MARKERS = (
    "BAD_UNITS",
    "LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT",
    "MARGIN_TOO_THIN_FOR_MIN_LOT",
    "MIN_LOT",
    "REWARD_RISK_TOO_LOW",
    "RISK_ENGINE_PASS_MISSING",
    "LIVE_ORDER_GATEWAY_PREFLIGHT_MISSING",
)
OPERATOR_REVIEW_MARKERS = ("OPERATOR", "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED")
GUARDIAN_RECEIPT_OPERATOR_REVIEW_MARKERS = ("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",)
GUARDIAN_MARKERS = ("GUARDIAN",)
CURRENT_INTENT_OWNED_BLOCKERS = (
    "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
)
FAILED_EXACT_REPLAY_MARKERS = (
    "S5_TP_PATH_DOES_NOT_RECONSTRUCT_OBSERVED_TP_FILLS",
    "STOP_S5_TRIGGER_OR_TP_PATH_REPLAY_FAILED",
)


@dataclass(frozen=True)
class ActiveOpportunityBoardSummary:
    status: str
    output_path: Path
    report_path: Path
    top_lane_id: str | None
    live_permission_allowed: bool


class ActiveOpportunityBoard:
    """Build the read-only multi-pair/multi-vehicle active opportunity board."""

    def __init__(
        self,
        *,
        active_trader_contract_path: Path = DEFAULT_ACTIVE_TRADER_CONTRACT,
        trader_goal_loop_path: Path = DEFAULT_TRADER_GOAL_LOOP_ORCHESTRATOR,
        payoff_shape_diagnosis_path: Path = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
        harvest_live_grade_path: Path = DEFAULT_HARVEST_LIVE_GRADE_PATH,
        proof_pack_queue_path: Path = DEFAULT_AS_PROOF_PACK_QUEUE,
        lane_candidate_board_path: Path = DEFAULT_AS_LANE_CANDIDATE_BOARD,
        portfolio_4x_path_planner_path: Path = DEFAULT_PORTFOLIO_4X_PATH_PLANNER,
        live_order_request_path: Path = DEFAULT_LIVE_ORDER_REQUEST,
        broker_snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        verification_ledger_path: Path = DEFAULT_VERIFICATION_LEDGER,
        execution_ledger_db_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
        strategy_profile_path: Path = DEFAULT_STRATEGY_PROFILE,
        guardian_receipt_consumption_path: Path = DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
        guardian_receipt_operator_review_path: Path = DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
        replay_artifact_paths: list[Path] | None = None,
        output_path: Path = DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
        report_path: Path = DEFAULT_ACTIVE_OPPORTUNITY_BOARD_REPORT,
        now_utc: datetime | None = None,
    ) -> None:
        self.paths = {
            "active_trader_contract": active_trader_contract_path,
            "trader_goal_loop_orchestrator": trader_goal_loop_path,
            "payoff_shape_diagnosis": payoff_shape_diagnosis_path,
            "harvest_live_grade_path": harvest_live_grade_path,
            "as_proof_pack_queue": proof_pack_queue_path,
            "as_lane_candidate_board": lane_candidate_board_path,
            "portfolio_4x_path_planner": portfolio_4x_path_planner_path,
            "live_order_request": live_order_request_path,
            "broker_snapshot": broker_snapshot_path,
            "order_intents": order_intents_path,
            "verification_ledger": verification_ledger_path,
            "strategy_profile": strategy_profile_path,
            "guardian_receipt_consumption": guardian_receipt_consumption_path,
            "guardian_receipt_operator_review": guardian_receipt_operator_review_path,
        }
        self.execution_ledger_db_path = execution_ledger_db_path
        self.replay_artifact_paths = replay_artifact_paths if replay_artifact_paths is not None else _default_replay_artifacts()
        self.output_path = output_path
        self.report_path = report_path
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> ActiveOpportunityBoardSummary:
        payload = self.build_payload()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload), encoding="utf-8")
        top_lane = payload.get("top_lane") if isinstance(payload.get("top_lane"), dict) else {}
        return ActiveOpportunityBoardSummary(
            status=str(payload["status"]),
            output_path=self.output_path,
            report_path=self.report_path,
            top_lane_id=top_lane.get("lane_id"),
            live_permission_allowed=bool(payload["live_permission_allowed"]),
        )

    def build_payload(self) -> dict[str, Any]:
        artifacts = {name: _load_json_artifact(path) for name, path in self.paths.items()}
        replay_artifacts = {
            str(path.relative_to(ROOT) if path.is_absolute() and path.is_relative_to(ROOT) else path): _load_json_artifact(path)
            for path in self.replay_artifact_paths
        }
        lanes: dict[str, dict[str, Any]] = {}

        _add_order_intent_lanes(lanes, artifacts["order_intents"])
        _add_proof_queue_lanes(lanes, artifacts["as_proof_pack_queue"])
        _add_portfolio_lanes(lanes, artifacts["portfolio_4x_path_planner"])
        _add_lane_board_lanes(lanes, artifacts["as_lane_candidate_board"])
        _add_payoff_lanes(lanes, artifacts["payoff_shape_diagnosis"])
        _add_harvest_lanes(lanes, artifacts["harvest_live_grade_path"])
        _add_replay_lanes(lanes, replay_artifacts)
        _attach_strategy_profile(lanes, artifacts["strategy_profile"])
        _attach_verification_ledger(lanes, artifacts["verification_ledger"])

        execution_ledger_summary = _execution_ledger_summary(self.execution_ledger_db_path)
        guardian_routing_clear = _guardian_routing_clear(
            artifacts["guardian_receipt_consumption"],
            artifacts["guardian_receipt_operator_review"],
        )
        guardian_intent_blockers_stale = _guardian_intent_blockers_stale(
            artifacts["guardian_receipt_consumption"],
            artifacts["order_intents"],
        )
        for lane in lanes.values():
            _finalize_lane(
                lane,
                guardian_routing_clear=guardian_routing_clear,
                guardian_intent_blockers_stale=guardian_intent_blockers_stale,
            )

        ranked_lanes = sorted(
            lanes.values(),
            key=lambda lane: (
                -float(lane.get("_rank_score") or 0.0),
                str(lane.get("pair")),
                str(lane.get("direction")),
                str(lane.get("strategy_family")),
                str(lane.get("vehicle")),
            ),
        )
        public_lanes = [_public_lane(lane) for lane in ranked_lanes]
        top_lane = public_lanes[0] if public_lanes else {}
        coverage_summary = _coverage_summary(public_lanes)
        pair_coverage = _coverage_by(public_lanes, "pair")
        strategy_family_coverage = _coverage_by(public_lanes, "strategy_family")
        vehicle_coverage = _coverage_by(public_lanes, "vehicle")
        no_trade_reasons = _no_trade_reasons(public_lanes)
        stale_source_reasons = _stale_source_reasons(public_lanes)
        artifact_index = _artifact_index(artifacts, replay_artifacts, self.execution_ledger_db_path)
        active_contract = artifacts["active_trader_contract"]
        goal_loop = artifacts["trader_goal_loop_orchestrator"]

        payload = {
            "schema_version": BOARD_VERSION,
            "status": _board_status(public_lanes),
            "generated_at_utc": self.now_utc.isoformat(),
            "read_only": True,
            "live_side_effects": [],
            "live_permission_allowed": False,
            "coverage_summary": coverage_summary,
            "ranked_active_lanes": public_lanes,
            "top_lane": top_lane,
            "pair_coverage": pair_coverage,
            "strategy_family_coverage": strategy_family_coverage,
            "vehicle_coverage": vehicle_coverage,
            "no_trade_reasons": no_trade_reasons,
            "stale_source_reasons": stale_source_reasons,
            "next_active_path": _next_active_path(top_lane, active_contract),
            "four_x_progress_hypothesis": _four_x_progress_hypothesis(
                top_lane,
                public_lanes,
                active_contract,
                goal_loop,
            ),
            "root_improvement_target": _root_improvement_target(top_lane, active_contract, goal_loop),
            "expected_edge_improvement": _expected_edge_improvement(top_lane, active_contract, goal_loop),
            "do_not_do": _do_not_do(),
            "artifact_index": artifact_index,
            "execution_ledger_summary": execution_ledger_summary,
            "global_safety": {
                "proof_queue_count_zero_is_live_permission": False,
                "operator_decision_inference_allowed": False,
                "broker_state_mutation_allowed": False,
                "live_permission_allowed": False,
                "guardian_receipt_normal_routing_allowed": guardian_routing_clear,
            },
        }
        if payload["live_permission_allowed"]:
            raise ValueError("active opportunity board must never grant live permission")
        if any(lane.get("live_permission_allowed") for lane in public_lanes):
            raise ValueError("active opportunity board lane must never grant live permission")
        return payload


def _default_replay_artifacts() -> list[Path]:
    patterns = ("*replay*.json", "*proof*.json")
    paths: list[Path] = []
    data_dir = ROOT / "data"
    for pattern in patterns:
        for path in data_dir.glob(pattern):
            if path.name in {"as_proof_pack_queue.json"}:
                continue
            if path not in paths:
                paths.append(path)
    return sorted(paths)


def _load_json_artifact(path: Path) -> dict[str, Any]:
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


def _artifact_index(
    artifacts: dict[str, dict[str, Any]],
    replay_artifacts: dict[str, dict[str, Any]],
    execution_ledger_db_path: Path,
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for name, artifact in artifacts.items():
        entries[name] = _artifact_entry(artifact)
    entries["execution_ledger_db"] = {
        "path": str(execution_ledger_db_path),
        "status": "present" if execution_ledger_db_path.exists() else "missing",
        "size_bytes": execution_ledger_db_path.stat().st_size if execution_ledger_db_path.exists() else None,
    }
    entries["replay_artifacts"] = {name: _artifact_entry(artifact) for name, artifact in replay_artifacts.items()}
    combined = hashlib.sha256(json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {"artifacts": entries, "combined_sha256": combined}


def _artifact_entry(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": artifact.get("_path"),
        "status": artifact.get("_artifact_status"),
        "sha256": artifact.get("_sha256"),
        "generated_at_utc": artifact.get("generated_at_utc") or artifact.get("fetched_at_utc"),
    }


def _add_order_intent_lanes(lanes: dict[str, dict[str, Any]], artifact: dict[str, Any]) -> None:
    for row in _list(artifact.get("results")):
        intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
        parsed = _parse_lane_id(str(row.get("lane_id") or ""))
        pair = _first_str(intent.get("pair"), row.get("pair"), parsed.get("pair"), "UNKNOWN")
        direction = _first_str(intent.get("side"), intent.get("direction"), row.get("side"), parsed.get("direction"), "UNKNOWN")
        strategy = _first_str(intent.get("method"), intent.get("strategy_family"), row.get("method"), parsed.get("strategy_family"), "UNKNOWN")
        vehicle = _normalize_vehicle(_first_str(intent.get("order_type"), row.get("order_type"), parsed.get("vehicle"), "UNKNOWN"))
        lane_id = str(row.get("lane_id") or _synthetic_lane_id("intent", pair, direction, strategy, vehicle))
        lane = _ensure_lane(lanes, lane_id, pair, direction, strategy, vehicle)
        lane["source_refs"].append("data/order_intents.json")
        lane["order_intent_status"] = row.get("status")
        lane["risk_allowed"] = bool(row.get("risk_allowed"))
        lane["payoff_shape"] = _payoff_shape_from_intent(intent)
        metrics = row.get("risk_metrics") if isinstance(row.get("risk_metrics"), dict) else {}
        lane["expected_edge_jpy"] = _first_number(
            lane.get("expected_edge_jpy"),
            metrics.get("expected_edge_jpy"),
            metrics.get("expectancy_jpy"),
            _metadata(intent).get("capture_take_profit_expectancy_jpy"),
        )
        lane["spread_pips"] = _first_number(lane.get("spread_pips"), metrics.get("spread_pips"))
        lane["risk_jpy"] = _first_number(lane.get("risk_jpy"), metrics.get("risk_jpy"))
        lane["reward_jpy"] = _first_number(lane.get("reward_jpy"), metrics.get("reward_jpy"))
        lane["units"] = _first_number(lane.get("units"), intent.get("units"))
        intent_blockers = _string_list(row.get("live_blocker_codes"))
        risk_issue_codes = _issue_codes(row.get("risk_issues"))
        strategy_issue_codes = _issue_codes(row.get("strategy_issues"))
        lane["blockers"].extend(intent_blockers)
        lane["blockers"].extend(risk_issue_codes)
        lane["blockers"].extend(strategy_issue_codes)
        lane["order_intent_blockers"].extend(intent_blockers)
        lane["order_intent_blockers"].extend(risk_issue_codes)
        lane["order_intent_blockers"].extend(strategy_issue_codes)
        lane["strategy_issue_codes"].extend(strategy_issue_codes)
        lane["risk_issue_codes"].extend(risk_issue_codes)


def _add_proof_queue_lanes(lanes: dict[str, dict[str, Any]], artifact: dict[str, Any]) -> None:
    for key, source_status in (("queue", "proof_queue"), ("rejected_candidates", "proof_rejected")):
        for row in _list(artifact.get(key)):
            pair, direction, strategy, vehicle, payoff = _fields_from_row(row)
            lane = _ensure_lane(
                lanes,
                str(row.get("lane_id") or _synthetic_lane_id(source_status, pair, direction, strategy, vehicle)),
                pair,
                direction,
                strategy,
                vehicle,
            )
            lane["source_refs"].append(f"data/as_proof_pack_queue.json:{key}")
            lane["proof_status"] = _first_str(row.get("proof_classification"), lane.get("proof_status"), "UNKNOWN")
            lane["can_enter_proof_pack"] = bool(row.get("can_enter_proof_pack"))
            lane["can_create_live_permission"] = bool(row.get("can_create_live_permission"))
            lane["proof_distance"] = _first_number(lane.get("proof_distance"), row.get("proof_distance"))
            lane["payoff_shape"] = _first_str(payoff, lane.get("payoff_shape"), "UNKNOWN")
            lane["expected_edge_jpy"] = _first_number(lane.get("expected_edge_jpy"), row.get("expected_jpy_per_trade"))
            lane["blockers"].extend(_string_list(row.get("current_blockers")))
            if source_status == "proof_rejected":
                lane["blockers"].extend(_string_list(row.get("rejection_reasons")))


def _add_portfolio_lanes(lanes: dict[str, dict[str, Any]], artifact: dict[str, Any]) -> None:
    for row in _list(artifact.get("candidate_rankings")):
        pair, direction, strategy, vehicle, payoff = _fields_from_row(row)
        lane = _ensure_lane(
            lanes,
            str(row.get("lane_id") or _synthetic_lane_id("portfolio", pair, direction, strategy, vehicle)),
            pair,
            direction,
            strategy,
            vehicle,
        )
        lane["source_refs"].append("data/portfolio_4x_path_planner.json:candidate_rankings")
        lane["portfolio_rank_score"] = _first_number(lane.get("portfolio_rank_score"), row.get("rank_score"))
        lane["expected_edge_jpy"] = _first_number(lane.get("expected_edge_jpy"), row.get("expected_jpy_per_trade"))
        lane["proof_status"] = _first_str(row.get("proof_classification"), lane.get("proof_status"), "UNKNOWN")
        lane["can_enter_proof_pack"] = bool(row.get("can_enter_proof_pack")) or bool(lane.get("can_enter_proof_pack"))
        lane["can_create_live_permission"] = bool(row.get("can_create_live_permission")) or bool(
            lane.get("can_create_live_permission")
        )
        lane["payoff_shape"] = _first_str(payoff, lane.get("payoff_shape"), "UNKNOWN")
        lane["proof_distance"] = _first_number(lane.get("proof_distance"), row.get("proof_distance"))
        lane["blockers"].extend(_string_list(row.get("current_blockers")))
        lane["blockers"].extend(_string_list(row.get("math_exclusion_reasons")))


def _add_lane_board_lanes(lanes: dict[str, dict[str, Any]], artifact: dict[str, Any]) -> None:
    for key in ("closest_candidate_to_proof_pack",):
        row = artifact.get(key)
        if not isinstance(row, dict):
            continue
        pair, direction, strategy, vehicle, payoff = _fields_from_row(row)
        lane = _ensure_lane(
            lanes,
            str(row.get("lane_id") or _synthetic_lane_id("lane_board", pair, direction, strategy, vehicle)),
            pair,
            direction,
            strategy,
            vehicle,
        )
        lane["source_refs"].append(f"data/as_lane_candidate_board.json:{key}")
        lane["proof_status"] = _first_str(row.get("proof_classification"), lane.get("proof_status"), "UNKNOWN")
        lane["can_enter_proof_pack"] = bool(row.get("can_enter_proof_pack")) or bool(lane.get("can_enter_proof_pack"))
        lane["can_create_live_permission"] = bool(row.get("can_create_live_permission")) or bool(
            lane.get("can_create_live_permission")
        )
        lane["proof_distance"] = _first_number(lane.get("proof_distance"), row.get("proof_distance"))
        lane["payoff_shape"] = _first_str(payoff, lane.get("payoff_shape"), "UNKNOWN")
        lane["blockers"].extend(_string_list(row.get("current_blockers")))


def _add_payoff_lanes(lanes: dict[str, dict[str, Any]], artifact: dict[str, Any]) -> None:
    for key in ("harvest_candidates", "partial_tp_runner_candidates", "runner_candidates", "no_trade_shapes"):
        for row in _list(artifact.get(key)):
            pair, direction, strategy, vehicle, payoff = _fields_from_row(row)
            target_lane_ids = _target_lane_ids(lanes, row, key, pair, direction, strategy, vehicle)
            for lane_id in target_lane_ids:
                lane_vehicle = lanes.get(lane_id, {}).get("vehicle") or vehicle
                lane = _ensure_lane(lanes, lane_id, pair, direction, strategy, str(lane_vehicle))
                _attach_payoff_row(lane, row, key, payoff)


def _target_lane_ids(
    lanes: dict[str, dict[str, Any]],
    row: dict[str, Any],
    prefix: str,
    pair: str,
    direction: str,
    strategy: str,
    vehicle: str,
) -> list[str]:
    if row.get("lane_id"):
        return [str(row["lane_id"])]
    matching = _matching_lane_ids(lanes, pair, direction, strategy, vehicle=vehicle)
    if matching:
        return matching
    return [_synthetic_lane_id(prefix, pair, direction, strategy, vehicle)]


def _attach_payoff_row(lane: dict[str, Any], row: dict[str, Any], key: str, payoff: str) -> None:
    lane["source_refs"].append(f"data/payoff_shape_diagnosis.json:{key}")
    default_payoff = "RUNNER" if "runner" in key else "HARVEST" if key == "harvest_candidates" else lane.get("payoff_shape")
    lane["payoff_shape"] = _first_str(payoff, default_payoff, "UNKNOWN")
    lane["harvest_classification"] = _first_str(row.get("classification"), lane.get("harvest_classification"), "")
    lane["live_promotion_allowed"] = bool(row.get("live_promotion_allowed")) or bool(lane.get("live_promotion_allowed"))
    lane["expected_edge_jpy"] = _first_number(
        lane.get("expected_edge_jpy"),
        row.get("take_profit_expectancy_jpy"),
        row.get("overall_expectancy_jpy_per_trade"),
        row.get("runner_tail_estimated_jpy"),
    )
    lane["proof_gap_trades"] = _first_number(lane.get("proof_gap_trades"), row.get("proof_gap_trades"))
    month_blocker = row.get("month_scale_blocker") if isinstance(row.get("month_scale_blocker"), dict) else None
    if month_blocker:
        lane["blockers"].append("MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE")
        lane["blockers"].extend(_string_list((month_blocker.get("block_reasons") or {}).keys()))
    if key == "no_trade_shapes":
        reason = row.get("reason_code")
        if reason:
            lane["blockers"].append(str(reason))


def _add_harvest_lanes(lanes: dict[str, dict[str, Any]], artifact: dict[str, Any]) -> None:
    for key in ("ranked_harvest_candidates", "closest_harvest_candidate"):
        rows = artifact.get(key)
        if isinstance(rows, dict):
            rows = [rows]
        for row in _list(rows):
            pair, direction, strategy, vehicle, payoff = _fields_from_row(row)
            current_intent = row.get("current_intent_best") if isinstance(row.get("current_intent_best"), dict) else {}
            target_lane_ids = [str(row["lane_id"])] if row.get("lane_id") else []
            if not target_lane_ids and current_intent.get("lane_id"):
                target_lane_ids = [str(current_intent["lane_id"])]
            if not target_lane_ids:
                target_lane_ids = _preferred_harvest_lane_ids(lanes, pair, direction, strategy, vehicle=vehicle)
            if not target_lane_ids:
                target_lane_ids = [_synthetic_lane_id("harvest", pair, direction, strategy, vehicle)]
            for lane_id in target_lane_ids:
                lane_vehicle = vehicle if vehicle != "UNKNOWN" else lanes.get(lane_id, {}).get("vehicle", vehicle)
                lane = _ensure_lane(lanes, lane_id, pair, direction, strategy, str(lane_vehicle))
                _attach_harvest_row(lane, row, key, payoff)


def _attach_harvest_row(lane: dict[str, Any], row: dict[str, Any], key: str, payoff: str) -> None:
    lane["source_refs"].append(f"data/harvest_live_grade_path.json:{key}")
    lane["payoff_shape"] = _first_str(payoff, "HARVEST", lane.get("payoff_shape"), "UNKNOWN")
    lane["harvest_classification"] = _first_str(row.get("classification"), lane.get("harvest_classification"), "")
    lane["live_promotion_allowed"] = bool(row.get("live_promotion_allowed")) or bool(lane.get("live_promotion_allowed"))
    lane["can_create_live_permission"] = bool(row.get("can_create_live_permission")) or bool(
        lane.get("can_create_live_permission")
    )
    lane["can_enter_proof_pack"] = (
        bool(row.get("planner_can_enter_proof_pack"))
        or bool(row.get("actual_proof_queue_member"))
        or bool(lane.get("can_enter_proof_pack"))
    )
    lane["proof_gap_trades"] = _first_number(lane.get("proof_gap_trades"), row.get("proof_gap_trades"))
    lane["portfolio_rank_score"] = _first_number(lane.get("portfolio_rank_score"), row.get("rank_score"))
    tp_proof = row.get("tp_proof") if isinstance(row.get("tp_proof"), dict) else {}
    lane["expected_edge_jpy"] = _first_number(
        lane.get("expected_edge_jpy"),
        tp_proof.get("take_profit_expectancy_jpy"),
    )
    lane["blockers"].extend(_string_list(row.get("promotion_blockers")))


def _preferred_harvest_lane_ids(
    lanes: dict[str, dict[str, Any]],
    pair: str,
    direction: str,
    strategy: str,
    *,
    vehicle: str | None = None,
) -> list[str]:
    matching = _matching_lane_ids(lanes, pair, direction, strategy, vehicle=vehicle)
    if not matching:
        return []
    proof_pack = [lane_id for lane_id in matching if lanes[lane_id].get("can_enter_proof_pack")]
    if proof_pack:
        return proof_pack
    portfolio = [lane_id for lane_id in matching if lanes[lane_id].get("portfolio_rank_score") is not None]
    if portfolio:
        return portfolio
    return matching


def _add_replay_lanes(lanes: dict[str, dict[str, Any]], artifacts: dict[str, dict[str, Any]]) -> None:
    for source_name, artifact in artifacts.items():
        if artifact.get("_artifact_status") != "present":
            continue
        row = _best_replay_row(artifact)
        pair, direction, strategy, vehicle, payoff = _fields_from_row(row)
        if pair == "UNKNOWN" and direction == "UNKNOWN" and strategy == "UNKNOWN":
            continue
        target_lane_ids = [str(row["lane_id"])] if row.get("lane_id") else _matching_lane_ids(
            lanes,
            pair,
            direction,
            strategy,
            vehicle=vehicle,
        )
        if not target_lane_ids:
            target_lane_ids = [_synthetic_lane_id("replay", pair, direction, strategy, vehicle)]
        for lane_id in target_lane_ids:
            lane = _ensure_lane(lanes, lane_id, pair, direction, strategy, vehicle)
            _attach_replay_artifact(lane, artifact, source_name, payoff)


def _attach_replay_artifact(lane: dict[str, Any], artifact: dict[str, Any], source_name: str, payoff: str) -> None:
    lane["source_refs"].append(source_name)
    lane["payoff_shape"] = _first_str(payoff, lane.get("payoff_shape"), "UNKNOWN")
    lane["replay_status"] = _first_str(
        artifact.get("status"),
        artifact.get("bidask_replay_status"),
        artifact.get("s5_bidask_replay_status"),
        artifact.get("classification"),
        lane.get("replay_status"),
        "UNKNOWN",
    )
    lane["expected_edge_jpy"] = _first_number(
        lane.get("expected_edge_jpy"),
        artifact.get("net_expectancy_after_bidask"),
        artifact.get("net_expectancy_after_bidask_slippage"),
        (artifact.get("exact_shape_replay") or {}).get("expectancy_jpy") if isinstance(artifact.get("exact_shape_replay"), dict) else None,
    )
    lane["replay_sample_count"] = _first_number(
        lane.get("replay_sample_count"),
        artifact.get("replay_sample_count"),
        (artifact.get("exact_shape_replay") or {}).get("sample_count") if isinstance(artifact.get("exact_shape_replay"), dict) else None,
    )
    lane["blockers"].extend(_codes_from_blockers(artifact.get("remaining_blockers")))
    lane["blockers"].extend(_codes_from_blockers(artifact.get("proof_queue_blockers_if_positive")))
    if artifact.get("live_permission_allowed") is True:
        lane["blockers"].append("UNEXPECTED_REPLAY_ARTIFACT_LIVE_PERMISSION_TRUE")


def _attach_strategy_profile(lanes: dict[str, dict[str, Any]], artifact: dict[str, Any]) -> None:
    profiles = _list(artifact.get("profiles"))
    profile_keys = {
        (str(row.get("pair")), str(row.get("direction")), str(row.get("method") or "UNKNOWN")): row
        for row in profiles
        if isinstance(row, dict)
    }
    for lane in lanes.values():
        key = (str(lane.get("pair")), str(lane.get("direction")), str(lane.get("strategy_family")))
        profile = profile_keys.get(key)
        if profile:
            lane["strategy_profile_status"] = "PROFILE_PRESENT"
            lane["source_refs"].append("data/strategy_profile.json")
            blocker_code = _first_str(profile.get("blocker_code"), profile.get("reason_code"))
            if blocker_code:
                lane["blockers"].append(blocker_code)
            elif (_float(profile.get("order_blocked")) or 0.0) > 0.0:
                lane["blockers"].append("STRATEGY_PROFILE_ORDER_BLOCKED_COUNT")
        elif artifact.get("_artifact_status") == "present":
            lane["strategy_profile_status"] = lane.get("strategy_profile_status") or "PROFILE_NOT_MATCHED"


def _attach_verification_ledger(lanes: dict[str, dict[str, Any]], artifact: dict[str, Any]) -> None:
    for row in _list(artifact.get("blocking_evidence")) + _list(artifact.get("learning_evidence")):
        subject_id = row.get("subject_id")
        if not subject_id or str(subject_id) not in lanes:
            continue
        lane = lanes[str(subject_id)]
        lane["source_refs"].append("data/verification_ledger.json")
        check_name = row.get("check_name")
        if check_name:
            lane["blockers"].append(str(check_name))


def _best_replay_row(artifact: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in ("target_shape", "required_shape", "requested_shape", "stale_candidate_shape"):
        value = artifact.get(key)
        if isinstance(value, str):
            row.update(_parse_shape(value))
            break
        if isinstance(value, dict):
            row.update(value)
            break
    for key in ("lane_id",):
        if artifact.get(key):
            row[key] = artifact.get(key)
    for nested_key in ("current_queue_context", "as_candidate"):
        nested = artifact.get(nested_key)
        if isinstance(nested, dict):
            row = {**row, **nested}
            break
    if not row and isinstance(artifact.get("current_order_intents"), list) and artifact["current_order_intents"]:
        first = artifact["current_order_intents"][0]
        if isinstance(first, dict):
            row.update(first)
    return row


def _ensure_lane(
    lanes: dict[str, dict[str, Any]],
    lane_id: str,
    pair: str,
    direction: str,
    strategy_family: str,
    vehicle: str,
) -> dict[str, Any]:
    lane_id = lane_id or _synthetic_lane_id("unknown", pair, direction, strategy_family, vehicle)
    if lane_id not in lanes:
        lanes[lane_id] = {
            "lane_id": lane_id,
            "pair": pair or "UNKNOWN",
            "direction": direction or "UNKNOWN",
            "strategy_family": strategy_family or "UNKNOWN",
            "vehicle": _normalize_vehicle(vehicle or "UNKNOWN"),
            "payoff_shape": "UNKNOWN",
            "status": "NO_TRADE_WITH_CAUSE",
            "expected_edge_jpy": None,
            "proof_status": "UNKNOWN",
            "replay_status": "UNKNOWN",
            "spread_status": "UNKNOWN",
            "risk_status": "UNKNOWN",
            "guardian_status": "UNKNOWN",
            "operator_review_status": "UNKNOWN",
            "live_permission_allowed": False,
            "next_action": "",
            "blockers": [],
            "source_refs": [],
            "risk_issue_codes": [],
            "strategy_issue_codes": [],
            "order_intent_blockers": [],
            "stale_source_blockers": [],
        }
    lane = lanes[lane_id]
    for key, value in (
        ("pair", pair),
        ("direction", direction),
        ("strategy_family", strategy_family),
        ("vehicle", _normalize_vehicle(vehicle)),
    ):
        if lane.get(key) in (None, "", "UNKNOWN") and value:
            lane[key] = value
    return lane


def _matching_lane_ids(
    lanes: dict[str, dict[str, Any]],
    pair: str,
    direction: str,
    strategy: str,
    *,
    vehicle: str | None = None,
) -> list[str]:
    return [
        lane_id
        for lane_id, lane in lanes.items()
        if lane.get("pair") == pair
        and lane.get("direction") == direction
        and lane.get("strategy_family") == strategy
        and lane.get("vehicle") != "UNKNOWN"
        and (vehicle is None or vehicle == "UNKNOWN" or lane.get("vehicle") == vehicle)
    ]


def _finalize_lane(
    lane: dict[str, Any],
    *,
    guardian_routing_clear: bool,
    guardian_intent_blockers_stale: bool,
) -> None:
    lane["blockers"] = _unique(_string_list(lane.get("blockers")))
    lane["source_refs"] = _unique(_string_list(lane.get("source_refs")))
    lane["risk_issue_codes"] = _unique(_string_list(lane.get("risk_issue_codes")))
    lane["strategy_issue_codes"] = _unique(_string_list(lane.get("strategy_issue_codes")))
    lane["order_intent_blockers"] = _unique(_string_list(lane.get("order_intent_blockers")))
    _suppress_stale_guardian_receipt_blockers(
        lane,
        guardian_routing_clear=guardian_routing_clear,
        guardian_intent_blockers_stale=guardian_intent_blockers_stale,
    )
    _suppress_stale_current_intent_owned_blockers(lane)
    lane["spread_status"] = _spread_status(lane)
    lane["risk_status"] = _risk_status(lane)
    lane["guardian_status"] = _marker_status(lane["blockers"], GUARDIAN_MARKERS, "BLOCKED", "NOT_BLOCKED")
    lane["operator_review_status"] = _marker_status(lane["blockers"], OPERATOR_REVIEW_MARKERS, "REQUIRED", "NOT_REQUIRED")
    lane["proof_status"] = _computed_proof_status(lane)
    lane["replay_status"] = _computed_replay_status(lane)
    lane["status"] = _classify_lane(lane)
    if lane["status"] == "NO_TRADE_WITH_CAUSE" and not lane["blockers"]:
        lane["blockers"].append("NO_CURRENT_EXECUTABLE_INTENT")
    lane["next_action"] = _lane_next_action(lane)
    lane["_rank_score"] = _rank_score(lane)


def _guardian_routing_clear(consumption: dict[str, Any], operator_review: dict[str, Any]) -> bool:
    if consumption.get("_artifact_status") != "present" or operator_review.get("_artifact_status") != "present":
        return False
    return consumption.get("normal_routing_allowed") is True and operator_review.get("normal_routing_allowed") is True


def _guardian_intent_blockers_stale(consumption: dict[str, Any], order_intents: dict[str, Any]) -> bool:
    if consumption.get("normal_routing_allowed") is not True:
        return False
    consumption_generated = _parse_utc(consumption.get("generated_at_utc"))
    intent_generated = _parse_utc(order_intents.get("generated_at_utc"))
    if consumption_generated is None or intent_generated is None:
        return False
    return intent_generated < consumption_generated


def _suppress_stale_guardian_receipt_blockers(
    lane: dict[str, Any],
    *,
    guardian_routing_clear: bool,
    guardian_intent_blockers_stale: bool,
) -> None:
    if not guardian_routing_clear:
        return
    current_intent_blockers = _string_list(lane.get("order_intent_blockers"))
    if "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED" in current_intent_blockers and not guardian_intent_blockers_stale:
        return
    blockers = _string_list(lane.get("blockers"))
    if "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED" not in blockers:
        return
    lane["blockers"] = [code for code in blockers if code != "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"]
    lane["stale_source_blockers"] = _unique(
        _string_list(lane.get("stale_source_blockers")) + ["GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"]
    )


def _suppress_stale_current_intent_owned_blockers(lane: dict[str, Any]) -> None:
    current_intent_blockers = set(_string_list(lane.get("order_intent_blockers")))
    blockers = _string_list(lane.get("blockers"))
    stale_codes = [
        code
        for code in CURRENT_INTENT_OWNED_BLOCKERS
        if code in blockers and code not in current_intent_blockers
    ]
    if not stale_codes:
        return
    lane["blockers"] = [code for code in blockers if code not in stale_codes]
    lane["stale_source_blockers"] = _unique(_string_list(lane.get("stale_source_blockers")) + stale_codes)


def _classify_lane(lane: dict[str, Any]) -> str:
    blockers = lane.get("blockers") or []
    has_negative = _has_marker(blockers, NEGATIVE_BLOCKER_MARKERS)
    has_evidence = _has_evidence_path(lane)
    if lane.get("order_intent_status") == "LIVE_READY" and lane.get("risk_allowed") and not blockers:
        return "LIVE_READY"
    if _has_marker(blockers, FAILED_EXACT_REPLAY_MARKERS):
        return "NO_TRADE_WITH_CAUSE"
    if _has_marker(blockers, GUARDIAN_RECEIPT_OPERATOR_REVIEW_MARKERS):
        return "OPERATOR_REVIEW_REQUIRED"
    if has_negative:
        return "NO_TRADE_WITH_CAUSE"
    if _has_marker(blockers, OPERATOR_REVIEW_MARKERS):
        return "OPERATOR_REVIEW_REQUIRED"
    if _is_scout_ready_candidate(lane):
        return "SCOUT_READY"
    if has_evidence:
        return "EVIDENCE_ACQUISITION"
    if _is_harvest_ready_candidate(lane):
        return "HARVEST_READY"
    return "NO_TRADE_WITH_CAUSE"


def _has_evidence_path(lane: dict[str, Any]) -> bool:
    blockers = lane.get("blockers") or []
    proof_status = str(lane.get("proof_status") or "")
    replay_status = str(lane.get("replay_status") or "")
    if lane.get("can_enter_proof_pack") and not _has_marker(blockers, NEGATIVE_BLOCKER_MARKERS):
        return True
    if "EVIDENCE_GAP" in proof_status:
        return True
    if _has_marker(blockers, EVIDENCE_BLOCKER_MARKERS):
        return True
    if "PASSED_STILL_BLOCKED" in replay_status or "MISSING" in replay_status:
        return True
    proof_gap = _float(lane.get("proof_gap_trades"))
    if proof_gap is not None and proof_gap <= 3:
        return True
    return False


def _is_scout_ready_candidate(lane: dict[str, Any]) -> bool:
    blockers = lane.get("blockers") or []
    if blockers:
        return False
    risk_status = str(lane.get("risk_status") or "")
    return "MIN_LOT_FEASIBLE" in risk_status and lane.get("risk_jpy") is not None


def _is_harvest_ready_candidate(lane: dict[str, Any]) -> bool:
    blockers = lane.get("blockers") or []
    if _has_marker(blockers, NEGATIVE_BLOCKER_MARKERS):
        return False
    classification = str(lane.get("harvest_classification") or "")
    payoff = str(lane.get("payoff_shape") or "")
    return "HARVEST" in payoff and ("TP_PROVEN" in classification or "PROOF_FLOOR" in classification)


def _rank_score(lane: dict[str, Any]) -> float:
    score = float(STATUS_PRIORITY.get(str(lane.get("status")), 0))
    score += _float(lane.get("portfolio_rank_score")) or 0.0
    edge = _float(lane.get("expected_edge_jpy"))
    if edge is not None:
        score += max(min(edge / 100.0, 50.0), -80.0)
    proof_gap = _float(lane.get("proof_gap_trades"))
    if proof_gap is not None:
        score += max(0.0, 30.0 - proof_gap)
    if lane.get("can_enter_proof_pack"):
        score += 25.0
    if lane.get("vehicle") == "LIMIT" and "LIMIT" in str(lane.get("replay_status")):
        score += 15.0
    if lane.get("vehicle") == "UNKNOWN":
        score -= 260.0
    if _has_marker(lane.get("blockers") or [], NEGATIVE_BLOCKER_MARKERS):
        score -= 160.0
    if "NO_CURRENT_EXECUTABLE_INTENT" in (lane.get("blockers") or []):
        score -= 220.0
    score -= min(len(lane.get("blockers") or []) * 4.0, 80.0)
    return round(score, 4)


def _spread_status(lane: dict[str, Any]) -> str:
    if _has_marker(lane.get("blockers") or [], SPREAD_BLOCKER_MARKERS):
        return "BLOCKED"
    if lane.get("spread_pips") is not None:
        return "OBSERVED_NOT_BLOCKED"
    return "UNKNOWN"


def _risk_status(lane: dict[str, Any]) -> str:
    blockers = lane.get("blockers") or []
    if _has_marker(blockers, RISK_BLOCKER_MARKERS):
        return "BLOCKED"
    if lane.get("risk_allowed"):
        return "ALLOWED_BY_DRY_RUN"
    if lane.get("units") and lane.get("risk_jpy") is not None:
        return "MIN_LOT_FEASIBLE_RISK_NOT_LIVE_READY"
    return "UNKNOWN"


def _computed_proof_status(lane: dict[str, Any]) -> str:
    status = str(lane.get("proof_status") or "UNKNOWN")
    if lane.get("can_enter_proof_pack") and status == "UNKNOWN":
        status = "EVIDENCE_GAP"
    proof_gap = lane.get("proof_gap_trades")
    if proof_gap is not None:
        status = f"{status};proof_gap={proof_gap}"
    return status


def _computed_replay_status(lane: dict[str, Any]) -> str:
    status = str(lane.get("replay_status") or "UNKNOWN")
    blockers = lane.get("blockers") or []
    if _has_marker(blockers, ("BIDASK_REPLAY_NEGATIVE",)):
        return "NEGATIVE"
    return status


def _lane_next_action(lane: dict[str, Any]) -> str:
    status = lane.get("status")
    lane_key = _lane_key(lane)
    if _has_marker(lane.get("blockers") or [], FAILED_EXACT_REPLAY_MARKERS):
        return (
            f"No trade for {lane_key}; consume the failed exact replay as not SCOUT-ready, "
            "wait for new independent trigger/TP-path evidence, and do not repeat the same exact replay."
        )
    if status == "LIVE_READY":
        return f"Keep {lane_key} visible for verifier/gateway checks only; this board grants no live permission."
    if status == "EVIDENCE_ACQUISITION":
        return f"Acquire or canonicalize proof/replay evidence for {lane_key}; do not send or mix vehicles."
    if status == "SCOUT_READY":
        return f"Prepare read-only SCOUT judgement material for {lane_key}; gateway permission remains false."
    if status == "HARVEST_READY":
        return f"Run HARVEST readiness checks for {lane_key} while preserving negative/replay blockers."
    if status == "OPERATOR_REVIEW_REQUIRED":
        if _has_marker(lane.get("blockers") or [], GUARDIAN_RECEIPT_OPERATOR_REVIEW_MARKERS):
            return f"Package guardian receipt operator-review evidence for {lane_key}; do not infer approval."
        return f"Package operator/guardian review evidence for {lane_key}; do not infer approval."
    return f"No trade for {lane_key}; preserve blocker cause and compare another pair/vehicle."


def _public_lane(lane: dict[str, Any]) -> dict[str, Any]:
    public = {
        "lane_id": str(lane.get("lane_id") or ""),
        "pair": str(lane.get("pair") or "UNKNOWN"),
        "direction": str(lane.get("direction") or "UNKNOWN"),
        "strategy_family": str(lane.get("strategy_family") or "UNKNOWN"),
        "vehicle": str(lane.get("vehicle") or "UNKNOWN"),
        "payoff_shape": str(lane.get("payoff_shape") or "UNKNOWN"),
        "status": str(lane.get("status") or "NO_TRADE_WITH_CAUSE"),
        "expected_edge_jpy": _json_number_or_none(lane.get("expected_edge_jpy")),
        "proof_status": str(lane.get("proof_status") or "UNKNOWN"),
        "replay_status": str(lane.get("replay_status") or "UNKNOWN"),
        "spread_status": str(lane.get("spread_status") or "UNKNOWN"),
        "risk_status": str(lane.get("risk_status") or "UNKNOWN"),
        "guardian_status": str(lane.get("guardian_status") or "UNKNOWN"),
        "operator_review_status": str(lane.get("operator_review_status") or "UNKNOWN"),
        "live_permission_allowed": False,
        "next_action": str(lane.get("next_action") or ""),
        "blockers": _unique(_string_list(lane.get("blockers"))),
        "stale_source_blockers": _unique(_string_list(lane.get("stale_source_blockers"))),
        "rank_score": _json_number_or_none(lane.get("_rank_score")),
        "source_refs": _unique(_string_list(lane.get("source_refs"))),
    }
    if public["status"] not in LANE_STATUSES:
        raise ValueError(f"unknown lane status: {public['status']}")
    return public


def _coverage_summary(lanes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs_scanned": sorted({lane["pair"] for lane in lanes if lane.get("pair") != "UNKNOWN"}),
        "strategy_families_scanned": sorted(
            {lane["strategy_family"] for lane in lanes if lane.get("strategy_family") != "UNKNOWN"}
        ),
        "vehicles_scanned": sorted({lane["vehicle"] for lane in lanes if lane.get("vehicle") != "UNKNOWN"}),
        "total_lanes": len(lanes),
        "live_ready_count": _count_status(lanes, "LIVE_READY"),
        "harvest_ready_count": _count_status(lanes, "HARVEST_READY"),
        "scout_ready_count": _count_status(lanes, "SCOUT_READY"),
        "evidence_acquisition_count": _count_status(lanes, "EVIDENCE_ACQUISITION"),
        "operator_review_required_count": _count_status(lanes, "OPERATOR_REVIEW_REQUIRED"),
        "no_trade_count": _count_status(lanes, "NO_TRADE_WITH_CAUSE"),
    }


def _coverage_by(lanes: list[dict[str, Any]], field: str) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for lane in lanes:
        key = str(lane.get(field) or "UNKNOWN")
        row = coverage.setdefault(
            key,
            {
                "total_lanes": 0,
                "statuses": {},
                "pairs": set(),
                "strategy_families": set(),
                "vehicles": set(),
                "best_lane_id": None,
                "best_status": None,
            },
        )
        row["total_lanes"] += 1
        row["statuses"][lane["status"]] = row["statuses"].get(lane["status"], 0) + 1
        row["pairs"].add(lane["pair"])
        row["strategy_families"].add(lane["strategy_family"])
        row["vehicles"].add(lane["vehicle"])
        if row["best_lane_id"] is None:
            row["best_lane_id"] = lane["lane_id"]
            row["best_status"] = lane["status"]
    return {
        key: {
            **{k: v for k, v in row.items() if k not in {"pairs", "strategy_families", "vehicles"}},
            "pairs": sorted(row["pairs"]),
            "strategy_families": sorted(row["strategy_families"]),
            "vehicles": sorted(row["vehicles"]),
        }
        for key, row in sorted(coverage.items())
    }


def _no_trade_reasons(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for lane in lanes:
        if lane.get("status") != "NO_TRADE_WITH_CAUSE":
            continue
        blockers = lane.get("blockers") or ["UNSPECIFIED_NO_TRADE_CAUSE"]
        for blocker in blockers:
            row = counts.setdefault(str(blocker), {"code": str(blocker), "count": 0, "example_lane_ids": []})
            row["count"] += 1
            if len(row["example_lane_ids"]) < 5:
                row["example_lane_ids"].append(lane["lane_id"])
    return sorted(counts.values(), key=lambda row: (-row["count"], row["code"]))


def _stale_source_reasons(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for lane in lanes:
        for blocker in lane.get("stale_source_blockers") or []:
            row = counts.setdefault(str(blocker), {"code": str(blocker), "count": 0, "example_lane_ids": []})
            row["count"] += 1
            if len(row["example_lane_ids"]) < 5:
                row["example_lane_ids"].append(lane["lane_id"])
    return sorted(counts.values(), key=lambda row: (-row["count"], row["code"]))


def _board_status(lanes: list[dict[str, Any]]) -> str:
    if not lanes:
        return "NO_LANES_FOUND"
    if any(lane["status"] == "LIVE_READY" for lane in lanes):
        return "BOARD_BUILT_LIVE_READY_DIAGNOSTIC_ONLY"
    if any(lane["status"] in {"EVIDENCE_ACQUISITION", "SCOUT_READY", "HARVEST_READY"} for lane in lanes):
        return "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY"
    if any(lane["status"] == "OPERATOR_REVIEW_REQUIRED" for lane in lanes):
        return "BOARD_BUILT_OPERATOR_REVIEW_REQUIRED_READ_ONLY"
    return "BOARD_BUILT_NO_TRADE_WITH_CAUSE"


def _next_active_path(top_lane: dict[str, Any], active_contract: dict[str, Any]) -> str:
    if not top_lane:
        return "NO_TRADE_WITH_CAUSE: no local lane candidates found."
    active_path = active_contract.get("selected_active_path")
    if top_lane["status"] == "EVIDENCE_ACQUISITION":
        return (
            f"EVIDENCE_ACQUISITION: {top_lane['lane_id']} is the closest read-only path; "
            f"active_contract={active_path or 'UNKNOWN'}."
        )
    return f"{top_lane['status']}: {top_lane['lane_id']} is top-ranked; active_contract={active_path or 'UNKNOWN'}."


def _four_x_progress_hypothesis(
    top_lane: dict[str, Any],
    lanes: list[dict[str, Any]],
    active_contract: dict[str, Any],
    goal_loop: dict[str, Any],
) -> str:
    inherited = active_contract.get("four_x_progress_hypothesis") or goal_loop.get("four_x_progress_hypothesis")
    pairs = sorted({lane["pair"] for lane in lanes if lane.get("pair") != "UNKNOWN"})
    if not top_lane:
        return str(inherited or "No active 4x path is visible in local artifacts.")
    return (
        f"Scanned {len(pairs)} pairs and {len(lanes)} pair/direction/family/vehicle lanes. "
        f"Current closest path is {top_lane['lane_id']} ({top_lane['status']}), but live permission remains false. "
        f"{inherited or '4x progress requires evidence quality improvement before execution.'}"
    )


def _root_improvement_target(top_lane: dict[str, Any], active_contract: dict[str, Any], goal_loop: dict[str, Any]) -> str:
    inherited = active_contract.get("root_improvement_target") or goal_loop.get("root_improvement_target")
    if not top_lane:
        return str(inherited or "Restore candidate evidence coverage before selecting a lane.")
    if top_lane.get("status") == "EVIDENCE_ACQUISITION":
        return f"Parallelize evidence acquisition across comparable lanes, starting with {top_lane['lane_id']}; {inherited or ''}".strip()
    return str(inherited or f"Improve {top_lane['lane_id']} without relaxing gates.")


def _expected_edge_improvement(top_lane: dict[str, Any], active_contract: dict[str, Any], goal_loop: dict[str, Any]) -> str:
    inherited = active_contract.get("expected_edge_improvement") or goal_loop.get("expected_edge_improvement")
    edge = top_lane.get("expected_edge_jpy") if top_lane else None
    if edge is not None:
        return f"Top lane expected_edge_jpy={edge}; improvement is evidence quality/ranking coverage, not live permission. {inherited or ''}".strip()
    return str(inherited or "Expected improvement is avoiding single-pair fixation and excluding negative-payoff lanes honestly.")


def _render_report(payload: dict[str, Any]) -> str:
    summary = payload.get("coverage_summary") if isinstance(payload.get("coverage_summary"), dict) else {}
    top = payload.get("top_lane") if isinstance(payload.get("top_lane"), dict) else {}
    lines = [
        "# Active Opportunity Board",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Read-only: `{payload.get('read_only')}`",
        f"- Live permission allowed: `{payload.get('live_permission_allowed')}`",
        f"- Total lanes: `{summary.get('total_lanes')}`",
        f"- Pairs scanned: `{len(summary.get('pairs_scanned') or [])}`",
        f"- Vehicles scanned: `{', '.join(summary.get('vehicles_scanned') or [])}`",
        "",
        "## Top Lane",
        "",
    ]
    if top:
        lines.extend(
            [
                f"- Lane: `{top.get('lane_id')}`",
                f"- Pair / direction / family / vehicle: `{top.get('pair')}` / `{top.get('direction')}` / `{top.get('strategy_family')}` / `{top.get('vehicle')}`",
                f"- Status: `{top.get('status')}`",
                f"- Expected edge JPY: `{top.get('expected_edge_jpy')}`",
                f"- Proof / replay: `{top.get('proof_status')}` / `{top.get('replay_status')}`",
                f"- Next action: {top.get('next_action')}",
            ]
        )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- LIVE_READY diagnostic count: `{summary.get('live_ready_count')}`",
            f"- HARVEST_READY count: `{summary.get('harvest_ready_count')}`",
            f"- SCOUT_READY count: `{summary.get('scout_ready_count')}`",
            f"- EVIDENCE_ACQUISITION count: `{summary.get('evidence_acquisition_count')}`",
            f"- OPERATOR_REVIEW_REQUIRED count: `{summary.get('operator_review_required_count')}`",
            f"- NO_TRADE count: `{summary.get('no_trade_count')}`",
            "",
            "## Active Path",
            "",
            payload.get("next_active_path", ""),
            "",
            payload.get("four_x_progress_hypothesis", ""),
            "",
            f"- Root improvement target: {payload.get('root_improvement_target')}",
            f"- Expected edge improvement: {payload.get('expected_edge_improvement')}",
            "",
            "## Top No-Trade Causes",
            "",
        ]
    )
    for reason in (payload.get("no_trade_reasons") or [])[:12]:
        lines.append(f"- `{reason.get('code')}`: {reason.get('count')}")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "This board does not authorize live order entry, SCOUT execution, gateway permission, cancellation, close, broker-state mutation, launchd load/reload, gate relaxation, 4x deficit lot backsolve, secret disclosure, or inferred operator approval.",
            "",
        ]
    )
    return "\n".join(lines)


def _do_not_do() -> list[str]:
    return [
        "do_not_send_live_order",
        "do_not_cancel_or_close",
        "do_not_launchd_load_reload",
        "do_not_mutate_broker_state",
        "do_not_relax_gates",
        "do_not_hide_negative_expectancy_or_month_scale_replay_negative",
        "do_not_treat_proof_queue_count_zero_as_live_permission",
        "do_not_backsolve_lot_from_4x_deficit",
        "do_not_print_secrets_tokens_credentials",
        "do_not_infer_operator_decision",
    ]


def _execution_ledger_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path), "event_count": 0, "verification_observation_count": 0}
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            event_count = _sqlite_count(con, "execution_events")
            observation_count = _sqlite_count(con, "verification_observations")
            latest_ts = con.execute("select max(ts_utc) from execution_events").fetchone()[0]
        finally:
            con.close()
        return {
            "status": "present",
            "path": str(path),
            "event_count": event_count,
            "verification_observation_count": observation_count,
            "latest_execution_ts_utc": latest_ts,
        }
    except sqlite3.Error as exc:
        return {"status": "unreadable", "path": str(path), "error": str(exc)}


def _sqlite_count(con: sqlite3.Connection, table: str) -> int:
    try:
        return int(con.execute(f"select count(*) from {table}").fetchone()[0])
    except sqlite3.Error:
        return 0


def _fields_from_row(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    shape = row.get("shape_key") or row.get("target_shape") or row.get("required_shape")
    parsed = _parse_shape(shape) if isinstance(shape, str) else {}
    lane_parsed = _parse_lane_id(str(row.get("lane_id") or ""))
    pair = _first_str(row.get("pair"), parsed.get("pair"), lane_parsed.get("pair"), "UNKNOWN")
    direction = _first_str(row.get("side"), row.get("direction"), parsed.get("direction"), lane_parsed.get("direction"), "UNKNOWN")
    strategy = _first_str(row.get("method"), row.get("strategy_family"), parsed.get("strategy_family"), lane_parsed.get("strategy_family"), "UNKNOWN")
    vehicle = _normalize_vehicle(_first_str(row.get("order_type"), row.get("vehicle"), parsed.get("vehicle"), lane_parsed.get("vehicle"), "UNKNOWN"))
    payoff = _first_str(row.get("exit_shape"), row.get("payoff_shape"), parsed.get("payoff_shape"), "UNKNOWN")
    return pair, direction, strategy, vehicle, payoff


def _parse_lane_id(lane_id: str) -> dict[str, str]:
    parts = lane_id.split(":")
    if len(parts) < 4:
        return {}
    return {
        "pair": parts[1],
        "direction": parts[2],
        "strategy_family": parts[3],
        "vehicle": parts[4] if len(parts) > 4 else "UNKNOWN",
    }


def _parse_shape(shape: Any) -> dict[str, str]:
    if not isinstance(shape, str):
        return {}
    parts = shape.split("|")
    parsed: dict[str, str] = {}
    if len(parts) >= 1:
        parsed["pair"] = parts[0]
    if len(parts) >= 2:
        parsed["direction"] = parts[1]
    if len(parts) >= 3:
        parsed["strategy_family"] = parts[2]
    if len(parts) >= 4:
        parsed["vehicle"] = parts[3]
    if len(parts) >= 5:
        parsed["payoff_shape"] = parts[4]
    return parsed


def _synthetic_lane_id(prefix: str, pair: str, direction: str, strategy: str, vehicle: str) -> str:
    return f"{prefix}:{pair or 'UNKNOWN'}:{direction or 'UNKNOWN'}:{strategy or 'UNKNOWN'}:{_normalize_vehicle(vehicle or 'UNKNOWN')}"


def _payoff_shape_from_intent(intent: dict[str, Any]) -> str:
    metadata = _metadata(intent)
    return _first_str(
        metadata.get("opportunity_mode"),
        metadata.get("tp_target_intent"),
        metadata.get("tp_execution_mode"),
        "UNKNOWN",
    )


def _metadata(intent: dict[str, Any]) -> dict[str, Any]:
    value = intent.get("metadata")
    return value if isinstance(value, dict) else {}


def _normalize_vehicle(value: Any) -> str:
    text = str(value or "UNKNOWN").upper()
    if text in {"STOP-ENTRY", "STOP_ENTRY"}:
        return "STOP"
    if text in {"LIMIT", "MARKET", "STOP", "UNKNOWN"}:
        return text
    return text


def _marker_status(blockers: list[str], markers: tuple[str, ...], blocked: str, clear: str) -> str:
    return blocked if _has_marker(blockers, markers) else clear


def _has_marker(blockers: list[str], markers: tuple[str, ...]) -> bool:
    return any(any(marker in str(blocker) for marker in markers) for blocker in blockers)


def _count_status(lanes: list[dict[str, Any]], status: str) -> int:
    return sum(1 for lane in lanes if lane.get("status") == status)


def _issue_codes(value: Any) -> list[str]:
    codes: list[str] = []
    for item in _list(value):
        if isinstance(item, dict):
            if str(item.get("severity") or "").upper() != "BLOCK":
                continue
            code = item.get("code") or item.get("check_name")
            if code:
                codes.append(str(code))
        elif item:
            codes.append(str(item))
    return codes


def _codes_from_blockers(value: Any) -> list[str]:
    codes: list[str] = []
    for item in _list(value):
        if isinstance(item, dict):
            code = item.get("code") or item.get("row_code") or item.get("blocker")
            if code:
                codes.append(str(code))
        elif item:
            codes.append(str(item))
    return codes


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [str(key) for key, val in value.items() if val]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, (set, tuple)):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _first_str(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text and text != "None":
            return text
    return ""


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _float(value)
        if number is not None:
            return number
    return None


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_number_or_none(value: Any) -> int | float | None:
    number = _float(value)
    if number is None:
        return None
    if number.is_integer():
        return int(number)
    return number


def _lane_key(lane: dict[str, Any]) -> str:
    return "|".join(
        [
            str(lane.get("pair") or "UNKNOWN"),
            str(lane.get("direction") or "UNKNOWN"),
            str(lane.get("strategy_family") or "UNKNOWN"),
            str(lane.get("vehicle") or "UNKNOWN"),
        ]
    )

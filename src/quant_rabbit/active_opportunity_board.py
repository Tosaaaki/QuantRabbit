from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
    DEFAULT_ACTIVE_OPPORTUNITY_BOARD_REPORT,
    DEFAULT_ACTIVE_TRADER_CONTRACT,
    DEFAULT_AS_LANE_CANDIDATE_BOARD,
    DEFAULT_AS_PROOF_PACK_QUEUE,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_CAPTURE_ECONOMICS,
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
BIDASK_REPLAY_PRECISION_RULES_PATH = Path(__file__).with_name("bidask_replay_precision_rules.json")
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
GUARDIAN_MARKERS = ("GUARDIAN",)
OPERATOR_REVIEW_CONSUMPTION_CLEAR_STATUSES = {
    "OPERATOR_REVIEW_CLEARS_RECEIPT",
    "OPERATOR_REVIEW_DURABLY_CONSUMED_RECEIPT",
}
CURRENT_INTENT_OWNED_BLOCKERS = (
    "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
)
CURRENT_LIVE_READY_STALE_DIAGNOSTIC_BLOCKERS = (
    "REALIZED_NEGATIVE_NO_POSITIVE_TP_SHAPE",
)
FAILED_EXACT_REPLAY_MARKERS = (
    "S5_TP_PATH_DOES_NOT_RECONSTRUCT_OBSERVED_TP_FILLS",
    "STOP_S5_TRIGGER_OR_TP_PATH_REPLAY_FAILED",
)
BIDASK_REPLAY_NEGATIVE_BLOCKER = "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"
BIDASK_REPLAY_EVIDENCE_REFRESH_BLOCKER = "BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED"
TP_PROVEN_ROTATION_BLOCKER = "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"
ENTRY_DROUGHT_RECOVERY_BLOCKER = "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH"
ENTRY_DROUGHT_CURRENT_INTENT_BLOCKER = "ENTRY_DROUGHT_RECOVERY_REQUIRES_CURRENT_INTENT"
ENTRY_DROUGHT_PAIR_SIDE_FALLBACK_BLOCKER = "ENTRY_DROUGHT_PAIR_SIDE_FALLBACK_REQUIRES_LANE_MAPPING"
LOCAL_TP_PROOF_ZERO_TRADES_BLOCKER = "LOCAL_TP_PROOF_ZERO_TRADES"
LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR_BLOCKER = "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR"
BROAD_TP_PROOF_NOT_EXACT_VEHICLE_BLOCKER = "BROAD_TP_PROOF_NOT_EXACT_VEHICLE"
TP_PROVEN_HARVEST_REPAIR_TARGET = "TP_PROVEN_HARVEST_BLOCKER_REPAIR_REQUIRED"
STALE_PROOF_QUEUE_ABSENCE_BLOCKERS = ("NOT_IN_PROOF_QUEUE", "PROOF_QUEUE_EMPTY_NO_LIVE_PERMISSION")
TP_PROOF_COLLECTION_MIN_TRADES = 5
BIDASK_REPLAY_NEGATIVE_MAX_AGE_HOURS = 72
BIDASK_REPLAY_NEGATIVE_LAST_DAY_MAX_AGE_DAYS = 3
ENTRY_DROUGHT_LOOKBACK_DAYS = 45
ENTRY_DROUGHT_RECENT_DAYS = 3
ENTRY_DROUGHT_MIN_ACCEPTED = 3
ENTRY_DROUGHT_MIN_FILLS = 2
ENTRY_DROUGHT_MIN_CLOSED_PL_JPY = 300.0
ENTRY_DROUGHT_MIN_EXPECTANCY_JPY = 100.0


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
        capture_economics_path: Path = DEFAULT_CAPTURE_ECONOMICS,
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
            "capture_economics": capture_economics_path,
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
        _attach_capture_economics_local_tp(
            lanes,
            artifacts["capture_economics"],
            exact_vehicle_tp_metrics=_exact_vehicle_take_profit_metrics(self.execution_ledger_db_path),
        )
        _add_proof_queue_lanes(lanes, artifacts["as_proof_pack_queue"])
        _add_portfolio_lanes(lanes, artifacts["portfolio_4x_path_planner"])
        _add_lane_board_lanes(lanes, artifacts["as_lane_candidate_board"])
        _add_payoff_lanes(lanes, artifacts["payoff_shape_diagnosis"])
        _add_harvest_lanes(lanes, artifacts["harvest_live_grade_path"], artifacts["order_intents"])
        _add_replay_lanes(lanes, replay_artifacts)
        _attach_strategy_profile(lanes, artifacts["strategy_profile"])
        _attach_verification_ledger(lanes, artifacts["verification_ledger"])
        _attach_goal_loop_edge_improvement_context(lanes, artifacts["trader_goal_loop_orchestrator"])
        _add_execution_recovery_lanes(
            lanes,
            self.execution_ledger_db_path,
            now_utc=self.now_utc,
        )
        _attach_packaged_pair_side_bidask_negative_evidence(lanes)

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
                now_utc=self.now_utc,
            )

        ranked_lanes = sorted(
            lanes.values(),
            key=_rank_sort_key,
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
            "entry_recovery_summary": _entry_recovery_summary(public_lanes),
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
        _attach_local_tp_proof_context(lane, intent)
        _attach_bidask_negative_evidence(lane, intent, intent_blockers)


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


def _add_harvest_lanes(
    lanes: dict[str, dict[str, Any]],
    artifact: dict[str, Any],
    order_intents: dict[str, Any],
) -> None:
    stale_vs_order_intents = _artifact_is_older_than(artifact, order_intents)
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
                _attach_harvest_row(
                    lane,
                    row,
                    key,
                    payoff,
                    stale_vs_order_intents=stale_vs_order_intents,
                )


def _attach_harvest_row(
    lane: dict[str, Any],
    row: dict[str, Any],
    key: str,
    payoff: str,
    *,
    stale_vs_order_intents: bool = False,
) -> None:
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
    promotion_blockers = _string_list(row.get("promotion_blockers"))
    if stale_vs_order_intents and lane.get("order_intent_status"):
        lane["stale_source_blockers"].extend(promotion_blockers)
    else:
        lane["blockers"].extend(promotion_blockers)


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
        if not _verification_row_is_blocking(row):
            continue
        subject_id = row.get("subject_id")
        if not subject_id or str(subject_id) not in lanes:
            continue
        lane = lanes[str(subject_id)]
        blocker_codes = _verification_ledger_blocker_codes(row)
        if not blocker_codes:
            continue
        lane["source_refs"].append("data/verification_ledger.json")
        lane["blockers"].extend(blocker_codes)


def _verification_row_is_blocking(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    return str(row.get("status") or "").upper() == "BLOCK" or str(row.get("severity") or "").upper() == "BLOCK"


def _verification_ledger_blocker_codes(row: dict[str, Any]) -> list[str]:
    check_name = str(row.get("check_name") or "")
    if check_name == "lane_blockers":
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        return _blocker_object_codes(evidence.get("blockers"))
    return [check_name] if check_name else []


def _attach_goal_loop_edge_improvement_context(lanes: dict[str, dict[str, Any]], artifact: dict[str, Any]) -> None:
    if artifact.get("selected_next_work_type") != "EDGE_IMPROVEMENT_EXPERIMENT":
        return
    state = artifact.get("edge_improvement_state") if isinstance(artifact.get("edge_improvement_state"), dict) else {}
    targets = _goal_loop_edge_targets(state)
    if not targets:
        return
    for lane in lanes.values():
        for target in targets:
            if not _lane_matches_target(lane, target):
                continue
            lane["edge_improvement_candidate"] = True
            lane["edge_improvement_target"] = _shape_from_target(target)
            lane["source_refs"].append("data/trader_goal_loop_orchestrator.json:edge_improvement_state")
            break


def _add_execution_recovery_lanes(
    lanes: dict[str, dict[str, Any]],
    execution_ledger_db_path: Path,
    *,
    now_utc: datetime,
) -> None:
    candidates = _execution_entry_recovery_candidates(execution_ledger_db_path, now_utc=now_utc)
    for candidate in candidates:
        lane_id = str(candidate.get("lane_id") or "")
        pair = str(candidate.get("pair") or "UNKNOWN")
        direction = str(candidate.get("direction") or "UNKNOWN")
        strategy = str(candidate.get("strategy_family") or "UNKNOWN")
        vehicle = _normalize_vehicle(candidate.get("vehicle") or "UNKNOWN")
        target_lane_id = lane_id or _synthetic_lane_id("entry_recovery", pair, direction, strategy, vehicle)
        lane = _ensure_lane(lanes, target_lane_id, pair, direction, strategy, vehicle)
        history = {
            "lookback_days": ENTRY_DROUGHT_LOOKBACK_DAYS,
            "recent_days": ENTRY_DROUGHT_RECENT_DAYS,
            "accepted_before_recent": candidate.get("accepted_before_recent"),
            "fills_before_recent": candidate.get("fills_before_recent"),
            "recent_accepted": candidate.get("recent_accepted"),
            "recent_fills": candidate.get("recent_fills"),
            "closed_trades": candidate.get("closed_trades"),
            "closed_pl_jpy": candidate.get("closed_pl_jpy"),
            "win_rate": candidate.get("win_rate"),
            "profit_source": candidate.get("profit_source"),
            "first_entry_ts_utc": candidate.get("first_entry_ts_utc"),
            "last_entry_ts_utc": candidate.get("last_entry_ts_utc"),
            "last_closed_ts_utc": candidate.get("last_closed_ts_utc"),
            "historical_lane_id": lane_id,
        }
        lane["entry_recovery_candidate"] = True
        lane["entry_recovery_history"] = history
        lane["source_refs"].append("data/execution_ledger.db:entry_recovery")
        recovery_blockers = [ENTRY_DROUGHT_RECOVERY_BLOCKER]
        if history.get("profit_source") == "pair_side_fallback":
            recovery_blockers.append(ENTRY_DROUGHT_PAIR_SIDE_FALLBACK_BLOCKER)
        if not lane.get("order_intent_status"):
            recovery_blockers.append(ENTRY_DROUGHT_CURRENT_INTENT_BLOCKER)
        lane["blockers"].extend(recovery_blockers)


def _execution_entry_recovery_candidates(path: Path, *, now_utc: datetime) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    history_start = now_utc - timedelta(days=ENTRY_DROUGHT_LOOKBACK_DAYS)
    recent_start = now_utc - timedelta(days=ENTRY_DROUGHT_RECENT_DAYS)
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                """
                select ts_utc,event_type,lane_id,pair,side,trade_id,realized_pl_jpy,raw_json
                from execution_events
                where ts_utc >= ?
                  and event_type in ('ORDER_ACCEPTED','ORDER_FILLED','TRADE_CLOSED')
                order by ts_utc asc
                """,
                (history_start.isoformat(),),
            ).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return []

    entries: dict[tuple[str, str, str], dict[str, Any]] = {}
    pair_side_pnl: dict[tuple[str, str], dict[str, Any]] = {}
    exact_pnl: dict[tuple[str, str, str], dict[str, Any]] = {}
    trade_open_lanes: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        if str(row["event_type"] or "") != "ORDER_FILLED":
            continue
        lane_id = str(row["lane_id"] or "")
        trade_id = str(row["trade_id"] or "")
        pair = str(row["pair"] or "UNKNOWN")
        side = str(row["side"] or "UNKNOWN")
        if lane_id and trade_id:
            trade_open_lanes.setdefault(trade_id, (lane_id, pair, side))

    for row in rows:
        ts = _parse_utc(row["ts_utc"])
        if ts is None:
            continue
        pair = str(row["pair"] or "UNKNOWN")
        side = str(row["side"] or "UNKNOWN")
        lane_id = str(row["lane_id"] or "")
        trade_id = str(row["trade_id"] or "")
        event_type = str(row["event_type"] or "")
        key = (lane_id, pair, side)
        if event_type in {"ORDER_ACCEPTED", "ORDER_FILLED"}:
            parsed = _parse_lane_id(lane_id)
            group = entries.setdefault(
                key,
                {
                    "lane_id": lane_id,
                    "pair": pair,
                    "direction": side,
                    "strategy_family": parsed.get("strategy_family") or "UNKNOWN",
                    "vehicle_counts": {},
                    "accepted_before_recent": 0,
                    "fills_before_recent": 0,
                    "recent_accepted": 0,
                    "recent_fills": 0,
                    "first_entry_ts_utc": None,
                    "last_entry_ts_utc": None,
                },
            )
            vehicle = _vehicle_from_execution_raw(row["raw_json"])
            if vehicle:
                group["vehicle_counts"][vehicle] = int(group["vehicle_counts"].get(vehicle, 0)) + 1
            group["first_entry_ts_utc"] = group["first_entry_ts_utc"] or ts.isoformat()
            group["last_entry_ts_utc"] = ts.isoformat()
            if ts >= recent_start:
                if event_type == "ORDER_ACCEPTED":
                    group["recent_accepted"] += 1
                elif event_type == "ORDER_FILLED":
                    group["recent_fills"] += 1
            else:
                if event_type == "ORDER_ACCEPTED":
                    group["accepted_before_recent"] += 1
                elif event_type == "ORDER_FILLED":
                    group["fills_before_recent"] += 1
            continue

        if event_type == "TRADE_CLOSED" and ts < recent_start:
            pl = _float(row["realized_pl_jpy"]) or 0.0
            _accumulate_closed_pnl(pair_side_pnl.setdefault((pair, side), {}), pl, ts)
            close_key = key if lane_id else trade_open_lanes.get(trade_id)
            if close_key:
                _accumulate_closed_pnl(exact_pnl.setdefault(close_key, {}), pl, ts)

    candidates: list[dict[str, Any]] = []
    for key, group in entries.items():
        accepted = int(group["accepted_before_recent"])
        fills = int(group["fills_before_recent"])
        if accepted < ENTRY_DROUGHT_MIN_ACCEPTED and fills < ENTRY_DROUGHT_MIN_FILLS:
            continue
        if int(group["recent_accepted"]) > 0 or int(group["recent_fills"]) > 0:
            continue
        lane_id, pair, side = key
        pnl = exact_pnl.get(key) or pair_side_pnl.get((pair, side)) or {}
        closed_pl = _float(pnl.get("closed_pl_jpy")) or 0.0
        closed_trades = int(pnl.get("closed_trades") or 0)
        if not _entry_recovery_profit_is_material(closed_pl, closed_trades):
            continue
        vehicle = _dominant_vehicle(group.get("vehicle_counts"))
        candidate = {
            **{k: v for k, v in group.items() if k != "vehicle_counts"},
            "vehicle": vehicle or _parse_lane_id(lane_id).get("vehicle") or "UNKNOWN",
            "closed_trades": closed_trades,
            "closed_pl_jpy": round(closed_pl, 4),
            "closed_expectancy_jpy": round(closed_pl / closed_trades, 4) if closed_trades else None,
            "win_rate": _json_number_or_none(pnl.get("win_rate")),
            "profit_source": "exact_lane" if key in exact_pnl else "pair_side_fallback",
            "last_closed_ts_utc": pnl.get("last_closed_ts_utc"),
        }
        candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda row: (
            -float(row.get("closed_pl_jpy") or 0.0),
            -int(row.get("accepted_before_recent") or 0),
            str(row.get("pair") or ""),
            str(row.get("direction") or ""),
        ),
    )


def _accumulate_closed_pnl(bucket: dict[str, Any], pl: float, ts: datetime) -> None:
    bucket["closed_trades"] = int(bucket.get("closed_trades") or 0) + 1
    bucket["closed_pl_jpy"] = float(bucket.get("closed_pl_jpy") or 0.0) + pl
    if pl > 0:
        bucket["winning_trades"] = int(bucket.get("winning_trades") or 0) + 1
    wins = int(bucket.get("winning_trades") or 0)
    trades = int(bucket.get("closed_trades") or 0)
    bucket["win_rate"] = wins / trades if trades else None
    bucket["last_closed_ts_utc"] = ts.isoformat()


def _entry_recovery_profit_is_material(closed_pl_jpy: float, closed_trades: int) -> bool:
    if closed_trades <= 0:
        return False
    if closed_pl_jpy < ENTRY_DROUGHT_MIN_CLOSED_PL_JPY:
        return False
    return (closed_pl_jpy / closed_trades) >= ENTRY_DROUGHT_MIN_EXPECTANCY_JPY


def _vehicle_from_execution_raw(raw_json: Any) -> str:
    if not raw_json:
        return ""
    try:
        payload = json.loads(str(raw_json))
    except json.JSONDecodeError:
        return ""
    order_type = str(payload.get("type") or "").upper()
    if order_type == "LIMIT_ORDER":
        return "LIMIT"
    if order_type == "STOP_ORDER":
        return "STOP"
    if order_type == "MARKET_ORDER":
        return "MARKET"
    return ""


def _dominant_vehicle(counts: Any) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]


def _goal_loop_edge_targets(state: dict[str, Any]) -> list[dict[str, str]]:
    target_values: list[str] = []
    for key in ("target_shape", "target", "target_shape_key", "closest_harvest_candidate_id"):
        value = state.get(key)
        if isinstance(value, str):
            target_values.append(value)
    for experiment in _list(state.get("experiments")):
        if isinstance(experiment, dict) and isinstance(experiment.get("target"), str):
            target_values.append(str(experiment["target"]))

    targets: list[dict[str, str]] = []
    for value in target_values:
        parsed = _parse_shape(value)
        if not (parsed.get("pair") and parsed.get("direction") and parsed.get("strategy_family")):
            continue
        if parsed not in targets:
            targets.append(parsed)
    return targets


def _lane_matches_target(lane: dict[str, Any], target: dict[str, str]) -> bool:
    if lane.get("pair") != target.get("pair"):
        return False
    if lane.get("direction") != target.get("direction"):
        return False
    if lane.get("strategy_family") != target.get("strategy_family"):
        return False
    target_vehicle = _normalize_vehicle(target.get("vehicle"))
    return target_vehicle == "UNKNOWN" or lane.get("vehicle") == target_vehicle


def _shape_from_target(target: dict[str, str]) -> str:
    parts = [
        target.get("pair", "UNKNOWN"),
        target.get("direction", "UNKNOWN"),
        target.get("strategy_family", "UNKNOWN"),
    ]
    vehicle = _normalize_vehicle(target.get("vehicle"))
    if vehicle != "UNKNOWN":
        parts.append(vehicle)
    return "|".join(parts)


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
    now_utc: datetime,
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
    _suppress_stale_proof_queue_absence_blockers(lane)
    _suppress_stale_current_intent_owned_blockers(lane)
    _suppress_live_ready_stale_diagnostic_blockers(lane)
    _mark_bidask_negative_evidence_refresh(lane, now_utc=now_utc)
    _mark_tp_proven_harvest_repair_candidate(lane)
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
    if consumption.get("_artifact_status") != "present":
        return False
    if consumption.get("normal_routing_allowed") is not True:
        return False
    if operator_review.get("_artifact_status") == "present" and operator_review.get("normal_routing_allowed") is True:
        return True
    return _consumption_has_durable_operator_review_clearance(consumption)


def _consumption_has_durable_operator_review_clearance(consumption: dict[str, Any]) -> bool:
    rows = consumption.get("classifications") if isinstance(consumption.get("classifications"), list) else []
    if not rows:
        return False
    normal_rows = [row for row in rows if isinstance(row, dict)]
    if len(normal_rows) != len(rows):
        return False
    if any(row.get("normal_routing_allowed") is not True for row in normal_rows):
        return False
    return any(
        row.get("operator_review_required") is True
        and str(row.get("operator_review_status") or "") in OPERATOR_REVIEW_CONSUMPTION_CLEAR_STATUSES
        for row in normal_rows
    )


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


def _suppress_stale_proof_queue_absence_blockers(lane: dict[str, Any]) -> None:
    if not (
        lane.get("can_enter_proof_pack")
        or "data/as_proof_pack_queue.json:queue" in _string_list(lane.get("source_refs"))
    ):
        return
    blockers = _string_list(lane.get("blockers"))
    stale_codes = [code for code in STALE_PROOF_QUEUE_ABSENCE_BLOCKERS if code in blockers]
    if not stale_codes:
        return
    lane["blockers"] = [code for code in blockers if code not in stale_codes]
    lane["stale_source_blockers"] = _unique(_string_list(lane.get("stale_source_blockers")) + stale_codes)


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


def _suppress_live_ready_stale_diagnostic_blockers(lane: dict[str, Any]) -> None:
    if lane.get("order_intent_status") != "LIVE_READY" or not lane.get("risk_allowed"):
        return
    current_intent_blockers = set(_string_list(lane.get("order_intent_blockers")))
    blockers = _string_list(lane.get("blockers"))
    stale_codes = [
        code
        for code in CURRENT_LIVE_READY_STALE_DIAGNOSTIC_BLOCKERS
        if code in blockers and code not in current_intent_blockers
    ]
    if not stale_codes:
        return
    lane["blockers"] = [code for code in blockers if code not in stale_codes]
    lane["stale_source_blockers"] = _unique(_string_list(lane.get("stale_source_blockers")) + stale_codes)


def _attach_local_tp_proof_context(lane: dict[str, Any], intent: dict[str, Any]) -> None:
    metadata = _metadata(intent)
    if not metadata:
        return
    proof = dict(lane.get("local_tp_proof") or {})
    for key in (
        "attach_take_profit_on_fill",
        "capture_take_profit_avg_loss_jpy",
        "capture_take_profit_avg_win_jpy",
        "capture_take_profit_expectancy_jpy",
        "capture_take_profit_losses",
        "capture_take_profit_scope",
        "capture_take_profit_scope_key",
        "capture_take_profit_trades",
        "capture_take_profit_wins",
        "tp_execution_mode",
        "tp_target_intent",
        "tp_target_source",
    ):
        if key in metadata:
            proof[key] = metadata.get(key)
    if proof:
        lane["local_tp_proof"] = proof


def _attach_capture_economics_local_tp(
    lanes: dict[str, dict[str, Any]],
    artifact: dict[str, Any],
    *,
    exact_vehicle_tp_metrics: dict[tuple[str, str, str, str], dict[str, Any]] | None = None,
) -> None:
    if artifact.get("_artifact_status") != "present":
        return
    floor = int(
        _float((artifact.get("segment_repair_priorities") or {}).get("scoped_tp_proof_min_exit_trades"))
        or _float(artifact.get("min_sample_for_verdict"))
        or 20
    )
    for lane in lanes.values():
        proof = lane.get("local_tp_proof")
        if not isinstance(proof, dict) or not proof:
            continue
        pair = str(lane.get("pair") or "")
        direction = str(lane.get("direction") or "")
        method = str(lane.get("strategy_family") or "")
        if not pair or not direction or not method:
            continue
        broad_metrics, broad_scope_name, broad_scope_key = _capture_scoped_exit_metrics(
            artifact,
            pair=pair,
            side=direction,
            method=method,
            exit_reason="TAKE_PROFIT_ORDER",
        )
        enriched = dict(proof)
        vehicle = _normalize_vehicle(lane.get("vehicle"))
        exact_scope_available = exact_vehicle_tp_metrics is not None and vehicle not in {"", "UNKNOWN"}
        exact_scope_key = f"{pair}|{direction}|{method}|{vehicle}|TAKE_PROFIT_ORDER"
        if exact_scope_available:
            metrics = exact_vehicle_tp_metrics.get((pair.upper(), direction.upper(), method.upper(), vehicle.upper()), {})
            scope_name = "PAIR_SIDE_METHOD_VEHICLE"
            scope_key = exact_scope_key
            enriched["capture_take_profit_metrics_source"] = "data/execution_ledger.db:exact_vehicle_take_profit"
        else:
            metrics = broad_metrics
            scope_name = broad_scope_name
            scope_key = broad_scope_key
            enriched["capture_take_profit_metrics_source"] = "data/capture_economics.json"
        enriched["capture_take_profit_scope"] = scope_name
        enriched["capture_take_profit_scope_key"] = scope_key
        enriched["capture_take_profit_proof_floor"] = floor
        if broad_metrics:
            enriched["broad_capture_take_profit_scope"] = broad_scope_name
            enriched["broad_capture_take_profit_scope_key"] = broad_scope_key
            enriched["broad_capture_take_profit_trades"] = _first_int(broad_metrics.get("trades"), 0)
            enriched["broad_capture_take_profit_wins"] = _first_int(broad_metrics.get("wins"), 0)
            enriched["broad_capture_take_profit_losses"] = _first_int(broad_metrics.get("losses"), 0)
            enriched["broad_capture_take_profit_expectancy_jpy"] = _json_number_or_none(
                broad_metrics.get("expectancy_jpy_per_trade")
            )
        if metrics:
            enriched["capture_take_profit_trades"] = _first_int(metrics.get("trades"), 0)
            enriched["capture_take_profit_wins"] = _first_int(metrics.get("wins"), 0)
            enriched["capture_take_profit_losses"] = _first_int(metrics.get("losses"), 0)
            enriched["capture_take_profit_expectancy_jpy"] = _json_number_or_none(
                metrics.get("expectancy_jpy_per_trade")
            )
            enriched["capture_take_profit_avg_win_jpy"] = _json_number_or_none(metrics.get("avg_win_jpy"))
            enriched["capture_take_profit_avg_loss_jpy"] = _json_number_or_none(metrics.get("avg_loss_jpy"))
        else:
            enriched["capture_take_profit_trades"] = 0
            enriched["capture_take_profit_wins"] = 0
            enriched["capture_take_profit_losses"] = 0
            enriched["capture_take_profit_expectancy_jpy"] = 0.0
            enriched["capture_take_profit_avg_win_jpy"] = 0.0
            enriched["capture_take_profit_avg_loss_jpy"] = 0.0
            enriched["capture_take_profit_zero_trade"] = True
        lane["local_tp_proof"] = enriched
        _append_broad_tp_not_exact_vehicle_blocker(lane, enriched)
        _append_local_tp_proof_floor_blocker(lane, enriched)


def _capture_scoped_exit_metrics(
    artifact: dict[str, Any],
    *,
    pair: str,
    side: str,
    method: str,
    exit_reason: str,
) -> tuple[dict[str, Any], str, str]:
    by_method = artifact.get("by_pair_side_method_exit_reason")
    method_scopes = _nested_dict_get(by_method, pair, side)
    if method_scopes is not None:
        method_exits = method_scopes.get(method)
        if isinstance(method_exits, dict):
            metrics = method_exits.get(exit_reason)
            if isinstance(metrics, dict):
                return metrics, "PAIR_SIDE_METHOD", f"{pair}|{side}|{method}|{exit_reason}"
            return {}, "MISSING_METHOD_EXIT", f"{pair}|{side}|{method}|{exit_reason}"
        return {}, "MISSING_METHOD_SCOPE", f"{pair}|{side}|{method}|{exit_reason}"
    by_pair_side = artifact.get("by_pair_side_exit_reason")
    pair_side_metrics = _nested_dict_get(by_pair_side, pair, side, exit_reason)
    if pair_side_metrics is not None:
        return pair_side_metrics, "PAIR_SIDE", f"{pair}|{side}|{exit_reason}"
    return {}, "MISSING_SCOPED", f"{pair}|{side}|{method}|{exit_reason}"


def _exact_vehicle_take_profit_metrics(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            columns = {
                str(row["name"])
                for row in con.execute("pragma table_info(execution_events)").fetchall()
                if "name" in row.keys()
            }
            required = {"ts_utc", "event_type", "trade_id", "lane_id", "pair", "side", "realized_pl_jpy", "exit_reason"}
            if not required.issubset(columns):
                return None
            rows = con.execute(
                """
                WITH fills AS (
                    SELECT
                        trade_id,
                        MAX(NULLIF(lane_id, '')) AS lane_id,
                        MAX(NULLIF(pair, '')) AS pair,
                        MAX(NULLIF(side, '')) AS side,
                        MAX(NULLIF(exit_reason, '')) AS entry_reason
                    FROM execution_events
                    WHERE event_type = 'ORDER_FILLED'
                      AND COALESCE(trade_id, '') != ''
                    GROUP BY trade_id
                ),
                closes AS (
                    SELECT
                        e.trade_id AS trade_id,
                        MAX(NULLIF(e.lane_id, '')) AS close_lane_id,
                        MAX(NULLIF(e.pair, '')) AS pair,
                        MAX(NULLIF(e.side, '')) AS side,
                        SUM(e.realized_pl_jpy) AS realized_pl_jpy,
                        (
                            SELECT e2.exit_reason
                            FROM execution_events e2
                            WHERE e2.trade_id = e.trade_id
                              AND e2.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
                              AND e2.realized_pl_jpy IS NOT NULL
                            ORDER BY e2.ts_utc DESC
                            LIMIT 1
                        ) AS final_exit_reason
                    FROM execution_events e
                    WHERE e.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
                      AND e.realized_pl_jpy IS NOT NULL
                      AND COALESCE(e.trade_id, '') != ''
                    GROUP BY e.trade_id
                )
                SELECT
                    COALESCE(f.lane_id, c.close_lane_id, '') AS entry_lane_id,
                    COALESCE(f.entry_reason, '') AS entry_reason,
                    COALESCE(f.pair, c.pair, '') AS pair,
                    COALESCE(f.side, c.side, '') AS side,
                    COUNT(*) AS trades,
                    SUM(CASE WHEN c.realized_pl_jpy > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN c.realized_pl_jpy < 0 THEN 1 ELSE 0 END) AS losses,
                    SUM(c.realized_pl_jpy) AS net_jpy,
                    SUM(CASE WHEN c.realized_pl_jpy > 0 THEN c.realized_pl_jpy ELSE 0 END) AS win_jpy,
                    SUM(CASE WHEN c.realized_pl_jpy < 0 THEN c.realized_pl_jpy ELSE 0 END) AS loss_jpy
                FROM closes c
                LEFT JOIN fills f ON f.trade_id = c.trade_id
                WHERE c.final_exit_reason = 'TAKE_PROFIT_ORDER'
                  AND COALESCE(f.lane_id, c.close_lane_id, '') != ''
                GROUP BY 1, 2, 3, 4
                """
            ).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return None

    accum: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for row in rows:
        lane_id = str(row["entry_lane_id"] or "")
        parsed = _parse_lane_id(lane_id)
        pair = str(parsed.get("pair") or row["pair"] or "UNKNOWN").upper()
        direction = str(parsed.get("direction") or row["side"] or "UNKNOWN").upper()
        method = str(parsed.get("strategy_family") or "UNKNOWN").upper()
        vehicle = _normalize_vehicle(parsed.get("vehicle") or row["entry_reason"] or "UNKNOWN")
        if "UNKNOWN" in {pair, direction, method, vehicle}:
            continue
        trades = int(row["trades"] or 0)
        wins = int(row["wins"] or 0)
        losses = int(row["losses"] or 0)
        net_jpy = float(row["net_jpy"] or 0.0)
        win_jpy = float(row["win_jpy"] or 0.0)
        loss_jpy = float(row["loss_jpy"] or 0.0)
        key = (pair, direction, method, vehicle)
        slot = accum.setdefault(
            key,
            {"trades": 0.0, "wins": 0.0, "losses": 0.0, "net_jpy": 0.0, "win_jpy": 0.0, "loss_jpy": 0.0},
        )
        slot["trades"] += trades
        slot["wins"] += wins
        slot["losses"] += losses
        slot["net_jpy"] += net_jpy
        slot["win_jpy"] += win_jpy
        slot["loss_jpy"] += loss_jpy

    metrics: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for key, slot in accum.items():
        trades = int(slot["trades"])
        wins = int(slot["wins"])
        losses = int(slot["losses"])
        net_jpy = float(slot["net_jpy"])
        win_jpy = float(slot["win_jpy"])
        loss_jpy = float(slot["loss_jpy"])
        metrics[key] = {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "expectancy_jpy_per_trade": round(net_jpy / trades, 4) if trades else 0.0,
            "avg_win_jpy": round(win_jpy / wins, 4) if wins else 0.0,
            "avg_loss_jpy": round(abs(loss_jpy) / losses, 4) if losses else 0.0,
            "net_jpy": round(net_jpy, 4),
            "source_scope": "PAIR_SIDE_METHOD_VEHICLE",
        }
    return metrics


def _nested_dict_get(root: object, *keys: str) -> dict[str, Any] | None:
    cursor: object = root
    for key in keys:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor if isinstance(cursor, dict) else None


def _append_local_tp_proof_floor_blocker(lane: dict[str, Any], proof: dict[str, Any]) -> None:
    blockers = _string_list(lane.get("blockers"))
    trades = _first_int(proof.get("capture_take_profit_trades"), 0)
    if (
        TP_PROVEN_ROTATION_BLOCKER not in blockers
        and BROAD_TP_PROOF_NOT_EXACT_VEHICLE_BLOCKER not in blockers
        and not (0 < trades < TP_PROOF_COLLECTION_MIN_TRADES)
    ):
        return
    if trades <= 0:
        lane["blockers"] = _unique(blockers + [LOCAL_TP_PROOF_ZERO_TRADES_BLOCKER])
    elif trades < TP_PROOF_COLLECTION_MIN_TRADES:
        lane["blockers"] = _unique(blockers + [LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR_BLOCKER])


def _append_broad_tp_not_exact_vehicle_blocker(lane: dict[str, Any], proof: dict[str, Any]) -> None:
    broad_trades = _float(proof.get("broad_capture_take_profit_trades"))
    exact_trades = _float(proof.get("capture_take_profit_trades"))
    if broad_trades is None or exact_trades is None or broad_trades <= exact_trades:
        return
    if str(proof.get("capture_take_profit_scope") or "") != "PAIR_SIDE_METHOD_VEHICLE":
        return
    proof["broad_capture_take_profit_not_used_as_exact_vehicle_proof"] = True
    blockers = _string_list(lane.get("blockers"))
    lane["blockers"] = _unique(blockers + [BROAD_TP_PROOF_NOT_EXACT_VEHICLE_BLOCKER])


def _mark_tp_proven_harvest_repair_candidate(lane: dict[str, Any]) -> None:
    if _has_marker(lane.get("blockers") or [], FAILED_EXACT_REPLAY_MARKERS):
        return
    if not _local_tp_proof_floor_met(lane):
        return
    blockers = _string_list(lane.get("blockers"))
    repair_markers = (
        NEGATIVE_BLOCKER_MARKERS
        + RISK_BLOCKER_MARKERS
        + SPREAD_BLOCKER_MARKERS
        + ("FORECAST", "HARVEST_TP_STRUCTURE_MISSING", "EXHAUSTION_RANGE_CHASE", "PATTERN_REVERSAL_CHASE")
    )
    if not _has_marker(blockers, repair_markers):
        return
    lane["edge_improvement_candidate"] = True
    lane["edge_improvement_target"] = _shape_from_target(lane)
    lane["tp_proven_harvest_repair_target"] = TP_PROVEN_HARVEST_REPAIR_TARGET


def _local_tp_proof_floor_met(lane: dict[str, Any]) -> bool:
    proof = lane.get("local_tp_proof")
    if not isinstance(proof, dict):
        return False
    if proof.get("attach_take_profit_on_fill") is not True:
        return False
    if str(proof.get("tp_execution_mode") or "") != "ATTACHED_TECHNICAL_TP":
        return False
    if str(proof.get("tp_target_intent") or "") != "HARVEST":
        return False
    if str(proof.get("capture_take_profit_scope") or "") != "PAIR_SIDE_METHOD_VEHICLE":
        return False
    trades = _float(proof.get("capture_take_profit_trades"))
    losses = _float(proof.get("capture_take_profit_losses"))
    expectancy = _float(proof.get("capture_take_profit_expectancy_jpy"))
    floor = _float(proof.get("capture_take_profit_proof_floor")) or float(TP_PROOF_COLLECTION_MIN_TRADES)
    return (
        trades is not None
        and trades >= floor
        and (losses is None or losses == 0)
        and expectancy is not None
        and expectancy > 0
    )


def _attach_bidask_negative_evidence(
    lane: dict[str, Any],
    intent: dict[str, Any],
    intent_blockers: list[str],
) -> None:
    if BIDASK_REPLAY_NEGATIVE_BLOCKER not in intent_blockers:
        return
    evidence = _metadata(intent).get("bidask_replay_precision_negative")
    if not isinstance(evidence, dict):
        return
    audit_report = _first_str(evidence.get("audit_report"))
    audit_path = _resolve_artifact_path(audit_report) if audit_report else None
    payload: dict[str, Any] = {}
    for key in (
        "name",
        "pair",
        "side",
        "direction",
        "granularity",
        "samples",
        "active_days",
        "first_day",
        "last_day",
        "directional_hit_rate",
        "avg_final_pips",
        "median_final_pips",
        "avg_mae_pips",
        "avg_mfe_pips",
        "positive_days",
        "negative_days",
        "positive_day_rate",
        "avg_daily_realized_pips",
        "daily_stability_status",
        "audit_report",
        "rule_set_generated_at_utc",
        "rule_set_source",
        "price_truth_coverage",
    ):
        if key in evidence:
            payload[key] = evidence[key]
    if audit_report:
        payload["audit_report_exists"] = bool(audit_path and audit_path.exists())
        payload["audit_report_resolved_path"] = str(audit_path) if audit_path else ""
    lane["bidask_negative_evidence"] = payload


def _attach_packaged_pair_side_bidask_negative_evidence(lanes: dict[str, dict[str, Any]]) -> None:
    rules = _packaged_bidask_negative_rules()
    if not rules:
        return
    by_pair_side: dict[tuple[str, str], dict[str, Any]] = {}
    for rule in rules:
        if rule.get("blocks_live_support") is False:
            continue
        pair = _first_str(rule.get("pair")).upper()
        side = _first_str(rule.get("side")).upper()
        if not pair or side not in {"LONG", "SHORT"}:
            continue
        current = by_pair_side.get((pair, side))
        if current is None or _packaged_negative_rule_sort_key(rule) > _packaged_negative_rule_sort_key(current):
            by_pair_side[(pair, side)] = rule

    for lane in lanes.values():
        pair = _first_str(lane.get("pair")).upper()
        side = _first_str(lane.get("direction")).upper()
        rule = by_pair_side.get((pair, side))
        if not rule:
            continue
        blockers = _string_list(lane.get("blockers"))
        if not blockers and lane.get("units") and lane.get("risk_jpy") is not None:
            continue
        if BIDASK_REPLAY_NEGATIVE_BLOCKER not in blockers:
            blockers.append(BIDASK_REPLAY_NEGATIVE_BLOCKER)
            lane["blockers"] = blockers
        packaged_payload = _packaged_bidask_negative_payload(rule)
        existing_evidence = lane.get("bidask_negative_evidence")
        if not isinstance(existing_evidence, dict):
            lane["bidask_negative_evidence"] = packaged_payload
        elif _should_replace_intent_bidask_evidence_with_packaged(existing_evidence, packaged_payload):
            packaged_payload["replaced_intent_bidask_negative_evidence"] = {
                "audit_report": existing_evidence.get("audit_report"),
                "audit_report_exists": existing_evidence.get("audit_report_exists"),
                "rule_set_generated_at_utc": existing_evidence.get("rule_set_generated_at_utc"),
                "last_day": existing_evidence.get("last_day"),
            }
            lane["bidask_negative_evidence"] = packaged_payload


def _packaged_bidask_negative_rules() -> list[dict[str, Any]]:
    try:
        payload = json.loads(BIDASK_REPLAY_PRECISION_RULES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_rules = payload.get("negative_rules") if isinstance(payload, dict) else None
    if not isinstance(raw_rules, list):
        return []
    rules: list[dict[str, Any]] = []
    for raw in raw_rules:
        if not isinstance(raw, dict):
            continue
        pair = _first_str(raw.get("pair")).upper()
        direction = _first_str(raw.get("direction")).upper()
        side = _first_str(raw.get("side")).upper()
        if not side:
            side = "LONG" if direction == "UP" else "SHORT" if direction == "DOWN" else ""
        if not pair or side not in {"LONG", "SHORT"}:
            continue
        rule = dict(raw)
        rule["pair"] = pair
        rule["side"] = side
        if direction:
            rule["direction"] = direction
        rule.setdefault("rule_set_generated_at_utc", payload.get("generated_at_utc"))
        rule.setdefault("rule_set_source", payload.get("generated_from") or str(BIDASK_REPLAY_PRECISION_RULES_PATH))
        if isinstance(payload.get("price_truth_coverage"), dict):
            rule.setdefault("price_truth_coverage", payload["price_truth_coverage"])
        rules.append(rule)
    return rules


def _packaged_negative_rule_sort_key(rule: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        _float(rule.get("samples")) or 0.0,
        _float(rule.get("active_days")) or 0.0,
        -(_float(rule.get("optimized_profit_factor")) or 0.0),
        -(_float(rule.get("positive_day_rate")) or 0.0),
    )


def _packaged_bidask_negative_payload(rule: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in (
        "name",
        "pair",
        "side",
        "direction",
        "granularity",
        "samples",
        "active_days",
        "first_day",
        "last_day",
        "directional_hit_rate",
        "avg_final_pips",
        "median_final_pips",
        "avg_mae_pips",
        "avg_mfe_pips",
        "positive_days",
        "negative_days",
        "positive_day_rate",
        "avg_daily_realized_pips",
        "optimized_profit_factor",
        "daily_stability_status",
        "audit_report",
        "rule_set_generated_at_utc",
        "rule_set_source",
        "price_truth_coverage",
    ):
        if key in rule:
            payload[key] = rule[key]
    audit_report = _first_str(rule.get("audit_report"))
    if audit_report:
        audit_path = _resolve_artifact_path(audit_report)
        payload["audit_report_exists"] = bool(audit_path and audit_path.exists())
        payload["audit_report_resolved_path"] = str(audit_path) if audit_path else ""
    payload["packaged_pair_side_supplement"] = True
    return payload


def _should_replace_intent_bidask_evidence_with_packaged(
    existing: dict[str, Any],
    packaged: dict[str, Any],
) -> bool:
    if not packaged.get("packaged_pair_side_supplement"):
        return False
    if packaged.get("audit_report_exists") is not True:
        return False
    if existing.get("packaged_pair_side_supplement"):
        return False
    if existing.get("audit_report_exists") is False:
        return True
    if not _first_str(existing.get("audit_report")):
        return True
    packaged_generated_at = _parse_utc(packaged.get("rule_set_generated_at_utc"))
    existing_generated_at = _parse_utc(existing.get("rule_set_generated_at_utc"))
    if packaged_generated_at is not None and (
        existing_generated_at is None or packaged_generated_at > existing_generated_at
    ):
        return True
    packaged_last_day = _parse_utc(packaged.get("last_day"))
    existing_last_day = _parse_utc(existing.get("last_day"))
    return packaged_last_day is not None and (
        existing_last_day is None or packaged_last_day > existing_last_day
    )


def _mark_bidask_negative_evidence_refresh(lane: dict[str, Any], *, now_utc: datetime) -> None:
    blockers = _string_list(lane.get("blockers"))
    if BIDASK_REPLAY_NEGATIVE_BLOCKER not in blockers:
        return
    evidence = lane.get("bidask_negative_evidence")
    if not isinstance(evidence, dict) or not evidence:
        return

    reasons: list[str] = []
    audit_report = _first_str(evidence.get("audit_report"))
    if (
        audit_report
        and evidence.get("audit_report_exists") is False
        and not evidence.get("packaged_pair_side_supplement")
    ):
        reasons.append("BIDASK_REPLAY_AUDIT_REPORT_MISSING")
    if not audit_report:
        reasons.append("BIDASK_REPLAY_AUDIT_REPORT_NOT_DECLARED")

    generated_at = _parse_utc(evidence.get("rule_set_generated_at_utc"))
    if generated_at is None:
        reasons.append("BIDASK_REPLAY_RULE_SET_TIMESTAMP_MISSING")
    elif now_utc - generated_at > timedelta(hours=BIDASK_REPLAY_NEGATIVE_MAX_AGE_HOURS):
        reasons.append("BIDASK_REPLAY_RULE_SET_STALE")

    last_day = _parse_utc(evidence.get("last_day"))
    if last_day is None:
        reasons.append("BIDASK_REPLAY_LAST_DAY_MISSING")
    elif (now_utc.date() - last_day.date()).days > BIDASK_REPLAY_NEGATIVE_LAST_DAY_MAX_AGE_DAYS:
        if _bidask_price_truth_complete(evidence) and not reasons:
            evidence["last_day_refresh_bypassed_by_price_truth_coverage"] = True
            evidence["last_day_refresh_bypass_reason"] = (
                "PRICE_TRUTH_OK with zero missing bid/ask truth; stale last_day means no newer "
                "matching negative-rule samples were observed in the refreshed source report."
            )
        else:
            reasons.append("BIDASK_REPLAY_LAST_DAY_STALE")

    if not reasons:
        evidence["refresh_required"] = False
        evidence["refresh_status"] = "CURRENT"
        return

    evidence["refresh_required"] = True
    evidence["refresh_status"] = "REQUIRED"
    evidence["refresh_max_age_hours"] = BIDASK_REPLAY_NEGATIVE_MAX_AGE_HOURS
    evidence["refresh_last_day_max_age_days"] = BIDASK_REPLAY_NEGATIVE_LAST_DAY_MAX_AGE_DAYS
    lane["evidence_refresh_reasons"] = _unique(_string_list(lane.get("evidence_refresh_reasons")) + reasons)
    if BIDASK_REPLAY_EVIDENCE_REFRESH_BLOCKER not in blockers:
        lane["blockers"] = blockers + [BIDASK_REPLAY_EVIDENCE_REFRESH_BLOCKER]


def _bidask_price_truth_complete(evidence: dict[str, Any]) -> bool:
    truth = evidence.get("price_truth_coverage")
    if not isinstance(truth, dict):
        return False
    status = str(truth.get("status") or "").upper()
    if status != "PRICE_TRUTH_OK":
        return False
    missing_samples = _float(truth.get("missing_price_truth_samples"))
    missing_groups = _float(truth.get("missing_price_window_group_count"))
    if missing_samples is not None and missing_samples > 0:
        return False
    if missing_groups is not None and missing_groups > 0:
        return False
    return True


def _bidask_negative_evidence_refresh_required(lane: dict[str, Any]) -> bool:
    if BIDASK_REPLAY_EVIDENCE_REFRESH_BLOCKER in _string_list(lane.get("blockers")):
        return True
    evidence = lane.get("bidask_negative_evidence")
    return isinstance(evidence, dict) and evidence.get("refresh_required") is True


def _resolve_artifact_path(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


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
    if has_negative and _bidask_negative_evidence_refresh_required(lane):
        return "EVIDENCE_ACQUISITION"
    if has_negative and _local_tp_proof_acquisition_required(lane):
        return "EVIDENCE_ACQUISITION"
    if _tp_proven_harvest_repair_required(lane):
        return "EVIDENCE_ACQUISITION"
    if has_negative and _edge_improvement_evidence_required(lane):
        return "EVIDENCE_ACQUISITION"
    if _entry_drought_recovery_required(lane):
        return "EVIDENCE_ACQUISITION"
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


def _local_tp_proof_acquisition_required(lane: dict[str, Any]) -> bool:
    blockers = _string_list(lane.get("blockers"))
    if str(lane.get("vehicle") or "").upper() == "MARKET":
        return False
    proof = lane.get("local_tp_proof")
    if not isinstance(proof, dict):
        return False
    if TP_PROVEN_ROTATION_BLOCKER not in blockers and BROAD_TP_PROOF_NOT_EXACT_VEHICLE_BLOCKER not in blockers:
        current_trades = _float(proof.get("capture_take_profit_trades"))
        if current_trades is None or current_trades <= 0:
            return False
    if proof.get("attach_take_profit_on_fill") is not True:
        return False
    if str(proof.get("tp_execution_mode") or "") != "ATTACHED_TECHNICAL_TP":
        return False
    if str(proof.get("tp_target_intent") or "") != "HARVEST":
        return False
    trades = _float(proof.get("capture_take_profit_trades"))
    if trades is not None:
        if trades <= 0:
            return False
        losses = _float(proof.get("capture_take_profit_losses"))
        expectancy = _float(proof.get("capture_take_profit_expectancy_jpy"))
        floor = _float(proof.get("capture_take_profit_proof_floor")) or float(TP_PROOF_COLLECTION_MIN_TRADES)
        if losses is None or losses > 0:
            return False
        if expectancy is None or expectancy <= 0:
            return False
        return trades < floor
    scope = str(proof.get("capture_take_profit_scope") or "")
    if scope.startswith("MISSING") or scope in {"", "None", "UNKNOWN"}:
        return True
    required_numeric = (
        "capture_take_profit_expectancy_jpy",
        "capture_take_profit_losses",
        "capture_take_profit_trades",
        "capture_take_profit_wins",
    )
    return any(proof.get(key) is None for key in required_numeric)


def _tp_proven_harvest_repair_required(lane: dict[str, Any]) -> bool:
    if lane.get("edge_improvement_candidate") is not True:
        return False
    if lane.get("tp_proven_harvest_repair_target") != TP_PROVEN_HARVEST_REPAIR_TARGET:
        return False
    return _local_tp_proof_floor_met(lane)


def _edge_improvement_evidence_required(lane: dict[str, Any]) -> bool:
    if lane.get("edge_improvement_candidate") is not True:
        return False
    if not lane.get("can_enter_proof_pack"):
        return False
    if not _has_evidence_path(lane):
        return False
    return _has_positive_harvest_or_edge_evidence(lane)


def _has_positive_harvest_or_edge_evidence(lane: dict[str, Any]) -> bool:
    proof = lane.get("local_tp_proof")
    if isinstance(proof, dict):
        trades = _float(proof.get("capture_take_profit_trades"))
        losses = _float(proof.get("capture_take_profit_losses"))
        expectancy = _float(proof.get("capture_take_profit_expectancy_jpy"))
        if (
            trades is not None
            and trades >= TP_PROOF_COLLECTION_MIN_TRADES
            and (losses is None or losses == 0)
            and expectancy is not None
            and expectancy > 0
        ):
            return True
    edge = _float(lane.get("expected_edge_jpy"))
    return edge is not None and edge > 0


def _entry_drought_recovery_required(lane: dict[str, Any]) -> bool:
    if lane.get("entry_recovery_candidate") is not True:
        return False
    history = lane.get("entry_recovery_history")
    if not isinstance(history, dict):
        return False
    accepted = _first_int(history.get("accepted_before_recent"), 0)
    fills = _first_int(history.get("fills_before_recent"), 0)
    closed_pl = _float(history.get("closed_pl_jpy")) or 0.0
    closed_trades = _first_int(history.get("closed_trades"), 0)
    recent_accepted = _first_int(history.get("recent_accepted"), 0)
    recent_fills = _first_int(history.get("recent_fills"), 0)
    if history.get("profit_source") == "pair_side_fallback" and not lane.get("order_intent_status"):
        return False
    if accepted < ENTRY_DROUGHT_MIN_ACCEPTED and fills < ENTRY_DROUGHT_MIN_FILLS:
        return False
    if not _entry_recovery_profit_is_material(closed_pl, closed_trades):
        return False
    return recent_accepted == 0 and recent_fills == 0


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
    if _local_tp_proof_floor_met(lane):
        score += 80.0
    elif _local_tp_proof_positive_partial(lane):
        score += 25.0
    if lane.get("can_enter_proof_pack"):
        score += 25.0
    if lane.get("entry_recovery_candidate") is True:
        history = lane.get("entry_recovery_history") if isinstance(lane.get("entry_recovery_history"), dict) else {}
        profit_source = str(history.get("profit_source") or "")
        if profit_source == "exact_lane":
            score += min((_float(history.get("closed_pl_jpy")) or 0.0) / 250.0, 35.0)
            score += min((_float(history.get("accepted_before_recent")) or 0.0) * 2.0, 20.0)
        else:
            score += min((_float(history.get("accepted_before_recent")) or 0.0), 8.0)
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


def _rank_sort_key(lane: dict[str, Any]) -> tuple[Any, ...]:
    blockers = lane.get("blockers") or []
    return (
        -float(STATUS_PRIORITY.get(str(lane.get("status")), 0)),
        _tp_proof_sort_bucket(lane),
        _current_intent_sort_bucket(lane),
        _entry_recovery_sort_bucket(lane),
        _vehicle_sort_bucket(lane),
        _live_unlock_blocker_sort_bucket(lane),
        len(blockers),
        -float(lane.get("_rank_score") or 0.0),
        str(lane.get("pair")),
        str(lane.get("direction")),
        str(lane.get("strategy_family")),
        str(lane.get("vehicle")),
    )


def _tp_proof_sort_bucket(lane: dict[str, Any]) -> int:
    if _local_tp_proof_floor_met(lane):
        return 0
    proof = lane.get("local_tp_proof")
    if not isinstance(proof, dict):
        return 3
    trades = _float(proof.get("capture_take_profit_trades"))
    losses = _float(proof.get("capture_take_profit_losses"))
    expectancy = _float(proof.get("capture_take_profit_expectancy_jpy"))
    if trades is not None and trades > 0 and (losses is None or losses == 0) and expectancy is not None and expectancy > 0:
        if trades >= TP_PROOF_COLLECTION_MIN_TRADES:
            return 1
        return 2
    return 3


def _local_tp_proof_positive_partial(lane: dict[str, Any]) -> bool:
    proof = lane.get("local_tp_proof")
    if not isinstance(proof, dict):
        return False
    trades = _float(proof.get("capture_take_profit_trades"))
    losses = _float(proof.get("capture_take_profit_losses"))
    expectancy = _float(proof.get("capture_take_profit_expectancy_jpy"))
    return (
        trades is not None
        and trades > 0
        and (losses is None or losses == 0)
        and expectancy is not None
        and expectancy > 0
    )


def _entry_recovery_sort_bucket(lane: dict[str, Any]) -> int:
    if lane.get("entry_recovery_candidate") is not True:
        return 0
    history = lane.get("entry_recovery_history") if isinstance(lane.get("entry_recovery_history"), dict) else {}
    if history.get("profit_source") == "exact_lane":
        return 0
    if not lane.get("order_intent_status"):
        return 4
    return 2


def _live_unlock_blocker_sort_bucket(lane: dict[str, Any]) -> int:
    blockers = lane.get("blockers") or []
    if _has_marker(blockers, FAILED_EXACT_REPLAY_MARKERS):
        return 4
    if LOCAL_TP_PROOF_ZERO_TRADES_BLOCKER in blockers:
        return 3
    if _has_marker(blockers, NEGATIVE_BLOCKER_MARKERS):
        return 2
    if lane.get("vehicle") == "UNKNOWN" or "NO_CURRENT_EXECUTABLE_INTENT" in blockers:
        return 5
    return 0


def _vehicle_sort_bucket(lane: dict[str, Any]) -> int:
    vehicle = str(lane.get("vehicle") or "").upper()
    if vehicle == "LIMIT":
        return 0
    if vehicle in {"STOP", "STOP-ENTRY", "STOP_ENTRY"}:
        return 1
    if vehicle == "MARKET":
        return 2
    return 3


def _current_intent_sort_bucket(lane: dict[str, Any]) -> int:
    blockers = lane.get("blockers") or []
    if lane.get("order_intent_status") and lane.get("vehicle") != "UNKNOWN":
        return 0
    if "NO_CURRENT_EXECUTABLE_INTENT" in blockers:
        return 2
    if lane.get("vehicle") == "UNKNOWN":
        return 3
    return 1


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
    if _bidask_negative_evidence_refresh_required(lane):
        return "NEGATIVE_EVIDENCE_REFRESH_REQUIRED"
    if _has_marker(blockers, ("BIDASK_REPLAY_NEGATIVE",)):
        return "NEGATIVE"
    return status


def _lane_next_action(lane: dict[str, Any]) -> str:
    status = lane.get("status")
    lane_key = _lane_key(lane)
    blockers = _string_list(lane.get("blockers"))
    if _has_marker(lane.get("blockers") or [], FAILED_EXACT_REPLAY_MARKERS):
        return (
            f"No trade for {lane_key}; consume the failed exact replay as not SCOUT-ready, "
            "wait for new independent trigger/TP-path evidence, and do not repeat the same exact replay."
        )
    if status == "LIVE_READY":
        return f"Keep {lane_key} visible for verifier/gateway checks only; this board grants no live permission."
    if status == "EVIDENCE_ACQUISITION":
        if lane.get("edge_improvement_candidate") is True:
            target = lane.get("edge_improvement_target") or lane_key
            return (
                f"Run read-only EDGE_IMPROVEMENT_EXPERIMENT for {target}; preserve negative/month-scale blockers, "
                "canonicalize proof/replay/sample gaps, rerank, and do not send."
            )
        if lane.get("entry_recovery_candidate") is True:
            history = lane.get("entry_recovery_history") if isinstance(lane.get("entry_recovery_history"), dict) else {}
            accepted = history.get("accepted_before_recent")
            fills = history.get("fills_before_recent")
            pl = history.get("closed_pl_jpy")
            source = history.get("profit_source") or "execution_ledger"
            if source == "pair_side_fallback":
                return (
                    f"Map historical pair/side recovery profit back to an exact lane for {lane_key}; historical "
                    f"accepted={accepted}, fills={fills}, pair_side_closed_pl_jpy={pl}, but the profitable closes "
                    "were not attributed to this lane. Keep every current blocker visible, mine the tagless/manual "
                    "winner pattern, rebuild exact proof/replay for the mapped lane, and do not send."
                )
            return (
                f"Run entry-frequency recovery analysis for {lane_key}; historical accepted={accepted}, "
                f"fills={fills}, closed_pl_jpy={pl} ({source}) but recent entries are zero. Re-tune forecast/pattern "
                "selection and bid/ask or local-TP proof for this lane while preserving every current blocker. Do not send."
            )
        if _bidask_negative_evidence_refresh_required(lane):
            reasons = ", ".join(_string_list(lane.get("evidence_refresh_reasons"))[:3])
            suffix = f" ({reasons})" if reasons else ""
            return (
                f"Refresh exact S5 bid/ask replay evidence for {lane_key}{suffix}; "
                "keep the negative blocker visible, rebuild/package bidask replay precision rules, and do not send."
            )
        if _local_tp_proof_acquisition_required(lane):
            proof = lane.get("local_tp_proof") if isinstance(lane.get("local_tp_proof"), dict) else {}
            scope_key = proof.get("capture_take_profit_scope_key") or lane_key
            return (
                f"Collect exact local TAKE_PROFIT_ORDER proof for {scope_key}; require positive expectancy, "
                "zero TP losses, and positive Wilson-stressed expectancy before reranking. Do not send."
            )
        return f"Acquire or canonicalize proof/replay evidence for {lane_key}; do not send or mix vehicles."
    if status == "SCOUT_READY":
        return f"Prepare read-only SCOUT judgement material for {lane_key}; gateway permission remains false."
    if status == "HARVEST_READY":
        return f"Run HARVEST readiness checks for {lane_key} while preserving negative/replay blockers."
    if status == "OPERATOR_REVIEW_REQUIRED":
        if lane.get("edge_improvement_candidate") is True:
            target = lane.get("edge_improvement_target") or lane_key
            return (
                f"Package guardian receipt operator-review evidence for {lane_key}; do not infer approval. "
                f"After review clears, run read-only EDGE_IMPROVEMENT_EXPERIMENT for {target}; preserve "
                "bid/ask, forecast, risk, and profitability blockers, rerank, and do not send."
            )
        if _has_marker(lane.get("blockers") or [], GUARDIAN_RECEIPT_OPERATOR_REVIEW_MARKERS):
            return f"Package guardian receipt operator-review evidence for {lane_key}; do not infer approval."
        return f"Package operator/guardian review evidence for {lane_key}; do not infer approval."
    if LOCAL_TP_PROOF_ZERO_TRADES_BLOCKER in blockers:
        proof = lane.get("local_tp_proof") if isinstance(lane.get("local_tp_proof"), dict) else {}
        floor = proof.get("capture_take_profit_proof_floor") or 20
        scope_key = proof.get("capture_take_profit_scope_key") or lane_key
        return (
            f"No trade for {lane_key}; exact local TAKE_PROFIT_ORDER proof for {scope_key} is 0/{floor}. "
            "Wait for new local TP receipts or an explicitly approved proof-collection scout, then rerank."
        )
    if LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR_BLOCKER in blockers:
        proof = lane.get("local_tp_proof") if isinstance(lane.get("local_tp_proof"), dict) else {}
        trades = proof.get("capture_take_profit_trades") or 0
        return (
            f"No trade for {lane_key}; local TAKE_PROFIT_ORDER proof has only {trades} sample(s), below "
            f"the {TP_PROOF_COLLECTION_MIN_TRADES}-trade proof-collection floor. Preserve the blocker and rerank."
        )
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
    if lane.get("evidence_refresh_reasons"):
        public["evidence_refresh_reasons"] = _unique(_string_list(lane.get("evidence_refresh_reasons")))
    if isinstance(lane.get("bidask_negative_evidence"), dict):
        public["bidask_negative_evidence"] = lane["bidask_negative_evidence"]
    if isinstance(lane.get("local_tp_proof"), dict):
        public["local_tp_proof"] = lane["local_tp_proof"]
    if lane.get("edge_improvement_candidate") is True:
        public["edge_improvement_candidate"] = True
        public["edge_improvement_target"] = str(lane.get("edge_improvement_target") or "")
    if lane.get("tp_proven_harvest_repair_target"):
        public["tp_proven_harvest_repair_target"] = str(lane.get("tp_proven_harvest_repair_target") or "")
    if lane.get("entry_recovery_candidate") is True:
        public["entry_recovery_candidate"] = True
        if isinstance(lane.get("entry_recovery_history"), dict):
            public["entry_recovery_history"] = lane["entry_recovery_history"]
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
        "entry_recovery_candidate_count": sum(1 for lane in lanes if lane.get("entry_recovery_candidate") is True),
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


def _entry_recovery_summary(lanes: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [lane for lane in lanes if lane.get("entry_recovery_candidate") is True]
    ranked = sorted(
        candidates,
        key=lambda lane: (
            -float(((lane.get("entry_recovery_history") or {}).get("closed_pl_jpy") or 0.0)),
            -float(((lane.get("entry_recovery_history") or {}).get("accepted_before_recent") or 0.0)),
            str(lane.get("lane_id") or ""),
        ),
    )
    return {
        "candidate_count": len(candidates),
        "top_candidates": [
            {
                "lane_id": lane.get("lane_id"),
                "pair": lane.get("pair"),
                "direction": lane.get("direction"),
                "strategy_family": lane.get("strategy_family"),
                "vehicle": lane.get("vehicle"),
                "status": lane.get("status"),
                "blockers": lane.get("blockers", [])[:8],
                "history": lane.get("entry_recovery_history") or {},
            }
            for lane in ranked[:8]
        ],
    }


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
            f"- Entry recovery candidates: `{summary.get('entry_recovery_candidate_count')}`",
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
    entry_recovery = payload.get("entry_recovery_summary") if isinstance(payload.get("entry_recovery_summary"), dict) else {}
    if entry_recovery.get("candidate_count"):
        lines.extend(["", "## Entry Recovery", ""])
        for lane in (entry_recovery.get("top_candidates") or [])[:5]:
            history = lane.get("history") if isinstance(lane.get("history"), dict) else {}
            lines.append(
                f"- `{lane.get('lane_id')}`: accepted `{history.get('accepted_before_recent')}`, "
                f"fills `{history.get('fills_before_recent')}`, closed_pl_jpy `{history.get('closed_pl_jpy')}`, "
                f"blockers `{', '.join(lane.get('blockers') or [])}`"
            )
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
    if text in {"STOP-ENTRY", "STOP_ENTRY", "STOP_ORDER"}:
        return "STOP"
    if text == "LIMIT_ORDER":
        return "LIMIT"
    if text == "MARKET_ORDER":
        return "MARKET"
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


def _blocker_object_codes(value: Any) -> list[str]:
    codes: list[str] = []
    for item in _list(value):
        if not isinstance(item, dict):
            continue
        if "severity" in item and str(item.get("severity") or "").upper() != "BLOCK":
            continue
        code = item.get("code") or item.get("row_code") or item.get("blocker")
        if code:
            codes.append(str(code))
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


def _artifact_is_older_than(source: dict[str, Any], newer: dict[str, Any]) -> bool:
    source_generated = _parse_utc(source.get("generated_at_utc") or source.get("fetched_at_utc"))
    newer_generated = _parse_utc(newer.get("generated_at_utc") or newer.get("fetched_at_utc"))
    if source_generated is None or newer_generated is None:
        return False
    return source_generated < newer_generated


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


def _first_int(value: Any, default: int) -> int:
    number = _float(value)
    if number is None:
        return default
    return int(number)


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

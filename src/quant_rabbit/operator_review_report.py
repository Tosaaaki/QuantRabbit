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
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
    DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
    DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER,
    DEFAULT_NON_EURUSD_PROOF_LANE_MAPPER,
    DEFAULT_OPERATOR_REVIEW_REPORT,
    DEFAULT_OPERATOR_REVIEW_REPORT_MD,
    DEFAULT_PREDICTIVE_SCOUT_FORWARD_PROOF,
    DEFAULT_QR_TRADER_RUN_WATCHDOG,
)


REPORT_VERSION = "operator_review_report_v1"

STATUS_DATA_INCOMPLETE = "OPERATOR_REVIEW_DATA_INCOMPLETE"
STATUS_NOT_SELECTED = "OPERATOR_REVIEW_NOT_SELECTED"
STATUS_STILL_BLOCKED = "OPERATOR_REVIEW_STILL_BLOCKED"
STATUS_CLEARED_OTHER_BLOCKERS_REMAIN = "OPERATOR_REVIEW_CLEARED_OTHER_BLOCKERS_REMAIN"
STATUS_MATERIAL_READY = "OPERATOR_REVIEW_MATERIAL_READY_NO_LIVE_PERMISSION"

OPERATOR_REVIEW_BLOCKERS = {
    "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
    "GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW",
    "OPERATOR_REVIEW_REQUIRED",
}

DO_NOT_DO = [
    "live order",
    "cancel order",
    "close position",
    "modify TP/SL",
    "launchd change",
    "gate relaxation",
    "negative expectancy suppression",
    "spread too wide suppression",
    "bidask negative suppression",
    "market-close loss as TP proof",
    "4x-deficit lot backsolve",
    "secret disclosure",
    "operator decision inference",
]


@dataclass(frozen=True)
class OperatorReviewReportSummary:
    status: str
    output_path: Path
    report_path: Path
    target_shape: str | None
    live_permission_allowed: bool


class OperatorReviewReport:
    """Build read-only operator review material for the current active lane.

    This report deliberately does not write `guardian_receipt_operator_review`.
    That file is the explicit operator decision artifact; this one is only the
    current evidence bundle the operator/trader should look at.
    """

    def __init__(
        self,
        *,
        active_trader_contract_path: Path = DEFAULT_ACTIVE_TRADER_CONTRACT,
        active_opportunity_board_path: Path = DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
        non_eurusd_live_grade_frontier_path: Path = DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER,
        non_eurusd_proof_lane_mapper_path: Path = DEFAULT_NON_EURUSD_PROOF_LANE_MAPPER,
        guardian_receipt_consumption_path: Path = DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
        guardian_receipt_operator_review_path: Path = DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
        qr_trader_run_watchdog_path: Path = DEFAULT_QR_TRADER_RUN_WATCHDOG,
        broker_snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        predictive_scout_forward_proof_path: Path = DEFAULT_PREDICTIVE_SCOUT_FORWARD_PROOF,
        output_path: Path = DEFAULT_OPERATOR_REVIEW_REPORT,
        report_path: Path = DEFAULT_OPERATOR_REVIEW_REPORT_MD,
        now_utc: datetime | None = None,
    ) -> None:
        self.paths = {
            "active_trader_contract": active_trader_contract_path,
            "active_opportunity_board": active_opportunity_board_path,
            "non_eurusd_live_grade_frontier": non_eurusd_live_grade_frontier_path,
            "non_eurusd_proof_lane_mapper": non_eurusd_proof_lane_mapper_path,
            "guardian_receipt_consumption": guardian_receipt_consumption_path,
            "guardian_receipt_operator_review": guardian_receipt_operator_review_path,
            "qr_trader_run_watchdog": qr_trader_run_watchdog_path,
            "broker_snapshot": broker_snapshot_path,
            "predictive_scout_forward_proof": predictive_scout_forward_proof_path,
        }
        self.output_path = output_path
        self.report_path = report_path
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> OperatorReviewReportSummary:
        payload = self.build_payload()
        if payload.get("read_only") is not True:
            raise ValueError("operator review report must be read-only")
        if payload.get("operator_review_material_only") is not True:
            raise ValueError("operator review report must remain material-only")
        if payload.get("live_permission_allowed") is not False:
            raise ValueError("operator review report must never grant live permission")
        if payload.get("live_side_effects") != []:
            raise ValueError("operator review report must not record live side effects")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(render_operator_review_report(payload), encoding="utf-8")
        return OperatorReviewReportSummary(
            status=str(payload["status"]),
            output_path=self.output_path,
            report_path=self.report_path,
            target_shape=payload.get("target_shape"),
            live_permission_allowed=False,
        )

    def build_payload(self) -> dict[str, Any]:
        artifacts = {name: _load_json_artifact(path) for name, path in self.paths.items()}
        contract = artifacts["active_trader_contract"]
        board = artifacts["active_opportunity_board"]
        frontier = artifacts["non_eurusd_live_grade_frontier"]
        consumption = artifacts["guardian_receipt_consumption"]
        guardian_review = artifacts["guardian_receipt_operator_review"]
        watchdog = artifacts["qr_trader_run_watchdog"]
        scout_proof = artifacts["predictive_scout_forward_proof"]

        board_top = _top_lane_from_board(board)
        contract_top = _top_lane_from_contract(contract)
        target_lane = _compact_lane(board_top or contract_top)
        target_shape = _lane_shape(target_lane)
        blockers = _merged_blockers(contract, target_lane)
        blocker_codes = [str(row.get("code")) for row in blockers if row.get("code")]
        operator_review_required = _operator_review_required(contract, target_lane, blocker_codes, consumption)
        guardian_state = _guardian_receipt_state(consumption, guardian_review)
        status = _status(
            artifacts=artifacts,
            selected_active_path=contract.get("selected_active_path"),
            operator_review_required=operator_review_required,
            guardian_state=guardian_state,
            blockers=blocker_codes,
        )
        success_eval = _success_condition_evaluation(
            contract=contract,
            target_lane=target_lane,
            board_top=_compact_lane(board_top),
            guardian_state=guardian_state,
        )
        payload = {
            "schema_version": REPORT_VERSION,
            "status": status,
            "generated_at_utc": self.now_utc.isoformat(),
            "read_only": True,
            "operator_review_material_only": True,
            "live_side_effects": [],
            "live_permission_allowed": False,
            "target_shape": target_shape,
            "target_lane": target_lane,
            "top_lane": _compact_lane(board_top),
            "top_non_eurusd_lane": _compact_lane(frontier.get("top_non_eurusd_lane")),
            "selected_active_path": contract.get("selected_active_path"),
            "operator_review_required": operator_review_required,
            "guardian_receipt_id": guardian_state.get("receipt_event_id"),
            "guardian_receipt_state": guardian_state,
            "current_guardian_consumption": _guardian_summary(consumption),
            "guardian_operator_review": _guardian_summary(guardian_review),
            "watchdog_state": _watchdog_summary(watchdog),
            "blockers": blockers,
            "proof_summary": _proof_summary(target_lane),
            "predictive_scout_forward_proof": _predictive_scout_proof_summary(
                scout_proof
            ),
            "risk_summary": _risk_summary(target_lane, frontier, contract),
            "recommendation_summary": _recommendation_summary(
                status=status,
                target_lane=target_lane,
                contract=contract,
                guardian_state=guardian_state,
                blockers=blocker_codes,
            ),
            "success_condition": _success_condition(),
            "success_condition_evaluation": success_eval,
            "do_not_do": list(DO_NOT_DO),
            "source_artifacts": {name: _source_summary(artifact) for name, artifact in artifacts.items()},
        }
        return payload


def render_operator_review_report(payload: dict[str, Any]) -> str:
    lane = payload.get("target_lane") if isinstance(payload.get("target_lane"), dict) else {}
    guardian = payload.get("guardian_receipt_state") if isinstance(payload.get("guardian_receipt_state"), dict) else {}
    proof = payload.get("proof_summary") if isinstance(payload.get("proof_summary"), dict) else {}
    scout_proof = (
        payload.get("predictive_scout_forward_proof")
        if isinstance(payload.get("predictive_scout_forward_proof"), dict)
        else {}
    )
    risk = payload.get("risk_summary") if isinstance(payload.get("risk_summary"), dict) else {}
    recommendation = (
        payload.get("recommendation_summary") if isinstance(payload.get("recommendation_summary"), dict) else {}
    )
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    lines = [
        "# Operator Review Report",
        "",
        f"- Generated UTC: `{payload.get('generated_at_utc')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Target shape: `{payload.get('target_shape')}`",
        f"- Selected active path: `{payload.get('selected_active_path')}`",
        f"- Read only: `{payload.get('read_only')}`",
        f"- Live permission allowed: `{payload.get('live_permission_allowed')}`",
        f"- Live side effects: `{len(payload.get('live_side_effects') or [])}`",
        "",
        "## Current Lane",
        "",
        f"- Lane id: `{lane.get('lane_id')}`",
        f"- Pair/side/strategy/vehicle: `{lane.get('pair')}` `{lane.get('direction')}` `{lane.get('strategy_family')}` `{lane.get('vehicle')}`",
        f"- Lane status: `{lane.get('status')}`",
        f"- Next action: {lane.get('next_action') or recommendation.get('next_action') or 'none'}",
        "",
        "## Guardian Receipt",
        "",
        f"- Receipt event: `{guardian.get('receipt_event_id')}`",
        f"- Action/lifecycle: `{guardian.get('receipt_action')}` / `{guardian.get('receipt_lifecycle')}`",
        f"- Consumption status: `{guardian.get('consumption_status')}`",
        f"- Consumption normal routing allowed: `{guardian.get('consumption_normal_routing_allowed')}`",
        f"- Operator review status: `{guardian.get('operator_review_status')}`",
        f"- Operator review normal routing allowed: `{guardian.get('operator_review_normal_routing_allowed')}`",
        "",
        "## Proof And Risk",
        "",
        f"- TP proof: `{proof.get('tp_proof_count')}` / `{proof.get('tp_proof_floor')}`",
        f"- TP losses: `{proof.get('tp_losses')}`",
        f"- Expected edge JPY: `{proof.get('expected_edge_jpy')}`",
        f"- Risk status: `{risk.get('risk_status')}`",
        f"- Replay status: `{risk.get('replay_status')}`",
        f"- Spread status: `{risk.get('spread_status')}`",
        f"- Bid/ask status: `{risk.get('bidask_status')}`",
        f"- Predictive SCOUT proof status: `{scout_proof.get('status')}`",
        f"- Predictive SCOUT eligible vehicles: `{scout_proof.get('eligible_vehicle_count')}`",
        f"- Predictive SCOUT automatic promotion allowed: `{scout_proof.get('promotion_allowed')}`",
        "",
        "## Blockers",
        "",
    ]
    if not blockers:
        lines.append("- none")
    else:
        for blocker in blockers:
            if not isinstance(blocker, dict):
                continue
            lines.append(f"- `{blocker.get('code')}` ({blocker.get('source')})")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Recommended operator action: `{recommendation.get('recommended_operator_action')}`",
            f"- Next active path after review: `{recommendation.get('after_operator_review_next_path')}`",
            f"- Reason: {recommendation.get('reason')}",
            "",
            "## Safety Boundary",
            "",
        ]
    )
    for item in payload.get("do_not_do") or []:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _load_json_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_artifact_status": "missing", "_path": str(path), "_sha256": None}
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    result = dict(payload)
    result["_artifact_status"] = "present"
    result["_path"] = str(path)
    result["_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


def _source_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_status": artifact.get("_artifact_status"),
        "path": artifact.get("_path"),
        "sha256": artifact.get("_sha256"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "status": artifact.get("status"),
        "normal_routing_allowed": artifact.get("normal_routing_allowed"),
        "live_permission_allowed": artifact.get("live_permission_allowed"),
    }


def _predictive_scout_proof_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    vehicles = [
        vehicle
        for vehicle in artifact.get("vehicles", []) or []
        if isinstance(vehicle, dict)
    ]
    eligible = [
        vehicle
        for vehicle in vehicles
        if vehicle.get("statistically_eligible_for_operator_review") is True
    ]
    return {
        "artifact_status": artifact.get("_artifact_status"),
        "generated_at_utc": artifact.get("generated_at_utc"),
        "status": artifact.get("status"),
        "promotion_allowed": False,
        "future_profit_guaranteed": False,
        "vehicle_count": len(vehicles),
        "eligible_vehicle_count": len(eligible),
        "eligible_vehicles": [
            {
                "predictive_scout_vehicle_id": vehicle.get(
                    "predictive_scout_vehicle_id"
                ),
                "pair": vehicle.get("pair"),
                "side": vehicle.get("side"),
                "resolved_count": vehicle.get("resolved_count"),
                "net_jpy": vehicle.get("net_jpy"),
                "profit_factor": vehicle.get("profit_factor"),
                "one_sided_95_mean_lower_jpy": vehicle.get(
                    "one_sided_95_mean_lower_jpy"
                ),
                "positive_day_rate": vehicle.get("positive_day_rate"),
                "duplicate_signal_count": vehicle.get("duplicate_signal_count"),
            }
            for vehicle in eligible
        ],
        "requirements": artifact.get("requirements"),
    }


def _top_lane_from_board(board: dict[str, Any]) -> dict[str, Any]:
    top = board.get("top_lane")
    if isinstance(top, dict):
        return top
    lanes = board.get("ranked_active_lanes")
    if isinstance(lanes, list) and lanes and isinstance(lanes[0], dict):
        return lanes[0]
    return {}


def _top_lane_from_contract(contract: dict[str, Any]) -> dict[str, Any]:
    current = contract.get("current_state")
    if not isinstance(current, dict):
        return {}
    active_board = current.get("active_opportunity_board")
    if not isinstance(active_board, dict):
        return {}
    top = active_board.get("top_lane")
    return top if isinstance(top, dict) else {}


def _compact_lane(lane: Any) -> dict[str, Any]:
    if not isinstance(lane, dict):
        return {}
    proof = lane.get("local_tp_proof") if isinstance(lane.get("local_tp_proof"), dict) else {}
    bidask = lane.get("bidask_negative_evidence") if isinstance(lane.get("bidask_negative_evidence"), dict) else {}
    return {
        "lane_id": lane.get("lane_id"),
        "pair": lane.get("pair"),
        "direction": lane.get("direction") or lane.get("side"),
        "strategy_family": lane.get("strategy_family"),
        "vehicle": lane.get("vehicle"),
        "status": lane.get("status"),
        "blockers": _string_list(lane.get("blockers")),
        "stale_source_blockers": _string_list(lane.get("stale_source_blockers")),
        "next_action": lane.get("next_action"),
        "expected_edge_jpy": lane.get("expected_edge_jpy"),
        "risk_status": lane.get("risk_status"),
        "proof_status": lane.get("proof_status"),
        "replay_status": lane.get("replay_status"),
        "spread_status": lane.get("spread_status"),
        "guardian_status": lane.get("guardian_status"),
        "operator_review_status": lane.get("operator_review_status"),
        "rank_score": lane.get("rank_score"),
        "live_permission_allowed": lane.get("live_permission_allowed", False),
        "local_tp_proof": {
            "tp_execution_mode": proof.get("tp_execution_mode"),
            "tp_target_intent": proof.get("tp_target_intent"),
            "tp_target_source": proof.get("tp_target_source"),
            "capture_take_profit_scope_key": proof.get("capture_take_profit_scope_key"),
            "capture_take_profit_trades": proof.get("capture_take_profit_trades"),
            "capture_take_profit_wins": proof.get("capture_take_profit_wins"),
            "capture_take_profit_losses": proof.get("capture_take_profit_losses"),
            "capture_take_profit_expectancy_jpy": proof.get("capture_take_profit_expectancy_jpy"),
            "capture_take_profit_proof_floor": proof.get("capture_take_profit_proof_floor"),
            "capture_take_profit_zero_trade": proof.get("capture_take_profit_zero_trade"),
        },
        "bidask_negative_evidence": {
            "name": bidask.get("name"),
            "pair": bidask.get("pair"),
            "side": bidask.get("side"),
            "samples": bidask.get("samples"),
            "directional_hit_rate": bidask.get("directional_hit_rate"),
            "positive_day_rate": bidask.get("positive_day_rate"),
            "refresh_required": bidask.get("refresh_required"),
            "refresh_status": bidask.get("refresh_status"),
            "audit_report_exists": bidask.get("audit_report_exists"),
            "audit_report_resolved_path": bidask.get("audit_report_resolved_path"),
        },
    }


def _lane_shape(lane: dict[str, Any]) -> str | None:
    pair = lane.get("pair")
    side = lane.get("direction")
    strategy = lane.get("strategy_family")
    vehicle = lane.get("vehicle")
    if not all((pair, side, strategy, vehicle)):
        return None
    return f"{pair}|{side}|{strategy}|{vehicle}"


def _merged_blockers(contract: dict[str, Any], lane: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(code: Any, source: str, status: Any = None, blocks_live_permission: Any = True) -> None:
        if not code:
            return
        text = str(code)
        key = (text, source)
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "code": text,
                "source": source,
                "status": status,
                "blocks_live_permission": blocks_live_permission is not False,
            }
        )

    for code in _string_list(lane.get("blockers")):
        add(code, "active_opportunity_board.top_lane")
    for row in contract.get("remaining_blockers") or []:
        if isinstance(row, dict):
            add(row.get("code"), "active_trader_contract.remaining_blockers", row.get("status"), row.get("blocks_live_permission"))
        else:
            add(row, "active_trader_contract.remaining_blockers")
    return rows


def _operator_review_required(
    contract: dict[str, Any],
    lane: dict[str, Any],
    blocker_codes: list[str],
    consumption: dict[str, Any],
) -> bool:
    if contract.get("selected_active_path") == "OPERATOR_REVIEW_REPORT":
        return True
    if lane.get("status") == "OPERATOR_REVIEW_REQUIRED":
        return True
    if any(code in OPERATOR_REVIEW_BLOCKERS for code in blocker_codes):
        return True
    if consumption.get("normal_routing_allowed") is False:
        return True
    return False


def _guardian_receipt_state(consumption: dict[str, Any], guardian_review: dict[str, Any]) -> dict[str, Any]:
    consumption_row = _first_dict(consumption.get("classifications"))
    review_row = _first_dict(guardian_review.get("classifications"))
    receipt_event_id = (
        consumption_row.get("receipt_event_id")
        or review_row.get("receipt_event_id")
        or consumption.get("receipt_event_id")
        or guardian_review.get("receipt_event_id")
    )
    return {
        "receipt_event_id": receipt_event_id,
        "receipt_action": consumption_row.get("receipt_action") or review_row.get("receipt_action"),
        "receipt_lifecycle": consumption_row.get("receipt_lifecycle") or review_row.get("receipt_lifecycle"),
        "classification": consumption_row.get("classification"),
        "consumption_status": consumption.get("status"),
        "consumption_normal_routing_allowed": consumption.get("normal_routing_allowed"),
        "consumption_unresolved_issue_count": consumption.get("unresolved_issue_count"),
        "operator_review_status": guardian_review.get("status"),
        "operator_review_normal_routing_allowed": guardian_review.get("normal_routing_allowed"),
        "operator_review_unresolved_count": guardian_review.get("unresolved_review_count"),
        "operator_review_decision": consumption_row.get("operator_review_decision") or review_row.get("operator_decision"),
        "operator_review_reason": consumption_row.get("operator_review_reason") or review_row.get("reason"),
        "operator_review_expires_at_utc": review_row.get("expires_at_utc"),
        "same_event_emergency_active": consumption_row.get("same_event_emergency_active")
        if "same_event_emergency_active" in consumption_row
        else review_row.get("same_event_emergency_active"),
    }


def _guardian_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_status": payload.get("_artifact_status"),
        "status": payload.get("status"),
        "normal_routing_allowed": payload.get("normal_routing_allowed"),
        "unresolved_issue_count": payload.get("unresolved_issue_count"),
        "unresolved_review_count": payload.get("unresolved_review_count"),
        "current_p0_p1_blocks_routing": payload.get("current_p0_p1_blocks_routing"),
        "classifications": [_compact_guardian_row(row) for row in payload.get("classifications") or [] if isinstance(row, dict)],
    }


def _compact_guardian_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "receipt_event_id": row.get("receipt_event_id"),
        "receipt_action": row.get("receipt_action"),
        "receipt_lifecycle": row.get("receipt_lifecycle"),
        "classification": row.get("classification"),
        "operator_decision": row.get("operator_decision") or row.get("operator_review_decision"),
        "operator_review_status": row.get("operator_review_status") or row.get("clearance_status"),
        "normal_routing_allowed": row.get("normal_routing_allowed"),
        "same_event_emergency_active": row.get("same_event_emergency_active"),
        "generated_at_utc": row.get("generated_at_utc"),
        "expires_at_utc": row.get("expires_at_utc"),
        "reason": row.get("reason") or row.get("operator_review_reason") or row.get("clearance_reason"),
    }


def _watchdog_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_status": payload.get("_artifact_status"),
        "status": payload.get("status"),
        "runtime_status": payload.get("runtime_status"),
        "issue_status": payload.get("issue_status"),
        "overall_status": payload.get("overall_status"),
        "last_trader_run_at": payload.get("last_trader_run_at"),
        "minutes_since_last_run": payload.get("minutes_since_last_run"),
        "issues": payload.get("issues") if isinstance(payload.get("issues"), list) else [],
    }


def _proof_summary(lane: dict[str, Any]) -> dict[str, Any]:
    proof = lane.get("local_tp_proof") if isinstance(lane.get("local_tp_proof"), dict) else {}
    return {
        "tp_proof_count": proof.get("capture_take_profit_trades"),
        "tp_proof_floor": proof.get("capture_take_profit_proof_floor"),
        "tp_wins": proof.get("capture_take_profit_wins"),
        "tp_losses": proof.get("capture_take_profit_losses"),
        "tp_expectancy_jpy": proof.get("capture_take_profit_expectancy_jpy"),
        "tp_scope_key": proof.get("capture_take_profit_scope_key"),
        "proof_status": lane.get("proof_status"),
        "expected_edge_jpy": lane.get("expected_edge_jpy"),
    }


def _risk_summary(lane: dict[str, Any], frontier: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk_status": lane.get("risk_status"),
        "replay_status": lane.get("replay_status"),
        "spread_status": lane.get("spread_status"),
        "bidask_status": _bidask_status(lane),
        "guardian_status": lane.get("guardian_status"),
        "operator_review_status": lane.get("operator_review_status"),
        "frontier_status": frontier.get("status"),
        "frontier_next_active_path": frontier.get("next_active_path"),
        "contract_live_permission_allowed": contract.get("live_permission_allowed"),
        "live_permission_allowed": False,
    }


def _bidask_status(lane: dict[str, Any]) -> str:
    bidask = lane.get("bidask_negative_evidence") if isinstance(lane.get("bidask_negative_evidence"), dict) else {}
    if bidask.get("refresh_required") is True:
        return "REFRESH_REQUIRED"
    blockers = _string_list(lane.get("blockers"))
    if any("BIDASK_REPLAY_NEGATIVE" in code for code in blockers):
        return "NEGATIVE"
    if any("BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED" in code for code in blockers):
        return "REFRESH_REQUIRED"
    return "UNKNOWN" if not bidask else "PRESENT"


def _recommendation_summary(
    *,
    status: str,
    target_lane: dict[str, Any],
    contract: dict[str, Any],
    guardian_state: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    if status == STATUS_STILL_BLOCKED:
        action = "KEEP_BLOCKED_PENDING_EXPLICIT_OPERATOR_REVIEW_AND_EVIDENCE_REFRESH"
    elif status == STATUS_NOT_SELECTED:
        action = "DO_NOT_REVIEW_UNLESS_ACTIVE_CONTRACT_SELECTS_OPERATOR_REVIEW_REPORT"
    else:
        action = "REVIEW_MATERIAL_ONLY_NO_LIVE_PERMISSION"
    return {
        "recommended_operator_action": action,
        "reason": _recommendation_reason(status, guardian_state, blockers),
        "next_action": target_lane.get("next_action") or contract.get("next_trade_enabling_action"),
        "after_operator_review_next_path": _after_review_next_path(blockers),
        "does_not_create_operator_decision": True,
        "does_not_create_live_permission": True,
    }


def _recommendation_reason(status: str, guardian_state: dict[str, Any], blockers: list[str]) -> str:
    if status == STATUS_STILL_BLOCKED:
        receipt = guardian_state.get("receipt_event_id") or "unknown"
        if guardian_state.get("consumption_normal_routing_allowed") is False:
            return (
                f"guardian receipt {receipt} still has normal_routing_allowed=false; "
                "operator review material is packaged but approval is not inferred."
            )
        return "operator review is required before normal routing, and other live blockers remain visible."
    if status == STATUS_NOT_SELECTED:
        return "active-trader-contract did not select OPERATOR_REVIEW_REPORT for the current cycle."
    if blockers:
        return "operator review no longer appears to be the only blocker; keep the remaining live blockers visible."
    return "material is ready for human review only; live permission remains false."


def _after_review_next_path(blockers: list[str]) -> str:
    if any("BIDASK_REPLAY" in code for code in blockers):
        return "BIDASK_REPLAY_EVIDENCE_REFRESH"
    if any("LOCAL_TP_PROOF" in code or "TP_PROOF" in code for code in blockers):
        return "TP_PROOF_COLLECTION"
    if any("NEGATIVE_EXPECTANCY" in code or "MONTH_SCALE" in code for code in blockers):
        return "EXPECTANCY_REPAIR"
    return "ACTIVE_BOARD_RERANK"


def _success_condition() -> dict[str, Any]:
    return {
        "report_material_only": True,
        "target_must_match_current_active_board_top": True,
        "guardian_consumption_must_allow_normal_routing_before_entries": True,
        "fresh_explicit_operator_review_required_for_clearance": True,
        "live_permission_must_remain_false": True,
        "live_side_effects_must_remain_empty": True,
    }


def _success_condition_evaluation(
    *,
    contract: dict[str, Any],
    target_lane: dict[str, Any],
    board_top: dict[str, Any],
    guardian_state: dict[str, Any],
) -> dict[str, bool]:
    return {
        "active_contract_selects_operator_review_report": contract.get("selected_active_path") == "OPERATOR_REVIEW_REPORT",
        "target_shape_matches_active_board_top": _lane_shape(target_lane) == _lane_shape(board_top),
        "guardian_consumption_normal_routing_allowed": guardian_state.get("consumption_normal_routing_allowed") is True,
        "guardian_operator_review_normal_routing_allowed": guardian_state.get("operator_review_normal_routing_allowed") is True,
        "live_permission_allowed_false": True,
        "live_side_effects_empty": True,
        "explicit_operator_review_not_inferred": True,
    }


def _status(
    *,
    artifacts: dict[str, dict[str, Any]],
    selected_active_path: Any,
    operator_review_required: bool,
    guardian_state: dict[str, Any],
    blockers: list[str],
) -> str:
    critical = ("active_trader_contract", "active_opportunity_board")
    if any(artifacts[name].get("_artifact_status") != "present" for name in critical):
        return STATUS_DATA_INCOMPLETE
    if selected_active_path != "OPERATOR_REVIEW_REPORT" and not operator_review_required:
        return STATUS_NOT_SELECTED
    guardian_clear = (
        guardian_state.get("consumption_normal_routing_allowed") is True
        and guardian_state.get("operator_review_normal_routing_allowed") is True
    )
    if operator_review_required or not guardian_clear:
        return STATUS_STILL_BLOCKED
    if blockers:
        return STATUS_CLEARED_OTHER_BLOCKERS_REMAIN
    return STATUS_MATERIAL_READY


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return {}

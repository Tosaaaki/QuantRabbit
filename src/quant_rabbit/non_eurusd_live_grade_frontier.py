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
    DEFAULT_AS_PROOF_PACK_QUEUE,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_FORECAST_HISTORY,
    DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
    DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
    DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER,
    DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER_REPORT,
    DEFAULT_NON_EURUSD_PROOF_LANE_MAPPER,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
    DEFAULT_PORTFOLIO_4X_PATH_PLANNER,
    DEFAULT_PROJECTION_LEDGER,
    DEFAULT_VERIFICATION_LEDGER,
    ROOT,
)


FRONTIER_VERSION = "non_eurusd_live_grade_frontier_v1"

STATUS_NON_EURUSD_FOUND = "NON_EURUSD_FRONTIER_FOUND"
STATUS_ONLY_EURUSD_FOUND = "ONLY_EURUSD_FRONTIER_FOUND"
STATUS_ALL_NEGATIVE = "ALL_FRONTIER_BLOCKED_BY_NEGATIVE_EXPECTANCY"
STATUS_ALL_SPREAD_OR_FORECAST = "ALL_FRONTIER_BLOCKED_BY_SPREAD_OR_FORECAST"
STATUS_DATA_INCOMPLETE = "FRONTIER_DATA_INCOMPLETE"

REQUIRED_NON_EUR_PAIRS = ("AUD_CAD", "USD_CAD", "CAD_JPY", "USD_JPY", "AUD_JPY", "GBP_USD")
USD_CAD_LONG_BREAKOUT_SHAPE = ("USD_CAD", "LONG", "BREAKOUT_FAILURE")
TOP_N = 10
DEFAULT_TP_PROOF_FLOOR = 20

NEGATIVE_BLOCKERS = (
    "NEGATIVE_EXPECTANCY",
    "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
    "MONTH_SCALE_REPLAY_NEGATIVE",
    "MARKET_CLOSE_LEAK",
    "REPLAY_NEGATIVE",
)
BIDASK_BLOCKERS = (
    "BIDASK_REPLAY_NEGATIVE",
    "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
    "BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED",
)
BIDASK_NEGATIVE_BLOCKERS = (
    "BIDASK_REPLAY_NEGATIVE",
    "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
    "spread_included_bidask_replay_negative_for_exact_lane",
    "packaged_bidask_rule_live_block_negative_expectancy",
)
BIDASK_REFRESH_BLOCKERS = ("BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED",)
SPREAD_BLOCKERS = ("SPREAD_TOO_WIDE", "TARGET_TOO_THIN_FOR_SPREAD")
FORECAST_BLOCKERS = (
    "FORECAST_WATCH_ONLY",
    "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
    "TELEMETRY_FORECAST_NOT_EXECUTABLE_FOR_LIVE",
    "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
    "FORECAST_DIRECTIONAL_HIT_RATE_WEAK_FOR_LIVE",
    "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
    "TELEMETRY_FORECAST_HISTORY_REQUIRED_FOR_LIVE",
    "TELEMETRY_FORECAST_HISTORY_STALE_FOR_LIVE",
    "TELEMETRY_FORECAST_HISTORY_MISMATCH_FOR_LIVE",
)
LOSS_BUDGET_BLOCKERS = (
    "LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT",
    "MARGIN_TOO_THIN_FOR_MIN_LOT",
    "MIN_LOT",
    "BAD_UNITS",
    "REWARD_RISK_TOO_LOW",
    "RANGE_COUNTERTREND_RR_TOO_LOW",
)
GUARDIAN_REVIEW_BLOCKERS = (
    "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
    "GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW",
    "OPERATOR_REVIEW_REQUIRED",
)
GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKER = "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"
OPERATOR_REVIEW_CONSUMPTION_CLEAR_STATUSES = {
    "OPERATOR_REVIEW_CLEARS_RECEIPT",
    "OPERATOR_REVIEW_DURABLY_CONSUMED_RECEIPT",
}


@dataclass(frozen=True)
class NonEurusdLiveGradeFrontierSummary:
    status: str
    output_path: Path
    report_path: Path
    top_lane_id: str | None
    top_non_eurusd_lane_id: str | None
    live_permission_allowed: bool


class NonEurusdLiveGradeFrontier:
    """Rank current EUR/USD and non-EUR/USD lanes by live-grade proximity.

    The artifact is deliberately read-only: it does not create live permission,
    does not relax blockers, and does not treat historical pair-level profit as
    exact TP proof. Its job is to identify the shortest next evidence action.
    """

    def __init__(
        self,
        *,
        active_opportunity_board_path: Path = DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        non_eurusd_proof_lane_mapper_path: Path = DEFAULT_NON_EURUSD_PROOF_LANE_MAPPER,
        payoff_shape_diagnosis_path: Path = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
        proof_pack_queue_path: Path = DEFAULT_AS_PROOF_PACK_QUEUE,
        portfolio_4x_path_planner_path: Path = DEFAULT_PORTFOLIO_4X_PATH_PLANNER,
        execution_ledger_db_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
        verification_ledger_path: Path = DEFAULT_VERIFICATION_LEDGER,
        forecast_history_path: Path = DEFAULT_FORECAST_HISTORY,
        projection_ledger_path: Path = DEFAULT_PROJECTION_LEDGER,
        guardian_receipt_consumption_path: Path = DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
        guardian_receipt_operator_review_path: Path = DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
        replay_artifact_paths: list[Path] | None = None,
        output_path: Path = DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER,
        report_path: Path = DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER_REPORT,
        now_utc: datetime | None = None,
    ) -> None:
        self.paths = {
            "active_opportunity_board": active_opportunity_board_path,
            "order_intents": order_intents_path,
            "non_eurusd_proof_lane_mapper": non_eurusd_proof_lane_mapper_path,
            "payoff_shape_diagnosis": payoff_shape_diagnosis_path,
            "as_proof_pack_queue": proof_pack_queue_path,
            "portfolio_4x_path_planner": portfolio_4x_path_planner_path,
            "verification_ledger": verification_ledger_path,
            "guardian_receipt_consumption": guardian_receipt_consumption_path,
            "guardian_receipt_operator_review": guardian_receipt_operator_review_path,
        }
        self.execution_ledger_db_path = execution_ledger_db_path
        self.forecast_history_path = forecast_history_path
        self.projection_ledger_path = projection_ledger_path
        self.replay_artifact_paths = replay_artifact_paths if replay_artifact_paths is not None else _default_replay_artifacts()
        self.output_path = output_path
        self.report_path = report_path
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> NonEurusdLiveGradeFrontierSummary:
        payload = self.build_payload()
        if payload.get("live_permission_allowed") is not False:
            raise ValueError("non-EUR/USD live-grade frontier must never grant live permission")
        if payload.get("live_side_effects") != []:
            raise ValueError("non-EUR/USD live-grade frontier must not record live side effects")
        for lane in payload.get("ranked_frontier_lanes") or []:
            if isinstance(lane, dict) and lane.get("live_permission_allowed") is True:
                raise ValueError("frontier lane must never grant live permission")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload), encoding="utf-8")
        top_lane = payload.get("top_lane") if isinstance(payload.get("top_lane"), dict) else {}
        top_non = payload.get("top_non_eurusd_lane") if isinstance(payload.get("top_non_eurusd_lane"), dict) else {}
        return NonEurusdLiveGradeFrontierSummary(
            status=str(payload["status"]),
            output_path=self.output_path,
            report_path=self.report_path,
            top_lane_id=top_lane.get("lane_id"),
            top_non_eurusd_lane_id=top_non.get("lane_id"),
            live_permission_allowed=False,
        )

    def build_payload(self) -> dict[str, Any]:
        artifacts = {name: _load_json_artifact(path) for name, path in self.paths.items()}
        replay_artifacts = {
            _display_path(path): _load_json_artifact(path)
            for path in self.replay_artifact_paths
            if path.name not in {
                DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER.name,
                DEFAULT_NON_EURUSD_PROOF_LANE_MAPPER.name,
            }
        }
        forecast_artifacts = _forecast_artifact_summary(
            forecast_history_path=self.forecast_history_path,
            projection_ledger_path=self.projection_ledger_path,
        )
        lanes = _collect_lanes(
            active_board=artifacts["active_opportunity_board"],
            order_intents=artifacts["order_intents"],
            proof_mapper=artifacts["non_eurusd_proof_lane_mapper"],
        )
        stale_blocker_codes = _stale_source_blocker_codes(
            guardian_receipt_consumption=artifacts["guardian_receipt_consumption"],
            order_intents=artifacts["order_intents"],
        )
        frontier_universe = [row for row in lanes.values() if row.get("order_intent_status")]
        if not frontier_universe:
            frontier_universe = list(lanes.values())
        ranked = sorted(
            (_frontier_lane(row, stale_blocker_codes=stale_blocker_codes) for row in frontier_universe),
            key=_frontier_sort_key,
        )
        top_lanes = ranked[:TOP_N]
        top_lane = top_lanes[0] if top_lanes else {}
        non_eur_ranked = [lane for lane in ranked if lane.get("pair") != "EUR_USD"]
        top_non = non_eur_ranked[0] if non_eur_ranked else {}
        gaps = _gap_sets(ranked)
        required_checks = _required_checks(ranked, top_lane, top_non)
        status = _frontier_status(
            ranked=ranked,
            top_lanes=top_lanes,
            top_lane=top_lane,
            top_non=top_non,
            artifacts=artifacts,
        )
        payload = {
            "schema_version": FRONTIER_VERSION,
            "status": status,
            "generated_at_utc": self.now_utc.isoformat(),
            "read_only": True,
            "live_side_effects": [],
            "live_permission_allowed": False,
            "scanned_pairs": sorted({lane["pair"] for lane in ranked if lane.get("pair") != "UNKNOWN"}),
            "scanned_intents": len(_list(artifacts["order_intents"].get("results"))),
            "ranked_frontier_lanes": [_public_frontier_lane(lane) for lane in top_lanes],
            "top_lane": _public_frontier_lane(top_lane) if top_lane else {},
            "top_non_eurusd_lane": _public_frontier_lane(top_non) if top_non else {},
            "proof_floor_gaps": gaps["proof_floor_gaps"],
            "bidask_replay_gaps": gaps["bidask_replay_gaps"],
            "spread_gaps": gaps["spread_gaps"],
            "forecast_gaps": gaps["forecast_gaps"],
            "loss_budget_gaps": gaps["loss_budget_gaps"],
            "required_checks": required_checks,
            "source_artifacts": _source_artifacts(
                artifacts=artifacts,
                replay_artifacts=replay_artifacts,
                forecast_artifacts=forecast_artifacts,
                execution_ledger_db_path=self.execution_ledger_db_path,
            ),
            "execution_ledger_summary": _execution_ledger_summary(self.execution_ledger_db_path),
            "forecast_artifacts": forecast_artifacts,
            "next_active_path": _next_active_path(status, top_non or top_lane),
            "do_not_do": _do_not_do(),
        }
        return payload


def _default_replay_artifacts() -> list[Path]:
    data_dir = ROOT / "data"
    paths: list[Path] = []
    for pattern in ("*bidask*replay*.json", "*replay*.json", "*proof*.json"):
        for path in data_dir.glob(pattern):
            if path.name == "as_proof_pack_queue.json":
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
    result = dict(payload)
    result["_artifact_status"] = "present"
    result["_path"] = str(path)
    result["_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


def _stale_source_blocker_codes(
    *,
    guardian_receipt_consumption: dict[str, Any],
    order_intents: dict[str, Any],
) -> list[str]:
    stale_codes: list[str] = []
    if (
        _guardian_receipt_consumption_durably_clears_operator_review(guardian_receipt_consumption)
        and _order_intents_older_than_consumption(order_intents, guardian_receipt_consumption)
    ):
        stale_codes.append(GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKER)
    return stale_codes


def _guardian_receipt_consumption_durably_clears_operator_review(
    guardian_receipt_consumption: dict[str, Any],
) -> bool:
    if guardian_receipt_consumption.get("_artifact_status") != "present":
        return False
    if guardian_receipt_consumption.get("normal_routing_allowed") is not True:
        return False
    rows = guardian_receipt_consumption.get("classifications")
    if not isinstance(rows, list) or not rows:
        return False
    classification_rows = [row for row in rows if isinstance(row, dict)]
    if len(classification_rows) != len(rows):
        return False
    if any(row.get("normal_routing_allowed") is not True for row in classification_rows):
        return False
    return any(
        row.get("operator_review_required") is True
        and str(row.get("operator_review_status") or "") in OPERATOR_REVIEW_CONSUMPTION_CLEAR_STATUSES
        for row in classification_rows
    )


def _order_intents_older_than_consumption(
    order_intents: dict[str, Any],
    guardian_receipt_consumption: dict[str, Any],
) -> bool:
    intent_generated = _parse_utc(order_intents.get("generated_at_utc"))
    consumption_generated = _parse_utc(guardian_receipt_consumption.get("generated_at_utc"))
    if intent_generated is None or consumption_generated is None:
        return False
    return intent_generated < consumption_generated


def _collect_lanes(
    *,
    active_board: dict[str, Any],
    order_intents: dict[str, Any],
    proof_mapper: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    lanes: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(_list(order_intents.get("results"))):
        if not isinstance(row, dict):
            continue
        lane_id = str(row.get("lane_id") or "")
        if not lane_id:
            continue
        lane = _ensure_lane(lanes, lane_id)
        _merge_order_intent(lane, row, index)
    for index, row in enumerate(_list(active_board.get("ranked_active_lanes"))):
        if not isinstance(row, dict):
            continue
        lane_id = str(row.get("lane_id") or "")
        if not lane_id:
            continue
        lane = _ensure_lane(lanes, lane_id)
        _merge_board_lane(lane, row, index)
    mapper_by_id = {
        str(row.get("lane_id")): row
        for row in _list(proof_mapper.get("mapped_lanes"))
        if isinstance(row, dict) and row.get("lane_id")
    }
    for lane_id, mapped in mapper_by_id.items():
        lane = _ensure_lane(lanes, lane_id)
        _merge_mapper_lane(lane, mapped)
    return lanes


def _ensure_lane(lanes: dict[str, dict[str, Any]], lane_id: str) -> dict[str, Any]:
    if lane_id not in lanes:
        parsed = _parse_lane_id(lane_id)
        lanes[lane_id] = {
            "lane_id": lane_id,
            "pair": parsed.get("pair", "UNKNOWN"),
            "direction": parsed.get("direction", "UNKNOWN"),
            "strategy_family": parsed.get("strategy_family", "UNKNOWN"),
            "vehicle": parsed.get("vehicle", "UNKNOWN"),
            "status": "NO_TRADE_WITH_CAUSE",
            "blockers": [],
            "source_refs": [],
            "order_intent_index": None,
            "board_rank_index": None,
            "expected_edge_jpy": None,
            "risk_metrics": {},
            "local_tp_proof": {},
            "proof_floor": {},
            "proof_mapper_assessment": "",
            "stale_source_blockers": [],
        }
    return lanes[lane_id]


def _merge_order_intent(lane: dict[str, Any], row: dict[str, Any], index: int) -> None:
    intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    parsed = _parse_lane_id(str(row.get("lane_id") or ""))
    _fill_shape(
        lane,
        pair=_first_str(intent.get("pair"), row.get("pair"), parsed.get("pair")),
        direction=_first_str(intent.get("side"), intent.get("direction"), row.get("side"), parsed.get("direction")),
        strategy=_first_str(
            intent.get("method"),
            intent.get("strategy_family"),
            row.get("method"),
            metadata.get("method"),
            parsed.get("strategy_family"),
        ),
        vehicle=_first_str(intent.get("order_type"), row.get("order_type"), parsed.get("vehicle")),
    )
    lane["status"] = _prefer_status(str(row.get("status") or ""), str(lane.get("status") or ""))
    lane["order_intent_status"] = row.get("status")
    lane["risk_allowed"] = bool(row.get("risk_allowed"))
    lane["order_intent_index"] = index if lane.get("order_intent_index") is None else min(index, int(lane["order_intent_index"]))
    metrics = row.get("risk_metrics") if isinstance(row.get("risk_metrics"), dict) else {}
    lane["risk_metrics"] = {**(lane.get("risk_metrics") or {}), **metrics}
    lane["expected_edge_jpy"] = _first_number(
        lane.get("expected_edge_jpy"),
        metrics.get("expected_edge_jpy"),
        metrics.get("expectancy_jpy"),
        metadata.get("capture_take_profit_expectancy_jpy"),
    )
    _merge_local_tp_proof(lane, metadata)
    lane["blockers"].extend(_string_list(row.get("live_blocker_codes")))
    lane["blockers"].extend(_issue_codes(row.get("risk_issues")))
    lane["blockers"].extend(_issue_codes(row.get("strategy_issues")))
    lane["source_refs"].append("data/order_intents.json")


def _merge_board_lane(lane: dict[str, Any], row: dict[str, Any], index: int) -> None:
    _fill_shape(
        lane,
        pair=row.get("pair"),
        direction=row.get("direction") or row.get("side"),
        strategy=row.get("strategy_family") or row.get("method"),
        vehicle=row.get("vehicle") or row.get("order_type"),
    )
    lane["status"] = _prefer_status(str(lane.get("status") or ""), str(row.get("status") or ""))
    lane["active_board_status"] = row.get("status")
    lane["board_rank_index"] = index if lane.get("board_rank_index") is None else min(index, int(lane["board_rank_index"]))
    lane["expected_edge_jpy"] = _first_number(lane.get("expected_edge_jpy"), row.get("expected_edge_jpy"))
    lane["spread_status_board"] = row.get("spread_status")
    lane["replay_status_board"] = row.get("replay_status")
    lane["risk_status_board"] = row.get("risk_status")
    lane["proof_status_board"] = row.get("proof_status")
    if isinstance(row.get("local_tp_proof"), dict):
        lane["local_tp_proof"] = {**(lane.get("local_tp_proof") or {}), **row["local_tp_proof"]}
    lane["blockers"].extend(_string_list(row.get("blockers")))
    lane["stale_source_blockers"].extend(_string_list(row.get("stale_source_blockers")))
    lane["source_refs"].extend(_string_list(row.get("source_refs")))
    lane["source_refs"].append("data/active_opportunity_board.json")


def _merge_mapper_lane(lane: dict[str, Any], row: dict[str, Any]) -> None:
    _fill_shape(
        lane,
        pair=row.get("pair"),
        direction=row.get("side") or row.get("direction"),
        strategy=row.get("strategy_family"),
        vehicle=row.get("vehicle"),
    )
    lane["proof_mapper_assessment"] = row.get("promotion_assessment")
    if isinstance(row.get("proof_floor"), dict):
        lane["proof_floor"] = row["proof_floor"]
    lane["blockers"].extend(_string_list(row.get("blockers")))
    lane["blockers"].extend(_string_list(row.get("mapping_gaps")))
    lane["source_refs"].append("data/non_eurusd_proof_lane_mapper.json")


def _frontier_lane(lane: dict[str, Any], *, stale_blocker_codes: list[str]) -> dict[str, Any]:
    stale_source_blockers = _unique(_string_list(lane.get("stale_source_blockers")) + stale_blocker_codes)
    blockers = [
        code
        for code in _unique(_string_list(lane.get("blockers")))
        if code not in stale_source_blockers
    ]
    tp_count, tp_floor = _tp_proof_counts(lane)
    proof_remaining = max((tp_floor or 0) - (tp_count or 0), 0) if tp_floor is not None and tp_count is not None else None
    bidask_status = _bidask_status(lane, blockers)
    spread_status = _spread_status(lane, blockers)
    forecast_status = _forecast_status(blockers)
    loss_budget_status = _loss_budget_status(blockers)
    distance = _distance_label(
        lane=lane,
        blockers=blockers,
        proof_remaining=proof_remaining,
        bidask_status=bidask_status,
        spread_status=spread_status,
        forecast_status=forecast_status,
        loss_budget_status=loss_budget_status,
    )
    row = {
        **lane,
        "blockers": blockers,
        "stale_source_blockers": stale_source_blockers,
        "tp_proof_count": tp_count,
        "tp_proof_floor": tp_floor,
        "tp_proof_remaining": proof_remaining,
        "bidask_status": bidask_status,
        "spread_status": spread_status,
        "forecast_status": forecast_status,
        "loss_budget_status": loss_budget_status,
        "distance_to_live_ready": distance,
        "distance_score": _distance_score(
            lane=lane,
            blockers=blockers,
            proof_remaining=proof_remaining,
            bidask_status=bidask_status,
            spread_status=spread_status,
            forecast_status=forecast_status,
            loss_budget_status=loss_budget_status,
        ),
    }
    row["next_action"] = _next_action(row)
    return row


def _public_frontier_lane(lane: dict[str, Any]) -> dict[str, Any]:
    if not lane:
        return {}
    return {
        "lane_id": str(lane.get("lane_id") or ""),
        "pair": str(lane.get("pair") or "UNKNOWN"),
        "direction": str(lane.get("direction") or "UNKNOWN"),
        "strategy_family": str(lane.get("strategy_family") or "UNKNOWN"),
        "vehicle": str(lane.get("vehicle") or "UNKNOWN"),
        "status": str(lane.get("status") or "NO_TRADE_WITH_CAUSE"),
        "distance_to_live_ready": str(lane.get("distance_to_live_ready") or "UNKNOWN"),
        "tp_proof_count": _json_number_or_none(lane.get("tp_proof_count")),
        "tp_proof_floor": _json_number_or_none(lane.get("tp_proof_floor")),
        "bidask_status": str(lane.get("bidask_status") or "UNKNOWN"),
        "spread_status": str(lane.get("spread_status") or "UNKNOWN"),
        "forecast_status": str(lane.get("forecast_status") or "UNKNOWN"),
        "loss_budget_status": str(lane.get("loss_budget_status") or "UNKNOWN"),
        "expected_edge_jpy": _json_number_or_none(lane.get("expected_edge_jpy")),
        "blockers": _string_list(lane.get("blockers")),
        "next_action": str(lane.get("next_action") or ""),
    }


def _distance_score(
    *,
    lane: dict[str, Any],
    blockers: list[str],
    proof_remaining: int | None,
    bidask_status: str,
    spread_status: str,
    forecast_status: str,
    loss_budget_status: str,
) -> int:
    status = str(lane.get("status") or "")
    score = {
        "LIVE_READY": 0,
        "HARVEST_READY": 10,
        "SCOUT_READY": 12,
        "EVIDENCE_ACQUISITION": 18,
        "OPERATOR_REVIEW_REQUIRED": 24,
        "DRY_RUN_PASSED": 16,
        "DRY_RUN_BLOCKED": 30,
        "NO_TRADE_WITH_CAUSE": 38,
    }.get(status, 34)
    if lane.get("order_intent_status"):
        score -= 6
    if lane.get("active_board_status"):
        score -= 3
    if _has_marker(blockers, GUARDIAN_REVIEW_BLOCKERS):
        score += 8
    if _has_marker(blockers, NEGATIVE_BLOCKERS):
        score += 13
    if bidask_status in {"NEGATIVE", "REFRESH_REQUIRED"}:
        score += 9
    if spread_status == "BLOCKED":
        score += 9
    if forecast_status != "PASS":
        score += 7 if forecast_status == "BLOCKED" else 4
    if loss_budget_status == "BLOCKED":
        score += 7
    if proof_remaining is not None:
        score += min(max(proof_remaining, 0), DEFAULT_TP_PROOF_FLOOR)
    if str(lane.get("vehicle") or "") == "UNKNOWN":
        score += 12
    return max(score, 0)


def _distance_label(
    *,
    lane: dict[str, Any],
    blockers: list[str],
    proof_remaining: int | None,
    bidask_status: str,
    spread_status: str,
    forecast_status: str,
    loss_budget_status: str,
) -> str:
    if str(lane.get("status") or "") == "LIVE_READY" and not blockers:
        return "0_LIVE_READY_VERIFIER_GATE_ONLY"
    hard = []
    if _has_marker(blockers, GUARDIAN_REVIEW_BLOCKERS):
        hard.append("guardian_review")
    if _has_marker(blockers, NEGATIVE_BLOCKERS):
        hard.append("negative_expectancy")
    if bidask_status in {"NEGATIVE", "REFRESH_REQUIRED"}:
        hard.append("bidask_replay")
    if spread_status == "BLOCKED":
        hard.append("spread")
    if forecast_status != "PASS":
        hard.append("forecast")
    if loss_budget_status == "BLOCKED":
        hard.append("loss_budget")
    if proof_remaining and proof_remaining > 0:
        hard.append("tp_proof_floor")
    if not hard:
        return "1_NEAR_LIVE_READY_RECHECK_GATEWAY"
    if len(hard) <= 2:
        return "2_CLOSE_BUT_BLOCKED_BY_" + "_AND_".join(hard[:2]).upper()
    return "3_MULTI_GATE_BLOCKED_" + "_".join(hard[:3]).upper()


def _frontier_sort_key(lane: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(lane.get("distance_score") or 9999),
        str(lane.get("pair") or "") == "EUR_USD",
        int(lane.get("order_intent_index") if lane.get("order_intent_index") is not None else 999999),
        int(lane.get("board_rank_index") if lane.get("board_rank_index") is not None else 999999),
        str(lane.get("pair") or ""),
        str(lane.get("direction") or ""),
        str(lane.get("strategy_family") or ""),
        str(lane.get("vehicle") or ""),
    )


def _gap_sets(ranked: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    proof_floor_gaps: list[dict[str, Any]] = []
    bidask_replay_gaps: list[dict[str, Any]] = []
    spread_gaps: list[dict[str, Any]] = []
    forecast_gaps: list[dict[str, Any]] = []
    loss_budget_gaps: list[dict[str, Any]] = []
    for lane in ranked:
        public = _gap_lane_base(lane)
        if lane.get("tp_proof_floor") is not None and (lane.get("tp_proof_remaining") or 0) > 0:
            proof_floor_gaps.append(
                {
                    **public,
                    "current_tp_trades": lane.get("tp_proof_count"),
                    "required_tp_trades": lane.get("tp_proof_floor"),
                    "remaining_tp_trades": lane.get("tp_proof_remaining"),
                }
            )
        if lane.get("bidask_status") in {"NEGATIVE", "REFRESH_REQUIRED"}:
            bidask_replay_gaps.append(public)
        if lane.get("spread_status") == "BLOCKED":
            spread_gaps.append(public)
        if lane.get("forecast_status") != "PASS":
            forecast_gaps.append(public)
        if lane.get("loss_budget_status") == "BLOCKED":
            loss_budget_gaps.append(public)
    return {
        "proof_floor_gaps": proof_floor_gaps[:25],
        "bidask_replay_gaps": bidask_replay_gaps[:25],
        "spread_gaps": spread_gaps[:25],
        "forecast_gaps": forecast_gaps[:25],
        "loss_budget_gaps": loss_budget_gaps[:25],
    }


def _gap_lane_base(lane: dict[str, Any]) -> dict[str, Any]:
    return {
        "lane_id": lane.get("lane_id"),
        "pair": lane.get("pair"),
        "direction": lane.get("direction"),
        "strategy_family": lane.get("strategy_family"),
        "vehicle": lane.get("vehicle"),
        "status": lane.get("status"),
        "distance_to_live_ready": lane.get("distance_to_live_ready"),
        "blockers": _string_list(lane.get("blockers"))[:12],
        "next_action": lane.get("next_action"),
    }


def _required_checks(
    ranked: list[dict[str, Any]],
    top_lane: dict[str, Any],
    top_non: dict[str, Any],
) -> dict[str, Any]:
    eur_ranked = [lane for lane in ranked if lane.get("pair") == "EUR_USD"]
    required_pair_rows = {
        pair: [_public_frontier_lane(lane) for lane in ranked if lane.get("pair") == pair][:5]
        for pair in REQUIRED_NON_EUR_PAIRS
    }
    usd_cad_long_breakout = [
        _public_frontier_lane(lane)
        for lane in ranked
        if (
            lane.get("pair"),
            lane.get("direction"),
            lane.get("strategy_family"),
        )
        == USD_CAD_LONG_BREAKOUT_SHAPE
    ]
    top_eur = eur_ranked[0] if eur_ranked else {}
    return {
        "non_eurusd_closer_than_eurusd": bool(
            top_non
            and (
                not top_eur
                or int(top_non.get("distance_score") or 9999) <= int(top_eur.get("distance_score") or 9999)
            )
        ),
        "top_eurusd_lane": _public_frontier_lane(top_eur) if top_eur else {},
        "usd_cad_long_breakout_failure_blocker_breakdown": usd_cad_long_breakout,
        "required_pair_candidates": required_pair_rows,
        "spread_too_wide_not_ignored": bool(any(lane.get("spread_status") == "BLOCKED" for lane in ranked)),
        "bidask_negative_not_ignored": bool(
            any(lane.get("bidask_status") in {"NEGATIVE", "REFRESH_REQUIRED"} for lane in ranked)
        ),
        "next_evidence_lane": _public_frontier_lane(top_non or top_lane),
    }


def _frontier_status(
    *,
    ranked: list[dict[str, Any]],
    top_lanes: list[dict[str, Any]],
    top_lane: dict[str, Any],
    top_non: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> str:
    if (
        artifacts["active_opportunity_board"].get("_artifact_status") != "present"
        or artifacts["order_intents"].get("_artifact_status") != "present"
        or not ranked
    ):
        return STATUS_DATA_INCOMPLETE
    if top_lanes and all(_has_marker(lane.get("blockers") or [], NEGATIVE_BLOCKERS) for lane in top_lanes):
        return STATUS_ALL_NEGATIVE
    if top_lanes and all(
        _has_marker(lane.get("blockers") or [], SPREAD_BLOCKERS + FORECAST_BLOCKERS) for lane in top_lanes
    ):
        return STATUS_ALL_SPREAD_OR_FORECAST
    if top_non:
        return STATUS_NON_EURUSD_FOUND
    if top_lane.get("pair") == "EUR_USD":
        return STATUS_ONLY_EURUSD_FOUND
    return STATUS_NON_EURUSD_FOUND if top_non else STATUS_DATA_INCOMPLETE


def _next_active_path(status: str, lane: dict[str, Any]) -> str:
    if not lane:
        return "FRONTIER_DATA_INCOMPLETE"
    lane_id = lane.get("lane_id")
    if status == STATUS_ALL_NEGATIVE:
        if lane.get("bidask_status") == "NEGATIVE":
            return (
                f"BIDASK_NEGATIVE_PATTERN_REPAIR: current exact bid/ask replay is negative for {lane_id}; "
                "repair pattern/vehicle selection or lane-local TP proof before rerunning replay. Do not send."
            )
        return f"EVIDENCE_ACQUISITION: preserve negative expectancy and rebuild exact TP/bidask proof for {lane_id}."
    if status == STATUS_ALL_SPREAD_OR_FORECAST:
        return f"FORECAST_OR_SPREAD_REFRESH: refresh current forecast/spread packet for {lane_id}; do not ignore blockers."
    if lane.get("bidask_status") == "REFRESH_REQUIRED":
        return f"BIDASK_REPLAY_REFRESH: run exact read-only bid/ask replay for {lane_id}; do not send."
    if lane.get("bidask_status") == "NEGATIVE":
        return (
            f"BIDASK_NEGATIVE_PATTERN_REPAIR: current exact bid/ask replay is negative for {lane_id}; "
            "repair pattern/vehicle selection or lane-local TP proof before rerunning replay. Do not send."
        )
    if lane.get("tp_proof_remaining"):
        return f"TP_PROOF_COLLECTION: collect exact TAKE_PROFIT_ORDER proof for {lane_id}; do not mix market-close losses."
    if lane.get("forecast_status") != "PASS":
        return f"FORECAST_TELEMETRY_REPAIR: make forecast executable/auditable for {lane_id}."
    if lane.get("spread_status") == "BLOCKED":
        return f"SPREAD_RECHECK: wait for session/liquidity spread to clear for {lane_id}; do not override spread."
    return f"LIVE_PERMISSION_READY_CHECK: {lane_id} is nearest; verifier/gateway still decide."


def _next_action(lane: dict[str, Any]) -> str:
    lane_id = lane.get("lane_id")
    if lane.get("bidask_status") == "REFRESH_REQUIRED":
        return f"Refresh exact S5 bid/ask replay for {lane_id}; keep negative blocker visible."
    if lane.get("bidask_status") == "NEGATIVE":
        return (
            f"Repair bid/ask-negative pattern or vehicle shape for {lane_id}; "
            "do not repeat replay until the lane inputs change."
        )
    if _has_marker(lane.get("blockers") or [], NEGATIVE_BLOCKERS):
        return f"Build exact TP-proven rotation proof for {lane_id}; do not hide negative expectancy."
    if lane.get("spread_status") == "BLOCKED":
        return f"Recheck spread/session for {lane_id}; do not override spread too wide."
    if lane.get("forecast_status") != "PASS":
        return f"Repair forecast telemetry/current executable forecast for {lane_id}."
    if lane.get("loss_budget_status") == "BLOCKED":
        return f"Wait for loss-budget/min-lot feasibility for {lane_id}; do not backsolve lots from 4x deficit."
    if lane.get("tp_proof_remaining"):
        return f"Collect {lane.get('tp_proof_remaining')} exact TP proof sample(s) for {lane_id}."
    return f"Keep {lane_id} on verifier/gateway path; frontier grants no live permission."


def _tp_proof_counts(lane: dict[str, Any]) -> tuple[int | None, int | None]:
    proof = lane.get("local_tp_proof") if isinstance(lane.get("local_tp_proof"), dict) else {}
    mapper_floor = lane.get("proof_floor") if isinstance(lane.get("proof_floor"), dict) else {}
    if mapper_floor:
        count = _first_int(mapper_floor.get("current_tp_trades"), None)
        floor = _first_int(mapper_floor.get("required_tp_trades"), None)
        if count is not None or floor is not None:
            return int(count or 0), int(floor or DEFAULT_TP_PROOF_FLOOR)
    count = _first_int(
        proof.get("capture_take_profit_trades"),
    )
    floor = _first_int(
        proof.get("capture_take_profit_proof_floor"),
    )
    if count is None and floor is None:
        return None, None
    return int(count or 0), int(floor or DEFAULT_TP_PROOF_FLOOR)


def _bidask_status(lane: dict[str, Any], blockers: list[str]) -> str:
    replay_status = str(lane.get("replay_status_board") or "").upper()
    if "NEGATIVE" in replay_status:
        return "NEGATIVE"
    if _has_marker(blockers, BIDASK_NEGATIVE_BLOCKERS):
        return "NEGATIVE"
    if "REFRESH" in replay_status:
        return "REFRESH_REQUIRED"
    if _has_marker(blockers, BIDASK_REFRESH_BLOCKERS):
        return "REFRESH_REQUIRED"
    return "PASS" if lane.get("order_intent_status") else "UNKNOWN"


def _spread_status(lane: dict[str, Any], blockers: list[str]) -> str:
    board_status = str(lane.get("spread_status_board") or "").upper()
    if board_status == "BLOCKED" or _has_marker(blockers, SPREAD_BLOCKERS):
        return "BLOCKED"
    if lane.get("risk_metrics", {}).get("spread_pips") is not None or board_status:
        return "PASS"
    return "UNKNOWN"


def _forecast_status(blockers: list[str]) -> str:
    if _has_marker(blockers, FORECAST_BLOCKERS):
        return "BLOCKED"
    return "PASS"


def _loss_budget_status(blockers: list[str]) -> str:
    if _has_marker(blockers, LOSS_BUDGET_BLOCKERS):
        return "BLOCKED"
    return "PASS"


def _merge_local_tp_proof(lane: dict[str, Any], metadata: dict[str, Any]) -> None:
    keys = {
        "capture_take_profit_trades",
        "capture_take_profit_wins",
        "capture_take_profit_losses",
        "capture_take_profit_expectancy_jpy",
        "capture_take_profit_proof_floor",
        "capture_take_profit_scope",
        "capture_take_profit_scope_key",
    }
    updates = {key: metadata.get(key) for key in keys if metadata.get(key) is not None}
    if updates:
        lane["local_tp_proof"] = {**(lane.get("local_tp_proof") or {}), **updates}


def _fill_shape(lane: dict[str, Any], *, pair: Any, direction: Any, strategy: Any, vehicle: Any) -> None:
    for key, value in (
        ("pair", pair),
        ("direction", direction),
        ("strategy_family", strategy),
        ("vehicle", _normalize_vehicle(vehicle)),
    ):
        if value in (None, "", "UNKNOWN"):
            continue
        if lane.get(key) in (None, "", "UNKNOWN"):
            lane[key] = str(value).upper()


def _prefer_status(left: str, right: str) -> str:
    priority = {
        "LIVE_READY": 7,
        "HARVEST_READY": 6,
        "SCOUT_READY": 5,
        "EVIDENCE_ACQUISITION": 4,
        "OPERATOR_REVIEW_REQUIRED": 3,
        "DRY_RUN_PASSED": 3,
        "DRY_RUN_BLOCKED": 2,
        "NO_TRADE_WITH_CAUSE": 1,
        "": 0,
    }
    return left if priority.get(left, 0) >= priority.get(right, 0) else right


def _forecast_artifact_summary(*, forecast_history_path: Path, projection_ledger_path: Path) -> dict[str, Any]:
    return {
        "forecast_history": _jsonl_summary(forecast_history_path),
        "projection_ledger": _jsonl_summary(projection_ledger_path),
    }


def _jsonl_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing", "rows": 0, "sha256": None}
    raw = path.read_bytes()
    rows = 0
    malformed = 0
    latest = None
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        rows += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(item, dict):
            ts = item.get("timestamp_utc") or item.get("timestamp_emitted_utc") or item.get("generated_at_utc")
            if isinstance(ts, str) and (latest is None or ts > latest):
                latest = ts
    return {
        "path": str(path),
        "status": "present",
        "rows": rows,
        "malformed_rows": malformed,
        "latest_timestamp_utc": latest,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _execution_ledger_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing"}
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = con.execute(
                """
                select count(*) as rows,
                       sum(case when event_type='TRADE_CLOSED' then 1 else 0 end) as closed_rows,
                       max(ts_utc) as latest_ts_utc
                from execution_events
                """
            ).fetchone()
        finally:
            con.close()
    except sqlite3.Error as exc:
        return {"path": str(path), "status": "unreadable", "error": str(exc)}
    return {
        "path": str(path),
        "status": "present",
        "rows": int(row[0] or 0),
        "closed_rows": int(row[1] or 0),
        "latest_ts_utc": row[2],
    }


def _source_artifacts(
    *,
    artifacts: dict[str, dict[str, Any]],
    replay_artifacts: dict[str, dict[str, Any]],
    forecast_artifacts: dict[str, Any],
    execution_ledger_db_path: Path,
) -> dict[str, Any]:
    return {
        **{name: _artifact_entry(artifact) for name, artifact in artifacts.items()},
        "replay_artifacts": {name: _artifact_entry(artifact) for name, artifact in replay_artifacts.items()},
        "forecast_artifacts": forecast_artifacts,
        "execution_ledger_db": {
            "path": str(execution_ledger_db_path),
            "status": "present" if execution_ledger_db_path.exists() else "missing",
            "size_bytes": execution_ledger_db_path.stat().st_size if execution_ledger_db_path.exists() else None,
        },
    }


def _artifact_entry(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": artifact.get("_path"),
        "status": artifact.get("_artifact_status"),
        "sha256": artifact.get("_sha256"),
        "generated_at_utc": artifact.get("generated_at_utc") or artifact.get("fetched_at_utc"),
    }


def _render_report(payload: dict[str, Any]) -> str:
    checks = payload.get("required_checks") if isinstance(payload.get("required_checks"), dict) else {}
    lines = [
        "# Non-EUR/USD Live-Grade Frontier",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Read-only: `{payload.get('read_only')}`",
        f"- Live permission allowed: `{payload.get('live_permission_allowed')}`",
        f"- Scanned intents: `{payload.get('scanned_intents')}`",
        f"- Scanned pairs: `{', '.join(payload.get('scanned_pairs') or [])}`",
        "",
        "## Top Lane",
        "",
        _lane_report_line(payload.get("top_lane") if isinstance(payload.get("top_lane"), dict) else {}),
        "",
        "## Top Non-EUR/USD Lane",
        "",
        _lane_report_line(payload.get("top_non_eurusd_lane") if isinstance(payload.get("top_non_eurusd_lane"), dict) else {}),
        "",
        "## Ranked Frontier",
        "",
    ]
    for lane in payload.get("ranked_frontier_lanes") or []:
        if not isinstance(lane, dict):
            continue
        lines.append(_lane_report_line(lane))
    lines.extend(
        [
            "",
            "## Required Checks",
            "",
            f"- Non-EUR/USD closer than EUR/USD: `{checks.get('non_eurusd_closer_than_eurusd')}`",
            f"- Spread too wide not ignored: `{checks.get('spread_too_wide_not_ignored')}`",
            f"- Bid/ask negative not ignored: `{checks.get('bidask_negative_not_ignored')}`",
            "",
            "### USD_CAD LONG BREAKOUT_FAILURE",
            "",
        ]
    )
    for lane in checks.get("usd_cad_long_breakout_failure_blocker_breakdown") or []:
        lines.append(_lane_report_line(lane))
    lines.extend(
        [
            "",
            "## Gaps",
            "",
            f"- Proof floor gaps: `{len(payload.get('proof_floor_gaps') or [])}`",
            f"- Bid/ask replay gaps: `{len(payload.get('bidask_replay_gaps') or [])}`",
            f"- Spread gaps: `{len(payload.get('spread_gaps') or [])}`",
            f"- Forecast gaps: `{len(payload.get('forecast_gaps') or [])}`",
            f"- Loss budget gaps: `{len(payload.get('loss_budget_gaps') or [])}`",
            "",
            "## Next Active Path",
            "",
            str(payload.get("next_active_path") or ""),
            "",
            "## Safety",
            "",
            "This artifact is read-only. It does not send, cancel, close, change launchd, relax gates, hide negative expectancy, ignore spread or bid/ask blockers, mix market-close loss into TP proof, backsolve lots from the 4x gap, infer operator approval, or expose secrets.",
            "",
        ]
    )
    return "\n".join(lines)


def _lane_report_line(lane: dict[str, Any]) -> str:
    if not lane:
        return "- No lane."
    blockers = ", ".join(_string_list(lane.get("blockers"))[:5])
    return (
        f"- `{lane.get('lane_id')}` `{lane.get('distance_to_live_ready')}` "
        f"bidask `{lane.get('bidask_status')}` spread `{lane.get('spread_status')}` "
        f"forecast `{lane.get('forecast_status')}` loss_budget `{lane.get('loss_budget_status')}` "
        f"TP `{lane.get('tp_proof_count')}/{lane.get('tp_proof_floor')}` blockers `{blockers}`"
    )


def _do_not_do() -> list[str]:
    return [
        "do_not_send_live_order",
        "do_not_cancel_order",
        "do_not_close_position",
        "do_not_change_launchd",
        "do_not_relax_gates",
        "do_not_hide_negative_expectancy",
        "do_not_ignore_spread_too_wide",
        "do_not_ignore_bidask_replay_negative",
        "do_not_mix_market_close_loss_into_tp_proof",
        "do_not_backsolve_lot_size_from_4x_gap",
        "do_not_print_secrets",
    ]


def _parse_lane_id(lane_id: str) -> dict[str, str]:
    parts = str(lane_id or "").split(":")
    if len(parts) < 4:
        return {}
    return {
        "pair": parts[1].upper(),
        "direction": parts[2].upper(),
        "strategy_family": parts[3].upper(),
        "vehicle": _normalize_vehicle(parts[4] if len(parts) > 4 else "UNKNOWN"),
    }


def _normalize_vehicle(value: Any) -> str:
    text = str(value or "UNKNOWN").upper()
    if text in {"STOP_ORDER", "STOP_ENTRY", "STOP-ENTRY"}:
        return "STOP"
    if text == "LIMIT_ORDER":
        return "LIMIT"
    if text == "MARKET_ORDER":
        return "MARKET"
    if text in {"LIMIT", "MARKET", "STOP"}:
        return text
    return "UNKNOWN" if text in {"", "NONE", "NULL"} else text


def _issue_codes(value: Any) -> list[str]:
    codes: list[str] = []
    for item in _list(value):
        if isinstance(item, dict):
            code = item.get("code")
            if code:
                codes.append(str(code))
        elif item:
            codes.append(str(item))
    return codes


def _has_marker(values: list[Any], markers: tuple[str, ...]) -> bool:
    return any(any(marker in str(value) for marker in markers) for value in values)


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if str(item)]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _first_str(*values: Any) -> str:
    for value in values:
        if value not in (None, "", "UNKNOWN"):
            return str(value).upper()
    return "UNKNOWN"


def _first_number(*values: Any) -> float | None:
    for value in values:
        parsed = _float(value)
        if parsed is not None:
            return parsed
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _float(value)
        if parsed is not None:
            return int(parsed)
    return None


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_number_or_none(value: Any) -> int | float | None:
    parsed = _float(value)
    if parsed is None:
        return None
    if parsed.is_integer():
        return int(parsed)
    return round(parsed, 6)


def _display_path(path: Path) -> str:
    try:
        if path.is_absolute() and path.is_relative_to(ROOT):
            return str(path.relative_to(ROOT))
    except ValueError:
        pass
    return str(path)

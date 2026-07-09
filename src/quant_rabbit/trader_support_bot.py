from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.execution_timing_contracts import (
    MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND,
    TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC,
    TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
    repair_replay_contract_from_payload,
)
from quant_rabbit.guardian_receipt_consumption import consumption_status_summary
from quant_rabbit.guardian_receipt_operator_review import operator_review_status_summary
from quant_rabbit.paths import (
    DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
    DEFAULT_ACTIVE_TRADER_CONTRACT,
    DEFAULT_BIDASK_REPLAY_VALIDATION,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_CAPTURE_ECONOMICS,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_EXECUTION_TIMING_AUDIT,
    DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
    DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
    DEFAULT_OANDA_UNIVERSAL_ROTATION_MINING,
    DEFAULT_OANDA_UNIVERSAL_ROTATION_PACKAGED_RULES,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_POSITION_GUARDIAN_EXECUTION,
    DEFAULT_POSITION_GUARDIAN_HEARTBEAT,
    DEFAULT_POSITION_GUARDIAN_MANAGEMENT,
    DEFAULT_POSITION_MANAGEMENT,
    DEFAULT_PROFIT_CAPTURE_BOT,
    DEFAULT_PROFITABILITY_ACCEPTANCE,
    DEFAULT_QR_TRADER_RUN_WATCHDOG,
    DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER,
    DEFAULT_SELF_IMPROVEMENT_AUDIT,
    DEFAULT_TRADER_SUPPORT_BOT,
    DEFAULT_TRADER_SUPPORT_BOT_REPORT,
    ROOT,
    effective_oanda_universal_rotation_path,
)
from quant_rabbit.operator_manual import (
    JPY_FRESH_ADD_BLOCK_CODE,
    is_operator_manual_position,
    operator_manual_position_packets_from_payload,
)
from quant_rabbit.risk import (
    FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE,
    FORECAST_MARKET_SUPPORT_MIN_SAMPLES,
    FORECAST_MARKET_SUPPORT_MIN_SIGNAL_CONFIDENCE,
)


STATUS_READY = "SUPPORT_READY"
STATUS_BLOCKED = "SUPPORT_BLOCKED"
GUARDIAN_LABEL = "com.quantrabbit.position-guardian"
GUARDIAN_BLOCKER = "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE"
GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKER = "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"
PROFIT_CAPTURE_MISS = "LOSS_CLOSE_PROFIT_CAPTURE_MISSED"
PERSISTENT_DISCIPLINE = "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"
REPAIR_BASKET_SELF_IMPROVEMENT_EXEMPT_P0_CODES = frozenset({PERSISTENT_DISCIPLINE})
TP_HARVEST_REPAIR_MODE = "TP_HARVEST_REPAIR"
OPERATOR_REVIEW_CONSUMPTION_CLEAR_STATUSES = {
    "OPERATOR_REVIEW_CLEARS_RECEIPT",
    "OPERATOR_REVIEW_DURABLY_CONSUMED_RECEIPT",
}
RANGE_FORECAST_ROTATION_BLOCKER = "RANGE_FORECAST_REQUIRES_RANGE_ROTATION"
RANGE_ROTATION_METHOD = "RANGE_ROTATION"
RANGE_ROTATION_COUNTERPART_MISSING = "RANGE_ROTATION_COUNTERPART_MISSING"
OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED = "OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED"
OANDA_AUDIT_ONLY_LOCAL_TP_EDGE_REQUEST = "PROVE_OANDA_AUDIT_ONLY_LOCAL_TP_EDGE"
OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_UNPROVED_STATUS = (
    "OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_UNPROVED"
)
OANDA_AUDIT_ONLY_REPLAY_HISTORY_DAYS = 120
OANDA_AUDIT_ONLY_REPLAY_HISTORY_MIN_COVERED_DAYS = 119.0
DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST = "REPAIR_DIRECTIONAL_INVERSION_COUNTERFACTUAL"
DIRECTIONAL_INVERSION_REPLAY_WAIT_STATUS = "WAITING_FOR_DIRECTIONAL_INVERSION_REPLAY_EVIDENCE"
DIRECTIONAL_INVERSION_REPLAY_REJECTED = "CONTRARIAN_REPLAY_REJECTED"
DIRECTIONAL_INVERSION_REPLAY_LIVE_SUPPORTED = "CONTRARIAN_REPLAY_LIVE_GRADE_SUPPORTED"
DIRECTIONAL_INVERSION_REPLAY_RANK_ONLY = "CONTRARIAN_REPLAY_RANK_ONLY"
DIRECTIONAL_INVERSION_REPLAY_EVIDENCE_PRESENT = "REPEATED_SPREAD_INCLUDED_EVIDENCE_PRESENT"
DIRECTIONAL_INVERSION_REPLAY_EVIDENCE_MISSING = "MISSING_REPEATED_SPREAD_INCLUDED_EVIDENCE"
DIRECTIONAL_INVERSION_REPLAY_PRESERVED_REQUIRES_REFRESH = (
    "PRESERVED_SPREAD_INCLUDED_EVIDENCE_REQUIRES_REFRESH"
)
DIRECTIONAL_INVERSION_REPLAY_SECTIONS = (
    "high_precision_inversion_selectors",
    "qualified_inversion_selectors",
)
DIRECTIONAL_INVERSION_MIN_REPLAY_SAMPLES = 2
DIRECTIONAL_INVERSION_MIN_ACTIVE_DAYS = 2
OANDA_AUDIT_ONLY_REPLAY_HISTORY_GRANULARITIES = "S5,M5"
OANDA_AUDIT_ONLY_REPLAY_HISTORY_ROOT = ROOT / "logs" / "replay" / "oanda_history"
OANDA_HISTORY_FILENAME_RE = re.compile(
    r"^(?P<pair>[A-Z]{3}_[A-Z]{3})_(?P<granularity>S5|M5)_BA_"
    r"(?P<start>\d{8}T\d{6}Z)_(?P<end>\d{8}T\d{6}Z)\.jsonl(?:\.gz)?$"
)
FORECAST_FRONTIER_EVIDENCE_WAIT_STATUS = "FORECAST_FRONTIER_WAITING_FOR_LIVE_PRECISION_EVIDENCE"
FRONTIER_QUOTE_FRESHNESS_WAIT_STATUS = "FRONTIER_WAITING_FOR_FRESH_QUOTE"
FRONTIER_MARGIN_CAPACITY_WAIT_STATUS = "FRONTIER_MARGIN_CAPACITY_WAIT"
ORDER_INTENTS_ARTIFACT_REFRESH_WAIT_STATUS = "ORDER_INTENTS_ARTIFACT_REFRESH_REQUIRED"
PROTECTIVE_FRONTIER_GUARDRAIL_STATUS = "FRONTIER_PROTECTIVE_GUARDRAIL_ACTIVE"
FRONTIER_STRATEGY_PROFILE_EVIDENCE_WAIT_STATUS = "FRONTIER_STRATEGY_PROFILE_WAITING_FOR_NEW_EVIDENCE"
BIDASK_REPLAY_WAIT_STATUS = "BIDASK_REPLAY_WAITING_FOR_FORECAST_SAMPLE_COVERAGE"
BIDASK_COVERAGE_DETAIL_PRESENT = "DETAIL_PRESENT"
BIDASK_COVERAGE_DETAIL_SUMMARY_ONLY = "SUMMARY_ONLY_REPACKAGE_REQUIRED"
BIDASK_COVERAGE_DETAIL_MISSING = "MISSING"
BIDASK_COVERAGE_DETAIL_NOT_REQUIRED = "NO_UNDER_SAMPLED_DETAIL_NEEDED"
TP_PROGRESS_GUARDIAN_WAIT_STATUS = "WAITING_FOR_POSITION_GUARDIAN_LIVE_EVIDENCE"
TP_PROGRESS_LIVE_EVIDENCE_WAIT_STATUS = "WAITING_FOR_LIVE_EVIDENCE_WINDOW"
POSITION_GUARDIAN_LOCK_WAIT_STATUS = "WAITING_FOR_POSITION_GUARDIAN_LOCK_RELEASE"
PENDING_CANCEL_REVIEW_CODE = "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED"
PENDING_CANCEL_RECEIPT_WAIT_STATUS = "WAITING_FOR_TRADER_CANCEL_RECEIPT"
QUOTE_FRESHNESS_BLOCKER_CODES = {
    "STALE_QUOTE",
    "TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE",
}
FORECAST_FRONTIER_BLOCKER_CODES = {
    "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
    "TELEMETRY_FORECAST_NOT_EXECUTABLE_FOR_LIVE",
    "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
}
FRONTIER_MARGIN_CAPACITY_BLOCKER_CODES = {
    "MARGIN_TOO_THIN_FOR_MIN_LOT",
    "LOSS_AND_MARGIN_TOO_THIN_FOR_MIN_LOT",
}
FRONTIER_GUARDRAIL_BLOCKER_CODES = {
    "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
    "LOSS_ASYMMETRY_GUARD_EXCEEDED",
    "REWARD_RISK_TOO_LOW",
    "EXHAUSTION_RANGE_CHASE",
    "BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE",
    "BREAKOUT_FAILURE_MARKET_NOT_RETESTED",
    "PATTERN_REVERSAL_CHASE",
    "RANGE_ROTATION_BROADER_LOCATION_CHASE",
    "RANGE_MARKET_NOT_AT_RAIL",
    "TREND_MARKET_NOT_OPERATING_TREND",
}
MONTH_SCALE_RESIDUAL_BLOCKER_CODES = {
    "MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCKED",
    "MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED",
}
PROFITABILITY_ACCEPTANCE_ARTIFACT_REFRESH_WAIT_STATUS = (
    "PROFITABILITY_ACCEPTANCE_ARTIFACT_REFRESH_REQUIRED"
)
REPAIR_EXEMPTION_CODES = {PERSISTENT_DISCIPLINE, "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE"}
GLOBAL_UNLOCK_BLOCKERS = {
    GUARDIAN_BLOCKER,
    "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE",
    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
}
ACCEPTANCE_LEAK_LOOKBACK_DAYS = 7
REPAIR_REQUEST_CONTRACT_VERSION = "trader_support_repair_request_v1"
REPAIR_AUTOMATION_ALLOWED_ACTIONS = [
    "read_artifacts",
    "edit_code",
    "edit_tests",
    "edit_runtime_contract_docs",
    "run_unit_tests",
    "commit",
    "sync_live_runtime",
]
REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS = [
    "order_send",
    "order_cancel",
    "position_close",
    "launchd_load",
    "launchd_reload",
]
REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS = [
    "direct_oanda_order_write",
    "direct_oanda_trade_close",
    "direct_launchd_mutation",
    "model_api_call_from_quantrabbit_code",
]


@dataclass(frozen=True)
class TraderSupportSummary:
    status: str
    output_path: Path
    report_path: Path
    blockers: list[dict[str, Any]]
    operator_actions: list[dict[str, Any]]
    metrics: dict[str, Any]


def _default_receipt_artifact_path(
    *,
    requested_path: Path,
    default_path: Path,
    output_path: Path,
) -> Path:
    if requested_path != default_path or output_path == DEFAULT_TRADER_SUPPORT_BOT:
        return requested_path
    if output_path.parent.name == default_path.parent.name:
        return output_path.parent / default_path.name
    return requested_path


class TraderSupportBot:
    """Read-only operator panel for live trader support loops.

    The support bot does not send orders and does not load launchd agents. It
    turns existing broker/runtime artifacts into one gate report so the trader
    can see why profit capture, entries, or acceptance are blocked.
    """

    def __init__(
        self,
        *,
        broker_snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        position_management_path: Path = DEFAULT_POSITION_MANAGEMENT,
        position_guardian_management_path: Path = DEFAULT_POSITION_GUARDIAN_MANAGEMENT,
        position_guardian_execution_path: Path = DEFAULT_POSITION_GUARDIAN_EXECUTION,
        position_guardian_heartbeat_path: Path = DEFAULT_POSITION_GUARDIAN_HEARTBEAT,
        self_improvement_audit_path: Path = DEFAULT_SELF_IMPROVEMENT_AUDIT,
        profitability_acceptance_path: Path = DEFAULT_PROFITABILITY_ACCEPTANCE,
        capture_economics_path: Path = DEFAULT_CAPTURE_ECONOMICS,
        execution_timing_audit_path: Path = DEFAULT_EXECUTION_TIMING_AUDIT,
        profit_capture_bot_path: Path = DEFAULT_PROFIT_CAPTURE_BOT,
        qr_trader_run_watchdog_path: Path = DEFAULT_QR_TRADER_RUN_WATCHDOG,
        guardian_receipt_consumption_path: Path = DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
        guardian_receipt_operator_review_path: Path = DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
        active_trader_contract_path: Path = DEFAULT_ACTIVE_TRADER_CONTRACT,
        active_opportunity_board_path: Path = DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
        non_eurusd_live_grade_frontier_path: Path = DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER,
        oanda_rotation_mining_path: Path = DEFAULT_OANDA_UNIVERSAL_ROTATION_MINING,
        oanda_rotation_packaged_path: Path = DEFAULT_OANDA_UNIVERSAL_ROTATION_PACKAGED_RULES,
        bidask_replay_validation_path: Path | None = None,
        oanda_history_root: Path | None = None,
        output_path: Path = DEFAULT_TRADER_SUPPORT_BOT,
        report_path: Path = DEFAULT_TRADER_SUPPORT_BOT_REPORT,
        now_utc: datetime | None = None,
    ) -> None:
        self.broker_snapshot_path = broker_snapshot_path
        self.order_intents_path = order_intents_path
        self.target_state_path = target_state_path
        self.position_management_path = position_management_path
        self.position_guardian_management_path = position_guardian_management_path
        self.position_guardian_execution_path = position_guardian_execution_path
        self.position_guardian_heartbeat_path = position_guardian_heartbeat_path
        self.self_improvement_audit_path = self_improvement_audit_path
        self.profitability_acceptance_path = profitability_acceptance_path
        self.capture_economics_path = _default_receipt_artifact_path(
            requested_path=capture_economics_path,
            default_path=DEFAULT_CAPTURE_ECONOMICS,
            output_path=output_path,
        )
        self.execution_timing_audit_path = execution_timing_audit_path
        self.profit_capture_bot_path = profit_capture_bot_path
        self.qr_trader_run_watchdog_path = qr_trader_run_watchdog_path
        self.guardian_receipt_consumption_path = _default_receipt_artifact_path(
            requested_path=guardian_receipt_consumption_path,
            default_path=DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
            output_path=output_path,
        )
        self.guardian_receipt_operator_review_path = _default_receipt_artifact_path(
            requested_path=guardian_receipt_operator_review_path,
            default_path=DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
            output_path=output_path,
        )
        self.active_trader_contract_path = _default_receipt_artifact_path(
            requested_path=active_trader_contract_path,
            default_path=DEFAULT_ACTIVE_TRADER_CONTRACT,
            output_path=output_path,
        )
        self.active_opportunity_board_path = _default_receipt_artifact_path(
            requested_path=active_opportunity_board_path,
            default_path=DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
            output_path=output_path,
        )
        self.non_eurusd_live_grade_frontier_path = _default_receipt_artifact_path(
            requested_path=non_eurusd_live_grade_frontier_path,
            default_path=DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER,
            output_path=output_path,
        )
        self.oanda_rotation_mining_path = oanda_rotation_mining_path
        self.oanda_rotation_packaged_path = oanda_rotation_packaged_path
        self._read_oanda_rotation = (
            output_path == DEFAULT_TRADER_SUPPORT_BOT
            or oanda_rotation_mining_path != DEFAULT_OANDA_UNIVERSAL_ROTATION_MINING
            or oanda_rotation_packaged_path != DEFAULT_OANDA_UNIVERSAL_ROTATION_PACKAGED_RULES
        )
        self.bidask_replay_validation_path = bidask_replay_validation_path or DEFAULT_BIDASK_REPLAY_VALIDATION
        self._read_bidask_replay_validation = (
            bidask_replay_validation_path is not None or output_path == DEFAULT_TRADER_SUPPORT_BOT
        )
        self.oanda_history_root = (
            oanda_history_root
            if oanda_history_root is not None
            else OANDA_AUDIT_ONLY_REPLAY_HISTORY_ROOT
            if output_path == DEFAULT_TRADER_SUPPORT_BOT
            else _infer_oanda_history_root(
                output_path=output_path,
                artifact_paths=[
                    order_intents_path,
                    broker_snapshot_path,
                    target_state_path,
                    profitability_acceptance_path,
                    capture_economics_path,
                    self_improvement_audit_path,
                    execution_timing_audit_path,
                    oanda_rotation_mining_path,
                    bidask_replay_validation_path or DEFAULT_BIDASK_REPLAY_VALIDATION,
                ],
            )
        )
        self.output_path = output_path
        self.report_path = report_path
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> TraderSupportSummary:
        payload = self.build_payload()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload), encoding="utf-8")
        return TraderSupportSummary(
            status=payload["status"],
            output_path=self.output_path,
            report_path=self.report_path,
            blockers=payload["blockers"],
            operator_actions=payload["operator_actions"],
            metrics=payload["metrics"],
        )

    def build_payload(self) -> dict[str, Any]:
        broker = _read_json(self.broker_snapshot_path)
        intents = _read_json(self.order_intents_path)
        target = _read_json(self.target_state_path)
        position_management = _read_json(self.position_management_path)
        guardian_management = _read_json(self.position_guardian_management_path)
        self_improvement = _read_json(self.self_improvement_audit_path)
        profitability = _read_json(self.profitability_acceptance_path)
        capture_economics = _read_json(self.capture_economics_path)
        timing = _read_json(self.execution_timing_audit_path)
        profit_capture_bot = _read_json(self.profit_capture_bot_path)
        qr_trader_run_watchdog = _watchdog_summary(_read_json(self.qr_trader_run_watchdog_path))
        guardian_receipt_consumption = consumption_status_summary(
            _read_json(self.guardian_receipt_consumption_path)
        )
        guardian_receipt_operator_review = operator_review_status_summary(
            _read_json(self.guardian_receipt_operator_review_path)
        )
        active_trader_contract = _read_json(self.active_trader_contract_path)
        active_opportunity_board = _read_json(self.active_opportunity_board_path)
        non_eurusd_live_grade_frontier = _read_json(self.non_eurusd_live_grade_frontier_path)
        oanda_rotation_effective_path = (
            effective_oanda_universal_rotation_path(
                self.oanda_rotation_mining_path,
                self.oanda_rotation_packaged_path,
            )
            if self._read_oanda_rotation
            else self.oanda_rotation_mining_path
        )
        oanda_rotation = (
            _read_json(oanda_rotation_effective_path)
            if self._read_oanda_rotation
            else {"_missing": True, "_path": str(oanda_rotation_effective_path)}
        )
        bidask_replay_validation = (
            _read_bidask_replay_validation(self.bidask_replay_validation_path)
            if self._read_bidask_replay_validation
            else {"_missing": True, "_path": str(self.bidask_replay_validation_path)}
        )

        guardian = _guardian_status(
            now_utc=self.now_utc,
            execution_path=self.position_guardian_execution_path,
            heartbeat_path=self.position_guardian_heartbeat_path,
        )
        target_summary = _target_summary(target)
        broker_summary = _broker_summary(
            broker,
            target=target_summary,
            oanda_rotation=oanda_rotation,
            bidask_replay_validation=bidask_replay_validation,
        )
        p0_findings = _p0_findings(self_improvement)
        profit_capture = _profit_capture_summary(self_improvement, timing)
        current_profit_capture = _current_profit_capture_summary(profit_capture_bot)
        entry = _entry_readiness_summary(
            intents,
            oanda_rotation=oanda_rotation,
            guardian=guardian,
            current_p0_findings=p0_findings,
            guardian_receipt_consumption=guardian_receipt_consumption,
            guardian_receipt_operator_review=guardian_receipt_operator_review,
            active_trader_contract=active_trader_contract,
            active_opportunity_board=active_opportunity_board,
            non_eurusd_live_grade_frontier=non_eurusd_live_grade_frontier,
        )
        artifact_freshness = _artifact_freshness_summary(
            broker=broker_summary,
            entry=entry,
        )
        entry["artifact_freshness"] = artifact_freshness
        oanda_history_coverage = _oanda_audit_only_history_coverage(
            [
                str(item.get("pair"))
                for item in entry["oanda_audit_only_local_tp_proof_required"]
                if isinstance(item, dict) and item.get("pair")
            ],
            history_root=self.oanda_history_root,
        )
        bidask_current_intent_history_coverage = _bidask_current_intent_history_coverage(
            entry.get("intent_pairs") if isinstance(entry.get("intent_pairs"), list) else [],
            history_root=self.oanda_history_root,
        )
        position = _position_support_summary(position_management, guardian_management, now_utc=self.now_utc)
        acceptance = _acceptance_summary(
            profitability,
            bidask_current_intent_history_coverage=bidask_current_intent_history_coverage,
        )
        acceptance_freshness = _profitability_acceptance_freshness_summary(
            profitability=profitability,
            capture_economics=capture_economics,
            entry=entry,
        )
        acceptance["artifact_freshness"] = acceptance_freshness

        blockers = _build_blockers(
            guardian=guardian,
            target=target_summary,
            profit_capture=profit_capture,
            p0_findings=p0_findings,
            entry=entry,
            acceptance=acceptance,
            qr_trader_run_watchdog=qr_trader_run_watchdog,
            guardian_receipt_consumption=guardian_receipt_consumption,
            guardian_receipt_operator_review=guardian_receipt_operator_review,
        )
        status = STATUS_BLOCKED if blockers else STATUS_READY
        actions = _operator_actions(
            status=status,
            blockers=blockers,
            guardian=guardian,
            profit_capture=profit_capture,
            entry=entry,
            acceptance=acceptance,
            broker=broker_summary,
            oanda_history_coverage=oanda_history_coverage,
            qr_trader_run_watchdog=qr_trader_run_watchdog,
            guardian_receipt_consumption=guardian_receipt_consumption,
            guardian_receipt_operator_review=guardian_receipt_operator_review,
        )
        repair_requests = _build_repair_requests(
            guardian=guardian,
            profit_capture=profit_capture,
            entry=entry,
            acceptance=acceptance,
            p0_findings=p0_findings,
            broker=broker_summary,
            target=target_summary,
            oanda_history_coverage=oanda_history_coverage,
        )
        send_allowed = (
            status == STATUS_READY
            and bool(entry["live_ready_lanes"])
            and (not guardian["required"] or bool(guardian["active"]))
            and guardian_receipt_consumption.get("normal_routing_allowed") is not False
            and not _guardian_receipt_operator_review_blocks_normal_routing(
                guardian_receipt_consumption,
                guardian_receipt_operator_review,
            )
        )
        repair_basket_self_improvement_blockers = _repair_basket_self_improvement_blockers(
            p0_findings,
            entry["repair_live_ready"],
        )
        repair_basket_send_allowed = (
            bool(entry["repair_live_ready"])
            and (not guardian["required"] or bool(guardian["active"]))
            and not repair_basket_self_improvement_blockers
            and not _guardian_receipt_operator_review_blocks_normal_routing(
                guardian_receipt_consumption,
                guardian_receipt_operator_review,
            )
        )
        _annotate_operational_target_firepower(
            acceptance["target_firepower"],
            audit_minimum_reachable=bool(
                acceptance["target_firepower"]["minimum_5pct_estimated_reachable"]
            ),
            send_allowed=send_allowed,
            status=status,
            blockers=blockers,
            guardian=guardian,
            entry=entry,
        )
        metrics = {
            "send_fresh_entries_allowed": send_allowed,
            "repair_basket_send_allowed": repair_basket_send_allowed,
            "repair_basket_self_improvement_blocker_codes": repair_basket_self_improvement_blockers,
            "guardian_active": guardian["active"],
            "guardian_heartbeat_fresh": guardian["heartbeat_fresh"],
            "live_ready_lanes": entry["live_ready_lanes"],
            "repair_live_ready_lanes": len(entry["repair_live_ready"]),
            "repair_basket_guardian_recovery_lanes": len(entry["repair_basket_guardian_recovery"]),
            "repair_frontier_lanes": len(entry["repair_frontier"]),
            "repair_frontier_superseded_by_range_forecast_lanes": len(
                entry["repair_frontier_superseded_by_range_forecast"]
            ),
            "oanda_audit_only_local_tp_proof_required_lanes": len(
                entry["oanda_audit_only_local_tp_proof_required"]
            ),
            "oanda_audit_only_with_replay_evidence_lanes": sum(
                1
                for item in entry["oanda_audit_only_local_tp_proof_required"]
                if item.get("oanda_replay_evidence_status") != "MISSING_OANDA_REPLAY_EVIDENCE"
            ),
            "repair_frontier_missing_range_rotation_counterpart_lanes": len(
                entry["repair_frontier_missing_range_rotation_counterpart"]
            ),
            "repair_frontier_after_support_clear_lanes": entry["repair_frontier_after_support_clear_lanes"],
            "repair_frontier_after_support_blocked_lanes": entry["repair_frontier_after_support_blocked_lanes"],
            "repair_frontier_after_support_top_blockers": entry["repair_frontier_remaining_blockers"],
            "near_ready_lane_count": entry["near_ready_lane_count"],
            "near_ready_blocker_groups": entry["near_ready_blocker_groups"],
            "shortest_live_ready_path_lane_id": entry["shortest_live_ready_path"].get("lane_id"),
            "shortest_live_ready_path_status": entry["shortest_live_ready_path"].get("status"),
            "shortest_live_ready_path_blocker_groups": entry["shortest_live_ready_path"].get("blocker_groups"),
            "active_path_lane_id": entry["active_path"].get("lane_id"),
            "active_path_pair": entry["active_path"].get("pair"),
            "active_path_status": entry["active_path"].get("status"),
            "active_path_source": entry["active_path"].get("source"),
            "stale_support_blocker_codes": entry["stale_support_blocker_codes"],
            "stale_support_blocker_lanes": entry["stale_support_blocker_lanes"],
            "order_intents_stale_against_broker_snapshot": artifact_freshness[
                "order_intents_stale_against_broker_snapshot"
            ],
            "order_intents_staleness_seconds": artifact_freshness["order_intents_staleness_seconds"],
            "order_intents_generated_at_utc": artifact_freshness["order_intents_generated_at_utc"],
            "broker_snapshot_fetched_at_utc": artifact_freshness["broker_snapshot_fetched_at_utc"],
            "global_unlock_frontier_lanes": len(entry["global_unlock_frontier"]),
            "month_scale_residual_blocked_intent_count": entry["month_scale_residual_blocked_intent_count"],
            "profit_capture_missed_loss_closes": profit_capture["missed_loss_closes"],
            "profit_capture_estimated_gap_jpy": profit_capture["estimated_gap_jpy"],
            "profit_capture_actual_loss_close_pl_jpy": profit_capture["actual_loss_close_pl_jpy"],
            "profit_capture_counterfactual_pl_jpy": profit_capture["counterfactual_profit_capture_pl_jpy"],
            "profit_capture_counterfactual_delta_jpy": profit_capture["counterfactual_profit_capture_delta_jpy"],
            "profit_capture_counterfactual_jpy": profit_capture["counterfactual_profit_capture_jpy"],
            "profit_capture_repair_replay_contract": profit_capture["repair_replay_contract"],
            "profit_capture_repair_replay_contract_present": profit_capture[
                "repair_replay_contract_present"
            ],
            "profit_capture_repair_replay_triggered": profit_capture["repair_replay_triggered"],
            "profit_capture_repair_replay_delta_jpy": profit_capture["repair_replay_delta_jpy"],
            "profit_capture_repair_replay_jpy": profit_capture["repair_replay_jpy"],
            "profit_capture_bankable_positions": current_profit_capture["bankable_positions"],
            "profit_capture_watch_positions": current_profit_capture["watch_positions"],
            "profit_capture_blocked_positions": current_profit_capture["blocked_positions"],
            "open_trader_positions": broker_summary["trader_positions"],
            "open_positions": broker_summary["positions"],
            "unknown_owner_positions": broker_summary["unknown_owner_positions"],
            "operator_manual_positions": broker_summary["operator_manual_positions"],
            "operator_manual_jpy_fresh_add_block_active": broker_summary[
                "operator_manual_jpy_fresh_add_block_active"
            ],
            "operator_manual_jpy_fresh_add_block_code": broker_summary[
                "operator_manual_jpy_fresh_add_block_code"
            ],
            "directional_inversion_counterfactual_count": len(
                broker_summary["directional_inversion_counterfactuals"]
            ),
            "directional_inversion_counterfactual_minimum_5pct_count": sum(
                1
                for item in broker_summary["directional_inversion_counterfactuals"]
                if item.get("would_clear_minimum_5pct")
            ),
            "directional_inversion_counterfactual_actionable_count": sum(
                1
                for item in broker_summary["directional_inversion_counterfactuals"]
                if _directional_inversion_counterfactual_is_actionable(item)
            ),
            "directional_inversion_counterfactual_replay_rejected_count": sum(
                1
                for item in broker_summary["directional_inversion_counterfactuals"]
                if (
                    isinstance(item.get("replay_verification"), dict)
                    and item["replay_verification"].get("status") == DIRECTIONAL_INVERSION_REPLAY_REJECTED
                )
            ),
            "target_remaining_jpy": target_summary["remaining_target_jpy"],
            "profitability_status": acceptance["status"],
            "profitability_acceptance_artifact_status": acceptance_freshness["status"],
            "profitability_acceptance_generated_at_utc": acceptance_freshness[
                "profitability_acceptance_generated_at_utc"
            ],
            "profitability_acceptance_stale_against_inputs": acceptance_freshness[
                "profitability_acceptance_stale_against_inputs"
            ],
            "profitability_acceptance_stale_input_names": acceptance_freshness["stale_input_names"],
            "capture_economics_generated_at_utc": acceptance_freshness["capture_economics_generated_at_utc"],
            "target_firepower_status": acceptance["target_firepower"]["status"],
            "target_firepower_minimum_5pct_estimated_reachable": acceptance["target_firepower"][
                "minimum_5pct_estimated_reachable"
            ],
            "target_firepower_target_10pct_estimated_reachable": acceptance["target_firepower"][
                "target_10pct_estimated_reachable"
            ],
            "target_firepower_operational_minimum_5pct_reachable": acceptance["target_firepower"][
                "operational_minimum_5pct_reachable"
            ],
            "target_firepower_operational_blocker_codes": acceptance["target_firepower"][
                "operational_blocker_codes"
            ],
            "acceptance_evidence_collection_count": len(
                acceptance["repair_plan"].get("evidence_collection_items", [])
            ),
            "repair_request_count": len(repair_requests),
            "repair_request_codes": [item["code"] for item in repair_requests],
            "qr_trader_run_watchdog_status": qr_trader_run_watchdog.get("status"),
            "qr_trader_run_watchdog_severity": qr_trader_run_watchdog.get("severity"),
            "qr_trader_run_watchdog_minutes_since_last_run": qr_trader_run_watchdog.get(
                "minutes_since_last_run"
            ),
            "qr_trader_run_watchdog_missed_expected_window": qr_trader_run_watchdog.get(
                "missed_expected_window"
            ),
            "qr_trader_guardian_receipt_issue_codes": [
                item.get("code")
                for item in qr_trader_run_watchdog.get("guardian_receipt_issues", [])
                if isinstance(item, dict)
            ],
            "guardian_receipt_consumption_status": guardian_receipt_consumption.get("status"),
            "guardian_receipt_consumption_normal_routing_allowed": guardian_receipt_consumption.get(
                "normal_routing_allowed"
            ),
            "guardian_receipt_consumption_unresolved_issue_count": guardian_receipt_consumption.get(
                "unresolved_issue_count"
            ),
            "guardian_receipt_consumption_classifications": [
                item.get("classification")
                for item in guardian_receipt_consumption.get("classifications", [])
                if isinstance(item, dict)
            ],
            "guardian_receipt_operator_review_status": guardian_receipt_operator_review.get("status"),
            "guardian_receipt_operator_review_normal_routing_allowed": guardian_receipt_operator_review.get(
                "normal_routing_allowed"
            ),
            "guardian_receipt_operator_review_unresolved_review_count": guardian_receipt_operator_review.get(
                "unresolved_review_count"
            ),
            "guardian_receipt_operator_review_decisions": [
                item.get("operator_decision")
                for item in guardian_receipt_operator_review.get("classifications", [])
                if isinstance(item, dict)
            ],
            "guardian_receipt_current_unclassified_issue_count": len(
                _current_guardian_receipt_unclassified_issues(
                    qr_trader_run_watchdog,
                    guardian_receipt_consumption,
                )
            ),
            "guardian_receipt_current_unclassified_issue_event_ids": [
                _guardian_receipt_event_id(item)
                for item in _current_guardian_receipt_unclassified_issues(
                    qr_trader_run_watchdog,
                    guardian_receipt_consumption,
                )
            ],
            "guardian_receipt_recommended_next_action": _guardian_receipt_consumption_next_action(
                qr_trader_run_watchdog,
                guardian_receipt_consumption,
            ),
            "repair_basket_lane_ids": [item["lane_id"] for item in entry["repair_live_ready"]],
            "repair_basket_guardian_recovery_lane_ids": [
                item["lane_id"] for item in entry["repair_basket_guardian_recovery"]
            ],
            "oanda_audit_only_local_tp_proof_required_lane_ids": [
                item["lane_id"] for item in entry["oanda_audit_only_local_tp_proof_required"]
            ],
        }
        generated = self.now_utc.isoformat()
        return {
            "artifact_paths": {
                "broker_snapshot": str(self.broker_snapshot_path),
                "order_intents": str(self.order_intents_path),
                "target_state": str(self.target_state_path),
                "position_management": str(self.position_management_path),
                "position_guardian_management": str(self.position_guardian_management_path),
                "position_guardian_execution": str(self.position_guardian_execution_path),
                "position_guardian_heartbeat": str(self.position_guardian_heartbeat_path),
                "self_improvement_audit": str(self.self_improvement_audit_path),
                "profitability_acceptance": str(self.profitability_acceptance_path),
                "capture_economics": str(self.capture_economics_path),
                "execution_timing_audit": str(self.execution_timing_audit_path),
                "profit_capture_bot": str(self.profit_capture_bot_path),
                "qr_trader_run_watchdog": str(self.qr_trader_run_watchdog_path),
                "guardian_receipt_consumption": str(self.guardian_receipt_consumption_path),
                "guardian_receipt_operator_review": str(self.guardian_receipt_operator_review_path),
                "active_trader_contract": str(self.active_trader_contract_path),
                "active_opportunity_board": str(self.active_opportunity_board_path),
                "non_eurusd_live_grade_frontier": str(self.non_eurusd_live_grade_frontier_path),
                "oanda_rotation_mining": str(self.oanda_rotation_mining_path),
                "oanda_rotation_packaged": str(self.oanda_rotation_packaged_path),
                "oanda_rotation_effective": str(oanda_rotation_effective_path),
                "bidask_replay_validation": str(self.bidask_replay_validation_path),
                "oanda_history_root": str(self.oanda_history_root),
            },
            "generated_at_utc": generated,
            "status": status,
            "blockers": blockers,
            "operator_actions": actions,
            "repair_requests": repair_requests,
            "metrics": metrics,
            "guardian": guardian,
            "broker": broker_summary,
            "target": target_summary,
            "profit_capture": profit_capture,
            "qr_trader_run_watchdog": qr_trader_run_watchdog,
            "guardian_receipt_consumption": guardian_receipt_consumption,
            "guardian_receipt_operator_review": guardian_receipt_operator_review,
            "current_profit_capture": current_profit_capture,
            "entry_readiness": entry,
            "oanda_history_coverage": oanda_history_coverage,
            "bidask_current_intent_history_coverage": bidask_current_intent_history_coverage,
            "position_support": position,
            "self_improvement": {
                "status": self_improvement.get("status") if isinstance(self_improvement, dict) else None,
                "p0_findings": p0_findings,
                "p0_count": len(p0_findings),
            },
            "profitability_acceptance": acceptance,
            "read_only": True,
            "live_side_effects": [],
        }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _watchdog_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("_missing"):
        return {
            "status": "UNAVAILABLE",
            "severity": "INFO",
            "missing": True,
            "path": payload.get("_path") if isinstance(payload, dict) else None,
            "guardian_receipt_issues": [],
        }
    guardian = payload.get("guardian_receipt") if isinstance(payload.get("guardian_receipt"), dict) else {}
    guardian_issues = guardian.get("issues") if isinstance(guardian.get("issues"), list) else []
    return {
        "status": payload.get("status"),
        "severity": payload.get("severity"),
        "missing": False,
        "generated_at_utc": payload.get("generated_at_utc"),
        "last_trader_run_at": payload.get("last_trader_run_at"),
        "last_trader_run_source": payload.get("last_trader_run_source"),
        "last_trader_run_path": payload.get("last_trader_run_path"),
        "rejected_timestamp_candidates": payload.get("rejected_timestamp_candidates")
        if isinstance(payload.get("rejected_timestamp_candidates"), list)
        else [],
        "minutes_since_last_run": payload.get("minutes_since_last_run"),
        "missed_expected_window": payload.get("missed_expected_window"),
        "expected_cadence_minutes": payload.get("expected_cadence_minutes"),
        "grace_minutes": payload.get("grace_minutes"),
        "suspected_cause": payload.get("suspected_cause"),
        "recommended_operator_action": payload.get("recommended_operator_action"),
        "guardian_receipt": guardian,
        "guardian_receipt_issues": guardian_issues,
        "issues": payload.get("issues") if isinstance(payload.get("issues"), list) else [],
        "artifact_paths": payload.get("artifact_paths") if isinstance(payload.get("artifact_paths"), dict) else {},
    }


def _watchdog_counts_as_support_blocker(payload: dict[str, Any]) -> bool:
    if payload.get("missing") or payload.get("status") == "UNAVAILABLE":
        return False
    return str(payload.get("status") or "") in {"BROKEN", "STALE", "UNKNOWN"}


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


def _read_bidask_replay_validation(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("_missing"):
        return payload
    payload = dict(payload)
    payload.setdefault("_path", str(path))
    history: list[dict[str, Any]] = []
    seen: set[Path] = set()
    candidates = [path]
    if path.parent.exists():
        candidates.extend(sorted(path.parent.glob("oanda_history_replay_validate_*.json")))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        try:
            item = _read_json(candidate)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if item.get("_missing"):
            continue
        item = dict(item)
        item["_path"] = str(candidate)
        history.append(item)
    if history:
        history.sort(
            key=lambda item: _parse_utc(item.get("generated_at_utc")) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        payload["_validation_history"] = history
    return payload


def _age_seconds(value: Any, *, now_utc: datetime) -> float | None:
    parsed = _parse_utc(value)
    if parsed is None:
        return None
    return max(0.0, (now_utc - parsed).total_seconds())


def _truthy(raw: Any) -> bool:
    return str(raw).strip() in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def _truthy_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _truthy(raw)


def _int_env(name: str, *, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _guardian_status(*, now_utc: datetime, execution_path: Path, heartbeat_path: Path) -> dict[str, Any]:
    label = os.environ.get("QR_POSITION_GUARDIAN_LABEL", GUARDIAN_LABEL)
    plist = Path(
        os.environ.get(
            "QR_POSITION_GUARDIAN_PLIST",
            str(Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"),
        )
    ).expanduser()
    interval = _int_env("QR_POSITION_GUARDIAN_INTERVAL", default=30, minimum=15)
    max_age = _int_env("QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS", default=interval * 4, minimum=interval)
    heartbeat_required = _truthy_env("QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT", default=True)
    required = _truthy_env("QR_REQUIRE_POSITION_GUARDIAN_ACTIVE", default=True)
    paths = _guardian_heartbeat_candidates(
        execution_path=execution_path,
        heartbeat_path=heartbeat_path,
    )
    heartbeat = _freshest_guardian_heartbeat(paths, now_utc=now_utc, max_age=max_age)
    runtime_lock = _live_runtime_lock_status(now_utc=now_utc)
    env_active_raw = os.environ.get("QR_POSITION_GUARDIAN_ACTIVE")

    status: dict[str, Any] = {
        "required": required,
        "active": False,
        "active_source": "launchd+heartbeat",
        "label": label,
        "plist_path": str(plist),
        "plist_exists": plist.exists(),
        "launchd_loaded": None,
        "heartbeat_required": heartbeat_required,
        "heartbeat_max_age_seconds": max_age,
        "heartbeat_fresh": heartbeat["fresh"],
        "heartbeat_age_seconds": heartbeat["age_seconds"],
        "heartbeat_generated_at_utc": heartbeat["generated_at_utc"],
        "heartbeat_path": heartbeat["path"],
        "heartbeat_status": heartbeat["status"],
        "heartbeat_candidates": [str(path) for path in paths],
        "live_runtime_lock": runtime_lock,
        "live_runtime_lock_active": runtime_lock["active"],
        "live_runtime_lock_command": runtime_lock["command"],
        "live_runtime_lock_age_seconds": runtime_lock["age_seconds"],
        "live_runtime_lock_pid": runtime_lock["pid"],
    }
    if env_active_raw is not None:
        env_active = _truthy(env_active_raw)
        status["env_active"] = env_active_raw
        status["launchd_loaded"] = bool(env_active)
        status["active_source"] = "env+heartbeat"
        status["active"] = bool(env_active and (heartbeat["fresh"] or not heartbeat_required))
        if env_active and heartbeat_required and not heartbeat["fresh"]:
            status["active_source"] = "live_runtime_lock_busy" if runtime_lock["active"] else "stale_heartbeat"
        return status
    if not plist.exists():
        status["active_source"] = "plist_missing"
        status["launchd_loaded"] = False
        return status
    loaded = _launchd_loaded(label)
    status["launchd_loaded"] = loaded["loaded"]
    if loaded.get("error"):
        status["launchctl_error"] = loaded["error"]
        status["active_source"] = "launchctl_unavailable"
    status["active"] = bool(loaded["loaded"] and (heartbeat["fresh"] or not heartbeat_required))
    if loaded["loaded"] and heartbeat_required and not heartbeat["fresh"]:
        status["active_source"] = "live_runtime_lock_busy" if runtime_lock["active"] else "stale_heartbeat"
    return status


def _guardian_heartbeat_candidates(*, execution_path: Path, heartbeat_path: Path) -> list[Path]:
    execution_env = os.environ.get("QR_POSITION_GUARDIAN_EXECUTION")
    heartbeat_env = os.environ.get("QR_POSITION_GUARDIAN_HEARTBEAT")
    paths = [
        Path(execution_env or str(execution_path)).expanduser(),
        Path(heartbeat_env or str(heartbeat_path)).expanduser(),
    ]
    if execution_env is None and _same_path(execution_path, DEFAULT_POSITION_GUARDIAN_EXECUTION):
        paths.append(_default_live_root() / "data" / DEFAULT_POSITION_GUARDIAN_EXECUTION.name)
    if heartbeat_env is None and _same_path(heartbeat_path, DEFAULT_POSITION_GUARDIAN_HEARTBEAT):
        paths.append(_default_live_root() / "data" / DEFAULT_POSITION_GUARDIAN_HEARTBEAT.name)
    return _dedupe_paths(paths)


def _default_live_root() -> Path:
    return Path(
        os.environ.get("QR_SYNC_LIVE_ROOT") or str(ROOT.parent / "QuantRabbit-live")
    ).expanduser()


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = _path_key(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _same_path(left: Path, right: Path) -> bool:
    return _path_key(left) == _path_key(right)


def _path_key(path: Path) -> str:
    try:
        return str(path.expanduser().resolve(strict=False))
    except OSError:
        return str(path.expanduser())


def _live_runtime_lock_status(*, now_utc: datetime) -> dict[str, Any]:
    lock_dir = Path(
        os.environ.get("QR_AUTOTRADE_LOCK_DIR") or str(ROOT / ".quant_rabbit_live.lock")
    ).expanduser()
    held_by_current_process = os.environ.get("QR_AUTOTRADE_LOCK_HELD") == "1"
    pid = _read_int_file(lock_dir / "pid")
    command = _read_text_file(lock_dir / "command")
    started_at = _read_text_file(lock_dir / "started_at_utc")
    active = bool(held_by_current_process)
    stale = False
    if pid is not None:
        if _process_is_running(pid):
            active = True
        else:
            stale = True
    age = _age_seconds(started_at, now_utc=now_utc) if started_at else None
    return {
        "path": str(lock_dir),
        "exists": lock_dir.exists(),
        "active": active,
        "stale": stale,
        "held_by_current_process": held_by_current_process,
        "pid": pid if pid is not None else (os.getpid() if held_by_current_process else None),
        "command": command,
        "started_at_utc": started_at,
        "age_seconds": round(age, 3) if age is not None else None,
    }


def _read_text_file(path: Path) -> str | None:
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    return value or None


def _read_int_file(path: Path) -> int | None:
    value = _read_text_file(path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _launchd_loaded(label: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["launchctl", "list", label],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        return {"loaded": False, "error": exc.__class__.__name__}
    if proc.returncode == 0:
        return {"loaded": True}
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        return {"loaded": False, "error": exc.__class__.__name__}
    return {"loaded": proc.returncode == 0}


def _freshest_guardian_heartbeat(paths: list[Path], *, now_utc: datetime, max_age: int) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        generated = payload.get("generated_at_utc")
        age = _age_seconds(generated, now_utc=now_utc)
        if age is None:
            continue
        item = {
            "path": str(path),
            "generated_at_utc": generated,
            "age_seconds": round(age, 3),
            "fresh": age <= max_age,
            "status": payload.get("status"),
        }
        if best is None or (item["age_seconds"] or 0) < (best["age_seconds"] or 0):
            best = item
    if best is None:
        return {
            "path": None,
            "generated_at_utc": None,
            "age_seconds": None,
            "fresh": False,
            "status": "MISSING",
        }
    return best


def _broker_summary(
    payload: dict[str, Any],
    *,
    target: dict[str, Any] | None = None,
    oanda_rotation: dict[str, Any] | None = None,
    bidask_replay_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    positions = payload.get("positions") if isinstance(payload.get("positions"), list) else []
    orders = payload.get("orders") if isinstance(payload.get("orders"), list) else []
    trader_positions = [item for item in positions if str(item.get("owner") or "").lower() == "trader"]
    target = target if isinstance(target, dict) else {}
    rows = []
    for item in trader_positions:
        rows.append(
            {
                "trade_id": item.get("trade_id"),
                "pair": item.get("pair"),
                "side": item.get("side"),
                "units": item.get("units"),
                "unrealized_pl_jpy": _round_optional(item.get("unrealized_pl_jpy"), 3),
                "take_profit": item.get("take_profit"),
                "stop_loss": item.get("stop_loss"),
            }
        )
    open_rows = [_compact_open_position(item) for item in positions[:12] if isinstance(item, dict)]
    operator_manual_packets = operator_manual_position_packets_from_payload(payload)
    operator_manual_trade_ids = {
        str(trade_id)
        for packet in operator_manual_packets
        for trade_id in (packet.get("trade_ids") if isinstance(packet.get("trade_ids"), list) else [])
    }
    operator_manual_rows = [
        item
        for item in open_rows
        if is_operator_manual_position(item) or str(item.get("trade_id")) in operator_manual_trade_ids
    ]
    unknown_positions = [
        item
        for item in open_rows
        if str(item.get("owner") or "").lower() not in {"trader", "bot"}
        and not (is_operator_manual_position(item) or str(item.get("trade_id")) in operator_manual_trade_ids)
    ]
    operator_manual_jpy_block = any(
        packet.get("pair") == "USD_JPY"
        and packet.get("side") == "SHORT"
        and bool(packet.get("blocks_fresh_jpy_adds"))
        for packet in operator_manual_packets
    )
    account = payload.get("account") if isinstance(payload.get("account"), dict) else {}
    return {
        "fetched_at_utc": payload.get("fetched_at_utc") or account.get("fetched_at_utc"),
        "positions": len(positions),
        "orders": len(orders),
        "trader_positions": len(trader_positions),
        "profitable_trader_positions": sum(1 for item in trader_positions if _float(item.get("unrealized_pl_jpy")) > 0),
        "trader_unrealized_pl_jpy": _round_optional(
            sum(_float(item.get("unrealized_pl_jpy")) for item in trader_positions),
            3,
        ),
        "balance_jpy": _round_optional(account.get("balance_jpy"), 3),
        "nav_jpy": _round_optional(account.get("nav_jpy"), 3),
        "margin_available_jpy": _round_optional(account.get("margin_available_jpy"), 3),
        "open_trader_positions": rows,
        "open_positions": open_rows,
        "unknown_owner_positions": len(unknown_positions),
        "unknown_owner_position_examples": unknown_positions[:5],
        "operator_manual_positions": len(operator_manual_rows),
        "operator_manual_position_examples": operator_manual_rows[:5],
        "operator_manual_position_packets": operator_manual_packets,
        "operator_manual_jpy_fresh_add_block_active": operator_manual_jpy_block,
        "operator_manual_jpy_fresh_add_block_code": JPY_FRESH_ADD_BLOCK_CODE if operator_manual_jpy_block else None,
        "directional_inversion_counterfactuals": _directional_inversion_counterfactuals(
            open_rows,
            target=target,
            oanda_rotation=oanda_rotation,
            bidask_replay_validation=bidask_replay_validation,
        ),
    }


def _artifact_freshness_summary(
    *,
    broker: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    intents_generated_raw = entry.get("generated_at_utc")
    broker_fetched_raw = broker.get("fetched_at_utc")
    intents_generated = _parse_utc(intents_generated_raw)
    broker_fetched = _parse_utc(broker_fetched_raw)
    staleness_seconds: float | None = None
    stale = False
    status = "ARTIFACT_FRESHNESS_UNKNOWN"
    reason = "order_intents or broker snapshot timestamp is unavailable"
    if intents_generated is not None and broker_fetched is not None:
        staleness_seconds = (broker_fetched - intents_generated).total_seconds()
        stale = staleness_seconds > 1.0
        if stale:
            status = ORDER_INTENTS_ARTIFACT_REFRESH_WAIT_STATUS
            reason = (
                "broker_snapshot was fetched after order_intents was generated; "
                "regenerate intents before treating LIVE_READY or repair-frontier blockers as current"
            )
        else:
            status = "ORDER_INTENTS_ALIGNED_WITH_BROKER_SNAPSHOT"
            reason = "order_intents is generated from the current broker evidence packet"
    return {
        "status": status,
        "order_intents_generated_at_utc": intents_generated_raw,
        "broker_snapshot_fetched_at_utc": broker_fetched_raw,
        "order_intents_stale_against_broker_snapshot": stale,
        "order_intents_staleness_seconds": _round_optional(staleness_seconds, 3),
        "refresh_required": stale,
        "reason": reason,
        "refresh_commands": (
            [
                "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts",
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator",
            ]
            if stale
            else []
        ),
    }


def _profitability_acceptance_freshness_summary(
    *,
    profitability: dict[str, Any],
    capture_economics: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    acceptance_generated_raw = profitability.get("generated_at_utc") or profitability.get("generated_at")
    capture_generated_raw = capture_economics.get("generated_at_utc") or capture_economics.get("generated_at")
    intents_generated_raw = entry.get("generated_at_utc")
    acceptance_generated = _parse_utc(acceptance_generated_raw)
    capture_generated = _parse_utc(capture_generated_raw)
    intents_generated = _parse_utc(intents_generated_raw)

    stale_inputs: list[str] = []
    if (
        acceptance_generated is not None
        and capture_generated is not None
        and (capture_generated - acceptance_generated).total_seconds() > 1.0
    ):
        stale_inputs.append("capture_economics")
    if (
        acceptance_generated is not None
        and intents_generated is not None
        and (intents_generated - acceptance_generated).total_seconds() > 1.0
    ):
        stale_inputs.append("order_intents")

    capture_after_intents = (
        capture_generated is not None
        and intents_generated is not None
        and (capture_generated - intents_generated).total_seconds() > 1.0
    )
    stale = bool(stale_inputs)
    status = (
        PROFITABILITY_ACCEPTANCE_ARTIFACT_REFRESH_WAIT_STATUS
        if stale
        else "PROFITABILITY_ACCEPTANCE_ALIGNED_WITH_INPUTS"
        if acceptance_generated is not None
        else "PROFITABILITY_ACCEPTANCE_FRESHNESS_UNKNOWN"
    )
    if stale:
        if capture_after_intents:
            reason = (
                "profitability_acceptance is older than current input evidence, and capture_economics "
                "is newer than order_intents; regenerate intents before recomputing acceptance"
            )
        else:
            reason = (
                "profitability_acceptance is older than current input evidence: "
                + ", ".join(stale_inputs)
            )
    elif acceptance_generated is None:
        reason = "profitability_acceptance generated timestamp is unavailable"
    else:
        reason = "profitability_acceptance is aligned with current capture_economics and order_intents"

    refresh_commands = []
    if stale:
        if capture_after_intents:
            refresh_commands.append(
                "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts"
            )
        refresh_commands.extend(
            [
                "PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance",
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator",
            ]
        )

    return {
        "status": status,
        "profitability_acceptance_generated_at_utc": acceptance_generated_raw,
        "capture_economics_generated_at_utc": capture_generated_raw,
        "order_intents_generated_at_utc": intents_generated_raw,
        "profitability_acceptance_stale_against_inputs": stale,
        "stale_input_names": stale_inputs,
        "capture_economics_generated_after_order_intents": capture_after_intents,
        "refresh_required": stale,
        "reason": reason,
        "refresh_commands": refresh_commands,
    }


def _compact_open_position(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    packet = raw.get("operator_manual_position") if isinstance(raw.get("operator_manual_position"), dict) else None
    return {
        "trade_id": item.get("trade_id"),
        "pair": item.get("pair"),
        "side": item.get("side"),
        "units": item.get("units"),
        "owner": item.get("owner"),
        "unrealized_pl_jpy": _round_optional(item.get("unrealized_pl_jpy"), 3),
        "take_profit": item.get("take_profit") or item.get("take_profit_price"),
        "stop_loss": item.get("stop_loss") or item.get("stop_loss_price"),
        "operator_manual_position": packet,
        "raw": {"operator_manual_position": packet} if packet else {},
    }


def _directional_inversion_counterfactuals(
    positions: list[dict[str, Any]],
    *,
    target: dict[str, Any],
    oanda_rotation: dict[str, Any] | None = None,
    bidask_replay_validation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    remaining_minimum = _float(target.get("remaining_minimum_jpy"))
    minimum_target = _float(target.get("minimum_target_jpy"))
    minimum_needed = remaining_minimum if remaining_minimum > 0 else minimum_target
    rows: list[dict[str, Any]] = []
    for item in positions:
        upl = _float(item.get("unrealized_pl_jpy"))
        if upl >= 0:
            continue
        side = str(item.get("side") or "").upper()
        opposite = "SHORT" if side == "LONG" else "LONG" if side == "SHORT" else "UNKNOWN"
        counterfactual_profit = -upl
        row = {
            "trade_id": item.get("trade_id"),
            "pair": item.get("pair"),
            "actual_side": side or item.get("side"),
            "opposite_side": opposite,
            "units": item.get("units"),
            "owner": item.get("owner"),
            "actual_unrealized_pl_jpy": _round_optional(upl, 3),
            "opposite_gross_counterfactual_pl_jpy": _round_optional(counterfactual_profit, 3),
            "remaining_minimum_jpy": _round_optional(remaining_minimum, 3),
            "minimum_target_jpy": _round_optional(minimum_target, 3),
            "would_clear_minimum_5pct": bool(minimum_needed > 0 and counterfactual_profit >= minimum_needed),
            "counterfactual_basis": (
                "gross sign-flip of current broker-truth unrealized P/L; requires spread-included "
                "entry timing replay before it can become a live inversion rule"
            ),
        }
        replay_verification = _directional_inversion_replay_verification(
            row,
            bidask_replay_validation,
        )
        if replay_verification is not None:
            row["replay_verification"] = replay_verification
        inversion_evidence = _matching_inversion_replay_evidence(
            row,
            oanda_rotation,
            allow_preserved=False,
        )
        preserved_inversion_evidence = (
            None
            if inversion_evidence is not None
            else _matching_inversion_replay_evidence(
                row,
                oanda_rotation,
                allow_preserved=True,
            )
        )
        row["has_repeated_spread_included_inversion_evidence"] = bool(inversion_evidence)
        if inversion_evidence is not None:
            row["inversion_replay_evidence_status"] = DIRECTIONAL_INVERSION_REPLAY_EVIDENCE_PRESENT
        elif preserved_inversion_evidence is not None:
            row["inversion_replay_evidence_status"] = (
                DIRECTIONAL_INVERSION_REPLAY_PRESERVED_REQUIRES_REFRESH
            )
        else:
            row["inversion_replay_evidence_status"] = DIRECTIONAL_INVERSION_REPLAY_EVIDENCE_MISSING
        if inversion_evidence is not None:
            row["inversion_replay_evidence"] = inversion_evidence
        elif preserved_inversion_evidence is not None:
            row["preserved_inversion_replay_evidence"] = preserved_inversion_evidence
        rows.append(row)
    rows.sort(
        key=lambda item: (
            not bool(item.get("would_clear_minimum_5pct")),
            -_float(item.get("opposite_gross_counterfactual_pl_jpy")),
            str(item.get("pair") or ""),
        )
    )
    return rows[:12]


def _matching_inversion_replay_evidence(
    item: dict[str, Any],
    oanda_rotation: dict[str, Any] | None,
    *,
    allow_preserved: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(oanda_rotation, dict):
        return None
    pair = str(item.get("pair") or "").upper()
    actual_side = str(item.get("actual_side") or "").upper()
    opposite_side = str(item.get("opposite_side") or "").upper()
    if not pair or not actual_side or not opposite_side:
        return None
    candidates: list[dict[str, Any]] = []
    for section_rank, section in enumerate(DIRECTIONAL_INVERSION_REPLAY_SECTIONS):
        rows = oanda_rotation.get(section) if isinstance(oanda_rotation.get(section), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("pair") or "").upper() != pair:
                continue
            if str(row.get("qualification") or "").upper() != "PASS":
                continue
            source_side = str(row.get("source_side") or row.get("actual_side") or "").upper()
            selected_side = str(row.get("selected_side") or row.get("opposite_side") or row.get("side") or "").upper()
            if source_side != actual_side or selected_side != opposite_side:
                continue
            preserved_from_existing = _oanda_rotation_row_is_preserved(row)
            if preserved_from_existing and not allow_preserved:
                continue
            if not (row.get("source_shape") or row.get("shape") or row.get("method")):
                continue
            if _float(row.get("validation_n") or row.get("all_n")) < DIRECTIONAL_INVERSION_MIN_REPLAY_SAMPLES:
                continue
            if _float(row.get("active_days")) < DIRECTIONAL_INVERSION_MIN_ACTIVE_DAYS:
                continue
            if _float(row.get("validation_profit_factor") or row.get("all_profit_factor")) <= 1.0:
                continue
            if _float(row.get("validation_inversion_edge_atr") or row.get("validation_avg_realized_atr")) <= 0.0:
                continue
            candidates.append(
                {
                    "_section_rank": section_rank,
                    "_sort_validation_n": _float(row.get("validation_n") or row.get("all_n")),
                    "_sort_profit_factor": _float(
                        row.get("validation_profit_factor") or row.get("all_profit_factor")
                    ),
                    "_sort_win_rate": _float(row.get("validation_win_rate") or row.get("all_win_rate")),
                    "_sort_active_days": _float(row.get("active_days")),
                    "_sort_edge_atr": _float(
                        row.get("validation_inversion_edge_atr") or row.get("validation_avg_realized_atr")
                    ),
                    "source_section": section,
                    "pair": pair,
                    "source_side": source_side,
                    "selected_side": selected_side,
                    "source_shape": row.get("source_shape"),
                    "shape": row.get("shape"),
                    "method": row.get("method"),
                    "exit_shape": row.get("exit_shape"),
                    "qualification": row.get("qualification"),
                    "validation_n": row.get("validation_n"),
                    "validation_win_rate": row.get("validation_win_rate"),
                    "validation_profit_factor": row.get("validation_profit_factor"),
                    "active_days": row.get("active_days"),
                    "positive_day_rate": row.get("positive_day_rate"),
                    "validation_inversion_edge_atr": row.get("validation_inversion_edge_atr"),
                    "preserved_from_existing_packaged_artifact": preserved_from_existing,
                    "preserved_from_source_report": row.get("preserved_from_source_report"),
                    "preserved_from_generated_at_utc": row.get("preserved_from_generated_at_utc"),
                    "preserved_during_packaging_source_report": row.get(
                        "preserved_during_packaging_source_report"
                    ),
                    "preserved_during_packaging_generated_at_utc": row.get(
                        "preserved_during_packaging_generated_at_utc"
                    ),
                }
            )
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            int(row["_section_rank"]),
            -_float(row["_sort_validation_n"]),
            -_float(row["_sort_profit_factor"]),
            -_float(row["_sort_win_rate"]),
            -_float(row["_sort_active_days"]),
            -_float(row["_sort_edge_atr"]),
        )
    )
    best = dict(candidates[0])
    for key in list(best):
        if key.startswith("_sort_") or key == "_section_rank":
            best.pop(key, None)
    return best


def _oanda_rotation_row_is_preserved(row: dict[str, Any]) -> bool:
    return bool(
        row.get("preserved_from_existing_packaged_artifact")
        or row.get("preserved_because_narrow_source")
    )


def _directional_inversion_counterfactual_has_viable_replay_path(item: dict[str, Any]) -> bool:
    if not item.get("would_clear_minimum_5pct"):
        return False
    replay = item.get("replay_verification")
    if isinstance(replay, dict) and replay.get("status") == DIRECTIONAL_INVERSION_REPLAY_REJECTED:
        return False
    return True


def _directional_inversion_counterfactual_is_actionable(item: dict[str, Any]) -> bool:
    return _directional_inversion_counterfactual_has_viable_replay_path(item) and bool(
        item.get("has_repeated_spread_included_inversion_evidence")
    )


def _directional_inversion_replay_verification(
    item: dict[str, Any],
    validation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(validation, dict) or validation.get("_missing"):
        return None
    pair = str(item.get("pair") or "").upper()
    if not pair:
        return None
    validation = _bidask_replay_validation_for_pair(validation, pair)
    precision = validation.get("precision_rules")
    if not isinstance(precision, dict):
        return None
    pair_filter = _upper_string_list(validation.get("pair_filter"))
    history_pairs = _upper_string_list(validation.get("history_pairs"))
    in_scope = pair in pair_filter if pair_filter else (pair in history_pairs if history_pairs else True)
    if not in_scope:
        return {
            "status": "PAIR_NOT_IN_REPLAY_SCOPE",
            "pair": pair,
            "generated_at_utc": validation.get("generated_at_utc"),
            "source_path": validation.get("_path"),
            "pair_filter": pair_filter,
            "history_pairs": history_pairs[:12],
        }

    contrarian_rules = _rules_for_pair(precision.get("contrarian_edge_rules"), pair)
    daily_stable_contrarian_rules = _rules_for_pair(precision.get("daily_stable_contrarian_edge_rules"), pair)
    negative_rules = _rules_for_pair(precision.get("negative_rules"), pair)
    adoption = precision.get("adoption_summary") if isinstance(precision.get("adoption_summary"), dict) else {}
    if daily_stable_contrarian_rules:
        status = DIRECTIONAL_INVERSION_REPLAY_LIVE_SUPPORTED
    elif contrarian_rules:
        status = DIRECTIONAL_INVERSION_REPLAY_RANK_ONLY
    elif negative_rules:
        status = DIRECTIONAL_INVERSION_REPLAY_REJECTED
    else:
        status = "CONTRARIAN_REPLAY_NO_PAIR_SUPPORT"
    return {
        "status": status,
        "pair": pair,
        "generated_at_utc": validation.get("generated_at_utc"),
        "source_path": validation.get("_path"),
        "granularity": validation.get("granularity"),
        "pair_filter": pair_filter,
        "evaluated_rows": validation.get("evaluated_rows"),
        "price_truth_status": (
            validation.get("price_truth_coverage", {}).get("status")
            if isinstance(validation.get("price_truth_coverage"), dict)
            else None
        ),
        "adoption_summary": adoption,
        "contrarian_edge_rules": len(contrarian_rules),
        "daily_stable_contrarian_edge_rules": len(daily_stable_contrarian_rules),
        "negative_rules": len(negative_rules),
        "negative_rule_names": [str(rule.get("name")) for rule in negative_rules[:5] if rule.get("name")],
    }


def _bidask_replay_validation_for_pair(validation: dict[str, Any], pair: str) -> dict[str, Any]:
    history = validation.get("_validation_history")
    if not isinstance(history, list):
        return validation
    for item in history:
        if isinstance(item, dict) and _bidask_replay_validation_in_scope(item, pair):
            return item
    return validation


def _bidask_replay_validation_in_scope(validation: dict[str, Any], pair: str) -> bool:
    pair_filter = _upper_string_list(validation.get("pair_filter"))
    if pair_filter:
        return pair in pair_filter
    history_pairs = _upper_string_list(validation.get("history_pairs"))
    if history_pairs:
        return pair in history_pairs
    return True


def _upper_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).upper() for item in value if str(item or "").strip()]


def _rules_for_pair(value: Any, pair: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict) and _rule_pair(item) == pair]


def _rule_pair(rule: dict[str, Any]) -> str | None:
    raw_pair = rule.get("pair") or rule.get("instrument")
    if isinstance(raw_pair, str) and raw_pair.strip():
        return raw_pair.upper()
    name = str(rule.get("name") or "").upper()
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] and parts[1]:
        return f"{parts[0]}_{parts[1]}"
    return None


def _target_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "campaign_day_jst": payload.get("campaign_day_jst"),
        "target_open": str(payload.get("status") or "") == "PURSUE_TARGET",
        "progress_pct": _round_optional(payload.get("progress_pct"), 4),
        "minimum_progress_pct": _round_optional(payload.get("minimum_progress_pct"), 4),
        "minimum_target_jpy": _round_optional(payload.get("minimum_target_jpy"), 3),
        "remaining_minimum_jpy": _round_optional(payload.get("remaining_minimum_jpy"), 3),
        "remaining_target_jpy": _round_optional(payload.get("remaining_target_jpy"), 3),
        "realized_pl_jpy": _round_optional(payload.get("realized_pl_jpy"), 3),
        "current_equity_jpy": _round_optional(payload.get("current_equity_jpy"), 3),
        "current_equity_raw": _round_optional(payload.get("current_equity_raw"), 3),
        "capital_flows_30d": _round_optional(payload.get("capital_flows_30d"), 3),
        "funding_adjusted_equity": _round_optional(payload.get("funding_adjusted_equity"), 3),
        "rolling_30d_multiplier_raw": _round_optional(payload.get("rolling_30d_multiplier_raw"), 6),
        "rolling_30d_multiplier_funding_adjusted": _round_optional(
            payload.get("rolling_30d_multiplier_funding_adjusted"),
            6,
        ),
        "remaining_to_4x_raw": _round_optional(payload.get("remaining_to_4x_raw"), 3),
        "remaining_to_4x_funding_adjusted": _round_optional(payload.get("remaining_to_4x_funding_adjusted"), 3),
        "required_calendar_daily_return_raw": _round_optional(payload.get("required_calendar_daily_return_raw"), 6),
        "required_active_day_return_raw": _round_optional(payload.get("required_active_day_return_raw"), 6),
        "required_calendar_daily_return_funding_adjusted": _round_optional(
            payload.get("required_calendar_daily_return_funding_adjusted"),
            6,
        ),
        "required_active_day_return_funding_adjusted": _round_optional(
            payload.get("required_active_day_return_funding_adjusted"),
            6,
        ),
        "performance_basis": payload.get("performance_basis"),
        "sizing_basis": payload.get("sizing_basis"),
        "target_trades_per_day": payload.get("target_trades_per_day"),
    }


def _p0_findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    return [
        {
            "code": item.get("code"),
            "message": item.get("message"),
            "next_action": item.get("next_action"),
            "evidence": item.get("evidence"),
        }
        for item in findings
        if str(item.get("priority") or "").upper() == "P0"
    ]


def _repair_basket_self_improvement_blockers(
    p0_findings: list[dict[str, Any]],
    repair_live_ready: list[dict[str, Any]] | None = None,
) -> list[str]:
    profit_capture_repair_ready = _has_tp_harvest_repair_basket(repair_live_ready)
    blockers: list[str] = []
    for item in p0_findings:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "SELF_IMPROVEMENT_P0").strip()
        if not code or code in REPAIR_BASKET_SELF_IMPROVEMENT_EXEMPT_P0_CODES:
            continue
        if code == PROFIT_CAPTURE_MISS and profit_capture_repair_ready:
            continue
        blockers.append(code)
    return list(dict.fromkeys(blockers))


def _has_tp_harvest_repair_basket(repair_live_ready: list[dict[str, Any]] | None) -> bool:
    if not repair_live_ready:
        return False
    return any(
        isinstance(item, dict)
        and item.get("status") == "LIVE_READY"
        and item.get("repair_mode") == TP_HARVEST_REPAIR_MODE
        for item in repair_live_ready
    )


def _profit_capture_summary(self_improvement: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
    all_findings = self_improvement.get("findings") if isinstance(self_improvement.get("findings"), list) else []
    finding = next(
        (
            item
            for item in all_findings
            if isinstance(item, dict)
            and item.get("code") == PROFIT_CAPTURE_MISS
            and str(item.get("priority") or "").upper() == "P0"
        ),
        None,
    )
    if finding is None:
        finding = next(
            (
                item
                for item in all_findings
                if isinstance(item, dict) and item.get("code") == PROFIT_CAPTURE_MISS
            ),
            None,
        )
    evidence = finding.get("evidence") if isinstance(finding, dict) and isinstance(finding.get("evidence"), dict) else {}
    summary = timing.get("summary") if isinstance(timing.get("summary"), dict) else {}
    repair_replay_contract = repair_replay_contract_from_payload(timing)
    repair_replay_contract_present = repair_replay_contract == TP_PROGRESS_REPAIR_REPLAY_CONTRACT
    missed = evidence.get("loss_closes_profit_capture_missed", summary.get("loss_closes_profit_capture_missed"))
    gap = evidence.get("loss_close_estimated_capture_gap_jpy", summary.get("loss_close_estimated_capture_gap_jpy"))
    actual_pl = evidence.get("loss_close_actual_pl_jpy", summary.get("loss_close_actual_pl_jpy"))
    counterfactual_pl = evidence.get(
        "loss_close_counterfactual_profit_capture_pl_jpy",
        summary.get("loss_close_counterfactual_profit_capture_pl_jpy"),
    )
    counterfactual_delta = evidence.get(
        "loss_close_counterfactual_profit_capture_delta_jpy",
        summary.get("loss_close_counterfactual_profit_capture_delta_jpy"),
    )
    counterfactual_jpy = evidence.get(
        "loss_close_counterfactual_profit_capture_jpy",
        summary.get("loss_close_counterfactual_profit_capture_jpy"),
    )
    repair_replay_triggered = evidence.get(
        "loss_closes_repair_replay_triggered",
        summary.get("loss_closes_repair_replay_triggered"),
    )
    repair_replay_delta = evidence.get(
        "loss_close_repair_replay_delta_jpy",
        summary.get("loss_close_repair_replay_delta_jpy"),
    )
    repair_replay_jpy = evidence.get(
        "loss_close_repair_replay_profit_capture_jpy",
        summary.get("loss_close_repair_replay_profit_capture_jpy"),
    )
    split_present = "post_repair_live_evidence_loss_closes_repair_replay_triggered" in summary or (
        "post_repair_live_evidence_loss_closes_repair_replay_triggered" in evidence
    )
    post_sample_count = int(
        _float(
            evidence.get(
                "post_repair_live_evidence_loss_closes_audited",
                summary.get("post_repair_live_evidence_loss_closes_audited"),
            )
        )
    )
    post_missed = int(
        _float(
            evidence.get(
                "post_repair_live_evidence_loss_closes_profit_capture_missed",
                summary.get("post_repair_live_evidence_loss_closes_profit_capture_missed"),
            )
        )
    )
    post_repair_replay_triggered = int(
        _float(
            evidence.get(
                "post_repair_live_evidence_loss_closes_repair_replay_triggered",
                summary.get("post_repair_live_evidence_loss_closes_repair_replay_triggered"),
            )
        )
    )
    pre_missed = int(
        _float(
            evidence.get(
                "pre_repair_historical_loss_closes_profit_capture_missed",
                summary.get("pre_repair_historical_loss_closes_profit_capture_missed"),
            )
        )
    )
    pre_repair_replay_triggered = int(
        _float(
            evidence.get(
                "pre_repair_historical_loss_closes_repair_replay_triggered",
                summary.get("pre_repair_historical_loss_closes_repair_replay_triggered"),
            )
        )
    )
    top = evidence.get("top_profit_capture_misses") if isinstance(evidence.get("top_profit_capture_misses"), list) else []
    top_repair = (
        evidence.get("top_repair_replay_triggers")
        if isinstance(evidence.get("top_repair_replay_triggers"), list)
        else []
    )
    if not top_repair:
        rows = timing.get("loss_close_regrets") if isinstance(timing.get("loss_close_regrets"), list) else []
        top_repair = [
            row
            for row in rows
            if isinstance(row, dict) and row.get("repair_replay_triggered_before_loss_close")
        ][:5]
    status = "OK"
    if split_present and post_sample_count <= 0 and _float(missed) > 0:
        status = "WAITING_FOR_POST_REPAIR_SAMPLE"
    elif split_present and post_repair_replay_triggered <= 0 and _float(missed) > 0:
        status = "HISTORICAL_DIAGNOSTIC_ONLY"
    elif _float(repair_replay_triggered if split_present else missed) > 0:
        status = "PROFIT_CAPTURE_REPAIR_REQUIRED"
    if status == "WAITING_FOR_POST_REPAIR_SAMPLE":
        clearance_condition = (
            "wait for the first post-repair live loss-close sample after "
            f"{summary.get('tp_progress_repair_live_evidence_boundary_utc') or TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC}; "
            "do not require historical pre-repair replay triggers to become zero"
        )
    elif status == "HISTORICAL_DIAGNOSTIC_ONLY":
        clearance_condition = (
            "post-repair production-gate replay remains clean; historical pre-repair "
            "misses stay diagnostic and must not be used as the clearance condition"
        )
    elif _float(missed) > 0:
        clearance_condition = (
            "execution-timing-audit reports zero post-repair live-evidence "
            "loss_closes_repair_replay_triggered under the current production-gate "
            "replay contract, and position guardian is proven active before fresh "
            "entries resume; raw or pre-repair misses remain diagnostic unless "
            "post-repair production-gate replay also triggers"
        )
    else:
        clearance_condition = "no missed TP-progress loss close in the active timing audit"
    return {
        "status": status,
        "missed_loss_closes": int(_float(missed)),
        "estimated_gap_jpy": _round_optional(gap, 3),
        "actual_loss_close_pl_jpy": _round_optional(actual_pl, 3),
        "counterfactual_profit_capture_pl_jpy": _round_optional(counterfactual_pl, 3),
        "counterfactual_profit_capture_delta_jpy": _round_optional(counterfactual_delta, 3),
        "counterfactual_profit_capture_jpy": _round_optional(counterfactual_jpy, 3),
        "repair_replay_triggered": int(_float(repair_replay_triggered)),
        "repair_replay_contract": repair_replay_contract,
        "repair_replay_contract_present": repair_replay_contract_present,
        "repair_evidence_split_present": split_present,
        "tp_progress_repair_live_evidence_boundary_utc": summary.get(
            "tp_progress_repair_live_evidence_boundary_utc"
        )
        or evidence.get("tp_progress_repair_live_evidence_boundary_utc")
        or TP_PROGRESS_REPAIR_LIVE_EVIDENCE_BOUNDARY_UTC,
        "tp_progress_repair_live_evidence_status": summary.get(
            "tp_progress_repair_live_evidence_status"
        )
        or evidence.get("tp_progress_repair_live_evidence_status"),
        "pre_repair_historical_missed_loss_closes": pre_missed,
        "pre_repair_historical_repair_replay_triggered": pre_repair_replay_triggered,
        "post_repair_live_evidence_loss_closes_audited": post_sample_count,
        "post_repair_live_evidence_missed_loss_closes": post_missed,
        "post_repair_live_evidence_repair_replay_triggered": (
            post_repair_replay_triggered
        ),
        "repair_replay_delta_jpy": _round_optional(repair_replay_delta, 3),
        "repair_replay_jpy": _round_optional(repair_replay_jpy, 3),
        "stop_loss_missed": evidence.get("stop_loss_closes_profit_capture_missed", summary.get("stop_loss_closes_profit_capture_missed")),
        "top_misses": top[:5],
        "top_repair_replay_triggers": top_repair[:5],
        "message": finding.get("message") if finding else None,
        "next_action": finding.get("next_action") if finding else None,
        "clearance_condition": clearance_condition,
        "verification_command": MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND,
    }


def _current_profit_capture_summary(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    positions = payload.get("positions") if isinstance(payload.get("positions"), list) else []
    return {
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "open_trader_positions": metrics.get("open_trader_positions"),
        "bankable_positions": metrics.get("bankable_positions", 0),
        "watch_positions": metrics.get("watch_positions", 0),
        "blocked_positions": metrics.get("blocked_positions", 0),
        "historical_repair_replay_triggered": metrics.get("historical_repair_replay_triggered"),
        "historical_repair_replay_delta_jpy": metrics.get("historical_repair_replay_delta_jpy"),
        "top_gate_statuses": [
            {
                "trade_id": item.get("trade_id"),
                "pair": item.get("pair"),
                "side": item.get("side"),
                "gate_status": item.get("gate_status"),
                "tp_progress": item.get("tp_progress"),
                "capture_trigger": item.get("capture_trigger"),
            }
            for item in positions[:5]
            if isinstance(item, dict)
        ],
    }


def _entry_readiness_summary(
    payload: dict[str, Any],
    *,
    oanda_rotation: dict[str, Any] | None = None,
    guardian: dict[str, Any] | None = None,
    current_p0_findings: list[dict[str, Any]] | None = None,
    guardian_receipt_consumption: dict[str, Any] | None = None,
    guardian_receipt_operator_review: dict[str, Any] | None = None,
    active_trader_contract: dict[str, Any] | None = None,
    active_opportunity_board: dict[str, Any] | None = None,
    non_eurusd_live_grade_frontier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    intent_pairs = _intent_pairs(results)
    live_ready = [item for item in results if item.get("status") == "LIVE_READY"]
    oanda_evidence_by_vehicle = _oanda_replay_evidence_by_vehicle(oanda_rotation or {})
    codes = Counter()
    stale_support_blocker_codes = _stale_support_blocker_codes(
        results,
        guardian=guardian or {},
        current_p0_findings=current_p0_findings or [],
        guardian_receipt_consumption=guardian_receipt_consumption or {},
        guardian_receipt_operator_review=guardian_receipt_operator_review or {},
    )
    repair_frontier: list[dict[str, Any]] = []
    repair_live_ready: list[dict[str, Any]] = []
    repair_basket_guardian_recovery: list[dict[str, Any]] = []
    repair_frontier_superseded_by_range_forecast: list[dict[str, Any]] = []
    repair_frontier_missing_range_rotation_counterpart: list[dict[str, Any]] = []
    oanda_audit_only_local_tp_proof_required: list[dict[str, Any]] = []
    global_unlock_frontier: list[dict[str, Any]] = []
    month_scale_residual_blocked_intents: list[dict[str, Any]] = []
    near_ready_lanes: list[dict[str, Any]] = []
    range_rotation_counterparts = _range_rotation_counterparts(results)
    for item in results:
        for code in item.get("live_blocker_codes") or []:
            codes[str(code)] += 1
        blocker_codes = [str(code) for code in item.get("live_blocker_codes") or []]
        metadata = _intent_metadata(item)
        if str(item.get("status") or "") != "LIVE_READY" and blocker_codes:
            near_ready_lanes.append(
                _near_ready_lane_summary(
                    item,
                    blocker_codes,
                    metadata,
                    stale_support_blocker_codes=stale_support_blocker_codes,
                )
            )
        residual_blockers = [code for code in blocker_codes if code in MONTH_SCALE_RESIDUAL_BLOCKER_CODES]
        if residual_blockers:
            intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
            context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
            month_scale_residual_blocked_intents.append(
                {
                    "lane_id": item.get("lane_id"),
                    "status": item.get("status"),
                    "pair": intent.get("pair"),
                    "side": intent.get("side"),
                    "method": context.get("method"),
                    "order_type": intent.get("order_type"),
                    "blocker_codes": residual_blockers,
                    "residual_group": metadata.get("month_scale_residual_loss_group"),
                }
            )
        if (
            metadata.get("positive_rotation_oanda_campaign_audit_only") is True
            or OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED in blocker_codes
        ):
            intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
            context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
            vehicle_key = (
                metadata.get("positive_rotation_oanda_campaign_matching_vehicle_key")
                or metadata.get("oanda_campaign_vehicle_key")
            )
            replay_evidence = _compact_oanda_replay_evidence(vehicle_key, oanda_evidence_by_vehicle)
            oanda_audit_only_local_tp_proof_required.append(
                {
                    "lane_id": item.get("lane_id"),
                    "status": item.get("status"),
                    "pair": intent.get("pair"),
                    "side": intent.get("side"),
                    "method": context.get("method"),
                    "order_type": intent.get("order_type"),
                    "reward_jpy": _intent_reward_jpy(item, metadata),
                    "risk_jpy": _intent_risk_jpy(item, metadata),
                    "capture_take_profit_scope": metadata.get("capture_take_profit_scope"),
                    "capture_take_profit_scope_key": metadata.get("capture_take_profit_scope_key"),
                    "oanda_vehicle_key": vehicle_key,
                    "oanda_replay_evidence_status": replay_evidence["evidence_status"],
                    "oanda_replay_live_permission": replay_evidence["live_permission"],
                    "oanda_replay_evidence": replay_evidence,
                    "remaining_blocker_codes_after_guardian": [
                        code for code in blocker_codes if code != GUARDIAN_BLOCKER
                    ],
                    "local_proof_required": True,
                    "historical_replay_can_clear_local_tp_proof": False,
                    "local_tp_proof_clearance_condition": (
                        "exact PAIR_SIDE_METHOD TAKE_PROFIT_ORDER receipts for this "
                        "pair/side/method must show positive expectancy, zero TP losses, "
                        "and positive Wilson-stressed expectancy, unless the exact "
                        "OANDA HARVEST vehicle has already been promoted to live-grade "
                        "by current-risk or normal-cap 5% firepower scaling"
                    ),
                }
            )
        tp_proof = _tp_proof_summary(metadata)
        if str(item.get("status") or "") != "LIVE_READY" and blocker_codes:
            remaining_after_global = [code for code in blocker_codes if code not in GLOBAL_UNLOCK_BLOCKERS]
            if not remaining_after_global:
                intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
                context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
                unlock_item = {
                    "lane_id": item.get("lane_id"),
                    "status": item.get("status"),
                    "pair": intent.get("pair"),
                    "side": intent.get("side"),
                    "method": context.get("method"),
                    "order_type": intent.get("order_type"),
                    "reward_jpy": _intent_reward_jpy(item, metadata),
                    "risk_jpy": _intent_risk_jpy(item, metadata),
                    "global_blocker_codes": blocker_codes,
                    "remaining_blocker_codes_after_global_unlock": remaining_after_global,
                    "repair_mode": metadata.get("self_improvement_p0_repair_mode")
                    or metadata.get("positive_rotation_mode"),
                }
                if tp_proof:
                    unlock_item["tp_proof"] = tp_proof
                global_unlock_frontier.append(unlock_item)
        if metadata.get("self_improvement_p0_repair_live_ready") is True:
            exempt = set(REPAIR_EXEMPTION_CODES)
            stale_support = set(stale_support_blocker_codes)
            remaining_after_support = [
                code
                for code in blocker_codes
                if code != GUARDIAN_BLOCKER and code not in stale_support and code not in exempt
            ]
            intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
            context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
            repair_item = {
                "lane_id": item.get("lane_id"),
                "status": item.get("status"),
                "pair": intent.get("pair"),
                "side": intent.get("side"),
                "method": context.get("method"),
                "order_type": intent.get("order_type"),
                "reward_jpy": _intent_reward_jpy(item, metadata),
                "risk_jpy": _intent_risk_jpy(item, metadata),
                "repair_mode": metadata.get("self_improvement_p0_repair_mode"),
                "matrix_repair_profile_status": metadata.get("matrix_repair_profile_status"),
                "blocker_codes": blocker_codes,
                "remaining_blocker_codes_after_guardian_and_repair_exemption": remaining_after_support,
            }
            if tp_proof:
                repair_item["tp_proof"] = tp_proof
            forecast_support = _forecast_frontier_support_summary(metadata)
            if forecast_support:
                repair_item["forecast_support"] = forecast_support
            if (
                RANGE_FORECAST_ROTATION_BLOCKER in blocker_codes
                and str(context.get("method") or "").upper() != RANGE_ROTATION_METHOD
            ):
                counterpart = range_rotation_counterparts.get((intent.get("pair"), intent.get("side")))
                if counterpart:
                    repair_item["superseded_reason"] = (
                        "RANGE forecast requires RANGE_ROTATION; non-rotation repair candidate is not an executable repair frontier"
                    )
                    repair_item["superseded_by_range_rotation_lane_id"] = counterpart
                    repair_frontier_superseded_by_range_forecast.append(repair_item)
                else:
                    remaining_with_gap = [
                        code
                        for code in remaining_after_support
                        if code != RANGE_FORECAST_ROTATION_BLOCKER
                    ]
                    remaining_with_gap.append(RANGE_ROTATION_COUNTERPART_MISSING)
                    repair_item["missing_range_rotation_counterpart_for"] = [intent.get("pair"), intent.get("side")]
                    repair_item["missing_range_rotation_counterpart_reason"] = (
                        "RANGE forecast blocked a non-rotation repair lane, but order_intents has no same pair/side "
                        "RANGE_ROTATION counterpart to inspect. Treat this as candidate coverage debt, not as a "
                        "superseded executable frontier lane."
                    )
                    repair_item["remaining_blocker_codes_after_guardian_and_repair_exemption"] = remaining_with_gap
                    repair_frontier_missing_range_rotation_counterpart.append(repair_item)
                continue
            repair_frontier.append(repair_item)
            if item.get("status") == "LIVE_READY" and not blocker_codes:
                repair_live_ready.append(repair_item)
            elif GUARDIAN_BLOCKER in blocker_codes and not remaining_after_support:
                repair_basket_guardian_recovery.append(repair_item)
    repair_frontier.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    repair_live_ready.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    repair_basket_guardian_recovery.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    repair_frontier_superseded_by_range_forecast.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    repair_frontier_missing_range_rotation_counterpart.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    oanda_audit_only_local_tp_proof_required.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    global_unlock_frontier.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    month_scale_residual_blocked_intents.sort(key=lambda item: str(item.get("lane_id") or ""))
    near_ready_lanes.sort(key=_near_ready_lane_sort_key)
    remaining_repair_blockers = _repair_frontier_remaining_blockers(repair_frontier)
    active_path = _active_path_summary(
        active_trader_contract or {},
        active_opportunity_board or {},
        non_eurusd_live_grade_frontier or {},
    )
    shortest_live_ready_path = _shortest_live_ready_path(near_ready_lanes, active_path=active_path)
    return {
        "generated_at_utc": payload.get("generated_at_utc"),
        "lanes": len(results),
        "intent_pairs": intent_pairs,
        "live_ready_lanes": len(live_ready),
        "live_ready_lane_ids": [item.get("lane_id") for item in live_ready],
        "top_blockers": [{"code": code, "count": count} for code, count in codes.most_common(12)],
        "guardian_blocked_lanes": codes.get(GUARDIAN_BLOCKER, 0),
        "stale_support_blocker_codes": stale_support_blocker_codes,
        "stale_support_blocker_lanes": sum(
            1
            for item in near_ready_lanes
            if any(code in stale_support_blocker_codes for code in item.get("blocker_codes") or [])
        ),
        "repair_frontier": repair_frontier[:12],
        "repair_frontier_superseded_by_range_forecast": repair_frontier_superseded_by_range_forecast[:12],
        "repair_frontier_missing_range_rotation_counterpart": repair_frontier_missing_range_rotation_counterpart[:12],
        "oanda_audit_only_local_tp_proof_required": oanda_audit_only_local_tp_proof_required[:12],
        "repair_live_ready": repair_live_ready[:12],
        "repair_basket_guardian_recovery": repair_basket_guardian_recovery[:12],
        "repair_frontier_after_support_clear_lanes": sum(
            1 for item in repair_frontier if not item["remaining_blocker_codes_after_guardian_and_repair_exemption"]
        ),
        "repair_frontier_after_support_blocked_lanes": sum(
            1 for item in repair_frontier if item["remaining_blocker_codes_after_guardian_and_repair_exemption"]
        ),
        "repair_frontier_remaining_blockers": remaining_repair_blockers,
        "near_ready_lane_count": len(near_ready_lanes),
        "near_ready_lanes": near_ready_lanes[:10],
        "near_ready_blocker_groups": _near_ready_blocker_groups(near_ready_lanes),
        "shortest_live_ready_path": shortest_live_ready_path,
        "active_path": active_path,
        "global_unlock_frontier": global_unlock_frontier[:12],
        "month_scale_residual_blocked_intents": month_scale_residual_blocked_intents[:12],
        "month_scale_residual_blocked_intent_count": len(month_scale_residual_blocked_intents),
    }


def _intent_pairs(results: list[dict[str, Any]]) -> list[str]:
    pairs: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
        pair = intent.get("pair") or item.get("pair")
        pair_text = str(pair or "").strip().upper()
        if pair_text:
            pairs.add(pair_text)
    return sorted(pairs)


def _near_ready_lane_summary(
    item: dict[str, Any],
    blocker_codes: list[str],
    metadata: dict[str, Any],
    *,
    stale_support_blocker_codes: list[str] | None = None,
) -> dict[str, Any]:
    intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
    context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
    risk_metrics = item.get("risk_metrics") if isinstance(item.get("risk_metrics"), dict) else {}
    stale_support_blockers = set(stale_support_blocker_codes or [])
    groups = sorted(
        {
            _near_ready_blocker_group(
                code,
                stale_support_blocker_codes=stale_support_blockers,
            )
            for code in blocker_codes
        }
    )
    remaining_after_stale_clear = [
        code
        for code in blocker_codes
        if _near_ready_blocker_group(
            code,
            stale_support_blocker_codes=stale_support_blockers,
        )
        != "stale_artifact"
    ]
    remaining_after_global = [code for code in blocker_codes if code not in GLOBAL_UNLOCK_BLOCKERS]
    return {
        "lane_id": item.get("lane_id"),
        "status": item.get("status"),
        "pair": intent.get("pair"),
        "side": intent.get("side"),
        "method": context.get("method"),
        "order_type": intent.get("order_type"),
        "reward_jpy": _intent_reward_jpy(item, metadata),
        "risk_jpy": _intent_risk_jpy(item, metadata),
        "reward_risk": _round_optional(risk_metrics.get("reward_risk"), 6),
        "units": intent.get("units"),
        "blocker_codes": blocker_codes,
        "blocker_count": len(set(blocker_codes)),
        "blocker_groups": groups,
        "remaining_blocker_codes_after_stale_artifacts_clear": remaining_after_stale_clear,
        "remaining_blocker_codes_after_global_unlock": remaining_after_global,
        "evidence_needed": _near_ready_evidence_needed(blocker_codes),
        "remains_unsafe_after_stale_artifacts_clear": bool(remaining_after_stale_clear),
        "ordinary_fresh_entries_must_remain_blocked": True,
    }


def _near_ready_lane_sort_key(item: dict[str, Any]) -> tuple[int, int, float, str]:
    remaining = item.get("remaining_blocker_codes_after_global_unlock")
    remaining_count = len(remaining) if isinstance(remaining, list) else 0
    return (
        int(item.get("blocker_count") or 0),
        remaining_count,
        -_float(item.get("reward_jpy")),
        str(item.get("lane_id") or ""),
    )


def _active_path_summary(
    active_trader_contract: dict[str, Any],
    active_opportunity_board: dict[str, Any],
    non_eurusd_live_grade_frontier: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(active_trader_contract, dict):
        active_trader_contract = {}
    if not isinstance(active_opportunity_board, dict):
        active_opportunity_board = {}
    if not isinstance(non_eurusd_live_grade_frontier, dict):
        non_eurusd_live_grade_frontier = {}

    contract_state = (
        active_trader_contract.get("current_state")
        if isinstance(active_trader_contract.get("current_state"), dict)
        else {}
    )
    contract_board = (
        contract_state.get("active_opportunity_board")
        if isinstance(contract_state.get("active_opportunity_board"), dict)
        else {}
    )
    contract_frontier = (
        contract_state.get("non_eurusd_live_grade_frontier")
        if isinstance(contract_state.get("non_eurusd_live_grade_frontier"), dict)
        else {}
    )
    sources: list[tuple[str, dict[str, Any]]] = [
        (
            "active_trader_contract.current_state.active_opportunity_board.top_lane",
            _dict_or_empty(contract_board.get("top_lane")),
        ),
        ("active_opportunity_board.top_lane", _dict_or_empty(active_opportunity_board.get("top_lane"))),
        (
            "active_trader_contract.current_state.non_eurusd_live_grade_frontier.top_non_eurusd_lane",
            _dict_or_empty(contract_frontier.get("top_non_eurusd_lane")),
        ),
        (
            "non_eurusd_live_grade_frontier.top_non_eurusd_lane",
            _dict_or_empty(non_eurusd_live_grade_frontier.get("top_non_eurusd_lane")),
        ),
    ]
    for source, lane in sources:
        lane_id = str(lane.get("lane_id") or "").strip()
        if not lane_id:
            continue
        blockers = [str(code) for code in lane.get("blockers") or [] if str(code).strip()]
        return {
            "status": lane.get("status") or active_trader_contract.get("status"),
            "lane_id": lane_id,
            "pair": lane.get("pair"),
            "side": lane.get("direction") or lane.get("side"),
            "method": lane.get("strategy_family") or lane.get("method"),
            "order_type": lane.get("vehicle") or lane.get("order_type"),
            "blocker_codes": blockers,
            "expected_edge_jpy": lane.get("expected_edge_jpy"),
            "spread_status": lane.get("spread_status"),
            "forecast_status": lane.get("forecast_status"),
            "loss_budget_status": lane.get("loss_budget_status"),
            "next_action": active_trader_contract.get("next_trade_enabling_action")
            or active_trader_contract.get("next_prompt")
            or lane.get("next_action")
            or active_opportunity_board.get("next_active_path")
            or non_eurusd_live_grade_frontier.get("next_active_path"),
            "target_shape": active_trader_contract.get("target_shape"),
            "contract_status": active_trader_contract.get("status"),
            "board_status": active_opportunity_board.get("status") or contract_board.get("status"),
            "frontier_status": non_eurusd_live_grade_frontier.get("status") or contract_frontier.get("status"),
            "generated_at_utc": active_trader_contract.get("generated_at_utc")
            or active_opportunity_board.get("generated_at_utc")
            or non_eurusd_live_grade_frontier.get("generated_at_utc"),
            "live_permission": bool(lane.get("live_permission_allowed") is True)
            and bool(active_trader_contract.get("live_permission_allowed") is True),
            "ordinary_fresh_entries_must_remain_blocked": True,
            "source": source,
        }
    target_shape = str(active_trader_contract.get("target_shape") or "").strip()
    if target_shape:
        parts = target_shape.split("|")
        return {
            "status": active_trader_contract.get("status"),
            "lane_id": None,
            "pair": parts[0] if len(parts) > 0 else None,
            "side": parts[1] if len(parts) > 1 else None,
            "method": parts[2] if len(parts) > 2 else None,
            "order_type": parts[3] if len(parts) > 3 else None,
            "blocker_codes": [
                str(code)
                for code in active_trader_contract.get("remaining_blockers") or []
                if str(code).strip()
            ],
            "next_action": active_trader_contract.get("next_trade_enabling_action")
            or active_trader_contract.get("next_prompt"),
            "target_shape": target_shape,
            "contract_status": active_trader_contract.get("status"),
            "live_permission": bool(active_trader_contract.get("live_permission_allowed") is True),
            "ordinary_fresh_entries_must_remain_blocked": True,
            "source": "active_trader_contract.target_shape",
        }
    return {
        "status": "NO_ACTIVE_PATH_ARTIFACT",
        "lane_id": None,
        "pair": None,
        "side": None,
        "method": None,
        "order_type": None,
        "blocker_codes": [],
        "next_action": None,
        "live_permission": False,
        "ordinary_fresh_entries_must_remain_blocked": True,
        "source": None,
    }


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _shortest_live_ready_path(
    near_ready_lanes: list[dict[str, Any]],
    *,
    active_path: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_lane_id = str((active_path or {}).get("lane_id") or "").strip()
    active_shape_selected = active_lane_id or any(
        str((active_path or {}).get(key) or "").strip()
        for key in ("pair", "side", "method", "order_type")
    )
    if active_shape_selected:
        for lane in near_ready_lanes:
            if not _active_path_matches_lane(active_path or {}, lane):
                continue
            evidence_needed = [
                str(item)
                for item in lane.get("evidence_needed") or []
                if str(item).strip()
            ]
            blocker_codes = [
                str(item)
                for item in lane.get("blocker_codes") or []
                if str(item).strip()
            ]
            first_next_step = str((active_path or {}).get("next_action") or "").strip()
            if not first_next_step:
                first_next_step = (
                    evidence_needed[0]
                    if evidence_needed
                    else "clear the lane's named active-path blockers in refreshed broker, forecast, replay, and risk evidence"
                )
            return {
                "status": "ACTIVE_PATH_BLOCKED_NEAR_READY_LANE",
                "lane_id": lane.get("lane_id"),
                "pair": lane.get("pair"),
                "side": lane.get("side"),
                "method": lane.get("method"),
                "order_type": lane.get("order_type"),
                "current_status": lane.get("status"),
                "blocker_codes": blocker_codes,
                "blocker_groups": lane.get("blocker_groups") or [],
                "evidence_needed": evidence_needed,
                "first_next_step": first_next_step,
                "live_permission": False,
                "ordinary_fresh_entries_must_remain_blocked": True,
                "selection_basis": "active_trader_contract",
                "active_path": active_path,
            }
        blocker_codes = [
            str(item)
            for item in (active_path or {}).get("blocker_codes") or []
            if str(item).strip()
        ]
        next_action = str((active_path or {}).get("next_action") or "").strip()
        return {
            "status": "ACTIVE_PATH_SELECTED_BUT_NOT_IN_NEAR_READY_INTENTS",
            "lane_id": active_lane_id or None,
            "pair": (active_path or {}).get("pair"),
            "side": (active_path or {}).get("side"),
            "method": (active_path or {}).get("method"),
            "order_type": (active_path or {}).get("order_type"),
            "current_status": (active_path or {}).get("status"),
            "blocker_codes": blocker_codes,
            "blocker_groups": sorted(
                {_near_ready_blocker_group(code, stale_support_blocker_codes=set()) for code in blocker_codes}
            ),
            "evidence_needed": _near_ready_evidence_needed(blocker_codes),
            "first_next_step": next_action
            or "refresh order_intents/support after active_opportunity_board selects this lane",
            "live_permission": False,
            "ordinary_fresh_entries_must_remain_blocked": True,
            "selection_basis": "active_trader_contract",
            "active_path": active_path,
        }
    if not near_ready_lanes:
        return {
            "status": "NO_NEAR_READY_DIAGNOSTIC_LANES",
            "lane_id": None,
            "live_permission": False,
            "ordinary_fresh_entries_must_remain_blocked": True,
            "selection_basis": "no blocked intent rows were available to rank",
            "evidence_needed": [],
            "blocker_codes": [],
            "blocker_groups": [],
        }
    lane = near_ready_lanes[0]
    evidence_needed = [
        str(item)
        for item in lane.get("evidence_needed") or []
        if str(item).strip()
    ]
    blocker_codes = [
        str(item)
        for item in lane.get("blocker_codes") or []
        if str(item).strip()
    ]
    return {
        "status": "BLOCKED_NEAR_READY_LANE",
        "lane_id": lane.get("lane_id"),
        "pair": lane.get("pair"),
        "side": lane.get("side"),
        "method": lane.get("method"),
        "order_type": lane.get("order_type"),
        "current_status": lane.get("status"),
        "blocker_codes": blocker_codes,
        "blocker_groups": lane.get("blocker_groups") or [],
        "evidence_needed": evidence_needed,
        "first_next_step": (
            evidence_needed[0]
            if evidence_needed
            else "clear the lane's named blockers in refreshed broker, forecast, replay, and risk evidence"
        ),
        "remaining_blocker_codes_after_global_unlock": (
            lane.get("remaining_blocker_codes_after_global_unlock") or []
        ),
        "remaining_blocker_codes_after_stale_artifacts_clear": (
            lane.get("remaining_blocker_codes_after_stale_artifacts_clear") or []
        ),
        "remains_unsafe_after_stale_artifacts_clear": bool(
            lane.get("remains_unsafe_after_stale_artifacts_clear")
        ),
        "ordinary_fresh_entries_must_remain_blocked": True,
        "live_permission": False,
        "selection_basis": (
            "ranked by fewest blockers, then fewest blockers after global unlock, "
            "then highest reward, using existing order_intents diagnostics"
        ),
    }


def _active_path_matches_lane(active_path: dict[str, Any], lane: dict[str, Any]) -> bool:
    active_lane_id = str(active_path.get("lane_id") or "").strip()
    lane_id = str(lane.get("lane_id") or "").strip()
    if active_lane_id:
        return lane_id == active_lane_id

    comparisons = (
        ("pair", "pair"),
        ("side", "side"),
        ("method", "method"),
        ("order_type", "order_type"),
    )
    for active_key, lane_key in comparisons:
        active_value = str(active_path.get(active_key) or "").strip().upper()
        if not active_value:
            continue
        lane_value = str(lane.get(lane_key) or "").strip().upper()
        if lane_value != active_value:
            return False
    return True


def _near_ready_blocker_groups(near_ready_lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for item in near_ready_lanes:
        lane_id = str(item.get("lane_id") or "")
        for group in item.get("blocker_groups") or []:
            group_text = str(group)
            counts[group_text] += 1
            if lane_id and len(examples[group_text]) < 3:
                examples[group_text].append(lane_id)
    return [
        {
            "group": group,
            "count": count,
            "example_lane_ids": examples.get(group, []),
        }
        for group, count in counts.most_common()
    ]


def _stale_support_blocker_codes(
    results: list[dict[str, Any]],
    *,
    guardian: dict[str, Any],
    current_p0_findings: list[dict[str, Any]],
    guardian_receipt_consumption: dict[str, Any],
    guardian_receipt_operator_review: dict[str, Any],
) -> list[str]:
    """Support-only stale classifier for blockers embedded in older intents.

    This does not change LIVE_READY status or gateway validation. It only keeps
    the support panel from treating a cleared guardian P0, copied into an older
    `order_intents` packet, as current repair work.
    """

    blocker_counts = Counter(
        str(code)
        for item in results
        if isinstance(item, dict)
        for code in item.get("live_blocker_codes") or []
    )
    current_p0_codes = {
        str(item.get("code") or "")
        for item in current_p0_findings
        if isinstance(item, dict)
    }
    stale_codes: list[str] = []
    if (
        blocker_counts.get(GUARDIAN_BLOCKER, 0) > 0
        and guardian.get("required")
        and guardian.get("active")
        and GUARDIAN_BLOCKER not in current_p0_codes
    ):
        stale_codes.append(GUARDIAN_BLOCKER)
    if (
        blocker_counts.get(GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKER, 0) > 0
        and not _guardian_receipt_operator_review_blocks_normal_routing(
            guardian_receipt_consumption,
            guardian_receipt_operator_review,
        )
    ):
        stale_codes.append(GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKER)
    return stale_codes


def _near_ready_blocker_group(
    code: str,
    *,
    stale_support_blocker_codes: set[str] | None = None,
) -> str:
    text = str(code or "").upper()
    if text in (stale_support_blocker_codes or set()):
        return "stale_artifact"
    if text in GLOBAL_UNLOCK_BLOCKERS:
        return "global"
    if text in QUOTE_FRESHNESS_BLOCKER_CODES or "STALE" in text:
        return "stale_artifact"
    if text in MONTH_SCALE_RESIDUAL_BLOCKER_CODES or text.startswith("MONTH_SCALE_"):
        return "month_scale_replay"
    if text in FORECAST_FRONTIER_BLOCKER_CODES or "FORECAST" in text or "TELEMETRY" in text:
        return "forecast_telemetry"
    if "BIDASK" in text or text == OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED:
        return "bidask_replay"
    if text in FRONTIER_MARGIN_CAPACITY_BLOCKER_CODES or "MARGIN" in text or "MIN_LOT" in text:
        return "margin_or_min_lot"
    if "REWARD_RISK" in text or "SPREAD" in text or "TARGET_TOO_THIN" in text:
        return "geometry_or_spread"
    if text in FRONTIER_GUARDRAIL_BLOCKER_CODES:
        return "protective_guardrail"
    if "RANGE_" in text or "REWARD_RISK" in text or text.endswith("_RR_TOO_LOW"):
        return "geometry_or_spread"
    if text in {"CHART_DIRECTION_CONFLICT", "HARVEST_TP_STRUCTURE_MISSING"}:
        return "geometry_or_spread"
    if "MATRIX_REPAIR" in text:
        return "strategy_profile"
    if "OPERATOR_MANUAL" in text or "MANUAL" in text:
        return "manual_overlap"
    if "STRATEGY" in text or "PROFILE" in text or "BLOCK_UNTIL_NEW_EVIDENCE" in text:
        return "strategy_profile"
    if "NEGATIVE_EXPECTANCY" in text or "TP_PROVEN" in text:
        return "profitability"
    return "other"


def _near_ready_evidence_needed(blocker_codes: list[str]) -> list[str]:
    groups = {_near_ready_blocker_group(code) for code in blocker_codes}
    needs: list[str] = []
    if "stale_artifact" in groups:
        needs.append("fresh broker snapshot, forecasts, projections, and order_intents from the same evidence packet")
    if "global" in groups:
        needs.append("clear global support/profitability P0s before ordinary fresh entries")
    if "forecast_telemetry" in groups:
        needs.append("fresh executable forecast telemetry with projection-ledger scoring for the lane pair/side")
    if "bidask_replay" in groups:
        needs.append(
            "spread-included bid/ask replay evidence that is non-negative for the exact pair/side/method"
        )
    if "month_scale_replay" in groups:
        needs.append(
            "744h execution-timing replay where the matching residual pair/side/method is non-negative or absent"
        )
    if "protective_guardrail" in groups or "geometry_or_spread" in groups:
        needs.append(
            "current rail/entry geometry proving the lane is not a chase and has acceptable reward/risk after spread"
        )
    if "margin_or_min_lot" in groups:
        needs.append("raw-NAV/margin capacity sufficient for the broker 1000-unit production floor and risk budget")
    if "strategy_profile" in groups:
        needs.append("new strategy-profile evidence for the same pair/side/method instead of BLOCK_UNTIL_NEW_EVIDENCE")
    if "manual_overlap" in groups:
        needs.append("explicit operator approval or changed broker truth before overlapping manual/operator exposure")
    if "profitability" in groups:
        needs.append(
            "TP-proven positive payoff evidence or non-negative system expectancy before ordinary fresh turnover"
        )
    return list(dict.fromkeys(needs))


def _tp_proof_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "positive_rotation_mode",
        "positive_rotation_pessimistic_expectancy_jpy",
        "capture_take_profit_scope",
        "capture_take_profit_scope_key",
        "capture_take_profit_trades",
        "capture_take_profit_wins",
        "capture_take_profit_losses",
        "capture_take_profit_expectancy_jpy",
        "capture_take_profit_avg_win_jpy",
        "capture_take_profit_avg_loss_jpy",
    )
    summary = {key: metadata.get(key) for key in keys if metadata.get(key) is not None}
    if not summary:
        return {}
    return summary


def _oanda_replay_evidence_by_vehicle(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    firepower = payload.get("campaign_firepower") if isinstance(payload.get("campaign_firepower"), dict) else {}
    sections: list[tuple[int, str, list[Any]]] = []
    high_precision = firepower.get("high_precision") if isinstance(firepower.get("high_precision"), dict) else {}
    evidence_queue = firepower.get("evidence_queue") if isinstance(firepower.get("evidence_queue"), dict) else {}
    sections.append((0, "high_precision", high_precision.get("top_vehicles") or []))
    sections.append((1, "evidence_queue", evidence_queue.get("top_vehicles") or []))
    sections.append((2, "live_grade_evidence_queue", payload.get("live_grade_evidence_queue") or []))
    out: dict[str, dict[str, Any]] = {}
    seen_priority: dict[str, int] = {}
    for priority, section, rows in sections:
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            vehicle_key = row.get("vehicle_key")
            if not vehicle_key:
                continue
            key = str(vehicle_key)
            if key in out and seen_priority.get(key, 99) <= priority:
                continue
            compact = dict(row)
            compact["firepower_section"] = row.get("firepower_section") or section
            out[key] = compact
            seen_priority[key] = priority
    return out


def _compact_oanda_replay_evidence(
    vehicle_key: Any,
    evidence_by_vehicle: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = str(vehicle_key or "")
    row = evidence_by_vehicle.get(key)
    if not row:
        return {
            "vehicle_key": key or None,
            "evidence_status": "MISSING_OANDA_REPLAY_EVIDENCE",
            "live_permission": False,
            "live_permission_reason": "No matching OANDA candle replay vehicle was found for this intent key.",
        }
    return {
        "vehicle_key": key,
        "firepower_section": row.get("firepower_section"),
        "evidence_status": row.get("evidence_status") or "OANDA_REPLAY_EVIDENCE",
        "validation_n": row.get("validation_n"),
        "validation_win_rate": _round_optional(row.get("validation_win_rate"), 6),
        "validation_win_wilson95_lower": _round_optional(row.get("validation_win_wilson95_lower"), 6),
        "validation_profit_factor": _round_optional(row.get("validation_profit_factor"), 6),
        "validation_avg_realized_pips": _round_optional(row.get("validation_avg_realized_pips"), 6),
        "validation_expectancy_r": _round_optional(row.get("validation_expectancy_r"), 6),
        "active_days": row.get("active_days"),
        "positive_day_rate": _round_optional(row.get("positive_day_rate"), 6),
        "estimated_return_pct_per_active_day_at_observed_frequency": _round_optional(
            row.get("estimated_return_pct_per_active_day_at_observed_frequency"),
            6,
        ),
        "trades_needed_for_minimum_5pct": row.get("trades_needed_for_minimum_5pct"),
        "trades_needed_for_target_10pct": row.get("trades_needed_for_target_10pct"),
        "live_permission": False,
        "live_permission_reason": (
            "OANDA candle replay is audit-only until local broker TP proof, current forecast, "
            "spread, risk, strategy-profile, and gateway checks all pass."
        ),
    }


def _range_rotation_counterparts(results: list[dict[str, Any]]) -> dict[tuple[Any, Any], Any]:
    counterparts: dict[tuple[Any, Any], Any] = {}
    for item in results:
        intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
        context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
        if str(context.get("method") or "").upper() != RANGE_ROTATION_METHOD:
            continue
        key = (intent.get("pair"), intent.get("side"))
        if key not in counterparts:
            counterparts[key] = item.get("lane_id")
    return counterparts


def _forecast_frontier_support_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    support = (
        metadata.get("forecast_market_support")
        if isinstance(metadata.get("forecast_market_support"), dict)
        else {}
    )
    unselected = (
        support.get("unselected_signals")
        if isinstance(support.get("unselected_signals"), list)
        else []
    )
    top_signal = next((item for item in unselected if isinstance(item, dict)), {})
    summary: dict[str, Any] = {}
    for key in (
        "forecast_direction",
        "forecast_confidence",
        "forecast_horizon_min",
        "forecast_market_support_ok",
        "forecast_market_support_reason",
    ):
        if metadata.get(key) is not None:
            summary[key] = metadata.get(key)
    if support:
        if "forecast_market_support_ok" not in summary and support.get("ok") is not None:
            summary["forecast_market_support_ok"] = support.get("ok")
        if "forecast_market_support_reason" not in summary and support.get("reason"):
            summary["forecast_market_support_reason"] = support.get("reason")
        for key in (
            "unselected_projection_count",
            "best_unselected_hit_rate",
            "best_unselected_samples",
        ):
            if support.get(key) is not None:
                summary[key] = support.get(key)
    if top_signal:
        signal_summaries = []
        for signal in unselected[:3]:
            if not isinstance(signal, dict):
                continue
            signal_summary: dict[str, Any] = {}
            for key in (
                "name",
                "direction",
                "live_precision_ok",
                "lead_time_min",
                "hit_rate",
                "economic_hit_rate",
                "samples",
                "calibration_samples",
                "economic_samples",
                "confidence",
            ):
                if signal.get(key) is not None:
                    signal_summary[key] = signal.get(key)
            if signal_summary:
                signal_summaries.append(signal_summary)
        if signal_summaries:
            summary["unselected_signal_examples"] = signal_summaries
        signal_summary: dict[str, Any] = {}
        for key in (
            "name",
            "direction",
            "live_precision_ok",
            "lead_time_min",
            "hit_rate",
            "economic_hit_rate",
            "samples",
            "calibration_samples",
            "economic_samples",
            "confidence",
        ):
            if top_signal.get(key) is not None:
                signal_summary[key] = top_signal.get(key)
        if signal_summary:
            summary["top_unselected_signal"] = signal_summary
    return summary


def _repair_frontier_remaining_blockers(repair_frontier: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    reward_by_code: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    co_blockers_by_code: dict[str, Counter[str]] = defaultdict(Counter)
    forecast_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matrix_status_by_code: dict[str, Counter[str]] = defaultdict(Counter)
    for item in repair_frontier:
        lane_id = str(item.get("lane_id") or "")
        reward = _float(item.get("reward_jpy"))
        matrix_status = str(item.get("matrix_repair_profile_status") or "").strip()
        remaining = [
            str(code)
            for code in item.get("remaining_blocker_codes_after_guardian_and_repair_exemption") or []
            if str(code).strip()
        ]
        for code in sorted(set(remaining)):
            counts[code] += 1
            reward_by_code[code] += reward
            for other in sorted(set(remaining) - {code}):
                co_blockers_by_code[code][other] += 1
            if lane_id and len(examples[code]) < 3:
                examples[code].append(lane_id)
            if matrix_status:
                matrix_status_by_code[code][matrix_status] += 1
            if (
                code in FORECAST_FRONTIER_BLOCKER_CODES
                and isinstance(item.get("forecast_support"), dict)
                and len(forecast_examples[code]) < 3
            ):
                forecast_examples[code].append(
                    {
                        "lane_id": lane_id,
                        "pair": item.get("pair"),
                        "side": item.get("side"),
                        "method": item.get("method"),
                        "order_type": item.get("order_type"),
                        "forecast_support": item.get("forecast_support"),
                    }
                )
    rows = []
    for code, count in counts.most_common(12):
        row = {
            "code": code,
            "count": count,
            "reward_jpy": _round_optional(reward_by_code[code], 3),
            "example_lane_ids": examples[code],
        }
        if forecast_examples.get(code):
            row["forecast_support_examples"] = forecast_examples[code]
        if matrix_status_by_code.get(code):
            row["matrix_repair_profile_status_counts"] = dict(matrix_status_by_code[code])
        co_blockers = [
            other
            for other, _ in sorted(
                co_blockers_by_code[code].items(),
                key=lambda pair: (-pair[1], pair[0]),
            )[:6]
        ]
        if co_blockers:
            row["co_blocker_codes"] = co_blockers
        rows.append(row)
    rows.sort(key=_frontier_blocker_sort_key)
    return rows


def _frontier_blocker_sort_key(row: dict[str, Any]) -> tuple[int, int, float, str]:
    code = str(row.get("code") or "")
    if _frontier_blocker_waits_for_quote_refresh(row):
        causal_rank = 0
    elif _frontier_blocker_waits_for_live_precision_evidence(row):
        causal_rank = 0
    elif _frontier_blocker_waits_for_strategy_profile_evidence(row):
        causal_rank = 0
    elif _frontier_blocker_waits_for_margin_capacity(row):
        causal_rank = 1
    elif code == OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED:
        causal_rank = 1
    elif code in FRONTIER_GUARDRAIL_BLOCKER_CODES:
        causal_rank = 4
    else:
        causal_rank = 2
    return (
        causal_rank,
        -int(row.get("count") or 0),
        -_float(row.get("reward_jpy")),
        code,
    )


def _frontier_blocker_is_protective_guardrail(row: dict[str, Any]) -> bool:
    return str(row.get("code") or "") in FRONTIER_GUARDRAIL_BLOCKER_CODES


def _frontier_blocker_waits_for_strategy_profile_evidence(row: dict[str, Any]) -> bool:
    if str(row.get("code") or "") != "MATRIX_REPAIR_REJECT_CONTEXT":
        return False
    status_counts = (
        row.get("matrix_repair_profile_status_counts")
        if isinstance(row.get("matrix_repair_profile_status_counts"), dict)
        else {}
    )
    status = str(row.get("matrix_repair_profile_status") or "")
    return "BLOCK_UNTIL_NEW_EVIDENCE" in status_counts or status == "BLOCK_UNTIL_NEW_EVIDENCE"


def _frontier_blocker_waits_for_quote_refresh(row: dict[str, Any]) -> bool:
    return str(row.get("code") or "") in QUOTE_FRESHNESS_BLOCKER_CODES


def _frontier_blocker_waits_for_margin_capacity(row: dict[str, Any]) -> bool:
    return str(row.get("code") or "") in FRONTIER_MARGIN_CAPACITY_BLOCKER_CODES


def _intent_metadata(item: dict[str, Any]) -> dict[str, Any]:
    intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    return metadata


def _intent_reward_jpy(item: dict[str, Any], metadata: dict[str, Any]) -> float | None:
    risk_metrics = item.get("risk_metrics") if isinstance(item.get("risk_metrics"), dict) else {}
    return _round_optional(
        metadata.get("sizing_actual_reward_jpy")
        if metadata.get("sizing_actual_reward_jpy") is not None
        else risk_metrics.get("reward_jpy"),
        3,
    )


def _intent_risk_jpy(item: dict[str, Any], metadata: dict[str, Any]) -> float | None:
    risk_metrics = item.get("risk_metrics") if isinstance(item.get("risk_metrics"), dict) else {}
    return _round_optional(
        metadata.get("sizing_actual_risk_jpy")
        if metadata.get("sizing_actual_risk_jpy") is not None
        else risk_metrics.get("risk_jpy"),
        3,
    )


def _position_support_summary(
    position_management: dict[str, Any],
    guardian_management: dict[str, Any],
    *,
    now_utc: datetime,
) -> dict[str, Any]:
    current_positions = (
        position_management.get("positions") if isinstance(position_management.get("positions"), list) else []
    )
    guardian_positions = (
        guardian_management.get("positions") if isinstance(guardian_management.get("positions"), list) else []
    )
    generated = guardian_management.get("generated_at_utc")
    return {
        "position_management_action": position_management.get("action"),
        "position_management_generated_at_utc": position_management.get("generated_at_utc"),
        "managed_positions": _compact_position_actions(current_positions),
        "guardian_management_generated_at_utc": generated,
        "guardian_management_age_seconds": _round_optional(_age_seconds(generated, now_utc=now_utc), 3),
        "guardian_managed_positions": _compact_position_actions(guardian_positions),
    }


def _compact_position_actions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for item in rows[:12]:
        compact.append(
            {
                "trade_id": item.get("trade_id"),
                "pair": item.get("pair"),
                "side": item.get("side"),
                "action": item.get("action"),
                "close_review_action": item.get("close_review_action"),
                "unrealized_pl_jpy": _round_optional(item.get("unrealized_pl_jpy"), 3),
                "remaining_reward_jpy": _round_optional(item.get("remaining_reward_jpy"), 3),
                "remaining_risk_jpy": _round_optional(item.get("remaining_risk_jpy"), 3),
            }
        )
    return compact


def _acceptance_summary(
    payload: dict[str, Any],
    *,
    bidask_current_intent_history_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    capture = metrics.get("capture_economics") if isinstance(metrics.get("capture_economics"), dict) else {}
    firepower = metrics.get("oanda_campaign_firepower") if isinstance(metrics.get("oanda_campaign_firepower"), dict) else {}
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    return {
        "status": payload.get("status"),
        "blockers": blockers[:8],
        "capture_economics": capture,
        "target_firepower": _target_firepower_summary(firepower),
        "repair_plan": _acceptance_repair_plan(
            payload,
            bidask_current_intent_history_coverage=bidask_current_intent_history_coverage,
        ),
    }


def _acceptance_repair_plan(
    payload: dict[str, Any],
    *,
    bidask_current_intent_history_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    p0_findings = [
        item
        for item in findings
        if isinstance(item, dict) and str(item.get("priority") or "").upper() == "P0"
    ]
    evidence_findings = [
        item
        for item in findings
        if isinstance(item, dict)
        and str(item.get("code") or "") in {
            "BIDASK_REPLAY_PRICE_TRUTH_PARTIAL",
            "BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN",
            "BIDASK_REPLAY_SUPPORT_NOT_DAILY_STABLE",
            "BIDASK_CONTRARIAN_EDGE_NOT_DAILY_STABLE",
        }
    ]
    fallback_blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    items: list[dict[str, Any]] = []
    if p0_findings:
        for finding in p0_findings[:12]:
            items.append(
                _acceptance_repair_item(
                    finding,
                    metrics,
                    bidask_current_intent_history_coverage=bidask_current_intent_history_coverage,
                )
            )
    elif fallback_blockers:
        for blocker in fallback_blockers[:12]:
            code, message = _split_blocker(str(blocker))
            items.append(
                _acceptance_repair_item(
                    {
                        "priority": "P0",
                        "code": code,
                        "message": message,
                        "next_action": "Inspect profitability_acceptance findings for the concrete red invariant.",
                        "evidence": {},
                    },
                    metrics,
                    bidask_current_intent_history_coverage=bidask_current_intent_history_coverage,
                )
            )
    evidence_items = [
        _acceptance_repair_item(
            finding,
            metrics,
            bidask_current_intent_history_coverage=bidask_current_intent_history_coverage,
        )
        for finding in evidence_findings[:8]
    ]
    commands: list[str] = []
    seen = set()
    for item in [*items, *evidence_items]:
        command = item.get("verification_command")
        if not command or command in seen:
            continue
        seen.add(command)
        commands.append(str(command))
    return {
        "status": payload.get("status"),
        "p0_count": len(p0_findings) if p0_findings else len(fallback_blockers),
        "items": items,
        "evidence_collection_items": evidence_items,
        "next_verification_commands": commands[:8],
        "loop_breaker": (
            "Rerunning profitability-acceptance alone cannot clear these P0s; the listed proof condition "
            "must change first. Evidence-collection items also require fresh replay/mining output before "
            "they can graduate into live-grade turnover."
            if items
            else "No acceptance P0 repair item is active; work evidence-collection items before claiming "
            "new high-turn firepower."
            if evidence_items
            else "No acceptance P0 repair item is active."
        ),
    }


def _acceptance_repair_item(
    finding: dict[str, Any],
    metrics: dict[str, Any],
    *,
    bidask_current_intent_history_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    code = str(finding.get("code") or "UNKNOWN_ACCEPTANCE_BLOCKER")
    evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
    condition, command, summary = _acceptance_clearance_for_code(
        code,
        evidence,
        metrics,
        bidask_current_intent_history_coverage=bidask_current_intent_history_coverage,
    )
    return {
        "code": code,
        "priority": str(finding.get("priority") or "P0"),
        "message": finding.get("message"),
        "next_action": finding.get("next_action"),
        "clearance_condition": condition,
        "verification_command": command,
        "evidence_summary": summary,
    }


def _bidask_price_truth_fetch_required(bidask: dict[str, Any]) -> bool:
    price_truth = (
        bidask.get("price_truth_coverage")
        if isinstance(bidask.get("price_truth_coverage"), dict)
        else {}
    )
    status = str(price_truth.get("status") or "").upper()
    missing_samples = _float(price_truth.get("missing_price_truth_samples"))
    missing_windows = _float(price_truth.get("missing_price_window_group_count"))
    fetch_count = _float(price_truth.get("history_fetch_command_count"))
    if status == "PRICE_TRUTH_OK" and missing_samples <= 0 and missing_windows <= 0 and fetch_count <= 0:
        return False
    if status in {"PARTIAL_PRICE_TRUTH", "PRICE_TRUTH_PARTIAL", "MISSING_PRICE_TRUTH"}:
        return True
    if missing_samples > 0 or missing_windows > 0 or fetch_count > 0:
        return True
    return bool(bidask.get("history_fetch_command")) and status != "PRICE_TRUTH_OK"


def _bidask_summary_requires_price_truth_fetch(summary: dict[str, Any]) -> bool:
    if "price_truth_fetch_required" in summary:
        return bool(summary.get("price_truth_fetch_required"))
    return _bidask_price_truth_fetch_required(summary)


def _bidask_sample_coverage_detail_status(sample_coverage: dict[str, Any]) -> str:
    if not sample_coverage:
        return BIDASK_COVERAGE_DETAIL_MISSING
    for key in ("under_sampled_pair_direction_examples", "pair_coverage_examples"):
        examples = sample_coverage.get(key)
        if isinstance(examples, list) and examples:
            return BIDASK_COVERAGE_DETAIL_PRESENT
    gap_reasons = sample_coverage.get("under_sampled_gap_reason_counts")
    if isinstance(gap_reasons, dict) and gap_reasons:
        return BIDASK_COVERAGE_DETAIL_PRESENT
    under_sampled = sample_coverage.get("under_sampled_pair_directions")
    if isinstance(under_sampled, list) and under_sampled:
        return BIDASK_COVERAGE_DETAIL_PRESENT
    if _float(under_sampled) > 0 or _float(sample_coverage.get("under_sampled_pair_direction_count")) > 0:
        return BIDASK_COVERAGE_DETAIL_SUMMARY_ONLY
    return BIDASK_COVERAGE_DETAIL_NOT_REQUIRED


def _acceptance_clearance_for_code(
    code: str,
    evidence: dict[str, Any],
    metrics: dict[str, Any],
    *,
    bidask_current_intent_history_coverage: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    if code == "SELF_IMPROVEMENT_P0_PRESENT":
        p0_findings = evidence.get("p0_findings") if isinstance(evidence.get("p0_findings"), list) else []
        p0_codes = [
            str(item.get("code"))
            for item in p0_findings
            if isinstance(item, dict) and item.get("code")
        ]
        return (
            "self_improvement_audit has zero P0 findings, or the only remaining discipline finding has "
            "been demoted by verified clean gateway close recovery",
            "PYTHONPATH=src python3 -m quant_rabbit.cli self-improvement-audit",
            {"current_p0_codes": p0_codes[:8], "current_p0_count": len(p0_codes) or None},
        )
    if code == "NEGATIVE_EXPECTANCY_ACTIVE":
        capture = metrics.get("capture_economics") if isinstance(metrics.get("capture_economics"), dict) else {}
        capture = capture or evidence
        overall = capture.get("overall") if isinstance(capture.get("overall"), dict) else capture
        return (
            "capture_economics.status is no longer NEGATIVE_EXPECTANCY, or entries are limited to exact "
            "TP-proven repair/harvest shapes with positive expectancy evidence",
            "PYTHONPATH=src python3 -m quant_rabbit.cli capture-economics",
            {
                "status": capture.get("status"),
                "expectancy_jpy_per_trade": _round_optional(
                    overall.get("expectancy_jpy_per_trade"),
                    3,
                ),
                "net_jpy": _round_optional(overall.get("net_jpy"), 3),
                "trades": overall.get("trades"),
                "profit_factor": _round_optional(overall.get("profit_factor"), 3),
            },
        )
    if code == "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE":
        segments = evidence.get("segments") if isinstance(evidence.get("segments"), list) else []
        return (
            "no TP-proven segment remains net-damaged by MARKET_ORDER_TRADE_CLOSE leakage; preserve broker TP "
            "and guardian capture instead of scaling market-close loss paths",
            "PYTHONPATH=src python3 -m quant_rabbit.cli capture-economics",
            {"damaged_segments": len(segments), "top_segments": segments[:3]},
        )
    if code == "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK":
        latest = evidence.get("latest_loss_close_ts_utc") or _latest_ts_from_examples(evidence)
        return (
            "recent_leak_loss_closes is zero inside the 7-day acceptance window, or each loss-side market "
            "close has contained-risk timing plus durable gateway/GPT close proof",
            MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND,
            {
                "recent_leak_loss_closes": evidence.get("recent_leak_loss_closes"),
                "recent_leak_loss_net_jpy": _round_optional(evidence.get("recent_leak_loss_net_jpy"), 3),
                "latest_loss_close_ts_utc": latest,
                "earliest_auto_clear_if_no_new_leak_utc": _plus_days_iso(
                    latest,
                    ACCEPTANCE_LEAK_LOOKBACK_DAYS,
                ),
                "timing_labels": evidence.get("recent_loss_timing_label_counts"),
            },
        )
    if code == "LOSS_CLOSE_GATE_EVIDENCE_MISSING":
        latest = _latest_ts_from_examples(evidence)
        return (
            "every recent GPT loss-side market close has durable close_gate_evidence in verification_observations, "
            "or the missing-evidence closes age out of the 7-day acceptance window without new leaks",
            "PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit",
            {
                "missing_close_gate_evidence": evidence.get(
                    "recent_close_gate_missing_loss_closes",
                    evidence.get("recent_close_gate_unverified_loss_closes"),
                ),
                "missing_close_gate_net_jpy": _round_optional(
                    evidence.get(
                        "recent_close_gate_missing_loss_net_jpy",
                        evidence.get("recent_close_gate_unverified_loss_net_jpy"),
                    ),
                    3,
                ),
                "not_passing_close_gate_evidence": evidence.get(
                    "recent_close_gate_not_passing_loss_closes"
                ),
                "missing_receipt_evidence_present": evidence.get(
                    "recent_close_gate_missing_receipt_evidence_present_loss_closes"
                ),
                "missing_receipt_evidence_present_net_jpy": _round_optional(
                    evidence.get(
                        "recent_close_gate_missing_receipt_evidence_present_loss_net_jpy"
                    ),
                    3,
                ),
                "missing_receipt_evidence_absent": evidence.get(
                    "recent_close_gate_missing_receipt_evidence_absent_loss_closes"
                ),
                "missing_receipt_evidence_absent_net_jpy": _round_optional(
                    evidence.get(
                        "recent_close_gate_missing_receipt_evidence_absent_loss_net_jpy"
                    ),
                    3,
                ),
                "latest_missing_evidence_ts_utc": latest,
                "earliest_auto_clear_if_no_new_missing_utc": _plus_days_iso(
                    latest,
                    ACCEPTANCE_LEAK_LOOKBACK_DAYS,
                ),
                "example_trade_ids": _example_trade_ids(evidence),
            },
        )
    if code == "LOSS_CLOSE_GATE_EVIDENCE_NOT_PASSING":
        latest = _latest_ts_from_examples(evidence)
        return (
            "every recent GPT loss-side market close has PASS close_gate_evidence in verification_observations, "
            "or the failed-evidence closes age out of the 7-day acceptance window without new leaks",
            "PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit",
            {
                "not_passing_close_gate_evidence": evidence.get(
                    "recent_close_gate_not_passing_loss_closes",
                    evidence.get("recent_close_gate_unverified_loss_closes"),
                ),
                "not_passing_close_gate_net_jpy": _round_optional(
                    evidence.get(
                        "recent_close_gate_not_passing_loss_net_jpy",
                        evidence.get("recent_close_gate_unverified_loss_net_jpy"),
                    ),
                    3,
                ),
                "missing_close_gate_evidence": evidence.get(
                    "recent_close_gate_missing_loss_closes"
                ),
                "latest_not_passing_evidence_ts_utc": latest,
                "earliest_auto_clear_if_no_new_not_passing_utc": _plus_days_iso(
                    latest,
                    ACCEPTANCE_LEAK_LOOKBACK_DAYS,
                ),
                "example_trade_ids": _example_trade_ids(evidence),
            },
        )
    if code == "TP_PROGRESS_REPLAY_REPAIR_UNPROVED":
        return (
            "execution-timing-audit reports zero post-repair live-evidence "
            "loss_closes_repair_replay_triggered under the current production-gate replay "
            "contract after the TP-progress TAKE_PROFIT_MARKET path and position guardian "
            "have had a live window to capture executable plus P/L before red closes",
            MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND,
            {
                "loss_closes_profit_capture_missed": evidence.get(
                    "loss_closes_profit_capture_missed"
                ),
                "loss_closes_repair_replay_triggered": evidence.get(
                    "loss_closes_repair_replay_triggered"
                ),
                "pre_repair_historical_loss_closes_repair_replay_triggered": evidence.get(
                    "pre_repair_historical_loss_closes_repair_replay_triggered"
                ),
                "post_repair_live_evidence_loss_closes_audited": evidence.get(
                    "post_repair_live_evidence_loss_closes_audited"
                ),
                "post_repair_live_evidence_loss_closes_repair_replay_triggered": evidence.get(
                    "post_repair_live_evidence_loss_closes_repair_replay_triggered"
                ),
                "counterfactual_profit_capture_delta_jpy": _round_optional(
                    evidence.get("counterfactual_profit_capture_delta_jpy"),
                    3,
                ),
                "counterfactual_profit_capture_jpy": _round_optional(
                    evidence.get("counterfactual_profit_capture_jpy"),
                    3,
                ),
                "example_trade_ids": _example_trade_ids_from_top_misses(evidence),
                "clearance_condition": evidence.get("clearance_condition"),
            },
        )
    if code == "MONTH_SCALE_LOSS_CLOSE_REPLAY_REQUIRED":
        return (
            "the active execution-timing-audit covers at least 720 hours with the current "
            "TP-progress production-gate replay contract before TP-positive / market-close-negative "
            "economics are treated as repaired",
            MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND,
            {
                "take_profit_net_jpy": _round_optional(evidence.get("take_profit_net_jpy"), 3),
                "market_close_net_jpy": _round_optional(evidence.get("market_close_net_jpy"), 3),
                "window_lookback_hours": _round_optional(evidence.get("window_lookback_hours"), 3),
                "required_lookback_hours": _round_optional(
                    evidence.get("required_lookback_hours"),
                    3,
                ),
                "repair_replay_contract": evidence.get("repair_replay_contract"),
                "repair_replay_contract_present": evidence.get("repair_replay_contract_present"),
            },
        )
    if code == "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE":
        residual_groups = (
            evidence.get("top_repair_replay_residual_groups")
            if isinstance(evidence.get("top_repair_replay_residual_groups"), list)
            else []
        )
        residual_method_rollups = (
            evidence.get("top_repair_replay_residual_method_rollups")
            if isinstance(evidence.get("top_repair_replay_residual_method_rollups"), list)
            else []
        )
        tp_progress_method_rollups = (
            evidence.get("top_tp_progress_repair_residual_method_rollups")
            if isinstance(evidence.get("top_tp_progress_repair_residual_method_rollups"), list)
            else []
        )
        entry_quality_method_rollups = (
            evidence.get("top_entry_quality_residual_method_rollups")
            if isinstance(evidence.get("top_entry_quality_residual_method_rollups"), list)
            else []
        )
        return (
            "month-scale production-gate replay is non-negative, or the top residual "
            "pair/side/method groups are removed by close-gate, TP-capture, or entry-selection "
            "changes before turnover is scaled",
            MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND,
            {
                "window_lookback_hours": _round_optional(evidence.get("window_lookback_hours"), 3),
                "loss_closes_profit_capture_missed": evidence.get(
                    "loss_closes_profit_capture_missed"
                ),
                "loss_closes_repair_replay_triggered": evidence.get(
                    "loss_closes_repair_replay_triggered"
                ),
                "repair_replay_counterfactual_pl_jpy": _round_optional(
                    evidence.get("repair_replay_counterfactual_pl_jpy"),
                    3,
                ),
                "active_counterfactual_profit_capture_pl_jpy": _round_optional(
                    evidence.get("active_counterfactual_profit_capture_pl_jpy"),
                    3,
                ),
                "counterfactual_profit_capture_delta_jpy": _round_optional(
                    evidence.get("counterfactual_profit_capture_delta_jpy"),
                    3,
                ),
                "top_repair_replay_residual_groups": residual_groups[:3],
                "top_repair_replay_residual_method_rollups": residual_method_rollups[:3],
                "top_tp_progress_repair_residual_method_rollups": tp_progress_method_rollups[:3],
                "top_entry_quality_residual_method_rollups": entry_quality_method_rollups[:3],
            },
        )
    if code == "TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED":
        return (
            "position guardian is proven active with a fresh heartbeat, then the TP-progress "
            "production-gate replay is rerun after a live window and reports zero "
            "loss_closes_repair_replay_triggered",
            "scripts/install-position-guardian.sh --status",
            {
                "guardian_profit_capture_inactive": evidence.get(
                    "guardian_profit_capture_inactive"
                ),
                "loss_closes_profit_capture_missed": evidence.get(
                    "loss_closes_profit_capture_missed"
                ),
                "loss_closes_repair_replay_triggered": evidence.get(
                    "loss_closes_repair_replay_triggered"
                ),
                "repair_replay_contract": evidence.get("repair_replay_contract"),
                "example_trade_ids": _example_trade_ids_from_top_misses(evidence),
                "clearance_condition": evidence.get("clearance_condition"),
            },
        )
    if code == "TP_PROGRESS_REPAIR_REPLAY_CONTRACT_MISSING":
        return (
            "execution-timing-audit is regenerated by the current runtime and includes "
            f"{TP_PROGRESS_REPAIR_REPLAY_CONTRACT} before TP-progress repair is judged clean",
            MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND,
            {
                "loss_closes_profit_capture_missed": evidence.get(
                    "loss_closes_profit_capture_missed"
                ),
                "repair_replay_contract": evidence.get("repair_replay_contract"),
                "required_repair_replay_contract": evidence.get(
                    "required_repair_replay_contract",
                    TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                ),
                "example_trade_ids": _example_trade_ids_from_top_misses(evidence),
                "clearance_condition": evidence.get("clearance_condition"),
            },
        )
    if code in {
        "BIDASK_REPLAY_PRICE_TRUTH_PARTIAL",
        "BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN",
        "BIDASK_REPLAY_SUPPORT_NOT_DAILY_STABLE",
        "BIDASK_CONTRARIAN_EDGE_NOT_DAILY_STABLE",
    }:
        bidask = metrics.get("bidask_replay_rules") if isinstance(metrics.get("bidask_replay_rules"), dict) else {}
        bidask = bidask or evidence
        examples = bidask.get("rank_only_examples") if isinstance(bidask.get("rank_only_examples"), list) else []
        price_truth = (
            bidask.get("price_truth_coverage")
            if isinstance(bidask.get("price_truth_coverage"), dict)
            else {}
        )
        sample_coverage = (
            bidask.get("forecast_sample_coverage_summary")
            if isinstance(bidask.get("forecast_sample_coverage_summary"), dict)
            else {}
        )
        coverage_detail_status = _bidask_sample_coverage_detail_status(sample_coverage)
        coverage_detail_repackage_required = (
            coverage_detail_status == BIDASK_COVERAGE_DETAIL_SUMMARY_ONLY
        )
        price_truth_fetch_required = _bidask_price_truth_fetch_required(bidask)
        validation_command = bidask.get("replay_validation_command") or (
            "python3 scripts/oanda_history_replay_validate.py "
            "--forecast-history data/forecast_history.jsonl "
            "--granularity S5"
        )
        source_report = str(
            bidask.get("source_report")
            or "logs/reports/forecast_improvement/oanda_history_replay_validate_latest.json"
        )
        coverage_detail_repackage_command = None
        if coverage_detail_repackage_required:
            coverage_detail_repackage_command = (
                "python3 scripts/package_bidask_replay_precision_rules.py "
                f"--source-report {source_report} "
                "--output src/quant_rabbit/bidask_replay_precision_rules.json"
            )
        if code == "BIDASK_REPLAY_PRICE_TRUTH_PARTIAL" or price_truth_fetch_required:
            condition = (
                "missing OANDA bid/ask price-truth windows are fetched and the refreshed replay "
                "report either reaches PRICE_TRUTH_OK or proves that any remaining candidate is "
                "only rank/audit evidence"
            )
        elif code == "BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN":
            condition = (
                "OANDA bid/ask price truth is complete for loaded samples; collect more forecast_history "
                "samples across the under-sampled pair-directions, then rerun replay validation and require "
                "global all-currency sample coverage to graduate from UNDER_SAMPLED before claiming "
                "all-currency high-turn readiness"
            )
        elif code == "BIDASK_REPLAY_SUPPORT_NOT_DAILY_STABLE":
            condition = (
                "OANDA bid/ask price truth is already complete; wait for additional forecast_history "
                "coverage or new candidate days, then rerun replay validation and require at least one "
                "support rule to graduate from rank-only to live-grade DAILY_STABLE before counting it "
                "as high-turn daily firepower"
            )
        else:
            condition = (
                "OANDA bid/ask price truth is already complete; wait for additional forecast_history "
                "coverage or new candidate days, then rerun replay validation and require at least one "
                "contrarian replay rule to graduate from rank-only to DAILY_STABLE before counting weak "
                "forecast inversion as live-grade turnover firepower"
            )
        history_fetch_command = bidask.get("history_fetch_command") if price_truth_fetch_required else None
        return (
            condition,
            str(validation_command),
            {
                "replay_evidence_status": (
                    "PRICE_TRUTH_FETCH_REQUIRED"
                    if price_truth_fetch_required
                    else BIDASK_REPLAY_WAIT_STATUS
                ),
                "price_truth_fetch_required": price_truth_fetch_required,
                "support_rules": bidask.get("support_rules"),
                "daily_stable_support_rules": bidask.get("daily_stable_support_rules"),
                "rank_only_support_rules": bidask.get("rank_only_support_rules"),
                "edge_rules": bidask.get("edge_rules"),
                "daily_stable_edge_rules": bidask.get("daily_stable_edge_rules"),
                "rank_only_edge_rules": bidask.get("rank_only_edge_rules"),
                "contrarian_edge_rules": bidask.get("contrarian_edge_rules"),
                "daily_stable_contrarian_edge_rules": bidask.get("daily_stable_contrarian_edge_rules"),
                "rank_only_contrarian_edge_rules": bidask.get("rank_only_contrarian_edge_rules"),
                "negative_rules": bidask.get("negative_rules"),
                "packaged_source_report": bidask.get("source_report"),
                "packaged_by": bidask.get("packaged_by"),
                "packaged_generated_at_utc": bidask.get("generated_at_utc"),
                "packaged_history_dirs": bidask.get("history_dirs"),
                "price_truth_coverage": bidask.get("price_truth_coverage"),
                "forecast_sample_coverage_summary": sample_coverage,
                "forecast_sample_coverage_detail_status": coverage_detail_status,
                "coverage_detail_repackage_required": coverage_detail_repackage_required,
                "coverage_detail_repackage_command": coverage_detail_repackage_command,
                "coverage_detail_repair_action": (
                    "confirm the source report and history_dirs match the broad packaged replay "
                    "before repackaging; do not replace all-currency evidence with a narrower latest shard"
                    if coverage_detail_repackage_required
                    else None
                ),
                "minimum_directional_samples_for_precision_rule": sample_coverage.get(
                    "min_directional_samples_for_precision_rule"
                ),
                "minimum_active_days_for_daily_stability": sample_coverage.get(
                    "min_active_days_for_daily_stability"
                ),
                "under_sampled_gap_reason_counts": sample_coverage.get(
                    "under_sampled_gap_reason_counts"
                ),
                "under_sampled_pair_direction_examples": sample_coverage.get(
                    "under_sampled_pair_direction_examples"
                ),
                "under_sampled_pair_direction_detail_count": (
                    sample_coverage.get("under_sampled_pair_direction_detail_count")
                    if sample_coverage.get("under_sampled_pair_direction_detail_count") is not None
                    else _list_len(sample_coverage.get("under_sampled_pair_direction_examples"))
                ),
                "under_sampled_pair_direction_examples_omitted": sample_coverage.get(
                    "under_sampled_pair_direction_examples_omitted"
                ),
                "under_sampled_pair_direction_details": (
                    sample_coverage.get("under_sampled_pair_direction_details")
                    if isinstance(sample_coverage.get("under_sampled_pair_direction_details"), list)
                    else sample_coverage.get("under_sampled_pair_direction_examples")
                ),
                "pair_coverage_count": (
                    sample_coverage.get("pair_coverage_count")
                    if sample_coverage.get("pair_coverage_count") is not None
                    else _list_len(sample_coverage.get("pair_coverage_examples"))
                ),
                "pair_coverage_examples_omitted": sample_coverage.get(
                    "pair_coverage_examples_omitted"
                ),
                "pair_coverage": (
                    sample_coverage.get("pair_coverage")
                    if isinstance(sample_coverage.get("pair_coverage"), list)
                    else sample_coverage.get("pair_coverage_examples")
                ),
                "pair_coverage_examples": sample_coverage.get("pair_coverage_examples"),
                "daily_stability_requirements": bidask.get("daily_stability_requirements"),
                "history_fetch_command": history_fetch_command,
                "stale_history_fetch_command_suppressed": (
                    not price_truth_fetch_required and bool(bidask.get("history_fetch_command"))
                ),
                "history_fetch_command_count": price_truth.get("history_fetch_command_count"),
                "history_fetch_command_mode": price_truth.get("history_fetch_command_mode"),
                "missing_price_window_group_count": price_truth.get("missing_price_window_group_count"),
                "all_currency_sample_coverage_status": price_truth.get(
                    "all_currency_sample_coverage_status"
                ),
                "global_currency_validation_blocked": price_truth.get(
                    "global_currency_validation_blocked"
                ),
                "under_sampled_pair_direction_count": (
                    bidask.get("under_sampled_pair_direction_count")
                    if bidask.get("under_sampled_pair_direction_count") is not None
                    else price_truth.get("under_sampled_pair_direction_count")
                ),
                "under_sampled_pair_directions": (
                    bidask.get("under_sampled_pair_directions")
                    if bidask.get("under_sampled_pair_directions") is not None
                    else price_truth.get("under_sampled_pair_directions")
                ),
                "under_sampled_missing_evaluated_samples": (
                    bidask.get("under_sampled_missing_evaluated_samples")
                    if bidask.get("under_sampled_missing_evaluated_samples") is not None
                    else price_truth.get("under_sampled_missing_evaluated_samples")
                ),
                "rank_only_examples": examples[:3],
                **(
                    {
                        "current_intent_local_history_coverage": bidask_current_intent_history_coverage,
                    }
                    if isinstance(bidask_current_intent_history_coverage, dict)
                    else {}
                ),
            },
        )
    if code == "EXECUTION_LEDGER_GATEWAY_RECEIPT_STREAM_STALE":
        return (
            "execution_ledger gateway receipt stream is fresh enough to classify recent broker market-close "
            "truth against local GPT/gateway receipts",
            "PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit",
            {
                "gateway_event_stream_lag_minutes": evidence.get("gateway_event_stream_lag_minutes"),
                "latest_gateway_market_close_ts_utc": evidence.get("latest_gateway_market_close_ts_utc"),
            },
        )
    return (
        "the named acceptance P0 finding disappears from profitability_acceptance after its evidence metric changes",
        "PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance",
        {key: evidence.get(key) for key in sorted(evidence)[:6]},
    )


def _split_blocker(raw: str) -> tuple[str, str]:
    if ":" not in raw:
        return raw.strip() or "UNKNOWN_ACCEPTANCE_BLOCKER", raw.strip()
    code, message = raw.split(":", 1)
    return code.strip() or "UNKNOWN_ACCEPTANCE_BLOCKER", message.strip()


def _latest_ts_from_examples(evidence: dict[str, Any]) -> str | None:
    examples = evidence.get("examples") if isinstance(evidence.get("examples"), list) else []
    latest: datetime | None = None
    latest_raw: str | None = None
    for item in examples:
        if not isinstance(item, dict):
            continue
        raw = (
            item.get("ts_utc")
            or item.get("closed_at_utc")
            or item.get("trade_close_ts_utc")
            or item.get("timestamp_utc")
        )
        parsed = _parse_utc(raw)
        if parsed is None:
            continue
        if latest is None or parsed > latest:
            latest = parsed
            latest_raw = parsed.isoformat()
    return latest_raw


def _plus_days_iso(value: Any, days: int) -> str | None:
    parsed = _parse_utc(value)
    if parsed is None:
        return None
    return (parsed + timedelta(days=days)).isoformat()


def _example_trade_ids(evidence: dict[str, Any]) -> list[str]:
    examples = evidence.get("examples") if isinstance(evidence.get("examples"), list) else []
    trade_ids: list[str] = []
    for item in examples:
        if not isinstance(item, dict):
            continue
        trade_id = item.get("trade_id")
        if trade_id is None:
            continue
        trade_ids.append(str(trade_id))
    return trade_ids[:8]


def _example_trade_ids_from_top_misses(evidence: dict[str, Any]) -> list[str]:
    top: list[Any] = []
    for key in ("top_profit_capture_misses", "top_repair_replay_triggers"):
        values = evidence.get(key)
        if isinstance(values, list):
            top.extend(values)
    trade_ids: list[str] = []
    for item in top:
        if not isinstance(item, dict):
            continue
        trade_id = item.get("trade_id")
        if trade_id is None:
            continue
        trade_ids.append(str(trade_id))
    trade_ids = list(dict.fromkeys(trade_ids))
    return trade_ids[:8]


def _target_firepower_summary(payload: dict[str, Any]) -> dict[str, Any]:
    high_precision = _firepower_bucket_summary(payload.get("high_precision"))
    evidence_queue = _firepower_bucket_summary(payload.get("evidence_queue"))
    bucket_name, bucket = _best_firepower_bucket(
        high_precision=high_precision,
        evidence_queue=evidence_queue,
    )
    minimum = _float(payload.get("minimum_return_pct")) or 5.0
    target = _float(payload.get("target_return_pct")) or 10.0
    estimated_return = _float(bucket.get("estimated_return_pct_per_active_day_at_observed_frequency"))
    return {
        "status": payload.get("status"),
        "target_open": payload.get("target_open"),
        "minimum_return_pct": _round_optional(minimum, 3),
        "target_return_pct": _round_optional(target, 3),
        "per_trade_risk_pct_lens": _round_optional(payload.get("per_trade_risk_pct_lens"), 3),
        "best_bucket": bucket_name,
        "minimum_5pct_estimated_reachable": bool(estimated_return >= minimum),
        "target_10pct_estimated_reachable": bool(estimated_return >= target),
        "audit_only_no_live_permission": True,
        "operational_minimum_5pct_reachable": False,
        "operational_blocker_codes": ["OPERATIONAL_CONTEXT_NOT_EVALUATED"],
        "operational_basis": "audit-only firepower has no live permission until support gates are clear",
        "high_precision": high_precision,
        "evidence_queue": evidence_queue,
        "contract": payload.get("contract"),
    }


def _annotate_operational_target_firepower(
    target_firepower: dict[str, Any],
    *,
    audit_minimum_reachable: bool,
    send_allowed: bool,
    status: str,
    blockers: list[dict[str, Any]],
    guardian: dict[str, Any],
    entry: dict[str, Any],
) -> None:
    operational_blockers: list[str] = []
    if not audit_minimum_reachable:
        operational_blockers.append("AUDIT_FIREPOWER_BELOW_5PCT")
    if status != STATUS_READY:
        operational_blockers.extend(
            str(item.get("code") or "UNKNOWN_SUPPORT_BLOCKER")
            for item in blockers
            if isinstance(item, dict)
        )
    if _guardian_counts_as_inactive_for_support(guardian):
        operational_blockers.append("POSITION_GUARDIAN_INACTIVE")
    if _float(entry.get("live_ready_lanes")) <= 0:
        operational_blockers.append("NO_LIVE_READY_LANES")
    artifact_freshness = (
        entry.get("artifact_freshness") if isinstance(entry.get("artifact_freshness"), dict) else {}
    )
    if artifact_freshness.get("order_intents_stale_against_broker_snapshot"):
        operational_blockers.append("ORDER_INTENTS_STALE_AGAINST_BROKER_SNAPSHOT")
    if not send_allowed:
        operational_blockers.append("FRESH_ENTRY_SEND_NOT_ALLOWED")
    operational_blockers = list(dict.fromkeys(item for item in operational_blockers if item))
    target_firepower["operational_minimum_5pct_reachable"] = bool(
        audit_minimum_reachable and send_allowed and not operational_blockers
    )
    target_firepower["operational_blocker_codes"] = operational_blockers
    target_firepower["operational_basis"] = (
        "live-ready support gates clear and audit firepower clears 5% floor"
        if target_firepower["operational_minimum_5pct_reachable"]
        else "audit-only firepower is blocked from live use until support gates clear"
    )


def _firepower_bucket_summary(raw: Any) -> dict[str, Any]:
    bucket = raw if isinstance(raw, dict) else {}
    return {
        "estimated_return_pct_per_active_day_at_observed_frequency": _round_optional(
            bucket.get("estimated_return_pct_per_active_day_at_observed_frequency"),
            6,
        ),
        "weighted_return_pct_per_trade_at_risk_lens": _round_optional(
            bucket.get("weighted_return_pct_per_trade_at_risk_lens"),
            6,
        ),
        "observed_attempts_per_active_day": _round_optional(
            bucket.get("observed_attempts_per_active_day"),
            6,
        ),
        "trades_needed_for_minimum_5pct_at_weighted_expectancy": bucket.get(
            "trades_needed_for_minimum_5pct_at_weighted_expectancy"
        ),
        "trades_needed_for_target_10pct_at_weighted_expectancy": bucket.get(
            "trades_needed_for_target_10pct_at_weighted_expectancy"
        ),
        "pair_count": bucket.get("pair_count"),
        "unique_vehicle_count": bucket.get("unique_vehicle_count"),
        "top_vehicle_keys": list(bucket.get("top_vehicle_keys") or [])[:5],
    }


def _best_firepower_bucket(
    *,
    high_precision: dict[str, Any],
    evidence_queue: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    candidates = [
        ("high_precision", high_precision),
        ("evidence_queue", evidence_queue),
    ]
    candidates.sort(
        key=lambda item: _float(item[1].get("estimated_return_pct_per_active_day_at_observed_frequency")),
        reverse=True,
    )
    name, bucket = candidates[0]
    if _float(bucket.get("estimated_return_pct_per_active_day_at_observed_frequency")) <= 0:
        return None, {}
    return name, bucket


def _build_blockers(
    *,
    guardian: dict[str, Any],
    target: dict[str, Any],
    profit_capture: dict[str, Any],
    p0_findings: list[dict[str, Any]],
    entry: dict[str, Any],
    acceptance: dict[str, Any],
    qr_trader_run_watchdog: dict[str, Any],
    guardian_receipt_consumption: dict[str, Any],
    guardian_receipt_operator_review: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if _watchdog_counts_as_support_blocker(qr_trader_run_watchdog):
        status = qr_trader_run_watchdog.get("status")
        severity = "P0" if status in {"BROKEN", "STALE"} else "P1"
        blockers.append(
            {
                "code": "QR_TRADER_SCHEDULED_RUN_WATCHDOG_BLOCKED",
                "severity": severity,
                "message": (
                    f"qr-trader scheduled-run watchdog status is {status}; "
                    f"last_run={qr_trader_run_watchdog.get('last_trader_run_at')} "
                    f"minutes_since={qr_trader_run_watchdog.get('minutes_since_last_run')}"
                ),
            }
        )
    if (
        guardian_receipt_consumption.get("normal_routing_allowed") is False
        and not guardian_receipt_consumption.get("missing")
    ):
        classifications = [
            str(item.get("classification") or "")
            for item in guardian_receipt_consumption.get("classifications", [])
            if isinstance(item, dict)
        ]
        blockers.append(
            {
                "code": "GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING",
                "severity": "P0",
                "message": (
                    "guardian receipt consumption status does not allow normal new-entry routing; "
                    f"status={guardian_receipt_consumption.get('status')} "
                    f"classifications={','.join(classifications) or 'none'}"
                ),
            }
        )
    if _guardian_receipt_operator_review_blocks_normal_routing(
        guardian_receipt_consumption,
        guardian_receipt_operator_review,
    ):
        decisions = [
            str(item.get("operator_decision") or "")
            for item in guardian_receipt_operator_review.get("classifications", [])
            if isinstance(item, dict)
        ]
        blockers.append(
            {
                "code": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING",
                "severity": "P0",
                "message": (
                    "guardian receipt operator review does not allow normal new-entry routing; "
                    f"status={guardian_receipt_operator_review.get('status')} "
                    f"decisions={','.join(decisions) or 'none'}"
                ),
            }
        )
    for issue in qr_trader_run_watchdog.get("guardian_receipt_issues", []) or []:
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity") or "WARN")
        blockers.append(
            {
                "code": str(issue.get("code") or "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER"),
                "severity": severity,
                "message": str(issue.get("message") or "guardian receipt was not consumed by qr-trader"),
            }
        )
    if _guardian_counts_as_inactive_for_support(guardian):
        blockers.append(
            {
                "code": "POSITION_GUARDIAN_INACTIVE",
                "severity": "P0",
                "message": (
                    "position guardian is required but not proven active with a fresh heartbeat; "
                    "fresh entries stay blocked because plus P/L can reverse between full cycles"
                ),
            }
        )
    profit_capture_block_severity = _profit_capture_support_blocker_severity(profit_capture)
    if profit_capture_block_severity:
        blockers.append(
            {
                "code": PROFIT_CAPTURE_MISS,
                "severity": profit_capture_block_severity,
                "message": _profit_capture_miss_message(profit_capture),
            }
        )
    if p0_findings:
        blockers.append(
            {
                "code": "SELF_IMPROVEMENT_P0_PRESENT",
                "severity": "P0",
                "message": f"self-improvement audit has {len(p0_findings)} P0 finding(s)",
            }
        )
    if target.get("target_open") and entry["live_ready_lanes"] == 0:
        artifact_freshness = (
            entry.get("artifact_freshness") if isinstance(entry.get("artifact_freshness"), dict) else {}
        )
        stale_intents = bool(artifact_freshness.get("order_intents_stale_against_broker_snapshot"))
        blockers.append(
            {
                "code": "NO_LIVE_READY_LANES",
                "severity": "P1",
                "message": (
                    "daily target is open but no lane is LIVE_READY; order_intents is older than "
                    "broker_snapshot, so refresh the evidence packet before treating blocker counts as current"
                    if stale_intents
                    else "daily target is open but no lane is LIVE_READY"
                ),
            }
        )
        if stale_intents:
            blockers.append(
                {
                    "code": "ORDER_INTENTS_STALE_AGAINST_BROKER_SNAPSHOT",
                    "severity": "P1",
                    "message": artifact_freshness.get("reason")
                    or "broker_snapshot is fresher than order_intents",
                }
            )
    if str(acceptance.get("status") or "") not in {"", "PROFITABILITY_ACCEPTANCE_PASSED", "PASSED"}:
        blockers.append(
            {
                "code": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                "severity": "P0",
                "message": f"profitability acceptance status is {acceptance.get('status')}",
            }
        )
    acceptance_freshness = (
        acceptance.get("artifact_freshness")
        if isinstance(acceptance.get("artifact_freshness"), dict)
        else {}
    )
    if acceptance_freshness.get("profitability_acceptance_stale_against_inputs"):
        blockers.append(
            {
                "code": "PROFITABILITY_ACCEPTANCE_STALE_AGAINST_INPUTS",
                "severity": "P0",
                "message": acceptance_freshness.get("reason")
                or "profitability acceptance is stale against current input evidence",
            }
        )
    return blockers


def _guardian_receipt_operator_review_blocks_normal_routing(
    guardian_receipt_consumption: dict[str, Any],
    guardian_receipt_operator_review: dict[str, Any],
) -> bool:
    if guardian_receipt_operator_review.get("missing"):
        return False
    if guardian_receipt_operator_review.get("normal_routing_allowed") is not False:
        return False
    return not _guardian_receipt_consumption_durably_clears_operator_review(
        guardian_receipt_consumption,
    )


def _guardian_receipt_consumption_durably_clears_operator_review(
    guardian_receipt_consumption: dict[str, Any],
) -> bool:
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


def _profit_capture_support_blocker_severity(profit_capture: dict[str, Any]) -> str | None:
    if _float(profit_capture.get("missed_loss_closes")) <= 0:
        return None
    if str(profit_capture.get("status") or "") in {
        "WAITING_FOR_POST_REPAIR_SAMPLE",
        "HISTORICAL_DIAGNOSTIC_ONLY",
    }:
        return None
    production_gate_block = (
        not profit_capture.get("repair_replay_contract_present")
        or int(_float(profit_capture.get("repair_replay_triggered"))) > 0
    )
    return "P0" if production_gate_block else "P1"


def _profit_capture_miss_message(profit_capture: dict[str, Any]) -> str:
    message = (
        profit_capture.get("message")
        or f"{profit_capture['missed_loss_closes']} losing close(s) missed TP-progress capture"
    )
    repair_triggers = int(_float(profit_capture.get("repair_replay_triggered")))
    if repair_triggers > 0:
        repair_delta = profit_capture.get("repair_replay_delta_jpy")
        if repair_delta is None:
            return f"{message}; production-gate replay triggers={repair_triggers}"
        return f"{message}; production-gate replay triggers={repair_triggers} delta={repair_delta} JPY"
    delta = profit_capture.get("counterfactual_profit_capture_delta_jpy")
    if delta is None:
        return message
    return f"{message}; conservative candle counterfactual delta={delta} JPY"


def _infer_oanda_history_root(*, output_path: Path, artifact_paths: list[Path]) -> Path:
    for path in artifact_paths:
        root = _project_root_from_artifact_path(path)
        if root is not None:
            return root / "logs" / "replay" / "oanda_history"
    return output_path.parent.parent / "logs" / "replay" / "oanda_history"


def _project_root_from_artifact_path(path: Path) -> Path | None:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    parts = candidate.parts
    for index in range(len(parts) - 1, 0, -1):
        if parts[index] in {"data", "logs", "src", "docs"}:
            return Path(*parts[:index])
    return None


def _oanda_audit_only_history_coverage(pairs: list[str], *, history_root: Path) -> dict[str, Any]:
    required_pairs = sorted({str(pair).upper() for pair in pairs if str(pair or "").strip()})
    required_granularities = [
        item.strip().upper()
        for item in OANDA_AUDIT_ONLY_REPLAY_HISTORY_GRANULARITIES.split(",")
        if item.strip()
    ]
    if not required_pairs:
        return {
            "status": "NOT_REQUIRED",
            "history_root": str(history_root),
            "required_pairs": [],
            "required_granularities": required_granularities,
            "covered_pairs_by_granularity": {granularity: [] for granularity in required_granularities},
            "missing_pairs_by_granularity": {granularity: [] for granularity in required_granularities},
            "fetch_commands": [],
            "complete": True,
        }

    best_files: dict[tuple[str, str], dict[str, Any]] = {}
    if history_root.exists():
        for path in history_root.rglob("*.jsonl*"):
            match = OANDA_HISTORY_FILENAME_RE.match(path.name)
            if not match:
                continue
            pair = match.group("pair")
            granularity = match.group("granularity")
            if pair not in required_pairs or granularity not in required_granularities:
                continue
            row = _oanda_history_file_summary(path, match)
            if row is None:
                continue
            key = (pair, granularity)
            current = best_files.get(key)
            if current is None or _oanda_history_file_rank(row) > _oanda_history_file_rank(current):
                best_files[key] = row

    covered: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    selected_files: list[dict[str, Any]] = []
    for granularity in required_granularities:
        covered[granularity] = []
        missing[granularity] = []
        for pair in required_pairs:
            row = best_files.get((pair, granularity))
            if row and _float(row.get("history_days")) >= OANDA_AUDIT_ONLY_REPLAY_HISTORY_MIN_COVERED_DAYS:
                covered[granularity].append(pair)
                selected_files.append(row)
            else:
                missing[granularity].append(pair)

    fetch_commands = _oanda_history_fetch_commands(missing)
    complete = not fetch_commands
    return {
        "status": "LOCAL_HISTORY_COMPLETE" if complete else "PRICE_TRUTH_FETCH_REQUIRED",
        "history_root": str(history_root),
        "required_pairs": required_pairs,
        "required_granularities": required_granularities,
        "required_history_days": OANDA_AUDIT_ONLY_REPLAY_HISTORY_DAYS,
        "minimum_covered_history_days": OANDA_AUDIT_ONLY_REPLAY_HISTORY_MIN_COVERED_DAYS,
        "covered_pairs_by_granularity": covered,
        "missing_pairs_by_granularity": missing,
        "selected_files": selected_files,
        "fetch_commands": fetch_commands,
        "complete": complete,
    }


def _bidask_current_intent_history_coverage(
    pairs: list[str],
    *,
    history_root: Path,
) -> dict[str, Any]:
    coverage = _oanda_audit_only_history_coverage(pairs, history_root=history_root)
    status = str(coverage.get("status") or "")
    if status == "PRICE_TRUTH_FETCH_REQUIRED":
        status = "LOCAL_HISTORY_GAP"
    elif status == "LOCAL_HISTORY_COMPLETE":
        status = "LOCAL_HISTORY_COMPLETE"
    elif status == "NOT_REQUIRED":
        status = "NOT_REQUIRED"
    coverage["status"] = status
    coverage["coverage_basis"] = "CURRENT_ORDER_INTENT_PAIRS_LOCAL_OANDA_HISTORY"
    coverage["diagnostic_only"] = True
    coverage["price_truth_fetch_required"] = False
    coverage["clears_bidask_replay_gate"] = False
    coverage["reason"] = (
        "local BA candle availability for the current order_intents pair universe; "
        "fetch commands here are read-only diagnostics and do not clear packaged "
        "bid/ask replay price-truth or forecast-sample gates by themselves"
    )
    return coverage


def _oanda_history_file_summary(path: Path, match: re.Match[str]) -> dict[str, Any] | None:
    start = _parse_oanda_history_filename_time(match.group("start"))
    end = _parse_oanda_history_filename_time(match.group("end"))
    if start is None or end is None or end <= start:
        return None
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    return {
        "pair": match.group("pair"),
        "granularity": match.group("granularity"),
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "history_days": round((end - start).total_seconds() / 86400.0, 6),
        "size_bytes": size_bytes,
        "path": str(path),
    }


def _parse_oanda_history_filename_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _oanda_history_file_rank(row: dict[str, Any]) -> tuple[float, str, int]:
    return (
        _float(row.get("history_days")),
        str(row.get("end_utc") or ""),
        int(_float(row.get("size_bytes"))),
    )


def _oanda_history_fetch_command(*, pairs: list[str], granularities: str) -> str:
    pair_arg = ",".join(sorted({str(pair) for pair in pairs if str(pair)})) or "EUR_USD"
    return (
        "PYTHONPATH=src python3 scripts/oanda_history_fetch.py "
        f"--pairs {pair_arg} --granularities {granularities} --price BA "
        f"--days {OANDA_AUDIT_ONLY_REPLAY_HISTORY_DAYS} --output-dir logs/replay/oanda_history"
    )


def _oanda_history_fetch_commands(missing_pairs_by_granularity: dict[str, list[str]]) -> list[str]:
    granularities_by_pair_set: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for granularity, pairs in missing_pairs_by_granularity.items():
        pair_set = tuple(sorted({str(pair) for pair in pairs if str(pair)}))
        if pair_set:
            granularities_by_pair_set[pair_set].append(str(granularity))
    return [
        _oanda_history_fetch_command(
            pairs=list(pair_set),
            granularities=",".join(granularities),
        )
        for pair_set, granularities in granularities_by_pair_set.items()
    ]


def _usable_oanda_history_coverage(
    coverage: dict[str, Any] | None,
    pairs: list[str],
) -> dict[str, Any]:
    required_pairs = sorted({str(pair).upper() for pair in pairs if str(pair or "").strip()})
    if isinstance(coverage, dict) and coverage.get("required_pairs") == required_pairs:
        return coverage
    return _oanda_audit_only_missing_history_coverage(required_pairs)


def _oanda_audit_only_missing_history_coverage(pairs: list[str]) -> dict[str, Any]:
    required_pairs = sorted({str(pair).upper() for pair in pairs if str(pair or "").strip()})
    required_granularities = [
        item.strip().upper()
        for item in OANDA_AUDIT_ONLY_REPLAY_HISTORY_GRANULARITIES.split(",")
        if item.strip()
    ]
    return {
        "status": "PRICE_TRUTH_FETCH_REQUIRED",
        "history_root": str(OANDA_AUDIT_ONLY_REPLAY_HISTORY_ROOT),
        "required_pairs": required_pairs,
        "required_granularities": required_granularities,
        "required_history_days": OANDA_AUDIT_ONLY_REPLAY_HISTORY_DAYS,
        "minimum_covered_history_days": OANDA_AUDIT_ONLY_REPLAY_HISTORY_MIN_COVERED_DAYS,
        "covered_pairs_by_granularity": {granularity: [] for granularity in required_granularities},
        "missing_pairs_by_granularity": {
            granularity: required_pairs for granularity in required_granularities
        },
        "selected_files": [],
        "fetch_commands": [
            _oanda_history_fetch_command(
                pairs=required_pairs,
                granularities=OANDA_AUDIT_ONLY_REPLAY_HISTORY_GRANULARITIES,
            )
        ],
        "complete": False,
    }


def _oanda_history_coverage_is_complete(history_coverage: dict[str, Any]) -> bool:
    fetch_commands = (
        history_coverage.get("fetch_commands")
        if isinstance(history_coverage.get("fetch_commands"), list)
        else []
    )
    return bool(history_coverage.get("complete")) and not fetch_commands


def _oanda_audit_only_candidate_has_clearable_replay(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get("historical_replay_can_clear_local_tp_proof")
        or candidate.get("oanda_replay_live_permission")
    )


def _oanda_audit_only_candidates_have_clearable_replay(
    candidates: list[dict[str, Any]],
) -> bool:
    return any(_oanda_audit_only_candidate_has_clearable_replay(candidate) for candidate in candidates)


def _guardian_waits_for_live_runtime_lock(guardian: dict[str, Any]) -> bool:
    return bool(
        guardian.get("required")
        and not guardian.get("active")
        and guardian.get("launchd_loaded")
        and not guardian.get("heartbeat_fresh")
        and guardian.get("live_runtime_lock_active")
    )


def _guardian_counts_as_inactive_for_support(guardian: dict[str, Any]) -> bool:
    return bool(
        guardian.get("required")
        and not guardian.get("active")
        and not _guardian_waits_for_live_runtime_lock(guardian)
    )


def _operator_actions(
    *,
    status: str,
    blockers: list[dict[str, Any]],
    guardian: dict[str, Any],
    profit_capture: dict[str, Any],
    entry: dict[str, Any],
    acceptance: dict[str, Any],
    broker: dict[str, Any] | None = None,
    oanda_history_coverage: dict[str, Any] | None = None,
    qr_trader_run_watchdog: dict[str, Any] | None = None,
    guardian_receipt_consumption: dict[str, Any] | None = None,
    guardian_receipt_operator_review: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    broker = broker if isinstance(broker, dict) else {}
    qr_trader_run_watchdog = (
        qr_trader_run_watchdog if isinstance(qr_trader_run_watchdog, dict) else {}
    )
    guardian_receipt_consumption = (
        guardian_receipt_consumption if isinstance(guardian_receipt_consumption, dict) else {}
    )
    guardian_receipt_operator_review = (
        guardian_receipt_operator_review if isinstance(guardian_receipt_operator_review, dict) else {}
    )
    actions: list[dict[str, Any]] = [
        {
            "code": "REFRESH_SUPPORT_PANEL",
            "command": "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
            "requires_explicit_operator_approval": False,
            "reason": "read-only status refresh for trader operations",
        }
    ]
    if _watchdog_counts_as_support_blocker(qr_trader_run_watchdog) or qr_trader_run_watchdog.get(
        "guardian_receipt_issues"
    ):
        actions.append(
            {
                "code": "REFRESH_QR_TRADER_RUN_WATCHDOG",
                "command": "PYTHONPATH=src python3 tools/qr_trader_run_watchdog.py",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "read-only scheduled-run watchdog refresh; does not run qr-trader, Codex wake, "
                    "or broker calls by default"
                ),
            }
        )
    if (
        qr_trader_run_watchdog.get("guardian_receipt_issues")
        and guardian_receipt_consumption.get("normal_routing_allowed") is not True
    ):
        classifications = guardian_receipt_consumption.get("classifications")
        if isinstance(classifications, list) and classifications:
            actions.append(
                {
                    "code": "REVIEW_GUARDIAN_RECEIPT_CONSUMPTION",
                    "command": "sed -n '1,160p' docs/guardian_receipt_consumption_report.md",
                    "requires_explicit_operator_approval": False,
                    "reason": _guardian_receipt_consumption_next_action(
                        qr_trader_run_watchdog,
                        guardian_receipt_consumption,
                    ),
                }
            )
        else:
            actions.append(
                {
                    "code": "WRITE_GUARDIAN_RECEIPT_CONSUMPTION",
                    "command": "PYTHONPATH=src python3 -m quant_rabbit.cli trader-draft-decision",
                    "requires_explicit_operator_approval": False,
                    "reason": _guardian_receipt_consumption_next_action(
                        qr_trader_run_watchdog,
                        guardian_receipt_consumption,
                    ),
                }
            )
    if _guardian_receipt_operator_review_blocks_normal_routing(
        guardian_receipt_consumption,
        guardian_receipt_operator_review,
    ):
        actions.append(
            {
                "code": "REVIEW_GUARDIAN_RECEIPT_OPERATOR_REVIEW",
                "command": "sed -n '1,200p' docs/guardian_receipt_operator_review_report.md",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "operator review artifact exists but does not clear normal routing; inspect the "
                    "decision row before any fresh-entry routing"
                ),
            }
        )
    recommended = qr_trader_run_watchdog.get("recommended_operator_action")
    if isinstance(recommended, dict) and recommended.get("code") not in {
        None,
        "NO_OPERATOR_ACTION_REQUIRED",
    }:
        actions.append(
            {
                "code": f"QR_TRADER_WATCHDOG_{recommended.get('code')}",
                "command": recommended.get("command") or "inspect qr-trader scheduled-run watchdog report",
                "requires_explicit_operator_approval": bool(
                    recommended.get("requires_explicit_operator_approval")
                ),
                "reason": recommended.get("reason") or "qr-trader run watchdog recommended action",
            }
        )
    if guardian["required"] and not guardian["active"]:
        actions.extend(
            [
                {
                    "code": "CHECK_POSITION_GUARDIAN_PREFLIGHT",
                    "command": "scripts/install-position-guardian.sh --check",
                    "requires_explicit_operator_approval": False,
                    "reason": "verify live runtime can safely run the fast profit-capture guardian",
                },
                {
                    "code": "CHECK_POSITION_GUARDIAN_STATUS",
                    "command": "scripts/install-position-guardian.sh --status",
                    "requires_explicit_operator_approval": False,
                    "reason": "show plist, launchd, and heartbeat state without changing live services",
                },
            ]
        )
        if _guardian_waits_for_live_runtime_lock(guardian):
            actions.append(
                {
                    "code": "WAIT_FOR_LIVE_RUNTIME_LOCK_RELEASE",
                    "command": "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                    "requires_explicit_operator_approval": False,
                    "reason": "guardian launchd is loaded, but heartbeat is stale because another live runtime cycle currently owns the shared lock",
                }
            )
        else:
            actions.append(
                {
                    "code": "LOAD_POSITION_GUARDIAN_ONLY_IF_APPROVED",
                    "command": "scripts/install-position-guardian.sh",
                    "requires_explicit_operator_approval": True,
                    "reason": "loading launchd changes live support services and must be explicit",
                }
            )
    profit_capture_block_severity = _profit_capture_support_blocker_severity(profit_capture)
    profit_capture_status = str(profit_capture.get("status") or "")
    if profit_capture_block_severity:
        actions.append(
            {
                "code": "RECHECK_TIMING_CAPTURE_MISSES",
                "command": MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND,
                "requires_explicit_operator_approval": False,
                "reason": "recompute TP-progress misses and confirm capture gap is shrinking",
            }
        )
    elif profit_capture_status == "WAITING_FOR_POST_REPAIR_SAMPLE":
        actions.append(
            {
                "code": "WAIT_FOR_POST_REPAIR_TP_PROGRESS_SAMPLE",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "pre-repair TP-progress misses are historical diagnostics; wait for a "
                    "post-repair live loss-close sample before rerunning the repair loop"
                ),
            }
        )
    elif profit_capture_status == "HISTORICAL_DIAGNOSTIC_ONLY":
        actions.append(
            {
                "code": "MONITOR_POST_REPAIR_TP_PROGRESS_EVIDENCE",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "post-repair production-gate replay is clean; keep monitoring without "
                    "treating pre-repair misses as a live blocker"
                ),
            }
        )
    artifact_freshness = (
        entry.get("artifact_freshness") if isinstance(entry.get("artifact_freshness"), dict) else {}
    )
    if artifact_freshness.get("order_intents_stale_against_broker_snapshot"):
        actions.append(
            {
                "code": "REGENERATE_INTENTS_FROM_CURRENT_BROKER_SNAPSHOT",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts",
                "requires_explicit_operator_approval": False,
                "reason": "order_intents is older than broker_snapshot; refresh intents before ranking live blockers",
            }
        )
    acceptance_freshness = (
        acceptance.get("artifact_freshness")
        if isinstance(acceptance.get("artifact_freshness"), dict)
        else {}
    )
    if acceptance_freshness.get("profitability_acceptance_stale_against_inputs"):
        refresh_commands = acceptance_freshness.get("refresh_commands")
        command = (
            " && ".join(str(item) for item in refresh_commands if str(item).strip())
            if isinstance(refresh_commands, list) and refresh_commands
            else "PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance"
        )
        actions.append(
            {
                "code": "REFRESH_PROFITABILITY_ACCEPTANCE_FROM_CURRENT_INPUTS",
                "command": command,
                "requires_explicit_operator_approval": False,
                "reason": acceptance_freshness.get("reason")
                or "profitability_acceptance is stale against current input evidence",
            }
        )
    stale_support_blocker_codes = [
        str(code)
        for code in entry.get("stale_support_blocker_codes") or []
        if str(code).strip()
    ]
    if stale_support_blocker_codes:
        actions.append(
            {
                "code": "REGENERATE_INTENTS_FROM_CURRENT_SUPPORT_STATE",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "order_intents still contains cleared support blocker(s): "
                    f"{', '.join(stale_support_blocker_codes)}"
                ),
            }
        )
    if entry["live_ready_lanes"] == 0:
        actions.append(
            {
                "code": "REFRESH_EVIDENCE_PACKET",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli cycle-refresh --daily-risk-pct 10",
                "requires_explicit_operator_approval": False,
                "reason": "refresh forecasts, intents, sidecar audits, and route from current broker truth",
            }
        )
    if int(_float(broker.get("unknown_owner_positions"))) > 0:
        actions.append(
            {
                "code": "REVIEW_UNKNOWN_OWNER_EXPOSURE",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "broker truth includes manual/tagless exposure that is TP-managed only; "
                    "classify, adopt, or manually resolve it before treating margin capacity as repairable"
                ),
            }
        )
    if acceptance.get("status") == "PROFITABILITY_ACCEPTANCE_BLOCKED":
        actions.append(
            {
                "code": "READ_ACCEPTANCE_BLOCKERS",
                "command": "sed -n '1,220p' docs/profitability_acceptance_report.md",
                "requires_explicit_operator_approval": False,
                "reason": "inspect red/green acceptance invariants before increasing turnover",
            }
        )
    repair_plan = acceptance.get("repair_plan") if isinstance(acceptance.get("repair_plan"), dict) else {}
    repair_items = repair_plan.get("items") if isinstance(repair_plan.get("items"), list) else []
    evidence_items = (
        repair_plan.get("evidence_collection_items")
        if isinstance(repair_plan.get("evidence_collection_items"), list)
        else []
    )
    if repair_items:
        actions.append(
            {
                "code": "FOLLOW_ACCEPTANCE_REPAIR_PLAN",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "use profitability_acceptance.repair_plan clearance conditions; rerunning acceptance "
                    "alone will loop until those proof metrics change"
                ),
            }
        )
    if evidence_items:
        first_evidence = evidence_items[0] if isinstance(evidence_items[0], dict) else {}
        summary = (
            first_evidence.get("evidence_summary")
            if isinstance(first_evidence.get("evidence_summary"), dict)
            else {}
        )
        price_truth_fetch_required = _bidask_summary_requires_price_truth_fetch(summary)
        history_fetch = summary.get("history_fetch_command")
        if history_fetch and price_truth_fetch_required:
            actions.append(
                {
                    "code": "FETCH_BIDASK_REPLAY_HISTORY",
                    "command": str(history_fetch),
                    "requires_explicit_operator_approval": False,
                    "reason": "read-only OANDA BA candle fetch for rank-only bid/ask replay validation",
                }
            )
        validation_command = first_evidence.get("verification_command")
        if validation_command and price_truth_fetch_required:
            actions.append(
                {
                    "code": "VALIDATE_BIDASK_REPLAY_HISTORY",
                    "command": str(validation_command),
                    "requires_explicit_operator_approval": False,
                    "reason": "rerun historical bid/ask replay before treating replay support as live-grade",
                }
            )
    repair_codes = {str(item.get("code")) for item in repair_items if isinstance(item, dict)}
    if "LOSS_CLOSE_GATE_EVIDENCE_MISSING" in repair_codes:
        actions.append(
            {
                "code": "VERIFY_CLOSE_GATE_EVIDENCE",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit",
                "requires_explicit_operator_approval": False,
                "reason": "confirm future GPT loss-side closes have durable PASS close_gate_evidence",
            }
        )
    if "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK" in repair_codes:
        actions.append(
            {
                "code": "RECHECK_LOSS_CLOSE_LEAK_WINDOW",
                "command": MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND,
                "requires_explicit_operator_approval": False,
                "reason": "verify the 7-day loss-close leak window is shrinking before adding turnover",
            }
        )
    if {
        "TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED",
        "TP_PROGRESS_REPLAY_REPAIR_UNPROVED",
        "TP_PROGRESS_REPAIR_REPLAY_CONTRACT_MISSING",
        "MONTH_SCALE_LOSS_CLOSE_REPLAY_REQUIRED",
        "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
    } & repair_codes:
        actions.append(
            {
                "code": "VERIFY_TP_PROGRESS_REPLAY_REPAIR",
                "command": MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND,
                "requires_explicit_operator_approval": False,
                "reason": (
                    "prove the OANDA candle replay TP-progress miss has cleared at the "
                    "required coverage before treating high-turnover profit capture as repaired"
                ),
            }
        )
    if entry.get("global_unlock_frontier"):
        actions.append(
            {
                "code": "WORK_GLOBAL_UNLOCK_FRONTIER",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "some lanes are blocked only by global support/profitability gates; "
                    "clear those gates before adding unrelated indicators"
                ),
            }
        )
    if entry.get("repair_frontier_remaining_blockers"):
        top = entry["repair_frontier_remaining_blockers"][0]
        actions.append(
            {
                "code": "WORK_REPAIR_FRONTIER_REMAINING_BLOCKERS",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "guardian/global repairs are not enough; top remaining repair-frontier blocker is "
                    f"{top.get('code')} across {top.get('count')} lane(s)"
                ),
            }
        )
    if entry.get("oanda_audit_only_local_tp_proof_required"):
        candidates = _oanda_audit_only_local_tp_candidates(entry)
        pairs = sorted(
            {
                str(item.get("pair"))
                for item in candidates
                if item.get("pair")
            }
        )
        pair_arg = ",".join(pairs) if pairs else "EUR_USD"
        coverage = _usable_oanda_history_coverage(oanda_history_coverage, pairs)
        fetch_commands = coverage.get("fetch_commands") if isinstance(coverage, dict) else []
        history_complete = _oanda_history_coverage_is_complete(coverage)
        replay_can_clear = _oanda_audit_only_candidates_have_clearable_replay(candidates)
        if fetch_commands:
            for fetch_command in fetch_commands:
                actions.append(
                    {
                        "code": "MINE_LOCAL_TP_PROOF_FOR_OANDA_AUDIT_ONLY",
                        "command": str(fetch_command),
                        "requires_explicit_operator_approval": False,
                        "reason": (
                            "OANDA campaign firepower is audit-only for these lanes; fetch only missing "
                            "spread-included candle truth before treating the candidate as improved evidence"
                        ),
                    }
                )
        if history_complete and not replay_can_clear:
            actions.append(
                {
                    "code": "WAIT_FOR_OANDA_AUDIT_ONLY_LOCAL_TP_PROOF",
                    "command": "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                    "requires_explicit_operator_approval": False,
                    "reason": (
                        "local OANDA S5/M5 history is already complete and the latest replay did not "
                        "clear local TP proof; wait for new TAKE_PROFIT_ORDER receipts, new forecast/"
                        "candle evidence, or live-grade HARVEST promotion before rerunning the replay "
                        "mining/package loop"
                    ),
                }
            )
        else:
            actions.append(
                {
                    "code": "VALIDATE_OANDA_AUDIT_ONLY_BIDASK_REPLAY",
                    "command": (
                        "PYTHONPATH=src python3 scripts/oanda_history_replay_validate.py "
                        "--history-dir logs/replay/oanda_history --granularity S5"
                    ),
                    "requires_explicit_operator_approval": False,
                    "reason": (
                        "score forecast_history against local OANDA S5 bid/ask candles with spread included; "
                        "this proves whether the prediction side actually made money on historical candles"
                    ),
                }
            )
            actions.append(
                {
                    "code": "MINE_OANDA_AUDIT_ONLY_CAMPAIGN_FIREPOWER",
                    "command": (
                        "PYTHONPATH=src python3 scripts/oanda_universal_rotation_miner.py "
                        "--history-root logs/replay/oanda_history --history-glob '*_M5_BA_*.jsonl' "
                        f"--pairs {pair_arg}"
                    ),
                    "requires_explicit_operator_approval": False,
                    "reason": (
                        "rerun the multi-month M5 bid/ask candle miner for the audit-only pairs so "
                        "improved range/failed-break/pullback vehicles are tested before reinsertion"
                    ),
                }
            )
            actions.append(
                {
                    "code": "PACKAGE_OANDA_AUDIT_ONLY_FIREPOWER_RULES_AFTER_REVIEW",
                    "command": "PYTHONPATH=src python3 scripts/package_oanda_universal_rotation_rules.py",
                    "requires_explicit_operator_approval": False,
                    "reason": (
                        "packages mined OANDA replay rows into the tracked runtime rule artifact; "
                        "Codex may run this after reviewing mining evidence, then test, commit, and sync "
                        "before live runtime uses the new evidence"
                    ),
                }
            )
            actions.append(
                {
                    "code": "RERUN_INTENTS_AFTER_OANDA_AUDIT_ONLY_REPLAY",
                    "command": "PYTHONPATH=src python3 -m quant_rabbit.cli cycle-refresh --daily-risk-pct 10",
                    "requires_explicit_operator_approval": False,
                    "reason": (
                        "rerun the full evidence/intents/acceptance loop after replay mining so improved "
                        "evidence is revalidated instead of leaving the old blocked packet in place"
                    ),
                }
            )
    target_firepower = acceptance.get("target_firepower") if isinstance(acceptance.get("target_firepower"), dict) else {}
    if target_firepower.get("minimum_5pct_estimated_reachable"):
        actions.append(
            {
                "code": "WORK_TARGET_FIREPOWER_BLOCKERS",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "firepower audit estimates enough turnover for the 5% floor, but live permission "
                    "still depends on clearing acceptance, guardian, and lane blockers"
                ),
            }
        )
    if status == STATUS_READY:
        actions.append(
            {
                "code": "RUN_NEXT_TRADER_CYCLE",
                "command": "scripts/run-autotrade-live.sh --send",
                "requires_explicit_operator_approval": True,
                "reason": "support gates are green; live send still requires normal trader/gateway validation",
            }
        )
    # Keep blocker-specific actions unique while preserving order.
    seen = set()
    unique = []
    for action in actions:
        code = action["code"]
        if code in seen:
            continue
        seen.add(code)
        unique.append(action)
    return unique


def _guardian_receipt_consumption_next_action(
    qr_trader_run_watchdog: dict[str, Any],
    guardian_receipt_consumption: dict[str, Any],
) -> str:
    issues = (
        qr_trader_run_watchdog.get("guardian_receipt_issues")
        if isinstance(qr_trader_run_watchdog.get("guardian_receipt_issues"), list)
        else []
    )
    if not issues:
        return "No unresolved guardian receipt issue is currently blocking normal routing."
    classifications = (
        guardian_receipt_consumption.get("classifications")
        if isinstance(guardian_receipt_consumption.get("classifications"), list)
        else []
    )
    unclassified_current_issues = _current_guardian_receipt_unclassified_issues(
        qr_trader_run_watchdog,
        guardian_receipt_consumption,
    )
    if unclassified_current_issues:
        event_ids = [
            _guardian_receipt_event_id(item) or "UNKNOWN"
            for item in unclassified_current_issues[:3]
        ]
        suffix = f" Current unclassified event(s): {', '.join(event_ids)}."
        return (
            "Run the qr-trader draft/preflight path to write a durable guardian_receipt_consumption "
            "classification for the current watchdog receipt issue before any ordinary fresh entry."
            + suffix
        )
    normal_allowed = guardian_receipt_consumption.get("normal_routing_allowed")
    if normal_allowed is True:
        return (
            "Guardian receipt issues have a normal-routing-allowed classification; "
            "continue normal verifier/gateway flow."
        )
    if not classifications:
        return (
            "Run the qr-trader draft/preflight path to write a durable guardian_receipt_consumption "
            "classification before any ordinary fresh entry."
        )
    labels = [
        str(item.get("classification") or "")
        for item in classifications
        if isinstance(item, dict) and str(item.get("classification") or "")
    ]
    if "NEEDS_OPERATOR_REVIEW" in labels:
        return (
            "Operator/trader must review NEEDS_OPERATOR_REVIEW guardian receipt rows and record a precise "
            "non-action reason or use an approved protection/risk-reduction path; ordinary fresh entries remain blocked."
        )
    return (
        "Guardian receipt consumption does not allow normal routing; inspect the classification report before any "
        "ordinary fresh entry and keep protection/reporting paths separate."
    )


def _current_guardian_receipt_unclassified_issues(
    qr_trader_run_watchdog: dict[str, Any],
    guardian_receipt_consumption: dict[str, Any],
) -> list[dict[str, Any]]:
    issues = (
        qr_trader_run_watchdog.get("guardian_receipt_issues")
        if isinstance(qr_trader_run_watchdog.get("guardian_receipt_issues"), list)
        else []
    )
    if not issues:
        return []
    classifications = (
        guardian_receipt_consumption.get("classifications")
        if isinstance(guardian_receipt_consumption.get("classifications"), list)
        else []
    )
    classified_keys = {
        _guardian_receipt_match_key(item)
        for item in classifications
        if isinstance(item, dict)
    }
    classified_keys.discard(("", "", ""))
    missing: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        key = _guardian_receipt_match_key(issue)
        if not any(key) or key not in classified_keys:
            missing.append(issue)
    return missing


def _guardian_receipt_match_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _guardian_receipt_event_id(record),
        str(record.get("receipt_action") or record.get("action") or "").upper(),
        str(record.get("receipt_lifecycle") or "").upper(),
    )


def _guardian_receipt_event_id(record: dict[str, Any]) -> str:
    return str(record.get("receipt_event_id") or record.get("event_id") or "").strip()


def _frontier_blocker_waits_for_live_precision_evidence(top: dict[str, Any]) -> bool:
    code = str(top.get("code") or "")
    if code not in FORECAST_FRONTIER_BLOCKER_CODES:
        return False
    examples = (
        top.get("forecast_support_examples")
        if isinstance(top.get("forecast_support_examples"), list)
        else []
    )
    if not any(
        isinstance(item, dict) and isinstance(item.get("forecast_support"), dict)
        for item in examples
    ):
        return False
    has_wait_signal = False
    for example in examples:
        if not isinstance(example, dict) or not isinstance(example.get("forecast_support"), dict):
            continue
        support = example["forecast_support"]
        if support.get("forecast_market_support_ok") is True:
            return False
        top_signal = (
            support.get("top_unselected_signal")
            if isinstance(support.get("top_unselected_signal"), dict)
            else {}
        )
        if top_signal.get("live_precision_ok") is True and _forecast_example_has_actionable_unclear_limit_support(
            example,
            support,
        ):
            return False
        reason = str(support.get("forecast_market_support_reason") or "").lower()
        direction = str(support.get("forecast_direction") or "").upper()
        if top_signal.get("live_precision_ok") is False:
            has_wait_signal = True
        if top_signal.get("live_precision_ok") is True:
            has_wait_signal = True
        if (
            "unselected" in reason
            or "no executable direction" in reason
            or "no current projection" in reason
            or "audited support floors" in reason
        ):
            has_wait_signal = True
        if direction in {"UNCLEAR", "RANGE", ""}:
            has_wait_signal = True
    return has_wait_signal


def _forecast_example_has_actionable_unclear_limit_support(
    example: dict[str, Any],
    support: dict[str, Any],
) -> bool:
    direction = str(support.get("forecast_direction") or "").upper()
    if direction != "UNCLEAR":
        return False
    if str(example.get("order_type") or "").upper() != "LIMIT":
        return False
    side = str(example.get("side") or "").upper()
    expected_signal_direction = "UP" if side == "LONG" else "DOWN" if side == "SHORT" else ""
    if not expected_signal_direction:
        return False
    opposite_signal_direction = "DOWN" if expected_signal_direction == "UP" else "UP"
    signals = (
        support.get("unselected_signal_examples")
        if isinstance(support.get("unselected_signal_examples"), list)
        else []
    )
    top_signal = support.get("top_unselected_signal")
    if not signals and isinstance(top_signal, dict):
        signals = [top_signal]
    has_same_side_signal = False
    for signal in signals:
        if not isinstance(signal, dict) or not _forecast_frontier_signal_strong_enough(signal):
            continue
        signal_direction = str(signal.get("direction") or "").upper()
        if signal_direction == opposite_signal_direction:
            return False
        if signal_direction == expected_signal_direction:
            has_same_side_signal = True
    return has_same_side_signal


def _forecast_frontier_signal_strong_enough(signal: dict[str, Any]) -> bool:
    samples = _int_like(
        signal.get("samples")
        if signal.get("samples") is not None
        else signal.get("calibration_samples")
        if signal.get("calibration_samples") is not None
        else signal.get("economic_samples")
    )
    return (
        signal.get("live_precision_ok") is True
        and _float(signal.get("confidence")) >= FORECAST_MARKET_SUPPORT_MIN_SIGNAL_CONFIDENCE
        and _float(signal.get("hit_rate")) >= FORECAST_MARKET_SUPPORT_MIN_DIRECTIONAL_HIT_RATE
        and samples >= FORECAST_MARKET_SUPPORT_MIN_SAMPLES
    )


def _oanda_audit_only_local_tp_candidates(entry: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = (
        entry.get("oanda_audit_only_local_tp_proof_required")
        if isinstance(entry.get("oanda_audit_only_local_tp_proof_required"), list)
        else []
    )
    rows = [item for item in candidates if isinstance(item, dict)]
    rows.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    return rows


def _build_repair_requests(
    *,
    guardian: dict[str, Any],
    profit_capture: dict[str, Any],
    entry: dict[str, Any],
    acceptance: dict[str, Any],
    p0_findings: list[dict[str, Any]] | None = None,
    broker: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
    oanda_history_coverage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    broker = broker if isinstance(broker, dict) else {}
    target = target if isinstance(target, dict) else {}
    p0_findings = p0_findings if isinstance(p0_findings, list) else []
    repair_plan = acceptance.get("repair_plan") if isinstance(acceptance.get("repair_plan"), dict) else {}
    raw_items = repair_plan.get("items") if isinstance(repair_plan.get("items"), list) else []
    items = [item for item in raw_items if isinstance(item, dict)]
    item_by_code = {str(item.get("code")): item for item in items if item.get("code")}
    raw_evidence_items = (
        repair_plan.get("evidence_collection_items")
        if isinstance(repair_plan.get("evidence_collection_items"), list)
        else []
    )
    evidence_items = [item for item in raw_evidence_items if isinstance(item, dict)]
    requests: list[dict[str, Any]] = []

    unknown_positions = (
        broker.get("unknown_owner_position_examples")
        if isinstance(broker.get("unknown_owner_position_examples"), list)
        else []
    )
    if int(_float(broker.get("unknown_owner_positions"))) > 0:
        requests.append(
            _repair_request(
                code="REVIEW_UNKNOWN_OWNER_EXPOSURE",
                priority="P1",
                status="OPERATOR_REVIEW_RECOMMENDED",
                source_findings=[
                    "BROKER_TRUTH_UNKNOWN_OWNER_EXPOSURE",
                    *[
                        str(item.get("code"))
                        for item in entry.get("repair_frontier_remaining_blockers", [])
                        if isinstance(item, dict)
                        and item.get("code") in FRONTIER_MARGIN_CAPACITY_BLOCKER_CODES
                    ],
                ],
                problem=(
                    "Broker truth includes manual/tagless exposure. It is TP-managed only by contract "
                    "and is not a fresh-entry send gate by itself; broker-truth margin capacity still "
                    "flows through normal intent and gateway risk checks."
                ),
                why_now=(
                    "Daily 5% firepower estimates can be mathematically reachable while broker margin "
                    "is consumed by a tagless live position. Codex should surface this as read-only "
                    "operator context without treating review itself as approval-bound code work or "
                    "loosening margin, min-lot, or forecast gates."
                ),
                evidence_summary={
                    "unknown_owner_positions": broker.get("unknown_owner_positions"),
                    "examples": unknown_positions[:5],
                    "margin_available_jpy": broker.get("margin_available_jpy"),
                    "nav_jpy": broker.get("nav_jpy"),
                },
                clearance_conditions=[
                    "If the operator changes this exposure externally or adopts it into a supported owner path, rerun broker-snapshot, generate-intents, trader-support-bot, and profitability-acceptance before treating margin blockers as current.",
                    "Do not close, stop-loss, cancel, or otherwise mutate this manual/tagless exposure from the repair queue; only existing approved gateways or explicit operator action may change live broker state.",
                ],
                verification_commands=[
                    "PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json",
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                ],
                suggested_files=[
                    "src/quant_rabbit/broker/client.py",
                    "src/quant_rabbit/strategy/position_manager.py",
                    "src/quant_rabbit/trader_support_bot.py",
                    "tests/test_trader_support_bot.py",
                ],
                required_tests=[
                    "Manual/tagless exposure is surfaced as read-only operator context without becoming an approval-bound Codex repair.",
                    "Support still keeps normal fresh-entry gates unchanged; unknown exposure does not become an implicit live permission, forced close, or fresh-entry blocker.",
                    "Margin-thin guidance continues to rely on broker-truth margin, min-lot, and risk caps instead of loosening them.",
                ],
                requires_explicit_operator_approval=False,
            )
        )

    pending_cancel_finding = next(
        (
            item
            for item in p0_findings
            if str(item.get("code") or "") == PENDING_CANCEL_REVIEW_CODE
        ),
        None,
    )
    if pending_cancel_finding is not None:
        evidence = (
            pending_cancel_finding.get("evidence")
            if isinstance(pending_cancel_finding.get("evidence"), dict)
            else {}
        )
        cancel_order_ids = [
            str(item)
            for item in evidence.get("cancel_review_order_ids", []) or []
            if str(item)
        ]
        requests.append(
            _repair_request(
                code=PENDING_CANCEL_REVIEW_CODE,
                priority="P0",
                status=PENDING_CANCEL_RECEIPT_WAIT_STATUS,
                source_findings=[PENDING_CANCEL_REVIEW_CODE],
                problem=(
                    "Self-improvement found trader-owned pending entry orders that no longer have "
                    "a current LIVE_READY matching candidate."
                ),
                why_now=(
                    "A stale or lower-quality pending entry can keep campaign exposure ambiguous and "
                    "consume margin. The repair loop must surface the exact order ids for a fresh "
                    "TRADE-with-cancel_order_ids or CANCEL_PENDING receipt instead of hiding them behind "
                    "a generic self-improvement P0 count."
                ),
                evidence_summary={
                    "cancel_review_order_ids": list(dict.fromkeys(cancel_order_ids)),
                    "groups": evidence.get("groups") or [],
                    "orders": evidence.get("orders") or [],
                    "source_next_action": pending_cancel_finding.get("next_action"),
                },
                clearance_conditions=[
                    (
                        "Refresh broker snapshot, order_intents, and trader-prompt-route. If the named "
                        "current trader-owned pending ids still exist and no current LIVE_READY "
                        "replacement basket exists, write a fresh accepted CANCEL_PENDING receipt naming "
                        "only those ids."
                    ),
                    (
                        "If a current LIVE_READY replacement basket exists, write a fresh accepted TRADE "
                        "receipt with cancel_order_ids for stale or lower-priority pending entries; gateway "
                        "validation must still re-check broker truth before any cancel or send."
                    ),
                    (
                        "Do not direct-cancel from support/orchestrator/Codex repair work. Clearance is a "
                        "new broker snapshot without the ids, or a consumed gateway receipt that canceled "
                        "or replaced them through the approved path."
                    ),
                ],
                verification_commands=[
                    "PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json",
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-prompt-route",
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator",
                ],
                suggested_files=[
                    "docs/SKILL_trader.md",
                    "docs/trader_prompts/35_position_management.md",
                    "docs/trader_prompts/40_verify_execute.md",
                    "src/quant_rabbit/gpt_trader.py",
                    "src/quant_rabbit/automation.py",
                    "tests/test_gpt_trader.py",
                    "tests/test_autotrade_cycle.py",
                ],
                required_tests=[
                    "Support bot emits a non-actionable repair request with the exact pending order ids when self-improvement raises PENDING_ENTRY_CANCEL_REVIEW_REQUIRED.",
                    "Repair orchestrator keeps the request in the waiting queue and does not mark it as Codex implementation work or live permission.",
                    "Gateway cancellation still requires an accepted CANCEL_PENDING receipt or accepted TRADE receipt with verified cancel_order_ids.",
                ],
            )
        )

    inversion_counterfactuals = (
        broker.get("directional_inversion_counterfactuals")
        if isinstance(broker.get("directional_inversion_counterfactuals"), list)
        else []
    )
    target_clearing_inversions = [
        item
        for item in inversion_counterfactuals
        if isinstance(item, dict) and _directional_inversion_counterfactual_has_viable_replay_path(item)
    ]
    if target_clearing_inversions:
        pairs = sorted({str(item.get("pair")) for item in target_clearing_inversions if item.get("pair")})
        has_repeated_inversion_evidence = any(
            bool(item.get("has_repeated_spread_included_inversion_evidence"))
            for item in target_clearing_inversions
        )
        requests.append(
            _repair_request(
                code=DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST,
                priority="P0",
                status=(
                    "READY_FOR_CODE_OR_EVIDENCE_REPAIR"
                    if has_repeated_inversion_evidence
                    else DIRECTIONAL_INVERSION_REPLAY_WAIT_STATUS
                ),
                source_findings=[
                    "BROKER_TRUTH_OPPOSITE_SIDE_WOULD_CLEAR_MINIMUM_5PCT",
                    (
                        "DIRECTIONAL_INVERSION_REPLAY_EVIDENCE_PRESENT"
                        if has_repeated_inversion_evidence
                        else "DIRECTIONAL_INVERSION_REPLAY_EVIDENCE_MISSING"
                    ),
                    *[
                        str(code)
                        for blocker in entry.get("repair_frontier_remaining_blockers", [])
                        if isinstance(blocker, dict)
                        for code in [blocker.get("code")]
                        if code
                    ],
                ],
                problem=(
                    "A current broker-truth losing position has an opposite-side gross P/L "
                    "counterfactual large enough to clear the daily 5% minimum target."
                ),
                why_now=(
                    "This is the direct version of the operator complaint: repeating TP/leak summaries "
                    "misses the simple directional failure. Codex must test whether the system should "
                    "have selected, inverted, or at least elevated the opposite-side forecast lane before "
                    "adding unrelated entry frequency."
                ),
                evidence_summary={
                    "target": {
                        "campaign_day_jst": target.get("campaign_day_jst"),
                        "minimum_target_jpy": target.get("minimum_target_jpy"),
                        "remaining_minimum_jpy": target.get("remaining_minimum_jpy"),
                        "remaining_target_jpy": target.get("remaining_target_jpy"),
                    },
                    "counterfactuals": target_clearing_inversions[:8],
                    "pairs": pairs,
                    "repair_frontier_remaining_blockers": entry.get("repair_frontier_remaining_blockers", [])[:8],
                },
                clearance_conditions=[
                    (
                        "Build a read-only inversion audit over forecast_history/projection_ledger and "
                        "spread-included OANDA candles; only repeated pair/side/method evidence may "
                        "promote an inversion rule."
                    ),
                    (
                        "The current invalid chase and low-reward shapes must remain blocked; a "
                        "directional inversion repair cannot bypass RiskEngine, strategy profile, "
                        "margin, forecast telemetry freshness, TP proof, or gateway validation."
                    ),
                ],
                verification_commands=[
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                    (
                        "PYTHONPATH=src python3 scripts/oanda_history_replay_validate.py "
                        f"--forecast-history data/forecast_history.jsonl --pairs {','.join(pairs) if pairs else 'EUR_USD'} "
                        "--granularity S5 --auto-history-min-days 30"
                    ),
                    "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts",
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator --trader-request forecast",
                ],
                suggested_files=[
                    "src/quant_rabbit/strategy/directional_forecaster.py",
                    "src/quant_rabbit/strategy/intent_generator.py",
                    "src/quant_rabbit/forecast_precision.py",
                    "src/quant_rabbit/trader_support_bot.py",
                    "tests/test_directional_forecaster.py",
                    "tests/test_intent_generator.py",
                    "tests/test_trader_support_bot.py",
                    "tests/test_trader_repair_orchestrator.py",
                ],
                required_tests=[
                    "Regression: a one-off losing position does not blindly invert live direction without repeated audited evidence.",
                    "Positive path: broker-truth opposite-side counterfactual that clears the daily 5% minimum becomes a P0 Codex repair request.",
                    "Safety path: invalid opposite-side lanes remain blocked by forecast, chase, reward/risk, margin, TP-proof, and gateway gates until the audited escape clears.",
                ],
            )
        )

    if "LOSS_CLOSE_GATE_EVIDENCE_MISSING" in item_by_code:
        item = item_by_code["LOSS_CLOSE_GATE_EVIDENCE_MISSING"]
        evidence_summary = (
            item.get("evidence_summary")
            if isinstance(item.get("evidence_summary"), dict)
            else {}
        )
        missing_with_receipt_evidence = _int_like(
            evidence_summary.get("missing_receipt_evidence_present")
        )
        missing_without_receipt_evidence = _int_like(
            evidence_summary.get("missing_receipt_evidence_absent")
        )
        missing_breakdown_known = (
            "missing_receipt_evidence_present" in evidence_summary
            or "missing_receipt_evidence_absent" in evidence_summary
        )
        if missing_breakdown_known and missing_with_receipt_evidence <= 0 and missing_without_receipt_evidence > 0:
            requests.append(
                _repair_request(
                    code="REVIEW_CLOSE_GATE_EVIDENCE_FAILURES",
                    priority="P0",
                    status="HISTORICAL_ACCEPTANCE_WINDOW_ACTIVE",
                    source_findings=[
                        code
                        for code in (
                            "LOSS_CLOSE_GATE_EVIDENCE_MISSING",
                            "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
                        )
                        if code in item_by_code
                    ],
                    problem=(
                        "Recent loss-side GPT closes lack durable close_gate_evidence because the "
                        "accepted close receipts themselves did not contain evidence."
                    ),
                    why_now=(
                        "A code-repair loop must not synthesize evidence for historical closes. "
                        "Acceptance remains red until these missing-evidence closes age out or "
                        "future loss-side closes carry PASS evidence."
                    ),
                    evidence_summary=evidence_summary,
                    clearance_conditions=[item.get("clearance_condition")],
                    verification_commands=[item.get("verification_command")],
                    suggested_files=[
                        "src/quant_rabbit/gpt_trader.py",
                        "src/quant_rabbit/automation.py",
                        "src/quant_rabbit/verification_ledger.py",
                        "tests/test_gpt_trader.py",
                        "tests/test_autotrade_cycle.py",
                        "tests/test_verification_ledger.py",
                    ],
                    required_tests=[
                        "Regression: an accepted GPT loss-side close without close_gate_evidence is blocked before broker close.",
                        "Positive path: a future contained GPT loss-side close with PASS close_gate_evidence clears this blocker.",
                        "Loop guard: support bot labels historical receipt-missing close evidence as an acceptance-window wait, not code repair.",
                    ],
                )
            )
        else:
            requests.append(
                _repair_request(
                    code="REPAIR_CLOSE_GATE_EVIDENCE_PERSISTENCE",
                    priority="P0",
                    status="READY_FOR_CODE_REPAIR",
                    source_findings=[
                        code
                        for code in (
                            "LOSS_CLOSE_GATE_EVIDENCE_MISSING",
                            "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
                        )
                        if code in item_by_code
                    ],
                    problem=(
                        "Loss-side market closes can remain unverified because durable close_gate_evidence "
                        "is missing from the verification ledger."
                    ),
                    why_now=(
                        "Profitability acceptance will keep blocking high-turn scaling until every recent "
                        "GPT/gateway loss close has PASS close_gate_evidence or ages out."
                    ),
                    evidence_summary=evidence_summary,
                    clearance_conditions=[item.get("clearance_condition")],
                    verification_commands=[item.get("verification_command")],
                    suggested_files=[
                        "src/quant_rabbit/gpt_trader.py",
                        "src/quant_rabbit/verification_ledger.py",
                        "src/quant_rabbit/profitability_acceptance.py",
                        "tests/test_gpt_trader.py",
                        "tests/test_verification_ledger.py",
                        "tests/test_cli.py",
                    ],
                    required_tests=[
                        "Regression: a GPT loss-side close without PASS close_gate_evidence remains blocked.",
                        "Positive path: a contained GPT loss-side close with matching PASS close_gate_evidence clears the acceptance blocker.",
                        "Ledger path: close_gate_evidence is written to verification_observations for accepted and rejected CLOSE receipts.",
                    ],
                )
            )

    if "LOSS_CLOSE_GATE_EVIDENCE_NOT_PASSING" in item_by_code:
        item = item_by_code["LOSS_CLOSE_GATE_EVIDENCE_NOT_PASSING"]
        requests.append(
            _repair_request(
                code="REVIEW_CLOSE_GATE_EVIDENCE_FAILURES",
                priority="P0",
                status="HISTORICAL_ACCEPTANCE_WINDOW_ACTIVE",
                source_findings=[
                    code
                    for code in (
                        "LOSS_CLOSE_GATE_EVIDENCE_NOT_PASSING",
                        "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
                    )
                    if code in item_by_code
                ],
                problem=(
                    "Recent GPT loss-side market closes now have durable close_gate_evidence, but that "
                    "evidence is BLOCK rather than PASS."
                ),
                why_now=(
                    "A code-repair loop must not synthesize PASS for historical closes. Acceptance remains "
                    "red until the failed-evidence closes age out or future GPT closes prove Gate A/B with PASS."
                ),
                evidence_summary=item.get("evidence_summary"),
                clearance_conditions=[item.get("clearance_condition")],
                verification_commands=[item.get("verification_command")],
                suggested_files=[
                    "src/quant_rabbit/gpt_trader.py",
                    "src/quant_rabbit/verification_ledger.py",
                    "src/quant_rabbit/profitability_acceptance.py",
                    "tests/test_gpt_trader.py",
                    "tests/test_verification_ledger.py",
                    "tests/test_cli.py",
                ],
                required_tests=[
                    "Regression: historical BLOCK close_gate_evidence remains a P0 acceptance blocker.",
                    "Positive path: a future contained GPT loss-side close with PASS close_gate_evidence clears this blocker.",
                    "Loop guard: support bot does not label already-persisted BLOCK evidence as a persistence code repair.",
                ],
            )
        )

    tp_codes = [
        code
        for code in (
            "TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED",
            "TP_PROGRESS_REPLAY_REPAIR_UNPROVED",
            "TP_PROGRESS_REPAIR_REPLAY_CONTRACT_MISSING",
        )
        if code in item_by_code
    ]
    if tp_codes:
        top_item = item_by_code[tp_codes[0]]
        tp_evidence = (
            top_item.get("evidence_summary")
            if isinstance(top_item.get("evidence_summary"), dict)
            else {}
        )
        current_guardian_inactive = _guardian_counts_as_inactive_for_support(guardian)
        current_guardian_lock_wait = _guardian_waits_for_live_runtime_lock(guardian)
        tp_replay_trigger_value = tp_evidence.get(
            "post_repair_live_evidence_loss_closes_repair_replay_triggered"
        )
        if tp_replay_trigger_value is None:
            tp_replay_trigger_value = tp_evidence.get("loss_closes_repair_replay_triggered")
        tp_has_replay_triggers = int(_float(tp_replay_trigger_value)) > 0
        tp_contract_missing = "TP_PROGRESS_REPAIR_REPLAY_CONTRACT_MISSING" in tp_codes
        tp_waits_for_operator_guardian = (
            not tp_contract_missing and current_guardian_inactive and not current_guardian_lock_wait
        )
        tp_waits_for_live_evidence = (
            not tp_contract_missing
            and tp_has_replay_triggers
            and (
                "TP_PROGRESS_REPLAY_REPAIR_UNPROVED" in tp_codes
                or (
                    "TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED" in tp_codes
                    and not current_guardian_inactive
                )
                or current_guardian_lock_wait
            )
        )
        verification_commands = [
            item_by_code[code].get("verification_command")
            for code in tp_codes
            if isinstance(item_by_code.get(code), dict)
        ]
        if tp_waits_for_live_evidence:
            verification_commands = [
                MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND,
                "PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance",
            ]
        tp_request_status = (
            TP_PROGRESS_GUARDIAN_WAIT_STATUS
            if tp_waits_for_operator_guardian
            else POSITION_GUARDIAN_LOCK_WAIT_STATUS
            if current_guardian_lock_wait
            else TP_PROGRESS_LIVE_EVIDENCE_WAIT_STATUS
            if tp_waits_for_live_evidence
            else "READY_FOR_CODE_REPAIR"
        )
        requests.append(
            _repair_request(
                code="REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                priority="P0",
                status=tp_request_status,
                source_findings=tp_codes,
                problem=(
                    "Historical losing closes still show executable TP-progress profit that was not "
                    "captured before the close turned red."
                ),
                why_now=(
                    "Adding entry frequency before this capture path is proved clean repeats the "
                    "known TAKE_PROFIT_ORDER plus / MARKET_ORDER_TRADE_CLOSE minus leak."
                ),
                evidence_summary={
                    **tp_evidence,
                    "current_guardian_active": guardian.get("active"),
                    "current_guardian_active_source": guardian.get("active_source"),
                    "current_guardian_heartbeat_fresh": guardian.get("heartbeat_fresh"),
                    "current_guardian_live_runtime_lock_active": guardian.get("live_runtime_lock_active"),
                    "current_guardian_live_runtime_lock_command": guardian.get("live_runtime_lock_command"),
                    "current_guardian_live_runtime_lock_age_seconds": guardian.get(
                        "live_runtime_lock_age_seconds"
                    ),
                    "current_guardian_deferred_by_live_runtime_lock": current_guardian_lock_wait,
                    "guardian_inactive_evidence_status": (
                        "STALE_CURRENT_GUARDIAN_ACTIVE"
                        if tp_evidence.get("guardian_profit_capture_inactive")
                        and guardian.get("active")
                        else "CURRENT_GUARDIAN_LOCK_BUSY"
                        if tp_evidence.get("guardian_profit_capture_inactive")
                        and current_guardian_lock_wait
                        else None
                    ),
                },
                clearance_conditions=[
                    item_by_code[code].get("clearance_condition")
                    for code in tp_codes
                    if isinstance(item_by_code.get(code), dict)
                ],
                verification_commands=verification_commands,
                suggested_files=[
                    "src/quant_rabbit/profit_capture_bot.py",
                    "src/quant_rabbit/execution_timing_audit.py",
                    "src/quant_rabbit/strategy/position_manager.py",
                    "src/quant_rabbit/strategy/position_protection_gateway.py",
                    "tests/test_execution_timing_audit.py",
                    "tests/test_trader_support_bot.py",
                ],
                required_tests=[
                    "Regression: TP-progress winner that later closes red is surfaced as repair_replay_triggered.",
                    "Positive path: production-gate replay reports zero loss_closes_repair_replay_triggered after the capture repair.",
                    "Safety path: support/profit-capture bots remain read-only and do not close positions directly.",
                ],
                requires_explicit_operator_approval=tp_waits_for_operator_guardian,
            )
        )

    if "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE" in item_by_code:
        item = item_by_code["MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"]
        residual_block_status = _month_scale_residual_block_status(
            item.get("evidence_summary"),
            entry,
        )
        residual_request_status = (
            "RESIDUAL_GROUPS_ALREADY_BLOCKED_WAITING_FOR_REPLAY"
            if residual_block_status["current_residual_blocked_intents_count"]
            else "READY_FOR_CODE_REPAIR"
        )
        evidence_summary = item.get("evidence_summary")
        if not isinstance(evidence_summary, dict):
            evidence_summary = {}
        evidence_summary = {
            **evidence_summary,
            "current_residual_block_status": residual_block_status,
        }
        requests.append(
            _repair_request(
                code="REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY",
                priority="P0",
                status=residual_request_status,
                source_findings=["MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"],
                problem=(
                    "Month-scale replay remains net negative after TP-progress repair; residual "
                    "pair/side/method groups must be blocked, reclassified, or given a tested escape."
                ),
                why_now=(
                    "The daily target cannot be scaled by frequency while the 744h production-gate replay "
                    "is still negative for the active residual groups."
                ),
                evidence_summary=evidence_summary,
                clearance_conditions=[item.get("clearance_condition")],
                verification_commands=[item.get("verification_command")],
                suggested_files=[
                    "src/quant_rabbit/execution_timing_audit.py",
                    "src/quant_rabbit/profitability_acceptance.py",
                    "src/quant_rabbit/strategy/intent_generator.py",
                    "tests/test_execution_timing_audit.py",
                    "tests/test_profitability_acceptance_replay_repair.py",
                    "tests/test_intent_generator.py",
                ],
                required_tests=[
                    "Regression: a matching residual pair/side/method cannot become LIVE_READY while month replay is negative.",
                    "Positive path: a non-matching pair/side/method with current evidence is not blocked by the residual group.",
                    "Metric path: TP-progress residuals and entry-quality residuals stay split in acceptance evidence.",
                ],
            )
        )

    if _guardian_counts_as_inactive_for_support(guardian):
        requests.append(
            _repair_request(
                code="RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT",
                priority="P0",
                status="OPERATOR_APPROVAL_REQUIRED",
                source_findings=["POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE"],
                problem=(
                    "Fresh entries are blocked because the fast position guardian is not proven active "
                    "with a fresh heartbeat."
                ),
                why_now=(
                    "The guardian is the between-cycle support path that should bank TP-progress profit "
                    "before it reverses into a later market-close loss."
                ),
                evidence_summary={
                    "active": guardian.get("active"),
                    "active_source": guardian.get("active_source"),
                    "launchd_loaded": guardian.get("launchd_loaded"),
                    "heartbeat_fresh": guardian.get("heartbeat_fresh"),
                    "heartbeat_age_seconds": guardian.get("heartbeat_age_seconds"),
                    "live_runtime_lock_active": guardian.get("live_runtime_lock_active"),
                    "live_runtime_lock_command": guardian.get("live_runtime_lock_command"),
                    "live_runtime_lock_age_seconds": guardian.get("live_runtime_lock_age_seconds"),
                },
                clearance_conditions=[
                    "scripts/install-position-guardian.sh --check passes, then an operator explicitly approves load/reload and heartbeat becomes fresh."
                ],
                verification_commands=[
                    "scripts/install-position-guardian.sh --check",
                    "scripts/install-position-guardian.sh --status",
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                ],
                suggested_files=[
                    "scripts/install-position-guardian.sh",
                    "scripts/run-position-guardian-live.sh",
                    "tests/test_position_guardian_install.py",
                    "tests/test_trader_support_bot.py",
                ],
                required_tests=[
                    "Preflight blocks unsynced live runtime before launchd can be loaded.",
                    "Support bot shows guardian recovery candidates without loading launchd itself.",
                    "Guardian stale heartbeat without an active live runtime lock requires explicit operator approval before load/reload.",
                ],
                requires_explicit_operator_approval=True,
            )
        )

    oanda_local_tp_candidates = _oanda_audit_only_local_tp_candidates(entry)
    if oanda_local_tp_candidates:
        pairs = sorted({str(item.get("pair")) for item in oanda_local_tp_candidates if item.get("pair")})
        pair_arg = ",".join(pairs) if pairs else "EUR_USD"
        history_coverage = _usable_oanda_history_coverage(oanda_history_coverage, pairs)
        fetch_commands = (
            history_coverage.get("fetch_commands")
            if isinstance(history_coverage.get("fetch_commands"), list)
            else []
        )
        source_findings = [
            OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED,
            *[
                str(code)
                for item in oanda_local_tp_candidates
                for code in item.get("remaining_blocker_codes_after_guardian", [])
                if code
            ],
        ]
        target_firepower = (
            acceptance.get("target_firepower")
            if isinstance(acceptance.get("target_firepower"), dict)
            else {}
        )
        history_complete = _oanda_history_coverage_is_complete(history_coverage)
        replay_can_clear = _oanda_audit_only_candidates_have_clearable_replay(oanda_local_tp_candidates)
        replay_loop_exhausted = history_complete and not replay_can_clear
        request_status = (
            OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_UNPROVED_STATUS
            if replay_loop_exhausted
            else "READY_FOR_READ_ONLY_EVIDENCE_COLLECTION"
        )
        request_why_now = (
            "The missing OANDA candle-truth collection has already been completed, and the "
            "latest replay still cannot clear local TP proof for these audit-only candidates. "
            "Repeating validate/mine/package with the same inputs is a loop bug; wait for new "
            "TAKE_PROFIT_ORDER receipts, new forecast/candle evidence, or exact HARVEST vehicle "
            "promotion before making this Codex-actionable again."
            if replay_loop_exhausted
            else (
                "This turns the loop from repeating forecast blockers into a concrete precision "
                "improvement path: fetch spread-included bid/ask truth, validate forecast_history, "
                "mine the exact pair/side/method vehicles, package reviewed rules, then require "
                "either local TAKE_PROFIT_ORDER proof or current-risk / normal-cap 5% firepower "
                "promotion before live permission."
            )
        )
        request_verification_commands = (
            [
                "PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance",
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator --trader-request forecast",
            ]
            if replay_loop_exhausted
            else [
                *fetch_commands,
                (
                    "PYTHONPATH=src python3 scripts/oanda_history_replay_validate.py "
                    "--history-dir logs/replay/oanda_history --granularity S5"
                ),
                (
                    "PYTHONPATH=src python3 scripts/oanda_universal_rotation_miner.py "
                    "--history-root logs/replay/oanda_history --history-glob '*_M5_BA_*.jsonl' "
                    f"--pairs {pair_arg}"
                ),
                "PYTHONPATH=src python3 scripts/package_oanda_universal_rotation_rules.py",
                "PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance",
                "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts",
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator --trader-request forecast",
            ]
        )
        requests.append(
            _repair_request(
                code=OANDA_AUDIT_ONLY_LOCAL_TP_EDGE_REQUEST,
                priority="P1",
                status=request_status,
                source_findings=list(dict.fromkeys(source_findings)),
                problem=(
                    "OANDA audit-only forecast/rotation candidates may help the 5% target, but they "
                    "are not local TP-proven or live-grade replay edges yet."
                ),
                why_now=request_why_now,
                evidence_summary={
                    "candidate_count": len(oanda_local_tp_candidates),
                    "pairs": pairs,
                    "top_candidates": oanda_local_tp_candidates[:8],
                    "top_reward_jpy": _round_optional(
                        max((_float(item.get("reward_jpy")) for item in oanda_local_tp_candidates), default=0.0),
                        3,
                    ),
                    "target_firepower": target_firepower,
                    "history_coverage": history_coverage,
                    "local_tp_proof_required": True,
                    "history_complete": history_complete,
                    "historical_replay_can_clear_local_tp_proof": replay_can_clear,
                    "read_only_replay_loop_exhausted": replay_loop_exhausted,
                    "next_evidence_required": (
                        "new local TAKE_PROFIT_ORDER receipts, new forecast/candle evidence, "
                        "or exact OANDA HARVEST vehicle promotion to current-risk / normal-cap "
                        "5% live-grade firepower"
                        if replay_loop_exhausted
                        else None
                    ),
                },
                clearance_conditions=[
                    (
                        "For each promoted candidate, exact pair/side/method TAKE_PROFIT_ORDER local "
                        "receipts must show positive expectancy, zero TP losses, and positive "
                        "Wilson-stressed expectancy, or the exact OANDA HARVEST vehicle must clear "
                        "current-risk / normal-cap 5% firepower scaling. Replay that does not clear "
                        "that live-grade test remains read-only."
                    ),
                    (
                        "Do not rerun the same OANDA validate/mine/package loop while local S5/M5 "
                        "history is complete and read-only replay cannot clear the proof; wait for "
                        "new local TP receipts, new forecast/candle evidence, or an exact HARVEST "
                        "vehicle promotion before reclassifying this request as Codex-actionable."
                        if replay_loop_exhausted
                        else (
                            "After packaging reviewed replay rules, rerun profitability-acceptance, "
                            "generate-intents, trader-support-bot, and trader-repair-orchestrator; "
                            "the old forecast blocker packet must not be reused."
                        )
                    ),
                ],
                verification_commands=request_verification_commands,
                suggested_files=[
                    "scripts/oanda_history_fetch.py",
                    "scripts/oanda_history_replay_validate.py",
                    "scripts/oanda_universal_rotation_miner.py",
                    "scripts/package_oanda_universal_rotation_rules.py",
                    "src/quant_rabbit/forecast_precision.py",
                    "src/quant_rabbit/strategy/intent_generator.py",
                    "src/quant_rabbit/trader_support_bot.py",
                    "tests/test_trader_support_bot.py",
                    "tests/test_trader_repair_orchestrator.py",
                    "tests/test_oanda_universal_rotation_miner.py",
                ],
                required_tests=[
                    "Regression: OANDA audit-only candidates remain blocked from live permission without exact local TP proof or live-grade current-risk firepower.",
                    "Positive path: support/orchestrator emits a Codex-readable precision work order with fetch, replay, mining, packaging, and refresh commands.",
                    "Safety path: read-only replay evidence does not send orders, close positions, mutate launchd, or bypass forecast/risk gates.",
                ],
            )
        )

    if evidence_items:
        first = evidence_items[0]
        source_codes = [str(item.get("code")) for item in evidence_items if item.get("code")]
        all_currency_sample_thin = "BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN" in source_codes
        summary = first.get("evidence_summary") if isinstance(first.get("evidence_summary"), dict) else {}
        price_truth_fetch_required = _bidask_summary_requires_price_truth_fetch(summary)
        requests.append(
            _repair_request(
                code="COLLECT_BIDASK_REPLAY_EVIDENCE",
                priority=str(first.get("priority") or "P1"),
                status=(
                    "READY_FOR_READ_ONLY_EVIDENCE_COLLECTION"
                    if price_truth_fetch_required
                    else BIDASK_REPLAY_WAIT_STATUS
                ),
                source_findings=[str(item.get("code")) for item in evidence_items if item.get("code")],
                problem=(
                    "Bid/ask replay support is partially missing OANDA price truth, so it cannot be "
                    "counted as live-grade high-turn firepower."
                    if price_truth_fetch_required
                    else "Bid/ask replay has OANDA price truth for loaded samples, but all-currency "
                    "pair-direction sample coverage is still too thin to claim high-turn readiness."
                    if all_currency_sample_thin
                    else "Bid/ask replay price truth is complete, but support remains rank-only because "
                    "forecast sample coverage or daily stability is not yet strong enough."
                ),
                why_now=(
                    "Historical replay may expose usable short-horizon edges, but only after missing "
                    "spread-included OANDA BA candles are fetched and replayed."
                    if price_truth_fetch_required
                    else "Fetching the same candle windows again cannot create the missing pair-direction "
                    "forecast samples; collect more forecast_history coverage, then rerun the read-only "
                    "bid/ask replay validator."
                    if all_currency_sample_thin
                    else "Repeating the same candle fetch cannot create more forecast_history samples or "
                    "repair daily concentration; wait for new forecast evidence before treating these "
                    "rank-only rules as operational firepower."
                ),
                evidence_summary=summary,
                clearance_conditions=[first.get("clearance_condition")],
                verification_commands=[
                    summary.get("history_fetch_command") if price_truth_fetch_required else None,
                    first.get("verification_command"),
                ],
                suggested_files=[
                    "scripts/oanda_history_fetch.py",
                    "scripts/oanda_history_replay_validate.py",
                    "scripts/oanda_universal_rotation_miner.py",
                    "src/quant_rabbit/profitability_acceptance.py",
                    "tests/test_trader_support_bot.py",
                ],
                required_tests=[
                    "Read-only fetch/validation commands are surfaced without granting live permission.",
                    "Rank-only replay rules cannot clear profitability acceptance until daily-stable requirements pass.",
                ],
            )
        )

    frontier_blockers = (
        entry.get("repair_frontier_remaining_blockers")
        if isinstance(entry.get("repair_frontier_remaining_blockers"), list)
        else []
    )
    if frontier_blockers:
        top = frontier_blockers[0] if isinstance(frontier_blockers[0], dict) else {}
        code = str(top.get("code") or "UNKNOWN_REPAIR_FRONTIER_BLOCKER")
        artifact_freshness = (
            entry.get("artifact_freshness") if isinstance(entry.get("artifact_freshness"), dict) else {}
        )
        waits_for_artifact_refresh = bool(
            artifact_freshness.get("order_intents_stale_against_broker_snapshot")
        )
        waits_for_quote_refresh = _frontier_blocker_waits_for_quote_refresh(top)
        waits_for_forecast_evidence = _frontier_blocker_waits_for_live_precision_evidence(top)
        waits_for_strategy_profile_evidence = _frontier_blocker_waits_for_strategy_profile_evidence(top)
        waits_for_margin_capacity = _frontier_blocker_waits_for_margin_capacity(top)
        protective_guardrail_active = _frontier_blocker_is_protective_guardrail(top)
        frontier_evidence_summary = top
        source_findings = [code]
        if waits_for_artifact_refresh:
            frontier_status = ORDER_INTENTS_ARTIFACT_REFRESH_WAIT_STATUS
            frontier_problem = (
                "Repair-frontier lanes were generated before the current broker snapshot, so the "
                "top blocker ranking may be artifact-stale."
            )
            frontier_why_now = (
                "Broker truth changed after order_intents was generated; treating the stale blocker "
                "as causal can send the repair loop into margin or gate work that may no longer apply."
            )
            frontier_clearance = [
                "Regenerate order_intents from the current broker_snapshot before selecting frontier repair work.",
                "After regeneration, rerun trader-support-bot and trader-repair-orchestrator from the same evidence packet.",
            ]
            frontier_verification_commands = list(
                artifact_freshness.get("refresh_commands")
                or [
                    "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts",
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                ]
            )
            frontier_evidence_summary = {
                **top,
                "artifact_freshness": artifact_freshness,
            }
            source_findings = ["ORDER_INTENTS_STALE_AGAINST_BROKER_SNAPSHOT", code]
        elif waits_for_quote_refresh:
            frontier_status = FRONTIER_QUOTE_FRESHNESS_WAIT_STATUS
            frontier_problem = (
                "Repair-frontier lanes are blocked because the current broker quote or forecast telemetry "
                "quote is stale."
            )
            frontier_why_now = (
                "Quote freshness is runtime broker truth, not strategy code; forcing the gate through "
                "stale prices would turn a transient refresh wait into live entry risk."
            )
            frontier_clearance = [
                f"{code} clears only after broker snapshot and generated intents use a quote inside the live freshness window.",
                "Do not loosen RiskEngine quote freshness or common entry gates; refresh broker truth and regenerate intents instead.",
            ]
            frontier_verification_commands = [
                "PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json",
                "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts",
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
            ]
        elif waits_for_forecast_evidence:
            frontier_status = FORECAST_FRONTIER_EVIDENCE_WAIT_STATUS
            frontier_problem = (
                "Repair-frontier lanes are forecast-blocked because current forecasts are not executable "
                "and supporting historical projections are not live-precision proof yet."
            )
            frontier_why_now = (
                "This prevents rank-only or long-lead historical projections from being promoted into live "
                "entries before bid/ask replay or fresh live evidence proves the forecast is tradable now."
            )
            frontier_clearance = [
                f"{code} clears only after current forecasts become executable, or bid/ask replay/live evidence marks the supporting projection live_precision_ok.",
                "Until then, collect read-only OANDA bid/ask replay evidence instead of editing entry gates to force LIVE_READY.",
            ]
            frontier_verification_commands = [
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "PYTHONPATH=src python3 scripts/oanda_history_replay_validate.py --forecast-history data/forecast_history.jsonl --granularity S5 --auto-history-min-days 30",
                "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts",
            ]
        elif waits_for_strategy_profile_evidence:
            frontier_status = FRONTIER_STRATEGY_PROFILE_EVIDENCE_WAIT_STATUS
            frontier_problem = (
                "Repair-frontier lanes are strategy-profile blocked by BLOCK_UNTIL_NEW_EVIDENCE; "
                "this is evidence wait, not Codex gate-loosening work."
            )
            frontier_why_now = (
                "A BLOCK_UNTIL_NEW_EVIDENCE matrix profile means the current vehicle must prove a new "
                "risk-resized receipt or market-structure edge before it can be reopened."
            )
            frontier_clearance = [
                f"{code} clears only after the exact lane profile is no longer BLOCK_UNTIL_NEW_EVIDENCE on refreshed order_intents.",
                "Do not auto-promote strategy_profile BLOCK_UNTIL_NEW_EVIDENCE or relax common entry gates to force this repair lane live-ready.",
            ]
            frontier_verification_commands = [
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts",
            ]
        elif waits_for_margin_capacity:
            frontier_status = FRONTIER_MARGIN_CAPACITY_WAIT_STATUS
            frontier_problem = (
                "Repair-frontier lanes are blocked by the minimum production lot, current risk budget, "
                "geometry, or margin capacity, not by a missing Codex code path."
            )
            frontier_why_now = (
                "The 1000u floor prevents structurally spread-dominated micro orders; lowering it to force a "
                "repair lane live would recreate the sub-lot loss loop."
            )
            frontier_clearance = [
                f"{code} clears only when free margin, daily risk budget, and current geometry can fund at least the minimum production lot.",
                "Wait for open positions to harvest TP/free margin, reduce other broker exposure through approved paths, or regenerate intents after broker truth changes.",
                "Do not lower MIN_PRODUCTION_LOT_UNITS, bypass RiskEngine, or emit sub-1000u live receipts as a repair.",
            ]
            frontier_verification_commands = [
                "PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json",
                "PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10",
                "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts",
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
            ]
        elif protective_guardrail_active:
            frontier_status = PROTECTIVE_FRONTIER_GUARDRAIL_STATUS
            frontier_problem = (
                "Repair-frontier lanes are blocked by protective geometry or chase guards; this is not a "
                "Codex gate-loosening task."
            )
            frontier_why_now = (
                "The current example is a bad entry shape; relaxing reward/risk or chase blockers would "
                "recreate the loss loop."
            )
            frontier_clearance = [
                f"{code} remains a protective guard until a fresh lane has valid reward/risk, retest geometry, and no chase-pattern blockers.",
                "Do not edit common entry gates to force this blocked lane live-ready; generate or wait for a better-shaped lane instead.",
            ]
            frontier_verification_commands = [
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts",
            ]
        else:
            frontier_status = "READY_FOR_CODE_OR_EVIDENCE_REPAIR"
            frontier_problem = (
                "After global support gates are removed, repair-frontier lanes still have lane-local blockers."
            )
            frontier_why_now = (
                "This is the next non-guardian blocker that would keep high-turn repair baskets from becoming executable."
            )
            frontier_clearance = [
                f"{code} disappears from repair_frontier_remaining_blockers or gains a tested downgrade path."
            ]
            frontier_verification_commands = [
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts",
            ]
        requests.append(
            _repair_request(
                code="REPAIR_FRONTIER_LANE_BLOCKER",
                priority="P1",
                status=frontier_status,
                source_findings=source_findings,
                problem=frontier_problem,
                why_now=frontier_why_now,
                evidence_summary=frontier_evidence_summary,
                clearance_conditions=frontier_clearance,
                verification_commands=frontier_verification_commands,
                suggested_files=[
                    "src/quant_rabbit/strategy/intent_generator.py",
                    "scripts/oanda_history_replay_validate.py",
                    "tests/test_intent_generator.py",
                    "tests/test_trader_support_bot.py",
                ],
                required_tests=[
                    "Regression: the original invalid blocker shape remains blocked.",
                    "Positive path: a current valid repair-frontier lane is allowed or explicitly downgraded with an escape condition.",
                ],
            )
        )

    return _unique_repair_requests(requests)


def _month_scale_residual_block_status(
    evidence_summary: Any,
    entry: dict[str, Any],
) -> dict[str, Any]:
    evidence = evidence_summary if isinstance(evidence_summary, dict) else {}
    current_blocked = (
        entry.get("month_scale_residual_blocked_intents")
        if isinstance(entry.get("month_scale_residual_blocked_intents"), list)
        else []
    )
    blocked_rows = [item for item in current_blocked if isinstance(item, dict)]
    blocker_counts: Counter[str] = Counter()
    for item in blocked_rows:
        for code in item.get("blocker_codes") or []:
            if str(code).strip():
                blocker_counts[str(code)] += 1
    residual_groups = (
        evidence.get("top_repair_replay_residual_groups")
        if isinstance(evidence.get("top_repair_replay_residual_groups"), list)
        else []
    )
    group_keys = [
        _residual_group_key(group)
        for group in residual_groups
        if isinstance(group, dict) and _residual_group_key(group) is not None
    ]
    blocked_keys = {
        _residual_group_key(item)
        for item in blocked_rows
        if _residual_group_key(item) is not None
    }
    matched_keys = [key for key in group_keys if key in blocked_keys]
    return {
        "current_residual_blocked_intents_count": len(blocked_rows),
        "current_residual_blocker_counts": dict(blocker_counts),
        "current_residual_blocked_examples": blocked_rows[:5],
        "top_residual_group_count": len(group_keys),
        "top_residual_groups_with_current_blocked_intent": len(matched_keys),
        "top_residual_groups_without_current_intent": [
            "|".join(key) for key in group_keys if key not in blocked_keys
        ][:8],
        "status": (
            "CURRENT_INTENTS_BLOCK_RESIDUAL_GROUPS_WAIT_FOR_744H_REPLAY"
            if blocked_rows
            else "NO_CURRENT_RESIDUAL_BLOCKED_INTENTS"
        ),
        "clearance_condition": (
            "Rerun execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80 and "
            "profitability-acceptance; if month-scale replay is still negative after current "
            "residual blockers are active, inspect the next unblocked residual frontier instead "
            "of reimplementing the same block."
        ),
    }


def _residual_group_key(item: dict[str, Any]) -> tuple[str, str, str] | None:
    pair = str(item.get("pair") or "").upper()
    side = str(item.get("side") or "").upper()
    method = str(item.get("method") or "").upper()
    if not pair or not side or not method:
        return None
    return pair, side, method


def _repair_request(
    *,
    code: str,
    priority: str,
    status: str,
    source_findings: list[str],
    problem: str,
    why_now: str,
    evidence_summary: Any,
    clearance_conditions: list[Any],
    verification_commands: list[Any],
    suggested_files: list[str],
    required_tests: list[str],
    requires_explicit_operator_approval: bool = False,
) -> dict[str, Any]:
    return {
        "contract_version": REPAIR_REQUEST_CONTRACT_VERSION,
        "code": code,
        "priority": priority,
        "status": status,
        "source_findings": [str(item) for item in source_findings if str(item)],
        "problem": problem,
        "why_now": why_now,
        "evidence_summary": evidence_summary if isinstance(evidence_summary, dict) else {},
        "clearance_conditions": [str(item) for item in clearance_conditions if item],
        "verification_commands": list(dict.fromkeys(str(item) for item in verification_commands if item)),
        "suggested_files": suggested_files,
        "required_tests": required_tests,
        "automation_contract": {
            "codex_may_execute": REPAIR_AUTOMATION_ALLOWED_ACTIONS,
            "commit_and_live_sync_required": True,
            "quant_rabbit_code_may_call_model_api": False,
            "live_side_effects_allowed": [],
            "requires_explicit_operator_approval_for": REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS,
            "forbidden_direct_actions": REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS,
            "orders_closes_launchd_policy": (
                "Order send, order cancel, position close, and launchd load/reload must go through "
                "explicit operator approval or the existing gateway path; a repair request alone is never live permission."
            ),
        },
        "requires_explicit_operator_approval": requires_explicit_operator_approval,
        "read_only": True,
        "live_side_effects": [],
    }


def repair_requests_from_support_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild repair requests from an already-written support-bot payload.

    Older or partially refreshed support artifacts can contain embedded
    acceptance/guardian/frontier evidence while missing the top-level
    `repair_requests` array. The repair orchestrator uses this to avoid
    treating a blocked support panel as "no repair work".
    """
    if not isinstance(payload, dict):
        return []
    guardian = payload.get("guardian") if isinstance(payload.get("guardian"), dict) else {}
    profit_capture = (
        payload.get("profit_capture") if isinstance(payload.get("profit_capture"), dict) else {}
    )
    entry = (
        payload.get("entry_readiness")
        if isinstance(payload.get("entry_readiness"), dict)
        else {}
    )
    acceptance = (
        payload.get("profitability_acceptance")
        if isinstance(payload.get("profitability_acceptance"), dict)
        else {}
    )
    self_improvement = (
        payload.get("self_improvement") if isinstance(payload.get("self_improvement"), dict) else {}
    )
    p0_findings = (
        self_improvement.get("p0_findings")
        if isinstance(self_improvement.get("p0_findings"), list)
        else _p0_findings(self_improvement)
    )
    broker = payload.get("broker") if isinstance(payload.get("broker"), dict) else {}
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    oanda_history_coverage = (
        payload.get("oanda_history_coverage")
        if isinstance(payload.get("oanda_history_coverage"), dict)
        else None
    )
    if not any((guardian, profit_capture, entry, acceptance, broker, target)):
        return []
    return _build_repair_requests(
        guardian=guardian,
        profit_capture=profit_capture,
        entry=entry,
        acceptance=acceptance,
        p0_findings=p0_findings,
        broker=broker,
        target=target,
        oanda_history_coverage=oanda_history_coverage,
    )


def _unique_repair_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for request in requests:
        code = str(request.get("code") or "")
        if not code or code in seen:
            continue
        seen.add(code)
        unique.append(request)
    return unique[:12]


def _render_report(payload: dict[str, Any]) -> str:
    guardian = payload["guardian"]
    entry = payload["entry_readiness"]
    profit = payload["profit_capture"]
    watchdog = (
        payload.get("qr_trader_run_watchdog")
        if isinstance(payload.get("qr_trader_run_watchdog"), dict)
        else {}
    )
    consumption = (
        payload.get("guardian_receipt_consumption")
        if isinstance(payload.get("guardian_receipt_consumption"), dict)
        else {}
    )
    current_profit = payload["current_profit_capture"]
    broker = payload["broker"]
    target = payload["target"]
    acceptance = payload["profitability_acceptance"]
    firepower = acceptance.get("target_firepower", {})
    acceptance_repair = acceptance.get("repair_plan", {})
    acceptance_freshness = (
        acceptance.get("artifact_freshness")
        if isinstance(acceptance.get("artifact_freshness"), dict)
        else {}
    )
    lines = [
        "# Trader Support Bot Report",
        "",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        f"- Status: `{payload['status']}`",
        f"- Read only: `{payload['read_only']}`",
        f"- Live side effects: `{len(payload['live_side_effects'])}`",
        "",
        "## Support Gates",
        "",
        "| Gate | Value |",
        "|---|---|",
        f"| Fresh entry send allowed | `{payload['metrics']['send_fresh_entries_allowed']}` |",
        f"| Repair basket send allowed | `{payload['metrics']['repair_basket_send_allowed']}` |",
        f"| Guardian active | `{guardian['active']}` source=`{guardian['active_source']}` |",
        f"| Guardian heartbeat fresh | `{guardian['heartbeat_fresh']}` age=`{guardian['heartbeat_age_seconds']}`s |",
        f"| qr-trader scheduled-run watchdog | `{watchdog.get('status')}` severity=`{watchdog.get('severity')}` minutes_since=`{watchdog.get('minutes_since_last_run')}` |",
        f"| Guardian receipt consumption | `{consumption.get('status')}` normal_routing_allowed=`{consumption.get('normal_routing_allowed')}` unresolved=`{consumption.get('unresolved_issue_count')}` |",
        f"| LIVE_READY lanes | `{entry['live_ready_lanes']}` / `{entry['lanes']}` |",
        f"| Near-ready diagnostic lanes | `{entry.get('near_ready_lane_count')}` |",
        f"| Active path lane | `{entry.get('active_path', {}).get('lane_id')}` pair=`{entry.get('active_path', {}).get('pair')}` source=`{entry.get('active_path', {}).get('source')}` |",
        f"| Order intents freshness | `{entry.get('artifact_freshness', {}).get('status')}` staleness=`{entry.get('artifact_freshness', {}).get('order_intents_staleness_seconds')}`s |",
        f"| Profitability acceptance freshness | `{acceptance_freshness.get('status')}` stale_inputs=`{acceptance_freshness.get('stale_input_names')}` |",
        f"| Repair LIVE_READY lanes | `{len(entry['repair_live_ready'])}` |",
        f"| Repair lanes after guardian recovery | `{len(entry['repair_basket_guardian_recovery'])}` |",
        f"| Repair frontier lanes | `{len(entry['repair_frontier'])}` |",
        f"| RANGE forecast superseded repair lanes | `{len(entry['repair_frontier_superseded_by_range_forecast'])}` |",
        f"| OANDA audit-only local TP proof required | `{len(entry['oanda_audit_only_local_tp_proof_required'])}` |",
        f"| RANGE forecast missing counterpart lanes | `{len(entry['repair_frontier_missing_range_rotation_counterpart'])}` |",
        f"| Repair frontier clear after support | `{entry['repair_frontier_after_support_clear_lanes']}` |",
        f"| Repair frontier blocked after support | `{entry['repair_frontier_after_support_blocked_lanes']}` |",
        f"| Global unlock frontier lanes | `{len(entry['global_unlock_frontier'])}` |",
        f"| Profit-capture misses | `{profit['missed_loss_closes']}` gap=`{profit['estimated_gap_jpy']}` JPY "
        f"counterfactual_delta=`{profit['counterfactual_profit_capture_delta_jpy']}` JPY "
        f"repair_replay_triggered=`{profit['repair_replay_triggered']}` "
        f"repair_delta=`{profit['repair_replay_delta_jpy']}` JPY |",
        f"| Current profit-capture positions | bankable=`{current_profit['bankable_positions']}` watch=`{current_profit['watch_positions']}` blocked=`{current_profit['blocked_positions']}` |",
        f"| Open positions | `{broker['positions']}` unknown_owner=`{broker.get('unknown_owner_positions')}` operator_manual=`{broker.get('operator_manual_positions')}` |",
        f"| Operator manual JPY add guard | `{broker.get('operator_manual_jpy_fresh_add_block_active')}` code=`{broker.get('operator_manual_jpy_fresh_add_block_code')}` |",
        f"| Open trader positions | `{broker['trader_positions']}` upl=`{broker['trader_unrealized_pl_jpy']}` JPY |",
        f"| Directional inversion 5% counterfactuals | `{sum(1 for item in broker.get('directional_inversion_counterfactuals', []) if item.get('would_clear_minimum_5pct'))}` |",
        f"| Target remaining | `{target['remaining_target_jpy']}` JPY |",
        f"| Rolling 30d equity | raw=`{target.get('current_equity_raw')}` funding_adjusted=`{target.get('funding_adjusted_equity')}` capital_flows_30d=`{target.get('capital_flows_30d')}` JPY |",
        f"| Rolling 30d multiplier | raw=`{target.get('rolling_30d_multiplier_raw')}` funding_adjusted=`{target.get('rolling_30d_multiplier_funding_adjusted')}` |",
        f"| Remaining to 4x | raw=`{target.get('remaining_to_4x_raw')}` funding_adjusted=`{target.get('remaining_to_4x_funding_adjusted')}` JPY |",
        f"| Required return | calendar_funding_adjusted=`{target.get('required_calendar_daily_return_funding_adjusted')}` active_funding_adjusted=`{target.get('required_active_day_return_funding_adjusted')}` |",
        f"| Target basis | performance=`{target.get('performance_basis')}` sizing=`{target.get('sizing_basis')}` |",
        f"| Firepower 5% audit estimate | `{firepower.get('minimum_5pct_estimated_reachable')}` best=`{firepower.get('best_bucket')}` |",
        f"| Firepower 5% operational reachable | `{firepower.get('operational_minimum_5pct_reachable')}` blockers=`{firepower.get('operational_blocker_codes')}` |",
        "",
        "## Blockers",
        "",
    ]
    if payload["blockers"]:
        for blocker in payload["blockers"]:
            lines.append(f"- `{blocker['severity']}` `{blocker['code']}`: {blocker['message']}")
    else:
        lines.append("- none")
    near_ready = entry.get("near_ready_lanes") if isinstance(entry.get("near_ready_lanes"), list) else []
    lines.extend(["", "## Near-Ready Lanes", ""])
    if near_ready:
        shortest_path = entry.get("shortest_live_ready_path") or {}
        lines.extend(
            [
                (
                    "- Shortest path: "
                    f"`{shortest_path.get('lane_id')}` status=`{shortest_path.get('status')}` "
                    f"basis=`{shortest_path.get('selection_basis')}` "
                    f"next=`{shortest_path.get('first_next_step')}`"
                ),
                (
                    "- Shortest path remains non-executable: "
                    f"live_permission=`{shortest_path.get('live_permission')}` "
                    f"ordinary_fresh_entries_must_remain_blocked="
                    f"`{shortest_path.get('ordinary_fresh_entries_must_remain_blocked')}`"
                ),
                "",
            ]
        )
        lines.extend(
            [
                "| Lane | Pair | Side | Method | Status | Blockers | Evidence needed |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for item in near_ready[:10]:
            lines.append(
                "| "
                f"`{item.get('lane_id')}` | `{item.get('pair')}` | `{item.get('side')}` | "
                f"`{item.get('method')}` | `{item.get('status')}` | "
                f"`{', '.join(item.get('blocker_codes') or [])}` | "
                f"{'; '.join(item.get('evidence_needed') or [])} |"
            )
        lines.append("")
        lines.append(
            "- Ordinary fresh entries must remain blocked for every near-ready diagnostic row until "
            "its blockers clear in refreshed broker/forecast/replay evidence."
        )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## qr-trader Scheduled Run Watchdog",
            "",
            f"- Status: `{watchdog.get('status')}` severity=`{watchdog.get('severity')}`",
            f"- Generated: `{watchdog.get('generated_at_utc')}`",
            f"- Last run evidence: `{watchdog.get('last_trader_run_at')}`",
            f"- Last run source: `{watchdog.get('last_trader_run_source')}` path=`{watchdog.get('last_trader_run_path')}`",
            f"- Minutes since last run: `{watchdog.get('minutes_since_last_run')}`",
            f"- Missed expected window: `{watchdog.get('missed_expected_window')}`",
            f"- Suspected cause: {watchdog.get('suspected_cause') or 'none'}",
            "",
            "## Guardian Receipt Consumption",
            "",
            f"- Status: `{consumption.get('status')}`",
            f"- Generated: `{consumption.get('generated_at_utc')}`",
            f"- Normal routing allowed: `{consumption.get('normal_routing_allowed')}`",
            f"- Unresolved issue count: `{consumption.get('unresolved_issue_count')}`",
            "- Recommended next action: "
            + _guardian_receipt_consumption_next_action(watchdog, consumption),
        ]
    )
    classifications = (
        consumption.get("classifications")
        if isinstance(consumption.get("classifications"), list)
        else []
    )
    if classifications:
        for item in classifications:
            if isinstance(item, dict):
                lines.append(
                    "- "
                    f"`{item.get('classification')}` issue=`{item.get('issue_code')}` "
                    f"event=`{item.get('receipt_event_id')}` action=`{item.get('receipt_action')}` "
                    f"lifecycle=`{item.get('receipt_lifecycle')}` "
                    f"normal_routing_allowed=`{item.get('normal_routing_allowed')}`"
                )
    else:
        lines.append("- Classifications: none")
    lines.extend(["", "## Guardian Receipt Issues", ""])
    guardian_receipt_issues = (
        watchdog.get("guardian_receipt_issues")
        if isinstance(watchdog.get("guardian_receipt_issues"), list)
        else []
    )
    if guardian_receipt_issues:
        lines.append("- Guardian receipt issues:")
        for issue in guardian_receipt_issues:
            if isinstance(issue, dict):
                lines.append(
                    f"  - `{issue.get('severity')}` `{issue.get('code')}`: {issue.get('message')}"
                )
    else:
        lines.append("- Guardian receipt issues: none")
    lines.extend(["", "## Operator Manual Positions", ""])
    operator_packets = (
        broker.get("operator_manual_position_packets")
        if isinstance(broker.get("operator_manual_position_packets"), list)
        else []
    )
    if operator_packets:
        lines.extend(
            [
                "| Pair | Side | Units | Avg Entry | UPL JPY | Pip Value | Thesis | State | Margin | Harvest Zone | Invalidation Evidence |",
                "|---|---|---:|---:|---:|---:|---|---|---|---|---|",
            ]
        )
        for packet in operator_packets:
            margin = packet.get("margin_pressure") if isinstance(packet.get("margin_pressure"), dict) else {}
            lines.append(
                f"| `{packet.get('pair')}` | `{packet.get('side')}` | `{packet.get('units')}` | "
                f"`{packet.get('avg_entry')}` | `{packet.get('unrealized_pl_jpy')}` | "
                f"`{packet.get('pip_value_jpy_per_pip')}` | {packet.get('thesis') or ''} | "
                f"`{packet.get('thesis_state')}` | `{margin.get('state')}` | "
                f"{packet.get('harvest_zone') or ''} | {packet.get('exact_invalidation_evidence') or ''} |"
            )
        lines.append("")
        lines.append(
            "- Management rule: observe, TP-assist, and report only; no SL, loss-side close, or averaging unless the operator explicitly asks."
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Repair Requests", ""])
    repair_requests = (
        payload.get("repair_requests") if isinstance(payload.get("repair_requests"), list) else []
    )
    if repair_requests:
        lines.extend(
            [
                "| Code | Status | Source | Verify |",
                "|---|---|---|---|",
            ]
        )
        for request in repair_requests[:8]:
            source = ", ".join(f"`{code}`" for code in request.get("source_findings", []) or []) or "none"
            verify = ", ".join(
                f"`{command}`" for command in request.get("verification_commands", [])[:2]
            ) or "none"
            approval = " approval-required" if request.get("requires_explicit_operator_approval") else ""
            lines.append(
                f"| `{request.get('code')}` | `{request.get('status')}`{approval} | {source} | {verify} |"
            )
        lines.append("")
        lines.append(
            "- Automation contract: Codex may edit code/tests/docs, run tests, commit, and sync live; "
            "orders, cancels, closes, and launchd changes require explicit approval or the existing gateway path."
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Bid/Ask Replay Coverage", ""])
    bidask_request = _repair_request_by_code(repair_requests, "COLLECT_BIDASK_REPLAY_EVIDENCE")
    if bidask_request:
        evidence = (
            bidask_request.get("evidence_summary")
            if isinstance(bidask_request.get("evidence_summary"), dict)
            else {}
        )
        lines.extend(_render_bidask_replay_coverage_report(evidence))
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Profit Capture Repair",
            "",
            f"- Status: `{profit['status']}`",
            f"- Actual loss-close PL JPY: `{profit['actual_loss_close_pl_jpy']}`",
            f"- Counterfactual profit-capture PL JPY: `{profit['counterfactual_profit_capture_pl_jpy']}`",
            f"- Counterfactual profit-capture delta JPY: `{profit['counterfactual_profit_capture_delta_jpy']}`",
            f"- Production-gate replay triggered: `{profit['repair_replay_triggered']}`",
            f"- Production-gate replay delta JPY: `{profit['repair_replay_delta_jpy']}`",
            f"- Clearance condition: {profit['clearance_condition']}",
            f"- Verify: `{profit['verification_command']}`",
        ]
    )
    if profit.get("top_repair_replay_triggers"):
        lines.extend(
            [
                "",
                "| Repair Trade | Pair | Side | Exit | Trigger UTC | Repair pips | Noise floor | Repair JPY | Delta JPY |",
                "|---|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for item in profit["top_repair_replay_triggers"][:5]:
            lines.append(
                f"| `{item.get('trade_id')}` | `{item.get('pair')}` | `{item.get('side')}` | "
                f"`{item.get('exit_reason')}` | `{item.get('repair_trigger_at_utc') or item.get('repair_replay_trigger_at_utc')}` | "
                f"`{_round_optional(item.get('repair_profit_pips') or item.get('repair_replay_profit_pips'), 4)}` | "
                f"`{_round_optional(item.get('repair_noise_floor_pips') or item.get('repair_replay_noise_floor_pips'), 4)}` | "
                f"`{_round_optional(item.get('repair_counterfactual_jpy') or item.get('repair_replay_counterfactual_jpy'), 3)}` | "
                f"`{_round_optional(item.get('repair_counterfactual_delta_jpy') or item.get('repair_replay_counterfactual_net_improvement_jpy'), 3)}` |"
            )
    if profit.get("top_misses"):
        lines.extend(
            [
                "",
                "| Trade | Pair | Side | Exit | MFE JPY | TP progress | Counterfactual JPY | Delta JPY | Realized JPY |",
                "|---|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in profit["top_misses"][:5]:
            lines.append(
                f"| `{item.get('trade_id')}` | `{item.get('pair')}` | `{item.get('side')}` | "
                f"`{item.get('exit_reason')}` | "
                f"`{_round_optional(item.get('estimated_mfe_jpy_before_loss_close'), 3)}` | "
                f"`{_round_optional(item.get('tp_progress_before_loss_close'), 4)}` | "
                f"`{_round_optional(item.get('profit_capture_counterfactual_jpy'), 3)}` | "
                f"`{_round_optional(item.get('profit_capture_counterfactual_net_improvement_jpy'), 3)}` | "
                f"`{_round_optional(item.get('realized_pl_jpy'), 3)}` |"
            )
    else:
        lines.append("- Missed capture examples: none")
    lines.extend(
        [
            "",
            "## Guardian",
            "",
            f"- Required: `{guardian['required']}`",
            f"- Label: `{guardian['label']}`",
            f"- Plist: `{guardian['plist_path']}` exists=`{guardian['plist_exists']}`",
            f"- Launchd loaded: `{guardian['launchd_loaded']}`",
            f"- Heartbeat path: `{guardian['heartbeat_path']}`",
            f"- Heartbeat generated: `{guardian['heartbeat_generated_at_utc']}`",
            "",
            "## Directional Inversion Counterfactuals",
            "",
        ]
    )
    inversion_rows = (
        broker.get("directional_inversion_counterfactuals")
        if isinstance(broker.get("directional_inversion_counterfactuals"), list)
        else []
    )
    if inversion_rows:
        lines.extend(
            [
                "| Trade | Owner | Pair | Actual | Opposite | Actual UPL JPY | Opposite gross JPY | Clears 5% minimum | Replay status | Evidence status |",
                "|---|---|---|---|---|---:|---:|---|---|---|",
            ]
        )
        for item in inversion_rows[:8]:
            replay = item.get("replay_verification") if isinstance(item.get("replay_verification"), dict) else {}
            lines.append(
                f"| `{item.get('trade_id')}` | `{item.get('owner')}` | `{item.get('pair')}` | "
                f"`{item.get('actual_side')}` | `{item.get('opposite_side')}` | "
                f"`{item.get('actual_unrealized_pl_jpy')}` | "
                f"`{item.get('opposite_gross_counterfactual_pl_jpy')}` | "
                f"`{item.get('would_clear_minimum_5pct')}` | "
                f"`{replay.get('status') or 'UNVERIFIED'}` | "
                f"`{item.get('inversion_replay_evidence_status')}` |"
            )
        lines.append("")
        lines.append(
            "- Counterfactuals are gross sign-flips of current broker-truth unrealized P/L; they are repair evidence, not live inversion permission."
        )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Top Intent Blockers",
            "",
        ]
    )
    if entry["top_blockers"]:
        for item in entry["top_blockers"]:
            lines.append(f"- `{item['code']}`: `{item['count']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Guardian Recovery Candidates", ""])
    if entry["repair_basket_guardian_recovery"]:
        lines.extend(
            [
                "| Lane | Pair | Side | Method | Reward JPY | TP proof | Remaining blockers after guardian |",
                "|---|---|---|---|---:|---|---|",
            ]
        )
        for item in entry["repair_basket_guardian_recovery"][:8]:
            remaining = ", ".join(
                item["remaining_blocker_codes_after_guardian_and_repair_exemption"]
            ) or "none"
            tp_proof = _format_tp_proof_report_cell(item.get("tp_proof"))
            lines.append(
                f"| `{item['lane_id']}` | `{item['pair']}` | `{item['side']}` | "
                f"`{item['method']}` | `{item['reward_jpy']}` | {tp_proof} | `{remaining}` |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Repair Frontier Blockers After Support", ""])
    if entry["repair_frontier_remaining_blockers"]:
        lines.extend(
            [
                "| Blocker | Lanes | Reward JPY | Examples |",
                "|---|---:|---:|---|",
            ]
        )
        for item in entry["repair_frontier_remaining_blockers"][:8]:
            examples = ", ".join(f"`{lane_id}`" for lane_id in item["example_lane_ids"]) or "none"
            lines.append(
                f"| `{item['code']}` | `{item['count']}` | `{item['reward_jpy']}` | {examples} |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Repair Frontier", ""])
    if entry["repair_frontier"]:
        lines.extend(
            [
                "| Lane | Pair | Side | Method | Reward JPY | Remaining blockers after support |",
                "|---|---|---|---|---:|---|",
            ]
        )
        for item in entry["repair_frontier"][:8]:
            remaining = ", ".join(item["remaining_blocker_codes_after_guardian_and_repair_exemption"]) or "none"
            lines.append(
                f"| `{item['lane_id']}` | `{item['pair']}` | `{item['side']}` | "
                f"`{item['method']}` | `{item['reward_jpy']}` | `{remaining}` |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## OANDA Audit-Only Local TP Proof Required", ""])
    if entry["oanda_audit_only_local_tp_proof_required"]:
        history_coverage = (
            payload.get("oanda_history_coverage")
            if isinstance(payload.get("oanda_history_coverage"), dict)
            else {}
        )
        history_complete = _oanda_history_coverage_is_complete(history_coverage)
        replay_can_clear = _oanda_audit_only_candidates_have_clearable_replay(
            entry["oanda_audit_only_local_tp_proof_required"]
        )
        missing_history = history_coverage.get("missing_pairs_by_granularity")
        if isinstance(missing_history, dict):
            missing_bits = [
                f"{granularity}:{','.join(pairs) or 'none'}"
                for granularity, pairs in missing_history.items()
                if isinstance(pairs, list)
            ]
            lines.append(
                f"- Local OANDA history coverage: `{history_coverage.get('status')}`; missing "
                f"{'; '.join(missing_bits) or 'none'}."
            )
        if history_complete and not replay_can_clear:
            lines.append(
                "- Local OANDA history is complete and the current replay still cannot clear "
                "local TP proof. Do not rerun validate/mine/package until new local TP receipts, "
                "new forecast/candle evidence, or exact HARVEST live-grade promotion changes the inputs."
            )
        lines.append(
            "- These candidates still lack live-grade replay or local TP proof. Escape condition: "
            "exact pair/side/method TAKE_PROFIT_ORDER proof with positive expectancy, zero TP "
            "losses, and positive Wilson-stressed expectancy, or exact OANDA HARVEST vehicle "
            "promotion through current-risk / normal-cap 5% firepower scaling."
        )
        lines.append("")
        lines.extend(
            [
                "| Lane | Pair | Side | Method | Scope | Vehicle | Replay evidence | Historical clears local proof | Remaining blockers after guardian |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for item in entry["oanda_audit_only_local_tp_proof_required"][:8]:
            remaining = ", ".join(item["remaining_blocker_codes_after_guardian"]) or "none"
            replay = _format_oanda_replay_report_cell(item)
            lines.append(
                f"| `{item['lane_id']}` | `{item['pair']}` | `{item['side']}` | "
                f"`{item['method']}` | `{item.get('capture_take_profit_scope')}` | "
                f"`{item.get('oanda_vehicle_key')}` | {replay} | "
                f"`{item.get('historical_replay_can_clear_local_tp_proof')}` | `{remaining}` |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## RANGE Forecast Superseded Repair Lanes", ""])
    if entry["repair_frontier_superseded_by_range_forecast"]:
        lines.extend(
            [
                "| Lane | Pair | Side | Method | Reward JPY | Range counterpart |",
                "|---|---|---|---|---:|---|",
            ]
        )
        for item in entry["repair_frontier_superseded_by_range_forecast"][:8]:
            counterpart = item.get("superseded_by_range_rotation_lane_id") or "missing"
            lines.append(
                f"| `{item['lane_id']}` | `{item['pair']}` | `{item['side']}` | "
                f"`{item['method']}` | `{item['reward_jpy']}` | `{counterpart}` |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## RANGE Forecast Missing Counterpart Repair Lanes", ""])
    if entry["repair_frontier_missing_range_rotation_counterpart"]:
        lines.extend(
            [
                "| Lane | Pair | Side | Method | Reward JPY | Missing counterpart |",
                "|---|---|---|---|---:|---|",
            ]
        )
        for item in entry["repair_frontier_missing_range_rotation_counterpart"][:8]:
            missing_pair_side = item.get("missing_range_rotation_counterpart_for") or []
            missing = ":".join(str(part) for part in missing_pair_side if part) or "unknown"
            lines.append(
                f"| `{item['lane_id']}` | `{item['pair']}` | `{item['side']}` | "
                f"`{item['method']}` | `{item['reward_jpy']}` | `{missing}` |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Global Unlock Frontier", ""])
    if entry["global_unlock_frontier"]:
        lines.extend(
            [
                "| Lane | Pair | Side | Method | Reward JPY | TP proof | Global blockers |",
                "|---|---|---|---|---:|---|---|",
            ]
        )
        for item in entry["global_unlock_frontier"][:8]:
            blockers = ", ".join(item["global_blocker_codes"]) or "none"
            tp_proof = _format_tp_proof_report_cell(item.get("tp_proof"))
            lines.append(
                f"| `{item['lane_id']}` | `{item['pair']}` | `{item['side']}` | "
                f"`{item['method']}` | `{item['reward_jpy']}` | {tp_proof} | `{blockers}` |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Target Firepower Evidence", ""])
    if firepower.get("status"):
        lines.append(f"- Status: `{firepower.get('status')}`")
        lines.append(f"- Best bucket: `{firepower.get('best_bucket')}`")
        lines.append(f"- 5% estimated reachable: `{firepower.get('minimum_5pct_estimated_reachable')}`")
        lines.append(f"- 10% estimated reachable: `{firepower.get('target_10pct_estimated_reachable')}`")
        lines.append("- Audit only, no live permission grant: `true`")
        for label in ("high_precision", "evidence_queue"):
            bucket = firepower.get(label) if isinstance(firepower.get(label), dict) else {}
            if bucket.get("estimated_return_pct_per_active_day_at_observed_frequency") is None:
                continue
            lines.append(
                f"- `{label}` return/day=`{bucket.get('estimated_return_pct_per_active_day_at_observed_frequency')}` "
                f"trades_needed_5pct=`{bucket.get('trades_needed_for_minimum_5pct_at_weighted_expectancy')}` "
                f"vehicles=`{bucket.get('top_vehicle_keys')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Acceptance Repair Plan", ""])
    if acceptance_repair.get("items"):
        lines.append(f"- Loop breaker: {acceptance_repair.get('loop_breaker')}")
        lines.extend(
            [
                "",
                "| Code | Clearance condition | Verify |",
                "|---|---|---|",
            ]
        )
        for item in acceptance_repair["items"][:8]:
            lines.append(
                f"| `{item['code']}` | {item['clearance_condition']} | "
                f"`{item['verification_command']}` |"
            )
    else:
        lines.append("- none")
    evidence_items = (
        acceptance_repair.get("evidence_collection_items")
        if isinstance(acceptance_repair.get("evidence_collection_items"), list)
        else []
    )
    lines.extend(["", "## Acceptance Evidence Collection", ""])
    if evidence_items:
        lines.extend(
            [
                "| Code | Clearance condition | Verify |",
                "|---|---|---|",
            ]
        )
        for item in evidence_items[:8]:
            lines.append(
                f"| `{item['code']}` | {item['clearance_condition']} | "
                f"`{item['verification_command']}` |"
            )
            summary = item.get("evidence_summary") if isinstance(item.get("evidence_summary"), dict) else {}
            fetch = summary.get("history_fetch_command")
            if fetch:
                lines.append(f"| `{item['code']}:fetch` | Fetch fresh BA candles | `{fetch}` |")
    else:
        lines.append("- none")
    lines.extend(["", "## Current Profit Capture", ""])
    if current_profit["top_gate_statuses"]:
        for item in current_profit["top_gate_statuses"]:
            lines.append(
                f"- `{item['trade_id']}` `{item['pair']}` `{item['side']}` "
                f"gate=`{item['gate_status']}` progress=`{item['tp_progress']}` trigger=`{item['capture_trigger']}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Open Trader Positions", ""])
    if broker["open_trader_positions"]:
        for item in broker["open_trader_positions"]:
            lines.append(
                f"- `{item['trade_id']}` `{item['pair']}` `{item['side']}` "
                f"units=`{item['units']}` upl=`{item['unrealized_pl_jpy']}` TP=`{item['take_profit']}` SL=`{item['stop_loss']}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Operator Actions", ""])
    for action in payload["operator_actions"]:
        approval = " approval-required" if action.get("requires_explicit_operator_approval") else ""
        lines.append(f"- `{action['code']}`{approval}: `{action['command']}` — {action['reason']}")
    return "\n".join(lines) + "\n"


def _format_oanda_replay_report_cell(item: dict[str, Any]) -> str:
    evidence = item.get("oanda_replay_evidence") if isinstance(item.get("oanda_replay_evidence"), dict) else {}
    if not evidence or evidence.get("evidence_status") == "MISSING_OANDA_REPLAY_EVIDENCE":
        return "`missing`"
    parts = [
        f"status={evidence.get('evidence_status')}",
        f"n={evidence.get('validation_n')}",
        f"win={evidence.get('validation_win_rate')}",
        f"pf={evidence.get('validation_profit_factor')}",
        f"day%={evidence.get('estimated_return_pct_per_active_day_at_observed_frequency')}",
        f"5%trades={evidence.get('trades_needed_for_minimum_5pct')}",
    ]
    return "`" + " ".join(str(part) for part in parts) + "`"


def _repair_request_by_code(requests: list[Any], code: str) -> dict[str, Any]:
    for item in requests:
        if isinstance(item, dict) and item.get("code") == code:
            return item
    return {}


def _render_bidask_replay_coverage_report(evidence: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    price_truth = (
        evidence.get("price_truth_coverage")
        if isinstance(evidence.get("price_truth_coverage"), dict)
        else {}
    )
    daily = (
        evidence.get("daily_stability_requirements")
        if isinstance(evidence.get("daily_stability_requirements"), dict)
        else {}
    )
    lines.append(
        "- Status: "
        f"`{evidence.get('all_currency_sample_coverage_status')}` "
        f"replay=`{evidence.get('replay_evidence_status')}` "
        f"price_truth=`{price_truth.get('status')}` "
        f"fetch_required=`{evidence.get('price_truth_fetch_required')}`"
    )
    lines.append(
        "- Rule counts: "
        f"edge=`{evidence.get('edge_rules')}` daily_stable_edge=`{evidence.get('daily_stable_edge_rules')}` "
        f"support=`{evidence.get('support_rules')}` daily_stable_support=`{evidence.get('daily_stable_support_rules')}` "
        f"rank_only_support=`{evidence.get('rank_only_support_rules')}` "
        f"negative=`{evidence.get('negative_rules')}`"
    )
    if daily:
        lines.append(
            "- Daily stability requirement: "
            f"active_days>=`{daily.get('min_active_days')}` "
            f"max_daily_share<=`{daily.get('max_daily_sample_share')}` "
            f"positive_day_rate>=`{daily.get('min_positive_day_rate')}`"
        )
    lines.append(
        "- Under-sampled pair-directions: "
        f"`{evidence.get('under_sampled_pair_direction_count')}` "
        f"missing_evaluated_samples=`{evidence.get('under_sampled_missing_evaluated_samples')}`"
    )
    examples = evidence.get("under_sampled_pair_direction_examples")
    if isinstance(examples, list) and examples:
        lines.append("")
        lines.extend(
            [
                "| Pair | Direction | Samples | Active days | Gap reasons |",
                "|---|---|---:|---:|---|",
            ]
        )
        for item in examples[:8]:
            if not isinstance(item, dict):
                continue
            reasons = item.get("gap_reasons") or item.get("reasons") or []
            if isinstance(reasons, list):
                reason_text = ", ".join(str(reason) for reason in reasons)
            else:
                reason_text = str(reasons or "")
            lines.append(
                f"| `{item.get('pair')}` | `{item.get('direction')}` | "
                f"`{item.get('evaluated_samples') or item.get('samples')}` | "
                f"`{item.get('active_days')}` | {reason_text or 'none'} |"
            )
    directions = evidence.get("under_sampled_pair_directions")
    if isinstance(directions, list) and directions:
        lines.append(
            "- First under-sampled directions: "
            + ", ".join(f"`{item}`" for item in directions[:12])
        )
    history = (
        evidence.get("current_intent_local_history_coverage")
        if isinstance(evidence.get("current_intent_local_history_coverage"), dict)
        else {}
    )
    if history:
        missing = history.get("missing_pairs_by_granularity")
        if isinstance(missing, dict):
            bits = [
                f"{granularity}:{','.join(str(pair) for pair in pairs) or 'none'}"
                for granularity, pairs in missing.items()
                if isinstance(pairs, list)
            ]
            lines.append(
                "- Current-intent local BA history: "
                f"`{history.get('status')}` complete=`{history.get('complete')}` "
                f"diagnostic_only=`{history.get('diagnostic_only')}` missing={'; '.join(bits) or 'none'}"
            )
        fetch_commands = history.get("fetch_commands")
        if isinstance(fetch_commands, list) and fetch_commands:
            lines.append("- Diagnostic history fetch command: `" + str(fetch_commands[0]) + "`")
    command = evidence.get("replay_validation_command") or evidence.get("verification_command")
    if command:
        lines.append("- Replay validation command: `" + str(command) + "`")
    lines.append(
        "- Live permission: remains blocked until forecast sample coverage graduates from "
        "`UNDER_SAMPLED` and replay rules become daily-stable/live-grade."
    )
    return lines


def _format_tp_proof_report_cell(value: Any) -> str:
    proof = value if isinstance(value, dict) else {}
    if not proof:
        return "`missing`"
    parts: list[str] = []
    mode = proof.get("positive_rotation_mode")
    if mode is not None:
        parts.append(f"mode={mode}")
    scope = proof.get("capture_take_profit_scope")
    if scope is not None:
        parts.append(f"scope={scope}")
    trades = proof.get("capture_take_profit_trades")
    losses = proof.get("capture_take_profit_losses")
    if trades is not None or losses is not None:
        parts.append(f"trades={trades}")
        parts.append(f"losses={losses}")
    expectancy = _round_optional(proof.get("capture_take_profit_expectancy_jpy"), 3)
    if expectancy is not None:
        parts.append(f"tp_exp_jpy={expectancy}")
    pessimistic = _round_optional(proof.get("positive_rotation_pessimistic_expectancy_jpy"), 3)
    if pessimistic is not None:
        parts.append(f"pess_exp_jpy={pessimistic}")
    return "`" + " ".join(parts) + "`" if parts else "`missing`"


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _int_like(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _round_optional(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None

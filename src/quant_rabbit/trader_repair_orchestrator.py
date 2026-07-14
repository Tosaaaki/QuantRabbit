from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_TRADER_REPAIR_ORCHESTRATOR,
    DEFAULT_TRADER_REPAIR_ORCHESTRATOR_REPORT,
    DEFAULT_TRADER_SUPPORT_BOT,
)
from quant_rabbit.trader_support_bot import (
    BIDASK_REPLAY_WAIT_STATUS,
    DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST,
    DIRECTIONAL_INVERSION_REPLAY_WAIT_STATUS,
    FRONTIER_MARGIN_CAPACITY_WAIT_STATUS,
    FRONTIER_PROOF_EVIDENCE_BLOCKER_CODES,
    FRONTIER_PROOF_EVIDENCE_WAIT_STATUS,
    FRONTIER_QUOTE_FRESHNESS_WAIT_STATUS,
    FRONTIER_STRATEGY_PROFILE_EVIDENCE_WAIT_STATUS,
    OANDA_AUDIT_ONLY_LOCAL_TP_EDGE_REQUEST,
    OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_UNPROVED_STATUS,
    PENDING_CANCEL_RECEIPT_WAIT_STATUS,
    PENDING_CANCEL_REVIEW_CODE,
    POSITION_GUARDIAN_LOCK_WAIT_STATUS,
    REPAIR_AUTOMATION_ALLOWED_ACTIONS,
    REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS,
    REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS,
    TP_PROGRESS_GUARDIAN_WAIT_STATUS,
    TP_PROGRESS_LIVE_EVIDENCE_WAIT_STATUS,
    repair_requests_from_support_payload,
)


CONTRACT_VERSION = "trader_repair_orchestrator_v1"
LOOP_ENGINEERING_PROMPT_VERSION = "loop_engineering_prompt_v1"
STATUS_READY = "READY_FOR_CODEX_REPAIR"
STATUS_APPROVAL_REQUIRED = "OPERATOR_APPROVAL_REQUIRED"
STATUS_NO_REQUESTS = "NO_REPAIR_REQUESTS"
STATUS_BLOCKED = "ORCHESTRATOR_BLOCKED"
AUTOMATION_READY = "READY_FOR_CODEX_IMPLEMENTATION"
READ_ONLY_EVIDENCE_WORK_STATUS = "READ_ONLY_EVIDENCE_WORK"
READ_ONLY_EVIDENCE_WAIT_STATUS = "WAITING_FOR_MATERIAL_EVIDENCE"
READ_ONLY_EVIDENCE_MAPPING_REQUIRED_STATUS = "ACTIVE_LANE_ACTION_MAPPING_REQUIRED"
ACTIVE_LANE_EVIDENCE_CANDIDATE_STATUS = "ACTIVE_LANE_EVIDENCE_CANDIDATE"
AUTOMATION_OPERATOR_APPROVAL = "WAITING_FOR_OPERATOR_APPROVAL"
AUTOMATION_LIVE_EVIDENCE_WINDOW = "WAITING_FOR_LIVE_EVIDENCE_WINDOW"
AUTOMATION_EVIDENCE = "WAITING_FOR_EVIDENCE"
NON_ACTIONABLE_REPAIR_STATUSES = {
    "FORECAST_FRONTIER_WAITING_FOR_LIVE_PRECISION_EVIDENCE",
    "FRONTIER_PROTECTIVE_GUARDRAIL_ACTIVE",
    "HISTORICAL_ACCEPTANCE_WINDOW_ACTIVE",
    "RESIDUAL_GROUPS_ALREADY_BLOCKED_WAITING_FOR_REPLAY",
    FRONTIER_MARGIN_CAPACITY_WAIT_STATUS,
    FRONTIER_PROOF_EVIDENCE_WAIT_STATUS,
    FRONTIER_QUOTE_FRESHNESS_WAIT_STATUS,
    FRONTIER_STRATEGY_PROFILE_EVIDENCE_WAIT_STATUS,
    TP_PROGRESS_GUARDIAN_WAIT_STATUS,
    TP_PROGRESS_LIVE_EVIDENCE_WAIT_STATUS,
    PENDING_CANCEL_RECEIPT_WAIT_STATUS,
    BIDASK_REPLAY_WAIT_STATUS,
    DIRECTIONAL_INVERSION_REPLAY_WAIT_STATUS,
    OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_UNPROVED_STATUS,
    POSITION_GUARDIAN_LOCK_WAIT_STATUS,
}
CODEX_ACTIONABLE_REPAIR_STATUSES = {
    "READY_FOR_CODE_REPAIR",
    "READY_FOR_CODE_OR_EVIDENCE_REPAIR",
    "READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
}
REPAIR_DEPENDENCY_RANK = {
    "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY": 0,
    "RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT": 1,
    PENDING_CANCEL_REVIEW_CODE: 2,
    DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST: 2,
    "REPAIR_CLOSE_GATE_EVIDENCE_PERSISTENCE": 3,
    "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY": 4,
    "REPAIR_FRONTIER_LANE_BLOCKER": 5,
    OANDA_AUDIT_ONLY_LOCAL_TP_EDGE_REQUEST: 6,
    "COLLECT_BIDASK_REPLAY_EVIDENCE": 7,
    "REVIEW_CLOSE_GATE_EVIDENCE_FAILURES": 90,
}
REPAIR_SELECTION_REASONS = {
    "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY": (
        "Direct TP-progress profit-capture repair outranks residual-entry scaling because "
        "TAKE_PROFIT_ORDER gains are still being erased by later loss-side market closes."
    ),
    "REPAIR_CLOSE_GATE_EVIDENCE_PERSISTENCE": (
        "Close-gate evidence persistence is a loss-leak repair, but it follows the direct "
        "TP-progress capture path when both are actionable."
    ),
    DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST: (
        "A broker-truth opposite-side counterfactual would clear the +5% pace marker inside "
        "the rolling 30d 4x campaign, so Codex must audit forecast inversion and opposite-lane "
        "suppression before adding unrelated entry frequency."
    ),
    "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY": (
        "Residual entry-quality repair is scaling work after TP-progress capture and close "
        "discipline have a proved live path; if matching current intents are already blocked, "
        "Codex must wait for a 744h replay instead of reimplementing the same block."
    ),
    OANDA_AUDIT_ONLY_LOCAL_TP_EDGE_REQUEST: (
        "OANDA audit-only forecast candidates are actionable as read-only precision work: "
        "fetch bid/ask truth, validate replay, mine exact vehicles, package reviewed rules, "
        "and keep live permission blocked until local TP receipts or live-grade current-risk "
        "firepower prove the exact HARVEST edge."
    ),
    "REVIEW_CLOSE_GATE_EVIDENCE_FAILURES": (
        "Historical BLOCK close-gate evidence must wait for future PASS evidence or age-out; "
        "Codex must not synthesize PASS for old closes."
    ),
}
OPERATOR_REVIEW_APPROVAL_BLOCKERS = {
    "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
    "GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW",
    "GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING",
}
DIRECT_LIVE_PERMISSION_BLOCKERS = {
    "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
    "GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW",
    "GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING",
    "PROFITABILITY_ACCEPTANCE_BLOCKED",
    "NEGATIVE_EXPECTANCY_ACTIVE",
    "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
    "MARKET_CLOSE_LEAK_FAMILY_BLOCKED",
    "SELF_IMPROVEMENT_P0_PRESENT",
    "MEMORY_HEALTH_BLOCKED",
    "NO_LIVE_READY_LANES",
    "TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE",
}
PROOF_CANDIDATE_BLOCKERS = {
    "spread_included_bidask_replay_negative_for_exact_lane",
    "packaged_bidask_rule_live_block_negative_expectancy",
    "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
    "S5_DAILY_SAMPLE_CONCENTRATED",
    "S5_POSITIVE_DAY_RATE_LOW",
}
GATEWAY_LIVE_READY_STATUSES = {"ACCEPTED", "STAGED", "SENT", "LIVE_READY"}
EVIDENCE_ACTION_PROGRESS_WATERMARK_CONTRACT = "evidence_action_progress_v1"
EVIDENCE_ACTION_MATERIAL_SOURCE_NEXT_ACTION = "next_action"
EVIDENCE_ACTION_MATERIAL_SOURCE_PROGRESS_RECEIPT = "progress_receipt"
_EVIDENCE_ACTION_PROGRESS_RECEIPT_FIELDS = frozenset(
    {
        "action_id",
        "category",
        "progress_watermark_contract",
        "progress_watermark_origin",
        "success_condition",
        "success_condition_evaluation",
    }
)
# The proof/support artifacts are rebuilt sequentially in one cycle. Five
# minutes is the operational drift window used here to distinguish same-packet
# evidence from an older cycle without tying the rule to market risk.
PROOF_EMPTY_REASON_FRESHNESS_MAX_AGE_SECONDS = 5 * 60
PROOF_EMPTY_REASON_FUTURE_TOLERANCE_SECONDS = 1.0
FRESHNESS_PRIMARY_PRIORITY = {
    "FRESH": 0,
    "MISSING": 1,
    "STALE": 2,
    "CONTRADICTED": 99,
}

ACTIVE_LANE_WAIT_ACTION = "WAIT_FOR_RANGE_RAIL_RECHECK"
ACTIVE_LANE_VERIFY_PROJECTIONS_ACTION = "VERIFY_TRIGGER_PROJECTIONS"
ACTIVE_LANE_ENTRY_DROUGHT_ACTION = "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH"
ACTIVE_LANE_FORECAST_PATTERN_ACTION = "FORECAST_PATTERN_REFRESH"
ACTIVE_LANE_RANGE_RAIL_ACTION = "RANGE_RAIL_GEOMETRY_REPAIR"
ACTIVE_LANE_EXACT_TP_PROOF_ACTION = "EXACT_TP_PROOF_COLLECTION"
ACTIVE_LANE_REPRICE_RANGE_ACTION = "REPRICE_RANGE_ROTATION_COUNTERPART"
ACTIVE_LANE_TRIGGER_PROOF_ACTION = "TRIGGER_PROJECTION_TO_LIMIT_PROOF"
ACTIVE_LANE_RANGE_READY_PROOF_ACTION = "RANGE_ROTATION_GEOMETRY_READY_PROOF_BLOCKED"
ACTIVE_LANE_CURRENT_INTENT_REGEN_ACTION = "CURRENT_INTENT_REGEN_REQUIRED"
ACTIVE_LANE_NON_MARKET_PROOF_ROUTE_ACTION = "NON_MARKET_TP_PROOF_ROUTE_REQUIRED"
ACTIVE_LANE_KEEP_BLOCKED_ACTION = "KEEP_BLOCKED_WITH_CAUSE"
ACTIVE_LANE_PRESERVE_BLOCKERS_ACTION = "PRESERVE_SPREAD_AND_EXPECTANCY_BLOCKERS"
ACTIVE_LANE_UNRESOLVED_ACTION = "UNRESOLVED_ACTIVE_LANE_ACTION"
ACTIVE_LANE_DISPATCH_HISTORY_LIMIT = 512
ACTIVE_LANE_ACK_COMMAND_TEMPLATE = (
    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator "
    "--ack-active-lane-dispatch {material_digest}"
)


@dataclass(frozen=True)
class TraderRepairOrchestratorSummary:
    status: str
    output_path: Path
    report_path: Path
    selected_request_code: str | None
    repair_request_count: int
    actionable_request_count: int
    approval_required_request_count: int
    waiting_request_count: int


class TraderRepairOrchestrator:
    """Build a machine-readable repair queue for the Codex work loop.

    This command is deliberately read-only. It translates the support bot's
    repair requests into an execution contract: Codex may edit/test/commit/sync
    code, while order sends, cancels, position closes, and launchd mutations
    remain outside this automation unless an existing gateway or explicit
    operator approval authorizes them.
    """

    def __init__(
        self,
        *,
        support_bot_path: Path = DEFAULT_TRADER_SUPPORT_BOT,
        output_path: Path = DEFAULT_TRADER_REPAIR_ORCHESTRATOR,
        report_path: Path = DEFAULT_TRADER_REPAIR_ORCHESTRATOR_REPORT,
        proof_queue_path: Path | None = None,
        as_lane_board_path: Path | None = None,
        portfolio_4x_path_planner_path: Path | None = None,
        live_order_request_path: Path | None = None,
        broker_snapshot_path: Path | None = None,
        trader_request: str | None = None,
        ack_active_lane_dispatch: str | None = None,
        now_utc: datetime | None = None,
    ) -> None:
        self.support_bot_path = support_bot_path
        self.output_path = output_path
        self.report_path = report_path
        artifact_root = output_path.parent
        self.proof_queue_path = proof_queue_path or artifact_root / "as_proof_pack_queue.json"
        self.as_lane_board_path = as_lane_board_path or artifact_root / "as_lane_candidate_board.json"
        self.portfolio_4x_path_planner_path = (
            portfolio_4x_path_planner_path or artifact_root / "portfolio_4x_path_planner.json"
        )
        if live_order_request_path:
            self.live_order_request_path = live_order_request_path
        elif output_path == DEFAULT_TRADER_REPAIR_ORCHESTRATOR:
            self.live_order_request_path = DEFAULT_LIVE_ORDER_REQUEST
        else:
            self.live_order_request_path = artifact_root / "live_order_request.json"
        if broker_snapshot_path:
            self.broker_snapshot_path = broker_snapshot_path
        elif output_path == DEFAULT_TRADER_REPAIR_ORCHESTRATOR:
            self.broker_snapshot_path = DEFAULT_BROKER_SNAPSHOT
        else:
            self.broker_snapshot_path = artifact_root / "broker_snapshot.json"
        self.trader_request = trader_request or ""
        self.ack_active_lane_dispatch = str(ack_active_lane_dispatch or "").strip()
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> TraderRepairOrchestratorSummary:
        with _exclusive_orchestrator_state_lock(self.output_path):
            payload = self.build_payload()
            _atomic_publish_orchestrator_outputs(
                output_path=self.output_path,
                output_content=(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                ),
                report_path=self.report_path,
                report_content=_render_report(payload),
            )
        selected = payload.get("selected_request") if isinstance(payload.get("selected_request"), dict) else {}
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        return TraderRepairOrchestratorSummary(
            status=str(payload.get("status") or STATUS_BLOCKED),
            output_path=self.output_path,
            report_path=self.report_path,
            selected_request_code=selected.get("code"),
            repair_request_count=int(metrics.get("repair_request_count") or 0),
            actionable_request_count=int(metrics.get("actionable_request_count") or 0),
            approval_required_request_count=int(metrics.get("approval_required_request_count") or 0),
            waiting_request_count=int(metrics.get("waiting_request_count") or 0),
        )

    def build_payload(self) -> dict[str, Any]:
        previous_payload, previous_output_recovery = _read_previous_orchestrator_output(
            self.output_path
        )
        previous_work_order = (
            previous_payload.get("codex_work_order")
            if isinstance(previous_payload.get("codex_work_order"), dict)
            else {}
        )
        previous_evidence_actions = _previous_evidence_actions(previous_payload)
        previous_current_state = _previous_orchestrator_current_state(previous_payload)
        support = _read_json(self.support_bot_path)
        proof_queue = _read_optional_json(self.proof_queue_path)
        as_board = _read_optional_json(self.as_lane_board_path)
        portfolio_planner = _read_optional_json(self.portfolio_4x_path_planner_path)
        live_order_request = _read_optional_json(self.live_order_request_path)
        broker_snapshot = _read_optional_json(self.broker_snapshot_path)
        raw_requests = support.get("repair_requests") if isinstance(support.get("repair_requests"), list) else []
        requests = [item for item in raw_requests if isinstance(item, dict)]
        request_source = "top_level_repair_requests"
        recovered_from_embedded_support = False
        if not requests:
            recovered_requests = repair_requests_from_support_payload(support)
            if recovered_requests:
                requests = recovered_requests
                request_source = "embedded_support_payload"
                recovered_from_embedded_support = True
        queue = [_queue_item(item, trader_request=self.trader_request) for item in requests]
        queue.sort(key=_queue_sort_key)
        actionable = [item for item in queue if item["automation_status"] == AUTOMATION_READY]
        approval_required = [item for item in queue if item["automation_status"] == AUTOMATION_OPERATOR_APPROVAL]
        waiting = [
            item
            for item in queue
            if item["automation_status"] not in {AUTOMATION_READY, AUTOMATION_OPERATOR_APPROVAL}
        ]
        selected = _select_request(actionable, trader_request=self.trader_request)
        execution_contract = _execution_contract()
        approval_boundary = _approval_boundary(execution_contract)
        if actionable:
            status = STATUS_READY
        elif approval_required:
            status = STATUS_APPROVAL_REQUIRED
        elif requests:
            status = STATUS_BLOCKED
        else:
            status = STATUS_NO_REQUESTS
        loop_prompt = _loop_engineering_prompt(
            support=support,
            queue=queue,
            actionable=actionable,
            approval_required=approval_required,
            waiting=waiting,
            selected=selected,
            status=status,
            trader_request=self.trader_request,
            approval_boundary=approval_boundary,
            proof_queue=proof_queue,
            as_board=as_board,
            portfolio_planner=portfolio_planner,
            live_order_request=live_order_request,
            broker_snapshot=broker_snapshot,
            previous_evidence_actions=previous_evidence_actions,
            previous_current_state=previous_current_state,
        )
        payload = {
            "contract_version": CONTRACT_VERSION,
            "generated_at_utc": self.now_utc.isoformat(),
            "status": status,
            "trader_request": self.trader_request,
            "selected_request_code": selected.get("code") if selected else None,
            "repair_request_count": len(queue),
            "actionable_request_count": len(actionable),
            "approval_required_request_count": len(approval_required),
            "waiting_request_count": len(waiting),
            "artifact_paths": {
                "trader_support_bot": str(self.support_bot_path),
                "as_proof_pack_queue": str(self.proof_queue_path),
                "as_lane_candidate_board": str(self.as_lane_board_path),
                "portfolio_4x_path_planner": str(self.portfolio_4x_path_planner_path),
                "live_order_request": str(self.live_order_request_path),
                "broker_snapshot": str(self.broker_snapshot_path),
                "output": str(self.output_path),
                "report": str(self.report_path),
            },
            "selected_request": selected,
            "queue": queue,
            "actionable_requests": actionable,
            "approval_required_requests": approval_required,
            "queue_summary": {
                "selected_request_code": selected.get("code") if selected else None,
                "actionable_request_codes": [str(item.get("code")) for item in actionable if item.get("code")],
                "approval_required_request_codes": [
                    str(item.get("code")) for item in approval_required if item.get("code")
                ],
                "waiting_request_codes": [str(item.get("code")) for item in waiting if item.get("code")],
            },
            "metrics": {
                "repair_request_count": len(queue),
                "actionable_request_count": len(actionable),
                "approval_required_request_count": len(approval_required),
                "waiting_request_count": len(waiting),
                "selected_request_code": selected.get("code") if selected else None,
                "support_status": support.get("status"),
                "repair_request_source": request_source,
                "recovered_from_embedded_support": recovered_from_embedded_support,
                "previous_output_recovery": previous_output_recovery,
            },
            "execution_contract": execution_contract,
            "approval_boundary": approval_boundary,
            "loop_engineering_prompt": loop_prompt,
            "codex_work_order": _codex_work_order(
                selected,
                status=status,
                trader_request=self.trader_request,
                execution_contract=execution_contract,
                approval_boundary=approval_boundary,
                output_path=self.output_path,
                report_path=self.report_path,
                loop_prompt=loop_prompt,
                previous_work_order=previous_work_order,
                ack_active_lane_dispatch=self.ack_active_lane_dispatch,
            ),
            "read_only": True,
            "live_side_effects": [],
        }
        proof_empty_reason = loop_prompt.get("current_state", {}).get("proof_queue_empty_reason")
        next_evidence_actions = loop_prompt.get("current_state", {}).get("next_evidence_actions")
        if proof_empty_reason:
            payload["proof_queue_empty_reason"] = proof_empty_reason
        if next_evidence_actions:
            payload["next_evidence_actions"] = next_evidence_actions
        evidence_action_progress = loop_prompt.get("current_state", {}).get(
            "evidence_action_progress"
        )
        if evidence_action_progress:
            payload["evidence_action_progress"] = evidence_action_progress
        return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path), "repair_requests": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_previous_orchestrator_output(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, "NO_PREVIOUS_OUTPUT"
    try:
        return _read_optional_json(path), "PREVIOUS_OUTPUT_VALID"
    except (OSError, json.JSONDecodeError, ValueError):
        # The previous output is only the anti-loop watermark. A truncated
        # self-artifact must not permanently brick this read-only command; the
        # next atomic publish establishes a new valid baseline.
        return {}, "CORRUPT_PREVIOUS_OUTPUT_IGNORED"


def _previous_evidence_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions = payload.get("next_evidence_actions")
    if not isinstance(actions, list):
        loop_prompt = (
            payload.get("loop_engineering_prompt")
            if isinstance(payload.get("loop_engineering_prompt"), dict)
            else {}
        )
        current_state = (
            loop_prompt.get("current_state")
            if isinstance(loop_prompt.get("current_state"), dict)
            else {}
        )
        actions = current_state.get("next_evidence_actions")
    if not isinstance(actions, list):
        return []
    return [item for item in actions if isinstance(item, dict)]


def _previous_orchestrator_current_state(payload: dict[str, Any]) -> dict[str, Any]:
    loop_prompt = (
        payload.get("loop_engineering_prompt")
        if isinstance(payload.get("loop_engineering_prompt"), dict)
        else {}
    )
    current_state = loop_prompt.get("current_state")
    return current_state if isinstance(current_state, dict) else {}


def _atomic_write_text(path: Path, content: str) -> None:
    temp_path = _prepare_atomic_text(path, content)
    try:
        _replace_prepared_text(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


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


def _atomic_publish_orchestrator_outputs(
    *,
    output_path: Path,
    output_content: str,
    report_path: Path,
    report_content: str,
) -> None:
    """Publish the derivative report first and authoritative ACK state last."""

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
def _exclusive_orchestrator_state_lock(output_path: Path):
    """Serialize read/build/publish so a stale refresh cannot undo an ACK."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _execution_contract() -> dict[str, Any]:
    return {
        "codex_may_execute": list(REPAIR_AUTOMATION_ALLOWED_ACTIONS),
        "code_change_loop": [
            "read_artifacts",
            "inspect_suggested_files",
            "edit_code",
            "edit_tests",
            "edit_runtime_contract_docs",
            "run_targeted_tests",
            "run_required_verification_commands",
            "commit",
            "sync_live_runtime",
        ],
        "commit_and_live_sync_required": True,
        "report_artifacts_are_runtime_drift": True,
        "quant_rabbit_code_may_call_model_api": False,
        "live_side_effects_allowed": [],
        "requires_explicit_operator_approval_for": list(REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS),
        "forbidden_direct_actions": list(REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS),
        "orders_closes_launchd_policy": (
            "Order send, order cancel, position close, and launchd load/reload must go through "
            "explicit operator approval or the existing gateway path. This orchestrator is never live permission."
        ),
    }


def _approval_boundary(execution_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "read_only_until_gateway_or_operator_approval": True,
        "live_side_effects_allowed": [],
        "requires_explicit_operator_approval_for": list(
            execution_contract.get("requires_explicit_operator_approval_for") or []
        ),
        "forbidden_direct_actions": list(execution_contract.get("forbidden_direct_actions") or []),
        "existing_gateway_paths": {
            "order_send": "LiveOrderGateway",
            "order_cancel": "LiveOrderGateway",
            "position_close": "PositionProtectionGateway",
            "launchd_load_or_reload": "operator_explicit_approval_after_preflight",
        },
        "quant_rabbit_code_may_call_model_api": False,
        "policy": execution_contract.get("orders_closes_launchd_policy"),
    }


def _loop_engineering_prompt(
    *,
    support: dict[str, Any],
    queue: list[dict[str, Any]],
    actionable: list[dict[str, Any]],
    approval_required: list[dict[str, Any]],
    waiting: list[dict[str, Any]],
    selected: dict[str, Any],
    status: str,
    trader_request: str,
    approval_boundary: dict[str, Any],
    proof_queue: dict[str, Any],
    as_board: dict[str, Any],
    portfolio_planner: dict[str, Any],
    live_order_request: dict[str, Any],
    broker_snapshot: dict[str, Any],
    previous_evidence_actions: list[dict[str, Any]],
    previous_current_state: dict[str, Any],
) -> dict[str, Any]:
    """Build the continuously updated prompt for the 5% campaign repair loop."""

    target = support.get("target") if isinstance(support.get("target"), dict) else {}
    guardian = support.get("guardian") if isinstance(support.get("guardian"), dict) else {}
    entry = support.get("entry_readiness") if isinstance(support.get("entry_readiness"), dict) else {}
    artifact_freshness = (
        entry.get("artifact_freshness") if isinstance(entry.get("artifact_freshness"), dict) else {}
    )
    acceptance = (
        support.get("profitability_acceptance")
        if isinstance(support.get("profitability_acceptance"), dict)
        else {}
    )
    target_firepower = (
        acceptance.get("target_firepower")
        if isinstance(acceptance.get("target_firepower"), dict)
        else {}
    )
    waiting_p0 = _p0_requests(waiting)
    current_state = {
        "orchestrator_status": status,
        "support_status": support.get("status"),
        "support_generated_at_utc": support.get("generated_at_utc"),
        "broker_snapshot_generated_at_utc": _broker_snapshot_timestamp(broker_snapshot),
        "support_blocker_codes": _support_blocker_codes(support.get("blockers")),
        "campaign_day_jst": target.get("campaign_day_jst"),
        "target_status": target.get("status"),
        "current_equity_raw": target.get("current_equity_raw") or target.get("current_equity_jpy"),
        "capital_flows_30d": target.get("capital_flows_30d"),
        "funding_adjusted_equity": target.get("funding_adjusted_equity"),
        "rolling_30d_multiplier_raw": target.get("rolling_30d_multiplier_raw"),
        "rolling_30d_multiplier_funding_adjusted": target.get(
            "rolling_30d_multiplier_funding_adjusted"
        ),
        "remaining_to_4x_raw": target.get("remaining_to_4x_raw"),
        "remaining_to_4x_funding_adjusted": target.get("remaining_to_4x_funding_adjusted"),
        "required_calendar_daily_return_funding_adjusted": target.get(
            "required_calendar_daily_return_funding_adjusted"
        ),
        "required_active_day_return_funding_adjusted": target.get(
            "required_active_day_return_funding_adjusted"
        ),
        "performance_basis": target.get("performance_basis"),
        "sizing_basis": target.get("sizing_basis"),
        "pace_state": target.get("pace_state"),
        "minimum_progress_pct": target.get("minimum_progress_pct"),
        "remaining_minimum_jpy": target.get("remaining_minimum_jpy"),
        "progress_pct": target.get("progress_pct"),
        "remaining_target_jpy": target.get("remaining_target_jpy"),
        "target_trades_per_day": target.get("target_trades_per_day"),
        "guardian_active": guardian.get("active"),
        "guardian_active_source": guardian.get("active_source"),
        "guardian_heartbeat_status": guardian.get("heartbeat_status"),
        "guardian_live_runtime_lock_active": guardian.get("live_runtime_lock_active"),
        "guardian_live_runtime_lock_command": guardian.get("live_runtime_lock_command"),
        "guardian_live_runtime_lock_age_seconds": guardian.get("live_runtime_lock_age_seconds"),
        "live_ready_lanes": entry.get("live_ready_lanes"),
        "guardian_blocked_lanes": entry.get("guardian_blocked_lanes"),
        "order_intents_stale_against_broker_snapshot": artifact_freshness.get(
            "order_intents_stale_against_broker_snapshot"
        ),
        "order_intents_generated_at_utc": artifact_freshness.get("order_intents_generated_at_utc"),
        "broker_snapshot_fetched_at_utc": artifact_freshness.get("broker_snapshot_fetched_at_utc"),
        "order_intents_staleness_seconds": artifact_freshness.get("order_intents_staleness_seconds"),
        "profitability_acceptance_status": acceptance.get("status"),
        "profitability_blockers": list(acceptance.get("blockers") or [])[:8],
        "operational_minimum_5pct_reachable": target_firepower.get(
            "operational_minimum_5pct_reachable"
        ),
        "audit_minimum_5pct_estimated_reachable": target_firepower.get(
            "minimum_5pct_estimated_reachable"
        ),
        "operational_blocker_codes": list(target_firepower.get("operational_blocker_codes") or [])[:12],
        "selected_request_code": selected.get("code") if selected else None,
        "selected_request_priority": selected.get("priority") if selected else None,
        "actionable_request_codes": _queue_codes(actionable),
        "approval_required_request_codes": _queue_codes(approval_required),
        "waiting_request_codes": _queue_codes(waiting),
        "waiting_p0_request_codes": _queue_codes(waiting_p0),
        "primary_waiting_p0_request_code": waiting_p0[0].get("code") if waiting_p0 else None,
        "selected_request_is_auxiliary_to_waiting_p0": bool(
            selected
            and waiting_p0
            and _priority_rank(selected.get("priority")) > _priority_rank(waiting_p0[0].get("priority"))
        ),
        "approval_required_details": _approval_required_details(approval_required),
    }
    current_state["execution_frontier"] = _execution_frontier_summary(
        entry=entry,
        queue=queue,
    )
    current_state["active_lane_evidence_work"] = _active_lane_evidence_work(entry)
    if current_state["active_lane_evidence_work"]:
        current_state["active_lane_action_code"] = current_state[
            "active_lane_evidence_work"
        ].get("action_code")
        current_state["active_lane_evidence_material_digest"] = current_state[
            "active_lane_evidence_work"
        ].get("material_digest")
    current_state["profitability_rca_summary"] = _profitability_rca_summary(
        acceptance=acceptance,
        target_firepower=target_firepower,
    )
    current_state["as_proof_state"] = _as_proof_state(
        proof_queue=proof_queue,
        as_board=as_board,
        portfolio_planner=portfolio_planner,
        live_order_request=live_order_request,
    )
    current_state.update(current_state["as_proof_state"])
    proof_queue_empty_reason = _proof_queue_empty_reason(current_state)
    if proof_queue_empty_reason:
        current_state["proof_queue_empty_reason"] = proof_queue_empty_reason
        fresh_evidence_actions = _next_evidence_actions_for_empty_proof_queue(
            proof_queue_empty_reason,
            current_state=current_state,
        )
        (
            current_state["next_evidence_actions"],
            evidence_action_progress,
        ) = _reconcile_evidence_action_progress(
            fresh_evidence_actions,
            previous_evidence_actions=previous_evidence_actions,
            previous_current_state=previous_current_state,
            current_state=current_state,
        )
        if evidence_action_progress:
            current_state["evidence_action_progress"] = evidence_action_progress
    elif previous_evidence_actions:
        _, evidence_action_progress = _reconcile_evidence_action_progress(
            [],
            previous_evidence_actions=previous_evidence_actions,
            previous_current_state=previous_current_state,
            current_state=current_state,
        )
        if evidence_action_progress:
            current_state["evidence_action_progress"] = evidence_action_progress
    artifact_contradictions = _artifact_contradictions(current_state)
    current_state["artifact_contradictions"] = artifact_contradictions
    current_state["artifact_contradiction_codes"] = [
        str(item.get("code")) for item in artifact_contradictions if item.get("code")
    ]
    current_hypothesis = _loop_current_hypothesis(
        actionable=actionable,
        approval_required=approval_required,
        waiting=waiting,
        selected=selected,
        current_state=current_state,
    )
    next_loop = _loop_next_steps(
        actionable=actionable,
        approval_required=approval_required,
        waiting=waiting,
        selected=selected,
    )
    if artifact_contradictions:
        next_loop.insert(
            0,
            "Resolve artifact contradictions before using blocker counts: "
            + "; ".join(str(item.get("message")) for item in artifact_contradictions if item.get("message")),
        )
    if _proof_queue_is_empty(current_state):
        next_loop.insert(
            0,
            "Treat the A/S proof queue as empty: no live permission can be created until "
            "as-live-ready-evidence-loop and as-4x-proof-path rebuild a non-negative, "
            "proof-ready candidate without guardian/profitability/gateway blockers.",
        )
    verification_commands = _loop_verification_commands(
        selected=selected,
        approval_required=approval_required,
        waiting=waiting,
    )
    if _proof_queue_is_empty(current_state):
        verification_commands = _dedupe_strings(
            [
                "PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop",
                "PYTHONPATH=src python3 -m quant_rabbit.cli as-4x-proof-path",
                *verification_commands,
            ]
        )
    if artifact_contradictions:
        refresh_commands: list[str] = []
        for item in artifact_contradictions:
            refresh_commands.extend(_dedupe_strings(item.get("refresh_commands")))
        verification_commands = _dedupe_strings([*refresh_commands, *verification_commands])
    anti_loop_rules = [
        "Do not treat audit-only 4x or +5% firepower as operational reachability unless operational_minimum_5pct_reachable is true and current LIVE_READY/proof/gateway gates agree.",
        "A proof_queue_count of 0, can_create_live_permission_count of 0, or gateway NO_LIVE_READY_INTENT is a no-send state, not a prompt to synthesize a receipt.",
        "If the selected actionable request is lower priority than a waiting P0 blocker, treat it as auxiliary evidence work, not as clearing the P0 or proving operational 5%.",
        "If fresher support state contradicts intent blocker counts, classify that blocker as artifact-stale and refresh the evidence packet before selecting repair work from it.",
        "Do not rerun profitability-acceptance as the fix unless an input artifact, gateway proof, or live evidence window changed first.",
        "If OANDA audit-only S5/M5 history is complete and replay cannot clear local TP proof, do not rerun validate/mine/package; wait for new local TP receipts, new forecast/candle evidence, or exact HARVEST live-grade promotion.",
        "Do not derive lot size from remaining_to_4x_funding_adjusted or remaining_minimum_jpy; sizing remains raw-NAV/risk/ATR/margin/gateway based.",
        "Do not lower MIN_PRODUCTION_LOT_UNITS, bypass MARGIN_TOO_THIN_FOR_MIN_LOT "
        "or LOSS_AND_MARGIN_TOO_THIN_FOR_MIN_LOT, synthesize PASS close evidence, "
        "or loosen protective market-structure guards without a failing regression and a positive-path test.",
        "Do not send orders, cancel orders, close positions, mutate launchd, or call model APIs from QuantRabbit code outside the existing gateway or explicit operator approval boundary.",
        "If the top item is waiting for live evidence, collect or wait for the named evidence; do not reimplement the same already-blocking guard.",
    ]
    self_review_questions = [
        "What single blocker currently prevents rolling 30d 4x pace or the +5% pace marker, and is it causal rather than merely frequent?",
        "Does the next action change broker state, launchd state, order state, or position state? If yes, which approved gateway or explicit operator approval covers it?",
        "Is any blocker contradicted by fresher support state, and did I refresh the stale artifact before treating it as causal?",
        "Do proof_queue_count, can_create_live_permission_count, and gateway_status all allow a future gateway path, or am I trying to route from zero proof?",
        "Am I trying to recover profit by increasing churn while capture economics is negative, or by preserving a TP-proven HARVEST shape?",
        "What evidence would prove this loop iteration worked, and which command will refresh that evidence?",
        "Is a TP-proven lane blocked only by margin or broker exposure, and would clearing it require broker-state change outside Codex?",
        "What is the strongest counterargument that the current blocker is actually a correct protective guardrail?",
    ]
    prompt_text = _render_loop_prompt_text(
        current_state=current_state,
        current_hypothesis=current_hypothesis,
        next_loop=next_loop,
        verification_commands=verification_commands,
        self_review_questions=self_review_questions,
        anti_loop_rules=anti_loop_rules,
        trader_request=trader_request,
    )
    return {
        "version": LOOP_ENGINEERING_PROMPT_VERSION,
        "objective": (
            "Drive QuantRabbit toward rolling 30-calendar-day 4x funding-adjusted equity by "
            "repeatedly selecting the highest-causal blocker, taking only approved code/evidence "
            "actions, and verifying that operational reachability improves without bypassing broker truth. "
            "+5% remains a pace marker, review trigger, and protection milestone rather than a forced-churn target."
        ),
        "trader_request": trader_request,
        "current_state": current_state,
        "current_hypothesis": current_hypothesis,
        "next_loop": next_loop,
        "verification_commands": verification_commands,
        "self_review_questions": self_review_questions,
        "anti_loop_rules": anti_loop_rules,
        "approval_boundary": approval_boundary,
        "prompt_text": prompt_text,
    }


def _queue_codes(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("code")) for item in items if item.get("code")]


def _p0_requests(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if _priority_rank(item.get("priority")) == 0]


def _approval_required_details(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for item in items[:5]:
        evidence = item.get("evidence_summary") if isinstance(item.get("evidence_summary"), dict) else {}
        examples = evidence.get("examples") if isinstance(evidence.get("examples"), list) else []
        compact_examples: list[dict[str, Any]] = []
        for example in examples[:3]:
            if not isinstance(example, dict):
                continue
            compact_examples.append(
                {
                    key: example.get(key)
                    for key in [
                        "trade_id",
                        "pair",
                        "side",
                        "units",
                        "owner",
                        "take_profit",
                        "stop_loss",
                        "unrealized_pl_jpy",
                    ]
                    if example.get(key) is not None
                }
            )
        detail = {
            "code": item.get("code"),
            "status": item.get("repair_status") or item.get("status"),
            "problem": item.get("problem"),
            "clearance_conditions": _dedupe_strings(item.get("clearance_conditions"))[:3],
            "examples": compact_examples,
        }
        details.append({key: value for key, value in detail.items() if value})
    return details


def _execution_frontier_summary(
    *,
    entry: dict[str, Any],
    queue: list[dict[str, Any]],
) -> dict[str, Any]:
    repair_frontier = (
        entry.get("repair_frontier") if isinstance(entry.get("repair_frontier"), list) else []
    )
    global_unlock_frontier = (
        entry.get("global_unlock_frontier")
        if isinstance(entry.get("global_unlock_frontier"), list)
        else []
    )
    remaining_blockers = (
        entry.get("repair_frontier_remaining_blockers")
        if isinstance(entry.get("repair_frontier_remaining_blockers"), list)
        else []
    )
    summary: dict[str, Any] = {
        "repair_frontier_lanes": len(repair_frontier),
        "repair_frontier_top_lanes": [
            _compact_frontier_lane(item)
            for item in repair_frontier[:3]
            if isinstance(item, dict)
        ],
        "repair_frontier_top_blockers": [
            _compact_frontier_blocker(item)
            for item in remaining_blockers[:5]
            if isinstance(item, dict)
        ],
        "global_unlock_frontier_lanes": len(global_unlock_frontier),
        "global_unlock_frontier_top_lanes": [
            _compact_frontier_lane(item)
            for item in global_unlock_frontier[:3]
            if isinstance(item, dict)
        ],
    }
    unknown_owner = _unknown_owner_context(queue)
    if unknown_owner:
        summary["unknown_owner_context"] = unknown_owner
    return summary


def _active_lane_evidence_work(entry: dict[str, Any]) -> dict[str, Any]:
    shortest = (
        entry.get("shortest_live_ready_path")
        if isinstance(entry.get("shortest_live_ready_path"), dict)
        else {}
    )
    active_path = (
        shortest.get("active_path")
        if isinstance(shortest.get("active_path"), dict)
        else {}
    )
    next_action = str(
        active_path.get("next_action")
        or shortest.get("first_next_step")
        or shortest.get("next_action")
        or ""
    ).strip()
    lane_id = str(active_path.get("lane_id") or shortest.get("lane_id") or "").strip()
    if not lane_id or not next_action:
        return {}
    if active_path.get("live_permission") is True or shortest.get("live_permission") is True:
        return {}

    blocker_codes = _dedupe_strings(
        list(active_path.get("blocker_codes") or [])
        + list(shortest.get("blocker_codes") or [])
    )
    action_code = _active_lane_action_code(next_action)
    suggested_commands = _active_lane_evidence_commands(action_code)
    suggested_command_steps = _active_lane_command_steps(suggested_commands)
    material_state = _active_lane_material_state(
        shortest=shortest,
        active_path=active_path,
        action_code=action_code,
        blocker_codes=blocker_codes,
        suggested_command_steps=suggested_command_steps,
    )
    material_digest = _canonical_sha256(material_state)
    work = {
        "status": ACTIVE_LANE_EVIDENCE_CANDIDATE_STATUS,
        "schema_version": "active_lane_evidence_work_v2",
        "source": "trader_support_bot.entry_readiness.shortest_live_ready_path",
        "lane_id": lane_id,
        "pair": active_path.get("pair") or shortest.get("pair"),
        "side": active_path.get("side") or shortest.get("side"),
        "method": active_path.get("method") or shortest.get("method"),
        "order_type": active_path.get("order_type") or shortest.get("order_type"),
        "current_status": shortest.get("status") or active_path.get("status"),
        "active_path_status": active_path.get("status"),
        "selection_basis": shortest.get("selection_basis"),
        "next_action": next_action,
        "action_code": action_code,
        "blocker_codes": blocker_codes[:24],
        "command_plan_available": bool(suggested_commands),
        "material_state": material_state,
        "material_digest": material_digest,
        "read_only": True,
        "live_side_effects": [],
        "live_permission_allowed": False,
    }
    return {key: value for key, value in work.items() if value not in (None, [], {})}


def _active_lane_action_code(next_action: str) -> str:
    """Return one current stage; future/follow-up tokens never become current work."""

    normalized = " ".join(str(next_action or "").split())
    upper = normalized.upper()

    # active_trader_contract can append a supplemental/parallel frontier lane
    # after the board-primary action. A concrete secondary action, including a
    # fired guardian trigger, must never hide the selected board-lane action.
    parallel_boundaries = [
        index
        for marker in (
            " PAIR THIS WITH FRONTIER EVIDENCE",
            " PAIR THIS WITH NON_EURUSD_LIVE_GRADE_FRONTIER EVIDENCE",
            " PARALLEL NON_EURUSD_LIVE_GRADE_FRONTIER EVIDENCE",
        )
        if (index := upper.find(marker)) >= 0
    ]
    if parallel_boundaries:
        primary_end = min(parallel_boundaries)
        normalized = normalized[:primary_end].rstrip()
        upper = upper[:primary_end].rstrip()

    # A fired guardian event explicitly invalidates the old wait token, which
    # remains in the prose only as a "do not repeat" warning.
    if (
        "CONTRACT_ADD_TRIGGER FIRED" in upper
        or "DO NOT REPEAT WAIT_FOR_RANGE_RAIL_RECHECK" in upper
    ) and (
        ACTIVE_LANE_REPRICE_RANGE_ACTION in upper
        or "REPRICE THE RANGE_ROTATION COUNTERPART" in upper
        or "REPRICE RANGE_ROTATION COUNTERPART" in upper
    ):
        return ACTIVE_LANE_REPRICE_RANGE_ACTION

    safe_match = re.search(
        r"\bNEXT\s+SAFE\s+(?:TUNING\s+)?ACTION\s+(?:IS|=|:)\s*([A-Z][A-Z0-9_-]*)",
        upper,
    )
    if safe_match is not None:
        prefix = upper[max(0, safe_match.start() - 32) : safe_match.start()]
        if "FOLLOW-UP" in prefix or "FOLLOW UP" in prefix or "LATER" in prefix:
            safe_match = None
    wait_index = upper.find(ACTIVE_LANE_WAIT_ACTION)
    if wait_index >= 0 and (safe_match is None or wait_index < safe_match.start()):
        return ACTIVE_LANE_WAIT_ACTION
    if safe_match is not None:
        return _normalize_active_lane_action_code(safe_match.group(1))
    generic_matches = [
        match
        for pattern in (
            r"\bNEXT\s+(?:TRADE[-_\s]+ENABLING\s+)?ACTION\s+(?:IS|=|:)\s*([A-Z][A-Z0-9_-]*)",
            r"\bACTION\s+IS\s+([A-Z][A-Z0-9_-]*)",
        )
        if (match := re.search(pattern, upper)) is not None
    ]
    if generic_matches:
        explicit_match = min(generic_matches, key=lambda item: item.start())
        return _normalize_active_lane_action_code(explicit_match.group(1))
    if (
        ACTIVE_LANE_REPRICE_RANGE_ACTION in upper
        or "REPRICE THE RANGE_ROTATION COUNTERPART" in upper
        or "REPRICE RANGE_ROTATION COUNTERPART" in upper
    ):
        return ACTIVE_LANE_REPRICE_RANGE_ACTION
    if ACTIVE_LANE_RANGE_READY_PROOF_ACTION in upper:
        return ACTIVE_LANE_RANGE_READY_PROOF_ACTION
    if (
        ACTIVE_LANE_EXACT_TP_PROOF_ACTION in upper
        or "TP_PROOF_COLLECTION" in upper
        or "COLLECT EXACT LOCAL TAKE_PROFIT_ORDER PROOF" in upper
        or "COLLECT EXACT TAKE_PROFIT_ORDER PROOF" in upper
    ):
        return ACTIVE_LANE_EXACT_TP_PROOF_ACTION
    if ACTIVE_LANE_TRIGGER_PROOF_ACTION in upper:
        return ACTIVE_LANE_TRIGGER_PROOF_ACTION
    if ACTIVE_LANE_RANGE_RAIL_ACTION in upper or "RANGE RAIL GEOMETRY REPAIR" in upper:
        return ACTIVE_LANE_RANGE_RAIL_ACTION
    if (
        ACTIVE_LANE_FORECAST_PATTERN_ACTION in upper
        or "FORECAST-PATTERN REFRESH" in upper
        or "REFRESH_FORECAST_RANGE_BOX" in upper
    ):
        return ACTIVE_LANE_FORECAST_PATTERN_ACTION
    if (
        "ENTRY_FREQUENCY_RECOVERY" in upper
        or "ENTRY-FREQUENCY" in upper
        or "ENTRY DROUGHT" in upper
    ):
        return ACTIVE_LANE_ENTRY_DROUGHT_ACTION
    if ACTIVE_LANE_VERIFY_PROJECTIONS_ACTION in upper:
        return ACTIVE_LANE_VERIFY_PROJECTIONS_ACTION
    return ACTIVE_LANE_UNRESOLVED_ACTION


def _normalize_active_lane_action_code(action_code: str) -> str:
    normalized = str(action_code or "").strip().upper().replace("-", "_")
    aliases = {
        "REFRESH_FORECAST_RANGE_BOX": ACTIVE_LANE_FORECAST_PATTERN_ACTION,
    }
    return aliases.get(normalized, normalized or ACTIVE_LANE_UNRESOLVED_ACTION)


def _active_lane_evidence_commands(action_code: str) -> list[str]:
    stage_commands = {
        ACTIVE_LANE_VERIFY_PROJECTIONS_ACTION: [
            "PYTHONPATH=src python3 -m quant_rabbit.cli verify-projections",
            "PYTHONPATH=src python3 -m quant_rabbit.cli forecast-pattern-refresh",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
        ],
        ACTIVE_LANE_TRIGGER_PROOF_ACTION: [
            "PYTHONPATH=src python3 -m quant_rabbit.cli verify-projections",
            "PYTHONPATH=src python3 -m quant_rabbit.cli forecast-pattern-refresh",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
        ],
        ACTIVE_LANE_ENTRY_DROUGHT_ACTION: [
            "PYTHONPATH=src python3 -m quant_rabbit.cli entry-frequency-recovery",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
        ],
        ACTIVE_LANE_FORECAST_PATTERN_ACTION: [
            "PYTHONPATH=src python3 -m quant_rabbit.cli forecast-pattern-refresh",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
        ],
        ACTIVE_LANE_RANGE_RAIL_ACTION: [
            "PYTHONPATH=src python3 -m quant_rabbit.cli range-rail-geometry-repair",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
            "PYTHONPATH=src python3 -m quant_rabbit.cli guardian-trigger-contract",
            "PYTHONPATH=src python3 -m quant_rabbit.cli guardian-event-router",
        ],
        ACTIVE_LANE_EXACT_TP_PROOF_ACTION: [
            "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
            "PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop",
            "PYTHONPATH=src python3 -m quant_rabbit.cli as-4x-proof-path",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-opportunity-board",
            "PYTHONPATH=src python3 -m quant_rabbit.cli non-eurusd-proof-lane-mapper",
            "PYTHONPATH=src python3 -m quant_rabbit.cli non-eurusd-live-grade-frontier",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
        ],
        ACTIVE_LANE_RANGE_READY_PROOF_ACTION: [
            "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
            "PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop",
            "PYTHONPATH=src python3 -m quant_rabbit.cli as-4x-proof-path",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-opportunity-board",
            "PYTHONPATH=src python3 -m quant_rabbit.cli non-eurusd-proof-lane-mapper",
            "PYTHONPATH=src python3 -m quant_rabbit.cli non-eurusd-live-grade-frontier",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
        ],
        ACTIVE_LANE_REPRICE_RANGE_ACTION: [
            "PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json",
            "PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10",
            "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-opportunity-board",
            "PYTHONPATH=src python3 -m quant_rabbit.cli non-eurusd-proof-lane-mapper",
            "PYTHONPATH=src python3 -m quant_rabbit.cli non-eurusd-live-grade-frontier",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
            "PYTHONPATH=src python3 -m quant_rabbit.cli entry-frequency-recovery",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
            "PYTHONPATH=src python3 -m quant_rabbit.cli forecast-pattern-refresh",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
            "PYTHONPATH=src python3 -m quant_rabbit.cli range-rail-geometry-repair",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
        ],
        ACTIVE_LANE_CURRENT_INTENT_REGEN_ACTION: [
            "PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json",
            "PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10",
            "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-opportunity-board",
            "PYTHONPATH=src python3 -m quant_rabbit.cli non-eurusd-proof-lane-mapper",
            "PYTHONPATH=src python3 -m quant_rabbit.cli non-eurusd-live-grade-frontier",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
        ],
        ACTIVE_LANE_NON_MARKET_PROOF_ROUTE_ACTION: [
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-opportunity-board",
            "PYTHONPATH=src python3 -m quant_rabbit.cli non-eurusd-proof-lane-mapper",
            "PYTHONPATH=src python3 -m quant_rabbit.cli non-eurusd-live-grade-frontier",
            "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
        ],
    }
    commands = list(stage_commands.get(action_code, []))
    if not commands:
        return []
    commands.extend(
        [
            "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
            ACTIVE_LANE_ACK_COMMAND_TEMPLATE,
            "PYTHONPATH=src python3 -m quant_rabbit.cli trader-goal-loop-orchestrator",
        ]
    )
    # The initial support refresh in proof-collection stages feeds A/S. The
    # common trailing support refresh is intentionally repeated after board /
    # contract updates so repair and goal-loop consume the terminal state.
    return commands


def _active_lane_command_steps(commands: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "command": command,
            "required": True,
            "ok_rcs": [0, 2] if "quant_rabbit.cli trader-support-bot" in command else [0],
        }
        for command in commands
    ]


def _active_lane_material_state(
    *,
    shortest: dict[str, Any],
    active_path: dict[str, Any],
    action_code: str,
    blocker_codes: list[str],
    suggested_command_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    material_scalars = [
        "status",
        "current_status",
        "lane_id",
        "pair",
        "side",
        "method",
        "order_type",
        "target_shape",
        "selection_basis",
        "contract_status",
        "board_status",
        "frontier_status",
        "expected_edge_jpy",
        "spread_status",
        "forecast_status",
        "forecast_box_status",
        "loss_budget_status",
        "proof_status",
        "replay_status",
        "risk_status",
        "rail_status",
        "counterpart_geometry_status",
        "trigger_projection_status",
        "guardian_trigger_status",
    ]
    state: dict[str, Any] = {
        "schema_version": "active_lane_material_state_v1",
        "action_code": action_code,
        "blocker_codes": sorted(set(blocker_codes)),
        "blocker_groups": sorted(
            set(_dedupe_strings(shortest.get("blocker_groups")))
        ),
        "command_plan_digest": _canonical_sha256(suggested_command_steps),
        "shortest_path": {
            key: shortest.get(key)
            for key in material_scalars
            if shortest.get(key) is not None
        },
        "active_path": {
            key: active_path.get(key)
            for key in material_scalars
            if active_path.get(key) is not None
        },
    }
    for key in ("local_tp_proof", "tp_proof", "success_condition", "evidence_watermark"):
        value = active_path.get(key)
        if value not in (None, [], {}):
            state["active_path"][key] = _stable_material_value(value)
    return state


def _stable_material_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _stable_material_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
            if str(key).lower()
            not in {
                "generated_at_utc",
                "fetched_at_utc",
                "updated_at_utc",
                "created_at_utc",
                "age_seconds",
                "path",
                "sha256",
            }
        }
    if isinstance(value, list):
        normalized = [_stable_material_value(item) for item in value]
        if all(not isinstance(item, (dict, list)) for item in normalized):
            return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))
        return normalized
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compact_frontier_lane(item: dict[str, Any]) -> dict[str, Any]:
    tp_proof = item.get("tp_proof") if isinstance(item.get("tp_proof"), dict) else {}
    compact = {
        "lane_id": item.get("lane_id"),
        "pair": item.get("pair"),
        "side": item.get("side"),
        "method": item.get("method"),
        "order_type": item.get("order_type"),
        "status": item.get("status"),
        "repair_mode": item.get("repair_mode"),
        "reward_jpy": item.get("reward_jpy"),
        "risk_jpy": item.get("risk_jpy"),
        "remaining_blocker_codes": (
            item.get("remaining_blocker_codes_after_guardian_and_repair_exemption")
            or item.get("remaining_blocker_codes_after_global_unlock")
            or item.get("blocker_codes")
            or item.get("global_blocker_codes")
        ),
        "tp_proof": {
            key: tp_proof.get(key)
            for key in [
                "positive_rotation_mode",
                "capture_take_profit_scope",
                "capture_take_profit_scope_key",
                "capture_take_profit_trades",
                "capture_take_profit_losses",
                "positive_rotation_pessimistic_expectancy_jpy",
            ]
            if tp_proof.get(key) is not None
        },
    }
    return {key: value for key, value in compact.items() if value not in (None, [], {})}


def _compact_frontier_blocker(item: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "code": item.get("code"),
        "count": item.get("count"),
        "example_lane_ids": item.get("example_lane_ids"),
        "co_blocker_codes": item.get("co_blocker_codes"),
        "reward_jpy": item.get("reward_jpy"),
    }
    return {key: value for key, value in compact.items() if value not in (None, [], {})}


def _unknown_owner_context(queue: list[dict[str, Any]]) -> dict[str, Any]:
    request = next(
        (
            item
            for item in queue
            if str(item.get("code") or "") == "REVIEW_UNKNOWN_OWNER_EXPOSURE"
        ),
        None,
    )
    if not request:
        return {}
    evidence = (
        request.get("evidence_summary")
        if isinstance(request.get("evidence_summary"), dict)
        else {}
    )
    examples = evidence.get("examples") if isinstance(evidence.get("examples"), list) else []
    compact_examples: list[dict[str, Any]] = []
    for example in examples[:3]:
        if not isinstance(example, dict):
            continue
        compact_examples.append(
            {
                key: example.get(key)
                for key in [
                    "trade_id",
                    "pair",
                    "side",
                    "units",
                    "owner",
                    "take_profit",
                    "stop_loss",
                    "unrealized_pl_jpy",
                ]
                if example.get(key) is not None
            }
        )
    context = {
        "status": request.get("repair_status") or request.get("status"),
        "unknown_owner_positions": evidence.get("unknown_owner_positions"),
        "margin_available_jpy": evidence.get("margin_available_jpy"),
        "nav_jpy": evidence.get("nav_jpy"),
        "examples": compact_examples,
    }
    return {key: value for key, value in context.items() if value not in (None, [], {})}


def _approval_action_summary(item: dict[str, Any]) -> str:
    details = _approval_required_details([item])
    if not details:
        return f"Run read-only preflight/status checks for {item.get('code')}."
    detail = details[0]
    code = detail.get("code") or item.get("code")
    example_text = _format_approval_examples(detail.get("examples"))
    clearance = _format_approval_clearance(detail.get("clearance_conditions"))
    suffix = " ".join(part for part in [example_text, clearance] if part)
    if suffix:
        return f"Run read-only preflight/status checks for {code}: {suffix}"
    return f"Run read-only preflight/status checks for {code}."


def _format_approval_examples(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return ""
    parts: list[str] = []
    for example in value[:3]:
        if not isinstance(example, dict):
            continue
        trade = example.get("trade_id")
        pair = example.get("pair")
        side = example.get("side")
        units = example.get("units")
        owner = example.get("owner")
        tp = example.get("take_profit")
        sl = example.get("stop_loss")
        pl = example.get("unrealized_pl_jpy")
        fragment = " ".join(str(item) for item in [pair, side] if item)
        if units is not None:
            fragment = f"{fragment} {units}u".strip()
        if trade:
            fragment = f"trade_id={trade} {fragment}".strip()
        if owner:
            fragment = f"{fragment} owner={owner}".strip()
        if tp is not None:
            fragment = f"{fragment} TP={tp}".strip()
        if sl is not None:
            fragment = f"{fragment} SL={sl}".strip()
        if pl is not None:
            fragment = f"{fragment} unrealized_jpy={pl}".strip()
        if fragment:
            parts.append(fragment)
    if not parts:
        return ""
    return "approval target(s): " + "; ".join(parts) + "."


def _format_approval_clearance(value: Any) -> str:
    conditions = [item.rstrip(" .") for item in _dedupe_strings(value)]
    if not conditions:
        return ""
    return "clearance: " + " / ".join(conditions[:2]) + "."


def _support_blocker_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    codes: list[str] = []
    for item in value:
        if isinstance(item, dict):
            code = item.get("code")
        else:
            code = item
        if code:
            codes.append(str(code))
    return _dedupe_strings(codes)


def _prefixed_blocker_codes(value: Any) -> list[str]:
    codes: list[str] = []
    for item in _dedupe_strings(value):
        code = item.split(":", 1)[0].strip()
        if code:
            codes.append(code)
    return _dedupe_strings(codes)


def _artifact_contradictions(current_state: dict[str, Any]) -> list[dict[str, Any]]:
    contradictions: list[dict[str, Any]] = []
    guardian_active = current_state.get("guardian_active") is True
    guardian_blocked_lanes = _optional_positive_int(current_state.get("guardian_blocked_lanes"))
    if guardian_active and guardian_blocked_lanes > 0:
        contradictions.append(
            {
                "code": "GUARDIAN_ACTIVE_BUT_INTENTS_CARRY_GUARDIAN_BLOCKERS",
                "message": (
                    "support proves position guardian active, but order_intents still carries "
                    f"{guardian_blocked_lanes} guardian-inactive blocker(s); treat those counts "
                    "as stale until intents/support/orchestrator are regenerated from the same evidence packet"
                ),
                "refresh_commands": [
                    "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts",
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator",
                ],
            }
        )
    if current_state.get("order_intents_stale_against_broker_snapshot") is True:
        contradictions.append(
            {
                "code": "ORDER_INTENTS_STALE_AGAINST_BROKER_SNAPSHOT",
                "message": (
                    "broker_snapshot is fresher than order_intents "
                    f"(broker={current_state.get('broker_snapshot_fetched_at_utc')}, "
                    f"intents={current_state.get('order_intents_generated_at_utc')}); "
                    "refresh generated intents before treating live-ready counts or repair-frontier blockers as causal"
                ),
                "refresh_commands": [
                    "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts",
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator",
                ],
            }
        )
    return contradictions


def _optional_positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _loop_current_hypothesis(
    *,
    actionable: list[dict[str, Any]],
    approval_required: list[dict[str, Any]],
    waiting: list[dict[str, Any]],
    selected: dict[str, Any],
    current_state: dict[str, Any],
) -> str:
    if selected:
        waiting_p0_codes = current_state.get("waiting_p0_request_codes")
        if (
            current_state.get("selected_request_is_auxiliary_to_waiting_p0")
            and isinstance(waiting_p0_codes, list)
            and waiting_p0_codes
        ):
            primary_p0 = str(waiting_p0_codes[0])
            problem = str(selected.get("problem") or "").rstrip(".")
            return (
                f"The causal P0 blocker remains {primary_p0} waiting on evidence; "
                f"the selected actionable loop is auxiliary work on {selected.get('code')}: "
                f"{problem}. Do not treat this as operational 5% proof until "
                "the waiting P0 clearance evidence changes."
            )
        return (
            f"The next useful loop is implementation work on {selected.get('code')}: "
            f"{selected.get('problem')}"
        )
    if approval_required:
        item = approval_required[0]
        return (
            f"The next blocker is approval-bound, not Codex code work: {item.get('code')} "
            f"({item.get('problem')})."
        )
    if waiting:
        item = waiting[0]
        return (
            f"The next blocker is evidence-window work: {item.get('code')} "
            f"({item.get('repair_status')}). Do not edit gates until the clearance evidence changes."
        )
    if current_state.get("live_ready_lanes") == 0:
        return "The target is open but no LIVE_READY lanes exist; refresh support evidence and rank live blocker families."
    return "No repair request is queued; continue broker-truth refresh, support bot generation, and acceptance checks."


def _loop_next_steps(
    *,
    actionable: list[dict[str, Any]],
    approval_required: list[dict[str, Any]],
    waiting: list[dict[str, Any]],
    selected: dict[str, Any],
) -> list[str]:
    if selected:
        steps: list[str] = []
        waiting_p0 = _p0_requests(waiting)
        if (
            waiting_p0
            and _priority_rank(selected.get("priority"))
            > _priority_rank(waiting_p0[0].get("priority"))
        ):
            steps.append(
                "Keep the waiting P0 blockers in scope: "
                f"{', '.join(_queue_codes(waiting_p0))}. "
                "The selected work is auxiliary until their clearance evidence changes."
            )
        steps.extend(
            [
                "Read the selected request evidence and suggested files before editing.",
                "Implement the smallest code/test/doc change that directly clears the selected clearance condition.",
                "Run targeted tests and listed verification commands; do not count report reruns as repair unless the underlying evidence changed.",
                "Commit with Codex attribution, sync live runtime, and verify live HEAD matches the promoted commit.",
            ]
        )
        return steps
    if approval_required:
        first = approval_required[0]
        return [
            _approval_action_summary(first),
            "Do not load/reload launchd, send/cancel orders, or close positions without explicit operator approval or an existing gateway path.",
            "After approval-bound state changes externally, rerun trader-support-bot and trader-repair-orchestrator to regenerate this prompt.",
        ]
    if waiting:
        first = waiting[0]
        return [
            f"Treat {first.get('code')} as waiting for evidence, not implementation.",
            "Run only the listed read-only verification/evidence commands and compare the new artifact against the clearance condition.",
            "If the same blocker repeats unchanged, move to the next actionable request or report the live-evidence wait instead of rewriting the same guard.",
        ]
    return [
        "Refresh trader-support-bot and trader-repair-orchestrator.",
        "If the target remains open with no LIVE_READY lanes, rank blocker families before proposing code.",
    ]


def _loop_verification_commands(
    *,
    selected: dict[str, Any],
    approval_required: list[dict[str, Any]],
    waiting: list[dict[str, Any]],
) -> list[str]:
    raw: list[str] = []
    if selected:
        raw.extend(_dedupe_strings(selected.get("final_verification_commands")))
    else:
        for item in [*approval_required[:2], *waiting[:3]]:
            raw.extend(_dedupe_strings(item.get("verification_commands")))
    raw.extend(
        [
            "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
            "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator",
        ]
    )
    return _dedupe_strings(raw)


def _render_loop_prompt_text(
    *,
    current_state: dict[str, Any],
    current_hypothesis: str,
    next_loop: list[str],
    verification_commands: list[str],
    self_review_questions: list[str],
    anti_loop_rules: list[str],
    trader_request: str,
) -> str:
    lines = [
        "QuantRabbit loop engineering prompt:",
        "",
        "Goal: pursue rolling 30d/monthly 4x funding-adjusted equity; use +5% daily as a pace/review/protection marker, not as a promise of market returns or a lot-sizing shortcut.",
        f"Trader request: {trader_request or '(none)'}",
        f"State: orchestrator={current_state.get('orchestrator_status')}, "
        f"target={current_state.get('target_status')}, live_ready={current_state.get('live_ready_lanes')}, "
        f"proof_queue={current_state.get('proof_queue_count')}, "
        f"live_permission_candidates={current_state.get('can_create_live_permission_count')}, "
        f"rejected_proof_candidates={current_state.get('rejected_proof_candidate_count')}, "
        f"gateway={current_state.get('gateway_status')}, "
        f"guardian_active={current_state.get('guardian_active')}, "
        f"guardian_lock={current_state.get('guardian_live_runtime_lock_active')}, "
        f"operational_5pct={current_state.get('operational_minimum_5pct_reachable')}, "
        f"audit_5pct={current_state.get('audit_minimum_5pct_estimated_reachable')}.",
        "4x target math: "
        f"funding_adjusted_equity={current_state.get('funding_adjusted_equity')}, "
        f"current_equity_raw={current_state.get('current_equity_raw')}, "
        f"capital_flows_30d={current_state.get('capital_flows_30d')}, "
        f"rolling_30d_multiplier_funding_adjusted={current_state.get('rolling_30d_multiplier_funding_adjusted')}, "
        f"remaining_to_4x_funding_adjusted={current_state.get('remaining_to_4x_funding_adjusted')}, "
        f"required_calendar_daily_return_funding_adjusted={current_state.get('required_calendar_daily_return_funding_adjusted')}, "
        f"required_active_day_return_funding_adjusted={current_state.get('required_active_day_return_funding_adjusted')}, "
        f"pace_state={current_state.get('pace_state')}, "
        f"performance_basis={current_state.get('performance_basis')}, "
        f"sizing_basis={current_state.get('sizing_basis')}.",
        f"Queue: selected={current_state.get('selected_request_code')}, "
        f"waiting_p0={', '.join(current_state.get('waiting_p0_request_codes') or []) or '(none)'}, "
        f"support_blockers={', '.join(current_state.get('support_blocker_codes') or []) or '(none)'}.",
        "A/S proof state: "
        f"{_render_as_proof_state_for_prompt(current_state.get('as_proof_state'))}.",
        "A/S proof empty reason: "
        f"{_render_proof_queue_empty_reason_for_prompt(current_state.get('proof_queue_empty_reason'))}.",
        "Next evidence actions: "
        f"{_render_next_evidence_actions_for_prompt(current_state.get('next_evidence_actions'))}.",
        "Execution frontier: "
        f"{_render_execution_frontier_for_prompt(current_state.get('execution_frontier'))}.",
        "Profitability RCA: "
        f"{_render_profitability_rca_for_prompt(current_state.get('profitability_rca_summary'))}.",
        "Approval required details: "
        f"{_render_approval_details_for_prompt(current_state.get('approval_required_details'))}.",
        "Artifact contradictions: "
        f"{', '.join(current_state.get('artifact_contradiction_codes') or []) or '(none)'}.",
        f"Hypothesis: {current_hypothesis}",
        "",
        "Next loop:",
    ]
    lines.extend(f"- {item}" for item in next_loop)
    lines.extend(["", "Self-review questions:"])
    lines.extend(f"- {item}" for item in self_review_questions)
    lines.extend(["", "Anti-loop rules:"])
    lines.extend(f"- {item}" for item in anti_loop_rules)
    lines.extend(["", "Verification commands:"])
    lines.extend(f"- {item}" for item in verification_commands)
    return "\n".join(lines)


def _render_as_proof_state_for_prompt(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "(missing)"
    parts = [
        _format_optional_count("queue", value.get("proof_queue_count")),
        _format_optional_count("proof_ready", value.get("proof_ready_count")),
        _format_optional_count("live_permission", value.get("can_create_live_permission_count")),
        _format_optional_count("rejected", value.get("rejected_proof_candidate_count")),
    ]
    for key, label in [
        ("as_live_ready_path_exists", "as_path"),
        ("proof_normal_routing_status", "routing"),
        ("proof_routing_allowed", "routing_allowed"),
        ("portfolio_status", "portfolio"),
        ("portfolio_can_reach_4x_now", "can_reach_4x_now"),
        ("gateway_status", "gateway"),
        ("proof_primary_blocker", "primary_blocker"),
    ]:
        if value.get(key) is not None:
            parts.append(f"{label}={value.get(key)}")
    blockers = _dedupe_strings(value.get("proof_global_blockers"))[:4]
    if blockers:
        parts.append("global_blockers=" + ",".join(blockers))
    return "; ".join(part for part in parts if part) or "(missing)"


def _render_proof_queue_empty_reason_for_prompt(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "(none)"
    categories: list[str] = []
    for item in value.get("categories", []):
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        reason_code = item.get("reason_code")
        status = item.get("status")
        if category:
            freshness = item.get("freshness") if isinstance(item.get("freshness"), dict) else {}
            freshness_status = freshness.get("status")
            categories.append(
                "/".join(
                    str(part)
                    for part in [
                        category,
                        status,
                        reason_code,
                        f"rank={item.get('causal_rank')}",
                        f"depth={item.get('blocking_depth')}",
                        freshness_status,
                    ]
                    if part
                )
            )
    parts = [
        f"status={value.get('status')}",
        f"primary={value.get('primary_category')}",
        f"primary_rank={value.get('primary_causal_rank')}",
        f"primary_depth={value.get('primary_blocking_depth')}",
        _format_optional_count("queue", value.get("proof_queue_count")),
        _format_optional_count("live_permission", value.get("can_create_live_permission_count")),
    ]
    if categories:
        parts.append("categories=" + " | ".join(categories))
    return "; ".join(part for part in parts if part)


def _render_next_evidence_actions_for_prompt(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "(none)"
    actions: list[str] = []
    for item in value[:6]:
        if not isinstance(item, dict):
            continue
        action_id = item.get("action_id")
        category = item.get("category")
        if action_id:
            text = str(action_id)
            if category:
                text += f"[{category}]"
            evaluation = (
                item.get("success_condition_evaluation")
                if isinstance(item.get("success_condition_evaluation"), dict)
                else {}
            )
            if evaluation.get("status"):
                text += f":{evaluation.get('status')}"
            actions.append(text)
    return ", ".join(actions) if actions else "(none)"


def _render_execution_frontier_for_prompt(value: Any) -> str:
    if not isinstance(value, dict):
        return "(none)"
    parts: list[str] = []
    repair_lanes = value.get("repair_frontier_top_lanes")
    if isinstance(repair_lanes, list) and repair_lanes:
        top = repair_lanes[0] if isinstance(repair_lanes[0], dict) else {}
        lane_id = top.get("lane_id")
        blockers = ", ".join(_dedupe_strings(top.get("remaining_blocker_codes")))
        tp_proof = top.get("tp_proof") if isinstance(top.get("tp_proof"), dict) else {}
        proof_bits = [
            str(tp_proof.get("positive_rotation_mode") or ""),
            _format_optional_count("tp_trades", tp_proof.get("capture_take_profit_trades")),
        ]
        proof_text = "/".join(part for part in proof_bits if part)
        detail = f"repair_top={lane_id}" if lane_id else "repair_top=(unknown)"
        if blockers:
            detail += f" blocked_by={blockers}"
        if proof_text:
            detail += f" proof={proof_text}"
        parts.append(detail)
    blocker_rows = value.get("repair_frontier_top_blockers")
    if isinstance(blocker_rows, list) and blocker_rows:
        blocker_parts: list[str] = []
        for item in blocker_rows[:3]:
            if not isinstance(item, dict) or not item.get("code"):
                continue
            count = item.get("count")
            examples = ", ".join(_dedupe_strings(item.get("example_lane_ids"))[:2])
            co_blockers = ", ".join(_dedupe_strings(item.get("co_blocker_codes"))[:3])
            text = str(item.get("code"))
            if count is not None:
                text += f"({count})"
            if examples:
                text += f": {examples}"
            if co_blockers:
                text += f" co_blocked_by={co_blockers}"
            blocker_parts.append(text)
        if blocker_parts:
            parts.append("frontier_blockers=" + " | ".join(blocker_parts))
    global_lanes = value.get("global_unlock_frontier_top_lanes")
    if isinstance(global_lanes, list) and global_lanes:
        top = global_lanes[0] if isinstance(global_lanes[0], dict) else {}
        lane_id = top.get("lane_id")
        tp_proof = top.get("tp_proof") if isinstance(top.get("tp_proof"), dict) else {}
        scope = tp_proof.get("capture_take_profit_scope")
        if lane_id:
            parts.append(f"global_unlock_top={lane_id} tp_scope={scope or 'unknown'}")
    unknown = value.get("unknown_owner_context")
    if isinstance(unknown, dict) and unknown:
        unknown_parts = [
            _format_optional_count("unknown_owner_positions", unknown.get("unknown_owner_positions")),
            _format_optional_count("margin_available_jpy", unknown.get("margin_available_jpy")),
        ]
        examples = unknown.get("examples") if isinstance(unknown.get("examples"), list) else []
        if examples and isinstance(examples[0], dict):
            example = examples[0]
            fragment = " ".join(
                str(item)
                for item in [
                    example.get("pair"),
                    example.get("side"),
                    f"{example.get('units')}u" if example.get("units") is not None else None,
                    f"trade_id={example.get('trade_id')}" if example.get("trade_id") else None,
                ]
                if item
            )
            if fragment:
                unknown_parts.append(f"example={fragment}")
        text = ", ".join(part for part in unknown_parts if part)
        if text:
            parts.append(text)
    return "; ".join(parts) if parts else "(none)"


def _profitability_rca_summary(
    *,
    acceptance: dict[str, Any],
    target_firepower: dict[str, Any],
) -> dict[str, Any]:
    capture = (
        acceptance.get("capture_economics")
        if isinstance(acceptance.get("capture_economics"), dict)
        else {}
    )
    overall = capture.get("overall") if isinstance(capture.get("overall"), dict) else {}
    take_profit = (
        capture.get("take_profit") if isinstance(capture.get("take_profit"), dict) else {}
    )
    market_close = (
        capture.get("market_close") if isinstance(capture.get("market_close"), dict) else {}
    )
    summary = {
        "profitability_acceptance_status": acceptance.get("status"),
        "capture_economics_status": capture.get("status"),
        "overall_expectancy_jpy_per_trade": overall.get("expectancy_jpy_per_trade"),
        "overall_net_jpy": overall.get("net_jpy"),
        "overall_trades": overall.get("trades"),
        "take_profit_expectancy_jpy_per_trade": take_profit.get("expectancy_jpy_per_trade"),
        "take_profit_net_jpy": take_profit.get("net_jpy"),
        "take_profit_trades": take_profit.get("trades"),
        "market_close_expectancy_jpy_per_trade": market_close.get("expectancy_jpy_per_trade"),
        "market_close_net_jpy": market_close.get("net_jpy"),
        "market_close_trades": market_close.get("trades"),
        "tp_proven_market_close_leak_segments": capture.get(
            "tp_proven_market_close_leak_segments"
        ),
        "acceptance_blocker_codes": _prefixed_blocker_codes(acceptance.get("blockers"))[:8],
        "operational_minimum_5pct_reachable": target_firepower.get(
            "operational_minimum_5pct_reachable"
        ),
        "audit_minimum_5pct_estimated_reachable": target_firepower.get(
            "minimum_5pct_estimated_reachable"
        ),
        "operational_blocker_codes": list(
            target_firepower.get("operational_blocker_codes") or []
        )[:8],
    }
    return {key: value for key, value in summary.items() if value not in (None, [], {})}


def _as_proof_state(
    *,
    proof_queue: dict[str, Any],
    as_board: dict[str, Any],
    portfolio_planner: dict[str, Any],
    live_order_request: dict[str, Any],
) -> dict[str, Any]:
    proof_summary = (
        proof_queue.get("summary") if isinstance(proof_queue.get("summary"), dict) else {}
    )
    board_firepower = (
        as_board.get("firepower_board_summary")
        if isinstance(as_board.get("firepower_board_summary"), dict)
        else {}
    )
    board_blocker = (
        as_board.get("exact_blocker_preventing_live_ready")
        if isinstance(as_board.get("exact_blocker_preventing_live_ready"), dict)
        else {}
    )
    planner_summary = (
        portfolio_planner.get("summary")
        if isinstance(portfolio_planner.get("summary"), dict)
        else {}
    )
    state = {
        "proof_queue_generated_at_utc": proof_queue.get("generated_at_utc"),
        "as_board_generated_at_utc": as_board.get("generated_at_utc"),
        "portfolio_planner_generated_at_utc": portfolio_planner.get("generated_at_utc"),
        "proof_queue_count": _first_present(
            proof_summary.get("queue_count"),
            len(proof_queue.get("queue")) if isinstance(proof_queue.get("queue"), list) else None,
        ),
        "proof_ready_count": _first_present(
            proof_summary.get("proof_ready_count"),
            planner_summary.get("proof_ready_candidates"),
        ),
        "can_create_live_permission_count": _first_present(
            proof_summary.get("can_create_live_permission_count"),
            board_firepower.get("can_create_live_permission_rows"),
        ),
        "rejected_proof_candidate_count": _first_present(
            proof_summary.get("rejected_candidate_count"),
            len(proof_queue.get("rejected_candidates"))
            if isinstance(proof_queue.get("rejected_candidates"), list)
            else None,
            planner_summary.get("planner_rejected_candidates"),
        ),
        "rejected_proof_candidate_reasons": _rejected_proof_candidate_reasons(
            proof_queue.get("rejected_candidates")
        ),
        "rejected_proof_candidate_examples": _rejected_proof_candidate_examples(
            proof_queue.get("rejected_candidates")
        ),
        "as_live_ready_path_exists": _first_present(
            proof_summary.get("as_live_ready_path_exists"),
            as_board.get("as_live_ready_path_exists"),
        ),
        "proof_normal_routing_status": _first_present(
            as_board.get("normal_routing_status"),
            portfolio_planner.get("normal_routing_status"),
        ),
        "proof_routing_allowed": as_board.get("routing_allowed"),
        "proof_primary_blocker": board_blocker.get("primary"),
        "proof_global_blockers": _dedupe_strings(board_blocker.get("global_blockers"))[:8],
        "proof_p0_rows": _dedupe_strings(board_blocker.get("p0_rows"))[:8],
        "portfolio_status": portfolio_planner.get("portfolio_status"),
        "portfolio_can_reach_4x_now": portfolio_planner.get("can_reach_4x_now"),
        "portfolio_can_create_live_permission": planner_summary.get("can_create_live_permission"),
        "portfolio_standalone_live_ready_candidates": planner_summary.get(
            "standalone_live_ready_candidates"
        ),
        "portfolio_standalone_math_candidates_meeting_required_return": planner_summary.get(
            "standalone_math_candidates_meeting_required_return"
        ),
        "live_order_request_generated_at_utc": live_order_request.get("generated_at_utc"),
        "gateway_status": live_order_request.get("status"),
        "gateway_issue_codes": _gateway_issue_codes(live_order_request),
    }
    return {key: value for key, value in state.items() if value not in (None, [], {})}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _broker_snapshot_timestamp(payload: dict[str, Any]) -> Any:
    account = payload.get("account") if isinstance(payload.get("account"), dict) else {}
    return _first_present(
        payload.get("generated_at_utc"),
        payload.get("fetched_at_utc"),
        account.get("generated_at_utc"),
        account.get("fetched_at_utc"),
    )


def _gateway_issue_codes(payload: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for key in ("risk_issues", "strategy_issues", "issues"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                code = item.get("code") or item.get("issue_code")
            else:
                code = item
            if code:
                codes.append(str(code))
    return _dedupe_strings(codes)


def _proof_queue_is_empty(current_state: dict[str, Any]) -> bool:
    proof_queue_count = current_state.get("proof_queue_count")
    live_permission_count = current_state.get("can_create_live_permission_count")
    return proof_queue_count == 0 or live_permission_count == 0


def _proof_queue_empty_reason(current_state: dict[str, Any]) -> dict[str, Any]:
    if not _proof_queue_is_empty(current_state):
        return {}

    proof_queue_count = _optional_positive_int(current_state.get("proof_queue_count"))
    live_permission_count = _optional_positive_int(
        current_state.get("can_create_live_permission_count")
    )
    categories: list[dict[str, Any]] = []

    rejected_count = _optional_positive_int(current_state.get("rejected_proof_candidate_count"))
    if rejected_count:
        categories.append(
            {
                "category": "rejected_proof_candidates",
                "status": "BLOCKING",
                "reason_code": "ALL_CURRENT_PROOF_CANDIDATES_REJECTED_BEFORE_QUEUE",
                "candidate_count": rejected_count,
                "evidence_ref": "data/as_proof_pack_queue.json:rejected_candidates",
                "rejection_reasons": _dedupe_strings(
                    current_state.get("rejected_proof_candidate_reasons")
                )[:8],
                "examples": current_state.get("rejected_proof_candidate_examples") or [],
            }
        )

    routing_blocked = (
        current_state.get("proof_normal_routing_status") == "BLOCKED"
        or current_state.get("proof_routing_allowed") is False
        or current_state.get("as_live_ready_path_exists") is False
    )
    if routing_blocked:
        categories.append(
            {
                "category": "lane_board",
                "status": "BLOCKING",
                "reason_code": "LANE_BOARD_NORMAL_ROUTING_BLOCKED",
                "evidence_ref": "data/as_lane_candidate_board.json",
                "normal_routing_status": current_state.get("proof_normal_routing_status"),
                "routing_allowed": current_state.get("proof_routing_allowed"),
                "primary_blocker": current_state.get("proof_primary_blocker"),
                "global_blockers": _dedupe_strings(current_state.get("proof_global_blockers"))[:8],
                "p0_rows": _dedupe_strings(current_state.get("proof_p0_rows"))[:8],
            }
        )

    portfolio_blocked = (
        current_state.get("portfolio_status") is not None
        and (
            current_state.get("portfolio_status") != "LIVE_READY_PORTFOLIO"
            or current_state.get("portfolio_can_reach_4x_now") is False
            or current_state.get("portfolio_can_create_live_permission") is False
        )
    )
    if portfolio_blocked:
        categories.append(
            {
                "category": "portfolio_planner",
                "status": "BLOCKING",
                "reason_code": "PORTFOLIO_PLANNER_HAS_NO_LIVE_PERMISSION_PATH",
                "evidence_ref": "data/portfolio_4x_path_planner.json",
                "portfolio_status": current_state.get("portfolio_status"),
                "can_reach_4x_now": current_state.get("portfolio_can_reach_4x_now"),
                "can_create_live_permission": current_state.get(
                    "portfolio_can_create_live_permission"
                ),
                "standalone_live_ready_candidates": current_state.get(
                    "portfolio_standalone_live_ready_candidates"
                ),
                "standalone_math_candidates_meeting_required_return": current_state.get(
                    "portfolio_standalone_math_candidates_meeting_required_return"
                ),
            }
        )

    gateway_status = current_state.get("gateway_status")
    gateway_blocked = gateway_status not in (None, "ACCEPTED", "STAGED", "SENT", "LIVE_READY")
    if gateway_blocked or not gateway_status:
        categories.append(
            {
                "category": "gateway_issue",
                "status": "BLOCKING" if gateway_blocked else "UNKNOWN",
                "reason_code": (
                    "GATEWAY_HAS_NO_LIVE_READY_INTENT"
                    if gateway_status
                    else "GATEWAY_STATUS_MISSING"
                ),
                "evidence_ref": "data/live_order_request.json",
                "gateway_status": gateway_status,
                "gateway_issue_codes": _dedupe_strings(current_state.get("gateway_issue_codes"))[:8],
            }
        )

    categories = [
        _annotate_empty_reason_category(item, current_state)
        for item in categories
    ]
    primary = _primary_empty_reason_category(categories)
    primary_category = str(primary.get("category")) if primary else "unknown"
    reason = {
        "status": "EMPTY",
        "primary_category": primary_category,
        "primary_reason_code": primary.get("reason_code") if primary else None,
        "primary_causal_rank": primary.get("causal_rank") if primary else None,
        "primary_blocking_depth": primary.get("blocking_depth") if primary else None,
        "proof_queue_count": proof_queue_count,
        "can_create_live_permission_count": live_permission_count,
        "categories": categories,
        "summary": (
            "A/S proof queue is empty because no current candidate both entered the proof "
            "queue and could create live permission; each listed category is a blocking "
            "or missing evidence surface, not live permission."
        ),
    }
    return {key: value for key, value in reason.items() if value not in (None, [], {})}


def _annotate_empty_reason_category(
    category: dict[str, Any],
    current_state: dict[str, Any],
) -> dict[str, Any]:
    annotated = dict(category)
    metrics = _empty_reason_category_metrics(annotated, current_state)
    annotated["freshness"] = _category_freshness(annotated, current_state)
    annotated["causal_rank"] = metrics["causal_rank"]
    annotated["blocking_depth"] = metrics["blocking_depth"]
    annotated["causal_blocker_codes"] = metrics["causal_blocker_codes"]
    annotated["causal_basis"] = metrics["causal_basis"]
    return {key: value for key, value in annotated.items() if value not in (None, [], {})}


def _empty_reason_category_metrics(
    category: dict[str, Any],
    current_state: dict[str, Any],
) -> dict[str, Any]:
    name = str(category.get("category") or "")
    blocker_codes = _category_blocker_codes(category)
    direct_live_blockers = [
        code for code in blocker_codes if code in DIRECT_LIVE_PERMISSION_BLOCKERS
    ]
    proof_blockers = [code for code in blocker_codes if code in PROOF_CANDIDATE_BLOCKERS]

    if name == "lane_board":
        if direct_live_blockers or category.get("primary_blocker"):
            causal_rank = 0
            basis = "normal_routing_or_primary_live_permission_blocker"
        else:
            causal_rank = 2
            basis = "lane_board_routing_surface"
        blocking_depth = _count_true(
            category.get("normal_routing_status") == "BLOCKED",
            category.get("routing_allowed") is False,
            current_state.get("as_live_ready_path_exists") is False,
            current_state.get("can_create_live_permission_count") == 0,
            bool(direct_live_blockers),
        )
    elif name == "rejected_proof_candidates":
        causal_rank = 1 if proof_blockers or category.get("candidate_count") else 3
        basis = "candidate_rejected_before_proof_queue"
        blocking_depth = _count_true(
            bool(category.get("candidate_count")),
            current_state.get("proof_queue_count") == 0,
            current_state.get("proof_ready_count") == 0,
            current_state.get("can_create_live_permission_count") == 0,
            bool(proof_blockers),
        )
    elif name == "portfolio_planner":
        causal_rank = 3
        basis = "portfolio_path_is_downstream_of_live_ready_proof"
        blocking_depth = _count_true(
            category.get("portfolio_status") != "LIVE_READY_PORTFOLIO",
            category.get("can_reach_4x_now") is False,
            category.get("can_create_live_permission") is False,
            current_state.get("can_create_live_permission_count") == 0,
        )
    elif name == "gateway_issue":
        gateway_direct = [
            code for code in _dedupe_strings(category.get("gateway_issue_codes"))
            if code in DIRECT_LIVE_PERMISSION_BLOCKERS
        ]
        causal_rank = 0 if gateway_direct else 4
        basis = (
            "gateway_reports_direct_live_permission_blocker"
            if gateway_direct
            else "gateway_status_is_downstream_of_proof_and_receipt"
        )
        blocking_depth = _count_true(
            category.get("gateway_status") not in GATEWAY_LIVE_READY_STATUSES,
            current_state.get("can_create_live_permission_count") == 0,
            bool(gateway_direct),
        )
    else:
        causal_rank = 9
        basis = "unknown_empty_proof_queue_surface"
        blocking_depth = 1 if category.get("status") == "BLOCKING" else 0

    return {
        "causal_rank": causal_rank,
        "blocking_depth": blocking_depth,
        "causal_blocker_codes": direct_live_blockers or proof_blockers or blocker_codes[:4],
        "causal_basis": basis,
    }


def _category_blocker_codes(category: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for key in (
        "primary_blocker",
        "reason_code",
        "rejection_reasons",
        "global_blockers",
        "p0_rows",
        "gateway_issue_codes",
    ):
        value = category.get(key)
        if isinstance(value, list):
            codes.extend(str(item) for item in value if item)
        elif value:
            codes.append(str(value))
    return _dedupe_strings(codes)


def _category_freshness(category: dict[str, Any], current_state: dict[str, Any]) -> dict[str, Any]:
    name = str(category.get("category") or "")
    if name == "rejected_proof_candidates":
        generated_at = current_state.get("proof_queue_generated_at_utc")
        source = "data/as_proof_pack_queue.json"
        required_reference_keys = ("broker_snapshot", "trader_support_bot")
    elif name == "lane_board":
        generated_at = current_state.get("as_board_generated_at_utc")
        source = "data/as_lane_candidate_board.json"
        required_reference_keys = ("broker_snapshot", "trader_support_bot")
    elif name == "portfolio_planner":
        generated_at = current_state.get("portfolio_planner_generated_at_utc")
        source = "data/portfolio_4x_path_planner.json"
        required_reference_keys = (
            "broker_snapshot",
            "trader_support_bot",
            "as_proof_pack_queue",
            "as_lane_candidate_board",
        )
    elif name == "gateway_issue":
        generated_at = current_state.get("live_order_request_generated_at_utc")
        source = "data/live_order_request.json"
        required_reference_keys = (
            "broker_snapshot",
            "trader_support_bot",
            "as_proof_pack_queue",
            "as_lane_candidate_board",
            "portfolio_4x_path_planner",
        )
    else:
        generated_at = None
        source = category.get("evidence_ref")
        required_reference_keys = ("broker_snapshot", "trader_support_bot")
    freshness = _freshness_for_generated_at(
        generated_at,
        current_state=current_state,
        source=str(source or ""),
        required_reference_keys=required_reference_keys,
    )
    if (
        name == "gateway_issue"
        and freshness.get("status") == "CONTRADICTED"
        and _no_send_gateway_is_aligned_with_current_upstream_blocks(current_state)
    ):
        freshness = {
            **freshness,
            "status": "FRESH",
            "freshness_reason": (
                "no-send gateway status is aligned with current proof/portfolio blockers; "
                "it need not postdate downstream read-only evidence while no proof-ready "
                "or live-permission-capable lane exists"
            ),
            "dependency_lag_exempted": True,
        }
    return freshness


def _no_send_gateway_is_aligned_with_current_upstream_blocks(current_state: dict[str, Any]) -> bool:
    gateway_status = str(current_state.get("gateway_status") or "").upper()
    if gateway_status not in {
        "NO_ACTION",
        "NO_LIVE_READY_INTENT",
        "GPT_WAIT",
        "GPT_REQUEST_EVIDENCE",
        "ACCEPTED_REQUEST_EVIDENCE_BLOCKS_CAMPAIGN_RECOVERY",
        "STALE_ACCEPTED_REQUEST_EVIDENCE_BLOCKS_CAMPAIGN_RECOVERY",
    }:
        return False
    if _optional_positive(current_state.get("proof_ready_count")):
        return False
    if _optional_positive(current_state.get("can_create_live_permission_count")):
        return False
    if _optional_positive(current_state.get("portfolio_standalone_live_ready_candidates")):
        return False
    if _optional_positive(current_state.get("portfolio_standalone_math_candidates_meeting_required_return")):
        return False
    if _truthy(current_state.get("portfolio_can_create_live_permission")):
        return False
    if _truthy(current_state.get("portfolio_can_reach_4x_now")):
        return False
    if _truthy(current_state.get("as_live_ready_path_exists")):
        return False
    return True


def _primary_empty_reason_category(categories: list[dict[str, Any]]) -> dict[str, Any]:
    blocking = [item for item in categories if item.get("status") == "BLOCKING"] or categories
    if not blocking:
        return {}
    eligible = [
        item
        for item in blocking
        if not (
            isinstance(item.get("freshness"), dict)
            and item["freshness"].get("status") == "CONTRADICTED"
        )
    ]
    blocking = eligible or [
        item
        for item in categories
        if not (
            isinstance(item.get("freshness"), dict)
            and item["freshness"].get("status") == "CONTRADICTED"
        )
    ]
    if not blocking:
        return {}
    return sorted(
        blocking,
        key=lambda item: (
            _freshness_primary_priority(item.get("freshness")),
            _safe_int(item.get("causal_rank"), default=99),
            -_safe_int(item.get("blocking_depth"), default=0),
            str(item.get("category") or ""),
        ),
    )[0]


def _freshness_for_generated_at(
    generated_at: Any,
    *,
    current_state: dict[str, Any],
    source: str,
    required_reference_keys: tuple[str, ...],
) -> dict[str, Any]:
    generated_dt = _parse_utc_timestamp(generated_at)
    references = _artifact_timestamp_references(current_state)
    available_refs = [item for item in references.values() if item[1] is not None]
    reference_label, reference_dt = _latest_timestamp_reference(available_refs)
    required_refs = [
        references[key]
        for key in required_reference_keys
        if key in references and references[key][1] is not None
    ]
    newest_required_label, newest_required_dt = _latest_timestamp_reference(required_refs)
    max_age = PROOF_EMPTY_REASON_FRESHNESS_MAX_AGE_SECONDS

    if generated_dt is None:
        status = "MISSING"
        age_seconds = None
        reason = "artifact timestamp is missing or unreadable"
    elif reference_dt is None:
        status = "MISSING"
        age_seconds = None
        reason = "freshness reference timestamp is missing"
    else:
        age_seconds = (reference_dt - generated_dt).total_seconds()
        future_skew_seconds = (generated_dt - datetime.now(timezone.utc)).total_seconds()
        dependency_lag_seconds = (
            (newest_required_dt - generated_dt).total_seconds()
            if newest_required_dt is not None
            else None
        )
        if future_skew_seconds > PROOF_EMPTY_REASON_FUTURE_TOLERANCE_SECONDS:
            status = "CONTRADICTED"
            reason = "artifact timestamp is in the future relative to orchestrator runtime"
        elif (
            dependency_lag_seconds is not None
            and dependency_lag_seconds > PROOF_EMPTY_REASON_FUTURE_TOLERANCE_SECONDS
        ):
            status = "CONTRADICTED"
            reason = f"artifact predates required upstream evidence: {newest_required_label}"
        elif age_seconds > max_age:
            status = "STALE"
            reason = "artifact timestamp lags the newest compared evidence packet"
        else:
            status = "FRESH"
            reason = "artifact timestamp is aligned with the compared evidence packet"

    result = {
        "status": status,
        "generated_at_utc": generated_at,
        "source": source,
        "freshness_age_seconds": _round_seconds(age_seconds),
        "freshness_reference_timestamp": reference_dt.isoformat() if reference_dt else None,
        "freshness_reference_source": reference_label,
        "freshness_max_age_seconds": max_age,
        "freshness_reason": reason,
    }
    return result


def _artifact_timestamp_references(current_state: dict[str, Any]) -> dict[str, tuple[str, datetime | None]]:
    return {
        "broker_snapshot": (
            "data/broker_snapshot.json",
            _parse_utc_timestamp(current_state.get("broker_snapshot_generated_at_utc")),
        ),
        "trader_support_bot": (
            "data/trader_support_bot.json",
            _parse_utc_timestamp(current_state.get("support_generated_at_utc")),
        ),
        "as_proof_pack_queue": (
            "data/as_proof_pack_queue.json",
            _parse_utc_timestamp(current_state.get("proof_queue_generated_at_utc")),
        ),
        "as_lane_candidate_board": (
            "data/as_lane_candidate_board.json",
            _parse_utc_timestamp(current_state.get("as_board_generated_at_utc")),
        ),
        "portfolio_4x_path_planner": (
            "data/portfolio_4x_path_planner.json",
            _parse_utc_timestamp(current_state.get("portfolio_planner_generated_at_utc")),
        ),
        "live_order_request": (
            "data/live_order_request.json",
            _parse_utc_timestamp(current_state.get("live_order_request_generated_at_utc")),
        ),
    }


def _latest_timestamp_reference(
    references: list[tuple[str, datetime | None]]
) -> tuple[str | None, datetime | None]:
    available = [(label, value) for label, value in references if value is not None]
    if not available:
        return None, None
    return max(available, key=lambda item: item[1])


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _round_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def _freshness_primary_priority(value: Any) -> int:
    if not isinstance(value, dict):
        return FRESHNESS_PRIMARY_PRIORITY["MISSING"]
    return FRESHNESS_PRIMARY_PRIORITY.get(
        str(value.get("status") or "MISSING"),
        FRESHNESS_PRIMARY_PRIORITY["MISSING"],
    )


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _count_true(*values: bool) -> int:
    return sum(1 for value in values if value)


def _next_evidence_actions_for_empty_proof_queue(
    reason: dict[str, Any],
    *,
    current_state: dict[str, Any],
) -> list[dict[str, Any]]:
    if not reason:
        return []
    categories = {
        str(item.get("category"))
        for item in reason.get("categories", [])
        if isinstance(item, dict) and item.get("category")
    }
    actions: list[dict[str, Any]] = []

    if "rejected_proof_candidates" in categories:
        actions.append(
            _evidence_action(
                current_state,
                {
                    "action_id": "collect_exact_tp_or_live_grade_harvest_evidence",
                    "category": "rejected_proof_candidates",
                    "commands": [],
                    "success_condition_text": (
                        "New local TAKE_PROFIT_ORDER receipts or exact HARVEST live-grade "
                        "promotion moves a candidate from rejected_candidates into queue/proof_ready."
                    ),
                    "success_condition": _success_condition(
                        "any",
                        [
                            _condition_check(
                                "proof_queue_count",
                                "gt",
                                _count_watermark(current_state.get("proof_queue_count")),
                            ),
                            _condition_check(
                                "proof_ready_count",
                                "gt",
                                _count_watermark(current_state.get("proof_ready_count")),
                            ),
                            _condition_check(
                                "can_create_live_permission_count",
                                "gt",
                                _count_watermark(
                                    current_state.get("can_create_live_permission_count")
                                ),
                            ),
                            _condition_check(
                                "rejected_proof_candidate_count",
                                "lt",
                                _count_watermark(
                                    current_state.get("rejected_proof_candidate_count")
                                ),
                            ),
                        ],
                        description=(
                            "A candidate entered the proof queue, became proof-ready/live-permission "
                            "capable, or the rejected candidate count decreased after evidence refresh."
                        ),
                    ),
                    "read_only": True,
                    "live_side_effects": [],
                }
            )
        )
    if "lane_board" in categories:
        actions.append(
            _evidence_action(
                current_state,
                {
                    "action_id": "refresh_lane_board_after_input_evidence_changes",
                    "category": "lane_board",
                    "commands": [
                        "PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance",
                        "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts",
                        "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                        "PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop",
                    ],
                    "success_condition_text": (
                        "normal_routing_status is no longer BLOCKED and "
                        "can_create_live_permission_count becomes positive."
                    ),
                    "success_condition": _success_condition(
                        "all",
                        [
                            _condition_check("proof_normal_routing_status", "neq", "BLOCKED"),
                            _condition_check("proof_routing_allowed", "is_true"),
                            _condition_check(
                                "can_create_live_permission_count",
                                "gt",
                                _count_watermark(
                                    current_state.get("can_create_live_permission_count")
                                ),
                            ),
                        ],
                        description=(
                            "Lane-board normal routing reopened and at least one current row can "
                            "create live permission after the full evidence refresh."
                        ),
                    ),
                    "read_only": True,
                    "live_side_effects": [],
                }
            )
        )
    if "portfolio_planner" in categories:
        actions.append(
            _evidence_action(
                current_state,
                {
                    "action_id": "refresh_portfolio_4x_path_planner",
                    "category": "portfolio_planner",
                    "commands": [
                        "PYTHONPATH=src python3 -m quant_rabbit.cli as-4x-proof-path",
                    ],
                    "success_condition_text": (
                        "portfolio_status becomes LIVE_READY_PORTFOLIO, an exact boolean live "
                        "capability flag becomes true, or proof-ready candidates advance."
                    ),
                    "success_condition": _success_condition(
                        "any",
                        [
                            _condition_check(
                                "portfolio_status",
                                "eq",
                                "LIVE_READY_PORTFOLIO",
                            ),
                            _condition_check(
                                "portfolio_can_create_live_permission",
                                "is_true",
                            ),
                            _condition_check("portfolio_can_reach_4x_now", "is_true"),
                            _condition_check(
                                "proof_ready_count",
                                "gt",
                                _count_watermark(current_state.get("proof_ready_count")),
                            ),
                        ],
                        description=(
                            "The planner finds a live-permission-capable path, reaches the 4x path "
                            "gate, or sees proof-ready candidates after refresh."
                        ),
                    ),
                    "read_only": True,
                    "live_side_effects": [],
                }
            )
        )
    if "gateway_issue" in categories:
        actions.append(
            _evidence_action(
                current_state,
                {
                    "action_id": "refresh_gateway_evidence_without_sending",
                    "category": "gateway_issue",
                    "commands": [
                        "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator",
                    ],
                    "success_condition_text": (
                        "data/live_order_request.json carries a current accepted/staged/sent "
                        "gateway status only after the normal verifier/gateway path authorizes it."
                    ),
                    "success_condition": _success_condition(
                        "any",
                        [
                            _condition_check(
                                "gateway_status",
                                "in",
                                sorted(GATEWAY_LIVE_READY_STATUSES),
                            ),
                            _condition_check(
                                "can_create_live_permission_count",
                                "gt",
                                _count_watermark(
                                    current_state.get("can_create_live_permission_count")
                                ),
                            ),
                        ],
                        description=(
                            "Gateway evidence now reports an authorized live-ready/staged/sent "
                            "state, or upstream proof can create live permission."
                        ),
                    ),
                    "read_only": True,
                    "live_side_effects": [],
                }
            )
        )
    return actions


def _evidence_action(current_state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    action["progress_watermark_contract"] = (
        EVIDENCE_ACTION_PROGRESS_WATERMARK_CONTRACT
    )
    condition = action.get("success_condition")
    if isinstance(condition, dict):
        action["success_condition_evaluation"] = _evaluate_success_condition(
            condition,
            current_state,
        )
        action["progress_watermark_origin"] = _evidence_action_watermark_origin(
            action,
            current_state=current_state,
        )
    return action


def _evidence_action_watermark_origin(
    action: dict[str, Any],
    *,
    current_state: dict[str, Any],
) -> dict[str, Any]:
    condition = action.get("success_condition")
    checks = condition.get("checks") if isinstance(condition, dict) else []
    fields = [
        str(check.get("field") or "")
        for check in checks
        if isinstance(check, dict) and str(check.get("field") or "")
    ]
    condition_state = {
        field: current_state.get(field)
        for field in dict.fromkeys(fields)
    }
    return {
        "contract": EVIDENCE_ACTION_PROGRESS_WATERMARK_CONTRACT,
        "action_id": str(action.get("action_id") or ""),
        "category": str(action.get("category") or ""),
        "condition_sha256": _canonical_sha256(condition),
        "condition_state": condition_state,
        "condition_state_sha256": _canonical_sha256(condition_state),
    }


def _reconcile_evidence_action_progress(
    fresh_actions: list[dict[str, Any]],
    *,
    previous_evidence_actions: list[dict[str, Any]],
    previous_current_state: dict[str, Any],
    current_state: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Carry an unmet action watermark across one or more verifier refreshes."""

    previous_by_id = _unique_evidence_actions_by_id(previous_evidence_actions)

    progress: list[dict[str, Any]] = []
    pending_previous: dict[str, dict[str, Any]] = {}
    for action_id, action in previous_by_id.items():
        if not _evidence_action_has_pending_valid_watermark(
            action,
            previous_current_state=previous_current_state,
        ):
            continue
        condition = action["success_condition"]
        evaluation = _evaluate_success_condition(condition, current_state)
        pending_previous[action_id] = action
        if evaluation.get("status") == "MET" and evaluation.get("passed") is True:
            progress.append(
                {
                    "action_id": action_id,
                    "category": str(action.get("category") or ""),
                    "progress_watermark_contract": (
                        EVIDENCE_ACTION_PROGRESS_WATERMARK_CONTRACT
                    ),
                    "progress_watermark_origin": action.get(
                        "progress_watermark_origin"
                    ),
                    "success_condition": condition,
                    "success_condition_evaluation": evaluation,
                }
            )

    reconciled: list[dict[str, Any]] = []
    for fresh in fresh_actions:
        action = dict(fresh)
        action["progress_watermark_source"] = "CURRENT_REFRESH"
        action_id = str(action.get("action_id") or "").strip()
        previous = pending_previous.get(action_id)
        if previous is not None and _evidence_action_watermark_shapes_match(
            previous,
            action,
        ):
            condition = previous["success_condition"]
            action["success_condition"] = condition
            action["progress_watermark_origin"] = previous[
                "progress_watermark_origin"
            ]
            action["success_condition_evaluation"] = _evaluate_success_condition(
                condition,
                current_state,
            )
            action["progress_watermark_source"] = "PREVIOUS_REFRESH"
        reconciled.append(action)
    return reconciled, progress


def _unique_evidence_actions_by_id(
    actions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for action in actions:
        action_id = str(action.get("action_id") or "").strip()
        if not action_id:
            continue
        if action_id in by_id:
            duplicate_ids.add(action_id)
            continue
        by_id[action_id] = action
    for action_id in duplicate_ids:
        by_id.pop(action_id, None)
    return by_id


def validated_evidence_action_material(
    action: Any,
    *,
    current_state: Any,
    source: str,
) -> dict[str, Any] | None:
    """Return trusted repeat material for a canonical evidence action or receipt.

    Serialized hashes and evaluations are treated only as assertions. The
    condition, frozen origin state, and current evaluation must all reproduce
    the canonical repair-orchestrator action before any of them can influence a
    downstream repeat guard.
    """

    if not isinstance(action, dict) or not isinstance(current_state, dict):
        return None
    if source not in {
        EVIDENCE_ACTION_MATERIAL_SOURCE_NEXT_ACTION,
        EVIDENCE_ACTION_MATERIAL_SOURCE_PROGRESS_RECEIPT,
    }:
        return None
    if (
        action.get("progress_watermark_contract")
        != EVIDENCE_ACTION_PROGRESS_WATERMARK_CONTRACT
    ):
        return None

    action_id = str(action.get("action_id") or "").strip()
    category = str(action.get("category") or "").strip()
    if not action_id or not category:
        return None
    condition = action.get("success_condition")
    if not _success_condition_is_carryable(condition):
        return None
    evaluation = action.get("success_condition_evaluation")
    if not isinstance(evaluation, dict):
        return None

    origin_state = _validated_evidence_action_watermark_origin(action)
    if origin_state is None:
        return None
    canonical_action = _canonical_evidence_action_for_origin(
        action_id=action_id,
        origin_state=origin_state,
    )
    if canonical_action is None:
        return None

    if source == EVIDENCE_ACTION_MATERIAL_SOURCE_NEXT_ACTION:
        if _evidence_action_semantics(action) != _evidence_action_semantics(
            canonical_action
        ):
            return None
    else:
        if set(action) != _EVIDENCE_ACTION_PROGRESS_RECEIPT_FIELDS:
            return None
        if (
            str(action.get("action_id") or "")
            != str(canonical_action.get("action_id") or "")
            or str(action.get("category") or "")
            != str(canonical_action.get("category") or "")
            or condition != canonical_action.get("success_condition")
        ):
            return None

    recomputed_evaluation = _evaluate_success_condition(condition, current_state)
    if evaluation != recomputed_evaluation:
        return None
    evaluation_status = recomputed_evaluation.get("status")
    evaluation_passed = recomputed_evaluation.get("passed")
    if (evaluation_status, evaluation_passed) not in {
        ("NOT_MET", False),
        ("MET", True),
    }:
        return None
    if source == EVIDENCE_ACTION_MATERIAL_SOURCE_PROGRESS_RECEIPT and (
        evaluation_status,
        evaluation_passed,
    ) != ("MET", True):
        return None

    canonical_condition = canonical_action["success_condition"]
    return {
        "source": source,
        "action_id": action_id,
        "category": category,
        "condition_sha256": _canonical_sha256(canonical_condition),
        "origin_condition_state_sha256": _canonical_sha256(origin_state),
        "evaluation_sha256": _canonical_sha256(recomputed_evaluation),
        "evaluation_status": evaluation_status,
        "evaluation_passed": evaluation_passed,
    }


def _evidence_action_has_pending_valid_watermark(
    action: dict[str, Any],
    *,
    previous_current_state: dict[str, Any],
) -> bool:
    material = validated_evidence_action_material(
        action,
        current_state=previous_current_state,
        source=EVIDENCE_ACTION_MATERIAL_SOURCE_NEXT_ACTION,
    )
    return bool(
        material is not None
        and material.get("evaluation_status") == "NOT_MET"
        and material.get("evaluation_passed") is False
    )


def _validated_evidence_action_watermark_origin(
    action: dict[str, Any],
) -> dict[str, Any] | None:
    origin = action.get("progress_watermark_origin")
    if not isinstance(origin, dict) or set(origin) != {
        "contract",
        "action_id",
        "category",
        "condition_sha256",
        "condition_state",
        "condition_state_sha256",
    }:
        return None
    if origin.get("contract") != EVIDENCE_ACTION_PROGRESS_WATERMARK_CONTRACT:
        return None
    if str(origin.get("action_id") or "") != str(action.get("action_id") or ""):
        return None
    if str(origin.get("category") or "") != str(action.get("category") or ""):
        return None
    condition = action.get("success_condition")
    if origin.get("condition_sha256") != _canonical_sha256(condition):
        return None
    condition_state = origin.get("condition_state")
    if not isinstance(condition_state, dict):
        return None
    if origin.get("condition_state_sha256") != _canonical_sha256(condition_state):
        return None
    checks = condition.get("checks") if isinstance(condition, dict) else []
    expected_fields = {
        str(check.get("field") or "")
        for check in checks
        if isinstance(check, dict) and str(check.get("field") or "")
    }
    if set(condition_state) != expected_fields:
        return None
    for check in checks:
        if not isinstance(check, dict):
            return None
        field = str(check.get("field") or "")
        if field not in _STRICT_COUNT_FIELDS:
            continue
        watermark = _count_watermark(check.get("value"))
        if watermark is None or condition_state.get(field) != watermark:
            return None
    origin_evaluation = _evaluate_success_condition(condition, condition_state)
    if origin_evaluation.get("status") != "NOT_MET" or origin_evaluation.get(
        "passed"
    ) is not False:
        return None
    return condition_state


def _canonical_evidence_action_for_origin(
    *,
    action_id: str,
    origin_state: dict[str, Any],
) -> dict[str, Any] | None:
    reason = {
        "categories": [
            {"category": category}
            for category in (
                "rejected_proof_candidates",
                "lane_board",
                "portfolio_planner",
                "gateway_issue",
            )
        ]
    }
    canonical = _next_evidence_actions_for_empty_proof_queue(
        reason,
        current_state=origin_state,
    )
    return _unique_evidence_actions_by_id(canonical).get(action_id)


def _success_condition_is_carryable(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("schema_version") != "success_condition_v1":
        return False
    if str(value.get("mode") or "").lower() not in {"all", "any"}:
        return False
    checks = value.get("checks")
    if not isinstance(checks, list) or not checks:
        return False
    for check in checks:
        if not isinstance(check, dict):
            return False
        field = str(check.get("field") or "")
        operator = str(check.get("operator") or "")
        if not field or not operator:
            return False
        if field in _STRICT_COUNT_FIELDS and _count_watermark(check.get("value")) is None:
            return False
    return True


def _evidence_action_watermark_shapes_match(
    previous: dict[str, Any],
    fresh: dict[str, Any],
) -> bool:
    return _evidence_action_semantics(
        previous,
        mask_count_watermarks=True,
    ) == _evidence_action_semantics(
        fresh,
        mask_count_watermarks=True,
    )


def _evidence_action_semantics(
    action: dict[str, Any],
    *,
    mask_count_watermarks: bool = False,
) -> dict[str, Any]:
    semantics = {
        key: value
        for key, value in action.items()
        if key
        not in {
            "progress_watermark_origin",
            "progress_watermark_source",
            "success_condition_evaluation",
        }
    }
    condition = semantics.get("success_condition")
    if mask_count_watermarks and isinstance(condition, dict):
        semantics["success_condition"] = _masked_count_watermark_condition(condition)
    return semantics


def _masked_count_watermark_condition(condition: dict[str, Any]) -> dict[str, Any]:
    masked = dict(condition)
    checks = condition.get("checks")
    if not isinstance(checks, list):
        return masked
    masked_checks: list[Any] = []
    for raw in checks:
        if not isinstance(raw, dict):
            masked_checks.append(raw)
            continue
        check = dict(raw)
        if str(check.get("field") or "") in _STRICT_COUNT_FIELDS:
            check["value"] = "__FROZEN_NON_NEGATIVE_INTEGER_WATERMARK__"
        masked_checks.append(check)
    masked["checks"] = masked_checks
    return masked


def _success_condition(
    mode: str,
    checks: list[dict[str, Any]],
    *,
    description: str,
) -> dict[str, Any]:
    return {
        "description": description,
        "schema_version": "success_condition_v1",
        "verification_scope": (
            "Evaluate these checks against loop_engineering_prompt.current_state from "
            "the next verifier/orchestrator refresh."
        ),
        "mode": mode,
        "checks": checks,
    }


def _condition_check(field: str, operator: str, value: Any = None) -> dict[str, Any]:
    check = {
        "field": field,
        "operator": operator,
    }
    if value is not None:
        check["value"] = value
    return check


def _count_watermark(value: Any) -> int | None:
    """Return an exact non-negative count, or no progress baseline for malformed input."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


_STRICT_COUNT_FIELDS = frozenset(
    {
        "proof_queue_count",
        "proof_ready_count",
        "can_create_live_permission_count",
        "rejected_proof_candidate_count",
    }
)


def _evaluate_success_condition(
    condition: dict[str, Any],
    current_state: dict[str, Any],
) -> dict[str, Any]:
    checks = condition.get("checks")
    if not isinstance(checks, list) or not checks:
        return {
            "status": "UNKNOWN",
            "passed": False,
            "reason": "success_condition has no machine-readable checks",
            "checks": [],
        }
    evaluated = [
        _evaluate_condition_check(check, current_state)
        for check in checks
        if isinstance(check, dict)
    ]
    if not evaluated:
        return {
            "status": "UNKNOWN",
            "passed": False,
            "reason": "success_condition checks were not JSON objects",
            "checks": [],
        }
    mode = str(condition.get("mode") or "all").lower()
    if mode == "any":
        passed = any(item.get("passed") for item in evaluated)
    else:
        mode = "all"
        passed = all(item.get("passed") for item in evaluated)
    return {
        "status": "MET" if passed else "NOT_MET",
        "passed": passed,
        "mode": mode,
        "checks": evaluated,
    }


def _evaluate_condition_check(
    check: dict[str, Any],
    current_state: dict[str, Any],
) -> dict[str, Any]:
    field = str(check.get("field") or "")
    operator = str(check.get("operator") or "")
    expected = check.get("value")
    actual = current_state.get(field)
    if field in _STRICT_COUNT_FIELDS:
        actual_count = _count_watermark(actual)
        expected_count = _count_watermark(expected)
        passed = (
            actual_count is not None
            and expected_count is not None
            and _condition_passes(actual_count, operator, expected_count)
        )
    else:
        passed = _condition_passes(actual, operator, expected)
    return {
        "field": field,
        "operator": operator,
        "expected": expected,
        "actual": actual,
        "passed": passed,
    }


def _condition_passes(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "eq":
        return actual == expected
    if operator == "neq":
        return actual is not None and actual != expected
    actual_number = _as_float(actual)
    expected_number = _as_float(expected)
    if operator == "gt":
        return (
            actual_number is not None
            and expected_number is not None
            and actual_number > expected_number
        )
    if operator == "gte":
        return (
            actual_number is not None
            and expected_number is not None
            and actual_number >= expected_number
        )
    if operator == "lt":
        return (
            actual_number is not None
            and expected_number is not None
            and actual_number < expected_number
        )
    if operator == "lte":
        return (
            actual_number is not None
            and expected_number is not None
            and actual_number <= expected_number
        )
    if operator == "in":
        return isinstance(expected, list) and actual in expected
    if operator == "not_in":
        return isinstance(expected, list) and actual not in expected
    if operator == "is_true":
        return actual is True
    if operator == "is_false":
        return actual is False
    return False


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rejected_proof_candidate_reasons(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    reasons: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        reasons.extend(_dedupe_strings(item.get("rejection_reasons")))
        reasons.extend(_dedupe_strings(item.get("current_blockers")))
    return _dedupe_strings(reasons)


def _rejected_proof_candidate_examples(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    examples: list[dict[str, Any]] = []
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        example = {
            "lane_id": item.get("lane_id"),
            "pair": item.get("pair"),
            "side": item.get("side"),
            "method": item.get("method"),
            "order_type": item.get("order_type"),
            "rejection_reasons": _dedupe_strings(item.get("rejection_reasons"))[:4],
        }
        examples.append({key: val for key, val in example.items() if val not in (None, [], {})})
    return examples


def _render_profitability_rca_for_prompt(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "(none)"
    parts: list[str] = []
    capture_status = value.get("capture_economics_status") or value.get(
        "profitability_acceptance_status"
    )
    if capture_status:
        parts.append(f"capture={capture_status}")
    for key, label in [
        ("overall_expectancy_jpy_per_trade", "overall_exp_jpy"),
        ("overall_net_jpy", "overall_net_jpy"),
        ("take_profit_expectancy_jpy_per_trade", "tp_exp_jpy"),
        ("take_profit_net_jpy", "tp_net_jpy"),
        ("market_close_expectancy_jpy_per_trade", "market_close_exp_jpy"),
        ("market_close_net_jpy", "market_close_net_jpy"),
    ]:
        metric = value.get(key)
        if metric is not None:
            parts.append(f"{label}={metric}")
    leak_segments = value.get("tp_proven_market_close_leak_segments")
    if leak_segments is not None:
        parts.append(f"tp_market_close_leak_segments={leak_segments}")
    acceptance_blockers = _dedupe_strings(value.get("acceptance_blocker_codes"))[:4]
    if acceptance_blockers:
        parts.append("acceptance_blockers=" + ",".join(acceptance_blockers))
    operational_blockers = _dedupe_strings(value.get("operational_blocker_codes"))[:4]
    if operational_blockers:
        parts.append("operational_blockers=" + ",".join(operational_blockers))
    return "; ".join(parts) if parts else "(none)"


def _format_optional_count(label: str, value: Any) -> str:
    if value is None:
        return ""
    return f"{label}={value}"


def _render_approval_details_for_prompt(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "(none)"
    rendered: list[str] = []
    for detail in value[:3]:
        if not isinstance(detail, dict):
            continue
        code = detail.get("code")
        examples = _format_approval_examples(detail.get("examples"))
        clearance = _format_approval_clearance(detail.get("clearance_conditions"))
        parts = [str(code)] if code else []
        parts.extend(part for part in [examples, clearance] if part)
        if parts:
            rendered.append(" ".join(parts))
    return " | ".join(rendered) if rendered else "(none)"


def _codex_work_order(
    selected: dict[str, Any],
    *,
    status: str,
    trader_request: str,
    execution_contract: dict[str, Any],
    approval_boundary: dict[str, Any],
    output_path: Path,
    report_path: Path,
    loop_prompt: dict[str, Any] | None = None,
    previous_work_order: dict[str, Any] | None = None,
    ack_active_lane_dispatch: str = "",
) -> dict[str, Any]:
    proof_state = _work_order_proof_state(loop_prompt)
    dispatched_material_digests = _previous_dispatched_active_lane_material_digests(
        previous_work_order
    )
    pending_material_digest = _previous_pending_active_lane_material_digest(
        previous_work_order
    )
    acknowledged_material_digest = str(ack_active_lane_dispatch or "").strip()
    acknowledgement_replayed = False
    if acknowledged_material_digest:
        if acknowledged_material_digest == pending_material_digest:
            dispatched_material_digests = _bounded_digest_history(
                [*dispatched_material_digests, acknowledged_material_digest]
            )
            pending_material_digest = ""
        elif acknowledged_material_digest in dispatched_material_digests:
            # A report/fsync failure can occur after the authoritative JSON ACK
            # state commits, including a commit that already opened a newer
            # material dispatch. Retrying the completed digest is idempotent and
            # must preserve any newer pending work.
            acknowledgement_replayed = True
        elif pending_material_digest:
            raise ValueError(
                "--ack-active-lane-dispatch does not match the pending material digest"
            )
        else:
            raise ValueError(
                "--ack-active-lane-dispatch requires an existing pending or previously completed exact digest"
            )
    last_dispatched_material_digest = (
        dispatched_material_digests[-1] if dispatched_material_digests else ""
    )
    if not selected:
        if status == STATUS_NO_REQUESTS:
            read_only_work = _read_only_evidence_work_order(
                proof_state=proof_state,
                status=status,
                trader_request=trader_request,
                approval_boundary=approval_boundary,
                loop_prompt=loop_prompt,
                dispatched_material_digests=dispatched_material_digests,
                pending_material_digest=pending_material_digest,
                acknowledged_material_digest=acknowledged_material_digest,
                acknowledgement_replayed=acknowledgement_replayed,
            )
            if read_only_work:
                return read_only_work
        return {
            "status": "NO_ACTIONABLE_CODEX_WORK",
            "orchestrator_status": status,
            "trader_request": trader_request,
            "reason": "No selected READY_FOR_CODEX_IMPLEMENTATION request is available.",
            "proof_state": proof_state,
            "approval_boundary": approval_boundary,
            "last_dispatched_material_digest": last_dispatched_material_digest or None,
            "dispatched_material_digests": dispatched_material_digests,
            "pending_material_digest": pending_material_digest or None,
            "acknowledged_material_digest": acknowledged_material_digest or None,
            "acknowledgement_replayed": acknowledgement_replayed,
        }
    return {
        "status": selected.get("automation_status"),
        "orchestrator_status": status,
        "selected_request_code": selected.get("code"),
        "priority": selected.get("priority"),
        "repair_status": selected.get("repair_status"),
        "dependency_rank": selected.get("dependency_rank"),
        "selection_reason": selected.get("selection_reason"),
        "trader_request": trader_request,
        "objective": selected.get("problem"),
        "why_now": selected.get("why_now"),
        "source_findings": selected.get("source_findings") or [],
        "evidence_summary": selected.get("evidence_summary") or {},
        "clearance_conditions": selected.get("clearance_conditions") or [],
        "suggested_files": selected.get("suggested_files") or [],
        "required_tests": selected.get("required_tests") or [],
        "targeted_test_commands": selected.get("targeted_test_commands") or [],
        "verification_commands": selected.get("verification_commands") or [],
        "final_verification_commands": selected.get("final_verification_commands") or [],
        "implementation_loop": selected.get("implementation_loop") or [],
        "deliverables": [
            "code_patch_or_documented_no_code_change",
            "regression_tests_for_the_named_failure",
            "positive_path_tests_for_the_allowed_shape",
            "updated_runtime_contract_docs_when_behavior_changes",
            "passing_targeted_tests",
            "passing_required_verification_commands",
            "passing_full_unittest_discover",
            "git_commit_with_codex_attribution",
            "verified_live_runtime_sync",
        ],
        "artifact_paths": {
            "orchestrator_json": str(output_path),
            "orchestrator_report": str(report_path),
        },
        "proof_state": proof_state,
        "commit_and_live_sync_required": bool(
            execution_contract.get("commit_and_live_sync_required", True)
        ),
        "quant_rabbit_code_may_call_model_api": False,
        "approval_boundary": approval_boundary,
        "last_dispatched_material_digest": last_dispatched_material_digest or None,
        "dispatched_material_digests": dispatched_material_digests,
        "pending_material_digest": pending_material_digest or None,
        "acknowledged_material_digest": acknowledged_material_digest or None,
        "acknowledgement_replayed": acknowledgement_replayed,
        "automation_prompt": (
            "Implement the selected QuantRabbit repair using the suggested files and tests. "
            "Do not send orders, cancel orders, close positions, mutate launchd, or add model API calls "
            "inside QuantRabbit code; those actions require the existing gateway path or explicit operator approval."
        ),
    }


def _work_order_proof_state(loop_prompt: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(loop_prompt, dict):
        return {}
    current_state = loop_prompt.get("current_state")
    if not isinstance(current_state, dict):
        return {}
    keys = [
        "proof_queue_count",
        "proof_ready_count",
        "can_create_live_permission_count",
        "rejected_proof_candidate_count",
        "as_live_ready_path_exists",
        "proof_normal_routing_status",
        "proof_routing_allowed",
        "proof_primary_blocker",
        "proof_global_blockers",
        "portfolio_status",
        "portfolio_can_reach_4x_now",
        "gateway_status",
        "gateway_issue_codes",
        "proof_queue_empty_reason",
        "next_evidence_actions",
        "evidence_action_progress",
        "active_lane_evidence_work",
        "active_lane_action_code",
        "active_lane_evidence_material_digest",
    ]
    return {
        key: current_state.get(key)
        for key in keys
        if current_state.get(key) not in (None, [], {})
    }


def _read_only_evidence_work_order(
    *,
    proof_state: dict[str, Any],
    status: str,
    trader_request: str,
    approval_boundary: dict[str, Any],
    loop_prompt: dict[str, Any] | None,
    dispatched_material_digests: list[str],
    pending_material_digest: str,
    acknowledged_material_digest: str,
    acknowledgement_replayed: bool,
) -> dict[str, Any]:
    active_work = proof_state.get("active_lane_evidence_work")
    if not isinstance(active_work, dict) or not active_work.get("lane_id"):
        return {}
    material_digest = str(active_work.get("material_digest") or "")
    action_code = str(active_work.get("action_code") or ACTIVE_LANE_UNRESOLVED_ACTION)
    command_template = _active_lane_evidence_commands(action_code)
    commands = [
        command.format(material_digest=material_digest)
        if "{material_digest}" in command
        else command
        for command in command_template
    ]
    command_steps = [
        {
            **step,
            "command": (
                str(step.get("command") or "").format(material_digest=material_digest)
                if "{material_digest}" in str(step.get("command") or "")
                else str(step.get("command") or "")
            ),
        }
        for step in _active_lane_command_steps(command_template)
    ]
    previous_history = _bounded_digest_history(dispatched_material_digests)
    previous_digest = previous_history[-1] if previous_history else ""
    repeated_unchanged = bool(material_digest and material_digest in previous_history)
    dispatchable_stage = bool(command_template)
    wait_only_stage = _active_lane_wait_only_action(action_code)
    unsupported_stage = not dispatchable_stage and not wait_only_stage
    execution_pending = dispatchable_stage and not repeated_unchanged
    new_dispatch_issued = bool(
        execution_pending and material_digest != pending_material_digest
    )
    superseded_pending_material_digest = (
        pending_material_digest
        if pending_material_digest and pending_material_digest != material_digest
        else ""
    )
    current_pending_material_digest = material_digest if execution_pending else ""
    dispatch_allowed = execution_pending
    last_dispatched_material_digest = (
        previous_history[-1] if previous_history else ""
    )
    work_status = (
        READ_ONLY_EVIDENCE_WORK_STATUS
        if execution_pending
        else (
            READ_ONLY_EVIDENCE_MAPPING_REQUIRED_STATUS
            if unsupported_stage
            else READ_ONLY_EVIDENCE_WAIT_STATUS
        )
    )
    current_state = (
        loop_prompt.get("current_state")
        if isinstance(loop_prompt, dict)
        and isinstance(loop_prompt.get("current_state"), dict)
        else {}
    )
    material_change_condition = _success_condition(
        "any",
        [
            _condition_check(
                "active_lane_evidence_material_digest",
                "neq",
                material_digest,
            ),
            _condition_check("active_lane_action_code", "neq", action_code),
        ],
        description=(
            "Reopen this lane only after its material evidence watermark or current action stage changes."
        ),
    )
    condition_evaluation = _evaluate_success_condition(
        material_change_condition,
        current_state,
    )
    if repeated_unchanged:
        reason_code = "MATERIAL_EVIDENCE_UNCHANGED"
        selection_reason = (
            "The same lane action and material evidence watermark were acknowledged as completed; "
            "wait for proof, blocker, lane, or action-stage change instead of repeating it."
        )
    elif wait_only_stage:
        reason_code = "ACTION_STAGE_WAIT"
        selection_reason = (
            f"The current lane stage {action_code} is explicitly wait-only; future follow-up "
            "actions must not run before its material condition changes."
        )
    elif unsupported_stage:
        reason_code = "ACTION_STAGE_MAPPING_REQUIRED"
        selection_reason = (
            f"The current lane stage {action_code} has no approved command mapping. Surface "
            "the contract gap for code repair instead of silently waiting or guessing a command."
        )
    elif not new_dispatch_issued:
        reason_code = "DISPATCH_PENDING_ACK"
        selection_reason = (
            "The same stable active-lane dispatch is still pending completion acknowledgement. "
            "Preserve its dispatch id and command plan; do not create a second dispatch."
        )
    else:
        reason_code = "NEW_OR_CHANGED_MATERIAL_EVIDENCE"
        selection_reason = (
            "No repair request is queued, and the active lane exposes one current, "
            "blocker-preserving read-only evidence stage that has not been dispatched for "
            "this material watermark."
        )
    suggested_files = (
        [
            "src/quant_rabbit/trader_repair_orchestrator.py",
            "tests/test_trader_repair_orchestrator.py",
            "docs/AGENT_CONTRACT.md",
        ]
        if unsupported_stage
        else []
    )
    targeted_test_commands = (
        ["PYTHONPATH=src python3 -m unittest tests.test_trader_repair_orchestrator -v"]
        if unsupported_stage
        else []
    )
    deliverables = (
        [
            "explicit_action_code_to_read_only_command_mapping_or_documented_wait_classification",
            "regression_test_for_the_producer_action_text",
            "updated_runtime_contract_docs",
            "passing_targeted_and_full_tests",
            "git_commit_with_codex_attribution",
            "verified_live_runtime_sync",
        ]
        if unsupported_stage
        else [
            "read_only_artifact_refresh_or_documented_wait_condition",
            "no_live_side_effects",
            "updated_support_goal_loop_orchestrator_artifacts",
        ]
    )
    return {
        "status": work_status,
        "orchestrator_status": status,
        "trader_request": trader_request,
        "objective": (
            f"Advance or wait for material read-only lane evidence for {active_work.get('lane_id')} "
            "without repeating an unchanged stage."
        ),
        "selection_reason": selection_reason,
        "reason_code": reason_code,
        "active_lane_evidence_work": {
            **active_work,
            "status": work_status,
            "command_plan_available": dispatch_allowed,
        },
        "action_code": action_code,
        "material_digest": material_digest,
        "previous_material_digest": previous_digest or None,
        "last_dispatched_material_digest": last_dispatched_material_digest or None,
        "dispatched_material_digests": previous_history,
        "pending_material_digest": current_pending_material_digest or None,
        "pending_dispatch_id": current_pending_material_digest or None,
        "acknowledged_material_digest": acknowledged_material_digest or None,
        "acknowledgement_replayed": acknowledgement_replayed,
        "superseded_pending_material_digest": superseded_pending_material_digest or None,
        "dispatch_allowed": dispatch_allowed,
        "new_dispatch_issued": new_dispatch_issued,
        "execution_pending": execution_pending,
        "repeat_suppressed": repeated_unchanged,
        "suggested_commands": commands if dispatch_allowed else [],
        "suggested_command_steps": command_steps if dispatch_allowed else [],
        "suggested_files": suggested_files,
        "required_tests": (
            ["unknown action stays fail-closed", "mapped action advances exactly one stage"]
            if unsupported_stage
            else []
        ),
        "targeted_test_commands": targeted_test_commands,
        "verification_commands": [
            "python3 -m json.tool data/trader_support_bot.json >/dev/null",
            "python3 -m json.tool data/trader_goal_loop_orchestrator.json >/dev/null",
        ],
        "final_verification_commands": [
            "python3 -m json.tool data/trader_repair_orchestrator.json >/dev/null",
        ],
        "deliverables": deliverables,
        "proof_state": {
            **proof_state,
            "active_lane_evidence_work": {
                **active_work,
                "status": work_status,
                "command_plan_available": dispatch_allowed,
            },
        },
        "material_change_condition": material_change_condition,
        "material_change_condition_evaluation": condition_evaluation,
        "commit_and_live_sync_required": unsupported_stage,
        "quant_rabbit_code_may_call_model_api": False,
        "approval_boundary": approval_boundary,
        "live_side_effects": [],
        "live_permission_allowed": False,
        "automation_prompt": (
            (
                f"Implement an explicit fail-closed mapping or wait classification for {action_code}, "
                "add regression tests, update the runtime contract, commit, and sync live. "
                "Do not guess from prose or add live side effects."
            )
            if unsupported_stage
            else (
                "Execute only the suggested read-only evidence commands when useful, then inspect the "
                "refreshed support/goal-loop state. Do not send orders, cancel orders, close positions, "
                "mutate launchd, relax gates, or treat this work order as live permission."
            )
        ),
        "loop_prompt_version": (
            loop_prompt.get("version") if isinstance(loop_prompt, dict) else None
        ),
    }


def _active_lane_wait_only_action(action_code: str) -> bool:
    return action_code in {
        ACTIVE_LANE_WAIT_ACTION,
        ACTIVE_LANE_KEEP_BLOCKED_ACTION,
        ACTIVE_LANE_PRESERVE_BLOCKERS_ACTION,
    }


def _previous_dispatched_active_lane_material_digests(
    previous_work_order: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(previous_work_order, dict):
        return []
    history = _dedupe_strings(previous_work_order.get("dispatched_material_digests"))
    last = str(previous_work_order.get("last_dispatched_material_digest") or "").strip()
    if last:
        history.append(last)
    has_dispatch_lifecycle = any(
        key in previous_work_order
        for key in (
            "pending_material_digest",
            "pending_dispatch_id",
            "execution_pending",
            "new_dispatch_issued",
        )
    )
    if (
        previous_work_order.get("status") == READ_ONLY_EVIDENCE_WORK_STATUS
        and not has_dispatch_lifecycle
    ):
        direct = str(previous_work_order.get("material_digest") or "").strip()
        if direct:
            history.append(direct)
        active_work = previous_work_order.get("active_lane_evidence_work")
        if isinstance(active_work, dict):
            nested = str(active_work.get("material_digest") or "").strip()
            if nested:
                history.append(nested)
    return _bounded_digest_history(history)


def _previous_pending_active_lane_material_digest(
    previous_work_order: dict[str, Any] | None,
) -> str:
    if not isinstance(previous_work_order, dict):
        return ""
    direct = str(previous_work_order.get("pending_material_digest") or "").strip()
    if direct:
        return direct
    if (
        previous_work_order.get("status") == READ_ONLY_EVIDENCE_WORK_STATUS
        and previous_work_order.get("dispatch_allowed") is True
    ):
        return str(previous_work_order.get("material_digest") or "").strip()
    return ""


def _bounded_digest_history(values: list[str]) -> list[str]:
    recent: list[str] = []
    for value in values:
        digest = str(value or "").strip()
        if not digest:
            continue
        if digest in recent:
            recent.remove(digest)
        recent.append(digest)
    return recent[-ACTIVE_LANE_DISPATCH_HISTORY_LIMIT:]


def _queue_item(request: dict[str, Any], *, trader_request: str) -> dict[str, Any]:
    contract = request.get("automation_contract") if isinstance(request.get("automation_contract"), dict) else {}
    dependency_wait = _approval_dependency_wait(request) or _operator_review_dependency_wait(request)
    explicit = bool(request.get("requires_explicit_operator_approval")) or dependency_wait is not None
    allowed_actions = [str(item) for item in contract.get("codex_may_execute", []) or []]
    approval_actions = [str(item) for item in contract.get("requires_explicit_operator_approval_for", []) or []]
    forbidden_actions = [str(item) for item in contract.get("forbidden_direct_actions", []) or []]
    if not allowed_actions:
        allowed_actions = list(REPAIR_AUTOMATION_ALLOWED_ACTIONS)
    if not approval_actions:
        approval_actions = list(REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS)
    if dependency_wait and dependency_wait.get("requires_explicit_operator_approval_for"):
        approval_actions = _dedupe_strings(
            [
                *approval_actions,
                *dependency_wait.get("requires_explicit_operator_approval_for", []),
            ]
        )
    if not forbidden_actions:
        forbidden_actions = list(REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS)

    automation_status = _automation_status(request, requires_explicit_operator_approval=explicit)
    verification_commands = _dedupe_strings(request.get("verification_commands"))
    targeted_tests = _targeted_test_commands(request.get("suggested_files"))
    dependency_rank = _dependency_rank(request.get("code"))
    return {
        "code": request.get("code"),
        "priority": request.get("priority"),
        "source_findings": _dedupe_strings(request.get("source_findings")),
        "repair_status": request.get("status"),
        "automation_status": automation_status,
        "dependency_rank": dependency_rank,
        "selection_reason": _selection_reason(request.get("code")),
        "match_score": _match_score(request, trader_request),
        "requires_explicit_operator_approval": explicit,
        "approval_dependency": dependency_wait,
        "problem": request.get("problem"),
        "why_now": request.get("why_now"),
        "evidence_summary": (
            request.get("evidence_summary")
            if isinstance(request.get("evidence_summary"), dict)
            else {}
        ),
        "clearance_conditions": _dedupe_strings(request.get("clearance_conditions")),
        "verification_commands": verification_commands,
        "suggested_files": _dedupe_strings(request.get("suggested_files")),
        "required_tests": _dedupe_strings(request.get("required_tests")),
        "targeted_test_commands": targeted_tests,
        "final_verification_commands": _dedupe_strings(
            [
                *targeted_tests,
                *verification_commands,
                "PYTHONPATH=src python3 -m unittest discover -s tests -v",
                "git status --short",
                "scripts/sync-live-runtime.sh",
            ]
        ),
        "implementation_loop": (
            [
                "read_support_artifacts",
                "inspect_suggested_files",
                "implement_code_and_tests",
                "run_targeted_tests",
                "run_required_verification_commands",
                "commit_with_codex_attribution",
                "sync_live_runtime",
                "verify_live_head_matches_promoted_commit",
            ]
            if not explicit
            and automation_status == AUTOMATION_READY
            else [
                "run_preflight_read_only_checks",
                (
                    "wait_for_explicit_operator_approval"
                    if automation_status == AUTOMATION_OPERATOR_APPROVAL
                    else "wait_for_required_live_evidence_or_acceptance_window"
                ),
                (
                    "execute_only_the_approved_external_or_gateway_action"
                    if automation_status == AUTOMATION_OPERATOR_APPROVAL
                    else "refresh_support_artifacts_without_live_side_effects"
                ),
                "refresh_support_artifacts",
            ]
        ),
        "automation_contract": {
            "codex_may_execute": allowed_actions,
            "commit_and_live_sync_required": bool(contract.get("commit_and_live_sync_required", True)),
            "quant_rabbit_code_may_call_model_api": False,
            "live_side_effects_allowed": [],
            "requires_explicit_operator_approval_for": approval_actions,
            "forbidden_direct_actions": forbidden_actions,
            "orders_closes_launchd_policy": contract.get("orders_closes_launchd_policy")
            or _execution_contract()["orders_closes_launchd_policy"],
        },
        "read_only": True,
        "live_side_effects": [],
    }


def _select_request(queue: list[dict[str, Any]], *, trader_request: str) -> dict[str, Any]:
    if not queue:
        return {}
    if trader_request.strip():
        return min(
            queue,
            key=lambda item: (
                -int(item.get("match_score") or 0),
                _priority_rank(item.get("priority")),
                _dependency_rank(item.get("code")),
                str(item.get("code") or ""),
            ),
        )
    return queue[0]


def _queue_sort_key(item: dict[str, Any]) -> tuple[int, int, int, int, str]:
    automation_rank = _automation_rank(item.get("automation_status"))
    priority_rank = _priority_rank(item.get("priority"))
    return (
        automation_rank,
        priority_rank,
        _dependency_rank(item.get("code")),
        -int(item.get("match_score") or 0),
        str(item.get("code") or ""),
    )


def _automation_status(
    request: dict[str, Any],
    *,
    requires_explicit_operator_approval: bool,
) -> str:
    if requires_explicit_operator_approval:
        return AUTOMATION_OPERATOR_APPROVAL
    if _request_waits_for_proof_evidence(request):
        return AUTOMATION_LIVE_EVIDENCE_WINDOW
    status = str(request.get("status") or "").upper()
    if status in NON_ACTIONABLE_REPAIR_STATUSES:
        return AUTOMATION_LIVE_EVIDENCE_WINDOW
    if status in CODEX_ACTIONABLE_REPAIR_STATUSES or status.startswith("READY_FOR_CODE"):
        return AUTOMATION_READY
    return AUTOMATION_EVIDENCE


def _request_waits_for_proof_evidence(request: dict[str, Any]) -> bool:
    if str(request.get("code") or "") != "REPAIR_FRONTIER_LANE_BLOCKER":
        return False
    evidence = (
        request.get("evidence_summary")
        if isinstance(request.get("evidence_summary"), dict)
        else {}
    )
    blocker_codes = {
        str(evidence.get("code") or ""),
        *(
            str(item)
            for item in request.get("source_findings", [])
            if isinstance(item, str)
        ),
    }
    return bool(blocker_codes & FRONTIER_PROOF_EVIDENCE_BLOCKER_CODES)


def _approval_dependency_wait(request: dict[str, Any]) -> dict[str, Any] | None:
    code = str(request.get("code") or "")
    if code != "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY":
        return None
    evidence = request.get("evidence_summary") if isinstance(request.get("evidence_summary"), dict) else {}
    if evidence.get("current_guardian_active") is True and evidence.get("current_guardian_heartbeat_fresh") is True:
        return None
    if (
        evidence.get("current_guardian_live_runtime_lock_active") is True
        or evidence.get("current_guardian_active_source") == "live_runtime_lock_busy"
    ):
        return None
    source_findings = set(_dedupe_strings(request.get("source_findings")))
    if not evidence.get("guardian_profit_capture_inactive"):
        return None
    replay_trigger_value = evidence.get(
        "post_repair_live_evidence_loss_closes_repair_replay_triggered"
    )
    if replay_trigger_value is None:
        replay_trigger_value = evidence.get("loss_closes_repair_replay_triggered")
    replay_triggered = _optional_positive(replay_trigger_value)
    if "TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED" not in source_findings and not replay_triggered:
        return None
    return {
        "code": "RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT",
        "reason": (
            "TP-progress production-gate replay is already present, but the repaired "
            "path cannot be proved while position guardian is inactive; loading or "
            "reloading launchd needs explicit operator approval."
        ),
        "clearance": (
            "Run scripts/install-position-guardian.sh --check, receive explicit "
            "operator approval to load/reload guardian, then wait for a fresh "
            "heartbeat and rerun execution-timing-audit after a live evidence window."
        ),
    }


def _operator_review_dependency_wait(request: dict[str, Any]) -> dict[str, Any] | None:
    if not _request_targets_operator_review_clearance(request):
        return None
    return {
        "code": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
        "reason": (
            "The selected repair request is blocked by an expired guardian receipt that "
            "requires a fresh explicit operator review. Codex must not clear this by "
            "changing intent, risk, or gateway code."
        ),
        "clearance": (
            "Record an explicit local operator decision via the guardian receipt operator-review "
            "flow, then rebuild guardian-receipt-consumption, generate-intents, "
            "trader-support-bot, and trader-repair-orchestrator."
        ),
        "requires_explicit_operator_approval_for": ["guardian_operator_review"],
    }


def _request_targets_operator_review_clearance(request: dict[str, Any]) -> bool:
    code = str(request.get("code") or "")
    if code != "REPAIR_FRONTIER_LANE_BLOCKER":
        return False
    evidence = request.get("evidence_summary") if isinstance(request.get("evidence_summary"), dict) else {}
    primary = str(evidence.get("code") or evidence.get("blocker_code") or "").upper()
    source_findings = {item.upper() for item in _dedupe_strings(request.get("source_findings"))}
    if primary in OPERATOR_REVIEW_APPROVAL_BLOCKERS:
        return True
    return bool(source_findings.intersection(OPERATOR_REVIEW_APPROVAL_BLOCKERS))


def _optional_positive(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _automation_rank(value: Any) -> int:
    status = str(value or "")
    if status == AUTOMATION_READY:
        return 0
    if status == AUTOMATION_OPERATOR_APPROVAL:
        return 1
    return 2


def _dependency_rank(code: Any) -> int:
    return REPAIR_DEPENDENCY_RANK.get(str(code or ""), 50)


def _selection_reason(code: Any) -> str:
    return REPAIR_SELECTION_REASONS.get(
        str(code or ""),
        "Repair queue position is based on automation status, priority, dependency rank, request match, and code.",
    )


def _priority_rank(value: Any) -> int:
    text = str(value or "").upper()
    if text == "P0":
        return 0
    if text == "P1":
        return 1
    if text == "P2":
        return 2
    return 9


def _match_score(request: dict[str, Any], trader_request: str) -> int:
    terms = _terms(trader_request)
    if not terms:
        return 0
    haystack = " ".join(
        str(part)
        for part in [
            request.get("code"),
            request.get("priority"),
            request.get("status"),
            request.get("problem"),
            request.get("why_now"),
            " ".join(_dedupe_strings(request.get("source_findings"))),
            " ".join(_dedupe_strings(request.get("suggested_files"))),
        ]
        if part
    ).lower()
    return sum(1 for term in terms if term in haystack)


def _terms(text: str) -> list[str]:
    raw = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9_./:-]+|[ぁ-んァ-ン一-龯]+", raw)
    synonyms = {
        "利益": "profit",
        "利確": "profit",
        "決済": "close",
        "修正": "repair",
        "証拠": "evidence",
        "検証": "verification",
        "ガーディアン": "guardian",
        "予測": "forecast",
        "精度": "precision",
        "逆": "inversion",
        "反対": "opposite",
        "目標": "target",
    }
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        mapped = synonyms.get(token)
        if mapped:
            expanded.append(mapped)
    return list(dict.fromkeys(expanded))


def _dedupe_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raw = [value]
    elif isinstance(value, list):
        raw = value
    else:
        raw = [value]
    return list(dict.fromkeys(str(item) for item in raw if str(item)))


def _targeted_test_commands(files: Any) -> list[str]:
    commands: list[str] = []
    for item in _dedupe_strings(files):
        path = Path(item)
        if path.parts and path.parts[0] == "tests" and path.suffix == ".py":
            module = ".".join(path.with_suffix("").parts)
            commands.append(f"PYTHONPATH=src python3 -m unittest {module} -v")
    return list(dict.fromkeys(commands))


def _render_report(payload: dict[str, Any]) -> str:
    selected = payload.get("selected_request") if isinstance(payload.get("selected_request"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    contract = payload.get("execution_contract") if isinstance(payload.get("execution_contract"), dict) else {}
    work_order = payload.get("codex_work_order") if isinstance(payload.get("codex_work_order"), dict) else {}
    boundary = payload.get("approval_boundary") if isinstance(payload.get("approval_boundary"), dict) else {}
    loop_prompt = (
        payload.get("loop_engineering_prompt")
        if isinstance(payload.get("loop_engineering_prompt"), dict)
        else {}
    )
    loop_state = (
        loop_prompt.get("current_state")
        if isinstance(loop_prompt.get("current_state"), dict)
        else {}
    )
    lines = [
        "# Trader Repair Orchestrator Report",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Trader request: `{payload.get('trader_request') or ''}`",
        f"- Selected request: `{selected.get('code') if selected else None}`",
        f"- Actionable requests: `{metrics.get('actionable_request_count')}`",
        f"- Approval-required requests: `{metrics.get('approval_required_request_count')}`",
        f"- Waiting requests: `{metrics.get('waiting_request_count')}`",
        f"- Repair request source: `{metrics.get('repair_request_source')}`",
        f"- Recovered from embedded support: `{metrics.get('recovered_from_embedded_support')}`",
        f"- Read only: `{payload.get('read_only')}`",
        f"- Live side effects: `{len(payload.get('live_side_effects') or [])}`",
        "",
        "## Execution Contract",
        "",
        f"- Codex may execute: `{', '.join(contract.get('codex_may_execute') or [])}`",
        f"- Approval required for: `{', '.join(contract.get('requires_explicit_operator_approval_for') or [])}`",
        f"- Forbidden direct actions: `{', '.join(contract.get('forbidden_direct_actions') or [])}`",
        f"- Commit and live sync required: `{contract.get('commit_and_live_sync_required')}`",
        f"- QuantRabbit code may call model API: `{contract.get('quant_rabbit_code_may_call_model_api')}`",
        f"- Policy: {contract.get('orders_closes_launchd_policy')}",
        "",
        "## Codex Work Order",
        "",
        f"- Status: `{work_order.get('status')}`",
        f"- Selection reason: {work_order.get('selection_reason')}",
        f"- Objective: {work_order.get('objective')}",
        f"- Reason code: `{work_order.get('reason_code')}`",
        f"- Active lane action: `{work_order.get('action_code')}`",
        f"- Material digest: `{work_order.get('material_digest')}`",
        f"- Pending dispatch id: `{work_order.get('pending_dispatch_id')}`",
        f"- Execution pending: `{work_order.get('execution_pending')}`",
        f"- Acknowledged digest: `{work_order.get('acknowledged_material_digest')}`",
        f"- Dispatch allowed: `{work_order.get('dispatch_allowed')}`",
        f"- Repeat suppressed: `{work_order.get('repeat_suppressed')}`",
        f"- Suggested commands: `{', '.join(work_order.get('suggested_commands') or [])}`",
        f"- Evidence summary keys: `{', '.join(sorted((work_order.get('evidence_summary') or {}).keys()))}`",
        f"- Deliverables: `{', '.join(work_order.get('deliverables') or [])}`",
        f"- Final verification: `{', '.join(work_order.get('final_verification_commands') or [])}`",
        f"- Commit and live sync required: `{work_order.get('commit_and_live_sync_required')}`",
        f"- Read-only until gateway or approval: `{boundary.get('read_only_until_gateway_or_operator_approval')}`",
        f"- Approval required for: `{', '.join(boundary.get('requires_explicit_operator_approval_for') or [])}`",
        f"- Existing gateway paths: `{json.dumps(boundary.get('existing_gateway_paths') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Loop Engineering Prompt",
        "",
        f"- Version: `{loop_prompt.get('version')}`",
        f"- Objective: {loop_prompt.get('objective')}",
        f"- Hypothesis: {loop_prompt.get('current_hypothesis')}",
        f"- 4x funding-adjusted multiplier: `{loop_state.get('rolling_30d_multiplier_funding_adjusted')}`",
        f"- Remaining to 4x funding-adjusted: `{loop_state.get('remaining_to_4x_funding_adjusted')}`",
        f"- Required calendar daily return funding-adjusted: `{loop_state.get('required_calendar_daily_return_funding_adjusted')}`",
        f"- Required active-day return funding-adjusted: `{loop_state.get('required_active_day_return_funding_adjusted')}`",
        f"- Performance basis: `{loop_state.get('performance_basis')}`",
        f"- Sizing basis: `{loop_state.get('sizing_basis')}`",
        f"- Pace state: `{loop_state.get('pace_state')}`",
        f"- Operational 5pct reachable: `{loop_state.get('operational_minimum_5pct_reachable')}`",
        f"- Audit 5pct estimated reachable: `{loop_state.get('audit_minimum_5pct_estimated_reachable')}`",
        f"- Live-ready lanes: `{loop_state.get('live_ready_lanes')}`",
        f"- Proof queue count: `{loop_state.get('proof_queue_count')}`",
        f"- Proof-ready count: `{loop_state.get('proof_ready_count')}`",
        f"- Live permission candidates: `{loop_state.get('can_create_live_permission_count')}`",
        f"- Rejected proof candidates: `{loop_state.get('rejected_proof_candidate_count')}`",
        f"- A/S proof primary blocker: `{loop_state.get('proof_primary_blocker')}`",
        f"- Portfolio status: `{loop_state.get('portfolio_status')}`",
        f"- Gateway status: `{loop_state.get('gateway_status')}`",
        f"- Proof queue empty reason: `{json.dumps(loop_state.get('proof_queue_empty_reason') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Next evidence actions: `{json.dumps(loop_state.get('next_evidence_actions') or [], ensure_ascii=False, sort_keys=True)}`",
        f"- Guardian active: `{loop_state.get('guardian_active')}`",
        f"- Artifact contradictions: `{', '.join(loop_state.get('artifact_contradiction_codes') or [])}`",
        f"- Actionable: `{', '.join(loop_state.get('actionable_request_codes') or [])}`",
        f"- Approval required: `{', '.join(loop_state.get('approval_required_request_codes') or [])}`",
        f"- Waiting: `{', '.join(loop_state.get('waiting_request_codes') or [])}`",
        f"- Next loop: `{'; '.join(loop_prompt.get('next_loop') or [])}`",
        f"- Verification: `{', '.join(loop_prompt.get('verification_commands') or [])}`",
        "",
        "```text",
        str(loop_prompt.get("prompt_text") or ""),
        "```",
        "",
        "## Queue",
        "",
    ]
    queue = payload.get("queue") if isinstance(payload.get("queue"), list) else []
    if queue:
        lines.extend(
            [
                "| Code | Priority | Repair status | Automation | Match | Dependency | Clearance | Verify |",
                "|---|---|---|---|---:|---:|---|---|",
            ]
        )
        for item in queue[:12]:
            verify = ", ".join(f"`{command}`" for command in item.get("verification_commands", [])[:2]) or "none"
            clearance_conditions = _dedupe_strings(item.get("clearance_conditions"))
            clearance = clearance_conditions[0] if clearance_conditions else "none"
            lines.append(
                f"| `{item.get('code')}` | `{item.get('priority')}` | "
                f"`{item.get('repair_status')}` | `{item.get('automation_status')}` | "
                f"`{item.get('match_score')}` | `{item.get('dependency_rank')}` | "
                f"{clearance} | {verify} |"
            )
    else:
        lines.append("- none")
    if selected:
        lines.extend(
            [
                "",
                "## Selected Request",
                "",
                f"- Code: `{selected.get('code')}`",
                f"- Dependency rank: `{selected.get('dependency_rank')}`",
                f"- Selection reason: {selected.get('selection_reason')}",
                f"- Problem: {selected.get('problem')}",
                f"- Why now: {selected.get('why_now')}",
                f"- Suggested files: `{', '.join(selected.get('suggested_files') or [])}`",
                f"- Targeted tests: `{', '.join(selected.get('targeted_test_commands') or [])}`",
                f"- Final verification: `{', '.join(selected.get('final_verification_commands') or [])}`",
            ]
        )
    return "\n".join(lines) + "\n"

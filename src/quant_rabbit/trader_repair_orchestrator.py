from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_TRADER_REPAIR_ORCHESTRATOR,
    DEFAULT_TRADER_REPAIR_ORCHESTRATOR_REPORT,
    DEFAULT_TRADER_SUPPORT_BOT,
)
from quant_rabbit.trader_support_bot import (
    BIDASK_REPLAY_WAIT_STATUS,
    DIRECTIONAL_INVERSION_COUNTERFACTUAL_REQUEST,
    DIRECTIONAL_INVERSION_REPLAY_WAIT_STATUS,
    FRONTIER_MARGIN_CAPACITY_WAIT_STATUS,
    FRONTIER_QUOTE_FRESHNESS_WAIT_STATUS,
    OANDA_AUDIT_ONLY_LOCAL_TP_EDGE_REQUEST,
    OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_UNPROVED_STATUS,
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
AUTOMATION_OPERATOR_APPROVAL = "WAITING_FOR_OPERATOR_APPROVAL"
AUTOMATION_LIVE_EVIDENCE_WINDOW = "WAITING_FOR_LIVE_EVIDENCE_WINDOW"
AUTOMATION_EVIDENCE = "WAITING_FOR_EVIDENCE"
NON_ACTIONABLE_REPAIR_STATUSES = {
    "FORECAST_FRONTIER_WAITING_FOR_LIVE_PRECISION_EVIDENCE",
    "FRONTIER_PROTECTIVE_GUARDRAIL_ACTIVE",
    "HISTORICAL_ACCEPTANCE_WINDOW_ACTIVE",
    "RESIDUAL_GROUPS_ALREADY_BLOCKED_WAITING_FOR_REPLAY",
    FRONTIER_MARGIN_CAPACITY_WAIT_STATUS,
    FRONTIER_QUOTE_FRESHNESS_WAIT_STATUS,
    TP_PROGRESS_GUARDIAN_WAIT_STATUS,
    TP_PROGRESS_LIVE_EVIDENCE_WAIT_STATUS,
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
        "A broker-truth opposite-side counterfactual would clear the 5% minimum target, so "
        "Codex must audit forecast inversion and opposite-lane suppression before adding "
        "unrelated entry frequency."
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
        trader_request: str | None = None,
        now_utc: datetime | None = None,
    ) -> None:
        self.support_bot_path = support_bot_path
        self.output_path = output_path
        self.report_path = report_path
        self.trader_request = trader_request or ""
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> TraderRepairOrchestratorSummary:
        payload = self.build_payload()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload), encoding="utf-8")
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
        support = _read_json(self.support_bot_path)
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
            ),
            "read_only": True,
            "live_side_effects": [],
        }
        return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path), "repair_requests": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


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
        "support_blocker_codes": _support_blocker_codes(support.get("blockers")),
        "campaign_day_jst": target.get("campaign_day_jst"),
        "target_status": target.get("status"),
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
    current_state["profitability_rca_summary"] = _profitability_rca_summary(
        acceptance=acceptance,
        target_firepower=target_firepower,
    )
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
    verification_commands = _loop_verification_commands(
        selected=selected,
        approval_required=approval_required,
        waiting=waiting,
    )
    if artifact_contradictions:
        refresh_commands: list[str] = []
        for item in artifact_contradictions:
            refresh_commands.extend(_dedupe_strings(item.get("refresh_commands")))
        verification_commands = _dedupe_strings([*refresh_commands, *verification_commands])
    anti_loop_rules = [
        "Do not treat audit-only 5% firepower as operational reachability unless operational_minimum_5pct_reachable is true.",
        "If the selected actionable request is lower priority than a waiting P0 blocker, treat it as auxiliary evidence work, not as clearing the P0 or proving operational 5%.",
        "If fresher support state contradicts intent blocker counts, classify that blocker as artifact-stale and refresh the evidence packet before selecting repair work from it.",
        "Do not rerun profitability-acceptance as the fix unless an input artifact, gateway proof, or live evidence window changed first.",
        "If OANDA audit-only S5/M5 history is complete and replay cannot clear local TP proof, do not rerun validate/mine/package; wait for new local TP receipts, new forecast/candle evidence, or exact HARVEST live-grade promotion.",
        "Do not lower MIN_PRODUCTION_LOT_UNITS, bypass MARGIN_TOO_THIN_FOR_MIN_LOT, synthesize PASS close evidence, or loosen protective market-structure guards without a failing regression and a positive-path test.",
        "Do not send orders, cancel orders, close positions, mutate launchd, or call model APIs from QuantRabbit code outside the existing gateway or explicit operator approval boundary.",
        "If the top item is waiting for live evidence, collect or wait for the named evidence; do not reimplement the same already-blocking guard.",
    ]
    self_review_questions = [
        "What single blocker currently prevents operational 5% reachability, and is it causal rather than merely frequent?",
        "Does the next action change broker state, launchd state, order state, or position state? If yes, which approved gateway or explicit operator approval covers it?",
        "Is any blocker contradicted by fresher support state, and did I refresh the stale artifact before treating it as causal?",
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
            "Drive QuantRabbit toward the daily 5% minimum from starting equity by repeatedly "
            "selecting the highest-causal blocker, taking only approved code/evidence actions, "
            "and verifying that operational reachability improves without bypassing broker truth."
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
        "Goal: pursue the 5% daily minimum as an audit/repair obligation, not as a promise of market returns.",
        f"Trader request: {trader_request or '(none)'}",
        f"State: orchestrator={current_state.get('orchestrator_status')}, "
        f"target={current_state.get('target_status')}, live_ready={current_state.get('live_ready_lanes')}, "
        f"guardian_active={current_state.get('guardian_active')}, "
        f"guardian_lock={current_state.get('guardian_live_runtime_lock_active')}, "
        f"operational_5pct={current_state.get('operational_minimum_5pct_reachable')}, "
        f"audit_5pct={current_state.get('audit_minimum_5pct_estimated_reachable')}.",
        f"Queue: selected={current_state.get('selected_request_code')}, "
        f"waiting_p0={', '.join(current_state.get('waiting_p0_request_codes') or []) or '(none)'}, "
        f"support_blockers={', '.join(current_state.get('support_blocker_codes') or []) or '(none)'}.",
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
) -> dict[str, Any]:
    if not selected:
        return {
            "status": "NO_ACTIONABLE_CODEX_WORK",
            "orchestrator_status": status,
            "trader_request": trader_request,
            "reason": "No selected READY_FOR_CODEX_IMPLEMENTATION request is available.",
            "approval_boundary": approval_boundary,
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
        "commit_and_live_sync_required": bool(execution_contract.get("commit_and_live_sync_required", True)),
        "quant_rabbit_code_may_call_model_api": False,
        "approval_boundary": approval_boundary,
        "automation_prompt": (
            "Implement the selected QuantRabbit repair using the suggested files and tests. "
            "Do not send orders, cancel orders, close positions, mutate launchd, or add model API calls "
            "inside QuantRabbit code; those actions require the existing gateway path or explicit operator approval."
        ),
    }


def _queue_item(request: dict[str, Any], *, trader_request: str) -> dict[str, Any]:
    contract = request.get("automation_contract") if isinstance(request.get("automation_contract"), dict) else {}
    dependency_wait = _approval_dependency_wait(request)
    explicit = bool(request.get("requires_explicit_operator_approval")) or dependency_wait is not None
    allowed_actions = [str(item) for item in contract.get("codex_may_execute", []) or []]
    approval_actions = [str(item) for item in contract.get("requires_explicit_operator_approval_for", []) or []]
    forbidden_actions = [str(item) for item in contract.get("forbidden_direct_actions", []) or []]
    if not allowed_actions:
        allowed_actions = list(REPAIR_AUTOMATION_ALLOWED_ACTIONS)
    if not approval_actions:
        approval_actions = list(REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS)
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
                    "execute_only_the_approved_gateway_or_launchd_action"
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
    status = str(request.get("status") or "").upper()
    if status in NON_ACTIONABLE_REPAIR_STATUSES:
        return AUTOMATION_LIVE_EVIDENCE_WINDOW
    if status in CODEX_ACTIONABLE_REPAIR_STATUSES or status.startswith("READY_FOR_CODE"):
        return AUTOMATION_READY
    return AUTOMATION_EVIDENCE


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
    replay_triggered = _optional_positive(evidence.get("loss_closes_repair_replay_triggered"))
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


def _optional_positive(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


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
        f"- Operational 5pct reachable: `{loop_state.get('operational_minimum_5pct_reachable')}`",
        f"- Audit 5pct estimated reachable: `{loop_state.get('audit_minimum_5pct_estimated_reachable')}`",
        f"- Live-ready lanes: `{loop_state.get('live_ready_lanes')}`",
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
                "| Code | Priority | Automation | Match | Dependency | Verify |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        for item in queue[:12]:
            verify = ", ".join(f"`{command}`" for command in item.get("verification_commands", [])[:2]) or "none"
            lines.append(
                f"| `{item.get('code')}` | `{item.get('priority')}` | "
                f"`{item.get('automation_status')}` | `{item.get('match_score')}` | "
                f"`{item.get('dependency_rank')}` | {verify} |"
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

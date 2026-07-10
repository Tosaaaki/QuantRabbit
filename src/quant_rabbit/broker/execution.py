from __future__ import annotations

import math
import hashlib
import json
import os
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from quant_rabbit.models import (
    BrokerSnapshot,
    MarketContext,
    OrderIntent,
    OrderType,
    Owner,
    Quote,
    RiskMetrics,
    Side,
    TradeMethod,
)


def _trader_sl_repair_disabled() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
from quant_rabbit.paths import (
    ROOT as _QR_ROOT,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_DAILY_TARGET_REPORT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_EXECUTION_LEDGER_REPORT,
    DEFAULT_GUARDIAN_ACTION_RECEIPT,
    DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
    DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_LIVE_ORDER_STAGE_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_QR_TRADER_RUN_WATCHDOG,
    DEFAULT_STRATEGY_PROFILE,
)
from quant_rabbit.execution_ledger import ExecutionLedger
from quant_rabbit.target import DailyTargetLedger
from quant_rabbit.predictive_scout import (
    PREDICTIVE_SCOUT_CLOCK_SKEW_SECONDS,
    PREDICTIVE_SCOUT_MAX_CONCURRENT,
    PREDICTIVE_SCOUT_MAX_SENT_PER_CAMPAIGN_DAY,
    predictive_scout_broker_signal_ids,
    predictive_scout_concurrent_count,
    predictive_scout_experiment_id,
    predictive_scout_geometry_claimed,
    predictive_scout_intent_claimed,
    predictive_scout_intent_issues,
    predictive_scout_metadata_supported,
    predictive_scout_policy,
    predictive_scout_signal_id,
    predictive_scout_vehicle_id,
)
from quant_rabbit.guardian_events import guardian_action_gateway_issues
from quant_rabbit.guardian_receipt_consumption import (
    BLOCK_NEW_ENTRY_CODE as GUARDIAN_RECEIPT_BLOCK_NEW_ENTRY_CODE,
    WATCHDOG_BLOCK_NEW_ENTRY_CODE as GUARDIAN_WATCHDOG_BLOCK_NEW_ENTRY_CODE,
    guardian_receipt_new_entry_blockers_from_paths,
)
from quant_rabbit.risk import (
    MIN_PRODUCTION_LOT_UNITS,
    RiskEngine,
    RiskIssue,
    RiskPolicy,
    _min_lot_test_override_active,
    resolve_max_loss_jpy,
)
from quant_rabbit.risk import DEFAULT_SPECS, estimate_required_margin_jpy, margin_budget_jpy
from quant_rabbit.self_improvement_guards import (
    forecast_adverse_path_exempted_by_tp_harvest_repair,
    forecast_adverse_path_new_risk_blocker,
    intent_matches_profitability_worst_segment,
    oanda_firepower_repair_current_risk_reaches_minimum,
    p0_code_exempted_by_tp_harvest_repair,
    profitability_p0_worst_segment,
)
from quant_rabbit.strategy.intent_generator import (
    MACRO_EVENT_MAX_RISK_PCT_NAV,
    _daily_risk_budget_from_state,
    _expired_pending_projection_count,
    _per_trade_risk_from_state,
)
from quant_rabbit.strategy.profile import StrategyProfile


class ExecutionClient(Protocol):
    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot: ...

    def post_order_json(self, order_request: dict[str, Any]) -> dict[str, Any]: ...

    def account_summary(self, *, now_utc: datetime | None = None) -> Any: ...

    def transactions_since_id(self, transaction_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LiveOrderStageSummary:
    status: str
    lane_id: str | None
    output_path: Path
    report_path: Path
    sent: bool
    risk_issues: int
    strategy_issues: int
    sent_count: int = 0
    lane_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PrePostReconciliationResult:
    intent: OrderIntent
    snapshot: BrokerSnapshot
    risk: Any
    order_request: dict[str, Any] | None
    attached_stop_metrics: dict[str, Any] | None
    max_loss_jpy: float
    portfolio_loss_cap_jpy: float | None
    size_multiple: float
    order_build_issues: list[dict[str, str]]
    issues: tuple[RiskIssue, ...]
    evidence: dict[str, Any]


@dataclass(frozen=True)
class _OrdinaryEntryPostClaimResult:
    evidence: dict[str, Any]
    issue: dict[str, str] | None = None


# Portfolio occupancy needs to scale with the campaign pace. Three active FX
# windows (Asia, London, NY) is the coarse session model already used by the
# archive outcome evidence; the fixed policy cap remains the fallback floor.
ACTIVE_FX_SESSION_BUCKETS_PER_DAY = 3

# Synthetic loss distance for trader-owned SL-free exposure in basket math.
# This mirrors `risk._open_portfolio_risk_jpy`: it is a structural accounting
# proxy for no-broker-SL mode, not an execution stop. Replace both copies with
# a shared policy field once SL-free risk accounting is centralized.
SL_FREE_SYNTHETIC_RISK_PIPS = 30.0

# Broker client-extension IDs are audit identifiers, not market/risk knobs. The
# prefix lets execution_ledger distinguish QuantRabbit-created orders from
# manual client IDs while the comment carries the full lane for attribution.
CLIENT_ORDER_ID_PREFIX = "qrv1"
TARGET_PATH_MAIN_ROLES = {"MAIN", "HERO", "PATH_A", "5PCT_PATH", "GUARANTEE_5", "PACE_5"}
TARGET_PATH_SUPPORT_ROLES = {"SCOUT", "RELOAD", "SECOND_SHOT", "SUPPORT", "PATH_B"}
TARGET_PATH_ATTACK_STACK_SLOTS = {"NOW", "RELOAD", "SECOND_SHOT"}
TARGET_PATH_GRADE_RANK = {"C": 0, "B-": 1, "B0": 2, "B": 2, "B+": 3, "A": 4, "S": 5}

# DailyTargetLedger is the final campaign state machine at the broker boundary.
# Only PURSUE_TARGET authorizes fresh risk.  The explicit deny set keeps known
# terminal/repair states auditable, while the allow-list makes a newly added or
# malformed status fail closed instead of silently becoming send permission.
PRE_POST_FRESH_ENTRY_ALLOWED_TARGET_STATUSES = frozenset({"PURSUE_TARGET"})
PRE_POST_FRESH_ENTRY_FORBIDDEN_TARGET_STATUSES = frozenset(
    {"TARGET_REACHED_PROTECT", "RISK_BUDGET_EXHAUSTED", "REPAIR_REQUIRED"}
)

class LiveOrderGateway:
    """Stage or send one OANDA order after the live risk contract passes."""

    def __init__(
        self,
        *,
        client: ExecutionClient,
        strategy_profile: Path = DEFAULT_STRATEGY_PROFILE,
        output_path: Path = DEFAULT_LIVE_ORDER_REQUEST,
        report_path: Path = DEFAULT_LIVE_ORDER_STAGE_REPORT,
        live_enabled: bool = False,
        max_loss_jpy: float | None = None,
        max_loss_pct: float | None = None,
        risk_equity_jpy: float | None = None,
        portfolio_loss_cap_jpy: float | None = None,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        target_report_path: Path | None = None,
        self_improvement_audit: Path | None = None,
        verified_decision_path: Path | None = None,
        guardian_action_receipt_path: Path | None = DEFAULT_GUARDIAN_ACTION_RECEIPT,
        qr_trader_run_watchdog_path: Path | None = DEFAULT_QR_TRADER_RUN_WATCHDOG,
        guardian_receipt_consumption_path: Path | None = DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
        guardian_receipt_operator_review_path: Path | None = DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
        broker_snapshot_path: Path | None = DEFAULT_BROKER_SNAPSHOT,
        execution_ledger_db_path: Path | None = None,
        execution_ledger_report_path: Path | None = None,
        predictive_scout_canonical_ledger_db_path: Path | None = None,
    ) -> None:
        self.client = client
        self.strategy_profile = strategy_profile
        self.output_path = output_path
        self.report_path = report_path
        self.live_enabled = live_enabled
        self.max_loss_jpy = max_loss_jpy
        self.max_loss_pct = max_loss_pct
        self.risk_equity_jpy = risk_equity_jpy
        self.portfolio_loss_cap_jpy = portfolio_loss_cap_jpy
        self.target_state_path = target_state_path
        self.target_report_path = target_report_path or (
            DEFAULT_DAILY_TARGET_REPORT
            if target_state_path == DEFAULT_DAILY_TARGET_STATE
            else target_state_path.with_suffix(".md")
        )
        self.execution_ledger_db_path = execution_ledger_db_path
        self.execution_ledger_report_path = execution_ledger_report_path
        self.predictive_scout_canonical_ledger_db_path = (
            predictive_scout_canonical_ledger_db_path
        )
        self.self_improvement_audit = self_improvement_audit
        # When the automation cycle stages a receipt that gpt-trader-decision
        # just verified, this points at the ACCEPTED verification artifact so
        # the LATEST_GPT_DECISION_STALE audit finding can be recognized as
        # already repaired. Manual stage-live-order paths leave it None.
        self.verified_decision_path = verified_decision_path
        self.guardian_action_receipt_path = guardian_action_receipt_path
        self.qr_trader_run_watchdog_path = (
            qr_trader_run_watchdog_path
            if qr_trader_run_watchdog_path != DEFAULT_QR_TRADER_RUN_WATCHDOG
            or output_path == DEFAULT_LIVE_ORDER_REQUEST
            else output_path.parent / DEFAULT_QR_TRADER_RUN_WATCHDOG.name
        )
        self.guardian_receipt_consumption_path = (
            guardian_receipt_consumption_path
            if guardian_receipt_consumption_path != DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION
            or output_path == DEFAULT_LIVE_ORDER_REQUEST
            else output_path.parent / DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION.name
        )
        self.guardian_receipt_operator_review_path = (
            guardian_receipt_operator_review_path
            if guardian_receipt_operator_review_path != DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW
            or output_path == DEFAULT_LIVE_ORDER_REQUEST
            else output_path.parent / DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW.name
        )
        self.broker_snapshot_path = (
            broker_snapshot_path
            if broker_snapshot_path != DEFAULT_BROKER_SNAPSHOT
            or output_path == DEFAULT_LIVE_ORDER_REQUEST
            else output_path.parent / DEFAULT_BROKER_SNAPSHOT.name
        )

    def run(
        self,
        *,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        lane_id: str | None = None,
        size_multiple: float = 1.0,
        send: bool = False,
        confirm_live: bool = False,
    ) -> LiveOrderStageSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        intents_payload = json.loads(intents_path.read_text())
        selected = _select_intent(intents_payload, lane_id)
        if selected is None:
            result = {
                "generated_at_utc": generated_at,
                "status": "NO_INTENT",
                "lane_id": lane_id,
                "order_request": None,
                "risk_issues": [],
                "strategy_issues": [],
                "send_requested": send,
                "sent": False,
                "response": None,
            }
            self._write_result(result)
            self._write_report(result)
            return LiveOrderStageSummary("NO_INTENT", lane_id, self.output_path, self.report_path, False, 0, 0)

        selected_lane_id = str(selected.get("lane_id") or "")
        intent = _intent_from_json(selected["intent"])
        intent = _intent_with_gateway_metadata(intent, selected_lane_id)
        requested_units = intent.units
        scaled_units, scale_issues, size_multiple = _scaled_units_for_intent(intent, size_multiple)
        if scaled_units is not None:
            intent = replace(intent, units=scaled_units)
        portfolio_position_cap = _portfolio_position_cap_from_state()
        default_max_loss_jpy = _per_trade_risk_from_state()
        max_loss_jpy = resolve_max_loss_jpy(
            max_loss_jpy=self.max_loss_jpy,
            max_loss_pct=self.max_loss_pct,
            equity_jpy=self.risk_equity_jpy,
            default_max_loss_jpy=default_max_loss_jpy,
            label="stage-live-order risk cap",
        )
        # AGENT_CONTRACT §3.5: portfolio cap (open + candidate exposure for the
        # day) is the whole-day risk budget, not the per-trade slice. Using
        # `max_loss_jpy` here would treat the per-shot cap as a portfolio
        # ceiling, blocking every additional shot once one position opens.
        # Resolve again immediately before broker validation. Automation may
        # have passed a cap computed earlier in the cycle; a close synced in
        # between must only tighten that value, never reopen capacity.
        portfolio_loss_cap = (
            self.portfolio_loss_cap_jpy
            if send and self.execution_ledger_db_path is not None
            else self._resolve_portfolio_loss_cap_jpy()
        )
        validate_live_enabled = self.live_enabled if send else True
        snapshot, risk, quote_refresh_attempts = self._snapshot_and_validate_intent(
            snapshot_pairs=_snapshot_pairs(intents_payload, intent),
            intent=intent,
            max_loss_jpy=max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            validate_live_enabled=validate_live_enabled,
            allow_basket_pending=False,
            portfolio_position_cap=portfolio_position_cap,
        )
        intent, risk, passive_reprice_issues = self._reprice_crossed_passive_limit(
            intent=intent,
            risk=risk,
            snapshot=snapshot,
            max_loss_jpy=max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            validate_live_enabled=validate_live_enabled,
            allow_basket_pending=False,
            portfolio_position_cap=portfolio_position_cap,
        )
        scale_issues.extend(passive_reprice_issues)
        intent, risk, size_multiple, loss_cap_scale_issues = self._clip_intent_to_loss_cap(
            intent=intent,
            risk=risk,
            snapshot=snapshot,
            max_loss_jpy=max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            validate_live_enabled=validate_live_enabled,
            allow_basket_pending=False,
            portfolio_position_cap=portfolio_position_cap,
            requested_units=requested_units,
            size_multiple=size_multiple,
        )
        scale_issues.extend(loss_cap_scale_issues)
        order_request, order_build_issues = _build_order_request(intent)
        (
            intent,
            risk,
            order_request,
            attached_stop_metrics,
            size_multiple,
            attached_stop_scale_issues,
            order_build_issues,
        ) = self._clip_intent_to_attached_stop_cap(
            intent=intent,
            risk=risk,
            snapshot=snapshot,
            max_loss_jpy=max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            cumulative_risk_jpy=0.0,
            validate_live_enabled=validate_live_enabled,
            allow_basket_pending=False,
            portfolio_position_cap=portfolio_position_cap,
            requested_units=requested_units,
            size_multiple=size_multiple,
            order_request=order_request,
            order_build_issues=order_build_issues,
        )
        scale_issues.extend(attached_stop_scale_issues)
        order_build_issues.extend(
            _predictive_scout_built_order_issues(intent, order_request)
        )
        strategy_issues = tuple(
            issue.__dict__ for issue in StrategyProfile.load(self.strategy_profile).validate(intent, for_live_send=True)
        )
        risk_issues = [issue.__dict__ for issue in risk.issues]
        intent_status_issues = _intent_status_issues(selected)
        projection_expiry_issues = _projection_expiry_status_issues(
            selected=selected,
            intents_path=intents_path,
            validation_time_utc=datetime.now(timezone.utc),
        )
        self_improvement_issues = _self_improvement_gateway_issues(
            self.self_improvement_audit,
            verified_decision_path=self.verified_decision_path,
            selected=selected,
        )
        gpt_verified_decision_issues = _gpt_verified_decision_live_send_issues(
            self.verified_decision_path,
            selected_lane_id=selected_lane_id,
            intents_payload=intents_payload,
            send=send,
            require_receipt=predictive_scout_intent_claimed(intent),
            intent=intent,
        )
        send_issues = _send_guard_issues(send=send, confirm_live=confirm_live, lane_id=lane_id)
        target_path_issues = _target_path_live_send_issues(intent, send=send)
        guardian_action_issues = guardian_action_gateway_issues(
            intent_metadata=intent.metadata,
            pair=intent.pair,
            thesis=intent.thesis,
            action_receipt_path=self.guardian_action_receipt_path,
        )
        guardian_receipt_consumption_issues = self._guardian_receipt_consumption_gateway_issues(risk_issues)
        predictive_scout_issues = predictive_scout_intent_issues(
            intent,
            snapshot=snapshot,
            data_root=intents_path.parent,
            validation_time_utc=datetime.now(timezone.utc),
            execution_ledger_db_path=self._predictive_scout_canonical_ledger_path(
                intents_path,
                live_send=send,
            ),
        )
        predictive_scout_issues.extend(
            _predictive_scout_ledger_path_issues(
                intent,
                intents_path=intents_path,
                configured_db_path=self.execution_ledger_db_path,
                canonical_db_path=self._predictive_scout_canonical_ledger_path(
                    intents_path,
                    live_send=send,
                ),
            )
        )
        sl_lint, sl_lint_issues = _sl_lint_result(
            intent=intent,
            snapshot=snapshot,
            order_request=order_request,
            metrics=risk.metrics,
            attached_stop_metrics=attached_stop_metrics,
        )
        all_blocked = (
            any(issue["severity"] == "BLOCK" for issue in risk_issues)
            or any(issue["severity"] == "BLOCK" for issue in strategy_issues)
            or any(issue["severity"] == "BLOCK" for issue in intent_status_issues)
            or any(issue["severity"] == "BLOCK" for issue in projection_expiry_issues)
            or any(issue["severity"] == "BLOCK" for issue in self_improvement_issues)
            or any(issue["severity"] == "BLOCK" for issue in gpt_verified_decision_issues)
            or any(issue["severity"] == "BLOCK" for issue in send_issues)
            or any(issue["severity"] == "BLOCK" for issue in target_path_issues)
            or any(issue["severity"] == "BLOCK" for issue in guardian_action_issues)
            or any(issue["severity"] == "BLOCK" for issue in guardian_receipt_consumption_issues)
            or any(issue["severity"] == "BLOCK" for issue in predictive_scout_issues)
            or any(issue["severity"] == "BLOCK" for issue in sl_lint_issues)
            or any(issue.severity == "BLOCK" for issue in scale_issues)
        )
        all_blocked = all_blocked or any(issue["severity"] == "BLOCK" for issue in order_build_issues)
        response = None
        sent = False
        entry_thesis_record = None
        entry_thesis_issues: list[dict[str, str]] = []
        ordinary_entry_claim: dict[str, Any] = {"status": "NOT_RUN"}
        ordinary_entry_claim_issues: list[dict[str, str]] = []
        pre_post_reconciliation: dict[str, Any] = {"status": "NOT_RUN"}
        status = "BLOCKED" if all_blocked else "STAGED"
        if send and order_request is not None and not all_blocked:
            predictive_scout_issues.extend(
                _predictive_scout_pre_post_issues(
                    intent,
                    validation_time_utc=datetime.now(timezone.utc),
                )
            )
            if any(issue["severity"] == "BLOCK" for issue in predictive_scout_issues):
                all_blocked = True
                status = "BLOCKED"
        if send and order_request is not None and not all_blocked:
            reconciliation = self._pre_post_reconcile(
                intent=intent,
                verified_intent=_intent_with_gateway_metadata(
                    _intent_from_json(selected["intent"]),
                    selected_lane_id,
                ),
                snapshot=snapshot,
                risk=risk,
                order_request=order_request,
                attached_stop_metrics=attached_stop_metrics,
                max_loss_jpy=max_loss_jpy,
                portfolio_loss_cap_jpy=portfolio_loss_cap,
                size_multiple=size_multiple,
                order_build_issues=order_build_issues,
                requested_units=requested_units,
                snapshot_pairs=_snapshot_pairs(intents_payload, intent),
                portfolio_position_cap=portfolio_position_cap,
                intents_path=intents_path,
                ignore_pending_order_ids=(),
            )
            intent = reconciliation.intent
            snapshot = reconciliation.snapshot
            risk = reconciliation.risk
            order_request = reconciliation.order_request
            attached_stop_metrics = reconciliation.attached_stop_metrics
            max_loss_jpy = reconciliation.max_loss_jpy
            portfolio_loss_cap = reconciliation.portfolio_loss_cap_jpy
            size_multiple = reconciliation.size_multiple
            order_build_issues = reconciliation.order_build_issues
            scale_issues.extend(reconciliation.issues)
            pre_post_reconciliation = reconciliation.evidence
            risk_issues = [issue.__dict__ for issue in risk.issues]
            sl_lint, sl_lint_issues = _sl_lint_result(
                intent=intent,
                snapshot=snapshot,
                order_request=order_request,
                metrics=risk.metrics,
                attached_stop_metrics=attached_stop_metrics,
            )
            final_lint_blocks = [
                str(issue.get("code") or "FINAL_SL_LINT_BLOCKED")
                for issue in sl_lint_issues
                if issue.get("severity") == "BLOCK"
            ]
            if final_lint_blocks:
                pre_post_reconciliation = {
                    **pre_post_reconciliation,
                    "status": "BLOCKED",
                    "final_check_failed": True,
                    "final_blocking_codes": [
                        *pre_post_reconciliation.get("final_blocking_codes", []),
                        *final_lint_blocks,
                    ],
                }
            if (
                any(issue.severity == "BLOCK" for issue in reconciliation.issues)
                or any(issue.severity == "BLOCK" for issue in risk.issues)
                or any(issue.get("severity") == "BLOCK" for issue in order_build_issues)
                or any(issue.get("severity") == "BLOCK" for issue in sl_lint_issues)
            ):
                all_blocked = True
                status = "BLOCKED"
        if send and order_request is not None and not all_blocked:
            reservation_issue = self._reserve_predictive_scout_post(
                intent=intent,
                order_request=order_request,
                lane_id=selected_lane_id,
                intents_path=intents_path,
                snapshot=snapshot,
            )
            if reservation_issue is not None:
                predictive_scout_issues.append(reservation_issue)
                all_blocked = True
                status = "BLOCKED"
        if send and order_request is not None and not all_blocked:
            claim_result = self._reserve_ordinary_entry_post(
                intent=intent,
                order_request=order_request,
                lane_id=selected_lane_id,
            )
            ordinary_entry_claim = claim_result.evidence
            if claim_result.issue is not None:
                ordinary_entry_claim_issues.append(claim_result.issue)
                all_blocked = True
                status = "BLOCKED"
        if send and order_request is not None and not all_blocked:
            try:
                response = self.client.post_order_json(order_request)
            except Exception as exc:
                ordinary_entry_claim, _ = self._finalize_ordinary_entry_post_claim(
                    ordinary_entry_claim,
                    status="POST_OUTCOME_UNKNOWN",
                    broker_outcome={
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                )
                raise
            ordinary_entry_claim, claim_finalize_issue = self._finalize_ordinary_entry_post_claim(
                ordinary_entry_claim,
                status="BROKER_RESPONSE_RECORDED",
                broker_outcome=response,
            )
            if claim_finalize_issue is not None:
                ordinary_entry_claim_issues.append(claim_finalize_issue)
            sent = True
            status = "SENT"
            if claim_finalize_issue is not None:
                status = "SENT_WITH_ENTRY_CLAIM_GAP"
            # Record entry thesis (2026-05-15, user directive: 「どの視点で
            # エントリーしたのか、時間がたって今のポジ状況はエントリー
            # したときと市況が変わってないか」). Best-effort: never raises
            # back into the live order path. Reads the latest forecast
            # for this pair that trader_brain wrote earlier in the cycle
            # and snapshots it alongside the SENT trade_id.
            try:
                from quant_rabbit.strategy.entry_thesis_ledger import (
                    record_entry_thesis_from_response_result,
                )
                record_result = record_entry_thesis_from_response_result(
                    response=response,
                    intent=intent,
                    data_root=_QR_ROOT / "data",
                )
                entry_thesis_record = record_result.to_dict()
                if record_result.status in {"FAILED", "DISABLED"}:
                    status = "SENT_WITH_ENTRY_THESIS_GAP"
                    entry_thesis_issues.append(
                        {
                            "severity": "BLOCK",
                            "code": "ENTRY_THESIS_RECORD_MISSING",
                            "message": record_result.issue or "entry thesis sidecar could not be verified after send",
                        }
                    )
            except Exception:
                status = "SENT_WITH_ENTRY_THESIS_GAP"
                entry_thesis_record = {"status": "FAILED", "issue": "entry thesis recorder raised unexpectedly"}
                entry_thesis_issues.append(
                    {
                        "severity": "BLOCK",
                        "code": "ENTRY_THESIS_RECORD_MISSING",
                        "message": "entry thesis recorder raised unexpectedly",
                    }
                )
        result = {
            "generated_at_utc": generated_at,
            "status": status,
            "lane_id": selected_lane_id,
            "order_request": order_request,
            "predictive_scout": predictive_scout_intent_claimed(intent),
            "predictive_scout_receipt": _predictive_scout_receipt_from_intent(intent),
            "context_evidence": _context_evidence_from_intent(intent),
            "sizing_evidence": _sizing_evidence_from_intent(
                intent,
                gateway_max_loss_jpy=max_loss_jpy,
                requested_units=requested_units,
                scaled_units=intent.units,
            ),
            "target_path_receipt": _target_path_receipt_from_intent(
                intent,
                risk_metrics=asdict(risk.metrics) if risk.metrics else None,
                order_request=order_request,
                requested_units=requested_units,
                final_units=intent.units,
                sent=sent,
            ),
            "risk_metrics": asdict(risk.metrics) if risk.metrics else None,
            "attached_stop_risk_metrics": attached_stop_metrics,
            "sl_lint": sl_lint,
            "risk_issues": [
                *risk_issues,
                *intent_status_issues,
                *projection_expiry_issues,
                *self_improvement_issues,
                *gpt_verified_decision_issues,
                *send_issues,
                *target_path_issues,
                *guardian_action_issues,
                *guardian_receipt_consumption_issues,
                *predictive_scout_issues,
                *sl_lint_issues,
                *order_build_issues,
                *[issue.__dict__ for issue in scale_issues],
                *entry_thesis_issues,
                *ordinary_entry_claim_issues,
            ],
            "strategy_issues": list(strategy_issues),
            "send_requested": send,
            "sent": sent,
            "response": response,
            "entry_thesis_record": entry_thesis_record,
            "ordinary_entry_claim": ordinary_entry_claim,
            "pre_post_reconciliation": pre_post_reconciliation,
            "snapshot": {
                "fetched_at_utc": snapshot.fetched_at_utc.isoformat(),
                "positions": len(snapshot.positions),
                "orders": len(snapshot.orders),
                "quotes": len(snapshot.quotes),
            },
            "quote_refresh_attempts": quote_refresh_attempts,
            "size_multiple": size_multiple,
            "requested_units": requested_units,
            "scaled_units": intent.units,
            "portfolio_position_cap": portfolio_position_cap,
        }
        self._write_result(result)
        durable, durability_issue = self._record_predictive_scout_receipt_durably(
            result,
            intents_path=intents_path,
        )
        if durable is not None:
            result["predictive_scout_receipt_durable"] = durable
        if durability_issue is not None:
            result["risk_issues"].append(durability_issue)
            status = "SENT_WITH_SCOUT_RECEIPT_GAP"
            result["status"] = status
        self._write_result(result)
        self._write_report(result)
        return LiveOrderStageSummary(
            status=status,
            lane_id=selected_lane_id,
            output_path=self.output_path,
            report_path=self.report_path,
            sent=sent,
            risk_issues=len(result["risk_issues"]),
            strategy_issues=len(strategy_issues),
            sent_count=1 if sent else 0,
            lane_ids=(selected_lane_id,) if selected_lane_id else (),
        )

    def run_batch(
        self,
        *,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        lane_ids: tuple[str, ...],
        size_multiples: dict[str, float] | None = None,
        ignore_pending_order_ids: tuple[str, ...] = (),
        send: bool = False,
        confirm_live: bool = False,
    ) -> LiveOrderStageSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        intents_payload = json.loads(intents_path.read_text())
        unique_lane_ids = tuple(dict.fromkeys(lane_id for lane_id in lane_ids if lane_id))
        selected_items = [(_select_intent(intents_payload, lane_id), lane_id) for lane_id in unique_lane_ids]
        selected_items = [(selected, lane_id) for selected, lane_id in selected_items if selected is not None]
        if not selected_items:
            result = {
                "generated_at_utc": generated_at,
                "status": "NO_INTENT",
                "lane_id": None,
                "lane_ids": list(unique_lane_ids),
                "orders": [],
                "risk_issues": [],
                "strategy_issues": [],
                "send_requested": send,
                "sent": False,
                "sent_count": 0,
            }
            self._write_result(result)
            self._write_report(result)
            return LiveOrderStageSummary("NO_INTENT", None, self.output_path, self.report_path, False, 0, 0)

        predictive_scout_items = [
            (selected, lane_id)
            for selected, lane_id in selected_items
            if _selected_intent_is_predictive_scout(selected)
        ]
        single_predictive_scout = len(predictive_scout_items) == 1 and len(selected_items) == 1
        if predictive_scout_items and not single_predictive_scout:
            issue = RiskIssue(
                "PREDICTIVE_SCOUT_SINGLE_ORDER_ONLY",
                "predictive SCOUT is a one-order forward experiment and cannot be staged or sent through basket/mixed execution",
            )
            order_results = [
                _blocked_batch_result(
                    generated_at=generated_at,
                    selected=selected,
                    lane_id=requested_lane_id,
                    send=send,
                    issue=issue,
                )
                for selected, requested_lane_id in selected_items
            ]
            result = {
                "generated_at_utc": generated_at,
                "status": "BLOCKED",
                "lane_id": order_results[0].get("lane_id"),
                "lane_ids": [item.get("lane_id") for item in order_results],
                "orders": order_results,
                "risk_issues": [issue for item in order_results for issue in item["risk_issues"]],
                "strategy_issues": [],
                "send_requested": send,
                "sent": False,
                "sent_count": 0,
                "staged_count": 0,
                "blocked_count": len(order_results),
            }
            self._write_result(result)
            self._write_report(result)
            return LiveOrderStageSummary(
                status="BLOCKED",
                lane_id=result["lane_id"],
                output_path=self.output_path,
                report_path=self.report_path,
                sent=False,
                risk_issues=len(result["risk_issues"]),
                strategy_issues=0,
                sent_count=0,
                lane_ids=tuple(lane_id for lane_id in result["lane_ids"] if lane_id),
            )

        ignored_pending_order_ids = _order_id_tuple(ignore_pending_order_ids)
        initial_snapshot = self.client.snapshot(_snapshot_pairs(intents_payload, _intent_from_json(selected_items[0][0]["intent"])))
        validation_snapshot = _snapshot_without_trader_pending_orders(initial_snapshot, ignored_pending_order_ids)
        initial_occupancy = _trader_entry_occupancy(validation_snapshot)
        portfolio_position_cap = _portfolio_position_cap_from_state()
        sent_count = 0
        accepted_count = 0
        validation_cumulative_risk_jpy = 0.0
        validation_cumulative_margin_jpy = 0.0
        accepted_risk_jpy = 0.0
        accepted_margin_jpy = 0.0
        seen_geometry = set(_pending_geometry_keys(validation_snapshot))
        seen_parent_lanes: set[str] = set(_pending_parent_lane_keys(validation_snapshot))
        accepted_pair_sides: dict[str, Side] = {}
        order_results: list[dict[str, Any]] = []
        batch_risk_issues = 0
        batch_strategy_issues = 0

        for selected, requested_lane_id in selected_items:
            parent_lane = _selected_parent_lane_key(selected, requested_lane_id)
            if parent_lane in seen_parent_lanes:
                blocked = _blocked_batch_result(
                    generated_at=generated_at,
                    selected=selected,
                    lane_id=requested_lane_id,
                    send=send,
                    issue=RiskIssue(
                        "BASKET_DUPLICATE_PARENT_LANE",
                        f"{requested_lane_id} is another execution variant of parent lane {parent_lane}; "
                        "send only one timing variant for a thesis in the same basket",
                    ),
                )
                order_results.append(blocked)
                batch_risk_issues += len(blocked["risk_issues"])
                continue
            candidate_intent = _intent_from_json(selected["intent"])
            accepted_side = accepted_pair_sides.get(candidate_intent.pair)
            if accepted_side is not None and accepted_side != candidate_intent.side and not _intent_declares_hedge(candidate_intent):
                blocked = _blocked_batch_result(
                    generated_at=generated_at,
                    selected=selected,
                    lane_id=requested_lane_id,
                    send=send,
                    issue=RiskIssue(
                        "BASKET_OPPOSING_PAIR_SIDE",
                        f"{requested_lane_id} would add {candidate_intent.pair} {candidate_intent.side.value} "
                        f"after {accepted_side.value} in the same basket; do not create same-pair "
                        "opposite exposure unless the intent explicitly declares HEDGE",
                    ),
                )
                order_results.append(blocked)
                batch_risk_issues += len(blocked["risk_issues"])
                continue
            active_occupancy = initial_occupancy + accepted_count
            if active_occupancy >= portfolio_position_cap:
                blocked = _blocked_batch_result(
                    generated_at=generated_at,
                    selected=selected,
                    lane_id=requested_lane_id,
                    send=send,
                    issue=RiskIssue(
                        "BASKET_PORTFOLIO_POSITION_LIMIT",
                        f"basket already has {active_occupancy} trader entries/pending orders; "
                        f"cap is {portfolio_position_cap}",
                    ),
                )
                order_results.append(blocked)
                batch_risk_issues += len(blocked["risk_issues"])
                continue

            item_result = self._run_one_selected(
                selected=selected,
                intents_payload=intents_payload,
                generated_at=generated_at,
                lane_id_arg=requested_lane_id,
                size_multiple=(size_multiples or {}).get(requested_lane_id, 1.0),
                send=send,
                confirm_live=confirm_live,
                allow_basket_pending=not single_predictive_scout,
                cumulative_risk_jpy=validation_cumulative_risk_jpy,
                cumulative_margin_jpy=validation_cumulative_margin_jpy,
                seen_geometry=seen_geometry,
                portfolio_position_cap=portfolio_position_cap,
                intents_path=intents_path,
                ignore_pending_order_ids=ignored_pending_order_ids,
            )
            order_results.append(item_result)
            batch_risk_issues += len(item_result["risk_issues"])
            batch_strategy_issues += len(item_result["strategy_issues"])
            if item_result.get("sent") or (not send and item_result.get("status") == "STAGED"):
                accepted_count += 1
                seen_parent_lanes.add(parent_lane)
                accepted_pair_sides.setdefault(candidate_intent.pair, candidate_intent.side)
                metrics = item_result.get("risk_metrics") if isinstance(item_result.get("risk_metrics"), dict) else {}
                receipt_risk_jpy = _receipt_risk_jpy(item_result)
                accepted_risk_jpy += receipt_risk_jpy
                accepted_margin_jpy += float(metrics.get("estimated_margin_jpy") or 0.0)
                # Dry-run staged orders are not visible in broker truth, so
                # carry them synthetically. Live sends are verified against the
                # next fresh broker snapshot to avoid double-counting margin.
                if not item_result.get("sent"):
                    validation_cumulative_risk_jpy += receipt_risk_jpy
                    validation_cumulative_margin_jpy += float(metrics.get("estimated_margin_jpy") or 0.0)
                geometry_key = item_result.get("geometry_key")
                if geometry_key:
                    seen_geometry.add(tuple(geometry_key))
            if item_result.get("sent"):
                sent_count += 1

        staged_count = sum(1 for item in order_results if item.get("status") == "STAGED")
        blocked_count = sum(1 for item in order_results if item.get("status") == "BLOCKED")
        entry_thesis_gap_count = sum(
            1 for item in order_results if item.get("status") == "SENT_WITH_ENTRY_THESIS_GAP"
        )
        if send:
            if entry_thesis_gap_count and blocked_count:
                status = "PARTIAL_SENT_WITH_ENTRY_THESIS_GAP"
            elif entry_thesis_gap_count:
                status = "SENT_WITH_ENTRY_THESIS_GAP"
            elif sent_count and blocked_count:
                status = "PARTIAL_SENT"
            elif sent_count:
                status = "SENT"
            else:
                status = "BLOCKED"
        else:
            status = "STAGED" if staged_count else "BLOCKED"
        result = {
            "generated_at_utc": generated_at,
            "status": status,
            "lane_id": order_results[0].get("lane_id") if order_results else None,
            "lane_ids": [item.get("lane_id") for item in order_results],
            "orders": order_results,
            "target_path_receipts": [
                item.get("target_path_receipt")
                for item in order_results
                if isinstance(item.get("target_path_receipt"), dict)
            ],
            "risk_issues": [issue for item in order_results for issue in item.get("risk_issues", [])],
            "strategy_issues": [issue for item in order_results for issue in item.get("strategy_issues", [])],
            "send_requested": send,
            "sent": sent_count > 0,
            "sent_count": sent_count,
            "staged_count": staged_count,
            "blocked_count": blocked_count,
            "cumulative_risk_jpy": accepted_risk_jpy,
            "cumulative_margin_jpy": accepted_margin_jpy,
            "initial_trader_entry_occupancy": initial_occupancy,
            "ignored_pending_order_ids": list(ignored_pending_order_ids),
            "portfolio_position_cap": portfolio_position_cap,
        }
        self._write_result(result)
        durable, durability_issue = self._record_predictive_scout_receipt_durably(
            result,
            intents_path=intents_path,
        )
        if durable is not None:
            result["predictive_scout_receipt_durable"] = durable
        if durability_issue is not None:
            result["risk_issues"].append(durability_issue)
            batch_risk_issues += 1
            status = "SENT_WITH_SCOUT_RECEIPT_GAP"
            result["status"] = status
        self._write_result(result)
        self._write_report(result)
        return LiveOrderStageSummary(
            status=status,
            lane_id=result["lane_id"],
            output_path=self.output_path,
            report_path=self.report_path,
            sent=sent_count > 0,
            risk_issues=batch_risk_issues,
            strategy_issues=batch_strategy_issues,
            sent_count=sent_count,
            lane_ids=tuple(lane_id for lane_id in result["lane_ids"] if lane_id),
        )

    def _run_one_selected(
        self,
        *,
        selected: dict[str, Any],
        intents_payload: dict[str, Any],
        generated_at: str,
        lane_id_arg: str | None,
        size_multiple: float,
        send: bool,
        confirm_live: bool,
        allow_basket_pending: bool = False,
        cumulative_risk_jpy: float = 0.0,
        cumulative_margin_jpy: float = 0.0,
        seen_geometry: set[tuple[object, ...]] | None = None,
        portfolio_position_cap: int | None = None,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        ignore_pending_order_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        selected_lane_id = str(selected.get("lane_id") or "")
        intent = _intent_from_json(selected["intent"])
        intent = _intent_with_gateway_metadata(intent, selected_lane_id)
        requested_units = intent.units
        scaled_units, scale_issues, size_multiple = _scaled_units_for_intent(intent, size_multiple)
        if scaled_units is not None:
            intent = replace(intent, units=scaled_units)
        max_loss_jpy = self._resolve_gateway_max_loss_jpy()
        portfolio_loss_cap = (
            self.portfolio_loss_cap_jpy
            if send and self.execution_ledger_db_path is not None
            else self._resolve_portfolio_loss_cap_jpy()
        )
        validate_live_enabled = self.live_enabled if send else True
        snapshot, risk, quote_refresh_attempts = self._snapshot_and_validate_intent(
            snapshot_pairs=_snapshot_pairs(intents_payload, intent),
            intent=intent,
            max_loss_jpy=max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            validate_live_enabled=validate_live_enabled,
            allow_basket_pending=allow_basket_pending,
            portfolio_position_cap=portfolio_position_cap or _portfolio_position_cap_from_state(),
            ignore_pending_order_ids=ignore_pending_order_ids,
        )
        intent, risk, passive_reprice_issues = self._reprice_crossed_passive_limit(
            intent=intent,
            risk=risk,
            snapshot=snapshot,
            max_loss_jpy=max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            validate_live_enabled=validate_live_enabled,
            allow_basket_pending=allow_basket_pending,
            portfolio_position_cap=portfolio_position_cap or _portfolio_position_cap_from_state(),
        )
        scale_issues.extend(passive_reprice_issues)
        intent, risk, size_multiple, loss_cap_scale_issues = self._clip_intent_to_loss_cap(
            intent=intent,
            risk=risk,
            snapshot=snapshot,
            max_loss_jpy=max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            validate_live_enabled=validate_live_enabled,
            allow_basket_pending=allow_basket_pending,
            portfolio_position_cap=portfolio_position_cap or _portfolio_position_cap_from_state(),
            requested_units=requested_units,
            size_multiple=size_multiple,
        )
        scale_issues.extend(loss_cap_scale_issues)
        if allow_basket_pending and risk.metrics is not None:
            scale_multiple, scale_issue = _basket_size_multiple(
                intent=intent,
                snapshot=snapshot,
                metrics=risk.metrics,
                portfolio_loss_cap=portfolio_loss_cap,
                cumulative_risk_jpy=cumulative_risk_jpy,
                cumulative_margin_jpy=cumulative_margin_jpy,
            )
            if scale_issue is not None:
                scale_issues.append(scale_issue)
            if 0.0 < scale_multiple < 1.0:
                scaled_units, extra_scale_issues = _scaled_units(
                    intent.units,
                    scale_multiple,
                    sub_min_lot_mode="block",
                )
                scale_issues.extend(extra_scale_issues)
                if scaled_units is not None:
                    intent = replace(intent, units=scaled_units)
                    risk = self._validate_intent(
                        intent=intent,
                        snapshot=snapshot,
                        max_loss_jpy=max_loss_jpy,
                        portfolio_loss_cap=portfolio_loss_cap,
                        validate_live_enabled=validate_live_enabled,
                        allow_basket_pending=allow_basket_pending,
                        portfolio_position_cap=portfolio_position_cap or _portfolio_position_cap_from_state(),
                    )
                    size_multiple *= scale_multiple
        order_request, order_build_issues = _build_order_request(intent)
        (
            intent,
            risk,
            order_request,
            attached_stop_metrics,
            size_multiple,
            attached_stop_scale_issues,
            order_build_issues,
        ) = self._clip_intent_to_attached_stop_cap(
            intent=intent,
            risk=risk,
            snapshot=snapshot,
            max_loss_jpy=max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            cumulative_risk_jpy=cumulative_risk_jpy,
            validate_live_enabled=validate_live_enabled,
            allow_basket_pending=allow_basket_pending,
            portfolio_position_cap=portfolio_position_cap or _portfolio_position_cap_from_state(),
            requested_units=requested_units,
            size_multiple=size_multiple,
            order_request=order_request,
            order_build_issues=order_build_issues,
        )
        scale_issues.extend(attached_stop_scale_issues)
        order_build_issues.extend(
            _predictive_scout_built_order_issues(intent, order_request)
        )
        strategy_issues = tuple(
            issue.__dict__ for issue in StrategyProfile.load(self.strategy_profile).validate(intent, for_live_send=True)
        )
        risk_issues = [issue.__dict__ for issue in risk.issues]
        if allow_basket_pending:
            risk_issues.extend(
                issue.__dict__
                for issue in _basket_issues(
                    intent=intent,
                    snapshot=snapshot,
                    metrics=risk.metrics,
                    portfolio_loss_cap=portfolio_loss_cap,
                    cumulative_risk_jpy=cumulative_risk_jpy,
                    cumulative_margin_jpy=cumulative_margin_jpy,
                    seen_geometry=seen_geometry or set(),
                )
            )
        intent_status_issues = _intent_status_issues(selected)
        projection_expiry_issues = _projection_expiry_status_issues(
            selected=selected,
            intents_path=intents_path,
            validation_time_utc=datetime.now(timezone.utc),
        )
        self_improvement_issues = _self_improvement_gateway_issues(
            self.self_improvement_audit,
            verified_decision_path=self.verified_decision_path,
            selected=selected,
        )
        gpt_verified_decision_issues = _gpt_verified_decision_live_send_issues(
            self.verified_decision_path,
            selected_lane_id=selected_lane_id,
            intents_payload=intents_payload,
            send=send,
            require_receipt=predictive_scout_intent_claimed(intent),
            intent=intent,
        )
        send_issues = _send_guard_issues(send=send, confirm_live=confirm_live, lane_id=lane_id_arg)
        target_path_issues = _target_path_live_send_issues(intent, send=send)
        guardian_action_issues = guardian_action_gateway_issues(
            intent_metadata=intent.metadata,
            pair=intent.pair,
            thesis=intent.thesis,
            action_receipt_path=self.guardian_action_receipt_path,
        )
        guardian_receipt_consumption_issues = self._guardian_receipt_consumption_gateway_issues(risk_issues)
        predictive_scout_issues = predictive_scout_intent_issues(
            intent,
            snapshot=snapshot,
            data_root=intents_path.parent,
            validation_time_utc=datetime.now(timezone.utc),
            execution_ledger_db_path=self._predictive_scout_canonical_ledger_path(
                intents_path,
                live_send=send,
            ),
        )
        predictive_scout_issues.extend(
            _predictive_scout_ledger_path_issues(
                intent,
                intents_path=intents_path,
                configured_db_path=self.execution_ledger_db_path,
                canonical_db_path=self._predictive_scout_canonical_ledger_path(
                    intents_path,
                    live_send=send,
                ),
            )
        )
        sl_lint, sl_lint_issues = _sl_lint_result(
            intent=intent,
            snapshot=snapshot,
            order_request=order_request,
            metrics=risk.metrics,
            attached_stop_metrics=attached_stop_metrics,
        )
        all_blocked = (
            any(issue["severity"] == "BLOCK" for issue in risk_issues)
            or any(issue["severity"] == "BLOCK" for issue in strategy_issues)
            or any(issue["severity"] == "BLOCK" for issue in intent_status_issues)
            or any(issue["severity"] == "BLOCK" for issue in projection_expiry_issues)
            or any(issue["severity"] == "BLOCK" for issue in self_improvement_issues)
            or any(issue["severity"] == "BLOCK" for issue in gpt_verified_decision_issues)
            or any(issue["severity"] == "BLOCK" for issue in send_issues)
            or any(issue["severity"] == "BLOCK" for issue in target_path_issues)
            or any(issue["severity"] == "BLOCK" for issue in guardian_action_issues)
            or any(issue["severity"] == "BLOCK" for issue in guardian_receipt_consumption_issues)
            or any(issue["severity"] == "BLOCK" for issue in predictive_scout_issues)
            or any(issue["severity"] == "BLOCK" for issue in sl_lint_issues)
            or any(issue.severity == "BLOCK" for issue in scale_issues)
        )
        all_blocked = all_blocked or any(issue["severity"] == "BLOCK" for issue in order_build_issues)
        response = None
        sent = False
        entry_thesis_record = None
        entry_thesis_issues: list[dict[str, str]] = []
        ordinary_entry_claim: dict[str, Any] = {"status": "NOT_RUN"}
        ordinary_entry_claim_issues: list[dict[str, str]] = []
        pre_post_reconciliation: dict[str, Any] = {"status": "NOT_RUN"}
        status = "BLOCKED" if all_blocked else "STAGED"
        if send and order_request is not None and not all_blocked:
            predictive_scout_issues.extend(
                _predictive_scout_pre_post_issues(
                    intent,
                    validation_time_utc=datetime.now(timezone.utc),
                )
            )
            if any(issue["severity"] == "BLOCK" for issue in predictive_scout_issues):
                all_blocked = True
                status = "BLOCKED"
        if send and order_request is not None and not all_blocked:
            final_position_cap = portfolio_position_cap or _portfolio_position_cap_from_state()
            reconciliation = self._pre_post_reconcile(
                intent=intent,
                verified_intent=_intent_with_gateway_metadata(
                    _intent_from_json(selected["intent"]),
                    selected_lane_id,
                ),
                snapshot=snapshot,
                risk=risk,
                order_request=order_request,
                attached_stop_metrics=attached_stop_metrics,
                max_loss_jpy=max_loss_jpy,
                portfolio_loss_cap_jpy=portfolio_loss_cap,
                size_multiple=size_multiple,
                order_build_issues=order_build_issues,
                requested_units=requested_units,
                snapshot_pairs=_snapshot_pairs(intents_payload, intent),
                portfolio_position_cap=final_position_cap,
                intents_path=intents_path,
                ignore_pending_order_ids=ignore_pending_order_ids,
            )
            intent = reconciliation.intent
            snapshot = reconciliation.snapshot
            risk = reconciliation.risk
            order_request = reconciliation.order_request
            attached_stop_metrics = reconciliation.attached_stop_metrics
            max_loss_jpy = reconciliation.max_loss_jpy
            portfolio_loss_cap = reconciliation.portfolio_loss_cap_jpy
            size_multiple = reconciliation.size_multiple
            order_build_issues = reconciliation.order_build_issues
            scale_issues.extend(reconciliation.issues)
            pre_post_reconciliation = reconciliation.evidence
            risk_issues = [issue.__dict__ for issue in risk.issues]
            sl_lint, sl_lint_issues = _sl_lint_result(
                intent=intent,
                snapshot=snapshot,
                order_request=order_request,
                metrics=risk.metrics,
                attached_stop_metrics=attached_stop_metrics,
            )
            final_lint_blocks = [
                str(issue.get("code") or "FINAL_SL_LINT_BLOCKED")
                for issue in sl_lint_issues
                if issue.get("severity") == "BLOCK"
            ]
            if final_lint_blocks:
                pre_post_reconciliation = {
                    **pre_post_reconciliation,
                    "status": "BLOCKED",
                    "final_check_failed": True,
                    "final_blocking_codes": [
                        *pre_post_reconciliation.get("final_blocking_codes", []),
                        *final_lint_blocks,
                    ],
                }
            if (
                any(issue.severity == "BLOCK" for issue in reconciliation.issues)
                or any(issue.severity == "BLOCK" for issue in risk.issues)
                or any(issue.get("severity") == "BLOCK" for issue in order_build_issues)
                or any(issue.get("severity") == "BLOCK" for issue in sl_lint_issues)
            ):
                all_blocked = True
                status = "BLOCKED"
        if send and order_request is not None and not all_blocked:
            reservation_issue = self._reserve_predictive_scout_post(
                intent=intent,
                order_request=order_request,
                lane_id=selected_lane_id,
                intents_path=intents_path,
                snapshot=snapshot,
            )
            if reservation_issue is not None:
                predictive_scout_issues.append(reservation_issue)
                all_blocked = True
                status = "BLOCKED"
        if send and order_request is not None and not all_blocked:
            claim_result = self._reserve_ordinary_entry_post(
                intent=intent,
                order_request=order_request,
                lane_id=selected_lane_id,
            )
            ordinary_entry_claim = claim_result.evidence
            if claim_result.issue is not None:
                ordinary_entry_claim_issues.append(claim_result.issue)
                all_blocked = True
                status = "BLOCKED"
        if send and order_request is not None and not all_blocked:
            try:
                response = self.client.post_order_json(order_request)
            except Exception as exc:
                ordinary_entry_claim, _ = self._finalize_ordinary_entry_post_claim(
                    ordinary_entry_claim,
                    status="POST_OUTCOME_UNKNOWN",
                    broker_outcome={
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                )
                raise
            ordinary_entry_claim, claim_finalize_issue = self._finalize_ordinary_entry_post_claim(
                ordinary_entry_claim,
                status="BROKER_RESPONSE_RECORDED",
                broker_outcome=response,
            )
            if claim_finalize_issue is not None:
                ordinary_entry_claim_issues.append(claim_finalize_issue)
            sent = True
            status = "SENT"
            if claim_finalize_issue is not None:
                status = "SENT_WITH_ENTRY_CLAIM_GAP"
            # Entry thesis recording — see comment in run() for rationale.
            try:
                from quant_rabbit.strategy.entry_thesis_ledger import (
                    record_entry_thesis_from_response_result,
                )
                record_result = record_entry_thesis_from_response_result(
                    response=response,
                    intent=intent,
                    data_root=_QR_ROOT / "data",
                )
                entry_thesis_record = record_result.to_dict()
                if record_result.status in {"FAILED", "DISABLED"}:
                    status = "SENT_WITH_ENTRY_THESIS_GAP"
                    entry_thesis_issues.append(
                        {
                            "severity": "BLOCK",
                            "code": "ENTRY_THESIS_RECORD_MISSING",
                            "message": record_result.issue or "entry thesis sidecar could not be verified after send",
                        }
                    )
            except Exception:
                status = "SENT_WITH_ENTRY_THESIS_GAP"
                entry_thesis_record = {"status": "FAILED", "issue": "entry thesis recorder raised unexpectedly"}
                entry_thesis_issues.append(
                    {
                        "severity": "BLOCK",
                        "code": "ENTRY_THESIS_RECORD_MISSING",
                        "message": "entry thesis recorder raised unexpectedly",
                    }
                )
        return {
            "generated_at_utc": generated_at,
            "status": status,
            "lane_id": selected_lane_id,
            "order_request": order_request,
            "predictive_scout": predictive_scout_intent_claimed(intent),
            "predictive_scout_receipt": _predictive_scout_receipt_from_intent(intent),
            "context_evidence": _context_evidence_from_intent(intent),
            "sizing_evidence": _sizing_evidence_from_intent(
                intent,
                gateway_max_loss_jpy=max_loss_jpy,
                requested_units=requested_units,
                scaled_units=intent.units,
            ),
            "target_path_receipt": _target_path_receipt_from_intent(
                intent,
                risk_metrics=asdict(risk.metrics) if risk.metrics else None,
                order_request=order_request,
                requested_units=requested_units,
                final_units=intent.units,
                sent=sent,
            ),
            "risk_metrics": asdict(risk.metrics) if risk.metrics else None,
            "attached_stop_risk_metrics": attached_stop_metrics,
            "sl_lint": sl_lint,
            "risk_issues": [
                *risk_issues,
                *intent_status_issues,
                *projection_expiry_issues,
                *self_improvement_issues,
                *gpt_verified_decision_issues,
                *send_issues,
                *target_path_issues,
                *guardian_action_issues,
                *guardian_receipt_consumption_issues,
                *predictive_scout_issues,
                *sl_lint_issues,
                *order_build_issues,
                *[issue.__dict__ for issue in scale_issues],
                *entry_thesis_issues,
                *ordinary_entry_claim_issues,
            ],
            "strategy_issues": list(strategy_issues),
            "send_requested": send,
            "sent": sent,
            "response": response,
            "entry_thesis_record": entry_thesis_record,
            "ordinary_entry_claim": ordinary_entry_claim,
            "pre_post_reconciliation": pre_post_reconciliation,
            "snapshot": {
                "fetched_at_utc": snapshot.fetched_at_utc.isoformat(),
                "positions": len(snapshot.positions),
                "orders": len(snapshot.orders),
                "quotes": len(snapshot.quotes),
            },
            "quote_refresh_attempts": quote_refresh_attempts,
            "size_multiple": size_multiple,
            "requested_units": requested_units,
            "scaled_units": intent.units,
            "portfolio_position_cap": portfolio_position_cap or _portfolio_position_cap_from_state(),
            "geometry_key": list(_intent_geometry_key(intent)),
        }

    def _guardian_receipt_consumption_gateway_issues(self, risk_issues: list[dict[str, Any]]) -> list[dict[str, str]]:
        if any(
            issue.get("code") in {GUARDIAN_RECEIPT_BLOCK_NEW_ENTRY_CODE, GUARDIAN_WATCHDOG_BLOCK_NEW_ENTRY_CODE}
            for issue in risk_issues
            if isinstance(issue, dict)
        ):
            return []
        return guardian_receipt_new_entry_blockers_from_paths(
            watchdog_path=self.qr_trader_run_watchdog_path,
            consumption_path=self.guardian_receipt_consumption_path,
            operator_review_path=self.guardian_receipt_operator_review_path,
            broker_snapshot_path=self.broker_snapshot_path,
        )

    def _resolve_gateway_max_loss_jpy(self) -> float:
        default_max_loss_jpy = _per_trade_risk_from_state()
        return resolve_max_loss_jpy(
            max_loss_jpy=self.max_loss_jpy,
            max_loss_pct=self.max_loss_pct,
            equity_jpy=self.risk_equity_jpy,
            default_max_loss_jpy=default_max_loss_jpy,
            label="stage-live-order risk cap",
        )

    def _resolve_portfolio_loss_cap_jpy(self) -> float | None:
        persisted_capacity = _daily_risk_budget_from_state(self.target_state_path)
        if self.portfolio_loss_cap_jpy is None:
            return persisted_capacity
        if persisted_capacity is None:
            return self.portfolio_loss_cap_jpy
        return min(self.portfolio_loss_cap_jpy, persisted_capacity)

    def _pre_post_reconcile(
        self,
        *,
        intent: OrderIntent,
        verified_intent: OrderIntent,
        snapshot: BrokerSnapshot,
        risk: Any,
        order_request: dict[str, Any] | None,
        attached_stop_metrics: dict[str, Any] | None,
        max_loss_jpy: float,
        portfolio_loss_cap_jpy: float | None,
        size_multiple: float,
        order_build_issues: list[dict[str, str]],
        requested_units: int,
        snapshot_pairs: tuple[str, ...],
        portfolio_position_cap: int,
        intents_path: Path,
        ignore_pending_order_ids: tuple[str, ...] = (),
    ) -> _PrePostReconciliationResult:
        """Reconcile loss capacity against broker truth immediately before POST.

        The whole sequence intentionally lives inside the gateway so direct,
        batch, AutoTradeCycle, and guardian entry paths share the same final
        defense. The refreshed target publishes capacity *before* open risk;
        this method then subtracts fresh open, pending, and candidate risk once
        from the exact snapshot whose lastTransactionID matched the ledger.
        """

        evidence: dict[str, Any] = {
            "status": "SKIPPED",
            "execution_ledger_db_path": (
                str(self.execution_ledger_db_path)
                if self.execution_ledger_db_path is not None
                else None
            ),
            "target_state_path": str(self.target_state_path),
        }
        def blocked(code: str, message: str) -> _PrePostReconciliationResult:
            issue = RiskIssue(code, message)
            evidence.update({"status": "BLOCKED", "issue_code": code, "message": message})
            return _PrePostReconciliationResult(
                intent=intent,
                snapshot=snapshot,
                risk=risk,
                order_request=order_request,
                attached_stop_metrics=attached_stop_metrics,
                max_loss_jpy=max_loss_jpy,
                portfolio_loss_cap_jpy=portfolio_loss_cap_jpy,
                size_multiple=size_multiple,
                order_build_issues=order_build_issues,
                issues=(issue,),
                evidence=evidence,
            )

        if self.execution_ledger_db_path is None:
            return blocked(
                "PRE_POST_LEDGER_PATH_MISSING",
                "live send requires an execution_ledger_db_path for final broker/target reconciliation",
            )

        if not hasattr(self.client, "account_summary") or not hasattr(
            self.client,
            "transactions_since_id",
        ):
            return blocked(
                "PRE_POST_LEDGER_SYNC_UNAVAILABLE",
                "live send requires account_summary and transactions_since_id so the execution ledger can sync before POST",
            )

        report_path = self.execution_ledger_report_path or self.execution_ledger_db_path.with_suffix(".md")
        ledger = ExecutionLedger(
            db_path=self.execution_ledger_db_path,
            report_path=report_path,
        )
        try:
            sync_summary = ledger.sync_oanda_transactions(self.client)
        except Exception as exc:
            return blocked(
                "PRE_POST_LEDGER_SYNC_FAILED",
                f"execution ledger sync failed immediately before POST: {type(exc).__name__}: {exc}",
            )
        evidence.update(
            {
                "ledger_sync_status": sync_summary.status,
                "ledger_last_transaction_id": sync_summary.last_transaction_id,
            }
        )
        if sync_summary.status != "SYNCED" or not sync_summary.last_transaction_id:
            return blocked(
                "PRE_POST_LEDGER_NOT_SYNCED",
                f"execution ledger status is {sync_summary.status} with last transaction "
                f"{sync_summary.last_transaction_id!r}; a baseline/legacy ledger cannot authorize a live POST",
            )

        try:
            fresh_snapshot = self.client.snapshot(snapshot_pairs)
        except Exception as exc:
            return blocked(
                "PRE_POST_BROKER_SNAPSHOT_FAILED",
                f"fresh broker snapshot failed immediately before POST: {type(exc).__name__}: {exc}",
            )
        fresh_snapshot = _snapshot_without_trader_pending_orders(
            fresh_snapshot,
            ignore_pending_order_ids,
        )
        broker_transaction_id = (
            str(fresh_snapshot.account.last_transaction_id or "").strip()
            if fresh_snapshot.account is not None
            else ""
        )
        ledger_transaction_id = str(sync_summary.last_transaction_id or "").strip()
        evidence.update(
            {
                "broker_last_transaction_id": broker_transaction_id or None,
                "snapshot_fetched_at_utc": fresh_snapshot.fetched_at_utc.isoformat(),
            }
        )
        if not broker_transaction_id or broker_transaction_id != ledger_transaction_id:
            return blocked(
                "PRE_POST_TRANSACTION_ID_MISMATCH",
                f"ledger lastTransactionID {ledger_transaction_id or 'missing'} does not match fresh broker snapshot "
                f"{broker_transaction_id or 'missing'}; retry after both views converge",
            )

        macro_event_confidence_sizing = _metadata_truthy(
            (intent.metadata or {}).get("macro_event_confidence_sizing")
        )
        macro_event_fresh_nav_value: float | None = None
        macro_event_fresh_nav_recheck: dict[str, Any] = {
            "status": "NOT_APPLICABLE"
        }
        if macro_event_confidence_sizing:
            account = fresh_snapshot.account
            fresh_nav = account.nav_jpy if account is not None else None
            try:
                parsed_fresh_nav = float(fresh_nav)
            except (TypeError, ValueError):
                parsed_fresh_nav = math.nan
            if not math.isfinite(parsed_fresh_nav) or parsed_fresh_nav <= 0.0:
                macro_event_fresh_nav_recheck = {
                    "status": "BLOCKED",
                    "fresh_nav_jpy": fresh_nav,
                    "max_risk_pct_nav": MACRO_EVENT_MAX_RISK_PCT_NAV,
                    "issue_code": "PRE_POST_MACRO_EVENT_FRESH_NAV_MISSING",
                }
                evidence["macro_event_fresh_nav_recheck"] = (
                    macro_event_fresh_nav_recheck
                )
                return blocked(
                    "PRE_POST_MACRO_EVENT_FRESH_NAV_MISSING",
                    "macro-event confidence sizing requires finite positive NAV from the "
                    "same fresh broker snapshot used immediately before POST",
                )
            macro_event_fresh_nav_value = parsed_fresh_nav

        try:
            target_summary = DailyTargetLedger(
                state_path=self.target_state_path,
                report_path=self.target_report_path,
                execution_ledger_path=self.execution_ledger_db_path,
            ).run(
                snapshot=fresh_snapshot,
                now_utc=fresh_snapshot.fetched_at_utc,
            )
            target_payload = json.loads(self.target_state_path.read_text())
        except Exception as exc:
            return blocked(
                "PRE_POST_TARGET_REFRESH_FAILED",
                f"daily target could not refresh from the synced ledger and matching snapshot: "
                f"{type(exc).__name__}: {exc}",
            )
        if not isinstance(target_payload, dict):
            return blocked(
                "PRE_POST_TARGET_STATE_LEGACY",
                "refreshed daily target state is not a JSON object",
            )

        required_fields = (
            "realized_loss_spent_jpy",
            "daily_loss_capacity_before_open_jpy",
            "base_per_trade_risk_budget_jpy",
            "per_trade_risk_budget_jpy",
        )
        parsed: dict[str, float] = {}
        for field in required_fields:
            raw = target_payload.get(field)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return blocked(
                    "PRE_POST_TARGET_STATE_LEGACY",
                    f"refreshed daily target state lacks numeric {field}; legacy state cannot authorize POST",
                )
            if not math.isfinite(value) or value < 0.0:
                return blocked(
                    "PRE_POST_TARGET_STATE_LEGACY",
                    f"refreshed daily target state has invalid {field}={raw!r}",
                )
            parsed[field] = value

        final_capacity = parsed["daily_loss_capacity_before_open_jpy"]
        final_per_trade = parsed["per_trade_risk_budget_jpy"]
        evidence.update(
            {
                "status": "REFRESHED",
                "target_status": target_summary.status,
                **parsed,
            }
        )
        # Preserve the more specific exhausted-capacity diagnosis when both
        # the numeric ledger and its state-machine status say risk is gone.
        if final_capacity <= 0.0 or final_per_trade <= 0.0:
            return blocked(
                "PRE_POST_DAILY_LOSS_CAPACITY_EXHAUSTED",
                f"fresh daily loss capacity is {final_capacity:.4f} JPY and final per-trade cap is "
                f"{final_per_trade:.4f} JPY; block new entry until the next campaign day",
            )
        final_target_status = str(target_summary.status or "").strip().upper()
        if final_target_status in PRE_POST_FRESH_ENTRY_FORBIDDEN_TARGET_STATUSES:
            return blocked(
                "PRE_POST_TARGET_STATUS_BLOCKS_FRESH_ENTRY",
                f"fresh daily target status {final_target_status} forbids new entry risk; "
                "discard the stale intent packet and follow the target protection/repair state",
            )
        if final_target_status not in PRE_POST_FRESH_ENTRY_ALLOWED_TARGET_STATUSES:
            return blocked(
                "PRE_POST_TARGET_STATUS_UNRECOGNIZED",
                f"fresh daily target status {final_target_status or 'missing'} is not an "
                "explicit fresh-entry-allowed state",
            )

        macro_event_fresh_absolute_cap: float | None = None
        if macro_event_confidence_sizing:
            assert macro_event_fresh_nav_value is not None
            nav_cap = macro_event_fresh_nav_value * (
                MACRO_EVENT_MAX_RISK_PCT_NAV / 100.0
            )
            macro_event_fresh_absolute_cap = min(
                final_per_trade,
                final_capacity,
                nav_cap,
            )
            macro_event_fresh_nav_recheck = {
                "status": "APPLIED",
                "fresh_nav_jpy": macro_event_fresh_nav_value,
                "max_risk_pct_nav": MACRO_EVENT_MAX_RISK_PCT_NAV,
                "fresh_nav_cap_jpy": nav_cap,
                "fresh_absolute_cap_jpy": macro_event_fresh_absolute_cap,
            }

        final_max_loss = min(float(max_loss_jpy), final_per_trade)
        if macro_event_fresh_absolute_cap is not None:
            final_max_loss = min(
                final_max_loss,
                macro_event_fresh_absolute_cap,
            )
        final_portfolio_cap = (
            final_capacity
            if portfolio_loss_cap_jpy is None
            else min(float(portfolio_loss_cap_jpy), final_capacity)
        )

        # Revalidate geometry/margin on the matching fresh snapshot without a
        # portfolio cap in RiskEngine. Portfolio capacity is handled once below
        # by the basket accounting helper, which includes open + pending + this
        # candidate and starts cumulative risk at zero.
        final_risk = self._validate_intent(
            intent=intent,
            snapshot=fresh_snapshot,
            max_loss_jpy=final_max_loss,
            portfolio_loss_cap=None,
            validate_live_enabled=True,
            allow_basket_pending=True,
            portfolio_position_cap=portfolio_position_cap,
        )
        final_intent, final_risk, final_size_multiple, sizing_issues = self._clip_intent_to_loss_cap(
            intent=intent,
            risk=final_risk,
            snapshot=fresh_snapshot,
            max_loss_jpy=final_max_loss,
            portfolio_loss_cap=None,
            validate_live_enabled=True,
            allow_basket_pending=True,
            portfolio_position_cap=portfolio_position_cap,
            requested_units=requested_units,
            size_multiple=size_multiple,
        )
        final_issues: list[RiskIssue] = list(sizing_issues)
        if final_risk.metrics is not None:
            capacity_scale, capacity_issue = _basket_size_multiple(
                intent=final_intent,
                snapshot=fresh_snapshot,
                metrics=final_risk.metrics,
                portfolio_loss_cap=final_portfolio_cap,
                cumulative_risk_jpy=0.0,
                cumulative_margin_jpy=0.0,
            )
            if capacity_issue is not None:
                final_issues.append(capacity_issue)
            if 0.0 < capacity_scale < 1.0:
                scaled_units, scaled_issues = _scaled_units(
                    final_intent.units,
                    capacity_scale,
                    sub_min_lot_mode="block",
                )
                final_issues.extend(scaled_issues)
                if scaled_units is not None:
                    final_intent = replace(final_intent, units=scaled_units)
                    final_size_multiple = abs(scaled_units) / abs(requested_units) if requested_units else 0.0
                    final_risk = self._validate_intent(
                        intent=final_intent,
                        snapshot=fresh_snapshot,
                        max_loss_jpy=final_max_loss,
                        portfolio_loss_cap=None,
                        validate_live_enabled=True,
                        allow_basket_pending=True,
                        portfolio_position_cap=portfolio_position_cap,
                    )

        final_order_request, final_order_build_issues = _build_order_request(final_intent)
        (
            final_intent,
            final_risk,
            final_order_request,
            final_attached_stop,
            final_size_multiple,
            attached_stop_issues,
            final_order_build_issues,
        ) = self._clip_intent_to_attached_stop_cap(
            intent=final_intent,
            risk=final_risk,
            snapshot=fresh_snapshot,
            max_loss_jpy=final_max_loss,
            portfolio_loss_cap=final_portfolio_cap,
            cumulative_risk_jpy=0.0,
            validate_live_enabled=True,
            allow_basket_pending=True,
            portfolio_position_cap=portfolio_position_cap,
            requested_units=requested_units,
            size_multiple=final_size_multiple,
            order_request=final_order_request,
            order_build_issues=final_order_build_issues,
        )
        final_issues.extend(attached_stop_issues)
        final_issues.extend(
            _basket_issues(
                intent=final_intent,
                snapshot=fresh_snapshot,
                metrics=final_risk.metrics,
                portfolio_loss_cap=final_portfolio_cap,
                cumulative_risk_jpy=0.0,
                cumulative_margin_jpy=0.0,
                seen_geometry=set(),
            )
        )
        fresh_pending_issues, fresh_pending_evidence = _fresh_pending_entry_issues(
            intent=final_intent,
            snapshot=fresh_snapshot,
            portfolio_position_cap=portfolio_position_cap,
        )
        final_issues.extend(fresh_pending_issues)
        target_path_final_recheck: dict[str, Any] = {"status": "NOT_APPLICABLE"}
        if _target_path_contract_present(dict(final_intent.metadata or {})):
            raw_remaining_minimum = target_payload.get("remaining_minimum_jpy")
            raw_minimum_progress = target_payload.get("minimum_progress_pct")
            try:
                fresh_remaining_minimum = float(raw_remaining_minimum)
                fresh_minimum_progress = float(raw_minimum_progress)
            except (TypeError, ValueError):
                fresh_remaining_minimum = math.nan
                fresh_minimum_progress = math.nan

            if (
                not math.isfinite(fresh_remaining_minimum)
                or fresh_remaining_minimum < 0.0
                or not math.isfinite(fresh_minimum_progress)
            ):
                issue = RiskIssue(
                    "PRE_POST_TARGET_PATH_STATE_INVALID",
                    "target-path live send requires fresh numeric remaining_minimum_jpy and "
                    "minimum_progress_pct from the reconciled daily target state",
                )
                final_issues.append(issue)
                target_path_final_recheck = {
                    "status": "BLOCKED",
                    "remaining_minimum_jpy": raw_remaining_minimum,
                    "minimum_progress_pct": raw_minimum_progress,
                    "issue_codes": [issue.code],
                }
            else:
                # The dry-run receipt can be older than a just-synced winning
                # close.  Overlay every 5%-progress alias with broker/ledger
                # truth before re-running the live target-path contract; stale
                # receipt metadata must not reopen B-grade risk after the floor.
                final_metadata = dict(final_intent.metadata or {})
                for key in (
                    "daily_progress_pct",
                    "day_progress_pct",
                    "total_day_progress_pct",
                ):
                    final_metadata.pop(key, None)
                final_metadata.update(
                    {
                        "remaining_to_5pct_yen": fresh_remaining_minimum,
                        "remaining_to_5pct": fresh_remaining_minimum,
                        "remaining_minimum_jpy": fresh_remaining_minimum,
                        "minimum_progress_pct": fresh_minimum_progress,
                    }
                )
                final_intent = replace(final_intent, metadata=final_metadata)
                target_path_recheck_issues = _target_path_live_send_issues(
                    final_intent,
                    send=True,
                )
                for target_path_issue in target_path_recheck_issues:
                    final_issues.append(
                        RiskIssue(
                            str(target_path_issue.get("code") or "TARGET_PATH_FINAL_RECHECK_FAILED"),
                            str(
                                target_path_issue.get("message")
                                or "target-path final live-send recheck failed"
                            ),
                            str(target_path_issue.get("severity") or "BLOCK"),
                        )
                    )
                target_path_final_recheck = {
                    "status": (
                        "BLOCKED"
                        if any(
                            issue.get("severity") == "BLOCK"
                            for issue in target_path_recheck_issues
                        )
                        else "PASSED"
                    ),
                    "remaining_minimum_jpy": fresh_remaining_minimum,
                    "minimum_progress_pct": fresh_minimum_progress,
                    "final_units": final_intent.units,
                    "issue_codes": [
                        str(issue.get("code") or "TARGET_PATH_FINAL_RECHECK_FAILED")
                        for issue in target_path_recheck_issues
                    ],
                }
        if predictive_scout_intent_claimed(verified_intent):
            verified_shape = (
                verified_intent.units,
                *_intent_geometry_key(verified_intent),
            )
            final_shape = (
                final_intent.units,
                *_intent_geometry_key(final_intent),
            )
            if final_shape != verified_shape:
                final_issues.append(
                    RiskIssue(
                        "PREDICTIVE_SCOUT_PRE_POST_MUTATION_REQUIRES_REVERIFY",
                        "fresh risk reconciliation changed predictive SCOUT units or order geometry; "
                        "do not mutate an already verified experiment—regenerate and reverify the exact tier/digest",
                    )
                )
        final_order_build_issues.extend(
            _predictive_scout_built_order_issues(final_intent, final_order_request)
        )
        for scout_issue in predictive_scout_intent_issues(
            final_intent,
            snapshot=fresh_snapshot,
            data_root=intents_path.parent,
            validation_time_utc=datetime.now(timezone.utc),
            execution_ledger_db_path=self._predictive_scout_canonical_ledger_path(
                intents_path,
                live_send=True,
            ),
        ):
            final_issues.append(
                RiskIssue(
                    str(scout_issue.get("code") or "PREDICTIVE_SCOUT_FINAL_RECHECK_FAILED"),
                    str(scout_issue.get("message") or "predictive SCOUT final recheck failed"),
                    str(scout_issue.get("severity") or "BLOCK"),
                )
            )
        blocking_codes = [
            issue.code
            for issue in (*final_risk.issues, *final_issues)
            if issue.severity == "BLOCK"
        ]
        blocking_codes.extend(
            str(issue.get("code") or "FINAL_ORDER_BUILD_BLOCKED")
            for issue in final_order_build_issues
            if issue.get("severity") == "BLOCK"
        )
        evidence.update(
            {
                "status": "BLOCKED" if blocking_codes else "PASSED",
                "final_check_failed": bool(blocking_codes),
                "final_blocking_codes": blocking_codes,
                "final_max_loss_jpy": final_max_loss,
                "final_portfolio_loss_cap_jpy": final_portfolio_cap,
                "final_units": final_intent.units,
                "macro_event_fresh_nav_recheck": macro_event_fresh_nav_recheck,
                "fresh_pending_reconciliation": fresh_pending_evidence,
                "target_path_final_recheck": target_path_final_recheck,
            }
        )
        return _PrePostReconciliationResult(
            intent=final_intent,
            snapshot=fresh_snapshot,
            risk=final_risk,
            order_request=final_order_request,
            attached_stop_metrics=final_attached_stop,
            max_loss_jpy=final_max_loss,
            portfolio_loss_cap_jpy=final_portfolio_cap,
            size_multiple=final_size_multiple,
            order_build_issues=final_order_build_issues,
            issues=tuple(final_issues),
            evidence=evidence,
        )

    def _validate_intent(
        self,
        *,
        intent: OrderIntent,
        snapshot: BrokerSnapshot,
        max_loss_jpy: float,
        portfolio_loss_cap: float | None,
        validate_live_enabled: bool,
        allow_basket_pending: bool,
        portfolio_position_cap: int,
    ):
        return RiskEngine(
            policy=RiskPolicy(
                allow_protected_trader_position_adds=True,
                block_new_entries_with_pending_entry_orders=not allow_basket_pending,
                max_loss_jpy=max_loss_jpy,
                max_portfolio_positions=portfolio_position_cap,
                max_portfolio_loss_jpy=None if allow_basket_pending else portfolio_loss_cap,
            ),
            live_enabled=validate_live_enabled,
            guardian_receipt_watchdog_path=self.qr_trader_run_watchdog_path,
            guardian_receipt_consumption_path=self.guardian_receipt_consumption_path,
            guardian_receipt_operator_review_path=self.guardian_receipt_operator_review_path,
            guardian_receipt_broker_snapshot_path=self.broker_snapshot_path,
        ).validate(intent, snapshot, for_live_send=True)

    def _reprice_crossed_passive_limit(
        self,
        *,
        intent: OrderIntent,
        risk,
        snapshot: BrokerSnapshot,
        max_loss_jpy: float,
        portfolio_loss_cap: float | None,
        validate_live_enabled: bool,
        allow_basket_pending: bool,
        portfolio_position_cap: int,
    ):
        repriced_intent, issue = _repriced_crossed_passive_limit_intent(
            intent=intent,
            snapshot=snapshot,
            risk=risk,
        )
        if issue is None:
            return intent, risk, []
        repriced_risk = self._validate_intent(
            intent=repriced_intent,
            snapshot=snapshot,
            max_loss_jpy=max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            validate_live_enabled=validate_live_enabled,
            allow_basket_pending=allow_basket_pending,
            portfolio_position_cap=portfolio_position_cap,
        )
        return repriced_intent, repriced_risk, [issue]

    def _clip_intent_to_loss_cap(
        self,
        *,
        intent: OrderIntent,
        risk,
        snapshot: BrokerSnapshot,
        max_loss_jpy: float,
        portfolio_loss_cap: float | None,
        validate_live_enabled: bool,
        allow_basket_pending: bool,
        portfolio_position_cap: int,
        requested_units: int,
        size_multiple: float,
    ):
        metrics = risk.metrics
        effective_max_loss_jpy = _attached_stop_loss_cap_jpy(intent, max_loss_jpy) or max_loss_jpy
        if metrics is None or metrics.risk_jpy <= effective_max_loss_jpy:
            issues: list[RiskIssue] = []
        else:
            scale = _capacity_scale(abs(intent.units), metrics.risk_jpy, effective_max_loss_jpy)
            scaled_units, issues = _scaled_units(intent.units, scale, sub_min_lot_mode="block")
            if scaled_units is None:
                return intent, risk, size_multiple, issues
            original_units = intent.units
            intent = replace(intent, units=scaled_units)
            risk = self._validate_intent(
                intent=intent,
                snapshot=snapshot,
                max_loss_jpy=max_loss_jpy,
                portfolio_loss_cap=portfolio_loss_cap,
                validate_live_enabled=validate_live_enabled,
                allow_basket_pending=allow_basket_pending,
                portfolio_position_cap=portfolio_position_cap,
            )
            if requested_units:
                size_multiple = abs(scaled_units) / abs(requested_units)
            issues = [
                RiskIssue(
                    "SIZE_MULTIPLE_CLIPPED_TO_LOSS_CAP",
                    f"scaled units {original_units}u would risk {metrics.risk_jpy:.0f} JPY "
                    f"above cap {effective_max_loss_jpy:.0f} JPY; clipped to {scaled_units}u.",
                    "WARN",
                ),
                *issues,
            ]
        return self._clip_intent_to_pair_margin_cap(
            intent=intent,
            risk=risk,
            snapshot=snapshot,
            max_loss_jpy=max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            validate_live_enabled=validate_live_enabled,
            allow_basket_pending=allow_basket_pending,
            portfolio_position_cap=portfolio_position_cap,
            requested_units=requested_units,
            size_multiple=size_multiple,
            prior_issues=issues,
        )

    def _clip_intent_to_attached_stop_cap(
        self,
        *,
        intent: OrderIntent,
        risk,
        snapshot: BrokerSnapshot,
        max_loss_jpy: float,
        portfolio_loss_cap: float | None,
        cumulative_risk_jpy: float,
        validate_live_enabled: bool,
        allow_basket_pending: bool,
        portfolio_position_cap: int,
        requested_units: int,
        size_multiple: float,
        order_request: dict[str, Any] | None,
        order_build_issues: list[dict[str, str]],
    ):
        metrics = risk.metrics
        attached_stop = _attached_stop_risk_metrics(intent, order_request, metrics)
        cap = _attached_stop_effective_loss_cap_jpy(
            intent=intent,
            policy_max_loss_jpy=max_loss_jpy,
            snapshot=snapshot,
            portfolio_loss_cap=portfolio_loss_cap,
            cumulative_risk_jpy=cumulative_risk_jpy,
        )
        if (
            metrics is None
            or order_request is None
            or attached_stop is None
            or cap is None
            or attached_stop["risk_jpy"] <= cap
        ):
            return intent, risk, order_request, attached_stop, size_multiple, [], order_build_issues

        if attached_stop.get("basis") == "DISASTER_SL":
            portfolio_remaining = _portfolio_loss_remaining_jpy(
                snapshot=snapshot,
                portfolio_loss_cap=portfolio_loss_cap,
                cumulative_risk_jpy=cumulative_risk_jpy,
            )
            if portfolio_remaining is not None and attached_stop["risk_jpy"] > portfolio_remaining:
                return (
                    intent,
                    risk,
                    order_request,
                    attached_stop,
                    size_multiple,
                    [
                        RiskIssue(
                            "DISASTER_STOP_PORTFOLIO_CAP_EXCEEDED",
                            f"disaster stop risk {attached_stop['risk_jpy']:.0f} JPY exceeds "
                            f"portfolio remaining capacity {portfolio_remaining:.0f} JPY; keep units "
                            "sized by the expected invalidation and skip this lane until the basket "
                            "can absorb the full catastrophe-stop exposure.",
                        )
                    ],
                    order_build_issues,
                )
            return intent, risk, order_request, attached_stop, size_multiple, [], order_build_issues

        original_units = intent.units
        original_attached_risk_jpy = float(attached_stop["risk_jpy"])
        scale = _capacity_scale(abs(intent.units), original_attached_risk_jpy, cap)
        scaled_units, scaled_issues = _scaled_units(intent.units, scale, sub_min_lot_mode="block")
        if scaled_units is None:
            return (
                intent,
                risk,
                order_request,
                attached_stop,
                size_multiple,
                [
                    RiskIssue(
                        "ATTACHED_STOP_LOSS_CAP_BELOW_MIN_LOT",
                        f"attached broker SL risk {original_attached_risk_jpy:.0f} JPY exceeds effective "
                        f"cap {cap:.0f} JPY, and fitting the cap would require a sub-"
                        f"{MIN_PRODUCTION_LOT_UNITS}u order; skip this lane until current geometry "
                        "or the broker-side catastrophe stop fits the production minimum lot.",
                    )
                ],
                order_build_issues,
            )

        intent = replace(intent, units=scaled_units)
        risk = self._validate_intent(
            intent=intent,
            snapshot=snapshot,
            max_loss_jpy=max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            validate_live_enabled=validate_live_enabled,
            allow_basket_pending=allow_basket_pending,
            portfolio_position_cap=portfolio_position_cap,
        )
        if requested_units:
            size_multiple = abs(scaled_units) / abs(requested_units)
        order_request, order_build_issues = _build_order_request(intent)
        attached_stop = _attached_stop_risk_metrics(intent, order_request, risk.metrics)
        issues = [
            RiskIssue(
                "SIZE_MULTIPLE_CLIPPED_TO_ATTACHED_STOP_CAP",
                f"attached broker SL risk {original_attached_risk_jpy:.0f} JPY for {original_units}u "
                f"exceeded effective cap {cap:.0f} JPY; clipped to {scaled_units}u before staging.",
                "WARN",
            ),
            *scaled_issues,
        ]
        if attached_stop is not None and attached_stop["risk_jpy"] > cap:
            issues.append(
                RiskIssue(
                    "ATTACHED_STOP_LOSS_CAP_EXCEEDED",
                    f"attached broker SL risk {attached_stop['risk_jpy']:.0f} JPY still exceeds "
                    f"effective cap {cap:.0f} JPY after size clipping; block live send.",
                )
            )
        return intent, risk, order_request, attached_stop, size_multiple, issues, order_build_issues

    def _clip_intent_to_pair_margin_cap(
        self,
        *,
        intent: OrderIntent,
        risk,
        snapshot: BrokerSnapshot,
        max_loss_jpy: float,
        portfolio_loss_cap: float | None,
        validate_live_enabled: bool,
        allow_basket_pending: bool,
        portfolio_position_cap: int,
        requested_units: int,
        size_multiple: float,
        prior_issues: list[RiskIssue],
    ):
        if not any(issue.code == "PAIR_MARGIN_CONCENTRATION_LIMIT" for issue in risk.issues):
            return intent, risk, size_multiple, prior_issues
        metrics = risk.metrics
        account = snapshot.account
        cap_pct = RiskPolicy().max_same_pair_margin_utilization_pct
        if (
            metrics is None
            or account is None
            or account.nav_jpy <= 0
            or cap_pct is None
            or cap_pct <= 0
            or metrics.estimated_margin_jpy is None
            or metrics.estimated_margin_jpy <= 0
        ):
            return intent, risk, size_multiple, prior_issues
        cap_jpy = account.nav_jpy * (cap_pct / 100.0)
        if metrics.estimated_margin_jpy <= cap_jpy:
            return intent, risk, size_multiple, prior_issues
        scale = _capacity_scale(abs(intent.units), metrics.estimated_margin_jpy, cap_jpy)
        scaled_units, issues = _scaled_units(intent.units, scale, sub_min_lot_mode="block")
        if scaled_units is None:
            return intent, risk, size_multiple, [*prior_issues, *issues]
        original_units = intent.units
        intent = replace(intent, units=scaled_units)
        risk = self._validate_intent(
            intent=intent,
            snapshot=snapshot,
            max_loss_jpy=max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            validate_live_enabled=validate_live_enabled,
            allow_basket_pending=allow_basket_pending,
            portfolio_position_cap=portfolio_position_cap,
        )
        if requested_units:
            size_multiple = abs(scaled_units) / abs(requested_units)
        return intent, risk, size_multiple, [
            *prior_issues,
            RiskIssue(
                "SIZE_MULTIPLE_CLIPPED_TO_PAIR_MARGIN_CAP",
                f"scaled units {original_units}u would use {metrics.estimated_margin_jpy:.0f} JPY "
                f"same-pair margin above cap {cap_jpy:.0f} JPY; clipped to {scaled_units}u.",
                "WARN",
            ),
            *issues,
        ]

    def _snapshot_and_validate_intent(
        self,
        *,
        snapshot_pairs: tuple[str, ...],
        intent: OrderIntent,
        max_loss_jpy: float,
        portfolio_loss_cap: float | None,
        validate_live_enabled: bool,
        allow_basket_pending: bool,
        portfolio_position_cap: int,
        ignore_pending_order_ids: tuple[str, ...] = (),
    ):
        snapshot = self.client.snapshot(snapshot_pairs)
        snapshot = _snapshot_without_trader_pending_orders(snapshot, ignore_pending_order_ids)
        risk = self._validate_intent(
            intent=intent,
            snapshot=snapshot,
            max_loss_jpy=max_loss_jpy,
            portfolio_loss_cap=portfolio_loss_cap,
            validate_live_enabled=validate_live_enabled,
            allow_basket_pending=allow_basket_pending,
            portfolio_position_cap=portfolio_position_cap,
        )
        refresh_attempts = 0
        if not _risk_has_blocking_stale_quote(risk):
            return snapshot, risk, refresh_attempts

        retry_sleep_seconds = _gateway_stale_quote_retry_sleep_seconds()
        for _ in range(_gateway_stale_quote_retry_attempts()):
            if retry_sleep_seconds > 0:
                time.sleep(retry_sleep_seconds)
            snapshot = self.client.snapshot(snapshot_pairs)
            snapshot = _snapshot_without_trader_pending_orders(snapshot, ignore_pending_order_ids)
            risk = self._validate_intent(
                intent=intent,
                snapshot=snapshot,
                max_loss_jpy=max_loss_jpy,
                portfolio_loss_cap=portfolio_loss_cap,
                validate_live_enabled=validate_live_enabled,
                allow_basket_pending=allow_basket_pending,
                portfolio_position_cap=portfolio_position_cap,
            )
            refresh_attempts += 1
            if not _risk_has_blocking_stale_quote(risk):
                break
        return snapshot, risk, refresh_attempts

    def _write_result(self, result: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _predictive_scout_canonical_ledger_path(
        self,
        intents_path: Path,
        *,
        live_send: bool,
    ) -> Path:
        if self.predictive_scout_canonical_ledger_db_path is not None:
            return self.predictive_scout_canonical_ledger_db_path
        # Live collection has one process-wide SSOT.  Dry-run/staging may use
        # an isolated intent-adjacent ledger so tests and offline review do not
        # mutate production state.
        if live_send:
            return DEFAULT_EXECUTION_LEDGER_DB
        return intents_path.parent / DEFAULT_EXECUTION_LEDGER_DB.name

    def _reserve_ordinary_entry_post(
        self,
        *,
        intent: OrderIntent,
        order_request: dict[str, Any],
        lane_id: str,
    ) -> _OrdinaryEntryPostClaimResult:
        """Durably consume one ordinary GPT entry signal before network I/O.

        Predictive SCOUT keeps its stricter vehicle reservation table.  For an
        ordinary lane, two independent uniqueness constraints are required:
        the same accepted decision cannot be reused for the same parent lane,
        and a new decision cannot relabel an already-consumed parent/forecast
        cycle as a fresh signal.  Consequently ADD needs both a new receipt and
        a new forecast cycle.  Claims are never deleted or released: a process
        crash after this commit is an ambiguous broker outcome and retries fail
        closed until a genuinely new signal is verified.
        """

        if predictive_scout_intent_claimed(intent):
            return _OrdinaryEntryPostClaimResult(
                evidence={"status": "NOT_APPLICABLE_PREDICTIVE_SCOUT"}
            )
        if self.verified_decision_path is None:
            return _OrdinaryEntryPostClaimResult(
                evidence={"status": "NOT_APPLICABLE_NO_GPT_RECEIPT"}
            )
        # Real live sends already require the configured canonical ledger in
        # _pre_post_reconcile.  The adjacent fallback preserves durable claim
        # semantics for isolated/test gateways whose reconciliation is mocked;
        # it cannot weaken a real send because missing sync capability/path is
        # blocked before this method is reached.
        claim_db_path = self.execution_ledger_db_path or (
            DEFAULT_EXECUTION_LEDGER_DB
            if self.output_path == DEFAULT_LIVE_ORDER_REQUEST
            else self.output_path.parent / DEFAULT_EXECUTION_LEDGER_DB.name
        )

        try:
            payload = json.loads(self.verified_decision_path.read_text())
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return _OrdinaryEntryPostClaimResult(
                evidence={"status": "BLOCKED", "issue_code": "ORDINARY_ENTRY_CLAIM_RECEIPT_UNREADABLE"},
                issue=RiskIssue(
                    "ORDINARY_ENTRY_CLAIM_RECEIPT_UNREADABLE",
                    f"verified GPT receipt changed or became unreadable at the pre-POST claim boundary: {exc}",
                ).__dict__,
            )
        decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
        receipt_status = str(payload.get("status") or "").strip().upper()
        action = str(decision.get("action") or "").strip().upper()
        selected_lanes = _gpt_receipt_selected_lane_ids(decision)
        blocking_verification_codes = [
            str(item.get("code") or "").strip()
            for item in payload.get("verification_issues", []) or []
            if isinstance(item, dict)
            and str(item.get("severity") or "BLOCK").upper() == "BLOCK"
            and str(item.get("code") or "").strip()
        ]
        if (
            receipt_status != "ACCEPTED"
            or action not in {"TRADE", "ADD"}
            or not lane_id
            or lane_id not in selected_lanes
            or not isinstance(decision.get("market_read_first"), dict)
            or bool(blocking_verification_codes)
        ):
            return _OrdinaryEntryPostClaimResult(
                evidence={
                    "status": "BLOCKED",
                    "issue_code": "ORDINARY_ENTRY_CLAIM_RECEIPT_MISMATCH",
                    "receipt_status": receipt_status or None,
                    "action": action or None,
                },
                issue=RiskIssue(
                    "ORDINARY_ENTRY_CLAIM_RECEIPT_MISMATCH",
                    "pre-POST one-shot claim requires the same ACCEPTED TRADE/ADD receipt to still name the selected lane",
                ).__dict__,
            )

        metadata = dict(intent.metadata or {})
        parent_lane_id = str(metadata.get("parent_lane_id") or "").strip()
        derived_parent_lane_id = _parent_lane_id_from_lane_id(lane_id)
        if parent_lane_id and parent_lane_id != derived_parent_lane_id:
            return _OrdinaryEntryPostClaimResult(
                evidence={
                    "status": "BLOCKED",
                    "issue_code": "ORDINARY_ENTRY_CLAIM_PARENT_LANE_MISMATCH",
                    "metadata_parent_lane_id": parent_lane_id,
                    "derived_parent_lane_id": derived_parent_lane_id,
                },
                issue=RiskIssue(
                    "ORDINARY_ENTRY_CLAIM_PARENT_LANE_MISMATCH",
                    "intent parent_lane_id does not match the selected execution variant; "
                    "refusing a relabeled one-shot claim",
                ).__dict__,
            )
        if not parent_lane_id:
            parent_lane_id = derived_parent_lane_id
        forecast_cycle_id = str(metadata.get("forecast_cycle_id") or "").strip()
        receipt_generated_at = str(
            decision.get("generated_at_utc")
            or payload.get("generated_at_utc")
            or ""
        ).strip()
        if not parent_lane_id or not forecast_cycle_id or not receipt_generated_at:
            missing = [
                name
                for name, value in (
                    ("parent_lane_id", parent_lane_id),
                    ("forecast_cycle_id", forecast_cycle_id),
                    ("receipt_generated_at_utc", receipt_generated_at),
                )
                if not value
            ]
            return _OrdinaryEntryPostClaimResult(
                evidence={
                    "status": "BLOCKED",
                    "issue_code": "ORDINARY_ENTRY_CLAIM_IDENTITY_MISSING",
                    "missing_fields": missing,
                },
                issue=RiskIssue(
                    "ORDINARY_ENTRY_CLAIM_IDENTITY_MISSING",
                    "ordinary GPT-authorized live entry lacks durable one-shot identity: "
                    + ", ".join(missing),
                ).__dict__,
            )

        receipt_identity = {
            "status": receipt_status,
            "decision": decision,
            # A legacy verified artifact may keep the model timestamp only at
            # top level; include the resolved value so rerunning a verifier
            # around the same model decision cannot manufacture a new claim.
            "receipt_generated_at_utc": receipt_generated_at,
        }
        receipt_sha256 = hashlib.sha256(
            json.dumps(
                receipt_identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        order_request_sha256 = hashlib.sha256(
            json.dumps(
                order_request,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        claim_id = hashlib.sha256(
            f"{receipt_sha256}\0{parent_lane_id}\0{forecast_cycle_id}".encode("utf-8")
        ).hexdigest()
        claimed_at = datetime.now(timezone.utc).isoformat()
        base_evidence = {
            "claim_id": claim_id,
            "receipt_sha256": receipt_sha256,
            "receipt_generated_at_utc": receipt_generated_at,
            "action": action,
            "lane_id": lane_id,
            "parent_lane_id": parent_lane_id,
            "forecast_cycle_id": forecast_cycle_id,
            "order_request_sha256": order_request_sha256,
            "ledger_db_path": str(claim_db_path),
        }

        conn: sqlite3.Connection | None = None
        try:
            claim_db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(claim_db_path), timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ordinary_live_entry_signal_claims (
                    claim_id TEXT PRIMARY KEY,
                    receipt_sha256 TEXT NOT NULL,
                    receipt_generated_at_utc TEXT NOT NULL,
                    action TEXT NOT NULL,
                    lane_id TEXT NOT NULL,
                    parent_lane_id TEXT NOT NULL,
                    forecast_cycle_id TEXT NOT NULL,
                    order_request_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    claimed_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    broker_outcome_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ordinary_entry_claim_receipt_parent
                ON ordinary_live_entry_signal_claims(receipt_sha256, parent_lane_id)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ordinary_entry_claim_parent_cycle
                ON ordinary_live_entry_signal_claims(parent_lane_id, forecast_cycle_id)
                """
            )
            prior_receipt = conn.execute(
                """
                SELECT claim_id, forecast_cycle_id, status
                FROM ordinary_live_entry_signal_claims
                WHERE receipt_sha256 = ? AND parent_lane_id = ?
                """,
                (receipt_sha256, parent_lane_id),
            ).fetchone()
            if prior_receipt is not None:
                conn.rollback()
                same_cycle = str(prior_receipt["forecast_cycle_id"] or "") == forecast_cycle_id
                code = (
                    "ORDINARY_ENTRY_RECEIPT_ALREADY_CLAIMED"
                    if same_cycle
                    else "ORDINARY_ENTRY_RECEIPT_PARENT_ALREADY_CLAIMED"
                )
                return _OrdinaryEntryPostClaimResult(
                    evidence={
                        **base_evidence,
                        "status": "DUPLICATE_BLOCKED",
                        "issue_code": code,
                        "existing_claim_id": prior_receipt["claim_id"],
                        "existing_claim_status": prior_receipt["status"],
                    },
                    issue=RiskIssue(
                        code,
                        "the same accepted GPT receipt already consumed this parent lane; "
                        "a changed intent/cycle cannot replay that authorization",
                    ).__dict__,
                )
            prior_cycle = conn.execute(
                """
                SELECT claim_id, receipt_sha256, status
                FROM ordinary_live_entry_signal_claims
                WHERE parent_lane_id = ? AND forecast_cycle_id = ?
                """,
                (parent_lane_id, forecast_cycle_id),
            ).fetchone()
            if prior_cycle is not None:
                conn.rollback()
                return _OrdinaryEntryPostClaimResult(
                    evidence={
                        **base_evidence,
                        "status": "DUPLICATE_BLOCKED",
                        "issue_code": "ORDINARY_ENTRY_FORECAST_CYCLE_ALREADY_CLAIMED",
                        "existing_claim_id": prior_cycle["claim_id"],
                        "existing_claim_status": prior_cycle["status"],
                    },
                    issue=RiskIssue(
                        "ORDINARY_ENTRY_FORECAST_CYCLE_ALREADY_CLAIMED",
                        "this parent lane/forecast cycle already owns a durable broker-POST claim; "
                        "a new receipt alone is not independent ADD/TRADE evidence",
                    ).__dict__,
                )
            conn.execute(
                """
                INSERT INTO ordinary_live_entry_signal_claims(
                    claim_id, receipt_sha256, receipt_generated_at_utc, action,
                    lane_id, parent_lane_id, forecast_cycle_id,
                    order_request_sha256, status, claimed_at_utc, updated_at_utc,
                    broker_outcome_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'RESERVED_PRE_POST', ?, ?, NULL)
                """,
                (
                    claim_id,
                    receipt_sha256,
                    receipt_generated_at,
                    action,
                    lane_id,
                    parent_lane_id,
                    forecast_cycle_id,
                    order_request_sha256,
                    claimed_at,
                    claimed_at,
                ),
            )
            conn.commit()
        except (OSError, sqlite3.Error, ValueError) as exc:
            if conn is not None:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
            return _OrdinaryEntryPostClaimResult(
                evidence={
                    **base_evidence,
                    "status": "BLOCKED",
                    "issue_code": "ORDINARY_ENTRY_CLAIM_RESERVATION_FAILED",
                },
                issue=RiskIssue(
                    "ORDINARY_ENTRY_CLAIM_RESERVATION_FAILED",
                    "refusing broker POST because the ordinary GPT entry signal could not be durably claimed first: "
                    f"{type(exc).__name__}: {exc}",
                ).__dict__,
            )
        finally:
            if conn is not None:
                conn.close()
        return _OrdinaryEntryPostClaimResult(
            evidence={
                **base_evidence,
                "status": "RESERVED_PRE_POST",
                "claimed_at_utc": claimed_at,
            }
        )

    def _finalize_ordinary_entry_post_claim(
        self,
        claim_evidence: dict[str, Any],
        *,
        status: str,
        broker_outcome: Any,
    ) -> tuple[dict[str, Any], dict[str, str] | None]:
        claim_id = str(claim_evidence.get("claim_id") or "").strip()
        raw_claim_db_path = str(claim_evidence.get("ledger_db_path") or "").strip()
        if not claim_id or not raw_claim_db_path:
            return claim_evidence, None
        claim_db_path = Path(raw_claim_db_path)
        now = datetime.now(timezone.utc).isoformat()
        try:
            outcome_json = json.dumps(
                broker_outcome,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            with sqlite3.connect(str(claim_db_path), timeout=5.0) as conn:
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("BEGIN IMMEDIATE")
                updated = conn.execute(
                    """
                    UPDATE ordinary_live_entry_signal_claims
                    SET status = ?, updated_at_utc = ?, broker_outcome_json = ?
                    WHERE claim_id = ? AND status = 'RESERVED_PRE_POST'
                    """,
                    (status, now, outcome_json, claim_id),
                ).rowcount
                if updated != 1:
                    raise ValueError(
                        f"ordinary live-entry claim {claim_id} was missing or no longer RESERVED_PRE_POST"
                    )
            return {
                **claim_evidence,
                "status": status,
                "updated_at_utc": now,
            }, None
        except (OSError, sqlite3.Error, ValueError) as exc:
            # The original committed reservation remains consumed.  Never
            # delete/release it merely because post-response annotation failed.
            return {
                **claim_evidence,
                "status": "FINALIZE_FAILED_RESERVATION_RETAINED",
                "finalize_error": f"{type(exc).__name__}: {exc}",
            }, RiskIssue(
                "ORDINARY_ENTRY_CLAIM_FINALIZE_FAILED",
                "broker POST returned but its one-shot claim outcome could not be annotated; "
                "the reservation remains consumed and must not be retried",
            ).__dict__

    def _record_predictive_scout_receipt_durably(
        self,
        result: dict[str, Any],
        *,
        intents_path: Path,
    ) -> tuple[bool | None, dict[str, str] | None]:
        if result.get("sent") is not True or not _result_contains_predictive_scout(result):
            return None, None
        db_path = self._predictive_scout_canonical_ledger_path(
            intents_path,
            live_send=True,
        )
        report_path = self.execution_ledger_report_path or (
            self.report_path.parent / DEFAULT_EXECUTION_LEDGER_REPORT.name
        )
        try:
            ExecutionLedger(db_path=db_path, report_path=report_path).record_gateway_receipt(
                kind="live_order",
                receipt_path=self.output_path,
            )
        except (OSError, sqlite3.Error, json.JSONDecodeError, ValueError) as exc:
            return False, RiskIssue(
                "PREDICTIVE_SCOUT_RECEIPT_DURABILITY_GAP",
                "broker accepted predictive SCOUT but its vehicle receipt was not durably indexed; "
                f"do not send another SCOUT until ledger reconciliation succeeds: {exc}",
            ).__dict__
        return True, None

    def _reserve_predictive_scout_post(
        self,
        *,
        intent: OrderIntent,
        order_request: dict[str, Any],
        lane_id: str,
        intents_path: Path,
        snapshot: BrokerSnapshot,
    ) -> dict[str, str] | None:
        if not predictive_scout_intent_claimed(intent):
            return None
        db_path = self._predictive_scout_canonical_ledger_path(
            intents_path,
            live_send=True,
        )
        report_path = self.execution_ledger_report_path or (
            self.report_path.parent / DEFAULT_EXECUTION_LEDGER_REPORT.name
        )
        reserved_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "generated_at_utc": reserved_at,
            "status": "PREDICTIVE_SCOUT_POST_RESERVED",
            "lane_id": lane_id,
            "order_request": order_request,
            "predictive_scout": True,
            "predictive_scout_post_reserved": True,
            "predictive_scout_receipt": _predictive_scout_receipt_from_intent(intent),
            "send_requested": True,
            "sent": False,
            "response": None,
        }
        scout_receipt = payload.get("predictive_scout_receipt")
        signal_id = (
            str(scout_receipt.get("predictive_scout_signal_id") or "")
            if isinstance(scout_receipt, dict)
            else ""
        )
        experiment_id = (
            str(scout_receipt.get("predictive_scout_experiment_id") or "")
            if isinstance(scout_receipt, dict)
            else ""
        )
        vehicle_id = (
            str(scout_receipt.get("predictive_scout_vehicle_id") or "")
            if isinstance(scout_receipt, dict)
            else ""
        )
        expires_at_utc = (
            str(scout_receipt.get("predictive_scout_expires_at_utc") or "")
            if isinstance(scout_receipt, dict)
            else ""
        )
        policy_path = intents_path.parent.parent / "config" / "predictive_scout_policy.json"
        policy = predictive_scout_policy(policy_path)
        metadata = intent.metadata or {}
        candidate_risk_jpy = _positive_float(
            metadata.get("predictive_scout_fresh_actual_initial_risk_jpy")
        )
        broker_active_risk_jpy = _metadata_float(
            metadata,
            "predictive_scout_active_initial_risk_jpy",
        )
        concurrent_risk_cap_jpy = _positive_float(
            metadata.get("predictive_scout_concurrent_risk_cap_jpy")
        )
        try:
            max_daily = int(policy.get("max_sent_per_campaign_day"))
            max_concurrent = int(policy.get("max_concurrent"))
        except (TypeError, ValueError):
            return RiskIssue(
                "PREDICTIVE_SCOUT_POST_RESERVATION_FAILED",
                "refusing broker POST because the predictive SCOUT policy does not provide "
                "valid daily and concurrent reservation caps",
            ).__dict__
        if not (
            0 < max_daily <= PREDICTIVE_SCOUT_MAX_SENT_PER_CAMPAIGN_DAY
            and 0 < max_concurrent <= PREDICTIVE_SCOUT_MAX_CONCURRENT
        ):
            return RiskIssue(
                "PREDICTIVE_SCOUT_POST_RESERVATION_FAILED",
                "refusing broker POST because predictive SCOUT reservation caps exceed the "
                "canonical shared campaign limits",
            ).__dict__
        if (
            candidate_risk_jpy is None
            or broker_active_risk_jpy is None
            or broker_active_risk_jpy < 0.0
            or concurrent_risk_cap_jpy is None
        ):
            return RiskIssue(
                "PREDICTIVE_SCOUT_POST_RESERVATION_FAILED",
                "refusing broker POST because fresh candidate, broker-active, and concurrent-cap JPY risk are not durably available",
            ).__dict__
        try:
            reservation = ExecutionLedger(
                db_path=db_path,
                report_path=report_path,
            ).reserve_predictive_scout_gateway_payload(
                kind="live_order",
                receipt_path=Path(f"{self.output_path}.predictive-scout-reservation"),
                payload=payload,
                signal_id=signal_id,
                experiment_id=experiment_id,
                vehicle_id=vehicle_id,
                expires_at_utc=expires_at_utc,
                max_daily=max_daily,
                max_concurrent=max_concurrent,
                broker_active_total=predictive_scout_concurrent_count(snapshot),
                candidate_risk_jpy=candidate_risk_jpy,
                broker_active_risk_jpy=broker_active_risk_jpy,
                concurrent_risk_cap_jpy=concurrent_risk_cap_jpy,
                broker_active_signal_ids=predictive_scout_broker_signal_ids(snapshot),
            )
        except (OSError, sqlite3.Error, json.JSONDecodeError, ValueError) as exc:
            return RiskIssue(
                "PREDICTIVE_SCOUT_POST_RESERVATION_FAILED",
                "refusing broker POST because the predictive SCOUT vehicle/experiment reservation "
                f"could not be durably indexed first: {exc}",
            ).__dict__
        if reservation.status == "DUPLICATE_SIGNAL":
            return RiskIssue(
                "PREDICTIVE_SCOUT_EXPERIMENT_ALREADY_RESERVED",
                "this SCOUT vehicle/forecast_cycle signal already owns a durable broker-POST reservation; refusing duplicate evidence and duplicate exposure",
            ).__dict__
        if reservation.status == "DAILY_CAP_REACHED":
            return RiskIssue(
                "PREDICTIVE_SCOUT_DAILY_CAP_REACHED_AT_RESERVATION",
                f"atomic SCOUT reservation found {reservation.daily_reserved} broker-POST slots "
                f"already consumed today (cap {max_daily})",
            ).__dict__
        if reservation.status == "CONCURRENT_CAP_REACHED":
            return RiskIssue(
                "PREDICTIVE_SCOUT_CONCURRENT_CAP_REACHED_AT_RESERVATION",
                f"atomic SCOUT reservation found {reservation.active_slots} broker/reflection-aware "
                f"active slots already occupied (cap {max_concurrent})",
            ).__dict__
        if reservation.status == "CONCURRENT_RISK_CAP_REACHED":
            return RiskIssue(
                "PREDICTIVE_SCOUT_CONCURRENT_NAV_RISK_CAP_REACHED_AT_RESERVATION",
                f"atomic SCOUT reservation found aggregate initial risk {reservation.aggregate_risk_jpy:.4f} JPY above current cap {reservation.concurrent_risk_cap_jpy:.4f} JPY",
            ).__dict__
        if not reservation.reserved:
            return RiskIssue(
                "PREDICTIVE_SCOUT_POST_RESERVATION_FAILED",
                f"predictive SCOUT reservation returned unsupported status {reservation.status}",
            ).__dict__
        return None

    def _write_report(self, result: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Live Order Stage Report",
            "",
            f"- Generated at UTC: `{result['generated_at_utc']}`",
            f"- Status: `{result['status']}`",
            f"- Lane: `{result.get('lane_id')}`",
            f"- Lanes: `{', '.join(str(item) for item in result.get('lane_ids', []) if item) or result.get('lane_id')}`",
            f"- Requested units: `{result.get('requested_units')}` size multiple: `{result.get('size_multiple')}` scaled units:`{result.get('scaled_units')}`",
            f"- Send requested: `{result.get('send_requested')}`",
            f"- Sent: `{result.get('sent')}`",
            f"- Sent count: `{result.get('sent_count', 1 if result.get('sent') else 0)}`",
            f"- Quote refresh attempts: `{result.get('quote_refresh_attempts', 0)}`",
            f"- Portfolio position cap: `{result.get('portfolio_position_cap')}`",
            "",
            "## Order Request",
            "",
        ]
        batch_orders = result.get("orders") if isinstance(result.get("orders"), list) else []
        if batch_orders:
            for item in batch_orders:
                order = item.get("order_request")
                lane_id = item.get("lane_id")
                lines.append(f"- `{lane_id}` status=`{item.get('status')}` sent=`{item.get('sent')}`")
                if not order:
                    continue
                lines.append(f"  - quote_refresh_attempts=`{item.get('quote_refresh_attempts', 0)}`")
                lines.append(f"  - `{order['instrument']}` `{order['type']}` units=`{order['units']}`")
                if "price" in order:
                    lines.append(f"  - price: `{order['price']}`")
                if order.get("takeProfitOnFill"):
                    lines.append(f"  - takeProfitOnFill: `{order['takeProfitOnFill']['price']}`")
                if order.get("stopLossOnFill"):
                    lines.append(f"  - stopLossOnFill: `{order['stopLossOnFill']['price']}`")
                else:
                    lines.append("  - stopLossOnFill: `none (SL-free)`")
                metrics = item.get("risk_metrics") if isinstance(item.get("risk_metrics"), dict) else None
                if metrics:
                    lines.append(
                        f"  - intent risk: `{metrics['risk_jpy']:.1f} JPY` reward=`{metrics['reward_jpy']:.1f} JPY` "
                        f"rr=`{metrics['reward_risk']:.2f}` margin=`{metrics.get('estimated_margin_jpy', 0.0):.1f} JPY`"
                    )
                attached_stop = item.get("attached_stop_risk_metrics")
                if isinstance(attached_stop, dict):
                    lines.append(
                        f"  - attached broker SL: `{attached_stop['basis']}` price=`{attached_stop['price']}` "
                        f"loss=`{attached_stop['loss_pips']:.1f}pip` risk=`{attached_stop['risk_jpy']:.1f} JPY`"
                    )
                lines.extend(_sl_lint_report_lines(item.get("sl_lint"), prefix="  - "))
                lines.extend(_sizing_evidence_report_lines(item.get("sizing_evidence"), prefix="  - "))
                lines.extend(_target_path_receipt_report_lines(item.get("target_path_receipt"), prefix="  - "))
        order = None if batch_orders else result.get("order_request")
        if order:
            lines.append(f"- `{order['instrument']}` `{order['type']}` units=`{order['units']}`")
            if "price" in order:
                lines.append(f"- price: `{order['price']}`")
            if order.get("takeProfitOnFill"):
                lines.append(f"- takeProfitOnFill: `{order['takeProfitOnFill']['price']}`")
            if order.get("stopLossOnFill"):
                lines.append(f"- stopLossOnFill: `{order['stopLossOnFill']['price']}`")
            else:
                lines.append("- stopLossOnFill: `none (SL-free)`")
            metrics = result.get("risk_metrics") if isinstance(result.get("risk_metrics"), dict) else None
            if metrics:
                margin_tail = ""
                if metrics.get("estimated_margin_jpy") is not None:
                    margin_tail = f" margin=`{metrics['estimated_margin_jpy']:.1f} JPY`"
                    after = metrics.get("margin_utilization_after_pct")
                    cap = metrics.get("max_margin_utilization_pct")
                    if after is not None and cap is not None:
                        margin_tail += f" margin_after=`{after:.1f}%/{cap:.1f}%`"
                lines.append(
                    f"- intent risk: `{metrics['risk_jpy']:.1f} JPY` reward=`{metrics['reward_jpy']:.1f} JPY` "
                    f"rr=`{metrics['reward_risk']:.2f}` spread=`{metrics['spread_pips']:.1f}pip`{margin_tail}"
                )
            attached_stop = result.get("attached_stop_risk_metrics")
            if isinstance(attached_stop, dict):
                lines.append(
                    f"- attached broker SL: `{attached_stop['basis']}` price=`{attached_stop['price']}` "
                    f"loss=`{attached_stop['loss_pips']:.1f}pip` risk=`{attached_stop['risk_jpy']:.1f} JPY`"
                )
            lines.extend(_sl_lint_report_lines(result.get("sl_lint"), prefix="- "))
            lines.extend(_sizing_evidence_report_lines(result.get("sizing_evidence"), prefix="- "))
            lines.extend(_target_path_receipt_report_lines(result.get("target_path_receipt"), prefix="- "))
        elif not batch_orders:
            lines.append("- none")
        lines.extend(["", "## Issues", ""])
        issues = [*result.get("risk_issues", []), *result.get("strategy_issues", [])]
        if issues:
            for issue in issues:
                lines.append(f"- `{issue['severity']}` {issue['code']}: {issue['message']}")
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "## Send Contract",
                "",
                "- This command stages by default and sends nothing.",
                "- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _select_intent(payload: dict[str, Any], lane_id: str | None) -> dict[str, Any] | None:
    candidates = [item for item in payload.get("results", []) or [] if isinstance(item, dict) and item.get("intent")]
    if lane_id:
        for item in candidates:
            if item.get("lane_id") == lane_id:
                return item
        return None
    for item in candidates:
        if item.get("status") == "LIVE_READY":
            return item
    return candidates[0] if candidates else None


def _snapshot_pairs(payload: dict[str, Any], intent: OrderIntent) -> tuple[str, ...]:
    pairs = {intent.pair, "USD_JPY"}
    for item in payload.get("results", []) or []:
        if isinstance(item, dict) and isinstance(item.get("intent"), dict):
            pair = item["intent"].get("pair")
            if pair:
                pairs.add(str(pair))
    return tuple(sorted(pairs))


def _result_contains_predictive_scout(payload: Any) -> bool:
    if isinstance(payload, dict):
        if payload.get("predictive_scout") is True:
            return True
        return any(_result_contains_predictive_scout(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_result_contains_predictive_scout(value) for value in payload)
    return False


def _scaled_units(
    units: int,
    size_multiple: float,
    *,
    sub_min_lot_mode: str = "clamp",
) -> tuple[int | None, list[RiskIssue]]:
    if not math.isfinite(size_multiple) or size_multiple <= 0:
        return None, [RiskIssue("INVALID_SIZE_MULTIPLE", "size_multiple must be a finite positive number")]
    scaled_abs = int(math.floor(abs(units) * size_multiple))
    if scaled_abs < 1:
        return None, [RiskIssue("SIZE_MULTIPLIER_TOO_SMALL", "scaled units would round to zero")]
    original_abs = abs(int(units))
    if (
        original_abs >= MIN_PRODUCTION_LOT_UNITS
        and scaled_abs < MIN_PRODUCTION_LOT_UNITS
        and not _min_lot_test_override_active()
    ):
        if sub_min_lot_mode == "block":
            return None, [
                RiskIssue(
                    "BASKET_CAPACITY_BELOW_MIN_LOT",
                    f"capacity scaling would reduce {original_abs}u to {scaled_abs}u, below the "
                    f"{MIN_PRODUCTION_LOT_UNITS}u production floor; skip this lane until "
                    "risk or margin room can fit the minimum lot.",
                )
            ]
        scaled_abs = MIN_PRODUCTION_LOT_UNITS
        scaled = scaled_abs if units >= 0 else -scaled_abs
        return scaled, [
            RiskIssue(
                "SIZE_MULTIPLE_CLAMPED_TO_MIN_LOT",
                f"size_multiple {size_multiple:.4g} would reduce {original_abs}u to "
                f"{int(math.floor(original_abs * size_multiple))}u; keeping "
                f"{MIN_PRODUCTION_LOT_UNITS}u production floor.",
                "WARN",
            )
        ]
    scaled = scaled_abs if units >= 0 else -scaled_abs
    return scaled, []


def _scaled_units_for_intent(intent: OrderIntent, size_multiple: float) -> tuple[int | None, list[RiskIssue], float]:
    if predictive_scout_intent_claimed(intent):
        if not math.isfinite(size_multiple) or size_multiple <= 0:
            return None, [RiskIssue("INVALID_SIZE_MULTIPLE", "size_multiple must be a finite positive number")], size_multiple
        if not math.isclose(size_multiple, 1.0, rel_tol=0.0, abs_tol=1e-9):
            return (
                None,
                [
                    RiskIssue(
                        "PREDICTIVE_SCOUT_SIZE_MULTIPLE_FORBIDDEN",
                        "predictive SCOUT units are fixed by current-NAV/SL risk before verification; "
                        f"size_multiple must remain 1.0, received {size_multiple:.4g}",
                    )
                ],
                size_multiple,
            )
        return intent.units, [], 1.0
    if not _intent_declares_hedge(intent):
        scaled_units, issues = _scaled_units(intent.units, size_multiple)
        return scaled_units, issues, size_multiple
    if not math.isfinite(size_multiple) or size_multiple <= 0:
        return None, [RiskIssue("INVALID_SIZE_MULTIPLE", "size_multiple must be a finite positive number")], size_multiple
    if math.isclose(size_multiple, 1.0):
        return intent.units, [], 1.0
    return (
        intent.units,
        [
            RiskIssue(
                "HEDGE_SIZE_MULTIPLE_IGNORED",
                "HEDGE units are capped by the uncovered opposite exposure; "
                f"kept {intent.units}u instead of applying size_multiple {size_multiple:.4g}.",
                "WARN",
            )
        ],
        1.0,
    )


def _portfolio_position_cap_from_state(
    state_path: Path = DEFAULT_DAILY_TARGET_STATE,
    *,
    policy: RiskPolicy | None = None,
) -> int:
    risk_policy = policy or RiskPolicy()
    base_cap = int(risk_policy.max_portfolio_positions)
    target_trades = _target_trades_per_day_from_state(state_path)
    if target_trades is None:
        return base_cap
    session_floor = math.ceil(target_trades / ACTIVE_FX_SESSION_BUCKETS_PER_DAY)
    return max(base_cap, session_floor)


def _target_trades_per_day_from_state(state_path: Path = DEFAULT_DAILY_TARGET_STATE) -> int | None:
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text())
        value = int(payload.get("target_trades_per_day") or 0)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if value > 0 else None


def _blocked_batch_result(*, generated_at: str, selected: dict[str, Any], lane_id: str, send: bool, issue: RiskIssue) -> dict[str, Any]:
    return {
        "generated_at_utc": generated_at,
        "status": "BLOCKED",
        "lane_id": str(selected.get("lane_id") or lane_id),
        "order_request": None,
        "risk_metrics": None,
        "risk_issues": [issue.__dict__],
        "strategy_issues": [],
        "send_requested": send,
        "sent": False,
        "response": None,
        "size_multiple": None,
        "requested_units": None,
        "scaled_units": None,
    }


def _selected_intent_is_predictive_scout(selected: dict[str, Any]) -> bool:
    intent = selected.get("intent") if isinstance(selected.get("intent"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    market_context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
    return predictive_scout_geometry_claimed(
        metadata,
        pair=str(intent.get("pair") or ""),
        side=str(intent.get("side") or ""),
        order_type=str(intent.get("order_type") or ""),
        method=str(market_context.get("method") or "") or None,
    )


def _receipt_risk_jpy(result: dict[str, Any]) -> float:
    metrics = result.get("risk_metrics") if isinstance(result.get("risk_metrics"), dict) else {}
    attached_stop = (
        result.get("attached_stop_risk_metrics")
        if isinstance(result.get("attached_stop_risk_metrics"), dict)
        else {}
    )
    return max(
        _positive_float(metrics.get("risk_jpy")) or 0.0,
        _positive_float(attached_stop.get("risk_jpy")) or 0.0,
    )


def _selected_parent_lane_key(selected: dict[str, Any], requested_lane_id: str | None) -> str:
    intent = selected.get("intent") if isinstance(selected, dict) else None
    metadata = intent.get("metadata") if isinstance(intent, dict) and isinstance(intent.get("metadata"), dict) else {}
    parent = metadata.get("parent_lane_id") if isinstance(metadata, dict) else None
    if isinstance(parent, str) and parent.strip():
        return parent.strip()
    lane_id = str(selected.get("lane_id") or requested_lane_id or "")
    return _parent_lane_id_from_lane_id(lane_id)


def _parent_lane_id_from_lane_id(lane_id: str) -> str:
    for suffix in (":MARKET", ":LIMIT", ":STOP"):
        if lane_id.endswith(suffix):
            return lane_id[: -len(suffix)]
    return lane_id


def _basket_size_multiple(
    *,
    intent: OrderIntent,
    snapshot: BrokerSnapshot,
    metrics,
    portfolio_loss_cap: float | None,
    cumulative_risk_jpy: float,
    cumulative_margin_jpy: float,
) -> tuple[float, RiskIssue | None]:
    scale = 1.0
    reasons: list[str] = []
    requested_units = abs(intent.units)
    pending_risk, pending_margin, pending_issue = _pending_risk_margin_jpy(snapshot)
    open_risk, open_issue = _open_trader_position_risk_jpy(snapshot)
    if pending_issue is not None:
        return 0.0, pending_issue
    if open_issue is not None:
        return 0.0, open_issue
    if portfolio_loss_cap is not None:
        remaining_risk = portfolio_loss_cap - open_risk - pending_risk - cumulative_risk_jpy
        if remaining_risk <= 0:
            return 0.0, RiskIssue(
                "BASKET_RISK_CAP_REACHED",
                f"basket risk capacity is exhausted: open {open_risk:.0f} + pending {pending_risk:.0f} "
                f"+ batch {cumulative_risk_jpy:.0f} >= cap {portfolio_loss_cap:.0f} JPY",
            )
        if metrics.risk_jpy > remaining_risk > 0:
            scale = min(scale, _capacity_scale(requested_units, metrics.risk_jpy, remaining_risk))
            reasons.append(f"risk room {remaining_risk:.0f} JPY")

    account = snapshot.account
    max_margin_pct = RiskPolicy().max_margin_utilization_pct
    if account is not None and max_margin_pct is not None:
        candidate_margin = max(0.0, float(metrics.estimated_margin_jpy or 0.0))
        if candidate_margin > 0:
            margin_room = margin_budget_jpy(account, max_margin_utilization_pct=max_margin_pct)
            remaining_margin = margin_room - pending_margin - cumulative_margin_jpy
            if remaining_margin <= 0:
                return 0.0, RiskIssue(
                    "BASKET_MARGIN_CAP_REACHED",
                    f"basket margin capacity is exhausted: pending {pending_margin:.0f} + batch {cumulative_margin_jpy:.0f} "
                    f">= room {margin_room:.0f} JPY",
                )
            if candidate_margin > remaining_margin > 0:
                scale = min(scale, _capacity_scale(requested_units, candidate_margin, remaining_margin))
                reasons.append(f"margin room {remaining_margin:.0f} JPY")

    if scale < 1.0:
        return scale, RiskIssue(
            "BASKET_DOWNSIZED_FOR_CAPACITY",
            f"{intent.pair} {intent.side.value} downsized to fit basket {' and '.join(reasons)}",
            "WARN",
        )
    return 1.0, None


def _capacity_scale(requested_units: int, current_value: float, capacity: float) -> float:
    """Scale units to fit a linear risk/margin capacity after integer flooring.

    OANDA order size is integer units while risk and margin are decimal account
    values. When we are capacity-bound, leave one unit of granularity as
    headroom so recomputing broker-truth risk/margin cannot round back over the
    cap by a few JPY and self-block the staged basket.
    """
    if requested_units <= 0 or current_value <= 0 or capacity <= 0:
        return 0.0
    value_per_unit = current_value / requested_units
    if value_per_unit <= 0:
        return 0.0
    max_units = math.floor(capacity / value_per_unit)
    if max_units < requested_units:
        max_units = max(0, max_units - 1)
    return max_units / requested_units


def _basket_issues(
    *,
    intent: OrderIntent,
    snapshot: BrokerSnapshot,
    metrics,
    portfolio_loss_cap: float | None,
    cumulative_risk_jpy: float,
    cumulative_margin_jpy: float,
    seen_geometry: set[tuple[object, ...]],
) -> list[RiskIssue]:
    issues: list[RiskIssue] = []
    geometry = _intent_geometry_key(intent)
    geometry_keys = {geometry}
    original_entry = _positive_float((intent.metadata or {}).get("gateway_passive_limit_original_entry"))
    if original_entry is not None:
        geometry_keys.add(_intent_geometry_key(intent, entry=original_entry))
    if any(key in seen_geometry for key in geometry_keys):
        issues.append(
            RiskIssue(
                "BASKET_DUPLICATE_GEOMETRY",
                f"{intent.pair} {intent.side.value} {intent.order_type.value} shares an existing entry/tp/sl geometry",
            )
        )
    if metrics is None:
        return issues
    pending_risk, pending_margin, pending_issue = _pending_risk_margin_jpy(snapshot)
    open_risk, open_issue = _open_trader_position_risk_jpy(snapshot)
    if pending_issue is not None:
        issues.append(pending_issue)
    if open_issue is not None:
        issues.append(open_issue)
    if portfolio_loss_cap is not None and pending_issue is None and open_issue is None:
        total_risk = open_risk + pending_risk + cumulative_risk_jpy + metrics.risk_jpy
        if total_risk > portfolio_loss_cap:
            issues.append(
                RiskIssue(
                    "BASKET_PORTFOLIO_LOSS_CAP_EXCEEDED",
                    f"basket worst-case loss {total_risk:.0f} JPY exceeds daily cap {portfolio_loss_cap:.0f} JPY",
                )
            )
    account = snapshot.account
    max_margin_pct = RiskPolicy().max_margin_utilization_pct
    if account is not None and max_margin_pct is not None:
        candidate_margin = max(0.0, float(metrics.estimated_margin_jpy or 0.0))
        if candidate_margin > 0:
            margin_room = margin_budget_jpy(account, max_margin_utilization_pct=max_margin_pct)
            total_margin = pending_margin + cumulative_margin_jpy + candidate_margin
            if total_margin > margin_room:
                issues.append(
                    RiskIssue(
                        "BASKET_MARGIN_UTILIZATION_CAP_EXCEEDED",
                        f"basket candidate margin {total_margin:.0f} JPY exceeds remaining {max_margin_pct:.1f}% "
                        f"margin room {margin_room:.0f} JPY",
                    )
                )
    return issues


def _trader_entry_occupancy(snapshot: BrokerSnapshot) -> int:
    positions = sum(1 for position in snapshot.positions if position.owner == Owner.TRADER)
    orders = sum(1 for order in snapshot.orders if _is_trader_pending_entry(order))
    return positions + orders


def _order_id_tuple(order_ids: tuple[str, ...] | list[str] | set[str] | None) -> tuple[str, ...]:
    if not order_ids:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for order_id in order_ids:
        value = str(order_id or "").strip()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return tuple(normalized)


def _snapshot_without_trader_pending_orders(
    snapshot: BrokerSnapshot,
    order_ids: tuple[str, ...] | list[str] | set[str] | None,
) -> BrokerSnapshot:
    ignored = set(_order_id_tuple(order_ids))
    if not ignored:
        return snapshot
    orders = tuple(
        order
        for order in snapshot.orders
        if not (order.order_id in ignored and _is_trader_pending_entry(order))
    )
    return snapshot if len(orders) == len(snapshot.orders) else replace(snapshot, orders=orders)


def _pending_geometry_keys(snapshot: BrokerSnapshot) -> tuple[tuple[object, ...], ...]:
    keys: list[tuple[object, ...]] = []
    for order in snapshot.orders:
        if not _is_trader_pending_entry(order):
            continue
        key = _pending_order_geometry_key(order)
        if key is not None:
            keys.append(key)
    return tuple(keys)


def _pending_parent_lane_keys(snapshot: BrokerSnapshot) -> tuple[str, ...]:
    keys: list[str] = []
    for order in snapshot.orders:
        if not _is_trader_pending_entry(order):
            continue
        key = _pending_order_parent_lane_key(order)
        if key:
            keys.append(key)
    return tuple(keys)


def _fresh_pending_entry_issues(
    *,
    intent: OrderIntent,
    snapshot: BrokerSnapshot,
    portfolio_position_cap: int,
) -> tuple[list[RiskIssue], dict[str, Any]]:
    """Rebuild duplicate and occupancy state from the final broker snapshot.

    Initial validation can race with another gateway process or a broker-side
    pending-order reflection.  This check intentionally derives every key from
    the snapshot fetched after ledger sync, rather than carrying the initial
    process-local ``seen_geometry`` set across the network boundary.
    """

    issues: list[RiskIssue] = []
    pending_geometry = set(_pending_geometry_keys(snapshot))
    pending_parents = set(_pending_parent_lane_keys(snapshot))
    geometry_keys = {_intent_geometry_key(intent)}
    original_entry = _positive_float(
        (intent.metadata or {}).get("gateway_passive_limit_original_entry")
    )
    if original_entry is not None:
        geometry_keys.add(_intent_geometry_key(intent, entry=original_entry))
    if geometry_keys & pending_geometry:
        issues.append(
            RiskIssue(
                "PRE_POST_DUPLICATE_PENDING_GEOMETRY",
                f"fresh broker truth already has trader pending geometry matching "
                f"{intent.pair} {intent.side.value} {intent.order_type.value}; refuse a second POST",
            )
        )

    lane_id = _gateway_lane_id(intent)
    parent_lane_id = str((intent.metadata or {}).get("parent_lane_id") or "").strip()
    if not parent_lane_id:
        parent_lane_id = _parent_lane_id_from_lane_id(lane_id)
    if parent_lane_id and parent_lane_id in pending_parents:
        issues.append(
            RiskIssue(
                "PRE_POST_DUPLICATE_PENDING_PARENT_LANE",
                f"fresh broker truth already has a trader pending entry for parent lane "
                f"{parent_lane_id}; one timing variant cannot be posted twice",
            )
        )

    occupancy = _trader_entry_occupancy(snapshot)
    if occupancy >= portfolio_position_cap:
        issues.append(
            RiskIssue(
                "PRE_POST_PORTFOLIO_POSITION_LIMIT",
                f"fresh broker truth has {occupancy} trader positions/pending entries; "
                f"portfolio cap is {portfolio_position_cap}",
            )
        )

    return issues, {
        "status": "BLOCKED" if any(issue.severity == "BLOCK" for issue in issues) else "PASSED",
        "pending_entry_count": sum(
            1 for order in snapshot.orders if _is_trader_pending_entry(order)
        ),
        "pending_geometry_count": len(pending_geometry),
        "pending_parent_lane_count": len(pending_parents),
        "candidate_parent_lane_id": parent_lane_id or None,
        "trader_entry_occupancy": occupancy,
        "portfolio_position_cap": portfolio_position_cap,
        "issue_codes": [issue.code for issue in issues],
    }


def _pending_risk_margin_jpy(snapshot: BrokerSnapshot) -> tuple[float, float, RiskIssue | None]:
    risk = 0.0
    margin = 0.0
    for order in snapshot.orders:
        if not _is_trader_pending_entry(order):
            continue
        if not order.pair or order.price is None or not order.units:
            return 0.0, 0.0, RiskIssue("PENDING_RISK_UNKNOWN", f"pending order {order.order_id} is missing pair/price/units")
        spec = DEFAULT_SPECS.get(order.pair)
        if spec is None:
            return 0.0, 0.0, RiskIssue("PENDING_RISK_UNKNOWN", f"pending order {order.order_id} pair {order.pair} is unsupported")
        quote_to_jpy = _quote_to_jpy(order.pair, snapshot)
        if quote_to_jpy is None:
            return 0.0, 0.0, RiskIssue("PENDING_RISK_UNKNOWN", f"missing conversion quote for pending order {order.order_id}")
        side = Side.LONG if order.units > 0 else Side.SHORT
        sl = _raw_dependent_price(order.raw, "stopLossOnFill")
        if sl is None:
            if _trader_sl_repair_disabled():
                loss_pips = SL_FREE_SYNTHETIC_RISK_PIPS
            else:
                return 0.0, 0.0, RiskIssue("PENDING_RISK_UNKNOWN", f"pending order {order.order_id} has no stopLossOnFill")
        elif side == Side.LONG:
            loss_pips = (order.price - sl) * spec.pip_factor
        else:
            loss_pips = (sl - order.price) * spec.pip_factor
        jpy_per_pip = (abs(order.units) / spec.pip_factor) * quote_to_jpy
        risk += max(0.0, loss_pips) * jpy_per_pip
        margin += estimate_required_margin_jpy(
            units=abs(order.units),
            entry_price=order.price,
            quote_to_jpy=quote_to_jpy,
            spec=spec,
        )
    return risk, margin, None


def _open_trader_position_risk_jpy(snapshot: BrokerSnapshot) -> tuple[float, RiskIssue | None]:
    risk = 0.0
    sl_free_active = _trader_sl_repair_disabled()
    for position in snapshot.positions:
        if position.owner != Owner.TRADER:
            continue
        spec = DEFAULT_SPECS.get(position.pair)
        if spec is None:
            return 0.0, RiskIssue("BASKET_OPEN_RISK_UNKNOWN", f"position {position.trade_id} pair {position.pair} is unsupported")
        quote_to_jpy = _quote_to_jpy(position.pair, snapshot)
        if quote_to_jpy is None:
            return 0.0, RiskIssue("BASKET_OPEN_RISK_UNKNOWN", f"missing conversion quote for position {position.trade_id}")
        if position.stop_loss is None:
            if sl_free_active:
                loss_pips = SL_FREE_SYNTHETIC_RISK_PIPS
            else:
                return 0.0, RiskIssue("BASKET_OPEN_RISK_UNKNOWN", f"trader position {position.trade_id} has no SL")
        else:
            if position.side == Side.LONG:
                loss_pips = (position.entry_price - position.stop_loss) * spec.pip_factor
            else:
                loss_pips = (position.stop_loss - position.entry_price) * spec.pip_factor
        jpy_per_pip = (position.units / spec.pip_factor) * quote_to_jpy
        risk += max(0.0, loss_pips) * jpy_per_pip
    return risk, None


def _is_trader_pending_entry(order) -> bool:
    if order.owner != Owner.TRADER or order.trade_id:
        return False
    return str(order.order_type or "").upper() in {"LIMIT", "STOP", "STOP-ENTRY", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}


def _pending_order_geometry_key(order) -> tuple[object, ...] | None:
    if not order.pair or order.price is None or not order.units:
        return None
    side = Side.LONG if order.units > 0 else Side.SHORT
    tp = _raw_dependent_price(order.raw, "takeProfitOnFill")
    sl = _raw_dependent_price(order.raw, "stopLossOnFill")
    order_type = "STOP-ENTRY" if str(order.order_type).upper() == "STOP" else str(order.order_type).upper()
    return (order.pair, side.value, order_type, _price_key(order.pair, order.price), _price_key(order.pair, tp), _price_key(order.pair, sl))


def _pending_order_parent_lane_key(order) -> str | None:
    raw = order.raw if isinstance(order.raw, dict) else {}
    for extension_key in ("clientExtensions", "tradeClientExtensions"):
        extension = raw.get(extension_key)
        if not isinstance(extension, dict):
            continue
        parent_lane_id = _parent_lane_id_from_extension_comment(
            extension.get("comment")
        )
        if parent_lane_id:
            return parent_lane_id
        lane_id = _lane_id_from_extension_comment(extension.get("comment"))
        if lane_id:
            return _parent_lane_id_from_lane_id(lane_id)
    return None


def _parent_lane_id_from_extension_comment(comment: Any) -> str | None:
    if not isinstance(comment, str):
        return None
    for token in comment.split():
        if token.startswith("parent="):
            parent_lane_id = token[len("parent="):].strip()
            return parent_lane_id or None
    return None


def _lane_id_from_extension_comment(comment: Any) -> str | None:
    if not isinstance(comment, str):
        return None
    for token in comment.split():
        if token.startswith("lane="):
            lane_id = token[len("lane="):].strip()
            return lane_id or None
    return None


def _intent_geometry_key(intent: OrderIntent, *, entry: float | None = None) -> tuple[object, ...]:
    return (
        intent.pair,
        intent.side.value,
        intent.order_type.value,
        _price_key(intent.pair, intent.entry if entry is None else entry),
        _price_key(intent.pair, intent.tp if _attach_take_profit_on_fill(intent) else None),
        _price_key(intent.pair, _attached_stop_loss_price(intent)),
    )


def _intent_declares_hedge(intent: OrderIntent) -> bool:
    return str((intent.metadata or {}).get("position_intent") or "").upper() == "HEDGE"


def _attached_stop_loss_price(intent: OrderIntent) -> float | None:
    initial_sl_on = os.environ.get("QR_NEW_ENTRY_INITIAL_SL", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }
    if initial_sl_on or not _trader_sl_repair_disabled() or _requires_intent_stop_on_fill(intent):
        return intent.sl
    disaster_sl = (intent.metadata or {}).get("disaster_sl")
    if disaster_sl is None:
        return None
    try:
        return float(disaster_sl)
    except (TypeError, ValueError):
        return None


def _requires_intent_stop_on_fill(intent: OrderIntent) -> bool:
    metadata = intent.metadata or {}
    if predictive_scout_metadata_supported(metadata):
        return True
    if metadata.get("broker_stop_loss_mode") == "INTENT_SL":
        return True
    return (
        metadata.get("campaign_role") == "OANDA_FIREPOWER_ROUTE"
        and metadata.get("positive_rotation_oanda_campaign_firepower_vehicle_match") is True
    )


def _attached_stop_risk_metrics(
    intent: OrderIntent,
    order_request: dict[str, Any] | None,
    metrics: RiskMetrics | None,
) -> dict[str, Any] | None:
    if not order_request or metrics is None:
        return None
    attached_sl = _raw_dependent_price(order_request, "stopLossOnFill")
    if attached_sl is None:
        return None
    spec = DEFAULT_SPECS.get(intent.pair)
    if spec is None:
        return None
    try:
        entry = float(order_request.get("price") or metrics.entry_price)
    except (TypeError, ValueError):
        entry = metrics.entry_price
    if intent.side == Side.LONG:
        loss_pips = (entry - attached_sl) * spec.pip_factor
    else:
        loss_pips = (attached_sl - entry) * spec.pip_factor
    loss_pips = max(0.0, loss_pips)
    risk_jpy = loss_pips * metrics.jpy_per_pip
    basis = "BROKER_ATTACHED_SL"
    tolerance = 0.5 / spec.pip_factor
    disaster_sl_raw = (intent.metadata or {}).get("disaster_sl")
    try:
        disaster_sl = None if disaster_sl_raw is None else float(disaster_sl_raw)
    except (TypeError, ValueError):
        disaster_sl = None
    if disaster_sl is not None and abs(attached_sl - disaster_sl) <= tolerance:
        basis = "DISASTER_SL"
    elif abs(attached_sl - intent.sl) <= tolerance:
        basis = "INTENT_SL"
    return {
        "basis": basis,
        "price": _price(intent.pair, attached_sl),
        "loss_pips": loss_pips,
        "risk_jpy": risk_jpy,
        "intent_sl": _price(intent.pair, intent.sl),
        "intent_loss_pips": metrics.loss_pips,
        "intent_risk_jpy": metrics.risk_jpy,
        "risk_delta_jpy": risk_jpy - metrics.risk_jpy,
        "loss_delta_pips": loss_pips - metrics.loss_pips,
    }


SL_LINT_ALLOWED_PURPOSES = {
    "protection": "protection",
    "thesis_invalidation": "thesis invalidation",
    "thesis invalidation": "thesis invalidation",
    "emergency_only": "emergency only",
    "emergency only": "emergency only",
}


def _sl_lint_result(
    *,
    intent: OrderIntent,
    snapshot: BrokerSnapshot,
    order_request: dict[str, Any] | None,
    metrics: RiskMetrics | None,
    attached_stop_metrics: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    metadata = intent.metadata or {}
    attached_sl = _raw_dependent_price(order_request or {}, "stopLossOnFill")
    purpose = _sl_lint_purpose(intent, attached_stop_metrics)
    invalidation_price = _sl_lint_invalidation_price(intent, attached_sl)
    invalidation_structure = _sl_lint_first_text(
        metadata,
        "invalidation_structure",
        "sl_invalidation_structure",
        "structure_invalidation",
        "thesis_invalidation_structure",
    )
    if not invalidation_structure and intent.market_context is not None:
        invalidation_structure = intent.market_context.invalidation
    what_proves_dead = _sl_lint_first_text(
        metadata,
        "what_proves_thesis_dead",
        "thesis_dead_condition",
        "sl_thesis_dead_condition",
        "invalidation_condition",
    )
    if not what_proves_dead and intent.market_context is not None:
        what_proves_dead = intent.market_context.invalidation

    noise_floor_pips = _sl_lint_normal_noise_floor_pips(intent, snapshot, metrics)
    battle_zone_pips = _sl_lint_battle_zone_pips(intent, noise_floor_pips)
    base = {
        "status": "PASS",
        "attached_stop_present": attached_sl is not None,
        "stop_mode": "BROKER_SL" if attached_sl is not None else "NO_BROKER_SL",
        "sl_purpose": purpose,
        "attached_stop_price": _price(intent.pair, attached_sl) if attached_sl is not None else None,
        "attached_stop_basis": (
            attached_stop_metrics.get("basis") if isinstance(attached_stop_metrics, dict) else None
        ),
        "invalidation_price": _price(intent.pair, invalidation_price) if invalidation_price is not None else None,
        "invalidation_structure": invalidation_structure or "not supplied",
        "why_sl_outside_normal_noise": "no broker trading SL attached; intent invalidation remains management evidence",
        "what_proves_thesis_dead": what_proves_dead or "not supplied",
        "normal_noise_floor_pips": round(noise_floor_pips, 4),
        "battle_zone_pips": round(battle_zone_pips, 4),
        "theme_group": _jpy_theme_group(intent),
        "jpy_theme_invalidation_price": _sl_lint_theme_invalidation_price(intent),
        "issues": [],
    }
    if attached_sl is None:
        return base, []

    issues: list[dict[str, str]] = []
    emergency_only = purpose == "emergency only"
    attached_loss_pips = None
    if isinstance(attached_stop_metrics, dict):
        attached_loss_pips = _positive_float(attached_stop_metrics.get("loss_pips"))
    if attached_loss_pips is None and metrics is not None:
        attached_loss_pips = _sl_lint_loss_pips(intent, attached_sl, metrics.entry_price)
    if attached_loss_pips is not None:
        base["attached_stop_loss_pips"] = round(attached_loss_pips, 4)
        base["why_sl_outside_normal_noise"] = (
            f"attached stop is {attached_loss_pips:.1f}pip from entry vs normal-noise floor "
            f"{noise_floor_pips:.1f}pip derived from live spread/ATR context"
        )
        if attached_loss_pips + 1e-6 < noise_floor_pips:
            issues.append(
                _sl_lint_issue(
                    "SL_LINT_NORMAL_NOISE_BAND",
                    f"{intent.pair} {intent.side.value} broker SL is {attached_loss_pips:.1f}pip from entry, "
                    f"inside the {noise_floor_pips:.1f}pip normal-noise floor; this is not thesis invalidation",
                    severity="WARN" if emergency_only else "BLOCK",
                )
            )

    major_levels = _sl_lint_major_levels(intent, attached_sl)
    for level in major_levels:
        distance_pips = _sl_lint_distance_pips(intent.pair, attached_sl, level["price"])
        if distance_pips is not None and distance_pips <= battle_zone_pips:
            base["major_figure_battle_zone"] = {
                "level": _price(intent.pair, level["price"]),
                "source": level["source"],
                "distance_pips": round(distance_pips, 4),
            }
            issues.append(
                _sl_lint_issue(
                    "SL_LINT_MAJOR_FIGURE_BATTLE_ZONE",
                    f"{intent.pair} broker SL {_price(intent.pair, attached_sl)} sits {distance_pips:.1f}pip from "
                    f"{level['source']} {_price(intent.pair, level['price'])}; avoid major-figure battle zones "
                    "unless this is emergency-only protection",
                    severity="WARN" if emergency_only else "BLOCK",
                )
            )
            break

    wick_zone = _sl_lint_matching_zone(intent, attached_sl, _SL_LINT_WICK_ZONE_KEYS)
    if wick_zone is not None:
        base["recent_wick_stop_run_zone"] = wick_zone
        issues.append(
            _sl_lint_issue(
                "SL_LINT_RECENT_WICK_STOP_RUN_ZONE",
                f"{intent.pair} broker SL {_price(intent.pair, attached_sl)} is inside {wick_zone['label']} "
                f"{wick_zone['low']}..{wick_zone['high']}; wick/stop-run zones do not prove thesis failure",
                severity="WARN" if emergency_only else "BLOCK",
            )
        )

    event_zone = _sl_lint_matching_zone(intent, attached_sl, _SL_LINT_EVENT_ZONE_KEYS)
    event_text_hit = _sl_lint_event_intervention_text_hit(intent)
    if event_zone is not None or (event_text_hit and "major_figure_battle_zone" in base):
        if event_zone is not None:
            base["event_intervention_zone"] = event_zone
            message = (
                f"{intent.pair} broker SL {_price(intent.pair, attached_sl)} is inside {event_zone['label']} "
                f"{event_zone['low']}..{event_zone['high']}; event/intervention zones need wider or emergency-only handling"
            )
        else:
            message = (
                f"{intent.pair} broker SL is in a JPY major-figure zone while the receipt cites intervention/event risk; "
                "that zone needs shared thesis invalidation or emergency-only handling"
            )
        issues.append(
            _sl_lint_issue(
                "SL_LINT_EVENT_INTERVENTION_ZONE",
                message,
                severity="WARN" if emergency_only else "BLOCK",
            )
        )

    theme_issue = _sl_lint_jpy_theme_issue(
        intent=intent,
        snapshot=snapshot,
        attached_sl=attached_sl,
        emergency_only=emergency_only,
    )
    if theme_issue is not None:
        issues.append(theme_issue)

    status = "BLOCK" if any(issue["severity"] == "BLOCK" for issue in issues) else ("WARN" if issues else "PASS")
    base["status"] = status
    base["issues"] = issues
    return base, issues


def _sl_lint_purpose(intent: OrderIntent, attached_stop_metrics: dict[str, Any] | None) -> str:
    if isinstance(attached_stop_metrics, dict) and attached_stop_metrics.get("basis") == "DISASTER_SL":
        return "emergency only"
    raw = str((intent.metadata or {}).get("sl_purpose") or "").strip().lower().replace("-", "_")
    if raw in SL_LINT_ALLOWED_PURPOSES:
        return SL_LINT_ALLOWED_PURPOSES[raw]
    mode = str((intent.metadata or {}).get("broker_stop_loss_mode") or "").strip().upper()
    if mode in {"DISASTER_SL", "EMERGENCY_ONLY"}:
        return "emergency only"
    if mode in {"PROTECTION", "PROTECTIVE"}:
        return "protection"
    return "thesis invalidation"


def _sl_lint_invalidation_price(intent: OrderIntent, attached_sl: float | None) -> float | None:
    metadata = intent.metadata or {}
    for key in (
        "invalidation_price",
        "forecast_invalidation_price",
        "sl_invalidation_price",
        "thesis_invalidation_price",
        "jpy_theme_invalidation_price",
        "theme_invalidation_price",
    ):
        value = _positive_float(metadata.get(key))
        if value is not None:
            return value
    return attached_sl if attached_sl is not None else _positive_float(intent.sl)


def _sl_lint_first_text(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _sl_lint_normal_noise_floor_pips(
    intent: OrderIntent,
    snapshot: BrokerSnapshot,
    metrics: RiskMetrics | None,
) -> float:
    spread_pips = None
    if metrics is not None:
        spread_pips = _positive_float(metrics.spread_pips)
    if spread_pips is None:
        quote = snapshot.quotes.get(intent.pair)
        spec = DEFAULT_SPECS.get(intent.pair)
        if quote is not None and spec is not None:
            spread_pips = abs(quote.ask - quote.bid) * spec.pip_factor
    floor = (spread_pips or 0.0) * RiskPolicy().min_stop_spread_multiple
    metadata = intent.metadata or {}
    for key in ("tp_atr_pips", "atr_pips", "m15_atr_pips", "h1_atr_pips", "normal_noise_floor_pips"):
        value = _positive_float(metadata.get(key))
        if value is not None:
            floor = max(floor, value)
    return max(0.0, floor)


def _sl_lint_battle_zone_pips(intent: OrderIntent, noise_floor_pips: float) -> float:
    metadata = intent.metadata or {}
    width = noise_floor_pips
    for key in ("level_cluster_radius_pips", "major_figure_radius_pips", "tp_atr_pips", "atr_pips"):
        value = _positive_float(metadata.get(key))
        if value is not None:
            width = max(width, value)
    return width


def _sl_lint_loss_pips(intent: OrderIntent, stop_price: float, entry_price: float) -> float | None:
    spec = DEFAULT_SPECS.get(intent.pair)
    if spec is None:
        return None
    if intent.side == Side.LONG:
        return max(0.0, (entry_price - stop_price) * spec.pip_factor)
    return max(0.0, (stop_price - entry_price) * spec.pip_factor)


def _sl_lint_distance_pips(pair: str, first: float, second: float) -> float | None:
    spec = DEFAULT_SPECS.get(pair)
    if spec is None:
        return None
    return abs(first - second) * spec.pip_factor


def _sl_lint_major_levels(intent: OrderIntent, attached_sl: float) -> list[dict[str, Any]]:
    metadata = intent.metadata or {}
    levels: list[dict[str, Any]] = []
    for key in ("nearest_levels_above", "nearest_levels_below", "key_levels", "major_levels"):
        raw = metadata.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or item.get("label") or key).strip()
            level_price = _positive_float(item.get("price"))
            if level_price is None:
                continue
            if "round" in source.lower() or "major" in source.lower() or key == "major_levels":
                levels.append({"price": level_price, "source": source or key})
    if intent.pair.endswith("_JPY") and _sl_lint_implicit_jpy_major_figure_active(intent):
        figure = float(round(attached_sl))
        if figure > 0:
            levels.append({"price": figure, "source": "JPY integer major figure"})
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, str]] = set()
    for level in levels:
        key = (round(float(level["price"]), _price_precision(intent.pair)), str(level["source"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(level)
    return deduped


def _sl_lint_implicit_jpy_major_figure_active(intent: OrderIntent) -> bool:
    text = _sl_lint_text_blob(intent).lower()
    return any(
        marker in text
        for marker in (
            "major figure",
            "round number",
            "psychological",
            "intervention",
            "rate check",
            "event risk",
            "介入",
        )
    )


_SL_LINT_WICK_ZONE_KEYS = (
    ("recent_wick_zone_low", "recent_wick_zone_high", "recent wick zone"),
    ("recent_stop_run_zone_low", "recent_stop_run_zone_high", "recent stop-run zone"),
    ("stop_run_zone_low", "stop_run_zone_high", "stop-run zone"),
    ("wick_zone_low", "wick_zone_high", "wick zone"),
)
_SL_LINT_EVENT_ZONE_KEYS = (
    ("event_risk_zone_low", "event_risk_zone_high", "event risk zone"),
    ("intervention_risk_zone_low", "intervention_risk_zone_high", "intervention risk zone"),
)


def _sl_lint_matching_zone(
    intent: OrderIntent,
    attached_sl: float,
    zone_keys: tuple[tuple[str, str, str], ...],
) -> dict[str, Any] | None:
    metadata = intent.metadata or {}
    for low_key, high_key, label in zone_keys:
        low = _positive_float(metadata.get(low_key))
        high = _positive_float(metadata.get(high_key))
        if low is None or high is None:
            continue
        lower, upper = sorted((low, high))
        if lower <= attached_sl <= upper:
            return {
                "label": label,
                "low": _price(intent.pair, lower),
                "high": _price(intent.pair, upper),
            }
    return None


def _sl_lint_event_intervention_text_hit(intent: OrderIntent) -> bool:
    text = _sl_lint_text_blob(intent).lower()
    return any(
        marker in text
        for marker in (
            "intervention",
            "rate check",
            "event risk",
            "boj",
            "mof",
            "jpy short",
            "yen intervention",
            "介入",
        )
    )


def _sl_lint_text_blob(intent: OrderIntent) -> str:
    parts = [intent.thesis, intent.reason]
    if intent.market_context is not None:
        parts.extend(
            [
                intent.market_context.regime,
                intent.market_context.narrative,
                intent.market_context.chart_story,
                intent.market_context.invalidation,
                intent.market_context.event_risk,
                intent.market_context.session,
            ]
        )
    for key, value in (intent.metadata or {}).items():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (int, float, bool)):
            parts.append(f"{key}={value}")
        elif isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value if isinstance(item, str))
    return " ".join(str(part) for part in parts if part)


def _jpy_theme_group(intent: OrderIntent) -> str | None:
    if not intent.pair.endswith("_JPY"):
        return None
    return "JPY_WEAKNESS" if intent.side == Side.LONG else "JPY_STRENGTH_REVERSAL"


def _sl_lint_theme_invalidation_price(intent: OrderIntent) -> str | None:
    for key in ("jpy_theme_invalidation_price", "theme_invalidation_price"):
        value = _positive_float((intent.metadata or {}).get(key))
        if value is not None:
            return _price(intent.pair, value)
    return None


def _sl_lint_jpy_theme_issue(
    *,
    intent: OrderIntent,
    snapshot: BrokerSnapshot,
    attached_sl: float,
    emergency_only: bool,
) -> dict[str, str] | None:
    theme = _jpy_theme_group(intent)
    if theme is None or _sl_lint_theme_invalidation_price(intent) is not None:
        return None
    conflicts: list[str] = []
    for position in snapshot.positions:
        if position.owner != Owner.TRADER or not position.pair.endswith("_JPY") or position.pair == intent.pair:
            continue
        if _jpy_theme_group_for(position.pair, position.side) == theme:
            conflicts.append(f"position {position.trade_id} {position.pair} {position.side.value}")
    for order in snapshot.orders:
        if not _is_trader_pending_entry(order) or not order.pair or not order.pair.endswith("_JPY") or order.pair == intent.pair:
            continue
        if not order.units:
            continue
        side = Side.LONG if order.units > 0 else Side.SHORT
        if _jpy_theme_group_for(order.pair, side) == theme:
            conflicts.append(f"pending order {order.order_id} {order.pair} {side.value}")
    if not conflicts:
        return None
    return _sl_lint_issue(
        "SL_LINT_JPY_THEME_INVALIDATION_REQUIRED",
        f"{intent.pair} {intent.side.value} broker SL {_price(intent.pair, attached_sl)} creates a separate "
        f"{theme} stop while {'; '.join(conflicts)} already carries the same JPY theme. Use one theme-level "
        "invalidation and shared JPY-theme risk budget instead of per-pair tight broker SLs.",
        severity="WARN" if emergency_only else "BLOCK",
    )


def _jpy_theme_group_for(pair: str, side: Side) -> str | None:
    if not pair.endswith("_JPY"):
        return None
    return "JPY_WEAKNESS" if side == Side.LONG else "JPY_STRENGTH_REVERSAL"


def _sl_lint_issue(code: str, message: str, *, severity: str = "BLOCK") -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def _sl_lint_report_lines(value: Any, *, prefix: str) -> list[str]:
    if not isinstance(value, dict) or not value:
        return []
    lines = [
        f"{prefix}SL_LINT: status=`{value.get('status')}` purpose=`{value.get('sl_purpose')}` "
        f"invalidation=`{value.get('invalidation_price')}` noise_floor=`{value.get('normal_noise_floor_pips')}pip`"
    ]
    dead = str(value.get("what_proves_thesis_dead") or "").strip()
    if dead and dead != "not supplied":
        lines.append(f"{prefix}SL proves dead: `{dead}`")
    issue_codes = [
        str(issue.get("code"))
        for issue in value.get("issues", [])
        if isinstance(issue, dict) and issue.get("code")
    ]
    if issue_codes:
        lines.append(f"{prefix}SL_LINT issues: `{', '.join(issue_codes)}`")
    return lines


def _attached_stop_loss_cap_jpy(intent: OrderIntent, policy_max_loss_jpy: float | None) -> float | None:
    """Return the effective cap for broker-attached stop risk.

    RiskEngine validates `intent.sl` before the gateway builds an OANDA order.
    In SL-free runtime the gateway may attach a farther catastrophe stop, so
    the fillable broker order needs one more cap check against the same
    machine-readable loss budget. This is not a new sizing policy; it prevents
    the executable `stopLossOnFill` from exceeding the cap that made the intent
    acceptable.
    """

    metadata = intent.metadata or {}
    caps = [
        value
        for value in (
            _positive_float(metadata.get("max_loss_jpy")),
            policy_max_loss_jpy
            if policy_max_loss_jpy is not None and policy_max_loss_jpy > 0
            else None,
        )
        if value is not None
    ]
    # A generated receipt may carry the cap that was current earlier in the
    # cycle.  Fresh pre-POST reconciliation supplies the current policy cap;
    # metadata may tighten that value but must never override it upward.
    cap = min(caps) if caps else None

    asymmetry_cap = _loss_asymmetry_cap_from_metadata(metadata)
    if asymmetry_cap is not None:
        cap = asymmetry_cap if cap is None else min(cap, asymmetry_cap)
    return cap


def _attached_stop_effective_loss_cap_jpy(
    *,
    intent: OrderIntent,
    policy_max_loss_jpy: float | None,
    snapshot: BrokerSnapshot,
    portfolio_loss_cap: float | None,
    cumulative_risk_jpy: float,
) -> float | None:
    cap = _attached_stop_loss_cap_jpy(intent, policy_max_loss_jpy)
    portfolio_remaining = _portfolio_loss_remaining_jpy(
        snapshot=snapshot,
        portfolio_loss_cap=portfolio_loss_cap,
        cumulative_risk_jpy=cumulative_risk_jpy,
    )
    if portfolio_remaining is not None:
        cap = portfolio_remaining if cap is None else min(cap, portfolio_remaining)
    return cap


def _portfolio_loss_remaining_jpy(
    *,
    snapshot: BrokerSnapshot,
    portfolio_loss_cap: float | None,
    cumulative_risk_jpy: float,
) -> float | None:
    if portfolio_loss_cap is None:
        return None
    pending_risk, _, pending_issue = _pending_risk_margin_jpy(snapshot)
    open_risk, open_issue = _open_trader_position_risk_jpy(snapshot)
    if pending_issue is not None or open_issue is not None:
        return None
    return max(0.0, portfolio_loss_cap - open_risk - pending_risk - max(0.0, cumulative_risk_jpy))


def _loss_asymmetry_cap_from_metadata(metadata: dict[str, Any]) -> float | None:
    mode = str(metadata.get("loss_asymmetry_guard_mode") or "").upper()
    if mode == "TP_PROVEN_RELAXED":
        return None
    status = str(metadata.get("capture_economics_status") or "").upper()
    active = str(metadata.get("loss_asymmetry_guard_active") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    avg_win = _positive_float(metadata.get("capture_avg_win_jpy"))
    avg_loss = _positive_float(metadata.get("capture_avg_loss_jpy"))
    if not active and not (status == "NEGATIVE_EXPECTANCY" and avg_win is not None and avg_loss is not None and avg_loss > avg_win):
        return None
    explicit_cap = _positive_float(metadata.get("loss_asymmetry_guard_loss_cap_jpy"))
    if explicit_cap is not None:
        return explicit_cap
    return avg_win


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _attach_take_profit_on_fill(intent: OrderIntent) -> bool:
    if os.environ.get("QR_DISABLE_AUTO_TP", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        return False
    raw = (intent.metadata or {}).get("attach_take_profit_on_fill")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return True


def _raw_dependent_price(raw: dict[str, Any], key: str) -> float | None:
    payload = raw.get(key) if isinstance(raw, dict) else None
    if not isinstance(payload, dict):
        return None
    value = payload.get("price")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quote_to_jpy(pair: str, snapshot: BrokerSnapshot) -> float | None:
    quote_ccy = pair.split("_", 1)[1]
    if quote_ccy == "JPY":
        return 1.0
    home_conversion = snapshot.home_conversions.get(quote_ccy)
    if home_conversion is not None and home_conversion > 0:
        return float(home_conversion)
    conversion_quote = snapshot.quotes.get(f"{quote_ccy}_JPY")
    if conversion_quote is not None:
        return max(conversion_quote.bid, conversion_quote.ask)
    return None


def _price_key(pair: str, value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), _price_precision(pair))


def _repriced_crossed_passive_limit_intent(
    *,
    intent: OrderIntent,
    snapshot: BrokerSnapshot,
    risk,
) -> tuple[OrderIntent, RiskIssue | None]:
    if intent.order_type != OrderType.LIMIT or intent.entry is None:
        return intent, None
    issue_codes = {issue.code for issue in getattr(risk, "issues", ())}
    quote = snapshot.quotes.get(intent.pair)
    if quote is None:
        return intent, None

    precision = _price_precision(intent.pair)
    tick = _price_tick(intent.pair)
    original_entry = float(intent.entry)
    candidate: float | None = None
    message: str | None = None

    if (
        intent.side == Side.LONG
        and "LIMIT_ENTRY_NOT_BELOW_MARKET" in issue_codes
        and original_entry >= quote.ask
    ):
        if predictive_scout_intent_claimed(intent):
            return intent, RiskIssue(
                "PREDICTIVE_SCOUT_TRIGGER_CROSSED",
                (
                    f"predictive SCOUT LONG trigger {original_entry:.{precision}f} is no longer passive "
                    f"at ask={quote.ask:.{precision}f}; expire the exact replay geometry instead of repricing it"
                ),
            )
        candidate = float(_price(intent.pair, quote.ask - tick))
        if candidate <= 0.0 or candidate >= quote.ask or candidate >= original_entry:
            return intent, None
        message = (
            f"LONG limit repriced passively from {original_entry:.{precision}f} to "
            f"{candidate:.{precision}f} below ask={quote.ask:.{precision}f}; no MARKET conversion"
        )
    elif (
        intent.side == Side.SHORT
        and "LIMIT_ENTRY_NOT_ABOVE_MARKET" in issue_codes
        and original_entry <= quote.bid
    ):
        if predictive_scout_intent_claimed(intent):
            return intent, RiskIssue(
                "PREDICTIVE_SCOUT_TRIGGER_CROSSED",
                (
                    f"predictive SCOUT SHORT trigger {original_entry:.{precision}f} is no longer passive "
                    f"at bid={quote.bid:.{precision}f}; expire the exact replay geometry instead of repricing it"
                ),
            )
        candidate = float(_price(intent.pair, quote.bid + tick))
        if candidate <= quote.bid or candidate <= original_entry:
            return intent, None
        message = (
            f"SHORT limit repriced passively from {original_entry:.{precision}f} to "
            f"{candidate:.{precision}f} above bid={quote.bid:.{precision}f}; no MARKET conversion"
        )
    if candidate is None or message is None:
        return intent, None

    metadata = dict(intent.metadata or {})
    metadata.update(
        {
            "gateway_passive_limit_repriced": True,
            "gateway_passive_limit_original_entry": round(original_entry, precision),
            "gateway_passive_limit_repriced_entry": round(candidate, precision),
            "gateway_passive_limit_quote_bid": round(float(quote.bid), precision),
            "gateway_passive_limit_quote_ask": round(float(quote.ask), precision),
        }
    )
    return (
        replace(intent, entry=candidate, metadata=metadata),
        RiskIssue("LIMIT_ENTRY_REPRICED_PASSIVE", message, "WARN"),
    )


def _send_guard_issues(*, send: bool, confirm_live: bool, lane_id: str | None) -> list[dict[str, str]]:
    issues: list[RiskIssue] = []
    if send and not confirm_live:
        issues.append(RiskIssue("LIVE_CONFIRMATION_REQUIRED", "live send requires --confirm-live"))
    if send and not lane_id:
        issues.append(RiskIssue("LANE_ID_REQUIRED_FOR_SEND", "live send requires an explicit --lane-id"))
    if send and _truthy_env("QR_REQUIRE_POSITION_GUARDIAN_ACTIVE", default=False) and not _position_guardian_active():
        issues.append(
            RiskIssue(
                "POSITION_GUARDIAN_INACTIVE_FOR_SEND",
                "fresh entry live send requires an active position guardian with a recent heartbeat so "
                "TP-progress profit can be captured between full trader cycles; load "
                "com.quantrabbit.position-guardian, repair stale guardian execution, or set "
                "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE=0 for an explicit operator override",
            )
        )
    return [issue.__dict__ for issue in issues]


def _gpt_verified_decision_live_send_issues(
    verified_decision_path: Path | None,
    *,
    selected_lane_id: str | None,
    intents_payload: dict[str, Any],
    send: bool,
    require_receipt: bool = False,
    intent: OrderIntent | None = None,
) -> list[dict[str, str]]:
    if not send:
        return []
    if verified_decision_path is None:
        if not require_receipt:
            return []
        return [
            RiskIssue(
                "GPT_FRESH_TRADE_RECEIPT_REQUIRED_FOR_LIVE_SEND",
                "predictive SCOUT live send requires a fresh verified ACCEPTED TRADE receipt; "
                "direct gateway invocation without AI trader verification is forbidden",
            ).__dict__
        ]
    if not verified_decision_path.exists():
        return [
            RiskIssue(
                "GPT_FRESH_TRADE_RECEIPT_REQUIRED_FOR_LIVE_SEND",
                f"live fresh entry send requires a verified ACCEPTED TRADE receipt: {verified_decision_path}",
            ).__dict__
        ]
    try:
        payload = json.loads(verified_decision_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [
            RiskIssue(
                "GPT_VERIFIED_RECEIPT_UNREADABLE_FOR_LIVE_SEND",
                f"verified GPT decision receipt is unreadable before live send: {verified_decision_path}: {exc}",
            ).__dict__
        ]

    issues: list[RiskIssue] = []
    status = str(payload.get("status") or "").upper()
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    action = str(decision.get("action") or "").upper()
    if status != "ACCEPTED":
        issues.append(
            RiskIssue(
                "GPT_VERIFIED_RECEIPT_NOT_ACCEPTED_FOR_LIVE_SEND",
                f"verified GPT receipt status is {status or 'missing'}; fresh entry send requires ACCEPTED TRADE.",
            )
        )
    allowed_actions = {"TRADE"} if require_receipt else {"TRADE", "ADD"}
    if action not in allowed_actions:
        issues.append(
            RiskIssue(
                "GPT_FRESH_TRADE_RECEIPT_REQUIRED_FOR_LIVE_SEND",
                f"verified GPT receipt action is {action or 'missing'}; "
                + (
                    "predictive SCOUT requires an exact TRADE receipt (ADD is not a forward experiment authorization)."
                    if require_receipt
                    else "WAIT/REQUEST_EVIDENCE/non-TRADE cannot send fresh risk."
                ),
            )
        )

    selected_receipt_lanes = _gpt_receipt_selected_lane_ids(decision)
    if action in allowed_actions and not selected_receipt_lanes:
        issues.append(
            RiskIssue(
                "GPT_SELECTED_LANE_MISSING_FOR_LIVE_SEND",
                "verified GPT TRADE/ADD receipt must name selected_lane_id or selected_lane_ids.",
            )
        )
    if selected_lane_id and selected_receipt_lanes and selected_lane_id not in selected_receipt_lanes:
        issues.append(
            RiskIssue(
                "GPT_SELECTED_LANE_MISMATCH_FOR_LIVE_SEND",
                f"gateway selected lane {selected_lane_id} is not in GPT receipt lanes: "
                f"{', '.join(selected_receipt_lanes)}",
            )
        )

    market_read = decision.get("market_read_first")
    if action in allowed_actions and not isinstance(market_read, dict):
        issues.append(
            RiskIssue(
                "GPT_MARKET_READ_FIRST_REQUIRED_FOR_LIVE_SEND",
                "fresh entry send requires current market_read_first on the verified GPT receipt.",
            )
        )
    if require_receipt and intent is not None and isinstance(market_read, dict):
        if not _market_read_supports_intent(market_read, intent):
            issues.append(
                RiskIssue(
                    "GPT_SCOUT_MARKET_READ_MISMATCH_FOR_LIVE_SEND",
                    f"verified predictive SCOUT receipt must contain a current {intent.pair} "
                    f"{intent.side.value} market-read prediction",
                )
            )
        issues.extend(
            RiskIssue(code, message)
            for code, message in _predictive_scout_verified_packet_mismatches(
                payload,
                selected_lane_id=selected_lane_id,
                intent=intent,
            )
        )

    blocking_verification_codes = [
        str(item.get("code") or "").strip()
        for item in payload.get("verification_issues", []) or []
        if isinstance(item, dict) and str(item.get("severity") or "BLOCK").upper() == "BLOCK"
    ]
    blocking_verification_codes = [code for code in blocking_verification_codes if code]
    if blocking_verification_codes:
        issues.append(
            RiskIssue(
                "GPT_VERIFIED_RECEIPT_HAS_BLOCKING_ISSUES",
                "verified GPT receipt still has BLOCK issue(s): "
                + ", ".join(blocking_verification_codes[:8]),
            )
        )

    receipt_generated_at = payload.get("generated_at_utc") or decision.get("generated_at_utc")
    receipt_ts = _parse_utc_timestamp(receipt_generated_at)
    intents_ts = _parse_utc_timestamp(intents_payload.get("generated_at_utc"))
    if receipt_ts is None:
        issues.append(
            RiskIssue(
                "GPT_VERIFIED_RECEIPT_TIMESTAMP_REQUIRED_FOR_LIVE_SEND",
                "verified GPT receipt must include generated_at_utc so gateway can prove it is fresh.",
            )
        )
    elif intents_ts is not None and receipt_ts < intents_ts:
        issues.append(
            RiskIssue(
                "GPT_RECEIPT_STALE_FOR_ORDER_INTENTS",
                "verified GPT receipt predates the order_intents packet; rerun gpt-trader-decision for "
                "current intents before sending fresh risk.",
            )
        )
    if require_receipt and intent is not None:
        now = datetime.now(timezone.utc)
        scout_generated = _parse_utc_timestamp(
            (intent.metadata or {}).get("predictive_scout_generated_at_utc")
        )
        try:
            scout_ttl_minutes = float(
                (intent.metadata or {}).get("predictive_scout_ttl_minutes")
            )
        except (TypeError, ValueError):
            scout_ttl_minutes = 0.0
        if intents_ts is None:
            issues.append(
                RiskIssue(
                    "GPT_SCOUT_INTENTS_TIMESTAMP_REQUIRED_FOR_LIVE_SEND",
                    "predictive SCOUT requires a timestamped order_intents packet so the AI receipt "
                    "cannot be replayed against an unversioned signal",
                )
            )
        if receipt_ts is not None:
            if receipt_ts > now + timedelta(seconds=PREDICTIVE_SCOUT_CLOCK_SKEW_SECONDS):
                issues.append(
                    RiskIssue(
                        "GPT_SCOUT_RECEIPT_FROM_FUTURE",
                        "predictive SCOUT AI receipt timestamp exceeds the allowed clock-skew window",
                    )
                )
            if (
                scout_generated is not None
                and receipt_ts
                < scout_generated - timedelta(seconds=PREDICTIVE_SCOUT_CLOCK_SKEW_SECONDS)
            ):
                issues.append(
                    RiskIssue(
                        "GPT_SCOUT_RECEIPT_PREDATES_SIGNAL",
                        "predictive SCOUT AI receipt predates the current forecast-cycle signal",
                    )
                )
            if scout_ttl_minutes <= 0.0 or (
                now - receipt_ts
            ).total_seconds() > scout_ttl_minutes * 60.0:
                issues.append(
                    RiskIssue(
                        "GPT_SCOUT_RECEIPT_STALE_FOR_SIGNAL",
                        "predictive SCOUT AI receipt is older than the signal's bounded TTL",
                    )
                )

    return [issue.__dict__ for issue in issues]


def _market_read_supports_intent(
    market_read: dict[str, Any],
    intent: OrderIntent,
) -> bool:
    expected_pair = intent.pair.replace("_", "").upper()
    expected_side = intent.side.value
    for key in ("next_30m_prediction", "next_2h_prediction", "best_trade_if_forced"):
        prediction = market_read.get(key)
        if not isinstance(prediction, dict):
            continue
        pair = str(prediction.get("pair") or "").replace("_", "").upper()
        direction = str(prediction.get("direction") or prediction.get("side") or "").upper()
        side = (
            "LONG"
            if direction in {"LONG", "UP", "BUY", "BULLISH"}
            else "SHORT"
            if direction in {"SHORT", "DOWN", "SELL", "BEARISH"}
            else ""
        )
        if pair == expected_pair and side == expected_side:
            return True
    return False


def _predictive_scout_verified_packet_mismatches(
    payload: dict[str, Any],
    *,
    selected_lane_id: str | None,
    intent: OrderIntent,
) -> list[tuple[str, str]]:
    packet = payload.get("input_packet") if isinstance(payload.get("input_packet"), dict) else {}
    lanes = packet.get("lanes") if isinstance(packet.get("lanes"), list) else []
    lane = next(
        (
            item
            for item in lanes
            if isinstance(item, dict)
            and str(item.get("lane_id") or "") == str(selected_lane_id or "")
        ),
        None,
    )
    if not isinstance(lane, dict):
        return [
            (
                "GPT_SCOUT_VERIFIED_PACKET_LANE_MISSING",
                "verified AI receipt must preserve the exact predictive SCOUT lane from its input packet",
            )
        ]
    scout = lane.get("predictive_scout") if isinstance(lane.get("predictive_scout"), dict) else {}
    metadata = intent.metadata or {}
    expected = {
        "pair": intent.pair,
        "direction": intent.side.value,
        "order_type": intent.order_type.value,
        "units": abs(int(intent.units)),
        "entry": intent.entry,
        "tp": intent.tp,
        "sl": intent.sl,
        "forecast_cycle_id": str(metadata.get("forecast_cycle_id") or ""),
        "rule_digest": str(metadata.get("predictive_scout_rule_digest") or ""),
        "risk_tier": str(metadata.get("predictive_scout_risk_tier") or "").upper(),
        "planned_initial_risk_jpy": _optional_float(
            metadata.get("predictive_scout_planned_initial_risk_jpy")
        ),
        "sizing_digest": str(
            metadata.get("predictive_scout_sizing_digest") or ""
        ),
        "generated_at_utc": str(metadata.get("predictive_scout_generated_at_utc") or ""),
        "expires_at_utc": str(metadata.get("predictive_scout_expires_at_utc") or ""),
    }
    actual = {
        "pair": str(lane.get("pair") or ""),
        "direction": str(lane.get("direction") or "").upper(),
        "order_type": str(lane.get("order_type") or "").upper(),
        "units": abs(_optional_int(lane.get("units")) or 0),
        "entry": _optional_float(lane.get("entry")),
        "tp": _optional_float(lane.get("tp")),
        "sl": _optional_float(lane.get("sl")),
        "forecast_cycle_id": str(scout.get("forecast_cycle_id") or ""),
        "rule_digest": str(scout.get("rule_digest") or ""),
        "risk_tier": str(scout.get("risk_tier") or "").upper(),
        "planned_initial_risk_jpy": _optional_float(
            scout.get("planned_initial_risk_jpy")
        ),
        "sizing_digest": str(scout.get("sizing_digest") or ""),
        "generated_at_utc": str(scout.get("generated_at_utc") or ""),
        "expires_at_utc": str(scout.get("expires_at_utc") or ""),
    }
    if actual != expected:
        return [
            (
                "GPT_SCOUT_SIGNAL_MISMATCH_FOR_LIVE_SEND",
                "verified AI packet does not match the current SCOUT forecast cycle, authenticated rule, or exact order geometry",
            )
        ]
    return []


def _gpt_receipt_selected_lane_ids(decision: dict[str, Any]) -> tuple[str, ...]:
    lane_ids: list[str] = []
    selected_lane_ids = decision.get("selected_lane_ids")
    if isinstance(selected_lane_ids, list):
        lane_ids.extend(str(lane_id).strip() for lane_id in selected_lane_ids if str(lane_id).strip())
    primary = str(decision.get("selected_lane_id") or "").strip()
    if primary:
        lane_ids.append(primary)
    return tuple(dict.fromkeys(lane_ids))


def _target_path_live_send_issues(intent: OrderIntent, *, send: bool) -> list[dict[str, str]]:
    metadata = dict(intent.metadata or {})
    if not send or not _target_path_contract_present(metadata):
        return []

    issues: list[RiskIssue] = []
    grade = _target_path_grade(metadata)
    rank = TARGET_PATH_GRADE_RANK.get(grade)
    role = _target_path_role(metadata)
    slot = _target_path_attack_stack_slot(metadata)
    remaining_5 = _target_path_remaining_to_5pct(metadata)
    progress_pct = _target_path_progress_pct(metadata)
    base_reached = (
        (remaining_5 is not None and remaining_5 <= 0)
        or (_metadata_float(metadata, "minimum_progress_pct") or 0.0) >= 100.0
        or (progress_pct is not None and progress_pct >= 5.0)
    )

    if not _truthy_env("QR_TARGET_PATH_LIVE_ENABLED", default=False):
        issues.append(
            RiskIssue(
                "TARGET_PATH_LIVE_DISABLED",
                "target-path live send requires QR_TARGET_PATH_LIVE_ENABLED=1; dry-run/stage remains available.",
            )
        )
    if str(metadata.get("valid_as_target_path") or "").strip().upper() != "YES":
        issues.append(
            RiskIssue(
                "TARGET_PATH_SIZING_NOT_VALID",
                "target-path live send requires sizing metadata valid_as_target_path=YES.",
            )
        )
    if not str(metadata.get("daily_target_mode") or metadata.get("target_mode") or "").strip():
        issues.append(
            RiskIssue("TARGET_PATH_DAILY_MODE_MISSING", "target-path live receipt requires daily target mode.")
        )
    if remaining_5 is None:
        issues.append(
            RiskIssue("TARGET_PATH_REMAINING_TO_5PCT_MISSING", "target-path live receipt requires remaining_to_5pct.")
        )
    if rank is None:
        issues.append(RiskIssue("TARGET_PATH_GRADE_MISSING", "target-path live send requires grade S/A or B+ support."))
    elif rank <= TARGET_PATH_GRADE_RANK["B0"]:
        issues.append(RiskIssue("TARGET_PATH_GRADE_TOO_LOW", "B0/B-/C target-path risk is never live."))
    elif grade == "B+" and not _b_plus_target_path_support(role, slot):
        issues.append(
            RiskIssue(
                "B_PLUS_NOT_SUPPORT_TARGET_PATH",
                "B+ target-path live risk is allowed only as support/reload/second-shot, not NOW/main.",
            )
        )
    elif rank < TARGET_PATH_GRADE_RANK["A"] and grade != "B+":
        issues.append(RiskIssue("TARGET_PATH_GRADE_TOO_LOW", "target-path live send requires S/A or B+ support."))
    if base_reached and not _target_path_extension_gate_yes(metadata) and grade.startswith("B"):
        issues.append(
            RiskIssue(
                "BASE_TARGET_REACHED_B_RISK_BLOCKED",
                "+5% is reached and the 10% Extension Gate is NO; fresh B risk is blocked.",
            )
        )
    if not _target_path_board_mapped(metadata, role):
        issues.append(
            RiskIssue("TARGET_PATH_BOARD_MAPPING_MISSING", "target-path live send requires 5% PACE BOARD mapping.")
        )
    if not _target_path_attack_stack_mapped(metadata, slot):
        issues.append(
            RiskIssue(
                "PATH_ATTACK_STACK_MAPPING_MISSING",
                "target-path live send requires ATTACK STACK mapping with slot NOW / RELOAD / SECOND_SHOT.",
            )
        )
    if _metadata_truthy(metadata.get("same_thesis_lost_recently")) and _metadata_truthy(
        metadata.get("vehicle_unchanged_after_loss")
    ):
        issues.append(
            RiskIssue(
                "SAME_THESIS_LOST_RECENTLY",
                "same thesis lost recently and vehicle is unchanged; LIVE-LEARNING blocks repeat vehicle risk.",
            )
        )

    for code, names, label in (
        ("EXACT_PRETRADE_BLOCKED", ("exact_pretrade_passed", "exact_pretrade_ok"), "exact pretrade"),
        ("SPREAD_GUARD_BLOCKED", ("spread_guard_passed", "spread_ok"), "spread guard"),
        ("PRICING_PROBE_BLOCKED", ("pricing_probe_passed", "pricing_probe_ok"), "pricing probe"),
        ("FILL_GUARD_BLOCKED", ("fill_guard_passed", "fill_guard_ok"), "fill guard"),
    ):
        passed = _metadata_bool(metadata, *names)
        if passed is not True:
            issues.append(RiskIssue(code, f"target-path live send requires {label} proof."))

    suggested_units = _metadata_int(metadata, "suggested_units")
    if (
        suggested_units is not None
        and suggested_units > 0
        and abs(int(intent.units)) < max(1, int(suggested_units * 0.5))
    ):
        issues.append(
            RiskIssue(
                "TARGET_PATH_UNITS_UNDER_SIZED",
                "target-path live send final units are less than half of dry-run suggested units.",
            )
        )
    for key in ("suggested_units", "risk_yen", "risk_pct", "target_yen", "contribution_to_5pct"):
        if _metadata_float(metadata, key) is None:
            issues.append(
                RiskIssue(
                    "TARGET_PATH_RECEIPT_FIELD_MISSING",
                    f"target-path live receipt requires sizing metadata `{key}`.",
                )
            )
    for key in ("suggested_units", "risk_yen", "target_yen", "contribution_to_5pct"):
        value = _metadata_float(metadata, key)
        if value is not None and value <= 0:
            issues.append(
                RiskIssue(
                    "TARGET_PATH_RECEIPT_FIELD_INVALID",
                    f"target-path live receipt requires positive sizing metadata `{key}`.",
                )
            )
    return [issue.__dict__ for issue in issues]


def _target_path_contract_present(metadata: dict[str, Any]) -> bool:
    keys = {
        "daily_target_mode",
        "target_mode",
        "remaining_to_5pct",
        "remaining_to_5pct_yen",
        "remaining_minimum_jpy",
        "target_path_role",
        "path_role",
        "valid_as_target_path",
        "path_board_available",
        "five_pct_path_available",
        "attack_stack_available",
        "maps_to_attack_stack",
        "path_board_slot",
        "attack_stack_slot",
        "target_path_live_mode",
    }
    return any(key in metadata for key in keys)


def _target_path_receipt_from_intent(
    intent: OrderIntent,
    *,
    risk_metrics: dict[str, Any] | None,
    order_request: dict[str, Any] | None,
    requested_units: int | None,
    final_units: int | None,
    sent: bool,
) -> dict[str, Any] | None:
    metadata = dict(intent.metadata or {})
    if not _target_path_contract_present(metadata):
        return None
    order_request = order_request or {}
    client_extensions = (
        order_request.get("clientExtensions") if isinstance(order_request.get("clientExtensions"), dict) else {}
    )
    risk_metrics = risk_metrics or {}
    return {
        "daily_target_mode": str(metadata.get("daily_target_mode") or metadata.get("target_mode") or "").strip(),
        "remaining_to_5pct": _metadata_float(
            metadata,
            "remaining_to_5pct_yen",
            "remaining_to_5pct",
            "remaining_minimum_jpy",
        ),
        "rolling_30d_policy": str(metadata.get("rolling_30d_policy") or "").strip() or None,
        "current_equity_raw": _metadata_float(metadata, "current_equity_raw"),
        "capital_flows_30d": _metadata_float(metadata, "capital_flows_30d"),
        "funding_adjusted_equity": _metadata_float(metadata, "funding_adjusted_equity"),
        "rolling_30d_multiplier_raw": _metadata_float(metadata, "rolling_30d_multiplier_raw"),
        "rolling_30d_multiplier_funding_adjusted": _metadata_float(
            metadata,
            "rolling_30d_multiplier_funding_adjusted",
        ),
        "current_30d_multiplier": _metadata_float(metadata, "current_30d_multiplier"),
        "remaining_to_4x_raw": _metadata_float(metadata, "remaining_to_4x_raw"),
        "remaining_to_4x_funding_adjusted": _metadata_float(metadata, "remaining_to_4x_funding_adjusted"),
        "remaining_to_4x": _metadata_float(metadata, "remaining_to_4x"),
        "required_calendar_daily_return_raw": _metadata_float(metadata, "required_calendar_daily_return_raw"),
        "required_active_day_return_raw": _metadata_float(metadata, "required_active_day_return_raw"),
        "required_calendar_daily_return_funding_adjusted": _metadata_float(
            metadata,
            "required_calendar_daily_return_funding_adjusted",
        ),
        "required_active_day_return_funding_adjusted": _metadata_float(
            metadata,
            "required_active_day_return_funding_adjusted",
        ),
        "required_calendar_daily_return": _metadata_float(metadata, "required_calendar_daily_return"),
        "required_active_day_return": _metadata_float(metadata, "required_active_day_return"),
        "performance_basis": str(metadata.get("performance_basis") or "").strip() or None,
        "sizing_basis": str(metadata.get("sizing_basis") or "").strip() or None,
        "pace_state": str(metadata.get("pace_state") or "").strip() or None,
        "five_pct_path_role": _target_path_role(metadata),
        "attack_stack_slot": _target_path_attack_stack_slot(metadata),
        "grade": _target_path_grade(metadata),
        "suggested_units": _metadata_int(metadata, "suggested_units") or requested_units,
        "final_units": final_units,
        "risk_yen": _metadata_float(metadata, "risk_yen") or _metadata_float(risk_metrics, "risk_jpy"),
        "risk_pct": _metadata_float(metadata, "risk_pct"),
        "target_yen": _metadata_float(metadata, "target_yen") or _metadata_float(risk_metrics, "reward_jpy"),
        "contribution_to_5pct": _metadata_float(metadata, "contribution_to_5pct"),
        "live_order_gateway_receipt_id": str(client_extensions.get("id") or "").strip() or None,
        "live_order_sent": bool(sent),
        "target_path_live_enabled": _truthy_env("QR_TARGET_PATH_LIVE_ENABLED", default=False),
        "target_path_live_mode": str(metadata.get("target_path_live_mode") or "LIVE_LEARNING").strip().upper(),
    }


def _target_path_receipt_report_lines(value: Any, *, prefix: str) -> list[str]:
    if not isinstance(value, dict) or not value:
        return []
    return [
        (
            f"{prefix}target-path receipt: mode=`{value.get('daily_target_mode')}` "
            f"remaining_to_5pct=`{_fmt_jpy(value.get('remaining_to_5pct'))}` "
            f"remaining_to_4x_raw=`{_fmt_jpy(value.get('remaining_to_4x_raw'))}` "
            f"remaining_to_4x_funding_adjusted=`{_fmt_jpy(value.get('remaining_to_4x_funding_adjusted') or value.get('remaining_to_4x'))}` "
            f"multiplier_raw=`{value.get('rolling_30d_multiplier_raw')}` "
            f"multiplier_funding_adjusted=`{value.get('rolling_30d_multiplier_funding_adjusted') or value.get('current_30d_multiplier')}` "
            f"required_calendar_funding_adjusted=`{value.get('required_calendar_daily_return_funding_adjusted') or value.get('required_calendar_daily_return')}` "
            f"required_active_funding_adjusted=`{value.get('required_active_day_return_funding_adjusted') or value.get('required_active_day_return')}` "
            f"performance_basis=`{value.get('performance_basis')}` sizing_basis=`{value.get('sizing_basis')}` "
            f"pace=`{value.get('pace_state')}` "
            f"role=`{value.get('five_pct_path_role')}` slot=`{value.get('attack_stack_slot')}` "
            f"grade=`{value.get('grade')}` units=`{value.get('suggested_units')}->{value.get('final_units')}` "
            f"risk=`{_fmt_jpy(value.get('risk_yen'))}` risk_pct=`{value.get('risk_pct')}` "
            f"target=`{_fmt_jpy(value.get('target_yen'))}` "
            f"contribution_to_5pct=`{_fmt_jpy(value.get('contribution_to_5pct'))}` "
            f"receipt_id=`{value.get('live_order_gateway_receipt_id')}` live_order_sent=`{value.get('live_order_sent')}`"
        )
    ]


def _target_path_grade(metadata: dict[str, Any]) -> str:
    raw = metadata.get("conviction_grade") or metadata.get("grade") or metadata.get("allocation_band") or ""
    grade = str(raw).strip().upper().replace("_", "").replace(" ", "")
    return "B0" if grade == "B" else grade


def _target_path_role(metadata: dict[str, Any]) -> str:
    return str(metadata.get("target_path_role") or metadata.get("path_role") or "").strip().upper()


def _target_path_attack_stack_slot(metadata: dict[str, Any]) -> str:
    return str(metadata.get("attack_stack_slot") or metadata.get("campaign_role") or "").strip().upper()


def _b_plus_target_path_support(role: str, slot: str) -> bool:
    return role in TARGET_PATH_SUPPORT_ROLES or slot in {"RELOAD", "SECOND_SHOT"}


def _target_path_board_mapped(metadata: dict[str, Any], role: str) -> bool:
    board_available = _metadata_truthy(metadata.get("path_board_available")) or _metadata_truthy(
        metadata.get("five_pct_path_available")
    )
    return board_available and (
        role in TARGET_PATH_MAIN_ROLES
        or role in TARGET_PATH_SUPPORT_ROLES
        or bool(str(metadata.get("path_board_slot") or "").strip())
    )


def _target_path_attack_stack_mapped(metadata: dict[str, Any], slot: str) -> bool:
    return (
        _metadata_truthy(metadata.get("attack_stack_available"))
        and _metadata_truthy(metadata.get("maps_to_attack_stack"))
        and slot in TARGET_PATH_ATTACK_STACK_SLOTS
    )


def _target_path_remaining_to_5pct(metadata: dict[str, Any]) -> float | None:
    return _metadata_float(metadata, "remaining_to_5pct_yen", "remaining_to_5pct", "remaining_minimum_jpy")


def _target_path_progress_pct(metadata: dict[str, Any]) -> float | None:
    return _metadata_float(metadata, "daily_progress_pct", "day_progress_pct", "total_day_progress_pct")


def _target_path_extension_gate_yes(metadata: dict[str, Any]) -> bool:
    return _metadata_bool(metadata, "ten_pct_extension_gate", "extension_gate_10pct", "extension_gate") is True


def _metadata_bool(metadata: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _metadata_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _metadata_float(metadata: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in metadata:
            continue
        try:
            return float(metadata.get(key))
        except (TypeError, ValueError):
            return None
    return None


def _metadata_int(metadata: dict[str, Any], *keys: str) -> int | None:
    value = _metadata_float(metadata, *keys)
    return int(value) if value is not None else None


def _truthy_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _truthy_value(raw)


def _truthy_value(raw: str) -> bool:
    return raw.strip() in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def _position_guardian_active() -> bool:
    raw = os.environ.get("QR_POSITION_GUARDIAN_ACTIVE")
    if raw is not None:
        return _truthy_value(raw)

    label = os.environ.get("QR_POSITION_GUARDIAN_LABEL", "com.quantrabbit.position-guardian")
    plist = Path(
        os.environ.get(
            "QR_POSITION_GUARDIAN_PLIST",
            str(Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"),
        )
    )
    if not plist.exists():
        return False
    try:
        list_proc = subprocess.run(
            ["launchctl", "list", label],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return False
    if list_proc.returncode == 0:
        return _position_guardian_heartbeat_fresh()
    try:
        print_proc = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return False
    return print_proc.returncode == 0 and _position_guardian_heartbeat_fresh()


def _position_guardian_heartbeat_fresh() -> bool:
    raw_required = os.environ.get("QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT", "1")
    if not _truthy_value(raw_required):
        return True
    try:
        interval = int(float(os.environ.get("QR_POSITION_GUARDIAN_INTERVAL", "30")))
    except (TypeError, ValueError):
        interval = 30
    interval = max(15, interval)
    try:
        max_age = int(float(os.environ.get("QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS", str(interval * 4))))
    except (TypeError, ValueError):
        max_age = interval * 4
    max_age = max(interval, max_age)
    paths = (
        _env_path("QR_POSITION_GUARDIAN_EXECUTION", _QR_ROOT / "data" / "position_guardian_execution.json"),
        _env_path("QR_POSITION_GUARDIAN_HEARTBEAT", _QR_ROOT / "data" / "position_guardian.json"),
    )
    now = datetime.now(timezone.utc)
    for path in paths:
        if not path.exists():
            continue
        generated = None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            payload = {}
        if isinstance(payload, dict):
            generated = _parse_guardian_heartbeat_time(payload.get("generated_at_utc"))
        if generated is not None:
            age = (now - generated).total_seconds()
        else:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                continue
        if -60.0 <= age <= max_age:
            return True
    return False


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    path = Path(raw)
    if path.is_absolute():
        return path
    return _QR_ROOT / path


def _parse_guardian_heartbeat_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _risk_has_blocking_stale_quote(risk: Any) -> bool:
    for issue in getattr(risk, "issues", ()) or ():
        if str(getattr(issue, "severity", "")).upper() != "BLOCK":
            continue
        if str(getattr(issue, "code", "")).upper() in {"STALE_QUOTE", "STALE_CONVERSION_QUOTE"}:
            return True
    return False


def _gateway_stale_quote_retry_attempts() -> int:
    return _bounded_env_int("QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS", default=3, minimum=0, maximum=10)


def _gateway_stale_quote_retry_sleep_seconds() -> float:
    return _bounded_env_float("QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS", default=5.0, minimum=0.0, maximum=30.0)


def _bounded_env_int(name: str, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _bounded_env_float(name: str, *, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _intent_status_issues(selected: dict[str, Any]) -> list[dict[str, str]]:
    status = str(selected.get("status") or "")
    if status == "LIVE_READY":
        return []
    return [
        RiskIssue(
            "INTENT_NOT_LIVE_READY",
            f"stage-live-order requires a LIVE_READY receipt, got {status or 'missing'}",
        ).__dict__
    ]


def _projection_expiry_status_issues(
    *,
    selected: dict[str, Any],
    intents_path: Path,
    validation_time_utc: datetime,
) -> list[dict[str, str]]:
    if str(selected.get("status") or "") != "LIVE_READY":
        return []
    expired = _expired_pending_projection_count(
        data_root=intents_path.parent,
        validation_time_utc=validation_time_utc,
    )
    if expired <= 0:
        return []
    return [
        RiskIssue(
            "TELEMETRY_PROJECTION_PENDING_EXPIRED_FOR_LIVE",
            (
                f"projection_ledger.jsonl has {expired} expired PENDING projection(s); "
                "rerun verify-projections before staging or sending new live exposure."
            ),
        ).__dict__
    ]


def _self_improvement_gateway_issues(
    path: Path | None,
    *,
    verified_decision_path: Path | None = None,
    selected: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Block fresh entry staging when self-improvement has an active P0 gate."""
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [
            RiskIssue(
                "SELF_IMPROVEMENT_AUDIT_UNREADABLE",
                f"self-improvement audit is unreadable before live order staging: {path}: {exc}",
            ).__dict__
        ]
    verification_postdates_audit = _accepted_verification_postdates(
        verified_decision_path,
        audit_generated_at=payload.get("generated_at_utc"),
    )
    worst_segment = profitability_p0_worst_segment(payload)
    p0_repair_selected = _selected_intent_is_self_improvement_profitability_repair(
        selected,
        worst_segment=worst_segment,
    )
    selected_intent = selected.get("intent") if isinstance(selected, dict) else {}
    selected_metadata = (
        selected_intent.get("metadata") if isinstance(selected_intent, dict) else {}
    )
    predictive_scout_repair_selected = bool(
        p0_repair_selected
        and isinstance(selected_metadata, dict)
        and str(selected_metadata.get("self_improvement_p0_repair_mode") or "")
        == "PREDICTIVE_SCOUT_FORWARD_EVIDENCE"
    )
    forecast_repair_selected = forecast_adverse_path_exempted_by_tp_harvest_repair(selected)
    blockers: list[str] = []
    for item in payload.get("findings", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("priority") or "").upper() != "P0":
            continue
        code = str(item.get("code") or "SELF_IMPROVEMENT_P0")
        if p0_code_exempted_by_tp_harvest_repair(
            code,
            p0_repair_selected=p0_repair_selected,
        ):
            continue
        if (
            code == "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED"
            and p0_repair_selected
            and not predictive_scout_repair_selected
            and _verified_trade_covers_pending_cancel_review(
                verified_decision_path,
                finding=item,
                audit_generated_at=payload.get("generated_at_utc"),
            )
        ):
            continue
        if code == "LATEST_GPT_DECISION_STALE" and verification_postdates_audit:
            # The decision being staged was verified ACCEPTED after this audit
            # ran, so the audit's stale-decision verdict is about an older
            # receipt (mirrors gpt_trader._self_improvement_trade_blockers).
            # With asynchronous audit cadence and a slower decision cadence the
            # streak otherwise re-blocks the first staging attempt of every
            # fresh receipt. Manual stage-live-order paths do not pass
            # verified_decision_path and keep the strict streak gate.
            continue
        if _self_improvement_gateway_non_blocker(code, item):
            continue
        message = str(item.get("message") or code)
        blockers.append(f"{code}: {message}")
    forecast_blocker = forecast_adverse_path_new_risk_blocker(payload)
    if forecast_blocker is not None and not forecast_repair_selected:
        blockers.append(f"{forecast_blocker['code']}: {forecast_blocker['message']}")
    if not blockers:
        return []
    return [
        RiskIssue(
            "SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER",
            "self-improvement blocks new live entry risk: " + "; ".join(blockers[:3]),
        ).__dict__
    ]


def _verified_trade_covers_pending_cancel_review(
    verified_decision_path: Path | None,
    *,
    finding: dict[str, Any],
    audit_generated_at: Any,
) -> bool:
    required_ids = _pending_cancel_review_order_ids_from_finding(finding)
    if not required_ids or verified_decision_path is None or not verified_decision_path.exists():
        return False
    if audit_generated_at and not _accepted_verification_postdates(
        verified_decision_path,
        audit_generated_at=audit_generated_at,
    ):
        return False
    try:
        payload = json.loads(verified_decision_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if str(payload.get("status") or "").upper() != "ACCEPTED":
        return False
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    if str(decision.get("action") or "").upper() != "TRADE":
        return False
    cancel_ids = {
        str(order_id or "").strip()
        for order_id in decision.get("cancel_order_ids", []) or []
        if str(order_id or "").strip()
    }
    return required_ids <= cancel_ids


def _pending_cancel_review_order_ids_from_finding(finding: dict[str, Any]) -> set[str]:
    order_ids: set[str] = set()

    def add_many(values: Any) -> None:
        for value in values or []:
            text = str(value or "").strip()
            if text:
                order_ids.add(text)

    add_many(finding.get("cancel_review_order_ids"))
    evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
    add_many(evidence.get("cancel_review_order_ids"))
    for item in evidence.get("orders", []) or []:
        if isinstance(item, dict):
            add_many((item.get("order_id"),))
    for group in evidence.get("groups", []) or []:
        if isinstance(group, dict):
            add_many(group.get("order_ids"))
    return order_ids


def _selected_intent_is_self_improvement_profitability_repair(
    selected: dict[str, Any] | None,
    *,
    worst_segment: dict[str, str] | None = None,
) -> bool:
    if not isinstance(selected, dict):
        return False
    intent = selected.get("intent") if isinstance(selected.get("intent"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    if metadata.get("self_improvement_p0_repair_live_ready") is not True:
        return False
    mode = str(metadata.get("self_improvement_p0_repair_mode") or "")
    if mode == "PREDICTIVE_SCOUT_FORWARD_EVIDENCE":
        if not predictive_scout_metadata_supported(metadata):
            return False
        if str(intent.get("order_type") or "").strip().upper() != OrderType.LIMIT.value:
            return False
        try:
            units = abs(int(intent.get("units")))
        except (TypeError, ValueError):
            return False
        if units < MIN_PRODUCTION_LOT_UNITS:
            return False
        if intent_matches_profitability_worst_segment(intent, worst_segment):
            return False
        return True
    if mode != "TP_HARVEST_REPAIR":
        return False
    if not oanda_firepower_repair_current_risk_reaches_minimum(metadata):
        return False
    if intent_matches_profitability_worst_segment(intent, worst_segment):
        return False
    return True


def _accepted_verification_postdates(
    verified_decision_path: Path | None,
    *,
    audit_generated_at: Any,
) -> bool:
    if verified_decision_path is None or not verified_decision_path.exists():
        return False
    audit_ts = _parse_utc_timestamp(audit_generated_at)
    if audit_ts is None:
        return False
    try:
        payload = json.loads(verified_decision_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if str(payload.get("status") or "").upper() != "ACCEPTED":
        return False
    generated = payload.get("generated_at_utc")
    if generated is None and isinstance(payload.get("decision"), dict):
        generated = payload["decision"].get("generated_at_utc")
    decision_ts = _parse_utc_timestamp(generated)
    return decision_ts is not None and decision_ts > audit_ts


def _predictive_scout_pre_post_issues(
    intent: OrderIntent,
    *,
    validation_time_utc: datetime,
) -> list[dict[str, str]]:
    metadata = intent.metadata or {}
    if not predictive_scout_intent_claimed(intent):
        return []
    now = validation_time_utc
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    expires_at = _parse_utc_timestamp(metadata.get("predictive_scout_expires_at_utc"))
    if expires_at is not None and expires_at.astimezone(timezone.utc) > now:
        return []
    return [
        RiskIssue(
            "PREDICTIVE_SCOUT_EXPIRED_BEFORE_POST",
            "predictive SCOUT TTL expired during gateway work; do not POST the stale forward experiment",
        ).__dict__
    ]


def _predictive_scout_ledger_path_issues(
    intent: OrderIntent,
    *,
    intents_path: Path,
    configured_db_path: Path | None,
    canonical_db_path: Path,
) -> list[dict[str, str]]:
    if not predictive_scout_intent_claimed(intent) or configured_db_path is None:
        return []
    canonical = canonical_db_path
    if configured_db_path.expanduser().resolve() == canonical.expanduser().resolve():
        return []
    return [
        RiskIssue(
            "PREDICTIVE_SCOUT_CANONICAL_LEDGER_REQUIRED",
            "predictive SCOUT checks, atomic signal claim, reconciliation, and forward proof must use "
            f"the canonical intent ledger {canonical}; configured ledger {configured_db_path} is forbidden",
        ).__dict__
    ]


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _self_improvement_gateway_non_blocker(code: str, finding: dict[str, Any]) -> bool:
    if code not in SELF_IMPROVEMENT_GATEWAY_NON_BLOCKER_CODES:
        return False
    evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
    streak = _optional_int(evidence.get("current_streak"))
    if streak is None:
        return True
    return streak < SELF_IMPROVEMENT_STALE_DECISION_PERSISTENT_STREAK


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# `LATEST_GPT_DECISION_STALE` is repaired by producing/verifying a current GPT
# decision, so one fresh repair pass must not be self-blocked by the previous
# audit. Once the audit history says the same stale-decision P0 is persistent,
# staging fresh entry risk is blocked until the stale receipt is cleared.
SELF_IMPROVEMENT_GATEWAY_NON_BLOCKER_CODES = frozenset({"LATEST_GPT_DECISION_STALE"})
SELF_IMPROVEMENT_STALE_DECISION_PERSISTENT_STREAK = 2


def _build_order_request(intent: OrderIntent) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    try:
        return _oanda_order_request(intent), []
    except ValueError as exc:
        return None, [RiskIssue("ORDER_REQUEST_INVALID", str(exc)).__dict__]


def _predictive_scout_built_order_issues(
    intent: OrderIntent,
    order_request: dict[str, Any] | None,
) -> list[dict[str, str]]:
    if not predictive_scout_intent_claimed(intent):
        return []
    if order_request is None:
        return [
            RiskIssue(
                "PREDICTIVE_SCOUT_ORDER_REQUEST_MISSING",
                "predictive SCOUT requires a complete broker order request",
            ).__dict__
        ]
    expected_tp = _price(intent.pair, intent.tp)
    expected_sl = _price(intent.pair, intent.sl)
    actual_tp = (order_request.get("takeProfitOnFill") or {}).get("price")
    actual_sl = (order_request.get("stopLossOnFill") or {}).get("price")
    issues: list[dict[str, str]] = []
    if actual_tp != expected_tp:
        issues.append(
            RiskIssue(
                "PREDICTIVE_SCOUT_ATTACHED_TP_MISMATCH",
                "predictive SCOUT must POST its exact intent TP; a global TP kill switch or altered dependent order blocks this experiment",
            ).__dict__
        )
    if actual_sl != expected_sl:
        issues.append(
            RiskIssue(
                "PREDICTIVE_SCOUT_ATTACHED_SL_MISMATCH",
                "predictive SCOUT must POST its exact intent SL as stopLossOnFill",
            ).__dict__
        )
    if order_request.get("timeInForce") != "GTD" or not order_request.get("gtdTime"):
        issues.append(
            RiskIssue(
                "PREDICTIVE_SCOUT_GTD_MISSING",
                "predictive SCOUT pending order must be broker-expiring GTD, never ordinary GTC",
            ).__dict__
        )
    if order_request.get("positionFill") != "OPEN_ONLY":
        issues.append(
            RiskIssue(
                "PREDICTIVE_SCOUT_OPEN_ONLY_REQUIRED",
                "predictive SCOUT must use positionFill=OPEN_ONLY so it cannot reduce or net "
                "manual/operator exposure if account hedging settings change",
            ).__dict__
        )
    return issues


def _predictive_scout_receipt_from_intent(intent: OrderIntent) -> dict[str, Any] | None:
    metadata = intent.metadata or {}
    if not predictive_scout_intent_claimed(intent):
        return None
    rule = metadata.get("bidask_replay_precision_seed_rule")
    risk_plan = metadata.get("predictive_scout_risk_plan")
    return {
        "predictive_scout": True,
        "predictive_scout_vehicle_id": predictive_scout_vehicle_id(intent),
        "predictive_scout_experiment_id": predictive_scout_experiment_id(intent),
        "predictive_scout_signal_id": predictive_scout_signal_id(intent),
        "pair": intent.pair,
        "side": intent.side.value,
        "order_type": intent.order_type.value,
        "units": abs(int(intent.units)),
        "entry": intent.entry,
        "take_profit": intent.tp,
        "stop_loss": intent.sl,
        "predictive_scout_risk_tier": metadata.get("predictive_scout_risk_tier"),
        "predictive_scout_nav_jpy_at_sizing": metadata.get(
            "predictive_scout_nav_jpy_at_sizing"
        ),
        "predictive_scout_max_risk_pct_nav": metadata.get(
            "predictive_scout_max_risk_pct_nav"
        ),
        "predictive_scout_max_loss_jpy": metadata.get(
            "predictive_scout_max_loss_jpy"
        ),
        "predictive_scout_effective_max_loss_jpy": metadata.get(
            "predictive_scout_effective_max_loss_jpy"
        ),
        "predictive_scout_planned_initial_risk_jpy": metadata.get(
            "predictive_scout_planned_initial_risk_jpy"
        ),
        "predictive_scout_planned_initial_risk_pct_nav": metadata.get(
            "predictive_scout_planned_initial_risk_pct_nav"
        ),
        "predictive_scout_fresh_actual_initial_risk_jpy": metadata.get(
            "predictive_scout_fresh_actual_initial_risk_jpy"
        ),
        "predictive_scout_active_initial_risk_jpy": metadata.get(
            "predictive_scout_active_initial_risk_jpy"
        ),
        "predictive_scout_aggregate_initial_risk_jpy": metadata.get(
            "predictive_scout_aggregate_initial_risk_jpy"
        ),
        "predictive_scout_concurrent_risk_cap_jpy": metadata.get(
            "predictive_scout_concurrent_risk_cap_jpy"
        ),
        "predictive_scout_sizing_digest": metadata.get(
            "predictive_scout_sizing_digest"
        ),
        "predictive_scout_risk_plan": (
            dict(risk_plan) if isinstance(risk_plan, dict) else None
        ),
        "forecast_cycle_id": metadata.get("forecast_cycle_id"),
        "forecast_direction": metadata.get("forecast_direction"),
        "forecast_confidence": metadata.get("forecast_confidence"),
        "forecast_horizon_min": metadata.get("forecast_horizon_min"),
        "predictive_scout_source": metadata.get("predictive_scout_source"),
        "predictive_scout_rule_name": metadata.get("predictive_scout_rule_name"),
        "predictive_scout_rule_digest": metadata.get("predictive_scout_rule_digest"),
        "predictive_scout_hypothesis": metadata.get("predictive_scout_hypothesis"),
        "predictive_scout_vehicle_proof_status": metadata.get(
            "predictive_scout_vehicle_proof_status"
        ),
        "predictive_scout_promotion_allowed": metadata.get("predictive_scout_promotion_allowed"),
        "predictive_scout_generated_at_utc": metadata.get("predictive_scout_generated_at_utc"),
        "predictive_scout_expires_at_utc": metadata.get("predictive_scout_expires_at_utc"),
        "bidask_replay_precision_seed_rule": dict(rule) if isinstance(rule, dict) else None,
    }


def _context_evidence_from_intent(intent: OrderIntent) -> dict[str, Any]:
    """Persist entry context refs on gateway receipts for later P/L attribution.

    Entry-thesis sidecars also store this payload after a successful fill, but
    the gateway receipt is the durable fallback used by execution-ledger and
    outcome-mart joins when sidecar recording is delayed or fails.
    """

    try:
        from quant_rabbit.strategy.entry_thesis_ledger import _build_context_evidence

        evidence = _build_context_evidence(intent=intent, metadata=dict(intent.metadata or {}), forecast={})
    except Exception:
        return {}
    return evidence if isinstance(evidence, dict) else {}


def _sizing_evidence_from_intent(
    intent: OrderIntent,
    *,
    gateway_max_loss_jpy: float | None,
    requested_units: int | None,
    scaled_units: int | None,
) -> dict[str, Any]:
    metadata = dict(intent.metadata or {})
    out: dict[str, Any] = {
        "requested_units": requested_units,
        "scaled_units": scaled_units,
    }
    if gateway_max_loss_jpy is not None:
        out["gateway_max_loss_jpy"] = round(float(gateway_max_loss_jpy), 4)

    active = str(metadata.get("loss_asymmetry_guard_active") or "").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }
    if active:
        out["loss_asymmetry_guard_active"] = True
        for source_key, output_key in (
            ("loss_asymmetry_guard_loss_cap_jpy", "loss_asymmetry_guard_loss_cap_jpy"),
            ("loss_asymmetry_guard_base_max_loss_jpy", "loss_asymmetry_guard_base_max_loss_jpy"),
            ("loss_asymmetry_guard_effective_max_loss_jpy", "loss_asymmetry_guard_effective_max_loss_jpy"),
            ("capture_avg_win_jpy", "capture_avg_win_jpy"),
            ("capture_avg_loss_jpy", "capture_avg_loss_jpy"),
            ("capture_take_profit_trades", "capture_take_profit_trades"),
            ("capture_take_profit_expectancy_jpy", "capture_take_profit_expectancy_jpy"),
        ):
            value = _positive_float(metadata.get(source_key))
            if value is not None:
                out[output_key] = round(value, 4)
        mode = str(metadata.get("loss_asymmetry_guard_mode") or "").strip()
        if mode:
            out["loss_asymmetry_guard_mode"] = mode
        relaxed = metadata.get("loss_asymmetry_guard_relaxed")
        if isinstance(relaxed, bool):
            out["loss_asymmetry_guard_relaxed"] = relaxed
        reason = str(metadata.get("loss_asymmetry_guard_relaxation_reason") or "").strip()
        if reason:
            out["loss_asymmetry_guard_relaxation_reason"] = reason
        status = str(metadata.get("capture_economics_status") or "").strip()
        if status:
            out["capture_economics_status"] = status
        basis = str(metadata.get("loss_asymmetry_guard_basis") or "").strip()
        if basis:
            out["loss_asymmetry_guard_basis"] = basis
    return {key: value for key, value in out.items() if value is not None}


def _sizing_evidence_report_lines(value: Any, *, prefix: str) -> list[str]:
    if not isinstance(value, dict) or not value:
        return []
    lines: list[str] = []
    units = _fmt_units_transition(value.get("requested_units"), value.get("scaled_units"))
    if value.get("loss_asymmetry_guard_active"):
        cap = value.get("loss_asymmetry_guard_loss_cap_jpy")
        base = value.get("loss_asymmetry_guard_base_max_loss_jpy")
        effective = value.get("loss_asymmetry_guard_effective_max_loss_jpy")
        avg_win = value.get("capture_avg_win_jpy")
        avg_loss = value.get("capture_avg_loss_jpy")
        status = value.get("capture_economics_status") or "UNKNOWN"
        mode = value.get("loss_asymmetry_guard_mode") or "CAP_AVG_WIN"
        tp_trades = value.get("capture_take_profit_trades")
        tp_expectancy = value.get("capture_take_profit_expectancy_jpy")
        lines.append(
            f"{prefix}sizing guard: `LOSS_ASYMMETRY` mode=`{mode}` units=`{units}` status=`{status}` "
            f"cap=`{_fmt_jpy(cap)}` base_cap=`{_fmt_jpy(base)}` effective_cap=`{_fmt_jpy(effective)}` "
            f"avg_win/avg_loss=`{_fmt_jpy(avg_win)}`/`{_fmt_jpy(avg_loss)}` "
            f"tp_exits=`{_fmt_count(tp_trades)}` tp_expectancy=`{_fmt_jpy(tp_expectancy)}`"
        )
    elif value.get("gateway_max_loss_jpy") is not None:
        lines.append(
            f"{prefix}sizing cap: units=`{units}` gateway_max_loss=`{_fmt_jpy(value.get('gateway_max_loss_jpy'))}`"
        )
    return lines


def _fmt_units_transition(requested: Any, scaled: Any) -> str:
    if requested is None and scaled is None:
        return "n/a"
    if requested is None:
        return str(scaled)
    if scaled is None or scaled == requested:
        return str(requested)
    return f"{requested}->{scaled}"


def _fmt_count(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "n/a"


def _fmt_jpy(value: Any) -> str:
    try:
        return f"{float(value):.1f} JPY"
    except (TypeError, ValueError):
        return "n/a"


def _oanda_order_request(intent: OrderIntent) -> dict[str, Any]:
    signed_units = intent.units if intent.side == Side.LONG else -intent.units
    order_type = _oanda_order_type(intent.order_type)
    order: dict[str, Any] = {
        "type": order_type,
        "instrument": intent.pair,
        "units": str(signed_units),
        "positionFill": _position_fill(intent),
        "clientExtensions": _client_extensions(intent),
        "tradeClientExtensions": _client_extensions(intent),
    }
    # TP attachment (2026-05-15): `intent.tp` remains the virtual target used
    # by risk validation and reports. Per-intent metadata can omit the broker
    # cap for strong-trend runners; `QR_DISABLE_AUTO_TP=1` remains the global
    # emergency override.
    if _attach_take_profit_on_fill(intent):
        order["takeProfitOnFill"] = {"price": _price(intent.pair, intent.tp)}
    # SL handling. Two production modes can coexist:
    #
    # - SL-attached mode (default 2026-05-13 onward, post 00:33 JST
    #   mass-close incident): the operator demands every NEW entry
    #   carry a broker-side `stopLossOnFill` so a panic-close trader
    #   cannot bypass it via the autonomous CLOSE path. Controlled by
    #   `QR_NEW_ENTRY_INITIAL_SL=1`. Existing positions are not
    #   retroactively given an SL — they remain SL-free.
    #
    # - SL-free legacy mode (`QR_TRADER_DISABLE_SL_REPAIR=1`, no
    #   `QR_NEW_ENTRY_INITIAL_SL`): the original 2026-05-07 directive
    #   「SLいらない」 widening; the broker enforces no SL and the
    #   intent.sl is sizing/math only. Kept for backward compatibility.
    initial_sl_on = os.environ.get("QR_NEW_ENTRY_INITIAL_SL", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }
    if initial_sl_on or not _trader_sl_repair_disabled() or _requires_intent_stop_on_fill(intent):
        order["stopLossOnFill"] = {"price": _price(intent.pair, intent.sl)}
    else:
        # Disaster stop (2026-06-11, operator-approved 「SLの件もやっていい」):
        # in SL-free runtime a NEW entry still carries a broker-side
        # CATASTROPHE stop when intent metadata provides one (H4 ATR ×
        # QR_DISASTER_SL_H4_ATR_MULT × session widening, computed by
        # intent_generator strictly beyond the expected intent.sl). It is
        # not a trading stop: it never trails, it does not enter sizing or
        # reward/risk math, and existing positions are never retro-fitted.
        # Its only job is bounding the tail when the market dislocates
        # inside the full-trader blind window between cycles.
        disaster_sl = (intent.metadata or {}).get("disaster_sl")
        if disaster_sl is not None:
            try:
                order["stopLossOnFill"] = {"price": _price(intent.pair, float(disaster_sl))}
            except (TypeError, ValueError):
                pass
    if intent.order_type == OrderType.MARKET:
        order["timeInForce"] = "FOK"
    else:
        if intent.entry is None:
            raise ValueError("pending orders require entry")
        order["price"] = _price(intent.pair, intent.entry)
        if predictive_scout_intent_claimed(intent):
            expires_at = _parse_utc_timestamp(
                (intent.metadata or {}).get("predictive_scout_expires_at_utc")
            )
            if expires_at is None:
                raise ValueError("predictive SCOUT pending orders require predictive_scout_expires_at_utc")
            order["timeInForce"] = "GTD"
            order["gtdTime"] = expires_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        else:
            order["timeInForce"] = "GTC"
    return order


def _oanda_order_type(order_type: OrderType) -> str:
    if order_type == OrderType.STOP_ENTRY:
        return "STOP"
    return order_type.value


def _position_fill(intent: OrderIntent) -> str:
    if predictive_scout_intent_claimed(intent):
        return "OPEN_ONLY"
    raw = str(intent.metadata.get("position_fill") or "").upper()
    if raw in {"DEFAULT", "OPEN_ONLY", "REDUCE_FIRST", "REDUCE_ONLY"}:
        return raw
    if str(intent.metadata.get("position_intent") or "").upper() in {"HEDGE", "PYRAMID"}:
        return "OPEN_ONLY"
    return "DEFAULT"


def _price(pair: str, value: float) -> str:
    return f"{value:.{_price_precision(pair)}f}"


def _price_precision(pair: str) -> int:
    return 3 if pair.endswith("_JPY") else 5


def _price_tick(pair: str) -> float:
    return 10 ** -_price_precision(pair)


def _intent_with_gateway_metadata(intent: OrderIntent, lane_id: str) -> OrderIntent:
    if not lane_id:
        return intent
    metadata = dict(intent.metadata or {})
    metadata.setdefault("lane_id", lane_id)
    return replace(intent, metadata=metadata)


def _client_extensions(intent: OrderIntent) -> dict[str, str]:
    return {
        "id": _client_order_id(intent),
        "tag": Owner.TRADER.value,
        "comment": _comment(intent),
    }


def _client_order_id(intent: OrderIntent) -> str:
    lane_id = _gateway_lane_id(intent)
    seed = "|".join(
        (
            lane_id,
            intent.pair,
            intent.side.value,
            intent.order_type.value,
            datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        )
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    base = f"{CLIENT_ORDER_ID_PREFIX}-{intent.pair.replace('_', '')}-{intent.side.value[0]}-{digest}"
    if predictive_scout_intent_claimed(intent):
        # The exact signal id belongs in the broker extension id, not the
        # already space-constrained comment.  This preserves the broker-visible
        # role, vehicle, and lane prefix while allowing atomic reservations to
        # reconcile one exact claim rather than a same-vehicle approximation.
        base = f"{base}-{predictive_scout_signal_id(intent)}"
    return base[:128]


def _comment(intent: OrderIntent) -> str:
    desk = str(intent.metadata.get("desk") or "vnext")
    role = str(intent.metadata.get("campaign_role") or "")
    lane_id = _gateway_lane_id(intent)
    parent_lane_id = str(intent.metadata.get("parent_lane_id") or "").strip()
    role_part = f" role={role}" if role else ""
    vehicle_part = (
        f" vehicle={predictive_scout_vehicle_id(intent)}"
        if predictive_scout_intent_claimed(intent)
        else ""
    )
    parent_part = (
        f" parent={parent_lane_id}"
        if parent_lane_id and parent_lane_id != lane_id
        else ""
    )
    lane_part = f" lane={lane_id}" if lane_id else ""
    # SCOUT concurrency is reconstructed from the broker's truncated comment.
    # Put the bounded campaign role before potentially long lane text so the
    # one-active-SCOUT guard cannot lose its identity at the 128-byte boundary.
    text = f"qr-vnext{role_part}{vehicle_part}{parent_part}{lane_part} desk={desk}".strip()
    return text[:128]


def _gateway_lane_id(intent: OrderIntent) -> str:
    metadata = intent.metadata or {}
    return str(metadata.get("lane_id") or metadata.get("parent_lane_id") or "").strip()


def _intent_from_json(payload: dict[str, Any]) -> OrderIntent:
    return OrderIntent(
        pair=str(payload["pair"]).upper(),
        side=Side.parse(str(payload["side"])),
        order_type=OrderType.parse(str(payload["order_type"])),
        units=int(payload["units"]),
        entry=float(payload["entry"]) if payload.get("entry") is not None else None,
        tp=float(payload["tp"]),
        sl=float(payload["sl"]),
        thesis=str(payload.get("thesis") or ""),
        reason=str(payload.get("reason") or ""),
        owner=Owner(str(payload.get("owner") or Owner.TRADER.value)),
        market_context=_market_context_from_json(payload.get("market_context")),
        metadata=dict(payload.get("metadata") or {}),
    )


def _market_context_from_json(payload: object) -> MarketContext | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("market_context must be an object")
    return MarketContext(
        regime=str(payload.get("regime") or ""),
        narrative=str(payload.get("narrative") or ""),
        chart_story=str(payload.get("chart_story") or ""),
        method=TradeMethod.parse(str(payload.get("method") or "")),
        invalidation=str(payload.get("invalidation") or ""),
        event_risk=str(payload.get("event_risk") or ""),
        session=str(payload.get("session") or ""),
    )

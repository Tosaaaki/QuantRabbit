from __future__ import annotations

import math
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from contextlib import closing
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
    DEFAULT_FORECAST_HISTORY,
    DEFAULT_GUARDIAN_ACTION_RECEIPT,
    DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
    DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_LIVE_ORDER_STAGE_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_QR_TRADER_RUN_WATCHDOG,
    DEFAULT_STRATEGY_PROFILE,
)
from quant_rabbit.execution_ledger import ExecutionLedger
from quant_rabbit.forecast_precision import hit_rate_wilson_lower
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.capture_economics import (
    EXECUTION_COST_FLOOR_CONTRACT,
    execution_cost_floor_from_surface,
    evaluate_exact_vehicle_net_edge,
    exact_vehicle_metrics_from_surface,
    read_attributed_net_outcomes,
    read_exact_vehicle_allocation_surface,
)
from quant_rabbit.decision_execution_lineage import (
    DEFAULT_MARKET_READ_EXECUTION_LINKS,
    DecisionExecutionLineageError,
    append_execution_link,
    attach_lineage_metadata,
    build_execution_link,
    lineage_from_metadata,
    read_verified_decision_lineage,
)
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
from quant_rabbit.guardian_tuning_overrides import (
    clear_guardian_tuning_validation_cache,
    guardian_tuning_validation_cycle,
)
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
    _loss_asymmetry_tp_proof_collection_shape_allowed,
    _loss_asymmetry_tp_relaxation_shape_allowed,
    _min_lot_test_override_active,
    m15_recovery_micro_claimed,
    resolve_max_loss_jpy,
    validate_m15_recovery_micro_live_claim,
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
    LOSS_ASYMMETRY_TP_RELAX_MIN_EXIT_TRADES,
    MACRO_EVENT_MAX_RISK_PCT_NAV,
    _daily_risk_budget_from_state,
    _exact_vehicle_take_profit_metrics,
    _expired_pending_projection_count,
    _per_trade_risk_from_state,
)
from quant_rabbit.strategy.directional_forecaster import (
    M15_RECOVERY_MICRO_MAX_UNITS,
)
from quant_rabbit.strategy.profile import StrategyProfile


# Every fresh entry is bounded by the same current-equity policy used by the
# macro confidence path.  The macro signal may choose a smaller fraction but
# cannot own or bypass this absolute pre-POST boundary.
FRESH_ENTRY_MAX_RISK_PCT_NAV = MACRO_EVENT_MAX_RISK_PCT_NAV
# Exact broker-TP history below five completed zero-loss lifecycles is not a
# usable capital-allocation exception. This is an evidence-collection floor,
# not a market tuning parameter; the all-exit contradiction check still wins.
CAPITAL_ALLOCATION_MIN_EXACT_TP_TRADES = 5
M15_RECOVERY_EDGE_COLLECTION_BASIS = "M15_RECOVERY_EDGE_COLLECTION"
M15_RECOVERY_PREBOUNDED_REASON = "M15_RECOVERY_MICRO_PREBOUNDED_CONTRACT"
TARGET_PATH_LIVE_LEARNING_EDGE_COLLECTION_BASIS = (
    "TARGET_PATH_LIVE_LEARNING_EDGE_COLLECTION"
)
TARGET_PATH_LIVE_LEARNING_PREBOUNDED_REASON = (
    "TARGET_PATH_LIVE_LEARNING_PREBOUNDED_CONTRACT"
)
FORECAST_S5_SECONDS = 5
FORECAST_S5_MAX_WINDOW_SECONDS = 15 * 60
# AI is a supervisory/tuning layer in the fast-bot architecture.  Keep its
# broker-entry authority fixed off at the last shared order gateway so a stale
# scheduler, replayed verified receipt, or accidentally re-enabled Guardian
# handoff cannot turn an AI market read into a broker POST.  This is deliberately
# scoped to explicit AI author identities; deterministic protection uses the
# separate PositionProtectionGateway and a future FAST_BOT author identity is
# not rejected by this authority boundary.
AI_ORDER_AUTHORITY = "NONE"
AI_ORDER_AUTHOR_KINDS = frozenset({"AI", "CODEX_MARKET_READ"})
PRE_ENTRY_FORECAST_CYCLE_RE = re.compile(
    r"^pre-entry-forecast-refresh:"
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})):"
)


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


@dataclass(frozen=True)
class _FinalPrePostBoundaryResult:
    evidence: dict[str, Any]
    issues: tuple[RiskIssue, ...] = ()


@dataclass(frozen=True)
class _VerifiedDecisionReceiptFreeze:
    """One-read receipt facts propagated through every broker boundary."""

    sha256: str | None
    numeric_allocation_required: bool
    numeric_requirement_issue: dict[str, str] | None
    capital_allocation_edge_basis: str | None
    execution_cost_floor_sha256: str | None
    guardian_provenance_issue: dict[str, str] | None = None
    guardian_action_receipt_material_contract: str | None = None
    guardian_action_receipt_baseline_pairs: tuple[str, ...] = ()
    guardian_action_receipt_scope_state_sha256: str | None = None


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
        pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
        forecast_history_path: Path = DEFAULT_FORECAST_HISTORY,
        execution_ledger_db_path: Path | None = None,
        execution_ledger_report_path: Path | None = None,
        predictive_scout_canonical_ledger_db_path: Path | None = None,
        market_read_execution_links_path: Path | None = None,
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
        self.market_read_execution_links_path = (
            market_read_execution_links_path
            if market_read_execution_links_path is not None
            else (
                DEFAULT_MARKET_READ_EXECUTION_LINKS
                if output_path == DEFAULT_LIVE_ORDER_REQUEST
                else output_path.parent / DEFAULT_MARKET_READ_EXECUTION_LINKS.name
            )
        )
        self.self_improvement_audit = self_improvement_audit
        # Every fresh-entry live send is bound to the ACCEPTED receipt that
        # gpt-trader-decision just verified.  Dry-run/staging may leave this
        # unset, but the broker boundary fails closed when ``send`` is true.
        # This keeps direct/manual gateway invocation from bypassing the GPT
        # lane and capital-allocation decision.
        self.verified_decision_path = verified_decision_path
        self.guardian_action_receipt_path = (
            guardian_action_receipt_path
            if guardian_action_receipt_path != DEFAULT_GUARDIAN_ACTION_RECEIPT
            or output_path == DEFAULT_LIVE_ORDER_REQUEST
            else output_path.parent / DEFAULT_GUARDIAN_ACTION_RECEIPT.name
        )
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
        self.pair_charts_path = pair_charts_path
        self.forecast_history_path = (
            forecast_history_path
            if forecast_history_path != DEFAULT_FORECAST_HISTORY
            or output_path == DEFAULT_LIVE_ORDER_REQUEST
            else output_path.parent / DEFAULT_FORECAST_HISTORY.name
        )

    @guardian_tuning_validation_cycle()
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
        verified_decision_receipt_at_entry = (
            _freeze_verified_decision_receipt(self.verified_decision_path)
            if send
            else _VerifiedDecisionReceiptFreeze(None, False, None, None, None)
        )
        verified_decision_sha_at_entry = (
            verified_decision_receipt_at_entry.sha256
        )
        numeric_allocation_recheck_required = (
            verified_decision_receipt_at_entry.numeric_allocation_required
        )
        numeric_allocation_requirement_issue = (
            verified_decision_receipt_at_entry.numeric_requirement_issue
        )
        guardian_provenance_issue = (
            verified_decision_receipt_at_entry.guardian_provenance_issue
        )
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
        verified_intent = intent
        intent, decision_lineage_issues = self._intent_with_verified_decision_lineage(
            intent,
            selected_lane_id=selected_lane_id,
        )
        requested_units = intent.units
        authorized_size_multiple = size_multiple
        scaled_units, scale_issues, size_multiple = _scaled_units_for_intent(intent, size_multiple)
        authorized_units = abs(int(scaled_units)) if scaled_units is not None else None
        capital_allocation_validated = False
        pre_reservation_capital_allocation_validated = False
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
            issue.__dict__
            for issue in _strategy_profile_live_send_issues(
                intent,
                StrategyProfile.load(self.strategy_profile).validate(
                    intent,
                    for_live_send=True,
                ),
            )
        )
        risk_issues = [issue.__dict__ for issue in risk.issues]
        risk_issues.extend(
            issue.__dict__ for issue in _selected_lane_metadata_issues(intent)
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
            expected_sha256=verified_decision_sha_at_entry,
            predictive_scout_trade_only=predictive_scout_intent_claimed(intent),
            intent=intent,
            base_units=requested_units,
            authorized_size_multiple=authorized_size_multiple,
            authorized_units=authorized_units,
            final_units=intent.units,
            order_request=order_request,
        )
        gpt_verified_decision_issues.extend(decision_lineage_issues)
        if numeric_allocation_requirement_issue is not None:
            gpt_verified_decision_issues.append(
                numeric_allocation_requirement_issue
            )
        if guardian_provenance_issue is not None:
            gpt_verified_decision_issues.append(guardian_provenance_issue)
        send_issues = _send_guard_issues(send=send, confirm_live=confirm_live, lane_id=lane_id)
        target_path_issues = _target_path_live_send_issues(intent, send=send)
        guardian_action_issues = guardian_action_gateway_issues(
            intent_metadata=intent.metadata,
            pair=intent.pair,
            thesis=intent.thesis,
            action_receipt_path=self.guardian_action_receipt_path,
        )
        guardian_receipt_consumption_issues = self._guardian_receipt_consumption_gateway_issues(
            risk_issues,
            risk=risk,
            snapshot=snapshot,
        )
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
        final_pre_post_boundary_issues: list[dict[str, str]] = []
        market_read_execution_link: dict[str, Any] = {"status": "NOT_RUN"}
        pre_post_reconciliation: dict[str, Any] = {"status": "NOT_RUN"}
        reservation_order_request_sha256: str | None = None
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
                verified_intent=verified_intent,
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
                numeric_allocation_recheck_required=(
                    numeric_allocation_recheck_required
                ),
                expected_receipt_sha256=verified_decision_sha_at_entry,
                expected_capital_allocation_edge_basis=(
                    verified_decision_receipt_at_entry.capital_allocation_edge_basis
                ),
                expected_execution_cost_floor_sha256=(
                    verified_decision_receipt_at_entry.execution_cost_floor_sha256
                ),
            )
            # The final deep attestation is useful only while building this
            # reconciliation result.  Do not carry it across reservations or
            # the broker POST boundary inside the outer gateway phase.
            clear_guardian_tuning_validation_cache()
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
            final_gpt_allocation_issues = _gpt_verified_decision_live_send_issues(
                self.verified_decision_path,
                selected_lane_id=selected_lane_id,
                intents_payload=intents_payload,
                send=send,
                expected_sha256=verified_decision_sha_at_entry,
                predictive_scout_trade_only=predictive_scout_intent_claimed(intent),
                intent=intent,
                base_units=requested_units,
                authorized_size_multiple=authorized_size_multiple,
                authorized_units=authorized_units,
                final_units=intent.units,
                order_request=order_request,
            )
            gpt_verified_decision_issues.extend(final_gpt_allocation_issues)
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
                or any(
                    issue.get("severity") == "BLOCK"
                    for issue in final_gpt_allocation_issues
                )
                or any(issue.get("severity") == "BLOCK" for issue in order_build_issues)
                or any(issue.get("severity") == "BLOCK" for issue in sl_lint_issues)
            ):
                all_blocked = True
                status = "BLOCKED"
        if (
            send
            and order_request is not None
            and self.verified_decision_path is not None
        ):
            receipt_change_issue = _verified_decision_receipt_change_issue(
                self.verified_decision_path,
                expected_sha256=verified_decision_sha_at_entry,
            )
            if receipt_change_issue is not None:
                gpt_verified_decision_issues.append(receipt_change_issue)
                all_blocked = True
                status = "BLOCKED"
            else:
                pre_reservation_capital_allocation_validated = (
                    _gpt_capital_allocation_binding_validated(
                        self.verified_decision_path,
                        expected_sha256=verified_decision_sha_at_entry,
                        issues=gpt_verified_decision_issues,
                    )
                    and _pre_post_capital_allocation_edge_validated(
                        pre_post_reconciliation
                    )
                )
        if send and order_request is not None and not all_blocked:
            reservation_order_request_sha256 = _canonical_order_request_sha256(
                order_request
            )
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
            final_boundary = self._final_pre_post_boundary(
                verified_intent=verified_intent,
                final_intent=intent,
                early_reconciliation=pre_post_reconciliation,
                expected_receipt_sha256=verified_decision_sha_at_entry,
                expected_capital_allocation_edge_basis=(
                    verified_decision_receipt_at_entry.capital_allocation_edge_basis
                ),
                expected_execution_cost_floor_sha256=(
                    verified_decision_receipt_at_entry.execution_cost_floor_sha256
                ),
                expected_guardian_action_receipt_material_contract=(
                    verified_decision_receipt_at_entry.guardian_action_receipt_material_contract
                ),
                expected_guardian_action_receipt_baseline_pairs=(
                    verified_decision_receipt_at_entry.guardian_action_receipt_baseline_pairs
                ),
                expected_guardian_action_receipt_scope_state_sha256=(
                    verified_decision_receipt_at_entry.guardian_action_receipt_scope_state_sha256
                ),
                expected_order_request_sha256=reservation_order_request_sha256,
                order_request=order_request,
                ordinary_entry_claim=ordinary_entry_claim,
            )
            pre_post_reconciliation = {
                **pre_post_reconciliation,
                "final_post_reservation_boundary": final_boundary.evidence,
            }
            final_pre_post_boundary_issues.extend(
                issue.__dict__ for issue in final_boundary.issues
            )
            final_boundary_blocked = any(
                issue.severity == "BLOCK" for issue in final_boundary.issues
            )
            capital_allocation_validated = bool(
                pre_reservation_capital_allocation_validated
                and not final_boundary_blocked
                and final_boundary.evidence.get("status") == "PASSED"
            )
            if final_boundary_blocked:
                ordinary_entry_claim, claim_finalize_issue = (
                    self._finalize_ordinary_entry_post_claim(
                        ordinary_entry_claim,
                        status="FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
                        broker_outcome={
                            "post_attempted": False,
                            "final_post_reservation_boundary": final_boundary.evidence,
                        },
                    )
                )
                if claim_finalize_issue is not None:
                    ordinary_entry_claim_issues.append(claim_finalize_issue)
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
            market_read_execution_link, lineage_link_issue = self._record_market_read_execution_link(
                intent=intent,
                lane_id=selected_lane_id,
                order_request=order_request,
                response=response,
                ordinary_entry_claim=ordinary_entry_claim,
            )
            if lineage_link_issue is not None:
                ordinary_entry_claim_issues.append(lineage_link_issue)
                status = "SENT_WITH_DECISION_LINEAGE_GAP"
        result = {
            "generated_at_utc": generated_at,
            "status": status,
            "lane_id": selected_lane_id,
            "order_request": order_request,
            "decision_lineage": _decision_lineage_receipt_from_intent(intent, order_request),
            "range_vehicle_candidate_receipt": _range_vehicle_candidate_receipt_from_intent(
                intent, order_request
            ),
            "predictive_scout": predictive_scout_intent_claimed(intent),
            "predictive_scout_receipt": _predictive_scout_receipt_from_intent(intent),
            "context_evidence": _context_evidence_from_intent(intent),
            "sizing_evidence": _sizing_evidence_from_intent(
                intent,
                gateway_max_loss_jpy=max_loss_jpy,
                requested_units=requested_units,
                scaled_units=intent.units,
                authorized_size_multiple=authorized_size_multiple,
                authorized_units=authorized_units,
                capital_allocation_validated=capital_allocation_validated,
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
                *final_pre_post_boundary_issues,
            ],
            "strategy_issues": list(strategy_issues),
            "send_requested": send,
            "sent": sent,
            "response": response,
            "entry_thesis_record": entry_thesis_record,
            "ordinary_entry_claim": ordinary_entry_claim,
            "market_read_execution_link": market_read_execution_link,
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

    @guardian_tuning_validation_cycle()
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
        decision_lineage_gap_count = sum(
            1
            for item in order_results
            if item.get("status") == "SENT_WITH_DECISION_LINEAGE_GAP"
        )
        if send:
            if decision_lineage_gap_count and blocked_count:
                status = "PARTIAL_SENT_WITH_DECISION_LINEAGE_GAP"
            elif decision_lineage_gap_count:
                status = "SENT_WITH_DECISION_LINEAGE_GAP"
            elif entry_thesis_gap_count and blocked_count:
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
            "decision_lineages": [
                item.get("decision_lineage")
                for item in order_results
                if isinstance(item.get("decision_lineage"), dict)
            ],
            "market_read_execution_links": [
                item.get("market_read_execution_link")
                for item in order_results
                if isinstance(item.get("market_read_execution_link"), dict)
                and item.get("market_read_execution_link", {}).get("status")
                not in {"NOT_RUN", "NOT_APPLICABLE_NO_MARKET_READ_LINEAGE"}
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
        verified_decision_receipt_at_entry = (
            _freeze_verified_decision_receipt(self.verified_decision_path)
            if send
            else _VerifiedDecisionReceiptFreeze(None, False, None, None, None)
        )
        verified_decision_sha_at_entry = (
            verified_decision_receipt_at_entry.sha256
        )
        numeric_allocation_recheck_required = (
            verified_decision_receipt_at_entry.numeric_allocation_required
        )
        numeric_allocation_requirement_issue = (
            verified_decision_receipt_at_entry.numeric_requirement_issue
        )
        guardian_provenance_issue = (
            verified_decision_receipt_at_entry.guardian_provenance_issue
        )
        selected_lane_id = str(selected.get("lane_id") or "")
        intent = _intent_from_json(selected["intent"])
        intent = _intent_with_gateway_metadata(intent, selected_lane_id)
        verified_intent = intent
        intent, decision_lineage_issues = self._intent_with_verified_decision_lineage(
            intent,
            selected_lane_id=selected_lane_id,
        )
        requested_units = intent.units
        authorized_size_multiple = size_multiple
        scaled_units, scale_issues, size_multiple = _scaled_units_for_intent(intent, size_multiple)
        authorized_units = abs(int(scaled_units)) if scaled_units is not None else None
        capital_allocation_validated = False
        pre_reservation_capital_allocation_validated = False
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
            issue.__dict__
            for issue in _strategy_profile_live_send_issues(
                intent,
                StrategyProfile.load(self.strategy_profile).validate(
                    intent,
                    for_live_send=True,
                ),
            )
        )
        risk_issues = [issue.__dict__ for issue in risk.issues]
        risk_issues.extend(
            issue.__dict__ for issue in _selected_lane_metadata_issues(intent)
        )
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
            expected_sha256=verified_decision_sha_at_entry,
            predictive_scout_trade_only=predictive_scout_intent_claimed(intent),
            intent=intent,
            base_units=requested_units,
            authorized_size_multiple=authorized_size_multiple,
            authorized_units=authorized_units,
            final_units=intent.units,
            order_request=order_request,
        )
        gpt_verified_decision_issues.extend(decision_lineage_issues)
        if numeric_allocation_requirement_issue is not None:
            gpt_verified_decision_issues.append(
                numeric_allocation_requirement_issue
            )
        if guardian_provenance_issue is not None:
            gpt_verified_decision_issues.append(guardian_provenance_issue)
        send_issues = _send_guard_issues(send=send, confirm_live=confirm_live, lane_id=lane_id_arg)
        target_path_issues = _target_path_live_send_issues(intent, send=send)
        guardian_action_issues = guardian_action_gateway_issues(
            intent_metadata=intent.metadata,
            pair=intent.pair,
            thesis=intent.thesis,
            action_receipt_path=self.guardian_action_receipt_path,
        )
        guardian_receipt_consumption_issues = self._guardian_receipt_consumption_gateway_issues(
            risk_issues,
            risk=risk,
            snapshot=snapshot,
        )
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
        final_pre_post_boundary_issues: list[dict[str, str]] = []
        market_read_execution_link: dict[str, Any] = {"status": "NOT_RUN"}
        pre_post_reconciliation: dict[str, Any] = {"status": "NOT_RUN"}
        reservation_order_request_sha256: str | None = None
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
                verified_intent=verified_intent,
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
                numeric_allocation_recheck_required=(
                    numeric_allocation_recheck_required
                ),
                expected_receipt_sha256=verified_decision_sha_at_entry,
                expected_capital_allocation_edge_basis=(
                    verified_decision_receipt_at_entry.capital_allocation_edge_basis
                ),
                expected_execution_cost_floor_sha256=(
                    verified_decision_receipt_at_entry.execution_cost_floor_sha256
                ),
            )
            clear_guardian_tuning_validation_cache()
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
            final_gpt_allocation_issues = _gpt_verified_decision_live_send_issues(
                self.verified_decision_path,
                selected_lane_id=selected_lane_id,
                intents_payload=intents_payload,
                send=send,
                expected_sha256=verified_decision_sha_at_entry,
                predictive_scout_trade_only=predictive_scout_intent_claimed(intent),
                intent=intent,
                base_units=requested_units,
                authorized_size_multiple=authorized_size_multiple,
                authorized_units=authorized_units,
                final_units=intent.units,
                order_request=order_request,
            )
            gpt_verified_decision_issues.extend(final_gpt_allocation_issues)
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
                or any(
                    issue.get("severity") == "BLOCK"
                    for issue in final_gpt_allocation_issues
                )
                or any(issue.get("severity") == "BLOCK" for issue in order_build_issues)
                or any(issue.get("severity") == "BLOCK" for issue in sl_lint_issues)
            ):
                all_blocked = True
                status = "BLOCKED"
        if (
            send
            and order_request is not None
            and self.verified_decision_path is not None
        ):
            receipt_change_issue = _verified_decision_receipt_change_issue(
                self.verified_decision_path,
                expected_sha256=verified_decision_sha_at_entry,
            )
            if receipt_change_issue is not None:
                gpt_verified_decision_issues.append(receipt_change_issue)
                all_blocked = True
                status = "BLOCKED"
            else:
                pre_reservation_capital_allocation_validated = (
                    _gpt_capital_allocation_binding_validated(
                        self.verified_decision_path,
                        expected_sha256=verified_decision_sha_at_entry,
                        issues=gpt_verified_decision_issues,
                    )
                    and _pre_post_capital_allocation_edge_validated(
                        pre_post_reconciliation
                    )
                )
        if send and order_request is not None and not all_blocked:
            reservation_order_request_sha256 = _canonical_order_request_sha256(
                order_request
            )
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
            final_boundary = self._final_pre_post_boundary(
                verified_intent=verified_intent,
                final_intent=intent,
                early_reconciliation=pre_post_reconciliation,
                expected_receipt_sha256=verified_decision_sha_at_entry,
                expected_capital_allocation_edge_basis=(
                    verified_decision_receipt_at_entry.capital_allocation_edge_basis
                ),
                expected_execution_cost_floor_sha256=(
                    verified_decision_receipt_at_entry.execution_cost_floor_sha256
                ),
                expected_guardian_action_receipt_material_contract=(
                    verified_decision_receipt_at_entry.guardian_action_receipt_material_contract
                ),
                expected_guardian_action_receipt_baseline_pairs=(
                    verified_decision_receipt_at_entry.guardian_action_receipt_baseline_pairs
                ),
                expected_guardian_action_receipt_scope_state_sha256=(
                    verified_decision_receipt_at_entry.guardian_action_receipt_scope_state_sha256
                ),
                expected_order_request_sha256=reservation_order_request_sha256,
                order_request=order_request,
                ordinary_entry_claim=ordinary_entry_claim,
            )
            pre_post_reconciliation = {
                **pre_post_reconciliation,
                "final_post_reservation_boundary": final_boundary.evidence,
            }
            final_pre_post_boundary_issues.extend(
                issue.__dict__ for issue in final_boundary.issues
            )
            final_boundary_blocked = any(
                issue.severity == "BLOCK" for issue in final_boundary.issues
            )
            capital_allocation_validated = bool(
                pre_reservation_capital_allocation_validated
                and not final_boundary_blocked
                and final_boundary.evidence.get("status") == "PASSED"
            )
            if final_boundary_blocked:
                ordinary_entry_claim, claim_finalize_issue = (
                    self._finalize_ordinary_entry_post_claim(
                        ordinary_entry_claim,
                        status="FINAL_BOUNDARY_BLOCKED_RESERVATION_RETAINED",
                        broker_outcome={
                            "post_attempted": False,
                            "final_post_reservation_boundary": final_boundary.evidence,
                        },
                    )
                )
                if claim_finalize_issue is not None:
                    ordinary_entry_claim_issues.append(claim_finalize_issue)
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
            market_read_execution_link, lineage_link_issue = self._record_market_read_execution_link(
                intent=intent,
                lane_id=selected_lane_id,
                order_request=order_request,
                response=response,
                ordinary_entry_claim=ordinary_entry_claim,
            )
            if lineage_link_issue is not None:
                ordinary_entry_claim_issues.append(lineage_link_issue)
                status = "SENT_WITH_DECISION_LINEAGE_GAP"
        return {
            "generated_at_utc": generated_at,
            "status": status,
            "lane_id": selected_lane_id,
            "order_request": order_request,
            "decision_lineage": _decision_lineage_receipt_from_intent(intent, order_request),
            "range_vehicle_candidate_receipt": _range_vehicle_candidate_receipt_from_intent(
                intent, order_request
            ),
            "predictive_scout": predictive_scout_intent_claimed(intent),
            "predictive_scout_receipt": _predictive_scout_receipt_from_intent(intent),
            "context_evidence": _context_evidence_from_intent(intent),
            "sizing_evidence": _sizing_evidence_from_intent(
                intent,
                gateway_max_loss_jpy=max_loss_jpy,
                requested_units=requested_units,
                scaled_units=intent.units,
                authorized_size_multiple=authorized_size_multiple,
                authorized_units=authorized_units,
                capital_allocation_validated=capital_allocation_validated,
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
                *final_pre_post_boundary_issues,
            ],
            "strategy_issues": list(strategy_issues),
            "send_requested": send,
            "sent": sent,
            "response": response,
            "entry_thesis_record": entry_thesis_record,
            "ordinary_entry_claim": ordinary_entry_claim,
            "market_read_execution_link": market_read_execution_link,
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

    def _guardian_receipt_consumption_gateway_issues(
        self,
        risk_issues: list[dict[str, Any]],
        *,
        risk: Any,
        snapshot: BrokerSnapshot,
    ) -> list[dict[str, Any]]:
        if any(
            issue.get("code") in {GUARDIAN_RECEIPT_BLOCK_NEW_ENTRY_CODE, GUARDIAN_WATCHDOG_BLOCK_NEW_ENTRY_CODE}
            for issue in risk_issues
            if isinstance(issue, dict)
        ):
            return []
        metrics = risk.metrics if isinstance(risk.metrics, RiskMetrics) else None
        account = snapshot.account
        current_margin_utilization_pct = None
        account_payload: dict[str, Any] = {}
        if account is not None:
            account_payload = {
                "nav_jpy": account.nav_jpy,
                "margin_used_jpy": account.margin_used_jpy,
                "margin_available_jpy": account.margin_available_jpy,
            }
            if account.nav_jpy > 0:
                current_margin_utilization_pct = (
                    account.margin_used_jpy / account.nav_jpy * 100.0
                )
        return guardian_receipt_new_entry_blockers_from_paths(
            watchdog_path=self.qr_trader_run_watchdog_path,
            consumption_path=self.guardian_receipt_consumption_path,
            operator_review_path=self.guardian_receipt_operator_review_path,
            broker_snapshot_path=self.broker_snapshot_path,
            broker_snapshot_payload={"account": account_payload},
            allow_p1_margin_warning=True,
            current_margin_utilization_pct=current_margin_utilization_pct,
            projected_margin_utilization_pct=(
                metrics.margin_utilization_after_pct if metrics is not None else None
            ),
            max_margin_utilization_pct=(
                metrics.max_margin_utilization_pct if metrics is not None else None
            ),
            margin_available_jpy=(
                account.margin_available_jpy if account is not None else None
            ),
        )

    def _intent_with_verified_decision_lineage(
        self,
        intent: OrderIntent,
        *,
        selected_lane_id: str,
    ) -> tuple[OrderIntent, list[dict[str, str]]]:
        """Bind one selected intent to the exact accepted GPT market read.

        This is an audit binding only.  It cannot authorize a send and is
        attached before the ordinary verifier/risk/gateway checks run.  A
        malformed current CODEX_MARKET_READ lineage is a BLOCK; a historical
        receipt without that contract remains lineage-unattributed rather than
        being guessed from pair or time.
        """

        try:
            lineage = read_verified_decision_lineage(
                self.verified_decision_path,
                selected_lane_id=selected_lane_id,
            )
            metadata = attach_lineage_metadata(intent.metadata, lineage)
        except DecisionExecutionLineageError as exc:
            return intent, [
                RiskIssue(
                    "GPT_DECISION_EXECUTION_LINEAGE_INVALID",
                    f"current accepted GPT entry cannot be bound to immutable market-read lineage: {exc}",
                ).__dict__
            ]
        return replace(intent, metadata=metadata), []

    def _record_market_read_execution_link(
        self,
        *,
        intent: OrderIntent,
        lane_id: str,
        order_request: dict[str, Any] | None,
        response: Any,
        ordinary_entry_claim: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str] | None]:
        """Append an exact-ID link only after an actual gateway response."""

        try:
            lineage = lineage_from_metadata(intent.metadata)
        except DecisionExecutionLineageError as exc:
            return {
                "status": "INVALID_INTENT_LINEAGE",
                "path": str(self.market_read_execution_links_path),
                "error": str(exc),
            }, RiskIssue(
                "MARKET_READ_EXECUTION_LINK_INVALID",
                f"broker POST returned but its GPT/market-read lineage is invalid: {exc}",
            ).__dict__
        if lineage is None:
            return {
                "status": "NOT_APPLICABLE_NO_MARKET_READ_LINEAGE",
                "path": str(self.market_read_execution_links_path),
            }, None
        if not isinstance(response, dict):
            return {
                "status": "ACTUAL_GATEWAY_RESPONSE_MISSING",
                "path": str(self.market_read_execution_links_path),
            }, RiskIssue(
                "MARKET_READ_EXECUTION_LINK_RESPONSE_MISSING",
                "broker POST returned without a JSON object, so explicit order/fill/trade ids cannot be linked",
            ).__dict__
        metadata = dict(intent.metadata or {})
        client_extensions = (
            order_request.get("clientExtensions")
            if isinstance(order_request, dict)
            and isinstance(order_request.get("clientExtensions"), dict)
            else {}
        )
        try:
            order_request_sha256 = str(
                ordinary_entry_claim.get("order_request_sha256") or ""
            ).strip() or (
                hashlib.sha256(
                    json.dumps(
                        order_request or {},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            )
            link = build_execution_link(
                lineage=lineage,
                gateway_response=response,
                lane_id=lane_id,
                parent_lane_id=str(metadata.get("parent_lane_id") or "") or None,
                forecast_cycle_id=str(metadata.get("forecast_cycle_id") or "") or None,
                claim_id=str(ordinary_entry_claim.get("claim_id") or "") or None,
                order_request_sha256=order_request_sha256,
                client_extension_id=str(client_extensions.get("id") or "") or None,
            )
            return append_execution_link(self.market_read_execution_links_path, link), None
        except (DecisionExecutionLineageError, OSError) as exc:
            return {
                "status": "WRITE_FAILED",
                "path": str(self.market_read_execution_links_path),
                "error": f"{type(exc).__name__}: {exc}",
            }, RiskIssue(
                "MARKET_READ_EXECUTION_LINK_WRITE_FAILED",
                "broker POST returned but its explicit OANDA ids could not be appended to the decision lineage: "
                f"{type(exc).__name__}: {exc}",
            ).__dict__

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
        numeric_allocation_recheck_required: bool = False,
        expected_receipt_sha256: str | None = None,
        expected_capital_allocation_edge_basis: str | None = None,
        expected_execution_cost_floor_sha256: str | None = None,
    ) -> _PrePostReconciliationResult:
        """Reconcile loss capacity against broker truth immediately before POST.

        The whole sequence intentionally lives inside the gateway so direct,
        batch, AutoTradeCycle, and guardian entry paths share the same final
        defense. The refreshed target publishes capacity *before* open risk;
        this method then subtracts fresh open, pending, and candidate risk once
        from the exact snapshot whose lastTransactionID matched the ledger.
        """

        clear_guardian_tuning_validation_cache()
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

        fresh_base_risk = None
        fresh_forecast_s5_path_proof = None
        if numeric_allocation_recheck_required:
            try:
                # Re-measure the immutable GPT baseline units against the same
                # fresh broker snapshot used for the final order. A deliberately
                # high validation cap keeps RiskEngine from clipping the proof;
                # the ordinary current loss/NAV/capacity gates are applied below.
                fresh_base_risk = self._validate_intent(
                    intent=verified_intent,
                    snapshot=fresh_snapshot,
                    max_loss_jpy=max(float(max_loss_jpy), 1.0e18),
                    portfolio_loss_cap=None,
                    validate_live_enabled=True,
                    allow_basket_pending=True,
                    portfolio_position_cap=portfolio_position_cap,
                )
            except Exception as exc:
                return blocked(
                    "PRE_POST_GPT_ALLOCATION_BASE_RISK_REMEASURE_FAILED",
                    "fresh GPT allocation proof could not remeasure the original "
                    f"base units with RiskEngine: {type(exc).__name__}: {exc}",
                )

        (
            capital_allocation_edge_recheck,
            capital_allocation_edge_issue,
        ) = _capital_allocation_edge_pre_post_recheck(
            verified_intent,
            ledger_path=self.execution_ledger_db_path,
            expected_edge_basis=(
                expected_capital_allocation_edge_basis
                if numeric_allocation_recheck_required
                else None
            ),
            expected_execution_cost_floor_sha256=(
                expected_execution_cost_floor_sha256
                if numeric_allocation_recheck_required
                else None
            ),
            execution_cost_floor_required=(
                numeric_allocation_recheck_required
            ),
        )
        evidence["capital_allocation_edge_recheck"] = (
            capital_allocation_edge_recheck
        )
        if capital_allocation_edge_issue is not None:
            return blocked(
                capital_allocation_edge_issue.code,
                capital_allocation_edge_issue.message,
            )

        (
            intent,
            tp_proven_fallback_cap_jpy,
            tp_proven_fresh_recheck,
            tp_proven_recheck_issue,
        ) = _tp_proven_pre_post_recheck(
            intent,
            ledger_path=self.execution_ledger_db_path,
        )
        evidence["tp_proven_fresh_recheck"] = tp_proven_fresh_recheck
        if tp_proven_recheck_issue is not None:
            return blocked(
                tp_proven_recheck_issue.code,
                tp_proven_recheck_issue.message,
            )

        macro_event_confidence_sizing = _metadata_truthy(
            (intent.metadata or {}).get("macro_event_confidence_sizing")
        )
        account = fresh_snapshot.account
        fresh_nav = account.nav_jpy if account is not None else None
        try:
            parsed_fresh_nav = float(fresh_nav)
        except (TypeError, ValueError):
            parsed_fresh_nav = math.nan
        if not math.isfinite(parsed_fresh_nav) or parsed_fresh_nav <= 0.0:
            issue_code = (
                "PRE_POST_MACRO_EVENT_FRESH_NAV_MISSING"
                if macro_event_confidence_sizing
                else "PRE_POST_FRESH_NAV_MISSING"
            )
            fresh_nav_recheck = {
                "status": "BLOCKED",
                "fresh_nav_jpy": fresh_nav,
                "max_risk_pct_nav": FRESH_ENTRY_MAX_RISK_PCT_NAV,
                "issue_code": issue_code,
            }
            evidence["fresh_nav_recheck"] = fresh_nav_recheck
            if macro_event_confidence_sizing:
                evidence["macro_event_fresh_nav_recheck"] = fresh_nav_recheck
            return blocked(
                issue_code,
                "fresh entry sizing requires finite positive NAV from the same "
                "broker snapshot used immediately before POST",
            )
        fresh_nav_cap_jpy = parsed_fresh_nav * (
            FRESH_ENTRY_MAX_RISK_PCT_NAV / 100.0
        )
        evidence["fresh_nav_recheck"] = {
            "status": "APPLIED",
            "fresh_nav_jpy": parsed_fresh_nav,
            "max_risk_pct_nav": FRESH_ENTRY_MAX_RISK_PCT_NAV,
            "fresh_nav_cap_jpy": fresh_nav_cap_jpy,
        }
        macro_event_fresh_nav_value: float | None = (
            parsed_fresh_nav if macro_event_confidence_sizing else None
        )
        macro_event_fresh_nav_recheck: dict[str, Any] = {
            "status": "NOT_APPLICABLE"
        }

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

        final_max_loss = min(
            float(max_loss_jpy),
            final_per_trade,
            fresh_nav_cap_jpy,
        )
        if tp_proven_fallback_cap_jpy is not None:
            final_max_loss = min(final_max_loss, tp_proven_fallback_cap_jpy)
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
        if tp_proven_fresh_recheck.get("status") == "CLIPPED_TO_AVG_WIN":
            final_issues.append(
                RiskIssue(
                    "PRE_POST_TP_PROOF_REVOKED_CLIPPED_TO_AVG_WIN",
                    "fresh execution-ledger evidence no longer satisfies the exact "
                    "pair/side/method/vehicle TP relaxation contract; final risk was "
                    "clipped to the observed average-winner cap before POST",
                    "WARN",
                )
            )
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
        m15_recovery_early = validate_m15_recovery_micro_live_claim(
            final_intent,
            fresh_snapshot,
            pair_charts_path=self.pair_charts_path,
            forecast_history_path=self.forecast_history_path,
            validation_time_utc=fresh_snapshot.fetched_at_utc,
        )
        evidence["m15_recovery_micro_recheck"] = m15_recovery_early.evidence
        final_issues.extend(m15_recovery_early.issues)
        m15_recovery_prebounded_allowed = bool(
            m15_recovery_early.applicable and m15_recovery_early.allowed
        )
        if (
            m15_recovery_early.applicable
            and abs(final_intent.units) > abs(intent.units)
        ):
            final_issues.append(
                RiskIssue(
                    "M15_RECOVERY_GATEWAY_UPSIZE_FORBIDDEN",
                    "pre-POST reconciliation may reduce M15 recovery units for current capacity but must never increase producer units",
                )
            )
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
        if numeric_allocation_recheck_required:
            # Bind the path proof to the exact reconciled rails and units that
            # will be reserved, after every loss-cap/attachment/target refresh.
            fresh_forecast_s5_path_proof = _forecast_s5_no_touch_proof(
                self.client,
                intent=final_intent,
                snapshot=fresh_snapshot,
                m15_recovery_prebounded=m15_recovery_prebounded_allowed,
            )
        (
            capital_allocation_numeric_recheck,
            capital_allocation_numeric_issue,
        ) = _capital_allocation_numeric_pre_post_recheck(
            verified_intent=verified_intent,
            final_intent=final_intent,
            fresh_snapshot=fresh_snapshot,
            fresh_base_risk=fresh_base_risk,
            required=numeric_allocation_recheck_required,
            forecast_s5_path_proof=fresh_forecast_s5_path_proof,
            execution_cost_floor=(
                capital_allocation_edge_recheck.get("execution_cost_floor")
                if isinstance(capital_allocation_edge_recheck, dict)
                else None
            ),
            capital_allocation_edge_recheck=(
                capital_allocation_edge_recheck
            ),
            m15_recovery_prebounded=m15_recovery_prebounded_allowed,
        )
        if (
            numeric_allocation_recheck_required
            and capital_allocation_numeric_issue is None
        ):
            price_bound_proof, price_bound_issue = (
                _capital_allocation_market_price_bound(
                    intent=final_intent,
                    snapshot=fresh_snapshot,
                    effective_max_loss_jpy=final_max_loss,
                    execution_cost_floor=(
                        capital_allocation_edge_recheck.get(
                            "execution_cost_floor"
                        )
                        if isinstance(
                            capital_allocation_edge_recheck, dict
                        )
                        else None
                    ),
                    portfolio_loss_remaining_jpy=(
                        _portfolio_loss_remaining_jpy(
                            snapshot=fresh_snapshot,
                            portfolio_loss_cap=final_portfolio_cap,
                            cumulative_risk_jpy=0.0,
                        )
                    ),
                    m15_recovery_prebounded=m15_recovery_prebounded_allowed,
                )
            )
            if price_bound_issue is not None:
                final_issues.append(price_bound_issue)
            elif price_bound_proof.get("status") == "PASSED":
                price_bound = float(price_bound_proof["price_bound"])
                final_order_request = {
                    **final_order_request,
                    "priceBound": price_bound_proof["price_bound_text"],
                }
                bound_snapshot = _snapshot_at_market_price_bound(
                    fresh_snapshot,
                    intent=final_intent,
                    price_bound=price_bound,
                )
                bound_verified_intent = replace(
                    verified_intent,
                    entry=price_bound,
                )
                bound_final_intent = replace(
                    final_intent,
                    entry=price_bound,
                )
                bound_base_risk = self._validate_intent(
                    intent=bound_verified_intent,
                    snapshot=bound_snapshot,
                    max_loss_jpy=1.0e18,
                    portfolio_loss_cap=None,
                    validate_live_enabled=True,
                    allow_basket_pending=True,
                    portfolio_position_cap=portfolio_position_cap,
                )
                bound_final_risk = self._validate_intent(
                    intent=bound_final_intent,
                    snapshot=bound_snapshot,
                    max_loss_jpy=final_max_loss,
                    portfolio_loss_cap=None,
                    validate_live_enabled=True,
                    allow_basket_pending=True,
                    portfolio_position_cap=portfolio_position_cap,
                )
                bound_risk_issues = list(bound_final_risk.issues)
                bound_risk_issues.extend(
                    _basket_issues(
                        intent=bound_final_intent,
                        snapshot=bound_snapshot,
                        metrics=bound_final_risk.metrics,
                        portfolio_loss_cap=final_portfolio_cap,
                        cumulative_risk_jpy=0.0,
                        cumulative_margin_jpy=0.0,
                        seen_geometry=set(),
                    )
                )
                bound_attached_stop = _attached_stop_risk_metrics(
                    bound_final_intent,
                    final_order_request,
                    bound_final_risk.metrics,
                )
                bound_outcome_cost_jpy = float(
                    price_bound_proof.get("outcome_cost_jpy") or 0.0
                )
                if bound_attached_stop is not None:
                    bound_attached_stop = {
                        **bound_attached_stop,
                        "execution_outcome_cost_jpy": (
                            bound_outcome_cost_jpy
                        ),
                        "cost_adjusted_total_downside_jpy": (
                            float(bound_attached_stop["risk_jpy"])
                            + bound_outcome_cost_jpy
                        ),
                    }
                    if bound_attached_stop.get("basis") == "DISASTER_SL":
                        bound_disaster_remaining = (
                            _portfolio_loss_remaining_jpy(
                                snapshot=bound_snapshot,
                                portfolio_loss_cap=final_portfolio_cap,
                                cumulative_risk_jpy=0.0,
                            )
                        )
                        if (
                            bound_disaster_remaining is not None
                            and float(
                                bound_attached_stop[
                                    "cost_adjusted_total_downside_jpy"
                                ]
                            )
                            > bound_disaster_remaining
                        ):
                            bound_risk_issues.append(
                                RiskIssue(
                                    "PRICE_BOUND_DISASTER_STOP_PORTFOLIO_CAP_EXCEEDED",
                                    "worst-fill disaster-stop risk exceeds fresh "
                                    "portfolio remaining capacity",
                                )
                            )
                    elif (
                        float(
                            bound_attached_stop[
                                "cost_adjusted_total_downside_jpy"
                            ]
                        )
                        > final_max_loss
                    ):
                        bound_risk_issues.append(
                            RiskIssue(
                                "PRICE_BOUND_ATTACHED_STOP_LOSS_CAP_EXCEEDED",
                                "worst-fill attached intent-stop risk exceeds "
                                "the fresh effective loss/NAV cap",
                            )
                        )
                bound_blockers = [
                    issue for issue in bound_risk_issues
                    if issue.severity == "BLOCK"
                ]
                if bound_blockers:
                    final_issues.append(
                        RiskIssue(
                            "PRE_POST_GPT_ALLOCATION_PRICE_BOUND_RISK_RECHECK_FAILED",
                            "worst-fill priceBound failed fresh RiskEngine/basket "
                            "validation: "
                            + ", ".join(issue.code for issue in bound_blockers),
                        )
                    )
                bound_material = dict(price_bound_proof)
                bound_material.pop("proof_sha256", None)
                bound_material.update(
                    {
                        "risk_engine_status": (
                            "BLOCKED" if bound_blockers else "PASSED"
                        ),
                        "risk_engine_metrics": (
                            asdict(bound_final_risk.metrics)
                            if isinstance(bound_final_risk.metrics, RiskMetrics)
                            else None
                        ),
                        "risk_engine_entry_basis": (
                            "SYNTHETIC_WORST_FILL_ONLY_RESERVED_INTENT_UNCHANGED"
                        ),
                        "risk_engine_blocking_codes": [
                            issue.code for issue in bound_blockers
                        ],
                        "take_profit_attached": isinstance(
                            final_order_request.get("takeProfitOnFill"),
                            dict,
                        ),
                        "stop_loss_attached": isinstance(
                            final_order_request.get("stopLossOnFill"),
                            dict,
                        ),
                        "attached_stop_risk": bound_attached_stop,
                    }
                )
                price_bound_proof = {
                    **bound_material,
                    "proof_sha256": _canonical_json_sha256(bound_material),
                }
                (
                    bound_numeric_recheck,
                    bound_numeric_issue,
                ) = _capital_allocation_numeric_pre_post_recheck(
                    verified_intent=bound_verified_intent,
                    final_intent=bound_final_intent,
                    fresh_snapshot=bound_snapshot,
                    fresh_base_risk=bound_base_risk,
                    required=True,
                    forecast_s5_path_proof=fresh_forecast_s5_path_proof,
                    execution_cost_floor=(
                        capital_allocation_edge_recheck.get(
                            "execution_cost_floor"
                        )
                        if isinstance(
                            capital_allocation_edge_recheck, dict
                        )
                        else None
                    ),
                    capital_allocation_edge_recheck=(
                        capital_allocation_edge_recheck
                    ),
                    market_entry_slippage_embedded=True,
                )
                capital_allocation_numeric_recheck = (
                    _numeric_proof_with_market_price_bound(
                        bound_numeric_recheck,
                        price_bound_proof,
                    )
                )
                if bound_numeric_issue is not None:
                    final_issues.append(
                        RiskIssue(
                            "PRE_POST_GPT_ALLOCATION_PRICE_BOUND_NUMERIC_REPROOF_FAILED",
                            "worst-fill priceBound no longer preserves the "
                            "floor-unit Wilson EV/quarter-Kelly allocation: "
                            f"{bound_numeric_issue.message}",
                        )
                    )
            else:
                # HEDGE and the strictly pre-bounded predictive SCOUT contract
                # do not acquire a new MARKET priceBound, but their bypass is
                # still frozen into the numeric proof.
                capital_allocation_numeric_recheck = (
                    _numeric_proof_with_market_price_bound(
                        capital_allocation_numeric_recheck,
                        price_bound_proof,
                    )
                )
        evidence["capital_allocation_numeric_recheck"] = (
            capital_allocation_numeric_recheck
        )
        if capital_allocation_numeric_issue is not None:
            final_issues.append(capital_allocation_numeric_issue)
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
                "portfolio_position_cap": portfolio_position_cap,
                "final_units": final_intent.units,
                "macro_event_fresh_nav_recheck": macro_event_fresh_nav_recheck,
                "tp_proven_fresh_recheck": tp_proven_fresh_recheck,
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

    def _final_pre_post_boundary(
        self,
        *,
        verified_intent: OrderIntent,
        final_intent: OrderIntent,
        early_reconciliation: dict[str, Any],
        expected_receipt_sha256: str | None,
        expected_capital_allocation_edge_basis: str | None,
        expected_execution_cost_floor_sha256: str | None,
        expected_order_request_sha256: str | None,
        order_request: dict[str, Any],
        ordinary_entry_claim: dict[str, Any],
        expected_guardian_action_receipt_material_contract: str | None = None,
        expected_guardian_action_receipt_baseline_pairs: tuple[str, ...] = (),
        expected_guardian_action_receipt_scope_state_sha256: str | None = None,
    ) -> _FinalPrePostBoundaryResult:
        """Fence reservations from the broker POST with immutable current truth.

        The earlier reconciliation is allowed to rebuild and downsize an order.
        Once a durable one-shot reservation exists, this boundary is validation
        only: it never changes units, switches the allocation basis, or releases
        the reservation. Any broker transaction after the earlier risk snapshot
        invalidates that snapshot and requires a genuinely fresh signal.
        """

        clear_guardian_tuning_validation_cache()

        early_edge = (
            early_reconciliation.get("capital_allocation_edge_recheck")
            if isinstance(
                early_reconciliation.get("capital_allocation_edge_recheck"),
                dict,
            )
            else {}
        )
        early_numeric = (
            early_reconciliation.get("capital_allocation_numeric_recheck")
            if isinstance(
                early_reconciliation.get("capital_allocation_numeric_recheck"),
                dict,
            )
            else {}
        )
        early_numeric_required_value = early_numeric.get("required")
        early_numeric_required_marker_valid = isinstance(
            early_numeric_required_value,
            bool,
        )
        # Do not reread/reclassify the receipt after the early proof. A missing
        # or malformed marker is treated as required and then blocked below.
        numeric_allocation_required = (
            early_numeric_required_value
            if early_numeric_required_marker_valid
            else True
        )
        early_numeric_frozen = _capital_allocation_numeric_proof_is_frozen(
            early_numeric,
            order_request=order_request,
            required=numeric_allocation_required,
            expected_intent=final_intent,
        )
        early_price_bound = (
            early_numeric.get("market_price_bound")
            if isinstance(early_numeric.get("market_price_bound"), dict)
            else {}
        )
        expected_reserved_price_bound = (
            str(early_price_bound.get("price_bound_text") or "")
            if numeric_allocation_required
            and str(early_price_bound.get("status") or "").upper()
            == "PASSED"
            else None
        )
        early_ledger_transaction_id = str(
            early_reconciliation.get("ledger_last_transaction_id") or ""
        ).strip()
        early_broker_transaction_id = str(
            early_reconciliation.get("broker_last_transaction_id") or ""
        ).strip()
        reservation_kind = (
            "PREDICTIVE_SCOUT"
            if predictive_scout_intent_claimed(verified_intent)
            else "ORDINARY_ENTRY"
            if ordinary_entry_claim.get("claim_id")
            else "NONE"
        )
        evidence: dict[str, Any] = {
            "status": "BLOCKED",
            "execution_ledger_db_path": (
                str(self.execution_ledger_db_path)
                if self.execution_ledger_db_path is not None
                else None
            ),
            "early_ledger_last_transaction_id": (
                early_ledger_transaction_id or None
            ),
            "early_broker_last_transaction_id": (
                early_broker_transaction_id or None
            ),
            "early_edge_basis": early_edge.get("basis"),
            "early_edge_proof_key": early_edge.get("proof_key"),
            "early_numeric_proof_sha256": early_numeric.get("proof_sha256"),
            "early_numeric_proof_status": early_numeric.get("status"),
            "early_numeric_proof_frozen": early_numeric_frozen,
            "numeric_allocation_recheck_required": numeric_allocation_required,
            "reservation_kind": reservation_kind,
            "reservation_retained_on_block": reservation_kind != "NONE",
            "post_attempted": False,
        }
        issues: list[RiskIssue] = []

        if not early_numeric_required_marker_valid:
            issues.append(
                RiskIssue(
                    "FINAL_PRE_POST_NUMERIC_REQUIREMENT_MARKER_INVALID",
                    "early reconciliation did not freeze an exact boolean numeric "
                    "allocation requirement; do not downgrade unreadable receipt "
                    "state to NOT_APPLICABLE",
                )
            )

        if not early_numeric_frozen:
            issues.append(
                RiskIssue(
                    "FINAL_PRE_POST_EARLY_NUMERIC_ALLOCATION_PROOF_INVALID",
                    "post-reservation broker fence requires an immutable early "
                    "fresh-NAV/quote/base-risk numeric allocation proof whose "
                    "floor-unit cap matches the reserved order",
                )
            )

        if (
            str(early_reconciliation.get("status") or "").upper() != "PASSED"
            or not early_ledger_transaction_id
            or early_ledger_transaction_id != early_broker_transaction_id
            or str(early_edge.get("status") or "").upper()
            not in {"PASSED", "BYPASSED"}
        ):
            issues.append(
                RiskIssue(
                    "FINAL_PRE_POST_EARLY_RECONCILIATION_INVALID",
                    "post-reservation broker fence requires the earlier reconciliation, "
                    "transaction-id match, and allocation edge proof to have passed",
                )
            )

        sync_summary = None
        if self.execution_ledger_db_path is None:
            issues.append(
                RiskIssue(
                    "FINAL_PRE_POST_LEDGER_PATH_MISSING",
                    "post-reservation broker fence requires the canonical execution ledger",
                )
            )
        else:
            report_path = self.execution_ledger_report_path or (
                self.execution_ledger_db_path.with_suffix(".md")
            )
            try:
                sync_summary = ExecutionLedger(
                    db_path=self.execution_ledger_db_path,
                    report_path=report_path,
                ).sync_oanda_transactions(self.client)
            except Exception as exc:
                issues.append(
                    RiskIssue(
                        "FINAL_PRE_POST_LEDGER_SYNC_FAILED",
                        "execution ledger re-sync failed after durable reservation and "
                        f"before broker POST: {type(exc).__name__}: {exc}",
                    )
                )

        final_ledger_transaction_id = ""
        if sync_summary is not None:
            final_ledger_transaction_id = str(
                sync_summary.last_transaction_id or ""
            ).strip()
            evidence.update(
                {
                    "final_ledger_sync_status": sync_summary.status,
                    "final_ledger_last_transaction_id": (
                        final_ledger_transaction_id or None
                    ),
                }
            )
            if sync_summary.status != "SYNCED" or not final_ledger_transaction_id:
                issues.append(
                    RiskIssue(
                        "FINAL_PRE_POST_LEDGER_NOT_SYNCED",
                        "post-reservation execution ledger must finish SYNCED with a "
                        "non-empty lastTransactionID before broker POST",
                    )
                )

        final_edge: dict[str, Any] = {"status": "NOT_RUN"}
        if (
            sync_summary is not None
            and sync_summary.status == "SYNCED"
            and final_ledger_transaction_id
            and self.execution_ledger_db_path is not None
        ):
            try:
                final_edge, final_edge_issue = (
                    _capital_allocation_edge_pre_post_recheck(
                        verified_intent,
                        ledger_path=self.execution_ledger_db_path,
                        expected_edge_basis=(
                            expected_capital_allocation_edge_basis
                            if numeric_allocation_required
                            else None
                        ),
                        expected_execution_cost_floor_sha256=(
                            expected_execution_cost_floor_sha256
                            if numeric_allocation_required
                            else None
                        ),
                        execution_cost_floor_required=(
                            numeric_allocation_required
                        ),
                    )
                )
            except Exception as exc:
                final_edge_issue = RiskIssue(
                    "FINAL_PRE_POST_GPT_ALLOCATION_EDGE_RECHECK_FAILED",
                    "post-reservation same-basis allocation reproof raised before "
                    f"broker POST: {type(exc).__name__}: {exc}",
                )
                final_edge = {
                    "status": "BLOCKED",
                    "issue_code": final_edge_issue.code,
                }
            if final_edge_issue is not None:
                issues.append(final_edge_issue)
            if (
                final_edge.get("basis") != early_edge.get("basis")
                or final_edge.get("proof_key") != early_edge.get("proof_key")
            ):
                issues.append(
                    RiskIssue(
                        "FINAL_PRE_POST_GPT_ALLOCATION_EDGE_BASIS_CHANGED",
                        "post-reservation allocation reproof must preserve the exact "
                        "earlier evidence basis and pair/side/method/vehicle proof key",
                    )
                )
        evidence["capital_allocation_edge_recheck"] = final_edge

        final_numeric: dict[str, Any] = {"status": "NOT_APPLICABLE"}
        final_numeric_snapshot_transaction_id = ""
        final_numeric_frozen = not numeric_allocation_required
        final_reserved_risk_evidence: dict[str, Any] = {
            "status": "NOT_APPLICABLE"
        }
        final_price_bound_proof: dict[str, Any] = {
            "status": "NOT_APPLICABLE"
        }
        final_bound_snapshot: BrokerSnapshot | None = None
        final_boundary_snapshot: BrokerSnapshot | None = None
        final_m15_snapshot: BrokerSnapshot | None = None
        final_reserved_intent: OrderIntent | None = None
        final_numeric_m15_recovery_allowed = False
        try:
            final_boundary_snapshot = self.client.snapshot(
                (verified_intent.pair,)
            )
            final_m15_snapshot = final_boundary_snapshot
        except Exception as exc:
            issues.append(
                RiskIssue(
                    "FINAL_PRE_POST_BROKER_SNAPSHOT_FAILED",
                    "post-reservation final broker snapshot failed before "
                    f"POST: {type(exc).__name__}: {exc}",
                )
            )
            if numeric_allocation_required:
                final_numeric = {
                    "status": "BLOCKED",
                    "issue_code": "FINAL_PRE_POST_BROKER_SNAPSHOT_FAILED",
                }
                final_numeric_frozen = False
        if final_boundary_snapshot is not None:
            final_numeric_snapshot_transaction_id = (
                str(
                    final_boundary_snapshot.account.last_transaction_id
                    if final_boundary_snapshot.account is not None
                    else ""
                ).strip()
            )
            try:
                final_reserved_intent = _intent_from_reserved_order_request(
                    final_intent,
                    order_request=order_request,
                    expected_price_bound=expected_reserved_price_bound,
                )
                raw_final_max_loss = early_reconciliation.get(
                    "final_max_loss_jpy"
                )
                if (
                    isinstance(raw_final_max_loss, bool)
                    or not isinstance(raw_final_max_loss, (int, float))
                    or not math.isfinite(float(raw_final_max_loss))
                    or float(raw_final_max_loss) <= 0.0
                ):
                    raise ValueError(
                        "early reconciliation has no finite positive final_max_loss_jpy"
                    )
                final_account = final_boundary_snapshot.account
                final_nav_jpy = (
                    float(final_account.nav_jpy)
                    if final_account is not None
                    and isinstance(final_account.nav_jpy, (int, float))
                    and not isinstance(final_account.nav_jpy, bool)
                    and math.isfinite(float(final_account.nav_jpy))
                    and float(final_account.nav_jpy) > 0.0
                    else None
                )
                if final_nav_jpy is None:
                    raise ValueError(
                        "final broker snapshot has no finite positive NAV"
                    )
                final_nav_hard_cap_jpy = final_nav_jpy * (
                    FRESH_ENTRY_MAX_RISK_PCT_NAV / 100.0
                )
                final_reserved_max_loss_jpy = min(
                    float(raw_final_max_loss),
                    final_nav_hard_cap_jpy,
                )
                macro_event_confidence_sizing = _metadata_truthy(
                    (verified_intent.metadata or {}).get(
                        "macro_event_confidence_sizing"
                    )
                )
                final_macro_nav_cap_jpy = (
                    final_nav_jpy * (MACRO_EVENT_MAX_RISK_PCT_NAV / 100.0)
                    if macro_event_confidence_sizing
                    else None
                )
                if final_macro_nav_cap_jpy is not None:
                    final_reserved_max_loss_jpy = min(
                        final_reserved_max_loss_jpy,
                        final_macro_nav_cap_jpy,
                    )
                risk_validation_snapshot = final_boundary_snapshot
                risk_validation_intent = final_reserved_intent
                if m15_recovery_micro_claimed(final_reserved_intent):
                    final_numeric_m15_recovery_allowed = (
                        validate_m15_recovery_micro_live_claim(
                            final_reserved_intent,
                            final_boundary_snapshot,
                            pair_charts_path=self.pair_charts_path,
                            forecast_history_path=self.forecast_history_path,
                            validation_time_utc=(
                                final_boundary_snapshot.fetched_at_utc
                            ),
                        ).allowed
                    )
                if numeric_allocation_required:
                    (
                        final_price_bound_proof,
                        final_price_bound_issue,
                    ) = _capital_allocation_market_price_bound(
                        intent=final_reserved_intent,
                        snapshot=final_boundary_snapshot,
                        effective_max_loss_jpy=final_reserved_max_loss_jpy,
                        reserved_price_bound=order_request.get("priceBound"),
                        execution_cost_floor=(
                            final_edge.get("execution_cost_floor")
                            if isinstance(final_edge, dict)
                            else None
                        ),
                        portfolio_loss_remaining_jpy=(
                            _portfolio_loss_remaining_jpy(
                                snapshot=final_boundary_snapshot,
                                portfolio_loss_cap=(
                                    float(
                                        early_reconciliation.get(
                                            "final_portfolio_loss_cap_jpy"
                                        )
                                    )
                                    if isinstance(
                                        early_reconciliation.get(
                                            "final_portfolio_loss_cap_jpy"
                                        ),
                                        (int, float),
                                    )
                                    and not isinstance(
                                        early_reconciliation.get(
                                            "final_portfolio_loss_cap_jpy"
                                        ),
                                        bool,
                                    )
                                    else None
                                ),
                                cumulative_risk_jpy=0.0,
                            )
                        ),
                        m15_recovery_prebounded=(
                            final_numeric_m15_recovery_allowed
                        ),
                    )
                    if final_price_bound_issue is not None:
                        issues.append(
                            RiskIssue(
                                "FINAL_PRE_POST_GPT_ALLOCATION_PRICE_BOUND_REPROOF_FAILED",
                                final_price_bound_issue.message,
                            )
                        )
                    elif final_price_bound_proof.get("status") == "PASSED":
                        final_bound_snapshot = _snapshot_at_market_price_bound(
                            final_boundary_snapshot,
                            intent=final_reserved_intent,
                            price_bound=float(
                                final_price_bound_proof["price_bound"]
                            ),
                        )
                        risk_validation_snapshot = final_bound_snapshot
                        risk_validation_intent = replace(
                            final_reserved_intent,
                            entry=float(final_price_bound_proof["price_bound"]),
                        )
                raw_position_cap = early_reconciliation.get(
                    "portfolio_position_cap"
                )
                final_position_cap = (
                    raw_position_cap
                    if isinstance(raw_position_cap, int)
                    and not isinstance(raw_position_cap, bool)
                    and raw_position_cap > 0
                    else _portfolio_position_cap_from_state()
                )
                final_reserved_risk = self._validate_intent(
                    intent=risk_validation_intent,
                    snapshot=risk_validation_snapshot,
                    max_loss_jpy=final_reserved_max_loss_jpy,
                    portfolio_loss_cap=None,
                    validate_live_enabled=True,
                    allow_basket_pending=True,
                    portfolio_position_cap=final_position_cap,
                )
                final_reserved_issues = list(final_reserved_risk.issues)
                raw_final_portfolio_cap = early_reconciliation.get(
                    "final_portfolio_loss_cap_jpy"
                )
                final_portfolio_cap = (
                    float(raw_final_portfolio_cap)
                    if isinstance(raw_final_portfolio_cap, (int, float))
                    and not isinstance(raw_final_portfolio_cap, bool)
                    and math.isfinite(float(raw_final_portfolio_cap))
                    and float(raw_final_portfolio_cap) >= 0.0
                    else None
                )
                final_reserved_issues.extend(
                    _basket_issues(
                        intent=risk_validation_intent,
                        snapshot=risk_validation_snapshot,
                        metrics=final_reserved_risk.metrics,
                        portfolio_loss_cap=final_portfolio_cap,
                        cumulative_risk_jpy=0.0,
                        cumulative_margin_jpy=0.0,
                        seen_geometry=set(),
                    )
                )
                final_attached_stop = _attached_stop_risk_metrics(
                    risk_validation_intent,
                    order_request,
                    final_reserved_risk.metrics,
                )
                final_outcome_cost_jpy = float(
                    final_price_bound_proof.get("outcome_cost_jpy") or 0.0
                )
                if final_attached_stop is not None:
                    final_attached_stop = {
                        **final_attached_stop,
                        "execution_outcome_cost_jpy": (
                            final_outcome_cost_jpy
                        ),
                        "cost_adjusted_total_downside_jpy": (
                            float(final_attached_stop["risk_jpy"])
                            + final_outcome_cost_jpy
                        ),
                    }
                    if final_attached_stop.get("basis") == "DISASTER_SL":
                        final_disaster_remaining = (
                            _portfolio_loss_remaining_jpy(
                                snapshot=risk_validation_snapshot,
                                portfolio_loss_cap=final_portfolio_cap,
                                cumulative_risk_jpy=0.0,
                            )
                        )
                        if (
                            final_disaster_remaining is not None
                            and float(
                                final_attached_stop[
                                    "cost_adjusted_total_downside_jpy"
                                ]
                            )
                            > final_disaster_remaining
                        ):
                            final_reserved_issues.append(
                                RiskIssue(
                                    "DISASTER_STOP_PORTFOLIO_CAP_EXCEEDED",
                                    "post-reservation disaster-stop risk "
                                    f"{float(final_attached_stop['risk_jpy']):.0f} JPY "
                                    "exceeds final portfolio remaining capacity "
                                    f"{final_disaster_remaining:.0f} JPY",
                                )
                            )
                    elif (
                        float(
                            final_attached_stop[
                                "cost_adjusted_total_downside_jpy"
                            ]
                        )
                        > final_reserved_max_loss_jpy
                    ):
                        final_reserved_issues.append(
                            RiskIssue(
                                "ATTACHED_STOP_LOSS_CAP_EXCEEDED",
                                "post-reservation attached intent-stop risk "
                                f"{float(final_attached_stop['risk_jpy']):.0f} JPY "
                                "exceeds final effective max loss "
                                f"{final_reserved_max_loss_jpy:.0f} JPY",
                            )
                        )
                final_reserved_blockers = [
                    issue
                    for issue in final_reserved_issues
                    if issue.severity == "BLOCK"
                ]
                if final_price_bound_proof.get("status") == "PASSED":
                    bound_material = dict(final_price_bound_proof)
                    bound_material.pop("proof_sha256", None)
                    bound_material.update(
                        {
                            "risk_engine_status": (
                                "BLOCKED"
                                if final_reserved_blockers
                                else "PASSED"
                            ),
                            "risk_engine_metrics": (
                                asdict(final_reserved_risk.metrics)
                                if isinstance(
                                    final_reserved_risk.metrics,
                                    RiskMetrics,
                                )
                                else None
                            ),
                            "risk_engine_entry_basis": (
                                "SYNTHETIC_WORST_FILL_ONLY_RESERVED_INTENT_UNCHANGED"
                            ),
                            "risk_engine_blocking_codes": [
                                issue.code
                                for issue in final_reserved_blockers
                            ],
                            "take_profit_attached": isinstance(
                                order_request.get("takeProfitOnFill"),
                                dict,
                            ),
                            "stop_loss_attached": isinstance(
                                order_request.get("stopLossOnFill"),
                                dict,
                            ),
                            "attached_stop_risk": final_attached_stop,
                        }
                    )
                    final_price_bound_proof = {
                        **bound_material,
                        "proof_sha256": _canonical_json_sha256(
                            bound_material
                        ),
                    }
                final_reserved_metrics = (
                    asdict(final_reserved_risk.metrics)
                    if isinstance(final_reserved_risk.metrics, RiskMetrics)
                    else None
                )
                final_reserved_risk_jpy = (
                    final_reserved_risk.metrics.risk_jpy
                    if isinstance(final_reserved_risk.metrics, RiskMetrics)
                    else None
                )
                final_reserved_material = {
                    "status": (
                        "BLOCKED" if final_reserved_blockers else "PASSED"
                    ),
                    "snapshot_fetched_at_utc": (
                        final_boundary_snapshot.fetched_at_utc.isoformat()
                    ),
                    "quote_timestamp_utc": (
                        final_boundary_snapshot.quotes[
                            verified_intent.pair
                        ].timestamp_utc.isoformat()
                        if verified_intent.pair
                        in final_boundary_snapshot.quotes
                        else None
                    ),
                    "risk_quote_basis": (
                        "RESERVED_PRICE_BOUND_WORST_FILL"
                        if final_bound_snapshot is not None
                        else "FINAL_EXECUTABLE_QUOTE"
                    ),
                    "actual_executable_entry": (
                        final_boundary_snapshot.quotes[
                            final_reserved_intent.pair
                        ].ask
                        if final_reserved_intent.side == Side.LONG
                        else final_boundary_snapshot.quotes[
                            final_reserved_intent.pair
                        ].bid
                    ),
                    "worst_fill_entry": (
                        final_price_bound_proof.get("price_bound")
                        if final_bound_snapshot is not None
                        else None
                    ),
                    "pair": final_reserved_intent.pair,
                    "side": final_reserved_intent.side.value,
                    "order_type": final_reserved_intent.order_type.value,
                    "units": abs(final_reserved_intent.units),
                    "entry": final_reserved_intent.entry,
                    "tp": final_reserved_intent.tp,
                    "sl": final_reserved_intent.sl,
                    "final_nav_jpy": final_nav_jpy,
                    "fresh_entry_max_risk_pct_nav": (
                        FRESH_ENTRY_MAX_RISK_PCT_NAV
                    ),
                    "final_nav_hard_cap_jpy": final_nav_hard_cap_jpy,
                    "final_macro_nav_cap_jpy": final_macro_nav_cap_jpy,
                    "early_final_max_loss_jpy": float(raw_final_max_loss),
                    "reserved_effective_max_loss_jpy": (
                        final_reserved_max_loss_jpy
                    ),
                    "reserved_risk_nav_pct": (
                        final_reserved_risk_jpy / final_nav_jpy * 100.0
                        if final_reserved_risk_jpy is not None
                        else None
                    ),
                    "max_loss_cap_basis": (
                        "MIN_EARLY_FINAL_AND_FINAL_NAV_HARD_CAP_AND_OPTIONAL_MACRO_NAV_CAP"
                    ),
                    "portfolio_position_cap": final_position_cap,
                    "final_portfolio_loss_cap_jpy": final_portfolio_cap,
                    "metrics": final_reserved_metrics,
                    "take_profit_attached": (
                        isinstance(
                            order_request.get("takeProfitOnFill"),
                            dict,
                        )
                    ),
                    "stop_loss_attached": (
                        isinstance(
                            order_request.get("stopLossOnFill"),
                            dict,
                        )
                    ),
                    "attached_stop_risk": final_attached_stop,
                    "market_price_bound": final_price_bound_proof,
                    "blocking_codes": [
                        issue.code for issue in final_reserved_blockers
                    ],
                }
                final_reserved_risk_evidence = {
                    **final_reserved_material,
                    "proof_sha256": _canonical_json_sha256(
                        final_reserved_material
                    ),
                }
                if final_reserved_risk.metrics is None:
                    issues.append(
                        RiskIssue(
                            "FINAL_PRE_POST_RESERVED_RISK_METRICS_MISSING",
                            "post-reservation exact reserved order has no fresh "
                            "RiskMetrics; broker POST is forbidden",
                        )
                    )
                issues.extend(final_reserved_blockers)
            except Exception as exc:
                issues.append(
                    RiskIssue(
                        "FINAL_PRE_POST_RESERVED_RISK_RECHECK_FAILED",
                        "post-reservation exact broker order could not be rebuilt "
                        f"and risk-validated: {type(exc).__name__}: {exc}",
                    )
                )
                final_reserved_risk_evidence = {
                    "status": "BLOCKED",
                    "issue_code": (
                        "FINAL_PRE_POST_RESERVED_RISK_RECHECK_FAILED"
                    ),
                }

            if numeric_allocation_required:
                if (
                    not final_numeric_snapshot_transaction_id
                    or final_numeric_snapshot_transaction_id
                    != final_ledger_transaction_id
                ):
                    issues.append(
                        RiskIssue(
                            "FINAL_PRE_POST_NUMERIC_SNAPSHOT_TRANSACTION_ID_MISMATCH",
                            "post-reservation numeric snapshot lastTransactionID "
                            f"{final_numeric_snapshot_transaction_id or 'missing'} does not "
                            "match the just-synced execution ledger "
                            f"{final_ledger_transaction_id or 'missing'}",
                        )
                    )
                try:
                    if final_reserved_intent is None:
                        raise ValueError(
                            "exact reserved order intent is unavailable"
                        )
                    numeric_reproof_snapshot = (
                        final_bound_snapshot
                        if final_bound_snapshot is not None
                        else final_boundary_snapshot
                    )
                    numeric_verified_intent = (
                        replace(
                            verified_intent,
                            entry=float(final_price_bound_proof["price_bound"]),
                        )
                        if final_bound_snapshot is not None
                        else verified_intent
                    )
                    numeric_final_intent = (
                        replace(
                            final_reserved_intent,
                            entry=float(final_price_bound_proof["price_bound"]),
                        )
                        if final_bound_snapshot is not None
                        else final_reserved_intent
                    )
                    final_base_risk = self._validate_intent(
                        intent=numeric_verified_intent,
                        snapshot=numeric_reproof_snapshot,
                        max_loss_jpy=1.0e18,
                        portfolio_loss_cap=None,
                        validate_live_enabled=True,
                        allow_basket_pending=True,
                        portfolio_position_cap=(
                            _portfolio_position_cap_from_state()
                        ),
                    )
                    final_forecast_s5_path_proof = (
                        _forecast_s5_no_touch_proof(
                            self.client,
                            intent=final_reserved_intent,
                            snapshot=final_boundary_snapshot,
                            m15_recovery_prebounded=(
                                final_numeric_m15_recovery_allowed
                            ),
                        )
                    )
                    final_numeric, final_numeric_issue = (
                        _capital_allocation_numeric_pre_post_recheck(
                            verified_intent=numeric_verified_intent,
                            final_intent=numeric_final_intent,
                            fresh_snapshot=numeric_reproof_snapshot,
                            fresh_base_risk=final_base_risk,
                            required=True,
                            forecast_s5_path_proof=(
                                final_forecast_s5_path_proof
                            ),
                            execution_cost_floor=(
                                final_edge.get("execution_cost_floor")
                                if isinstance(final_edge, dict)
                                else None
                            ),
                            capital_allocation_edge_recheck=final_edge,
                            market_entry_slippage_embedded=(
                                final_bound_snapshot is not None
                            ),
                            m15_recovery_prebounded=(
                                final_numeric_m15_recovery_allowed
                            ),
                        )
                    )
                    final_numeric = _numeric_proof_with_market_price_bound(
                        final_numeric,
                        final_price_bound_proof,
                    )
                except Exception as exc:
                    final_numeric_issue = RiskIssue(
                        "FINAL_PRE_POST_GPT_ALLOCATION_NUMERIC_REPROOF_FAILED",
                        "post-reservation numeric allocation reproof raised "
                        f"before broker POST: {type(exc).__name__}: {exc}",
                    )
                    final_numeric = {
                        "status": "BLOCKED",
                        "issue_code": final_numeric_issue.code,
                    }
                if final_numeric_issue is not None:
                    issues.append(
                        RiskIssue(
                            (
                                "FINAL_" + final_numeric_issue.code
                                if final_numeric_issue.code.startswith("PRE_POST_")
                                else final_numeric_issue.code
                            ),
                            final_numeric_issue.message,
                            final_numeric_issue.severity,
                        )
                    )
                final_numeric_frozen = (
                    _capital_allocation_numeric_proof_is_frozen(
                        final_numeric,
                        order_request=order_request,
                        required=True,
                        expected_intent=final_intent,
                    )
                )
                if not final_numeric_frozen:
                    issues.append(
                        RiskIssue(
                            "FINAL_PRE_POST_NUMERIC_ALLOCATION_PROOF_INVALID",
                            "post-reservation fresh numeric allocation proof is "
                            "not content-addressed to the immutable reserved order",
                        )
                    )
        evidence.update(
            {
                "final_numeric_snapshot_last_transaction_id": (
                    final_numeric_snapshot_transaction_id or None
                ),
                "capital_allocation_numeric_recheck": final_numeric,
                "final_numeric_proof_sha256": final_numeric.get(
                    "proof_sha256"
                ),
                "final_numeric_proof_frozen": final_numeric_frozen,
                "final_reserved_risk_recheck": (
                    final_reserved_risk_evidence
                ),
            }
        )

        # S5 is a historical read, but time still passes while its complete
        # candle path and worst-fill proof are verified. Read one coherent
        # broker snapshot afterward so account truth and a passive RANGE quote
        # cannot straddle different moments. Improvements never relax the
        # conservative coherent snapshot used above.
        final_broker_fence: dict[str, Any] = {
            "status": "NOT_APPLICABLE"
        }
        final_range_quote_fence: dict[str, Any] = {
            "status": "NOT_APPLICABLE"
        }
        final_broker_transaction_id = final_numeric_snapshot_transaction_id
        if numeric_allocation_required:
            final_snapshot_account = (
                final_boundary_snapshot.account
                if final_boundary_snapshot is not None
                else None
            )
            try:
                if final_snapshot_account is None:
                    raise ValueError("final coherent snapshot account is missing")
                observed_snapshot = self.client.snapshot(
                    (verified_intent.pair,)
                )
                final_m15_snapshot = observed_snapshot
                observed_account = observed_snapshot.account
                if observed_account is None:
                    raise ValueError("post-S5 broker snapshot account is missing")

                def finite_nonnegative(value: Any) -> float | None:
                    if isinstance(value, bool) or not isinstance(
                        value,
                        (int, float),
                    ):
                        return None
                    parsed = float(value)
                    return (
                        parsed
                        if math.isfinite(parsed) and parsed >= 0.0
                        else None
                    )

                observed_tx = str(
                    getattr(observed_account, "last_transaction_id", "")
                    or ""
                ).strip()
                observed_nav = _positive_float(
                    getattr(observed_account, "nav_jpy", None)
                )
                observed_margin_used = finite_nonnegative(
                    getattr(observed_account, "margin_used_jpy", None)
                )
                observed_margin_available = finite_nonnegative(
                    getattr(observed_account, "margin_available_jpy", None)
                )
                basis_nav = _positive_float(final_snapshot_account.nav_jpy)
                basis_margin_used = finite_nonnegative(
                    final_snapshot_account.margin_used_jpy
                )
                basis_margin_available = finite_nonnegative(
                    final_snapshot_account.margin_available_jpy
                )
                values_valid = None not in (
                    observed_nav,
                    observed_margin_used,
                    observed_margin_available,
                    basis_nav,
                    basis_margin_used,
                    basis_margin_available,
                )
                transaction_matches = bool(
                    observed_tx
                    and observed_tx == final_numeric_snapshot_transaction_id
                    and observed_tx == final_ledger_transaction_id
                )
                nav_worsened = bool(
                    values_valid and observed_nav + 1e-9 < basis_nav
                )
                margin_used_worsened = bool(
                    values_valid
                    and observed_margin_used > basis_margin_used + 1e-9
                )
                margin_available_worsened = bool(
                    values_valid
                    and observed_margin_available + 1e-9
                    < basis_margin_available
                )
                account_fence_passed = bool(
                    values_valid
                    and transaction_matches
                    and not nav_worsened
                    and not margin_used_worsened
                    and not margin_available_worsened
                )
                range_fence_intent = final_reserved_intent or final_intent
                (
                    final_range_quote_fence,
                    final_range_quote_issue,
                ) = _post_s5_tp_proven_range_quote_fence(
                    intent=range_fence_intent,
                    snapshot=observed_snapshot,
                    prior_snapshot=final_boundary_snapshot,
                )
                range_quote_passed = str(
                    final_range_quote_fence.get("status") or ""
                ).upper() in {"PASSED", "NOT_APPLICABLE"}
                fence_passed = account_fence_passed and range_quote_passed
                observed_quote = observed_snapshot.quotes.get(
                    verified_intent.pair
                )
                final_broker_fence = {
                    "status": "PASSED" if fence_passed else "BLOCKED",
                    "read_order": (
                        "AFTER_FINAL_S5_AND_WORST_FILL_REPROOF_"
                        "SAME_BROKER_SNAPSHOT"
                    ),
                    "adopted_risk_basis": (
                        "EARLIER_COHERENT_PAIR_SNAPSHOT_CONSERVATIVE"
                    ),
                    "account_and_quote_same_snapshot": True,
                    "observed_snapshot_fetched_at_utc": (
                        observed_snapshot.fetched_at_utc.isoformat()
                    ),
                    "snapshot_last_transaction_id": (
                        final_numeric_snapshot_transaction_id or None
                    ),
                    "ledger_last_transaction_id": (
                        final_ledger_transaction_id or None
                    ),
                    "observed_last_transaction_id": observed_tx or None,
                    "transaction_matches": transaction_matches,
                    "snapshot_nav_jpy": basis_nav,
                    "observed_nav_jpy": observed_nav,
                    "snapshot_margin_used_jpy": basis_margin_used,
                    "observed_margin_used_jpy": observed_margin_used,
                    "snapshot_margin_available_jpy": basis_margin_available,
                    "observed_margin_available_jpy": (
                        observed_margin_available
                    ),
                    "nav_worsened": nav_worsened,
                    "margin_used_worsened": margin_used_worsened,
                    "margin_available_worsened": margin_available_worsened,
                    "account_fence_passed": account_fence_passed,
                    "range_quote_fence_status": (
                        final_range_quote_fence.get("status")
                    ),
                    "range_quote_fence_proof_sha256": (
                        final_range_quote_fence.get("proof_sha256")
                    ),
                    "observed_quote_bid": (
                        float(observed_quote.bid)
                        if observed_quote is not None
                        else None
                    ),
                    "observed_quote_ask": (
                        float(observed_quote.ask)
                        if observed_quote is not None
                        else None
                    ),
                }
                final_broker_transaction_id = observed_tx
                if final_range_quote_issue is not None:
                    issues.append(final_range_quote_issue)
                if not account_fence_passed:
                    issues.append(
                        RiskIssue(
                            "FINAL_PRE_POST_ACCOUNT_FENCE_FAILED",
                            "the same post-S5 broker snapshot detected changed "
                            "transaction truth, invalid account values, or worse "
                            "NAV/margin; the durable reservation is retained and "
                            "no POST is allowed",
                        )
                    )
            except Exception as exc:
                final_broker_fence = {
                    "status": "BLOCKED",
                    "read_order": (
                        "AFTER_FINAL_S5_AND_WORST_FILL_REPROOF_"
                        "SAME_BROKER_SNAPSHOT"
                    ),
                    "account_and_quote_same_snapshot": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                final_broker_transaction_id = ""
                issues.append(
                    RiskIssue(
                        "FINAL_PRE_POST_ACCOUNT_FENCE_FAILED",
                        "post-S5 account-and-quote broker snapshot fence could not be completed: "
                        f"{type(exc).__name__}: {exc}",
                    )
                )
        evidence["post_s5_broker_fence"] = final_broker_fence
        evidence["post_s5_account_fence"] = final_broker_fence
        evidence["post_s5_tp_proven_range_quote_fence"] = (
            final_range_quote_fence
        )
        evidence["final_broker_last_transaction_id"] = (
            final_broker_transaction_id or None
        )

        if final_ledger_transaction_id and final_broker_transaction_id:
            if final_ledger_transaction_id != final_broker_transaction_id:
                issues.append(
                    RiskIssue(
                        "FINAL_PRE_POST_TRANSACTION_ID_MISMATCH",
                        "post-reservation execution ledger and final broker "
                        "lastTransactionID differ; no broker POST is allowed",
                    )
                )
            elif (
                final_ledger_transaction_id != early_ledger_transaction_id
                or final_broker_transaction_id != early_broker_transaction_id
            ):
                issues.append(
                    RiskIssue(
                        "FINAL_PRE_POST_TRANSACTION_ID_CHANGED_AFTER_RESERVATION",
                        "broker transaction truth changed after the risk snapshot and "
                        "during durable reservation; rebuild and reverify a fresh signal",
                    )
                )
        elif sync_summary is not None and not any(
            issue.code
            in {
                "FINAL_PRE_POST_BROKER_SNAPSHOT_FAILED",
                "FINAL_PRE_POST_GPT_ALLOCATION_SNAPSHOT_FAILED",
            }
            for issue in issues
        ):
            issues.append(
                RiskIssue(
                    "FINAL_PRE_POST_TRANSACTION_ID_MISSING",
                    "post-reservation ledger and broker lastTransactionID must both be present",
                )
            )
        evidence["transaction_id_unchanged"] = bool(
            early_ledger_transaction_id
            and early_ledger_transaction_id == early_broker_transaction_id
            and early_ledger_transaction_id == final_ledger_transaction_id
            and early_ledger_transaction_id == final_broker_transaction_id
            and (
                not numeric_allocation_required
                or early_ledger_transaction_id
                == final_numeric_snapshot_transaction_id
            )
        )

        current_receipt_sha256 = _file_sha256(self.verified_decision_path)
        receipt_sha_matches: bool | None = None
        if self.verified_decision_path is not None:
            receipt_sha_matches = bool(
                expected_receipt_sha256 is not None
                and current_receipt_sha256 == expected_receipt_sha256
            )
            receipt_issue = _verified_decision_receipt_change_issue(
                self.verified_decision_path,
                expected_sha256=expected_receipt_sha256,
            )
            if receipt_issue is not None:
                issues.append(
                    RiskIssue(
                        str(receipt_issue.get("code") or "GPT_RECEIPT_CHANGED"),
                        str(
                            receipt_issue.get("message")
                            or "verified GPT receipt changed before broker POST"
                        ),
                        str(receipt_issue.get("severity") or "BLOCK"),
                    )
                )
        evidence.update(
            {
                "verified_decision_expected_sha256": expected_receipt_sha256,
                "verified_decision_current_sha256": current_receipt_sha256,
                "verified_decision_sha_matches": receipt_sha_matches,
            }
        )

        current_order_request_sha256 = _canonical_order_request_sha256(
            order_request
        )
        claimed_order_request_sha256 = str(
            ordinary_entry_claim.get("order_request_sha256") or ""
        ).strip()
        order_sha_matches = bool(
            expected_order_request_sha256
            and current_order_request_sha256 == expected_order_request_sha256
            and (
                not claimed_order_request_sha256
                or claimed_order_request_sha256 == current_order_request_sha256
            )
        )
        evidence.update(
            {
                "expected_order_request_sha256": expected_order_request_sha256,
                "current_order_request_sha256": current_order_request_sha256,
                "claimed_order_request_sha256": (
                    claimed_order_request_sha256 or None
                ),
                "order_request_sha_matches": order_sha_matches,
            }
        )
        if not order_sha_matches:
            issues.append(
                RiskIssue(
                    "FINAL_PRE_POST_ORDER_REQUEST_CHANGED_AFTER_RESERVATION",
                    "broker order request changed after durable reservation; do not "
                    "send an order that is not bound to the consumed claim",
                )
            )

        # Re-read the canonical chart artifact at the last gateway boundary.
        # The producer receipt is never permission by itself: the same exact
        # pair row, root clock, digest, proof shape, current broker spread, and
        # bounded units must still agree after all reservations/reproofs.
        m15_final_evidence: dict[str, Any] = {"status": "NOT_APPLICABLE"}
        if m15_recovery_micro_claimed(verified_intent):
            final_recovery_intent = final_reserved_intent or final_intent
            if final_m15_snapshot is None:
                issues.append(
                    RiskIssue(
                        "FINAL_PRE_POST_M15_RECOVERY_SNAPSHOT_MISSING",
                        "the final M15 recovery fence has no fresh broker snapshot",
                    )
                )
                m15_final_evidence = {
                    "status": "BLOCKED",
                    "blocking_codes": [
                        "FINAL_PRE_POST_M15_RECOVERY_SNAPSHOT_MISSING"
                    ],
                }
            else:
                final_recovery = validate_m15_recovery_micro_live_claim(
                    final_recovery_intent,
                    final_m15_snapshot,
                    pair_charts_path=self.pair_charts_path,
                    forecast_history_path=self.forecast_history_path,
                    validation_time_utc=final_m15_snapshot.fetched_at_utc,
                )
                m15_final_evidence = final_recovery.evidence
                issues.extend(final_recovery.issues)
                early_recovery = (
                    early_reconciliation.get("m15_recovery_micro_recheck")
                    if isinstance(
                        early_reconciliation.get(
                            "m15_recovery_micro_recheck"
                        ),
                        dict,
                    )
                    else {}
                )
                immutable_fields = (
                    "source_sha256",
                    "pair_chart_sha256",
                    "root_generated_at_utc",
                    "receipt_sha256",
                    "pair",
                    "side",
                    "order_type",
                    "producer_units",
                    "forecast_binding_sha256",
                    "lane_binding_sha256",
                )
                source_unchanged = bool(
                    early_recovery.get("status") == "PASSED"
                    and final_recovery.allowed
                    and all(
                        early_recovery.get(field)
                        == m15_final_evidence.get(field)
                        for field in immutable_fields
                    )
                )
                expected_receipt = (
                    (verified_intent.metadata or {}).get(
                        "m15_recovery_micro_receipt"
                    )
                )
                final_receipt = (
                    (final_recovery_intent.metadata or {}).get(
                        "m15_recovery_micro_receipt"
                    )
                )
                receipt_unchanged = bool(
                    isinstance(expected_receipt, dict)
                    and final_receipt == expected_receipt
                    and m15_final_evidence.get("receipt_sha256")
                    == expected_receipt.get("receipt_sha256")
                )
                units_not_upsized = bool(
                    final_recovery_intent.units > 0
                    and final_recovery_intent.units <= verified_intent.units
                    and final_recovery_intent.units
                    <= M15_RECOVERY_MICRO_MAX_UNITS
                )
                m15_final_evidence = {
                    **m15_final_evidence,
                    "early_source_unchanged": source_unchanged,
                    "receipt_unchanged": receipt_unchanged,
                    "units_not_upsized": units_not_upsized,
                    "final_gateway_live_permission_granted": False,
                }
                if not source_unchanged:
                    issues.append(
                        RiskIssue(
                            "FINAL_PRE_POST_M15_RECOVERY_SOURCE_CHANGED",
                            "the canonical pair_charts artifact, root clock, selected pair row, or early recovery proof changed before POST",
                        )
                    )
                if not receipt_unchanged:
                    issues.append(
                        RiskIssue(
                            "FINAL_PRE_POST_M15_RECOVERY_RECEIPT_CHANGED",
                            "the M15 recovery receipt changed after GPT/early risk verification",
                        )
                    )
                if not units_not_upsized:
                    issues.append(
                        RiskIssue(
                            "FINAL_PRE_POST_M15_RECOVERY_UPSIZE_FORBIDDEN",
                            "the final reserved M15 recovery order exceeds its verified producer units or the 999u ceiling",
                        )
                    )
        evidence["m15_recovery_micro_final_recheck"] = m15_final_evidence

        # This is deliberately the final mutable-file read before the caller's
        # broker POST. Rebuild the exact full Guardian scope that GPT reviewed;
        # unrelated routine receipt rotation canonicalizes to the same state,
        # while selected-pair meaning and every global safety break fail closed.
        from quant_rabbit.market_read_overlay import (
            GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT,
            canonical_json_sha256,
            guardian_action_receipt_scope_material,
        )

        guardian_pairs = tuple(
            expected_guardian_action_receipt_baseline_pairs
        )
        guardian_expected_sha = str(
            expected_guardian_action_receipt_scope_state_sha256 or ""
        ).strip().lower()
        guardian_provenance_valid = bool(
            expected_guardian_action_receipt_material_contract
            == GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT
            and guardian_pairs
            and guardian_pairs == tuple(sorted(set(guardian_pairs)))
            and all(pair in DEFAULT_SPECS for pair in guardian_pairs)
            and verified_intent.pair in guardian_pairs
            and final_intent.pair in guardian_pairs
            and re.fullmatch(r"[0-9a-f]{64}", guardian_expected_sha)
            is not None
        )
        if not guardian_provenance_valid:
            issues.append(
                RiskIssue(
                    "FINAL_PRE_POST_GUARDIAN_RECEIPT_PROVENANCE_INVALID",
                    "post-reservation Guardian fence requires the frozen material "
                    "contract, reviewed pair scope containing the selected pair, "
                    "and a canonical scope-state digest",
                )
            )

        guardian_material: dict[str, Any]
        guardian_recheck_error: str | None = None
        try:
            guardian_material = guardian_action_receipt_scope_material(
                self.guardian_action_receipt_path,
                baseline_pairs=guardian_pairs,
                as_of=datetime.now(timezone.utc),
            )
            guardian_current_sha = canonical_json_sha256(guardian_material)
        except Exception as exc:
            # The helper is intentionally fail-closed, but a final catch keeps
            # an unforeseen parser/filesystem error from skipping this fence.
            guardian_recheck_error = f"{type(exc).__name__}: {exc}"
            guardian_material = {
                "parse_status": "UNREADABLE",
                "baseline_pairs": list(guardian_pairs),
            }
            guardian_current_sha = None

        guardian_parse_status = str(
            guardian_material.get("parse_status") or "INVALID"
        ).strip().upper()
        guardian_global_safety = guardian_material.get("global_safety") is True
        guardian_digest_matches = bool(
            guardian_provenance_valid
            and guardian_current_sha == guardian_expected_sha
        )
        guardian_recheck = {
            "status": "PASSED",
            "material_contract": (
                expected_guardian_action_receipt_material_contract
            ),
            "baseline_pairs": list(guardian_pairs),
            "parse_status": guardian_parse_status,
            "scope": guardian_material.get("scope"),
            "time_state": guardian_material.get("time_state"),
            "global_safety": guardian_global_safety,
            "p1_margin_warning_observed": guardian_material.get(
                "p1_margin_warning_observed"
            )
            is True,
            "margin_contract": guardian_material.get("margin_contract"),
            "global_reasons": list(
                guardian_material.get("global_reasons") or []
            ),
            "expected_scope_state_sha256": guardian_expected_sha or None,
            "current_scope_state_sha256": guardian_current_sha,
            "digest_matches": guardian_digest_matches,
            "raw_read_immediately_before_post": True,
            "error": guardian_recheck_error,
        }
        if guardian_parse_status != "VALID":
            issues.append(
                RiskIssue(
                    "FINAL_PRE_POST_GUARDIAN_RECEIPT_UNAVAILABLE_OR_INVALID",
                    "canonical Guardian action receipt was missing, unreadable, "
                    "oversized, or invalid at the final pre-POST boundary",
                )
            )
        if guardian_global_safety:
            issues.append(
                RiskIssue(
                    "FINAL_PRE_POST_GUARDIAN_GLOBAL_SAFETY_BLOCK",
                    "Guardian reported a portfolio/global safety condition or a "
                    "broken routing/receipt contract immediately before POST",
                )
            )
        if not guardian_digest_matches:
            issues.append(
                RiskIssue(
                    "FINAL_PRE_POST_GUARDIAN_RECEIPT_SCOPE_CHANGED",
                    "Guardian meaning in the GPT-reviewed pair scope changed after "
                    "decision verification; rebuild the market read before POST",
                )
            )
        if any(
            issue.code.startswith("FINAL_PRE_POST_GUARDIAN_")
            for issue in issues
        ):
            guardian_recheck["status"] = "BLOCKED"
        evidence["guardian_action_receipt_scope_recheck"] = guardian_recheck

        blocking_codes = [
            issue.code for issue in issues if issue.severity == "BLOCK"
        ]
        evidence.update(
            {
                "status": "BLOCKED" if blocking_codes else "PASSED",
                "blocking_codes": blocking_codes,
            }
        )
        return _FinalPrePostBoundaryResult(
            evidence=evidence,
            issues=tuple(issues),
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
            pair_charts_path=self.pair_charts_path,
            forecast_history_path=self.forecast_history_path,
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
        asymmetry_metadata_issue = _loss_asymmetry_nonfinite_issue(intent.metadata or {})
        if asymmetry_metadata_issue is None:
            asymmetry_metadata_issue = _loss_asymmetry_avg_win_issue(
                intent,
                getattr(risk, "metrics", None),
            )
        if asymmetry_metadata_issue is not None:
            return (
                intent,
                risk,
                order_request,
                None,
                size_multiple,
                [asymmetry_metadata_issue],
                order_build_issues,
            )
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
            or action != "TRADE"
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
                    "pre-POST one-shot claim requires the same ACCEPTED schema-v2 TRADE receipt to still name the selected lane",
                ).__dict__,
            )

        try:
            current_lineage = read_verified_decision_lineage(
                self.verified_decision_path,
                selected_lane_id=lane_id,
            )
            intent_lineage = lineage_from_metadata(intent.metadata)
        except DecisionExecutionLineageError as exc:
            return _OrdinaryEntryPostClaimResult(
                evidence={
                    "status": "BLOCKED",
                    "issue_code": "ORDINARY_ENTRY_CLAIM_LINEAGE_INVALID",
                    "error": str(exc),
                },
                issue=RiskIssue(
                    "ORDINARY_ENTRY_CLAIM_LINEAGE_INVALID",
                    "pre-POST one-shot claim could not revalidate immutable GPT market-read lineage: "
                    f"{exc}",
                ).__dict__,
            )
        if (current_lineage is None) != (intent_lineage is None) or (
            current_lineage is not None
            and intent_lineage is not None
            and (
                current_lineage.decision_receipt_id != intent_lineage.decision_receipt_id
                or current_lineage.market_read_prediction_id
                != intent_lineage.market_read_prediction_id
                or current_lineage.lineage_token != intent_lineage.lineage_token
            )
        ):
            return _OrdinaryEntryPostClaimResult(
                evidence={
                    "status": "BLOCKED",
                    "issue_code": "ORDINARY_ENTRY_CLAIM_LINEAGE_MISMATCH",
                },
                issue=RiskIssue(
                    "ORDINARY_ENTRY_CLAIM_LINEAGE_MISMATCH",
                    "selected intent lineage no longer matches the accepted GPT receipt at the pre-POST boundary",
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
        order_request_sha256 = _canonical_order_request_sha256(order_request)
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
            "decision_lineage": current_lineage.to_dict() if current_lineage else None,
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
                    decision_receipt_id TEXT,
                    market_read_prediction_id TEXT,
                    decision_lineage_token TEXT,
                    status TEXT NOT NULL,
                    claimed_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    broker_outcome_json TEXT
                )
                """
            )
            existing_columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(ordinary_live_entry_signal_claims)"
                )
            }
            for column in (
                "decision_receipt_id",
                "market_read_prediction_id",
                "decision_lineage_token",
            ):
                if column not in existing_columns:
                    conn.execute(
                        f"ALTER TABLE ordinary_live_entry_signal_claims ADD COLUMN {column} TEXT"
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
                    order_request_sha256, decision_receipt_id,
                    market_read_prediction_id, decision_lineage_token,
                    status, claimed_at_utc, updated_at_utc,
                    broker_outcome_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'RESERVED_PRE_POST', ?, ?, NULL)
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
                    current_lineage.decision_receipt_id if current_lineage else None,
                    current_lineage.market_read_prediction_id if current_lineage else None,
                    current_lineage.lineage_token if current_lineage else None,
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
            with closing(sqlite3.connect(str(claim_db_path), timeout=5.0)) as conn, conn:
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
            # delete/release it merely because outcome annotation failed.
            return {
                **claim_evidence,
                "status": "FINALIZE_FAILED_RESERVATION_RETAINED",
                "finalize_error": f"{type(exc).__name__}: {exc}",
            }, RiskIssue(
                "ORDINARY_ENTRY_CLAIM_FINALIZE_FAILED",
                "the one-shot claim outcome could not be annotated; "
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
            "decision_lineage": _decision_lineage_receipt_from_intent(intent, order_request),
            "range_vehicle_candidate_receipt": _range_vehicle_candidate_receipt_from_intent(
                intent, order_request
            ),
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
    if m15_recovery_micro_claimed(intent):
        if not math.isfinite(size_multiple) or size_multiple <= 0:
            return None, [RiskIssue("INVALID_SIZE_MULTIPLE", "size_multiple must be a finite positive number")], size_multiple
        if size_multiple > 1.0 + 1e-9:
            return (
                None,
                [
                    RiskIssue(
                        "M15_RECOVERY_GATEWAY_UPSIZE_FORBIDDEN",
                        "the bounded M15 recovery gateway may downsize for current risk/margin but must never upsize producer units",
                    )
                ],
                size_multiple,
            )
        scaled_units, issues = _scaled_units(intent.units, size_multiple)
        return scaled_units, issues, size_multiple
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
        margin_room = margin_budget_jpy(
            account,
            max_margin_utilization_pct=max_margin_pct,
        )
        remaining_margin = margin_room - pending_margin - cumulative_margin_jpy
        if remaining_margin <= 0:
            return 0.0, RiskIssue(
                "BASKET_MARGIN_CAP_REACHED",
                f"basket margin capacity is exhausted: pending {pending_margin:.0f} + batch {cumulative_margin_jpy:.0f} "
                f">= room {margin_room:.0f} JPY",
            )
        if candidate_margin > remaining_margin:
            scale = min(
                scale,
                _capacity_scale(
                    requested_units,
                    candidate_margin,
                    remaining_margin,
                ),
            )
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
        margin_room = margin_budget_jpy(
            account,
            max_margin_utilization_pct=max_margin_pct,
        )
        total_margin = pending_margin + cumulative_margin_jpy + candidate_margin
        if margin_room <= 0 or total_margin > margin_room:
            issues.append(
                RiskIssue(
                    "BASKET_MARGIN_UTILIZATION_CAP_EXCEEDED",
                    f"basket candidate margin {total_margin:.0f} JPY cannot fit remaining {max_margin_pct:.1f}% "
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


def _freeze_verified_decision_receipt(
    path: Path | None,
) -> _VerifiedDecisionReceiptFreeze:
    """Read a verified decision once and freeze every POST-critical identity.

    The mutable path is never consulted again for edge-basis or execution-cost
    selection. Later path reads may only attest that these exact bytes still
    occupy the path; they cannot change the frozen semantics.
    """

    if path is None:
        return _VerifiedDecisionReceiptFreeze(None, False, None, None, None)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return _VerifiedDecisionReceiptFreeze(
            None,
            True,
            {
                "severity": "BLOCK",
                "code": (
                    "GPT_CAPITAL_ALLOCATION_REQUIREMENT_UNREADABLE_AT_GATEWAY_ENTRY"
                ),
                "message": (
                    "verified GPT receipt could not be read while freezing "
                    "numeric allocation applicability and signed evidence: "
                    f"{type(exc).__name__}: {exc}"
                ),
            },
            None,
            None,
        )
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        return _VerifiedDecisionReceiptFreeze(
            receipt_sha256,
            True,
            {
                "severity": "BLOCK",
                "code": (
                    "GPT_CAPITAL_ALLOCATION_REQUIREMENT_UNREADABLE_AT_GATEWAY_ENTRY"
                ),
                "message": (
                    "verified GPT receipt is not strict JSON while freezing "
                    "numeric allocation applicability and signed evidence: "
                    f"{type(exc).__name__}: {exc}"
                ),
            },
            None,
            None,
        )
    if not isinstance(payload, dict) or not isinstance(
        payload.get("decision"),
        dict,
    ):
        return _VerifiedDecisionReceiptFreeze(
            receipt_sha256,
            True,
            {
                "severity": "BLOCK",
                "code": (
                    "GPT_CAPITAL_ALLOCATION_REQUIREMENT_UNREADABLE_AT_GATEWAY_ENTRY"
                ),
                "message": (
                    "verified GPT receipt lacks an object decision while freezing "
                    "numeric allocation applicability and signed evidence"
                ),
            },
            None,
            None,
        )
    decision = payload["decision"]
    provenance = (
        decision.get("decision_provenance")
        if isinstance(decision.get("decision_provenance"), dict)
        else {}
    )
    allocation = decision.get("capital_allocation")
    numeric_required = bool(
        isinstance(allocation, dict)
        and provenance.get("schema_version") == 2
        and str(provenance.get("author_kind") or "").strip().upper()
        == "CODEX_MARKET_READ"
        and str(decision.get("action") or "").strip().upper() == "TRADE"
        and str(allocation.get("decision") or "").strip().upper()
        == "ALLOCATE"
    )
    edge_basis = str(
        provenance.get("capital_allocation_edge_basis") or ""
    ).strip().upper()
    if edge_basis not in {
        "EXACT_VEHICLE_ALL_EXIT_NET",
        "EXACT_VEHICLE_TAKE_PROFIT",
        M15_RECOVERY_EDGE_COLLECTION_BASIS,
        TARGET_PATH_LIVE_LEARNING_EDGE_COLLECTION_BASIS,
        "PREDICTIVE_SCOUT_FORWARD_EVIDENCE",
        "HEDGE_RISK_REDUCTION",
    }:
        edge_basis = None
    cost_sha256 = str(
        provenance.get("execution_cost_floor_sha256") or ""
    ).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", cost_sha256) is None:
        cost_sha256 = None
    from quant_rabbit.market_read_overlay import (
        GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT,
    )

    guardian_contract = provenance.get(
        "guardian_action_receipt_material_contract"
    )
    raw_guardian_pairs = provenance.get(
        "guardian_action_receipt_baseline_pairs"
    )
    guardian_sha256 = str(
        provenance.get("guardian_action_receipt_scope_state_sha256") or ""
    ).strip().lower()
    lane_ids: list[str] = []
    raw_lane_ids = decision.get("selected_lane_ids")
    if isinstance(raw_lane_ids, list):
        lane_ids.extend(
            item for item in raw_lane_ids if isinstance(item, str)
        )
    raw_lane_id = decision.get("selected_lane_id")
    if isinstance(raw_lane_id, str) and raw_lane_id not in lane_ids:
        lane_ids.append(raw_lane_id)
    expected_guardian_pairs = tuple(
        sorted(
            {
                token.strip().upper()
                for lane_id in lane_ids
                for token in lane_id.split(":")
                if token.strip().upper() in DEFAULT_SPECS
            }
        )
    )
    guardian_pairs = (
        tuple(raw_guardian_pairs)
        if isinstance(raw_guardian_pairs, list)
        and all(isinstance(pair, str) for pair in raw_guardian_pairs)
        else ()
    )
    guardian_provenance_required = bool(
        provenance.get("schema_version") == 2
        and str(provenance.get("author_kind") or "").strip().upper()
        == "CODEX_MARKET_READ"
        and str(decision.get("action") or "").strip().upper() == "TRADE"
    )
    guardian_provenance_valid = bool(
        guardian_contract == GUARDIAN_ACTION_RECEIPT_MATERIAL_CONTRACT
        and guardian_pairs
        and guardian_pairs == tuple(sorted(set(guardian_pairs)))
        and all(pair in DEFAULT_SPECS for pair in guardian_pairs)
        and set(expected_guardian_pairs).issubset(set(guardian_pairs))
        and expected_guardian_pairs
        and re.fullmatch(r"[0-9a-f]{64}", guardian_sha256) is not None
    )
    guardian_issue = None
    if guardian_provenance_required and not guardian_provenance_valid:
        guardian_issue = {
            "severity": "BLOCK",
            "code": "GPT_GUARDIAN_ACTION_RECEIPT_PROVENANCE_INVALID_AT_GATEWAY_ENTRY",
            "message": (
                "schema-v2 CODEX_MARKET_READ TRADE must freeze the exact Guardian "
                "material contract, sorted execution pairs, and scope-state digest"
            ),
        }
    return _VerifiedDecisionReceiptFreeze(
        sha256=receipt_sha256,
        numeric_allocation_required=numeric_required,
        numeric_requirement_issue=None,
        capital_allocation_edge_basis=edge_basis,
        execution_cost_floor_sha256=cost_sha256,
        guardian_provenance_issue=guardian_issue,
        guardian_action_receipt_material_contract=(
            str(guardian_contract)
            if guardian_provenance_valid
            else None
        ),
        guardian_action_receipt_baseline_pairs=(
            guardian_pairs if guardian_provenance_valid else ()
        ),
        guardian_action_receipt_scope_state_sha256=(
            guardian_sha256 if guardian_provenance_valid else None
        ),
    )


def _codex_numeric_allocation_requirement_at_entry(
    path: Path | None,
    *,
    expected_sha256: str | None,
) -> tuple[bool, dict[str, str] | None]:
    """Freeze numeric-proof applicability from one exact entry receipt read."""

    if path is None:
        return False, None
    frozen = _freeze_verified_decision_receipt(path)
    if frozen.sha256 is None:
        return True, frozen.numeric_requirement_issue
    if expected_sha256 is None or frozen.sha256 != expected_sha256:
        return True, {
            "severity": "BLOCK",
            "code": "GPT_CAPITAL_ALLOCATION_REQUIREMENT_RECEIPT_CHANGED_AT_GATEWAY_ENTRY",
            "message": (
                "verified GPT receipt bytes changed while freezing numeric "
                "allocation applicability"
            ),
        }
    return frozen.numeric_allocation_required, frozen.numeric_requirement_issue


def _verified_execution_cost_floor_sha256(
    path: Path | None,
    *,
    expected_sha256: str | None = None,
) -> str | None:
    """Read the auto-stamped cost proof identity from one verified receipt."""

    if path is None:
        return None
    try:
        receipt_bytes = path.read_bytes()
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    if (
        expected_sha256 is not None
        and hashlib.sha256(receipt_bytes).hexdigest() != expected_sha256
    ):
        return None
    try:
        payload = json.loads(receipt_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    decision = (
        payload.get("decision")
        if isinstance(payload, dict)
        and isinstance(payload.get("decision"), dict)
        else {}
    )
    provenance = (
        decision.get("decision_provenance")
        if isinstance(decision.get("decision_provenance"), dict)
        else {}
    )
    value = str(provenance.get("execution_cost_floor_sha256") or "")
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None


def _verified_capital_allocation_edge_basis(
    path: Path | None,
    *,
    expected_sha256: str | None = None,
) -> str | None:
    """Read the board-selected edge basis from signed decision provenance."""

    if path is None:
        return None
    try:
        receipt_bytes = path.read_bytes()
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    if (
        expected_sha256 is not None
        and hashlib.sha256(receipt_bytes).hexdigest() != expected_sha256
    ):
        return None
    try:
        payload = json.loads(receipt_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    decision = (
        payload.get("decision")
        if isinstance(payload, dict)
        and isinstance(payload.get("decision"), dict)
        else {}
    )
    provenance = (
        decision.get("decision_provenance")
        if isinstance(decision.get("decision_provenance"), dict)
        else {}
    )
    value = str(
        provenance.get("capital_allocation_edge_basis") or ""
    ).strip().upper()
    return value if value in {
        "EXACT_VEHICLE_ALL_EXIT_NET",
        "EXACT_VEHICLE_TAKE_PROFIT",
        M15_RECOVERY_EDGE_COLLECTION_BASIS,
        TARGET_PATH_LIVE_LEARNING_EDGE_COLLECTION_BASIS,
        "PREDICTIVE_SCOUT_FORWARD_EVIDENCE",
        "HEDGE_RISK_REDUCTION",
    } else None


def _numeric_predictive_scout_contract(intent: OrderIntent) -> bool:
    metadata = dict(intent.metadata or {})
    method = (
        intent.market_context.method.value
        if intent.market_context is not None
        and isinstance(intent.market_context.method, TradeMethod)
        else ""
    )
    return bool(
        predictive_scout_intent_claimed(intent)
        and predictive_scout_metadata_supported(metadata)
        and intent.order_type == OrderType.LIMIT
        and method == "BREAKOUT_FAILURE"
        and str(metadata.get("desk") or "").strip().lower()
        == "failure_trader"
        and str(metadata.get("campaign_role") or "").strip().upper()
        == "BIDASK_REPLAY_CONTRARIAN_SCOUT"
        and metadata.get("attach_take_profit_on_fill") is True
        and str(metadata.get("tp_execution_mode") or "").strip().upper()
        == "ATTACHED_TECHNICAL_TP"
    )


def _m15_recovery_prebounded_intent_claim(
    intent: OrderIntent | None,
) -> bool:
    """Rebuild the immutable part of the recovery collection contract.

    This helper deliberately does not grant live permission.  The caller must
    still run ``validate_m15_recovery_micro_live_claim`` against the current
    chart/history/broker snapshot.  It only prevents a copied edge-basis label
    from entering the non-MARKET numeric/transport bypass.
    """

    if not isinstance(intent, OrderIntent):
        return False
    metadata = dict(intent.metadata or {})
    method = (
        intent.market_context.method.value
        if intent.market_context is not None
        and isinstance(intent.market_context.method, TradeMethod)
        else ""
    )
    if (
        not m15_recovery_micro_claimed(intent)
        or method != "BREAKOUT_FAILURE"
        or intent.order_type != OrderType.STOP_ENTRY
        or intent.units.__class__ is not int
        or not 1 <= intent.units <= M15_RECOVERY_MICRO_MAX_UNITS
        or str(metadata.get("positive_rotation_mode") or "").strip().upper()
        not in {"TP_PROOF_COLLECTION_HARVEST", "TP_PROVEN_HARVEST"}
        or str(metadata.get("position_intent") or "NEW").strip().upper()
        == "HEDGE"
        or metadata.get("attach_take_profit_on_fill") is not True
        or str(metadata.get("tp_execution_mode") or "").strip().upper()
        != "ATTACHED_TECHNICAL_TP"
        or str(metadata.get("tp_target_intent") or "").strip().upper()
        != "HARVEST"
        or str(metadata.get("opportunity_mode") or "").strip().upper()
        != "HARVEST"
    ):
        return False
    try:
        from quant_rabbit.strategy.intent_generator import (
            positive_rotation_proof_acquisition_contract,
        )
        from quant_rabbit.strategy.m15_recovery_contract import (
            validate_forecast_binding,
            validate_lane_binding,
        )

        receipt = metadata.get("m15_recovery_micro_receipt")
        forecast_binding = metadata.get("forecast_m15_recovery_binding")
        lane_binding = metadata.get("m15_recovery_lane_binding")
        if not all(
            isinstance(value, dict)
            for value in (receipt, forecast_binding, lane_binding)
        ):
            return False
        forecast_valid, _ = validate_forecast_binding(
            forecast_binding,
            recovery_receipt=receipt,
            metadata=metadata,
        )
        lane_valid, _ = validate_lane_binding(
            lane_binding,
            forecast_binding=forecast_binding,
            pair=intent.pair,
            side=intent.side.value,
            method=method,
            order_type=intent.order_type.value,
            entry=intent.entry,
            tp=intent.tp,
            sl=intent.sl,
            current_units=intent.units,
            metadata=metadata,
        )
        proof = positive_rotation_proof_acquisition_contract(intent)
    except (TypeError, ValueError, OverflowError, AttributeError):
        return False
    mode = str(metadata.get("positive_rotation_mode") or "").strip().upper()
    return bool(
        forecast_valid
        and lane_valid
        and proof.get("positive_rotation_mode") == mode
        and proof.get("reachable") is True
        and not proof.get("failed_checks")
    )


def _m15_recovery_edge_collection_prebounded_intent_claim(
    intent: OrderIntent | None,
) -> bool:
    """Bind the distinct collection basis without upgrading it to TP-proven."""

    return bool(
        _m15_recovery_prebounded_intent_claim(intent)
        and isinstance(intent, OrderIntent)
        and str(
            (intent.metadata or {}).get("positive_rotation_mode") or ""
        ).strip().upper()
        == "TP_PROOF_COLLECTION_HARVEST"
    )


def _range_tp_proven_prebounded_intent_claim(intent: OrderIntent) -> bool:
    """Recognize—but do not authorize—the narrow passive RANGE claim."""

    metadata = dict(intent.metadata or {})
    method = (
        intent.market_context.method.value
        if intent.market_context is not None
        and isinstance(intent.market_context.method, TradeMethod)
        else ""
    )
    expected_entry_side = (
        "SUPPORT" if intent.side == Side.LONG else "RESISTANCE"
    )
    return bool(
        intent.order_type == OrderType.LIMIT
        and intent.entry is not None
        and method in {"RANGE_ROTATION", "BREAKOUT_FAILURE"}
        and str(metadata.get("forecast_direction") or "").strip().upper()
        == "RANGE"
        and str(
            metadata.get("forecast_directional_calibration_name") or ""
        ).strip().lower()
        == "directional_forecast_range"
        and str(metadata.get("position_intent") or "NEW").strip().upper()
        != "HEDGE"
        and metadata.get("attach_take_profit_on_fill") is True
        and str(metadata.get("tp_execution_mode") or "").strip().upper()
        == "ATTACHED_TECHNICAL_TP"
        and str(metadata.get("tp_target_intent") or "").strip().upper()
        == "HARVEST"
        and str(metadata.get("opportunity_mode") or "").strip().upper()
        == "HARVEST"
        and str(metadata.get("positive_rotation_mode") or "").strip().upper()
        == "TP_PROVEN_HARVEST"
        and metadata.get("positive_rotation_live_ready") is True
        and str(metadata.get("loss_asymmetry_guard_mode") or "").strip().upper()
        == "TP_PROVEN_RELAXED"
        and str(metadata.get("geometry_model") or "").strip().upper()
        == "RANGE_RAIL_LIMIT"
        and metadata.get("range_tp_is_inside_box") is True
        and metadata.get("range_sl_outside_box") is True
        and str(metadata.get("range_entry_side") or "").strip().upper()
        == expected_entry_side
    )


def _post_s5_tp_proven_range_quote_fence(
    *,
    intent: OrderIntent,
    snapshot: BrokerSnapshot,
    prior_snapshot: BrokerSnapshot | None,
) -> tuple[dict[str, Any], RiskIssue | None]:
    """Keep a frozen RANGE LIMIT passive and untouched after the final S5 read."""

    if not _range_tp_proven_prebounded_intent_claim(intent):
        return {"status": "NOT_APPLICABLE"}, None

    def finished(material: dict[str, Any]) -> dict[str, Any]:
        return {
            **material,
            "proof_sha256": _canonical_json_sha256(material),
        }

    quote = snapshot.quotes.get(intent.pair)
    prior_quote = (
        prior_snapshot.quotes.get(intent.pair)
        if prior_snapshot is not None
        else None
    )
    target, invalidation, barrier_basis = _forecast_s5_barriers(intent)
    quote_values = (
        (quote.bid, quote.ask)
        if quote is not None
        else (None, None)
    )
    quote_valid = bool(
        quote is not None
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in quote_values
        )
        and float(quote.bid) <= float(quote.ask)
    )
    geometry_valid = bool(
        barrier_basis == "RANGE_RAILS"
        and intent.entry is not None
        and target is not None
        and invalidation is not None
        and all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in (
                intent.entry,
                intent.tp,
                intent.sl,
                target,
                invalidation,
            )
        )
    )
    quote_time_monotonic = bool(
        quote is not None
        and (
            prior_quote is None
            or quote.timestamp_utc.astimezone(timezone.utc)
            >= prior_quote.timestamp_utc.astimezone(timezone.utc)
        )
    )
    snapshot_time_monotonic = bool(
        prior_snapshot is None
        or snapshot.fetched_at_utc.astimezone(timezone.utc)
        >= prior_snapshot.fetched_at_utc.astimezone(timezone.utc)
    )
    entry_touched = False
    tp_touched = False
    target_touched = False
    invalidation_touched = False
    sl_touched = False
    if quote_valid and geometry_valid and quote is not None:
        if intent.side == Side.LONG:
            entry_touched = float(quote.ask) <= float(intent.entry)
            tp_touched = float(quote.ask) >= float(intent.tp)
            target_touched = float(quote.ask) >= float(target)
            invalidation_touched = float(quote.bid) <= float(invalidation)
            sl_touched = float(quote.bid) <= float(intent.sl)
        else:
            entry_touched = float(quote.bid) >= float(intent.entry)
            tp_touched = float(quote.bid) <= float(intent.tp)
            target_touched = float(quote.bid) <= float(target)
            invalidation_touched = float(quote.ask) >= float(invalidation)
            sl_touched = float(quote.ask) >= float(intent.sl)
    passed = bool(
        quote_valid
        and geometry_valid
        and quote_time_monotonic
        and snapshot_time_monotonic
        and not entry_touched
        and not tp_touched
        and not target_touched
        and not invalidation_touched
        and not sl_touched
    )
    reason = (
        "POST_S5_RANGE_QUOTE_UNTOUCHED"
        if passed
        else "POST_S5_RANGE_QUOTE_INVALID"
        if not quote_valid
        else "POST_S5_RANGE_GEOMETRY_INVALID"
        if not geometry_valid
        else "POST_S5_RANGE_QUOTE_TIME_REGRESSED"
        if not quote_time_monotonic or not snapshot_time_monotonic
        else "POST_S5_RANGE_LIMIT_ENTRY_CROSSED"
        if entry_touched
        else "POST_S5_RANGE_TP_OR_TARGET_TOUCHED"
        if tp_touched or target_touched
        else "POST_S5_RANGE_INVALIDATION_OR_SL_TOUCHED"
    )
    material = {
        "status": "PASSED" if passed else "BLOCKED",
        "reason": reason,
        "pair": intent.pair,
        "side": intent.side.value,
        "order_type": intent.order_type.value,
        "entry": intent.entry,
        "tp": intent.tp,
        "sl": intent.sl,
        "forecast_target": target,
        "forecast_invalidation": invalidation,
        "barrier_basis": barrier_basis,
        "quote_bid": float(quote.bid) if quote_valid and quote is not None else None,
        "quote_ask": float(quote.ask) if quote_valid and quote is not None else None,
        "quote_timestamp_utc": (
            quote.timestamp_utc.astimezone(timezone.utc).isoformat()
            if quote is not None
            else None
        ),
        "prior_quote_timestamp_utc": (
            prior_quote.timestamp_utc.astimezone(timezone.utc).isoformat()
            if prior_quote is not None
            else None
        ),
        "snapshot_fetched_at_utc": snapshot.fetched_at_utc.astimezone(
            timezone.utc
        ).isoformat(),
        "prior_snapshot_fetched_at_utc": (
            prior_snapshot.fetched_at_utc.astimezone(timezone.utc).isoformat()
            if prior_snapshot is not None
            else None
        ),
        "quote_valid": quote_valid,
        "geometry_valid": geometry_valid,
        "quote_time_monotonic": quote_time_monotonic,
        "snapshot_time_monotonic": snapshot_time_monotonic,
        "entry_touched": entry_touched,
        "tp_touched": tp_touched,
        "target_touched": target_touched,
        "invalidation_touched": invalidation_touched,
        "sl_touched": sl_touched,
        "repriced": False,
    }
    evidence = finished(material)
    if passed:
        return evidence, None
    issue_code = (
        "FINAL_PRE_POST_TP_PROVEN_RANGE_TRIGGER_CROSSED"
        if entry_touched
        else "FINAL_PRE_POST_TP_PROVEN_RANGE_QUOTE_FENCE_FAILED"
    )
    return evidence, RiskIssue(
        issue_code,
        "the exact TP-proven RANGE LIMIT or one of its frozen rails was crossed "
        "after the final S5 read; retain the reservation and expire the geometry "
        "without repricing or broker POST",
    )


def _range_tp_proven_basis_from_fresh_edge(
    intent: OrderIntent,
    edge_recheck: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Rebuild the RANGE fixed-cap receipt from the just-synced cohort.

    The original positive-rotation fields identify the reviewed contract, but
    their counts and Wilson stress are never trusted at POST time.  Recompute
    them from the fresh exact TP row carried by the edge recheck.
    """

    if not _range_tp_proven_prebounded_intent_claim(intent):
        return None
    edge = edge_recheck if isinstance(edge_recheck, dict) else {}
    method = intent.market_context.method.value
    expected_key = [
        intent.pair.upper(),
        intent.side.value.upper(),
        method.upper(),
        "LIMIT",
    ]
    current_tp = (
        edge.get("current_tp")
        if isinstance(edge.get("current_tp"), dict)
        else {}
    )
    fresh_capture = (
        edge.get("fresh_capture")
        if isinstance(edge.get("fresh_capture"), dict)
        else {}
    )
    ledger_sha = str(edge.get("ledger_surface_sha256") or "")
    trades = current_tp.get("trades")
    wins = current_tp.get("wins")
    losses = current_tp.get("losses")
    avg_win = current_tp.get("avg_win_jpy")
    expectancy = current_tp.get("expectancy_jpy")
    loss_proxy = fresh_capture.get("avg_loss_jpy")
    if (
        edge.get("status") != "PASSED"
        or edge.get("basis") != "EXACT_VEHICLE_TAKE_PROFIT"
        or edge.get("proof_key") != expected_key
        or not re.fullmatch(r"[0-9a-f]{64}", ledger_sha)
        or current_tp.get("arithmetic_consistent") is not True
        or isinstance(trades, bool)
        or not isinstance(trades, int)
        or trades < 20
        or isinstance(wins, bool)
        or not isinstance(wins, int)
        or wins < 0
        or wins > trades
        or isinstance(losses, bool)
        or not isinstance(losses, int)
        or losses != 0
        or isinstance(avg_win, bool)
        or not isinstance(avg_win, (int, float))
        or not math.isfinite(float(avg_win))
        or float(avg_win) <= 0.0
        or isinstance(expectancy, bool)
        or not isinstance(expectancy, (int, float))
        or not math.isfinite(float(expectancy))
        or float(expectancy) <= 0.0
        or isinstance(loss_proxy, bool)
        or not isinstance(loss_proxy, (int, float))
        or not math.isfinite(float(loss_proxy))
        or float(loss_proxy) <= 0.0
    ):
        return None
    hit_rate = wins / trades
    wilson_lower = hit_rate_wilson_lower(hit_rate, trades)
    if wilson_lower is None:
        return None
    loss_proxy = float(loss_proxy)
    pessimistic = (
        wilson_lower * float(avg_win)
        - (1.0 - wilson_lower) * loss_proxy
    )
    if pessimistic <= 0.0:
        return None

    fresh_metadata = dict(intent.metadata or {})
    fresh_metadata.update(
        {
            "capture_take_profit_trades": trades,
            "capture_take_profit_wins": wins,
            "capture_take_profit_losses": losses,
            "capture_take_profit_expectancy_jpy": float(expectancy),
            "capture_take_profit_avg_win_jpy": float(avg_win),
            "positive_rotation_tp_trades": trades,
            "positive_rotation_tp_wins": wins,
            "positive_rotation_loss_proxy_jpy": loss_proxy,
            "positive_rotation_tp_win_rate_lower": round(
                wilson_lower, 6
            ),
            "positive_rotation_pessimistic_expectancy_jpy": round(
                pessimistic, 4
            ),
        }
    )
    from quant_rabbit.market_read_overlay import (
        _tp_proven_harvest_range_economic_basis,
    )

    return _tp_proven_harvest_range_economic_basis(
        intent={
            "pair": intent.pair,
            "side": intent.side.value,
            "order_type": intent.order_type.value,
            "units": intent.units,
            "entry": intent.entry,
            "tp": intent.tp,
            "sl": intent.sl,
        },
        metadata=fresh_metadata,
        method=method,
        positive_tp_edge=True,
        execution_ledger_surface_sha256=ledger_sha,
    )


def _forecast_s5_barriers(
    intent: OrderIntent,
) -> tuple[float | None, float | None, str]:
    """Return side-aware barriers for directional or RANGE path reproof."""

    metadata = dict(intent.metadata or {})
    if _range_tp_proven_prebounded_intent_claim(intent):
        low = _positive_float(metadata.get("forecast_range_low_price"))
        high = _positive_float(metadata.get("forecast_range_high_price"))
        if low is None or high is None or low >= high:
            return None, None, "RANGE_RAILS_INVALID"
        if intent.side == Side.LONG:
            return high, low, "RANGE_RAILS"
        return low, high, "RANGE_RAILS"
    return (
        _positive_float(metadata.get("forecast_target_price")),
        _positive_float(metadata.get("forecast_invalidation_price")),
        "DIRECTIONAL_FORECAST",
    )


def _forecast_s5_no_touch_proof(
    client: Any,
    *,
    intent: OrderIntent,
    snapshot: BrokerSnapshot,
    m15_recovery_prebounded: bool = False,
) -> dict[str, Any]:
    """Prove no TP/target/invalidation touch from forecast emission to quote."""

    def finished(material: dict[str, Any]) -> dict[str, Any]:
        return {
            **material,
            "proof_sha256": _canonical_json_sha256(material),
        }

    metadata = dict(intent.metadata or {})
    hedge = str(metadata.get("position_intent") or "").strip().upper() == "HEDGE"
    scout = _numeric_predictive_scout_contract(intent)
    target_path_live_learning = (
        _target_path_live_learning_allocation_claim(intent)
    )
    m15_recovery = bool(
        m15_recovery_prebounded
        and _m15_recovery_prebounded_intent_claim(intent)
    )
    if hedge or scout or m15_recovery or target_path_live_learning:
        return finished(
            {
                "status": "BYPASSED",
                "reason": (
                    "HEDGE_RISK_REDUCTION_PREBOUNDED_CONTRACT"
                    if hedge
                    else TARGET_PATH_LIVE_LEARNING_PREBOUNDED_REASON
                    if target_path_live_learning
                    else M15_RECOVERY_PREBOUNDED_REASON
                    if m15_recovery
                    else "PREDICTIVE_SCOUT_PREBOUNDED_CONTRACT"
                ),
                "pair": intent.pair,
            }
        )

    cycle_id = str(metadata.get("forecast_cycle_id") or "").strip()
    match = PRE_ENTRY_FORECAST_CYCLE_RE.match(cycle_id)
    if match is None:
        return finished(
            {
                "status": "BLOCKED",
                "reason": "FORECAST_CYCLE_ID_NOT_PRODUCTION_ISO",
                "pair": intent.pair,
                "forecast_cycle_id": cycle_id or None,
            }
        )
    emission = _parse_utc_timestamp(match.group("ts"))
    quote = snapshot.quotes.get(intent.pair)
    quote_time = quote.timestamp_utc if quote is not None else None
    if emission is None or quote_time is None:
        return finished(
            {
                "status": "BLOCKED",
                "reason": "FORECAST_OR_QUOTE_TIME_MISSING",
                "pair": intent.pair,
                "forecast_cycle_id": cycle_id,
            }
        )
    emission = emission.astimezone(timezone.utc)
    quote_time = quote_time.astimezone(timezone.utc)
    window_seconds = (quote_time - emission).total_seconds()
    if window_seconds < 0.0 or window_seconds > FORECAST_S5_MAX_WINDOW_SECONDS:
        return finished(
            {
                "status": "BLOCKED",
                "reason": "FORECAST_S5_WINDOW_OUT_OF_RANGE",
                "pair": intent.pair,
                "forecast_emitted_at_utc": emission.isoformat(),
                "quote_timestamp_utc": quote_time.isoformat(),
                "window_seconds": window_seconds,
                "maximum_window_seconds": FORECAST_S5_MAX_WINDOW_SECONDS,
            }
        )

    def floor_s5(value: datetime) -> datetime:
        epoch = value.timestamp()
        return datetime.fromtimestamp(
            math.floor(epoch / FORECAST_S5_SECONDS) * FORECAST_S5_SECONDS,
            tz=timezone.utc,
        )

    first_start = floor_s5(emission)
    last_start = floor_s5(quote_time)
    query_to = last_start + timedelta(seconds=FORECAST_S5_SECONDS)
    query = {
        "granularity": "S5",
        "from": first_start.isoformat().replace("+00:00", "Z"),
        "to": query_to.isoformat().replace("+00:00", "Z"),
        "price": "M",
        "includeFirst": "true",
        "smooth": "false",
    }
    get_json = getattr(client, "get_json", None)
    if not callable(get_json):
        return finished(
            {
                "status": "BLOCKED",
                "reason": "FORECAST_S5_CLIENT_UNAVAILABLE",
                "pair": intent.pair,
                "query": query,
            }
        )
    try:
        payload = get_json(
            f"/v3/instruments/{intent.pair}/candles",
            query,
        )
    except Exception as exc:
        return finished(
            {
                "status": "BLOCKED",
                "reason": "FORECAST_S5_FETCH_FAILED",
                "pair": intent.pair,
                "query": query,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    raw_candles = payload.get("candles") if isinstance(payload, dict) else None
    if (
        not isinstance(raw_candles, list)
        or str(payload.get("instrument") or "") != intent.pair
        or str(payload.get("granularity") or "").upper() != "S5"
    ):
        return finished(
            {
                "status": "BLOCKED",
                "reason": "FORECAST_S5_PAYLOAD_INVALID",
                "pair": intent.pair,
                "query": query,
            }
        )
    parsed: list[tuple[datetime, float, float, float, float, bool]] = []
    malformed = False
    for item in raw_candles:
        if not isinstance(item, dict):
            malformed = True
            break
        timestamp = _parse_utc_timestamp(item.get("time"))
        mid = item.get("mid") if isinstance(item.get("mid"), dict) else None
        try:
            raw_values = tuple(mid[key] for key in ("o", "h", "l", "c"))
            if any(isinstance(value, bool) for value in raw_values):
                raise TypeError("boolean OHLC is not numeric broker price truth")
            values = tuple(float(value) for value in raw_values)
        except (KeyError, TypeError, ValueError):
            malformed = True
            break
        complete = item.get("complete")
        if (
            timestamp is None
            or not isinstance(complete, bool)
            or any(not math.isfinite(value) or value <= 0.0 for value in values)
            or values[2] > values[0]
            or values[2] > values[3]
            or values[1] < values[0]
            or values[1] < values[3]
            or values[2] > values[1]
        ):
            malformed = True
            break
        normalized_ts = timestamp.astimezone(timezone.utc)
        if first_start <= normalized_ts <= last_start:
            parsed.append(
                (
                    normalized_ts,
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    complete,
                )
            )
    parsed.sort(key=lambda row: row[0])
    expected_count = int(
        (last_start - first_start).total_seconds() / FORECAST_S5_SECONDS
    ) + 1
    timestamps = [row[0] for row in parsed]
    coverage_ok = bool(
        not malformed
        and len(parsed) == expected_count
        and len(set(timestamps)) == len(timestamps)
        and timestamps
        and timestamps[0] == first_start
        and timestamps[-1] == last_start
        and timestamps[0] <= emission
        < timestamps[0] + timedelta(seconds=FORECAST_S5_SECONDS)
        and timestamps[-1] <= quote_time
        < timestamps[-1] + timedelta(seconds=FORECAST_S5_SECONDS)
        and all(
            (right - left).total_seconds() == FORECAST_S5_SECONDS
            for left, right in zip(timestamps, timestamps[1:])
        )
        and all(row[5] is True for row in parsed[:-1])
    )
    forecast_target, forecast_invalidation, barrier_basis = (
        _forecast_s5_barriers(intent)
    )
    range_path = barrier_basis == "RANGE_RAILS"
    barriers_valid = bool(
        forecast_target is not None
        and forecast_invalidation is not None
        and intent.tp > 0.0
    )
    tp_touched = False
    target_touched = False
    invalidation_touched = False
    order_entry_touched = False
    order_sl_touched = False
    same_candle_both_touched = False
    first_tp_touch_utc: str | None = None
    first_target_touch_utc: str | None = None
    first_invalidation_touch_utc: str | None = None
    first_order_entry_touch_utc: str | None = None
    first_order_sl_touch_utc: str | None = None
    if coverage_ok and barriers_valid:
        for candle_ts, _, high, low, _, _ in parsed:
            if intent.side == Side.LONG:
                candle_tp = high >= intent.tp
                candle_target = high >= forecast_target
                candle_invalidation = low <= forecast_invalidation
                candle_entry = bool(
                    range_path
                    and intent.entry is not None
                    and low <= intent.entry
                )
                candle_sl = bool(range_path and low <= intent.sl)
            else:
                candle_tp = low <= intent.tp
                candle_target = low <= forecast_target
                candle_invalidation = high >= forecast_invalidation
                candle_entry = bool(
                    range_path
                    and intent.entry is not None
                    and high >= intent.entry
                )
                candle_sl = bool(range_path and high >= intent.sl)
            tp_touched = tp_touched or candle_tp
            target_touched = target_touched or candle_target
            invalidation_touched = invalidation_touched or candle_invalidation
            order_entry_touched = order_entry_touched or candle_entry
            order_sl_touched = order_sl_touched or candle_sl
            if candle_tp and first_tp_touch_utc is None:
                first_tp_touch_utc = candle_ts.isoformat()
            if candle_target and first_target_touch_utc is None:
                first_target_touch_utc = candle_ts.isoformat()
            if candle_invalidation and first_invalidation_touch_utc is None:
                first_invalidation_touch_utc = candle_ts.isoformat()
            if candle_entry and first_order_entry_touch_utc is None:
                first_order_entry_touch_utc = candle_ts.isoformat()
            if candle_sl and first_order_sl_touch_utc is None:
                first_order_sl_touch_utc = candle_ts.isoformat()
            same_candle_both_touched = same_candle_both_touched or (
                (candle_tp or candle_target) and candle_invalidation
            )
    no_touch = bool(
        coverage_ok
        and barriers_valid
        and not tp_touched
        and not target_touched
        and not invalidation_touched
        and not order_entry_touched
        and not order_sl_touched
    )
    reason = (
        "FORECAST_S5_NO_TOUCH_PROVEN"
        if no_touch
        else "FORECAST_S5_CANDLE_INVALID"
        if malformed
        else "FORECAST_S5_COVERAGE_INCOMPLETE"
        if not coverage_ok
        else "FORECAST_S5_BARRIER_INVALID"
        if not barriers_valid
        else "FORECAST_S5_BOTH_BARRIERS_TOUCHED"
        if same_candle_both_touched
        else "FORECAST_S5_TP_OR_TARGET_TOUCHED"
        if tp_touched or target_touched
        else "FORECAST_S5_RANGE_ENTRY_OR_SL_TOUCHED"
        if order_entry_touched or order_sl_touched
        else "FORECAST_S5_INVALIDATION_TOUCHED"
    )
    candle_rows = [
        {
            "time": row[0].isoformat(),
            "o": row[1],
            "h": row[2],
            "l": row[3],
            "c": row[4],
            "complete": row[5],
        }
        for row in parsed
    ]
    material = {
        "status": "PASSED" if no_touch else "BLOCKED",
        "reason": reason,
        "pair": intent.pair,
        "side": intent.side.value,
        "forecast_cycle_id": cycle_id,
        "forecast_emitted_at_utc": emission.isoformat(),
        "quote_timestamp_utc": quote_time.isoformat(),
        "window_seconds": window_seconds,
        "from_floor_expansion_seconds": (
            emission - first_start
        ).total_seconds(),
        "to_ceil_expansion_seconds": (
            query_to - quote_time
        ).total_seconds(),
        "query": query,
        "response_instrument": payload.get("instrument"),
        "response_granularity": payload.get("granularity"),
        "expected_candles": expected_count,
        "observed_candles": len(parsed),
        "first_candle_start_utc": (
            timestamps[0].isoformat() if timestamps else None
        ),
        "last_candle_start_utc": (
            timestamps[-1].isoformat() if timestamps else None
        ),
        "tail_complete": parsed[-1][5] if parsed else None,
        # Keep the canonical rows in the proof. A digest alone cannot be
        # semantically revalidated at the next boundary: any arbitrary 64-byte
        # string could otherwise be wrapped in a newly valid outer hash.
        "candle_rows": candle_rows,
        "candle_rows_sha256": _canonical_json_sha256(candle_rows),
        "max_high": max((row[2] for row in parsed), default=None),
        "min_low": min((row[3] for row in parsed), default=None),
        "coverage_ok": coverage_ok,
        "malformed_candle": malformed,
        "order_tp": intent.tp,
        "order_entry": intent.entry,
        "order_sl": intent.sl,
        "barrier_basis": barrier_basis,
        "forecast_target": forecast_target,
        "forecast_invalidation": forecast_invalidation,
        "tp_touched": tp_touched,
        "target_touched": target_touched,
        "invalidation_touched": invalidation_touched,
        "order_entry_touched": order_entry_touched,
        "order_sl_touched": order_sl_touched,
        "same_candle_both_touched": same_candle_both_touched,
        "first_tp_touch_utc": first_tp_touch_utc,
        "first_target_touch_utc": first_target_touch_utc,
        "first_invalidation_touch_utc": first_invalidation_touch_utc,
        "first_order_entry_touch_utc": first_order_entry_touch_utc,
        "first_order_sl_touch_utc": first_order_sl_touch_utc,
        "no_touch": no_touch,
    }
    return finished(material)


def _forecast_s5_path_proof_valid(
    proof: Any,
    *,
    intent: OrderIntent,
    snapshot: BrokerSnapshot,
    bypass_reason: str | None,
) -> bool:
    if not isinstance(proof, dict):
        return False
    proof_sha256 = str(proof.get("proof_sha256") or "")
    material = dict(proof)
    material.pop("proof_sha256", None)
    if proof_sha256 != _canonical_json_sha256(material):
        return False
    if str(proof.get("pair") or "") != intent.pair:
        return False
    if bypass_reason is not None:
        return bool(
            str(proof.get("status") or "").upper() == "BYPASSED"
            and str(proof.get("reason") or "") == bypass_reason
        )
    metadata = dict(intent.metadata or {})
    quote = snapshot.quotes.get(intent.pair)
    if quote is None:
        return False
    expected_target, expected_invalidation, expected_barrier_basis = (
        _forecast_s5_barriers(intent)
    )
    if expected_target is None or expected_invalidation is None:
        return False
    cycle_id = str(metadata.get("forecast_cycle_id") or "").strip()
    match = PRE_ENTRY_FORECAST_CYCLE_RE.match(cycle_id)
    emission = _parse_utc_timestamp(match.group("ts")) if match else None
    if emission is None:
        return False
    emission = emission.astimezone(timezone.utc)
    quote_time = quote.timestamp_utc.astimezone(timezone.utc)
    first_start = datetime.fromtimestamp(
        math.floor(emission.timestamp() / FORECAST_S5_SECONDS)
        * FORECAST_S5_SECONDS,
        tz=timezone.utc,
    )
    last_start = datetime.fromtimestamp(
        math.floor(quote_time.timestamp() / FORECAST_S5_SECONDS)
        * FORECAST_S5_SECONDS,
        tz=timezone.utc,
    )
    query_to = last_start + timedelta(seconds=FORECAST_S5_SECONDS)
    expected_query = {
        "granularity": "S5",
        "from": first_start.isoformat().replace("+00:00", "Z"),
        "to": query_to.isoformat().replace("+00:00", "Z"),
        "price": "M",
        "includeFirst": "true",
        "smooth": "false",
    }
    window_seconds = (quote_time - emission).total_seconds()
    expected_count = int(
        (last_start - first_start).total_seconds() / FORECAST_S5_SECONDS
    ) + 1
    rows = proof.get("candle_rows")
    if not isinstance(rows, list) or len(rows) != expected_count:
        return False
    parsed_rows: list[tuple[datetime, float, float, float, float, bool]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "time",
            "o",
            "h",
            "l",
            "c",
            "complete",
        }:
            return False
        timestamp = _parse_utc_timestamp(row.get("time"))
        values = (row.get("o"), row.get("h"), row.get("l"), row.get("c"))
        if (
            timestamp is None
            or not isinstance(row.get("complete"), bool)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in values
            )
        ):
            return False
        open_price, high, low, close = (float(value) for value in values)
        if (
            low > open_price
            or low > close
            or high < open_price
            or high < close
            or low > high
        ):
            return False
        parsed_rows.append(
            (
                timestamp.astimezone(timezone.utc),
                open_price,
                high,
                low,
                close,
                row["complete"],
            )
        )
    row_times = [row[0] for row in parsed_rows]
    if not (
        row_times[0] == first_start
        and row_times[-1] == last_start
        and len(set(row_times)) == len(row_times)
        and all(
            (right - left).total_seconds() == FORECAST_S5_SECONDS
            for left, right in zip(row_times, row_times[1:])
        )
        and row_times[0] <= emission
        < row_times[0] + timedelta(seconds=FORECAST_S5_SECONDS)
        and row_times[-1] <= quote_time
        < row_times[-1] + timedelta(seconds=FORECAST_S5_SECONDS)
        and all(row[5] is True for row in parsed_rows[:-1])
    ):
        return False
    recomputed_touch = {
        "tp_touched": False,
        "target_touched": False,
        "invalidation_touched": False,
        "same_candle_both_touched": False,
        "order_entry_touched": False,
        "order_sl_touched": False,
        "first_tp_touch_utc": None,
        "first_target_touch_utc": None,
        "first_invalidation_touch_utc": None,
        "first_order_entry_touch_utc": None,
        "first_order_sl_touch_utc": None,
    }
    for candle_time, _, high, low, _, _ in parsed_rows:
        if intent.side == Side.LONG:
            candle_tp = high >= intent.tp
            candle_target = high >= float(expected_target)
            candle_invalidation = low <= float(expected_invalidation)
            candle_entry = bool(
                expected_barrier_basis == "RANGE_RAILS"
                and intent.entry is not None
                and low <= intent.entry
            )
            candle_sl = bool(
                expected_barrier_basis == "RANGE_RAILS"
                and low <= intent.sl
            )
        else:
            candle_tp = low <= intent.tp
            candle_target = low <= float(expected_target)
            candle_invalidation = high >= float(expected_invalidation)
            candle_entry = bool(
                expected_barrier_basis == "RANGE_RAILS"
                and intent.entry is not None
                and high >= intent.entry
            )
            candle_sl = bool(
                expected_barrier_basis == "RANGE_RAILS"
                and high >= intent.sl
            )
        for key, touched in (
            ("tp_touched", candle_tp),
            ("target_touched", candle_target),
            ("invalidation_touched", candle_invalidation),
        ):
            recomputed_touch[key] = recomputed_touch[key] or touched
        if candle_tp and recomputed_touch["first_tp_touch_utc"] is None:
            recomputed_touch["first_tp_touch_utc"] = candle_time.isoformat()
        if (
            candle_target
            and recomputed_touch["first_target_touch_utc"] is None
        ):
            recomputed_touch["first_target_touch_utc"] = candle_time.isoformat()
        if (
            candle_invalidation
            and recomputed_touch["first_invalidation_touch_utc"] is None
        ):
            recomputed_touch["first_invalidation_touch_utc"] = (
                candle_time.isoformat()
            )
        for key, touched in (
            ("order_entry_touched", candle_entry),
            ("order_sl_touched", candle_sl),
        ):
            recomputed_touch[key] = recomputed_touch[key] or touched
        if (
            candle_entry
            and recomputed_touch["first_order_entry_touch_utc"] is None
        ):
            recomputed_touch["first_order_entry_touch_utc"] = (
                candle_time.isoformat()
            )
        if (
            candle_sl
            and recomputed_touch["first_order_sl_touch_utc"] is None
        ):
            recomputed_touch["first_order_sl_touch_utc"] = (
                candle_time.isoformat()
            )
        recomputed_touch["same_candle_both_touched"] = bool(
            recomputed_touch["same_candle_both_touched"]
            or ((candle_tp or candle_target) and candle_invalidation)
        )
    if any(proof.get(key) != value for key, value in recomputed_touch.items()):
        return False
    rows_sha256 = _canonical_json_sha256(rows)
    numeric_fields = (
        (proof.get("order_tp"), intent.tp),
        (proof.get("forecast_target"), expected_target),
        (proof.get("forecast_invalidation"), expected_invalidation),
    )
    if any(
        expected is None
        or isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(float(actual))
        or not math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for actual, expected in numeric_fields
    ):
        return False
    if expected_barrier_basis == "RANGE_RAILS":
        for actual, expected in (
            (proof.get("order_entry"), intent.entry),
            (proof.get("order_sl"), intent.sl),
        ):
            if (
                expected is None
                or isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(float(actual))
                or not math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                return False
    return bool(
        str(proof.get("status") or "").upper() == "PASSED"
        and str(proof.get("reason") or "")
        == "FORECAST_S5_NO_TOUCH_PROVEN"
        and str(proof.get("barrier_basis") or "")
        == expected_barrier_basis
        and str(proof.get("side") or "").upper() == intent.side.value
        and str(proof.get("forecast_cycle_id") or "")
        == cycle_id
        and str(proof.get("quote_timestamp_utc") or "")
        == quote_time.isoformat()
        and str(proof.get("forecast_emitted_at_utc") or "")
        == emission.isoformat()
        and isinstance(proof.get("window_seconds"), (int, float))
        and not isinstance(proof.get("window_seconds"), bool)
        and math.isclose(
            float(proof["window_seconds"]),
            window_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and 0.0 <= window_seconds <= FORECAST_S5_MAX_WINDOW_SECONDS
        and proof.get("query") == expected_query
        and "count" not in proof.get("query", {})
        and proof.get("expected_candles") == expected_count
        and proof.get("observed_candles") == expected_count
        and str(proof.get("first_candle_start_utc") or "")
        == first_start.isoformat()
        and str(proof.get("last_candle_start_utc") or "")
        == last_start.isoformat()
        and isinstance(proof.get("tail_complete"), bool)
        and proof.get("candle_rows_sha256") == rows_sha256
        and proof.get("tail_complete") is parsed_rows[-1][5]
        and isinstance(proof.get("max_high"), (int, float))
        and not isinstance(proof.get("max_high"), bool)
        and math.isclose(
            float(proof["max_high"]),
            max(row[2] for row in parsed_rows),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and isinstance(proof.get("min_low"), (int, float))
        and not isinstance(proof.get("min_low"), bool)
        and math.isclose(
            float(proof["min_low"]),
            min(row[3] for row in parsed_rows),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and float(proof["min_low"]) <= float(proof["max_high"])
        and proof.get("coverage_ok") is True
        and proof.get("malformed_candle") is False
        and proof.get("no_touch") is True
        and proof.get("tp_touched") is False
        and proof.get("target_touched") is False
        and proof.get("invalidation_touched") is False
        and proof.get("order_entry_touched") is False
        and proof.get("order_sl_touched") is False
        and proof.get("same_candle_both_touched") is False
        and str(proof.get("response_instrument") or "") == intent.pair
        and str(proof.get("response_granularity") or "").upper() == "S5"
    )


def _capital_allocation_market_price_bound(
    *,
    intent: OrderIntent,
    snapshot: BrokerSnapshot,
    effective_max_loss_jpy: float,
    reserved_price_bound: Any = None,
    execution_cost_floor: dict[str, Any] | None = None,
    portfolio_loss_remaining_jpy: float | None = None,
    m15_recovery_prebounded: bool = False,
) -> tuple[dict[str, Any], RiskIssue | None]:
    """Derive/reprove the worst MARKET fill allowed by EV and risk truth.

    OANDA MARKET orders otherwise have an unbounded fill price.  A signal can
    pass at the observed ask/bid and still lose its Wilson EV, quarter-Kelly
    budget, or current-NAV loss cap in a gap before FOK execution.  This proof
    derives a side-aware, tick-rounded `priceBound`; when a reserved bound is
    supplied it validates that exact immutable transport value.
    """

    def finished(material: dict[str, Any]) -> dict[str, Any]:
        return {
            **material,
            "proof_sha256": _canonical_json_sha256(material),
        }

    # Lazy for the same import-cycle reason as the numeric ceiling adapter.
    # One shared constant prevents the order transport cap from drifting away
    # from the overlay/gateway quarter-Kelly sizing contract.
    from quant_rabbit.market_read_overlay import (
        CAPITAL_ALLOCATION_KELLY_FRACTION,
    )

    kelly_fraction = float(CAPITAL_ALLOCATION_KELLY_FRACTION)
    submitted_tp = intent.tp
    submitted_sl = intent.sl
    intent = _intent_with_broker_price_precision(intent)
    metadata = dict(intent.metadata or {})
    hedge = str(metadata.get("position_intent") or "").strip().upper() == "HEDGE"
    scout = _numeric_predictive_scout_contract(intent)
    range_tp_prebounded = _range_tp_proven_prebounded_intent_claim(intent)
    m15_recovery = bool(
        m15_recovery_prebounded
        and _m15_recovery_prebounded_intent_claim(intent)
    )
    if hedge or scout or range_tp_prebounded or m15_recovery:
        bypass_reason = (
            "HEDGE_RISK_REDUCTION_PREBOUNDED_CONTRACT"
            if hedge
            else M15_RECOVERY_PREBOUNDED_REASON
            if m15_recovery
            else "PREDICTIVE_SCOUT_PREBOUNDED_CONTRACT"
            if scout
            else "TP_PROVEN_RANGE_NONMARKET_PREBOUNDED_CONTRACT"
        )
        if (range_tp_prebounded or m15_recovery) and reserved_price_bound is not None:
            evidence = finished(
                {
                    "status": "BLOCKED",
                    "reason": (
                        "M15_RECOVERY_PRICE_BOUND_FORBIDDEN"
                        if m15_recovery
                        else "RANGE_LIMIT_PRICE_BOUND_FORBIDDEN"
                    ),
                    "pair": intent.pair,
                    "side": intent.side.value,
                    "price_bound_text": None,
                }
            )
            return evidence, RiskIssue(
                "PRE_POST_GPT_ALLOCATION_PRICE_BOUND_REPROOF_FAILED",
                (
                    "M15 recovery STOP-ENTRY must not carry a MARKET priceBound"
                    if m15_recovery
                    else "passive RANGE LIMIT must not carry a MARKET priceBound"
                ),
            )
        evidence = finished(
            {
                "status": "BYPASSED",
                "reason": bypass_reason,
                "pair": intent.pair,
                "side": intent.side.value,
                "price_bound_text": None,
            }
        )
        return evidence, None

    quote = snapshot.quotes.get(intent.pair)
    account = snapshot.account
    quote_currency = (
        intent.pair.split("_", 1)[1] if "_" in intent.pair else ""
    )
    quote_to_jpy = (
        1.0
        if quote_currency == "JPY"
        else _positive_float(snapshot.home_conversions.get(quote_currency))
    )
    nav_jpy = (
        _positive_float(account.nav_jpy) if account is not None else None
    )
    max_loss_cap = _positive_float(effective_max_loss_jpy)
    cost_floor = (
        dict(execution_cost_floor)
        if isinstance(execution_cost_floor, dict)
        else {}
    )
    claimed_cost_sha = str(cost_floor.get("proof_sha256") or "")
    cost_material = dict(cost_floor)
    cost_material.pop("proof_sha256", None)
    cost_digest_valid = bool(
        re.fullmatch(r"[0-9a-f]{64}", claimed_cost_sha)
        and claimed_cost_sha == _canonical_json_sha256(cost_material)
    )
    entry_slippage_p95 = _nonnegative_float(
        cost_floor.get("market_entry_adverse_p95_pips")
    )
    audited_exit_p95 = _nonnegative_float(
        cost_floor.get("audited_protected_exit_adverse_p95_pips")
    )
    take_profit_exit_p95 = _nonnegative_float(
        cost_floor.get("take_profit_exit_adverse_p95_pips")
    )
    stop_loss_exit_p95 = _nonnegative_float(
        cost_floor.get("stop_loss_exit_adverse_p95_pips")
    )
    if take_profit_exit_p95 is None:
        take_profit_exit_p95 = audited_exit_p95
    if stop_loss_exit_p95 is None:
        stop_loss_exit_p95 = audited_exit_p95
    financing_per_unit = _nonnegative_float(
        cost_floor.get("financing_adverse_stress_jpy_per_unit")
    )
    cost_method = (
        intent.market_context.method.value
        if intent.market_context is not None
        and isinstance(intent.market_context.method, TradeMethod)
        else "UNKNOWN"
    )
    expected_cost_scope_key = "|".join(
        (
            intent.pair.upper(),
            intent.side.value.upper(),
            str(cost_method).upper(),
            "MARKET",
        )
    )
    cost_floor_valid = bool(
        cost_floor.get("status") == "PASSED"
        and cost_floor.get("contract") == EXECUTION_COST_FLOOR_CONTRACT
        and str(cost_floor.get("scope_key") or "").upper()
        == expected_cost_scope_key
        and cost_floor.get("spread_double_count_forbidden") is True
        and cost_digest_valid
        and entry_slippage_p95 is not None
        and audited_exit_p95 is not None
        and take_profit_exit_p95 is not None
        and stop_loss_exit_p95 is not None
        and financing_per_unit is not None
    )
    raw_hit_rate = metadata.get("forecast_directional_economic_hit_rate")
    raw_samples = metadata.get("forecast_directional_economic_samples")
    hit_rate = (
        float(raw_hit_rate)
        if isinstance(raw_hit_rate, (int, float))
        and not isinstance(raw_hit_rate, bool)
        and math.isfinite(float(raw_hit_rate))
        and 0.0 <= float(raw_hit_rate) <= 1.0
        else None
    )
    samples = (
        raw_samples
        if isinstance(raw_samples, int)
        and not isinstance(raw_samples, bool)
        and raw_samples > 0
        else None
    )
    wilson_lower = hit_rate_wilson_lower(hit_rate, samples)
    units = abs(int(intent.units))
    current_entry = (
        quote.ask
        if quote is not None and intent.side == Side.LONG
        else quote.bid
        if quote is not None and intent.side == Side.SHORT
        else None
    )
    spread = (
        quote.ask - quote.bid
        if quote is not None
        and math.isfinite(float(quote.ask))
        and math.isfinite(float(quote.bid))
        and quote.ask > quote.bid > 0.0
        else None
    )
    tick = _price_tick(intent.pair)
    invalid_inputs = bool(
        intent.order_type != OrderType.MARKET
        or units <= 0
        or quote is None
        or current_entry is None
        or not math.isfinite(float(current_entry))
        or float(current_entry) <= 0.0
        or spread is None
        or quote_to_jpy is None
        or nav_jpy is None
        or max_loss_cap is None
        or not cost_floor_valid
        or wilson_lower is None
        or not 0.0 < float(wilson_lower) < 1.0
        or not math.isfinite(float(intent.tp))
        or not math.isfinite(float(intent.sl))
        or intent.tp <= 0.0
        or intent.sl <= 0.0
    )
    if invalid_inputs:
        evidence = finished(
            {
                "status": "BLOCKED",
                "reason": "PRICE_BOUND_INPUTS_INVALID",
                "pair": intent.pair,
                "side": intent.side.value,
                "order_type": intent.order_type.value,
                "units": units,
                "current_executable_entry": current_entry,
                "spread": spread,
                "quote_to_jpy": quote_to_jpy,
                "nav_jpy": nav_jpy,
                "effective_max_loss_jpy": max_loss_cap,
                "economic_wilson95_lower": wilson_lower,
                "price_bound_text": None,
            }
        )
        return evidence, RiskIssue(
            "PRE_POST_GPT_ALLOCATION_PRICE_BOUND_INPUTS_INVALID",
            "ordinary schema-v2 MARKET allocation cannot derive a finite "
            "worst-fill priceBound from the fresh quote, conversion, NAV, "
            "Wilson probability, TP/SL, units, and effective loss cap",
        )

    assert current_entry is not None
    assert quote_to_jpy is not None
    assert nav_jpy is not None
    assert max_loss_cap is not None
    assert wilson_lower is not None
    monetary_per_price = units * quote_to_jpy
    pip_factor = instrument_pip_factor(intent.pair)
    current_spread_pips = spread * pip_factor
    # Entry and TP/SL prices are already executable-side prices, so the normal
    # bid/ask spread is in the geometry.  Only separately observed protected-
    # order slippage belongs here, and TP/SL tails must remain outcome-specific.
    take_profit_exit_stress_pips = float(take_profit_exit_p95)
    stop_loss_exit_stress_pips = float(stop_loss_exit_p95)
    take_profit_exit_stress_jpy = (
        take_profit_exit_stress_pips * units / pip_factor * quote_to_jpy
    )
    stop_loss_exit_stress_jpy = (
        stop_loss_exit_stress_pips * units / pip_factor * quote_to_jpy
    )
    financing_stress_jpy = float(financing_per_unit) * units
    win_outcome_cost_jpy = take_profit_exit_stress_jpy + financing_stress_jpy
    loss_outcome_cost_jpy = stop_loss_exit_stress_jpy + financing_stress_jpy
    expected_outcome_cost_jpy = (
        float(wilson_lower) * win_outcome_cost_jpy
        + (1.0 - float(wilson_lower)) * loss_outcome_cost_jpy
    )
    # Compatibility field consumed by the final loss-cap recheck.  It is the
    # loss-path cost, not a probability-weighted expected cost.
    outcome_cost_jpy = loss_outcome_cost_jpy
    total_range_price = (
        intent.tp - intent.sl
        if intent.side == Side.LONG
        else intent.sl - intent.tp
    )
    total_range_jpy = total_range_price * monetary_per_price
    nav_hard_cap_jpy = nav_jpy * (FRESH_ENTRY_MAX_RISK_PCT_NAV / 100.0)
    quarter_kelly_nav_jpy = (
        nav_jpy * kelly_fraction
    )
    discriminant = (
        (total_range_jpy + quarter_kelly_nav_jpy) ** 2
        - 4.0
        * float(wilson_lower)
        * quarter_kelly_nav_jpy
        * total_range_jpy
    )
    kelly_risk_cap_jpy = (
        (
            2.0
            * float(wilson_lower)
            * quarter_kelly_nav_jpy
            * total_range_jpy
        )
        / (
            total_range_jpy
            + quarter_kelly_nav_jpy
            + math.sqrt(max(0.0, discriminant))
        )
        if total_range_jpy > 0.0
        and quarter_kelly_nav_jpy > 0.0
        and discriminant >= 0.0
        else None
    )
    if kelly_risk_cap_jpy is None or kelly_risk_cap_jpy <= 0.0:
        evidence = finished(
            {
                "status": "BLOCKED",
                "reason": "PRICE_BOUND_QUARTER_KELLY_ROOT_INVALID",
                "pair": intent.pair,
                "side": intent.side.value,
                "total_range_jpy": total_range_jpy,
                "quarter_kelly_nav_jpy": quarter_kelly_nav_jpy,
                "discriminant": discriminant,
                "price_bound_text": None,
            }
        )
        return evidence, RiskIssue(
            "PRE_POST_GPT_ALLOCATION_PRICE_BOUND_KELLY_INVALID",
            "ordinary MARKET allocation has no positive quarter-Kelly "
            "worst-fill root at the fresh NAV",
        )

    net_downside_caps = [
        max_loss_cap,
        nav_hard_cap_jpy,
        kelly_risk_cap_jpy,
        float(wilson_lower) * total_range_jpy,
    ]
    portfolio_remaining = _nonnegative_float(portfolio_loss_remaining_jpy)
    if portfolio_remaining is not None:
        net_downside_caps.append(portfolio_remaining)
    risk_cap_jpy = min(net_downside_caps) - outcome_cost_jpy
    if risk_cap_jpy <= 0.0:
        evidence = finished(
            {
                "status": "BLOCKED",
                "reason": "EXECUTION_COST_EXHAUSTS_PRICE_BOUND_RISK_CAP",
                "pair": intent.pair,
                "side": intent.side.value,
                "outcome_cost_jpy": outcome_cost_jpy,
                "net_downside_caps": net_downside_caps,
                "execution_cost_floor": cost_floor,
                "price_bound_text": None,
            }
        )
        return evidence, RiskIssue(
            "PRE_POST_GPT_ALLOCATION_PRICE_BOUND_COST_EXHAUSTS_CAP",
            "dynamic exit/financing cost exhausts the MARKET priceBound risk capacity",
        )
    risk_distance = risk_cap_jpy / monetary_per_price
    ev_raw_risk_cap_jpy = (
        float(wilson_lower) * total_range_jpy - expected_outcome_cost_jpy
    )
    ev_zero_entry = (
        intent.sl + ev_raw_risk_cap_jpy / monetary_per_price
        if intent.side == Side.LONG
        else intent.sl - ev_raw_risk_cap_jpy / monetary_per_price
    )
    if intent.side == Side.LONG:
        cap_entry = intent.sl + risk_distance
        raw_adverse_limit = min(cap_entry, ev_zero_entry, intent.tp)
        # LONG upper bound rounds down; subtract an infinitesimal tick ratio so
        # a root lying exactly on a tick moves one full tick inside strict EV.
        derived_price_bound = (
            math.floor(raw_adverse_limit / tick - 1e-9) * tick
        )
    else:
        cap_entry = intent.sl - risk_distance
        raw_adverse_limit = max(cap_entry, ev_zero_entry, intent.tp)
        # SHORT lower bound rounds up for the symmetric strict boundary.
        derived_price_bound = (
            math.ceil(raw_adverse_limit / tick + 1e-9) * tick
        )
    derived_price_bound = round(
        derived_price_bound,
        _price_precision(intent.pair),
    )
    price_bound_text = _price(intent.pair, derived_price_bound)

    supplied_bound = None
    supplied_bound_valid = reserved_price_bound is None
    if reserved_price_bound is not None:
        try:
            supplied_bound = float(str(reserved_price_bound).strip())
        except (TypeError, ValueError):
            supplied_bound = None
        supplied_bound_valid = bool(
            supplied_bound is not None
            and math.isfinite(supplied_bound)
            and supplied_bound > 0.0
            and str(reserved_price_bound).strip()
            == _price(intent.pair, supplied_bound)
        )
    worst_fill_entry = (
        supplied_bound if supplied_bound is not None else derived_price_bound
    )
    bound_not_looser = bool(
        supplied_bound_valid
        and (
            worst_fill_entry <= derived_price_bound + 1e-12
            if intent.side == Side.LONG
            else worst_fill_entry >= derived_price_bound - 1e-12
        )
    )
    current_inside_bound = bool(
        float(current_entry) <= worst_fill_entry + 1e-12
        if intent.side == Side.LONG
        else float(current_entry) >= worst_fill_entry - 1e-12
    )
    entry_slippage_allowance_pips = (
        (worst_fill_entry - float(current_entry)) * pip_factor
        if intent.side == Side.LONG
        else (float(current_entry) - worst_fill_entry) * pip_factor
    )
    entry_slippage_allowance_passed = bool(
        entry_slippage_allowance_pips + 1e-9 >= float(entry_slippage_p95)
    )
    geometry_passed = bool(
        intent.sl < worst_fill_entry < intent.tp
        if intent.side == Side.LONG
        else intent.tp < worst_fill_entry < intent.sl
    )
    worst_risk_jpy = (
        abs(worst_fill_entry - intent.sl) * monetary_per_price
    )
    worst_reward_jpy = (
        abs(intent.tp - worst_fill_entry) * monetary_per_price
    )
    worst_net_risk_jpy = worst_risk_jpy + loss_outcome_cost_jpy
    worst_net_reward_jpy = worst_reward_jpy - win_outcome_cost_jpy
    worst_reward_risk = (
        worst_net_reward_jpy / worst_net_risk_jpy
        if worst_net_risk_jpy > 0.0 and worst_net_reward_jpy > 0.0
        else None
    )
    worst_ev_lower_jpy = (
        float(wilson_lower) * worst_net_reward_jpy
        - (1.0 - float(wilson_lower)) * worst_net_risk_jpy
    )
    full_kelly_fraction = (
        float(wilson_lower)
        - (1.0 - float(wilson_lower)) / worst_reward_risk
        if worst_reward_risk is not None and worst_reward_risk > 0.0
        else None
    )
    worst_quarter_kelly_budget_jpy = (
        nav_jpy
        * kelly_fraction
        * full_kelly_fraction
        if full_kelly_fraction is not None
        else None
    )
    ev_passed = bool(
        worst_ev_lower_jpy > 0.0
        and not math.isclose(
            worst_ev_lower_jpy,
            0.0,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    )
    kelly_passed = bool(
        worst_quarter_kelly_budget_jpy is not None
        and worst_quarter_kelly_budget_jpy > 0.0
        and worst_net_risk_jpy <= worst_quarter_kelly_budget_jpy + 1e-9
    )
    cap_passed = bool(
        worst_net_risk_jpy <= max_loss_cap + 1e-9
        and worst_net_risk_jpy <= nav_hard_cap_jpy + 1e-9
        and (
            portfolio_remaining is None
            or worst_net_risk_jpy <= portfolio_remaining + 1e-9
        )
    )
    next_adverse_entry = (
        derived_price_bound + tick
        if intent.side == Side.LONG
        else derived_price_bound - tick
    )
    next_risk_jpy = abs(next_adverse_entry - intent.sl) * monetary_per_price
    next_reward_jpy = abs(intent.tp - next_adverse_entry) * monetary_per_price
    next_net_risk_jpy = next_risk_jpy + loss_outcome_cost_jpy
    next_net_reward_jpy = next_reward_jpy - win_outcome_cost_jpy
    next_geometry = bool(
        intent.sl < next_adverse_entry < intent.tp
        if intent.side == Side.LONG
        else intent.tp < next_adverse_entry < intent.sl
    )
    next_reward_risk = (
        next_net_reward_jpy / next_net_risk_jpy
        if next_geometry
        and next_net_risk_jpy > 0.0
        and next_net_reward_jpy > 0.0
        else None
    )
    next_ev_jpy = (
        float(wilson_lower) * next_net_reward_jpy
        - (1.0 - float(wilson_lower)) * next_net_risk_jpy
    )
    next_full_kelly = (
        float(wilson_lower)
        - (1.0 - float(wilson_lower)) / next_reward_risk
        if next_reward_risk is not None and next_reward_risk > 0.0
        else None
    )
    next_kelly_budget_jpy = (
        nav_jpy * kelly_fraction * next_full_kelly
        if next_full_kelly is not None
        else None
    )
    next_adverse_tick_passes = bool(
        next_geometry
        and next_ev_jpy > 0.0
        and next_kelly_budget_jpy is not None
        and next_kelly_budget_jpy > 0.0
        and next_net_risk_jpy <= next_kelly_budget_jpy + 1e-9
        and next_net_risk_jpy <= max_loss_cap + 1e-9
        and next_net_risk_jpy <= nav_hard_cap_jpy + 1e-9
        and (
            portfolio_remaining is None
            or next_net_risk_jpy <= portfolio_remaining + 1e-9
        )
    )
    next_adverse_tick_fails = not next_adverse_tick_passes
    passed = bool(
        supplied_bound_valid
        and bound_not_looser
        and current_inside_bound
        and entry_slippage_allowance_passed
        and geometry_passed
        and ev_passed
        and kelly_passed
        and cap_passed
        and next_adverse_tick_fails
    )
    reason = (
        "WORST_FILL_PRICE_BOUND_PROVEN"
        if passed
        else "RESERVED_PRICE_BOUND_INVALID"
        if not supplied_bound_valid
        else "RESERVED_PRICE_BOUND_LOOSER_THAN_FRESH_CEILING"
        if not bound_not_looser
        else "CURRENT_EXECUTABLE_PRICE_OUTSIDE_BOUND"
        if not current_inside_bound
        else "PRICE_BOUND_ENTRY_SLIPPAGE_ALLOWANCE_BELOW_P95"
        if not entry_slippage_allowance_passed
        else "WORST_FILL_TP_SL_GEOMETRY_INVALID"
        if not geometry_passed
        else "WORST_FILL_CONSERVATIVE_EV_NOT_POSITIVE"
        if not ev_passed
        else "WORST_FILL_QUARTER_KELLY_CAP_EXCEEDED"
        if not kelly_passed
        else "PRICE_BOUND_NOT_MAXIMAL_ON_TICK_GRID"
        if not next_adverse_tick_fails
        else "WORST_FILL_LOSS_CAP_EXCEEDED"
    )
    material = {
        "status": "PASSED" if passed else "BLOCKED",
        "reason": reason,
        "contract": "QR_GPT_MARKET_WORST_FILL_PRICE_BOUND_V1",
        "pair": intent.pair,
        "side": intent.side.value,
        "method": cost_method,
        "expected_execution_cost_scope_key": expected_cost_scope_key,
        "units": units,
        "current_executable_entry": float(current_entry),
        "spread": spread,
        "submitted_tp": submitted_tp,
        "submitted_sl": submitted_sl,
        "tp": intent.tp,
        "sl": intent.sl,
        "tp_transport_text": _price(intent.pair, intent.tp),
        "sl_transport_text": _price(intent.pair, intent.sl),
        "dependent_price_basis": "EXACT_OANDA_TRANSPORT_PRECISION",
        "quote_to_jpy": quote_to_jpy,
        "monetary_per_price": monetary_per_price,
        "nav_jpy": nav_jpy,
        "nav_hard_cap_pct": FRESH_ENTRY_MAX_RISK_PCT_NAV,
        "nav_hard_cap_jpy": nav_hard_cap_jpy,
        "effective_max_loss_jpy": max_loss_cap,
        "portfolio_loss_remaining_jpy": portfolio_remaining,
        "execution_cost_floor": cost_floor,
        "execution_cost_floor_digest_valid": cost_digest_valid,
        "market_entry_adverse_p95_pips": entry_slippage_p95,
        "audited_protected_exit_adverse_p95_pips": audited_exit_p95,
        "fresh_spread_pips": current_spread_pips,
        "take_profit_exit_adverse_p95_pips": take_profit_exit_p95,
        "stop_loss_exit_adverse_p95_pips": stop_loss_exit_p95,
        "take_profit_exit_execution_stress_pips": (
            take_profit_exit_stress_pips
        ),
        "stop_loss_exit_execution_stress_pips": stop_loss_exit_stress_pips,
        "take_profit_exit_execution_stress_jpy": take_profit_exit_stress_jpy,
        "stop_loss_exit_execution_stress_jpy": stop_loss_exit_stress_jpy,
        # Backward-readable aliases for the protected loss path.  Frozen-proof
        # validation below uses the explicit win/loss fields for new proofs.
        "exit_execution_stress_pips": stop_loss_exit_stress_pips,
        "exit_execution_stress_jpy": stop_loss_exit_stress_jpy,
        "financing_adverse_stress_jpy_per_unit": financing_per_unit,
        "financing_stress_jpy": financing_stress_jpy,
        "win_outcome_cost_jpy": win_outcome_cost_jpy,
        "loss_outcome_cost_jpy": loss_outcome_cost_jpy,
        "expected_outcome_cost_jpy": expected_outcome_cost_jpy,
        "outcome_cost_jpy": outcome_cost_jpy,
        "economic_hit_rate": hit_rate,
        "economic_samples": samples,
        "economic_wilson95_lower": wilson_lower,
        "kelly_fractional_multiplier": (
            kelly_fraction
        ),
        "kelly_fraction_source": (
            "quant_rabbit.market_read_overlay."
            "CAPITAL_ALLOCATION_KELLY_FRACTION"
        ),
        "total_tp_sl_range_jpy": total_range_jpy,
        "quarter_kelly_root_net_downside_cap_jpy": kelly_risk_cap_jpy,
        "effective_worst_fill_raw_risk_cap_jpy": risk_cap_jpy,
        "ev_zero_entry": ev_zero_entry,
        "loss_cap_entry": cap_entry,
        "raw_adverse_limit": raw_adverse_limit,
        "tick": tick,
        "rounding": (
            "LONG_FLOOR_ONE_TICK_AT_EXACT_ROOT"
            if intent.side == Side.LONG
            else "SHORT_CEIL_ONE_TICK_AT_EXACT_ROOT"
        ),
        "derived_safe_price_bound": derived_price_bound,
        "derived_safe_price_bound_text": price_bound_text,
        "next_adverse_tick_entry": next_adverse_entry,
        "next_adverse_tick_risk_jpy": next_risk_jpy,
        "next_adverse_tick_reward_jpy": next_reward_jpy,
        "next_adverse_tick_net_risk_jpy": next_net_risk_jpy,
        "next_adverse_tick_net_reward_jpy": next_net_reward_jpy,
        "next_adverse_tick_ev_lower_jpy": next_ev_jpy,
        "next_adverse_tick_quarter_kelly_budget_jpy": (
            next_kelly_budget_jpy
        ),
        "next_adverse_tick_passes": next_adverse_tick_passes,
        "next_adverse_tick_fails": next_adverse_tick_fails,
        "reserved_price_bound": supplied_bound,
        "reserved_price_bound_text": (
            str(reserved_price_bound).strip()
            if reserved_price_bound is not None
            else None
        ),
        "price_bound": worst_fill_entry,
        "price_bound_text": _price(intent.pair, worst_fill_entry),
        "bound_not_looser_than_fresh_ceiling": bound_not_looser,
        "current_executable_price_inside_bound": current_inside_bound,
        "entry_slippage_allowance_pips": entry_slippage_allowance_pips,
        "entry_slippage_allowance_meets_p95": (
            entry_slippage_allowance_passed
        ),
        "worst_fill_geometry_passed": geometry_passed,
        "worst_fill_risk_jpy": worst_risk_jpy,
        "worst_fill_reward_jpy": worst_reward_jpy,
        "worst_fill_net_risk_jpy": worst_net_risk_jpy,
        "worst_fill_net_reward_jpy": worst_net_reward_jpy,
        "worst_fill_reward_risk": worst_reward_risk,
        "worst_fill_ev_lower_jpy": worst_ev_lower_jpy,
        "worst_fill_ev_strictly_positive": ev_passed,
        "worst_fill_full_kelly_fraction": full_kelly_fraction,
        "worst_fill_quarter_kelly_budget_jpy": (
            worst_quarter_kelly_budget_jpy
        ),
        "worst_fill_quarter_kelly_passed": kelly_passed,
        "worst_fill_loss_caps_passed": cap_passed,
    }
    evidence = finished(material)
    if passed:
        return evidence, None
    return evidence, RiskIssue(
        "PRE_POST_GPT_ALLOCATION_PRICE_BOUND_REPROOF_FAILED",
        "reserved ordinary MARKET priceBound does not preserve strict Wilson "
        f"EV, quarter-Kelly risk, current-NAV/effective loss caps, and TP/SL "
        f"geometry at the worst fill ({reason})",
    )


def _snapshot_at_market_price_bound(
    snapshot: BrokerSnapshot,
    *,
    intent: OrderIntent,
    price_bound: float,
) -> BrokerSnapshot:
    quote = snapshot.quotes[intent.pair]
    spread = quote.ask - quote.bid
    if intent.side == Side.LONG:
        bound_quote = replace(
            quote,
            ask=price_bound,
            bid=price_bound - spread,
        )
    else:
        bound_quote = replace(
            quote,
            bid=price_bound,
            ask=price_bound + spread,
        )
    return replace(
        snapshot,
        quotes={**snapshot.quotes, intent.pair: bound_quote},
    )


def _numeric_proof_with_market_price_bound(
    numeric_proof: dict[str, Any],
    price_bound_proof: dict[str, Any],
) -> dict[str, Any]:
    material = dict(numeric_proof)
    material.pop("proof_sha256", None)
    material["market_price_bound"] = price_bound_proof
    return {
        **material,
        "proof_sha256": _canonical_json_sha256(material),
    }


def _capital_allocation_numeric_pre_post_recheck(
    *,
    verified_intent: OrderIntent,
    final_intent: OrderIntent,
    fresh_snapshot: BrokerSnapshot,
    fresh_base_risk: Any,
    required: bool,
    forecast_s5_path_proof: dict[str, Any] | None = None,
    execution_cost_floor: dict[str, Any] | None = None,
    capital_allocation_edge_recheck: dict[str, Any] | None = None,
    market_entry_slippage_embedded: bool = False,
    m15_recovery_prebounded: bool = False,
) -> tuple[dict[str, Any], RiskIssue | None]:
    """Recompute GPT's numeric ceiling from fresh broker truth at base units."""

    # Lazy by design: market_read_overlay imports risk policy but never this
    # gateway module. Keeping the private pure-math reuse here avoids widening
    # module initialization into a broker/execution dependency cycle.
    from quant_rabbit.market_read_overlay import (
        _capital_allocation_numeric_ceiling,
    )

    base_units = abs(int(verified_intent.units))
    final_units = abs(int(final_intent.units))
    final_effective_multiple = (
        final_units / base_units if base_units > 0 else math.inf
    )
    if not required:
        material = {
            "status": "NOT_APPLICABLE",
            "required": False,
            "base_units": base_units,
            "final_units": final_units,
            "final_effective_multiple": final_effective_multiple,
            "fresh_snapshot_fetched_at_utc": (
                fresh_snapshot.fetched_at_utc.isoformat()
            ),
        }
        return {
            **material,
            "proof_sha256": _canonical_json_sha256(material),
        }, None

    metadata = dict(verified_intent.metadata or {})
    predictive_scout = _numeric_predictive_scout_contract(verified_intent)
    hedge = str(metadata.get("position_intent") or "").strip().upper() == "HEDGE"
    target_path_live_learning_claim = (
        _target_path_live_learning_allocation_claim(verified_intent)
    )
    range_tp_claim = _range_tp_proven_prebounded_intent_claim(
        verified_intent
    )
    m15_recovery_claim = bool(
        m15_recovery_prebounded
        and _m15_recovery_prebounded_intent_claim(
            verified_intent
        )
    )
    range_economic_basis = _range_tp_proven_basis_from_fresh_edge(
        verified_intent,
        capital_allocation_edge_recheck,
    )
    range_geometry_frozen = bool(
        not range_tp_claim
        or (
            final_intent.pair == verified_intent.pair
            and final_intent.side == verified_intent.side
            and final_intent.order_type == verified_intent.order_type
            and final_intent.entry == verified_intent.entry
            and final_intent.tp == verified_intent.tp
            and final_intent.sl == verified_intent.sl
        )
    )
    m15_recovery_geometry_frozen = bool(
        not m15_recovery_claim
        or (
            final_intent.pair == verified_intent.pair
            and final_intent.side == verified_intent.side
            and final_intent.order_type == verified_intent.order_type
            and final_intent.entry == verified_intent.entry
            and final_intent.tp == verified_intent.tp
            and final_intent.sl == verified_intent.sl
            and 0 < abs(final_intent.units) <= abs(verified_intent.units)
        )
    )
    target_path_live_learning_geometry_frozen = bool(
        not target_path_live_learning_claim
        or (
            final_intent.pair == verified_intent.pair
            and final_intent.side == verified_intent.side
            and final_intent.order_type == verified_intent.order_type
            and final_intent.entry == verified_intent.entry
            and final_intent.tp == verified_intent.tp
            and final_intent.sl == verified_intent.sl
            and 0 < abs(final_intent.units) <= abs(verified_intent.units)
        )
    )
    quote = fresh_snapshot.quotes.get(verified_intent.pair)
    account = fresh_snapshot.account
    quote_currency = (
        verified_intent.pair.split("_", 1)[1]
        if "_" in verified_intent.pair
        else ""
    )
    quote_to_jpy = (
        1.0
        if quote_currency == "JPY"
        else fresh_snapshot.home_conversions.get(quote_currency)
    )
    metrics = getattr(fresh_base_risk, "metrics", None)
    risk_metrics = asdict(metrics) if isinstance(metrics, RiskMetrics) else {}
    intent_payload = {
        "pair": verified_intent.pair,
        "side": verified_intent.side.value,
        "order_type": verified_intent.order_type.value,
        "units": verified_intent.units,
        "entry": verified_intent.entry,
        "tp": verified_intent.tp,
        "sl": verified_intent.sl,
        # The shared numeric helper binds the dynamic execution-cost proof to
        # pair/side/method/MARKET.  Dropping the verified method here silently
        # changed a valid TREND_CONTINUATION proof into UNKNOWN during the
        # pre-POST reproof and blocked every ordinary MARKET order.
        "market_context": (
            {"method": verified_intent.market_context.method.value}
            if verified_intent.market_context is not None
            and isinstance(
                verified_intent.market_context.method,
                TradeMethod,
            )
            else {}
        ),
    }
    numeric_ceiling, raw_fresh_max_multiple = _capital_allocation_numeric_ceiling(
        intent=intent_payload,
        metadata=metadata,
        risk_metrics=risk_metrics,
        account_nav_jpy=account.nav_jpy if account is not None else None,
        broker_bid=quote.bid if quote is not None else None,
        broker_ask=quote.ask if quote is not None else None,
        broker_quote_to_jpy=quote_to_jpy,
        predictive_scout=predictive_scout,
        hedge=hedge,
        range_economic_basis=range_economic_basis,
        forecast_current_binding_mode=(
            "RANGE_FRESH_PASSIVE_RAIL"
            if range_tp_claim
            else "DIRECTIONAL_FRESH_MARKET_DRIFT"
        ),
        execution_cost_floor=execution_cost_floor,
        market_entry_slippage_embedded=(
            market_entry_slippage_embedded
        ),
        m15_recovery_prebounded=(
            m15_recovery_claim and m15_recovery_geometry_frozen
        ),
        target_path_live_learning_prebounded=(
            target_path_live_learning_claim
            and target_path_live_learning_geometry_frozen
        ),
    )
    proof_is_bypass = str(numeric_ceiling.get("reason") or "") in {
        "PREDICTIVE_SCOUT_PREBOUNDED_CONTRACT",
        "HEDGE_RISK_REDUCTION_PREBOUNDED_CONTRACT",
        "TP_PROVEN_RANGE_NONMARKET_PREBOUNDED_CONTRACT",
        M15_RECOVERY_PREBOUNDED_REASON,
        TARGET_PATH_LIVE_LEARNING_PREBOUNDED_REASON,
    }
    path_proof_is_bypass = str(numeric_ceiling.get("reason") or "") in {
        "PREDICTIVE_SCOUT_PREBOUNDED_CONTRACT",
        "HEDGE_RISK_REDUCTION_PREBOUNDED_CONTRACT",
        M15_RECOVERY_PREBOUNDED_REASON,
        TARGET_PATH_LIVE_LEARNING_PREBOUNDED_REASON,
    }
    path_proof = (
        dict(forecast_s5_path_proof)
        if isinstance(forecast_s5_path_proof, dict)
        else {"status": "MISSING", "reason": "FORECAST_S5_PATH_PROOF_MISSING"}
    )
    bypass_reason = (
        str(numeric_ceiling.get("reason") or "")
        if path_proof_is_bypass
        else None
    )
    path_proof_passed = _forecast_s5_path_proof_valid(
        path_proof,
        intent=final_intent,
        snapshot=fresh_snapshot,
        bypass_reason=bypass_reason,
    )
    fresh_max_multiple = (
        float(raw_fresh_max_multiple)
        if proof_is_bypass or path_proof_passed
        else 0.0
    )
    numeric_inputs_passed = bool(
        isinstance(raw_fresh_max_multiple, (int, float))
        and not isinstance(raw_fresh_max_multiple, bool)
        and math.isfinite(float(raw_fresh_max_multiple))
        and float(raw_fresh_max_multiple) > 0.0
        and path_proof_passed
        and range_geometry_frozen
        and m15_recovery_geometry_frozen
        and target_path_live_learning_geometry_frozen
    )
    fresh_unit_cap = (
        math.floor(base_units * float(fresh_max_multiple) + 1e-12)
        if numeric_inputs_passed and base_units > 0
        else 0
    )
    final_ratio_passed = bool(
        numeric_inputs_passed
        and base_units > 0
        and final_units > 0
        and final_units <= base_units
        and final_units <= fresh_unit_cap
    )
    status = (
        "BYPASSED"
        if proof_is_bypass and final_ratio_passed
        else "PASSED"
        if numeric_inputs_passed and final_ratio_passed
        else "BLOCKED"
    )
    material = {
        "status": status,
        "required": True,
        "risk_metrics_source": "FRESH_RISK_ENGINE_ORIGINAL_BASE_UNITS",
        "fresh_snapshot_fetched_at_utc": (
            fresh_snapshot.fetched_at_utc.isoformat()
        ),
        "fresh_quote_timestamp_utc": (
            quote.timestamp_utc.isoformat() if quote is not None else None
        ),
        "base_units": base_units,
        "final_units": final_units,
        "final_effective_multiple": final_effective_multiple,
        "fresh_max_multiple": float(fresh_max_multiple),
        "raw_numeric_max_multiple": float(raw_fresh_max_multiple),
        "fresh_unit_cap_floor": fresh_unit_cap,
        "numeric_inputs_passed": numeric_inputs_passed,
        "final_ratio_passed": final_ratio_passed,
        "fresh_base_risk_metrics": risk_metrics,
        "fresh_base_risk_issue_codes_context": [
            str(getattr(issue, "code", ""))
            for issue in getattr(fresh_base_risk, "issues", ())
            if str(getattr(issue, "code", ""))
        ],
        "numeric_helper_binding": (
            "LAZY_PURE_MATH_IMPORT:quant_rabbit.market_read_overlay."
            "_capital_allocation_numeric_ceiling;NO_EXECUTION_IMPORT_IN_OVERLAY"
        ),
        "range_tp_prebounded_claimed": range_tp_claim,
        "range_tp_geometry_frozen": range_geometry_frozen,
        "range_tp_fresh_edge_basis": range_economic_basis,
        "m15_recovery_prebounded_claimed": m15_recovery_claim,
        "m15_recovery_geometry_frozen": m15_recovery_geometry_frozen,
        "target_path_live_learning_prebounded_claimed": (
            target_path_live_learning_claim
        ),
        "target_path_live_learning_geometry_frozen": (
            target_path_live_learning_geometry_frozen
        ),
        "forecast_s5_path_proof": path_proof,
        "forecast_s5_path_proof_passed": path_proof_passed,
        "numeric_ceiling": numeric_ceiling,
    }
    evidence = {
        **material,
        "proof_sha256": _canonical_json_sha256(material),
    }
    if not numeric_inputs_passed:
        if (
            target_path_live_learning_claim
            and not target_path_live_learning_geometry_frozen
        ):
            return evidence, RiskIssue(
                "PRE_POST_GPT_ALLOCATION_TARGET_PATH_GEOMETRY_MUTATED",
                "target-path live-learning pair/side/vehicle/entry/TP/SL or producer unit ceiling changed after the signed allocation",
            )
        if m15_recovery_claim and not m15_recovery_geometry_frozen:
            return evidence, RiskIssue(
                "PRE_POST_GPT_ALLOCATION_M15_RECOVERY_GEOMETRY_MUTATED",
                "M15 recovery pair/side/vehicle/entry/TP/SL or producer unit ceiling changed after the signed allocation",
            )
        if not range_geometry_frozen:
            return evidence, RiskIssue(
                "PRE_POST_GPT_ALLOCATION_RANGE_GEOMETRY_MUTATED",
                "TP-proven RANGE LIMIT entry/TP/SL/vehicle changed after the signed allocation; expire and rebuild the lane",
            )
        if not path_proof_passed:
            return evidence, RiskIssue(
                "PRE_POST_GPT_ALLOCATION_FORECAST_S5_PATH_REPROOF_FAILED",
                "fresh GPT allocation requires complete OANDA S5 mid-candle "
                "no-touch coverage from forecast emission through the current "
                f"quote ({path_proof.get('reason') or 'unknown'}); discard the "
                "stale/consumed signal",
            )
        return evidence, RiskIssue(
            "PRE_POST_GPT_ALLOCATION_NUMERIC_REPROOF_FAILED",
            "fresh broker NAV/quote/conversion, base-unit RiskMetrics, forecast "
            "geometry, economic Wilson EV, or quarter-Kelly proof no longer "
            f"authorizes risk ({numeric_ceiling.get('reason') or 'unknown'}); "
            "discard the stale GPT allocation",
        )
    if not final_ratio_passed:
        return evidence, RiskIssue(
            "PRE_POST_GPT_ALLOCATION_FRESH_CEILING_EXCEEDED",
            f"final units {final_units} exceed floor(base units {base_units} * "
            f"fresh maximum {float(fresh_max_multiple):.12f})={fresh_unit_cap}; reduce "
            "units or obtain a fresh GPT allocation receipt",
        )
    return evidence, None


def _capital_allocation_edge_pre_post_recheck(
    intent: OrderIntent,
    *,
    ledger_path: Path,
    expected_edge_basis: str | None = None,
    expected_execution_cost_floor_sha256: str | None = None,
    execution_cost_floor_required: bool = False,
) -> tuple[dict[str, Any], RiskIssue | None]:
    """Reprove the selected GPT allocation edge from the just-synced ledger.

    The intent's embedded metrics identify which evidence basis the reviewed
    allocation used. Fresh ledger data may preserve or revoke that same basis;
    an all-exit proof is never allowed to fall back to a profitable TP subset.
    """

    metadata = dict(intent.metadata or {})
    read_at_utc = datetime.now(timezone.utc).isoformat()
    method = (
        intent.market_context.method.value
        if intent.market_context is not None
        and isinstance(intent.market_context.method, TradeMethod)
        else "UNKNOWN"
    )
    vehicle = _tp_proven_vehicle(intent.order_type)
    exact_key = (
        intent.pair.upper(),
        intent.side.value.upper(),
        method.upper(),
        vehicle,
    )
    scope_key = "|".join(exact_key)

    if execution_cost_floor_required and expected_edge_basis is None:
        evidence = {
            "status": "BLOCKED",
            "issue_code": (
                "PRE_POST_GPT_ALLOCATION_SIGNED_EDGE_BASIS_UNREADABLE"
            ),
            "basis": None,
            "expected_basis": None,
            "proof_key": list(exact_key),
            "read_at_utc": read_at_utc,
        }
        return evidence, RiskIssue(
            "PRE_POST_GPT_ALLOCATION_SIGNED_EDGE_BASIS_UNREADABLE",
            "numeric GPT allocation must preserve the edge basis from the exact "
            "receipt bytes frozen at gateway entry; missing, swapped, or unreadable "
            "provenance cannot fall back to an intent-inferred basis",
        )

    if (
        execution_cost_floor_required
        and expected_edge_basis
        in {
            "EXACT_VEHICLE_ALL_EXIT_NET",
            "EXACT_VEHICLE_TAKE_PROFIT",
        }
        and expected_execution_cost_floor_sha256 is None
    ):
        evidence = {
            "status": "BLOCKED",
            "issue_code": (
                "PRE_POST_GPT_ALLOCATION_SIGNED_COST_FLOOR_UNREADABLE"
            ),
            "basis": expected_edge_basis,
            "expected_basis": expected_edge_basis,
            "proof_key": list(exact_key),
            "read_at_utc": read_at_utc,
        }
        return evidence, RiskIssue(
            "PRE_POST_GPT_ALLOCATION_SIGNED_COST_FLOOR_UNREADABLE",
            "numeric GPT allocation must preserve the execution-cost proof SHA "
            "from the exact receipt bytes frozen at gateway entry; missing or "
            "unreadable provenance cannot adopt a later ledger proof",
        )

    if _target_path_live_learning_allocation_claim(intent):
        if expected_edge_basis not in {
            None,
            TARGET_PATH_LIVE_LEARNING_EDGE_COLLECTION_BASIS,
        }:
            return {
                "status": "BLOCKED",
                "basis": TARGET_PATH_LIVE_LEARNING_EDGE_COLLECTION_BASIS,
                "expected_basis": expected_edge_basis,
                "proof_key": list(exact_key),
                "read_at_utc": read_at_utc,
            }, RiskIssue(
                "PRE_POST_GPT_ALLOCATION_EDGE_BASIS_MISMATCH",
                "signed board edge basis does not match bounded target-path live learning",
            )
        return {
            "status": "BYPASSED",
            "basis": TARGET_PATH_LIVE_LEARNING_EDGE_COLLECTION_BASIS,
            "expected_basis": expected_edge_basis,
            "proof_key": list(exact_key),
            "read_at_utc": read_at_utc,
        }, None
    if str(metadata.get("position_intent") or "").strip().upper() == "HEDGE":
        if expected_edge_basis not in {None, "HEDGE_RISK_REDUCTION"}:
            return {
                "status": "BLOCKED",
                "basis": "HEDGE_RISK_REDUCTION",
                "expected_basis": expected_edge_basis,
                "proof_key": list(exact_key),
                "read_at_utc": read_at_utc,
            }, RiskIssue(
                "PRE_POST_GPT_ALLOCATION_EDGE_BASIS_MISMATCH",
                "signed board edge basis does not match HEDGE risk reduction",
            )
        return {
            "status": "BYPASSED",
            "basis": "HEDGE_RISK_REDUCTION",
            "expected_basis": expected_edge_basis,
            "proof_key": list(exact_key),
            "read_at_utc": read_at_utc,
        }, None
    if (
        predictive_scout_intent_claimed(intent)
        and predictive_scout_metadata_supported(metadata)
    ):
        if expected_edge_basis not in {
            None,
            "PREDICTIVE_SCOUT_FORWARD_EVIDENCE",
        }:
            return {
                "status": "BLOCKED",
                "basis": "PREDICTIVE_SCOUT_FORWARD_EVIDENCE",
                "expected_basis": expected_edge_basis,
                "proof_key": list(exact_key),
                "read_at_utc": read_at_utc,
            }, RiskIssue(
                "PRE_POST_GPT_ALLOCATION_EDGE_BASIS_MISMATCH",
                "signed board edge basis does not match predictive scout evidence",
            )
        return {
            "status": "BYPASSED",
            "basis": "PREDICTIVE_SCOUT_FORWARD_EVIDENCE",
            "expected_basis": expected_edge_basis,
            "proof_key": list(exact_key),
            "read_at_utc": read_at_utc,
        }, None

    claimed_net = {
        "trades": metadata.get("capture_exact_vehicle_net_trades"),
        "wins": metadata.get("capture_exact_vehicle_net_wins"),
        "losses": metadata.get("capture_exact_vehicle_net_losses"),
        "net_jpy": metadata.get("capture_exact_vehicle_net_jpy"),
        "expectancy_jpy_per_trade": metadata.get(
            "capture_exact_vehicle_net_expectancy_jpy"
        ),
        "avg_win_jpy": metadata.get("capture_exact_vehicle_net_avg_win_jpy"),
        "avg_loss_jpy": metadata.get("capture_exact_vehicle_net_avg_loss_jpy"),
        "unresolved_realized_trades": metadata.get(
            "capture_exact_vehicle_net_unresolved_realized_trades"
        ),
        "unresolved_realized_net_jpy": metadata.get(
            "capture_exact_vehicle_net_unresolved_realized_net_jpy"
        ),
    }
    claimed_net_evaluation = evaluate_exact_vehicle_net_edge(claimed_net)
    claimed_net_binding = bool(
        method != "UNKNOWN"
        and vehicle in {"LIMIT", "MARKET", "STOP"}
        and str(metadata.get("capture_exact_vehicle_net_scope") or "").upper()
        == "PAIR_SIDE_METHOD_VEHICLE"
        and str(metadata.get("capture_exact_vehicle_net_scope_key") or "").upper()
        == f"{scope_key}|ALL_AUDITED_EXITS"
        and str(metadata.get("capture_exact_vehicle_net_vehicle") or "").upper()
        == vehicle
        and str(metadata.get("capture_exact_vehicle_net_metrics_source") or "")
        == "data/execution_ledger.db:exact_vehicle_net"
        and str(metadata.get("capture_exact_vehicle_net_exit_scope") or "").upper()
        == "ALL_AUDITED_EXITS"
    )
    claimed_tp = {
        "trades": metadata.get("capture_take_profit_trades"),
        "wins": metadata.get("capture_take_profit_wins"),
        "losses": metadata.get("capture_take_profit_losses"),
        "net_jpy": metadata.get("capture_take_profit_net_jpy"),
        "expectancy_jpy_per_trade": metadata.get(
            "capture_take_profit_expectancy_jpy"
        ),
        "avg_win_jpy": metadata.get("capture_take_profit_avg_win_jpy"),
        "avg_loss_jpy": metadata.get("capture_take_profit_avg_loss_jpy"),
    }
    claimed_tp_evaluation = evaluate_exact_vehicle_net_edge(claimed_tp)
    claimed_tp_binding = bool(
        method != "UNKNOWN"
        and vehicle in {"LIMIT", "MARKET", "STOP"}
        and metadata.get("attach_take_profit_on_fill") is True
        and str(metadata.get("tp_execution_mode") or "").upper()
        == "ATTACHED_TECHNICAL_TP"
        and metadata.get("capture_take_profit_exact_vehicle_required") is True
        and str(metadata.get("capture_take_profit_scope") or "").upper()
        == "PAIR_SIDE_METHOD_VEHICLE"
        and str(metadata.get("capture_take_profit_scope_key") or "").upper()
        == f"{scope_key}|TAKE_PROFIT_ORDER"
        and str(metadata.get("capture_take_profit_vehicle") or "").upper()
        == vehicle
        and str(metadata.get("capture_take_profit_metrics_source") or "")
        == "data/execution_ledger.db:exact_vehicle_take_profit"
        and claimed_tp_evaluation["arithmetic_consistent"] is True
        and (claimed_tp_evaluation.get("trades") or 0)
        >= CAPITAL_ALLOCATION_MIN_EXACT_TP_TRADES
        and claimed_tp_evaluation.get("losses") == 0
        and (claimed_tp_evaluation.get("net_jpy") or 0.0) > 0.0
        and (claimed_tp_evaluation.get("expectancy_jpy") or 0.0) > 0.0
        and (claimed_tp_evaluation.get("avg_win_jpy") or 0.0) > 0.0
        and claimed_net_evaluation["blocks_tp_exception"] is not True
    )
    m15_recovery_collection_intent = (
        _m15_recovery_edge_collection_prebounded_intent_claim(intent)
    )
    m15_recovery_collection_claim = bool(
        expected_edge_basis == M15_RECOVERY_EDGE_COLLECTION_BASIS
        and claimed_tp_binding
        and m15_recovery_collection_intent
    )
    original_basis = (
        expected_edge_basis
        if expected_edge_basis == "EXACT_VEHICLE_ALL_EXIT_NET"
        and not m15_recovery_collection_intent
        and claimed_net_binding
        and claimed_net_evaluation["proven"] is True
        else expected_edge_basis
        if expected_edge_basis == "EXACT_VEHICLE_TAKE_PROFIT"
        and claimed_tp_binding
        and not m15_recovery_collection_intent
        else expected_edge_basis
        if m15_recovery_collection_claim
        else "EXACT_VEHICLE_ALL_EXIT_NET"
        if expected_edge_basis is None
        and not m15_recovery_collection_intent
        and claimed_net_binding
        and claimed_net_evaluation["proven"] is True
        else "EXACT_VEHICLE_TAKE_PROFIT"
        if expected_edge_basis is None
        and not m15_recovery_collection_intent
        and claimed_tp_binding
        else None
    )
    if original_basis is None:
        evidence = {
            "status": "BLOCKED",
            "issue_code": "PRE_POST_GPT_ALLOCATION_EDGE_BASIS_UNPROVEN",
            "basis": None,
            "expected_basis": expected_edge_basis,
            "proof_key": list(exact_key),
            "read_at_utc": read_at_utc,
            "claimed_all_exit": claimed_net_evaluation,
            "claimed_tp": claimed_tp_evaluation,
        }
        return evidence, RiskIssue(
            "PRE_POST_GPT_ALLOCATION_EDGE_BASIS_UNPROVEN",
            "the verified GPT allocation does not carry an arithmetically valid exact "
            "all-exit or bounded exact-TP evidence basis for this pair/side/method/vehicle",
        )

    surface = read_exact_vehicle_allocation_surface(ledger_path)
    net_metrics = exact_vehicle_metrics_from_surface(
        surface,
        field="exact_vehicle_net",
    )
    tp_metrics = exact_vehicle_metrics_from_surface(
        surface,
        field="exact_vehicle_take_profit",
    )
    if net_metrics is None or tp_metrics is None:
        evidence = {
            "status": "BLOCKED",
            "issue_code": "PRE_POST_GPT_ALLOCATION_LEDGER_READ_FAILED",
            "basis": original_basis,
            "proof_key": list(exact_key),
            "read_at_utc": read_at_utc,
            "ledger_parse_status": surface.get("parse_status"),
        }
        return evidence, RiskIssue(
            "PRE_POST_GPT_ALLOCATION_LEDGER_READ_FAILED",
            "the post-sync execution ledger cannot produce a complete exact-vehicle "
            "capital-allocation surface immediately before POST",
        )

    current_net = dict(net_metrics.get(exact_key) or {})
    current_tp = dict(tp_metrics.get(exact_key) or {})
    current_net_evaluation = evaluate_exact_vehicle_net_edge(current_net)
    current_tp_evaluation = evaluate_exact_vehicle_net_edge(current_tp)
    current_net_contract = bool(
        str(current_net.get("source_scope") or "").upper()
        == "PAIR_SIDE_METHOD_VEHICLE"
        and str(current_net.get("exit_scope") or "").upper()
        == "ALL_AUDITED_EXITS"
    )
    current_tp_contract = bool(
        str(current_tp.get("source_scope") or "").upper()
        == "PAIR_SIDE_METHOD_VEHICLE"
        and str(current_tp.get("exit_scope") or "").upper()
        == "PURE_TAKE_PROFIT_LIFECYCLE"
    )
    execution_cost_floor = execution_cost_floor_from_surface(
        surface,
        exact_key=exact_key,
        as_of=datetime.now(timezone.utc),
    )
    current_cost_sha = str(
        execution_cost_floor.get("proof_sha256") or ""
    )
    cost_floor_passed = bool(
        original_basis == M15_RECOVERY_EDGE_COLLECTION_BASIS
        or not execution_cost_floor_required
        or (
            execution_cost_floor.get("status") == "PASSED"
            and expected_execution_cost_floor_sha256 is not None
            and current_cost_sha == expected_execution_cost_floor_sha256
        )
    )
    if original_basis == "EXACT_VEHICLE_ALL_EXIT_NET":
        passed = bool(
            current_net_contract
            and current_net_evaluation["proven"] is True
            and cost_floor_passed
        )
        failed_checks = [] if passed else ["ALL_EXIT_EDGE_NO_LONGER_PROVEN"]
    else:
        current_tp_passed = bool(
            current_tp_contract
            and current_tp_evaluation["arithmetic_consistent"] is True
            and (current_tp_evaluation.get("trades") or 0)
            >= CAPITAL_ALLOCATION_MIN_EXACT_TP_TRADES
            and current_tp_evaluation.get("losses") == 0
            and (current_tp_evaluation.get("net_jpy") or 0.0) > 0.0
            and (current_tp_evaluation.get("expectancy_jpy") or 0.0) > 0.0
            and (current_tp_evaluation.get("avg_win_jpy") or 0.0) > 0.0
        )
        all_exit_allows_tp = bool(
            current_net_contract
            and current_net_evaluation["blocks_tp_exception"] is not True
        )
        recovery_collection_window = bool(
            original_basis != M15_RECOVERY_EDGE_COLLECTION_BASIS
            or (
                m15_recovery_collection_claim
                and CAPITAL_ALLOCATION_MIN_EXACT_TP_TRADES
                <= (current_tp_evaluation.get("trades") or 0)
                < 20
            )
        )
        recovery_same_tp_row = bool(
            original_basis != M15_RECOVERY_EDGE_COLLECTION_BASIS
            or all(
                current_tp_evaluation.get(key)
                == claimed_tp_evaluation.get(key)
                for key in (
                    "trades",
                    "wins",
                    "losses",
                    "net_jpy",
                    "expectancy_jpy",
                    "avg_win_jpy",
                    "avg_loss_jpy",
                )
            )
        )
        passed = bool(
            current_tp_passed
            and all_exit_allows_tp
            and recovery_collection_window
            and recovery_same_tp_row
            and cost_floor_passed
        )
        failed_checks = []
        if not current_tp_passed:
            failed_checks.append("EXACT_TP_EDGE_NO_LONGER_PROVEN")
        if not all_exit_allows_tp:
            failed_checks.append("ALL_EXIT_EVIDENCE_SUPPRESSES_TP_EXCEPTION")
        if not recovery_collection_window:
            failed_checks.append("M15_RECOVERY_COLLECTION_WINDOW_INVALID")
        if not recovery_same_tp_row:
            failed_checks.append("M15_RECOVERY_EXACT_TP_ROW_CHANGED")
    if not cost_floor_passed:
        failed_checks.append("EXECUTION_COST_FLOOR_STALE_OR_MISMATCHED")
    fresh_capture = _fresh_capture_payoff_metrics(ledger_path)

    evidence = {
        "status": "PASSED" if passed else "BLOCKED",
        "issue_code": None if passed else "PRE_POST_GPT_ALLOCATION_EDGE_STALE",
        "basis": original_basis,
        "expected_basis": expected_edge_basis,
        "proof_key": list(exact_key),
        "read_at_utc": read_at_utc,
        "ledger_surface_sha256": surface.get("allocation_surface_sha256"),
        "current_all_exit": current_net_evaluation,
        "current_tp": current_tp_evaluation,
        "fresh_capture": fresh_capture,
        "execution_cost_floor": execution_cost_floor,
        "expected_execution_cost_floor_sha256": (
            expected_execution_cost_floor_sha256
        ),
        "failed_checks": failed_checks,
    }
    if passed:
        return evidence, None
    return evidence, RiskIssue(
        "PRE_POST_GPT_ALLOCATION_EDGE_STALE",
        f"the post-sync {original_basis} evidence for {scope_key} no longer authorizes "
        "the reviewed capital allocation; refresh GPT market-read and sizing before POST",
    )


def _tp_proven_pre_post_recheck(
    intent: OrderIntent,
    *,
    ledger_path: Path,
) -> tuple[OrderIntent, float | None, dict[str, Any], RiskIssue | None]:
    """Reprove an embedded TP relaxation from the just-synced live ledger.

    Intent generation can legitimately mark a lane ``TP_PROVEN_RELAXED`` and
    then wait while the trader verifies/stages it. A newly synced close in that
    interval can invalidate the zero-loss exact-vehicle proof. The gateway must
    therefore treat the embedded metrics as a claim, not as final POST
    authority. A readable mismatch falls back to the fresh average-winner cap;
    an unreadable ledger cannot safely determine either proof or fallback cap.
    """

    metadata = dict(intent.metadata or {})
    if str(metadata.get("loss_asymmetry_guard_mode") or "").upper() != "TP_PROVEN_RELAXED":
        return intent, None, {"status": "NOT_APPLICABLE"}, None
    # A verified recovery lane content-addresses its producer TP proof into the
    # lane binding.  The fresh ledger read below is a second, pre-POST proof;
    # it must not rewrite those signed source fields and thereby invalidate the
    # very lane it just reproved.
    m15_recovery_bound = _m15_recovery_prebounded_intent_claim(intent)

    read_at_utc = datetime.now(timezone.utc).isoformat()
    exact_metrics = _exact_vehicle_take_profit_metrics(ledger_path)
    capture_metrics = _fresh_capture_payoff_metrics(ledger_path)
    if exact_metrics is None or capture_metrics is None:
        evidence = {
            "status": "BLOCKED",
            "issue_code": "PRE_POST_TP_PROOF_LEDGER_READ_FAILED",
            "ledger_path": str(ledger_path),
            "read_at_utc": read_at_utc,
            "freshness": "POST_SYNC_LEDGER_READ_REQUIRED",
        }
        return (
            intent,
            None,
            evidence,
            RiskIssue(
                "PRE_POST_TP_PROOF_LEDGER_READ_FAILED",
                "TP_PROVEN_RELAXED requires a readable post-sync execution ledger "
                "for exact TP proof and the average-winner fallback cap immediately "
                "before POST",
            ),
        )

    vehicle = _tp_proven_vehicle(intent.order_type)
    method = (
        intent.market_context.method.value
        if intent.market_context is not None
        and isinstance(intent.market_context.method, TradeMethod)
        else "UNKNOWN"
    )
    proof_key = (
        intent.pair.upper(),
        intent.side.value.upper(),
        method.upper(),
        vehicle,
    )
    proof = dict(exact_metrics.get(proof_key) or {})
    tp_trades = _nonnegative_int(proof.get("trades"))
    tp_losses = _nonnegative_int(proof.get("losses"))
    tp_expectancy = _finite_float(proof.get("expectancy_jpy_per_trade"))
    tp_avg_win = _finite_float(proof.get("avg_win_jpy"))

    failed_checks: list[str] = []
    if intent.order_type == OrderType.MARKET:
        failed_checks.append("MARKET_VEHICLE")
    if metadata.get("attach_take_profit_on_fill") is not True or not _attach_take_profit_on_fill(intent):
        failed_checks.append("ATTACHED_TP_NOT_EXECUTABLE")
    if str(metadata.get("tp_execution_mode") or "").upper() != "ATTACHED_TECHNICAL_TP":
        failed_checks.append("TP_EXECUTION_MODE_MISMATCH")
    if str(metadata.get("tp_target_intent") or "").upper() != "HARVEST":
        failed_checks.append("TP_TARGET_NOT_HARVEST")
    if method == "UNKNOWN":
        failed_checks.append("METHOD_MISSING")
    if vehicle not in {"LIMIT", "STOP"}:
        failed_checks.append("VEHICLE_UNSUPPORTED")
    if tp_trades < LOSS_ASYMMETRY_TP_RELAX_MIN_EXIT_TRADES:
        failed_checks.append("TP_SAMPLE_BELOW_FLOOR")
    if tp_losses != 0:
        failed_checks.append("TP_LOSS_PRESENT")
    if tp_expectancy is None or tp_expectancy <= 0.0:
        failed_checks.append("TP_EXPECTANCY_NOT_POSITIVE")
    if tp_avg_win is None or tp_avg_win <= 0.0:
        failed_checks.append("TP_AVG_WIN_NOT_POSITIVE")

    scope_key = (
        f"{proof_key[0]}|{proof_key[1]}|{proof_key[2]}|{proof_key[3]}|"
        "TAKE_PROFIT_ORDER"
    )
    fresh_metadata = {
        "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
        "capture_take_profit_scope_key": scope_key,
        "capture_take_profit_exact_vehicle_required": True,
        "capture_take_profit_vehicle": vehicle,
        "capture_take_profit_metrics_source": (
            "data/execution_ledger.db:pre_post_exact_vehicle_take_profit"
        ),
        "capture_take_profit_trades": tp_trades,
        "capture_take_profit_wins": _nonnegative_int(proof.get("wins")),
        "capture_take_profit_losses": tp_losses,
        "capture_take_profit_expectancy_jpy": tp_expectancy,
        "capture_take_profit_avg_win_jpy": tp_avg_win,
        "capture_take_profit_avg_loss_jpy": _finite_float(
            proof.get("avg_loss_jpy")
        ),
    }
    if not m15_recovery_bound:
        metadata.update(fresh_metadata)
    metadata.update(
        {
            "capture_economics_pre_post_read_at_utc": read_at_utc,
            "capture_economics_latest_realized_ts_utc": capture_metrics.get(
                "latest_realized_ts_utc"
            ),
        }
    )
    evidence: dict[str, Any] = {
        "status": "PASSED" if not failed_checks else "CLIPPED_TO_AVG_WIN",
        "ledger_path": str(ledger_path),
        "read_at_utc": read_at_utc,
        "freshness": "SYNCED_LEDGER_MATCHED_BROKER_TRANSACTION_ID",
        "proof_key": list(proof_key),
        "scope_key": scope_key,
        "required_tp_trades": LOSS_ASYMMETRY_TP_RELAX_MIN_EXIT_TRADES,
        "tp_trades": tp_trades,
        "tp_losses": tp_losses,
        "tp_expectancy_jpy": tp_expectancy,
        "failed_checks": failed_checks,
        "fresh_capture": capture_metrics,
        "producer_bound_metadata_preserved": m15_recovery_bound,
        "fresh_tp_metadata": fresh_metadata if m15_recovery_bound else None,
    }
    if not failed_checks:
        metadata.update(
            {
                "loss_asymmetry_guard_relaxed": True,
                "loss_asymmetry_guard_relaxation_reason": (
                    "fresh pre-POST exact pair/side/method/vehicle attached-TP "
                    "proof still has the required sample, zero TP losses, and "
                    "positive expectancy"
                ),
                "pre_post_tp_proven_recheck_status": "PASSED",
            }
        )
        return replace(intent, metadata=metadata), None, evidence, None

    fresh_avg_win = _positive_float(capture_metrics.get("avg_win_jpy"))
    embedded_cap = _positive_float(metadata.get("loss_asymmetry_guard_loss_cap_jpy"))
    if fresh_avg_win is None:
        evidence.update(
            {
                "status": "BLOCKED",
                "issue_code": "PRE_POST_TP_PROOF_AVG_WIN_CAP_MISSING",
            }
        )
        return (
            intent,
            None,
            evidence,
            RiskIssue(
                "PRE_POST_TP_PROOF_AVG_WIN_CAP_MISSING",
                "fresh exact TP proof no longer passes and the synced ledger has "
                "no positive average winner with which to cap the order safely",
            ),
        )

    # The embedded cap may only tighten the freshly recomputed average winner;
    # it cannot substitute for missing current payoff evidence or widen it.
    fallback_cap = min(
        cap for cap in (fresh_avg_win, embedded_cap) if cap is not None
    )
    metadata.update(
        {
            "loss_asymmetry_guard_mode": "CAP_AVG_WIN",
            "loss_asymmetry_guard_relaxed": False,
            "loss_asymmetry_guard_relaxation_reason": (
                "fresh pre-POST exact TP proof failed: " + ",".join(failed_checks)
            ),
            "loss_asymmetry_guard_loss_cap_jpy": round(fallback_cap, 4),
            "loss_asymmetry_guard_effective_max_loss_jpy": round(fallback_cap, 4),
            "capture_economics_trades": capture_metrics.get("trades"),
            "capture_avg_win_jpy": fresh_avg_win,
            "capture_avg_loss_jpy": capture_metrics.get("avg_loss_jpy"),
            "capture_economics_stale": False,
            "pre_post_tp_proven_recheck_status": "CLIPPED_TO_AVG_WIN",
        }
    )
    evidence["fallback_avg_win_cap_jpy"] = fallback_cap
    evidence["embedded_avg_win_cap_jpy"] = embedded_cap
    return replace(intent, metadata=metadata), fallback_cap, evidence, None


def _fresh_capture_payoff_metrics(ledger_path: Path) -> dict[str, Any] | None:
    """Read the current gateway-attributed payoff basis without writing artifacts."""

    rows = read_attributed_net_outcomes(ledger_path)
    if rows is None:
        return None

    realized: list[float] = []
    timestamps: list[str] = []
    try:
        for row in rows:
            timestamps.append(str(row.ts_utc or ""))
            realized.append(float(row.realized_pl_jpy))
    except (AttributeError, TypeError, ValueError):
        return None
    wins = [value for value in realized if value > 0.0]
    losses = [value for value in realized if value < 0.0]
    trades = len(wins) + len(losses)
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = abs(sum(losses) / len(losses)) if losses else None
    expectancy = (sum(wins) + sum(losses)) / trades if trades else None
    return {
        "trades": trades,
        "wins": len(wins),
        "losses": len(losses),
        "avg_win_jpy": round(avg_win, 4) if avg_win is not None else None,
        "avg_loss_jpy": round(avg_loss, 4) if avg_loss is not None else None,
        "expectancy_jpy_per_trade": (
            round(expectancy, 4) if expectancy is not None else None
        ),
        "latest_realized_ts_utc": max(timestamps) if timestamps else None,
    }


def _tp_proven_vehicle(order_type: OrderType) -> str:
    if order_type == OrderType.LIMIT:
        return "LIMIT"
    if order_type == OrderType.STOP_ENTRY:
        return "STOP"
    if order_type == OrderType.MARKET:
        return "MARKET"
    return "UNKNOWN"


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _nonnegative_int(value: object) -> int:
    parsed = _finite_float(value)
    return max(0, int(parsed)) if parsed is not None else 0


def _loss_asymmetry_cap_from_metadata(metadata: dict[str, Any]) -> float | None:
    # HEDGE orders manage existing exposure instead of adding ordinary fresh
    # one-way risk.  RiskEngine and the independent gateway evidence check both
    # exempt this shape, so the attached-stop sizing boundary must not silently
    # reintroduce the average-winner cap and under-size the protection order.
    if str(metadata.get("position_intent") or "NEW").strip().upper() == "HEDGE":
        return None
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
    guard_required = active or status == "NEGATIVE_EXPECTANCY"
    if not guard_required or avg_win is None:
        return None
    explicit_cap = _positive_float(metadata.get("loss_asymmetry_guard_loss_cap_jpy"))
    candidates = [value for value in (explicit_cap, avg_win) if value is not None]
    # A legacy/forged firepower receipt may claim the normal cap in its
    # explicit/effective fields. The executable attached-stop boundary must use
    # the tighter observed average winner just like RiskEngine.
    return min(candidates) if candidates else None


def _loss_asymmetry_avg_win_issue(
    intent: OrderIntent,
    metrics: RiskMetrics | None,
) -> RiskIssue | None:
    """Require the realized local average winner at the attached-stop boundary.

    This gateway check is independent of ``RiskEngine`` so a hand-built or
    stale receipt cannot use a declared/OANDA cap after deleting the current
    capture-economics basis. Exact TP-proven and one-unit local TP collection
    receipts keep their existing evidence-bound exceptions.
    """

    metadata = intent.metadata or {}
    if str(metadata.get("position_intent") or "NEW").upper() == "HEDGE":
        return None
    status = str(metadata.get("capture_economics_status") or "").upper()
    active = str(metadata.get("loss_asymmetry_guard_active") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not (active or status == "NEGATIVE_EXPECTANCY"):
        return None
    mode = str(metadata.get("loss_asymmetry_guard_mode") or "").upper()
    if mode == "TP_PROVEN_RELAXED" and _loss_asymmetry_tp_relaxation_shape_allowed(
        intent,
        metadata,
    ):
        return None
    if (
        mode == "TP_PROOF_COLLECTION_MIN_LOT"
        and metrics is not None
        and _loss_asymmetry_tp_proof_collection_shape_allowed(
            intent,
            metadata,
            metrics,
        )
    ):
        return None
    if _positive_float(metadata.get("capture_avg_win_jpy")) is not None:
        return None
    return RiskIssue(
        "LOSS_ASYMMETRY_GUARD_CAP_MISSING",
        "the attached-stop gateway requires finite positive "
        "capture_avg_win_jpy from realized local payoff evidence for ordinary "
        "fresh risk under an active loss-asymmetry guard or "
        "NEGATIVE_EXPECTANCY; a declared, effective, or legacy OANDA "
        "firepower cap cannot substitute.",
    )


def _loss_asymmetry_nonfinite_issue(metadata: dict[str, Any]) -> RiskIssue | None:
    mode = str(metadata.get("loss_asymmetry_guard_mode") or "").upper()
    status = str(metadata.get("capture_economics_status") or "").upper()
    active = str(metadata.get("loss_asymmetry_guard_active") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not (active or status == "NEGATIVE_EXPECTANCY" or mode):
        return None
    invalid_fields: list[str] = []
    for key in (
        "capture_avg_win_jpy",
        "capture_avg_loss_jpy",
        "loss_asymmetry_guard_loss_cap_jpy",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            invalid_fields.append(key)
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            invalid_fields.append(key)
            continue
        if not math.isfinite(parsed):
            invalid_fields.append(key)
    if not invalid_fields:
        return None
    return RiskIssue(
        "LOSS_ASYMMETRY_GUARD_NONFINITE",
        "loss-asymmetry metadata contains a non-finite or malformed numeric "
        f"field ({', '.join(invalid_fields)}); refresh capture-economics before "
        "building or sending a broker order.",
    )


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _nonnegative_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


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
        if (
            predictive_scout_intent_claimed(intent)
            or _range_tp_proven_prebounded_intent_claim(intent)
        ):
            range_contract = _range_tp_proven_prebounded_intent_claim(intent)
            return intent, RiskIssue(
                (
                    "TP_PROVEN_RANGE_TRIGGER_CROSSED"
                    if range_contract
                    else "PREDICTIVE_SCOUT_TRIGGER_CROSSED"
                ),
                (
                    f"{'TP-proven RANGE' if range_contract else 'predictive SCOUT'} "
                    f"LONG trigger {original_entry:.{precision}f} is no longer passive "
                    f"at ask={quote.ask:.{precision}f}; expire the exact geometry instead of repricing it"
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
        if (
            predictive_scout_intent_claimed(intent)
            or _range_tp_proven_prebounded_intent_claim(intent)
        ):
            range_contract = _range_tp_proven_prebounded_intent_claim(intent)
            return intent, RiskIssue(
                (
                    "TP_PROVEN_RANGE_TRIGGER_CROSSED"
                    if range_contract
                    else "PREDICTIVE_SCOUT_TRIGGER_CROSSED"
                ),
                (
                    f"{'TP-proven RANGE' if range_contract else 'predictive SCOUT'} "
                    f"SHORT trigger {original_entry:.{precision}f} is no longer passive "
                    f"at bid={quote.bid:.{precision}f}; expire the exact geometry instead of repricing it"
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


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _canonical_order_request_sha256(order_request: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            order_request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _capital_allocation_numeric_proof_is_frozen(
    evidence: Any,
    *,
    order_request: dict[str, Any],
    required: bool,
    expected_intent: OrderIntent | None = None,
) -> bool:
    if not isinstance(evidence, dict):
        return False
    proof_sha256 = str(evidence.get("proof_sha256") or "")
    material = dict(evidence)
    material.pop("proof_sha256", None)
    if proof_sha256 != _canonical_json_sha256(material):
        return False
    status = str(evidence.get("status") or "").strip().upper()
    if required:
        if status not in {"PASSED", "BYPASSED"}:
            return False
    elif status not in {"NOT_APPLICABLE", "PASSED", "BYPASSED"}:
        return False
    base_units = evidence.get("base_units")
    final_units = evidence.get("final_units")
    if (
        isinstance(base_units, bool)
        or not isinstance(base_units, int)
        or base_units <= 0
        or isinstance(final_units, bool)
        or not isinstance(final_units, int)
        or final_units <= 0
    ):
        return False
    raw_order_units = order_request.get("units")
    order_units_text = str(raw_order_units or "").strip()
    if not order_units_text.lstrip("-").isdigit():
        return False
    if abs(int(order_units_text)) != final_units:
        return False
    expected_ratio = final_units / base_units
    stored_ratio = evidence.get("final_effective_multiple")
    if (
        isinstance(stored_ratio, bool)
        or not isinstance(stored_ratio, (int, float))
        or not math.isfinite(float(stored_ratio))
        or not math.isclose(
            float(stored_ratio),
            expected_ratio,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        return False
    if not required:
        return True
    path_proof = evidence.get("forecast_s5_path_proof")
    if not isinstance(path_proof, dict):
        return False
    path_sha256 = str(path_proof.get("proof_sha256") or "")
    path_material = dict(path_proof)
    path_material.pop("proof_sha256", None)
    if path_sha256 != _canonical_json_sha256(path_material):
        return False
    price_bound = evidence.get("market_price_bound")
    if not isinstance(price_bound, dict):
        return False
    price_bound_sha256 = str(price_bound.get("proof_sha256") or "")
    price_bound_material = dict(price_bound)
    price_bound_material.pop("proof_sha256", None)
    if price_bound_sha256 != _canonical_json_sha256(price_bound_material):
        return False
    price_bound_status = str(price_bound.get("status") or "").upper()
    numeric_ceiling = (
        evidence.get("numeric_ceiling")
        if isinstance(evidence.get("numeric_ceiling"), dict)
        else {}
    )
    if status == "BYPASSED":
        bypass_reason = str(numeric_ceiling.get("reason") or "")
        if (
            price_bound_status != "BYPASSED"
            or str(price_bound.get("reason") or "") != bypass_reason
            or "priceBound" in order_request
        ):
            return False
        if bypass_reason == M15_RECOVERY_PREBOUNDED_REASON:
            numeric_inputs = (
                numeric_ceiling.get("inputs")
                if isinstance(numeric_ceiling.get("inputs"), dict)
                else {}
            )
            expected_metadata = (
                dict(expected_intent.metadata or {})
                if expected_intent is not None
                else {}
            )
            expected_lane_binding = (
                expected_metadata.get("m15_recovery_lane_binding")
                if isinstance(
                    expected_metadata.get("m15_recovery_lane_binding"),
                    dict,
                )
                else {}
            )
            expected_order_type = (
                "LIMIT"
                if expected_intent is not None
                and expected_intent.order_type == OrderType.LIMIT
                else "STOP"
                if expected_intent is not None
                and expected_intent.order_type == OrderType.STOP_ENTRY
                else None
            )
            expected_price = (
                _price(expected_intent.pair, expected_intent.entry)
                if expected_intent is not None
                and expected_intent.entry is not None
                else None
            )
            if not (
                expected_intent is not None
                and _m15_recovery_prebounded_intent_claim(
                    expected_intent
                )
                and expected_order_type is not None
                and order_request.get("type") == expected_order_type
                and order_request.get("price") == expected_price
                and 0 < final_units <= base_units <= M15_RECOVERY_MICRO_MAX_UNITS
                and abs(expected_intent.units) == final_units
                and expected_metadata.get("m15_recovery_micro_units")
                == base_units
                and expected_lane_binding.get("producer_units") == base_units
                and evidence.get("m15_recovery_prebounded_claimed") is True
                and evidence.get("m15_recovery_geometry_frozen") is True
                and evidence.get("numeric_inputs_passed") is True
                and evidence.get("final_ratio_passed") is True
                and evidence.get("fresh_unit_cap_floor") == base_units
                and evidence.get("fresh_max_multiple") == 1.0
                and evidence.get("raw_numeric_max_multiple") == 1.0
                and numeric_ceiling.get("reason")
                == M15_RECOVERY_PREBOUNDED_REASON
                and numeric_ceiling.get("max_multiple") == 1.0
                and numeric_inputs.get("m15_recovery_prebounded") is True
                and str(path_proof.get("status") or "").upper()
                == "BYPASSED"
                and path_proof.get("reason")
                == M15_RECOVERY_PREBOUNDED_REASON
            ):
                return False
        elif bypass_reason == "TP_PROVEN_RANGE_NONMARKET_PREBOUNDED_CONTRACT":
            inputs = (
                numeric_ceiling.get("inputs")
                if isinstance(numeric_ceiling.get("inputs"), dict)
                else {}
            )
            geometry = (
                numeric_ceiling.get("geometry")
                if isinstance(numeric_ceiling.get("geometry"), dict)
                else {}
            )
            numeric_cost = (
                numeric_ceiling.get("execution_cost_floor")
                if isinstance(
                    numeric_ceiling.get("execution_cost_floor"), dict
                )
                else {}
            )
            numeric_cost_sha = str(numeric_cost.get("proof_sha256") or "")
            numeric_cost_material = dict(numeric_cost)
            numeric_cost_material.pop("proof_sha256", None)
            expected_method = (
                expected_intent.market_context.method.value
                if expected_intent is not None
                and expected_intent.market_context is not None
                and isinstance(
                    expected_intent.market_context.method,
                    TradeMethod,
                )
                else "UNKNOWN"
            )
            expected_scope = (
                "|".join(
                    (
                        expected_intent.pair.upper(),
                        expected_intent.side.value.upper(),
                        expected_method.upper(),
                        "LIMIT",
                    )
                )
                if expected_intent is not None
                else ""
            )
            path_range_valid = bool(
                str(path_proof.get("status") or "").upper() == "PASSED"
                and path_proof.get("reason")
                == "FORECAST_S5_NO_TOUCH_PROVEN"
                and path_proof.get("barrier_basis") == "RANGE_RAILS"
                and path_proof.get("no_touch") is True
                and path_proof.get("order_entry_touched") is False
                and path_proof.get("order_sl_touched") is False
            )
            range_basis = inputs.get("range_trade_economic_basis")
            fresh_basis = evidence.get("range_tp_fresh_edge_basis")
            if not (
                expected_intent is not None
                and _range_tp_proven_prebounded_intent_claim(expected_intent)
                and order_request.get("type") == "LIMIT"
                and order_request.get("price")
                == _price(expected_intent.pair, expected_intent.entry)
                and evidence.get("range_tp_prebounded_claimed") is True
                and evidence.get("range_tp_geometry_frozen") is True
                and isinstance(range_basis, dict)
                and isinstance(fresh_basis, dict)
                and range_basis == fresh_basis
                and inputs.get("range_trade_economic_basis_valid") is True
                and geometry.get("range_rails_bound") is True
                and geometry.get(
                    "forecast_outcome_conservatively_contains_order"
                )
                is True
                and geometry.get("range_risk_cap_passed") is True
                and geometry.get("passed") is True
                and numeric_ceiling.get("execution_cost_floor_valid") is True
                and numeric_ceiling.get(
                    "execution_cost_floor_digest_valid"
                )
                is True
                and numeric_cost.get("status") == "PASSED"
                and numeric_cost.get("contract")
                == EXECUTION_COST_FLOOR_CONTRACT
                and str(numeric_cost.get("scope_key") or "").upper()
                == expected_scope
                and numeric_cost_sha
                == _canonical_json_sha256(numeric_cost_material)
                and path_range_valid
            ):
                return False
        elif bypass_reason not in {
            "PREDICTIVE_SCOUT_PREBOUNDED_CONTRACT",
            "HEDGE_RISK_REDUCTION_PREBOUNDED_CONTRACT",
        }:
            return False
        elif not (
            str(path_proof.get("status") or "").upper() == "BYPASSED"
            and str(path_proof.get("reason") or "") == bypass_reason
        ):
            return False
    else:
        bound_text = str(price_bound.get("price_bound_text") or "")
        numeric_cost = (
            numeric_ceiling.get("execution_cost_floor")
            if isinstance(
                numeric_ceiling.get("execution_cost_floor"), dict
            )
            else {}
        )
        bound_cost = (
            price_bound.get("execution_cost_floor")
            if isinstance(price_bound.get("execution_cost_floor"), dict)
            else {}
        )
        numeric_cost_sha = str(numeric_cost.get("proof_sha256") or "")
        bound_cost_sha = str(bound_cost.get("proof_sha256") or "")
        numeric_cost_material = dict(numeric_cost)
        numeric_cost_material.pop("proof_sha256", None)
        bound_cost_material = dict(bound_cost)
        bound_cost_material.pop("proof_sha256", None)
        bound_pair = str(price_bound.get("pair") or "").strip().upper()
        bound_side = str(price_bound.get("side") or "").strip().upper()
        bound_method = str(price_bound.get("method") or "").strip().upper()
        proof_cost_scope = "|".join(
            (bound_pair, bound_side, bound_method, "MARKET")
        )
        authoritative_method = (
            expected_intent.market_context.method.value
            if expected_intent is not None
            and expected_intent.market_context is not None
            and isinstance(
                expected_intent.market_context.method,
                TradeMethod,
            )
            else "UNKNOWN"
        )
        expected_cost_scope = (
            "|".join(
                (
                    expected_intent.pair.upper(),
                    expected_intent.side.value.upper(),
                    authoritative_method.upper(),
                    "MARKET",
                )
            )
            if expected_intent is not None
            else ""
        )
        cost_proofs_frozen = bool(
            numeric_cost.get("status") == "PASSED"
            and bound_cost.get("status") == "PASSED"
            and numeric_cost.get("contract")
            == EXECUTION_COST_FLOOR_CONTRACT
            and bound_cost.get("contract")
            == EXECUTION_COST_FLOOR_CONTRACT
            and numeric_cost.get("spread_double_count_forbidden") is True
            and bound_cost.get("spread_double_count_forbidden") is True
            and bound_pair
            and bound_side in {"LONG", "SHORT"}
            and bound_method
            and bound_method != "UNKNOWN"
            and price_bound.get("expected_execution_cost_scope_key")
            == proof_cost_scope
            and proof_cost_scope == expected_cost_scope
            and str(numeric_cost.get("scope_key") or "").upper()
            == expected_cost_scope
            and str(bound_cost.get("scope_key") or "").upper()
            == expected_cost_scope
            and numeric_cost_sha
            and numeric_cost_sha == bound_cost_sha
            and numeric_cost_sha
            == _canonical_json_sha256(numeric_cost_material)
            and bound_cost_sha
            == _canonical_json_sha256(bound_cost_material)
        )
        numeric_entry = (
            numeric_ceiling
            .get("inputs", {})
            .get("broker_executable_entry")
            if isinstance(numeric_ceiling, dict)
            else None
        )
        def finite_number(value: Any) -> float | None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None

        outcome_cost = finite_number(price_bound.get("outcome_cost_jpy"))
        loss_outcome_cost = finite_number(
            price_bound.get("loss_outcome_cost_jpy")
        )
        win_outcome_cost = finite_number(
            price_bound.get("win_outcome_cost_jpy")
        )
        stop_loss_execution_cost = finite_number(
            price_bound.get("exit_execution_stress_jpy")
        )
        take_profit_execution_cost = finite_number(
            price_bound.get("take_profit_exit_execution_stress_jpy")
        )
        financing_cost = finite_number(
            price_bound.get("financing_stress_jpy")
        )
        entry_p95 = finite_number(
            price_bound.get("market_entry_adverse_p95_pips")
        )
        cost_entry_p95 = finite_number(
            bound_cost.get("market_entry_adverse_p95_pips")
        )
        entry_allowance = finite_number(
            price_bound.get("entry_slippage_allowance_pips")
        )
        worst_gross_risk = finite_number(
            price_bound.get("worst_fill_risk_jpy")
        )
        worst_gross_reward = finite_number(
            price_bound.get("worst_fill_reward_jpy")
        )
        worst_net_risk = finite_number(
            price_bound.get("worst_fill_net_risk_jpy")
        )
        worst_net_reward = finite_number(
            price_bound.get("worst_fill_net_reward_jpy")
        )
        next_gross_risk = finite_number(
            price_bound.get("next_adverse_tick_risk_jpy")
        )
        next_gross_reward = finite_number(
            price_bound.get("next_adverse_tick_reward_jpy")
        )
        next_net_risk = finite_number(
            price_bound.get("next_adverse_tick_net_risk_jpy")
        )
        next_net_reward = finite_number(
            price_bound.get("next_adverse_tick_net_reward_jpy")
        )
        portfolio_remaining = finite_number(
            price_bound.get("portfolio_loss_remaining_jpy")
        )
        cost_identities_valid = bool(
            outcome_cost is not None
            and outcome_cost > 0.0
            and loss_outcome_cost is not None
            and loss_outcome_cost > 0.0
            and win_outcome_cost is not None
            and win_outcome_cost >= 0.0
            and stop_loss_execution_cost is not None
            and stop_loss_execution_cost >= 0.0
            and take_profit_execution_cost is not None
            and take_profit_execution_cost >= 0.0
            and financing_cost is not None
            and financing_cost >= 0.0
            and math.isclose(
                outcome_cost,
                loss_outcome_cost,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            and math.isclose(
                loss_outcome_cost,
                stop_loss_execution_cost + financing_cost,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            and math.isclose(
                win_outcome_cost,
                take_profit_execution_cost + financing_cost,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            and entry_p95 is not None
            and cost_entry_p95 is not None
            and math.isclose(
                entry_p95,
                cost_entry_p95,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and entry_allowance is not None
            and entry_allowance + 1e-12 >= entry_p95
            and None not in {
                worst_gross_risk,
                worst_gross_reward,
                worst_net_risk,
                worst_net_reward,
                next_gross_risk,
                next_gross_reward,
                next_net_risk,
                next_net_reward,
            }
            and math.isclose(
                float(worst_net_risk),
                float(worst_gross_risk) + loss_outcome_cost,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(worst_net_reward),
                float(worst_gross_reward) - win_outcome_cost,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(next_net_risk),
                float(next_gross_risk) + loss_outcome_cost,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(next_net_reward),
                float(next_gross_reward) - win_outcome_cost,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            and (
                portfolio_remaining is None
                or float(worst_net_risk) <= portfolio_remaining + 1e-9
            )
        )
        if (
            price_bound_status != "PASSED"
            or price_bound.get("bound_not_looser_than_fresh_ceiling") is not True
            or price_bound.get("current_executable_price_inside_bound") is not True
            or price_bound.get("worst_fill_geometry_passed") is not True
            or price_bound.get("worst_fill_ev_strictly_positive") is not True
            or price_bound.get("worst_fill_quarter_kelly_passed") is not True
            or price_bound.get("worst_fill_loss_caps_passed") is not True
            or price_bound.get("next_adverse_tick_fails") is not True
            or price_bound.get("entry_slippage_allowance_meets_p95") is not True
            or not cost_proofs_frozen
            or numeric_ceiling.get("execution_cost_floor_valid") is not True
            or numeric_ceiling.get("execution_cost_floor_digest_valid") is not True
            or price_bound.get("execution_cost_floor_digest_valid") is not True
            or not cost_identities_valid
            or price_bound.get("risk_engine_status") != "PASSED"
            or order_request.get("priceBound") != bound_text
            or not isinstance(numeric_entry, (int, float))
            or isinstance(numeric_entry, bool)
            or not math.isclose(
                float(numeric_entry),
                float(price_bound.get("price_bound")),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return False
    fresh_max = evidence.get("fresh_max_multiple")
    fresh_unit_cap = evidence.get("fresh_unit_cap_floor")
    if (
        isinstance(fresh_max, bool)
        or not isinstance(fresh_max, (int, float))
        or not math.isfinite(float(fresh_max))
        or float(fresh_max) <= 0.0
        or isinstance(fresh_unit_cap, bool)
        or not isinstance(fresh_unit_cap, int)
        or fresh_unit_cap != math.floor(base_units * float(fresh_max) + 1e-12)
        or final_units > fresh_unit_cap
        or evidence.get("numeric_inputs_passed") is not True
        or evidence.get("final_ratio_passed") is not True
    ):
        return False
    return True


def _verified_decision_receipt_change_issue(
    path: Path | None,
    *,
    expected_sha256: str | None,
) -> dict[str, str] | None:
    if path is None:
        return None
    if expected_sha256 is None:
        return RiskIssue(
            "GPT_CAPITAL_ALLOCATION_RECEIPT_CHANGED_BEFORE_POST",
            "configured GPT decision receipt was not readable at gateway entry; refresh and reverify before broker POST",
        ).__dict__
    current_sha256 = _file_sha256(path)
    if current_sha256 == expected_sha256:
        return None
    return RiskIssue(
        "GPT_CAPITAL_ALLOCATION_RECEIPT_CHANGED_BEFORE_POST",
        "verified GPT decision receipt changed or disappeared after gateway validation; refresh and reverify before broker POST",
    ).__dict__


def _gpt_capital_allocation_binding_validated(
    path: Path | None,
    *,
    expected_sha256: str | None,
    issues: list[dict[str, Any]],
) -> bool:
    """Report authorization only after the exact live receipt passed every GPT gate."""

    if (
        path is None
        or expected_sha256 is None
        or _file_sha256(path) != expected_sha256
        or any(
            str(issue.get("severity") or "BLOCK").upper() == "BLOCK"
            for issue in issues
            if isinstance(issue, dict)
        )
    ):
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    provenance = (
        decision.get("decision_provenance")
        if isinstance(decision.get("decision_provenance"), dict)
        else {}
    )
    return bool(
        str(payload.get("status") or "").upper() == "ACCEPTED"
        and str(decision.get("action") or "").upper() == "TRADE"
        and provenance.get("author_kind") == "CODEX_MARKET_READ"
        and provenance.get("schema_version") == 2
        and isinstance(decision.get("capital_allocation"), dict)
    )


def _pre_post_capital_allocation_edge_validated(
    evidence: dict[str, Any],
) -> bool:
    recheck = evidence.get("capital_allocation_edge_recheck")
    return bool(
        isinstance(recheck, dict)
        and str(recheck.get("status") or "").upper()
        in {"PASSED", "BYPASSED"}
    )


def _gpt_verified_decision_live_send_issues(
    verified_decision_path: Path | None,
    *,
    selected_lane_id: str | None,
    intents_payload: dict[str, Any],
    send: bool,
    expected_sha256: str | None,
    predictive_scout_trade_only: bool = False,
    intent: OrderIntent | None = None,
    base_units: int | None = None,
    authorized_size_multiple: float | None = None,
    authorized_units: int | None = None,
    final_units: int | None = None,
    order_request: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    if not send:
        return []
    if verified_decision_path is None:
        return [
            RiskIssue(
                "GPT_FRESH_TRADE_RECEIPT_REQUIRED_FOR_LIVE_SEND",
                "every fresh-entry live send requires a fresh verified ACCEPTED schema-v2 TRADE receipt; "
                "direct gateway invocation without AI trader verification is forbidden",
            ).__dict__
        ]
    try:
        receipt_bytes = verified_decision_path.read_bytes()
    except OSError as exc:
        return [
            RiskIssue(
                "GPT_VERIFIED_RECEIPT_UNREADABLE_FOR_LIVE_SEND",
                f"verified GPT decision receipt is unreadable before live send: {verified_decision_path}: {exc}",
            ).__dict__
        ]
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if expected_sha256 is None or receipt_sha256 != expected_sha256:
        return [
            RiskIssue(
                "GPT_VERIFIED_RECEIPT_BYTES_MISMATCH_FOR_LIVE_SEND",
                "verified GPT decision bytes differ from the receipt frozen at gateway entry; "
                "live allocation validation cannot switch between receipt versions",
            ).__dict__
        ]
    try:
        payload = json.loads(receipt_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
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
    # The current CODEX_MARKET_READ allocation schema exists only on TRADE.
    # ADD/HEDGE are intent roles, not an alternate broker authorization: they
    # still require an exact schema-v2 TRADE receipt so units cannot bypass the
    # GPT-authored capital-allocation contract.
    allowed_actions = {"TRADE"}
    if action not in allowed_actions:
        issues.append(
            RiskIssue(
                "GPT_FRESH_TRADE_RECEIPT_REQUIRED_FOR_LIVE_SEND",
                f"verified GPT receipt action is {action or 'missing'}; "
                + (
                    "predictive SCOUT requires an exact TRADE receipt (ADD is not a forward experiment authorization)."
                    if predictive_scout_trade_only
                    else "ADD/WAIT/REQUEST_EVIDENCE/non-TRADE cannot send fresh risk; "
                    "encode ADD/HEDGE as the intent role under a schema-v2 TRADE allocation."
                ),
            )
        )

    selected_receipt_lanes = _gpt_receipt_selected_lane_ids(decision)
    if action in allowed_actions and not selected_receipt_lanes:
        issues.append(
            RiskIssue(
                "GPT_SELECTED_LANE_MISSING_FOR_LIVE_SEND",
                "verified GPT TRADE receipt must name selected_lane_id or selected_lane_ids.",
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

    provenance = (
        decision.get("decision_provenance")
        if isinstance(decision.get("decision_provenance"), dict)
        else {}
    )
    issues.extend(_ai_order_authority_live_send_issues(provenance))
    if action == "TRADE":
        issues.extend(
            _codex_capital_allocation_live_send_issues(
                decision=decision,
                selected_lane_id=selected_lane_id,
                base_units=base_units,
                authorized_size_multiple=authorized_size_multiple,
                authorized_units=authorized_units,
                final_units=final_units,
                order_request=order_request,
                intent=intent,
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
    if predictive_scout_trade_only and intent is not None and isinstance(market_read, dict):
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
    if predictive_scout_trade_only and intent is not None:
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


def _ai_order_authority_live_send_issues(
    provenance: dict[str, Any],
) -> list[RiskIssue]:
    """Reject explicit AI order authors while AI order authority is NONE.

    The author check is intentionally exact rather than prefix-based.  Unknown
    or malformed receipt authors continue to fail through the existing verified
    receipt contract, while an explicit future ``FAST_BOT`` provenance is left
    for its own deterministic gateway contract instead of being mislabeled AI.
    """

    author_kind = str(provenance.get("author_kind") or "").strip().upper()
    if (
        AI_ORDER_AUTHORITY == "NONE"
        and author_kind in AI_ORDER_AUTHOR_KINDS
    ):
        return [
            RiskIssue(
                "AI_ORDER_AUTHORITY_NONE",
                f"{author_kind} may supervise or tune the trading bot but cannot authorize a broker entry while AI_ORDER_AUTHORITY=NONE",
            )
        ]
    return []


def verified_trade_size_multiple(
    verified_decision_path: Path | None,
) -> float | None:
    """Return the exact bounded GPT TRADE multiplier for gateway forwarding.

    This is intentionally narrower than receipt verification.  Callers use it
    only to preserve the GPT-authored number when invoking ``run``; the gateway
    still re-reads and validates the complete receipt, allocation hashes,
    selected units, lane, freshness, and final broker request.
    """
    if verified_decision_path is None or not verified_decision_path.exists():
        return None
    try:
        payload = json.loads(verified_decision_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if str(payload.get("status") or "").upper() != "ACCEPTED":
        return None
    decision = payload.get("decision")
    if not isinstance(decision, dict) or str(decision.get("action") or "").upper() != "TRADE":
        return None
    allocation = decision.get("capital_allocation")
    if not isinstance(allocation, dict) or str(allocation.get("decision") or "").upper() != "ALLOCATE":
        return None
    raw_multiple = allocation.get("size_multiple")
    if (
        not isinstance(raw_multiple, (int, float))
        or isinstance(raw_multiple, bool)
        or float(raw_multiple) not in {0.5, 0.75, 1.0}
    ):
        return None
    return float(raw_multiple)


def _codex_capital_allocation_live_send_issues(
    *,
    decision: dict[str, Any],
    selected_lane_id: str | None,
    base_units: int | None,
    authorized_size_multiple: float | None,
    authorized_units: int | None,
    final_units: int | None,
    order_request: dict[str, Any] | None,
    intent: OrderIntent | None,
) -> list[RiskIssue]:
    issues: list[RiskIssue] = []
    provenance = (
        decision.get("decision_provenance")
        if isinstance(decision.get("decision_provenance"), dict)
        else {}
    )
    allocation = (
        decision.get("capital_allocation")
        if isinstance(decision.get("capital_allocation"), dict)
        else None
    )
    if provenance.get("author_kind") != "CODEX_MARKET_READ":
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_AUTHOR_MISMATCH",
                "accepted TRADE must be authored by the fresh CODEX_MARKET_READ allocation path",
            )
        )
    if allocation is None:
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_REQUIRED_FOR_LIVE_SEND",
                "CODEX_MARKET_READ TRADE requires a verified bounded capital_allocation receipt",
            )
        )
        return issues
    expected_fields = {
        "decision",
        "lane_id",
        "size_multiple",
        "selected_units",
        "allocation_board_sha256",
        "rationale",
    }
    if set(allocation) != expected_fields:
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_SCHEMA_INVALID",
                "capital_allocation fields do not match the exact v2 execution schema",
            )
        )
    if provenance.get("schema_version") != 2:
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_SCHEMA_INVALID",
                "CODEX market-read provenance must use schema_version 2",
            )
        )
    claimed_edge_basis = str(
        provenance.get("capital_allocation_edge_basis") or ""
    ).strip().upper()
    m15_recovery_collection_intent = (
        _m15_recovery_edge_collection_prebounded_intent_claim(intent)
    )
    target_path_live_learning_intent = (
        isinstance(intent, OrderIntent)
        and _target_path_live_learning_allocation_claim(intent)
    )
    edge_basis_allowed = bool(
        (
            claimed_edge_basis == M15_RECOVERY_EDGE_COLLECTION_BASIS
            and m15_recovery_collection_intent
        )
        or (
            claimed_edge_basis
            == TARGET_PATH_LIVE_LEARNING_EDGE_COLLECTION_BASIS
            and target_path_live_learning_intent
        )
        or (
            not m15_recovery_collection_intent
            and not target_path_live_learning_intent
            and claimed_edge_basis
            in {
                "EXACT_VEHICLE_ALL_EXIT_NET",
                "EXACT_VEHICLE_TAKE_PROFIT",
                "PREDICTIVE_SCOUT_FORWARD_EVIDENCE",
                "HEDGE_RISK_REDUCTION",
            }
        )
    )
    if not edge_basis_allowed:
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_EDGE_BASIS_MISSING",
                "signed market-read provenance must freeze the board-selected edge basis",
            )
        )
    if str(decision.get("action") or "").upper() != "TRADE" or str(
        allocation.get("decision") or ""
    ).upper() != "ALLOCATE":
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_ACTION_MISMATCH",
                "live fresh entry requires TRADE plus capital_allocation.decision=ALLOCATE",
            )
        )
    allocation_lane = str(allocation.get("lane_id") or "")
    receipt_lanes = _gpt_receipt_selected_lane_ids(decision)
    if (
        not selected_lane_id
        or allocation_lane != selected_lane_id
        or receipt_lanes != (selected_lane_id,)
    ):
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_LANE_MISMATCH",
                "capital allocation, the single verified receipt lane, and the gateway lane must match exactly",
            )
        )

    raw_multiple = allocation.get("size_multiple")
    allowed_ratios = {0.5: (1, 2), 0.75: (3, 4), 1.0: (1, 1)}
    receipt_multiple = (
        float(raw_multiple)
        if isinstance(raw_multiple, (int, float))
        and not isinstance(raw_multiple, bool)
        and float(raw_multiple) in allowed_ratios
        else None
    )
    if receipt_multiple is None:
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_MULTIPLE_MISMATCH",
                "capital allocation size_multiple must be exactly one of 0.5, 0.75, or 1.0",
            )
        )
    raw_selected_units = allocation.get("selected_units")
    selected_units = (
        raw_selected_units
        if isinstance(raw_selected_units, int) and not isinstance(raw_selected_units, bool)
        else None
    )
    if selected_units is None or selected_units <= 0:
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_SELECTED_UNITS_MISMATCH",
                "capital allocation selected_units must be a positive integer",
            )
        )
    normalized_base_units = (
        abs(base_units)
        if isinstance(base_units, int) and not isinstance(base_units, bool)
        else None
    )
    if receipt_multiple is not None and normalized_base_units is not None:
        numerator, denominator = allowed_ratios[receipt_multiple]
        expected_units = normalized_base_units * numerator // denominator
        if expected_units <= 0 or selected_units != expected_units:
            issues.append(
                RiskIssue(
                    "GPT_CAPITAL_ALLOCATION_SELECTED_UNITS_MISMATCH",
                    f"verified allocation units must equal integer floor of current intent units: {expected_units}",
                )
            )
    else:
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_SELECTED_UNITS_MISMATCH",
                "gateway could not bind allocation to positive current intent base units",
            )
        )
    if (
        receipt_multiple is None
        or not isinstance(authorized_size_multiple, (int, float))
        or isinstance(authorized_size_multiple, bool)
        or not math.isfinite(float(authorized_size_multiple))
        or float(authorized_size_multiple) != receipt_multiple
    ):
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_MULTIPLE_MISMATCH",
                "gateway caller size_multiple does not exactly match the verified GPT allocation",
            )
        )
    if (
        selected_units is None
        or not isinstance(authorized_units, int)
        or isinstance(authorized_units, bool)
        or authorized_units != selected_units
    ):
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_PRECLIP_MISMATCH",
                "gateway pre-risk scaled units do not exactly match the verified GPT allocation",
            )
        )

    board_sha = str(allocation.get("allocation_board_sha256") or "")
    provenance_board_sha = str(
        provenance.get("capital_allocation_board_sha256") or ""
    )
    if (
        len(board_sha) != 64
        or any(char not in "0123456789abcdef" for char in board_sha)
        or board_sha != provenance_board_sha
    ):
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_BOARD_MISMATCH",
                "capital-allocation board digest does not match verified provenance",
            )
        )
    canonical_allocation = json.dumps(
        allocation,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    allocation_sha = hashlib.sha256(canonical_allocation).hexdigest()
    if allocation_sha != str(provenance.get("capital_allocation_sha256") or ""):
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_BOARD_MISMATCH",
                "capital_allocation content digest does not match verified provenance",
            )
        )
    provenance_multiple = provenance.get("authorized_size_multiple")
    if (
        receipt_multiple is not None
        and (
            not isinstance(provenance_multiple, (int, float))
            or isinstance(provenance_multiple, bool)
            or not math.isfinite(float(provenance_multiple))
            or float(provenance_multiple) != receipt_multiple
        )
    ):
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_MULTIPLE_MISMATCH",
                "provenance authorized_size_multiple does not match the allocation",
            )
        )
    provenance_units = provenance.get("authorized_units")
    if selected_units is not None and (
        not isinstance(provenance_units, int)
        or isinstance(provenance_units, bool)
        or provenance_units != selected_units
    ):
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_SELECTED_UNITS_MISMATCH",
                "provenance authorized_units does not match the allocation",
            )
        )
    if not str(allocation.get("rationale") or "").strip():
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_SCHEMA_INVALID",
                "capital allocation rationale must be non-empty",
            )
        )

    normalized_final_units = (
        abs(final_units)
        if isinstance(final_units, int) and not isinstance(final_units, bool)
        else None
    )
    if (
        selected_units is not None
        and (
            normalized_final_units is None
            or normalized_final_units <= 0
            or normalized_final_units > selected_units
        )
    ):
        issues.append(
            RiskIssue(
                "GPT_CAPITAL_ALLOCATION_FINAL_UNITS_EXCEEDED",
                "risk-adjusted final intent units must remain positive and no greater than GPT-authorized units",
            )
        )
    if order_request is not None and normalized_final_units is not None:
        try:
            order_units = int(order_request.get("units"))
        except (TypeError, ValueError, OverflowError):
            order_units = 0
        expected_sign = 1 if intent is not None and intent.side == Side.LONG else -1
        if (
            abs(order_units) != normalized_final_units
            or order_units * expected_sign <= 0
            or (selected_units is not None and abs(order_units) > selected_units)
        ):
            issues.append(
                RiskIssue(
                    "GPT_CAPITAL_ALLOCATION_ORDER_UNITS_MISMATCH",
                    "broker order units must match the final intent sign and stay within GPT-authorized units",
                )
            )
    return issues


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


def _target_path_live_learning_allocation_claim(intent: OrderIntent) -> bool:
    """Re-identify the signed micro-risk target-path collection route."""

    metadata = dict(intent.metadata or {})
    if (
        str(metadata.get("target_path_live_mode") or "").strip().upper()
        != "LIVE_LEARNING"
        or str(metadata.get("valid_as_target_path") or "").strip().upper()
        != "YES"
        or intent.order_type != OrderType.LIMIT
        or not 0 < abs(int(intent.units)) <= 1_000
    ):
        return False
    risk_pct = _metadata_float(metadata, "risk_pct")
    risk_yen = _metadata_float(metadata, "risk_yen")
    target_yen = _metadata_float(metadata, "target_yen")
    suggested_units = _metadata_int(metadata, "suggested_units")
    if (
        risk_pct is None
        or risk_pct <= 0.0
        or risk_pct > 0.15
        or risk_yen is None
        or risk_yen <= 0.0
        or target_yen is None
        or target_yen <= 0.0
        or target_yen / risk_yen < 1.5
        or suggested_units != abs(int(intent.units))
    ):
        return False
    if intent.side == Side.LONG:
        geometry_valid = intent.sl < intent.entry < intent.tp
    else:
        geometry_valid = intent.tp < intent.entry < intent.sl
    if not geometry_valid:
        return False
    return not any(
        str(issue.get("severity") or "BLOCK").upper() == "BLOCK"
        for issue in _target_path_live_send_issues(intent, send=True)
    )


def _strategy_profile_live_send_issues(
    intent: OrderIntent,
    issues: tuple[Any, ...] | list[Any],
) -> tuple[Any, ...]:
    """Apply the profile gate without undoing the signed micro-learning path.

    A valid target-path live-learning intent is deliberately bounded to a
    passive LIMIT, at most 1,000 units, at most 0.15% NAV risk, and at least
    1.5 reward/risk.  Its purpose is to collect the exact execution evidence
    needed to move a WATCH_ONLY profile forward.  Reapplying only the generic
    WATCH_ONLY ``STRATEGY_NOT_ELIGIBLE`` result after that contract validates
    makes the evidence-collection route impossible.  Every other profile
    issue remains intact.
    """

    normalized = tuple(issues)
    if not _target_path_live_learning_allocation_claim(intent):
        return normalized
    return tuple(
        issue
        for issue in normalized
        if str(getattr(issue, "code", "")) != "STRATEGY_NOT_ELIGIBLE"
    )


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
        order_request = _oanda_order_request(intent)
    except ValueError as exc:
        return None, [RiskIssue("ORDER_REQUEST_INVALID", str(exc)).__dict__]
    mismatches: list[str] = []

    def require_grid(label: str, raw_value: Any) -> None:
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError):
            mismatches.append(label)
            return
        if (
            not math.isfinite(parsed)
            or parsed <= 0.0
            or not math.isclose(
                parsed,
                float(_price(intent.pair, parsed)),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            mismatches.append(label)

    if "price" in order_request:
        require_grid("entry", intent.entry)
    if isinstance(order_request.get("takeProfitOnFill"), dict):
        require_grid("take_profit", intent.tp)
    if isinstance(order_request.get("stopLossOnFill"), dict):
        initial_sl_on = os.environ.get(
            "QR_NEW_ENTRY_INITIAL_SL",
            "",
        ).strip() in {"1", "true", "TRUE", "yes", "YES"}
        if (
            initial_sl_on
            or not _trader_sl_repair_disabled()
            or _requires_intent_stop_on_fill(intent)
        ):
            require_grid("stop_loss", intent.sl)
        else:
            require_grid("disaster_stop", (intent.metadata or {}).get("disaster_sl"))
    issues = []
    if mismatches:
        issues.append(
            RiskIssue(
                "BROKER_PRICE_PRECISION_MISMATCH",
                "broker-carried price field(s) are off the instrument tick "
                f"grid ({', '.join(mismatches)}); regenerate the exact intent "
                "instead of validating one float and POSTing its rounded value",
            ).__dict__
        )
    return order_request, issues


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


def _range_vehicle_candidate_receipt_from_intent(
    intent: OrderIntent,
    order_request: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Bind a final gateway request to its immutable RANGE candidate shape.

    Candidate generation cannot prove activation, final repricing, or broker
    acceptance.  This receipt carries the stable identity into the gateway
    artifact while keeping broker truth and live permission explicitly
    separate.  Sent predictive-SCOUT results are subsequently indexed by the
    existing durable gateway-receipt path.
    """

    metadata = intent.metadata or {}
    candidate_id = str(metadata.get("range_vehicle_candidate_id") or "")
    shape_sha = str(metadata.get("range_vehicle_shape_sha256") or "")
    projection_sha = str(
        metadata.get("range_vehicle_gateway_projection_sha256") or ""
    )
    if not candidate_id and not shape_sha and not projection_sha:
        return None
    if not all(
        len(value) == 64 and all(char in "0123456789abcdef" for char in value)
        for value in (candidate_id, shape_sha, projection_sha)
    ):
        return {
            "status": "INVALID_CANDIDATE_IDENTITY",
            "candidate_id": candidate_id or None,
            "vehicle_shape_sha256": shape_sha or None,
            "gateway_contract_projection_sha256": projection_sha or None,
            "live_permission_allowed": False,
        }
    request_sha = (
        _canonical_json_sha256(order_request)
        if isinstance(order_request, dict)
        else None
    )
    take_profit_on_fill = (
        order_request.get("takeProfitOnFill")
        if isinstance(order_request, dict)
        and isinstance(order_request.get("takeProfitOnFill"), dict)
        else {}
    )
    stop_loss_on_fill = (
        order_request.get("stopLossOnFill")
        if isinstance(order_request, dict)
        and isinstance(order_request.get("stopLossOnFill"), dict)
        else {}
    )
    return {
        "status": (
            "CANDIDATE_IDENTITY_CARRIED_WITH_FINAL_GATEWAY_ORDER_REQUEST"
            if request_sha is not None
            else "CANDIDATE_PRESENT_ORDER_REQUEST_MISSING"
        ),
        "candidate_id": candidate_id,
        "vehicle_shape_sha256": shape_sha,
        "gateway_contract_projection_sha256": projection_sha,
        "final_order_request_sha256": request_sha,
        "final_order_contract": (
            {
                "instrument": order_request.get("instrument"),
                "type": order_request.get("type"),
                "units": order_request.get("units"),
                "price": order_request.get("price"),
                "time_in_force": order_request.get("timeInForce"),
                "gtd_time": order_request.get("gtdTime"),
                "position_fill": order_request.get("positionFill"),
                "take_profit_on_fill": take_profit_on_fill.get("price"),
                "stop_loss_on_fill": stop_loss_on_fill.get("price"),
            }
            if isinstance(order_request, dict)
            else None
        ),
        "candidate_ledger_row_proved": False,
        "join_caveat": (
            "this gateway receipt binds the carried identity claim to the final request; "
            "a separate candidate-ledger receipt join is required to prove the source row"
        ),
        "broker_acceptance_proved": False,
        "live_permission_allowed": False,
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


def _decision_lineage_receipt_from_intent(
    intent: OrderIntent,
    order_request: dict[str, Any] | None,
) -> dict[str, Any] | None:
    try:
        lineage = lineage_from_metadata(intent.metadata)
    except DecisionExecutionLineageError as exc:
        return {
            "status": "INVALID",
            "error": str(exc),
            "live_permission": False,
        }
    if lineage is None:
        return None
    client_extensions = (
        order_request.get("clientExtensions")
        if isinstance(order_request, dict)
        and isinstance(order_request.get("clientExtensions"), dict)
        else {}
    )
    client_extension_id = str(client_extensions.get("id") or "") or None
    return {
        "status": "BOUND_TO_SELECTED_INTENT",
        **lineage.to_dict(),
        "client_extension_id": client_extension_id,
        "broker_lineage_token_embedded": bool(
            client_extension_id and lineage.lineage_token in client_extension_id
        ),
        "live_permission": False,
    }


def _sizing_evidence_from_intent(
    intent: OrderIntent,
    *,
    gateway_max_loss_jpy: float | None,
    requested_units: int | None,
    scaled_units: int | None,
    authorized_size_multiple: float | None = None,
    authorized_units: int | None = None,
    capital_allocation_validated: bool = False,
) -> dict[str, Any]:
    metadata = dict(intent.metadata or {})
    out: dict[str, Any] = {
        "requested_units": requested_units,
        "scaled_units": scaled_units,
        "base_units": abs(requested_units) if requested_units is not None else None,
        "caller_size_multiple": authorized_size_multiple,
        "caller_preclip_units": authorized_units,
        "capital_allocation_validated": capital_allocation_validated,
        "authorized_size_multiple": (
            authorized_size_multiple if capital_allocation_validated else None
        ),
        "authorized_units": authorized_units if capital_allocation_validated else None,
        "final_units": abs(scaled_units) if scaled_units is not None else None,
        "final_effective_size_multiple": (
            round(abs(scaled_units) / abs(requested_units), 8)
            if scaled_units is not None and requested_units
            else None
        ),
        "allocation_status": (
            "RISK_DOWNSIZED"
            if capital_allocation_validated
            and authorized_units is not None
            and scaled_units is not None
            and abs(scaled_units) < authorized_units
            else "AUTHORIZED_EXACT"
            if capital_allocation_validated
            and authorized_units is not None
            and scaled_units is not None
            else None
        ),
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


def _intent_with_broker_price_precision(intent: OrderIntent) -> OrderIntent:
    return replace(
        intent,
        tp=float(_price(intent.pair, intent.tp)),
        sl=float(_price(intent.pair, intent.sl)),
    )


def _price_precision(pair: str) -> int:
    return 3 if pair.endswith("_JPY") else 5


def _price_tick(pair: str) -> float:
    return 10 ** -_price_precision(pair)


def _intent_with_gateway_metadata(intent: OrderIntent, lane_id: str) -> OrderIntent:
    if not lane_id:
        return intent
    metadata = dict(intent.metadata or {})
    submitted_lane_id = str(metadata.get("lane_id") or "").strip()
    if submitted_lane_id and submitted_lane_id != lane_id:
        metadata["submitted_lane_id"] = submitted_lane_id
        metadata["selected_lane_metadata_mismatch"] = {
            "submitted_lane_id": submitted_lane_id,
            "selected_lane_id": lane_id,
        }
    else:
        # These fields are gateway-owned audit evidence.  Exact/missing source
        # metadata must not inherit a stale mismatch marker from an older
        # serialized intent.
        metadata.pop("submitted_lane_id", None)
        metadata.pop("selected_lane_metadata_mismatch", None)
    # The result selected from order_intents is the execution identity.  Risk
    # and broker extensions must never inherit a sibling/parent lane embedded
    # in the intent payload, because that can bypass an exact-lane tightening.
    metadata["lane_id"] = lane_id
    return replace(intent, metadata=metadata)


def _selected_lane_metadata_issues(intent: OrderIntent) -> tuple[RiskIssue, ...]:
    metadata = intent.metadata or {}
    mismatch = metadata.get("selected_lane_metadata_mismatch")
    if not isinstance(mismatch, dict):
        return ()
    selected_lane_id = str(mismatch.get("selected_lane_id") or "").strip()
    submitted_lane_id = str(mismatch.get("submitted_lane_id") or "").strip()
    if (
        not selected_lane_id
        or not submitted_lane_id
        or selected_lane_id == submitted_lane_id
    ):
        return ()
    return (
        RiskIssue(
            "SELECTED_LANE_METADATA_MISMATCH",
            "gateway selected lane "
            f"{selected_lane_id} conflicts with intent metadata lane "
            f"{submitted_lane_id}; the selected lane remains authoritative, but "
            "live send is blocked until the intent is regenerated with one "
            "exact identity",
        ),
    )


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
    lineage_token = str(
        (intent.metadata or {}).get("gpt_decision_lineage_token") or ""
    ).strip()
    if lineage_token:
        # OANDA caps ClientExtensions.id at 128 characters.  The token is a
        # short content address for both full 256-bit lineage ids; owner stays
        # in `tag` and the full lane stays in `comment` unchanged.
        suffix = f"-{lineage_token}"
        base = f"{base[: max(0, 128 - len(suffix))]}{suffix}"
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


def _intent_from_reserved_order_request(
    final_intent: OrderIntent,
    *,
    order_request: dict[str, Any],
    expected_price_bound: str | None = None,
) -> OrderIntent:
    """Verify reserved transport semantics without replacing virtual rails."""

    expected = _oanda_order_request(final_intent)
    scalar_keys = (
        "type",
        "instrument",
        "units",
        "positionFill",
        "timeInForce",
        "price",
        "gtdTime",
    )
    for key in scalar_keys:
        expected_present = key in expected
        actual_present = key in order_request
        if expected_present != actual_present or (
            expected_present and order_request.get(key) != expected.get(key)
        ):
            raise ValueError(
                f"reserved order {key} does not match the final reconciled intent"
            )
    for key in ("takeProfitOnFill", "stopLossOnFill"):
        expected_payload = expected.get(key)
        actual_payload = order_request.get(key)
        if (expected_payload is None) != (actual_payload is None):
            raise ValueError(
                f"reserved order {key} presence does not match attachment policy"
            )
        if expected_payload is not None:
            if not isinstance(actual_payload, dict) or actual_payload.get(
                "price"
            ) != expected_payload.get("price"):
                raise ValueError(
                    f"reserved order {key}.price does not match final intent policy"
                )
    actual_price_bound = order_request.get("priceBound")
    if (expected_price_bound is None) != (actual_price_bound is None) or (
        expected_price_bound is not None
        and actual_price_bound != expected_price_bound
    ):
        raise ValueError(
            "reserved order priceBound does not match the frozen reconciled "
            "worst-fill proof"
        )
    return final_intent


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

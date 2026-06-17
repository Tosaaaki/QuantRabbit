from __future__ import annotations

import math
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
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
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_LIVE_ORDER_STAGE_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_STRATEGY_PROFILE,
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
from quant_rabbit.strategy.intent_generator import (
    _daily_risk_budget_from_state,
    _expired_pending_projection_count,
    _per_trade_risk_from_state,
)
from quant_rabbit.strategy.profile import StrategyProfile


class ExecutionClient(Protocol):
    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot: ...

    def post_order_json(self, order_request: dict[str, Any]) -> dict[str, Any]: ...


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
        self_improvement_audit: Path | None = None,
        verified_decision_path: Path | None = None,
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
        self.self_improvement_audit = self_improvement_audit
        # When the automation cycle stages a receipt that gpt-trader-decision
        # just verified, this points at the ACCEPTED verification artifact so
        # the LATEST_GPT_DECISION_STALE audit finding can be recognized as
        # already repaired. Manual stage-live-order paths leave it None.
        self.verified_decision_path = verified_decision_path

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
        portfolio_loss_cap = (
            self.portfolio_loss_cap_jpy
            if self.portfolio_loss_cap_jpy is not None
            else _daily_risk_budget_from_state(DEFAULT_DAILY_TARGET_STATE)
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
        )
        send_issues = _send_guard_issues(send=send, confirm_live=confirm_live, lane_id=lane_id)
        all_blocked = (
            any(issue["severity"] == "BLOCK" for issue in risk_issues)
            or any(issue["severity"] == "BLOCK" for issue in strategy_issues)
            or any(issue["severity"] == "BLOCK" for issue in intent_status_issues)
            or any(issue["severity"] == "BLOCK" for issue in projection_expiry_issues)
            or any(issue["severity"] == "BLOCK" for issue in self_improvement_issues)
            or any(issue["severity"] == "BLOCK" for issue in send_issues)
            or any(issue.severity == "BLOCK" for issue in scale_issues)
        )
        order_request, order_build_issues = _build_order_request(intent)
        all_blocked = all_blocked or any(issue["severity"] == "BLOCK" for issue in order_build_issues)
        response = None
        sent = False
        entry_thesis_record = None
        entry_thesis_issues: list[dict[str, str]] = []
        status = "BLOCKED" if all_blocked else "STAGED"
        if send and order_request is not None and not all_blocked:
            response = self.client.post_order_json(order_request)
            sent = True
            status = "SENT"
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
            "context_evidence": _context_evidence_from_intent(intent),
            "risk_metrics": asdict(risk.metrics) if risk.metrics else None,
            "attached_stop_risk_metrics": _attached_stop_risk_metrics(intent, order_request, risk.metrics),
            "risk_issues": [
                *risk_issues,
                *intent_status_issues,
                *projection_expiry_issues,
                *self_improvement_issues,
                *send_issues,
                *order_build_issues,
                *[issue.__dict__ for issue in scale_issues],
                *entry_thesis_issues,
            ],
            "strategy_issues": list(strategy_issues),
            "send_requested": send,
            "sent": sent,
            "response": response,
            "entry_thesis_record": entry_thesis_record,
            "snapshot": {
                "fetched_at_utc": snapshot.fetched_at_utc.isoformat(),
                "positions": len(snapshot.positions),
                "orders": len(snapshot.orders),
                "quotes": len(snapshot.quotes),
            },
            "quote_refresh_attempts": quote_refresh_attempts,
            "size_multiple": size_multiple,
            "requested_units": requested_units,
            "scaled_units": scaled_units,
            "portfolio_position_cap": portfolio_position_cap,
        }
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
                allow_basket_pending=True,
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
                accepted_risk_jpy += float(metrics.get("risk_jpy") or 0.0)
                accepted_margin_jpy += float(metrics.get("estimated_margin_jpy") or 0.0)
                # Dry-run staged orders are not visible in broker truth, so
                # carry them synthetically. Live sends are verified against the
                # next fresh broker snapshot to avoid double-counting margin.
                if not item_result.get("sent"):
                    validation_cumulative_risk_jpy += float(metrics.get("risk_jpy") or 0.0)
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
        portfolio_loss_cap = self._resolve_portfolio_loss_cap_jpy()
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
        )
        send_issues = _send_guard_issues(send=send, confirm_live=confirm_live, lane_id=lane_id_arg)
        all_blocked = (
            any(issue["severity"] == "BLOCK" for issue in risk_issues)
            or any(issue["severity"] == "BLOCK" for issue in strategy_issues)
            or any(issue["severity"] == "BLOCK" for issue in intent_status_issues)
            or any(issue["severity"] == "BLOCK" for issue in projection_expiry_issues)
            or any(issue["severity"] == "BLOCK" for issue in self_improvement_issues)
            or any(issue["severity"] == "BLOCK" for issue in send_issues)
            or any(issue.severity == "BLOCK" for issue in scale_issues)
        )
        order_request, order_build_issues = _build_order_request(intent)
        all_blocked = all_blocked or any(issue["severity"] == "BLOCK" for issue in order_build_issues)
        response = None
        sent = False
        entry_thesis_record = None
        entry_thesis_issues: list[dict[str, str]] = []
        status = "BLOCKED" if all_blocked else "STAGED"
        if send and order_request is not None and not all_blocked:
            response = self.client.post_order_json(order_request)
            sent = True
            status = "SENT"
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
            "context_evidence": _context_evidence_from_intent(intent),
            "risk_metrics": asdict(risk.metrics) if risk.metrics else None,
            "attached_stop_risk_metrics": _attached_stop_risk_metrics(intent, order_request, risk.metrics),
            "risk_issues": [
                *risk_issues,
                *intent_status_issues,
                *projection_expiry_issues,
                *self_improvement_issues,
                *send_issues,
                *order_build_issues,
                *[issue.__dict__ for issue in scale_issues],
                *entry_thesis_issues,
            ],
            "strategy_issues": list(strategy_issues),
            "send_requested": send,
            "sent": sent,
            "response": response,
            "entry_thesis_record": entry_thesis_record,
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
        return (
            self.portfolio_loss_cap_jpy
            if self.portfolio_loss_cap_jpy is not None
            else _daily_risk_budget_from_state(DEFAULT_DAILY_TARGET_STATE)
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
        ).validate(intent, snapshot, for_live_send=True)

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
        if metrics is None or metrics.risk_jpy <= max_loss_jpy:
            issues: list[RiskIssue] = []
        else:
            scale = _capacity_scale(abs(intent.units), metrics.risk_jpy, max_loss_jpy)
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
                    f"above cap {max_loss_jpy:.0f} JPY; clipped to {scaled_units}u.",
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


def _selected_parent_lane_key(selected: dict[str, Any], requested_lane_id: str | None) -> str:
    intent = selected.get("intent") if isinstance(selected, dict) else None
    metadata = intent.get("metadata") if isinstance(intent, dict) and isinstance(intent.get("metadata"), dict) else {}
    parent = metadata.get("parent_lane_id") if isinstance(metadata, dict) else None
    if isinstance(parent, str) and parent.strip():
        return parent.strip()
    lane_id = str(selected.get("lane_id") or requested_lane_id or "")
    return _parent_lane_id_from_lane_id(lane_id)


def _parent_lane_id_from_lane_id(lane_id: str) -> str:
    if lane_id.endswith(":MARKET"):
        return lane_id[: -len(":MARKET")]
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
    if geometry in seen_geometry:
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
        lane_id = _lane_id_from_extension_comment(extension.get("comment"))
        if lane_id:
            return _parent_lane_id_from_lane_id(lane_id)
    return None


def _lane_id_from_extension_comment(comment: Any) -> str | None:
    if not isinstance(comment, str):
        return None
    for token in comment.split():
        if token.startswith("lane="):
            lane_id = token[len("lane="):].strip()
            return lane_id or None
    return None


def _intent_geometry_key(intent: OrderIntent) -> tuple[object, ...]:
    return (
        intent.pair,
        intent.side.value,
        intent.order_type.value,
        _price_key(intent.pair, intent.entry),
        _price_key(intent.pair, intent.tp if _attach_take_profit_on_fill(intent) else None),
        _price_key(intent.pair, _attached_stop_loss_price(intent)),
    )


def _intent_declares_hedge(intent: OrderIntent) -> bool:
    return str((intent.metadata or {}).get("position_intent") or "").upper() == "HEDGE"


def _attached_stop_loss_price(intent: OrderIntent) -> float | None:
    initial_sl_on = os.environ.get("QR_NEW_ENTRY_INITIAL_SL", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }
    if initial_sl_on or not _trader_sl_repair_disabled():
        return intent.sl
    disaster_sl = (intent.metadata or {}).get("disaster_sl")
    if disaster_sl is None:
        return None
    try:
        return float(disaster_sl)
    except (TypeError, ValueError):
        return None


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
    precision = 3 if pair.endswith("_JPY") else 5
    return round(float(value), precision)


def _send_guard_issues(*, send: bool, confirm_live: bool, lane_id: str | None) -> list[dict[str, str]]:
    issues: list[RiskIssue] = []
    if send and not confirm_live:
        issues.append(RiskIssue("LIVE_CONFIRMATION_REQUIRED", "live send requires --confirm-live"))
    if send and not lane_id:
        issues.append(RiskIssue("LANE_ID_REQUIRED_FOR_SEND", "live send requires an explicit --lane-id"))
    return [issue.__dict__ for issue in issues]


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
    blockers: list[str] = []
    for item in payload.get("findings", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("priority") or "").upper() != "P0":
            continue
        code = str(item.get("code") or "SELF_IMPROVEMENT_P0")
        if code == "LATEST_GPT_DECISION_STALE" and verification_postdates_audit:
            # The decision being staged was verified ACCEPTED after this audit
            # ran, so the audit's stale-decision verdict is about an older
            # receipt (mirrors gpt_trader._self_improvement_trade_blockers).
            # With a 20-minute audit cadence and a slower decision cadence the
            # streak otherwise re-blocks the first staging attempt of every
            # fresh receipt. Manual stage-live-order paths do not pass
            # verified_decision_path and keep the strict streak gate.
            continue
        if _self_improvement_gateway_non_blocker(code, item):
            continue
        message = str(item.get("message") or code)
        blockers.append(f"{code}: {message}")
    if not blockers:
        return []
    return [
        RiskIssue(
            "SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER",
            "self-improvement P0 blocks new live entry risk: " + "; ".join(blockers[:3]),
        ).__dict__
    ]


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
    if initial_sl_on or not _trader_sl_repair_disabled():
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
        # inside the 20-minute blind window between cycles.
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
        order["timeInForce"] = "GTC"
    return order


def _oanda_order_type(order_type: OrderType) -> str:
    if order_type == OrderType.STOP_ENTRY:
        return "STOP"
    return order_type.value


def _position_fill(intent: OrderIntent) -> str:
    raw = str(intent.metadata.get("position_fill") or "").upper()
    if raw in {"DEFAULT", "OPEN_ONLY", "REDUCE_FIRST", "REDUCE_ONLY"}:
        return raw
    if str(intent.metadata.get("position_intent") or "").upper() in {"HEDGE", "PYRAMID"}:
        return "OPEN_ONLY"
    return "DEFAULT"


def _price(pair: str, value: float) -> str:
    precision = 3 if pair.endswith("_JPY") else 5
    return f"{value:.{precision}f}"


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
    return f"{CLIENT_ORDER_ID_PREFIX}-{intent.pair.replace('_', '')}-{intent.side.value[0]}-{digest}"[:128]


def _comment(intent: OrderIntent) -> str:
    desk = str(intent.metadata.get("desk") or "vnext")
    role = str(intent.metadata.get("campaign_role") or "")
    lane_id = _gateway_lane_id(intent)
    lane_part = f" lane={lane_id}" if lane_id else ""
    text = f"qr-vnext{lane_part} desk={desk} role={role}".strip()
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

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from quant_rabbit.analysis.sessions import tag_bar
from quant_rabbit.instruments import NORMAL_SPREAD_PIPS, instrument_pip_factor
from quant_rabbit.models import BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.operator_manual import is_operator_managed_manual_owner, operator_manual_tp_modify_blocked
from quant_rabbit.predictive_scout import predictive_scout_broker_raw_claimed
from quant_rabbit.paths import DEFAULT_POSITION_EXECUTION, DEFAULT_POSITION_EXECUTION_REPORT
from quant_rabbit.position_execution_contract import (
    POSITION_EXECUTION_SNAPSHOT_MAX_AGE_SECONDS,
    POSITION_EXECUTION_SNAPSHOT_MAX_FUTURE_SKEW_SECONDS,
)
from quant_rabbit.position_execution_evidence import (
    POSITION_EXECUTION_SNAPSHOT_EVIDENCE_FIELD,
    persist_position_execution_snapshot_evidence,
)
from quant_rabbit.risk import RiskPolicy, _spread_session_multiplier_from_tag
from quant_rabbit.strategy.position_manager import (
    ACTION_BREAK_EVEN_STOP,
    ACTION_EXTEND_TP,
    ACTION_HARVEST_TP,
    ACTION_HOLD_PROTECTED,
    ACTION_NARROW_TP,
    ACTION_PROFIT_PROTECT,
    ACTION_REPAIR_PROTECTION,
    ACTION_REPAIR_TAKE_PROFIT,
    ACTION_REVIEW_EXIT,
    ACTION_TAKE_PROFIT_MARKET,
    ManagedPosition,
    PositionManagementDecision,
)


POSITION_PROTECTION_CLOSE_PROVENANCE = "position_protection_gateway"


class PositionExecutionClient(Protocol):
    def replace_trade_dependent_orders(self, trade_id: str, order_request: dict[str, Any]) -> dict[str, Any]: ...

    def close_trade_with_provenance(
        self,
        trade_id: str,
        units: str = "ALL",
        *,
        provenance: str,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PositionExecutionSummary:
    status: str
    output_path: Path
    report_path: Path
    sent: bool
    actions: int
    blocked: int


class PositionProtectionGateway:
    """Execute only risk-reducing position-management actions."""

    def __init__(
        self,
        *,
        client: PositionExecutionClient,
        output_path: Path = DEFAULT_POSITION_EXECUTION,
        report_path: Path = DEFAULT_POSITION_EXECUTION_REPORT,
        live_enabled: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.client = client
        self.output_path = output_path
        self.report_path = report_path
        self.live_enabled = live_enabled
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def run(
        self,
        *,
        decision: PositionManagementDecision,
        snapshot: BrokerSnapshot,
        send: bool = False,
    ) -> PositionExecutionSummary:
        generated_at_utc = self.clock().astimezone(timezone.utc)
        generated_at = generated_at_utc.isoformat()
        snapshot_trade_id_duplicates = _duplicate_trade_ids(
            position.trade_id for position in snapshot.positions
        )
        decision_trade_id_duplicates = _duplicate_trade_ids(
            managed.trade_id for managed in decision.positions
        )
        snapshot_evidence: dict[str, Any] | None = None
        snapshot_evidence_error: str | None = None
        positions = {position.trade_id: position for position in snapshot.positions}
        actions = [
            self._plan_action(
                managed,
                positions.get(managed.trade_id),
                snapshot,
                generated_at_utc,
                decision.snapshot_fetched_at_utc,
            )
            for managed in decision.positions
        ]
        if send and any(action["request"] is not None for action in actions):
            try:
                snapshot_evidence = persist_position_execution_snapshot_evidence(
                    snapshot=snapshot,
                    receipt_path=self.output_path,
                )
            # This is the last local boundary before a broker POST.  Any
            # serialization/validation failure must become a durable block,
            # including malformed runtime dataclass fields.
            except Exception as exc:  # noqa: BLE001
                snapshot_evidence_error = str(exc)
        preflight_issues: list[dict[str, str]] = []
        if snapshot_trade_id_duplicates:
            preflight_issues.append(
                {
                    "severity": "BLOCK",
                    "code": "BROKER_SNAPSHOT_TRADE_ID_DUPLICATE",
                    "message": (
                        "broker snapshot contains duplicate trade ids; no position "
                        "mutation may be sent: "
                        + ",".join(snapshot_trade_id_duplicates)
                    ),
                }
            )
        if decision_trade_id_duplicates:
            preflight_issues.append(
                {
                    "severity": "BLOCK",
                    "code": "POSITION_DECISION_TRADE_ID_DUPLICATE",
                    "message": (
                        "position-management decision contains duplicate trade ids; "
                        "no position mutation may be sent: "
                        + ",".join(decision_trade_id_duplicates)
                    ),
                }
            )
        if snapshot_evidence_error is not None:
            preflight_issues.append(
                {
                    "severity": "BLOCK",
                    "code": "BROKER_SNAPSHOT_EVIDENCE_PERSIST_FAILED",
                    "message": (
                        "exact pre-send broker snapshot evidence could not be "
                        f"persisted: {snapshot_evidence_error}"
                    ),
                }
            )
        if preflight_issues:
            for action in actions:
                if action["request"] is not None:
                    action["issues"].extend(dict(issue) for issue in preflight_issues)
        if send and not self.live_enabled:
            for action in actions:
                if action["request"] is not None:
                    action["issues"].append(
                        {
                            "severity": "BLOCK",
                            "code": "LIVE_DISABLED",
                            "message": "position protection write requires QR_LIVE_ENABLED=1",
                        }
                    )

        responses: list[dict[str, Any] | None] = []
        for managed, action in zip(decision.positions, actions, strict=True):
            request = action["request"]
            blocked = _has_block(action)
            response = None
            if send and request is not None and not blocked:
                boundary_now = _aware_utc_datetime(self.clock())
                if boundary_now is not None:
                    action["send_boundary_checked_at_utc"] = (
                        boundary_now.isoformat()
                    )
                position = positions.get(managed.trade_id)
                boundary_issue = _send_boundary_identity_issue(
                    action=action,
                    request=request,
                    managed=managed,
                    position=position,
                    snapshot=snapshot,
                    decision_snapshot_fetched_at_utc=(
                        decision.snapshot_fetched_at_utc
                    ),
                    boundary_now_utc=boundary_now,
                )
                if boundary_issue is not None:
                    action["issues"].append(boundary_issue)
                    blocked = True
            if send and request is not None and not blocked:
                try:
                    action["broker_post_attempted"] = True
                    if request["type"] == "CLOSE":
                        response = _close_trade_with_supported_provenance(
                            self.client,
                            str(request["trade_id"]),
                            str(request.get("units") or "ALL"),
                            provenance=POSITION_PROTECTION_CLOSE_PROVENANCE,
                        )
                    elif request["type"] == "DEPENDENT_ORDER_REPLACE":
                        response = self.client.replace_trade_dependent_orders(
                            str(request["trade_id"]),
                            dict(request["order_request"]),
                        )
                    action["sent"] = True
                except Exception as exc:  # noqa: BLE001
                    action["issues"].append(
                        {
                            "severity": "BLOCK",
                            "code": _send_error_code(str(request["type"])),
                            "message": str(exc),
                        }
                    )
                    response = {"error": str(exc)}
            responses.append(response)
            action["response"] = response

        actionable = sum(1 for action in actions if action["request"] is not None)
        blocked_count = sum(1 for action in actions if _has_block(action))
        sent_count = sum(1 for action in actions if action.get("sent"))
        status = _status(actionable=actionable, blocked=blocked_count, sent=sent_count, send=send)
        result = {
            "generated_at_utc": generated_at,
            "cycle_step_run_id": os.environ.get("QR_CYCLE_STEP_RUN_ID"),
            POSITION_EXECUTION_SNAPSHOT_EVIDENCE_FIELD: snapshot_evidence,
            "status": status,
            "send_requested": send,
            "sent": sent_count > 0,
            "actions": actions,
        }
        self._write_result(result)
        self._write_report(result)
        return PositionExecutionSummary(
            status=status,
            output_path=self.output_path,
            report_path=self.report_path,
            sent=sent_count > 0,
            actions=actionable,
            blocked=blocked_count,
        )

    def _plan_action(
        self,
        managed: ManagedPosition,
        position: BrokerPosition | None,
        snapshot: BrokerSnapshot,
        generated_at_utc: datetime,
        decision_snapshot_fetched_at_utc: str | None,
    ) -> dict[str, Any]:
        action: dict[str, Any] = {
            "trade_id": managed.trade_id,
            "pair": managed.pair,
            "owner": managed.owner,
            "broker_position_identity": None,
            "management_action": managed.action,
            "reasons": list(managed.reasons),
            "request": None,
            "issues": [],
            "sent": False,
            "broker_post_attempted": False,
            "send_boundary_checked_at_utc": None,
            "response": None,
        }
        if position is None:
            action["issues"].append(
                {
                    "severity": "BLOCK",
                    "code": "POSITION_NOT_FOUND",
                    "message": "managed position is no longer open in broker snapshot",
                }
            )
            return action
        broker_owner = _position_owner_value(position.owner)
        broker_pair = str(position.pair or "").strip()
        broker_trade_id = str(position.trade_id or "").strip()
        snapshot_fetched_at = _aware_utc_datetime(snapshot.fetched_at_utc)
        action["trade_id"] = broker_trade_id
        action["pair"] = broker_pair
        action["owner"] = broker_owner
        if snapshot_fetched_at is not None:
            action["broker_position_identity"] = {
                "snapshot_fetched_at_utc": snapshot_fetched_at.isoformat(),
                "trade_id": broker_trade_id,
                "pair": broker_pair,
                "owner": broker_owner,
            }
        manual_tp_owner = is_operator_managed_manual_owner(position.owner)
        if position.owner != Owner.TRADER and not manual_tp_owner:
            action["issues"].append(
                {
                    "severity": "BLOCK",
                    "code": "NON_TRADER_POSITION",
                    "message": f"refusing to modify external position id={position.trade_id}",
                }
            )
            return action
        if (
            predictive_scout_broker_raw_claimed(position.raw)
            and managed.action != ACTION_HOLD_PROTECTED
        ):
            action["issues"].append(
                {
                    "severity": "BLOCK",
                    "code": "PREDICTIVE_SCOUT_EXIT_GEOMETRY_FROZEN",
                    "message": (
                        "predictive SCOUT must resolve through its exact entry-time broker TP/SL; "
                        "position-management close and dependent-order replacement are forbidden"
                    ),
                }
            )
            return action
        if managed.action == ACTION_HOLD_PROTECTED:
            return action
        if managed.action == ACTION_TAKE_PROFIT_MARKET:
            if manual_tp_owner:
                action["issues"].append(
                    {
                        "severity": "BLOCK",
                        "code": "MANUAL_POSITION_CLOSE_FORBIDDEN",
                        "message": "manual/tagless positions are TP-managed only; market close is forbidden",
                    }
                )
                return action
            if position.unrealized_pl_jpy <= 0:
                action["issues"].append(
                    {
                        "severity": "BLOCK",
                        "code": "PROFIT_MARKET_CLOSE_NOT_PROFITABLE",
                        "message": (
                            "TAKE_PROFIT_MARKET requires current broker unrealized P/L to be positive; "
                            f"upl={position.unrealized_pl_jpy}"
                        ),
                    }
                )
                return action
            spread_issue = _market_close_spread_issue(position, snapshot)
            if spread_issue:
                action["issues"].append(spread_issue)
                return action
            identity_issue = _broker_position_identity_issue(
                managed=managed,
                position=position,
                snapshot_fetched_at_utc=snapshot_fetched_at,
                decision_snapshot_fetched_at_utc=decision_snapshot_fetched_at_utc,
                generated_at_utc=generated_at_utc,
            )
            if identity_issue:
                action["issues"].append(identity_issue)
                return action
            action["request"] = {
                "type": "CLOSE",
                "trade_id": position.trade_id,
                "units": "ALL",
                "provenance": POSITION_PROTECTION_CLOSE_PROVENANCE,
            }
            return action
        if managed.action == ACTION_REVIEW_EXIT:
            if manual_tp_owner:
                action["issues"].append(
                    {
                        "severity": "BLOCK",
                        "code": "MANUAL_POSITION_CLOSE_FORBIDDEN",
                        "message": "manual/tagless positions are TP-managed only; loss close is forbidden",
                    }
                )
                return action
            review_exit_issue = _review_exit_gate_issue(managed)
            if review_exit_issue:
                action["issues"].append(review_exit_issue)
                return action
            spread_issue = _market_close_spread_issue(position, snapshot)
            if spread_issue:
                action["issues"].append(spread_issue)
                return action
            identity_issue = _broker_position_identity_issue(
                managed=managed,
                position=position,
                snapshot_fetched_at_utc=snapshot_fetched_at,
                decision_snapshot_fetched_at_utc=decision_snapshot_fetched_at_utc,
                generated_at_utc=generated_at_utc,
            )
            if identity_issue:
                action["issues"].append(identity_issue)
                return action
            action["request"] = {
                "type": "CLOSE",
                "trade_id": position.trade_id,
                "units": "ALL",
                "provenance": POSITION_PROTECTION_CLOSE_PROVENANCE,
            }
            return action
        # Adaptive TP actions fire a TP-only DEPENDENT_ORDER_REPLACE through the
        # same path as REPAIR/PROFIT_PROTECT (user 2026-05-08「ミクロとマクロの
        #視点」「確実に利益を取って」).
        if managed.action not in {
            ACTION_REPAIR_PROTECTION,
            ACTION_BREAK_EVEN_STOP,
            ACTION_PROFIT_PROTECT,
            ACTION_REPAIR_TAKE_PROFIT,
            ACTION_HARVEST_TP,
            ACTION_NARROW_TP,
            ACTION_EXTEND_TP,
        }:
            return action

        quote = snapshot.quotes.get(position.pair)
        order_request: dict[str, Any] = {}
        if managed.recommended_stop_loss is not None:
            if manual_tp_owner:
                action["issues"].append(
                    {
                        "severity": "BLOCK",
                        "code": "MANUAL_POSITION_STOP_LOSS_FORBIDDEN",
                        "message": "manual/tagless positions are TP-managed only; stop-loss writes are forbidden",
                    }
                )
            else:
                stop_issue = _stop_update_issue(position, float(managed.recommended_stop_loss), quote)
                if stop_issue:
                    action["issues"].append(stop_issue)
                else:
                    order_request["stopLoss"] = {
                        "timeInForce": "GTC",
                        "price": _price(position.pair, float(managed.recommended_stop_loss)),
                    }
        if managed.recommended_take_profit is not None:
            if operator_manual_tp_modify_blocked(position):
                action["issues"].append(
                    {
                        "severity": "BLOCK",
                        "code": "OPERATOR_MANUAL_TP_MODIFY_FORBIDDEN",
                        "message": (
                            "operator-manual packet sets auto_tp_modify_allowed=false; "
                            "take-profit replacement requires explicit operator authorization"
                        ),
                    }
                )
                return action
            current_tp = position.take_profit
            new_tp = float(managed.recommended_take_profit)
            tp_changed = current_tp is None or abs(new_tp - current_tp) > 1e-7
            if tp_changed:
                tp_issue = _take_profit_issue(position, new_tp, quote)
                if tp_issue:
                    action["issues"].append(tp_issue)
                else:
                    # Allow updating an existing TP, not just setting a missing one,
                    # so the trader can move TP closer to harvest as the move
                    # extends or push it out as the structure widens.
                    order_request["takeProfit"] = {
                        "timeInForce": "GTC",
                        "price": _price(position.pair, new_tp),
                    }
        if order_request:
            identity_issue = _broker_position_identity_issue(
                managed=managed,
                position=position,
                snapshot_fetched_at_utc=snapshot_fetched_at,
                decision_snapshot_fetched_at_utc=decision_snapshot_fetched_at_utc,
                generated_at_utc=generated_at_utc,
            )
            if identity_issue:
                action["issues"].append(identity_issue)
                return action
            action["request"] = {
                "type": "DEPENDENT_ORDER_REPLACE",
                "trade_id": position.trade_id,
                "order_request": order_request,
            }
        return action

    def _write_result(self, result: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, result: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Position Execution Report",
            "",
            f"- Generated at UTC: `{result['generated_at_utc']}`",
            f"- Status: `{result['status']}`",
            f"- Send requested: `{result['send_requested']}`",
            f"- Sent: `{result['sent']}`",
            "",
            "## Actions",
            "",
        ]
        actions = result.get("actions", [])
        if not actions:
            lines.append("- none")
        for action in actions:
            request = action.get("request")
            lines.append(
                f"- `{action['trade_id']}` `{action['pair']}` owner=`{action.get('owner')}` management=`{action['management_action']}` "
                f"request=`{request['type'] if request else 'none'}` sent=`{action.get('sent')}`"
            )
            if request and request["type"] == "DEPENDENT_ORDER_REPLACE":
                lines.append(f"  - order_request: `{json.dumps(request['order_request'], sort_keys=True)}`")
            for issue in action.get("issues", []):
                lines.append(f"  - `{issue['severity']}` {issue['code']}: {issue['message']}")
        lines.extend(
            [
                "",
                "## Execution Contract",
                "",
                "- Trader-owned position writes are risk-reducing only: close the trade, create missing protection, place profit-only break-even/profit-lock, tighten an existing SL, or update TP.",
                "- Manual/tagless position writes are TP-only profit management; SL writes and market closes are forbidden.",
                "- Existing SL cannot be widened. Existing TP may be moved only by TP-management actions.",
                "- Live execution requires the autotrade send path and `QR_LIVE_ENABLED=1`.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _status(*, actionable: int, blocked: int, sent: int, send: bool) -> str:
    if blocked and actionable == 0:
        return "BLOCKED"
    if actionable == 0:
        return "NO_ACTION"
    if blocked >= actionable:
        return "BLOCKED"
    if sent and blocked:
        return "PARTIAL_SENT_WITH_BLOCKS"
    if sent:
        return "SENT"
    if send and blocked:
        return "BLOCKED"
    return "STAGED"


def _position_owner_value(owner: Any) -> str:
    value = getattr(owner, "value", owner)
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {item.value for item in Owner} else Owner.UNKNOWN.value


def _duplicate_trade_ids(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        trade_id = str(value or "").strip()
        if trade_id in seen:
            duplicates.add(trade_id)
        seen.add(trade_id)
    return tuple(sorted(duplicates))


def _aware_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _broker_position_identity_issue(
    *,
    managed: ManagedPosition,
    position: BrokerPosition,
    snapshot_fetched_at_utc: datetime | None,
    decision_snapshot_fetched_at_utc: str | None,
    generated_at_utc: datetime,
) -> dict[str, str] | None:
    if (
        snapshot_fetched_at_utc is None
        or not str(position.trade_id or "").strip()
        or not str(position.pair or "").strip()
    ):
        return {
            "severity": "BLOCK",
            "code": "BROKER_SNAPSHOT_TIME_INVALID",
            "message": "broker position identity and snapshot time must be complete",
        }
    decision_snapshot_fetched_at = _aware_utc_datetime(
        decision_snapshot_fetched_at_utc
    )
    if decision_snapshot_fetched_at != snapshot_fetched_at_utc:
        return {
            "severity": "BLOCK",
            "code": "BROKER_SNAPSHOT_DECISION_MISMATCH",
            "message": (
                "position-management decision is not bound to the exact broker "
                "snapshot used for execution"
            ),
        }
    snapshot_age_seconds = (
        generated_at_utc - snapshot_fetched_at_utc
    ).total_seconds()
    if (
        snapshot_age_seconds
        < -POSITION_EXECUTION_SNAPSHOT_MAX_FUTURE_SKEW_SECONDS
        or snapshot_age_seconds > POSITION_EXECUTION_SNAPSHOT_MAX_AGE_SECONDS
    ):
        return {
            "severity": "BLOCK",
            "code": "BROKER_SNAPSHOT_STALE",
            "message": (
                "broker snapshot is outside the position execution freshness window: "
                f"age_seconds={snapshot_age_seconds:.3f} "
                f"max={POSITION_EXECUTION_SNAPSHOT_MAX_AGE_SECONDS}"
            ),
        }
    broker_pair = str(position.pair or "").strip()
    if str(managed.pair or "").strip() != broker_pair:
        return {
            "severity": "BLOCK",
            "code": "BROKER_POSITION_PAIR_MISMATCH",
            "message": (
                "managed pair contradicts the exact broker-snapshot position "
                f"identity for trade id={position.trade_id}"
            ),
        }
    return None


def _send_boundary_identity_issue(
    *,
    action: dict[str, Any],
    request: dict[str, Any],
    managed: ManagedPosition,
    position: BrokerPosition | None,
    snapshot: BrokerSnapshot,
    decision_snapshot_fetched_at_utc: str | None,
    boundary_now_utc: datetime | None,
) -> dict[str, str] | None:
    """Re-prove identity and freshness immediately before each broker POST."""

    if boundary_now_utc is None:
        return {
            "severity": "BLOCK",
            "code": "BROKER_EXECUTION_CLOCK_INVALID",
            "message": "position execution clock must be timezone-aware at send boundary",
        }
    if position is None:
        return {
            "severity": "BLOCK",
            "code": "POSITION_NOT_FOUND",
            "message": "managed position is absent at the broker send boundary",
        }
    snapshot_fetched_at = _aware_utc_datetime(snapshot.fetched_at_utc)
    issue = _broker_position_identity_issue(
        managed=managed,
        position=position,
        snapshot_fetched_at_utc=snapshot_fetched_at,
        decision_snapshot_fetched_at_utc=decision_snapshot_fetched_at_utc,
        generated_at_utc=boundary_now_utc,
    )
    if issue is not None:
        return issue

    broker_trade_id = str(position.trade_id or "").strip()
    broker_pair = str(position.pair or "").strip()
    broker_owner = _position_owner_value(position.owner)
    expected_identity = {
        "snapshot_fetched_at_utc": snapshot_fetched_at.isoformat(),
        "trade_id": broker_trade_id,
        "pair": broker_pair,
        "owner": broker_owner,
    }
    if (
        action.get("trade_id") != broker_trade_id
        or action.get("pair") != broker_pair
        or action.get("owner") != broker_owner
        or action.get("broker_position_identity") != expected_identity
        or str(request.get("trade_id") or "").strip() != broker_trade_id
    ):
        return {
            "severity": "BLOCK",
            "code": "BROKER_POSITION_IDENTITY_CHANGED",
            "message": (
                "planned action/request no longer matches the exact broker-snapshot "
                "trade, pair, and owner at the send boundary"
            ),
        }
    return None


def _send_error_code(request_type: str) -> str:
    if request_type == "CLOSE":
        return "POSITION_CLOSE_SEND_FAILED"
    if request_type == "DEPENDENT_ORDER_REPLACE":
        return "POSITION_PROTECTION_SEND_FAILED"
    return "POSITION_ACTION_SEND_FAILED"


def _close_trade_with_supported_provenance(
    client: PositionExecutionClient,
    trade_id: str,
    units: str,
    *,
    provenance: str,
) -> dict[str, Any]:
    close_with_provenance = getattr(client, "close_trade_with_provenance", None)
    class_method = getattr(type(client), "close_trade_with_provenance", None)
    if callable(close_with_provenance) and callable(class_method):
        return close_with_provenance(trade_id, units, provenance=provenance)
    raise RuntimeError(
        "position close requires close_trade_with_provenance; "
        "raw close_trade fallback is disabled by the position execution contract"
    )


def _has_block(action: dict[str, Any]) -> bool:
    return any(issue.get("severity") == "BLOCK" for issue in action.get("issues", []))


def _review_exit_gate_issue(managed: ManagedPosition) -> dict[str, str] | None:
    if not _auto_close_disabled():
        return None
    reason_text = " ".join(str(reason) for reason in managed.reasons).lower()
    if "gpt-close: accepted gpt_trader close receipt passed gate a/b" in reason_text:
        return None
    if (
        "next-generation entry thesis ledger present" in reason_text
        and "structural loss-cut remains executable" in reason_text
        and _structural_auto_close_enabled()
        and _structural_review_exit_reason(managed.reasons)
    ):
        return None
    return {
        "severity": "BLOCK",
        "code": "REVIEW_EXIT_GATE_AB_REQUIRED",
        "message": (
            "QR_DISABLE_AUTO_CLOSE=1 blocks loss-side REVIEW_EXIT unless the action is backed by "
            "an accepted gpt_trader CLOSE receipt or QR_ALLOW_STRUCTURAL_AUTO_CLOSE=1 explicit opt-in"
        ),
    }


def _auto_close_disabled() -> bool:
    return os.environ.get("QR_DISABLE_AUTO_CLOSE", "").strip().lower() in {"1", "true", "yes"}


def _structural_auto_close_enabled() -> bool:
    return os.environ.get("QR_ALLOW_STRUCTURAL_AUTO_CLOSE", "").strip().lower() in {"1", "true", "yes"}


def _structural_review_exit_reason(reasons: tuple[str, ...]) -> bool:
    for reason in reasons:
        text = str(reason)
        if not text.startswith("loss-cut:"):
            continue
        lowered = text.lower()
        if "close-confirmed structural break" in lowered:
            return True
        if "structural ob broken" in lowered:
            return True
    return False


def _position_close_spread_override_enabled() -> bool:
    return os.environ.get("QR_POSITION_CLOSE_SPREAD_OVERRIDE", "").strip().lower() in {"1", "true", "yes"}


def _snapshot_session_tag(snapshot: BrokerSnapshot) -> str | None:
    fetched_at = getattr(snapshot, "fetched_at_utc", None)
    if fetched_at is None:
        return None
    marker = tag_bar(fetched_at)
    tag = marker.tag.value if hasattr(marker.tag, "value") else str(marker.tag)
    return tag or None


def _market_close_spread_issue(position: BrokerPosition, snapshot: BrokerSnapshot) -> dict[str, str] | None:
    if _position_close_spread_override_enabled():
        return None
    quote = snapshot.quotes.get(position.pair)
    if quote is None:
        return {
            "severity": "BLOCK",
            "code": "POSITION_CLOSE_QUOTE_MISSING",
            "message": (
                f"{position.pair} quote missing from latest broker snapshot; market CLOSE spread cost cannot be "
                "bounded. Refresh broker-snapshot or set QR_POSITION_CLOSE_SPREAD_OVERRIDE=1 for explicit operator override."
            ),
        }
    normal_spread = NORMAL_SPREAD_PIPS.get(position.pair)
    if normal_spread is None:
        return {
            "severity": "BLOCK",
            "code": "POSITION_CLOSE_SPREAD_BASELINE_MISSING",
            "message": (
                f"{position.pair} normal spread baseline is missing; market CLOSE spread cost cannot be bounded. "
                "Add the broker-spec baseline or set QR_POSITION_CLOSE_SPREAD_OVERRIDE=1 for explicit operator override."
            ),
        }
    spread_pips = abs(quote.ask - quote.bid) * instrument_pip_factor(position.pair)
    max_spread_multiple = RiskPolicy().max_spread_multiple
    session_tag = _snapshot_session_tag(snapshot)
    session_mult = _spread_session_multiplier_from_tag(session_tag)
    effective_spread_cap_mult = max_spread_multiple * session_mult
    spread_cap = normal_spread * effective_spread_cap_mult
    if spread_pips > spread_cap:
        return {
            "severity": "BLOCK",
            "code": "POSITION_CLOSE_SPREAD_TOO_WIDE",
            "message": (
                f"{position.pair} market CLOSE spread {spread_pips:.1f}pip exceeds "
                f"{effective_spread_cap_mult:.2f}x normal {normal_spread:.1f}pip "
                f"(policy={max_spread_multiple:.1f}x, session={session_tag or 'UNKNOWN'}, "
                f"session_mult={session_mult:.2f}); defer to TP/SL/protection "
                "until liquidity normalizes or set QR_POSITION_CLOSE_SPREAD_OVERRIDE=1 for explicit operator override."
            ),
        }
    return None


def _stop_update_issue(position: BrokerPosition, new_stop: float, quote: Quote | None) -> dict[str, str] | None:
    if position.stop_loss is not None:
        if position.side == Side.LONG and new_stop <= position.stop_loss:
            return {
                "severity": "BLOCK",
                "code": "SL_NOT_TIGHTER",
                "message": f"LONG SL update would not tighten: current={position.stop_loss} proposed={new_stop}",
            }
        if position.side == Side.SHORT and new_stop >= position.stop_loss:
            return {
                "severity": "BLOCK",
                "code": "SL_NOT_TIGHTER",
                "message": f"SHORT SL update would not tighten: current={position.stop_loss} proposed={new_stop}",
            }
    if quote is not None:
        if position.side == Side.LONG and new_stop >= quote.bid:
            return {
                "severity": "BLOCK",
                "code": "SL_NOT_MARKET_VALID",
                "message": f"LONG SL must stay below bid: bid={quote.bid} proposed={new_stop}",
            }
        if position.side == Side.SHORT and new_stop <= quote.ask:
            return {
                "severity": "BLOCK",
                "code": "SL_NOT_MARKET_VALID",
                "message": f"SHORT SL must stay above ask: ask={quote.ask} proposed={new_stop}",
            }
    return None


def _take_profit_issue(position: BrokerPosition, take_profit: float, quote: Quote | None) -> dict[str, str] | None:
    if quote is not None:
        if position.side == Side.LONG and take_profit <= quote.ask:
            return {
                "severity": "BLOCK",
                "code": "TP_NOT_MARKET_VALID",
                "message": f"LONG TP must stay above ask: ask={quote.ask} proposed={take_profit}",
            }
        if position.side == Side.SHORT and take_profit >= quote.bid:
            return {
                "severity": "BLOCK",
                "code": "TP_NOT_MARKET_VALID",
                "message": f"SHORT TP must stay below bid: bid={quote.bid} proposed={take_profit}",
            }
    if position.side == Side.LONG and take_profit <= position.entry_price:
        return {
            "severity": "BLOCK",
            "code": "TP_NOT_REWARD_SIDE",
            "message": f"LONG TP must stay above entry: entry={position.entry_price} proposed={take_profit}",
        }
    if position.side == Side.SHORT and take_profit >= position.entry_price:
        return {
            "severity": "BLOCK",
            "code": "TP_NOT_REWARD_SIDE",
            "message": f"SHORT TP must stay below entry: entry={position.entry_price} proposed={take_profit}",
        }
    return None


def _price(pair: str, value: float) -> str:
    precision = 3 if pair.endswith("_JPY") else 5
    return f"{value:.{precision}f}"

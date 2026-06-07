from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from quant_rabbit.models import BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.paths import DEFAULT_POSITION_EXECUTION, DEFAULT_POSITION_EXECUTION_REPORT
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


class PositionExecutionClient(Protocol):
    def replace_trade_dependent_orders(self, trade_id: str, order_request: dict[str, Any]) -> dict[str, Any]: ...

    def close_trade(self, trade_id: str, units: str = "ALL") -> dict[str, Any]: ...


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
    ) -> None:
        self.client = client
        self.output_path = output_path
        self.report_path = report_path
        self.live_enabled = live_enabled

    def run(
        self,
        *,
        decision: PositionManagementDecision,
        snapshot: BrokerSnapshot,
        send: bool = False,
    ) -> PositionExecutionSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        positions = {position.trade_id: position for position in snapshot.positions}
        actions = [
            self._plan_action(managed, positions.get(managed.trade_id), snapshot)
            for managed in decision.positions
        ]
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
        for action in actions:
            request = action["request"]
            blocked = _has_block(action)
            response = None
            if send and request is not None and not blocked:
                if request["type"] == "CLOSE":
                    response = self.client.close_trade(str(request["trade_id"]), str(request.get("units") or "ALL"))
                elif request["type"] == "DEPENDENT_ORDER_REPLACE":
                    response = self.client.replace_trade_dependent_orders(
                        str(request["trade_id"]),
                        dict(request["order_request"]),
                    )
                action["sent"] = True
            responses.append(response)
            action["response"] = response

        actionable = sum(1 for action in actions if action["request"] is not None)
        blocked_count = sum(1 for action in actions if _has_block(action))
        sent_count = sum(1 for action in actions if action.get("sent"))
        status = _status(actionable=actionable, blocked=blocked_count, sent=sent_count, send=send)
        result = {
            "generated_at_utc": generated_at,
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
    ) -> dict[str, Any]:
        action: dict[str, Any] = {
            "trade_id": managed.trade_id,
            "pair": managed.pair,
            "owner": managed.owner,
            "management_action": managed.action,
            "reasons": list(managed.reasons),
            "request": None,
            "issues": [],
            "sent": False,
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
        manual_tp_owner = position.owner in {Owner.MANUAL, Owner.UNKNOWN}
        if position.owner not in {Owner.TRADER, Owner.MANUAL, Owner.UNKNOWN}:
            action["issues"].append(
                {
                    "severity": "BLOCK",
                    "code": "NON_TRADER_POSITION",
                    "message": f"refusing to modify external position id={position.trade_id}",
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
            action["request"] = {"type": "CLOSE", "trade_id": position.trade_id, "units": "ALL"}
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
            action["request"] = {"type": "CLOSE", "trade_id": position.trade_id, "units": "ALL"}
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
    ):
        return None
    return {
        "severity": "BLOCK",
        "code": "REVIEW_EXIT_GATE_AB_REQUIRED",
        "message": (
            "QR_DISABLE_AUTO_CLOSE=1 blocks loss-side REVIEW_EXIT unless the action is backed by "
            "an accepted gpt_trader CLOSE receipt or next-generation structural loss-cut evidence"
        ),
    }


def _auto_close_disabled() -> bool:
    return os.environ.get("QR_DISABLE_AUTO_CLOSE", "").strip().lower() in {"1", "true", "yes"}


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

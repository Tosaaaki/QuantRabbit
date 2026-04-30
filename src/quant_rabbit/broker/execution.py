from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from quant_rabbit.models import BrokerSnapshot, MarketContext, OrderIntent, OrderType, Owner, Quote, Side, TradeMethod
from quant_rabbit.paths import (
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_LIVE_ORDER_STAGE_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_STRATEGY_PROFILE,
)
from quant_rabbit.risk import RiskEngine, RiskIssue
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
    ) -> None:
        self.client = client
        self.strategy_profile = strategy_profile
        self.output_path = output_path
        self.report_path = report_path
        self.live_enabled = live_enabled

    def run(
        self,
        *,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        lane_id: str | None = None,
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
        snapshot = self.client.snapshot(_snapshot_pairs(intents_payload, intent))
        validate_live_enabled = self.live_enabled if send else True
        risk = RiskEngine(live_enabled=validate_live_enabled).validate(intent, snapshot, for_live_send=True)
        strategy_issues = tuple(
            issue.__dict__ for issue in StrategyProfile.load(self.strategy_profile).validate(intent, for_live_send=True)
        )
        risk_issues = [issue.__dict__ for issue in risk.issues]
        send_issues = _send_guard_issues(send=send, confirm_live=confirm_live, lane_id=lane_id)
        all_blocked = (
            any(issue["severity"] == "BLOCK" for issue in risk_issues)
            or any(issue["severity"] == "BLOCK" for issue in strategy_issues)
            or any(issue["severity"] == "BLOCK" for issue in send_issues)
        )
        order_request = _oanda_order_request(intent)
        response = None
        sent = False
        status = "BLOCKED" if all_blocked else "STAGED"
        if send and not all_blocked:
            response = self.client.post_order_json(order_request)
            sent = True
            status = "SENT"
        result = {
            "generated_at_utc": generated_at,
            "status": status,
            "lane_id": selected_lane_id,
            "order_request": order_request,
            "risk_issues": [*risk_issues, *send_issues],
            "strategy_issues": list(strategy_issues),
            "send_requested": send,
            "sent": sent,
            "response": response,
            "snapshot": {
                "fetched_at_utc": snapshot.fetched_at_utc.isoformat(),
                "positions": len(snapshot.positions),
                "orders": len(snapshot.orders),
                "quotes": len(snapshot.quotes),
            },
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
        )

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
            f"- Send requested: `{result.get('send_requested')}`",
            f"- Sent: `{result.get('sent')}`",
            "",
            "## Order Request",
            "",
        ]
        order = result.get("order_request")
        if order:
            lines.append(f"- `{order['instrument']}` `{order['type']}` units=`{order['units']}`")
            if "price" in order:
                lines.append(f"- price: `{order['price']}`")
            lines.append(f"- takeProfitOnFill: `{order['takeProfitOnFill']['price']}`")
            lines.append(f"- stopLossOnFill: `{order['stopLossOnFill']['price']}`")
        else:
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


def _send_guard_issues(*, send: bool, confirm_live: bool, lane_id: str | None) -> list[dict[str, str]]:
    issues: list[RiskIssue] = []
    if send and not confirm_live:
        issues.append(RiskIssue("LIVE_CONFIRMATION_REQUIRED", "live send requires --confirm-live"))
    if send and not lane_id:
        issues.append(RiskIssue("LANE_ID_REQUIRED_FOR_SEND", "live send requires an explicit --lane-id"))
    return [issue.__dict__ for issue in issues]


def _oanda_order_request(intent: OrderIntent) -> dict[str, Any]:
    signed_units = intent.units if intent.side == Side.LONG else -intent.units
    order_type = _oanda_order_type(intent.order_type)
    order: dict[str, Any] = {
        "type": order_type,
        "instrument": intent.pair,
        "units": str(signed_units),
        "positionFill": "DEFAULT",
        "takeProfitOnFill": {"price": _price(intent.pair, intent.tp)},
        "stopLossOnFill": {"price": _price(intent.pair, intent.sl)},
        "clientExtensions": {"tag": Owner.TRADER.value, "comment": _comment(intent)},
        "tradeClientExtensions": {"tag": Owner.TRADER.value, "comment": _comment(intent)},
    }
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


def _price(pair: str, value: float) -> str:
    precision = 3 if pair.endswith("_JPY") else 5
    return f"{value:.{precision}f}"


def _comment(intent: OrderIntent) -> str:
    desk = str(intent.metadata.get("desk") or "vnext")
    role = str(intent.metadata.get("campaign_role") or "")
    text = f"qr-vnext {desk} {role}".strip()
    return text[:128]


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

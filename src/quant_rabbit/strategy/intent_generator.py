from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.models import BrokerOrder, BrokerPosition, BrokerSnapshot, MarketContext, OrderIntent, OrderType, Owner, Quote, Side, TradeMethod
from quant_rabbit.paths import (
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_ORDER_INTENT_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_STRATEGY_PROFILE,
)
from quant_rabbit.risk import RiskEngine
from quant_rabbit.strategy.profile import StrategyProfile


PIP_FACTORS = {
    "USD_JPY": 100,
    "EUR_JPY": 100,
    "GBP_JPY": 100,
    "AUD_JPY": 100,
    "EUR_USD": 10000,
    "GBP_USD": 10000,
    "AUD_USD": 10000,
}


@dataclass(frozen=True)
class GeneratedIntent:
    lane_id: str
    status: str
    intent: dict[str, Any] | None
    risk_allowed: bool | None
    risk_issues: tuple[dict[str, Any], ...]
    strategy_issues: tuple[dict[str, Any], ...]
    live_blockers: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class IntentGenerationSummary:
    output_path: Path
    report_path: Path
    candidates_seen: int
    generated: int
    needs_snapshot: int
    dry_run_passed: int
    live_ready: int


class IntentGenerator:
    def __init__(
        self,
        *,
        campaign_plan: Path = DEFAULT_CAMPAIGN_PLAN,
        strategy_profile: Path = DEFAULT_STRATEGY_PROFILE,
        output_path: Path = DEFAULT_ORDER_INTENTS,
        report_path: Path = DEFAULT_ORDER_INTENT_REPORT,
    ) -> None:
        self.campaign_plan = campaign_plan
        self.strategy_profile = strategy_profile
        self.output_path = output_path
        self.report_path = report_path

    def run(self, *, snapshot_path: Path | None = None, max_candidates: int = 12) -> IntentGenerationSummary:
        plan = json.loads(self.campaign_plan.read_text())
        lanes = [lane for lane in plan.get("lanes", []) if _lane_can_attempt(lane)]
        snapshot = _snapshot_from_json(json.loads(snapshot_path.read_text())) if snapshot_path else None
        strategy_profile = StrategyProfile.load(self.strategy_profile) if self.strategy_profile.exists() else None
        results: list[GeneratedIntent] = []
        for lane in lanes[:max_candidates]:
            results.append(self._build_for_lane(lane, snapshot, strategy_profile))
        generated_at = datetime.now(timezone.utc).isoformat()
        self._write_output(results, generated_at, snapshot_path)
        self._write_report(results, generated_at, snapshot_path)
        return IntentGenerationSummary(
            output_path=self.output_path,
            report_path=self.report_path,
            candidates_seen=len(lanes),
            generated=sum(1 for item in results if item.intent is not None),
            needs_snapshot=sum(1 for item in results if item.status == "NEEDS_BROKER_SNAPSHOT"),
            dry_run_passed=sum(1 for item in results if item.status == "DRY_RUN_PASSED"),
            live_ready=sum(1 for item in results if item.status == "LIVE_READY"),
        )

    def _build_for_lane(
        self,
        lane: dict[str, Any],
        snapshot: BrokerSnapshot | None,
        strategy_profile: StrategyProfile | None,
    ) -> GeneratedIntent:
        lane_id = _lane_id(lane)
        pair = str(lane["pair"])
        direction = str(lane["direction"])
        if snapshot is None:
            return GeneratedIntent(
                lane_id=lane_id,
                status="NEEDS_BROKER_SNAPSHOT",
                intent=None,
                risk_allowed=None,
                risk_issues=(),
                strategy_issues=(),
                live_blockers=("broker snapshot is required to price entry/TP/SL",),
                note="Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.",
            )
        quote = snapshot.quotes.get(pair)
        if quote is None:
            return GeneratedIntent(
                lane_id=lane_id,
                status="MISSING_QUOTE",
                intent=None,
                risk_allowed=False,
                risk_issues=({"code": "MISSING_QUOTE", "message": f"missing quote for {pair}", "severity": "BLOCK"},),
                strategy_issues=(),
                live_blockers=(f"snapshot has no quote for {pair}",),
                note="Cannot build priced intent without a live quote.",
            )
        intent = _intent_from_lane(lane, quote)
        risk = RiskEngine().validate(intent, snapshot, for_live_send=False)
        strategy_issues = tuple(
            issue.__dict__ for issue in (strategy_profile.validate(intent, for_live_send=False) if strategy_profile else ())
        )
        live_strategy_issues = tuple(
            issue.__dict__ for issue in (strategy_profile.validate(intent, for_live_send=True) if strategy_profile else ())
        )
        live_blockers = tuple(issue["message"] for issue in live_strategy_issues if issue.get("severity") == "BLOCK")
        risk_issues = tuple(issue.__dict__ for issue in risk.issues)
        if risk.allowed and not live_blockers:
            status = "LIVE_READY"
        elif risk.allowed:
            status = "DRY_RUN_PASSED"
        else:
            status = "DRY_RUN_BLOCKED"
        return GeneratedIntent(
            lane_id=lane_id,
            status=status,
            intent=_intent_to_json(intent),
            risk_allowed=risk.allowed,
            risk_issues=risk_issues,
            strategy_issues=strategy_issues,
            live_blockers=live_blockers,
            note="Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.",
        )

    def _write_output(
        self,
        results: list[GeneratedIntent],
        generated_at: str,
        snapshot_path: Path | None,
    ) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": generated_at,
            "campaign_plan": str(self.campaign_plan),
            "strategy_profile": str(self.strategy_profile),
            "snapshot_path": str(snapshot_path) if snapshot_path else None,
            "results": [asdict(item) for item in results],
        }
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(
        self,
        results: list[GeneratedIntent],
        generated_at: str,
        snapshot_path: Path | None,
    ) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Order Intents Report",
            "",
            f"- Generated at UTC: `{generated_at}`",
            f"- Campaign plan: `{self.campaign_plan}`",
            f"- Snapshot: `{snapshot_path if snapshot_path else 'none'}`",
            f"- Results: `{len(results)}`",
            "",
            "## Status Counts",
            "",
        ]
        counts: dict[str, int] = {}
        for item in results:
            counts[item.status] = counts.get(item.status, 0) + 1
        for status, count in sorted(counts.items()):
            lines.append(f"- `{status}`: `{count}`")
        lines.extend(["", "## Candidates", ""])
        for item in results:
            lines.append(f"- `{item.lane_id}` status=`{item.status}`")
            lines.append(f"  - note: {item.note}")
            if item.intent:
                intent = item.intent
                lines.append(
                    f"  - intent: `{intent['pair']} {intent['side']} {intent['order_type']}` "
                    f"units={intent['units']} entry={intent.get('entry')} tp={intent['tp']} sl={intent['sl']}"
                )
            for issue in item.risk_issues:
                lines.append(f"  - risk {issue['severity']}: {issue['code']} {issue['message']}")
            for issue in item.strategy_issues:
                lines.append(f"  - strategy {issue['severity']}: {issue['code']} {issue['message']}")
            for blocker in item.live_blockers:
                lines.append(f"  - live blocker: {blocker}")
        lines.extend(
            [
                "",
                "## Completion Rule",
                "",
                "- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.",
                "- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.",
                "- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _lane_can_attempt(lane: dict[str, Any]) -> bool:
    return lane.get("adoption") in {"ORDER_INTENT_REQUIRED", "RISK_REPAIR_DRY_RUN", "TRIGGER_RECEIPT_REQUIRED"} and lane.get(
        "direction"
    ) in {"LONG", "SHORT"}


def _lane_id(lane: dict[str, Any]) -> str:
    return f"{lane.get('desk')}:{lane.get('pair')}:{lane.get('direction')}:{lane.get('method')}"


def _intent_from_lane(lane: dict[str, Any], quote: Quote) -> OrderIntent:
    pair = str(lane["pair"])
    side = Side.parse(str(lane["direction"]))
    method = TradeMethod.parse(str(lane["method"]))
    order_type = _order_type_for(method)
    entry, tp, sl = _geometry(pair, side, order_type, quote)
    thesis = f"{lane['desk']} {pair} {side.value} {method.value}: {lane['required_receipt']}"
    context = MarketContext(
        regime=f"{method.value} campaign lane",
        narrative=str(lane.get("reason") or ""),
        chart_story=" | ".join(str(item) for item in lane.get("story_examples", [])[:2]) or "campaign lane requires current chart read",
        method=method,
        invalidation=f"invalid if SL {sl} trades or campaign overlay vetoes the setup",
        event_risk="; ".join(str(item) for item in lane.get("blockers", [])[:2]),
        session="generated dry-run",
    )
    return OrderIntent(
        pair=pair,
        side=side,
        order_type=order_type,
        units=1000,
        entry=entry,
        tp=tp,
        sl=sl,
        thesis=thesis,
        owner=Owner.TRADER,
        market_context=context,
        metadata={
            "desk": lane.get("desk"),
            "adoption": lane.get("adoption"),
            "campaign_role": lane.get("campaign_role"),
            "required_receipt": lane.get("required_receipt"),
        },
    )


def _order_type_for(method: TradeMethod) -> OrderType:
    if method == TradeMethod.RANGE_ROTATION:
        return OrderType.LIMIT
    return OrderType.STOP_ENTRY


def _geometry(pair: str, side: Side, order_type: OrderType, quote: Quote) -> tuple[float, float, float]:
    pip_factor = PIP_FACTORS[pair]
    pip = 1.0 / pip_factor
    spread_pips = abs(quote.ask - quote.bid) * pip_factor
    stop_pips = max(spread_pips * 6.0, 8.0)
    reward_pips = stop_pips * 1.5
    trigger_offset_pips = max(spread_pips * 2.0, 2.0)
    if order_type == OrderType.LIMIT:
        entry = quote.bid - trigger_offset_pips * pip if side == Side.LONG else quote.ask + trigger_offset_pips * pip
    else:
        entry = quote.ask + trigger_offset_pips * pip if side == Side.LONG else quote.bid - trigger_offset_pips * pip
    if side == Side.LONG:
        tp = entry + reward_pips * pip
        sl = entry - stop_pips * pip
    else:
        tp = entry - reward_pips * pip
        sl = entry + stop_pips * pip
    return _round_price(pair, entry), _round_price(pair, tp), _round_price(pair, sl)


def _round_price(pair: str, value: float) -> float:
    return round(value, 3 if pair.endswith("_JPY") else 5)


def _intent_to_json(intent: OrderIntent) -> dict[str, Any]:
    return {
        "pair": intent.pair,
        "side": intent.side.value,
        "order_type": intent.order_type.value,
        "units": intent.units,
        "entry": intent.entry,
        "tp": intent.tp,
        "sl": intent.sl,
        "thesis": intent.thesis,
        "owner": intent.owner.value,
        "market_context": {
            "regime": intent.market_context.regime if intent.market_context else "",
            "narrative": intent.market_context.narrative if intent.market_context else "",
            "chart_story": intent.market_context.chart_story if intent.market_context else "",
            "method": intent.market_context.method.value if intent.market_context else "",
            "invalidation": intent.market_context.invalidation if intent.market_context else "",
            "event_risk": intent.market_context.event_risk if intent.market_context else "",
            "session": intent.market_context.session if intent.market_context else "",
        },
        "metadata": intent.metadata,
    }


def _snapshot_from_json(payload: dict[str, Any]) -> BrokerSnapshot:
    positions = tuple(
        BrokerPosition(
            trade_id=str(item["trade_id"]),
            pair=str(item["pair"]),
            side=Side.parse(str(item["side"])),
            units=int(item["units"]),
            entry_price=float(item["entry_price"]),
            unrealized_pl_jpy=float(item.get("unrealized_pl_jpy") or 0.0),
            take_profit=float(item["take_profit"]) if item.get("take_profit") is not None else None,
            stop_loss=float(item["stop_loss"]) if item.get("stop_loss") is not None else None,
            owner=Owner(str(item.get("owner") or Owner.UNKNOWN.value)),
        )
        for item in payload.get("positions", []) or []
    )
    orders = tuple(
        BrokerOrder(
            order_id=str(item["order_id"]),
            pair=item.get("pair"),
            order_type=str(item.get("order_type") or ""),
            trade_id=item.get("trade_id"),
            price=float(item["price"]) if item.get("price") is not None else None,
            state=item.get("state"),
            units=int(item["units"]) if item.get("units") is not None else None,
        )
        for item in payload.get("orders", []) or []
    )
    quotes = {}
    for pair, item in (payload.get("quotes") or {}).items():
        timestamp = item.get("timestamp_utc")
        quotes[pair] = Quote(
            pair=pair,
            bid=float(item["bid"]),
            ask=float(item["ask"]),
            timestamp_utc=datetime.fromisoformat(timestamp) if timestamp else datetime.now(timezone.utc),
        )
    fetched = payload.get("fetched_at_utc")
    return BrokerSnapshot(
        fetched_at_utc=datetime.fromisoformat(fetched) if fetched else datetime.now(timezone.utc),
        positions=positions,
        orders=orders,
        quotes=quotes,
    )

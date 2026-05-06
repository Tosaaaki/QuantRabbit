from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from quant_rabbit.paths import (
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_GPT_TRADER_DECISION_REPORT,
    DEFAULT_MARKET_STORY_PROFILE,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_STRATEGY_PROFILE,
)


ALLOWED_ACTIONS = ("TRADE", "WAIT", "CANCEL_PENDING", "PROTECT", "TIGHTEN_SL", "CLOSE", "REQUEST_EVIDENCE")
ALLOWED_CONFIDENCE = ("LOW", "MEDIUM", "HIGH")
ALLOWED_METHODS = ("TREND_CONTINUATION", "RANGE_ROTATION", "BREAKOUT_FAILURE", "EVENT_RISK", "POSITION_MANAGEMENT")
# Matches the CLI generate-intents breadth used by the scheduled trader. The
# verifier also keeps every LIVE_READY lane even when a smaller cap is passed,
# because the operator may cite any executable lane visible in order_intents.
DEFAULT_GPT_MAX_LANES = 56


@dataclass(frozen=True)
class GPTTraderDecision:
    action: str
    selected_lane_id: str | None
    cancel_order_ids: tuple[str, ...]
    confidence: str
    thesis: str
    method: str
    narrative: str
    chart_story: str
    invalidation: str
    rejected_alternatives: tuple[str, ...]
    risk_notes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    operator_summary: str


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    message: str
    severity: str = "BLOCK"


@dataclass(frozen=True)
class VerificationResult:
    allowed: bool
    issues: tuple[VerificationIssue, ...]


@dataclass(frozen=True)
class GPTTraderSummary:
    status: str
    output_path: Path
    report_path: Path
    action: str | None
    selected_lane_id: str | None
    allowed: bool
    issues: int


class TraderModelProvider(Protocol):
    def decide(self, input_packet: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]: ...


class StaticTraderProvider:
    def __init__(self, decision: dict[str, Any]) -> None:
        self.decision = decision

    def decide(self, input_packet: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return dict(self.decision)


class GPTTraderBrain:
    """Build a broker-truth packet and verify a Codex-created decision receipt."""

    def __init__(
        self,
        *,
        provider: TraderModelProvider | None = None,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        campaign_plan_path: Path = DEFAULT_CAMPAIGN_PLAN,
        strategy_profile_path: Path = DEFAULT_STRATEGY_PROFILE,
        market_story_profile_path: Path = DEFAULT_MARKET_STORY_PROFILE,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        output_path: Path = DEFAULT_GPT_TRADER_DECISION,
        report_path: Path = DEFAULT_GPT_TRADER_DECISION_REPORT,
        max_lanes: int = DEFAULT_GPT_MAX_LANES,
    ) -> None:
        self.provider = provider
        self.intents_path = intents_path
        self.campaign_plan_path = campaign_plan_path
        self.strategy_profile_path = strategy_profile_path
        self.market_story_profile_path = market_story_profile_path
        self.target_state_path = target_state_path
        self.output_path = output_path
        self.report_path = report_path
        self.max_lanes = max_lanes

    def run(self, *, snapshot_path: Path) -> GPTTraderSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        packet = self._input_packet(snapshot_path)
        if self.provider is None:
            raise RuntimeError("Codex GPT verifier requires a decision response JSON")
        raw_decision = self.provider.decide(packet, GPT_TRADER_SCHEMA)
        decision = _decision_from_payload(raw_decision)
        verification = DecisionVerifier(packet).verify(decision)
        status = "ACCEPTED" if verification.allowed else "REJECTED"
        result = {
            "generated_at_utc": generated_at,
            "status": status,
            "decision": asdict(decision),
            "verification_issues": [asdict(issue) for issue in verification.issues],
            "input_packet": packet,
        }
        self._write_result(result)
        self._write_report(result)
        return GPTTraderSummary(
            status=status,
            output_path=self.output_path,
            report_path=self.report_path,
            action=decision.action,
            selected_lane_id=decision.selected_lane_id,
            allowed=verification.allowed,
            issues=len(verification.issues),
        )

    def _input_packet(self, snapshot_path: Path) -> dict[str, Any]:
        snapshot = _load_json(snapshot_path)
        intents = _load_json(self.intents_path)
        campaign = _load_json(self.campaign_plan_path)
        strategy = _load_json(self.strategy_profile_path)
        story = _load_json(self.market_story_profile_path)
        target = _load_json(self.target_state_path) if self.target_state_path.exists() else {}
        lanes = _lane_packet(intents, campaign, strategy, story, max_lanes=self.max_lanes)
        refs = _allowed_refs(snapshot=snapshot, target=target, lanes=lanes)
        return {
            "contract": {
                "allowed_actions": list(ALLOWED_ACTIONS),
                "trade_requires_live_ready_lane": True,
                "pending_entry_blocks_new_trade": True,
                "protected_trader_position_adds_require_portfolio_validation": True,
                "model_output_is_advisory_until_verified": True,
            },
            "broker_snapshot": _snapshot_packet(snapshot),
            "daily_target": _target_packet(target),
            "lanes": lanes,
            "allowed_evidence_refs": refs,
        }

    def _write_result(self, result: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, result: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        decision = result["decision"]
        lines = [
            "# GPT Trader Decision Report",
            "",
            f"- Generated at UTC: `{result['generated_at_utc']}`",
            f"- Status: `{result['status']}`",
            f"- Action: `{decision.get('action')}`",
            f"- Selected lane: `{decision.get('selected_lane_id')}`",
            f"- Cancel order ids: `{', '.join(decision.get('cancel_order_ids') or []) or 'none'}`",
            f"- Confidence: `{decision.get('confidence')}`",
            f"- Operator summary: {decision.get('operator_summary')}",
            "",
            "## Verification Issues",
            "",
        ]
        issues = result.get("verification_issues", [])
        if issues:
            for issue in issues:
                lines.append(f"- `{issue['severity']}` {issue['code']}: {issue['message']}")
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "## Decision Contract",
                "",
                "- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.",
                "- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.",
                "- Evidence refs must come from the input packet; invented refs reject the decision.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


class DecisionVerifier:
    def __init__(self, input_packet: dict[str, Any]) -> None:
        self.packet = input_packet
        self.lanes = {str(lane["lane_id"]): lane for lane in input_packet.get("lanes", [])}
        self.allowed_refs = set(str(ref) for ref in input_packet.get("allowed_evidence_refs", []))

    def verify(self, decision: GPTTraderDecision) -> VerificationResult:
        issues: list[VerificationIssue] = []
        if decision.action not in ALLOWED_ACTIONS:
            issues.append(VerificationIssue("BAD_ACTION", f"unsupported action {decision.action!r}"))
        if decision.confidence not in ALLOWED_CONFIDENCE:
            issues.append(VerificationIssue("BAD_CONFIDENCE", f"unsupported confidence {decision.confidence!r}"))
        if decision.method not in ALLOWED_METHODS:
            issues.append(VerificationIssue("BAD_METHOD", f"unsupported method {decision.method!r}"))
        if not decision.evidence_refs:
            issues.append(VerificationIssue("MISSING_EVIDENCE_REFS", "decision must cite packet evidence refs"))
        unknown_refs = sorted(set(decision.evidence_refs) - self.allowed_refs)
        if unknown_refs:
            issues.append(VerificationIssue("UNKNOWN_EVIDENCE_REF", f"unknown evidence refs: {', '.join(unknown_refs)}"))

        broker = self.packet.get("broker_snapshot", {})
        positions = int(broker.get("positions") or 0)
        lane = self.lanes.get(decision.selected_lane_id or "")
        tradeable_lanes = _tradeable_live_ready_lanes(self.packet)
        exposure_blockers = _trade_exposure_blockers(self.packet)

        if decision.action == "TRADE":
            if not decision.selected_lane_id:
                issues.append(VerificationIssue("LANE_REQUIRED", "TRADE requires selected_lane_id"))
            if exposure_blockers:
                issues.append(VerificationIssue("EXPOSURE_BLOCKS_TRADE", "; ".join(exposure_blockers[:3])))
            if lane is None:
                issues.append(VerificationIssue("UNKNOWN_LANE", f"selected lane is not in packet: {decision.selected_lane_id}"))
            else:
                if lane.get("status") != "LIVE_READY":
                    issues.append(VerificationIssue("LANE_NOT_LIVE_READY", f"lane status is {lane.get('status')}"))
                if lane.get("method") != decision.method:
                    issues.append(VerificationIssue("METHOD_MISMATCH", "decision method does not match selected lane"))
                if lane.get("risk_blockers") or lane.get("strategy_blockers") or lane.get("live_blockers"):
                    issues.append(VerificationIssue("LANE_HAS_BLOCKERS", "selected lane still carries blockers"))
            for field_name, value in (
                ("thesis", decision.thesis),
                ("narrative", decision.narrative),
                ("chart_story", decision.chart_story),
                ("invalidation", decision.invalidation),
            ):
                if not value.strip():
                    issues.append(VerificationIssue("INCOMPLETE_TRADE_DECISION", f"TRADE missing {field_name}"))
        elif decision.action in {"WAIT", "REQUEST_EVIDENCE"}:
            if decision.selected_lane_id is not None:
                issues.append(VerificationIssue("WAIT_SELECTED_LANE", f"{decision.action} must not select a lane"))
            if _target_requires_entry(self.packet) and not exposure_blockers and tradeable_lanes:
                cited_live_ready = _cited_live_ready_lanes(decision, tradeable_lanes)
                if decision.action == "REQUEST_EVIDENCE":
                    issues.append(
                        VerificationIssue(
                            "REQUEST_EVIDENCE_WITH_LIVE_READY_LANES",
                            "REQUEST_EVIDENCE is stale or contradictory because the packet already contains "
                            f"tradeable LIVE_READY lanes: {', '.join(tradeable_lanes[:3])}",
                        )
                    )
                elif not cited_live_ready:
                    issues.append(
                        VerificationIssue(
                            "WAIT_MISSING_LIVE_READY_REJECTION",
                            "WAIT must cite at least one current LIVE_READY lane evidence ref when clean "
                            "tradeable lanes exist and the daily target is still open",
                        )
                    )
        elif decision.action == "CANCEL_PENDING":
            pending_order_ids = set(_pending_entry_order_ids(self.packet))
            if decision.selected_lane_id is not None:
                issues.append(VerificationIssue("CANCEL_SELECTED_LANE", "CANCEL_PENDING must not select a trade lane"))
            if not pending_order_ids:
                issues.append(VerificationIssue("NO_PENDING_ENTRY", "CANCEL_PENDING requires a pending entry order"))
            if not decision.cancel_order_ids:
                issues.append(
                    VerificationIssue(
                        "MISSING_CANCEL_ORDER_IDS",
                        "CANCEL_PENDING must name the pending entry order ids to cancel",
                    )
                )
            unknown_cancel_ids = sorted(set(decision.cancel_order_ids) - pending_order_ids)
            if unknown_cancel_ids:
                issues.append(
                    VerificationIssue(
                        "UNKNOWN_CANCEL_ORDER_ID",
                        "cancel_order_ids must match current pending entry orders: "
                        + ", ".join(unknown_cancel_ids),
                    )
                )
        elif decision.action in {"PROTECT", "TIGHTEN_SL", "CLOSE"}:
            if positions <= 0:
                issues.append(VerificationIssue("NO_OPEN_POSITION", f"{decision.action} requires an open position"))

        return VerificationResult(allowed=not any(issue.severity == "BLOCK" for issue in issues), issues=tuple(issues))


GPT_TRADER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action",
        "selected_lane_id",
        "confidence",
        "thesis",
        "method",
        "narrative",
        "chart_story",
        "invalidation",
        "rejected_alternatives",
        "risk_notes",
        "evidence_refs",
        "operator_summary",
    ],
    "properties": {
        "action": {"type": "string", "enum": list(ALLOWED_ACTIONS)},
        "selected_lane_id": {"type": ["string", "null"]},
        "cancel_order_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": list(ALLOWED_CONFIDENCE)},
        "thesis": {"type": "string"},
        "method": {"type": "string", "enum": list(ALLOWED_METHODS)},
        "narrative": {"type": "string"},
        "chart_story": {"type": "string"},
        "invalidation": {"type": "string"},
        "rejected_alternatives": {"type": "array", "items": {"type": "string"}},
        "risk_notes": {"type": "array", "items": {"type": "string"}},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "operator_summary": {"type": "string"},
    },
}


def _decision_from_payload(payload: dict[str, Any]) -> GPTTraderDecision:
    selected_lane_id = payload.get("selected_lane_id")
    return GPTTraderDecision(
        action=str(payload.get("action") or ""),
        selected_lane_id=str(selected_lane_id) if selected_lane_id is not None else None,
        cancel_order_ids=tuple(str(item) for item in payload.get("cancel_order_ids", []) or []),
        confidence=str(payload.get("confidence") or ""),
        thesis=str(payload.get("thesis") or ""),
        method=str(payload.get("method") or ""),
        narrative=str(payload.get("narrative") or ""),
        chart_story=str(payload.get("chart_story") or ""),
        invalidation=str(payload.get("invalidation") or ""),
        rejected_alternatives=tuple(str(item) for item in payload.get("rejected_alternatives", []) or []),
        risk_notes=tuple(str(item) for item in payload.get("risk_notes", []) or []),
        evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", []) or []),
        operator_summary=str(payload.get("operator_summary") or ""),
    )


def _lane_packet(
    intents: dict[str, Any],
    campaign: dict[str, Any],
    strategy: dict[str, Any],
    story: dict[str, Any],
    *,
    max_lanes: int,
) -> list[dict[str, Any]]:
    campaign_index = {f"{lane.get('desk')}:{lane.get('pair')}:{lane.get('direction')}:{lane.get('method')}": lane for lane in campaign.get("lanes", []) or []}
    strategy_index = {(item.get("pair"), item.get("direction")): item for item in strategy.get("profiles", []) or []}
    story_index = {item.get("pair"): item for item in story.get("pair_profiles", []) or []}
    lanes: list[dict[str, Any]] = []
    for result in intents.get("results", []) or []:
        if not isinstance(result, dict) or not isinstance(result.get("intent"), dict):
            continue
        intent = result["intent"]
        lane_id = str(result.get("lane_id") or "")
        pair = str(intent.get("pair") or "")
        direction = str(intent.get("side") or "")
        context = intent.get("market_context") or {}
        risk_blockers = _block_issues(result.get("risk_issues"))
        strategy_blockers = _block_issues(result.get("strategy_issues"))
        lanes.append(
            {
                "lane_id": lane_id,
                "evidence_ref": f"intent:{lane_id}",
                "status": result.get("status"),
                "pair": pair,
                "direction": direction,
                "method": context.get("method") or "",
                "order_type": intent.get("order_type"),
                "entry": intent.get("entry"),
                "tp": intent.get("tp"),
                "sl": intent.get("sl"),
                "units": intent.get("units"),
                "risk_metrics": _small_dict(
                    result.get("risk_metrics"),
                    ("entry_price", "loss_pips", "reward_pips", "risk_jpy", "reward_jpy", "reward_risk", "spread_pips", "jpy_per_pip"),
                ),
                "thesis": intent.get("thesis"),
                "narrative": context.get("narrative") or "",
                "chart_story": context.get("chart_story") or "",
                "invalidation": context.get("invalidation") or "",
                "risk_blockers": risk_blockers,
                "strategy_blockers": strategy_blockers,
                "live_blockers": list(result.get("live_blockers", []) or []),
                "campaign": _small_dict(campaign_index.get(lane_id), ("adoption", "campaign_role", "required_receipt")),
                "strategy": _small_dict(strategy_index.get((pair, direction)), ("status", "pretrade_net_jpy", "live_net_jpy", "live_worst_jpy", "required_fix")),
                "story": _small_dict(story_index.get(pair), ("methods", "themes", "examples")),
            }
        )
    if max_lanes <= 0 or len(lanes) <= max_lanes:
        return lanes
    capped = lanes[:max_lanes]
    capped_ids = {str(lane.get("lane_id") or "") for lane in capped}
    for lane in lanes[max_lanes:]:
        lane_id = str(lane.get("lane_id") or "")
        if lane_id and lane_id not in capped_ids and lane.get("status") == "LIVE_READY":
            capped.append(lane)
            capped_ids.add(lane_id)
    return capped


def _snapshot_packet(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_ref": "broker:snapshot",
        "fetched_at_utc": snapshot.get("fetched_at_utc"),
        "positions": len(snapshot.get("positions", []) or []),
        "orders": len(snapshot.get("orders", []) or []),
        "position_summaries": [
            {
                "trade_id": item.get("trade_id"),
                "pair": item.get("pair"),
                "side": item.get("side"),
                "units": item.get("units"),
                "take_profit": item.get("take_profit"),
                "stop_loss": item.get("stop_loss"),
                "owner": item.get("owner"),
            }
            for item in (snapshot.get("positions", []) or [])[:5]
        ],
        "pending_orders": [
            {
                "order_id": item.get("order_id"),
                "pair": item.get("pair"),
                "order_type": item.get("order_type"),
                "trade_id": item.get("trade_id"),
                "price": item.get("price"),
                "units": item.get("units"),
                "owner": item.get("owner"),
            }
            for item in (snapshot.get("orders", []) or [])[:5]
        ],
    }


def _target_packet(target: dict[str, Any]) -> dict[str, Any]:
    if not target:
        return {"evidence_ref": "target:daily", "status": "missing"}
    return {
        "evidence_ref": "target:daily",
        "status": target.get("status"),
        "target_jpy": target.get("target_jpy"),
        "progress_jpy": target.get("progress_jpy"),
        "remaining_target_jpy": target.get("remaining_target_jpy"),
        "remaining_risk_budget_jpy": target.get("remaining_risk_budget_jpy"),
    }


def _allowed_refs(*, snapshot: dict[str, Any], target: dict[str, Any], lanes: list[dict[str, Any]]) -> list[str]:
    # Per docs/SKILL_trader.md the playbook prescribes a richer set of evidence
    # refs than the base broker/target/lane triple — the trader is required to
    # cite per-pair charts, cross-asset, flow, levels, currency strength,
    # economic calendar, and COT data. The verifier therefore must accept these
    # refs as known; otherwise every well-formed decision is rejected with
    # UNKNOWN_EVIDENCE_REF and the cycle never reaches the gateway.
    timeframes = ("M5", "M15", "H1")
    structure_keys = ("structure",)
    cross_assets = ("dxy", "USB10Y_USD", "USB02Y_USD", "spx", "gold", "oil", "btc")
    refs = ["broker:snapshot", "target:daily"]
    pairs: set[str] = set()
    currencies: set[str] = set()
    for lane in lanes:
        lane_id = lane["lane_id"]
        pair = str(lane.get("pair") or "")
        direction = str(lane.get("direction") or "")
        if pair:
            pairs.add(pair)
            for currency in pair.split("_"):
                if currency:
                    currencies.add(currency)
        refs.extend(
            [
                str(lane["evidence_ref"]),
                f"campaign:{lane_id}",
                f"strategy:{pair}:{direction}",
                f"story:{pair}",
                f"intent:{lane_id}",
            ]
        )
    for pair in pairs:
        for tf in timeframes:
            refs.append(f"chart:{pair}:{tf}")
        for key in structure_keys:
            refs.append(f"chart:{pair}:{key}")
        refs.extend(
            [
                f"flow:{pair}",
                f"levels:{pair}",
                f"calendar:{pair}",
                f"strength:{pair}",
                f"cross:correlations:{pair}",
            ]
        )
    for currency in currencies:
        refs.append(f"cot:{currency}")
        refs.append(f"strength:{currency}")
        refs.append(f"calendar:{currency}")
    for asset in cross_assets:
        refs.append(f"cross:{asset}")
    refs.extend(["cross:dxy", "cross:correlations"])
    return sorted(set(refs))


def _block_issues(items: object) -> list[str]:
    blockers: list[str] = []
    for item in items or []:
        if isinstance(item, dict) and item.get("severity") == "BLOCK":
            blockers.append(str(item.get("message") or item.get("code") or "block"))
    return blockers


def _small_dict(payload: object, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {key: payload.get(key) for key in keys if key in payload}


def _has_pending_entry_order(packet: dict[str, Any]) -> bool:
    return bool(_pending_entry_order_ids(packet))


def _pending_entry_order_ids(packet: dict[str, Any]) -> list[str]:
    snapshot = packet.get("broker_snapshot", {})
    order_ids: list[str] = []
    for order in snapshot.get("pending_orders", []) or []:
        if order.get("trade_id"):
            continue
        order_type = str(order.get("order_type") or "").upper()
        order_id = str(order.get("order_id") or "")
        if order_id and order_type in {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}:
            order_ids.append(order_id)
    return order_ids


def _trade_exposure_blockers(packet: dict[str, Any]) -> list[str]:
    snapshot = packet.get("broker_snapshot", {})
    blockers: list[str] = []
    for position in snapshot.get("position_summaries", []) or []:
        owner = str(position.get("owner") or "")
        if owner in {"manual", "unknown"}:
            continue
        if (
            owner == "trader"
            and position.get("take_profit") is not None
            and position.get("stop_loss") is not None
        ):
            continue
        blockers.append(
            f"non-layerable position {position.get('pair')} {position.get('side')} id={position.get('trade_id')}"
        )
    if _has_pending_entry_order(packet):
        blockers.append("pending entry order is open")
    return blockers


def _target_requires_entry(packet: dict[str, Any]) -> bool:
    target = packet.get("daily_target", {})
    remaining = target.get("remaining_target_jpy")
    try:
        return float(remaining or 0.0) > 0.0 and target.get("status") != "TARGET_REACHED_PROTECT"
    except (TypeError, ValueError):
        return False


def _tradeable_live_ready_lanes(packet: dict[str, Any]) -> list[str]:
    lanes: list[str] = []
    for lane in packet.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        if lane.get("status") != "LIVE_READY":
            continue
        if lane.get("risk_blockers") or lane.get("strategy_blockers") or lane.get("live_blockers"):
            continue
        lane_id = str(lane.get("lane_id") or "")
        if lane_id:
            lanes.append(lane_id)
    return lanes


def _cited_live_ready_lanes(decision: GPTTraderDecision, lane_ids: list[str]) -> list[str]:
    refs = set(decision.evidence_refs)
    return [lane_id for lane_id in lane_ids if f"intent:{lane_id}" in refs]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())

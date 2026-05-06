from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.models import BrokerSnapshot, Owner, Side, TradeMethod
from quant_rabbit.paths import (
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_MARKET_STORY_PROFILE,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_TRADER_SETTINGS,
    DEFAULT_STRATEGY_PROFILE,
    DEFAULT_TRADER_DECISION,
    DEFAULT_TRADER_DECISION_REPORT,
)


JPY_CROSSES = {"AUD_JPY", "EUR_JPY", "GBP_JPY", "USD_JPY"}
PENDING_ENTRY_TYPES = {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}
ACTION_SEND_ENTRY = "SEND_ENTRY"
ACTION_MONITOR_EXISTING = "MONITOR_EXISTING_EXPOSURE"
ACTION_NO_TRADE = "NO_TRADE"

# A historical worst loss becomes "large" only after it exceeds 1.8x the
# current per-trade cap. This preserves the old proportional warning behavior
# (-900 JPY when the cap was 500 JPY), while keeping the threshold tied to the
# active campaign cap instead of a stale JPY literal.
HISTORICAL_LARGE_LOSS_CAP_MULTIPLE = 1.8

# Risk-geometry scoring buckets are fractions of the current per-trade cap:
# <=60% is high-quality unused risk budget, <=90% is acceptable, above 90%
# leaves little room for spread/slippage drift. These are scoring weights only;
# the RiskEngine remains the executable risk authority.
LOW_RISK_CAP_FRACTION = 0.60
MEDIUM_RISK_CAP_FRACTION = 0.90

# Pending-order replacement tolerance. This is deliberately a spread multiple,
# not a fixed pip distance: a valid trigger can drift by more raw pips in thin
# liquidity, while liquid tape should tolerate less. Above this many current
# spreads, the pending price is no longer the same executable neighborhood.
PENDING_ENTRY_REPLACE_SPREAD_MULT = 8.0

# Narrative penalties are score/ranking inputs, not risk gates. The JPY
# intervention penalty must be larger than a normal positive-evidence boost so
# rate-check / intervention risk actually reduces size while still leaving a
# current LIVE_READY receipt executable under AGENT_CONTRACT §6.
JPY_INTERVENTION_SCORE_PENALTY = 90.0
JPY_LIQUIDITY_SCORE_PENALTY = 25.0


@dataclass(frozen=True)
class LaneScore:
    lane_id: str
    pair: str
    direction: str
    method: str
    order_type: str
    entry: float | None
    status: str
    score: float
    action: str
    blockers: tuple[str, ...]
    rationale: tuple[str, ...]
    size_multiple: float = 1.0
    judgment: tuple[str, ...] = ()
    spread_pips: float | None = None


@dataclass(frozen=True)
class TraderDecision:
    action: str
    selected_lane_id: str | None
    generated_at_utc: str
    reason: str
    scores: tuple[LaneScore, ...]
    positions: int
    orders: int
    selected_lane_score: float | None = None
    selected_lane_size_multiple: float | None = None
    pending_cancel_order_ids: tuple[str, ...] = ()
    loss_cap_jpy: float | None = None
    loss_cap_source: str = ""


@dataclass(frozen=True)
class TraderSettings:
    score_bias: float = 0.0
    score_size_enabled: bool = True
    size_multiple_min: float = 0.7
    size_multiple_max: float = 1.8
    size_multiple_anchor_score: float = 110.0
    size_multiple_per_score_point: float = 0.005
    default_max_loss_jpy: float | None = None
    default_max_loss_pct: float | None = None


class TraderBrain:
    """Compare live-ready lanes using mined history, market story, and current risk state."""

    def __init__(
        self,
        *,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        campaign_plan_path: Path = DEFAULT_CAMPAIGN_PLAN,
        strategy_profile_path: Path = DEFAULT_STRATEGY_PROFILE,
        market_story_profile_path: Path = DEFAULT_MARKET_STORY_PROFILE,
        trader_settings_path: Path = DEFAULT_TRADER_SETTINGS,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        output_path: Path = DEFAULT_TRADER_DECISION,
        report_path: Path = DEFAULT_TRADER_DECISION_REPORT,
    ) -> None:
        self.intents_path = intents_path
        self.campaign_plan_path = campaign_plan_path
        self.strategy_profile_path = strategy_profile_path
        self.market_story_profile_path = market_story_profile_path
        self.trader_settings_path = trader_settings_path
        self.target_state_path = target_state_path
        self.output_path = output_path
        self.report_path = report_path

    def run(self, snapshot: BrokerSnapshot) -> TraderDecision:
        generated_at = datetime.now(timezone.utc).isoformat()
        intents_payload = _load_json(self.intents_path)
        campaign_payload = _load_json(self.campaign_plan_path)
        strategy_payload = _load_json(self.strategy_profile_path)
        story_payload = _load_json(self.market_story_profile_path)
        strategies = _strategy_index(strategy_payload)
        stories = _story_index(story_payload)
        campaign = _campaign_index(campaign_payload)
        trader_settings = _load_trader_settings(self.trader_settings_path)
        loss_cap_jpy, loss_cap_source = _resolve_trader_loss_cap(
            strategy_payload=strategy_payload,
            settings=trader_settings,
            target_state_path=self.target_state_path,
            snapshot=snapshot,
        )
        positions = len(snapshot.positions)
        orders = len(snapshot.orders)
        pending_entries = _pending_entry_order_count(snapshot)
        portfolio_add_allowed = _portfolio_add_allowed(snapshot)
        exposure_blockers = () if portfolio_add_allowed else _exposure_blockers(snapshot)
        scores = tuple(
            sorted(
                (
                    self._score_lane(
                        result,
                        strategies,
                        stories,
                        campaign,
                        exposure_blockers,
                        trader_settings,
                        loss_cap_jpy=loss_cap_jpy,
                    )
                    for result in intents_payload.get("results", [])
                    if isinstance(result, dict) and isinstance(result.get("intent"), dict)
                ),
                key=lambda item: item.score,
                reverse=True,
            )
        )
        if exposure_blockers or pending_entries:
            pending_cancel_order_ids = _contaminated_pending_order_ids(snapshot, scores)
            decision = TraderDecision(
                action=ACTION_MONITOR_EXISTING,
                selected_lane_id=None,
                selected_lane_score=None,
                selected_lane_size_multiple=None,
                generated_at_utc=generated_at,
                reason="Pending entry or non-layerable exposure is open; evaluate but do not add fresh risk.",
                scores=scores,
                positions=positions,
                orders=orders,
                pending_cancel_order_ids=pending_cancel_order_ids,
                loss_cap_jpy=loss_cap_jpy,
                loss_cap_source=loss_cap_source,
            )
        else:
            selected = _select_entry_lane(
                scores,
                target_state_path=self.target_state_path,
                snapshot=snapshot,
            )
            if selected:
                decision = TraderDecision(
                    action=ACTION_SEND_ENTRY,
                    selected_lane_id=selected.lane_id,
                    selected_lane_score=selected.score,
                    selected_lane_size_multiple=selected.size_multiple,
                    generated_at_utc=generated_at,
                    reason=_entry_selection_reason(scores, selected),
                    scores=scores,
                    positions=positions,
                    orders=orders,
                    loss_cap_jpy=loss_cap_jpy,
                    loss_cap_source=loss_cap_source,
                )
            else:
                decision = TraderDecision(
                    action=ACTION_NO_TRADE,
                    selected_lane_id=None,
                    selected_lane_score=None,
                    selected_lane_size_multiple=None,
                    generated_at_utc=generated_at,
                    reason="No lane cleared trader-brain discretionary gates.",
                    scores=scores,
                    positions=positions,
                    orders=orders,
                    loss_cap_jpy=loss_cap_jpy,
                    loss_cap_source=loss_cap_source,
                )
        self._write(decision)
        return decision

    def _score_lane(
        self,
        result: dict[str, Any],
        strategies: dict[tuple[str, str], dict[str, Any]],
        stories: dict[str, dict[str, Any]],
        campaign: dict[str, dict[str, Any]],
        exposure_blockers: tuple[str, ...],
        settings: TraderSettings,
        *,
        loss_cap_jpy: float | None,
    ) -> LaneScore:
        intent = result["intent"]
        lane_id = str(result.get("lane_id") or "")
        pair = str(intent.get("pair") or "")
        direction = str(intent.get("side") or "")
        method = str((intent.get("market_context") or {}).get("method") or "")
        order_type = str(intent.get("order_type") or "")
        entry = _optional_float(intent.get("entry"))
        status = str(result.get("status") or "")
        risk_metrics = result.get("risk_metrics") if isinstance(result.get("risk_metrics"), dict) else {}
        spread_pips = _optional_float(risk_metrics.get("spread_pips"))
        strategy = strategies.get((pair, direction), {})
        story = stories.get(pair, {})
        parent_lane_id = str((intent.get("metadata") or {}).get("parent_lane_id") or "")
        lane = campaign.get(lane_id) or campaign.get(parent_lane_id) or campaign.get(_parent_lane_id(lane_id)) or {}
        blockers: list[str] = list(exposure_blockers)
        rationale: list[str] = []
        score = 0.0

        if status == "LIVE_READY":
            score += 100.0
            rationale.append("live-ready risk/profile receipt")
        elif status == "DRY_RUN_PASSED":
            score += 35.0
            blockers.append("strategy still has live blockers")
        else:
            score -= 250.0
            blockers.append(f"intent status is {status}")

        profile_status = str(strategy.get("status") or "")
        if profile_status == "CANDIDATE":
            score += 25.0
            rationale.append("strategy profile candidate")
        elif profile_status:
            score -= 25.0
            blockers.append(f"strategy profile is {profile_status}")
        else:
            score -= 40.0
            blockers.append("missing strategy profile")

        pretrade_net = float(strategy.get("pretrade_net_jpy") or 0.0)
        live_net = float(strategy.get("live_net_jpy") or 0.0)
        live_worst = _optional_float(strategy.get("live_worst_jpy"))
        if loss_cap_jpy is None:
            blockers.append("trader loss cap missing for historical evidence scaling")
        else:
            pretrade_component = _clamp(pretrade_net / loss_cap_jpy, -25.0, 25.0)
            live_component = _clamp(live_net / loss_cap_jpy, -30.0, 30.0)
            if status != "LIVE_READY" or pretrade_component > 0:
                score += pretrade_component
            if status != "LIVE_READY" or live_component > 0:
                score += live_component
        if pretrade_net > 0:
            rationale.append(f"positive pretrade evidence {pretrade_net:.0f} JPY")
        if live_net > 0:
            rationale.append(f"positive live evidence {live_net:.0f} JPY")
        if live_net < 0:
            rationale.append(f"negative live execution history {live_net:.0f} JPY; current receipt is the authority")
            if status != "LIVE_READY":
                score -= 20.0
        if loss_cap_jpy is not None and live_worst is not None and live_worst <= -loss_cap_jpy:
            if live_worst <= -(loss_cap_jpy * HISTORICAL_LARGE_LOSS_CAP_MULTIPLE):
                rationale.append(
                    f"historical live worst loss is large: {live_worst:.0f} JPY; current receipt repairs sizing"
                )
            else:
                rationale.append(f"old worst loss repaired only by current sizing: {live_worst:.0f} JPY")
            if status != "LIVE_READY":
                score -= 8.0

        method_pressure = int((story.get("methods") or {}).get(method, 0))
        score += _clamp(method_pressure * 0.25, 0.0, 30.0)
        if method_pressure:
            rationale.append(f"market-story method pressure {method_pressure}")
        themes = dict(story.get("themes") or {})
        examples = tuple(str(item) for item in story.get("examples", [])[:4])
        score += _method_theme_score(method, themes, rationale)
        score += _campaign_score(lane, rationale)
        score += _narrative_risk_score(pair, direction, method, themes, examples, blockers, rationale, status=status)
        score += _technical_consensus_score(
            intent=intent,
            method=method,
            status=status,
            strategy=strategy,
            story=story,
            lane=lane,
            risk_metrics=risk_metrics,
            method_pressure=method_pressure,
            loss_cap_jpy=loss_cap_jpy,
            rationale=rationale,
            blockers=blockers,
        )
        score += _direction_conflict_penalty(result, rationale)
        if order_type.upper() == "MARKET" and status == "LIVE_READY":
            score += 5.0
            rationale.append("market receipt can execute the current quote instead of waiting for a trigger")

        if result.get("risk_issues"):
            blockers.extend(str(issue.get("message") or issue.get("code")) for issue in result.get("risk_issues", []))
            score -= 100.0
        if result.get("live_blockers"):
            blockers.extend(str(item) for item in result.get("live_blockers", []))
            score -= 100.0

        gate_blockers, judgment = _discretionary_gate_check(
            intent=intent,
            status=status,
            profile_status=profile_status,
            strategy=strategy,
            lane=lane,
            method=method,
            method_pressure=method_pressure,
        )
        blockers.extend(gate_blockers)

        adjusted_score = round(score + settings.score_bias, 2)
        size_multiple = _size_multiple(adjusted_score, settings)
        action = ACTION_SEND_ENTRY if status == "LIVE_READY" and not blockers else ACTION_NO_TRADE
        return LaneScore(
            lane_id=lane_id,
            pair=pair,
            direction=direction,
            method=method,
            order_type=order_type,
            entry=entry,
            status=status,
            score=adjusted_score,
            size_multiple=size_multiple,
            action=action,
            blockers=tuple(blockers[:8]),
            rationale=tuple(rationale[:8]),
            judgment=tuple(judgment[:8]),
            spread_pips=spread_pips,
        )

    def _write(self, decision: TraderDecision) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(
                {
                    "action": decision.action,
                    "selected_lane_id": decision.selected_lane_id,
                    "selected_lane_score": decision.selected_lane_score,
                    "selected_lane_size_multiple": decision.selected_lane_size_multiple,
                    "generated_at_utc": decision.generated_at_utc,
                    "reason": decision.reason,
                    "positions": decision.positions,
                    "orders": decision.orders,
                    "pending_cancel_order_ids": list(decision.pending_cancel_order_ids),
                    "loss_cap_jpy": decision.loss_cap_jpy,
                    "loss_cap_source": decision.loss_cap_source,
                    "scores": [asdict(item) for item in decision.scores],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Trader Decision Report",
            "",
            f"- Generated at UTC: `{decision.generated_at_utc}`",
            f"- Action: `{decision.action}`",
            f"- Selected lane: `{decision.selected_lane_id}`",
            f"- Selected lane score: `{decision.selected_lane_score}`",
            f"- Selected lane size multiple: `{decision.selected_lane_size_multiple}`",
            f"- Positions: `{decision.positions}`",
            f"- Orders: `{decision.orders}`",
            f"- Pending cancel ids: `{', '.join(decision.pending_cancel_order_ids) if decision.pending_cancel_order_ids else 'none'}`",
            f"- Loss cap: `{decision.loss_cap_jpy if decision.loss_cap_jpy is not None else 'missing'}` (`{decision.loss_cap_source or 'missing'}`)",
            f"- Reason: {decision.reason}",
            "",
            "## Ranked Lanes",
            "",
        ]
        for item in decision.scores[:12]:
            lines.append(
                f"- `{item.lane_id}` score=`{item.score}` action=`{item.action}` "
                f"`{item.pair} {item.direction} {item.method}`"
            )
            lines.append(f"  - size_multiple: `{item.size_multiple}`")
            if item.rationale:
                lines.append(f"  - why: {'; '.join(item.rationale)}")
            if item.judgment:
                lines.append(f"  - judgment: {'; '.join(item.judgment)}")
            if item.blockers:
                lines.append(f"  - blockers: {'; '.join(item.blockers)}")
        lines.extend(
            [
                "",
                "## Trader-Brain Contract",
                "",
                "- This layer must compare lanes; it must not send the first live-ready candidate mechanically.",
                "- Scores rank attention only; live entry requires explicit discretionary gates, not a single score threshold.",
                "- Pending entry or non-layerable exposure makes TraderBrain monitor-only; automation may pass compatible pending entries to gateway basket validation.",
                "- JPY-cross long trades are penalized when intervention / thin-liquidity themes are active.",
                "- The execution gateway remains the final authority for live risk.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _select_entry_lane(
    scores: tuple[LaneScore, ...],
    *,
    target_state_path: Path,
    snapshot: BrokerSnapshot,
) -> LaneScore | None:
    sendable = tuple(item for item in scores if item.action == ACTION_SEND_ENTRY)
    if not sendable:
        return None
    if _target_open_needs_immediate_entry(target_state_path, snapshot):
        market = next((item for item in sendable if item.order_type.upper() == "MARKET"), None)
        if market is not None:
            return market
    return sendable[0]


def _entry_selection_reason(scores: tuple[LaneScore, ...], selected: LaneScore) -> str:
    top_sendable = next((item for item in scores if item.action == ACTION_SEND_ENTRY), None)
    if selected.order_type.upper() == "MARKET":
        if top_sendable is not None and top_sendable.lane_id != selected.lane_id:
            return f"Selected live-ready MARKET lane for target-open immediate exposure: {selected.lane_id}"
        return f"Selected highest-scoring live-ready MARKET lane for target-open immediate exposure: {selected.lane_id}"
    return f"Selected highest-scoring live-ready lane: {selected.lane_id}"


def _target_open_needs_immediate_entry(target_state_path: Path, snapshot: BrokerSnapshot) -> bool:
    if not _target_open(target_state_path):
        return False
    trader_positions = sum(1 for position in snapshot.positions if position.owner == Owner.TRADER)
    return trader_positions == 0 and _pending_entry_order_count(snapshot) == 0


def _target_open(target_state_path: Path) -> bool:
    target = _load_json(target_state_path)
    if str(target.get("status") or "") != "PURSUE_TARGET":
        return False
    return float(target.get("remaining_target_jpy") or 0.0) > 0


def load_trader_settings(path: Path) -> TraderSettings:
    return _load_trader_settings(path)


def _load_trader_settings(path: Path) -> TraderSettings:
    payload = _load_json(path)
    settings_payload = payload.get("size_by_score")
    if not isinstance(settings_payload, dict):
        settings_payload = {}
    risk_payload = payload.get("risk")
    if not isinstance(risk_payload, dict):
        risk_payload = {}
    score_bias = _coalesce_float(settings_payload.get("score_bias"), 0.0)
    score_size_enabled = settings_payload.get("enabled")
    if not isinstance(score_size_enabled, bool):
        score_size_enabled = True
    size_multiple_min = _coalesce_float(settings_payload.get("size_multiple_min"), 0.7)
    size_multiple_max = _coalesce_float(settings_payload.get("size_multiple_max"), 1.8)
    if size_multiple_max < size_multiple_min:
        size_multiple_min, size_multiple_max = size_multiple_max, size_multiple_min
    if size_multiple_min <= 0:
        size_multiple_min = 0.05
    size_multiple_anchor_score = _coalesce_float(settings_payload.get("size_multiple_anchor_score"), 110.0)
    size_multiple_per_score_point = _coalesce_float(
        settings_payload.get("size_multiple_per_score_point"), 0.005
    )
    if size_multiple_per_score_point < 0:
        size_multiple_per_score_point = 0.0
    default_max_loss_jpy = _optional_float(risk_payload.get("max_loss_jpy"))
    default_max_loss_pct = _optional_float(risk_payload.get("max_loss_pct"))
    return TraderSettings(
        score_bias=score_bias,
        score_size_enabled=bool(score_size_enabled),
        size_multiple_min=size_multiple_min,
        size_multiple_max=size_multiple_max,
        size_multiple_anchor_score=size_multiple_anchor_score,
        size_multiple_per_score_point=size_multiple_per_score_point,
        default_max_loss_jpy=default_max_loss_jpy,
        default_max_loss_pct=default_max_loss_pct,
    )


def _size_multiple(score: float, settings: TraderSettings) -> float:
    if not settings.score_size_enabled:
        return 1.0
    multiple = 1.0 + ((score - settings.size_multiple_anchor_score) * settings.size_multiple_per_score_point)
    return round(_clamp(multiple, settings.size_multiple_min, settings.size_multiple_max), 2)


def _coalesce_float(value: object, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _resolve_trader_loss_cap(
    *,
    strategy_payload: dict[str, Any],
    settings: TraderSettings,
    target_state_path: Path,
    snapshot: BrokerSnapshot,
) -> tuple[float | None, str]:
    cap = _loss_cap_from_target_state(target_state_path)
    if cap is not None:
        return cap, f"daily target state {target_state_path}"
    cap = _loss_cap_from_strategy_payload(strategy_payload)
    if cap is not None:
        return cap, "strategy profile system_contract.loss_cap_jpy"
    if settings.default_max_loss_jpy is not None and settings.default_max_loss_jpy > 0:
        return round(settings.default_max_loss_jpy, 4), "trader settings risk.max_loss_jpy"
    if (
        settings.default_max_loss_pct is not None
        and settings.default_max_loss_pct > 0
        and snapshot.account is not None
        and snapshot.account.balance_jpy > 0
    ):
        cap = snapshot.account.balance_jpy * (settings.default_max_loss_pct / 100.0)
        return round(cap, 4), "trader settings risk.max_loss_pct of broker balance"
    return None, "missing loss cap"


def _loss_cap_from_strategy_payload(payload: dict[str, Any]) -> float | None:
    contract = payload.get("system_contract")
    if not isinstance(contract, dict):
        return None
    return _positive_float(contract.get("loss_cap_jpy"))


def _loss_cap_from_target_state(path: Path) -> float | None:
    payload = _load_json(path)
    return _positive_float(payload.get("per_trade_risk_budget_jpy"))


def _positive_float(value: object) -> float | None:
    parsed = _optional_float(value)
    if parsed is None or parsed <= 0:
        return None
    return round(parsed, 4)


def _strategy_index(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in payload.get("profiles", []) or []:
        if isinstance(item, dict):
            pair = str(item.get("pair") or "")
            direction = str(item.get("direction") or "")
            if pair and direction:
                index[(pair, direction)] = item
    return index


def _story_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in payload.get("pair_profiles", []) or []:
        if isinstance(item, dict) and item.get("pair"):
            index[str(item["pair"])] = item
    return index


def _campaign_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for lane in payload.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        lane_id = f"{lane.get('desk')}:{lane.get('pair')}:{lane.get('direction')}:{lane.get('method')}"
        index[lane_id] = lane
    return index


def _parent_lane_id(lane_id: str) -> str:
    if lane_id.endswith(":MARKET"):
        return lane_id[: -len(":MARKET")]
    return lane_id


def _exposure_blockers(snapshot: BrokerSnapshot) -> tuple[str, ...]:
    blockers: list[str] = []
    for position in snapshot.positions:
        if position.owner in {Owner.MANUAL, Owner.UNKNOWN}:
            continue
        blockers.append(f"open position exists: {position.pair} {position.side.value} id={position.trade_id}")
    for order in snapshot.orders:
        if order.owner in {Owner.MANUAL, Owner.UNKNOWN}:
            continue
        if not order.trade_id and order.order_type.upper() in PENDING_ENTRY_TYPES:
            blockers.append(f"pending entry exists: {order.pair} {order.order_type} id={order.order_id}")
    return tuple(blockers)


def _pending_entry_order_count(snapshot: BrokerSnapshot) -> int:
    return sum(
        1
        for order in snapshot.orders
        if not order.trade_id and order.order_type.upper() in PENDING_ENTRY_TYPES
        and order.owner not in {Owner.MANUAL, Owner.UNKNOWN}
    )


def _portfolio_add_allowed(snapshot: BrokerSnapshot) -> bool:
    trader_positions = tuple(position for position in snapshot.positions if position.owner == Owner.TRADER)
    if not trader_positions:
        return False
    return all(
        position.owner == Owner.TRADER and position.stop_loss is not None and position.take_profit is not None
        for position in trader_positions
    )


def _contaminated_pending_order_ids(snapshot: BrokerSnapshot, scores: tuple[LaneScore, ...]) -> tuple[str, ...]:
    scores_by_key: dict[tuple[str, str, str], list[LaneScore]] = {}
    for score in scores:
        key = (score.pair, score.direction, _normalized_entry_type(score.order_type))
        scores_by_key.setdefault(key, []).append(score)
    contaminated: list[str] = []
    for order in snapshot.orders:
        if order.trade_id or order.order_type.upper() not in PENDING_ENTRY_TYPES:
            continue
        direction = _order_direction(order.units)
        if not order.pair or direction is None:
            continue
        if order.owner != Owner.TRADER:
            continue
        compatible_scores = scores_by_key.get((order.pair, direction, _normalized_entry_type(order.order_type)), [])
        if not compatible_scores:
            contaminated.append(order.order_id)
            continue
        if not any(_keeps_pending_order(order, score) for score in compatible_scores):
            contaminated.append(order.order_id)
    return tuple(contaminated)


def _order_direction(units: int | None) -> str | None:
    if units is None:
        return None
    if units > 0:
        return Side.LONG.value
    if units < 0:
        return Side.SHORT.value
    return None


def _same_entry_type(order_type: str, lane_order_type: str) -> bool:
    return _normalized_entry_type(order_type) == _normalized_entry_type(lane_order_type)


def _normalized_entry_type(order_type: str) -> str:
    return "STOP-ENTRY" if order_type.upper() == "STOP" else order_type.upper()


def _keeps_pending_order(order: BrokerOrder, score: LaneScore) -> bool:
    blocker_text = " ".join(score.blockers).upper()
    if "INTERVENTION" in blocker_text or "VISUAL STORY" in blocker_text:
        return False
    if score.action != ACTION_SEND_ENTRY and not _blocked_only_by_existing_pending(score):
        return False
    return not _entry_drift_exceeds_current_spread(order.pair or "", order.price, score.entry, score.spread_pips)


def _blocked_only_by_existing_pending(score: LaneScore) -> bool:
    return bool(score.blockers) and all(str(item).startswith("pending entry exists:") for item in score.blockers)


def _entry_drift_pips(pair: str, order_price: float | None, lane_entry: float | None) -> float:
    if order_price is None or lane_entry is None:
        return 0.0
    pip_factor = 100 if pair.endswith("_JPY") else 10000
    return abs(order_price - lane_entry) * pip_factor


def _entry_drift_exceeds_current_spread(
    pair: str,
    order_price: float | None,
    lane_entry: float | None,
    spread_pips: float | None,
) -> bool:
    if spread_pips is None or spread_pips <= 0:
        return False
    # Pending entries should not be canceled just because the next broker snapshot
    # reprices the same setup by a few ticks. The replacement threshold is tied to
    # current spread, so it expands in thin liquidity and tightens in normal tape.
    return _entry_drift_pips(pair, order_price, lane_entry) > spread_pips * PENDING_ENTRY_REPLACE_SPREAD_MULT


def _method_theme_score(method: str, themes: dict[str, Any], rationale: list[str]) -> float:
    score = 0.0
    if method == TradeMethod.TREND_CONTINUATION.value and int(themes.get("momentum") or 0) > 0:
        score += 12.0
        rationale.append("momentum theme supports trend")
    if method == TradeMethod.RANGE_ROTATION.value and int(themes.get("range_rail") or 0) > 0:
        score += 12.0
        rationale.append("range rail theme supports rotation")
    if method == TradeMethod.BREAKOUT_FAILURE.value and int(themes.get("breakout_failure") or 0) > 0:
        score += 14.0
        rationale.append("breakout-failure theme supports trap/reclaim")
    if int(themes.get("event_risk") or 0) > 0:
        score -= 8.0
        rationale.append("event risk requires restraint")
    if int(themes.get("spread_liquidity") or 0) > 0:
        score -= 10.0
        rationale.append("spread/liquidity theme reduces urgency")
    return score


def _campaign_score(lane: dict[str, Any], rationale: list[str]) -> float:
    role = str(lane.get("campaign_role") or "")
    adoption = str(lane.get("adoption") or "")
    score = 0.0
    if "NOW" in role:
        score += 14.0
        rationale.append(f"campaign role {role}")
    elif "BACKUP" in role:
        score += 6.0
    if adoption == "ORDER_INTENT_REQUIRED":
        score += 8.0
    return score


def _narrative_risk_score(
    pair: str,
    direction: str,
    method: str,
    themes: dict[str, Any],
    examples: tuple[str, ...],
    blockers: list[str],
    rationale: list[str],
    *,
    status: str,
) -> float:
    # AGENT_CONTRACT §6: current discretionary narrative concerns size the lane
    # down via score -> size_multiple. They MUST NOT block the lane in prose. Per §3.5
    # per_trade_risk_budget_jpy already shrinks the per-shot exposure; layering
    # an "intervention narrative" or "visual story rejected" gate on top is an
    # invented threshold not enumerated in §3.5/§9/§10/§11. Surface the concern
    # in rationale so the operator/GPT sees it, but let the lane stay tradable.
    text = " ".join(examples).upper()
    score = 0.0
    is_live_ready = status == "LIVE_READY"
    if pair in JPY_CROSSES and direction == Side.LONG.value:
        intervention = int(themes.get("intervention") or 0)
        liquidity = int(themes.get("spread_liquidity") or 0)
        if intervention or "INTERVENTION" in text or "RATE CHECK" in text:
            score -= JPY_INTERVENTION_SCORE_PENALTY
            rationale.append("JPY-cross long under intervention/rate-check narrative; size multiple reduced")
        if liquidity or "GOLDEN WEEK" in text:
            score -= JPY_LIQUIDITY_SCORE_PENALTY
            rationale.append("JPY liquidity theme requires smaller/fewer entries")
    if "WAIT" in text:
        if is_live_ready:
            rationale.append("stale narrative WAIT language ignored for live-ready receipt")
        else:
            score -= 18.0
            rationale.append("recent narrative contained WAIT language")
    if "NO:" in text and method == TradeMethod.RANGE_ROTATION.value:
        if is_live_ready:
            rationale.append("stale visual rejection marker ignored for live-ready receipt")
        else:
            score -= 28.0
            rationale.append("visual story rejected range rotation; size multiple reduced")
    if "TREND-BULL" in text and direction == Side.LONG.value:
        score += 10.0
    if "TREND-BEAR" in text and direction == Side.SHORT.value:
        score += 10.0
    return score


def _technical_consensus_score(
    *,
    intent: dict[str, Any],
    method: str,
    status: str,
    strategy: dict[str, Any],
    story: dict[str, Any],
    lane: dict[str, Any],
    risk_metrics: dict[str, Any] | None,
    method_pressure: int,
    loss_cap_jpy: float | None,
    rationale: list[str],
    blockers: list[str],
) -> float:
    score = 0.0
    support_ticks = 0
    evidence_ticks = 0

    positive_evidence_n = int(strategy.get("positive_evidence_n") or 0)
    positive_tail_jpy = float(strategy.get("positive_tail_jpy") or 0.0)
    positive_best_jpy = float(strategy.get("positive_best_jpy") or 0.0)
    seat_discovered = int(strategy.get("seat_discovered") or 0)
    seat_orderable = int(strategy.get("seat_orderable") or 0)
    seat_captured = int(strategy.get("seat_captured") or 0)
    live_worst = _optional_float(strategy.get("live_worst_jpy"))
    required_fix = str(strategy.get("required_fix") or "")

    # Evidence depth and quality are one pillar.
    if positive_evidence_n >= 120:
        score += 8.0
        evidence_ticks += 1
        rationale.append(f"broad positive evidence count={positive_evidence_n}")
    elif positive_evidence_n >= 40:
        score += 5.0
        evidence_ticks += 1
        rationale.append(f"positive evidence count={positive_evidence_n}")
    elif positive_evidence_n > 0:
        score += 1.0
        rationale.append(f"some positive evidence count={positive_evidence_n}")
    else:
        if status == "LIVE_READY":
            rationale.append("missing positive mined evidence on this pair/direction; advisory only")
        else:
            score -= 3.0
            rationale.append("missing positive mined evidence on this pair/direction; repair required before live-ready")

    if seat_orderable > 0 and seat_discovered > 0:
        capture_rate = seat_captured / seat_discovered
        if capture_rate >= 0.50:
            score += 4.0
            support_ticks += 1
            rationale.append(f"high capture quality={capture_rate:.0%} ({seat_captured}/{seat_discovered})")
        elif capture_rate >= 0.30:
            score += 2.0
            support_ticks += 1
        elif capture_rate >= 0.10:
            score += 0.0
        else:
            if status == "LIVE_READY":
                rationale.append(f"low capture rate={capture_rate:.0%} ({seat_captured}/{seat_discovered}); advisory only")
            else:
                score -= 4.0
                rationale.append(
                    f"low capture rate={capture_rate:.0%} ({seat_captured}/{seat_discovered}); repair required before live-ready"
                )
    if positive_tail_jpy > 0:
        score += 2.0
    if positive_best_jpy > 0:
        score += 1.5

    # Risk geometry quality is second pillar.
    risk_metrics = risk_metrics or {}
    reward_risk = _optional_float(risk_metrics.get("reward_risk"))
    spread_pips = _optional_float(risk_metrics.get("spread_pips"))
    risk_jpy = _optional_float(risk_metrics.get("risk_jpy"))
    reward_jpy = _optional_float(risk_metrics.get("reward_jpy"))
    plan_rr = _optional_float(lane.get("target_reward_risk"))

    if reward_risk is None or spread_pips is None or risk_jpy is None:
        score -= 2.5
        blockers.append("missing dry-run risk geometry metric")
    else:
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        if method == TradeMethod.RANGE_ROTATION.value and metadata.get("geometry_model") == "RANGE_RAIL_LIMIT":
            score += 8.0
            support_ticks += 1
            rationale.append(
                f"range LIMIT is anchored to {metadata.get('range_entry_side')} rail "
                f"{metadata.get('range_support')}–{metadata.get('range_resistance')}"
            )
        if reward_risk >= 1.2:
            score += 2.0
            support_ticks += 1
        else:
            score -= 6.0
            blockers.append(f"reward/risk below minimum floor: {reward_risk:.2f}R")
        if reward_risk >= 2.0:
            score += 2.5
            rationale.append(f"reward/risk geometry supports edge: {reward_risk:.2f}R")
        if spread_pips <= 1.2:
            score += 2.0
            support_ticks += 1
            rationale.append(f"tight spread={spread_pips:.1f}pip")
        elif spread_pips <= 2.0:
            score += 0.8
        else:
            score -= 2.0
            blockers.append(f"wide spread for fresh edge={spread_pips:.1f}pip")
        if loss_cap_jpy is None:
            blockers.append("trader loss cap missing for risk geometry ranking")
        else:
            risk_fraction = risk_jpy / loss_cap_jpy
            if risk_fraction <= LOW_RISK_CAP_FRACTION:
                score += 2.0
                support_ticks += 1
            elif risk_fraction <= MEDIUM_RISK_CAP_FRACTION:
                score += 1.0
            else:
                score -= 2.0
        if plan_rr is not None and reward_risk >= plan_rr * 0.85:
            score += 1.5
            rationale.append(f"geometry reward/risk {reward_risk:.2f}R matches lane target {plan_rr:.2f}R")
        if reward_jpy is not None and reward_jpy > risk_jpy:
            score += 1.0

    # Strategy context and campaign contract consistency is third pillar.
    if method_pressure <= 0 and not story.get("methods"):
        if status == "LIVE_READY" and support_ticks >= 2 and evidence_ticks >= 1:
            rationale.append("entry can still pass with strong statistical edge despite weak live story pressure")
        elif status == "LIVE_READY":
            rationale.append("live technical story lacks method pressure; advisory only")
        else:
            score -= 5.0
            blockers.append("live technical story lacks method pressure for this setup")

    if required_fix and "watch-only" in required_fix.lower():
        if status == "LIVE_READY":
            rationale.append("strategy required_fix still mentions watch-only restrictions; advisory only")
        else:
            score -= 2.0
            rationale.append("strategy required_fix still mentions watch-only restrictions; repair required before live-ready")

    if (
        loss_cap_jpy is not None
        and live_worst is not None
        and live_worst <= -(loss_cap_jpy * HISTORICAL_LARGE_LOSS_CAP_MULTIPLE)
    ):
        if status != "LIVE_READY":
            score -= 2.0

    if status == "LIVE_READY" and intent.get("order_type"):
        support_ratio = (support_ticks + 1) / 5.0
        if support_ratio >= 0.7:
            score += 3.0
        elif support_ratio <= 0.2:
            score -= 3.0

    score += _story_fusion_score(
        method=method,
        direction=str(intent.get("side") or ""),
        examples=tuple(str(item) for item in story.get("examples", ())),
        score=score,
        rationale=rationale,
        blockers=blockers,
        support_ticks=support_ticks,
        status=status,
    )

    return score


def _story_fusion_score(
    *,
    method: str,
    direction: str,
    examples: tuple[str, ...],
    score: float,
    rationale: list[str],
    blockers: list[str],
    support_ticks: int,
    status: str,
) -> float:
    delta = 0.0
    if not examples:
        if status == "LIVE_READY":
            rationale.append("story has no concrete technical/news/chart examples; advisory only")
            return 0.0
        blockers.append("story has no concrete technical/news/chart examples")
        return -3.0

    source_counts: dict[str, int] = {}
    for item in examples:
        upper = item.upper()
        source = "OTHER"
        if upper.startswith("NEWS_DIGEST"):
            source = "NEWS"
        elif upper.startswith("NEWS_FLOW"):
            source = "FLOW"
        elif upper.startswith("QUALITY_AUDIT"):
            source = "QUALITY"
        source_counts[source] = source_counts.get(source, 0) + 1
        if "TREND-BULL" in upper and direction == Side.LONG.value:
            delta += 1.5
            support_ticks += 1
        elif "TREND-BEAR" in upper and direction == Side.SHORT.value:
            delta += 1.5
            support_ticks += 1
        elif "TREND-BULL" in upper and direction == Side.SHORT.value:
            delta -= 1.2
        elif "TREND-BEAR" in upper and direction == Side.LONG.value:
            delta -= 1.2

    method_token = {
        "TREND_CONTINUATION": "TREND",
        "RANGE_ROTATION": "RANGE",
        "BREAKOUT_FAILURE": "BREAKOUT",
        "EVENT_RISK": "EVENT",
        "POSITION_MANAGEMENT": "POSITION",
    }.get(method, "")

    source_diversity = len(source_counts)
    if source_diversity >= 2:
        delta += 2.0
        rationale.append(f"multi-source story coverage ({', '.join(sorted(source_counts))})")
        if source_counts.get("QUALITY", 0) > 0 and source_counts.get("NEWS", 0) > 0:
            delta += 1.0
            rationale.append("news and chart-quality evidence agree on setup")
    else:
        if status == "LIVE_READY":
            rationale.append("story evidence lacks source diversity; advisory only")
        else:
            delta -= 1.0
            blockers.append("story evidence lacks source diversity")

    conflict_hits = sum(1 for item in examples if "NO:" in item.upper())
    if conflict_hits >= 2:
        if status == "LIVE_READY":
            rationale.append("story explicitly contains mixed rejection markers; advisory only")
        else:
            delta -= 2.5
            blockers.append("story explicitly contains mixed rejection markers")
    elif conflict_hits == 1:
        if status != "LIVE_READY":
            delta -= 1.0

    if method_token and any(method_token in item.upper() for item in examples):
        delta += 2.0
        support_ticks += 1
        rationale.append(f"story examples confirm method token={method_token}")

    if support_ticks >= 3 and score >= 80:
        delta += 1.0
    return delta


def _direction_conflict_penalty(result: dict[str, Any], rationale: list[str]) -> float:
    intent = result.get("intent") or {}
    pair = str(intent.get("pair") or "")
    direction = str(intent.get("side") or "")
    context = intent.get("market_context") or {}
    narrative = f"{context.get('narrative') or ''} {context.get('chart_story') or ''}".upper()
    if pair == "EUR_USD" and "DIRECTIONLESS" in narrative:
        rationale.append("EUR_USD narrative is directionless; require cleaner proof")
        return -10.0 if direction else 0.0
    return 0.0


def _discretionary_gate_check(
    *,
    intent: dict[str, Any],
    status: str,
    profile_status: str,
    strategy: dict[str, Any],
    lane: dict[str, Any],
    method: str,
    method_pressure: int,
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    judgment: list[str] = []
    if status == "LIVE_READY":
        judgment.append("fresh live-ready receipt exists")
    else:
        blockers.append(f"receipt is not live-ready: {status}")

    if profile_status == "CANDIDATE":
        judgment.append("strategy profile is live-eligible")
    elif profile_status:
        blockers.append(f"strategy profile is not live-eligible: {profile_status}")
    else:
        blockers.append("missing strategy profile")

    if not str(intent.get("thesis") or "").strip():
        blockers.append("missing trader thesis")
    context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else None
    if context is None:
        blockers.append("missing market context")
    else:
        missing_context = [
            name
            for name in ("regime", "narrative", "chart_story", "method", "invalidation")
            if not str(context.get(name) or "").strip()
        ]
        if missing_context:
            blockers.append(f"incomplete market context: {', '.join(missing_context)}")
        else:
            judgment.append("thesis, narrative, chart story, method, and invalidation are explicit")

    if _optional_float(intent.get("tp")) is None or _optional_float(intent.get("sl")) is None:
        blockers.append("missing TP/SL geometry")
    if int(intent.get("units") or 0) <= 0:
        blockers.append("missing executable units")

    adoption = str(lane.get("adoption") or "")
    if adoption in {"ORDER_INTENT_REQUIRED", "RISK_REPAIR_DRY_RUN", "TRIGGER_RECEIPT_REQUIRED"}:
        judgment.append(f"campaign lane is executable after receipts: {adoption}")
    else:
        blockers.append(f"campaign lane is not executable: {adoption or 'missing'}")

    pretrade_net = float(strategy.get("pretrade_net_jpy") or 0.0)
    live_net = float(strategy.get("live_net_jpy") or 0.0)
    if pretrade_net > 0 or live_net > 0 or strategy.get("receipt_promotion"):
        judgment.append("mined or repaired edge evidence is positive")
    elif status == "LIVE_READY" and profile_status == "CANDIDATE":
        judgment.append("past edge evidence is weak/negative, but the current live-ready receipt remains executable")
    else:
        blockers.append("no positive mined or repaired edge evidence")

    if method and method_pressure > 0:
        judgment.append(f"current story contains method pressure for {method}")
    elif method and (
        int(strategy.get("positive_evidence_n") or 0) >= 40
        or str(strategy.get("status")) == "CANDIDATE"
    ):
        judgment.append("market story is weak, but evidence is strong enough to keep under review")
    else:
        blockers.append("market story does not support the selected method")
    return blockers, judgment


def _optional_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

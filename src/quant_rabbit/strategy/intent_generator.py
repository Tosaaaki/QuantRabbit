from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.models import BrokerOrder, BrokerPosition, BrokerSnapshot, MarketContext, OrderIntent, OrderType, Owner, Quote, Side, TradeMethod
from quant_rabbit.paths import (
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_ORDER_INTENT_REPORT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_STRATEGY_PROFILE,
)
from quant_rabbit.risk import RiskEngine, RiskPolicy, resolve_max_loss_jpy
from quant_rabbit.strategy.profile import StrategyProfile


# Geometry tuning constants. Per AGENT_CONTRACT §3.5, every constant on the
# trader risk path needs a market-reality reason. These are *minimums* / floors,
# not the truth — the actual stop distance is the larger of ATR-derived and
# spread-derived candidates.
#
# - GEOMETRY_ATR_MULT: 1.0 ATR is a "typical move" of the timeframe. Trader
#   takes a setup expecting the next ATR to either prove the thesis (toward TP)
#   or invalidate it (through SL). Tighter than 1.0 means routine noise hits SL;
#   wider would needlessly enlarge worst-case loss.
# - GEOMETRY_SPREAD_FLOOR_MULT: 6.0 × spread protects against broker fill jitter
#   and wick noise around the entry. Must be >= RiskPolicy.min_stop_spread_multiple
#   (currently 5.0) — the validator rejects stops thinner than 5× spread, so the
#   generator pre-emptively floors at 6× to leave a small safety margin.
# - GEOMETRY_ATR_TIMEFRAME: M5 is the operating timeframe of the scalp / swing
#   trader. M15 / H1 ATR is also available in pair_charts but reflects slower
#   structure than the trader is reacting to.
GEOMETRY_ATR_MULT = 1.0
GEOMETRY_SPREAD_FLOOR_MULT = 6.0
GEOMETRY_ATR_TIMEFRAME = "M5"


def _per_trade_risk_from_state(state_path: Path = DEFAULT_DAILY_TARGET_STATE) -> float | None:
    """Return per-trade JPY cap from the daily target ledger, or None if unavailable.

    Reads `per_trade_risk_budget_jpy` (= daily_risk_budget_jpy / target_trades_per_day),
    written by DailyTargetLedger. Falling back to `daily_risk_budget_jpy` (the
    whole-day total) would mean a single trade can burn the day's entire risk
    budget, which is exactly the failure mode this split was built to remove.
    Per AGENT_CONTRACT §3.5: no silent literal fallback; if the file is missing
    or the value is missing/zero, return None and let the caller raise.
    """
    return _state_field(state_path, "per_trade_risk_budget_jpy")


def _daily_risk_budget_from_state(state_path: Path = DEFAULT_DAILY_TARGET_STATE) -> float | None:
    """Return whole-day JPY risk budget from the daily target ledger.

    Per AGENT_CONTRACT §3.5 the **portfolio** cap (open + candidate exposure)
    must be the day's total budget, NOT the per-trade slice. Reusing the
    per-trade cap as the portfolio cap silently blocks every additional shot
    once one position opens, because `open_risk + candidate_risk` immediately
    exceeds a per-shot limit. Returns None when the ledger is absent so the
    caller can decide whether to skip the portfolio gate (no-op) rather than
    invent a JPY literal.
    """
    return _state_field(state_path, "daily_risk_budget_jpy")


def _state_field(state_path: Path, key: str) -> float | None:
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    raw = payload.get(key)
    try:
        value = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _load_pair_charts(charts_path: Path = DEFAULT_PAIR_CHARTS) -> dict[str, dict[str, Any]] | None:
    """Load pair_charts.json indexed by pair name.

    Returns a dict like {"EUR_USD": {"M5": {"atr_pips": 5.2, "regime": ...}, ...}, ...}.
    Returns None when the file is absent / malformed — the caller must decide
    whether to BLOCK the cycle (production) or proceed without ATR (tests).
    """
    if not charts_path.exists():
        return None
    try:
        payload = json.loads(charts_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    raw_charts = payload.get("charts")
    if not isinstance(raw_charts, list):
        return None
    indexed: dict[str, dict[str, Any]] = {}
    for chart in raw_charts:
        pair = chart.get("pair")
        if not isinstance(pair, str):
            continue
        per_tf: dict[str, Any] = {"dominant_regime": chart.get("dominant_regime")}
        for view in chart.get("views", []) or []:
            granularity = view.get("granularity")
            if isinstance(granularity, str):
                per_tf[granularity] = view.get("indicators", {}) or {}
        indexed[pair] = per_tf
    return indexed if indexed else None


def _atr_pips_for(pair: str, charts: dict[str, dict[str, Any]] | None, timeframe: str = GEOMETRY_ATR_TIMEFRAME) -> float | None:
    """Look up ATR(pips) for a pair on the given timeframe. None when missing."""
    if charts is None:
        return None
    per_tf = charts.get(pair)
    if not per_tf:
        return None
    indicators = per_tf.get(timeframe)
    if not isinstance(indicators, dict):
        return None
    raw = indicators.get("atr_pips")
    try:
        value = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


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
    risk_metrics: dict[str, Any] | None
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
        pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
        max_loss_jpy: float | None = None,
        max_loss_pct: float | None = None,
        risk_equity_jpy: float | None = None,
    ) -> None:
        self.campaign_plan = campaign_plan
        self.strategy_profile = strategy_profile
        self.output_path = output_path
        self.report_path = report_path
        self.pair_charts_path = pair_charts_path
        self.max_loss_jpy = max_loss_jpy
        self.max_loss_pct = max_loss_pct
        self.risk_equity_jpy = risk_equity_jpy

    def run(self, *, snapshot_path: Path | None = None, max_candidates: int = 12) -> IntentGenerationSummary:
        plan = json.loads(self.campaign_plan.read_text())
        lanes = [lane for lane in plan.get("lanes", []) if _lane_can_attempt(lane)]
        snapshot = _snapshot_from_json(json.loads(snapshot_path.read_text())) if snapshot_path else None
        strategy_profile = StrategyProfile.load(self.strategy_profile) if self.strategy_profile.exists() else None
        # Pull equity-derived per-trade cap from daily_target_state.json when
        # neither explicit JPY nor pct arguments were supplied. This is the
        # day's risk budget already divided by the target trade pace, so each
        # generated lane sizes against one shot — not the whole day. No JPY
        # literal fallback (§3.5).
        default_cap = _per_trade_risk_from_state()
        max_loss_jpy = resolve_max_loss_jpy(
            max_loss_jpy=self.max_loss_jpy,
            max_loss_pct=self.max_loss_pct,
            equity_jpy=self.risk_equity_jpy,
            default_max_loss_jpy=default_cap,
            label="generate-intents risk cap",
        )
        # Whole-day cap on `open_risk + candidate_risk` (portfolio cap).
        # Distinct from `max_loss_jpy` (per-trade cap). Per AGENT_CONTRACT §3.5
        # these caps must not be conflated: feeding the per-trade slice as the
        # portfolio cap blocks every additional shot once one position opens,
        # because `open_risk` is already on that single-shot scale. None when
        # the ledger is missing — the validator skips the portfolio gate
        # rather than synthesizing a literal.
        portfolio_loss_cap = _daily_risk_budget_from_state()
        # Load ATR / regime per pair from pair_charts.json. None when the file
        # is missing — _build_for_lane will surface MISSING_ATR_DATA so the
        # operator sees that geometry was built without market context.
        pair_charts = _load_pair_charts(self.pair_charts_path)
        results: list[GeneratedIntent] = []
        for lane in lanes[:max_candidates]:
            results.append(
                self._build_for_lane(
                    lane,
                    snapshot,
                    strategy_profile,
                    max_loss_jpy=max_loss_jpy,
                    portfolio_loss_cap=portfolio_loss_cap,
                    pair_charts=pair_charts,
                )
            )
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
        *,
        max_loss_jpy: float,
        portfolio_loss_cap: float | None = None,
        pair_charts: dict[str, dict[str, Any]] | None = None,
    ) -> GeneratedIntent:
        lane_id = _lane_id(lane)
        pair = str(lane["pair"])
        direction = str(lane["direction"])
        if snapshot is None:
            return GeneratedIntent(
                lane_id=lane_id,
                status="NEEDS_BROKER_SNAPSHOT",
                intent=None,
                risk_metrics=None,
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
                risk_metrics=None,
                risk_allowed=False,
                risk_issues=({"code": "MISSING_QUOTE", "message": f"missing quote for {pair}", "severity": "BLOCK"},),
                strategy_issues=(),
                live_blockers=(f"snapshot has no quote for {pair}",),
                note="Cannot build priced intent without a live quote.",
            )
        atr_pips = _atr_pips_for(pair, pair_charts)
        intent = _intent_from_lane(
            lane, quote, snapshot, max_loss_jpy=max_loss_jpy, atr_pips=atr_pips
        )
        risk = RiskEngine(
            policy=RiskPolicy(
                block_new_entries_with_pending_entry_orders=False,
                allow_protected_trader_position_adds=True,
                max_portfolio_loss_jpy=portfolio_loss_cap,
            )
        ).validate(
            intent,
            snapshot,
            for_live_send=False,
        )
        strategy_issues = tuple(
            issue.__dict__ for issue in (strategy_profile.validate(intent, for_live_send=False) if strategy_profile else ())
        )
        live_strategy_issues = tuple(
            issue.__dict__ for issue in (strategy_profile.validate(intent, for_live_send=True) if strategy_profile else ())
        )
        live_blockers = tuple(issue["message"] for issue in live_strategy_issues if issue.get("severity") == "BLOCK")
        risk_issues = list(issue.__dict__ for issue in risk.issues)
        # Per AGENT_CONTRACT §3.5: surface MISSING_ATR_DATA as a BLOCK issue so
        # the operator sees that geometry was built without market context.
        # No silent fallback — the lane is blocked from going LIVE_READY until
        # pair_charts are refreshed.
        if atr_pips is None:
            risk_issues.append(
                {
                    "code": "MISSING_ATR_DATA",
                    "message": (
                        f"pair_charts.json has no atr_pips for {pair} {GEOMETRY_ATR_TIMEFRAME}; "
                        "geometry is using spread-only floor (no ATR scaling)."
                    ),
                    "severity": "BLOCK",
                }
            )
            # Force DRY_RUN_BLOCKED downstream.
            risk_allowed = False
        else:
            risk_allowed = risk.allowed
        risk_issues = tuple(risk_issues)
        if risk_allowed and not live_blockers:
            status = "LIVE_READY"
        elif risk_allowed:
            status = "DRY_RUN_PASSED"
        else:
            status = "DRY_RUN_BLOCKED"
        return GeneratedIntent(
            lane_id=lane_id,
            status=status,
            intent=_intent_to_json(intent),
            risk_metrics=asdict(risk.metrics) if risk.metrics else None,
            risk_allowed=risk_allowed,
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
            if item.risk_metrics:
                lines.append(
                    f"  - risk metrics: risk=`{item.risk_metrics['risk_jpy']:.1f} JPY` "
                    f"reward=`{item.risk_metrics['reward_jpy']:.1f} JPY` "
                    f"rr=`{item.risk_metrics['reward_risk']:.2f}` spread=`{item.risk_metrics['spread_pips']:.1f}pip`"
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


def _intent_from_lane(
    lane: dict[str, Any],
    quote: Quote,
    snapshot: BrokerSnapshot,
    *,
    max_loss_jpy: float,
    atr_pips: float | None = None,
) -> OrderIntent:
    pair = str(lane["pair"])
    side = Side.parse(str(lane["direction"]))
    method = TradeMethod.parse(str(lane["method"]))
    order_type = _order_type_for(method)
    target_reward_risk = _target_reward_risk(lane)
    entry, tp, sl = _geometry(
        pair, side, order_type, quote, reward_risk=target_reward_risk, atr_pips=atr_pips
    )
    units = _risk_budgeted_units(pair, entry, sl, max_loss_jpy=max_loss_jpy, snapshot=snapshot)
    thesis = f"{lane['desk']} {pair} {side.value} {method.value} {target_reward_risk:.2f}R: {lane['required_receipt']}"
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
        units=units,
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
            "target_reward_risk": target_reward_risk,
            "evidence_tail_jpy": lane.get("evidence_tail_jpy"),
            "evidence_best_jpy": lane.get("evidence_best_jpy"),
            "sizing_rule": f"floor units to the largest broker size under the {max_loss_jpy:.0f} JPY loss cap",
            "max_loss_jpy": max_loss_jpy,
        },
    )


def _order_type_for(method: TradeMethod) -> OrderType:
    if method == TradeMethod.RANGE_ROTATION:
        return OrderType.LIMIT
    return OrderType.STOP_ENTRY


def _geometry(
    pair: str,
    side: Side,
    order_type: OrderType,
    quote: Quote,
    *,
    reward_risk: float = 1.5,
    atr_pips: float | None = None,
) -> tuple[float, float, float]:
    """Build (entry, tp, sl) prices.

    Stop distance comes from market reality, not a fixed pip literal:
        stop_pips = max(atr_pips * GEOMETRY_ATR_MULT, spread_pips * GEOMETRY_SPREAD_FLOOR_MULT)

    When atr_pips is None (pair_charts missing), we fall back to a *spread-only*
    distance; the caller is responsible for emitting MISSING_ATR so the operator
    sees that geometry was built without the primary market input.
    """
    pip_factor = PIP_FACTORS[pair]
    pip = 1.0 / pip_factor
    spread_pips = abs(quote.ask - quote.bid) * pip_factor
    spread_floor = spread_pips * GEOMETRY_SPREAD_FLOOR_MULT
    if atr_pips is not None and atr_pips > 0:
        stop_pips = max(atr_pips * GEOMETRY_ATR_MULT, spread_floor)
    else:
        stop_pips = spread_floor
    reward_pips = stop_pips * reward_risk
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


def _target_reward_risk(lane: dict[str, Any]) -> float:
    try:
        value = float(lane.get("target_reward_risk") or 1.5)
    except (TypeError, ValueError):
        value = 1.5
    return round(min(8.0, max(1.2, value)), 2)


def _round_price(pair: str, value: float) -> float:
    return round(value, 3 if pair.endswith("_JPY") else 5)


def _risk_budgeted_units(pair: str, entry: float, sl: float, *, max_loss_jpy: float, snapshot: BrokerSnapshot) -> int:
    pip_factor = PIP_FACTORS[pair]
    stop_pips = abs(entry - sl) * pip_factor
    if stop_pips <= 0:
        return 1
    quote_to_jpy = _quote_to_jpy(pair, snapshot)
    if quote_to_jpy is None:
        return 1
    max_units = max_loss_jpy * pip_factor / (stop_pips * quote_to_jpy)
    if max_units >= 1000:
        return max(1000, int(max_units // 1000) * 1000)
    return max(1, int(max_units))


def _quote_to_jpy(pair: str, snapshot: BrokerSnapshot) -> float | None:
    quote_ccy = pair.split("_", 1)[1]
    if quote_ccy == "JPY":
        return 1.0
    conversion_quote = snapshot.quotes.get(f"{quote_ccy}_JPY")
    if conversion_quote is None:
        return None
    return max(conversion_quote.bid, conversion_quote.ask)


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
            owner=Owner(str(item.get("owner") or Owner.UNKNOWN.value)),
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
    account = _account_summary_from_payload(payload.get("account"))
    return BrokerSnapshot(
        fetched_at_utc=datetime.fromisoformat(fetched) if fetched else datetime.now(timezone.utc),
        positions=positions,
        orders=orders,
        quotes=quotes,
        account=account,
    )


def _account_summary_from_payload(payload: object):
    from quant_rabbit.models import AccountSummary

    if not isinstance(payload, dict):
        return None
    fetched = payload.get("fetched_at_utc")
    return AccountSummary(
        nav_jpy=float(payload.get("nav_jpy") or 0.0),
        balance_jpy=float(payload.get("balance_jpy") or 0.0),
        unrealized_pl_jpy=float(payload.get("unrealized_pl_jpy") or 0.0),
        margin_used_jpy=float(payload.get("margin_used_jpy") or 0.0),
        margin_available_jpy=float(payload.get("margin_available_jpy") or 0.0),
        pl_jpy=float(payload.get("pl_jpy") or 0.0),
        financing_jpy=float(payload.get("financing_jpy") or 0.0),
        last_transaction_id=str(payload.get("last_transaction_id") or ""),
        fetched_at_utc=(
            datetime.fromisoformat(fetched) if isinstance(fetched, str) and fetched else datetime.now(timezone.utc)
        ),
    )

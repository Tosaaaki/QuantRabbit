from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.models import TradeMethod
from quant_rabbit.paths import (
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_CAMPAIGN_REPORT,
    DEFAULT_MARKET_STORY_PROFILE,
    DEFAULT_STRATEGY_PROFILE,
)
from quant_rabbit.risk import RiskPolicy


DESK_METHODS: dict[str, TradeMethod] = {
    "trend_trader": TradeMethod.TREND_CONTINUATION,
    "range_trader": TradeMethod.RANGE_ROTATION,
    "failure_trader": TradeMethod.BREAKOUT_FAILURE,
    "event_risk_trader": TradeMethod.EVENT_RISK,
    "position_manager": TradeMethod.POSITION_MANAGEMENT,
}


@dataclass(frozen=True)
class StrategyEvidence:
    pair: str
    direction: str
    status: str
    required_fix: str
    pretrade_net_jpy: float = 0.0
    live_net_jpy: float = 0.0
    live_worst_jpy: float | None = None
    positive_evidence_n: int = 0
    positive_best_jpy: float = 0.0
    positive_tail_jpy: float = 0.0
    target_reward_risk: float = 1.5
    seat_missed: int = 0
    seat_captured: int = 0

    @property
    def key(self) -> str:
        return f"{self.pair} {self.direction}"


@dataclass(frozen=True)
class MarketStoryEvidence:
    pair: str
    methods: dict[str, int]
    themes: dict[str, int]
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeskLane:
    desk: str
    pair: str
    direction: str
    method: str
    adoption: str
    campaign_role: str
    reason: str
    required_receipt: str
    target_reward_risk: float = 1.5
    evidence_tail_jpy: float = 0.0
    evidence_best_jpy: float = 0.0
    blockers: tuple[str, ...] = ()
    story_examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class DailyCampaignPlan:
    generated_at_utc: str
    start_balance_jpy: float
    target_jpy: float
    target_return_pct: float
    lanes: tuple[DeskLane, ...]
    director_verdict: str
    coverage_gap: str
    operating_rule: str
    loss_cap_jpy: float
    loss_cap_source: str


@dataclass(frozen=True)
class CampaignSummary:
    report_path: Path
    plan_path: Path
    target_jpy: float
    lanes: int
    actionable_lanes: int
    rejected_lanes: int


class CampaignPlanner:
    def __init__(
        self,
        *,
        strategy_profile: Path = DEFAULT_STRATEGY_PROFILE,
        market_story_profile: Path = DEFAULT_MARKET_STORY_PROFILE,
        report_path: Path = DEFAULT_CAMPAIGN_REPORT,
        plan_path: Path = DEFAULT_CAMPAIGN_PLAN,
    ) -> None:
        self.strategy_profile = strategy_profile
        self.market_story_profile = market_story_profile
        self.report_path = report_path
        self.plan_path = plan_path

    def run(self, *, start_balance_jpy: float, target_return_pct: float = 10.0) -> CampaignSummary:
        strategies = _load_strategy_profile(self.strategy_profile)
        loss_cap_jpy, loss_cap_source = _load_strategy_loss_cap(self.strategy_profile)
        stories = _load_market_story_profile(self.market_story_profile)
        lanes = tuple(self._build_lanes(strategies, stories, loss_cap_jpy=loss_cap_jpy))
        target_jpy = round(start_balance_jpy * (target_return_pct / 100.0), 2)
        plan = DailyCampaignPlan(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            start_balance_jpy=start_balance_jpy,
            target_jpy=target_jpy,
            target_return_pct=target_return_pct,
            lanes=lanes,
            director_verdict=_director_verdict(lanes),
            coverage_gap=_coverage_gap(lanes, target_jpy),
            operating_rule=(
                "The 10% number is a campaign target, not a profit guarantee. "
                f"No lane becomes live without current market_context, entry, TP, SL, <={loss_cap_jpy:.0f} JPY risk, and >=1.2R."
            ),
            loss_cap_jpy=loss_cap_jpy,
            loss_cap_source=loss_cap_source,
        )
        self._write_plan(plan)
        self._write_report(plan)
        return CampaignSummary(
            report_path=self.report_path,
            plan_path=self.plan_path,
            target_jpy=target_jpy,
            lanes=len(lanes),
            actionable_lanes=sum(1 for lane in lanes if lane.adoption not in {"REJECTED", "RISK_OVERLAY"}),
            rejected_lanes=sum(1 for lane in lanes if lane.adoption == "REJECTED"),
        )

    def _build_lanes(
        self,
        strategies: dict[str, list[StrategyEvidence]],
        stories: dict[str, MarketStoryEvidence],
        *,
        loss_cap_jpy: float,
    ) -> list[DeskLane]:
        lanes: list[DeskLane] = []
        for desk, method in DESK_METHODS.items():
            for pair, story in sorted(stories.items()):
                method_count = story.methods.get(method.value, 0)
                if method_count <= 0:
                    continue
                if desk in {"event_risk_trader", "position_manager"}:
                    lanes.append(_overlay_lane(desk, pair, method, story))
                    continue
                for strategy in strategies.get(pair, []):
                    lanes.append(_strategy_lane(desk, method, story, strategy, loss_cap_jpy=loss_cap_jpy))
        return sorted(lanes, key=_lane_sort_key)

    def _write_plan(self, plan: DailyCampaignPlan) -> None:
        self.plan_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(plan)
        self.plan_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, plan: DailyCampaignPlan) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Daily 10% Campaign Plan",
            "",
            f"- Generated at UTC: `{plan.generated_at_utc}`",
            f"- Start balance: `{plan.start_balance_jpy:.0f} JPY`",
            f"- Target return: `{plan.target_return_pct:.1f}%`",
            f"- Target JPY: `{plan.target_jpy:.0f} JPY`",
            f"- Per-trade loss cap: `{plan.loss_cap_jpy:.0f} JPY` (`{plan.loss_cap_source}`)",
            f"- Director verdict: `{plan.director_verdict}`",
            f"- Coverage gap: {plan.coverage_gap}",
            "",
            "## Operating Rule",
            "",
            plan.operating_rule,
            "",
            "## Trader Desks",
            "",
        ]
        for lane in plan.lanes:
            lines.append(
                f"- `{lane.desk}` `{lane.pair} {lane.direction}` method=`{lane.method}` "
                f"adoption=`{lane.adoption}` role=`{lane.campaign_role}` target_rr=`{lane.target_reward_risk:.2f}`"
            )
            lines.append(f"  - reason: {lane.reason}")
            lines.append(f"  - receipt: {lane.required_receipt}")
            for blocker in lane.blockers:
                lines.append(f"  - blocker: {blocker}")
            for example in lane.story_examples[:2]:
                lines.append(f"  - story: {example}")
        lines.extend(
            [
                "",
                "## Portfolio Director Rules",
                "",
                "- Multiple desks may propose at the same time, but one pair/direction cannot bypass the risk gateway through a different desk name.",
                "- The campaign target requires a lane ladder: NOW, BACKUP, RUNNER/ADD, and explicit blockers for missing coverage.",
                "- Event and position-management desks can veto or shrink every other desk when liquidity, news, margin, or open-risk story dominates.",
                "- If no desk can produce current order-intent receipts, the correct output is `coverage gap remains`, not forced trading.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _strategy_lane(
    desk: str,
    method: TradeMethod,
    story: MarketStoryEvidence,
    strategy: StrategyEvidence,
    *,
    loss_cap_jpy: float,
) -> DeskLane:
    adoption, campaign_role, receipt, blockers = _adoption_for_strategy(strategy, method, loss_cap_jpy=loss_cap_jpy)
    target_rr = _lane_target_reward_risk(strategy, adoption, loss_cap_jpy=loss_cap_jpy)
    reason = (
        f"{strategy.status}; pretrade_net={strategy.pretrade_net_jpy:.1f}, "
        f"live_net={strategy.live_net_jpy:.1f}, worst={strategy.live_worst_jpy}; "
        f"positive_tail={strategy.positive_tail_jpy:.1f}, target_rr={target_rr:.2f}; "
        f"story method pressure={story.methods.get(method.value, 0)}"
    )
    if adoption in {"ORDER_INTENT_REQUIRED", "RISK_REPAIR_DRY_RUN", "TRIGGER_RECEIPT_REQUIRED"} and target_rr >= 3.0:
        campaign_role = f"{campaign_role}_RUNNER"
        receipt = f"{receipt} Runner TP should target about {target_rr:.2f}R only when current tape supports the hold."
    return DeskLane(
        desk=desk,
        pair=strategy.pair,
        direction=strategy.direction,
        method=method.value,
        adoption=adoption,
        campaign_role=campaign_role,
        reason=reason,
        required_receipt=receipt,
        target_reward_risk=target_rr,
        evidence_tail_jpy=strategy.positive_tail_jpy,
        evidence_best_jpy=strategy.positive_best_jpy,
        blockers=blockers,
        story_examples=story.examples[:2],
    )


def _overlay_lane(desk: str, pair: str, method: TradeMethod, story: MarketStoryEvidence) -> DeskLane:
    theme_text = ", ".join(f"{key}={value}" for key, value in sorted(story.themes.items(), key=lambda item: -item[1])[:4])
    return DeskLane(
        desk=desk,
        pair=pair,
        direction="BOTH",
        method=method.value,
        adoption="RISK_OVERLAY",
        campaign_role="VETO_OR_RESIZE",
        reason=f"overlay from market story themes: {theme_text}",
        required_receipt="Any other desk must name how this overlay changes size, timing, stop, or pass decision.",
        target_reward_risk=1.0,
        story_examples=story.examples[:2],
    )


def _adoption_for_strategy(
    strategy: StrategyEvidence,
    method: TradeMethod,
    *,
    loss_cap_jpy: float,
) -> tuple[str, str, str, tuple[str, ...]]:
    cap_text = f"{loss_cap_jpy:.0f} JPY"
    if strategy.status == "CANDIDATE":
        return (
            "ORDER_INTENT_REQUIRED",
            "NOW_OR_BACKUP",
            f"Create a current order intent with market_context, entry, TP, SL, risk <={cap_text}, and >=1.2R.",
            (),
        )
    if strategy.status == "RISK_REPAIR_CANDIDATE":
        return (
            "RISK_REPAIR_DRY_RUN",
            "NOW_IF_REPAIRED",
            f"Produce a dry-run receipt proving old edge survives with <={cap_text} risk before live use.",
            ("old sizing broke the loss cap", strategy.required_fix),
        )
    if strategy.status == "MINE_MISSED_EDGE":
        receipt = "Arm only a trigger/pending-entry receipt; no market chase."
        if method == TradeMethod.RANGE_ROTATION:
            receipt = "Use exact rail/box order intent only; missed move is not participation."
        return (
            "TRIGGER_RECEIPT_REQUIRED",
            "BACKUP_OR_RELOAD",
            receipt,
            (strategy.required_fix,),
        )
    if strategy.status == "WATCH_ONLY":
        return (
            "WATCH_ONLY",
            "OBSERVE",
            "Wait for more evidence; do not count this toward 10% coverage.",
            (strategy.required_fix,),
        )
    return (
        "REJECTED",
        "NONE",
        "Do not trade until new evidence changes the strategy profile.",
        (strategy.required_fix,),
    )


def _director_verdict(lanes: tuple[DeskLane, ...]) -> str:
    if any(lane.adoption == "ORDER_INTENT_REQUIRED" for lane in lanes):
        return "BUILD_ORDER_INTENTS"
    if any(lane.adoption in {"RISK_REPAIR_DRY_RUN", "TRIGGER_RECEIPT_REQUIRED"} for lane in lanes):
        return "REPAIR_BEFORE_LIVE"
    return "NO_LIVE_CAMPAIGN_YET"


def _coverage_gap(lanes: tuple[DeskLane, ...], target_jpy: float) -> str:
    live_ready = [lane for lane in lanes if lane.adoption == "ORDER_INTENT_REQUIRED"]
    repair = [lane for lane in lanes if lane.adoption in {"RISK_REPAIR_DRY_RUN", "TRIGGER_RECEIPT_REQUIRED"}]
    if live_ready:
        return (
            f"{len(live_ready)} lanes can attempt receipts, but target {target_jpy:.0f} JPY still requires "
            "actual TP/SL geometry and current tape confirmation."
        )
    if repair:
        return (
            f"Target {target_jpy:.0f} JPY has no live-ready coverage yet; "
            f"{len(repair)} lanes need risk repair or trigger receipts first."
        )
    return f"Target {target_jpy:.0f} JPY has no evidence-backed coverage yet."


def _lane_sort_key(lane: DeskLane) -> tuple[int, str, str, str]:
    rank = {
        "ORDER_INTENT_REQUIRED": 0,
        "RISK_REPAIR_DRY_RUN": 1,
        "TRIGGER_RECEIPT_REQUIRED": 2,
        "RISK_OVERLAY": 3,
        "WATCH_ONLY": 4,
        "REJECTED": 5,
    }.get(lane.adoption, 9)
    return (rank, lane.desk, lane.pair, lane.direction)


def _load_strategy_profile(path: Path) -> dict[str, list[StrategyEvidence]]:
    payload = json.loads(path.read_text())
    by_pair: dict[str, list[StrategyEvidence]] = {}
    for item in payload.get("profiles", []):
        if not isinstance(item, dict):
            continue
        pair = str(item.get("pair") or "")
        direction = str(item.get("direction") or "")
        if not pair or not direction:
            continue
        evidence = StrategyEvidence(
            pair=pair,
            direction=direction,
            status=str(item.get("status") or "WATCH_ONLY"),
            required_fix=str(item.get("required_fix") or ""),
            pretrade_net_jpy=float(item.get("pretrade_net_jpy") or 0.0),
            live_net_jpy=float(item.get("live_net_jpy") or 0.0),
            live_worst_jpy=_optional_float(item.get("live_worst_jpy")),
            positive_evidence_n=int(item.get("positive_evidence_n") or 0),
            positive_best_jpy=float(item.get("positive_best_jpy") or 0.0),
            positive_tail_jpy=float(item.get("positive_tail_jpy") or 0.0),
            target_reward_risk=float(item.get("target_reward_risk") or 1.5),
            seat_missed=int(item.get("seat_missed") or 0),
            seat_captured=int(item.get("seat_captured") or 0),
        )
        by_pair.setdefault(pair, []).append(evidence)
    for values in by_pair.values():
        values.sort(key=lambda item: (item.status, item.direction))
    return by_pair


def _load_strategy_loss_cap(path: Path) -> tuple[float, str]:
    payload = json.loads(path.read_text())
    contract = payload.get("system_contract", {}) if isinstance(payload, dict) else {}
    raw = contract.get("loss_cap_jpy") if isinstance(contract, dict) else None
    try:
        cap = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        cap = 0.0
    if cap > 0:
        source = str(contract.get("loss_cap_source") or "strategy profile system_contract")
        return round(cap, 4), source
    policy_cap = RiskPolicy().max_loss_jpy
    if policy_cap is None or policy_cap <= 0:
        raise ValueError(f"strategy profile has no usable loss cap: {path}")
    return round(float(policy_cap), 4), "RiskPolicy.max_loss_jpy library default"


def _load_market_story_profile(path: Path) -> dict[str, MarketStoryEvidence]:
    payload = json.loads(path.read_text())
    stories: dict[str, MarketStoryEvidence] = {}
    for item in payload.get("pair_profiles", []):
        if not isinstance(item, dict):
            continue
        pair = str(item.get("pair") or "")
        if not pair:
            continue
        stories[pair] = MarketStoryEvidence(
            pair=pair,
            methods={str(key): int(value) for key, value in dict(item.get("methods") or {}).items()},
            themes={str(key): int(value) for key, value in dict(item.get("themes") or {}).items()},
            examples=tuple(str(example) for example in item.get("examples", [])[:5]),
        )
    return stories


def _optional_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _lane_target_reward_risk(strategy: StrategyEvidence, adoption: str, *, loss_cap_jpy: float) -> float:
    if adoption in {"WATCH_ONLY", "REJECTED", "RISK_OVERLAY"}:
        return 1.0
    evidence_rr = float(strategy.target_reward_risk or 1.5)
    if strategy.positive_tail_jpy > 0:
        evidence_rr = max(evidence_rr, strategy.positive_tail_jpy / loss_cap_jpy)
    if strategy.positive_best_jpy > 0:
        evidence_rr = max(evidence_rr, strategy.positive_best_jpy / loss_cap_jpy)
    return round(min(8.0, max(1.5, evidence_rr)), 2)

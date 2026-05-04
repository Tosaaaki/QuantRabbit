from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_COVERAGE_OPTIMIZATION,
    DEFAULT_COVERAGE_OPTIMIZATION_REPORT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_REPLAY_BACKTEST,
)


@dataclass(frozen=True)
class CoverageLane:
    lane_id: str
    status: str
    pair: str
    direction: str
    order_type: str
    units: int
    risk_jpy: float
    reward_jpy: float
    reward_risk: float
    counts_live_ready: bool
    counts_after_promotion: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class CoverageOptimizationSummary:
    output_path: Path
    report_path: Path
    status: str
    remaining_target_jpy: float
    live_ready_reward_jpy: float
    potential_reward_jpy: float
    live_ready_lanes: int
    promotion_candidate_lanes: int
    action_items: int
    sequential_ladder_reward_jpy: float = 0.0
    sequential_ladder_steps: int = 0


class CoverageOptimizer:
    """Turn campaign pressure into executable coverage and explicit gap receipts."""

    def __init__(
        self,
        *,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        replay_path: Path = DEFAULT_REPLAY_BACKTEST,
        output_path: Path = DEFAULT_COVERAGE_OPTIMIZATION,
        report_path: Path = DEFAULT_COVERAGE_OPTIMIZATION_REPORT,
    ) -> None:
        self.intents_path = intents_path
        self.target_state_path = target_state_path
        self.replay_path = replay_path
        self.output_path = output_path
        self.report_path = report_path

    def run(self) -> CoverageOptimizationSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        intents = _load_json(self.intents_path)
        target = _load_json(self.target_state_path)
        replay = _load_json(self.replay_path) if self.replay_path.exists() else {}
        lanes = tuple(_coverage_lane(item) for item in intents.get("results", []) or [] if _has_intent(item))
        remaining_target = _remaining_target(target)
        remaining_risk_budget = _remaining_risk_budget(target)
        live_ready_reward = _round(sum(lane.reward_jpy for lane in lanes if lane.counts_live_ready))
        potential_reward = _round(sum(lane.reward_jpy for lane in lanes if lane.counts_live_ready or lane.counts_after_promotion))
        live_ready_risk = _round(sum(lane.risk_jpy for lane in lanes if lane.counts_live_ready))
        sequential_ladder = _sequential_ladder(lanes, remaining_target, remaining_risk_budget)
        replay_gap = _replay_gap(replay)
        blockers = tuple(
            _blockers(
                target=target,
                remaining_target=remaining_target,
                remaining_risk_budget=remaining_risk_budget,
                live_ready_reward=live_ready_reward,
                potential_reward=potential_reward,
                live_ready_risk=live_ready_risk,
                sequential_ladder=sequential_ladder,
                lanes=lanes,
                replay_gap=replay_gap,
            )
        )
        action_items = tuple(
            _action_items(
                remaining_target=remaining_target,
                live_ready_reward=live_ready_reward,
                potential_reward=potential_reward,
                live_ready_risk=live_ready_risk,
                remaining_risk_budget=remaining_risk_budget,
                sequential_ladder=sequential_ladder,
                lanes=lanes,
                replay_gap=replay_gap,
            )
        )
        status = _status(
            target=target,
            blockers=blockers,
            remaining_target=remaining_target,
            live_ready_reward=live_ready_reward,
            potential_reward=potential_reward,
        )
        payload = {
            "generated_at_utc": generated_at,
            "status": status,
            "intents_path": str(self.intents_path),
            "target_state_path": str(self.target_state_path),
            "replay_path": str(self.replay_path) if self.replay_path.exists() else None,
            "remaining_target_jpy": remaining_target,
            "remaining_risk_budget_jpy": remaining_risk_budget,
            "live_ready_reward_jpy": live_ready_reward,
            "live_ready_risk_jpy": live_ready_risk,
            "potential_reward_jpy": potential_reward,
            "sequential_ladder_reward_jpy": sequential_ladder["reward_jpy"],
            "sequential_ladder_risk_jpy": sequential_ladder["risk_jpy"],
            "sequential_ladder_steps": sequential_ladder["steps"],
            "sequential_ladder_lane_ids": sequential_ladder["lane_ids"],
            "coverage_pct": _round((live_ready_reward / remaining_target) * 100.0) if remaining_target else 100.0,
            "sequential_ladder_coverage_pct": _round((sequential_ladder["reward_jpy"] / remaining_target) * 100.0)
            if remaining_target
            else 100.0,
            "potential_coverage_pct": _round((potential_reward / remaining_target) * 100.0) if remaining_target else 100.0,
            "replay_gap": replay_gap,
            "blockers": list(blockers),
            "action_items": list(action_items),
            "lanes": [asdict(lane) for lane in lanes],
        }
        self._write_output(payload)
        self._write_report(payload)
        return CoverageOptimizationSummary(
            output_path=self.output_path,
            report_path=self.report_path,
            status=status,
            remaining_target_jpy=remaining_target,
            live_ready_reward_jpy=live_ready_reward,
            potential_reward_jpy=potential_reward,
            live_ready_lanes=sum(1 for lane in lanes if lane.counts_live_ready),
            promotion_candidate_lanes=sum(1 for lane in lanes if lane.counts_after_promotion),
            action_items=len(action_items),
            sequential_ladder_reward_jpy=sequential_ladder["reward_jpy"],
            sequential_ladder_steps=int(sequential_ladder["steps"]),
        )

    def _write_output(self, payload: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, payload: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Coverage Optimization Report",
            "",
            f"- Generated at UTC: `{payload['generated_at_utc']}`",
            f"- Status: `{payload['status']}`",
            f"- Remaining target: `{payload['remaining_target_jpy']:.0f} JPY`",
            f"- Live-ready reward: `{payload['live_ready_reward_jpy']:.0f} JPY` (`{payload['coverage_pct']:.1f}%`)",
            f"- Sequential ladder reward: `{payload['sequential_ladder_reward_jpy']:.0f} JPY` "
            f"(`{payload['sequential_ladder_coverage_pct']:.1f}%`, steps=`{payload['sequential_ladder_steps']}`)",
            f"- Potential reward after promotions: `{payload['potential_reward_jpy']:.0f} JPY` (`{payload['potential_coverage_pct']:.1f}%`)",
            f"- Remaining risk budget: `{payload['remaining_risk_budget_jpy']:.0f} JPY`",
            "",
            "## Blockers",
            "",
        ]
        if payload["blockers"]:
            lines.extend(f"- {item}" for item in payload["blockers"])
        else:
            lines.append("- none")
        lines.extend(["", "## Action Items", ""])
        if payload["action_items"]:
            lines.extend(f"- {item}" for item in payload["action_items"])
        else:
            lines.append("- none")
        lines.extend(["", "## Lanes", ""])
        for lane in payload["lanes"]:
            lines.append(
                f"- `{lane['lane_id']}` status=`{lane['status']}` reward=`{lane['reward_jpy']:.0f}` "
                f"risk=`{lane['risk_jpy']:.0f}` rr=`{lane['reward_risk']:.2f}` "
                f"live_ready=`{lane['counts_live_ready']}` promotion_candidate=`{lane['counts_after_promotion']}`"
            )
            for blocker in lane["blockers"][:3]:
                lines.append(f"  - blocker: {blocker}")
        lines.extend(
            [
                "",
                "## Coverage Contract",
                "",
                "- Coverage is executable reward from current receipts, not a profit guarantee.",
                "- `DRY_RUN_PASSED` lanes count only as potential coverage until strategy blockers are promoted by receipts.",
                "- A target gap remains a product blocker until it is closed by live-ready, risk-valid lanes or a no-market gap receipt.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _coverage_lane(result: dict[str, Any]) -> CoverageLane:
    intent = result["intent"]
    pair = str(intent.get("pair") or "")
    direction = str(intent.get("side") or "")
    units = int(intent.get("units") or 0)
    risk_jpy, reward_jpy, reward_risk = _risk_reward_from_result(result)
    blockers = tuple(_result_blockers(result))
    status = str(result.get("status") or "")
    return CoverageLane(
        lane_id=str(result.get("lane_id") or ""),
        status=status,
        pair=pair,
        direction=direction,
        order_type=str(intent.get("order_type") or ""),
        units=units,
        risk_jpy=risk_jpy,
        reward_jpy=reward_jpy,
        reward_risk=reward_risk,
        counts_live_ready=status == "LIVE_READY" and not blockers,
        counts_after_promotion=status == "DRY_RUN_PASSED" and not _has_risk_block(result),
        blockers=blockers,
    )


def _result_blockers(result: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    metrics = result.get("risk_metrics") if isinstance(result.get("risk_metrics"), dict) else {}
    required_metrics = ("risk_jpy", "reward_jpy", "reward_risk")
    if not all(_optional_float(metrics.get(key)) is not None for key in required_metrics):
        blockers.append("broker-truth risk metrics are missing; rerun generate-intents with a current broker snapshot")
    for issue in result.get("risk_issues", []) or []:
        if isinstance(issue, dict) and issue.get("severity") == "BLOCK":
            blockers.append(str(issue.get("message") or issue.get("code") or "risk block"))
    for issue in result.get("strategy_issues", []) or []:
        if isinstance(issue, dict) and issue.get("severity") == "BLOCK":
            blockers.append(str(issue.get("message") or issue.get("code") or "strategy block"))
    blockers.extend(str(item) for item in result.get("live_blockers", []) or [])
    return blockers


def _risk_reward_from_result(result: dict[str, Any]) -> tuple[float, float, float]:
    metrics = result.get("risk_metrics") if isinstance(result.get("risk_metrics"), dict) else {}
    risk_jpy = _optional_float(metrics.get("risk_jpy"))
    reward_jpy = _optional_float(metrics.get("reward_jpy"))
    reward_risk = _optional_float(metrics.get("reward_risk"))
    if risk_jpy is not None and reward_jpy is not None and reward_risk is not None:
        return _round(risk_jpy), _round(reward_jpy), _round(reward_risk)
    return 0.0, 0.0, 0.0


def _blockers(
    *,
    target: dict[str, Any],
    remaining_target: float,
    remaining_risk_budget: float,
    live_ready_reward: float,
    potential_reward: float,
    live_ready_risk: float,
    sequential_ladder: dict[str, Any],
    lanes: tuple[CoverageLane, ...],
    replay_gap: str | None,
) -> list[str]:
    blockers: list[str] = []
    if not target:
        blockers.append("daily target state is missing; run daily-target-state or plan-campaign")
    if target.get("status") == "REPAIR_REQUIRED":
        blockers.append("daily target ledger requires protection repair before fresh risk")
    if remaining_risk_budget <= 0 and remaining_target > 0:
        blockers.append("remaining risk budget is unavailable")
    if live_ready_reward < remaining_target:
        blockers.append(f"live-ready reward misses remaining target by {remaining_target - live_ready_reward:.0f} JPY")
    if potential_reward < remaining_target:
        blockers.append(f"even promoted dry-run reward misses remaining target by {remaining_target - potential_reward:.0f} JPY")
    if remaining_target > 0 and not any(lane.counts_live_ready for lane in lanes):
        blockers.append("no LIVE_READY lanes exist")
    max_lane_risk = max((lane.risk_jpy for lane in lanes if lane.counts_live_ready), default=0.0)
    if remaining_target > 0 and max_lane_risk > remaining_risk_budget > 0:
        blockers.append("at least one live-ready lane exceeds the remaining per-entry risk budget")
    elif (
        remaining_target > 0
        and live_ready_risk > remaining_risk_budget > 0
        and float(sequential_ladder.get("reward_jpy") or 0.0) < remaining_target
    ):
        blockers.append("live-ready ladder risk exceeds remaining daily risk budget and sequential coverage still misses target")
    if remaining_target > 0 and replay_gap:
        blockers.append(replay_gap)
    return blockers


def _action_items(
    *,
    remaining_target: float,
    live_ready_reward: float,
    potential_reward: float,
    live_ready_risk: float,
    remaining_risk_budget: float,
    sequential_ladder: dict[str, Any],
    lanes: tuple[CoverageLane, ...],
    replay_gap: str | None,
) -> list[str]:
    items: list[str] = []
    promotion_candidates = [lane for lane in lanes if lane.counts_after_promotion]
    if promotion_candidates:
        items.append(f"promote {len(promotion_candidates)} dry-run receipts only after their strategy blockers clear")
    if live_ready_reward < remaining_target:
        reward_gap = remaining_target - live_ready_reward
        avg_reward = _average_reward(lanes) or 1.0
        items.append(f"build at least {math.ceil(reward_gap / avg_reward)} additional live-ready trigger receipts")
    if potential_reward < remaining_target:
        items.append("expand lane generation across timing windows or pairs; current repaired ladder cannot cover target")
    if live_ready_risk > remaining_risk_budget > 0 and float(sequential_ladder.get("reward_jpy") or 0.0) >= remaining_target:
        items.append("execute coverage as a sequential ladder; do not deploy all live-ready lanes as simultaneous exposure")
    blocked_pairs = sorted({lane.pair for lane in lanes if lane.blockers and not lane.counts_live_ready})
    if blocked_pairs:
        items.append(f"repair blockers for: {', '.join(blocked_pairs[:8])}")
    if replay_gap:
        items.append("rerun replay/backtest after coverage changes and keep gap reasons as product blockers")
    return items


def _status(
    *,
    target: dict[str, Any],
    blockers: tuple[str, ...],
    remaining_target: float,
    live_ready_reward: float,
    potential_reward: float,
) -> str:
    if not target:
        return "COVERAGE_GAP"
    if remaining_target <= 0:
        return "TARGET_REACHED_PROTECT"
    hard = [
        item
        for item in blockers
        if not item.startswith("live-ready reward misses") and item != "no LIVE_READY lanes exist"
    ]
    if live_ready_reward >= remaining_target and not hard:
        return "LIVE_READY_COVERAGE_READY"
    if (live_ready_reward >= remaining_target or potential_reward >= remaining_target) and _has_replay_evidence_gap(hard):
        return "COVERAGE_REQUIRES_REPLAY_EVIDENCE"
    if potential_reward >= remaining_target:
        return "COVERAGE_REQUIRES_PROFILE_PROMOTION"
    return "COVERAGE_GAP"


def _remaining_target(target: dict[str, Any]) -> float:
    if not target:
        return 0.0
    return _round(float(target.get("remaining_target_jpy") or target.get("target_jpy") or 0.0))


def _remaining_risk_budget(target: dict[str, Any]) -> float:
    if not target:
        return 0.0
    return _round(float(target.get("remaining_risk_budget_jpy") or 0.0))


def _sequential_ladder(
    lanes: tuple[CoverageLane, ...],
    remaining_target: float,
    remaining_risk_budget: float,
) -> dict[str, Any]:
    selected: list[CoverageLane] = []
    reward = 0.0
    risk = 0.0
    candidates = sorted(
        (lane for lane in lanes if lane.counts_live_ready and lane.risk_jpy > 0 and lane.reward_jpy > 0),
        key=lambda lane: (lane.reward_risk, lane.reward_jpy),
        reverse=True,
    )
    for lane in candidates:
        if remaining_risk_budget > 0 and lane.risk_jpy > remaining_risk_budget:
            continue
        selected.append(lane)
        reward += lane.reward_jpy
        risk = max(risk, lane.risk_jpy)
        if remaining_target > 0 and reward >= remaining_target:
            break
    return {
        "reward_jpy": _round(reward),
        "risk_jpy": _round(risk),
        "steps": len(selected),
        "lane_ids": [lane.lane_id for lane in selected],
    }


def _replay_gap(replay: dict[str, Any]) -> str | None:
    summary = replay.get("summary") if isinstance(replay.get("summary"), dict) else {}
    days = int(summary.get("days") or 0)
    if not days:
        return None
    covered = int(summary.get("evidence_target_covered") or 0)
    if covered < days:
        return f"replay evidence covers target on {covered}/{days} days"
    return None


def _average_reward(lanes: tuple[CoverageLane, ...]) -> float:
    rewards = [lane.reward_jpy for lane in lanes if lane.reward_jpy > 0 and not lane.blockers]
    if not rewards:
        rewards = [lane.reward_jpy for lane in lanes if lane.reward_jpy > 0]
    return _round(sum(rewards) / len(rewards)) if rewards else 0.0


def _has_intent(item: object) -> bool:
    return isinstance(item, dict) and isinstance(item.get("intent"), dict)


def _has_risk_block(result: dict[str, Any]) -> bool:
    return any(
        isinstance(issue, dict) and issue.get("severity") == "BLOCK"
        for issue in result.get("risk_issues", []) or []
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _has_replay_evidence_gap(blockers: list[str]) -> bool:
    return any(item.startswith("replay evidence covers target") for item in blockers)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _round(value: float) -> float:
    return round(value, 4)

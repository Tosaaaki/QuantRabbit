from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from quant_rabbit.paths import (
    DEFAULT_AI_ATTACK_ADVICE,
    DEFAULT_AI_ATTACK_ADVICE_REPORT,
    DEFAULT_AI_TEST_BOT_BACKTEST,
    DEFAULT_COVERAGE_OPTIMIZATION,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_OUTCOME_MART,
    DEFAULT_ORDER_INTENTS,
)


# Advisory weights only. They rank already-LIVE_READY receipts for Codex
# attention; they never resize risk, bypass gateway validation, or create a
# production threshold. If these need tuning, replace them with an evaluated
# advisor backtest rather than treating them as risk constants.
ADVISORY_REWARD_COVERAGE_SCORE_CAP = 40.0
ADVISORY_REWARD_RISK_SCORE_MULT = 6.0
ADVISORY_POSITIVE_EDGE_SCORE = 25.0
ADVISORY_NEGATIVE_EDGE_PENALTY = 25.0
ADVISORY_OUTCOME_MART_EDGE_SCORE = 15.0
ADVISORY_OUTCOME_MART_NEGATIVE_PENALTY = 15.0
ADVISORY_OUTCOME_MART_MIN_TRIALS = 5
ADVISORY_MARKET_PARTICIPATION_SCORE = 5.0
REPORT_LANE_LIMIT = 12


@dataclass(frozen=True)
class AttackLaneAdvice:
    lane_id: str
    pair: str
    direction: str
    method: str
    order_type: str
    status: str
    reward_jpy: float
    risk_jpy: float
    reward_risk: float
    score: float
    historical_edge_jpy: float | None
    historical_edge_trades: int
    historical_edge_buckets: tuple[str, ...]
    archive_method_edge_jpy: float | None
    archive_method_avg_jpy: float | None
    archive_method_trials: int
    archive_method_key: str | None
    archive_condition_edge_jpy: float | None
    archive_condition_avg_jpy: float | None
    archive_condition_trials: int
    archive_condition_key: str | None
    rationale: tuple[str, ...]
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttackAdviceSummary:
    output_path: Path
    report_path: Path
    status: str
    live_ready_lanes: int
    recommended_now_lanes: int
    recommended_reward_jpy: float
    coverage_pct: float
    blockers: int


class AttackAdvisor:
    """Read-only attack governor for current LIVE_READY receipts.

    This component is intentionally not an executor. It turns current
    `order_intents` plus deduped AI-test-bot history into a ranked advisory
    packet that Codex automation may cite when choosing lanes. Broker writes
    still flow only through `gpt-trader-decision` and `LiveOrderGateway`.
    """

    def __init__(
        self,
        *,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        ai_backtest_path: Path = DEFAULT_AI_TEST_BOT_BACKTEST,
        outcome_mart_path: Path = DEFAULT_OUTCOME_MART,
        coverage_path: Path = DEFAULT_COVERAGE_OPTIMIZATION,
        output_path: Path = DEFAULT_AI_ATTACK_ADVICE,
        report_path: Path = DEFAULT_AI_ATTACK_ADVICE_REPORT,
    ) -> None:
        self.intents_path = intents_path
        self.target_state_path = target_state_path
        self.ai_backtest_path = ai_backtest_path
        self.outcome_mart_path = outcome_mart_path
        self.coverage_path = coverage_path
        self.output_path = output_path
        self.report_path = report_path

    def run(self) -> AttackAdviceSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        intents = _load_json(self.intents_path)
        target = _load_json(self.target_state_path)
        ai_backtest = _load_optional_json(self.ai_backtest_path)
        outcome_mart = _load_optional_json(self.outcome_mart_path)
        coverage = _load_optional_json(self.coverage_path)
        edge_index = _edge_index(ai_backtest or {})
        outcome_index = _outcome_index(outcome_mart or {})
        condition_index = _condition_index(outcome_mart or {})
        intent_session_bucket = _session_bucket_from_timestamp(intents.get("generated_at_utc"))
        remaining_target = _positive_float(target.get("remaining_target_jpy"))
        remaining_risk = _positive_float(target.get("remaining_risk_budget_jpy"))
        lanes = tuple(
            _attack_lane(
                item,
                edge_index=edge_index,
                outcome_index=outcome_index,
                condition_index=condition_index,
                intent_session_bucket=intent_session_bucket,
                remaining_target_jpy=remaining_target,
            )
            for item in intents.get("results", []) or []
            if isinstance(item, dict) and isinstance(item.get("intent"), dict)
        )
        live_ready = tuple(lane for lane in lanes if lane.status == "LIVE_READY" and not lane.blockers)
        ranked = tuple(sorted(live_ready, key=lambda lane: (lane.score, lane.reward_jpy, lane.lane_id), reverse=True))
        recommended = _recommend_now(ranked, remaining_risk_jpy=remaining_risk, remaining_target_jpy=remaining_target)
        watchlist = tuple(lane for lane in ranked if lane.lane_id not in {item.lane_id for item in recommended})[:REPORT_LANE_LIMIT]
        live_ready_reward = _round(sum(lane.reward_jpy for lane in _dedupe_geometry(live_ready)))
        recommended_reward = _round(sum(lane.reward_jpy for lane in recommended))
        recommended_risk = _round(sum(lane.risk_jpy for lane in recommended))
        coverage_pct = _round((live_ready_reward / remaining_target) * 100.0) if remaining_target else 0.0
        recommended_coverage_pct = _round((recommended_reward / remaining_target) * 100.0) if remaining_target else 0.0
        blockers = tuple(
            _blockers(
                intents=intents,
                target=target,
                remaining_target=remaining_target,
                live_ready=live_ready,
                recommended=recommended,
            )
        )
        action_items = tuple(
            _action_items(
                remaining_target=remaining_target,
                live_ready_reward=live_ready_reward,
                recommended_reward=recommended_reward,
                live_ready=live_ready,
                edge_index=edge_index,
                outcome_index=outcome_index,
                condition_index=condition_index,
                coverage=coverage or {},
            )
        )
        status = _status(blockers=blockers, recommended=recommended, remaining_target=remaining_target, recommended_reward=recommended_reward)
        payload = {
            "generated_at_utc": generated_at,
            "status": status,
            "read_only": True,
            "live_permission": False,
            "intents_path": str(self.intents_path),
            "target_state_path": str(self.target_state_path),
            "ai_backtest_path": str(self.ai_backtest_path) if self.ai_backtest_path.exists() else None,
            "outcome_mart_path": str(self.outcome_mart_path) if self.outcome_mart_path.exists() else None,
            "coverage_path": str(self.coverage_path) if self.coverage_path.exists() else None,
            "remaining_target_jpy": remaining_target,
            "remaining_risk_budget_jpy": remaining_risk,
            "live_ready_lanes": len(live_ready),
            "live_ready_reward_jpy": live_ready_reward,
            "coverage_pct": coverage_pct,
            "recommended_now_lane_ids": [lane.lane_id for lane in recommended],
            "recommended_now_reward_jpy": recommended_reward,
            "recommended_now_risk_jpy": recommended_risk,
            "recommended_now_coverage_pct": recommended_coverage_pct,
            "watchlist_lane_ids": [lane.lane_id for lane in watchlist],
            "required_additional_reward_jpy": _round(max((remaining_target or 0.0) - live_ready_reward, 0.0)),
            "required_additional_live_ready_lanes": _required_additional_lanes(
                remaining_target=remaining_target,
                live_ready_reward=live_ready_reward,
                live_ready=live_ready,
            ),
            "settings_advice": _settings_advice(),
            "blockers": list(blockers),
            "action_items": list(action_items),
            "lanes": [asdict(lane) for lane in ranked],
            "blocked_lanes": [asdict(lane) for lane in lanes if lane.blockers],
        }
        self._write_output(payload)
        self._write_report(payload)
        return AttackAdviceSummary(
            output_path=self.output_path,
            report_path=self.report_path,
            status=status,
            live_ready_lanes=len(live_ready),
            recommended_now_lanes=len(recommended),
            recommended_reward_jpy=recommended_reward,
            coverage_pct=coverage_pct,
            blockers=len(blockers),
        )

    def _write_output(self, payload: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, payload: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# AI Attack Advice Report",
            "",
            f"- Generated at UTC: `{payload['generated_at_utc']}`",
            f"- Status: `{payload['status']}`",
            f"- Read only: `{payload['read_only']}`",
            f"- Live permission: `{payload['live_permission']}`",
            f"- Live-ready lanes: `{payload['live_ready_lanes']}`",
            f"- Live-ready reward: `{payload['live_ready_reward_jpy']:.0f} JPY` (`{payload['coverage_pct']:.1f}%`)",
            f"- Recommended now: `{len(payload['recommended_now_lane_ids'])}` lanes, "
            f"reward=`{payload['recommended_now_reward_jpy']:.0f} JPY`, risk=`{payload['recommended_now_risk_jpy']:.0f} JPY`",
            f"- Required additional reward: `{payload['required_additional_reward_jpy']:.0f} JPY`",
            f"- Required additional live-ready lanes: `{payload['required_additional_live_ready_lanes']}`",
            "",
            "## Recommended Now",
            "",
        ]
        if payload["recommended_now_lane_ids"]:
            lane_map = {lane["lane_id"]: lane for lane in payload["lanes"]}
            for lane_id in payload["recommended_now_lane_ids"]:
                lane = lane_map[lane_id]
                lines.append(
                    f"- `{lane_id}` score=`{lane['score']:.1f}` reward=`{lane['reward_jpy']:.0f}` "
                    f"risk=`{lane['risk_jpy']:.0f}` rr=`{lane['reward_risk']:.2f}` "
                    f"hist_edge=`{lane['historical_edge_jpy']}` "
                    f"condition=`{lane['archive_condition_key']}` "
                    f"condition_edge=`{lane['archive_condition_edge_jpy']}` "
                    f"method_edge=`{lane['archive_method_edge_jpy']}`"
                )
        else:
            lines.append("- none")
        lines.extend(["", "## Watchlist", ""])
        if payload["watchlist_lane_ids"]:
            for lane_id in payload["watchlist_lane_ids"][:REPORT_LANE_LIMIT]:
                lines.append(f"- `{lane_id}`")
        else:
            lines.append("- none")
        lines.extend(["", "## Blockers", ""])
        if payload["blockers"]:
            lines.extend(f"- {item}" for item in payload["blockers"])
        else:
            lines.append("- none")
        lines.extend(["", "## Action Items", ""])
        if payload["action_items"]:
            lines.extend(f"- {item}" for item in payload["action_items"])
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "## Contract",
                "",
                "- This advice is read-only and never places, stages, or resizes broker orders.",
                "- `LiveOrderGateway` remains the final broker-truth and risk authority.",
                "- Do not raise loss caps from this report; fix coverage by adding validated lanes or improving execution evidence.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _attack_lane(
    item: dict[str, Any],
    *,
    edge_index: dict[tuple[str, str], dict[str, Any]],
    outcome_index: dict[tuple[str, str, str], dict[str, Any]],
    condition_index: dict[tuple[str, str, str, str], dict[str, Any]],
    intent_session_bucket: str,
    remaining_target_jpy: float | None,
) -> AttackLaneAdvice:
    intent = item["intent"]
    lane_id = str(item.get("lane_id") or "")
    pair = str(intent.get("pair") or "")
    direction = str(intent.get("side") or "")
    context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    method = str(context.get("method") or "")
    order_type = str(intent.get("order_type") or "")
    status = str(item.get("status") or "")
    metrics = item.get("risk_metrics") if isinstance(item.get("risk_metrics"), dict) else {}
    reward = _positive_float(metrics.get("reward_jpy"))
    risk = _positive_float(metrics.get("risk_jpy"))
    rr = _positive_float(metrics.get("reward_risk")) or 0.0
    blockers: list[str] = []
    if status != "LIVE_READY":
        blockers.append(f"status is {status}")
    if reward is None or risk is None:
        blockers.append("broker-truth reward/risk metrics are missing")
    edge = edge_index.get((pair, direction), {})
    edge_jpy = _optional_float(edge.get("managed_net_jpy"))
    edge_trades = int(edge.get("trades") or 0)
    archive_edge = _best_outcome_edge(outcome_index, pair=pair, direction=direction, method=method)
    archive_net_jpy = _optional_float(archive_edge.get("net_jpy")) if archive_edge else None
    archive_avg_jpy = _optional_float(archive_edge.get("avg_jpy")) if archive_edge else None
    archive_trials = int(archive_edge.get("outcome_n") or 0) if archive_edge else 0
    archive_key = str(archive_edge.get("key") or "") if archive_edge else None
    condition_edge = _best_condition_edge(
        condition_index,
        method=method,
        order_type=order_type,
        session=_first_condition_token(
            context.get("session_bucket"),
            metadata.get("session_bucket"),
            context.get("session"),
            intent_session_bucket,
        ),
        regime=_first_condition_token(
            metadata.get("regime_state"),
            context.get("regime_state"),
            context.get("regime"),
        ),
    )
    condition_net_jpy = _optional_float(condition_edge.get("net_jpy")) if condition_edge else None
    condition_avg_jpy = _optional_float(condition_edge.get("avg_jpy")) if condition_edge else None
    condition_trials = int(condition_edge.get("outcome_n") or 0) if condition_edge else 0
    condition_key = str(condition_edge.get("key") or "") if condition_edge else None
    rationale: list[str] = []
    score = 0.0
    if reward is not None:
        if remaining_target_jpy and remaining_target_jpy > 0:
            score += min(ADVISORY_REWARD_COVERAGE_SCORE_CAP, (reward / remaining_target_jpy) * 100.0)
            rationale.append("adds target coverage")
        else:
            rationale.append("target state missing or complete")
    if rr > 0:
        score += rr * ADVISORY_REWARD_RISK_SCORE_MULT
        rationale.append("reward/risk efficiency")
    if edge_jpy is not None and edge_jpy > 0:
        score += ADVISORY_POSITIVE_EDGE_SCORE
        rationale.append("positive AI-test-bot pair/direction edge")
    elif edge_jpy is not None and edge_jpy < 0:
        score -= ADVISORY_NEGATIVE_EDGE_PENALTY
        rationale.append("negative AI-test-bot pair/direction edge")
    condition_is_actionable = condition_trials >= ADVISORY_OUTCOME_MART_MIN_TRIALS
    if condition_net_jpy is not None and condition_is_actionable and condition_net_jpy > 0:
        score += ADVISORY_OUTCOME_MART_EDGE_SCORE
        rationale.append("positive archive condition edge")
    elif condition_net_jpy is not None and condition_is_actionable and condition_net_jpy < 0:
        score -= ADVISORY_OUTCOME_MART_NEGATIVE_PENALTY
        rationale.append("negative archive condition edge")
    elif condition_net_jpy is not None and not condition_is_actionable:
        rationale.append("archive condition edge below sample floor")
    if order_type == "MARKET":
        score += ADVISORY_MARKET_PARTICIPATION_SCORE
        rationale.append("immediate market participation")
    return AttackLaneAdvice(
        lane_id=lane_id,
        pair=pair,
        direction=direction,
        method=method,
        order_type=order_type,
        status=status,
        reward_jpy=_round(reward or 0.0),
        risk_jpy=_round(risk or 0.0),
        reward_risk=_round(rr),
        score=_round(score),
        historical_edge_jpy=_round(edge_jpy) if edge_jpy is not None else None,
        historical_edge_trades=edge_trades,
        historical_edge_buckets=tuple(str(bucket) for bucket in edge.get("buckets", ()) or ()),
        archive_method_edge_jpy=_round(archive_net_jpy) if archive_net_jpy is not None else None,
        archive_method_avg_jpy=_round(archive_avg_jpy) if archive_avg_jpy is not None else None,
        archive_method_trials=archive_trials,
        archive_method_key=archive_key,
        archive_condition_edge_jpy=_round(condition_net_jpy) if condition_net_jpy is not None else None,
        archive_condition_avg_jpy=_round(condition_avg_jpy) if condition_avg_jpy is not None else None,
        archive_condition_trials=condition_trials,
        archive_condition_key=condition_key,
        rationale=tuple(rationale),
        blockers=tuple(blockers),
    )


def _recommend_now(
    lanes: tuple[AttackLaneAdvice, ...],
    *,
    remaining_risk_jpy: float | None,
    remaining_target_jpy: float | None,
) -> tuple[AttackLaneAdvice, ...]:
    selected: list[AttackLaneAdvice] = []
    seen_geometry: set[tuple[str, str, str, float, float, float]] = set()
    risk_used = 0.0
    reward = 0.0
    for lane in lanes:
        geometry = _geometry_key(lane)
        if geometry in seen_geometry:
            continue
        if remaining_risk_jpy is not None and remaining_risk_jpy > 0 and risk_used + lane.risk_jpy > remaining_risk_jpy:
            continue
        selected.append(lane)
        seen_geometry.add(geometry)
        risk_used += lane.risk_jpy
        reward += lane.reward_jpy
        if remaining_target_jpy is not None and remaining_target_jpy > 0 and reward >= remaining_target_jpy:
            break
    return tuple(selected)


def _dedupe_geometry(lanes: tuple[AttackLaneAdvice, ...]) -> tuple[AttackLaneAdvice, ...]:
    selected: list[AttackLaneAdvice] = []
    seen: set[tuple[str, str, str, float, float, float]] = set()
    for lane in lanes:
        key = _geometry_key(lane)
        if key in seen:
            continue
        seen.add(key)
        selected.append(lane)
    return tuple(selected)


def _geometry_key(lane: AttackLaneAdvice) -> tuple[str, str, str, float, float, float]:
    # AttackAdvice does not need exact entry coordinates for the report; using
    # lane-level economics avoids double-counting identical method clones.
    return (lane.pair, lane.direction, lane.order_type, lane.reward_jpy, lane.risk_jpy, lane.reward_risk)


def _edge_index(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in payload.get("bucket_contributions", []) or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("bucket") or "")
        parts = label.split(":")
        if len(parts) < 3:
            continue
        key = (parts[1], parts[2])
        current = index.setdefault(key, {"managed_net_jpy": 0.0, "trades": 0, "buckets": []})
        current["managed_net_jpy"] = _round(float(current["managed_net_jpy"]) + float(item.get("managed_net_jpy") or 0.0))
        current["trades"] = int(current["trades"]) + int(item.get("trades") or 0)
        current["buckets"].append(label)
    return index


def _outcome_index(payload: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in payload.get("method_edges", []) or []:
        if not isinstance(item, dict):
            continue
        pair = str(item.get("pair") or "")
        direction = str(item.get("direction") or "")
        method = str(item.get("method") or "")
        if not pair or not direction or not method:
            continue
        index[(pair, direction, method)] = item
    return index


def _condition_index(payload: dict[str, Any]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for item in payload.get("condition_edges", []) or []:
        if not isinstance(item, dict):
            continue
        method = _condition_token(item.get("method"))
        order_type = _condition_token(item.get("order_type"))
        session = _condition_token(item.get("session_bucket"))
        regime = _condition_token(item.get("regime"))
        if not method:
            continue
        index[(method, order_type or "UNSPECIFIED", session or "UNSPECIFIED", regime or "UNSPECIFIED")] = item
    return index


def _best_outcome_edge(
    index: dict[tuple[str, str, str], dict[str, Any]],
    *,
    pair: str,
    direction: str,
    method: str,
) -> dict[str, Any]:
    exact = index.get((pair, direction, method))
    if exact:
        return exact
    return index.get((pair, direction, "UNSPECIFIED"), {})


def _best_condition_edge(
    index: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    method: str,
    order_type: str,
    session: str,
    regime: str,
) -> dict[str, Any]:
    method_key = _condition_token(method)
    order_key = _condition_token(order_type) or "UNSPECIFIED"
    session_key = _condition_token(session) or "UNSPECIFIED"
    regime_key = _condition_token(regime) or "UNSPECIFIED"
    for key in (
        (method_key, order_key, session_key, regime_key),
        (method_key, order_key, session_key, "UNSPECIFIED"),
        (method_key, order_key, "UNSPECIFIED", regime_key),
        (method_key, order_key, "UNSPECIFIED", "UNSPECIFIED"),
        (method_key, "UNSPECIFIED", session_key, regime_key),
        (method_key, "UNSPECIFIED", session_key, "UNSPECIFIED"),
        (method_key, "UNSPECIFIED", "UNSPECIFIED", regime_key),
        (method_key, "UNSPECIFIED", "UNSPECIFIED", "UNSPECIFIED"),
    ):
        edge = index.get(key)
        if edge:
            return edge
    return {}


def _condition_token(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper().replace("/", "_").replace(" ", "_")
    if not text or text == "GENERATED_DRY-RUN":
        return ""
    if "CAMPAIGN_LANE" in text:
        return ""
    if text == "NEWYORK":
        return "NY"
    return text


def _first_condition_token(*values: object) -> str:
    for value in values:
        token = _condition_token(value)
        if token:
            return token
    return ""


def _session_bucket_from_timestamp(value: object) -> str:
    if not value:
        return ""
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return ""
    hour = parsed.hour
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 21:
        return "NY"
    return "ROLLOVER"


def _blockers(
    *,
    intents: dict[str, Any],
    target: dict[str, Any],
    remaining_target: float | None,
    live_ready: tuple[AttackLaneAdvice, ...],
    recommended: tuple[AttackLaneAdvice, ...],
) -> Iterator[str]:
    if not intents:
        yield "order intents are missing; run generate-intents first"
    if not target:
        yield "daily target state is missing; run daily-target-state first"
    if remaining_target is None or remaining_target <= 0:
        yield "remaining target is missing or already complete"
    if not live_ready:
        yield "no LIVE_READY lanes are available for attack advice"
    if live_ready and not recommended:
        yield "LIVE_READY lanes exist but none fit current remaining risk budget"


def _action_items(
    *,
    remaining_target: float | None,
    live_ready_reward: float,
    recommended_reward: float,
    live_ready: tuple[AttackLaneAdvice, ...],
    edge_index: dict[tuple[str, str], dict[str, Any]],
    outcome_index: dict[tuple[str, str, str], dict[str, Any]],
    condition_index: dict[tuple[str, str, str, str], dict[str, Any]],
    coverage: dict[str, Any],
) -> Iterator[str]:
    if remaining_target and live_ready_reward < remaining_target:
        gap = _round(remaining_target - live_ready_reward)
        yield f"build additional LIVE_READY receipts for {gap:.0f} JPY of target coverage"
    if remaining_target and recommended_reward < remaining_target and live_ready:
        yield "use recommended_now as the first verified basket, then keep generating sequential ladder receipts"
    if not edge_index:
        yield "run ai-test-bot-backtest before advice so pair/direction history is grounded"
    if not condition_index and not outcome_index:
        yield "run build-outcome-mart before advice so archive condition history is grounded"
    if condition_index and live_ready and not any(lane.archive_condition_key for lane in live_ready):
        yield "current LIVE_READY lanes lack usable session/regime condition keys; archive condition edges are report-only until intents carry current market buckets"
    coverage_status = str(coverage.get("status") or "")
    if coverage_status and coverage_status != "LIVE_READY_COVERAGE_READY":
        yield f"resolve coverage optimizer status {coverage_status} before treating attack advice as certified"


def _status(
    *,
    blockers: tuple[str, ...],
    recommended: tuple[AttackLaneAdvice, ...],
    remaining_target: float | None,
    recommended_reward: float,
) -> str:
    if not recommended:
        return "NO_ATTACK_ADVICE"
    if remaining_target and recommended_reward >= remaining_target and not blockers:
        return "ATTACK_COVERAGE_READY"
    return "ATTACK_PARTIAL"


def _required_additional_lanes(
    *,
    remaining_target: float | None,
    live_ready_reward: float,
    live_ready: tuple[AttackLaneAdvice, ...],
) -> int | None:
    if not remaining_target or remaining_target <= live_ready_reward:
        return 0
    if not live_ready:
        return None
    avg_reward = sum(lane.reward_jpy for lane in live_ready) / len(live_ready)
    if avg_reward <= 0:
        return None
    return int(math.ceil((remaining_target - live_ready_reward) / avg_reward))


def _settings_advice() -> dict[str, Any]:
    return {
        "do_not_raise_loss_cap": True,
        "do_not_enable_second_trader": True,
        "safe_parameter_surface": [
            "lane ranking",
            "archive outcome feature weighting",
            "selected_lane_ids basket",
            "condition priority",
            "additional receipt backlog priority",
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _positive_float(value: object) -> float | None:
    parsed = _optional_float(value)
    return parsed if parsed is not None and parsed > 0 else None


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float) -> float:
    return round(value, 4)

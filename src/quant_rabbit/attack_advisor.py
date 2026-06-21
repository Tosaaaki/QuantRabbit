from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from quant_rabbit.forecast_precision import (
    oanda_universal_rotation_precision_assessment,
    projection_precision_edge_summary,
)
from quant_rabbit.paths import (
    DEFAULT_AI_ATTACK_ADVICE,
    DEFAULT_AI_ATTACK_ADVICE_REPORT,
    DEFAULT_AI_TEST_BOT_BACKTEST,
    DEFAULT_COVERAGE_OPTIMIZATION,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_OUTCOME_MART,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PROJECTION_LEDGER,
)
from quant_rabbit.strategy.projection_ledger import directional_calibration_signal_name


# Advisory weights only. They rank already-LIVE_READY receipts for Codex
# attention; they never resize risk, bypass gateway validation, or create a
# production threshold. If these need tuning, replace them with an evaluated
# advisor backtest rather than treating them as risk constants.
ADVISORY_REWARD_COVERAGE_SCORE_CAP = 40.0
ADVISORY_REWARD_RISK_SCORE_MULT = 6.0
ADVISORY_POSITIVE_EDGE_SCORE = 25.0
ADVISORY_RESEARCH_POSITIVE_EDGE_SCORE = 8.0
ADVISORY_OUTCOME_MART_EDGE_SCORE = 15.0
ADVISORY_OUTCOME_MART_UNVALIDATED_EDGE_SCORE = 4.0
ADVISORY_OUTCOME_MART_MIN_TRIALS = 5
ADVISORY_MARKET_PARTICIPATION_SCORE = 5.0
# Projection economic edges are current calibration evidence, not realized
# trade profitability. Keep the boost below walk-forward outcome evidence and
# certified AI backtests so it ranks already-LIVE_READY lanes without implying
# live permission or a solved daily guarantee.
ADVISORY_PROJECTION_ECONOMIC_EDGE_SCORE = 14.0
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
    archive_condition_validation_outcomes: int
    archive_condition_validation_actual_net_jpy: float | None
    archive_condition_validation_hit_rate_pct: float | None
    rationale: tuple[str, ...]
    learning_influences: tuple[str, ...] = ()
    learning_influence_details: tuple[dict[str, object], ...] = ()
    learning_score_delta: float = 0.0
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
        projection_ledger_path: Path = DEFAULT_PROJECTION_LEDGER,
        output_path: Path = DEFAULT_AI_ATTACK_ADVICE,
        report_path: Path = DEFAULT_AI_ATTACK_ADVICE_REPORT,
    ) -> None:
        self.intents_path = intents_path
        self.target_state_path = target_state_path
        self.ai_backtest_path = ai_backtest_path
        self.outcome_mart_path = outcome_mart_path
        self.coverage_path = coverage_path
        self.projection_ledger_path = projection_ledger_path
        self.output_path = output_path
        self.report_path = report_path

    def run(self) -> AttackAdviceSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        intents = _load_json(self.intents_path)
        target = _load_json(self.target_state_path)
        ai_backtest = _load_optional_json(self.ai_backtest_path)
        outcome_mart = _load_optional_json(self.outcome_mart_path)
        coverage = _load_optional_json(self.coverage_path)
        p0_shadow = _coverage_p0_shadow_live_ready(coverage or {})
        edge_index = _edge_index(ai_backtest or {})
        outcome_index = _outcome_index(outcome_mart or {})
        condition_index = _condition_index(outcome_mart or {})
        condition_validation_index = _condition_validation_index(outcome_mart or {})
        projection_edge_index = _projection_economic_edge_index(self.projection_ledger_path)
        intent_session_bucket = _session_bucket_from_timestamp(intents.get("generated_at_utc"))
        remaining_target = _positive_float(target.get("remaining_target_jpy"))
        remaining_risk = _positive_float(target.get("remaining_risk_budget_jpy"))
        lanes = tuple(
            _attack_lane(
                item,
                edge_index=edge_index,
                outcome_index=outcome_index,
                condition_index=condition_index,
                condition_validation_index=condition_validation_index,
                projection_edge_index=projection_edge_index,
                intent_session_bucket=intent_session_bucket,
                remaining_target_jpy=remaining_target,
            )
            for item in intents.get("results", []) or []
            if isinstance(item, dict) and isinstance(item.get("intent"), dict)
        )
        live_ready = tuple(lane for lane in lanes if lane.status == "LIVE_READY" and not lane.blockers)
        # E (2026-05-13): precision filter — drop lanes that the
        # operator's B/D filters in trader_brain would penalise so
        # the advice surface matches the prefilter SEND_ENTRY view.
        # Pulls per-lane chart context from the intent metadata
        # already populated by intent_generator (_chart_context_for).
        precision_index = _precision_index_from_intents(intents)
        precision_filtered_reasons = {
            lane.lane_id: reason
            for lane in live_ready
            for reason in (_precision_lane_blocker(lane, precision_index),)
            if reason
        }
        live_ready_filtered = tuple(
            lane for lane in live_ready
            if lane.lane_id not in precision_filtered_reasons
        )
        ranked = tuple(sorted(live_ready_filtered, key=lambda lane: (lane.score, lane.reward_jpy, lane.lane_id), reverse=True))
        recommended = _recommend_now(ranked, remaining_risk_jpy=remaining_risk, remaining_target_jpy=remaining_target)
        watchlist = tuple(lane for lane in ranked if lane.lane_id not in {item.lane_id for item in recommended})[:REPORT_LANE_LIMIT]
        live_ready_reward = _round(sum(lane.reward_jpy for lane in _dedupe_geometry(live_ready)))
        recommended_reward = _round(sum(lane.reward_jpy for lane in recommended))
        recommended_risk = _round(sum(lane.risk_jpy for lane in recommended))
        coverage_pct = _round((live_ready_reward / remaining_target) * 100.0) if remaining_target else 0.0
        recommended_coverage_pct = _round((recommended_reward / remaining_target) * 100.0) if remaining_target else 0.0
        matrix_supported_repair_queue = _matrix_supported_repair_queue(coverage or {})
        projection_edge_activation_queue = _projection_edge_activation_queue(intents, projection_edge_index)
        blockers = tuple(
            _blockers(
                intents=intents,
                target=target,
                remaining_target=remaining_target,
                live_ready=live_ready,
                live_ready_filtered=live_ready_filtered,
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
                projection_edge_activation_queue=projection_edge_activation_queue,
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
            "projection_ledger_path": str(self.projection_ledger_path) if self.projection_ledger_path.exists() else None,
            "projection_economic_precision_edges": len(projection_edge_index),
            "remaining_target_jpy": remaining_target,
            "remaining_risk_budget_jpy": remaining_risk,
            "live_ready_lanes": len(live_ready),
            "precision_filtered_lane_ids": list(precision_filtered_reasons),
            "precision_filtered_reasons": precision_filtered_reasons,
            "live_ready_reward_jpy": live_ready_reward,
            "coverage_pct": coverage_pct,
            "recommended_now_lane_ids": [lane.lane_id for lane in recommended],
            "recommended_now_reward_jpy": recommended_reward,
            "recommended_now_risk_jpy": recommended_risk,
            "recommended_now_coverage_pct": recommended_coverage_pct,
            "watchlist_lane_ids": [lane.lane_id for lane in watchlist],
            "matrix_supported_repair_queue": matrix_supported_repair_queue,
            "projection_edge_activation_queue": projection_edge_activation_queue,
            "self_improvement_p0_shadow_live_ready": p0_shadow,
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
            f"- Projection economic precision edges: `{payload['projection_economic_precision_edges']}`",
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
                    f"condition_validated_net=`{lane['archive_condition_validation_actual_net_jpy']}` "
                    f"method_edge=`{lane['archive_method_edge_jpy']}` "
                    f"learning_delta=`{lane.get('learning_score_delta', 0.0):.1f}` "
                    f"learning=`{', '.join(lane.get('learning_influences') or []) or 'none'}`"
                )
        else:
            lines.append("- none")
        lines.extend(["", "## Watchlist", ""])
        if payload["watchlist_lane_ids"]:
            for lane_id in payload["watchlist_lane_ids"][:REPORT_LANE_LIMIT]:
                lines.append(f"- `{lane_id}`")
        else:
            lines.append("- none")
        lines.extend(["", "## Matrix-Supported Repair Queue", ""])
        if payload.get("matrix_supported_repair_queue"):
            for item in payload["matrix_supported_repair_queue"][:REPORT_LANE_LIMIT]:
                pair = item.get("pair") or ""
                direction = item.get("direction") or ""
                state = item.get("coverage_state") or ""
                status = item.get("strategy_profile_status") or ""
                support = item.get("matrix_support_count")
                net = item.get("managed_net_jpy")
                blockers = item.get("top_blockers") if isinstance(item.get("top_blockers"), list) else []
                blocker = blockers[0] if blockers else ""
                lines.append(
                    f"- `{pair} {direction}` state=`{state}` profile=`{status}` "
                    f"matrix_support=`{support}` managed_net=`{net}` blocker=`{blocker}`"
                )
        else:
            lines.append("- none")
        lines.extend(["", "## Projection Edge Activation Queue", ""])
        if payload.get("projection_edge_activation_queue"):
            for item in payload["projection_edge_activation_queue"][:REPORT_LANE_LIMIT]:
                lines.append(
                    f"- `{item.get('signal_name')}` bucket=`{item.get('bucket')}` "
                    f"direction=`{item.get('edge_direction') or 'EITHER'}` "
                    f"status=`{item.get('activation_status')}` "
                    f"repair=`{item.get('primary_repair_category')}` "
                    f"economic_Wilson95_lower=`{item.get('economic_hit_rate_wilson_lower')}` "
                    f"matched_lanes=`{item.get('matched_lane_count')}` "
                    f"blocker=`{((item.get('top_blockers') or [''])[0])}`"
                )
        else:
            lines.append("- none")
        lines.extend(["", "## Precision Filtered", ""])
        if payload.get("precision_filtered_reasons"):
            for lane_id, reason in payload["precision_filtered_reasons"].items():
                lines.append(f"- `{lane_id}`: {reason}")
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
    condition_validation_index: dict[tuple[str, str], dict[str, Any]],
    projection_edge_index: dict[tuple[str, str, str], dict[str, Any]],
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
            normalizer=_session_condition_token,
        ),
        regime=_first_condition_token(
            metadata.get("regime_state"),
            context.get("regime_state"),
            context.get("regime"),
            normalizer=_regime_condition_token,
        ),
    )
    condition_net_jpy = _optional_float(condition_edge.get("net_jpy")) if condition_edge else None
    condition_avg_jpy = _optional_float(condition_edge.get("avg_jpy")) if condition_edge else None
    condition_trials = int(condition_edge.get("outcome_n") or 0) if condition_edge else 0
    condition_key = str(condition_edge.get("key") or "") if condition_edge else None
    condition_validation = _condition_validation_for_edge(
        condition_validation_index,
        condition_key=condition_key,
        condition_net_jpy=condition_net_jpy,
    )
    condition_validation_outcomes = int(condition_validation.get("outcomes") or 0) if condition_validation else 0
    condition_validation_actual_net_jpy = (
        _optional_float(condition_validation.get("actual_net_jpy")) if condition_validation else None
    )
    condition_validation_hit_rate_pct = (
        _optional_float(condition_validation.get("directional_hit_rate_pct")) if condition_validation else None
    )
    rationale: list[str] = []
    learning_influences: list[str] = []
    learning_details: list[dict[str, object]] = []
    learning_score_delta = 0.0
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
        source_status = str(edge.get("source_status") or "")
        if source_status == "TARGET_COVERAGE_CERTIFIED":
            delta = ADVISORY_POSITIVE_EDGE_SCORE
            influence = "ai_backtest_certified_positive_edge"
        else:
            delta = ADVISORY_RESEARCH_POSITIVE_EDGE_SCORE
            influence = "ai_backtest_research_positive_edge"
        score += delta
        learning_score_delta += delta
        learning_influences.append(influence)
        learning_details.append(
            {
                "source": "ai_backtest",
                "influence": influence,
                "score_delta": delta,
                "source_status": source_status,
                "edge_jpy": _round(edge_jpy),
                "trades": edge_trades,
            }
        )
        rationale.append(f"positive AI-test-bot pair/direction edge ({source_status}, +{delta:.1f})")
    elif edge_jpy is not None and edge_jpy < 0:
        rationale.append(
            "negative AI-test-bot pair/direction edge is advisory only; "
            "current LIVE_READY rank is not demoted"
        )
    condition_is_actionable = condition_trials >= ADVISORY_OUTCOME_MART_MIN_TRIALS
    if condition_net_jpy is not None and condition_is_actionable and condition_net_jpy > 0:
        if (
            condition_validation_outcomes >= ADVISORY_OUTCOME_MART_MIN_TRIALS
            and condition_validation_actual_net_jpy is not None
            and condition_validation_actual_net_jpy <= 0
        ):
            rationale.append("positive archive condition edge failed walk-forward validation")
        elif (
            0 < condition_validation_outcomes < ADVISORY_OUTCOME_MART_MIN_TRIALS
            and condition_validation_actual_net_jpy is not None
            and condition_validation_actual_net_jpy <= 0
        ):
            rationale.append(
                "positive archive condition edge has negative partial walk-forward validation; "
                "no learning boost until validation recovers"
            )
        elif (
            condition_validation_outcomes >= ADVISORY_OUTCOME_MART_MIN_TRIALS
            and condition_validation_actual_net_jpy is not None
            and condition_validation_actual_net_jpy > 0
        ):
            delta = ADVISORY_OUTCOME_MART_EDGE_SCORE
            score += delta
            learning_score_delta += delta
            learning_influences.append("outcome_mart_walk_forward_positive_edge")
            learning_details.append(
                {
                    "source": "outcome_mart",
                    "influence": "outcome_mart_walk_forward_positive_edge",
                    "score_delta": delta,
                    "condition_key": condition_key,
                    "outcomes": condition_validation_outcomes,
                    "actual_net_jpy": _round(condition_validation_actual_net_jpy),
                    "hit_rate_pct": _round(condition_validation_hit_rate_pct)
                    if condition_validation_hit_rate_pct is not None
                    else None,
                }
            )
            rationale.append(f"positive archive condition edge passed walk-forward validation (+{delta:.1f})")
        else:
            delta = ADVISORY_OUTCOME_MART_UNVALIDATED_EDGE_SCORE
            score += delta
            learning_score_delta += delta
            learning_influences.append("outcome_mart_unvalidated_positive_edge")
            learning_details.append(
                {
                    "source": "outcome_mart",
                    "influence": "outcome_mart_unvalidated_positive_edge",
                    "score_delta": delta,
                    "condition_key": condition_key,
                    "outcomes": condition_trials,
                    "validation_outcomes": condition_validation_outcomes,
                }
            )
            rationale.append(f"positive archive condition edge lacks walk-forward validation floor (+{delta:.1f})")
    elif condition_net_jpy is not None and condition_is_actionable and condition_net_jpy < 0:
        if (
            condition_validation_outcomes >= ADVISORY_OUTCOME_MART_MIN_TRIALS
            and condition_validation_actual_net_jpy is not None
            and condition_validation_actual_net_jpy < 0
        ):
            rationale.append("negative archive condition edge validated walk-forward; advisory only")
        else:
            rationale.append("negative archive condition edge; advisory only")
    elif condition_net_jpy is not None and not condition_is_actionable:
        rationale.append("archive condition edge below sample floor")
    oanda_delta, oanda_detail, oanda_rationale = _oanda_rotation_rank_edge(
        intent=intent,
        metadata=metadata,
        metrics=metrics,
        pair=pair,
        direction=direction,
        order_type=order_type,
        method=method,
    )
    if oanda_rationale:
        rationale.append(oanda_rationale)
    if oanda_delta > 0.0 and oanda_detail:
        score += oanda_delta
        learning_score_delta += oanda_delta
        learning_influences.append("oanda_universal_rotation_rank_edge")
        learning_details.append(oanda_detail)
    projection_delta, projection_detail, projection_rationale = _projection_economic_rank_edge(
        metadata=metadata,
        projection_edge_index=projection_edge_index,
        pair=pair,
    )
    if projection_rationale:
        rationale.append(projection_rationale)
    if projection_delta > 0.0 and projection_detail:
        score += projection_delta
        learning_score_delta += projection_delta
        learning_influences.append("projection_economic_precision_rank_edge")
        learning_details.append(projection_detail)
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
        archive_condition_validation_outcomes=condition_validation_outcomes,
        archive_condition_validation_actual_net_jpy=_round(condition_validation_actual_net_jpy)
        if condition_validation_actual_net_jpy is not None
        else None,
        archive_condition_validation_hit_rate_pct=_round(condition_validation_hit_rate_pct)
        if condition_validation_hit_rate_pct is not None
        else None,
        rationale=tuple(rationale),
        learning_influences=tuple(learning_influences),
        learning_influence_details=tuple(learning_details),
        learning_score_delta=_round(learning_score_delta),
        blockers=tuple(blockers),
    )


def _projection_economic_rank_edge(
    *,
    metadata: dict[str, Any],
    projection_edge_index: dict[tuple[str, str, str], dict[str, Any]],
    pair: str,
) -> tuple[float, dict[str, object] | None, str | None]:
    if not projection_edge_index:
        return 0.0, None, None
    support = metadata.get("forecast_market_support")
    if not isinstance(support, dict):
        return 0.0, None, None
    if support.get("ok") is not True:
        return 0.0, None, None
    regime = _projection_regime_token(
        metadata.get("regime_state")
        or metadata.get("dominant_regime")
        or metadata.get("forecast_regime")
    )
    candidates: list[dict[str, Any]] = []
    for signal in support.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        if signal.get("live_precision_ok") is False:
            continue
        candidates.extend(
            _projection_edge_matches_for_signal(
                signal,
                projection_edge_index=projection_edge_index,
                pair=pair,
                regime=regime,
                support_direction=support.get("direction"),
            )
        )
    if not candidates:
        return 0.0, None, None
    best = max(
        candidates,
        key=lambda item: (
            float(item.get("economic_hit_rate_wilson_lower") or 0.0),
            float(item.get("economic_hit_rate") or 0.0),
            int(item.get("economic_samples") or 0),
            float(item.get("hit_rate_wilson_lower") or 0.0),
        ),
    )
    detail: dict[str, object] = {
        "source": "projection_ledger",
        "influence": "projection_economic_precision_rank_edge",
        "score_delta": ADVISORY_PROJECTION_ECONOMIC_EDGE_SCORE,
        "rank_only": True,
        "signal_name": str(best.get("signal_name") or ""),
        "edge_direction": str(best.get("edge_direction") or ""),
        "bucket": str(best.get("bucket") or ""),
        "pair": str(best.get("pair") or ""),
        "regime": str(best.get("regime") or ""),
        "samples": int(best.get("samples") or 0),
        "hit_rate": _round(float(best.get("hit_rate") or 0.0)),
        "hit_rate_wilson_lower": _round(float(best.get("hit_rate_wilson_lower") or 0.0)),
        "economic_samples": int(best.get("economic_samples") or 0),
        "economic_hit_rate": _round(float(best.get("economic_hit_rate") or 0.0)),
        "economic_hit_rate_wilson_lower": _round(
            float(best.get("economic_hit_rate_wilson_lower") or 0.0)
        ),
        "timeout_rate": _round(float(best.get("timeout_rate") or 0.0)),
    }
    rationale = (
        "projection economic precision rank edge "
        f"(+{ADVISORY_PROJECTION_ECONOMIC_EDGE_SCORE:.1f}): {best.get('signal_name')} "
        f"direction={best.get('edge_direction') or 'EITHER'} "
        f"bucket={best.get('bucket')} economic_Wilson95_lower="
        f"{float(best.get('economic_hit_rate_wilson_lower') or 0.0):.2f} "
        f"economic_hit={float(best.get('economic_hit_rate') or 0.0):.2f} "
        f"n={int(best.get('economic_samples') or 0)} rank-only"
    )
    return ADVISORY_PROJECTION_ECONOMIC_EDGE_SCORE, detail, rationale


def _projection_edge_matches_for_signal(
    signal: dict[str, Any],
    *,
    projection_edge_index: dict[tuple[str, str, str], dict[str, Any]],
    pair: str,
    regime: str,
    support_direction: object | None = None,
) -> list[dict[str, Any]]:
    signal_direction = _projection_signal_direction(signal, fallback=support_direction)
    names = _projection_signal_candidate_names(signal, signal_direction)
    if not names:
        return []
    regimes = [regime, "_all_regimes"] if regime else ["_all_regimes"]
    pairs = [pair, "_all_pairs"] if pair else ["_all_pairs"]
    specificity_order = [
        (pair_key, regime_key)
        for pair_key in pairs
        for regime_key in regimes
    ]
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for name in names:
        for pair_key, regime_key in specificity_order:
            item = projection_edge_index.get((name, pair_key, regime_key))
            if not item:
                continue
            if not _projection_edge_direction_matches(item, signal_direction):
                continue
            key = (str(item.get("signal_name") or ""), str(item.get("bucket") or ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            break
    return out


def _projection_signal_direction(signal: dict[str, Any], *, fallback: object | None = None) -> str:
    for raw in (signal.get("direction"), fallback):
        text = str(raw or "").strip().upper()
        if text in {"UP", "DOWN", "RANGE", "EITHER"}:
            return text
    return ""


def _projection_signal_candidate_names(signal: dict[str, Any], signal_direction: str) -> list[str]:
    raw_names: list[str] = []
    for raw in (signal.get("calibration_name"), signal.get("name")):
        text = str(raw or "").strip()
        if text and text not in raw_names:
            raw_names.append(text)
    if signal_direction not in {"UP", "DOWN", "RANGE"}:
        return raw_names

    names: list[str] = []
    for raw in raw_names:
        alias = directional_calibration_signal_name(raw, signal_direction)
        if alias and alias not in names:
            names.append(alias)
        # A base signal name pools all directions. Once a direction is known,
        # only a matching direction-specific alias is precise enough for rank
        # activation; otherwise the opposite side can inherit a stale edge.
        if _projection_signal_name_direction(raw) == signal_direction and raw not in names:
            names.append(raw)
    return names


def _projection_signal_name_direction(signal_name: object) -> str:
    text = str(signal_name or "").strip().lower()
    if text.endswith("_up"):
        return "UP"
    if text.endswith("_down"):
        return "DOWN"
    if text.endswith("_range"):
        return "RANGE"
    return ""


def _projection_signal_base_name(signal_name: object) -> str:
    text = str(signal_name or "").strip()
    direction = _projection_signal_name_direction(text)
    if not direction:
        return text
    suffix = f"_{direction.lower()}"
    return text[: -len(suffix)]


def _projection_edge_direction_matches(edge: dict[str, Any], signal_direction: str) -> bool:
    edge_direction = str(edge.get("edge_direction") or "").upper()
    if signal_direction in {"UP", "DOWN", "RANGE"}:
        return edge_direction == signal_direction
    if signal_direction == "EITHER":
        return edge_direction in {"", "EITHER"}
    return True


def _projection_regime_token(value: object) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if not text:
        return ""
    if "RANGE" in text:
        return "RANGE"
    if "TREND" in text or "IMPULSE" in text:
        return "TREND"
    if "REVERSAL" in text:
        return "REVERSAL_RISK"
    if "UNCLEAR" in text or "TRANSITION" in text:
        return "UNCLEAR"
    return text


def _oanda_rotation_rank_edge(
    *,
    intent: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    pair: str,
    direction: str,
    order_type: str,
    method: str,
) -> tuple[float, dict[str, object] | None, str | None]:
    assessment = metadata.get("oanda_universal_rotation_precision_assessment")
    if not isinstance(assessment, dict):
        assessment_metadata = _oanda_metadata_with_execution_costs(
            metadata,
            spread_pips=_optional_float(metrics.get("spread_pips")),
        )
        assessment = oanda_universal_rotation_precision_assessment(
            assessment_metadata,
            pair=pair,
            side=direction,
            order_type=order_type,
            method=method,
            entry=_optional_float(intent.get("entry")),
            take_profit=_optional_float(intent.get("tp")),
            stop_loss=_optional_float(intent.get("sl")),
        )
    supports = [
        item for item in assessment.get("rank_only_supports", [])
        if isinstance(item, dict)
    ]
    if not supports:
        return 0.0, None, None
    raw_delta = _optional_float(assessment.get("score_delta")) or 0.0
    scale, scale_rationale = _oanda_capture_rotation_scale(metadata)
    delta = _round(raw_delta * scale)
    best = (
        assessment.get("primary_rank_support")
        if isinstance(assessment.get("primary_rank_support"), dict)
        else max(
            supports,
            key=lambda item: (
                float(item.get("rank_score_bonus") or 0.0),
                float(item.get("capital_efficiency_score") or 0.0),
                float(item.get("validation_win_wilson95_lower") or 0.0),
                float(item.get("validation_profit_factor") or 0.0),
                int(item.get("validation_samples") or 0),
            ),
        )
    )
    raw_live_gap_reasons = best.get("live_gap_reasons")
    live_gap_reasons = (
        [str(item) for item in raw_live_gap_reasons]
        if isinstance(raw_live_gap_reasons, list)
        else ["NONE"]
    )
    if scale_rationale and delta <= 0.0:
        return 0.0, None, scale_rationale
    detail: dict[str, object] = {
        "source": "oanda_universal_rotation",
        "influence": "oanda_universal_rotation_rank_edge",
        "score_delta": delta,
        "raw_score_delta": _round(raw_delta),
        "rule_name": str(best.get("name") or ""),
        "rank_only": True,
        "validation_samples": int(best.get("validation_samples") or 0),
        "validation_win_rate": _round(float(best.get("validation_win_rate") or 0.0)),
        "validation_wilson95_lower": _round(float(best.get("validation_win_wilson95_lower") or 0.0)),
        "validation_profit_factor": _round(float(best.get("validation_profit_factor") or 0.0)),
        "live_gap_reasons": live_gap_reasons,
        "rule_source_section": str(best.get("rule_source_section") or ""),
    }
    if scale != 1.0:
        detail["capture_rotation_score_scale"] = _round(scale)
    rationale = (
        "OANDA universal rotation rank edge "
        f"(+{delta:.1f}): {best.get('name')} "
        f"valid_n={int(best.get('validation_samples') or 0)} "
        f"win={float(best.get('validation_win_rate') or 0.0):.2f} "
        f"Wilson95_lower={float(best.get('validation_win_wilson95_lower') or 0.0):.2f} "
        f"PF={float(best.get('validation_profit_factor') or 0.0):.2f} "
        f"rank-only live_gap={','.join(str(item) for item in live_gap_reasons)}"
    )
    if scale_rationale:
        rationale = f"{rationale}; {scale_rationale}"
    return delta, detail, rationale


def _oanda_metadata_with_execution_costs(
    metadata: dict[str, Any],
    *,
    spread_pips: float | None,
) -> dict[str, Any]:
    out = dict(metadata)
    atr_pips = _optional_float(
        out.get("oanda_m5_atr_pips")
        if out.get("oanda_m5_atr_pips") is not None
        else out.get("m5_atr_pips")
    )
    if spread_pips is not None and spread_pips > 0.0 and atr_pips is not None and atr_pips > 0.0:
        spread_atr = spread_pips / atr_pips
        out["oanda_m5_spread_atr"] = round(spread_atr, 6)
    return out


def _oanda_capture_rotation_scale(metadata: dict[str, Any]) -> tuple[float, str | None]:
    status = str(metadata.get("capture_economics_status") or "").upper()
    if status != "NEGATIVE_EXPECTANCY":
        return 1.0, None
    if metadata.get("positive_rotation_live_ready") is not True:
        return (
            0.0,
            "capture economics is NEGATIVE_EXPECTANCY; OANDA rank-only rotation edge is size-neutral "
            "until positive_rotation_live_ready proves TP HARVEST capture",
        )
    if metadata.get("positive_rotation_minimum_floor_reachable") is not True:
        return (
            1.0,
            "positive rotation lacks daily 5% floor firepower proof; keep OANDA rank-only ordering active "
            "but do not treat the daily floor as solved",
        )
    return 1.0, None


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
    if not _ai_backtest_edge_indexable(payload):
        return index
    source_status = str(payload.get("status") or "UNKNOWN")
    for item in payload.get("bucket_contributions", []) or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("bucket") or "")
        parts = label.split(":")
        if len(parts) < 3:
            continue
        key = (parts[1], parts[2])
        current = index.setdefault(
            key,
            {"managed_net_jpy": 0.0, "trades": 0, "buckets": [], "source_status": source_status},
        )
        current["managed_net_jpy"] = _round(float(current["managed_net_jpy"]) + float(item.get("managed_net_jpy") or 0.0))
        current["trades"] = int(current["trades"]) + int(item.get("trades") or 0)
        current["buckets"].append(label)
    return index


def _ai_backtest_edge_indexable(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    if payload.get("live_permission") is True:
        return False
    status = str(payload.get("status") or "")
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    if status == "TARGET_COVERAGE_CERTIFIED":
        return not blockers
    if status != "RESEARCH_PROFITABLE_NOT_CERTIFIED":
        return False
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    selected_trades = int(_optional_float(summary.get("selected_trades")) or 0)
    managed_net = _optional_float(summary.get("total_managed_net_jpy")) or 0.0
    profit_factor = _optional_float(summary.get("profit_factor")) or 0.0
    return selected_trades >= 30 and managed_net > 0 and profit_factor > 1.0


def _projection_economic_edge_index(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        from quant_rabbit.risk import (
            FORECAST_LIVE_PRECISION_MIN_SAMPLES,
            FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER,
        )
        from quant_rabbit.strategy.projection_ledger import compute_hit_rates

        hit_rates = compute_hit_rates(path.parent)
    except Exception:
        return {}
    filtered = {
        signal_name: buckets
        for signal_name, buckets in (hit_rates or {}).items()
        if not str(signal_name or "").startswith("directional_forecast")
    }
    edges = projection_precision_edge_summary(
        filtered,
        min_wilson_lower=FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER,
        min_samples=FORECAST_LIVE_PRECISION_MIN_SAMPLES,
        limit=10_000,
    )
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    directional_base_keys: set[tuple[str, str, str]] = set()
    for item in edges:
        if not isinstance(item, dict):
            continue
        signal_name = str(item.get("signal_name") or "")
        pair = str(item.get("pair") or "")
        regime = str(item.get("regime") or "")
        if not signal_name or not pair or not regime:
            continue
        edge_direction = _projection_signal_name_direction(signal_name)
        if edge_direction:
            directional_base_keys.add((_projection_signal_base_name(signal_name), pair, regime))

    for item in edges:
        if not isinstance(item, dict):
            continue
        signal_name = str(item.get("signal_name") or "")
        pair = str(item.get("pair") or "")
        regime = str(item.get("regime") or "")
        if not signal_name or not pair or not regime:
            continue
        edge_direction = _projection_signal_name_direction(signal_name)
        if not edge_direction and (signal_name, pair, regime) in directional_base_keys:
            continue
        enriched = dict(item)
        enriched["edge_direction"] = edge_direction or "EITHER"
        index[(signal_name, pair, regime)] = enriched
    return index


def _projection_edge_activation_queue(
    intents: dict[str, Any],
    projection_edge_index: dict[tuple[str, str, str], dict[str, Any]],
    *,
    limit: int = REPORT_LANE_LIMIT,
) -> list[dict[str, Any]]:
    if not projection_edge_index:
        return []
    active_keys: set[tuple[str, str]] = set()
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    results = [
        item for item in intents.get("results", []) or []
        if isinstance(item, dict) and isinstance(item.get("intent"), dict)
    ]

    for result in results:
        intent = result["intent"]
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        support = metadata.get("forecast_market_support")
        if not isinstance(support, dict):
            continue
        pair = str(intent.get("pair") or "")
        regime = _projection_regime_token(
            metadata.get("regime_state")
            or metadata.get("dominant_regime")
            or metadata.get("forecast_regime")
        )
        lane_id = str(result.get("lane_id") or "")
        status = str(result.get("status") or "")
        blockers = _intent_issue_strings(result)
        support_ok = support.get("ok") is True

        for source, signals in (
            ("selected", support.get("signals") or []),
            ("unselected", support.get("unselected_signals") or []),
        ):
            if not isinstance(signals, list):
                continue
            for signal in signals:
                if not isinstance(signal, dict):
                    continue
                matches = _projection_edge_matches_for_signal(
                    signal,
                    projection_edge_index=projection_edge_index,
                    pair=pair,
                    regime=regime,
                    support_direction=support.get("direction"),
                )
                for edge in matches:
                    key = _projection_edge_key(edge)
                    if not key:
                        continue
                    if (
                        source == "selected"
                        and status == "LIVE_READY"
                        and support_ok
                        and not blockers
                        and signal.get("live_precision_ok") is not False
                    ):
                        active_keys.add(key)
                        continue
                    row = rows.setdefault(key, _projection_edge_activation_row(edge))
                    _append_unique(row["matched_lane_ids"], lane_id)
                    _append_unique(row["signal_sources"], source)
                    if status != "LIVE_READY" or blockers:
                        _append_unique(row["blocked_lane_ids"], lane_id)
                    if signal.get("live_precision_ok") is False:
                        row["current_precision_failure_count"] = int(row["current_precision_failure_count"]) + 1
                    for blocker in blockers:
                        _append_unique(row["top_blockers"], blocker)
                        if not blocker.startswith("status is "):
                            _append_unique(row["non_status_blockers"], blocker)
                        category = _projection_edge_repair_category(blocker)
                        if category:
                            row["repair_category_hits"][category] = int(row["repair_category_hits"].get(category, 0)) + 1

    for edge in projection_edge_index.values():
        key = _projection_edge_key(edge)
        if not key or key in active_keys or key in rows:
            continue
        rows[key] = _projection_edge_activation_row(edge)

    out: list[dict[str, Any]] = []
    for key, row in rows.items():
        if key in active_keys:
            continue
        matched = row["matched_lane_ids"]
        blocked = row["blocked_lane_ids"]
        sources = set(row["signal_sources"])
        precision_failures = int(row["current_precision_failure_count"])
        if precision_failures:
            activation_status = "CURRENT_PRECISION_FAILED"
            action = "wait for current live_precision_ok support before ranking this edge"
        elif "selected" in sources and (blocked or matched):
            activation_status = "SURFACED_BUT_BLOCKED"
            action = "repair current lane blockers before using this rank-only edge"
        elif "unselected" in sources:
            activation_status = "SURFACED_UNSELECTED"
            action = "resolve forecast direction conflict before using this rank-only edge"
        else:
            activation_status = "EDGE_READY_NO_CURRENT_SIGNAL"
            action = "refresh market context and wait for the detector to fire again"
        repair_categories = _projection_edge_repair_category_summary(row["repair_category_hits"])
        primary_repair_category = (
            str(repair_categories[0]["category"])
            if repair_categories
            else _projection_edge_default_repair_category(activation_status)
        )
        primary_repair_action = _projection_edge_repair_action(
            primary_repair_category,
            activation_status=activation_status,
        )
        row["activation_status"] = activation_status
        row["activation_action"] = primary_repair_action or action
        row["primary_repair_category"] = primary_repair_category
        row["primary_repair_action"] = primary_repair_action or action
        row["repair_categories"] = repair_categories
        row["matched_lane_count"] = len(matched)
        row["blocked_lane_count"] = len(blocked)
        row["matched_lane_ids"] = matched[:6]
        row["blocked_lane_ids"] = blocked[:6]
        row["signal_sources"] = row["signal_sources"][:4]
        row["top_blockers"] = (row["non_status_blockers"] or row["top_blockers"])[:6]
        row["non_status_blockers"] = row["non_status_blockers"][:6]
        row.pop("repair_category_hits", None)
        out.append(row)

    out = _projection_edge_activation_collapse_broader_duplicates(out)
    out.sort(key=_projection_edge_activation_sort_key)
    return out[: max(0, int(limit))]


def _projection_edge_activation_row(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "signal_name": str(edge.get("signal_name") or ""),
        "edge_direction": str(edge.get("edge_direction") or ""),
        "bucket": str(edge.get("bucket") or ""),
        "pair": str(edge.get("pair") or ""),
        "regime": str(edge.get("regime") or ""),
        "samples": int(edge.get("samples") or 0),
        "hit_rate": _round(float(edge.get("hit_rate") or 0.0)),
        "hit_rate_wilson_lower": _round(float(edge.get("hit_rate_wilson_lower") or 0.0)),
        "economic_samples": int(edge.get("economic_samples") or 0),
        "economic_hit_rate": _round(float(edge.get("economic_hit_rate") or 0.0)),
        "economic_hit_rate_wilson_lower": _round(
            float(edge.get("economic_hit_rate_wilson_lower") or 0.0)
        ),
        "timeout_rate": _round(float(edge.get("timeout_rate") or 0.0)),
        "rank_only": True,
        "matched_lane_ids": [],
        "blocked_lane_ids": [],
        "signal_sources": [],
        "top_blockers": [],
        "non_status_blockers": [],
        "repair_category_hits": {},
        "current_precision_failure_count": 0,
    }


def _projection_edge_repair_category(blocker: str) -> str:
    text = str(blocker or "").upper()
    if not text or text.startswith("STATUS IS "):
        return ""
    if "SPREAD" in text or "LIQUIDITY" in text:
        return "SPREAD_LIQUIDITY_WAIT"
    if (
        "RISK-RESIZED" in text
        or "RISK_RESIZED" in text
        or "LOSS CAP" in text
        or "LOSS_CAP" in text
        or "MARGIN" in text
        or "MIN_LOT" in text
    ):
        return "RISK_RESIZE_DRY_RUN"
    if (
        "BLOCK_UNTIL_NEW_EVIDENCE" in text
        or "WATCH_ONLY" in text
        or "STRATEGY_PROFILE" in text
        or "MINED STRATEGY PROFILE" in text
    ):
        return "STRATEGY_PROFILE_REPAIR"
    if (
        "MATRIX_REPAIR_REJECT_CONTEXT" in text
        or "REJECT CONTEXT" in text
        or "MARKET_CONTEXT" in text
        or "CONFLUENCE" in text
        or "SCORE_BALANCE" in text
        or "QUOTE JPY STRENGTH" in text
    ):
        return "MARKET_CONTEXT_REPAIR"
    if (
        "FORECAST" in text
        or "PREDICTION" in text
        or "UNSELECTED_DIRECTION" in text
        or "AUDITED" in text
        or "PROJECTION" in text
    ):
        return "FORECAST_SUPPORT_REPAIR"
    if (
        "EXHAUSTION" in text
        or "PATTERN" in text
        or "CHART_DIRECTION_CONFLICT" in text
        or "BOS" in text
        or "CHOCH" in text
    ):
        return "STRUCTURE_TIMING_REPAIR"
    return "OTHER_BLOCKER"


def _projection_edge_repair_category_summary(counts: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category, count in counts.items():
        rows.append({"category": str(category), "count": int(count or 0)})
    rows.sort(
        key=lambda item: (
            -int(item.get("count") or 0),
            _projection_edge_repair_category_priority(str(item.get("category") or "")),
            str(item.get("category") or ""),
        )
    )
    return rows


def _projection_edge_repair_category_priority(category: str) -> int:
    order = {
        "RISK_RESIZE_DRY_RUN": 0,
        "FORECAST_SUPPORT_REPAIR": 1,
        "MARKET_CONTEXT_REPAIR": 2,
        "STRATEGY_PROFILE_REPAIR": 3,
        "STRUCTURE_TIMING_REPAIR": 4,
        "SPREAD_LIQUIDITY_WAIT": 5,
        "CURRENT_PRECISION_REPAIR": 6,
        "DETECTOR_REFRESH_WAIT": 7,
        "OTHER_BLOCKER": 8,
    }
    return order.get(str(category or ""), 99)


def _projection_edge_default_repair_category(activation_status: str) -> str:
    if activation_status == "CURRENT_PRECISION_FAILED":
        return "CURRENT_PRECISION_REPAIR"
    if activation_status == "SURFACED_UNSELECTED":
        return "FORECAST_SUPPORT_REPAIR"
    if activation_status == "EDGE_READY_NO_CURRENT_SIGNAL":
        return "DETECTOR_REFRESH_WAIT"
    return "OTHER_BLOCKER"


def _projection_edge_repair_action(category: str, *, activation_status: str) -> str:
    if activation_status == "EDGE_READY_NO_CURRENT_SIGNAL":
        return "refresh market context and wait for this detector to fire before ranking the edge"
    if category == "RISK_RESIZE_DRY_RUN":
        return "produce a risk-resized dry-run receipt that fits the current loss cap before promotion"
    if category == "FORECAST_SUPPORT_REPAIR":
        return "repair forecast/support alignment; do not trade the edge until current direction support clears"
    if category == "MARKET_CONTEXT_REPAIR":
        return "refresh market-context matrix and wait for reject context to clear before using the edge"
    if category == "STRATEGY_PROFILE_REPAIR":
        return "mine or promote fresh bounded evidence for this pair/side/method; no auto-promotion"
    if category == "STRUCTURE_TIMING_REPAIR":
        return "wait for market-structure timing proof such as retest, rejection, or close-confirmed structure"
    if category == "SPREAD_LIQUIDITY_WAIT":
        return "wait for session/liquidity spread normalization before treating the edge as usable"
    if category == "CURRENT_PRECISION_REPAIR":
        return "wait for current live_precision_ok support before ranking this edge"
    return "repair current lane blockers before using this rank-only edge"


def _projection_edge_key(edge: dict[str, Any]) -> tuple[str, str] | None:
    signal_name = str(edge.get("signal_name") or "")
    bucket = str(edge.get("bucket") or "")
    if not signal_name or not bucket:
        return None
    return (signal_name, bucket)


def _projection_edge_activation_collapse_broader_duplicates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer actionable pair/regime rows over their broader fallback copies."""

    signatures = {
        (
            str(row.get("signal_name") or ""),
            str(row.get("edge_direction") or ""),
            str(row.get("pair") or ""),
            str(row.get("regime") or ""),
        )
        for row in rows
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        signal = str(row.get("signal_name") or "")
        direction = str(row.get("edge_direction") or "")
        pair = str(row.get("pair") or "")
        regime = str(row.get("regime") or "")
        if not signal:
            out.append(row)
            continue
        if pair != "_all_pairs" and regime == "_all_regimes":
            if any(
                s == signal and d == direction and p == pair and r != "_all_regimes"
                for s, d, p, r in signatures
            ):
                continue
        if pair == "_all_pairs" and regime != "_all_regimes":
            if any(
                s == signal and d == direction and p != "_all_pairs" and r == regime
                for s, d, p, r in signatures
            ):
                continue
        if pair == "_all_pairs" and regime == "_all_regimes":
            if any(
                s == signal and d == direction and (p != "_all_pairs" or r != "_all_regimes")
                for s, d, p, r in signatures
            ):
                continue
        out.append(row)
    return out


def _projection_edge_activation_sort_key(row: dict[str, Any]) -> tuple[int, int, float, float, int, str, str]:
    status_rank = {
        "SURFACED_BUT_BLOCKED": 0,
        "CURRENT_PRECISION_FAILED": 1,
        "SURFACED_UNSELECTED": 2,
        "EDGE_READY_NO_CURRENT_SIGNAL": 3,
    }.get(str(row.get("activation_status") or ""), 9)
    broad_rank = 1 if str(row.get("pair") or "").startswith("_all") else 0
    return (
        status_rank,
        broad_rank,
        -float(row.get("economic_hit_rate_wilson_lower") or 0.0),
        -float(row.get("economic_hit_rate") or 0.0),
        -int(row.get("economic_samples") or 0),
        str(row.get("signal_name") or ""),
        str(row.get("bucket") or ""),
    )


def _intent_issue_strings(item: dict[str, Any]) -> list[str]:
    out: list[str] = []
    status = str(item.get("status") or "")
    if status and status != "LIVE_READY":
        out.append(f"status is {status}")
    for key in ("live_blockers", "risk_issues", "strategy_issues", "blockers"):
        value = item.get(key)
        if not isinstance(value, list):
            continue
        for raw in value:
            text = ""
            if isinstance(raw, dict):
                text = str(raw.get("code") or raw.get("message") or raw.get("reason") or "").strip()
            else:
                text = str(raw or "").strip()
            if text:
                _append_unique(out, text)
    return out


def _append_unique(items: list[Any], value: Any) -> None:
    if value and value not in items:
        items.append(value)


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
    items = list(payload.get("condition_edges", []) or [])
    items.extend(payload.get("condition_rollups", []) or [])
    for item in items:
        if not isinstance(item, dict):
            continue
        method = _condition_token(item.get("method"))
        order_type = _condition_token(item.get("order_type"))
        session = _session_condition_token(item.get("session_bucket"))
        regime = _regime_condition_token(item.get("regime"))
        if not method:
            continue
        index[(method, order_type or "UNSPECIFIED", session or "UNSPECIFIED", regime or "UNSPECIFIED")] = item
    return index


def _condition_validation_index(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    validation = payload.get("condition_validation") if isinstance(payload.get("condition_validation"), dict) else {}
    for item in validation.get("matched_edges", []) or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "")
        predicted_edge = str(item.get("predicted_edge") or "")
        if key and predicted_edge:
            index[(key, predicted_edge)] = item
    return index


def _condition_validation_for_edge(
    index: dict[tuple[str, str], dict[str, Any]],
    *,
    condition_key: str | None,
    condition_net_jpy: float | None,
) -> dict[str, Any]:
    if not condition_key or condition_net_jpy is None or condition_net_jpy == 0:
        return {}
    predicted_edge = "POSITIVE" if condition_net_jpy > 0 else "NEGATIVE"
    return index.get((condition_key, predicted_edge), {})


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
    session_key = _session_condition_token(session) or "UNSPECIFIED"
    regime_key = _regime_condition_token(regime) or "UNSPECIFIED"
    for key in (
        (method_key, order_key, session_key, regime_key),
        (method_key, order_key, session_key, "ALL"),
        (method_key, order_key, "ALL", regime_key),
        (method_key, order_key, "ALL", "ALL"),
        (method_key, "ALL", session_key, regime_key),
        (method_key, "ALL", session_key, "ALL"),
        (method_key, "ALL", "ALL", regime_key),
        (method_key, "ALL", "ALL", "ALL"),
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


def _session_condition_token(value: object) -> str:
    text = _condition_token(value).replace("-", "_")
    if not text:
        return ""
    if "LONDON" in text:
        return "LONDON"
    if text in {"NY", "NEWYORK", "NEW_YORK"}:
        return "NY"
    if text.startswith("NY_") or "NEWYORK" in text or "SILVER_BULLET" in text:
        return "NY"
    if text == "TOKYO" or "TOKYO" in text or "ASIA" in text:
        return "ASIA"
    if "ROLLOVER" in text or "OFF_HOURS" in text:
        return "ROLLOVER"
    return text


def _regime_condition_token(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper().replace("/", "_").replace(" ", "_").replace("-", "_")
    if not text:
        return ""
    if "CAMPAIGN_LANE" in text and "CURRENT" not in text:
        return ""
    if "SQUEEZE" in text or "BREAKOUT_PENDING" in text:
        return "SQUEEZE"
    if "RANGE" in text or "ROTATION" in text or "MEAN_REVERT" in text:
        return "RANGE"
    if "QUIET" in text or "STABLE" in text or "THIN_LIQUIDITY" in text:
        return "QUIET"
    if "TREND" in text or "BULL" in text or "BEAR" in text or "IMPULSE" in text:
        return "TRENDING"
    if "TRANSITION" in text or "FRICTION" in text or "HEADLINE" in text:
        return "TRANSITION"
    return text


def _first_condition_token(*values: object, normalizer=_condition_token) -> str:
    for value in values:
        token = normalizer(value)
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
    live_ready_filtered: tuple[AttackLaneAdvice, ...],
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
    if live_ready and not live_ready_filtered:
        yield "LIVE_READY lanes exist but precision filters excluded every advice lane"
    elif live_ready_filtered and not recommended:
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
    projection_edge_activation_queue: list[dict[str, Any]],
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
    p0_shadow = _coverage_p0_shadow_live_ready(coverage)
    if p0_shadow.get("count"):
        lanes = ", ".join(str(lane) for lane in (p0_shadow.get("lane_ids") or [])[:3])
        yield (
            f"preserve {int(p0_shadow.get('count') or 0)} otherwise-live-ready P0-gated receipt(s) "
            f"for the first post-recovery basket; send remains blocked by profitability P0: {lanes}"
        )
    matrix_queue = _matrix_supported_repair_queue(coverage)
    if matrix_queue:
        preview = "; ".join(
            f"{item.get('pair')} {item.get('direction')} ({(item.get('top_blockers') or [''])[0]})"
            for item in matrix_queue[:3]
        )
        yield f"repair matrix-supported profitable edges before broad exploration: {preview}"
    if projection_edge_activation_queue:
        preview = "; ".join(
            f"{item.get('signal_name')} {item.get('bucket')} ({item.get('primary_repair_category')})"
            for item in projection_edge_activation_queue[:3]
        )
        yield f"activate projection economic precision edges only after current blockers clear: {preview}"
    coverage_status = str(coverage.get("status") or "")
    if coverage_status and coverage_status != "LIVE_READY_COVERAGE_READY":
        yield f"resolve coverage optimizer status {coverage_status} before treating attack advice as certified"


def _matrix_supported_repair_queue(coverage: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = (
        coverage.get("artifact_diagnostics")
        if isinstance(coverage.get("artifact_diagnostics"), dict)
        else {}
    )
    profitable = (
        diagnostics.get("profitable_bucket_coverage")
        if isinstance(diagnostics.get("profitable_bucket_coverage"), dict)
        else {}
    )
    rows = profitable.get("matrix_supported_repair_queue")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "pair": str(row.get("pair") or ""),
                "direction": str(row.get("direction") or ""),
                "coverage_state": str(row.get("coverage_state") or ""),
                "strategy_profile_status": str(row.get("strategy_profile_status") or ""),
                "strategy_profile_required_fix": str(row.get("strategy_profile_required_fix") or ""),
                "managed_net_jpy": _round(float(row.get("managed_net_jpy") or 0.0)),
                "raw_net_jpy": _round(float(row.get("raw_net_jpy") or 0.0)),
                "trades": int(row.get("trades") or 0),
                "days": int(row.get("days") or 0),
                "matrix_support_count": int(row.get("matrix_support_count") or 0),
                "matrix_reject_count": int(row.get("matrix_reject_count") or 0),
                "matrix_warning_count": int(row.get("matrix_warning_count") or 0),
                "matrix_support_context": _string_list(row.get("matrix_support_context"))[:4],
                "matrix_cross_asset_context": _string_list(row.get("matrix_cross_asset_context"))[:4],
                "top_blockers": _string_list(row.get("top_blockers"))[:4],
            }
        )
    return out[:REPORT_LANE_LIMIT]


def _coverage_p0_shadow_live_ready(coverage: dict[str, Any]) -> dict[str, Any]:
    diagnostics = (
        coverage.get("artifact_diagnostics")
        if isinstance(coverage.get("artifact_diagnostics"), dict)
        else {}
    )
    shadow = diagnostics.get("self_improvement_p0_shadow_live_ready")
    if not isinstance(shadow, dict):
        return {"count": 0, "lane_ids": (), "reward_jpy": 0.0, "risk_jpy": 0.0}
    return {
        "count": int(shadow.get("count") or 0),
        "lane_ids": tuple(str(item) for item in shadow.get("lane_ids", ()) or ()),
        "reward_jpy": _round(float(shadow.get("reward_jpy") or 0.0)),
        "risk_jpy": _round(float(shadow.get("risk_jpy") or 0.0)),
        "send_blocked": bool(shadow.get("send_blocked")),
        "blocker_code": str(shadow.get("blocker_code") or ""),
        "candidates": tuple(
            item for item in shadow.get("candidates", ()) or () if isinstance(item, dict)
        )[:REPORT_LANE_LIMIT],
    }


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item)]


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


# E — precision filter constants (2026-05-13). Same statistical
# boundaries as trader_brain's B/D filters so the prefilter and the
# advice surface agree on what counts as "extreme" or "fractured".
ATTACK_PRECISION_EXTREME_HIGH = 0.95
ATTACK_PRECISION_EXTREME_LOW = 0.05
ATTACK_PRECISION_TF_AGREEMENT_MIN = 2.0 / 3.0


def _precision_index_from_intents(intents: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map lane_id -> precision context pulled from intent.metadata.

    Producer is intent_generator._chart_context_for; consumer is
    `_precision_lane_passes` to drop lanes that the operator's B/D
    filters would penalise. Lanes without metadata or without the
    relevant fields fall through (no signal == no filter).
    """
    out: dict[str, dict[str, Any]] = {}
    for item in intents.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        lane_id = item.get("lane_id")
        if not isinstance(lane_id, str):
            continue
        intent_doc = item.get("intent") if isinstance(item.get("intent"), dict) else {}
        context_doc = intent_doc.get("market_context") if isinstance(intent_doc.get("market_context"), dict) else {}
        metadata = intent_doc.get("metadata") if isinstance(intent_doc.get("metadata"), dict) else {}
        out[lane_id] = {
            "price_percentile_24h": _optional_float(metadata.get("price_percentile_24h")),
            "price_percentile_7d": _optional_float(metadata.get("price_percentile_7d")),
            "entry_price_percentile_24h": _optional_float(metadata.get("entry_price_percentile_24h")),
            "entry_price_percentile_7d": _optional_float(metadata.get("entry_price_percentile_7d")),
            "tf_agreement_score": _optional_float(metadata.get("tf_agreement_score")),
            "range_24h_sigma_multiple": _optional_float(metadata.get("range_24h_sigma_multiple")),
            "direction": (intent_doc.get("side") or "").upper(),
            "method": str(context_doc.get("method") or "").upper(),
            "order_type": str(intent_doc.get("order_type") or "").upper(),
            "geometry_model": str(metadata.get("geometry_model") or "").upper(),
            "range_tp_is_inside_box": bool(metadata.get("range_tp_is_inside_box")),
            "range_sl_outside_box": bool(metadata.get("range_sl_outside_box")),
        }
    return out


def _precision_lane_passes(lane: AttackLaneAdvice, idx: dict[str, dict[str, Any]]) -> bool:
    return _precision_lane_blocker(lane, idx) is None


def _precision_lane_blocker(lane: AttackLaneAdvice, idx: dict[str, dict[str, Any]]) -> str | None:
    """E: drop lanes from the advice surface if their direction is
    same-side at a 24h percentile extreme, or if M15/M30/H1 are too
    fractured to chase. Missing context = lane passes (no filter
    when the data isn't there to make the call — AGENT_CONTRACT §3.5).
    """
    ctx = idx.get(lane.lane_id)
    if ctx is None:
        return None
    direction = (lane.direction or "").upper()
    percentiles = _precision_location_percentiles(ctx)
    for label, ppct in percentiles:
        if direction == "LONG" and ppct >= ATTACK_PRECISION_EXTREME_HIGH:
            return (
                f"LONG at {label} price percentile {ppct:.2f} >= "
                f"{ATTACK_PRECISION_EXTREME_HIGH:.2f}"
            )
        if direction == "SHORT" and ppct <= ATTACK_PRECISION_EXTREME_LOW:
            return (
                f"SHORT at {label} price percentile {ppct:.2f} <= "
                f"{ATTACK_PRECISION_EXTREME_LOW:.2f}"
            )
    tf_agree = ctx.get("tf_agreement_score")
    if tf_agree is not None and tf_agree < ATTACK_PRECISION_TF_AGREEMENT_MIN:
        # A passive range-rail LIMIT is not chasing the fractured operating TF;
        # it waits for price to return to an executable rail. IntentGenerator
        # and RiskEngine still own the broader-location, spread, and geometry
        # gates, so keep these lanes visible as attack candidates instead of
        # deleting the only low-cost fallback when the first advised lane fails
        # broker-truth spread revalidation.
        if (
            ctx.get("method") == "RANGE_ROTATION"
            and ctx.get("order_type") in {"LIMIT", "LIMIT_ORDER"}
            and ctx.get("geometry_model") == "RANGE_RAIL_LIMIT"
            and bool(ctx.get("range_tp_is_inside_box"))
            and bool(ctx.get("range_sl_outside_box"))
        ):
            return None
        return (
            f"tf agreement {tf_agree:.2f} < "
            f"{ATTACK_PRECISION_TF_AGREEMENT_MIN:.2f}"
        )
    return None


def _precision_location_percentiles(ctx: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    """Return the location points the advice precision filter should use."""
    use_entry = ctx.get("order_type") in {"LIMIT", "LIMIT_ORDER"}
    points: list[tuple[str, float]] = []
    for horizon in ("24h", "7d"):
        key = f"price_percentile_{horizon}"
        label = horizon
        if use_entry and ctx.get(f"entry_price_percentile_{horizon}") is not None:
            key = f"entry_price_percentile_{horizon}"
            label = f"{horizon} entry"
        value = ctx.get(key)
        if value is not None:
            points.append((label, float(value)))
    return tuple(points)

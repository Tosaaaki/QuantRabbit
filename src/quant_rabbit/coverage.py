from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.paths import (
    DEFAULT_AI_TEST_BOT_BACKTEST,
    DEFAULT_COVERAGE_OPTIMIZATION,
    DEFAULT_COVERAGE_OPTIMIZATION_REPORT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_MARKET_CONTEXT_MATRIX,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_REPLAY_BACKTEST,
    DEFAULT_STRATEGY_PROFILE,
)
from quant_rabbit.strategy.receipt_promotion import NON_PROMOTABLE_RISK_CODES


# Coverage inputs are current-cycle artifacts. One hour is an operational
# scheduler freshness boundary, not a market edge or risk threshold; replace it
# with an explicit cycle id once every artifact carries one.
COVERAGE_INTENTS_STALE_AFTER_SECONDS = 3600.0
DISCOVERY_EVIDENCE_SOURCE_TABLES = frozenset({"pretrade_outcomes", "seat_outcomes"})
HARVEST_REWARD_RISK_MAX = 1.35
RUNNER_REWARD_RISK_MIN = 2.0
QUOTE_STALE_ISSUE_CODES = frozenset({"STALE_QUOTE", "TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE"})


@dataclass(frozen=True)
class CoverageLane:
    lane_id: str
    status: str
    pair: str
    direction: str
    order_type: str
    entry: float | None
    tp: float | None
    sl: float | None
    units: int
    risk_jpy: float
    reward_jpy: float
    reward_risk: float
    opportunity_mode: str
    issue_codes: tuple[str, ...]
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
        ai_backtest_path: Path = DEFAULT_AI_TEST_BOT_BACKTEST,
        strategy_profile_path: Path = DEFAULT_STRATEGY_PROFILE,
        output_path: Path = DEFAULT_COVERAGE_OPTIMIZATION,
        report_path: Path = DEFAULT_COVERAGE_OPTIMIZATION_REPORT,
        market_context_matrix_path: Path = DEFAULT_MARKET_CONTEXT_MATRIX,
    ) -> None:
        self.intents_path = intents_path
        self.target_state_path = target_state_path
        self.replay_path = replay_path
        self.ai_backtest_path = ai_backtest_path
        self.strategy_profile_path = strategy_profile_path
        self.output_path = output_path
        self.report_path = report_path
        self.market_context_matrix_path = market_context_matrix_path

    def run(self) -> CoverageOptimizationSummary:
        now_utc = datetime.now(timezone.utc)
        generated_at = now_utc.isoformat()
        intents = _load_json(self.intents_path)
        target = _load_json(self.target_state_path)
        replay = _load_json(self.replay_path) if self.replay_path.exists() else {}
        ai_backtest = _load_json(self.ai_backtest_path) if self.ai_backtest_path.exists() else {}
        strategy_profile = _load_json(self.strategy_profile_path) if self.strategy_profile_path.exists() else {}
        market_context_matrix = _load_json(self.market_context_matrix_path) if self.market_context_matrix_path.exists() else {}
        lanes = tuple(_coverage_lane(item) for item in intents.get("results", []) or [] if _has_intent(item))
        artifact_diagnostics = _artifact_diagnostics(
            intents=intents,
            intents_path=self.intents_path,
            ai_backtest=ai_backtest,
            ai_backtest_path=self.ai_backtest_path,
            strategy_profile=strategy_profile,
            strategy_profile_path=self.strategy_profile_path,
            market_context_matrix=market_context_matrix,
            market_context_matrix_path=self.market_context_matrix_path,
            now_utc=now_utc,
        )
        remaining_target = _remaining_target(target)
        remaining_risk_budget = _remaining_risk_budget(target)
        raw_live_ready_lanes = tuple(lane for lane in lanes if lane.counts_live_ready)
        live_ready_lanes = _dedupe_exact_geometry(raw_live_ready_lanes)
        potential_lanes = _dedupe_exact_geometry(
            tuple(lane for lane in lanes if lane.counts_live_ready or lane.counts_after_promotion)
        )
        duplicate_live_ready_lanes = len(raw_live_ready_lanes) - len(live_ready_lanes)
        live_ready_reward = _round(sum(lane.reward_jpy for lane in live_ready_lanes))
        potential_reward = _round(sum(lane.reward_jpy for lane in potential_lanes))
        live_ready_risk = _round(sum(lane.risk_jpy for lane in live_ready_lanes))
        raw_live_ready_reward = _round(sum(lane.reward_jpy for lane in raw_live_ready_lanes))
        raw_live_ready_risk = _round(sum(lane.risk_jpy for lane in raw_live_ready_lanes))
        sequential_ladder = _sequential_ladder(live_ready_lanes, remaining_target, remaining_risk_budget)
        opportunity_modes = _opportunity_mode_summary(lanes, remaining_target)
        replay_gap = _replay_gap(replay)
        blockers = tuple(
            _blockers(
                target=target,
                remaining_target=remaining_target,
                remaining_risk_budget=remaining_risk_budget,
                live_ready_reward=live_ready_reward,
                potential_reward=potential_reward,
                live_ready_risk=live_ready_risk,
                duplicate_live_ready_lanes=duplicate_live_ready_lanes,
                sequential_ladder=sequential_ladder,
                lanes=lanes,
                replay_gap=replay_gap,
                artifact_diagnostics=artifact_diagnostics,
            )
        )
        action_items = tuple(
            _action_items(
                remaining_target=remaining_target,
                live_ready_reward=live_ready_reward,
                potential_reward=potential_reward,
                live_ready_risk=live_ready_risk,
                remaining_risk_budget=remaining_risk_budget,
                duplicate_live_ready_lanes=duplicate_live_ready_lanes,
                sequential_ladder=sequential_ladder,
                opportunity_modes=opportunity_modes,
                lanes=lanes,
                replay_gap=replay_gap,
                artifact_diagnostics=artifact_diagnostics,
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
            "ai_backtest_path": str(self.ai_backtest_path) if self.ai_backtest_path.exists() else None,
            "remaining_target_jpy": remaining_target,
            "remaining_risk_budget_jpy": remaining_risk_budget,
            "live_ready_reward_jpy": live_ready_reward,
            "live_ready_risk_jpy": live_ready_risk,
            "raw_live_ready_reward_jpy": raw_live_ready_reward,
            "raw_live_ready_risk_jpy": raw_live_ready_risk,
            "unique_live_ready_lanes": len(live_ready_lanes),
            "duplicate_live_ready_lanes": duplicate_live_ready_lanes,
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
            "opportunity_modes": opportunity_modes,
            "replay_gap": replay_gap,
            "artifact_diagnostics": artifact_diagnostics,
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
            live_ready_lanes=len(live_ready_lanes),
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
            f"- Unique live-ready lanes: `{payload['unique_live_ready_lanes']}` "
            f"(duplicates removed=`{payload['duplicate_live_ready_lanes']}`)",
            f"- Sequential ladder reward: `{payload['sequential_ladder_reward_jpy']:.0f} JPY` "
            f"(`{payload['sequential_ladder_coverage_pct']:.1f}%`, steps=`{payload['sequential_ladder_steps']}`)",
            f"- Potential reward after promotions: `{payload['potential_reward_jpy']:.0f} JPY` (`{payload['potential_coverage_pct']:.1f}%`)",
            f"- Remaining risk budget: `{payload['remaining_risk_budget_jpy']:.0f} JPY`",
            "",
            "## Artifact Diagnostics",
            "",
        ]
        diagnostics = payload.get("artifact_diagnostics") if isinstance(payload.get("artifact_diagnostics"), dict) else {}
        market_status = diagnostics.get("current_market_status") if isinstance(diagnostics.get("current_market_status"), dict) else {}
        lines.extend(
            [
                f"- Intent generated at UTC: `{diagnostics.get('intents_generated_at_utc') or 'unknown'}`",
                f"- Intent age seconds: `{diagnostics.get('intents_age_seconds')}`",
                f"- Intent stale: `{diagnostics.get('intents_artifact_stale')}`",
                f"- Current FX open: `{market_status.get('is_fx_open')}`",
                f"- Market closed reason: `{market_status.get('closed_reason') or 'none'}`",
                f"- Matrix missing: `{diagnostics.get('market_context_matrix_missing')}`",
                f"- Requires market evidence refresh: `{diagnostics.get('requires_market_evidence_refresh')}`",
                f"- All lanes spread-blocked: `{diagnostics.get('all_lanes_spread_blocked')}`",
                f"- All lanes quote-stale: `{diagnostics.get('all_lanes_quote_stale')}`",
                f"- Spread-normalized candidates: `{diagnostics.get('spread_normalized_candidate_count')}` "
                f"reward=`{diagnostics.get('spread_normalized_candidate_reward_jpy')}`",
                f"- Spread-normalized no-live-blocker candidates: `{diagnostics.get('spread_normalized_no_live_blocker_count')}` "
                f"reward=`{diagnostics.get('spread_normalized_no_live_blocker_reward_jpy')}`",
                f"- Spread-normalized live-blocker counts: `{diagnostics.get('spread_normalized_live_blocker_counts') or {}}`",
                f"- Quote-stale result count: `{diagnostics.get('quote_stale_result_count')}`",
                f"- Quote-normalized candidates: `{diagnostics.get('quote_normalized_candidate_count')}` "
                f"reward=`{diagnostics.get('quote_normalized_candidate_reward_jpy')}`",
                f"- Quote-normalized no-live-blocker candidates: `{diagnostics.get('quote_normalized_no_live_blocker_count')}` "
                f"reward=`{diagnostics.get('quote_normalized_no_live_blocker_reward_jpy')}`",
                f"- Quote-normalized live-blocker counts: `{diagnostics.get('quote_normalized_live_blocker_counts') or {}}`",
                f"- Risk block issue counts: `{diagnostics.get('risk_block_issue_counts') or {}}`",
                "",
            ]
        )
        modes = payload.get("opportunity_modes") if isinstance(payload.get("opportunity_modes"), dict) else {}
        if modes:
            lines.extend(["## Opportunity Modes", ""])
            for key in ("HARVEST", "RUNNER", "BALANCED"):
                item = modes.get(key) if isinstance(modes.get(key), dict) else None
                if not item:
                    continue
                blockers = item.get("top_blockers") if isinstance(item.get("top_blockers"), list) else []
                issue_codes = item.get("top_issue_codes") if isinstance(item.get("top_issue_codes"), list) else []
                top_blockers = ", ".join(str(blocker.get("label")) for blocker in blockers[:3] if isinstance(blocker, dict))
                top_codes = ", ".join(str(issue.get("code")) for issue in issue_codes[:4] if isinstance(issue, dict))
                lines.append(
                    f"- `{key}` lanes=`{item.get('lanes')}` live_ready=`{item.get('live_ready_lanes')}` "
                    f"promotion_candidates=`{item.get('promotion_candidate_lanes')}` "
                    f"total_reward=`{item.get('reward_jpy')}` live_reward=`{item.get('live_ready_reward_jpy')}` "
                    f"potential_reward=`{item.get('potential_reward_jpy')}` codes=`{top_codes or 'none'}` "
                    f"blockers=`{top_blockers or 'none'}`"
                )
            lines.append("")
        bucket_diag = diagnostics.get("profitable_bucket_coverage") if isinstance(diagnostics.get("profitable_bucket_coverage"), dict) else {}
        if bucket_diag:
            lines.extend(
                [
                    "## Profitable Bucket Coverage",
                    "",
                    f"- AI backtest status: `{bucket_diag.get('source_status')}`",
                    f"- Positive historical pair/directions: `{bucket_diag.get('positive_pair_directions')}`",
                    f"- Positive managed net: `{bucket_diag.get('positive_managed_net_jpy')}`",
                    f"- Coverage states: `{bucket_diag.get('state_counts') or {}}`",
                    "",
                ]
            )
            for item in bucket_diag.get("top_edges", []) or []:
                if not isinstance(item, dict):
                    continue
                blockers = item.get("top_blockers") if isinstance(item.get("top_blockers"), list) else []
                lines.append(
                    f"- `{item.get('pair')} {item.get('direction')}` state=`{item.get('coverage_state')}` "
                    f"managed_net=`{item.get('managed_net_jpy')}` current_lanes=`{item.get('current_lane_count')}` "
                    f"spread_normalized=`{item.get('spread_normalized_candidate_count')}` "
                    f"no_live_blocker=`{item.get('spread_normalized_no_live_blocker_count')}` "
                    f"profile=`{item.get('strategy_profile_status')}` "
                    f"matrix_supports=`{item.get('matrix_support_count')}` matrix_rejects=`{item.get('matrix_reject_count')}`"
                )
                if item.get("strategy_profile_status"):
                    lines.append(
                        "  - strategy profile: "
                        f"live_net=`{item.get('strategy_profile_live_net_jpy')}` "
                        f"pretrade_net=`{item.get('strategy_profile_pretrade_net_jpy')}` "
                        f"seat_net=`{item.get('strategy_profile_seat_net_jpy')}` "
                        f"fix={item.get('strategy_profile_required_fix') or 'none'}"
                    )
                xasset = item.get("matrix_cross_asset_context") if isinstance(item.get("matrix_cross_asset_context"), list) else []
                if xasset:
                    lines.append(f"  - cross/context assets: {', '.join(str(ref) for ref in xasset[:3])}")
                if blockers:
                    lines.append(f"  - top blockers: {', '.join(str(blocker) for blocker in blockers[:3])}")
            lines.append("")
            discovery_edges = (
                bucket_diag.get("discovery_evidence_edges")
                if isinstance(bucket_diag.get("discovery_evidence_edges"), list)
                else []
            )
            if discovery_edges:
                lines.extend(
                    [
                        "## Discovery Evidence Edges",
                        "",
                        f"- Advisory, not summed into primary historical P/L: `{bucket_diag.get('discovery_evidence_not_summed')}`",
                        "",
                    ]
                )
                for item in discovery_edges[:8]:
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"- `{item.get('bucket')}` net=`{item.get('managed_net_jpy')}` "
                        f"trades=`{item.get('trades')}` win_rate=`{item.get('win_rate_pct')}` "
                        f"coverage=`{item.get('coverage_state')}` "
                        f"matrix_supports=`{item.get('matrix_support_count')}` matrix_rejects=`{item.get('matrix_reject_count')}`"
                    )
                lines.append("")
            repair_queue = (
                bucket_diag.get("matrix_supported_repair_queue")
                if isinstance(bucket_diag.get("matrix_supported_repair_queue"), list)
                else []
            )
            if repair_queue:
                lines.extend(["## Matrix-Supported Repair Queue", ""])
                for item in repair_queue[:8]:
                    if not isinstance(item, dict):
                        continue
                    blockers = item.get("top_blockers") if isinstance(item.get("top_blockers"), list) else []
                    contexts = item.get("matrix_support_context") if isinstance(item.get("matrix_support_context"), list) else []
                    lines.append(
                        f"- `{item.get('pair')} {item.get('direction')}` state=`{item.get('coverage_state')}` "
                        f"managed_net=`{item.get('managed_net_jpy')}` supports=`{item.get('matrix_support_count')}` "
                        f"rejects=`{item.get('matrix_reject_count')}` layers=`{item.get('matrix_support_layers') or []}`"
                    )
                    if contexts:
                        lines.append(f"  - same-side context: {', '.join(str(ref) for ref in contexts[:3])}")
                    if blockers:
                        lines.append(f"  - repair blocker: {blockers[0]}")
                lines.append("")
        lines.extend(
            [
                "## Blockers",
                "",
            ]
        )
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
        entry=_optional_float(intent.get("entry")),
        tp=_optional_float(intent.get("tp")),
        sl=_optional_float(intent.get("sl")),
        units=units,
        risk_jpy=risk_jpy,
        reward_jpy=reward_jpy,
        reward_risk=reward_risk,
        opportunity_mode=_opportunity_mode(result, reward_risk),
        issue_codes=tuple(_result_issue_codes(result)),
        counts_live_ready=status == "LIVE_READY" and not blockers,
        counts_after_promotion=(
            status == "DRY_RUN_PASSED"
            and not _has_risk_block(result)
            and not _has_non_promotable_risk(result)
        ),
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


def _result_issue_codes(result: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for issue_key in ("risk_issues", "strategy_issues"):
        for issue in result.get(issue_key, []) or []:
            if not isinstance(issue, dict):
                continue
            code = str(issue.get("code") or issue.get("message") or "").strip()
            if code:
                codes.append(code)
    return codes


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
    duplicate_live_ready_lanes: int,
    sequential_ladder: dict[str, Any],
    lanes: tuple[CoverageLane, ...],
    replay_gap: str | None,
    artifact_diagnostics: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if artifact_diagnostics.get("market_context_matrix_missing"):
        blockers.append("market context matrix artifact is missing; refresh market-context-matrix before judging GPT/intent coverage")
    if artifact_diagnostics.get("all_lanes_spread_blocked"):
        market_status = artifact_diagnostics.get("current_market_status")
        is_fx_open = market_status.get("is_fx_open") if isinstance(market_status, dict) else None
        if is_fx_open is False:
            blockers.append("current FX market is closed and all intent lanes are spread-blocked; refresh broker truth after market open before judging strategy coverage")
        if artifact_diagnostics.get("intents_artifact_stale"):
            blockers.append("order_intents artifact is stale and all intent lanes are spread-blocked; rerun broker-snapshot/generate-intents before judging strategy coverage")
        if is_fx_open is not False and not artifact_diagnostics.get("intents_artifact_stale"):
            blockers.append("all current intent lanes are spread-blocked; refresh broker truth after spreads normalize before judging strategy coverage")
    if artifact_diagnostics.get("all_lanes_quote_stale"):
        if artifact_diagnostics.get("intents_artifact_stale"):
            blockers.append("order_intents artifact is stale and all intent lanes are quote-stale; rerun broker-snapshot/generate-intents before judging strategy coverage")
        else:
            blockers.append("all current intent lanes are quote-stale; rerun broker-snapshot/generate-intents before judging strategy coverage")
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
    duplicate_live_ready_lanes: int,
    sequential_ladder: dict[str, Any],
    opportunity_modes: dict[str, Any],
    lanes: tuple[CoverageLane, ...],
    replay_gap: str | None,
    artifact_diagnostics: dict[str, Any],
) -> list[str]:
    items: list[str] = []
    promotion_candidates = [lane for lane in lanes if lane.counts_after_promotion]
    evidence_refresh_required = bool(artifact_diagnostics.get("requires_market_evidence_refresh"))
    if artifact_diagnostics.get("market_context_matrix_missing"):
        items.append("run market-context-matrix so gold/oil/SPX/DXY/rates/news context reaches coverage and GPT packets")
    if evidence_refresh_required:
        items.append("refresh broker-snapshot and generate-intents after quotes and spreads are tradable; recompute coverage before adding new strategy lanes")
    if duplicate_live_ready_lanes:
        items.append("dedupe same entry/tp/sl receipts before considering multi-entry execution")
    if promotion_candidates:
        items.append(f"promote {len(promotion_candidates)} dry-run receipts only after their strategy blockers clear")
    if live_ready_reward < remaining_target:
        if evidence_refresh_required:
            items.append("defer live-ready reward sizing until fresh spreads and quotes separate market-closure noise from true discovery failure")
            spread_candidates = int(artifact_diagnostics.get("spread_normalized_candidate_count") or 0)
            spread_reward = float(artifact_diagnostics.get("spread_normalized_candidate_reward_jpy") or 0.0)
            no_live_blocker_candidates = int(artifact_diagnostics.get("spread_normalized_no_live_blocker_count") or 0)
            if spread_candidates:
                items.append(
                    f"after spread refresh, re-evaluate {spread_candidates} spread-normalized candidates "
                    f"({spread_reward:.0f} JPY reward; {no_live_blocker_candidates} have no remaining live blockers)"
                )
                blocker_counts = artifact_diagnostics.get("spread_normalized_live_blocker_counts")
                if isinstance(blocker_counts, dict) and blocker_counts:
                    top_blockers = ", ".join(f"{key}={value}" for key, value in list(blocker_counts.items())[:4])
                    items.append(
                        "repair spread-normalized live blockers before treating spread refresh as enough: "
                        f"{top_blockers}"
                    )
            quote_candidates = int(artifact_diagnostics.get("quote_normalized_candidate_count") or 0)
            quote_reward = float(artifact_diagnostics.get("quote_normalized_candidate_reward_jpy") or 0.0)
            quote_no_live_blocker_candidates = int(
                artifact_diagnostics.get("quote_normalized_no_live_blocker_count") or 0
            )
            if quote_candidates:
                items.append(
                    f"after quote refresh, re-evaluate {quote_candidates} quote-normalized candidates "
                    f"({quote_reward:.0f} JPY reward; {quote_no_live_blocker_candidates} have no remaining live blockers)"
                )
                quote_blocker_counts = artifact_diagnostics.get("quote_normalized_live_blocker_counts")
                if isinstance(quote_blocker_counts, dict) and quote_blocker_counts:
                    top_blockers = ", ".join(f"{key}={value}" for key, value in list(quote_blocker_counts.items())[:4])
                    items.append(
                        "repair quote-normalized live blockers before treating quote refresh as enough: "
                        f"{top_blockers}"
                    )
        else:
            reward_gap = remaining_target - live_ready_reward
            avg_reward = _average_reward(lanes) or 1.0
            items.append(f"build at least {math.ceil(reward_gap / avg_reward)} additional live-ready trigger receipts")
    mode_repair = _opportunity_mode_repair_item(opportunity_modes)
    if mode_repair and not evidence_refresh_required:
        items.append(mode_repair)
    profitable_bucket_diag = (
        artifact_diagnostics.get("profitable_bucket_coverage")
        if isinstance(artifact_diagnostics.get("profitable_bucket_coverage"), dict)
        else {}
    )
    if remaining_target > live_ready_reward and profitable_bucket_diag:
        repair_queue = profitable_bucket_diag.get("matrix_supported_repair_queue")
        if isinstance(repair_queue, list) and repair_queue:
            labels = []
            for item in repair_queue[:4]:
                if not isinstance(item, dict):
                    continue
                label = f"{item.get('pair')} {item.get('direction')}"
                blockers = item.get("top_blockers") if isinstance(item.get("top_blockers"), list) else []
                if blockers:
                    label += f" ({blockers[0]})"
                labels.append(label)
            if labels:
                items.append(
                    "prioritize matrix-supported profitable repairs before broad exploration: "
                    + "; ".join(labels)
                )
        blocked_top = profitable_bucket_diag.get("blocked_or_missing_top")
        if isinstance(blocked_top, list) and blocked_top:
            labels = []
            for item in blocked_top[:4]:
                if not isinstance(item, dict):
                    continue
                label = f"{item.get('pair')} {item.get('direction')} {item.get('coverage_state')}"
                blockers = item.get("top_blockers") if isinstance(item.get("top_blockers"), list) else []
                if blockers:
                    label += f" ({blockers[0]})"
                labels.append(label)
            if labels:
                items.append(
                    "repair historical-profitable bucket coverage before widening discovery: "
                    + "; ".join(labels)
                )
        discovery_edges = profitable_bucket_diag.get("discovery_evidence_edges")
        if isinstance(discovery_edges, list) and discovery_edges:
            labels = []
            for item in discovery_edges[:4]:
                if not isinstance(item, dict):
                    continue
                labels.append(
                    f"{item.get('bucket')} net={float(item.get('managed_net_jpy') or 0.0):.0f}"
                )
            if labels:
                items.append(
                    "promote advisory discovery evidence into primary selection tests before live sizing: "
                    + "; ".join(labels)
                )
    if potential_reward < remaining_target and not evidence_refresh_required:
        items.append("expand lane generation across timing windows or pairs; current repaired ladder cannot cover target")
    if live_ready_risk > remaining_risk_budget > 0 and float(sequential_ladder.get("reward_jpy") or 0.0) >= remaining_target:
        items.append("execute coverage as a sequential ladder; do not deploy all live-ready lanes as simultaneous exposure")
    blocked_pairs = sorted({lane.pair for lane in lanes if lane.blockers and not lane.counts_live_ready})
    if blocked_pairs and not evidence_refresh_required:
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
    if _has_market_evidence_gap(hard):
        return "COVERAGE_GAP"
    if (live_ready_reward >= remaining_target or potential_reward >= remaining_target) and _has_replay_evidence_gap(hard):
        return "COVERAGE_REQUIRES_REPLAY_EVIDENCE"
    if potential_reward >= remaining_target:
        return "COVERAGE_REQUIRES_PROFILE_PROMOTION"
    return "COVERAGE_GAP"


def _artifact_diagnostics(
    *,
    intents: dict[str, Any],
    intents_path: Path,
    ai_backtest: dict[str, Any],
    ai_backtest_path: Path,
    strategy_profile: dict[str, Any],
    strategy_profile_path: Path,
    market_context_matrix: dict[str, Any],
    market_context_matrix_path: Path,
    now_utc: datetime,
) -> dict[str, Any]:
    results = tuple(item for item in intents.get("results", []) or [] if isinstance(item, dict))
    intents_generated_at = intents.get("generated_at_utc")
    parsed_generated_at = _parse_iso_datetime(intents_generated_at)
    intents_age_seconds = None
    if parsed_generated_at is not None:
        intents_age_seconds = _round(max(0.0, (now_utc - parsed_generated_at).total_seconds()))
    risk_block_issue_counts = _issue_code_counts(results, "risk_issues", block_only=True)
    strategy_block_issue_counts = _issue_code_counts(results, "strategy_issues", block_only=True)
    spread_normalized = _spread_normalized_candidate_summary(results)
    quote_normalized = _quote_stale_candidate_summary(results)
    all_lanes_spread_blocked = bool(results) and all(
        _result_has_block_issue_code(result, "risk_issues", "SPREAD_TOO_WIDE") for result in results
    )
    all_lanes_quote_stale = bool(results) and all(
        _result_has_any_block_issue_code(result, QUOTE_STALE_ISSUE_CODES) for result in results
    )
    market_status = compute_market_status(now_utc).to_dict()
    intents_artifact_stale = bool(
        intents_age_seconds is not None and intents_age_seconds > COVERAGE_INTENTS_STALE_AFTER_SECONDS
    )
    market_context_matrix_missing = not market_context_matrix_path.exists()
    requires_market_evidence_refresh = bool(
        market_context_matrix_missing
        or all_lanes_spread_blocked
        or all_lanes_quote_stale
    )
    return {
        "intents_path": str(intents_path),
        "intents_generated_at_utc": parsed_generated_at.isoformat() if parsed_generated_at else intents_generated_at,
        "intents_age_seconds": intents_age_seconds,
        "intents_artifact_stale": intents_artifact_stale,
        "intents_stale_after_seconds": COVERAGE_INTENTS_STALE_AFTER_SECONDS,
        "results_count": len(results),
        "status_counts": _status_counts(results),
        "risk_block_issue_counts": risk_block_issue_counts,
        "strategy_block_issue_counts": strategy_block_issue_counts,
        "profitable_bucket_coverage": _profitable_bucket_coverage_summary(
            ai_backtest=ai_backtest,
            ai_backtest_path=ai_backtest_path,
            strategy_profile=strategy_profile,
            strategy_profile_path=strategy_profile_path,
            market_context_matrix=market_context_matrix,
            results=results,
        ),
        **spread_normalized,
        **quote_normalized,
        "all_lanes_spread_blocked": all_lanes_spread_blocked,
        "all_lanes_quote_stale": all_lanes_quote_stale,
        "market_context_matrix_path": str(market_context_matrix_path),
        "market_context_matrix_missing": market_context_matrix_missing,
        "current_market_status": {
            "generated_at_utc": market_status.get("generated_at_utc"),
            "is_fx_open": market_status.get("is_fx_open"),
            "closed_reason": market_status.get("closed_reason"),
            "active_sessions": market_status.get("active_sessions") or [],
            "minutes_to_next_open": market_status.get("minutes_to_next_open"),
            "minutes_to_next_close": market_status.get("minutes_to_next_close"),
        },
        "requires_market_evidence_refresh": requires_market_evidence_refresh,
    }


def _spread_normalized_candidate_summary(results: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    no_live_blocker_candidates: list[dict[str, Any]] = []
    for result in results:
        if not _is_spread_normalized_candidate(result):
            continue
        risk_jpy, reward_jpy, reward_risk = _risk_reward_from_result(result)
        if risk_jpy <= 0 or reward_jpy <= 0:
            continue
        live_blockers = [str(item) for item in result.get("live_blockers", []) or [] if str(item).strip()]
        lane = {
            "lane_id": str(result.get("lane_id") or ""),
            "reward_jpy": reward_jpy,
            "risk_jpy": risk_jpy,
            "reward_risk": reward_risk,
            "live_blockers": live_blockers[:4],
        }
        candidates.append(lane)
        if not live_blockers:
            no_live_blocker_candidates.append(lane)
    top = sorted(candidates, key=lambda item: float(item["reward_jpy"]), reverse=True)[:8]
    live_blocker_counts = Counter(
        blocker
        for item in candidates
        for blocker in item.get("live_blockers", [])
        if isinstance(blocker, str) and blocker
    )
    return {
        "spread_normalized_candidate_count": len(candidates),
        "spread_normalized_candidate_reward_jpy": _round(sum(float(item["reward_jpy"]) for item in candidates)),
        "spread_normalized_no_live_blocker_count": len(no_live_blocker_candidates),
        "spread_normalized_no_live_blocker_reward_jpy": _round(
            sum(float(item["reward_jpy"]) for item in no_live_blocker_candidates)
        ),
        "spread_normalized_top_lane_ids": [str(item["lane_id"]) for item in top],
        "spread_normalized_top_candidates": top,
        "spread_normalized_live_blocker_counts": dict(live_blocker_counts.most_common(12)),
    }


def _quote_stale_candidate_summary(results: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    no_live_blocker_candidates: list[dict[str, Any]] = []
    quote_stale_results = [
        result for result in results if _result_has_any_block_issue_code(result, QUOTE_STALE_ISSUE_CODES)
    ]
    for result in quote_stale_results:
        if not _is_quote_normalized_candidate(result):
            continue
        risk_jpy, reward_jpy, reward_risk = _risk_reward_from_result(result)
        if risk_jpy <= 0 or reward_jpy <= 0:
            continue
        live_blockers = [str(item) for item in result.get("live_blockers", []) or [] if str(item).strip()]
        lane = {
            "lane_id": str(result.get("lane_id") or ""),
            "reward_jpy": reward_jpy,
            "risk_jpy": risk_jpy,
            "reward_risk": reward_risk,
            "live_blockers": live_blockers[:4],
        }
        candidates.append(lane)
        if not live_blockers:
            no_live_blocker_candidates.append(lane)
    top = sorted(candidates, key=lambda item: float(item["reward_jpy"]), reverse=True)[:8]
    live_blocker_counts = Counter(
        blocker
        for item in candidates
        for blocker in item.get("live_blockers", [])
        if isinstance(blocker, str) and blocker
    )
    return {
        "quote_stale_result_count": len(quote_stale_results),
        "quote_normalized_candidate_count": len(candidates),
        "quote_normalized_candidate_reward_jpy": _round(sum(float(item["reward_jpy"]) for item in candidates)),
        "quote_normalized_no_live_blocker_count": len(no_live_blocker_candidates),
        "quote_normalized_no_live_blocker_reward_jpy": _round(
            sum(float(item["reward_jpy"]) for item in no_live_blocker_candidates)
        ),
        "quote_normalized_top_lane_ids": [str(item["lane_id"]) for item in top],
        "quote_normalized_top_candidates": top,
        "quote_normalized_live_blocker_counts": dict(live_blocker_counts.most_common(12)),
    }


def _profitable_bucket_coverage_summary(
    *,
    ai_backtest: dict[str, Any],
    ai_backtest_path: Path,
    strategy_profile: dict[str, Any],
    strategy_profile_path: Path,
    market_context_matrix: dict[str, Any],
    results: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    edges = _profitable_bucket_edges(ai_backtest)
    if not edges:
        return {}
    profile_index = _strategy_profile_edge_index(strategy_profile)
    rows_by_edge: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
        key = (str(intent.get("pair") or ""), str(intent.get("side") or ""))
        if key in edges:
            rows_by_edge.setdefault(key, []).append(result)
    top_edges: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    for key, edge in sorted(edges.items(), key=lambda item: float(item[1]["managed_net_jpy"]), reverse=True):
        rows = tuple(rows_by_edge.get(key, ()))
        summary = _profitable_edge_current_summary(
            edge=edge,
            rows=rows,
            profile_entry=_strategy_profile_edge(profile_index, key),
            matrix_side=_matrix_side_payload(market_context_matrix, key[0], key[1]),
        )
        top_edges.append(summary)
        state_counts[str(summary["coverage_state"])] += 1
    blocked_or_missing = [
        item for item in top_edges if item.get("coverage_state") != "SPREAD_NORMALIZED_NO_LIVE_BLOCKER"
    ]
    matrix_supported_repair_queue = _matrix_supported_repair_queue(blocked_or_missing)
    summaries_by_key = {
        (str(item.get("pair") or ""), str(item.get("direction") or "")): item
        for item in top_edges
        if isinstance(item, dict)
    }
    discovery_evidence_edges = _discovery_evidence_edges(ai_backtest, summaries_by_key)
    return {
        "ai_backtest_path": str(ai_backtest_path) if ai_backtest_path.exists() else None,
        "strategy_profile_path": str(strategy_profile_path) if strategy_profile_path.exists() else None,
        "strategy_profile_missing": not strategy_profile_path.exists(),
        "source_status": str(ai_backtest.get("status") or "UNKNOWN"),
        "live_permission": bool(ai_backtest.get("live_permission") is True),
        "positive_pair_directions": len(edges),
        "positive_managed_net_jpy": _round(sum(float(item["managed_net_jpy"]) for item in edges.values())),
        "positive_trade_count": sum(int(item["trades"]) for item in edges.values()),
        "state_counts": dict(state_counts),
        "top_edges": top_edges[:12],
        "blocked_or_missing_top": blocked_or_missing[:8],
        "matrix_supported_repair_queue": matrix_supported_repair_queue[:8],
        "discovery_evidence_edges": discovery_evidence_edges[:8],
        "discovery_evidence_count": len(discovery_evidence_edges),
        "discovery_evidence_not_summed": True,
    }


def _profitable_bucket_edges(ai_backtest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(ai_backtest, dict) or not ai_backtest:
        return edges
    for item in ai_backtest.get("bucket_contributions", []) or []:
        if not isinstance(item, dict):
            continue
        managed_net = _optional_float(item.get("managed_net_jpy")) or 0.0
        if managed_net <= 0:
            continue
        key = _bucket_pair_direction(item.get("bucket"))
        if key is None:
            continue
        current = edges.setdefault(
            key,
            {
                "pair": key[0],
                "direction": key[1],
                "managed_net_jpy": 0.0,
                "raw_net_jpy": 0.0,
                "trades": 0,
                "days": 0,
                "best_trade_jpy": None,
                "worst_trade_jpy": None,
                "buckets": [],
            },
        )
        current["managed_net_jpy"] = _round(float(current["managed_net_jpy"]) + managed_net)
        current["raw_net_jpy"] = _round(float(current["raw_net_jpy"]) + (_optional_float(item.get("raw_net_jpy")) or 0.0))
        current["trades"] = int(current["trades"]) + int(item.get("trades") or 0)
        current["days"] = max(int(current["days"]), int(item.get("days") or 0))
        best = _optional_float(item.get("best_trade_jpy"))
        if best is not None:
            current["best_trade_jpy"] = (
                best if current["best_trade_jpy"] is None else max(float(current["best_trade_jpy"]), best)
            )
        worst = _optional_float(item.get("worst_trade_jpy"))
        if worst is not None:
            current["worst_trade_jpy"] = (
                worst if current["worst_trade_jpy"] is None else min(float(current["worst_trade_jpy"]), worst)
            )
        current["buckets"].append(str(item.get("bucket") or ""))
    return edges


def _discovery_evidence_edges(
    ai_backtest: dict[str, Any],
    summaries_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    if not isinstance(ai_backtest, dict) or not ai_backtest:
        return edges
    for item in ai_backtest.get("evidence_bucket_contributions", []) or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("bucket") or "")
        parts = label.split(":")
        if len(parts) < 3 or parts[0] not in DISCOVERY_EVIDENCE_SOURCE_TABLES:
            continue
        managed_net = _optional_float(item.get("managed_net_jpy")) or 0.0
        if managed_net <= 0:
            continue
        key = _bucket_pair_direction(label)
        if key is None:
            continue
        current = summaries_by_key.get(key, {})
        edges.append(
            {
                "bucket": label,
                "source_table": parts[0],
                "pair": key[0],
                "direction": key[1],
                "managed_net_jpy": _round(managed_net),
                "raw_net_jpy": _round(_optional_float(item.get("raw_net_jpy")) or 0.0),
                "trades": int(item.get("trades") or 0),
                "days": int(item.get("days") or 0),
                "win_rate_pct": _round(_optional_float(item.get("win_rate_pct")) or 0.0),
                "coverage_state": current.get("coverage_state"),
                "current_lane_count": current.get("current_lane_count"),
                "top_blockers": current.get("top_blockers") if isinstance(current.get("top_blockers"), list) else [],
                "matrix_support_count": current.get("matrix_support_count"),
                "matrix_reject_count": current.get("matrix_reject_count"),
                "matrix_support_context": current.get("matrix_support_context")
                if isinstance(current.get("matrix_support_context"), list)
                else [],
                "matrix_reject_context": current.get("matrix_reject_context")
                if isinstance(current.get("matrix_reject_context"), list)
                else [],
            }
        )
    return sorted(
        edges,
        key=lambda item: (
            -float(item.get("managed_net_jpy") or 0.0),
            -float(item.get("win_rate_pct") or 0.0),
            str(item.get("bucket") or ""),
        ),
    )


def _bucket_pair_direction(bucket: object) -> tuple[str, str] | None:
    parts = str(bucket or "").split(":")
    if len(parts) < 3:
        return None
    pair = parts[1].strip().upper()
    direction = parts[2].strip().upper()
    if not pair or direction not in {"LONG", "SHORT"}:
        return None
    return pair, direction


def _profitable_edge_current_summary(
    *,
    edge: dict[str, Any],
    rows: tuple[dict[str, Any], ...],
    profile_entry: dict[str, Any],
    matrix_side: dict[str, Any],
) -> dict[str, Any]:
    spread_normalized = tuple(result for result in rows if _is_spread_normalized_candidate(result))
    no_live_blocker = tuple(result for result in spread_normalized if not _live_blockers(result))
    if no_live_blocker:
        state = "SPREAD_NORMALIZED_NO_LIVE_BLOCKER"
    elif spread_normalized:
        state = "SPREAD_NORMALIZED_LIVE_BLOCKED"
    elif rows:
        state = "SURFACED_BUT_BLOCKED"
    else:
        state = "NO_CURRENT_LANE"
    rewards = [_risk_reward_from_result(result)[1] for result in rows]
    blocker_counts = Counter(
        blocker
        for result in rows
        for blocker in _current_blocker_labels(result)
        if blocker
    )
    matrix_summary = _matrix_side_summary(matrix_side)
    return {
        "pair": edge["pair"],
        "direction": edge["direction"],
        "coverage_state": state,
        "managed_net_jpy": edge["managed_net_jpy"],
        "raw_net_jpy": edge["raw_net_jpy"],
        "trades": edge["trades"],
        "days": edge["days"],
        "best_trade_jpy": edge["best_trade_jpy"],
        "worst_trade_jpy": edge["worst_trade_jpy"],
        "buckets": edge["buckets"][:4],
        "current_lane_count": len(rows),
        "current_status_counts": _status_counts(rows),
        "current_best_reward_jpy": _round(max(rewards, default=0.0)),
        "spread_normalized_candidate_count": len(spread_normalized),
        "spread_normalized_no_live_blocker_count": len(no_live_blocker),
        "top_blockers": [label for label, _count in blocker_counts.most_common(6)],
        **_strategy_profile_edge_summary(profile_entry),
        **matrix_summary,
    }


def _strategy_profile_edge_index(profile: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in profile.get("profiles", []) or []:
        if not isinstance(item, dict):
            continue
        pair = str(item.get("pair") or "").strip().upper()
        direction = str(item.get("direction") or "").strip().upper()
        if not pair or direction not in {"LONG", "SHORT"}:
            continue
        index.setdefault((pair, direction), []).append(item)
    return index


def _strategy_profile_edge(
    index: dict[tuple[str, str], list[dict[str, Any]]],
    key: tuple[str, str],
) -> dict[str, Any]:
    rows = index.get(key) or []
    if not rows:
        return {}
    pair_level = [item for item in rows if not str(item.get("method") or "").strip()]
    if pair_level:
        return dict(pair_level[0], profile_count=len(rows))
    status_rank = {
        "BLOCK_UNTIL_NEW_EVIDENCE": 0,
        "WATCH_ONLY": 1,
        "RISK_REPAIR_CANDIDATE": 2,
        "MINE_MISSED_EDGE": 3,
        "CANDIDATE": 4,
    }
    chosen = sorted(
        rows,
        key=lambda item: status_rank.get(str(item.get("status") or ""), 99),
    )[0]
    return dict(chosen, profile_count=len(rows))


def _strategy_profile_edge_summary(entry: dict[str, Any]) -> dict[str, Any]:
    if not entry:
        return {
            "strategy_profile_status": None,
            "strategy_profile_required_fix": None,
            "strategy_profile_blocks_live": None,
            "strategy_profile_profile_count": 0,
        }
    status = str(entry.get("status") or "WATCH_ONLY")
    return {
        "strategy_profile_status": status,
        "strategy_profile_method": str(entry.get("method") or "") or None,
        "strategy_profile_required_fix": entry.get("required_fix"),
        "strategy_profile_blocks_live": status not in {"CANDIDATE"},
        "strategy_profile_profile_count": int(entry.get("profile_count") or 1),
        "strategy_profile_pretrade_net_jpy": _optional_float(entry.get("pretrade_net_jpy")),
        "strategy_profile_live_net_jpy": _optional_float(entry.get("live_net_jpy")),
        "strategy_profile_live_worst_jpy": _optional_float(entry.get("live_worst_jpy")),
        "strategy_profile_seat_pl_n": entry.get("seat_pl_n"),
        "strategy_profile_seat_net_jpy": _optional_float(entry.get("seat_net_jpy")),
        "strategy_profile_seat_win_rate_pct": _optional_float(entry.get("seat_win_rate_pct")),
    }


def _matrix_side_payload(matrix: dict[str, Any], pair: str, direction: str) -> dict[str, Any]:
    pairs = matrix.get("pairs") if isinstance(matrix.get("pairs"), dict) else {}
    side_map = pairs.get(pair) if isinstance(pairs.get(pair), dict) else {}
    side = side_map.get(direction) if isinstance(side_map.get(direction), dict) else {}
    return side if isinstance(side, dict) else {}


def _matrix_side_summary(side: dict[str, Any]) -> dict[str, Any]:
    if not side:
        return {
            "matrix_ref": None,
            "matrix_support_count": 0,
            "matrix_reject_count": 0,
            "matrix_warning_count": 0,
            "matrix_strongest_support": None,
            "matrix_strongest_reject": None,
            "matrix_strongest_warning": None,
            "matrix_cross_asset_context": [],
            "matrix_support_context": [],
            "matrix_reject_context": [],
            "matrix_support_layers": [],
            "matrix_reject_layers": [],
            "same_side_matrix_context_supported": False,
        }
    support_context = _matrix_cross_asset_context(side, buckets=("supports",))
    reject_context = _matrix_cross_asset_context(side, buckets=("rejects",))
    support_layers = _matrix_layers(side, bucket="supports")
    reject_layers = _matrix_layers(side, bucket="rejects")
    support_count = int(side.get("support_count") or 0)
    reject_count = int(side.get("reject_count") or 0)
    return {
        "matrix_ref": side.get("evidence_ref"),
        "matrix_support_count": support_count,
        "matrix_reject_count": reject_count,
        "matrix_warning_count": int(side.get("warning_count") or 0),
        "matrix_strongest_support": side.get("strongest_support"),
        "matrix_strongest_reject": side.get("strongest_reject"),
        "matrix_strongest_warning": side.get("strongest_warning"),
        "matrix_cross_asset_context": _matrix_cross_asset_context(side),
        "matrix_support_context": support_context,
        "matrix_reject_context": reject_context,
        "matrix_support_layers": support_layers,
        "matrix_reject_layers": reject_layers,
        "same_side_matrix_context_supported": bool(
            support_context and support_count > reject_count
        ),
    }


def _matrix_cross_asset_context(side: dict[str, Any], *, buckets: tuple[str, ...] | None = None) -> list[str]:
    items: list[str] = []
    for key in buckets or ("supports", "rejects", "warnings", "horizon_conflicts"):
        for item in side.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            layer = str(item.get("layer") or "")
            refs = [str(ref) for ref in item.get("evidence_refs", []) or []]
            if layer not in {"cross_asset", "context_asset_chart"} and not any(
                ref.startswith("context_asset:") or ref.startswith("cross:") for ref in refs
            ):
                continue
            code = str(item.get("code") or layer or "context")
            message = str(item.get("message") or "").strip()
            items.append(f"{code}: {message}" if message else code)
            if len(items) >= 6:
                return items
    return items


def _matrix_layers(side: dict[str, Any], *, bucket: str) -> list[str]:
    layers = {
        str(item.get("layer") or "").strip()
        for item in side.get(bucket, []) or []
        if isinstance(item, dict) and str(item.get("layer") or "").strip()
    }
    return sorted(layers)


def _matrix_supported_repair_queue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if bool(row.get("same_side_matrix_context_supported"))
    ]
    out: list[dict[str, Any]] = []
    for row in sorted(
        candidates,
        key=lambda item: (
            -float(item.get("managed_net_jpy") or 0.0),
            str(item.get("pair") or ""),
            str(item.get("direction") or ""),
        ),
    ):
        out.append(
            {
                "pair": row.get("pair"),
                "direction": row.get("direction"),
                "coverage_state": row.get("coverage_state"),
                "managed_net_jpy": row.get("managed_net_jpy"),
                "current_lane_count": row.get("current_lane_count"),
                "spread_normalized_candidate_count": row.get("spread_normalized_candidate_count"),
                "spread_normalized_no_live_blocker_count": row.get("spread_normalized_no_live_blocker_count"),
                "top_blockers": [str(item) for item in (row.get("top_blockers") or [])[:4] if str(item).strip()],
                "strategy_profile_status": row.get("strategy_profile_status"),
                "strategy_profile_required_fix": row.get("strategy_profile_required_fix"),
                "matrix_ref": row.get("matrix_ref"),
                "matrix_support_count": row.get("matrix_support_count"),
                "matrix_reject_count": row.get("matrix_reject_count"),
                "matrix_support_layers": row.get("matrix_support_layers") if isinstance(row.get("matrix_support_layers"), list) else [],
                "matrix_reject_layers": row.get("matrix_reject_layers") if isinstance(row.get("matrix_reject_layers"), list) else [],
                "matrix_support_context": [
                    str(item) for item in (row.get("matrix_support_context") or [])[:4] if str(item).strip()
                ],
                "matrix_reject_context": [
                    str(item) for item in (row.get("matrix_reject_context") or [])[:4] if str(item).strip()
                ],
            }
        )
    return out


def _is_spread_normalized_candidate(result: dict[str, Any]) -> bool:
    non_spread_risk_blocks = [
        issue
        for issue in result.get("risk_issues", []) or []
        if isinstance(issue, dict)
        and issue.get("severity") == "BLOCK"
        and issue.get("code") != "SPREAD_TOO_WIDE"
    ]
    strategy_blocks = [
        issue
        for issue in result.get("strategy_issues", []) or []
        if isinstance(issue, dict) and issue.get("severity") == "BLOCK"
    ]
    return not non_spread_risk_blocks and not strategy_blocks


def _is_quote_normalized_candidate(result: dict[str, Any]) -> bool:
    block_codes: list[str] = []
    for issue_key in ("risk_issues", "strategy_issues"):
        for issue in result.get(issue_key, []) or []:
            if not isinstance(issue, dict) or issue.get("severity") != "BLOCK":
                continue
            block_codes.append(str(issue.get("code") or issue.get("message") or "UNKNOWN"))
    if not any(code in QUOTE_STALE_ISSUE_CODES for code in block_codes):
        return False
    return all(code in QUOTE_STALE_ISSUE_CODES for code in block_codes)


def _live_blockers(result: dict[str, Any]) -> list[str]:
    return [str(item) for item in result.get("live_blockers", []) or [] if str(item).strip()]


def _current_blocker_labels(result: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for issue in result.get("risk_issues", []) or []:
        if not isinstance(issue, dict) or issue.get("severity") != "BLOCK":
            continue
        code = str(issue.get("code") or issue.get("message") or "risk block")
        if code != "SPREAD_TOO_WIDE":
            labels.append(code)
    for issue in result.get("strategy_issues", []) or []:
        if not isinstance(issue, dict) or issue.get("severity") != "BLOCK":
            continue
        labels.append(str(issue.get("code") or issue.get("message") or "strategy block"))
    labels.extend(_live_blockers(result))
    return labels


def _parse_iso_datetime(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _issue_code_counts(results: tuple[dict[str, Any], ...], key: str, *, block_only: bool) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for result in results:
        for issue in result.get(key, []) or []:
            if not isinstance(issue, dict):
                continue
            if block_only and issue.get("severity") != "BLOCK":
                continue
            code = str(issue.get("code") or issue.get("message") or "UNKNOWN")
            counter[code] += 1
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _status_counts(results: tuple[dict[str, Any], ...]) -> dict[str, int]:
    counter: Counter[str] = Counter(str(result.get("status") or "UNKNOWN") for result in results)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _result_has_block_issue_code(result: dict[str, Any], key: str, code: str) -> bool:
    return any(
        isinstance(issue, dict) and issue.get("severity") == "BLOCK" and issue.get("code") == code
        for issue in result.get(key, []) or []
    )


def _result_has_any_block_issue_code(result: dict[str, Any], codes: frozenset[str]) -> bool:
    return any(
        isinstance(issue, dict)
        and issue.get("severity") == "BLOCK"
        and str(issue.get("code") or "") in codes
        for issue_key in ("risk_issues", "strategy_issues")
        for issue in result.get(issue_key, []) or []
    )


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


def _dedupe_exact_geometry(lanes: tuple[CoverageLane, ...]) -> tuple[CoverageLane, ...]:
    selected: dict[tuple[object, ...], CoverageLane] = {}
    for lane in lanes:
        key = (lane.pair, lane.direction, lane.order_type, lane.entry, lane.tp, lane.sl)
        current = selected.get(key)
        if current is None or (lane.reward_risk, lane.reward_jpy, lane.lane_id) > (
            current.reward_risk,
            current.reward_jpy,
            current.lane_id,
        ):
            selected[key] = lane
    return tuple(selected.values())


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


def _opportunity_mode(result: dict[str, Any], reward_risk: float) -> str:
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    market_context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
    target_intent = str(metadata.get("tp_target_intent") or metadata.get("campaign_role") or "").upper()
    method = str(market_context.get("method") or "").upper()
    if any(token in target_intent for token in ("RUNNER", "TRAIL", "EXTEND", "SWING", "HOLD", "ADD")):
        return "RUNNER"
    if reward_risk >= RUNNER_REWARD_RISK_MIN:
        return "RUNNER"
    if any(token in target_intent for token in ("HARVEST", "SCALP", "QUICK")):
        return "HARVEST"
    if 0.0 < reward_risk <= HARVEST_REWARD_RISK_MAX:
        return "HARVEST"
    if method == "TREND_CONTINUATION":
        return "RUNNER"
    if method in {"RANGE_ROTATION", "BREAKOUT_FAILURE"}:
        return "HARVEST"
    return "BALANCED"


def _opportunity_mode_summary(
    lanes: tuple[CoverageLane, ...],
    remaining_target: float,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {
        mode: {
            "lanes": 0,
            "live_ready_lanes": 0,
            "promotion_candidate_lanes": 0,
            "reward_jpy": 0.0,
            "live_ready_reward_jpy": 0.0,
            "potential_reward_jpy": 0.0,
            "coverage_pct": 0.0,
            "potential_coverage_pct": 0.0,
            "status_counts": {},
            "top_issue_codes": [],
            "top_blockers": [],
            "top_lanes": [],
        }
        for mode in ("HARVEST", "RUNNER", "BALANCED")
    }
    status_counts: dict[str, Counter[str]] = {mode: Counter() for mode in summaries}
    issue_code_counts: dict[str, Counter[str]] = {mode: Counter() for mode in summaries}
    blocker_counts: dict[str, Counter[str]] = {mode: Counter() for mode in summaries}
    top_lanes: dict[str, list[CoverageLane]] = {mode: [] for mode in summaries}
    for lane in lanes:
        mode = lane.opportunity_mode if lane.opportunity_mode in summaries else "BALANCED"
        item = summaries[mode]
        item["lanes"] += 1
        item["reward_jpy"] = _round(float(item["reward_jpy"]) + lane.reward_jpy)
        if lane.counts_live_ready:
            item["live_ready_lanes"] += 1
            item["live_ready_reward_jpy"] = _round(float(item["live_ready_reward_jpy"]) + lane.reward_jpy)
        if lane.counts_live_ready or lane.counts_after_promotion:
            item["potential_reward_jpy"] = _round(float(item["potential_reward_jpy"]) + lane.reward_jpy)
        if lane.counts_after_promotion:
            item["promotion_candidate_lanes"] += 1
        status_counts[mode][lane.status] += 1
        issue_code_counts[mode].update(lane.issue_codes)
        blocker_counts[mode].update(lane.blockers)
        top_lanes[mode].append(lane)
    for mode, item in summaries.items():
        item["coverage_pct"] = (
            _round((float(item["live_ready_reward_jpy"]) / remaining_target) * 100.0)
            if remaining_target
            else 100.0
        )
        item["potential_coverage_pct"] = (
            _round((float(item["potential_reward_jpy"]) / remaining_target) * 100.0)
            if remaining_target
            else 100.0
        )
        item["status_counts"] = dict(sorted(status_counts[mode].items(), key=lambda pair: (-pair[1], pair[0])))
        item["top_issue_codes"] = [
            {"code": code, "count": count}
            for code, count in issue_code_counts[mode].most_common(12)
        ]
        item["top_blockers"] = [
            {"label": label, "count": count}
            for label, count in blocker_counts[mode].most_common(6)
        ]
        item["top_lanes"] = [
            {
                "lane_id": lane.lane_id,
                "status": lane.status,
                "reward_jpy": lane.reward_jpy,
                "reward_risk": lane.reward_risk,
                "blockers": list(lane.blockers[:3]),
            }
            for lane in sorted(top_lanes[mode], key=lambda lane: (lane.reward_jpy, lane.reward_risk), reverse=True)[:6]
        ]
    return summaries


def _opportunity_mode_repair_item(opportunity_modes: dict[str, Any]) -> str | None:
    labels: list[str] = []
    for mode in ("HARVEST", "RUNNER"):
        item = opportunity_modes.get(mode) if isinstance(opportunity_modes.get(mode), dict) else {}
        if int(item.get("lanes") or 0) <= 0 or int(item.get("live_ready_lanes") or 0) > 0:
            continue
        issue_codes = item.get("top_issue_codes") if isinstance(item.get("top_issue_codes"), list) else []
        code_labels = [
            str(issue.get("code"))
            for issue in issue_codes[:3]
            if isinstance(issue, dict) and str(issue.get("code") or "").strip()
        ]
        if code_labels:
            reason_text = "top codes: " + ", ".join(code_labels)
        else:
            blockers = item.get("top_blockers") if isinstance(item.get("top_blockers"), list) else []
            blocker_labels = [
                str(blocker.get("label"))
                for blocker in blockers[:2]
                if isinstance(blocker, dict) and str(blocker.get("label") or "").strip()
            ]
            reason_text = "top blockers: " + (", ".join(blocker_labels) if blocker_labels else "no explicit blocker label")
        labels.append(f"{mode} lanes={int(item.get('lanes') or 0)} {reason_text}")
    if len(labels) >= 2:
        return "repair both harvest and runner opportunity paths instead of treating coverage as one pool: " + "; ".join(labels)
    if labels:
        return "repair the visible opportunity path before broad exploration: " + labels[0]
    return None


def _has_intent(item: object) -> bool:
    return isinstance(item, dict) and isinstance(item.get("intent"), dict)


def _has_risk_block(result: dict[str, Any]) -> bool:
    return any(
        isinstance(issue, dict) and issue.get("severity") == "BLOCK"
        for issue in result.get("risk_issues", []) or []
    )


def _has_non_promotable_risk(result: dict[str, Any]) -> bool:
    """Return True when strategy promotion cannot clear the live-readiness gap."""

    for issue in result.get("risk_issues", []) or []:
        if not isinstance(issue, dict):
            continue
        if str(issue.get("code") or "") in NON_PROMOTABLE_RISK_CODES:
            return True
    return False


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _has_replay_evidence_gap(blockers: list[str]) -> bool:
    return any(item.startswith("replay evidence covers target") for item in blockers)


def _has_market_evidence_gap(blockers: list[str]) -> bool:
    return any(
        item.startswith("market context matrix artifact is missing")
        or item.startswith("current FX market is closed")
        or item.startswith("order_intents artifact is stale")
        or item.startswith("all current intent lanes are spread-blocked")
        or item.startswith("all current intent lanes are quote-stale")
        for item in blockers
    )


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _round(value: float) -> float:
    return round(value, 4)

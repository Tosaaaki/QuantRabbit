#!/usr/bin/env python3
"""Build read-only 4x A/S proof-path artifacts.

The script reads local QuantRabbit artifacts and the execution ledger. It does
not call OANDA, stage orders, send orders, cancel orders, close trades, or
modify SL/TP.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AS_LOOP_PATH = ROOT / "tools" / "build_as_live_ready_evidence_loop.py"
MANUAL_TRADE_ID = "472987"
MANUAL_TP_ORDER_ID = "472988"
HISTORICAL_TARGETS = (
    ("range_trader:GBP_USD:LONG:RANGE_ROTATION", "UP"),
    ("trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION", "DOWN"),
)
AUDJPY_REPAIR_LANES = (
    "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE",
    "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT",
)


def main() -> int:
    generated_at = _now()
    as_loop = _load_as_loop()
    context = _context(as_loop)
    payloads = {
        "data/post_gate_expectancy_gap_trace.json": build_post_gate_gap_trace(generated_at, context),
        "data/historical_only_to_fresh_proof_replay.json": build_historical_only_replay(generated_at, context),
        "data/audjpy_short_breakout_failure_repair_proof.json": build_audjpy_repair(generated_at, context),
        "data/audjpy_short_breakout_failure_limit_proof_pack.json": build_audjpy_limit_proof_pack(generated_at, context),
        "data/portfolio_4x_path_planner.json": build_portfolio_planner(generated_at, context),
    }
    markdown = {
        "docs/post_gate_expectancy_gap_trace.md": post_gate_gap_md(payloads["data/post_gate_expectancy_gap_trace.json"]),
        "docs/historical_only_to_fresh_proof_replay.md": historical_replay_md(payloads["data/historical_only_to_fresh_proof_replay.json"]),
        "docs/audjpy_short_breakout_failure_repair_proof.md": audjpy_repair_md(payloads["data/audjpy_short_breakout_failure_repair_proof.json"]),
        "docs/audjpy_short_breakout_failure_limit_proof_pack.md": audjpy_limit_proof_pack_md(payloads["data/audjpy_short_breakout_failure_limit_proof_pack.json"]),
        "docs/portfolio_4x_path_planner.md": portfolio_planner_md(payloads["data/portfolio_4x_path_planner.json"]),
    }
    for rel, payload in payloads.items():
        _write_json(rel, payload)
        print(f"wrote {rel}")
    for rel, text in markdown.items():
        _write_text(rel, text)
        print(f"wrote {rel}")
    return 0


def build_post_gate_gap_trace(generated_at: str, ctx: dict[str, Any]) -> dict[str, Any]:
    as_loop = ctx["as_loop"]
    attributed = ctx["attributed_outcomes"]
    blocked = ctx["blocked"]
    excluded_ids = {MANUAL_TRADE_ID} | blocked["market_close_trade_ids"] | blocked["residual_trade_ids"]
    post_gate_rows = [row for row in attributed if row.trade_id not in excluded_ids]
    scope_metrics = as_loop._bucket_metrics(post_gate_rows)
    families = _family_rollups(post_gate_rows, as_loop)
    trades = [_outcome_dict(row) for row in sorted(post_gate_rows, key=lambda item: item.realized_pl_jpy)]
    net = _float(scope_metrics.get("net_jpy")) or 0.0
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_post_gate_expectancy_gap_trace",
        "source_artifacts": [
            "data/execution_ledger.db",
            "data/post_gate_capture_economics_decomposition.json",
            "data/profitability_acceptance.json",
            "data/month_scale_residual_family_table.json",
            "data/market_close_leak_trade_table.json",
        ],
        "scope": {
            "name": "manual_excluded_plus_both_market_close_leak_and_residual_family_filters",
            "definition": (
                "Trader-attributed realized rows excluding the operator-manual EUR_USD trade, "
                "current MARKET_CLOSE_LEAK_FAMILY_BLOCKED trade IDs, and current month-scale "
                "residual-family blocked trade IDs."
            ),
            "metrics": scope_metrics,
            "remaining_gap_to_zero_jpy": _round(max(0.0, -net)),
            "negative_expectancy_active_should_remain": True,
            "can_create_live_permission": False,
        },
        "blocked_trade_sets": {
            "manual_trade_ids": [MANUAL_TRADE_ID],
            "market_close_leak_trade_ids": sorted(blocked["market_close_trade_ids"]),
            "residual_family_trade_ids": sorted(blocked["residual_trade_ids"]),
        },
        "family_trace": families,
        "trade_trace": trades,
        "largest_loss_trades": trades[:20],
        "positive_offset_families": [row for row in sorted(families, key=lambda item: item["net_jpy"], reverse=True) if row_positive(row)][:12],
        "manual_position_safety": manual_position_safety(ctx["broker"]),
        "permission_boundary": (
            "The post-filter scope is still negative by -1194.4656 JPY at the current ledger state. "
            "Even if a filtered diagnostic scope becomes non-negative, it cannot create live permission "
            "without regenerated acceptance, fresh lane proof, risk pass, gateway pass, GPT verifier, and guardian clearance."
        ),
        "live_side_effects": [],
    }


def build_historical_only_replay(generated_at: str, ctx: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for lane_id, direction in HISTORICAL_TARGETS:
        candidate = ctx["candidate_by_lane"].get(lane_id) or {}
        proof_queue_row = ctx["proof_by_lane"].get(lane_id) or {}
        row = _historical_target_row(lane_id, direction, candidate, proof_queue_row, ctx)
        rows.append(row)
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_historical_only_to_fresh_proof_replay",
        "source_artifacts": [
            "data/rolling_30d_4x_firepower_board.json",
            "data/as_proof_pack_queue.json",
            "data/order_intents.json",
            str(ctx["fresh_replay_relpath"]),
            "data/execution_timing_audit.json",
            "data/memory_health.json",
            "data/broker_snapshot.json",
        ],
        "fresh_replay_summary": replay_summary(ctx["fresh_replay"], ctx["fresh_replay_relpath"]),
        "rolling_30d_target_math": ctx["firepower"].get("target_math"),
        "execution_timing_744h_summary": timing_summary(ctx["timing"]),
        "classification_values": ["PROOF_READY", "REPAIR_REQUIRED", "EVIDENCE_GAP", "REJECTED", "HISTORICAL_ONLY"],
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "proof_ready": sum(1 for row in rows if row["classification"] == "PROOF_READY"),
            "historical_only": sum(1 for row in rows if row["classification"] == "HISTORICAL_ONLY"),
            "can_create_live_permission": 0,
            "fresh_s5_price_truth_status": _nested(ctx["fresh_replay"], "price_truth_coverage", "status"),
            "fresh_s5_adoption_level": _nested(ctx["fresh_replay"], "price_truth_coverage", "adoption_level"),
        },
        "live_side_effects": [],
    }


def build_audjpy_repair(generated_at: str, ctx: dict[str, Any]) -> dict[str, Any]:
    required_pct = _float(_nested(ctx["firepower"], "target_math", "required_calendar_daily_return_funding_adjusted_pct"))
    rows = []
    for lane_id in AUDJPY_REPAIR_LANES:
        candidate = ctx["candidate_by_lane"].get(lane_id) or {}
        queue_row = ctx["proof_by_lane"].get(lane_id) or {}
        fresh = fresh_direction_evidence(ctx["fresh_replay"], "AUD_JPY", "DOWN")
        daily_pct = _float(candidate.get("expected_daily_return_pct_on_funding_adjusted_equity"))
        rows.append(
            {
                "lane_id": lane_id,
                "order_type": candidate.get("order_type"),
                "classification": "REPAIR_REQUIRED",
                "decision": "portfolio_component_after_repair_only",
                "standalone_4x": bool(daily_pct is not None and required_pct is not None and daily_pct >= required_pct),
                "standalone_4x_gap_pct": _round((required_pct or 0.0) - (daily_pct or 0.0)),
                "portfolio_component_candidate_after_repair": True,
                "can_enter_proof_pack": bool(queue_row.get("can_enter_proof_pack")),
                "can_create_live_permission": False,
                "expected_daily_return_pct_on_funding_adjusted_equity": daily_pct,
                "expected_jpy_per_trade": candidate.get("expected_jpy_per_trade"),
                "estimated_trades_per_day_available": candidate.get("estimated_trades_per_day_available"),
                "required_trades_per_day_to_contribute_to_30d_4x": candidate.get("required_trades_per_day_to_contribute_to_30d_4x"),
                "margin": {
                    "realistic_units": candidate.get("realistic_units"),
                    "margin_requirement_realistic_size_jpy": candidate.get("margin_requirement_realistic_size_jpy"),
                    "margin_requirement_min_lot_jpy": candidate.get("margin_requirement_min_lot_jpy"),
                    "broker_margin_context": candidate.get("broker_margin_context"),
                    "risk_allowed": candidate.get("risk_allowed"),
                },
                "fresh_s5_bidask_replay": fresh,
                "packaged_source_evidence": candidate.get("source_evidence"),
                "current_blockers": candidate.get("current_blockers") or [],
                "missing_proof": queue_row.get("missing_proof") or {},
                "repair_requirements": audjpy_repair_requirements(candidate, queue_row, fresh),
            }
        )
    preferred = min(rows, key=lambda item: (len(item["current_blockers"]), item["order_type"] != "LIMIT")) if rows else None
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_audjpy_short_breakout_failure_repair_proof",
        "source_artifacts": [
            "data/rolling_30d_4x_firepower_board.json",
            "data/as_proof_pack_queue.json",
            "data/order_intents.json",
            str(ctx["fresh_replay_relpath"]),
            "data/broker_snapshot.json",
        ],
        "target_math": ctx["firepower"].get("target_math"),
        "summary": {
            "status": "REPAIR_REQUIRED",
            "standalone_4x_path_exists": any(row["standalone_4x"] for row in rows),
            "portfolio_component_after_repair_exists": bool(rows),
            "can_create_live_permission": False,
            "preferred_repair_shape": preferred.get("lane_id") if preferred else None,
            "reason": "AUD_JPY SHORT BREAKOUT_FAILURE has proof-collection value but no fresh live-grade S5 proof, no fresh executable forecast, no risk/gateway/GPT pass, and guardian remains blocked.",
        },
        "rows": rows,
        "live_side_effects": [],
    }


def build_audjpy_limit_proof_pack(generated_at: str, ctx: dict[str, Any]) -> dict[str, Any]:
    lane_id = "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT"
    candidate = ctx["candidate_by_lane"].get(lane_id) or {}
    queue_row = ctx["proof_by_lane"].get(lane_id) or {}
    intent_row = ctx["intent_by_lane"].get(lane_id) or {}
    risk = intent_row.get("risk_metrics") if isinstance(intent_row.get("risk_metrics"), dict) else {}
    intent = intent_row.get("intent") if isinstance(intent_row.get("intent"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    fresh = fresh_direction_evidence(ctx["fresh_replay"], "AUD_JPY", "DOWN")
    missing = queue_row.get("missing_proof") or {}
    blockers = candidate.get("current_blockers") or []
    expected_jpy = _float(candidate.get("expected_jpy_per_trade"))
    trades_per_day = _float(candidate.get("estimated_trades_per_day_available"))
    active_day_contribution = expected_jpy * trades_per_day if expected_jpy is not None and trades_per_day is not None else None
    direction_coverage = fresh.get("direction_coverage") if isinstance(fresh.get("direction_coverage"), dict) else {}
    forecast_coverage = (
        ctx["fresh_replay"].get("forecast_sample_coverage")
        if isinstance(ctx["fresh_replay"].get("forecast_sample_coverage"), dict)
        else {}
    )
    min_directional_samples = _int(forecast_coverage.get("min_directional_samples_for_precision_rule")) or 30
    min_active_days = _int(forecast_coverage.get("min_active_days_for_daily_stability")) or 3
    fresh_samples = _int(direction_coverage.get("evaluated_samples")) or 0
    fresh_active_days = _int(direction_coverage.get("evaluated_active_days")) or 0
    daily_stability_status = _nested(fresh, "contrarian_or_rank_only_rules", 0, "daily_stability_status")
    stale_quote_blocked = "STALE_QUOTE" in blockers or "TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE" in blockers
    range_forecast_requires_range_rotation = "RANGE_FORECAST_REQUIRES_RANGE_ROTATION" in blockers
    forecast_market_support_ok = metadata.get("forecast_market_support_ok")
    forecast_executable = (
        missing.get("forecast_executable_proof") is True
        and forecast_market_support_ok is not False
        and not stale_quote_blocked
        and not range_forecast_requires_range_rotation
    )
    daily_stability_ok = bool(fresh.get("live_grade_support") is True and daily_stability_status not in {"INSUFFICIENT_ACTIVE_DAYS", "REJECTED_DAILY_STABILITY"})
    checks = {
        "fresh_744h_replay": bool(missing.get("fresh_744h_replay") is True),
        "s5_bidask_spread_included_replay": bool(missing.get("s5_bidask_spread_included_replay") is True),
        "sample_count_floor": fresh_samples >= min_directional_samples,
        "active_day_floor": fresh_active_days >= min_active_days,
        "daily_stability_floor": daily_stability_ok,
        "forecast_executability": forecast_executable,
        "geometry_proof": bool(missing.get("geometry_proof") is True),
        "attached_tp_proof": bool(missing.get("attached_tp_proof") is True),
        "reward_risk": bool((_float(risk.get("reward_risk")) or 0.0) >= 1.0),
        "margin_feasibility": bool(candidate.get("realistic_units") and candidate.get("margin_requirement_realistic_size_jpy")),
        "risk_engine_pass": bool(missing.get("risk_engine_pass") is True),
        "live_order_gateway_pass": bool(missing.get("live_order_gateway_pass") is True),
        "gpt_verifier_pass": bool(missing.get("gpt_verifier_pass") is True),
        "guardian_operator_review_clear": bool(missing.get("no_guardian_operator_review_blocker") is True),
    }
    failed = [key for key, value in checks.items() if value is not True]
    if candidate.get("source_evidence", {}).get("historical_only"):
        classification = "HISTORICAL_ONLY"
    elif not candidate:
        classification = "EVIDENCE_GAP"
    elif any(code in blockers for code in ("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION")):
        classification = "PORTFOLIO_COMPONENT_REPAIR_REQUIRED"
    elif failed:
        classification = "EVIDENCE_GAP"
    else:
        classification = "PROOF_READY"
    if classification == "PROOF_READY" and (checks["risk_engine_pass"] is not True or checks["live_order_gateway_pass"] is not True):
        classification = "PORTFOLIO_COMPONENT_REPAIR_REQUIRED"
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_audjpy_short_breakout_failure_limit_proof_pack",
        "lane_id": lane_id,
        "source_artifacts": [
            "data/order_intents.json",
            "data/as_proof_pack_queue.json",
            "data/rolling_30d_4x_firepower_board.json",
            str(ctx["fresh_replay_relpath"]),
            "data/execution_timing_audit.json",
            "data/broker_snapshot.json",
        ],
        "classification_values": [
            "PROOF_READY",
            "PORTFOLIO_COMPONENT_REPAIR_REQUIRED",
            "EVIDENCE_GAP",
            "REJECTED",
            "HISTORICAL_ONLY",
        ],
        "classification": classification,
        "standalone_4x": False,
        "portfolio_component_possible_after_repair": classification == "PORTFOLIO_COMPONENT_REPAIR_REQUIRED",
        "can_create_live_permission": False,
        "decision": "keep_as_repair_required_no_permission",
        "fresh_744h_replay": timing_summary(ctx["timing"]),
        "fresh_s5_bidask_spread_included_replay": fresh,
        "sample_and_stability": {
            "candidate_samples": candidate.get("sample_count"),
            "candidate_active_days": candidate.get("active_days"),
            "fresh_replay_evaluated_samples": _nested(fresh, "direction_coverage", "evaluated_samples"),
            "fresh_replay_evaluated_active_days": _nested(fresh, "direction_coverage", "evaluated_active_days"),
            "min_directional_samples_for_precision_rule": min_directional_samples,
            "min_active_days_for_daily_stability": min_active_days,
            "daily_stability_status": daily_stability_status,
            "daily_stability_gap": _nested(fresh, "contrarian_or_rank_only_rules", 0, "daily_stability_gap"),
        },
        "forecast_executability": {
            "forecast_direction": metadata.get("forecast_direction"),
            "forecast_confidence": metadata.get("forecast_confidence"),
            "forecast_market_support_ok": forecast_market_support_ok,
            "stale_quote_blocked": stale_quote_blocked,
            "range_forecast_requires_range_rotation": range_forecast_requires_range_rotation,
            "executable": forecast_executable,
        },
        "geometry": {
            "order_type": candidate.get("order_type"),
            "entry": intent.get("entry"),
            "tp": intent.get("tp"),
            "sl": intent.get("sl"),
            "reward_pips": risk.get("reward_pips"),
            "loss_pips": risk.get("loss_pips"),
            "reward_risk": risk.get("reward_risk"),
            "spread_pips": risk.get("spread_pips"),
            "blockers": [code for code in blockers if "CHASE" in code or "HARVEST_TP_STRUCTURE" in code or "RANGE_FORECAST" in code],
        },
        "attached_tp_proof": {
            "attach_take_profit_on_fill": metadata.get("attach_take_profit_on_fill"),
            "capture_take_profit_scope_key": metadata.get("capture_take_profit_scope_key"),
            "capture_take_profit_trades": metadata.get("capture_take_profit_trades"),
            "capture_take_profit_wins": metadata.get("capture_take_profit_wins"),
            "capture_take_profit_expectancy_jpy": metadata.get("capture_take_profit_expectancy_jpy"),
            "positive_rotation_mode": metadata.get("positive_rotation_mode"),
            "proof_collection_ready": metadata.get("positive_rotation_proof_collection_ready"),
        },
        "economics": {
            "expected_jpy_per_trade": candidate.get("expected_jpy_per_trade"),
            "estimated_trades_per_day_available": candidate.get("estimated_trades_per_day_available"),
            "expected_active_day_contribution_jpy": _round(active_day_contribution),
            "expected_daily_return_pct_on_funding_adjusted_equity": candidate.get("expected_daily_return_pct_on_funding_adjusted_equity"),
            "required_calendar_daily_return_pct": _nested(ctx["firepower"], "target_math", "required_calendar_daily_return_funding_adjusted_pct"),
            "required_trades_per_day_to_contribute_to_30d_4x": candidate.get("required_trades_per_day_to_contribute_to_30d_4x"),
        },
        "margin_and_risk": {
            "realistic_units": candidate.get("realistic_units"),
            "risk_allowed": candidate.get("risk_allowed"),
            "risk_jpy": risk.get("risk_jpy"),
            "estimated_margin_jpy": candidate.get("margin_requirement_realistic_size_jpy"),
            "margin_requirement_min_lot_jpy": candidate.get("margin_requirement_min_lot_jpy"),
            "broker_margin_context": candidate.get("broker_margin_context"),
        },
        "verifier_gateway_guardian": {
            "risk_engine_pass": checks["risk_engine_pass"],
            "live_order_gateway_pass": checks["live_order_gateway_pass"],
            "gpt_verifier_pass": checks["gpt_verifier_pass"],
            "guardian_operator_review_clear": checks["guardian_operator_review_clear"],
            "blockers": blockers,
        },
        "proof_checks": checks,
        "failed_checks": failed,
        "missing_proof": missing,
        "live_side_effects": [],
    }


def build_portfolio_planner(generated_at: str, ctx: dict[str, Any]) -> dict[str, Any]:
    all_candidates = all_firepower_candidates(ctx)
    rankings = [portfolio_rank_row(row, ctx) for row in all_candidates]
    rankings.sort(key=lambda item: (-item["rank_score"], item["proof_distance"], item["lane_id"]))
    required_pct = _float(_nested(ctx["firepower"], "target_math", "required_calendar_daily_return_funding_adjusted_pct")) or 0.0
    standalone = [row for row in rankings if (row.get("expected_daily_return_pct_on_funding_adjusted_equity") or 0.0) >= required_pct]
    mathematical_basket = accumulate_unique_basket(rankings, required_pct)
    repair_only_basket = accumulate_unique_basket(
        [row for row in rankings if row["proof_classification"] == "REPAIR_REQUIRED"],
        required_pct,
    )
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_portfolio_4x_path_planner",
        "source_artifacts": [
            "data/order_intents.json",
            "data/rolling_30d_4x_firepower_board.json",
            "data/as_proof_pack_queue.json",
            "data/audjpy_short_breakout_failure_limit_proof_pack.json",
            "data/profitability_acceptance.json",
            "data/trader_support_bot.json",
            "data/broker_snapshot.json",
        ],
        "target_math": ctx["firepower"].get("target_math"),
        "portfolio_status": "NO_LIVE_READY_PORTFOLIO",
        "can_reach_4x_now": False,
        "normal_routing_status": "BLOCKED",
        "live_ready_lanes": 0,
        "ranking_contract": "Only non-hard-excluded candidates are ranked. Ranking is repair-priority only and cannot create permission.",
        "summary": {
            "non_hard_excluded_candidates": len(all_candidates),
            "standalone_math_candidates_meeting_required_return": len(standalone),
            "standalone_live_ready_candidates": 0,
            "proof_ready_candidates": sum(1 for row in rankings if row["proof_classification"] == "PROOF_READY"),
            "repair_required_candidates": sum(1 for row in rankings if row["proof_classification"] == "REPAIR_REQUIRED"),
            "historical_only_candidates": sum(1 for row in rankings if row["proof_classification"] == "HISTORICAL_ONLY"),
            "can_create_live_permission": False,
        },
        "standalone_math_candidates": standalone[:8],
        "fastest_mathematical_basket": mathematical_basket,
        "repair_only_basket": repair_only_basket,
        "candidate_rankings": rankings[:30],
        "global_blockers": global_blockers(ctx),
        "manual_position_safety": manual_position_safety(ctx["broker"]),
        "live_side_effects": [],
    }


def _context(as_loop: Any) -> dict[str, Any]:
    acceptance = _load_json("data/profitability_acceptance.json")
    residual = _load_json("data/month_scale_residual_family_table.json")
    market_close = _load_json("data/market_close_leak_trade_table.json")
    blocked = as_loop._blocked_sets(acceptance, residual, market_close)
    attributed = as_loop._load_outcomes(ROOT / "data" / "execution_ledger.db", as_loop.ATTRIBUTED_REALIZED_SQL)
    fresh_replay_relpath = Path("logs/reports/forecast_improvement/oanda_history_replay_validate_latest.json")
    fresh_replay = _load_json(str(fresh_replay_relpath))
    firepower = _load_json("data/rolling_30d_4x_firepower_board.json")
    proof_queue = _load_json("data/as_proof_pack_queue.json")
    order_intents = _load_json("data/order_intents.json")
    broker = _load_json("data/broker_snapshot.json")
    daily = _load_json("data/daily_target_state.json")
    candidate_by_lane = {str(row.get("lane_id")): row for row in firepower.get("candidates") or []}
    proof_by_lane = {str(row.get("lane_id")): row for row in proof_queue.get("queue") or []}
    intent_by_lane = {str(row.get("lane_id")): row for row in order_intents.get("results") or []}
    return {
        "as_loop": as_loop,
        "acceptance": acceptance,
        "residual": residual,
        "market_close": market_close,
        "blocked": blocked,
        "attributed_outcomes": attributed,
        "fresh_replay": fresh_replay,
        "fresh_replay_relpath": fresh_replay_relpath,
        "firepower": firepower,
        "proof_queue": proof_queue,
        "order_intents": order_intents,
        "broker": broker,
        "daily": daily,
        "timing": _load_json("data/execution_timing_audit.json"),
        "memory": _load_json("data/memory_health.json"),
        "support": _load_json("data/trader_support_bot.json"),
        "candidate_by_lane": candidate_by_lane,
        "proof_by_lane": proof_by_lane,
        "intent_by_lane": intent_by_lane,
    }


def _historical_target_row(
    lane_id: str,
    direction: str,
    candidate: dict[str, Any],
    proof_queue_row: dict[str, Any],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    pair = str(candidate.get("pair") or _lane_part(lane_id, 1))
    fresh = fresh_direction_evidence(ctx["fresh_replay"], pair, direction)
    blockers = candidate.get("current_blockers") or []
    daily_pct = _float(candidate.get("expected_daily_return_pct_on_funding_adjusted_equity"))
    required_pct = _float(_nested(ctx["firepower"], "target_math", "required_calendar_daily_return_funding_adjusted_pct"))
    forecast_executable = not any(code in blockers for code in ("STALE_QUOTE", "TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE"))
    geometry_ok = not any(
        token in code
        for code in blockers
        for token in ("CHASE", "CONFLICT", "REQUIRES_RANGE_ROTATION", "STRATEGY_NOT_ELIGIBLE")
    )
    margin_ok = bool(candidate.get("risk_allowed") is True and (candidate.get("realistic_units") or 0) > 0)
    classification = "HISTORICAL_ONLY"
    if fresh.get("live_grade_support") and forecast_executable and geometry_ok and margin_ok:
        classification = "PROOF_READY"
    return {
        "lane_id": lane_id,
        "pair": pair,
        "side": candidate.get("side"),
        "method": candidate.get("method"),
        "order_type": candidate.get("order_type"),
        "classification": classification,
        "fresh_replay_status": fresh.get("status"),
        "can_create_live_permission": False,
        "meets_4x_required_daily_return_prefilter": bool(daily_pct is not None and required_pct is not None and daily_pct >= required_pct),
        "expected_daily_return_pct_on_funding_adjusted_equity": daily_pct,
        "required_calendar_daily_return_funding_adjusted_pct": required_pct,
        "expected_jpy_per_trade": candidate.get("expected_jpy_per_trade"),
        "estimated_trades_per_day_available": candidate.get("estimated_trades_per_day_available"),
        "packaged_historical_evidence": candidate.get("source_evidence"),
        "fresh_s5_bidask_replay": fresh,
        "fresh_744h_replay": timing_summary(ctx["timing"]),
        "proof_checks": {
            "fresh_forecast_executable": forecast_executable,
            "geometry_proof": geometry_ok,
            "margin_and_risk_engine_pass": margin_ok,
            "live_order_gateway_pass": candidate.get("status") == "LIVE_READY",
            "gpt_verifier_pass": False,
            "guardian_clear": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED" not in blockers,
            "s5_live_grade": bool(fresh.get("live_grade_support")),
        },
        "margin": {
            "realistic_units": candidate.get("realistic_units"),
            "margin_requirement_realistic_size_jpy": candidate.get("margin_requirement_realistic_size_jpy"),
            "broker_margin_context": candidate.get("broker_margin_context"),
            "risk_allowed": candidate.get("risk_allowed"),
        },
        "current_blockers": blockers,
        "missing_proof": proof_queue_row.get("missing_proof") or {},
        "decision": "rank_repair_work_only_no_live_permission",
    }


def fresh_direction_evidence(replay: dict[str, Any], pair: str, direction: str) -> dict[str, Any]:
    pair_cov = next((row for row in _nested(replay, "forecast_sample_coverage", "pairs") or [] if row.get("pair") == pair), {})
    under = next(
        (
            row
            for row in _nested(replay, "forecast_sample_coverage", "under_sampled_pair_directions") or []
            if row.get("pair") == pair and row.get("direction") == direction
        ),
        {},
    )
    segment = next(
        (
            row
            for row in _nested(replay, "segments", "by_pair_direction") or []
            if row.get("pair") == pair and row.get("direction") == direction
        ),
        {},
    )
    direct_rules = _rules_for_direction(replay, pair, direction)
    contrarian_rules = [
        row
        for row in (_nested(replay, "precision_rules", "contrarian_edge_rules") or [])
        + (_nested(replay, "precision_rules", "rejected_daily_stability_segments") or [])
        if row.get("pair") == pair and row.get("direction") == direction
    ]
    live_grade = any(rule.get("live_grade") is True for rule in direct_rules + contrarian_rules)
    status = "EVIDENCE_GAP"
    if live_grade:
        status = "PROOF_READY"
    elif any(rule.get("adoption_status") == "LIVE_BLOCK_NEGATIVE_EXPECTANCY" for rule in direct_rules):
        status = "REJECTED_NEGATIVE_EXPECTANCY"
    elif under:
        status = "EVIDENCE_GAP_UNDER_SAMPLED"
    return {
        "status": status,
        "pair": pair,
        "direction": direction,
        "price_truth_status": _nested(replay, "price_truth_coverage", "status"),
        "adoption_level": _nested(replay, "price_truth_coverage", "adoption_level"),
        "price_truth_blockers": _nested(replay, "price_truth_coverage", "blockers") or [],
        "pair_coverage": pair_cov,
        "direction_coverage": under,
        "direction_segment": segment,
        "direct_rules": direct_rules[:5],
        "contrarian_or_rank_only_rules": contrarian_rules[:5],
        "live_grade_support": live_grade,
        "can_create_live_permission": False,
    }


def _rules_for_direction(replay: dict[str, Any], pair: str, direction: str) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    precision = replay.get("precision_rules") if isinstance(replay.get("precision_rules"), dict) else {}
    for key in ("edge_rules", "daily_stable_edge_rules", "negative_rules"):
        for row in precision.get(key) or []:
            if row.get("pair") == pair and row.get("direction") == direction:
                buckets.append(row)
    return buckets


def all_firepower_candidates(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    as_loop = ctx["as_loop"]
    rows = []
    required_daily_jpy = None
    funding_adjusted = _float(ctx["daily"].get("funding_adjusted_equity"))
    required_pct = _float(ctx["daily"].get("required_calendar_daily_return_funding_adjusted"))
    if funding_adjusted is not None and required_pct is not None:
        required_daily_jpy = funding_adjusted * required_pct / 100.0
    for result in ctx["order_intents"].get("results") or []:
        row = as_loop._candidate_firepower_row(
            result,
            daily=ctx["daily"],
            broker=ctx["broker"],
            blocked=ctx["blocked"],
            required_daily_jpy=required_daily_jpy,
        )
        if not row.get("hard_excluded"):
            rows.append(row)
    return rows


def portfolio_rank_row(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    proof = ctx["proof_by_lane"].get(row.get("lane_id")) or {}
    classification = proof.get("proof_classification") or proof_classification_from_row(row)
    proof_distance = _int(proof.get("proof_distance"))
    if proof_distance is None:
        proof_distance = _int(row.get("proof_gap_count")) or 99
    daily_pct = _float(row.get("expected_daily_return_pct_on_funding_adjusted_equity")) or 0.0
    margin = _float(row.get("margin_requirement_realistic_size_jpy"))
    evidence_score = {"PROOF_READY": 100.0, "REPAIR_REQUIRED": 55.0, "EVIDENCE_GAP": 40.0, "HISTORICAL_ONLY": 20.0, "REJECTED": 0.0}.get(classification, 10.0)
    margin_score = 12.0 if margin and margin < 5000 else 4.0 if margin else 0.0
    rank_score = _round(daily_pct * 8.0 + evidence_score + margin_score - proof_distance * 4.0 - len(row.get("current_blockers") or []) * 1.5)
    return {
        "lane_id": row.get("lane_id"),
        "pair": row.get("pair"),
        "side": row.get("side"),
        "method": row.get("method"),
        "order_type": row.get("order_type"),
        "proof_classification": classification,
        "rank_score": rank_score,
        "proof_distance": proof_distance,
        "expected_daily_return_pct_on_funding_adjusted_equity": row.get("expected_daily_return_pct_on_funding_adjusted_equity"),
        "expected_jpy_per_trade": row.get("expected_jpy_per_trade"),
        "estimated_trades_per_day_available": row.get("estimated_trades_per_day_available"),
        "realistic_units": row.get("realistic_units"),
        "margin_requirement_realistic_size_jpy": row.get("margin_requirement_realistic_size_jpy"),
        "risk_allowed": row.get("risk_allowed"),
        "concentration_key": f"{row.get('pair')}|{row.get('side')}|{row.get('method')}",
        "current_blocker_count": len(row.get("current_blockers") or []),
        "current_blockers": row.get("current_blockers") or [],
        "can_create_live_permission": False,
        "can_enter_proof_pack": bool(row.get("can_enter_proof_pack")),
    }


def accumulate_unique_basket(rows: list[dict[str, Any]], required_pct: float) -> dict[str, Any]:
    seen: set[str] = set()
    basket: list[dict[str, Any]] = []
    total = 0.0
    for row in sorted(rows, key=lambda item: -(_float(item.get("expected_daily_return_pct_on_funding_adjusted_equity")) or 0.0)):
        key = str(row.get("concentration_key"))
        if key in seen:
            continue
        seen.add(key)
        basket.append(row)
        total += _float(row.get("expected_daily_return_pct_on_funding_adjusted_equity")) or 0.0
        if total >= required_pct:
            break
    return {
        "required_daily_return_pct": required_pct,
        "sum_expected_daily_return_pct": _round(total),
        "mathematically_reaches_required_return": bool(total >= required_pct),
        "currently_live_eligible": False,
        "reason": "Basket is mathematical repair-priority only; at least one required gate remains blocked on every component.",
        "components": basket,
    }


def proof_classification_from_row(row: dict[str, Any]) -> str:
    source = row.get("source_evidence") if isinstance(row.get("source_evidence"), dict) else {}
    blockers = row.get("current_blockers") or []
    if row.get("hard_excluded"):
        return "REJECTED"
    if source.get("historical_only"):
        return "HISTORICAL_ONLY"
    if any("NEGATIVE_EXPECTANCY" in code or "SPREAD_TOO_WIDE" in code or "STALE_QUOTE" in code for code in blockers):
        return "REPAIR_REQUIRED"
    return "EVIDENCE_GAP"


def audjpy_repair_requirements(candidate: dict[str, Any], queue_row: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    requirements = []
    missing = queue_row.get("missing_proof") or {}
    for key, value in missing.items():
        if value is not True:
            requirements.append(key)
    if fresh.get("live_grade_support") is not True:
        requirements.append("fresh_s5_bidask_live_grade_support")
    if candidate.get("expected_daily_return_pct_on_funding_adjusted_equity") and candidate.get("expected_daily_return_pct_on_funding_adjusted_equity") < 5.0:
        requirements.append("portfolio_pairing_or_higher_frequency_required_for_4x")
    return sorted(set(requirements))


def _family_rollups(rows: list[Any], as_loop: Any) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Any]] = defaultdict(list)
    for row in rows:
        groups[(row.pair, row.side, row.method, row.exit_reason)].append(row)
    output = []
    for (pair, side, method, exit_reason), group in groups.items():
        metrics = as_loop._bucket_metrics(group)
        output.append(
            {
                "family_id": f"{pair}|{side}|{method}|{exit_reason}",
                "pair": pair,
                "side": side,
                "method": method,
                "exit_reason": exit_reason,
                **metrics,
                "trade_ids": [row.trade_id for row in sorted(group, key=lambda item: item.ts_utc)],
                "largest_losses": [_outcome_dict(row) for row in sorted(group, key=lambda item: item.realized_pl_jpy)[:5]],
                "largest_wins": [
                    _outcome_dict(row)
                    for row in sorted(
                        (item for item in group if item.realized_pl_jpy > 0),
                        key=lambda item: item.realized_pl_jpy,
                        reverse=True,
                    )[:5]
                ],
                "can_create_live_permission": False,
            }
        )
    return sorted(output, key=lambda item: (item["net_jpy"], item["family_id"]))


def _outcome_dict(row: Any) -> dict[str, Any]:
    return {
        "ts_utc": row.ts_utc,
        "trade_id": row.trade_id,
        "pair": row.pair,
        "side": row.side,
        "lane_id": row.lane_id,
        "method": row.method,
        "exit_reason": row.exit_reason,
        "family_id": f"{row.pair}|{row.side}|{row.method}|{row.exit_reason}",
        "realized_pl_jpy": _round(row.realized_pl_jpy),
    }


def replay_summary(replay: dict[str, Any], relpath: Path) -> dict[str, Any]:
    return {
        "path": str(relpath),
        "generated_at_utc": replay.get("generated_at_utc"),
        "granularity": replay.get("granularity"),
        "pair_filter": replay.get("pair_filter"),
        "price_truth_coverage": replay.get("price_truth_coverage"),
        "forecast_sample_coverage": replay.get("forecast_sample_coverage"),
        "adoption_summary": _nested(replay, "precision_rules", "adoption_summary"),
    }


def timing_summary(timing: dict[str, Any]) -> dict[str, Any]:
    summary = timing.get("summary") if isinstance(timing.get("summary"), dict) else {}
    return {
        "generated_at_utc": timing.get("generated_at_utc"),
        "status": timing.get("status"),
        "lookback_basis": "execution_timing_audit current local artifact",
        "loss_closes_audited": summary.get("loss_closes_audited"),
        "historical_pre_repair_loss_closes_profit_capture_missed": summary.get("historical_pre_repair_loss_closes_profit_capture_missed"),
        "historical_pre_repair_loss_closes_repair_replay_triggered": summary.get("historical_pre_repair_loss_closes_repair_replay_triggered"),
        "loss_close_repair_replay_delta_jpy": summary.get("loss_close_repair_replay_delta_jpy"),
        "post_repair_live_evidence_loss_closes_audited": summary.get("post_repair_live_evidence_loss_closes_audited"),
        "post_repair_live_evidence_loss_closes_profit_capture_missed": summary.get("post_repair_live_evidence_loss_closes_profit_capture_missed"),
        "permission_boundary": "System-level 744h timing replay is diagnostic; it is not a lane-specific live permission receipt.",
    }


def global_blockers(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "profitability_acceptance_status": ctx["acceptance"].get("status"),
        "profitability_acceptance_blockers": ctx["acceptance"].get("blockers") or [],
        "support_status": ctx["support"].get("status"),
        "support_blockers": ctx["support"].get("blockers") or [],
        "memory_status": ctx["memory"].get("status"),
    }


def manual_position_safety(broker: dict[str, Any]) -> dict[str, Any]:
    positions = broker.get("positions") if isinstance(broker.get("positions"), list) else []
    orders = broker.get("orders") if isinstance(broker.get("orders"), list) else []
    manual = next((row for row in positions if str(row.get("trade_id")) == MANUAL_TRADE_ID), {})
    expected_tp = next((row for row in orders if str(row.get("order_id") or row.get("id")) == MANUAL_TP_ORDER_ID), {})
    active_tp = next((row for row in orders if str(row.get("trade_id")) == MANUAL_TRADE_ID), expected_tp)
    raw_tp = _nested(manual, "raw", "takeProfitOrder") if isinstance(manual.get("raw"), dict) else {}
    current_tp_order_id = (
        active_tp.get("order_id")
        or active_tp.get("id")
        or (raw_tp.get("id") if isinstance(raw_tp, dict) else None)
    )
    replaced_expected_tp = (
        isinstance(raw_tp, dict)
        and str(raw_tp.get("replacesOrderID") or "") == MANUAL_TP_ORDER_ID
        and str(current_tp_order_id or "") != MANUAL_TP_ORDER_ID
    )
    return {
        "manual_trade_id": MANUAL_TRADE_ID,
        "expected_manual_tp_order_id": MANUAL_TP_ORDER_ID,
        "current_manual_tp_order_id": current_tp_order_id,
        "expected_tp_order_present": bool(expected_tp),
        "expected_tp_replaced_in_broker_truth": bool(replaced_expected_tp),
        "position_present": bool(manual),
        "tp_order_present": bool(active_tp),
        "position_owner": manual.get("owner"),
        "position_pair": manual.get("pair"),
        "position_side": manual.get("side"),
        "position_units": manual.get("units"),
        "position_unrealized_pl_jpy": manual.get("unrealized_pl_jpy"),
        "take_profit_price": manual.get("take_profit") or active_tp.get("price"),
        "tp_replaces_order_id": raw_tp.get("replacesOrderID") if isinstance(raw_tp, dict) else None,
        "tp_create_time": raw_tp.get("createTime") if isinstance(raw_tp, dict) else active_tp.get("createTime"),
        "management_intent": _nested(manual, "operator_manual_position", "management_intent"),
        "system_pl_counted": _nested(manual, "operator_manual_position", "system_pl_counted"),
        "no_live_side_effects": True,
        "untouched_by_this_run": True,
    }


def row_positive(row: dict[str, Any]) -> bool:
    return (_float(row.get("net_jpy")) or 0.0) > 0.0


def post_gate_gap_md(payload: dict[str, Any]) -> str:
    scope = payload.get("scope") or {}
    metrics = scope.get("metrics") or {}
    safety = payload.get("manual_position_safety") or {}
    lines = [
        "# Post-Gate Expectancy Gap Trace",
        "",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Scope: `{scope.get('name')}`",
        f"- Trades: `{metrics.get('trades')}`; net JPY: `{metrics.get('net_jpy')}`; expectancy/trade: `{metrics.get('expectancy_jpy_per_trade')}`",
        f"- Remaining gap to zero: `{scope.get('remaining_gap_to_zero_jpy')}` JPY",
        f"- Can create live permission: `{scope.get('can_create_live_permission')}`",
        "",
        "## Largest Loss Families",
        "",
        "| family | trades | net JPY | expectancy | max loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in (payload.get("family_trace") or [])[:12]:
        lines.append(f"| `{row.get('family_id')}` | {row.get('trades')} | {row.get('net_jpy')} | {row.get('expectancy_jpy_per_trade')} | {row.get('max_loss_jpy')} |")
    lines.extend(["", "## Largest Loss Trades", "", "| trade | family | P/L JPY |", "|---|---|---:|"])
    for row in (payload.get("largest_loss_trades") or [])[:12]:
        lines.append(f"| `{row.get('trade_id')}` | `{row.get('family_id')}` | {row.get('realized_pl_jpy')} |")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            f"- EUR_USD `{MANUAL_TRADE_ID}` remains manual/read-only and excluded.",
            f"- Expected TP `{safety.get('expected_manual_tp_order_id')}` active: `{safety.get('expected_tp_order_present')}`; current TP: `{safety.get('current_manual_tp_order_id')}` at `{safety.get('take_profit_price')}`.",
            f"- Replaced expected TP in broker truth: `{safety.get('expected_tp_replaced_in_broker_truth')}`; no live side effects from this run: `{safety.get('untouched_by_this_run')}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def historical_replay_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Historical-Only To Fresh Proof Replay",
        "",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Fresh replay: `{_nested(payload, 'fresh_replay_summary', 'path')}`",
        f"- Price truth: `{_nested(payload, 'fresh_replay_summary', 'price_truth_coverage', 'status')}`",
        f"- Adoption level: `{_nested(payload, 'fresh_replay_summary', 'price_truth_coverage', 'adoption_level')}`",
        "",
        "| lane | class | daily % | fresh S5 status | forecast | geometry | margin | permission |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for row in payload.get("rows") or []:
        checks = row.get("proof_checks") or {}
        lines.append(
            f"| `{row.get('lane_id')}` | `{row.get('classification')}` | {row.get('expected_daily_return_pct_on_funding_adjusted_equity')} | `{row.get('fresh_replay_status')}` | `{checks.get('fresh_forecast_executable')}` | `{checks.get('geometry_proof')}` | `{checks.get('margin_and_risk_engine_pass')}` | `{row.get('can_create_live_permission')}` |"
        )
    lines.extend(["", "## 744h Replay Boundary", "", f"- {payload.get('execution_timing_744h_summary')}"])
    return "\n".join(lines) + "\n"


def audjpy_repair_md(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# AUD_JPY SHORT BREAKOUT_FAILURE Repair Proof",
        "",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Status: `{summary.get('status')}`",
        f"- Standalone 4x path exists: `{summary.get('standalone_4x_path_exists')}`",
        f"- Preferred repair shape: `{summary.get('preferred_repair_shape')}`",
        f"- Can create live permission: `{summary.get('can_create_live_permission')}`",
        "",
        "| lane | order | daily % | standalone gap % | fresh S5 | proof pack | permission |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in payload.get("rows") or []:
        lines.append(
            f"| `{row.get('lane_id')}` | `{row.get('order_type')}` | {row.get('expected_daily_return_pct_on_funding_adjusted_equity')} | {row.get('standalone_4x_gap_pct')} | `{_nested(row, 'fresh_s5_bidask_replay', 'status')}` | `{row.get('can_enter_proof_pack')}` | `{row.get('can_create_live_permission')}` |"
        )
    lines.extend(["", "## Missing Repair Requirements", ""])
    for row in payload.get("rows") or []:
        lines.append(f"- `{row.get('lane_id')}`: {', '.join(row.get('repair_requirements') or [])}")
    return "\n".join(lines) + "\n"


def audjpy_limit_proof_pack_md(payload: dict[str, Any]) -> str:
    economics = payload.get("economics") or {}
    geometry = payload.get("geometry") or {}
    margin = payload.get("margin_and_risk") or {}
    lines = [
        "# AUD_JPY SHORT BREAKOUT_FAILURE LIMIT Proof Pack",
        "",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Lane: `{payload.get('lane_id')}`",
        f"- Classification: `{payload.get('classification')}`",
        f"- Standalone 4x: `{payload.get('standalone_4x')}`",
        f"- Portfolio component possible after repair: `{payload.get('portfolio_component_possible_after_repair')}`",
        f"- Can create live permission: `{payload.get('can_create_live_permission')}`",
        "",
        "## Economics",
        "",
        f"- Expected JPY/trade: `{economics.get('expected_jpy_per_trade')}`",
        f"- Estimated trades/day: `{economics.get('estimated_trades_per_day_available')}`",
        f"- Expected active-day contribution: `{economics.get('expected_active_day_contribution_jpy')}` JPY",
        f"- Expected daily return on funding-adjusted equity: `{economics.get('expected_daily_return_pct_on_funding_adjusted_equity')}`%",
        f"- Required calendar daily return: `{economics.get('required_calendar_daily_return_pct')}`%",
        "",
        "## Geometry / Margin",
        "",
        f"- Entry / TP / SL: `{geometry.get('entry')}` / `{geometry.get('tp')}` / `{geometry.get('sl')}`",
        f"- Reward/risk: `{geometry.get('reward_risk')}`; reward/loss pips `{geometry.get('reward_pips')}` / `{geometry.get('loss_pips')}`",
        f"- Units / risk / margin: `{margin.get('realistic_units')}` / `{margin.get('risk_jpy')}` / `{margin.get('estimated_margin_jpy')}`",
        "",
        "## Failed Checks",
        "",
    ]
    for item in payload.get("failed_checks") or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Current Blockers", ""])
    for item in (payload.get("verifier_gateway_guardian") or {}).get("blockers") or []:
        lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


def portfolio_planner_md(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    basket = payload.get("fastest_mathematical_basket") or {}
    lines = [
        "# Portfolio 4x Path Planner",
        "",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Status: `{payload.get('portfolio_status')}`",
        f"- Can reach 4x now: `{payload.get('can_reach_4x_now')}`",
        f"- Non-hard-excluded candidates: `{summary.get('non_hard_excluded_candidates')}`",
        f"- Standalone math candidates: `{summary.get('standalone_math_candidates_meeting_required_return')}`",
        f"- Fastest mathematical basket reaches required return: `{basket.get('mathematically_reaches_required_return')}`; live eligible `{basket.get('currently_live_eligible')}`",
        "",
        "## Top Ranked Repair Work",
        "",
        "| rank | lane | class | daily % | score | distance | units | blockers |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate((payload.get("candidate_rankings") or [])[:15], start=1):
        lines.append(
            f"| {idx} | `{row.get('lane_id')}` | `{row.get('proof_classification')}` | {row.get('expected_daily_return_pct_on_funding_adjusted_equity')} | {row.get('rank_score')} | {row.get('proof_distance')} | {row.get('realistic_units')} | {row.get('current_blocker_count')} |"
        )
    lines.extend(["", "## Global Blockers", "", f"- `{payload.get('global_blockers')}`"])
    return "\n".join(lines) + "\n"


def _load_as_loop() -> Any:
    spec = importlib.util.spec_from_file_location("as_loop", AS_LOOP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {AS_LOOP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["as_loop"] = module
    spec.loader.exec_module(module)
    return module


def _load_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(rel: str, payload: Any) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(rel: str, text: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _nested(payload: Any, *keys: Any) -> Any:
    cur = payload
    for key in keys:
        if isinstance(cur, list) and isinstance(key, int):
            if 0 <= key < len(cur):
                cur = cur[key]
                continue
            return None
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _lane_part(lane_id: str, index: int) -> str:
    parts = str(lane_id or "").split(":")
    return parts[index] if len(parts) > index else ""


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 4) -> float | None:
    number = _float(value)
    if number is None:
        return None
    return round(number, digits)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())

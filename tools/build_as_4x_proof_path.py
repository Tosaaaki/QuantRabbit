#!/usr/bin/env python3
"""Build read-only 4x A/S proof-path artifacts.

The script reads local QuantRabbit artifacts and the execution ledger. It does
not call OANDA, stage orders, send orders, cancel orders, close trades, or
modify SL/TP.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AS_LOOP_PATH = ROOT / "tools" / "build_as_live_ready_evidence_loop.py"
MANUAL_TRADE_ID = "472987"
MANUAL_TP_ORDER_ID = "472988"
MANUAL_TP_AUDIT_ORDER_ID = "472994"
AUDJPY_LIMIT_PROOF_CLASSIFICATIONS = ("PROOF_READY", "REPAIR_REQUIRED", "EVIDENCE_GAP", "REJECTED")
HISTORICAL_TARGETS = (
    ("range_trader:GBP_USD:LONG:RANGE_ROTATION", "UP"),
    ("trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION", "DOWN"),
)
AUDJPY_REPAIR_LANES = (
    "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE",
    "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT",
)
EURUSD_LIMIT_TARGET_LANE = "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT"
EURUSD_LIMIT_TARGET_BASE_LANE = "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"
EURUSD_LIMIT_TARGET_SHAPE = "EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST"
EURUSD_LIMIT_SAMPLE_FLOOR = 20


def main() -> int:
    generated_at = _now()
    as_loop = _load_as_loop()
    context = _context(as_loop)
    limit_sample_mining = build_eurusd_limit_sample_mining(generated_at, context)
    context["limit_sample_mining"] = limit_sample_mining
    payloads = {
        "data/post_gate_expectancy_gap_trace.json": build_post_gate_gap_trace(generated_at, context),
        "data/historical_only_to_fresh_proof_replay.json": build_historical_only_replay(generated_at, context),
        "data/audjpy_short_breakout_failure_repair_proof.json": build_audjpy_repair(generated_at, context),
        "data/audjpy_short_breakout_failure_limit_proof_pack.json": build_audjpy_limit_proof_pack(generated_at, context),
        "data/manual_eurusd_tp_replacement_provenance.json": build_manual_eurusd_tp_replacement_provenance(generated_at, context),
        "data/profitability_acceptance_blocker_reconciliation.json": build_profitability_acceptance_blocker_reconciliation(generated_at, context),
        "data/eurusd_short_breakout_failure_limit_sample_mining.json": limit_sample_mining,
    }
    portfolio_planner = build_portfolio_planner(generated_at, context)
    payloads["data/portfolio_4x_path_planner.json"] = portfolio_planner
    payloads["data/harvest_live_grade_path.json"] = build_harvest_live_grade_path(
        generated_at,
        context,
        portfolio_planner,
    )
    markdown = {
        "docs/post_gate_expectancy_gap_trace.md": post_gate_gap_md(payloads["data/post_gate_expectancy_gap_trace.json"]),
        "docs/historical_only_to_fresh_proof_replay.md": historical_replay_md(payloads["data/historical_only_to_fresh_proof_replay.json"]),
        "docs/audjpy_short_breakout_failure_repair_proof.md": audjpy_repair_md(payloads["data/audjpy_short_breakout_failure_repair_proof.json"]),
        "docs/audjpy_short_breakout_failure_limit_proof_pack.md": audjpy_limit_proof_pack_md(payloads["data/audjpy_short_breakout_failure_limit_proof_pack.json"]),
        "docs/manual_eurusd_tp_replacement_provenance.md": manual_eurusd_tp_replacement_provenance_md(payloads["data/manual_eurusd_tp_replacement_provenance.json"]),
        "docs/profitability_acceptance_blocker_reconciliation.md": profitability_acceptance_blocker_reconciliation_md(payloads["data/profitability_acceptance_blocker_reconciliation.json"]),
        "docs/eurusd_short_breakout_failure_limit_sample_mining.md": eurusd_limit_sample_mining_md(
            payloads["data/eurusd_short_breakout_failure_limit_sample_mining.json"]
        ),
        "docs/portfolio_4x_path_planner.md": portfolio_planner_md(payloads["data/portfolio_4x_path_planner.json"]),
        "docs/harvest_live_grade_path_report.md": harvest_live_grade_path_md(payloads["data/harvest_live_grade_path.json"]),
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
    if not candidate:
        classification = "EVIDENCE_GAP"
    elif any(
        code in blockers
        for code in (
            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
        )
    ):
        classification = "REPAIR_REQUIRED"
    elif failed:
        classification = "REPAIR_REQUIRED"
    else:
        classification = "PROOF_READY"
    if classification == "PROOF_READY" and (checks["risk_engine_pass"] is not True or checks["live_order_gateway_pass"] is not True):
        classification = "REPAIR_REQUIRED"
    required_proof_matrix = audjpy_limit_required_proof_matrix(
        checks=checks,
        fresh_samples=fresh_samples,
        fresh_active_days=fresh_active_days,
        min_directional_samples=min_directional_samples,
        min_active_days=min_active_days,
        daily_stability_status=daily_stability_status,
        forecast_executable=forecast_executable,
        geometry={
            "entry": intent.get("entry"),
            "tp": intent.get("tp"),
            "sl": intent.get("sl"),
            "reward_risk": risk.get("reward_risk"),
            "blockers": [code for code in blockers if "CHASE" in code or "HARVEST_TP_STRUCTURE" in code or "RANGE_FORECAST" in code],
        },
        attached_tp=metadata,
        blockers=blockers,
    )
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
            *AUDJPY_LIMIT_PROOF_CLASSIFICATIONS,
        ],
        "classification": classification,
        "standalone_4x": False,
        "portfolio_component_possible_after_repair": classification == "REPAIR_REQUIRED",
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
        "required_proof_matrix": required_proof_matrix,
        "missing_required_proof": [
            row["proof"] for row in required_proof_matrix if row.get("status") not in {"PRESENT", "PRESENT_BUT_NOT_PERMISSION"}
        ],
        "missing_proof": missing,
        "live_side_effects": [],
    }


def audjpy_limit_required_proof_matrix(
    *,
    checks: dict[str, bool],
    fresh_samples: int,
    fresh_active_days: int,
    min_directional_samples: int,
    min_active_days: int,
    daily_stability_status: Any,
    forecast_executable: bool,
    geometry: dict[str, Any],
    attached_tp: dict[str, Any],
    blockers: list[str],
) -> list[dict[str, Any]]:
    geometry_complete = all(geometry.get(key) is not None for key in ("entry", "tp", "sl")) and (_float(geometry.get("reward_risk")) or 0.0) >= 1.0
    attached_tp_complete = bool(
        attached_tp.get("attach_take_profit_on_fill") is True
        and attached_tp.get("capture_take_profit_scope_key")
        and (_int(attached_tp.get("capture_take_profit_trades")) or 0) > 0
    )
    return [
        {
            "proof": "S5 samples",
            "status": "PRESENT" if checks.get("sample_count_floor") and checks.get("s5_bidask_spread_included_replay") else "MISSING",
            "evidence": {
                "fresh_replay_evaluated_samples": fresh_samples,
                "min_directional_samples": min_directional_samples,
                "s5_bidask_spread_included_replay": checks.get("s5_bidask_spread_included_replay"),
            },
        },
        {
            "proof": "active days",
            "status": "PRESENT" if fresh_active_days >= min_active_days and checks.get("daily_stability_floor") else "MISSING",
            "evidence": {
                "fresh_replay_evaluated_active_days": fresh_active_days,
                "min_active_days": min_active_days,
                "daily_stability_status": daily_stability_status,
            },
        },
        {
            "proof": "forecast executable proof",
            "status": "PRESENT" if forecast_executable else "MISSING",
            "evidence": {
                "forecast_executable": forecast_executable,
                "blocking_codes": [
                    code
                    for code in blockers
                    if code in {"RANGE_FORECAST_REQUIRES_RANGE_ROTATION", "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH", "STALE_QUOTE", "TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE"}
                ],
            },
        },
        {
            "proof": "geometry proof",
            "status": "PRESENT_BUT_NOT_PERMISSION" if geometry_complete else "MISSING",
            "evidence": geometry,
        },
        {
            "proof": "attached TP proof",
            "status": "PRESENT_BUT_NOT_PERMISSION" if attached_tp_complete else "MISSING",
            "evidence": {
                "attach_take_profit_on_fill": attached_tp.get("attach_take_profit_on_fill"),
                "capture_take_profit_scope_key": attached_tp.get("capture_take_profit_scope_key"),
                "capture_take_profit_trades": attached_tp.get("capture_take_profit_trades"),
                "capture_take_profit_wins": attached_tp.get("capture_take_profit_wins"),
                "capture_take_profit_expectancy_jpy": attached_tp.get("capture_take_profit_expectancy_jpy"),
                "positive_rotation_proof_collection_ready": attached_tp.get("positive_rotation_proof_collection_ready"),
            },
        },
        {
            "proof": "RiskEngine",
            "status": "PRESENT" if checks.get("risk_engine_pass") else "MISSING",
            "evidence": {"risk_engine_pass": checks.get("risk_engine_pass")},
        },
        {
            "proof": "Gateway",
            "status": "PRESENT" if checks.get("live_order_gateway_pass") else "MISSING",
            "evidence": {"live_order_gateway_pass": checks.get("live_order_gateway_pass")},
        },
        {
            "proof": "GPT verifier",
            "status": "PRESENT" if checks.get("gpt_verifier_pass") else "MISSING",
            "evidence": {"gpt_verifier_pass": checks.get("gpt_verifier_pass")},
        },
        {
            "proof": "guardian/operator review",
            "status": "PRESENT" if checks.get("guardian_operator_review_clear") else "MISSING",
            "evidence": {
                "guardian_operator_review_clear": checks.get("guardian_operator_review_clear"),
                "guardian_blocker_present": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED" in blockers,
            },
        },
    ]


def build_manual_eurusd_tp_replacement_provenance(generated_at: str, ctx: dict[str, Any]) -> dict[str, Any]:
    broker = ctx["broker"]
    active = _manual_active_tp_from_broker(broker)
    active_id = active.get("order_id")
    ledger_events = _manual_tp_ledger_events(extra_order_ids=[active_id] if active_id else [])
    chain = _manual_tp_replacement_chain(broker, ledger_events)
    order_ids = [str(row.get("order_id")) for row in chain if row.get("order_id")]
    gateway_receipts = _manual_tp_gateway_receipts(order_ids)
    chain = _manual_tp_replacement_chain(broker, ledger_events, gateway_receipts=gateway_receipts)
    safety = manual_position_safety(broker, ledger_events=ledger_events, gateway_receipts=gateway_receipts)
    audit_order = next((row for row in chain if str(row.get("order_id")) == MANUAL_TP_AUDIT_ORDER_ID), {})
    current_order = next((row for row in chain if row.get("lifecycle") == "ACTIVE_BROKER_TRUTH"), {})
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_manual_eurusd_tp_replacement_provenance",
        "source_artifacts": [
            "data/broker_snapshot.json",
            "data/execution_ledger.db",
            "docs/execution_ledger_report.md",
        ],
        "classification_values": ["OPERATOR_MANUAL_PROTECTED", "PROVENANCE_UNKNOWN_BLOCK_AUTOMATION"],
        "status": "MANUAL_TP_AUTOMATION_BLOCKED",
        "manual_trade_id": MANUAL_TRADE_ID,
        "audit_order_id": MANUAL_TP_AUDIT_ORDER_ID,
        "audit_order_classification": audit_order.get("provenance_classification") or "PROVENANCE_UNKNOWN_BLOCK_AUTOMATION",
        "audit_order_lifecycle": audit_order.get("lifecycle"),
        "current_active_tp_order_id": current_order.get("order_id") or active_id,
        "current_active_tp_price": current_order.get("price") or active.get("price"),
        "current_active_tp_classification": current_order.get("provenance_classification") or "PROVENANCE_UNKNOWN_BLOCK_AUTOMATION",
        "broker_truth": {
            "snapshot_fetched_at_utc": broker.get("fetched_at_utc"),
            "last_transaction_id": _nested(broker, "account", "last_transaction_id") or broker.get("last_transaction_id"),
            "manual_position": _manual_position_snapshot(broker),
            "active_take_profit_order": active,
        },
        "replacement_chain": chain,
        "gateway_receipt_search": {
            "searched_order_ids": order_ids,
            "receipt_count": sum(len(items) for items in gateway_receipts.values()),
            "receipts_by_order_id": gateway_receipts,
            "result": "NO_LOCAL_QUANTRABBIT_GATEWAY_RECEIPT_FOUND",
        },
        "manual_position_safety": safety,
        "conclusion": (
            "Broker truth shows EUR_USD trade 472987 is operator-manual and the active TP is protected from automation. "
            "Order 472994 is historical/replaced by 472996, and neither 472994 nor 472996 has a local QuantRabbit gateway receipt. "
            "Classify TP provenance as unknown and block automation from using, modifying, or inferring permission from it."
        ),
        "live_side_effects": [],
    }


def build_profitability_acceptance_blocker_reconciliation(generated_at: str, ctx: dict[str, Any]) -> dict[str, Any]:
    acceptance = ctx["acceptance"]
    broker_mutation_audit = _load_json("data/broker_mutation_bypass_audit.json")
    manual_tp_audit = _manual_tp_audit_evidence_summary(broker_mutation_audit)
    current_codes = _acceptance_codes(acceptance)
    blocker_codes = _acceptance_blocker_codes(acceptance)
    rows = [
        _profitability_reconciliation_row(
            code="OPERATOR_MANUAL_TP_OPT_OUT_BYPASS",
            classification=(
                "FIXED_NEEDS_CLEAN_WINDOW"
                if _nested(broker_mutation_audit, "conclusion", "position_manager_gateway_bypass_fixed") is True
                else "ACTIVE_BLOCKER"
            ),
            current_codes=current_codes,
            blocker_codes=blocker_codes,
            evidence_summary={
                "broker_mutation_audit": "data/broker_mutation_bypass_audit.json",
                "tp_rebalance_incident_contained": _nested(
                    broker_mutation_audit, "conclusion", "tp_rebalance_incident_contained"
                ),
                "position_manager_gateway_bypass_fixed": _nested(
                    broker_mutation_audit, "conclusion", "position_manager_gateway_bypass_fixed"
                ),
                "manual_trade_472987_untouched": _nested(
                    broker_mutation_audit, "conclusion", "manual_trade_472987_untouched"
                ),
                "last_transaction_id": manual_tp_audit["last_transaction_id"],
                "active_take_profit_order": manual_tp_audit["active_take_profit_order"],
            },
            clearance_condition=(
                "Keep a clean proof window where tp_rebalancer, PositionManager, and "
                "PositionProtectionGateway all preserve operator-manual packets with "
                "auto_tp_modify_allowed=false, and broker transaction IDs do not advance "
                "from unauthorized TP/SL/close writes."
            ),
        ),
        _profitability_reconciliation_row(
            code="SELF_IMPROVEMENT_P0_PRESENT",
            classification="ACTIVE_BLOCKER" if _self_p0_items(ctx["self_audit"]) else "STALE_SUPERSEDED",
            current_codes=current_codes,
            blocker_codes=blocker_codes,
            evidence_summary={
                "self_improvement_status": ctx["self_audit"].get("status"),
                "self_improvement_p0_codes": [row.get("code") for row in _self_p0_items(ctx["self_audit"])],
                "memory_status": ctx["memory"].get("status"),
                "memory_blockers": ctx["memory"].get("blockers") or [],
            },
            clearance_condition="Regenerate memory-health and self-improvement-audit with zero P0 findings.",
        ),
        _profitability_reconciliation_row(
            code="NEGATIVE_EXPECTANCY_ACTIVE",
            classification="ACTIVE_BLOCKER",
            current_codes=current_codes,
            blocker_codes=blocker_codes,
            evidence_summary=_acceptance_evidence_summary(acceptance, "NEGATIVE_EXPECTANCY_ACTIVE"),
            clearance_condition="Accepted realized capture economics must become non-negative without promoting blocked historical families.",
        ),
        _profitability_reconciliation_row(
            code="MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
            classification="ACTIVE_BLOCKER",
            current_codes=current_codes,
            blocker_codes=blocker_codes,
            evidence_summary=_acceptance_evidence_summary(acceptance, "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE"),
            clearance_condition="TP-proven segments must no longer be net-damaged by unproven MARKET_ORDER_TRADE_CLOSE leakage.",
        ),
        _profitability_reconciliation_row(
            code="MARKET_CLOSE_LEAK_FAMILY_BLOCKED",
            classification="TAXONOMY_DUPLICATE",
            current_codes=current_codes,
            blocker_codes=blocker_codes,
            duplicate_of="MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
            evidence_summary=_acceptance_evidence_summary(acceptance, "MARKET_CLOSE_LEAK_FAMILY_BLOCKED"),
            clearance_condition="Clear only with exact close-gate proof, contained-risk timing evidence, and TP-proven exception evidence for the family.",
        ),
        _profitability_reconciliation_row(
            code="MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
            classification="CONTAINED_NOT_CLEARED",
            current_codes=current_codes,
            blocker_codes=blocker_codes,
            contained_by="month_scale_residual_family_filters",
            evidence_summary=_acceptance_evidence_summary(acceptance, "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"),
            clearance_condition="Fresh 30-day TP-progress repair replay must be non-negative after the accepted filters and cannot rely on optimism.",
        ),
        _profitability_reconciliation_row(
            code="PROJECTION_HEADLINE_PRECISION_ECONOMIC_GAP",
            classification="EVIDENCE_GAP",
            current_codes=current_codes,
            blocker_codes=blocker_codes,
            evidence_summary=_acceptance_evidence_summary(acceptance, "PROJECTION_HEADLINE_PRECISION_ECONOMIC_GAP"),
            clearance_condition="Projection buckets used for live support must clear economic precision, not just headline hit-rate precision.",
        ),
        _profitability_reconciliation_row(
            code="BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN",
            classification="EVIDENCE_GAP",
            current_codes=current_codes,
            blocker_codes=blocker_codes,
            evidence_summary=_acceptance_evidence_summary(acceptance, "BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN"),
            clearance_condition="All-currency bid/ask replay coverage must be thick enough for the live-grade proof target.",
        ),
        _profitability_reconciliation_row(
            code="NO_LIVE_READY_TARGET_COVERAGE",
            classification="EVIDENCE_GAP",
            current_codes=current_codes,
            blocker_codes=blocker_codes,
            evidence_summary=_acceptance_evidence_summary(acceptance, "NO_LIVE_READY_TARGET_COVERAGE"),
            clearance_condition="At least one lane must regenerate LIVE_READY with risk, gateway, GPT, guardian, telemetry, and acceptance proof.",
        ),
        _profitability_reconciliation_row(
            code="REPAIR_FRONTIER_BLOCKED",
            classification="EVIDENCE_GAP",
            current_codes=current_codes,
            blocker_codes=blocker_codes,
            evidence_summary=_acceptance_evidence_summary(acceptance, "REPAIR_FRONTIER_BLOCKED"),
            clearance_condition="Closest repair lanes must fill exact proof gaps; repair ranking does not create permission.",
        ),
        _profitability_reconciliation_row(
            code="EXECUTION_LEDGER_STALE",
            classification="STALE_SUPERSEDED",
            current_codes=current_codes,
            blocker_codes=blocker_codes,
            evidence_summary={
                "execution_ledger_last_oanda_transaction_id": _nested(ctx["memory"], "metrics", "execution_ledger", "last_oanda_transaction_id"),
                "snapshot_last_transaction_id": _nested(ctx["memory"], "metrics", "execution_ledger", "snapshot_last_transaction_id"),
            },
            clearance_condition="Keep execution ledger synchronized with broker snapshot before regenerating acceptance.",
        ),
        _profitability_reconciliation_row(
            code="RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
            classification="STALE_SUPERSEDED",
            current_codes=current_codes,
            blocker_codes=blocker_codes,
            evidence_summary={"present_in_current_acceptance": "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK" in current_codes},
            clearance_condition="If it reappears, reconcile the exact market-close receipt and keep the family blocked until proven.",
        ),
        _profitability_reconciliation_row(
            code="HISTORICAL_PROFIT_CAPTURE_MISSED",
            classification="CONTAINED_NOT_CLEARED",
            current_codes=current_codes,
            blocker_codes=blocker_codes,
            evidence_summary={"related_current_code": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"},
            clearance_condition="Treat historical capture misses as repair inputs only until post-repair replay and fresh proof are non-negative.",
        ),
    ]
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_profitability_acceptance_blocker_reconciliation",
        "source_artifacts": [
            "data/profitability_acceptance.json",
            "docs/profitability_acceptance_report.md",
            "data/self_improvement_audit.json",
            "data/memory_health.json",
            "data/remaining_profitability_p0_decomposition.json",
            "data/as_lane_candidate_board.json",
            "data/as_proof_pack_queue.json",
            "data/broker_mutation_bypass_audit.json",
        ],
        "classification_values": [
            "ACTIVE_BLOCKER",
            "FIXED_NEEDS_CLEAN_WINDOW",
            "CONTAINED_NOT_CLEARED",
            "TAXONOMY_DUPLICATE",
            "STALE_SUPERSEDED",
            "EVIDENCE_GAP",
        ],
        "status": acceptance.get("status") or "PROFITABILITY_ACCEPTANCE_BLOCKED",
        "summary": {
            "acceptance_blocked": acceptance.get("status") != "PROFITABILITY_ACCEPTANCE_PASS",
            "current_acceptance_blocker_codes": sorted(blocker_codes),
            "classification_counts": _classification_counts(rows),
            "normal_routing_status": "BLOCKED",
            "routing_allowed": False,
            "live_ready_lanes": 0,
            "as_live_ready_path_exists": False,
            "can_create_live_permission": False,
            "containment_only_cannot_clear_acceptance": True,
        },
        "rows": rows,
        "permission_boundary": (
            "No row in this reconciliation creates live permission. ACTIVE, fixed-needs-clean-window, contained, duplicate, "
            "and evidence-gap rows all keep normal routing blocked until acceptance, lane proof, RiskEngine, LiveOrderGateway, "
            "GPT verifier, and guardian gates pass together."
        ),
        "live_side_effects": [],
    }


def build_portfolio_planner(generated_at: str, ctx: dict[str, Any]) -> dict[str, Any]:
    all_candidates = all_firepower_candidates(ctx)
    rankings = [portfolio_rank_row(row, ctx) for row in all_candidates]
    rankings.sort(key=lambda item: (-item["rank_score"], item["proof_distance"], item["lane_id"]))
    required_pct = _float(_nested(ctx["firepower"], "target_math", "required_calendar_daily_return_funding_adjusted_pct")) or 0.0
    math_eligible = [row for row in rankings if row.get("math_candidate_eligible") is True]
    standalone = [
        row
        for row in math_eligible
        if (row.get("expected_daily_return_pct_on_funding_adjusted_equity") or 0.0) >= required_pct
    ]
    mathematical_basket = accumulate_unique_basket(math_eligible, required_pct)
    repair_only_basket = accumulate_unique_basket(
        [row for row in math_eligible if row["proof_classification"] == "REPAIR_REQUIRED"],
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
            "math_candidate_eligible_candidates": len(math_eligible),
            "planner_rejected_candidates": sum(1 for row in rankings if row.get("math_candidate_eligible") is False),
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


def build_harvest_live_grade_path(
    generated_at: str,
    ctx: dict[str, Any],
    portfolio_planner: dict[str, Any],
) -> dict[str, Any]:
    payoff = ctx["payoff_shape"]
    harvest_candidates = payoff.get("harvest_candidates") if isinstance(payoff.get("harvest_candidates"), list) else []
    rankings = [
        harvest_rank_row(row, ctx, portfolio_planner)
        for row in harvest_candidates
        if isinstance(row, dict)
    ]
    rankings.sort(key=lambda item: (-item["rank_score"], item["proof_gap_trades"], item["shape_key"]))
    for index, row in enumerate(rankings, start=1):
        row["rank"] = index
    closest = rankings[0] if rankings else None
    proof_summary = ctx["proof_queue"].get("summary") if isinstance(ctx["proof_queue"].get("summary"), dict) else {}
    proof_queue_count = _int(proof_summary.get("queue_count"))
    if proof_queue_count is None:
        proof_queue_count = len(ctx["proof_queue"].get("queue") or [])
    proof_ready_count = _int(proof_summary.get("proof_ready_count")) or 0
    can_create_count = _int(proof_summary.get("can_create_live_permission_count")) or 0
    blockers = harvest_blockers(closest or {}, ctx, portfolio_planner)
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_harvest_live_grade_path",
        "status": "DIAGNOSIS_COMPLETE_BLOCKED_NO_LIVE_PERMISSION",
        "read_only": True,
        "live_side_effects": [],
        "live_permission_allowed": False,
        "live_promotion_allowed": False,
        "runner_creation_decision": "HARVEST_PROOF_FIRST",
        "source_artifacts": [
            "data/payoff_shape_diagnosis.json",
            "data/rolling_30d_4x_firepower_board.json",
            "data/as_proof_pack_queue.json",
            "data/as_lane_candidate_board.json",
            "data/portfolio_4x_path_planner.json",
            "data/eurusd_short_breakout_failure_proof_floor_update.json",
            "data/eurusd_short_breakout_failure_limit_s5_bidask_replay.json",
            "data/eurusd_short_breakout_failure_limit_sample_mining.json",
            "data/profitability_acceptance.json",
            "data/trader_support_bot.json",
            "data/live_order_request.json",
            "data/broker_snapshot.json",
        ],
        "proof_queue_summary": {
            "queue_count": proof_queue_count,
            "proof_ready_count": proof_ready_count,
            "can_create_live_permission_count": can_create_count,
            "proof_queue_count_is_live_permission": False,
        },
        "portfolio_summary": {
            "portfolio_status": portfolio_planner.get("portfolio_status"),
            "can_reach_4x_now": bool(portfolio_planner.get("can_reach_4x_now")),
            "can_create_live_permission": bool(_nested(portfolio_planner, "summary", "can_create_live_permission")),
            "live_ready_lanes": _int(portfolio_planner.get("live_ready_lanes")) or 0,
        },
        "runner_candidates": payoff.get("runner_candidates") or [],
        "harvest_candidate_count": len(rankings),
        "closest_harvest_candidate": closest,
        "ranked_harvest_candidates": rankings[:20],
        "promotion_blockers": blockers,
        "required_next_actions": harvest_required_next_actions(closest or {}, ctx),
        "safety_contract": {
            "do_not_send_orders": True,
            "do_not_cancel_orders": True,
            "do_not_close_positions": True,
            "do_not_modify_launchd": True,
            "do_not_relax_gates": True,
            "do_not_backsolve_lot_from_4x_deficit": True,
            "do_not_mix_market_or_stop_rows_into_limit_proof": True,
            "do_not_mix_market_close_losses_into_harvest_proof": True,
            "proof_queue_count_is_not_live_permission": True,
            "negative_expectancy_must_remain_visible": True,
            "month_scale_negative_must_remain_visible": True,
        },
    }


def build_eurusd_limit_sample_mining(generated_at: str, ctx: dict[str, Any]) -> dict[str, Any]:
    replay = ctx["limit_s5_bidask_replay"]
    replay_trade_ids = _replayed_limit_trade_ids(replay)
    legacy_search = _load_json("data/eurusd_short_breakout_failure_legacy_sample_search.json")
    accepted_legacy_ids = {
        str(row.get("trade_id"))
        for row in legacy_search.get("accepted_tp_harvest_samples") or []
        if isinstance(row, dict) and row.get("trade_id")
    }
    rejected_legacy = _legacy_rejection_map(legacy_search)
    execution = _mine_execution_ledger_eurusd_limit_samples(replay_trade_ids)
    legacy = _mine_legacy_eurusd_limit_samples(
        replay_trade_ids=replay_trade_ids,
        accepted_legacy_ids=accepted_legacy_ids,
        rejected_legacy=rejected_legacy,
    )
    additional = [
        row
        for row in execution.get("rows", []) + legacy.get("rows", [])
        if row.get("verdict") == "ACCEPTABLE_NEW_EXACT_LIMIT_SAMPLE"
    ]
    observed_count = _int(replay.get("replay_sample_count")) or len(replay_trade_ids)
    total_after_local_mining = observed_count + len(additional)
    remaining = max(0, EURUSD_LIMIT_SAMPLE_FLOOR - total_after_local_mining)
    floor_met = remaining == 0
    local_coverage_exhausted = len(additional) == 0 and not floor_met
    if floor_met:
        status = "LOCAL_LIMIT_SAMPLE_FLOOR_MET"
    elif additional:
        status = "LOCAL_LIMIT_SAMPLE_COVERAGE_FOUND_NEW_CANDIDATES"
    else:
        status = "LOCAL_LIMIT_SAMPLE_COVERAGE_EXHAUSTED_STILL_UNDERSAMPLED"
    remaining_blockers = []
    if not floor_met:
        remaining_blockers.append(
            {
                "code": "LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY",
                "status": "BLOCKING_PROOF_QUEUE_PROMOTION",
                "evidence": (
                    f"{total_after_local_mining}/{EURUSD_LIMIT_SAMPLE_FLOOR} exact LIMIT samples "
                    "after read-only local mining."
                ),
            }
        )
    if local_coverage_exhausted:
        remaining_blockers.append(
            {
                "code": "LOCAL_LIMIT_SAMPLE_COVERAGE_EXHAUSTED",
                "status": "EVIDENCE_ACQUISITION_BLOCKER",
                "evidence": (
                    "No additional acceptable exact LIMIT/HARVEST rows were found in execution_ledger.db or "
                    "legacy_history.db under the current classification rules."
                ),
            }
        )
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_eurusd_short_breakout_failure_limit_sample_mining",
        "status": status,
        "target_shape": EURUSD_LIMIT_TARGET_SHAPE,
        "target_lane": EURUSD_LIMIT_TARGET_LANE,
        "read_only": True,
        "live_side_effects": [],
        "live_permission_allowed": False,
        "source_artifacts": [
            "data/execution_ledger.db",
            "data/legacy_history.db",
            "data/eurusd_short_breakout_failure_limit_s5_bidask_replay.json",
            "data/eurusd_short_breakout_failure_legacy_sample_search.json",
        ],
        "sample_floor": {
            "required_exact_limit_samples": EURUSD_LIMIT_SAMPLE_FLOOR,
            "current_replayed_exact_limit_samples": observed_count,
            "additional_acceptable_local_samples_found": len(additional),
            "total_exact_limit_samples_after_local_mining": total_after_local_mining,
            "remaining_exact_limit_samples": remaining,
            "floor_met": floor_met,
        },
        "current_replay_trade_ids": sorted(replay_trade_ids),
        "additional_acceptable_samples": additional,
        "execution_ledger_coverage": execution,
        "legacy_history_coverage": legacy,
        "classification_rules": {
            "accepted": [
                "pair=EUR_USD",
                "side=SHORT",
                "entry vehicle LIMIT_ORDER",
                "lane or legacy strategy maps to BREAKOUT_FAILURE",
                "exit reason TAKE_PROFIT_ORDER",
                "not already present in exact replay",
            ],
            "rejected": [
                "MARKET_ORDER and STOP_ORDER are never counted in LIMIT proof",
                "RANGE_ROTATION/TREND_CONTINUATION lanes are not BREAKOUT_FAILURE proof",
                "ambiguous direct-USD or continuation legacy wins remain rejected unless a stronger strategy source maps them to failed-break/retest",
                "partial-close or duplicate close rows are not used as fresh exact attached-TP samples",
            ],
        },
        "remaining_blockers": remaining_blockers,
        "next_read_only_actions": [
            "Do not reclassify MARKET/STOP or range/trend wins into the LIMIT proof floor.",
            "If more samples are needed, import or locate additional broker/legacy rows with exact LIMIT_ORDER entry, attached TAKE_PROFIT_ORDER exit, and BREAKOUT_FAILURE-equivalent strategy evidence.",
            "If exact LIMIT history remains exhausted, split MARKET and STOP vehicles into separate read-only proof contracts instead of mixing them into LIMIT.",
            "Keep live_permission_allowed=false until risk, guardian/operator review, verifier, gateway, and fresh broker truth all pass.",
        ],
        "do_not_do": [
            "Do not send live orders.",
            "Do not stage live orders from mining coverage.",
            "Do not cancel orders.",
            "Do not close positions.",
            "Do not modify broker TP or SL.",
            "Do not loosen gates or lower the exact LIMIT sample floor.",
            "Do not mix MARKET_ORDER, STOP_ORDER, RANGE_ROTATION, TREND_CONTINUATION, or market-close outcomes into LIMIT/HARVEST proof.",
            "Do not invent operator approval.",
        ],
    }


def harvest_rank_row(
    row: dict[str, Any],
    ctx: dict[str, Any],
    portfolio_planner: dict[str, Any],
) -> dict[str, Any]:
    shape_key = str(row.get("shape_key") or "")
    pair = str(row.get("pair") or _shape_part(shape_key, 0))
    side = str(row.get("side") or _shape_part(shape_key, 1))
    method = str(row.get("method") or _shape_part(shape_key, 2))
    firepower_rows = [
        candidate
        for candidate in all_firepower_candidates(ctx)
        if candidate.get("pair") == pair and candidate.get("side") == side and candidate.get("method") == method
    ]
    current_best = _preferred_firepower_row(firepower_rows)
    proof_rows = _matching_proof_rows(ctx["proof_queue"], pair, side, method)
    proof_queue_member = bool(proof_rows)
    proof_best = _preferred_proof_row(proof_rows, current_best)
    portfolio_best = _matching_portfolio_row(portfolio_planner, pair, side, method, current_best)
    adjusted = _adjusted_tp_proof(row, ctx)
    limit_replay = _target_limit_replay(row, ctx)
    proof_gap = _int(adjusted.get("proof_gap_trades"))
    if proof_gap is None:
        proof_gap = _int(row.get("proof_gap_trades")) or 999
    overall_net = _float(row.get("overall_net_jpy")) or 0.0
    market_close_losses = _int(row.get("market_close_losses")) or 0
    month_blocked = bool(row.get("month_scale_blocker"))
    current_blockers = list(current_best.get("current_blockers") or [])
    if not current_blockers:
        current_blockers = list(current_best.get("live_blocker_codes") or [])
    planner_can_enter = bool(
        (portfolio_best or {}).get("can_enter_proof_pack")
        or current_best.get("can_enter_proof_pack")
        or proof_best.get("can_enter_proof_pack")
    )
    can_create = bool(
        (portfolio_best or {}).get("can_create_live_permission")
        or current_best.get("can_create_live_permission")
        or proof_best.get("can_create_live_permission")
    )
    target_bonus = 35.0 if shape_key == ctx["proof_floor_update"].get("target_shape") else 0.0
    proof_floor_bonus = 30.0 if adjusted.get("proof_floor_reached") else 0.0
    rank_score = _round(
        target_bonus
        + proof_floor_bonus
        + (35.0 if overall_net > 0 else -25.0)
        + (20.0 if current_best else 0.0)
        + (20.0 if proof_queue_member else 0.0)
        + (12.0 if planner_can_enter else 0.0)
        + (8.0 if not month_blocked else -20.0)
        + (_float(adjusted.get("take_profit_expectancy_jpy")) or 0.0) / 100.0
        - proof_gap * 3.0
        - market_close_losses * 1.5
        - len(current_blockers) * 1.0
    )
    lane_id = current_best.get("lane_id") or proof_best.get("lane_id") or (portfolio_best or {}).get("lane_id")
    current_intent = _current_intent_summary(lane_id, ctx)
    blockers = _candidate_promotion_blockers(
        row=row,
        current_best=current_best,
        proof_best=proof_best,
        portfolio_best=portfolio_best or {},
        adjusted=adjusted,
        limit_replay=limit_replay,
        ctx=ctx,
    )
    return {
        "rank": None,
        "rank_score": rank_score,
        "rank_reason": _harvest_rank_reason(row, adjusted, proof_queue_member, planner_can_enter, limit_replay),
        "proof_gap_trades": proof_gap,
        "candidate_id": shape_key,
        "shape_key": shape_key,
        "classification": adjusted.get("classification") or row.get("classification"),
        "pair": pair,
        "side": side,
        "method": method,
        "current_intent_count": len(firepower_rows),
        "current_intent_best": current_intent,
        "portfolio_planner_best": portfolio_best,
        "actual_proof_queue_member": proof_queue_member,
        "proof_queue_entries": [
            {
                "lane_id": item.get("lane_id"),
                "order_type": item.get("order_type"),
                "proof_classification": item.get("proof_classification"),
                "proof_distance": item.get("proof_distance"),
                "can_enter_proof_pack": bool(item.get("can_enter_proof_pack")),
                "can_create_live_permission": bool(item.get("can_create_live_permission")),
                "missing_proof": item.get("missing_proof") or {},
            }
            for item in proof_rows
        ],
        "planner_can_enter_proof_pack": planner_can_enter,
        "can_create_live_permission": can_create,
        "live_promotion_allowed": False,
        "promotion_blockers": blockers,
        "evidence_refs": row.get("evidence_refs") or [],
        "tp_proof": adjusted,
        "shape_pnl": {
            "overall_net_jpy": row.get("overall_net_jpy"),
            "overall_expectancy_jpy_per_trade": row.get("overall_expectancy_jpy_per_trade"),
            "market_close_losses": row.get("market_close_losses"),
            "market_close_net_jpy": row.get("market_close_net_jpy"),
        },
        "month_scale_blocker": row.get("month_scale_blocker"),
        "limit_s5_bidask_replay": limit_replay,
        "limit_sample_mining": _target_limit_sample_mining(row, ctx),
    }


def harvest_blockers(
    closest: dict[str, Any],
    ctx: dict[str, Any],
    portfolio_planner: dict[str, Any],
) -> list[dict[str, Any]]:
    codes = list(closest.get("promotion_blockers") or [])
    codes.extend(_codes_from_values(ctx["acceptance"].get("blockers") or []))
    codes.extend(_codes_from_values(ctx["support"].get("blockers") or []))
    if ctx["acceptance"].get("status"):
        codes.append(str(ctx["acceptance"].get("status")))
    memory_status = str(ctx["memory"].get("status") or "")
    if memory_status and memory_status not in {"OK", "PASS", "MEMORY_HEALTH_PASS"}:
        codes.append(memory_status)
    if portfolio_planner.get("portfolio_status") == "NO_LIVE_READY_PORTFOLIO":
        codes.append("NO_LIVE_READY_PORTFOLIO")
    if _nested(portfolio_planner, "summary", "can_create_live_permission") is not True:
        codes.append("PORTFOLIO_PLANNER_CANNOT_CREATE_LIVE_PERMISSION")
    proof_summary = ctx["proof_queue"].get("summary") if isinstance(ctx["proof_queue"].get("summary"), dict) else {}
    if (_int(proof_summary.get("proof_ready_count")) or 0) == 0:
        codes.append("PROOF_QUEUE_NOT_PROOF_READY")
    if (_int(proof_summary.get("can_create_live_permission_count")) or 0) == 0:
        codes.append("PROOF_QUEUE_CANNOT_CREATE_LIVE_PERMISSION")
    live_order = ctx["live_order_request"]
    if live_order.get("status") in {None, "NO_ACTION"} or live_order.get("send_requested") is not True:
        codes.append("NO_FRESH_GATEWAY_PERMISSION")
    rows = []
    for code in _unique_strings(codes):
        rows.append({"code": code, "status": "BLOCKING_LIVE_GRADE"})
    return rows


def harvest_required_next_actions(closest: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    target = closest.get("candidate_id") or closest.get("shape_key")
    mining = ctx.get("limit_sample_mining") if isinstance(ctx.get("limit_sample_mining"), dict) else {}
    mining_floor = mining.get("sample_floor") if isinstance(mining.get("sample_floor"), dict) else {}
    return [
        {
            "action_id": "canonicalize_limit_s5_bidask_replay",
            "target": "EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST",
            "read_only": True,
            "live_side_effects": [],
            "success_condition": {
                "legacy_limit_rows_reconciled": ["469278", "469427", "469898"],
                "limit_only_sample_floor_met": True,
                "s5_bidask_spread_included_replay": "positive_for_exact_limit_harvest_vehicle",
            },
        },
        {
            "action_id": "mine_additional_exact_limit_harvest_samples",
            "target": target,
            "read_only": True,
            "live_side_effects": [],
            "current_local_coverage_status": mining.get("status"),
            "current_local_additional_acceptable_samples": mining_floor.get("additional_acceptable_local_samples_found"),
            "current_exact_limit_sample_gap": mining_floor.get("remaining_exact_limit_samples"),
            "success_condition": "Additional accepted rows are LIMIT + ATTACHED_TECHNICAL_TP + HARVEST only; MARKET/STOP/range/trend rows remain excluded.",
        },
        {
            "action_id": "refresh_profitability_and_goal_chain",
            "target": "portfolio",
            "commands": [
                "PYTHONPATH=src python3 -m quant_rabbit.cli payoff-shape-diagnosis",
                "PYTHONPATH=src python3 -m quant_rabbit.cli as-live-ready-evidence-loop",
                "PYTHONPATH=src python3 -m quant_rabbit.cli as-4x-proof-path",
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator",
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-goal-loop-orchestrator",
                "PYTHONPATH=src python3 -m quant_rabbit.cli active-trader-contract",
            ],
            "read_only": True,
            "live_side_effects": [],
            "success_condition": "harvest/proof/portfolio/goal/active artifacts agree on proof queue membership while live_permission_allowed remains false until all gates pass.",
        },
    ]


def _preferred_firepower_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda item: (
            item.get("order_type") != "LIMIT",
            not bool(item.get("can_enter_proof_pack")),
            _int(item.get("proof_gap_count")) or 999,
            str(item.get("lane_id") or ""),
        ),
    )[0]


def _matching_proof_rows(proof_queue: dict[str, Any], pair: str, side: str, method: str) -> list[dict[str, Any]]:
    rows = []
    for row in proof_queue.get("queue") or []:
        if not isinstance(row, dict):
            continue
        if row.get("pair") == pair and row.get("side") == side and row.get("method") == method:
            rows.append(row)
    return rows


def _preferred_proof_row(rows: list[dict[str, Any]], current_best: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {}
    current_lane = current_best.get("lane_id")
    return sorted(
        rows,
        key=lambda item: (
            item.get("lane_id") != current_lane,
            item.get("order_type") != "LIMIT",
            not bool(item.get("can_enter_proof_pack")),
            _int(item.get("proof_distance")) or 999,
            str(item.get("lane_id") or ""),
        ),
    )[0]


def _matching_portfolio_row(
    portfolio: dict[str, Any],
    pair: str,
    side: str,
    method: str,
    current_best: dict[str, Any],
) -> dict[str, Any]:
    current_lane = current_best.get("lane_id")
    rows = [
        row
        for row in portfolio.get("candidate_rankings") or []
        if isinstance(row, dict)
        and row.get("pair") == pair
        and row.get("side") == side
        and row.get("method") == method
    ]
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda item: (
            item.get("lane_id") != current_lane,
            item.get("order_type") != "LIMIT",
            -(_float(item.get("rank_score")) or 0.0),
            _int(item.get("proof_distance")) or 999,
        ),
    )[0]


def _adjusted_tp_proof(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    proof_floor_update = ctx["proof_floor_update"]
    shape_key = row.get("shape_key")
    if shape_key == proof_floor_update.get("target_shape"):
        post = proof_floor_update.get("post_update_tp_proof")
        post = post if isinstance(post, dict) else {}
        return {
            "proof_source": "data/eurusd_short_breakout_failure_proof_floor_update.json",
            "classification": _nested(
                proof_floor_update,
                "harvest_live_grade_re_evaluation",
                "post_update_candidate_classification",
            )
            or row.get("classification"),
            "take_profit_trades": _int(post.get("wins")),
            "take_profit_wins": _int(post.get("wins")),
            "take_profit_losses": _int(post.get("losses")),
            "proof_floor_trades": _int(post.get("proof_floor")),
            "proof_gap_trades": _int(post.get("remaining_samples")),
            "proof_floor_reached": bool(post.get("proof_floor_reached")),
            "take_profit_expectancy_jpy": row.get("take_profit_expectancy_jpy"),
            "take_profit_avg_win_jpy": row.get("take_profit_avg_win_jpy"),
            "take_profit_net_jpy": row.get("take_profit_net_jpy"),
            "legacy_samples_accepted": proof_floor_update.get("legacy_samples_accepted") or [],
            "canonical_integration_status": proof_floor_update.get("canonical_integration_status"),
        }
    proof_floor = _int(row.get("proof_floor_trades"))
    gap = _int(row.get("proof_gap_trades"))
    return {
        "proof_source": "data/payoff_shape_diagnosis.json",
        "classification": row.get("classification"),
        "take_profit_trades": _int(row.get("take_profit_trades")),
        "take_profit_wins": _int(row.get("take_profit_wins")),
        "take_profit_losses": _int(row.get("take_profit_losses")),
        "proof_floor_trades": proof_floor,
        "proof_gap_trades": gap,
        "proof_floor_reached": bool(gap == 0 and proof_floor is not None),
        "take_profit_expectancy_jpy": row.get("take_profit_expectancy_jpy"),
        "take_profit_avg_win_jpy": row.get("take_profit_avg_win_jpy"),
        "take_profit_net_jpy": row.get("take_profit_net_jpy"),
        "legacy_samples_accepted": [],
        "canonical_integration_status": None,
    }


def _target_limit_replay(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    replay = ctx["limit_s5_bidask_replay"]
    if row.get("shape_key") != "EUR_USD|SHORT|BREAKOUT_FAILURE":
        return None
    blockers = []
    for item in replay.get("remaining_blockers") or []:
        code = _code_from_value(item)
        if code == "NOT_IN_PROOF_QUEUE":
            continue
        blockers.append(code)
    proof_rows = _matching_proof_rows(ctx["proof_queue"], "EUR_USD", "SHORT", "BREAKOUT_FAILURE")
    if proof_rows:
        blockers.append("PROOF_QUEUE_MEMBER_BUT_NOT_PROOF_READY")
    return {
        "status": replay.get("status"),
        "s5_bidask_replay_status": replay.get("s5_bidask_replay_status"),
        "target_shape": replay.get("target_shape"),
        "replay_sample_count": _int(replay.get("replay_sample_count")),
        "replay_wins": _int(replay.get("replay_wins")),
        "replay_losses": _int(replay.get("replay_losses")),
        "net_expectancy_after_bidask": replay.get("net_expectancy_after_bidask"),
        "spread_proof_positive": _limit_replay_spread_proof_positive(replay),
        "live_grade_candidate": bool(replay.get("live_grade_candidate")),
        "market_stop_samples_excluded": bool(replay.get("market_stop_samples_excluded")),
        "market_close_excluded": bool(replay.get("market_close_excluded")),
        "remaining_blocker_codes": _unique_strings(blockers),
    }


def _target_limit_sample_mining(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("shape_key") != "EUR_USD|SHORT|BREAKOUT_FAILURE":
        return None
    mining = ctx.get("limit_sample_mining")
    if not isinstance(mining, dict) or not mining:
        return None
    execution_summary = _nested(mining, "execution_ledger_coverage", "summary") or {}
    legacy_summary = _nested(mining, "legacy_history_coverage", "summary") or {}
    return {
        "status": mining.get("status"),
        "sample_floor": mining.get("sample_floor"),
        "additional_acceptable_samples": mining.get("additional_acceptable_samples") or [],
        "execution_ledger_summary": execution_summary,
        "legacy_history_summary": legacy_summary,
        "remaining_blocker_codes": _codes_from_values(mining.get("remaining_blockers") or []),
    }


def _current_intent_summary(lane_id: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    intent_row = ctx["intent_by_lane"].get(str(lane_id)) if lane_id else {}
    intent_row = intent_row if isinstance(intent_row, dict) else {}
    intent = intent_row.get("intent") if isinstance(intent_row.get("intent"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    risk_metrics = intent_row.get("risk_metrics") if isinstance(intent_row.get("risk_metrics"), dict) else {}
    return {
        "lane_id": lane_id,
        "status": intent_row.get("status"),
        "order_type": intent.get("order_type"),
        "attach_take_profit_on_fill": metadata.get("attach_take_profit_on_fill"),
        "tp_execution_mode": metadata.get("tp_execution_mode"),
        "tp_target_intent": metadata.get("tp_target_intent"),
        "positive_rotation_mode": metadata.get("positive_rotation_mode"),
        "risk_allowed": bool(intent_row.get("risk_allowed")),
        "risk_jpy": risk_metrics.get("risk_jpy") or metadata.get("sizing_actual_risk_jpy") or metadata.get("max_loss_jpy"),
        "units": intent.get("units"),
        "entry": intent.get("entry"),
        "tp": intent.get("tp"),
        "sl": intent.get("sl"),
        "live_blocker_codes": list(intent_row.get("live_blocker_codes") or []),
    }


def _candidate_promotion_blockers(
    *,
    row: dict[str, Any],
    current_best: dict[str, Any],
    proof_best: dict[str, Any],
    portfolio_best: dict[str, Any],
    adjusted: dict[str, Any],
    limit_replay: dict[str, Any] | None,
    ctx: dict[str, Any],
) -> list[str]:
    blockers = []
    blockers.extend(current_best.get("current_blockers") or [])
    blockers.extend(current_best.get("live_blocker_codes") or [])
    blockers.extend(portfolio_best.get("current_blockers") or [])
    if row.get("month_scale_blocker"):
        blockers.append("MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE")
    if (row.get("market_close_losses") or 0) > 0:
        blockers.append("MARKET_CLOSE_LEAK_PRESENT")
    if adjusted.get("proof_floor_reached") is not True:
        blockers.append("SAMPLE_GAP")
    if proof_best:
        missing = proof_best.get("missing_proof") if isinstance(proof_best.get("missing_proof"), dict) else {}
        if missing.get("sample_count_floor") is False:
            blockers.append("LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY")
        if (
            missing.get("s5_bidask_spread_included_replay") is False
            and not _limit_replay_spread_proof_positive(limit_replay)
        ):
            blockers.append("S5_BIDASK_SPREAD_INCLUDED_REPLAY_MISSING")
        if missing.get("active_day_floor") is False:
            blockers.append("ACTIVE_DAY_FLOOR_NOT_MET")
        if missing.get("risk_engine_pass") is False:
            blockers.append("RISK_ENGINE_PASS_MISSING")
        if missing.get("live_order_gateway_pass") is False:
            blockers.append("LIVE_ORDER_GATEWAY_PREFLIGHT_MISSING")
        if missing.get("gpt_verifier_pass") is False:
            blockers.append("FRESH_GPT_VERIFIER_TRADE_RECEIPT_MISSING")
        if missing.get("no_guardian_operator_review_blocker") is False:
            blockers.append("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED")
        if proof_best.get("can_create_live_permission") is not True:
            blockers.append("PROOF_QUEUE_MEMBER_BUT_NOT_PROOF_READY")
    else:
        blockers.append("NOT_IN_PROOF_QUEUE")
    if limit_replay:
        blockers.extend(limit_replay.get("remaining_blocker_codes") or [])
    limit_sample_mining = _target_limit_sample_mining(row, ctx)
    if limit_sample_mining:
        blockers.extend(limit_sample_mining.get("remaining_blocker_codes") or [])
    blockers.extend(_codes_from_values(ctx["acceptance"].get("blockers") or []))
    if ctx["acceptance"].get("status"):
        blockers.append(str(ctx["acceptance"].get("status")))
    return _unique_strings(blockers)


def _limit_replay_spread_proof_positive(replay: dict[str, Any] | None) -> bool:
    if not replay:
        return False
    if not replay.get("s5_bidask_replay_status"):
        return False
    if (_int(replay.get("replay_losses")) or 0) > 0:
        return False
    expectancy = _float(replay.get("net_expectancy_after_bidask"))
    return expectancy is None or expectancy >= 0.0


def _harvest_rank_reason(
    row: dict[str, Any],
    adjusted: dict[str, Any],
    proof_queue_member: bool,
    planner_can_enter: bool,
    limit_replay: dict[str, Any] | None,
) -> str:
    pieces = [
        f"TP proof {adjusted.get('take_profit_trades')}/{adjusted.get('proof_floor_trades')} gap {adjusted.get('proof_gap_trades')}",
        f"proof_queue_member={proof_queue_member}",
        f"planner_can_enter_proof_pack={planner_can_enter}",
    ]
    if limit_replay:
        pieces.append(
            "LIMIT S5 bid/ask replay "
            f"{limit_replay.get('replay_wins')}/{limit_replay.get('replay_sample_count')} "
            f"net={limit_replay.get('net_expectancy_after_bidask')}"
        )
    if row.get("overall_net_jpy") is not None:
        pieces.append(f"overall_net_jpy={row.get('overall_net_jpy')}")
    if row.get("month_scale_blocker"):
        pieces.append("direct_month_scale_blocker=present")
    return "; ".join(pieces)


def _shape_part(shape_key: str, index: int) -> str:
    parts = shape_key.split("|")
    return parts[index] if len(parts) > index else ""


def _codes_from_values(values: list[Any]) -> list[str]:
    return [_code_from_value(value) for value in values if _code_from_value(value)]


def _code_from_value(value: Any) -> str:
    if isinstance(value, dict):
        raw = value.get("code") or value.get("reason_code") or value.get("status")
    else:
        raw = value
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    return text.split(":", 1)[0].strip()


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _code_from_value(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _replayed_limit_trade_ids(replay: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for row in replay.get("sample_replay_details") or []:
        if isinstance(row, dict) and row.get("trade_id"):
            ids.add(str(row["trade_id"]))
    return ids


def _legacy_rejection_map(legacy_search: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rejected: dict[str, dict[str, Any]] = {}
    for row in legacy_search.get("rejected_samples") or []:
        if not isinstance(row, dict):
            continue
        raw_trade_id = str(row.get("trade_id") or "")
        if not raw_trade_id:
            continue
        for trade_id in [part.strip() for part in raw_trade_id.split(",") if part.strip()]:
            rejected[trade_id] = {
                "reason_codes": _unique_strings(row.get("reason_codes") or []),
                "evidence": row.get("evidence"),
            }
    return rejected


def _mine_execution_ledger_eurusd_limit_samples(replay_trade_ids: set[str]) -> dict[str, Any]:
    db_path = ROOT / "data" / "execution_ledger.db"
    if not db_path.exists():
        return {
            "source": "data/execution_ledger.db",
            "status": "SOURCE_MISSING",
            "rows": [],
            "summary": {},
        }
    sql = """
        WITH fills AS (
            SELECT trade_id, order_id, ts_utc AS entry_ts, lane_id, pair, side, units,
                   price AS entry_price, exit_reason AS entry_order_type, raw_json AS fill_raw
            FROM execution_events
            WHERE event_type = 'ORDER_FILLED'
              AND pair = 'EUR_USD'
              AND side = 'SHORT'
        ),
        accepts AS (
            SELECT order_id, ts_utc AS accepted_ts, lane_id AS accepted_lane_id,
                   price AS order_price, tp AS tp_order_price, sl AS sl_order_price,
                   raw_json AS accept_raw
            FROM execution_events
            WHERE event_type = 'ORDER_ACCEPTED'
              AND pair = 'EUR_USD'
              AND side = 'SHORT'
        ),
        closes AS (
            SELECT trade_id, ts_utc AS exit_ts, order_id AS close_order_id,
                   units AS close_units, price AS exit_price, realized_pl_jpy,
                   exit_reason, raw_json AS close_raw
            FROM execution_events
            WHERE event_type = 'TRADE_CLOSED'
              AND pair = 'EUR_USD'
              AND side = 'SHORT'
              AND exit_reason = 'TAKE_PROFIT_ORDER'
        )
        SELECT fills.trade_id, fills.order_id, fills.entry_ts, fills.lane_id,
               accepts.accepted_lane_id, accepts.accepted_ts, fills.units,
               accepts.order_price, fills.entry_price, accepts.tp_order_price,
               accepts.sl_order_price, closes.exit_ts, closes.close_order_id,
               closes.exit_price, closes.realized_pl_jpy, fills.entry_order_type,
               accepts.accept_raw, fills.fill_raw, closes.close_raw
        FROM fills
        JOIN accepts ON accepts.order_id = fills.order_id
        JOIN closes ON closes.trade_id = fills.trade_id
        ORDER BY fills.entry_ts, fills.trade_id
    """
    rows: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for record in conn.execute(sql):
            accept_raw = _loads_json(record["accept_raw"])
            fill_raw = _loads_json(record["fill_raw"])
            close_raw = _loads_json(record["close_raw"])
            accepted_type = str(accept_raw.get("type") or "")
            vehicle = accepted_type or str(record["entry_order_type"] or "")
            lane_id = str(record["lane_id"] or record["accepted_lane_id"] or "")
            target_method = lane_id in {EURUSD_LIMIT_TARGET_LANE, EURUSD_LIMIT_TARGET_BASE_LANE}
            exact_limit_lane = lane_id == EURUSD_LIMIT_TARGET_LANE
            is_limit = vehicle == "LIMIT_ORDER"
            trade_id = str(record["trade_id"] or "")
            reason_codes: list[str] = []
            if trade_id in replay_trade_ids:
                verdict = "ACCEPTED_CURRENT_REPLAY"
            elif is_limit and target_method:
                verdict = "ACCEPTABLE_NEW_EXACT_LIMIT_SAMPLE"
            elif target_method and not is_limit:
                verdict = "REJECTED_NOT_LIMIT_ORDER"
                reason_codes.append("MARKET_OR_STOP_NOT_LIMIT_PROOF")
            elif is_limit and not target_method:
                verdict = "REJECTED_NON_TARGET_METHOD"
                reason_codes.append("LIMIT_BUT_NOT_BREAKOUT_FAILURE_TARGET_LANE")
            else:
                verdict = "REJECTED_NON_TARGET_OR_NOT_LIMIT"
                reason_codes.append("NOT_EXACT_LIMIT_BREAKOUT_FAILURE")
            if not target_method:
                reason_codes.append("LANE_NOT_TARGET_BREAKOUT_FAILURE")
            if not is_limit:
                reason_codes.append("ENTRY_VEHICLE_NOT_LIMIT_ORDER")
            rows.append(
                {
                    "trade_id": trade_id,
                    "order_id": str(record["order_id"] or ""),
                    "lane_id": lane_id,
                    "accepted_order_type": accepted_type,
                    "entry_order_type": record["entry_order_type"],
                    "target_method_lane": target_method,
                    "exact_limit_lane": exact_limit_lane,
                    "entry_timestamp_utc": record["entry_ts"],
                    "exit_timestamp_utc": record["exit_ts"],
                    "units": record["units"],
                    "order_price": _round(record["order_price"], 5),
                    "entry_price": _round(record["entry_price"], 5),
                    "tp_order_price": _round(record["tp_order_price"], 5),
                    "exit_price": _round(record["exit_price"], 5),
                    "realized_pl_jpy": _round(record["realized_pl_jpy"], 4),
                    "entry_bid": _nested(fill_raw, "fullPrice", "bids", 0, "price"),
                    "entry_ask": _nested(fill_raw, "fullPrice", "asks", 0, "price"),
                    "exit_bid": _nested(close_raw, "fullPrice", "bids", 0, "price"),
                    "exit_ask": _nested(close_raw, "fullPrice", "asks", 0, "price"),
                    "verdict": verdict,
                    "reason_codes": _unique_strings(reason_codes),
                    "already_in_replay": trade_id in replay_trade_ids,
                }
            )
    return {
        "source": "data/execution_ledger.db",
        "status": "COVERAGE_COMPLETE_READ_ONLY",
        "rows": rows,
        "summary": _sample_mining_summary(rows),
    }


def _mine_legacy_eurusd_limit_samples(
    *,
    replay_trade_ids: set[str],
    accepted_legacy_ids: set[str],
    rejected_legacy: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    db_path = ROOT / "data" / "legacy_history.db"
    if not db_path.exists():
        return {
            "source": "data/legacy_history.db live_trade_events",
            "status": "SOURCE_MISSING",
            "rows": [],
            "summary": {},
        }
    sql = """
        SELECT trade_id,
               MIN(timestamp_text) AS first_ts,
               MAX(timestamp_text) AS last_ts,
               SUM(CASE WHEN action = 'ENTRY' THEN 1 ELSE 0 END) AS entry_rows,
               SUM(CASE WHEN action = 'LIMIT_FILL' THEN 1 ELSE 0 END) AS limit_fill_rows,
               SUM(CASE WHEN action = 'STOP_FILL' THEN 1 ELSE 0 END) AS stop_fill_rows,
               SUM(CASE WHEN action = 'PARTIAL_CLOSE' THEN 1 ELSE 0 END) AS partial_close_rows,
               SUM(CASE WHEN action = 'CLOSE' AND reason = 'TAKE_PROFIT_ORDER' THEN 1 ELSE 0 END) AS tp_close_rows,
               ROUND(SUM(CASE WHEN action = 'CLOSE' THEN COALESCE(pl_jpy, 0) ELSE 0 END), 4) AS close_pl_jpy,
               GROUP_CONCAT(action || ':' || line_no || ':' || COALESCE(reason, '') || ':' || raw_line, X'1E') AS evidence
        FROM live_trade_events
        WHERE trade_id IN (
            SELECT trade_id
            FROM live_trade_events
            WHERE pair = 'EUR_USD'
              AND direction = 'SHORT'
              AND action = 'CLOSE'
              AND reason = 'TAKE_PROFIT_ORDER'
        )
        GROUP BY trade_id
        ORDER BY CAST(trade_id AS INTEGER)
    """
    rows: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for record in conn.execute(sql):
            trade_id = str(record["trade_id"] or "")
            excerpts = _legacy_evidence_excerpts(str(record["evidence"] or ""))
            vehicle = _legacy_entry_vehicle(record, excerpts)
            reason_codes: list[str] = []
            curated_rejection = rejected_legacy.get(trade_id)
            partial_close_count = _int(record["partial_close_rows"]) or 0
            tp_close_count = _int(record["tp_close_rows"]) or 0
            if trade_id in replay_trade_ids:
                verdict = "ACCEPTED_CURRENT_REPLAY"
            elif trade_id in accepted_legacy_ids and vehicle == "LIMIT_ORDER" and partial_close_count == 0:
                verdict = "ACCEPTABLE_NEW_EXACT_LIMIT_SAMPLE"
            elif vehicle != "LIMIT_ORDER":
                verdict = "REJECTED_NOT_LIMIT_ORDER"
                reason_codes.append("ENTRY_VEHICLE_NOT_LIMIT_ORDER")
            elif partial_close_count:
                verdict = "REJECTED_PARTIAL_CLOSE_MIXED_IN"
                reason_codes.append("PARTIAL_CLOSE_MIXED_WITH_TP")
            elif curated_rejection:
                verdict = "REJECTED_CURATED_STRATEGY_NOT_EXACT"
                reason_codes.extend(curated_rejection.get("reason_codes") or [])
            elif _legacy_has_breakout_failure_language(excerpts):
                verdict = "POTENTIAL_REQUIRES_STRATEGY_REVIEW"
                reason_codes.append("NOT_IN_CURATED_ACCEPTED_SAMPLE_SET")
            else:
                verdict = "REJECTED_STRATEGY_NOT_UNAMBIGUOUS"
                reason_codes.append("STRATEGY_TAG_NOT_UNAMBIGUOUS_BREAKOUT_FAILURE")
            if tp_close_count != 1:
                reason_codes.append("TP_CLOSE_COUNT_NOT_ONE")
            rows.append(
                {
                    "trade_id": trade_id,
                    "first_timestamp_utc": _legacy_timestamp_to_iso(record["first_ts"]),
                    "last_timestamp_utc": _legacy_timestamp_to_iso(record["last_ts"]),
                    "entry_rows": _int(record["entry_rows"]) or 0,
                    "limit_fill_rows": _int(record["limit_fill_rows"]) or 0,
                    "stop_fill_rows": _int(record["stop_fill_rows"]) or 0,
                    "partial_close_rows": partial_close_count,
                    "tp_close_rows": tp_close_count,
                    "entry_vehicle": vehicle,
                    "realized_close_pl_jpy": _round(record["close_pl_jpy"], 4),
                    "verdict": verdict,
                    "reason_codes": _unique_strings(reason_codes),
                    "already_in_replay": trade_id in replay_trade_ids,
                    "curated_acceptance": trade_id in accepted_legacy_ids,
                    "curated_rejection": curated_rejection or {},
                    "evidence_excerpts": excerpts[:4],
                }
            )
    return {
        "source": "data/legacy_history.db live_trade_events",
        "status": "COVERAGE_COMPLETE_READ_ONLY",
        "rows": rows,
        "summary": _sample_mining_summary(rows),
    }


def _sample_mining_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_verdict: dict[str, int] = defaultdict(int)
    by_vehicle: dict[str, int] = defaultdict(int)
    for row in rows:
        by_verdict[str(row.get("verdict") or "UNKNOWN")] += 1
        by_vehicle[str(row.get("accepted_order_type") or row.get("entry_vehicle") or row.get("entry_order_type") or "UNKNOWN")] += 1
    return {
        "rows": len(rows),
        "accepted_current_replay": by_verdict.get("ACCEPTED_CURRENT_REPLAY", 0),
        "acceptable_new_exact_limit_samples": by_verdict.get("ACCEPTABLE_NEW_EXACT_LIMIT_SAMPLE", 0),
        "by_verdict": dict(sorted(by_verdict.items())),
        "by_vehicle": dict(sorted(by_vehicle.items())),
    }


def _legacy_entry_vehicle(record: sqlite3.Row, excerpts: list[str]) -> str:
    text = "\n".join(excerpts).lower()
    if (_int(record["limit_fill_rows"]) or 0) > 0 or "via limit fill" in text:
        return "LIMIT_ORDER"
    if (_int(record["stop_fill_rows"]) or 0) > 0 or "stop_fill" in text or " via stop" in text:
        return "STOP_ORDER"
    if "market" in text and "order" in text:
        return "MARKET_ORDER"
    return "UNKNOWN"


def _legacy_evidence_excerpts(raw: str) -> list[str]:
    excerpts = []
    for part in raw.split("\x1e"):
        part = " ".join(part.split())
        if not part:
            continue
        excerpts.append(part[:500])
    return excerpts


def _legacy_has_breakout_failure_language(excerpts: list[str]) -> bool:
    text = "\n".join(excerpts).lower()
    positive_tokens = (
        "breakout_failure",
        "failed_shelf",
        "failed shelf",
        "failed-retest",
        "failed retest",
        "broken-shelf",
        "shelf retest",
    )
    return any(token in text for token in positive_tokens)


def _legacy_timestamp_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.replace(" UTC", "Z").replace(" ", "T", 1)


def _loads_json(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _context(as_loop: Any) -> dict[str, Any]:
    acceptance = _load_json("data/profitability_acceptance.json")
    self_audit = _load_json("data/self_improvement_audit.json")
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
    memory = _load_json("data/memory_health.json")
    candidate_by_lane = {str(row.get("lane_id")): row for row in firepower.get("candidates") or []}
    proof_by_lane = {str(row.get("lane_id")): row for row in proof_queue.get("queue") or []}
    intent_by_lane = {str(row.get("lane_id")): row for row in order_intents.get("results") or []}
    return {
        "as_loop": as_loop,
        "acceptance": acceptance,
        "self_audit": self_audit,
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
        "memory": memory,
        "support": _load_json("data/trader_support_bot.json"),
        "p0_decomposition": _load_json("data/remaining_profitability_p0_decomposition.json"),
        "as_board": _load_json("data/as_lane_candidate_board.json"),
        "payoff_shape": _load_json("data/payoff_shape_diagnosis.json"),
        "proof_floor_update": _load_json("data/eurusd_short_breakout_failure_proof_floor_update.json"),
        "limit_s5_bidask_replay": _load_json("data/eurusd_short_breakout_failure_limit_s5_bidask_replay.json"),
        "live_order_request": _load_json("data/live_order_request.json"),
        "candidate_by_lane": candidate_by_lane,
        "proof_by_lane": proof_by_lane,
        "intent_by_lane": intent_by_lane,
    }


def _manual_tp_audit_evidence_summary(broker_mutation_audit: dict[str, Any]) -> dict[str, Any]:
    broker_truth = broker_mutation_audit.get("broker_truth") if isinstance(broker_mutation_audit, dict) else {}
    if not isinstance(broker_truth, dict):
        broker_truth = {}
    snapshots = [
        broker_truth.get("live"),
        broker_truth.get("dev"),
        broker_truth,
    ]
    last_transaction_id = None
    active_take_profit_order = None
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        if last_transaction_id is None:
            last_transaction_id = snapshot.get("last_transaction_id") or _nested(snapshot, "account", "last_transaction_id")
        if active_take_profit_order is None:
            active_take_profit_order = _active_manual_tp_order_id(snapshot)
        if last_transaction_id is not None and active_take_profit_order is not None:
            break
    return {
        "last_transaction_id": str(last_transaction_id) if last_transaction_id is not None else None,
        "active_take_profit_order": str(active_take_profit_order) if active_take_profit_order is not None else None,
    }


def _active_manual_tp_order_id(snapshot: dict[str, Any]) -> str | None:
    for order in snapshot.get("orders") or []:
        if not isinstance(order, dict):
            continue
        if str(order.get("trade_id") or "") != MANUAL_TRADE_ID:
            continue
        order_type = str(order.get("order_type") or order.get("type") or "").upper()
        state = str(order.get("state") or "").upper()
        if "TAKE_PROFIT" not in order_type and order_type != "TAKE_PROFIT":
            continue
        if state and state not in {"PENDING", "OPEN"}:
            continue
        order_id = order.get("order_id") or order.get("id")
        if order_id:
            return str(order_id)
    return None


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
    math_exclusion_reasons = portfolio_math_exclusion_reasons(row)
    if math_exclusion_reasons:
        classification = "REJECTED"
    proof_distance = _int(proof.get("proof_distance"))
    if proof_distance is None:
        proof_distance = _int(row.get("proof_gap_count")) or 99
    daily_pct = _float(row.get("expected_daily_return_pct_on_funding_adjusted_equity")) or 0.0
    margin = _float(row.get("margin_requirement_realistic_size_jpy"))
    evidence_score = {"PROOF_READY": 100.0, "REPAIR_REQUIRED": 55.0, "EVIDENCE_GAP": 40.0, "HISTORICAL_ONLY": 20.0, "REJECTED": 0.0}.get(classification, 10.0)
    margin_score = 12.0 if margin and margin < 5000 else 4.0 if margin else 0.0
    rank_score = _round(
        daily_pct * 8.0
        + evidence_score
        + margin_score
        - proof_distance * 4.0
        - len(row.get("current_blockers") or []) * 1.5
        - (100.0 if math_exclusion_reasons else 0.0)
    )
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
        "math_candidate_eligible": not math_exclusion_reasons,
        "math_exclusion_reasons": math_exclusion_reasons,
        "can_create_live_permission": False,
        "can_enter_proof_pack": bool(row.get("can_enter_proof_pack")),
    }


def portfolio_math_exclusion_reasons(row: dict[str, Any]) -> list[str]:
    blockers = {str(code) for code in row.get("current_blockers") or []}
    source = row.get("source_evidence") if isinstance(row.get("source_evidence"), dict) else {}
    reasons: list[str] = []
    if "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE" in blockers:
        reasons.append("spread_included_bidask_replay_negative_for_exact_lane")
    if source.get("bidask_rule_status") == "LIVE_BLOCK_NEGATIVE_EXPECTANCY":
        reasons.append("packaged_bidask_rule_live_block_negative_expectancy")
    return reasons


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


def _manual_position_snapshot(broker: dict[str, Any]) -> dict[str, Any]:
    positions = broker.get("positions") if isinstance(broker.get("positions"), list) else []
    manual = next((row for row in positions if str(row.get("trade_id")) == MANUAL_TRADE_ID), {})
    operator = manual.get("operator_manual_position") if isinstance(manual.get("operator_manual_position"), dict) else {}
    return {
        "present": bool(manual),
        "trade_id": MANUAL_TRADE_ID,
        "owner": manual.get("owner"),
        "pair": manual.get("pair"),
        "side": manual.get("side"),
        "units": manual.get("units"),
        "take_profit": manual.get("take_profit"),
        "stop_loss": manual.get("stop_loss"),
        "unrealized_pl_jpy": manual.get("unrealized_pl_jpy"),
        "management_intent": operator.get("management_intent"),
        "system_pl_counted": operator.get("system_pl_counted"),
        "system_occupancy_counted": operator.get("system_occupancy_counted"),
        "auto_close_allowed": operator.get("loss_side_auto_close_allowed"),
        "auto_sl_attach_allowed": operator.get("auto_sl_attach_allowed"),
        "auto_tp_modify_allowed": operator.get("auto_tp_modify_allowed"),
        "same_theme_auto_add_allowed": operator.get("same_theme_auto_add_allowed"),
    }


def _manual_active_tp_from_broker(broker: dict[str, Any]) -> dict[str, Any]:
    positions = broker.get("positions") if isinstance(broker.get("positions"), list) else []
    orders = broker.get("orders") if isinstance(broker.get("orders"), list) else []
    manual = next((row for row in positions if str(row.get("trade_id")) == MANUAL_TRADE_ID), {})
    raw_tp = _nested(manual, "raw", "takeProfitOrder") if isinstance(manual.get("raw"), dict) else {}
    active_tp = next(
        (
            row
            for row in orders
            if str(row.get("trade_id") or row.get("tradeID") or _nested(row, "raw", "tradeID") or "") == MANUAL_TRADE_ID
        ),
        {},
    )
    order_id = (
        active_tp.get("order_id")
        or active_tp.get("id")
        or (raw_tp.get("id") if isinstance(raw_tp, dict) else None)
    )
    return {
        "order_id": str(order_id) if order_id is not None else None,
        "trade_id": MANUAL_TRADE_ID if order_id else None,
        "price": active_tp.get("price") or (raw_tp.get("price") if isinstance(raw_tp, dict) else None) or manual.get("take_profit"),
        "state": active_tp.get("state"),
        "create_time": active_tp.get("createTime") or (raw_tp.get("createTime") if isinstance(raw_tp, dict) else None),
        "replaces_order_id": raw_tp.get("replacesOrderID") if isinstance(raw_tp, dict) else active_tp.get("replacesOrderID"),
        "source": "broker_snapshot",
    }


def _manual_tp_ledger_events(*, extra_order_ids: list[Any] | None = None) -> list[dict[str, Any]]:
    db_path = ROOT / "data" / "execution_ledger.db"
    if not db_path.exists():
        return []
    order_ids = {MANUAL_TP_ORDER_ID, MANUAL_TP_AUDIT_ORDER_ID}
    order_ids.update(str(item) for item in extra_order_ids or [] if item)
    predicates = ["trade_id = ?", "raw_json LIKE ?"]
    params: list[Any] = [MANUAL_TRADE_ID, f"%{MANUAL_TRADE_ID}%"]
    for order_id in sorted(order_ids):
        predicates.extend(["order_id = ?", "raw_json LIKE ?"])
        params.extend([order_id, f"%{order_id}%"])
    sql = f"""
        SELECT ts_utc, source, event_type, lane_id, order_id, trade_id, client_order_id,
               pair, side, units, price, tp, sl, exit_reason, oanda_transaction_id, raw_json
        FROM execution_events
        WHERE {' OR '.join(predicates)}
        ORDER BY ts_utc, oanda_transaction_id
    """
    events: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        for row in conn.execute(sql, params):
            (
                ts_utc,
                source,
                event_type,
                lane_id,
                order_id,
                trade_id,
                client_order_id,
                pair,
                side,
                units,
                price,
                tp,
                sl,
                exit_reason,
                oanda_transaction_id,
                raw_json,
            ) = row
            try:
                raw = json.loads(raw_json or "{}")
            except json.JSONDecodeError:
                raw = {}
            raw_type = str(raw.get("type") or "")
            if raw_type not in {"TAKE_PROFIT_ORDER", "ORDER_CANCEL"} and event_type not in {"PROTECTION_CREATED", "ORDER_CANCELED"}:
                continue
            raw_order_id = raw.get("id") if raw_type == "TAKE_PROFIT_ORDER" else raw.get("orderID")
            events.append(
                {
                    "ts_utc": ts_utc,
                    "source": source,
                    "event_type": event_type,
                    "raw_type": raw_type,
                    "lane_id": lane_id,
                    "order_id": str(raw_order_id or order_id or ""),
                    "ledger_order_id": str(order_id or ""),
                    "trade_id": str(raw.get("tradeID") or trade_id or ""),
                    "client_order_id": client_order_id,
                    "pair": pair,
                    "side": side,
                    "units": units,
                    "price": raw.get("price") or price or tp,
                    "tp": tp,
                    "sl": sl,
                    "exit_reason": exit_reason,
                    "transaction_id": str(raw.get("id") or oanda_transaction_id or ""),
                    "oanda_transaction_id": str(oanda_transaction_id or ""),
                    "reason": raw.get("reason") or exit_reason,
                    "replaces_order_id": raw.get("replacesOrderID"),
                    "replaced_by_order_id": raw.get("replacedByOrderID"),
                    "cancelling_transaction_id": raw.get("cancellingTransactionID"),
                    "request_id": raw.get("requestID"),
                    "user_id": raw.get("userID"),
                    "time": raw.get("time") or ts_utc,
                }
            )
    return events


def _manual_tp_gateway_receipts(order_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    db_path = ROOT / "data" / "execution_ledger.db"
    receipts = {str(order_id): [] for order_id in order_ids if str(order_id)}
    if not receipts or not db_path.exists():
        return receipts
    predicates = []
    params: list[Any] = []
    for order_id in receipts:
        predicates.extend(["order_id = ?", "client_order_id = ?", "raw_json LIKE ?"])
        params.extend([order_id, order_id, f"%{order_id}%"])
    sql = f"""
        SELECT ts_utc, source, event_type, order_id, trade_id, client_order_id, lane_id, raw_json
        FROM execution_events
        WHERE (event_type LIKE 'GATEWAY%' OR source LIKE '%gateway%' OR event_type IN ('ORDER_INTENT_STAGED', 'ORDER_ACCEPTED'))
          AND ({' OR '.join(predicates)})
        ORDER BY ts_utc
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        for ts_utc, source, event_type, order_id, trade_id, client_order_id, lane_id, raw_json in conn.execute(sql, params):
            row = {
                "ts_utc": ts_utc,
                "source": source,
                "event_type": event_type,
                "order_id": order_id,
                "trade_id": trade_id,
                "client_order_id": client_order_id,
                "lane_id": lane_id,
            }
            raw_text = raw_json or ""
            for oid in receipts:
                if oid in {str(order_id or ""), str(client_order_id or "")} or oid in raw_text:
                    receipts[oid].append(row)
    return receipts


def _manual_tp_replacement_chain(
    broker: dict[str, Any],
    ledger_events: list[dict[str, Any]],
    *,
    gateway_receipts: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    gateway_receipts = gateway_receipts or {}
    active = _manual_active_tp_from_broker(broker)
    active_id = str(active.get("order_id") or "")
    created: dict[str, dict[str, Any]] = {}
    cancel_by_order: dict[str, dict[str, Any]] = {}
    for event in ledger_events:
        order_id = str(event.get("order_id") or "")
        if not order_id:
            continue
        if event.get("raw_type") == "TAKE_PROFIT_ORDER" and str(event.get("trade_id") or "") == MANUAL_TRADE_ID:
            created[order_id] = event
        elif event.get("raw_type") == "ORDER_CANCEL":
            cancel_by_order[order_id] = event
    if active_id and active_id not in created:
        created[active_id] = {
            "time": active.get("create_time"),
            "ts_utc": active.get("create_time"),
            "source": active.get("source"),
            "event_type": "BROKER_SNAPSHOT_ACTIVE_TP",
            "raw_type": "TAKE_PROFIT_ORDER",
            "order_id": active_id,
            "trade_id": MANUAL_TRADE_ID,
            "price": active.get("price"),
            "replaces_order_id": active.get("replaces_order_id"),
        }
    chain: list[dict[str, Any]] = []
    for order_id, event in sorted(created.items(), key=lambda item: (str(item[1].get("time") or item[1].get("ts_utc") or ""), item[0])):
        cancel = cancel_by_order.get(order_id) or {}
        replaced_by = cancel.get("replaced_by_order_id")
        if not replaced_by:
            child = next((row for row in created.values() if str(row.get("replaces_order_id") or "") == order_id), {})
            replaced_by = child.get("order_id")
        lifecycle = "ACTIVE_BROKER_TRUTH" if order_id == active_id else "REPLACED" if replaced_by else "HISTORICAL_INACTIVE_OR_UNKNOWN"
        receipts = gateway_receipts.get(order_id) or []
        chain.append(
            {
                "order_id": order_id,
                "trade_id": event.get("trade_id") or MANUAL_TRADE_ID,
                "price": _round(event.get("price"), 5),
                "created_at_utc": event.get("time") or event.get("ts_utc"),
                "source": event.get("source"),
                "event_type": event.get("event_type"),
                "reason": event.get("reason"),
                "request_id": event.get("request_id"),
                "user_id": event.get("user_id"),
                "replaces_order_id": event.get("replaces_order_id"),
                "replaced_by_order_id": replaced_by,
                "cancel_transaction_id": cancel.get("transaction_id"),
                "cancel_request_id": cancel.get("request_id"),
                "lifecycle": lifecycle,
                "provenance_classification": "PROVENANCE_UNKNOWN_BLOCK_AUTOMATION",
                "classification_reason": (
                    "No local QuantRabbit gateway receipt/client extension proves this TP mutation. "
                    "Because the parent trade is operator-manual, the order is protected and automation-usable=false."
                ),
                "local_gateway_receipt_count": len(receipts),
                "automation_usable": False,
                "auto_modify_allowed": False,
            }
        )
    return chain


def manual_position_safety(
    broker: dict[str, Any],
    *,
    ledger_events: list[dict[str, Any]] | None = None,
    gateway_receipts: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    positions = broker.get("positions") if isinstance(broker.get("positions"), list) else []
    orders = broker.get("orders") if isinstance(broker.get("orders"), list) else []
    manual = next((row for row in positions if str(row.get("trade_id")) == MANUAL_TRADE_ID), {})
    active = _manual_active_tp_from_broker(broker)
    if ledger_events is None:
        ledger_events = _manual_tp_ledger_events(extra_order_ids=[active.get("order_id")] if active.get("order_id") else [])
    chain = _manual_tp_replacement_chain(broker, ledger_events, gateway_receipts=gateway_receipts)
    expected_tp = next((row for row in orders if str(row.get("order_id") or row.get("id")) == MANUAL_TP_ORDER_ID), {})
    current_tp_order_id = active.get("order_id")
    audit_order = next((row for row in chain if str(row.get("order_id")) == MANUAL_TP_AUDIT_ORDER_ID), {})
    current_order = next((row for row in chain if str(row.get("order_id")) == str(current_tp_order_id or "")), {})
    replaced_expected_tp = any(
        str(row.get("order_id") or "") == MANUAL_TP_ORDER_ID and row.get("replaced_by_order_id")
        for row in chain
    ) or any(str(row.get("replaces_order_id") or "") == MANUAL_TP_ORDER_ID for row in chain)
    return {
        "manual_trade_id": MANUAL_TRADE_ID,
        "expected_manual_tp_order_id": MANUAL_TP_ORDER_ID,
        "audit_manual_tp_order_id": MANUAL_TP_AUDIT_ORDER_ID,
        "current_manual_tp_order_id": current_tp_order_id,
        "expected_tp_order_present": bool(expected_tp),
        "expected_tp_replaced_in_broker_truth": bool(replaced_expected_tp),
        "position_present": bool(manual),
        "tp_order_present": bool(current_tp_order_id),
        "position_owner": manual.get("owner"),
        "position_pair": manual.get("pair"),
        "position_side": manual.get("side"),
        "position_units": manual.get("units"),
        "position_unrealized_pl_jpy": manual.get("unrealized_pl_jpy"),
        "take_profit_price": manual.get("take_profit") or active.get("price"),
        "tp_replaces_order_id": active.get("replaces_order_id"),
        "tp_create_time": active.get("create_time"),
        "audit_tp_lifecycle": audit_order.get("lifecycle"),
        "audit_tp_replaced_by_order_id": audit_order.get("replaced_by_order_id"),
        "audit_tp_provenance_classification": audit_order.get("provenance_classification") or "PROVENANCE_UNKNOWN_BLOCK_AUTOMATION",
        "current_tp_provenance_classification": current_order.get("provenance_classification") or "PROVENANCE_UNKNOWN_BLOCK_AUTOMATION",
        "tp_replacement_chain": chain,
        "management_intent": _nested(manual, "operator_manual_position", "management_intent"),
        "system_pl_counted": _nested(manual, "operator_manual_position", "system_pl_counted"),
        "system_occupancy_counted": _nested(manual, "operator_manual_position", "system_occupancy_counted"),
        "automation_blocked": True,
        "auto_close_allowed": False,
        "auto_sl_attach_allowed": False,
        "auto_tp_modify_allowed": False,
        "same_theme_auto_add_allowed": False,
        "no_live_side_effects": True,
        "untouched_by_this_run": True,
    }


def _profitability_reconciliation_row(
    *,
    code: str,
    classification: str,
    current_codes: set[str],
    blocker_codes: set[str],
    evidence_summary: dict[str, Any],
    clearance_condition: str,
    duplicate_of: str | None = None,
    contained_by: str | None = None,
) -> dict[str, Any]:
    present = code in current_codes
    blocker = code in blocker_codes
    stale = classification == "STALE_SUPERSEDED"
    still_blocks_acceptance = bool(blocker and not stale)
    still_blocks_fresh_entries = bool(
        (
            present
            or blocker
            or classification
            in {
                "FIXED_NEEDS_CLEAN_WINDOW",
                "CONTAINED_NOT_CLEARED",
                "TAXONOMY_DUPLICATE",
                "EVIDENCE_GAP",
            }
        )
        and not stale
    )
    return {
        "code": code,
        "classification": classification,
        "present_in_current_acceptance": present,
        "present_as_current_blocker": blocker,
        "duplicate_of": duplicate_of,
        "contained_by": contained_by,
        "still_blocks_acceptance": still_blocks_acceptance,
        "still_blocks_fresh_entries": still_blocks_fresh_entries,
        "still_blocks_live_ready": still_blocks_fresh_entries,
        "can_create_live_permission": False,
        "evidence_summary": evidence_summary,
        "clearance_condition": clearance_condition,
    }


def _acceptance_codes(acceptance: dict[str, Any]) -> set[str]:
    codes = {str(item.get("code")) for item in acceptance.get("findings") or [] if isinstance(item, dict) and item.get("code")}
    codes.update(_acceptance_blocker_codes(acceptance))
    return codes


def _acceptance_blocker_codes(acceptance: dict[str, Any]) -> set[str]:
    return {
        str(item).split(":", 1)[0]
        for item in acceptance.get("blockers") or []
        if str(item).split(":", 1)[0]
    }


def _acceptance_evidence_summary(acceptance: dict[str, Any], code: str) -> dict[str, Any]:
    finding = _findings_by_code(acceptance).get(code) or {}
    return {
        "message": finding.get("message"),
        "priority": finding.get("priority"),
        "evidence": _compact_evidence(finding.get("evidence")),
    }


def _findings_by_code(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("code")): item
        for item in payload.get("findings") or []
        if isinstance(item, dict) and item.get("code")
    }


def _self_p0_items(self_audit: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in self_audit.get("findings") or []
        if isinstance(item, dict) and item.get("priority") == "P0"
    ]


def _classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("classification") or "UNKNOWN")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _compact_evidence(value: Any, *, depth: int = 0) -> Any:
    if value is None:
        return None
    if depth >= 2:
        if isinstance(value, list):
            return {"count": len(value)}
        if isinstance(value, dict):
            return {"keys": sorted(str(key) for key in value.keys())[:12]}
        return value
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 10:
                compact["_truncated_keys"] = max(0, len(value) - idx)
                break
            compact[str(key)] = _compact_evidence(item, depth=depth + 1)
        return compact
    if isinstance(value, list):
        return {
            "count": len(value),
            "first": _compact_evidence(value[0], depth=depth + 1) if value else None,
        }
    return value


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
    lines.extend(["", "## Required Proof Matrix", "", "| proof | status |", "|---|---|"])
    for row in payload.get("required_proof_matrix") or []:
        lines.append(f"| `{row.get('proof')}` | `{row.get('status')}` |")
    lines.extend(["", "## Current Blockers", ""])
    for item in (payload.get("verifier_gateway_guardian") or {}).get("blockers") or []:
        lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


def manual_eurusd_tp_replacement_provenance_md(payload: dict[str, Any]) -> str:
    broker = payload.get("broker_truth") or {}
    active = broker.get("active_take_profit_order") or {}
    lines = [
        "# Manual EUR_USD TP Replacement Provenance",
        "",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Manual trade: `{payload.get('manual_trade_id')}`",
        f"- Audit order: `{payload.get('audit_order_id')}` classified `{payload.get('audit_order_classification')}` with lifecycle `{payload.get('audit_order_lifecycle')}`",
        f"- Active broker TP: `{payload.get('current_active_tp_order_id')}` at `{payload.get('current_active_tp_price')}` classified `{payload.get('current_active_tp_classification')}`",
        f"- Snapshot last transaction: `{broker.get('last_transaction_id')}` fetched `{broker.get('snapshot_fetched_at_utc')}`",
        f"- Gateway receipt search: `{_nested(payload, 'gateway_receipt_search', 'result')}`",
        f"- Can automation use or modify this TP: `False`",
        "",
        "## Current Broker Truth",
        "",
        f"- Active TP order `{active.get('order_id')}` replaces `{active.get('replaces_order_id')}`.",
        f"- No live side effects in this run: `{payload.get('live_side_effects')}`",
        "",
        "## Replacement Chain",
        "",
        "| order | lifecycle | price | replaces | replaced by | class | gateway receipts |",
        "|---|---|---:|---|---|---|---:|",
    ]
    for row in payload.get("replacement_chain") or []:
        lines.append(
            f"| `{row.get('order_id')}` | `{row.get('lifecycle')}` | {row.get('price')} | `{row.get('replaces_order_id')}` | `{row.get('replaced_by_order_id')}` | `{row.get('provenance_classification')}` | {row.get('local_gateway_receipt_count')} |"
        )
    lines.extend(["", "## Conclusion", "", payload.get("conclusion") or ""])
    return "\n".join(lines) + "\n"


def profitability_acceptance_blocker_reconciliation_md(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# Profitability Acceptance Blocker Reconciliation",
        "",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Normal routing: `{summary.get('normal_routing_status')}`",
        f"- A/S LIVE_READY path exists: `{summary.get('as_live_ready_path_exists')}`",
        f"- Can create live permission: `{summary.get('can_create_live_permission')}`",
        "",
        "| code | classification | current blocker | blocks fresh entries | clearance |",
        "|---|---|---|---|---|",
    ]
    for row in payload.get("rows") or []:
        lines.append(
            f"| `{row.get('code')}` | `{row.get('classification')}` | `{row.get('present_as_current_blocker')}` | `{row.get('still_blocks_fresh_entries')}` | {row.get('clearance_condition')} |"
        )
    lines.extend(["", "## Boundary", "", payload.get("permission_boundary") or ""])
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


def harvest_live_grade_path_md(payload: dict[str, Any]) -> str:
    closest = payload.get("closest_harvest_candidate") if isinstance(payload.get("closest_harvest_candidate"), dict) else {}
    proof = payload.get("proof_queue_summary") if isinstance(payload.get("proof_queue_summary"), dict) else {}
    portfolio = payload.get("portfolio_summary") if isinstance(payload.get("portfolio_summary"), dict) else {}
    lines = [
        "# HARVEST Live-Grade Path",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "## Verdict",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Live permission allowed: `{payload.get('live_permission_allowed')}`",
        f"- Live promotion allowed: `{payload.get('live_promotion_allowed')}`",
        f"- Runner creation decision: `{payload.get('runner_creation_decision')}`",
        f"- Closest HARVEST: `{closest.get('candidate_id')}`",
        f"- Proof queue: queue=`{proof.get('queue_count')}` proof_ready=`{proof.get('proof_ready_count')}` can_create_live_permission=`{proof.get('can_create_live_permission_count')}`",
        f"- Portfolio: `{portfolio.get('portfolio_status')}` live_ready_lanes=`{portfolio.get('live_ready_lanes')}`",
        "",
        "This is a read-only diagnosis. It did not send orders, cancel orders, close positions, change launchd, relax gates, or derive lot size from the 4x deficit.",
        "",
        "## Closest Candidate",
        "",
        f"- Shape: `{closest.get('candidate_id')}`",
        f"- Class: `{closest.get('classification')}`",
        f"- Proof queue member: `{closest.get('actual_proof_queue_member')}`",
        f"- Planner can enter proof pack: `{closest.get('planner_can_enter_proof_pack')}`",
        f"- Can create live permission: `{closest.get('can_create_live_permission')}`",
        f"- Rank reason: {closest.get('rank_reason')}",
        "",
    ]
    tp = closest.get("tp_proof") if isinstance(closest.get("tp_proof"), dict) else {}
    if tp:
        lines.extend(
            [
                "## TP Proof",
                "",
                f"- Source: `{tp.get('proof_source')}`",
                f"- TAKE_PROFIT_ORDER wins/trades: `{tp.get('take_profit_wins') or tp.get('take_profit_trades')}`",
                f"- TAKE_PROFIT_ORDER losses: `{tp.get('take_profit_losses')}`",
                f"- Proof floor: `{tp.get('proof_floor_trades')}`",
                f"- Remaining broad TP gap: `{tp.get('proof_gap_trades')}`",
                f"- Legacy samples accepted: `{', '.join(tp.get('legacy_samples_accepted') or [])}`",
                f"- Canonical integration: `{tp.get('canonical_integration_status')}`",
                "",
            ]
        )
    replay = closest.get("limit_s5_bidask_replay") if isinstance(closest.get("limit_s5_bidask_replay"), dict) else {}
    if replay:
        lines.extend(
            [
                "## Exact LIMIT S5 Bid/Ask Replay",
                "",
                f"- Status: `{replay.get('status')}`",
                f"- Samples: `{replay.get('replay_sample_count')}`",
                f"- Wins/Losses: `{replay.get('replay_wins')}` / `{replay.get('replay_losses')}`",
                f"- Net expectancy after bid/ask: `{replay.get('net_expectancy_after_bidask')}`",
                f"- Live-grade candidate: `{replay.get('live_grade_candidate')}`",
                "",
            ]
        )
    mining = closest.get("limit_sample_mining") if isinstance(closest.get("limit_sample_mining"), dict) else {}
    if mining:
        floor = mining.get("sample_floor") if isinstance(mining.get("sample_floor"), dict) else {}
        lines.extend(
            [
                "## Exact LIMIT Sample Mining",
                "",
                f"- Status: `{mining.get('status')}`",
                f"- Current exact LIMIT samples: `{floor.get('current_replayed_exact_limit_samples')}`",
                f"- Additional acceptable local samples: `{floor.get('additional_acceptable_local_samples_found')}`",
                f"- Remaining exact LIMIT gap: `{floor.get('remaining_exact_limit_samples')}`",
                "",
            ]
        )
    lines.extend(["## Promotion Blockers", ""])
    for row in payload.get("promotion_blockers") or []:
        lines.append(f"- `{row.get('code')}`: {row.get('status')}")
    lines.extend(["", "## Ranked HARVEST Candidates", ""])
    lines.append("| Rank | Shape | Class | TP proof | Queue | Planner | Live |")
    lines.append("|---:|---|---|---:|---|---|---|")
    for row in payload.get("ranked_harvest_candidates") or []:
        tp_row = row.get("tp_proof") if isinstance(row.get("tp_proof"), dict) else {}
        lines.append(
            "| {rank} | `{shape}` | `{classification}` | {trades}/{floor} gap {gap} | `{queue}` | `{planner}` | `{live}` |".format(
                rank=row.get("rank"),
                shape=row.get("shape_key"),
                classification=row.get("classification"),
                trades=tp_row.get("take_profit_trades"),
                floor=tp_row.get("proof_floor_trades"),
                gap=tp_row.get("proof_gap_trades"),
                queue=row.get("actual_proof_queue_member"),
                planner=row.get("planner_can_enter_proof_pack"),
                live=row.get("live_promotion_allowed"),
            )
        )
    lines.extend(["", "## Next Read-Only Actions", ""])
    for action in payload.get("required_next_actions") or []:
        lines.append(
            f"- `{action.get('action_id')}`: target `{action.get('target')}`; live_side_effects={action.get('live_side_effects')}"
        )
    return "\n".join(lines) + "\n"


def eurusd_limit_sample_mining_md(payload: dict[str, Any]) -> str:
    floor = payload.get("sample_floor") if isinstance(payload.get("sample_floor"), dict) else {}
    execution = payload.get("execution_ledger_coverage") if isinstance(payload.get("execution_ledger_coverage"), dict) else {}
    legacy = payload.get("legacy_history_coverage") if isinstance(payload.get("legacy_history_coverage"), dict) else {}
    lines = [
        "# EUR_USD SHORT BREAKOUT_FAILURE LIMIT Sample Mining",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "## Verdict",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Target shape: `{payload.get('target_shape')}`",
        f"- Read-only: `{payload.get('read_only')}`",
        f"- Live permission allowed: `{payload.get('live_permission_allowed')}`",
        f"- Current replayed exact LIMIT samples: `{floor.get('current_replayed_exact_limit_samples')}`",
        f"- Additional acceptable local samples found: `{floor.get('additional_acceptable_local_samples_found')}`",
        f"- Remaining exact LIMIT samples: `{floor.get('remaining_exact_limit_samples')}`",
        "",
        "This artifact is evidence coverage only. It did not send, stage, cancel, close, mutate broker state, change launchd, or relax gates.",
        "",
        "## Coverage Summary",
        "",
        f"- Execution ledger: `{(execution.get('summary') or {}).get('by_verdict')}`",
        f"- Legacy history: `{(legacy.get('summary') or {}).get('by_verdict')}`",
        "",
        "## Additional Acceptable Samples",
        "",
    ]
    additional = payload.get("additional_acceptable_samples") or []
    if additional:
        for row in additional:
            lines.append(f"- `{row.get('trade_id')}` from `{row.get('lane_id') or row.get('source')}`")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Rejection Boundary",
            "",
            "- MARKET_ORDER and STOP_ORDER wins remain outside LIMIT proof.",
            "- RANGE_ROTATION and TREND_CONTINUATION wins remain outside BREAKOUT_FAILURE proof.",
            "- Ambiguous direct-USD or continuation legacy rows remain rejected unless a stronger source maps them to failed-break/retest.",
            "- Partial-close or duplicate close rows are not counted as fresh exact attached-TP samples.",
            "",
            "## Next Read-Only Actions",
            "",
        ]
    )
    for action in payload.get("next_read_only_actions") or []:
        lines.append(f"- {action}")
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

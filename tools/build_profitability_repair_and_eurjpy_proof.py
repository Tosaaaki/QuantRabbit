#!/usr/bin/env python3
"""Build profitability repair plans and EUR_JPY SHORT proof artifacts.

This tool is read-only with respect to broker state. It reads local artifacts,
the local execution ledger, and local/live replay reports; it writes evidence
JSON and Markdown reports for operator review.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = Path("/Users/tossaki/App/QuantRabbit-live")

USD_JPY_TARGET_LANE = "failure_trader:USD_JPY:LONG:BREAKOUT_FAILURE:LIMIT"
EUR_USD_LEAK_LANE = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"
EUR_USD_MANUAL_TRADE_ID = "472987"

S5_PROBE_DIR = Path("logs/reports/forecast_improvement/eurjpy_short_local_tp_probe")


def main() -> int:
    generated_at = _now()
    artifacts = _build_all(generated_at)
    for path, payload in artifacts["json"].items():
        _write_json(path, payload)
        print(f"wrote {path}")
    for path, text in artifacts["markdown"].items():
        _write_text(path, text)
        print(f"wrote {path}")
    return 0


def _build_all(generated_at: str) -> dict[str, dict[Path, Any]]:
    acceptance = _load_json(Path("data/profitability_acceptance.json"))
    capture = _load_json(Path("data/capture_economics.json"))
    profit_capture = _load_json(Path("data/profit_capture_bot.json"))
    timing = _load_json(Path("data/execution_timing_audit.json"))
    order_intents = _load_json(Path("data/order_intents.json"))
    old_board = _load_json(Path("data/as_lane_candidate_board.json"))
    usdjpy = _load_json(Path("data/usdjpy_long_breakout_failure_tp_proof.json"))
    daily = _load_json(Path("data/daily_target_state.json"))
    broker = _load_json(Path("data/broker_snapshot.json"))
    support = _load_json(Path("data/trader_support_bot.json"))
    repair_orchestrator = _load_json(Path("data/trader_repair_orchestrator.json"))
    bidask_rules = _load_json(Path("src/quant_rabbit/bidask_replay_precision_rules.json"))
    s5_report_path = _latest(S5_PROBE_DIR / "oanda_history_replay_validate_*.json")
    miner_report_path = _latest(S5_PROBE_DIR / "oanda_universal_rotation_mining_*.json")
    s5_report = _load_json(s5_report_path)
    miner_report = _load_json(miner_report_path)

    blocker_decomposition = _build_blocker_decomposition(
        acceptance=acceptance,
        capture=capture,
        profit_capture=profit_capture,
        timing=timing,
        bidask_rules=bidask_rules,
    )
    market_close = _build_market_close_plan(
        generated_at=generated_at,
        acceptance=acceptance,
        capture=capture,
        broker=broker,
    )
    tp_replay = _build_tp_replay_plan(
        generated_at=generated_at,
        acceptance=acceptance,
        profit_capture=profit_capture,
        timing=timing,
        broker=broker,
    )
    eurjpy_proof = _build_eurjpy_proof(
        generated_at=generated_at,
        s5_report_path=s5_report_path,
        miner_report_path=miner_report_path,
        s5_report=s5_report,
        miner_report=miner_report,
        order_intents=order_intents,
        bidask_rules=bidask_rules,
    )
    board = _build_board(
        generated_at=generated_at,
        order_intents=order_intents,
        old_board=old_board,
        usdjpy=usdjpy,
        eurjpy_proof=eurjpy_proof,
        acceptance=acceptance,
        capture=capture,
        daily=daily,
        broker=broker,
        support=support,
        repair_orchestrator=repair_orchestrator,
        blocker_decomposition=blocker_decomposition,
    )

    return {
        "json": {
            Path("data/market_close_leak_repair_plan.json"): market_close,
            Path("data/tp_progress_replay_repair_plan.json"): tp_replay,
            Path("data/eurjpy_short_local_tp_proof.json"): eurjpy_proof,
            Path("data/as_lane_candidate_board.json"): board,
        },
        "markdown": {
            Path("docs/market_close_leak_repair_plan.md"): _market_close_md(market_close),
            Path("docs/tp_progress_replay_repair_plan.md"): _tp_replay_md(tp_replay),
            Path("docs/eurjpy_short_local_tp_proof_report.md"): _eurjpy_md(eurjpy_proof),
            Path("docs/as_lane_candidate_board.md"): _board_md(board),
        },
    }


def _build_blocker_decomposition(
    *,
    acceptance: dict[str, Any],
    capture: dict[str, Any],
    profit_capture: dict[str, Any],
    timing: dict[str, Any],
    bidask_rules: dict[str, Any],
) -> list[dict[str, Any]]:
    findings = _findings_by_code(acceptance)
    capture_overall = capture.get("overall") if isinstance(capture.get("overall"), dict) else {}
    repair_summary = capture.get("repair_summary") if isinstance(capture.get("repair_summary"), dict) else {}
    market_segment = _market_close_segment(acceptance, capture)
    month_ev = _evidence(findings.get("MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"))
    bidask_ev = _evidence(findings.get("BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN"))
    adoption = bidask_ev.get("adoption_summary") or bidask_rules.get("adoption_summary") or {}

    return [
        {
            "code": "NEGATIVE_EXPECTANCY_ACTIVE",
            "current_evidence": {
                "status": capture.get("status"),
                "trades": capture_overall.get("trades"),
                "expectancy_jpy_per_trade": capture_overall.get("expectancy_jpy_per_trade"),
                "net_jpy": capture_overall.get("net_jpy"),
                "win_rate": capture_overall.get("win_rate"),
                "payoff_ratio": capture_overall.get("payoff_ratio"),
                "dominant_loss_exit_reason": repair_summary.get("dominant_loss_exit_reason"),
                "dominant_loss_exit_net_jpy": repair_summary.get("dominant_loss_exit_net_jpy"),
                "strongest_positive_exit_reason": repair_summary.get("strongest_positive_exit_reason"),
                "strongest_positive_exit_net_jpy": repair_summary.get("strongest_positive_exit_net_jpy"),
            },
            "pair_strategy_families": _top_capture_families(capture),
            "attribution": "SYSTEM_LEDGER_AFTER_OPERATOR_MANUAL_EXCLUSION",
            "clearing_condition": "capture_economics must be non-negative, or fresh entries must be restricted to exact attached-TP HARVEST repair lanes with local proof.",
            "required_code_or_evidence_change": "Repair market-close leakage, entry-quality residuals, and local TP proof; then rerun capture-economics and profitability-acceptance.",
            "fresh_entries_must_remain_blocked": True,
        },
        {
            "code": "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
            "current_evidence": {
                "present": "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE" in findings,
                "segment": market_segment,
            },
            "pair_strategy_families": ["EUR_USD LONG BREAKOUT_FAILURE"],
            "attribution": "SYSTEM_GATEWAY_ATTRIBUTED_ONLY; operator manual rows excluded",
            "clearing_condition": "No TP-proven segment remains net-damaged by MARKET_ORDER_TRADE_CLOSE leakage.",
            "required_code_or_evidence_change": "Ban or constrain loss-side system market closes without durable close-gate evidence; preserve attached TP/HARVEST exits.",
            "fresh_entries_must_remain_blocked": True,
        },
        {
            "code": "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
            "current_evidence": {
                "present_in_current_profitability_acceptance": "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK" in findings,
                "stale_or_runtime_mentions_exist": True,
                "note": "Current data/profitability_acceptance.json does not raise this 7-day P0, while stale/runtime sidecars still mention it.",
                "execution_timing_status": timing.get("status"),
            },
            "pair_strategy_families": ["recent gateway loss-side market closes when re-raised"],
            "attribution": "CURRENT_ACCEPTANCE_ABSENT; stale mentions are not permission evidence",
            "clearing_condition": "If re-raised, recent loss-side gateway market closes must be zero or have contained-risk timing plus durable close-gate proof.",
            "required_code_or_evidence_change": "Refresh acceptance after any changed timing/capture input; do not synthesize contained-risk proof.",
            "fresh_entries_must_remain_blocked": True,
        },
        {
            "code": "HISTORICAL_PROFIT_CAPTURE_MISSED",
            "current_evidence": {
                "status": profit_capture.get("status"),
                "metrics": profit_capture.get("metrics"),
                "top_blocker_codes": [item.get("code") for item in profit_capture.get("blockers") or [] if isinstance(item, dict)],
            },
            "pair_strategy_families": _top_profit_capture_families(profit_capture),
            "attribution": "SYSTEM_EXIT_MANAGEMENT_HISTORY; operator manual EUR_USD is excluded",
            "clearing_condition": "Post-repair live evidence stays clean and the 744h replay residual becomes non-negative or ages out.",
            "required_code_or_evidence_change": "Keep TP-progress production-gate evidence, rerun profit-capture-bot and execution-timing-audit after changed input.",
            "fresh_entries_must_remain_blocked": True,
        },
        {
            "code": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
            "current_evidence": {
                key: month_ev.get(key)
                for key in (
                    "window_lookback_hours",
                    "loss_closes_profit_capture_missed",
                    "loss_closes_repair_replay_triggered",
                    "counterfactual_profit_capture_delta_jpy",
                    "counterfactual_profit_capture_jpy",
                    "repair_replay_counterfactual_pl_jpy",
                )
            },
            "pair_strategy_families": _compact_residual_groups(month_ev.get("top_repair_replay_residual_groups") or [], limit=10),
            "attribution": "SYSTEM_REPLAY_RESIDUALS; matching pair/side/method groups only are executable blockers",
            "clearing_condition": "Rerun 744h execution-timing-audit and profitability-acceptance until replay is non-negative or matching residual groups disappear.",
            "required_code_or_evidence_change": "Repair entry-quality, close-gate, and TP-progress filters for listed residual groups; prove via replay before live permission.",
            "fresh_entries_must_remain_blocked": True,
        },
        {
            "code": "BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN",
            "current_evidence": {
                "adoption_summary": adoption,
                "forecast_sample_coverage_summary": {
                    key: (bidask_ev.get("forecast_sample_coverage_summary") or {}).get(key)
                    for key in ("pair_count", "min_directional_samples_for_precision_rule", "min_active_days_for_daily_stability")
                },
                "price_truth_coverage_status": (bidask_ev.get("price_truth_coverage") or {}).get("status"),
            },
            "pair_strategy_families": ["all-currency high-turn readiness coverage; EUR_JPY has negative direction rule despite price truth"],
            "attribution": "REPLAY_EVIDENCE_COVERAGE; not operator/manual",
            "clearing_condition": "All-currency or target-lane S5 bid/ask evidence must meet sample, active-day, and daily-stability floors for the exact shape.",
            "required_code_or_evidence_change": "Collect more forecast/history samples or produce exact pair/side/shape TP proof; package only with raw audit present.",
            "fresh_entries_must_remain_blocked": True,
        },
    ]


def _build_market_close_plan(
    *,
    generated_at: str,
    acceptance: dict[str, Any],
    capture: dict[str, Any],
    broker: dict[str, Any],
) -> dict[str, Any]:
    segment = _market_close_segment(acceptance, capture)
    ledger_trades = _ledger_market_close_leak_trades()
    published_ids = list(segment.get("market_close_loss_trade_ids") or [])
    ledger_ids = [item["trade_id"] for item in ledger_trades]
    manual = _manual_position(broker)
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_evidence",
        "source_artifacts": [
            "data/profitability_acceptance.json",
            "data/capture_economics.json",
            "data/execution_ledger.db",
            "data/broker_snapshot.json",
        ],
        "blocker": "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
        "segment": segment,
        "artifact_count_check": {
            "capture_market_close_losses": segment.get("market_close_losses"),
            "published_loss_trade_ids": published_ids,
            "published_loss_trade_id_count": len(published_ids),
            "ledger_reconciled_loss_trade_ids": ledger_ids,
            "ledger_reconciled_loss_trade_id_count": len(ledger_ids),
            "ledger_extra_not_in_published_examples": sorted(set(ledger_ids) - set(published_ids)),
            "published_missing_from_ledger_reconcile": sorted(set(published_ids) - set(ledger_ids)),
            "interpretation": "Use the ledger list for trade-level repair, while preserving the acceptance artifact's published IDs as blocker evidence.",
        },
        "market_close_leak_trades": ledger_trades,
        "manual_position_guard": {
            "trade_id": EUR_USD_MANUAL_TRADE_ID,
            "present": bool(manual),
            "classification": _nested(manual, "operator_manual_position", "classification"),
            "management_intent": _nested(manual, "operator_manual_position", "management_intent"),
            "system_pl_counted": _nested(manual, "operator_manual_position", "system_pl_counted"),
            "auto_sl_attach_allowed": _nested(manual, "operator_manual_position", "auto_sl_attach_allowed"),
            "auto_tp_modify_allowed": _nested(manual, "operator_manual_position", "auto_tp_modify_allowed"),
            "same_theme_auto_add_allowed": _nested(manual, "operator_manual_position", "same_theme_auto_add_allowed"),
        },
        "repair_plan": {
            "banned_or_constrained_close_path": [
                "Ban loss-side SYSTEM_GATEWAY MARKET_ORDER_TRADE_CLOSE on TP-proven HARVEST lanes unless a durable close-gate packet proves thesis invalidation and contained risk.",
                "Do not use operator/manual positions as system close evidence or system P/L repair material.",
                "Do not replace attached broker TP with discretionary market-close leakage on the same lane.",
            ],
            "evidence_required_before_market_close_allowed": [
                "fresh broker quote and spread snapshot at close decision time",
                "position owner and lane provenance linking the close request to a system-owned trade",
                "hard close-gate evidence: thesis invalidation, risk containment, and no same-direction support conflict",
                "execution-timing post-close replay showing market close preserves edge versus attached TP/HARVEST",
                "ledger receipt tying ORDER_ACCEPTED TRADE_CLOSE to the original system lane",
            ],
            "prevent_tp_edge_from_being_dominated_by_close_leakage": [
                "Prefer attached broker TAKE_PROFIT_ORDER and TP-progress profit-capture paths for proven shapes.",
                "Reject fresh entries for a lane if its historical market-close loss net exceeds its TP edge.",
                "Keep market-close loss examples counted against system edge until capture_economics and profitability_acceptance clear.",
                "Require a fresh replay/capture packet before increasing exposure on EUR_USD LONG BREAKOUT_FAILURE.",
            ],
            "clearing_conditions": [
                "capture_economics no longer reports MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
                "EUR_USD LONG BREAKOUT_FAILURE market_close_net_jpy no longer dominates TAKE_PROFIT_ORDER edge",
                "execution_timing_audit shows close-gate proof or no loss-side market-close leakage for matching system lanes",
            ],
        },
        "fresh_entries_must_remain_blocked": True,
        "live_side_effects": [],
    }


def _build_tp_replay_plan(
    *,
    generated_at: str,
    acceptance: dict[str, Any],
    profit_capture: dict[str, Any],
    timing: dict[str, Any],
    broker: dict[str, Any],
) -> dict[str, Any]:
    findings = _findings_by_code(acceptance)
    ev = _evidence(findings.get("MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"))
    residuals = ev.get("top_repair_replay_residual_groups") or []
    method_rollups = ev.get("top_repair_replay_residual_method_rollups") or []
    manual = _manual_position(broker)
    required_delta = abs(float(ev.get("repair_replay_counterfactual_pl_jpy") or 0.0))
    top_residual_sum = sum(abs(float(item.get("repair_replay_pl_jpy") or 0.0)) for item in residuals)
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_evidence",
        "source_artifacts": [
            "data/profitability_acceptance.json",
            "data/profit_capture_bot.json",
            "data/execution_timing_audit.json",
            "data/broker_snapshot.json",
        ],
        "blocker": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
        "current_evidence": {
            key: ev.get(key)
            for key in (
                "window_lookback_hours",
                "loss_closes_profit_capture_missed",
                "loss_closes_repair_replay_triggered",
                "counterfactual_profit_capture_delta_jpy",
                "counterfactual_profit_capture_jpy",
                "repair_replay_counterfactual_pl_jpy",
                "active_counterfactual_profit_capture_pl_jpy",
                "raw_counterfactual_profit_capture_pl_jpy",
            )
        },
        "profit_capture_context": {
            "status": profit_capture.get("status"),
            "metrics": profit_capture.get("metrics"),
            "top_misses": _limit(profit_capture.get("top_misses"), 8),
            "top_repair_replay_triggers": _limit(profit_capture.get("top_repair_replay_triggers"), 8),
        },
        "execution_timing_context": {
            "status": timing.get("status"),
            "summary": timing.get("summary"),
        },
        "losing_residual_groups": [_diagnose_residual(item) for item in residuals],
        "method_rollups": method_rollups,
        "manual_eurusd_exclusion": {
            "trade_id": EUR_USD_MANUAL_TRADE_ID,
            "present": bool(manual),
            "classification": _nested(manual, "operator_manual_position", "classification"),
            "management_intent": _nested(manual, "operator_manual_position", "management_intent"),
            "excluded_from_system_pl": _nested(manual, "operator_manual_position", "system_pl_counted") is False,
        },
        "filter_to_make_replay_non_negative": {
            "required_replay_pl_improvement_jpy": round(required_delta, 4),
            "top_residual_abs_loss_jpy": round(top_residual_sum, 4),
            "filter": "Block or repair matching pair/side/method residual groups with BELOW_TP_PROGRESS_GATE, NO_PROFIT_CANDIDATE, or BELOW_NOISE_FLOOR until the 744h replay is non-negative.",
            "not_a_broad_method_ban": True,
            "manual_eurusd_must_stay_excluded": True,
        },
        "proof_required_before_live_permission": [
            "rerun execution-timing-audit --lookback-hours 744 --post-close-hours 6",
            "rerun profitability-acceptance and confirm MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE cleared or matching residual group disappeared",
            "show exact local TP proof for the candidate pair/side/method/exit shape",
            "show close-gate evidence for any loss-side market close path",
            "show RiskEngine, LiveOrderGateway, and fresh GPT-5.5 TRADE/ADD receipt after evidence clears",
        ],
        "fresh_entries_must_remain_blocked": True,
        "live_side_effects": [],
    }


def _build_eurjpy_proof(
    *,
    generated_at: str,
    s5_report_path: Path,
    miner_report_path: Path,
    s5_report: dict[str, Any],
    miner_report: dict[str, Any],
    order_intents: dict[str, Any],
    bidask_rules: dict[str, Any],
) -> dict[str, Any]:
    intents = _eurjpy_short_intents(order_intents)
    s5_direction = _s5_down_summary(s5_report)
    packaged_negative = _eurjpy_packaged_negative_rule(bidask_rules)
    candidates = _eurjpy_candidates(s5_report, miner_report, packaged_negative)
    classification = "REJECTED_NEGATIVE_EXPECTANCY"
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_local_tp_proof",
        "classification": classification,
        "as_candidate": False,
        "live_ready_allowed": False,
        "source_artifacts": [
            str(s5_report_path),
            str(miner_report_path),
            "data/order_intents.json",
            "src/quant_rabbit/bidask_replay_precision_rules.json",
        ],
        "fresh_s5_bidask_probe": {
            "generated_at_utc": s5_report.get("generated_at_utc"),
            "summary": s5_report.get("summary"),
            "eurjpy_down": s5_direction,
            "interpretation": "Fresh current EUR_JPY S5 proof is under the 30-sample floor; positive TP/SL variants are not permission evidence.",
        },
        "packaged_bidask_negative_context": packaged_negative,
        "current_order_intents": intents,
        "tested_candidates": candidates,
        "decision": {
            "can_become_as": False,
            "reason": "Broad EUR_JPY SHORT S5 and pair-shape evidence is negative; narrow positive confluences are post-hoc, audit-only, and below live-grade sample/Wilson/active-day requirements.",
            "missing_evidence": [
                "non-negative spread-included S5 bid/ask replay for the exact EUR_JPY SHORT lane shape",
                "sufficient validation samples and active days for the selected session/location bucket",
                "exact local TP scope for EUR_JPY|SHORT|strategy|TAKE_PROFIT_ORDER",
                "fresh order_intent emitted with units > 0 and all RiskEngine/LiveOrderGateway blockers cleared",
                "fresh GPT-5.5 TRADE/ADD receipt after proof and risk gates pass",
            ],
            "risk_engine_constraints_if_retested": [
                "attached broker TAKE_PROFIT_ORDER only",
                "LIMIT or explicitly proven entry type only; do not reuse proof across STOP/MARKET/LIMIT",
                "spread cap and stale-quote gate must pass at send time",
                "units must come from current loss/margin budgets, not from deposit-inflated raw NAV performance",
            ],
            "live_order_gateway_blockers": _common_live_blockers(intents),
        },
        "safety": _safety_packet(),
        "live_side_effects": [],
    }


def _build_board(
    *,
    generated_at: str,
    order_intents: dict[str, Any],
    old_board: dict[str, Any],
    usdjpy: dict[str, Any],
    eurjpy_proof: dict[str, Any],
    acceptance: dict[str, Any],
    capture: dict[str, Any],
    daily: dict[str, Any],
    broker: dict[str, Any],
    support: dict[str, Any],
    repair_orchestrator: dict[str, Any],
    blocker_decomposition: list[dict[str, Any]],
) -> dict[str, Any]:
    results = order_intents.get("results") if isinstance(order_intents.get("results"), list) else []
    live_ready = [item for item in results if item.get("status") == "LIVE_READY"]
    usdjpy_current = [item for item in results if (item.get("intent") or {}).get("pair") == "USD_JPY"]
    eurjpy_short = [item for item in results if (item.get("intent") or {}).get("pair") == "EUR_JPY" and (item.get("intent") or {}).get("side") == "SHORT"]
    manual = _manual_position(broker)
    target_lane_present = any(item.get("lane_id") == USD_JPY_TARGET_LANE for item in results)
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_board_rebuild",
        "source_artifacts": [
            "data/order_intents.json",
            "data/usdjpy_long_breakout_failure_tp_proof.json",
            "data/eurjpy_short_local_tp_proof.json",
            "data/profitability_acceptance.json",
            "data/capture_economics.json",
            "data/daily_target_state.json",
            "data/broker_snapshot.json",
        ],
        "order_intents_generated_at_utc": order_intents.get("generated_at_utc"),
        "previous_board_generated_at_utc": old_board.get("generated_at_utc"),
        "total_lanes": len(results),
        "live_ready_lanes": len(live_ready),
        "routing_allowed": False,
        "normal_routing_status": "BLOCKED",
        "usd_jpy_rejection_state": {
            "target_lane": USD_JPY_TARGET_LANE,
            "target_lane_present_in_current_order_intents": target_lane_present,
            "present_only_in_stale_board": bool(usdjpy.get("stale_candidate_shape")) and not target_lane_present,
            "as_allowed": False,
            "live_ready_allowed": False,
            "verdict": usdjpy.get("verdict"),
            "exact_blockers": [item.get("blocker") for item in usdjpy.get("remaining_clearing_conditions") or []],
            "stale_packaged_rule_excluded_from_permission": True,
            "usd_jpy_current_lanes_candidate_only": [_lane_summary(item) for item in usdjpy_current],
        },
        "eur_jpy_short_proof_result": {
            "classification": eurjpy_proof.get("classification"),
            "as_candidate": eurjpy_proof.get("as_candidate"),
            "live_ready_allowed": eurjpy_proof.get("live_ready_allowed"),
            "current_lanes": [_lane_summary(item) for item in eurjpy_short],
            "decision": eurjpy_proof.get("decision"),
        },
        "profitability_blockers": blocker_decomposition,
        "next_best_candidates": _next_best_candidates(repair_orchestrator, acceptance, eurjpy_proof),
        "shortest_path": {
            "status": "blocked_no_as_live_ready_lane",
            "steps": [
                "Keep USD_JPY stale packaged rule excluded; do not promote USD_JPY LONG without fresh exact TP proof.",
                "Repair EUR_USD LONG BREAKOUT_FAILURE market-close leakage or keep that lane out of fresh entry routing.",
                "Clear the 744h month-scale TP-progress replay residuals for matching pair/side/method groups.",
                "Find a pair/side/strategy with spread-included local TP proof, sufficient samples/days, and non-negative replay.",
                "Regenerate order_intents and require LIVE_READY plus RiskEngine, LiveOrderGateway, guardian/operator review, and GPT-5.5 TRADE/ADD receipt.",
            ],
        },
        "shortest_path_to_10pct_extension_day": {
            "status": "none_currently",
            "reason": "No A/S LIVE_READY lane exists and funding-adjusted 30d multiplier remains below expansion threshold.",
        },
        "funding_adjusted_30d_status": {
            "performance_basis": daily.get("performance_basis"),
            "current_30d_multiplier": daily.get("current_30d_multiplier"),
            "current_equity_raw": daily.get("current_equity_raw"),
            "funding_adjusted_equity": daily.get("funding_adjusted_equity"),
            "capital_flows_30d": daily.get("capital_flows_30d"),
            "capital_flow_count_30d": daily.get("capital_flow_count_30d"),
            "deposit_100000_jpy_excluded_from_performance": True,
        },
        "manual_eurusd_472987": {
            "present": bool(manual),
            "classification": _nested(manual, "operator_manual_position", "classification"),
            "management_intent": _nested(manual, "operator_manual_position", "management_intent"),
            "system_pl_counted": _nested(manual, "operator_manual_position", "system_pl_counted"),
            "auto_sl_attach_allowed": _nested(manual, "operator_manual_position", "auto_sl_attach_allowed"),
            "auto_tp_modify_allowed": _nested(manual, "operator_manual_position", "auto_tp_modify_allowed"),
            "same_theme_auto_add_allowed": _nested(manual, "operator_manual_position", "same_theme_auto_add_allowed"),
        },
        "safety": _safety_packet(),
        "live_side_effects": [],
        "capture_status": capture.get("status"),
        "support_status": support.get("status"),
    }


def _eurjpy_candidates(
    s5_report: dict[str, Any],
    miner_report: dict[str, Any],
    packaged_negative: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    s5_down = _s5_down_summary(s5_report)
    candidates.append(
        {
            "candidate": "S5 forecast-history EUR_JPY SHORT broad DOWN",
            "strategy": "GENERIC_SHORT",
            "side": "SHORT",
            "session_or_location_bucket": "current forecast_history DOWN; no strategy bucket",
            "tp_sl_shape": s5_down.get("best_exit"),
            "sample_count": s5_down.get("n"),
            "active_days": _nested(s5_down, "daily_stability", "active_days"),
            "avg_pips": _nested(s5_down, "best_exit", "avg_realized_pips"),
            "profit_factor": _nested(s5_down, "best_exit", "profit_factor"),
            "hit_rate": _nested(s5_down, "summary", "hit_rate"),
            "positive_day_rate": _nested(s5_down, "daily_stability", "positive_day_rate"),
            "max_adverse": {"metric": "avg_mae_pips", "value": _nested(s5_down, "summary", "avg_mae_pips")},
            "drawdown": {"metric": "worst_daily_realized_pips", "value": _nested(s5_down, "daily_stability", "worst_daily_realized_pips")},
            "evidence_sufficient": False,
            "not_overfit": False,
            "classification": "REJECTED_UNDER_SAMPLED",
            "reason": "n is below the precision-rule sample floor; broad final-pip/adverse metrics remain weak.",
        }
    )
    if packaged_negative:
        candidates.append(
            {
                "candidate": "Packaged/live S5 EUR_JPY DOWN negative rule",
                "strategy": "GENERIC_SHORT",
                "side": "SHORT",
                "session_or_location_bucket": "all packaged DOWN samples",
                "tp_sl_shape": {
                    "take_profit_pips": packaged_negative.get("optimized_take_profit_pips"),
                    "stop_loss_pips": packaged_negative.get("optimized_stop_loss_pips"),
                },
                "sample_count": packaged_negative.get("samples"),
                "active_days": packaged_negative.get("active_days"),
                "avg_pips": packaged_negative.get("optimized_avg_realized_pips"),
                "profit_factor": packaged_negative.get("optimized_profit_factor"),
                "hit_rate": packaged_negative.get("directional_hit_rate"),
                "positive_day_rate": packaged_negative.get("positive_day_rate"),
                "max_adverse": {"metric": "avg_mae_pips", "value": packaged_negative.get("avg_mae_pips")},
                "drawdown": {"metric": "worst_daily_realized_pips", "value": packaged_negative.get("worst_daily_realized_pips")},
                "evidence_sufficient": False,
                "not_overfit": True,
                "classification": "REJECTED_NEGATIVE_EXPECTANCY",
                "reason": "Large-sample packaged context blocks generic EUR_JPY SHORT live use.",
            }
        )
    candidates.extend(_shape_candidate_rows(miner_report))
    return candidates


def _shape_candidate_rows(miner_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wanted = {
        "trend_continuation": "TREND_CONTINUATION SHORT",
        "range_reversion": "RANGE_ROTATION SHORT",
        "failed_break_fade": "BREAKOUT_FAILURE SHORT",
        "range_reclaim": "failed acceptance / major-figure fade proxy",
        "exhaustion_chase": "major-figure exhaustion fade proxy",
    }
    pair_shapes = [
        item
        for item in miner_report.get("top_pair_shapes") or []
        if item.get("pair") == "EUR_JPY" and item.get("side") == "SHORT"
    ]
    best_by_shape: dict[str, dict[str, Any]] = {}
    for item in pair_shapes:
        shape = str(item.get("shape") or "")
        if shape not in wanted:
            continue
        current = best_by_shape.get(shape)
        if current is None or float(item.get("validation_avg_realized_pips") or -9999) > float(current.get("validation_avg_realized_pips") or -9999):
            best_by_shape[shape] = item
    for shape, label in wanted.items():
        item = best_by_shape.get(shape)
        if item is None:
            rows.append(
                {
                    "candidate": label,
                    "strategy": label.split()[0],
                    "side": "SHORT",
                    "session_or_location_bucket": "not available in focused miner output",
                    "evidence_sufficient": False,
                    "classification": "EVIDENCE_GAP",
                    "reason": "No machine-readable candidate row for this shape.",
                }
            )
            continue
        rows.append(_miner_candidate(label, item, broad=True))

    selectors = [
        item
        for item in miner_report.get("qualified_directional_selectors") or []
        if item.get("pair") == "EUR_JPY" and item.get("selected_side") == "SHORT"
    ]
    for item in selectors[:2]:
        rows.append(_miner_candidate("session-specific RANGE_ROTATION SHORT", item, broad=False))
    queue = [
        item
        for item in miner_report.get("live_grade_evidence_queue") or []
        if item.get("pair") == "EUR_JPY" and item.get("side") == "SHORT"
    ]
    if queue:
        rows.append(_miner_candidate("live-grade evidence queue best EUR_JPY SHORT", queue[0], broad=False, queue=True))
    return rows


def _miner_candidate(label: str, item: dict[str, Any], *, broad: bool, queue: bool = False) -> dict[str, Any]:
    validation_days = item.get("validation_days_tail") if isinstance(item.get("validation_days_tail"), list) else []
    worst_day = None
    if validation_days:
        worst_day = min((day.get("avg_realized_atr") for day in validation_days if day.get("avg_realized_atr") is not None), default=None)
    blockers = item.get("blockers") or item.get("live_grade_gap_reasons") or []
    avg = item.get("validation_avg_realized_pips")
    pf = item.get("validation_profit_factor")
    n = item.get("validation_n")
    positive = bool(avg is not None and avg > 0 and pf is not None and pf > 1)
    sufficient = bool(positive and not blockers and broad is False and n and n >= 30)
    classification = "REJECTED_UNDER_SAMPLED" if positive else "REJECTED_NEGATIVE_EXPECTANCY"
    if broad and positive:
        classification = "EVIDENCE_GAP"
    if queue:
        classification = "REJECTED_UNDER_SAMPLED"
    return {
        "candidate": label,
        "strategy": item.get("shape"),
        "side": item.get("side") or item.get("selected_side"),
        "session_or_location_bucket": item.get("confluence") or item.get("feature") or "all pair-shape samples",
        "tp_sl_shape": item.get("exit_shape"),
        "sample_count": n,
        "active_days": item.get("active_days"),
        "avg_pips": avg,
        "profit_factor": pf,
        "hit_rate": item.get("validation_win_rate"),
        "positive_day_rate": item.get("positive_day_rate"),
        "max_adverse": {"metric": "not_reported_by_shape_miner", "value": None, "proxy": {"validation_stop_first_rate": item.get("validation_stop_first_rate")}},
        "drawdown": {"metric": "worst_validation_day_avg_atr", "value": worst_day},
        "evidence_sufficient": sufficient,
        "not_overfit": bool(broad and item.get("qualification") != "PASS"),
        "blockers": blockers,
        "classification": classification,
        "reason": _candidate_reason(item, positive=positive, broad=broad, queue=queue),
    }


def _candidate_reason(item: dict[str, Any], *, positive: bool, broad: bool, queue: bool) -> str:
    if queue:
        return "Miner marks this as evidence collection only; live_permission is false and validation/active-day/Wilson gaps remain."
    if broad and not positive:
        return "Broad pair-shape validation is negative after spread and stop-first ambiguity."
    if positive:
        return "Positive only after session/confluence selection; sample count and live-grade requirements are insufficient for A/S permission."
    return "Validation expectancy or profit factor is below live threshold."


def _s5_down_summary(report: dict[str, Any]) -> dict[str, Any]:
    grids = report.get("segment_exit_grids") if isinstance(report.get("segment_exit_grids"), dict) else {}
    rows = grids.get("by_pair_direction") if isinstance(grids.get("by_pair_direction"), list) else []
    for row in rows:
        if row.get("pair") == "EUR_JPY" and row.get("direction") == "DOWN":
            return row
    segments = report.get("segments") if isinstance(report.get("segments"), dict) else {}
    for row in segments.get("by_pair_direction") or []:
        if row.get("pair") == "EUR_JPY" and row.get("direction") == "DOWN":
            return row
    return {}


def _eurjpy_packaged_negative_rule(rules: dict[str, Any]) -> dict[str, Any] | None:
    for item in rules.get("negative_rules") or []:
        if item.get("pair") == "EUR_JPY" and item.get("direction") == "DOWN":
            return item
    return None


def _eurjpy_short_intents(order_intents: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in order_intents.get("results") or []:
        intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
        if intent.get("pair") == "EUR_JPY" and intent.get("side") == "SHORT":
            rows.append(_lane_summary(item))
    return rows


def _common_live_blockers(intents: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    for item in intents:
        for code in item.get("live_blocker_codes") or []:
            seen.add(str(code))
    return sorted(seen)


def _market_close_segment(acceptance: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    capture_segment: dict[str, Any] = {}
    priorities = capture.get("segment_repair_priorities") if isinstance(capture.get("segment_repair_priorities"), dict) else {}
    for segment in priorities.get("items") or []:
        if segment.get("pair") == "EUR_USD" and segment.get("side") == "LONG" and segment.get("method") == "BREAKOUT_FAILURE":
            capture_segment = segment
            break
    findings = _findings_by_code(acceptance)
    for segment in (_evidence(findings.get("MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE")).get("segments") or []):
        if segment.get("pair") == "EUR_USD" and segment.get("side") == "LONG" and segment.get("method") == "BREAKOUT_FAILURE":
            merged = dict(capture_segment)
            merged.update(segment)
            return merged
    return capture_segment


def _ledger_market_close_leak_trades() -> list[dict[str, Any]]:
    path = ROOT / "data/execution_ledger.db"
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows: list[dict[str, Any]] = []
    query = """
        SELECT *
        FROM execution_events
        WHERE event_type = 'GATEWAY_TRADE_CLOSE_RECONCILED'
          AND lane_id = ?
        ORDER BY ts_utc
    """
    for rec in conn.execute(query, (EUR_USD_LEAK_LANE,)):
        raw = _json_or_empty(rec["raw_json"])
        close_tx = _close_tx(raw)
        close = conn.execute("SELECT * FROM execution_events WHERE oanda_transaction_id = ?", (close_tx,)).fetchone() if close_tx else None
        fill = conn.execute(
            "SELECT * FROM execution_events WHERE event_type = 'ORDER_FILLED' AND trade_id = ? ORDER BY ts_utc LIMIT 1",
            (rec["trade_id"],),
        ).fetchone()
        if close is None or close["realized_pl_jpy"] is None or float(close["realized_pl_jpy"]) >= 0:
            continue
        rows.append(
            {
                "trade_id": str(rec["trade_id"]),
                "pair": "EUR_USD",
                "side": "LONG",
                "strategy": "BREAKOUT_FAILURE",
                "lane_id": EUR_USD_LEAK_LANE,
                "entry_time_utc": fill["ts_utc"] if fill else None,
                "entry_price": fill["price"] if fill else None,
                "entry_units": fill["units"] if fill else None,
                "exit_time_utc": close["ts_utc"],
                "exit_price": close["price"],
                "realized_pl_jpy": close["realized_pl_jpy"],
                "close_reason": close["exit_reason"],
                "close_transaction_id": close_tx,
                "system_or_operator": "SYSTEM_GATEWAY",
                "gateway_reconciled": True,
                "campaign_exposure_recovery": "NO_CAMPAIGN_EXPOSURE_RECOVERY_EVIDENCE_IN_LEDGER",
                "should_be_excluded_or_counted": "COUNT_AGAINST_SYSTEM_EDGE",
                "operator_manual_excluded": True,
            }
        )
    conn.close()
    return rows


def _diagnose_residual(item: dict[str, Any]) -> dict[str, Any]:
    reasons = item.get("block_reasons") if isinstance(item.get("block_reasons"), dict) else {}
    if "NO_PROFIT_CANDIDATE" in reasons:
        diagnosis = "bad_entry_or_no_positive_excursion"
    elif "BELOW_TP_PROGRESS_GATE" in reasons:
        diagnosis = "entry_quality_or_premature_exit_below_tp_progress_gate"
    elif "BELOW_NOISE_FLOOR" in reasons:
        diagnosis = "missed_capture_but_below_noise_floor"
    else:
        diagnosis = "residual_needs_manual_review"
    return {
        "pair": item.get("pair"),
        "strategy": item.get("method"),
        "side": item.get("side"),
        "exit_reason": item.get("exit_reason"),
        "loss_closes": item.get("loss_closes"),
        "actual_pl_jpy": item.get("actual_pl_jpy"),
        "repair_replay_pl_jpy": item.get("repair_replay_pl_jpy"),
        "repair_replay_triggered": item.get("repair_replay_triggered"),
        "block_reasons": reasons,
        "examples": item.get("examples"),
        "diagnosis": diagnosis,
        "live_permission_filter": "block matching pair/side/method until replay clears or exact TP proof overrides with all gates passed",
    }


def _lane_summary(item: dict[str, Any]) -> dict[str, Any]:
    intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
    metrics = item.get("risk_metrics") if isinstance(item.get("risk_metrics"), dict) else {}
    return {
        "lane_id": item.get("lane_id"),
        "pair": intent.get("pair"),
        "side": intent.get("side"),
        "order_type": intent.get("order_type"),
        "entry": intent.get("entry"),
        "tp": intent.get("tp"),
        "sl": intent.get("sl"),
        "units": intent.get("units"),
        "status": item.get("status"),
        "live_blocker_codes": item.get("live_blocker_codes"),
        "risk_allowed": item.get("risk_allowed"),
        "reward_risk": metrics.get("reward_risk"),
        "spread_pips": metrics.get("spread_pips"),
    }


def _next_best_candidates(
    repair_orchestrator: dict[str, Any],
    acceptance: dict[str, Any],
    eurjpy_proof: dict[str, Any],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    frontier = repair_orchestrator.get("execution_frontier") if isinstance(repair_orchestrator.get("execution_frontier"), dict) else {}
    if frontier:
        examples.append(
            {
                "name": "repair_frontier",
                "status": "blocked_repair_context_only",
                "evidence": frontier,
                "why_not_live": "repair frontier still inherits profitability and guardian/operator-review blockers",
            }
        )
    findings = _findings_by_code(acceptance)
    repair_ev = _evidence(findings.get("REPAIR_FRONTIER_BLOCKED"))
    if repair_ev:
        examples.append(
            {
                "name": "profitability_acceptance_repair_frontier",
                "status": "blocked",
                "evidence": repair_ev,
                "why_not_live": "REPAIR_FRONTIER_BLOCKED remains active",
            }
        )
    examples.append(
        {
            "name": "EUR_JPY SHORT local TP proof",
            "status": eurjpy_proof.get("classification"),
            "why_not_live": (eurjpy_proof.get("decision") or {}).get("reason"),
        }
    )
    examples.append(
        {
            "name": "USD_JPY LONG BREAKOUT_FAILURE LIMIT",
            "status": "rejected_current_replay_negative_undersampled",
            "why_not_live": "stale packaged rule excluded; fresh exact TP10/SL7 proof failed",
        }
    )
    return examples


def _top_capture_families(capture: dict[str, Any]) -> list[dict[str, Any]]:
    priorities = capture.get("segment_repair_priorities") if isinstance(capture.get("segment_repair_priorities"), dict) else {}
    rows = []
    for item in priorities.get("items") or []:
        rows.append(
            {
                "pair": item.get("pair"),
                "side": item.get("side"),
                "strategy": item.get("method"),
                "net_jpy": item.get("net_jpy"),
                "market_close_net_jpy": item.get("market_close_net_jpy"),
                "priority_class": item.get("priority_class"),
            }
        )
    return rows[:10]


def _top_profit_capture_families(profit_capture: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in profit_capture.get("top_misses") or []:
        rows.append(
            {
                "trade_id": item.get("trade_id"),
                "pair": item.get("pair"),
                "side": item.get("side"),
                "lane_id": item.get("lane_id"),
                "realized_pl_jpy": item.get("realized_pl_jpy"),
                "counterfactual_profit_capture_delta_jpy": item.get("counterfactual_profit_capture_delta_jpy"),
            }
        )
    return rows[:8]


def _compact_residual_groups(groups: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return [
        {
            "pair": item.get("pair"),
            "side": item.get("side"),
            "strategy": item.get("method"),
            "exit_reason": item.get("exit_reason"),
            "loss_closes": item.get("loss_closes"),
            "repair_replay_pl_jpy": item.get("repair_replay_pl_jpy"),
            "block_reasons": item.get("block_reasons"),
        }
        for item in groups[:limit]
    ]


def _market_close_md(plan: dict[str, Any]) -> str:
    rows = plan.get("market_close_leak_trades") or []
    lines = [
        "# Market Close Leak Repair Plan",
        "",
        f"- Generated: `{plan.get('generated_at_utc')}`",
        f"- Blocker: `{plan.get('blocker')}`",
        f"- Fresh entries blocked: `{plan.get('fresh_entries_must_remain_blocked')}`",
        "",
        "## Evidence",
        "",
        "| trade_id | pair | side | strategy | entry | exit | P/L JPY | close reason | attribution | campaign recovery | count/exclude |",
        "|---|---|---|---|---|---|---:|---|---|---|---|",
    ]
    for item in rows:
        lines.append(
            "| {trade_id} | {pair} | {side} | {strategy} | {entry_time_utc} @ {entry_price} | {exit_time_utc} @ {exit_price} | {realized_pl_jpy} | {close_reason} | {system_or_operator} | {campaign_exposure_recovery} | {should_be_excluded_or_counted} |".format(**item)
        )
    count_check = plan.get("artifact_count_check") or {}
    lines.extend(
        [
            "",
            "## Count Check",
            "",
            f"- Capture market-close losses: `{count_check.get('capture_market_close_losses')}`",
            f"- Published blocker IDs: `{', '.join(count_check.get('published_loss_trade_ids') or [])}`",
            f"- Ledger reconciled loss IDs: `{', '.join(count_check.get('ledger_reconciled_loss_trade_ids') or [])}`",
            f"- Ledger extra not in published examples: `{', '.join(count_check.get('ledger_extra_not_in_published_examples') or []) or 'none'}`",
            "",
            "## Repair",
            "",
        ]
    )
    for item in (plan.get("repair_plan") or {}).get("banned_or_constrained_close_path") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Required evidence before market close:")
    for item in (plan.get("repair_plan") or {}).get("evidence_required_before_market_close_allowed") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("TP-edge protection:")
    for item in (plan.get("repair_plan") or {}).get("prevent_tp_edge_from_being_dominated_by_close_leakage") or []:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _tp_replay_md(plan: dict[str, Any]) -> str:
    lines = [
        "# TP Progress Replay Repair Plan",
        "",
        f"- Generated: `{plan.get('generated_at_utc')}`",
        f"- Blocker: `{plan.get('blocker')}`",
        f"- Fresh entries blocked: `{plan.get('fresh_entries_must_remain_blocked')}`",
        "",
        "## Current Evidence",
        "",
    ]
    for key, value in (plan.get("current_evidence") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Residual Groups",
            "",
            "| pair | side | strategy | exit | diagnosis | replay P/L JPY | block reasons |",
            "|---|---|---|---|---|---:|---|",
        ]
    )
    for item in plan.get("losing_residual_groups") or []:
        lines.append(
            f"| {item.get('pair')} | {item.get('side')} | {item.get('strategy')} | {item.get('exit_reason')} | {item.get('diagnosis')} | {item.get('repair_replay_pl_jpy')} | {json.dumps(item.get('block_reasons'), sort_keys=True)} |"
        )
    lines.extend(
        [
            "",
            "## Clearing Filter",
            "",
            f"- Required replay improvement: `{(plan.get('filter_to_make_replay_non_negative') or {}).get('required_replay_pl_improvement_jpy')}` JPY",
            f"- Filter: {(plan.get('filter_to_make_replay_non_negative') or {}).get('filter')}",
            f"- Manual EUR_USD `{EUR_USD_MANUAL_TRADE_ID}` excluded: `{(plan.get('manual_eurusd_exclusion') or {}).get('excluded_from_system_pl')}`",
            "",
            "## Proof Before Live",
            "",
        ]
    )
    for item in plan.get("proof_required_before_live_permission") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _eurjpy_md(proof: dict[str, Any]) -> str:
    lines = [
        "# EUR_JPY SHORT Local TP Proof Report",
        "",
        f"- Generated: `{proof.get('generated_at_utc')}`",
        f"- Classification: `{proof.get('classification')}`",
        f"- A/S candidate: `{proof.get('as_candidate')}`",
        f"- LIVE_READY allowed: `{proof.get('live_ready_allowed')}`",
        "",
        "## Decision",
        "",
        (proof.get("decision") or {}).get("reason", ""),
        "",
        "## Candidates",
        "",
        "| candidate | bucket | n | days | avg pips | PF | hit | pos-day | adverse | drawdown | class | sufficient |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for item in proof.get("tested_candidates") or []:
        lines.append(
            f"| {item.get('candidate')} | {item.get('session_or_location_bucket')} | {item.get('sample_count')} | {item.get('active_days')} | {item.get('avg_pips')} | {item.get('profit_factor')} | {item.get('hit_rate')} | {item.get('positive_day_rate')} | {json.dumps(item.get('max_adverse'), sort_keys=True)} | {json.dumps(item.get('drawdown'), sort_keys=True)} | {item.get('classification')} | {item.get('evidence_sufficient')} |"
        )
    lines.extend(
        [
            "",
            "## Missing Evidence",
            "",
        ]
    )
    for item in (proof.get("decision") or {}).get("missing_evidence") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Safety",
            "",
        ]
    )
    for key, value in (proof.get("safety") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def _board_md(board: dict[str, Any]) -> str:
    lines = [
        "# A/S Lane Candidate Board",
        "",
        f"- Generated: `{board.get('generated_at_utc')}`",
        f"- Order intents generated: `{board.get('order_intents_generated_at_utc')}`",
        f"- Total lanes: `{board.get('total_lanes')}`",
        f"- LIVE_READY lanes: `{board.get('live_ready_lanes')}`",
        f"- Normal routing: `{board.get('normal_routing_status')}`",
        "",
        "## USD_JPY",
        "",
    ]
    usd = board.get("usd_jpy_rejection_state") or {}
    lines.extend(
        [
            f"- Target lane present in current order_intents: `{usd.get('target_lane_present_in_current_order_intents')}`",
            f"- Present only in stale board: `{usd.get('present_only_in_stale_board')}`",
            f"- A/S allowed: `{usd.get('as_allowed')}`",
            f"- LIVE_READY allowed: `{usd.get('live_ready_allowed')}`",
            f"- Stale packaged rule excluded: `{usd.get('stale_packaged_rule_excluded_from_permission')}`",
            f"- Blockers: `{', '.join(usd.get('exact_blockers') or [])}`",
            "",
            "## EUR_JPY SHORT",
            "",
        ]
    )
    eur = board.get("eur_jpy_short_proof_result") or {}
    lines.extend(
        [
            f"- Classification: `{eur.get('classification')}`",
            f"- A/S candidate: `{eur.get('as_candidate')}`",
            f"- LIVE_READY allowed: `{eur.get('live_ready_allowed')}`",
            f"- Reason: `{((eur.get('decision') or {}).get('reason'))}`",
            "",
            "## Profitability Blockers",
            "",
            "| blocker | attribution | clearing condition | fresh entries blocked |",
            "|---|---|---|---|",
        ]
    )
    for item in board.get("profitability_blockers") or []:
        lines.append(
            f"| `{item.get('code')}` | {item.get('attribution')} | {item.get('clearing_condition')} | `{item.get('fresh_entries_must_remain_blocked')}` |"
        )
    lines.extend(
        [
            "",
            "## Next Best Candidates",
            "",
        ]
    )
    for item in board.get("next_best_candidates") or []:
        lines.append(f"- `{item.get('name')}`: `{item.get('status')}` - {item.get('why_not_live')}")
    lines.extend(
        [
            "",
            "## Shortest Path",
            "",
            f"- Status: `{(board.get('shortest_path') or {}).get('status')}`",
        ]
    )
    for item in (board.get("shortest_path") or {}).get("steps") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Funding / Manual Safety",
            "",
            f"- Funding-adjusted 30d multiplier: `{(board.get('funding_adjusted_30d_status') or {}).get('current_30d_multiplier')}`",
            f"- 100,000 JPY deposit excluded from performance: `{(board.get('funding_adjusted_30d_status') or {}).get('deposit_100000_jpy_excluded_from_performance')}`",
            f"- EUR_USD `{EUR_USD_MANUAL_TRADE_ID}` classification: `{(board.get('manual_eurusd_472987') or {}).get('classification')}`",
            f"- EUR_USD `{EUR_USD_MANUAL_TRADE_ID}` intent: `{(board.get('manual_eurusd_472987') or {}).get('management_intent')}`",
            f"- System P/L counted: `{(board.get('manual_eurusd_472987') or {}).get('system_pl_counted')}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _findings_by_code(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("code")): item for item in data.get("findings") or [] if isinstance(item, dict)}


def _evidence(finding: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(finding, dict):
        return {}
    evidence = finding.get("evidence")
    return evidence if isinstance(evidence, dict) else {}


def _manual_position(broker: dict[str, Any]) -> dict[str, Any]:
    for position in broker.get("positions") or []:
        if str(position.get("trade_id")) == EUR_USD_MANUAL_TRADE_ID:
            return position
    return {}


def _nested(data: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = data or {}
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _old_board_has_lane(board: dict[str, Any], lane_id: str) -> bool:
    return any(item.get("lane_id") == lane_id for item in board.get("candidates") or [])


def _limit(value: Any, limit: int) -> Any:
    if isinstance(value, list):
        return value[:limit]
    return value


def _json_or_empty(raw: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _close_tx(raw: dict[str, Any]) -> str | None:
    value = raw.get("close_transaction_id") or raw.get("close_transactionID") or raw.get("close_oanda_transaction_id")
    if value:
        return str(value)
    uid = str(raw.get("close_event_uid") or "")
    parts = uid.split(":")
    if len(parts) > 1:
        return parts[1]
    return None


def _safety_packet() -> dict[str, Any]:
    return {
        "broker_state_modified": False,
        "orders_placed": False,
        "orders_cancelled": False,
        "positions_closed": False,
        "sl_tp_modified": False,
        "execution_flags_enabled": False,
        "normal_routing_remains_blocked": True,
        "raw_nav_usage": "sizing_and_risk_only",
        "performance_kpi": "funding_adjusted_equity",
        "deposit_100000_jpy_is_capital_flow_not_pl": True,
    }


def _latest(pattern: Path) -> Path:
    paths = sorted((ROOT / pattern.parent).glob(pattern.name))
    if paths:
        return paths[-1].relative_to(ROOT)
    live_paths = sorted((LIVE_ROOT / pattern.parent).glob(pattern.name))
    if live_paths:
        return live_paths[-1]
    raise FileNotFoundError(pattern)


def _load_json(path: Path) -> dict[str, Any]:
    actual = path if path.is_absolute() else ROOT / path
    if not actual.exists():
        return {}
    payload = json.loads(actual.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    actual = ROOT / path
    actual.parent.mkdir(parents=True, exist_ok=True)
    actual.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    actual = ROOT / path
    actual.parent.mkdir(parents=True, exist_ok=True)
    actual.write_text(text, encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

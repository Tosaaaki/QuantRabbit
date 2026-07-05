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
MARKET_CLOSE_LEAK_FAMILY_TRADE_IDS = (
    "470356",
    "470353",
    "470730",
    "471174",
    "471089",
    "471255",
    "472280",
)
MARKET_CLOSE_LEAK_FAMILY_MANUAL_EXCLUDED_TRADE_IDS = (EUR_USD_MANUAL_TRADE_ID,)
PRIORITY_RESIDUAL_FAMILY_KEYS = {
    ("GBP_USD", "LONG", "BREAKOUT_FAILURE"),
    ("AUD_USD", "LONG", "RANGE_ROTATION"),
    ("AUD_USD", "SHORT", "RANGE_ROTATION"),
    ("EUR_USD", "LONG", "RANGE_ROTATION"),
    ("EUR_USD", "SHORT", "RANGE_ROTATION"),
    ("NZD_CAD", "SHORT", "RANGE_ROTATION"),
    ("EUR_JPY", "LONG", "RANGE_ROTATION"),
}

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
    harvest_gate = _load_json(Path("data/tp_progress_harvest_gate_evidence.json"))
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
    market_close_table = _build_market_close_leak_trade_table(
        generated_at=generated_at,
        acceptance=acceptance,
        capture=capture,
        timing=timing,
        broker=broker,
    )
    month_residuals = _build_month_scale_tp_replay_residuals(
        generated_at=generated_at,
        acceptance=acceptance,
        timing=timing,
        broker=broker,
    )
    residual_family_table = _build_month_scale_residual_family_table(
        generated_at=generated_at,
        timing=timing,
        harvest_gate=harvest_gate,
        broker=broker,
    )
    historical_missed = _build_historical_profit_capture_missed_table(
        generated_at=generated_at,
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
        residual_family_table=residual_family_table,
    )

    return {
        "json": {
            Path("data/market_close_leak_repair_plan.json"): market_close,
            Path("data/market_close_leak_trade_table.json"): market_close_table,
            Path("data/tp_progress_replay_repair_plan.json"): tp_replay,
            Path("data/month_scale_tp_replay_residuals.json"): month_residuals,
            Path("data/month_scale_residual_family_table.json"): residual_family_table,
            Path("data/historical_profit_capture_missed_table.json"): historical_missed,
            Path("data/eurjpy_short_local_tp_proof.json"): eurjpy_proof,
            Path("data/as_lane_candidate_board.json"): board,
        },
        "markdown": {
            Path("docs/market_close_leak_repair_plan.md"): _market_close_md(market_close),
            Path("docs/market_close_leak_trade_table.md"): _market_close_trade_table_md(market_close_table),
            Path("docs/tp_progress_replay_repair_plan.md"): _tp_replay_md(tp_replay),
            Path("docs/month_scale_tp_replay_residuals.md"): _month_scale_tp_replay_residuals_md(month_residuals),
            Path("docs/month_scale_residual_family_table.md"): _month_scale_residual_family_table_md(residual_family_table),
            Path("docs/historical_profit_capture_missed_table.md"): _historical_profit_capture_missed_md(historical_missed),
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
    month_ev = _month_scale_replay_metrics(acceptance)
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
    ev = _month_scale_replay_metrics(acceptance)
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


def _build_market_close_leak_trade_table(
    *,
    generated_at: str,
    acceptance: dict[str, Any],
    capture: dict[str, Any],
    timing: dict[str, Any],
    broker: dict[str, Any],
) -> dict[str, Any]:
    findings = _findings_by_code(acceptance)
    segment = _market_close_segment(acceptance, capture)
    rows = []
    for item in _ledger_market_close_leak_trades():
        rows.append(
            {
                "trade_id": item.get("trade_id"),
                "pair": item.get("pair"),
                "side": item.get("side"),
                "strategy": item.get("strategy"),
                "lane_id": item.get("lane_id"),
                "entry_price": item.get("entry_price"),
                "entry_time_utc": item.get("entry_time_utc"),
                "close_price": item.get("exit_price"),
                "close_time_utc": item.get("exit_time_utc"),
                "realized_pl_jpy": item.get("realized_pl_jpy"),
                "close_reason": item.get("close_reason"),
                "system_gateway_or_operator_manual": item.get("system_or_operator"),
                "campaign_exposure_recovery": "no",
                "campaign_exposure_recovery_evidence": item.get("campaign_exposure_recovery"),
                "should_count_against_system_edge": True,
                "system_edge_count_reason": "system gateway close reconciled to a system lane; operator manual EUR_USD 472987 is excluded separately",
                "exact_trade_id_covered_by_regression_tests": False,
                "already_covered_by_regression_tests": "behavioral coverage only; this live trade id is not a fixture",
                "regression_test_coverage": [
                    "tests/test_profitability_acceptance_replay_repair.py",
                    "tests/test_capture_economics.py",
                    "tests/test_gpt_trader.py",
                ],
            }
        )
    manual = _manual_position(broker)
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_trade_decomposition",
        "source_artifacts": [
            "data/profitability_acceptance.json",
            "data/capture_economics.json",
            "data/execution_timing_audit.json",
            "data/execution_ledger.db",
            "data/broker_snapshot.json",
        ],
        "blockers": {
            "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE": {
                "present": "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE" in findings,
                "segment": segment,
                "contributing_trade_ids": [row.get("trade_id") for row in rows],
                "contributing_trade_count": len(rows),
            },
            "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK": {
                "present_in_current_profitability_acceptance": "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK" in findings,
                "contributing_trade_ids": [],
                "contributing_trade_count": 0,
                "status": "stale_or_runtime_mention_only_in_current_evidence",
                "execution_timing_status": timing.get("status"),
                "note": "Current profitability_acceptance does not raise this blocker; stale mentions remain non-permission evidence.",
            },
        },
        "manual_eurusd_guard": {
            "trade_id": EUR_USD_MANUAL_TRADE_ID,
            "present": bool(manual),
            "classification": _nested(manual, "operator_manual_position", "classification"),
            "management_intent": _nested(manual, "operator_manual_position", "management_intent"),
            "system_pl_counted": _nested(manual, "operator_manual_position", "system_pl_counted"),
            "system_occupancy_counted": False,
            "auto_close_allowed": False,
            "auto_sl_attach_allowed": _nested(manual, "operator_manual_position", "auto_sl_attach_allowed"),
            "auto_tp_modify_allowed": _nested(manual, "operator_manual_position", "auto_tp_modify_allowed"),
            "same_theme_auto_add_allowed": _nested(manual, "operator_manual_position", "same_theme_auto_add_allowed"),
        },
        "trades": rows,
        "mitigation_by_family": [
            {
                "family": "TP_PROVEN_BREAKOUT_FAILURE_SYSTEM_MARKET_CLOSE",
                "entry_or_close_path_must_be_banned": "loss-side SYSTEM_GATEWAY MARKET_ORDER_TRADE_CLOSE on TP-proven HARVEST/BREAKOUT_FAILURE lanes without durable close-gate evidence",
                "evidence_that_would_allow_again": [
                    "fresh quote/spread packet at close decision time",
                    "system-owned trade provenance and lane receipt",
                    "thesis invalidation plus contained-risk close-gate proof",
                    "744h execution timing replay showing the market-close path is non-negative versus attached TP/profit capture",
                    "profitability_acceptance clears MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
                ],
                "fix_type": ["CODE_FIX", "ROUTING_GATE", "REPLAY_FILTER"],
                "files_modules_to_update": [
                    "src/quant_rabbit/gpt_trader.py",
                    "src/quant_rabbit/automation.py",
                    "src/quant_rabbit/profitability_acceptance.py",
                    "src/quant_rabbit/capture_economics.py",
                    "src/quant_rabbit/trader_support_bot.py",
                    "src/quant_rabbit/strategy/trader_brain.py",
                ],
                "tests_required": [
                    "tests/test_gpt_trader.py",
                    "tests/test_profitability_acceptance_replay_repair.py",
                    "tests/test_capture_economics.py",
                    "tests/test_trader_brain.py",
                ],
            },
            {
                "family": "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
                "entry_or_close_path_must_be_banned": "recent loss-side gateway market-close path if refreshed acceptance re-raises the blocker",
                "evidence_that_would_allow_again": [
                    "recent loss-side gateway market closes are zero",
                    "or each loss close has contained-risk timing evidence plus durable close-gate receipt",
                    "profitability_acceptance refresh keeps the blocker absent",
                ],
                "fix_type": ["EVIDENCE_GAP", "ROUTING_GATE"],
                "files_modules_to_update": [
                    "src/quant_rabbit/profitability_acceptance.py",
                    "src/quant_rabbit/execution_timing_audit.py",
                    "src/quant_rabbit/trader_support_bot.py",
                ],
                "tests_required": [
                    "tests/test_profitability_acceptance_replay_repair.py",
                    "tests/test_execution_timing_audit.py",
                    "tests/test_trader_support_bot.py",
                ],
            },
            {
                "family": "OPERATOR_MANUAL_EXCLUSION",
                "entry_or_close_path_must_be_banned": "using operator/manual EUR_USD 472987 as system P/L, system occupancy, auto-close, SL attach, TP modification, or same-theme add evidence",
                "evidence_that_would_allow_again": [
                    "explicit operator reclassification and new proof packet; current artifact says OPERATOR_MANUAL / KEEP",
                ],
                "fix_type": ["MANUAL_EXCLUSION"],
                "files_modules_to_update": [
                    "src/quant_rabbit/broker_snapshot.py",
                    "src/quant_rabbit/trader_support_bot.py",
                    "src/quant_rabbit/position_manager.py",
                    "src/quant_rabbit/risk_engine.py",
                ],
                "tests_required": [
                    "tests/test_trader_support_bot.py",
                    "tests/test_position_manager.py",
                    "tests/test_risk_engine.py",
                ],
            },
        ],
        "gate_definitions": [
            {
                "blocker_code": "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
                "current_evidence": "EUR_USD LONG BREAKOUT_FAILURE has positive TP expectancy but negative system MARKET_ORDER_TRADE_CLOSE leakage; ledger rows enumerate the loss trades.",
                "exact_clearing_condition": "No TP-proven segment remains net-damaged by loss-side system MARKET_ORDER_TRADE_CLOSE, and refreshed acceptance no longer raises the blocker.",
                "artifact_that_proves_it": [
                    "data/capture_economics.json",
                    "data/profitability_acceptance.json",
                    "data/market_close_leak_trade_table.json",
                ],
                "verifier_riskengine_liveordergateway_implication": "Verifier must reject loss closes lacking durable close-gate proof; RiskEngine must keep matching fresh entries blocked; LiveOrderGateway must not execute loss-side market closes on this family without a fresh close receipt.",
                "can_create_as_permission": False,
            },
            {
                "blocker_code": "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
                "current_evidence": "Absent from current profitability_acceptance; stale/runtime mentions exist and do not grant permission.",
                "exact_clearing_condition": "Refresh acceptance keeps the blocker absent, or if re-raised, every recent loss-side gateway market close is removed or proven contained-risk.",
                "artifact_that_proves_it": [
                    "data/profitability_acceptance.json",
                    "data/execution_timing_audit.json",
                    "data/market_close_leak_trade_table.json",
                ],
                "verifier_riskengine_liveordergateway_implication": "If re-raised, normal routing stays blocked and close-path receipts must be reviewed before any matching lane can route.",
                "can_create_as_permission": False,
            },
        ],
        "fresh_entries_must_remain_blocked": True,
        "live_side_effects": [],
    }


def _build_month_scale_tp_replay_residuals(
    *,
    generated_at: str,
    acceptance: dict[str, Any],
    timing: dict[str, Any],
    broker: dict[str, Any],
) -> dict[str, Any]:
    ev = _month_scale_replay_metrics(acceptance)
    residuals = [_diagnose_residual(item) for item in ev.get("top_repair_replay_residual_groups") or []]
    current_residual = _float_or_none(ev.get("repair_replay_counterfactual_pl_jpy"))
    required = abs(current_residual or 0.0)
    known_abs = sum(abs(float(item.get("repair_replay_pl_jpy") or 0.0)) for item in residuals)
    manual = _manual_position(broker)
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_replay_residual_decomposition",
        "source_artifacts": [
            "data/profitability_acceptance.json",
            "data/execution_timing_audit.json",
            "data/broker_snapshot.json",
        ],
        "blocker": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
        "replay_window": {
            "lookback_hours": ev.get("window_lookback_hours") or _nested(timing, "window", "lookback_hours"),
            "from_utc": _nested(timing, "window", "from_utc"),
            "to_utc": _nested(timing, "window", "to_utc"),
            "post_close_hours": _nested(timing, "window", "post_close_hours"),
        },
        "pl_summary": {
            "baseline_actual_loss_close_pl_jpy": _nested(timing, "summary", "loss_close_actual_pl_jpy"),
            "raw_counterfactual_profit_capture_pl_jpy": ev.get("raw_counterfactual_profit_capture_pl_jpy"),
            "active_counterfactual_profit_capture_pl_jpy": ev.get("active_counterfactual_profit_capture_pl_jpy"),
            "current_residual_pl_jpy": current_residual,
            "improved_pl_after_missed_capture_repair_jpy": ev.get("repair_replay_counterfactual_pl_jpy"),
            "counterfactual_profit_capture_jpy": ev.get("counterfactual_profit_capture_jpy"),
            "counterfactual_profit_capture_delta_jpy": ev.get("counterfactual_profit_capture_delta_jpy"),
            "loss_closes_profit_capture_missed": ev.get("loss_closes_profit_capture_missed"),
            "loss_closes_repair_replay_triggered": ev.get("loss_closes_repair_replay_triggered"),
        },
        "residual_losing_families": residuals,
        "pair_side_strategy_groups": residuals,
        "method_rollups": ev.get("top_repair_replay_residual_method_rollups") or [],
        "bad_entry_vs_bad_exit_vs_missed_capture": _residual_reason_rollup(residuals),
        "manual_eurusd_excluded": {
            "trade_id": EUR_USD_MANUAL_TRADE_ID,
            "excluded": bool(manual) and _nested(manual, "operator_manual_position", "system_pl_counted") is False,
            "classification": _nested(manual, "operator_manual_position", "classification"),
            "management_intent": _nested(manual, "operator_manual_position", "management_intent"),
        },
        "filter_to_make_replay_non_negative": {
            "required_replay_pl_improvement_jpy": round(required, 4),
            "known_top_residual_abs_loss_jpy": round(known_abs, 4),
            "additional_tail_loss_to_cover_jpy": round(max(required - known_abs, 0.0), 4),
            "filter": "Block or repair every matching pair/side/method residual group with NO_PROFIT_CANDIDATE, BELOW_TP_PROGRESS_GATE, or BELOW_NOISE_FLOOR until the full 744h replay is non-negative; the top residual table alone is not permission evidence.",
            "manual_eurusd_must_stay_excluded": True,
        },
        "gate_definitions": [
            {
                "blocker_code": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
                "current_evidence": "744h replay improves historical missed captures but remains net negative after the TP-progress production-gate repair.",
                "exact_clearing_condition": "Rerun execution-timing-audit --lookback-hours 744 --post-close-hours 6 and profitability-acceptance; blocker clears only when replay P/L is non-negative or matching residual groups disappear.",
                "artifact_that_proves_it": [
                    "data/execution_timing_audit.json",
                    "data/profitability_acceptance.json",
                    "data/month_scale_tp_replay_residuals.json",
                ],
                "verifier_riskengine_liveordergateway_implication": "Verifier and RiskEngine must block matching residual pair/side/method groups from A/S; LiveOrderGateway cannot route them even if order_intents produce units.",
                "can_create_as_permission": False,
            }
        ],
        "fresh_entries_must_remain_blocked": True,
        "live_side_effects": [],
    }


def _build_month_scale_residual_family_table(
    *,
    generated_at: str,
    timing: dict[str, Any],
    harvest_gate: dict[str, Any],
    broker: dict[str, Any],
) -> dict[str, Any]:
    raw_rows = timing.get("loss_close_regrets") if isinstance(timing.get("loss_close_regrets"), list) else []
    rows = [
        _residual_family_replay_row(row)
        for row in raw_rows
        if isinstance(row, dict)
    ]
    family_excluded_ids = set(MARKET_CLOSE_LEAK_FAMILY_TRADE_IDS)
    manual_excluded_ids = set(MARKET_CLOSE_LEAK_FAMILY_MANUAL_EXCLUDED_TRADE_IDS)
    included = [
        row
        for row in rows
        if row["trade_id"] not in family_excluded_ids
        and row["trade_id"] not in manual_excluded_ids
    ]
    negative_groups = _residual_family_groups(included)
    blocked_keys = {
        (item["pair"], item["side"], item["method"])
        for item in negative_groups
        if float(item.get("residual_pl_jpy") or 0.0) < 0.0
    }
    after_filter_rows = [
        row
        for row in included
        if (row["pair"], row["side"], row["method"]) not in blocked_keys
    ]
    harvest_source_timing_generated = _nested(
        harvest_gate,
        "source",
        "execution_timing_audit_generated_at_utc",
    )
    harvest_matches_timing = bool(
        timing.get("generated_at_utc")
        and harvest_source_timing_generated == timing.get("generated_at_utc")
    )
    after_proposed = (
        (harvest_gate.get("month_scale_replay") or {}).get("after_proposed_gates")
        if harvest_matches_timing
        and isinstance((harvest_gate.get("month_scale_replay") or {}).get("after_proposed_gates"), dict)
        else {}
    )
    headline_baseline = _float_or_none(after_proposed.get("baseline_pl_jpy"))
    headline_improved = _float_or_none(after_proposed.get("improved_pl_jpy"))
    if headline_baseline is None:
        headline_baseline = sum(float(row["realized_pl_jpy"] or 0.0) for row in included)
    if headline_improved is None:
        headline_improved = sum(float(row["repair_replay_pl_jpy"] or 0.0) for row in included)
    after_filter_residual = sum(float(row["repair_replay_pl_jpy"] or 0.0) for row in after_filter_rows)
    families = [_classify_residual_family(item) for item in negative_groups]
    priority_families = [item for item in families if item.get("priority_focus")]
    tail_families = [item for item in families if not item.get("priority_focus")]
    manual = _manual_position(broker)
    excluded_trade_ids = sorted(
        {
            trade_id
            for item in families
            for trade_id in item.get("trade_ids", []) or []
        }
    )
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_month_scale_residual_family_gate",
        "source_artifacts": [
            "data/execution_timing_audit.json",
            "data/tp_progress_harvest_gate_evidence.json",
            "data/broker_snapshot.json",
        ],
        "source_execution_timing_audit": {
            "generated_at_utc": timing.get("generated_at_utc"),
            "path": "data/execution_timing_audit.json",
            "window": timing.get("window"),
            "repair_replay_contract": _nested(timing, "precision", "profit_capture_repair_replay_contract"),
        },
        "source_harvest_gate_evidence": {
            "generated_at_utc": harvest_gate.get("generated_at_utc"),
            "source_execution_timing_audit_generated_at_utc": _nested(
                harvest_gate,
                "source",
                "execution_timing_audit_generated_at_utc",
            ),
            "source_consistent_with_execution_timing": harvest_matches_timing,
            "basis": _nested(harvest_gate, "month_scale_replay", "basis"),
        },
        "summary": {
            "family_count": len(families),
            "priority_family_count": len(priority_families),
            "tail_family_count": len(tail_families),
            "negative_trade_count": sum(len(item.get("trade_ids", []) or []) for item in families),
            "priority_residual_pl_jpy": round(sum(float(item.get("residual_pl_jpy") or 0.0) for item in priority_families), 4),
            "tail_residual_pl_jpy": round(sum(float(item.get("residual_pl_jpy") or 0.0) for item in tail_families), 4),
            "all_can_create_live_permission_false": all(
                item.get("can_create_live_permission") is False for item in families
            ),
        },
        "replay_before_residual_family_filters": {
            "baseline_pl_jpy": round(headline_baseline, 4),
            "improved_pl_jpy": round(headline_improved, 4),
            "residual_pl_jpy": round(headline_improved, 4),
            "basis": "durable TP-progress harvest gate evidence after EUR_USD market-close family/manual exclusions when available",
            "clears_month_scale_tp_progress_replay_still_negative": headline_improved >= 0.0,
        },
        "replay_after_residual_family_filters": {
            "baseline_pl_jpy": round(headline_baseline, 4),
            "improved_pl_jpy": round(headline_improved, 4),
            "residual_pl_jpy": round(after_filter_residual, 4),
            "residual_family_filters_active": True,
            "excluded_family_count": len(blocked_keys),
            "excluded_trade_ids": excluded_trade_ids,
            "remaining_residual_groups": _remaining_negative_groups(after_filter_rows),
            "clears_month_scale_tp_progress_replay_still_negative": after_filter_residual >= 0.0,
            "basis": "Same 744h replay after blocking every negative pair/side/method residual family from fresh permission.",
        },
        "families": families,
        "residual_tail": {
            "family_count": len(tail_families),
            "residual_pl_jpy": round(sum(float(item.get("residual_pl_jpy") or 0.0) for item in tail_families), 4),
            "families": [
                {
                    "family": item.get("family"),
                    "residual_pl_jpy": item.get("residual_pl_jpy"),
                    "cause": item.get("cause"),
                    "proposed_action": item.get("proposed_action"),
                    "current_blocker_code": item.get("current_blocker_code"),
                }
                for item in tail_families
            ],
        },
        "market_close_leak_family_gate_active": True,
        "tp_progress_harvest_gate_active": True,
        "manual_eurusd_472987_excluded": bool(manual)
        and _nested(manual, "operator_manual_position", "system_pl_counted") is False,
        "no_unproven_fresh_entry_promotion": True,
        "all_negative_families_can_create_live_permission_false": all(
            item.get("can_create_live_permission") is False for item in families
        ),
        "gate_definitions": [
            {
                "blocker_code": "MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED",
                "current_evidence": "The same pair/side/method remains negative after TP-progress repair because no executable profit candidate or acceptable geometry exists.",
                "exact_clearing_condition": "The matching family disappears from a refreshed 744h replay, or exact local TP/geometry/forecast proof makes the filtered replay non-negative and order_intents no longer carry residual metadata.",
                "verifier_riskengine_liveordergateway_implication": "GPT verifier and RiskEngine block residual metadata; LiveOrderGateway receives no LIVE_READY receipt for matching families.",
                "can_create_as_permission": False,
            },
            {
                "blocker_code": "MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCKED",
                "current_evidence": "The TP-progress diagnostic row remains negative or below bid/ask/noise proof after the repair replay.",
                "exact_clearing_condition": "Spread-included non-negative replay plus current production-gate evidence removes the matching residual family.",
                "verifier_riskengine_liveordergateway_implication": "The family is historical-only until the exact repair proof is refreshed and acceptance clears.",
                "can_create_as_permission": False,
            },
        ],
        "safety": {
            **_safety_packet(),
            "market_close_leak_family_gate_active": True,
            "tp_progress_harvest_gate_active": True,
            "manual_eurusd_472987_excluded": bool(manual)
            and _nested(manual, "operator_manual_position", "system_pl_counted") is False,
            "no_unproven_fresh_entry_promotion": True,
            "all_negative_families_can_create_live_permission_false": all(
                item.get("can_create_live_permission") is False for item in families
            ),
        },
        "live_side_effects": [],
    }


def _residual_family_replay_row(row: dict[str, Any]) -> dict[str, Any]:
    actual = _float_or_none(row.get("realized_pl_jpy")) or 0.0
    counterfactual = _float_or_none(row.get("repair_replay_counterfactual_pl_jpy"))
    trigger_at = _parse_utc(row.get("repair_replay_trigger_at_utc"))
    close_at = _parse_utc(row.get("close_at_utc"))
    before_close = bool(trigger_at and close_at and trigger_at < close_at)
    profit_pips = _float_or_none(row.get("repair_replay_profit_pips"))
    noise_floor_pips = _float_or_none(row.get("repair_replay_noise_floor_pips"))
    above_noise = (
        profit_pips is not None
        and noise_floor_pips is not None
        and profit_pips > noise_floor_pips
    )
    executable = bool(
        row.get("repair_replay_triggered_before_loss_close")
        and before_close
        and above_noise
        and counterfactual is not None
        and counterfactual > actual
        and str(row.get("repair_replay_exit") or "") == "TP_PROGRESS_PRODUCTION_GATE_REPLAY"
    )
    block_reason = str(row.get("repair_replay_block_reason") or "").strip()
    return {
        "trade_id": str(row.get("trade_id") or ""),
        "pair": str(row.get("pair") or "UNKNOWN").strip() or "UNKNOWN",
        "side": str(row.get("side") or "UNKNOWN").strip().upper() or "UNKNOWN",
        "method": _method_from_lane_id(str(row.get("lane_id") or "")),
        "lane_id": row.get("lane_id"),
        "exit_reason": str(row.get("exit_reason") or "UNKNOWN").strip().upper(),
        "realized_pl_jpy": round(actual, 4),
        "repair_replay_pl_jpy": round(counterfactual if executable and counterfactual is not None else actual, 4),
        "repair_replay_counterfactual_pl_jpy": round(counterfactual, 4) if counterfactual is not None else None,
        "repair_replay_triggered": executable,
        "raw_repair_replay_triggered_before_loss_close": bool(
            row.get("repair_replay_triggered_before_loss_close")
        ),
        "profit_capture_missed_before_loss_close": bool(
            row.get("profit_capture_missed_before_loss_close")
        ),
        "repair_replay_block_reason": block_reason or None,
        "trigger_status": "CURRENT_RULE_TRIGGER" if executable else (block_reason or "NO_PROFIT_CANDIDATE"),
        "residual_scope": _residual_family_scope(row, executable=executable),
    }


def _residual_family_scope(row: dict[str, Any], *, executable: bool) -> str:
    if executable:
        return "TP_PROGRESS_REPAIR_TRIGGERED"
    if row.get("profit_capture_missed_before_loss_close"):
        return "TP_PROGRESS_DIAGNOSTIC_BLOCKED"
    return "ENTRY_QUALITY_OR_CLOSE_RESIDUAL"


def _residual_family_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        residual = _float_or_none(row.get("repair_replay_pl_jpy"))
        if residual is None or residual >= 0.0:
            continue
        key = (row["pair"], row["side"], row["method"])
        item = groups.setdefault(
            key,
            {
                "pair": key[0],
                "side": key[1],
                "method": key[2],
                "trade_ids": [],
                "loss_closes": 0,
                "realized_pl_jpy": 0.0,
                "counterfactual_pl_after_tp_progress_repair_jpy": 0.0,
                "residual_pl_jpy": 0.0,
                "exit_reasons": {},
                "block_reasons": {},
                "residual_scopes": {},
                "examples": [],
            },
        )
        item["trade_ids"].append(row["trade_id"])
        item["loss_closes"] = int(item["loss_closes"]) + 1
        item["realized_pl_jpy"] = float(item["realized_pl_jpy"]) + float(row["realized_pl_jpy"])
        item["counterfactual_pl_after_tp_progress_repair_jpy"] = float(
            item["counterfactual_pl_after_tp_progress_repair_jpy"]
        ) + residual
        item["residual_pl_jpy"] = float(item["residual_pl_jpy"]) + residual
        _increment_count(item["exit_reasons"], row.get("exit_reason"))
        _increment_count(item["block_reasons"], row.get("repair_replay_block_reason") or row.get("trigger_status"))
        _increment_count(item["residual_scopes"], row.get("residual_scope"))
        if len(item["examples"]) < 4:
            item["examples"].append(
                {
                    "trade_id": row["trade_id"],
                    "lane_id": row.get("lane_id"),
                    "realized_pl_jpy": row.get("realized_pl_jpy"),
                    "repair_replay_pl_jpy": row.get("repair_replay_pl_jpy"),
                    "repair_replay_block_reason": row.get("repair_replay_block_reason"),
                    "exit_reason": row.get("exit_reason"),
                    "residual_scope": row.get("residual_scope"),
                }
            )
    result = []
    for item in groups.values():
        result.append(
            {
                **item,
                "trade_ids": sorted(str(value) for value in item.get("trade_ids") or []),
                "realized_pl_jpy": round(float(item.get("realized_pl_jpy") or 0.0), 4),
                "counterfactual_pl_after_tp_progress_repair_jpy": round(
                    float(item.get("counterfactual_pl_after_tp_progress_repair_jpy") or 0.0),
                    4,
                ),
                "residual_pl_jpy": round(float(item.get("residual_pl_jpy") or 0.0), 4),
                "exit_reasons": _sorted_count_dict(item.get("exit_reasons")),
                "block_reasons": _sorted_count_dict(item.get("block_reasons")),
                "residual_scopes": _sorted_count_dict(item.get("residual_scopes")),
            }
        )
    result.sort(
        key=lambda item: (
            float(item.get("residual_pl_jpy") or 0.0),
            str(item.get("pair") or ""),
            str(item.get("side") or ""),
            str(item.get("method") or ""),
        )
    )
    return result


def _classify_residual_family(item: dict[str, Any]) -> dict[str, Any]:
    key = (
        str(item.get("pair") or "UNKNOWN"),
        str(item.get("side") or "UNKNOWN"),
        str(item.get("method") or "UNKNOWN"),
    )
    block_reasons = item.get("block_reasons") if isinstance(item.get("block_reasons"), dict) else {}
    exit_reasons = item.get("exit_reasons") if isinstance(item.get("exit_reasons"), dict) else {}
    residual_scopes = item.get("residual_scopes") if isinstance(item.get("residual_scopes"), dict) else {}
    cause = _residual_family_cause(key, block_reasons, exit_reasons)
    proposed_action = _residual_family_action(key, cause, block_reasons)
    blocker_code = (
        "MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCKED"
        if "TP_PROGRESS_DIAGNOSTIC_BLOCKED" in residual_scopes
        else "MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED"
    )
    exact_evidence = _residual_family_exact_evidence(key, cause, proposed_action)
    priority = key in PRIORITY_RESIDUAL_FAMILY_KEYS
    can_ever_be_as = proposed_action not in {"BAN_FAMILY", "HISTORICAL_ONLY", "OPERATOR_MANUAL_EXCLUDE"}
    return {
        "family": f"{key[0]} {key[1]} {key[2]}",
        "pair": key[0],
        "side": key[1],
        "strategy": key[2],
        "method": key[2],
        "trade_ids": item.get("trade_ids") or [],
        "loss_closes": item.get("loss_closes"),
        "realized_pl_jpy": item.get("realized_pl_jpy"),
        "counterfactual_pl_after_tp_progress_repair_jpy": item.get(
            "counterfactual_pl_after_tp_progress_repair_jpy"
        ),
        "residual_pl_jpy": item.get("residual_pl_jpy"),
        "cause": cause,
        "current_blocker_code": blocker_code,
        "proposed_action": proposed_action,
        "can_create_live_permission": False,
        "can_ever_be_as": can_ever_be_as,
        "decision": "BAN_NOW_REPAIR_ONLY_WITH_EXACT_EVIDENCE" if can_ever_be_as else "BAN_NOW_HISTORICAL_ONLY",
        "priority_focus": priority,
        "residual_tail": not priority,
        "exit_reasons": exit_reasons,
        "block_reasons": block_reasons,
        "residual_scopes": residual_scopes,
        "examples": item.get("examples") or [],
        "exact_evidence_needed_to_repair": exact_evidence,
        "as_status": "NO_CURRENT_A_S; possible only after exact evidence and fresh LIVE_READY regeneration"
        if can_ever_be_as
        else "NO_CURRENT_A_S; historical-only until a new family proof exists",
    }


def _residual_family_cause(
    key: tuple[str, str, str],
    block_reasons: dict[str, Any],
    exit_reasons: dict[str, Any],
) -> str:
    method = key[2]
    if "BELOW_NOISE_FLOOR" in block_reasons:
        return "NEGATIVE_BIDASK_REPLAY"
    if "NO_PROFIT_CANDIDATE" in block_reasons:
        if method == "RANGE_ROTATION":
            return "RANGE_CHASE"
        return "FORECAST_NOT_EXECUTABLE"
    if "BELOW_TP_PROGRESS_GATE" in block_reasons:
        if method == "RANGE_ROTATION":
            return "RANGE_CHASE"
        if "STOP_LOSS_ORDER" in exit_reasons:
            return "BAD_ENTRY"
        return "BAD_EXIT"
    return "UNKNOWN"


def _residual_family_action(
    key: tuple[str, str, str],
    cause: str,
    block_reasons: dict[str, Any],
) -> str:
    method = key[2]
    if cause == "NEGATIVE_BIDASK_REPLAY":
        return "REQUIRE_BIDASK_NON_NEGATIVE"
    if cause == "FORECAST_NOT_EXECUTABLE":
        return "REQUIRE_FORECAST_EXECUTABLE"
    if cause == "RANGE_CHASE" or method == "RANGE_ROTATION":
        return "REQUIRE_GEOMETRY_REPAIR"
    if "BELOW_TP_PROGRESS_GATE" in block_reasons:
        return "REQUIRE_LOCAL_TP_PROOF"
    return "BAN_FAMILY"


def _residual_family_exact_evidence(
    key: tuple[str, str, str],
    cause: str,
    proposed_action: str,
) -> list[str]:
    pair, side, method = key
    evidence = [
        f"fresh 744h execution-timing-audit where {pair} {side} {method} no longer has negative residual replay P/L",
        "profitability-acceptance regenerated from the same timing artifact and not stale against inputs",
        "order_intents regenerated with no month_scale_residual_loss_group metadata for the matching family",
        "RiskEngine and LiveOrderGateway validation on a fresh broker snapshot after all blockers clear",
        "fresh GPT TRADE/ADD receipt only after the lane is already LIVE_READY",
    ]
    if proposed_action == "REQUIRE_GEOMETRY_REPAIR":
        evidence.extend(
            [
                "RANGE_ROTATION broad-location proof: LONG entries only in broad discount/lower half and SHORT entries only in premium/upper half",
                "TP geometry proof that target lies inside the current range and SL lies outside invalidation without negative reward/risk distortion",
            ]
        )
    elif proposed_action == "REQUIRE_LOCAL_TP_PROOF":
        evidence.extend(
            [
                "spread-included local TP proof for the exact pair/side/method/order-type shape",
                "close-gate proof if any MARKET_ORDER_TRADE_CLOSE path is retained",
            ]
        )
    elif proposed_action == "REQUIRE_BIDASK_NON_NEGATIVE":
        evidence.append("bid/ask replay for the exact family is non-negative after spread, noise floor, samples, and active-day checks")
    elif proposed_action == "REQUIRE_FORECAST_EXECUTABLE":
        evidence.append("fresh forecast packet marks the family executable with direction/method agreement and no watch-only fallback")
    if cause == "BAD_EXIT":
        evidence.append("loss-side close path proves thesis invalidation and contained risk; otherwise use attached TP/HARVEST only")
    return evidence


def _remaining_negative_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pair": item.get("pair"),
            "side": item.get("side"),
            "method": item.get("method"),
            "residual_pl_jpy": item.get("residual_pl_jpy"),
            "trade_ids": item.get("trade_ids"),
        }
        for item in _residual_family_groups(rows)
        if float(item.get("residual_pl_jpy") or 0.0) < 0.0
    ]


def _increment_count(target: Any, raw_key: Any) -> None:
    if not isinstance(target, dict):
        return
    key = str(raw_key or "UNKNOWN")
    target[key] = int(target.get(key) or 0) + 1


def _sorted_count_dict(value: Any) -> dict[str, int]:
    source = value if isinstance(value, dict) else {}
    return dict(
        sorted(
            ((str(key), int(count)) for key, count in source.items()),
            key=lambda item: (-item[1], item[0]),
        )
    )


def _parse_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _build_historical_profit_capture_missed_table(
    *,
    generated_at: str,
    timing: dict[str, Any],
    broker: dict[str, Any],
) -> dict[str, Any]:
    rows = [_missed_capture_row(item) for item in _missed_capture_regrets(timing)]
    manual = _manual_position(broker)
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_profit_capture_decomposition",
        "source_artifacts": [
            "data/execution_timing_audit.json",
            "data/profit_capture_bot.json",
            "data/broker_snapshot.json",
        ],
        "blocker": "HISTORICAL_PROFIT_CAPTURE_MISSED",
        "summary": {
            "missed_capture_count": len(rows),
            "repair_replay_triggered_count": sum(1 for row in rows if row.get("capture_would_be_allowed_under_current_rules")),
            "historical_actual_loss_close_pl_jpy": _nested(timing, "summary", "loss_close_actual_pl_jpy"),
            "historical_counterfactual_profit_capture_delta_jpy": _nested(timing, "summary", "loss_close_counterfactual_profit_capture_delta_jpy"),
            "historical_repair_replay_delta_jpy": _nested(timing, "summary", "loss_close_repair_replay_delta_jpy"),
            "post_repair_live_missed_count": _nested(timing, "summary", "post_repair_live_evidence_loss_closes_profit_capture_missed"),
        },
        "manual_eurusd_guard": {
            "trade_id": EUR_USD_MANUAL_TRADE_ID,
            "present": bool(manual),
            "classification": _nested(manual, "operator_manual_position", "classification"),
            "management_intent": _nested(manual, "operator_manual_position", "management_intent"),
            "system_pl_counted": _nested(manual, "operator_manual_position", "system_pl_counted"),
        },
        "missed_captures": rows,
        "gate_definitions": [
            {
                "blocker_code": "HISTORICAL_PROFIT_CAPTURE_MISSED",
                "current_evidence": "14 recent loss closes had executable profit-capture opportunities before loss close; production-gate replay triggers on 13.",
                "exact_clearing_condition": "Post-repair live evidence remains clean and refreshed 744h replay residuals are non-negative or age out of the month-scale blocker.",
                "artifact_that_proves_it": [
                    "data/profit_capture_bot.json",
                    "data/execution_timing_audit.json",
                    "data/historical_profit_capture_missed_table.json",
                ],
                "verifier_riskengine_liveordergateway_implication": "Profit-capture trigger logic stays required; RiskEngine cannot treat historical repair as A/S permission until replay and profitability blockers clear.",
                "can_create_as_permission": False,
            }
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
    residual_family_table: dict[str, Any],
) -> dict[str, Any]:
    results = order_intents.get("results") if isinstance(order_intents.get("results"), list) else []
    live_ready = [item for item in results if item.get("status") == "LIVE_READY"]
    usdjpy_current = [item for item in results if (item.get("intent") or {}).get("pair") == "USD_JPY"]
    eurjpy_short = [item for item in results if (item.get("intent") or {}).get("pair") == "EUR_JPY" and (item.get("intent") or {}).get("side") == "SHORT"]
    audusd_current = [item for item in results if (item.get("intent") or {}).get("pair") == "AUD_USD"]
    manual = _manual_position(broker)
    target_lane_present = any(item.get("lane_id") == USD_JPY_TARGET_LANE for item in results)
    finding_codes = set(_findings_by_code(acceptance))
    month_residual_line = (
        "Profitability acceptance still raises NEGATIVE_EXPECTANCY_ACTIVE, MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE, and MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE."
        if "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE" in finding_codes
        else "Profitability acceptance no longer raises MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE; residual-family gates still block every negative historical family from fresh A/S permission."
    )
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
            "data/market_close_leak_trade_table.json",
            "data/month_scale_tp_replay_residuals.json",
            "data/month_scale_residual_family_table.json",
            "data/historical_profit_capture_missed_table.json",
        ],
        "repair_analysis_artifacts": [
            "data/market_close_leak_trade_table.json",
            "docs/market_close_leak_trade_table.md",
            "data/month_scale_tp_replay_residuals.json",
            "docs/month_scale_tp_replay_residuals.md",
            "data/month_scale_residual_family_table.json",
            "docs/month_scale_residual_family_table.md",
            "data/historical_profit_capture_missed_table.json",
            "docs/historical_profit_capture_missed_table.md",
        ],
        "order_intents_generated_at_utc": order_intents.get("generated_at_utc"),
        "previous_board_generated_at_utc": old_board.get("generated_at_utc"),
        "total_lanes": len(results),
        "live_ready_lanes": len(live_ready),
        "as_live_ready_path_exists": False,
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
        "aud_usd_rejection_state": {
            "classification": "REJECTED",
            "as_candidate": False,
            "live_ready_allowed": False,
            "reason": "All current AUD_USD lanes are DRY_RUN_BLOCKED with negative-expectancy, month-scale residual, stale quote, spread, loss-budget, bid/ask replay, and guardian review blockers.",
            "current_lanes": [_lane_summary(item) for item in audusd_current],
        },
        "month_scale_residual_family_gate": {
            "family_count": _nested(residual_family_table, "summary", "family_count"),
            "priority_family_count": _nested(residual_family_table, "summary", "priority_family_count"),
            "tail_family_count": _nested(residual_family_table, "summary", "tail_family_count"),
            "baseline_pl_jpy": _nested(residual_family_table, "replay_after_residual_family_filters", "baseline_pl_jpy"),
            "improved_pl_jpy": _nested(residual_family_table, "replay_after_residual_family_filters", "improved_pl_jpy"),
            "residual_pl_jpy": _nested(residual_family_table, "replay_after_residual_family_filters", "residual_pl_jpy"),
            "clears_month_scale_tp_progress_replay_still_negative": _nested(
                residual_family_table,
                "replay_after_residual_family_filters",
                "clears_month_scale_tp_progress_replay_still_negative",
            ),
            "all_negative_families_can_create_live_permission_false": residual_family_table.get(
                "all_negative_families_can_create_live_permission_false"
            ),
            "priority_families": [
                {
                    "family": item.get("family"),
                    "cause": item.get("cause"),
                    "current_blocker_code": item.get("current_blocker_code"),
                    "proposed_action": item.get("proposed_action"),
                    "residual_pl_jpy": item.get("residual_pl_jpy"),
                    "can_create_live_permission": item.get("can_create_live_permission"),
                    "can_ever_be_as": item.get("can_ever_be_as"),
                }
                for item in residual_family_table.get("families", []) or []
                if item.get("priority_focus")
            ],
        },
        "profitability_blockers": blocker_decomposition,
        "next_best_candidates": _next_best_candidates(repair_orchestrator, acceptance, eurjpy_proof),
        "blocker_hierarchy": [
            "No order_intents row is LIVE_READY.",
            "Normal routing remains BLOCKED.",
            month_residual_line,
            "Residual family table blocks every negative pair/side/method group from fresh A/S permission until exact proof clears the matching family.",
            "USD_JPY LONG BREAKOUT_FAILURE LIMIT remains rejected until fresh exact positive proof exists.",
            "EUR_JPY SHORT remains rejected/evidence-gap until spread-included non-negative proof exists.",
            "AUD_USD remains rejected by current DRY_RUN_BLOCKED lanes and month-scale residual blockers.",
            "Guardian/operator review and fresh GPT TRADE/ADD receipt requirements remain unfulfilled.",
        ],
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
            "broker_nav_jpy": _nested(broker, "account", "nav_jpy"),
            "current_equity_raw_equals_broker_nav": daily.get("current_equity_raw") == _nested(broker, "account", "nav_jpy"),
            "funding_adjusted_equity": daily.get("funding_adjusted_equity"),
            "capital_flows_30d": daily.get("capital_flows_30d"),
            "capital_flow_count_30d": daily.get("capital_flow_count_30d"),
            "deposit_100000_jpy_excluded_from_performance": True,
            "rolling_30d_multiplier_funding_adjusted_is_kpi": daily.get("performance_basis") == "funding_adjusted",
        },
        "manual_eurusd_472987": {
            "present": bool(manual),
            "classification": _nested(manual, "operator_manual_position", "classification"),
            "management_intent": _nested(manual, "operator_manual_position", "management_intent"),
            "system_pl_counted": _nested(manual, "operator_manual_position", "system_pl_counted"),
            "system_occupancy_counted": False,
            "auto_close_allowed": False,
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
          AND (lane_id = ? OR lane_id LIKE ?)
        ORDER BY ts_utc
    """
    for rec in conn.execute(query, (EUR_USD_LEAK_LANE, f"{EUR_USD_LEAK_LANE}:%")):
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
                "lane_id": rec["lane_id"],
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
        family = "BAD_ENTRY_NO_PROFIT_CANDIDATE"
    elif "BELOW_TP_PROGRESS_GATE" in reasons:
        diagnosis = "entry_quality_or_premature_exit_below_tp_progress_gate"
        family = "BAD_ENTRY_OR_BAD_EXIT_BELOW_TP_PROGRESS_GATE"
    elif "BELOW_NOISE_FLOOR" in reasons:
        diagnosis = "missed_capture_but_below_noise_floor"
        family = "MISSED_CAPTURE_BELOW_NOISE_FLOOR"
    else:
        diagnosis = "residual_needs_manual_review"
        family = "MANUAL_REVIEW"
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
        "bad_entry_vs_bad_exit_vs_missed_capture": family,
        "live_permission_filter": "block matching pair/side/method until replay clears or exact TP proof overrides with all gates passed",
    }


def _residual_reason_rollup(residuals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rollup: dict[str, dict[str, Any]] = {}
    for item in residuals:
        family = str(item.get("bad_entry_vs_bad_exit_vs_missed_capture") or "MANUAL_REVIEW")
        bucket = rollup.setdefault(
            family,
            {
                "groups": 0,
                "loss_closes": 0,
                "repair_replay_pl_jpy": 0.0,
                "examples": [],
            },
        )
        bucket["groups"] += 1
        bucket["loss_closes"] += int(item.get("loss_closes") or 0)
        bucket["repair_replay_pl_jpy"] = round(float(bucket["repair_replay_pl_jpy"]) + float(item.get("repair_replay_pl_jpy") or 0.0), 4)
        for example in item.get("examples") or []:
            if len(bucket["examples"]) < 5:
                bucket["examples"].append(example.get("trade_id"))
    return rollup


def _missed_capture_regrets(timing: dict[str, Any]) -> list[dict[str, Any]]:
    rows = timing.get("loss_close_regrets") if isinstance(timing.get("loss_close_regrets"), list) else []
    return [row for row in rows if isinstance(row, dict) and row.get("profit_capture_missed_before_loss_close")]


def _missed_capture_row(row: dict[str, Any]) -> dict[str, Any]:
    triggered = bool(row.get("repair_replay_triggered_before_loss_close"))
    capture_price = _capture_price_from_pips(row)
    lane_id = row.get("lane_id")
    if lane_id:
        system_or_manual = "SYSTEM"
    else:
        system_or_manual = "UNATTRIBUTED_HISTORY_NOT_OPERATOR_MANUAL_472987"
    trigger = row.get("repair_replay_exit") if triggered else row.get("repair_replay_block_reason") or row.get("profit_capture_counterfactual_exit")
    return {
        "trade_id": str(row.get("trade_id")),
        "pair": row.get("pair"),
        "side": row.get("side"),
        "lane_id": lane_id,
        "max_favorable_excursion": {
            "mfe_pips_before_loss_close": row.get("mfe_pips_before_loss_close"),
            "estimated_mfe_jpy_before_loss_close": row.get("estimated_mfe_jpy_before_loss_close"),
            "mfe_at_utc": row.get("mfe_at_utc"),
            "tp_progress_before_loss_close": row.get("tp_progress_before_loss_close"),
        },
        "executable_capture_price_time": {
            "price": capture_price,
            "time_utc": row.get("repair_replay_trigger_at_utc") or row.get("mfe_at_utc"),
            "price_source": "entry_plus_repair_replay_profit_pips" if triggered else "entry_plus_raw_counterfactual_profit_pips",
            "profit_pips": row.get("repair_replay_profit_pips") if triggered else row.get("profit_capture_counterfactual_pips"),
            "spread_pips": row.get("repair_replay_spread_pips"),
            "noise_floor_pips": row.get("repair_replay_noise_floor_pips"),
        },
        "missed_capture_amount": {
            "raw_counterfactual_net_improvement_jpy": row.get("profit_capture_counterfactual_net_improvement_jpy"),
            "repair_replay_net_improvement_jpy": row.get("repair_replay_counterfactual_net_improvement_jpy"),
            "repair_replay_counterfactual_pl_jpy": row.get("repair_replay_counterfactual_pl_jpy"),
        },
        "current_close_result": {
            "entry_price": row.get("entry"),
            "entry_time_utc": row.get("fill_at_utc"),
            "close_price": row.get("close_price"),
            "close_time_utc": row.get("close_at_utc"),
            "realized_pl_jpy": row.get("realized_pl_jpy"),
            "close_reason": row.get("exit_reason"),
        },
        "system_or_manual": system_or_manual,
        "capture_would_be_allowed_under_current_rules": triggered,
        "current_rules_block_reason": None if triggered else row.get("repair_replay_block_reason"),
        "exact_trigger_that_should_have_fired": trigger,
        "operator_manual_eurusd_472987": False,
    }


def _capture_price_from_pips(row: dict[str, Any]) -> float | None:
    entry = _float_or_none(row.get("entry"))
    if entry is None:
        return None
    pips = _float_or_none(row.get("repair_replay_profit_pips"))
    if pips is None:
        pips = _float_or_none(row.get("profit_capture_counterfactual_pips"))
    if pips is None:
        return None
    pair = str(row.get("pair") or "")
    pip_size = 0.01 if pair.endswith("_JPY") else 0.0001
    side = str(row.get("side") or "").upper()
    price = entry + pips * pip_size if side == "LONG" else entry - pips * pip_size
    return round(price, 3 if pair.endswith("_JPY") else 5)


def _method_from_lane_id(lane_id: str) -> str:
    parts = [part for part in str(lane_id or "").split(":") if part]
    if len(parts) >= 4:
        return str(parts[3] or "UNKNOWN").strip().upper()
    return "UNKNOWN"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _market_close_trade_table_md(table: dict[str, Any]) -> str:
    blocker = (table.get("blockers") or {}).get("MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE") or {}
    recent = (table.get("blockers") or {}).get("RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK") or {}
    lines = [
        "# Market Close Leak Trade Table",
        "",
        f"- Generated: `{table.get('generated_at_utc')}`",
        f"- MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE present: `{blocker.get('present')}`",
        f"- RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK present in current acceptance: `{recent.get('present_in_current_profitability_acceptance')}`",
        f"- Fresh entries blocked: `{table.get('fresh_entries_must_remain_blocked')}`",
        "",
        "## Contributing Trades",
        "",
        "| trade_id | pair | side | strategy | entry | close | P/L JPY | close reason | attribution | campaign recovery | counts against system edge | regression coverage |",
        "|---|---|---|---|---|---|---:|---|---|---|---|---|",
    ]
    for item in table.get("trades") or []:
        lines.append(
            "| {trade_id} | {pair} | {side} | {strategy} | {entry_time_utc} @ {entry_price} | {close_time_utc} @ {close_price} | {realized_pl_jpy} | {close_reason} | {system_gateway_or_operator_manual} | {campaign_exposure_recovery} | {should_count_against_system_edge} | {already_covered_by_regression_tests} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Recent Gateway Loss Leak",
            "",
            f"- Current contributing trades: `{recent.get('contributing_trade_count')}`",
            f"- Status: `{recent.get('status')}`",
            f"- Note: {recent.get('note')}",
            "",
            "## Mitigation Families",
            "",
            "| family | fix type | banned path | evidence to allow again | files/modules | tests |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in table.get("mitigation_by_family") or []:
        lines.append(
            f"| `{item.get('family')}` | `{', '.join(item.get('fix_type') or [])}` | {item.get('entry_or_close_path_must_be_banned')} | {', '.join(item.get('evidence_that_would_allow_again') or [])} | `{', '.join(item.get('files_modules_to_update') or [])}` | `{', '.join(item.get('tests_required') or [])}` |"
        )
    lines.extend(
        [
            "",
            "## Gate Definitions",
            "",
        ]
    )
    for item in table.get("gate_definitions") or []:
        lines.append(f"- `{item.get('blocker_code')}`: {item.get('exact_clearing_condition')} Can create A/S permission: `{item.get('can_create_as_permission')}`")
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


def _month_scale_tp_replay_residuals_md(table: dict[str, Any]) -> str:
    pl = table.get("pl_summary") or {}
    filt = table.get("filter_to_make_replay_non_negative") or {}
    manual = table.get("manual_eurusd_excluded") or {}
    lines = [
        "# Month-Scale TP Replay Residuals",
        "",
        f"- Generated: `{table.get('generated_at_utc')}`",
        f"- Blocker: `{table.get('blocker')}`",
        f"- Replay window: `{json.dumps(table.get('replay_window'), sort_keys=True)}`",
        f"- Baseline actual loss-close P/L JPY: `{pl.get('baseline_actual_loss_close_pl_jpy')}`",
        f"- Improved P/L after missed-capture repair JPY: `{pl.get('improved_pl_after_missed_capture_repair_jpy')}`",
        f"- Current residual P/L JPY: `{pl.get('current_residual_pl_jpy')}`",
        f"- Manual EUR_USD `{EUR_USD_MANUAL_TRADE_ID}` excluded: `{manual.get('excluded')}`",
        "",
        "## Residual Groups",
        "",
        "| pair | side | strategy | exit | repair P/L JPY | loss closes | family | block reasons | examples |",
        "|---|---|---|---|---:|---:|---|---|---|",
    ]
    for item in table.get("residual_losing_families") or []:
        examples = ",".join(str(example.get("trade_id")) for example in item.get("examples") or [])
        lines.append(
            f"| {item.get('pair')} | {item.get('side')} | {item.get('strategy')} | {item.get('exit_reason')} | {item.get('repair_replay_pl_jpy')} | {item.get('loss_closes')} | {item.get('bad_entry_vs_bad_exit_vs_missed_capture')} | {json.dumps(item.get('block_reasons'), sort_keys=True)} | {examples} |"
        )
    lines.extend(
        [
            "",
            "## Rollup",
            "",
            f"- Bad entry / bad exit / missed capture: `{json.dumps(table.get('bad_entry_vs_bad_exit_vs_missed_capture'), sort_keys=True)}`",
            f"- Required improvement to non-negative: `{filt.get('required_replay_pl_improvement_jpy')}` JPY",
            f"- Known top residual abs loss: `{filt.get('known_top_residual_abs_loss_jpy')}` JPY",
            f"- Additional tail loss to cover: `{filt.get('additional_tail_loss_to_cover_jpy')}` JPY",
            f"- Filter: {filt.get('filter')}",
            "",
            "## Gate Definitions",
            "",
        ]
    )
    for item in table.get("gate_definitions") or []:
        lines.append(f"- `{item.get('blocker_code')}`: {item.get('exact_clearing_condition')} Can create A/S permission: `{item.get('can_create_as_permission')}`")
    return "\n".join(lines) + "\n"


def _month_scale_residual_family_table_md(table: dict[str, Any]) -> str:
    summary = table.get("summary") or {}
    before = table.get("replay_before_residual_family_filters") or {}
    after = table.get("replay_after_residual_family_filters") or {}
    source = table.get("source_execution_timing_audit") or {}
    harvest = table.get("source_harvest_gate_evidence") or {}
    lines = [
        "# Month-Scale Residual Family Table",
        "",
        f"- Generated: `{table.get('generated_at_utc')}`",
        f"- Execution timing generated: `{source.get('generated_at_utc')}`",
        f"- Harvest evidence generated: `{harvest.get('generated_at_utc')}`",
        f"- Harvest source matches timing: `{harvest.get('source_consistent_with_execution_timing')}`",
        f"- Family count: `{summary.get('family_count')}`",
        f"- Priority families: `{summary.get('priority_family_count')}`",
        f"- Tail families: `{summary.get('tail_family_count')}`",
        f"- All negative families can create live permission: `{not table.get('all_negative_families_can_create_live_permission_false')}`",
        "",
        "## Replay",
        "",
        f"- Before filters baseline P/L JPY: `{before.get('baseline_pl_jpy')}`",
        f"- Before filters improved/residual P/L JPY: `{before.get('improved_pl_jpy')}`",
        f"- After residual-family filters residual P/L JPY: `{after.get('residual_pl_jpy')}`",
        f"- MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE clears after filters: `{after.get('clears_month_scale_tp_progress_replay_still_negative')}`",
        f"- Excluded family count: `{after.get('excluded_family_count')}`",
        f"- Remaining residual groups: `{json.dumps(after.get('remaining_residual_groups') or [], sort_keys=True)}`",
        "",
        "## Families",
        "",
        "| family | trades | realized P/L | counterfactual P/L | residual P/L | cause | blocker | action | A/S now | can ever A/S | priority |",
        "|---|---:|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for item in table.get("families") or []:
        lines.append(
            f"| {item.get('family')} | {','.join(item.get('trade_ids') or [])} | {item.get('realized_pl_jpy')} | {item.get('counterfactual_pl_after_tp_progress_repair_jpy')} | {item.get('residual_pl_jpy')} | `{item.get('cause')}` | `{item.get('current_blocker_code')}` | `{item.get('proposed_action')}` | `{item.get('can_create_live_permission')}` | `{item.get('can_ever_be_as')}` | `{item.get('priority_focus')}` |"
        )
    lines.extend(
        [
            "",
            "## Priority Repair Requirements",
            "",
        ]
    )
    for item in table.get("families") or []:
        if not item.get("priority_focus"):
            continue
        lines.append(f"### {item.get('family')}")
        lines.append(f"- Decision: `{item.get('decision')}`")
        lines.append(f"- Cause/action: `{item.get('cause')}` / `{item.get('proposed_action')}`")
        lines.append(f"- A/S status: `{item.get('as_status')}`")
        for evidence in item.get("exact_evidence_needed_to_repair") or []:
            lines.append(f"- {evidence}")
        lines.append("")
    tail = table.get("residual_tail") or {}
    lines.extend(
        [
            "## Residual Tail",
            "",
            f"- Tail family count: `{tail.get('family_count')}`",
            f"- Tail residual P/L JPY: `{tail.get('residual_pl_jpy')}`",
        ]
    )
    for item in tail.get("families") or []:
        lines.append(
            f"- `{item.get('family')}`: `{item.get('residual_pl_jpy')}` JPY, `{item.get('cause')}`, `{item.get('proposed_action')}`, blocker `{item.get('current_blocker_code')}`"
        )
    lines.extend(
        [
            "",
            "## Gate Definitions",
            "",
        ]
    )
    for item in table.get("gate_definitions") or []:
        lines.append(f"- `{item.get('blocker_code')}`: {item.get('exact_clearing_condition')} Can create A/S permission: `{item.get('can_create_as_permission')}`")
    return "\n".join(lines) + "\n"


def _historical_profit_capture_missed_md(table: dict[str, Any]) -> str:
    summary = table.get("summary") or {}
    lines = [
        "# Historical Profit-Capture Missed Table",
        "",
        f"- Generated: `{table.get('generated_at_utc')}`",
        f"- Blocker: `{table.get('blocker')}`",
        f"- Missed captures: `{summary.get('missed_capture_count')}`",
        f"- Current-rule replay triggers: `{summary.get('repair_replay_triggered_count')}`",
        f"- Post-repair live missed count: `{summary.get('post_repair_live_missed_count')}`",
        "",
        "## Missed Captures",
        "",
        "| trade_id | pair | side | MFE | executable capture | missed amount JPY | close result | system/manual | allowed now | trigger |",
        "|---|---|---|---|---|---:|---|---|---|---|",
    ]
    for item in table.get("missed_captures") or []:
        mfe = item.get("max_favorable_excursion") or {}
        cap = item.get("executable_capture_price_time") or {}
        amt = item.get("missed_capture_amount") or {}
        close = item.get("current_close_result") or {}
        lines.append(
            f"| {item.get('trade_id')} | {item.get('pair')} | {item.get('side')} | {mfe.get('mfe_pips_before_loss_close')}p / {mfe.get('estimated_mfe_jpy_before_loss_close')} JPY @ {mfe.get('mfe_at_utc')} | {cap.get('time_utc')} @ {cap.get('price')} | {amt.get('repair_replay_net_improvement_jpy')} | {close.get('close_time_utc')} @ {close.get('close_price')} = {close.get('realized_pl_jpy')} ({close.get('close_reason')}) | {item.get('system_or_manual')} | {item.get('capture_would_be_allowed_under_current_rules')} | {item.get('exact_trigger_that_should_have_fired')} |"
        )
    lines.extend(
        [
            "",
            "## Gate Definitions",
            "",
        ]
    )
    for item in table.get("gate_definitions") or []:
        lines.append(f"- `{item.get('blocker_code')}`: {item.get('exact_clearing_condition')} Can create A/S permission: `{item.get('can_create_as_permission')}`")
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
        f"- A/S LIVE_READY path exists: `{board.get('as_live_ready_path_exists')}`",
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
            "## AUD_USD",
            "",
        ]
    )
    aud = board.get("aud_usd_rejection_state") or {}
    lines.extend(
        [
            f"- Classification: `{aud.get('classification')}`",
            f"- A/S candidate: `{aud.get('as_candidate')}`",
            f"- LIVE_READY allowed: `{aud.get('live_ready_allowed')}`",
            f"- Reason: `{aud.get('reason')}`",
            "",
            "## Month-Scale Residual Family Gate",
            "",
        ]
    )
    residual = board.get("month_scale_residual_family_gate") or {}
    lines.extend(
        [
            f"- Family count: `{residual.get('family_count')}`",
            f"- Priority families: `{residual.get('priority_family_count')}`",
            f"- Tail families: `{residual.get('tail_family_count')}`",
            f"- Before filters improved P/L JPY: `{residual.get('improved_pl_jpy')}`",
            f"- After filters residual P/L JPY: `{residual.get('residual_pl_jpy')}`",
            f"- Month-scale replay clears after filters: `{residual.get('clears_month_scale_tp_progress_replay_still_negative')}`",
            f"- All negative families can create live permission: `{not residual.get('all_negative_families_can_create_live_permission_false')}`",
            "",
            "| family | residual P/L | cause | blocker | action | A/S now | can ever A/S |",
            "|---|---:|---|---|---|---|---|",
        ]
    )
    for item in residual.get("priority_families") or []:
        lines.append(
            f"| {item.get('family')} | {item.get('residual_pl_jpy')} | `{item.get('cause')}` | `{item.get('current_blocker_code')}` | `{item.get('proposed_action')}` | `{item.get('can_create_live_permission')}` | `{item.get('can_ever_be_as')}` |"
        )
    lines.extend(
        [
            "",
            "## Repair Analysis Artifacts",
            "",
        ]
    )
    for item in board.get("repair_analysis_artifacts") or []:
        lines.append(f"- `{item}`")
    lines.extend(
        [
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
    lines.append("")
    lines.append("Blocker hierarchy:")
    for item in board.get("blocker_hierarchy") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Required sequence:")
    for item in (board.get("shortest_path") or {}).get("steps") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Funding / Manual Safety",
            "",
            f"- Funding-adjusted 30d multiplier: `{(board.get('funding_adjusted_30d_status') or {}).get('current_30d_multiplier')}`",
            f"- Current equity raw equals broker NAV: `{(board.get('funding_adjusted_30d_status') or {}).get('current_equity_raw_equals_broker_nav')}`",
            f"- Funding-adjusted multiplier is KPI: `{(board.get('funding_adjusted_30d_status') or {}).get('rolling_30d_multiplier_funding_adjusted_is_kpi')}`",
            f"- 100,000 JPY deposit excluded from performance: `{(board.get('funding_adjusted_30d_status') or {}).get('deposit_100000_jpy_excluded_from_performance')}`",
            f"- EUR_USD `{EUR_USD_MANUAL_TRADE_ID}` classification: `{(board.get('manual_eurusd_472987') or {}).get('classification')}`",
            f"- EUR_USD `{EUR_USD_MANUAL_TRADE_ID}` intent: `{(board.get('manual_eurusd_472987') or {}).get('management_intent')}`",
            f"- System P/L counted: `{(board.get('manual_eurusd_472987') or {}).get('system_pl_counted')}`",
            f"- System occupancy counted: `{(board.get('manual_eurusd_472987') or {}).get('system_occupancy_counted')}`",
            f"- Auto close allowed: `{(board.get('manual_eurusd_472987') or {}).get('auto_close_allowed')}`",
            f"- Auto SL attach allowed: `{(board.get('manual_eurusd_472987') or {}).get('auto_sl_attach_allowed')}`",
            f"- Auto TP modify allowed: `{(board.get('manual_eurusd_472987') or {}).get('auto_tp_modify_allowed')}`",
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


def _month_scale_replay_metrics(acceptance: dict[str, Any]) -> dict[str, Any]:
    findings = _findings_by_code(acceptance)
    evidence = _evidence(findings.get("MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"))
    if evidence:
        return evidence
    metrics = acceptance.get("metrics") if isinstance(acceptance.get("metrics"), dict) else {}
    replay = (
        metrics.get("profit_capture_replay_repair")
        if isinstance(metrics.get("profit_capture_replay_repair"), dict)
        else {}
    )
    return dict(replay)


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

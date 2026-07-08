"""Read-only HARVEST/RUNNER payoff-shape diagnosis.

This module does not call OANDA, stage orders, cancel orders, close positions,
modify launchd, or grant live permission. It combines existing closed-trade,
replay, timing-audit, and order-intent evidence into a machine-readable surface
for deciding whether the current 4x path is HARVEST-led, RUNNER-led, mixed, or
not tradable.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from quant_rabbit.capture_economics import (
    MIN_SAMPLE_FOR_VERDICT,
    RealizedOutcome,
    _ATTRIBUTED_REALIZED_SQL,
    _bucket_metrics,
    _lane_method,
)
from quant_rabbit.paths import (
    DEFAULT_CAPTURE_ECONOMICS,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_EXECUTION_TIMING_AUDIT,
    DEFAULT_MONTH_SCALE_TP_REPLAY_RESIDUALS,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS_REPORT,
DEFAULT_REPLAY_BACKTEST,
)


# Compact report caps. They only limit display size; the source artifacts retain
# the full evidence, and these constants are not trading thresholds.
ITEM_LIMIT = 12
EXAMPLE_LIMIT = 5

TAKE_PROFIT_EXIT_REASON = "TAKE_PROFIT_ORDER"
MARKET_CLOSE_EXIT_REASON = "MARKET_ORDER_TRADE_CLOSE"
STOP_LOSS_EXIT_REASON = "STOP_LOSS_ORDER"
MARGIN_CLOSEOUT_EXIT_REASON = "MARKET_ORDER_MARGIN_CLOSEOUT"
DEFAULT_EURUSD_SHORT_BREAKOUT_FAILURE_PROOF_FLOOR_UPDATE = (
    DEFAULT_PAYOFF_SHAPE_DIAGNOSIS.parent / "eurusd_short_breakout_failure_proof_floor_update.json"
)


@dataclass(frozen=True)
class PayoffShapeDiagnosisSummary:
    output_path: Path
    report_path: Path
    status: str
    overall_payoff_shape_verdict: str
    harvest_candidates: int
    runner_candidates: int
    partial_tp_runner_candidates: int
    no_trade_shapes: int


def build_payoff_shape_diagnosis(
    *,
    ledger_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
    capture_economics_path: Path = DEFAULT_CAPTURE_ECONOMICS,
    execution_timing_audit_path: Path = DEFAULT_EXECUTION_TIMING_AUDIT,
    order_intents_path: Path = DEFAULT_ORDER_INTENTS,
    replay_backtest_path: Path = DEFAULT_REPLAY_BACKTEST,
    month_scale_residuals_path: Path = DEFAULT_MONTH_SCALE_TP_REPLAY_RESIDUALS,
    proof_floor_update_path: Path = DEFAULT_EURUSD_SHORT_BREAKOUT_FAILURE_PROOF_FLOOR_UPDATE,
    output_path: Path = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS,
    report_path: Path = DEFAULT_PAYOFF_SHAPE_DIAGNOSIS_REPORT,
) -> PayoffShapeDiagnosisSummary:
    """Build and persist the payoff-shape diagnosis payload."""

    generated_at = datetime.now(timezone.utc).isoformat()
    realized = _load_realized_outcomes(ledger_path)
    capture = _load_json(capture_economics_path)
    timing = _load_json(execution_timing_audit_path)
    intents = _load_json(order_intents_path)
    replay = _load_json(replay_backtest_path)
    month_scale = _load_json(month_scale_residuals_path)
    proof_floor_update = _load_json(proof_floor_update_path)

    shape_rows = _group_realized(realized, lambda row: _shape_key(row.pair, row.side, row.method))
    family_stats = _family_stats(realized, month_scale)
    pair_stats = _pair_stats(realized, month_scale)
    session_stats, intent_replay_packets, intent_blocker_shapes = _session_stats_and_replay(intents)
    month_scale_blocks = _month_scale_shape_blocks(month_scale)

    timing_runner_cases = _missed_runner_cases(timing)
    overheld_harvest_cases = _overheld_harvest_cases(timing)
    proof_reconciliation = _canonical_proof_reconciliation(proof_floor_update)
    harvest_candidates = _apply_canonical_proof_reconciliation(
        _harvest_candidates(shape_rows, month_scale_blocks, timing),
        proof_reconciliation,
    )
    runner_candidates = _runner_candidates(shape_rows, timing_runner_cases, month_scale_blocks)
    partial_candidates = _partial_tp_runner_candidates(
        harvest_candidates=harvest_candidates,
        runner_cases=timing_runner_cases,
        shape_rows=shape_rows,
        month_scale_blocks=month_scale_blocks,
    )
    no_trade_shapes = _no_trade_shapes(
        shape_rows=shape_rows,
        month_scale=month_scale,
        month_scale_blocks=month_scale_blocks,
        intent_blocker_shapes=intent_blocker_shapes,
        intent_replay_packets=intent_replay_packets,
    )
    mfe_mae_summary = _mfe_mae_summary(
        timing=timing,
        intent_replay_packets=intent_replay_packets,
    )
    verdict = _overall_verdict(
        capture=capture,
        replay=replay,
        month_scale=month_scale,
        harvest_candidates=harvest_candidates,
        runner_candidates=runner_candidates,
        partial_candidates=partial_candidates,
    )
    recommendations = _recommendations(
        verdict=verdict,
        capture=capture,
        replay=replay,
        month_scale=month_scale,
        harvest_candidates=harvest_candidates,
        runner_candidates=runner_candidates,
        partial_candidates=partial_candidates,
        no_trade_shapes=no_trade_shapes,
    )
    next_actions = _next_evidence_actions(
        month_scale=month_scale,
        harvest_candidates=harvest_candidates,
        runner_candidates=runner_candidates,
        no_trade_shapes=no_trade_shapes,
    )

    status = "OK"
    missing_sources = [
        str(path)
        for path, payload in (
            (capture_economics_path, capture),
            (execution_timing_audit_path, timing),
            (order_intents_path, intents),
            (replay_backtest_path, replay),
            (month_scale_residuals_path, month_scale),
        )
        if not payload
    ]
    if not realized or missing_sources:
        status = "PARTIAL_DATA"

    payload = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "status": status,
        "source_artifacts": {
            "execution_ledger_db": str(ledger_path),
            "capture_economics": str(capture_economics_path),
            "execution_timing_audit": str(execution_timing_audit_path),
            "order_intents": str(order_intents_path),
            "replay_backtest": str(replay_backtest_path),
            "month_scale_tp_replay_residuals": str(month_scale_residuals_path),
            "proof_floor_update": str(proof_floor_update_path),
        },
        "missing_source_artifacts": missing_sources,
        "safety_contract": {
            "read_only": True,
            "no_live_order": True,
            "no_cancel": True,
            "no_close": True,
            "no_launchd_change": True,
            "no_gate_relaxation": True,
            "proof_queue_count_is_not_live_permission": True,
            "no_4x_deficit_lot_backsolve": True,
            "negative_expectancy_not_hidden": True,
        },
        "overall_payoff_shape_verdict": verdict,
        "canonical_proof_reconciliation": proof_reconciliation,
        "harvest_candidates": harvest_candidates,
        "runner_candidates": runner_candidates,
        "partial_tp_runner_candidates": partial_candidates,
        "no_trade_shapes": no_trade_shapes,
        "family_stats": family_stats,
        "pair_stats": pair_stats,
        "session_stats": session_stats,
        "mfe_mae_summary": mfe_mae_summary,
        "missed_runner_cases": timing_runner_cases,
        "overheld_harvest_cases": overheld_harvest_cases,
        "payoff_shape_recommendations": recommendations,
        "next_evidence_actions": next_actions,
        "live_side_effects": [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _write_report(payload, report_path)

    return PayoffShapeDiagnosisSummary(
        output_path=output_path,
        report_path=report_path,
        status=status,
        overall_payoff_shape_verdict=verdict["classification"],
        harvest_candidates=len(harvest_candidates),
        runner_candidates=len(runner_candidates),
        partial_tp_runner_candidates=len(partial_candidates),
        no_trade_shapes=len(no_trade_shapes),
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_realized_outcomes(path: Path) -> list[RealizedOutcome]:
    if not path.exists():
        return []
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            rows = conn.execute(_ATTRIBUTED_REALIZED_SQL).fetchall()
    except sqlite3.Error:
        return []
    outcomes: list[RealizedOutcome] = []
    for ts, trade_id, pair, side, lane_id, reason, pl in rows:
        if pl is None:
            continue
        outcomes.append(
            RealizedOutcome(
                ts_utc=str(ts or ""),
                trade_id=str(trade_id or ""),
                pair=str(pair or "UNKNOWN"),
                side=str(side or "UNKNOWN"),
                lane_id=str(lane_id or ""),
                method=_lane_method(str(lane_id or "")),
                exit_reason=str(reason or "UNKNOWN"),
                realized_pl_jpy=float(pl),
            )
        )
    return outcomes


def _shape_key(pair: str, side: str, method: str) -> str:
    return f"{str(pair or 'UNKNOWN').upper()}|{str(side or 'UNKNOWN').upper()}|{str(method or 'UNKNOWN').upper()}"


def _split_shape_key(key: str) -> tuple[str, str, str]:
    parts = (key or "UNKNOWN|UNKNOWN|UNKNOWN").split("|")
    parts = (parts + ["UNKNOWN", "UNKNOWN", "UNKNOWN"])[:3]
    return parts[0], parts[1], parts[2]


def _group_realized(
    rows: Iterable[RealizedOutcome],
    key_fn: Any,
) -> dict[str, list[RealizedOutcome]]:
    groups: dict[str, list[RealizedOutcome]] = {}
    for row in rows:
        groups.setdefault(str(key_fn(row)), []).append(row)
    return groups


def _stats_from_rows(rows: list[RealizedOutcome]) -> dict[str, Any]:
    metrics = _bucket_metrics(rows)
    by_exit = {
        reason: _bucket_metrics([row for row in rows if row.exit_reason == reason])
        for reason in sorted({row.exit_reason for row in rows})
    }
    metrics["by_exit_reason"] = by_exit
    metrics["take_profit_order"] = by_exit.get(TAKE_PROFIT_EXIT_REASON, {"trades": 0})
    metrics["market_order_trade_close"] = by_exit.get(MARKET_CLOSE_EXIT_REASON, {"trades": 0})
    metrics["stop_loss_order"] = by_exit.get(STOP_LOSS_EXIT_REASON, {"trades": 0})
    metrics["margin_closeout"] = by_exit.get(MARGIN_CLOSEOUT_EXIT_REASON, {"trades": 0})
    return metrics


def _family_stats(realized: list[RealizedOutcome], month_scale: dict[str, Any]) -> dict[str, Any]:
    groups = _group_realized(realized, lambda row: row.method or "UNKNOWN")
    residual_by_method = {
        str(row.get("method") or row.get("strategy") or "UNKNOWN").upper(): row
        for row in _as_list(month_scale.get("method_rollups"))
    }
    out: dict[str, Any] = {}
    for method, rows in sorted(groups.items()):
        stats = _stats_from_rows(rows)
        residual = residual_by_method.get(method.upper())
        out[method] = {
            "closed_trade_stats": _compact_metrics(stats),
            "take_profit_order": stats.get("take_profit_order"),
            "market_order_trade_close": stats.get("market_order_trade_close"),
            "stop_loss_order": stats.get("stop_loss_order"),
            "month_scale_residual": _compact_residual(residual),
            "expectancy_breakdown": _family_expectancy_breakdown(stats, residual),
        }
    return out


def _pair_stats(realized: list[RealizedOutcome], month_scale: dict[str, Any]) -> dict[str, Any]:
    groups = _group_realized(realized, lambda row: row.pair or "UNKNOWN")
    residual_pairs: dict[str, list[dict[str, Any]]] = {}
    for row in _as_list(month_scale.get("residual_losing_families")):
        residual_pairs.setdefault(str(row.get("pair") or "UNKNOWN").upper(), []).append(row)
    out: dict[str, Any] = {}
    for pair, rows in sorted(groups.items()):
        stats = _stats_from_rows(rows)
        out[pair] = {
            "closed_trade_stats": _compact_metrics(stats),
            "take_profit_order": stats.get("take_profit_order"),
            "market_order_trade_close": stats.get("market_order_trade_close"),
            "stop_loss_order": stats.get("stop_loss_order"),
            "month_scale_residual_net_jpy": _round(sum(_num(r.get("actual_pl_jpy")) for r in residual_pairs.get(pair.upper(), []))),
            "month_scale_residual_families": [
                _compact_residual(row) for row in residual_pairs.get(pair.upper(), [])[:EXAMPLE_LIMIT]
            ],
        }
    return out


def _session_stats_and_replay(
    order_intents: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sessions: dict[str, dict[str, Any]] = {}
    replay_packets: list[dict[str, Any]] = []
    blocker_shapes: list[dict[str, Any]] = []

    for result in _as_list(order_intents.get("results")):
        intent = result.get("intent") if isinstance(result, dict) else None
        if not isinstance(intent, dict):
            continue
        meta = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        pair = str(intent.get("pair") or meta.get("pair") or "UNKNOWN").upper()
        side = str(intent.get("side") or meta.get("side") or "UNKNOWN").upper()
        market_context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
        method = str(
            meta.get("method")
            or market_context.get("method")
            or _lane_method(str(result.get("lane_id") or ""))
            or "UNKNOWN"
        ).upper()
        session = str(meta.get("session_current_tag") or meta.get("session_bucket") or "UNKNOWN").upper()
        status = str(result.get("status") or "UNKNOWN").upper()
        live_blocker_codes = [str(code) for code in _as_list(result.get("live_blocker_codes"))]
        risk_issues = [
            str(issue.get("code") or "")
            for issue in _as_list(result.get("risk_issues"))
            if isinstance(issue, dict)
        ]
        codes = [code for code in live_blocker_codes + risk_issues if code]
        session_row = sessions.setdefault(
            session,
            {
                "candidate_count": 0,
                "risk_allowed_count": 0,
                "live_ready_count": 0,
                "status_counts": Counter(),
                "blocker_codes": Counter(),
                "payoff_shape_modes": Counter(),
                "reward_risk_values": [],
                "risk_jpy_values": [],
                "examples": [],
            },
        )
        session_row["candidate_count"] += 1
        if result.get("risk_allowed") is True:
            session_row["risk_allowed_count"] += 1
        if status == "LIVE_READY":
            session_row["live_ready_count"] += 1
        session_row["status_counts"][status] += 1
        session_row["blocker_codes"].update(codes)
        shape_mode = str(meta.get("tp_target_intent") or meta.get("opportunity_mode") or "UNKNOWN").upper()
        session_row["payoff_shape_modes"][shape_mode] += 1
        metrics = result.get("risk_metrics") if isinstance(result.get("risk_metrics"), dict) else {}
        reward_risk = _maybe_num(metrics.get("reward_risk"))
        risk_jpy = _maybe_num(metrics.get("risk_jpy"))
        if reward_risk is not None:
            session_row["reward_risk_values"].append(reward_risk)
        if risk_jpy is not None:
            session_row["risk_jpy_values"].append(risk_jpy)
        if len(session_row["examples"]) < EXAMPLE_LIMIT:
            session_row["examples"].append(
                {
                    "lane_id": result.get("lane_id"),
                    "pair": pair,
                    "side": side,
                    "method": method,
                    "status": status,
                    "live_blocker_codes": live_blocker_codes[:6],
                    "tp_target_intent": meta.get("tp_target_intent"),
                    "tp_execution_mode": meta.get("tp_execution_mode"),
                }
            )

        for packet in _extract_replay_packets(meta):
            packet.update({"pair": pair, "side": side, "method": method, "session": session, "lane_id": result.get("lane_id")})
            replay_packets.append(packet)
            if _num(packet.get("avg_final_pips")) < 0 or packet.get("blocks_live_support") is True:
                blocker_shapes.append(
                    {
                        "source": "order_intents",
                        "reason_code": "SPREAD_INCLUDED_REPLAY_NEGATIVE",
                        "pair": pair,
                        "side": side,
                        "method": method,
                        "session": session,
                        "lane_id": result.get("lane_id"),
                        "avg_final_pips": packet.get("avg_final_pips"),
                        "avg_mfe_pips": packet.get("avg_mfe_pips"),
                        "avg_mae_pips": packet.get("avg_mae_pips"),
                        "samples": packet.get("samples"),
                        "live_promotion_allowed": False,
                    }
                )

        hard_negative_codes = [
            code
            for code in codes
            if code
            in {
                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                "MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED",
                "MONTH_SCALE_RESIDUAL_LOSS_REPAIR_BLOCKED",
                "STRATEGY_NOT_ELIGIBLE",
            }
        ]
        if hard_negative_codes:
            blocker_shapes.append(
                {
                    "source": "order_intents",
                    "reason_code": "CURRENT_INTENT_BLOCKED",
                    "pair": pair,
                    "side": side,
                    "method": method,
                    "session": session,
                    "lane_id": result.get("lane_id"),
                    "blocker_codes": sorted(set(hard_negative_codes)),
                    "status": status,
                    "live_promotion_allowed": False,
                }
            )

    compact: dict[str, Any] = {}
    for session, row in sorted(sessions.items()):
        compact[session] = {
            "candidate_count": row["candidate_count"],
            "risk_allowed_count": row["risk_allowed_count"],
            "live_ready_count": row["live_ready_count"],
            "status_counts": dict(row["status_counts"].most_common()),
            "top_blocker_codes": dict(row["blocker_codes"].most_common(10)),
            "payoff_shape_modes": dict(row["payoff_shape_modes"].most_common()),
            "avg_reward_risk": _avg(row["reward_risk_values"]),
            "avg_risk_jpy": _avg(row["risk_jpy_values"]),
            "examples": row["examples"],
        }
    return compact, replay_packets, blocker_shapes


def _extract_replay_packets(meta: dict[str, Any]) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            has_replay_shape = any(
                key in value
                for key in (
                    "avg_mfe_pips",
                    "avg_mae_pips",
                    "avg_final_pips",
                    "optimized_profit_factor",
                    "directional_hit_rate",
                )
            )
            if has_replay_shape:
                packets.append(
                    {
                        "source_key": path,
                        "name": value.get("name"),
                        "avg_mfe_pips": _maybe_num(value.get("avg_mfe_pips")),
                        "avg_mae_pips": _maybe_num(value.get("avg_mae_pips")),
                        "avg_final_pips": _maybe_num(value.get("avg_final_pips")),
                        "optimized_profit_factor": _maybe_num(value.get("optimized_profit_factor")),
                        "optimized_win_rate": _maybe_num(value.get("optimized_win_rate")),
                        "samples": _maybe_int(value.get("samples")),
                        "active_days": _maybe_int(value.get("active_days")),
                        "live_grade": value.get("live_grade"),
                        "blocks_live_support": value.get("blocks_live_support"),
                        "adoption_status": value.get("adoption_status"),
                    }
                )
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    walk(child, f"{path}[{index}]")

    walk(meta, "")
    return packets


def _harvest_candidates(
    shape_rows: dict[str, list[RealizedOutcome]],
    month_scale_blocks: dict[str, dict[str, Any]],
    timing: dict[str, Any],
) -> list[dict[str, Any]]:
    canceled_rollups = {
        _shape_key(row.get("pair"), row.get("side"), row.get("method")): row
        for row in _as_list((timing.get("canceled_order_regret_by_shape") or {}).get("items"))
        if isinstance(row, dict)
    }
    candidates: list[dict[str, Any]] = []
    for key, rows in shape_rows.items():
        pair, side, method = _split_shape_key(key)
        stats = _stats_from_rows(rows)
        tp = stats.get("take_profit_order") or {}
        tp_trades = int(tp.get("trades") or 0)
        tp_losses = int(tp.get("losses") or 0)
        tp_expectancy = _num(tp.get("expectancy_jpy_per_trade"))
        tp_avg_win = _num(tp.get("avg_win_jpy"))
        if tp_trades <= 0 or tp_losses > 0 or tp_expectancy <= 0 or tp_avg_win <= 0:
            continue
        month_block = month_scale_blocks.get(key)
        proof_gap = max(0, MIN_SAMPLE_FOR_VERDICT - tp_trades)
        candidates.append(
            {
                "shape_key": key,
                "pair": pair,
                "side": side,
                "method": method,
                "classification": "HARVEST_POSITIVE_TP_PROVEN" if proof_gap == 0 else "HARVEST_POSITIVE_THIN_SAMPLE",
                "live_promotion_allowed": False,
                "live_permission_reason": (
                    "blocked_by_month_scale_replay"
                    if month_block
                    else "diagnostic_only_requires_current_forecast_spread_strategy_margin_gateway"
                ),
                "take_profit_trades": tp_trades,
                "take_profit_wins": int(tp.get("wins") or 0),
                "take_profit_losses": tp_losses,
                "take_profit_expectancy_jpy": tp.get("expectancy_jpy_per_trade"),
                "take_profit_avg_win_jpy": tp.get("avg_win_jpy"),
                "take_profit_net_jpy": tp.get("net_jpy"),
                "proof_floor_trades": MIN_SAMPLE_FOR_VERDICT,
                "proof_gap_trades": proof_gap,
                "overall_net_jpy": stats.get("net_jpy"),
                "overall_expectancy_jpy_per_trade": stats.get("expectancy_jpy_per_trade"),
                "market_close_net_jpy": (stats.get("market_order_trade_close") or {}).get("net_jpy"),
                "market_close_losses": (stats.get("market_order_trade_close") or {}).get("losses"),
                "month_scale_blocker": _compact_residual(month_block),
                "timing_regret": _compact_timing_rollup(canceled_rollups.get(key)),
                "evidence_refs": [
                    f"capture:shape:{pair}:{side}:{method}:TAKE_PROFIT_ORDER",
                    "data/capture_economics.json",
                    "data/execution_timing_audit.json",
                ],
            }
        )
    return sorted(
        candidates,
        key=lambda item: (
            int(item.get("proof_gap_trades") or 0),
            -_num(item.get("take_profit_net_jpy")),
            str(item.get("shape_key")),
        ),
    )[:ITEM_LIMIT]


def _canonical_proof_reconciliation(proof_floor_update: dict[str, Any]) -> dict[str, Any]:
    if not proof_floor_update:
        return {
            "status": "MISSING",
            "applied": False,
            "live_permission_allowed": False,
            "live_side_effects": [],
        }
    target_shape = str(proof_floor_update.get("target_shape") or "")
    post = proof_floor_update.get("post_update_tp_proof") if isinstance(proof_floor_update.get("post_update_tp_proof"), dict) else {}
    pre = proof_floor_update.get("pre_update_tp_proof") if isinstance(proof_floor_update.get("pre_update_tp_proof"), dict) else {}
    accepted = _as_list(proof_floor_update.get("accepted_sample_checks"))
    accepted_ids = [str(row.get("trade_id")) for row in accepted if isinstance(row, dict) and row.get("trade_id")]
    accepted_net = _round(sum(_num(row.get("realized_pl_jpy")) for row in accepted if isinstance(row, dict)))
    post_wins = _maybe_int(post.get("wins"))
    post_losses = _maybe_int(post.get("losses"))
    post_gap = _maybe_int(post.get("remaining_samples"))
    proof_floor = _maybe_int(post.get("proof_floor")) or _maybe_int(pre.get("proof_floor")) or MIN_SAMPLE_FOR_VERDICT
    checks = proof_floor_update.get("required_checks") if isinstance(proof_floor_update.get("required_checks"), dict) else {}
    required_checks_passed = bool(checks) and all(
        isinstance(row, dict) and row.get("passed") is True
        for row in checks.values()
    )
    applied = (
        proof_floor_update.get("read_only") is True
        and proof_floor_update.get("live_permission_allowed") is False
        and target_shape == "EUR_USD|SHORT|BREAKOUT_FAILURE"
        and bool(post.get("proof_floor_reached"))
        and (post_wins or 0) >= proof_floor
        and (post_losses or 0) == 0
        and required_checks_passed
    )
    return {
        "status": "APPLIED_TO_PAYOFF_SHAPE_DIAGNOSIS" if applied else "NOT_APPLIED",
        "applied": applied,
        "source_artifact": "data/eurusd_short_breakout_failure_proof_floor_update.json",
        "target_shape": target_shape or None,
        "canonical_integration_status": proof_floor_update.get("canonical_integration_status"),
        "accepted_legacy_sample_trade_ids": accepted_ids,
        "accepted_legacy_sample_net_jpy": accepted_net,
        "pre_update_tp_proof": pre,
        "post_update_tp_proof": post,
        "required_checks_passed": required_checks_passed,
        "read_only": True,
        "live_permission_allowed": False,
        "live_side_effects": [],
        "safety_note": (
            "This reconciles broad broker TAKE_PROFIT_ORDER proof only. It does not make the exact "
            "LIMIT/HARVEST vehicle live-grade, does not mix MARKET/STOP rows into LIMIT proof, and does "
            "not clear risk, verifier, gateway, guardian, operator, or negative-expectancy blockers."
        ),
    }


def _apply_canonical_proof_reconciliation(
    candidates: list[dict[str, Any]],
    reconciliation: dict[str, Any],
) -> list[dict[str, Any]]:
    if not reconciliation.get("applied"):
        return candidates
    target = reconciliation.get("target_shape")
    accepted_count = len(_as_list(reconciliation.get("accepted_legacy_sample_trade_ids")))
    accepted_net = _num(reconciliation.get("accepted_legacy_sample_net_jpy"))
    post = reconciliation.get("post_update_tp_proof") if isinstance(reconciliation.get("post_update_tp_proof"), dict) else {}
    updated: list[dict[str, Any]] = []
    for row in candidates:
        if row.get("shape_key") != target:
            updated.append(row)
            continue
        wins = _maybe_int(post.get("wins")) or int(row.get("take_profit_wins") or row.get("take_profit_trades") or 0)
        losses = _maybe_int(post.get("losses")) or 0
        proof_floor = _maybe_int(post.get("proof_floor")) or MIN_SAMPLE_FOR_VERDICT
        proof_gap = _maybe_int(post.get("remaining_samples"))
        if proof_gap is None:
            proof_gap = max(0, proof_floor - wins)
        original_net = _num(row.get("take_profit_net_jpy"))
        reconciled_net = _round(original_net + accepted_net)
        expectancy = _round((reconciled_net or 0.0) / wins) if wins > 0 and reconciled_net is not None else row.get("take_profit_expectancy_jpy")
        new_row = dict(row)
        new_row.update(
            {
                "classification": "HARVEST_PROOF_FLOOR_REACHED_EVIDENCE_ONLY" if proof_gap == 0 else row.get("classification"),
                "take_profit_trades": wins,
                "take_profit_wins": wins,
                "take_profit_losses": losses,
                "take_profit_net_jpy": reconciled_net,
                "take_profit_expectancy_jpy": expectancy,
                "take_profit_avg_win_jpy": expectancy if losses == 0 else row.get("take_profit_avg_win_jpy"),
                "proof_floor_trades": proof_floor,
                "proof_gap_trades": proof_gap,
                "canonical_proof_reconciliation": {
                    "source_artifact": reconciliation.get("source_artifact"),
                    "accepted_legacy_sample_trade_ids": reconciliation.get("accepted_legacy_sample_trade_ids"),
                    "accepted_legacy_sample_count": accepted_count,
                    "accepted_legacy_sample_net_jpy": reconciliation.get("accepted_legacy_sample_net_jpy"),
                    "canonical_integration_status": reconciliation.get("canonical_integration_status"),
                    "scope": "broad_take_profit_order_proof_only_not_exact_limit_sample_floor",
                },
            }
        )
        refs = list(new_row.get("evidence_refs") or [])
        refs.append(str(reconciliation.get("source_artifact")))
        new_row["evidence_refs"] = sorted(set(ref for ref in refs if ref))
        updated.append(new_row)
    return sorted(
        updated,
        key=lambda item: (
            int(item.get("proof_gap_trades") or 0),
            -_num(item.get("take_profit_net_jpy")),
            str(item.get("shape_key")),
        ),
    )[:ITEM_LIMIT]


def _runner_candidates(
    shape_rows: dict[str, list[RealizedOutcome]],
    runner_cases: list[dict[str, Any]],
    month_scale_blocks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    case_counts = Counter(str(case.get("shape_key") or "") for case in runner_cases)
    case_tail_jpy: Counter[str] = Counter()
    for case in runner_cases:
        case_tail_jpy[str(case.get("shape_key") or "")] += _num(case.get("estimated_post_close_favorable_jpy"))

    candidates: list[dict[str, Any]] = []
    for key, rows in shape_rows.items():
        pair, side, method = _split_shape_key(key)
        stats = _stats_from_rows(rows)
        if _num(stats.get("net_jpy")) <= 0:
            continue
        avg_win = _num(stats.get("avg_win_jpy"))
        avg_loss = _num(stats.get("avg_loss_jpy"))
        payoff = _maybe_num(stats.get("payoff_ratio"))
        breakeven = _maybe_num(stats.get("breakeven_payoff_at_win_rate"))
        clearly_beats_loss = avg_loss > 0 and avg_win > avg_loss
        clears_breakeven = payoff is not None and breakeven is not None and payoff > breakeven
        has_runner_tail = case_counts.get(key, 0) > 0
        if not (clearly_beats_loss and clears_breakeven and has_runner_tail):
            continue
        month_block = month_scale_blocks.get(key)
        candidates.append(
            {
                "shape_key": key,
                "pair": pair,
                "side": side,
                "method": method,
                "classification": "RUNNER_POSITIVE_WITH_MFE_EXTENSION",
                "live_promotion_allowed": False,
                "live_permission_reason": (
                    "blocked_by_month_scale_replay"
                    if month_block
                    else "diagnostic_only_runner_requires_current_forecast_and_gateway_validation"
                ),
                "trades": stats.get("trades"),
                "net_jpy": stats.get("net_jpy"),
                "expectancy_jpy_per_trade": stats.get("expectancy_jpy_per_trade"),
                "avg_win_jpy": stats.get("avg_win_jpy"),
                "avg_loss_jpy": stats.get("avg_loss_jpy"),
                "payoff_ratio": stats.get("payoff_ratio"),
                "breakeven_payoff_at_win_rate": stats.get("breakeven_payoff_at_win_rate"),
                "missed_runner_case_count": int(case_counts.get(key, 0)),
                "estimated_tail_jpy": _round(case_tail_jpy.get(key, 0.0)),
                "month_scale_blocker": _compact_residual(month_block),
                "evidence_refs": [
                    f"capture:shape:{pair}:{side}:{method}",
                    f"timing:runner:{pair}:{side}:{method}",
                ],
            }
        )
    return sorted(candidates, key=lambda item: (-_num(item.get("estimated_tail_jpy")), str(item.get("shape_key"))))[:ITEM_LIMIT]


def _partial_tp_runner_candidates(
    *,
    harvest_candidates: list[dict[str, Any]],
    runner_cases: list[dict[str, Any]],
    shape_rows: dict[str, list[RealizedOutcome]],
    month_scale_blocks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    cases_by_shape: dict[str, list[dict[str, Any]]] = {}
    for case in runner_cases:
        cases_by_shape.setdefault(str(case.get("shape_key") or ""), []).append(case)
    candidates: list[dict[str, Any]] = []
    for harvest in harvest_candidates:
        key = str(harvest.get("shape_key") or "")
        rows = shape_rows.get(key, [])
        stats = _stats_from_rows(rows)
        market = stats.get("market_order_trade_close") or {}
        runner_tail_cases = cases_by_shape.get(key, [])
        has_tail = bool(runner_tail_cases)
        has_market_close_leak = _num(market.get("net_jpy")) < 0
        if not (has_tail or has_market_close_leak):
            continue
        candidates.append(
            {
                "shape_key": key,
                "pair": harvest.get("pair"),
                "side": harvest.get("side"),
                "method": harvest.get("method"),
                "classification": "PARTIAL_TP_THEN_SMALL_RUNNER_DIAGNOSTIC",
                "live_promotion_allowed": False,
                "tp1_evidence": {
                    "take_profit_trades": harvest.get("take_profit_trades"),
                    "take_profit_losses": harvest.get("take_profit_losses"),
                    "take_profit_expectancy_jpy": harvest.get("take_profit_expectancy_jpy"),
                    "take_profit_avg_win_jpy": harvest.get("take_profit_avg_win_jpy"),
                    "proof_gap_trades": harvest.get("proof_gap_trades"),
                },
                "runner_tail_case_count": len(runner_tail_cases),
                "runner_tail_estimated_jpy": _round(sum(_num(case.get("estimated_post_close_favorable_jpy")) for case in runner_tail_cases)),
                "market_close_leak_net_jpy": market.get("net_jpy"),
                "market_close_losses": market.get("losses"),
                "risk_reduction_logic": (
                    "TP1 must reduce realized downside first; the runner leg is diagnostic only until "
                    "month-scale replay and close-gate leakage are non-negative."
                ),
                "month_scale_blocker": _compact_residual(month_scale_blocks.get(key)),
                "evidence_refs": [
                    f"capture:shape:{key}:TAKE_PROFIT_ORDER",
                    f"timing:runner:{key}",
                ],
            }
        )
    return sorted(
        candidates,
        key=lambda item: (
            -_num(item.get("runner_tail_estimated_jpy")),
            _num(item.get("market_close_leak_net_jpy")),
            str(item.get("shape_key")),
        ),
    )[:ITEM_LIMIT]


def _missed_runner_cases(timing: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in _as_list(timing.get("market_close_counterfactuals")):
        if not isinstance(row, dict):
            continue
        label = str(row.get("post_close_path_label") or "")
        if label != "PROFIT_CLOSE_LEFT_RUNNER_UPSIDE" and row.get("tp_touched_after_market_close") is not True:
            continue
        method = _lane_method(str(row.get("lane_id") or ""))
        key = _shape_key(row.get("pair"), row.get("side"), method)
        cases.append(
            {
                "shape_key": key,
                "trade_id": row.get("trade_id"),
                "lane_id": row.get("lane_id"),
                "pair": row.get("pair"),
                "side": row.get("side"),
                "method": method,
                "close_at_utc": row.get("close_at_utc"),
                "realized_pl_jpy": row.get("realized_pl_jpy"),
                "post_close_path_label": label,
                "post_close_favorable_pips": row.get("post_close_favorable_pips"),
                "estimated_post_close_favorable_jpy": row.get("estimated_post_close_favorable_jpy"),
                "post_close_adverse_pips": row.get("post_close_adverse_pips"),
                "estimated_post_close_adverse_jpy": row.get("estimated_post_close_adverse_jpy"),
                "tp_touched_after_market_close": row.get("tp_touched_after_market_close"),
                "tp_touch_minutes_after_market_close": row.get("tp_touch_minutes_after_market_close"),
                "evidence_ref": "timing:market_close_counterfactuals",
            }
        )
    return sorted(cases, key=lambda item: -_num(item.get("estimated_post_close_favorable_jpy")))[:ITEM_LIMIT]


def _overheld_harvest_cases(timing: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in _as_list(timing.get("loss_close_regrets")):
        if not isinstance(row, dict):
            continue
        had_positive = row.get("had_positive_mfe_before_loss_close") is True
        missed_capture = row.get("profit_capture_missed_before_loss_close") is True
        repair_triggered = row.get("repair_replay_triggered_before_loss_close") is True
        if not (missed_capture or repair_triggered or had_positive):
            continue
        method = _lane_method(str(row.get("lane_id") or ""))
        key = _shape_key(row.get("pair"), row.get("side"), method)
        cases.append(
            {
                "shape_key": key,
                "trade_id": row.get("trade_id"),
                "lane_id": row.get("lane_id"),
                "pair": row.get("pair"),
                "side": row.get("side"),
                "method": method,
                "exit_reason": row.get("exit_reason"),
                "realized_pl_jpy": row.get("realized_pl_jpy"),
                "mfe_pips_before_loss_close": row.get("mfe_pips_before_loss_close"),
                "estimated_mfe_jpy_before_loss_close": row.get("estimated_mfe_jpy_before_loss_close"),
                "tp_progress_before_loss_close": row.get("tp_progress_before_loss_close"),
                "decision_lag_minutes_after_first_positive": row.get("decision_lag_minutes_after_first_positive"),
                "profit_capture_missed_before_loss_close": missed_capture,
                "repair_replay_triggered_before_loss_close": repair_triggered,
                "repair_replay_counterfactual_net_improvement_jpy": row.get("repair_replay_counterfactual_net_improvement_jpy"),
                "repair_replay_block_reason": row.get("repair_replay_block_reason"),
                "evidence_ref": "timing:loss_close_regrets",
            }
        )
    return sorted(cases, key=lambda item: (_num(item.get("realized_pl_jpy")), -_num(item.get("estimated_mfe_jpy_before_loss_close"))))[:ITEM_LIMIT]


def _no_trade_shapes(
    *,
    shape_rows: dict[str, list[RealizedOutcome]],
    month_scale: dict[str, Any],
    month_scale_blocks: dict[str, dict[str, Any]],
    intent_blocker_shapes: list[dict[str, Any]],
    intent_replay_packets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    no_trade: dict[str, dict[str, Any]] = {}

    for key, rows in shape_rows.items():
        pair, side, method = _split_shape_key(key)
        stats = _stats_from_rows(rows)
        tp = stats.get("take_profit_order") or {}
        tp_positive = int(tp.get("trades") or 0) > 0 and int(tp.get("losses") or 0) == 0 and _num(tp.get("expectancy_jpy_per_trade")) > 0
        if _num(stats.get("net_jpy")) < 0 and not tp_positive:
            no_trade[key] = {
                "shape_key": key,
                "pair": pair,
                "side": side,
                "method": method,
                "reason_code": "REALIZED_NEGATIVE_NO_POSITIVE_TP_SHAPE",
                "net_jpy": stats.get("net_jpy"),
                "expectancy_jpy_per_trade": stats.get("expectancy_jpy_per_trade"),
                "trades": stats.get("trades"),
                "live_promotion_allowed": False,
                "evidence_refs": ["data/execution_ledger.db", "data/capture_economics.json"],
            }

    for key, residual in month_scale_blocks.items():
        pair, side, method = _split_shape_key(key)
        row = no_trade.setdefault(
            key,
            {
                "shape_key": key,
                "pair": pair,
                "side": side,
                "method": method,
                "reason_code": "MONTH_SCALE_REPLAY_NEGATIVE",
                "live_promotion_allowed": False,
                "evidence_refs": [],
            },
        )
        row["month_scale_blocker"] = _compact_residual(residual)
        row["reason_code"] = "MONTH_SCALE_REPLAY_NEGATIVE"
        row.setdefault("evidence_refs", []).append("data/month_scale_tp_replay_residuals.json")

    for blocked in intent_blocker_shapes:
        key = _shape_key(blocked.get("pair"), blocked.get("side"), blocked.get("method"))
        row = no_trade.setdefault(
            key,
            {
                "shape_key": key,
                "pair": blocked.get("pair"),
                "side": blocked.get("side"),
                "method": blocked.get("method"),
                "reason_code": str(blocked.get("reason_code") or "CURRENT_INTENT_BLOCKED"),
                "live_promotion_allowed": False,
                "evidence_refs": [],
            },
        )
        row.setdefault("current_intent_blockers", []).append(_drop_none(blocked))
        row.setdefault("evidence_refs", []).append("data/order_intents.json")

    for packet in intent_replay_packets:
        if _num(packet.get("avg_final_pips")) >= 0 and packet.get("blocks_live_support") is not True:
            continue
        key = _shape_key(packet.get("pair"), packet.get("side"), packet.get("method"))
        row = no_trade.setdefault(
            key,
            {
                "shape_key": key,
                "pair": packet.get("pair"),
                "side": packet.get("side"),
                "method": packet.get("method"),
                "reason_code": "SPREAD_INCLUDED_REPLAY_NEGATIVE",
                "live_promotion_allowed": False,
                "evidence_refs": [],
            },
        )
        row.setdefault("negative_replay_packets", []).append(
            {
                "name": packet.get("name"),
                "session": packet.get("session"),
                "avg_final_pips": packet.get("avg_final_pips"),
                "avg_mfe_pips": packet.get("avg_mfe_pips"),
                "avg_mae_pips": packet.get("avg_mae_pips"),
                "samples": packet.get("samples"),
                "adoption_status": packet.get("adoption_status"),
            }
        )
        row.setdefault("evidence_refs", []).append("data/order_intents.json")

    if month_scale.get("fresh_entries_must_remain_blocked") is True:
        no_trade["GLOBAL|ALL|MONTH_SCALE"] = {
            "shape_key": "GLOBAL|ALL|MONTH_SCALE",
            "pair": "GLOBAL",
            "side": "ALL",
            "method": "MONTH_SCALE",
            "reason_code": str(month_scale.get("blocker") or "MONTH_SCALE_REPLAY_NEGATIVE"),
            "fresh_entries_must_remain_blocked": True,
            "live_promotion_allowed": False,
            "evidence_refs": ["data/month_scale_tp_replay_residuals.json"],
        }

    return sorted(
        (_dedupe_evidence_refs(row) for row in no_trade.values()),
        key=lambda item: (
            _num(item.get("net_jpy")),
            str(item.get("reason_code")),
            str(item.get("shape_key")),
        ),
    )[:ITEM_LIMIT]


def _mfe_mae_summary(
    *,
    timing: dict[str, Any],
    intent_replay_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "execution_timing": {
            "summary": timing.get("summary") or {},
            "loss_close_mfe": {
                "audited": len(_as_list(timing.get("loss_close_regrets"))),
                "had_positive_mfe": sum(1 for row in _as_list(timing.get("loss_close_regrets")) if isinstance(row, dict) and row.get("had_positive_mfe_before_loss_close") is True),
                "estimated_mfe_jpy": _round(sum(_num(row.get("estimated_mfe_jpy_before_loss_close")) for row in _as_list(timing.get("loss_close_regrets")) if isinstance(row, dict))),
            },
            "canceled_order_mfe": {
                "audited": len(_as_list(timing.get("canceled_order_regrets"))),
                "entry_touched": sum(1 for row in _as_list(timing.get("canceled_order_regrets")) if isinstance(row, dict) and row.get("entry_touched_after_cancel") is True),
                "tp_touched": sum(1 for row in _as_list(timing.get("canceled_order_regrets")) if isinstance(row, dict) and row.get("tp_touched_after_cancel") is True),
                "estimated_missed_mfe_jpy": _round(sum(_num(row.get("estimated_missed_mfe_jpy")) for row in _as_list(timing.get("canceled_order_regrets")) if isinstance(row, dict))),
            },
            "post_close_runner": {
                "audited": len(_as_list(timing.get("market_close_counterfactuals"))),
                "left_runner_upside": sum(1 for row in _as_list(timing.get("market_close_counterfactuals")) if isinstance(row, dict) and row.get("post_close_path_label") == "PROFIT_CLOSE_LEFT_RUNNER_UPSIDE"),
                "estimated_followthrough_jpy": _round(sum(_num(row.get("estimated_post_close_favorable_jpy")) for row in _as_list(timing.get("market_close_counterfactuals")) if isinstance(row, dict) and row.get("post_close_path_label") == "PROFIT_CLOSE_LEFT_RUNNER_UPSIDE")),
            },
        },
        "order_intent_bidask_replay": {
            "overall": _aggregate_replay_packets(intent_replay_packets),
            "by_family": _aggregate_replay_by(intent_replay_packets, "method"),
            "by_pair": _aggregate_replay_by(intent_replay_packets, "pair"),
            "by_session": _aggregate_replay_by(intent_replay_packets, "session"),
            "top_negative_packets": sorted(
                [
                    {
                        "lane_id": packet.get("lane_id"),
                        "pair": packet.get("pair"),
                        "side": packet.get("side"),
                        "method": packet.get("method"),
                        "session": packet.get("session"),
                        "name": packet.get("name"),
                        "avg_final_pips": packet.get("avg_final_pips"),
                        "avg_mfe_pips": packet.get("avg_mfe_pips"),
                        "avg_mae_pips": packet.get("avg_mae_pips"),
                        "samples": packet.get("samples"),
                        "adoption_status": packet.get("adoption_status"),
                    }
                    for packet in intent_replay_packets
                    if _maybe_num(packet.get("avg_final_pips")) is not None and _num(packet.get("avg_final_pips")) < 0
                ],
                key=lambda item: _num(item.get("avg_final_pips")),
            )[:ITEM_LIMIT],
        },
    }


def _month_scale_shape_blocks(month_scale: dict[str, Any]) -> dict[str, dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {}
    for row in _as_list(month_scale.get("residual_losing_families")):
        if not isinstance(row, dict):
            continue
        key = _shape_key(row.get("pair"), row.get("side"), row.get("strategy") or row.get("method"))
        blocks[key] = row
    return blocks


def _overall_verdict(
    *,
    capture: dict[str, Any],
    replay: dict[str, Any],
    month_scale: dict[str, Any],
    harvest_candidates: list[dict[str, Any]],
    runner_candidates: list[dict[str, Any]],
    partial_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    capture_status = str(capture.get("status") or "UNKNOWN")
    replay_net = _num((replay.get("summary") or {}).get("total_historical_net_jpy"))
    month_blocked = month_scale.get("fresh_entries_must_remain_blocked") is True
    harvest_net = sum(_num(row.get("take_profit_net_jpy")) for row in harvest_candidates)
    runner_tail = sum(_num(row.get("estimated_tail_jpy")) for row in runner_candidates)
    partial_tail = sum(_num(row.get("runner_tail_estimated_jpy")) for row in partial_candidates)

    if not harvest_candidates and not runner_candidates and not partial_candidates:
        classification = "NO_TRADE_DOMINANT"
        primary_reason = "No positive HARVEST/RUNNER shape survived spread-included replay and realized expectancy checks."
    elif partial_candidates:
        classification = "MIXED_HARVEST_PRIMARY"
        primary_reason = (
            "Broker TP/HARVEST buckets are the only clearly paying realized exit family, "
            "while timing audit shows some leftover runner tail; use TP-first only as diagnostic evidence."
        )
    elif harvest_candidates and not runner_candidates:
        classification = "HARVEST_PRIMARY"
        primary_reason = "Positive evidence is concentrated in attached/broker TP capture; RUNNER payoff is not proved."
    elif runner_candidates and not harvest_candidates:
        classification = "RUNNER_PRIMARY"
        primary_reason = "Positive evidence is concentrated in shapes whose average win beats average loss and MFE extends."
    else:
        classification = "MIXED"
        primary_reason = "Both TP capture and runner extension have positive evidence, but each still needs current live gates."

    live_promotion_allowed = (
        capture_status == "POSITIVE_EXPECTANCY"
        and replay_net >= 0
        and not month_blocked
        and classification not in {"NO_TRADE_DOMINANT"}
    )
    if capture_status == "NEGATIVE_EXPECTANCY" or month_blocked or replay_net < 0:
        live_promotion_allowed = False

    return {
        "classification": classification,
        "primary_reason": primary_reason,
        "live_promotion_allowed": live_promotion_allowed,
        "live_promotion_blockers": [
            blocker
            for blocker, active in (
                ("NEGATIVE_EXPECTANCY", capture_status == "NEGATIVE_EXPECTANCY"),
                ("MONTH_SCALE_REPLAY_NEGATIVE", month_blocked),
                ("REPLAY_BACKTEST_NEGATIVE", replay_net < 0),
            )
            if active
        ],
        "capture_economics_status": capture_status,
        "replay_backtest_total_historical_net_jpy": (replay.get("summary") or {}).get("total_historical_net_jpy"),
        "month_scale_blocker": month_scale.get("blocker"),
        "harvest_candidate_net_jpy": _round(harvest_net),
        "runner_candidate_tail_jpy": _round(runner_tail),
        "partial_tp_runner_tail_jpy": _round(partial_tail),
        "interpretation_for_4x": (
            "4x requires a mixed shape with HARVEST as the accounting base and only evidence-scoped runners. "
            "RUNNER-only promotion is not supported while market-close leakage and month-scale replay remain negative."
            if classification == "MIXED_HARVEST_PRIMARY"
            else primary_reason
        ),
    }


def _recommendations(
    *,
    verdict: dict[str, Any],
    capture: dict[str, Any],
    replay: dict[str, Any],
    month_scale: dict[str, Any],
    harvest_candidates: list[dict[str, Any]],
    runner_candidates: list[dict[str, Any]],
    partial_candidates: list[dict[str, Any]],
    no_trade_shapes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = [
        {
            "code": "NO_LIVE_PERMISSION_FROM_DIAGNOSIS",
            "action": "keep diagnosis read-only; do not send, cancel, close, modify launchd, or relax gates",
            "live_permission_allowed": False,
            "evidence": "live_side_effects is []",
        },
        {
            "code": "PROOF_QUEUE_COUNT_IS_NOT_PERMISSION",
            "action": "treat proof_queue_count=0 or any firepower estimate as evidence status only, never as a live permission",
            "live_permission_allowed": False,
        },
        {
            "code": "NO_4X_DEFICIT_LOT_BACKSOLVE",
            "action": "do not derive lot size from remaining_to_4x; this diagnosis contains no unit sizing output",
            "live_permission_allowed": False,
        },
    ]
    if str(capture.get("status") or "") == "NEGATIVE_EXPECTANCY":
        recommendations.append(
            {
                "code": "NEGATIVE_EXPECTANCY_VISIBLE",
                "action": "keep fresh entries blocked except exact TP-proven HARVEST shapes that pass every existing current gate",
                "capture_status": capture.get("status"),
                "dominant_loss_exit_reason": (capture.get("repair_summary") or {}).get("dominant_loss_exit_reason"),
                "live_permission_allowed": False,
            }
        )
    if month_scale.get("fresh_entries_must_remain_blocked") is True:
        recommendations.append(
            {
                "code": "MONTH_SCALE_REPLAY_BLOCKS_PROMOTION",
                "action": "do not promote matching family/pair/session while month-scale replay remains negative",
                "blocker": month_scale.get("blocker"),
                "current_residual_pl_jpy": (month_scale.get("pl_summary") or {}).get("current_residual_pl_jpy"),
                "live_permission_allowed": False,
            }
        )
    if harvest_candidates:
        recommendations.append(
            {
                "code": "HARVEST_BASE_KEEP_NARROW",
                "action": "preserve positive broker-TP HARVEST shapes, but only as exact-shape, spread-included evidence candidates",
                "candidate_count": len(harvest_candidates),
                "top_shape": harvest_candidates[0].get("shape_key"),
                "live_permission_allowed": False,
            }
        )
    if partial_candidates:
        recommendations.append(
            {
                "code": "PARTIAL_TP_RUNNER_REPLAY_NEXT",
                "action": "simulate TP1 bank plus small runner leg for top HARVEST shapes before changing live behavior",
                "candidate_count": len(partial_candidates),
                "top_shape": partial_candidates[0].get("shape_key"),
                "live_permission_allowed": False,
            }
        )
    if not runner_candidates:
        recommendations.append(
            {
                "code": "RUNNER_ONLY_UNPROVED",
                "action": "do not switch to RUNNER-only; require average win greater than average loss plus MFE-extension evidence",
                "live_permission_allowed": False,
            }
        )
    if no_trade_shapes:
        recommendations.append(
            {
                "code": "NO_TRADE_SHAPES_STAY_BLOCKED",
                "action": "keep spread/slippage-included negative and month-scale-negative shapes as NO_TRADE until replay clears",
                "shape_count": len(no_trade_shapes),
                "top_reason": no_trade_shapes[0].get("reason_code"),
                "live_permission_allowed": False,
            }
        )
    recommendations.append(
        {
            "code": "FOUR_X_PAYOFF_SHAPE_VERDICT",
            "action": verdict.get("interpretation_for_4x"),
            "classification": verdict.get("classification"),
            "live_permission_allowed": verdict.get("live_promotion_allowed"),
        }
    )
    return recommendations


def _next_evidence_actions(
    *,
    month_scale: dict[str, Any],
    harvest_candidates: list[dict[str, Any]],
    runner_candidates: list[dict[str, Any]],
    no_trade_shapes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions = [
        {
            "code": "REFRESH_MONTH_SCALE_TIMING_AUDIT",
            "command": "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80",
            "read_only": True,
            "success_condition": {
                "artifact": "data/execution_timing_audit.json",
                "checks": [
                    {"json_path": "window.lookback_hours", "operator": ">=", "value": 720},
                    {"json_path": "summary.loss_closes_repair_replay_triggered", "operator": "==", "value": 0},
                ],
            },
        },
        {
            "code": "REFRESH_CAPTURE_ECONOMICS_AFTER_LEDGER_SYNC",
            "command": "PYTHONPATH=src python3 -m quant_rabbit.cli capture-economics",
            "read_only": True,
            "success_condition": {
                "artifact": "data/capture_economics.json",
                "checks": [
                    {"json_path": "status", "operator": "in", "value": ["POSITIVE_EXPECTANCY", "NEGATIVE_EXPECTANCY"]},
                    {"json_path": "repair_summary.dominant_loss_exit_reason", "operator": "!=", "value": None},
                ],
            },
        },
        {
            "code": "RECHECK_MONTH_SCALE_RESIDUALS",
            "command": "PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance",
            "read_only": True,
            "success_condition": {
                "artifact": "data/month_scale_tp_replay_residuals.json",
                "checks": [
                    {"json_path": "fresh_entries_must_remain_blocked", "operator": "==", "value": False},
                    {"json_path": "pl_summary.current_residual_pl_jpy", "operator": ">=", "value": 0},
                ],
            },
            "current_result": {
                "fresh_entries_must_remain_blocked": month_scale.get("fresh_entries_must_remain_blocked"),
                "current_residual_pl_jpy": (month_scale.get("pl_summary") or {}).get("current_residual_pl_jpy"),
            },
        },
    ]
    if harvest_candidates:
        actions.append(
            {
                "code": "COLLECT_EXACT_HARVEST_TP_PROOF_GAPS",
                "read_only": True,
                "target_shapes": [
                    {
                        "shape_key": row.get("shape_key"),
                        "proof_gap_trades": row.get("proof_gap_trades"),
                        "take_profit_losses": row.get("take_profit_losses"),
                    }
                    for row in harvest_candidates
                    if int(row.get("proof_gap_trades") or 0) > 0
                ][:EXAMPLE_LIMIT],
                "success_condition": {
                    "checks": [
                        {"field": "take_profit_trades", "operator": ">=", "value": MIN_SAMPLE_FOR_VERDICT},
                        {"field": "take_profit_losses", "operator": "==", "value": 0},
                        {"field": "take_profit_expectancy_jpy", "operator": ">", "value": 0},
                    ],
                },
            }
        )
    if runner_candidates:
        actions.append(
            {
                "code": "RUNNER_TAIL_COUNTERFACTUAL_REPLAY",
                "read_only": True,
                "target_shapes": [row.get("shape_key") for row in runner_candidates[:EXAMPLE_LIMIT]],
                "success_condition": {
                    "checks": [
                        {"field": "avg_win_jpy", "operator": ">", "field_rhs": "avg_loss_jpy"},
                        {"field": "missed_runner_case_count", "operator": ">", "value": 0},
                        {"field": "month_scale_blocker", "operator": "==", "value": None},
                    ],
                },
            }
        )
    if no_trade_shapes:
        actions.append(
            {
                "code": "NO_TRADE_ESCAPE_CONDITION",
                "read_only": True,
                "target_shapes": [row.get("shape_key") for row in no_trade_shapes[:EXAMPLE_LIMIT]],
                "success_condition": {
                    "checks": [
                        {"field": "spread_included_replay_net", "operator": ">=", "value": 0},
                        {"field": "month_scale_replay_net", "operator": ">=", "value": 0},
                        {"field": "current_live_blocker_codes", "operator": "not_contains_any", "value": ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION", "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"]},
                    ],
                },
            }
        )
    return actions


def _family_expectancy_breakdown(stats: dict[str, Any], residual: dict[str, Any] | None) -> dict[str, Any]:
    tp = stats.get("take_profit_order") or {}
    market = stats.get("market_order_trade_close") or {}
    return {
        "shape_positive_only_if": (
            "TAKE_PROFIT_ORDER"
            if _num(tp.get("expectancy_jpy_per_trade")) > 0 and int(tp.get("losses") or 0) == 0
            else None
        ),
        "market_close_leak_active": _num(market.get("net_jpy")) < 0,
        "average_win_too_small_vs_average_loss": _num(stats.get("avg_win_jpy")) <= _num(stats.get("avg_loss_jpy")) if _num(stats.get("avg_loss_jpy")) > 0 else False,
        "month_scale_residual_negative": _num((residual or {}).get("actual_pl_jpy")) < 0,
    }


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "trades": metrics.get("trades", 0),
        "wins": metrics.get("wins", 0),
        "losses": metrics.get("losses", 0),
        "win_rate": metrics.get("win_rate"),
        "avg_win_jpy": metrics.get("avg_win_jpy"),
        "avg_loss_jpy": metrics.get("avg_loss_jpy"),
        "payoff_ratio": metrics.get("payoff_ratio"),
        "breakeven_payoff_at_win_rate": metrics.get("breakeven_payoff_at_win_rate"),
        "expectancy_jpy_per_trade": metrics.get("expectancy_jpy_per_trade"),
        "net_jpy": metrics.get("net_jpy"),
    }


def _compact_residual(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    return {
        "pair": row.get("pair"),
        "side": row.get("side"),
        "method": row.get("method") or row.get("strategy"),
        "actual_pl_jpy": row.get("actual_pl_jpy"),
        "repair_replay_pl_jpy": row.get("repair_replay_pl_jpy"),
        "repair_replay_triggered": row.get("repair_replay_triggered"),
        "block_reasons": row.get("block_reasons"),
        "diagnosis": row.get("diagnosis"),
        "live_permission_filter": row.get("live_permission_filter"),
    }


def _compact_timing_rollup(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    return {
        "orders": row.get("orders"),
        "entry_touch_after_cancel_rate": row.get("entry_touch_after_cancel_rate"),
        "tp_touched_after_cancel_rate": row.get("tp_touched_after_cancel_rate"),
        "avg_entry_touch_after_cancel_minutes": row.get("avg_entry_touch_after_cancel_minutes"),
        "avg_tp_touch_after_cancel_minutes": row.get("avg_tp_touch_after_cancel_minutes"),
        "estimated_missed_mfe_jpy": row.get("estimated_missed_mfe_jpy"),
        "priority_class": row.get("priority_class"),
    }


def _aggregate_replay_by(packets: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for packet in packets:
        groups.setdefault(str(packet.get(field) or "UNKNOWN"), []).append(packet)
    return {key: _aggregate_replay_packets(rows) for key, rows in sorted(groups.items())}


def _aggregate_replay_packets(packets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "packets": len(packets),
        "sample_sum": sum(int(packet.get("samples") or 0) for packet in packets),
        "avg_mfe_pips": _avg([_num(packet.get("avg_mfe_pips")) for packet in packets if _maybe_num(packet.get("avg_mfe_pips")) is not None]),
        "avg_mae_pips": _avg([_num(packet.get("avg_mae_pips")) for packet in packets if _maybe_num(packet.get("avg_mae_pips")) is not None]),
        "avg_final_pips": _avg([_num(packet.get("avg_final_pips")) for packet in packets if _maybe_num(packet.get("avg_final_pips")) is not None]),
        "negative_avg_final_packets": sum(1 for packet in packets if _maybe_num(packet.get("avg_final_pips")) is not None and _num(packet.get("avg_final_pips")) < 0),
        "blocks_live_support_packets": sum(1 for packet in packets if packet.get("blocks_live_support") is True),
    }


def _write_report(payload: dict[str, Any], report_path: Path) -> None:
    verdict = payload.get("overall_payoff_shape_verdict") or {}
    lines = [
        "# Payoff Shape Diagnosis Report",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Verdict: `{verdict.get('classification')}`",
        f"- Live promotion allowed: `{verdict.get('live_promotion_allowed')}`",
        f"- Live side effects: `{payload.get('live_side_effects')}`",
        "",
        "## 4x Payoff Shape Verdict",
        "",
        str(verdict.get("interpretation_for_4x") or verdict.get("primary_reason") or ""),
        "",
        "## HARVEST Candidates",
        "",
        "| shape | class | TP n | TP exp | proof gap | market close net | live permission |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("harvest_candidates", []):
        lines.append(
            f"| `{row.get('shape_key')}` | `{row.get('classification')}` | {row.get('take_profit_trades')} "
            f"| {row.get('take_profit_expectancy_jpy')} | {row.get('proof_gap_trades')} "
            f"| {row.get('market_close_net_jpy')} | `{row.get('live_promotion_allowed')}` |"
        )
    lines += [
        "",
        "## RUNNER Candidates",
        "",
        "| shape | n | exp | avg win | avg loss | tail cases | tail JPY | live permission |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("runner_candidates", []):
        lines.append(
            f"| `{row.get('shape_key')}` | {row.get('trades')} | {row.get('expectancy_jpy_per_trade')} "
            f"| {row.get('avg_win_jpy')} | {row.get('avg_loss_jpy')} "
            f"| {row.get('missed_runner_case_count')} | {row.get('estimated_tail_jpy')} "
            f"| `{row.get('live_promotion_allowed')}` |"
        )
    lines += [
        "",
        "## Partial TP + Runner",
        "",
        "| shape | TP exp | runner cases | runner tail JPY | market close leak |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("partial_tp_runner_candidates", []):
        tp = row.get("tp1_evidence") or {}
        lines.append(
            f"| `{row.get('shape_key')}` | {tp.get('take_profit_expectancy_jpy')} "
            f"| {row.get('runner_tail_case_count')} | {row.get('runner_tail_estimated_jpy')} "
            f"| {row.get('market_close_leak_net_jpy')} |"
        )
    lines += [
        "",
        "## NO_TRADE Shapes",
        "",
        "| shape | reason | net | live permission |",
        "|---|---|---:|---|",
    ]
    for row in payload.get("no_trade_shapes", []):
        lines.append(
            f"| `{row.get('shape_key')}` | `{row.get('reason_code')}` | {row.get('net_jpy')} "
            f"| `{row.get('live_promotion_allowed')}` |"
        )
    lines += [
        "",
        "## MFE / MAE Summary",
        "",
        "```json",
        json.dumps(payload.get("mfe_mae_summary"), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Recommendations",
        "",
    ]
    for row in payload.get("payoff_shape_recommendations", []):
        lines.append(f"- `{row.get('code')}`: {row.get('action')}")
    lines += [
        "",
        "## Next Evidence Actions",
        "",
    ]
    for row in payload.get("next_evidence_actions", []):
        command = row.get("command")
        suffix = f" - `{command}`" if command else ""
        lines.append(f"- `{row.get('code')}`{suffix}")
    lines += [
        "",
        "## Safety",
        "",
        "- No live orders were sent.",
        "- No orders were canceled.",
        "- No positions were closed.",
        "- No launchd state was changed.",
        "- No gate was relaxed.",
        "- Negative expectancy and month-scale replay blockers remain visible.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _maybe_num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> float:
    return _maybe_num(value) or 0.0


def _maybe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | int | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _avg(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return _round(sum(clean) / len(clean))


def _drop_none(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value is not None}


def _dedupe_evidence_refs(row: dict[str, Any]) -> dict[str, Any]:
    refs = row.get("evidence_refs")
    if isinstance(refs, list):
        row["evidence_refs"] = sorted(set(str(ref) for ref in refs if str(ref)))
    return row

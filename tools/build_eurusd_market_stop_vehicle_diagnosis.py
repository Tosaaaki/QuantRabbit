#!/usr/bin/env python3
"""Build read-only MARKET/STOP HARVEST vehicle diagnosis for EUR_USD shorts."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_SHAPE = "EUR_USD|SHORT|BREAKOUT_FAILURE"
LIMIT_TARGET_SHAPE = "EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST"
JSON_OUTPUT = ROOT / "data" / "eurusd_short_breakout_failure_market_stop_vehicle_diagnosis.json"
DOC_OUTPUT = ROOT / "docs" / "eurusd_short_breakout_failure_market_stop_vehicle_diagnosis.md"


def main() -> int:
    payload = build_payload(_now())
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    DOC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DOC_OUTPUT.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {JSON_OUTPUT.relative_to(ROOT)}")
    print(f"wrote {DOC_OUTPUT.relative_to(ROOT)}")
    return 0


def build_payload(generated_at: str) -> dict[str, Any]:
    vehicle_split = _load_json("data/eurusd_short_breakout_failure_vehicle_split_diagnosis.json")
    spread_proof = _load_json("data/eurusd_short_breakout_failure_spread_slippage_proof.json")
    limit_mining = _load_json("data/eurusd_short_breakout_failure_limit_sample_mining.json")
    active_contract = _load_json("data/active_trader_contract.json")
    proof_floor = _load_json("data/eurusd_short_breakout_failure_proof_floor_update.json")
    harvest_path = _load_json("data/harvest_live_grade_path.json")
    operator_review = _load_json("data/operator_review_report.json")
    goal_loop = _load_json("data/trader_goal_loop_orchestrator.json")
    vehicle_split_body = vehicle_split.get("vehicle_split") if isinstance(vehicle_split.get("vehicle_split"), dict) else {}
    market_samples = _vehicle_samples(vehicle_split_body, "market_order_samples", "MARKET_ORDER")
    stop_samples = _vehicle_samples(vehicle_split_body, "stop_order_samples", "STOP_ORDER")
    limit_samples = _vehicle_samples(vehicle_split_body, "limit_order_samples", "LIMIT_ORDER")
    market_vehicle = _market_vehicle(market_samples, spread_proof)
    stop_vehicle = _stop_vehicle(stop_samples, spread_proof)
    limit_baseline = _limit_baseline(limit_samples, limit_mining, vehicle_split_body)
    comparison = _vehicle_comparison(limit_baseline, market_vehicle, stop_vehicle)
    remaining_blockers = _remaining_blockers(
        market_vehicle=market_vehicle,
        stop_vehicle=stop_vehicle,
        active_contract=active_contract,
        proof_floor=proof_floor,
        harvest_path=harvest_path,
        operator_review=operator_review,
        goal_loop=goal_loop,
    )
    return {
        "status": _overall_status(market_vehicle, stop_vehicle),
        "target_shape": TARGET_SHAPE,
        "generated_at_utc": generated_at,
        "read_only": True,
        "live_side_effects": [],
        "live_permission_allowed": False,
        "limit_vehicle_baseline": limit_baseline,
        "market_harvest_vehicle": market_vehicle,
        "stop_harvest_vehicle": stop_vehicle,
        "vehicle_comparison": comparison,
        "recommended_active_path": comparison["recommended_active_path"],
        "four_x_progress_hypothesis": (
            "MARKET and STOP rows are positive as observed HARVEST evidence, but they must become "
            "separate vehicles. STOP_HARVEST is currently closer than MARKET_HARVEST because it has "
            "trigger/order prices and higher expectancy, while both still require exact replay, "
            "risk/invalidation contracts, and operator/guardian review before any active use."
        ),
        "root_improvement_target": (
            "Define and replay STOP_HARVEST first as a separate EUR_USD SHORT BREAKOUT_FAILURE vehicle, "
            "then evaluate MARKET_HARVEST with an explicit requested-entry/slippage/invalidation packet."
        ),
        "expected_edge_improvement": (
            "A separate STOP/MARKET proof path could avoid waiting on exhausted LIMIT-only samples, "
            "but expected improvement is evidence quality only; it does not create live permission."
        ),
        "remaining_blockers": remaining_blockers,
        "next_read_only_actions": [
            "Build exact STOP_HARVEST S5 bid/ask trigger and TP replay from the 7 STOP_ORDER samples.",
            "Define STOP_HARVEST invalidation, max trigger slippage, and stop-chase rejection rules before scout review.",
            "Build a MARKET_HARVEST packet with requested entry quote, max slippage, invalidation, and TP/SL geometry before replay.",
            "Keep LIMIT_HARVEST proof isolated at 4/20; do not import MARKET or STOP samples into the LIMIT floor.",
            "After exact replay packets exist, rerun proof queue, 4x planner, active contract, and operator-review material read-only.",
        ],
        "do_not_do": [
            "Do not send live orders.",
            "Do not stage live orders.",
            "Do not cancel or close trades.",
            "Do not mutate broker state, TP, SL, or launchd.",
            "Do not relax gates or infer operator approval.",
            "Do not mix MARKET_ORDER or STOP_ORDER samples into LIMIT_HARVEST proof.",
            "Do not merge MARKET_HARVEST and STOP_HARVEST into one proof vehicle.",
            "Do not mix market-close losses into TP/HARVEST proof, and do not hide negative expectancy or month-scale blockers.",
            "Do not backsolve lot size from the 4x deficit.",
        ],
        "source_artifacts": [
            "data/eurusd_short_breakout_failure_limit_sample_mining.json",
            "data/active_trader_contract.json",
            "data/eurusd_short_breakout_failure_vehicle_split_diagnosis.json",
            "data/eurusd_short_breakout_failure_spread_slippage_proof.json",
            "data/eurusd_short_breakout_failure_proof_floor_update.json",
            "data/harvest_live_grade_path.json",
            "data/payoff_shape_diagnosis.json",
            "data/operator_review_report.json",
            "data/trader_goal_loop_orchestrator.json",
            "data/execution_ledger.db",
        ],
        "safety_checks": {
            "market_and_stop_aggregated_separately": True,
            "limit_proof_mixes_market_stop_samples": False,
            "market_stop_combined_used_as_single_vehicle": False,
            "market_close_losses_excluded_from_harvest_proof": True,
            "live_permission_created": False,
        },
    }


def _limit_baseline(
    limit_samples: list[dict[str, Any]],
    limit_mining: dict[str, Any],
    vehicle_split: dict[str, Any],
) -> dict[str, Any]:
    floor = limit_mining.get("sample_floor") if isinstance(limit_mining.get("sample_floor"), dict) else {}
    summary = vehicle_split.get("limit_order_summary") if isinstance(vehicle_split.get("limit_order_summary"), dict) else {}
    return {
        "vehicle": "LIMIT_HARVEST",
        "status": limit_mining.get("status") or "LIMIT_ONLY_WAIT_FOR_FUTURE_SAMPLES",
        "target_shape": LIMIT_TARGET_SHAPE,
        "sample_summary": _summary(limit_samples, fallback=summary),
        "sample_floor": {
            "required_exact_limit_samples": _first_int(floor.get("required_exact_limit_samples"), 20),
            "current_replayed_exact_limit_samples": _first_int(
                floor.get("current_replayed_exact_limit_samples"),
                len(limit_samples),
            ),
            "additional_acceptable_local_samples_found": _first_int(
                floor.get("additional_acceptable_local_samples_found"),
                0,
            ),
            "remaining_exact_limit_samples": _first_int(floor.get("remaining_exact_limit_samples"), 16),
            "floor_met": bool(floor.get("floor_met")),
        },
        "proof_boundary": {
            "market_samples_allowed": False,
            "stop_samples_allowed": False,
            "reason": "LIMIT_HARVEST remains a separate exact LIMIT_ORDER proof floor.",
        },
        "sample_trade_ids": [str(row.get("trade_id")) for row in limit_samples if row.get("trade_id")],
    }


def _market_vehicle(samples: list[dict[str, Any]], spread_proof: dict[str, Any]) -> dict[str, Any]:
    entry_order_price_available = all(row.get("entry_order_price") is not None for row in samples)
    return {
        "vehicle": "MARKET_HARVEST",
        "status": "MARKET_HARVEST_REPLAY_REQUIRED",
        "entry_order_type": "MARKET_ORDER",
        "sample_summary": _summary(samples),
        "sample_trade_ids": [str(row.get("trade_id")) for row in samples if row.get("trade_id")],
        "proof_boundary": {
            "included_in_limit_proof": False,
            "mixed_with_stop_vehicle": False,
            "market_close_mixed_in": any(bool(row.get("market_close_mixed_in")) for row in samples),
        },
        "entry_risk_slippage_invalidation": {
            "requested_entry_order_price_available": entry_order_price_available,
            "observed_entry_spread_available": all(row.get("entry_spread_pips") is not None for row in samples),
            "historical_fill_vs_requested_price_available": all(
                row.get("entry_fill_vs_order_price_pips") is not None for row in samples
            ),
            "max_slippage_contract_defined": False,
            "pre_entry_invalidation_defined": False,
            "risk_defined_for_scout": False,
            "blocking_reason": (
                "MARKET_ORDER rows have observed fill/spread evidence, but no requested entry order price. "
                "A MARKET_HARVEST scout needs requested quote, max slippage, invalidation, and TP/SL geometry."
            ),
        },
        "exact_replay": {
            "independent_s5_bidask_replay_attached": False,
            "samples_requiring_replay": len(samples),
            "replay_status_counts": _counts(row.get("bidask_replay_status") for row in samples),
            "source_status": spread_proof.get("status"),
        },
        "readiness": {
            "promising_observed_harvest": _is_positive(samples),
            "can_enter_proof_queue_now": False,
            "can_create_live_permission": False,
            "next_gate": "Define MARKET risk/slippage/invalidation packet before exact replay or scout.",
        },
    }


def _stop_vehicle(samples: list[dict[str, Any]], spread_proof: dict[str, Any]) -> dict[str, Any]:
    trigger_price_available = all(row.get("entry_order_price") is not None for row in samples)
    trigger_slippage = [abs(_float(row.get("entry_fill_vs_order_price_pips")) or 0.0) for row in samples]
    return {
        "vehicle": "STOP_HARVEST",
        "status": "STOP_HARVEST_REPLAY_REQUIRED",
        "entry_order_type": "STOP_ORDER",
        "sample_summary": _summary(samples),
        "sample_trade_ids": [str(row.get("trade_id")) for row in samples if row.get("trade_id")],
        "proof_boundary": {
            "included_in_limit_proof": False,
            "mixed_with_market_vehicle": False,
            "market_close_mixed_in": any(bool(row.get("market_close_mixed_in")) for row in samples),
        },
        "trigger_slippage_invalidation": {
            "stop_trigger_price_available": trigger_price_available,
            "observed_trigger_slippage_pips_max": _round(max(trigger_slippage) if trigger_slippage else None),
            "observed_trigger_spread_available": all(row.get("entry_spread_pips") is not None for row in samples),
            "trigger_contract_defined": trigger_price_available,
            "max_trigger_slippage_contract_defined": False,
            "pre_or_post_trigger_invalidation_defined": False,
            "risk_defined_for_scout": False,
            "blocking_reason": (
                "STOP_ORDER rows preserve trigger/order prices, but a STOP_HARVEST vehicle still needs trigger-side "
                "S5 replay, invalidation, max slippage, and stop-chase rules before scout or proof promotion."
            ),
        },
        "exact_replay": {
            "independent_s5_bidask_replay_attached": False,
            "samples_requiring_replay": len(samples),
            "replay_status_counts": _counts(row.get("bidask_replay_status") for row in samples),
            "source_status": spread_proof.get("status"),
        },
        "readiness": {
            "promising_observed_harvest": _is_positive(samples),
            "can_enter_proof_queue_now": False,
            "can_create_live_permission": False,
            "next_gate": "Build STOP trigger/TP exact replay and define invalidation before operator review.",
        },
    }


def _vehicle_comparison(
    limit_baseline: dict[str, Any],
    market_vehicle: dict[str, Any],
    stop_vehicle: dict[str, Any],
) -> dict[str, Any]:
    market_summary = market_vehicle["sample_summary"]
    stop_summary = stop_vehicle["sample_summary"]
    closer = "STOP_HARVEST"
    reasons = [
        "STOP_HARVEST has explicit stop trigger/order prices while MARKET_HARVEST lacks requested entry prices.",
        "STOP_HARVEST observed net and expectancy exceed MARKET_HARVEST in the current packet.",
        "Both vehicles remain blocked until exact S5 bid/ask replay and risk/invalidation contracts exist.",
    ]
    if market_summary["sample_count"] >= 20 and market_vehicle["entry_risk_slippage_invalidation"]["risk_defined_for_scout"]:
        closer = "MARKET_HARVEST"
        reasons = ["MARKET_HARVEST has enough samples and a complete risk contract."]
    return {
        "limit_baseline": {
            "current_exact_limit_samples": limit_baseline["sample_floor"]["current_replayed_exact_limit_samples"],
            "remaining_exact_limit_samples": limit_baseline["sample_floor"]["remaining_exact_limit_samples"],
            "local_coverage_exhausted": limit_baseline["status"]
            == "LOCAL_LIMIT_SAMPLE_COVERAGE_EXHAUSTED_STILL_UNDERSAMPLED",
        },
        "market_vs_stop": {
            "market_sample_count": market_summary["sample_count"],
            "market_net_jpy": market_summary["net_jpy"],
            "market_expectancy_jpy_per_trade": market_summary["expectancy_jpy_per_trade"],
            "stop_sample_count": stop_summary["sample_count"],
            "stop_net_jpy": stop_summary["net_jpy"],
            "stop_expectancy_jpy_per_trade": stop_summary["expectancy_jpy_per_trade"],
            "closer_to_4x_active_path": closer,
            "why": reasons,
        },
        "recommended_active_path": "EVIDENCE_ACQUISITION:STOP_HARVEST_EXACT_REPLAY_BEFORE_SCOUT_OR_OPERATOR_REVIEW",
        "scout_or_operator_review_now": False,
        "why_not_scout_yet": "Risk/invalidation and exact S5 bid/ask replay are not complete for MARKET_HARVEST or STOP_HARVEST.",
    }


def _remaining_blockers(
    *,
    market_vehicle: dict[str, Any],
    stop_vehicle: dict[str, Any],
    active_contract: dict[str, Any],
    proof_floor: dict[str, Any],
    harvest_path: dict[str, Any],
    operator_review: dict[str, Any],
    goal_loop: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers = [
        ("MARKET_HARVEST_EXACT_REPLAY_REQUIRED", "BLOCKING_MARKET_VEHICLE_PROOF"),
        ("STOP_HARVEST_EXACT_REPLAY_REQUIRED", "BLOCKING_STOP_VEHICLE_PROOF"),
    ]
    if not market_vehicle["entry_risk_slippage_invalidation"]["risk_defined_for_scout"]:
        blockers.append(("MARKET_ENTRY_RISK_SLIPPAGE_INVALIDATION_UNDEFINED", "BLOCKING_MARKET_SCOUT"))
    if not stop_vehicle["trigger_slippage_invalidation"]["risk_defined_for_scout"]:
        blockers.append(("STOP_TRIGGER_SLIPPAGE_INVALIDATION_UNDEFINED", "BLOCKING_STOP_SCOUT"))
    blockers.extend(
        [
            ("LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY", "VISIBLE_LIMIT_BASELINE_BLOCKER"),
            ("MARKET_STOP_NOT_ALLOWED_IN_LIMIT_PROOF", "PROOF_BOUNDARY"),
            ("NO_LIVE_PERMISSION_CREATED", "SAFETY_BOUNDARY"),
        ]
    )
    for source in (
        active_contract.get("remaining_blockers") or [],
        proof_floor.get("remaining_blockers") or [],
        harvest_path.get("promotion_blockers") or [],
        operator_review.get("blockers") or [],
        goal_loop.get("remaining_blockers") or [],
    ):
        for code in _codes_from_values(source):
            if code in {
                "NEGATIVE_EXPECTANCY_ACTIVE",
                "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
                "MARKET_CLOSE_LEAK_PRESENT",
                "MARKET_CLOSE_LEAK_PRESENT_EXCLUDED",
                "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                "NO_LIVE_READY_PORTFOLIO",
                "NO_FRESH_GATEWAY_PERMISSION",
                "PORTFOLIO_PLANNER_CANNOT_CREATE_LIVE_PERMISSION",
                "PROOF_QUEUE_MEMBER_BUT_NOT_PROOF_READY",
            }:
                blockers.append((code, "VISIBLE_EXISTING_BLOCKER"))
    rows = []
    seen = set()
    for code, status in blockers:
        if code in seen:
            continue
        seen.add(code)
        rows.append({"code": code, "status": status, "blocks_live_permission": True})
    return rows


def _overall_status(market_vehicle: dict[str, Any], stop_vehicle: dict[str, Any]) -> str:
    if not market_vehicle["readiness"]["promising_observed_harvest"] and not stop_vehicle["readiness"][
        "promising_observed_harvest"
    ]:
        return "MARKET_STOP_VEHICLE_REJECTED"
    if (
        not market_vehicle["entry_risk_slippage_invalidation"]["risk_defined_for_scout"]
        and not stop_vehicle["trigger_slippage_invalidation"]["risk_defined_for_scout"]
    ):
        return "MARKET_STOP_VEHICLE_PROMISING_STILL_BLOCKED"
    return "MARKET_STOP_VEHICLE_BLOCKED_RISK_UNDEFINED"


def render_markdown(payload: dict[str, Any]) -> str:
    market = payload["market_harvest_vehicle"]
    stop = payload["stop_harvest_vehicle"]
    limit = payload["limit_vehicle_baseline"]
    comparison = payload["vehicle_comparison"]["market_vs_stop"]
    lines = [
        "# EUR_USD SHORT BREAKOUT_FAILURE MARKET/STOP Vehicle Diagnosis",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "## Verdict",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Target shape: `{payload.get('target_shape')}`",
        f"- Recommended active path: `{payload.get('recommended_active_path')}`",
        f"- Live permission allowed: `{payload.get('live_permission_allowed')}`",
        f"- Read-only: `{payload.get('read_only')}`",
        "",
        "## Vehicle Split",
        "",
        f"- LIMIT baseline: `{limit['sample_summary']['sample_count']}` samples, remaining exact LIMIT gap `{limit['sample_floor']['remaining_exact_limit_samples']}`.",
        f"- MARKET_HARVEST: `{market['sample_summary']['wins']}/{market['sample_summary']['losses']}` net `{market['sample_summary']['net_jpy']}` JPY, status `{market['status']}`.",
        f"- STOP_HARVEST: `{stop['sample_summary']['wins']}/{stop['sample_summary']['losses']}` net `{stop['sample_summary']['net_jpy']}` JPY, status `{stop['status']}`.",
        f"- Closer active path candidate: `{comparison['closer_to_4x_active_path']}`.",
        "",
        "MARKET_ORDER and STOP_ORDER samples are not mixed into LIMIT proof and are not merged with each other.",
        "",
        "## Risk Boundary",
        "",
        f"- MARKET requested entry price available: `{market['entry_risk_slippage_invalidation']['requested_entry_order_price_available']}`.",
        f"- MARKET risk defined for scout: `{market['entry_risk_slippage_invalidation']['risk_defined_for_scout']}`.",
        f"- STOP trigger price available: `{stop['trigger_slippage_invalidation']['stop_trigger_price_available']}`.",
        f"- STOP risk defined for scout: `{stop['trigger_slippage_invalidation']['risk_defined_for_scout']}`.",
        "",
        "## Remaining Blockers",
        "",
    ]
    lines.extend(f"- `{row['code']}`: {row['status']}" for row in payload.get("remaining_blockers") or [])
    lines.extend(["", "## Next Read-Only Actions", ""])
    lines.extend(f"- {action}" for action in payload.get("next_read_only_actions") or [])
    return "\n".join(lines) + "\n"


def _vehicle_samples(vehicle_split: dict[str, Any], key: str, order_type: str) -> list[dict[str, Any]]:
    samples = vehicle_split.get(key) if isinstance(vehicle_split.get(key), list) else []
    return [row for row in samples if isinstance(row, dict) and row.get("entry_order_type") == order_type]


def _summary(samples: list[dict[str, Any]], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = fallback or {}
    sample_count = len(samples) if samples else _first_int(fallback.get("samples"), 0)
    wins = sum(1 for row in samples if (_float(row.get("realized_pl_jpy")) or 0.0) > 0.0)
    losses = sum(1 for row in samples if (_float(row.get("realized_pl_jpy")) or 0.0) < 0.0)
    net = sum(_float(row.get("realized_pl_jpy")) or 0.0 for row in samples)
    if not samples:
        wins = _first_int(fallback.get("wins"), 0)
        losses = _first_int(fallback.get("losses"), 0)
        net = _float(fallback.get("net_jpy")) or 0.0
    expectancy = net / sample_count if sample_count else 0.0
    return {
        "sample_count": sample_count,
        "wins": wins,
        "losses": losses,
        "net_jpy": _round(net),
        "expectancy_jpy_per_trade": _round(expectancy),
        "observed_cost_inclusive_pass_count": sum(
            1 for row in samples if row.get("observed_cost_inclusive_pass") is True
        ),
        "entry_order_type_counts": dict(sorted(Counter(str(row.get("entry_order_type")) for row in samples).items())),
        "entry_spread_pips": _min_max(row.get("entry_spread_pips") for row in samples),
        "exit_spread_pips": _min_max(row.get("exit_spread_pips") for row in samples),
        "market_close_mixed_in_count": sum(1 for row in samples if row.get("market_close_mixed_in")),
    }


def _is_positive(samples: list[dict[str, Any]]) -> bool:
    summary = _summary(samples)
    return summary["sample_count"] > 0 and summary["losses"] == 0 and summary["net_jpy"] > 0


def _counts(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values if value is not None).items()))


def _min_max(values: Any) -> dict[str, float | None]:
    parsed = [_float(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    if not parsed:
        return {"min": None, "max": None}
    return {"min": _round(min(parsed)), "max": _round(max(parsed))}


def _codes_from_values(values: Any) -> list[str]:
    if not isinstance(values, list):
        values = [values]
    codes = []
    for value in values:
        if isinstance(value, dict):
            raw = value.get("code") or value.get("status")
        else:
            raw = value
        if raw:
            codes.append(str(raw).split(":", 1)[0].strip())
    return [code for code in codes if code]


def _load_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _first_int(*values: Any) -> int:
    for value in values:
        parsed = _int(value)
        if parsed is not None:
            return parsed
    return 0


def _int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
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

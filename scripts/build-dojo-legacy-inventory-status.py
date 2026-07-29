#!/usr/bin/env python3
"""Merge the 82-family inventory with completed legacy replay attempts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


BASELINE_EVALUATED = {
    "trend_ma",
    "pulse_break",
    "m1_scalper",
    "range_fader",
    "trend_breakout",
    "session_open",
    "pullback_continuation",
    "failed_break_reverse",
    "donchian55",
    "bb_rsi",
    "momentum_pulse",
    "vol_compression_break",
    "bb_rsi_fast",
    "micro_vwap_revert",
    "range_break",
    "pullback_ema",
    "level_reactor",
    "vwap_bound_revert",
    "trend_momentum",
    "compression_revert",
    "trend_retest",
    "impulse_break_s5",
    "impulse_retest_s5",
    "impulse_momentum_s5",
    "pullback_s5",
    "vwap_magnet_s5",
    "stop_run_reversal",
}

EVIDENCE_ONLY = {
    "basic",
    "fast_scalp",
    "london_momentum",
    "ma_rsi_macd",
    "macro_core",
    "macro_tech_fusion",
    "manual_spike",
    "manual_swing",
    "micro_core",
    "micro_multistrat",
    "micro_pullback_fib",
    "micro_range_revert_lite",
    "mirror_spike",
    "mirror_spike_s5",
    "mirror_spike_tight",
    "mm_lite",
    "mtf_breakout",
    "onepip_maker_s1",
    "pullback_runner_s5",
    "pullback_scalp",
    "range_bounce",
    "range_compression_break",
    "scalp_core",
    "scalp_multistrat",
    "scalp_precision",
    "scalp_reversal_nwave",
    "spike_reversal",
    "squeeze_break_s5",
    "tech_fusion",
    "trend_pullback",
    "vol_spike_rider",
    "vol_squeeze",
}

WRAPPER_RELATIONSHIPS = {
    "scalp_ping_5s_b": "scalp_ping_5s",
    "scalp_ping_5s_c": "scalp_ping_5s",
    "scalp_ping_5s_d": "scalp_ping_5s",
    "scalp_ping_5s_flow": "scalp_ping_5s",
    "scalp_macd_rsi_div_b": "scalp_macd_rsi_div",
    "compression_revert": "micro_runtime",
    "level_reactor": "micro_runtime",
    "momentum_burst": "micro_runtime",
    "momentum_pulse": "micro_runtime",
    "momentum_stack": "micro_runtime",
    "pullback_ema": "micro_runtime",
    "range_break": "micro_runtime",
    "trend_momentum": "micro_runtime",
    "trend_retest": "micro_runtime",
    "micro_vwap_revert": "micro_runtime",
    "vwap_bound_revert": "micro_runtime",
    "scalp_drought_revert": "scalp_precision",
    "scalp_precision_lowvol": "scalp_precision",
    "scalp_vwap_revert": "scalp_precision",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--priority-replay", type=Path, required=True)
    parser.add_argument("--vm-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inventory_payload = json.loads(args.inventory.read_text(encoding="utf-8"))
    priority_payload = json.loads(args.priority_replay.read_text(encoding="utf-8"))
    vm_payload = json.loads(args.vm_evidence.read_text(encoding="utf-8"))
    inventory = list(inventory_payload["inventory"])
    family_ids = {str(row["strategy_id"]) for row in inventory}
    attempted = {
        str(result["strategy"]): result
        for result in priority_payload["priority_results"]
    }
    attempted_ids = set(attempted)
    recoverable_remaining = (
        family_ids - BASELINE_EVALUATED - attempted_ids - EVIDENCE_ONLY
    )
    expected_union = (
        BASELINE_EVALUATED
        | attempted_ids
        | recoverable_remaining
        | EVIDENCE_ONLY
    )
    if len(inventory) != 82 or family_ids != expected_union:
        raise SystemExit(
            "inventory partition mismatch: "
            f"rows={len(inventory)} union={len(expected_union)} "
            f"missing={sorted(family_ids - expected_union)} "
            f"extra={sorted(expected_union - family_ids)}"
        )

    updated: list[dict[str, object]] = []
    for source_row in inventory:
        row = dict(source_row)
        family = str(row["strategy_id"])
        if family in BASELINE_EVALUATED:
            current_status = "evaluated_or_attempted_before_this_run"
            decision = "see_prior_results"
        elif family in attempted_ids:
            current_status = "evaluated_or_attempted_priority_run"
            decision = str(attempted[family]["decision"])
            row["priority_replay"] = {
                "entry_cohort_size": attempted[family]["entry_cohort_size"],
                "best_arm": attempted[family]["best_arm"],
                "best_protected_arm": attempted[family]["best_protected_arm"],
            }
        elif family in EVIDENCE_ONLY:
            current_status = "unevaluated_evidence_only"
            decision = "証拠不足"
        else:
            current_status = "unevaluated_implementation_recoverable"
            decision = "未評価"
        row["current_evaluation_status"] = current_status
        row["current_decision"] = decision
        row["wrapper_target"] = WRAPPER_RELATIONSHIPS.get(family)
        updated.append(row)

    output_payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authority": "NONE",
        "live_permission": False,
        "broker_order_mutation": False,
        "counts": {
            "normalized_families": len(inventory),
            "evaluated_or_attempted": len(BASELINE_EVALUATED) + len(attempted_ids),
            "not_yet_evaluated": len(recoverable_remaining) + len(EVIDENCE_ONLY),
            "unevaluated_implementation_recoverable": len(recoverable_remaining),
            "unevaluated_evidence_only": len(EVIDENCE_ONLY),
            "thin_wrapper_relationships": len(WRAPPER_RELATIONSHIPS),
            "vm_instances": len(vm_payload.get("instances") or []),
            "direct_vm_family_links": sum(
                len(value)
                for value in (vm_payload.get("instance_worker_links") or {}).values()
            ),
            "systemd_units": len(vm_payload.get("systemd_units") or []),
        },
        "partitions": {
            "evaluated_or_attempted_before_this_run": sorted(BASELINE_EVALUATED),
            "evaluated_or_attempted_priority_run": sorted(attempted_ids),
            "unevaluated_implementation_recoverable": sorted(recoverable_remaining),
            "unevaluated_evidence_only": sorted(EVIDENCE_ONLY),
        },
        "wrapper_relationships": WRAPPER_RELATIONSHIPS,
        "vm_instances": vm_payload.get("instances") or [],
        "instance_worker_links": vm_payload.get("instance_worker_links") or {},
        "families": updated,
        "sources": {
            "inventory": str(args.inventory),
            "priority_replay": str(args.priority_replay),
            "vm_evidence": str(args.vm_evidence),
        },
    }
    args.output.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_payload["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

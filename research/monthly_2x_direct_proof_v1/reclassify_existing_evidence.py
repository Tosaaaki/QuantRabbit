from __future__ import annotations

from decimal import Decimal, getcontext
from hashlib import sha256
import json
from math import log
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = Path(__file__).with_name("MONTHLY_2X_DIRECT_PROOF_V1.json")

EXPECTED_SOURCES = {
    "research/monthly_3x_direct_proof/2026-08-10/report_v1.json": "bac260fa7c5e819cca7c5b46a6e774c45737c5fd2a9a36477c7de910418ec92c",
    "research/monthly_3x_direct_proof/2026-08-10/x_mtf_result_v3.json": "bb5cd3df0755b608cb75cb6c487eee3ee0ca449e3544a0ad8f117c6348ac0220",
    "research/monthly_3x_direct_proof/2026-08-10/mtf_long_window_report_v4.json": "6ea40d0dc40294cae942f3ca3ee1d4b4c0b0a4fc5a3e7535de08115701f61cf9",
    "research/monthly_3x_direct_proof/2026-08-10/family_fusion_report_v5.json": "0b865e6d86867b262a065b57e81cfef8989d715739829cc4ab91de69d005a345",
    "research/monthly_3x_direct_proof/2026-08-10/currency_rotation_v6_report.json": "abe3badaf6310566623d556e387d2c969ac3318df2a3b8cc8b86cbb1281025e0",
    "research/monthly_3x_direct_proof/2026-08-10/slow_currency_rotation_v7_report.json": "36b28fd45a27d40d7f59656a2b7c08ed10abd7c37e5b49b5279188647f1c72b9",
    "research/monthly_3x_growth_engine/2026-08-10/growth_report_v1.json": "424346e8574300b369e89d6cfe4271600d80cf7a40e28bb92140822f31aafb55",
    "research/monthly_3x_growth_engine/2026-08-10/growth_grid_v1.jsonl": "9ff2e311fdb8d29fbf3b4bdf666dbd555a0cfe0edbe8b0202a8ea92152806c1e",
    "research/financial_oracle_v2/2026-08-10/exit_report_v1.json": "a5d404fa054ed93d0fc34634374223c390297f87fc42b18662e1ffe150d5cf0b",
    "research/loss_close_paired_robustness/robustness_report_v2.json": "ce44d29152725f6e6361f28ef5b41224459cbec11be4daffa71377f47cfe7069",
    "research/python_ecosystem_audit/2026-08-10/real_shadow_report.json": "bab89ab3a1e7d9825379fe8a7a66195547cdf426d5d211917ecfdbae705907f8",
    "research/system_utilization_rca/2026-08-10/utilization_report_v1.json": "3eb9a953068f15764c8a4b2967619ecd4737b338a61fff00fabee98f6a1022b8",
    "research/decision_time_execution_evidence/2026-08-10/coverage_report_v1.json": "ee84f61dd89a6069155c4e2bdc4e8200460bb9be2d0a361857e1e10eb0da25b1",
    "research/always_available_profit_proof/2026-08-10/proof_report_v1.json": "b681499d0d7c0267ea22067f46114f1d394bbf66dace71e12634824e75d12123",
    "research/capital_preservation_gate/2026-08-10/report_v1.json": "00c9b74b2c98b4ef584fe0b76a5b88658b52425d5c44f6b142a3c8d968b08a4f"
}


def read_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_jsonl(relative: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (ROOT / relative).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_sources() -> dict[str, str]:
    actual = {
        relative: sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in EXPECTED_SOURCES
    }
    mismatches = {
        relative: digest
        for relative, digest in actual.items()
        if digest != EXPECTED_SOURCES[relative]
    }
    if mismatches:
        raise RuntimeError(f"source hash mismatch: {sorted(mismatches)}")
    return actual


def main() -> dict[str, Any]:
    getcontext().prec = 50
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    source_hashes = verify_sources()

    direct = read_json("research/monthly_3x_direct_proof/2026-08-10/report_v1.json")
    x_mtf = read_json("research/monthly_3x_direct_proof/2026-08-10/x_mtf_result_v3.json")
    mtf = read_json("research/monthly_3x_direct_proof/2026-08-10/mtf_long_window_report_v4.json")
    fusion = read_json("research/monthly_3x_direct_proof/2026-08-10/family_fusion_report_v5.json")
    rotation = read_json("research/monthly_3x_direct_proof/2026-08-10/currency_rotation_v6_report.json")
    slow_rotation = read_json("research/monthly_3x_direct_proof/2026-08-10/slow_currency_rotation_v7_report.json")
    growth = read_json("research/monthly_3x_growth_engine/2026-08-10/growth_report_v1.json")
    growth_grid = read_jsonl("research/monthly_3x_growth_engine/2026-08-10/growth_grid_v1.jsonl")
    exits = read_json("research/financial_oracle_v2/2026-08-10/exit_report_v1.json")
    hedge = read_json("research/loss_close_paired_robustness/robustness_report_v2.json")
    oss = read_json("research/python_ecosystem_audit/2026-08-10/real_shadow_report.json")
    utilization = read_json("research/system_utilization_rca/2026-08-10/utilization_report_v1.json")
    execution = read_json("research/decision_time_execution_evidence/2026-08-10/coverage_report_v1.json")
    positive_vehicle = read_json("research/always_available_profit_proof/2026-08-10/proof_report_v1.json")
    preservation = read_json("research/capital_preservation_gate/2026-08-10/report_v1.json")

    baseline = Decimal(
        str(
            growth["corrected_64d_validation_baseline_at_1x_75pct_cap"][
                "rolling_30d_equity_multiple_max"
            ]
        )
    )
    target = Decimal(str(contract["capital_and_target"]["minimum_rolling_30d_multiple_after_all_costs"]))
    start = Decimal(str(contract["capital_and_target"]["starting_equity_jpy"]))
    linear_gap = target - baseline
    required_factor = target / baseline
    log_gap = log(float(target)) - log(float(baseline))

    capped = [
        row
        for row in growth_grid
        if row["window"] == "QUADRUPLE_64D"
        and row["split"] == "VALIDATION"
        and row["cohort_margin_peak_jpy"] <= contract["capital_and_target"]["gross_margin_cap_jpy"]
        and row["realized_max_drawdown_jpy"] <= contract["capital_and_target"]["maximum_drawdown_jpy"]
        and row["ruin"] is False
    ]
    capped_best = max(capped, key=lambda row: row["rolling_30d_equity_multiple_max"])
    two_x_rows = [row for row in capped if row["rolling_30d_equity_multiple_max"] >= float(target)]

    families = [
        {
            "family": "existing_multidimensional_sweeps",
            "status": "TARGET_PATH_NOT_YET_PROVEN",
            "evidence": {
                "grid_rows": direct["grid_rows"],
                "stable_multiwindow_candidates": direct["stable_multiwindow_candidate_count"],
                "prior_target_passes": direct["monthly_3x_pass_count"]
            },
            "decisive_reason": "no stable multiwindow candidate exists before applying either the 3x or 2x terminal target"
        },
        {
            "family": "technical_fusion_and_mtf",
            "status": "TARGET_PATH_NOT_YET_PROVEN",
            "evidence": {
                "mtf_train_plateau": mtf["train_plateau_count"],
                "mtf_stable_32d_64d": mtf["stable_32d_64d_count"],
                "fusion_train_plateau": fusion["train_plateau_count"],
                "fusion_stable_32d_64d": fusion["stable_32d_64d_count"],
                "x_mtf_validation_eligible": x_mtf["counts"]["validation_eligible"]
            },
            "decisive_reason": "MTF and fusion have zero TRAIN plateau and zero 32d/64d stability; the X-MTF subarm is separately not evaluable"
        },
        {
            "family": "trailing_break_even_partial_take_profit",
            "status": "NOT_EVALUABLE",
            "evidence": {
                "exit_status": exits["status"],
                "strict_path_episodes": exits["strict_path_episodes"],
                "blockers": exits["decisive_blockers"]
            },
            "decisive_reason": "causal fee/financing, account margin/netting, partial-fill depth, and unwind evidence are incomplete"
        },
        {
            "family": "hedging",
            "status": "TARGET_PATH_NOT_YET_PROVEN",
            "evidence": {
                "overall_decision": hedge["overall_decision"],
                "arm_decisions": hedge["final_decisions"]
            },
            "decisive_reason": "all four preregistered hedge arms were rejected and the STOP-loss cohort does not establish monthly entry-strategy profitability"
        },
        {
            "family": "dynamic_lot_inventory_exposure",
            "status": "TARGET_PATH_NOT_YET_PROVEN",
            "evidence": {
                "cap_compliant_best_rolling_30d_multiple": capped_best["rolling_30d_equity_multiple_max"],
                "cap_compliant_best_paired_lcb_jpy": capped_best["paired_lcb_jpy"],
                "cap_compliant_2x_rows": len(two_x_rows),
                "validation_success_count": growth["validation_success_count"]
            },
            "decisive_reason": "no cap-compliant row reaches 2x; the best cap-compliant row also has a negative paired LCB and no TRAIN plateau"
        },
        {
            "family": "x_derived_methods",
            "status": "NOT_EVALUABLE",
            "evidence": {
                "x_mtf_status": x_mtf["status"],
                "x_mtf_validation_eligible": x_mtf["counts"]["validation_eligible"],
                "x_system_classifications": [
                    item["classification"]
                    for item in utilization["inventory"]
                    if item["system_id"].startswith("x_")
                ]
            },
            "decisive_reason": "the fixed archive has no eligible validation trades and no complete 30-day proof interval"
        },
        {
            "family": "oss_adapters",
            "status": "TARGET_PATH_NOT_YET_PROVEN",
            "evidence": {
                "profitability_increment_jpy": oss["profitability_increment_jpy_attributed_to_adapters"],
                "strategy_decision": oss["strategy_decision"]
            },
            "decisive_reason": "the adapters preserve calculation parity but create zero profitability increment and no trading edge"
        },
        {
            "family": "currency_rotation_28",
            "status": "TARGET_PATH_NOT_YET_PROVEN",
            "evidence": {
                "fast_sources": rotation["source_count"],
                "fast_train_plateau": rotation["train_plateau_count"],
                "fast_stable_32d_64d": rotation["stable_32d_64d_count"],
                "slow_train_plateau": slow_rotation["train_plateau_count"],
                "slow_stable_32d_64d": slow_rotation["stable_32d_64d_count"]
            },
            "decisive_reason": "both fast and slow 28-pair rotations have zero TRAIN plateau and zero 32d/64d stability"
        },
        {
            "family": "conditional_positive_vehicle",
            "status": "TARGET_PATH_NOT_YET_PROVEN",
            "evidence": {
                "trades": positive_vehicle["proof"]["exact_realized_after_bidask"]["trades"],
                "net_jpy": positive_vehicle["proof"]["exact_realized_after_bidask"]["net_jpy"],
                "next_proof_count": positive_vehicle["next_proof_count"]
            },
            "decisive_reason": "four positive receipts prove conditional existence only, not 30-day frequency, stability, margin, or 2x compounding"
        },
        {
            "family": "capital_preservation_gate",
            "status": "NOT_EVALUABLE",
            "evidence": {
                "actions": preservation["actions"],
                "profit_generation_status": preservation["profit_generation_status"],
                "strict_policy_net_jpy": preservation["strict_policy_realized_net_jpy"]
            },
            "decisive_reason": "the gate proves fail-closed capital preservation, not profit generation"
        }
    ]

    return {
        "contract": contract["contract_id"],
        "inheritance_commit": contract["scope"]["inheritance_commit"],
        "source_hashes_verified": True,
        "source_hashes": source_hashes,
        "gap": {
            "baseline_rolling_30d_multiple": float(baseline),
            "target_multiple": float(target),
            "linear_multiple_gap": float(linear_gap),
            "linear_equity_gap_jpy": float(start * linear_gap),
            "required_factor_from_baseline": float(required_factor),
            "required_increment_from_baseline_pct": float((required_factor - 1) * 100),
            "log_growth_gap": log_gap,
            "linear_and_log_agree_on_shortfall": linear_gap > 0 and log_gap > 0
        },
        "families": families,
        "system_admission": {
            "strict_eligible_decisions": execution["strict_eligible"],
            "stage_coverage": execution["overall_stage_coverage"],
            "status": "FAIL_CLOSED_FOR_EXECUTABLE_2X_PROOF"
        },
        "overall_status": "TARGET_PATH_NOT_YET_PROVEN",
        "dominant_blocker": "no preregistered family has both a positive TRAIN LCB plateau and stable 32d/64d validation; strict decision-time cost, margin, fill, and unwind coverage is zero",
        "next_independent_hypothesis": {
            "id": "EVENT_DRIVEN_CROSS_ASSET_DISLOCATION_V1",
            "mechanism": "timestamped macro-release surprise plus synchronized cross-asset reaction and side-aware FX bid/ask execution",
            "independence": "new event-driven opportunity and data source, not a threshold retune of observed technical, MTF, hedge, sizing, or rotation families",
            "required_before_evaluation": "preregister source lineage, completed-bar/event chronology, after-cost fill/unwind/margin evidence, and keep holdout unread"
        },
        "holdout_read": False,
        "live_paper_broker_order_deploy_touched": False
    }


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2, sort_keys=True))

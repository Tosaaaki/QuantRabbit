#!/usr/bin/env python3
"""Summarize a complete 12-coordinate paired inventory counterfactual."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
from typing import Any

from quant_rabbit.dojo_paired_inventory_counterfactual import (
    CADENCE_IDS,
    RESULT_CONTRACT,
)
from quant_rabbit.dojo_portfolio_replay_reducer import canonical_portfolio_sha256


def _load(path: Path) -> dict[str, Any]:
    with path.resolve(strict=True).open("rb") as handle:
        return json.load(handle)


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _valid_result(path: Path, plan_sha256: str) -> dict[str, Any]:
    result = _load(path)
    claimed = result.get("result_sha256")
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    if (
        result.get("contract") != RESULT_CONTRACT
        or result.get("plan_sha256") != plan_sha256
        or claimed != canonical_portfolio_sha256(body)
        or result.get("classification") != "EXPERIMENTAL_UNRANKED"
    ):
        raise ValueError(f"invalid paired result: {path}")
    cadence_rows = result.get("cadence_rows", [])
    if (
        len(cadence_rows) != len(CADENCE_IDS)
        or {row.get("cadence_id") for row in cadence_rows} != set(CADENCE_IDS)
    ):
        raise ValueError(f"cadence denominator is invalid: {path}")
    for cadence_row in cadence_rows:
        if cadence_row.get("provider_model_call_count") != 0:
            raise ValueError(f"unexpected provider-model call: {path}")
        for decision in cadence_row.get("intervention_audit_log", []):
            packet = decision.get("packet", {})
            if (
                decision.get("future_information_used") is not False
                or int(decision["input_available_through_epoch"])
                > int(decision["decision_epoch"])
                or packet.get("terminal_result_visible") is not False
                or packet.get("future_quote_visible") is not False
                or packet.get("append_wall_clock_visible") is not False
            ):
                raise ValueError(f"causal audit failed: {path}")
    return result


def _oos_aggregate(cadence_row: dict[str, Any]) -> dict[str, float]:
    blocks = cadence_row["oos_block_rows"]
    if (
        len(blocks) != 8
        or len({row["block_id"] for row in blocks}) != 8
        or any(row.get("status") != "MEASURED_EXPERIMENTAL" for row in blocks)
    ):
        raise ValueError("OOS fixed denominator is incomplete")
    return {
        "bot_only_net_jpy": sum(float(row["bot_only_net_jpy"]) for row in blocks),
        "ai_managed_net_jpy": sum(
            float(row["ai_managed_net_jpy"]) for row in blocks
        ),
        "bot_only_max_within_block_drawdown_fraction": max(
            float(row["bot_only_max_drawdown_fraction"]) for row in blocks
        ),
        "ai_max_within_block_drawdown_fraction": max(
            float(row["ai_managed_max_drawdown_fraction"]) for row in blocks
        ),
        "bot_only_peak_margin_usage_fraction": max(
            float(row["bot_only_peak_margin_usage_fraction"]) for row in blocks
        ),
        "ai_peak_margin_usage_fraction": max(
            float(row["ai_managed_peak_margin_usage_fraction"]) for row in blocks
        ),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load(args.plan)
    job = _load(args.job_result)
    runtimes = _load(args.coordinate_runtimes)["coordinate_runtimes"]
    baseline = job["portfolio_results_by_coordinate"]
    paths = sorted(args.results_dir.glob("*.paired-inventory.json"))
    if len(paths) != 12:
        raise ValueError("fixed denominator requires exactly 12 result files")
    results: dict[str, dict[str, Any]] = {}
    result_file_manifest = []
    for path in paths:
        result = _valid_result(path, plan["plan_sha256"])
        coordinate_id = result["coordinate_id"]
        if coordinate_id in results:
            raise ValueError(f"duplicate coordinate result: {coordinate_id}")
        results[coordinate_id] = result
        result_file_manifest.append(
            {
                "coordinate_id": coordinate_id,
                "absolute_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "embedded_result_sha256": result["result_sha256"],
                "intervention_audit_entry_count": sum(
                    len(row["intervention_audit_log"])
                    for row in result["cadence_rows"]
                ),
            }
        )
    if set(results) != set(baseline):
        raise ValueError("paired results differ from the immutable 12-coordinate baseline")

    rows_by_cadence: dict[str, list[dict[str, Any]]] = {
        cadence: [] for cadence in CADENCE_IDS
    }
    for coordinate_id, result in sorted(results.items()):
        base = baseline[coordinate_id]
        scenario = runtimes[coordinate_id]["cost_scenario"]
        family = result["family_id"]
        for cadence_row in result["cadence_rows"]:
            oos = _oos_aggregate(cadence_row)
            portfolio = cadence_row["portfolio_result"]
            base_net = float(base["end_equity_jpy"]) - float(
                base["start_equity_jpy"]
            )
            ai_net = float(portfolio["end_equity_jpy"]) - float(
                portfolio["start_equity_jpy"]
            )
            rows_by_cadence[cadence_row["cadence_id"]].append(
                {
                    "coordinate_id": coordinate_id,
                    "family_id": family,
                    "cost_scenario": scenario,
                    "bot_only_net_jpy": base_net,
                    "ai_managed_net_jpy": ai_net,
                    "net_delta_jpy": ai_net - base_net,
                    "paired_oos_policy_effect_net_delta_jpy": ai_net - base_net,
                    "bot_only_full_run_realized_pnl_jpy": base[
                        "realized_pnl_jpy"
                    ],
                    "ai_full_run_realized_pnl_jpy": portfolio[
                        "realized_pnl_jpy"
                    ],
                    "bot_only_full_run_expectancy_jpy_per_trade": (
                        None
                        if int(base["trade_count"]) == 0
                        else base_net / int(base["trade_count"])
                    ),
                    "bot_only_full_run_trade_count": base["trade_count"],
                    "ai_full_run_trade_count": portfolio["trade_count"],
                    "bot_only_win_rate": None,
                    "bot_only_profit_factor": None,
                    "ai_close_metrics": cadence_row["ai_close_metrics"],
                    "bot_only_full_run_max_drawdown_fraction": base[
                        "max_drawdown_fraction"
                    ],
                    "ai_full_run_max_drawdown_fraction": portfolio[
                        "max_drawdown_fraction"
                    ],
                    "full_run_max_drawdown_delta": (
                        float(portfolio["max_drawdown_fraction"])
                        - float(base["max_drawdown_fraction"])
                    ),
                    "bot_only_max_within_block_drawdown_fraction": oos[
                        "bot_only_max_within_block_drawdown_fraction"
                    ],
                    "ai_max_within_block_drawdown_fraction": oos[
                        "ai_max_within_block_drawdown_fraction"
                    ],
                    "max_within_block_drawdown_delta": (
                        oos["ai_max_within_block_drawdown_fraction"]
                        - oos["bot_only_max_within_block_drawdown_fraction"]
                    ),
                    "bot_only_oos_balance_change_proxy_jpy": oos[
                        "bot_only_net_jpy"
                    ],
                    "ai_oos_balance_change_proxy_jpy": oos[
                        "ai_managed_net_jpy"
                    ],
                    "bot_only_peak_margin_usage_fraction": base[
                        "peak_margin_usage_fraction"
                    ],
                    "ai_peak_margin_usage_fraction": portfolio[
                        "peak_margin_usage_fraction"
                    ],
                    "bot_only_margin_closeouts": base["margin_closeouts"],
                    "ai_margin_closeouts": portfolio["margin_closeouts"],
                    "bot_only_ruin_events": base["ruin_event_count"],
                    "ai_ruin_events": portfolio["ruin_event_count"],
                    "bot_only_full_run_transaction_cost_jpy": base[
                        "transaction_cost_jpy"
                    ],
                    "ai_full_run_transaction_cost_jpy": portfolio[
                        "transaction_cost_jpy"
                    ],
                    "ai_provider_cost_jpy": 0.0,
                    "model_authored_policy_evaluation_count": cadence_row.get(
                        "model_authored_policy_evaluation_count",
                        cadence_row["ai_call_count"],
                    ),
                    "provider_model_call_count": cadence_row[
                        "provider_model_call_count"
                    ],
                    "intervention_count": cadence_row["intervention_count"],
                    "loss_avoided_proxy_jpy": max(0.0, ai_net - base_net),
                    "missed_upside_proxy_jpy": max(0.0, base_net - ai_net),
                    "tp_profit_retained_jpy": None,
                    "oos_block_rows": cadence_row["oos_block_rows"],
                }
            )

    cadence_summary = []
    for cadence in CADENCE_IDS:
        rows = rows_by_cadence[cadence]
        deltas = [float(row["net_delta_jpy"]) for row in rows]
        cadence_summary.append(
            {
                "cadence_id": cadence,
                "coordinate_count": len(rows),
                "positive_delta_coordinate_count": sum(value > 0 for value in deltas),
                "equal_delta_coordinate_count": sum(value == 0 for value in deltas),
                "negative_delta_coordinate_count": sum(value < 0 for value in deltas),
                "median_net_delta_jpy": median(deltas),
                "minimum_net_delta_jpy": min(deltas),
                "maximum_net_delta_jpy": max(deltas),
                "median_max_within_block_drawdown_delta": median(
                    float(row["max_within_block_drawdown_delta"]) for row in rows
                ),
                "median_full_run_max_drawdown_delta": median(
                    float(row["full_run_max_drawdown_delta"]) for row in rows
                ),
                "aggregate_model_authored_decision_count": sum(
                    int(row["model_authored_policy_evaluation_count"])
                    for row in rows
                ),
                "aggregate_provider_model_call_count": 0,
            }
        )
    cadence_summary.sort(
        key=lambda row: (
            -float(row["median_net_delta_jpy"]),
            -int(row["positive_delta_coordinate_count"]),
            float(row["median_max_within_block_drawdown_delta"]),
            int(row["aggregate_model_authored_decision_count"]),
            row["cadence_id"],
        )
    )
    experimental_best = cadence_summary[0]["cadence_id"]
    comparison_rows = rows_by_cadence[experimental_best]

    paired_family_gates = []
    for family in sorted({row["family_id"] for row in comparison_rows}):
        family_rows = [
            row for row in comparison_rows if row["family_id"] == family
        ]
        scenarios = {row["cost_scenario"] for row in family_rows}
        passed_economic_direction = (
            scenarios == {"BASE", "STRESS"}
            and all(float(row["net_delta_jpy"]) > 0 for row in family_rows)
            and all(
                float(row["full_run_max_drawdown_delta"]) <= 0
                for row in family_rows
            )
            and all(
                float(row["max_within_block_drawdown_delta"]) <= 0
                for row in family_rows
            )
            and all(
                int(row["ai_margin_closeouts"])
                <= int(row["bot_only_margin_closeouts"])
                and int(row["ai_ruin_events"]) <= int(row["bot_only_ruin_events"])
                for row in family_rows
            )
        )
        paired_family_gates.append(
            {
                "family_id": family,
                "cadence_id": experimental_best,
                "base_stress_direction_consistent_improvement": (
                    passed_economic_direction
                ),
                "promotion_gate_passed": False,
                "promotion_blockers": [
                    "SOURCE_QUOTE_COVERAGE_NOT_PROVED",
                    "WORN_TRAIN_RESEARCHER_PRIOR_AGGREGATE_OUTCOME_EXPOSURE",
                    "ACTUAL_MODEL_CHECKPOINT_CALLS_NOT_EXECUTED",
                    "BOT_ONLY_TRADE_LEVEL_PROFIT_FACTOR_NOT_IN_IMMUTABLE_EVIDENCE",
                    "TP_PROFIT_RETAINED_NOT_IDENTIFIABLE_FROM_IMMUTABLE_BASELINE",
                ],
            }
        )

    body = {
        "contract": "QR_DOJO_PAIRED_INVENTORY_COUNTERFACTUAL_SUMMARY_V1",
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "source_job_sha256": job["job_sha256"],
        "classification": "EXPERIMENTAL_UNRANKED",
        "fixed_coordinate_count": 12,
        "fixed_cadence_count": 7,
        "paired_oos_measured_cell_count": 84,
        "phase_b_model_authored_policy_cell_count": 84,
        "phase_b_actual_model_checkpoint_cell_count": 0,
        "source_quote_coverage_proved": False,
        "bot_only_trade_level_profit_factor_available": False,
        "tp_profit_retained_available": False,
        "oos_drawdown_scope": (
            "MAXIMUM_WITHIN_EACH_NON_OVERLAPPING_OOS_BLOCK; "
            "CONTINUOUS_CROSS_BLOCK_OOS_DRAWDOWN_NOT_RECORDED_IN_V2"
        ),
        "paired_net_effect_scope": (
            "FULL_RUN_FINAL_SETTLEMENT_NET_DIFFERENCE; CALIBRATION PREFIX IS "
            "IDENTICAL AND POLICY ACTIVATES ONLY IN OOS, SO THE BETWEEN-ARM "
            "DELTA IS THE OOS POLICY EFFECT. ABSOLUTE NET VALUES RETAIN THE "
            "SHARED CALIBRATION PREFIX."
        ),
        "oos_transaction_cost_scope": (
            "FULL_RUN NET AND PAIRED DELTA ARE AFTER EXECUTION COST AND FINAL "
            "SETTLEMENT; ABSOLUTE TRANSACTION COST FIELDS ARE FULL_RUN BECAUSE "
            "V2 DID NOT SNAPSHOT COST BY OOS BLOCK"
        ),
        "result_file_manifest": sorted(
            result_file_manifest, key=lambda row: row["coordinate_id"]
        ),
        "causal_audit_validated": True,
        "cadence_summary": cadence_summary,
        "experimental_best_cadence_id": experimental_best,
        "comparison_rows": comparison_rows,
        "paired_family_gates": paired_family_gates,
        "selected_policy": (
            "FROZEN_MODEL_AUTHORED_CAUSAL_INVENTORY_POLICY_WITH_"
            "CALIBRATION_STATE_PERCENTILES"
        ),
        "paper_shadow_or_olhc_candidates": [],
        "ranking_status": "UNRANKED",
        "profit_guaranteed": False,
        "coordinate_account_results_sum_allowed": False,
        "authority": {
            "research_only": True,
            "paper_replay_only": True,
            "live_permission": False,
            "broker_mutation_allowed": False,
            "order_authority": "NONE",
            "automatic_deployment_allowed": False,
            "promotion_eligible": False,
        },
    }
    return {**body, "summary_sha256": canonical_portfolio_sha256(body)}


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# r13 2025-01 OHLC paired inventory counterfactual",
        "",
        f"- Status: `{summary['classification']}`",
        f"- Fixed denominator: {summary['fixed_coordinate_count']} coordinates × "
        f"{summary['fixed_cadence_count']} cadences = "
        f"{summary['paired_oos_measured_cell_count']} measured cells",
        f"- Experimental best cadence: `{summary['experimental_best_cadence_id']}`",
        "- Actual provider-model checkpoint cells: 0/84",
        "- OLHC/paper-shadow candidates: none",
        "",
        "## Twelve-account comparison",
        "",
        "| family | cost | bot full-run net | policy full-run net | paired OOS effect | bot DD | policy DD | policy OOS+terminal PF | policy expectancy | evals | provider calls | interventions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        summary["comparison_rows"],
        key=lambda item: (item["family_id"], item["cost_scenario"]),
    ):
        ai_pf = row["ai_close_metrics"]["profit_factor"]
        ai_expectancy = row["ai_close_metrics"]["expectancy_jpy_per_close"]
        lines.append(
            f"| {row['family_id']} | {row['cost_scenario']} | "
            f"{row['bot_only_net_jpy']:.2f} | {row['ai_managed_net_jpy']:.2f} | "
            f"{row['paired_oos_policy_effect_net_delta_jpy']:.2f} | "
            f"{row['bot_only_full_run_max_drawdown_fraction']:.4f} | "
            f"{row['ai_full_run_max_drawdown_fraction']:.4f} | "
            f"{'N/A' if ai_pf is None else f'{ai_pf:.3f}'} | "
            f"{'N/A' if ai_expectancy is None else f'{ai_expectancy:.2f}'} | "
            f"{row['model_authored_policy_evaluation_count']} | "
            f"{row['provider_model_call_count']} | "
            f"{row['intervention_count']} |"
        )
    lines.extend(
        [
            "",
            "Both arms have an identical calibration prefix and the policy only "
            "activates in OOS. Therefore the final-settlement full-run net "
            "difference is the paired OOS policy effect; absolute net and DD "
            "retain the shared calibration prefix. The eight block balance "
            "proxies are diagnostic only because they exclude the terminal "
            "flat-settlement boundary. Bot-only PF/win rate and TP-profit-retained "
            "are N/A because the immutable r13 baseline does not expose OOS "
            "trade-level gross wins/losses or TP-attributed cash. Account rows "
            "are independent and must not be summed.",
            "",
            "The `policy evals` column is deterministic frozen-policy evaluation, "
            "not an AI provider call. Provider calls and provider cost are zero.",
            "",
            "## Economic and safety detail",
            "",
            "| family | cost | bot realized | policy realized | bot expectancy/trade | policy win rate | margin peak bot/policy | margin calls bot/policy | ruin bot/policy | execution cost bot/policy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        summary["comparison_rows"],
        key=lambda item: (item["family_id"], item["cost_scenario"]),
    ):
        bot_expectancy = row["bot_only_full_run_expectancy_jpy_per_trade"]
        ai_win_rate = row["ai_close_metrics"]["win_rate"]
        lines.append(
            f"| {row['family_id']} | {row['cost_scenario']} | "
            f"{row['bot_only_full_run_realized_pnl_jpy']:.2f} | "
            f"{row['ai_full_run_realized_pnl_jpy']:.2f} | "
            f"{'N/A' if bot_expectancy is None else f'{bot_expectancy:.2f}'} | "
            f"{'N/A' if ai_win_rate is None else f'{ai_win_rate:.3f}'} | "
            f"{row['bot_only_peak_margin_usage_fraction']:.4f}/"
            f"{row['ai_peak_margin_usage_fraction']:.4f} | "
            f"{row['bot_only_margin_closeouts']}/"
            f"{row['ai_margin_closeouts']} | "
            f"{row['bot_only_ruin_events']}/{row['ai_ruin_events']} | "
            f"{row['bot_only_full_run_transaction_cost_jpy']:.2f}/"
            f"{row['ai_full_run_transaction_cost_jpy']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Bot-only win rate and PF remain N/A. All intervention audit entries "
            "passed the no-future/no-terminal/no-wall-clock checks and are retained "
            "in the hash-manifested raw result files.",
            "",
            "## Promotion decision",
            "",
            "None. Source quote coverage is unproved, the month is worn TRAIN "
            "with prior aggregate outcome exposure, and no actual provider model "
            "was called at checkpoints. The paired deltas are experimental only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job-result", type=Path, required=True)
    parser.add_argument("--coordinate-runtimes", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args)
    _write_exclusive(
        args.output_json,
        json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n",
    )
    _write_exclusive(args.output_markdown, _markdown(summary).encode())
    print(summary["summary_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

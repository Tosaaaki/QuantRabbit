#!/usr/bin/env python3
"""Run the preregistered r13 AI inventory calibration and paired OOS cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_portfolio_replay_reducer import canonical_portfolio_sha256
from quant_rabbit.dojo_r13_ai_inventory_oos import (
    A_BOT_ONLY,
    B_INVENTORY_ONLY,
    CADENCE_SECONDS,
    C_FORECAST_INVENTORY,
    PHASE_A_POLICY_PROFILES,
    _atomic_json,
    deterministic_worker_response,
    load_prepared_coordinate,
    load_prepared_study,
    prepare_r13_inputs,
    score_forecasts_posthoc,
    simulate_partition,
)


def _metric_summary(cell: dict[str, Any]) -> dict[str, Any]:
    metrics = cell["metrics"]
    return {
        key: metrics[key]
        for key in (
            "net_after_all_costs_jpy",
            "profit_factor",
            "win_rate",
            "expectancy_jpy",
            "max_drawdown_fraction",
            "max_margin_utilization_fraction",
            "margin_call_count",
            "ruin_event_count",
            "tp_profit_retained_fraction",
            "loss_avoided_jpy",
            "missed_upside_jpy",
            "turnover_jpy",
            "scheduled_trade_count",
            "trade_count",
            "skipped_trade_count",
            "ai_decision_count",
            "ai_call_count",
            "ai_fallback_count",
            "ai_estimated_input_tokens",
            "ai_estimated_output_tokens",
            "ai_notional_cost_usd",
            "immutable_unobservable_background_cashflow_jpy",
        )
    }


def _admissible(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return (
        candidate["net_after_all_costs_jpy"]
        > baseline["net_after_all_costs_jpy"]
        and candidate["max_drawdown_fraction"]
        <= baseline["max_drawdown_fraction"] + 1e-12
        and candidate["margin_call_count"] <= baseline["margin_call_count"]
        and candidate["ruin_event_count"] <= baseline["ruin_event_count"]
    )


def _phase_b_operational_summary(
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in (B_INVENTORY_ONLY, C_FORECAST_INVENTORY):
        arm_cells = [cell for cell in cells if cell["arm"] == arm]
        audit_rows = [
            row
            for cell in arm_cells
            for row in cell["intervention_audit"]
        ]
        accepted_rows = [
            row
            for row in audit_rows
            if not row["fallback"]
            and row["attempted_worker_response_sha256"] is not None
        ]
        action_counts: dict[str, int] = {}
        for row in accepted_rows:
            action = str(row["action"]["type"])
            action_counts[action] = action_counts.get(action, 0) + 1
        summary[arm] = {
            "cell_count": len(arm_cells),
            "actual_ai_call_count": sum(
                cell["metrics"]["ai_call_count"] for cell in arm_cells
            ),
            "accepted_worker_response_count": len(accepted_rows),
            "schema_invalid_response_count": sum(
                row["failure_class"] == "DojoR13AIInventoryError"
                for row in audit_rows
            ),
            "preregistered_call_cap_fallback_count": sum(
                row["failure_class"] == "PREREGISTERED_CALL_CAP"
                for row in audit_rows
            ),
            "fallback_decision_count": sum(
                row["fallback"] for row in audit_rows
            ),
            "accepted_action_counts": action_counts,
            "estimated_input_tokens": sum(
                cell["metrics"]["ai_estimated_input_tokens"]
                for cell in arm_cells
            ),
            "estimated_output_tokens": sum(
                cell["metrics"]["ai_estimated_output_tokens"]
                for cell in arm_cells
            ),
            "notional_ai_cost_usd": sum(
                cell["metrics"]["ai_notional_cost_usd"]
                for cell in arm_cells
            ),
        }
    return summary


def calibrate(root: Path) -> dict[str, Any]:
    study, frames = load_prepared_study(root)
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for coordinate_ref in study["coordinates"]:
        coordinate = load_prepared_coordinate(
            root,
            study,
            coordinate_ref["coordinate_id"],
        )
        baseline_cell = simulate_partition(
            study=study,
            coordinate=coordinate,
            frames=frames,
            partition="CALIBRATION",
            arm=A_BOT_ONLY,
            cadence_id=None,
            policy_version="IMMUTABLE_R13",
            prompt_version="NONE",
            worker=None,
            capture_full_audit=False,
        )
        baseline = _metric_summary(baseline_cell)
        rows.append(
            {
                "coordinate_id": coordinate["coordinate_id"],
                "family_id": coordinate["family_id"],
                "cost_scenario": coordinate["cost_scenario"],
                "arm": A_BOT_ONLY,
                "cadence_id": None,
                "policy_version": "IMMUTABLE_R13",
                "admissible": True,
                "metrics": baseline,
                "cell_sha256": baseline_cell["cell_sha256"],
            }
        )
        for arm in (B_INVENTORY_ONLY, C_FORECAST_INVENTORY):
            candidates: list[dict[str, Any]] = []
            for policy_version in PHASE_A_POLICY_PROFILES:
                for cadence_id in CADENCE_SECONDS:
                    cell = simulate_partition(
                        study=study,
                        coordinate=coordinate,
                        frames=frames,
                        partition="CALIBRATION",
                        arm=arm,
                        cadence_id=cadence_id,
                        policy_version=policy_version,
                        prompt_version="PHASE_A_DETERMINISTIC_V1",
                        worker=lambda packet, policy=policy_version: (
                            deterministic_worker_response(
                                packet,
                                policy_id=policy,
                            )
                        ),
                        capture_full_audit=False,
                    )
                    metrics = _metric_summary(cell)
                    row = {
                        "coordinate_id": coordinate["coordinate_id"],
                        "family_id": coordinate["family_id"],
                        "cost_scenario": coordinate["cost_scenario"],
                        "arm": arm,
                        "cadence_id": cadence_id,
                        "policy_version": policy_version,
                        "admissible": _admissible(metrics, baseline),
                        "metrics": metrics,
                        "cell_sha256": cell["cell_sha256"],
                    }
                    rows.append(row)
                    candidates.append(row)
            eligible = [row for row in candidates if row["admissible"]]
            ranking_pool = eligible if eligible else candidates
            selected = max(
                ranking_pool,
                key=lambda row: (
                    row["metrics"]["net_after_all_costs_jpy"],
                    -row["metrics"]["max_drawdown_fraction"],
                    -row["metrics"]["ai_decision_count"],
                    row["cadence_id"],
                    row["policy_version"],
                ),
            )
            selections.append(
                {
                    "coordinate_id": coordinate["coordinate_id"],
                    "family_id": coordinate["family_id"],
                    "cost_scenario": coordinate["cost_scenario"],
                    "arm": arm,
                    "cadence_id": selected["cadence_id"],
                    "policy_version": selected["policy_version"],
                    "calibration_admissible": selected["admissible"],
                    "calibration_net_delta_jpy": (
                        selected["metrics"]["net_after_all_costs_jpy"]
                        - baseline["net_after_all_costs_jpy"]
                    ),
                    "selection_rule": (
                        "MAX_NET_THEN_MIN_DD_THEN_MIN_CALLS_WITH_RISK_GATES"
                        if eligible
                        else "NO_ADMISSIBLE_CELL_MAX_NET_DIAGNOSTIC_ONLY"
                    ),
                    "selected_cell_sha256": selected["cell_sha256"],
                }
            )
        print(
            json.dumps(
                {
                    "phase": "CALIBRATION_COORDINATE_COMPLETE",
                    "coordinate_id": coordinate["coordinate_id"],
                    "family_id": coordinate["family_id"],
                    "cost_scenario": coordinate["cost_scenario"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    body = {
        "contract": "QR_DOJO_R13_AI_INVENTORY_PHASE_A_CALIBRATION_V1",
        "schema_version": 1,
        "study_sha256": study["study_sha256"],
        "partition": "CALIBRATION",
        "future_oos_accessed_during_selection": False,
        "policy_profiles": PHASE_A_POLICY_PROFILES,
        "cadence_seconds": CADENCE_SECONDS,
        "selection_count": len(selections),
        "selections": selections,
        "cell_count": len(rows),
        "cells": rows,
    }
    result = {
        **body,
        "calibration_sha256": canonical_portfolio_sha256(body),
    }
    _atomic_json(root / "phase-a-calibration.json", result)
    return result


def deterministic_oos(root: Path) -> dict[str, Any]:
    study, frames = load_prepared_study(root)
    calibration_path = root / "phase-a-calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_claimed = calibration.pop("calibration_sha256")
    if calibration_claimed != canonical_portfolio_sha256(calibration):
        raise ValueError("calibration selection changed")
    calibration["calibration_sha256"] = calibration_claimed
    selection_index = {
        (row["coordinate_id"], row["arm"]): row
        for row in calibration["selections"]
    }
    rows: list[dict[str, Any]] = []
    for coordinate_ref in study["coordinates"]:
        coordinate = load_prepared_coordinate(
            root,
            study,
            coordinate_ref["coordinate_id"],
        )
        baseline_cell = simulate_partition(
            study=study,
            coordinate=coordinate,
            frames=frames,
            partition="OOS",
            arm=A_BOT_ONLY,
            cadence_id=None,
            policy_version="IMMUTABLE_R13",
            prompt_version="NONE",
            worker=None,
            capture_full_audit=False,
        )
        baseline = _metric_summary(baseline_cell)
        rows.append(
            {
                "coordinate_id": coordinate["coordinate_id"],
                "family_id": coordinate["family_id"],
                "cost_scenario": coordinate["cost_scenario"],
                "arm": A_BOT_ONLY,
                "cadence_id": None,
                "policy_version": "IMMUTABLE_R13",
                "metrics": baseline,
                "cell_sha256": baseline_cell["cell_sha256"],
            }
        )
        for arm in (B_INVENTORY_ONLY, C_FORECAST_INVENTORY):
            selection = selection_index[(coordinate["coordinate_id"], arm)]
            policy_version = selection["policy_version"]
            cell = simulate_partition(
                study=study,
                coordinate=coordinate,
                frames=frames,
                partition="OOS",
                arm=arm,
                cadence_id=selection["cadence_id"],
                policy_version=policy_version,
                prompt_version="PHASE_A_DETERMINISTIC_V1",
                worker=lambda packet, policy=policy_version: (
                    deterministic_worker_response(
                        packet,
                        policy_id=policy,
                    )
                ),
                capture_full_audit=False,
            )
            rows.append(
                {
                    "coordinate_id": coordinate["coordinate_id"],
                    "family_id": coordinate["family_id"],
                    "cost_scenario": coordinate["cost_scenario"],
                    "arm": arm,
                    "cadence_id": selection["cadence_id"],
                    "policy_version": policy_version,
                    "calibration_admissible": selection[
                        "calibration_admissible"
                    ],
                    "metrics": _metric_summary(cell),
                    "cell_sha256": cell["cell_sha256"],
                }
            )
        print(
            json.dumps(
                {
                    "phase": "DETERMINISTIC_OOS_COORDINATE_COMPLETE",
                    "coordinate_id": coordinate["coordinate_id"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    body = {
        "contract": "QR_DOJO_R13_AI_INVENTORY_PHASE_A_OOS_DIAGNOSTIC_V1",
        "schema_version": 1,
        "study_sha256": study["study_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "partition": "OOS",
        "acting_model": "DETERMINISTIC_PHASE_A_NOT_ACTUAL_AI",
        "cell_count": len(rows),
        "cells": rows,
    }
    result = {**body, "result_sha256": canonical_portfolio_sha256(body)}
    _atomic_json(root / "phase-a-oos-diagnostic.json", result)
    return result


def worker_session(
    *,
    root: Path,
    coordinate_id: str,
    arm: str,
    output: Path,
    max_ai_calls: int,
    worker_id: str,
    worker_model: str,
) -> dict[str, Any]:
    if arm not in (B_INVENTORY_ONLY, C_FORECAST_INVENTORY):
        raise ValueError("worker session arm must be B or C")
    study, frames = load_prepared_study(root)
    coordinate = load_prepared_coordinate(root, study, coordinate_id)
    calibration = json.loads(
        (root / "phase-a-calibration.json").read_text(encoding="utf-8")
    )
    calibration_claimed = calibration.pop("calibration_sha256")
    if calibration_claimed != canonical_portfolio_sha256(calibration):
        raise ValueError("calibration selection changed")
    calibration["calibration_sha256"] = calibration_claimed
    selection = next(
        row
        for row in calibration["selections"]
        if row["coordinate_id"] == coordinate_id and row["arm"] == arm
    )
    emitted_packet_hashes: list[str] = []

    def request_worker(packet: dict[str, Any]) -> dict[str, Any]:
        emitted_packet_hashes.append(packet["packet_sha256"])
        packet_path = (
            output.parent
            / "packets"
            / f"{packet['packet_sha256']}.json"
        )
        _atomic_json(packet_path, packet)
        print(
            json.dumps(
                {
                    "kind": "WORKER_PACKET",
                    "session_contract": (
                        "FRESH_CONTEXT_PACKET_ONLY_NO_REPLAY_RESULT_ACCESS_V1"
                    ),
                    "packet_file": str(packet_path.resolve()),
                    "packet": packet,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        line = input()
        payload = json.loads(line)
        if isinstance(payload, dict) and set(payload) == {"response"}:
            payload = payload["response"]
        if not isinstance(payload, dict):
            raise ValueError("worker response must be a JSON object")
        return payload

    cell = simulate_partition(
        study=study,
        coordinate=coordinate,
        frames=frames,
        partition="OOS",
        arm=arm,
        cadence_id=selection["cadence_id"],
        policy_version=selection["policy_version"],
        prompt_version="PHASE_B_FRESH_WORKER_V1",
        worker=request_worker,
        max_ai_calls=max_ai_calls,
        capture_full_audit=True,
    )
    body = {
        "contract": "QR_DOJO_R13_AI_INVENTORY_PHASE_B_WORKER_SESSION_V1",
        "schema_version": 1,
        "study_sha256": study["study_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "coordinate_id": coordinate_id,
        "family_id": coordinate["family_id"],
        "cost_scenario": coordinate["cost_scenario"],
        "arm": arm,
        "cadence_id": selection["cadence_id"],
        "policy_version": selection["policy_version"],
        "worker_boundary": (
            "SEPARATE_FRESH_AGENT_PER_PACKET_OR_EQUIVALENT_REQUIRED"
        ),
        "worker_id": worker_id,
        "worker_model": worker_model,
        "max_actual_ai_calls_preregistered": max_ai_calls,
        "profit_conditioned_retry_allowed": False,
        "emitted_packet_hashes": emitted_packet_hashes,
        "cell": cell,
    }
    result = {**body, "session_sha256": canonical_portfolio_sha256(body)}
    _atomic_json(output, result)
    print(
        json.dumps(
            {
                "kind": "SESSION_COMPLETE",
                "coordinate_id": coordinate_id,
                "arm": arm,
                "actual_ai_calls": cell["metrics"]["ai_call_count"],
                "fallbacks": cell["metrics"]["ai_fallback_count"],
                "session_sha256": result["session_sha256"],
                "economic_result_withheld_from_worker": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def aggregate_phase_b(root: Path) -> dict[str, Any]:
    study, frames = load_prepared_study(root)
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    candidate_cells: list[dict[str, Any]] = []
    for coordinate_ref in study["coordinates"]:
        coordinate = load_prepared_coordinate(
            root,
            study,
            coordinate_ref["coordinate_id"],
        )
        baseline_cell = simulate_partition(
            study=study,
            coordinate=coordinate,
            frames=frames,
            partition="OOS",
            arm=A_BOT_ONLY,
            cadence_id=None,
            policy_version="IMMUTABLE_R13",
            prompt_version="NONE",
            worker=None,
            capture_full_audit=False,
        )
        cells = {A_BOT_ONLY: baseline_cell}
        rows.append(
            {
                "coordinate_id": coordinate["coordinate_id"],
                "family_id": coordinate["family_id"],
                "cost_scenario": coordinate["cost_scenario"],
                "arm": A_BOT_ONLY,
                "metrics": _metric_summary(baseline_cell),
                "cell_sha256": baseline_cell["cell_sha256"],
            }
        )
        for arm in (B_INVENTORY_ONLY, C_FORECAST_INVENTORY):
            path = (
                root
                / "phase-b-sessions"
                / arm
                / f"{coordinate['coordinate_id']}.json"
            )
            session = json.loads(path.read_text(encoding="utf-8"))
            claimed = session.pop("session_sha256")
            if claimed != canonical_portfolio_sha256(session):
                raise ValueError(f"worker session changed: {path}")
            cell = session["cell"]
            cell_claimed = cell["cell_sha256"]
            cell_body = {
                key: value
                for key, value in cell.items()
                if key != "cell_sha256"
            }
            if cell_claimed != canonical_portfolio_sha256(cell_body):
                raise ValueError(f"worker cell changed: {path}")
            cells[arm] = cell
            candidate_cells.append(cell)
            rows.append(
                {
                    "coordinate_id": coordinate["coordinate_id"],
                    "family_id": coordinate["family_id"],
                    "cost_scenario": coordinate["cost_scenario"],
                    "arm": arm,
                    "cadence_id": session["cadence_id"],
                    "policy_version": session["policy_version"],
                    "worker_model": session["worker_model"],
                    "metrics": _metric_summary(cell),
                    "forecast_evaluation_actual_worker": cell[
                        "forecast_evaluation_actual_worker"
                    ],
                    "cell_sha256": cell_claimed,
                    "session_sha256": claimed,
                }
            )
        a = cells[A_BOT_ONLY]["metrics"]
        b = cells[B_INVENTORY_ONLY]["metrics"]
        c = cells[C_FORECAST_INVENTORY]["metrics"]
        b_pass = (
            b["net_after_all_costs_jpy"] > a["net_after_all_costs_jpy"]
            and b["max_drawdown_fraction"]
            <= a["max_drawdown_fraction"] + 1e-12
            and b["margin_call_count"] <= a["margin_call_count"]
            and b["ruin_event_count"] <= a["ruin_event_count"]
        )
        c_vs_a_pass = (
            c["net_after_all_costs_jpy"] > a["net_after_all_costs_jpy"]
            and c["max_drawdown_fraction"]
            <= a["max_drawdown_fraction"] + 1e-12
            and c["margin_call_count"] <= a["margin_call_count"]
            and c["ruin_event_count"] <= a["ruin_event_count"]
        )
        c_vs_b_pass = (
            c["net_after_all_costs_jpy"] > b["net_after_all_costs_jpy"]
            and c["max_drawdown_fraction"]
            <= b["max_drawdown_fraction"] + 1e-12
            and c["margin_call_count"] <= b["margin_call_count"]
            and c["ruin_event_count"] <= b["ruin_event_count"]
        )
        comparisons.append(
            {
                "coordinate_id": coordinate["coordinate_id"],
                "family_id": coordinate["family_id"],
                "cost_scenario": coordinate["cost_scenario"],
                "a_net_jpy": a["net_after_all_costs_jpy"],
                "b_net_jpy": b["net_after_all_costs_jpy"],
                "c_net_jpy": c["net_after_all_costs_jpy"],
                "b_minus_a_net_jpy": (
                    b["net_after_all_costs_jpy"]
                    - a["net_after_all_costs_jpy"]
                ),
                "c_minus_a_net_jpy": (
                    c["net_after_all_costs_jpy"]
                    - a["net_after_all_costs_jpy"]
                ),
                "c_minus_b_net_jpy": (
                    c["net_after_all_costs_jpy"]
                    - b["net_after_all_costs_jpy"]
                ),
                "a_max_dd": a["max_drawdown_fraction"],
                "b_max_dd": b["max_drawdown_fraction"],
                "c_max_dd": c["max_drawdown_fraction"],
                "b_vs_a_gate": b_pass,
                "c_vs_a_gate": c_vs_a_pass,
                "c_vs_b_gate": c_vs_b_pass,
            }
        )
    family_decisions = []
    for family_id in sorted({row["family_id"] for row in comparisons}):
        family_rows = [
            row for row in comparisons if row["family_id"] == family_id
        ]
        scenarios = {row["cost_scenario"] for row in family_rows}
        passed = (
            scenarios == {"BASE", "STRESS"}
            and all(
                row["c_vs_a_gate"] and row["c_vs_b_gate"]
                for row in family_rows
            )
            and all(row["c_minus_a_net_jpy"] > 0 for row in family_rows)
        )
        family_decisions.append(
            {
                "family_id": family_id,
                "status": (
                    "CANDIDATE_TO_JANUARY_OLHC"
                    if passed
                    else "REJECT"
                ),
                "base_stress_direction_consistent": (
                    scenarios == {"BASE", "STRESS"}
                    and len(
                        {
                            row["c_minus_a_net_jpy"] > 0
                            for row in family_rows
                        }
                    )
                    == 1
                ),
                "coordinate_count": len(family_rows),
            }
        )
    body = {
        "contract": "QR_DOJO_R13_AI_INVENTORY_PHASE_B_OOS_RESULT_V1",
        "schema_version": 1,
        "study_sha256": study["study_sha256"],
        "partition": "OOS",
        "source_quote_coverage_proved": False,
        "classification": (
            "EXPERIMENTAL_SAME_INCOMPLETE_SOURCE_PAIRED_DIFFERENCE"
        ),
        "cell_count": len(rows),
        "actual_phase_b_cell_count": len(rows) - 12,
        "comparison_count": len(comparisons),
        "phase_b_operational_summary": _phase_b_operational_summary(
            candidate_cells
        ),
        "forecast_summary_actual_worker": score_forecasts_posthoc(
            forecast_rows=[
                forecast
                for cell in candidate_cells
                if cell["arm"] == C_FORECAST_INVENTORY
                for forecast in cell["forecast_rows"]
                if not forecast["fallback"]
            ],
            frames=frames,
        ),
        "cells": rows,
        "comparisons": comparisons,
        "family_decisions": family_decisions,
        "oracle_d_used_for_acting_or_claims": False,
        "next_month_bot_only_claimed": False,
    }
    result = {**body, "result_sha256": canonical_portfolio_sha256(body)}
    _atomic_json(root / "phase-b-oos-result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "calibrate",
            "deterministic-oos",
            "worker-session",
            "aggregate-phase-b",
        ),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument(
        "--calibration-end-epoch",
        type=int,
        default=1737158400,
    )
    parser.add_argument("--coordinate-id")
    parser.add_argument("--arm")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-ai-calls", type=int, default=3)
    parser.add_argument("--worker-id", default="codex-fresh-worker")
    parser.add_argument("--worker-model", default="gpt-5")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.baseline_root is None or not args.job_id:
            parser.error("prepare requires --baseline-root and --job-id")
        result = prepare_r13_inputs(
            baseline_root=args.baseline_root,
            job_id=args.job_id,
            output_root=args.root,
            calibration_end_epoch=args.calibration_end_epoch,
        )
    elif args.command == "calibrate":
        result = calibrate(args.root)
    elif args.command == "deterministic-oos":
        result = deterministic_oos(args.root)
    elif args.command == "worker-session":
        if not args.coordinate_id or not args.arm or args.output is None:
            parser.error(
                "worker-session requires --coordinate-id, --arm, and --output"
            )
        worker_session(
            root=args.root,
            coordinate_id=args.coordinate_id,
            arm=args.arm,
            output=args.output,
            max_ai_calls=args.max_ai_calls,
            worker_id=args.worker_id,
            worker_model=args.worker_model,
        )
        return
    else:
        result = aggregate_phase_b(args.root)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "contract": result["contract"],
                "cells": result.get(
                    "cell_count",
                    result.get("coordinate_count"),
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

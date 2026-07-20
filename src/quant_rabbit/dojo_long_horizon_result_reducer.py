"""Independent fixed-denominator scorer for the DOJO long-horizon TRAIN.

The execution state machine proves control-plane identity and retains every
``COMPLETE`` or ``FAILED`` coordinate.  This module is the separate economic
reducer: it validates the complete 348-job / 32,112-coordinate handoff before
opening any result, then recomputes monthly multiples, continuous-account
chains, risk/ruin gates and aligned leave-one-out concentration diagnostics.

Historical results are worn TRAIN evidence.  Even a perfect arithmetic result
cannot promote a strategy, mutate configuration, authorize an order or grant
live permission.  This module is pure and has no filesystem, broker, network,
model or tuning side effect.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Final

from quant_rabbit.dojo_long_horizon_execution import (
    DojoLongHorizonExecutionError,
    validate_long_horizon_terminal_reducer_bundle,
)
from quant_rabbit.dojo_long_horizon_plan import (
    M5_BINDING_ID,
    STARTING_EQUITY_JPY,
    canonical_sha256,
)


CONTRACT: Final = "QR_DOJO_LONG_HORIZON_RESULT_SCORECARD_V1"
ECONOMIC_RUNNER_OUTPUT_CONTRACT: Final = (
    "QR_DOJO_LONG_HORIZON_ECONOMIC_RUNNER_OUTPUT_REQUIREMENTS_V1"
)
SCHEMA_VERSION: Final = 1
EVIDENCE_TIER: Final = "WORN_HISTORICAL_TRAIN_ONLY"
MAX_REPORTED_ISSUES: Final = 100
_INDEPENDENT_ECONOMIC_EVIDENCE_BLOCKER: Final = (
    "COMPACT_ECONOMIC_EVIDENCE_NOT_INDEPENDENTLY_REEXECUTED"
)


class DojoLongHorizonResultReducerError(ValueError):
    """The terminal denominator or scorer inputs are malformed."""


def _authority() -> dict[str, Any]:
    return {
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "diagnostic_only": True,
        "forward_proof_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
        "trainer_may_change_live_configuration": False,
    }


def long_horizon_economic_runner_output_requirements() -> dict[str, Any]:
    """Describe the minimum replay inputs an independent scorer must receive.

    Hashes of runner-authored aggregate numbers are not computation proofs.
    A future accepted runner must retain enough causal input to let this
    separately pinned scorer invoke the trusted portfolio reducer again and
    compare every terminal claim.  The current terminal contract supplies
    digests only, so these requirements remain unmet and the official gate is
    deliberately closed.
    """

    body = {
        "contract": ECONOMIC_RUNNER_OUTPUT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": EVIDENCE_TIER,
        "artifact_cardinality": {
            "one_shared_source_stream_packet_per_job": True,
            "one_causal_reducer_input_transcript_per_coordinate": True,
            "one_terminal_or_typed_failure_per_coordinate": True,
            "failed_coordinates_remain_in_fixed_denominator": True,
        },
        "job_source_packet": {
            "required": True,
            "fields": [
                "job_sha256",
                "source_binding_id",
                "source_slice_receipt",
                "source_slice_bytes_sha256",
                "ordered_quote_batch_payloads",
                "ordered_quote_batch_chain_sha256",
                "first_source_cursor",
                "last_source_cursor",
                "expected_quote_pair_set_sha256",
                "observed_quote_pair_set_sha256",
                "missing_quote_count",
                "synthetic_quote_count",
            ],
            "same_ordered_quote_batches_fanned_to_all_coordinates": True,
            "source_receipt_hash_without_bound_source_bytes_is_sufficient": False,
        },
        "coordinate_reducer_input": {
            "required": True,
            "fields": [
                "coordinate_id",
                "job_sha256",
                "portfolio_policy",
                "portfolio_policy_sha256",
                "worker_binding_set_sha256",
                "post_exit_snapshots",
                "all_worker_proposal_batches",
                "risk_reducing_intents",
                "new_risk_intents",
                "reducer_frame_order_sha256",
                "terminal_settlement_input",
            ],
            "one_decision_context_per_timestamp": True,
            "worker_reported_fill_pnl_cost_or_margin_is_trusted": False,
            "event_only_fill_exit_ledger_is_sufficient": False,
        },
        "independent_reexecution": {
            "required": True,
            "trusted_reducer": (
                "quant_rabbit.dojo_portfolio_replay_reducer.reduce_portfolio_replay"
            ),
            "trusted_reducer_code_sha256_must_equal_plan": True,
            "runner_aggregate_result_is_an_input_to_reducer": False,
            "terminal_fields_recomputed_and_compared": [
                "starting_balance_jpy",
                "starting_equity_jpy",
                "ending_balance_jpy",
                "ending_equity_jpy",
                "minimum_mtm_equity_jpy",
                "minimum_free_margin_jpy",
                "max_mtm_drawdown_fraction",
                "peak_margin_usage_fraction",
                "margin_closeout_count",
                "ruin_event_count",
                "trade_count",
                "fill_count",
                "margin_reject_count",
                "financing_jpy",
                "transaction_cost_jpy",
            ],
            "recompute_fill_prices_from_quotes": True,
            "recompute_exit_prices_from_quotes": True,
            "recompute_fx_conversion_from_synchronized_quotes": True,
            "recompute_spread_slippage_fees_and_financing": True,
            "recompute_every_timestamp_mtm_and_margin": True,
        },
        "current_trusted_reducer_gap": {
            "current_output_contract": (
                "QR_DOJO_SHARED_ACCOUNT_PORTFOLIO_REPLAY_V1"
            ),
            "current_output_is_sufficient_for_long_horizon_cell": False,
            "missing_or_ambiguous_independent_metrics": [
                "minimum_mtm_equity_jpy",
                "minimum_free_margin_jpy",
                "peak_margin_usage_fraction_at_one_timestamp",
                "ruin_event_count",
                "trade_count_semantics",
                "margin_reject_count_semantics",
                "transaction_cost_jpy_including_every_fee_component",
                "quote_coverage_complete",
                "active_worker_ack_complete",
            ],
            "peak_margin_jpy_divided_by_unaligned_peak_equity_is_allowed": False,
            "position_closes_may_silently_substitute_for_trade_count": False,
        },
        "continuous_carry": {
            "full_carry_state_bytes_required": True,
            "carry_state_sha256_required": True,
            "next_month_predecessor_must_equal_prior_carry_bytes": True,
            "equity_only_or_state_hash_only_handoff_is_sufficient": False,
        },
        "failure_evidence": {
            "typed_failure_document_required": True,
            "failure_document_bytes_must_match_evidence_sha256": True,
            "source_or_reducer_failure_must_be_independently_reproducible": True,
            "partial_or_zero_economics_allowed": False,
            "failure_may_reduce_denominator": False,
        },
        "lopo": {
            "separate_causal_reexecution_per_fold_required": True,
            "post_hoc_subtraction_from_full_portfolio_allowed": False,
            "comparison_alignment": [
                "month",
                "evaluation_mode",
                "intrabar_path",
                "cost_scenario",
            ],
        },
        "current_event_only_compact_ledger_limitations": [
            "NO_ORDERED_SOURCE_QUOTE_PAYLOADS",
            "NO_ALL_WORKER_PROPOSAL_TRANSCRIPT",
            "FILL_PRICE_NOT_RECOMPUTABLE_FROM_SOURCE_QUOTES",
            "CONVERSION_RATE_NOT_RECOMPUTABLE_FROM_SOURCE_QUOTES",
            "FINANCING_NOT_RECOMPUTABLE_FROM_SEALED_POLICY_AND_TIME",
            "MTM_AND_MARGIN_PATH_ARE_SELF_REPORTED",
            "FULL_CONTINUOUS_CARRY_BYTES_NOT_BOUND_TO_TERMINAL_CELL",
            "FAILURE_EVIDENCE_BYTES_NOT_OPENED",
        ],
        "authority": _authority(),
    }
    return {**body, "requirements_sha256": canonical_sha256(body)}


def _official_gate(
    cell_reported_blockers: Sequence[str],
) -> tuple[bool, list[str]]:
    """Keep the official gate closed until causal inputs are re-executed."""

    return False, [
        _INDEPENDENT_ECONOMIC_EVIDENCE_BLOCKER,
        *cell_reported_blockers,
    ]


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DojoLongHorizonResultReducerError(f"{field} must be a sequence")
    return value


def _isclose(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    left_number = float(left)
    right_number = float(right)
    return (
        math.isfinite(left_number)
        and math.isfinite(right_number)
        and math.isclose(
            left_number,
            right_number,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    )


def _multiple(cell: Mapping[str, Any]) -> float | None:
    if cell["status"] != "COMPLETE":
        return None
    start = float(cell["starting_equity_jpy"])
    end = float(cell["ending_equity_jpy"])
    if start <= 0.0:
        return None
    result = end / start
    return result if math.isfinite(result) else None


def _profit_drop_fraction(full_multiple: float, lopo_multiple: float) -> float:
    """Return path/mode/month-aligned relative profit loss.

    When the full portfolio itself has no positive monthly profit, a weaker
    fold is maximally concentrated and an equal/better fold has zero measured
    drop.  This prevents a negative denominator from making concentration look
    artificially good.
    """

    full_profit = full_multiple - 1.0
    if full_profit <= 0.0:
        return 1.0 if lopo_multiple < full_multiple else 0.0
    return max(0.0, (full_multiple - lopo_multiple) / full_profit)


def _flatten(
    schedule: Mapping[str, Any],
    terminals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job, terminal in zip(schedule["jobs"], terminals, strict=True):
        for coordinate, cell in zip(job["coordinates"], terminal["cells"], strict=True):
            if coordinate["coordinate_id"] != cell["coordinate_id"]:
                raise DojoLongHorizonResultReducerError(
                    "validated terminal coordinate order drifted from schedule"
                )
            records.append(
                {
                    "job": job,
                    "coordinate": coordinate,
                    "cell": cell,
                }
            )
    return records


def _denominator_summary(
    *,
    plan: Mapping[str, Any],
    schedule: Mapping[str, Any],
    terminals: Sequence[Mapping[str, Any]],
    handoffs: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    complete = sum(row["cell"]["status"] == "COMPLETE" for row in records)
    failed_ids = [
        row["cell"]["coordinate_id"]
        for row in records
        if row["cell"]["status"] == "FAILED"
    ]
    expected = plan["exact_denominator"]["total_required_result_cell_count"]
    exact_coverage = (
        len(terminals) == schedule["stream_job_count"]
        and len(handoffs) == schedule["stream_job_count"]
        and len(records) == expected == schedule["result_coordinate_count"]
    )
    return {
        "expected_job_count": schedule["stream_job_count"],
        "observed_job_count": len(terminals),
        "expected_coordinate_count": expected,
        "observed_coordinate_count": len(records),
        "effective_weighted_coordinate_count": schedule[
            "effective_weighted_result_coordinate_count"
        ],
        "complete_coordinate_count": complete,
        "failed_coordinate_count": len(failed_ids),
        "failed_coordinate_ids_sha256": canonical_sha256(failed_ids),
        "failed_coordinate_id_sample": failed_ids[:MAX_REPORTED_ISSUES],
        "failed_coordinates_retained_in_denominator": True,
        "failed_coordinates_zero_filled": False,
        "exact_coverage": exact_coverage,
        "all_coordinates_complete": exact_coverage and not failed_ids,
    }


def _continuous_chain_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    chains: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    independent_reset_issues: list[str] = []
    independent_failed = 0
    for record in records:
        coordinate = record["coordinate"]
        cell = record["cell"]
        if coordinate["evaluation_mode"] == "CONTINUOUS_ACCOUNT":
            chain_id = coordinate["continuous_account_chain_id"]
            if not isinstance(chain_id, str):
                raise DojoLongHorizonResultReducerError(
                    "continuous coordinate is missing its sealed chain identity"
                )
            chains[chain_id].append(record)
            continue
        if cell["status"] != "COMPLETE":
            independent_failed += 1
            continue
        if not _isclose(
            cell["starting_equity_jpy"], STARTING_EQUITY_JPY
        ) or not _isclose(cell["starting_balance_jpy"], STARTING_EQUITY_JPY):
            independent_reset_issues.append(cell["coordinate_id"])

    chain_issues: list[dict[str, Any]] = []
    failed_chain_cells = 0
    for chain_id in sorted(chains):
        rows = sorted(
            chains[chain_id],
            key=lambda row: row["coordinate"]["continuous_chain_ordinal"],
        )
        issue_codes: list[str] = []
        ordinals = [row["coordinate"]["continuous_chain_ordinal"] for row in rows]
        if ordinals != list(range(len(rows))):
            issue_codes.append("NON_CONTIGUOUS_ORDINALS")
        if not rows[0]["coordinate"]["continuous_chain_first"]:
            issue_codes.append("FIRST_MARKER_MISSING")
        if not rows[-1]["coordinate"]["continuous_chain_last"]:
            issue_codes.append("LAST_MARKER_MISSING")

        first = rows[0]["cell"]
        if first["status"] != "COMPLETE":
            failed_chain_cells += 1
            issue_codes.append("FIRST_CELL_FAILED")
        elif not _isclose(
            first["starting_equity_jpy"], STARTING_EQUITY_JPY
        ) or not _isclose(first["starting_balance_jpy"], STARTING_EQUITY_JPY):
            issue_codes.append("INITIAL_EQUITY_NOT_RESET")

        for prior, current in zip(rows, rows[1:]):
            prior_coordinate = prior["coordinate"]
            current_coordinate = current["coordinate"]
            prior_cell = prior["cell"]
            current_cell = current["cell"]
            if (
                prior_cell["status"] != "COMPLETE"
                or current_cell["status"] != "COMPLETE"
            ):
                failed_chain_cells += int(current_cell["status"] != "COMPLETE")
                issue_codes.append("CHAIN_CELL_FAILED")
                continue
            if (
                current_coordinate["predecessor_state_slot_id"]
                != prior_coordinate["carry_out_state_slot_id"]
            ):
                issue_codes.append("STATE_SLOT_CHAIN_MISMATCH")
            if (
                current_cell["predecessor_state_sha256"]
                != prior_cell["carry_out_state_sha256"]
            ):
                issue_codes.append("STATE_SHA_CHAIN_MISMATCH")
            if not _isclose(
                current_cell["starting_equity_jpy"], prior_cell["ending_equity_jpy"]
            ):
                issue_codes.append("EQUITY_CHAIN_MISMATCH")
            if not _isclose(
                current_cell["starting_balance_jpy"], prior_cell["ending_balance_jpy"]
            ):
                issue_codes.append("BALANCE_CHAIN_MISMATCH")
        if issue_codes:
            chain_issues.append(
                {
                    "continuous_account_chain_id": chain_id,
                    "issue_codes": sorted(set(issue_codes)),
                }
            )

    return {
        "continuous_chain_count": len(chains),
        "continuous_chain_issue_count": len(chain_issues),
        "continuous_failed_cell_count": failed_chain_cells,
        "continuous_chain_issues_sha256": canonical_sha256(chain_issues),
        "continuous_chain_issue_sample": chain_issues[:MAX_REPORTED_ISSUES],
        "continuous_chain_gate_pass": not chain_issues and failed_chain_cells == 0,
        "independent_coordinate_count": sum(
            row["coordinate"]["evaluation_mode"] == "INDEPENDENT_MONTH"
            for row in records
        ),
        "independent_failed_cell_count": independent_failed,
        "independent_reset_issue_count": len(independent_reset_issues),
        "independent_reset_issue_ids_sha256": canonical_sha256(
            independent_reset_issues
        ),
        "independent_reset_issue_sample": independent_reset_issues[
            :MAX_REPORTED_ISSUES
        ],
        "independent_reset_gate_pass": (
            independent_failed == 0 and not independent_reset_issues
        ),
    }


def _risk_summary(
    records: Sequence[Mapping[str, Any]], *, gates: Mapping[str, Any]
) -> dict[str, Any]:
    scenario_drawdowns: dict[str, list[float]] = {"BASE": [], "STRESS": []}
    peak_margins: list[float] = []
    minimum_equities: list[float] = []
    minimum_free_margins: list[float] = []
    closeouts = 0
    ruin_events = 0
    failed = 0
    violation_ids: list[str] = []
    for record in records:
        coordinate = record["coordinate"]
        cell = record["cell"]
        if cell["status"] != "COMPLETE":
            failed += 1
            continue
        scenario = coordinate["cost_scenario"]
        drawdown = float(cell["max_mtm_drawdown_fraction"])
        margin = float(cell["peak_margin_usage_fraction"])
        minimum_equity = float(cell["minimum_mtm_equity_jpy"])
        minimum_free_margin = float(cell["minimum_free_margin_jpy"])
        scenario_drawdowns[scenario].append(drawdown)
        peak_margins.append(margin)
        minimum_equities.append(minimum_equity)
        minimum_free_margins.append(minimum_free_margin)
        closeouts += int(cell["margin_closeout_count"])
        ruin_events += int(cell["ruin_event_count"])
        dd_limit = (
            gates["base_max_drawdown_fraction_max"]
            if scenario == "BASE"
            else gates["stress_max_drawdown_fraction_max"]
        )
        if (
            drawdown > dd_limit
            or margin > gates["peak_margin_fraction_max"]
            or int(cell["margin_closeout_count"]) > 0
            or int(cell["ruin_event_count"]) > 0
            or minimum_equity <= 0.0
            or minimum_free_margin <= 0.0
            or float(cell["ending_equity_jpy"]) <= 0.0
        ):
            violation_ids.append(cell["coordinate_id"])

    base_max = max(scenario_drawdowns["BASE"], default=None)
    stress_max = max(scenario_drawdowns["STRESS"], default=None)
    peak_margin = max(peak_margins, default=None)
    minimum_equity = min(minimum_equities, default=None)
    minimum_free_margin = min(minimum_free_margins, default=None)
    gate_pass = (
        failed == 0
        and base_max is not None
        and base_max <= gates["base_max_drawdown_fraction_max"]
        and stress_max is not None
        and stress_max <= gates["stress_max_drawdown_fraction_max"]
        and peak_margin is not None
        and peak_margin <= gates["peak_margin_fraction_max"]
        and closeouts <= gates["margin_closeout_count_max"]
        and ruin_events == 0
        and minimum_equity is not None
        and minimum_equity > 0.0
        and minimum_free_margin is not None
        and minimum_free_margin > 0.0
        and not violation_ids
    )
    return {
        "scope": "ALL_M5_M1_MAIN_AND_LOPO_COORDINATES",
        "failed_coordinate_count": failed,
        "observed_base_max_drawdown_fraction": base_max,
        "observed_stress_max_drawdown_fraction": stress_max,
        "observed_peak_margin_usage_fraction": peak_margin,
        "observed_minimum_mtm_equity_jpy": minimum_equity,
        "observed_minimum_free_margin_jpy": minimum_free_margin,
        "observed_margin_closeout_count": closeouts,
        "observed_ruin_event_count": ruin_events,
        "risk_or_ruin_violation_count": len(violation_ids),
        "risk_or_ruin_violation_ids_sha256": canonical_sha256(violation_ids),
        "risk_or_ruin_violation_id_sample": violation_ids[:MAX_REPORTED_ISSUES],
        "gate_pass": gate_pass,
    }


def _monthly_main_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    months: Sequence[str],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    main: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for record in records:
        coordinate = record["coordinate"]
        job = record["job"]
        if (
            job["source_binding_id"] != M5_BINDING_ID
            or coordinate["stage"] != "PORTFOLIO_MAIN"
        ):
            continue
        key = (
            job["month"],
            coordinate["evaluation_mode"],
            job["intrabar_path"],
            coordinate["cost_scenario"],
        )
        if key in main:
            raise DojoLongHorizonResultReducerError(
                "M5 PORTFOLIO_MAIN coordinate is duplicated"
            )
        main[key] = record

    modes: dict[str, Any] = {}
    all_month_rows: list[dict[str, Any]] = []
    for mode in ("INDEPENDENT_MONTH", "CONTINUOUS_ACCOUNT"):
        mode_rows: list[dict[str, Any]] = []
        for month in months:
            cells: dict[str, dict[str, Any]] = {}
            complete = True
            for scenario in ("BASE", "STRESS"):
                cells[scenario] = {}
                for path in ("OHLC", "OLHC"):
                    record = main.get((month, mode, path, scenario))
                    if record is None:
                        raise DojoLongHorizonResultReducerError(
                            "M5 PORTFOLIO_MAIN fixed denominator is incomplete"
                        )
                    cell = record["cell"]
                    multiple = _multiple(cell)
                    if multiple is None:
                        complete = False
                    cells[scenario][path] = {
                        "coordinate_id": cell["coordinate_id"],
                        "status": cell["status"],
                        "monthly_ending_multiple": multiple,
                    }
            stress_multiples = [
                cells["STRESS"][path]["monthly_ending_multiple"]
                for path in ("OHLC", "OLHC")
            ]
            pessimistic = (
                min(float(value) for value in stress_multiples)
                if all(value is not None for value in stress_multiples)
                else None
            )
            reached = (
                pessimistic is not None and pessimistic >= gates["target_multiple"]
            )
            losing = pessimistic is not None and pessimistic < 1.0
            month_row = {
                "month": month,
                "evaluation_mode": mode,
                "cells": cells,
                "all_four_main_cells_complete": complete,
                "pessimistic_stress_multiple": pessimistic,
                "reached_3x": reached,
                "losing_month": losing,
            }
            mode_rows.append(month_row)
            all_month_rows.append(month_row)
        hit_count = sum(row["reached_3x"] for row in mode_rows)
        losing_months = [row["month"] for row in mode_rows if row["losing_month"]]
        unknown_months = [
            row["month"]
            for row in mode_rows
            if row["pessimistic_stress_multiple"] is None
        ]
        multiples = [
            float(row["pessimistic_stress_multiple"])
            for row in mode_rows
            if row["pessimistic_stress_multiple"] is not None
        ]
        modes[mode] = {
            "month_count": len(mode_rows),
            "months": mode_rows,
            "three_x_hit_count": hit_count,
            "three_x_hit_rate": hit_count / len(mode_rows),
            "losing_month_count": len(losing_months),
            "losing_months": losing_months,
            "unknown_month_count": len(unknown_months),
            "unknown_months": unknown_months,
            "worst_pessimistic_stress_multiple": min(multiples, default=None),
            "every_month_3x": hit_count == len(mode_rows) and not unknown_months,
            "zero_losing_months": not losing_months and not unknown_months,
        }
    return {
        "month_denominator": list(months),
        "required_month_count_per_mode": len(months),
        "modes": modes,
        "all_mode_month_rows_sha256": canonical_sha256(all_month_rows),
        "independent_and_continuous_every_month_3x": all(
            row["every_month_3x"] for row in modes.values()
        ),
        "independent_and_continuous_zero_losing_months": all(
            row["zero_losing_months"] for row in modes.values()
        ),
    }


def _lopo_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    main: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for record in records:
        coordinate = record["coordinate"]
        job = record["job"]
        if (
            job["source_binding_id"] == M5_BINDING_ID
            and coordinate["stage"] == "PORTFOLIO_MAIN"
        ):
            main[
                (
                    job["month"],
                    coordinate["evaluation_mode"],
                    job["intrabar_path"],
                    coordinate["cost_scenario"],
                )
            ] = record

    stage_limits = {
        "PAIR_LOPO": gates["pair_lopo_profit_drop_fraction_max"],
        "FAMILY_LOPO": gates["family_lopo_profit_drop_fraction_max"],
        "CURRENCY_LOPO": gates["currency_lopo_profit_drop_fraction_max"],
    }
    stage_rows: dict[str, list[Mapping[str, Any]]] = {
        stage: [] for stage in stage_limits
    }
    for record in records:
        coordinate = record["coordinate"]
        if (
            record["job"]["source_binding_id"] == M5_BINDING_ID
            and coordinate["stage"] in stage_rows
        ):
            stage_rows[coordinate["stage"]].append(record)

    result: dict[str, Any] = {}
    all_pass = True
    for stage, rows in stage_rows.items():
        expected_count = (
            int(expected_counts[stage]) if expected_counts is not None else len(rows)
        )
        if len(rows) != expected_count:
            raise DojoLongHorizonResultReducerError(
                f"{stage} comparison denominator drifted from the sealed plan"
            )
        limit = float(stage_limits[stage])
        comparison_digest_rows: list[dict[str, Any]] = []
        by_label: dict[str, dict[str, Any]] = {}
        unknown = 0
        over_limit = 0
        worst: dict[str, Any] | None = None
        for record in rows:
            job = record["job"]
            coordinate = record["coordinate"]
            cell = record["cell"]
            key = (
                job["month"],
                coordinate["evaluation_mode"],
                job["intrabar_path"],
                coordinate["cost_scenario"],
            )
            full = main.get(key)
            if full is None:
                raise DojoLongHorizonResultReducerError(
                    "LOPO coordinate has no aligned M5 PORTFOLIO_MAIN result"
                )
            full_multiple = _multiple(full["cell"])
            lopo_multiple = _multiple(cell)
            drop = (
                _profit_drop_fraction(full_multiple, lopo_multiple)
                if full_multiple is not None and lopo_multiple is not None
                else None
            )
            label = coordinate["fold_label"]
            if not isinstance(label, str):
                raise DojoLongHorizonResultReducerError(
                    "LOPO coordinate is missing its sealed fold label"
                )
            label_row = by_label.setdefault(
                label,
                {
                    "comparison_count": 0,
                    "complete_comparison_count": 0,
                    "unknown_comparison_count": 0,
                    "over_limit_count": 0,
                    "maximum_profit_drop_fraction": None,
                },
            )
            label_row["comparison_count"] += 1
            if drop is None:
                unknown += 1
                label_row["unknown_comparison_count"] += 1
            else:
                label_row["complete_comparison_count"] += 1
                current_max = label_row["maximum_profit_drop_fraction"]
                if current_max is None or drop > current_max:
                    label_row["maximum_profit_drop_fraction"] = drop
                if drop > limit:
                    over_limit += 1
                    label_row["over_limit_count"] += 1
                if worst is None or drop > worst["profit_drop_fraction"]:
                    worst = {
                        "coordinate_id": cell["coordinate_id"],
                        "full_coordinate_id": full["cell"]["coordinate_id"],
                        "month": job["month"],
                        "evaluation_mode": coordinate["evaluation_mode"],
                        "intrabar_path": job["intrabar_path"],
                        "cost_scenario": coordinate["cost_scenario"],
                        "fold_label": label,
                        "profit_drop_fraction": drop,
                    }
            comparison_digest_rows.append(
                {
                    "coordinate_id": cell["coordinate_id"],
                    "full_coordinate_id": full["cell"]["coordinate_id"],
                    "profit_drop_fraction": drop,
                }
            )
        gate_pass = unknown == 0 and over_limit == 0 and bool(rows)
        all_pass = all_pass and gate_pass
        result[stage] = {
            "expected_comparison_count": expected_count,
            "comparison_count": len(rows),
            "complete_comparison_count": len(rows) - unknown,
            "unknown_comparison_count": unknown,
            "over_limit_count": over_limit,
            "profit_drop_fraction_max": limit,
            "maximum_profit_drop_fraction": (
                worst["profit_drop_fraction"] if worst is not None else None
            ),
            "worst_aligned_comparison": worst,
            "by_label": dict(sorted(by_label.items())),
            "comparison_rows_sha256": canonical_sha256(comparison_digest_rows),
            "gate_pass": gate_pass,
        }
    return {"stages": result, "all_lopo_gates_pass": all_pass}


def _m1_context_summary(
    records: Sequence[Mapping[str, Any]], *, expected_count: int | None = None
) -> dict[str, Any]:
    contexts: dict[str, dict[str, Any]] = {}
    for record in records:
        job = record["job"]
        if job["granularity"] != "M1":
            continue
        binding = job["source_binding_id"]
        row = contexts.setdefault(
            binding,
            {
                "coordinate_count": 0,
                "complete_coordinate_count": 0,
                "failed_coordinate_count": 0,
                "aggregation_weight_sum": 0.0,
            },
        )
        row["coordinate_count"] += 1
        row["aggregation_weight_sum"] += float(
            record["coordinate"]["aggregation_weight"]
        )
        if record["cell"]["status"] == "COMPLETE":
            row["complete_coordinate_count"] += 1
        else:
            row["failed_coordinate_count"] += 1
    all_complete = bool(contexts) and all(
        row["complete_coordinate_count"] == row["coordinate_count"]
        and row["failed_coordinate_count"] == 0
        for row in contexts.values()
    )
    observed_count = sum(row["coordinate_count"] for row in contexts.values())
    sealed_expected = observed_count if expected_count is None else expected_count
    if observed_count != sealed_expected:
        raise DojoLongHorizonResultReducerError(
            "M1 precision denominator drifted from the sealed plan"
        )
    return {
        "context_policy": "CORE5_AND_FULL28_ARE_DISTINCT_FULL_WEIGHT_CONTEXTS",
        "expected_coordinate_count": sealed_expected,
        "observed_coordinate_count": observed_count,
        "contexts": dict(sorted(contexts.items())),
        "all_contexts_complete": all_complete,
        "m1_economics_substitute_for_m5_monthly_gate": False,
    }


def score_long_horizon_results(
    *,
    plan: Mapping[str, Any],
    schedule: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
    terminal_manifests: Sequence[Mapping[str, Any]],
    reducer_handoffs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score the exact sealed long-horizon denominator, fail closed.

    Both control-plane artifact sets are mandatory.  Missing jobs, missing
    coordinates, reordered terminals, partial economics, self-rehashed drift
    or a mismatched reducer handoff fail before arithmetic starts.
    """

    terminals_input = _sequence(terminal_manifests, field="terminal_manifests")
    handoffs_input = _sequence(reducer_handoffs, field="reducer_handoffs")
    try:
        bundle = validate_long_horizon_terminal_reducer_bundle(
            terminals=terminals_input,
            handoffs=handoffs_input,
            schedule=schedule,
            plan=plan,
            execution_manifest=execution_manifest,
        )
        terminals = bundle["terminal_manifests"]
        handoffs = bundle["reducer_handoffs"]
    except DojoLongHorizonExecutionError as exc:
        raise DojoLongHorizonResultReducerError(
            "long-horizon terminal denominator is not independently valid"
        ) from exc

    # The combined execution validator above rebuilds and byte-compares the
    # plan, schedule and execution manifest before returning.  Rebuilding the
    # 32,112-coordinate schedule a second time here would add no trust.
    sealed_plan = plan
    sealed_schedule = schedule
    records = _flatten(sealed_schedule, terminals)
    denominator = _denominator_summary(
        plan=sealed_plan,
        schedule=sealed_schedule,
        terminals=terminals,
        handoffs=handoffs,
        records=records,
    )
    gates = sealed_plan["monthly_3x_diagnostic_gates"]
    chain = _continuous_chain_summary(records)
    risk = _risk_summary(records, gates=gates)
    monthly = _monthly_main_summary(
        records,
        months=sealed_plan["exact_denominator"]["months"],
        gates=gates,
    )
    expected_lopo_counts = {
        row["stage"]: row["result_cell_count"]
        for row in sealed_plan["exact_denominator"]["portfolio_stages"]
        if row["stage"] != "PORTFOLIO_MAIN"
    }
    lopo = _lopo_summary(
        records,
        gates=gates,
        expected_counts=expected_lopo_counts,
    )
    m1 = _m1_context_summary(
        records,
        expected_count=sealed_plan["exact_denominator"][
            "m1_precision_result_cell_count"
        ],
    )

    cell_reported_blockers: list[str] = []
    if not denominator["all_coordinates_complete"]:
        cell_reported_blockers.append("RESULT_DENOMINATOR_CONTAINS_FAILED_CELLS")
    if not monthly["independent_and_continuous_every_month_3x"]:
        cell_reported_blockers.append("NOT_EVERY_MONTH_3X_IN_BOTH_ACCOUNT_MODES")
    if not monthly["independent_and_continuous_zero_losing_months"]:
        cell_reported_blockers.append("LOSING_OR_UNKNOWN_MONTH_PRESENT")
    if not risk["gate_pass"]:
        cell_reported_blockers.append("RISK_MARGIN_OR_RUIN_GATE_FAILED")
    if not lopo["all_lopo_gates_pass"]:
        cell_reported_blockers.append(
            "PAIR_FAMILY_OR_CURRENCY_LOPO_GATE_FAILED"
        )
    if not chain["continuous_chain_gate_pass"]:
        cell_reported_blockers.append("CONTINUOUS_ACCOUNT_CHAIN_FAILED")
    if not chain["independent_reset_gate_pass"]:
        cell_reported_blockers.append("INDEPENDENT_MONTH_RESET_FAILED")
    if not m1["all_contexts_complete"]:
        cell_reported_blockers.append("M1_PRECISION_CONTEXT_INCOMPLETE")
    arithmetic_gate_pass, blockers = _official_gate(cell_reported_blockers)

    body = {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": EVIDENCE_TIER,
        "plan_sha256": sealed_plan["plan_sha256"],
        "schedule_sha256": sealed_schedule["schedule_sha256"],
        "execution_manifest_sha256": execution_manifest["execution_manifest_sha256"],
        "terminal_sha256_values_sha256": canonical_sha256(
            [row["terminal_sha256"] for row in terminals]
        ),
        "reducer_handoff_sha256_values_sha256": canonical_sha256(
            [row["reducer_handoff_sha256"] for row in handoffs]
        ),
        "evidence_boundary": {
            "terminal_control_plane_independently_validated": True,
            "terminal_cell_values_independently_aggregated": True,
            "terminal_economic_values_independently_recomputed": False,
            "compact_trade_evidence_reexecuted_by_this_scorer": False,
            "source_quote_bytes_opened_by_this_scorer": False,
            "worker_proposal_transcripts_opened_by_this_scorer": False,
            "continuous_carry_state_bytes_revalidated_by_this_scorer": False,
            "typed_failure_evidence_bytes_opened_by_this_scorer": False,
            "historical_worn_train_only": True,
            "prospective_proof": False,
        },
        "required_economic_runner_output": (
            long_horizon_economic_runner_output_requirements()
        ),
        "denominator": denominator,
        "monthly_m5_portfolio_main": monthly,
        "continuous_and_reset_integrity": chain,
        "risk_margin_and_ruin": risk,
        "lopo_concentration": lopo,
        "m1_precision_context_diagnostic": m1,
        "cell_reported_diagnostic_gate_pass": not cell_reported_blockers,
        "cell_reported_diagnostic_blockers": cell_reported_blockers,
        "arithmetic_gate_pass": arithmetic_gate_pass,
        "arithmetic_blockers": blockers,
        "diagnostic_target_reached": False,
        "market_return_guarantee": False,
        "promotion_eligible": False,
        "promotion_blockers": [
            "WORN_HISTORICAL_TRAIN_HAS_NO_PROMOTION_AUTHORITY",
            "FORWARD_PAPER_AND_SEPARATE_PROMOTION_CONTRACT_REQUIRED",
            *blockers,
        ],
        "authority": _authority(),
    }
    return {**body, "scorecard_sha256": canonical_sha256(body)}


__all__ = [
    "CONTRACT",
    "ECONOMIC_RUNNER_OUTPUT_CONTRACT",
    "DojoLongHorizonResultReducerError",
    "long_horizon_economic_runner_output_requirements",
    "score_long_horizon_results",
]

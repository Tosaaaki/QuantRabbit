from __future__ import annotations

from typing import Any

import pytest

from quant_rabbit.dojo_long_horizon_execution import (
    CELL_CONTRACT,
    build_long_horizon_execution_manifest,
)
from quant_rabbit.dojo_long_horizon_plan import (
    IMPLEMENTATION_DIGEST_KEYS,
    SOURCE_BINDING_IDS,
    build_long_horizon_train_plan,
    canonical_sha256,
)
from quant_rabbit.dojo_long_horizon_result_reducer import (
    DojoLongHorizonResultReducerError,
    _authority,
    _continuous_chain_summary,
    _lopo_summary,
    _m1_context_summary,
    _monthly_main_summary,
    _official_gate,
    _profit_drop_fraction,
    _risk_summary,
    long_horizon_economic_runner_output_requirements,
    score_long_horizon_results,
)
from quant_rabbit.dojo_long_horizon_schedule import (
    build_long_horizon_stream_schedule,
)


FAMILIES = ("breakout", "range_fade", "spike_fade")
WORKERS = (
    {"worker_id": "breakout-v1", "family_id": "breakout", "config_sha256": "a" * 64},
    {"worker_id": "range-v1", "family_id": "range_fade", "config_sha256": "b" * 64},
    {"worker_id": "spike-v1", "family_id": "spike_fade", "config_sha256": "c" * 64},
)


def _digests(keys: tuple[str, ...], offset: int) -> dict[str, str]:
    return {key: f"{index + offset:064x}" for index, key in enumerate(keys, 1)}


@pytest.fixture(scope="module")
def sealed_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = build_long_horizon_train_plan(
        portfolio_families=FAMILIES,
        source_digests=_digests(SOURCE_BINDING_IDS, 0),
        corpus_digests=_digests(SOURCE_BINDING_IDS, 10),
        implementation_digests=_digests(IMPLEMENTATION_DIGEST_KEYS, 20),
    )
    schedule = build_long_horizon_stream_schedule(plan, worker_bindings=WORKERS)
    manifest = build_long_horizon_execution_manifest(
        schedule,
        plan=plan,
        runner_binding={
            "runner_contract": "TEST_RUNNER_V1",
            "runner_code_sha256": "d" * 64,
            "result_contract": CELL_CONTRACT,
        },
        resource_policy={
            "max_resident_coordinates": 160,
            "max_rss_bytes": 2_147_483_648,
            "max_open_files": 256,
            "min_free_disk_bytes": 5_368_709_120,
            "max_checkpoint_bytes": 8_388_608,
            "max_terminal_bytes": 2_097_152,
            "max_parallel_jobs": 4,
        },
    )
    return plan, schedule, manifest


def _cell(
    coordinate_id: str,
    *,
    start: float = 200_000.0,
    end: float = 620_000.0,
    status: str = "COMPLETE",
    minimum: float = 190_000.0,
    ruin: int = 0,
    trade_count: int = 1,
    predecessor_sha: str | None = None,
    carry_sha: str | None = None,
) -> dict[str, Any]:
    if status == "FAILED":
        return {
            "coordinate_id": coordinate_id,
            "status": "FAILED",
            "starting_balance_jpy": None,
            "starting_equity_jpy": None,
            "ending_balance_jpy": None,
            "ending_equity_jpy": None,
            "minimum_mtm_equity_jpy": None,
            "minimum_free_margin_jpy": None,
            "max_mtm_drawdown_fraction": None,
            "peak_margin_usage_fraction": None,
            "margin_closeout_count": None,
            "ruin_event_count": None,
            "trade_count": None,
            "fill_count": None,
            "margin_reject_count": None,
            "financing_jpy": None,
            "transaction_cost_jpy": None,
            "predecessor_state_sha256": None,
            "carry_out_state_sha256": None,
        }
    return {
        "coordinate_id": coordinate_id,
        "status": "COMPLETE",
        "starting_balance_jpy": start,
        "starting_equity_jpy": start,
        "ending_balance_jpy": end,
        "ending_equity_jpy": end,
        "minimum_mtm_equity_jpy": minimum,
        "minimum_free_margin_jpy": start * 0.5,
        "max_mtm_drawdown_fraction": 0.05,
        "peak_margin_usage_fraction": 0.20,
        "margin_closeout_count": 0,
        "ruin_event_count": ruin,
        "trade_count": trade_count,
        "fill_count": trade_count * 2,
        "margin_reject_count": 0,
        "financing_jpy": 0.0,
        "transaction_cost_jpy": 10.0,
        "predecessor_state_sha256": predecessor_sha,
        "carry_out_state_sha256": carry_sha,
    }


def _record(
    *,
    coordinate_id: str,
    month: str = "2020-01",
    mode: str = "INDEPENDENT_MONTH",
    path: str = "OHLC",
    scenario: str = "STRESS",
    stage: str = "PORTFOLIO_MAIN",
    fold: str | None = None,
    source: str = "M5_EXACT28_2020_2026H1",
    granularity: str = "M5",
    cell: dict[str, Any] | None = None,
    chain_id: str | None = None,
    ordinal: int | None = None,
    predecessor_slot: str | None = None,
    carry_slot: str | None = None,
    first: bool = False,
    last: bool = False,
    aggregation_weight: float = 1.0,
) -> dict[str, Any]:
    return {
        "job": {
            "month": month,
            "intrabar_path": path,
            "source_binding_id": source,
            "granularity": granularity,
        },
        "coordinate": {
            "coordinate_id": coordinate_id,
            "evaluation_mode": mode,
            "cost_scenario": scenario,
            "stage": stage,
            "fold_label": fold,
            "continuous_account_chain_id": chain_id,
            "continuous_chain_ordinal": ordinal,
            "continuous_chain_first": first,
            "continuous_chain_last": last,
            "predecessor_state_slot_id": predecessor_slot,
            "carry_out_state_slot_id": carry_slot,
            "aggregation_weight": aggregation_weight,
        },
        "cell": cell or _cell(coordinate_id),
    }


def _gates() -> dict[str, Any]:
    return {
        "target_multiple": 3.0,
        "base_max_drawdown_fraction_max": 0.10,
        "stress_max_drawdown_fraction_max": 0.15,
        "peak_margin_fraction_max": 0.45,
        "margin_closeout_count_max": 0,
        "pair_lopo_profit_drop_fraction_max": 0.50,
        "family_lopo_profit_drop_fraction_max": 0.50,
        "currency_lopo_profit_drop_fraction_max": 0.50,
    }


def test_public_scorer_rejects_any_reduced_job_denominator(
    sealed_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    plan, schedule, manifest = sealed_inputs
    assert schedule["stream_job_count"] == 348
    assert schedule["result_coordinate_count"] == 32_112

    with pytest.raises(
        DojoLongHorizonResultReducerError,
        match="terminal denominator is not independently valid",
    ):
        score_long_horizon_results(
            plan=plan,
            schedule=schedule,
            execution_manifest=manifest,
            terminal_manifests=[],
            reducer_handoffs=[],
        )


def test_scorecard_authority_is_permanently_diagnostic_only() -> None:
    authority = _authority()

    assert authority["promotion_eligible"] is False
    assert authority["live_permission"] is False
    assert authority["order_authority"] == "NONE"
    assert authority["automatic_deployment_allowed"] is False
    assert authority["trainer_may_change_live_configuration"] is False


def test_self_reported_cells_cannot_open_the_official_economic_gate() -> None:
    gate_pass, blockers = _official_gate([])

    assert gate_pass is False
    assert blockers == [
        "COMPACT_ECONOMIC_EVIDENCE_NOT_INDEPENDENTLY_REEXECUTED",
        "SOURCE_QUOTE_COVERAGE_NOT_PROVED",
    ]

    gate_pass, blockers = _official_gate(["NOT_EVERY_MONTH_3X"])
    assert gate_pass is False
    assert blockers == [
        "COMPACT_ECONOMIC_EVIDENCE_NOT_INDEPENDENTLY_REEXECUTED",
        "SOURCE_QUOTE_COVERAGE_NOT_PROVED",
        "NOT_EVERY_MONTH_3X",
    ]

    gate_pass, blockers = _official_gate(
        [],
        independent_reexecution_passed=True,
        source_quote_coverage_proved=False,
    )
    assert gate_pass is False
    assert blockers == ["SOURCE_QUOTE_COVERAGE_NOT_PROVED"]

    gate_pass, blockers = _official_gate(
        [],
        independent_reexecution_passed=True,
        source_quote_coverage_proved=True,
    )
    assert gate_pass is True
    assert blockers == []


def test_runner_output_requirement_demands_causal_inputs_not_aggregate_hashes() -> None:
    requirement = long_horizon_economic_runner_output_requirements()

    assert (
        requirement["job_source_packet"][
            "source_receipt_hash_without_bound_source_bytes_is_sufficient"
        ]
        is False
    )
    assert (
        requirement["coordinate_reducer_input"][
            "event_only_fill_exit_ledger_is_sufficient"
        ]
        is False
    )
    assert requirement["independent_reexecution"]["required"] is True
    assert (
        requirement["independent_reexecution"][
            "runner_aggregate_result_is_an_input_to_reducer"
        ]
        is False
    )
    assert (
        requirement["current_trusted_reducer_gap"][
            "current_output_is_sufficient_for_long_horizon_cell"
        ]
        is False
    )
    assert (
        "minimum_mtm_equity_jpy"
        in requirement["current_trusted_reducer_gap"][
            "missing_or_ambiguous_independent_metrics"
        ]
    )
    assert requirement["continuous_carry"]["full_carry_state_bytes_required"] is True
    assert requirement["failure_evidence"]["failure_may_reduce_denominator"] is False
    assert (
        requirement["lopo"]["post_hoc_subtraction_from_full_portfolio_allowed"] is False
    )
    assert requirement["authority"]["live_permission"] is False
    unsigned = {
        key: value for key, value in requirement.items() if key != "requirements_sha256"
    }
    assert requirement["requirements_sha256"] == canonical_sha256(unsigned)


def test_zero_trade_complete_month_is_one_x_and_fails_three_x_without_becoming_failed() -> (
    None
):
    records: list[dict[str, Any]] = []
    for mode in ("INDEPENDENT_MONTH", "CONTINUOUS_ACCOUNT"):
        for scenario in ("BASE", "STRESS"):
            for path in ("OHLC", "OLHC"):
                coordinate_id = canonical_sha256([mode, scenario, path])
                records.append(
                    _record(
                        coordinate_id=coordinate_id,
                        mode=mode,
                        scenario=scenario,
                        path=path,
                        cell=_cell(
                            coordinate_id,
                            start=200_000.0,
                            end=200_000.0,
                            trade_count=0,
                        ),
                    )
                )

    result = _monthly_main_summary(records, months=("2020-01",), gates=_gates())

    for mode in ("INDEPENDENT_MONTH", "CONTINUOUS_ACCOUNT"):
        row = result["modes"][mode]["months"][0]
        assert row["all_four_main_cells_complete"] is True
        assert row["pessimistic_stress_multiple"] == 1.0
        assert row["reached_3x"] is False
    assert result["independent_and_continuous_every_month_3x"] is False


def test_failed_cell_stays_unknown_and_is_never_zero_filled() -> None:
    records: list[dict[str, Any]] = []
    for mode in ("INDEPENDENT_MONTH", "CONTINUOUS_ACCOUNT"):
        for scenario in ("BASE", "STRESS"):
            for path in ("OHLC", "OLHC"):
                coordinate_id = canonical_sha256([mode, scenario, path])
                status = (
                    "FAILED"
                    if (mode, scenario, path) == ("INDEPENDENT_MONTH", "STRESS", "OLHC")
                    else "COMPLETE"
                )
                records.append(
                    _record(
                        coordinate_id=coordinate_id,
                        mode=mode,
                        scenario=scenario,
                        path=path,
                        cell=_cell(coordinate_id, status=status),
                    )
                )

    result = _monthly_main_summary(records, months=("2020-01",), gates=_gates())
    independent = result["modes"]["INDEPENDENT_MONTH"]["months"][0]

    assert independent["cells"]["STRESS"]["OLHC"]["monthly_ending_multiple"] is None
    assert independent["pessimistic_stress_multiple"] is None
    assert result["modes"]["INDEPENDENT_MONTH"]["unknown_month_count"] == 1
    assert result["independent_and_continuous_every_month_3x"] is False


def test_lopo_compares_aligned_monthly_multiples_even_when_capital_levels_differ() -> (
    None
):
    full_id = "1" * 64
    pair_id = "2" * 64
    family_id = "3" * 64
    currency_id = "4" * 64
    records = [
        _record(
            coordinate_id=full_id,
            cell=_cell(full_id, start=1_000.0, end=3_000.0),
        ),
        _record(
            coordinate_id=pair_id,
            stage="PAIR_LOPO",
            fold="USD_JPY",
            cell=_cell(pair_id, start=10_000.0, end=25_000.0),
        ),
        _record(
            coordinate_id=family_id,
            stage="FAMILY_LOPO",
            fold="spike_fade",
            cell=_cell(family_id, start=100_000.0, end=250_000.0),
        ),
        _record(
            coordinate_id=currency_id,
            stage="CURRENCY_LOPO",
            fold="JPY",
            cell=_cell(currency_id, start=1_000_000.0, end=2_500_000.0),
        ),
    ]

    result = _lopo_summary(records, gates=_gates())

    assert _profit_drop_fraction(3.0, 2.5) == 0.25
    assert result["stages"]["PAIR_LOPO"]["maximum_profit_drop_fraction"] == 0.25
    assert result["stages"]["FAMILY_LOPO"]["maximum_profit_drop_fraction"] == 0.25
    assert result["stages"]["CURRENCY_LOPO"]["maximum_profit_drop_fraction"] == 0.25
    assert result["all_lopo_gates_pass"] is True


def test_continuous_state_sha_and_equity_mismatches_fail_the_chain_gate() -> None:
    first_id = "5" * 64
    second_id = "6" * 64
    first = _record(
        coordinate_id=first_id,
        mode="CONTINUOUS_ACCOUNT",
        month="2020-01",
        chain_id="7" * 64,
        ordinal=0,
        carry_slot="8" * 64,
        first=True,
        cell=_cell(
            first_id,
            start=200_000.0,
            end=620_000.0,
            carry_sha="9" * 64,
        ),
    )
    second = _record(
        coordinate_id=second_id,
        mode="CONTINUOUS_ACCOUNT",
        month="2020-02",
        chain_id="7" * 64,
        ordinal=1,
        predecessor_slot="8" * 64,
        carry_slot="a" * 64,
        last=True,
        cell=_cell(
            second_id,
            start=610_000.0,
            end=1_891_000.0,
            predecessor_sha="b" * 64,
            carry_sha="c" * 64,
        ),
    )

    result = _continuous_chain_summary([first, second])

    assert result["continuous_chain_gate_pass"] is False
    issue_codes = result["continuous_chain_issue_sample"][0]["issue_codes"]
    assert "STATE_SHA_CHAIN_MISMATCH" in issue_codes
    assert "EQUITY_CHAIN_MISMATCH" in issue_codes


def test_m1_contexts_are_distinct_full_weight_observations() -> None:
    records = [
        _record(
            coordinate_id="d" * 64,
            source="M1_CORE5_2020_2026H1",
            granularity="M1",
            aggregation_weight=1.0,
        ),
        _record(
            coordinate_id="e" * 64,
            source="M1_FULL28_2025_2026H1",
            granularity="M1",
            aggregation_weight=1.0,
        ),
    ]

    result = _m1_context_summary(records)

    assert result["all_contexts_complete"] is True
    assert result["contexts"]["M1_CORE5_2020_2026H1"]["aggregation_weight_sum"] == 1.0
    assert result["contexts"]["M1_FULL28_2025_2026H1"]["aggregation_weight_sum"] == 1.0
    assert (
        sum(row["aggregation_weight_sum"] for row in result["contexts"].values()) == 2.0
    )


@pytest.mark.parametrize(
    ("minimum_equity", "ruin_events"),
    [(0.0, 1), (-10.0, 1), (100.0, 1)],
)
def test_nonpositive_minimum_equity_or_ruin_event_fails_global_survival_gate(
    minimum_equity: float, ruin_events: int
) -> None:
    coordinate_id = "f" * 64
    records = [
        _record(
            coordinate_id=coordinate_id,
            cell=_cell(
                coordinate_id,
                minimum=minimum_equity,
                ruin=ruin_events,
            ),
        )
    ]

    result = _risk_summary(records, gates=_gates())

    assert result["observed_minimum_mtm_equity_jpy"] == minimum_equity
    assert result["observed_ruin_event_count"] == ruin_events
    assert result["gate_pass"] is False
    assert result["risk_or_ruin_violation_count"] == 1

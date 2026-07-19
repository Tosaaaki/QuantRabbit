from __future__ import annotations

import pytest

import quant_rabbit.dojo_long_horizon_schedule as schedule_module
from quant_rabbit.dojo_long_horizon_plan import (
    IMPLEMENTATION_DIGEST_KEYS,
    M1_CORE5_BINDING_ID,
    M1_FULL28_BINDING_ID,
    M5_BINDING_ID,
    SOURCE_BINDING_IDS,
    build_long_horizon_train_plan,
    canonical_sha256,
)
from quant_rabbit.dojo_long_horizon_schedule import (
    MAX_WORKERS,
    MAX_WORKERS_PER_FAMILY,
    DojoLongHorizonScheduleError,
    build_long_horizon_stream_schedule,
    validate_long_horizon_stream_schedule,
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
def plan() -> dict:
    return build_long_horizon_train_plan(
        portfolio_families=FAMILIES,
        source_digests=_digests(SOURCE_BINDING_IDS, 0),
        corpus_digests=_digests(SOURCE_BINDING_IDS, 10),
        implementation_digests=_digests(IMPLEMENTATION_DIGEST_KEYS, 20),
    )


@pytest.fixture(scope="module")
def schedule(plan: dict) -> dict:
    return build_long_horizon_stream_schedule(plan, worker_bindings=WORKERS)


def test_expands_exact_fixed_denominator_into_bounded_stream_jobs(
    plan: dict, schedule: dict
) -> None:
    assert schedule["stream_job_count"] == (78 + 78 + 18) * 2 == 348
    assert schedule["result_coordinate_count"] == 32_112
    assert (
        schedule["expected_result_coordinate_count"]
        == plan["exact_denominator"]["total_required_result_cell_count"]
    )
    assert sum(job["coordinate_count"] for job in schedule["jobs"]) == 32_112
    ids = [
        coordinate["coordinate_id"]
        for job in schedule["jobs"]
        for coordinate in job["coordinates"]
    ]
    assert len(ids) == len(set(ids)) == 32_112
    assert schedule["all_coordinate_ids_sha256"] == canonical_sha256(ids)
    assert validate_long_horizon_stream_schedule(schedule, plan=plan) == schedule


def test_each_job_requires_one_stream_fanned_out_before_decisions(
    schedule: dict,
) -> None:
    assert all(job["source_stream_instance_count"] == 1 for job in schedule["jobs"])
    assert all(
        job["fanout_before_any_coordinate_decision"] is True for job in schedule["jobs"]
    )
    assert all(
        job["coordinate_may_reopen_or_resort_source"] is False
        for job in schedule["jobs"]
    )
    assert all(job["synthetic_quote_count"] == 0 for job in schedule["jobs"])
    assert all(
        job["runtime_batch_chain_receipt_required"] is True for job in schedule["jobs"]
    )
    assert (
        schedule["execution_contract"][
            "runner_implementation_verified_by_this_artifact"
        ]
        is False
    )


def test_m5_job_fans_main_and_all_lopo_accounts_from_same_price_stream(
    schedule: dict,
) -> None:
    job = next(
        row
        for row in schedule["jobs"]
        if row["source_binding_id"] == M5_BINDING_ID
        and row["month"] == "2020-01"
        and row["intrabar_path"] == "OHLC"
    )
    # Per job: 2 account modes × 2 cost scenarios ×
    # (main + 28 pair LOPO + 3 family LOPO + 8 currency LOPO).
    assert job["coordinate_count"] == 2 * 2 * (1 + 28 + 3 + 8) == 160
    coordinate_ids = {coordinate["coordinate_id"] for coordinate in job["coordinates"]}
    assert len(coordinate_ids) == 160
    assert job["price_stream_id"]
    assert len(job["feed_pairs"]) == 28


def test_lopo_coordinate_removes_exact_exposure_not_the_feed(schedule: dict) -> None:
    m5_jobs = [
        job for job in schedule["jobs"] if job["source_binding_id"] == M5_BINDING_ID
    ]
    assert len(m5_jobs) == 78 * 2
    assert len({job["price_stream_id"] for job in m5_jobs}) == len(m5_jobs)
    assert all(job["coordinate_count"] == 160 for job in m5_jobs)
    job = m5_jobs[0]

    def selected(values: tuple[str, ...] | list[str], mask: str) -> list[str]:
        return [value for value, bit in zip(values, mask, strict=True) if bit == "1"]

    pair_fold = next(
        row
        for row in job["coordinates"]
        if row["stage"] == "PAIR_LOPO" and row["fold_label"] == "USD_JPY"
    )
    pair_fold_pairs = selected(job["feed_pairs"], pair_fold["trade_pair_mask"])
    assert "USD_JPY" not in pair_fold_pairs
    assert len(pair_fold_pairs) == pair_fold["trade_pair_count"] == 27

    family_fold = next(
        row
        for row in job["coordinates"]
        if row["stage"] == "FAMILY_LOPO" and row["fold_label"] == "spike_fade"
    )
    family_fold_families = selected(FAMILIES, family_fold["active_family_mask"])
    assert "spike_fade" not in family_fold_families
    assert len(family_fold_families) == family_fold["active_family_count"] == 2
    assert family_fold["active_worker_mask"] == "110"
    assert family_fold["active_worker_count"] == 2

    currency_fold = next(
        row
        for row in job["coordinates"]
        if row["stage"] == "CURRENCY_LOPO" and row["fold_label"] == "JPY"
    )
    currency_fold_pairs = selected(job["feed_pairs"], currency_fold["trade_pair_mask"])
    assert len(currency_fold_pairs) == currency_fold["trade_pair_count"] == 21
    assert all("JPY" not in pair.split("_") for pair in currency_fold_pairs)


def test_m1_rectangles_remain_distinct_and_keep_declared_overlap(
    schedule: dict,
) -> None:
    core = next(
        job
        for job in schedule["jobs"]
        if job["source_binding_id"] == M1_CORE5_BINDING_ID
        and job["month"] == "2025-01"
        and job["intrabar_path"] == "OHLC"
    )
    full = next(
        job
        for job in schedule["jobs"]
        if job["source_binding_id"] == M1_FULL28_BINDING_ID
        and job["month"] == "2025-01"
        and job["intrabar_path"] == "OHLC"
    )
    assert core["coordinate_count"] == 5 * 2 * 2 == 20
    assert full["coordinate_count"] == 28 * 2 * 2 == 112
    assert core["price_stream_id"] != full["price_stream_id"]
    assert set(core["feed_pairs"]).issubset(set(full["feed_pairs"]))

    def precision(job: dict, pair: str) -> dict:
        return next(
            row
            for row in job["coordinates"]
            if row["precision_pair"] == pair
            and row["evaluation_mode"] == "INDEPENDENT_MONTH"
            and row["cost_scenario"] == "BASE"
        )

    core_replica = precision(core, "USD_JPY")
    full_replica = precision(full, "USD_JPY")
    assert core_replica["replica_group_id"] == full_replica["replica_group_id"]
    assert core_replica["aggregation_weight"] == 0.5
    assert full_replica["aggregation_weight"] == 0.5
    assert core_replica["replica_expected_count"] == 2
    assert core_replica["replica_paired_consistency_required"] is True

    non_overlap = next(
        job
        for job in schedule["jobs"]
        if job["source_binding_id"] == M1_CORE5_BINDING_ID
        and job["month"] == "2024-12"
        and job["intrabar_path"] == "OHLC"
    )
    ordinary = precision(non_overlap, "USD_JPY")
    assert ordinary["aggregation_weight"] == 1.0
    assert ordinary["replica_expected_count"] == 1
    assert ordinary["replica_paired_consistency_required"] is False

    assert schedule["effective_weighted_result_coordinate_count"] == 31_392
    assert schedule["expected_effective_weighted_result_coordinate_count"] == 31_392
    replica_contract = schedule["m1_replica_contract"]
    assert replica_contract["overlap_pair_month_count"] == 5 * 18 == 90
    assert replica_contract["raw_overlap_replica_coordinate_count"] == 1_440
    assert replica_contract["effective_overlap_result_count"] == 720
    assert replica_contract["paired_terminal_consistency_required"] is True


def test_continuous_and_independent_accounts_are_separate_coordinates(
    schedule: dict,
) -> None:
    contract = schedule["execution_contract"]
    assert contract["one_shared_account_per_coordinate"] is True
    assert contract["account_state_shared_between_coordinates"] is False
    assert contract["continuous_account_state_chained_across_months"] is True
    assert contract["independent_month_state_reset"] is True
    assert contract["runtime_terminal_manifest_per_coordinate_required"] is True

    january = next(
        job
        for job in schedule["jobs"]
        if job["source_binding_id"] == M5_BINDING_ID
        and job["month"] == "2020-01"
        and job["intrabar_path"] == "OHLC"
    )
    february = next(
        job
        for job in schedule["jobs"]
        if job["source_binding_id"] == M5_BINDING_ID
        and job["month"] == "2020-02"
        and job["intrabar_path"] == "OHLC"
    )

    def main(job: dict, mode: str) -> dict:
        return next(
            row
            for row in job["coordinates"]
            if row["stage"] == "PORTFOLIO_MAIN"
            and row["evaluation_mode"] == mode
            and row["cost_scenario"] == "BASE"
        )

    january_continuous = main(january, "CONTINUOUS_ACCOUNT")
    february_continuous = main(february, "CONTINUOUS_ACCOUNT")
    assert (
        january_continuous["continuous_account_chain_id"]
        == february_continuous["continuous_account_chain_id"]
    )
    assert january_continuous["coordinate_id"] != february_continuous["coordinate_id"]
    assert january_continuous["continuous_chain_ordinal"] == 0
    assert january_continuous["continuous_chain_first"] is True
    assert january_continuous["continuous_chain_last"] is False
    assert january_continuous["predecessor_state_slot_id"] is None
    assert (
        january_continuous["carry_out_state_slot_id"]
        == february_continuous["carry_in_state_slot_id"]
        == february_continuous["predecessor_state_slot_id"]
    )
    assert february_continuous["continuous_chain_ordinal"] == 1
    assert february_continuous["continuous_chain_first"] is False

    final_job = next(
        job
        for job in schedule["jobs"]
        if job["source_binding_id"] == M5_BINDING_ID
        and job["month"] == "2026-06"
        and job["intrabar_path"] == "OHLC"
    )
    final_continuous = main(final_job, "CONTINUOUS_ACCOUNT")
    assert final_continuous["continuous_chain_ordinal"] == 77
    assert final_continuous["continuous_chain_last"] is True
    assert final_continuous["carry_out_state_slot_id"] is not None

    independent = main(january, "INDEPENDENT_MONTH")
    assert independent["continuous_account_chain_id"] is None
    assert independent["continuous_chain_ordinal"] is None
    assert independent["continuous_chain_first"] is False
    assert independent["continuous_chain_last"] is False
    assert independent["carry_in_state_slot_id"] is None
    assert independent["carry_out_state_slot_id"] is None
    assert independent["predecessor_state_slot_id"] is None


def test_worker_family_coverage_is_exact_and_authority_is_absent(
    plan: dict, schedule: dict
) -> None:
    missing = WORKERS[:-1]
    with pytest.raises(DojoLongHorizonScheduleError, match="every sealed"):
        build_long_horizon_stream_schedule(plan, worker_bindings=missing)
    duplicate = (
        WORKERS[0],
        {**WORKERS[1], "worker_id": WORKERS[0]["worker_id"]},
        WORKERS[2],
    )
    with pytest.raises(DojoLongHorizonScheduleError, match="sorted|unique"):
        build_long_horizon_stream_schedule(plan, worker_bindings=duplicate)
    bad_sha = tuple({**row} for row in WORKERS)
    bad_sha[0]["config_sha256"] = "0" * 64
    with pytest.raises(DojoLongHorizonScheduleError, match="non-zero"):
        build_long_horizon_stream_schedule(plan, worker_bindings=bad_sha)

    too_many_one_family = [
        {
            "worker_id": f"breakout-{index:02d}",
            "family_id": "breakout",
            "config_sha256": f"{index + 1:064x}",
        }
        for index in range(MAX_WORKERS_PER_FAMILY + 1)
    ]
    too_many_one_family.extend(
        [
            {
                "worker_id": "range-00",
                "family_id": "range_fade",
                "config_sha256": "e" * 64,
            },
            {
                "worker_id": "spike-00",
                "family_id": "spike_fade",
                "config_sha256": "f" * 64,
            },
        ]
    )
    with pytest.raises(DojoLongHorizonScheduleError, match="per-family"):
        build_long_horizon_stream_schedule(
            plan,
            worker_bindings=tuple(
                sorted(too_many_one_family, key=lambda row: row["worker_id"])
            ),
        )

    too_many_total = [
        {
            "worker_id": f"worker-{index:02d}",
            "family_id": FAMILIES[index % len(FAMILIES)],
            "config_sha256": f"{index + 1:064x}",
        }
        for index in range(MAX_WORKERS + 1)
    ]
    with pytest.raises(DojoLongHorizonScheduleError, match="hard cap"):
        build_long_horizon_stream_schedule(plan, worker_bindings=too_many_total)

    assert schedule["authority"] == {
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "diagnostic_only": True,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
    }


def test_worker_config_mutation_changes_every_execution_identity(
    plan: dict, schedule: dict
) -> None:
    changed_workers = tuple({**row} for row in WORKERS)
    changed_workers[0]["config_sha256"] = "d" * 64
    changed_worker_set_sha256 = canonical_sha256({"bindings": list(changed_workers)})
    _, changed_active_variants_sha256 = schedule_module._active_worker_variants(
        changed_workers, families=FAMILIES
    )
    assert changed_worker_set_sha256 != schedule["worker_set"]["worker_set_sha256"]
    original_job = schedule["jobs"][0]
    changed_stream_body = {
        "plan_sha256": plan["plan_sha256"],
        "worker_set_sha256": changed_worker_set_sha256,
        "active_worker_variants_sha256": changed_active_variants_sha256,
        "source_binding_id": original_job["source_binding_id"],
        "month": original_job["month"],
        "intrabar_path": original_job["intrabar_path"],
        "source_digest_sha256": original_job["source_digest_sha256"],
        "corpus_digest_sha256": original_job["corpus_digest_sha256"],
    }
    changed_price_stream_id = canonical_sha256(changed_stream_body)
    changed_coordinates = schedule_module._m5_coordinates(
        plan=plan,
        source_binding_id=original_job["source_binding_id"],
        month=original_job["month"],
        intrabar_path=original_job["intrabar_path"],
        price_stream_id=changed_price_stream_id,
        feed_pairs=original_job["feed_pairs"],
        worker_rows=changed_workers,
        plan_sha256=plan["plan_sha256"],
        worker_set_sha256=changed_worker_set_sha256,
        month_ordinal=0,
        month_count=78,
    )
    changed_coordinate_ids = [row["coordinate_id"] for row in changed_coordinates]
    changed_job_body = {
        key: value for key, value in original_job.items() if key != "job_sha256"
    }
    changed_job_body.update(
        {
            "worker_set_sha256": changed_worker_set_sha256,
            "active_worker_variants_sha256": changed_active_variants_sha256,
            "price_stream_id": changed_price_stream_id,
            "coordinates": changed_coordinates,
            "coordinate_ids_sha256": canonical_sha256(changed_coordinate_ids),
        }
    )
    changed_job_sha256 = canonical_sha256(changed_job_body)
    original_coordinate = next(
        row
        for row in original_job["coordinates"]
        if row["stage"] == "PORTFOLIO_MAIN"
        and row["evaluation_mode"] == "CONTINUOUS_ACCOUNT"
        and row["cost_scenario"] == "BASE"
    )
    changed_coordinate = next(
        row
        for row in changed_coordinates
        if row["stage"] == "PORTFOLIO_MAIN"
        and row["evaluation_mode"] == "CONTINUOUS_ACCOUNT"
        and row["cost_scenario"] == "BASE"
    )
    assert changed_price_stream_id != original_job["price_stream_id"]
    assert changed_job_sha256 != original_job["job_sha256"]
    assert changed_coordinate["coordinate_id"] != original_coordinate["coordinate_id"]
    assert (
        changed_coordinate["continuous_account_chain_id"]
        != original_coordinate["continuous_account_chain_id"]
    )
    assert (
        changed_coordinate["active_worker_bindings_sha256"]
        != original_coordinate["active_worker_bindings_sha256"]
    )
    assert (
        changed_coordinate["carry_in_state_slot_id"]
        != original_coordinate["carry_in_state_slot_id"]
    )


def test_rehashed_tamper_is_rejected_by_structural_rebuild(
    plan: dict, schedule: dict
) -> None:
    original_contract_value = schedule["execution_contract"][
        "account_state_shared_between_coordinates"
    ]
    original_schedule_sha256 = schedule["schedule_sha256"]
    schedule["execution_contract"]["account_state_shared_between_coordinates"] = True
    body = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    schedule["schedule_sha256"] = canonical_sha256(body)
    try:
        with pytest.raises(DojoLongHorizonScheduleError, match="drifted"):
            validate_long_horizon_stream_schedule(schedule, plan=plan)
    finally:
        schedule["execution_contract"]["account_state_shared_between_coordinates"] = (
            original_contract_value
        )
        schedule["schedule_sha256"] = original_schedule_sha256

    schedule["note"] = "looks harmless"
    try:
        with pytest.raises(DojoLongHorizonScheduleError, match="top-level"):
            validate_long_horizon_stream_schedule(schedule, plan=plan)
    finally:
        del schedule["note"]

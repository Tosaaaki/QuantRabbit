from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from quant_rabbit.dojo_long_horizon_execution import (
    CELL_CONTRACT,
    MAX_ATTEMPTS_PER_JOB,
    DojoLongHorizonExecutionError,
    LongHorizonExecutionSession,
    build_long_horizon_coordinate_result,
    build_long_horizon_execution_manifest,
    initialize_long_horizon_execution_state,
    validate_long_horizon_execution_manifest,
    validate_long_horizon_coordinate_result,
    validate_long_horizon_reducer_handoff,
    validate_long_horizon_terminal_manifest,
    validate_long_horizon_terminal_manifests,
)
from quant_rabbit.dojo_long_horizon_plan import (
    IMPLEMENTATION_DIGEST_KEYS,
    M5_BINDING_ID,
    SOURCE_BINDING_IDS,
    build_long_horizon_train_plan,
    canonical_sha256,
)
from quant_rabbit.dojo_long_horizon_schedule import (
    MAX_WORKERS,
    MAX_WORKERS_PER_FAMILY,
    build_long_horizon_stream_schedule,
)


FAMILIES = ("breakout", "range_fade", "spike_fade")
WORKERS = (
    {"worker_id": "breakout-v1", "family_id": "breakout", "config_sha256": "a" * 64},
    {"worker_id": "range-v1", "family_id": "range_fade", "config_sha256": "b" * 64},
    {"worker_id": "spike-v1", "family_id": "spike_fade", "config_sha256": "c" * 64},
)
RUNNER_BINDING = {
    "runner_contract": "QR_TEST_BROKER_FREE_RUNNER_V1",
    "runner_code_sha256": "d" * 64,
    "result_contract": CELL_CONTRACT,
}
RESOURCE_POLICY = {
    "max_resident_coordinates": 160,
    "max_rss_bytes": 2_147_483_648,
    "max_open_files": 256,
    "min_free_disk_bytes": 5_368_709_120,
    "max_checkpoint_bytes": 8_388_608,
    "max_terminal_bytes": 2_097_152,
    "max_parallel_jobs": 4,
}


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


@pytest.fixture(scope="module")
def manifest(plan: dict, schedule: dict) -> dict:
    return build_long_horizon_execution_manifest(
        schedule,
        plan=plan,
        runner_binding=RUNNER_BINDING,
        resource_policy=RESOURCE_POLICY,
    )


def _job(schedule: dict, *, month: str, path: str = "OHLC") -> dict:
    return next(
        job
        for job in schedule["jobs"]
        if job["source_binding_id"] == M5_BINDING_ID
        and job["month"] == month
        and job["intrabar_path"] == path
    )


def _complete_result(
    *,
    job: dict,
    claim: dict,
    coordinate: dict,
    predecessor_state_sha256: str | None = None,
) -> dict:
    identity = coordinate["coordinate_id"]
    return build_long_horizon_coordinate_result(
        job=job,
        claim=claim,
        coordinate_id=identity,
        status="COMPLETE",
        starting_balance_jpy=200_000.0,
        starting_equity_jpy=200_000.0,
        ending_balance_jpy=201_000.0,
        ending_equity_jpy=200_900.0,
        minimum_mtm_equity_jpy=198_000.0,
        minimum_free_margin_jpy=100_000.0,
        max_mtm_drawdown_fraction=0.01,
        peak_margin_usage_fraction=0.25,
        margin_closeout_count=0,
        ruin_event_count=0,
        trade_count=1,
        fill_count=2,
        margin_reject_count=0,
        financing_jpy=-10.0,
        transaction_cost_jpy=20.0,
        source_slice_receipt_sha256=canonical_sha256({"source": job["job_sha256"]}),
        batch_chain_sha256=canonical_sha256({"batch": job["job_sha256"]}),
        compact_evidence_sha256=canonical_sha256({"cell": identity}),
        quote_coverage_complete=True,
        active_worker_ack_complete=True,
        predecessor_state_sha256=predecessor_state_sha256,
        carry_out_state_sha256=(
            canonical_sha256({"carry": identity})
            if coordinate["carry_out_state_slot_id"] is not None
            else None
        ),
    )


def _failed_result(*, job: dict, claim: dict, coordinate: dict, code: str) -> dict:
    return build_long_horizon_coordinate_result(
        job=job,
        claim=claim,
        coordinate_id=coordinate["coordinate_id"],
        status="FAILED",
        failure={
            "code": code,
            "retryable": False,
            "evidence_sha256": canonical_sha256(
                {"failure": coordinate["coordinate_id"], "code": code}
            ),
        },
    )


def test_manifest_seals_jobs_workers_resources_and_no_authority(
    plan: dict, schedule: dict, manifest: dict
) -> None:
    assert manifest["job_count"] == schedule["stream_job_count"] == 348
    assert manifest["result_coordinate_count"] == 32_112
    assert manifest["resource_limits"]["max_workers_global"] == MAX_WORKERS == 32
    assert (
        manifest["resource_limits"]["max_workers_per_family"]
        == MAX_WORKERS_PER_FAMILY
        == 8
    )
    assert (
        manifest["resource_limits"]["max_attempts_per_job"] == MAX_ATTEMPTS_PER_JOB == 1
    )
    assert manifest["resource_enforcement"] == {
        "runner_must_measure_limits": True,
        "state_machine_observes_process_metrics": False,
        "resource_limit_action": "PAUSE_AND_RESUME_SAME_CLAIM",
        "resource_limit_may_create_failed_terminal": False,
        "resource_limit_may_create_new_claim": False,
        "unbounded_execution_allowed": False,
    }
    assert manifest["authority"]["broker_mutation_allowed"] is False
    assert manifest["authority"]["live_permission"] is False
    assert (
        validate_long_horizon_execution_manifest(manifest, schedule=schedule, plan=plan)
        == manifest
    )


def test_manifest_rejects_resource_or_content_address_drift(
    plan: dict, schedule: dict, manifest: dict
) -> None:
    tampered = copy.deepcopy(manifest)
    tampered["jobs"][0]["coordinate_count"] += 1
    tampered["jobs"][0]["execution_job_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered["jobs"][0].items()
            if key != "execution_job_sha256"
        }
    )
    tampered["all_execution_job_ids_sha256"] = canonical_sha256(
        [row["execution_job_sha256"] for row in tampered["jobs"]]
    )
    tampered["execution_manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered.items()
            if key != "execution_manifest_sha256"
        }
    )
    with pytest.raises(DojoLongHorizonExecutionError, match="drifted"):
        validate_long_horizon_execution_manifest(tampered, schedule=schedule, plan=plan)


def test_failed_payload_cannot_hide_partial_or_zero_economics(
    schedule: dict, manifest: dict
) -> None:
    job = schedule["jobs"][0]
    claim = {
        "contract": "QR_DOJO_LONG_HORIZON_JOB_CLAIM_V1",
        "schema_version": 1,
        "execution_manifest_sha256": manifest["execution_manifest_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "job_sha256": job["job_sha256"],
        "attempt_ordinal": 1,
        "runner_id": "runner-test",
        "required_predecessor_state_slot_count": 0,
        "required_predecessor_state_slot_ids_sha256": canonical_sha256([]),
        "authority": manifest["authority"],
    }
    claim["claim_sha256"] = canonical_sha256(claim)
    coordinate = job["coordinates"][0]
    failed = _failed_result(
        job=job, claim=claim, coordinate=coordinate, code="RUNNER_FAILED"
    )
    tampered = copy.deepcopy(failed)
    tampered["ending_equity_jpy"] = 0.0
    tampered["cell_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "cell_sha256"}
    )
    with pytest.raises(DojoLongHorizonExecutionError, match="masquerade"):
        validate_long_horizon_coordinate_result(tampered, job=job, claim=claim)
    zero_trade = _complete_result(job=job, claim=claim, coordinate=coordinate)
    zero_trade["trade_count"] = 0
    zero_trade["fill_count"] = 0
    zero_trade["ending_balance_jpy"] = zero_trade["starting_balance_jpy"]
    zero_trade["ending_equity_jpy"] = zero_trade["starting_equity_jpy"]
    zero_trade["cell_sha256"] = canonical_sha256(
        {key: value for key, value in zero_trade.items() if key != "cell_sha256"}
    )
    assert (
        validate_long_horizon_coordinate_result(zero_trade, job=job, claim=claim)[
            "trade_count"
        ]
        == 0
    )
    hidden_ruin = copy.deepcopy(zero_trade)
    hidden_ruin["minimum_mtm_equity_jpy"] = -1.0
    hidden_ruin["ruin_event_count"] = 0
    hidden_ruin["cell_sha256"] = canonical_sha256(
        {key: value for key, value in hidden_ruin.items() if key != "cell_sha256"}
    )
    with pytest.raises(DojoLongHorizonExecutionError, match="ruin event"):
        validate_long_horizon_coordinate_result(hidden_ruin, job=job, claim=claim)


def test_o_excl_resume_failed_absorbing_and_coordinate_scoped_carry(
    tmp_path: Path, plan: dict, schedule: dict, manifest: dict
) -> None:
    state = tmp_path / "long-execution"
    initialized = initialize_long_horizon_execution_state(
        state,
        schedule=schedule,
        plan=plan,
        runner_binding=RUNNER_BINDING,
        resource_policy=RESOURCE_POLICY,
    )
    assert initialized == manifest
    session = LongHorizonExecutionSession(state, schedule=schedule, plan=plan)
    january = _job(schedule, month="2020-01")
    february = _job(schedule, month="2020-02")

    jan_handoff = session.claim_job(
        job_sha256=january["job_sha256"],
        runner_id="runner-jan",
    )
    claim = jan_handoff["claim"]
    assert jan_handoff["runnable_coordinate_count"] == january["coordinate_count"]
    assert jan_handoff["predecessor_blocked_coordinate_count"] == 0
    assert (
        session.claim_job(
            job_sha256=january["job_sha256"],
            runner_id="runner-jan",
        )["claim"]
        == claim
    )
    with pytest.raises(DojoLongHorizonExecutionError, match="active claim"):
        session.claim_job(
            job_sha256=january["job_sha256"],
            runner_id="competing-runner",
        )

    # Claiming February does not withhold its independent coordinates while
    # continuous coordinates wait for exact January predecessor cells.
    feb_handoff = session.claim_job(
        job_sha256=february["job_sha256"],
        runner_id="runner-feb",
    )
    assert feb_handoff["runnable_coordinate_count"] > 0
    assert feb_handoff["predecessor_blocked_coordinate_count"] > 0
    feb_coordinate_by_id = {
        row["coordinate_id"]: row for row in february["coordinates"]
    }
    assert all(
        feb_coordinate_by_id[coordinate_id]["evaluation_mode"] == "INDEPENDENT_MONTH"
        for coordinate_id in feb_handoff["runnable_coordinate_ids"]
    )
    assert all(
        feb_coordinate_by_id[coordinate_id]["evaluation_mode"] == "CONTINUOUS_ACCOUNT"
        for coordinate_id in feb_handoff["predecessor_blocked_coordinate_ids"]
    )

    first = january["coordinates"][0]
    first_result = _complete_result(job=january, claim=claim, coordinate=first)
    session.record_result(
        claim_sha256=claim["claim_sha256"],
        result=first_result,
    )
    resumed = session.resume(claim_sha256=claim["claim_sha256"])
    assert resumed["recorded_coordinate_count"] == 1
    assert resumed["pending_coordinate_count"] == january["coordinate_count"] - 1
    altered = copy.deepcopy(first_result)
    altered["ending_equity_jpy"] += 1.0
    altered["cell_sha256"] = canonical_sha256(
        {key: value for key, value in altered.items() if key != "cell_sha256"}
    )
    with pytest.raises(DojoLongHorizonExecutionError, match="different bytes"):
        session.record_result(
            claim_sha256=claim["claim_sha256"],
            result=altered,
        )

    failed_continuous = next(
        coordinate
        for coordinate in january["coordinates"]
        if coordinate["evaluation_mode"] == "CONTINUOUS_ACCOUNT"
    )
    jan_results = []
    for coordinate in january["coordinates"][1:]:
        if coordinate["coordinate_id"] == failed_continuous["coordinate_id"]:
            jan_results.append(
                _failed_result(
                    job=january,
                    claim=claim,
                    coordinate=coordinate,
                    code="RUNNER_FAILED",
                )
            )
        else:
            jan_results.append(
                _complete_result(job=january, claim=claim, coordinate=coordinate)
            )
    session.record_results(
        claim_sha256=claim["claim_sha256"],
        results=jan_results,
    )
    sealed_january = session.seal(claim_sha256=claim["claim_sha256"])
    january_terminal = sealed_january["terminal_manifest"]
    assert january_terminal["terminal_status"] == "FAILED"
    assert january_terminal["failed_coordinate_count"] == 1
    assert (
        validate_long_horizon_terminal_manifest(
            january_terminal,
            schedule=schedule,
            plan=plan,
            execution_manifest=manifest,
        )
        == january_terminal
    )
    assert (
        validate_long_horizon_reducer_handoff(
            sealed_january["reducer_handoff"],
            terminal=january_terminal,
            schedule=schedule,
            plan=plan,
            execution_manifest=manifest,
        )
        == sealed_january["reducer_handoff"]
    )
    with pytest.raises(DojoLongHorizonExecutionError, match="absorbing"):
        session.claim_job(
            job_sha256=january["job_sha256"],
            runner_id="free-retry",
        )
    assert len(list((state / "claims" / january["job_sha256"]).glob("*.json"))) == 1

    # Resume dynamically publishes the exact failed predecessor only into the
    # corresponding continuous coordinate.  Other continuous chains receive
    # valid carry; already-runnable independent coordinates remain untouched.
    feb_resumed = session.resume(claim_sha256=feb_handoff["claim"]["claim_sha256"])
    feb_cell_dir = (
        state
        / "cells"
        / february["job_sha256"]
        / f"attempt-0001-{feb_handoff['claim']['claim_sha256']}"
    )
    generated = [json.loads(path.read_text()) for path in feb_cell_dir.glob("*.json")]
    assert len(generated) == 1
    assert generated[0]["status"] == "FAILED"
    assert generated[0]["failure"]["code"] == "PREDECESSOR_FAILED"
    generated_coordinate = feb_coordinate_by_id[generated[0]["coordinate_id"]]
    assert generated_coordinate["evaluation_mode"] == "CONTINUOUS_ACCOUNT"
    assert feb_resumed["runnable_coordinate_count"] == (
        february["coordinate_count"] - 1
    )
    assert feb_resumed["predecessor_blocked_coordinate_count"] == 0

    carry_by_slot = {
        item["state_slot_id"]: item["state_sha256"]
        for item in feb_resumed["ready_predecessor_carry_receipts"]
    }
    feb_results = []
    generated_id = generated[0]["coordinate_id"]
    for coordinate in february["coordinates"]:
        if coordinate["coordinate_id"] == generated_id:
            continue
        predecessor = coordinate["predecessor_state_slot_id"]
        feb_results.append(
            _complete_result(
                job=february,
                claim=feb_handoff["claim"],
                coordinate=coordinate,
                predecessor_state_sha256=(
                    carry_by_slot[predecessor] if predecessor is not None else None
                ),
            )
        )
    session.record_results(
        claim_sha256=feb_handoff["claim"]["claim_sha256"],
        results=feb_results,
    )
    sealed_february = session.seal(claim_sha256=feb_handoff["claim"]["claim_sha256"])
    assert sealed_february["terminal_manifest"]["terminal_status"] == "FAILED"
    assert sealed_february["terminal_manifest"]["failed_coordinate_count"] == 1

    status = session.status()
    assert status["terminal_job_count"] == 2
    assert status["failed_terminal_job_count"] == 2
    assert status["complete_job_count"] == 0
    assert status["profit_or_3x_status_computed"] is False
    assert status["all_cells_complete"] is False


def test_bulk_validator_requires_all_348_unique_job_terminals(
    plan: dict, schedule: dict, manifest: dict
) -> None:
    with pytest.raises(DojoLongHorizonExecutionError, match="exact job denominator"):
        validate_long_horizon_terminal_manifests(
            [], schedule=schedule, plan=plan, execution_manifest=manifest
        )


def test_cli_status_has_no_broker_or_profit_authority(
    tmp_path: Path, plan: dict, schedule: dict
) -> None:
    state = tmp_path / "cli-state"
    initialize_long_horizon_execution_state(
        state,
        schedule=schedule,
        plan=plan,
        runner_binding=RUNNER_BINDING,
        resource_policy=RESOURCE_POLICY,
    )
    plan_path = tmp_path / "plan.json"
    schedule_path = tmp_path / "schedule.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run-dojo-long-horizon-execution.py",
            "status",
            "--state-dir",
            str(state),
            "--plan",
            str(plan_path),
            "--schedule",
            str(schedule_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["profit_or_3x_status_computed"] is False
    assert result["authority"]["broker_mutation_allowed"] is False
    assert result["authority"]["live_permission"] is False

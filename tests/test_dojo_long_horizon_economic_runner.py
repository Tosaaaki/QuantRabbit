from __future__ import annotations

import copy
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import quant_rabbit.dojo_long_horizon_economic_runner as economic_runner_module
import quant_rabbit.dojo_long_horizon_source_manifest as source_manifest_contract

from quant_rabbit.dojo_long_horizon_economic_runner import (
    BUILTIN_NO_INTENT_RUNTIME_BINDING_SHA256,
    build_month_source_slice_receipt,
    builtin_no_intent_runtime_factory,
    run_long_horizon_economic_job,
    validate_month_source_slice_receipt,
)
from quant_rabbit.dojo_long_horizon_execution import (
    CELL_CONTRACT,
    LongHorizonExecutionSession,
    initialize_long_horizon_execution_state,
)
from quant_rabbit.dojo_long_horizon_plan import (
    IMPLEMENTATION_DIGEST_KEYS,
    M1_CORE5_BINDING_ID,
    SOURCE_BINDING_IDS,
    build_long_horizon_train_plan,
    canonical_sha256,
)
from quant_rabbit.dojo_long_horizon_schedule import (
    build_long_horizon_stream_schedule,
)
from quant_rabbit.dojo_portfolio_replay_reducer import seal_portfolio_policy


WORKERS = (
    {
        "worker_id": "test_worker_a",
        "family_id": "test_family_a",
        "config_sha256": "a" * 64,
    },
    {
        "worker_id": "test_worker_b",
        "family_id": "test_family_b",
        "config_sha256": "b" * 64,
    },
)
CATALOG = [
    {**worker, "owner_id": f"test_owner_{index}"}
    for index, worker in enumerate(WORKERS, 1)
]
RUNTIME_SHA = BUILTIN_NO_INTENT_RUNTIME_BINDING_SHA256
PLANS_BY_JOB: dict[str, dict[str, Any]] = {}


@pytest.fixture(autouse=True)
def _tiny_manifest_uses_production_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    # Production calls the full fixed-denominator pure verifier.  This focused
    # test uses the real top-level contract and nested seals but only five tiny
    # pair shards, so replace only the 28-pair/78-month denominator check.
    monkeypatch.setattr(
        economic_runner_module,
        "verify_long_horizon_source_manifest_seal",
        lambda value: dict(value),
    )


def _digests(keys: tuple[str, ...], offset: int) -> dict[str, str]:
    return {key: f"{index + offset:064x}" for index, key in enumerate(keys, 1)}


@pytest.fixture(scope="module")
def sealed_handoff(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    plan = build_long_horizon_train_plan(
        portfolio_families=("test_family_a", "test_family_b"),
        source_digests=_digests(SOURCE_BINDING_IDS, 0),
        corpus_digests=_digests(SOURCE_BINDING_IDS, 10),
        implementation_digests=_digests(IMPLEMENTATION_DIGEST_KEYS, 20),
    )
    schedule = build_long_horizon_stream_schedule(
        plan, worker_bindings=WORKERS
    )
    job = next(
        row
        for row in schedule["jobs"]
        if row["source_binding_id"] == M1_CORE5_BINDING_ID
        and row["month"] == "2020-01"
        and row["intrabar_path"] == "OHLC"
    )
    state = tmp_path_factory.mktemp("economic-state")
    initialize_long_horizon_execution_state(
        state,
        schedule=schedule,
        plan=plan,
        runner_binding={
            "runner_contract": "QR_TEST_ECONOMIC_RUNNER_V1",
            "runner_code_sha256": "d" * 64,
            "result_contract": CELL_CONTRACT,
        },
        resource_policy={
            "max_resident_coordinates": 160,
            "max_rss_bytes": 2_147_483_648,
            "max_open_files": 256,
            "min_free_disk_bytes": 1,
            "max_checkpoint_bytes": 8_388_608,
            "max_terminal_bytes": 2_097_152,
            "max_parallel_jobs": 1,
        },
    )
    session = LongHorizonExecutionSession(state, schedule=schedule, plan=plan)
    PLANS_BY_JOB[job["job_sha256"]] = plan
    return session.claim_job(job_sha256=job["job_sha256"], runner_id="economic-test")


def _source_row(epoch: int, bump: float) -> dict[str, Any]:
    openings = {
        "AUD_USD": 0.6500 + bump,
        "EUR_USD": 1.1000 + bump,
        "GBP_USD": 1.3000 + bump,
        "NZD_USD": 0.6100 + bump,
        "USD_JPY": 145.00 + bump,
    }
    quotes = []
    for pair, bid_open in sorted(openings.items()):
        spread = 0.02 if pair == "USD_JPY" else 0.0002
        span = 0.10 if pair == "USD_JPY" else 0.0010
        bid = [bid_open, bid_open + span, bid_open - span, bid_open + span / 4]
        ask = [value + spread for value in bid]
        quotes.append({"pair": pair, "bid": bid, "ask": ask})
    return {"complete": True, "epoch": epoch, "granularity": "M1", "quotes": quotes}


def _write_source(
    root: Path, job: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "2020-01.jsonl"
    rows = [_source_row(1_577_836_800, 0.0), _source_row(1_577_836_860, 0.0001)]
    path.write_bytes(
        b"".join(
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
            for row in rows
        )
    )
    raw_root = root / "raw"
    raw_root.mkdir()
    physical_rows = []
    coverage_rows = []
    for pair in job["feed_pairs"]:
        raw_path = raw_root / f"{pair}.jsonl.gz"
        with gzip.open(raw_path, "wt", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                quote = next(item for item in row["quotes"] if item["pair"] == pair)
                handle.write(
                    json.dumps(
                        {
                            "ask": dict(
                                zip(("o", "h", "l", "c"), quote["ask"], strict=True)
                            ),
                            "bid": dict(
                                zip(("o", "h", "l", "c"), quote["bid"], strict=True)
                            ),
                            "complete": True,
                            "granularity": "M1",
                            "pair": pair,
                            "price": "BA",
                            "time": datetime.fromtimestamp(
                                row["epoch"], timezone.utc
                            ).strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                            "volume": 1,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        raw_bytes = raw_path.read_bytes()
        physical_id = canonical_sha256({"pair": pair, "path": raw_path.name})
        physical_rows.append(
            {
                "physical_shard_id": physical_id,
                "root_kind": "M1",
                "relative_path": raw_path.name,
                "file_size_bytes": len(raw_bytes),
                "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "pair": pair,
                "granularity": "M1",
            }
        )
        coverage_body = {
            "pair": pair,
            "month": "2020-01",
            "physical_shard_id": physical_id,
            "row_count": len(rows),
            "first_observed_utc": "2020-01-01T00:00:00+00:00",
            "last_observed_utc": "2020-01-01T00:01:00+00:00",
            "request_window_completion_report_proved": True,
            "missing_slot_legitimacy_proved": True,
            "calendar_open_quote_coverage_proved": True,
        }
        coverage_rows.append(
            {
                **coverage_body,
                "coverage_cell_sha256": canonical_sha256(coverage_body),
            }
        )
    physical_ids = [row["physical_shard_id"] for row in physical_rows]
    binding = {
        "binding_id": job["source_binding_id"],
        "granularity": job["granularity"],
        "pairs": list(job["feed_pairs"]),
        "months": [job["month"]],
        "source_digest_sha256": job["source_digest_sha256"],
        "corpus_digest_sha256": job["corpus_digest_sha256"],
        "physical_shard_ids_sha256": canonical_sha256(physical_ids),
        "month_pair_coverage": coverage_rows,
    }
    manifest_body = {
        "contract": source_manifest_contract.CONTRACT,
        "schema_version": source_manifest_contract.SCHEMA_VERSION,
        "classification": source_manifest_contract.CLASSIFICATION,
        "source_roots": {"M1": str(raw_root), "M5": str(raw_root)},
        "study_period": {
            "from_utc": job["from_utc"],
            "to_utc": job["to_utc"],
            "half_open": True,
        },
        "selection_policy": "TEST_EXACT_PAIR_MONTH_SHARDS",
        "acquisition_identity": {
            "provider": "TEST_ONLY",
            "request_window_completion_report_proved": True,
        },
        "row_validation": {
            "strict_json": True,
            "complete_bid_ask_ohlc_only": True,
            "synthetic_rows_allowed": False,
        },
        "binding_count": 1,
        "bindings": [binding],
        "physical_shard_count": len(physical_rows),
        "physical_shards": physical_rows,
        "physical_shard_ids_sha256": canonical_sha256(physical_ids),
        "duplicate_equivalence_record_count": 0,
        "duplicate_equivalence_records": [],
        "duplicate_equivalence_records_sha256": canonical_sha256([]),
        "plan_digest_inputs": {
            "source_digests": {
                job["source_binding_id"]: job["source_digest_sha256"]
            },
            "corpus_digests": {
                job["source_binding_id"]: job["corpus_digest_sha256"]
            },
        },
        "summary_only_admission_allowed": False,
        "raw_rows_embedded": False,
        "authority": source_manifest_contract._authority(),
    }
    manifest = {
        **manifest_body,
        "source_manifest_sha256": canonical_sha256(manifest_body),
    }
    source_manifest_contract._validate_structure(manifest)
    receipt = build_month_source_slice_receipt(
        source_root=root,
        relative_path=path.name,
        job=job,
        source_manifest=manifest,
    )
    return receipt, manifest


def _policy(
    bindings: Sequence[Mapping[str, Any]],
    pairs: Sequence[str],
    tradable_pairs: Sequence[str],
    label: str,
) -> dict:
    return seal_portfolio_policy(
        {
            "policy_id": f"economic-test-{label}",
            "expected_quote_pairs": list(pairs),
            "tradable_pairs": list(tradable_pairs),
            "active_worker_bindings": list(bindings),
            "leverage": 20,
            "margin_closeout_fraction": 0.9,
            "max_margin_utilization_fraction": 0.45,
            "max_portfolio_stop_risk_fraction": 0.10,
            "max_open_and_pending_total": 8,
            "max_open_and_pending_per_pair": 2,
            "max_open_and_pending_per_family": 8,
            "max_currency_gross_notional_fraction": 10.0,
            "max_cluster_gross_notional_fraction": 10.0,
            "max_lock_seconds": 86_400,
            "slippage_by_pair": [
                {
                    "pair": pair,
                    "entry_slippage_price": 0.0001,
                    "exit_slippage_price": 0.0001,
                }
                for pair in pairs
            ],
            "financing_by_pair": [
                {
                    "pair": pair,
                    "long_cost_jpy_per_unit_day": 0.0,
                    "short_cost_jpy_per_unit_day": 0.0,
                }
                for pair in pairs
            ],
            "conversion_routes": [
                {
                    "currency": "USD",
                    "pair": "USD_JPY",
                    "orientation": "JPY_PER_CURRENCY",
                }
            ],
            "correlation_bindings": [],
        }
    )


def _runtime_rows(handoff: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    job = handoff["job"]
    digests = PLANS_BY_JOB[job["job_sha256"]]["implementation_binding"]["digests"]
    result = {}
    for coordinate in job["coordinates"]:
        trade_pairs = [
            pair
            for pair, bit in zip(
                job["feed_pairs"], coordinate["trade_pair_mask"], strict=True
            )
            if bit == "1"
        ]
        policy = _policy(
            CATALOG,
            job["feed_pairs"],
            trade_pairs,
            f"{coordinate['cost_scenario'].lower()}-{coordinate['coordinate_id'][:8]}",
        )
        cost_sha = digests[
            "base_cost_policy_sha256"
            if coordinate["cost_scenario"] == "BASE"
            else "stress_cost_policy_sha256"
        ]
        policy_binding = canonical_sha256(
            {
                "coordinate_id": coordinate["coordinate_id"],
                "cost_scenario": coordinate["cost_scenario"],
                "portfolio_policy_sha256": policy["policy_sha256"],
                "cost_policy_sha256": cost_sha,
                "risk_policy_sha256": digests["risk_policy_sha256"],
                "replay_engine_sha256": digests["replay_engine_sha256"],
                "allocator_policy": coordinate["allocator_policy"],
                "initial_balance_jpy": 200_000,
            }
        )
        result[coordinate["coordinate_id"]] = {
            "coordinate_id": coordinate["coordinate_id"],
            "cost_scenario": coordinate["cost_scenario"],
            "trade_pairs": trade_pairs,
            "portfolio_policy": policy,
            "cost_policy_sha256": cost_sha,
            "risk_policy_sha256": digests["risk_policy_sha256"],
            "replay_engine_sha256": digests["replay_engine_sha256"],
            "portfolio_policy_binding_sha256": policy_binding,
        }
    return result


class NoIntentRuntime:
    def __init__(
        self,
        bindings: Sequence[Mapping[str, Any]],
        prior_state: Mapping[str, Any] | None,
    ) -> None:
        self.bindings = [dict(row) for row in bindings]
        self.calls = int((prior_state or {}).get("calls", 0))

    def propose(self, snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
        self.calls += 1
        return [
            {
                **binding,
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "risk_reducing_intents": [],
                "new_risk_intents": [],
            }
            for binding in self.bindings
        ]

    def export_state(self) -> dict[str, int]:
        return {"calls": self.calls}


def _no_intent_factory(
    _coordinate: Mapping[str, Any],
    bindings: Sequence[Mapping[str, Any]],
    prior_state: Mapping[str, Any] | None,
) -> NoIntentRuntime:
    return NoIntentRuntime(bindings, prior_state)


def test_single_source_stream_fans_out_before_incremental_economics(
    tmp_path: Path, sealed_handoff: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, source_manifest = _write_source(tmp_path, sealed_handoff["job"])
    import quant_rabbit.dojo_long_horizon_economic_runner as runner

    real_open = runner.os.open
    source_path = (tmp_path / receipt["relative_path"]).resolve()
    source_open_count = 0

    def counted_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal source_open_count
        if Path(path).resolve() == source_path:
            source_open_count += 1
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(runner.os, "open", counted_open)
    result = run_long_horizon_economic_job(
        runner_handoff=sealed_handoff,
        plan=PLANS_BY_JOB[sealed_handoff["job"]["job_sha256"]],
        source_root=tmp_path,
        source_manifest=source_manifest,
        source_slice_receipt=receipt,
        worker_catalog=CATALOG,
        coordinate_runtimes=_runtime_rows(sealed_handoff),
        worker_runtime_factory=builtin_no_intent_runtime_factory,
        worker_runtime_binding_sha256=RUNTIME_SHA,
    )

    assert source_open_count == result["source_open_count"] == 1
    assert result["source_row_count"] == 2
    assert result["quote_batch_count"] == 8
    assert result["coordinate_result_count"] == 20
    assert result["complete_coordinate_count"] == 20
    assert result["failed_coordinate_count"] == 0
    assert len(result["portfolio_results_by_coordinate"]) == 20
    assert len(result["economic_carry_states_by_slot"]) == 10
    assert all(row["status"] == "COMPLETE" for row in result["coordinate_results"])
    assert all(row["trade_count"] == 0 for row in result["coordinate_results"])
    assert all(row["fill_count"] == 0 for row in result["coordinate_results"])
    assert all(
        replay["processed_coordinate_count"] == 8
        for replay in result["portfolio_results_by_coordinate"].values()
    )
    assert result["partial_economics_reported"] is False
    assert result["authority"]["live_permission"] is False


def test_source_byte_drift_invalidates_every_coordinate(
    tmp_path: Path, sealed_handoff: dict[str, Any]
) -> None:
    receipt, source_manifest = _write_source(tmp_path, sealed_handoff["job"])
    path = tmp_path / receipt["relative_path"]
    raw = path.read_bytes()
    assert b"145.02" in raw
    path.write_bytes(raw.replace(b"145.02", b"145.03", 1))

    result = run_long_horizon_economic_job(
        runner_handoff=sealed_handoff,
        plan=PLANS_BY_JOB[sealed_handoff["job"]["job_sha256"]],
        source_root=tmp_path,
        source_manifest=source_manifest,
        source_slice_receipt=receipt,
        worker_catalog=CATALOG,
        coordinate_runtimes=_runtime_rows(sealed_handoff),
        worker_runtime_factory=builtin_no_intent_runtime_factory,
        worker_runtime_binding_sha256=RUNTIME_SHA,
    )

    assert result["complete_coordinate_count"] == 0
    assert result["failed_coordinate_count"] == 20
    assert {
        row["failure"]["code"] for row in result["coordinate_results"]
    } == {"SOURCE_STREAM_FAILURE"}
    assert result["portfolio_results_by_coordinate"] == {}
    assert result["partial_economics_reported"] is False


def test_source_receipt_is_job_and_hash_bound(
    tmp_path: Path, sealed_handoff: dict[str, Any]
) -> None:
    receipt, source_manifest = _write_source(tmp_path, sealed_handoff["job"])
    assert (
        validate_month_source_slice_receipt(
            receipt,
            job=sealed_handoff["job"],
            source_manifest=source_manifest,
        )
        == receipt
    )
    tampered = copy.deepcopy(receipt)
    tampered["file_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="seal is invalid"):
        validate_month_source_slice_receipt(
            tampered,
            job=sealed_handoff["job"],
            source_manifest=source_manifest,
        )


def test_unproved_calendar_gap_coverage_fails_every_cell(
    tmp_path: Path, sealed_handoff: dict[str, Any]
) -> None:
    _receipt, source_manifest = _write_source(tmp_path, sealed_handoff["job"])
    coverage_rows = source_manifest["bindings"][0]["month_pair_coverage"]
    for index, row in enumerate(coverage_rows):
        body = {
            **{key: value for key, value in row.items() if key != "coverage_cell_sha256"},
            "missing_slot_legitimacy_proved": False,
            "calendar_open_quote_coverage_proved": False,
        }
        coverage_rows[index] = {
            **body,
            "coverage_cell_sha256": canonical_sha256(body),
        }
    manifest_body = {
        key: value
        for key, value in source_manifest.items()
        if key != "source_manifest_sha256"
    }
    source_manifest = {
        **manifest_body,
        "source_manifest_sha256": canonical_sha256(manifest_body),
    }
    receipt = build_month_source_slice_receipt(
        source_root=tmp_path,
        relative_path="2020-01.jsonl",
        job=sealed_handoff["job"],
        source_manifest=source_manifest,
    )
    result = run_long_horizon_economic_job(
        runner_handoff=sealed_handoff,
        plan=PLANS_BY_JOB[sealed_handoff["job"]["job_sha256"]],
        source_root=tmp_path,
        source_manifest=source_manifest,
        source_slice_receipt=receipt,
        worker_catalog=CATALOG,
        coordinate_runtimes=_runtime_rows(sealed_handoff),
        worker_runtime_factory=builtin_no_intent_runtime_factory,
        worker_runtime_binding_sha256=RUNTIME_SHA,
    )
    assert result["job_status"] == "INCOMPLETE_FAILED"
    assert result["complete_coordinate_count"] == 0
    assert result["portfolio_results_by_coordinate"] == {}
    assert result["partial_economics_reported"] is False
    assert {
        row["failure"]["code"] for row in result["coordinate_results"]
    } == {"SOURCE_QUOTE_COVERAGE_UNPROVEN"}


def test_external_python_worker_is_rejected_before_factory_side_effect(
    tmp_path: Path, sealed_handoff: dict[str, Any]
) -> None:
    receipt, source_manifest = _write_source(tmp_path, sealed_handoff["job"])
    called = False

    def side_effectful_factory(*_args: Any) -> NoIntentRuntime:
        nonlocal called
        called = True
        raise AssertionError("must not execute")

    with pytest.raises(ValueError, match="external in-process worker code is forbidden"):
        run_long_horizon_economic_job(
            runner_handoff=sealed_handoff,
            plan=PLANS_BY_JOB[sealed_handoff["job"]["job_sha256"]],
            source_root=tmp_path,
            source_manifest=source_manifest,
            source_slice_receipt=receipt,
            worker_catalog=CATALOG,
            coordinate_runtimes=_runtime_rows(sealed_handoff),
            worker_runtime_factory=side_effectful_factory,
            worker_runtime_binding_sha256="f" * 64,
        )
    assert called is False

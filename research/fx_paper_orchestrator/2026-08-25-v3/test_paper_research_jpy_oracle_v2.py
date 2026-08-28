from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from fractions import Fraction
from pathlib import Path

import pytest

import paper_research_jpy_oracle_v2 as oracle
from paper_research_jpy_oracle_golden_v2 import build_golden_payload


START_NS = 1_767_225_600_000_000_000  # 2026-01-01 00:00:00 UTC
PROVIDER = "ORACLE_FIXTURE"


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal(value: dict, field: str) -> dict:
    value[field] = oracle.embedded_hash(value, field)
    return value


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")
    return path


def artifact(root: Path, path: Path, label: str) -> dict:
    data = path.read_bytes()
    return {
        "artifact_id": label,
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def registry_payload() -> dict:
    return seal({
        "schema_version": 1,
        "registry_id": "FROZEN_FX_INSTRUMENT_REGISTRY_V1",
        "instruments": {
            "EUR_USD": {"pip_ticks": 10, "price_scale": 100_000},
            "USD_JPY": {"pip_ticks": 1, "price_scale": 100},
        },
    }, "registry_sha256")


def source_rows(*, future_delta: int = 0) -> list[dict]:
    offsets = (0, 1, 2, 301, 302, 360, 361, 662, 900)
    rows: list[dict] = []
    for seconds in offsets:
        for instrument, arrival_offset in (("EUR_USD", 100_000_000), ("USD_JPY", 200_000_000)):
            sequence = offsets.index(seconds) + 1
            if instrument == "EUR_USD":
                bid = 110_000 + sequence * 8
                ask = bid + 12
                scale = 100_000
            else:
                bid = 15_000 + sequence * 3
                ask = bid + 2
                scale = 100
            if seconds >= 662:
                bid += future_delta
                ask += future_delta
            source = START_NS + seconds * 1_000_000_000
            rows.append({
                "schema_version": 1,
                "provider_id": PROVIDER,
                "instrument": instrument,
                "bid_ticks": bid,
                "ask_ticks": ask,
                "tick_scale": scale,
                "source_ts_ns": source,
                "arrival_ts_ns": source + arrival_offset,
                "provider_event_id": f"{instrument}-{sequence}",
                "sequence": sequence,
                "heartbeat": False,
                "quality_flags": [],
            })
    return sorted(rows, key=lambda row: (
        row["arrival_ts_ns"], row["source_ts_ns"], row["provider_id"],
        row["instrument"], row["sequence"],
    ))


def write_source(root: Path, rows: list[dict], registry: dict) -> tuple[Path, Path, list[dict]]:
    blob = root / "inputs" / "source.jsonl"
    blob.parent.mkdir(parents=True, exist_ok=True)
    lines = [canonical(row) + b"\n" for row in rows]
    blob_bytes = b"".join(lines)
    blob.write_bytes(blob_bytes)
    enriched: list[dict] = []
    prefix = "0" * 64
    for row, line in zip(rows, lines):
        event_hash = hashlib.sha256(line).hexdigest()
        prefix = hashlib.sha256(canonical({
            "previous_hash": prefix,
            "source_event_sha256": event_hash,
        })).hexdigest()
        enriched.append({
            **row,
            "source_event_sha256": event_hash,
            "source_prefix_root_sha256": prefix,
        })
    policies = []
    stream_keys = sorted({
        (row["provider_id"], row["instrument"])
        for row in rows
    })
    for provider_id, instrument in stream_keys:
        stream = [
            row for row in rows
            if (row["provider_id"], row["instrument"])
            == (provider_id, instrument)
        ]
        policies.append({
            "provider_id": provider_id,
            "instrument": instrument,
            "sequence_required": True,
            "first_sequence": stream[0]["sequence"],
            "last_sequence": stream[-1]["sequence"],
            "event_count": len(stream),
            "max_source_gap_ns": 400_000_000_000,
            "max_arrival_gap_ns": 400_000_000_000,
        })
    manifest = seal({
        "schema_version": 2,
        "source_bytes_sha256": hashlib.sha256(blob_bytes).hexdigest(),
        "source_size_bytes": len(blob_bytes),
        "event_count": len(rows),
        "first_source_ts_ns": min(row["source_ts_ns"] for row in rows),
        "last_source_ts_ns": max(row["source_ts_ns"] for row in rows),
        "provider_allowlist": sorted({row["provider_id"] for row in rows}),
        "instrument_registry_sha256": registry["registry_sha256"],
        "stream_policies": policies,
        "lossless": True,
    }, "manifest_sha256")
    return blob, write_json(root / "inputs" / "source_manifest.json", manifest), enriched


def proposal_row(enriched: list[dict], *, ordinal: int, event_index: int, direction: int, max_age_seconds: int = 300) -> dict:
    event = enriched[event_index]
    available = [row for row in enriched if row["arrival_ts_ns"] <= event["arrival_ts_ns"]]
    return {
        "proposal_ordinal": ordinal,
        "decision_source_ts_ns": event["source_ts_ns"],
        "decision_arrival_ts_ns": event["arrival_ts_ns"],
        "available_at_ns": event["arrival_ts_ns"],
        "decision_source_event_sha256": event["source_event_sha256"],
        "completed_data_watermark_source_ts_ns": max(row["source_ts_ns"] for row in available),
        "completed_data_prefix_root_sha256": available[-1]["source_prefix_root_sha256"],
        "instrument": event["instrument"],
        "direction": direction,
        "notional_jpy_micros": 28_000 * 1_000_000,
        "max_age_ns": max_age_seconds * 1_000_000_000,
        "worker_key": "FIXED_DETECTOR",
        "action": "ENTER",
    }


def fixture(
    root: Path,
    *,
    rows: list[dict] | None = None,
    proposal_specs: list[tuple[int, int]] | None = None,
    future_delta: int = 0,
    registry_mutation: callable | None = None,
    execution_mutation: callable | None = None,
    inventory_mutation: callable | None = None,
    evaluation_mutation: callable | None = None,
    proposal_max_age_seconds: int = 300,
) -> tuple[dict, dict]:
    root.mkdir(parents=True, exist_ok=True)
    oracle_output_root(root)
    registry = registry_payload()
    if registry_mutation is not None:
        registry.pop("registry_sha256")
        registry_mutation(registry)
        registry["instruments"] = dict(sorted(registry["instruments"].items()))
        seal(registry, "registry_sha256")
    registry_path = write_json(root / "inputs" / "instrument_registry.json", registry)
    source = source_rows(future_delta=future_delta) if rows is None else rows
    blob, source_manifest_path, enriched = write_source(root, source, registry)
    if proposal_specs is None:
        # EUR seq1 and USDJPY seq6 decisions.
        proposal_specs = [(0, 1), (11, -1)]
    proposal = seal({
        "schema_version": 2,
        "candidate_key": "ORACLE-V2-FIXTURE",
        "provenance": {
            "detector_code_sha256": "1" * 64,
            "detector_policy_sha256": "2" * 64,
            "generator_policy_sha256": "3" * 64,
            "source_acquisition_contract_sha256": "4" * 64,
        },
        "rows": [
            proposal_row(
                enriched,
                ordinal=index,
                event_index=event_index,
                direction=direction,
                max_age_seconds=proposal_max_age_seconds,
            )
            for index, (event_index, direction) in enumerate(proposal_specs, 1)
        ],
    }, "proposal_sha256")
    execution = {
        "schema_version": 2,
        "policy_id": "FROZEN_EXECUTION_POLICY_V2",
        "arms": {
            "RAW_SIGNAL": {
                "latency_ns": 0,
                "slippage_micropips_per_side": 0,
                "commission_ppm_per_side": 0,
                "financing_ppm_per_day": 0,
                "raw_mid": True,
            },
            "EXECUTABLE_BASE": {
                "latency_ns": 500_000_000,
                "slippage_micropips_per_side": 100_000,
                "commission_ppm_per_side": 2,
                "financing_ppm_per_day": 1,
                "raw_mid": False,
            },
            "ADVERSE_STRESS": {
                "latency_ns": 1_500_000_000,
                "slippage_micropips_per_side": 300_000,
                "commission_ppm_per_side": 6,
                "financing_ppm_per_day": 3,
                "raw_mid": False,
            },
        },
        "max_trade_quote_staleness_ns": 400_000_000_000,
    }
    if execution_mutation is not None:
        execution_mutation(execution)
    seal(execution, "execution_policy_sha256")
    inventory = seal({
        "schema_version": 2,
        "policy_id": "FROZEN_INVENTORY_POLICY_V2",
        "max_gross_notional_jpy_micros": 200_000 * 1_000_000,
        "max_currency_notional_jpy_micros": 200_000 * 1_000_000,
        "max_open_positions": 4,
        "same_pair_collision": "REJECT_NEW",
        "terminal_liquidation": True,
    }, "inventory_policy_sha256")
    if inventory_mutation is not None:
        inventory.pop("inventory_policy_sha256")
        inventory_mutation(inventory)
        seal(inventory, "inventory_policy_sha256")
    accounting = seal({
        "schema_version": 2,
        "policy_id": "FROZEN_ACCOUNTING_POLICY_V2",
        "jpy_micros_per_yen": 1_000_000,
        "base_microunits_per_unit": 1_000_000,
        "max_conversion_staleness_ns": 400_000_000_000,
        "supported_quote_currencies": ["CAD", "CHF", "JPY", "USD"],
        "asset_conversion_side": "BID",
        "liability_conversion_side": "ASK",
        "positive_cost_rounding": "CEILING",
    }, "accounting_policy_sha256")
    evaluation = {
        "schema_version": 2,
        "policy_id": "FROZEN_EVALUATION_POLICY_V2",
        "period_start_ts_ns": START_NS,
        "period_end_ts_ns": START_NS + 901_000_000_000,
        "initial_equity_jpy_micros": 200_000 * 1_000_000,
        "margin_notional_cap_jpy_micros": 200_000 * 1_000_000,
        "margin_rate_bps": 500,
        "max_gross_to_equity_bps": 20_000,
        "cvar_tail_bps": 500,
        "cluster_window_ns": 3_600_000_000_000,
        "full_month_ids": [],
        "holdout_state": "UNOPENED",
    }
    if evaluation_mutation is not None:
        evaluation_mutation(evaluation)
    seal(evaluation, "evaluation_policy_sha256")
    authority = seal({
        "schema_version": 2,
        "policy_id": "FROZEN_PAPER_AUTHORITY_V1",
        **oracle.AUTHORITY,
    }, "authority_policy_sha256")
    values = {
        "proposal": proposal,
        "execution_policy": execution,
        "inventory_policy": inventory,
        "accounting_policy": accounting,
        "evaluation_policy": evaluation,
        "authority_policy": authority,
    }
    paths = {
        key: write_json(root / "inputs" / f"{key}.json", value)
        for key, value in values.items()
    }
    request = {
        "schema_version": 2,
        "source_blob": artifact(root, blob, "source_blob"),
        "source_manifest": artifact(root, source_manifest_path, "source_manifest"),
        "proposal": artifact(root, paths["proposal"], "proposal"),
        "execution_policy": artifact(root, paths["execution_policy"], "execution_policy"),
        "inventory_policy": artifact(root, paths["inventory_policy"], "inventory_policy"),
        "accounting_policy": artifact(root, paths["accounting_policy"], "accounting_policy"),
        "evaluation_policy": artifact(root, paths["evaluation_policy"], "evaluation_policy"),
        "instrument_registry": artifact(root, registry_path, "instrument_registry"),
        "authority_policy": artifact(root, paths["authority_policy"], "authority_policy"),
        "output_directory": "oracle_output",
    }
    return request, {**values, "registry": registry, "source_rows": enriched}


def oracle_output_root(root: Path) -> Path:
    result = root / "oracle-publish"
    result.mkdir(mode=0o700, exist_ok=True)
    result.chmod(0o700)
    return result


def run(root: Path, request: dict) -> dict:
    return oracle.execute(
        request,
        trusted_input_root=root,
        trusted_output_root=oracle_output_root(root),
    )


def exclusive_rename_at(root_fd: int, source: str, destination: str) -> None:
    if oracle._lstat_at(root_fd, destination) is not None:
        raise FileExistsError(destination)
    os.rename(source, destination, src_dir_fd=root_fd, dst_dir_fd=root_fd)


def leave_recoverable_oracle_stage(
    root: Path,
    request: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    original = oracle._write_file_at

    def fail_before_manifest(
        directory_fd: int, name: str, data: bytes, mode: int = 0o600
    ) -> None:
        if name == "oracle_manifest.json":
            raise OSError("injected recoverable manifest fault")
        original(directory_fd, name, data, mode)

    monkeypatch.setattr(oracle, "_write_file_at", fail_before_manifest)
    with pytest.raises(OSError, match="recoverable manifest fault"):
        run(root, request)
    monkeypatch.setattr(oracle, "_write_file_at", original)
    stages = list(oracle_output_root(root).glob(".oracle_output.*.stage"))
    assert len(stages) == 1
    return stages[0]


def ledger(root: Path, output: str = "oracle_output") -> list[dict]:
    return [
        json.loads(line)
        for line in (
            oracle_output_root(root) / output / "oracle_ledger.jsonl"
        ).read_text().splitlines()
    ]


def golden_request(root: Path) -> tuple[dict, dict]:
    payload = build_golden_payload()
    inputs = payload["inputs"]
    input_dir = root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_path = input_dir / "source_blob.jsonl"
    source_path.write_bytes(inputs["source_blob_utf8"].encode("utf-8"))
    paths = {
        label: write_json(input_dir / f"{label}.json", inputs[label])
        for label in (
            "source_manifest",
            "proposal",
            "execution_policy",
            "inventory_policy",
            "accounting_policy",
            "evaluation_policy",
            "instrument_registry",
            "authority_policy",
        )
    }
    request = {
        "schema_version": 2,
        "source_blob": artifact(root, source_path, "source_blob"),
        **{label: artifact(root, path, label) for label, path in paths.items()},
        "output_directory": "oracle_output",
    }
    return request, payload


def test_accounting_only_oracle_replays_all_arms_with_exact_chain(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    result = run(tmp_path, request)
    manifest = result["manifest"]
    assert manifest["classification"] == oracle.CLASSIFICATION
    assert manifest["causal_signal_admission"] is False
    assert manifest["release_evidence_eligible"] is False
    assert manifest["authority"] == oracle.AUTHORITY
    assert manifest["oracle_ledger_row_count"] == 6
    assert manifest["producer_result_or_metrics_used"] is False
    records = ledger(tmp_path)
    assert records[0]["previous_hash"] == "0" * 64
    assert records[-1]["record_hash"] == manifest["oracle_ledger_terminal_hash"]
    assert hashlib.sha256(
        (
            oracle_output_root(tmp_path)
            / "oracle_output"
            / "oracle_ledger.jsonl"
        ).read_bytes()
    ).hexdigest() == manifest["oracle_ledger_sha256"]
    assert all(record["external_order_count"] == 0 for record in records)
    assert all(manifest["oracle_metrics"]["arms"][arm]["terminal_open_positions"] == 0 for arm in oracle.ARMS)
    assert manifest["oracle_metrics"]["same_signal_ids_all_arms"] is True
    assert manifest["oracle_metrics"]["common_gross_reference_shared"] is True
    assert (
        oracle_output_root(tmp_path) / "oracle_output" / "COMMIT.json"
    ).is_file()


def test_hand_derived_golden_ledger_and_metrics_match_exactly(tmp_path: Path) -> None:
    request, golden = golden_request(tmp_path)
    result = run(tmp_path, request)
    expected = golden["expected"]
    actual_ledger = (
        oracle_output_root(tmp_path) / "oracle_output" / "oracle_ledger.jsonl"
    ).read_bytes()
    assert actual_ledger == expected["ledger_utf8"].encode("utf-8")
    assert hashlib.sha256(actual_ledger).hexdigest() == (
        "fe8520a5b77a37c6cc2b5f22db109fb9094253a714a8ee578831fa46a03e8145"
    )
    assert len(actual_ledger) == 9_100
    assert [row["record_hash"] for row in ledger(tmp_path)] == [
        "aff4f6c0bef1d1c9180b537af6542eff740c3d71393dd54373594cec208f2924",
        "1b4b05f847eac8e30002d0810323f85e4e1c60e90f6d6b5f312224b7451111f1",
        "16f8185488f2744abcacbcce1e155b10450f46d852962d92f20db0749e41f858",
    ]
    assert result["manifest"]["oracle_metrics"] == expected["oracle_metrics"]
    assert expected["oracle_metrics"]["metrics_sha256"] == (
        "2f3b49b229949adfe4c1ee12fc1f77f2cd8467a19c77ea5065ae5151a92c47f0"
    )


def test_same_pair_collision_is_enforced_at_fill_arrival(tmp_path: Path) -> None:
    # EUR seq1 decision enters on seq2; EUR seq2 decision attempts seq3 while open.
    request, _ = fixture(tmp_path, proposal_specs=[(0, 1), (2, -1)])
    run(tmp_path, request)
    records = ledger(tmp_path)
    for arm in oracle.ARMS:
        arm_rows = [row for row in records if row["arm"] == arm]
        assert arm_rows[0]["status"] == "FILLED_CLOSED"
        assert arm_rows[1]["status"] == "SAME_PAIR_COLLISION_REJECTED"


def test_signed_currency_exposure_values_base_and_quote_nodes_independently(
    tmp_path: Path,
) -> None:
    request, _ = fixture(tmp_path)
    run(tmp_path, request)
    filled = next(row for row in ledger(tmp_path) if row["arm"] == "RAW_SIGNAL" and row["proposal_ordinal"] == 1)
    exposure = filled["signed_currency_exposure_after_entry_jpy_micros"]
    marked_notional = filled["gross_open_notional_after_entry_jpy_micros"]
    units = filled["units_micros"]
    eur_asset_exact = (
        Fraction(units, 1_000_000)
        * Fraction(110_016, 100_000)
        * Fraction(15_003, 100)
        * 1_000_000
    )
    usd_liability_exact = -(
        Fraction(units, 1_000_000)
        * Fraction(110_022, 100_000)
        * Fraction(15_005, 100)
        * 1_000_000
    )
    assert exposure == {
        "EUR": (
            eur_asset_exact.numerator + eur_asset_exact.denominator - 1
        ) // eur_asset_exact.denominator,
        "USD": usd_liability_exact.numerator // usd_liability_exact.denominator,
    }
    assert exposure["EUR"] != -exposure["USD"]
    assert max(abs(value) for value in exposure.values()) != marked_notional
    assert 0 < filled["filled_notional_jpy_micros"] <= filled[
        "target_notional_jpy_micros"
    ]
    assert marked_notional != filled["filled_notional_jpy_micros"]


def test_elapsed_and_financing_use_fill_arrival_clock(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    run(tmp_path, request)
    arm_rows = {
        row["arm"]: row
        for row in ledger(tmp_path)
        if row["proposal_ordinal"] == 1
    }
    for filled in arm_rows.values():
        assert filled["elapsed_ns"] == (
            filled["exit_arrival_ts_ns"] - filled["entry_arrival_ts_ns"]
        )
        assert filled["elapsed_ns"] == 300_000_000_000
    assert arm_rows["EXECUTABLE_BASE"]["financing_jpy_micros"] > 0


def test_same_arrival_exit_releases_inventory_before_new_entry(tmp_path: Path) -> None:
    # Proposal 1 enters EUR seq2 and is due exactly at seq4. Proposal 2 is
    # decided at seq3 and enters on that same seq4 arrival. Every arm uses the
    # same latency here; ADVERSE remains strictly worse through direct costs.
    def zero_latency(execution: dict) -> None:
        execution["arms"]["EXECUTABLE_BASE"]["latency_ns"] = 0
        execution["arms"]["ADVERSE_STRESS"]["latency_ns"] = 0

    request, _ = fixture(
        tmp_path,
        proposal_specs=[(0, 1), (4, -1)],
        execution_mutation=zero_latency,
    )
    run(tmp_path, request)
    rows = ledger(tmp_path)
    for arm in oracle.ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        assert [row["status"] for row in arm_rows] == ["FILLED_CLOSED", "FILLED_CLOSED"]
        assert arm_rows[0]["exit_arrival_ts_ns"] == arm_rows[1]["entry_arrival_ts_ns"]


def test_quote_staleness_boundary_is_inclusive_then_fails_closed() -> None:
    event = {
        "source_ts_ns": 100,
        "arrival_ts_ns": 100,
        "bid_ticks": 100,
        "ask_ticks": 102,
        "tick_scale": 100,
    }
    books = {"USD_JPY": [event]}
    assert oracle._latest_causal(books, "USD_JPY", 200, 200, 100) is event
    with pytest.raises(oracle.OracleError, match="stale causal BBO"):
        oracle._latest_causal(books, "USD_JPY", 201, 200, 100)
    with pytest.raises(oracle.OracleError, match="stale causal BBO"):
        oracle._latest_causal(books, "USD_JPY", 200, 201, 100)


def test_quote_transit_age_is_part_of_staleness() -> None:
    delayed = {
        "source_ts_ns": 100,
        "arrival_ts_ns": 1_000,
        "bid_ticks": 100,
        "ask_ticks": 102,
        "tick_scale": 100,
    }
    with pytest.raises(oracle.OracleError, match="stale causal BBO"):
        oracle._latest_causal({"USD_JPY": [delayed]}, "USD_JPY", 100, 1_000, 100)


def test_sign_aware_conversion_and_negative_flooring_vectors() -> None:
    usd_jpy = {
        "source_ts_ns": 100, "arrival_ts_ns": 100,
        "bid_ticks": 100, "ask_ticks": 101, "tick_scale": 1,
    }
    usd_cad = {
        "source_ts_ns": 100, "arrival_ts_ns": 100,
        "bid_ticks": 125, "ask_ticks": 126, "tick_scale": 100,
    }
    books = {"USD_JPY": [usd_jpy], "USD_CAD": [usd_cad]}
    registry = {
        "USD_CAD": {"pip_ticks": 1, "price_scale": 100},
        "USD_JPY": {"pip_ticks": 1, "price_scale": 1},
    }
    assert oracle._convert_to_jpy(
        oracle.Fraction(1), "USD", 100, 100, books, 1,
        registry=registry,
    ) == 100
    assert oracle._convert_to_jpy(
        oracle.Fraction(-1), "USD", 100, 100, books, 1,
        registry=registry,
    ) == -101
    assert oracle._convert_to_jpy(
        oracle.Fraction(126), "CAD", 100, 100, books, 1,
        registry=registry,
    ) == 10_000
    assert oracle._convert_to_jpy(
        oracle.Fraction(-125), "CAD", 100, 100, books, 1,
        registry=registry,
    ) == -10_100
    assert oracle._asset_micros(oracle.Fraction(-1, 3)) == -333_334


@pytest.mark.parametrize("direction", [1, -1])
def test_e2e_no_causal_fill_drag_reconciles_for_both_gross_signs(
    tmp_path: Path, direction: int
) -> None:
    def no_executable_fill(execution: dict) -> None:
        execution["arms"]["EXECUTABLE_BASE"]["latency_ns"] = 10_000_000_000_000
        execution["arms"]["ADVERSE_STRESS"]["latency_ns"] = 10_000_000_000_001

    request, _ = fixture(
        tmp_path,
        proposal_specs=[(0, direction)],
        execution_mutation=no_executable_fill,
    )
    run(tmp_path, request)
    for row in ledger(tmp_path):
        if row["arm"] == "RAW_SIGNAL":
            assert row["status"] == "FILLED_CLOSED"
            continue
        assert row["status"] == "NO_CAUSAL_FILL"
        assert row["latency_spread_slippage_drag_jpy_micros"] == row["common_gross_pnl_jpy_micros"]
        assert row["admission_opportunity_drag_jpy_micros"] == 0
        assert row["common_gross_pnl_jpy_micros"] - row["net_pnl_jpy_micros"] == (
            row["latency_spread_slippage_drag_jpy_micros"]
        )


def test_multiple_providers_for_same_instrument_fail_closed() -> None:
    registry = registry_payload()
    events = []
    for index, provider in enumerate(("A", "B"), 1):
        events.append({
            "schema_version": 1,
            "provider_id": provider,
            "instrument": "USD_JPY",
            "bid_ticks": 15_000 + index,
            "ask_ticks": 15_002 + index,
            "tick_scale": 100,
            "source_ts_ns": START_NS + index,
            "arrival_ts_ns": START_NS + index,
            "provider_event_id": f"{provider}-1",
            "sequence": 1,
            "heartbeat": False,
            "quality_flags": [],
        })
    blob = b"".join(canonical(event) + b"\n" for event in events)
    manifest = seal({
        "schema_version": 2,
        "source_bytes_sha256": hashlib.sha256(blob).hexdigest(),
        "source_size_bytes": len(blob),
        "event_count": 2,
        "first_source_ts_ns": START_NS + 1,
        "last_source_ts_ns": START_NS + 2,
        "provider_allowlist": ["A", "B"],
        "instrument_registry_sha256": registry["registry_sha256"],
        "stream_policies": [
            {
                "provider_id": provider,
                "instrument": "USD_JPY",
                "sequence_required": True,
                "first_sequence": 1,
                "last_sequence": 1,
                "event_count": 1,
                "max_source_gap_ns": 1,
                "max_arrival_gap_ns": 1,
            }
            for provider in ("A", "B")
        ],
        "lossless": True,
    }, "manifest_sha256")
    with pytest.raises(oracle.OracleError, match="multiple providers"):
        oracle._parse_source(blob, manifest, registry, oracle._validate_instrument_registry(registry))


def test_duplicate_provider_event_identity_in_one_stream_fails_closed(
    tmp_path: Path,
) -> None:
    rows = source_rows()
    same_stream = [row for row in rows if row["instrument"] == "EUR_USD"]
    same_stream[1]["provider_event_id"] = same_stream[0]["provider_event_id"]
    request, _ = fixture(tmp_path, rows=rows)
    with pytest.raises(oracle.OracleError, match="duplicate provider event identity"):
        run(tmp_path, request)


def test_provider_event_id_scope_is_provider_and_instrument(
    tmp_path: Path,
) -> None:
    rows = source_rows()
    shared_id = "SHARED-ACROSS-DISTINCT-STREAMS"
    next(row for row in rows if row["instrument"] == "EUR_USD")[
        "provider_event_id"
    ] = shared_id
    next(row for row in rows if row["instrument"] == "USD_JPY")[
        "provider_event_id"
    ] = shared_id
    request, _ = fixture(tmp_path, rows=rows)
    assert run(tmp_path, request)["manifest"]["oracle_ledger_row_count"] == 6


def test_provider_event_id_scope_distinguishes_provider_and_instrument(
    tmp_path: Path,
) -> None:
    rows = source_rows()
    shared_id = "SHARED-ACROSS-PROVIDER-AND-INSTRUMENT"
    next(row for row in rows if row["instrument"] == "EUR_USD")[
        "provider_event_id"
    ] = shared_id
    for row in rows:
        if row["instrument"] == "USD_JPY":
            row["provider_id"] = "SECOND_PROVIDER"
    next(row for row in rows if row["instrument"] == "USD_JPY")[
        "provider_event_id"
    ] = shared_id
    rows.sort(key=lambda row: (
        row["arrival_ts_ns"], row["source_ts_ns"], row["provider_id"],
        row["instrument"], row["sequence"],
    ))
    request, _ = fixture(tmp_path, rows=rows)
    assert run(tmp_path, request)["manifest"]["oracle_ledger_row_count"] == 6


def test_repeated_null_provider_event_ids_are_permitted(tmp_path: Path) -> None:
    rows = source_rows()
    for row in rows:
        row["provider_event_id"] = None
    request, _ = fixture(tmp_path, rows=rows)
    assert run(tmp_path, request)["manifest"]["oracle_ledger_row_count"] == 6


def test_missing_provider_event_id_field_fails_exact_source_schema(
    tmp_path: Path,
) -> None:
    rows = source_rows()
    rows[0].pop("provider_event_id")
    request, _ = fixture(tmp_path, rows=rows)
    with pytest.raises(oracle.OracleError, match="source BBO record schema mismatch"):
        run(tmp_path, request)


def test_tiny_notional_gets_all_arm_dispositions_instead_of_aborting(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    proposal_path = tmp_path / request["proposal"]["relative_path"]
    proposal = json.loads(proposal_path.read_text())
    proposal["rows"][0]["notional_jpy_micros"] = 1
    proposal.pop("proposal_sha256")
    seal(proposal, "proposal_sha256")
    write_json(proposal_path, proposal)
    request["proposal"] = artifact(tmp_path, proposal_path, "proposal")
    result = run(tmp_path, request)
    tiny = [row for row in ledger(tmp_path) if row["proposal_ordinal"] == 1]
    assert len(tiny) == 3
    assert {row["arm"] for row in tiny} == set(oracle.ARMS)
    assert {row["status"] for row in tiny} == {"SIZE_ROUNDED_TO_ZERO"}
    assert result["manifest"]["oracle_metrics"]["all_proposals_have_all_arm_dispositions"] is True


def test_positive_fractional_costs_round_up(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    run(tmp_path, request)
    filled = next(row for row in ledger(tmp_path) if row["arm"] == "EXECUTABLE_BASE" and row["proposal_ordinal"] == 1)
    # Each side uses its own causal executable notional and is ceiled
    # independently; the target notional is never reused as a cost basis.
    assert filled["commission_jpy_micros"] == 112_016
    assert filled["financing_basis_notional_jpy_micros"] == filled[
        "filled_notional_jpy_micros"
    ]
    # A fractional positive financing debit must never round down to zero.
    assert oracle._positive_cost_micros(oracle.Fraction(1, 10_000)) == 1


def test_fractional_commission_is_ceiled_independently_per_side() -> None:
    event = {
        "bid_ticks": 10_000,
        "ask_ticks": 10_002,
        "tick_scale": 100,
        "source_ts_ns": 100,
        "arrival_ts_ns": 100,
    }
    policy = {
        "raw_mid": False,
        "slippage_micropips_per_side": 0,
        "commission_ppm_per_side": 1,
        "financing_ppm_per_day": 0,
    }
    position = {
        "proposal": {"instrument": "USD_JPY", "direction": 1, "notional_jpy_micros": 1},
        "policy": policy,
        "entry": event,
        "entry_price": oracle.Fraction(10_002, 100),
        "units_micros": 1,
        "entry_notional_exact_jpy_micros": oracle.Fraction(10_002, 100),
    }
    value = oracle._position_value(
        position,
        event,
        {"USD_JPY": [event]},
        {"max_conversion_staleness_ns": 1},
        {"USD_JPY": {"price_scale": 100, "pip_ticks": 1}},
    )
    assert value["commission_jpy_micros"] == 2


@pytest.mark.parametrize(
    ("reason", "latency_expected"),
    [("NO_CAUSAL_FILL", True), ("SAME_PAIR_COLLISION_REJECTED", False)],
)
@pytest.mark.parametrize("gross", [50, -50])
def test_rejected_drag_attribution_is_mutually_exclusive_and_reconciles(
    reason: str, latency_expected: bool, gross: int
) -> None:
    proposal = {
        "proposal_ordinal": 1,
        "instrument": "EUR_USD",
        "direction": 1,
        "notional_jpy_micros": 100,
    }
    common = {
        "entry": {"source_event_sha256": "1" * 64},
        "exit": {"source_event_sha256": "2" * 64},
        "gross_pnl_jpy_micros": gross,
    }
    record = oracle._rejected(
        proposal, "3" * 64, "4" * 64, "EXECUTABLE_BASE", reason, common
    )
    latency = record["latency_spread_slippage_drag_jpy_micros"]
    admission = record["admission_opportunity_drag_jpy_micros"]
    assert (latency != 0) is latency_expected
    assert (admission != 0) is not latency_expected
    assert latency + admission == record["common_gross_pnl_jpy_micros"] - record["net_pnl_jpy_micros"]


def test_identical_currency_and_inverse_registry_pairs_are_rejected() -> None:
    with pytest.raises(oracle.OracleError, match="must differ"):
        oracle._pair("USD_USD")
    payload = seal({
        "schema_version": 1,
        "registry_id": "FROZEN_FX_INSTRUMENT_REGISTRY_V1",
        "instruments": {
            "EUR_USD": {"pip_ticks": 10, "price_scale": 100_000},
            "USD_EUR": {"pip_ticks": 10, "price_scale": 100_000},
        },
    }, "registry_sha256")
    with pytest.raises(oracle.OracleError, match="inverse duplicate"):
        oracle._validate_instrument_registry(payload)


def test_accounting_uses_exact_rational_arithmetic_for_large_tick_values() -> None:
    numerator = 8_439_773_459_423_196_600_373_401_704_310_476_485_109
    denominator = 279_443_915_986
    expected = numerator * 1_000_000 // denominator
    assert oracle._asset_micros(oracle.Fraction(numerator, denominator)) == expected
    assert expected == 30_202_029_733_386_734_448_141_701_487_551_657


def test_equal_arrival_conversion_uses_full_completed_batch_watermark() -> None:
    books = {
        "EUR_USD": [{
            "source_ts_ns": 100, "arrival_ts_ns": 200,
            "bid_ticks": 100, "ask_ticks": 101, "tick_scale": 100,
        }],
        "USD_JPY": [
            {"source_ts_ns": 90, "arrival_ts_ns": 190, "bid_ticks": 100, "ask_ticks": 101, "tick_scale": 100},
            {"source_ts_ns": 150, "arrival_ts_ns": 200, "bid_ticks": 200, "ask_ticks": 201, "tick_scale": 100},
        ],
    }
    watermark = oracle._arrival_watermark_from_books(books, 200)
    assert watermark == 150
    assert oracle._convert_to_jpy(
        oracle.Fraction(1, 1), "USD", watermark, 200, books, 200,
        registry={
            "EUR_USD": {"pip_ticks": 1, "price_scale": 100},
            "USD_JPY": {"pip_ticks": 1, "price_scale": 100},
        },
    ) == oracle.Fraction(2, 1)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda policy: policy["arms"]["ADVERSE_STRESS"].__setitem__("commission_ppm_per_side", 1),
        lambda policy: policy["arms"]["ADVERSE_STRESS"].update(policy["arms"]["EXECUTABLE_BASE"]),
    ],
)
def test_adverse_policy_cannot_be_easier_or_identical(tmp_path: Path, mutator: callable) -> None:
    request, _ = fixture(tmp_path, execution_mutation=mutator)
    with pytest.raises(oracle.OracleError, match="ADVERSE"):
        run(tmp_path, request)


def test_unknown_request_root_override_is_rejected(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    request["input_root"] = str(tmp_path)
    with pytest.raises(oracle.OracleError, match="request schema mismatch"):
        run(tmp_path, request)


@pytest.mark.parametrize(
    "relative_path",
    ["inputs/./source.jsonl", "inputs/.hidden", "inputs/" + "a" * 129],
)
def test_noncanonical_or_oversized_artifact_path_is_rejected(
    tmp_path: Path, relative_path: str
) -> None:
    request, _ = fixture(tmp_path)
    request["source_blob"] = dict(request["source_blob"])
    request["source_blob"]["relative_path"] = relative_path
    with pytest.raises(oracle.OracleError, match="unsafe component"):
        run(tmp_path, request)


def test_request_schema_freezes_artifact_roles_and_runtime_path_grammar() -> None:
    schema = json.loads((Path(__file__).parent / "paper_research_jpy_oracle_schema_v2.json").read_text())
    for role in (
        "source_blob", "source_manifest", "proposal", "execution_policy",
        "inventory_policy", "accounting_policy", "evaluation_policy",
        "instrument_registry", "authority_policy",
    ):
        role_schema = schema["$defs"][f"{role}_artifact"]
        assert role_schema["allOf"][1]["properties"]["artifact_id"]["const"] == role
    pattern = schema["$defs"]["artifact"]["properties"]["relative_path"]["pattern"]
    assert re.fullmatch(pattern, "inputs/source.jsonl")
    assert re.fullmatch(pattern, "inputs/./source.jsonl") is None
    assert re.fullmatch(pattern, "inputs/.hidden") is None


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"output_directory":"a","output_directory":"b","schema_version":2}\n', "duplicate"),
        (b'{"schema_version":-0}\n', "negative zero"),
        (b'{"schema_version":2, "x":1}\n', "canonical"),
    ],
)
def test_strict_decoder_rejects_duplicate_negative_zero_and_noncanonical(raw: bytes, message: str) -> None:
    with pytest.raises(oracle.OracleError, match=message):
        oracle.strict_json(raw, "adversarial")


def test_bool_as_numeric_schema_version_is_rejected(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    request["schema_version"] = True
    with pytest.raises(oracle.OracleError, match="integer"):
        run(tmp_path, request)


def test_authority_flip_is_rejected_even_when_resealed(tmp_path: Path) -> None:
    request, values = fixture(tmp_path)
    authority = dict(values["authority_policy"])
    authority["live_authority"] = True
    seal(authority, "authority_policy_sha256")
    path = write_json(tmp_path / "inputs" / "authority_policy.json", authority)
    request["authority_policy"] = artifact(tmp_path, path, "authority_policy")
    with pytest.raises(oracle.OracleError, match="paper authority.*mismatch"):
        run(tmp_path, request)


def test_provider_sequence_and_scale_are_frozen(tmp_path: Path) -> None:
    for index, mutation in enumerate(("provider", "sequence", "scale")):
        case = tmp_path / mutation
        rows = source_rows()
        if mutation == "provider":
            rows[0]["provider_id"] = "EVIL"
        elif mutation == "sequence":
            stream_index = next(i for i, row in enumerate(rows) if row["instrument"] == "EUR_USD" and row["sequence"] == 2)
            rows[stream_index]["sequence"] = 99
        else:
            rows[0]["tick_scale"] *= 10
        request, _ = fixture(case, rows=rows)
        with pytest.raises(oracle.OracleError):
            run(case, request)


def test_dangling_output_symlink_is_rejected_without_escape(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-missing"
    (oracle_output_root(tmp_path) / "oracle_output").symlink_to(
        outside, target_is_directory=True
    )
    with pytest.raises(oracle.OracleError, match="output leaf"):
        run(tmp_path, request)
    assert not outside.exists()


def test_input_artifact_symlink_is_rejected(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    proposal_path = tmp_path / request["proposal"]["relative_path"]
    real = proposal_path.with_name("proposal-real.json")
    proposal_path.rename(real)
    proposal_path.symlink_to(real)
    with pytest.raises((oracle.OracleError, OSError)):
        run(tmp_path, request)


def test_future_price_perturbation_preserves_prior_signal_identity(tmp_path: Path) -> None:
    roots = [tmp_path / "base", tmp_path / "future"]
    signal_ids = []
    prefix_roots = []
    for root, delta in zip(roots, (0, 1000)):
        request, values = fixture(root, future_delta=delta, proposal_specs=[(0, 1)])
        run(root, request)
        record = next(row for row in ledger(root) if row["arm"] == "RAW_SIGNAL")
        signal_ids.append(record["signal_id"])
        prefix_roots.append(values["proposal"]["rows"][0]["completed_data_prefix_root_sha256"])
    assert signal_ids[0] == signal_ids[1]
    assert prefix_roots[0] == prefix_roots[1]


def test_launcher_code_fd_mismatch_fails_before_execution(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    bad = tmp_path / "bad-code.py"
    bad.write_bytes(b"# different\n")
    request_file = tmp_path / "request.json"
    request_file.write_bytes(canonical(request) + b"\n")
    input_fd = oracle._open_trusted_directory(tmp_path, "input")
    output_fd = oracle._open_trusted_directory(oracle_output_root(tmp_path), "output")
    code_fd = os.open(bad, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with pytest.raises(oracle.OracleError, match="code FD differs"):
            oracle.execute_from_fds(
                request_file.read_bytes(),
                input_root_fd=input_fd,
                output_root_fd=output_fd,
                code_fd=code_fd,
            )
    finally:
        os.close(code_fd)
        os.close(input_fd)
        os.close(output_fd)


def test_direct_oracle_rejects_same_input_output_inode(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    with pytest.raises(oracle.OracleError, match="distinct directory inodes"):
        oracle.execute(
            request,
            trusted_input_root=tmp_path,
            trusted_output_root=tmp_path,
        )


def test_writable_runtime_fd_is_rejected(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    module_path = Path(oracle.__file__).resolve()
    input_fd = oracle._open_trusted_directory(tmp_path, "input")
    output_fd = oracle._open_trusted_directory(oracle_output_root(tmp_path), "output")
    code_fd = os.open(module_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    contract_fd = os.open(module_path.parent / oracle.CONTRACT_NAME, os.O_RDONLY)
    schema_fd = os.open(module_path.parent / oracle.SCHEMA_NAME, os.O_RDONLY)
    try:
        with pytest.raises(oracle.OracleError, match="read-only"):
            oracle.execute_from_fds(
                canonical(request) + b"\n",
                input_root_fd=input_fd,
                output_root_fd=output_fd,
                code_fd=code_fd,
                contract_fd=contract_fd,
                schema_fd=schema_fd,
            )
    finally:
        os.close(schema_fd)
        os.close(contract_fd)
        os.close(code_fd)
        os.close(output_fd)
        os.close(input_fd)


@pytest.mark.parametrize("bad_label", ["code", "contract", "schema"])
def test_every_runtime_artifact_fd_is_exactly_bound(
    tmp_path: Path, bad_label: str
) -> None:
    request, _ = fixture(tmp_path)
    module_path = Path(oracle.__file__).resolve()
    paths = {
        "code": module_path,
        "contract": module_path.parent / oracle.CONTRACT_NAME,
        "schema": module_path.parent / oracle.SCHEMA_NAME,
    }
    bad = tmp_path / f"bad-{bad_label}"
    bad.write_bytes(b"different\n")
    descriptors = {
        label: os.open(bad if label == bad_label else path, os.O_RDONLY)
        for label, path in paths.items()
    }
    input_fd = oracle._open_trusted_directory(tmp_path, "input")
    output_fd = oracle._open_trusted_directory(oracle_output_root(tmp_path), "output")
    try:
        with pytest.raises(oracle.OracleError, match=f"{bad_label} FD differs"):
            oracle.execute_from_fds(
                canonical(request) + b"\n",
                input_root_fd=input_fd,
                output_root_fd=output_fd,
                code_fd=descriptors["code"],
                contract_fd=descriptors["contract"],
                schema_fd=descriptors["schema"],
            )
    finally:
        os.close(output_fd)
        os.close(input_fd)
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_existing_complete_output_is_idempotent_exact_readback(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    first = run(tmp_path, request)
    second = run(tmp_path, request)
    assert first["manifest"]["oracle_root_sha256"] == second["manifest"]["oracle_root_sha256"]
    assert first["manifest"]["oracle_ledger_sha256"] == second["manifest"]["oracle_ledger_sha256"]


def test_manifest_unknown_profit_field_is_not_part_of_exact_output_schema(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    run(tmp_path, request)
    manifest_path = oracle_output_root(tmp_path) / "oracle_output" / "oracle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["profit_gate_pass"] = True
    manifest["oracle_root_sha256"] = oracle.embedded_hash(manifest, "oracle_root_sha256")
    manifest_path.write_bytes(canonical(manifest) + b"\n")
    with pytest.raises(oracle.OracleError):
        run(tmp_path, request)


def test_margin_guard_fails_on_ratio_breach_or_closeout() -> None:
    evaluation = {
        "initial_equity_jpy_micros": 100,
        "period_start_ts_ns": START_NS,
        "period_end_ts_ns": START_NS + 1,
        "cvar_tail_bps": 500,
        "cluster_window_ns": 1,
        "margin_notional_cap_jpy_micros": 1_000,
    }
    record = {
        "status": "FILLED_CLOSED",
        "exit_disposition": "MARGIN_CLOSEOUT",
        "signal_id": "a" * 64,
        "economic_lot_id": "a" * 64,
        "instrument": "USD_JPY",
        "entry_arrival_ts_ns": START_NS,
        "net_pnl_jpy_micros": 0,
        "common_gross_pnl_jpy_micros": 0,
        "realized_cost_jpy_micros": 0,
        "fill_sizing_drag_jpy_micros": 0,
        "latency_spread_slippage_drag_jpy_micros": 0,
        "commission_jpy_micros": 0,
        "financing_jpy_micros": 0,
        "admission_opportunity_drag_jpy_micros": 0,
        "economic_net_pnl_jpy_micros_numerator": 0,
        "economic_net_pnl_jpy_micros_denominator": 1,
    }
    risk = [{
        "arrival_ts_ns": START_NS,
        "marked_equity_jpy_micros": 100,
        "gross_notional_jpy_micros": 10,
        "required_margin_jpy_micros": 1,
        "free_margin_jpy_micros": 99,
        "margin_ratio_pass": False,
    }]
    metrics = oracle._arm_metrics(
        [record], [], risk, [], {}, {}, {}, evaluation, 1
    )
    assert metrics["margin_guard_pass"] is False


def test_currency_time_cluster_is_connected_and_ticket_partition_invariant() -> None:
    evaluation = {
        "cluster_window_ns": 1_000,
        "initial_equity_jpy_micros": 1_000_000,
        "cvar_tail_bps": 5_000,
    }
    eur = {
        "status": "FILLED_CLOSED", "instrument": "EUR_USD", "entry_arrival_ts_ns": 1,
        "net_pnl_jpy_micros": -100, "signal_id": "a" * 64,
        "economic_lot_id": "a" * 64,
        "economic_net_pnl_jpy_micros_numerator": -100,
        "economic_net_pnl_jpy_micros_denominator": 1,
    }
    jpy = {
        "status": "FILLED_CLOSED", "instrument": "USD_JPY", "entry_arrival_ts_ns": 1,
        "net_pnl_jpy_micros": 40, "signal_id": "b" * 64,
        "economic_lot_id": "b" * 64,
        "economic_net_pnl_jpy_micros_numerator": 40,
        "economic_net_pnl_jpy_micros_denominator": 1,
    }
    n_eff, cvar, cvar_return, observations = oracle._cluster_metrics([eur, jpy], evaluation)
    assert n_eff == 1
    assert observations[0]["currency_nodes"] == ["EUR", "JPY", "USD"]
    split = [
        {**eur, "net_pnl_jpy_micros": -51,
         "economic_net_pnl_jpy_micros_numerator": -50},
        {**eur, "net_pnl_jpy_micros": -51,
         "economic_net_pnl_jpy_micros_numerator": -50},
        jpy,
    ]
    split_result = oracle._cluster_metrics(split, evaluation)
    assert split_result[:3] == (n_eff, cvar, cvar_return)


def test_cluster_cvar_selects_three_cluster_tail_without_ticket_weighting() -> None:
    evaluation = {
        "cluster_window_ns": 1_000,
        "initial_equity_jpy_micros": 1_000,
        "cvar_tail_bps": 5_000,
    }
    records = [
        {"status": "FILLED_CLOSED", "instrument": "EUR_USD",
         "entry_arrival_ts_ns": timestamp, "net_pnl_jpy_micros": pnl,
         "signal_id": token * 64, "economic_lot_id": token * 64,
         "economic_net_pnl_jpy_micros_numerator": pnl,
         "economic_net_pnl_jpy_micros_denominator": 1}
        for timestamp, pnl, token in ((1, -100, "a"), (1_001, -50, "b"), (2_001, 100, "c"))
    ]
    n_eff, cvar, cvar_return, observations = oracle._cluster_metrics(records, evaluation)
    assert n_eff == len(observations) == 3
    assert cvar == -75
    assert cvar_return == "-0.075000000000000000"
    assert [item["cluster_id"] for item in observations] == sorted(
        item["cluster_id"] for item in observations
    )


@pytest.mark.parametrize(
    ("instrument_spec", "direction", "opening", "expected_delta"),
    [
        ({"price_scale": 100_000, "pip_ticks": 10}, 1, True, 1_000_000),
        ({"price_scale": 100_000, "pip_ticks": 10}, -1, True, -1_000_000),
        ({"price_scale": 100, "pip_ticks": 1}, 1, False, -100_000),
        ({"price_scale": 100, "pip_ticks": 1}, -1, False, 100_000),
    ],
)
def test_pip_scaled_slippage_is_directional_and_instrument_normalized(
    instrument_spec: dict, direction: int, opening: bool, expected_delta: int
) -> None:
    event = {
        "bid_ticks": 10_000,
        "ask_ticks": 10_012,
        "tick_scale": instrument_spec["price_scale"],
    }
    policy = {"raw_mid": False, "slippage_micropips_per_side": 100_000}
    _, numerator, _ = oracle._execution_price_parts(
        event,
        direction,
        opening=opening,
        policy=policy,
        instrument_spec=instrument_spec,
    )
    buy = (opening and direction > 0) or (not opening and direction < 0)
    base_ticks = event["ask_ticks"] if buy else event["bid_ticks"]
    assert numerator - base_ticks * oracle.PRICE_SUBPIP_SCALE == expected_delta


def test_cluster_return_denominator_does_not_depend_on_future_exit_pnl() -> None:
    evaluation = {
        "cluster_window_ns": 100,
        "initial_equity_jpy_micros": 1_000,
        "cvar_tail_bps": 5_000,
    }
    records = [
        {"status": "FILLED_CLOSED", "instrument": "EUR_USD", "entry_arrival_ts_ns": 1, "net_pnl_jpy_micros": -100, "signal_id": "a" * 64, "economic_lot_id": "a" * 64, "economic_net_pnl_jpy_micros_numerator": -100, "economic_net_pnl_jpy_micros_denominator": 1},
        {"status": "FILLED_CLOSED", "instrument": "USD_JPY", "entry_arrival_ts_ns": 201, "net_pnl_jpy_micros": -10, "signal_id": "b" * 64, "economic_lot_id": "b" * 64, "economic_net_pnl_jpy_micros_numerator": -10, "economic_net_pnl_jpy_micros_denominator": 1},
    ]
    first_observations = oracle._cluster_metrics(records, evaluation)[3]
    first = next(
        item["signed_return"]
        for item in first_observations
        if item["time_bucket"] == 2
    )
    records[0] = {
        **records[0],
        "net_pnl_jpy_micros": -500,
        "economic_net_pnl_jpy_micros_numerator": -500,
    }
    second_observations = oracle._cluster_metrics(records, evaluation)[3]
    second = next(
        item["signed_return"]
        for item in second_observations
        if item["time_bucket"] == 2
    )
    assert first == second == "-0.010000000000000000"


def test_rejected_rows_keep_economic_lot_identity_distinct_from_signal_id() -> None:
    proposal = {
        "proposal_ordinal": 1,
        "instrument": "USD_JPY",
        "direction": 1,
        "notional_jpy_micros": 28_000_000_000,
    }
    signal_id = "a" * 64
    economic_lot_id = "b" * 64
    row = oracle._rejected(
        proposal,
        signal_id,
        economic_lot_id,
        "EXECUTABLE_BASE",
        "GROSS_CAP_REJECTED",
        None,
    )

    assert row["signal_id"] == signal_id
    assert row["economic_lot_id"] == economic_lot_id
    assert row["economic_lot_id"] != row["signal_id"]


def test_mark_valuation_financing_uses_current_arrival_cutoff(tmp_path: Path) -> None:
    request, values = fixture(tmp_path)
    rows = values["source_rows"]
    entry = next(row for row in rows if row["instrument"] == "USD_JPY" and row["sequence"] == 1)
    mark = next(row for row in rows if row["instrument"] == "USD_JPY" and row["sequence"] == 2)
    policy = values["execution_policy"]["arms"]["EXECUTABLE_BASE"]
    entry_price, _, _ = oracle._execution_price_parts(
        entry, 1, opening=True, policy=policy, instrument_spec={"price_scale": 100, "pip_ticks": 1}
    )
    position = {
        "proposal": {"instrument": "USD_JPY", "direction": 1, "notional_jpy_micros": 28_000_000_000},
        "policy": policy,
        "entry": entry,
        "entry_price": entry_price,
        "units_micros": 1_000_000,
        "entry_notional_exact_jpy_micros": entry_price * oracle.JPY_MICROS_PER_YEN,
    }
    books = {
        "USD_JPY": [row for row in rows if row["instrument"] == "USD_JPY"],
    }
    cutoff = entry["arrival_ts_ns"] + oracle.DAY_NS // 2
    values_at_cutoff = oracle._position_value(
        position, mark, books, values["accounting_policy"], {"USD_JPY": {"price_scale": 100, "pip_ticks": 1}},
        valuation_source_watermark_ns=mark["source_ts_ns"], valuation_arrival_ns=cutoff,
    )
    assert values_at_cutoff["elapsed_ns"] == oracle.DAY_NS // 2
    assert values_at_cutoff["financing_jpy_micros"] == 76


def test_full_month_grid_keeps_zero_activity_months() -> None:
    jan = int(__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc).timestamp()) * 1_000_000_000
    apr = int(__import__("datetime").datetime(2026, 4, 1, tzinfo=__import__("datetime").timezone.utc).timestamp()) * 1_000_000_000
    assert oracle._complete_months(jan, apr) == ["2026-01", "2026-02", "2026-03"]


def test_arm_metrics_retains_partial_months_and_uses_running_marked_peak() -> None:
    from datetime import datetime, timezone

    start = int(datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    end = int(datetime(2026, 3, 15, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    initial = 100_000_000
    evaluation = {
        "period_start_ts_ns": start,
        "period_end_ts_ns": end,
        "initial_equity_jpy_micros": initial,
        "margin_notional_cap_jpy_micros": initial,
        "margin_rate_bps": 500,
        "max_gross_to_equity_bps": 10_000,
        "cvar_tail_bps": 500,
        "cluster_window_ns": oracle.HOUR_NS,
    }
    risk = [
        {"arrival_ts_ns": start + 1, "marked_equity_jpy_micros": 100_000_000,
         "gross_notional_jpy_micros": 0, "required_margin_jpy_micros": 0,
         "free_margin_jpy_micros": 100_000_000, "margin_ratio_pass": True},
        {"arrival_ts_ns": start + 2, "marked_equity_jpy_micros": 120_000_000,
         "gross_notional_jpy_micros": 0, "required_margin_jpy_micros": 0,
         "free_margin_jpy_micros": 120_000_000, "margin_ratio_pass": True},
        {"arrival_ts_ns": start + 3, "marked_equity_jpy_micros": 90_000_000,
         "gross_notional_jpy_micros": 0, "required_margin_jpy_micros": 0,
         "free_margin_jpy_micros": 90_000_000, "margin_ratio_pass": True},
        {"arrival_ts_ns": start + 4, "marked_equity_jpy_micros": 110_000_000,
         "gross_notional_jpy_micros": 0, "required_margin_jpy_micros": 0,
         "free_margin_jpy_micros": 110_000_000, "margin_ratio_pass": True},
    ]
    metrics = oracle._arm_metrics(
        [], [], risk, [], {}, {}, {}, evaluation, oracle.DAY_NS,
    )
    assert [(item["month_id"], item["comparable_full_month"]) for item in metrics["monthly"]] == [
        ("2026-01", False), ("2026-02", True), ("2026-03", False),
    ]
    assert all(item["equity_multiple"] == "1.000000000000000000" for item in metrics["monthly"])
    assert metrics["max_drawdown_jpy_micros"] == 30_000_000
    assert metrics["max_drawdown_ratio"] == "0.250000000000000000"


def test_month_end_executable_mark_is_in_drawdown_observation_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime, timezone

    start = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    feb = int(datetime(2026, 2, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    end = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    evaluation = {
        "period_start_ts_ns": start,
        "period_end_ts_ns": end,
        "initial_equity_jpy_micros": 100,
        "margin_notional_cap_jpy_micros": 1_000,
        "margin_rate_bps": 500,
        "max_gross_to_equity_bps": 10_000,
        "cvar_tail_bps": 500,
        "cluster_window_ns": oracle.HOUR_NS,
    }
    risk = [
        {"arrival_ts_ns": start + 1, "marked_equity_jpy_micros": 120,
         "gross_notional_jpy_micros": 0, "required_margin_jpy_micros": 0,
         "free_margin_jpy_micros": 120, "margin_ratio_pass": True},
        {"arrival_ts_ns": feb + 1, "marked_equity_jpy_micros": 115,
         "gross_notional_jpy_micros": 0, "required_margin_jpy_micros": 0,
         "free_margin_jpy_micros": 115, "margin_ratio_pass": True},
    ]

    def boundary_equity(*args: object, **kwargs: object) -> int:
        del kwargs
        cutoff = args[1]
        if cutoff == feb - 1:
            return 80
        if cutoff == end - 1:
            return 115
        return 100

    monkeypatch.setattr(oracle, "_equity_at", boundary_equity)
    metrics = oracle._arm_metrics(
        [], [], risk, [], {}, {}, {}, evaluation, oracle.DAY_NS,
    )
    assert metrics["max_drawdown_jpy_micros"] == 40
    assert metrics["max_drawdown_ratio"] == "0.333333333333333334"
    assert metrics["minimum_marked_equity_jpy_micros"] == 80


def test_absolute_and_ratio_drawdown_maxima_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = {
        "period_start_ts_ns": START_NS,
        "period_end_ts_ns": START_NS + 10,
        "initial_equity_jpy_micros": 100,
        "margin_notional_cap_jpy_micros": 2_000,
        "margin_rate_bps": 500,
        "max_gross_to_equity_bps": 10_000,
        "cvar_tail_bps": 500,
        "cluster_window_ns": oracle.HOUR_NS,
    }
    risk = [
        {"arrival_ts_ns": START_NS + index, "marked_equity_jpy_micros": equity,
         "gross_notional_jpy_micros": 0, "required_margin_jpy_micros": 0,
         "free_margin_jpy_micros": equity, "margin_ratio_pass": True}
        for index, equity in enumerate((100, 50, 1_000, 800), 1)
    ]
    monkeypatch.setattr(
        oracle,
        "_equity_at",
        lambda *args, **kwargs: 800 if args[1] == START_NS + 9 else 100,
    )
    metrics = oracle._arm_metrics([], [], risk, [], {}, {}, {}, evaluation, 1)
    assert metrics["max_drawdown_jpy_micros"] == 200
    assert metrics["max_drawdown_ratio"] == "0.500000000000000000"


def test_cross_month_open_position_uses_causal_executable_month_end_mtm() -> None:
    from datetime import datetime, timezone

    boundary = int(datetime(2026, 2, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    entry = {
        "instrument": "USD_JPY", "source_ts_ns": boundary - 3_600_000_000_000,
        "arrival_ts_ns": boundary - 3_600_000_000_000, "bid_ticks": 10_000,
        "ask_ticks": 10_002, "tick_scale": 100,
    }
    mark = {
        "instrument": "USD_JPY", "source_ts_ns": boundary - 1,
        "arrival_ts_ns": boundary - 1, "bid_ticks": 10_100,
        "ask_ticks": 10_102, "tick_scale": 100,
    }
    policy = {
        "raw_mid": False, "slippage_micropips_per_side": 0,
        "commission_ppm_per_side": 0, "financing_ppm_per_day": 0,
    }
    position = {
        "proposal": {"instrument": "USD_JPY", "direction": 1, "notional_jpy_micros": 100_000_000},
        "entry": entry,
        "entry_price": oracle.Fraction(10_002, 100),
        "units_micros": 1_000_000,
        "entry_notional_exact_jpy_micros": oracle.Fraction(10_002, 100) * oracle.JPY_MICROS_PER_YEN,
        "policy": policy,
        "closed_record": {"exit_arrival_ts_ns": boundary + 1, "net_pnl_jpy_micros": 999_999_999},
    }
    evaluation = {"initial_equity_jpy_micros": 100_000_000}
    equity = oracle._equity_at(
        [position], boundary - 1, [entry, mark], {"USD_JPY": [entry, mark]},
        {"max_conversion_staleness_ns": oracle.HOUR_NS},
        {"USD_JPY": {"price_scale": 100, "pip_ticks": 1}},
        evaluation, oracle.HOUR_NS,
    )
    # Long liquidation uses BID 101.00, not midpoint 101.01.  The still-open
    # position is marked; the deliberately huge future close result is ignored.
    assert equity == 100_980_000


def test_terminal_liquidation_uses_period_cutoff_clock_not_quote_arrival(tmp_path: Path) -> None:
    request, _ = fixture(
        tmp_path,
        proposal_specs=[(0, 1)],
        proposal_max_age_seconds=2_000,
    )
    result = run(tmp_path, request)
    cutoff = START_NS + 901_000_000_000 - 1
    rows = ledger(tmp_path)
    for row in rows:
        assert row["exit_disposition"] == "TERMINAL_LIQUIDATION"
        assert row["exit_arrival_ts_ns"] == cutoff
        assert row["exit_source_reference"]["arrival_ts_ns"] < cutoff
        assert row["elapsed_ns"] == cutoff - row["entry_arrival_ts_ns"]
    assert all(
        result["manifest"]["oracle_metrics"]["arms"][arm]["terminal_open_positions"] == 0
        for arm in oracle.ARMS
    )


def test_terminal_preliquidation_mark_sets_exact_minimum_free_margin(
    tmp_path: Path,
) -> None:
    request, _ = fixture(tmp_path)
    result = run(tmp_path, request)
    adverse = result["manifest"]["oracle_metrics"]["arms"]["ADVERSE_STRESS"]

    # The final real quote arrives before the period cutoff.  Financing keeps
    # accruing until the terminal valuation clock, so the pre-liquidation state
    # is one micro lower than the last source-event state:
    #   199_988_381_412 equity - 1_400_521_838 margin.
    assert adverse["minimum_marked_equity_jpy_micros"] == 199_988_381_412
    assert adverse["maximum_required_margin_jpy_micros"] == 1_400_521_838
    assert adverse["minimum_free_margin_jpy_micros"] == 198_587_859_574


def test_no_tick_month_boundary_financing_breach_closes_before_reporting_pass(
    tmp_path: Path,
) -> None:
    # Reuse the independently hand-built frozen reference fixture bytes, not a
    # producer result.  Its last real quote predates the February boundary;
    # only accrued financing at the no-tick checkpoint crosses the risk gate.
    from test_paper_research_double_entry_reference_v2 import (
        END_NS,
        _boundary_risk_artifacts,
        _build_canonical_fixture,
    )

    raw_artifacts = _boundary_risk_artifacts(
        _build_canonical_fixture(), END_NS - 1
    )
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True)
    request: dict[str, object] = {
        "schema_version": 2,
        "output_directory": "oracle_output",
    }
    for label, raw in raw_artifacts.items():
        suffix = ".jsonl" if label == "source_blob" else ".json"
        path = input_dir / f"{label}{suffix}"
        path.write_bytes(raw)
        request[label] = artifact(tmp_path, path, label)

    result = run(tmp_path, request)
    rows = ledger(tmp_path)
    for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS"):
        row = next(item for item in rows if item["arm"] == arm)
        assert row["exit_disposition"] == "MARGIN_CLOSEOUT"
        assert row["exit_arrival_ts_ns"] == END_NS - 1
        assert row["exit_source_reference"]["arrival_ts_ns"] < END_NS - 1
        metrics = result["manifest"]["oracle_metrics"]["arms"][arm]
        assert metrics["minimum_free_margin_jpy_micros"] < 0
        assert metrics["margin_guard_pass"] is False
        assert metrics["terminal_open_positions"] == 0


def test_nonpositive_equity_forces_closeout_and_halts_later_proposals(tmp_path: Path) -> None:
    rows = source_rows()
    crashed: list[dict] = []
    for row in rows:
        changed = dict(row)
        if row["instrument"] == "USD_JPY" and row["sequence"] >= 4:
            changed["bid_ticks"] = row["sequence"]
            changed["ask_ticks"] = row["sequence"] + 1
        crashed.append(changed)

    def small_equity(evaluation: dict) -> None:
        evaluation["initial_equity_jpy_micros"] = 15_000_000_000

    request, _ = fixture(
        tmp_path,
        rows=crashed,
        proposal_specs=[(1, 1), (7, 1)],
        evaluation_mutation=small_equity,
        proposal_max_age_seconds=600,
    )
    result = run(tmp_path, request)
    rows_by_arm = {
        arm: [row for row in ledger(tmp_path) if row["arm"] == arm]
        for arm in oracle.ARMS
    }
    for arm, arm_rows in rows_by_arm.items():
        assert arm_rows[0]["exit_disposition"] == "MARGIN_CLOSEOUT"
        assert arm_rows[1]["status"] == "ACCOUNT_HALTED"
        metrics = result["manifest"]["oracle_metrics"]["arms"][arm]
        assert metrics["minimum_marked_equity_jpy_micros"] <= 0
        assert metrics["margin_guard_pass"] is False
        assert metrics["terminal_open_positions"] == 0


def test_ledger_only_stage_recovers_without_stale_lock_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = fixture(tmp_path)
    original = oracle._write_file_at

    def fail_before_manifest(directory_fd: int, name: str, data: bytes, mode: int = 0o600) -> None:
        if name == "oracle_manifest.json":
            raise OSError("injected manifest fault")
        original(directory_fd, name, data, mode)

    monkeypatch.setattr(oracle, "_write_file_at", fail_before_manifest)
    with pytest.raises(OSError, match="injected"):
        run(tmp_path, request)
    monkeypatch.setattr(oracle, "_write_file_at", original)
    result = run(tmp_path, request)
    assert result["manifest"]["status"] == "COMPLETE"
    assert (oracle_output_root(tmp_path) / "oracle_output" / "COMMIT.json").is_file()


def test_dangling_child_in_recoverable_stage_is_quarantined_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = fixture(tmp_path)
    stage = leave_recoverable_oracle_stage(tmp_path, request, monkeypatch)
    poisoned_child = stage / "intent.json"
    poisoned_child.unlink()
    poisoned_child.symlink_to("missing-stage-child")
    publish_root = oracle_output_root(tmp_path)
    lock_path = publish_root / ".oracle_output.lock"
    lock_before = lock_path.stat()

    with pytest.raises(oracle.OracleError, match="FAILED_VISIBLE_PARTIAL_OUTPUT"):
        run(tmp_path, request)

    failed = list(publish_root.glob(".oracle_output.*.failed"))
    assert len(failed) == 1
    failed_info = failed[0].stat()
    assert not stage.exists()
    assert (failed[0] / "intent.json").is_symlink()
    assert os.readlink(failed[0] / "intent.json") == "missing-stage-child"
    assert not (publish_root / "oracle_output").exists()
    lock_after_quarantine = lock_path.stat()
    assert (lock_after_quarantine.st_dev, lock_after_quarantine.st_ino) == (
        lock_before.st_dev,
        lock_before.st_ino,
    )
    assert lock_after_quarantine.st_nlink == 1
    assert lock_after_quarantine.st_mode & 0o777 == 0o600

    with pytest.raises(
        oracle.OracleError, match="prior partial output failure is preserved"
    ):
        run(tmp_path, request)

    failed_after_retry = list(publish_root.glob(".oracle_output.*.failed"))
    assert failed_after_retry == failed
    assert (failed_after_retry[0].stat().st_dev, failed_after_retry[0].stat().st_ino) == (
        failed_info.st_dev,
        failed_info.st_ino,
    )
    assert not stage.exists()
    assert not (publish_root / "oracle_output").exists()


def test_native_recovered_stage_destination_collision_preserves_both_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = fixture(tmp_path)
    stage = leave_recoverable_oracle_stage(tmp_path, request, monkeypatch)
    marker = b"preserve-colliding-destination\n"

    def collide(root_fd: int, source: str, destination: str) -> None:
        del root_fd, source
        target = oracle_output_root(tmp_path) / destination
        target.mkdir(mode=0o700)
        (target / "marker.txt").write_bytes(marker)
        raise FileExistsError(destination)

    monkeypatch.setattr(oracle, "_RENAME_EXCLUSIVE", collide)
    with pytest.raises(FileExistsError):
        run(tmp_path, request)
    assert stage.is_dir()
    assert (
        oracle_output_root(tmp_path) / "oracle_output" / "marker.txt"
    ).read_bytes() == marker
    (oracle_output_root(tmp_path) / "oracle_output" / "marker.txt").unlink()
    (oracle_output_root(tmp_path) / "oracle_output").rmdir()

    monkeypatch.setattr(oracle, "_RENAME_EXCLUSIVE", exclusive_rename_at)
    recovered = run(tmp_path, request)
    assert recovered["manifest"]["status"] == "COMPLETE"
    assert (oracle_output_root(tmp_path) / "oracle_output" / "COMMIT.json").is_file()


def test_native_recovered_stage_substitution_never_succeeds_and_can_reverify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = fixture(tmp_path)
    stage = leave_recoverable_oracle_stage(tmp_path, request, monkeypatch)
    held_name = stage.name + ".held"

    def substitute_then_quarantine(
        root_fd: int, source: str, destination: str
    ) -> None:
        if destination == "oracle_output":
            os.rename(source, held_name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
            os.mkdir(source, 0o700, dir_fd=root_fd)
            os.rename(source, destination, src_dir_fd=root_fd, dst_dir_fd=root_fd)
            return
        if destination.endswith(".failed"):
            os.rename(held_name, destination, src_dir_fd=root_fd, dst_dir_fd=root_fd)
            return
        raise AssertionError("unexpected native rename target")

    monkeypatch.setattr(oracle, "_RENAME_EXCLUSIVE", substitute_then_quarantine)
    with pytest.raises(oracle.OracleError, match="STAGE_PATH_SUBSTITUTED"):
        run(tmp_path, request)
    assert not (
        oracle_output_root(tmp_path) / "oracle_output" / "COMMIT.json"
    ).exists()
    held = oracle_output_root(tmp_path) / held_name
    assert held.is_dir()
    assert list(oracle_output_root(tmp_path).glob(".oracle_output.*.failed")) == []
    (oracle_output_root(tmp_path) / "oracle_output").rmdir()
    held.rename(stage)

    monkeypatch.setattr(oracle, "_RENAME_EXCLUSIVE", exclusive_rename_at)
    recovered = run(tmp_path, request)
    assert recovered["manifest"]["status"] == "COMPLETE"


def test_native_recovered_stage_post_rename_root_fsync_fault_reverifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = fixture(tmp_path)
    leave_recoverable_oracle_stage(tmp_path, request, monkeypatch)
    original_fsync = os.fsync
    renamed = False
    failed = False

    def rename_then_flag(root_fd: int, source: str, destination: str) -> None:
        nonlocal renamed
        exclusive_rename_at(root_fd, source, destination)
        renamed = True

    def fail_first_post_rename_fsync(descriptor: int) -> None:
        nonlocal failed
        if renamed and not failed:
            failed = True
            raise OSError("injected post-rename root fsync fault")
        original_fsync(descriptor)

    monkeypatch.setattr(oracle, "_RENAME_EXCLUSIVE", rename_then_flag)
    monkeypatch.setattr(oracle.os, "fsync", fail_first_post_rename_fsync)
    with pytest.raises(OSError, match="post-rename root fsync fault"):
        run(tmp_path, request)
    assert failed is True
    assert (oracle_output_root(tmp_path) / "oracle_output" / "COMMIT.json").is_file()

    monkeypatch.setattr(oracle.os, "fsync", original_fsync)
    recovered = run(tmp_path, request)
    assert recovered["manifest"]["status"] == "COMPLETE"


def test_native_stage_path_substitution_never_returns_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = fixture(tmp_path)

    def substitute_then_rename(root_fd: int, source: str, destination: str) -> None:
        held = source + ".held"
        os.rename(source, held, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.mkdir(source, 0o700, dir_fd=root_fd)
        os.rename(source, destination, src_dir_fd=root_fd, dst_dir_fd=root_fd)

    monkeypatch.setattr(oracle, "_RENAME_EXCLUSIVE", substitute_then_rename)
    with pytest.raises(oracle.OracleError, match="published output inode mismatch"):
        run(tmp_path, request)
    assert not (
        oracle_output_root(tmp_path) / "oracle_output" / "COMMIT.json"
    ).exists()


def test_native_final_path_substitution_during_validation_never_returns_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = fixture(tmp_path)

    def exclusive_rename(root_fd: int, source: str, destination: str) -> None:
        if oracle._lstat_at(root_fd, destination) is not None:
            raise FileExistsError(destination)
        os.rename(source, destination, src_dir_fd=root_fd, dst_dir_fd=root_fd)

    original_validate = oracle._validate_complete_output_fd
    validations = 0

    def validate_then_substitute(*args: object, **kwargs: object) -> dict:
        nonlocal validations
        result = original_validate(*args, **kwargs)
        validations += 1
        if validations == 2:
            publish_root = oracle_output_root(tmp_path)
            os.rename(
                publish_root / "oracle_output",
                publish_root / ".oracle_output.detached",
            )
            (publish_root / "oracle_output").mkdir(mode=0o700)
        return result

    monkeypatch.setattr(oracle, "_RENAME_EXCLUSIVE", exclusive_rename)
    monkeypatch.setattr(oracle, "_validate_complete_output_fd", validate_then_substitute)
    with pytest.raises(oracle.OracleError, match="pathname changed during validation"):
        run(tmp_path, request)
    assert validations == 2
    assert not (
        oracle_output_root(tmp_path) / "oracle_output" / "COMMIT.json"
    ).exists()


def test_extra_output_file_invalidates_exact_commit(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    run(tmp_path, request)
    (
        oracle_output_root(tmp_path) / "oracle_output" / "unexpected.txt"
    ).write_text("x")
    with pytest.raises(oracle.OracleError, match="file set"):
        run(tmp_path, request)


def test_hardlinked_lock_is_rejected_without_corrupting_victim(tmp_path: Path) -> None:
    request, _ = fixture(tmp_path)
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"preserve-me\n")
    victim.chmod(0o600)
    os.link(victim, oracle_output_root(tmp_path) / ".oracle_output.lock")
    with pytest.raises(oracle.OracleError, match="lock file"):
        run(tmp_path, request)
    assert victim.read_bytes() == b"preserve-me\n"


def test_named_lock_replacement_never_quarantines_active_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = fixture(tmp_path)
    publish_root = oracle_output_root(tmp_path)
    original = oracle._write_file_at
    replaced = False

    def replace_lock_after_first_stage_write(
        directory_fd: int, name: str, data: bytes, mode: int = 0o600
    ) -> None:
        nonlocal replaced
        original(directory_fd, name, data, mode)
        if not replaced:
            lock_path = publish_root / ".oracle_output.lock"
            lock_path.unlink()
            lock_path.write_bytes(b"")
            lock_path.chmod(0o600)
            replaced = True

    monkeypatch.setattr(oracle, "_write_file_at", replace_lock_after_first_stage_write)
    with pytest.raises(oracle.LockIdentityError):
        run(tmp_path, request)
    assert replaced is True
    assert len(list(publish_root.glob(".oracle_output.*.stage"))) == 1
    assert list(publish_root.glob(".oracle_output.*.failed")) == []
    assert not (publish_root / "oracle_output").exists()


def test_named_lock_replacement_at_commit_boundary_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = fixture(tmp_path)
    publish_root = oracle_output_root(tmp_path)
    renamed = False

    def rename_then_replace_lock(
        root_fd: int, source: str, destination: str
    ) -> None:
        nonlocal renamed
        exclusive_rename_at(root_fd, source, destination)
        lock_path = publish_root / ".oracle_output.lock"
        lock_path.unlink()
        lock_path.write_bytes(b"")
        lock_path.chmod(0o600)
        renamed = True

    monkeypatch.setattr(oracle, "_RENAME_EXCLUSIVE", rename_then_replace_lock)
    with pytest.raises(oracle.LockIdentityError):
        run(tmp_path, request)
    assert renamed is True
    assert (publish_root / "oracle_output" / "COMMIT.json").is_file()
    assert list(publish_root.glob(".oracle_output.*.failed")) == []


def test_oracle_import_graph_has_no_runner_process_network_or_dynamic_dependency() -> None:
    tree = ast.parse(Path(oracle.__file__).read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    forbidden = {
        "paper_research_jpy_oracle_v1", "paper_research_oracle_verifier_v2",
        "paper_research_template_runner_v3", "paper_research_system_v3",
        "jpy_accounting_v2", "shadow_jpy_accounting_v1", "result_validator",
        "socket", "requests", "subprocess", "ctypes", "importlib",
        "builtins", "_posixsubprocess",
    }
    assert imports.isdisjoint(forbidden)
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {"float", "eval", "exec", "__import__"}
        for node in ast.walk(tree)
    )

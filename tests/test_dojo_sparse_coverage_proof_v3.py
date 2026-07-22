from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import quant_rabbit.dojo_sparse_coverage_proof_v3 as coverage_module

from quant_rabbit.dojo_sparse_coverage_proof_v3 import (
    COVERAGE_INPUT_CONTRACT,
    EXPECTED_OPEN,
    LEGITIMATE_NO_CANDLE,
    MARKET_CALENDAR_SPEC_CONTRACT,
    NO_CANDLE_REASON,
    OBSERVED,
    SCHEMA_VERSION,
    VERIFIED_CLOSED,
    DojoSparseCoverageProofV3Error,
    build_coverage_proof_receipt,
    build_market_closure_receipt,
    build_no_candle_receipt,
    build_observed_candle,
    build_sealed_market_calendar_artifact,
    canonical_json_bytes,
    canonical_sha256,
    verify_coverage_proof_receipt,
    verify_sealed_market_calendar_artifact,
    write_json_exclusive,
)


START = 1_735_689_600
END = START + 900
PROVIDER = "OANDA"
PRODUCER = "oanda-candle-acquirer"
VERIFIER = "independent-response-verifier"
PAIRS = ("EUR_USD", "USD_JPY")
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _calendar_spec() -> dict[str, Any]:
    closure = build_market_closure_receipt(
        provider=PROVIDER,
        reason="PROVIDER_PUBLISHED_HOLIDAY_CLOSURE",
        effective_from_epoch=START + 300,
        effective_to_epoch=START + 600,
        provider_evidence_sha256=SHA_A,
        independent_verification_artifact_sha256=SHA_D,
        source_producer_id=PRODUCER,
        independent_verifier_id=VERIFIER,
    )
    return {
        "contract": MARKET_CALENDAR_SPEC_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "provider": PROVIDER,
        "calendar_id": "oanda-2025-new-year-evidence-v1",
        "granularity": "M5",
        "from_epoch": START,
        "to_epoch": END,
        "source_producer_id": PRODUCER,
        "independent_verifier_id": VERIFIER,
        "closure_receipts": [closure],
        "slots": [
            {"epoch": START, "state": EXPECTED_OPEN},
            {
                "epoch": START + 300,
                "state": VERIFIED_CLOSED,
                "closure_receipt_sha256": closure["receipt_sha256"],
            },
            {"epoch": START + 600, "state": EXPECTED_OPEN},
        ],
    }


def _coverage_input(calendar: dict[str, Any]) -> dict[str, Any]:
    eur_observed = build_observed_candle(
        pair="EUR_USD",
        epoch=START,
        bid=[1.1000, 1.1010, 1.0990, 1.1005],
        ask=[1.1002, 1.1012, 1.0992, 1.1007],
        source_artifact_sha256=SHA_B,
    )
    jpy_observed = build_observed_candle(
        pair="USD_JPY",
        epoch=START + 600,
        bid=[150.0, 150.1, 149.9, 150.05],
        ask=[150.02, 150.12, 149.92, 150.07],
        source_artifact_sha256=SHA_C,
    )
    eur_no_candle = build_no_candle_receipt(
        provider=PROVIDER,
        pair="EUR_USD",
        effective_from_epoch=START + 600,
        effective_to_epoch=START + 900,
        request_from_epoch=START,
        request_to_epoch=END,
        provider_request_sha256=SHA_A,
        provider_response_sha256=SHA_B,
        independent_verification_artifact_sha256=SHA_D,
        calendar_artifact_sha256=calendar["calendar_artifact_sha256"],
        source_producer_id=PRODUCER,
        independent_verifier_id=VERIFIER,
    )
    jpy_no_candle = build_no_candle_receipt(
        provider=PROVIDER,
        pair="USD_JPY",
        effective_from_epoch=START,
        effective_to_epoch=START + 300,
        request_from_epoch=START,
        request_to_epoch=END,
        provider_request_sha256=SHA_A,
        provider_response_sha256=SHA_C,
        independent_verification_artifact_sha256=SHA_D,
        calendar_artifact_sha256=calendar["calendar_artifact_sha256"],
        source_producer_id=PRODUCER,
        independent_verifier_id=VERIFIER,
    )
    return {
        "contract": COVERAGE_INPUT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "calendar_artifact_sha256": calendar["calendar_artifact_sha256"],
        "provider": PROVIDER,
        "granularity": "M5",
        "from_epoch": START,
        "to_epoch": END,
        "feed_pairs": list(PAIRS),
        "source_producer_id": PRODUCER,
        "independent_verifier_id": VERIFIER,
        "observed_candles": [eur_observed, jpy_observed],
        "no_candle_receipts": [eur_no_candle, jpy_no_candle],
        "classifications": [
            {
                "pair": "EUR_USD",
                "epoch": START,
                "classification": OBSERVED,
                "observed_candle_sha256": eur_observed["candle_sha256"],
            },
            {
                "pair": "EUR_USD",
                "epoch": START + 600,
                "classification": LEGITIMATE_NO_CANDLE,
                "no_candle_receipt_sha256": eur_no_candle["receipt_sha256"],
            },
            {
                "pair": "USD_JPY",
                "epoch": START,
                "classification": LEGITIMATE_NO_CANDLE,
                "no_candle_receipt_sha256": jpy_no_candle["receipt_sha256"],
            },
            {
                "pair": "USD_JPY",
                "epoch": START + 600,
                "classification": OBSERVED,
                "observed_candle_sha256": jpy_observed["candle_sha256"],
            },
        ],
    }


def _sealed_body(value: dict[str, Any], field: str) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != field}
    return {**body, field: canonical_sha256(body)}


def _valid_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    calendar = build_sealed_market_calendar_artifact(_calendar_spec())
    return calendar, _coverage_input(calendar)


def test_complete_partition_remains_candidate_only_without_authenticated_bytes() -> (
    None
):
    calendar, coverage = _valid_inputs()
    receipt = build_coverage_proof_receipt(
        calendar_artifact=calendar,
        coverage_input=coverage,
    )

    assert calendar["grid_slot_count"] == 3
    assert calendar["expected_open_slot_count"] == 2
    assert calendar["verified_closed_slot_count"] == 1
    assert receipt["expected_pair_slot_count"] == 4
    assert receipt["classified_pair_slot_count"] == 4
    assert receipt["observed_pair_slot_count"] == 2
    assert receipt["legitimate_no_candle_pair_slot_count"] == 2
    assert receipt["unclassified_pair_slot_count"] == 0
    assert receipt["proof_eligible"] is False
    assert receipt["sparse_calendar_coverage_proved"] is False
    assert receipt["proof_classification"] == (
        "CANDIDATE_ONLY_UNAUTHENTICATED_SOURCE_REFERENCES"
    )
    assert receipt["coverage_input_sha256"] == canonical_sha256(coverage)
    assert receipt["authority"]["existing_generation_upgrade_allowed"] is False
    assert receipt["authority"]["requires_new_generation_binding"] is True
    assert verify_sealed_market_calendar_artifact(calendar) == calendar
    assert (
        verify_coverage_proof_receipt(
            receipt,
            calendar_artifact=calendar,
            coverage_input=coverage,
        )
        == receipt
    )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "out_of_range"])
def test_calendar_rejects_grid_slot_missing_duplicate_or_out_of_range(
    mutation: str,
) -> None:
    spec = _calendar_spec()
    if mutation == "missing":
        del spec["slots"][1]
    elif mutation == "duplicate":
        spec["slots"][1]["epoch"] = START
    else:
        spec["slots"][2]["epoch"] = END

    with pytest.raises(DojoSparseCoverageProofV3Error):
        build_sealed_market_calendar_artifact(spec)


def test_calendar_never_infers_a_holiday_without_independent_receipt() -> None:
    spec = _calendar_spec()
    spec["closure_receipts"] = []

    with pytest.raises(
        DojoSparseCoverageProofV3Error,
        match="lacks its independent closure receipt",
    ):
        build_sealed_market_calendar_artifact(spec)


@pytest.mark.parametrize("mutation", ["provider", "reason", "interval", "seal"])
def test_calendar_rejects_closure_receipt_mismatch(mutation: str) -> None:
    spec = _calendar_spec()
    receipt = spec["closure_receipts"][0]
    if mutation == "provider":
        receipt["provider"] = "OTHER_PROVIDER"
        spec["closure_receipts"][0] = _sealed_body(receipt, "receipt_sha256")
    elif mutation == "reason":
        receipt["reason"] = "GUESSED_HOLIDAY"
        spec["closure_receipts"][0] = _sealed_body(receipt, "receipt_sha256")
    elif mutation == "interval":
        receipt["effective_from_epoch"] = START
        spec["closure_receipts"][0] = _sealed_body(receipt, "receipt_sha256")
    else:
        receipt["receipt_sha256"] = "0" * 64

    with pytest.raises(DojoSparseCoverageProofV3Error):
        build_sealed_market_calendar_artifact(spec)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "out_of_range"])
def test_pair_slot_classification_rejects_missing_duplicate_or_out_of_range(
    mutation: str,
) -> None:
    calendar, coverage = _valid_inputs()
    if mutation == "missing":
        coverage["classifications"].pop()
    elif mutation == "duplicate":
        coverage["classifications"][-1] = copy.deepcopy(coverage["classifications"][0])
    else:
        coverage["classifications"][-1]["epoch"] = END

    with pytest.raises(DojoSparseCoverageProofV3Error):
        build_coverage_proof_receipt(
            calendar_artifact=calendar,
            coverage_input=coverage,
        )


def test_observed_classification_must_match_exact_sealed_candle() -> None:
    calendar, coverage = _valid_inputs()
    coverage["classifications"][0]["observed_candle_sha256"] = coverage[
        "observed_candles"
    ][1]["candle_sha256"]

    with pytest.raises(
        DojoSparseCoverageProofV3Error,
        match="differs from its sealed candle",
    ):
        build_coverage_proof_receipt(
            calendar_artifact=calendar,
            coverage_input=coverage,
        )


def test_observed_candle_payload_tamper_is_rejected_before_classification() -> None:
    calendar, coverage = _valid_inputs()
    coverage["observed_candles"][0]["bid"][3] = 1.1006

    with pytest.raises(
        DojoSparseCoverageProofV3Error,
        match="observed candle content or seal differs",
    ):
        build_coverage_proof_receipt(
            calendar_artifact=calendar,
            coverage_input=coverage,
        )


@pytest.mark.parametrize("mutation", ["provider", "reason", "interval", "seal"])
def test_no_candle_receipt_provider_reason_interval_and_seal_are_bound(
    mutation: str,
) -> None:
    calendar, coverage = _valid_inputs()
    receipt = coverage["no_candle_receipts"][0]
    if mutation == "provider":
        receipt["provider"] = "OTHER_PROVIDER"
        coverage["no_candle_receipts"][0] = _sealed_body(receipt, "receipt_sha256")
    elif mutation == "reason":
        receipt["reason"] = "HOLIDAY_GUESS"
        coverage["no_candle_receipts"][0] = _sealed_body(receipt, "receipt_sha256")
    elif mutation == "interval":
        receipt["effective_from_epoch"] = START
        receipt["effective_to_epoch"] = START + 300
        coverage["no_candle_receipts"][0] = _sealed_body(receipt, "receipt_sha256")
    else:
        receipt["receipt_sha256"] = "0" * 64

    with pytest.raises(DojoSparseCoverageProofV3Error):
        build_coverage_proof_receipt(
            calendar_artifact=calendar,
            coverage_input=coverage,
        )


def test_no_candle_receipt_requires_a_distinct_independent_verifier() -> None:
    calendar, _coverage = _valid_inputs()

    with pytest.raises(
        DojoSparseCoverageProofV3Error,
        match="producer and independent verifier must differ",
    ):
        build_no_candle_receipt(
            provider=PROVIDER,
            pair="EUR_USD",
            effective_from_epoch=START,
            effective_to_epoch=START + 300,
            request_from_epoch=START,
            request_to_epoch=END,
            provider_request_sha256=SHA_A,
            provider_response_sha256=SHA_B,
            independent_verification_artifact_sha256=SHA_C,
            calendar_artifact_sha256=calendar["calendar_artifact_sha256"],
            source_producer_id=PRODUCER,
            independent_verifier_id=PRODUCER,
        )


def test_extra_unclassified_evidence_is_rejected() -> None:
    calendar, coverage = _valid_inputs()
    coverage["observed_candles"][1] = copy.deepcopy(coverage["observed_candles"][0])

    with pytest.raises(DojoSparseCoverageProofV3Error, match="duplicated"):
        build_coverage_proof_receipt(
            calendar_artifact=calendar,
            coverage_input=coverage,
        )


def test_numeric_schema_version_and_unsorted_evidence_are_not_canonical() -> None:
    calendar, coverage = _valid_inputs()
    coverage["schema_version"] = 3.0
    with pytest.raises(DojoSparseCoverageProofV3Error, match="identity differs"):
        build_coverage_proof_receipt(
            calendar_artifact=calendar,
            coverage_input=coverage,
        )

    calendar, coverage = _valid_inputs()
    coverage["observed_candles"].reverse()
    with pytest.raises(
        DojoSparseCoverageProofV3Error,
        match="canonical validated form",
    ):
        build_coverage_proof_receipt(
            calendar_artifact=calendar,
            coverage_input=coverage,
        )


def test_resealed_true_eligibility_claim_never_verifies() -> None:
    calendar, coverage = _valid_inputs()
    receipt = build_coverage_proof_receipt(
        calendar_artifact=calendar,
        coverage_input=coverage,
    )
    receipt["proof_eligible"] = True
    receipt = _sealed_body(receipt, "receipt_sha256")

    with pytest.raises(
        DojoSparseCoverageProofV3Error,
        match="independent reconstruction",
    ):
        verify_coverage_proof_receipt(
            receipt,
            calendar_artifact=calendar,
            coverage_input=coverage,
        )


def test_python_bool_integer_equality_cannot_bypass_canonical_verification() -> None:
    calendar, coverage = _valid_inputs()
    receipt = build_coverage_proof_receipt(
        calendar_artifact=calendar,
        coverage_input=coverage,
    )
    receipt["authority"]["live_permission"] = 0

    with pytest.raises(
        DojoSparseCoverageProofV3Error,
        match="independent reconstruction",
    ):
        verify_coverage_proof_receipt(
            receipt,
            calendar_artifact=calendar,
            coverage_input=coverage,
        )


def test_cli_seals_generates_verifies_and_recovers_idempotently(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run-dojo-sparse-coverage-proof-v3.py"
    )
    spec_path = tmp_path / "calendar-spec.json"
    calendar_path = tmp_path / "calendar.json"
    coverage_path = tmp_path / "coverage.json"
    receipt_path = tmp_path / "receipt.json"
    spec_path.write_text(json.dumps(_calendar_spec()), encoding="utf-8")

    seal = subprocess.run(
        [
            sys.executable,
            str(script),
            "seal-calendar",
            "--spec",
            str(spec_path),
            "--output",
            str(calendar_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert seal.returncode == 0, seal.stdout + seal.stderr
    calendar = json.loads(calendar_path.read_text(encoding="utf-8"))
    coverage_path.write_text(json.dumps(_coverage_input(calendar)), encoding="utf-8")
    generate_command = [
        sys.executable,
        str(script),
        "generate",
        "--calendar",
        str(calendar_path),
        "--coverage-input",
        str(coverage_path),
        "--output",
        str(receipt_path),
    ]
    generated = subprocess.run(
        generate_command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stdout + generated.stderr
    first_bytes = receipt_path.read_bytes()
    result = json.loads(generated.stdout)
    assert result["proof_eligible"] is False
    assert result["proof_classification"] == (
        "CANDIDATE_ONLY_UNAUTHENTICATED_SOURCE_REFERENCES"
    )
    assert result["authority"]["existing_generation_upgrade_allowed"] is False

    verified = subprocess.run(
        [
            sys.executable,
            str(script),
            "verify",
            "--calendar",
            str(calendar_path),
            "--coverage-input",
            str(coverage_path),
            "--receipt",
            str(receipt_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert json.loads(verified.stdout)["status"] == (
        "SPARSE_COVERAGE_DIAGNOSTIC_VERIFIED"
    )

    second = subprocess.run(
        generate_command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert json.loads(second.stdout)["status"] == "SPARSE_COVERAGE_DIAGNOSTIC_WRITTEN"
    assert receipt_path.read_bytes() == first_bytes


def test_cli_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run-dojo-sparse-coverage-proof-v3.py"
    )
    bad_spec = tmp_path / "bad-spec.json"
    bad_spec.write_text('{"contract":"x","contract":"y"}', encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "seal-calendar",
            "--spec",
            str(bad_spec),
            "--output",
            str(tmp_path / "unused.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["status"] == "BLOCKED"
    assert not (tmp_path / "unused.json").exists()


def test_write_once_accepts_same_bytes_but_rejects_different_bytes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "diagnostic.json"
    first = {"candidate_only": True}

    assert write_json_exclusive(output, first) == output
    first_bytes = output.read_bytes()
    assert write_json_exclusive(output, copy.deepcopy(first)) == output
    assert output.read_bytes() == first_bytes

    with pytest.raises(
        DojoSparseCoverageProofV3Error,
        match="already exists with different bytes",
    ):
        write_json_exclusive(output, {"candidate_only": False})

    assert output.read_bytes() == first_bytes


def test_write_once_rejects_retry_when_durable_anchor_is_missing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "diagnostic.json"
    value = {"candidate_only": True}
    assert write_json_exclusive(output, value) == output
    first_bytes = output.read_bytes()
    anchor = next(tmp_path.glob(".qr-coverage-*.pending"))
    anchor.unlink()

    with pytest.raises(
        DojoSparseCoverageProofV3Error,
        match="durable pending anchor is missing",
    ):
        write_json_exclusive(output, value)

    assert output.read_bytes() == first_bytes
    assert not anchor.exists()


def test_write_once_rejects_same_directory_replacement_without_deleting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "diagnostic.json"
    real_link = os.link
    replacement = b"forged replacement\n"
    swapped = False

    def swap_after_link(*args: Any, **kwargs: Any) -> None:
        nonlocal swapped
        real_link(*args, **kwargs)
        if not swapped and output.exists():
            swapped = True
            output.unlink()
            output.write_bytes(replacement)

    monkeypatch.setattr(coverage_module.os, "link", swap_after_link)

    with pytest.raises(
        DojoSparseCoverageProofV3Error,
        match="publication conflicts|final entry differs",
    ):
        write_json_exclusive(output, {"candidate_only": True})

    assert swapped is True
    assert output.read_bytes() == replacement


def test_write_once_rechecks_directory_entry_after_payload_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "diagnostic.json"
    real_fsync = os.fsync
    replacement = b"late forged replacement\n"
    swapped = False

    def swap_after_directory_fsync(descriptor: int) -> None:
        nonlocal swapped
        real_fsync(descriptor)
        if (
            not swapped
            and output.exists()
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
        ):
            swapped = True
            output.unlink()
            output.write_bytes(replacement)

    monkeypatch.setattr(
        coverage_module.os,
        "fsync",
        swap_after_directory_fsync,
    )

    with pytest.raises(
        DojoSparseCoverageProofV3Error,
        match="final entry differs|final pathname identity differs",
    ):
        write_json_exclusive(output, {"candidate_only": True})

    assert swapped is True
    assert output.read_bytes() == replacement


def test_write_once_recovers_after_crash_before_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "diagnostic.json"
    value = {"candidate_only": True}
    expected = canonical_json_bytes(value) + b"\n"
    real_link = os.link
    crashed = False

    def crash_before_link(*args: Any, **kwargs: Any) -> None:
        nonlocal crashed
        if not crashed:
            crashed = True
            raise OSError("simulated crash before publish")
        real_link(*args, **kwargs)

    monkeypatch.setattr(coverage_module.os, "link", crash_before_link)

    with pytest.raises(OSError, match="simulated crash before publish"):
        write_json_exclusive(output, value)

    pending = list(tmp_path.glob(".qr-coverage-*.pending"))
    assert crashed is True
    assert not output.exists()
    assert len(pending) == 1
    assert pending[0].read_bytes() == expected

    monkeypatch.setattr(coverage_module.os, "link", real_link)
    assert write_json_exclusive(output, value) == output
    assert output.read_bytes() == expected
    anchor = next(tmp_path.glob(".qr-coverage-*.pending"))
    assert anchor.read_bytes() == expected
    assert anchor.stat().st_ino == output.stat().st_ino


def test_write_once_repairs_partial_pending_after_write_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "diagnostic.json"
    value = {"candidate_only": True}
    expected = canonical_json_bytes(value) + b"\n"
    real_write = os.write
    crashed = False

    def partial_write_then_crash(descriptor: int, data: Any) -> int:
        nonlocal crashed
        if not crashed:
            crashed = True
            partial_size = max(1, len(data) // 2)
            real_write(descriptor, data[:partial_size])
            raise OSError("simulated crash during pending write")
        return real_write(descriptor, data)

    monkeypatch.setattr(coverage_module.os, "write", partial_write_then_crash)
    with pytest.raises(OSError, match="simulated crash during pending write"):
        write_json_exclusive(output, value)

    pending = next(tmp_path.glob(".qr-coverage-*.pending"))
    assert crashed is True
    assert not output.exists()
    assert 0 < pending.stat().st_size < len(expected)

    monkeypatch.setattr(coverage_module.os, "write", real_write)
    assert write_json_exclusive(output, value) == output
    assert output.read_bytes() == expected
    anchor = next(tmp_path.glob(".qr-coverage-*.pending"))
    assert anchor.read_bytes() == expected
    assert anchor.stat().st_ino == output.stat().st_ino


def test_write_once_recovers_after_crash_before_publish_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "diagnostic.json"
    value = {"candidate_only": True}
    expected = canonical_json_bytes(value) + b"\n"
    real_fsync = os.fsync
    crashed = False

    def crash_before_directory_fsync(descriptor: int) -> None:
        nonlocal crashed
        if (
            not crashed
            and output.exists()
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
        ):
            crashed = True
            raise OSError("simulated crash before directory fsync")
        real_fsync(descriptor)

    monkeypatch.setattr(
        coverage_module.os,
        "fsync",
        crash_before_directory_fsync,
    )

    with pytest.raises(OSError, match="simulated crash before directory fsync"):
        write_json_exclusive(output, value)

    assert crashed is True
    assert output.read_bytes() == expected
    assert len(list(tmp_path.glob(".qr-coverage-*.pending"))) == 1

    retry_directory_fsyncs = 0

    def count_retry_directory_fsync(descriptor: int) -> None:
        nonlocal retry_directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            retry_directory_fsyncs += 1
        real_fsync(descriptor)

    monkeypatch.setattr(
        coverage_module.os,
        "fsync",
        count_retry_directory_fsync,
    )
    assert write_json_exclusive(output, value) == output
    assert retry_directory_fsyncs == 1
    assert output.read_bytes() == expected
    anchor = next(tmp_path.glob(".qr-coverage-*.pending"))
    assert anchor.read_bytes() == expected
    assert anchor.stat().st_ino == output.stat().st_ino


def test_durable_pending_anchor_is_never_unlinked_and_replacement_survives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "diagnostic.json"
    value = {"candidate_only": True}
    assert write_json_exclusive(output, value) == output
    pending = next(tmp_path.glob(".qr-coverage-*.pending"))
    displaced = tmp_path / ".original-pending-displaced"
    replacement_bytes = b"replacement must survive\n"
    pending.rename(displaced)
    pending.write_bytes(replacement_bytes)

    def forbid_unlink(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            f"durable anchor must not be unlinked: {args!r} {kwargs!r}"
        )

    monkeypatch.setattr(coverage_module.os, "unlink", forbid_unlink)
    with pytest.raises(
        DojoSparseCoverageProofV3Error,
        match="pending",
    ):
        write_json_exclusive(output, value)

    assert pending.read_bytes() == replacement_bytes
    assert displaced.read_bytes() == output.read_bytes()


def test_no_candle_reason_is_fixed_to_provider_response_omission() -> None:
    assert NO_CANDLE_REASON == "COMPLETE_PROVIDER_RESPONSE_OMITTED_CANDLE"

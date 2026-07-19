from __future__ import annotations

import copy
import math

import pytest

from quant_rabbit.dojo_conveyor_resource_gate import (
    BACKPRESSURE,
    DISK_RESERVE_BYTES,
    REQUEST_CONTRACT,
    REVIEW_REQUIRED,
    START_ALLOWED,
    DojoConveyorResourceGateError,
    canonical_sha256,
    plan_conveyor_start,
    required_free_bytes,
    verify_resource_gate_decision,
)


def _request() -> dict:
    predicted = 1_000_000_003
    archive_temp = 2_000_000_000
    minimum = required_free_bytes(predicted, archive_temp)
    return {
        "contract": REQUEST_CONTRACT,
        "schema_version": 1,
        "prior_result": {
            "terminal": True,
            "result_bound": True,
            "result_binding_sha256": "1" * 64,
            "lineage_tip_sha256": "2" * 64,
        },
        "drive_compact_report": {
            "remote_verified": True,
            "report_sha256": "3" * 64,
            "metadata_receipt_sha256": "4" * 64,
        },
        "resources": {
            "active_trainer_count": 0,
            "remote_unverified_generation_count": 0,
            "compression_upload_active_count": 0,
            "free_bytes": minimum,
            "predicted_next_output_bytes": predicted,
            "archive_temp_bytes": archive_temp,
        },
        "fixed_bindings": {
            "corpus_sha256": "5" * 64,
            "scorer_sha256": "6" * 64,
            "study_sha256": "7" * 64,
        },
        "authority": {
            "model_invocation_allowed": False,
            "runner_invocation_allowed": False,
            "filesystem_write_allowed": False,
            "filesystem_delete_allowed": False,
            "broker_mutation_allowed": False,
            "live_permission": False,
            "order_authority": "NONE",
        },
    }


def test_exact_boundary_allows_next_sealed_run_without_granting_authority() -> None:
    request = _request()

    decision = plan_conveyor_start(request)

    assert decision["status"] == START_ALLOWED
    assert decision["review_required_reasons"] == []
    assert decision["backpressure_reasons"] == []
    assert decision["required_free_bytes"] == request["resources"]["free_bytes"]
    assert decision["free_bytes_headroom"] == 0
    assert all(decision["checks"].values())
    assert decision["authority"] == request["authority"]
    assert all(
        value is False or value == "NONE" for value in decision["authority"].values()
    )
    assert "START_ALLOWED_GRANTS_NO_RUNNER_AUTHORITY" in decision["limitations"]
    assert decision["request_sha256"] == canonical_sha256(request)
    body = {key: value for key, value in decision.items() if key != "decision_sha256"}
    assert decision["decision_sha256"] == canonical_sha256(body)
    assert verify_resource_gate_decision(decision, request) == decision


def test_required_disk_uses_integer_ceiling_without_float_rounding() -> None:
    assert required_free_bytes(0, 0) == DISK_RESERVE_BYTES
    assert required_free_bytes(1, 0) == DISK_RESERVE_BYTES + 2
    assert required_free_bytes(4, 9) == DISK_RESERVE_BYTES + 5 + 9
    huge = 2**80 + 1
    assert required_free_bytes(huge, 7) == (5 * huge + 3) // 4 + 7 + DISK_RESERVE_BYTES


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("active_trainer_count", "ACTIVE_TRAINER_PRESENT"),
        (
            "remote_unverified_generation_count",
            "REMOTE_UNVERIFIED_GENERATION_PRESENT",
        ),
        ("compression_upload_active_count", "COMPRESSION_UPLOAD_ACTIVE"),
    ],
)
def test_occupied_conveyor_resource_applies_backpressure(
    field: str, reason: str
) -> None:
    request = _request()
    request["resources"][field] = 1

    decision = plan_conveyor_start(request)

    assert decision["status"] == BACKPRESSURE
    assert decision["review_required_reasons"] == []
    assert decision["backpressure_reasons"] == [reason]


def test_one_byte_below_disk_floor_applies_backpressure() -> None:
    request = _request()
    request["resources"]["free_bytes"] -= 1

    decision = plan_conveyor_start(request)

    assert decision["status"] == BACKPRESSURE
    assert decision["backpressure_reasons"] == ["INSUFFICIENT_FREE_BYTES"]
    assert decision["free_bytes_headroom"] == -1
    assert decision["checks"]["free_bytes_sufficient"] is False


@pytest.mark.parametrize(
    ("section", "field", "reason"),
    [
        ("prior_result", "terminal", "PRIOR_RESULT_NOT_TERMINAL"),
        ("prior_result", "result_bound", "PRIOR_RESULT_NOT_BOUND"),
        (
            "drive_compact_report",
            "remote_verified",
            "DRIVE_COMPACT_REPORT_NOT_REMOTE_VERIFIED",
        ),
    ],
)
def test_unsealed_or_unverified_handoff_requires_review(
    section: str, field: str, reason: str
) -> None:
    request = _request()
    request[section][field] = False

    decision = plan_conveyor_start(request)

    assert decision["status"] == REVIEW_REQUIRED
    assert decision["review_required_reasons"] == [reason]


@pytest.mark.parametrize(
    ("section", "field", "reason"),
    [
        (
            "prior_result",
            "result_binding_sha256",
            "INVALID_PRIOR_RESULT_BINDING_SHA256",
        ),
        ("prior_result", "lineage_tip_sha256", "INVALID_LINEAGE_TIP_SHA256"),
        (
            "drive_compact_report",
            "report_sha256",
            "INVALID_DRIVE_COMPACT_REPORT_SHA256",
        ),
        (
            "drive_compact_report",
            "metadata_receipt_sha256",
            "INVALID_DRIVE_METADATA_RECEIPT_SHA256",
        ),
        ("fixed_bindings", "corpus_sha256", "INVALID_CORPUS_SHA256"),
        ("fixed_bindings", "scorer_sha256", "INVALID_SCORER_SHA256"),
        ("fixed_bindings", "study_sha256", "INVALID_STUDY_SHA256"),
    ],
)
@pytest.mark.parametrize("invalid", ["", "a" * 63, "A" * 64, "g" * 64])
def test_every_binding_requires_lowercase_64_hex(
    section: str, field: str, reason: str, invalid: str
) -> None:
    request = _request()
    request[section][field] = invalid

    decision = plan_conveyor_start(request)

    assert decision["status"] == REVIEW_REQUIRED
    assert decision["review_required_reasons"] == [reason]
    assert decision["checks"]["all_binding_sha256_valid"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_invocation_allowed", True),
        ("runner_invocation_allowed", True),
        ("filesystem_write_allowed", True),
        ("filesystem_delete_allowed", True),
        ("broker_mutation_allowed", True),
        ("live_permission", True),
        ("order_authority", "START"),
    ],
)
def test_authority_claims_require_review_and_are_never_echoed(
    field: str, value: object
) -> None:
    request = _request()
    request["authority"][field] = value

    decision = plan_conveyor_start(request)

    assert decision["status"] == REVIEW_REQUIRED
    assert decision["review_required_reasons"] == ["AUTHORITY_NOT_RESEARCH_ONLY"]
    assert decision["checks"]["authority_research_only"] is False
    assert (
        decision["authority"][field] is False or decision["authority"][field] == "NONE"
    )


def test_review_required_takes_precedence_but_preserves_backpressure_diagnosis() -> (
    None
):
    request = _request()
    request["prior_result"]["result_bound"] = False
    request["resources"]["active_trainer_count"] = 2

    decision = plan_conveyor_start(request)

    assert decision["status"] == REVIEW_REQUIRED
    assert decision["review_required_reasons"] == ["PRIOR_RESULT_NOT_BOUND"]
    assert decision["backpressure_reasons"] == ["ACTIVE_TRAINER_PRESENT"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request.update({"unknown": None}),
        lambda request: request["prior_result"].update({"unknown": None}),
        lambda request: request["resources"].pop("free_bytes"),
        lambda request: request["fixed_bindings"].update({"unknown": "x"}),
        lambda request: request["authority"].pop("live_permission"),
    ],
)
def test_exact_schema_rejects_unknown_or_missing_fields(mutation) -> None:
    request = _request()
    mutation(request)

    with pytest.raises(DojoConveyorResourceGateError, match="keys mismatch"):
        plan_conveyor_start(request)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("active_trainer_count", True),
        ("remote_unverified_generation_count", False),
        ("compression_upload_active_count", 1.0),
        ("free_bytes", -1),
        ("predicted_next_output_bytes", -1),
        ("archive_temp_bytes", math.nan),
        ("archive_temp_bytes", math.inf),
    ],
)
def test_resource_numbers_reject_bool_float_nan_infinity_and_negative(
    field: str, value: object
) -> None:
    request = _request()
    request["resources"][field] = value

    with pytest.raises(DojoConveyorResourceGateError, match="non-negative integer"):
        plan_conveyor_start(request)


def test_boolean_schema_version_is_not_integer_one() -> None:
    request = _request()
    request["schema_version"] = True

    with pytest.raises(DojoConveyorResourceGateError, match="integer 1"):
        plan_conveyor_start(request)


def test_boolean_evidence_fields_are_exact_booleans() -> None:
    request = _request()
    request["prior_result"]["terminal"] = 1

    with pytest.raises(DojoConveyorResourceGateError, match="boolean"):
        plan_conveyor_start(request)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_canonical_sha_rejects_nonfinite_numbers(bad: float) -> None:
    with pytest.raises(DojoConveyorResourceGateError, match="finite"):
        canonical_sha256({"bad": bad})


def test_canonical_sha_is_independent_of_mapping_order() -> None:
    first = {"a": 1, "b": {"c": 2}}
    second = {"b": {"c": 2}, "a": 1}

    assert canonical_sha256(first) == canonical_sha256(second)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda decision: decision.update({"status": START_ALLOWED}),
        lambda decision: decision["checks"].update({"free_bytes_sufficient": True}),
        lambda decision: decision["authority"].update(
            {"runner_invocation_allowed": True}
        ),
        lambda decision: decision.update({"unknown": None}),
    ],
)
def test_verifier_rejects_rehashed_or_unhashed_tampering(mutate) -> None:
    request = _request()
    request["resources"]["free_bytes"] -= 1
    decision = copy.deepcopy(plan_conveyor_start(request))
    mutate(decision)
    if "unknown" not in decision:
        body = {
            key: value for key, value in decision.items() if key != "decision_sha256"
        }
        decision["decision_sha256"] = canonical_sha256(body)

    with pytest.raises(DojoConveyorResourceGateError):
        verify_resource_gate_decision(decision, request)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda decision: decision.update({"free_bytes_headroom": False}),
        lambda decision: decision["authority"].update({"runner_invocation_allowed": 0}),
    ],
)
def test_verifier_does_not_treat_boolean_as_equal_integer(mutate) -> None:
    request = _request()
    decision = copy.deepcopy(plan_conveyor_start(request))
    mutate(decision)
    body = {key: value for key, value in decision.items() if key != "decision_sha256"}
    decision["decision_sha256"] = canonical_sha256(body)

    with pytest.raises(DojoConveyorResourceGateError):
        verify_resource_gate_decision(decision, request)


def test_request_is_copied_before_return_and_decision_is_deterministic() -> None:
    request = _request()
    expected = plan_conveyor_start(request)
    reordered = dict(reversed(list(request.items())))

    assert plan_conveyor_start(reordered) == expected
    request["resources"]["free_bytes"] = 0
    assert expected["status"] == START_ALLOWED

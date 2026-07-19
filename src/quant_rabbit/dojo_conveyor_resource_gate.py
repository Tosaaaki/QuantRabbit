"""Pure, fail-closed resource gate for the DOJO research conveyor.

The gate decides whether a *separately authorised* orchestrator may consider
starting the next sealed TRAIN run.  It does not read the filesystem, call a
model, start a runner, mutate broker state, or grant any of those authorities.

Evidence defects require human/agent review.  Healthy evidence with occupied
workers, an unfinished Drive handoff, or insufficient disk creates ordinary
backpressure.  ``START_ALLOWED`` is therefore a scheduling recommendation,
never execution or live-trading permission.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any, Final


REQUEST_CONTRACT: Final = "QR_DOJO_CONVEYOR_RESOURCE_GATE_REQUEST_V1"
DECISION_CONTRACT: Final = "QR_DOJO_CONVEYOR_RESOURCE_GATE_DECISION_V1"
SCHEMA_VERSION: Final = 1

DISK_RESERVE_BYTES: Final = 20 * 1024**3
PREDICTED_OUTPUT_NUMERATOR: Final = 5
PREDICTED_OUTPUT_DENOMINATOR: Final = 4

START_ALLOWED: Final = "START_ALLOWED"
BACKPRESSURE: Final = "BACKPRESSURE"
REVIEW_REQUIRED: Final = "REVIEW_REQUIRED"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_REQUEST_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "prior_result",
        "drive_compact_report",
        "resources",
        "fixed_bindings",
        "authority",
    }
)
_PRIOR_RESULT_KEYS = frozenset(
    {
        "terminal",
        "result_bound",
        "result_binding_sha256",
        "lineage_tip_sha256",
    }
)
_DRIVE_REPORT_KEYS = frozenset(
    {
        "remote_verified",
        "report_sha256",
        "metadata_receipt_sha256",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "active_trainer_count",
        "remote_unverified_generation_count",
        "compression_upload_active_count",
        "free_bytes",
        "predicted_next_output_bytes",
        "archive_temp_bytes",
    }
)
_FIXED_BINDING_KEYS = frozenset(
    {
        "corpus_sha256",
        "scorer_sha256",
        "study_sha256",
    }
)
_AUTHORITY_KEYS = frozenset(
    {
        "model_invocation_allowed",
        "runner_invocation_allowed",
        "filesystem_write_allowed",
        "filesystem_delete_allowed",
        "broker_mutation_allowed",
        "live_permission",
        "order_authority",
    }
)
_DECISION_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "request_sha256",
        "status",
        "review_required_reasons",
        "backpressure_reasons",
        "required_free_bytes",
        "observed_free_bytes",
        "free_bytes_headroom",
        "checks",
        "authority",
        "limitations",
        "decision_sha256",
    }
)
_CHECK_KEYS = frozenset(
    {
        "terminal_prior_result",
        "prior_result_bound",
        "drive_compact_report_remote_verified",
        "all_binding_sha256_valid",
        "authority_research_only",
        "active_trainer_zero",
        "remote_unverified_generation_zero",
        "compression_upload_idle",
        "free_bytes_sufficient",
    }
)

_NO_AUTHORITY: Final = {
    "model_invocation_allowed": False,
    "runner_invocation_allowed": False,
    "filesystem_write_allowed": False,
    "filesystem_delete_allowed": False,
    "broker_mutation_allowed": False,
    "live_permission": False,
    "order_authority": "NONE",
}
_LIMITATIONS: Final = (
    "PURE_PLANNER_NO_SIDE_EFFECTS",
    "START_ALLOWED_GRANTS_NO_RUNNER_AUTHORITY",
    "NO_MODEL_OR_BROKER_AUTHORITY",
    "NO_FILESYSTEM_WRITE_OR_DELETE_AUTHORITY",
)


class DojoConveyorResourceGateError(ValueError):
    """The resource-gate request or decision is structurally invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict canonical JSON bytes for a finite JSON value."""

    _validate_json(value, "value")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoConveyorResourceGateError(
            f"value is not canonical JSON: {exc}"
        ) from exc


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def required_free_bytes(
    predicted_next_output_bytes: int,
    archive_temp_bytes: int,
) -> int:
    """Return ``ceil(1.25 * predicted) + archive temp + 20 GiB`` exactly."""

    predicted = _nonnegative_int(
        predicted_next_output_bytes, "predicted_next_output_bytes"
    )
    archive_temp = _nonnegative_int(archive_temp_bytes, "archive_temp_bytes")
    scaled_prediction = (
        PREDICTED_OUTPUT_NUMERATOR * predicted + PREDICTED_OUTPUT_DENOMINATOR - 1
    ) // PREDICTED_OUTPUT_DENOMINATOR
    return scaled_prediction + archive_temp + DISK_RESERVE_BYTES


def plan_conveyor_start(request: Mapping[str, Any]) -> dict[str, Any]:
    """Build a sealed, side-effect-free decision for the next TRAIN run."""

    normalized = _validate_request(request)
    prior = normalized["prior_result"]
    drive = normalized["drive_compact_report"]
    resources = normalized["resources"]
    bindings = normalized["fixed_bindings"]
    authority = normalized["authority"]

    binding_checks = (
        ("INVALID_PRIOR_RESULT_BINDING_SHA256", prior["result_binding_sha256"]),
        ("INVALID_LINEAGE_TIP_SHA256", prior["lineage_tip_sha256"]),
        ("INVALID_DRIVE_COMPACT_REPORT_SHA256", drive["report_sha256"]),
        (
            "INVALID_DRIVE_METADATA_RECEIPT_SHA256",
            drive["metadata_receipt_sha256"],
        ),
        ("INVALID_CORPUS_SHA256", bindings["corpus_sha256"]),
        ("INVALID_SCORER_SHA256", bindings["scorer_sha256"]),
        ("INVALID_STUDY_SHA256", bindings["study_sha256"]),
    )
    invalid_binding_reasons = [
        reason for reason, value in binding_checks if not _is_sha256(value)
    ]
    authority_research_only = authority == _NO_AUTHORITY

    review_reasons: list[str] = []
    if not prior["terminal"]:
        review_reasons.append("PRIOR_RESULT_NOT_TERMINAL")
    if not prior["result_bound"]:
        review_reasons.append("PRIOR_RESULT_NOT_BOUND")
    if not drive["remote_verified"]:
        review_reasons.append("DRIVE_COMPACT_REPORT_NOT_REMOTE_VERIFIED")
    review_reasons.extend(invalid_binding_reasons)
    if not authority_research_only:
        review_reasons.append("AUTHORITY_NOT_RESEARCH_ONLY")

    minimum_free = required_free_bytes(
        resources["predicted_next_output_bytes"],
        resources["archive_temp_bytes"],
    )
    backpressure_reasons: list[str] = []
    if resources["active_trainer_count"] > 0:
        backpressure_reasons.append("ACTIVE_TRAINER_PRESENT")
    if resources["remote_unverified_generation_count"] > 0:
        backpressure_reasons.append("REMOTE_UNVERIFIED_GENERATION_PRESENT")
    if resources["compression_upload_active_count"] > 0:
        backpressure_reasons.append("COMPRESSION_UPLOAD_ACTIVE")
    if resources["free_bytes"] < minimum_free:
        backpressure_reasons.append("INSUFFICIENT_FREE_BYTES")

    if review_reasons:
        status = REVIEW_REQUIRED
    elif backpressure_reasons:
        status = BACKPRESSURE
    else:
        status = START_ALLOWED

    checks = {
        "terminal_prior_result": prior["terminal"],
        "prior_result_bound": prior["result_bound"],
        "drive_compact_report_remote_verified": drive["remote_verified"],
        "all_binding_sha256_valid": not invalid_binding_reasons,
        "authority_research_only": authority_research_only,
        "active_trainer_zero": resources["active_trainer_count"] == 0,
        "remote_unverified_generation_zero": resources[
            "remote_unverified_generation_count"
        ]
        == 0,
        "compression_upload_idle": resources["compression_upload_active_count"] == 0,
        "free_bytes_sufficient": resources["free_bytes"] >= minimum_free,
    }
    body = {
        "contract": DECISION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "request_sha256": canonical_sha256(normalized),
        "status": status,
        "review_required_reasons": review_reasons,
        "backpressure_reasons": backpressure_reasons,
        "required_free_bytes": minimum_free,
        "observed_free_bytes": resources["free_bytes"],
        "free_bytes_headroom": resources["free_bytes"] - minimum_free,
        "checks": checks,
        "authority": dict(_NO_AUTHORITY),
        "limitations": list(_LIMITATIONS),
    }
    return {**body, "decision_sha256": canonical_sha256(body)}


def verify_resource_gate_decision(
    decision: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify exact schema, request binding, semantics, and decision seal."""

    normalized_decision = _mapping(decision, "decision")
    _exact_keys(normalized_decision, _DECISION_KEYS, "decision")
    _literal(normalized_decision["contract"], DECISION_CONTRACT, "decision.contract")
    _schema_version(normalized_decision["schema_version"], "decision.schema_version")
    _sha_field(normalized_decision["request_sha256"], "decision.request_sha256")
    _sha_field(normalized_decision["decision_sha256"], "decision.decision_sha256")
    _status(normalized_decision["status"], "decision.status")
    _string_list(
        normalized_decision["review_required_reasons"],
        "decision.review_required_reasons",
    )
    _string_list(
        normalized_decision["backpressure_reasons"],
        "decision.backpressure_reasons",
    )
    _nonnegative_int(
        normalized_decision["required_free_bytes"],
        "decision.required_free_bytes",
    )
    _nonnegative_int(
        normalized_decision["observed_free_bytes"],
        "decision.observed_free_bytes",
    )
    _integer(
        normalized_decision["free_bytes_headroom"],
        "decision.free_bytes_headroom",
    )
    _string_list(normalized_decision["limitations"], "decision.limitations")
    checks = _mapping(normalized_decision["checks"], "decision.checks")
    _exact_keys(checks, _CHECK_KEYS, "decision.checks")
    for key, value in checks.items():
        _boolean(value, f"decision.checks.{key}")
    output_authority = _mapping(normalized_decision["authority"], "decision.authority")
    _exact_keys(output_authority, _AUTHORITY_KEYS, "decision.authority")
    for key in _AUTHORITY_KEYS - {"order_authority"}:
        _boolean(output_authority[key], f"decision.authority.{key}")
    _string(
        output_authority["order_authority"],
        "decision.authority.order_authority",
    )
    if output_authority != _NO_AUTHORITY:
        raise DojoConveyorResourceGateError(
            "decision.authority must grant no authority"
        )

    expected = plan_conveyor_start(request)
    if normalized_decision != expected:
        raise DojoConveyorResourceGateError(
            "decision does not match the canonical request-derived decision"
        )
    body = {
        key: value
        for key, value in normalized_decision.items()
        if key != "decision_sha256"
    }
    if normalized_decision["decision_sha256"] != canonical_sha256(body):
        raise DojoConveyorResourceGateError("decision_sha256 mismatch")
    return dict(normalized_decision)


# Descriptive aliases for callers which use build/evaluate vocabulary.
build_resource_gate_plan = plan_conveyor_start
evaluate_resource_gate = plan_conveyor_start
plan_next_run_start = plan_conveyor_start


def _validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(request, "request")
    _exact_keys(value, _REQUEST_KEYS, "request")
    _literal(value["contract"], REQUEST_CONTRACT, "request.contract")
    _schema_version(value["schema_version"], "request.schema_version")

    prior = _mapping(value["prior_result"], "request.prior_result")
    _exact_keys(prior, _PRIOR_RESULT_KEYS, "request.prior_result")
    _boolean(prior["terminal"], "request.prior_result.terminal")
    _boolean(prior["result_bound"], "request.prior_result.result_bound")
    _string(
        prior["result_binding_sha256"], "request.prior_result.result_binding_sha256"
    )
    _string(prior["lineage_tip_sha256"], "request.prior_result.lineage_tip_sha256")

    drive = _mapping(value["drive_compact_report"], "request.drive_compact_report")
    _exact_keys(drive, _DRIVE_REPORT_KEYS, "request.drive_compact_report")
    _boolean(
        drive["remote_verified"],
        "request.drive_compact_report.remote_verified",
    )
    _string(drive["report_sha256"], "request.drive_compact_report.report_sha256")
    _string(
        drive["metadata_receipt_sha256"],
        "request.drive_compact_report.metadata_receipt_sha256",
    )

    resources = _mapping(value["resources"], "request.resources")
    _exact_keys(resources, _RESOURCE_KEYS, "request.resources")
    for key, item in resources.items():
        _nonnegative_int(item, f"request.resources.{key}")

    bindings = _mapping(value["fixed_bindings"], "request.fixed_bindings")
    _exact_keys(bindings, _FIXED_BINDING_KEYS, "request.fixed_bindings")
    for key, item in bindings.items():
        _string(item, f"request.fixed_bindings.{key}")

    authority = _mapping(value["authority"], "request.authority")
    _exact_keys(authority, _AUTHORITY_KEYS, "request.authority")
    for key in _AUTHORITY_KEYS - {"order_authority"}:
        _boolean(authority[key], f"request.authority.{key}")
    _string(authority["order_authority"], "request.authority.order_authority")

    # A canonical round trip produces ordinary dict/list primitives and severs
    # references to caller-owned mutable containers without filesystem I/O.
    _validate_json(value, "request")
    return json.loads(canonical_json_bytes(value))


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoConveyorResourceGateError(f"{field} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], field: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise DojoConveyorResourceGateError(f"{field} keys must be strings")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise DojoConveyorResourceGateError(
            f"{field} keys mismatch: missing={missing}, unknown={unknown}"
        )


def _literal(value: Any, expected: str, field: str) -> None:
    if not isinstance(value, str) or value != expected:
        raise DojoConveyorResourceGateError(f"{field} must be {expected!r}")


def _schema_version(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != SCHEMA_VERSION:
        raise DojoConveyorResourceGateError(f"{field} must be integer {SCHEMA_VERSION}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise DojoConveyorResourceGateError(f"{field} must be a boolean")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise DojoConveyorResourceGateError(f"{field} must be a string")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DojoConveyorResourceGateError(f"{field} must be a non-negative integer")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DojoConveyorResourceGateError(f"{field} must be an integer")
    return value


def _status(value: Any, field: str) -> str:
    status = _string(value, field)
    if status not in {START_ALLOWED, BACKPRESSURE, REVIEW_REQUIRED}:
        raise DojoConveyorResourceGateError(f"{field} is not a known status")
    return status


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise DojoConveyorResourceGateError(f"{field} must be an array")
    for index, item in enumerate(value):
        _string(item, f"{field}[{index}]")
    return value


def _sha_field(value: Any, field: str) -> str:
    if not _is_sha256(value):
        raise DojoConveyorResourceGateError(f"{field} must be lowercase 64-hex SHA-256")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_json(value: Any, field: str, *, depth: int = 0) -> None:
    if depth > 64:
        raise DojoConveyorResourceGateError(f"{field} exceeds JSON depth limit")
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DojoConveyorResourceGateError(f"{field} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{field}[{index}]", depth=depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DojoConveyorResourceGateError(
                    f"{field} object keys must be strings"
                )
            _validate_json(item, f"{field}.{key}", depth=depth + 1)
        return
    raise DojoConveyorResourceGateError(
        f"{field} contains unsupported JSON type {type(value).__name__}"
    )

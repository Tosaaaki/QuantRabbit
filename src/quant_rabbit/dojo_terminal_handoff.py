"""Fail-closed terminal hand-off for the DOJO research conveyor.

The coordinator admits a trainer result only as one inseparable terminal
bundle: ``run.json``, ``evaluation.json``, and ``cells.json``.  It rejects an
adjacent ``run_failure.json``, replays the fixed denominator and evaluation
through :mod:`dojo_ai_trainer_packet`, and only then appends a local candidate
lineage ``RESULT_BOUND`` event.

There are two explicit timing classifications.  A result whose lineage study
already exists at hand-off is
``LINEAGE_PRESENT_AT_HANDOFF_NO_PREREGISTRATION_CLAIM``.  This is a
presence-only observation and does not claim the study existed before run
start.  A completed run
registered after it started must be invoked as
``RETROSPECTIVE_ADMIN_BINDING``; that classification is preserved even when a
retry observes a partially-created or already-bound lineage.  Neither class is
proof or promotion evidence.  In particular, retrospective administration can
never manufacture pre-registration.

The hand-off receipt store is a local first-write SHA chain.  Each receipt is
written with ``O_EXCL``, file ``fsync``, and directory ``fsync`` under a stable
advisory lock.  It remains local diagnostic evidence: the filesystem owner can
still delete and rebuild the whole store, so there is no external monotonic
witness and no model, runner, Drive, broker, order, or live authority.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Final

from quant_rabbit.dojo_ai_trainer_packet import (
    MAX_BOUND_ARTIFACT_BYTES,
    DojoAITrainerPacketError,
    validate_terminal_result_bundle,
)
from quant_rabbit.dojo_bot_trainer import DojoBotTrainerError, verify_sealed_study
from quant_rabbit.dojo_candidate_lineage_registry import (
    CandidateLineageError,
    CandidateLineageSnapshot,
    bind_result,
    initialize_registry,
    seal_study_attempt,
    verify_registry,
)


RECEIPT_CONTRACT: Final = "QR_DOJO_TERMINAL_HANDOFF_RECEIPT_V1"
STORE_STATUS_CONTRACT: Final = "QR_DOJO_TERMINAL_HANDOFF_STORE_STATUS_V1"
SCHEMA_VERSION: Final = 1

LINEAGE_PRESENT_AT_HANDOFF: Final = (
    "LINEAGE_PRESENT_AT_HANDOFF_NO_PREREGISTRATION_CLAIM"
)
RETROSPECTIVE_ADMIN_BINDING: Final = "RETROSPECTIVE_ADMIN_BINDING"
ABSENT_CAS_TOKEN: Final = "ABSENT"

MAX_RECEIPTS: Final = 64
MAX_RECEIPT_BYTES: Final = 512 * 1024
_RECEIPT_NAME = re.compile(r"[0-9]{6}\.json\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}\Z")

_AUTHORITY: Final = {
    "proof_eligible": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
}
_LIMITATIONS: Final = (
    "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
    "LOCAL_FIRST_WRITE_ONLY_NO_EXTERNAL_MONOTONIC_WITNESS",
    "RETROSPECTIVE_ADMIN_BINDING_NEVER_PROVES_PREREGISTRATION",
    "LINEAGE_PRESENT_AT_HANDOFF_DOES_NOT_PROVE_PREREGISTRATION",
    "NO_MODEL_RUNNER_DRIVE_BROKER_ORDER_OR_LIVE_AUTHORITY",
    "FULL_TERMINAL_BUNDLE_REQUIRED_EVALUATION_ONLY_BINDING_FORBIDDEN",
)

_RECEIPT_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "sequence",
        "recorded_at_utc",
        "previous_receipt_sha256",
        "binding_timing_classification",
        "lineage_branch",
        "terminal_bundle",
        "sealed_study",
        "lineage_before",
        "lineage_after",
        "checks",
        "limitations",
        *_AUTHORITY,
        "receipt_sha256",
    }
)
_ARTIFACT_KEYS = frozenset(
    {"artifact_relpath", "artifact_sha256", "artifact_size_bytes"}
)
_TERMINAL_KEYS = frozenset(
    {
        "terminal_dir_relpath",
        "run",
        "evaluation",
        "cells",
        "run_sha256",
        "evaluation_sha256",
        "study_sha256",
        "status",
        "expected_cell_count",
        "observed_cell_count",
        "failed_cell_count",
    }
)
_STUDY_KEYS = frozenset(
    {
        "artifact_relpath",
        "artifact_sha256",
        "artifact_size_bytes",
        "study_sha256",
        "attempt_ordinal",
    }
)
_LINEAGE_BEFORE_KEYS = frozenset(
    {
        "state",
        "registry_id",
        "lineage_prefix",
        "latest_sequence",
        "latest_event_sha256",
        "study_count",
        "result_count",
    }
)
_LINEAGE_AFTER_KEYS = frozenset(
    {
        "registry_id",
        "lineage_prefix",
        "latest_sequence",
        "latest_event_sha256",
        "attempt_ordinal",
        "study_sha256",
        "evaluation_sha256",
        "evaluation_artifact_sha256",
        "evaluation_artifact_size_bytes",
        "result_event_sha256",
        "result_event_sequence",
        "result_binding_sha256",
    }
)
_CHECK_KEYS = frozenset(
    {
        "terminal_bundle_complete",
        "run_failure_absent",
        "terminal_status_valid",
        "fixed_denominator_complete",
        "study_sha_consistent",
        "authority_research_only",
        "evaluation_rebuilt_from_cells",
        "full_terminal_bundle_required",
        "lineage_result_bound",
    }
)
_LINEAGE_BRANCHES = frozenset(
    {
        "UNINITIALIZED_LINEAGE_CREATED_SEALED_AND_BOUND",
        "EXISTING_GENESIS_SEALED_AND_BOUND",
        "EXISTING_PENDING_STUDY_BOUND",
        "EXISTING_EXACT_RESULT_REUSED",
    }
)


class DojoTerminalHandoffError(ValueError):
    """The terminal bundle, lineage branch, CAS token, or receipt is unsafe."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict deterministic JSON bytes without a trailing newline."""

    _validate_json(value, "value")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoTerminalHandoffError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def coordinate_terminal_handoff(
    *,
    terminal_dir: Path,
    sealed_study_path: Path,
    lineage_events_dir: Path,
    artifact_root: Path,
    receipt_events_dir: Path,
    expected_lineage_tip_sha256: str,
    expected_receipt_tip_sha256: str,
    binding_timing_classification: str,
    event_at_utc: datetime | str,
    registry_id: str | None = None,
    lineage_prefix: str | None = None,
    created_by: str | None = None,
) -> dict[str, Any]:
    """Validate, lineage-bind, and receipt one complete terminal run.

    ``expected_*_tip_sha256`` is either an exact lowercase SHA-256 or the
    literal ``ABSENT``.  The latter is valid only when the corresponding store
    has no events.  Lineage initialization and study sealing are allowed only
    under the explicit retrospective classification.
    """

    classification = _classification(binding_timing_classification)
    lineage_expected = _cas_token(
        expected_lineage_tip_sha256, "expected_lineage_tip_sha256"
    )
    receipt_expected = _cas_token(
        expected_receipt_tip_sha256, "expected_receipt_tip_sha256"
    )
    timestamp = _utc_text(event_at_utc, "event_at_utc")
    root = _real_directory(Path(artifact_root), "artifact root")
    receipt_dir = _ensure_real_child_directory(
        Path(receipt_events_dir), label="receipt events directory"
    )
    lock_fd = _open_lock(receipt_dir)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        receipts = verify_receipt_store(receipt_dir)
        observed_receipt_tip = (
            receipts[-1]["receipt_sha256"] if receipts else ABSENT_CAS_TOKEN
        )
        if observed_receipt_tip != receipt_expected:
            raise DojoTerminalHandoffError(
                "stale receipt-store tip; reload before hand-off"
            )
        if len(receipts) >= MAX_RECEIPTS:
            raise DojoTerminalHandoffError("terminal hand-off receipt store is full")

        before = _load_lineage_or_absent(Path(lineage_events_dir), root)
        observed_lineage_tip = (
            before.latest_event_sha256 if before is not None else ABSENT_CAS_TOKEN
        )
        if observed_lineage_tip != lineage_expected:
            raise DojoTerminalHandoffError("stale lineage tip; reload before hand-off")
        if before is not None and _parse_utc(timestamp, "event_at_utc") < _parse_utc(
            before.latest_event_at_utc, "lineage latest event time"
        ):
            raise DojoTerminalHandoffError(
                "hand-off timestamp predates the verified lineage tip"
            )

        bundle = _read_terminal_bundle(Path(terminal_dir), root)
        study_reference = _read_json_artifact(
            Path(sealed_study_path), root, "sealed study artifact"
        )
        try:
            sealed_study = verify_sealed_study(
                study_reference["value"],
                study_reference["value"]["source_digests"],
            )
            run, evaluation, cells = validate_terminal_result_bundle(
                run=bundle["run"]["value"],
                evaluation=bundle["evaluation"]["value"],
                cells=bundle["cells"]["value"],
                sealed_study=sealed_study,
            )
        except (KeyError, DojoBotTrainerError, DojoAITrainerPacketError) as exc:
            raise DojoTerminalHandoffError(
                f"terminal bundle validation failed: {exc}"
            ) from exc

        # A second stable read closes the ordinary writer race between terminal
        # discovery and lineage mutation.  The runner writes run.json last, so
        # a healthy completed directory is quiescent here.
        bundle_again = _read_terminal_bundle(Path(terminal_dir), root)
        study_again = _read_json_artifact(
            Path(sealed_study_path), root, "sealed study artifact"
        )
        if not _same_bundle(bundle, bundle_again) or not _same_reference(
            study_reference, study_again
        ):
            raise DojoTerminalHandoffError(
                "terminal bundle or sealed study changed during validation"
            )

        lineage_before = _lineage_before_artifact(before)
        after, branch = _bind_lineage(
            snapshot=before,
            classification=classification,
            lineage_events_dir=Path(lineage_events_dir),
            artifact_root=root,
            sealed_study_path=Path(sealed_study_path).absolute(),
            evaluation_path=Path(terminal_dir).absolute() / "evaluation.json",
            study_reference=study_reference,
            sealed_study=sealed_study,
            evaluation=evaluation,
            evaluation_reference=bundle["evaluation"],
            event_at_utc=timestamp,
            registry_id=registry_id,
            lineage_prefix=lineage_prefix,
            created_by=created_by,
        )

        # The lineage verifier re-reads the exact bound evaluation.  Re-read the
        # other two terminal files as well before sealing their hashes into the
        # receipt, so a changed run/cells artifact cannot receive a hand-off.
        final_bundle = _read_terminal_bundle(Path(terminal_dir), root)
        if not _same_bundle(bundle, final_bundle):
            raise DojoTerminalHandoffError(
                "terminal bundle changed before hand-off receipt publication"
            )
        after = verify_registry(Path(lineage_events_dir), artifact_root=root)
        lineage_after = _lineage_after_artifact(after, evaluation)
        receipt = _build_receipt(
            sequence=len(receipts),
            recorded_at_utc=timestamp,
            previous_receipt_sha256=(
                None if not receipts else receipts[-1]["receipt_sha256"]
            ),
            classification=classification,
            branch=branch,
            bundle=bundle,
            normalized_run=run,
            study_reference=study_reference,
            sealed_study=sealed_study,
            lineage_before=lineage_before,
            lineage_after=lineage_after,
        )
        _write_receipt_exclusive(receipt_dir, receipt)
        return verify_handoff_receipt(receipt)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def verify_handoff_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Verify one terminal hand-off receipt and its research-only boundary."""

    row = _exact_mapping(value, _RECEIPT_KEYS, "handoff receipt")
    if row["contract"] != RECEIPT_CONTRACT or row["schema_version"] != 1:
        raise DojoTerminalHandoffError("handoff receipt contract/version drifted")
    sequence = _nonnegative_integer(row["sequence"], "sequence")
    _utc_text(row["recorded_at_utc"], "recorded_at_utc")
    previous = row["previous_receipt_sha256"]
    if (sequence == 0) is not (previous is None):
        raise DojoTerminalHandoffError("receipt genesis/previous binding drifted")
    if previous is not None:
        _sha(previous, "previous_receipt_sha256")
    classification = _classification(row["binding_timing_classification"])
    branch = row["lineage_branch"]
    if branch not in _LINEAGE_BRANCHES:
        raise DojoTerminalHandoffError("handoff lineage branch is unsupported")
    if (
        branch
        in {
            "UNINITIALIZED_LINEAGE_CREATED_SEALED_AND_BOUND",
            "EXISTING_GENESIS_SEALED_AND_BOUND",
        }
        and classification != RETROSPECTIVE_ADMIN_BINDING
    ):
        raise DojoTerminalHandoffError(
            "after-start lineage administration is not classified retrospective"
        )
    _verify_terminal_summary(row["terminal_bundle"])
    _verify_study_summary(row["sealed_study"])
    _verify_lineage_before(row["lineage_before"])
    _verify_lineage_after(row["lineage_after"])
    terminal = row["terminal_bundle"]
    study = row["sealed_study"]
    after = row["lineage_after"]
    if not (
        terminal["study_sha256"] == study["study_sha256"] == after["study_sha256"]
        and terminal["evaluation_sha256"] == after["evaluation_sha256"]
        and terminal["evaluation"]["artifact_sha256"]
        == after["evaluation_artifact_sha256"]
        and terminal["evaluation"]["artifact_size_bytes"]
        == after["evaluation_artifact_size_bytes"]
        and study["attempt_ordinal"] == after["attempt_ordinal"]
    ):
        raise DojoTerminalHandoffError(
            "terminal, study, and lineage result bindings diverge"
        )
    before = row["lineage_before"]
    if before["state"] == "VERIFIED" and (
        before["registry_id"] != after["registry_id"]
        or before["lineage_prefix"] != after["lineage_prefix"]
        or before["latest_sequence"] > after["latest_sequence"]
    ):
        raise DojoTerminalHandoffError("lineage identity or sequence changed")
    checks = _exact_mapping(row["checks"], _CHECK_KEYS, "handoff checks")
    if any(checks[key] is not True for key in _CHECK_KEYS):
        raise DojoTerminalHandoffError("handoff receipt contains a failed check")
    if row["limitations"] != list(_LIMITATIONS):
        raise DojoTerminalHandoffError("handoff receipt limitations drifted")
    _verify_authority(row)
    claimed = _sha(row["receipt_sha256"], "receipt_sha256")
    body = {key: item for key, item in row.items() if key != "receipt_sha256"}
    if canonical_sha256(body) != claimed:
        raise DojoTerminalHandoffError("handoff receipt SHA-256 mismatch")
    return _json_copy(row)


def verify_receipt_store(receipt_events_dir: Path) -> list[dict[str, Any]]:
    """Verify the complete ordinal receipt chain in one real directory."""

    directory = _real_directory(Path(receipt_events_dir), "receipt events directory")
    directory_fd = _open_directory(directory, "receipt events directory")
    try:
        names = sorted(
            name for name in os.listdir(directory_fd) if name != ".handoff.lock"
        )
        if any(_RECEIPT_NAME.fullmatch(name) is None for name in names):
            raise DojoTerminalHandoffError(
                "receipt events directory contains an unexpected file"
            )
        if len(names) > MAX_RECEIPTS:
            raise DojoTerminalHandoffError("terminal hand-off receipt store is full")
        expected_names = [f"{sequence:06d}.json" for sequence in range(len(names))]
        if names != expected_names:
            raise DojoTerminalHandoffError("receipt store has an ordinal gap or fork")
        receipts: list[dict[str, Any]] = []
        previous: str | None = None
        previous_time: datetime | None = None
        for sequence, name in enumerate(names):
            raw = _read_regular_at(
                directory_fd,
                name,
                maximum_bytes=MAX_RECEIPT_BYTES,
                label=f"handoff receipt {name}",
            )
            receipt = verify_handoff_receipt(_strict_json(raw, f"receipt {name}"))
            if raw != canonical_json_bytes(receipt) + b"\n":
                raise DojoTerminalHandoffError(
                    "handoff receipt bytes are not canonical JSON"
                )
            if receipt["sequence"] != sequence:
                raise DojoTerminalHandoffError("handoff receipt sequence drifted")
            if receipt["previous_receipt_sha256"] != previous:
                raise DojoTerminalHandoffError("handoff receipt SHA chain drifted")
            current_time = _parse_utc(receipt["recorded_at_utc"], "recorded_at_utc")
            if previous_time is not None and current_time < previous_time:
                raise DojoTerminalHandoffError("handoff receipt time moved backward")
            receipts.append(receipt)
            previous = receipt["receipt_sha256"]
            previous_time = current_time
        remaining = sorted(
            name for name in os.listdir(directory_fd) if name != ".handoff.lock"
        )
        if remaining != names:
            raise DojoTerminalHandoffError("receipt store changed while being verified")
        return receipts
    finally:
        os.close(directory_fd)


def receipt_store_status(receipt_events_dir: Path) -> dict[str, Any]:
    receipts = verify_receipt_store(receipt_events_dir)
    return {
        "contract": STORE_STATUS_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "receipt_count": len(receipts),
        "latest_receipt_sha256": (receipts[-1]["receipt_sha256"] if receipts else None),
        "latest_binding_timing_classification": (
            receipts[-1]["binding_timing_classification"] if receipts else None
        ),
        **_AUTHORITY,
    }


def _bind_lineage(
    *,
    snapshot: CandidateLineageSnapshot | None,
    classification: str,
    lineage_events_dir: Path,
    artifact_root: Path,
    sealed_study_path: Path,
    evaluation_path: Path,
    study_reference: Mapping[str, Any],
    sealed_study: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    evaluation_reference: Mapping[str, Any],
    event_at_utc: str,
    registry_id: str | None,
    lineage_prefix: str | None,
    created_by: str | None,
) -> tuple[CandidateLineageSnapshot, str]:
    current = snapshot
    if current is None:
        if classification != RETROSPECTIVE_ADMIN_BINDING:
            raise DojoTerminalHandoffError(
                "uninitialized lineage requires RETROSPECTIVE_ADMIN_BINDING"
            )
        if registry_id is None or lineage_prefix is None or created_by is None:
            raise DojoTerminalHandoffError(
                "retrospective lineage initialization requires registry_id, "
                "lineage_prefix, and created_by"
            )
        current = initialize_registry(
            lineage_events_dir,
            artifact_root=artifact_root,
            registry_id=_identifier(registry_id, "registry_id"),
            lineage_prefix=_identifier(lineage_prefix, "lineage_prefix"),
            created_by=_identifier(created_by, "created_by"),
            event_at_utc=event_at_utc,
        )
        current = seal_study_attempt(
            lineage_events_dir,
            artifact_root=artifact_root,
            sealed_study_path=sealed_study_path,
            expected_tip_sha256=current.latest_event_sha256,
            event_at_utc=event_at_utc,
        )
        branch = "UNINITIALIZED_LINEAGE_CREATED_SEALED_AND_BOUND"
    elif not current.studies:
        if classification != RETROSPECTIVE_ADMIN_BINDING:
            raise DojoTerminalHandoffError(
                "genesis-only lineage requires RETROSPECTIVE_ADMIN_BINDING"
            )
        _require_optional_identity_matches(current, registry_id, lineage_prefix)
        current = seal_study_attempt(
            lineage_events_dir,
            artifact_root=artifact_root,
            sealed_study_path=sealed_study_path,
            expected_tip_sha256=current.latest_event_sha256,
            event_at_utc=event_at_utc,
        )
        branch = "EXISTING_GENESIS_SEALED_AND_BOUND"
    elif len(current.studies) == len(current.results) + 1:
        _require_optional_identity_matches(current, registry_id, lineage_prefix)
        _require_exact_study_binding(current, study_reference, sealed_study)
        branch = "EXISTING_PENDING_STUDY_BOUND"
    elif len(current.studies) == len(current.results):
        _require_optional_identity_matches(current, registry_id, lineage_prefix)
        _require_exact_study_binding(current, study_reference, sealed_study)
        _require_exact_result_binding(
            current, evaluation, evaluation_reference=evaluation_reference
        )
        return current, "EXISTING_EXACT_RESULT_REUSED"
    else:  # verifier should make this unreachable; keep the coordinator closed.
        raise DojoTerminalHandoffError("candidate lineage state is not bindable")

    current = bind_result(
        lineage_events_dir,
        artifact_root=artifact_root,
        evaluation_path=evaluation_path,
        expected_tip_sha256=current.latest_event_sha256,
        event_at_utc=event_at_utc,
    )
    _require_exact_result_binding(
        current, evaluation, evaluation_reference=evaluation_reference
    )
    return current, branch


def _require_optional_identity_matches(
    snapshot: CandidateLineageSnapshot,
    registry_id: str | None,
    lineage_prefix: str | None,
) -> None:
    if (
        registry_id is not None
        and _identifier(registry_id, "registry_id") != snapshot.registry_id
    ):
        raise DojoTerminalHandoffError("supplied registry_id differs from lineage")
    if (
        lineage_prefix is not None
        and _identifier(lineage_prefix, "lineage_prefix") != snapshot.lineage_prefix
    ):
        raise DojoTerminalHandoffError("supplied lineage_prefix differs from lineage")


def _require_exact_study_binding(
    snapshot: CandidateLineageSnapshot,
    study_reference: Mapping[str, Any],
    sealed_study: Mapping[str, Any],
) -> None:
    study = snapshot.studies[-1]
    if (
        study["study_sha256"] != sealed_study["study_sha256"]
        or study["study_artifact_sha256"] != study_reference["artifact_sha256"]
        or study["study_artifact_size_bytes"] != study_reference["artifact_size_bytes"]
    ):
        raise DojoTerminalHandoffError(
            "lineage pending/latest study differs from the terminal sealed study"
        )


def _require_exact_result_binding(
    snapshot: CandidateLineageSnapshot,
    evaluation: Mapping[str, Any],
    *,
    evaluation_reference: Mapping[str, Any] | None = None,
) -> None:
    if not snapshot.results or len(snapshot.studies) != len(snapshot.results):
        raise DojoTerminalHandoffError("lineage result was not bound")
    result = snapshot.results[-1]
    if (
        result["study_sha256"] != evaluation["study_sha256"]
        or result["evaluation_sha256"] != evaluation["evaluation_sha256"]
    ):
        raise DojoTerminalHandoffError(
            "lineage result differs from the complete terminal bundle"
        )
    if evaluation_reference is not None and (
        result["evaluation_artifact_sha256"] != evaluation_reference["artifact_sha256"]
        or result["evaluation_artifact_size_bytes"]
        != evaluation_reference["artifact_size_bytes"]
    ):
        raise DojoTerminalHandoffError(
            "lineage result does not bind the exact terminal evaluation bytes"
        )


def _build_receipt(
    *,
    sequence: int,
    recorded_at_utc: str,
    previous_receipt_sha256: str | None,
    classification: str,
    branch: str,
    bundle: Mapping[str, Any],
    normalized_run: Mapping[str, Any],
    study_reference: Mapping[str, Any],
    sealed_study: Mapping[str, Any],
    lineage_before: Mapping[str, Any],
    lineage_after: Mapping[str, Any],
) -> dict[str, Any]:
    denominator = normalized_run["fixed_denominator"]
    body = {
        "contract": RECEIPT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "recorded_at_utc": recorded_at_utc,
        "previous_receipt_sha256": previous_receipt_sha256,
        "binding_timing_classification": classification,
        "lineage_branch": branch,
        "terminal_bundle": {
            "terminal_dir_relpath": bundle["terminal_dir_relpath"],
            "run": _artifact_summary(bundle["run"]),
            "evaluation": _artifact_summary(bundle["evaluation"]),
            "cells": _artifact_summary(bundle["cells"]),
            "run_sha256": normalized_run["run_sha256"],
            "evaluation_sha256": normalized_run["evaluation_sha256"],
            "study_sha256": normalized_run["study_sha256"],
            "status": normalized_run["status"],
            "expected_cell_count": denominator["expected_cell_count"],
            "observed_cell_count": denominator["observed_cell_count"],
            "failed_cell_count": denominator["failed_cell_count"],
        },
        "sealed_study": {
            **_artifact_summary(study_reference),
            "study_sha256": sealed_study["study_sha256"],
            "attempt_ordinal": sealed_study["study"]["search_budget"][
                "attempt_ordinal"
            ],
        },
        "lineage_before": _json_copy(lineage_before),
        "lineage_after": _json_copy(lineage_after),
        "checks": {key: True for key in sorted(_CHECK_KEYS)},
        "limitations": list(_LIMITATIONS),
        **_AUTHORITY,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _lineage_before_artifact(
    snapshot: CandidateLineageSnapshot | None,
) -> dict[str, Any]:
    if snapshot is None:
        return {
            "state": "ABSENT",
            "registry_id": None,
            "lineage_prefix": None,
            "latest_sequence": None,
            "latest_event_sha256": None,
            "study_count": 0,
            "result_count": 0,
        }
    return {
        "state": "VERIFIED",
        "registry_id": snapshot.registry_id,
        "lineage_prefix": snapshot.lineage_prefix,
        "latest_sequence": snapshot.latest_sequence,
        "latest_event_sha256": snapshot.latest_event_sha256,
        "study_count": len(snapshot.studies),
        "result_count": len(snapshot.results),
    }


def _lineage_after_artifact(
    snapshot: CandidateLineageSnapshot, evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    _require_exact_result_binding(snapshot, evaluation)
    result = snapshot.results[-1]
    result_event = next(
        event
        for event in reversed(snapshot.events)
        if event["event_type"] == "RESULT_BOUND"
    )
    return {
        "registry_id": snapshot.registry_id,
        "lineage_prefix": snapshot.lineage_prefix,
        "latest_sequence": snapshot.latest_sequence,
        "latest_event_sha256": snapshot.latest_event_sha256,
        "attempt_ordinal": result["attempt_ordinal"],
        "study_sha256": result["study_sha256"],
        "evaluation_sha256": result["evaluation_sha256"],
        "evaluation_artifact_sha256": result["evaluation_artifact_sha256"],
        "evaluation_artifact_size_bytes": result["evaluation_artifact_size_bytes"],
        "result_event_sha256": result_event["event_sha256"],
        "result_event_sequence": result_event["sequence"],
        "result_binding_sha256": canonical_sha256(result),
    }


def _load_lineage_or_absent(
    events_dir: Path, artifact_root: Path
) -> CandidateLineageSnapshot | None:
    candidate = events_dir.absolute()
    try:
        state = candidate.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DojoTerminalHandoffError(
            "cannot inspect lineage events directory"
        ) from exc
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise DojoTerminalHandoffError("lineage events path must be a real directory")
    try:
        names = os.listdir(candidate)
    except OSError as exc:
        raise DojoTerminalHandoffError("cannot list lineage events directory") from exc
    if not names:
        return None
    try:
        return verify_registry(candidate, artifact_root=artifact_root)
    except CandidateLineageError as exc:
        raise DojoTerminalHandoffError(f"candidate lineage is invalid: {exc}") from exc


def _read_terminal_bundle(terminal_dir: Path, artifact_root: Path) -> dict[str, Any]:
    directory = _real_directory(terminal_dir, "terminal directory")
    relative_dir = _relative_path(artifact_root, directory, "terminal directory")
    directory_fd = _open_directory(directory, "terminal directory")
    try:
        names = set(os.listdir(directory_fd))
        required = {"run.json", "evaluation.json", "cells.json"}
        missing = sorted(required - names)
        if missing:
            raise DojoTerminalHandoffError(
                "terminal bundle is incomplete; missing " + ", ".join(missing)
            )
        if "run_failure.json" in names:
            raise DojoTerminalHandoffError(
                "run_failure.json coexists with terminal artifacts"
            )
        result: dict[str, Any] = {"terminal_dir_relpath": relative_dir}
        for key, name in (
            ("run", "run.json"),
            ("evaluation", "evaluation.json"),
            ("cells", "cells.json"),
        ):
            raw = _read_regular_at(
                directory_fd,
                name,
                maximum_bytes=MAX_BOUND_ARTIFACT_BYTES,
                label=f"terminal {name}",
            )
            value = _strict_json(raw, f"terminal {name}")
            result[key] = {
                "artifact_relpath": f"{relative_dir}/{name}",
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "artifact_size_bytes": len(raw),
                "value": value,
            }
        names_after = set(os.listdir(directory_fd))
        if "run_failure.json" in names_after:
            raise DojoTerminalHandoffError(
                "run_failure.json appeared while terminal bundle was read"
            )
        if not required.issubset(names_after):
            raise DojoTerminalHandoffError("terminal bundle changed while being read")
        return result
    finally:
        os.close(directory_fd)


def _read_json_artifact(path: Path, root: Path, label: str) -> dict[str, Any]:
    candidate = path.absolute()
    relpath = _relative_path(root, candidate, label)
    try:
        expected = candidate.lstat()
    except OSError as exc:
        raise DojoTerminalHandoffError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(expected.st_mode) or stat.S_ISLNK(expected.st_mode):
        raise DojoTerminalHandoffError(f"{label} must be a regular file")
    parent = _real_directory(candidate.parent, f"{label} parent")
    parent_fd = _open_directory(parent, f"{label} parent")
    try:
        raw = _read_regular_at(
            parent_fd,
            candidate.name,
            maximum_bytes=MAX_BOUND_ARTIFACT_BYTES,
            label=label,
        )
    finally:
        os.close(parent_fd)
    return {
        "artifact_relpath": relpath,
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_size_bytes": len(raw),
        "value": _strict_json(raw, label),
    }


def _same_bundle(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left["terminal_dir_relpath"] == right["terminal_dir_relpath"] and all(
        _same_reference(left[key], right[key]) for key in ("run", "evaluation", "cells")
    )


def _same_reference(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        _artifact_summary(left) == _artifact_summary(right)
        and left["value"] == right["value"]
    )


def _artifact_summary(reference: Mapping[str, Any]) -> dict[str, Any]:
    return {key: reference[key] for key in sorted(_ARTIFACT_KEYS)}


def _verify_terminal_summary(value: Any) -> None:
    row = _exact_mapping(value, _TERMINAL_KEYS, "terminal bundle summary")
    _relpath(row["terminal_dir_relpath"], "terminal_dir_relpath")
    for key in ("run", "evaluation", "cells"):
        _verify_artifact_summary(row[key], f"terminal {key}")
    _sha(row["run_sha256"], "run_sha256")
    _sha(row["evaluation_sha256"], "evaluation_sha256")
    _sha(row["study_sha256"], "study_sha256")
    if row["status"] not in {"COMPLETE", "COMPLETE_WITH_FAILED_CELLS"}:
        raise DojoTerminalHandoffError("terminal status is not complete")
    expected = _positive_integer(row["expected_cell_count"], "expected_cell_count")
    observed = _positive_integer(row["observed_cell_count"], "observed_cell_count")
    failed = _nonnegative_integer(row["failed_cell_count"], "failed_cell_count")
    if observed != expected or failed > expected:
        raise DojoTerminalHandoffError("receipt terminal denominator is incomplete")


def _verify_study_summary(value: Any) -> None:
    row = _exact_mapping(value, _STUDY_KEYS, "sealed study summary")
    _verify_artifact_summary({key: row[key] for key in _ARTIFACT_KEYS}, "sealed study")
    _sha(row["study_sha256"], "study_sha256")
    _positive_integer(row["attempt_ordinal"], "attempt_ordinal")


def _verify_artifact_summary(value: Any, label: str) -> None:
    row = _exact_mapping(value, _ARTIFACT_KEYS, label)
    _relpath(row["artifact_relpath"], f"{label}.artifact_relpath")
    _sha(row["artifact_sha256"], f"{label}.artifact_sha256")
    _positive_integer(row["artifact_size_bytes"], f"{label}.artifact_size_bytes")


def _verify_lineage_before(value: Any) -> None:
    row = _exact_mapping(value, _LINEAGE_BEFORE_KEYS, "lineage before")
    if row["state"] == "ABSENT":
        if (
            any(
                row[key] is not None
                for key in (
                    "registry_id",
                    "lineage_prefix",
                    "latest_sequence",
                    "latest_event_sha256",
                )
            )
            or row["study_count"] != 0
            or row["result_count"] != 0
        ):
            raise DojoTerminalHandoffError("absent lineage summary is inconsistent")
        return
    if row["state"] != "VERIFIED":
        raise DojoTerminalHandoffError("lineage before state is unsupported")
    _identifier(row["registry_id"], "lineage before registry_id")
    _identifier(row["lineage_prefix"], "lineage before lineage_prefix")
    _nonnegative_integer(row["latest_sequence"], "lineage before sequence")
    _sha(row["latest_event_sha256"], "lineage before tip")
    studies = _nonnegative_integer(row["study_count"], "study_count")
    results = _nonnegative_integer(row["result_count"], "result_count")
    if results > studies or studies > results + 1:
        raise DojoTerminalHandoffError("lineage before counts are inconsistent")


def _verify_lineage_after(value: Any) -> None:
    row = _exact_mapping(value, _LINEAGE_AFTER_KEYS, "lineage after")
    _identifier(row["registry_id"], "lineage after registry_id")
    _identifier(row["lineage_prefix"], "lineage after lineage_prefix")
    _nonnegative_integer(row["latest_sequence"], "lineage after sequence")
    for field in (
        "latest_event_sha256",
        "study_sha256",
        "evaluation_sha256",
        "evaluation_artifact_sha256",
        "result_event_sha256",
        "result_binding_sha256",
    ):
        _sha(row[field], f"lineage after {field}")
    _positive_integer(row["attempt_ordinal"], "lineage after attempt_ordinal")
    _positive_integer(
        row["evaluation_artifact_size_bytes"],
        "lineage after evaluation_artifact_size_bytes",
    )
    _positive_integer(row["result_event_sequence"], "result_event_sequence")
    if (
        row["latest_sequence"] != row["result_event_sequence"]
        or row["latest_event_sha256"] != row["result_event_sha256"]
    ):
        raise DojoTerminalHandoffError("lineage after is not terminal RESULT_BOUND")


def _write_receipt_exclusive(directory: Path, receipt: Mapping[str, Any]) -> None:
    verified = verify_handoff_receipt(receipt)
    payload = canonical_json_bytes(verified) + b"\n"
    if len(payload) > MAX_RECEIPT_BYTES:
        raise DojoTerminalHandoffError("handoff receipt byte limit exceeded")
    directory_fd = _open_directory(directory, "receipt events directory")
    descriptor: int | None = None
    try:
        name = f"{verified['sequence']:06d}.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            written = handle.write(payload)
            if written != len(payload):
                raise DojoTerminalHandoffError("handoff receipt write was incomplete")
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(directory_fd)
    except FileExistsError as exc:
        raise DojoTerminalHandoffError(
            "handoff receipt slot already exists; reload current tip"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _ensure_real_child_directory(path: Path, *, label: str) -> Path:
    candidate = path.absolute()
    if candidate == candidate.parent:
        raise DojoTerminalHandoffError(f"{label} cannot be a filesystem root")
    try:
        state = candidate.lstat()
    except FileNotFoundError:
        parent = _real_directory(candidate.parent, f"{label} parent")
        parent_fd = _open_directory(parent, f"{label} parent")
        try:
            try:
                os.mkdir(candidate.name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            os.fsync(parent_fd)
        except OSError as exc:
            raise DojoTerminalHandoffError(f"cannot create {label}") from exc
        finally:
            os.close(parent_fd)
        state = candidate.lstat()
    except OSError as exc:
        raise DojoTerminalHandoffError(f"cannot inspect {label}") from exc
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise DojoTerminalHandoffError(f"{label} must be a real directory")
    return _real_directory(candidate, label)


def _open_lock(directory: Path) -> int:
    directory_fd = _open_directory(directory, "receipt events directory")
    try:
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(".handoff.lock", flags, 0o600, dir_fd=directory_fd)
        state = os.fstat(descriptor)
        if not stat.S_ISREG(state.st_mode):
            os.close(descriptor)
            raise DojoTerminalHandoffError("handoff lock is not a regular file")
        return descriptor
    except OSError as exc:
        raise DojoTerminalHandoffError("cannot open handoff lock") from exc
    finally:
        os.close(directory_fd)


def _read_regular_at(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    try:
        expected = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise DojoTerminalHandoffError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(expected.st_mode)
        or expected.st_size <= 0
        or expected.st_size > maximum_bytes
    ):
        raise DojoTerminalHandoffError(
            f"{label} must be a bounded nonempty regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            raw = handle.read(maximum_bytes + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise DojoTerminalHandoffError(f"{label} could not be read safely") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_dev != expected.st_dev
        or before.st_ino != expected.st_ino
        or len(raw) != before.st_size
    ):
        raise DojoTerminalHandoffError(f"{label} changed while being read")
    return raw


def _strict_json(raw: bytes, label: str) -> Any:
    def reject_constant(token: str) -> None:
        raise DojoTerminalHandoffError(
            f"{label} contains forbidden non-finite JSON token {token}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DojoTerminalHandoffError(
                    f"{label} contains duplicate JSON key {key}"
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except DojoTerminalHandoffError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DojoTerminalHandoffError(f"{label} is not strict JSON") from exc
    _validate_json(value, label)
    return value


def _real_directory(path: Path, label: str) -> Path:
    candidate = path.absolute()
    try:
        expected = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise DojoTerminalHandoffError(f"{label} is unavailable") from exc
    if (
        resolved != candidate
        or not stat.S_ISDIR(expected.st_mode)
        or stat.S_ISLNK(expected.st_mode)
    ):
        raise DojoTerminalHandoffError(f"{label} must be a real directory")
    return candidate


def _open_directory(path: Path, label: str) -> int:
    try:
        expected = path.lstat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        actual = os.fstat(descriptor)
    except OSError as exc:
        raise DojoTerminalHandoffError(f"cannot open {label}") from exc
    if (
        not stat.S_ISDIR(actual.st_mode)
        or actual.st_dev != expected.st_dev
        or actual.st_ino != expected.st_ino
    ):
        os.close(descriptor)
        raise DojoTerminalHandoffError(f"{label} changed while being opened")
    return descriptor


def _relative_path(root: Path, path: Path, label: str) -> str:
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise DojoTerminalHandoffError(f"{label} escapes artifact root") from exc
    if not relative.parts:
        raise DojoTerminalHandoffError(f"{label} cannot equal artifact root")
    return _relpath(PurePosixPath(*relative.parts).as_posix(), label)


def _relpath(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DojoTerminalHandoffError(f"{label} is not a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DojoTerminalHandoffError(f"{label} is not a safe relative path")
    return path.as_posix()


def _classification(value: Any) -> str:
    if value not in {LINEAGE_PRESENT_AT_HANDOFF, RETROSPECTIVE_ADMIN_BINDING}:
        raise DojoTerminalHandoffError(
            "binding_timing_classification must be "
            "LINEAGE_PRESENT_AT_HANDOFF_NO_PREREGISTRATION_CLAIM or "
            "RETROSPECTIVE_ADMIN_BINDING"
        )
    return str(value)


def _cas_token(value: Any, label: str) -> str:
    if value == ABSENT_CAS_TOKEN:
        return ABSENT_CAS_TOKEN
    return _sha(value, label)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DojoTerminalHandoffError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise DojoTerminalHandoffError(f"{label} is invalid")
    return value


def _positive_integer(value: Any, label: str) -> int:
    number = _nonnegative_integer(value, label)
    if number <= 0:
        raise DojoTerminalHandoffError(f"{label} must be positive")
    return number


def _nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DojoTerminalHandoffError(f"{label} must be a non-negative integer")
    return value


def _utc_text(value: datetime | str, label: str) -> str:
    instant = _parse_utc(value, label)
    return instant.isoformat().replace("+00:00", "Z")


def _parse_utc(value: datetime | str, label: str) -> datetime:
    try:
        instant = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
    except (TypeError, ValueError) as exc:
        raise DojoTerminalHandoffError(f"{label} must be an ISO-8601 instant") from exc
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise DojoTerminalHandoffError(f"{label} must be timezone-aware")
    return instant.astimezone(timezone.utc)


def _verify_authority(value: Mapping[str, Any]) -> None:
    for key, expected in _AUTHORITY.items():
        if value.get(key) != expected:
            raise DojoTerminalHandoffError(
                "terminal hand-off exceeds research-only authority"
            )


def _exact_mapping(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise DojoTerminalHandoffError(f"{label} schema mismatch")
    return _json_copy(value)


def _validate_json(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DojoTerminalHandoffError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DojoTerminalHandoffError(f"{label} contains a non-string key")
            _validate_json(item, f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_json(item, f"{label}[{index}]")
        return
    raise DojoTerminalHandoffError(f"{label} is not JSON")


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


__all__ = [
    "ABSENT_CAS_TOKEN",
    "LINEAGE_PRESENT_AT_HANDOFF",
    "RETROSPECTIVE_ADMIN_BINDING",
    "DojoTerminalHandoffError",
    "canonical_json_bytes",
    "canonical_sha256",
    "coordinate_terminal_handoff",
    "receipt_store_status",
    "verify_handoff_receipt",
    "verify_receipt_store",
]

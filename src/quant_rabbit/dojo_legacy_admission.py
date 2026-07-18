"""Fail-closed admission boundary for pre-DOJO positive artifacts.

Legacy research files are allowed to remain readable as diagnostics, but a
builder must not translate them back into an "adopted" strategy merely because
their internal digest verifies.  Positive reuse requires all of the following:

* exactly one content-addressed current goal board in the registry;
* a canonically sealed ``QR_DOJO_GOAL_BOARD_V2`` board;
* board-level proof admission enabled by reviewed proof verifiers;
* explicit, trusted ``EDGE_PROVEN`` registration of every positive source; and
* source-local declarations that the artifact is prospective, non-diagnostic,
  and eligible for promotion.

This module grants no live or order authority.  It only prevents semantic
rollback from the current DOJO validity boundary to historical self-attested
JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_goal_board import canonical_sha256


GOAL_BOARD_CONTRACT: Final = "QR_DOJO_GOAL_BOARD_V2"
GOAL_BOARD_SCHEMA_VERSION: Final = 2
MAX_BOARD_BYTES: Final = 4 * 1024 * 1024
MAX_SOURCE_BYTES: Final = 64 * 1024 * 1024
MAX_LANES: Final = 256
_BOARD_NAME_RE: Final = re.compile(r"dojo_goal_board_[0-9]{8}_([0-9a-f]{64})\.json\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


class LegacyAdmissionError(ValueError):
    """Raised when legacy positive evidence is not admitted by current DOJO."""


def _reject_json_constant(value: str) -> None:
    raise LegacyAdmissionError(f"non-finite JSON constant is forbidden: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise LegacyAdmissionError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _read_regular_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    try:
        state = path.lstat()
    except OSError as exc:
        raise LegacyAdmissionError(f"cannot stat JSON artifact {path}: {exc}") from exc
    if not stat.S_ISREG(state.st_mode):
        raise LegacyAdmissionError(f"JSON artifact must be a regular file: {path}")
    if state.st_size <= 0 or state.st_size > maximum_bytes:
        raise LegacyAdmissionError(
            f"JSON artifact size is outside 1..{maximum_bytes} bytes: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read(maximum_bytes + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise LegacyAdmissionError(f"cannot read JSON artifact {path}: {exc}") from exc
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != before.st_size
    ):
        raise LegacyAdmissionError(f"JSON artifact changed while reading: {path}")
    return payload


def load_strict_json_with_sha256(
    path: Path, *, maximum_bytes: int
) -> tuple[dict[str, Any], str]:
    """Load one bounded regular JSON file and hash the exact parsed bytes."""

    payload = _read_regular_bytes(path, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except LegacyAdmissionError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LegacyAdmissionError(f"cannot parse JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LegacyAdmissionError(f"JSON artifact must contain one object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def load_strict_json(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    """Load one bounded regular JSON file without following a final symlink."""

    value, _ = load_strict_json_with_sha256(path, maximum_bytes=maximum_bytes)
    return value


def regular_file_sha256(path: Path, *, maximum_bytes: int = MAX_SOURCE_BYTES) -> str:
    """Hash one bounded regular file while rejecting final-component symlinks."""

    try:
        state = path.lstat()
    except OSError as exc:
        raise LegacyAdmissionError(
            f"cannot stat positive source {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(state.st_mode):
        raise LegacyAdmissionError(f"positive source must be a regular file: {path}")
    if state.st_size <= 0 or state.st_size > maximum_bytes:
        raise LegacyAdmissionError(
            f"positive source size is outside 1..{maximum_bytes} bytes: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise LegacyAdmissionError(
            f"cannot hash positive source {path}: {exc}"
        ) from exc
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_size != state.st_size
    ):
        raise LegacyAdmissionError(f"positive source changed while hashing: {path}")
    return digest.hexdigest()


def _require_boolean(value: Any, *, field: str, expected: bool) -> None:
    if value is not expected:
        raise LegacyAdmissionError(
            f"legacy positive source requires {field}={str(expected).lower()}"
        )


def _validate_goal_board(value: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
    required = {
        "contract",
        "schema_version",
        "board_sha256",
        "proof_admission",
        "lane_evaluations",
        "order_authority",
        "live_permission",
        "broker_mutation_allowed",
    }
    missing = sorted(required - set(value))
    if missing:
        raise LegacyAdmissionError(f"goal board is missing required fields: {missing}")
    if value["contract"] != GOAL_BOARD_CONTRACT:
        raise LegacyAdmissionError("goal board contract is not QR_DOJO_GOAL_BOARD_V2")
    if (
        isinstance(value["schema_version"], bool)
        or not isinstance(value["schema_version"], int)
        or value["schema_version"] != GOAL_BOARD_SCHEMA_VERSION
    ):
        raise LegacyAdmissionError("goal board schema version is unsupported")
    declared_sha = value["board_sha256"]
    if not isinstance(declared_sha, str) or _SHA256_RE.fullmatch(declared_sha) is None:
        raise LegacyAdmissionError("goal board SHA-256 is malformed")
    body = dict(value)
    body.pop("board_sha256")
    if canonical_sha256(body) != declared_sha:
        raise LegacyAdmissionError("goal board canonical SHA-256 does not verify")
    match = _BOARD_NAME_RE.fullmatch(path.name)
    if match is None or match.group(1) != declared_sha:
        raise LegacyAdmissionError(
            "goal board filename must contain its canonical SHA-256"
        )
    _require_boolean(
        value["live_permission"], field="goal_board.live_permission", expected=False
    )
    _require_boolean(
        value["broker_mutation_allowed"],
        field="goal_board.broker_mutation_allowed",
        expected=False,
    )
    if value["order_authority"] != "NONE":
        raise LegacyAdmissionError("goal board order_authority must remain NONE")
    proof_admission = value["proof_admission"]
    if not isinstance(proof_admission, dict):
        raise LegacyAdmissionError("goal board proof_admission must be an object")
    for key in (
        "promotion_possible",
        "self_asserted_json_can_promote",
        "trusted_proof_contracts",
    ):
        if key not in proof_admission:
            raise LegacyAdmissionError(f"goal board proof_admission is missing {key}")
    if not isinstance(proof_admission["promotion_possible"], bool):
        raise LegacyAdmissionError("promotion_possible must be boolean")
    if proof_admission["self_asserted_json_can_promote"] is not False:
        raise LegacyAdmissionError("self-asserted JSON must never promote")
    trusted_contracts = proof_admission["trusted_proof_contracts"]
    if not isinstance(trusted_contracts, list) or any(
        not isinstance(item, str) or not item for item in trusted_contracts
    ):
        raise LegacyAdmissionError("trusted_proof_contracts must be a string array")
    lanes = value["lane_evaluations"]
    if not isinstance(lanes, list) or len(lanes) > MAX_LANES:
        raise LegacyAdmissionError(f"lane_evaluations must contain 0..{MAX_LANES} rows")
    lane_ids: set[str] = set()
    for index, lane in enumerate(lanes):
        if not isinstance(lane, dict) or not isinstance(lane.get("lane_id"), str):
            raise LegacyAdmissionError(f"lane_evaluations[{index}] is malformed")
        if lane["lane_id"] in lane_ids:
            raise LegacyAdmissionError("goal board lane IDs must be unique")
        lane_ids.add(lane["lane_id"])
    return dict(value)


def load_current_goal_board(registry_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Load the unique content-addressed board from the active registry."""

    try:
        state = registry_dir.lstat()
    except OSError as exc:
        raise LegacyAdmissionError(f"cannot stat DOJO registry: {exc}") from exc
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise LegacyAdmissionError("DOJO registry must be one real directory")
    candidates = sorted(
        path
        for path in registry_dir.iterdir()
        if _BOARD_NAME_RE.fullmatch(path.name) is not None
    )
    if len(candidates) != 1:
        raise LegacyAdmissionError(
            "DOJO registry must contain exactly one current content-addressed "
            f"goal board; found={len(candidates)}"
        )
    board_path = candidates[0]
    board = load_strict_json(board_path, maximum_bytes=MAX_BOARD_BYTES)
    return board_path, _validate_goal_board(board, path=board_path)


def _eligible_positive_source(name: str, value: Mapping[str, Any]) -> None:
    _require_boolean(
        value.get("diagnostic_only"),
        field=f"{name}.diagnostic_only",
        expected=False,
    )
    _require_boolean(
        value.get("historical_only"),
        field=f"{name}.historical_only",
        expected=False,
    )
    _require_boolean(
        value.get("forward_proof_eligible"),
        field=f"{name}.forward_proof_eligible",
        expected=True,
    )
    _require_boolean(
        value.get("promotion_allowed"),
        field=f"{name}.promotion_allowed",
        expected=True,
    )


def admit_legacy_positive_sources(
    *,
    board_path: Path,
    board: Mapping[str, Any],
    positive_sources: Mapping[str, tuple[Path, Mapping[str, Any]]],
    loaded_source_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a sealed admission receipt or reject every unsafe positive reuse."""

    if not positive_sources:
        raise LegacyAdmissionError("at least one positive source is required")
    current_path, current_board = load_current_goal_board(board_path.parent)
    if current_path != board_path or current_board != dict(board):
        raise LegacyAdmissionError(
            "goal board changed after admission input was loaded"
        )
    proof_admission = current_board["proof_admission"]
    if proof_admission["promotion_possible"] is not True:
        raise LegacyAdmissionError(
            "current DOJO goal board does not permit evidence promotion"
        )
    trusted_contracts = set(proof_admission["trusted_proof_contracts"])
    if not trusted_contracts:
        raise LegacyAdmissionError("current DOJO has no reviewed proof verifier")

    registrations: dict[str, dict[str, str]] = {}
    used_lane_ids: set[str] = set()
    for source_name, (source_path, source_value) in sorted(positive_sources.items()):
        _eligible_positive_source(source_name, source_value)
        source_sha = regular_file_sha256(source_path)
        if loaded_source_sha256 is not None:
            loaded_sha = loaded_source_sha256.get(source_name)
            if loaded_sha != source_sha:
                raise LegacyAdmissionError(
                    f"{source_name} changed after its admitted JSON was parsed"
                )
        matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for lane in board["lane_evaluations"]:
            evidence = lane.get("evidence_verification")
            if not isinstance(evidence, dict):
                continue
            if (
                evidence.get("actual_sha256") == source_sha
                and evidence.get("expected_sha256") == source_sha
            ):
                matches.append((lane, evidence))
        if len(matches) != 1:
            raise LegacyAdmissionError(
                f"{source_name} is not uniquely registered by current DOJO; "
                f"matches={len(matches)}"
            )
        lane, evidence = matches[0]
        lane_id = str(lane["lane_id"])
        if lane_id in used_lane_ids:
            raise LegacyAdmissionError(
                f"{source_name} reuses a DOJO lane already assigned to another source"
            )
        used_lane_ids.add(lane_id)
        if (
            lane.get("declared_status") != "EDGE_PROVEN"
            or lane.get("edge_status") != "EDGE_PROVEN"
        ):
            raise LegacyAdmissionError(
                f"{source_name} is not registered as an EDGE_PROVEN lane"
            )
        if (
            evidence.get("trusted") is not True
            or evidence.get("status") != "VERIFIED"
            or evidence.get("blocker") is not None
        ):
            raise LegacyAdmissionError(
                f"{source_name} lacks trusted verifier-derived evidence"
            )
        evidence_contract = evidence.get("contract")
        if evidence_contract not in trusted_contracts:
            raise LegacyAdmissionError(
                f"{source_name} evidence contract is not trusted by current DOJO"
            )
        registrations[source_name] = {
            "lane_id": lane_id,
            "source_file_sha256": source_sha,
            "evidence_contract": str(evidence_contract),
        }

    body = {
        "contract": "QR_DOJO_LEGACY_POSITIVE_ADMISSION_V1",
        "schema_version": 1,
        "goal_board_path": os.fspath(board_path),
        "goal_board_sha256": current_board["board_sha256"],
        "registrations": registrations,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return {**body, "admission_sha256": canonical_sha256(body)}


def positive_source_names() -> Sequence[str]:
    """Names whose evidence is translated into positive adoption semantics."""

    return (
        "survivor_lock",
        "validation_replication",
        "monthly_distribution",
        "all_weather_attribution",
        "cell_gating",
        "lane_addition",
    )

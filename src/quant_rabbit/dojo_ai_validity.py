"""Local first-write validity ledger for the prospective DOJO AI artifact DAG.

This ledger detects byte rewrites, slot reuse, invalid-parent descendants, and
in-process forks.  It is intentionally *not* described as externally monotonic:
the local owner can delete and recreate the entire directory until an external
signed witness is added.  Consequently every status remains diagnostic and
proof/promotion/live eligibility is false.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_ai_discretion import canonical_sha256


GENESIS_CONTRACT = "QR_DOJO_AI_VALIDITY_GENESIS_V1"
EVENT_CONTRACT = "QR_DOJO_AI_VALIDITY_EVENT_V1"
REGISTRY_DIRNAME = "validity"
MAX_EVENTS = 1_024
MAX_EVENT_BYTES = 65_536
MAX_TOTAL_BYTES = 4 * 1024 * 1024
MAX_PARENTS = 256
MAX_GRAPH_DEPTH = 128
_HEX64 = re.compile(r"[0-9a-f]{64}")
_LOGICAL_ID = re.compile(r"[a-z0-9][a-z0-9._/-]{0,239}")
_REASON = re.compile(r"[A-Z][A-Z0-9_]{2,119}")


class DojoAIValidityError(ValueError):
    """The local first-write ledger or artifact DAG is invalid."""


@dataclass(frozen=True)
class RegistrySnapshot:
    registry_id: str
    event_count: int
    latest_sequence: int
    latest_event_sha256: str
    committed: Mapping[str, Mapping[str, Any]]
    invalidated: frozenset[str]
    invalid_paths: Mapping[str, tuple[str, ...]]
    external_witness_status: str = "ABSENT"
    proof_eligible: bool = False
    promotion_eligible: bool = False
    live_permission: bool = False


def initialize_registry(
    run_dir: Path,
    *,
    precommit_path: Path,
    start_path: Path,
    created_at_utc: datetime,
) -> RegistrySnapshot:
    """Create genesis once, binding exact precommit/start raw bytes."""

    root = _safe_run_dir(run_dir)
    events = root / REGISTRY_DIRNAME / "events"
    genesis_path = events / "000000.json"
    if genesis_path.is_file():
        return verify_registry(root)
    if genesis_path.exists() or genesis_path.is_symlink():
        raise DojoAIValidityError("validity genesis path is unsafe")
    precommit = _artifact_row(root, precommit_path, "precommit", ())
    start = _artifact_row(root, start_path, "start", ("precommit",))
    if precommit["contract"] != "QR_DOJO_AI_FORWARD_PRECOMMIT_V3":
        raise DojoAIValidityError("validity genesis requires V3 precommit")
    if start["contract"] != "QR_DOJO_AI_FORWARD_START_V3":
        raise DojoAIValidityError("validity genesis requires V3 start")
    if precommit["artifact_sha256"] not in start["referenced_sha256s"]:
        raise DojoAIValidityError("validity start does not bind precommit")
    created = _utc(created_at_utc, "created_at_utc")
    registry_id = hashlib.sha256(
        (precommit["raw_sha256"] + start["raw_sha256"]).encode("ascii")
    ).hexdigest()
    body = {
        "contract": GENESIS_CONTRACT,
        "schema_version": 1,
        "event_kind": "GENESIS",
        "registry_id": registry_id,
        "sequence": 0,
        "previous_event_sha256": None,
        "created_at_utc": _iso(created),
        "artifacts": [precommit, start],
        "policy": {
            "first_write_slot_immutable": True,
            "slot_reuse_allowed": False,
            "transitive_invalidity_fail_closed": True,
            "invalidity_reversible": False,
            "event_pruning_allowed": False,
            "max_events": MAX_EVENTS,
            "max_total_bytes": MAX_TOTAL_BYTES,
            "max_graph_depth": MAX_GRAPH_DEPTH,
            "external_monotonic_witness_required_for_proof": True,
        },
        "external_witness_status": "ABSENT",
        "external_witness_receipt_sha256": None,
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
    }
    genesis = _seal(body, "event_sha256")
    _write_json_new(genesis_path, genesis, root)
    return verify_registry(root)


def append_artifact_commit(
    run_dir: Path,
    *,
    logical_id: str,
    artifact_path: Path,
    parent_logical_ids: Sequence[str],
    committed_at_utc: datetime,
) -> RegistrySnapshot:
    """Commit one previously unseen logical slot and its exact raw bytes."""

    root = _safe_run_dir(run_dir)
    logical = _logical(logical_id)
    parents = _parent_ids(parent_logical_ids)
    snapshot = verify_registry(root)
    existing = snapshot.committed.get(logical)
    candidate = _artifact_row(root, artifact_path, logical, parents)
    if existing is not None:
        if dict(existing) != candidate:
            raise DojoAIValidityError("SLOT_REUSE_CONFLICT")
        return snapshot
    if len(snapshot.committed) >= MAX_EVENTS:
        raise DojoAIValidityError("REGISTRY_CAPACITY_EXHAUSTED")
    for parent in parents:
        parent_row = snapshot.committed.get(parent)
        if parent_row is None:
            raise DojoAIValidityError(f"missing committed parent: {parent}")
        if parent in snapshot.invalidated:
            raise DojoAIValidityError(f"invalid parent blocks descendant: {parent}")
        if parent_row["artifact_sha256"] not in candidate["referenced_sha256s"]:
            raise DojoAIValidityError(
                f"artifact bytes do not bind declared parent: {parent}"
            )
    event = _event(
        snapshot,
        event_kind="ARTIFACT_COMMITTED",
        created_at_utc=committed_at_utc,
        payload={"artifact": candidate},
    )
    _append_event(root, event)
    return verify_registry(root)


def append_invalidation(
    run_dir: Path,
    *,
    logical_id: str,
    reason_code: str,
    evidence_sha256: str,
    invalidated_at_utc: datetime,
) -> RegistrySnapshot:
    """Irreversibly invalidate one committed root; descendants fail at read time."""

    root = _safe_run_dir(run_dir)
    logical = _logical(logical_id)
    reason = _reason(reason_code)
    evidence = _sha(evidence_sha256, "evidence_sha256")
    snapshot = verify_registry(root)
    if logical not in snapshot.committed:
        raise DojoAIValidityError("cannot invalidate an uncommitted slot")
    explicit = _explicit_invalidations(root)
    prior = explicit.get(logical)
    if prior is not None:
        if prior["reason_code"] != reason or prior["evidence_sha256"] != evidence:
            raise DojoAIValidityError("invalidation replacement is forbidden")
        return snapshot
    event = _event(
        snapshot,
        event_kind="ARTIFACT_INVALIDATED",
        created_at_utc=invalidated_at_utc,
        payload={
            "target_logical_id": logical,
            "target_artifact_sha256": snapshot.committed[logical][
                "artifact_sha256"
            ],
            "reason_code": reason,
            "evidence_sha256": evidence,
            "descendant_policy": "TRANSITIVE_FAIL_CLOSED",
            "reversible": False,
        },
    )
    _append_event(root, event)
    return verify_registry(root)


def verify_registry(run_dir: Path) -> RegistrySnapshot:
    """Verify strict chain, raw bytes, parent DAG, and transitive invalidity."""

    root = _safe_run_dir(run_dir)
    events_dir = root / REGISTRY_DIRNAME / "events"
    if not events_dir.is_dir() or events_dir.is_symlink():
        raise DojoAIValidityError("validity events directory is absent or unsafe")
    entries = sorted(events_dir.iterdir())
    if not entries or len(entries) > MAX_EVENTS:
        raise DojoAIValidityError("validity event count is invalid")
    expected_names = [f"{index:06d}.json" for index in range(len(entries))]
    if [path.name for path in entries] != expected_names:
        raise DojoAIValidityError("validity event sequence has a gap or extra file")
    total_bytes = 0
    previous_sha: str | None = None
    registry_id: str | None = None
    committed: dict[str, dict[str, Any]] = {}
    explicit_invalid: dict[str, dict[str, Any]] = {}
    latest_sha = ""
    for sequence, path in enumerate(entries):
        if not path.is_file() or path.is_symlink():
            raise DojoAIValidityError("validity event path is unsafe")
        raw = path.read_bytes()
        total_bytes += len(raw)
        if len(raw) > MAX_EVENT_BYTES or total_bytes > MAX_TOTAL_BYTES:
            raise DojoAIValidityError("validity registry byte bound exceeded")
        event = _strict_json_bytes(raw)
        _validate_seal(event, "event_sha256")
        if event.get("sequence") != sequence:
            raise DojoAIValidityError("validity event sequence drifted")
        if event.get("previous_event_sha256") != previous_sha:
            raise DojoAIValidityError("validity event chain drifted")
        if event.get("external_witness_status") != "ABSENT" or event.get(
            "external_witness_receipt_sha256"
        ) is not None:
            raise DojoAIValidityError("unsupported or forged witness status")
        if any(event.get(key) is not False for key in (
            "proof_eligible", "promotion_eligible", "live_permission"
        )):
            raise DojoAIValidityError("validity event exceeds diagnostic authority")
        if sequence == 0:
            if event.get("contract") != GENESIS_CONTRACT or event.get("event_kind") != "GENESIS":
                raise DojoAIValidityError("validity genesis identity drifted")
            registry_id = _sha(event.get("registry_id"), "registry_id")
            artifacts = event.get("artifacts")
            if not isinstance(artifacts, list) or len(artifacts) != 2:
                raise DojoAIValidityError("validity genesis artifact set drifted")
            for row in artifacts:
                validated = _validate_artifact_row(root, row)
                logical = validated["logical_id"]
                if logical in committed:
                    raise DojoAIValidityError("validity genesis reuses a slot")
                committed[logical] = validated
        else:
            if event.get("contract") != EVENT_CONTRACT or event.get("registry_id") != registry_id:
                raise DojoAIValidityError("validity event identity drifted")
            kind = event.get("event_kind")
            if kind == "ARTIFACT_COMMITTED":
                row = _validate_artifact_row(root, event.get("artifact"))
                logical = row["logical_id"]
                if logical in committed:
                    raise DojoAIValidityError("validity ledger reuses a logical slot")
                for parent in row["parent_logical_ids"]:
                    if parent not in committed:
                        raise DojoAIValidityError("validity artifact has missing/later parent")
                    if committed[parent]["artifact_sha256"] not in row["referenced_sha256s"]:
                        raise DojoAIValidityError("validity artifact omits declared parent bytes")
                committed[logical] = row
            elif kind == "ARTIFACT_INVALIDATED":
                logical = _logical(event.get("target_logical_id"))
                if logical not in committed or logical in explicit_invalid:
                    raise DojoAIValidityError("validity invalidation target is invalid or repeated")
                if event.get("target_artifact_sha256") != committed[logical]["artifact_sha256"]:
                    raise DojoAIValidityError("validity invalidation target digest drifted")
                _reason(event.get("reason_code"))
                _sha(event.get("evidence_sha256"), "evidence_sha256")
                if event.get("descendant_policy") != "TRANSITIVE_FAIL_CLOSED" or event.get("reversible") is not False:
                    raise DojoAIValidityError("validity invalidation policy drifted")
                explicit_invalid[logical] = event
            else:
                raise DojoAIValidityError("validity event kind is unsupported")
        previous_sha = event["event_sha256"]
        latest_sha = previous_sha
    if registry_id is None:
        raise DojoAIValidityError("validity registry lacks genesis")
    invalid, paths = _invalid_closure(committed, explicit_invalid)
    _assert_depth(committed)
    return RegistrySnapshot(
        registry_id=registry_id,
        event_count=len(entries),
        latest_sequence=len(entries) - 1,
        latest_event_sha256=latest_sha,
        committed=committed,
        invalidated=frozenset(invalid),
        invalid_paths=paths,
    )


def assert_artifacts_valid(run_dir: Path, logical_ids: Sequence[str]) -> RegistrySnapshot:
    snapshot = verify_registry(run_dir)
    for raw in logical_ids:
        logical = _logical(raw)
        if logical not in snapshot.committed:
            raise DojoAIValidityError(f"artifact slot is not committed: {logical}")
        if logical in snapshot.invalidated:
            path = " -> ".join(snapshot.invalid_paths[logical])
            raise DojoAIValidityError(f"artifact is transitively invalid: {path}")
    return snapshot


def status_artifact(run_dir: Path) -> dict[str, Any]:
    snapshot = verify_registry(run_dir)
    return {
        "contract": "QR_DOJO_AI_VALIDITY_STATUS_V1",
        "registry_id": snapshot.registry_id,
        "event_count": snapshot.event_count,
        "latest_sequence": snapshot.latest_sequence,
        "latest_event_sha256": snapshot.latest_event_sha256,
        "committed_artifact_count": len(snapshot.committed),
        "invalidated_artifact_count": len(snapshot.invalidated),
        "invalidated_logical_ids": sorted(snapshot.invalidated),
        "external_witness_status": snapshot.external_witness_status,
        "limitations": [
            "LOCAL_OWNER_CAN_DELETE_OR_RECREATE_ENTIRE_LEDGER",
            "EXTERNAL_SIGNED_MONOTONIC_WITNESS_ABSENT",
        ],
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
    }


def _event(
    snapshot: RegistrySnapshot,
    *,
    event_kind: str,
    created_at_utc: datetime,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "contract": EVENT_CONTRACT,
        "schema_version": 1,
        "event_kind": event_kind,
        "registry_id": snapshot.registry_id,
        "sequence": snapshot.latest_sequence + 1,
        "previous_event_sha256": snapshot.latest_event_sha256,
        "created_at_utc": _iso(_utc(created_at_utc, "created_at_utc")),
        **_snapshot(payload),
        "external_witness_status": "ABSENT",
        "external_witness_receipt_sha256": None,
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
    }
    return _seal(body, "event_sha256")


def _append_event(root: Path, event: Mapping[str, Any]) -> None:
    sequence = event.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise DojoAIValidityError("validity event sequence is invalid")
    path = root / REGISTRY_DIRNAME / "events" / f"{sequence:06d}.json"
    _write_json_new(path, event, root)


def _artifact_row(
    root: Path,
    artifact_path: Path,
    logical_id: str,
    parents: Sequence[str],
) -> dict[str, Any]:
    logical = _logical(logical_id)
    parent_ids = _parent_ids(parents)
    path = _safe_artifact_path(root, artifact_path)
    raw = path.read_bytes()
    value = _strict_json_bytes(raw)
    contract = value.get("contract")
    if not isinstance(contract, str) or not contract.startswith("QR_DOJO_AI_"):
        raise DojoAIValidityError("validity artifact contract is unsupported")
    own_key, own_sha = _detect_own_seal(value)
    refs = sorted(_all_sha256s(value))
    return {
        "logical_id": logical,
        "relative_path": path.relative_to(root).as_posix(),
        "contract": contract,
        "own_seal_field": own_key,
        "artifact_sha256": own_sha,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_size_bytes": len(raw),
        "parent_logical_ids": list(parent_ids),
        "referenced_sha256s": refs,
    }


def _validate_artifact_row(root: Path, raw: Any) -> dict[str, Any]:
    row = _mapping(raw, "validity artifact row")
    expected_keys = {
        "logical_id",
        "relative_path",
        "contract",
        "own_seal_field",
        "artifact_sha256",
        "raw_sha256",
        "raw_size_bytes",
        "parent_logical_ids",
        "referenced_sha256s",
    }
    if set(row) != expected_keys:
        raise DojoAIValidityError("validity artifact row shape drifted")
    candidate = _artifact_row(
        root,
        root / str(row["relative_path"]),
        str(row["logical_id"]),
        row["parent_logical_ids"],
    )
    if row != candidate:
        raise DojoAIValidityError("TAMPERED_ARTIFACT_BYTES_OR_METADATA")
    return row


def _detect_own_seal(value: Mapping[str, Any]) -> tuple[str, str]:
    matches: list[tuple[str, str]] = []
    for key, digest in value.items():
        if not key.endswith("_sha256") or not isinstance(digest, str):
            continue
        if _HEX64.fullmatch(digest) is None:
            continue
        body = {name: item for name, item in value.items() if name != key}
        if canonical_sha256(body) == digest:
            matches.append((key, digest))
    if len(matches) != 1:
        raise DojoAIValidityError("artifact must expose one canonical own seal")
    return matches[0]


def _all_sha256s(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for item in value.values():
            found.update(_all_sha256s(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_all_sha256s(item))
    elif isinstance(value, str) and _HEX64.fullmatch(value):
        found.add(value)
    return found


def _invalid_closure(
    committed: Mapping[str, Mapping[str, Any]],
    explicit: Mapping[str, Mapping[str, Any]],
) -> tuple[set[str], dict[str, tuple[str, ...]]]:
    invalid = set(explicit)
    paths: dict[str, tuple[str, ...]] = {key: (key,) for key in explicit}
    changed = True
    while changed:
        changed = False
        for logical, row in committed.items():
            if logical in invalid:
                continue
            bad = next(
                (parent for parent in row["parent_logical_ids"] if parent in invalid),
                None,
            )
            if bad is not None:
                invalid.add(logical)
                paths[logical] = paths[bad] + (logical,)
                changed = True
    return invalid, paths


def _assert_depth(committed: Mapping[str, Mapping[str, Any]]) -> None:
    depth: dict[str, int] = {}
    for logical, row in committed.items():
        parent_depth = max((depth[parent] for parent in row["parent_logical_ids"]), default=0)
        depth[logical] = parent_depth + 1
        if depth[logical] > MAX_GRAPH_DEPTH:
            raise DojoAIValidityError("validity graph depth bound exceeded")


def _explicit_invalidations(root: Path) -> dict[str, dict[str, Any]]:
    events_dir = root / REGISTRY_DIRNAME / "events"
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(events_dir.iterdir())[1:]:
        event = _strict_json_bytes(path.read_bytes())
        if event.get("event_kind") == "ARTIFACT_INVALIDATED":
            result[str(event["target_logical_id"])] = event
    return result


def _parent_ids(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DojoAIValidityError("parent logical ids must be a sequence")
    result = tuple(_logical(item) for item in value)
    if len(result) > MAX_PARENTS or len(set(result)) != len(result):
        raise DojoAIValidityError("parent logical ids are duplicated or too large")
    return result


def _logical(value: Any) -> str:
    if not isinstance(value, str) or _LOGICAL_ID.fullmatch(value) is None or ".." in value:
        raise DojoAIValidityError("logical id is invalid")
    return value


def _reason(value: Any) -> str:
    if not isinstance(value, str) or _REASON.fullmatch(value) is None:
        raise DojoAIValidityError("invalidation reason code is invalid")
    return value


def _safe_run_dir(path: Path) -> Path:
    root = path.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise DojoAIValidityError("run directory is absent or unsafe")
    return root


def _safe_artifact_path(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    try:
        relative_candidate = candidate.relative_to(root)
    except ValueError as exc:
        raise DojoAIValidityError("artifact path escaped the run directory") from exc
    cursor = root
    for part in relative_candidate.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise DojoAIValidityError("artifact path contains a symlink")
    resolved = candidate.resolve(strict=True)
    if resolved == root or root not in resolved.parents:
        raise DojoAIValidityError("artifact path escaped the run directory")
    if not resolved.is_file() or candidate.is_symlink():
        raise DojoAIValidityError("artifact path is absent or unsafe")
    if REGISTRY_DIRNAME in resolved.relative_to(root).parts:
        raise DojoAIValidityError("validity ledger cannot commit itself as data")
    return resolved


def _write_json_new(path: Path, value: Mapping[str, Any], root: Path) -> None:
    payload = _canonical_bytes(value) + b"\n"
    if len(payload) > MAX_EVENT_BYTES:
        raise DojoAIValidityError("validity event byte bound exceeded")
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve(strict=True)
    if resolved_parent == root or root not in resolved_parent.parents:
        raise DojoAIValidityError("validity event write escaped run directory")
    if path.exists() or path.is_symlink():
        raise DojoAIValidityError("validity event slot already exists")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _strict_json_bytes(raw: bytes) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise DojoAIValidityError(f"non-finite JSON number: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DojoAIValidityError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoAIValidityError("strict artifact JSON parse failed") from exc
    return _mapping(value, "validity JSON artifact")


def _seal(body: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = _snapshot(body)
    value[key] = canonical_sha256(value)
    return value


def _validate_seal(value: Mapping[str, Any], key: str) -> None:
    digest = _sha(value.get(key), key)
    body = {name: item for name, item in value.items() if name != key}
    if canonical_sha256(body) != digest:
        raise DojoAIValidityError(f"{key} mismatch")


def _snapshot(value: Any) -> Any:
    try:
        return json.loads(_canonical_bytes(value))
    except (TypeError, ValueError) as exc:
        raise DojoAIValidityError("validity artifact is not strict JSON") from exc


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoAIValidityError(f"{label} must be an object")
    return _snapshot(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise DojoAIValidityError(f"{label} is not sha256")
    return value


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DojoAIValidityError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | (os.O_DIRECTORY if hasattr(os, "O_DIRECTORY") else 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "DojoAIValidityError",
    "EVENT_CONTRACT",
    "GENESIS_CONTRACT",
    "RegistrySnapshot",
    "append_artifact_commit",
    "append_invalidation",
    "assert_artifacts_valid",
    "initialize_registry",
    "status_artifact",
    "verify_registry",
]

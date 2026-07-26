"""Fail-closed controller boundary for replay proof promotion.

The pure metric evaluator in this module never authorizes a room.  Only
``issue_paper_ai_inventory_launch_preflight`` may promote a candidate.  That
controller derives every path from one repository root, validates the actual
candidate chain and replay files, appends ``PAPER_ELIGIBLE`` durably, and emits
an immutable token for a future isolated ``paper-ai-inventory`` room.
"""

from __future__ import annotations

import base64
import binascii
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from quant_rabbit.dojo_autonomous_improvement import (
    append_candidate_event,
    validate_research_root,
)
from quant_rabbit.dojo_replay_gates import (
    PROOF_MANIFEST_CONTRACT,
    canonical_proof_manifest_sha256,
    evaluate_inventory_release_proof_ladder,
)
from quant_rabbit.dojo_replay_worker_receipt import (
    ReplayWorkerReceiptError,
    verify_trusted_replay_worker_receipt,
)


LIFECYCLE_DECISION_CONTRACT = "QR_DOJO_REPLAY_LIFECYCLE_DECISION_V1"
PROOF_ARTIFACT_CONTRACT = "QR_DOJO_REPLAY_PROOF_ARTIFACT_V1"
JOB_MANIFEST_CONTRACT = "QR_DOJO_REPLAY_JOB_MANIFEST_V1"
SOURCE_MANIFEST_CONTRACT = "QR_VIRTUAL_REPLAY_SOURCE_MANIFEST_V1"
FUTURE_REGISTRY_CONTRACT = "QR_DOJO_PAPER_ROOM_REGISTRY_V1"
CANDIDATE_SPEC_CONTRACT = "QR_DOJO_AUTONOMOUS_CANDIDATE_SPEC_V1"
REPLAY_JOB_OWNER_CONTRACT = "QR_DOJO_REPLAY_JOB_OWNER_V1"
REPLAY_OUTPUT_MANIFEST_CONTRACT = "QR_DOJO_REPLAY_OUTPUT_MANIFEST_V1"
LAUNCH_PREFLIGHT_CONTRACT = "QR_DOJO_AI_INVENTORY_LAUNCH_PREFLIGHT_V1"
CANONICAL_RESEARCH_RELATIVE_ROOT = Path("research/data/dojo_autonomous_improvement_v1")
CANONICAL_PAPER_AI_ROOMS_RELATIVE_ROOT = Path(
    "research/data/dojo_paper_ai_inventory_v1/rooms"
)
CANONICAL_SOURCE_CAPTURE_MANIFEST_RELATIVE_ROOT = Path(
    "research/data/dojo_paper_ai_inventory_v1/source_capture/manifests"
)
CANONICAL_REPLAY_RELATIVE_ROOT = Path("replay")
WINDOWS = ("TRAIN", "VAL", "S5")
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,254}$")
_SOURCE_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SOURCE_CAPTURE_MANIFEST_KEYS = frozenset(
    {
        "contract",
        "manifest_id",
        "capture_key_id",
        "ed25519_public_key_base64",
        "allowed_source_roles",
        "allowed_provider_kinds",
        "source_adapters",
        "paper_only",
        "order_authority",
        "live_permission",
        "manifest_sha256",
    }
)
_SOURCE_ADAPTER_KEYS = frozenset(
    {
        "source_role",
        "provider_kind",
        "adapter_id",
        "adapter_module",
        "adapter_callable",
        "adapter_executable_sha256",
        "adapter_config_sha256",
    }
)
_SHA_FIELDS = (
    "candidate_id",
    "spec_sha256",
    "policy_sha256",
    "job_manifest_sha256",
    "git_head_sha256",
    "artifact_manifest_sha256",
    "artifact_sha256",
)


class DojoReplayLifecycleError(ValueError):
    """A preregistration, artifact, or future experiment binding is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_unique_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DojoReplayLifecycleError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTITY_RE.fullmatch(value) is None:
        raise DojoReplayLifecycleError(f"{label} is invalid")
    return value


def _paper_guard(value: Mapping[str, Any], label: str) -> None:
    if (
        value.get("paper_only") is not True
        or value.get("order_authority") != "NONE"
        or value.get("live_permission") is not False
    ):
        raise DojoReplayLifecycleError(
            f"{label} must be paper-only with order authority NONE"
        )


def _utc(value: Any, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DojoReplayLifecycleError(f"{label} must be a UTC timestamp") from exc
    else:
        raise DojoReplayLifecycleError(f"{label} must be a UTC timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DojoReplayLifecycleError(f"{label} must be a UTC timestamp")
    return parsed


def _json_bytes(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        raise DojoReplayLifecycleError(f"{label} must be immutable bytes")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DojoReplayLifecycleError(f"{label} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise DojoReplayLifecycleError(f"{label} must be a JSON object")
    return value


def _verify_self_digest(
    value: Mapping[str, Any],
    digest_field: str,
    label: str,
) -> str:
    claimed = _sha(value.get(digest_field), f"{label}.{digest_field}")
    body = {key: item for key, item in value.items() if key != digest_field}
    if claimed != _canonical_sha256(body):
        raise DojoReplayLifecycleError(f"{label} digest mismatch")
    return claimed


def _windows(value: Any, label: str) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(WINDOWS):
        raise DojoReplayLifecycleError(
            f"{label} must contain exactly TRAIN, VAL, and S5"
        )
    normalized: dict[str, dict[str, str]] = {}
    intervals: list[tuple[datetime, datetime, str]] = []
    for name in WINDOWS:
        item = value.get(name)
        if not isinstance(item, Mapping) or set(item) != {
            "from_utc",
            "to_utc",
            "source_sha256",
        }:
            raise DojoReplayLifecycleError(f"{label}.{name} binding is incomplete")
        start = _utc(item.get("from_utc"), f"{label}.{name}.from_utc")
        end = _utc(item.get("to_utc"), f"{label}.{name}.to_utc")
        if start >= end:
            raise DojoReplayLifecycleError(
                f"{label}.{name} window is empty or reversed"
            )
        source_sha256 = _sha(
            item.get("source_sha256"),
            f"{label}.{name}.source_sha256",
        )
        normalized[name] = {
            "from_utc": str(item["from_utc"]),
            "to_utc": str(item["to_utc"]),
            "source_sha256": source_sha256,
        }
        intervals.append((start, end, name))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise DojoReplayLifecycleError(
                f"{label} reuses or overlaps {previous[2]} and {current[2]}"
            )
    return normalized


def _candidate_spec(raw: bytes) -> tuple[dict[str, Any], str]:
    spec = _json_bytes(raw, "candidate spec")
    if spec.get("contract") != CANDIDATE_SPEC_CONTRACT:
        raise DojoReplayLifecycleError("candidate spec contract is invalid")
    if spec.get("family") != "INVENTORY_RELEASE":
        raise DojoReplayLifecycleError("candidate family must be INVENTORY_RELEASE")
    _paper_guard(spec, "candidate spec")
    candidate_id = _sha(spec.get("candidate_id"), "candidate_id")
    spec_sha256 = _verify_self_digest(spec, "spec_sha256", "candidate spec")
    adapter_id = _identity(spec.get("adapter_id"), "candidate adapter_id")
    model_id = _identity(spec.get("model_id"), "candidate model_id")
    config_sha256 = _sha(spec.get("config_sha256"), "candidate config_sha256")
    producer_id = _identity(spec.get("producer_id"), "candidate producer_id")
    source_capture_manifest_sha256 = _sha(
        spec.get("source_capture_manifest_sha256"),
        "candidate source_capture_manifest_sha256",
    )
    windows = _windows(spec.get("windows"), "candidate spec windows")
    gates = spec.get("risk_gates")
    if not isinstance(gates, Mapping):
        raise DojoReplayLifecycleError("candidate risk_gates are missing")
    expected_gates = {
        "min_settlements_per_independent_arm": 30,
        "min_active_days_per_independent_arm": 20,
        "min_independent_stress_pf": 1.25,
        "positive_net": True,
        "positive_expectancy": True,
        "worst_day_not_worse": True,
        "drawdown_not_worse": True,
        "margin_ruin_not_worse": True,
        "unresolved_end_exposure": False,
    }
    for field, expected in expected_gates.items():
        if gates.get(field) != expected:
            raise DojoReplayLifecycleError(
                f"candidate risk gate {field} is not preregistered"
            )
    if spec.get("end_of_replay_forced_close_benefit") is not False:
        raise DojoReplayLifecycleError(
            "candidate permits end-of-replay forced-close benefit"
        )
    return {
        "candidate_id": candidate_id,
        "spec_sha256": spec_sha256,
        "adapter_id": adapter_id,
        "model_id": model_id,
        "config_sha256": config_sha256,
        "producer_id": producer_id,
        "source_capture_manifest_sha256": source_capture_manifest_sha256,
        "windows": windows,
    }, _raw_sha256(raw)


def _job_manifest(
    raw: bytes,
    *,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    job = _json_bytes(raw, "job manifest")
    if job.get("contract") != JOB_MANIFEST_CONTRACT:
        raise DojoReplayLifecycleError("job manifest contract is invalid")
    _paper_guard(job, "job manifest")
    manifest_sha256 = _verify_self_digest(job, "manifest_sha256", "job manifest")
    if job.get("candidate_id") != candidate["candidate_id"]:
        raise DojoReplayLifecycleError("job candidate binding mismatch")
    if job.get("spec_sha256") != candidate["spec_sha256"]:
        raise DojoReplayLifecycleError("job spec binding mismatch")
    for field in (
        "adapter_id",
        "model_id",
        "config_sha256",
        "producer_id",
        "source_capture_manifest_sha256",
    ):
        if job.get(field) != candidate[field]:
            raise DojoReplayLifecycleError(f"job {field} binding mismatch")
    argv = job.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(
            not isinstance(item, str)
            or not item
            or len(item) > 16_384
            or "\x00" in item
            for item in argv
        )
    ):
        raise DojoReplayLifecycleError("job argv is invalid")
    argv_sha256 = _sha(job.get("argv_sha256"), "job argv_sha256")
    if argv_sha256 != _canonical_sha256(argv):
        raise DojoReplayLifecycleError("job argv digest mismatch")
    policy_sha256 = _sha(job.get("policy_sha256"), "policy_sha256")
    git_head = job.get("git_head")
    if (
        not isinstance(git_head, str)
        or len(git_head) != 40
        or any(character not in "0123456789abcdef" for character in git_head)
    ):
        raise DojoReplayLifecycleError("git_head must be a full commit SHA")
    git_head_sha256 = _sha(job.get("git_head_sha256"), "git_head_sha256")
    if git_head_sha256 != _raw_sha256(git_head.encode("ascii")):
        raise DojoReplayLifecycleError("git_head digest mismatch")
    output_manifest_sha256 = _sha(
        job.get("output_manifest_sha256"), "output_manifest_sha256"
    )
    files = job.get("files")
    if not isinstance(files, list):
        raise DojoReplayLifecycleError("job files must be a list")
    file_shas = {item.get("sha256") for item in files if isinstance(item, Mapping)}
    for window in WINDOWS:
        if candidate["windows"][window]["source_sha256"] not in file_shas:
            raise DojoReplayLifecycleError(
                f"job does not bind the {window} source manifest"
            )
    return {
        "manifest_sha256": manifest_sha256,
        "policy_sha256": policy_sha256,
        "git_head": git_head,
        "git_head_sha256": git_head_sha256,
        "output_manifest_sha256": output_manifest_sha256,
        "argv": list(argv),
        "argv_sha256": argv_sha256,
        "adapter_id": candidate["adapter_id"],
        "model_id": candidate["model_id"],
        "config_sha256": candidate["config_sha256"],
        "producer_id": candidate["producer_id"],
        "source_capture_manifest_sha256": candidate["source_capture_manifest_sha256"],
    }


def _source_manifests(
    raw_by_window: Any,
    *,
    windows: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    if not isinstance(raw_by_window, Mapping) or set(raw_by_window) != set(WINDOWS):
        raise DojoReplayLifecycleError(
            "source_manifest_bytes must contain exactly TRAIN, VAL, and S5"
        )
    bindings: dict[str, str] = {}
    for window in WINDOWS:
        raw = raw_by_window[window]
        source = _json_bytes(raw, f"{window} source manifest")
        if source.get("contract") != SOURCE_MANIFEST_CONTRACT:
            raise DojoReplayLifecycleError(
                f"{window} source manifest contract is invalid"
            )
        digest = _raw_sha256(raw)
        if digest != windows[window]["source_sha256"]:
            raise DojoReplayLifecycleError(f"{window} source manifest digest mismatch")
        bindings[window] = digest
    return bindings


def _proof_artifact(
    raw: bytes,
    *,
    candidate: Mapping[str, Any],
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], str, datetime]:
    artifact = _json_bytes(raw, "proof artifact")
    if artifact.get("contract") != PROOF_ARTIFACT_CONTRACT:
        raise DojoReplayLifecycleError("proof artifact contract is invalid")
    _paper_guard(artifact, "proof artifact")
    artifact_sha256 = _verify_self_digest(artifact, "artifact_sha256", "proof artifact")
    if raw != _canonical_bytes(artifact) + b"\n":
        raise DojoReplayLifecycleError(
            "proof artifact bytes are not canonical sealed bytes"
        )
    expected = {
        "candidate_id": candidate["candidate_id"],
        "spec_sha256": candidate["spec_sha256"],
        "policy_sha256": job["policy_sha256"],
        "job_manifest_sha256": job["manifest_sha256"],
        "git_head": job["git_head"],
        "git_head_sha256": job["git_head_sha256"],
        "artifact_manifest_sha256": job["output_manifest_sha256"],
    }
    for field, value in expected.items():
        if artifact.get(field) != value:
            raise DojoReplayLifecycleError(f"proof artifact {field} binding mismatch")
    if (
        _windows(artifact.get("windows"), "proof artifact windows")
        != (candidate["windows"])
    ):
        raise DojoReplayLifecycleError("proof artifact windows changed")
    completed_at = _utc(
        artifact.get("completed_at_utc"),
        "proof artifact completed_at_utc",
    )
    if not isinstance(artifact.get("arms"), list):
        raise DojoReplayLifecycleError("proof artifact arms must be a list")
    return artifact, artifact_sha256, completed_at


def _future_registry(
    raw: bytes,
    *,
    candidate: Mapping[str, Any],
    job: Mapping[str, Any],
    artifact_sha256: str,
    completed_at: datetime,
) -> dict[str, Any]:
    registry = _json_bytes(raw, "future registry")
    if registry.get("contract") != FUTURE_REGISTRY_CONTRACT:
        raise DojoReplayLifecycleError("future registry contract is invalid")
    _paper_guard(registry, "future registry")
    if registry.get("proof_mode") != "candidate":
        raise DojoReplayLifecycleError("future registry proof_mode must be candidate")
    experiment_id = registry.get("experiment_id")
    if (
        not isinstance(experiment_id, str)
        or not experiment_id.startswith("paper-ai-inventory-")
        or Path(experiment_id).name != experiment_id
    ):
        raise DojoReplayLifecycleError(
            "future registry must target a paper-ai-inventory experiment"
        )
    window = registry.get("window")
    if not isinstance(window, Mapping):
        raise DojoReplayLifecycleError("future registry window is missing")
    start = _utc(window.get("start_utc"), "future registry start_utc")
    end = _utc(window.get("end_utc"), "future registry end_utc")
    if start <= completed_at or start >= end:
        raise DojoReplayLifecycleError(
            "future registry is not strictly after proof completion"
        )
    binding = registry.get("proof_binding")
    expected_binding = {
        "candidate_id": candidate["candidate_id"],
        "spec_sha256": candidate["spec_sha256"],
        "policy_sha256": job["policy_sha256"],
        "job_manifest_sha256": job["manifest_sha256"],
        "proof_artifact_sha256": artifact_sha256,
        "git_head": job["git_head"],
        "git_head_sha256": job["git_head_sha256"],
        "adapter_id": candidate["adapter_id"],
        "model_id": candidate["model_id"],
        "config_sha256": candidate["config_sha256"],
        "producer_id": candidate["producer_id"],
        "source_capture_manifest_sha256": candidate["source_capture_manifest_sha256"],
    }
    if not isinstance(binding, Mapping) or dict(binding) != expected_binding:
        raise DojoReplayLifecycleError("future registry proof binding mismatch")
    rooms = registry.get("rooms")
    if not isinstance(rooms, list) or not rooms:
        raise DojoReplayLifecycleError("future registry rooms are missing")
    room_ids: set[str] = set()
    for room in rooms:
        if (
            not isinstance(room, Mapping)
            or room.get("candidate_id") != candidate["candidate_id"]
            or room.get("adapter_id") != candidate["adapter_id"]
            or room.get("model_id") != candidate["model_id"]
            or room.get("config_sha256") != candidate["config_sha256"]
            or room.get("producer_id") != candidate["producer_id"]
            or room.get("source_capture_manifest_sha256")
            != candidate["source_capture_manifest_sha256"]
            or not isinstance(room.get("room_id"), str)
            or not room["room_id"].startswith("paper-ai-inventory-")
            or Path(room["room_id"]).name != room["room_id"]
            or room["room_id"] in room_ids
        ):
            raise DojoReplayLifecycleError("future registry room binding is invalid")
        room_ids.add(room["room_id"])
    return {
        "experiment_id": experiment_id,
        "registry_sha256": _raw_sha256(raw),
        "start_utc": str(window["start_utc"]),
        "end_utc": str(window["end_utc"]),
        "room_ids": sorted(room_ids),
    }


def canonical_proof_artifact_bytes(
    artifact_body: Mapping[str, Any],
) -> bytes:
    """Return canonical newline-terminated bytes with an internal digest."""

    if not isinstance(artifact_body, Mapping):
        raise DojoReplayLifecycleError("proof artifact body must be an object")
    body = dict(artifact_body)
    body.pop("artifact_sha256", None)
    sealed = {**body, "artifact_sha256": _canonical_sha256(body)}
    return _canonical_bytes(sealed) + b"\n"


def seal_proof_artifact_exclusive(
    path: Path | str,
    artifact_bytes: bytes,
) -> dict[str, Any]:
    """Create one immutable proof artifact with O_EXCL and durable fsync."""

    target = Path(path)
    artifact = _json_bytes(artifact_bytes, "proof artifact")
    if artifact.get("contract") != PROOF_ARTIFACT_CONTRACT:
        raise DojoReplayLifecycleError("proof artifact contract is invalid")
    artifact_sha256 = _verify_self_digest(artifact, "artifact_sha256", "proof artifact")
    if artifact_bytes != _canonical_bytes(artifact) + b"\n":
        raise DojoReplayLifecycleError(
            "proof artifact bytes are not canonical sealed bytes"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as exc:
        raise DojoReplayLifecycleError(
            "proof artifact already exists; overwrite is forbidden"
        ) from exc
    try:
        view = memoryview(artifact_bytes)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short proof artifact write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(
        target.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return {
        "path": str(target),
        "artifact_sha256": artifact_sha256,
        "bytes_sha256": _raw_sha256(artifact_bytes),
        "size_bytes": len(artifact_bytes),
        "exclusive": True,
        "fsynced": True,
    }


def evaluate_replay_proof(
    *,
    candidate_spec_bytes: bytes,
    job_manifest_bytes: bytes,
    source_manifest_bytes: Mapping[str, bytes],
    proof_artifact_bytes: bytes,
    future_registry_bytes: bytes,
) -> dict[str, Any]:
    """Evaluate supplied bytes without granting lifecycle eligibility.

    Caller-supplied bytes can demonstrate that a metric envelope is internally
    consistent.  They are not trusted replay provenance and therefore cannot
    produce a ``PAPER_ELIGIBLE`` payload or launch token.
    """

    try:
        candidate, _ = _candidate_spec(candidate_spec_bytes)
        job = _job_manifest(job_manifest_bytes, candidate=candidate)
        _source_manifests(
            source_manifest_bytes,
            windows=candidate["windows"],
        )
        artifact, artifact_sha256, completed_at = _proof_artifact(
            proof_artifact_bytes,
            candidate=candidate,
            job=job,
        )
        _future_registry(
            future_registry_bytes,
            candidate=candidate,
            job=job,
            artifact_sha256=artifact_sha256,
            completed_at=completed_at,
        )
    except DojoReplayLifecycleError as exc:
        return {
            "contract": LIFECYCLE_DECISION_CONTRACT,
            "decision": "MEASUREMENT_BLOCKED",
            "proof_eligible": False,
            "reason": str(exc),
            "gate_decision": None,
            "paper_eligible_event_payload": None,
            "ledger_append_performed": False,
            "paper_room_launched": False,
            "append_controller": "issue_paper_ai_inventory_launch_preflight",
        }

    proof_manifest: dict[str, Any] = {
        "contract": PROOF_MANIFEST_CONTRACT,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "candidate_id": candidate["candidate_id"],
        "spec_sha256": candidate["spec_sha256"],
        "policy_sha256": job["policy_sha256"],
        "artifact_manifest_sha256": job["output_manifest_sha256"],
        "windows": candidate["windows"],
        "arms": artifact["arms"],
    }
    expected_manifest_sha256 = canonical_proof_manifest_sha256(proof_manifest)
    proof_manifest["manifest_sha256"] = expected_manifest_sha256
    gate_decision = evaluate_inventory_release_proof_ladder(
        proof_manifest,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if gate_decision.get("decision") != "PROOF_ELIGIBLE":
        return {
            "contract": LIFECYCLE_DECISION_CONTRACT,
            "decision": "PROOF_REJECTED",
            "proof_eligible": False,
            "reason": "sealed replay proof did not pass independent gates",
            "gate_decision": gate_decision,
            "paper_eligible_event_payload": None,
            "ledger_append_performed": False,
            "paper_room_launched": False,
            "append_controller": "issue_paper_ai_inventory_launch_preflight",
        }

    return {
        "contract": LIFECYCLE_DECISION_CONTRACT,
        "decision": "PROOF_ELIGIBLE_UNTRUSTED",
        "proof_eligible": True,
        "reason": (
            "metric envelope passed, but filesystem provenance and candidate "
            "lifecycle were not authenticated"
        ),
        "gate_decision": gate_decision,
        "paper_eligible_event_payload": None,
        "ledger_append_performed": False,
        "paper_room_launched": False,
        "append_controller": "issue_paper_ai_inventory_launch_preflight",
    }


def canonical_research_root(repository_root: Path | str) -> Path:
    """Return the one accepted autonomous-improvement root."""

    root = _repository_root(repository_root)
    return root / CANONICAL_RESEARCH_RELATIVE_ROOT


def canonical_paper_ai_rooms_root(repository_root: Path | str) -> Path:
    """Return the one accepted future paper-AI room root."""

    root = _repository_root(repository_root)
    return root / CANONICAL_PAPER_AI_ROOMS_RELATIVE_ROOT


def issue_paper_ai_inventory_launch_preflight(
    repository_root: Path | str,
    *,
    candidate_id: str,
    future_registry_path: Path,
    recorded_at_utc: datetime | str,
) -> dict[str, Any]:
    """Promote one proven candidate and issue immutable per-room tokens.

    The controller accepts no candidate ledger path, artifact bytes, metrics,
    room root, or token destination from its caller.  Those are all derived
    from the canonical repository layout.  It never starts a room.
    """

    root = _repository_root(repository_root)
    recorded = _utc(recorded_at_utc, "recorded_at_utc")
    candidate_id = _sha(candidate_id, "candidate_id")
    inspected = _inspect_canonical_candidate(
        root,
        candidate_id=candidate_id,
        future_registry_path=future_registry_path,
        require_paper_eligible=False,
    )
    if recorded >= _utc(
        inspected["registry"]["start_utc"], "future registry start_utc"
    ):
        raise DojoReplayLifecycleError(
            "launch preflight must be issued before the future window starts"
        )

    candidate_dir = inspected["candidate_dir"]
    claim_body = {
        "contract": "QR_DOJO_PAPER_ELIGIBLE_APPEND_CLAIM_V1",
        "candidate_id": candidate_id,
        "proof_artifact_bytes_sha256": inspected["proof_artifact_bytes_sha256"],
        "replay_worker_receipt_sha256": inspected["replay_worker_receipt_sha256"],
        "adapter_id": inspected["candidate"]["adapter_id"],
        "model_id": inspected["candidate"]["model_id"],
        "config_sha256": inspected["candidate"]["config_sha256"],
        "producer_id": inspected["candidate"]["producer_id"],
        "source_capture_manifest_sha256": inspected["candidate"][
            "source_capture_manifest_sha256"
        ],
        "future_registry_sha256": inspected["registry"]["registry_sha256"],
        "recorded_at_utc": _canonical_utc(recorded),
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    claim = {**claim_body, "claim_sha256": _canonical_sha256(claim_body)}
    claim_path = candidate_dir / "paper_eligible_append.claim.json"
    _write_exclusive_fsynced(claim_path, _canonical_bytes(claim) + b"\n")

    event_payload = {
        **inspected["eligibility_payload"],
        "candidate_id": candidate_id,
        "implementation_commit_sha256": inspected["job"]["git_head_sha256"],
        "future_experiment_id": inspected["registry"]["experiment_id"],
        "append_claim_sha256": claim["claim_sha256"],
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    try:
        event, appended = append_candidate_event(
            inspected["research_root"],
            event_type="PAPER_ELIGIBLE",
            payload=event_payload,
            recorded_at_utc=recorded,
        )
        if not appended:
            raise DojoReplayLifecycleError(
                "PAPER_ELIGIBLE was not exclusively appended"
            )
        validated = validate_research_root(inspected["research_root"])
        if (
            validated.get("status") != "VALID"
            or validated["candidate"]["tip_sha256"] != event["event_sha256"]
        ):
            raise DojoReplayLifecycleError(
                "candidate chain did not seal at PAPER_ELIGIBLE"
            )

        tokens: dict[str, dict[str, str]] = {}
        for room_id in inspected["registry"]["room_ids"]:
            room_root = (
                canonical_paper_ai_rooms_root(root)
                / inspected["registry"]["experiment_id"]
                / room_id
            )
            room_root.mkdir(parents=True, exist_ok=True)
            token_body = {
                "contract": LAUNCH_PREFLIGHT_CONTRACT,
                "candidate_id": candidate_id,
                "adapter_id": inspected["candidate"]["adapter_id"],
                "model_id": inspected["candidate"]["model_id"],
                "config_sha256": inspected["candidate"]["config_sha256"],
                "producer_id": inspected["candidate"]["producer_id"],
                "source_capture_manifest_sha256": inspected["candidate"][
                    "source_capture_manifest_sha256"
                ],
                "spec_sha256": inspected["candidate"]["spec_sha256"],
                "policy_sha256": inspected["job"]["policy_sha256"],
                "experiment_id": inspected["registry"]["experiment_id"],
                "room_id": room_id,
                "paper_eligible_event_sha256": event["event_sha256"],
                "candidate_lifecycle_ledger_tip_sha256": event["event_sha256"],
                "append_claim_sha256": claim["claim_sha256"],
                "job_manifest_sha256": inspected["job"]["manifest_sha256"],
                "job_owner_sha256": inspected["job_owner"]["owner_sha256"],
                "proof_artifact_sha256": inspected["artifact_sha256"],
                "proof_artifact_bytes_sha256": inspected["proof_artifact_bytes_sha256"],
                "proof_manifest_sha256": inspected["proof_manifest_sha256"],
                "replay_worker_receipt_sha256": inspected[
                    "replay_worker_receipt_sha256"
                ],
                "source_manifest_sha256s": inspected["source_bindings"],
                "future_registry_sha256": inspected["registry"]["registry_sha256"],
                "future_window": {
                    "start_utc": inspected["registry"]["start_utc"],
                    "end_utc": inspected["registry"]["end_utc"],
                },
                "git_head": inspected["job"]["git_head"],
                "git_head_sha256": inspected["job"]["git_head_sha256"],
                "issued_at_utc": _canonical_utc(recorded),
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
                "paper_room_launched": False,
            }
            token = {
                **token_body,
                "launch_preflight_token_sha256": _canonical_sha256(token_body),
            }
            token_path = room_root / "launch_preflight.json"
            _write_exclusive_fsynced(
                token_path,
                _canonical_bytes(token) + b"\n",
            )
            tokens[room_id] = {
                "path": str(token_path),
                "launch_preflight_token_sha256": token["launch_preflight_token_sha256"],
            }
    except Exception:
        # The exclusive claim is intentionally retained as a crash marker.
        # It prevents a second controller from silently repeating a partially
        # completed eligibility transition.
        raise

    return {
        "contract": LIFECYCLE_DECISION_CONTRACT,
        "decision": "PAPER_ELIGIBLE_APPENDED",
        "candidate_id": candidate_id,
        "paper_eligible_event_sha256": event["event_sha256"],
        "candidate_lifecycle_ledger_tip_sha256": event["event_sha256"],
        "launch_preflight_tokens": tokens,
        "ledger_append_performed": True,
        "exclusive_append_claim": True,
        "fsynced": True,
        "paper_room_launched": False,
    }


def verify_paper_ai_inventory_launch_preflight(
    repository_root: Path | str,
    *,
    experiment_id: str,
    room_id: str,
) -> dict[str, Any]:
    """Revalidate a canonical room token and all evidence behind it."""

    root = _repository_root(repository_root)
    for label, value in (
        ("experiment_id", experiment_id),
        ("room_id", room_id),
    ):
        if (
            not isinstance(value, str)
            or not value.startswith("paper-ai-inventory-")
            or Path(value).name != value
        ):
            raise DojoReplayLifecycleError(f"{label} is not an isolated AI id")
    token_path = (
        canonical_paper_ai_rooms_root(root)
        / experiment_id
        / room_id
        / "launch_preflight.json"
    )
    token_raw = _read_canonical_file(
        token_path,
        root=canonical_paper_ai_rooms_root(root),
        label="launch preflight token",
    )
    token = _json_bytes(token_raw, "launch preflight token")
    if token.get("contract") != LAUNCH_PREFLIGHT_CONTRACT:
        raise DojoReplayLifecycleError("launch preflight contract is invalid")
    _paper_guard(token, "launch preflight")
    if token.get("paper_room_launched") is not False:
        raise DojoReplayLifecycleError("launch preflight launch state is invalid")
    claimed = _sha(
        token.get("launch_preflight_token_sha256"),
        "launch_preflight_token_sha256",
    )
    body = {
        key: value
        for key, value in token.items()
        if key != "launch_preflight_token_sha256"
    }
    if claimed != _canonical_sha256(body):
        raise DojoReplayLifecycleError("launch preflight digest mismatch")
    if token.get("experiment_id") != experiment_id or token.get("room_id") != room_id:
        raise DojoReplayLifecycleError("launch preflight room binding mismatch")

    candidate_id = _sha(token.get("candidate_id"), "candidate_id")
    registry_path = _registry_path_from_token(root, candidate_id)
    inspected = _inspect_canonical_candidate(
        root,
        candidate_id=candidate_id,
        future_registry_path=registry_path,
        require_paper_eligible=True,
    )
    expected = {
        "adapter_id": inspected["candidate"]["adapter_id"],
        "model_id": inspected["candidate"]["model_id"],
        "config_sha256": inspected["candidate"]["config_sha256"],
        "producer_id": inspected["candidate"]["producer_id"],
        "source_capture_manifest_sha256": inspected["candidate"][
            "source_capture_manifest_sha256"
        ],
        "spec_sha256": inspected["candidate"]["spec_sha256"],
        "policy_sha256": inspected["job"]["policy_sha256"],
        "paper_eligible_event_sha256": inspected["paper_eligible_event_sha256"],
        "candidate_lifecycle_ledger_tip_sha256": inspected[
            "candidate_lifecycle_ledger_tip_sha256"
        ],
        "append_claim_sha256": inspected["append_claim_sha256"],
        "job_manifest_sha256": inspected["job"]["manifest_sha256"],
        "job_owner_sha256": inspected["job_owner"]["owner_sha256"],
        "proof_artifact_sha256": inspected["artifact_sha256"],
        "proof_artifact_bytes_sha256": inspected["proof_artifact_bytes_sha256"],
        "proof_manifest_sha256": inspected["proof_manifest_sha256"],
        "replay_worker_receipt_sha256": inspected["replay_worker_receipt_sha256"],
        "source_manifest_sha256s": inspected["source_bindings"],
        "future_registry_sha256": inspected["registry"]["registry_sha256"],
        "git_head": inspected["job"]["git_head"],
        "git_head_sha256": inspected["job"]["git_head_sha256"],
        "future_window": {
            "start_utc": inspected["registry"]["start_utc"],
            "end_utc": inspected["registry"]["end_utc"],
        },
    }
    for field, value in expected.items():
        if token.get(field) != value:
            raise DojoReplayLifecycleError(f"launch preflight {field} binding mismatch")
    if room_id not in inspected["registry"]["room_ids"]:
        raise DojoReplayLifecycleError("launch preflight room is not registered")
    _validate_all_room_tokens(
        root,
        registry=inspected["registry"],
        expected=expected,
        candidate_id=candidate_id,
        paper_eligible_event_sha256=inspected["paper_eligible_event_sha256"],
    )
    return dict(token)


def _inspect_canonical_candidate(
    repository_root: Path,
    *,
    candidate_id: str,
    future_registry_path: Path,
    require_paper_eligible: bool,
) -> dict[str, Any]:
    research_root = canonical_research_root(repository_root)
    candidate_dir = research_root / "candidates" / candidate_id
    replay_root = candidate_dir / CANONICAL_REPLAY_RELATIVE_ROOT
    paths = {
        "spec": candidate_dir / "spec.json",
        "job": replay_root / "job_manifest.json",
        "owner": replay_root / "job_owner.json",
        "output": replay_root / "output_manifest.json",
        "artifact": replay_root / "proof_artifact.json",
        "worker_receipt": replay_root / "worker_receipt.json",
    }
    research = validate_research_root(research_root)
    if research.get("status") != "VALID":
        raise DojoReplayLifecycleError("canonical candidate research root is invalid")
    ledger_path = research_root / "candidate_ledger.jsonl"
    with _shared_file_lock(ledger_path) as handle:
        rows = _ledger_rows(handle.read())
        candidate, candidate_spec_bytes_sha256 = _candidate_spec(
            _read_canonical_file(
                paths["spec"],
                root=candidate_dir,
                label="candidate spec",
                canonical_required=False,
            )
        )
        if candidate["candidate_id"] != candidate_id:
            raise DojoReplayLifecycleError("candidate directory/spec mismatch")
        source_capture_manifest = _validate_source_capture_manifest(
            repository_root,
            candidate["source_capture_manifest_sha256"],
        )
        chain = _candidate_chain(
            rows,
            candidate_id=candidate_id,
            require_paper_eligible=require_paper_eligible,
        )

        job_raw = _read_canonical_file(
            paths["job"], root=replay_root, label="job manifest"
        )
        job = _job_manifest(job_raw, candidate=candidate)
        expected_job_lock = {
            "job_manifest_sha256": job["manifest_sha256"],
            "git_head_sha256": job["git_head_sha256"],
            "spec_sha256": candidate["spec_sha256"],
            "policy_sha256": job["policy_sha256"],
            "output_manifest_sha256": job["output_manifest_sha256"],
            "argv": job["argv"],
            "argv_sha256": job["argv_sha256"],
            "adapter_id": candidate["adapter_id"],
            "model_id": candidate["model_id"],
            "config_sha256": candidate["config_sha256"],
            "producer_id": candidate["producer_id"],
            "source_capture_manifest_sha256": candidate[
                "source_capture_manifest_sha256"
            ],
        }
        for field, value in expected_job_lock.items():
            if chain["job_lock"].get(field) != value:
                raise DojoReplayLifecycleError(
                    f"candidate chain does not bind job {field}"
                )
        _validate_git_head(repository_root, job)
        job_value = _json_bytes(job_raw, "job manifest")

        source_raw: dict[str, bytes] = {}
        source_file_bindings: list[dict[str, str]] = []
        expected_job_files: list[dict[str, str]] = []
        for window in WINDOWS:
            source_path = replay_root / "source_manifests" / f"{window}.json"
            raw = _read_canonical_file(
                source_path,
                root=replay_root,
                label=f"{window} source manifest",
            )
            source_raw[window] = raw
            expected_job_files.append(
                {
                    "path": str(source_path.relative_to(repository_root)),
                    "sha256": _raw_sha256(raw),
                }
            )
            source_file_bindings.extend(
                _validate_source_file_bytes(
                    _json_bytes(raw, f"{window} source manifest"),
                    repository_root=repository_root,
                    window=window,
                )
            )
        if job_value.get("files") != expected_job_files:
            raise DojoReplayLifecycleError(
                "job manifest files are not the canonical source manifests"
            )
        source_bindings = _source_manifests(
            source_raw,
            windows=candidate["windows"],
        )

        output_raw = _read_canonical_file(
            paths["output"], root=replay_root, label="output manifest"
        )
        if _raw_sha256(output_raw) != job["output_manifest_sha256"]:
            raise DojoReplayLifecycleError("actual output manifest digest mismatch")
        output = _json_bytes(output_raw, "output manifest")
        if (
            output.get("contract") != REPLAY_OUTPUT_MANIFEST_CONTRACT
            or output.get("candidate_id") != candidate_id
            or output.get("spec_sha256") != candidate["spec_sha256"]
            or output.get("policy_sha256") != job["policy_sha256"]
            or output.get("git_head") != job["git_head"]
            or output.get("source_manifest_sha256s") != source_bindings
            or output.get("adapter_id") != candidate["adapter_id"]
            or output.get("model_id") != candidate["model_id"]
            or output.get("config_sha256") != candidate["config_sha256"]
            or output.get("producer_id") != candidate["producer_id"]
            or output.get("source_capture_manifest_sha256")
            != candidate["source_capture_manifest_sha256"]
        ):
            raise DojoReplayLifecycleError("output manifest binding mismatch")
        _paper_guard(output, "output manifest")

        owner_raw = _read_canonical_file(
            paths["owner"], root=replay_root, label="job owner"
        )
        job_owner = _job_owner(
            owner_raw,
            candidate_id=candidate_id,
            job=job,
            job_lock=chain["job_lock"],
            replay_root=replay_root,
            repository_root=repository_root,
        )

        artifact_raw = _read_canonical_file(
            paths["artifact"], root=replay_root, label="proof artifact"
        )
        artifact, artifact_sha256, completed_at = _proof_artifact(
            artifact_raw,
            candidate=candidate,
            job=job,
        )
        proof_manifest = {
            "contract": PROOF_MANIFEST_CONTRACT,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "candidate_id": candidate_id,
            "spec_sha256": candidate["spec_sha256"],
            "policy_sha256": job["policy_sha256"],
            "artifact_manifest_sha256": job["output_manifest_sha256"],
            "windows": candidate["windows"],
            "arms": artifact["arms"],
        }
        proof_manifest_sha256 = canonical_proof_manifest_sha256(proof_manifest)
        proof_manifest["manifest_sha256"] = proof_manifest_sha256
        gate = evaluate_inventory_release_proof_ladder(
            proof_manifest,
            expected_manifest_sha256=proof_manifest_sha256,
        )
        if gate.get("decision") != "PROOF_ELIGIBLE":
            raise DojoReplayLifecycleError(
                "actual proof artifact failed independent replay gates"
            )
        proof_bytes_sha256 = _raw_sha256(artifact_raw)
        try:
            worker_receipt = verify_trusted_replay_worker_receipt(
                repository_root,
                paths["worker_receipt"],
            )
        except ReplayWorkerReceiptError as exc:
            raise DojoReplayLifecycleError(
                f"trusted replay worker receipt failed: {exc}"
            ) from exc
        expected_worker_bindings = {
            "adapter_id": candidate["adapter_id"],
            "model_id": candidate["model_id"],
            "config_sha256": candidate["config_sha256"],
            "producer_id": candidate["producer_id"],
            "argv": job["argv"],
            "argv_sha256": job["argv_sha256"],
            "git_head": job["git_head"],
            "git_head_sha256": job["git_head_sha256"],
            "candidate_id": candidate_id,
            "spec_sha256": candidate["spec_sha256"],
            "policy_sha256": job["policy_sha256"],
            "job_manifest_sha256": job["manifest_sha256"],
            "output_manifest_sha256": job["output_manifest_sha256"],
            "source_files": source_file_bindings,
            "windows": candidate["windows"],
            "results_artifact_path": str(
                paths["artifact"].relative_to(repository_root)
            ),
            "results_artifact_sha256": proof_bytes_sha256,
        }
        for field, value in expected_worker_bindings.items():
            if worker_receipt.get(field) != value:
                raise DojoReplayLifecycleError(
                    f"trusted replay worker receipt {field} binding mismatch"
                )
        if (
            _utc(
                worker_receipt.get("completed_at_utc"),
                "replay worker receipt completed_at_utc",
            )
            < completed_at
        ):
            raise DojoReplayLifecycleError(
                "replay worker receipt predates the proof artifact"
            )
        worker_receipt_sha256 = _sha(
            worker_receipt.get("receipt_sha256"),
            "replay worker receipt_sha256",
        )
        passed = chain["replay_passed"].get("payload", {})
        expected_pass_bindings = {
            "proof_artifact_sha256": artifact_sha256,
            "proof_artifact_bytes_sha256": proof_bytes_sha256,
            "proof_manifest_sha256": proof_manifest_sha256,
            "job_manifest_sha256": job["manifest_sha256"],
            "replay_worker_receipt_sha256": worker_receipt_sha256,
        }
        for field, value in expected_pass_bindings.items():
            if passed.get(field) != value:
                raise DojoReplayLifecycleError(
                    f"REPLAY_PASSED does not bind actual {field}"
                )

        registry_raw = _read_future_registry(
            repository_root, candidate_id, future_registry_path
        )
        registry = _future_registry(
            registry_raw,
            candidate=candidate,
            job=job,
            artifact_sha256=artifact_sha256,
            completed_at=completed_at,
        )
        eligibility_payload = {
            "spec_sha256": candidate["spec_sha256"],
            "candidate_spec_bytes_sha256": candidate_spec_bytes_sha256,
            "policy_sha256": job["policy_sha256"],
            "job_manifest_sha256": job["manifest_sha256"],
            "job_manifest_bytes_sha256": _raw_sha256(job_raw),
            "job_owner_sha256": job_owner["owner_sha256"],
            "source_manifest_sha256s": source_bindings,
            "proof_artifact_sha256": artifact_sha256,
            "proof_artifact_bytes_sha256": proof_bytes_sha256,
            "proof_manifest_sha256": proof_manifest_sha256,
            "replay_worker_receipt_sha256": worker_receipt_sha256,
            "adapter_id": candidate["adapter_id"],
            "model_id": candidate["model_id"],
            "config_sha256": candidate["config_sha256"],
            "producer_id": candidate["producer_id"],
            "future_registry_sha256": registry["registry_sha256"],
            "future_window": {
                "start_utc": registry["start_utc"],
                "end_utc": registry["end_utc"],
            },
        }
        result = {
            "research_root": research_root,
            "candidate_dir": candidate_dir,
            "candidate": candidate,
            "source_capture_manifest": source_capture_manifest,
            "job": job,
            "job_owner": job_owner,
            "source_bindings": source_bindings,
            "artifact_sha256": artifact_sha256,
            "proof_artifact_bytes_sha256": proof_bytes_sha256,
            "proof_manifest_sha256": proof_manifest_sha256,
            "replay_worker_receipt_sha256": worker_receipt_sha256,
            "registry": registry,
            "eligibility_payload": eligibility_payload,
        }
        if require_paper_eligible:
            eligible = chain["paper_eligible"]
            payload = eligible.get("payload", {})
            claim_raw = _read_canonical_file(
                candidate_dir / "paper_eligible_append.claim.json",
                root=candidate_dir,
                label="paper eligibility append claim",
            )
            claim = _json_bytes(claim_raw, "paper eligibility append claim")
            if claim.get("contract") != "QR_DOJO_PAPER_ELIGIBLE_APPEND_CLAIM_V1":
                raise DojoReplayLifecycleError(
                    "paper eligibility append claim contract is invalid"
                )
            _paper_guard(claim, "paper eligibility append claim")
            claim_sha256 = _verify_self_digest(
                claim,
                "claim_sha256",
                "paper eligibility append claim",
            )
            expected_claim = {
                "candidate_id": candidate_id,
                "proof_artifact_bytes_sha256": proof_bytes_sha256,
                "replay_worker_receipt_sha256": worker_receipt_sha256,
                "adapter_id": candidate["adapter_id"],
                "model_id": candidate["model_id"],
                "config_sha256": candidate["config_sha256"],
                "producer_id": candidate["producer_id"],
                "source_capture_manifest_sha256": candidate[
                    "source_capture_manifest_sha256"
                ],
                "future_registry_sha256": registry["registry_sha256"],
            }
            for field, value in expected_claim.items():
                if claim.get(field) != value:
                    raise DojoReplayLifecycleError(
                        f"paper eligibility append claim {field} mismatch"
                    )
            _utc(
                claim.get("recorded_at_utc"),
                "paper eligibility append claim recorded_at_utc",
            )
            for field, value in {
                **eligibility_payload,
                "candidate_id": candidate_id,
                "implementation_commit_sha256": job["git_head_sha256"],
                "future_experiment_id": registry["experiment_id"],
                "append_claim_sha256": claim_sha256,
            }.items():
                if payload.get(field) != value:
                    raise DojoReplayLifecycleError(
                        f"PAPER_ELIGIBLE {field} binding mismatch"
                    )
            result["paper_eligible_event_sha256"] = eligible["event_sha256"]
            result["candidate_lifecycle_ledger_tip_sha256"] = research["candidate"][
                "tip_sha256"
            ]
            result["append_claim_sha256"] = claim_sha256
        return result


def _validate_all_room_tokens(
    repository_root: Path,
    *,
    registry: Mapping[str, Any],
    expected: Mapping[str, Any],
    candidate_id: str,
    paper_eligible_event_sha256: str,
) -> None:
    rooms_root = canonical_paper_ai_rooms_root(repository_root)
    for room_id in registry["room_ids"]:
        path = (
            rooms_root / registry["experiment_id"] / room_id / "launch_preflight.json"
        )
        raw = _read_canonical_file(
            path,
            root=rooms_root,
            label=f"{room_id} launch preflight token",
        )
        item = _json_bytes(raw, f"{room_id} launch preflight token")
        if item.get("contract") != LAUNCH_PREFLIGHT_CONTRACT:
            raise DojoReplayLifecycleError(
                f"{room_id} launch preflight contract is invalid"
            )
        _paper_guard(item, f"{room_id} launch preflight")
        digest = _sha(
            item.get("launch_preflight_token_sha256"),
            f"{room_id} launch_preflight_token_sha256",
        )
        body = {
            key: value
            for key, value in item.items()
            if key != "launch_preflight_token_sha256"
        }
        if digest != _canonical_sha256(body):
            raise DojoReplayLifecycleError(
                f"{room_id} launch preflight digest mismatch"
            )
        if (
            item.get("candidate_id") != candidate_id
            or item.get("experiment_id") != registry["experiment_id"]
            or item.get("room_id") != room_id
            or item.get("paper_eligible_event_sha256") != paper_eligible_event_sha256
            or item.get("paper_room_launched") is not False
        ):
            raise DojoReplayLifecycleError(
                f"{room_id} launch preflight identity mismatch"
            )
        for field, value in expected.items():
            if item.get(field) != value:
                raise DojoReplayLifecycleError(
                    f"{room_id} launch preflight {field} mismatch"
                )


def _candidate_chain(
    rows: list[dict[str, Any]],
    *,
    candidate_id: str,
    require_paper_eligible: bool,
) -> dict[str, Any]:
    registrations = [
        row
        for row in rows
        if row.get("event_type") == "CANDIDATE_PREREGISTERED"
        and row.get("payload", {}).get("candidate_id") == candidate_id
    ]
    if len(registrations) != 1:
        raise DojoReplayLifecycleError(
            "candidate preregistration is missing or ambiguous"
        )
    relevant = [
        row
        for row in rows
        if row.get("payload", {}).get("candidate_id") == candidate_id
    ]
    jobs = [
        row
        for row in relevant
        if row.get("event_type") in {"REPLAY_STARTED", "REPLAY_RETRY_STARTED"}
    ]
    passes = [row for row in relevant if row.get("event_type") == "REPLAY_PASSED"]
    eligible = [row for row in relevant if row.get("event_type") == "PAPER_ELIGIBLE"]
    if not jobs or len(passes) != 1:
        raise DojoReplayLifecycleError(
            "candidate must bind one passed replay and at least one sealed job"
        )
    selected_job = jobs[-1]
    if rows.index(selected_job) >= rows.index(passes[0]):
        raise DojoReplayLifecycleError("candidate replay transition order is invalid")
    job_lock = selected_job.get("payload", {}).get("job_lock")
    if not isinstance(job_lock, Mapping):
        raise DojoReplayLifecycleError("candidate replay job_lock is missing")
    if require_paper_eligible:
        if len(eligible) != 1 or rows[-1] != eligible[0]:
            raise DojoReplayLifecycleError(
                "candidate chain is not terminal at PAPER_ELIGIBLE"
            )
    elif eligible:
        raise DojoReplayLifecycleError("candidate is already PAPER_ELIGIBLE")
    return {
        "job_lock": dict(job_lock),
        "replay_passed": passes[0],
        "paper_eligible": eligible[0] if eligible else None,
    }


def _job_owner(
    raw: bytes,
    *,
    candidate_id: str,
    job: Mapping[str, str],
    job_lock: Mapping[str, Any],
    replay_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    owner = _json_bytes(raw, "job owner")
    if owner.get("contract") != REPLAY_JOB_OWNER_CONTRACT:
        raise DojoReplayLifecycleError("job owner contract is invalid")
    _paper_guard(owner, "job owner")
    owner_sha256 = _verify_self_digest(owner, "owner_sha256", "job owner")
    expected = {
        "candidate_id": candidate_id,
        "job_manifest_sha256": job["manifest_sha256"],
        "pid": job_lock.get("pid"),
        "screen_name": job_lock.get("screen_name"),
        "process_command_sha256": job_lock.get("process_command_sha256"),
        "output_directory": str(replay_root.relative_to(repository_root)),
        "status": "COMPLETED",
    }
    for field, value in expected.items():
        if owner.get(field) != value:
            raise DojoReplayLifecycleError(f"job owner {field} binding mismatch")
    if job_lock.get("job_owner_sha256") != owner_sha256:
        raise DojoReplayLifecycleError("candidate chain does not bind job owner")
    _utc(owner.get("completed_at_utc"), "job owner completed_at_utc")
    return dict(owner)


def _validate_source_file_bytes(
    source: Mapping[str, Any],
    *,
    repository_root: Path,
    window: str,
) -> list[dict[str, str]]:
    files = source.get("files")
    if not isinstance(files, list) or not files:
        raise DojoReplayLifecycleError(f"{window} source files are missing")
    seen: set[str] = set()
    bindings: list[dict[str, str]] = []
    for item in files:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256"}
            or not isinstance(item.get("path"), str)
        ):
            raise DojoReplayLifecycleError(f"{window} source file binding is invalid")
        path_text = str(item["path"])
        if path_text in seen:
            raise DojoReplayLifecycleError(f"{window} source file is duplicated")
        seen.add(path_text)
        path = repository_root / path_text
        raw = _read_canonical_file(
            path,
            root=repository_root,
            label=f"{window} source file",
            canonical_required=False,
        )
        expected_sha256 = _sha(item.get("sha256"), "source file sha256")
        if _raw_sha256(raw) != expected_sha256:
            raise DojoReplayLifecycleError(
                f"{window} source file bytes digest mismatch"
            )
        bindings.append(
            {
                "window": window,
                "path": path_text,
                "sha256": expected_sha256,
            }
        )
    return bindings


def _validate_source_capture_manifest(
    repository_root: Path,
    file_sha256: str,
) -> dict[str, Any]:
    manifest_root = repository_root / CANONICAL_SOURCE_CAPTURE_MANIFEST_RELATIVE_ROOT
    path = manifest_root / f"{file_sha256}.json"
    raw = _read_regular_no_follow(
        path,
        allowed_root=manifest_root,
        max_bytes=256 * 1024,
        label="source capture manifest",
    )
    if _raw_sha256(raw) != file_sha256:
        raise DojoReplayLifecycleError(
            "source capture manifest raw bytes digest mismatch"
        )
    manifest = _json_bytes(raw, "source capture manifest")
    if raw != _canonical_bytes(manifest) + b"\n":
        raise DojoReplayLifecycleError(
            "source capture manifest bytes are not canonical"
        )
    if set(manifest) != _SOURCE_CAPTURE_MANIFEST_KEYS:
        raise DojoReplayLifecycleError("source capture manifest schema is invalid")
    if manifest.get("contract") != "QR_DOJO_AI_SOURCE_CAPTURE_MANIFEST_V1":
        raise DojoReplayLifecycleError("source capture manifest contract is invalid")
    _identity(manifest.get("manifest_id"), "source capture manifest_id")
    _identity(manifest.get("capture_key_id"), "source capture key id")
    public_key = manifest.get("ed25519_public_key_base64")
    if not isinstance(public_key, str) or not public_key:
        raise DojoReplayLifecycleError("source capture public key is invalid")
    try:
        decoded_public_key = base64.b64decode(public_key, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise DojoReplayLifecycleError("source capture public key is invalid") from exc
    if (
        len(decoded_public_key) != 32
        or base64.b64encode(decoded_public_key).decode("ascii") != public_key
    ):
        raise DojoReplayLifecycleError(
            "source capture public key is not canonical raw Ed25519"
        )
    roles = manifest.get("allowed_source_roles")
    if (
        not isinstance(roles, list)
        or not roles
        or any(
            not isinstance(item, str) or _SOURCE_ROLE_RE.fullmatch(item) is None
            for item in roles
        )
        or roles != sorted(set(roles))
    ):
        raise DojoReplayLifecycleError(
            "source capture allowed_source_roles are invalid"
        )
    providers = manifest.get("allowed_provider_kinds")
    if (
        not isinstance(providers, list)
        or not providers
        or any(
            not isinstance(item, str) or _IDENTITY_RE.fullmatch(item) is None
            for item in providers
        )
        or providers != sorted(set(providers))
    ):
        raise DojoReplayLifecycleError(
            "source capture allowed_provider_kinds are invalid"
        )
    raw_adapters = manifest.get("source_adapters")
    if not isinstance(raw_adapters, list) or not raw_adapters:
        raise DojoReplayLifecycleError("source capture source_adapters are missing")
    adapters: list[dict[str, str]] = []
    for item in raw_adapters:
        if not isinstance(item, Mapping) or set(item) != _SOURCE_ADAPTER_KEYS:
            raise DojoReplayLifecycleError(
                "source capture source_adapters schema is invalid"
            )
        source_role = item.get("source_role")
        if (
            not isinstance(source_role, str)
            or _SOURCE_ROLE_RE.fullmatch(source_role) is None
        ):
            raise DojoReplayLifecycleError(
                "source capture adapter source_role is invalid"
            )
        adapters.append(
            {
                "source_role": source_role,
                "provider_kind": _identity(
                    item.get("provider_kind"),
                    "source capture adapter provider_kind",
                ),
                "adapter_id": _identity(
                    item.get("adapter_id"),
                    "source capture adapter_id",
                ),
                "adapter_module": _identity(
                    item.get("adapter_module"),
                    "source capture adapter_module",
                ),
                "adapter_callable": _identity(
                    item.get("adapter_callable"),
                    "source capture adapter_callable",
                ),
                "adapter_executable_sha256": _sha(
                    item.get("adapter_executable_sha256"),
                    "source capture adapter_executable_sha256",
                ),
                "adapter_config_sha256": _sha(
                    item.get("adapter_config_sha256"),
                    "source capture adapter_config_sha256",
                ),
            }
        )
    if adapters != sorted(adapters, key=lambda item: item["source_role"]):
        raise DojoReplayLifecycleError(
            "source capture source_adapters are not sorted by source_role"
        )
    adapter_roles = [item["source_role"] for item in adapters]
    if len(adapter_roles) != len(set(adapter_roles)) or adapter_roles != roles:
        raise DojoReplayLifecycleError(
            "source capture adapters do not cover allowed roles exactly"
        )
    if sorted({item["provider_kind"] for item in adapters}) != providers:
        raise DojoReplayLifecycleError(
            "source capture adapter providers do not match allowlist"
        )
    _paper_guard(manifest, "source capture manifest")
    semantic_sha256 = _verify_self_digest(
        manifest,
        "manifest_sha256",
        "source capture manifest",
    )
    return {
        "file_sha256": file_sha256,
        "manifest_sha256": semantic_sha256,
        "manifest_id": manifest["manifest_id"],
        "capture_key_id": manifest["capture_key_id"],
        "allowed_source_roles": list(roles),
        "allowed_provider_kinds": list(providers),
        "source_adapters": adapters,
    }


def _read_regular_no_follow(
    path: Path,
    *,
    allowed_root: Path,
    max_bytes: int,
    label: str,
) -> bytes:
    descriptor: int | None = None
    try:
        root = allowed_root.resolve(strict=True)
        target = Path(os.path.abspath(path))
        relative = target.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise OSError("symlink component")
        lexical_stat = os.lstat(target)
        if stat.S_ISLNK(lexical_stat.st_mode):
            raise OSError("symlink target")
        descriptor = os.open(
            target,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_dev != lexical_stat.st_dev
            or before.st_ino != lexical_stat.st_ino
            or before.st_size <= 0
            or before.st_size > max_bytes
        ):
            raise OSError("invalid regular file")
        raw = bytearray()
        while len(raw) <= max_bytes:
            chunk = os.read(
                descriptor,
                min(64 * 1024, max_bytes + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
    except (OSError, ValueError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise DojoReplayLifecycleError(
            f"{label} is missing or outside its canonical root"
        ) from exc
    os.close(descriptor)
    if (
        len(raw) > max_bytes
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise DojoReplayLifecycleError(f"{label} changed while reading")
    return bytes(raw)


def _read_future_registry(
    repository_root: Path,
    candidate_id: str,
    supplied: Path,
) -> bytes:
    if not isinstance(supplied, Path):
        raise DojoReplayLifecycleError("future_registry_path must be an explicit Path")
    expected_name = f"dojo_paper_rooms_ai_inventory_{candidate_id}.json"
    expected = repository_root / "config" / expected_name
    try:
        if supplied.resolve(strict=True) != expected.resolve(strict=True):
            raise DojoReplayLifecycleError(
                "future registry is not at its canonical config path"
            )
    except OSError as exc:
        raise DojoReplayLifecycleError("future registry is missing") from exc
    return _read_canonical_file(
        expected,
        root=repository_root / "config",
        label="future registry",
    )


def _registry_path_from_token(repository_root: Path, candidate_id: str) -> Path:
    return (
        repository_root
        / "config"
        / f"dojo_paper_rooms_ai_inventory_{candidate_id}.json"
    )


def _validate_git_head(repository_root: Path, job: Mapping[str, str]) -> None:
    try:
        process = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DojoReplayLifecycleError("repository git HEAD is unavailable") from exc
    current = process.stdout.strip()
    if current != job["git_head"]:
        raise DojoReplayLifecycleError("replay git HEAD is not the repository HEAD")


def _repository_root(value: Path | str) -> Path:
    if not isinstance(value, (Path, str)):
        raise DojoReplayLifecycleError("repository_root must be a path")
    try:
        root = Path(value).resolve(strict=True)
    except OSError as exc:
        raise DojoReplayLifecycleError("repository_root is missing") from exc
    if not root.is_dir():
        raise DojoReplayLifecycleError("repository_root is not a directory")
    return root


def _read_canonical_file(
    path: Path,
    *,
    root: Path,
    label: str,
    canonical_required: bool = True,
) -> bytes:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
        if path.is_symlink() or not resolved.is_file():
            raise OSError
        raw = resolved.read_bytes()
    except (OSError, ValueError) as exc:
        raise DojoReplayLifecycleError(
            f"{label} is missing or outside its canonical root"
        ) from exc
    if canonical_required:
        value = _json_bytes(raw, label)
        if raw != _canonical_bytes(value) + b"\n":
            raise DojoReplayLifecycleError(f"{label} bytes are not canonical")
    return raw


@contextmanager
def _shared_file_lock(path: Path) -> Iterator[Any]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise DojoReplayLifecycleError("candidate ledger is unavailable") from exc
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        yield handle
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _ledger_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            rows.append(_json_bytes(line.encode("utf-8"), "candidate ledger row"))
    if not rows:
        raise DojoReplayLifecycleError("candidate ledger is empty")
    return rows


def _write_exclusive_fsynced(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise DojoReplayLifecycleError(
            f"exclusive lifecycle artifact already exists: {path.name}"
        ) from exc
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short exclusive lifecycle write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _canonical_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

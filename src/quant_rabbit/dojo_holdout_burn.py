"""First-write burn registry for historical DOJO holdout windows.

The registry records which historical outcomes were already used, reserves a
locally unopened window relative to one declared selection lineage, burns that
reservation immediately before outcome reveal, and binds the eventual result.

This is deliberately a local discipline mechanism, not proof that nobody has
seen the data.  Every event and derived status therefore keeps proof, promotion,
live, and broker-mutation authority disabled.  An external monotonic witness
and independent custody would be needed to make a stronger claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


EVENT_CONTRACT: Final = "QR_DOJO_HISTORICAL_HOLDOUT_BURN_EVENT_V1"
STATUS_CONTRACT: Final = "QR_DOJO_HISTORICAL_HOLDOUT_BURN_STATUS_V1"
SCHEMA_VERSION: Final = 1
MAX_EVENTS: Final = 100_000
MAX_EVENT_BYTES: Final = 256 * 1024
MAX_TOTAL_BYTES: Final = 128 * 1024 * 1024

_EVENT_TYPES: Final = frozenset(
    {"GENESIS", "LEGACY_BURN", "RESERVATION", "BURN_INTENT", "RESULT_BOUND"}
)
_HEX64: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}\Z")
_CODE: Final = re.compile(r"[A-Z][A-Z0-9_.:-]{0,127}\Z")
_INSTRUMENT: Final = re.compile(r"[A-Z0-9]{2,12}_[A-Z0-9]{2,12}\Z")
_EVENT_NAME: Final = re.compile(r"[0-9]{6}\.json\Z")

_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "registry_id",
        "sequence",
        "event_type",
        "event_at_utc",
        "previous_event_sha256",
        "body",
        "external_witness_status",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "broker_mutation_allowed",
        "event_sha256",
    }
)
_GENESIS_KEYS: Final = frozenset({"created_by", "monotonicity_scope"})
_LEGACY_BURN_KEYS: Final = frozenset(
    {
        "burn_id",
        "task_kind",
        "selection_lineage_id",
        "instruments",
        "target_outcome_domain",
        "window_from_utc",
        "window_to_utc",
        "granularity",
        "input_modalities",
        "legacy_source",
        "legacy_evidence_sha256",
        "reason",
    }
)
_RESERVATION_KEYS: Final = frozenset(
    {
        "reservation_id",
        "candidate_id",
        "task_kind",
        "family_id",
        "selection_lineage_id",
        "unused_relative_to",
        "instruments",
        "target_outcome_domain",
        "window_from_utc",
        "window_to_utc",
        "granularity",
        "input_modalities",
        "prompt_set_sha256",
        "model_policy_sha256",
        "scorer_sha256",
        "code_sha256",
        "corpus_manifest_sha256",
        "custody_policy_sha256",
    }
)
_BURN_INTENT_KEYS: Final = frozenset(
    {
        "reservation_id",
        "reveal_material_sha256",
        "revealed_by",
        "permanence_acknowledged",
    }
)
_RESULT_BOUND_KEYS: Final = frozenset(
    {"reservation_id", "result_sha256", "result_contract", "bound_by"}
)


class HoldoutBurnError(ValueError):
    """The registry, requested transition, or supplied binding is invalid."""


@dataclass(frozen=True)
class HoldoutRegistrySnapshot:
    """Verified immutable view of the local event directory."""

    registry_id: str
    event_count: int
    latest_sequence: int
    latest_event_sha256: str
    latest_event_at_utc: str
    events: tuple[Mapping[str, Any], ...]
    legacy_burns: tuple[Mapping[str, Any], ...]
    reservations: Mapping[str, Mapping[str, Any]]
    burn_intents: Mapping[str, Mapping[str, Any]]
    results: Mapping[str, Mapping[str, Any]]
    external_witness_status: str = "ABSENT"
    proof_eligible: bool = False
    promotion_eligible: bool = False
    live_permission: bool = False
    broker_mutation_allowed: bool = False


def initialize_registry(
    events_dir: Path,
    *,
    registry_id: str,
    created_by: str,
    event_at_utc: datetime | str,
) -> HoldoutRegistrySnapshot:
    """Create exactly one genesis event in a new or empty real directory."""

    registry = _identifier(registry_id, "registry_id")
    body = {
        "created_by": _text(created_by, "created_by"),
        "monotonicity_scope": "LOCAL_FIRST_WRITE_ONLY",
    }
    directory_fd = _open_events_directory(Path(events_dir), create=True)
    try:
        names = _event_names(directory_fd)
        if names:
            raise HoldoutBurnError("holdout registry is already initialized")
        event = _new_event(
            registry_id=registry,
            sequence=0,
            event_type="GENESIS",
            event_at_utc=event_at_utc,
            previous_event_sha256=None,
            body=body,
        )
        _write_event_exclusive(directory_fd, "000000.json", event)
    finally:
        os.close(directory_fd)
    return verify_registry(Path(events_dir))


def append_legacy_burn(
    events_dir: Path,
    *,
    burn: Mapping[str, Any],
    event_at_utc: datetime | str,
) -> HoldoutRegistrySnapshot:
    """Record a known-used interval; later discovery invalidates reservations."""

    body = _normalize_legacy_burn(burn)
    snapshot = verify_registry(Path(events_dir))
    if any(row["burn_id"] == body["burn_id"] for row in snapshot.legacy_burns):
        raise HoldoutBurnError("legacy burn id is already recorded")
    # Distinct legacy audits may legitimately describe nested M1/M5/S5 windows.
    # They all burn reservations; only replacement of the same burn id is banned.
    return _append(Path(events_dir), snapshot, "LEGACY_BURN", body, event_at_utc)


def reserve_holdout(
    events_dir: Path,
    *,
    reservation: Mapping[str, Any],
    event_at_utc: datetime | str,
) -> HoldoutRegistrySnapshot:
    """Reserve one historical window relative to the declared selection lineage."""

    body = _normalize_reservation(reservation)
    snapshot = verify_registry(Path(events_dir))
    if body["reservation_id"] in snapshot.reservations:
        raise HoldoutBurnError("reservation id is already recorded")
    legacy_conflict = _first_overlap(body, snapshot.legacy_burns)
    if legacy_conflict is not None:
        raise HoldoutBurnError(
            "reservation overlaps a legacy burn: " f"{legacy_conflict['burn_id']}"
        )
    reservation_conflict = _first_overlap(body, snapshot.reservations.values())
    if reservation_conflict is not None:
        raise HoldoutBurnError(
            "reservation conflicts with an existing outcome window regardless of "
            "candidate, corpus, prompt, or granularity relabeling: "
            f"{reservation_conflict['reservation_id']}"
        )
    return _append(Path(events_dir), snapshot, "RESERVATION", body, event_at_utc)


def append_burn_intent(
    events_dir: Path,
    *,
    intent: Mapping[str, Any],
    event_at_utc: datetime | str,
) -> HoldoutRegistrySnapshot:
    """Permanently burn a reservation immediately before outcome reveal."""

    body = _normalize_burn_intent(intent)
    snapshot = verify_registry(Path(events_dir))
    reservation_id = body["reservation_id"]
    reservation = snapshot.reservations.get(reservation_id)
    if reservation is None:
        raise HoldoutBurnError("burn intent references an unknown reservation")
    if reservation_id in snapshot.burn_intents:
        raise HoldoutBurnError("reservation is already permanently burned")
    if _first_overlap(reservation, snapshot.legacy_burns) is not None:
        raise HoldoutBurnError("legacy overlap invalidated reservation before reveal")
    return _append(Path(events_dir), snapshot, "BURN_INTENT", body, event_at_utc)


def bind_result(
    events_dir: Path,
    *,
    result: Mapping[str, Any],
    event_at_utc: datetime | str,
) -> HoldoutRegistrySnapshot:
    """Bind result bytes to a burned reservation without changing eligibility."""

    body = _normalize_result(result)
    snapshot = verify_registry(Path(events_dir))
    reservation_id = body["reservation_id"]
    if reservation_id not in snapshot.reservations:
        raise HoldoutBurnError("result references an unknown reservation")
    if reservation_id not in snapshot.burn_intents:
        raise HoldoutBurnError("result cannot be bound before permanent burn intent")
    if reservation_id in snapshot.results:
        raise HoldoutBurnError("reservation result is already bound")
    return _append(Path(events_dir), snapshot, "RESULT_BOUND", body, event_at_utc)


def verify_registry(events_dir: Path) -> HoldoutRegistrySnapshot:
    """Verify filenames, strict canonical JSON, hash chain, and all transitions."""

    directory_fd = _open_events_directory(Path(events_dir), create=False)
    try:
        names = _event_names(directory_fd)
        if not names:
            raise HoldoutBurnError("holdout registry has no genesis event")
        if len(names) > MAX_EVENTS:
            raise HoldoutBurnError("holdout registry event count exceeds limit")
        expected = [f"{sequence:06d}.json" for sequence in range(len(names))]
        if names != expected:
            raise HoldoutBurnError(
                "holdout registry event sequence has a gap or extra file"
            )
        events: list[dict[str, Any]] = []
        total_bytes = 0
        for name in names:
            event, size = _read_event(directory_fd, name)
            total_bytes += size
            if total_bytes > MAX_TOTAL_BYTES:
                raise HoldoutBurnError("holdout registry total byte limit exceeded")
            events.append(event)
    finally:
        os.close(directory_fd)
    return _replay(events)


def status_artifact(events_dir: Path) -> dict[str, Any]:
    """Return conservative derived state; it never grants proof or live authority."""

    snapshot = verify_registry(Path(events_dir))
    reservation_rows: list[dict[str, Any]] = []
    for reservation_id, reservation in sorted(snapshot.reservations.items()):
        invalidators = [
            burn["burn_id"]
            for burn in snapshot.legacy_burns
            if _overlaps(reservation, burn)
        ]
        burned = reservation_id in snapshot.burn_intents
        result_bound = reservation_id in snapshot.results
        if invalidators:
            state = "INVALIDATED_BY_LEGACY_BURN"
        elif result_bound:
            state = "RESULT_BOUND"
        elif burned:
            state = "BURNED_PENDING_RESULT"
        else:
            state = "LOCALLY_UNOPENED_RELATIVE_TO_DECLARED_SELECTION_LINEAGE"
        reservation_rows.append(
            {
                **_snapshot(reservation),
                "state": state,
                "permanently_burned": burned,
                "result_bound": result_bound,
                "invalidating_legacy_burn_ids": invalidators,
                "burn_intent_event_sha256": (
                    snapshot.burn_intents[reservation_id]["event_sha256"]
                    if burned
                    else None
                ),
                "result_event_sha256": (
                    snapshot.results[reservation_id]["event_sha256"]
                    if result_bound
                    else None
                ),
                "proof_eligible": False,
                "promotion_eligible": False,
                "live_permission": False,
            }
        )
    return {
        "contract": STATUS_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "registry_id": snapshot.registry_id,
        "event_count": snapshot.event_count,
        "latest_sequence": snapshot.latest_sequence,
        "latest_event_sha256": snapshot.latest_event_sha256,
        "legacy_burn_count": len(snapshot.legacy_burns),
        "legacy_burns": [_snapshot(row) for row in snapshot.legacy_burns],
        "reservation_count": len(snapshot.reservations),
        "permanently_burned_reservation_count": len(snapshot.burn_intents),
        "result_bound_count": len(snapshot.results),
        "invalidated_reservation_count": sum(
            bool(row["invalidating_legacy_burn_ids"]) for row in reservation_rows
        ),
        "reservations": reservation_rows,
        "qualification_scope": (
            "LOCALLY_UNOPENED_RELATIVE_TO_DECLARED_SELECTION_LINEAGE"
        ),
        "external_witness_status": "ABSENT",
        "limitations": [
            "LOCAL_OWNER_CAN_DELETE_OR_RECREATE_ENTIRE_LEDGER",
            "LOCAL_REGISTRY_CANNOT_PROVE_NOBODY_SAW_HISTORICAL_OUTCOMES",
            "EXTERNAL_SIGNED_MONOTONIC_WITNESS_AND_INDEPENDENT_CUSTODY_ABSENT",
        ],
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }


def _append(
    events_dir: Path,
    snapshot: HoldoutRegistrySnapshot,
    event_type: str,
    body: Mapping[str, Any],
    event_at_utc: datetime | str,
) -> HoldoutRegistrySnapshot:
    if snapshot.event_count >= MAX_EVENTS:
        raise HoldoutBurnError("holdout registry capacity is exhausted")
    event_time = _utc_text(event_at_utc, "event_at_utc")
    if _parse_utc(event_time, "event_at_utc") < _parse_utc(
        snapshot.latest_event_at_utc, "latest_event_at_utc"
    ):
        raise HoldoutBurnError("event_at_utc cannot move backward")
    sequence = snapshot.latest_sequence + 1
    event = _new_event(
        registry_id=snapshot.registry_id,
        sequence=sequence,
        event_type=event_type,
        event_at_utc=event_time,
        previous_event_sha256=snapshot.latest_event_sha256,
        body=body,
    )
    directory_fd = _open_events_directory(events_dir, create=False)
    try:
        _write_event_exclusive(directory_fd, f"{sequence:06d}.json", event)
    finally:
        os.close(directory_fd)
    return verify_registry(events_dir)


def _new_event(
    *,
    registry_id: str,
    sequence: int,
    event_type: str,
    event_at_utc: datetime | str,
    previous_event_sha256: str | None,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    if event_type not in _EVENT_TYPES:
        raise HoldoutBurnError("unsupported holdout registry event type")
    value = {
        "contract": EVENT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "registry_id": _identifier(registry_id, "registry_id"),
        "sequence": sequence,
        "event_type": event_type,
        "event_at_utc": _utc_text(event_at_utc, "event_at_utc"),
        "previous_event_sha256": (
            None
            if previous_event_sha256 is None
            else _sha(previous_event_sha256, "previous_event_sha256")
        ),
        "body": _snapshot(body),
        "external_witness_status": "ABSENT",
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    value["event_sha256"] = _canonical_sha(value)
    return value


def _replay(events: Sequence[Mapping[str, Any]]) -> HoldoutRegistrySnapshot:
    previous_sha: str | None = None
    previous_time: datetime | None = None
    registry_id: str | None = None
    legacy_burns: list[dict[str, Any]] = []
    reservations: dict[str, dict[str, Any]] = {}
    burn_intents: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    verified_events: list[dict[str, Any]] = []

    for sequence, raw in enumerate(events):
        event = _mapping(raw, "holdout event")
        if set(event) != _TOP_LEVEL_KEYS:
            raise HoldoutBurnError("holdout event top-level shape drifted")
        if (
            event["contract"] != EVENT_CONTRACT
            or isinstance(event["schema_version"], bool)
            or event["schema_version"] != SCHEMA_VERSION
        ):
            raise HoldoutBurnError("holdout event contract or version drifted")
        if isinstance(event["sequence"], bool) or event["sequence"] != sequence:
            raise HoldoutBurnError("holdout event sequence drifted")
        if event["previous_event_sha256"] != previous_sha:
            raise HoldoutBurnError("holdout event hash chain drifted")
        event_sha = _sha(event["event_sha256"], "event_sha256")
        body_without_sha = {
            key: value for key, value in event.items() if key != "event_sha256"
        }
        if _canonical_sha(body_without_sha) != event_sha:
            raise HoldoutBurnError("holdout event_sha256 mismatch")
        if event["external_witness_status"] != "ABSENT" or any(
            event[field] is not False
            for field in (
                "proof_eligible",
                "promotion_eligible",
                "live_permission",
                "broker_mutation_allowed",
            )
        ):
            raise HoldoutBurnError("holdout event exceeds local diagnostic authority")
        event_time_text = _utc_text(event["event_at_utc"], "event_at_utc")
        event_time = _parse_utc(event_time_text, "event_at_utc")
        if previous_time is not None and event_time < previous_time:
            raise HoldoutBurnError("holdout event time moved backward")
        current_registry = _identifier(event["registry_id"], "registry_id")
        if sequence == 0:
            if event["event_type"] != "GENESIS" or previous_sha is not None:
                raise HoldoutBurnError("first holdout event must be genesis")
            registry_id = current_registry
            _normalize_genesis(event["body"])
        else:
            if current_registry != registry_id or event["event_type"] == "GENESIS":
                raise HoldoutBurnError("holdout registry identity or genesis drifted")
            event_type = event["event_type"]
            if event_type == "LEGACY_BURN":
                body = _normalize_legacy_burn(event["body"])
                if any(row["burn_id"] == body["burn_id"] for row in legacy_burns):
                    raise HoldoutBurnError("legacy burn id is repeated")
                legacy_burns.append(body)
            elif event_type == "RESERVATION":
                body = _normalize_reservation(event["body"])
                reservation_id = body["reservation_id"]
                if reservation_id in reservations:
                    raise HoldoutBurnError("reservation id is repeated")
                if _first_overlap(body, legacy_burns) is not None:
                    raise HoldoutBurnError(
                        "reservation overlaps an earlier legacy burn"
                    )
                if _first_overlap(body, reservations.values()) is not None:
                    raise HoldoutBurnError(
                        "overlapping holdout reservation is repeated"
                    )
                reservations[reservation_id] = body
            elif event_type == "BURN_INTENT":
                body = _normalize_burn_intent(event["body"])
                reservation_id = body["reservation_id"]
                reservation = reservations.get(reservation_id)
                if reservation is None or reservation_id in burn_intents:
                    raise HoldoutBurnError("burn transition is missing or repeated")
                if _first_overlap(reservation, legacy_burns) is not None:
                    raise HoldoutBurnError("invalidated reservation cannot be revealed")
                burn_intents[reservation_id] = {**body, "event_sha256": event_sha}
            elif event_type == "RESULT_BOUND":
                body = _normalize_result(event["body"])
                reservation_id = body["reservation_id"]
                if reservation_id not in reservations:
                    raise HoldoutBurnError("result references an unknown reservation")
                if reservation_id not in burn_intents or reservation_id in results:
                    raise HoldoutBurnError("result transition is missing or repeated")
                results[reservation_id] = {**body, "event_sha256": event_sha}
            else:
                raise HoldoutBurnError("unsupported holdout event type")
        previous_sha = event_sha
        previous_time = event_time
        verified_events.append(event)

    if registry_id is None or previous_sha is None or previous_time is None:
        raise HoldoutBurnError("holdout registry lacks verified genesis")
    return HoldoutRegistrySnapshot(
        registry_id=registry_id,
        event_count=len(verified_events),
        latest_sequence=len(verified_events) - 1,
        latest_event_sha256=previous_sha,
        latest_event_at_utc=_iso(previous_time),
        events=tuple(verified_events),
        legacy_burns=tuple(legacy_burns),
        reservations=reservations,
        burn_intents=burn_intents,
        results=results,
    )


def _normalize_genesis(value: Any) -> dict[str, Any]:
    body = _exact_mapping(value, _GENESIS_KEYS, "genesis")
    normalized = {
        "created_by": _text(body["created_by"], "created_by"),
        "monotonicity_scope": body["monotonicity_scope"],
    }
    if normalized["monotonicity_scope"] != "LOCAL_FIRST_WRITE_ONLY":
        raise HoldoutBurnError("genesis monotonicity scope drifted")
    if body != normalized:
        raise HoldoutBurnError("genesis body is not canonical")
    return normalized


def _normalize_legacy_burn(value: Any) -> dict[str, Any]:
    body = _exact_mapping(value, _LEGACY_BURN_KEYS, "legacy burn")
    normalized = {
        "burn_id": _identifier(body["burn_id"], "burn_id"),
        "task_kind": _code(body["task_kind"], "task_kind"),
        "selection_lineage_id": _identifier(
            body["selection_lineage_id"], "selection_lineage_id"
        ),
        "instruments": _string_set(body["instruments"], "instruments", instrument=True),
        "target_outcome_domain": _code(
            body["target_outcome_domain"], "target_outcome_domain"
        ),
        "window_from_utc": _utc_text(body["window_from_utc"], "window_from_utc"),
        "window_to_utc": _utc_text(body["window_to_utc"], "window_to_utc"),
        "granularity": _code(body["granularity"], "granularity"),
        "input_modalities": _string_set(
            body["input_modalities"], "input_modalities", instrument=False
        ),
        "legacy_source": _text(body["legacy_source"], "legacy_source"),
        "legacy_evidence_sha256": _sha(
            body["legacy_evidence_sha256"], "legacy_evidence_sha256"
        ),
        "reason": _code(body["reason"], "reason"),
    }
    _positive_window(normalized)
    return normalized


def _normalize_reservation(value: Any) -> dict[str, Any]:
    body = _exact_mapping(value, _RESERVATION_KEYS, "reservation")
    normalized = {
        "reservation_id": _identifier(body["reservation_id"], "reservation_id"),
        "candidate_id": _identifier(body["candidate_id"], "candidate_id"),
        "task_kind": _code(body["task_kind"], "task_kind"),
        "family_id": _identifier(body["family_id"], "family_id"),
        "selection_lineage_id": _identifier(
            body["selection_lineage_id"], "selection_lineage_id"
        ),
        "unused_relative_to": _text(body["unused_relative_to"], "unused_relative_to"),
        "instruments": _string_set(body["instruments"], "instruments", instrument=True),
        "target_outcome_domain": _code(
            body["target_outcome_domain"], "target_outcome_domain"
        ),
        "window_from_utc": _utc_text(body["window_from_utc"], "window_from_utc"),
        "window_to_utc": _utc_text(body["window_to_utc"], "window_to_utc"),
        "granularity": _code(body["granularity"], "granularity"),
        "input_modalities": _string_set(
            body["input_modalities"], "input_modalities", instrument=False
        ),
        "prompt_set_sha256": _sha(body["prompt_set_sha256"], "prompt_set_sha256"),
        "model_policy_sha256": _sha(body["model_policy_sha256"], "model_policy_sha256"),
        "scorer_sha256": _sha(body["scorer_sha256"], "scorer_sha256"),
        "code_sha256": _sha(body["code_sha256"], "code_sha256"),
        "corpus_manifest_sha256": _sha(
            body["corpus_manifest_sha256"], "corpus_manifest_sha256"
        ),
        "custody_policy_sha256": _sha(
            body["custody_policy_sha256"], "custody_policy_sha256"
        ),
    }
    _positive_window(normalized)
    return normalized


def _normalize_burn_intent(value: Any) -> dict[str, Any]:
    body = _exact_mapping(value, _BURN_INTENT_KEYS, "burn intent")
    normalized = {
        "reservation_id": _identifier(body["reservation_id"], "reservation_id"),
        "reveal_material_sha256": _sha(
            body["reveal_material_sha256"], "reveal_material_sha256"
        ),
        "revealed_by": _text(body["revealed_by"], "revealed_by"),
        "permanence_acknowledged": body["permanence_acknowledged"],
    }
    if normalized["permanence_acknowledged"] is not True:
        raise HoldoutBurnError("burn intent must acknowledge permanent consumption")
    return normalized


def _normalize_result(value: Any) -> dict[str, Any]:
    body = _exact_mapping(value, _RESULT_BOUND_KEYS, "result")
    return {
        "reservation_id": _identifier(body["reservation_id"], "reservation_id"),
        "result_sha256": _sha(body["result_sha256"], "result_sha256"),
        "result_contract": _identifier(body["result_contract"], "result_contract"),
        "bound_by": _text(body["bound_by"], "bound_by"),
    }


def _first_overlap(
    candidate: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    for row in rows:
        if _overlaps(candidate, row):
            return row
    return None


def _overlaps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    # Any scored outcome derived from an overlapping candle path consumes that
    # path.  Outcome-domain names are labels, not independent custody scopes:
    # ENTRY and EXIT answers for the same instrument/window both reveal future
    # prices and therefore must not be reserved separately.
    if not set(left["instruments"]).intersection(right["instruments"]):
        return False
    left_from = _parse_utc(left["window_from_utc"], "window_from_utc")
    left_to = _parse_utc(left["window_to_utc"], "window_to_utc")
    right_from = _parse_utc(right["window_from_utc"], "window_from_utc")
    right_to = _parse_utc(right["window_to_utc"], "window_to_utc")
    return left_from < right_to and right_from < left_to


def _positive_window(body: Mapping[str, Any]) -> None:
    start = _parse_utc(body["window_from_utc"], "window_from_utc")
    end = _parse_utc(body["window_to_utc"], "window_to_utc")
    if not start < end:
        raise HoldoutBurnError("holdout window must be positive and half-open")


def _open_events_directory(path: Path, *, create: bool) -> int:
    candidate = path.absolute()
    if candidate == candidate.parent:
        raise HoldoutBurnError("events directory cannot be a filesystem root")
    try:
        state = candidate.lstat()
    except FileNotFoundError:
        if not create:
            raise HoldoutBurnError("holdout events directory is absent") from None
        parent_fd = _open_real_directory(candidate.parent, "events parent")
        try:
            try:
                os.mkdir(candidate.name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            os.fsync(parent_fd)
        except OSError as exc:
            raise HoldoutBurnError(
                f"cannot create holdout events directory: {exc}"
            ) from exc
        finally:
            os.close(parent_fd)
        state = candidate.lstat()
    except OSError as exc:
        raise HoldoutBurnError(f"cannot stat holdout events directory: {exc}") from exc
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise HoldoutBurnError("holdout events directory must be a real directory")
    return _open_real_directory(candidate, "events directory")


def _open_real_directory(path: Path, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise HoldoutBurnError(f"cannot open {label}: {exc}") from exc


def _event_names(directory_fd: int) -> list[str]:
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as exc:
        raise HoldoutBurnError(f"cannot list holdout events directory: {exc}") from exc
    if any(_EVENT_NAME.fullmatch(name) is None for name in names):
        raise HoldoutBurnError("holdout events directory contains an unexpected file")
    return names


def _read_event(directory_fd: int, name: str) -> tuple[dict[str, Any], int]:
    try:
        state = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise HoldoutBurnError(f"cannot stat holdout event {name}: {exc}") from exc
    if not stat.S_ISREG(state.st_mode) or state.st_size <= 0:
        raise HoldoutBurnError("holdout event must be a nonempty regular file")
    if state.st_size > MAX_EVENT_BYTES:
        raise HoldoutBurnError("holdout event byte limit exceeded")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise HoldoutBurnError("holdout event must remain a regular file")
            raw = handle.read(MAX_EVENT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise HoldoutBurnError(f"cannot read holdout event {name}: {exc}") from exc
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_dev != state.st_dev
        or before.st_ino != state.st_ino
        or len(raw) != before.st_size
    ):
        raise HoldoutBurnError("holdout event changed while being read")
    value = _strict_json_bytes(raw)
    if raw != _canonical_bytes(value) + b"\n":
        raise HoldoutBurnError("holdout event bytes are not canonical JSON")
    return value, len(raw)


def _write_event_exclusive(
    directory_fd: int, name: str, event: Mapping[str, Any]
) -> None:
    payload = _canonical_bytes(event) + b"\n"
    if len(payload) > MAX_EVENT_BYTES:
        raise HoldoutBurnError("holdout event byte limit exceeded")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
        created = True
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(directory_fd)
    except FileExistsError as exc:
        raise HoldoutBurnError(
            "holdout event slot already exists; retry from head"
        ) from exc
    except BaseException:
        if created:
            try:
                os.unlink(name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except OSError:
                pass
        raise


def _strict_json_bytes(raw: bytes) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise HoldoutBurnError(f"non-finite JSON constant is forbidden: {token}")

    def reject_float(token: str) -> None:
        raise HoldoutBurnError(f"JSON floating point is forbidden: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HoldoutBurnError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
            parse_float=reject_float,
        )
    except HoldoutBurnError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HoldoutBurnError("strict holdout event JSON parse failed") from exc
    return _mapping(value, "holdout event")


def _exact_mapping(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    body = _mapping(value, label)
    if set(body) != keys:
        missing = sorted(keys - set(body))
        extra = sorted(set(body) - keys)
        raise HoldoutBurnError(
            f"{label} bindings are incomplete: missing={missing}, extra={extra}"
        )
    return body


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HoldoutBurnError(f"{label} must be an object")
    return _snapshot(value)


def _snapshot(value: Any) -> Any:
    try:
        return json.loads(_canonical_bytes(value))
    except (TypeError, ValueError) as exc:
        raise HoldoutBurnError("holdout value is not strict JSON") from exc


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise HoldoutBurnError(f"{label} must be a lowercase sha256")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise HoldoutBurnError(f"{label} is invalid")
    return value


def _code(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise HoldoutBurnError(f"{label} is invalid")
    normalized = value.strip().upper()
    if _CODE.fullmatch(normalized) is None:
        raise HoldoutBurnError(f"{label} is invalid")
    return normalized


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise HoldoutBurnError(f"{label} must be nonempty text")
    text = value.strip()
    if not text or len(text) > 2_048 or any(ord(character) < 32 for character in text):
        raise HoldoutBurnError(f"{label} must be bounded nonempty text")
    return text


def _string_set(value: Any, label: str, *, instrument: bool) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HoldoutBurnError(f"{label} must be a nonempty sequence")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise HoldoutBurnError(f"{label} contains a non-string value")
        candidate = item.strip().upper()
        if instrument:
            if _INSTRUMENT.fullmatch(candidate) is None:
                raise HoldoutBurnError(f"{label} contains an invalid instrument")
        elif _CODE.fullmatch(candidate) is None:
            raise HoldoutBurnError(f"{label} contains an invalid code")
        normalized.append(candidate)
    result = sorted(set(normalized))
    if not result or len(result) != len(normalized) or len(result) > 128:
        raise HoldoutBurnError(f"{label} is empty, duplicated, or too large")
    return result


def _utc_text(value: datetime | str | Any, label: str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise HoldoutBurnError(f"{label} must be timezone-aware")
        return _iso(value.astimezone(timezone.utc))
    if not isinstance(value, str):
        raise HoldoutBurnError(f"{label} must be exact UTC text")
    parsed = _parse_utc(value, label)
    normalized = _iso(parsed)
    if value != normalized:
        raise HoldoutBurnError(f"{label} must use canonical UTC Z notation")
    return normalized


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise HoldoutBurnError(f"{label} must be exact UTC Z text")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise HoldoutBurnError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise HoldoutBurnError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "EVENT_CONTRACT",
    "HoldoutBurnError",
    "HoldoutRegistrySnapshot",
    "STATUS_CONTRACT",
    "append_burn_intent",
    "append_legacy_burn",
    "bind_result",
    "initialize_registry",
    "reserve_holdout",
    "status_artifact",
    "verify_registry",
]

"""Parallel prospective-test registry (weakness ledger W20).

Serial future tests cap the rate at which PROVEN evidence can accumulate.
This registry lets several frozen candidates each pre-declare an independent
future window, while keeping the multiple-testing denominator honest: every
registration is append-only, hash-chained, made strictly before its window
opens, and counted per family for later White-Reality-Check style
correction.  Registration proves nothing; it only fixes, in advance, what
may later be evaluated unchanged.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

CONTRACT = "QR_PROSPECTIVE_REGISTRY_V1"
_SHA_LENGTH = 64


class ProspectiveRegistryError(ValueError):
    """Raised when a registration or the chain is invalid."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _aware(value: Any, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ProspectiveRegistryError(f"{label} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _sha(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != _SHA_LENGTH or any(c not in "0123456789abcdef" for c in text):
        raise ProspectiveRegistryError(f"{label} must be a lowercase sha256")
    return text


def empty_registry() -> dict[str, Any]:
    body = {"contract": CONTRACT, "schema_version": 1, "entries": []}
    return {**body, "registry_sha256": _canonical_sha(body)}


def _validate_registry(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(registry, Mapping):
        raise ProspectiveRegistryError("registry must be an object")
    body = {k: v for k, v in registry.items() if k != "registry_sha256"}
    if registry.get("registry_sha256") != _canonical_sha(body):
        raise ProspectiveRegistryError("registry digest is invalid")
    if registry.get("contract") != CONTRACT:
        raise ProspectiveRegistryError("registry contract is invalid")
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ProspectiveRegistryError("registry entries must be a list")
    previous = ""
    for index, entry in enumerate(entries):
        chained = {k: v for k, v in entry.items() if k != "entry_sha256"}
        if entry.get("entry_sha256") != _canonical_sha(chained):
            raise ProspectiveRegistryError(f"entry {index} digest is invalid")
        if entry.get("previous_entry_sha256") != (previous or None):
            raise ProspectiveRegistryError(f"entry {index} breaks the chain")
        previous = entry["entry_sha256"]
    return list(entries)


def register_candidate(
    registry: Mapping[str, Any],
    *,
    candidate_id: str,
    family_id: str,
    lock_sha256: str,
    window_from_utc: datetime,
    window_to_utc: datetime,
    registered_at_utc: datetime,
) -> dict[str, Any]:
    """Append one pre-declared future window; returns the new sealed registry."""

    entries = _validate_registry(registry)
    candidate = str(candidate_id).strip()
    family = str(family_id).strip().upper()
    if not candidate or not family:
        raise ProspectiveRegistryError("candidate and family identity are required")
    registered = _aware(registered_at_utc, "registered_at_utc")
    window_from = _aware(window_from_utc, "window_from_utc")
    window_to = _aware(window_to_utc, "window_to_utc")
    if not registered < window_from < window_to:
        raise ProspectiveRegistryError(
            "window must open strictly after registration and be positive"
        )
    for entry in entries:
        if entry["candidate_id"] == candidate:
            raise ProspectiveRegistryError("candidate is already registered")
    previous_sha = entries[-1]["entry_sha256"] if entries else None
    entry_body = {
        "candidate_id": candidate,
        "family_id": family,
        "lock_sha256": _sha(lock_sha256, "lock_sha256"),
        "window_from_utc": window_from.isoformat(),
        "window_to_utc": window_to.isoformat(),
        "registered_at_utc": registered.isoformat(),
        "previous_entry_sha256": previous_sha,
        "evaluation_allowed_from_utc": window_to.isoformat(),
        "reselection_allowed": False,
    }
    sealed_entry = {**entry_body, "entry_sha256": _canonical_sha(entry_body)}
    body = {
        "contract": CONTRACT,
        "schema_version": 1,
        "entries": [*entries, sealed_entry],
    }
    return {**body, "registry_sha256": _canonical_sha(body)}


def evaluation_admissible(
    registry: Mapping[str, Any],
    *,
    candidate_id: str,
    now_utc: datetime,
) -> dict[str, Any]:
    """Say whether one candidate's window may be evaluated now, fail-closed."""

    entries = _validate_registry(registry)
    now = _aware(now_utc, "now_utc")
    for entry in entries:
        if entry["candidate_id"] == str(candidate_id).strip():
            matured = now >= datetime.fromisoformat(entry["window_to_utc"])
            return {
                "candidate_id": entry["candidate_id"],
                "admissible": matured,
                "reason": "WINDOW_MATURED"
                if matured
                else "WINDOW_NOT_YET_MATURED",
                "lock_sha256": entry["lock_sha256"],
                "grants_positive_result": False,
            }
    return {
        "candidate_id": str(candidate_id).strip(),
        "admissible": False,
        "reason": "CANDIDATE_NOT_REGISTERED",
        "lock_sha256": None,
        "grants_positive_result": False,
    }


def family_denominators(registry: Mapping[str, Any]) -> dict[str, int]:
    """Per-family registration counts: the honest multiple-testing input."""

    counts: dict[str, int] = {}
    for entry in _validate_registry(registry):
        counts[entry["family_id"]] = counts.get(entry["family_id"], 0) + 1
    return counts

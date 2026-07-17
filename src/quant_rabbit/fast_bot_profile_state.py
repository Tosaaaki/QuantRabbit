"""Content-addressed activation and exact-lane quarantine state.

Activation is a research-routing receipt only.  It binds one immutable profile
catalog, one pair/horizon scope, and the catalog's pre-sealed fallback path.
Quarantine is terminal for the exact profile revision represented by that
catalog.  Neither artifact grants AI, order, broker, or live authority.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from quant_rabbit.fast_bot_profiles import (
    AI_ORDER_AUTHORITY,
    ORDER_AUTHORITY,
    PRIMARY_ELIGIBLE,
    FastBotProfileCatalog,
    LaneKey,
    ProfileContractError,
    _fallback_chain_profile_ids_validated,
    canonical_sha256,
    validate_fast_bot_profile_catalog,
)


ACTIVATION_RECEIPT_CONTRACT = "QR_FAST_BOT_PROFILE_ACTIVATION_RECEIPT_V1"
QUARANTINE_RECORD_CONTRACT = "QR_FAST_BOT_PROFILE_QUARANTINE_V1"

# One bounded state artifact may cite at most this many immutable evidence
# digests.  This protects artifact size and reviewability; it is not a market
# threshold, and a new schema should replace it if provenance needs expand.
MAX_EVIDENCE_DIGESTS = 32
# Quarantine reasons are protocol labels, not free-form narratives.  This
# audit-surface bound is unrelated to market behavior; longer prose belongs in
# content-addressed evidence rather than this schema revision.
MAX_REASON_CODE_CHARS = 96
# Python's canonical UTC rendering is 20 characters at whole-second precision
# and 27 with microseconds.  The upper bound rejects oversized parser input;
# timestamps needing more precision require a new artifact contract.
MAX_CANONICAL_UTC_CHARS = 27

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REASON_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")
_AUTHORITY_BODY: dict[str, object] = {
    "ai_order_authority": AI_ORDER_AUTHORITY,
    "order_authority": ORDER_AUTHORITY,
    "live_permission": False,
    "broker_mutation_allowed": False,
    "shadow_only": True,
}
_ACTIVATION_KEYS = frozenset(
    {
        "contract",
        "catalog_sha256",
        "pair",
        "profile_id",
        "profile_sha256",
        "horizon_lane",
        "fallback_chain_profile_ids",
        "fallback_chain_sha256",
        "evidence_sha256s",
        "activated_at_utc",
        *_AUTHORITY_BODY,
        "activation_receipt_sha256",
    }
)
_QUARANTINE_KEYS = frozenset(
    {
        "contract",
        "catalog_sha256",
        "pair",
        "profile_id",
        "profile_sha256",
        "horizon_lane",
        "reason_code",
        "evidence_sha256s",
        "quarantined_at_utc",
        "terminal_for_profile_revision",
        *_AUTHORITY_BODY,
        "quarantine_record_sha256",
    }
)


def build_activation_receipt(
    *,
    catalog: FastBotProfileCatalog | Mapping[str, Any],
    lane: LaneKey,
    evidence_sha256s: Sequence[str],
    activated_at_utc: datetime,
) -> dict[str, Any]:
    """Seal a research-primary request and its registry-owned fallback chain."""

    catalog = validate_fast_bot_profile_catalog(catalog)
    if lane.__class__ is not LaneKey:
        raise ProfileContractError("activation lane must be an exact LaneKey")
    profile = catalog.profile(lane.profile_id)
    if not profile.supports(lane.pair, lane.horizon_lane):
        raise ProfileContractError("activation profile does not support lane")
    if profile.activation_eligibility != PRIMARY_ELIGIBLE:
        raise ProfileContractError("profile is not research-primary eligible")
    evidence = _canonical_evidence(evidence_sha256s)
    chain = _fallback_chain_profile_ids_validated(catalog, lane)
    chain_sha = _fallback_chain_sha(catalog, lane, chain)
    body = {
        "contract": ACTIVATION_RECEIPT_CONTRACT,
        "catalog_sha256": catalog.catalog_sha256,
        "pair": lane.pair,
        "profile_id": lane.profile_id,
        "profile_sha256": profile.profile_sha256,
        "horizon_lane": lane.horizon_lane,
        "fallback_chain_profile_ids": list(chain),
        "fallback_chain_sha256": chain_sha,
        "evidence_sha256s": list(evidence),
        "activated_at_utc": _canonical_utc(activated_at_utc),
        **_AUTHORITY_BODY,
    }
    return {
        **body,
        "activation_receipt_sha256": canonical_sha256(body),
    }


def validate_activation_receipt(
    value: object,
    catalog: FastBotProfileCatalog | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and detach one activation receipt against exact catalog bytes."""

    catalog = validate_fast_bot_profile_catalog(catalog)
    return _validate_activation_receipt_against_catalog(value, catalog)


def _validate_activation_receipt_against_catalog(
    value: object,
    catalog: FastBotProfileCatalog,
) -> dict[str, Any]:
    """Validate one receipt against an already validated immutable catalog."""

    snapshot = _snapshot_mapping(value, label="activation receipt")
    _require_exact_keys(snapshot, _ACTIVATION_KEYS, label="activation receipt")
    if (
        snapshot["contract"].__class__ is not str
        or snapshot["contract"] != ACTIVATION_RECEIPT_CONTRACT
    ):
        raise ProfileContractError("activation receipt contract mismatch")
    _require_authority(snapshot, label="activation receipt")
    if (
        _require_sha(snapshot["catalog_sha256"], label="catalog_sha256")
        != catalog.catalog_sha256
    ):
        raise ProfileContractError("activation receipt catalog mismatch")

    lane = LaneKey(
        snapshot["pair"],
        snapshot["profile_id"],
        snapshot["horizon_lane"],
    )
    profile = catalog.profile(lane.profile_id)
    if not profile.supports(lane.pair, lane.horizon_lane):
        raise ProfileContractError("activation profile does not support lane")
    if profile.activation_eligibility != PRIMARY_ELIGIBLE:
        raise ProfileContractError("profile is not research-primary eligible")
    if (
        _require_sha(snapshot["profile_sha256"], label="profile_sha256")
        != profile.profile_sha256
    ):
        raise ProfileContractError("activation profile_sha256 mismatch")

    chain_raw = _require_json_list(
        snapshot["fallback_chain_profile_ids"],
        label="fallback_chain_profile_ids",
    )
    if any(item.__class__ is not str for item in chain_raw):
        raise ProfileContractError("fallback chain ids must be exact strings")
    chain = tuple(chain_raw)
    expected_chain = _fallback_chain_profile_ids_validated(catalog, lane)
    if chain != expected_chain:
        raise ProfileContractError("activation fallback chain is not registry sealed")
    if _require_sha(
        snapshot["fallback_chain_sha256"], label="fallback_chain_sha256"
    ) != _fallback_chain_sha(catalog, lane, chain):
        raise ProfileContractError("activation fallback chain digest mismatch")

    evidence_raw = _require_json_list(
        snapshot["evidence_sha256s"], label="evidence_sha256s"
    )
    evidence = _canonical_evidence(evidence_raw)
    if list(evidence) != evidence_raw:
        raise ProfileContractError("activation evidence is not canonical")
    activated_at_utc = _require_canonical_utc(
        snapshot["activated_at_utc"], label="activated_at_utc"
    )

    stored_sha = _require_sha(
        snapshot["activation_receipt_sha256"],
        label="activation_receipt_sha256",
    )
    # Rebuild every nested collection from validated immutable tuples.  A
    # shallow ``dict(value)`` would leave the caller's lists aliased into the
    # supposedly detached state and permit post-validation chain mutation.
    body = {
        "contract": ACTIVATION_RECEIPT_CONTRACT,
        "catalog_sha256": catalog.catalog_sha256,
        "pair": lane.pair,
        "profile_id": lane.profile_id,
        "profile_sha256": profile.profile_sha256,
        "horizon_lane": lane.horizon_lane,
        "fallback_chain_profile_ids": list(chain),
        "fallback_chain_sha256": _fallback_chain_sha(catalog, lane, chain),
        "evidence_sha256s": list(evidence),
        "activated_at_utc": activated_at_utc,
        **_AUTHORITY_BODY,
    }
    if canonical_sha256(body) != stored_sha:
        raise ProfileContractError("activation receipt digest mismatch")
    return {**body, "activation_receipt_sha256": stored_sha}


def build_quarantine_record(
    *,
    catalog: FastBotProfileCatalog | Mapping[str, Any],
    lane: LaneKey,
    reason_code: str,
    evidence_sha256s: Sequence[str],
    quarantined_at_utc: datetime,
) -> dict[str, Any]:
    """Seal a terminal quarantine for exactly one profile lane."""

    catalog = validate_fast_bot_profile_catalog(catalog)
    if lane.__class__ is not LaneKey:
        raise ProfileContractError("quarantine lane must be an exact LaneKey")
    profile = catalog.profile(lane.profile_id)
    if not profile.supports(lane.pair, lane.horizon_lane):
        raise ProfileContractError("quarantine profile does not support lane")
    reason = _require_reason(reason_code)
    evidence = _canonical_evidence(evidence_sha256s)
    body = {
        "contract": QUARANTINE_RECORD_CONTRACT,
        "catalog_sha256": catalog.catalog_sha256,
        "pair": lane.pair,
        "profile_id": lane.profile_id,
        "profile_sha256": profile.profile_sha256,
        "horizon_lane": lane.horizon_lane,
        "reason_code": reason,
        "evidence_sha256s": list(evidence),
        "quarantined_at_utc": _canonical_utc(quarantined_at_utc),
        "terminal_for_profile_revision": True,
        **_AUTHORITY_BODY,
    }
    return {
        **body,
        "quarantine_record_sha256": canonical_sha256(body),
    }


def validate_quarantine_record(
    value: object,
    catalog: FastBotProfileCatalog | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and detach one exact-lane quarantine record."""

    catalog = validate_fast_bot_profile_catalog(catalog)
    return _validate_quarantine_record_against_catalog(value, catalog)


def _validate_quarantine_record_against_catalog(
    value: object,
    catalog: FastBotProfileCatalog,
) -> dict[str, Any]:
    """Validate one quarantine against an already validated catalog."""

    snapshot = _snapshot_mapping(value, label="quarantine record")
    _require_exact_keys(snapshot, _QUARANTINE_KEYS, label="quarantine record")
    if (
        snapshot["contract"].__class__ is not str
        or snapshot["contract"] != QUARANTINE_RECORD_CONTRACT
    ):
        raise ProfileContractError("quarantine record contract mismatch")
    _require_authority(snapshot, label="quarantine record")
    if (
        _require_sha(snapshot["catalog_sha256"], label="catalog_sha256")
        != catalog.catalog_sha256
    ):
        raise ProfileContractError("quarantine record catalog mismatch")

    lane = LaneKey(
        snapshot["pair"],
        snapshot["profile_id"],
        snapshot["horizon_lane"],
    )
    profile = catalog.profile(lane.profile_id)
    if not profile.supports(lane.pair, lane.horizon_lane):
        raise ProfileContractError("quarantine profile does not support lane")
    if (
        _require_sha(snapshot["profile_sha256"], label="profile_sha256")
        != profile.profile_sha256
    ):
        raise ProfileContractError("quarantine profile_sha256 mismatch")
    reason_code = _require_reason(snapshot["reason_code"])

    evidence_raw = _require_json_list(
        snapshot["evidence_sha256s"], label="evidence_sha256s"
    )
    evidence = _canonical_evidence(evidence_raw)
    if list(evidence) != evidence_raw:
        raise ProfileContractError("quarantine evidence is not canonical")
    quarantined_at_utc = _require_canonical_utc(
        snapshot["quarantined_at_utc"], label="quarantined_at_utc"
    )
    if snapshot["terminal_for_profile_revision"] is not True:
        raise ProfileContractError("quarantine must be terminal for profile revision")

    stored_sha = _require_sha(
        snapshot["quarantine_record_sha256"],
        label="quarantine_record_sha256",
    )
    body = {
        "contract": QUARANTINE_RECORD_CONTRACT,
        "catalog_sha256": catalog.catalog_sha256,
        "pair": lane.pair,
        "profile_id": lane.profile_id,
        "profile_sha256": profile.profile_sha256,
        "horizon_lane": lane.horizon_lane,
        "reason_code": reason_code,
        "evidence_sha256s": list(evidence),
        "quarantined_at_utc": quarantined_at_utc,
        "terminal_for_profile_revision": True,
        **_AUTHORITY_BODY,
    }
    if canonical_sha256(body) != stored_sha:
        raise ProfileContractError("quarantine record digest mismatch")
    return {**body, "quarantine_record_sha256": stored_sha}


def index_activation_receipts(
    catalog: FastBotProfileCatalog | Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate receipts and reject even identical duplicate lane scopes."""

    catalog = validate_fast_bot_profile_catalog(catalog)
    return _index_activation_receipts_against_catalog(catalog, receipts)


def _index_activation_receipts_against_catalog(
    catalog: FastBotProfileCatalog,
    receipts: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index rows after the caller has validated the catalog exactly once."""

    index: dict[tuple[str, str], dict[str, Any]] = {}
    snapshot = _snapshot_sequence(receipts, label="activation receipts")
    scope_count = len({(lane.pair, lane.horizon_lane) for lane in catalog.lanes()})
    if len(snapshot) > scope_count:
        raise ProfileContractError("activation receipts exceed catalog scope count")
    for raw in snapshot:
        receipt = _validate_activation_receipt_against_catalog(raw, catalog)
        scope = (receipt["pair"], receipt["horizon_lane"])
        if scope in index:
            raise ProfileContractError("multiple activation receipts for one scope")
        index[scope] = receipt
    return index


def index_quarantine_records(
    catalog: FastBotProfileCatalog | Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[LaneKey, dict[str, Any]]:
    """Validate quarantine records and reject duplicate exact lanes."""

    catalog = validate_fast_bot_profile_catalog(catalog)
    return _index_quarantine_records_against_catalog(catalog, records)


def _index_quarantine_records_against_catalog(
    catalog: FastBotProfileCatalog,
    records: Sequence[Mapping[str, Any]],
) -> dict[LaneKey, dict[str, Any]]:
    """Index rows after the caller has validated the catalog exactly once."""

    index: dict[LaneKey, dict[str, Any]] = {}
    snapshot = _snapshot_sequence(records, label="quarantine records")
    if len(snapshot) > len(catalog.lanes()):
        raise ProfileContractError("quarantine records exceed catalog lane count")
    for raw in snapshot:
        record = _validate_quarantine_record_against_catalog(raw, catalog)
        lane = LaneKey(
            record["pair"],
            record["profile_id"],
            record["horizon_lane"],
        )
        if lane in index:
            raise ProfileContractError("duplicate quarantine record for exact lane")
        index[lane] = record
    return index


def _fallback_chain_sha(
    catalog: FastBotProfileCatalog,
    lane: LaneKey,
    chain: Sequence[str],
) -> str:
    return canonical_sha256(
        {
            "catalog_sha256": catalog.catalog_sha256,
            "pair": lane.pair,
            "horizon_lane": lane.horizon_lane,
            "profile_ids": list(chain),
        }
    )


def _canonical_evidence(value: object) -> tuple[str, ...]:
    snapshot = _snapshot_sequence(value, label="evidence_sha256s")
    if not snapshot:
        raise ProfileContractError("evidence_sha256s must not be empty")
    if len(snapshot) > MAX_EVIDENCE_DIGESTS:
        raise ProfileContractError("evidence_sha256s exceeds its bounded capacity")
    evidence = tuple(_require_sha(item, label="evidence SHA-256") for item in snapshot)
    if len(set(evidence)) != len(evidence):
        raise ProfileContractError("evidence_sha256s contains duplicates")
    return tuple(sorted(evidence))


def _canonical_utc(value: datetime) -> str:
    if value.__class__ is not datetime or value.tzinfo is None:
        raise ProfileContractError("artifact timestamp must be an aware datetime")
    utc = value.astimezone(timezone.utc)
    rendered = utc.isoformat(timespec="microseconds" if utc.microsecond else "seconds")
    return rendered.replace("+00:00", "Z")


def _require_canonical_utc(value: object, *, label: str) -> str:
    if (
        value.__class__ is not str
        or len(value) > MAX_CANONICAL_UTC_CHARS
        or not value.endswith("Z")
    ):
        raise ProfileContractError(f"{label} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ProfileContractError(f"{label} must be canonical UTC") from exc
    if _canonical_utc(parsed) != value:
        raise ProfileContractError(f"{label} must be canonical UTC")
    return value


def _require_reason(value: object) -> str:
    if (
        value.__class__ is not str
        or len(value) > MAX_REASON_CODE_CHARS
        or not _REASON_RE.fullmatch(value)
    ):
        raise ProfileContractError("reason_code must be a canonical token")
    return value


def _require_sha(value: object, *, label: str) -> str:
    if value.__class__ is not str or not _SHA256_RE.fullmatch(value):
        raise ProfileContractError(f"{label} must be a lowercase SHA-256")
    return value


def _snapshot_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileContractError(f"{label} must be a mapping")
    try:
        snapshot = dict(value)
    except Exception as exc:
        raise ProfileContractError(f"{label} snapshot is unreadable") from exc
    if any(key.__class__ is not str for key in snapshot):
        raise ProfileContractError(f"{label} keys must be exact strings")
    return snapshot


def _snapshot_sequence(value: object, *, label: str) -> tuple[Any, ...]:
    if value.__class__ not in {list, tuple}:
        raise ProfileContractError(f"{label} must be a list or tuple")
    try:
        return tuple(value)
    except Exception as exc:
        raise ProfileContractError(f"{label} snapshot is unreadable") from exc


def _require_json_list(value: object, *, label: str) -> list[Any]:
    if value.__class__ is not list:
        raise ProfileContractError(f"{label} must be a JSON list")
    return list(value)


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    if frozenset(value) != expected:
        raise ProfileContractError(f"{label} has non-canonical keys")


def _require_authority(value: Mapping[str, Any], *, label: str) -> None:
    for key, expected in _AUTHORITY_BODY.items():
        actual = value.get(key)
        if actual.__class__ is not expected.__class__ or actual != expected:
            raise ProfileContractError(f"{label} has unsafe {key}")

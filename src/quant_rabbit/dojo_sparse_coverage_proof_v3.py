"""Versioned fail-closed diagnostic for sparse M5 DOJO source coverage.

This module is deliberately independent from the existing long-horizon source
manifest and train-control paths.  It cannot upgrade an existing generation.
It diagnoses only that a candidate input supplied:

* one sealed, gap-free market-calendar row for every aligned M5 grid slot;
* independent provider evidence for every slot declared closed; and
* exactly one ``OBSERVED`` or independently receipted
  ``LEGITIMATE_NO_CANDLE`` classification for every pair/open-slot cell.

Content hashes do not authenticate provider or independent-verifier bytes.
Consequently this contract never emits promotion-eligible coverage proof.  It
never invents or carries a quote and has no broker, promotion, order, sizing,
or live authority.
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
from pathlib import Path
from typing import Any, Final


MARKET_CALENDAR_SPEC_CONTRACT: Final = "QR_DOJO_MARKET_CALENDAR_SPEC_V3"
MARKET_CALENDAR_ARTIFACT_CONTRACT: Final = "QR_DOJO_SEALED_MARKET_CALENDAR_V3"
MARKET_CLOSURE_RECEIPT_CONTRACT: Final = "QR_DOJO_INDEPENDENT_MARKET_CLOSURE_RECEIPT_V1"
COVERAGE_INPUT_CONTRACT: Final = "QR_DOJO_PAIR_SLOT_COVERAGE_INPUT_V3"
OBSERVED_CANDLE_CONTRACT: Final = "QR_DOJO_OBSERVED_M5_CANDLE_V1"
NO_CANDLE_RECEIPT_CONTRACT: Final = "QR_DOJO_INDEPENDENT_PROVIDER_NO_CANDLE_RECEIPT_V1"
COVERAGE_PROOF_RECEIPT_CONTRACT: Final = "QR_DOJO_SPARSE_COVERAGE_DIAGNOSTIC_V3"
SCHEMA_VERSION: Final = 3
EVIDENCE_SCHEMA_VERSION: Final = 1
GRANULARITY: Final = "M5"
CADENCE_SECONDS: Final = 300
EXPECTED_OPEN: Final = "EXPECTED_OPEN"
VERIFIED_CLOSED: Final = "VERIFIED_CLOSED"
OBSERVED: Final = "OBSERVED"
LEGITIMATE_NO_CANDLE: Final = "LEGITIMATE_NO_CANDLE"
NO_CANDLE_REASON: Final = "COMPLETE_PROVIDER_RESPONSE_OMITTED_CANDLE"
CLOSURE_REASONS: Final = frozenset(
    {
        "PROVIDER_PUBLISHED_WEEKLY_CLOSURE",
        "PROVIDER_PUBLISHED_MAINTENANCE_CLOSURE",
        "PROVIDER_PUBLISHED_HOLIDAY_CLOSURE",
        "PROVIDER_PUBLISHED_SPECIAL_CLOSURE",
    }
)
MAX_JSON_BYTES: Final = 256 * 1024 * 1024
# One proof artifact is intentionally month-sized.  Longer studies chain
# independently sealed months instead of building an unbounded in-memory
# pair×slot set or one giant archive object.
MAX_GRID_SLOTS: Final = 10_000
MAX_PAIRS: Final = 64
MAX_PAIR_SLOT_CELLS: Final = MAX_GRID_SLOTS * MAX_PAIRS

_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PAIR_RE: Final = re.compile(r"[A-Z]{3}_[A-Z]{3}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")

_AUTHORITY: Final = {
    "automatic_deployment_allowed": False,
    "broker_mutation_allowed": False,
    "coverage_promotion_proof_allowed": False,
    "existing_generation_upgrade_allowed": False,
    "live_permission": False,
    "order_authority": "NONE",
    "promotion_eligible": False,
    "requires_new_generation_binding": True,
    "research_source_coverage_only": True,
    "source_authentication_proved": False,
}

_CALENDAR_SPEC_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "provider",
        "calendar_id",
        "granularity",
        "from_epoch",
        "to_epoch",
        "source_producer_id",
        "independent_verifier_id",
        "closure_receipts",
        "slots",
    }
)
_CALENDAR_ARTIFACT_KEYS: Final = frozenset(
    {
        *_CALENDAR_SPEC_KEYS,
        "grid_slot_count",
        "expected_open_slot_count",
        "verified_closed_slot_count",
        "closure_receipts_sha256",
        "slots_sha256",
        "authority",
        "calendar_artifact_sha256",
    }
)
_CLOSURE_RECEIPT_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "provider",
        "reason",
        "effective_from_epoch",
        "effective_to_epoch",
        "provider_evidence_sha256",
        "independent_verification_artifact_sha256",
        "source_producer_id",
        "independent_verifier_id",
        "receipt_sha256",
    }
)
_OPEN_SLOT_KEYS: Final = frozenset({"epoch", "state"})
_CLOSED_SLOT_KEYS: Final = frozenset({"epoch", "state", "closure_receipt_sha256"})
_COVERAGE_INPUT_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "calendar_artifact_sha256",
        "provider",
        "granularity",
        "from_epoch",
        "to_epoch",
        "feed_pairs",
        "source_producer_id",
        "independent_verifier_id",
        "observed_candles",
        "no_candle_receipts",
        "classifications",
    }
)
_OBSERVED_CANDLE_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "pair",
        "epoch",
        "granularity",
        "complete",
        "bid",
        "ask",
        "source_artifact_sha256",
        "candle_sha256",
    }
)
_NO_CANDLE_RECEIPT_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "provider",
        "pair",
        "granularity",
        "reason",
        "effective_from_epoch",
        "effective_to_epoch",
        "request_from_epoch",
        "request_to_epoch",
        "provider_request_sha256",
        "provider_response_sha256",
        "independent_verification_artifact_sha256",
        "calendar_artifact_sha256",
        "source_producer_id",
        "independent_verifier_id",
        "receipt_sha256",
    }
)
_OBSERVED_CLASSIFICATION_KEYS: Final = frozenset(
    {"pair", "epoch", "classification", "observed_candle_sha256"}
)
_NO_CANDLE_CLASSIFICATION_KEYS: Final = frozenset(
    {"pair", "epoch", "classification", "no_candle_receipt_sha256"}
)
_PROOF_RECEIPT_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "calendar_artifact_sha256",
        "coverage_input_sha256",
        "provider",
        "granularity",
        "from_epoch",
        "to_epoch",
        "feed_pairs",
        "grid_slot_count",
        "expected_open_slot_count",
        "verified_closed_slot_count",
        "expected_pair_slot_count",
        "classified_pair_slot_count",
        "observed_pair_slot_count",
        "legitimate_no_candle_pair_slot_count",
        "unclassified_pair_slot_count",
        "duplicate_pair_slot_count",
        "out_of_range_pair_slot_count",
        "observed_candle_mismatch_count",
        "no_candle_receipt_mismatch_count",
        "observed_candles_sha256",
        "no_candle_receipts_sha256",
        "classifications_sha256",
        "complete_pair_slot_classification",
        "proof_classification",
        "sparse_calendar_coverage_proved",
        "proof_eligible",
        "authority",
        "receipt_sha256",
    }
)


class DojoSparseCoverageProofV3Error(ValueError):
    """The V3 source-coverage diagnostic is incomplete or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict canonical JSON bytes after rejecting non-string keys."""

    _require_json_shape(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoSparseCoverageProofV3Error("value is not strict JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def build_market_closure_receipt(
    *,
    provider: str,
    reason: str,
    effective_from_epoch: int,
    effective_to_epoch: int,
    provider_evidence_sha256: str,
    independent_verification_artifact_sha256: str,
    source_producer_id: str,
    independent_verifier_id: str,
) -> dict[str, Any]:
    """Seal independent provider evidence for one aligned closure interval."""

    body = {
        "contract": MARKET_CLOSURE_RECEIPT_CONTRACT,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "provider": _identifier(provider, field="closure provider"),
        "reason": _closure_reason(reason),
        "effective_from_epoch": _aligned_epoch(
            effective_from_epoch, field="closure effective_from_epoch"
        ),
        "effective_to_epoch": _aligned_epoch(
            effective_to_epoch, field="closure effective_to_epoch"
        ),
        "provider_evidence_sha256": _sha256(
            provider_evidence_sha256, field="closure provider evidence"
        ),
        "independent_verification_artifact_sha256": _sha256(
            independent_verification_artifact_sha256,
            field="closure independent verification artifact",
        ),
        "source_producer_id": _identifier(
            source_producer_id, field="closure source_producer_id"
        ),
        "independent_verifier_id": _identifier(
            independent_verifier_id, field="closure independent_verifier_id"
        ),
    }
    if body["effective_to_epoch"] <= body["effective_from_epoch"]:
        raise DojoSparseCoverageProofV3Error(
            "closure effective interval must be positive"
        )
    _require_independent_identities(
        body["source_producer_id"],
        body["independent_verifier_id"],
        field="closure receipt",
    )
    return {**body, "receipt_sha256": canonical_sha256(body)}


def verify_market_closure_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact_mapping(value, _CLOSURE_RECEIPT_KEYS, field="closure receipt")
    if row["contract"] != MARKET_CLOSURE_RECEIPT_CONTRACT or not _is_exact_integer(
        row["schema_version"], EVIDENCE_SCHEMA_VERSION
    ):
        raise DojoSparseCoverageProofV3Error("closure receipt identity differs")
    expected = build_market_closure_receipt(
        provider=row["provider"],
        reason=row["reason"],
        effective_from_epoch=row["effective_from_epoch"],
        effective_to_epoch=row["effective_to_epoch"],
        provider_evidence_sha256=row["provider_evidence_sha256"],
        independent_verification_artifact_sha256=row[
            "independent_verification_artifact_sha256"
        ],
        source_producer_id=row["source_producer_id"],
        independent_verifier_id=row["independent_verifier_id"],
    )
    if not _canonical_equal(row, expected):
        raise DojoSparseCoverageProofV3Error("closure receipt seal differs")
    return expected


def build_sealed_market_calendar_artifact(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and seal every aligned M5 grid slot without inferred holidays."""

    raw = _exact_top_mapping(spec, _CALENDAR_SPEC_KEYS, field="market calendar spec")
    if (
        raw["contract"] != MARKET_CALENDAR_SPEC_CONTRACT
        or not _is_exact_integer(raw["schema_version"], SCHEMA_VERSION)
        or raw["granularity"] != GRANULARITY
    ):
        raise DojoSparseCoverageProofV3Error("market calendar identity differs")
    provider = _identifier(raw["provider"], field="calendar provider")
    calendar_id = _identifier(raw["calendar_id"], field="calendar_id")
    start = _aligned_epoch(raw["from_epoch"], field="calendar from_epoch")
    end = _aligned_epoch(raw["to_epoch"], field="calendar to_epoch")
    if end <= start:
        raise DojoSparseCoverageProofV3Error("calendar interval must be positive")
    grid_count = (end - start) // CADENCE_SECONDS
    if not 1 <= grid_count <= MAX_GRID_SLOTS:
        raise DojoSparseCoverageProofV3Error("calendar grid exceeds its bound")
    producer = _identifier(
        raw["source_producer_id"], field="calendar source_producer_id"
    )
    verifier = _identifier(
        raw["independent_verifier_id"], field="calendar independent_verifier_id"
    )
    _require_independent_identities(producer, verifier, field="calendar")

    closure_raw = _sequence(raw["closure_receipts"], field="closure_receipts")
    if len(closure_raw) > grid_count:
        raise DojoSparseCoverageProofV3Error(
            "closure receipt count exceeds the calendar grid"
        )
    closures = [verify_market_closure_receipt(row) for row in closure_raw]
    closure_by_sha: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[int, int, str]] = []
    for closure in closures:
        digest = closure["receipt_sha256"]
        if digest in closure_by_sha:
            raise DojoSparseCoverageProofV3Error("closure receipt is duplicated")
        if (
            closure["provider"] != provider
            or closure["source_producer_id"] != producer
            or closure["independent_verifier_id"] != verifier
        ):
            raise DojoSparseCoverageProofV3Error(
                "closure receipt provider or independent identity differs"
            )
        if (
            closure["effective_from_epoch"] < start
            or closure["effective_to_epoch"] > end
        ):
            raise DojoSparseCoverageProofV3Error(
                "closure receipt effective interval is outside the calendar"
            )
        closure_by_sha[digest] = closure
        intervals.append(
            (
                closure["effective_from_epoch"],
                closure["effective_to_epoch"],
                digest,
            )
        )
    intervals.sort()
    for left, right in zip(intervals, intervals[1:], strict=False):
        if right[0] < left[1]:
            raise DojoSparseCoverageProofV3Error(
                "closure receipt effective intervals overlap"
            )
    closures.sort(
        key=lambda row: (
            row["effective_from_epoch"],
            row["effective_to_epoch"],
            row["receipt_sha256"],
        )
    )

    slot_raw = _sequence(raw["slots"], field="calendar slots")
    if len(slot_raw) != grid_count:
        raise DojoSparseCoverageProofV3Error(
            "calendar must classify every aligned M5 grid slot"
        )
    slots: list[dict[str, Any]] = []
    expected_epochs = list(range(start, end, CADENCE_SECONDS))
    used_closures: set[str] = set()
    for expected_epoch, value in zip(expected_epochs, slot_raw, strict=True):
        row = _mapping(value, field="calendar slot")
        state = row.get("state")
        keys = _OPEN_SLOT_KEYS if state == EXPECTED_OPEN else _CLOSED_SLOT_KEYS
        row = _exact_mapping(row, keys, field="calendar slot")
        epoch = _aligned_epoch(row["epoch"], field="calendar slot epoch")
        if epoch != expected_epoch:
            raise DojoSparseCoverageProofV3Error(
                "calendar slot is duplicated, missing, out of range, or unsorted"
            )
        if state == EXPECTED_OPEN:
            slots.append({"epoch": epoch, "state": EXPECTED_OPEN})
            continue
        if state != VERIFIED_CLOSED:
            raise DojoSparseCoverageProofV3Error("calendar slot state is invalid")
        digest = _sha256(
            row["closure_receipt_sha256"], field="closure receipt reference"
        )
        closure = closure_by_sha.get(digest)
        if closure is None:
            raise DojoSparseCoverageProofV3Error(
                "closed slot lacks its independent closure receipt"
            )
        if not (
            closure["effective_from_epoch"] <= epoch
            and epoch + CADENCE_SECONDS <= closure["effective_to_epoch"]
        ):
            raise DojoSparseCoverageProofV3Error(
                "closed slot differs from the receipt effective interval"
            )
        used_closures.add(digest)
        slots.append(
            {
                "epoch": epoch,
                "state": VERIFIED_CLOSED,
                "closure_receipt_sha256": digest,
            }
        )
    if used_closures != set(closure_by_sha):
        raise DojoSparseCoverageProofV3Error(
            "calendar contains an unreferenced closure receipt"
        )
    for interval_start, interval_end, digest in intervals:
        interval_epochs = range(interval_start, interval_end, CADENCE_SECONDS)
        if any(
            slots[(epoch - start) // CADENCE_SECONDS].get("closure_receipt_sha256")
            != digest
            for epoch in interval_epochs
        ):
            raise DojoSparseCoverageProofV3Error(
                "closure receipt interval is not classified closed in full"
            )
    open_count = sum(row["state"] == EXPECTED_OPEN for row in slots)
    if open_count < 1:
        raise DojoSparseCoverageProofV3Error(
            "calendar must contain at least one expected open slot"
        )
    body = {
        "contract": MARKET_CALENDAR_ARTIFACT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "provider": provider,
        "calendar_id": calendar_id,
        "granularity": GRANULARITY,
        "from_epoch": start,
        "to_epoch": end,
        "source_producer_id": producer,
        "independent_verifier_id": verifier,
        "closure_receipts": closures,
        "slots": slots,
        "grid_slot_count": grid_count,
        "expected_open_slot_count": open_count,
        "verified_closed_slot_count": grid_count - open_count,
        "closure_receipts_sha256": canonical_sha256(closures),
        "slots_sha256": canonical_sha256(slots),
        "authority": _json_copy(_AUTHORITY),
    }
    return {**body, "calendar_artifact_sha256": canonical_sha256(body)}


def verify_sealed_market_calendar_artifact(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    row = _exact_top_mapping(
        value, _CALENDAR_ARTIFACT_KEYS, field="sealed market calendar"
    )
    spec = {key: row[key] for key in _CALENDAR_SPEC_KEYS}
    spec["contract"] = MARKET_CALENDAR_SPEC_CONTRACT
    expected = build_sealed_market_calendar_artifact(spec)
    if not _canonical_equal(row, expected):
        raise DojoSparseCoverageProofV3Error(
            "sealed market calendar differs from its canonical reconstruction"
        )
    return expected


def build_observed_candle(
    *,
    pair: str,
    epoch: int,
    bid: Sequence[Any],
    ask: Sequence[Any],
    source_artifact_sha256: str,
) -> dict[str, Any]:
    """Seal one complete observed bid/ask OHLC candle."""

    normalized_bid = _ohlc(bid, field="observed bid")
    normalized_ask = _ohlc(ask, field="observed ask")
    if any(
        ask_value < bid_value
        for bid_value, ask_value in zip(normalized_bid, normalized_ask, strict=True)
    ):
        raise DojoSparseCoverageProofV3Error("observed candle has negative spread")
    body = {
        "contract": OBSERVED_CANDLE_CONTRACT,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "pair": _pair(pair),
        "epoch": _aligned_epoch(epoch, field="observed candle epoch"),
        "granularity": GRANULARITY,
        "complete": True,
        "bid": normalized_bid,
        "ask": normalized_ask,
        "source_artifact_sha256": _sha256(
            source_artifact_sha256, field="observed source artifact"
        ),
    }
    return {**body, "candle_sha256": canonical_sha256(body)}


def verify_observed_candle(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact_mapping(value, _OBSERVED_CANDLE_KEYS, field="observed candle")
    if (
        row["contract"] != OBSERVED_CANDLE_CONTRACT
        or not _is_exact_integer(row["schema_version"], EVIDENCE_SCHEMA_VERSION)
        or row["granularity"] != GRANULARITY
        or row["complete"] is not True
    ):
        raise DojoSparseCoverageProofV3Error("observed candle identity differs")
    expected = build_observed_candle(
        pair=row["pair"],
        epoch=row["epoch"],
        bid=row["bid"],
        ask=row["ask"],
        source_artifact_sha256=row["source_artifact_sha256"],
    )
    if not _canonical_equal(row, expected):
        raise DojoSparseCoverageProofV3Error("observed candle content or seal differs")
    return expected


def build_no_candle_receipt(
    *,
    provider: str,
    pair: str,
    effective_from_epoch: int,
    effective_to_epoch: int,
    request_from_epoch: int,
    request_to_epoch: int,
    provider_request_sha256: str,
    provider_response_sha256: str,
    independent_verification_artifact_sha256: str,
    calendar_artifact_sha256: str,
    source_producer_id: str,
    independent_verifier_id: str,
    reason: str = NO_CANDLE_REASON,
) -> dict[str, Any]:
    """Seal an independently verified provider omission for exactly one slot."""

    body = {
        "contract": NO_CANDLE_RECEIPT_CONTRACT,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "provider": _identifier(provider, field="no-candle provider"),
        "pair": _pair(pair),
        "granularity": GRANULARITY,
        "reason": _no_candle_reason(reason),
        "effective_from_epoch": _aligned_epoch(
            effective_from_epoch, field="no-candle effective_from_epoch"
        ),
        "effective_to_epoch": _aligned_epoch(
            effective_to_epoch, field="no-candle effective_to_epoch"
        ),
        "request_from_epoch": _aligned_epoch(
            request_from_epoch, field="no-candle request_from_epoch"
        ),
        "request_to_epoch": _aligned_epoch(
            request_to_epoch, field="no-candle request_to_epoch"
        ),
        "provider_request_sha256": _sha256(
            provider_request_sha256, field="provider request"
        ),
        "provider_response_sha256": _sha256(
            provider_response_sha256, field="provider response"
        ),
        "independent_verification_artifact_sha256": _sha256(
            independent_verification_artifact_sha256,
            field="independent verification artifact",
        ),
        "calendar_artifact_sha256": _sha256(
            calendar_artifact_sha256, field="no-candle calendar artifact"
        ),
        "source_producer_id": _identifier(
            source_producer_id, field="no-candle source_producer_id"
        ),
        "independent_verifier_id": _identifier(
            independent_verifier_id, field="no-candle independent_verifier_id"
        ),
    }
    if body["effective_to_epoch"] - body["effective_from_epoch"] != CADENCE_SECONDS:
        raise DojoSparseCoverageProofV3Error(
            "no-candle effective interval must be exactly one M5 slot"
        )
    if not (
        body["request_from_epoch"] <= body["effective_from_epoch"]
        and body["effective_to_epoch"] <= body["request_to_epoch"]
        and body["request_to_epoch"] > body["request_from_epoch"]
    ):
        raise DojoSparseCoverageProofV3Error(
            "no-candle request interval does not cover its effective slot"
        )
    _require_independent_identities(
        body["source_producer_id"],
        body["independent_verifier_id"],
        field="no-candle receipt",
    )
    return {**body, "receipt_sha256": canonical_sha256(body)}


def verify_no_candle_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact_mapping(value, _NO_CANDLE_RECEIPT_KEYS, field="no-candle receipt")
    if (
        row["contract"] != NO_CANDLE_RECEIPT_CONTRACT
        or not _is_exact_integer(row["schema_version"], EVIDENCE_SCHEMA_VERSION)
        or row["granularity"] != GRANULARITY
    ):
        raise DojoSparseCoverageProofV3Error("no-candle receipt identity differs")
    expected = build_no_candle_receipt(
        provider=row["provider"],
        pair=row["pair"],
        reason=row["reason"],
        effective_from_epoch=row["effective_from_epoch"],
        effective_to_epoch=row["effective_to_epoch"],
        request_from_epoch=row["request_from_epoch"],
        request_to_epoch=row["request_to_epoch"],
        provider_request_sha256=row["provider_request_sha256"],
        provider_response_sha256=row["provider_response_sha256"],
        independent_verification_artifact_sha256=row[
            "independent_verification_artifact_sha256"
        ],
        calendar_artifact_sha256=row["calendar_artifact_sha256"],
        source_producer_id=row["source_producer_id"],
        independent_verifier_id=row["independent_verifier_id"],
    )
    if not _canonical_equal(row, expected):
        raise DojoSparseCoverageProofV3Error(
            "no-candle provider evidence or receipt seal differs"
        )
    return expected


def build_coverage_proof_receipt(
    *,
    calendar_artifact: Mapping[str, Any],
    coverage_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive candidate diagnostics without claiming source authentication."""

    calendar = verify_sealed_market_calendar_artifact(calendar_artifact)
    coverage = _exact_top_mapping(
        coverage_input, _COVERAGE_INPUT_KEYS, field="coverage input"
    )
    if (
        coverage["contract"] != COVERAGE_INPUT_CONTRACT
        or not _is_exact_integer(coverage["schema_version"], SCHEMA_VERSION)
        or coverage["granularity"] != GRANULARITY
    ):
        raise DojoSparseCoverageProofV3Error("coverage input identity differs")
    calendar_sha = calendar["calendar_artifact_sha256"]
    coverage_calendar_sha = _sha256(
        coverage["calendar_artifact_sha256"],
        field="coverage calendar artifact",
    )
    coverage_provider = _identifier(coverage["provider"], field="coverage provider")
    coverage_start = _aligned_epoch(coverage["from_epoch"], field="coverage from_epoch")
    coverage_end = _aligned_epoch(coverage["to_epoch"], field="coverage to_epoch")
    if (
        coverage_calendar_sha != calendar_sha
        or coverage_provider != calendar["provider"]
        or coverage_start != calendar["from_epoch"]
        or coverage_end != calendar["to_epoch"]
    ):
        raise DojoSparseCoverageProofV3Error(
            "coverage input differs from the sealed calendar"
        )
    producer = _identifier(
        coverage["source_producer_id"], field="coverage source_producer_id"
    )
    verifier = _identifier(
        coverage["independent_verifier_id"],
        field="coverage independent_verifier_id",
    )
    _require_independent_identities(producer, verifier, field="coverage input")
    pairs = _feed_pairs(coverage["feed_pairs"])
    open_epochs = tuple(
        row["epoch"] for row in calendar["slots"] if row["state"] == EXPECTED_OPEN
    )
    expected_cell_count = len(pairs) * len(open_epochs)
    if expected_cell_count > MAX_PAIR_SLOT_CELLS:
        raise DojoSparseCoverageProofV3Error(
            "coverage pair-slot denominator exceeds its bound"
        )

    observed_raw = _sequence(coverage["observed_candles"], field="observed_candles")
    no_candle_raw = _sequence(
        coverage["no_candle_receipts"], field="no_candle_receipts"
    )
    classification_raw = _sequence(coverage["classifications"], field="classifications")
    if (
        len(classification_raw) != expected_cell_count
        or len(observed_raw) > expected_cell_count
        or len(no_candle_raw) > expected_cell_count
        or len(observed_raw) + len(no_candle_raw) != expected_cell_count
    ):
        raise DojoSparseCoverageProofV3Error(
            "pair-slot evidence/classification counts differ from the denominator"
        )
    observed_rows = [verify_observed_candle(row) for row in observed_raw]
    no_candle_rows = [verify_no_candle_receipt(row) for row in no_candle_raw]
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    expected_keys = {(pair, epoch) for pair in pairs for epoch in open_epochs}

    observed_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    observed_by_sha: dict[str, dict[str, Any]] = {}
    for row in observed_rows:
        key = (row["pair"], row["epoch"])
        if key in observed_by_key or row["candle_sha256"] in observed_by_sha:
            raise DojoSparseCoverageProofV3Error("observed candle is duplicated")
        if key not in expected_keys:
            raise DojoSparseCoverageProofV3Error(
                "observed candle pair/epoch is outside the expected open denominator"
            )
        observed_by_key[key] = row
        observed_by_sha[row["candle_sha256"]] = row

    no_candle_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    no_candle_by_sha: dict[str, dict[str, Any]] = {}
    for row in no_candle_rows:
        key = (row["pair"], row["effective_from_epoch"])
        if key in no_candle_by_key or row["receipt_sha256"] in no_candle_by_sha:
            raise DojoSparseCoverageProofV3Error("no-candle receipt is duplicated")
        if key not in expected_keys:
            raise DojoSparseCoverageProofV3Error(
                "no-candle pair/epoch is outside the expected open denominator"
            )
        if (
            row["provider"] != calendar["provider"]
            or row["calendar_artifact_sha256"] != calendar_sha
            or row["source_producer_id"] != producer
            or row["independent_verifier_id"] != verifier
            or row["request_from_epoch"] < calendar["from_epoch"]
            or row["request_to_epoch"] > calendar["to_epoch"]
        ):
            raise DojoSparseCoverageProofV3Error(
                "no-candle receipt provider, calendar, request window, or "
                "independent identity differs"
            )
        no_candle_by_key[key] = row
        no_candle_by_sha[row["receipt_sha256"]] = row

    observed_rows.sort(key=lambda row: (pair_index[row["pair"]], row["epoch"]))
    no_candle_rows.sort(
        key=lambda row: (
            pair_index[row["pair"]],
            row["effective_from_epoch"],
        )
    )

    classifications: list[dict[str, Any]] = []
    classified_keys: set[tuple[str, int]] = set()
    used_observed: set[str] = set()
    used_no_candle: set[str] = set()
    expected_order = [(pair, epoch) for pair in pairs for epoch in open_epochs]
    for expected_key, value in zip(expected_order, classification_raw, strict=True):
        raw_row = _mapping(value, field="pair-slot classification")
        classification = raw_row.get("classification")
        keys = (
            _OBSERVED_CLASSIFICATION_KEYS
            if classification == OBSERVED
            else _NO_CANDLE_CLASSIFICATION_KEYS
        )
        row = _exact_mapping(raw_row, keys, field="pair-slot classification")
        pair = _pair(row["pair"])
        epoch = _aligned_epoch(row["epoch"], field="classification epoch")
        key = (pair, epoch)
        if pair not in pair_index or key not in expected_keys:
            raise DojoSparseCoverageProofV3Error(
                "classification pair/epoch is outside the expected open denominator"
            )
        if key in classified_keys:
            raise DojoSparseCoverageProofV3Error(
                "pair-slot classification is duplicated"
            )
        if key != expected_key:
            raise DojoSparseCoverageProofV3Error(
                "pair-slot classification is missing, extra, or unsorted"
            )
        classified_keys.add(key)
        if classification == OBSERVED:
            digest = _sha256(
                row["observed_candle_sha256"], field="observed candle reference"
            )
            candle = observed_by_sha.get(digest)
            if candle is None or (candle["pair"], candle["epoch"]) != key:
                raise DojoSparseCoverageProofV3Error(
                    "OBSERVED classification differs from its sealed candle"
                )
            if digest in used_observed:
                raise DojoSparseCoverageProofV3Error(
                    "observed candle is referenced more than once"
                )
            used_observed.add(digest)
            classifications.append(
                {
                    "pair": pair,
                    "epoch": epoch,
                    "classification": OBSERVED,
                    "observed_candle_sha256": digest,
                }
            )
            continue
        if classification != LEGITIMATE_NO_CANDLE:
            raise DojoSparseCoverageProofV3Error(
                "pair-slot classification status is invalid"
            )
        digest = _sha256(
            row["no_candle_receipt_sha256"], field="no-candle receipt reference"
        )
        no_candle = no_candle_by_sha.get(digest)
        if (
            no_candle is None
            or (no_candle["pair"], no_candle["effective_from_epoch"]) != key
        ):
            raise DojoSparseCoverageProofV3Error(
                "LEGITIMATE_NO_CANDLE classification differs from its receipt"
            )
        if digest in used_no_candle:
            raise DojoSparseCoverageProofV3Error(
                "no-candle receipt is referenced more than once"
            )
        used_no_candle.add(digest)
        classifications.append(
            {
                "pair": pair,
                "epoch": epoch,
                "classification": LEGITIMATE_NO_CANDLE,
                "no_candle_receipt_sha256": digest,
            }
        )
    if classified_keys != expected_keys:
        raise DojoSparseCoverageProofV3Error(
            "not every expected pair/open-slot cell is classified"
        )
    if used_observed != set(observed_by_sha):
        raise DojoSparseCoverageProofV3Error(
            "coverage input contains an unclassified observed candle"
        )
    if used_no_candle != set(no_candle_by_sha):
        raise DojoSparseCoverageProofV3Error(
            "coverage input contains an unclassified no-candle receipt"
        )
    if set(observed_by_key) & set(no_candle_by_key):
        raise DojoSparseCoverageProofV3Error(
            "pair-slot cannot be both observed and no-candle"
        )
    if set(observed_by_key) | set(no_candle_by_key) != expected_keys:
        raise DojoSparseCoverageProofV3Error(
            "pair-slot evidence is not a complete exact partition"
        )

    normalized_input = {
        "contract": COVERAGE_INPUT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "calendar_artifact_sha256": calendar_sha,
        "provider": calendar["provider"],
        "granularity": GRANULARITY,
        "from_epoch": calendar["from_epoch"],
        "to_epoch": calendar["to_epoch"],
        "feed_pairs": list(pairs),
        "source_producer_id": producer,
        "independent_verifier_id": verifier,
        "observed_candles": observed_rows,
        "no_candle_receipts": no_candle_rows,
        "classifications": classifications,
    }
    if not _canonical_equal(coverage, normalized_input):
        raise DojoSparseCoverageProofV3Error(
            "coverage input differs from its canonical validated form"
        )
    observed_count = len(used_observed)
    no_candle_count = len(used_no_candle)
    body = {
        "contract": COVERAGE_PROOF_RECEIPT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "calendar_artifact_sha256": calendar_sha,
        "coverage_input_sha256": canonical_sha256(normalized_input),
        "provider": calendar["provider"],
        "granularity": GRANULARITY,
        "from_epoch": calendar["from_epoch"],
        "to_epoch": calendar["to_epoch"],
        "feed_pairs": list(pairs),
        "grid_slot_count": calendar["grid_slot_count"],
        "expected_open_slot_count": len(open_epochs),
        "verified_closed_slot_count": calendar["verified_closed_slot_count"],
        "expected_pair_slot_count": expected_cell_count,
        "classified_pair_slot_count": expected_cell_count,
        "observed_pair_slot_count": observed_count,
        "legitimate_no_candle_pair_slot_count": no_candle_count,
        "unclassified_pair_slot_count": 0,
        "duplicate_pair_slot_count": 0,
        "out_of_range_pair_slot_count": 0,
        "observed_candle_mismatch_count": 0,
        "no_candle_receipt_mismatch_count": 0,
        "observed_candles_sha256": canonical_sha256(observed_rows),
        "no_candle_receipts_sha256": canonical_sha256(no_candle_rows),
        "classifications_sha256": canonical_sha256(classifications),
        "complete_pair_slot_classification": True,
        "proof_classification": "CANDIDATE_ONLY_UNAUTHENTICATED_SOURCE_REFERENCES",
        "sparse_calendar_coverage_proved": False,
        "proof_eligible": False,
        "authority": _json_copy(_AUTHORITY),
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def verify_coverage_proof_receipt(
    receipt: Mapping[str, Any],
    *,
    calendar_artifact: Mapping[str, Any],
    coverage_input: Mapping[str, Any],
) -> dict[str, Any]:
    row = _exact_mapping(receipt, _PROOF_RECEIPT_KEYS, field="coverage proof receipt")
    if row["contract"] != COVERAGE_PROOF_RECEIPT_CONTRACT or not _is_exact_integer(
        row["schema_version"], SCHEMA_VERSION
    ):
        raise DojoSparseCoverageProofV3Error("coverage proof receipt identity differs")
    expected = build_coverage_proof_receipt(
        calendar_artifact=calendar_artifact,
        coverage_input=coverage_input,
    )
    if not _canonical_equal(row, expected):
        raise DojoSparseCoverageProofV3Error(
            "coverage proof receipt differs from independent reconstruction"
        )
    return expected


def read_bounded_json_artifact(path: Path, *, field: str) -> Any:
    """Read one stable regular JSON file and reject duplicate object keys."""

    target = Path(path)
    try:
        before = target.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoSparseCoverageProofV3Error(f"{field} is unavailable") from exc
    if (
        target.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or not 0 < before.st_size <= MAX_JSON_BYTES
    ):
        raise DojoSparseCoverageProofV3Error(
            f"{field} must be a bounded nonempty regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(target, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            raw = handle.read(MAX_JSON_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise DojoSparseCoverageProofV3Error(f"{field} could not be read") from exc
    if (
        opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
        or opened.st_size != before.st_size
        or opened.st_mtime_ns != before.st_mtime_ns
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
        or len(raw) != opened.st_size
        or len(raw) > MAX_JSON_BYTES
    ):
        raise DojoSparseCoverageProofV3Error(f"{field} changed while being read")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoSparseCoverageProofV3Error(f"{field} is not strict JSON") from exc


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    """Crash-recoverably publish canonical JSON without replacing ``path``.

    The deterministic pending pathname remains as a hard-link durability anchor.
    It consumes one directory entry, not a second payload allocation.  POSIX has
    no portable unlink-if-inode-matches primitive, so deleting that pathname
    would reintroduce a final check-to-unlink replacement race.
    """

    target = Path(path)
    if not target.name or target.name in {".", ".."}:
        raise DojoSparseCoverageProofV3Error("output path is invalid")
    try:
        parent = target.parent.resolve(strict=True)
    except OSError as exc:
        raise DojoSparseCoverageProofV3Error(
            "output parent directory is unavailable"
        ) from exc
    if not parent.is_dir():
        raise DojoSparseCoverageProofV3Error(
            "output parent must be an existing directory"
        )
    resolved_target = parent / target.name
    payload = canonical_json_bytes(value) + b"\n"
    if len(payload) > MAX_JSON_BYTES:
        raise DojoSparseCoverageProofV3Error("output exceeds its byte bound")
    pending_identity_material = (
        target.name.encode("utf-8") + b"\0" + hashlib.sha256(payload).digest()
    )
    pending_name = (
        ".qr-coverage-"
        + hashlib.sha256(pending_identity_material).hexdigest()
        + ".pending"
    )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_descriptor: int | None = None
    pending_descriptor: int | None = None
    try:
        directory_descriptor = os.open(parent, directory_flags)
        directory_identity = os.fstat(directory_descriptor)
        existing_final = _read_dirfd_entry(
            directory_descriptor,
            target.name,
            maximum_bytes=len(payload),
        )
        if existing_final is not None:
            if existing_final[1] != payload:
                raise DojoSparseCoverageProofV3Error(
                    "write-once output already exists with different bytes"
                )
            _verify_matching_pending_anchor(
                directory_descriptor,
                pending_name,
                payload=payload,
                final_identity=existing_final[0],
            )
            _verify_published_entry(
                directory_descriptor,
                target.name,
                payload=payload,
                expected_identity=existing_final[0],
            )
            os.fsync(directory_descriptor)
            _verify_matching_pending_anchor(
                directory_descriptor,
                pending_name,
                payload=payload,
                final_identity=existing_final[0],
            )
            _verify_published_entry(
                directory_descriptor,
                target.name,
                payload=payload,
                expected_identity=existing_final[0],
            )
            _verify_directory_identity(parent, directory_identity)
            return resolved_target

        pending_flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        pending_descriptor = os.open(
            pending_name,
            pending_flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        fcntl.flock(pending_descriptor, fcntl.LOCK_EX)
        pending_stat = os.fstat(pending_descriptor)
        if not stat.S_ISREG(pending_stat.st_mode):
            raise DojoSparseCoverageProofV3Error(
                "coverage diagnostic pending path is not a regular file"
            )
        pending_bytes = _read_locked_descriptor(
            pending_descriptor,
            maximum_bytes=len(payload),
        )
        if pending_bytes != payload:
            os.ftruncate(pending_descriptor, 0)
            os.lseek(pending_descriptor, 0, os.SEEK_SET)
            _write_all(pending_descriptor, payload)
        os.fsync(pending_descriptor)
        pending_identity = _verify_locked_pending(
            pending_descriptor,
            payload=payload,
        )

        try:
            os.link(
                pending_name,
                target.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            pass
        published = _read_dirfd_entry(
            directory_descriptor,
            target.name,
            maximum_bytes=len(payload),
        )
        if published is None or published[1] != payload:
            raise DojoSparseCoverageProofV3Error(
                "atomic no-replace publication conflicts with another output"
            )
        if published[0] != pending_identity:
            raise DojoSparseCoverageProofV3Error(
                "published output is not the sealed pending inode"
            )
        os.fsync(directory_descriptor)
        _verify_published_entry(
            directory_descriptor,
            target.name,
            payload=payload,
            expected_identity=pending_identity,
        )
        _verify_pending_anchor_path_identity(
            directory_descriptor,
            pending_name,
            expected_identity=pending_identity,
        )
        _verify_directory_identity(parent, directory_identity)
        return resolved_target
    finally:
        if pending_descriptor is not None:
            os.close(pending_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _read_locked_descriptor(descriptor: int, *, maximum_bytes: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    before = os.fstat(descriptor)
    chunks: list[bytes] = []
    remaining = maximum_bytes + 1
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    after = os.fstat(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(data) != before.st_size
        or len(data) > maximum_bytes
    ):
        raise DojoSparseCoverageProofV3Error(
            "coverage diagnostic pending bytes changed while read"
        )
    return data


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(payload):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("short coverage diagnostic pending write")
        written += count


def _verify_locked_pending(descriptor: int, *, payload: bytes) -> tuple[int, int]:
    data = _read_locked_descriptor(descriptor, maximum_bytes=len(payload))
    current = os.fstat(descriptor)
    if data != payload or current.st_size != len(payload):
        raise DojoSparseCoverageProofV3Error(
            "coverage diagnostic pending payload differs"
        )
    return current.st_dev, current.st_ino


def _read_dirfd_entry(
    directory_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
) -> tuple[tuple[int, int], bytes] | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except FileNotFoundError:
        return None
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise DojoSparseCoverageProofV3Error(
                "coverage diagnostic output path is not a regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(maximum_bytes + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        opened.st_dev != after.st_dev
        or opened.st_ino != after.st_ino
        or opened.st_size != after.st_size
        or opened.st_mtime_ns != after.st_mtime_ns
        or len(data) != opened.st_size
        or len(data) > maximum_bytes
    ):
        raise DojoSparseCoverageProofV3Error(
            "coverage diagnostic output changed while read"
        )
    return (opened.st_dev, opened.st_ino), data


def _verify_published_entry(
    directory_descriptor: int,
    name: str,
    *,
    payload: bytes,
    expected_identity: tuple[int, int],
) -> None:
    current = _read_dirfd_entry(
        directory_descriptor,
        name,
        maximum_bytes=len(payload),
    )
    if current is None or current[0] != expected_identity or current[1] != payload:
        raise DojoSparseCoverageProofV3Error(
            "coverage diagnostic final entry differs after publication"
        )
    final_stat = os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
    if (final_stat.st_dev, final_stat.st_ino) != expected_identity:
        raise DojoSparseCoverageProofV3Error(
            "coverage diagnostic final pathname identity differs"
        )


def _verify_matching_pending_anchor(
    directory_descriptor: int,
    pending_name: str,
    *,
    payload: bytes,
    final_identity: tuple[int, int],
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(
            pending_name,
            flags,
            dir_fd=directory_descriptor,
        )
    except FileNotFoundError:
        raise DojoSparseCoverageProofV3Error(
            "coverage diagnostic durable pending anchor is missing"
        )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        pending_payload = _read_locked_descriptor(
            descriptor,
            maximum_bytes=len(payload),
        )
        pending_stat = os.fstat(descriptor)
        pending_identity = (pending_stat.st_dev, pending_stat.st_ino)
        pending_path = os.stat(
            pending_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(pending_stat.st_mode)
            or not stat.S_ISREG(pending_path.st_mode)
            or pending_payload != payload
            or pending_identity != final_identity
            or (pending_path.st_dev, pending_path.st_ino) != final_identity
        ):
            raise DojoSparseCoverageProofV3Error(
                "existing output has a conflicting deterministic pending anchor"
            )
    finally:
        os.close(descriptor)


def _verify_pending_anchor_path_identity(
    directory_descriptor: int,
    pending_name: str,
    *,
    expected_identity: tuple[int, int],
) -> None:
    try:
        pending_path = os.stat(
            pending_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise DojoSparseCoverageProofV3Error(
            "coverage diagnostic durable pending anchor is missing"
        ) from exc
    if (
        not stat.S_ISREG(pending_path.st_mode)
        or (pending_path.st_dev, pending_path.st_ino) != expected_identity
    ):
        raise DojoSparseCoverageProofV3Error(
            "coverage diagnostic durable pending anchor identity differs"
        )


def _verify_directory_identity(parent: Path, expected: os.stat_result) -> None:
    current = os.stat(parent, follow_symlinks=False)
    if current.st_dev != expected.st_dev or current.st_ino != expected.st_ino:
        raise DojoSparseCoverageProofV3Error(
            "output parent directory identity changed during publication"
        )


def _reject_duplicate_object_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise DojoSparseCoverageProofV3Error(
                f"JSON object contains duplicate key: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise DojoSparseCoverageProofV3Error(
        f"JSON artifact contains non-finite constant: {value}"
    )


def _require_json_shape(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DojoSparseCoverageProofV3Error(
                    "canonical JSON object keys must be strings"
                )
            _require_json_shape(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _require_json_shape(item)
        return
    if value is None or value.__class__ in {bool, int, float, str}:
        if value.__class__ is float and not math.isfinite(value):
            raise DojoSparseCoverageProofV3Error("canonical JSON number must be finite")
        return
    raise DojoSparseCoverageProofV3Error("value is not strict JSON")


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _canonical_equal(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _exact_top_mapping(
    value: Any, keys: frozenset[str], *, field: str
) -> dict[str, Any]:
    """Check a potentially large top-level schema without a deep copy."""

    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoSparseCoverageProofV3Error(f"{field} must be an object")
    if set(value) != set(keys):
        raise DojoSparseCoverageProofV3Error(f"{field} schema is not exact")
    return dict(value)


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoSparseCoverageProofV3Error(f"{field} must be an object")
    copied = _json_copy(dict(value))
    if not isinstance(copied, dict):
        raise DojoSparseCoverageProofV3Error(f"{field} must be an object")
    return copied


def _exact_mapping(value: Any, keys: frozenset[str], *, field: str) -> dict[str, Any]:
    row = _mapping(value, field=field)
    if set(row) != set(keys):
        raise DojoSparseCoverageProofV3Error(f"{field} schema is not exact")
    return row


def _sequence(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DojoSparseCoverageProofV3Error(f"{field} must be an array")
    return list(value)


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise DojoSparseCoverageProofV3Error(f"{field} is not a bounded identifier")
    return value


def _pair(value: Any) -> str:
    if not isinstance(value, str) or _PAIR_RE.fullmatch(value) is None:
        raise DojoSparseCoverageProofV3Error("pair identity is invalid")
    if value[:3] == value[4:]:
        raise DojoSparseCoverageProofV3Error("pair currencies must differ")
    return value


def _feed_pairs(value: Any) -> tuple[str, ...]:
    rows = _sequence(value, field="feed_pairs")
    if not 1 <= len(rows) <= MAX_PAIRS:
        raise DojoSparseCoverageProofV3Error("feed pair count is outside its bound")
    pairs = tuple(_pair(row) for row in rows)
    if len(set(pairs)) != len(pairs):
        raise DojoSparseCoverageProofV3Error("feed_pairs contains a duplicate")
    return pairs


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoSparseCoverageProofV3Error(f"{field} must be a lowercase SHA-256")
    return value


def _aligned_epoch(value: Any, *, field: str) -> int:
    if value.__class__ is not int or value < 0 or value % CADENCE_SECONDS:
        raise DojoSparseCoverageProofV3Error(
            f"{field} must be a nonnegative M5-aligned integer"
        )
    return value


def _closure_reason(value: Any) -> str:
    if not isinstance(value, str) or value not in CLOSURE_REASONS:
        raise DojoSparseCoverageProofV3Error(
            "closure reason is not a provider-published closure class"
        )
    return str(value)


def _no_candle_reason(value: Any) -> str:
    if not isinstance(value, str) or value != NO_CANDLE_REASON:
        raise DojoSparseCoverageProofV3Error(
            "no-candle reason must name a complete provider response omission"
        )
    return str(value)


def _require_independent_identities(
    producer: str, verifier: str, *, field: str
) -> None:
    if producer == verifier:
        raise DojoSparseCoverageProofV3Error(
            f"{field} producer and independent verifier must differ"
        )


def _is_exact_integer(value: Any, expected: int) -> bool:
    return value.__class__ is int and value == expected


def _ohlc(value: Any, *, field: str) -> list[float]:
    rows = _sequence(value, field=field)
    if len(rows) != 4:
        raise DojoSparseCoverageProofV3Error(f"{field} must contain O,H,L,C")
    result: list[float] = []
    for item in rows:
        if item.__class__ not in {int, float}:
            raise DojoSparseCoverageProofV3Error(f"{field} values must be numeric")
        number = float(item)
        if not math.isfinite(number) or number <= 0.0:
            raise DojoSparseCoverageProofV3Error(
                f"{field} values must be finite and positive"
            )
        result.append(number)
    open_price, high, low, close = result
    if high < max(open_price, low, close) or low > min(open_price, high, close):
        raise DojoSparseCoverageProofV3Error(f"{field} geometry is invalid")
    return result


__all__ = [
    "CADENCE_SECONDS",
    "CLOSURE_REASONS",
    "COVERAGE_INPUT_CONTRACT",
    "COVERAGE_PROOF_RECEIPT_CONTRACT",
    "DojoSparseCoverageProofV3Error",
    "EXPECTED_OPEN",
    "GRANULARITY",
    "LEGITIMATE_NO_CANDLE",
    "MARKET_CALENDAR_ARTIFACT_CONTRACT",
    "MARKET_CALENDAR_SPEC_CONTRACT",
    "NO_CANDLE_REASON",
    "OBSERVED",
    "SCHEMA_VERSION",
    "VERIFIED_CLOSED",
    "build_coverage_proof_receipt",
    "build_market_closure_receipt",
    "build_no_candle_receipt",
    "build_observed_candle",
    "build_sealed_market_calendar_artifact",
    "canonical_json_bytes",
    "canonical_sha256",
    "read_bounded_json_artifact",
    "verify_coverage_proof_receipt",
    "verify_market_closure_receipt",
    "verify_no_candle_receipt",
    "verify_observed_candle",
    "verify_sealed_market_calendar_artifact",
    "write_json_exclusive",
]

"""Pure opt-in sparse source-slice V2 for DOJO economic replay.

The existing synchronized source contract remains untouched.  This module is
an independent vertical slice that binds already-authenticated raw
``pair -> epoch -> quote`` maps to a :mod:`dojo_sparse_replay` schedule.  Its
canonical rows contain only quotes actually observed at the union epoch.
Unavailable pairs retain availability and pair-local age metadata, never an
executable synthetic or carried quote.

The builder does not open or authenticate raw files.  It requires an upstream
authentication-receipt digest for every pair, recomputes the normalized quote
map identity from the supplied maps, and binds both identities.  All APIs are
pure and research-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from quant_rabbit.dojo_sparse_replay import (
    QUOTE_POLICY,
    SparseReplaySchedule,
    iter_observed_epoch_batches,
)


PARENT_QUOTE_MAP_BINDING_CONTRACT: Final = (
    "QR_DOJO_AUTHENTICATED_PARENT_QUOTE_MAP_BINDING_V1"
)
SPARSE_SOURCE_SLICE_CONTRACT: Final = "QR_DOJO_SPARSE_SOURCE_SLICE_V2"
SPARSE_SOURCE_CHAIN_ATTESTATION_CONTRACT: Final = (
    "QR_DOJO_SPARSE_SOURCE_CHAIN_ATTESTATION_V2"
)
SCHEMA_VERSION: Final = 2
PARENT_BINDING_SCHEMA_VERSION: Final = 1
GENESIS_SHA256: Final = "0" * 64

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}\Z")
_QUOTE_KEYS: Final = frozenset({"pair", "bid", "ask"})
_PARENT_BINDING_INPUT_KEYS: Final = frozenset(
    {
        "raw_source_id",
        "raw_artifact_sha256",
        "upstream_authentication_receipt_sha256",
    }
)
_PARENT_BINDING_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "pair",
        "granularity",
        "from_epoch",
        "to_epoch",
        "raw_source_id",
        "raw_artifact_sha256",
        "upstream_authentication_receipt_sha256",
        "upstream_raw_authentication_required",
        "raw_bytes_opened_by_this_builder",
        "normalized_quote_map_sha256",
        "observed_row_count",
        "first_observed_epoch",
        "last_observed_epoch",
        "synthetic_quote_count",
        "carry_forward_quote_count",
        "parent_binding_sha256",
    }
)
_RECEIPT_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "stream_id",
        "slice_id",
        "sequence",
        "prior_receipt_sha256",
        "granularity",
        "from_epoch",
        "to_epoch",
        "feed_pairs",
        "quote_policy",
        "schedule_coverage_receipt",
        "schedule_coverage_sha256",
        "availability_mask_sha256",
        "pair_local_quote_age_mask_sha256",
        "calendar_coverage_validated",
        "parent_quote_maps",
        "parent_quote_maps_sha256",
        "source_row_count",
        "observed_quote_count",
        "source_rows_sha256",
        "first_epoch",
        "last_epoch",
        "synthetic_quote_count",
        "carry_forward_quote_count",
        "complete",
        "lineage",
        "authority",
        "receipt_sha256",
    }
)
_AUTHORITY: Final = {
    "research_source_only": True,
    "historical_train_is_forward_proof": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
}


class DojoSparseSourceSliceV2Error(ValueError):
    """Sparse source rows, lineage, or parent identity are invalid."""


@dataclass(frozen=True)
class SparseSourceSliceV2:
    """In-memory canonical rows plus their content-addressed receipt."""

    rows: tuple[Mapping[str, Any], ...]
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class SparseSourceConsumerBatch:
    """Verified fresh-only consumer view at one union epoch."""

    epoch: int
    fresh_observed_quotes: tuple[tuple[str, Mapping[str, Any]], ...]
    fresh_executable_pairs: tuple[str, ...]
    unavailable_pairs: tuple[str, ...]
    pair_local_quote_age_seconds: tuple[tuple[str, int | None], ...]
    synthetic_quote_count: int = 0
    carry_forward_quote_count: int = 0

    def executable_quote(self, pair: str) -> Mapping[str, Any]:
        """Return a fresh quote or fail instead of carrying a stale quote."""

        for candidate, quote in self.fresh_observed_quotes:
            if candidate == pair:
                return quote
        raise DojoSparseSourceSliceV2Error(
            f"pair has no fresh executable quote at epoch {self.epoch}: {pair}"
        )


def canonical_source_sha256(value: Any) -> str:
    """Hash one strict JSON value canonically."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoSparseSourceSliceV2Error(
            "sparse source value is not strict canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def build_parent_quote_map_binding(
    *,
    schedule: SparseReplaySchedule,
    pair: str,
    quote_map: Mapping[int, Mapping[str, Any]],
    authentication: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an externally authenticated raw artifact to its quote-map truth."""

    _schedule(schedule)
    if pair not in schedule.feed_pairs:
        raise DojoSparseSourceSliceV2Error("parent pair is outside the schedule")
    auth = _exact_mapping(
        authentication,
        _PARENT_BINDING_INPUT_KEYS,
        field="parent authentication",
    )
    normalized = _normalized_pair_quote_rows(
        schedule=schedule,
        pair=pair,
        quote_map=quote_map,
    )
    body = {
        "contract": PARENT_QUOTE_MAP_BINDING_CONTRACT,
        "schema_version": PARENT_BINDING_SCHEMA_VERSION,
        "pair": pair,
        "granularity": schedule.granularity,
        "from_epoch": schedule.start_epoch,
        "to_epoch": schedule.end_epoch,
        "raw_source_id": _identifier(auth["raw_source_id"], field="raw_source_id"),
        "raw_artifact_sha256": _sha256(
            auth["raw_artifact_sha256"], field="raw_artifact_sha256"
        ),
        "upstream_authentication_receipt_sha256": _sha256(
            auth["upstream_authentication_receipt_sha256"],
            field="upstream_authentication_receipt_sha256",
        ),
        "upstream_raw_authentication_required": True,
        "raw_bytes_opened_by_this_builder": False,
        "normalized_quote_map_sha256": canonical_source_sha256(normalized),
        "observed_row_count": len(normalized),
        "first_observed_epoch": normalized[0]["epoch"],
        "last_observed_epoch": normalized[-1]["epoch"],
        "synthetic_quote_count": 0,
        "carry_forward_quote_count": 0,
    }
    return {**body, "parent_binding_sha256": canonical_source_sha256(body)}


def validate_parent_quote_map_binding(
    value: Mapping[str, Any],
    *,
    schedule: SparseReplaySchedule,
    pair: str,
    quote_map: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute a parent map binding from the exact supplied quote map."""

    row = _strict_mapping(value, field="parent quote map binding")
    _exact_mapping(row, _PARENT_BINDING_KEYS, field="parent quote map binding")
    expected = build_parent_quote_map_binding(
        schedule=schedule,
        pair=pair,
        quote_map=quote_map,
        authentication={
            "raw_source_id": row.get("raw_source_id"),
            "raw_artifact_sha256": row.get("raw_artifact_sha256"),
            "upstream_authentication_receipt_sha256": row.get(
                "upstream_authentication_receipt_sha256"
            ),
        },
    )
    if row != expected:
        raise DojoSparseSourceSliceV2Error(
            "parent quote map binding differs from supplied raw quote identity"
        )
    return row


def build_sparse_source_slice_v2(
    *,
    stream_id: str,
    slice_id: str,
    sequence: int,
    prior_receipt_sha256: str,
    schedule: SparseReplaySchedule,
    pair_quote_maps: Mapping[str, Mapping[int, Mapping[str, Any]]],
    parent_bindings: Mapping[str, Mapping[str, Any]],
) -> SparseSourceSliceV2:
    """Build canonical sparse rows and a complete V2 source receipt."""

    _schedule(schedule)
    normalized_sequence = _integer(sequence, field="sequence", minimum=0)
    prior_sha = _sha256(prior_receipt_sha256, field="prior_receipt_sha256")
    if (normalized_sequence == 0) != (prior_sha == GENESIS_SHA256):
        raise DojoSparseSourceSliceV2Error(
            "source lineage genesis/prior receipt relation is invalid"
        )
    if set(pair_quote_maps) != set(schedule.feed_pairs):
        raise DojoSparseSourceSliceV2Error(
            "quote maps must exactly equal the scheduled feed pairs"
        )
    if set(parent_bindings) != set(schedule.feed_pairs):
        raise DojoSparseSourceSliceV2Error(
            "parent bindings must exactly equal the scheduled feed pairs"
        )

    normalized_parent_bindings: list[dict[str, Any]] = []
    parent_quote_sha_by_key: dict[tuple[str, int], str] = {}
    for pair in schedule.feed_pairs:
        binding = validate_parent_quote_map_binding(
            parent_bindings[pair],
            schedule=schedule,
            pair=pair,
            quote_map=pair_quote_maps[pair],
        )
        normalized_parent_bindings.append(binding)
        for normalized_quote in _normalized_pair_quote_rows(
            schedule=schedule,
            pair=pair,
            quote_map=pair_quote_maps[pair],
        ):
            parent_quote_sha_by_key[(pair, normalized_quote["epoch"])] = (
                canonical_source_sha256(normalized_quote)
            )

    rows: list[dict[str, Any]] = []
    for ordinal, batch in enumerate(
        iter_observed_epoch_batches(pair_quote_maps, schedule=schedule)
    ):
        quotes: list[dict[str, Any]] = []
        for pair, raw_quote in batch.observations:
            normalized = _normalized_quote(
                raw_quote,
                pair=pair,
                epoch=batch.epoch,
                granularity=schedule.granularity,
            )
            quotes.append(
                {
                    "pair": pair,
                    "bid": normalized["bid"],
                    "ask": normalized["ask"],
                    "parent_quote_sha256": parent_quote_sha_by_key[(pair, batch.epoch)],
                }
            )
        observed_pairs = list(batch.observed_pairs)
        unavailable_pairs = list(batch.unavailable_pairs)
        age_map = dict(batch.pair_local_quote_age_seconds)
        row_body = {
            "ordinal": ordinal,
            "epoch": batch.epoch,
            "granularity": schedule.granularity,
            "complete": True,
            "observed_pairs": observed_pairs,
            "fresh_executable_pairs": observed_pairs,
            "unavailable_pairs": unavailable_pairs,
            "pair_local_quote_age_seconds": age_map,
            "availability_cell_sha256": canonical_source_sha256(
                {"epoch": batch.epoch, "batch_pairs": observed_pairs}
            ),
            "pair_local_quote_age_cell_sha256": canonical_source_sha256(
                {
                    "epoch": batch.epoch,
                    "pair_local_quote_age_seconds": age_map,
                }
            ),
            "quotes": quotes,
            "synthetic_quote_count": 0,
            "carry_forward_quote_count": 0,
        }
        rows.append(
            {**row_body, "source_row_sha256": canonical_source_sha256(row_body)}
        )

    coverage = schedule.coverage_receipt()
    observed_quote_count = sum(len(row["quotes"]) for row in rows)
    parent_summary = [
        {
            "pair": binding["pair"],
            "parent_binding_sha256": binding["parent_binding_sha256"],
            "raw_artifact_sha256": binding["raw_artifact_sha256"],
            "upstream_authentication_receipt_sha256": binding[
                "upstream_authentication_receipt_sha256"
            ],
            "normalized_quote_map_sha256": binding["normalized_quote_map_sha256"],
            "observed_row_count": binding["observed_row_count"],
        }
        for binding in normalized_parent_bindings
    ]
    receipt_body = {
        "contract": SPARSE_SOURCE_SLICE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "stream_id": _identifier(stream_id, field="stream_id"),
        "slice_id": _identifier(slice_id, field="slice_id"),
        "sequence": normalized_sequence,
        "prior_receipt_sha256": prior_sha,
        "granularity": schedule.granularity,
        "from_epoch": schedule.start_epoch,
        "to_epoch": schedule.end_epoch,
        "feed_pairs": list(schedule.feed_pairs),
        "quote_policy": QUOTE_POLICY,
        "schedule_coverage_receipt": coverage,
        "schedule_coverage_sha256": coverage["coverage_sha256"],
        "availability_mask_sha256": coverage["availability_mask_sha256"],
        "pair_local_quote_age_mask_sha256": coverage[
            "pair_local_quote_age_mask_sha256"
        ],
        "calendar_coverage_validated": True,
        "parent_quote_maps": parent_summary,
        "parent_quote_maps_sha256": canonical_source_sha256(parent_summary),
        "source_row_count": len(rows),
        "observed_quote_count": observed_quote_count,
        "source_rows_sha256": canonical_source_sha256(rows),
        "first_epoch": rows[0]["epoch"],
        "last_epoch": rows[-1]["epoch"],
        "synthetic_quote_count": 0,
        "carry_forward_quote_count": 0,
        "complete": True,
        "lineage": {
            "append_only_sequence_intent": True,
            "prior_receipt_bound": True,
            "caller_supplied_chain_fork_check_required": True,
            "external_monotonic_anchor_configured": False,
            "global_fork_absence_proven": False,
        },
        "authority": _strict_json_copy(_AUTHORITY),
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": canonical_source_sha256(receipt_body),
    }
    return SparseSourceSliceV2(
        rows=tuple(_strict_mapping(row, field="source row") for row in rows),
        receipt=_strict_mapping(receipt, field="source receipt"),
    )


def verify_sparse_source_slice_v2(
    source_slice: SparseSourceSliceV2,
    *,
    expected_stream_id: str,
    expected_slice_id: str,
    expected_sequence: int,
    expected_prior_receipt_sha256: str,
    schedule: SparseReplaySchedule,
    pair_quote_maps: Mapping[str, Mapping[int, Mapping[str, Any]]],
    parent_bindings: Mapping[str, Mapping[str, Any]],
) -> SparseSourceSliceV2:
    """Rebuild the complete slice and reject row, receipt, or lineage drift."""

    if not isinstance(source_slice, SparseSourceSliceV2):
        raise DojoSparseSourceSliceV2Error("source_slice must be a SparseSourceSliceV2")
    rows = tuple(_strict_mapping(row, field="source row") for row in source_slice.rows)
    receipt = _strict_mapping(source_slice.receipt, field="source receipt")
    _exact_mapping(receipt, _RECEIPT_KEYS, field="source receipt")
    expected = build_sparse_source_slice_v2(
        stream_id=expected_stream_id,
        slice_id=expected_slice_id,
        sequence=expected_sequence,
        prior_receipt_sha256=expected_prior_receipt_sha256,
        schedule=schedule,
        pair_quote_maps=pair_quote_maps,
        parent_bindings=parent_bindings,
    )
    if rows != expected.rows or receipt != expected.receipt:
        raise DojoSparseSourceSliceV2Error(
            "sparse source slice differs from authenticated parent truth"
        )
    return expected


def consume_sparse_source_slice_v2(
    source_slice: SparseSourceSliceV2,
    *,
    expected_stream_id: str,
    expected_slice_id: str,
    expected_sequence: int,
    expected_prior_receipt_sha256: str,
    schedule: SparseReplaySchedule,
    pair_quote_maps: Mapping[str, Mapping[int, Mapping[str, Any]]],
    parent_bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[SparseSourceConsumerBatch, ...]:
    """Verify then expose only fresh observed quotes as executable inputs."""

    verified = verify_sparse_source_slice_v2(
        source_slice,
        expected_stream_id=expected_stream_id,
        expected_slice_id=expected_slice_id,
        expected_sequence=expected_sequence,
        expected_prior_receipt_sha256=expected_prior_receipt_sha256,
        schedule=schedule,
        pair_quote_maps=pair_quote_maps,
        parent_bindings=parent_bindings,
    )
    batches: list[SparseSourceConsumerBatch] = []
    for row in verified.rows:
        observed_pairs = tuple(row["observed_pairs"])
        quote_pairs = tuple(quote["pair"] for quote in row["quotes"])
        if quote_pairs != observed_pairs or tuple(row["fresh_executable_pairs"]) != (
            observed_pairs
        ):
            raise DojoSparseSourceSliceV2Error(
                "source row executable pairs differ from fresh observations"
            )
        batches.append(
            SparseSourceConsumerBatch(
                epoch=row["epoch"],
                fresh_observed_quotes=tuple(
                    (quote["pair"], _strict_mapping(quote, field="quote"))
                    for quote in row["quotes"]
                ),
                fresh_executable_pairs=observed_pairs,
                unavailable_pairs=tuple(row["unavailable_pairs"]),
                pair_local_quote_age_seconds=tuple(
                    (pair, row["pair_local_quote_age_seconds"][pair])
                    for pair in schedule.feed_pairs
                ),
            )
        )
    return tuple(batches)


def verify_sparse_source_receipt_chain_v2(
    receipts: Sequence[Mapping[str, Any]],
    *,
    expected_stream_id: str,
    expected_start_sequence: int = 0,
    expected_prior_receipt_sha256: str = GENESIS_SHA256,
) -> dict[str, Any]:
    """Reject gaps, overlaps, and forks in the complete caller-supplied chain."""

    if isinstance(receipts, (str, bytes, bytearray)) or not receipts:
        raise DojoSparseSourceSliceV2Error("source receipt chain is empty")
    start_sequence = _integer(
        expected_start_sequence, field="expected_start_sequence", minimum=0
    )
    expected_prior = _sha256(
        expected_prior_receipt_sha256,
        field="expected_prior_receipt_sha256",
    )
    stream_id = _identifier(expected_stream_id, field="expected_stream_id")
    normalized: list[dict[str, Any]] = []
    seen_receipts: set[str] = set()
    previous_to: int | None = None
    for offset, raw in enumerate(receipts):
        row = _strict_mapping(raw, field="source receipt")
        _exact_mapping(row, _RECEIPT_KEYS, field="source receipt")
        body = {key: item for key, item in row.items() if key != "receipt_sha256"}
        if row["receipt_sha256"] != canonical_source_sha256(body):
            raise DojoSparseSourceSliceV2Error("source receipt self-hash is invalid")
        expected_sequence = start_sequence + offset
        if (
            row["stream_id"] != stream_id
            or row["sequence"] != expected_sequence
            or row["prior_receipt_sha256"] != expected_prior
        ):
            raise DojoSparseSourceSliceV2Error(
                "source receipt chain has a gap, alternate prior, or fork"
            )
        if row["receipt_sha256"] in seen_receipts:
            raise DojoSparseSourceSliceV2Error("source receipt is duplicated")
        if previous_to is not None and row["from_epoch"] != previous_to:
            raise DojoSparseSourceSliceV2Error(
                "source receipt windows have a gap or overlap"
            )
        if (
            row["complete"] is not True
            or row["synthetic_quote_count"] != 0
            or row["carry_forward_quote_count"] != 0
            or row["authority"] != _AUTHORITY
        ):
            raise DojoSparseSourceSliceV2Error(
                "source receipt chain contains an ineligible receipt"
            )
        seen_receipts.add(row["receipt_sha256"])
        normalized.append(row)
        expected_prior = row["receipt_sha256"]
        previous_to = row["to_epoch"]
    body = {
        "contract": SPARSE_SOURCE_CHAIN_ATTESTATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "stream_id": stream_id,
        "start_sequence": start_sequence,
        "terminal_sequence": start_sequence + len(normalized) - 1,
        "receipt_count": len(normalized),
        "receipt_sha256s": [row["receipt_sha256"] for row in normalized],
        "terminal_receipt_sha256": normalized[-1]["receipt_sha256"],
        "supplied_chain_gap_free": True,
        "supplied_chain_fork_free": True,
        "external_monotonic_anchor_configured": False,
        "global_fork_absence_proven": False,
        "authority": _strict_json_copy(_AUTHORITY),
    }
    return {**body, "attestation_sha256": canonical_source_sha256(body)}


def _normalized_pair_quote_rows(
    *,
    schedule: SparseReplaySchedule,
    pair: str,
    quote_map: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(quote_map, Mapping):
        raise DojoSparseSourceSliceV2Error("parent quote map must be an object")
    expected_epochs = schedule.epochs_for_pair(pair)
    if set(quote_map) != set(expected_epochs):
        raise DojoSparseSourceSliceV2Error(
            "parent quote map differs from scheduled observed epochs"
        )
    return [
        _normalized_quote(
            quote_map[epoch],
            pair=pair,
            epoch=epoch,
            granularity=schedule.granularity,
        )
        for epoch in expected_epochs
    ]


def _normalized_quote(
    value: Any, *, pair: str, epoch: int, granularity: str
) -> dict[str, Any]:
    row = _exact_mapping(value, _QUOTE_KEYS, field="parent quote")
    if row["pair"] != pair:
        raise DojoSparseSourceSliceV2Error("parent quote pair identity differs")
    bid = _ohlc(row["bid"], field="bid")
    ask = _ohlc(row["ask"], field="ask")
    if any(ask_value < bid_value for bid_value, ask_value in zip(bid, ask)):
        raise DojoSparseSourceSliceV2Error("parent quote has negative bid/ask spread")
    return {
        "pair": pair,
        "epoch": _integer(epoch, field="epoch", minimum=0),
        "granularity": granularity,
        "bid": bid,
        "ask": ask,
    }


def _ohlc(value: Any, *, field: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 4
    ):
        raise DojoSparseSourceSliceV2Error(f"{field} must contain O,H,L,C")
    prices = [
        _positive_price(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    ]
    open_price, high, low, close = prices
    if high < max(open_price, low, close) or low > min(open_price, high, close):
        raise DojoSparseSourceSliceV2Error(f"{field} OHLC geometry is invalid")
    return prices


def _positive_price(value: Any, *, field: str) -> float:
    if value.__class__ not in {int, float}:
        raise DojoSparseSourceSliceV2Error(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise DojoSparseSourceSliceV2Error(f"{field} must be finite and positive")
    return result


def _schedule(value: Any) -> SparseReplaySchedule:
    if not isinstance(value, SparseReplaySchedule):
        raise DojoSparseSourceSliceV2Error(
            "schedule must be a validated SparseReplaySchedule"
        )
    coverage = value.coverage_receipt()
    if (
        coverage["synthetic_quote_count"] != 0
        or coverage["carry_forward_quote_count"] != 0
        or coverage["quote_policy"] != QUOTE_POLICY
        or coverage["granularity"] not in {"M1", "M5"}
    ):
        raise DojoSparseSourceSliceV2Error("schedule is not sparse-source eligible")
    return value


def _strict_json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise DojoSparseSourceSliceV2Error(
            "sparse source value is not strict JSON"
        ) from exc


def _strict_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoSparseSourceSliceV2Error(f"{field} must be an object")
    result = _strict_json_copy(dict(value))
    if not isinstance(result, dict):
        raise DojoSparseSourceSliceV2Error(f"{field} must be an object")
    return result


def _exact_mapping(
    value: Any, keys: set[str] | frozenset[str], *, field: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoSparseSourceSliceV2Error(f"{field} must be an object")
    row = dict(value)
    if set(row) != set(keys):
        raise DojoSparseSourceSliceV2Error(f"{field} schema is not exact")
    return row


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise DojoSparseSourceSliceV2Error(f"{field} is not a bounded identifier")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DojoSparseSourceSliceV2Error(f"{field} is not a SHA-256 digest")
    return value


def _integer(value: Any, *, field: str, minimum: int) -> int:
    if value.__class__ is not int or value < minimum:
        raise DojoSparseSourceSliceV2Error(f"{field} must be an integer >= {minimum}")
    return value


__all__ = [
    "GENESIS_SHA256",
    "PARENT_QUOTE_MAP_BINDING_CONTRACT",
    "SCHEMA_VERSION",
    "SPARSE_SOURCE_CHAIN_ATTESTATION_CONTRACT",
    "SPARSE_SOURCE_SLICE_CONTRACT",
    "DojoSparseSourceSliceV2Error",
    "SparseSourceConsumerBatch",
    "SparseSourceSliceV2",
    "build_parent_quote_map_binding",
    "build_sparse_source_slice_v2",
    "canonical_source_sha256",
    "consume_sparse_source_slice_v2",
    "validate_parent_quote_map_binding",
    "verify_sparse_source_receipt_chain_v2",
    "verify_sparse_source_slice_v2",
]

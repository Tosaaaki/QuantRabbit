"""Immutable point-in-time evidence packets for future paper AI inventory.

This module is deliberately isolated from broker, model, network, and room
runtime code.  Its launch-safe builder reads strict content-addressed source
files from the dedicated future ``paper-ai-inventory`` research root, seals
them at one immutable cutoff, and persists a content-addressed packet.  The
older caller-assembled builder remains explicitly unsafe/test-only and must not
be wired into a room launch or decision path.

The packet is evidence, not an action or authority:

* ``paper_only`` is always true;
* ``order_authority`` is always ``NONE``;
* ``live_permission`` is always false;
* every timestamp is at or before the cutoff;
* the cutoff and writer clock must both be inside the DST-aware FX week; and
* verification returns only the packet's strict allowlist, never a source
  file path or access handle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_ai_source_capture import (
    AiSourceCaptureError,
    verify_ai_source_capture_receipt,
)
from quant_rabbit.dojo_replay_lifecycle import (
    LAUNCH_PREFLIGHT_CONTRACT as PAPER_AI_LAUNCH_PREFLIGHT_CONTRACT,
    DojoReplayLifecycleError,
    verify_paper_ai_inventory_launch_preflight,
)


DOJO_AI_EVIDENCE_PACKET_CONTRACT = "QR_DOJO_AI_INVENTORY_EVIDENCE_PACKET_V2"
VIRTUAL_BROKER_LEDGER_SNAPSHOT_CONTRACT = "QR_VIRTUAL_BROKER_LEDGER_SNAPSHOT_V1"
DEDICATED_EVIDENCE_ROOT = Path(
    "research/data/dojo_paper_ai_inventory_v1/evidence_packets"
)
DEDICATED_CANONICAL_SOURCE_ROOT = Path(
    "research/data/dojo_paper_ai_inventory_v1/canonical_sources"
)
LOW_LEVEL_BUILDER_LAUNCH_SAFE = False

MAX_PACKET_BYTES = 2 * 1024 * 1024
MAX_CANONICAL_SOURCE_BYTES = 4 * 1024 * 1024
MAX_PACKET_BUILD_LAG_SECONDS = 300
MAX_DYNAMIC_BINDING_AGE_SECONDS = 300
MAX_QUOTE_AGE_SECONDS = 180
MAX_CANDLE_AGE_SECONDS = 24 * 60 * 60
MAX_CANDLE_ROWS = 2_000
MAX_SOURCE_ROWS_PER_KIND = 64
MAX_TEXT_CHARS = 2_000
MAX_URL_CHARS = 2_048

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_HEAD_RE = re.compile(r"^[0-9a-f]{40}$")
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")
_GRANULARITY_RE = re.compile(
    r"^(?:S5|S10|S15|S30|M1|M2|M4|M5|M10|M15|M30|H1|H2|H3|H4|H6|H8|H12|D)$"
)

_INPUT_KEYS = frozenset(
    {
        "contract",
        "cutoff_utc",
        "bindings",
        "position",
        "entry_signal",
        "quote",
        "candles",
        "news_items",
        "calendar_items",
        "cross_asset_items",
        "dynamic_binding_max_age_seconds",
        "paper_only",
        "order_authority",
        "live_permission",
    }
)
_SEALED_KEYS = _INPUT_KEYS | frozenset(
    {
        "sealed_at_utc",
        "seal_lag_nanoseconds",
        "source_row_counts",
        "packet_sha256",
    }
)
_BINDING_KEYS = frozenset(
    {
        "launch_preflight_token_sha256",
        "git_head",
        "git_branch",
        "canonical_source_root",
        "experiment_id",
        "room_id",
        "session_contract_sha256",
        "candidate_id",
        "candidate_sha256",
        "spec_id",
        "spec_sha256",
        "policy_id",
        "policy_sha256",
        "paper_eligible_tip_sha256",
        "ledger_sha256",
        "ledger_observed_at_utc",
        "state_sha256",
        "state_observed_at_utc",
        "snapshot_sha256",
        "snapshot_observed_at_utc",
    }
)
_VIRTUAL_BROKER_LEDGER_SNAPSHOT_KEYS = frozenset(
    {
        "contract",
        "room_id",
        "observed_at_utc",
        "terminal_sha256",
        "rows",
    }
)
_VIRTUAL_BROKER_LEDGER_ROW_KEYS = frozenset(
    {"ts_utc", "event", "payload", "prev_sha", "sha"}
)
_PAPER_AI_LAUNCH_PREFLIGHT_KEYS = frozenset(
    {
        "contract",
        "candidate_id",
        "adapter_id",
        "model_id",
        "config_sha256",
        "producer_id",
        "spec_sha256",
        "policy_sha256",
        "experiment_id",
        "room_id",
        "paper_eligible_event_sha256",
        "candidate_lifecycle_ledger_tip_sha256",
        "append_claim_sha256",
        "job_manifest_sha256",
        "job_owner_sha256",
        "proof_artifact_sha256",
        "proof_artifact_bytes_sha256",
        "proof_manifest_sha256",
        "replay_worker_receipt_sha256",
        "source_manifest_sha256s",
        "source_capture_manifest_sha256",
        "future_registry_sha256",
        "future_window",
        "git_head",
        "git_head_sha256",
        "issued_at_utc",
        "paper_only",
        "order_authority",
        "live_permission",
        "paper_room_launched",
        "launch_preflight_token_sha256",
    }
)
_POSITION_KEYS = frozenset(
    {
        "position_id",
        "pair",
        "side",
        "units",
        "entry_price",
        "opened_at_utc",
        "observed_at_utc",
        "strategy_tag",
        "entry_context_sha256",
        "take_profit",
        "stop_loss",
        "remaining_ceiling_seconds",
        "unrealized_pl_jpy",
        "gross_same_currency_units",
        "net_same_currency_units",
        "margin_used_jpy",
        "capital_locked_jpy",
        "same_direction_position_count",
    }
)
_QUOTE_KEYS = frozenset(
    {
        "pair",
        "bid",
        "ask",
        "timestamp_utc",
        "source_sha256",
        "max_age_seconds",
    }
)
_ENTRY_SIGNAL_KEYS = frozenset(
    {
        "signal_identity_sha256",
        "pair",
        "side",
        "order_type",
        "units",
        "price",
        "strategy_tag",
        "entry_context_sha256",
        "tp_pips",
        "sl_pips",
        "observed_at_utc",
    }
)
_CANDLE_KEYS = frozenset(
    {
        "pair",
        "granularity",
        "started_at_utc",
        "completed_at_utc",
        "bid_o",
        "bid_h",
        "bid_l",
        "bid_c",
        "ask_o",
        "ask_h",
        "ask_l",
        "ask_c",
        "source_sha256",
        "max_age_seconds",
    }
)
_SOURCE_REQUIRED_KEYS = frozenset(
    {
        "source_id",
        "source_url",
        "title",
        "published_at_utc",
        "updated_at_utc",
        "fetched_at_utc",
        "observed_at_utc",
        "content_sha256",
    }
)
_SOURCE_OPTIONAL_KEYS = frozenset(
    {
        "subject",
        "fact",
        "actual",
        "consensus",
        "value",
        "unit",
        "change",
        "affected_currency",
        "transmission_chain",
        "observed_reaction",
        "contrary_evidence",
        "confidence",
    }
)
_SOURCE_KEYS = _SOURCE_REQUIRED_KEYS | _SOURCE_OPTIONAL_KEYS
_TRUSTED_REQUEST_KEYS = frozenset(
    {
        "contract",
        "cutoff_utc",
        "experiment_id",
        "room_id",
        "candidate_id",
        "spec_id",
        "policy_id",
        "source_files",
        "source_receipts",
        "dynamic_binding_max_age_seconds",
        "paper_only",
        "order_authority",
        "live_permission",
    }
)
_TRUSTED_SOURCE_FILE_KEYS = frozenset(
    {
        "session_contract",
        "candidate",
        "spec",
        "policy",
        "paper_eligible_event",
        "ledger",
        "state",
        "snapshot",
        "position",
        "entry_context",
        "entry_signal",
        "quote",
        "candles",
        "news_items",
        "calendar_items",
        "cross_asset_items",
    }
)


class EvidencePacketError(RuntimeError):
    """Base class for fail-closed evidence packet failures."""


class EvidencePacketIntegrityError(EvidencePacketError):
    """An immutable packet, root, or content digest cannot be trusted."""


class EvidencePacketMarketClosedError(EvidencePacketError):
    """New AI evidence is disabled while the deterministic FX week is closed."""


def build_ai_inventory_evidence_packet(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """UNSAFE/test-only low-level builder from caller-assembled rows.

    This compatibility surface accepts caller-populated digest fields and must
    never be used by a paper-room launch or decision path.  Launch code must
    call :func:`build_trusted_ai_inventory_evidence_packet`, which reads and
    hashes canonical source files itself.  The writer clock remains internal
    so even test callers cannot backdate the seal.
    """

    now = _utc_now().astimezone(timezone.utc)
    return _seal_packet(value, sealed_at=now)


def build_trusted_ai_inventory_evidence_packet(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a launch-safe packet from canonical on-disk source truth.

    The repository root cannot be supplied by a caller.  A short-lived
    caller-provided token is not accepted.  The per-room PAPER_ELIGIBLE
    preflight and its complete proof ladder are revalidated from the canonical
    lifecycle root using the request's experiment and room identifiers.
    Package-derived worktree, current Git HEAD/branch, and the fixed canonical
    source root are then bound into the packet.

    ``value`` contains identifiers and content-addressed filenames only.  It
    contains no digest field.  Every source must be a direct regular-file child
    of the dedicated canonical source root and be named ``<sha256>.json``.
    Bytes, filename digest, strict JSON, signed acquisition receipts, embedded
    timestamps, and scope identities are revalidated before the ordinary
    semantic sealer sees the assembled packet.  Filesystem mtimes are used
    only to detect a concurrent mutation during one read; they are never a
    chronology or provenance trust root.
    """

    request = _normalize_trusted_request(value)
    repository_root = _trusted_repository_root()
    now = _utc_now().astimezone(timezone.utc)
    preflight, git_head, git_branch = _verified_room_launch_preflight(
        repository_root=repository_root,
        request=request,
    )
    return _build_trusted_ai_inventory_evidence_packet_for_root(
        repository_root,
        request,
        preflight=preflight,
        runtime_git_head=git_head,
        runtime_git_branch=git_branch,
        sealed_at=now,
    )


def _build_trusted_ai_inventory_evidence_packet_for_root(
    repository_root: Path,
    value: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    runtime_git_head: str,
    runtime_git_branch: str,
    sealed_at: datetime,
) -> dict[str, Any]:
    """Internal implementation; tests inject the package-root resolver."""

    request = _normalize_trusted_request(value)

    cutoff = _parse_utc(request.get("cutoff_utc"), "cutoff_utc")
    source_root = _canonical_source_root(repository_root)
    source_files = _require_mapping(request.get("source_files"), "source_files")
    _require_exact_keys(source_files, _TRUSTED_SOURCE_FILE_KEYS, "source_files")
    sources = {
        role: _read_trusted_canonical_source(
            source_root,
            source_files[role],
            cutoff=cutoff,
            role=role,
        )
        for role in sorted(_TRUSTED_SOURCE_FILE_KEYS)
    }
    _verify_trusted_source_receipts(
        repository_root,
        request=request,
        sources=sources,
        preflight=preflight,
    )
    _require_trusted_sources_match_preflight(sources, preflight=preflight)
    assembled = _assemble_trusted_packet(
        request,
        sources,
        preflight=preflight,
        git_head=runtime_git_head,
        git_branch=runtime_git_branch,
    )
    return _seal_packet(assembled, sealed_at=sealed_at)


def entry_signal_identity_sha256(value: Mapping[str, Any]) -> str:
    """Return the identity shared by a signal producer, packet, and proxy.

    The digest covers every strict entry-signal field except the digest itself.
    Callers must provide the canonical values that they intend to seal; the
    packet builder independently normalizes, recomputes, and compares it.
    """

    if not isinstance(value, Mapping):
        raise TypeError("entry signal must be a mapping")
    snapshot = _snapshot_mapping(value, "entry signal")
    keys = set(snapshot)
    allowed_without_digest = _ENTRY_SIGNAL_KEYS - {"signal_identity_sha256"}
    if keys != _ENTRY_SIGNAL_KEYS and keys != allowed_without_digest:
        raise ValueError("entry signal identity input schema is invalid")
    body = {key: snapshot[key] for key in sorted(allowed_without_digest)}
    body["units"] = float(
        _require_number(body.get("units"), "entry_signal.units", positive=True)
    )
    for key in ("price", "tp_pips", "sl_pips"):
        if body.get(key) is not None:
            body[key] = float(
                _require_number(body.get(key), f"entry_signal.{key}", positive=True)
            )
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def write_ai_inventory_evidence_packet(
    repository_root: Path | str,
    value: Mapping[str, Any],
) -> Path:
    """UNSAFE/test-only persistence for caller-assembled low-level rows.

    Launch code must use :func:`write_trusted_ai_inventory_evidence_packet`.
    """

    packet = build_ai_inventory_evidence_packet(value)
    return _persist_sealed_packet(repository_root, packet)


def write_trusted_ai_inventory_evidence_packet(
    value: Mapping[str, Any],
) -> Path:
    """Build from trusted canonical sources and persist immutably."""

    request = _normalize_trusted_request(value)
    repository_root = _trusted_repository_root()
    now = _utc_now().astimezone(timezone.utc)
    preflight, git_head, git_branch = _verified_room_launch_preflight(
        repository_root=repository_root,
        request=request,
    )
    packet = _build_trusted_ai_inventory_evidence_packet_for_root(
        repository_root,
        request,
        preflight=preflight,
        runtime_git_head=git_head,
        runtime_git_branch=git_branch,
        sealed_at=now,
    )
    return _persist_sealed_packet(repository_root, packet)


def _normalize_trusted_request(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("trusted evidence request must be a mapping")
    request = _snapshot_mapping(value, "trusted evidence request")
    _require_exact_keys(request, _TRUSTED_REQUEST_KEYS, "trusted evidence request")
    if request.get("contract") != DOJO_AI_EVIDENCE_PACKET_CONTRACT:
        raise ValueError("trusted evidence request contract is invalid")
    if request.get("paper_only") is not True:
        raise ValueError("trusted evidence request must be paper_only=true")
    if request.get("order_authority") != "NONE":
        raise ValueError("trusted evidence request order_authority must be NONE")
    if request.get("live_permission") is not False:
        raise ValueError("trusted evidence request live_permission must be false")
    for key in ("experiment_id", "room_id", "candidate_id", "spec_id", "policy_id"):
        _require_id(request.get(key), f"trusted evidence request {key}")
    _parse_utc(request.get("cutoff_utc"), "cutoff_utc")
    return request


def _verified_room_launch_preflight(
    *,
    repository_root: Path,
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    """Load the canonical per-room PAPER_ELIGIBLE proof; accept no token."""

    try:
        raw_token = verify_paper_ai_inventory_launch_preflight(
            repository_root,
            experiment_id=request["experiment_id"],
            room_id=request["room_id"],
        )
    except (DojoReplayLifecycleError, OSError, TypeError, ValueError) as exc:
        raise EvidencePacketIntegrityError(
            "canonical per-room PAPER_ELIGIBLE preflight is invalid"
        ) from exc
    try:
        token = _validate_verified_room_launch_preflight(raw_token, request=request)
    except EvidencePacketIntegrityError:
        raise
    except (TypeError, ValueError) as exc:
        raise EvidencePacketIntegrityError(
            "verified PAPER_ELIGIBLE preflight fields are invalid"
        ) from exc
    git_head, git_branch = _read_git_identity(repository_root)
    try:
        _require_git_head(git_head, "runtime Git HEAD")
        _require_codex_branch(git_branch, "runtime Git branch")
    except ValueError as exc:
        raise EvidencePacketIntegrityError("runtime Git identity is invalid") from exc
    if token["git_head"] != git_head:
        raise EvidencePacketIntegrityError(
            "PAPER_ELIGIBLE preflight Git HEAD no longer matches runtime"
        )
    return token, git_head, git_branch


def _validate_verified_room_launch_preflight(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidencePacketIntegrityError(
            "verified PAPER_ELIGIBLE preflight must be a mapping"
        )
    token = _snapshot_mapping(value, "verified PAPER_ELIGIBLE preflight")
    if set(token) != _PAPER_AI_LAUNCH_PREFLIGHT_KEYS:
        raise EvidencePacketIntegrityError(
            "verified PAPER_ELIGIBLE preflight schema is invalid"
        )
    if token.get("contract") != PAPER_AI_LAUNCH_PREFLIGHT_CONTRACT:
        raise EvidencePacketIntegrityError(
            "verified PAPER_ELIGIBLE preflight contract is invalid"
        )
    if token.get("paper_only") is not True:
        raise EvidencePacketIntegrityError(
            "verified PAPER_ELIGIBLE preflight must be paper_only=true"
        )
    if token.get("order_authority") != "NONE":
        raise EvidencePacketIntegrityError(
            "verified PAPER_ELIGIBLE preflight order_authority must be NONE"
        )
    if token.get("live_permission") is not False:
        raise EvidencePacketIntegrityError(
            "verified PAPER_ELIGIBLE preflight live_permission must be false"
        )
    if token.get("paper_room_launched") is not False:
        raise EvidencePacketIntegrityError(
            "verified PAPER_ELIGIBLE preflight launch state is invalid"
        )
    expected_values = {
        "experiment_id": request["experiment_id"],
        "room_id": request["room_id"],
        "candidate_id": request["candidate_id"],
    }
    for key, expected in expected_values.items():
        if token.get(key) != expected:
            raise EvidencePacketIntegrityError(
                f"verified PAPER_ELIGIBLE preflight {key} binding mismatch"
            )
    _require_git_head(token.get("git_head"), "PAPER_ELIGIBLE preflight git_head")
    for key in (
        "candidate_id",
        "config_sha256",
        "spec_sha256",
        "policy_sha256",
        "paper_eligible_event_sha256",
        "candidate_lifecycle_ledger_tip_sha256",
        "append_claim_sha256",
        "job_manifest_sha256",
        "job_owner_sha256",
        "proof_artifact_sha256",
        "proof_artifact_bytes_sha256",
        "proof_manifest_sha256",
        "replay_worker_receipt_sha256",
        "future_registry_sha256",
        "source_capture_manifest_sha256",
        "git_head_sha256",
        "launch_preflight_token_sha256",
    ):
        _require_sha(token.get(key), f"PAPER_ELIGIBLE preflight {key}")
    for key in ("adapter_id", "model_id", "producer_id"):
        _require_id(token.get(key), f"PAPER_ELIGIBLE preflight {key}")
    source_manifests = _require_mapping(
        token.get("source_manifest_sha256s"),
        "PAPER_ELIGIBLE preflight source_manifest_sha256s",
    )
    if not source_manifests:
        raise EvidencePacketIntegrityError(
            "verified PAPER_ELIGIBLE preflight source manifests are empty"
        )
    for window, digest in source_manifests.items():
        _require_id(window, "PAPER_ELIGIBLE preflight source manifest window")
        _require_sha(
            digest,
            f"PAPER_ELIGIBLE preflight source_manifest_sha256s.{window}",
        )
    body = {
        key: item
        for key, item in token.items()
        if key != "launch_preflight_token_sha256"
    }
    if (
        token["launch_preflight_token_sha256"]
        != hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    ):
        raise EvidencePacketIntegrityError(
            "verified PAPER_ELIGIBLE preflight digest mismatch"
        )
    issued_at = _parse_utc(
        token.get("issued_at_utc"), "PAPER_ELIGIBLE preflight issued_at_utc"
    )
    future_window = _require_mapping(
        token.get("future_window"), "PAPER_ELIGIBLE preflight future_window"
    )
    _require_exact_keys(
        future_window,
        frozenset({"start_utc", "end_utc"}),
        "PAPER_ELIGIBLE preflight future_window",
    )
    start = _parse_utc(
        future_window.get("start_utc"),
        "PAPER_ELIGIBLE preflight future_window.start_utc",
    )
    end = _parse_utc(
        future_window.get("end_utc"),
        "PAPER_ELIGIBLE preflight future_window.end_utc",
    )
    cutoff = _parse_utc(request.get("cutoff_utc"), "cutoff_utc")
    if issued_at >= start or start >= end:
        raise EvidencePacketIntegrityError(
            "verified PAPER_ELIGIBLE preflight time binding is invalid"
        )
    if cutoff < start or cutoff >= end:
        raise EvidencePacketIntegrityError(
            "evidence cutoff is outside PAPER_ELIGIBLE future window"
        )
    return token


def _require_trusted_sources_match_preflight(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    preflight: Mapping[str, Any],
) -> None:
    expected_source_digests = {
        "spec": preflight["spec_sha256"],
        "policy": preflight["policy_sha256"],
    }
    for role, expected in expected_source_digests.items():
        if sources[role]["sha256"] != expected:
            raise EvidencePacketIntegrityError(
                f"trusted {role} source does not match PAPER_ELIGIBLE preflight"
            )
    event = _trusted_document_mapping(sources, "paper_eligible_event")
    if (
        event.get("event_sha256") != preflight["paper_eligible_event_sha256"]
        or event.get("event_sha256")
        != preflight["candidate_lifecycle_ledger_tip_sha256"]
    ):
        raise EvidencePacketIntegrityError(
            "trusted PAPER_ELIGIBLE event does not match lifecycle tip"
        )


def _verify_trusted_source_receipts(
    repository_root: Path,
    *,
    request: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    preflight: Mapping[str, Any],
) -> None:
    receipts = _require_mapping(
        request.get("source_receipts"), "source_receipts"
    )
    _require_exact_keys(
        receipts, _TRUSTED_SOURCE_FILE_KEYS, "source_receipts"
    )
    cutoff = _parse_utc(request.get("cutoff_utc"), "cutoff_utc")
    for role in sorted(_TRUSTED_SOURCE_FILE_KEYS):
        receipt_sha = _require_sha(
            receipts.get(role), f"source_receipts.{role}"
        )
        try:
            verify_ai_source_capture_receipt(
                repository_root,
                experiment_id=request["experiment_id"],
                room_id=request["room_id"],
                candidate_id=request["candidate_id"],
                cutoff_utc=cutoff,
                source_role=role,
                source_sha256=sources[role]["sha256"],
                receipt_sha256=receipt_sha,
            )
        except (AiSourceCaptureError, OSError, TypeError, ValueError) as exc:
            raise EvidencePacketIntegrityError(
                f"trusted {role} source lacks a valid signed acquisition receipt"
            ) from exc


def _validate_virtual_broker_ledger_snapshot(
    value: object,
    *,
    room_id: str,
) -> tuple[str, str]:
    """Validate the complete broker ledger and return semantic tip/time."""

    snapshot = _require_mapping(value, "trusted virtual broker ledger")
    _require_exact_keys(
        snapshot,
        _VIRTUAL_BROKER_LEDGER_SNAPSHOT_KEYS,
        "trusted virtual broker ledger",
    )
    if snapshot.get("contract") != VIRTUAL_BROKER_LEDGER_SNAPSHOT_CONTRACT:
        raise EvidencePacketIntegrityError(
            "trusted virtual broker ledger contract is invalid"
        )
    if snapshot.get("room_id") != room_id:
        raise EvidencePacketIntegrityError(
            "trusted virtual broker ledger room binding mismatch"
        )
    observed_at = _parse_utc(
        snapshot.get("observed_at_utc"),
        "trusted virtual broker ledger observed_at_utc",
    )
    rows = snapshot.get("rows")
    if not isinstance(rows, list):
        raise EvidencePacketIntegrityError(
            "trusted virtual broker ledger rows must be an array"
        )
    expected_previous = "0" * 64
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            raise EvidencePacketIntegrityError(
                f"trusted virtual broker ledger rows[{index}] must be an object"
            )
        row = _snapshot_mapping(raw_row, f"trusted virtual broker ledger rows[{index}]")
        _require_exact_keys(
            row,
            _VIRTUAL_BROKER_LEDGER_ROW_KEYS,
            f"trusted virtual broker ledger rows[{index}]",
        )
        timestamp = _parse_utc(
            row.get("ts_utc"),
            f"trusted virtual broker ledger rows[{index}].ts_utc",
        )
        if timestamp > observed_at:
            raise EvidencePacketIntegrityError(
                "trusted virtual broker ledger row is after observation"
            )
        if not isinstance(row.get("event"), str) or not row["event"]:
            raise EvidencePacketIntegrityError(
                "trusted virtual broker ledger event is invalid"
            )
        if not isinstance(row.get("payload"), dict):
            raise EvidencePacketIntegrityError(
                "trusted virtual broker ledger payload is invalid"
            )
        if row.get("prev_sha") != expected_previous:
            raise EvidencePacketIntegrityError(
                "trusted virtual broker ledger prev_sha mismatch"
            )
        body = {key: row[key] for key in ("ts_utc", "event", "payload", "prev_sha")}
        expected_sha = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
        if row.get("sha") != expected_sha:
            raise EvidencePacketIntegrityError(
                "trusted virtual broker ledger sha mismatch"
            )
        expected_previous = expected_sha
    terminal = _require_sha(
        snapshot.get("terminal_sha256"),
        "trusted virtual broker ledger terminal_sha256",
    )
    if terminal != expected_previous:
        raise EvidencePacketIntegrityError(
            "trusted virtual broker ledger terminal_sha256 mismatch"
        )
    return terminal, _format_utc(observed_at)


def _trusted_repository_root() -> Path:
    """Resolve the worktree from this module, never from caller input."""

    try:
        root = Path(__file__).resolve(strict=True).parents[2]
        root_stat = root.lstat()
        git_stat = (root / ".git").lstat()
    except (IndexError, OSError) as exc:
        raise EvidencePacketIntegrityError(
            "package-derived repository root is unavailable"
        ) from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise EvidencePacketIntegrityError("package-derived repository root is unsafe")
    if stat.S_ISLNK(git_stat.st_mode) or not (
        stat.S_ISDIR(git_stat.st_mode) or stat.S_ISREG(git_stat.st_mode)
    ):
        raise EvidencePacketIntegrityError("package Git metadata is unsafe")
    return root


def _read_git_identity(repository_root: Path) -> tuple[str, str]:
    """Read one symbolic branch and HEAD from Git metadata fail-closed."""

    dot_git = repository_root / ".git"
    dot_git_stat = dot_git.lstat()
    if stat.S_ISLNK(dot_git_stat.st_mode):
        raise EvidencePacketIntegrityError("package Git metadata is unsafe")
    if stat.S_ISREG(dot_git_stat.st_mode):
        marker = _read_git_text(dot_git, "Git worktree pointer")
        if not marker.startswith("gitdir: "):
            raise EvidencePacketIntegrityError("Git worktree pointer is invalid")
        raw_git_dir = Path(marker.removeprefix("gitdir: ").strip())
        git_dir = (
            raw_git_dir if raw_git_dir.is_absolute() else repository_root / raw_git_dir
        )
    elif stat.S_ISDIR(dot_git_stat.st_mode):
        git_dir = dot_git
    else:
        raise EvidencePacketIntegrityError("package Git metadata is invalid")
    try:
        git_dir_stat = git_dir.lstat()
        git_dir = git_dir.resolve(strict=True)
    except OSError as exc:
        raise EvidencePacketIntegrityError("Git directory is unavailable") from exc
    if stat.S_ISLNK(git_dir_stat.st_mode) or not stat.S_ISDIR(git_dir_stat.st_mode):
        raise EvidencePacketIntegrityError("Git directory is unsafe")

    common_dir = git_dir
    common_dir_pointer = git_dir / "commondir"
    if common_dir_pointer.exists():
        raw_common_dir = Path(
            _read_git_text(common_dir_pointer, "Git common directory pointer")
        )
        candidate = (
            raw_common_dir if raw_common_dir.is_absolute() else git_dir / raw_common_dir
        )
        try:
            candidate_stat = candidate.lstat()
            common_dir = candidate.resolve(strict=True)
        except OSError as exc:
            raise EvidencePacketIntegrityError(
                "Git common directory is unavailable"
            ) from exc
        if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISDIR(
            candidate_stat.st_mode
        ):
            raise EvidencePacketIntegrityError("Git common directory is unsafe")

    head_text = _read_git_text(git_dir / "HEAD", "Git HEAD")
    if not head_text.startswith("ref: refs/heads/"):
        raise EvidencePacketIntegrityError(
            "trusted launch requires a symbolic branch, not detached HEAD"
        )
    ref_name = head_text.removeprefix("ref: ").strip()
    branch = ref_name.removeprefix("refs/heads/")
    if not branch.startswith("codex/") or not _ID_RE.fullmatch(branch):
        raise EvidencePacketIntegrityError("trusted launch branch must be under codex/")
    ref_parts = Path(ref_name)
    if ref_parts.is_absolute() or ".." in ref_parts.parts:
        raise EvidencePacketIntegrityError("Git branch reference is unsafe")

    head: str | None = None
    for base in (git_dir, common_dir):
        ref_path = base / ref_parts
        if ref_path.exists():
            head = _read_git_text(ref_path, "Git branch reference").lower()
            break
    if head is None:
        packed_refs = common_dir / "packed-refs"
        if packed_refs.exists():
            for line in _read_git_text(packed_refs, "Git packed refs").splitlines():
                if line.startswith(("#", "^")) or " " not in line:
                    continue
                candidate_head, candidate_ref = line.split(" ", 1)
                if candidate_ref == ref_name:
                    head = candidate_head.lower()
                    break
    if head is None or not _GIT_HEAD_RE.fullmatch(head):
        raise EvidencePacketIntegrityError("Git HEAD commit is invalid")
    return head, branch


def _read_git_text(path: Path, field: str) -> str:
    try:
        item_stat = path.lstat()
        if stat.S_ISLNK(item_stat.st_mode) or not stat.S_ISREG(item_stat.st_mode):
            raise EvidencePacketIntegrityError(f"{field} is unsafe")
        if item_stat.st_size <= 0 or item_stat.st_size > 1024 * 1024:
            raise EvidencePacketIntegrityError(f"{field} size is invalid")
        value = path.read_text(encoding="utf-8")
    except EvidencePacketIntegrityError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidencePacketIntegrityError(f"{field} is unreadable") from exc
    return value.strip()


def _persist_sealed_packet(
    repository_root: Path | str,
    packet: Mapping[str, Any],
) -> Path:
    raw = _canonical_json_bytes(packet) + b"\n"
    if len(raw) > MAX_PACKET_BYTES:
        raise EvidencePacketError("evidence packet exceeds bounded size")

    root = _dedicated_root(repository_root, create=True)
    packet_path = root / f"{packet['packet_sha256']}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(packet_path, flags, 0o600)
    except FileExistsError:
        existing = _read_exact_packet_file(packet_path)
        if existing != raw:
            raise EvidencePacketIntegrityError(
                "existing evidence packet is tampered or collides with its digest"
            )
        # Full semantic verification is required even for an identical retry.
        verify_ai_inventory_evidence_packet(repository_root, packet_path)
        return packet_path
    except OSError as exc:
        raise EvidencePacketError("exclusive evidence packet create failed") from exc

    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(root)
    except Exception:
        try:
            packet_path.unlink()
            _fsync_directory(root)
        except OSError:
            pass
        raise
    return packet_path


def verify_ai_inventory_evidence_packet(
    repository_root: Path | str,
    packet_path: Path | str,
) -> dict[str, Any]:
    """Authenticate a packet and return its strict model-facing allowlist.

    ``packet_path`` must name a direct regular-file child of the dedicated
    evidence root.  The returned mapping contains no filesystem path and no
    pointer to an original source; it contains only the sealed structured
    rows, source URLs, and their content digests.
    """

    root = _dedicated_root(repository_root, create=False)
    path = Path(packet_path)
    _require_direct_packet_child(root, path)
    raw = _read_exact_packet_file(path)
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise EvidencePacketIntegrityError(
            "evidence packet must be one canonical newline-terminated JSON object"
        )
    try:
        payload = json.loads(
            raw[:-1],
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidencePacketIntegrityError("evidence packet JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise EvidencePacketIntegrityError("evidence packet must be an object")
    if set(payload) != _SEALED_KEYS:
        raise EvidencePacketIntegrityError(
            "evidence packet top-level schema is invalid"
        )

    sealed_at = _parse_utc(payload.get("sealed_at_utc"), "sealed_at_utc")
    input_body = {key: payload[key] for key in _INPUT_KEYS}
    try:
        rebuilt = _seal_packet(input_body, sealed_at=sealed_at)
    except (TypeError, ValueError, EvidencePacketError) as exc:
        raise EvidencePacketIntegrityError(
            "evidence packet semantic verification failed"
        ) from exc
    expected_raw = _canonical_json_bytes(rebuilt) + b"\n"
    if raw != expected_raw:
        raise EvidencePacketIntegrityError(
            "evidence packet is noncanonical or its digest does not match"
        )
    expected_name = f"{rebuilt['packet_sha256']}.json"
    if path.name != expected_name:
        raise EvidencePacketIntegrityError(
            "evidence packet filename does not match content digest"
        )
    # Snapshot through canonical JSON so callers cannot mutate cached state.
    return json.loads(_canonical_json_bytes(rebuilt))


def _seal_packet(
    value: Mapping[str, Any],
    *,
    sealed_at: datetime,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("evidence packet input must be a mapping")
    body = _snapshot_mapping(value, "evidence packet input")
    _require_exact_keys(body, _INPUT_KEYS, "evidence packet input")

    if body.get("contract") != DOJO_AI_EVIDENCE_PACKET_CONTRACT:
        raise ValueError("evidence packet contract is invalid")
    if body.get("paper_only") is not True:
        raise ValueError("evidence packet must be paper_only=true")
    if body.get("order_authority") != "NONE":
        raise ValueError("evidence packet order_authority must be NONE")
    if body.get("live_permission") is not False:
        raise ValueError("evidence packet live_permission must be false")

    cutoff = _parse_utc(body.get("cutoff_utc"), "cutoff_utc")
    sealed_at = _require_aware_utc(sealed_at, "sealed_at")
    _require_market_open(cutoff, "cutoff")
    _require_market_open(sealed_at, "writer clock")
    if sealed_at < cutoff:
        raise ValueError("writer clock precedes cutoff")
    lag_ns = _nanoseconds_between(cutoff, sealed_at)
    if lag_ns > MAX_PACKET_BUILD_LAG_SECONDS * 1_000_000_000:
        raise ValueError("evidence packet build lag exceeds the fixed limit")

    dynamic_age = _require_int(
        body.get("dynamic_binding_max_age_seconds"),
        "dynamic_binding_max_age_seconds",
        minimum=1,
        maximum=MAX_DYNAMIC_BINDING_AGE_SECONDS,
    )
    bindings = _normalize_bindings(
        body.get("bindings"),
        cutoff=cutoff,
        max_age_seconds=dynamic_age,
    )
    position = _normalize_position(
        body.get("position"),
        cutoff=cutoff,
        max_age_seconds=dynamic_age,
    )
    entry_signal = _normalize_entry_signal(
        body.get("entry_signal"),
        position=position,
        cutoff=cutoff,
        max_age_seconds=dynamic_age,
    )
    quote = _normalize_quote(body.get("quote"), cutoff=cutoff)
    if position["pair"] != quote["pair"]:
        raise ValueError("position and quote pair mismatch")
    candles = _normalize_candles(
        body.get("candles"),
        cutoff=cutoff,
        position_pair=position["pair"],
    )
    news_items = _normalize_source_items(
        body.get("news_items"), kind="news", cutoff=cutoff
    )
    calendar_items = _normalize_source_items(
        body.get("calendar_items"), kind="calendar", cutoff=cutoff
    )
    cross_asset_items = _normalize_source_items(
        body.get("cross_asset_items"), kind="cross_asset", cutoff=cutoff
    )

    normalized: dict[str, Any] = {
        "contract": DOJO_AI_EVIDENCE_PACKET_CONTRACT,
        "cutoff_utc": _format_utc(cutoff),
        "bindings": bindings,
        "position": position,
        "entry_signal": entry_signal,
        "quote": quote,
        "candles": candles,
        "news_items": news_items,
        "calendar_items": calendar_items,
        "cross_asset_items": cross_asset_items,
        "dynamic_binding_max_age_seconds": dynamic_age,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "sealed_at_utc": _format_utc(sealed_at),
        "seal_lag_nanoseconds": lag_ns,
        "source_row_counts": {
            "candles": len(candles),
            "news": len(news_items),
            "calendar": len(calendar_items),
            "cross_assets": len(cross_asset_items),
        },
    }
    normalized["packet_sha256"] = _packet_sha256(normalized)
    return normalized


def _normalize_bindings(
    value: object,
    *,
    cutoff: datetime,
    max_age_seconds: int,
) -> dict[str, Any]:
    mapping = _require_mapping(value, "bindings")
    _require_exact_keys(mapping, _BINDING_KEYS, "bindings")
    normalized: dict[str, Any] = {
        "launch_preflight_token_sha256": _require_sha(
            mapping.get("launch_preflight_token_sha256"),
            "bindings.launch_preflight_token_sha256",
        ),
        "git_head": _require_git_head(mapping.get("git_head"), "bindings.git_head"),
        "git_branch": _require_codex_branch(
            mapping.get("git_branch"), "bindings.git_branch"
        ),
        "canonical_source_root": _require_canonical_source_root_binding(
            mapping.get("canonical_source_root")
        ),
    }
    for key in (
        "experiment_id",
        "room_id",
        "candidate_id",
        "spec_id",
        "policy_id",
    ):
        normalized[key] = _require_id(mapping.get(key), f"bindings.{key}")
    for key in (
        "session_contract_sha256",
        "candidate_sha256",
        "spec_sha256",
        "policy_sha256",
        "paper_eligible_tip_sha256",
        "ledger_sha256",
        "state_sha256",
        "snapshot_sha256",
    ):
        normalized[key] = _require_sha(mapping.get(key), f"bindings.{key}")
    for key in (
        "ledger_observed_at_utc",
        "state_observed_at_utc",
        "snapshot_observed_at_utc",
    ):
        observed = _parse_utc(mapping.get(key), f"bindings.{key}")
        _require_not_after(observed, cutoff, f"bindings.{key}")
        _require_fresh(
            observed,
            cutoff,
            max_age_seconds=max_age_seconds,
            field=f"bindings.{key}",
        )
        normalized[key] = _format_utc(observed)
    return normalized


def _normalize_position(
    value: object,
    *,
    cutoff: datetime,
    max_age_seconds: int,
) -> dict[str, Any]:
    mapping = _require_mapping(value, "position")
    _require_exact_keys(mapping, _POSITION_KEYS, "position")
    pair = _require_pair(mapping.get("pair"), "position.pair")
    side = mapping.get("side")
    if side not in {"FLAT", "LONG", "SHORT"}:
        raise ValueError("position.side must be FLAT, LONG, or SHORT")
    observed = _parse_utc(mapping.get("observed_at_utc"), "position.observed_at_utc")
    _require_not_after(observed, cutoff, "position.observed_at_utc")
    _require_fresh(
        observed,
        cutoff,
        max_age_seconds=max_age_seconds,
        field="position.observed_at_utc",
    )

    common: dict[str, Any] = {
        "pair": pair,
        "side": side,
        "observed_at_utc": _format_utc(observed),
        "strategy_tag": _require_id(
            mapping.get("strategy_tag"), "position.strategy_tag"
        ),
        "entry_context_sha256": _require_sha(
            mapping.get("entry_context_sha256"),
            "position.entry_context_sha256",
        ),
    }
    if side == "FLAT":
        expected_position_id = f"FLAT:{pair}"
        if mapping.get("position_id") != expected_position_id:
            raise ValueError("flat position_id must be the exact FLAT:<pair> identity")
        for key in ("entry_price", "opened_at_utc", "take_profit", "stop_loss"):
            if mapping.get(key) is not None:
                raise ValueError(f"flat position.{key} must be null")
        _require_exact_float_zero(mapping.get("units"), "position.units")
        for key in (
            "unrealized_pl_jpy",
            "gross_same_currency_units",
            "net_same_currency_units",
            "margin_used_jpy",
            "capital_locked_jpy",
        ):
            _require_exact_float_zero(mapping.get(key), f"position.{key}")
        _require_exact_int_zero(
            mapping.get("remaining_ceiling_seconds"),
            "position.remaining_ceiling_seconds",
        )
        _require_exact_int_zero(
            mapping.get("same_direction_position_count"),
            "position.same_direction_position_count",
        )
        return {
            "position_id": expected_position_id,
            **common,
            "units": 0.0,
            "entry_price": None,
            "opened_at_utc": None,
            "take_profit": None,
            "stop_loss": None,
            "remaining_ceiling_seconds": 0,
            "unrealized_pl_jpy": 0.0,
            "gross_same_currency_units": 0.0,
            "net_same_currency_units": 0.0,
            "margin_used_jpy": 0.0,
            "capital_locked_jpy": 0.0,
            "same_direction_position_count": 0,
        }

    opened = _parse_utc(mapping.get("opened_at_utc"), "position.opened_at_utc")
    _require_not_after(opened, cutoff, "position.opened_at_utc")
    position_id = _require_id(mapping.get("position_id"), "position.position_id")
    if position_id.startswith("FLAT:"):
        raise ValueError("open position cannot use a FLAT identity")
    units = _require_number(mapping.get("units"), "position.units", positive=True)
    result: dict[str, Any] = {
        "position_id": position_id,
        **common,
        "units": float(units),
        "entry_price": _require_number(
            mapping.get("entry_price"), "position.entry_price", positive=True
        ),
        "opened_at_utc": _format_utc(opened),
        "take_profit": _require_number(
            mapping.get("take_profit"), "position.take_profit", positive=True
        ),
        "stop_loss": _require_number(
            mapping.get("stop_loss"), "position.stop_loss", positive=True
        ),
        "remaining_ceiling_seconds": _require_int(
            mapping.get("remaining_ceiling_seconds"),
            "position.remaining_ceiling_seconds",
            minimum=0,
            maximum=7 * 24 * 60 * 60,
        ),
        "unrealized_pl_jpy": _require_number(
            mapping.get("unrealized_pl_jpy"), "position.unrealized_pl_jpy"
        ),
        "gross_same_currency_units": _require_number(
            mapping.get("gross_same_currency_units"),
            "position.gross_same_currency_units",
            minimum=0,
        ),
        "net_same_currency_units": _require_number(
            mapping.get("net_same_currency_units"),
            "position.net_same_currency_units",
        ),
        "margin_used_jpy": _require_number(
            mapping.get("margin_used_jpy"), "position.margin_used_jpy", minimum=0
        ),
        "capital_locked_jpy": _require_number(
            mapping.get("capital_locked_jpy"),
            "position.capital_locked_jpy",
            minimum=0,
        ),
        "same_direction_position_count": _require_int(
            mapping.get("same_direction_position_count"),
            "position.same_direction_position_count",
            minimum=1,
            maximum=1_000,
        ),
    }
    return result


def _normalize_entry_signal(
    value: object,
    *,
    position: Mapping[str, Any],
    cutoff: datetime,
    max_age_seconds: int,
) -> dict[str, Any] | None:
    side = position["side"]
    if side in {"LONG", "SHORT"}:
        if value is not None:
            raise ValueError("open position entry_signal must be null")
        return None
    if side != "FLAT":
        raise ValueError("entry signal position scope is invalid")

    mapping = _require_mapping(value, "entry_signal")
    _require_exact_keys(mapping, _ENTRY_SIGNAL_KEYS, "entry_signal")
    signal_side = mapping.get("side")
    if signal_side not in {"LONG", "SHORT"}:
        raise ValueError("entry_signal.side must be LONG or SHORT")
    order_type = mapping.get("order_type")
    if order_type not in {"MARKET", "LIMIT", "STOP"}:
        raise ValueError("entry_signal.order_type must be MARKET, LIMIT, or STOP")
    observed = _parse_utc(
        mapping.get("observed_at_utc"), "entry_signal.observed_at_utc"
    )
    _require_not_after(observed, cutoff, "entry_signal.observed_at_utc")
    _require_fresh(
        observed,
        cutoff,
        max_age_seconds=max_age_seconds,
        field="entry_signal.observed_at_utc",
    )
    pair = _require_pair(mapping.get("pair"), "entry_signal.pair")
    strategy_tag = _require_id(mapping.get("strategy_tag"), "entry_signal.strategy_tag")
    entry_context = _require_sha(
        mapping.get("entry_context_sha256"),
        "entry_signal.entry_context_sha256",
    )
    if pair != position["pair"]:
        raise ValueError("entry_signal pair does not match FLAT position scope")
    if strategy_tag != position["strategy_tag"]:
        raise ValueError("entry_signal strategy_tag does not match FLAT position scope")
    if entry_context != position["entry_context_sha256"]:
        raise ValueError(
            "entry_signal entry_context_sha256 does not match FLAT position scope"
        )

    units = float(
        _require_number(mapping.get("units"), "entry_signal.units", positive=True)
    )
    raw_price = mapping.get("price")
    if order_type == "MARKET":
        if raw_price is not None:
            raise ValueError("MARKET entry_signal.price must be null")
        price: int | float | None = None
    else:
        price = float(_require_number(raw_price, "entry_signal.price", positive=True))
    normalized: dict[str, Any] = {
        "signal_identity_sha256": _require_sha(
            mapping.get("signal_identity_sha256"),
            "entry_signal.signal_identity_sha256",
        ),
        "pair": pair,
        "side": signal_side,
        "order_type": order_type,
        "units": units,
        "price": price,
        "strategy_tag": strategy_tag,
        "entry_context_sha256": entry_context,
        "tp_pips": _optional_positive_float(
            mapping.get("tp_pips"), "entry_signal.tp_pips"
        ),
        "sl_pips": _optional_positive_float(
            mapping.get("sl_pips"), "entry_signal.sl_pips"
        ),
        "observed_at_utc": _format_utc(observed),
    }
    expected_identity = entry_signal_identity_sha256(normalized)
    if normalized["signal_identity_sha256"] != expected_identity:
        raise ValueError("entry_signal signal_identity_sha256 mismatch")
    return normalized


def _normalize_quote(value: object, *, cutoff: datetime) -> dict[str, Any]:
    mapping = _require_mapping(value, "quote")
    _require_exact_keys(mapping, _QUOTE_KEYS, "quote")
    timestamp = _parse_utc(mapping.get("timestamp_utc"), "quote.timestamp_utc")
    _require_not_after(timestamp, cutoff, "quote.timestamp_utc")
    max_age = _require_int(
        mapping.get("max_age_seconds"),
        "quote.max_age_seconds",
        minimum=1,
        maximum=MAX_QUOTE_AGE_SECONDS,
    )
    _require_fresh(
        timestamp,
        cutoff,
        max_age_seconds=max_age,
        field="quote.timestamp_utc",
    )
    bid = _require_number(mapping.get("bid"), "quote.bid", positive=True)
    ask = _require_number(mapping.get("ask"), "quote.ask", positive=True)
    if ask < bid:
        raise ValueError("quote ask is below bid")
    return {
        "pair": _require_pair(mapping.get("pair"), "quote.pair"),
        "bid": bid,
        "ask": ask,
        "timestamp_utc": _format_utc(timestamp),
        "source_sha256": _require_sha(
            mapping.get("source_sha256"), "quote.source_sha256"
        ),
        "max_age_seconds": max_age,
    }


def _normalize_candles(
    value: object,
    *,
    cutoff: datetime,
    position_pair: str,
) -> list[dict[str, Any]]:
    rows = _require_sequence(value, "candles", maximum=MAX_CANDLE_ROWS)
    if not rows:
        raise ValueError("candles must contain at least one completed row")
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for index, item in enumerate(rows):
        field = f"candles[{index}]"
        mapping = _require_mapping(item, field)
        _require_exact_keys(mapping, _CANDLE_KEYS, field)
        pair = _require_pair(mapping.get("pair"), f"{field}.pair")
        if pair != position_pair:
            raise ValueError(f"{field}.pair does not match active position")
        granularity = mapping.get("granularity")
        if not isinstance(granularity, str) or not _GRANULARITY_RE.fullmatch(
            granularity
        ):
            raise ValueError(f"{field}.granularity is invalid")
        started = _parse_utc(mapping.get("started_at_utc"), f"{field}.started_at_utc")
        completed = _parse_utc(
            mapping.get("completed_at_utc"), f"{field}.completed_at_utc"
        )
        if started >= completed:
            raise ValueError(f"{field} completion must follow start")
        _require_not_after(started, cutoff, f"{field}.started_at_utc")
        _require_not_after(completed, cutoff, f"{field}.completed_at_utc")
        max_age = _require_int(
            mapping.get("max_age_seconds"),
            f"{field}.max_age_seconds",
            minimum=1,
            maximum=MAX_CANDLE_AGE_SECONDS,
        )
        _require_fresh(
            completed,
            cutoff,
            max_age_seconds=max_age,
            field=f"{field}.completed_at_utc",
        )
        prices = {
            key: _require_number(mapping.get(key), f"{field}.{key}", positive=True)
            for key in (
                "bid_o",
                "bid_h",
                "bid_l",
                "bid_c",
                "ask_o",
                "ask_h",
                "ask_l",
                "ask_c",
            )
        }
        for prefix in ("bid", "ask"):
            low = prices[f"{prefix}_l"]
            high = prices[f"{prefix}_h"]
            open_price = prices[f"{prefix}_o"]
            close_price = prices[f"{prefix}_c"]
            if low > min(open_price, close_price) or high < max(
                open_price, close_price
            ):
                raise ValueError(f"{field} {prefix} OHLC geometry is invalid")
        for suffix in ("o", "h", "l", "c"):
            if prices[f"ask_{suffix}"] < prices[f"bid_{suffix}"]:
                raise ValueError(f"{field} ask is below bid")
        completed_text = _format_utc(completed)
        identity = (pair, granularity, completed_text)
        if identity in identities:
            raise ValueError("duplicate completed candle identity")
        identities.add(identity)
        normalized.append(
            {
                "pair": pair,
                "granularity": granularity,
                "started_at_utc": _format_utc(started),
                "completed_at_utc": completed_text,
                **prices,
                "source_sha256": _require_sha(
                    mapping.get("source_sha256"), f"{field}.source_sha256"
                ),
                "max_age_seconds": max_age,
            }
        )
    normalized.sort(
        key=lambda row: (
            row["pair"],
            row["granularity"],
            row["completed_at_utc"],
            row["source_sha256"],
        )
    )
    return normalized


def _normalize_source_items(
    value: object,
    *,
    kind: str,
    cutoff: datetime,
) -> list[dict[str, Any]]:
    rows = _require_sequence(value, f"{kind}_items", maximum=MAX_SOURCE_ROWS_PER_KIND)
    normalized: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    identities: set[tuple[str, str]] = set()
    for index, item in enumerate(rows):
        field = f"{kind}_items[{index}]"
        mapping = _require_mapping(item, field)
        keys = set(mapping)
        missing = _SOURCE_REQUIRED_KEYS - keys
        unknown = keys - _SOURCE_KEYS
        if missing or unknown:
            raise ValueError(
                f"{field} schema invalid; missing={sorted(missing)} "
                f"unknown={sorted(unknown)}"
            )
        source_id = _require_id(mapping.get("source_id"), f"{field}.source_id")
        content_sha = _require_sha(
            mapping.get("content_sha256"), f"{field}.content_sha256"
        )
        identity = (source_id, content_sha)
        if source_id in source_ids or identity in identities:
            raise ValueError(f"duplicate {kind} source row")
        source_ids.add(source_id)
        identities.add(identity)

        published = _parse_utc(
            mapping.get("published_at_utc"), f"{field}.published_at_utc"
        )
        updated = _parse_utc(mapping.get("updated_at_utc"), f"{field}.updated_at_utc")
        fetched = _parse_utc(mapping.get("fetched_at_utc"), f"{field}.fetched_at_utc")
        observed = _parse_utc(
            mapping.get("observed_at_utc"), f"{field}.observed_at_utc"
        )
        for timestamp_name, timestamp in (
            ("published_at_utc", published),
            ("updated_at_utc", updated),
            ("fetched_at_utc", fetched),
            ("observed_at_utc", observed),
        ):
            _require_not_after(timestamp, cutoff, f"{field}.{timestamp_name}")
        if updated < published:
            raise ValueError(f"{field}.updated_at_utc precedes publication")
        if fetched < updated:
            raise ValueError(f"{field}.fetched_at_utc precedes update")

        row: dict[str, Any] = {
            "source_id": source_id,
            "source_url": _require_https_url(
                mapping.get("source_url"), f"{field}.source_url"
            ),
            "title": _require_text(mapping.get("title"), f"{field}.title"),
            "published_at_utc": _format_utc(published),
            "updated_at_utc": _format_utc(updated),
            "fetched_at_utc": _format_utc(fetched),
            "observed_at_utc": _format_utc(observed),
            "content_sha256": content_sha,
        }
        for key in sorted(_SOURCE_OPTIONAL_KEYS - {"confidence"}):
            if key not in mapping:
                continue
            raw = mapping[key]
            if key in {"actual", "consensus", "value", "change"}:
                row[key] = _require_scalar(raw, f"{field}.{key}")
            else:
                row[key] = _require_text(raw, f"{field}.{key}")
        if "confidence" in mapping:
            row["confidence"] = _require_number(
                mapping.get("confidence"),
                f"{field}.confidence",
                minimum=0,
                maximum=1,
            )
        normalized.append(row)
    normalized.sort(
        key=lambda row: (
            row["published_at_utc"],
            row["updated_at_utc"],
            row["source_id"],
            row["content_sha256"],
        )
    )
    return normalized


def _packet_sha256(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "packet_sha256"}
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _assemble_trusted_packet(
    request: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    *,
    preflight: Mapping[str, Any],
    git_head: str,
    git_branch: str,
) -> dict[str, Any]:
    experiment_id = _require_id(
        request.get("experiment_id"), "trusted request experiment_id"
    )
    room_id = _require_id(request.get("room_id"), "trusted request room_id")
    candidate_id = _require_id(
        request.get("candidate_id"), "trusted request candidate_id"
    )
    spec_id = _require_id(request.get("spec_id"), "trusted request spec_id")
    policy_id = _require_id(request.get("policy_id"), "trusted request policy_id")

    session_contract = _trusted_document_mapping(sources, "session_contract")
    _require_source_identity(
        session_contract, "experiment_id", experiment_id, "session_contract"
    )
    _require_source_identity(session_contract, "room_id", room_id, "session_contract")
    candidate = _trusted_document_mapping(sources, "candidate")
    _require_source_identity(candidate, "candidate_id", candidate_id, "candidate")
    spec = _trusted_document_mapping(sources, "spec")
    _require_source_identity(spec, "spec_id", spec_id, "spec")
    policy = _trusted_document_mapping(sources, "policy")
    _require_source_identity(policy, "policy_id", policy_id, "policy")
    paper_eligible = _trusted_document_mapping(sources, "paper_eligible_event")
    _require_source_identity(
        paper_eligible, "candidate_id", candidate_id, "paper_eligible_event"
    )
    if paper_eligible.get("event_type") != "PAPER_ELIGIBLE":
        raise EvidencePacketIntegrityError(
            "paper_eligible_event is not a PAPER_ELIGIBLE lifecycle event"
        )
    ledger_sha256, ledger_observed_at_utc = _validate_virtual_broker_ledger_snapshot(
        sources["ledger"]["document"],
        room_id=room_id,
    )
    for role in ("state", "snapshot"):
        _require_source_identity(
            _trusted_document_mapping(sources, role), "room_id", room_id, role
        )

    position_raw = _trusted_document_mapping(sources, "position")
    expected_position_keys = _POSITION_KEYS - {"entry_context_sha256"}
    _require_exact_keys(position_raw, expected_position_keys, "trusted position")
    entry_context = _trusted_document_mapping(sources, "entry_context")
    if position_raw.get("pair") != entry_context.get("pair"):
        raise EvidencePacketIntegrityError(
            "entry_context pair does not match trusted position"
        )
    if position_raw.get("strategy_tag") != entry_context.get("strategy_tag"):
        raise EvidencePacketIntegrityError(
            "entry_context strategy_tag does not match trusted position"
        )
    position = {
        **position_raw,
        "entry_context_sha256": sources["entry_context"]["sha256"],
    }

    entry_signal_raw = sources["entry_signal"]["document"]
    if position_raw.get("side") in {"LONG", "SHORT"}:
        if entry_signal_raw is not None:
            raise EvidencePacketIntegrityError(
                "trusted open position entry_signal source must contain null"
            )
        entry_signal = None
    else:
        if not isinstance(entry_signal_raw, dict):
            raise EvidencePacketIntegrityError(
                "trusted FLAT entry_signal source must contain an object"
            )
        expected_signal_keys = _ENTRY_SIGNAL_KEYS - {
            "signal_identity_sha256",
            "entry_context_sha256",
        }
        _require_exact_keys(
            entry_signal_raw, expected_signal_keys, "trusted entry_signal"
        )
        entry_signal = {
            **entry_signal_raw,
            "entry_context_sha256": sources["entry_context"]["sha256"],
        }
        entry_signal["signal_identity_sha256"] = entry_signal_identity_sha256(
            entry_signal
        )

    quote_raw = _trusted_document_mapping(sources, "quote")
    _require_exact_keys(quote_raw, _QUOTE_KEYS - {"source_sha256"}, "trusted quote")
    quote = {**quote_raw, "source_sha256": sources["quote"]["sha256"]}

    candle_rows = _trusted_document_list(sources, "candles")
    candles: list[dict[str, Any]] = []
    for index, row in enumerate(candle_rows):
        if not isinstance(row, dict):
            raise EvidencePacketIntegrityError(
                f"trusted candles[{index}] must be an object"
            )
        _require_exact_keys(
            row,
            _CANDLE_KEYS - {"source_sha256"},
            f"trusted candles[{index}]",
        )
        candles.append({**row, "source_sha256": sources["candles"]["sha256"]})

    def trusted_source_rows(role: str) -> list[dict[str, Any]]:
        rows = _trusted_document_list(sources, role)
        result: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise EvidencePacketIntegrityError(
                    f"trusted {role}[{index}] must be an object"
                )
            keys = set(row)
            expected_required = _SOURCE_REQUIRED_KEYS - {"content_sha256"}
            missing = expected_required - keys
            unknown = keys - (_SOURCE_KEYS - {"content_sha256"})
            if missing or unknown:
                raise EvidencePacketIntegrityError(
                    f"trusted {role}[{index}] schema invalid; "
                    f"missing={sorted(missing)} unknown={sorted(unknown)}"
                )
            result.append({**row, "content_sha256": sources[role]["sha256"]})
        return result

    state = _trusted_document_mapping(sources, "state")
    snapshot = _trusted_document_mapping(sources, "snapshot")
    return {
        "contract": DOJO_AI_EVIDENCE_PACKET_CONTRACT,
        "cutoff_utc": request["cutoff_utc"],
        "bindings": {
            "launch_preflight_token_sha256": preflight["launch_preflight_token_sha256"],
            "git_head": git_head,
            "git_branch": git_branch,
            "canonical_source_root": DEDICATED_CANONICAL_SOURCE_ROOT.as_posix(),
            "experiment_id": experiment_id,
            "room_id": room_id,
            "session_contract_sha256": sources["session_contract"]["sha256"],
            "candidate_id": candidate_id,
            "candidate_sha256": sources["candidate"]["sha256"],
            "spec_id": spec_id,
            "spec_sha256": sources["spec"]["sha256"],
            "policy_id": policy_id,
            "policy_sha256": sources["policy"]["sha256"],
            "paper_eligible_tip_sha256": preflight[
                "candidate_lifecycle_ledger_tip_sha256"
            ],
            "ledger_sha256": ledger_sha256,
            "ledger_observed_at_utc": ledger_observed_at_utc,
            "state_sha256": sources["state"]["sha256"],
            "state_observed_at_utc": _required_source_timestamp(state, "state"),
            "snapshot_sha256": sources["snapshot"]["sha256"],
            "snapshot_observed_at_utc": _required_source_timestamp(
                snapshot, "snapshot"
            ),
        },
        "position": position,
        "entry_signal": entry_signal,
        "quote": quote,
        "candles": candles,
        "news_items": trusted_source_rows("news_items"),
        "calendar_items": trusted_source_rows("calendar_items"),
        "cross_asset_items": trusted_source_rows("cross_asset_items"),
        "dynamic_binding_max_age_seconds": request["dynamic_binding_max_age_seconds"],
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }


def _trusted_document_mapping(
    sources: Mapping[str, Mapping[str, Any]],
    role: str,
) -> dict[str, Any]:
    document = sources[role]["document"]
    if not isinstance(document, dict):
        raise EvidencePacketIntegrityError(
            f"trusted {role} source must contain an object"
        )
    return dict(document)


def _trusted_document_list(
    sources: Mapping[str, Mapping[str, Any]],
    role: str,
) -> list[Any]:
    document = sources[role]["document"]
    if not isinstance(document, list):
        raise EvidencePacketIntegrityError(
            f"trusted {role} source must contain an array"
        )
    return list(document)


def _require_source_identity(
    document: Mapping[str, Any],
    key: str,
    expected: str,
    role: str,
) -> None:
    if document.get(key) != expected:
        raise EvidencePacketIntegrityError(
            f"trusted {role} {key} does not match request scope"
        )


def _required_source_timestamp(document: Mapping[str, Any], role: str) -> str:
    value = document.get("observed_at_utc")
    _parse_utc(value, f"trusted {role}.observed_at_utc")
    assert isinstance(value, str)
    return value


def _read_trusted_canonical_source(
    root: Path,
    filename: object,
    *,
    cutoff: datetime,
    role: str,
) -> dict[str, Any]:
    if not isinstance(filename, str) or not re.fullmatch(
        r"[0-9a-f]{64}\.json", filename
    ):
        raise EvidencePacketIntegrityError(
            f"trusted {role} source filename must be content-addressed"
        )
    path = root / filename
    if path.parent != root:
        raise EvidencePacketIntegrityError(
            f"trusted {role} source escaped canonical root"
        )
    try:
        item_lstat = path.lstat()
    except OSError as exc:
        raise EvidencePacketIntegrityError(
            f"trusted {role} source is unavailable"
        ) from exc
    if stat.S_ISLNK(item_lstat.st_mode) or not stat.S_ISREG(item_lstat.st_mode):
        raise EvidencePacketIntegrityError(
            f"trusted {role} source must be a regular non-symlink file"
        )
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise EvidencePacketIntegrityError(
            f"trusted {role} source open failed"
        ) from exc
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > MAX_CANONICAL_SOURCE_BYTES
        ):
            raise EvidencePacketIntegrityError(
                f"trusted {role} source size/type is invalid"
            )
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            raw = handle.read(MAX_CANONICAL_SOURCE_BYTES + 1)
            after = os.fstat(handle.fileno())
    except EvidencePacketIntegrityError:
        raise
    except OSError as exc:
        raise EvidencePacketIntegrityError(
            f"trusted {role} source read failed"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
    if (
        len(raw) != before.st_size
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise EvidencePacketIntegrityError(
            f"trusted {role} source changed while reading"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if filename != f"{digest}.json":
        raise EvidencePacketIntegrityError(
            f"trusted {role} source digest does not match filename"
        )
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise EvidencePacketIntegrityError(
            f"trusted {role} source must be one canonical JSON document"
        )
    try:
        document = json.loads(
            raw[:-1],
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidencePacketIntegrityError(
            f"trusted {role} source JSON is invalid"
        ) from exc
    if _canonical_json_bytes(document) + b"\n" != raw:
        raise EvidencePacketIntegrityError(
            f"trusted {role} source JSON is noncanonical"
        )
    _reject_post_cutoff_source_timestamps(
        document, cutoff=cutoff, field=f"trusted {role}"
    )
    return {
        "document": document,
        "sha256": digest,
    }


def _reject_post_cutoff_source_timestamps(
    value: object,
    *,
    cutoff: datetime,
    field: str,
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{field}.{key}"
            if key.endswith("_utc") and item is not None:
                timestamp = _parse_utc(item, child)
                _require_not_after(timestamp, cutoff, child)
            _reject_post_cutoff_source_timestamps(item, cutoff=cutoff, field=child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_post_cutoff_source_timestamps(
                item, cutoff=cutoff, field=f"{field}[{index}]"
            )


def _dedicated_root(repository_root: Path | str, *, create: bool) -> Path:
    return _fixed_research_root(repository_root, DEDICATED_EVIDENCE_ROOT, create=create)


def _canonical_source_root(repository_root: Path | str) -> Path:
    return _fixed_research_root(
        repository_root, DEDICATED_CANONICAL_SOURCE_ROOT, create=False
    )


def _fixed_research_root(
    repository_root: Path | str,
    relative_root: Path,
    *,
    create: bool,
) -> Path:
    repo = Path(repository_root)
    if not repo.is_absolute():
        raise EvidencePacketIntegrityError("repository_root must be absolute")
    try:
        repo_stat = repo.lstat()
    except OSError as exc:
        raise EvidencePacketIntegrityError("repository_root is unavailable") from exc
    if stat.S_ISLNK(repo_stat.st_mode) or not stat.S_ISDIR(repo_stat.st_mode):
        raise EvidencePacketIntegrityError(
            "repository_root must be a real directory, not a symlink"
        )
    repo = repo.resolve(strict=True)
    current = repo
    for part in relative_root.parts:
        current = current / part
        try:
            item_stat = current.lstat()
        except FileNotFoundError:
            if not create:
                raise EvidencePacketIntegrityError(
                    "dedicated evidence root does not exist"
                )
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                item_stat = current.lstat()
                if stat.S_ISLNK(item_stat.st_mode) or not stat.S_ISDIR(
                    item_stat.st_mode
                ):
                    raise EvidencePacketIntegrityError(
                        "dedicated evidence root contains an unsafe component"
                    )
            else:
                _fsync_directory(current.parent)
                continue
        except OSError as exc:
            raise EvidencePacketIntegrityError(
                "dedicated evidence root is unavailable"
            ) from exc
        if stat.S_ISLNK(item_stat.st_mode) or not stat.S_ISDIR(item_stat.st_mode):
            raise EvidencePacketIntegrityError(
                "dedicated evidence root contains an unsafe component"
            )
    resolved = current.resolve(strict=True)
    if resolved != repo.joinpath(relative_root):
        raise EvidencePacketIntegrityError("dedicated evidence root escaped repository")
    return resolved


def _require_direct_packet_child(root: Path, path: Path) -> None:
    if not path.is_absolute():
        raise EvidencePacketIntegrityError("packet path must be absolute")
    try:
        item_stat = path.lstat()
    except OSError as exc:
        raise EvidencePacketIntegrityError("evidence packet is unavailable") from exc
    if stat.S_ISLNK(item_stat.st_mode) or not stat.S_ISREG(item_stat.st_mode):
        raise EvidencePacketIntegrityError(
            "evidence packet must be a direct regular file"
        )
    if path.parent.resolve(strict=True) != root:
        raise EvidencePacketIntegrityError(
            "evidence packet path escapes the dedicated root"
        )
    if not re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
        raise EvidencePacketIntegrityError("evidence packet filename is invalid")


def _read_exact_packet_file(path: Path) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise EvidencePacketIntegrityError("evidence packet is unavailable") from exc
    try:
        item_stat = os.fstat(fd)
        if not stat.S_ISREG(item_stat.st_mode):
            raise EvidencePacketIntegrityError("evidence packet is not a regular file")
        if item_stat.st_size <= 0 or item_stat.st_size > MAX_PACKET_BYTES:
            raise EvidencePacketIntegrityError("evidence packet size is invalid")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            raw = handle.read(MAX_PACKET_BYTES + 1)
    except EvidencePacketIntegrityError:
        raise
    except OSError as exc:
        raise EvidencePacketIntegrityError("evidence packet read failed") from exc
    finally:
        if fd >= 0:
            os.close(fd)
    if len(raw) != item_stat.st_size:
        raise EvidencePacketIntegrityError("evidence packet changed while reading")
    return raw


def _snapshot_mapping(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        snapshot = json.loads(
            raw,
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict JSON") from exc
    if not isinstance(snapshot, dict):
        raise TypeError(f"{field} must be a mapping")
    return snapshot


def _require_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return dict(value)


def _require_sequence(
    value: object,
    field: str,
    *,
    maximum: int,
) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    if len(value) > maximum:
        raise ValueError(f"{field} exceeds bounded row count")
    return list(value)


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    field: str,
) -> None:
    keys = set(value)
    if keys != expected:
        raise ValueError(
            f"{field} schema invalid; missing={sorted(expected - keys)} "
            f"unknown={sorted(keys - expected)}"
        )


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value or len(value) > 40:
        raise ValueError(f"{field} must be a bounded UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid timestamp") from exc
    return _require_aware_utc(parsed, field)


def _require_aware_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    utc = value.astimezone(timezone.utc)
    if utc.microsecond:
        return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return utc.isoformat(timespec="seconds").replace("+00:00", "Z")


def _nanoseconds_between(start: datetime, end: datetime) -> int:
    delta = end - start
    return (
        delta.days * 86_400 * 1_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
    )


def _require_not_after(value: datetime, cutoff: datetime, field: str) -> None:
    if value > cutoff:
        raise ValueError(f"{field} is after immutable cutoff")


def _require_fresh(
    value: datetime,
    cutoff: datetime,
    *,
    max_age_seconds: int,
    field: str,
) -> None:
    age_ns = _nanoseconds_between(value, cutoff)
    if age_ns < 0:
        raise ValueError(f"{field} is after immutable cutoff")
    if age_ns > max_age_seconds * 1_000_000_000:
        raise ValueError(f"{field} is stale")


def _require_market_open(value: datetime, field: str) -> None:
    try:
        status = compute_market_status(value)
    except Exception as exc:
        raise EvidencePacketError(f"FX market status unavailable for {field}") from exc
    if not status.is_fx_open:
        raise EvidencePacketMarketClosedError(
            f"new AI evidence packets are disabled while FX is closed ({field})"
        )


def _require_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError(f"{field} is invalid")
    return value


def _require_pair(value: object, field: str) -> str:
    if not isinstance(value, str) or not _PAIR_RE.fullmatch(value):
        raise ValueError(f"{field} is invalid")
    return value


def _require_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} is invalid")
    return value


def _require_git_head(value: object, field: str) -> str:
    if not isinstance(value, str) or not _GIT_HEAD_RE.fullmatch(value):
        raise ValueError(f"{field} is invalid")
    return value


def _require_codex_branch(value: object, field: str) -> str:
    branch = _require_id(value, field)
    if not branch.startswith("codex/"):
        raise ValueError(f"{field} must be under codex/")
    return branch


def _require_canonical_source_root_binding(value: object) -> str:
    expected = DEDICATED_CANONICAL_SOURCE_ROOT.as_posix()
    if value != expected:
        raise ValueError("bindings.canonical_source_root does not match the fixed root")
    return expected


def _require_int(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} is below minimum")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} exceeds maximum")
    return value


def _require_number(
    value: object,
    field: str,
    *,
    positive: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{field} must be positive")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} is below minimum")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} exceeds maximum")
    if isinstance(value, int):
        return value
    return 0.0 if number == 0 else number


def _require_exact_float_zero(value: object, field: str) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value != 0.0:
        raise ValueError(f"{field} must be exact float zero")


def _require_exact_int_zero(value: object, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != 0:
        raise ValueError(f"{field} must be exact integer zero")


def _optional_positive_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    return float(_require_number(value, field, positive=True))


def _require_text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_TEXT_CHARS
        or "\x00" in value
    ):
        raise ValueError(f"{field} is invalid")
    return value


def _require_scalar(value: object, field: str) -> str | int | float | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _require_text(value, field)
    return _require_number(value, field)


def _require_https_url(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_URL_CHARS:
        raise ValueError(f"{field} is invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(f"{field} must be a public HTTPS URL without credentials")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_unique_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise EvidencePacketError("evidence directory cannot be opened") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise EvidencePacketError("evidence directory fsync failed") from exc
    finally:
        os.close(fd)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

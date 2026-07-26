"""Fail-closed AI proposal producer for future paper inventory rooms.

The producer is deliberately dormant infrastructure.  It authenticates one
immutable evidence packet, constructs deterministic model-request bytes, and
normalizes one tightly bounded proposal.  It has no broker, runner, OANDA, or
order-authority dependency; its output is evidence for a separately gated
paper-only decision writer.

The authenticated external model adapter receives canonical UTF-8 JSON bytes
only.  It never receives an evidence filename, repository path, source handle,
or any unverified caller prose.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit import dojo_ai_evidence_packet as _evidence_packets


DOJO_AI_PROPOSAL_REQUEST_CONTRACT = "QR_DOJO_AI_INVENTORY_PROPOSAL_REQUEST_V2"
DOJO_AI_PRODUCER_RECEIPT_CONTRACT = "QR_DOJO_AI_INVENTORY_PRODUCER_RECEIPT_V1"
DOJO_AI_COMMAND_INVOKE_RECEIPT_CONTRACT = (
    "QR_DOJO_AI_INVENTORY_COMMAND_INVOKE_RECEIPT_V1"
)
DOJO_AI_SIGNED_MODEL_RESPONSE_CONTRACT = "QR_DOJO_AI_INVENTORY_SIGNED_MODEL_RESPONSE_V1"

AI_PROPOSAL_ACTIONS = frozenset(
    {
        "HOLD",
        "BLOCK_NEW",
        "ALLOW_NEW_VIRTUAL",
        "REDUCE_VIRTUAL",
        "CLOSE_VIRTUAL",
    }
)

MAX_MODEL_RESPONSE_BYTES = 64 * 1024
MAX_PRODUCER_RECEIPT_BYTES = 64 * 1024
MAX_REASON_CHARS = 2_000
MAX_ID_CHARS = 255
MAX_PRODUCER_EVIDENCE_AGE_SECONDS = 300
PRODUCER_RECEIPT_DIRECTORY = "producer_receipts"

_RESPONSE_KEYS = frozenset(
    {
        "action",
        "reason_code",
        "reason",
        "virtual_units",
        "confidence",
    }
)
_PRODUCER_RECEIPT_KEYS = frozenset(
    {
        "contract",
        "producer_id",
        "model_id",
        "evidence_packet_sha256",
        "request_sha256",
        "response_sha256",
        "action",
        "reason_code",
        "reason",
        "virtual_units",
        "confidence",
        "entry_signal_identity_sha256",
        "command_invoke_receipt",
        "produced_at_utc",
        "paper_only",
        "order_authority",
        "live_permission",
        "receipt_sha256",
    }
)
_COMMAND_MANIFEST_KEYS = frozenset(
    {
        "adapter_id",
        "model_id",
        "executable_path",
        "executable_sha256",
        "argv",
        "executor_uid",
        "executor_gid",
        "signature_key_id",
        "ed25519_public_key_base64",
        "timeout_seconds",
        "command_manifest_sha256",
    }
)
_COMMAND_INVOKE_RECEIPT_KEYS = frozenset(
    {
        "contract",
        "adapter_id",
        "model_id",
        "command_manifest_sha256",
        "executable_sha256",
        "executable_device",
        "executable_inode",
        "executor_uid",
        "executor_gid",
        "argv_sha256",
        "request_sha256",
        "response_sha256",
        "signed_response",
        "signature_key_id",
        "signature_base64",
        "signed_payload_sha256",
        "started_at_utc",
        "completed_at_utc",
        "exit_code",
        "invoke_receipt_sha256",
    }
)
_SIGNED_MODEL_RESPONSE_KEYS = frozenset(
    {
        "contract",
        "adapter_id",
        "model_id",
        "request_sha256",
        "response",
        "signature_key_id",
        "signature_base64",
    }
)
_LIFECYCLE_BINDING_KEYS = frozenset(
    {
        "adapter_id",
        "model_id",
        "config_sha256",
        "producer_id",
        "candidate_id",
        "experiment_id",
        "room_id",
        "future_window",
        "git_head",
        "launch_preflight_token_sha256",
    }
)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,254}$")
_REASON_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


_TRUSTED_COMMAND_ADAPTERS: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
_LOADED_COMMAND_ADAPTERS: dict[str, Mapping[str, Any]] = {}
_LOADED_COMMAND_ADAPTER_BINDINGS: dict[str, Mapping[str, Any]] = {}


class AllowlistedCommandModelAdapter:
    """Opaque reference to a command manifest pinned by trusted code.

    The reference contains no command, model id, path, or callback supplied by
    the caller.  An adapter id is usable only when it resolves through the
    module's explicit immutable allowlist.
    """

    __slots__ = ("_adapter_id",)

    def __init__(self, adapter_id: str) -> None:
        self._adapter_id = _require_id(adapter_id, "adapter_id")

    @property
    def adapter_id(self) -> str:
        return self._adapter_id


class AiInventoryProducerError(RuntimeError):
    """Base class for fail-closed producer failures."""


class AiInventoryProducerMarketClosedError(AiInventoryProducerError):
    """Model evaluation is disabled outside the DST-aware FX week."""


class AiInventoryProducerEvidenceError(AiInventoryProducerError):
    """The supplied evidence packet cannot be authenticated."""


class AiInventoryProducerModelError(AiInventoryProducerError):
    """The injected model failed before returning a valid response."""


class AiInventoryProducerResponseError(AiInventoryProducerError):
    """The model response is not a strict, bounded proposal object."""


class AiInventoryProducerReceiptError(AiInventoryProducerError):
    """A producer receipt cannot be written or authenticated."""


class AiInventoryProducerReceiptIntegrityError(AiInventoryProducerReceiptError):
    """A producer receipt, filename, or dedicated directory is untrusted."""


def command_adapter_manifest_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical digest used by the code-owned adapter allowlist."""

    manifest = _normalize_command_manifest(value, require_digest=False)
    return manifest["command_manifest_sha256"]


def load_sealed_command_model_adapter(
    adapter_id: str,
    config_sha256: str,
    *,
    experiment_id: str,
    room_id: str,
) -> AllowlistedCommandModelAdapter:
    """Load one Git/preflight-bound sidecar manifest from its fixed root.

    This is the only production loader.  It accepts no command, executable,
    key, config path, repository path, or model id from the caller.  A repeated
    load must resolve to the identical immutable manifest.
    """

    from quant_rabbit.dojo_ai_inventory_model_sidecar import (
        AiInventoryModelSidecarError,
        load_production_adapter_manifest,
    )

    requested = _require_id(adapter_id, "adapter_id")
    try:
        loaded = load_production_adapter_manifest(
            requested,
            config_sha256,
            experiment_id=experiment_id,
            room_id=room_id,
        )
    except AiInventoryModelSidecarError as exc:
        raise AiInventoryProducerModelError(
            "sealed sidecar command manifest could not be loaded"
        ) from exc
    if (
        not isinstance(loaded, Mapping)
        or set(loaded) != {"command_manifest", "lifecycle_binding"}
    ):
        raise AiInventoryProducerModelError(
            "sealed sidecar adapter registration schema is invalid"
        )
    manifest = _normalize_command_manifest(
        loaded["command_manifest"],
        require_digest=True,
    )
    lifecycle_binding = _normalize_lifecycle_binding(
        loaded["lifecycle_binding"]
    )
    if (
        manifest["adapter_id"] != lifecycle_binding["adapter_id"]
        or manifest["model_id"] != lifecycle_binding["model_id"]
        or lifecycle_binding["config_sha256"] != config_sha256
        or lifecycle_binding["experiment_id"] != experiment_id
        or lifecycle_binding["room_id"] != room_id
    ):
        raise AiInventoryProducerModelError(
            "sidecar manifest and lifecycle registration do not match"
        )
    existing = _LOADED_COMMAND_ADAPTERS.get(requested)
    if existing is not None and existing != manifest:
        raise AiInventoryProducerModelError(
            "loaded command adapter identity conflicts with prior manifest"
        )
    existing_binding = _LOADED_COMMAND_ADAPTER_BINDINGS.get(requested)
    if existing_binding is not None and existing_binding != lifecycle_binding:
        raise AiInventoryProducerModelError(
            "loaded command adapter lifecycle binding conflicts with prior registration"
        )
    _LOADED_COMMAND_ADAPTERS[requested] = MappingProxyType(manifest)
    _LOADED_COMMAND_ADAPTER_BINDINGS[requested] = MappingProxyType(
        lifecycle_binding
    )
    return AllowlistedCommandModelAdapter(requested)


def _normalize_lifecycle_binding(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AiInventoryProducerModelError(
            "sidecar lifecycle binding must be a mapping"
        )
    try:
        snapshot = json.loads(
            _canonical_json_bytes(value),
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AiInventoryProducerModelError(
            "sidecar lifecycle binding is not strict JSON"
        ) from exc
    if set(snapshot) != _LIFECYCLE_BINDING_KEYS:
        raise AiInventoryProducerModelError(
            "sidecar lifecycle binding schema is invalid"
        )
    window = snapshot.get("future_window")
    if not isinstance(window, Mapping) or set(window) != {"start_utc", "end_utc"}:
        raise AiInventoryProducerModelError(
            "sidecar lifecycle future window is invalid"
        )
    start = _model_utc(window.get("start_utc"), "future_window.start_utc")
    end = _model_utc(window.get("end_utc"), "future_window.end_utc")
    if start >= end:
        raise AiInventoryProducerModelError(
            "sidecar lifecycle future window is empty or reversed"
        )
    git_head = snapshot.get("git_head")
    if not isinstance(git_head, str) or not re.fullmatch(r"[0-9a-f]{40}", git_head):
        raise AiInventoryProducerModelError(
            "sidecar lifecycle Git HEAD is invalid"
        )
    return {
        "adapter_id": _model_id(snapshot.get("adapter_id"), "adapter_id"),
        "model_id": _model_id(snapshot.get("model_id"), "model_id"),
        "config_sha256": _model_sha(
            snapshot.get("config_sha256"), "config_sha256"
        ),
        "producer_id": _model_id(snapshot.get("producer_id"), "producer_id"),
        "candidate_id": _model_sha(
            snapshot.get("candidate_id"), "candidate_id"
        ),
        "experiment_id": _model_id(
            snapshot.get("experiment_id"), "experiment_id"
        ),
        "room_id": _model_id(snapshot.get("room_id"), "room_id"),
        "future_window": {
            "start_utc": window["start_utc"],
            "end_utc": window["end_utc"],
        },
        "git_head": git_head,
        "launch_preflight_token_sha256": _model_sha(
            snapshot.get("launch_preflight_token_sha256"),
            "launch_preflight_token_sha256",
        ),
    }


def _require_loaded_lifecycle_binding(
    *,
    adapter_id: str,
    producer_id: str,
    model_id: str,
    packet: Mapping[str, Any],
) -> None:
    binding = _LOADED_COMMAND_ADAPTER_BINDINGS.get(adapter_id)
    if binding is None:
        if adapter_id in _LOADED_COMMAND_ADAPTERS:
            raise AiInventoryProducerModelError(
                "production adapter has no lifecycle preregistration"
            )
        return
    packet_bindings = packet.get("bindings")
    if not isinstance(packet_bindings, Mapping):
        raise AiInventoryProducerEvidenceError(
            "evidence packet bindings are unavailable"
        )
    expected = {
        "producer_id": producer_id,
        "model_id": model_id,
        "experiment_id": packet_bindings.get("experiment_id"),
        "room_id": packet_bindings.get("room_id"),
        "candidate_id": packet_bindings.get("candidate_id"),
        "git_head": packet_bindings.get("git_head"),
        "launch_preflight_token_sha256": packet_bindings.get(
            "launch_preflight_token_sha256"
        ),
    }
    for field, observed in expected.items():
        if binding[field] != observed:
            raise AiInventoryProducerEvidenceError(
                f"evidence packet {field} does not match lifecycle preregistration"
            )
    cutoff = _model_utc(packet.get("cutoff_utc"), "packet cutoff_utc")
    start = _model_utc(
        binding["future_window"]["start_utc"],
        "future_window.start_utc",
    )
    end = _model_utc(
        binding["future_window"]["end_utc"],
        "future_window.end_utc",
    )
    if cutoff < start or cutoff >= end:
        raise AiInventoryProducerEvidenceError(
            "evidence packet cutoff is outside lifecycle future window"
        )


def produce_ai_inventory_proposal(
    evidence_packet: Mapping[str, Any] | bytes,
    adapter: AllowlistedCommandModelAdapter,
    *,
    producer_id: str,
    room_root: Path,
) -> dict[str, Any]:
    """Produce one authenticated paper-only AI inventory proposal.

    The internal wall clock is checked before packet processing or model
    invocation.  Tests may patch :func:`_utc_now`; callers cannot backdate the
    decision.  Only an executable and argv pinned in the code-owned command
    allowlist may receive canonical request bytes.  A durable producer receipt
    is written before any proposal is returned.
    """

    producer = _require_id(producer_id, "producer_id")
    if type(adapter) is not AllowlistedCommandModelAdapter:
        raise AiInventoryProducerModelError(
            "model adapter must be the exact allowlisted command adapter type"
        )
    preflight_at = _utc_now().astimezone(timezone.utc)
    _require_market_open(preflight_at)
    command_manifest = _trusted_command_manifest(adapter.adapter_id)
    model_name = command_manifest["model_id"]

    verified = _verify_packet_value(evidence_packet)
    _require_not_future(verified, preflight_at)
    _require_loaded_lifecycle_binding(
        adapter_id=adapter.adapter_id,
        producer_id=producer,
        model_id=model_name,
        packet=verified,
    )
    watermarks = _source_watermarks(verified)
    watermarks_sha256 = _sha256(_canonical_json_bytes(watermarks))

    request = {
        "contract": DOJO_AI_PROPOSAL_REQUEST_CONTRACT,
        "producer_id": producer,
        "model_id": model_name,
        "purpose": "PAPER_AI_INVENTORY_PROPOSAL_ONLY",
        "evidence_packet": verified,
        "source_watermarks": watermarks,
        "source_watermarks_sha256": watermarks_sha256,
        "required_response": {
            "exact_keys": sorted(_RESPONSE_KEYS),
            "actions": sorted(AI_PROPOSAL_ACTIONS),
            "virtual_units": (
                "null for HOLD/BLOCK_NEW/ALLOW_NEW_VIRTUAL; finite positive "
                "number bounded by observed position units for REDUCE/CLOSE"
            ),
            "confidence": "finite number in [0,1]",
        },
        "safety": {
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "proposal_is_not_an_action": True,
            "arbitrary_prose_has_no_authority": True,
        },
    }
    request_bytes = _canonical_json_bytes(request)
    request_sha256 = _sha256(request_bytes)
    raw_response, command_invoke_receipt, produced_at = _invoke_allowlisted_command(
        command_manifest,
        request_bytes=request_bytes,
        request_sha256=request_sha256,
    )
    _require_market_open(produced_at)
    _require_not_future(verified, produced_at)

    response = _parse_response(raw_response)
    normalized = _normalize_response(response, verified)
    response_sha256 = _sha256(raw_response)
    if command_invoke_receipt["response_sha256"] != response_sha256:
        raise AiInventoryProducerModelError("command invoke response digest mismatch")
    produced_at_utc = _format_utc(produced_at)
    entry_signal = verified.get("entry_signal")
    entry_signal_identity = (
        entry_signal["signal_identity_sha256"]
        if isinstance(entry_signal, Mapping)
        else None
    )
    producer_receipt = _seal_producer_receipt(
        {
            "contract": DOJO_AI_PRODUCER_RECEIPT_CONTRACT,
            "producer_id": producer,
            "model_id": model_name,
            "evidence_packet_sha256": verified["packet_sha256"],
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
            **normalized,
            "entry_signal_identity_sha256": entry_signal_identity,
            "command_invoke_receipt": command_invoke_receipt,
            "produced_at_utc": produced_at_utc,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        },
        require_digest=False,
    )
    _write_ai_inventory_producer_receipt(room_root, producer_receipt)

    return {
        **normalized,
        "ai_decision_binding": {
            "producer_id": producer,
            "model_id": model_name,
            "evidence_packet_sha256": verified["packet_sha256"],
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
            "observed_at_utc": verified["cutoff_utc"],
            "producer_receipt_sha256": producer_receipt["receipt_sha256"],
            "produced_at_utc": produced_at_utc,
        },
        "producer_receipt": producer_receipt,
    }


def _trusted_command_manifest(adapter_id: str) -> dict[str, Any]:
    configured = _TRUSTED_COMMAND_ADAPTERS.get(adapter_id)
    if configured is None:
        configured = _LOADED_COMMAND_ADAPTERS.get(adapter_id)
    if configured is None:
        raise AiInventoryProducerModelError(
            "model adapter is not present in the trusted command allowlist"
        )
    manifest = _normalize_command_manifest(configured, require_digest=True)
    if manifest["adapter_id"] != adapter_id:
        raise AiInventoryProducerModelError("trusted command adapter id mismatch")
    _authenticated_executable(manifest)
    return manifest


def _normalize_command_manifest(
    value: Mapping[str, Any],
    *,
    require_digest: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AiInventoryProducerModelError(
            "trusted command manifest must be a mapping"
        )
    try:
        snapshot = json.loads(
            _canonical_json_bytes(value),
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AiInventoryProducerModelError(
            "trusted command manifest is not strict JSON"
        ) from exc
    expected = (
        _COMMAND_MANIFEST_KEYS
        if require_digest
        else _COMMAND_MANIFEST_KEYS - {"command_manifest_sha256"}
    )
    if set(snapshot) != expected:
        raise AiInventoryProducerModelError(
            "trusted command manifest schema is invalid"
        )
    adapter_id = _model_id(snapshot.get("adapter_id"), "adapter_id")
    model_id = _model_id(snapshot.get("model_id"), "model_id")
    executable_path = snapshot.get("executable_path")
    if (
        not isinstance(executable_path, str)
        or not executable_path
        or len(executable_path) > 4_096
        or "\x00" in executable_path
        or not Path(executable_path).is_absolute()
    ):
        raise AiInventoryProducerModelError(
            "trusted command executable_path is invalid"
        )
    executable_sha256 = _model_sha(
        snapshot.get("executable_sha256"), "executable_sha256"
    )
    raw_argv = snapshot.get("argv")
    if (
        not isinstance(raw_argv, list)
        or not 1 <= len(raw_argv) <= 64
        or raw_argv[0] != executable_path
    ):
        raise AiInventoryProducerModelError("trusted command argv is invalid")
    argv: list[str] = []
    for item in raw_argv:
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 16_384
            or "\x00" in item
        ):
            raise AiInventoryProducerModelError(
                "trusted command argv contains an invalid item"
            )
        argv.append(item)
    executor_uid = _bounded_exact_int(
        snapshot.get("executor_uid"), "executor_uid", minimum=0
    )
    executor_gid = _bounded_exact_int(
        snapshot.get("executor_gid"), "executor_gid", minimum=0
    )
    signature_key_id = _model_id(snapshot.get("signature_key_id"), "signature_key_id")
    public_key_base64 = _canonical_ed25519_public_key(
        snapshot.get("ed25519_public_key_base64")
    )
    timeout_seconds = _bounded_exact_int(
        snapshot.get("timeout_seconds"),
        "timeout_seconds",
        minimum=1,
        maximum=300,
    )
    normalized: dict[str, Any] = {
        "adapter_id": adapter_id,
        "model_id": model_id,
        "executable_path": executable_path,
        "executable_sha256": executable_sha256,
        "argv": argv,
        "executor_uid": executor_uid,
        "executor_gid": executor_gid,
        "signature_key_id": signature_key_id,
        "ed25519_public_key_base64": public_key_base64,
        "timeout_seconds": timeout_seconds,
    }
    digest = _sha256(_canonical_json_bytes(normalized))
    if require_digest:
        supplied = _model_sha(
            snapshot.get("command_manifest_sha256"),
            "command_manifest_sha256",
        )
        if supplied != digest:
            raise AiInventoryProducerModelError(
                "trusted command manifest digest mismatch"
            )
    normalized["command_manifest_sha256"] = digest
    return normalized


def _authenticated_executable(manifest: Mapping[str, Any]) -> os.stat_result:
    path = Path(manifest["executable_path"])
    try:
        item_stat = path.lstat()
    except OSError as exc:
        raise AiInventoryProducerModelError(
            "trusted command executable is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(item_stat.st_mode)
        or not stat.S_ISREG(item_stat.st_mode)
        or not item_stat.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    ):
        raise AiInventoryProducerModelError(
            "trusted command executable is not a real executable file"
        )
    if (
        item_stat.st_uid != manifest["executor_uid"]
        or item_stat.st_gid != manifest["executor_gid"]
    ):
        raise AiInventoryProducerModelError("trusted command executable owner mismatch")
    digest, opened_stat = _digest_regular_file(path)
    if (
        digest != manifest["executable_sha256"]
        or opened_stat.st_dev != item_stat.st_dev
        or opened_stat.st_ino != item_stat.st_ino
    ):
        raise AiInventoryProducerModelError(
            "trusted command executable digest or identity mismatch"
        )
    return opened_stat


def _invoke_allowlisted_command(
    manifest: Mapping[str, Any],
    *,
    request_bytes: bytes,
    request_sha256: str,
) -> tuple[bytes, dict[str, Any], datetime]:
    started_at = _utc_now().astimezone(timezone.utc)
    _require_market_open(started_at)
    before = _authenticated_executable(manifest)
    try:
        completed = subprocess.run(
            list(manifest["argv"]),
            input=request_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=manifest["timeout_seconds"],
            check=False,
            cwd="/",
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AiInventoryProducerModelError(
            "allowlisted command model invocation failed"
        ) from exc
    completed_at = _utc_now().astimezone(timezone.utc)
    after = _authenticated_executable(manifest)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_size != after.st_size
    ):
        raise AiInventoryProducerModelError(
            "trusted command executable changed during invocation"
        )
    if completed.returncode != 0:
        raise AiInventoryProducerModelError(
            "allowlisted command model exited unsuccessfully"
        )
    raw_envelope = bytes(completed.stdout)
    if not raw_envelope or len(raw_envelope) > MAX_MODEL_RESPONSE_BYTES:
        raise AiInventoryProducerModelError(
            "allowlisted command model response size is invalid"
        )
    (
        signed_response,
        signature_key_id,
        signature_base64,
        signed_payload_sha256,
    ) = _verified_signed_model_response(
        raw_envelope,
        manifest=manifest,
        request_sha256=request_sha256,
    )
    raw_response = _canonical_json_bytes(signed_response)
    response_sha256 = _sha256(raw_response)
    invoke_body: dict[str, Any] = {
        "contract": DOJO_AI_COMMAND_INVOKE_RECEIPT_CONTRACT,
        "adapter_id": manifest["adapter_id"],
        "model_id": manifest["model_id"],
        "command_manifest_sha256": manifest["command_manifest_sha256"],
        "executable_sha256": manifest["executable_sha256"],
        "executable_device": before.st_dev,
        "executable_inode": before.st_ino,
        "executor_uid": before.st_uid,
        "executor_gid": before.st_gid,
        "argv_sha256": _sha256(_canonical_json_bytes(manifest["argv"])),
        "request_sha256": request_sha256,
        "response_sha256": response_sha256,
        "signed_response": signed_response,
        "signature_key_id": signature_key_id,
        "signature_base64": signature_base64,
        "signed_payload_sha256": signed_payload_sha256,
        "started_at_utc": _format_utc(started_at),
        "completed_at_utc": _format_utc(completed_at),
        "exit_code": 0,
    }
    invoke_body["invoke_receipt_sha256"] = _sha256(_canonical_json_bytes(invoke_body))
    return raw_response, invoke_body, completed_at


def _verified_signed_model_response(
    raw: bytes,
    *,
    manifest: Mapping[str, Any],
    request_sha256: str,
) -> tuple[dict[str, Any], str, str, str]:
    try:
        envelope = json.loads(
            raw,
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AiInventoryProducerModelError(
            "allowlisted command returned an invalid signed response envelope"
        ) from exc
    if not isinstance(envelope, dict) or set(envelope) != _SIGNED_MODEL_RESPONSE_KEYS:
        raise AiInventoryProducerModelError(
            "signed model response envelope schema is invalid"
        )
    if raw != _canonical_json_bytes(envelope):
        raise AiInventoryProducerModelError(
            "signed model response envelope bytes are not canonical"
        )
    if (
        envelope.get("contract") != DOJO_AI_SIGNED_MODEL_RESPONSE_CONTRACT
        or envelope.get("adapter_id") != manifest["adapter_id"]
        or envelope.get("model_id") != manifest["model_id"]
        or envelope.get("request_sha256") != request_sha256
        or envelope.get("signature_key_id") != manifest["signature_key_id"]
    ):
        raise AiInventoryProducerModelError("signed model response binding mismatch")
    response = _parse_response(envelope.get("response"))
    signature_base64 = _canonical_signature_base64(envelope.get("signature_base64"))
    signed_body = {
        key: envelope[key]
        for key in _SIGNED_MODEL_RESPONSE_KEYS
        if key != "signature_base64"
    }
    signed_payload = _canonical_json_bytes(signed_body)
    _verify_ed25519_signature(
        public_key_base64=manifest["ed25519_public_key_base64"],
        signature_base64=signature_base64,
        payload=signed_payload,
    )
    return (
        response,
        manifest["signature_key_id"],
        signature_base64,
        _sha256(signed_payload),
    )


def _write_ai_inventory_producer_receipt(
    room_root: Path,
    receipt: Mapping[str, Any],
) -> Path:
    """Exclusively persist a receipt at its internally derived destination."""

    sealed = _seal_producer_receipt(receipt, require_digest=True)
    raw = _canonical_json_bytes(sealed) + b"\n"
    if len(raw) > MAX_PRODUCER_RECEIPT_BYTES:
        raise AiInventoryProducerReceiptError(
            "producer receipt exceeds the fixed size bound"
        )
    receipt_root = _producer_receipt_root(room_root, create=True)
    path = receipt_root / f"{sealed['receipt_sha256']}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        existing = _read_receipt_file(path)
        if existing != raw:
            raise AiInventoryProducerReceiptIntegrityError(
                "existing producer receipt is tampered or collides"
            )
        verify_ai_inventory_producer_receipt(room_root, path)
        return path
    except OSError as exc:
        raise AiInventoryProducerReceiptError(
            "exclusive producer receipt create failed"
        ) from exc

    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(receipt_root)
    except Exception:
        try:
            path.unlink()
            _fsync_directory(receipt_root)
        except OSError:
            pass
        raise
    return path


def verify_ai_inventory_producer_receipt(
    room_root: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Authenticate one direct content-addressed producer receipt file."""

    root = _producer_receipt_root(room_root, create=False)
    path = _require_direct_receipt_child(root, receipt_path)
    raw = _read_receipt_file(path)
    if (
        not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
        or len(raw) > MAX_PRODUCER_RECEIPT_BYTES
    ):
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt must be one bounded canonical JSON row"
        )
    try:
        value = json.loads(
            raw[:-1],
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt JSON is invalid"
        ) from exc
    try:
        sealed = _seal_producer_receipt(value, require_digest=True)
    except (TypeError, ValueError, AiInventoryProducerReceiptError) as exc:
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt semantic verification failed"
        ) from exc
    expected = _canonical_json_bytes(sealed) + b"\n"
    if raw != expected:
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt is noncanonical"
        )
    if path.name != f"{sealed['receipt_sha256']}.json":
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt filename does not match its digest"
        )
    return json.loads(_canonical_json_bytes(sealed))


def _verify_packet_value(
    value: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Re-run the public packet verifier without exposing a path downstream."""

    raw = _packet_bytes(value)
    try:
        locator = json.loads(
            raw[:-1],
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AiInventoryProducerEvidenceError(
            "evidence packet JSON is invalid"
        ) from exc
    if not isinstance(locator, dict):
        raise AiInventoryProducerEvidenceError("evidence packet must be a JSON object")
    packet_sha256 = locator.get("packet_sha256")
    if not isinstance(packet_sha256, str) or not _SHA256_RE.fullmatch(packet_sha256):
        raise AiInventoryProducerEvidenceError(
            "evidence packet digest locator is invalid"
        )

    try:
        with tempfile.TemporaryDirectory(
            prefix="qr-dojo-ai-packet-verify-"
        ) as temporary:
            repository_root = Path(temporary).resolve()
            packet_root = repository_root / _evidence_packets.DEDICATED_EVIDENCE_ROOT
            packet_root.mkdir(parents=True, mode=0o700)
            packet_path = packet_root / f"{packet_sha256}.json"
            packet_path.write_bytes(raw)
            packet_path.chmod(0o600)
            verified = _evidence_packets.verify_ai_inventory_evidence_packet(
                repository_root, packet_path
            )
    except Exception as exc:
        raise AiInventoryProducerEvidenceError(
            "evidence packet authentication failed"
        ) from exc
    if not isinstance(verified, dict):
        raise AiInventoryProducerEvidenceError(
            "evidence verifier returned an invalid object"
        )
    return verified


def _packet_bytes(value: Mapping[str, Any] | bytes) -> bytes:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, Mapping):
        try:
            raw = _canonical_json_bytes(value) + b"\n"
        except (TypeError, ValueError) as exc:
            raise AiInventoryProducerEvidenceError(
                "evidence packet object is not canonical JSON"
            ) from exc
    else:
        raise TypeError("evidence_packet must be a mapping or bytes")
    if (
        not raw
        or len(raw) > _evidence_packets.MAX_PACKET_BYTES
        or not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
    ):
        raise AiInventoryProducerEvidenceError(
            "evidence packet bytes are not one bounded canonical JSON row"
        )
    return raw


def _require_not_future(packet: Mapping[str, Any], produced_at: datetime) -> None:
    parsed_fields: dict[str, datetime] = {}
    for field in ("cutoff_utc", "sealed_at_utc"):
        value = packet.get(field)
        if not isinstance(value, str):
            raise AiInventoryProducerEvidenceError(
                f"verified packet {field} is invalid"
            )
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise AiInventoryProducerEvidenceError(
                f"verified packet {field} is invalid"
            ) from exc
        if parsed.tzinfo is None or parsed.astimezone(timezone.utc) > produced_at:
            raise AiInventoryProducerEvidenceError(
                f"verified packet {field} is after producer clock"
            )
        parsed_fields[field] = parsed.astimezone(timezone.utc)
    if (
        produced_at - parsed_fields["cutoff_utc"]
    ).total_seconds() > MAX_PRODUCER_EVIDENCE_AGE_SECONDS:
        raise AiInventoryProducerEvidenceError(
            "verified packet is stale at producer clock"
        )


def _source_watermarks(packet: Mapping[str, Any]) -> dict[str, Any]:
    bindings = packet["bindings"]
    return {
        "cutoff_utc": packet["cutoff_utc"],
        "sealed_at_utc": packet["sealed_at_utc"],
        "ledger": {
            "sha256": bindings["ledger_sha256"],
            "observed_at_utc": bindings["ledger_observed_at_utc"],
        },
        "state": {
            "sha256": bindings["state_sha256"],
            "observed_at_utc": bindings["state_observed_at_utc"],
        },
        "snapshot": {
            "sha256": bindings["snapshot_sha256"],
            "observed_at_utc": bindings["snapshot_observed_at_utc"],
        },
        "quote": {
            "source_sha256": packet["quote"]["source_sha256"],
            "timestamp_utc": packet["quote"]["timestamp_utc"],
        },
        "entry_signal": (
            None
            if packet["entry_signal"] is None
            else {
                "signal_identity_sha256": packet["entry_signal"][
                    "signal_identity_sha256"
                ],
                "observed_at_utc": packet["entry_signal"]["observed_at_utc"],
            }
        ),
        "candles": [
            {
                "source_sha256": row["source_sha256"],
                "completed_at_utc": row["completed_at_utc"],
            }
            for row in packet["candles"]
        ],
        "news": [
            {
                "source_id": row["source_id"],
                "content_sha256": row["content_sha256"],
                "observed_at_utc": row["observed_at_utc"],
            }
            for row in packet["news_items"]
        ],
        "calendar": [
            {
                "source_id": row["source_id"],
                "content_sha256": row["content_sha256"],
                "observed_at_utc": row["observed_at_utc"],
            }
            for row in packet["calendar_items"]
        ],
        "cross_assets": [
            {
                "source_id": row["source_id"],
                "content_sha256": row["content_sha256"],
                "observed_at_utc": row["observed_at_utc"],
            }
            for row in packet["cross_asset_items"]
        ],
    }


def _parse_response(
    value: bytes | str | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, bytes):
        if not value or len(value) > MAX_MODEL_RESPONSE_BYTES:
            raise AiInventoryProducerResponseError(
                "model response bytes exceed the fixed bound"
            )
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        if not raw or len(raw) > MAX_MODEL_RESPONSE_BYTES:
            raise AiInventoryProducerResponseError(
                "model response text exceeds the fixed bound"
            )
    elif isinstance(value, Mapping):
        try:
            raw = _canonical_json_bytes(value)
        except (TypeError, ValueError) as exc:
            raise AiInventoryProducerResponseError(
                "model response mapping is not strict JSON"
            ) from exc
    else:
        raise AiInventoryProducerResponseError("model response must be a JSON object")
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AiInventoryProducerResponseError(
            "model response is not valid strict JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise AiInventoryProducerResponseError("model response must be a JSON object")
    if set(parsed) != _RESPONSE_KEYS:
        raise AiInventoryProducerResponseError("model response schema is invalid")
    return parsed


def _normalize_response(
    response: Mapping[str, Any],
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    action = response.get("action")
    if not isinstance(action, str) or action not in AI_PROPOSAL_ACTIONS:
        raise AiInventoryProducerResponseError("model action is invalid")

    reason_code = response.get("reason_code")
    if not isinstance(reason_code, str) or not _REASON_CODE_RE.fullmatch(reason_code):
        raise AiInventoryProducerResponseError("reason_code is invalid")

    reason = response.get("reason")
    if (
        not isinstance(reason, str)
        or not reason.strip()
        or len(reason) > MAX_REASON_CHARS
        or any(ord(character) < 32 for character in reason)
    ):
        raise AiInventoryProducerResponseError("reason is invalid")

    observed_side = packet["position"]["side"]
    observed_units = _finite_number(
        packet["position"]["units"], "observed position units"
    )
    entry_signal = packet.get("entry_signal")
    if action in {"BLOCK_NEW", "ALLOW_NEW_VIRTUAL"}:
        if observed_side != "FLAT" or not isinstance(entry_signal, Mapping):
            raise AiInventoryProducerResponseError(
                f"{action} requires flat evidence with an authenticated entry signal"
            )
    elif observed_side == "FLAT":
        raise AiInventoryProducerResponseError(
            f"{action} requires an authenticated open position"
        )
    if action in {"HOLD", "BLOCK_NEW", "ALLOW_NEW_VIRTUAL"}:
        if response.get("virtual_units") is not None:
            raise AiInventoryProducerResponseError(
                f"{action} requires virtual_units=null"
            )
        units: float | None = None
    else:
        units = _finite_number(response.get("virtual_units"), "virtual_units")
        if units <= 0:
            raise AiInventoryProducerResponseError(
                "inventory mutation virtual_units must be positive"
            )
    if action == "REDUCE_VIRTUAL":
        assert units is not None
        if not 0 < units < observed_units:
            raise AiInventoryProducerResponseError(
                "REDUCE_VIRTUAL units must be positive and below observed units"
            )
    elif action == "CLOSE_VIRTUAL" and units != observed_units:
        raise AiInventoryProducerResponseError(
            "CLOSE_VIRTUAL units must equal observed position units"
        )

    confidence = _finite_number(response.get("confidence"), "confidence")
    if not 0 <= confidence <= 1:
        raise AiInventoryProducerResponseError("confidence must be in [0,1]")

    return {
        "action": action,
        "reason_code": reason_code,
        "reason": reason,
        "virtual_units": None if units is None else float(units),
        "confidence": float(confidence),
    }


def _seal_producer_receipt(
    value: Mapping[str, Any],
    *,
    require_digest: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("producer receipt must be a mapping")
    try:
        snapshot = json.loads(
            _canonical_json_bytes(value),
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AiInventoryProducerReceiptError(
            "producer receipt is not strict JSON"
        ) from exc
    expected_keys = (
        _PRODUCER_RECEIPT_KEYS
        if require_digest
        else _PRODUCER_RECEIPT_KEYS - {"receipt_sha256"}
    )
    if set(snapshot) != expected_keys:
        raise AiInventoryProducerReceiptError("producer receipt schema is invalid")
    if snapshot.get("contract") != DOJO_AI_PRODUCER_RECEIPT_CONTRACT:
        raise AiInventoryProducerReceiptError("producer receipt contract is invalid")
    producer_id = _receipt_id(snapshot.get("producer_id"), "producer_id")
    model_id = _receipt_id(snapshot.get("model_id"), "model_id")
    evidence_sha = _receipt_sha(
        snapshot.get("evidence_packet_sha256"),
        "evidence_packet_sha256",
    )
    request_sha = _receipt_sha(snapshot.get("request_sha256"), "request_sha256")
    response_sha = _receipt_sha(snapshot.get("response_sha256"), "response_sha256")
    action = snapshot.get("action")
    if not isinstance(action, str) or action not in AI_PROPOSAL_ACTIONS:
        raise AiInventoryProducerReceiptError("producer receipt action is invalid")
    reason_code = snapshot.get("reason_code")
    if (
        not isinstance(reason_code, str)
        or _REASON_CODE_RE.fullmatch(reason_code) is None
    ):
        raise AiInventoryProducerReceiptError("producer receipt reason_code is invalid")
    reason = snapshot.get("reason")
    if (
        not isinstance(reason, str)
        or not reason.strip()
        or len(reason) > MAX_REASON_CHARS
        or any(ord(character) < 32 for character in reason)
    ):
        raise AiInventoryProducerReceiptError("producer receipt reason is invalid")

    raw_units = snapshot.get("virtual_units")
    if action in {"HOLD", "BLOCK_NEW", "ALLOW_NEW_VIRTUAL"}:
        if raw_units is not None:
            raise AiInventoryProducerReceiptError(
                "non-mutating receipt virtual_units must be null"
            )
        virtual_units: float | None = None
    else:
        if raw_units.__class__ is not float:
            raise AiInventoryProducerReceiptError(
                "mutating receipt virtual_units must be a canonical float"
            )
        virtual_units = _receipt_finite_number(
            raw_units, "virtual_units", positive=True
        )

    raw_confidence = snapshot.get("confidence")
    if raw_confidence.__class__ is not float:
        raise AiInventoryProducerReceiptError(
            "producer receipt confidence must be a canonical float"
        )
    confidence = _receipt_finite_number(
        raw_confidence, "confidence", minimum=0.0, maximum=1.0
    )

    raw_signal_identity = snapshot.get("entry_signal_identity_sha256")
    if action in {"BLOCK_NEW", "ALLOW_NEW_VIRTUAL"}:
        entry_signal_identity: str | None = _receipt_sha(
            raw_signal_identity, "entry_signal_identity_sha256"
        )
    else:
        if raw_signal_identity is not None:
            raise AiInventoryProducerReceiptError(
                "open-position receipt must not bind an entry signal"
            )
        entry_signal_identity = None

    command_invoke_receipt = _normalize_command_invoke_receipt(
        snapshot.get("command_invoke_receipt"),
        producer_model_id=model_id,
        request_sha256=request_sha,
        response_sha256=response_sha,
        normalized_response={
            "action": action,
            "reason_code": reason_code,
            "reason": reason,
            "virtual_units": virtual_units,
            "confidence": confidence,
        },
    )
    produced_at = _receipt_utc(snapshot.get("produced_at_utc"), "produced_at_utc")
    if command_invoke_receipt["completed_at_utc"] != produced_at:
        raise AiInventoryProducerReceiptError(
            "producer receipt clock does not match command completion"
        )
    if snapshot.get("paper_only") is not True:
        raise AiInventoryProducerReceiptError(
            "producer receipt must be paper_only=true"
        )
    if snapshot.get("order_authority") != "NONE":
        raise AiInventoryProducerReceiptError(
            "producer receipt order_authority must be NONE"
        )
    if snapshot.get("live_permission") is not False:
        raise AiInventoryProducerReceiptError(
            "producer receipt live_permission must be false"
        )

    normalized: dict[str, Any] = {
        "contract": DOJO_AI_PRODUCER_RECEIPT_CONTRACT,
        "producer_id": producer_id,
        "model_id": model_id,
        "evidence_packet_sha256": evidence_sha,
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        "action": action,
        "reason_code": reason_code,
        "reason": reason,
        "virtual_units": virtual_units,
        "confidence": confidence,
        "entry_signal_identity_sha256": entry_signal_identity,
        "command_invoke_receipt": command_invoke_receipt,
        "produced_at_utc": produced_at,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    digest = _sha256(_canonical_json_bytes(normalized))
    if require_digest:
        supplied_digest = _receipt_sha(snapshot.get("receipt_sha256"), "receipt_sha256")
        if supplied_digest != digest:
            raise AiInventoryProducerReceiptIntegrityError(
                "producer receipt digest mismatch"
            )
    normalized["receipt_sha256"] = digest
    return normalized


def _normalize_command_invoke_receipt(
    value: object,
    *,
    producer_model_id: str,
    request_sha256: str,
    response_sha256: str,
    normalized_response: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AiInventoryProducerReceiptError(
            "command invoke receipt must be a mapping"
        )
    try:
        snapshot = json.loads(
            _canonical_json_bytes(value),
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AiInventoryProducerReceiptError(
            "command invoke receipt is not strict JSON"
        ) from exc
    if set(snapshot) != _COMMAND_INVOKE_RECEIPT_KEYS:
        raise AiInventoryProducerReceiptError(
            "command invoke receipt schema is invalid"
        )
    if snapshot.get("contract") != DOJO_AI_COMMAND_INVOKE_RECEIPT_CONTRACT:
        raise AiInventoryProducerReceiptError(
            "command invoke receipt contract is invalid"
        )
    adapter_id = _receipt_id(snapshot.get("adapter_id"), "adapter_id")
    model_id = _receipt_id(snapshot.get("model_id"), "model_id")
    if model_id != producer_model_id:
        raise AiInventoryProducerReceiptError(
            "command invoke model does not match producer receipt"
        )
    manifest = _trusted_command_manifest(adapter_id)
    if manifest["model_id"] != model_id:
        raise AiInventoryProducerReceiptError(
            "command invoke model does not match trusted manifest"
        )
    command_manifest_sha = _receipt_sha(
        snapshot.get("command_manifest_sha256"),
        "command_manifest_sha256",
    )
    executable_sha = _receipt_sha(
        snapshot.get("executable_sha256"), "executable_sha256"
    )
    if (
        command_manifest_sha != manifest["command_manifest_sha256"]
        or executable_sha != manifest["executable_sha256"]
    ):
        raise AiInventoryProducerReceiptError(
            "command invoke manifest binding mismatch"
        )
    executable_stat = _authenticated_executable(manifest)
    executable_device = _bounded_receipt_int(
        snapshot.get("executable_device"),
        "executable_device",
        minimum=0,
    )
    executable_inode = _bounded_receipt_int(
        snapshot.get("executable_inode"),
        "executable_inode",
        minimum=1,
    )
    executor_uid = _bounded_receipt_int(
        snapshot.get("executor_uid"), "executor_uid", minimum=0
    )
    executor_gid = _bounded_receipt_int(
        snapshot.get("executor_gid"), "executor_gid", minimum=0
    )
    if (
        executable_device != executable_stat.st_dev
        or executable_inode != executable_stat.st_ino
        or executor_uid != manifest["executor_uid"]
        or executor_gid != manifest["executor_gid"]
    ):
        raise AiInventoryProducerReceiptError(
            "command invoke OS executor identity mismatch"
        )
    argv_sha = _receipt_sha(snapshot.get("argv_sha256"), "argv_sha256")
    if argv_sha != _sha256(_canonical_json_bytes(manifest["argv"])):
        raise AiInventoryProducerReceiptError("command invoke argv digest mismatch")
    bound_request = _receipt_sha(
        snapshot.get("request_sha256"), "invoke request_sha256"
    )
    bound_response = _receipt_sha(
        snapshot.get("response_sha256"), "invoke response_sha256"
    )
    signed_response = _parse_receipt_signed_response(snapshot.get("signed_response"))
    if (
        bound_request != request_sha256
        or bound_response != response_sha256
        or bound_response != _sha256(_canonical_json_bytes(signed_response))
        or signed_response != normalized_response
    ):
        raise AiInventoryProducerReceiptError(
            "command invoke request or response digest mismatch"
        )
    signature_key_id = _receipt_id(snapshot.get("signature_key_id"), "signature_key_id")
    if signature_key_id != manifest["signature_key_id"]:
        raise AiInventoryProducerReceiptError("command invoke signature key mismatch")
    signature_base64 = _receipt_signature_base64(snapshot.get("signature_base64"))
    signed_payload_sha = _receipt_sha(
        snapshot.get("signed_payload_sha256"),
        "signed_payload_sha256",
    )
    signed_body = {
        "contract": DOJO_AI_SIGNED_MODEL_RESPONSE_CONTRACT,
        "adapter_id": adapter_id,
        "model_id": model_id,
        "request_sha256": bound_request,
        "response": signed_response,
        "signature_key_id": signature_key_id,
    }
    signed_payload = _canonical_json_bytes(signed_body)
    if signed_payload_sha != _sha256(signed_payload):
        raise AiInventoryProducerReceiptIntegrityError(
            "command invoke signed payload digest mismatch"
        )
    _verify_ed25519_signature(
        public_key_base64=manifest["ed25519_public_key_base64"],
        signature_base64=signature_base64,
        payload=signed_payload,
        receipt_error=True,
    )
    started_at = _receipt_utc(snapshot.get("started_at_utc"), "invoke started_at_utc")
    completed_at = _receipt_utc(
        snapshot.get("completed_at_utc"), "invoke completed_at_utc"
    )
    if _utc_datetime(completed_at) < _utc_datetime(started_at):
        raise AiInventoryProducerReceiptError(
            "command invoke completion precedes start"
        )
    exit_code = _bounded_receipt_int(
        snapshot.get("exit_code"), "exit_code", minimum=0, maximum=0
    )
    normalized: dict[str, Any] = {
        "contract": DOJO_AI_COMMAND_INVOKE_RECEIPT_CONTRACT,
        "adapter_id": adapter_id,
        "model_id": model_id,
        "command_manifest_sha256": command_manifest_sha,
        "executable_sha256": executable_sha,
        "executable_device": executable_device,
        "executable_inode": executable_inode,
        "executor_uid": executor_uid,
        "executor_gid": executor_gid,
        "argv_sha256": argv_sha,
        "request_sha256": bound_request,
        "response_sha256": bound_response,
        "signed_response": signed_response,
        "signature_key_id": signature_key_id,
        "signature_base64": signature_base64,
        "signed_payload_sha256": signed_payload_sha,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "exit_code": exit_code,
    }
    digest = _sha256(_canonical_json_bytes(normalized))
    supplied = _receipt_sha(
        snapshot.get("invoke_receipt_sha256"),
        "invoke_receipt_sha256",
    )
    if supplied != digest:
        raise AiInventoryProducerReceiptIntegrityError(
            "command invoke receipt digest mismatch"
        )
    normalized["invoke_receipt_sha256"] = digest
    return normalized


def _producer_receipt_root(room_root: Path, *, create: bool) -> Path:
    if not isinstance(room_root, Path) or not room_root.is_absolute():
        raise AiInventoryProducerReceiptIntegrityError(
            "room_root must be an absolute Path"
        )
    try:
        root_stat = room_root.lstat()
    except OSError as exc:
        raise AiInventoryProducerReceiptIntegrityError(
            "room_root is unavailable"
        ) from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise AiInventoryProducerReceiptIntegrityError(
            "room_root must be a real directory"
        )
    root = room_root.resolve(strict=True)
    receipt_root = root / PRODUCER_RECEIPT_DIRECTORY
    try:
        receipt_stat = receipt_root.lstat()
    except FileNotFoundError:
        if not create:
            raise AiInventoryProducerReceiptIntegrityError(
                "producer receipt directory does not exist"
            )
        try:
            receipt_root.mkdir(mode=0o700)
            _fsync_directory(root)
        except OSError as exc:
            raise AiInventoryProducerReceiptError(
                "producer receipt directory create failed"
            ) from exc
    except OSError as exc:
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt directory is unavailable"
        ) from exc
    else:
        if stat.S_ISLNK(receipt_stat.st_mode) or not stat.S_ISDIR(receipt_stat.st_mode):
            raise AiInventoryProducerReceiptIntegrityError(
                "producer receipt directory is unsafe"
            )
    resolved = receipt_root.resolve(strict=True)
    if resolved != root / PRODUCER_RECEIPT_DIRECTORY:
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt directory escaped room_root"
        )
    return resolved


def _require_direct_receipt_child(root: Path, receipt_path: Path) -> Path:
    if not isinstance(receipt_path, Path) or not receipt_path.is_absolute():
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt path must be an absolute Path"
        )
    try:
        item_stat = receipt_path.lstat()
    except OSError as exc:
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt is unavailable"
        ) from exc
    if stat.S_ISLNK(item_stat.st_mode) or not stat.S_ISREG(item_stat.st_mode):
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt must be a direct regular file"
        )
    if receipt_path.parent.resolve(strict=True) != root:
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt path escapes its dedicated directory"
        )
    if re.fullmatch(r"[0-9a-f]{64}\.json", receipt_path.name) is None:
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt filename is invalid"
        )
    return receipt_path


def _read_receipt_file(path: Path) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt cannot be opened"
        ) from exc
    try:
        item_stat = os.fstat(fd)
        if not stat.S_ISREG(item_stat.st_mode):
            raise AiInventoryProducerReceiptIntegrityError(
                "producer receipt is not a regular file"
            )
        if item_stat.st_size <= 0 or item_stat.st_size > MAX_PRODUCER_RECEIPT_BYTES:
            raise AiInventoryProducerReceiptIntegrityError(
                "producer receipt size is invalid"
            )
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            raw = handle.read(MAX_PRODUCER_RECEIPT_BYTES + 1)
    except AiInventoryProducerReceiptIntegrityError:
        raise
    except OSError as exc:
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt read failed"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
    if len(raw) != item_stat.st_size:
        raise AiInventoryProducerReceiptIntegrityError(
            "producer receipt changed while reading"
        )
    return raw


def _digest_regular_file(path: Path) -> tuple[str, os.stat_result]:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise AiInventoryProducerModelError(
            "trusted command executable cannot be opened"
        ) from exc
    try:
        item_stat = os.fstat(fd)
        if not stat.S_ISREG(item_stat.st_mode):
            raise AiInventoryProducerModelError(
                "trusted command executable is not a regular file"
            )
        digest = hashlib.sha256()
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except AiInventoryProducerModelError:
        raise
    except OSError as exc:
        raise AiInventoryProducerModelError(
            "trusted command executable cannot be hashed"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
    return digest.hexdigest(), item_stat


def _model_id(value: object, field: str) -> str:
    try:
        return _require_id(value, field)
    except ValueError as exc:
        raise AiInventoryProducerModelError(
            f"trusted command {field} is invalid"
        ) from exc


def _model_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AiInventoryProducerModelError(f"trusted command {field} is invalid")
    return value


def _model_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise AiInventoryProducerModelError(f"{field} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AiInventoryProducerModelError(
            f"{field} must be a UTC timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise AiInventoryProducerModelError(f"{field} must be a UTC timestamp")
    return parsed.astimezone(timezone.utc)


def _canonical_ed25519_public_key(value: object) -> str:
    try:
        raw = _decode_canonical_base64(value)
    except ValueError as exc:
        raise AiInventoryProducerModelError(
            "trusted command Ed25519 public key is invalid"
        ) from exc
    if len(raw) != 32:
        raise AiInventoryProducerModelError(
            "trusted command Ed25519 public key length is invalid"
        )
    return str(value)


def _canonical_signature_base64(value: object) -> str:
    try:
        raw = _decode_canonical_base64(value)
    except ValueError as exc:
        raise AiInventoryProducerModelError(
            "signed model response signature is invalid"
        ) from exc
    if len(raw) != 64:
        raise AiInventoryProducerModelError(
            "signed model response signature length is invalid"
        )
    return str(value)


def _receipt_signature_base64(value: object) -> str:
    try:
        raw = _decode_canonical_base64(value)
    except ValueError as exc:
        raise AiInventoryProducerReceiptError(
            "producer receipt signature is invalid"
        ) from exc
    if len(raw) != 64:
        raise AiInventoryProducerReceiptError(
            "producer receipt signature length is invalid"
        )
    return str(value)


def _decode_canonical_base64(value: object) -> bytes:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError("base64 value is invalid")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("base64 value is invalid") from exc
    if base64.b64encode(raw).decode("ascii") != value:
        raise ValueError("base64 value is noncanonical")
    return raw


def _verify_ed25519_signature(
    *,
    public_key_base64: str,
    signature_base64: str,
    payload: bytes,
    receipt_error: bool = False,
) -> None:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as exc:
        error = (
            AiInventoryProducerReceiptIntegrityError
            if receipt_error
            else AiInventoryProducerModelError
        )
        raise error("Ed25519 verifier is unavailable") from exc
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            _decode_canonical_base64(public_key_base64)
        )
        public_key.verify(
            _decode_canonical_base64(signature_base64),
            payload,
        )
    except (InvalidSignature, ValueError) as exc:
        error = (
            AiInventoryProducerReceiptIntegrityError
            if receipt_error
            else AiInventoryProducerModelError
        )
        raise error("Ed25519 model response signature verification failed") from exc


def _parse_receipt_signed_response(value: object) -> dict[str, Any]:
    try:
        return _parse_response(value)  # type: ignore[arg-type]
    except AiInventoryProducerResponseError as exc:
        raise AiInventoryProducerReceiptError(
            "command invoke signed response is invalid"
        ) from exc


def _bounded_exact_int(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:  # noqa: E721 - reject bool, an int subclass
        raise AiInventoryProducerModelError(
            f"trusted command {field} must be an integer"
        )
    if minimum is not None and value < minimum:
        raise AiInventoryProducerModelError(f"trusted command {field} is below minimum")
    if maximum is not None and value > maximum:
        raise AiInventoryProducerModelError(f"trusted command {field} exceeds maximum")
    return value


def _bounded_receipt_int(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:  # noqa: E721 - reject bool, an int subclass
        raise AiInventoryProducerReceiptError(
            f"producer receipt {field} must be an integer"
        )
    if minimum is not None and value < minimum:
        raise AiInventoryProducerReceiptError(
            f"producer receipt {field} is below minimum"
        )
    if maximum is not None and value > maximum:
        raise AiInventoryProducerReceiptError(
            f"producer receipt {field} exceeds maximum"
        )
    return value


def _receipt_id(value: object, field: str) -> str:
    try:
        return _require_id(value, field)
    except ValueError as exc:
        raise AiInventoryProducerReceiptError(
            f"producer receipt {field} is invalid"
        ) from exc


def _receipt_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AiInventoryProducerReceiptError(f"producer receipt {field} is invalid")
    return value


def _receipt_finite_number(
    value: object,
    field: str,
    *,
    positive: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AiInventoryProducerReceiptError(
            f"producer receipt {field} must be numeric"
        )
    number = float(value)
    if not math.isfinite(number):
        raise AiInventoryProducerReceiptError(
            f"producer receipt {field} must be finite"
        )
    if positive and number <= 0:
        raise AiInventoryProducerReceiptError(
            f"producer receipt {field} must be positive"
        )
    if minimum is not None and number < minimum:
        raise AiInventoryProducerReceiptError(
            f"producer receipt {field} is below minimum"
        )
    if maximum is not None and number > maximum:
        raise AiInventoryProducerReceiptError(
            f"producer receipt {field} exceeds maximum"
        )
    return number


def _receipt_utc(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise AiInventoryProducerReceiptError(f"producer receipt {field} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AiInventoryProducerReceiptError(
            f"producer receipt {field} is invalid"
        ) from exc
    if parsed.tzinfo is None or _format_utc(parsed) != value:
        raise AiInventoryProducerReceiptError(
            f"producer receipt {field} is noncanonical"
        )
    return value


def _utc_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _require_market_open(value: datetime) -> None:
    try:
        status = compute_market_status(value)
    except Exception as exc:
        raise AiInventoryProducerError("FX market status is unavailable") from exc
    if not status.is_fx_open:
        raise AiInventoryProducerMarketClosedError(
            "AI inventory model evaluation is disabled while FX is closed"
        )


def _require_id(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > MAX_ID_CHARS
        or not _ID_RE.fullmatch(value)
    ):
        raise ValueError(f"{field} is invalid")
    return value


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AiInventoryProducerResponseError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise AiInventoryProducerResponseError(f"{field} must be finite")
    return number


def _normalized_number(value: float) -> int | float:
    if value == 0:
        return 0
    if value.is_integer():
        return int(value)
    return value


def _format_utc(value: datetime) -> str:
    utc = value.astimezone(timezone.utc)
    if utc.microsecond:
        return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return utc.isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise AiInventoryProducerReceiptError(
            "producer receipt directory cannot be opened"
        ) from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise AiInventoryProducerReceiptError(
            "producer receipt directory fsync failed"
        ) from exc
    finally:
        os.close(fd)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

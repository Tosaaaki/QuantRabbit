"""Externally signed model sidecar for future paper-only AI inventory rooms.

This module is deliberately not wired to an existing room.  It loads one
content-addressed configuration from a package-derived fixed root, authenticates
the nested model executable, verifies the model-owned Ed25519 signature using
only a pinned public key, and relays the exact canonical response envelope to
``dojo_ai_inventory_producer``.  It has no broker dependency or order
authority.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit import dojo_ai_evidence_packet as _evidence_packets
from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_replay_lifecycle import (
    LAUNCH_PREFLIGHT_CONTRACT,
    DojoReplayLifecycleError,
    verify_paper_ai_inventory_launch_preflight,
)


DOJO_AI_MODEL_SIDECAR_CONFIG_CONTRACT = "QR_DOJO_AI_INVENTORY_MODEL_SIDECAR_CONFIG_V1"
DOJO_AI_SIGNED_MODEL_RESPONSE_CONTRACT = "QR_DOJO_AI_INVENTORY_SIGNED_MODEL_RESPONSE_V1"
DOJO_AI_PROPOSAL_REQUEST_CONTRACT = "QR_DOJO_AI_INVENTORY_PROPOSAL_REQUEST_V2"
DEDICATED_MODEL_SIDECAR_CONFIG_ROOT = Path(
    "research/data/dojo_paper_ai_inventory_v1/model_sidecar_configs"
)

MAX_CONFIG_BYTES = 128 * 1024
MAX_EXECUTABLE_BYTES = 1024 * 1024 * 1024
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_REASON_CHARS = 2_000

AI_PROPOSAL_ACTIONS = frozenset(
    {
        "HOLD",
        "BLOCK_NEW",
        "ALLOW_NEW_VIRTUAL",
        "REDUCE_VIRTUAL",
        "CLOSE_VIRTUAL",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_HEAD_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,254}$")
_REASON_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_CONFIG_KEYS = frozenset(
    {
        "contract",
        "adapter_id",
        "model_id",
        "producer_id",
        "candidate_id",
        "experiment_id",
        "room_id",
        "future_window",
        "model_executable_path",
        "model_executable_sha256",
        "model_executable_device",
        "model_executable_inode",
        "model_executor_uid",
        "model_executor_gid",
        "model_argv",
        "model_timeout_seconds",
        "sidecar_executable_path",
        "sidecar_executable_sha256",
        "sidecar_executable_device",
        "sidecar_executable_inode",
        "sidecar_executor_uid",
        "sidecar_executor_gid",
        "sidecar_timeout_seconds",
        "signature_key_id",
        "ed25519_public_key_base64",
        "git_head",
        "git_branch",
        "paper_only",
        "order_authority",
        "live_permission",
        "config_sha256",
    }
)
_PAPER_AI_LAUNCH_PREFLIGHT_KEYS = frozenset(
    {
        "contract",
        "adapter_id",
        "model_id",
        "config_sha256",
        "producer_id",
        "candidate_id",
        "source_capture_manifest_sha256",
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
_REQUEST_KEYS = frozenset(
    {
        "contract",
        "producer_id",
        "model_id",
        "purpose",
        "evidence_packet",
        "source_watermarks",
        "source_watermarks_sha256",
        "required_response",
        "safety",
    }
)
_RESPONSE_KEYS = frozenset(
    {"action", "reason_code", "reason", "virtual_units", "confidence"}
)
_SIGNED_RESPONSE_KEYS = frozenset(
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


class AiInventoryModelSidecarError(RuntimeError):
    """Base class for fail-closed sidecar errors."""


class AiInventoryModelSidecarMarketClosedError(AiInventoryModelSidecarError):
    """Model evaluation is disabled outside the DST-aware FX week."""


class AiInventoryModelSidecarConfigError(AiInventoryModelSidecarError):
    """A sidecar configuration or bound local resource is untrusted."""


class AiInventoryModelSidecarModelError(AiInventoryModelSidecarError):
    """The nested model did not produce one strict bounded response."""


def model_sidecar_config_sha256(value: Mapping[str, Any]) -> str:
    """Return the digest for a strict config body without ``config_sha256``."""

    return _normalize_config(value, require_digest=False)["config_sha256"]


def load_production_adapter_manifest(
    adapter_id: str,
    config_sha256: str,
    *,
    experiment_id: str,
    room_id: str,
) -> dict[str, Any]:
    """Load one production adapter from the fixed root and fresh Git preflight.

    No repository or config path is accepted from the caller.  The returned
    mapping contains the producer command manifest and the verified lifecycle
    binding.  It contains only public signature-verification material.
    """

    repository_root = _trusted_repository_root()
    now = _utc_now().astimezone(timezone.utc)
    _require_market_open(now)
    git_head, git_branch = _read_git_identity(repository_root)
    preflight = _verified_room_launch_preflight(
        repository_root,
        experiment_id=experiment_id,
        room_id=room_id,
        git_head=git_head,
        now=now,
        require_active_window=False,
    )
    requested_config_sha = _require_sha(config_sha256, "config_sha256")
    if (
        preflight["adapter_id"] != adapter_id
        or preflight["config_sha256"] != requested_config_sha
    ):
        raise AiInventoryModelSidecarConfigError(
            "requested adapter/config does not match lifecycle preregistration"
        )
    config = _load_config_for_root(
        repository_root,
        requested_config_sha,
        expected_adapter_id=adapter_id,
        expected_model_id=preflight["model_id"],
        expected_producer_id=preflight["producer_id"],
        expected_candidate_id=preflight["candidate_id"],
        expected_experiment_id=preflight["experiment_id"],
        expected_room_id=preflight["room_id"],
        expected_future_window=preflight["future_window"],
        git_head=git_head,
        git_branch=git_branch,
    )
    _authenticated_file(
        Path(config["model_executable_path"]),
        expected_sha256=config["model_executable_sha256"],
        expected_device=config["model_executable_device"],
        expected_inode=config["model_executable_inode"],
        expected_uid=config["model_executor_uid"],
        expected_gid=config["model_executor_gid"],
        executable=True,
        role="nested model executable",
    )
    _authenticated_file(
        Path(config["sidecar_executable_path"]),
        expected_sha256=config["sidecar_executable_sha256"],
        expected_device=config["sidecar_executable_device"],
        expected_inode=config["sidecar_executable_inode"],
        expected_uid=config["sidecar_executor_uid"],
        expected_gid=config["sidecar_executor_gid"],
        executable=True,
        role="sidecar executable",
    )
    manifest: dict[str, Any] = {
        "adapter_id": config["adapter_id"],
        "model_id": config["model_id"],
        "executable_path": config["sidecar_executable_path"],
        "executable_sha256": config["sidecar_executable_sha256"],
        "argv": [
            config["sidecar_executable_path"],
            "--config-sha256",
            config["config_sha256"],
        ],
        "executor_uid": config["sidecar_executor_uid"],
        "executor_gid": config["sidecar_executor_gid"],
        "signature_key_id": config["signature_key_id"],
        "ed25519_public_key_base64": config["ed25519_public_key_base64"],
        "timeout_seconds": config["sidecar_timeout_seconds"],
    }
    manifest["command_manifest_sha256"] = _sha256(_canonical_json_bytes(manifest))
    return {
        "command_manifest": manifest,
        "lifecycle_binding": {
            "adapter_id": preflight["adapter_id"],
            "model_id": preflight["model_id"],
            "config_sha256": preflight["config_sha256"],
            "producer_id": preflight["producer_id"],
            "candidate_id": preflight["candidate_id"],
            "experiment_id": preflight["experiment_id"],
            "room_id": preflight["room_id"],
            "future_window": preflight["future_window"],
            "git_head": preflight["git_head"],
            "launch_preflight_token_sha256": preflight[
                "launch_preflight_token_sha256"
            ],
        },
    }


def run_model_sidecar(
    config_sha256: str,
    request_bytes: bytes,
) -> bytes:
    """Run the authenticated nested model and return one signed envelope."""

    repository_root = _trusted_repository_root()
    now = _utc_now().astimezone(timezone.utc)
    _require_market_open(now)
    git_head, git_branch = _read_git_identity(repository_root)
    config = _load_config_for_root(
        repository_root,
        config_sha256,
        expected_adapter_id=None,
        expected_model_id=None,
        expected_producer_id=None,
        expected_candidate_id=None,
        expected_experiment_id=None,
        expected_room_id=None,
        expected_future_window=None,
        git_head=git_head,
        git_branch=git_branch,
    )
    preflight = _verified_room_launch_preflight(
        repository_root,
        experiment_id=config["experiment_id"],
        room_id=config["room_id"],
        git_head=git_head,
        now=now,
        require_active_window=True,
    )
    if (
        config["config_sha256"] != preflight["config_sha256"]
        or config["adapter_id"] != preflight["adapter_id"]
        or config["model_id"] != preflight["model_id"]
        or config["producer_id"] != preflight["producer_id"]
        or config["candidate_id"] != preflight["candidate_id"]
        or config["future_window"] != preflight["future_window"]
    ):
        raise AiInventoryModelSidecarConfigError(
            "sidecar config launch preflight binding mismatch"
        )
    request, canonical_request = _parse_request(request_bytes, config=config)
    request_sha256 = _sha256(canonical_request)

    model_path = Path(config["model_executable_path"])
    before = _authenticated_file(
        model_path,
        expected_sha256=config["model_executable_sha256"],
        expected_device=config["model_executable_device"],
        expected_inode=config["model_executable_inode"],
        expected_uid=config["model_executor_uid"],
        expected_gid=config["model_executor_gid"],
        executable=True,
        role="nested model executable",
    )
    try:
        completed = subprocess.run(
            list(config["model_argv"]),
            input=canonical_request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=config["model_timeout_seconds"],
            check=False,
            cwd="/",
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AiInventoryModelSidecarModelError(
            "nested model invocation failed"
        ) from exc
    completed_at = _utc_now().astimezone(timezone.utc)
    _require_market_open(completed_at)
    after = _authenticated_file(
        model_path,
        expected_sha256=config["model_executable_sha256"],
        expected_device=config["model_executable_device"],
        expected_inode=config["model_executable_inode"],
        expected_uid=config["model_executor_uid"],
        expected_gid=config["model_executor_gid"],
        executable=True,
        role="nested model executable",
    )
    if before.st_mtime_ns != after.st_mtime_ns or before.st_size != after.st_size:
        raise AiInventoryModelSidecarConfigError(
            "nested model executable changed during invocation"
        )
    if completed.returncode != 0:
        raise AiInventoryModelSidecarModelError("nested model exited unsuccessfully")
    return _verify_nested_signed_envelope(
        bytes(completed.stdout),
        config=config,
        request_sha256=request_sha256,
    )


def _verified_room_launch_preflight(
    repository_root: Path,
    *,
    experiment_id: str,
    room_id: str,
    git_head: str,
    now: datetime,
    require_active_window: bool,
) -> dict[str, Any]:
    experiment = _paper_ai_room_id(experiment_id, "experiment_id")
    room = _paper_ai_room_id(room_id, "room_id")
    try:
        verified = verify_paper_ai_inventory_launch_preflight(
            repository_root,
            experiment_id=experiment,
            room_id=room,
        )
        token = json.loads(
            _canonical_json_bytes(verified),
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (
        DojoReplayLifecycleError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise AiInventoryModelSidecarConfigError(
            "canonical per-room PAPER_ELIGIBLE preflight is invalid"
        ) from exc
    if set(token) != _PAPER_AI_LAUNCH_PREFLIGHT_KEYS:
        raise AiInventoryModelSidecarConfigError(
            "canonical room preflight schema is invalid"
        )
    if (
        token.get("contract") != LAUNCH_PREFLIGHT_CONTRACT
        or token.get("paper_only") is not True
        or token.get("order_authority") != "NONE"
        or token.get("live_permission") is not False
        or token.get("paper_room_launched") is not False
        or token.get("experiment_id") != experiment
        or token.get("room_id") != room
        or token.get("git_head") != git_head
    ):
        raise AiInventoryModelSidecarConfigError(
            "canonical room preflight binding or safety is invalid"
        )
    for field in (
        "config_sha256",
        "candidate_id",
        "source_capture_manifest_sha256",
        "git_head_sha256",
        "replay_worker_receipt_sha256",
        "launch_preflight_token_sha256",
    ):
        _require_sha(token.get(field), f"room preflight {field}")
    for field in ("adapter_id", "model_id", "producer_id"):
        _require_id(token.get(field), f"room preflight {field}")
    body = {
        key: item
        for key, item in token.items()
        if key != "launch_preflight_token_sha256"
    }
    if token["launch_preflight_token_sha256"] != _sha256(_canonical_json_bytes(body)):
        raise AiInventoryModelSidecarConfigError(
            "canonical room preflight digest mismatch"
        )
    window = _future_window(token.get("future_window"))
    issued = _utc_timestamp(token.get("issued_at_utc"), "room preflight issued_at_utc")
    start = _utc_timestamp(
        window["start_utc"], "room preflight future_window.start_utc"
    )
    end = _utc_timestamp(window["end_utc"], "room preflight future_window.end_utc")
    checked_at = now.astimezone(timezone.utc)
    if issued >= start or start >= end or checked_at >= end:
        raise AiInventoryModelSidecarConfigError(
            "canonical room preflight future window is invalid or expired"
        )
    if require_active_window and checked_at < start:
        raise AiInventoryModelSidecarConfigError(
            "AI sidecar cannot run before the canonical future window"
        )
    token["future_window"] = window
    return token


def _load_config_for_root(
    repository_root: Path,
    config_sha256: str,
    *,
    expected_adapter_id: str | None,
    expected_model_id: str | None,
    expected_producer_id: str | None,
    expected_candidate_id: str | None,
    expected_experiment_id: str | None,
    expected_room_id: str | None,
    expected_future_window: Mapping[str, Any] | None,
    git_head: str,
    git_branch: str,
) -> dict[str, Any]:
    digest = _require_sha(config_sha256, "config_sha256")
    root = _fixed_config_root(repository_root)
    path = root / f"{digest}.json"
    raw = _read_regular_file(path, MAX_CONFIG_BYTES, role="sidecar config")
    if raw != raw.rstrip(b"\n") + b"\n" or raw.count(b"\n") != 1:
        raise AiInventoryModelSidecarConfigError(
            "sidecar config must be one canonical JSON row"
        )
    try:
        value = json.loads(
            raw[:-1],
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
        config = _normalize_config(value, require_digest=True)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AiInventoryModelSidecarConfigError(
            "sidecar config JSON is invalid"
        ) from exc
    if config["config_sha256"] != digest:
        raise AiInventoryModelSidecarConfigError(
            "sidecar config filename does not match sealed content"
        )
    if raw != _canonical_json_bytes(config) + b"\n":
        raise AiInventoryModelSidecarConfigError(
            "sidecar config bytes are not canonical"
        )
    if expected_adapter_id is not None and config["adapter_id"] != expected_adapter_id:
        raise AiInventoryModelSidecarConfigError(
            "sidecar adapter id does not match requested adapter"
        )
    if expected_model_id is not None and config["model_id"] != expected_model_id:
        raise AiInventoryModelSidecarConfigError(
            "sidecar model id does not match lifecycle preregistration"
        )
    if (
        expected_producer_id is not None
        and config["producer_id"] != expected_producer_id
    ):
        raise AiInventoryModelSidecarConfigError(
            "sidecar producer id does not match lifecycle preregistration"
        )
    if (
        expected_candidate_id is not None
        and config["candidate_id"] != expected_candidate_id
    ):
        raise AiInventoryModelSidecarConfigError(
            "sidecar candidate id does not match canonical room preflight"
        )
    if (
        expected_experiment_id is not None
        and config["experiment_id"] != expected_experiment_id
    ):
        raise AiInventoryModelSidecarConfigError(
            "sidecar experiment id does not match canonical room preflight"
        )
    if expected_room_id is not None and config["room_id"] != expected_room_id:
        raise AiInventoryModelSidecarConfigError(
            "sidecar room id does not match canonical room preflight"
        )
    if (
        expected_future_window is not None
        and config["future_window"] != expected_future_window
    ):
        raise AiInventoryModelSidecarConfigError(
            "sidecar future window does not match canonical room preflight"
        )
    if config["git_head"] != git_head or config["git_branch"] != git_branch:
        raise AiInventoryModelSidecarConfigError(
            "sidecar config Git binding no longer matches runtime"
        )
    return config


def _normalize_config(
    value: Mapping[str, Any],
    *,
    require_digest: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AiInventoryModelSidecarConfigError("sidecar config must be a mapping")
    try:
        snapshot = json.loads(
            _canonical_json_bytes(value),
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AiInventoryModelSidecarConfigError(
            "sidecar config is not strict JSON"
        ) from exc
    expected = _CONFIG_KEYS if require_digest else _CONFIG_KEYS - {"config_sha256"}
    if set(snapshot) != expected:
        raise AiInventoryModelSidecarConfigError("sidecar config schema is invalid")
    if snapshot.get("contract") != DOJO_AI_MODEL_SIDECAR_CONFIG_CONTRACT:
        raise AiInventoryModelSidecarConfigError("sidecar config contract is invalid")
    if (
        snapshot.get("paper_only") is not True
        or snapshot.get("order_authority") != "NONE"
        or snapshot.get("live_permission") is not False
    ):
        raise AiInventoryModelSidecarConfigError(
            "sidecar config safety authority is invalid"
        )
    model_path = _absolute_path(
        snapshot.get("model_executable_path"), "model_executable_path"
    )
    sidecar_path = _absolute_path(
        snapshot.get("sidecar_executable_path"), "sidecar_executable_path"
    )
    model_argv = _argv(
        snapshot.get("model_argv"),
        executable_path=model_path,
    )
    normalized: dict[str, Any] = {
        "contract": DOJO_AI_MODEL_SIDECAR_CONFIG_CONTRACT,
        "adapter_id": _require_id(snapshot.get("adapter_id"), "adapter_id"),
        "model_id": _require_id(snapshot.get("model_id"), "model_id"),
        "producer_id": _require_id(snapshot.get("producer_id"), "producer_id"),
        "candidate_id": _require_sha(snapshot.get("candidate_id"), "candidate_id"),
        "experiment_id": _paper_ai_room_id(
            snapshot.get("experiment_id"), "experiment_id"
        ),
        "room_id": _paper_ai_room_id(snapshot.get("room_id"), "room_id"),
        "future_window": _future_window(snapshot.get("future_window")),
        "model_executable_path": model_path,
        "model_executable_sha256": _require_sha(
            snapshot.get("model_executable_sha256"),
            "model_executable_sha256",
        ),
        "model_executable_device": _exact_int(
            snapshot.get("model_executable_device"),
            "model_executable_device",
            minimum=0,
        ),
        "model_executable_inode": _exact_int(
            snapshot.get("model_executable_inode"),
            "model_executable_inode",
            minimum=1,
        ),
        "model_executor_uid": _exact_int(
            snapshot.get("model_executor_uid"),
            "model_executor_uid",
            minimum=0,
        ),
        "model_executor_gid": _exact_int(
            snapshot.get("model_executor_gid"),
            "model_executor_gid",
            minimum=0,
        ),
        "model_argv": model_argv,
        "model_timeout_seconds": _exact_int(
            snapshot.get("model_timeout_seconds"),
            "model_timeout_seconds",
            minimum=1,
            maximum=300,
        ),
        "sidecar_executable_path": sidecar_path,
        "sidecar_executable_sha256": _require_sha(
            snapshot.get("sidecar_executable_sha256"),
            "sidecar_executable_sha256",
        ),
        "sidecar_executable_device": _exact_int(
            snapshot.get("sidecar_executable_device"),
            "sidecar_executable_device",
            minimum=0,
        ),
        "sidecar_executable_inode": _exact_int(
            snapshot.get("sidecar_executable_inode"),
            "sidecar_executable_inode",
            minimum=1,
        ),
        "sidecar_executor_uid": _exact_int(
            snapshot.get("sidecar_executor_uid"),
            "sidecar_executor_uid",
            minimum=0,
        ),
        "sidecar_executor_gid": _exact_int(
            snapshot.get("sidecar_executor_gid"),
            "sidecar_executor_gid",
            minimum=0,
        ),
        "sidecar_timeout_seconds": _exact_int(
            snapshot.get("sidecar_timeout_seconds"),
            "sidecar_timeout_seconds",
            minimum=1,
            maximum=300,
        ),
        "signature_key_id": _require_id(
            snapshot.get("signature_key_id"), "signature_key_id"
        ),
        "ed25519_public_key_base64": _public_key_base64(
            snapshot.get("ed25519_public_key_base64")
        ),
        "git_head": _git_head(snapshot.get("git_head")),
        "git_branch": _git_branch(snapshot.get("git_branch")),
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    digest = _sha256(_canonical_json_bytes(normalized))
    if require_digest and snapshot.get("config_sha256") != digest:
        raise AiInventoryModelSidecarConfigError("sidecar config digest mismatch")
    normalized["config_sha256"] = digest
    return normalized


def _parse_request(
    raw: bytes,
    *,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_REQUEST_BYTES:
        raise AiInventoryModelSidecarModelError("sidecar request size is invalid")
    try:
        request = json.loads(
            raw,
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AiInventoryModelSidecarModelError(
            "sidecar request is not strict JSON"
        ) from exc
    if not isinstance(request, dict) or set(request) != _REQUEST_KEYS:
        raise AiInventoryModelSidecarModelError("sidecar request schema is invalid")
    canonical = _canonical_json_bytes(request)
    if raw != canonical:
        raise AiInventoryModelSidecarModelError(
            "sidecar request bytes are not canonical"
        )
    safety = request.get("safety")
    if (
        request.get("contract") != DOJO_AI_PROPOSAL_REQUEST_CONTRACT
        or request.get("model_id") != config["model_id"]
        or request.get("producer_id") != config["producer_id"]
        or request.get("purpose") != "PAPER_AI_INVENTORY_PROPOSAL_ONLY"
        or not isinstance(safety, Mapping)
        or safety.get("paper_only") is not True
        or safety.get("order_authority") != "NONE"
        or safety.get("live_permission") is not False
    ):
        raise AiInventoryModelSidecarModelError(
            "sidecar request binding or safety is invalid"
        )
    return request, canonical


def _verify_nested_signed_envelope(
    raw: bytes,
    *,
    config: Mapping[str, Any],
    request_sha256: str,
) -> bytes:
    if not raw or len(raw) > MAX_RESPONSE_BYTES:
        raise AiInventoryModelSidecarModelError(
            "nested model signed response size is invalid"
        )
    try:
        envelope = json.loads(
            raw,
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AiInventoryModelSidecarModelError(
            "nested model signed response is not strict JSON"
        ) from exc
    if not isinstance(envelope, dict) or set(envelope) != _SIGNED_RESPONSE_KEYS:
        raise AiInventoryModelSidecarModelError(
            "nested model signed response schema is invalid"
        )
    canonical = _canonical_json_bytes(envelope)
    if raw != canonical:
        raise AiInventoryModelSidecarModelError(
            "nested model signed response bytes are not canonical"
        )
    if (
        envelope.get("contract") != DOJO_AI_SIGNED_MODEL_RESPONSE_CONTRACT
        or envelope.get("adapter_id") != config["adapter_id"]
        or envelope.get("model_id") != config["model_id"]
        or envelope.get("request_sha256") != request_sha256
        or envelope.get("signature_key_id") != config["signature_key_id"]
    ):
        raise AiInventoryModelSidecarModelError(
            "nested model signed response binding mismatch"
        )
    response = envelope.get("response")
    if not isinstance(response, Mapping):
        raise AiInventoryModelSidecarModelError(
            "nested model response must be an object"
        )
    _parse_response(_canonical_json_bytes(response))
    signature = _signature_bytes(envelope.get("signature_base64"))
    signed_body = {
        key: envelope[key]
        for key in _SIGNED_RESPONSE_KEYS
        if key != "signature_base64"
    }
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(
                config["ed25519_public_key_base64"],
                validate=True,
            )
        )
        public_key.verify(signature, _canonical_json_bytes(signed_body))
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise AiInventoryModelSidecarModelError(
            "nested model signed response signature is invalid"
        ) from exc
    return canonical


def _parse_response(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > MAX_RESPONSE_BYTES:
        raise AiInventoryModelSidecarModelError("nested model response size is invalid")
    try:
        response = json.loads(
            raw,
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AiInventoryModelSidecarModelError(
            "nested model response is not strict JSON"
        ) from exc
    if not isinstance(response, dict) or set(response) != _RESPONSE_KEYS:
        raise AiInventoryModelSidecarModelError(
            "nested model response schema is invalid"
        )
    canonical = _canonical_json_bytes(response)
    if raw not in (canonical, canonical + b"\n"):
        raise AiInventoryModelSidecarModelError(
            "nested model response bytes are not canonical"
        )
    action = response.get("action")
    reason_code = response.get("reason_code")
    reason = response.get("reason")
    if action not in AI_PROPOSAL_ACTIONS:
        raise AiInventoryModelSidecarModelError(
            "nested model response action is invalid"
        )
    if not isinstance(reason_code, str) or not _REASON_CODE_RE.fullmatch(reason_code):
        raise AiInventoryModelSidecarModelError(
            "nested model response reason_code is invalid"
        )
    if (
        not isinstance(reason, str)
        or not reason.strip()
        or len(reason) > MAX_REASON_CHARS
        or any(ord(character) < 32 for character in reason)
    ):
        raise AiInventoryModelSidecarModelError(
            "nested model response reason is invalid"
        )
    confidence = _finite_number(response.get("confidence"), "confidence")
    if not 0 <= confidence <= 1:
        raise AiInventoryModelSidecarModelError(
            "nested model confidence is outside [0,1]"
        )
    units = response.get("virtual_units")
    if action in {"HOLD", "BLOCK_NEW", "ALLOW_NEW_VIRTUAL"}:
        if units is not None:
            raise AiInventoryModelSidecarModelError(
                "non-mutating model response must use virtual_units=null"
            )
        canonical_units: float | None = None
    else:
        canonical_units = _finite_number(units, "virtual_units")
        if canonical_units <= 0:
            raise AiInventoryModelSidecarModelError(
                "mutating model response virtual_units must be positive"
            )
    return {
        "action": action,
        "reason_code": reason_code,
        "reason": reason,
        "virtual_units": canonical_units,
        "confidence": confidence,
    }


def _authenticated_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_device: int,
    expected_inode: int,
    expected_uid: int,
    expected_gid: int,
    executable: bool,
    role: str,
) -> os.stat_result:
    try:
        item_stat = path.lstat()
    except OSError as exc:
        raise AiInventoryModelSidecarConfigError(f"{role} is unavailable") from exc
    if stat.S_ISLNK(item_stat.st_mode) or not stat.S_ISREG(item_stat.st_mode):
        raise AiInventoryModelSidecarConfigError(f"{role} must be a real regular file")
    if executable and not item_stat.st_mode & (
        stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    ):
        raise AiInventoryModelSidecarConfigError(f"{role} is not executable")
    digest, opened_stat = _digest_file(path, MAX_EXECUTABLE_BYTES)
    if (
        digest != expected_sha256
        or opened_stat.st_dev != item_stat.st_dev
        or opened_stat.st_ino != item_stat.st_ino
        or opened_stat.st_dev != expected_device
        or opened_stat.st_ino != expected_inode
        or opened_stat.st_uid != expected_uid
        or opened_stat.st_gid != expected_gid
    ):
        raise AiInventoryModelSidecarConfigError(f"{role} identity or digest mismatch")
    return opened_stat


def _fixed_config_root(repository_root: Path) -> Path:
    try:
        repo_stat = repository_root.lstat()
    except OSError as exc:
        raise AiInventoryModelSidecarConfigError(
            "sidecar repository root is unavailable"
        ) from exc
    if stat.S_ISLNK(repo_stat.st_mode) or not stat.S_ISDIR(repo_stat.st_mode):
        raise AiInventoryModelSidecarConfigError(
            "sidecar repository root must be a real directory"
        )
    current = repository_root.resolve(strict=True)
    for part in DEDICATED_MODEL_SIDECAR_CONFIG_ROOT.parts:
        current = current / part
        try:
            item_stat = current.lstat()
        except OSError as exc:
            raise AiInventoryModelSidecarConfigError(
                "fixed sidecar config root is unavailable"
            ) from exc
        if stat.S_ISLNK(item_stat.st_mode) or not stat.S_ISDIR(item_stat.st_mode):
            raise AiInventoryModelSidecarConfigError(
                "fixed sidecar config root contains an unsafe component"
            )
    return current


def _trusted_repository_root() -> Path:
    try:
        return _evidence_packets._trusted_repository_root()
    except (ValueError, RuntimeError) as exc:
        raise AiInventoryModelSidecarConfigError(
            "package-derived repository root is unavailable"
        ) from exc


def _read_git_identity(repository_root: Path) -> tuple[str, str]:
    try:
        return _evidence_packets._read_git_identity(repository_root)
    except (ValueError, RuntimeError) as exc:
        raise AiInventoryModelSidecarConfigError(
            "sidecar Git identity is unavailable"
        ) from exc


def _read_regular_file(
    path: Path,
    limit: int,
    *,
    role: str,
    exact_mode: int | None = None,
    expected_uid: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        item_stat = path.lstat()
        fd = os.open(path, flags)
    except OSError as exc:
        raise AiInventoryModelSidecarConfigError(f"{role} cannot be opened") from exc
    try:
        opened_stat = os.fstat(fd)
        if (
            stat.S_ISLNK(item_stat.st_mode)
            or not stat.S_ISREG(item_stat.st_mode)
            or not stat.S_ISREG(opened_stat.st_mode)
            or item_stat.st_dev != opened_stat.st_dev
            or item_stat.st_ino != opened_stat.st_ino
        ):
            raise AiInventoryModelSidecarConfigError(
                f"{role} is not a stable regular file"
            )
        if exact_mode is not None and stat.S_IMODE(opened_stat.st_mode) != exact_mode:
            raise AiInventoryModelSidecarConfigError(
                f"{role} permissions must be {exact_mode:04o}"
            )
        if expected_uid is not None and opened_stat.st_uid != expected_uid:
            raise AiInventoryModelSidecarConfigError(f"{role} owner mismatch")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(65_536, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise AiInventoryModelSidecarConfigError(
                    f"{role} exceeds the fixed size bound"
                )
        return b"".join(chunks)
    finally:
        os.close(fd)


def _digest_file(path: Path, limit: int) -> tuple[str, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        item_stat = path.lstat()
        fd = os.open(path, flags)
        opened_stat = os.fstat(fd)
    except OSError as exc:
        raise AiInventoryModelSidecarConfigError(
            "authenticated executable cannot be opened"
        ) from exc
    try:
        if (
            stat.S_ISLNK(item_stat.st_mode)
            or not stat.S_ISREG(item_stat.st_mode)
            or not stat.S_ISREG(opened_stat.st_mode)
            or item_stat.st_dev != opened_stat.st_dev
            or item_stat.st_ino != opened_stat.st_ino
            or opened_stat.st_size > limit
        ):
            raise AiInventoryModelSidecarConfigError(
                "authenticated executable is not a stable bounded regular file"
            )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise AiInventoryModelSidecarConfigError(
                    "authenticated executable exceeds the size bound"
                )
            digest.update(chunk)
        return digest.hexdigest(), opened_stat
    finally:
        os.close(fd)


def _absolute_path(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 4_096
        or "\x00" in value
        or not Path(value).is_absolute()
    ):
        raise AiInventoryModelSidecarConfigError(f"{field} is invalid")
    return value


def _argv(value: object, *, executable_path: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 64
        or value[0] != executable_path
    ):
        raise AiInventoryModelSidecarConfigError("model_argv is invalid")
    argv: list[str] = []
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 16_384
            or "\x00" in item
        ):
            raise AiInventoryModelSidecarConfigError(
                "model_argv contains an invalid item"
            )
        argv.append(item)
    return argv


def _public_key_base64(value: object) -> str:
    if not isinstance(value, str):
        raise AiInventoryModelSidecarConfigError("Ed25519 public key is invalid")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise AiInventoryModelSidecarConfigError(
            "Ed25519 public key is invalid"
        ) from exc
    if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != value:
        raise AiInventoryModelSidecarConfigError("Ed25519 public key is invalid")
    return value


def _signature_bytes(value: object) -> bytes:
    if not isinstance(value, str):
        raise AiInventoryModelSidecarModelError(
            "nested model signature is invalid"
        )
    try:
        raw = base64.b64decode(value, validate=True)
    except (TypeError, ValueError) as exc:
        raise AiInventoryModelSidecarModelError(
            "nested model signature is invalid"
        ) from exc
    if len(raw) != 64 or base64.b64encode(raw).decode("ascii") != value:
        raise AiInventoryModelSidecarModelError(
            "nested model signature is invalid"
        )
    return raw


def _exact_int(
    value: object,
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:  # noqa: E721 - bool must fail closed
        raise AiInventoryModelSidecarConfigError(f"{field} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise AiInventoryModelSidecarConfigError(f"{field} is outside bounds")
    return value


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AiInventoryModelSidecarModelError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AiInventoryModelSidecarModelError(f"{field} must be finite")
    return result


def _require_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise AiInventoryModelSidecarConfigError(f"{field} is invalid")
    return value


def _require_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise AiInventoryModelSidecarConfigError(f"{field} is invalid")
    return value


def _git_head(value: object) -> str:
    if not isinstance(value, str) or not _GIT_HEAD_RE.fullmatch(value):
        raise AiInventoryModelSidecarConfigError("git_head is invalid")
    return value


def _git_branch(value: object) -> str:
    branch = _require_id(value, "git_branch")
    if not branch.startswith("codex/"):
        raise AiInventoryModelSidecarConfigError("git_branch must be under codex/")
    return branch


def _paper_ai_room_id(value: object, field: str) -> str:
    identifier = _require_id(value, field)
    if (
        not identifier.startswith("paper-ai-inventory-")
        or Path(identifier).name != identifier
    ):
        raise AiInventoryModelSidecarConfigError(
            f"{field} is not an isolated paper AI identifier"
        )
    return identifier


def _future_window(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"start_utc", "end_utc"}:
        raise AiInventoryModelSidecarConfigError("future_window schema is invalid")
    start_text = value.get("start_utc")
    end_text = value.get("end_utc")
    start = _utc_timestamp(start_text, "future_window.start_utc")
    end = _utc_timestamp(end_text, "future_window.end_utc")
    if start >= end:
        raise AiInventoryModelSidecarConfigError("future_window is empty or reversed")
    assert isinstance(start_text, str)
    assert isinstance(end_text, str)
    return {"start_utc": start_text, "end_utc": end_text}


def _utc_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise AiInventoryModelSidecarConfigError(f"{field} must be UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AiInventoryModelSidecarConfigError(f"{field} must be UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise AiInventoryModelSidecarConfigError(f"{field} must be UTC")
    return parsed.astimezone(timezone.utc)


def _strict_unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_market_open(value: datetime) -> None:
    status = compute_market_status(value.astimezone(timezone.utc))
    if not status.is_fx_open:
        raise AiInventoryModelSidecarMarketClosedError(
            "MARKET_CLOSED_AI_MODEL_SIDECAR_PAUSED"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args(argv)
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    try:
        envelope = run_model_sidecar(args.config_sha256, raw)
    except AiInventoryModelSidecarError as exc:
        print(
            f"sidecar_error:{type(exc).__name__}",
            file=sys.stderr,
        )
        return 2
    sys.stdout.buffer.write(envelope)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

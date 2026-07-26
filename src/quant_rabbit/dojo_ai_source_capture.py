"""Signed point-in-time source capture for paper-only AI inventory evidence.

The capture key is intentionally external to the repository.  A future
per-room PAPER_ELIGIBLE token must bind the exact public-key and adapter
manifest before either capture or verification is possible.  Production
capture dispatches only code-owned registered read-only adapters; the
caller-callback surface is explicitly unsafe/test-only.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import inspect
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_ai_source_adapters import (
    OANDA_COMPLETED_BID_ASK_CANDLES_ADAPTER_ID,
    OANDA_EXECUTABLE_QUOTE_ADAPTER_ID,
    SOURCE_ADAPTER_MODULE,
    DojoAiSourceAdapterError,
    acquire_oanda_completed_bid_ask_candles,
    acquire_oanda_executable_quote,
    canonical_source_adapter_config_bytes,
    source_adapter_capture_binding,
)
from quant_rabbit.dojo_replay_lifecycle import (
    DojoReplayLifecycleError,
    verify_paper_ai_inventory_launch_preflight,
)


SOURCE_CAPTURE_MANIFEST_CONTRACT = "QR_DOJO_AI_SOURCE_CAPTURE_MANIFEST_V1"
SOURCE_CAPTURE_RECEIPT_CONTRACT = "QR_DOJO_AI_SOURCE_CAPTURE_RECEIPT_V1"
CAPTURE_MANIFEST_TOKEN_FIELD = "source_capture_manifest_sha256"
CAPTURE_PRIVATE_KEY_ENV = "QR_DOJO_AI_CAPTURE_ED25519_PRIVATE_KEY_PATH"
CANONICAL_SOURCE_ROOT = Path(
    "research/data/dojo_paper_ai_inventory_v1/canonical_sources"
)
CAPTURE_ROOT = Path("research/data/dojo_paper_ai_inventory_v1/source_capture")
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_RECEIPTS = 100_000
LOW_LEVEL_CAPTURE_PRODUCTION_SAFE = False

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MANIFEST_KEYS = frozenset(
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
_RECEIPT_BODY_KEYS = frozenset(
    {
        "contract",
        "sequence",
        "previous_receipt_sha256",
        "experiment_id",
        "room_id",
        "candidate_id",
        "source_role",
        "canonical_source_sha256",
        "raw_source_bytes_sha256",
        "provider_kind",
        "adapter_id",
        "adapter_module",
        "adapter_callable",
        "adapter_executable_sha256",
        "adapter_config_sha256",
        "provider_timestamp_utc",
        "fetched_at_utc",
        "source_watermark_sha256",
        "cutoff_utc",
        "capture_manifest_file_sha256",
        "capture_manifest_sha256",
        "capture_key_id",
        "paper_only",
        "order_authority",
        "live_permission",
    }
)
_RECEIPT_KEYS = _RECEIPT_BODY_KEYS | frozenset(
    {"receipt_sha256", "signature_base64"}
)
_ADAPTER_KEYS = frozenset(
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
_REGISTERED_QUOTE_KEYS = frozenset(
    {
        "pair",
        "bid",
        "ask",
        "timestamp_utc",
        "max_age_seconds",
    }
)
_REGISTERED_CANDLE_KEYS = frozenset(
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
        "max_age_seconds",
    }
)
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_GRANULARITY_RE = re.compile(r"^[A-Z][A-Z0-9]{0,7}$")


class AiSourceCaptureError(RuntimeError):
    """Capture or verification failed closed."""


class AiSourceCaptureMarketClosedError(AiSourceCaptureError):
    """No fetch or capture is allowed while the FX week is closed."""


@dataclass(frozen=True)
class ReadOnlySourceAcquisition:
    """One atomic result returned by a read-only source adapter."""

    raw_bytes: bytes
    provider_timestamp_utc: datetime | str
    source_watermark_sha256: str


@dataclass(frozen=True)
class _RegisteredAdapter:
    adapter_id: str
    module: str
    callable_name: str
    acquire: Callable[[Mapping[str, Any]], object]


# Production registration is code-owned.  Runtime callers cannot add entries.
# Each adapter must be implemented and reviewed here before a lifecycle-bound
# manifest can activate it.
_REGISTERED_ADAPTERS: Mapping[str, _RegisteredAdapter] = MappingProxyType(
    {
        OANDA_EXECUTABLE_QUOTE_ADAPTER_ID: _RegisteredAdapter(
            adapter_id=OANDA_EXECUTABLE_QUOTE_ADAPTER_ID,
            module=SOURCE_ADAPTER_MODULE,
            callable_name="acquire_oanda_executable_quote",
            acquire=acquire_oanda_executable_quote,
        ),
        OANDA_COMPLETED_BID_ASK_CANDLES_ADAPTER_ID: _RegisteredAdapter(
            adapter_id=OANDA_COMPLETED_BID_ASK_CANDLES_ADAPTER_ID,
            module=SOURCE_ADAPTER_MODULE,
            callable_name="acquire_oanda_completed_bid_ask_candles",
            acquire=acquire_oanda_completed_bid_ask_candles,
        ),
    }
)
_ROLE_ADAPTER_IDS: Mapping[str, str] = MappingProxyType(
    {
        "quote": OANDA_EXECUTABLE_QUOTE_ADAPTER_ID,
        "candles": OANDA_COMPLETED_BID_ASK_CANDLES_ADAPTER_ID,
    }
)


def source_capture_manifest_sha256(value: Mapping[str, Any]) -> str:
    """Return the semantic manifest digest, excluding its digest field."""

    manifest = _normalize_manifest(value, require_digest=False)
    return manifest["manifest_sha256"]


def capture_registered_ai_source(
    *,
    experiment_id: str,
    room_id: str,
    candidate_id: str,
    source_role: str,
    cutoff_utc: datetime | str,
) -> dict[str, Any]:
    """Capture through one code-owned, manifest-bound read-only adapter."""

    role = _source_role(source_role)
    adapter_id = _ROLE_ADAPTER_IDS.get(role)
    if adapter_id is None:
        raise AiSourceCaptureError(
            "no code-owned production adapter is registered for source role"
        )
    registration = _REGISTERED_ADAPTERS.get(adapter_id)
    if registration is None or registration.adapter_id != adapter_id:
        raise AiSourceCaptureError("production source adapter registry is invalid")

    def acquire() -> ReadOnlySourceAcquisition:
        root = _trusted_repository_root()
        token = _verified_preflight(
            root,
            _capture_id(experiment_id, "experiment_id"),
            _capture_id(room_id, "room_id"),
            _sha(candidate_id, "candidate_id"),
        )
        manifest, _ = _bound_manifest(root, token)
        binding = _manifest_adapter(manifest, role)
        _verify_registered_adapter(registration, binding)
        config = _load_adapter_config(root, binding)
        try:
            result = registration.acquire(config)
        except DojoAiSourceAdapterError as exc:
            raise AiSourceCaptureError(
                "registered source adapter acquisition failed"
            ) from exc
        return _normalize_acquisition(result)

    return _capture_with_bound_adapter(
        experiment_id=experiment_id,
        room_id=room_id,
        candidate_id=candidate_id,
        source_role=role,
        cutoff_utc=cutoff_utc,
        expected_adapter_id=adapter_id,
        acquire=acquire,
    )


def capture_test_only_ai_source(
    *,
    experiment_id: str,
    room_id: str,
    candidate_id: str,
    source_role: str,
    cutoff_utc: datetime | str,
    provider_kind: str,
    acquire: Callable[[], ReadOnlySourceAcquisition],
) -> dict[str, Any]:
    """UNSAFE/test-only capture from a caller-provided callback.

    A lifecycle manifest must explicitly bind ``TEST_ONLY_CALLBACK``.  This
    surface is never production-safe and must not be used by room/controller
    code.
    """

    provider_kind = _capture_id(provider_kind, "provider_kind")
    return _capture_with_bound_adapter(
        experiment_id=experiment_id,
        room_id=room_id,
        candidate_id=candidate_id,
        source_role=source_role,
        cutoff_utc=cutoff_utc,
        expected_adapter_id="TEST_ONLY_CALLBACK",
        expected_provider_kind=provider_kind,
        acquire=acquire,
    )


def _capture_with_bound_adapter(
    *,
    experiment_id: str,
    room_id: str,
    candidate_id: str,
    source_role: str,
    cutoff_utc: datetime | str,
    expected_adapter_id: str,
    acquire: Callable[[], ReadOnlySourceAcquisition],
    expected_provider_kind: str | None = None,
) -> dict[str, Any]:
    if not callable(acquire):
        raise TypeError("read-only acquisition callback must be callable")
    root = _trusted_repository_root()
    experiment_id = _capture_id(experiment_id, "experiment_id")
    room_id = _capture_id(room_id, "room_id")
    candidate_id = _sha(candidate_id, "candidate_id")
    source_role = _source_role(source_role)
    cutoff = _utc(cutoff_utc, "cutoff_utc")

    pre_fetch_clock = _utc_now()
    _require_market_open(pre_fetch_clock, "pre-fetch clock")
    token = _verified_preflight(root, experiment_id, room_id, candidate_id)
    manifest, manifest_file_sha = _bound_manifest(root, token)
    adapter = _manifest_adapter(manifest, source_role)
    if adapter["adapter_id"] != expected_adapter_id:
        raise AiSourceCaptureError("capture adapter identity binding mismatch")
    if (
        expected_provider_kind is not None
        and adapter["provider_kind"] != expected_provider_kind
    ):
        raise AiSourceCaptureError("capture provider binding mismatch")
    _require_cutoff_in_future_window(cutoff, token)
    private_key = _load_private_key(manifest)

    acquisition = _normalize_acquisition(acquire())
    fetched_at = _utc_now()
    _require_market_open(fetched_at, "post-fetch clock")
    raw = acquisition.raw_bytes
    provider_timestamp = _utc(
        acquisition.provider_timestamp_utc, "provider_timestamp_utc"
    )
    source_watermark_sha256 = _sha(
        acquisition.source_watermark_sha256, "source_watermark_sha256"
    )
    if not isinstance(raw, bytes):
        raise AiSourceCaptureError("read-only acquisition must return bytes")
    if len(raw) <= 0 or len(raw) > MAX_SOURCE_BYTES:
        raise AiSourceCaptureError("captured source size is invalid")
    if provider_timestamp > fetched_at:
        raise AiSourceCaptureError("provider timestamp is after fetch")
    if fetched_at > cutoff:
        raise AiSourceCaptureError("fetch completed after immutable cutoff")
    source_document = _parse_canonical_json(raw, "captured source")
    _validate_acquisition_document(
        source_document,
        adapter=adapter,
        acquisition=acquisition,
        fetched_at=fetched_at,
    )

    source_sha = hashlib.sha256(raw).hexdigest()
    _write_canonical_source(root, source_sha, raw)
    receipt_root = _receipt_root(root, experiment_id, room_id, create=True)
    lock_path = receipt_root / ".capture.lock"
    try:
        lock_fd = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise AiSourceCaptureError("capture lock cannot be opened safely") from exc
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise AiSourceCaptureError("capture lock is not a regular file")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        rows = _verify_receipt_chain(
            receipt_root,
            manifest=manifest,
            manifest_file_sha=manifest_file_sha,
            lifecycle_token=token,
            experiment_id=experiment_id,
            room_id=room_id,
            candidate_id=candidate_id,
        )
        sequence = len(rows) + 1
        previous = rows[-1]["receipt_sha256"] if rows else "0" * 64
        body: dict[str, Any] = {
            "contract": SOURCE_CAPTURE_RECEIPT_CONTRACT,
            "sequence": sequence,
            "previous_receipt_sha256": previous,
            "experiment_id": experiment_id,
            "room_id": room_id,
            "candidate_id": candidate_id,
            "source_role": source_role,
            "canonical_source_sha256": source_sha,
            "raw_source_bytes_sha256": source_sha,
            "provider_kind": adapter["provider_kind"],
            "adapter_id": adapter["adapter_id"],
            "adapter_module": adapter["adapter_module"],
            "adapter_callable": adapter["adapter_callable"],
            "adapter_executable_sha256": adapter[
                "adapter_executable_sha256"
            ],
            "adapter_config_sha256": adapter["adapter_config_sha256"],
            "provider_timestamp_utc": _format_utc(provider_timestamp),
            "fetched_at_utc": _format_utc(fetched_at),
            "source_watermark_sha256": source_watermark_sha256,
            "cutoff_utc": _format_utc(cutoff),
            "capture_manifest_file_sha256": manifest_file_sha,
            "capture_manifest_sha256": manifest["manifest_sha256"],
            "capture_key_id": manifest["capture_key_id"],
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }
        receipt_sha = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        receipt = {
            **body,
            "receipt_sha256": receipt_sha,
            "signature_base64": base64.b64encode(
                private_key.sign(_canonical_bytes(body))
            ).decode("ascii"),
        }
        _write_receipt(receipt_root, receipt)
        return receipt
    finally:
        os.close(lock_fd)


def verify_ai_source_capture_receipt(
    repository_root: Path,
    *,
    experiment_id: str,
    room_id: str,
    candidate_id: str,
    cutoff_utc: datetime | str,
    source_role: str,
    source_sha256: str,
    receipt_sha256: str,
) -> dict[str, Any]:
    """Revalidate canonical lifecycle proof and the full signed receipt chain."""

    return _verify_ai_source_capture_receipt(
        repository_root,
        experiment_id=experiment_id,
        room_id=room_id,
        candidate_id=candidate_id,
        cutoff_utc=cutoff_utc,
        source_role=source_role,
        source_sha256=source_sha256,
        receipt_sha256=receipt_sha256,
        allow_test_only=False,
    )


def verify_test_only_ai_source_capture_receipt(
    repository_root: Path,
    *,
    experiment_id: str,
    room_id: str,
    candidate_id: str,
    cutoff_utc: datetime | str,
    source_role: str,
    source_sha256: str,
    receipt_sha256: str,
) -> dict[str, Any]:
    """UNSAFE/test-only verifier for TEST_ONLY_CALLBACK receipts."""

    return _verify_ai_source_capture_receipt(
        repository_root,
        experiment_id=experiment_id,
        room_id=room_id,
        candidate_id=candidate_id,
        cutoff_utc=cutoff_utc,
        source_role=source_role,
        source_sha256=source_sha256,
        receipt_sha256=receipt_sha256,
        allow_test_only=True,
    )


def _verify_ai_source_capture_receipt(
    repository_root: Path,
    *,
    experiment_id: str,
    room_id: str,
    candidate_id: str,
    cutoff_utc: datetime | str,
    source_role: str,
    source_sha256: str,
    receipt_sha256: str,
    allow_test_only: bool,
) -> dict[str, Any]:
    root = _repository_root(repository_root)
    experiment_id = _capture_id(experiment_id, "experiment_id")
    room_id = _capture_id(room_id, "room_id")
    candidate_id = _sha(candidate_id, "candidate_id")
    cutoff = _utc(cutoff_utc, "cutoff_utc")
    source_role = _source_role(source_role)
    source_sha256 = _sha(source_sha256, "source_sha256")
    receipt_sha256 = _sha(receipt_sha256, "receipt_sha256")
    token = _verified_preflight(root, experiment_id, room_id, candidate_id)
    manifest, manifest_file_sha = _bound_manifest(root, token)
    receipt_root = _receipt_root(root, experiment_id, room_id, create=False)
    rows = _verify_receipt_chain(
        receipt_root,
        manifest=manifest,
        manifest_file_sha=manifest_file_sha,
        lifecycle_token=token,
        experiment_id=experiment_id,
        room_id=room_id,
        candidate_id=candidate_id,
    )
    matches = [row for row in rows if row["receipt_sha256"] == receipt_sha256]
    if len(matches) != 1:
        raise AiSourceCaptureError("capture receipt is absent or duplicated")
    receipt = matches[0]
    if (
        receipt.get("adapter_id") == "TEST_ONLY_CALLBACK"
        and not allow_test_only
    ):
        raise AiSourceCaptureError(
            "test-only capture receipt is not production evidence"
        )
    bindings = {
        "candidate_id": candidate_id,
        "source_role": source_role,
        "canonical_source_sha256": source_sha256,
        "raw_source_bytes_sha256": source_sha256,
        "cutoff_utc": _format_utc(cutoff),
    }
    for key, value in bindings.items():
        if receipt.get(key) != value:
            raise AiSourceCaptureError(f"capture receipt {key} binding mismatch")
    _verify_canonical_source(root, source_sha256)
    return dict(receipt)


def _verified_preflight(
    root: Path,
    experiment_id: str,
    room_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    try:
        token = verify_paper_ai_inventory_launch_preflight(
            root,
            experiment_id=experiment_id,
            room_id=room_id,
        )
    except (DojoReplayLifecycleError, OSError, TypeError, ValueError) as exc:
        raise AiSourceCaptureError(
            "canonical PAPER_ELIGIBLE preflight is invalid"
        ) from exc
    if token.get("candidate_id") != candidate_id:
        raise AiSourceCaptureError("preflight candidate binding mismatch")
    if (
        token.get("paper_only") is not True
        or token.get("order_authority") != "NONE"
        or token.get("live_permission") is not False
    ):
        raise AiSourceCaptureError("preflight paper authority is invalid")
    return dict(token)


def _bound_manifest(
    root: Path,
    token: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    manifest_file_sha = _sha(
        token.get(CAPTURE_MANIFEST_TOKEN_FIELD),
        "capture manifest file sha256",
    )
    manifest_root = _fixed_root(root, CAPTURE_ROOT / "manifests", create=False)
    path = manifest_root / f"{manifest_file_sha}.json"
    raw = _read_regular_file(path, MAX_SOURCE_BYTES, "capture manifest")
    if hashlib.sha256(raw).hexdigest() != manifest_file_sha:
        raise AiSourceCaptureError("capture manifest file digest mismatch")
    manifest = _normalize_manifest(
        _parse_canonical_json(raw, "capture manifest"),
        require_digest=True,
    )
    return manifest, manifest_file_sha


def _normalize_manifest(
    value: Mapping[str, Any],
    *,
    require_digest: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AiSourceCaptureError("capture manifest must be a mapping")
    try:
        snapshot = json.loads(_canonical_bytes(value))
    except (TypeError, ValueError) as exc:
        raise AiSourceCaptureError("capture manifest is not canonical JSON") from exc
    expected = _MANIFEST_KEYS if require_digest else _MANIFEST_KEYS - {
        "manifest_sha256"
    }
    if set(snapshot) != expected:
        raise AiSourceCaptureError("capture manifest schema is invalid")
    if snapshot.get("contract") != SOURCE_CAPTURE_MANIFEST_CONTRACT:
        raise AiSourceCaptureError("capture manifest contract is invalid")
    manifest_id = _capture_id(snapshot.get("manifest_id"), "manifest_id")
    key_id = _capture_id(snapshot.get("capture_key_id"), "capture_key_id")
    public_key = _public_key(snapshot.get("ed25519_public_key_base64"))
    roles = _sorted_unique_strings(
        snapshot.get("allowed_source_roles"),
        "allowed_source_roles",
        validator=_source_role,
    )
    providers = _sorted_unique_strings(
        snapshot.get("allowed_provider_kinds"),
        "allowed_provider_kinds",
        validator=lambda item: _capture_id(item, "provider_kind"),
    )
    adapters = _normalize_adapters(snapshot.get("source_adapters"))
    if [item["source_role"] for item in adapters] != roles:
        raise AiSourceCaptureError(
            "capture manifest adapters do not cover allowed roles exactly"
        )
    if sorted({item["provider_kind"] for item in adapters}) != providers:
        raise AiSourceCaptureError(
            "capture manifest adapter providers do not match allowlist"
        )
    if (
        snapshot.get("paper_only") is not True
        or snapshot.get("order_authority") != "NONE"
        or snapshot.get("live_permission") is not False
    ):
        raise AiSourceCaptureError("capture manifest paper authority is invalid")
    normalized = {
        "contract": SOURCE_CAPTURE_MANIFEST_CONTRACT,
        "manifest_id": manifest_id,
        "capture_key_id": key_id,
        "ed25519_public_key_base64": public_key,
        "allowed_source_roles": roles,
        "allowed_provider_kinds": providers,
        "source_adapters": adapters,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    digest = hashlib.sha256(_canonical_bytes(normalized)).hexdigest()
    if require_digest and snapshot.get("manifest_sha256") != digest:
        raise AiSourceCaptureError("capture manifest semantic digest mismatch")
    return {**normalized, "manifest_sha256": digest}


def _normalize_adapters(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise AiSourceCaptureError("source_adapters must be a non-empty array")
    result: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _ADAPTER_KEYS:
            raise AiSourceCaptureError(
                f"source_adapters[{index}] schema is invalid"
            )
        result.append(
            {
                "source_role": _source_role(raw.get("source_role")),
                "provider_kind": _capture_id(
                    raw.get("provider_kind"), "provider_kind"
                ),
                "adapter_id": _capture_id(
                    raw.get("adapter_id"), "adapter_id"
                ),
                "adapter_module": _capture_id(
                    raw.get("adapter_module"), "adapter_module"
                ),
                "adapter_callable": _capture_id(
                    raw.get("adapter_callable"), "adapter_callable"
                ),
                "adapter_executable_sha256": _sha(
                    raw.get("adapter_executable_sha256"),
                    "adapter_executable_sha256",
                ),
                "adapter_config_sha256": _sha(
                    raw.get("adapter_config_sha256"),
                    "adapter_config_sha256",
                ),
            }
        )
    if result != sorted(result, key=lambda item: item["source_role"]):
        raise AiSourceCaptureError(
            "source_adapters must be sorted by source_role"
        )
    roles = [item["source_role"] for item in result]
    if len(roles) != len(set(roles)):
        raise AiSourceCaptureError("source_adapters contains duplicate roles")
    return result


def _manifest_adapter(
    manifest: Mapping[str, Any], source_role: str
) -> dict[str, str]:
    matches = [
        item
        for item in manifest["source_adapters"]
        if item["source_role"] == source_role
    ]
    if len(matches) != 1:
        raise AiSourceCaptureError(
            "source role has no unique manifest-bound adapter"
        )
    return dict(matches[0])


def _verify_registered_adapter(
    registration: _RegisteredAdapter,
    binding: Mapping[str, str],
) -> None:
    if (
        binding.get("adapter_id") != registration.adapter_id
        or binding.get("adapter_module") != registration.module
        or binding.get("adapter_callable") != registration.callable_name
        or registration.acquire.__module__ != registration.module
        or registration.acquire.__name__ != registration.callable_name
    ):
        raise AiSourceCaptureError("registered adapter identity mismatch")
    source_path_text = inspect.getsourcefile(registration.acquire)
    if source_path_text is None:
        raise AiSourceCaptureError("registered adapter source is unavailable")
    source_path = Path(source_path_text)
    if not source_path.is_absolute():
        source_path = source_path.resolve()
    raw = _read_regular_file(
        source_path, MAX_SOURCE_BYTES, "registered adapter executable"
    )
    if hashlib.sha256(raw).hexdigest() != binding.get(
        "adapter_executable_sha256"
    ):
        raise AiSourceCaptureError("registered adapter executable digest mismatch")


def _load_adapter_config(
    root: Path, binding: Mapping[str, str]
) -> dict[str, Any]:
    digest = _sha(
        binding.get("adapter_config_sha256"),
        "manifest adapter_config_sha256",
    )
    config_root = _fixed_root(
        root, CAPTURE_ROOT / "adapter_configs", create=False
    )
    raw = _read_regular_file(
        config_root / f"{digest}.json",
        MAX_SOURCE_BYTES,
        "registered adapter config",
    )
    if hashlib.sha256(raw).hexdigest() != digest:
        raise AiSourceCaptureError("registered adapter config digest mismatch")
    value = _parse_canonical_json(raw, "registered adapter config")
    if not isinstance(value, dict):
        raise AiSourceCaptureError(
            "registered adapter config must be a JSON object"
        )
    try:
        canonical = canonical_source_adapter_config_bytes(value)
        derived_binding = source_adapter_capture_binding(value)
    except DojoAiSourceAdapterError as exc:
        raise AiSourceCaptureError(
            "registered adapter config semantic validation failed"
        ) from exc
    if canonical != raw:
        raise AiSourceCaptureError(
            "registered adapter config bytes are not canonical"
        )
    if dict(binding) != derived_binding:
        raise AiSourceCaptureError(
            "registered adapter config does not match manifest binding"
        )
    return dict(value)


def _normalize_acquisition(value: object) -> ReadOnlySourceAcquisition:
    required = {
        "raw_bytes",
        "provider_timestamp_utc",
        "source_watermark_sha256",
    }
    if isinstance(value, Mapping):
        if set(value) != required:
            raise AiSourceCaptureError(
                "read-only adapter acquisition schema is invalid"
            )
        snapshot = dict(value)
    elif is_dataclass(value) and not isinstance(value, type):
        field_names = {field.name for field in fields(value)}
        if field_names != required:
            raise AiSourceCaptureError(
                "read-only adapter acquisition schema is invalid"
            )
        snapshot = {field: getattr(value, field) for field in required}
    else:
        raise AiSourceCaptureError(
            "read-only adapter returned an invalid acquisition"
        )
    raw = snapshot["raw_bytes"]
    if raw.__class__ is not bytes:
        raise AiSourceCaptureError("read-only acquisition raw_bytes is invalid")
    provider_at = _utc(
        snapshot["provider_timestamp_utc"],
        "read-only acquisition provider_timestamp_utc",
    )
    watermark = _sha(
        snapshot["source_watermark_sha256"],
        "read-only acquisition source_watermark_sha256",
    )
    return ReadOnlySourceAcquisition(
        raw_bytes=raw,
        provider_timestamp_utc=_format_utc(provider_at),
        source_watermark_sha256=watermark,
    )


def _validate_acquisition_document(
    value: object,
    *,
    adapter: Mapping[str, str],
    acquisition: ReadOnlySourceAcquisition,
    fetched_at: datetime,
) -> None:
    if not isinstance(value, (dict, list)):
        raise AiSourceCaptureError(
            "captured source must be a JSON object or array"
        )
    provider_at = _utc(
        acquisition.provider_timestamp_utc,
        "read-only acquisition provider_timestamp_utc",
    )
    if provider_at > fetched_at:
        raise AiSourceCaptureError("captured source provider time is invalid")
    if adapter["adapter_id"] == OANDA_EXECUTABLE_QUOTE_ADAPTER_ID:
        _validate_registered_quote(value, provider_at=provider_at)
    elif adapter["adapter_id"] == OANDA_COMPLETED_BID_ASK_CANDLES_ADAPTER_ID:
        _validate_registered_candles(
            value,
            provider_at=provider_at,
            fetched_at=fetched_at,
        )
    else:
        return
    raw_sha = hashlib.sha256(acquisition.raw_bytes).hexdigest()
    if acquisition.source_watermark_sha256 != raw_sha:
        raise AiSourceCaptureError(
            "registered source watermark does not match exact raw bytes"
        )


def _validate_registered_quote(
    value: object,
    *,
    provider_at: datetime,
) -> None:
    if not isinstance(value, dict) or set(value) != _REGISTERED_QUOTE_KEYS:
        raise AiSourceCaptureError(
            "registered quote source schema is invalid"
        )
    pair = value.get("pair")
    if not isinstance(pair, str) or not _PAIR_RE.fullmatch(pair):
        raise AiSourceCaptureError("registered quote pair is invalid")
    bid = _positive_number(value.get("bid"), "registered quote bid")
    ask = _positive_number(value.get("ask"), "registered quote ask")
    if ask < bid:
        raise AiSourceCaptureError("registered quote source is crossed")
    _bounded_exact_int(
        value.get("max_age_seconds"),
        "registered quote max_age_seconds",
        minimum=1,
        maximum=180,
    )
    observed = _utc(
        value.get("timestamp_utc"),
        "registered quote source timestamp_utc",
    )
    if observed != provider_at:
        raise AiSourceCaptureError(
            "registered quote source/provider time mismatch"
        )


def _validate_registered_candles(
    value: object,
    *,
    provider_at: datetime,
    fetched_at: datetime,
) -> None:
    if not isinstance(value, list) or not value:
        raise AiSourceCaptureError(
            "registered candle source must be a non-empty JSON array"
        )
    expected_pair: str | None = None
    expected_granularity: str | None = None
    expected_max_age: int | None = None
    previous_completed: datetime | None = None
    for index, row in enumerate(value):
        field = f"registered candle source[{index}]"
        if not isinstance(row, dict) or set(row) != _REGISTERED_CANDLE_KEYS:
            raise AiSourceCaptureError(f"{field} schema is invalid")
        pair = row.get("pair")
        granularity = row.get("granularity")
        if not isinstance(pair, str) or not _PAIR_RE.fullmatch(pair):
            raise AiSourceCaptureError(f"{field} pair is invalid")
        if not isinstance(granularity, str) or not _GRANULARITY_RE.fullmatch(
            granularity
        ):
            raise AiSourceCaptureError(f"{field} granularity is invalid")
        max_age = _bounded_exact_int(
            row.get("max_age_seconds"),
            f"{field} max_age_seconds",
            minimum=1,
            maximum=24 * 60 * 60,
        )
        if expected_pair is None:
            expected_pair = pair
            expected_granularity = granularity
            expected_max_age = max_age
        elif (
            pair != expected_pair
            or granularity != expected_granularity
            or max_age != expected_max_age
        ):
            raise AiSourceCaptureError(
                "registered candle batch identity is inconsistent"
            )
        started = _utc(row.get("started_at_utc"), f"{field} started_at_utc")
        completed = _utc(
            row.get("completed_at_utc"),
            f"{field} completed_at_utc",
        )
        if started >= completed or completed > fetched_at:
            raise AiSourceCaptureError(f"{field} time bounds are invalid")
        if previous_completed is not None and completed <= previous_completed:
            raise AiSourceCaptureError(
                "registered candles are not strictly chronological"
            )
        previous_completed = completed
        prices = {
            key: _positive_number(row.get(key), f"{field} {key}")
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
                raise AiSourceCaptureError(f"{field} OHLC geometry is invalid")
        for suffix in ("o", "h", "l", "c"):
            if prices[f"ask_{suffix}"] < prices[f"bid_{suffix}"]:
                raise AiSourceCaptureError(f"{field} ask is below bid")
    if previous_completed != provider_at:
        raise AiSourceCaptureError(
            "registered candle source/provider time mismatch"
        )


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AiSourceCaptureError(f"{field} is invalid")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise AiSourceCaptureError(f"{field} is invalid")
    return normalized


def _bounded_exact_int(
    value: object,
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise AiSourceCaptureError(f"{field} is invalid")
    return value


def _verify_receipt_chain(
    receipt_root: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_file_sha: str,
    lifecycle_token: Mapping[str, Any],
    experiment_id: str,
    room_id: str,
    candidate_id: str,
) -> list[dict[str, Any]]:
    paths = sorted(receipt_root.glob("[0-9]" * 8 + "-*.json"))
    if len(paths) > MAX_RECEIPTS:
        raise AiSourceCaptureError("capture receipt chain exceeds fixed bound")
    expected_previous = "0" * 64
    rows: list[dict[str, Any]] = []
    public_key = Ed25519PublicKey.from_public_bytes(
        base64.b64decode(
            manifest["ed25519_public_key_base64"], validate=True
        )
    )
    for sequence, path in enumerate(paths, 1):
        raw = _read_regular_file(path, 256 * 1024, "capture receipt")
        receipt = _parse_canonical_json(raw, "capture receipt")
        if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
            raise AiSourceCaptureError("capture receipt schema is invalid")
        body = {key: receipt[key] for key in _RECEIPT_BODY_KEYS}
        digest = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        expected_name = f"{sequence:08d}-{digest}.json"
        if (
            receipt.get("contract") != SOURCE_CAPTURE_RECEIPT_CONTRACT
            or receipt.get("sequence") != sequence
            or receipt.get("previous_receipt_sha256") != expected_previous
            or receipt.get("receipt_sha256") != digest
            or path.name != expected_name
        ):
            raise AiSourceCaptureError("capture receipt chain is invalid")
        if (
            receipt.get("capture_manifest_file_sha256") != manifest_file_sha
            or receipt.get("capture_manifest_sha256")
            != manifest["manifest_sha256"]
            or receipt.get("capture_key_id") != manifest["capture_key_id"]
        ):
            raise AiSourceCaptureError("capture receipt manifest binding mismatch")
        if (
            receipt.get("paper_only") is not True
            or receipt.get("order_authority") != "NONE"
            or receipt.get("live_permission") is not False
        ):
            raise AiSourceCaptureError("capture receipt paper authority is invalid")
        _validate_receipt_fields(
            receipt,
            sequence=sequence,
            manifest=manifest,
            manifest_file_sha=manifest_file_sha,
            lifecycle_token=lifecycle_token,
            experiment_id=experiment_id,
            room_id=room_id,
            candidate_id=candidate_id,
        )
        signature = _signature(receipt.get("signature_base64"))
        try:
            public_key.verify(signature, _canonical_bytes(body))
        except InvalidSignature as exc:
            raise AiSourceCaptureError(
                "capture receipt signature is invalid"
            ) from exc
        expected_previous = digest
        rows.append(dict(receipt))
    return rows


def _validate_receipt_fields(
    receipt: Mapping[str, Any],
    *,
    sequence: int,
    manifest: Mapping[str, Any],
    manifest_file_sha: str,
    lifecycle_token: Mapping[str, Any],
    experiment_id: str,
    room_id: str,
    candidate_id: str,
) -> None:
    if receipt.get("sequence") != sequence or isinstance(
        receipt.get("sequence"), bool
    ):
        raise AiSourceCaptureError("capture receipt sequence is invalid")
    if receipt.get("experiment_id") != experiment_id:
        raise AiSourceCaptureError("capture receipt experiment binding mismatch")
    if receipt.get("room_id") != room_id:
        raise AiSourceCaptureError("capture receipt room binding mismatch")
    if receipt.get("candidate_id") != candidate_id:
        raise AiSourceCaptureError("capture receipt candidate binding mismatch")
    _sha(
        receipt.get("previous_receipt_sha256"),
        "capture receipt previous_receipt_sha256",
    )
    source_role = _source_role(receipt.get("source_role"))
    if source_role not in manifest["allowed_source_roles"]:
        raise AiSourceCaptureError("capture receipt source role is not allowed")
    canonical_sha = _sha(
        receipt.get("canonical_source_sha256"),
        "capture receipt canonical_source_sha256",
    )
    raw_sha = _sha(
        receipt.get("raw_source_bytes_sha256"),
        "capture receipt raw_source_bytes_sha256",
    )
    if canonical_sha != raw_sha:
        raise AiSourceCaptureError("capture receipt source digest mismatch")
    provider_kind = _capture_id(
        receipt.get("provider_kind"), "capture receipt provider_kind"
    )
    if provider_kind not in manifest["allowed_provider_kinds"]:
        raise AiSourceCaptureError("capture receipt provider kind is not allowed")
    adapter = _manifest_adapter(manifest, source_role)
    for field in (
        "provider_kind",
        "adapter_id",
        "adapter_module",
        "adapter_callable",
        "adapter_executable_sha256",
        "adapter_config_sha256",
    ):
        if receipt.get(field) != adapter[field]:
            raise AiSourceCaptureError(
                f"capture receipt {field} binding mismatch"
            )
    provider_at = _utc(
        receipt.get("provider_timestamp_utc"),
        "capture receipt provider_timestamp_utc",
    )
    fetched_at = _utc(
        receipt.get("fetched_at_utc"), "capture receipt fetched_at_utc"
    )
    cutoff = _utc(receipt.get("cutoff_utc"), "capture receipt cutoff_utc")
    if provider_at > fetched_at or fetched_at > cutoff:
        raise AiSourceCaptureError("capture receipt time ordering is invalid")
    _require_market_open(fetched_at, "capture receipt fetched_at_utc")
    _require_cutoff_in_future_window(cutoff, lifecycle_token)
    _sha(
        receipt.get("source_watermark_sha256"),
        "capture receipt source_watermark_sha256",
    )
    if (
        receipt.get("capture_manifest_file_sha256") != manifest_file_sha
        or receipt.get("capture_manifest_sha256")
        != manifest["manifest_sha256"]
        or receipt.get("capture_key_id") != manifest["capture_key_id"]
    ):
        raise AiSourceCaptureError("capture receipt manifest binding mismatch")


def _write_canonical_source(root: Path, digest: str, raw: bytes) -> None:
    source_root = _fixed_root(root, CANONICAL_SOURCE_ROOT, create=True)
    path = source_root / f"{digest}.json"
    _write_exclusive_or_identical(path, raw, "canonical source")


def _verify_canonical_source(root: Path, digest: str) -> None:
    source_root = _fixed_root(root, CANONICAL_SOURCE_ROOT, create=False)
    path = source_root / f"{digest}.json"
    raw = _read_regular_file(path, MAX_SOURCE_BYTES, "canonical source")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise AiSourceCaptureError("canonical source digest mismatch")
    _parse_canonical_json(raw, "canonical source")


def _write_receipt(root: Path, receipt: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(receipt) + b"\n"
    path = root / (
        f"{receipt['sequence']:08d}-{receipt['receipt_sha256']}.json"
    )
    _write_exclusive_or_identical(path, raw, "capture receipt")


def _write_exclusive_or_identical(path: Path, raw: bytes, label: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        if _read_regular_file(path, max(len(raw), 1), label) != raw:
            raise AiSourceCaptureError(f"existing {label} is not identical")
        return
    except OSError as exc:
        raise AiSourceCaptureError(f"{label} exclusive create failed") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _load_private_key(manifest: Mapping[str, Any]) -> Ed25519PrivateKey:
    raw_path = os.environ.get(CAPTURE_PRIVATE_KEY_ENV)
    if not raw_path:
        raise AiSourceCaptureError("capture private key path is not configured")
    path = Path(raw_path)
    if not path.is_absolute():
        raise AiSourceCaptureError("capture private key path must be absolute")
    raw = _read_regular_file(path, 64 * 1024, "capture private key")
    try:
        key = serialization.load_pem_private_key(raw, password=None)
    except (TypeError, ValueError) as exc:
        raise AiSourceCaptureError("capture private key is invalid") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise AiSourceCaptureError("capture private key is not Ed25519")
    public_raw = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    actual = base64.b64encode(public_raw).decode("ascii")
    if actual != manifest["ed25519_public_key_base64"]:
        raise AiSourceCaptureError(
            "capture private key does not match bound manifest"
        )
    return key


def _receipt_root(
    root: Path,
    experiment_id: str,
    room_id: str,
    *,
    create: bool,
) -> Path:
    relative = CAPTURE_ROOT / "receipts" / experiment_id / room_id
    return _fixed_root(root, relative, create=create)


def _trusted_repository_root() -> Path:
    try:
        root = Path(__file__).resolve(strict=True).parents[2]
    except (IndexError, OSError) as exc:
        raise AiSourceCaptureError(
            "package-derived repository root is unavailable"
        ) from exc
    return _repository_root(root)


def _repository_root(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise AiSourceCaptureError("repository root must be an absolute Path")
    try:
        root_stat = value.lstat()
        git_stat = (value / ".git").lstat()
    except OSError as exc:
        raise AiSourceCaptureError("repository root is unavailable") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise AiSourceCaptureError("repository root is unsafe")
    if stat.S_ISLNK(git_stat.st_mode) or not (
        stat.S_ISDIR(git_stat.st_mode) or stat.S_ISREG(git_stat.st_mode)
    ):
        raise AiSourceCaptureError("repository Git metadata is unsafe")
    return value.resolve(strict=True)


def _fixed_root(root: Path, relative: Path, *, create: bool) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        try:
            item_stat = current.lstat()
        except FileNotFoundError:
            if not create:
                raise AiSourceCaptureError("fixed capture root does not exist")
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                item_stat = current.lstat()
                if stat.S_ISLNK(item_stat.st_mode) or not stat.S_ISDIR(
                    item_stat.st_mode
                ):
                    raise AiSourceCaptureError(
                        "fixed capture root is unsafe"
                    ) from None
            _fsync_directory(current.parent)
            continue
        except OSError as exc:
            raise AiSourceCaptureError("fixed capture root is unavailable") from exc
        if stat.S_ISLNK(item_stat.st_mode) or not stat.S_ISDIR(item_stat.st_mode):
            raise AiSourceCaptureError("fixed capture root is unsafe")
    resolved = current.resolve(strict=True)
    if resolved != root / relative:
        raise AiSourceCaptureError("fixed capture root escaped repository")
    return resolved


def _read_regular_file(path: Path, maximum: int, label: str) -> bytes:
    try:
        item_stat = path.lstat()
        if stat.S_ISLNK(item_stat.st_mode) or not stat.S_ISREG(item_stat.st_mode):
            raise AiSourceCaptureError(f"{label} is not a regular file")
        if item_stat.st_size <= 0 or item_stat.st_size > maximum:
            raise AiSourceCaptureError(f"{label} size is invalid")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            raw = handle.read(maximum + 1)
    except AiSourceCaptureError:
        raise
    except OSError as exc:
        raise AiSourceCaptureError(f"{label} cannot be read") from exc
    if len(raw) != item_stat.st_size:
        raise AiSourceCaptureError(f"{label} changed while reading")
    return raw


def _parse_canonical_json(raw: bytes, label: str) -> Any:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise AiSourceCaptureError(
            f"{label} must be one canonical newline-terminated JSON document"
        )
    try:
        value = json.loads(
            raw[:-1],
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AiSourceCaptureError(f"{label} JSON is invalid") from exc
    if _canonical_bytes(value) + b"\n" != raw:
        raise AiSourceCaptureError(f"{label} JSON is noncanonical")
    return value


def _public_key(value: object) -> str:
    if not isinstance(value, str):
        raise AiSourceCaptureError("capture public key is invalid")
    try:
        raw = base64.b64decode(value, validate=True)
        Ed25519PublicKey.from_public_bytes(raw)
    except (ValueError, TypeError) as exc:
        raise AiSourceCaptureError("capture public key is invalid") from exc
    if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != value:
        raise AiSourceCaptureError("capture public key is noncanonical")
    return value


def _signature(value: object) -> bytes:
    if not isinstance(value, str):
        raise AiSourceCaptureError("capture signature is invalid")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise AiSourceCaptureError("capture signature is invalid") from exc
    if len(raw) != 64 or base64.b64encode(raw).decode("ascii") != value:
        raise AiSourceCaptureError("capture signature is noncanonical")
    return raw


def _sorted_unique_strings(
    value: object,
    field: str,
    *,
    validator: Callable[[object], str],
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise AiSourceCaptureError(f"{field} must be a non-empty array")
    result = [validator(item) for item in value]
    if result != sorted(set(result)):
        raise AiSourceCaptureError(f"{field} must be sorted and unique")
    return result


def _require_cutoff_in_future_window(
    cutoff: datetime,
    token: Mapping[str, Any],
) -> None:
    window = token.get("future_window")
    if not isinstance(window, Mapping):
        raise AiSourceCaptureError("preflight future window is invalid")
    start = _utc(window.get("start_utc"), "future_window.start_utc")
    end = _utc(window.get("end_utc"), "future_window.end_utc")
    if cutoff < start or cutoff >= end:
        raise AiSourceCaptureError("capture cutoff is outside future window")


def _require_market_open(value: datetime, field: str) -> None:
    try:
        status = compute_market_status(value)
    except Exception as exc:
        raise AiSourceCaptureError(f"FX status unavailable for {field}") from exc
    if not status.is_fx_open:
        raise AiSourceCaptureMarketClosedError(
            f"read-only capture is disabled while FX is closed ({field})"
        )


def _capture_id(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not _ID_RE.fullmatch(value)
        or Path(value).name != value
    ):
        raise AiSourceCaptureError(f"{field} is invalid")
    return value


def _source_role(value: object) -> str:
    if not isinstance(value, str) or not _ROLE_RE.fullmatch(value):
        raise AiSourceCaptureError("source_role is invalid")
    return value


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise AiSourceCaptureError(f"{field} is invalid")
    return value


def _utc(value: datetime | str | object, field: str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise AiSourceCaptureError(f"{field} is invalid") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise AiSourceCaptureError(f"{field} is invalid")
    if parsed.tzinfo is None:
        raise AiSourceCaptureError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        os.fsync(descriptor)
    except OSError as exc:
        raise AiSourceCaptureError("capture directory fsync failed") from exc
    finally:
        if "descriptor" in locals():
            os.close(descriptor)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

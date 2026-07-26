"""Trusted external-worker receipts for autonomous DOJO replay proof.

Production ships with an empty worker allowlist.  A replay result therefore
cannot be promoted until trusted code preregisters the exact executable,
adapter configuration, producer identity, model identity, and Ed25519 public
key.  Receipt verification opens every bound file with ``O_NOFOLLOW`` and
checks the bytes again; a self-hashed metrics document is never authority.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


REPLAY_WORKER_RECEIPT_CONTRACT = "QR_DOJO_REPLAY_WORKER_RECEIPT_V1"
MAX_RECEIPT_BYTES = 256 * 1024
WINDOWS = ("TRAIN", "VAL", "S5")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,254}$")

_CONFIG_KEYS = frozenset(
    {
        "adapter_id",
        "model_id",
        "producer_id",
        "executable_path",
        "executable_sha256",
        "signature_key_id",
        "ed25519_public_key_base64",
        "config_sha256",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "contract",
        "adapter_id",
        "model_id",
        "config_sha256",
        "producer_id",
        "executable_path",
        "executable_sha256",
        "executable_device",
        "executable_inode",
        "executable_uid",
        "executable_gid",
        "argv",
        "argv_sha256",
        "git_head",
        "git_head_sha256",
        "candidate_id",
        "spec_sha256",
        "policy_sha256",
        "job_manifest_sha256",
        "output_manifest_sha256",
        "source_files",
        "windows",
        "costs",
        "intrabar_paths",
        "results_artifact_path",
        "results_artifact_sha256",
        "completed_at_utc",
        "paper_only",
        "order_authority",
        "live_permission",
        "signature_key_id",
        "signed_payload_sha256",
        "signature_base64",
        "receipt_sha256",
    }
)
_SIGNED_EXCLUSIONS = frozenset(
    {"signed_payload_sha256", "signature_base64", "receipt_sha256"}
)

# Deliberately empty in production.  Tests may patch this code-owned mapping
# with an ephemeral key and an exact executable manifest.
_TRUSTED_REPLAY_WORKERS: Mapping[str, Mapping[str, Any]] = MappingProxyType({})


class ReplayWorkerReceiptError(ValueError):
    """A replay worker or its externally signed receipt is untrusted."""


def replay_worker_config_sha256(value: Mapping[str, Any]) -> str:
    """Return the digest used by the immutable trusted-worker allowlist."""

    return _normalize_config(value, require_digest=False)["config_sha256"]


def validate_replay_worker_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one complete trusted-worker public config."""

    return _normalize_config(value, require_digest=True)


def verify_trusted_replay_worker_receipt(
    repository_root: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Verify one canonical signed receipt and every file it binds."""

    if not isinstance(repository_root, Path) or not isinstance(receipt_path, Path):
        raise ReplayWorkerReceiptError(
            "repository_root and receipt_path must be explicit Paths"
        )
    root = _resolved_directory(repository_root, "repository root")
    raw, _ = _read_regular_no_follow(
        receipt_path,
        allowed_root=root,
        max_bytes=MAX_RECEIPT_BYTES,
        label="replay worker receipt",
    )
    receipt = _strict_json(raw, "replay worker receipt")
    if raw != _canonical_bytes(receipt) + b"\n":
        raise ReplayWorkerReceiptError("replay worker receipt is not canonical")
    if set(receipt) != _RECEIPT_KEYS:
        raise ReplayWorkerReceiptError("replay worker receipt schema is invalid")
    if receipt.get("contract") != REPLAY_WORKER_RECEIPT_CONTRACT:
        raise ReplayWorkerReceiptError("replay worker receipt contract is invalid")
    _paper_guard(receipt, "replay worker receipt")

    adapter_id = _identifier(receipt.get("adapter_id"), "adapter_id")
    configured = _TRUSTED_REPLAY_WORKERS.get(adapter_id)
    if configured is None:
        raise ReplayWorkerReceiptError(
            "replay worker is absent from the production allowlist"
        )
    config = _normalize_config(configured, require_digest=True)
    for field in (
        "adapter_id",
        "model_id",
        "producer_id",
        "config_sha256",
        "executable_path",
        "executable_sha256",
        "signature_key_id",
    ):
        if receipt.get(field) != config[field]:
            raise ReplayWorkerReceiptError(
                f"replay worker receipt {field} allowlist mismatch"
            )

    executable_path = Path(config["executable_path"])
    executable_sha256, executable_stat = _digest_regular_no_follow(
        executable_path,
        allowed_root=None,
        label="replay worker executable",
    )
    if executable_stat.st_mode & 0o111 == 0:
        raise ReplayWorkerReceiptError("replay worker executable is not executable")
    expected_executable = {
        "executable_sha256": executable_sha256,
        "executable_device": executable_stat.st_dev,
        "executable_inode": executable_stat.st_ino,
        "executable_uid": executable_stat.st_uid,
        "executable_gid": executable_stat.st_gid,
    }
    for field, expected in expected_executable.items():
        if receipt.get(field) != expected or type(receipt.get(field)) is not type(
            expected
        ):
            raise ReplayWorkerReceiptError(
                f"replay worker receipt {field} executable mismatch"
            )

    argv = receipt.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or argv[0] != config["executable_path"]
        or any(
            not isinstance(item, str)
            or not item
            or len(item) > 16_384
            or "\x00" in item
            for item in argv
        )
    ):
        raise ReplayWorkerReceiptError("replay worker argv is invalid")
    if receipt.get("argv_sha256") != _sha256(_canonical_bytes(argv)):
        raise ReplayWorkerReceiptError("replay worker argv digest mismatch")

    git_head = receipt.get("git_head")
    if not isinstance(git_head, str) or _GIT_RE.fullmatch(git_head) is None:
        raise ReplayWorkerReceiptError("replay worker git_head is invalid")
    if receipt.get("git_head_sha256") != _sha256(git_head.encode("ascii")):
        raise ReplayWorkerReceiptError("replay worker git_head digest mismatch")
    for field in (
        "candidate_id",
        "spec_sha256",
        "policy_sha256",
        "job_manifest_sha256",
        "output_manifest_sha256",
        "results_artifact_sha256",
    ):
        _sha(receipt.get(field), field)

    windows = receipt.get("windows")
    if not isinstance(windows, Mapping) or set(windows) != set(WINDOWS):
        raise ReplayWorkerReceiptError(
            "replay worker windows must be TRAIN, VAL, and S5"
        )
    if receipt.get("costs") != ["BASE", "STRESS"]:
        raise ReplayWorkerReceiptError("replay worker costs are not BASE/STRESS")
    if receipt.get("intrabar_paths") != ["OHLC", "OLHC"]:
        raise ReplayWorkerReceiptError("replay worker intrabar paths are not OHLC/OLHC")

    source_files = receipt.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ReplayWorkerReceiptError("replay worker source files are missing")
    normalized_sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in source_files:
        if not isinstance(item, Mapping) or set(item) != {
            "window",
            "path",
            "sha256",
        }:
            raise ReplayWorkerReceiptError(
                "replay worker source file binding is invalid"
            )
        window = item.get("window")
        path_text = item.get("path")
        if (
            window not in WINDOWS
            or not isinstance(path_text, str)
            or not path_text
            or (window, path_text) in seen
        ):
            raise ReplayWorkerReceiptError(
                "replay worker source file identity is invalid"
            )
        seen.add((window, path_text))
        expected_sha256 = _sha(item.get("sha256"), "source file sha256")
        actual_sha256, _ = _digest_regular_no_follow(
            root / path_text,
            allowed_root=root,
            label=f"{window} bid-ask source",
        )
        if actual_sha256 != expected_sha256:
            raise ReplayWorkerReceiptError(
                f"{window} bid-ask source bytes digest mismatch"
            )
        normalized_sources.append(
            {
                "window": str(window),
                "path": path_text,
                "sha256": expected_sha256,
            }
        )
    if normalized_sources != source_files:
        raise ReplayWorkerReceiptError(
            "replay worker source file order is not canonical"
        )

    results_path = receipt.get("results_artifact_path")
    if not isinstance(results_path, str) or not results_path:
        raise ReplayWorkerReceiptError("results artifact path is invalid")
    actual_result_sha256, _ = _digest_regular_no_follow(
        root / results_path,
        allowed_root=root,
        label="replay results artifact",
    )
    if actual_result_sha256 != receipt["results_artifact_sha256"]:
        raise ReplayWorkerReceiptError("replay results artifact bytes mismatch")

    _identifier(receipt.get("model_id"), "model_id")
    _identifier(receipt.get("producer_id"), "producer_id")
    _identifier(receipt.get("signature_key_id"), "signature_key_id")
    _utc(receipt.get("completed_at_utc"), "completed_at_utc")

    signed_body = {
        key: receipt[key] for key in receipt if key not in _SIGNED_EXCLUSIONS
    }
    signed_payload = _canonical_bytes(signed_body)
    signed_payload_sha256 = _sha(
        receipt.get("signed_payload_sha256"),
        "signed_payload_sha256",
    )
    if signed_payload_sha256 != _sha256(signed_payload):
        raise ReplayWorkerReceiptError("signed replay payload digest mismatch")
    signature = _canonical_base64(receipt.get("signature_base64"), "signature")
    public_key = _canonical_base64(
        config["ed25519_public_key_base64"],
        "Ed25519 public key",
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            signed_payload,
        )
    except (ValueError, InvalidSignature) as exc:
        raise ReplayWorkerReceiptError(
            "replay worker Ed25519 signature is invalid"
        ) from exc

    receipt_body = {key: receipt[key] for key in receipt if key != "receipt_sha256"}
    receipt_sha256 = _sha(receipt.get("receipt_sha256"), "receipt_sha256")
    if receipt_sha256 != _sha256(_canonical_bytes(receipt_body)):
        raise ReplayWorkerReceiptError("replay worker receipt digest mismatch")
    return dict(receipt)


def _normalize_config(
    value: Mapping[str, Any],
    *,
    require_digest: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayWorkerReceiptError("trusted replay worker config is invalid")
    snapshot = _strict_json(_canonical_bytes(value), "trusted replay worker config")
    expected = _CONFIG_KEYS if require_digest else _CONFIG_KEYS - {"config_sha256"}
    if set(snapshot) != expected:
        raise ReplayWorkerReceiptError("trusted replay worker config schema is invalid")
    executable = snapshot.get("executable_path")
    if (
        not isinstance(executable, str)
        or not Path(executable).is_absolute()
        or not executable
        or "\x00" in executable
    ):
        raise ReplayWorkerReceiptError(
            "trusted replay worker executable path is invalid"
        )
    body = {
        "adapter_id": _identifier(snapshot.get("adapter_id"), "adapter_id"),
        "model_id": _identifier(snapshot.get("model_id"), "model_id"),
        "producer_id": _identifier(snapshot.get("producer_id"), "producer_id"),
        "executable_path": executable,
        "executable_sha256": _sha(
            snapshot.get("executable_sha256"),
            "executable_sha256",
        ),
        "signature_key_id": _identifier(
            snapshot.get("signature_key_id"),
            "signature_key_id",
        ),
        "ed25519_public_key_base64": base64.b64encode(
            _canonical_base64(
                snapshot.get("ed25519_public_key_base64"),
                "Ed25519 public key",
            )
        ).decode("ascii"),
    }
    digest = _sha256(_canonical_bytes(body))
    if require_digest and snapshot.get("config_sha256") != digest:
        raise ReplayWorkerReceiptError("trusted replay worker config digest mismatch")
    return {**body, "config_sha256": digest}


def _digest_regular_no_follow(
    path: Path,
    *,
    allowed_root: Path | None,
    label: str,
) -> tuple[str, os.stat_result]:
    digest = hashlib.sha256()
    descriptor, before = _open_regular_no_follow(
        path,
        allowed_root=allowed_root,
        label=label,
    )
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        after.st_size <= 0
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ReplayWorkerReceiptError(f"{label} changed while hashing")
    return digest.hexdigest(), after


def _read_regular_no_follow(
    path: Path,
    *,
    allowed_root: Path,
    max_bytes: int,
    label: str,
) -> tuple[bytes, os.stat_result]:
    descriptor, before = _open_regular_no_follow(
        path,
        allowed_root=allowed_root,
        label=label,
    )
    try:
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise ReplayWorkerReceiptError(f"{label} size is invalid")
        raw = bytearray()
        while len(raw) <= max_bytes:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(raw) > max_bytes
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ReplayWorkerReceiptError(f"{label} changed while reading")
    return bytes(raw), after


def _open_regular_no_follow(
    path: Path,
    *,
    allowed_root: Path | None,
    label: str,
) -> tuple[int, os.stat_result]:
    descriptor: int | None = None
    try:
        target = path if path.is_absolute() else path.absolute()
        if allowed_root is not None:
            root = allowed_root.resolve(strict=True)
            lexical = Path(os.path.abspath(target))
            lexical.relative_to(root)
            relative = lexical.relative_to(root)
            current = root
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    raise OSError("symlink component")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        lexical_stat = os.lstat(target)
        if stat.S_ISLNK(lexical_stat.st_mode):
            raise OSError("symlink target")
        descriptor = os.open(target, flags)
        item_stat = os.fstat(descriptor)
    except (OSError, ValueError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ReplayWorkerReceiptError(
            f"{label} is unavailable or outside its canonical root"
        ) from exc
    if (
        not stat.S_ISREG(item_stat.st_mode)
        or stat.S_ISLNK(lexical_stat.st_mode)
        or item_stat.st_dev != lexical_stat.st_dev
        or item_stat.st_ino != lexical_stat.st_ino
    ):
        os.close(descriptor)
        raise ReplayWorkerReceiptError(f"{label} is not a regular no-follow file")
    return descriptor, item_stat


def _resolved_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ReplayWorkerReceiptError(f"{label} is unavailable") from exc
    if not resolved.is_dir():
        raise ReplayWorkerReceiptError(f"{label} is not a directory")
    return resolved


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReplayWorkerReceiptError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ReplayWorkerReceiptError(f"{label} is not a JSON object")
    return value


def _paper_guard(value: Mapping[str, Any], label: str) -> None:
    if (
        value.get("paper_only") is not True
        or value.get("order_authority") != "NONE"
        or value.get("live_permission") is not False
    ):
        raise ReplayWorkerReceiptError(
            f"{label} must be paper-only with order authority NONE"
        )


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ReplayWorkerReceiptError(f"{label} is invalid")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ReplayWorkerReceiptError(f"{label} is not a SHA-256 digest")
    return value


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith(("Z", "+00:00")):
        raise ReplayWorkerReceiptError(f"{label} is not a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReplayWorkerReceiptError(f"{label} is not a UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ReplayWorkerReceiptError(f"{label} is not a UTC timestamp")
    return value


def _canonical_base64(value: Any, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ReplayWorkerReceiptError(f"{label} is invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ReplayWorkerReceiptError(f"{label} is invalid") from exc
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ReplayWorkerReceiptError(f"{label} is not canonical base64")
    return decoded


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")

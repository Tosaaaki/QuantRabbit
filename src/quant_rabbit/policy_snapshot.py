from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_SNAPSHOT_SCHEMA_VERSION = 1


class PolicySnapshotError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PolicyBinding:
    project_key: str
    broker_account_id: str
    environment: str
    revocation_epoch: int


def seal_policy_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a content-addressed control-plane snapshot for the offline hot path."""

    body = {str(key): value for key, value in payload.items() if key != "snapshot_sha256"}
    _validate_shape(body)
    return {**body, "snapshot_sha256": _canonical_sha256(body)}


def write_sealed_policy_snapshot(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish one control-plane snapshot after validating its shape."""

    sealed = seal_policy_snapshot(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise PolicySnapshotError("POLICY_SNAPSHOT_PATH_UNSAFE", "policy snapshot path is a symlink")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(sealed, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return sealed


def load_and_verify_policy_snapshot(
    path: Path,
    *,
    binding: PolicyBinding,
    now: datetime | None = None,
    required_source_pages: Sequence[str] = (),
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PolicySnapshotError(
            "POLICY_SNAPSHOT_MISSING",
            f"sealed policy snapshot is missing: {path}",
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicySnapshotError(
            "POLICY_SNAPSHOT_UNREADABLE",
            f"sealed policy snapshot is unreadable: {path}: {exc}",
        ) from exc
    if not isinstance(value, Mapping):
        raise PolicySnapshotError("POLICY_SNAPSHOT_INVALID", "policy snapshot must be an object")
    return verify_policy_snapshot(
        value,
        binding=binding,
        now=now,
        required_source_pages=required_source_pages,
    )


def verify_policy_snapshot(
    value: Mapping[str, Any],
    *,
    binding: PolicyBinding,
    now: datetime | None = None,
    required_source_pages: Sequence[str] = (),
) -> dict[str, Any]:
    snapshot = dict(value)
    stored_sha = snapshot.pop("snapshot_sha256", None)
    if not isinstance(stored_sha, str) or stored_sha != _canonical_sha256(snapshot):
        raise PolicySnapshotError(
            "POLICY_SNAPSHOT_TAMPERED",
            "policy snapshot content digest does not match",
        )
    _validate_shape(snapshot)

    current = _utc_now(now)
    issued = _parse_utc(snapshot["issued_at_utc"], "issued_at_utc")
    expires = _parse_utc(snapshot["expires_at_utc"], "expires_at_utc")
    if issued > current:
        raise PolicySnapshotError("POLICY_SNAPSHOT_NOT_YET_VALID", "policy snapshot is not yet valid")
    if expires <= current:
        raise PolicySnapshotError("POLICY_SNAPSHOT_EXPIRED", "policy snapshot has expired")

    expected = {
        "project_key": binding.project_key,
        "broker_account_id": binding.broker_account_id,
        "environment": binding.environment,
        "revocation_epoch": binding.revocation_epoch,
    }
    mismatched = [key for key, expected_value in expected.items() if snapshot.get(key) != expected_value]
    if mismatched:
        raise PolicySnapshotError(
            "POLICY_SNAPSHOT_BINDING_MISMATCH",
            "policy snapshot binding differs for: " + ", ".join(mismatched),
        )

    pages = snapshot.get("source_pages")
    page_ids = {
        str(page.get("page_id"))
        for page in pages
        if isinstance(page, Mapping) and isinstance(page.get("page_id"), str)
    }
    missing_pages = sorted(set(required_source_pages) - page_ids)
    if missing_pages:
        raise PolicySnapshotError(
            "POLICY_SNAPSHOT_SOURCE_MISMATCH",
            "policy snapshot does not bind required source pages: " + ", ".join(missing_pages),
        )

    hot_path = snapshot["hot_path"]
    required_rules = {
        "notion_access_allowed": False,
        "browser_access_allowed": False,
        "legacy_strategy_authority": "BASELINE_ONLY",
        "manual_positions": "NO_TOUCH",
    }
    mismatched_rules = [key for key, expected_value in required_rules.items() if hot_path.get(key) != expected_value]
    if mismatched_rules:
        raise PolicySnapshotError(
            "POLICY_SNAPSHOT_RULE_MISMATCH",
            "hot-path policy differs for: " + ", ".join(mismatched_rules),
        )
    return dict(value)


def _validate_shape(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != POLICY_SNAPSHOT_SCHEMA_VERSION:
        raise PolicySnapshotError("POLICY_SNAPSHOT_SCHEMA_MISMATCH", "unsupported policy snapshot schema")
    for field in (
        "policy_version",
        "project_key",
        "broker_account_id",
        "environment",
        "issued_at_utc",
        "expires_at_utc",
    ):
        if not isinstance(value.get(field), str) or not str(value[field]).strip():
            raise PolicySnapshotError("POLICY_SNAPSHOT_INVALID", f"{field} must be a non-empty string")
    epoch = value.get("revocation_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise PolicySnapshotError("POLICY_SNAPSHOT_INVALID", "revocation_epoch must be a non-negative integer")
    pages = value.get("source_pages")
    if not isinstance(pages, list) or not pages:
        raise PolicySnapshotError("POLICY_SNAPSHOT_INVALID", "source_pages must be a non-empty list")
    for page in pages:
        if not isinstance(page, Mapping):
            raise PolicySnapshotError("POLICY_SNAPSHOT_INVALID", "source page must be an object")
        for field in ("page_id", "last_edited_at_utc"):
            if not isinstance(page.get(field), str) or not str(page[field]).strip():
                raise PolicySnapshotError("POLICY_SNAPSHOT_INVALID", f"source page {field} is required")
        _parse_utc(page["last_edited_at_utc"], "source_pages.last_edited_at_utc")
    hot_path = value.get("hot_path")
    if not isinstance(hot_path, Mapping):
        raise PolicySnapshotError("POLICY_SNAPSHOT_INVALID", "hot_path must be an object")


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise PolicySnapshotError("POLICY_SNAPSHOT_INVALID", f"{label} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PolicySnapshotError("POLICY_SNAPSHOT_INVALID", f"invalid {label}: {value}") from exc
    if parsed.tzinfo is None:
        raise PolicySnapshotError("POLICY_SNAPSHOT_INVALID", f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise PolicySnapshotError("POLICY_SNAPSHOT_INVALID", "now must include a timezone")
    return current.astimezone(timezone.utc)

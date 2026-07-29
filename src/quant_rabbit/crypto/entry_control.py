from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_STATUSES = {
    "ACTIVE",
    "FORWARD_PAPER_ONLY",
    "PAUSE_NEW_ENTRIES",
    "QUARANTINE_NEW_ENTRIES",
}
BLOCKED_STATUSES = {
    "PAUSE_NEW_ENTRIES",
    "QUARANTINE_NEW_ENTRIES",
}


@dataclass(frozen=True)
class PaperEntryControlSnapshot:
    strategy: str
    status: str
    reason: str
    new_entries_allowed: bool
    existing_position_policy: str
    authority: str
    live_permission: bool
    policy_path: str
    policy_sha256: str
    valid: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "status": self.status,
            "reason": self.reason,
            "new_entries_allowed": self.new_entries_allowed,
            "existing_position_policy": self.existing_position_policy,
            "authority": self.authority,
            "live_permission": self.live_permission,
            "policy_path": self.policy_path,
            "policy_sha256": self.policy_sha256,
            "valid": self.valid,
        }


class PaperEntryControl:
    """Read a Paper-only entry control without touching open-position exits."""

    def __init__(
        self,
        path: Path,
        *,
        refresh_interval_sec: float = 1.0,
    ) -> None:
        self.path = path
        self.refresh_interval_ns = int(
            max(0.0, refresh_interval_sec) * 1_000_000_000
        )
        self._loaded_at_ns = 0
        self._policy_sha256 = ""
        self._controls: dict[str, dict[str, Any]] | None = None
        self._load_error = "ENTRY_CONTROL_NOT_LOADED"

    def snapshot(
        self,
        strategy: str,
        *,
        now_ns: int | None = None,
    ) -> PaperEntryControlSnapshot:
        current_ns = time.monotonic_ns() if now_ns is None else now_ns
        if (
            self._controls is None
            or current_ns - self._loaded_at_ns >= self.refresh_interval_ns
        ):
            self._reload(current_ns)
        if self._controls is None:
            return self._invalid_snapshot(strategy, self._load_error)
        raw = self._controls.get(strategy)
        if not isinstance(raw, dict):
            return self._invalid_snapshot(
                strategy,
                "ENTRY_CONTROL_STRATEGY_MISSING",
            )
        status = str(raw.get("status", "")).upper()
        reason = str(raw.get("reason", "")).strip()
        existing_policy = str(
            raw.get("existing_position_policy", "")
        ).upper()
        authority = str(raw.get("authority", "")).upper()
        live_permission = raw.get("live_permission")
        if (
            status not in ALLOWED_STATUSES
            or not reason
            or existing_policy != "RISK_CONTRACT"
            or authority != "NONE"
            or live_permission is not False
        ):
            return self._invalid_snapshot(
                strategy,
                "ENTRY_CONTROL_RECORD_INVALID",
            )
        return PaperEntryControlSnapshot(
            strategy=strategy,
            status=status,
            reason=reason,
            new_entries_allowed=status not in BLOCKED_STATUSES,
            existing_position_policy=existing_policy,
            authority=authority,
            live_permission=False,
            policy_path=str(self.path),
            policy_sha256=self._policy_sha256,
            valid=True,
        )

    def _reload(self, now_ns: int) -> None:
        self._loaded_at_ns = now_ns
        try:
            payload_bytes = self.path.read_bytes()
            payload = json.loads(payload_bytes)
            if payload.get("schema") != "QR_CRYPTO_ENTRY_CONTROL_V1":
                raise ValueError("unsupported entry-control schema")
            if (
                str(payload.get("authority", "")).upper() != "NONE"
                or payload.get("live_permission") is not False
            ):
                raise ValueError("unsafe entry-control authority")
            controls = payload.get("strategies")
            if not isinstance(controls, dict) or not controls:
                raise ValueError("entry-control strategies missing")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._controls = None
            self._policy_sha256 = ""
            self._load_error = f"ENTRY_CONTROL_UNAVAILABLE_{type(exc).__name__}"
            return
        self._controls = {
            str(key): dict(value)
            for key, value in controls.items()
            if isinstance(value, dict)
        }
        self._policy_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        self._load_error = ""

    def _invalid_snapshot(
        self,
        strategy: str,
        reason: str,
    ) -> PaperEntryControlSnapshot:
        return PaperEntryControlSnapshot(
            strategy=strategy,
            status="FAIL_CLOSED",
            reason=reason,
            new_entries_allowed=False,
            existing_position_policy="RISK_CONTRACT",
            authority="NONE",
            live_permission=False,
            policy_path=str(self.path),
            policy_sha256=self._policy_sha256,
            valid=False,
        )

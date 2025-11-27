"""
analysis.policy_bus
~~~~~~~~~~~~~~~~~~~
Shared in-memory policy exchange between the strategy orchestrator (main)
and worker coroutines. Provides a light-weight publish/subscribe contract
that can optionally persist the latest plan snapshot to disk for debugging.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


_LOCK = threading.Lock()
_POLICY: "PolicySnapshot" | None = None


def _default_pocket_policy() -> Dict[str, Any]:
    return {
        "enabled": True,
        "bias": "neutral",  # long | short | neutral
        "confidence": 0.0,
        "units_cap": None,  # optional hard cap (units)
        "entry_gates": {
            "allow_new": True,
            "require_retest": False,
            "spread_ok": True,
            "drift_ok": True,
        },
        "exit_profile": {
            "reverse_threshold": 70,
            "time_stop": None,
        },
        "be_profile": {
            "enabled": True,
            "trigger_pips": 0.0,
            "cooldown_sec": 0.0,
            "lock_ratio": 0.0,
            "min_lock_pips": 0.0,
        },
        "partial_profile": {
            "thresholds_pips": [],
            "fractions": [],
            "min_units": 0,
        },
        "strategies": [],
        "pending_orders": [],
    }


@dataclass
class PolicySnapshot:
    """
    Minimal schema for sharing orchestrator decisions with worker loops.
    The structure is intentionally permissive so that incremental fields
    can be introduced without breaking consumers (workers should default
    to fallbacks if a field is missing).
    """

    version: int
    generated_ts: float  # epoch seconds (UTC)
    air_score: float = 0.0
    uncertainty: float = 0.0
    event_lock: bool = False
    range_mode: bool = False
    notes: dict[str, Any] = field(default_factory=dict)
    pockets: dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "macro": _default_pocket_policy(),
            "micro": _default_pocket_policy(),
            "scalp": _default_pocket_policy(),
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Protect against dataclass mutability by deep copying via json round-trip
        return json.loads(json.dumps(data))


_DUMP_PATH: Optional[Path] = None
_DUMP_ENABLED = False


def _dump_policy(policy: PolicySnapshot) -> None:
    if not _DUMP_ENABLED or _DUMP_PATH is None:
        return
    try:
        _DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _DUMP_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(policy.to_dict(), ensure_ascii=False, indent=2))
        tmp.replace(_DUMP_PATH)
    except Exception:
        # Dump failures should not break trading logic; ignore silently.
        pass


def configure_dump(path: str | os.PathLike[str] | None) -> None:
    """
    Optionally configure a filesystem location to persist the latest policy.
    Passing None disables the dump feature.
    """
    global _DUMP_PATH, _DUMP_ENABLED
    if path is None:
        _DUMP_PATH = None
        _DUMP_ENABLED = False
        return
    _DUMP_PATH = Path(path)
    _DUMP_ENABLED = True


def publish(policy: PolicySnapshot | Dict[str, Any]) -> None:
    """
    Replace the current snapshot.
    Accepts either a PolicySnapshot instance or a dict matching the schema.
    """
    global _POLICY
    if isinstance(policy, dict):
        policy = PolicySnapshot(
            version=int(policy.get("version", 0)),
            generated_ts=float(policy.get("generated_ts", time.time())),
            air_score=float(policy.get("air_score", 0.0)),
            uncertainty=float(policy.get("uncertainty", 0.0)),
            event_lock=bool(policy.get("event_lock", False)),
            range_mode=bool(policy.get("range_mode", False)),
            notes=dict(policy.get("notes", {})),
            pockets=dict(policy.get("pockets", {})),
        )
    with _LOCK:
        _POLICY = policy
        _dump_policy(policy)


def latest(default: Optional[PolicySnapshot] = None) -> PolicySnapshot | None:
    """
    Return the latest snapshot. A defensive copy is returned so callers
    can mutate without affecting the shared state.
    """
    with _LOCK:
        current = _POLICY
    if current is None:
        return default
    data = current.to_dict()
    return PolicySnapshot(
        version=data["version"],
        generated_ts=data["generated_ts"],
        air_score=data.get("air_score", 0.0),
        uncertainty=data.get("uncertainty", 0.0),
        event_lock=data.get("event_lock", False),
        range_mode=data.get("range_mode", False),
        notes=data.get("notes", {}),
        pockets=data.get("pockets", {}),
    )


# Configure dump path from environment at import time for convenience.
_ENV_PATH = os.getenv("POLICY_BUS_DUMP_PATH")
if _ENV_PATH:
    configure_dump(_ENV_PATH)

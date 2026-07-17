"""AI regime supervision V2 rows (weakness ledger W2/W8/W15).

V1 supervision speaks only pair-level GO/CAUTION/STOP, so the six-hour AI
review cannot express "this is a RANGE day: prefer the range family, stand
the momentum family down".  V2 adds a declared regime and per-family rows.
Family rows reallocate attention among already-approved candidate families;
they never create a signal, change geometry, or grant live permission, and
a family whose declared regime affinity contradicts the declared regime is
forced down to CAUTION at validation time.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

CONTRACT = "QR_AI_REGIME_SUPERVISION_V2"
ACTIONS = frozenset({"GO", "CAUTION", "STOP"})
REGIMES = frozenset({"TREND", "RANGE", "SQUEEZE", "EVENT", "UNCLEAR"})
MAX_TTL = timedelta(hours=6)
AUTHORITY = {
    "creates_signals": False,
    "changes_geometry": False,
    "reuses_selection_evidence": False,
    "order_authority": "NONE",
    "live_permission": False,
    "broker_mutation_allowed": False,
}


class RegimeSupervisionError(ValueError):
    """Raised when a V2 supervision artifact fails validation."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _aware(value: Any, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RegimeSupervisionError(f"{label} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def build_regime_supervision_v2(
    *,
    pair: str,
    declared_regime: str,
    pair_action: str,
    family_rows: Sequence[Mapping[str, Any]],
    observed_at_utc: datetime,
    expires_at_utc: datetime,
    observation_sha256: str,
) -> dict[str, Any]:
    """Seal one pair's V2 supervision bound to one exact observation."""

    pair_name = str(pair).upper()
    if len(pair_name.split("_")) != 2:
        raise RegimeSupervisionError("pair identity is invalid")
    regime = str(declared_regime).upper()
    if regime not in REGIMES:
        raise RegimeSupervisionError("declared regime is invalid")
    action = str(pair_action).upper()
    if action not in ACTIONS:
        raise RegimeSupervisionError("pair action is invalid")
    observed = _aware(observed_at_utc, "observed_at_utc")
    expires = _aware(expires_at_utc, "expires_at_utc")
    if not observed < expires <= observed + MAX_TTL:
        raise RegimeSupervisionError("supervision TTL must be positive and <= 6h")

    sealed_rows: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for row in family_rows:
        family_id = str(row.get("family_id") or "").upper()
        if not family_id:
            raise RegimeSupervisionError("family_id is required")
        if family_id in seen_families:
            raise RegimeSupervisionError(f"duplicate family row: {family_id}")
        seen_families.add(family_id)
        family_action = str(row.get("action") or "").upper()
        if family_action not in ACTIONS:
            raise RegimeSupervisionError(f"family action is invalid: {family_id}")
        affinity = row.get("regime_affinity")
        if not isinstance(affinity, (list, tuple)) or not affinity:
            raise RegimeSupervisionError(
                f"family regime_affinity is required: {family_id}"
            )
        affinity_set = {str(item).upper() for item in affinity}
        if not affinity_set <= REGIMES:
            raise RegimeSupervisionError(
                f"family regime_affinity is invalid: {family_id}"
            )
        # A GO on a family that does not claim the declared regime is a
        # contradiction: demote to CAUTION instead of trusting the row.
        effective = family_action
        demoted = False
        if family_action == "GO" and regime not in affinity_set and regime != "UNCLEAR":
            effective = "CAUTION"
            demoted = True
        sealed_rows.append(
            {
                "family_id": family_id,
                "declared_action": family_action,
                "effective_action": effective,
                "regime_affinity": sorted(affinity_set),
                "demoted_for_regime_mismatch": demoted,
            }
        )
    if not sealed_rows:
        raise RegimeSupervisionError("at least one family row is required")

    body: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 2,
        "pair": pair_name,
        "declared_regime": regime,
        "pair_action": action,
        "family_rows": sealed_rows,
        "observed_at_utc": observed.isoformat(),
        "expires_at_utc": expires.isoformat(),
        "observation_sha256": str(observation_sha256),
        **AUTHORITY,
    }
    return {**body, "supervision_sha256": _canonical_sha(body)}


def effective_family_action(
    supervision: Mapping[str, Any],
    *,
    family_id: str,
    now_utc: datetime,
) -> str:
    """Resolve one family's effective action; fail closed to UNSUPERVISED."""

    if not isinstance(supervision, Mapping):
        raise RegimeSupervisionError("supervision must be an object")
    body = {k: v for k, v in supervision.items() if k != "supervision_sha256"}
    if supervision.get("supervision_sha256") != _canonical_sha(body):
        raise RegimeSupervisionError("supervision digest is invalid")
    if supervision.get("contract") != CONTRACT:
        raise RegimeSupervisionError("supervision contract is invalid")
    for key, value in AUTHORITY.items():
        if supervision.get(key) != value:
            raise RegimeSupervisionError(f"supervision authority is invalid: {key}")
    now = _aware(now_utc, "now_utc")
    if now >= datetime.fromisoformat(str(supervision["expires_at_utc"])):
        return "UNSUPERVISED"
    if supervision.get("pair_action") == "STOP":
        return "STOP"
    wanted = str(family_id).upper()
    for row in supervision.get("family_rows", ()):
        if row.get("family_id") == wanted:
            action = str(row.get("effective_action"))
            if action not in ACTIONS:
                raise RegimeSupervisionError("family effective action is invalid")
            # The pair-level action caps every family row.
            if supervision.get("pair_action") == "CAUTION" and action == "GO":
                return "CAUTION"
            return action
    return "UNSUPERVISED"

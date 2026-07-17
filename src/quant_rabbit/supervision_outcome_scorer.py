"""Supervision outcome scoring (weakness ledger W17).

GO/CAUTION/STOP rows steer capital attention, yet nothing measured whether
they were right.  This module scores expired supervision rows against the
realized stressed pips of the family they governed and seals a monthly
scorecard.  A family whose supervision accuracy falls below the declared
floor is listed for automatic CAUTION so an unreliable supervisor loses
influence instead of silently keeping it.  Scoring reads only realized,
already-sealed outcomes; it changes no authority by itself.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

CONTRACT = "QR_SUPERVISION_OUTCOME_SCORECARD_V1"
ACTIONS = frozenset({"GO", "CAUTION", "STOP"})
MIN_ROWS_PER_FAMILY = 10
ACCURACY_FLOOR = 0.5
CAUTION_TOLERANCE_PIPS = 5.0


class SupervisionScoringError(ValueError):
    """Raised when scoring inputs are malformed."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def row_hit(action: str, realized_stressed_pips: float) -> bool:
    """One row's verdict: did the action match what actually happened?

    GO is right when the governed family realized positive stressed pips,
    STOP is right when it realized non-positive, and CAUTION is right when
    the realized magnitude stayed inside the declared tolerance band.
    """

    name = str(action).upper()
    if name not in ACTIONS:
        raise SupervisionScoringError("action is invalid")
    if isinstance(realized_stressed_pips, bool) or not isinstance(
        realized_stressed_pips, (int, float)
    ):
        raise SupervisionScoringError("realized pips must be a number")
    pips = float(realized_stressed_pips)
    if not math.isfinite(pips):
        raise SupervisionScoringError("realized pips must be finite")
    if name == "GO":
        return pips > 0.0
    if name == "STOP":
        return pips <= 0.0
    return abs(pips) <= CAUTION_TOLERANCE_PIPS


def build_supervision_scorecard(
    rows: Sequence[Mapping[str, Any]],
    *,
    window_label: str,
) -> dict[str, Any]:
    """Seal per-family supervision accuracy over one review window.

    Each row: ``family_id``, ``action``, ``realized_stressed_pips`` — all
    from expired supervision rows and sealed outcome ledgers only.
    """

    if not rows:
        raise SupervisionScoringError("at least one scored row is required")
    per_family: dict[str, dict[str, int]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SupervisionScoringError(f"row {index} must be an object")
        family = str(row.get("family_id") or "").upper()
        if not family:
            raise SupervisionScoringError(f"row {index} family_id is required")
        hit = row_hit(row.get("action"), row.get("realized_stressed_pips"))
        bucket = per_family.setdefault(family, {"scored": 0, "hits": 0})
        bucket["scored"] += 1
        bucket["hits"] += int(hit)

    family_rows = []
    auto_caution: list[str] = []
    for family in sorted(per_family):
        bucket = per_family[family]
        accuracy = bucket["hits"] / bucket["scored"]
        measurable = bucket["scored"] >= MIN_ROWS_PER_FAMILY
        below_floor = measurable and accuracy < ACCURACY_FLOOR
        if below_floor:
            auto_caution.append(family)
        family_rows.append(
            {
                "family_id": family,
                "scored_rows": bucket["scored"],
                "hits": bucket["hits"],
                "accuracy": round(accuracy, 9),
                "measurable": measurable,
                "below_accuracy_floor": below_floor,
            }
        )

    body: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "window_label": str(window_label),
        "accuracy_floor": ACCURACY_FLOOR,
        "min_rows_per_family": MIN_ROWS_PER_FAMILY,
        "caution_tolerance_pips": CAUTION_TOLERANCE_PIPS,
        "family_rows": family_rows,
        "supervision_auto_caution_required": auto_caution,
        "measurement_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    return {**body, "scorecard_sha256": _canonical_sha(body)}

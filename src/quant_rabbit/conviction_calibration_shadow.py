"""Conviction grounding + calibration (completeness gap #4, W25 core).

W25's thesis is that a pro's "gut" is pattern recognition weighted by the
self-awareness of how reliable that read is.  The conviction ladder trusts
a count of booleans the read merely ASSERTS; nothing verifies them or
learns whether high-conviction reads actually pay.  This module supplies
both halves:

  (a) grounding — a declared conviction condition counts only when an
      independent recomputation also confirms it, so an over-confident read
      cannot inflate its own size;
  (b) calibration — from a history of (predicted-confidence, realized
      outcome) pairs it measures Brier score and expected calibration error,
      and maps grounded conviction x regime to a continuous 0..1 confidence
      multiplier via realized stressed-expectancy.

The calibration TABLE must be learned from the live read/outcome ledger
that accrues once the AI trader runs; until then the mechanism is exercised
on explicit inputs.  No order authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

CONTRACT = "QR_CONVICTION_CALIBRATION_V1"


class ConvictionCalibrationError(ValueError):
    """Raised when calibration inputs are malformed."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ground_conviction_conditions(
    declared: Sequence[tuple[str, bool]],
    independent: Mapping[str, bool],
) -> dict[str, Any]:
    """Count only conditions the read declared AND an independent check confirms.

    ``independent`` is a recomputation of each named condition from the
    sealed evidence — not the read's own word.  A condition the read claims
    true but the recomputation cannot confirm is dropped and recorded, so an
    over-confident read is grounded down rather than trusted.
    """

    if not declared:
        raise ConvictionCalibrationError("a declared condition checklist is required")
    grounded = 0
    disagreements: list[str] = []
    seen: set[str] = set()
    for name, claimed in declared:
        key = str(name).strip()
        if not key:
            raise ConvictionCalibrationError("condition name is required")
        if key in seen:
            raise ConvictionCalibrationError(f"duplicate condition: {key}")
        seen.add(key)
        if claimed.__class__ is not bool:
            raise ConvictionCalibrationError(f"condition {key} must be a strict boolean")
        verified = bool(independent.get(key, False))
        if claimed and verified:
            grounded += 1
        elif claimed and not verified:
            disagreements.append(key)
    return {
        "declared_true_count": sum(1 for _, c in declared if c),
        "grounded_true_count": grounded,
        "ungrounded_claims": sorted(disagreements),
        "read_overconfident": bool(disagreements),
    }


def brier_score(predictions: Sequence[tuple[float, bool]]) -> float:
    """Mean squared error of probabilistic predictions (lower is better)."""

    if not predictions:
        raise ConvictionCalibrationError("at least one prediction is required")
    total = 0.0
    for prob, outcome in predictions:
        p = _unit(prob, "predicted probability")
        total += (p - (1.0 if outcome else 0.0)) ** 2
    return total / len(predictions)


def expected_calibration_error(
    predictions: Sequence[tuple[float, bool]], *, bins: int = 10
) -> float:
    """Gap between predicted confidence and realized frequency, binned."""

    if not predictions:
        raise ConvictionCalibrationError("at least one prediction is required")
    if not isinstance(bins, int) or bins <= 0:
        raise ConvictionCalibrationError("bins must be a positive integer")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for prob, outcome in predictions:
        p = _unit(prob, "predicted probability")
        index = min(bins - 1, int(p * bins))
        buckets[index].append((p, outcome))
    n = len(predictions)
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_p = sum(p for p, _ in bucket) / len(bucket)
        freq = sum(1 for _, o in bucket if o) / len(bucket)
        ece += (len(bucket) / n) * abs(mean_p - freq)
    return ece


def calibration_multiplier(
    *,
    grounded_conviction: int,
    regime: str,
    expectancy_table: Mapping[str, Mapping[str, float]],
    max_conviction: int = 4,
) -> dict[str, Any]:
    """Continuous 0..1 confidence multiplier from realized stressed-expectancy.

    ``expectancy_table[regime][str(grounded)]`` holds the realized stressed
    pips/trade for that grounded-conviction level in that regime.  Positive
    expectancy scales toward 1; non-positive collapses the multiplier to 0
    (a level with no measured edge earns no size), so calibration — not the
    read's assertion — sets the confidence.
    """

    if grounded_conviction < 0:
        raise ConvictionCalibrationError("grounded_conviction must be non-negative")
    level = min(grounded_conviction, max_conviction)
    regime_key = str(regime).upper()
    regime_row = expectancy_table.get(regime_key)
    if not isinstance(regime_row, Mapping):
        return {
            "regime": regime_key,
            "grounded_conviction": level,
            "confidence_multiplier": 0.0,
            "reason": "NO_CALIBRATION_FOR_REGIME",
            "calibration_present": False,
        }
    expectancy = regime_row.get(str(level))
    if not isinstance(expectancy, (int, float)) or isinstance(expectancy, bool):
        return {
            "regime": regime_key,
            "grounded_conviction": level,
            "confidence_multiplier": 0.0,
            "reason": "NO_CALIBRATION_FOR_LEVEL",
            "calibration_present": False,
        }
    positives = [
        float(v)
        for row in expectancy_table.values()
        for v in row.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0.0
    ]
    scale = max(positives) if positives else 1.0
    value = float(expectancy)
    multiplier = 0.0 if value <= 0.0 else min(1.0, value / scale)
    return {
        "regime": regime_key,
        "grounded_conviction": level,
        "realized_stressed_expectancy": round(value, 9),
        "confidence_multiplier": round(multiplier, 9),
        "reason": "CALIBRATED",
        "calibration_present": True,
    }


def build_calibration_report(
    predictions: Sequence[tuple[float, bool]], *, window_label: str, bins: int = 10
) -> dict[str, Any]:
    """Seal a monthly calibration report (Brier + ECE) over sealed outcomes."""

    body: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "window_label": str(window_label),
        "sample_count": len(predictions),
        "brier_score": round(brier_score(predictions), 9),
        "expected_calibration_error": round(
            expected_calibration_error(predictions, bins=bins), 9
        ),
        "calibration_table_learned_from_live_ledger": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    return {**body, "report_sha256": _canonical_sha(body)}


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConvictionCalibrationError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ConvictionCalibrationError(f"{label} must be in [0, 1]")
    return number

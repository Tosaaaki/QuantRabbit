"""Fixed 30-month diagnostic score for DOJO historical walk-forward results.

This is an aggregator, not an evidence verifier.  It requires references to
externally verified cell results, but cannot independently re-execute them.
Consequently it never grants promotion or live authority even when every
diagnostic gate passes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date
from typing import Any, Final, Mapping, Sequence


CONTRACT: Final = "QR_DOJO_MONTHLY_MTM_DIAGNOSTIC_V1"
TARGET_MULTIPLE: Final = 3.0
PATHS: Final = ("OHLC", "OLHC")
SCENARIOS: Final = ("BASE", "STRESS")
EXPECTED_PAIRS: Final = (
    "AUD_USD",
    "EUR_USD",
    "GBP_USD",
    "NZD_USD",
    "USD_JPY",
)
EXPECTED_FAMILIES: Final = (
    "compression_break",
    "daily_break_pullback",
    "range_fade_limit",
    "spike_fade",
)
NORMAL_DRAWDOWN_MAX: Final = 0.10
STRESS_DRAWDOWN_MAX: Final = 0.15
PEAK_MARGIN_MAX: Final = 0.45
MARGIN_CLOSEOUT_MAX: Final = 0
LOPO_PROFIT_DROP_MAX: Final = 0.50
EVIDENCE_TIER: Final = "WORN_RETROSPECTIVE_DIAGNOSTIC"
_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


class DojoMonthlyMtmScoringError(ValueError):
    """The fixed historical diagnostic denominator is incomplete or malformed."""


def _months() -> tuple[str, ...]:
    result = []
    year, month = 2024, 1
    while (year, month) <= (2026, 6):
        result.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return tuple(result)


EXPECTED_MONTHS: Final = _months()


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoMonthlyMtmScoringError(f"{field} must be lowercase SHA-256")
    return value


def _number(value: Any, *, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoMonthlyMtmScoringError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise DojoMonthlyMtmScoringError(f"{field} is outside its finite range")
    return result


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DojoMonthlyMtmScoringError(f"{field} must be a non-negative integer")
    return value


def _exact(value: Any, keys: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise DojoMonthlyMtmScoringError(
            f"{field} must contain exactly: {','.join(sorted(keys))}"
        )
    return value


def _cell(value: Any, *, field: str) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "ending_multiple",
            "max_drawdown_fraction",
            "peak_margin_fraction",
            "margin_closeouts",
            "evidence_complete",
            "verified_result_sha256",
            "source_digest_sha256",
            "evidence_tier",
        },
        field=field,
    )
    if row["evidence_complete"] is not True:
        raise DojoMonthlyMtmScoringError(f"{field} evidence is incomplete")
    if row["evidence_tier"] != EVIDENCE_TIER:
        raise DojoMonthlyMtmScoringError(f"{field} evidence tier is not diagnostic")
    return {
        "ending_multiple": _number(
            row["ending_multiple"], field=f"{field}.ending_multiple", minimum=0.0
        ),
        "max_drawdown_fraction": _number(
            row["max_drawdown_fraction"],
            field=f"{field}.max_drawdown_fraction",
            minimum=0.0,
        ),
        "peak_margin_fraction": _number(
            row["peak_margin_fraction"],
            field=f"{field}.peak_margin_fraction",
            minimum=0.0,
        ),
        "margin_closeouts": _integer(
            row["margin_closeouts"], field=f"{field}.margin_closeouts"
        ),
        "evidence_complete": True,
        "verified_result_sha256": _sha(
            row["verified_result_sha256"],
            field=f"{field}.verified_result_sha256",
        ),
        "source_digest_sha256": _sha(
            row["source_digest_sha256"], field=f"{field}.source_digest_sha256"
        ),
        "evidence_tier": EVIDENCE_TIER,
    }


def _cells(value: Any, *, field: str) -> dict[str, dict[str, dict[str, Any]]]:
    outer = _exact(value, set(SCENARIOS), field=field)
    return {
        scenario: {
            path: _cell(
                _exact(outer[scenario], set(PATHS), field=f"{field}.{scenario}")[path],
                field=f"{field}.{scenario}.{path}",
            )
            for path in PATHS
        }
        for scenario in SCENARIOS
    }


def _profit_drop(full: float, remaining: float) -> float:
    profit = full - 1.0
    if profit <= 0.0:
        return 1.0 if remaining < full else 0.0
    return max(0.0, (full - remaining) / profit)


def _lopo(
    value: Any,
    *,
    labels: Sequence[str],
    full_by_path: Mapping[str, float],
    field: str,
) -> dict[str, Any]:
    rows = _exact(value, set(labels), field=field)
    scored = []
    for label in labels:
        paths = _exact(rows[label], set(PATHS), field=f"{field}.{label}")
        multiples = {
            path: _number(paths[path], field=f"{field}.{label}.{path}", minimum=0.0)
            for path in PATHS
        }
        drops = {
            path: _profit_drop(float(full_by_path[path]), multiples[path])
            for path in PATHS
        }
        scored.append(
            {
                "label": label,
                "ending_multiples": multiples,
                "path_aligned_profit_drop_fractions": drops,
                "maximum_profit_drop_fraction": max(drops.values()),
            }
        )
    maximum = max(row["maximum_profit_drop_fraction"] for row in scored)
    return {
        "rows": scored,
        "maximum_profit_drop_fraction": maximum,
        "gate_pass": maximum <= LOPO_PROFIT_DROP_MAX,
    }


def wilson95_lower_bound(hits: int, total: int) -> float:
    if isinstance(hits, bool) or isinstance(total, bool):
        raise DojoMonthlyMtmScoringError("Wilson counts must be integers")
    if not isinstance(hits, int) or not isinstance(total, int) or total <= 0:
        raise DojoMonthlyMtmScoringError("Wilson total must be positive")
    if hits < 0 or hits > total:
        raise DojoMonthlyMtmScoringError("Wilson hits are outside total")
    z = 1.959963984540054
    p = hits / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    spread = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2))
    return max(0.0, (center - spread) / denominator)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    rank = probability * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def score_monthly_mtm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the exact 2024-01..2026-06 retrospective denominator."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise DojoMonthlyMtmScoringError("rows must be a sequence")
    by_month: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _exact(
            raw,
            {"month", "cells", "pair_lopo", "family_lopo"},
            field=f"rows[{index}]",
        )
        month = row["month"]
        if not isinstance(month, str):
            raise DojoMonthlyMtmScoringError("month must be YYYY-MM")
        try:
            canonical = date.fromisoformat(f"{month}-01").strftime("%Y-%m")
        except ValueError as exc:
            raise DojoMonthlyMtmScoringError("month must be YYYY-MM") from exc
        if month != canonical or month in by_month:
            raise DojoMonthlyMtmScoringError("month is non-canonical or duplicated")
        by_month[month] = row
    if set(by_month) != set(EXPECTED_MONTHS):
        raise DojoMonthlyMtmScoringError(
            "rows must contain the exact 2024-01..2026-06 month denominator"
        )

    monthly = []
    for month in EXPECTED_MONTHS:
        row = by_month[month]
        cells = _cells(row["cells"], field=f"months.{month}.cells")
        full_by_path = {
            path: cells["STRESS"][path]["ending_multiple"] for path in PATHS
        }
        pessimistic = min(full_by_path.values())
        base_dd = max(cells["BASE"][path]["max_drawdown_fraction"] for path in PATHS)
        stress_dd = max(
            cells["STRESS"][path]["max_drawdown_fraction"] for path in PATHS
        )
        peak_margin = max(
            cells[scenario][path]["peak_margin_fraction"]
            for scenario in SCENARIOS
            for path in PATHS
        )
        closeouts = sum(
            cells[scenario][path]["margin_closeouts"]
            for scenario in SCENARIOS
            for path in PATHS
        )
        pair_lopo = _lopo(
            row["pair_lopo"],
            labels=EXPECTED_PAIRS,
            full_by_path=full_by_path,
            field=f"months.{month}.pair_lopo",
        )
        family_lopo = _lopo(
            row["family_lopo"],
            labels=EXPECTED_FAMILIES,
            full_by_path=full_by_path,
            field=f"months.{month}.family_lopo",
        )
        gates = {
            "normal_drawdown": base_dd <= NORMAL_DRAWDOWN_MAX,
            "stress_drawdown": stress_dd <= STRESS_DRAWDOWN_MAX,
            "peak_margin": peak_margin <= PEAK_MARGIN_MAX,
            "margin_closeout": closeouts <= MARGIN_CLOSEOUT_MAX,
            "pair_lopo": pair_lopo["gate_pass"],
            "family_lopo": family_lopo["gate_pass"],
        }
        monthly.append(
            {
                "month": month,
                "cells": cells,
                "pessimistic_stress_multiple": pessimistic,
                "reached_3x": pessimistic >= TARGET_MULTIPLE,
                "observed_normal_max_drawdown_fraction": base_dd,
                "observed_stress_max_drawdown_fraction": stress_dd,
                "observed_peak_margin_fraction": peak_margin,
                "observed_margin_closeouts": closeouts,
                "pair_lopo_concentration": pair_lopo,
                "family_lopo_concentration": family_lopo,
                "gates": {**gates, "all_pass": all(gates.values())},
            }
        )

    multiples = [row["pessimistic_stress_multiple"] for row in monthly]
    hits = sum(row["reached_3x"] for row in monthly)
    losing = [
        row["month"] for row in monthly if row["pessimistic_stress_multiple"] < 1.0
    ]
    blockers = []
    if hits != len(EXPECTED_MONTHS):
        blockers.append("NOT_EVERY_MONTH_3X")
    if not all(row["gates"]["all_pass"] for row in monthly):
        blockers.append("MONTHLY_RISK_OR_CONCENTRATION_GATE_FAILED")
    body = {
        "contract": CONTRACT,
        "schema_version": 1,
        "evidence_boundary": "CALLER_REFERENCES_NOT_REEXECUTED_OR_VERIFIED",
        "research_evidence_verified": False,
        "month_denominator": list(EXPECTED_MONTHS),
        "month_count": len(monthly),
        "months": monthly,
        "monthly_pessimistic_stress_multiples": multiples,
        "average_pessimistic_stress_multiple": sum(multiples) / len(multiples),
        "worst_pessimistic_stress_multiple": min(multiples),
        "p05_pessimistic_stress_multiple": _percentile(multiples, 0.05),
        "losing_months": losing,
        "losing_month_rate": len(losing) / len(monthly),
        "three_x_hit_count": hits,
        "three_x_hit_rate": hits / len(monthly),
        "three_x_hit_rate_wilson95_lcb": wilson95_lower_bound(hits, len(monthly)),
        "every_month_3x": hits == len(monthly),
        "every_month_gate_pass": all(row["gates"]["all_pass"] for row in monthly),
        "arithmetic_gate_pass": not blockers,
        "arithmetic_blockers": blockers,
        "promotion_eligible": False,
        "promotion_blockers": [
            "HISTORICAL_WORN_DIAGNOSTIC_HAS_NO_PROMOTION_AUTHORITY",
            *blockers,
        ],
        "live_permission": False,
        "order_authority": "NONE",
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**body, "scorecard_sha256": digest}


__all__ = [
    "CONTRACT",
    "EXPECTED_FAMILIES",
    "EXPECTED_MONTHS",
    "EXPECTED_PAIRS",
    "DojoMonthlyMtmScoringError",
    "score_monthly_mtm",
    "wilson95_lower_bound",
]

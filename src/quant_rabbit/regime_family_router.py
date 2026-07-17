"""Regime x volatility family router (weakness ledger W21 mechanism).

All-weather means one qualified family per regime x volatility cell.  This
router maps a declared market state to the families eligible for capital,
and — crucially — is HONEST about uncovered cells: a cell with no qualified
family returns UNCOVERED_CELL rather than silently routing to a family that
was never validated there.  It reads a pre-declared family catalog (each
family states its regime and volatility affinities and its promotion
status); it never invents coverage from live results.  Routing grants no
order authority; it only says which validated family MAY be considered.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

CONTRACT = "QR_REGIME_FAMILY_ROUTER_V1"
REGIMES = ("TREND", "RANGE", "SQUEEZE", "EVENT")
VOL_STATES = ("LOW", "HIGH")
PROMOTION_STATES = frozenset(
    {"UNPROVEN", "TRAIN_LOCKED", "VALIDATION_REPLICATED", "FUTURE_PROVEN"}
)


class RegimeFamilyRouterError(ValueError):
    """Raised when the family catalog or a routing query is malformed."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_family(family: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(family, Mapping):
        raise RegimeFamilyRouterError("family must be an object")
    family_id = str(family.get("family_id") or "").strip().upper()
    if not family_id:
        raise RegimeFamilyRouterError("family_id is required")
    regimes = {str(item).upper() for item in family.get("regime_affinity") or ()}
    if not regimes or not regimes <= set(REGIMES):
        raise RegimeFamilyRouterError(f"family regime_affinity is invalid: {family_id}")
    vols = {str(item).upper() for item in family.get("vol_affinity") or ()}
    if not vols or not vols <= set(VOL_STATES):
        raise RegimeFamilyRouterError(f"family vol_affinity is invalid: {family_id}")
    promotion = str(family.get("promotion_state") or "")
    if promotion not in PROMOTION_STATES:
        raise RegimeFamilyRouterError(f"family promotion_state is invalid: {family_id}")
    return {
        "family_id": family_id,
        "regime_affinity": sorted(regimes),
        "vol_affinity": sorted(vols),
        "promotion_state": promotion,
    }


def build_family_catalog(families: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Seal a pre-declared family catalog and map the full regime x vol grid."""

    if not families:
        raise RegimeFamilyRouterError("at least one family is required")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for family in families:
        row = _validate_family(family)
        if row["family_id"] in seen:
            raise RegimeFamilyRouterError(f"duplicate family: {row['family_id']}")
        seen.add(row["family_id"])
        validated.append(row)

    grid: list[dict[str, Any]] = []
    uncovered = 0
    for regime in REGIMES:
        for vol in VOL_STATES:
            eligible = [
                row["family_id"]
                for row in validated
                if regime in row["regime_affinity"] and vol in row["vol_affinity"]
            ]
            proven = [
                row["family_id"]
                for row in validated
                if row["family_id"] in eligible
                and row["promotion_state"]
                in {"VALIDATION_REPLICATED", "FUTURE_PROVEN"}
            ]
            covered = bool(eligible)
            if not covered:
                uncovered += 1
            grid.append(
                {
                    "regime": regime,
                    "vol_state": vol,
                    "eligible_families": sorted(eligible),
                    "proven_families": sorted(proven),
                    "covered": covered,
                    "proven_covered": bool(proven),
                }
            )

    body: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "families": validated,
        "grid": grid,
        "cell_count": len(grid),
        "uncovered_cell_count": uncovered,
        "all_weather_grid_covered": uncovered == 0,
        "all_weather_grid_proven": all(cell["proven_covered"] for cell in grid),
        "order_authority": "NONE",
        "live_permission": False,
    }
    return {**body, "catalog_sha256": _canonical_sha(body)}


def route_families(
    catalog: Mapping[str, Any], *, declared_regime: str, vol_state: str
) -> dict[str, Any]:
    """Return the families that MAY be considered for one declared cell."""

    body = {key: value for key, value in catalog.items() if key != "catalog_sha256"}
    if catalog.get("catalog_sha256") != _canonical_sha(body):
        raise RegimeFamilyRouterError("family catalog digest is invalid")
    regime = str(declared_regime).upper()
    vol = str(vol_state).upper()
    if regime == "UNCLEAR" or regime not in REGIMES or vol not in VOL_STATES:
        # An unclear or unknown state routes to nothing rather than guessing.
        return {
            "regime": regime,
            "vol_state": vol,
            "eligible_families": [],
            "proven_families": [],
            "routing_status": "UNCLEAR_STATE_NO_ROUTE",
            "order_authority": "NONE",
        }
    for cell in catalog.get("grid", ()):
        if cell["regime"] == regime and cell["vol_state"] == vol:
            return {
                "regime": regime,
                "vol_state": vol,
                "eligible_families": list(cell["eligible_families"]),
                "proven_families": list(cell["proven_families"]),
                "routing_status": (
                    "PROVEN_FAMILY_AVAILABLE"
                    if cell["proven_covered"]
                    else "ELIGIBLE_UNPROVEN_ONLY"
                    if cell["covered"]
                    else "UNCOVERED_CELL"
                ),
                "order_authority": "NONE",
            }
    return {
        "regime": regime,
        "vol_state": vol,
        "eligible_families": [],
        "proven_families": [],
        "routing_status": "UNCOVERED_CELL",
        "order_authority": "NONE",
    }

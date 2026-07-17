from __future__ import annotations

import pytest

from quant_rabbit.regime_family_router import (
    RegimeFamilyRouterError,
    build_family_catalog,
    route_families,
)


def _catalog():
    # The current honest state: a trend/high-vol survivor (validation
    # replicated) and a design-stage range family (unproven).  Low-vol trend
    # and the range x high-vol cell stay uncovered.
    return build_family_catalog(
        [
            {
                "family_id": "S5_SURVIVOR",
                "regime_affinity": ["TREND"],
                "vol_affinity": ["HIGH"],
                "promotion_state": "VALIDATION_REPLICATED",
            },
            {
                "family_id": "RANGE_RAIL",
                "regime_affinity": ["RANGE", "SQUEEZE"],
                "vol_affinity": ["LOW"],
                "promotion_state": "UNPROVEN",
            },
        ]
    )


def test_grid_is_honest_about_uncovered_cells() -> None:
    catalog = _catalog()

    assert catalog["cell_count"] == 8  # 4 regimes x 2 vol states
    assert catalog["all_weather_grid_covered"] is False
    assert catalog["all_weather_grid_proven"] is False
    # 8 cells, only TREND/HIGH and RANGE/LOW and SQUEEZE/LOW covered.
    assert catalog["uncovered_cell_count"] == 5


def test_routing_distinguishes_proven_unproven_and_uncovered() -> None:
    catalog = _catalog()

    proven = route_families(catalog, declared_regime="TREND", vol_state="HIGH")
    assert proven["routing_status"] == "PROVEN_FAMILY_AVAILABLE"
    assert proven["proven_families"] == ["S5_SURVIVOR"]

    unproven = route_families(catalog, declared_regime="RANGE", vol_state="LOW")
    assert unproven["routing_status"] == "ELIGIBLE_UNPROVEN_ONLY"
    assert unproven["eligible_families"] == ["RANGE_RAIL"]
    assert unproven["proven_families"] == []

    uncovered = route_families(catalog, declared_regime="RANGE", vol_state="HIGH")
    assert uncovered["routing_status"] == "UNCOVERED_CELL"
    assert uncovered["eligible_families"] == []

    unclear = route_families(catalog, declared_regime="UNCLEAR", vol_state="HIGH")
    assert unclear["routing_status"] == "UNCLEAR_STATE_NO_ROUTE"


def test_catalog_fail_closed_and_tamper_evident() -> None:
    with pytest.raises(RegimeFamilyRouterError, match="promotion_state"):
        build_family_catalog(
            [
                {
                    "family_id": "X",
                    "regime_affinity": ["TREND"],
                    "vol_affinity": ["HIGH"],
                    "promotion_state": "MAYBE",
                }
            ]
        )
    with pytest.raises(RegimeFamilyRouterError, match="duplicate"):
        build_family_catalog(
            [
                {"family_id": "X", "regime_affinity": ["TREND"], "vol_affinity": ["HIGH"], "promotion_state": "UNPROVEN"},
                {"family_id": "X", "regime_affinity": ["RANGE"], "vol_affinity": ["LOW"], "promotion_state": "UNPROVEN"},
            ]
        )

    catalog = _catalog()
    tampered = dict(catalog)
    tampered["uncovered_cell_count"] = 0
    with pytest.raises(RegimeFamilyRouterError, match="digest"):
        route_families(tampered, declared_regime="TREND", vol_state="HIGH")

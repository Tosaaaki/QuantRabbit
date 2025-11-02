from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.order_sizing import apply_min_units_floor


def test_floor_not_exceeding_risk():
    adjusted, info = apply_min_units_floor(
        base_units=500,
        min_units_cfg=1000,
        units_by_risk=800,
        units_by_margin=2000,
    )
    assert adjusted == 800
    assert info is not None
    assert info["applied"] is True
    assert info["effective_floor"] == 800


def test_floor_respects_margin_cap():
    adjusted, info = apply_min_units_floor(
        base_units=200,
        min_units_cfg=600,
        units_by_risk=1000,
        units_by_margin=450,
    )
    assert adjusted == 450
    assert info is not None
    assert info["applied"] is True
    assert info["adjusted_units"] == 450
    assert info["clipped_by"] == "margin"


def test_floor_skipped_when_base_already_high_enough():
    adjusted, info = apply_min_units_floor(
        base_units=1200,
        min_units_cfg=600,
        units_by_risk=1500,
        units_by_margin=2000,
    )
    assert adjusted == 1200
    assert info is None


def test_floor_never_applies_when_no_risk_capacity():
    adjusted, info = apply_min_units_floor(
        base_units=0,
        min_units_cfg=500,
        units_by_risk=0,
        units_by_margin=2000,
    )
    assert adjusted == 0
    assert info is None

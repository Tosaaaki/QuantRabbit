from __future__ import annotations

import pytest

from quant_rabbit.conviction_calibration_shadow import (
    ConvictionCalibrationError,
    brier_score,
    build_calibration_report,
    calibration_multiplier,
    expected_calibration_error,
    ground_conviction_conditions,
    _canonical_sha,
)


def test_grounding_drops_unverified_claims() -> None:
    declared = [
        ("REGIME_ALIGNED", True),
        ("SESSION_FAVORABLE", True),
        ("NO_EVENT_WINDOW", True),
    ]
    independent = {"REGIME_ALIGNED": True, "SESSION_FAVORABLE": False, "NO_EVENT_WINDOW": True}

    result = ground_conviction_conditions(declared, independent)

    assert result["declared_true_count"] == 3
    assert result["grounded_true_count"] == 2  # SESSION not independently confirmed
    assert result["ungrounded_claims"] == ["SESSION_FAVORABLE"]
    assert result["read_overconfident"] is True


def test_brier_and_ece_reward_calibration() -> None:
    # Perfectly calibrated & confident predictions -> low Brier, low ECE.
    good = [(0.9, True)] * 9 + [(0.9, False)] * 1
    # Overconfident wrong predictions -> high Brier.
    bad = [(0.9, False)] * 9 + [(0.9, True)] * 1
    assert brier_score(good) < brier_score(bad)
    assert expected_calibration_error(good) < 0.15
    assert expected_calibration_error(bad) > 0.5

    with pytest.raises(ConvictionCalibrationError, match="in .0, 1."):
        brier_score([(1.5, True)])


def test_multiplier_is_set_by_realized_expectancy_not_the_read() -> None:
    table = {
        "TREND": {"0": 1.0, "2": 5.0, "4": 10.0},
        "RANGE": {"0": -2.0, "2": -1.0, "4": 0.5},
    }
    # Highest realized expectancy in the table (10.0) maps to multiplier 1.0.
    strong = calibration_multiplier(grounded_conviction=4, regime="TREND", expectancy_table=table)
    assert strong["confidence_multiplier"] == pytest.approx(1.0)

    # A conviction level whose realized expectancy is negative earns 0 size,
    # regardless of how confident the read claimed to be.
    losing = calibration_multiplier(grounded_conviction=2, regime="RANGE", expectancy_table=table)
    assert losing["confidence_multiplier"] == 0.0

    missing = calibration_multiplier(grounded_conviction=4, regime="EVENT", expectancy_table=table)
    assert missing["calibration_present"] is False
    assert missing["confidence_multiplier"] == 0.0


def test_calibration_report_seals() -> None:
    report = build_calibration_report([(0.8, True), (0.2, False)], window_label="2026-07")
    body = {k: v for k, v in report.items() if k != "report_sha256"}
    assert report["report_sha256"] == _canonical_sha(body)
    assert report["sample_count"] == 2
    assert report["order_authority"] == "NONE"

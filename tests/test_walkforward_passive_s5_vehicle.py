from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.audit_walkforward_passive_s5_vehicle import (
    _candidate_rank,
    _filter_predictions,
    _entry_configurations,
    _positive_expectancy_candidate,
    _prediction_windows,
    _vehicle_geometries,
)


def test_truth_window_covers_full_vehicle_not_short_direction_horizon() -> None:
    entry = datetime(2026, 7, 1, tzinfo=timezone.utc)
    windows = _prediction_windows(
        [
            {
                "pair": "EUR_USD",
                "entry_timestamp_utc": entry.isoformat(),
                "future_timestamp_utc": (entry + timedelta(minutes=60)).isoformat(),
            }
        ],
        vehicle_horizon_min=1440.0,
        exit_grace_seconds=5.0,
    )
    assert windows["EUR_USD"] == [
        (entry, entry + timedelta(days=1, seconds=5))
    ]


def test_pair_filter_is_explicit_and_preserves_matching_rows() -> None:
    rows = [
        {"pair": "EUR_USD", "value": 1},
        {"pair": "AUD_JPY", "value": 2},
    ]
    assert _filter_predictions(rows, pairs={"AUD_JPY"}) == [
        {"pair": "AUD_JPY", "value": 2}
    ]


def test_vehicle_grid_never_accepts_loss_larger_than_take_profit() -> None:
    geometries = _vehicle_geometries()
    assert geometries
    assert all(reward > risk for reward, risk in geometries)


def test_entry_grid_compares_market_and_passive_limit() -> None:
    configurations = _entry_configurations()
    assert any(vehicle == "MARKET" for vehicle, _, _ in configurations)
    assert any(vehicle == "PASSIVE_LIMIT" for vehicle, _, _ in configurations)


def test_entry_grid_preserves_total_forecast_horizon() -> None:
    configurations = _entry_configurations(vehicle_horizon_min=240.0)
    assert configurations
    for vehicle, entry_ttl, max_hold in configurations:
        if vehicle == "MARKET":
            assert entry_ttl == 0.0
            assert max_hold == 240.0
        else:
            assert entry_ttl + max_hold == 240.0


def test_candidate_ranking_prefers_positive_expectancy_after_minimum_evidence() -> None:
    positive = {
        "validation_metrics": {
            "mean_conservative_pips": 1.0,
            "one_sided_95_mean_lower_pips": -2.0,
            "fills": 100,
        }
    }
    certain_loss = {
        "validation_metrics": {
            "mean_conservative_pips": -0.5,
            "one_sided_95_mean_lower_pips": -0.6,
            "fills": 100,
        }
    }
    assert _candidate_rank(positive) > _candidate_rank(certain_loss)


def test_positive_expectancy_candidate_does_not_require_positive_95pct_lower() -> None:
    validation = {
        "fills": 100,
        "mean_conservative_pips": 1.2,
        "conservative_profit_factor": 1.3,
    }
    holdout = {
        "fills": 100,
        "mean_conservative_pips": 0.4,
        "conservative_profit_factor": 1.1,
        "positive_day_rate": 0.6,
        "one_sided_95_mean_lower_pips": -0.8,
    }
    assert _positive_expectancy_candidate(
        validation,
        holdout,
        minimum_validation_fills=40,
        minimum_holdout_fills=40,
    )

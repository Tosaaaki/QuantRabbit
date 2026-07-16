from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.technical_forecast_evaluation import (
    choose_validation_threshold,
    directional_metrics,
    select_non_overlapping_predictions,
)


def _row(
    timestamp: datetime,
    *,
    pair: str = "EUR_USD",
    prediction: float = 2.0,
    long_pips: float = 3.0,
    short_pips: float = -4.0,
) -> dict[str, object]:
    return {
        "timestamp_utc": timestamp.isoformat(),
        "pair": pair,
        "predicted_pips": prediction,
        "long_pips": long_pips,
        "short_pips": short_pips,
    }


class TechnicalForecastEvaluationTest(unittest.TestCase):
    def test_selection_is_pair_local_and_non_overlapping(self) -> None:
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        selected = select_non_overlapping_predictions(
            [
                _row(start),
                _row(start + timedelta(minutes=10)),
                _row(start + timedelta(minutes=10), pair="USD_JPY"),
                _row(start + timedelta(minutes=60)),
            ],
            horizon_min=60,
            minimum_absolute_prediction_pips=1,
        )
        self.assertEqual(
            [(row["pair"], row["timestamp_utc"]) for row in selected],
            [
                ("EUR_USD", start.isoformat()),
                ("USD_JPY", (start + timedelta(minutes=10)).isoformat()),
                ("EUR_USD", (start + timedelta(minutes=60)).isoformat()),
            ],
        )

    def test_direction_uses_executable_side_return(self) -> None:
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        selected = select_non_overlapping_predictions(
            [
                _row(start, prediction=2, long_pips=3, short_pips=-5),
                _row(
                    start + timedelta(hours=1),
                    prediction=-2,
                    long_pips=-4,
                    short_pips=6,
                ),
            ],
            horizon_min=60,
            minimum_absolute_prediction_pips=0,
        )
        self.assertEqual([row["executed_pips"] for row in selected], [3.0, 6.0])
        self.assertEqual(directional_metrics(selected)["mean_pips"], 4.5)

    def test_threshold_is_chosen_only_from_supplied_validation_rows(self) -> None:
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        rows = [
            _row(start + timedelta(days=index), prediction=prediction, long_pips=result)
            for index, (prediction, result) in enumerate(
                [(1, -2), (2, -1), (3, 4), (4, 5)]
            )
        ]
        result = choose_validation_threshold(
            rows,
            horizon_min=60,
            thresholds_pips=[0, 3],
            minimum_trades=2,
            minimum_active_days=2,
        )
        self.assertEqual(result["selected"]["threshold_pips"], 3.0)
        self.assertEqual(result["selected"]["metrics"]["trades"], 2)

    def test_metrics_report_one_roundtrip_cost_identity(self) -> None:
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        metrics = directional_metrics(
            [
                {
                    **_row(start, long_pips=2.0),
                    "executed_pips": 2.0,
                    "gross_directional_pips": 3.5,
                    "roundtrip_spread_cost_pips": 1.5,
                },
                {
                    **_row(start + timedelta(hours=1), short_pips=1.0),
                    "executed_pips": 1.0,
                    "gross_directional_pips": 2.0,
                    "roundtrip_spread_cost_pips": 1.0,
                },
            ]
        )
        self.assertEqual(metrics["gross_directional_mean_pips"], 2.75)
        self.assertEqual(metrics["roundtrip_spread_cost_mean_pips"], 1.25)
        self.assertEqual(metrics["mean_pips"], 1.5)
        self.assertEqual(metrics["execution_identity_max_abs_pips"], 0.0)


if __name__ == "__main__":
    unittest.main()

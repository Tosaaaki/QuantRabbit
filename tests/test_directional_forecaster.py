"""Unit tests for strategy/directional_forecaster.py."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from quant_rabbit.strategy.directional_forecaster import synthesize_forecast


@dataclass
class _Sig:
    direction: str
    bonus_magnitude: float
    confidence: float
    rationale: str = ""


class ForecastGeometryTest(unittest.TestCase):
    def test_down_forecast_uses_only_below_current_targets(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M15",
                    "indicators": {"pip_size": 0.0001},
                    "structure": {
                        "liquidity": [
                            {"side": "EQ_LOW", "price": 1.1015, "indices": [1, 2, 3, 4]},
                            {"side": "EQ_HIGH", "price": 1.1020, "indices": [1, 2, 3, 4]},
                        ],
                        "swings": [
                            {"side": "HIGH", "price": 1.1020},
                            {"side": "LOW", "price": 1.1015},
                        ],
                    },
                },
                {
                    "granularity": "H1",
                    "indicators": {"pip_size": 0.0001},
                    "structure": {
                        "liquidity": [
                            {"side": "EQ_LOW", "price": 1.0980, "indices": [1, 2, 3, 4]},
                        ],
                        "swings": [
                            {"side": "LOW", "price": 1.0980},
                        ],
                    },
                },
            ]
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 30.0, 1.0, "breakdown")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertIsNotNone(forecast.target_price)
        self.assertLess(forecast.target_price, 1.1000)
        self.assertIsNotNone(forecast.invalidation_price)
        self.assertGreater(forecast.invalidation_price, 1.1000)


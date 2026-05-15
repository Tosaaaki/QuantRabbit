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
        self.assertIn("breakdown", " ".join(forecast.drivers_for))
        self.assertEqual(forecast.current_price, 1.1)
        self.assertGreater(forecast.down_score, 0.0)

    def test_strong_htf_downtrend_prevents_micro_up_signal_from_owning_forecast(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "SHORT_LEAN",
                "score_gap": -0.75,
                "tf_agreement_score": 1.0,
                "dominant_regime": "TREND_DOWN",
            },
            "views": [
                {
                    "granularity": "H1",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 48.0},
                    "structure": {"structure_events": []},
                },
                {
                    "granularity": "H4",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 35.0},
                    "structure": {"structure_events": []},
                },
                {
                    "granularity": "M1",
                    "regime": "RANGE",
                    "indicators": {"pip_size": 0.0001, "adx_14": 12.0},
                    "structure": {
                        "structure_events": [
                            {"kind": "BOS_UP", "close_confirmed": True, "broken_pivot_price": 1.1001}
                        ]
                    },
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[_Sig("UP", 40.0, 1.0, "M1-only bottom bounce")],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertNotEqual(forecast.direction, "UP")
        self.assertGreater(forecast.down_score, forecast.up_score)
        self.assertIn("SHORT_LEAN", " ".join(forecast.drivers_for))

    def test_countertrend_confirmation_ignores_stale_m15_h1_structure(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "SHORT_LEAN",
                "score_gap": -0.75,
                "tf_agreement_score": 1.0,
                "dominant_regime": "TREND_DOWN",
            },
            "views": [
                {
                    "granularity": "M15",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 38.0},
                    "structure": {
                        "structure_events": [
                            {"kind": "CHOCH_UP", "close_confirmed": True, "index": 100},
                            {"kind": "BOS_DOWN", "close_confirmed": True, "index": 194},
                        ]
                    },
                },
                {
                    "granularity": "H1",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 52.0},
                    "structure": {
                        "structure_events": [
                            {"kind": "BOS_UP", "close_confirmed": True, "index": 109},
                            {"kind": "BOS_DOWN", "close_confirmed": True, "index": 153},
                        ]
                    },
                },
                {
                    "granularity": "H4",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 35.0},
                    "structure": {"structure_events": []},
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[_Sig("UP", 90.0, 1.0, "stale reversal should not dominate")],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertNotEqual(forecast.direction, "UP")
        self.assertLess(forecast.up_score, 90.0)

    def test_countertrend_confirmation_accepts_latest_h1_reversal(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "SHORT_LEAN",
                "score_gap": -0.75,
                "tf_agreement_score": 1.0,
                "dominant_regime": "TREND_DOWN",
            },
            "views": [
                {
                    "granularity": "H1",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 52.0},
                    "structure": {
                        "structure_events": [
                            {"kind": "BOS_DOWN", "close_confirmed": True, "index": 153},
                            {"kind": "CHOCH_UP", "close_confirmed": True, "index": 168},
                        ]
                    },
                },
                {
                    "granularity": "H4",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 35.0},
                    "structure": {"structure_events": []},
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[_Sig("UP", 90.0, 1.0, "fresh H1 reversal")],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertIn("countertrend UP allowed", forecast.rationale_summary)

    def test_score_momentum_and_market_location_allow_early_turn_forecast(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "SHORT_LEAN",
                "score_gap": -0.30,
                "tf_agreement_score": 1.0,
                "dominant_regime": "TREND_DOWN",
                "price_percentile_24h": 0.04,
                "price_percentile_7d": 0.06,
                "score_momentum": {
                    "direction": "UP",
                    "elapsed_min": 60.0,
                    "long_score_delta": 0.24,
                    "short_score_delta": -0.26,
                    "score_gap_delta": 0.40,
                    "score_gap_slope_per_hour": 0.40,
                    "previous_score_gap": -0.70,
                    "current_score_gap": -0.30,
                },
            },
            "views": [
                {
                    "granularity": "H1",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 32.0},
                    "structure": {"structure_events": []},
                },
                {
                    "granularity": "H4",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001, "adx_14": 31.0},
                    "structure": {"structure_events": []},
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[_Sig("UP", 20.0, 1.0, "fresh lower sweep")],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertGreater(forecast.up_score, forecast.down_score)
        self.assertIn("score momentum", " ".join(forecast.drivers_for))
        self.assertIn("market location", " ".join(forecast.drivers_for))
        self.assertIn("score momentum gap", forecast.rationale_summary)

    def test_technical_family_scores_feed_forecast_before_flat_score_gap(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "TIED",
                "score_gap": 0.0,
                "dominant_regime": "TREND_UP",
            },
            "views": [
                {
                    "granularity": "M15",
                    "regime": "TREND_UP",
                    "indicators": {"pip_size": 0.0001},
                    "family_scores": {
                        "trend_score": 0.82,
                        "mean_rev_score": 0.05,
                        "breakout_score": 0.10,
                        "disagreement": 0.0,
                    },
                },
                {
                    "granularity": "H1",
                    "regime": "TREND_UP",
                    "indicators": {"pip_size": 0.0001},
                    "family_scores": {
                        "trend_score": 0.74,
                        "mean_rev_score": -0.10,
                        "breakout_score": 0.05,
                        "disagreement": 0.0,
                    },
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertGreater(forecast.up_score, 0.0)
        self.assertIn("technical trend family", " ".join(forecast.drivers_for))

    def test_forecast_calibration_prefers_direction_specific_history(self) -> None:
        hit_rates = {
            "directional_forecast": {
                "EUR_USD:TREND": {"hit_rate": 1.0, "samples": 100},
            },
            "directional_forecast_up": {
                "EUR_USD:TREND": {"hit_rate": 0.0, "samples": 100},
            },
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart={"views": [{"granularity": "M15", "indicators": {"pip_size": 0.0001}}]},
            current_price=1.1000,
            pattern_signals=[_Sig("UP", 60.0, 1.0, "up setup")],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
            hit_rates=hit_rates,
            regime="TREND",
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertLess(forecast.calibration_multiplier, 0.5)
        self.assertLess(forecast.confidence, 0.5)

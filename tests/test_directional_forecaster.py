"""Unit tests for strategy/directional_forecaster.py."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from quant_rabbit.strategy.directional_forecaster import (
    FORECAST_D_ANCHOR_HORIZON_MIN,
    FORECAST_EXECUTION_HORIZON_MIN,
    FORECAST_OPERATING_SWING_HORIZON_MIN,
    synthesize_forecast,
)


@dataclass
class _Sig:
    direction: str
    bonus_magnitude: float
    confidence: float
    rationale: str = ""


@dataclass
class _NamedSig:
    name: str
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

    def test_forecast_invalidation_ignores_levels_inside_operating_noise(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M5",
                    "indicators": {
                        "pip_size": 0.0001,
                        "atr_pips": 2.0,
                        "donchian_high": 1.10004,
                        "donchian_low": 1.09965,
                    },
                    "structure": {
                        "swings": [
                            {"side": "HIGH", "price": 1.10004},
                            {"side": "HIGH", "price": 1.10040},
                            {"side": "LOW", "price": 1.09965},
                        ],
                    },
                }
            ]
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 60.0, 1.0, "breakdown")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertEqual(forecast.invalidation_price, 1.1004)
        self.assertLess(forecast.target_price, 1.1000)

    def test_incomplete_noise_cleared_geometry_reduces_forecast_confidence(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M5",
                    "indicators": {
                        "pip_size": 0.0001,
                        "atr_pips": 2.0,
                        "donchian_high": 1.10040,
                        "donchian_low": 1.09992,
                    },
                    "structure": {
                        "swings": [
                            {"side": "HIGH", "price": 1.10040},
                            {"side": "LOW", "price": 1.09992},
                        ],
                    },
                }
            ]
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 60.0, 1.0, "breakdown")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertIsNone(forecast.target_price)
        self.assertEqual(forecast.invalidation_price, 1.1004)
        self.assertLess(forecast.confidence, forecast.raw_confidence)
        self.assertIn("robust forecast geometry missing target", forecast.rationale_summary)

    def test_forecast_geometry_clears_current_spread_noise_floor(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M5",
                    "indicators": {
                        "pip_size": 0.0001,
                        "atr_pips": 2.0,
                        "donchian_high": 1.10070,
                        "donchian_low": 1.09920,
                    },
                    "structure": {
                        "swings": [
                            {"side": "HIGH", "price": 1.10070},
                            {"side": "LOW", "price": 1.09920},
                        ],
                    },
                }
            ]
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 60.0, 1.0, "breakdown")],
            correlation_signals=[],
            paths=[],
            spread_pips=2.0,
        )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertIsNone(forecast.target_price)
        self.assertIsNone(forecast.invalidation_price)
        self.assertLess(forecast.confidence, forecast.raw_confidence)
        self.assertIn("robust forecast geometry missing target/invalidation", forecast.rationale_summary)

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

    def test_stable_range_phase_forecasts_range(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "TIED",
                "score_gap": 0.0,
                "dominant_regime": "UNCLEAR",
            },
            "views": [
                {
                    "granularity": "M15",
                    "regime": "RANGE",
                    "regime_reading": {"state": "RANGE", "confidence": 0.9, "atr_percentile": 30.0},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 14.0,
                        "choppiness_14": 66.0,
                        "close": 1.1050,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                    },
                },
                {
                    "granularity": "H1",
                    "regime": "RANGE",
                    "regime_reading": {"state": "RANGE", "confidence": 0.9, "atr_percentile": 35.0},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 16.0,
                        "choppiness_14": 64.0,
                        "close": 1.1050,
                        "donchian_low": 1.0990,
                        "donchian_high": 1.1110,
                    },
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1050,
            pattern_signals=[],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "RANGE")
        self.assertGreater(forecast.range_score, forecast.up_score)
        self.assertIn("inside stable range", " ".join(forecast.drivers_for))

    def test_contested_direction_inside_range_forecasts_range(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "TIED",
                "score_gap": 0.0,
                "dominant_regime": "RANGE",
            },
            "views": [
                {
                    "granularity": "M5",
                    "regime": "UNCLEAR",
                    "regime_reading": {"state": "TREND_WEAK", "confidence": 0.6},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 18.0,
                        "choppiness_14": 56.0,
                        "close": 1.1050,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                    },
                },
                {
                    "granularity": "M15",
                    "regime": "UNCLEAR",
                    "regime_reading": {"state": "TREND_WEAK", "confidence": 0.6},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 19.0,
                        "choppiness_14": 55.0,
                        "close": 1.1050,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                    },
                },
                {
                    "granularity": "H1",
                    "regime": "UNCLEAR",
                    "regime_reading": {"state": "TRANSITION", "confidence": 0.6},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 20.0,
                        "choppiness_14": 50.0,
                        "close": 1.1050,
                        "donchian_low": 1.0990,
                        "donchian_high": 1.1110,
                    },
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1050,
            pattern_signals=[_Sig("UP", 55.0, 1.0, "up evidence")],
            projection_signals=[_Sig("DOWN", 53.0, 1.0, "down evidence")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "RANGE")
        self.assertGreaterEqual(forecast.confidence, 0.55)
        self.assertIn("contested direction inside RANGE_FORMING", forecast.rationale_summary)
        self.assertIn("range forming", " ".join(forecast.drivers_for))

    def test_contested_direction_without_range_evidence_stays_unclear(self) -> None:
        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart={"views": [{"granularity": "M15", "indicators": {"pip_size": 0.0001}}]},
            current_price=1.1050,
            pattern_signals=[_Sig("UP", 55.0, 1.0, "up evidence")],
            projection_signals=[_Sig("DOWN", 53.0, 1.0, "down evidence")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertIn("contested", forecast.rationale_summary)

    def test_lower_half_range_location_contests_downside_chase_forecast(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "SHORT_LEAN",
                "score_gap": -0.4,
                "dominant_regime": "RANGE",
            },
            "views": [
                {
                    "granularity": "M5",
                    "regime": "RANGE",
                    "regime_reading": {"state": "RANGE", "confidence": 0.9},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 14.0,
                        "choppiness_14": 66.0,
                        "close": 1.1008,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                    },
                },
                {
                    "granularity": "M15",
                    "regime": "RANGE",
                    "regime_reading": {"state": "RANGE", "confidence": 0.9},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 16.0,
                        "choppiness_14": 64.0,
                        "close": 1.1010,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1120,
                    },
                },
                {
                    "granularity": "H1",
                    "regime": "RANGE",
                    "regime_reading": {"state": "RANGE", "confidence": 0.9},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 17.0,
                        "choppiness_14": 63.0,
                        "close": 1.1010,
                        "donchian_low": 1.0990,
                        "donchian_high": 1.1130,
                    },
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1010,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 35.0, 1.0, "downside pressure")],
            correlation_signals=[],
            paths=[],
        )

        self.assertNotEqual(forecast.direction, "DOWN")
        self.assertGreater(forecast.up_score, 0.0)
        self.assertIn("contested", forecast.rationale_summary)

    def test_breakout_pending_range_phase_blocks_range_rotation(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "TIED",
                "score_gap": 0.0,
                "dominant_regime": "RANGE",
                "range_24h_sigma_multiple": 0.7,
            },
            "views": [
                {
                    "granularity": "M15",
                    "regime": "RANGE",
                    "regime_reading": {"state": "BREAKOUT_PENDING", "confidence": 0.8, "atr_percentile": 12.0},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 22.0,
                        "choppiness_14": 50.0,
                        "bb_squeeze": 1,
                        "bb_width_percentile_100": 0.12,
                        "close": 1.1050,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                    },
                },
                {
                    "granularity": "H1",
                    "regime": "RANGE",
                    "regime_reading": {"state": "BREAKOUT_PENDING", "confidence": 0.8, "atr_percentile": 14.0},
                    "indicators": {
                        "pip_size": 0.0001,
                        "adx_14": 23.0,
                        "choppiness_14": 52.0,
                        "bb_squeeze": 1,
                        "bb_width_percentile_100": 0.15,
                        "close": 1.1050,
                        "donchian_low": 1.0990,
                        "donchian_high": 1.1110,
                    },
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1050,
            pattern_signals=[],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UNCLEAR")
        self.assertIn("breakout pending blocks RANGE rotation", forecast.rationale_summary)
        self.assertIn("BREAKOUT_PENDING", " ".join(forecast.drivers_for))

    def test_confirmed_range_breakout_can_forecast_direction(self) -> None:
        pair_chart = {
            "confluence": {
                "score_balance": "TIED",
                "score_gap": 0.0,
                "dominant_regime": "TRANSITION",
            },
            "views": [
                {
                    "granularity": "M15",
                    "regime": "RANGE",
                    "regime_reading": {"state": "BREAKOUT_PENDING", "confidence": 0.8, "atr_percentile": 12.0},
                    "indicators": {
                        "pip_size": 0.0001,
                        "close": 1.1101,
                        "donchian_low": 1.1000,
                        "donchian_high": 1.1100,
                        "linreg_slope_20": 0.4,
                    },
                    "family_scores": {"trend_score": 0.55, "breakout_score": 0.75, "disagreement": 0.1},
                },
                {
                    "granularity": "H1",
                    "regime": "RANGE",
                    "regime_reading": {"state": "BREAKOUT_PENDING", "confidence": 0.8, "atr_percentile": 15.0},
                    "indicators": {
                        "pip_size": 0.0001,
                        "close": 1.1101,
                        "donchian_low": 1.0990,
                        "donchian_high": 1.1100,
                        "linreg_slope_20": 0.3,
                    },
                    "family_scores": {"trend_score": 0.50, "breakout_score": 0.70, "disagreement": 0.1},
                },
            ],
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1101,
            pattern_signals=[],
            projection_signals=[],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertIn("range breakout confirmed UP", " ".join(forecast.drivers_for))

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

    def test_projection_signal_calibration_happens_before_forecast_selection(self) -> None:
        hit_rates = {
            "liquidity_sweep_low_up": {
                "EUR_USD:TREND": {"hit_rate": 0.0, "samples": 100},
            },
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart={"views": [{"granularity": "M15", "indicators": {"pip_size": 0.0001}}]},
            current_price=1.1000,
            pattern_signals=[_Sig("DOWN", 40.0, 1.0, "down setup")],
            projection_signals=[
                _NamedSig(
                    "liquidity_sweep_low",
                    "UP",
                    100.0,
                    1.0,
                    "bad historical sweep-low UP detector",
                )
            ],
            correlation_signals=[],
            paths=[],
            hit_rates=hit_rates,
            regime="TREND",
        )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertGreater(forecast.down_score, forecast.up_score)
        self.assertIn("[cal×", " ".join(forecast.drivers_against))

    def test_d_h4_anchor_extends_directional_forecast_horizon(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M5",
                    "indicators": {"pip_size": 0.0001, "atr_pips": 2.0},
                    "structure": {
                        "swings": [
                            {"side": "HIGH", "price": 1.1080},
                            {"side": "LOW", "price": 1.0975},
                        ]
                    },
                },
                {
                    "granularity": "H4",
                    "regime": "TREND_UP",
                    "indicators": {"pip_size": 0.0001, "adx_14": 34.0},
                    "family_scores": {"trend_score": 0.72, "disagreement": 0.0},
                },
                {
                    "granularity": "D",
                    "regime": "TREND_UP",
                    "indicators": {"pip_size": 0.0001, "adx_14": 31.0},
                    "family_scores": {"trend_score": 0.68, "disagreement": 0.0},
                },
            ]
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("UP", 35.0, 1.0, "daily continuation")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertEqual(forecast.horizon_min, FORECAST_D_ANCHOR_HORIZON_MIN)
        self.assertIn("D/H4 UP anchor", forecast.rationale_summary)
        self.assertIn("D/H4 UP anchor", " ".join(forecast.drivers_for))

    def test_h1_m30_anchor_uses_operating_swing_horizon(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M5",
                    "indicators": {"pip_size": 0.0001, "atr_pips": 2.0},
                    "structure": {
                        "swings": [
                            {"side": "HIGH", "price": 1.1025},
                            {"side": "LOW", "price": 1.0940},
                        ]
                    },
                },
                {
                    "granularity": "M30",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001},
                    "family_scores": {"trend_score": -0.70, "disagreement": 0.0},
                },
                {
                    "granularity": "H1",
                    "regime": "TREND_DOWN",
                    "indicators": {"pip_size": 0.0001},
                    "family_scores": {"trend_score": -0.74, "disagreement": 0.0},
                },
            ]
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("DOWN", 35.0, 1.0, "operating swing breakdown")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "DOWN")
        self.assertEqual(forecast.horizon_min, FORECAST_OPERATING_SWING_HORIZON_MIN)
        self.assertIn("H1 operating-swing DOWN", forecast.rationale_summary)
        self.assertIn("H1 operating-swing DOWN", " ".join(forecast.drivers_for))

    def test_micro_only_forecast_keeps_execution_horizon(self) -> None:
        pair_chart = {
            "views": [
                {
                    "granularity": "M5",
                    "regime": "TREND_UP",
                    "indicators": {"pip_size": 0.0001, "atr_pips": 2.0, "adx_14": 31.0},
                    "family_scores": {"trend_score": 0.80, "disagreement": 0.0},
                    "structure": {
                        "swings": [
                            {"side": "HIGH", "price": 1.1040},
                            {"side": "LOW", "price": 1.0975},
                        ]
                    },
                }
            ]
        }

        forecast = synthesize_forecast(
            pair="EUR_USD",
            pair_chart=pair_chart,
            current_price=1.1000,
            pattern_signals=[],
            projection_signals=[_Sig("UP", 35.0, 1.0, "M5 impulse")],
            correlation_signals=[],
            paths=[],
        )

        self.assertEqual(forecast.direction, "UP")
        self.assertEqual(forecast.horizon_min, FORECAST_EXECUTION_HORIZON_MIN)
        self.assertNotIn("anchor horizon", forecast.rationale_summary)


class ContestedRangePhaseEvidenceTest(unittest.TestCase):
    """_contested_range_raw_confidence phase-evidence path (2026-06-11).

    Live funnel regression: 19/28 pairs sat in measured boxes while the
    production signal book inflated directional scores enough that
    range_score < margin, so 60+ rotation lanes per cycle died as UNCLEAR.
    The phase detector's own confidence (>= 0.5 == its documented
    minimum-evidence bar) now qualifies RANGE resolution directly.
    """

    @staticmethod
    def _compute(phase: str, phase_conf: float, *, range_score: float, margin_pair=(100.0, 92.0)):
        from quant_rabbit.strategy.directional_forecaster import (
            _RangePhase,
            _contested_range_raw_confidence,
        )

        winner, runner_up = margin_pair
        return _contested_range_raw_confidence(
            phase=_RangePhase(
                phase=phase, confidence=phase_conf, direction=None,
                rationale="test", evidence=("e1",),
            ),
            winner_score=winner,
            runner_up_score=runner_up,
            range_score=range_score,
            margin=winner - runner_up,
        )

    def test_confident_box_qualifies_even_when_range_echo_lost_to_noise(self) -> None:
        # margin 8 > range_score 5, but the box itself measured 0.66.
        conf = self._compute("IN_RANGE", 0.66, range_score=5.0)
        self.assertIsNotNone(conf)
        # (directional_uncertainty 0.92 + phase 0.66) / 2 = 0.79
        self.assertAlmostEqual(conf, 0.79, places=2)

    def test_weak_box_below_min_evidence_stays_unclear(self) -> None:
        self.assertIsNone(self._compute("RANGE_FORMING", 0.49, range_score=5.0))

    def test_breakout_pending_box_never_qualifies(self) -> None:
        self.assertIsNone(self._compute("BREAKOUT_PENDING", 1.0, range_score=5.0))

    def test_strong_range_echo_path_unchanged(self) -> None:
        # range_score 20 >= margin 8 → original blend, phase conf irrelevant.
        conf = self._compute("IN_RANGE", 0.0, range_score=20.0)
        self.assertIsNotNone(conf)
        # (0.92 + min(1, 20/92)) / 2 ≈ 0.5687
        self.assertAlmostEqual(conf, (0.92 + 20.0 / 92.0) / 2.0, places=3)

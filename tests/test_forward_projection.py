"""Regression tests for forward projection direction semantics."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timezone

from quant_rabbit.strategy.forward_projection import (
    ProjectionSignal,
    aggregate_projection_score,
    detect_forward_projections,
)
from quant_rabbit.strategy.predictive_limit_orders import generate_limits_from_projections


_QUIET_SESSION_NOW = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)


@dataclass
class _Sig:
    name: str
    direction: str
    rationale: str


def _chart_with_liquidity(*, side: str, price: float) -> dict:
    return {
        "pair": "EUR_USD",
        "views": [
            {
                "granularity": "M5",
                "indicators": {"atr_pips": 10.0},
                "structure": {
                    "liquidity": [
                        {"side": side, "price": price, "indices": [1, 2, 3]},
                    ],
                },
            }
        ],
    }


class LiquiditySweepDirectionTest(unittest.TestCase):
    def test_equal_high_sweep_is_short_entry_bias(self) -> None:
        signals = detect_forward_projections(
            _chart_with_liquidity(side="EQ_HIGH", price=1.1003),
            pair="EUR_USD",
            current_price=1.1000,
            now=_QUIET_SESSION_NOW,
        )

        sweep = next(s for s in signals if s.name == "liquidity_sweep_high")
        self.assertEqual(sweep.direction, "DOWN")

        short_score, short_reasons = aggregate_projection_score(signals, "SHORT")
        long_score, long_reasons = aggregate_projection_score(signals, "LONG")

        self.assertGreater(short_score, 0.0, short_reasons)
        self.assertLess(long_score, 0.0, long_reasons)

    def test_equal_low_sweep_is_long_entry_bias(self) -> None:
        signals = detect_forward_projections(
            _chart_with_liquidity(side="EQ_LOW", price=1.0997),
            pair="EUR_USD",
            current_price=1.1000,
            now=_QUIET_SESSION_NOW,
        )

        sweep = next(s for s in signals if s.name == "liquidity_sweep_low")
        self.assertEqual(sweep.direction, "UP")

        long_score, long_reasons = aggregate_projection_score(signals, "LONG")
        short_score, short_reasons = aggregate_projection_score(signals, "SHORT")

        self.assertGreater(long_score, 0.0, long_reasons)
        self.assertLess(short_score, 0.0, short_reasons)

    def test_projection_score_prefers_direction_specific_calibration(self) -> None:
        signals = [
            ProjectionSignal(
                name="liquidity_sweep_high",
                timeframe="M5",
                direction="UP",
                lead_time_min=10,
                confidence=1.0,
                bonus_magnitude=10.0,
                rationale="EUR_USD sweep-high UP historically fails",
            )
        ]
        hit_rates = {
            "liquidity_sweep_high": {
                "EUR_USD:TREND": {"hit_rate": 1.0, "samples": 100},
            },
            "liquidity_sweep_high_up": {
                "EUR_USD:TREND": {"hit_rate": 0.0, "samples": 100},
            },
        }

        score, reasons = aggregate_projection_score(
            signals,
            "LONG",
            hit_rates=hit_rates,
            pair="EUR_USD",
            regime="TREND",
        )

        self.assertLess(score, 5.0)
        self.assertTrue(any("[cal×" in reason for reason in reasons), reasons)

    def test_predictive_limit_fades_equal_high_by_signal_name(self) -> None:
        signals = [
            _Sig(
                "liquidity_sweep_high",
                "DOWN",
                f"M5 equal-highs at {1.1003 + i * 0.00001:.5f} (3.0pip up)",
            )
            for i in range(4)
        ]

        orders = generate_limits_from_projections(
            pair="EUR_USD",
            pair_chart={
                "views": [{"granularity": "M15", "indicators": {"atr_pips": 10.0}}],
            },
            current_bid=1.1000,
            current_ask=1.1001,
            projection_signals=signals,
            paths=[],
        )

        self.assertTrue(orders)
        self.assertTrue(all(o.side == "SHORT" for o in orders))

    def test_predictive_limit_fades_equal_low_by_signal_name(self) -> None:
        signals = [
            _Sig(
                "liquidity_sweep_low",
                "UP",
                f"M5 equal-lows at {1.0997 - i * 0.00001:.5f} (3.0pip down)",
            )
            for i in range(4)
        ]

        orders = generate_limits_from_projections(
            pair="EUR_USD",
            pair_chart={
                "views": [{"granularity": "M15", "indicators": {"atr_pips": 10.0}}],
            },
            current_bid=1.1000,
            current_ask=1.1001,
            projection_signals=signals,
            paths=[],
        )

        self.assertTrue(orders)
        self.assertTrue(all(o.side == "LONG" for o in orders))

    def test_predictive_limit_allows_small_grade_b_early_turn_equal_low(self) -> None:
        signals = [
            _Sig(
                "liquidity_sweep_low",
                "UP",
                "M15 equal-lows at 1.09970 (3.0pip down)",
            )
        ]

        orders = generate_limits_from_projections(
            pair="EUR_USD",
            pair_chart={
                "confluence": {"price_percentile_24h": 0.05, "price_percentile_7d": 0.02},
                "views": [
                    {
                        "granularity": "M15",
                        "indicators": {
                            "atr_pips": 10.0,
                            "bb_lower": 1.0996,
                            "bb_middle": 1.1005,
                            "close": 1.09975,
                            "rsi_14": 31.0,
                            "williams_r_14": -92.0,
                        },
                        "structure": {"last_event": {"kind": "BOS_DOWN", "close_confirmed": False}},
                    }
                ],
            },
            current_bid=1.1000,
            current_ask=1.1001,
            projection_signals=signals,
            paths=[],
        )

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, "LONG")
        self.assertEqual(orders[0].grade, "B")
        self.assertLess(orders[0].units, 5000)

    def test_predictive_limit_rejects_thin_grade_b_without_extreme_context(self) -> None:
        signals = [
            _Sig(
                "liquidity_sweep_low",
                "UP",
                "M15 equal-lows at 1.09970 (3.0pip down)",
            )
        ]

        orders = generate_limits_from_projections(
            pair="EUR_USD",
            pair_chart={
                "confluence": {"price_percentile_24h": 0.45, "price_percentile_7d": 0.50},
                "views": [
                    {
                        "granularity": "M15",
                        "indicators": {
                            "atr_pips": 10.0,
                            "bb_lower": 1.0990,
                            "bb_middle": 1.1005,
                            "close": 1.1002,
                            "rsi_14": 49.0,
                            "williams_r_14": -45.0,
                        },
                        "structure": {"last_event": {"kind": "BOS_DOWN", "close_confirmed": False}},
                    }
                ],
            },
            current_bid=1.1000,
            current_ask=1.1001,
            projection_signals=signals,
            paths=[],
        )

        self.assertEqual(orders, [])

    def test_predictive_limit_dedupes_same_liquidity_level(self) -> None:
        signals = [
            _Sig("liquidity_sweep_low", "UP", "M5 equal-lows at 1.09970 (3.0pip down)"),
            _Sig("liquidity_sweep_low", "UP", "M15 equal-lows at 1.09970 (3.0pip down)"),
        ]

        orders = generate_limits_from_projections(
            pair="EUR_USD",
            pair_chart={
                "confluence": {"price_percentile_24h": 0.03},
                "views": [
                    {
                        "granularity": "M5",
                        "indicators": {
                            "atr_pips": 10.0,
                            "bb_lower": 1.0996,
                            "bb_middle": 1.1005,
                            "close": 1.0997,
                            "rsi_14": 35.0,
                        },
                    }
                ],
            },
            current_bid=1.1000,
            current_ask=1.1001,
            projection_signals=signals,
            paths=[],
        )

        self.assertEqual(len(orders), 1)

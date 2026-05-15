"""Regression tests for forward projection direction semantics."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from quant_rabbit.strategy.forward_projection import (
    aggregate_projection_score,
    detect_forward_projections,
)
from quant_rabbit.strategy.predictive_limit_orders import generate_limits_from_projections


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
        )

        sweep = next(s for s in signals if s.name == "liquidity_sweep_low")
        self.assertEqual(sweep.direction, "UP")

        long_score, long_reasons = aggregate_projection_score(signals, "LONG")
        short_score, short_reasons = aggregate_projection_score(signals, "SHORT")

        self.assertGreater(long_score, 0.0, long_reasons)
        self.assertLess(short_score, 0.0, short_reasons)

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

"""Unit tests for strategy/pattern_signals.py."""

from __future__ import annotations

import os
import unittest

from quant_rabbit.strategy.pattern_signals import (
    FAILED_BREAKOUT_BONUS,
    PATTERN_TOTAL_CAP,
    PatternSignal,
    aggregate_pattern_score,
    detect_pattern_signals,
)


def _view(
    *,
    tf: str = "M15",
    rsi: float | None = None,
    close: float | None = None,
    bb_upper: float | None = None,
    bb_lower: float | None = None,
    aroon_up: float | None = None,
    aroon_down: float | None = None,
    structure_events: list | None = None,
    dealing_range: dict | None = None,
) -> dict:
    ind = {}
    if rsi is not None: ind["rsi_14"] = rsi
    if close is not None: ind["close"] = close
    if bb_upper is not None: ind["bb_upper"] = bb_upper
    if bb_lower is not None: ind["bb_lower"] = bb_lower
    if aroon_up is not None: ind["aroon_up_14"] = aroon_up
    if aroon_down is not None: ind["aroon_down_14"] = aroon_down
    view = {"granularity": tf, "indicators": ind}
    if structure_events is not None:
        view["structure"] = {"structure_events": structure_events}
    if dealing_range is not None:
        view["smc"] = {"dealing_range": dealing_range}
    return view


def _chart(views: list[dict]) -> dict:
    return {"pair": "EUR_USD", "views": views}


class FailedBreakoutTest(unittest.TestCase):
    def setUp(self) -> None:
        if "QR_DISABLE_PATTERN_SIGNALS" in os.environ:
            del os.environ["QR_DISABLE_PATTERN_SIGNALS"]

    def test_wick_only_bos_up_fades_down(self) -> None:
        chart = _chart([_view(
            structure_events=[
                {"kind": "BOS_UP", "broken_pivot_price": 1.17, "close_confirmed": False, "index": 10},
            ],
        )])
        signals = detect_pattern_signals(chart)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].name, "failed_breakout")
        self.assertEqual(signals[0].direction, "DOWN")

    def test_close_confirmed_bos_does_not_fire(self) -> None:
        chart = _chart([_view(
            structure_events=[
                {"kind": "BOS_UP", "broken_pivot_price": 1.17, "close_confirmed": True, "index": 10},
            ],
        )])
        signals = detect_pattern_signals(chart)
        self.assertEqual(len(signals), 0)

    def test_wick_only_bos_down_fades_up(self) -> None:
        chart = _chart([_view(
            structure_events=[
                {"kind": "BOS_DOWN", "broken_pivot_price": 1.17, "close_confirmed": False, "index": 10},
            ],
        )])
        signals = detect_pattern_signals(chart)
        self.assertEqual(signals[0].direction, "UP")


class RSIExtremeTest(unittest.TestCase):
    def test_rsi_overbought_near_bb_upper_fires_down(self) -> None:
        chart = _chart([_view(rsi=75, close=1.180, bb_upper=1.181, bb_lower=1.165)])
        signals = detect_pattern_signals(chart)
        rsi_signals = [s for s in signals if "rsi_extreme" in s.name]
        self.assertEqual(len(rsi_signals), 1)
        self.assertEqual(rsi_signals[0].direction, "DOWN")

    def test_rsi_oversold_near_bb_lower_fires_up(self) -> None:
        chart = _chart([_view(rsi=25, close=1.166, bb_upper=1.181, bb_lower=1.165)])
        signals = detect_pattern_signals(chart)
        rsi_signals = [s for s in signals if "rsi_extreme" in s.name]
        self.assertEqual(rsi_signals[0].direction, "UP")

    def test_rsi_overbought_mid_BB_no_fire(self) -> None:
        """RSI 75 alone isn't enough — need price at the rail too."""
        chart = _chart([_view(rsi=75, close=1.173, bb_upper=1.181, bb_lower=1.165)])
        signals = detect_pattern_signals(chart)
        self.assertEqual([s for s in signals if "rsi_extreme" in s.name], [])


class DealingRangeTest(unittest.TestCase):
    def test_at_swing_high_fires_down(self) -> None:
        chart = _chart([_view(
            close=1.180, dealing_range={"swing_high": 1.180, "swing_low": 1.160},
        )])
        signals = detect_pattern_signals(chart)
        dr = [s for s in signals if "dealing_range" in s.name]
        self.assertEqual(dr[0].direction, "DOWN")

    def test_at_swing_low_fires_up(self) -> None:
        chart = _chart([_view(
            close=1.160, dealing_range={"swing_high": 1.180, "swing_low": 1.160},
        )])
        signals = detect_pattern_signals(chart)
        dr = [s for s in signals if "dealing_range" in s.name]
        self.assertEqual(dr[0].direction, "UP")

    def test_mid_range_no_fire(self) -> None:
        chart = _chart([_view(
            close=1.170, dealing_range={"swing_high": 1.180, "swing_low": 1.160},
        )])
        signals = detect_pattern_signals(chart)
        self.assertEqual([s for s in signals if "dealing_range" in s.name], [])


class AroonFlipTest(unittest.TestCase):
    def test_strong_up_momentum(self) -> None:
        chart = _chart([_view(aroon_up=85, aroon_down=15)])
        signals = detect_pattern_signals(chart)
        a = [s for s in signals if "aroon" in s.name]
        self.assertEqual(a[0].direction, "UP")

    def test_strong_down_momentum(self) -> None:
        chart = _chart([_view(aroon_up=10, aroon_down=80)])
        signals = detect_pattern_signals(chart)
        a = [s for s in signals if "aroon" in s.name]
        self.assertEqual(a[0].direction, "DOWN")


class AggregateScoreTest(unittest.TestCase):
    def test_aligned_signal_adds_full_magnitude(self) -> None:
        sig = PatternSignal("x", "M15", "UP", 1.0, 10.0, "test")
        total, _ = aggregate_pattern_score([sig], "LONG")
        self.assertAlmostEqual(total, 10.0)

    def test_opposed_signal_subtracts_half_magnitude(self) -> None:
        sig = PatternSignal("x", "M15", "DOWN", 1.0, 10.0, "test")
        total, _ = aggregate_pattern_score([sig], "LONG")
        self.assertAlmostEqual(total, -5.0)

    def test_total_clamped_to_cap(self) -> None:
        signals = [PatternSignal(f"s{i}", "M15", "UP", 1.0, 50.0, "test") for i in range(5)]
        total, _ = aggregate_pattern_score(signals, "LONG")
        self.assertAlmostEqual(total, PATTERN_TOTAL_CAP)

    def test_confidence_scales_contribution(self) -> None:
        sig = PatternSignal("x", "M15", "UP", 0.5, 10.0, "test")
        total, _ = aggregate_pattern_score([sig], "LONG")
        self.assertAlmostEqual(total, 5.0)


class KillSwitchTest(unittest.TestCase):
    def test_disabled_returns_empty(self) -> None:
        os.environ["QR_DISABLE_PATTERN_SIGNALS"] = "1"
        try:
            chart = _chart([_view(rsi=75, close=1.180, bb_upper=1.181, bb_lower=1.165)])
            self.assertEqual(detect_pattern_signals(chart), [])
        finally:
            del os.environ["QR_DISABLE_PATTERN_SIGNALS"]


if __name__ == "__main__":
    unittest.main()

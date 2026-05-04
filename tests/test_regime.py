from __future__ import annotations

import math
import random
import unittest

from quant_rabbit.analysis.regime import (
    RegimeReading,
    classify_regime,
)


def _ohlc_from_closes(closes: list[float], wiggle: float = 0.05) -> tuple[list[float], list[float]]:
    """Synthesize highs/lows that bracket each close by `wiggle`."""
    highs: list[float] = []
    lows: list[float] = []
    prev = closes[0]
    for c in closes:
        h = max(prev, c) + wiggle
        lo = min(prev, c) - wiggle
        highs.append(h)
        lows.append(lo)
        prev = c
    return highs, lows


class RegimeEmptyAndEdgeTest(unittest.TestCase):
    def test_empty_input_returns_unknown(self) -> None:
        result = classify_regime(closes=[], highs=[], lows=[])
        self.assertEqual(result.state, "UNKNOWN")
        self.assertEqual(result.confidence, 0.0)

    def test_single_value_returns_unknown(self) -> None:
        result = classify_regime(closes=[100.0], highs=[100.1], lows=[99.9])
        self.assertEqual(result.state, "UNKNOWN")

    def test_misaligned_inputs_returns_unknown(self) -> None:
        # highs / lows length mismatch.
        result = classify_regime(closes=[1.0, 2.0, 3.0], highs=[1.0], lows=[1.0])
        self.assertEqual(result.state, "UNKNOWN")

    def test_short_series_returns_unknown(self) -> None:
        closes = [100.0 + i * 0.01 for i in range(30)]
        highs, lows = _ohlc_from_closes(closes)
        result = classify_regime(closes=closes, highs=highs, lows=lows)
        # Hurst window default 200 cannot be filled -> UNKNOWN.
        self.assertEqual(result.state, "UNKNOWN")
        self.assertEqual(result.confidence, 0.0)


class RegimeTrendTest(unittest.TestCase):
    def test_strong_uptrend_yields_trend_strong_or_weak(self) -> None:
        # Linear ramp with mild noise — Hurst should rise above 0.55,
        # ADX should be elevated, Choppiness depressed.
        random.seed(42)
        closes = [100.0 + i * 0.05 + random.gauss(0, 0.02) for i in range(800)]
        highs, lows = _ohlc_from_closes(closes, wiggle=0.03)
        result = classify_regime(
            closes=closes,
            highs=highs,
            lows=lows,
            bars_per_year=500,
            hurst_window=200,
        )
        # A clean ramp should at minimum register as TREND_WEAK or
        # TREND_STRONG; specifically not RANGE.
        self.assertIn(result.state, ("TREND_STRONG", "TREND_WEAK"))
        self.assertNotEqual(result.state, "RANGE")
        self.assertIsNotNone(result.hurst)
        self.assertIsNotNone(result.adx)
        self.assertIsNotNone(result.choppiness)


class RegimeRangeTest(unittest.TestCase):
    def test_anti_persistent_series_yields_low_hurst(self) -> None:
        # AR(1) with negative autocorrelation = canonical anti-persistent
        # (mean-reverting) series. Hurst < 0.45 expected per the DFA
        # threshold convention (HURST_RANGE_THRESHOLD).
        random.seed(11)
        n = 800
        prev = 100.0
        mean_lvl = 100.0
        closes: list[float] = []
        for _ in range(n):
            eps = random.gauss(0, 0.05)
            cur = mean_lvl - 0.5 * (prev - mean_lvl) + eps
            closes.append(cur)
            prev = cur
        highs, lows = _ohlc_from_closes(closes, wiggle=0.02)
        result = classify_regime(
            closes=closes,
            highs=highs,
            lows=lows,
            bars_per_year=500,
            hurst_window=200,
        )
        self.assertNotEqual(result.state, "TREND_STRONG")
        self.assertNotEqual(result.state, "TREND_WEAK")
        self.assertIsNotNone(result.hurst)
        assert result.hurst is not None
        # Anti-persistent -> Hurst measurably <0.45 (the RANGE threshold).
        self.assertLess(result.hurst, 0.45)


class RegimeReadingDataclassTest(unittest.TestCase):
    def test_dataclass_is_frozen(self) -> None:
        r = RegimeReading(
            state="UNKNOWN",
            hurst=None,
            adx=None,
            choppiness=None,
            atr_percentile=None,
            confidence=0.0,
        )
        with self.assertRaises(Exception):
            r.state = "TREND_STRONG"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

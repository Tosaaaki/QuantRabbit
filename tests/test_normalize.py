from __future__ import annotations

import math
import unittest

from quant_rabbit.analysis.normalize import (
    NormalizedValue,
    normalize_indicator,
    rolling_percentile_rank,
    rolling_z,
)


class RollingZTest(unittest.TestCase):
    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(rolling_z([]), [])

    def test_window_underfilled_returns_none(self) -> None:
        out = rolling_z([1.0], window=10)
        self.assertEqual(out, [None])

    def test_all_none_input(self) -> None:
        out = rolling_z([None, None, None], window=10)
        self.assertEqual(out, [None, None, None])

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            rolling_z([1.0, 2.0, 3.0], window=1)

    def test_constant_series_yields_zero_z(self) -> None:
        # Constant -> stdev=0 -> defined Z=0 once window is filled.
        series = [5.0] * 30
        out = rolling_z(series, window=10)
        # First value can't have a Z (no prior history).
        self.assertIsNone(out[0])
        # By bar 10 we have 10 prior samples in the trailing window.
        self.assertEqual(out[15], 0.0)

    def test_skips_none_in_window(self) -> None:
        # Mix valid and None; the trailing window should still get filled.
        series: list[float | None] = [1.0, None, 2.0, None, 3.0, None, 4.0, None, 5.0]
        out = rolling_z(series, window=3)
        # Last entry should be a valid Z (3 prior valid samples: 2, 3, 4).
        self.assertIsNotNone(out[-1])

    def test_uptrend_yields_positive_z_at_end(self) -> None:
        series = [float(i) for i in range(50)]
        out = rolling_z(series, window=10)
        # Final value far above its trailing window mean -> positive Z.
        self.assertIsNotNone(out[-1])
        assert out[-1] is not None
        self.assertGreater(out[-1], 0.5)

    def test_sine_wave_z_oscillates(self) -> None:
        series = [math.sin(i * 0.2) for i in range(100)]
        out = rolling_z(series, window=20)
        # Some values should be positive and some negative.
        non_none = [v for v in out if v is not None]
        self.assertTrue(any(v > 0 for v in non_none))
        self.assertTrue(any(v < 0 for v in non_none))


class RollingPercentileTest(unittest.TestCase):
    def test_empty_input(self) -> None:
        self.assertEqual(rolling_percentile_rank([]), [])

    def test_single_value(self) -> None:
        # Window underfilled (need >= 2 samples) -> None.
        out = rolling_percentile_rank([42.0])
        self.assertEqual(out, [None])

    def test_all_none(self) -> None:
        out = rolling_percentile_rank([None, None, None])
        self.assertEqual(out, [None, None, None])

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            rolling_percentile_rank([1.0, 2.0], window=1)

    def test_max_value_yields_100_pct(self) -> None:
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = rolling_percentile_rank(series, window=10)
        # The highest sample at the end should land at 100.
        self.assertEqual(out[-1], 100.0)

    def test_min_value_yields_low_pct(self) -> None:
        # Reversed: last value is the smallest in its window.
        series = [5.0, 4.0, 3.0, 2.0, 1.0]
        out = rolling_percentile_rank(series, window=10)
        # The smallest sample at the end -> only itself is at-or-below.
        assert out[-1] is not None
        self.assertLess(out[-1], 30.0)


class NormalizeIndicatorTest(unittest.TestCase):
    def test_empty_input(self) -> None:
        self.assertEqual(normalize_indicator([]), [])

    def test_returns_normalized_value_for_each_input(self) -> None:
        series = [float(i) for i in range(30)]
        out = normalize_indicator(series, window_z=10, window_pct=20)
        self.assertEqual(len(out), 30)
        self.assertIsInstance(out[0], NormalizedValue)

    def test_underfilled_returns_raw_with_none_normalizations(self) -> None:
        out = normalize_indicator([3.14], window_z=10, window_pct=10)
        self.assertEqual(out[0].raw, 3.14)
        self.assertIsNone(out[0].z_score)
        self.assertIsNone(out[0].percentile)

    def test_none_raw_propagates(self) -> None:
        out = normalize_indicator([None, 1.0, 2.0, 3.0], window_z=2, window_pct=2)
        self.assertIsNone(out[0].raw)
        self.assertIsNone(out[0].z_score)
        self.assertIsNone(out[0].percentile)


if __name__ == "__main__":
    unittest.main()

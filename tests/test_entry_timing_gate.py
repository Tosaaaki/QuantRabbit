"""Unit tests for strategy/entry_timing_gate.py."""

from __future__ import annotations

import unittest

from quant_rabbit.strategy.entry_timing_gate import (
    ENTRY_TIMING_AGAINST_PENALTY,
    ENTRY_TIMING_ALIGNED_BONUS,
    ENTRY_TIMING_MIXED_PENALTY,
    check_entry_timing,
)


def _m5_view(*close_dirs: int) -> dict:
    """Build a pair_chart with M5 candles whose close-open sign matches
    each provided sign (+1 = up close, -1 = down close)."""
    candles = []
    for i, d in enumerate(close_dirs):
        o = 1.1000 + i * 0.0001
        c = o + (0.0002 if d > 0 else -0.0002)
        candles.append({"open": o, "close": c})
    return {
        "pair": "EUR_USD",
        "views": [{"timeframe": "M5", "candles": candles}],
    }


class EntryTimingGateTest(unittest.TestCase):
    def test_no_chart_returns_unknown(self) -> None:
        r = check_entry_timing(None, "LONG")
        self.assertEqual(r.state, "UNKNOWN")
        self.assertEqual(r.score_delta, 0.0)

    def test_no_m5_view_returns_unknown(self) -> None:
        r = check_entry_timing({"pair": "EUR_USD", "views": []}, "LONG")
        self.assertEqual(r.state, "UNKNOWN")

    def test_fewer_than_3_candles_returns_unknown(self) -> None:
        chart = _m5_view(+1, +1)
        r = check_entry_timing(chart, "LONG")
        self.assertEqual(r.state, "UNKNOWN")

    def test_aligned_long_all_up(self) -> None:
        chart = _m5_view(+1, +1, +1)
        r = check_entry_timing(chart, "LONG")
        self.assertEqual(r.state, "ALIGNED")
        self.assertAlmostEqual(r.score_delta, ENTRY_TIMING_ALIGNED_BONUS)
        self.assertIn("ALIGNED", r.rationale or "")

    def test_aligned_short_all_down(self) -> None:
        chart = _m5_view(-1, -1, -1)
        r = check_entry_timing(chart, "SHORT")
        self.assertEqual(r.state, "ALIGNED")
        self.assertAlmostEqual(r.score_delta, ENTRY_TIMING_ALIGNED_BONUS)

    def test_against_long_all_down(self) -> None:
        chart = _m5_view(-1, -1, -1)
        r = check_entry_timing(chart, "LONG")
        self.assertEqual(r.state, "AGAINST")
        self.assertAlmostEqual(r.score_delta, -ENTRY_TIMING_AGAINST_PENALTY)
        self.assertIn("AGAINST", r.rationale or "")

    def test_against_short_all_up(self) -> None:
        chart = _m5_view(+1, +1, +1)
        r = check_entry_timing(chart, "SHORT")
        self.assertEqual(r.state, "AGAINST")

    def test_mixed_returns_small_penalty(self) -> None:
        chart = _m5_view(+1, -1, +1)
        r = check_entry_timing(chart, "LONG")
        self.assertEqual(r.state, "MIXED")
        self.assertAlmostEqual(r.score_delta, -ENTRY_TIMING_MIXED_PENALTY)
        chart2 = _m5_view(-1, +1, -1)
        r2 = check_entry_timing(chart2, "SHORT")
        self.assertEqual(r2.state, "MIXED")

    def test_only_last_three_candles_count(self) -> None:
        """5-candle chart, last 3 are all UP → aligned LONG even if older candles disagree."""
        chart = _m5_view(-1, -1, +1, +1, +1)
        r = check_entry_timing(chart, "LONG")
        self.assertEqual(r.state, "ALIGNED")


if __name__ == "__main__":
    unittest.main()

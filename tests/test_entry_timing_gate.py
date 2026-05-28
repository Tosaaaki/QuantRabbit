"""Unit tests for strategy/entry_timing_gate.py."""

from __future__ import annotations

import unittest

from quant_rabbit.strategy.entry_timing_gate import (
    ENTRY_TIMING_AGAINST_PENALTY,
    ENTRY_TIMING_ALIGNED_BONUS,
    ENTRY_TIMING_MIXED_PENALTY,
    OPERATING_TF_MOMENTUM_AGAINST_PENALTY,
    check_entry_timing,
    check_operating_tf_momentum,
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


def _m5_chart_reader_view(*close_dirs: int) -> dict:
    """Build the shape emitted by chart_reader.PairChart.to_dict()."""
    candles = []
    for i, d in enumerate(close_dirs):
        o = 1.1000 + i * 0.0001
        c = o + (0.0002 if d > 0 else -0.0002)
        candles.append({"o": o, "c": c})
    return {
        "pair": "EUR_USD",
        "views": [{"granularity": "M5", "recent_candles": candles}],
    }


def _operating_tf_chart(*, m5: tuple[str, float], m15: tuple[str, float], m30: tuple[str, float]) -> dict:
    return {
        "pair": "EUR_USD",
        "views": [
            {"granularity": "M5", "regime": m5[0], "indicators": {"adx_14": m5[1]}},
            {"granularity": "M15", "regime": m15[0], "indicators": {"adx_14": m15[1]}},
            {"granularity": "M30", "regime": m30[0], "indicators": {"adx_14": m30[1]}},
        ],
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

    def test_reads_chart_reader_granularity_recent_candles_shape(self) -> None:
        chart = _m5_chart_reader_view(+1, +1, +1)
        r = check_entry_timing(chart, "SHORT")
        self.assertEqual(r.state, "AGAINST")
        self.assertAlmostEqual(r.score_delta, -ENTRY_TIMING_AGAINST_PENALTY)

    def test_operating_tf_momentum_blocks_short_against_up_stack(self) -> None:
        chart = _operating_tf_chart(
            m5=("TREND_UP", 44.0),
            m15=("TREND_UP", 32.0),
            m30=("IMPULSE_UP", 21.0),
        )

        r = check_operating_tf_momentum(chart, "SHORT")

        self.assertEqual(r.state, "AGAINST")
        self.assertAlmostEqual(r.score_delta, -OPERATING_TF_MOMENTUM_AGAINST_PENALTY)
        self.assertIn("M5 TREND_UP", r.rationale or "")
        self.assertIn("M30 IMPULSE_UP", r.rationale or "")

    def test_operating_tf_momentum_needs_multi_tf_counter_pressure(self) -> None:
        chart = _operating_tf_chart(
            m5=("TREND_UP", 28.0),
            m15=("RANGE", 14.0),
            m30=("RANGE", 18.0),
        )

        r = check_operating_tf_momentum(chart, "SHORT")

        self.assertEqual(r.state, "MIXED")
        self.assertEqual(r.score_delta, 0.0)


if __name__ == "__main__":
    unittest.main()

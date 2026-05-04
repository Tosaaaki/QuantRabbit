from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.analysis.candles import Candle
from quant_rabbit.analysis.chart_reader import build_pair_chart


def _series(start: float, step: float, n: int = 100) -> list[Candle]:
    base = datetime(2026, 5, 4, tzinfo=timezone.utc)
    out: list[Candle] = []
    prev = start
    for i in range(n):
        c = start + step * i
        h = max(prev, c) + abs(step) * 0.5 + 0.01
        low_value = min(prev, c) - abs(step) * 0.5 - 0.01
        out.append(Candle(base + timedelta(minutes=5 * i), prev, h, low_value, c, 1000, True))
        prev = c
    return out


class ChartReaderTest(unittest.TestCase):
    def test_uptrend_pair_scores_long_above_short(self) -> None:
        candles_by_tf = {
            "M5": _series(156.0, 0.04),
            "M15": _series(156.0, 0.05),
            "H1": _series(156.0, 0.08),
        }
        chart = build_pair_chart(
            "USD_JPY",
            client=None,  # type: ignore[arg-type]
            candles_by_tf=candles_by_tf,
        )
        self.assertGreater(chart.long_score, chart.short_score)
        self.assertIn("USD_JPY", chart.chart_story)
        self.assertEqual(len(chart.views), 3)

    def test_downtrend_pair_scores_short_above_long(self) -> None:
        candles_by_tf = {
            "M5": _series(160.0, -0.04),
            "M15": _series(160.0, -0.05),
            "H1": _series(160.0, -0.08),
        }
        chart = build_pair_chart(
            "USD_JPY",
            client=None,  # type: ignore[arg-type]
            candles_by_tf=candles_by_tf,
        )
        self.assertGreater(chart.short_score, chart.long_score)
        self.assertIn("TREND_DOWN", chart.dominant_regime)

    def test_chart_story_includes_indicator_fragments(self) -> None:
        candles_by_tf = {"M5": _series(157.0, 0.03)}
        chart = build_pair_chart(
            "USD_JPY",
            client=None,  # type: ignore[arg-type]
            timeframes=("M5",),
            candles_by_tf=candles_by_tf,
        )
        self.assertIn("ADX=", chart.chart_story)
        self.assertIn("ATR=", chart.chart_story)
        self.assertIn("RSI=", chart.chart_story)


if __name__ == "__main__":
    unittest.main()

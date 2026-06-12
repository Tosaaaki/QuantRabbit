from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from quant_rabbit.projection_truth import load_projection_candle_truth


class ProjectionTruthTest(unittest.TestCase):
    def test_candle_truth_deadline_stops_before_unbounded_pair_loop(self) -> None:
        calls: list[tuple[str, str, int]] = []

        def _fetcher(_client, pair: str, granularity: str, *, count: int):
            calls.append((pair, granularity, count))
            return [SimpleNamespace(close=1.1)]

        with mock.patch(
            "quant_rabbit.projection_truth.time.monotonic",
            side_effect=[0.0, 0.0, 1.2, 1.2],
        ):
            result = load_projection_candle_truth(
                object(),
                ["GBP_USD", "EUR_USD"],
                m1_count=2,
                m5_count=2,
                budget_seconds=1.0,
                fetcher=_fetcher,
            )

        self.assertEqual(calls, [("EUR_USD", "M1", 2)])
        self.assertTrue(result.deadline_exceeded)
        self.assertEqual(result.candle_granularity_counts, {"EUR_USD": {"M1": 1}})
        self.assertIn("_deadline", result.candle_errors)


if __name__ == "__main__":
    unittest.main()

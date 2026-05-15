from __future__ import annotations

import unittest

from quant_rabbit.analysis.score_momentum import attach_score_momentum


class ScoreMomentumTest(unittest.TestCase):
    def test_attaches_pair_score_slope_from_previous_snapshot(self) -> None:
        charts = [
            {
                "pair": "EUR_USD",
                "long_score": 0.29,
                "short_score": 0.69,
                "confluence": {"score_gap": -0.40},
            }
        ]
        previous = {
            "generated_at_utc": "2026-05-15T10:00:00+00:00",
            "charts": [
                {
                    "pair": "EUR_USD",
                    "long_score": 0.05,
                    "short_score": 0.95,
                    "confluence": {"score_gap": -0.90},
                }
            ],
        }

        attach_score_momentum(charts, previous, "2026-05-15T11:00:00+00:00")

        momentum = charts[0]["confluence"]["score_momentum"]
        self.assertEqual(momentum["direction"], "UP")
        self.assertEqual(momentum["elapsed_min"], 60.0)
        self.assertAlmostEqual(momentum["long_score_delta"], 0.24)
        self.assertAlmostEqual(momentum["short_score_delta"], -0.26)
        self.assertAlmostEqual(momentum["score_gap_delta"], 0.50)
        self.assertAlmostEqual(momentum["score_gap_slope_per_hour"], 0.50)

    def test_ignores_stale_previous_snapshot(self) -> None:
        charts = [
            {
                "pair": "EUR_USD",
                "long_score": 0.29,
                "short_score": 0.69,
                "confluence": {"score_gap": -0.40},
            }
        ]
        previous = {
            "generated_at_utc": "2026-05-15T00:00:00+00:00",
            "charts": [
                {
                    "pair": "EUR_USD",
                    "long_score": 0.05,
                    "short_score": 0.95,
                    "confluence": {"score_gap": -0.90},
                }
            ],
        }

        attach_score_momentum(charts, previous, "2026-05-15T11:00:00+00:00")

        self.assertNotIn("score_momentum", charts[0]["confluence"])


if __name__ == "__main__":
    unittest.main()

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
        self.assertEqual(momentum["baseline_lineage"], "PREVIOUS_PACKET")

    def test_same_cycle_reanchor_carries_prior_cycle_baseline(self) -> None:
        charts = [
            {
                "pair": "EUR_USD",
                "long_score": 0.35,
                "short_score": 0.63,
                "confluence": {"score_gap": -0.28},
            }
        ]
        previous = {
            "generated_at_utc": "2026-05-15T10:00:00+00:00",
            "charts": [
                {
                    "pair": "EUR_USD",
                    "long_score": 0.29,
                    "short_score": 0.69,
                    "confluence": {
                        "score_gap": -0.40,
                        "score_momentum": {
                            "baseline_lineage": "PREVIOUS_PACKET",
                            "previous_generated_at_utc": "2026-05-15T09:00:00+00:00",
                            "current_generated_at_utc": "2026-05-15T10:00:00+00:00",
                            "previous_score_gap": -0.90,
                            "current_score_gap": -0.40,
                            "score_gap_delta": 0.50,
                            "elapsed_min": 60.0,
                            "long_score_delta": 0.24,
                            "short_score_delta": -0.26,
                        },
                    },
                }
            ],
        }

        attach_score_momentum(charts, previous, "2026-05-15T10:05:00+00:00")

        momentum = charts[0]["confluence"]["score_momentum"]
        self.assertEqual(momentum["baseline_lineage"], "PRIOR_CYCLE_CARRIED")
        self.assertEqual(momentum["previous_generated_at_utc"], "2026-05-15T09:00:00+00:00")
        self.assertEqual(momentum["elapsed_min"], 65.0)
        self.assertAlmostEqual(momentum["long_score_delta"], 0.30)
        self.assertAlmostEqual(momentum["short_score_delta"], -0.32)
        self.assertAlmostEqual(momentum["score_gap_delta"], 0.62)
        self.assertAlmostEqual(momentum["score_gap_slope_per_hour"], 0.5723)

    def test_cycle_identity_carries_baseline_even_after_thirty_minutes(self) -> None:
        charts = [
            {
                "pair": "EUR_USD",
                "long_score": 0.35,
                "short_score": 0.63,
                "confluence": {"score_gap": -0.28},
            }
        ]
        previous = {
            "cycle_id": "cycle-a",
            "generated_at_utc": "2026-05-15T10:00:00+00:00",
            "charts": [
                {
                    "pair": "EUR_USD",
                    "long_score": 0.29,
                    "short_score": 0.69,
                    "confluence": {
                        "score_gap": -0.40,
                        "score_momentum": {
                            "baseline_lineage": "PREVIOUS_PACKET",
                            "previous_generated_at_utc": "2026-05-15T09:00:00+00:00",
                            "current_generated_at_utc": "2026-05-15T10:00:00+00:00",
                            "previous_score_gap": -0.90,
                            "current_score_gap": -0.40,
                            "score_gap_delta": 0.50,
                            "elapsed_min": 60.0,
                            "long_score_delta": 0.24,
                            "short_score_delta": -0.26,
                        },
                    },
                }
            ],
        }

        attach_score_momentum(
            charts,
            previous,
            "2026-05-15T10:45:00+00:00",
            cycle_id="cycle-a",
        )

        momentum = charts[0]["confluence"]["score_momentum"]
        self.assertEqual(momentum["baseline_lineage"], "PRIOR_CYCLE_CARRIED")
        self.assertEqual(momentum["previous_generated_at_utc"], "2026-05-15T09:00:00+00:00")
        self.assertEqual(momentum["elapsed_min"], 105.0)

    def test_distinct_cycle_identity_does_not_merge_short_interval_packets(self) -> None:
        charts = [
            {
                "pair": "EUR_USD",
                "long_score": 0.35,
                "short_score": 0.63,
                "confluence": {"score_gap": -0.28},
            }
        ]
        previous = {
            "cycle_id": "cycle-a",
            "generated_at_utc": "2026-05-15T10:00:00+00:00",
            "charts": [
                {
                    "pair": "EUR_USD",
                    "long_score": 0.29,
                    "short_score": 0.69,
                    "confluence": {"score_gap": -0.40},
                }
            ],
        }

        attach_score_momentum(
            charts,
            previous,
            "2026-05-15T10:05:00+00:00",
            cycle_id="cycle-b",
        )

        momentum = charts[0]["confluence"]["score_momentum"]
        self.assertEqual(momentum["baseline_lineage"], "PREVIOUS_PACKET")
        self.assertEqual(momentum["previous_generated_at_utc"], "2026-05-15T10:00:00+00:00")
        self.assertEqual(momentum["elapsed_min"], 5.0)

    def test_one_sided_cycle_identity_does_not_invent_momentum(self) -> None:
        for previous_cycle_id, current_cycle_id in (
            ("cycle-a", None),
            (None, "cycle-b"),
        ):
            with self.subTest(
                previous_cycle_id=previous_cycle_id,
                current_cycle_id=current_cycle_id,
            ):
                charts = [
                    {
                        "pair": "EUR_USD",
                        "long_score": 0.35,
                        "short_score": 0.63,
                        "confluence": {"score_gap": -0.28},
                    }
                ]
                previous = {
                    "generated_at_utc": "2026-05-15T10:00:00+00:00",
                    "charts": [
                        {
                            "pair": "EUR_USD",
                            "long_score": 0.29,
                            "short_score": 0.69,
                            "confluence": {"score_gap": -0.40},
                        }
                    ],
                }
                if previous_cycle_id is not None:
                    previous["cycle_id"] = previous_cycle_id

                attach_score_momentum(
                    charts,
                    previous,
                    "2026-05-15T10:05:00+00:00",
                    cycle_id=current_cycle_id,
                )

                self.assertNotIn("score_momentum", charts[0]["confluence"])

    def test_unbound_wrapper_cycle_identity_does_not_invent_momentum(self) -> None:
        charts = [
            {
                "pair": "EUR_USD",
                "long_score": 0.35,
                "short_score": 0.63,
                "confluence": {"score_gap": -0.28},
            }
        ]
        previous = {
            "cycle_id": "cycle-a",
            "generated_at_utc": "2026-05-15T10:00:00+00:00",
            "charts": [
                {
                    "pair": "EUR_USD",
                    "long_score": 0.29,
                    "short_score": 0.69,
                    "confluence": {"score_gap": -0.40},
                }
            ],
        }

        attach_score_momentum(
            charts,
            previous,
            "2026-05-15T10:05:00+00:00",
            cycle_id="fallback-wrapper-cycle",
            cycle_lineage_status="UNBOUND_WRAPPER",
        )

        self.assertNotIn("score_momentum", charts[0]["confluence"])

    def test_same_cycle_reanchor_without_lineage_does_not_invent_momentum(self) -> None:
        charts = [
            {
                "pair": "EUR_USD",
                "long_score": 0.35,
                "short_score": 0.63,
                "confluence": {"score_gap": -0.28},
            }
        ]
        previous = {
            "generated_at_utc": "2026-05-15T10:00:00+00:00",
            "charts": [
                {
                    "pair": "EUR_USD",
                    "long_score": 0.29,
                    "short_score": 0.69,
                    "confluence": {"score_gap": -0.40},
                }
            ],
        }

        attach_score_momentum(charts, previous, "2026-05-15T10:05:00+00:00")

        self.assertNotIn("score_momentum", charts[0]["confluence"])

    def test_same_cycle_reanchor_rejects_inconsistent_embedded_baseline(self) -> None:
        charts = [
            {
                "pair": "EUR_USD",
                "long_score": 0.35,
                "short_score": 0.63,
                "confluence": {"score_gap": -0.28},
            }
        ]
        previous = {
            "generated_at_utc": "2026-05-15T10:00:00+00:00",
            "charts": [
                {
                    "pair": "EUR_USD",
                    "long_score": 0.29,
                    "short_score": 0.69,
                    "confluence": {
                        "score_gap": -0.40,
                        "score_momentum": {
                            "baseline_lineage": "PREVIOUS_PACKET",
                            "previous_generated_at_utc": "2026-05-15T09:00:00+00:00",
                            "current_generated_at_utc": "2026-05-15T10:00:00+00:00",
                            "previous_score_gap": 0.90,
                            "current_score_gap": -0.40,
                            "score_gap_delta": -1.30,
                            "elapsed_min": 60.0,
                            "long_score_delta": 0.24,
                            "short_score_delta": -0.26,
                        },
                    },
                }
            ],
        }

        attach_score_momentum(charts, previous, "2026-05-15T10:05:00+00:00")

        self.assertNotIn("score_momentum", charts[0]["confluence"])

    def test_same_cycle_reanchor_rejects_unverified_embedded_lineage(self) -> None:
        base_momentum = {
            "baseline_lineage": "PREVIOUS_PACKET",
            "previous_generated_at_utc": "2026-05-15T09:00:00+00:00",
            "current_generated_at_utc": "2026-05-15T10:00:00+00:00",
            "previous_score_gap": -0.90,
            "current_score_gap": -0.40,
            "score_gap_delta": 0.50,
            "elapsed_min": 60.0,
            "long_score_delta": 0.24,
            "short_score_delta": -0.26,
        }
        invalid_fields = {
            "baseline_lineage": "UNKNOWN",
            "current_generated_at_utc": "2026-05-15T09:59:00+00:00",
            "score_gap_delta": float("nan"),
        }

        for field, invalid_value in invalid_fields.items():
            with self.subTest(field=field):
                charts = [
                    {
                        "pair": "EUR_USD",
                        "long_score": 0.35,
                        "short_score": 0.63,
                        "confluence": {"score_gap": -0.28},
                    }
                ]
                embedded = {**base_momentum, field: invalid_value}
                previous = {
                    "generated_at_utc": "2026-05-15T10:00:00+00:00",
                    "charts": [
                        {
                            "pair": "EUR_USD",
                            "long_score": 0.29,
                            "short_score": 0.69,
                            "confluence": {
                                "score_gap": -0.40,
                                "score_momentum": embedded,
                            },
                        }
                    ],
                }

                attach_score_momentum(charts, previous, "2026-05-15T10:05:00+00:00")

                self.assertNotIn("score_momentum", charts[0]["confluence"])

    def test_rejects_naive_evidence_clock_without_raising(self) -> None:
        charts = [
            {
                "pair": "EUR_USD",
                "long_score": 0.35,
                "short_score": 0.63,
                "confluence": {"score_gap": -0.28},
            }
        ]
        previous = {
            "generated_at_utc": "2026-05-15T10:00:00",
            "charts": [
                {
                    "pair": "EUR_USD",
                    "long_score": 0.29,
                    "short_score": 0.69,
                    "confluence": {"score_gap": -0.40},
                }
            ],
        }

        attach_score_momentum(charts, previous, "2026-05-15T11:00:00+00:00")

        self.assertNotIn("score_momentum", charts[0]["confluence"])

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

from __future__ import annotations

import unittest

from quant_rabbit.fast_bot_profit_candidate_audit import (
    candidate_admission_checks,
    candidate_metrics,
)


def _signal(name: str, day: str) -> dict:
    return {
        "signal_sha256": name,
        "generated_at_utc": f"{day}T00:00:00+00:00",
    }


def _outcome(name: str, pips: float, *, filled: bool = True) -> dict:
    return {
        "signal_sha256": name,
        "entry_experiment": {
            "arms": [
                {
                    "arm_id": "PASSIVE_NEAR_SIDE",
                    "filled": filled,
                    "realized_pips": pips,
                }
            ]
        },
    }


class FastBotProfitCandidateAuditTests(unittest.TestCase):
    def test_candidate_metrics_use_distinct_active_days_and_spread_truth(self) -> None:
        signals = [
            _signal("a", "2026-08-28"),
            _signal("b", "2026-08-31"),
            _signal("c", "2026-09-01"),
            _signal("d", "2026-09-01"),
        ]
        outcomes = {
            "a": _outcome("a", 3.0),
            "b": _outcome("b", -1.0),
            "c": _outcome("c", 3.0),
            "d": _outcome("d", 0.0, filled=False),
        }

        metrics = candidate_metrics(
            signals,
            outcomes,
            arm_id="PASSIVE_NEAR_SIDE",
        )

        self.assertEqual(metrics["resolved_signals"], 4)
        self.assertEqual(metrics["filled_signals"], 3)
        self.assertEqual(metrics["active_days"], 3)
        self.assertEqual(metrics["net_pips"], 5.0)
        self.assertEqual(metrics["profit_factor"], 6.0)
        self.assertEqual(metrics["positive_day_rate"], 0.666667)
        self.assertTrue(metrics["spread_included"])

    def test_three_sample_one_day_slice_cannot_be_admitted(self) -> None:
        checks = candidate_admission_checks(
            {
                "filled_signals": 3,
                "active_days": 1,
                "profit_factor": 1.478261,
                "pessimistic_expectancy_pips": -1.0,
                "positive_day_rate": 1.0,
                "maximum_daily_sample_share": 1.0,
            }
        )

        self.assertFalse(checks["minimum_samples"])
        self.assertFalse(checks["minimum_active_days"])
        self.assertFalse(checks["minimum_pessimistic_expectancy_pips"])
        self.assertFalse(checks["maximum_daily_sample_share"])
        self.assertFalse(all(checks.values()))


if __name__ == "__main__":
    unittest.main()

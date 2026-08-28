from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tools.refresh_oanda_spread_calibration import (
    _last_complete_business_days,
    _nearest_rank,
    _pair_calibration,
    _parse_oanda_time,
)


class RefreshOandaSpreadCalibrationTest(unittest.TestCase):
    def test_pre_session_end_uses_six_prior_complete_business_days(self) -> None:
        days = _last_complete_business_days(
            datetime(2026, 8, 28, 10, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(
            [item.isoformat() for item in days],
            [
                "2026-08-20",
                "2026-08-21",
                "2026-08-24",
                "2026-08-25",
                "2026-08-26",
                "2026-08-27",
            ],
        )

    def test_oanda_nanosecond_timestamp_is_parsed_without_changing_time(self) -> None:
        parsed = _parse_oanda_time("2026-08-27T14:55:00.000000000Z")
        self.assertEqual(parsed, datetime(2026, 8, 27, 14, 55, tzinfo=timezone.utc))

    def test_nearest_rank_and_baseline_are_deterministic(self) -> None:
        self.assertEqual(_nearest_rank([1.0, 3.0, 2.0, 4.0], 0.95), 4.0)
        samples = [
            {"endpoint_spread_pips": value}
            for value in ([1.0] * 205 + [2.0] * 11)
        ]
        calibration = _pair_calibration("EUR_USD", samples)
        self.assertEqual(calibration["sample_count"], 216)
        self.assertEqual(calibration["p95_pips"], 2.0)
        self.assertEqual(calibration["recommended_baseline_pips"], 0.8)


if __name__ == "__main__":
    unittest.main()

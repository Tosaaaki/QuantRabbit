from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "build_audjpy_limit_fresh_s5_bidask_replay.py"
    spec = importlib.util.spec_from_file_location("build_audjpy_limit_fresh_s5_bidask_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class AudJpyLimitFreshS5BidaskReplayTest(unittest.TestCase):
    def test_thresholds_require_daily_stability_and_positive_day_rate(self) -> None:
        stats = {
            "sample_count": 135,
            "active_days": 6,
            "max_daily_sample_share": 0.9111,
            "positive_day_rate": 0.3333,
            "expectancy_pips": 0.861481,
        }

        thresholds = builder._thresholds(stats)

        self.assertTrue(thresholds["sample_count_floor"])
        self.assertTrue(thresholds["active_day_floor"])
        self.assertFalse(thresholds["daily_stability_floor"])
        self.assertFalse(thresholds["positive_day_rate_floor"])
        self.assertTrue(thresholds["spread_included_expectancy_positive"])
        self.assertEqual(builder._classification(stats, thresholds), "REPAIR_REQUIRED")

    def test_zero_sample_shape_is_evidence_gap(self) -> None:
        stats = {
            "sample_count": 0,
            "active_days": 0,
            "max_daily_sample_share": None,
            "positive_day_rate": None,
            "expectancy_pips": None,
        }

        thresholds = builder._thresholds(stats)

        self.assertFalse(thresholds["sample_count_floor"])
        self.assertFalse(thresholds["active_day_floor"])
        self.assertFalse(thresholds["daily_stability_floor"])
        self.assertFalse(thresholds["positive_day_rate_floor"])
        self.assertFalse(thresholds["spread_included_expectancy_positive"])
        self.assertEqual(builder._classification(stats, thresholds), "EVIDENCE_GAP")

    def test_wilson_lower_bound_is_conservative(self) -> None:
        lower = builder._wilson_lower_bound(67, 135)

        self.assertIsNotNone(lower)
        assert lower is not None
        self.assertLess(lower, 67 / 135)
        self.assertGreater(lower, 0.40)


if __name__ == "__main__":
    unittest.main()

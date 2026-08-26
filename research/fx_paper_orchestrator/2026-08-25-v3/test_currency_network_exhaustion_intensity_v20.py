import unittest

from run_currency_network_exhaustion_intensity_v20 import tuning_median_positive


class CurrencyNetworkExhaustionIntensityTest(unittest.TestCase):
    def test_threshold_uses_only_positive_tuning_values(self):
        rows = [
            {"fill_time": "2026-03-12T00:00:00Z", "network_alignment": -9.0},
            {"fill_time": "2026-03-13T00:00:00Z", "network_alignment": 0.2},
            {"fill_time": "2026-04-01T00:00:00Z", "network_alignment": 0.8},
            {"fill_time": "2026-05-05T00:00:00Z", "network_alignment": 99.0},
        ]
        self.assertEqual(tuning_median_positive(rows), (0.5, 2))

    def test_threshold_fails_closed_without_positive_tuning_values(self):
        with self.assertRaisesRegex(ValueError, "no positive tuning"):
            tuning_median_positive([
                {"fill_time": "2026-04-01T00:00:00Z", "network_alignment": 0.0},
                {"fill_time": "2026-05-05T00:00:00Z", "network_alignment": 1.0},
            ])


if __name__ == "__main__":
    unittest.main()

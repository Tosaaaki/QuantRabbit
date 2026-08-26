import unittest

from run_liquid_major_universe_v9 import UNIVERSE, summarize


class LiquidMajorUniverseTest(unittest.TestCase):
    def test_universe_is_exactly_predefined_seven(self):
        self.assertEqual(UNIVERSE, {
            "AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY"
        })

    def test_summary_never_drops_signal_by_cost_arm(self):
        row = {
            "fill_time": "2026-05-01T00:00:00Z",
            "exit_time": "2026-05-02T00:00:00Z",
            "scores": {
                "RAW_SIGNAL": {"net_return": 0.001},
                "EXECUTABLE_BASE": {"net_return": -0.001},
                "ADVERSE_STRESS": {"net_return": -0.002},
            },
        }
        result = summarize([row], "2026-05-01", "2026-06-01")
        self.assertEqual({value["signals"] for value in result["arms"].values()}, {1})

    def test_period_requires_exit_before_end_boundary(self):
        row = {
            "fill_time": "2026-05-31T23:55:00Z",
            "exit_time": "2026-06-01T00:05:00Z",
            "scores": {arm: {"net_return": 0.001} for arm in (
                "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"
            )},
        }
        result = summarize([row], "2026-05-01", "2026-06-01")
        self.assertEqual(result["arms"]["RAW_SIGNAL"]["signals"], 0)


if __name__ == "__main__":
    unittest.main()

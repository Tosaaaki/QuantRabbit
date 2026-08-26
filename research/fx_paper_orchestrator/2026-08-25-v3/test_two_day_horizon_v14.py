import unittest
from datetime import datetime, timedelta, timezone

from causal_composite_indicators_v3 import Bar
from run_two_day_horizon_v14 import HORIZON, rescore


class TwoDayHorizonTest(unittest.TestCase):
    def fixture(self):
        rows = []
        for index in range(HORIZON + 1):
            mid = 1.0 + index * .00001
            stamp = (datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z")
            rows.append(Bar("EUR_USD", stamp, mid-.0001, mid, mid-.0002, mid-.0001,
                            mid+.0001, mid+.0002, mid, mid+.0001, 100))
        return rows

    def test_finite_horizon_and_cost_order(self):
        rows = self.fixture()
        values = [rescore(rows, 0, 1, arm)["net_return"] for arm in (
            "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"
        )]
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])

    def test_missing_exit_fails_closed(self):
        self.assertIsNone(rescore(self.fixture()[:-1], 0, -1, "RAW_SIGNAL"))


if __name__ == "__main__":
    unittest.main()

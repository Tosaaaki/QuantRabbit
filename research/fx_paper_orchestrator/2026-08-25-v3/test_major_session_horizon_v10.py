import unittest
from datetime import datetime, timedelta, timezone

from causal_composite_indicators_v3 import Bar
from run_major_session_horizon_v10 import HORIZON, rescore


def fixture():
    rows = []
    for index in range(HORIZON + 1):
        mid = 1.0 + index * .00001
        stamp = (datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z")
        rows.append(Bar("EUR_USD", stamp, mid-.0001, mid, mid-.0002, mid-.0001,
                        mid+.0001, mid+.0002, mid, mid+.0001, 100))
    return rows


class MajorSessionHorizonTest(unittest.TestCase):
    def test_finite_192_bar_horizon_and_cost_order(self):
        rows = fixture()
        raw = rescore(rows, 0, 1, "RAW_SIGNAL")
        base = rescore(rows, 0, 1, "EXECUTABLE_BASE")
        adverse = rescore(rows, 0, 1, "ADVERSE_STRESS")
        self.assertGreater(raw["net_return"], base["net_return"])
        self.assertGreater(base["net_return"], adverse["net_return"])

    def test_missing_exit_fails_closed(self):
        self.assertIsNone(rescore(fixture()[:-1], 0, 1, "RAW_SIGNAL"))


if __name__ == "__main__":
    unittest.main()

import unittest
from datetime import datetime, timedelta, timezone

from causal_composite_indicators_v3 import Bar
from run_auction_trap_horizon_v8 import HORIZON, rescore


def bars():
    output = []
    for index in range(HORIZON + 1):
        mid = 1.0 + index * 0.00001
        stamp = (datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z")
        output.append(Bar("EUR_USD", stamp, mid - .0001, mid, mid - .0002, mid - .0001,
                          mid + .0001, mid + .0002, mid, mid + .0001, 100))
    return output


class AuctionTrapHorizonTest(unittest.TestCase):
    def test_exact_fixed_horizon_and_cost_ordering(self):
        data = bars()
        raw = rescore(data, 0, 1, "RAW_SIGNAL")
        base = rescore(data, 0, 1, "EXECUTABLE_BASE")
        adverse = rescore(data, 0, 1, "ADVERSE_STRESS")
        self.assertIsNotNone(raw)
        self.assertGreater(raw["net_return"], base["net_return"])
        self.assertGreater(base["net_return"], adverse["net_return"])

    def test_missing_fixed_exit_is_not_fabricated(self):
        self.assertIsNone(rescore(bars()[:-1], 0, -1, "RAW_SIGNAL"))


if __name__ == "__main__":
    unittest.main()

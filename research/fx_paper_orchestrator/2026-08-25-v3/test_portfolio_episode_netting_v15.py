import unittest
from datetime import datetime, timedelta, timezone

from causal_composite_indicators_v3 import Bar
from run_portfolio_episode_netting_v15 import roundtrip_return, simulate_pair


def bars():
    rows = []
    for index in range(8):
        mid = 1.0 + index * .0001
        stamp = (datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z")
        rows.append(Bar("EUR_USD", stamp, mid-.0001, mid, mid-.0002, mid-.0001,
                        mid+.0001, mid+.0002, mid, mid+.0001, 100))
    return rows


class PortfolioEpisodeNettingTest(unittest.TestCase):
    def test_same_direction_does_not_add_or_extend(self):
        data = bars()
        signals = [
            {"pair": "EUR_USD", "fill_time": data[1].time, "exit_time": data[5].time, "direction": 1},
            {"pair": "EUR_USD", "fill_time": data[2].time, "exit_time": data[6].time, "direction": 1},
        ]
        _, audit = simulate_pair("EUR_USD", data, signals, "RAW_SIGNAL", "2026-05-01", "2026-05-02")
        self.assertEqual(audit["opens"], 1)
        self.assertEqual(audit["ignored_same_direction"], 1)
        self.assertEqual(audit["terminal_open_inventory"], 0)

    def test_opposite_signal_nets_before_reopening(self):
        data = bars()
        signals = [
            {"pair": "EUR_USD", "fill_time": data[1].time, "exit_time": data[5].time, "direction": 1},
            {"pair": "EUR_USD", "fill_time": data[2].time, "exit_time": data[6].time, "direction": -1},
        ]
        _, audit = simulate_pair("EUR_USD", data, signals, "RAW_SIGNAL", "2026-05-01", "2026-05-02")
        self.assertEqual(audit["reversals"], 1)
        self.assertEqual(audit["opens"], 2)
        self.assertEqual(audit["closes"], 2)

    def test_cost_ordering(self):
        data = bars()
        values = [roundtrip_return(data[0], data[-1], 1, arm, False) for arm in (
            "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"
        )]
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])


if __name__ == "__main__":
    unittest.main()

import unittest
from datetime import datetime, timedelta, timezone

from causal_composite_indicators_v3 import Bar
from run_opposite_close_only_v16 import simulate_pair


def bars():
    rows = []
    for index in range(8):
        mid = 1.0 + index * .0001
        stamp = (datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z")
        rows.append(Bar("EUR_USD", stamp, mid-.0001, mid, mid-.0002, mid-.0001,
                        mid+.0001, mid+.0002, mid, mid+.0001, 100))
    return rows


class OppositeCloseOnlyTest(unittest.TestCase):
    def test_opposite_signal_closes_without_same_signal_reopen(self):
        data = bars()
        signals = [
            {"pair": "EUR_USD", "fill_time": data[1].time, "exit_time": data[5].time, "direction": 1},
            {"pair": "EUR_USD", "fill_time": data[2].time, "exit_time": data[6].time, "direction": -1},
        ]
        _, audit = simulate_pair("EUR_USD", data, signals, "RAW_SIGNAL", "2026-05-01", "2026-05-02")
        self.assertEqual(audit["opens"], 1)
        self.assertEqual(audit["closes"], 1)
        self.assertEqual(audit["opposite_close_only"], 1)
        self.assertEqual(audit["terminal_open_inventory"], 0)


if __name__ == "__main__":
    unittest.main()

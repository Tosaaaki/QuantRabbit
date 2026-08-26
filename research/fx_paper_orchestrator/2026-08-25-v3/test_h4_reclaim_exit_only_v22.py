import unittest
from datetime import datetime, timedelta, timezone

from causal_composite_indicators_v3 import Bar
from run_h4_reclaim_exit_only_v22 import simulate_pair


def bars():
    rows = []
    for index in range(8):
        mid = 1.0 + index * 0.0001
        stamp = (datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z")
        rows.append(Bar(
            "EUR_USD", stamp,
            mid - 0.0001, mid, mid - 0.0002, mid - 0.0001,
            mid + 0.0001, mid + 0.0002, mid, mid + 0.0001, 100,
        ))
    return rows


class H4ReclaimExitOnlyTest(unittest.TestCase):
    def test_auxiliary_opposite_closes_without_reopening(self):
        data = bars()
        primary = [{
            "pair": "EUR_USD", "fill_time": data[1].time, "exit_time": data[6].time, "direction": -1,
        }]
        auxiliary = [{"pair": "EUR_USD", "fill_time": data[3].time, "direction": 1}]
        _, audit = simulate_pair(
            "EUR_USD", data, primary, auxiliary, "RAW_SIGNAL", "2026-05-01", "2026-05-02"
        )
        self.assertEqual(audit["opens"], 1)
        self.assertEqual(audit["auxiliary_exit_only"], 1)
        self.assertEqual(audit["closes"], 1)

    def test_auxiliary_cannot_open_when_flat(self):
        data = bars()
        auxiliary = [{"pair": "EUR_USD", "fill_time": data[3].time, "direction": 1}]
        _, audit = simulate_pair(
            "EUR_USD", data, [], auxiliary, "RAW_SIGNAL", "2026-05-01", "2026-05-02"
        )
        self.assertEqual(audit["opens"], 0)
        self.assertEqual(audit["auxiliary_flat_ignored"], 1)

    def test_same_direction_auxiliary_does_not_extend_expiry(self):
        data = bars()
        primary = [{
            "pair": "EUR_USD", "fill_time": data[1].time, "exit_time": data[4].time, "direction": 1,
        }]
        auxiliary = [{"pair": "EUR_USD", "fill_time": data[3].time, "direction": 1}]
        _, audit = simulate_pair(
            "EUR_USD", data, primary, auxiliary, "RAW_SIGNAL", "2026-05-01", "2026-05-02"
        )
        self.assertEqual(audit["auxiliary_same_direction_ignored"], 1)
        self.assertEqual(audit["closes"], 1)


if __name__ == "__main__":
    unittest.main()

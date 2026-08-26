import unittest
from datetime import datetime, timedelta, timezone

from causal_composite_indicators_v3 import Bar
from run_h4_countertrend_sweep_reclaim_v21 import HORIZON, signal_at


def bar(index, close, high=None, low=None, spread=0.0002):
    high = close + 0.0002 if high is None else high
    low = close - 0.0002 if low is None else low
    stamp = (datetime(2026, 3, 11, tzinfo=timezone.utc) + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z")
    return Bar(
        "EUR_USD", stamp,
        close - spread / 2, high - spread / 2, low - spread / 2, close - spread / 2,
        close + spread / 2, high + spread / 2, low + spread / 2, close + spread / 2, 100,
    )


class H4CountertrendSweepReclaimTest(unittest.TestCase):
    def fixture(self):
        bars = [bar(i, 1.0000 + i * 0.00005) for i in range(53 + HORIZON + 2)]
        bars[48] = bar(48, 1.00240)
        bars[49] = bar(49, 1.00220)
        bars[50] = bar(50, 1.00200)
        rail_low = min(item.mid_l for item in bars[39:51])
        bars[51] = bar(51, 1.00180, low=rail_low - 0.0003)
        bars[52] = bar(52, rail_low + 0.0004)
        return bars, 52

    def test_completed_h4_m15_m5_reclaim_generates_long(self):
        bars, index = self.fixture()
        signal = signal_at(bars, index)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["direction"], 1)

    def test_future_bars_cannot_change_signal(self):
        bars, index = self.fixture()
        before = signal_at(bars, index)
        bars[index + 10] = bar(index + 10, 9.0)
        self.assertEqual(before, signal_at(bars, index))

    def test_fill_bar_is_not_read_by_signal(self):
        bars, index = self.fixture()
        before = signal_at(bars, index)
        bars[index + 1] = bar(index + 1, 0.1)
        self.assertEqual(before, signal_at(bars, index))


if __name__ == "__main__":
    unittest.main()

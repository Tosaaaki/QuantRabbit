import unittest
from datetime import datetime, timedelta, timezone

from causal_composite_indicators_v3 import Bar
from run_asian_box_sweep_reclaim_v24 import detect_day_signal, raw_path_metrics, roundtrip_return


def bar(index, close=1.0000, high=1.0001, low=0.9999, spread=0.0002, stamp=None):
    stamp = stamp or (
        datetime(2026, 3, 11, tzinfo=timezone.utc) + timedelta(minutes=5 * index)
    ).isoformat().replace("+00:00", "Z")
    return Bar(
        "EUR_USD", stamp,
        close - spread / 2, high - spread / 2, low - spread / 2, close - spread / 2,
        close + spread / 2, high + spread / 2, low + spread / 2, close + spread / 2, 100,
    )


class AsianBoxSweepReclaimTest(unittest.TestCase):
    def fixture(self):
        return [bar(i) for i in range(192)]

    def test_lower_sweep_reclaim_generates_long_with_next_fill_and_fixed_exit(self):
        bars = self.fixture()
        bars[72] = bar(72, close=1.0000, high=1.00005, low=0.9997)
        signal = detect_day_signal("EUR_USD", bars)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["direction"], 1)
        self.assertEqual(signal["decision_time"], bars[72].time)
        self.assertEqual(signal["fill_time"], bars[73].time)
        self.assertEqual(signal["exit_time"], bars[191].time)

    def test_upper_sweep_reclaim_generates_short(self):
        bars = self.fixture()
        bars[72] = bar(72, close=1.0000, high=1.0003, low=0.99995)
        signal = detect_day_signal("EUR_USD", bars)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["direction"], -1)

    def test_ambiguous_two_sided_sweep_is_rejected(self):
        bars = self.fixture()
        bars[72] = bar(72, close=1.0000, high=1.0003, low=0.9997)
        self.assertIsNone(detect_day_signal("EUR_USD", bars))

    def test_first_unambiguous_signal_is_the_only_daily_signal(self):
        bars = self.fixture()
        bars[72] = bar(72, close=1.0000, high=1.00005, low=0.9997)
        bars[74] = bar(74, close=1.0000, high=1.0003, low=0.99995)
        signal = detect_day_signal("EUR_USD", bars)
        self.assertEqual(signal["decision_time"], bars[72].time)
        self.assertEqual(signal["direction"], 1)

    def test_future_and_fill_prices_cannot_change_decision(self):
        bars = self.fixture()
        bars[72] = bar(72, close=1.0000, high=1.00005, low=0.9997)
        before = detect_day_signal("EUR_USD", bars)
        bars[73] = bar(73, close=8.0, high=8.1, low=7.9)
        bars[191] = bar(191, close=0.2, high=0.3, low=0.1)
        after = detect_day_signal("EUR_USD", bars)
        self.assertEqual(before, after)

    def test_missing_or_duplicate_box_timestamp_fails_closed(self):
        missing = self.fixture()
        del missing[12]
        self.assertIsNone(detect_day_signal("EUR_USD", missing))
        duplicate = self.fixture()
        duplicate[12] = bar(12, stamp=duplicate[11].time)
        self.assertIsNone(detect_day_signal("EUR_USD", duplicate))

    def test_raw_short_return_uses_exact_inverse(self):
        entry = bar(73, close=1.0, high=1.01, low=0.99)
        exit_bar = bar(191, close=1.1, high=1.11, low=1.09)
        metrics = raw_path_metrics([entry, exit_bar], -1)
        self.assertAlmostEqual(metrics["gross_return"], 1.0 / 1.1 - 1.0)

    def test_source_nanosecond_utc_timestamps_are_accepted(self):
        bars = self.fixture()
        bars = [
            bar(index, stamp=item.time.replace("Z", ".000000000Z"))
            for index, item in enumerate(bars)
        ]
        bars[72] = bar(72, close=1.0000, high=1.00005, low=0.9997, stamp=bars[72].time)
        self.assertEqual(detect_day_signal("EUR_USD", bars)["direction"], 1)

    def test_cost_calculation_accepts_source_nanoseconds(self):
        entry = bar(73, stamp="2026-03-11T06:05:00.000000000Z")
        exit_bar = bar(191, close=1.001, high=1.0011, low=1.0009, stamp="2026-03-11T15:55:00.000000000Z")
        self.assertIsInstance(roundtrip_return(entry, exit_bar, 1, "ADVERSE_STRESS", False), float)


if __name__ == "__main__":
    unittest.main()

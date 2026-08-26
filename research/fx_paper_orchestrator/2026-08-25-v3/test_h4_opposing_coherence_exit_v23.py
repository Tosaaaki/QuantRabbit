import unittest
from datetime import datetime, timedelta, timezone

from causal_composite_indicators_v3 import Bar
from run_h4_opposing_coherence_exit_v23 import freeze_threshold, h4_candidate_at, nearest_rank


def bar(index, close, start=None):
    start = start or datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)
    stamp = (start + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z")
    return Bar(
        "EUR_USD", stamp,
        close - 0.0001, close, close - 0.0002, close - 0.0001,
        close + 0.0001, close + 0.0002, close, close + 0.0001, 100,
    )


class H4OpposingCoherenceExitTest(unittest.TestCase):
    def fixture(self):
        return [bar(index, 1.0 + index * 0.00001) for index in range(260)]

    def test_candidate_uses_two_completed_h4_blocks(self):
        bars = self.fixture()
        measured = h4_candidate_at(bars, 143)
        self.assertIsNotNone(measured)
        self.assertEqual(measured["direction"], 1)
        self.assertEqual(measured["fill_time"], bars[144].time)

    def test_future_and_fill_bars_cannot_change_candidate(self):
        bars = self.fixture()
        before = h4_candidate_at(bars, 143)
        bars[144] = bar(144, 0.1)
        bars[180] = bar(180, 9.0)
        self.assertEqual(before, h4_candidate_at(bars, 143))

    def test_noncontiguous_window_fails_closed(self):
        bars = self.fixture()
        bars[100] = bar(101, bars[100].mid_c)
        self.assertIsNone(h4_candidate_at(bars, 143))

    def test_q75_is_nearest_rank_and_tuning_only(self):
        candidates = [
            {"decision_time": "2026-03-12T00:00:00Z", "coherence": 1.0},
            {"decision_time": "2026-03-13T00:00:00Z", "coherence": 2.0},
            {"decision_time": "2026-04-01T00:00:00Z", "coherence": 3.0},
            {"decision_time": "2026-04-02T00:00:00Z", "coherence": 4.0},
            {"decision_time": "2026-05-02T00:00:00Z", "coherence": 999.0},
        ]
        self.assertEqual(nearest_rank([1.0, 2.0, 3.0, 4.0], 0.75), 3.0)
        self.assertEqual(freeze_threshold(candidates), (3.0, 4))


if __name__ == "__main__":
    unittest.main()

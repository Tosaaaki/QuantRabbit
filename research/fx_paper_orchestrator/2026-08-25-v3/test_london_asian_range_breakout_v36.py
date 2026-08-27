import unittest

from paper_replay_compat.fx_original_indicators import Bar
from run_london_asian_range_breakout_v36 import detect_day_signals


PAIRS = ("AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY")


def bars(pair: str, decision_close: float) -> list[Bar]:
    result = []
    for minute in range(0, 8 * 60, 5):
        hour, part = divmod(minute, 60)
        close = decision_close if minute == 7 * 60 + 55 else 1.0
        result.append(Bar(pair, f"2026-04-01T{hour:02d}:{part:02d}:00.000000000Z",
                          0.9999, 1.0009, 0.9990, close - .0001,
                          1.0001, 1.0011, 0.9992, close + .0001))
    return result


class LondonAsianRangeBreakoutV36Test(unittest.TestCase):
    def test_completed_close_above_asian_range_emits_long_with_next_bar_fill(self):
        rows = detect_day_signals({pair: bars(pair, 1.002) for pair in PAIRS})
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(row["direction"] == 1 for row in rows))
        self.assertTrue(all(row["decision_time"].endswith(".000000000Z") for row in rows))
        self.assertTrue(all(row["fill_time"].endswith("T08:00:00.000000000Z") for row in rows))

    def test_close_inside_asian_range_emits_nothing(self):
        self.assertEqual(detect_day_signals({pair: bars(pair, 1.0) for pair in PAIRS}), [])


if __name__ == "__main__":
    unittest.main()

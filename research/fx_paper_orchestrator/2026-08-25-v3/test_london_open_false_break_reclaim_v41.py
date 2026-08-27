import unittest

from paper_replay_compat.fx_original_indicators import Bar
from run_london_open_false_break_reclaim_v41 import TARGET_HOLD_SECONDS, detect_day_signals


PAIRS = ("AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY")


def bars(pair: str, sweep: str, decision_close: float = 1.0) -> list[Bar]:
    result = []
    for minute in range(0, 9 * 60, 5):
        hour, part = divmod(minute, 60)
        high, low = 1.0005, .9995
        if 6 * 60 <= minute <= 8 * 60 + 30:
            if sweep in {"upper", "both"} and minute == 6 * 60:
                high = 1.001
            if sweep in {"lower", "both"} and minute == 6 * 60 + 5:
                low = .999
        close = decision_close if minute == 8 * 60 + 55 else 1.0
        result.append(Bar(pair, f"2026-04-01T{hour:02d}:{part:02d}:00.000000000Z",
                          low - .00002, high - .00002, low - .00002, close - .00001,
                          low + .00002, high + .00002, low + .00002, close + .00001))
    return result


class LondonOpenFalseBreakReclaimV41Test(unittest.TestCase):
    def test_upper_only_sweep_and_completed_reclaim_emits_short(self):
        rows = detect_day_signals({pair: bars(pair, "upper") for pair in PAIRS})
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(row["direction"] == -1 for row in rows))
        self.assertTrue(all(row["decision_time"].endswith("T08:55:00.000000000Z") for row in rows))
        self.assertTrue(all(row["fill_time"].endswith("T09:00:00.000000000Z") for row in rows))
        self.assertTrue(all(row["exit_time"].endswith("T12:55:00.000000000Z") for row in rows))

    def test_lower_only_sweep_and_completed_reclaim_emits_long(self):
        rows = detect_day_signals({pair: bars(pair, "lower") for pair in PAIRS})
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(row["direction"] == 1 for row in rows))

    def test_both_sides_or_close_outside_range_fails_closed(self):
        self.assertEqual(detect_day_signals({pair: bars(pair, "both") for pair in PAIRS}), [])
        self.assertEqual(
            detect_day_signals({pair: bars(pair, "upper", 1.0006) for pair in PAIRS}), [],
        )

    def test_v39_v40_carry_duration_is_preserved(self):
        self.assertEqual(TARGET_HOLD_SECONDS, 14_100)


if __name__ == "__main__":
    unittest.main()

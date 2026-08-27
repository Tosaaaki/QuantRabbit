import unittest

from paper_replay_compat.fx_original_indicators import Bar
from run_london_overextension_fade_v38 import detect_day_signals


PAIRS = ("AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY")


def bars(pair: str, london_close: float) -> list[Bar]:
    result = []
    for minute in range(0, 12 * 60, 5):
        hour, part = divmod(minute, 60)
        close = london_close if minute == 11 * 60 + 55 else 1.0
        result.append(Bar(pair, f"2026-04-01T{hour:02d}:{part:02d}:00.000000000Z",
                          .99999, 1.0001, .9999, close - .00001,
                          1.00001, 1.00012, .99992, close + .00001))
    return result


class LondonOverextensionFadeV38Test(unittest.TestCase):
    def test_large_positive_london_displacement_emits_fade_short(self):
        rows = detect_day_signals({pair: bars(pair, 1.002) for pair in PAIRS})
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(row["direction"] == -1 for row in rows))
        self.assertTrue(all(row["fill_time"].endswith("T12:00:00.000000000Z") for row in rows))

    def test_small_displacement_below_asian_range_emits_nothing(self):
        self.assertEqual(detect_day_signals({pair: bars(pair, 1.00005) for pair in PAIRS}), [])


if __name__ == "__main__":
    unittest.main()

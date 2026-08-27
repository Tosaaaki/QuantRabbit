import unittest

from paper_replay_compat.fx_original_indicators import Bar
from run_london_fix_overextension_fade_v40 import TARGET_HOLD_SECONDS, detect_day_signals


PAIRS = ("AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY")


def bars(pair: str, decision_close: float) -> list[Bar]:
    result = []
    for minute in range(8 * 60, 16 * 60, 5):
        hour, part = divmod(minute, 60)
        close = decision_close if minute == 15 * 60 + 55 else 1.0
        result.append(Bar(pair, f"2026-04-01T{hour:02d}:{part:02d}:00.000000000Z",
                          .99999, 1.0001, .9999, close - .00001,
                          1.00001, 1.00012, .99992, close + .00001))
    return result


class LondonFixOverextensionFadeV40Test(unittest.TestCase):
    def test_large_positive_pre_fix_move_emits_fade_short(self):
        rows = detect_day_signals({pair: bars(pair, 1.002) for pair in PAIRS})
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(row["direction"] == -1 for row in rows))
        self.assertTrue(all(row["fill_time"].endswith("T16:00:00.000000000Z") for row in rows))

    def test_v39_carry_duration_is_preserved(self):
        self.assertEqual(TARGET_HOLD_SECONDS, 14_100)


if __name__ == "__main__":
    unittest.main()

import unittest

import test_london_asian_range_breakout_v36 as fixture
from run_london_asian_range_breakout_v37 import EVALUATION_END_EXCLUSIVE, detect_day_signals


class LondonAsianRangeBreakoutV37Test(unittest.TestCase):
    def test_fixed_evaluation_end_is_unchanged(self):
        self.assertEqual(EVALUATION_END_EXCLUSIVE, "2026-07-01")

    def test_evaluation_day_preserves_v36_signal_family(self):
        rows = detect_day_signals({pair: fixture.bars(pair, 1.002) for pair in fixture.PAIRS})
        self.assertEqual(len(rows), 7)

    def test_post_evaluation_day_is_ineligible_without_price_or_outcome_check(self):
        corpus = {pair: fixture.bars(pair, 1.002) for pair in fixture.PAIRS}
        shifted = {
            pair: [bar.__class__(bar.pair, bar.time.replace("2026-04-01", "2026-07-01"),
                                 bar.bid_o, bar.bid_h, bar.bid_l, bar.bid_c,
                                 bar.ask_o, bar.ask_h, bar.ask_l, bar.ask_c, bar.volume)
                   for bar in bars]
            for pair, bars in corpus.items()
        }
        self.assertEqual(detect_day_signals(shifted), [])


if __name__ == "__main__":
    unittest.main()

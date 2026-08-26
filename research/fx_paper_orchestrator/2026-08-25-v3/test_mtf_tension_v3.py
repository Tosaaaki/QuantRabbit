import unittest

import pandas as pd

from run_mtf_tension_v3 import selected_direction


class MtfTensionTest(unittest.TestCase):
    def test_pullback_is_mutually_exclusive_from_unanimous(self):
        unanimous = pd.Series({"M15_direction": 1, "H1_direction": 1, "H4_direction": 1})
        pullback = pd.Series({"M15_direction": -1, "H1_direction": 1, "H4_direction": 1})
        self.assertEqual(selected_direction(unanimous, "MTF_UNANIMOUS"), 1)
        self.assertIsNone(selected_direction(unanimous, "H4_H1_PULLBACK_CONTINUATION"))
        self.assertIsNone(selected_direction(pullback, "MTF_UNANIMOUS"))
        self.assertEqual(selected_direction(pullback, "H4_H1_PULLBACK_CONTINUATION"), 1)


if __name__ == "__main__":
    unittest.main()

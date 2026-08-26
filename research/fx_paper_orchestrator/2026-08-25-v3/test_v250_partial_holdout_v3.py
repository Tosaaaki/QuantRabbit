import unittest

from run_v250_partial_holdout_v3 import CANDIDATE, full_month_multiples


class FrozenHoldoutTest(unittest.TestCase):
    def test_candidate_identity_is_frozen(self):
        self.assertEqual(CANDIDATE, "M15_H8:ridge:P0.0")

    def test_partial_july_is_not_called_a_complete_month(self):
        values = {"2026-05": 1.1, "2026-06": .9, "2026-07": 9.0}
        self.assertEqual(full_month_multiples(values), {"2026-05": 1.1, "2026-06": .9})


if __name__ == "__main__":
    unittest.main()

import unittest

import run_london_overextension_carry_v39 as v39


class LondonOverextensionCarryV39Test(unittest.TestCase):
    def test_only_target_carry_is_shortened_to_raw_session_horizon(self):
        self.assertEqual(v39.TARGET_HOLD_SECONDS, 14_100)
        self.assertEqual(v39.HARD_MAX_AGE_SECONDS, 345_600)

    def test_parent_runner_is_frozen_v38(self):
        self.assertEqual(v39.frozen_v38.CYCLE_ID, "V38")


if __name__ == "__main__":
    unittest.main()

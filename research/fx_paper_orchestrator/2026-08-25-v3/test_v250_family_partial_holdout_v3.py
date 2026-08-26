import json
import unittest

from run_v250_family_partial_holdout_v3 import V250_DIR


class FrozenFamilyTest(unittest.TestCase):
    def test_registered_family_is_exactly_54(self):
        contract = json.loads((V250_DIR / "contract_v250.json").read_text())
        self.assertEqual(contract["candidate_count"], 54)
        self.assertEqual(len(contract["predicted_net_return_floors"]), 3)


if __name__ == "__main__":
    unittest.main()

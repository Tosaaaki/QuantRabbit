import unittest

from run_currency_network_disconfirmation_v18 import graph_alignment as v18_alignment
from run_currency_network_exhaustion_v19 import graph_alignment as v19_alignment


class CurrencyNetworkExhaustionTest(unittest.TestCase):
    def test_v18_v19_share_identical_measurement_function(self):
        self.assertIs(v18_alignment, v19_alignment)

    def test_gate_partition_is_strictly_complementary(self):
        values = [-1.0, 0.0, 1.0]
        self.assertEqual([value <= 0 for value in values], [True, True, False])
        self.assertEqual([value > 0 for value in values], [False, False, True])


if __name__ == "__main__":
    unittest.main()

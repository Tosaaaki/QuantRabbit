import unittest

from run_graph_inventory_netting_v3 import target_weights


class GraphInventoryTest(unittest.TestCase):
    def test_target_is_fixed_gross_and_sparse(self):
        values = {
            "AUD_USD": .01, "EUR_USD": .02, "GBP_USD": .03, "NZD_USD": .04,
            "USD_CAD": -.01, "USD_CHF": -.02, "USD_JPY": -.03,
            "AUD_CAD": .01, "AUD_CHF": .01, "AUD_JPY": .01,
            "EUR_CAD": .01, "EUR_CHF": .01, "EUR_JPY": .01,
            "GBP_CAD": .01, "GBP_CHF": .01, "GBP_JPY": .01,
            "NZD_CAD": .01, "NZD_CHF": .01, "NZD_JPY": .01,
            "AUD_EUR": .01, "AUD_GBP": .01, "AUD_NZD": .01,
            "EUR_GBP": .01, "EUR_NZD": .01, "GBP_NZD": .01,
            "CAD_CHF": .01, "CAD_JPY": .01, "CHF_JPY": .01,
        }
        target = target_weights(values, 4)
        self.assertEqual(len(target), 4)
        self.assertAlmostEqual(sum(abs(value) for value in target.values()), 1.0)


if __name__ == "__main__":
    unittest.main()

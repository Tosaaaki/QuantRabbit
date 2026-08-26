import unittest

from run_currency_network_disconfirmation_v18 import graph_alignment


class CurrencyNetworkDisconfirmationTest(unittest.TestCase):
    def test_target_pair_is_excluded(self):
        corpus = {pair: [] for pair in (
            "AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY"
        )}
        index = {pair: {} for pair in corpus}
        self.assertIsNone(graph_alignment("EUR_USD", "missing", -1, corpus, index))

    def test_gate_has_no_cost_parameter(self):
        self.assertEqual(graph_alignment.__code__.co_argcount, 5)


if __name__ == "__main__":
    unittest.main()

import unittest

from run_graph_residual_v3 import leave_one_out_consensus, worker_direction


class GraphResidualTest(unittest.TestCase):
    def test_target_pair_cannot_change_its_own_consensus(self):
        values = {
            "EUR_USD": .9,
            "EUR_GBP": .01, "EUR_JPY": .02, "EUR_CHF": .01,
            "GBP_USD": -.01, "JPY_USD": -.02, "CHF_USD": -.01,
        }
        first = leave_one_out_consensus(values, "EUR_USD")
        values["EUR_USD"] = -99
        second = leave_one_out_consensus(values, "EUR_USD")
        self.assertEqual(first, second)

    def test_workers_are_structurally_distinct(self):
        self.assertEqual(worker_direction("GRAPH_PROPAGATION", .01, .002), 1)
        self.assertEqual(worker_direction("GRAPH_RESIDUAL_REVERSION", .01, .002), -1)
        self.assertEqual(worker_direction("GRAPH_COHERENT_MOMENTUM", .01, .002), 1)
        self.assertIsNone(worker_direction("GRAPH_LAG_CATCHUP", .01, .002))
        self.assertEqual(worker_direction("GRAPH_LAG_CATCHUP", -.01, .002), 1)


if __name__ == "__main__":
    unittest.main()

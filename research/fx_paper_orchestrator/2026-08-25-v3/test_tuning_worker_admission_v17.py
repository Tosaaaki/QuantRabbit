import unittest

from run_tuning_worker_admission_v17 import select_workers


class TuningWorkerAdmissionTest(unittest.TestCase):
    def metrics(self, signals, raw, adverse):
        return {
            "RAW_SIGNAL": {"source_signals": signals, "sleeve_equity_multiple": raw},
            "EXECUTABLE_BASE": {"source_signals": signals, "sleeve_equity_multiple": raw},
            "ADVERSE_STRESS": {"source_signals": signals, "sleeve_equity_multiple": adverse},
        }

    def test_requires_density_and_raw_and_adverse(self):
        audit = {
            "EUR_USD": self.metrics(20, 1.01, 1.001),
            "GBP_USD": self.metrics(19, 1.02, 1.01),
            "USD_JPY": self.metrics(30, .99, 1.01),
        }
        self.assertEqual(select_workers(audit), ["EUR_USD"])


if __name__ == "__main__":
    unittest.main()

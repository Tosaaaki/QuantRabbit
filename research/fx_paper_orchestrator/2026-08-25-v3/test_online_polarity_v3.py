import unittest

from run_online_polarity_v3 import MIN_DAILY_CLUSTERS, polarity_state


def history(values):
    return [{
        "exit_time": f"2026-01-{i + 1:02d}T12:00:00+00:00",
        "returns": {"RAW_SIGNAL": value},
    } for i, value in enumerate(values)]


class OnlinePolarityTest(unittest.TestCase):
    def test_insufficient_matured_history_freezes(self):
        state = polarity_state(history([0.01] * (MIN_DAILY_CLUSTERS - 1)))
        self.assertEqual(state["mode"], "FREEZE")

    def test_strictly_positive_history_continues(self):
        state = polarity_state(history([0.01] * MIN_DAILY_CLUSTERS))
        self.assertEqual(state["mode"], "CONTINUE")

    def test_strictly_negative_history_inverts(self):
        state = polarity_state(history([-0.01] * MIN_DAILY_CLUSTERS))
        self.assertEqual(state["mode"], "INVERT")

    def test_uncertain_history_freezes(self):
        values = [0.01 if i % 2 else -0.01 for i in range(MIN_DAILY_CLUSTERS)]
        state = polarity_state(history(values))
        self.assertEqual(state["mode"], "FREEZE")


if __name__ == "__main__":
    unittest.main()

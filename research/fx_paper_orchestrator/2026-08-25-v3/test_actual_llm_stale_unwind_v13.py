import unittest
from datetime import datetime, timedelta, timezone

from causal_composite_indicators_v3 import Bar
from run_actual_llm_stale_unwind_v13 import score_policy


def fixture():
    rows = []
    for index in range(386):
        mid = 1.0 - max(0, index - 180) * .000001
        stamp = (datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z")
        rows.append(Bar("EUR_USD", stamp, mid-.0001, mid, mid-.0002, mid-.0001,
                        mid+.0001, mid+.0002, mid, mid+.0001, 100))
    return rows


class ActualLlmStaleUnwindTest(unittest.TestCase):
    def test_nonpositive_completed_impulse_unwinds_at_following_open(self):
        result = score_policy(fixture(), 0, 1, .01, "RAW_SIGNAL", 192, 12, 384)
        self.assertEqual(result["exit_reason"], "STALE_UNWIND")
        self.assertEqual(result["decision_age_m5_bars"], 192)
        self.assertEqual(result["exit_time"], fixture()[193].time)

    def test_cost_arms_share_exit_and_ordering(self):
        rows = fixture()
        results = [score_policy(rows, 0, 1, .01, arm, 192, 12, 384) for arm in (
            "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"
        )]
        self.assertEqual(len({row["exit_time"] for row in results}), 1)
        self.assertGreater(results[0]["net_return"], results[1]["net_return"])
        self.assertGreater(results[1]["net_return"], results[2]["net_return"])


if __name__ == "__main__":
    unittest.main()

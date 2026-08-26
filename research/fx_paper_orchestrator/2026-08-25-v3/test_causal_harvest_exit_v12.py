import unittest
from datetime import datetime, timedelta, timezone

from causal_composite_indicators_v3 import Bar
from run_causal_harvest_exit_v12 import MAX_AGE, nearest_rank, score_dynamic


def fixture(rise_at=10):
    rows = []
    for index in range(MAX_AGE + 2):
        mid = 1.001 if index >= rise_at else 1.0
        stamp = (datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z")
        rows.append(Bar("EUR_USD", stamp, mid-.0001, mid, mid-.0002, mid-.0001,
                        mid+.0001, mid+.0002, mid, mid+.0001, 100))
    return rows


class CausalHarvestExitTest(unittest.TestCase):
    def test_nearest_rank_is_deterministic(self):
        self.assertEqual(nearest_rank([4, 1, 3, 2], .35), 2)

    def test_target_hit_exits_only_at_following_open(self):
        rows = fixture(rise_at=10)
        result = score_dynamic(rows, 0, 1, .0005, "RAW_SIGNAL")
        self.assertTrue(result["tp_hit"])
        self.assertEqual(result["decision_age_m5_bars"], 10)
        self.assertEqual(result["exit_time"], rows[11].time)

    def test_unreached_target_has_finite_max_age(self):
        result = score_dynamic(fixture(rise_at=MAX_AGE + 1), 0, 1, .0005, "RAW_SIGNAL")
        self.assertFalse(result["tp_hit"])
        self.assertEqual(result["decision_age_m5_bars"], MAX_AGE)

    def test_cost_arms_do_not_change_exit_decision(self):
        rows = fixture(rise_at=10)
        results = [score_dynamic(rows, 0, 1, .0005, arm) for arm in (
            "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"
        )]
        self.assertEqual({row["exit_time"] for row in results}, {rows[11].time})
        self.assertGreater(results[0]["net_return"], results[1]["net_return"])
        self.assertGreater(results[1]["net_return"], results[2]["net_return"])


if __name__ == "__main__":
    unittest.main()

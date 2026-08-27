import unittest

from run_global_no_overlap_admission_v35 import HARD_MAX_AGE_SECONDS, apply_rule


def row(signal_id, day, pair, fill_time, score):
    return {
        "signal_id": signal_id,
        "utc_day": day,
        "pair": pair,
        "fill_time": fill_time,
        "diagnostics": {
            "native_asian_log_displacement": score,
            "training_abs_displacement_q75": 1.0,
        },
    }


class GlobalNoOverlapAdmissionV35Test(unittest.TestCase):
    def test_daily_rank_is_preserved_then_hard_age_spacing_is_applied(self):
        rows = [
            row("a", "2026-05-01", "AUD_USD", "2026-05-01T06:00:00.000000000Z", 2.0),
            row("b", "2026-05-01", "EUR_USD", "2026-05-01T06:00:00.000000000Z", 3.0),
            row("c", "2026-05-04", "GBP_USD", "2026-05-04T06:00:00.000000000Z", 4.0),
            row("d", "2026-05-05", "USD_JPY", "2026-05-05T06:00:00.000000000Z", 1.5),
        ]
        self.assertEqual(apply_rule(rows), {"b", "d"})

    def test_spacing_is_fixed_at_hard_max_age(self):
        self.assertEqual(HARD_MAX_AGE_SECONDS, 345600)


if __name__ == "__main__":
    unittest.main()

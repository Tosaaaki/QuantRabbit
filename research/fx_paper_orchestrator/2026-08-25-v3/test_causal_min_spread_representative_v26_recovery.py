from __future__ import annotations

import unittest
from datetime import timezone

import run_causal_min_spread_representative_v26 as frozen_v26
import run_causal_min_spread_representative_v26_recovery as recovery


class TimestampCompatibilityTest(unittest.TestCase):
    def test_actual_nine_digit_zero_fraction_is_exact(self):
        parsed = recovery.parse_v26_utc_timestamp("2026-05-01T11:55:00.000000000Z")
        self.assertEqual(parsed.isoformat(), "2026-05-01T11:55:00+00:00")
        self.assertEqual(parsed.tzinfo, timezone.utc)

    def test_microseconds_are_preserved_and_zero_padded(self):
        parsed = recovery.parse_v26_utc_timestamp("2026-05-01T11:55:00.1234Z")
        self.assertEqual(parsed.microsecond, 123400)

    def test_nonzero_submicrosecond_precision_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "nonzero sub-microsecond"):
            recovery.parse_v26_utc_timestamp("2026-05-01T11:55:00.000000001Z")

    def test_non_utc_or_noncanonical_timestamp_fails_closed(self):
        for value in (
            "2026-05-01T11:55:00+00:00",
            "2026-05-01 11:55:00Z",
            "2026-05-01T11:55:00.000000000+00:00",
        ):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "canonical UTC"):
                recovery.parse_v26_utc_timestamp(value)

    def test_wrapper_changes_only_the_parser_binding(self):
        original = frozen_v26.parse_time
        try:
            recovery.install_timestamp_compatibility()
            self.assertIs(frozen_v26.parse_time, recovery.parse_v26_utc_timestamp)
            self.assertIs(recovery.frozen_v26.run, frozen_v26.run)
            self.assertIs(recovery.frozen_v26.apply_rule, frozen_v26.apply_rule)
            self.assertIs(recovery.frozen_v26.arm_metrics, frozen_v26.arm_metrics)
        finally:
            frozen_v26.parse_time = original

    def test_prepared_wrapper_is_not_directly_executable(self):
        with self.assertRaisesRegex(RuntimeError, "not executable"):
            recovery.main()


if __name__ == "__main__":
    unittest.main()

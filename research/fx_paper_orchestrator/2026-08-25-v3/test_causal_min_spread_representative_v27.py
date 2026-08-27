from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from unittest import mock

import run_auction_trap_geometry_v7 as auction_v7
import run_causal_min_spread_representative_v26 as frozen_v26
import run_causal_min_spread_representative_v27 as v27
import run_portfolio_episode_netting_v15 as portfolio_v15


ROOT = Path(__file__).resolve().parent


class TimestampCompatibilityTest(unittest.TestCase):
    def test_accepts_canonical_one_to_nine_fraction_digits(self):
        for value in (
            "2026-05-01T11:55:00Z",
            "2026-05-01T11:55:00.1Z",
            "2026-05-01T11:55:00.123456Z",
            "2026-05-01T11:55:00.123456000Z",
            "2026-05-01T11:55:00.123456789Z",
            "2026-05-01T11:55:00.000000000Z",
        ):
            parsed = v27.parse_utc_nanoseconds(value)
            self.assertIsInstance(parsed, v27.EpochNanoseconds)

    def test_preserves_nonzero_nine_digit_order(self):
        left = v27.parse_utc_nanoseconds("2026-05-01T11:55:00.123456001Z")
        right = v27.parse_utc_nanoseconds("2026-05-01T11:55:00.123456789Z")
        self.assertLess(left, right)
        self.assertEqual((right - left).value, 788)

    def test_elapsed_seconds_preserves_nanosecond_delta(self):
        start = v27.parse_utc_nanoseconds("2026-05-01T05:55:00.000000001Z")
        end = v27.parse_utc_nanoseconds("2026-05-01T11:55:00.000000009Z")
        delta = end - start
        self.assertEqual(delta.value, 21_600_000_000_008)
        self.assertAlmostEqual(delta.total_seconds(), 21600.000000008, places=12)

    def test_boundary_timestamp_is_exact_across_utc_day(self):
        start = v27.parse_utc_nanoseconds("2026-05-31T23:59:59.999999999Z")
        end = v27.parse_utc_nanoseconds("2026-06-01T00:00:00.000000001Z")
        self.assertEqual((end - start).value, 2)

    def test_rejects_noncanonical_or_overprecision_input(self):
        for value in (
            "2026-05-01 11:55:00Z",
            "2026-05-01T11:55:00+00:00",
            "2026-05-01T11:55:00.0000000000Z",
        ):
            with self.assertRaisesRegex(ValueError, "not canonical UTC"):
                v27.parse_utc_nanoseconds(value)

    def test_installs_every_reachable_timestamp_binding(self):
        v27.install_timestamp_compatibility()
        self.assertIs(frozen_v26.parse_time, v27.parse_utc_nanoseconds)
        self.assertIs(portfolio_v15.timestamp, v27.parse_utc_nanoseconds)
        self.assertIs(auction_v7.timestamp, v27.parse_utc_nanoseconds)
        self.assertIs(
            frozen_v26.roundtrip_return.__globals__["timestamp"],
            v27.parse_utc_nanoseconds,
        )
        self.assertIs(
            frozen_v26.simulate_portfolio.__globals__["timestamp"],
            v27.parse_utc_nanoseconds,
        )

    def test_cost_path_accepts_nonzero_nine_digit_timestamps(self):
        v27.install_timestamp_compatibility()
        entry = frozen_v26.Bar(
            "EUR_USD", "2026-05-01T05:55:00.000000001Z",
            1.1000, 1.1001, 1.0999, 1.1000,
            1.1002, 1.1003, 1.1001, 1.1002,
        )
        exit_bar = frozen_v26.Bar(
            "EUR_USD", "2026-05-01T11:55:00.000000009Z",
            1.1010, 1.1011, 1.1009, 1.1010,
            1.1012, 1.1013, 1.1011, 1.1012,
        )
        value = frozen_v26.roundtrip_return(entry, exit_bar, 1, "EXECUTABLE_BASE", False)
        self.assertTrue(math.isfinite(value))


class FrozenStrategyDelegationTest(unittest.TestCase):
    def test_causal_score_delegates_without_extra_strategy_logic(self):
        sentinel = object()
        with mock.patch.object(frozen_v26, "causal_score", return_value=sentinel) as delegated:
            self.assertIs(v27.causal_score({"id": 1}, ["bar"], {"time": 0}), sentinel)
        delegated.assert_called_once_with({"id": 1}, ["bar"], {"time": 0})

    def test_apply_rule_delegates_without_extra_strategy_logic(self):
        sentinel = [{"signal_id": "S1"}]
        with mock.patch.object(frozen_v26, "apply_rule", return_value=sentinel) as delegated:
            self.assertIs(v27.apply_rule([{"signal_id": "S1"}], {"EUR_USD": []}), sentinel)
        delegated.assert_called_once_with([{"signal_id": "S1"}], {"EUR_USD": []})

    def test_v25_v26_evidence_remains_frozen_and_v26_rerun_forbidden(self):
        self.assertEqual(
            frozen_v26.sha256_file(ROOT / "CAUSAL_MIN_SPREAD_REPRESENTATIVE_PREREGISTRATION_V26.json"),
            "b4579b78ea045e1cbf778cbc3f643496823276e768dd7b7dad66bb6d14fd9c2d",
        )
        self.assertEqual(
            frozen_v26.sha256_file(ROOT / "run_causal_min_spread_representative_v26.py"),
            "3989c614e92b93bc107995f93952c15bdb75a5f43740882270e2c7d0916e67ab",
        )
        self.assertEqual(
            frozen_v26.sha256_file(ROOT / "V26_AUTHORIZED_RECOVERY_FAILURE.json"),
            "75cceae96df7be5a51955a0966f587d378a0328ddf4f9c4f4947c2b3ed154a2b",
        )
        state = json.loads((ROOT / "evidence/orchestrator_state_v2/state.json").read_text())
        self.assertEqual(state["cycles"]["V26"]["status"], "FAILED_AUTHORIZED_RECOVERY_NO_RERUN")
        failure = json.loads((ROOT / "V26_AUTHORIZED_RECOVERY_FAILURE.json").read_text())
        self.assertTrue(failure["next_work_order"]["v26_may_not_be_replayed"])
        self.assertFalse(v27.RUNTIME_COMPATIBILITY_PROVENANCE["v26_rerun_permitted"])

    def test_input_fraction_audit_records_exact_zero_observed_nonzero_tail(self):
        audit = json.loads((ROOT / "V27_TIMESTAMP_FRACTION_AUDIT.json").read_text())
        self.assertEqual(audit["source_binding"]["rows_scanned"], 182303)
        self.assertEqual(audit["source_binding"]["fraction_digit_counts"], {"9": 182303})
        self.assertEqual(audit["source_binding"]["fraction_7_to_9_nonzero_count"], 0)
        self.assertEqual(audit["parent_ledger_binding"]["timestamp_values_scanned"], 1500)
        self.assertEqual(audit["parent_ledger_binding"]["fraction_7_to_9_nonzero_count"], 0)


if __name__ == "__main__":
    unittest.main()

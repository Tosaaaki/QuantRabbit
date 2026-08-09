from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import real_shadow_core as core  # noqa: E402
import run_real_shadow as runner  # noqa: E402


class RealCohortShadowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run([sys.executable, str(HERE / "run_real_shadow.py")], check=True)
        cls.payload = json.loads((HERE / "real_shadow_payload.json").read_text(encoding="utf-8"))
        cls.report = json.loads((HERE / "real_shadow_report.json").read_text(encoding="utf-8"))
        cls.adapters = json.loads((HERE / "real_adapter_report.json").read_text(encoding="utf-8"))

    def test_real_cohort_rebuild_is_logically_reproducible(self) -> None:
        rebuilt = core.build_payload()
        self.assertEqual(core.logical_digest(rebuilt), core.logical_digest(self.payload))
        self.assertEqual(self.report["payload_digest"], core.logical_digest(rebuilt))

    def test_financial_invariants_reproduce_frozen_oracles(self) -> None:
        invariant = self.payload["financial_invariants"]
        self.assertTrue(invariant["exact_with_tolerance_1e_9"])
        self.assertLess(invariant["max_abs_net_diff_jpy"], 1e-9)
        self.assertLess(invariant["max_abs_retention_diff"], 1e-9)
        self.assertLess(invariant["max_abs_lcb_diff_jpy"], 1e-9)
        baseline = next(
            row for row in self.payload["candidate_summaries"]
            if row["window"] == "QUADRUPLE_64D" and row["method"] == "ALL_TRADES"
        )
        self.assertAlmostEqual(baseline["after_cost_net_jpy"], 15144.4802, places=9)

    def test_outer_chronology_embargo_and_holdout_contract(self) -> None:
        self.assertFalse(self.payload["holdout_read"])
        self.assertFalse(self.report["holdout_read"])
        for split in self.payload["manifest"]["splits"].values():
            self.assertEqual(split["embargo_seconds"], 3600)
            self.assertGreater(split["purged"], 0)
        self.assertEqual(self.payload["manifest"]["splits"]["QUADRUPLE_64D"]["validation"], 101)

    def test_oanda_financial_and_dukascopy_feature_boundaries_are_preserved(self) -> None:
        lineage = self.payload["manifest"]["lineage"]
        self.assertEqual(lineage["execution_source"], "OANDA_ACTUAL_AFTER_COST")
        self.assertEqual(lineage["feature_source"], "DUKASCOPY_DATAFEED_TICK")
        self.assertFalse(lineage["cross_source_fill_substitution"])
        for row in self.payload["episode_records"]:
            self.assertEqual(row["execution_source"], "OANDA_ACTUAL_AFTER_COST")
            self.assertTrue(row["source_boundary_preserved"])
            if row["price_action_features"] is not None:
                self.assertEqual(row["feature_source"], "DUKASCOPY_DATAFEED_TICK")

    def test_canonical_long_keys_unique_and_missing_not_zero(self) -> None:
        dims = self.payload["cube_axes"] + ["metric"]
        keys = [tuple(row[name] for name in dims) for row in self.payload["long_rows"]]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(any(row["value"] is None for row in self.payload["long_rows"]))
        cube = json.loads((HERE / "real_cube_sparse.json").read_text(encoding="utf-8"))
        self.assertTrue(cube["missing_is_absent_or_null_never_zero"])

    def test_xarray_numeric_and_missing_parity(self) -> None:
        result = self.adapters["xarray"]["adapter"]["result"]
        self.assertEqual(result["numeric_max_abs_diff"], 0.0)
        self.assertTrue(result["input_null_preserved"])
        self.assertTrue(result["known_absent_coordinate_is_nan"])

    def test_salib_is_train_only_and_exposes_unstable_real_ranking(self) -> None:
        result = self.adapters["salib"]["adapter"]["result"]
        self.assertTrue(result["ranking_fixed_on_train"])
        window = result["windows"]["QUADRUPLE_64D"]
        self.assertEqual(window["status"], "EXECUTED_TRAIN_ONLY_RANKING")
        self.assertFalse(window["validation_labels_used_for_ranking"])
        self.assertLess(window["train_to_validation_rank_agreement"], 0.5)

    def test_pymoo_matches_custom_front_but_margin_constraint_blocks_admission(self) -> None:
        windows = self.adapters["pymoo"]["adapter"]["result"]["windows"]
        self.assertTrue(all(window["front_exact_match"] for window in windows.values()))
        self.assertTrue(all(window["constrained_front_empty_due_to_margin"] for window in windows.values()))

    def test_mapie_matches_manual_bounds_without_policy_or_profit_change(self) -> None:
        result = self.adapters["mapie"]["adapter"]["result"]
        window = result["windows"]["QUADRUPLE_64D"]
        self.assertEqual(window["status"], "EXECUTED_OUTER_VALIDATION")
        self.assertEqual(window["manual_bound_max_abs_diff"], 0.0)
        self.assertFalse(window["validation_labels_used_for_fit_or_conformal"])
        self.assertEqual(result["incremental_net_jpy_attributed_to_adapter"], 0.0)

    def test_adapter_rollback_fails_closed_without_ambient_fallback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qr-adapter-rollback-") as tmp:
            with self.assertRaises(runner.CandidateUnavailable):
                runner.probe_candidate("xarray", Path(tmp))

    def test_capacity_and_adoption_are_research_only(self) -> None:
        disk = self.report["disk"]
        self.assertFalse(disk["hard_stop_free_lt_5gib"])
        self.assertFalse(disk["soft_pause_free_lt_8gib"])
        self.assertFalse(disk["run_owned_cap_exceeded"])
        self.assertFalse(disk["new_package_install"])
        self.assertEqual(self.report["profitability_increment_jpy_attributed_to_adapters"], 0.0)
        self.assertIn("NO_CHANGE_KEEP_ALL_TRADES_BASELINE", self.report["strategy_decision"])

    def test_dowhy_and_river_remain_held_and_unexecuted(self) -> None:
        self.assertNotIn("dowhy", self.adapters)
        self.assertNotIn("river", self.adapters)
        self.assertEqual(self.report["decisions"]["dowhy"]["decision"], "HOLD_UNCHANGED_NOT_RUN")
        self.assertEqual(self.report["decisions"]["river"]["decision"], "HOLD_UNCHANGED_NOT_RUN")

    def test_stdlib_independent_oracle_passes(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(HERE / "verify_real_shadow.py")],
            check=True, capture_output=True, text=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["checks"], 33)
        self.assertEqual(result["failed"], [])
        self.assertTrue(result["stdlib_only"])


if __name__ == "__main__":
    unittest.main()

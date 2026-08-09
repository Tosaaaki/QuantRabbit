from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent


class ExternalAdapterEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run([sys.executable, str(HERE / "run_external_adapters.py")], check=True)
        cls.report = json.loads((HERE / "external_adapter_report.json").read_text())

    def test_all_candidates_preserve_financial_oracle_and_are_deterministic(self) -> None:
        self.assertEqual(set(self.report), {"xarray", "salib", "pymoo", "dowhy", "mapie", "river"})
        for evidence in self.report.values():
            self.assertTrue(evidence["financial_oracle_unchanged"])
            self.assertTrue(evidence["adapter"]["deterministic_repeat"])
            self.assertEqual(evidence["fixture_records"], 252)
            self.assertEqual(evidence["long_rows"], 567)

    def test_xarray_reproduces_values_and_preserves_missing(self) -> None:
        result = self.report["xarray"]["adapter"]["result"]
        self.assertEqual(result["numeric_max_abs_diff"], 0.0)
        self.assertEqual(result["after_cost_sum_diff"], 0.0)
        self.assertEqual(result["lcb_sum_diff"], 0.0)
        self.assertTrue(result["known_absent_is_nan"])

    def test_salib_and_pymoo_match_bounded_oracles(self) -> None:
        self.assertEqual(self.report["salib"]["adapter"]["result"]["factorial_lookup_max_abs_diff"], 0.0)
        self.assertTrue(self.report["salib"]["adapter"]["result"]["training_only"])
        self.assertTrue(self.report["pymoo"]["adapter"]["result"]["front_exact_match"])
        self.assertTrue(self.report["pymoo"]["adapter"]["result"]["validation_only"])

    def test_dowhy_mapie_and_river_independent_checks(self) -> None:
        self.assertLess(self.report["dowhy"]["adapter"]["result"]["effect_abs_diff"], 1e-12)
        self.assertLess(self.report["mapie"]["adapter"]["result"]["manual_bound_max_abs_diff"], 1e-12)
        self.assertFalse(self.report["mapie"]["adapter"]["result"]["holdout_read"])
        self.assertEqual(self.report["river"]["adapter"]["result"]["mean_abs_diff"], 0.0)
        self.assertFalse(self.report["river"]["adapter"]["result"]["holdout_read"])

    def test_disk_and_sbom_contract(self) -> None:
        disk = json.loads((HERE / "disk_checkpoints.json").read_text())
        self.assertFalse(disk["output_cap_exceeded"])
        self.assertFalse(disk["outside_status_changed"])
        self.assertGreater(disk["free_final_bytes"], 8 * 1024**3)
        self.assertTrue(all(not checkpoint["hard_stop_lt_5gib"] for checkpoint in disk["checkpoints"]))
        self.assertTrue(all(not checkpoint["soft_pause_ge_1gib_decrease"] for checkpoint in disk["checkpoints"]))
        wheels = json.loads((HERE / "adapter_wheel_manifest.json").read_text())
        self.assertTrue(all(items and all(len(item["sha256"]) == 64 for item in items) for items in wheels.values()))
        sbom = json.loads((HERE / "adapter_sbom.json").read_text())
        self.assertTrue(all(items for items in sbom.values()))
        manifest_hashes = {item["sha256"] for items in wheels.values() for item in items}
        direct_lock = (HERE / "research_lock.txt").read_text()
        for candidate in ("xarray", "salib", "pymoo", "dowhy", "mapie", "river"):
            direct_wheel = next(item for item in wheels[candidate] if item["filename"].lower().startswith(candidate))
            self.assertIn(direct_wheel["sha256"], manifest_hashes)
            self.assertIn(direct_wheel["sha256"], direct_lock)


if __name__ == "__main__":
    unittest.main()

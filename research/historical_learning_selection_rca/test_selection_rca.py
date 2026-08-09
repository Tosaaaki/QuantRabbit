from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

from hypothesis import given, strategies as st
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SPEC = importlib.util.spec_from_file_location("selection_rca", ROOT / "run_selection_rca.py")
RCA = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = RCA
SPEC.loader.exec_module(RCA)


class SelectionRcaProperties(unittest.TestCase):
    def test_frozen_bindings_verify(self) -> None:
        actual = RCA.verify_frozen(REPO)
        self.assertEqual(actual, RCA.FROZEN)

    def test_coverage_binding_passes_missing_forecast(self) -> None:
        self.assertTrue(RCA.coverage_binding_take({"forecast_present": False}, -999.0))
        self.assertFalse(RCA.coverage_binding_take({"forecast_present": True}, -0.01))
        self.assertTrue(RCA.coverage_binding_take({"forecast_present": True}, 0.01))

    def test_train_threshold_api_cannot_receive_validation_labels(self) -> None:
        rows = [
            {"episode_id": str(index), "net_jpy": value, "pair": "USD_JPY", "side": "LONG", "units": 1000, "intended_price": 150.0, "financing_jpy": 0.0}
            for index, value in enumerate(([100.0, -10.0, 50.0, -5.0, 25.0] * 8))
        ]
        predictions = np.linspace(-2.0, 2.0, len(rows))
        first = RCA.choose_threshold(rows, predictions)
        validation_labels_that_must_not_matter = [999999.0, -999999.0]
        self.assertEqual(len(validation_labels_that_must_not_matter), 2)
        second = RCA.choose_threshold(rows, predictions)
        self.assertEqual(first["status"], second["status"])
        self.assertEqual(first["threshold_jpy"], second["threshold_jpy"])

    def test_pair_side_calibration_requires_twenty_oof_rows(self) -> None:
        rows = [{"pair": "USD_JPY", "side": "LONG", "net_jpy": 10.0}] * 19
        offsets = RCA.calibration_offsets(rows, np.zeros(19))
        self.assertNotIn("USD_JPY|LONG", offsets["groups"])
        rows.append({"pair": "USD_JPY", "side": "LONG", "net_jpy": 10.0})
        offsets = RCA.calibration_offsets(rows, np.zeros(20))
        self.assertEqual(offsets["groups"]["USD_JPY|LONG"], 10.0)

    @given(
        st.lists(st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False), min_size=1, max_size=100),
        st.lists(st.booleans(), min_size=1, max_size=100),
    )
    def test_incremental_identity(self, values: list[float], choices: list[bool]) -> None:
        size = min(len(values), len(choices))
        values, choices = values[:size], choices[:size]
        missed_winners = sum(value for value, take in zip(values, choices) if not take and value > 0)
        avoided_losers = -sum(value for value, take in zip(values, choices) if not take and value < 0)
        direct = sum(value if take else 0.0 for value, take in zip(values, choices)) - sum(values)
        self.assertAlmostEqual(direct, avoided_losers - missed_winners, places=8)

    def test_shadow_cohort_never_infers_skip(self) -> None:
        episodes = RCA.read_jsonl(REPO / "research/historical_learning_admission/all_entry_episodes_v1.jsonl")
        parent = RCA.load_parent(REPO)
        rows = RCA.enrich_rows(REPO, parent, episodes)
        shadow = RCA.shadow_rows(rows)
        self.assertEqual(len(shadow), 549)
        self.assertTrue(all(row["skip_inference"] == "NOT_INFERRED" for row in shadow))

    def test_post_feature_thesis_is_not_decision_feature(self) -> None:
        report = json.loads((ROOT / "selection_rca_report_v1.json").read_text())
        self.assertGreater(report["missingness"]["post_feature_only_thesis_records_labeled"], 0)
        self.assertLess(
            report["missingness"]["decision_time_thesis_coverage_labeled"],
            report["missingness"]["thesis_record_coverage_labeled"],
        )

    def test_mnar_is_not_claimed_identifiable(self) -> None:
        prereg = json.loads((ROOT / "preregister_v1.json").read_text())
        report = json.loads((ROOT / "selection_rca_report_v1.json").read_text())
        self.assertIn("cannot be proven", prereg["analyses"]["interpretation_guard"])
        self.assertIn("MNAR_NOT_IDENTIFIABLE", report["missingness"]["mnar_conclusion"])

    def test_x_task_files_are_read_only_inputs(self) -> None:
        report = json.loads((ROOT / "selection_rca_report_v1.json").read_text())
        self.assertFalse(report["x_contract"]["admitted"])
        self.assertTrue(all(path.startswith("research/x_fx_methods/") for path in report["x_contract"]["files_read_only"]))


if __name__ == "__main__":
    unittest.main()

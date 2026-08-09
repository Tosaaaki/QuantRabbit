import json
from pathlib import Path
import subprocess
import hashlib
import unittest


HERE = Path(__file__).resolve().parent


def rows(name):
    return [json.loads(line) for line in (HERE / name).read_text().splitlines() if line.strip()]


class FusionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["python3", str(HERE / "run_fusion.py")], check=True, cwd=HERE.parents[2])
        cls.inference = rows("inference_table_v1.jsonl")
        cls.outcomes = rows("outcome_table_v1.jsonl")
        cls.fused = rows("fused_decisions_v1.jsonl")
        cls.report = json.loads((HERE / "utilization_report_v1.json").read_text())
        cls.prereg = json.loads((HERE / "preregister_v1.json").read_text())

    def test_frozen_scope_and_no_side_effects(self):
        self.assertFalse(self.prereg["permissions"]["holdout_read"])
        self.assertFalse(self.report["holdout_read"])
        self.assertFalse(self.report["live_paper_broker_order_deploy_touched"])

    def test_exact_251_separate_outcomes(self):
        self.assertEqual(len(self.outcomes), 251)
        self.assertEqual(len({row["episode_id"] for row in self.outcomes}), 251)
        forbidden = {"actual_after_cost_net", "fill", "margin", "DD", "unwind", "terminal_reason"}
        self.assertFalse(any(forbidden.intersection(row) for row in self.inference))

    def test_all_systems_have_episode_rows(self):
        systems = {item["system_id"] for item in self.report["inventory"]}
        self.assertEqual(len(self.inference), 251 * len(systems))
        self.assertEqual({row["system_id"] for row in self.inference}, systems)
        self.assertEqual(len({(row["episode_id"], row["system_id"]) for row in self.inference}), len(self.inference))

    def test_required_inference_schema_and_hashes(self):
        required = set(self.prereg["inference_row_required_columns"])
        self.assertTrue(all(required.issubset(row) for row in self.inference))
        self.assertTrue(all(len(row["output_sha"]) == 64 for row in self.inference))

    def test_missing_is_null_not_zero(self):
        missing = [row for row in self.inference if row["missing_inputs"]]
        self.assertTrue(missing)
        self.assertTrue(all(row["probability_or_score"] is None for row in missing if row["system_id"] not in {"forecast", "price_action"}))

    def test_one_fused_answer_per_episode(self):
        self.assertEqual(len(self.fused), 251)
        self.assertEqual(len({row["decision_id"] for row in self.fused}), 251)
        self.assertTrue(all(row["action"] in {"TRADE", "WAIT", "SKIP", "MANAGE"} for row in self.fused))

    def test_no_all_trades_fallback(self):
        self.assertFalse(self.report["causal_bottleneck"]["all_trades_fallback_detected"])
        self.assertFalse(self.report["decision_utilization_kpi"]["all_trades_baseline_used_as_decision"])

    def test_margin_evidence_checkpoint(self):
        margin = self.report["lineage"]["margin_evidence_64d_validation"]
        self.assertEqual((margin["known"], margin["total"]), (15, 101))
        self.assertAlmostEqual(margin["coverage"], 15 / 101)

    def test_adapters_are_not_mislabeled_as_profit(self):
        classes = {item["system_id"]: item["classification"] for item in self.report["inventory"]}
        self.assertEqual(classes["xarray"], "GENERATED_ONLY")
        self.assertEqual(classes["mapie"], "DISCONNECTED")
        self.assertEqual(self.report["profitability_increment_attributed_to_fusion_jpy"], 0.0)

    def test_validation_is_not_used_for_fit(self):
        for window in self.report["fusion"].values():
            for candidate in window["candidates"].values():
                if candidate["status"] == "EVALUATED":
                    self.assertIn("TRAIN OOF", candidate["selection_rule"])

    def test_decision_time_constraints_fail_closed(self):
        self.assertEqual(self.report["final_trade_count"], 0)
        self.assertTrue(all(row["action"] != "TRADE" for row in self.fused))

    def test_cube_preserves_missingness(self):
        cube = json.loads((HERE / "inference_cube_sparse_v1.json").read_text())
        self.assertGreater(cube["missing_cells"], 0)
        self.assertIn("never zero", cube["missing_representation"])

    def test_rebuild_is_byte_reproducible(self):
        names = ["inference_table_v1.jsonl", "outcome_table_v1.jsonl", "fused_decisions_v1.jsonl", "inference_cube_sparse_v1.json", "utilization_report_v1.json"]
        before = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}
        subprocess.run(["python3", str(HERE / "run_fusion.py")], check=True, cwd=HERE.parents[2])
        after = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_audit  # noqa: E402
from audit_core import (  # noqa: E402
    REQUIRED_FIELDS,
    bid_ask_after_cost,
    build_cube,
    deterministic_conformal,
    drift_refutation,
    pairwise_interactions,
    pareto_front,
    placebo_refutation,
    records_to_long,
)


class EcosystemAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = build_audit.fixture_records()
        cls.rows = records_to_long(cls.records)
        cls.cube = build_cube(cls.rows)

    def test_canonical_long_table_contract_and_no_zero_for_missing(self) -> None:
        self.assertTrue(self.rows)
        for row in self.rows:
            self.assertTrue(set(REQUIRED_FIELDS).issubset(row))
            self.assertIn(row["split"], {"TRAIN", "VALIDATION"})
            self.assertNotEqual(row.get("admission_status"), "HOLDOUT")
        # Deliberately omitted coordinate remains sparse, rather than an invented 0.
        key = json.dumps(["VALIDATION", "M15", "AUD_JPY", "RANGE", "cube_shadow", "stress_plus_1pip", "margin_cap_70", "TIMEOUT"], separators=(",", ":"))
        self.assertNotIn(key, self.cube.values.get("after_cost_net_jpy", {}))

    def test_cube_reproduces_individual_metric_aggregation(self) -> None:
        group = [r for r in self.records if r["split"] == "TRAIN" and r["regime"] == "TREND" and r["method"] == "baseline" and r["cost"] == "observed_bid_ask" and r["risk"] == "margin_cap_92" and r["exit"] == "SL"]
        expected = sum(float(r["net_jpy"]) for r in group if r["net_jpy"] is not None)
        actual = self.cube.value("after_cost_net_jpy", split="TRAIN", timeframe="M5", pair="EUR_USD", regime="TREND", method="baseline", cost="observed_bid_ask", risk="margin_cap_92", exit="SL")
        self.assertEqual(actual, expected)

    def test_bid_ask_oracle_is_side_aware_and_cost_explicit(self) -> None:
        long = bid_ask_after_cost(side="LONG", entry_bid=100.0, entry_ask=100.2, exit_bid=101.0, exit_ask=101.2, units=1000, fee_jpy=2.0)
        short = bid_ask_after_cost(side="SHORT", entry_bid=100.0, entry_ask=100.2, exit_bid=99.0, exit_ask=99.2, units=1000, financing_jpy=3.0)
        self.assertEqual((long["entry_fill"], long["exit_fill"]), (100.2, 101.0))
        self.assertEqual((short["entry_fill"], short["exit_fill"]), (100.0, 99.2))
        self.assertAlmostEqual(long["net_jpy"], 798.0, places=9)
        self.assertAlmostEqual(short["net_jpy"], 797.0, places=9)

    def test_interactions_and_pareto_keep_multiple_metrics(self) -> None:
        interactions = pairwise_interactions(self.rows)
        self.assertTrue(interactions)
        candidates = build_audit._candidate_summaries(self.rows)
        front = pareto_front(candidates)
        self.assertTrue(all(row["split"] == "VALIDATION" for row in front))
        self.assertTrue(all("lcb_jpy" in row and "max_drawdown_jpy" in row for row in front))
        self.assertFalse(any(row.get("holdout") is True for row in front))

    def test_fallback_proofs_are_deterministic_and_not_claimed_external(self) -> None:
        values = [float(r["net_jpy"]) for r in self.records if r["net_jpy"] is not None]
        self.assertEqual(placebo_refutation(values), placebo_refutation(values))
        self.assertEqual(drift_refutation(values), drift_refutation(values))
        conformal = deterministic_conformal(values)
        self.assertEqual(conformal["status"], "EXECUTED_FALLBACK")
        self.assertGreaterEqual(conformal["coverage"], 0.0)
        self.assertLessEqual(conformal["coverage"], 1.0)

    def test_generated_artifact_is_reproducible_after_build(self) -> None:
        build_audit.main()
        first = hashlib.sha256((HERE / "canonical_long_table.jsonl").read_bytes()).hexdigest()
        build_audit.main()
        second = hashlib.sha256((HERE / "canonical_long_table.jsonl").read_bytes()).hexdigest()
        self.assertEqual(first, second)
        manifest = json.loads((HERE / "run_manifest.json").read_text())
        self.assertFalse(manifest["holdout_read"])
        self.assertFalse(manifest["live_paper_broker_order_deploy_touched"])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from datetime import datetime, timezone


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def read_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    if "." in normalized:
        prefix, suffix = normalized.split(".", 1)
        plus = suffix.find("+")
        fraction, offset = suffix[:plus], suffix[plus:]
        normalized = f"{prefix}.{fraction[:6]}{offset}"
    return datetime.fromisoformat(normalized).astimezone(timezone.utc)


class EvidenceLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = read_jsonl("evidence_ledger_v1.jsonl")
        cls.decisions = read_jsonl("fused_decisions_rerun_v1.jsonl")
        cls.report = json.loads((HERE / "coverage_report_v1.json").read_text(encoding="utf-8"))
        source = REPO / "research/historical_learning_admission/all_entry_episodes_v1.jsonl"
        cls.episodes = {
            row["episode_id"]: row
            for row in (json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip())
            if row.get("label_status") == "ACTUAL_AFTER_COST"
        }

    def test_exact_251_unique_rows(self) -> None:
        self.assertEqual(251, len(self.rows))
        self.assertEqual(251, len({row["decision_id"] for row in self.rows}))

    def test_output_hashes_bind_rows(self) -> None:
        for row in self.rows:
            output_sha = row.pop("output_sha")
            logical = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            row["output_sha"] = output_sha
            self.assertEqual(output_sha, logical)

    def test_exact_order_join_for_every_episode(self) -> None:
        for row in self.rows:
            self.assertTrue(row["candidate_order"]["coverage"])
            self.assertEqual("ACTUAL", row["candidate_order"]["evidence_kind"])
            self.assertIn(row["decision_id"], row["candidate_order"]["provenance"])

    def test_no_future_pricing_watermark(self) -> None:
        for row in self.rows:
            if not row["pricing"]["coverage"]:
                continue
            self.assertLessEqual(utc(row["pricing"]["value"]["watermark"]), utc(row["decision_time"]))
            self.assertGreaterEqual(row["pricing"]["value"]["watermark_age_seconds"], 0.0)
            self.assertLessEqual(row["pricing"]["value"]["watermark_age_seconds"], 15.0)

    def test_bid_ask_and_executable_side(self) -> None:
        for row in self.rows:
            if not row["pricing"]["coverage"]:
                continue
            value = row["pricing"]["value"]
            self.assertLessEqual(value["bid"], value["ask"])
            expected = value["ask"] if row["side"] == "LONG" else value["bid"]
            self.assertEqual(expected, value["executable_entry_side_price"])

    def test_mid_is_descriptive_only(self) -> None:
        for row in self.rows:
            value = row["candidate_order"]["value"]
            if value["order_type"] == "MARKET_ORDER":
                self.assertIsNone(value["entry_price"])

    def test_financing_is_not_backfilled(self) -> None:
        for row in self.rows:
            self.assertEqual("MISSING", row["costs"]["financing"]["evidence_kind"])
            self.assertFalse(row["costs"]["financing"]["coverage"])

    def test_margin_missing_remains_missing(self) -> None:
        for row in self.rows:
            margin = row["portfolio_margin"]
            self.assertEqual("MISSING", margin["evidence_kind"])
            self.assertIsNone(margin["value"]["margin_available"])
            self.assertIsNone(margin["value"]["margin_used"])

    def test_single_leg_close_does_not_prove_unwind(self) -> None:
        for row in self.rows:
            self.assertEqual("MISSING", row["exit_unwind"]["unwind_validity"]["evidence_kind"])
            self.assertFalse(row["exit_unwind"]["coverage"])

    def test_strict_fusion_never_trades_ineligible(self) -> None:
        ledger = {row["decision_id"]: row for row in self.rows}
        self.assertEqual(0, self.report["strict_eligible"])
        for decision in self.decisions:
            if decision["action"] == "TRADE":
                self.assertTrue(ledger[decision["decision_id"]]["strict_eligible"])
        self.assertEqual(0, sum(row["action"] == "TRADE" for row in self.decisions))

    def test_observed_execution_is_evaluation_only(self) -> None:
        for row in self.rows:
            self.assertTrue(row["observed_execution"]["evaluation_only"])
            if row["observed_execution"]["fill"]:
                self.assertTrue(row["observed_execution"]["fill"]["evaluation_only"])
                self.assertGreaterEqual(row["observed_execution"]["fill"]["delay_seconds"], 0.0)

    def test_window_counts_match_preregister(self) -> None:
        expected = {"INITIAL_16D": (13, 12), "DOUBLE_32D": (43, 31), "QUADRUPLE_64D": (145, 101)}
        for window, (train, validation) in expected.items():
            actual = self.report["by_window_split"][window]
            self.assertEqual(train, actual["TRAIN"]["episodes"])
            self.assertEqual(validation, actual["VALIDATION"]["episodes"])

    def test_independent_full_net(self) -> None:
        full = sum(float(row["net_jpy"]) for row in self.episodes.values())
        self.assertAlmostEqual(full, self.report["coverage_selection_bias_diagnostic"]["full_net_jpy"], places=9)

    def test_deterministic_regeneration(self) -> None:
        before = {name: digest(HERE / name) for name in ("evidence_ledger_v1.jsonl", "coverage_report_v1.json", "fused_decisions_rerun_v1.jsonl")}
        subprocess.run([sys.executable, str(HERE / "build_ledger.py")], cwd=REPO, check=True, capture_output=True, text=True)
        after = {name: digest(HERE / name) for name in before}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

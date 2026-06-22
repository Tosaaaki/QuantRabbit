from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "package_bidask_replay_precision_rules.py"
    spec = importlib.util.spec_from_file_location("package_bidask_replay_precision_rules", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


packager = _load_module()


class BidAskReplayPackagerTest(unittest.TestCase):
    def test_package_payload_preserves_adoption_and_truth_metadata(self) -> None:
        payload = {
            "generated_at_utc": "2026-06-22T16:00:00Z",
            "granularity": "S5",
            "truth_source": "local OANDA S5 bid/ask candles",
            "price_truth_coverage": {
                "status": "PRICE_TRUTH_OK",
                "reason": "All samples scored.",
                "adoption_level": "FULL_REPLAY_READY",
                "candidate_rule_validation_blocked": False,
                "global_currency_validation_blocked": False,
                "evaluated_rows": 650,
                "history_files": 140,
                "history_candles": 330150,
                "missing_price_truth_samples": 0,
            },
            "forecast_sample_coverage": {
                "min_directional_samples_for_precision_rule": 30,
                "min_active_days_for_daily_stability": 3,
                "pair_count": 28,
                "pair_direction_count": 42,
                "unscorable_no_market_samples": 12,
                "under_sampled_pair_directions": [{"pair": "GBP_USD"}],
            },
            "precision_rules": {
                "selection": {"edge_min_samples": 30},
                "adoption_summary": {
                    "live_grade_support_rules": 0,
                    "rank_only_support_rules": 3,
                    "negative_block_rules": 1,
                },
                "edge_rules": [
                    {
                        "name": "EUR_USD_DOWN_S5_BIDASK_HARVEST_TP5_SL7",
                        "pair": "EUR_USD",
                        "direction": "DOWN",
                        "side": "SHORT",
                        "samples": 226,
                        "adoption_status": "RANK_ONLY_NOT_DAILY_STABLE",
                    }
                ],
                "daily_stable_edge_rules": [],
                "contrarian_edge_rules": [],
                "daily_stable_contrarian_edge_rules": [],
                "negative_rules": [],
                "rejected_sampled_segments": [],
                "rejected_contrarian_segments": [],
                "rejected_daily_stability_segments": [
                    {
                        "name": "EUR_USD_DOWN_S5_BIDASK_HARVEST_TP5_SL7",
                        "adoption_status": "RANK_ONLY_NOT_DAILY_STABLE",
                    }
                ],
            },
        }

        packaged = packager.package_payload(payload, source_report=Path("latest.json"))

        self.assertEqual(packaged["generated_from"], "scripts/oanda_history_replay_validate.py")
        self.assertEqual(
            packaged["packaged_by"],
            "scripts/package_bidask_replay_precision_rules.py",
        )
        self.assertEqual(packaged["source_report"], "latest.json")
        self.assertEqual(packaged["price_truth_coverage"]["status"], "PRICE_TRUTH_OK")
        self.assertEqual(packaged["price_truth_coverage"]["evaluated_rows"], 650)
        self.assertEqual(packaged["adoption_summary"]["rank_only_support_rules"], 3)
        self.assertEqual(packaged["edge_rules"][0]["adoption_status"], "RANK_ONLY_NOT_DAILY_STABLE")
        self.assertEqual(
            packaged["forecast_sample_coverage_summary"]["under_sampled_pair_directions"],
            1,
        )

    def test_package_payload_rejects_partial_price_truth_by_default(self) -> None:
        payload = {
            "price_truth_coverage": {"status": "PARTIAL_PRICE_TRUTH"},
            "precision_rules": {"edge_rules": []},
        }

        with self.assertRaisesRegex(ValueError, "refusing to package partial"):
            packager.package_payload(payload, source_report=Path("latest.json"))

        packaged = packager.package_payload(
            payload,
            source_report=Path("latest.json"),
            allow_partial=True,
        )
        self.assertEqual(packaged["price_truth_coverage"]["status"], "PARTIAL_PRICE_TRUTH")


if __name__ == "__main__":
    unittest.main()

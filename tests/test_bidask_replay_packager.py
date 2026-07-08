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
            "history_dirs": [
                "tmp/qr_acceptance_check/oanda_history_s5/20260622T155928Z",
                "tmp/qr_acceptance_check/oanda_history_s5_windowed/20260622T163008Z",
            ],
            "granularity": "S5",
            "truth_source": "local OANDA S5 bid/ask candles",
            "price_truth_coverage": {
                "status": "PRICE_TRUTH_OK",
                "reason": "All samples scored.",
                "adoption_level": "PAIR_LOCAL_RANK_ONLY",
                "candidate_rule_validation_blocked": False,
                "global_currency_validation_blocked": True,
                "evaluated_rows": 650,
                "unscorable_no_market_rows": 12,
                "pending_future_truth_rows": 4,
                "history_files": 140,
                "history_candles": 330150,
                "missing_price_truth_samples": 0,
                "missing_price_window_group_count": 0,
                "unscorable_no_market_window_group_count": 2,
                "pending_future_truth_window_group_count": 1,
                "all_currency_sample_coverage_status": "UNDER_SAMPLED",
                "under_sampled_pair_direction_count": 1,
                "under_sampled_pair_directions": ["GBP_USD:DOWN"],
                "under_sampled_missing_evaluated_samples": 29,
                "history_fetch_command": (
                    "PYTHONPATH=src python3 scripts/oanda_history_fetch.py "
                    "--pairs AUD_USD --granularities S5 --price BA "
                    "--from 2026-06-01T00:00:00Z --to 2026-06-02T00:00:00Z "
                    "--output-dir logs/replay/oanda_history"
                ),
                "history_fetch_command_count": 1,
                "history_fetch_command_mode": "WINDOWED",
                "history_fetch_commands": [
                    {
                        "date": "2026-06-01",
                        "pairs": ["AUD_USD"],
                        "forecast_rows_missing_truth": 12,
                        "command": (
                            "PYTHONPATH=src python3 scripts/oanda_history_fetch.py "
                            "--pairs AUD_USD --granularities S5 --price BA "
                            "--from 2026-06-01T00:00:00Z --to 2026-06-02T00:00:00Z "
                            "--output-dir logs/replay/oanda_history"
                        ),
                    }
                ],
            },
            "forecast_sample_coverage": {
                "min_directional_samples_for_precision_rule": 30,
                "min_active_days_for_daily_stability": 3,
                "pair_count": 28,
                "pair_direction_count": 42,
                "unscorable_no_market_samples": 12,
                "pending_future_truth_samples": 4,
                "pairs": [
                    {
                        "pair": "GBP_USD",
                        "forecast_samples": 12,
                        "evaluated_samples": 1,
                        "missing_price_truth_samples": 0,
                        "pending_future_truth_samples": 4,
                        "unscorable_no_market_samples": 7,
                        "missing_evaluated_samples_to_min_directional": 29,
                    }
                ],
                "under_sampled_pair_directions": [
                    {
                        "pair": "GBP_USD",
                        "direction": "DOWN",
                        "forecast_samples": 12,
                        "forecast_active_days": 1,
                        "evaluated_samples": 1,
                        "evaluated_active_days": 1,
                        "unscorable_no_market_samples": 7,
                        "pending_future_truth_samples": 4,
                        "missing_price_truth_samples": 0,
                        "missing_evaluated_samples": 29,
                        "missing_active_days": 2,
                        "coverage_gap_reasons": [
                            "INSUFFICIENT_EVALUATED_SAMPLES",
                            "INSUFFICIENT_ACTIVE_DAYS",
                        ],
                    }
                ],
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
        packaged_from_live = packager.package_payload(
            payload,
            source_report=Path(
                "/Users/tossaki/App/QuantRabbit-live/logs/reports/forecast_improvement/latest.json"
            ),
        )
        self.assertEqual(
            packaged_from_live["source_report"],
            "logs/reports/forecast_improvement/latest.json",
        )
        self.assertEqual(
            packaged["history_dirs"],
            [
                "tmp/qr_acceptance_check/oanda_history_s5/20260622T155928Z",
                "tmp/qr_acceptance_check/oanda_history_s5_windowed/20260622T163008Z",
            ],
        )
        self.assertEqual(packaged["price_truth_coverage"]["status"], "PRICE_TRUTH_OK")
        self.assertEqual(packaged["price_truth_coverage"]["evaluated_rows"], 650)
        self.assertEqual(packaged["price_truth_coverage"]["pending_future_truth_rows"], 4)
        self.assertEqual(
            packaged["price_truth_coverage"]["pending_future_truth_window_group_count"],
            1,
        )
        self.assertEqual(
            packaged["price_truth_coverage"]["history_fetch_command_count"],
            1,
        )
        self.assertEqual(
            packaged["price_truth_coverage"]["history_fetch_commands"][0]["pairs"],
            ["AUD_USD"],
        )
        self.assertEqual(
            packaged["price_truth_coverage"]["all_currency_sample_coverage_status"],
            "UNDER_SAMPLED",
        )
        self.assertEqual(
            packaged["price_truth_coverage"]["under_sampled_pair_directions"],
            ["GBP_USD:DOWN"],
        )
        self.assertEqual(packaged["adoption_summary"]["rank_only_support_rules"], 3)
        self.assertEqual(packaged["edge_rules"][0]["adoption_status"], "RANK_ONLY_NOT_DAILY_STABLE")
        self.assertEqual(
            packaged["forecast_sample_coverage_summary"]["under_sampled_pair_directions"],
            1,
        )
        self.assertEqual(
            packaged["forecast_sample_coverage_summary"]["pending_future_truth_samples"],
            4,
        )
        self.assertEqual(
            packaged["forecast_sample_coverage_summary"]["under_sampled_gap_reason_counts"],
            {"INSUFFICIENT_ACTIVE_DAYS": 1, "INSUFFICIENT_EVALUATED_SAMPLES": 1},
        )
        self.assertEqual(
            packaged["forecast_sample_coverage_summary"]["under_sampled_pair_direction_examples"],
            [
                {
                    "pair": "GBP_USD",
                    "direction": "DOWN",
                    "forecast_samples": 12,
                    "forecast_active_days": 1,
                    "evaluated_samples": 1,
                    "evaluated_active_days": 1,
                    "unscorable_no_market_samples": 7,
                    "pending_future_truth_samples": 4,
                    "missing_price_truth_samples": 0,
                    "missing_evaluated_samples": 29,
                    "missing_active_days": 2,
                    "coverage_gap_reasons": [
                        "INSUFFICIENT_EVALUATED_SAMPLES",
                        "INSUFFICIENT_ACTIVE_DAYS",
                    ],
                }
            ],
        )
        self.assertEqual(
            packaged["forecast_sample_coverage_summary"]["under_sampled_pair_direction_details"],
            packaged["forecast_sample_coverage_summary"]["under_sampled_pair_direction_examples"],
        )
        self.assertEqual(
            packaged["forecast_sample_coverage_summary"]["under_sampled_pair_direction_detail_count"],
            1,
        )
        self.assertEqual(
            packaged["forecast_sample_coverage_summary"]["under_sampled_pair_direction_examples_omitted"],
            0,
        )
        self.assertEqual(
            packaged["forecast_sample_coverage_summary"]["pair_coverage_examples"],
            [
                {
                    "pair": "GBP_USD",
                    "forecast_samples": 12,
                    "evaluated_samples": 1,
                    "missing_price_truth_samples": 0,
                    "pending_future_truth_samples": 4,
                    "unscorable_no_market_samples": 7,
                    "missing_evaluated_samples_to_min_directional": 29,
                }
            ],
        )
        self.assertEqual(
            packaged["forecast_sample_coverage_summary"]["pair_coverage"],
            packaged["forecast_sample_coverage_summary"]["pair_coverage_examples"],
        )
        self.assertEqual(packaged["forecast_sample_coverage_summary"]["pair_coverage_count"], 1)
        self.assertEqual(
            packaged["forecast_sample_coverage_summary"]["pair_coverage_examples_omitted"],
            0,
        )

    def test_package_payload_preserves_full_coverage_details_beyond_examples(self) -> None:
        coverage_rows = [
            {
                "pair": f"PAIR_{index:02d}",
                "direction": "UP",
                "forecast_samples": 30 + index,
                "forecast_active_days": 3,
                "evaluated_samples": 30 + index,
                "evaluated_active_days": 3,
                "coverage_gap_reasons": ["PENDING_FUTURE_TRUTH_WINDOW"],
            }
            for index in range(25)
        ]
        pair_rows = [
            {
                "pair": f"PAIR_{index:02d}",
                "forecast_samples": 30 + index,
                "evaluated_samples": 30 + index,
            }
            for index in range(28)
        ]
        payload = {
            "price_truth_coverage": {"status": "PRICE_TRUTH_OK"},
            "forecast_sample_coverage": {
                "under_sampled_pair_directions": coverage_rows,
                "pairs": pair_rows,
            },
            "precision_rules": {"edge_rules": []},
        }

        packaged = packager.package_payload(payload, source_report=Path("latest.json"))
        summary = packaged["forecast_sample_coverage_summary"]

        self.assertEqual(summary["under_sampled_pair_direction_detail_count"], 25)
        self.assertEqual(len(summary["under_sampled_pair_direction_details"]), 25)
        self.assertEqual(len(summary["under_sampled_pair_direction_examples"]), 24)
        self.assertEqual(summary["under_sampled_pair_direction_examples_omitted"], 1)
        self.assertEqual(summary["pair_coverage_count"], 28)
        self.assertEqual(len(summary["pair_coverage"]), 28)
        self.assertEqual(len(summary["pair_coverage_examples"]), 24)
        self.assertEqual(summary["pair_coverage_examples_omitted"], 4)

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

    def test_pair_filtered_refresh_preserves_rules_for_unrefreshed_pairs(self) -> None:
        payload = {
            "generated_at_utc": "2026-07-08T12:30:00Z",
            "pair_filter": ["EUR_JPY"],
            "price_truth_coverage": {"status": "PRICE_TRUTH_OK"},
            "precision_rules": {
                "adoption_summary": {"negative_block_rules": 1},
                "negative_rules": [
                    {
                        "name": "EUR_JPY_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY",
                        "pair": "EUR_JPY",
                        "direction": "DOWN",
                        "samples": 1200,
                    }
                ],
            },
        }
        existing = {
            "generated_at_utc": "2026-07-03T14:52:18Z",
            "source_report": "logs/reports/forecast_improvement/old.json",
            "adoption_summary": {"negative_block_rules": 2},
            "negative_rules": [
                {
                    "name": "EUR_JPY_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY",
                    "pair": "EUR_JPY",
                    "direction": "DOWN",
                    "samples": 1147,
                },
                {
                    "name": "USD_JPY_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY",
                    "pair": "USD_JPY",
                    "direction": "DOWN",
                    "samples": 525,
                },
            ],
        }

        packaged = packager.package_payload(payload, source_report=Path("eurjpy.json"))
        packager.preserve_existing_rule_rows(packaged, existing)

        rows_by_name = {row["name"]: row for row in packaged["negative_rules"]}
        self.assertEqual(
            rows_by_name["EUR_JPY_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY"]["samples"],
            1200,
        )
        self.assertNotIn(
            "preserved_from_existing_packaged_artifact",
            rows_by_name["EUR_JPY_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY"],
        )
        preserved = rows_by_name["USD_JPY_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY"]
        self.assertTrue(preserved["preserved_from_existing_packaged_artifact"])
        self.assertTrue(preserved["preserved_because_pair_filtered_source"])
        self.assertEqual(
            preserved["preserved_from_source_report"],
            "logs/reports/forecast_improvement/old.json",
        )
        self.assertTrue(packaged["existing_rule_rows_preserved"])
        self.assertEqual(packaged["adoption_summary"]["negative_block_rules"], 2)


if __name__ == "__main__":
    unittest.main()

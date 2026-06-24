from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "package_oanda_universal_rotation_rules.py"
    spec = importlib.util.spec_from_file_location("package_oanda_universal_rotation_rules", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


packager = _load_module()


class OandaUniversalRotationPackagerTest(unittest.TestCase):
    def test_package_payload_preserves_inversion_selectors(self) -> None:
        payload = {
            "generated_at_utc": "2026-06-21T13:01:37Z",
            "qualified_inversion_selector_count": 1,
            "high_precision_inversion_selector_count": 0,
            "config": {
                "min_positive_day_rate": 0.55,
                "min_validation_win_rate": 0.52,
                "inversion_selector_min_samples": 60,
                "inversion_selector_confluence_sizes": [2, 3],
                "unused": "drop",
            },
            "campaign_firepower": {
                "status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
                "minimum_return_pct": 5.0,
                "target_return_pct": 10.0,
                "high_precision": {"unique_vehicle_count": 3, "top_vehicles": []},
            },
            "qualified_inversion_selectors": [
                {
                    "pair": "USD_JPY",
                    "shape": "trend_continuation",
                    "source_side": "SHORT",
                    "selected_side": "LONG",
                    "side": "LONG",
                    "exit_shape": "tp1_sl1",
                    "feature_a": "atr_regime:mid",
                    "feature_b": "session:ny",
                    "qualification": "PASS",
                    "train_n": 48,
                    "train_win_rate": 0.6,
                    "validation_n": 20,
                    "validation_win_rate": 0.65,
                    "validation_win_wilson95_lower": 0.432851,
                    "validation_avg_realized_pips": 3.2,
                    "validation_avg_realized_atr": 0.321626,
                    "validation_profit_factor": 2.172566,
                    "validation_inversion_edge_atr": 1.221626,
                    "source_validation_avg_realized_atr": -0.9,
                    "active_days": 8,
                    "positive_day_rate": 0.75,
                    "blockers": [],
                    "all_n": 68,
                }
            ],
        }

        packaged = packager.package_payload(payload, source_report=Path("latest.json"))

        row = packaged["qualified_inversion_selectors"][0]
        self.assertEqual(packaged["summary"]["qualified_inversion_selector_count"], 1)
        self.assertEqual(packaged["summary"]["high_precision_inversion_selector_count"], 0)
        self.assertEqual(packaged["config"]["inversion_selector_confluence_sizes"], [2, 3])
        self.assertEqual(
            packaged["campaign_firepower"]["status"],
            "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
        )
        self.assertEqual(packaged["campaign_firepower"]["high_precision"]["unique_vehicle_count"], 3)
        self.assertEqual(row["source_side"], "SHORT")
        self.assertEqual(row["selected_side"], "LONG")
        self.assertEqual(row["validation_inversion_edge_atr"], 1.221626)
        self.assertEqual(row["source_validation_avg_realized_atr"], -0.9)
        self.assertNotIn("blockers", row)
        self.assertNotIn("all_n", row)

    def test_merge_payloads_combines_pair_shard_reports_for_packaging(self) -> None:
        first = {
            "generated_at_utc": "2026-06-23T08:00:00Z",
            "history_files": 1,
            "history_pairs": 1,
            "history_files_discovered": 2,
            "history_pairs_discovered": 2,
            "history_pairs_discovered_order": ["AUD_USD", "USD_JPY"],
            "history_pair_selection": {"selected_pairs": ["AUD_USD"]},
            "scored_outcomes": 100,
            "inversion_scored_outcomes": 100,
            "high_precision_multi_confluence_count": 1,
            "qualified_multi_confluence_count": 4,
            "config": {
                "min_positive_day_rate": 0.55,
                "multi_confluence_sizes": [3],
            },
            "high_precision_multi_confluences": [
                {
                    "pair": "AUD_USD",
                    "side": "LONG",
                    "shape": "range_reversion",
                    "exit_shape": "tp1_sl1",
                    "confluence": "session:ny + atr_regime:high + body:flat",
                    "confluence_size": 3,
                    "validation_n": 20,
                    "validation_win_rate": 0.75,
                    "validation_avg_realized_atr": 0.6,
                }
            ],
            "campaign_firepower": {
                "high_precision": {
                    "top_vehicles": [
                        {
                            "vehicle_key": "AUD_USD|LONG|range_reversion|tp1_sl1",
                            "pair": "AUD_USD",
                            "side": "LONG",
                            "shape": "range_reversion",
                            "exit_shape": "tp1_sl1",
                            "confluence": "session:ny + atr_regime:high + body:flat",
                            "estimated_return_pct_per_active_day_at_observed_frequency": 3.0,
                            "estimated_return_pct_per_trade_at_risk_lens": 0.6,
                            "observed_attempts_per_active_day": 5.0,
                        }
                    ]
                },
                "evidence_queue": {"top_vehicles": []},
            },
        }
        second = {
            "generated_at_utc": "2026-06-23T08:01:00Z",
            "history_files": 1,
            "history_pairs": 1,
            "history_files_discovered": 2,
            "history_pairs_discovered": 2,
            "history_pairs_discovered_order": ["AUD_USD", "USD_JPY"],
            "history_pair_selection": {"selected_pairs": ["USD_JPY"]},
            "scored_outcomes": 200,
            "inversion_scored_outcomes": 200,
            "high_precision_multi_confluence_count": 1,
            "qualified_multi_confluence_count": 5,
            "config": {
                "min_validation_win_rate": 0.52,
                "multi_confluence_sizes": [4],
            },
            "high_precision_multi_confluences": [
                {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "shape": "pullback_continuation",
                    "exit_shape": "tp1.25_sl1",
                    "confluence": "session:london + atr_regime:mid + body:flat + wick_reject:1",
                    "confluence_size": 4,
                    "validation_n": 22,
                    "validation_win_rate": 0.77,
                    "validation_avg_realized_atr": 0.55,
                }
            ],
            "campaign_firepower": {
                "high_precision": {
                    "top_vehicles": [
                        {
                            "vehicle_key": "USD_JPY|SHORT|pullback_continuation|tp1.25_sl1",
                            "pair": "USD_JPY",
                            "side": "SHORT",
                            "shape": "pullback_continuation",
                            "exit_shape": "tp1.25_sl1",
                            "confluence": "session:london + atr_regime:mid + body:flat + wick_reject:1",
                            "estimated_return_pct_per_active_day_at_observed_frequency": 2.5,
                            "estimated_return_pct_per_trade_at_risk_lens": 0.5,
                            "observed_attempts_per_active_day": 5.0,
                        }
                    ]
                },
                "evidence_queue": {"top_vehicles": []},
            },
        }

        merged = packager.merge_payloads(
            [first, second],
            source_reports=[Path("shard0.json"), Path("shard1.json")],
        )
        packaged = packager.package_payload(merged, source_report=Path("merged.json"))

        self.assertEqual(merged["history_pairs"], 2)
        self.assertEqual(merged["history_pairs_discovered"], 2)
        self.assertEqual(merged["history_pair_selection"]["selected_pairs"], ["AUD_USD", "USD_JPY"])
        self.assertEqual(merged["scored_outcomes"], 300)
        self.assertEqual(merged["high_precision_multi_confluence_count"], 2)
        self.assertEqual(merged["qualified_multi_confluence_count"], 9)
        self.assertEqual(merged["config"]["multi_confluence_sizes"], [3, 4])
        self.assertEqual(merged["campaign_firepower"]["status"], "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED")
        self.assertEqual(
            merged["campaign_firepower"]["high_precision"][
                "estimated_return_pct_per_active_day_at_observed_frequency"
            ],
            5.5,
        )
        self.assertEqual(packaged["summary"]["history_pairs_discovered"], 2)
        self.assertEqual(
            packaged["summary"]["history_pair_selection"]["selected_pairs"],
            ["AUD_USD", "USD_JPY"],
        )
        self.assertEqual(
            {row["pair"] for row in packaged["high_precision_multi_confluences"]},
            {"AUD_USD", "USD_JPY"},
        )

    def test_preserves_existing_rule_rows_when_latest_report_is_top_n_excerpt(self) -> None:
        packaged = packager.package_payload(
            {
                "high_precision_multi_confluence_count": 126,
                "high_precision_multi_confluences": [
                    {
                        "pair": "USD_JPY",
                        "side": "LONG",
                        "shape": "trend_continuation",
                        "validation_n": 50,
                    }
                ],
            },
            source_report=Path("latest.json"),
        )
        existing = {
            "high_precision_multi_confluences": [
                {"pair": "USD_JPY", "side": "LONG", "shape": "trend_continuation"},
                {"pair": "GBP_JPY", "side": "SHORT", "shape": "range_rotation"},
            ]
        }

        packager.preserve_existing_rule_rows(packaged, existing)

        rows = packaged["high_precision_multi_confluences"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["pair"], "GBP_JPY")

    def test_merges_focused_rows_when_latest_report_is_narrower(self) -> None:
        packaged = packager.package_payload(
            {
                "high_precision_multi_confluence_count": 12,
                "qualified_multi_confluence_count": 120,
                "high_precision_multi_confluences": [
                    {
                        "pair": "AUD_USD",
                        "side": "SHORT",
                        "shape": "range_reversion",
                        "exit_shape": "tp1p25_sl1",
                        "feature_a": "session:rollover",
                        "validation_n": 15,
                        "validation_win_rate": 0.8667,
                    }
                ],
            },
            source_report=Path("focused.json"),
        )
        existing = {
            "summary": {
                "high_precision_multi_confluence_count": 149,
                "qualified_multi_confluence_count": 8292,
            },
            "high_precision_multi_confluences": [
                {
                    "pair": "GBP_JPY",
                    "side": "LONG",
                    "shape": "trend_continuation",
                    "exit_shape": "tp1_sl1",
                    "feature_a": "session:london_open",
                    "validation_n": 24,
                }
            ],
        }

        packager.preserve_existing_rule_rows(packaged, existing)

        rows = packaged["high_precision_multi_confluences"]
        self.assertEqual({row["pair"] for row in rows}, {"AUD_USD", "GBP_JPY"})
        self.assertEqual(rows[1]["pair"], "AUD_USD")
        self.assertEqual(rows[0]["pair"], "GBP_JPY")
        self.assertTrue(rows[0]["preserved_from_existing_packaged_artifact"])
        self.assertTrue(rows[0]["preserved_because_narrow_source"])
        self.assertNotIn("preserved_from_existing_packaged_artifact", rows[1])

    def test_does_not_preserve_existing_rule_rows_when_summary_confirms_smaller_universe(self) -> None:
        packaged = packager.package_payload(
            {
                "high_precision_multi_confluence_count": 1,
                "high_precision_multi_confluences": [
                    {
                        "pair": "USD_JPY",
                        "side": "LONG",
                        "shape": "trend_continuation",
                        "validation_n": 50,
                    }
                ],
            },
            source_report=Path("latest.json"),
        )
        existing = {
            "high_precision_multi_confluences": [
                {"pair": "USD_JPY", "side": "LONG", "shape": "trend_continuation"},
                {"pair": "GBP_JPY", "side": "SHORT", "shape": "range_rotation"},
            ]
        }

        packager.preserve_existing_rule_rows(packaged, existing)

        rows = packaged["high_precision_multi_confluences"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pair"], "USD_JPY")

    def test_preserves_broader_campaign_firepower_when_latest_report_is_narrower(self) -> None:
        payload = {
            "generated_at_utc": "2026-06-23T06:34:34Z",
            "history_pairs": 2,
            "scored_outcomes": 59232,
            "high_precision_multi_confluence_count": 44,
            "high_precision_pair_confluence_count": 0,
            "qualified_multi_confluence_count": 1141,
            "qualified_pair_confluence_count": 72,
            "campaign_firepower": {
                "status": "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED",
                "high_precision": {
                    "unique_vehicle_count": 8,
                    "estimated_return_pct_per_active_day_at_observed_frequency": 8.57,
                },
            },
            "high_precision_inversion_selectors": [],
        }
        existing = {
            "source_report": "logs/reports/forecast_improvement/oanda_universal_rotation_mining_latest.json",
            "campaign_firepower_source_report": "merged_oanda_universal_rotation_reports",
            "scope_metadata_source_report": "merged_oanda_universal_rotation_reports",
            "summary": {
                "high_precision_multi_confluence_count": 149,
                "high_precision_pair_confluence_count": 17,
                "qualified_multi_confluence_count": 8292,
                "qualified_pair_confluence_count": 629,
            },
            "campaign_firepower": {
                "status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
                "high_precision": {
                    "unique_vehicle_count": 24,
                    "estimated_return_pct_per_active_day_at_observed_frequency": 29.03,
                },
            },
            "high_precision_inversion_selectors": [
                {
                    "pair": "GBP_CHF",
                    "side": "SHORT",
                    "shape": "range_reversion",
                    "validation_n": 30,
                }
            ],
        }

        packaged = packager.package_payload(payload, source_report=Path("latest.json"))
        packager.preserve_existing_rule_rows(packaged, existing)
        packager.preserve_existing_campaign_firepower(packaged, existing)
        packager.preserve_existing_scope_metadata(packaged, existing)

        self.assertTrue(packaged["campaign_firepower_preserved_from_existing"])
        self.assertEqual(packaged["campaign_firepower"]["status"], "VERIFIED_TARGET_10_ROUTE_ESTIMATED")
        self.assertEqual(packaged["campaign_firepower"]["high_precision"]["unique_vehicle_count"], 24)
        self.assertEqual(
            packaged["campaign_firepower_source_report"],
            existing["campaign_firepower_source_report"],
        )
        preserved_inversion = packaged["high_precision_inversion_selectors"][0]
        self.assertEqual(preserved_inversion["pair"], "GBP_CHF")
        self.assertTrue(preserved_inversion["preserved_from_existing_packaged_artifact"])
        self.assertTrue(preserved_inversion["preserved_because_narrow_source"])
        self.assertEqual(
            preserved_inversion["preserved_from_source_report"],
            existing["source_report"],
        )
        self.assertEqual(packaged["summary"]["history_pairs"], 2)
        self.assertEqual(packaged["summary"]["high_precision_multi_confluence_count"], 149)
        self.assertEqual(packaged["summary"]["qualified_pair_confluence_count"], 629)
        self.assertTrue(packaged["scope_metadata_preserved_from_existing"])
        self.assertEqual(
            packaged["scope_metadata_source_report"],
            existing["scope_metadata_source_report"],
        )
        self.assertEqual(packaged["narrow_source_summary"]["history_pairs"], 2)

    def test_preservation_uses_original_scope_before_rule_rows_expand_counts(self) -> None:
        payload = {
            "generated_at_utc": "2026-06-23T10:09:16Z",
            "history_pairs": 2,
            "high_precision_multi_confluence_count": 44,
            "high_precision_pair_confluence_count": 0,
            "qualified_multi_confluence_count": 1139,
            "qualified_pair_confluence_count": 73,
            "campaign_firepower": {
                "status": "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED",
                "high_precision": {
                    "unique_vehicle_count": 8,
                    "estimated_return_pct_per_active_day_at_observed_frequency": 8.57,
                },
            },
            "high_precision_multi_confluences": [
                {
                    "pair": "GBP_JPY",
                    "side": "LONG",
                    "shape": "trend_continuation",
                    "exit_shape": "tp1_sl1",
                    "feature_a": "session:london_open",
                    "validation_n": 24,
                }
            ],
            "qualified_multi_confluences": [
                {
                    "pair": "GBP_JPY",
                    "side": "SHORT",
                    "shape": "range_reversion",
                    "exit_shape": "tp1_sl1",
                    "feature_a": "session:london_ny_overlap",
                    "validation_n": 21,
                }
            ],
        }
        existing = {
            "source_report": "merged_oanda_universal_rotation_reports",
            "summary": {
                "history_pairs": 8,
                "high_precision_multi_confluence_count": 44,
                "high_precision_pair_confluence_count": 6,
                "qualified_multi_confluence_count": 1684,
                "qualified_pair_confluence_count": 327,
            },
            "campaign_firepower": {
                "status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
                "high_precision": {
                    "unique_vehicle_count": 19,
                    "estimated_return_pct_per_active_day_at_observed_frequency": 26.9,
                },
            },
            "high_precision_multi_confluences": [
                {
                    "pair": "AUD_USD",
                    "side": "SHORT",
                    "shape": "range_reversion",
                    "exit_shape": "tp1_sl1",
                    "feature_a": "session:rollover",
                    "validation_n": 16,
                }
            ],
            "high_precision_pair_confluences": [
                {
                    "pair": "AUD_USD",
                    "side": "SHORT",
                    "shape": "range_reversion",
                    "exit_shape": "tp1_sl1",
                    "feature_a": "session:rollover",
                    "validation_n": 16,
                }
            ],
            "qualified_multi_confluences": [
                {
                    "pair": "AUD_USD",
                    "side": "SHORT",
                    "shape": "range_reversion",
                    "exit_shape": "tp1_sl1",
                    "feature_a": "session:rollover",
                    "validation_n": 16,
                }
            ],
            "qualified_pair_confluences": [
                {
                    "pair": "AUD_USD",
                    "side": "SHORT",
                    "shape": "range_reversion",
                    "exit_shape": "tp1_sl1",
                    "feature_a": "session:rollover",
                    "validation_n": 16,
                }
            ],
        }

        packaged = packager.package_payload(payload, source_report=Path("focused.json"))
        report_is_narrower = packager._packaged_report_is_narrower(packaged, existing)
        self.assertTrue(report_is_narrower)

        packager.preserve_existing_campaign_firepower(
            packaged,
            existing,
            report_is_narrower=report_is_narrower,
        )
        packager.preserve_existing_scope_metadata(
            packaged,
            existing,
            report_is_narrower=report_is_narrower,
        )
        packager.preserve_existing_rule_rows(
            packaged,
            existing,
            report_is_narrower=report_is_narrower,
        )

        self.assertTrue(packaged["campaign_firepower_preserved_from_existing"])
        self.assertEqual(packaged["campaign_firepower"]["status"], "VERIFIED_TARGET_10_ROUTE_ESTIMATED")
        self.assertEqual(packaged["campaign_firepower"]["high_precision"]["unique_vehicle_count"], 19)
        self.assertTrue(packaged["scope_metadata_preserved_from_existing"])
        self.assertEqual(packaged["summary"]["history_pairs"], 8)
        self.assertEqual(packaged["narrow_source_summary"]["history_pairs"], 2)
        self.assertEqual(
            {row["pair"] for row in packaged["high_precision_multi_confluences"]},
            {"AUD_USD", "GBP_JPY"},
        )
        self.assertEqual(packaged["summary"]["high_precision_pair_confluence_count"], 6)

    def test_preserves_broader_campaign_evidence_queue_when_high_precision_is_unchanged(self) -> None:
        payload = {
            "generated_at_utc": "2026-06-23T13:03:24Z",
            "high_precision_multi_confluence_count": 155,
            "high_precision_pair_confluence_count": 6,
            "qualified_multi_confluence_count": 5302,
            "qualified_pair_confluence_count": 327,
            "campaign_firepower": {
                "status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
                "per_trade_risk_pct_lens": 0.35,
                "high_precision": {
                    "top_vehicles": [
                        {
                            "vehicle_key": "USD_JPY|LONG|range_reversion|tp1_sl1",
                            "pair": "USD_JPY",
                            "side": "LONG",
                            "shape": "range_reversion",
                            "exit_shape": "tp1_sl1",
                            "estimated_return_pct_per_active_day_at_observed_frequency": 12.0,
                            "estimated_return_pct_per_trade_at_risk_lens": 0.8,
                            "observed_attempts_per_active_day": 15.0,
                        }
                    ]
                },
                "evidence_queue": {
                    "top_vehicles": [
                        {
                            "vehicle_key": "USD_JPY|SHORT|range_reversion|tp1_sl1",
                            "pair": "USD_JPY",
                            "side": "SHORT",
                            "shape": "range_reversion",
                            "exit_shape": "tp1_sl1",
                            "estimated_return_pct_per_active_day_at_observed_frequency": 1.2,
                            "estimated_return_pct_per_trade_at_risk_lens": 0.4,
                            "observed_attempts_per_active_day": 3.0,
                        }
                    ]
                },
            },
        }
        existing = {
            "source_report": "merged_oanda_universal_rotation_reports",
            "summary": {
                "high_precision_multi_confluence_count": 149,
                "high_precision_pair_confluence_count": 6,
                "qualified_multi_confluence_count": 8292,
                "qualified_pair_confluence_count": 629,
            },
            "campaign_firepower": {
                "status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
                "high_precision": {
                    "top_vehicles": [
                        {
                            "vehicle_key": "USD_JPY|LONG|range_reversion|tp1_sl1",
                            "pair": "USD_JPY",
                            "side": "LONG",
                            "shape": "range_reversion",
                            "exit_shape": "tp1_sl1",
                            "estimated_return_pct_per_active_day_at_observed_frequency": 10.0,
                            "estimated_return_pct_per_trade_at_risk_lens": 0.7,
                            "observed_attempts_per_active_day": 14.0,
                        }
                    ]
                },
                "evidence_queue": {
                    "top_vehicles": [
                        {
                            "vehicle_key": "USD_JPY|SHORT|range_reversion|tp1_sl1",
                            "pair": "USD_JPY",
                            "side": "SHORT",
                            "shape": "range_reversion",
                            "exit_shape": "tp1_sl1",
                            "estimated_return_pct_per_active_day_at_observed_frequency": 1.0,
                            "estimated_return_pct_per_trade_at_risk_lens": 0.3,
                            "observed_attempts_per_active_day": 3.0,
                        },
                        {
                            "vehicle_key": "GBP_USD|SHORT|range_reversion|tp1_sl1",
                            "pair": "GBP_USD",
                            "side": "SHORT",
                            "shape": "range_reversion",
                            "exit_shape": "tp1_sl1",
                            "estimated_return_pct_per_active_day_at_observed_frequency": 1.4,
                            "estimated_return_pct_per_trade_at_risk_lens": 0.35,
                            "observed_attempts_per_active_day": 4.0,
                        },
                        {
                            "vehicle_key": "AUD_USD|SHORT|range_reversion|tp1_sl1",
                            "pair": "AUD_USD",
                            "side": "SHORT",
                            "shape": "range_reversion",
                            "exit_shape": "tp1_sl1",
                            "estimated_return_pct_per_active_day_at_observed_frequency": 1.6,
                            "estimated_return_pct_per_trade_at_risk_lens": 0.4,
                            "observed_attempts_per_active_day": 4.0,
                        },
                    ]
                },
            },
        }

        packaged = packager.package_payload(payload, source_report=Path("focused.json"))
        report_is_narrower = packager._packaged_report_is_narrower(packaged, existing)
        self.assertTrue(report_is_narrower)

        packager.preserve_existing_campaign_firepower(
            packaged,
            existing,
            report_is_narrower=report_is_narrower,
        )

        self.assertTrue(packaged["campaign_firepower_preserved_from_existing"])
        self.assertEqual(packaged["campaign_firepower"]["high_precision"]["unique_vehicle_count"], 1)
        self.assertEqual(packaged["campaign_firepower"]["evidence_queue"]["unique_vehicle_count"], 3)
        self.assertEqual(packaged["campaign_firepower"]["evidence_queue"]["pair_count"], 3)
        self.assertEqual(packaged["campaign_firepower"]["per_trade_risk_pct_lens"], 0.35)
        self.assertEqual(
            {row["pair"] for row in packaged["campaign_firepower"]["evidence_queue"]["top_vehicles"]},
            {"AUD_USD", "GBP_USD", "USD_JPY"},
        )

    def test_preserves_broader_config_when_latest_report_is_narrower(self) -> None:
        packaged = packager.package_payload(
            {
                "high_precision_multi_confluence_count": 12,
                "qualified_multi_confluence_count": 120,
                "config": {
                    "min_positive_day_rate": 0.6,
                    "multi_confluence_sizes": [3],
                    "inversion_selector_confluence_sizes": [2],
                },
                "high_precision_multi_confluences": [],
            },
            source_report=Path("focused.json"),
        )
        existing = {
            "source_report": "logs/reports/forecast_improvement/oanda_universal_rotation_mining_latest.json",
            "summary": {
                "high_precision_multi_confluence_count": 149,
                "qualified_multi_confluence_count": 8292,
            },
            "config": {
                "min_positive_day_rate": 0.55,
                "min_validation_win_rate": 0.52,
                "multi_confluence_sizes": [3, 4],
                "inversion_selector_confluence_sizes": [2, 3],
            },
        }

        packager.preserve_existing_scope_metadata(packaged, existing)

        self.assertEqual(packaged["config"]["min_positive_day_rate"], 0.55)
        self.assertEqual(packaged["config"]["min_validation_win_rate"], 0.52)
        self.assertEqual(packaged["config"]["multi_confluence_sizes"], [3, 4])
        self.assertEqual(packaged["config"]["inversion_selector_confluence_sizes"], [2, 3])
        self.assertEqual(packaged["narrow_source_config"]["multi_confluence_sizes"], [3])
        self.assertEqual(
            packaged["scope_metadata_source_report"],
            existing["source_report"],
        )

    def test_uses_latest_campaign_firepower_when_report_is_not_narrower(self) -> None:
        payload = {
            "generated_at_utc": "2026-06-23T06:34:34Z",
            "high_precision_multi_confluence_count": 200,
            "qualified_multi_confluence_count": 9000,
            "campaign_firepower": {
                "status": "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED",
                "high_precision": {"unique_vehicle_count": 8},
            },
        }
        existing = {
            "summary": {
                "high_precision_multi_confluence_count": 149,
                "qualified_multi_confluence_count": 8292,
            },
            "campaign_firepower": {
                "status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
                "high_precision": {"unique_vehicle_count": 24},
            },
        }

        packaged = packager.package_payload(payload, source_report=Path("latest.json"))
        packager.preserve_existing_campaign_firepower(packaged, existing)

        self.assertNotIn("campaign_firepower_preserved_from_existing", packaged)
        self.assertEqual(packaged["campaign_firepower"]["status"], "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED")

    def test_preserves_section_rows_when_selector_scope_is_narrower(self) -> None:
        packaged = packager.package_payload(
            {
                "high_precision_multi_confluence_count": 200,
                "qualified_multi_confluence_count": 9000,
                "qualified_inversion_selector_count": 0,
                "qualified_inversion_selectors": [],
                "campaign_firepower": {
                    "status": "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED",
                    "high_precision": {"unique_vehicle_count": 8},
                },
            },
            source_report=Path("merged_shards.json"),
        )
        existing = {
            "summary": {
                "high_precision_multi_confluence_count": 44,
                "qualified_multi_confluence_count": 1385,
                "qualified_inversion_selector_count": 1,
            },
            "qualified_inversion_selectors": [
                {
                    "pair": "EUR_USD",
                    "source_side": "SHORT",
                    "selected_side": "LONG",
                    "shape": "trend_continuation",
                    "exit_shape": "tp1_sl1",
                    "validation_inversion_edge_atr": 0.4,
                }
            ],
            "campaign_firepower": {
                "status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
                "high_precision": {"unique_vehicle_count": 24},
            },
        }

        packager.preserve_existing_rule_rows(packaged, existing)
        packager.preserve_existing_campaign_firepower(packaged, existing)

        self.assertEqual(len(packaged["qualified_inversion_selectors"]), 1)
        self.assertEqual(packaged["qualified_inversion_selectors"][0]["pair"], "EUR_USD")
        self.assertEqual(packaged["summary"]["qualified_inversion_selector_count"], 1)
        self.assertNotIn("campaign_firepower_preserved_from_existing", packaged)
        self.assertEqual(packaged["campaign_firepower"]["status"], "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED")


if __name__ == "__main__":
    unittest.main()

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

        self.assertTrue(packaged["campaign_firepower_preserved_from_existing"])
        self.assertEqual(packaged["campaign_firepower"]["status"], "VERIFIED_TARGET_10_ROUTE_ESTIMATED")
        self.assertEqual(packaged["campaign_firepower"]["high_precision"]["unique_vehicle_count"], 24)
        self.assertEqual(packaged["campaign_firepower_source_report"], existing["source_report"])
        self.assertEqual(packaged["high_precision_inversion_selectors"], existing["high_precision_inversion_selectors"])
        self.assertEqual(packaged["summary"]["history_pairs"], 2)

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


if __name__ == "__main__":
    unittest.main()

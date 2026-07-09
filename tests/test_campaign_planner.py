from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quant_rabbit.strategy.ensemble import CampaignPlanner


def _write_oanda_firepower_report(root: Path, *, status: str = "VERIFIED_TARGET_10_ROUTE_ESTIMATED") -> Path:
    path = root / "oanda_universal_rotation_mining_latest.json"
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-06-21T00:00:00Z",
                "campaign_firepower": {
                    "status": status,
                    "high_precision": {
                        "unique_vehicle_count": 3,
                        "pair_count": 2,
                        "estimated_return_pct_per_active_day_at_observed_frequency": 11.0,
                        "top_vehicles": [
                            {
                                "vehicle_key": "GBP_USD|SHORT|range_reversion|tp1_sl1",
                                "pair": "GBP_USD",
                                "shape": "range_reversion",
                                "firepower_side": "SHORT",
                                "exit_shape": "tp1_sl1",
                                "validation_n": 30,
                                "validation_win_rate": 0.70,
                                "validation_win_wilson95_lower": 0.52,
                                "validation_profit_factor": 2.46,
                                "estimated_return_pct_per_active_day_at_observed_frequency": 1.43,
                                "live_permission": False,
                            },
                            {
                                "vehicle_key": "USD_JPY|LONG|range_reversion|tp1.25_sl1",
                                "pair": "USD_JPY",
                                "shape": "range_reversion",
                                "firepower_side": "LONG",
                                "exit_shape": "tp1.25_sl1",
                                "validation_n": 14,
                                "validation_win_rate": 0.86,
                                "validation_win_wilson95_lower": 0.60,
                                "validation_profit_factor": 8.89,
                                "estimated_return_pct_per_active_day_at_observed_frequency": 1.62,
                                "live_permission": False,
                            },
                            {
                                "vehicle_key": "USD_JPY|LONG|range_reversion|tp1_sl1",
                                "pair": "USD_JPY",
                                "shape": "range_reversion",
                                "firepower_side": "LONG",
                                "exit_shape": "tp1_sl1",
                                "validation_n": 22,
                                "validation_win_rate": 0.76,
                                "validation_win_wilson95_lower": 0.54,
                                "validation_profit_factor": 3.12,
                                "estimated_return_pct_per_active_day_at_observed_frequency": 0.80,
                                "live_permission": False,
                            },
                        ],
                    },
                },
            }
        )
    )
    return path


class CampaignPlannerTest(unittest.TestCase):
    def test_builds_multi_desk_plan_without_live_guarantee(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy = root / "strategy.json"
            story = root / "story.json"
            report = root / "campaign.md"
            plan_path = root / "campaign.json"
            strategy.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "status": "RISK_REPAIR_CANDIDATE",
                                "required_fix": "old sizing broke loss cap",
                                "pretrade_net_jpy": 1000,
                                "live_net_jpy": 500,
                                "live_worst_jpy": -700,
                                "positive_best_jpy": 2500,
                                "positive_tail_jpy": 1800,
                                "target_reward_risk": 3.6,
                            },
                            {
                                "pair": "USD_JPY",
                                "direction": "LONG",
                                "status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                "required_fix": "bad history",
                                "pretrade_net_jpy": -100,
                                "live_net_jpy": -1000,
                                "live_worst_jpy": -900,
                            },
                        ]
                    }
                )
            )
            story.write_text(
                json.dumps(
                    {
                        "pair_profiles": [
                            {
                                "pair": "EUR_USD",
                                "methods": {"TREND_CONTINUATION": 4, "RANGE_ROTATION": 2, "EVENT_RISK": 1},
                                "themes": {"momentum": 3, "event_risk": 1},
                                "examples": ["quality_audit: green staircase"],
                            },
                            {
                                "pair": "USD_JPY",
                                "methods": {"BREAKOUT_FAILURE": 3, "EVENT_RISK": 5},
                                "themes": {"intervention": 5},
                                "examples": ["news_digest: intervention risk"],
                            },
                        ]
                    }
                )
            )

            summary = CampaignPlanner(
                strategy_profile=strategy,
                market_story_profile=story,
                report_path=report,
                plan_path=plan_path,
            ).run(start_balance_jpy=200_000)

            self.assertEqual(summary.target_jpy, 20_000)
            self.assertGreaterEqual(summary.lanes, 4)
            payload = json.loads(plan_path.read_text())
            adoptions = {lane["adoption"] for lane in payload["lanes"]}
            self.assertIn("RISK_REPAIR_DRY_RUN", adoptions)
            self.assertIn("RISK_OVERLAY", adoptions)
            self.assertIn("REJECTED", adoptions)
            runner = next(lane for lane in payload["lanes"] if lane["pair"] == "EUR_USD")
            self.assertGreaterEqual(runner["target_reward_risk"], 3.6)
            self.assertIn("RUNNER", runner["campaign_role"])
            self.assertIn("target, not a profit guarantee", payload["operating_rule"])
            self.assertIn("Portfolio Director Rules", report.read_text())

    def test_prioritizes_high_tail_and_missed_pressure_within_same_adoption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy = root / "strategy.json"
            story = root / "story.json"
            plan_path = root / "campaign.json"
            strategy.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "direction": "SHORT",
                                "status": "CANDIDATE",
                                "required_fix": "ready",
                                "positive_best_jpy": 2992.49,
                                "positive_tail_jpy": 1213.22,
                                "seat_missed": 94,
                            },
                            {
                                "pair": "GBP_USD",
                                "direction": "LONG",
                                "status": "CANDIDATE",
                                "required_fix": "ready",
                                "positive_best_jpy": 4020.0,
                                "positive_tail_jpy": 2259.0,
                                "seat_missed": 46,
                            },
                        ]
                    }
                )
            )
            story.write_text(
                json.dumps(
                    {
                        "pair_profiles": [
                            {"pair": "EUR_USD", "methods": {"TREND_CONTINUATION": 2}, "themes": {}, "examples": []},
                            {"pair": "GBP_USD", "methods": {"TREND_CONTINUATION": 2}, "themes": {}, "examples": []},
                        ]
                    }
                )
            )

            CampaignPlanner(
                strategy_profile=strategy,
                market_story_profile=story,
                report_path=root / "campaign.md",
                plan_path=plan_path,
            ).run(start_balance_jpy=200_000)

            first = json.loads(plan_path.read_text())["lanes"][0]
            self.assertEqual(first["pair"], "EUR_USD")
            self.assertEqual(first["direction"], "SHORT")
            self.assertEqual(first["seat_missed"], 94)
            self.assertGreater(first["missed_reward_pressure_jpy"], 114000)

    def test_strategy_profile_candidate_without_market_story_still_builds_order_intents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy = root / "strategy.json"
            story = root / "story.json"
            plan_path = root / "campaign.json"
            strategy.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "AUD_CAD",
                                "direction": "LONG",
                                "status": "CANDIDATE",
                                "required_fix": "current execution ledger is positive",
                                "live_net_jpy": 1104.4,
                                "positive_best_jpy": 535.2,
                                "positive_tail_jpy": 535.2,
                            },
                            {
                                "pair": "NZD_CHF",
                                "direction": "LONG",
                                "status": "WATCH_ONLY",
                                "required_fix": "mixed evidence",
                            },
                        ]
                    }
                )
            )
            story.write_text(json.dumps({"pair_profiles": []}))

            summary = CampaignPlanner(
                strategy_profile=strategy,
                market_story_profile=story,
                report_path=root / "campaign.md",
                plan_path=plan_path,
            ).run(start_balance_jpy=200_000)

            payload = json.loads(plan_path.read_text())
            aud_lanes = [lane for lane in payload["lanes"] if lane["pair"] == "AUD_CAD"]
            self.assertEqual(summary.actionable_lanes, 3)
            self.assertEqual({lane["adoption"] for lane in aud_lanes}, {"ORDER_INTENT_REQUIRED"})
            self.assertEqual(
                {lane["method"] for lane in aud_lanes},
                {"RANGE_ROTATION", "BREAKOUT_FAILURE", "TREND_CONTINUATION"},
            )
            self.assertTrue(
                all("strategy_profile direct AUD_CAD LONG" in lane["story_examples"][0] for lane in aud_lanes)
            )
            self.assertFalse(any(lane["pair"] == "NZD_CHF" for lane in payload["lanes"]))

    def test_seeds_verified_oanda_firepower_lanes_ahead_of_story_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy = root / "strategy.json"
            story = root / "story.json"
            plan_path = root / "campaign.json"
            oanda = _write_oanda_firepower_report(root)
            strategy.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "direction": "SHORT",
                                "status": "CANDIDATE",
                                "required_fix": "ready",
                                "positive_best_jpy": 2992.49,
                                "positive_tail_jpy": 1213.22,
                                "seat_missed": 94,
                            }
                        ]
                    }
                )
            )
            story.write_text(
                json.dumps(
                    {
                        "pair_profiles": [
                            {"pair": "EUR_USD", "methods": {"TREND_CONTINUATION": 2}, "themes": {}, "examples": []}
                        ]
                    }
                )
            )

            CampaignPlanner(
                strategy_profile=strategy,
                market_story_profile=story,
                report_path=root / "campaign.md",
                plan_path=plan_path,
                oanda_rotation_mining=oanda,
            ).run(start_balance_jpy=200_000)

            lanes = json.loads(plan_path.read_text())["lanes"]
            first = lanes[0]
            self.assertTrue(first["oanda_campaign_firepower_seed"])
            self.assertEqual(first["pair"], "USD_JPY")
            self.assertEqual(first["direction"], "LONG")
            self.assertEqual(first["desk"], "range_trader")
            self.assertEqual(first["method"], "RANGE_ROTATION")
            self.assertEqual(first["campaign_role"], "OANDA_FIREPOWER_ROUTE")
            self.assertEqual(first["target_reward_risk"], 1.25)
            self.assertEqual(first["oanda_campaign_vehicle_count"], 2)
            self.assertEqual(
                first["oanda_campaign_vehicle_keys"],
                [
                    "USD_JPY|LONG|range_reversion|tp1.25_sl1",
                    "USD_JPY|LONG|range_reversion|tp1_sl1",
                ],
            )
            self.assertEqual(first["oanda_campaign_exit_shapes"], ["tp1.25_sl1", "tp1_sl1"])
            self.assertAlmostEqual(first["oanda_campaign_estimated_return_pct_per_active_day"], 2.42)
            self.assertFalse(first["oanda_campaign_live_permission"])
            self.assertTrue(any(lane["pair"] == "GBP_USD" for lane in lanes))

    def test_default_plan_uses_packaged_oanda_firepower_when_latest_log_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy = root / "strategy.json"
            story = root / "story.json"
            plan_path = root / "daily_campaign_plan.json"
            missing_latest = root / "logs" / "missing_oanda_latest.json"
            packaged = _write_oanda_firepower_report(root)
            strategy.write_text(json.dumps({"profiles": []}))
            story.write_text(json.dumps({"pair_profiles": []}))

            with mock.patch(
                "quant_rabbit.strategy.ensemble.DEFAULT_CAMPAIGN_PLAN",
                plan_path,
            ), mock.patch(
                "quant_rabbit.strategy.ensemble.DEFAULT_OANDA_UNIVERSAL_ROTATION_MINING",
                missing_latest,
            ), mock.patch(
                "quant_rabbit.strategy.ensemble.DEFAULT_OANDA_UNIVERSAL_ROTATION_PACKAGED_RULES",
                packaged,
            ):
                CampaignPlanner(
                    strategy_profile=strategy,
                    market_story_profile=story,
                    report_path=root / "campaign.md",
                    plan_path=plan_path,
                ).run(start_balance_jpy=200_000)

            first = json.loads(plan_path.read_text())["lanes"][0]
            self.assertTrue(first["oanda_campaign_firepower_seed"])
            self.assertEqual(first["pair"], "USD_JPY")
            self.assertEqual(first["direction"], "LONG")


if __name__ == "__main__":
    unittest.main()

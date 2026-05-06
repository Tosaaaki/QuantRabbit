from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.strategy.ensemble import CampaignPlanner


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


if __name__ == "__main__":
    unittest.main()

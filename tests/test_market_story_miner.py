from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.strategy.market_story import MarketStoryMiner


class MarketStoryMinerTest(unittest.TestCase):
    def test_mines_narrative_chart_story_and_method_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "legacy"
            logs = archive / "logs"
            collab = archive / "collab_trade"
            daily = collab / "daily" / "2026-04-30"
            logs.mkdir(parents=True)
            daily.mkdir(parents=True)
            collab.mkdir(exist_ok=True)

            (logs / "news_digest.md").write_text(
                "BOJ rate check: USD/JPY intervention risk. NFP spread window blocks tight SLs.\n"
                "EUR/USD follows FOMC split and USD softness.\n"
                "ECB/GDP collision is an event label, not a currency pair.\n"
            )
            (logs / "quality_audit.md").write_text(
                "EUR_USD chart tells me: clean green staircase into upper band = trend continuation.\n"
                "AUD_JPY story: upper rail range rotation only if the box holds.\n"
            )
            (collab / "state.md").write_text(
                "Market Narrative: USD_JPY live risk is unprotected; POSITION_MANAGEMENT before fresh entries.\n"
            )
            (daily / "state.md").write_text(
                "GBP_USD breakout failure requires rejection price and trapped side.\n"
                "NZD_USD range rail rotation is valid only at the box edge.\n"
            )

            report = root / "market_story.md"
            profile = root / "market_story.json"
            summary = MarketStoryMiner(archive, report, profile).run()

            self.assertEqual(summary.artifacts, 4)
            self.assertGreaterEqual(summary.story_lines, 4)
            payload = json.loads(profile.read_text())
            pairs = {item["pair"]: item for item in payload["pair_profiles"]}
            self.assertIn("EUR_USD", pairs)
            self.assertNotIn("ECB_GDP", pairs)
            self.assertIn("TREND_CONTINUATION", pairs["EUR_USD"]["methods"])
            self.assertIn("AUD_JPY", pairs)
            self.assertIn("RANGE_ROTATION", pairs["AUD_JPY"]["methods"])
            self.assertIn("NZD_USD", pairs)
            self.assertIn("RANGE_ROTATION", pairs["NZD_USD"]["methods"])
            self.assertIn("Method Switching Contract", report.read_text())

    def test_news_root_examples_precede_legacy_archive_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "legacy"
            archive_logs = archive / "logs"
            live_logs = root / "logs"
            archive_logs.mkdir(parents=True)
            live_logs.mkdir(parents=True)

            (archive_logs / "news_digest.md").write_text(
                "EUR_USD old archive breakout failure narrative.\n",
                encoding="utf-8",
            )
            (live_logs / "news_digest.md").write_text(
                "EUR_USD current live risk-on range retest narrative.\n",
                encoding="utf-8",
            )

            profile = root / "market_story.json"
            report = root / "market_story.md"
            MarketStoryMiner(archive=archive, report_path=report, profile_path=profile, news_root=live_logs).run()

            payload = json.loads(profile.read_text())
            pairs = {item["pair"]: item for item in payload["pair_profiles"]}
            self.assertTrue(pairs["EUR_USD"]["examples"][0].startswith("news_digest: EUR_USD current live"))
            artifact_lines = [line for line in report.read_text().splitlines() if line.startswith("- `")]
            self.assertTrue(artifact_lines[0].startswith("- `news/news_digest.md`"))


if __name__ == "__main__":
    unittest.main()

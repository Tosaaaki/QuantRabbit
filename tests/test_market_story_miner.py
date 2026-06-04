from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.strategy.market_story import MarketStoryMiner


def _build_history_db(db_path: Path, day_pnl: dict[str, float]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE legacy_records (
                source_table TEXT NOT NULL,
                source_id TEXT,
                session_date TEXT,
                pair TEXT,
                direction TEXT,
                pl REAL,
                execution_style TEXT,
                allocation_band TEXT,
                thesis TEXT,
                raw_json TEXT NOT NULL
            )
            """
        )
        for date, total in day_pnl.items():
            conn.execute(
                "INSERT INTO legacy_records (source_table, session_date, pl, raw_json) VALUES (?, ?, ?, ?)",
                ("trades", date, total, "{}"),
            )
        conn.commit()


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
            self.assertTrue(pairs["EUR_USD"]["examples"][0].startswith("news_digest [day=? pnl=?]: EUR_USD current live"))
            artifact_lines = [line for line in report.read_text().splitlines() if line.startswith("- `")]
            self.assertTrue(artifact_lines[0].startswith("- `news/news_digest.md`"))

    def test_examples_carry_outcome_attribution_and_date_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "legacy"
            logs = archive / "logs"
            daily_win = archive / "collab_trade" / "daily" / "2026-04-07"
            daily_loss = archive / "collab_trade" / "daily" / "2026-03-30"
            logs.mkdir(parents=True)
            daily_win.mkdir(parents=True)
            daily_loss.mkdir(parents=True)

            (daily_win / "state.md").write_text(
                "AUD_JPY LONG trend continuation: H4 ADX=43, M5 BB lower buy on staircase pullback.\n"
            )
            (daily_loss / "state.md").write_text(
                "AUD_JPY range rotation Edge C | Allocation C | Why: late chase rejection failed.\n"
            )
            (logs / "news_digest.md").write_text(
                "AUD_JPY general intervention risk discussion without a dated line.\n"
            )
            (logs / "quality_audit.md").write_text(
                "## 2026-04-07 audit\n"
                "AUD_JPY LONG | Edge B | Allocation B | Why: clean staircase, trend continuation.\n"
                "AUD_JPY LONG | Edge B | Allocation B | Why: duplicate same day should not fill every slot.\n"
                "## 2026-04-19 audit\n"
                "AUD_JPY WAIT | Edge C | Allocation C | Why: bounce active, fresh risk skipped.\n"
            )

            db_path = root / "history.db"
            _build_history_db(
                db_path,
                {
                    "2026-04-07": 11493.0,
                    "2026-03-30": -22360.0,
                    "2026-04-19": -654.0,
                },
            )

            profile = root / "story.json"
            report = root / "story.md"
            MarketStoryMiner(archive, report, profile, db_path=db_path).run()

            payload = json.loads(profile.read_text())
            pairs = {item["pair"]: item for item in payload["pair_profiles"]}
            examples = pairs["AUD_JPY"]["examples"]

            self.assertTrue(all(example.startswith(("daily_state [day=", "quality_audit [day=", "news_digest [day=")) for example in examples))
            self.assertIn("[day=2026-03-30 pnl=-22360]", examples[0])
            self.assertTrue(any("[day=2026-04-07 pnl=+11493]" in example for example in examples))
            dated_days = [
                example.split("[day=", 1)[1].split(" ", 1)[0]
                for example in examples
                if "[day=?" not in example
            ]
            self.assertEqual(len(dated_days), len(set(dated_days)))
            self.assertIn("Edge B | Allocation B", " ".join(examples))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.paths import effective_oanda_universal_rotation_path


class OandaUniversalRotationPathTest(unittest.TestCase):
    def test_preserved_packaged_scope_wins_over_newer_narrow_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = (
                root
                / "logs"
                / "reports"
                / "forecast_improvement"
                / "oanda_universal_rotation_mining_latest.json"
            )
            packaged = root / "src" / "quant_rabbit" / "oanda_universal_rotation_precision_rules.json"
            latest.parent.mkdir(parents=True)
            packaged.parent.mkdir(parents=True)
            latest.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-25T01:15:00Z",
                        "history_pair_selection": {
                            "selected_pair_count": 2,
                            "selected_pairs": ["EUR_JPY", "USD_JPY"],
                        },
                        "history_files": 2,
                        "history_pairs": 2,
                        "scored_outcomes": 20_000,
                        "summary": {"scored_outcomes": 20_000},
                    }
                ),
                encoding="utf-8",
            )
            packaged.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-25T00:51:00Z",
                        "source_report": (
                            "logs/reports/forecast_improvement/"
                            "oanda_universal_rotation_mining_latest.json"
                        ),
                        "scope_metadata_preserved_from_existing": True,
                        "summary": {
                            "history_pair_selection": {
                                "selected_pair_count": 6,
                                "selected_pairs": [
                                    "AUD_USD",
                                    "EUR_JPY",
                                    "GBP_JPY",
                                    "GBP_USD",
                                    "NZD_USD",
                                    "USD_JPY",
                                ],
                            },
                            "history_files": 6,
                            "history_pairs": 6,
                            "scored_outcomes": 267_328,
                        },
                    }
                ),
                encoding="utf-8",
            )

            path = effective_oanda_universal_rotation_path(latest, packaged)

            self.assertEqual(path, packaged)

    def test_newer_latest_wins_when_packaged_scope_does_not_cover_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = (
                root
                / "logs"
                / "reports"
                / "forecast_improvement"
                / "oanda_universal_rotation_mining_latest.json"
            )
            packaged = root / "src" / "quant_rabbit" / "oanda_universal_rotation_precision_rules.json"
            latest.parent.mkdir(parents=True)
            packaged.parent.mkdir(parents=True)
            latest.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-25T01:15:00Z",
                        "history_pair_selection": {
                            "selected_pair_count": 2,
                            "selected_pairs": ["EUR_JPY", "CAD_JPY"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            packaged.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-25T00:51:00Z",
                        "source_report": (
                            "logs/reports/forecast_improvement/"
                            "oanda_universal_rotation_mining_latest.json"
                        ),
                        "scope_metadata_preserved_from_existing": True,
                        "summary": {
                            "history_pair_selection": {
                                "selected_pair_count": 2,
                                "selected_pairs": ["EUR_JPY", "USD_JPY"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            path = effective_oanda_universal_rotation_path(latest, packaged)

            self.assertEqual(path, latest)


if __name__ == "__main__":
    unittest.main()

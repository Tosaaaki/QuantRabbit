from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.strategy.tf_weights import dynamic_tf_weights


class DynamicTfWeightsCalendarTest(unittest.TestCase):
    def test_timestamp_utc_calendar_event_triggers_event_risk_overlay(self) -> None:
        now = datetime(2026, 6, 5, 12, 25, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "economic_calendar.json"
            path.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "timestamp_utc": "2026-06-05T12:30:00+00:00",
                                "currency": "USD",
                                "impact": "High",
                                "title": "Non-Farm Employment Change",
                            }
                        ]
                    }
                )
            )

            _, label = dynamic_tf_weights(
                session="ASIA_RANGE",
                dominant_regime="RANGE",
                method="RANGE_ROTATION",
                pair="GBP_USD",
                calendar_path=path,
                now_utc=now,
            )

        self.assertIn("NY_OVERLAP", label)
        self.assertIn("news:USD:Non-Farm Employment Change", label)


if __name__ == "__main__":
    unittest.main()

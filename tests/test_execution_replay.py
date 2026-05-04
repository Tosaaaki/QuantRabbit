from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.execution_replay import ExecutionReplayer


class ExecutionReplayerTest(unittest.TestCase):
    def test_replays_stop_entry_to_take_profit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            prices = root / "prices.json"
            intents.write_text(json.dumps({"results": [_intent()]}))
            prices.write_text(
                json.dumps(
                    {
                        "ticks": [
                            {"timestamp_utc": "2026-05-01T00:00:00+00:00", "pair": "EUR_USD", "bid": 1.0998, "ask": 1.1000},
                            {"timestamp_utc": "2026-05-01T00:00:00+00:00", "pair": "USD_JPY", "bid": 157.0, "ask": 157.01},
                            {"timestamp_utc": "2026-05-01T00:01:00+00:00", "pair": "EUR_USD", "bid": 1.1010, "ask": 1.1011},
                            {"timestamp_utc": "2026-05-01T00:01:00+00:00", "pair": "USD_JPY", "bid": 157.0, "ask": 157.01},
                        ]
                    }
                )
            )

            summary = ExecutionReplayer(
                intents_path=intents,
                price_path=prices,
                output_path=root / "execution_replay.json",
                report_path=root / "execution_replay.md",
            ).run(target_jpy=150)

            self.assertEqual(summary.status, "TARGET_HIT")
            self.assertTrue(summary.target_hit)
            self.assertEqual(summary.closed, 1)
            self.assertAlmostEqual(summary.net_pl_jpy, 157.0)
            payload = json.loads((root / "execution_replay.json").read_text())
            self.assertEqual(payload["results"][0]["exit_reason"], "TP")
            self.assertEqual(payload["results"][0]["blockers"], [])

    def test_blocks_usd_quote_replay_without_conversion_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            prices = root / "prices.json"
            intents.write_text(json.dumps({"results": [_intent()]}))
            prices.write_text(
                json.dumps(
                    {
                        "ticks": [
                            {"timestamp_utc": "2026-05-01T00:00:00+00:00", "pair": "EUR_USD", "bid": 1.0998, "ask": 1.1000},
                            {"timestamp_utc": "2026-05-01T00:01:00+00:00", "pair": "EUR_USD", "bid": 1.1010, "ask": 1.1011},
                        ]
                    }
                )
            )

            summary = ExecutionReplayer(
                intents_path=intents,
                price_path=prices,
                output_path=root / "execution_replay.json",
                report_path=root / "execution_replay.md",
            ).run(target_jpy=150)

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.target_hit)
            payload = json.loads((root / "execution_replay.json").read_text())
            self.assertIn("conversion tick", payload["blockers"][0])

    def test_market_replay_uses_actual_fill_quote_not_stale_expected_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            prices = root / "prices.json"
            stale_market = _intent()
            stale_market["intent"] = {
                **stale_market["intent"],
                "order_type": "MARKET",
                "entry": 1.0900,
                "tp": 1.1010,
                "sl": 1.0995,
            }
            intents.write_text(json.dumps({"results": [stale_market]}))
            prices.write_text(
                json.dumps(
                    {
                        "ticks": [
                            {"timestamp_utc": "2026-05-01T00:00:00+00:00", "pair": "EUR_USD", "bid": 1.0998, "ask": 1.1000},
                            {"timestamp_utc": "2026-05-01T00:00:00+00:00", "pair": "USD_JPY", "bid": 157.0, "ask": 157.01},
                            {"timestamp_utc": "2026-05-01T00:01:00+00:00", "pair": "EUR_USD", "bid": 1.1010, "ask": 1.1011},
                            {"timestamp_utc": "2026-05-01T00:01:00+00:00", "pair": "USD_JPY", "bid": 157.0, "ask": 157.01},
                        ]
                    }
                )
            )

            summary = ExecutionReplayer(
                intents_path=intents,
                price_path=prices,
                output_path=root / "execution_replay.json",
                report_path=root / "execution_replay.md",
            ).run(target_jpy=1000)

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.target_hit)
            self.assertAlmostEqual(summary.net_pl_jpy, 157.0)


def _intent() -> dict:
    return {
        "lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
        "status": "LIVE_READY",
        "intent": {
            "pair": "EUR_USD",
            "side": "LONG",
            "order_type": "STOP-ENTRY",
            "units": 1000,
            "entry": 1.1000,
            "tp": 1.1010,
            "sl": 1.0995,
            "thesis": "test",
            "market_context": {
                "regime": "TREND_CONTINUATION campaign lane",
                "narrative": "trend continuation pressure",
                "chart_story": "trend staircase",
                "method": "TREND_CONTINUATION",
                "invalidation": "SL trades",
            },
        },
    }


if __name__ == "__main__":
    unittest.main()

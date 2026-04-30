from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.models import BrokerOrder, BrokerSnapshot, Owner, Quote
from quant_rabbit.strategy.trader_brain import ACTION_MONITOR_EXISTING, ACTION_SEND_ENTRY, TraderBrain


class TraderBrainTest(unittest.TestCase):
    def test_penalizes_jpy_intervention_and_selects_direct_usd_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = TraderBrain(
                intents_path=_intents(root),
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_strategy(root),
                market_story_profile_path=_stories(root),
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            self.assertEqual(decision.action, ACTION_SEND_ENTRY)
            self.assertEqual(decision.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION")
            ranked = json.loads((root / "decision.json").read_text())["scores"]
            aud = next(item for item in ranked if item["pair"] == "AUD_JPY")
            self.assertIn("JPY-cross long faces intervention", " ".join(aud["blockers"]))

    def test_existing_pending_order_forces_monitor_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = TraderBrain(
                intents_path=_intents(root),
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_strategy(root),
                market_story_profile_path=_stories(root),
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )
            snapshot = _snapshot(
                orders=(
                    BrokerOrder(
                        order_id="1",
                        pair="AUD_JPY",
                        order_type="STOP",
                        price=112.5,
                        state="PENDING",
                        units=1000,
                        owner=Owner.TRADER,
                    ),
                )
            )

            decision = brain.run(snapshot)

            self.assertEqual(decision.action, ACTION_MONITOR_EXISTING)
            self.assertIsNone(decision.selected_lane_id)
            self.assertEqual(decision.pending_cancel_order_ids, ("1",))


def _snapshot(*, orders=()) -> BrokerSnapshot:
    now = datetime.now(timezone.utc)
    return BrokerSnapshot(
        fetched_at_utc=now,
        orders=tuple(orders),
        quotes={
            "AUD_JPY": Quote("AUD_JPY", 112.49, 112.50, timestamp_utc=now),
            "EUR_USD": Quote("EUR_USD", 1.1720, 1.1721, timestamp_utc=now),
            "USD_JPY": Quote("USD_JPY", 157.00, 157.01, timestamp_utc=now),
        },
    )


def _intents(root: Path) -> Path:
    path = root / "intents.json"
    path.write_text(
        json.dumps(
            {
                "results": [
                    _result(
                        "failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE",
                        "AUD_JPY",
                        "LONG",
                        "BREAKOUT_FAILURE",
                    ),
                    _result(
                        "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                        "EUR_USD",
                        "LONG",
                        "TREND_CONTINUATION",
                    ),
                ]
            }
        )
    )
    return path


def _result(lane_id: str, pair: str, side: str, method: str) -> dict:
    return {
        "lane_id": lane_id,
        "status": "LIVE_READY",
        "risk_allowed": True,
        "risk_issues": [],
        "live_blockers": [],
        "intent": {
            "pair": pair,
            "side": side,
            "order_type": "STOP-ENTRY",
            "units": 1000,
            "entry": 1.1725 if pair == "EUR_USD" else 112.56,
            "tp": 1.1737 if pair == "EUR_USD" else 112.72,
            "sl": 1.1717 if pair == "EUR_USD" else 112.46,
            "thesis": "test",
            "owner": "trader",
            "market_context": {
                "regime": f"{method} campaign lane",
                "narrative": "test narrative",
                "chart_story": "trend-bull continuation",
                "method": method,
                "invalidation": "SL trades",
            },
        },
    }


def _campaign(root: Path) -> Path:
    path = root / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    _lane("failure_trader", "AUD_JPY", "LONG", "BREAKOUT_FAILURE"),
                    _lane("trend_trader", "EUR_USD", "LONG", "TREND_CONTINUATION"),
                ]
            }
        )
    )
    return path


def _lane(desk: str, pair: str, direction: str, method: str) -> dict:
    return {
        "desk": desk,
        "pair": pair,
        "direction": direction,
        "method": method,
        "adoption": "ORDER_INTENT_REQUIRED",
        "campaign_role": "NOW_OR_BACKUP",
    }


def _strategy(root: Path) -> Path:
    path = root / "strategy.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "AUD_JPY",
                        "direction": "LONG",
                        "status": "CANDIDATE",
                        "pretrade_net_jpy": 3000,
                        "live_net_jpy": 2000,
                        "live_worst_jpy": -700,
                    },
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "CANDIDATE",
                        "pretrade_net_jpy": 5000,
                        "live_net_jpy": 2500,
                        "live_worst_jpy": -400,
                    },
                ]
            }
        )
    )
    return path


def _stories(root: Path) -> Path:
    path = root / "stories.json"
    path.write_text(
        json.dumps(
            {
                "pair_profiles": [
                    {
                        "pair": "AUD_JPY",
                        "methods": {"BREAKOUT_FAILURE": 30},
                        "themes": {"breakout_failure": 4, "intervention": 3, "spread_liquidity": 2},
                        "examples": ["JPY intervention risk and rate check; WAIT on crosses"],
                    },
                    {
                        "pair": "EUR_USD",
                        "methods": {"TREND_CONTINUATION": 35},
                        "themes": {"momentum": 5},
                        "examples": ["EUR_USD trend-bull staircase continuation"],
                    },
                ]
            }
        )
    )
    return path


if __name__ == "__main__":
    unittest.main()

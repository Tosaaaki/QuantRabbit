from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.models import BrokerOrder, BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.strategy.trader_brain import ACTION_MONITOR_EXISTING, ACTION_NO_TRADE, ACTION_SEND_ENTRY, TraderBrain


class TraderBrainTest(unittest.TestCase):
    def test_jpy_intervention_sizes_down_but_does_not_block(self) -> None:
        # Per AGENT_CONTRACT §6, narrative concerns must size the lane down via
        # size_multiple, not block it in prose. The AUD_JPY lane MUST NOT
        # carry an "intervention" blocker that would drop it from the GPT
        # prefilter set; the concern surfaces in rationale + lower size_multiple
        # instead.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = TraderBrain(
                intents_path=_intents(root),
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            brain.run(_snapshot())

            ranked = json.loads((root / "decision.json").read_text())["scores"]
            aud = next(item for item in ranked if item["pair"] == "AUD_JPY")
            eur = next(item for item in ranked if item["pair"] == "EUR_USD")
            blocker_text = " ".join(aud["blockers"])
            self.assertNotIn("JPY-cross long faces intervention", blocker_text)
            self.assertNotIn("visual story explicitly rejected", blocker_text)
            # Score penalty must be deep enough that AUD_JPY ranks well below the
            # unaffected EUR_USD lane (intervention contributes ≥55 of the gap).
            self.assertGreater(eur["score"] - aud["score"], 50.0)
            self.assertLess(aud["size_multiple"], 1.0)

    def test_existing_pending_order_forces_monitor_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = TraderBrain(
                intents_path=_intents(root),
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
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
            self.assertEqual(decision.pending_cancel_order_ids, ())

    def test_keeps_pending_when_compatible_lane_exists_below_top_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = TraderBrain(
                intents_path=_mixed_entry_type_intents(root),
                campaign_plan_path=_mixed_campaign(root),
                strategy_profile_path=_eur_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )
            snapshot = _snapshot(
                orders=(
                    BrokerOrder(
                        order_id="trend-stop",
                        pair="EUR_USD",
                        order_type="STOP",
                        price=1.17252,
                        state="PENDING",
                        units=1000,
                        owner=Owner.TRADER,
                    ),
                )
            )

            decision = brain.run(snapshot)

            self.assertEqual(decision.action, ACTION_MONITOR_EXISTING)
            self.assertEqual(decision.pending_cancel_order_ids, ())

    def test_cancels_pending_only_when_same_type_lane_has_moved_outside_spread_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = TraderBrain(
                intents_path=_mixed_entry_type_intents(root),
                campaign_plan_path=_mixed_campaign(root),
                strategy_profile_path=_eur_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )
            snapshot = _snapshot(
                orders=(
                    BrokerOrder(
                        order_id="far-stop",
                        pair="EUR_USD",
                        order_type="STOP",
                        price=1.18000,
                        state="PENDING",
                        units=1000,
                        owner=Owner.TRADER,
                    ),
                )
            )

            decision = brain.run(snapshot)

            self.assertEqual(decision.pending_cancel_order_ids, ("far-stop",))

    def test_protected_trader_position_can_still_select_portfolio_add(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = TraderBrain(
                intents_path=_intents(root),
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )
            snapshot = _snapshot(
                positions=(
                    BrokerPosition(
                        trade_id="101",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=3000,
                        entry_price=1.1710,
                        take_profit=1.1750,
                        stop_loss=1.1710,
                        owner=Owner.TRADER,
                    ),
                )
            )

            decision = brain.run(snapshot)

            self.assertEqual(decision.action, ACTION_SEND_ENTRY)
            self.assertEqual(decision.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION")

    def test_refuses_live_ready_lane_without_trader_thesis_and_market_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "intents.json"
            payload = {"results": [_result("trend_trader:EUR_USD:LONG:TREND_CONTINUATION", "EUR_USD", "LONG", "TREND_CONTINUATION")]}
            payload["results"][0]["intent"]["thesis"] = ""
            payload["results"][0]["intent"].pop("market_context")
            path.write_text(json.dumps(payload))
            brain = TraderBrain(
                intents_path=path,
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            self.assertEqual(decision.action, ACTION_NO_TRADE)
            lane = decision.scores[0]
            self.assertEqual(lane.action, ACTION_NO_TRADE)
            self.assertIn("missing trader thesis", " ".join(lane.blockers))
            self.assertIn("missing market context", " ".join(lane.blockers))

    def test_historical_worst_loss_is_scaled_by_current_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy_path = _strategy(root, loss_cap_jpy=1000.0)
            brain = TraderBrain(
                intents_path=_intents(root),
                campaign_plan_path=_campaign(root),
                strategy_profile_path=strategy_path,
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            brain.run(_snapshot())

            payload = json.loads((root / "decision.json").read_text())
            aud = next(item for item in payload["scores"] if item["pair"] == "AUD_JPY")
            self.assertEqual(payload["loss_cap_jpy"], 1000.0)
            self.assertNotIn("old worst loss repaired", " ".join(aud["rationale"]))

    def test_daily_target_state_overrides_stale_strategy_contract_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state = _target_state(root, per_trade_risk_budget_jpy=400.0)
            strategy_path = _strategy(root, loss_cap_jpy=1000.0)
            brain = TraderBrain(
                intents_path=_intents(root),
                campaign_plan_path=_campaign(root),
                strategy_profile_path=strategy_path,
                market_story_profile_path=_stories(root),
                target_state_path=target_state,
                trader_settings_path=root / "settings.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            brain.run(_snapshot())

            payload = json.loads((root / "decision.json").read_text())
            self.assertEqual(payload["loss_cap_jpy"], 400.0)
            self.assertIn("daily target state", payload["loss_cap_source"])

    def test_historical_large_loss_warns_but_does_not_block_repaired_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy_path = _eur_strategy(root)
            payload = json.loads(strategy_path.read_text())
            payload["system_contract"]["loss_cap_jpy"] = 400.0
            payload["profiles"][0]["live_worst_jpy"] = -900.0
            strategy_path.write_text(json.dumps(payload))
            brain = TraderBrain(
                intents_path=_intents(root),
                campaign_plan_path=_campaign(root),
                strategy_profile_path=strategy_path,
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            eur = next(item for item in decision.scores if item.pair == "EUR_USD")
            self.assertEqual(eur.action, ACTION_SEND_ENTRY)
            self.assertIn("historical live worst loss is large", " ".join(eur.rationale))
            self.assertNotIn("historical live worst loss is large", " ".join(eur.blockers))
            self.assertEqual(decision.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION")


def _snapshot(*, orders=(), positions=()) -> BrokerSnapshot:
    now = datetime.now(timezone.utc)
    return BrokerSnapshot(
        fetched_at_utc=now,
        positions=tuple(positions),
        orders=tuple(orders),
        quotes={
            "AUD_JPY": Quote("AUD_JPY", 112.49, 112.50, timestamp_utc=now),
            "EUR_USD": Quote("EUR_USD", 1.1720, 1.1721, timestamp_utc=now),
            "USD_JPY": Quote("USD_JPY", 157.00, 157.01, timestamp_utc=now),
        },
    )


def _target_state(root: Path, *, per_trade_risk_budget_jpy: float) -> Path:
    path = root / "target.json"
    path.write_text(
        json.dumps(
            {
                "status": "PURSUE_TARGET",
                "remaining_target_jpy": 10_000.0,
                "daily_risk_budget_jpy": per_trade_risk_budget_jpy * 10,
                "target_trades_per_day": 10,
                "per_trade_risk_budget_jpy": per_trade_risk_budget_jpy,
            }
        )
    )
    return path


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


def _mixed_entry_type_intents(root: Path) -> Path:
    path = root / "mixed_intents.json"
    stop = _result("trend_trader:EUR_USD:LONG:TREND_CONTINUATION", "EUR_USD", "LONG", "TREND_CONTINUATION")
    stop["risk_metrics"] = {"risk_jpy": 100.0, "reward_jpy": 220.0, "reward_risk": 2.2, "spread_pips": 0.8}
    limit = _result("range_trader:EUR_USD:LONG:RANGE_ROTATION", "EUR_USD", "LONG", "RANGE_ROTATION")
    limit["risk_metrics"] = {"risk_jpy": 80.0, "reward_jpy": 300.0, "reward_risk": 3.75, "spread_pips": 0.8}
    limit["intent"] = {
        **limit["intent"],
        "order_type": "LIMIT",
        "entry": 1.17120,
        "tp": 1.17360,
        "sl": 1.17060,
    }
    path.write_text(json.dumps({"results": [limit, stop]}))
    return path


def _result(lane_id: str, pair: str, side: str, method: str) -> dict:
    return {
        "lane_id": lane_id,
        "status": "LIVE_READY",
        "risk_allowed": True,
        "risk_metrics": {"risk_jpy": 100.0, "reward_jpy": 220.0, "reward_risk": 2.2, "spread_pips": 0.8},
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


def _mixed_campaign(root: Path) -> Path:
    path = root / "mixed_campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    _lane("trend_trader", "EUR_USD", "LONG", "TREND_CONTINUATION"),
                    _lane("range_trader", "EUR_USD", "LONG", "RANGE_ROTATION"),
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


def _strategy(root: Path, *, loss_cap_jpy: float = 500.0) -> Path:
    path = root / "strategy.json"
    path.write_text(
        json.dumps(
            {
                "system_contract": {
                    "loss_cap_jpy": loss_cap_jpy,
                    "loss_cap_source": "test current campaign cap",
                },
                "profiles": [
                    {
                        "pair": "AUD_JPY",
                        "direction": "LONG",
                        "status": "CANDIDATE",
                        "pretrade_net_jpy": 3000,
                        "live_net_jpy": 2000,
                        "live_worst_jpy": -700,
                        "positive_evidence_n": 80,
                        "positive_tail_jpy": 900,
                        "positive_best_jpy": 1500,
                        "seat_discovered": 10,
                        "seat_orderable": 8,
                        "seat_captured": 4,
                    },
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "CANDIDATE",
                        "pretrade_net_jpy": 5000,
                        "live_net_jpy": 2500,
                        "live_worst_jpy": -400,
                        "positive_evidence_n": 120,
                        "positive_tail_jpy": 1200,
                        "positive_best_jpy": 2200,
                        "seat_discovered": 10,
                        "seat_orderable": 8,
                        "seat_captured": 5,
                    },
                ]
            }
        )
    )
    return path


def _eur_strategy(root: Path) -> Path:
    path = root / "eur_strategy.json"
    path.write_text(
        json.dumps(
            {
                "system_contract": {
                    "loss_cap_jpy": 500.0,
                    "loss_cap_source": "test current campaign cap",
                },
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "CANDIDATE",
                        "pretrade_net_jpy": 5000,
                        "live_net_jpy": 2500,
                        "live_worst_jpy": -400,
                        "positive_evidence_n": 120,
                        "positive_tail_jpy": 1200,
                        "positive_best_jpy": 2200,
                        "seat_discovered": 10,
                        "seat_orderable": 8,
                        "seat_captured": 5,
                    }
                ],
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
                        "examples": [
                            "news_digest: JPY intervention risk and rate check; WAIT on crosses",
                            "quality_audit: AUD_JPY trend-bull continuation but narrative-sensitive",
                        ],
                    },
                    {
                        "pair": "EUR_USD",
                        "methods": {"TREND_CONTINUATION": 35},
                        "themes": {"momentum": 5},
                        "examples": [
                            "news_digest: EUR_USD trend-bull macro continuation",
                            "quality_audit: EUR_USD trend-bull staircase continuation",
                        ],
                    },
                ]
            }
        )
    )
    return path


if __name__ == "__main__":
    unittest.main()

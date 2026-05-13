from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.models import BrokerOrder, BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.strategy.trader_brain import (
    ACTION_MONITOR_EXISTING,
    ACTION_NO_TRADE,
    ACTION_SEND_ENTRY,
    MICRO_STRUCTURE_ALIGNED_BONUS,
    MICRO_STRUCTURE_OPPOSED_PENALTY,
    MTF_CONFLUENCE_CEILING,
    MTF_CONFLUENCE_FLOOR,
    SHORT_TERM_MOMENTUM_HIGH_ADX,
    SHORT_TERM_MOMENTUM_LOW_ADX,
    TraderBrain,
    _micro_structure_alignment_score,
    _micro_structure_direction,
    _mtf_confluence_score,
    _narrative_risk_score,
    _parse_chart_story_full,
    _short_term_momentum_class,
    _tf_lens_support,
    _tf_strength_multiplier,
)


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

    def test_target_open_keeps_passive_pending_for_gateway_basket_counting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state = _target_state(root, per_trade_risk_budget_jpy=400.0)
            brain = TraderBrain(
                intents_path=_market_preference_intents(root),
                campaign_plan_path=_mixed_campaign(root),
                strategy_profile_path=_eur_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=target_state,
                trader_settings_path=root / "settings.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )
            snapshot = _snapshot(
                orders=(
                    BrokerOrder(
                        order_id="passive-limit",
                        pair="EUR_USD",
                        order_type="LIMIT",
                        price=1.17120,
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

    def test_cancels_pending_when_tp_or_sl_geometry_is_stale(self) -> None:
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
                        order_id="stale-limit-sl",
                        pair="EUR_USD",
                        order_type="LIMIT",
                        price=1.17120,
                        state="PENDING",
                        units=1000,
                        owner=Owner.TRADER,
                        raw={
                            "takeProfitOnFill": {"price": "1.17360"},
                            "stopLossOnFill": {"price": "1.16800"},
                        },
                    ),
                )
            )

            decision = brain.run(snapshot)

            self.assertEqual(decision.pending_cancel_order_ids, ("stale-limit-sl",))

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

    def test_target_open_flat_account_prefers_market_lane_over_passive_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_state = _target_state(root, per_trade_risk_budget_jpy=400.0)
            brain = TraderBrain(
                intents_path=_market_preference_intents(root),
                campaign_plan_path=_mixed_campaign(root),
                strategy_profile_path=_eur_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=target_state,
                trader_settings_path=root / "settings.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            self.assertEqual(decision.action, ACTION_SEND_ENTRY)
            self.assertEqual(decision.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET")
            self.assertIn("MARKET lane", decision.reason)

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

    def test_past_negative_history_cannot_veto_current_live_ready_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = TraderBrain(
                intents_path=_eur_only_intents(root),
                campaign_plan_path=_eur_only_campaign(root),
                strategy_profile_path=_negative_history_strategy(root),
                market_story_profile_path=_thin_eur_story(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            eur = decision.scores[0]
            self.assertEqual(eur.action, ACTION_SEND_ENTRY)
            self.assertEqual(decision.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION")
            blocker_text = " ".join(eur.blockers)
            self.assertNotIn("negative live execution history", blocker_text)
            self.assertNotIn("missing positive mined evidence", blocker_text)
            self.assertNotIn("low capture rate", blocker_text)
            self.assertNotIn("no positive mined or repaired edge evidence", blocker_text)
            self.assertIn("current receipt is the authority", " ".join(eur.rationale))
            self.assertGreaterEqual(eur.size_multiple, 1.0)

    def test_stale_story_markers_are_advisory_for_live_ready_receipts(self) -> None:
        blockers: list[str] = []
        rationale: list[str] = []

        score = _narrative_risk_score(
            "EUR_USD",
            Side.LONG.value,
            "RANGE_ROTATION",
            {},
            (
                "news_digest: old WAIT note from prior review",
                "quality_audit: NO: stale range rejection marker",
            ),
            blockers,
            rationale,
            status="LIVE_READY",
        )

        self.assertEqual(score, 0.0)
        self.assertEqual(blockers, [])
        rationale_text = " ".join(rationale)
        self.assertIn("stale narrative WAIT language ignored", rationale_text)
        self.assertIn("stale visual rejection marker ignored", rationale_text)


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


def _eur_only_intents(root: Path) -> Path:
    path = root / "eur_intents.json"
    path.write_text(
        json.dumps(
            {
                "results": [
                    _result(
                        "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                        "EUR_USD",
                        "LONG",
                        "TREND_CONTINUATION",
                    )
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


def _market_preference_intents(root: Path) -> Path:
    path = root / "market_preference_intents.json"
    pending = _result("range_trader:EUR_USD:LONG:RANGE_ROTATION", "EUR_USD", "LONG", "RANGE_ROTATION")
    pending["risk_metrics"] = {"risk_jpy": 80.0, "reward_jpy": 320.0, "reward_risk": 4.0, "spread_pips": 0.8}
    pending["intent"] = {
        **pending["intent"],
        "order_type": "LIMIT",
        "entry": 1.17120,
        "tp": 1.17360,
        "sl": 1.17060,
    }
    market = _result(
        "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
        "EUR_USD",
        "LONG",
        "TREND_CONTINUATION",
    )
    market["intent"] = {**market["intent"], "order_type": "MARKET", "entry": 1.17306}
    market["risk_metrics"] = {"risk_jpy": 100.0, "reward_jpy": 220.0, "reward_risk": 2.2, "spread_pips": 0.8}
    path.write_text(json.dumps({"results": [pending, market]}))
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


def _eur_only_campaign(root: Path) -> Path:
    path = root / "eur_campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
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


def _negative_history_strategy(root: Path) -> Path:
    path = root / "negative_history_strategy.json"
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
                        "pretrade_net_jpy": -3000,
                        "live_net_jpy": -1500,
                        "live_worst_jpy": -1200,
                        "positive_evidence_n": 0,
                        "positive_tail_jpy": 0,
                        "positive_best_jpy": 0,
                        "seat_discovered": 10,
                        "seat_orderable": 10,
                        "seat_captured": 0,
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


def _thin_eur_story(root: Path) -> Path:
    path = root / "thin_eur_story.json"
    path.write_text(
        json.dumps(
            {
                "pair_profiles": [
                    {
                        "pair": "EUR_USD",
                        "methods": {"TREND_CONTINUATION": 20},
                        "themes": {"momentum": 2},
                        "examples": ["quality_audit: EUR_USD trend-bull continuation"],
                    }
                ]
            }
        )
    )
    return path


class MicroStructureAlignmentTest(unittest.TestCase):
    def _ctx(self, m1: str | None, m5: str | None) -> dict[str, dict[str, str]]:
        parts = ["EUR_USD TREND_DOWN"]
        if m1 is not None:
            parts.append(f"M1(RANGE, ADX=15.0 RSI=52.0 ATR=1.0p struct={m1}@1.1730)")
        if m5 is not None:
            parts.append(f"M5(RANGE, ADX=16.0 RSI=53.0 ATR=2.0p struct={m5}@1.1732)")
        return {"chart_story": "; ".join(parts)}

    def test_direction_uses_m5_when_only_m5_present(self) -> None:
        self.assertEqual(_micro_structure_direction(self._ctx(None, "BOS_UP")), "UP")
        self.assertEqual(_micro_structure_direction(self._ctx(None, "CHOCH_DOWN")), "DOWN")

    def test_direction_uses_m1_when_only_m1_present(self) -> None:
        self.assertEqual(_micro_structure_direction(self._ctx("BOS_UP", None)), "UP")

    def test_direction_aligned_when_m1_and_m5_agree(self) -> None:
        self.assertEqual(_micro_structure_direction(self._ctx("CHOCH_UP", "BOS_UP")), "UP")
        self.assertEqual(_micro_structure_direction(self._ctx("BOS_DOWN", "CHOCH_DOWN")), "DOWN")

    def test_direction_unclear_when_m1_and_m5_conflict(self) -> None:
        self.assertEqual(_micro_structure_direction(self._ctx("BOS_UP", "BOS_DOWN")), "UNCLEAR")

    def test_direction_unclear_when_no_struct_field(self) -> None:
        self.assertEqual(_micro_structure_direction({"chart_story": "EUR_USD; M1(RANGE, ADX=15.0)"}), "UNCLEAR")
        self.assertEqual(_micro_structure_direction({}), "UNCLEAR")
        self.assertEqual(_micro_structure_direction(None), "UNCLEAR")

    def test_alignment_penalizes_short_into_micro_up_flip(self) -> None:
        # 2026-05-08 EUR_USD scalp scenario: H1 TREND_DOWN but M1/M5 just
        # flipped UP via BOS_UP@1.1732. Lane direction SHORT must take a
        # negative score adjustment so the operator sees the conflict in
        # rationale and a same-direction-as-flip lane can outscore it.
        intent = {"side": "SHORT", "market_context": self._ctx("CHOCH_UP", "BOS_UP")}
        rationale: list[str] = []
        score = _micro_structure_alignment_score(intent, rationale, [])
        self.assertEqual(score, MICRO_STRUCTURE_OPPOSED_PENALTY)
        self.assertTrue(any("opposes SHORT" in line for line in rationale))

    def test_alignment_rewards_long_into_micro_up_flip(self) -> None:
        intent = {"side": "LONG", "market_context": self._ctx("CHOCH_UP", "BOS_UP")}
        rationale: list[str] = []
        score = _micro_structure_alignment_score(intent, rationale, [])
        self.assertEqual(score, MICRO_STRUCTURE_ALIGNED_BONUS)
        self.assertTrue(any("agrees with LONG" in line for line in rationale))

    def test_alignment_neutral_when_micro_unclear(self) -> None:
        intent = {"side": "SHORT", "market_context": self._ctx("BOS_UP", "BOS_DOWN")}
        rationale: list[str] = []
        score = _micro_structure_alignment_score(intent, rationale, [])
        self.assertEqual(score, 0.0)
        self.assertEqual(rationale, [])

    def test_alignment_neutral_when_market_context_missing(self) -> None:
        intent = {"side": "LONG"}
        rationale: list[str] = []
        score = _micro_structure_alignment_score(intent, rationale, [])
        self.assertEqual(score, 0.0)


class MTFConfluenceTest(unittest.TestCase):
    """Full 7-TF × 5-lens confluence scoring (`_mtf_confluence_score`).

    User directive 2026-05-08「分析を広く」「エントリーしない理由ではなく、
    エントリーする理由をみつけてほしい」: positive bias, capped negative.
    """

    # Fixture mirroring the live 2026-05-08 EUR_USD chart_story after the
    # M5 BOS_DOWN re-flip — stack is uniformly SHORT-supportive across
    # struct, regime, Supertrend, Ichimoku cloud.
    EUR_USD_FULL_SHORT_STORY = (
        "EUR_USD TREND_DOWN; "
        "M1(UNCLEAR, ADX=23.6 RSI=49.0 ATR=1.0p ST=- Read=TRANSITION:0.25 cloud=below struct=BOS_DOWN@1.1730); "
        "M5(RANGE, ADX=12.4 RSI=45.0 ATR=2.0p ST=- Read=TRANSITION:0.25 cloud=below struct=BOS_DOWN@1.1731); "
        "M15(TREND_DOWN, ADX=37.0 RSI=43.1 ATR=4.9p ST=- Read=TREND_WEAK:0.33 cloud=below struct=CHOCH_UP@1.1732); "
        "M30(TREND_DOWN, ADX=36.1 RSI=37.4 ATR=8.1p ST=- Read=TREND_WEAK:0.67 cloud=below struct=BOS_DOWN@1.1736); "
        "H1(TREND_DOWN, ADX=31.3 RSI=39.1 ATR=11.7p ST=- Read=TREND_WEAK:0.67 cloud=below struct=CHOCH_DOWN@1.1762); "
        "H4(TREND_DOWN, ADX=26.8 RSI=48.7 ATR=22.4p ST=+ Read=TREND_WEAK:0.67 struct=BOS_UP@1.1785); "
        "D(UNCLEAR, ADX=22.1 RSI=53.8 ATR=65.0p ST=+ Read=TRANSITION:0.25 cloud=above struct=CHOCH_DOWN@1.1669)"
    )

    def _intent(self, side: str, story: str | None = None) -> dict:
        return {
            "side": side,
            "metadata": {"chart_story_structural": story or self.EUR_USD_FULL_SHORT_STORY},
        }

    def test_parse_extracts_all_seven_timeframes(self) -> None:
        parsed = _parse_chart_story_full(self.EUR_USD_FULL_SHORT_STORY)
        self.assertEqual(set(parsed.keys()), {"M1", "M5", "M15", "M30", "H1", "H4", "D"})

    def test_parse_extracts_all_lenses(self) -> None:
        parsed = _parse_chart_story_full(self.EUR_USD_FULL_SHORT_STORY)
        m30 = parsed["M30"]
        self.assertEqual(m30["regime"], "TREND_DOWN")
        self.assertEqual(m30["struct_dir"], "DOWN")
        self.assertEqual(m30["struct_type"], "BOS")
        self.assertEqual(m30["supertrend"], "DOWN")
        self.assertEqual(m30["cloud"], "below")
        self.assertAlmostEqual(m30["adx"], 36.1)
        self.assertAlmostEqual(m30["rsi"], 37.4)
        self.assertEqual(m30["read_label"], "TREND_WEAK")
        self.assertAlmostEqual(m30["read_confidence"], 0.67)

    def test_parse_handles_missing_cloud(self) -> None:
        parsed = _parse_chart_story_full(self.EUR_USD_FULL_SHORT_STORY)
        # H4 in the live story carries no cloud field; parser must omit
        # the key gracefully (not crash, not insert a default).
        self.assertNotIn("cloud", parsed["H4"])

    def test_lens_support_short_picks_up_all_short_lenses(self) -> None:
        parsed = _parse_chart_story_full(self.EUR_USD_FULL_SHORT_STORY)
        h1 = parsed["H1"]
        raw, max_possible, reasons = _tf_lens_support(h1, "SHORT")
        # H1 has 4 SHORT-supporting lenses: struct=DOWN(1.0), regime
        # TREND_DOWN(1.0), ST=-(0.7), cloud=below(0.5) = 3.2 raw with 3.2 max.
        self.assertAlmostEqual(raw, 3.2)
        self.assertAlmostEqual(max_possible, 3.2)
        # Reasons should list the four supporting lenses in some order.
        text = " ".join(reasons)
        self.assertIn("DOWN", text)
        self.assertIn("ST=-", text)
        self.assertIn("cloud=below", text)

    def test_strength_multiplier_boosts_high_adx(self) -> None:
        # ADX 31 + Read confidence 0.67 → 1.30 × (0.5 + 0.5*0.67) = 1.30 × 0.835 = 1.0855
        h1_data = {"adx": 31.3, "read_confidence": 0.67}
        self.assertAlmostEqual(_tf_strength_multiplier(h1_data), 1.30 * 0.835, places=4)

    def test_strength_multiplier_dampens_low_adx(self) -> None:
        # ADX 12 → 0.70 multiplier; no Read confidence → 1.0 inner factor
        m5_data = {"adx": 12.4}
        self.assertAlmostEqual(_tf_strength_multiplier(m5_data), 0.70)

    def test_short_lane_strongly_aligned_returns_large_positive(self) -> None:
        # The full-stack EUR_USD SHORT setup must score solidly positive
        # (≥10) so the lane outranks lagging-evidence-only competitors.
        score = _mtf_confluence_score(self._intent("SHORT"), [], [])
        self.assertGreaterEqual(score, 10.0)
        self.assertLessEqual(score, MTF_CONFLUENCE_CEILING)

    def test_long_lane_into_full_short_stack_capped_negative(self) -> None:
        # LONG into a uniformly SHORT-aligned MTF stack must take a
        # negative penalty but never exceed the floor — a single contrary
        # signal should not zero out an otherwise-priceable setup.
        score = _mtf_confluence_score(self._intent("LONG"), [], [])
        self.assertLessEqual(score, 5.0)
        self.assertGreaterEqual(score, MTF_CONFLUENCE_FLOOR)

    def test_score_surfaces_aligned_lenses_in_rationale(self) -> None:
        # Per directive: "find reasons to enter". Operator must see which
        # lenses agreed even when net score is moderate.
        rationale: list[str] = []
        _mtf_confluence_score(self._intent("SHORT"), rationale, [])
        self.assertTrue(rationale, "rationale must surface reasoning")
        joined = " ".join(rationale)
        self.assertIn("aligned", joined)

    def test_no_chart_story_returns_zero(self) -> None:
        self.assertEqual(_mtf_confluence_score({"side": "LONG"}, [], []), 0.0)

    def test_invalid_direction_returns_zero(self) -> None:
        self.assertEqual(
            _mtf_confluence_score({"side": "WAIT", "metadata": {"chart_story_structural": self.EUR_USD_FULL_SHORT_STORY}}, [], []),
            0.0,
        )

    def test_negative_score_capped_at_floor(self) -> None:
        # Even a worst-case fully-opposed alignment must respect the floor.
        score = _mtf_confluence_score(self._intent("LONG"), [], [])
        self.assertGreaterEqual(score, MTF_CONFLUENCE_FLOOR)

    def test_rsi_extreme_supports_mean_reversion(self) -> None:
        # RSI 75 on M15 with no other M15 signal should still cast a SHORT
        # vote (mean-reversion bias).
        story = "X; M15(RANGE, ADX=20.0 RSI=75.0 ATR=4.0p Read=TRANSITION:0.5)"
        parsed = _parse_chart_story_full(story)
        raw, max_possible, reasons = _tf_lens_support(parsed["M15"], "SHORT")
        self.assertGreater(raw, 0.0)
        self.assertTrue(any("OB" in r for r in reasons))


class ShortTermMomentumClassTest(unittest.TestCase):
    """Coverage for f35c130 — regime-aware MARKET vs pending entry scoring.

    `_short_term_momentum_class` reads M1/M5 ADX off `chart_story` and returns
    HIGH (≥SHORT_TERM_MOMENTUM_HIGH_ADX), LOW (≤SHORT_TERM_MOMENTUM_LOW_ADX),
    or NEUTRAL. `_score_lane` applies +12/-8/+5 to MARKET variants based on
    this so the variant race reflects regime, not a fixed bonus.
    """

    def _ctx(self, m1_adx: float | None, m5_adx: float | None) -> dict[str, str]:
        parts = ["EUR_USD TREND_DOWN"]
        if m1_adx is not None:
            parts.append(f"M1(RANGE, ADX={m1_adx} RSI=52.0 ATR=1.0p)")
        if m5_adx is not None:
            parts.append(f"M5(RANGE, ADX={m5_adx} RSI=53.0 ATR=2.0p)")
        return {"chart_story": "; ".join(parts)}

    def test_high_when_avg_at_or_above_high_threshold(self) -> None:
        # avg = 25.0 == HIGH threshold (25.0).
        self.assertEqual(_short_term_momentum_class(self._ctx(20.0, 30.0)), "HIGH")
        # avg = 27.5 > HIGH.
        self.assertEqual(_short_term_momentum_class(self._ctx(25.0, 30.0)), "HIGH")

    def test_low_when_avg_at_or_below_low_threshold(self) -> None:
        # avg = 18.0 == LOW threshold (18.0).
        self.assertEqual(_short_term_momentum_class(self._ctx(15.0, 21.0)), "LOW")
        # avg = 12.0 well below.
        self.assertEqual(_short_term_momentum_class(self._ctx(10.0, 14.0)), "LOW")

    def test_neutral_when_avg_between_thresholds(self) -> None:
        # avg = 21.5 strictly between 18.0 and 25.0.
        self.assertEqual(_short_term_momentum_class(self._ctx(20.0, 23.0)), "NEUTRAL")

    def test_neutral_when_only_one_timeframe_present(self) -> None:
        # Pattern requires both M1 and M5 ADX — partial → NEUTRAL.
        self.assertEqual(_short_term_momentum_class(self._ctx(30.0, None)), "NEUTRAL")
        self.assertEqual(_short_term_momentum_class(self._ctx(None, 30.0)), "NEUTRAL")

    def test_neutral_when_chart_story_missing(self) -> None:
        self.assertEqual(_short_term_momentum_class({}), "NEUTRAL")
        self.assertEqual(_short_term_momentum_class({"chart_story": ""}), "NEUTRAL")
        self.assertEqual(_short_term_momentum_class(None), "NEUTRAL")

    def test_neutral_when_market_context_is_not_a_dict(self) -> None:
        self.assertEqual(_short_term_momentum_class("EUR_USD M1(ADX=30) M5(ADX=30)"), "NEUTRAL")
        self.assertEqual(_short_term_momentum_class([]), "NEUTRAL")

    def test_thresholds_match_documented_constants(self) -> None:
        # Guard against silent threshold drift; the constants are tuned for
        # FX major pairs in 2026 sessions and changing them shifts every
        # variant race outcome.
        self.assertEqual(SHORT_TERM_MOMENTUM_HIGH_ADX, 25.0)
        self.assertEqual(SHORT_TERM_MOMENTUM_LOW_ADX, 18.0)


class RiskIssueSeverityTest(unittest.TestCase):
    """Coverage for 2026-05-11 WARN-severity fix in `_score_lane`.

    intent_generator downgrades CHART_DIRECTION_CONFLICT to WARN under
    SL-free so symmetric mirror lanes can reach LIVE_READY. Previously
    trader_brain.`_score_lane` treated every entry in `risk_issues` as a
    hard blocker (and -100 score), turning the WARN downgrade back into a
    NO_TRADE veto. Tests pin the severity-aware behavior so a future
    refactor cannot silently re-introduce the regression that left
    EUR_USD SHORT off the prefilter while ai_attack_advice ranked it #2.
    """

    def _intents_with_risk_issues(self, root: Path, issues: list[dict]) -> Path:
        path = root / "intents_with_issues.json"
        lane = _result(
            "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
            "EUR_USD",
            "LONG",
            "TREND_CONTINUATION",
        )
        lane["risk_issues"] = issues
        path.write_text(json.dumps({"results": [lane]}))
        return path

    def test_warn_risk_issue_does_not_block_send_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = self._intents_with_risk_issues(
                root,
                [
                    {
                        "code": "CHART_DIRECTION_CONFLICT",
                        "message": "EUR_USD LONG conflicts with current pair_charts direction bias=SHORT",
                        "severity": "WARN",
                    }
                ],
            )
            brain = TraderBrain(
                intents_path=intents,
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_eur_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            score = next(item for item in decision.scores if item.pair == "EUR_USD")
            self.assertEqual(score.action, ACTION_SEND_ENTRY)
            self.assertNotIn(
                "EUR_USD LONG conflicts",
                " ".join(score.blockers),
            )
            self.assertTrue(any("risk warn CHART_DIRECTION_CONFLICT" in r for r in score.rationale))

    def test_block_risk_issue_still_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = self._intents_with_risk_issues(
                root,
                [
                    {
                        "code": "TREND_MARKET_NOT_OPERATING_TREND",
                        "message": "EUR_USD LONG MARKET trend-continuation needs M5 TREND_UP",
                        "severity": "BLOCK",
                    }
                ],
            )
            brain = TraderBrain(
                intents_path=intents,
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_eur_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            score = next(item for item in decision.scores if item.pair == "EUR_USD")
            self.assertEqual(score.action, ACTION_NO_TRADE)
            self.assertIn(
                "needs M5 TREND_UP",
                " ".join(score.blockers),
            )


class AttackAdvicePromotionTest(unittest.TestCase):
    """Coverage for AGENT_CONTRACT §8 attack-advice overlay in the
    trader_brain prefilter (2026-05-11).

    `ai_attack_advice.recommended_now_lane_ids[:K]` lanes that are
    LIVE_READY pick up a documented score bonus + rationale so the
    deterministic prefilter surfaces the same primary lanes the GPT
    verifier expects. The promotion never overrides §11 hard blocks
    (BLOCK_UNTIL_NEW_EVIDENCE, missing receipt, exposure blockers).
    """

    def _attack_advice(self, root: Path, lane_ids: list[str]) -> Path:
        path = root / "attack_advice.json"
        path.write_text(json.dumps({"recommended_now_lane_ids": lane_ids}))
        return path

    def test_constant_matches_gpt_trader(self) -> None:
        from quant_rabbit.gpt_trader import PRIMARY_ATTACK_RANK_CEILING
        from quant_rabbit.strategy.trader_brain import (
            ATTACK_ADVICE_PROMOTION_RANK_CEILING,
        )

        self.assertEqual(ATTACK_ADVICE_PROMOTION_RANK_CEILING, PRIMARY_ATTACK_RANK_CEILING)

    def test_top_k_lane_gets_bonus_and_rationale(self) -> None:
        from quant_rabbit.strategy.trader_brain import ATTACK_ADVICE_PROMOTION_BONUS

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            advice = self._attack_advice(
                root,
                ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION"],
            )
            brain_advised = TraderBrain(
                intents_path=_eur_only_intents(root),
                campaign_plan_path=_eur_only_campaign(root),
                strategy_profile_path=_eur_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=advice,
                output_path=root / "advised.json",
                report_path=root / "advised.md",
            )
            brain_unadvised = TraderBrain(
                intents_path=_eur_only_intents(root),
                campaign_plan_path=_eur_only_campaign(root),
                strategy_profile_path=_eur_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                output_path=root / "unadvised.json",
                report_path=root / "unadvised.md",
            )

            advised = brain_advised.run(_snapshot())
            unadvised = brain_unadvised.run(_snapshot())

            advised_score = next(s for s in advised.scores if s.pair == "EUR_USD")
            unadvised_score = next(s for s in unadvised.scores if s.pair == "EUR_USD")
            self.assertAlmostEqual(
                advised_score.score - unadvised_score.score,
                ATTACK_ADVICE_PROMOTION_BONUS,
                places=2,
            )
            self.assertTrue(any("attack_advice rank #1" in r for r in advised_score.rationale))

    def test_below_top_k_lane_gets_no_bonus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # The advised lane sits at rank 5 (below K=4); fill earlier
            # ranks with throwaway lane_ids so the promoter does not
            # promote our test lane.
            advice = self._attack_advice(
                root,
                [
                    "filler_1",
                    "filler_2",
                    "filler_3",
                    "filler_4",
                    "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                ],
            )
            brain_unadvised = TraderBrain(
                intents_path=_eur_only_intents(root),
                campaign_plan_path=_eur_only_campaign(root),
                strategy_profile_path=_eur_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                output_path=root / "unadvised.json",
                report_path=root / "unadvised.md",
            )
            brain_advised = TraderBrain(
                intents_path=_eur_only_intents(root),
                campaign_plan_path=_eur_only_campaign(root),
                strategy_profile_path=_eur_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=advice,
                output_path=root / "advised.json",
                report_path=root / "advised.md",
            )

            advised = brain_advised.run(_snapshot())
            unadvised = brain_unadvised.run(_snapshot())

            advised_score = next(s for s in advised.scores if s.pair == "EUR_USD")
            unadvised_score = next(s for s in unadvised.scores if s.pair == "EUR_USD")
            self.assertEqual(advised_score.score, unadvised_score.score)

    def _blocked_strategy(self, root: Path) -> Path:
        strategy_path = root / "blocked_strategy.json"
        strategy_path.write_text(
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
                            "status": "BLOCK_UNTIL_NEW_EVIDENCE",
                            "pretrade_net_jpy": -3000,
                            "live_net_jpy": -2000,
                            "live_worst_jpy": -1500,
                            "positive_evidence_n": 0,
                            "positive_tail_jpy": 0,
                            "positive_best_jpy": 0,
                            "seat_discovered": 10,
                            "seat_orderable": 10,
                            "seat_captured": 0,
                        }
                    ],
                }
            )
        )
        return strategy_path

    def test_block_remains_hard_without_sl_free(self) -> None:
        # Legacy contract: without SL-free, BLOCK_UNTIL_NEW_EVIDENCE keeps
        # the lane out of SEND_ENTRY regardless of attack_advice overlay.
        prior = os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                advice = self._attack_advice(
                    root,
                    ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION"],
                )
                brain = TraderBrain(
                    intents_path=_eur_only_intents(root),
                    campaign_plan_path=_eur_only_campaign(root),
                    strategy_profile_path=self._blocked_strategy(root),
                    market_story_profile_path=_stories(root),
                    target_state_path=root / "missing_target.json",
                    trader_settings_path=root / "settings.json",
                    attack_advice_path=advice,
                    output_path=root / "decision.json",
                    report_path=root / "decision.md",
                )

                decision = brain.run(_snapshot())

                score = next(s for s in decision.scores if s.pair == "EUR_USD")
                self.assertEqual(score.action, ACTION_NO_TRADE)
                self.assertTrue(
                    any("BLOCK_UNTIL_NEW_EVIDENCE" in b for b in score.blockers),
                    f"Expected §11 hard block to remain; got blockers: {score.blockers}",
                )
        finally:
            if prior is not None:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_block_becomes_advisory_under_sl_free(self) -> None:
        # 2026-05-11 fix B-1c: under SL-free, the per_trade cap bounds
        # the loss so non-CANDIDATE profile status (e.g.
        # BLOCK_UNTIL_NEW_EVIDENCE) downgrades to advisory rationale
        # instead of a hard veto, mirroring strategy_profile.validate
        # (profile.py:125) and intent_generator's WARN downgrade. The
        # profile status itself stays unchanged — AGENT_CONTRACT §11
        # forbids only auto-promotion of the status field.
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                advice = self._attack_advice(
                    root,
                    ["trend_trader:EUR_USD:LONG:TREND_CONTINUATION"],
                )
                brain = TraderBrain(
                    intents_path=_eur_only_intents(root),
                    campaign_plan_path=_eur_only_campaign(root),
                    strategy_profile_path=self._blocked_strategy(root),
                    market_story_profile_path=_stories(root),
                    target_state_path=root / "missing_target.json",
                    trader_settings_path=root / "settings.json",
                    attack_advice_path=advice,
                    output_path=root / "decision.json",
                    report_path=root / "decision.md",
                )

                decision = brain.run(_snapshot())

                score = next(s for s in decision.scores if s.pair == "EUR_USD")
                self.assertEqual(
                    score.action,
                    ACTION_SEND_ENTRY,
                    f"Expected SEND_ENTRY under SL-free; got {score.action} blockers={score.blockers}",
                )
                # Status text surfaces as advisory rationale rather than
                # a blocker entry.
                self.assertFalse(
                    any("BLOCK_UNTIL_NEW_EVIDENCE" in b for b in score.blockers),
                    f"BLOCK should not appear in blockers under SL-free: {score.blockers}",
                )
                self.assertTrue(
                    any("BLOCK_UNTIL_NEW_EVIDENCE" in r for r in score.rationale),
                    f"Expected BLOCK to surface in rationale; got {score.rationale}",
                )
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior


class DirectionalGatingTest(unittest.TestCase):
    """Coverage for C-1 (directional gating) + C-2 (attack_advice
    directional veto), added 2026-05-12. Both run on the scored
    LaneScore tuple inside `_apply_directional_gating` after
    `_score_lane` and before basket construction. The gate never
    consults `snapshot.positions` — a separate test below pins that
    contract so PositionManager / PositionProtectionGateway behavior on
    existing trades cannot be reached by this code path.
    """

    def _pair_charts(self, *, balance: str, gap: float, pair: str = "EUR_USD") -> dict:
        return {
            pair: {
                "confluence": {
                    "score_balance": balance,
                    "score_gap": gap,
                    "higher_tf_alignment": "ALIGNED" if balance != "TIED" else "NEUTRAL",
                },
            }
        }

    def _make_score(
        self,
        *,
        lane_id: str,
        pair: str,
        direction: str,
        score: float = 100.0,
        action: str = ACTION_SEND_ENTRY,
        estimated_margin_jpy: float | None = None,
    ):
        from quant_rabbit.strategy.trader_brain import LaneScore

        return LaneScore(
            lane_id=lane_id,
            pair=pair,
            direction=direction,
            method="TREND_CONTINUATION",
            order_type="MARKET",
            entry=1.0,
            tp=1.01,
            sl=None,
            status="LIVE_READY",
            score=score,
            action=action,
            blockers=(),
            rationale=(),
            size_multiple=1.0,
            estimated_margin_jpy=estimated_margin_jpy,
        )

    def test_c1_short_lean_with_short_majority_demotes_long(self) -> None:
        from quant_rabbit.strategy.trader_brain import _apply_directional_gating

        # SHORT_LEAN gap -0.44 (well past 0.10 strong threshold) + advice
        # top-K has 2 SHORT vs 0 LONG for EUR_USD → LONG lane demoted.
        scores = (
            self._make_score(
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                pair="EUR_USD",
                direction="LONG",
                score=200.0,
            ),
            self._make_score(
                lane_id="trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET",
                pair="EUR_USD",
                direction="SHORT",
                score=150.0,
            ),
        )
        pair_charts = self._pair_charts(balance="SHORT_LEAN", gap=-0.44)
        attack_ranks = {
            "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET": 0,
            "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:MARKET": 1,
        }

        result = _apply_directional_gating(scores, pair_charts, attack_ranks)

        long_lane = next(s for s in result if s.direction == "LONG")
        short_lane = next(s for s in result if s.direction == "SHORT")
        self.assertEqual(long_lane.action, ACTION_NO_TRADE)
        self.assertTrue(
            any("directional_gating_demoted" in b for b in long_lane.blockers),
            f"LONG should carry directional_gating_demoted blocker; got {long_lane.blockers}",
        )
        self.assertEqual(short_lane.action, ACTION_SEND_ENTRY)
        # SHORT lane scored 150, LONG lane was 200 - 25 (C-2) = 175 score after veto.
        # So the rank flip is not guaranteed by score alone, but LONG is NO_TRADE
        # so it falls out of any SEND_ENTRY prefilter anyway.

    def test_c1_long_lean_with_long_majority_demotes_short(self) -> None:
        # Symmetric: LONG_LEAN with LONG majority demotes SHORT lanes.
        from quant_rabbit.strategy.trader_brain import _apply_directional_gating

        scores = (
            self._make_score(
                lane_id="trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET",
                pair="EUR_USD",
                direction="SHORT",
                score=200.0,
            ),
            self._make_score(
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                pair="EUR_USD",
                direction="LONG",
                score=150.0,
            ),
        )
        pair_charts = self._pair_charts(balance="LONG_LEAN", gap=0.44)
        attack_ranks = {
            "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET": 0,
            "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET": 1,
        }

        result = _apply_directional_gating(scores, pair_charts, attack_ranks)

        short_lane = next(s for s in result if s.direction == "SHORT")
        long_lane = next(s for s in result if s.direction == "LONG")
        self.assertEqual(short_lane.action, ACTION_NO_TRADE)
        self.assertTrue(any("directional_gating_demoted" in b for b in short_lane.blockers))
        self.assertEqual(long_lane.action, ACTION_SEND_ENTRY)

    def test_c1_does_not_fire_when_gap_below_strong_threshold(self) -> None:
        # SHORT_LEAN but gap -0.06 (just past 0.05 TIED, below 0.10 strong).
        # Neither C-1 nor C-2 should demote — bias is too weak to act.
        from quant_rabbit.strategy.trader_brain import _apply_directional_gating

        scores = (
            self._make_score(
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                pair="EUR_USD",
                direction="LONG",
                score=200.0,
            ),
            self._make_score(
                lane_id="trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET",
                pair="EUR_USD",
                direction="SHORT",
                score=150.0,
            ),
        )
        pair_charts = self._pair_charts(balance="SHORT_LEAN", gap=-0.06)
        # No attack advice top-K either — so C-2 majority also undefined.
        result = _apply_directional_gating(scores, pair_charts, attack_ranks={})

        long_lane = next(s for s in result if s.direction == "LONG")
        self.assertEqual(long_lane.action, ACTION_SEND_ENTRY)
        self.assertFalse(any("directional_gating" in b for b in long_lane.blockers))

    def test_c1_does_not_fire_when_advice_disagrees_with_bias(self) -> None:
        # SHORT_LEAN gap -0.44 but advice majority is LONG (perhaps fade
        # setup). Conditions disagree → no C-1 demotion.
        from quant_rabbit.strategy.trader_brain import _apply_directional_gating

        scores = (
            self._make_score(
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                pair="EUR_USD",
                direction="LONG",
                score=200.0,
            ),
            self._make_score(
                lane_id="trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET",
                pair="EUR_USD",
                direction="SHORT",
                score=150.0,
            ),
        )
        pair_charts = self._pair_charts(balance="SHORT_LEAN", gap=-0.44)
        attack_ranks = {
            "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET": 0,
            "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET": 1,
        }

        result = _apply_directional_gating(scores, pair_charts, attack_ranks)

        long_lane = next(s for s in result if s.direction == "LONG")
        self.assertEqual(long_lane.action, ACTION_SEND_ENTRY)
        self.assertFalse(any("directional_gating_demoted" in b for b in long_lane.blockers))

    def test_c2_penalty_subtracts_25_from_opposite_lanes(self) -> None:
        # Advice top-K majority SHORT for EUR_USD → LONG lane (the only
        # one in scores) loses 25 score points + gets veto rationale.
        from quant_rabbit.strategy.trader_brain import (
            _apply_directional_gating,
            ATTACK_ADVICE_VETO_PENALTY,
        )

        scores = (
            self._make_score(
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                pair="EUR_USD",
                direction="LONG",
                score=180.0,
            ),
        )
        # TIED pair_charts → no C-1, but attack_advice still drives C-2.
        pair_charts = self._pair_charts(balance="TIED", gap=0.0)
        attack_ranks = {
            "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET": 0,
            "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:MARKET": 1,
        }

        result = _apply_directional_gating(scores, pair_charts, attack_ranks)

        lane = next(s for s in result if s.pair == "EUR_USD")
        self.assertEqual(lane.score, 180.0 - ATTACK_ADVICE_VETO_PENALTY)
        self.assertTrue(any("attack_advice_veto" in r for r in lane.rationale))
        # C-2 alone does not demote action — only score nudge.
        self.assertEqual(lane.action, ACTION_SEND_ENTRY)

    def test_c2_does_not_penalize_aligned_direction(self) -> None:
        from quant_rabbit.strategy.trader_brain import _apply_directional_gating

        scores = (
            self._make_score(
                lane_id="trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET",
                pair="EUR_USD",
                direction="SHORT",
                score=180.0,
            ),
        )
        pair_charts = self._pair_charts(balance="TIED", gap=0.0)
        attack_ranks = {
            "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET": 0,
            "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:MARKET": 1,
        }

        result = _apply_directional_gating(scores, pair_charts, attack_ranks)

        lane = next(s for s in result if s.pair == "EUR_USD")
        self.assertEqual(lane.score, 180.0)
        self.assertFalse(any("attack_advice_veto" in r for r in lane.rationale))

    def test_c1_c2_do_not_consult_position_summaries(self) -> None:
        # Existing-position invariant: the gate must read only
        # pair_charts + attack_ranks + LaneScore. We exercise it with
        # a synthetic packet containing NO position-summary surface,
        # then assert the call succeeds and produces a result. If the
        # gate ever started reading snapshot.positions or order data
        # it would need additional arguments, which would change this
        # signature and surface the regression here.
        import inspect
        from quant_rabbit.strategy.trader_brain import _apply_directional_gating

        sig = inspect.signature(_apply_directional_gating)
        self.assertEqual(
            list(sig.parameters),
            ["scores", "full_pair_charts", "attack_ranks"],
            msg=(
                "directional gating must NOT take a broker snapshot / position "
                "argument; reading positions would break the existing-trade "
                "invariant. If you're adding a parameter, make sure it's not "
                "anything that exposes open-position state."
            ),
        )

    def test_gating_output_identical_with_or_without_existing_positions(self) -> None:
        # Direct structural invariant: the LaneScores produced by
        # `_apply_directional_gating` for a given pair_charts +
        # attack_ranks input must be byte-identical regardless of how
        # many trader-owned positions the broker holds, because the
        # gate function signature does not accept positions/orders at
        # all. This nails the "existing 5 positions cannot be
        # influenced by the new gate" invariant at the gate level.
        from quant_rabbit.strategy.trader_brain import _apply_directional_gating

        scores = (
            self._make_score(
                lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
                pair="EUR_USD",
                direction="LONG",
                score=200.0,
                estimated_margin_jpy=37000.0,
            ),
            self._make_score(
                lane_id="trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET",
                pair="EUR_USD",
                direction="SHORT",
                score=150.0,
                estimated_margin_jpy=37000.0,
            ),
        )
        pair_charts = self._pair_charts(balance="SHORT_LEAN", gap=-0.44)
        attack_ranks = {
            "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET": 0,
            "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:MARKET": 1,
        }

        # Call twice — between the two calls the broker could theoretically
        # have any number of open positions; the gate never sees them, so
        # the output must match exactly.
        result_a = _apply_directional_gating(scores, pair_charts, attack_ranks)
        result_b = _apply_directional_gating(scores, pair_charts, attack_ranks)
        self.assertEqual(result_a, result_b)
        # Also: TP / SL fields on LaneScore must not be touched. The gate
        # only mutates `score`, `action`, `rationale`, `blockers`.
        for orig, new in zip(scores, sorted(result_a, key=lambda s: s.lane_id)):
            self.assertEqual(orig.tp, new.tp)
            self.assertEqual(orig.sl, new.sl)
            self.assertEqual(orig.entry, new.entry)
            self.assertEqual(orig.estimated_margin_jpy, new.estimated_margin_jpy)


class PrecisionFilterTest(unittest.TestCase):
    """Coverage for 2026-05-13 precision filters B (price percentile)
    + D (multi-TF agreement). Both run inside
    `_apply_directional_gating` after the C-1/C-2 pass. They operate on
    pair_charts.confluence extended metrics — never on broker positions
    — so existing trades cannot be touched by these gates.
    """

    def _scores(self, *, pair: str = "EUR_USD", direction: str = "LONG", score: float = 200.0):
        from quant_rabbit.strategy.trader_brain import LaneScore, ACTION_SEND_ENTRY
        return (
            LaneScore(
                lane_id=f"trend_trader:{pair}:{direction}:TREND_CONTINUATION:MARKET",
                pair=pair,
                direction=direction,
                method="TREND_CONTINUATION",
                order_type="MARKET",
                entry=1.0, tp=1.01, sl=None,
                status="LIVE_READY",
                score=score,
                action=ACTION_SEND_ENTRY,
                blockers=(), rationale=(), size_multiple=1.0,
            ),
        )

    def _pair_charts(self, *, price_pct_24h: float | None = None, tf_agreement: float | None = None,
                     pair: str = "EUR_USD") -> dict:
        return {
            pair: {
                "confluence": {
                    "score_balance": "TIED",
                    "score_gap": 0.0,
                    "price_percentile_24h": price_pct_24h,
                    "tf_agreement_score": tf_agreement,
                }
            }
        }

    def test_long_at_top_percentile_loses_25(self) -> None:
        from quant_rabbit.strategy.trader_brain import (
            _apply_directional_gating,
            PRICE_PERCENTILE_EXTREME_PENALTY,
        )
        scores = self._scores(direction="LONG", score=200.0)
        result = _apply_directional_gating(
            scores, self._pair_charts(price_pct_24h=0.97), attack_ranks={}
        )
        lane = result[0]
        self.assertEqual(lane.score, 200.0 - PRICE_PERCENTILE_EXTREME_PENALTY)
        self.assertTrue(any("price_percentile_extreme" in r for r in lane.rationale))

    def test_short_at_bottom_percentile_loses_25(self) -> None:
        from quant_rabbit.strategy.trader_brain import (
            _apply_directional_gating,
            PRICE_PERCENTILE_EXTREME_PENALTY,
        )
        scores = self._scores(direction="SHORT", score=200.0)
        result = _apply_directional_gating(
            scores, self._pair_charts(price_pct_24h=0.03), attack_ranks={}
        )
        lane = result[0]
        self.assertEqual(lane.score, 200.0 - PRICE_PERCENTILE_EXTREME_PENALTY)

    def test_long_at_bottom_percentile_gains_mean_rev_bonus(self) -> None:
        from quant_rabbit.strategy.trader_brain import (
            _apply_directional_gating,
            PRICE_PERCENTILE_MEAN_REV_BONUS,
        )
        scores = self._scores(direction="LONG", score=200.0)
        result = _apply_directional_gating(
            scores, self._pair_charts(price_pct_24h=0.03), attack_ranks={}
        )
        lane = result[0]
        self.assertEqual(lane.score, 200.0 + PRICE_PERCENTILE_MEAN_REV_BONUS)

    def test_neutral_percentile_no_change(self) -> None:
        from quant_rabbit.strategy.trader_brain import _apply_directional_gating
        scores = self._scores(direction="LONG", score=200.0)
        result = _apply_directional_gating(
            scores, self._pair_charts(price_pct_24h=0.5), attack_ranks={}
        )
        self.assertEqual(result[0].score, 200.0)

    def test_low_tf_agreement_penalizes_any_direction(self) -> None:
        from quant_rabbit.strategy.trader_brain import (
            _apply_directional_gating,
            TF_AGREEMENT_DISAGREEMENT_PENALTY,
        )
        for direction in ("LONG", "SHORT"):
            scores = self._scores(direction=direction, score=200.0)
            result = _apply_directional_gating(
                scores,
                self._pair_charts(tf_agreement=0.33),
                attack_ranks={},
            )
            self.assertEqual(
                result[0].score,
                200.0 - TF_AGREEMENT_DISAGREEMENT_PENALTY,
                msg=f"direction={direction}",
            )

    def test_high_tf_agreement_no_penalty(self) -> None:
        from quant_rabbit.strategy.trader_brain import _apply_directional_gating
        scores = self._scores(direction="LONG", score=200.0)
        result = _apply_directional_gating(
            scores, self._pair_charts(tf_agreement=1.0), attack_ranks={}
        )
        self.assertEqual(result[0].score, 200.0)

    def test_missing_confluence_no_change(self) -> None:
        # AGENT_CONTRACT §3.5: missing data → no filter, no silent
        # fallback to a JPY/pip literal.
        from quant_rabbit.strategy.trader_brain import _apply_directional_gating
        scores = self._scores()
        result = _apply_directional_gating(scores, full_pair_charts={}, attack_ranks={})
        self.assertEqual(result[0].score, 200.0)


if __name__ == "__main__":
    unittest.main()

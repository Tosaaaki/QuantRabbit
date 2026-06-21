from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from quant_rabbit.models import BrokerOrder, BrokerPosition, BrokerSnapshot, Owner, Quote, Side, TradeMethod
from quant_rabbit.risk import RiskPolicy
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
    LaneScore,
    TraderBrain,
    _contaminated_pending_order_ids,
    _forecast_market_support_allows_low_confidence_live_ready,
    _micro_structure_alignment_score,
    _micro_structure_direction,
    _mtf_confluence_score,
    _narrative_risk_score,
    _oanda_universal_rotation_precision_score,
    _parse_chart_story_full,
    _forecast_lane_gate,
    _selection_reward_risk_floor,
    _short_term_momentum_class,
    _tf_lens_support,
    _tf_strength_multiplier,
)
from quant_rabbit.strategy.directional_forecaster import DirectionalForecast
from quant_rabbit.strategy.entry_thesis_ledger import PendingEntryThesis, record_pending_entry_thesis
from quant_rabbit.strategy.lane_history_ledger import LaneHistorySnapshot


class TraderBrainTest(unittest.TestCase):
    def test_selection_reward_risk_floor_uses_range_policy_floor(self) -> None:
        policy = RiskPolicy()

        self.assertEqual(
            _selection_reward_risk_floor(TradeMethod.RANGE_ROTATION.value, {"regime_state": "RANGE"}),
            policy.range_min_reward_risk,
        )
        self.assertEqual(
            _selection_reward_risk_floor(TradeMethod.BREAKOUT_FAILURE.value, {"geometry_model": "RANGE_DIRECTIONAL_MARKET"}),
            policy.range_min_reward_risk,
        )
        self.assertEqual(
            _selection_reward_risk_floor(TradeMethod.BREAKOUT_FAILURE.value, {"regime_state": "TREND_UP"}),
            policy.min_reward_risk,
        )

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
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=root / "missing_pair_charts.json",
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

    def test_live_ready_lane_keeps_send_entry_when_only_selection_spread_is_wide(self) -> None:
        """RiskEngine/Gateway own executable spread validation.

        TraderBrain may penalize a wide-spread LIVE_READY lane in ranking, but
        must not add a fixed-pip blocker after the lane already cleared current
        spread economics.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents_path = _eur_only_intents(root)
            payload = json.loads(intents_path.read_text())
            payload["results"][0]["risk_metrics"]["spread_pips"] = 2.4
            intents_path.write_text(json.dumps(payload))
            brain = TraderBrain(
                intents_path=intents_path,
                campaign_plan_path=_eur_only_campaign(root),
                strategy_profile_path=_eur_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            self.assertEqual(decision.action, ACTION_SEND_ENTRY)
            self.assertEqual(decision.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION")
            lane = decision.scores[0]
            self.assertEqual(lane.action, ACTION_SEND_ENTRY)
            self.assertFalse(any("wide spread for fresh edge" in item for item in lane.blockers))
            self.assertTrue(any("wide spread=2.4pip is advisory" in item for item in lane.rationale))

    def test_technical_precision_scores_positive_and_blocks_negative_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = _result(
                "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                "EUR_USD",
                "SHORT",
                "BREAKOUT_FAILURE",
            )
            bad = _result(
                "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:bad",
                "EUR_USD",
                "SHORT",
                "BREAKOUT_FAILURE",
            )
            for lane in (good, bad):
                lane["intent"]["order_type"] = "LIMIT"
                lane["intent"]["entry"] = 1.17330
                lane["intent"]["tp"] = 1.17280
                lane["intent"]["sl"] = 1.17370
                lane["intent"]["metadata"] = {
                    "forecast_direction": "DOWN",
                    "forecast_confidence": 0.23,
                    "chart_direction_bias": "SHORT",
                    "m1_atr_percentile_100": 0.10,
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                    "tp_target_intent": "HARVEST",
                    "opportunity_mode": "HARVEST",
                }
            good["intent"]["metadata"]["m5_ema_slope_5"] = -0.20
            bad["intent"]["metadata"]["m5_ema_slope_5"] = 0.20
            intents = root / "technical_precision_intents.json"
            intents.write_text(json.dumps({"results": [bad, good]}))
            campaign = root / "technical_precision_campaign.json"
            campaign.write_text(
                json.dumps(
                    {
                        "lanes": [
                            _lane("failure_trader", "EUR_USD", "SHORT", "BREAKOUT_FAILURE"),
                        ]
                    }
                )
            )
            brain = TraderBrain(
                intents_path=intents,
                campaign_plan_path=campaign,
                strategy_profile_path=_opposite_market_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            good_score = next(
                item for item in decision.scores
                if item.lane_id == "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"
            )
            bad_score = next(item for item in decision.scores if item.lane_id.endswith(":bad"))
            self.assertEqual(good_score.action, ACTION_SEND_ENTRY)
            self.assertEqual(bad_score.action, ACTION_NO_TRADE)
            self.assertGreater(good_score.score, bad_score.score)
            self.assertTrue(any("technical precision +24.0" in item for item in good_score.rationale))
            self.assertTrue(any("technical_harvest_negative_bucket" in item for item in bad_score.blockers))
            self.assertEqual(decision.selected_lane_id, good_score.lane_id)

    def test_bidask_replay_scores_positive_segment_and_blocks_negative_pair_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = _result(
                "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                "EUR_USD",
                "SHORT",
                "BREAKOUT_FAILURE",
            )
            bad = _result(
                "failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE:bad",
                "AUD_JPY",
                "LONG",
                "BREAKOUT_FAILURE",
            )
            contrarian = _result(
                "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:contrarian",
                "AUD_JPY",
                "SHORT",
                "BREAKOUT_FAILURE",
            )
            good["intent"]["order_type"] = "LIMIT"
            good["intent"]["entry"] = 1.17330
            good["intent"]["tp"] = 1.17280
            good["intent"]["sl"] = 1.17400
            good["intent"]["metadata"] = {
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.23,
                "chart_direction_bias": "SHORT",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
            }
            bad["intent"]["metadata"] = {
                "forecast_direction": "UP",
                "forecast_confidence": 0.87,
                "chart_direction_bias": "LONG",
            }
            contrarian["intent"]["order_type"] = "LIMIT"
            contrarian["intent"]["entry"] = 114.289
            contrarian["intent"]["tp"] = 114.189
            contrarian["intent"]["sl"] = 114.359
            contrarian["intent"]["metadata"] = {
                "forecast_direction": "UP",
                "forecast_confidence": 0.80,
                "forecast_horizon_min": 60,
                "chart_direction_bias": "LONG",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
            }
            intents = root / "bidask_precision_intents.json"
            intents.write_text(json.dumps({"results": [bad, good, contrarian]}))
            campaign = root / "bidask_precision_campaign.json"
            campaign.write_text(
                json.dumps(
                    {
                        "lanes": [
                            _lane("failure_trader", "EUR_USD", "SHORT", "BREAKOUT_FAILURE"),
                            _lane("failure_trader", "AUD_JPY", "LONG", "BREAKOUT_FAILURE"),
                            _lane("failure_trader", "AUD_JPY", "SHORT", "BREAKOUT_FAILURE"),
                        ]
                    }
                )
            )
            brain = TraderBrain(
                intents_path=intents,
                campaign_plan_path=campaign,
                strategy_profile_path=_opposite_market_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            good_score = next(
                item for item in decision.scores
                if item.lane_id == "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"
            )
            bad_score = next(item for item in decision.scores if item.lane_id.endswith(":bad"))
            contrarian_score = next(item for item in decision.scores if item.lane_id.endswith(":contrarian"))
            self.assertEqual(good_score.action, ACTION_SEND_ENTRY)
            self.assertEqual(bad_score.action, ACTION_NO_TRADE)
            self.assertTrue(any("bid/ask replay rank-only edge +6.0" in item for item in good_score.rationale))
            self.assertTrue(
                any("bid/ask replay rank-only contrarian edge +6.0" in item for item in contrarian_score.rationale)
            )
            self.assertTrue(any("bidask_replay_negative_bucket" in item for item in bad_score.blockers))
            self.assertEqual(decision.selected_lane_id, good_score.lane_id)

    def test_oanda_universal_rotation_is_rank_only_score_support(self) -> None:
        intent = {
            "metadata": {
                "forecast_direction": "DOWN",
                "chart_direction_bias": "SHORT",
                "m5_atr_percentile_100": 0.82,
                "session_bucket": "ASIA",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
            }
        }
        rationale: list[str] = []

        score = _oanda_universal_rotation_precision_score(
            intent=intent,
            pair="GBP_USD",
            direction="SHORT",
            order_type="LIMIT",
            method="RANGE_ROTATION",
            entry=1.30000,
            tp=1.29950,
            sl=1.30070,
            rationale=rationale,
        )

        self.assertEqual(score, 10.0)
        self.assertTrue(any("oanda universal rotation +10.0" in item for item in rationale))
        self.assertTrue(any("live_gap=VALIDATION_WIN_RATE_BELOW_90_PERCENT" in item for item in rationale))
        assessment = intent["metadata"]["oanda_universal_rotation_precision_assessment"]
        self.assertIsNone(assessment["primary_support"])
        self.assertTrue(assessment["primary_rank_support"]["rank_only"])

    def test_oanda_universal_rotation_derives_spread_regime_for_all_pair_selector(self) -> None:
        intent = {
            "metadata": {
                "forecast_direction": "DOWN",
                "chart_direction_bias": "SHORT",
                "m5_atr_pips": 5.0,
                "session_bucket": "LONDON_NY_OVERLAP",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
            }
        }
        rationale: list[str] = []

        score = _oanda_universal_rotation_precision_score(
            intent=intent,
            pair="EUR_USD",
            direction="SHORT",
            order_type="LIMIT",
            method="PULLBACK_CONTINUATION",
            entry=1.10000,
            tp=1.09950,
            sl=1.10070,
            spread_pips=1.0,
            rationale=rationale,
        )

        self.assertEqual(score, 8.0)
        self.assertEqual(intent["metadata"]["oanda_m5_spread_regime"], "mid")
        self.assertTrue(any("oanda universal rotation +8.0" in item for item in rationale))
        assessment = intent["metadata"]["oanda_universal_rotation_precision_assessment"]
        self.assertEqual(
            assessment["primary_rank_support"]["name"],
            "EUR_USD_SHORT_M5_PULLBACK_CONTINUATION_SESSION_LONDON_NY_OVERLAP_SPREAD_REGIME_MID_TP1P25_SL1",
        )

    def test_oanda_universal_rotation_scales_down_when_recent_lane_history_degrades(self) -> None:
        intent = {
            "metadata": {
                "forecast_direction": "DOWN",
                "chart_direction_bias": "SHORT",
                "m5_atr_pips": 5.0,
                "session_bucket": "LONDON_NY_OVERLAP",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
            }
        }
        lane_history = {
            ("EUR_USD", "SHORT", "PULLBACK_CONTINUATION"): LaneHistorySnapshot(
                pair="EUR_USD",
                direction="SHORT",
                sample_size=5,
                net_pl_jpy=-4000.0,
                modifier=-25.0,
                method="PULLBACK_CONTINUATION",
            )
        }
        rationale: list[str] = []

        score = _oanda_universal_rotation_precision_score(
            intent=intent,
            pair="EUR_USD",
            direction="SHORT",
            order_type="LIMIT",
            method="PULLBACK_CONTINUATION",
            entry=1.10000,
            tp=1.09950,
            sl=1.10070,
            spread_pips=1.0,
            lane_history=lane_history,
            rationale=rationale,
        )

        self.assertEqual(score, 0.0)
        self.assertTrue(any("recent lane history scales OANDA rotation edge x0.00" in item for item in rationale))
        assessment = intent["metadata"]["oanda_universal_rotation_precision_assessment"]
        self.assertEqual(assessment["raw_score_delta_before_recent_history_scale"], 8.0)
        self.assertEqual(assessment["recent_history_score_scale"], 0.0)

    def test_oanda_universal_rotation_is_size_neutral_when_capture_economics_negative_without_firepower(self) -> None:
        intent = {
            "metadata": {
                "forecast_direction": "DOWN",
                "chart_direction_bias": "SHORT",
                "m5_atr_pips": 5.0,
                "session_bucket": "LONDON_NY_OVERLAP",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "loss_asymmetry_guard_mode": "TP_PROVEN_RELAXED",
                "loss_asymmetry_guard_relaxed": True,
            }
        }
        rationale: list[str] = []

        score = _oanda_universal_rotation_precision_score(
            intent=intent,
            pair="EUR_USD",
            direction="SHORT",
            order_type="LIMIT",
            method="PULLBACK_CONTINUATION",
            entry=1.10000,
            tp=1.09950,
            sl=1.10070,
            spread_pips=1.0,
            rationale=rationale,
        )

        self.assertEqual(score, 0.0)
        self.assertTrue(any("OANDA rank-only rotation edge is size-neutral" in item for item in rationale))
        assessment = intent["metadata"]["oanda_universal_rotation_precision_assessment"]
        self.assertEqual(assessment["raw_score_delta_before_capture_rotation_scale"], 8.0)
        self.assertEqual(assessment["capture_rotation_score_scale"], 0.0)

    def test_oanda_universal_rotation_scores_tp_proven_even_when_daily_floor_unproved(self) -> None:
        intent = {
            "metadata": {
                "forecast_direction": "DOWN",
                "chart_direction_bias": "SHORT",
                "m5_atr_pips": 5.0,
                "session_bucket": "LONDON_NY_OVERLAP",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "positive_rotation_live_ready": True,
                "positive_rotation_minimum_floor_reachable": False,
            }
        }
        rationale: list[str] = []

        score = _oanda_universal_rotation_precision_score(
            intent=intent,
            pair="EUR_USD",
            direction="SHORT",
            order_type="LIMIT",
            method="PULLBACK_CONTINUATION",
            entry=1.10000,
            tp=1.09950,
            sl=1.10070,
            spread_pips=1.0,
            rationale=rationale,
        )

        self.assertEqual(score, 8.0)
        self.assertTrue(any("do not treat the daily floor as solved" in item for item in rationale))
        assessment = intent["metadata"]["oanda_universal_rotation_precision_assessment"]
        self.assertEqual(
            assessment["capture_rotation_rationale"],
            "positive rotation lacks daily 5% floor firepower proof; keep OANDA rank-only ordering active "
            "but do not treat the daily floor as solved",
        )
        self.assertNotIn("capture_rotation_score_scale", assessment)

    def test_oanda_universal_rotation_scores_when_positive_rotation_firepower_is_proved(self) -> None:
        intent = {
            "metadata": {
                "forecast_direction": "DOWN",
                "chart_direction_bias": "SHORT",
                "m5_atr_pips": 5.0,
                "session_bucket": "LONDON_NY_OVERLAP",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
                "opportunity_mode": "HARVEST",
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "positive_rotation_live_ready": True,
                "positive_rotation_minimum_floor_reachable": True,
            }
        }
        rationale: list[str] = []

        score = _oanda_universal_rotation_precision_score(
            intent=intent,
            pair="EUR_USD",
            direction="SHORT",
            order_type="LIMIT",
            method="PULLBACK_CONTINUATION",
            entry=1.10000,
            tp=1.09950,
            sl=1.10070,
            spread_pips=1.0,
            rationale=rationale,
        )

        self.assertEqual(score, 8.0)
        assessment = intent["metadata"]["oanda_universal_rotation_precision_assessment"]
        self.assertNotIn("capture_rotation_score_scale", assessment)

    def test_technical_rotation_scores_high_frequency_bucket_without_live_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rotation = _result(
                "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                "EUR_USD",
                "SHORT",
                "BREAKOUT_FAILURE",
            )
            plain = _result(
                "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:plain",
                "EUR_USD",
                "SHORT",
                "BREAKOUT_FAILURE",
            )
            for lane in (rotation, plain):
                lane["intent"]["order_type"] = "LIMIT"
                lane["intent"]["entry"] = 1.17330
                lane["intent"]["tp"] = 1.17280
                lane["intent"]["sl"] = 1.17370
                lane["intent"]["metadata"] = {
                    "forecast_direction": "DOWN",
                    "forecast_confidence": 0.63,
                    "chart_direction_bias": "SHORT",
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                    "tp_target_intent": "HARVEST",
                    "opportunity_mode": "HARVEST",
                }
            rotation["intent"]["metadata"].update(
                {
                    "m5_bb_pct_b": 0.40,
                    "m5_atr_percentile_100": 0.80,
                }
            )
            intents = root / "technical_rotation_intents.json"
            intents.write_text(json.dumps({"results": [plain, rotation]}))
            campaign = root / "technical_rotation_campaign.json"
            campaign.write_text(
                json.dumps(
                    {
                        "lanes": [
                            _lane("failure_trader", "EUR_USD", "SHORT", "BREAKOUT_FAILURE"),
                        ]
                    }
                )
            )
            brain = TraderBrain(
                intents_path=intents,
                campaign_plan_path=campaign,
                strategy_profile_path=_opposite_market_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            rotation_score = next(
                item for item in decision.scores
                if item.lane_id == "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"
            )
            plain_score = next(item for item in decision.scores if item.lane_id.endswith(":plain"))
            self.assertEqual(rotation_score.action, ACTION_SEND_ENTRY)
            self.assertGreater(rotation_score.score, plain_score.score)
            self.assertTrue(any("technical rotation +18.0" in item for item in rotation_score.rationale))
            self.assertEqual(decision.selected_lane_id, rotation_score.lane_id)

    def test_cycle_level_projection_and_correlation_context_is_built_once_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_charts_path = root / "pair_charts.json"
            pair_charts_path.write_text(
                json.dumps(
                    {
                        "charts": [
                            {
                                "pair": "AUD_JPY",
                                "views": [
                                    {
                                        "granularity": "M15",
                                        "recent_candles": [{"c": 112.0 + idx * 0.01} for idx in range(12)],
                                    }
                                ],
                            },
                            {
                                "pair": "EUR_USD",
                                "views": [
                                    {
                                        "granularity": "M15",
                                        "recent_candles": [{"c": 1.1700 + idx * 0.0001} for idx in range(12)],
                                    }
                                ],
                            },
                        ]
                    }
                )
            )
            built_map = {("AUD_JPY", "EUR_USD"): 0.8, ("EUR_USD", "AUD_JPY"): 0.8}
            seen_maps: list[dict[tuple[str, str], float] | None] = []

            def observed_detect(pair, charts, *, correlation_map=None):
                seen_maps.append(correlation_map)
                return []

            brain = TraderBrain(
                intents_path=_intents(root),
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=pair_charts_path,
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            with (
                mock.patch(
                    "quant_rabbit.strategy.trader_brain.build_correlation_map",
                    return_value=built_map,
                ) as build_mock,
                mock.patch(
                    "quant_rabbit.strategy.trader_brain.detect_correlation_lag",
                    side_effect=observed_detect,
                ),
                mock.patch(
                    "quant_rabbit.strategy.projection_ledger.compute_hit_rates",
                    return_value={},
                ) as hit_rates_mock,
            ):
                brain.run(_snapshot())

            self.assertEqual(build_mock.call_count, 1)
            self.assertEqual(hit_rates_mock.call_count, 1)
            self.assertGreaterEqual(len(seen_maps), 2)
            self.assertTrue(all(correlation_map is built_map for correlation_map in seen_maps))

    def test_existing_pending_order_does_not_force_monitor_only(self) -> None:
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

            self.assertEqual(decision.action, ACTION_SEND_ENTRY)
            self.assertIsNotNone(decision.selected_lane_id)
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

            self.assertEqual(decision.action, ACTION_SEND_ENTRY)
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

            self.assertEqual(decision.action, ACTION_SEND_ENTRY)
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

    def test_keeps_recovery_hedge_pending_despite_short_term_opposition(self) -> None:
        now = datetime.now(timezone.utc)
        pending = BrokerOrder(
            order_id="recovery-short-stop",
            pair="EUR_USD",
            order_type="STOP",
            price=1.16049,
            state="PENDING",
            units=-22_000,
            owner=Owner.TRADER,
            raw={
                "clientExtensions": {
                    "tag": "trader",
                    "comment": "qr-vnext failure_trader NOW_OR_BACKUP",
                },
                "takeProfitOnFill": {"price": "1.15675"},
            },
        )
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(
                BrokerPosition(
                    trade_id="trapped-long",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=22_000,
                    entry_price=1.16688,
                    owner=Owner.TRADER,
                    unrealized_pl_jpy=-23_000.0,
                ),
            ),
            orders=(pending,),
            quotes={"EUR_USD": Quote("EUR_USD", 1.16046, 1.16054, timestamp_utc=now)},
        )
        score = LaneScore(
            lane_id="failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
            pair="EUR_USD",
            direction="SHORT",
            method="BREAKOUT_FAILURE",
            order_type="STOP-ENTRY",
            entry=1.16030,
            tp=1.15669,
            sl=1.16133,
            status="LIVE_READY",
            score=116.78,
            action=ACTION_NO_TRADE,
            blockers=(
                "micro_structure_opposed: M1+M5 both struct opposite to SHORT",
                "forecast confidence 0.22 < 0.55 threshold",
            ),
            rationale=("recovery hedge may run against the stale score while it monetizes trapped exposure",),
            spread_pips=0.8,
            estimated_margin_jpy=0.0,
            hedge_recovery=True,
        )

        self.assertEqual(_contaminated_pending_order_ids(snapshot, (score,)), ())

    def test_aligned_forecast_text_does_not_contaminate_pending_entry(self) -> None:
        now = datetime.now(timezone.utc)
        pending = BrokerOrder(
            order_id="aligned-forecast-limit",
            pair="EUR_USD",
            order_type="LIMIT",
            price=1.17120,
            state="PENDING",
            units=1000,
            owner=Owner.TRADER,
            raw={
                "clientExtensions": {"tag": "trader"},
                "takeProfitOnFill": {"price": "1.17360"},
            },
        )
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            orders=(pending,),
            quotes={"EUR_USD": Quote("EUR_USD", 1.17110, 1.17118, timestamp_utc=now)},
        )
        score = LaneScore(
            lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
            pair="EUR_USD",
            direction="LONG",
            method="BREAKOUT_FAILURE",
            order_type="LIMIT",
            entry=1.17120,
            tp=1.17360,
            sl=None,
            status="LIVE_READY",
            score=18.0,
            action=ACTION_NO_TRADE,
            blockers=("wide spread for fresh edge=2.4pip",),
            rationale=("forecast UP aligned LONG; keep waiting for retest fill",),
            spread_pips=2.4,
            estimated_margin_jpy=12_000.0,
        )

        self.assertEqual(_contaminated_pending_order_ids(snapshot, (score,)), ())

    def test_snapshot_reuse_keeps_pending_with_ledger_thesis_when_forecast_softens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            record_pending_entry_thesis(
                PendingEntryThesis(
                    timestamp_utc=now.isoformat(),
                    order_id="ledger-backed-stop",
                    pair="GBP_CAD",
                    side="LONG",
                    entry_price=1.86796,
                    forecast_direction="UP",
                    forecast_confidence=0.636,
                    regime="UNCLEAR",
                    invalidation_price=1.86646,
                    target_price=1.86753,
                    key_drivers=["forecast UP at entry"],
                    lane_id="trend_trader:GBP_CAD:LONG:TREND_CONTINUATION",
                ),
                root,
            )
            pending = BrokerOrder(
                order_id="ledger-backed-stop",
                pair="GBP_CAD",
                order_type="STOP",
                price=1.86796,
                state="PENDING",
                units=3000,
                owner=Owner.TRADER,
                raw={},
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                orders=(pending,),
                quotes={"GBP_CAD": Quote("GBP_CAD", 1.86720, 1.86730, timestamp_utc=now)},
            )
            score = LaneScore(
                lane_id="trend_trader:GBP_CAD:LONG:TREND_CONTINUATION",
                pair="GBP_CAD",
                direction="LONG",
                method="TREND_CONTINUATION",
                order_type="STOP-ENTRY",
                entry=1.86796,
                tp=1.87852,
                sl=None,
                status="DRY_RUN_PASSED",
                score=22.0,
                action=ACTION_NO_TRADE,
                blockers=("forecast confidence 0.48 < 0.55 threshold",),
                rationale=("forecast UP conf 0.48 too low",),
                spread_pips=1.0,
                estimated_margin_jpy=10_000.0,
            )

            self.assertEqual(_contaminated_pending_order_ids(snapshot, (score,), data_root=root), ())

    def test_keeps_pending_when_invalidation_only_wicks_inside_buffer(self) -> None:
        now = datetime.now(timezone.utc)
        pending = BrokerOrder(
            order_id="buffered-long-stop",
            pair="EUR_USD",
            order_type="STOP",
            price=1.18000,
            state="PENDING",
            units=1000,
            owner=Owner.TRADER,
            raw={
                "createTime": now.isoformat(),
                "stopLossOnFill": {"price": "1.17200"},
            },
        )
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            orders=(pending,),
            quotes={"EUR_USD": Quote("EUR_USD", 1.17190, 1.17202, timestamp_utc=now)},
        )
        score = LaneScore(
            lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
            pair="EUR_USD",
            direction="LONG",
            method="TREND_CONTINUATION",
            order_type="STOP-ENTRY",
            entry=1.18000,
            tp=1.18400,
            sl=1.17200,
            status="DRY_RUN_PASSED",
            score=18.0,
            action=ACTION_NO_TRADE,
            blockers=("forecast confidence 0.48 < 0.55 threshold",),
            rationale=("same-side thesis remains represented",),
            spread_pips=1.2,
            estimated_margin_jpy=10_000.0,
        )

        self.assertEqual(_contaminated_pending_order_ids(snapshot, (score,)), ())

    def test_cancels_pending_when_invalidation_clears_buffer(self) -> None:
        now = datetime.now(timezone.utc)
        pending = BrokerOrder(
            order_id="buffer-breached-long-stop",
            pair="EUR_USD",
            order_type="STOP",
            price=1.18000,
            state="PENDING",
            units=1000,
            owner=Owner.TRADER,
            raw={
                "createTime": now.isoformat(),
                "stopLossOnFill": {"price": "1.17200"},
            },
        )
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            orders=(pending,),
            quotes={"EUR_USD": Quote("EUR_USD", 1.17175, 1.17187, timestamp_utc=now)},
        )
        score = LaneScore(
            lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
            pair="EUR_USD",
            direction="LONG",
            method="TREND_CONTINUATION",
            order_type="STOP-ENTRY",
            entry=1.18000,
            tp=1.18400,
            sl=1.17200,
            status="DRY_RUN_PASSED",
            score=18.0,
            action=ACTION_NO_TRADE,
            blockers=("forecast confidence 0.48 < 0.55 threshold",),
            rationale=("same-side thesis remains represented",),
            spread_pips=1.2,
            estimated_margin_jpy=10_000.0,
        )

        self.assertEqual(
            _contaminated_pending_order_ids(snapshot, (score,)),
            ("buffer-breached-long-stop",),
        )

    def test_cancels_pending_when_current_pair_scores_only_opposite_side(self) -> None:
        now = datetime.now(timezone.utc)
        pending = BrokerOrder(
            order_id="opposite-only-short-limit",
            pair="AUD_CAD",
            order_type="LIMIT",
            price=0.98980,
            state="PENDING",
            units=-8000,
            owner=Owner.TRADER,
            raw={
                "createTime": now.isoformat(),
                "clientExtensions": {"tag": "trader"},
                "takeProfitOnFill": {"price": "0.98870"},
                "stopLossOnFill": {"price": "0.99501"},
            },
        )
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            orders=(pending,),
            quotes={"AUD_CAD": Quote("AUD_CAD", 0.98942, 0.98950, timestamp_utc=now)},
        )
        opposite_only = LaneScore(
            lane_id="range_trader:AUD_CAD:LONG:RANGE_ROTATION",
            pair="AUD_CAD",
            direction="LONG",
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=0.98913,
            tp=0.99023,
            sl=0.98741,
            status="DRY_RUN_BLOCKED",
            score=24.0,
            action=ACTION_NO_TRADE,
            blockers=("forecast watch-only", "range location chase"),
            rationale=("current packet exposes only AUD_CAD LONG candidates",),
            spread_pips=2.2,
            estimated_margin_jpy=36_000.0,
        )

        self.assertEqual(
            _contaminated_pending_order_ids(snapshot, (opposite_only,)),
            ("opposite-only-short-limit",),
        )

    def test_recent_cancel_regret_preserves_anchored_pending_against_weak_opposite_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_cancel_regret_audit(root, pair="AUD_CAD", side="SHORT", order_type="LIMIT_ORDER")
            now = datetime.now(timezone.utc)
            pending = BrokerOrder(
                order_id="regret-short-limit",
                pair="AUD_CAD",
                order_type="LIMIT",
                price=0.98980,
                state="PENDING",
                units=-8000,
                owner=Owner.TRADER,
                raw={
                    "createTime": now.isoformat(),
                    "clientExtensions": {"tag": "trader"},
                    "takeProfitOnFill": {"price": "0.98870"},
                    "stopLossOnFill": {"price": "0.99501"},
                },
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                orders=(pending,),
                quotes={"AUD_CAD": Quote("AUD_CAD", 0.98942, 0.98950, timestamp_utc=now)},
            )
            weak_opposite = LaneScore(
                lane_id="range_trader:AUD_CAD:LONG:RANGE_ROTATION",
                pair="AUD_CAD",
                direction="LONG",
                method="RANGE_ROTATION",
                order_type="LIMIT",
                entry=0.98913,
                tp=0.99023,
                sl=0.98741,
                status="DRY_RUN_BLOCKED",
                score=24.0,
                action=ACTION_NO_TRADE,
                blockers=("forecast watch-only", "range location chase"),
                rationale=("current packet exposes only a weak AUD_CAD LONG candidate",),
                spread_pips=2.2,
                estimated_margin_jpy=36_000.0,
            )

            self.assertEqual(_contaminated_pending_order_ids(snapshot, (weak_opposite,), data_root=root), ())

    def test_pending_thesis_horizon_preserves_against_weak_opposite_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            record_pending_entry_thesis(
                PendingEntryThesis(
                    timestamp_utc=(now - timedelta(minutes=12)).isoformat(),
                    order_id="horizon-short-limit",
                    pair="AUD_CAD",
                    side="SHORT",
                    entry_price=0.98980,
                    forecast_direction="RANGE",
                    forecast_confidence=0.81,
                    regime="RANGE",
                    invalidation_price=0.99501,
                    target_price=0.98870,
                    lane_id="range_trader:AUD_CAD:SHORT:RANGE_ROTATION",
                    horizon_hours=1.0,
                ),
                root,
            )
            pending = BrokerOrder(
                order_id="horizon-short-limit",
                pair="AUD_CAD",
                order_type="LIMIT",
                price=0.98980,
                state="PENDING",
                units=-8000,
                owner=Owner.TRADER,
                raw={
                    "createTime": now.isoformat(),
                    "clientExtensions": {"tag": "trader"},
                    "takeProfitOnFill": {"price": "0.98870"},
                    "stopLossOnFill": {"price": "0.99501"},
                },
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                orders=(pending,),
                quotes={"AUD_CAD": Quote("AUD_CAD", 0.98942, 0.98950, timestamp_utc=now)},
            )
            weak_opposite = LaneScore(
                lane_id="range_trader:AUD_CAD:LONG:RANGE_ROTATION",
                pair="AUD_CAD",
                direction="LONG",
                method="RANGE_ROTATION",
                order_type="LIMIT",
                entry=0.98913,
                tp=0.99023,
                sl=0.98741,
                status="DRY_RUN_BLOCKED",
                score=24.0,
                action=ACTION_NO_TRADE,
                blockers=("forecast watch-only", "range location chase"),
                rationale=("current packet exposes only a weak AUD_CAD LONG candidate",),
                spread_pips=2.2,
                estimated_margin_jpy=36_000.0,
            )

            self.assertEqual(_contaminated_pending_order_ids(snapshot, (weak_opposite,), data_root=root), ())

    def test_recent_cancel_regret_does_not_preserve_against_tradeable_opposite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_cancel_regret_audit(root, pair="AUD_CAD", side="SHORT", order_type="LIMIT_ORDER")
            now = datetime.now(timezone.utc)
            pending = BrokerOrder(
                order_id="regret-short-limit",
                pair="AUD_CAD",
                order_type="LIMIT",
                price=0.98980,
                state="PENDING",
                units=-8000,
                owner=Owner.TRADER,
                raw={
                    "createTime": now.isoformat(),
                    "clientExtensions": {"tag": "trader"},
                    "takeProfitOnFill": {"price": "0.98870"},
                    "stopLossOnFill": {"price": "0.99501"},
                },
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                orders=(pending,),
                quotes={"AUD_CAD": Quote("AUD_CAD", 0.98942, 0.98950, timestamp_utc=now)},
            )
            tradeable_opposite = LaneScore(
                lane_id="range_trader:AUD_CAD:LONG:RANGE_ROTATION",
                pair="AUD_CAD",
                direction="LONG",
                method="RANGE_ROTATION",
                order_type="LIMIT",
                entry=0.98913,
                tp=0.99023,
                sl=0.98741,
                status="LIVE_READY",
                score=88.0,
                action=ACTION_SEND_ENTRY,
                blockers=(),
                rationale=("current packet exposes a tradeable AUD_CAD LONG replacement",),
                spread_pips=2.2,
                estimated_margin_jpy=36_000.0,
            )

            self.assertEqual(
                _contaminated_pending_order_ids(snapshot, (tradeable_opposite,), data_root=root),
                ("regret-short-limit",),
            )

    def test_keeps_pending_when_current_packet_has_no_pair_scores(self) -> None:
        now = datetime.now(timezone.utc)
        pending = BrokerOrder(
            order_id="unseen-pair-short-limit",
            pair="AUD_CAD",
            order_type="LIMIT",
            price=0.98980,
            state="PENDING",
            units=-8000,
            owner=Owner.TRADER,
            raw={
                "createTime": now.isoformat(),
                "clientExtensions": {"tag": "trader"},
                "takeProfitOnFill": {"price": "0.98870"},
                "stopLossOnFill": {"price": "0.99501"},
            },
        )
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            orders=(pending,),
            quotes={"AUD_CAD": Quote("AUD_CAD", 0.98942, 0.98950, timestamp_utc=now)},
        )
        unrelated_score = LaneScore(
            lane_id="range_trader:EUR_USD:LONG:RANGE_ROTATION",
            pair="EUR_USD",
            direction="LONG",
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.1600,
            tp=1.1620,
            sl=1.1585,
            status="DRY_RUN_BLOCKED",
            score=12.0,
            action=ACTION_NO_TRADE,
            blockers=("forecast confidence low",),
            rationale=("unrelated pair only",),
            spread_pips=0.8,
            estimated_margin_jpy=20_000.0,
        )

        self.assertEqual(_contaminated_pending_order_ids(snapshot, (unrelated_score,)), ())

    def test_snapshot_reuse_cancels_pending_when_ledger_invalidation_breaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            record_pending_entry_thesis(
                PendingEntryThesis(
                    timestamp_utc=now.isoformat(),
                    order_id="ledger-invalidated-stop",
                    pair="GBP_CAD",
                    side="LONG",
                    entry_price=1.86796,
                    forecast_direction="UP",
                    forecast_confidence=0.636,
                    regime="UNCLEAR",
                    invalidation_price=1.86646,
                    target_price=1.86753,
                    key_drivers=["forecast UP at entry"],
                    lane_id="trend_trader:GBP_CAD:LONG:TREND_CONTINUATION",
                ),
                root,
            )
            pending = BrokerOrder(
                order_id="ledger-invalidated-stop",
                pair="GBP_CAD",
                order_type="STOP",
                price=1.86796,
                state="PENDING",
                units=3000,
                owner=Owner.TRADER,
                raw={},
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                orders=(pending,),
                quotes={"GBP_CAD": Quote("GBP_CAD", 1.86620, 1.86630, timestamp_utc=now)},
            )
            score = LaneScore(
                lane_id="trend_trader:GBP_CAD:LONG:TREND_CONTINUATION",
                pair="GBP_CAD",
                direction="LONG",
                method="TREND_CONTINUATION",
                order_type="STOP-ENTRY",
                entry=1.86796,
                tp=1.87852,
                sl=None,
                status="DRY_RUN_PASSED",
                score=22.0,
                action=ACTION_NO_TRADE,
                blockers=("forecast confidence 0.48 < 0.55 threshold",),
                rationale=("forecast UP conf 0.48 too low",),
                spread_pips=1.0,
                estimated_margin_jpy=10_000.0,
            )

            self.assertEqual(
                _contaminated_pending_order_ids(snapshot, (score,), data_root=root),
                ("ledger-invalidated-stop",),
            )

    def test_opposed_forecast_text_still_contaminates_pending_entry(self) -> None:
        now = datetime.now(timezone.utc)
        pending = BrokerOrder(
            order_id="opposed-forecast-limit",
            pair="EUR_USD",
            order_type="LIMIT",
            price=1.17120,
            state="PENDING",
            units=1000,
            owner=Owner.TRADER,
            raw={
                "clientExtensions": {"tag": "trader"},
                "takeProfitOnFill": {"price": "1.17360"},
            },
        )
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            orders=(pending,),
            quotes={"EUR_USD": Quote("EUR_USD", 1.17110, 1.17118, timestamp_utc=now)},
        )
        score = LaneScore(
            lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
            pair="EUR_USD",
            direction="LONG",
            method="BREAKOUT_FAILURE",
            order_type="LIMIT",
            entry=1.17120,
            tp=1.17360,
            sl=None,
            status="DRY_RUN_PASSED",
            score=-12.0,
            action=ACTION_NO_TRADE,
            blockers=("forecast DOWN opposes LONG -> BLOCK",),
            rationale=("current forecast invalidates the pending long thesis",),
            spread_pips=2.4,
            estimated_margin_jpy=12_000.0,
        )

        self.assertEqual(_contaminated_pending_order_ids(snapshot, (score,)), ("opposed-forecast-limit",))

    def test_keeps_pending_entry_when_market_thesis_still_valid(self) -> None:
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
                        order_id="fresh-stop",
                        pair="EUR_USD",
                        order_type="STOP",
                        price=1.18000,
                        state="PENDING",
                        units=1000,
                        owner=Owner.TRADER,
                        raw={
                            "createTime": datetime.now(timezone.utc).isoformat(),
                            "stopLossOnFill": {"price": "1.16800"},
                        },
                    ),
                )
            )

            decision = brain.run(snapshot)

            self.assertEqual(decision.pending_cancel_order_ids, ())

    def test_keeps_old_pending_when_market_thesis_still_valid(self) -> None:
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
                        order_id="old-stop",
                        pair="EUR_USD",
                        order_type="STOP",
                        price=1.18000,
                        state="PENDING",
                        units=1000,
                        owner=Owner.TRADER,
                        raw={
                            "createTime": (datetime.now(timezone.utc) - timedelta(hours=13)).isoformat(),
                            "stopLossOnFill": {"price": "1.16800"},
                        },
                    ),
                )
            )

            decision = brain.run(snapshot)

            self.assertEqual(decision.pending_cancel_order_ids, ())

    def test_cancels_pending_when_current_market_thesis_flips_opposite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = TraderBrain(
                intents_path=_opposite_market_intents(root),
                campaign_plan_path=_opposite_market_campaign(root),
                strategy_profile_path=_opposite_market_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )
            snapshot = _snapshot(
                orders=(
                    BrokerOrder(
                        order_id="long-stop-flipped",
                        pair="EUR_USD",
                        order_type="STOP",
                        price=1.18000,
                        state="PENDING",
                        units=1000,
                        owner=Owner.TRADER,
                        raw={
                            "createTime": datetime.now(timezone.utc).isoformat(),
                            "stopLossOnFill": {"price": "1.16800"},
                        },
                    ),
                )
            )

            decision = brain.run(snapshot)

            self.assertEqual(decision.pending_cancel_order_ids, ("long-stop-flipped",))

    def test_cancels_fresh_pending_when_stop_side_invalidated(self) -> None:
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
                        order_id="invalidated-stop",
                        pair="EUR_USD",
                        order_type="STOP",
                        price=1.18000,
                        state="PENDING",
                        units=1000,
                        owner=Owner.TRADER,
                        raw={
                            "createTime": datetime.now(timezone.utc).isoformat(),
                            "stopLossOnFill": {"price": "1.17300"},
                        },
                    ),
                )
            )

            decision = brain.run(snapshot)

            self.assertEqual(decision.pending_cancel_order_ids, ("invalidated-stop",))

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

    def test_sl_free_tp_less_runner_can_still_select_portfolio_add(self) -> None:
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
                        trade_id="471232",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=7000,
                        entry_price=1.16768,
                        take_profit=None,
                        stop_loss=None,
                        owner=Owner.TRADER,
                    ),
                )
            )

            prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            prior_tp = os.environ.get("QR_ENABLE_MISSING_TP_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
            try:
                decision = brain.run(snapshot)
            finally:
                if prior_sl is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl
                if prior_tp is None:
                    os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
                else:
                    os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = prior_tp

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

    def test_blocks_live_ready_lane_when_m1_and_m5_structure_oppose_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "intents.json"
            lane = _result(
                "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                "EUR_USD",
                "LONG",
                "TREND_CONTINUATION",
            )
            lane["intent"]["market_context"]["chart_story"] = (
                "EUR_USD TREND_DOWN; "
                "M1(UNCLEAR, ADX=24.6 RSI=33.6 struct=CHOCH_DOWN@1.1649); "
                "M5(TREND_DOWN, ADX=50.7 RSI=25.0 struct=BOS_DOWN@1.1652)"
            )
            path.write_text(json.dumps({"results": [lane]}))
            brain = TraderBrain(
                intents_path=path,
                campaign_plan_path=_eur_only_campaign(root),
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
            self.assertTrue(any("micro_structure_opposed" in blocker for blocker in score.blockers))
            self.assertTrue(any("M1+M5 both struct opposite" in item for item in score.rationale))

    def test_pending_range_rail_limit_tolerates_opposed_micro_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "intents.json"
            campaign_path = root / "range_short_campaign.json"
            campaign_path.write_text(
                json.dumps({"lanes": [_lane("range_trader", "EUR_USD", "SHORT", "RANGE_ROTATION")]})
            )
            lane = _result(
                "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                "EUR_USD",
                "SHORT",
                "RANGE_ROTATION",
            )
            lane["intent"] = {
                **lane["intent"],
                "order_type": "LIMIT",
                "entry": 1.1750,
                "tp": 1.1725,
                "sl": 1.1762,
                "metadata": {
                    "geometry_model": "RANGE_RAIL_LIMIT",
                    "range_support": 1.1710,
                    "range_resistance": 1.1760,
                    "range_tp_is_inside_box": True,
                    "range_sl_outside_box": True,
                    "max_loss_jpy": 100.0,
                },
                "market_context": {
                    **lane["intent"]["market_context"],
                    "chart_story": (
                        "EUR_USD RANGE; "
                        "M1(UNCLEAR, ADX=24.6 RSI=63.6 struct=CHOCH_UP@1.1740); "
                        "M5(TREND_UP, ADX=50.7 RSI=75.0 struct=BOS_UP@1.1742)"
                    ),
                },
            }
            path.write_text(json.dumps({"results": [lane]}))
            brain = TraderBrain(
                intents_path=path,
                campaign_plan_path=campaign_path,
                strategy_profile_path=_opposite_market_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            score = next(item for item in decision.scores if item.pair == "EUR_USD")
            self.assertEqual(score.action, ACTION_SEND_ENTRY)
            self.assertFalse(any("micro_structure_opposed" in blocker for blocker in score.blockers))
            self.assertTrue(any("pending range-rail limit waits for retest" in item for item in score.rationale))

    def test_market_entry_blocks_when_chart_reader_m5_timing_opposes_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "intents.json"
            lane_id = "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET"
            lane = _result(lane_id, "EUR_USD", "SHORT", "TREND_CONTINUATION")
            lane["intent"] = {
                **lane["intent"],
                "order_type": "MARKET",
                "entry": 1.1720,
                "tp": 1.1690,
                "sl": 1.1730,
            }
            path.write_text(json.dumps({"results": [lane]}))
            brain = TraderBrain(
                intents_path=path,
                campaign_plan_path=_opposite_market_campaign(root),
                strategy_profile_path=_opposite_market_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=_entry_timing_pair_charts(root, +1, +1, +1),
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            score = next(item for item in decision.scores if item.lane_id == lane_id)
            self.assertEqual(score.action, ACTION_NO_TRADE)
            self.assertTrue(any("entry_timing_against_market" in blocker for blocker in score.blockers))
            self.assertTrue(any("entry timing hard block" in item for item in score.rationale))

    def test_pending_limit_is_not_hard_blocked_by_against_m5_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "intents.json"
            campaign_path = root / "range_short_campaign.json"
            campaign_path.write_text(
                json.dumps({"lanes": [_lane("range_trader", "EUR_USD", "SHORT", "RANGE_ROTATION")]})
            )
            lane_id = "range_trader:EUR_USD:SHORT:RANGE_ROTATION"
            lane = _result(lane_id, "EUR_USD", "SHORT", "RANGE_ROTATION")
            lane["intent"] = {
                **lane["intent"],
                "order_type": "LIMIT",
                "entry": 1.1750,
                "tp": 1.1725,
                "sl": 1.1762,
                "metadata": {
                    "geometry_model": "RANGE_RAIL_LIMIT",
                    "range_support": 1.1710,
                    "range_resistance": 1.1760,
                    "range_tp_is_inside_box": True,
                    "range_sl_outside_box": True,
                    "max_loss_jpy": 100.0,
                },
            }
            path.write_text(json.dumps({"results": [lane]}))
            brain = TraderBrain(
                intents_path=path,
                campaign_plan_path=campaign_path,
                strategy_profile_path=_opposite_market_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=_entry_timing_pair_charts(root, +1, +1, +1),
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            score = next(item for item in decision.scores if item.lane_id == lane_id)
            self.assertEqual(score.action, ACTION_SEND_ENTRY)
            self.assertFalse(any("entry_timing_against_market" in blocker for blocker in score.blockers))
            self.assertTrue(any("entry timing AGAINST" in item for item in score.rationale))

    def test_pending_range_limit_blocks_when_operating_tfs_strongly_oppose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "intents.json"
            campaign_path = root / "range_short_campaign.json"
            campaign_path.write_text(
                json.dumps({"lanes": [_lane("range_trader", "EUR_USD", "SHORT", "RANGE_ROTATION")]})
            )
            lane_id = "range_trader:EUR_USD:SHORT:RANGE_ROTATION"
            lane = _result(lane_id, "EUR_USD", "SHORT", "RANGE_ROTATION")
            lane["intent"] = {
                **lane["intent"],
                "order_type": "LIMIT",
                "entry": 1.16170,
                "tp": 1.16039,
                "sl": 1.16280,
                "metadata": {
                    "geometry_model": "RANGE_RAIL_LIMIT",
                    "range_support": 1.15860,
                    "range_resistance": 1.16180,
                    "range_tp_is_inside_box": True,
                    "range_sl_outside_box": True,
                    "max_loss_jpy": 100.0,
                },
                "market_context": {
                    **lane["intent"]["market_context"],
                    "chart_story": "EUR_USD RANGE upper rail retest",
                },
            }
            path.write_text(json.dumps({"results": [lane]}))
            brain = TraderBrain(
                intents_path=path,
                campaign_plan_path=campaign_path,
                strategy_profile_path=_opposite_market_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=_operating_tf_momentum_pair_charts(root),
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            score = next(item for item in decision.scores if item.lane_id == lane_id)
            self.assertEqual(score.action, ACTION_NO_TRADE)
            self.assertTrue(any("operating_tf_momentum_opposed" in blocker for blocker in score.blockers))
            self.assertTrue(any("range-rail LIMIT" in item for item in score.rationale))

    def test_preserves_pending_range_limit_when_operating_tfs_strongly_oppose_but_thesis_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "intents.json"
            campaign_path = root / "range_short_campaign.json"
            campaign_path.write_text(
                json.dumps({"lanes": [_lane("range_trader", "EUR_USD", "SHORT", "RANGE_ROTATION")]})
            )
            lane_id = "range_trader:EUR_USD:SHORT:RANGE_ROTATION"
            lane = _result(lane_id, "EUR_USD", "SHORT", "RANGE_ROTATION")
            lane["intent"] = {
                **lane["intent"],
                "order_type": "LIMIT",
                "entry": 1.16170,
                "tp": 1.16039,
                "sl": 1.16280,
                "metadata": {
                    "geometry_model": "RANGE_RAIL_LIMIT",
                    "range_support": 1.15860,
                    "range_resistance": 1.16180,
                    "range_tp_is_inside_box": True,
                    "range_sl_outside_box": True,
                    "max_loss_jpy": 100.0,
                },
                "market_context": {
                    **lane["intent"]["market_context"],
                    "chart_story": "EUR_USD RANGE upper rail retest",
                },
            }
            path.write_text(json.dumps({"results": [lane]}))
            brain = TraderBrain(
                intents_path=path,
                campaign_plan_path=campaign_path,
                strategy_profile_path=_opposite_market_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=_operating_tf_momentum_pair_charts(root),
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )
            snapshot = _snapshot(
                orders=(
                    BrokerOrder(
                        order_id="range-short-limit",
                        pair="EUR_USD",
                        order_type="LIMIT",
                        price=1.16170,
                        state="PENDING",
                        units=-1000,
                        owner=Owner.TRADER,
                        raw={
                            "clientExtensions": {"tag": "trader"},
                            "takeProfitOnFill": {"price": "1.16039"},
                            "stopLossOnFill": {"price": "1.16280"},
                        },
                    ),
                ),
                quotes={"EUR_USD": Quote("EUR_USD", 1.16155, 1.16165)},
            )

            decision = brain.run(snapshot)

            self.assertEqual(decision.pending_cancel_order_ids, ())
            report = (root / "decision.md").read_text()
            self.assertIn("operating TF hard block", report)

    def test_blocks_short_when_technicals_oppose_even_without_trend_regime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "intents.json"
            campaign_path = root / "range_short_campaign.json"
            campaign_path.write_text(
                json.dumps({"lanes": [_lane("range_trader", "EUR_USD", "SHORT", "RANGE_ROTATION")]})
            )
            lane_id = "range_trader:EUR_USD:SHORT:RANGE_ROTATION:MARKET"
            lane = _result(lane_id, "EUR_USD", "SHORT", "RANGE_ROTATION")
            lane["intent"] = {
                **lane["intent"],
                "order_type": "MARKET",
                "metadata": {
                    **lane["intent"].get("metadata", {}),
                    "max_loss_jpy": 100.0,
                },
            }
            path.write_text(json.dumps({"results": [lane]}))
            brain = TraderBrain(
                intents_path=path,
                campaign_plan_path=campaign_path,
                strategy_profile_path=_opposite_market_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=_technical_opposition_pair_charts(root),
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            score = next(item for item in decision.scores if item.lane_id == lane_id)
            self.assertEqual(score.action, ACTION_NO_TRADE)
            self.assertTrue(any("technical_entry_opposed" in blocker for blocker in score.blockers))
            self.assertTrue(any("technical hard block" in item for item in score.rationale))

    def test_live_ready_pending_trigger_not_vetoed_by_missing_profile_or_current_technical_opposition(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lane_id = "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"
                lane = _result(lane_id, "EUR_USD", "SHORT", "TREND_CONTINUATION")
                lane["intent"] = {
                    **lane["intent"],
                    "metadata": {
                        **lane["intent"].get("metadata", {}),
                        "adoption": "ORDER_INTENT_REQUIRED",
                    },
                }
                intents_path = root / "intents.json"
                intents_path.write_text(json.dumps({"results": [lane]}))
                strategy_path = root / "strategy.json"
                strategy_path.write_text(
                    json.dumps(
                        {
                            "system_contract": {
                                "loss_cap_jpy": 500.0,
                                "loss_cap_source": "test current campaign cap",
                            },
                            "profiles": [],
                        }
                    )
                )
                empty_story_path = root / "stories.json"
                empty_story_path.write_text(json.dumps({"pair_profiles": []}))
                empty_campaign_path = root / "campaign.json"
                empty_campaign_path.write_text(json.dumps({"lanes": []}))

                brain = TraderBrain(
                    intents_path=intents_path,
                    campaign_plan_path=empty_campaign_path,
                    strategy_profile_path=strategy_path,
                    market_story_profile_path=empty_story_path,
                    target_state_path=root / "missing_target.json",
                    trader_settings_path=root / "settings.json",
                    attack_advice_path=root / "missing_attack_advice.json",
                    pair_charts_path=_technical_opposition_pair_charts(root),
                    output_path=root / "decision.json",
                    report_path=root / "decision.md",
                )

                decision = brain.run(_snapshot())

                score = next(item for item in decision.scores if item.lane_id == lane_id)
                self.assertEqual(score.action, ACTION_SEND_ENTRY)
                self.assertEqual(decision.selected_lane_id, lane_id)
                blocker_text = " ".join(score.blockers)
                self.assertNotIn("missing strategy profile", blocker_text)
                self.assertNotIn("campaign lane is not executable", blocker_text)
                self.assertNotIn("no positive mined or repaired edge evidence", blocker_text)
                self.assertNotIn("market story does not support", blocker_text)
                self.assertNotIn("technical_entry_opposed", blocker_text)
                self.assertTrue(any("technical caution" in item for item in score.rationale))
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

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

    def test_negative_seat_pnl_ranks_down_without_blocking_live_ready_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy_path = _eur_strategy(root)
            payload = json.loads(strategy_path.read_text())
            payload["profiles"][0]["seat_pl_n"] = 12
            payload["profiles"][0]["seat_net_jpy"] = -3000.0
            payload["profiles"][0]["seat_win_rate_pct"] = 16.7
            strategy_path.write_text(json.dumps(payload))
            brain = TraderBrain(
                intents_path=_eur_only_intents(root),
                campaign_plan_path=_eur_only_campaign(root),
                strategy_profile_path=strategy_path,
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            eur = decision.scores[0]
            self.assertEqual(eur.action, ACTION_SEND_ENTRY)
            self.assertFalse(any("negative seat discovery" in item for item in eur.blockers))
            self.assertIn("negative seat discovery PnL -3000 JPY", " ".join(eur.rationale))

    def test_market_lane_requires_trigger_when_history_and_daily_review_are_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "trader_overrides.json").write_text(
                json.dumps(
                    {
                        "expires_at_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                        "bias_overrides": {"EUR_USD": {"LONG": -20.0}},
                    }
                )
            )
            brain = TraderBrain(
                intents_path=_market_preference_intents(root),
                campaign_plan_path=_mixed_campaign(root),
                strategy_profile_path=_negative_history_strategy(root),
                market_story_profile_path=_stories(root),
                target_state_path=root / "missing_target.json",
                trader_settings_path=root / "settings.json",
                attack_advice_path=root / "missing_attack_advice.json",
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "decision.json",
                report_path=root / "decision.md",
            )

            decision = brain.run(_snapshot())

            market = next(item for item in decision.scores if item.lane_id.endswith(":MARKET"))
            self.assertEqual(market.action, ACTION_NO_TRADE)
            self.assertTrue(any("daily-review headwind" in item for item in market.blockers))
            self.assertEqual(decision.selected_lane_id, "range_trader:EUR_USD:LONG:RANGE_ROTATION")

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


class ForecastLaneGateTest(unittest.TestCase):
    def _forecast(self, direction: str, confidence: float = 0.8) -> DirectionalForecast:
        return DirectionalForecast(
            pair="EUR_USD",
            direction=direction,
            confidence=confidence,
            invalidation_price=None,
            target_price=None,
            horizon_min=60,
            drivers_for=(),
            drivers_against=(),
            rationale_summary="test forecast",
        )

    def test_range_forecast_allows_executable_range_rotation(self) -> None:
        intent = {
            "metadata": {
                "geometry_model": "RANGE_RAIL_LIMIT",
                "range_support": 1.171,
                "range_resistance": 1.176,
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
            }
        }

        ok, reason = _forecast_lane_gate(
            self._forecast("RANGE"),
            direction="LONG",
            method="RANGE_ROTATION",
            intent=intent,
        )

        self.assertTrue(ok)
        self.assertIn("supports range rotation", reason)

    def test_range_forecast_blocks_trend_chase_without_range_geometry(self) -> None:
        ok, reason = _forecast_lane_gate(
            self._forecast("RANGE"),
            direction="LONG",
            method="TREND_CONTINUATION",
            intent={"metadata": {}},
        )

        self.assertFalse(ok)
        self.assertIn("requires executable RANGE_ROTATION", reason)

    def test_unclear_forecast_blocks_fresh_range_rotation(self) -> None:
        intent = {
            "metadata": {
                "geometry_model": "RANGE_RAIL_LIMIT",
                "range_support": 1.171,
                "range_resistance": 1.176,
                "range_tp_is_inside_box": True,
                "range_sl_outside_box": True,
            }
        }

        ok, reason = _forecast_lane_gate(
            self._forecast("UNCLEAR"),
            direction="SHORT",
            method="RANGE_ROTATION",
            intent=intent,
        )

        self.assertFalse(ok)
        self.assertIn("no executable directional or RANGE edge", reason)

    def test_directional_forecast_blocks_opposite_side(self) -> None:
        ok, reason = _forecast_lane_gate(
            self._forecast("UP"),
            direction="SHORT",
            method="TREND_CONTINUATION",
            intent={},
        )

        self.assertFalse(ok)
        self.assertIn("opposes SHORT", reason)

    def test_market_support_allows_near_miss_directional_forecast(self) -> None:
        intent = {
            "metadata": {
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.47,
                "forecast_raw_confidence": 0.64,
                "chart_direction_bias": "SHORT",
                "forecast_market_support": {
                    "ok": True,
                    "direction": "DOWN",
                    "bootstrap_projection_support": True,
                    "aligned_projection_count": 3,
                    "best_hit_rate": 0.94,
                    "best_samples": 100,
                },
            }
        }

        self.assertTrue(
            _forecast_market_support_allows_low_confidence_live_ready(
                intent,
                side="SHORT",
                forecast=self._forecast("DOWN", confidence=0.47),
                min_confidence=0.55,
            )
        )

    def test_market_support_rejects_genuine_weak_directional_forecast(self) -> None:
        intent = {
            "metadata": {
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.36,
                "forecast_raw_confidence": 0.64,
                "chart_direction_bias": "SHORT",
                "forecast_market_support": {
                    "ok": True,
                    "direction": "DOWN",
                    "bootstrap_projection_support": True,
                },
            }
        }

        self.assertFalse(
            _forecast_market_support_allows_low_confidence_live_ready(
                intent,
                side="SHORT",
                forecast=self._forecast("DOWN", confidence=0.36),
                min_confidence=0.55,
            )
        )

    def test_market_support_uses_aligned_hit_rate_not_either_timing_for_direction(self) -> None:
        intent = {
            "metadata": {
                "forecast_direction": "DOWN",
                "forecast_confidence": 0.47,
                "forecast_raw_confidence": 0.64,
                "chart_direction_bias": "SHORT",
                "forecast_market_support": {
                    "ok": True,
                    "direction": "DOWN",
                    "aligned_projection_count": 1,
                    "timing_projection_count": 1,
                    "best_hit_rate": 0.90,
                    "best_samples": 100,
                    "best_aligned_hit_rate": 0.40,
                    "best_aligned_samples": 100,
                    "best_timing_hit_rate": 0.90,
                    "best_timing_samples": 100,
                },
            }
        }

        self.assertFalse(
            _forecast_market_support_allows_low_confidence_live_ready(
                intent,
                side="SHORT",
                forecast=self._forecast("DOWN", confidence=0.47),
                min_confidence=0.55,
            )
        )

    def test_market_support_rejects_timing_only_known_weak_direction_bucket(self) -> None:
        intent = {
            "order_type": "STOP-ENTRY",
            "metadata": {
                "forecast_direction": "UP",
                "forecast_confidence": 0.58,
                "forecast_raw_confidence": 0.66,
                "chart_direction_bias": "LONG",
                "forecast_directional_calibration_name": "directional_forecast_up",
                "forecast_directional_hit_rate": 0.12,
                "forecast_directional_samples": 18,
                "forecast_market_support": {
                    "ok": True,
                    "aligned_projection_count": 0,
                    "timing_projection_count": 1,
                    "best_hit_rate": 0.88,
                    "best_samples": 100,
                    "best_timing_hit_rate": 0.88,
                    "best_timing_samples": 100,
                },
            },
        }

        self.assertFalse(
            _forecast_market_support_allows_low_confidence_live_ready(
                intent,
                side="LONG",
                forecast=self._forecast("UP", confidence=0.58),
                min_confidence=0.65,
            )
        )

    def test_market_support_rejects_range_edge_stop_entry_fakeout(self) -> None:
        intent = {
            "order_type": "STOP-ENTRY",
            "metadata": {
                "forecast_direction": "UP",
                "forecast_confidence": 0.45,
                "forecast_raw_confidence": 0.63,
                "chart_direction_bias": "LONG",
                "m5_regime": "RANGE",
                "range_phase": "RANGE_STABLE",
                "tf_regime_map": {
                    "M5": {"classification": "RANGE", "range_position": 0.95},
                    "M15": {"classification": "RANGE", "range_position": 0.81},
                },
                "forecast_market_support": {
                    "ok": True,
                    "direction": "UP",
                    "aligned_projection_count": 1,
                    "timing_projection_count": 1,
                    "best_hit_rate": 0.90,
                    "best_samples": 100,
                    "best_aligned_hit_rate": 0.90,
                    "best_aligned_samples": 100,
                    "signals": [
                        {
                            "name": "macro_event_nowcast_central_bank",
                            "direction": "UP",
                            "confidence": 0.79,
                            "hit_rate": 0.90,
                            "samples": 100,
                        }
                    ],
                },
            },
        }

        self.assertFalse(
            _forecast_market_support_allows_low_confidence_live_ready(
                intent,
                side="LONG",
                forecast=self._forecast("UP", confidence=0.45),
                min_confidence=0.65,
            )
        )

    def test_market_support_rejects_out_of_horizon_macro_event_support(self) -> None:
        intent = {
            "order_type": "STOP-ENTRY",
            "metadata": {
                "forecast_direction": "UP",
                "forecast_confidence": 0.46,
                "forecast_raw_confidence": 0.66,
                "forecast_horizon_min": 180,
                "chart_direction_bias": "LONG",
                "m5_regime": "TREND_UP",
                "forecast_market_support": {
                    "ok": True,
                    "direction": "UP",
                    "aligned_projection_count": 1,
                    "best_hit_rate": 0.88,
                    "best_samples": 75,
                    "best_aligned_hit_rate": 0.88,
                    "best_aligned_samples": 75,
                    "signals": [
                        {
                            "name": "macro_event_nowcast_central_bank",
                            "direction": "UP",
                            "confidence": 0.79,
                            "hit_rate": 0.88,
                            "samples": 75,
                            "lead_time_min": 3797.0,
                        }
                    ],
                },
            },
        }

        self.assertFalse(
            _forecast_market_support_allows_low_confidence_live_ready(
                intent,
                side="LONG",
                forecast=self._forecast("UP", confidence=0.46),
                min_confidence=0.65,
            )
        )


def _write_cancel_regret_audit(root: Path, *, pair: str, side: str, order_type: str) -> None:
    now = datetime.now(timezone.utc)
    (root / "execution_timing_audit.json").write_text(
        json.dumps(
            {
                "generated_at_utc": now.isoformat(),
                "status": "OK",
                "canceled_order_regrets": [
                    {
                        "order_id": "prior-cancel-regret",
                        "pair": pair,
                        "side": side,
                        "order_type": order_type,
                        "entry_touched_after_cancel": True,
                        "tp_touched_after_cancel": False,
                        "sl_touched_after_cancel": False,
                        "mfe_pips_after_cancel_entry": 4.0,
                    }
                ],
            }
        )
    )


def _snapshot(*, orders=(), positions=(), quotes=None) -> BrokerSnapshot:
    now = datetime.now(timezone.utc)
    default_quotes = {
        "AUD_JPY": Quote("AUD_JPY", 112.49, 112.50, timestamp_utc=now),
        "EUR_USD": Quote("EUR_USD", 1.1720, 1.1721, timestamp_utc=now),
        "USD_JPY": Quote("USD_JPY", 157.00, 157.01, timestamp_utc=now),
    }
    if quotes:
        default_quotes.update(quotes)
    return BrokerSnapshot(
        fetched_at_utc=now,
        positions=tuple(positions),
        orders=tuple(orders),
        quotes=default_quotes,
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


def _entry_timing_pair_charts(root: Path, *close_dirs: int) -> Path:
    path = root / "pair_charts.json"
    candles = []
    for i, direction in enumerate(close_dirs):
        open_price = 1.1700 + i * 0.0001
        close_price = open_price + (0.0002 if direction > 0 else -0.0002)
        candles.append({"o": open_price, "c": close_price})
    path.write_text(
        json.dumps(
            {
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "views": [
                            {
                                "granularity": "M5",
                                "recent_candles": candles,
                            }
                        ],
                    }
                ]
            }
        )
    )
    return path


def _operating_tf_momentum_pair_charts(root: Path) -> Path:
    path = root / "pair_charts.json"
    path.write_text(
        json.dumps(
            {
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "views": [
                            {
                                "granularity": "M5",
                                "regime": "TREND_UP",
                                "indicators": {"adx_14": 44.0},
                                "recent_candles": [
                                    {"o": 1.1609, "c": 1.1611},
                                    {"o": 1.1611, "c": 1.1615},
                                    {"o": 1.1615, "c": 1.1617},
                                ],
                            },
                            {
                                "granularity": "M15",
                                "regime": "TREND_UP",
                                "indicators": {"adx_14": 32.0},
                            },
                            {
                                "granularity": "M30",
                                "regime": "IMPULSE_UP",
                                "indicators": {"adx_14": 21.0},
                            },
                        ],
                    }
                ]
            }
        )
    )
    return path


def _technical_opposition_pair_charts(root: Path) -> Path:
    path = root / "pair_charts.json"
    path.write_text(
        json.dumps(
            {
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "views": [
                            {
                                "granularity": tf,
                                "regime": "UNCLEAR",
                                "indicators": {
                                    "rsi_14": 72.0,
                                    "macd_hist": 0.0002,
                                    "supertrend_dir": 1,
                                    "ichimoku_cloud_pos": 1,
                                    "plus_di_14": 35.0,
                                    "minus_di_14": 10.0,
                                },
                                "structure": {"last_event": {"kind": "CHOCH_UP", "close_confirmed": True}},
                            }
                            for tf in ("M5", "M15")
                        ],
                    }
                ]
            }
        )
    )
    return path


def _opposite_market_intents(root: Path) -> Path:
    path = root / "opposite_market_intents.json"
    stale_long = _result("trend_trader:EUR_USD:LONG:TREND_CONTINUATION", "EUR_USD", "LONG", "TREND_CONTINUATION")
    stale_long["status"] = "DRY_RUN_PASSED"
    stale_long["risk_metrics"] = {"risk_jpy": 100.0, "reward_jpy": 180.0, "reward_risk": 1.8, "spread_pips": 0.8}
    fresh_short = _result("trend_trader:EUR_USD:SHORT:TREND_CONTINUATION", "EUR_USD", "SHORT", "TREND_CONTINUATION")
    fresh_short["risk_metrics"] = {"risk_jpy": 100.0, "reward_jpy": 300.0, "reward_risk": 3.0, "spread_pips": 0.8}
    fresh_short["intent"] = {
        **fresh_short["intent"],
        "entry": 1.17150,
        "tp": 1.16900,
        "sl": 1.17240,
        "market_context": {
            **fresh_short["intent"]["market_context"],
            "chart_story": "trend-bear continuation",
        },
    }
    path.write_text(json.dumps({"results": [stale_long, fresh_short]}))
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


def _opposite_market_campaign(root: Path) -> Path:
    path = root / "opposite_market_campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    _lane("trend_trader", "EUR_USD", "LONG", "TREND_CONTINUATION"),
                    _lane("trend_trader", "EUR_USD", "SHORT", "TREND_CONTINUATION"),
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


def _opposite_market_strategy(root: Path) -> Path:
    path = root / "opposite_market_strategy.json"
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
                        "pretrade_net_jpy": 0,
                        "live_net_jpy": 0,
                        "live_worst_jpy": -400,
                        "positive_evidence_n": 0,
                        "positive_tail_jpy": 0,
                        "positive_best_jpy": 0,
                        "seat_discovered": 10,
                        "seat_orderable": 8,
                        "seat_captured": 1,
                    },
                    {
                        "pair": "EUR_USD",
                        "direction": "SHORT",
                        "status": "CANDIDATE",
                        "pretrade_net_jpy": 5000,
                        "live_net_jpy": 2500,
                        "live_worst_jpy": -300,
                        "positive_evidence_n": 150,
                        "positive_tail_jpy": 1600,
                        "positive_best_jpy": 2600,
                        "seat_discovered": 10,
                        "seat_orderable": 8,
                        "seat_captured": 6,
                    },
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

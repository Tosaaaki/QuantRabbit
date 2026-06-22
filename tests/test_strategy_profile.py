from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.models import MarketContext, OrderIntent, OrderType, Owner, Side, TradeMethod
from quant_rabbit.strategy.profile import StrategyProfile


class StrategyProfileTest(unittest.TestCase):
    def test_blocks_history_rejected_profile_for_live_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "USD_JPY",
                                "direction": "LONG",
                                "status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                "required_fix": "bad history",
                                "live_net_jpy": -1200.5,
                                "live_n": 4,
                                "pretrade_net_jpy": -300.25,
                                "pretrade_n": 3,
                                "seat_net_jpy": -2200.0,
                                "seat_win_rate_pct": 25.0,
                                "seat_pl_n": 8,
                                "top_block_reasons": ["live tape negative", "pretrade negative"],
                            },
                            {
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "status": "RISK_REPAIR_CANDIDATE",
                                "required_fix": "cap risk",
                            },
                        ]
                    }
                )
            )
            profile = StrategyProfile.load(path)
            prior = _disable_sl_free_for_test()
            try:
                blocked = profile.validate(_intent("USD_JPY"), for_live_send=True)
                repair_dry = profile.validate(_intent("EUR_USD"), for_live_send=False)
                repair_live = profile.validate(_intent("EUR_USD"), for_live_send=True)
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

            self.assertEqual(blocked[0].code, "STRATEGY_NOT_ELIGIBLE")
            self.assertEqual(blocked[0].severity, "BLOCK")
            evidence = profile.issue_evidence(_intent("USD_JPY"))
            self.assertEqual(evidence["profile_status"], "BLOCK_UNTIL_NEW_EVIDENCE")
            self.assertEqual(evidence["required_fix"], "bad history")
            self.assertEqual(evidence["live_net_jpy"], -1200.5)
            self.assertEqual(evidence["pretrade_net_jpy"], -300.25)
            self.assertEqual(evidence["seat_win_rate_pct"], 25.0)
            self.assertEqual(evidence["top_block_reasons"], ["live tape negative", "pretrade negative"])
            self.assertEqual(repair_dry[0].code, "STRATEGY_RISK_REPAIR_REQUIRED")
            self.assertEqual(repair_dry[0].severity, "WARN")
            self.assertEqual(repair_live[0].severity, "BLOCK")

    def test_method_specific_profile_cannot_authorize_another_strategy_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "method": "BREAKOUT_FAILURE",
                                "status": "CANDIDATE",
                                "required_fix": "failed-breakout edge only",
                            }
                        ]
                    }
                )
            )
            profile = StrategyProfile.load(path)

            prior = _disable_sl_free_for_test()
            try:
                breakout = profile.validate(
                    _intent("EUR_USD", method=TradeMethod.BREAKOUT_FAILURE),
                    for_live_send=True,
                )
                trend = profile.validate(
                    _intent("EUR_USD", method=TradeMethod.TREND_CONTINUATION),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

            self.assertEqual(breakout, ())
            self.assertEqual(trend[0].code, "STRATEGY_METHOD_PROFILE_MISSING")
            self.assertEqual(trend[0].severity, "BLOCK")

    def test_mine_missed_edge_blocks_market_live_send_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="MINE_MISSED_EDGE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.MARKET,
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_TRIGGER_RECEIPT_REQUIRED")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_mine_missed_edge_allows_pending_trigger_receipt_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="MINE_MISSED_EDGE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.STOP_ENTRY,
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_TRIGGER_RECEIPT_REQUIRED")
        self.assertEqual(issues[0].severity, "WARN")

    def test_block_until_new_evidence_blocks_live_send_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="BLOCK_UNTIL_NEW_EVIDENCE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.STOP_ENTRY,
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_block_until_new_evidence_range_rail_scout_is_advisory_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                "required_fix": "require a new vehicle or market-structure proof",
                                "positive_evidence_n": 1,
                                "positive_tail_jpy": 295.0,
                                "live_net_jpy": -120.0,
                                "live_n": 3,
                            }
                        ]
                    }
                )
            )
            profile = StrategyProfile.load(path)
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.RANGE_ROTATION,
                        order_type=OrderType.LIMIT,
                        metadata={
                            "forecast_seed": True,
                            "forecast_direction": "RANGE",
                            "forecast_confidence": 0.78,
                            "geometry_model": "RANGE_RAIL_LIMIT",
                            "range_entry_side": "support",
                            "range_tp_is_inside_box": True,
                            "range_sl_outside_box": True,
                            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                            "opportunity_mode": "HARVEST",
                            "chart_direction_bias": "SHORT",
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "WARN")

    def test_block_until_new_evidence_range_rail_without_positive_evidence_still_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_pair_side_profile(Path(tmp), status="BLOCK_UNTIL_NEW_EVIDENCE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.RANGE_ROTATION,
                        order_type=OrderType.LIMIT,
                        metadata={
                            "forecast_seed": True,
                            "forecast_direction": "RANGE",
                            "forecast_confidence": 0.78,
                            "geometry_model": "RANGE_RAIL_LIMIT",
                            "range_entry_side": "support",
                            "range_tp_is_inside_box": True,
                            "range_sl_outside_box": True,
                            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                            "opportunity_mode": "HARVEST",
                            "chart_direction_bias": "SHORT",
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_block_until_new_evidence_oanda_firepower_vehicle_is_advisory_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_pair_side_profile(Path(tmp), status="BLOCK_UNTIL_NEW_EVIDENCE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.RANGE_ROTATION,
                        order_type=OrderType.LIMIT,
                        metadata=_oanda_firepower_metadata(),
                    ),
                    for_live_send=True,
                )
                market_issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.RANGE_ROTATION,
                        order_type=OrderType.MARKET,
                        metadata=_oanda_firepower_metadata(),
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "WARN")
        self.assertEqual(len(market_issues), 1)
        self.assertEqual(market_issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(market_issues[0].severity, "BLOCK")

    def test_block_until_new_evidence_incomplete_oanda_firepower_still_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_pair_side_profile(Path(tmp), status="BLOCK_UNTIL_NEW_EVIDENCE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.RANGE_ROTATION,
                        order_type=OrderType.LIMIT,
                        metadata=_oanda_firepower_metadata(vehicle_count=0),
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_block_until_new_evidence_failed_break_limit_scout_is_advisory_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "pair": "EUR_USD",
                                "direction": "SHORT",
                                "status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                "required_fix": "require a new vehicle or market-structure proof",
                                "positive_evidence_n": 3,
                                "positive_tail_jpy": 401.0,
                                "live_net_jpy": -908.0,
                                "live_n": 4,
                            }
                        ]
                    }
                )
            )
            profile = StrategyProfile.load(path)
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        side=Side.SHORT,
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.LIMIT,
                        metadata={
                            "forecast_seed": True,
                            "forecast_direction": "DOWN",
                            "forecast_confidence": 0.23,
                            "chart_direction_bias": "SHORT",
                            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                            "tp_target_intent": "HARVEST",
                            "opportunity_mode": "HARVEST",
                            "forecast_market_support": {
                                "ok": True,
                                "direction": "DOWN",
                                "aligned_projection_count": 1,
                            },
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "WARN")

    def test_block_until_new_evidence_failed_break_market_still_blocks_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_pair_side_profile(Path(tmp), status="BLOCK_UNTIL_NEW_EVIDENCE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.MARKET,
                        metadata={
                            "forecast_seed": True,
                            "forecast_direction": "UP",
                            "forecast_confidence": 0.72,
                            "chart_direction_bias": "LONG",
                            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                            "tp_target_intent": "HARVEST",
                            "opportunity_mode": "HARVEST",
                            "forecast_market_support": {
                                "ok": True,
                                "direction": "UP",
                                "aligned_projection_count": 1,
                            },
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_watch_only_blocks_live_send_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="WATCH_ONLY"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.STOP_ENTRY,
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_watch_only_forecast_seed_pending_trigger_is_advisory_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="WATCH_ONLY"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.STOP_ENTRY,
                        metadata={"forecast_seed": True},
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "WARN")

    def test_watch_only_forecast_seed_market_still_blocks_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="WATCH_ONLY"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.MARKET,
                        metadata={"forecast_seed": True},
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_watch_only_range_rotation_rail_seed_is_advisory_despite_local_fade_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_pair_side_profile(Path(tmp), status="WATCH_ONLY"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.RANGE_ROTATION,
                        order_type=OrderType.LIMIT,
                        metadata={
                            "forecast_seed": True,
                            "forecast_direction": "RANGE",
                            "forecast_confidence": 0.78,
                            "geometry_model": "RANGE_RAIL_LIMIT",
                            "range_entry_side": "support",
                            "chart_direction_bias": "LONG",
                            "m5_long_bias": 0.125,
                            "m5_short_bias": 0.75,
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "WARN")

    def test_watch_only_upper_rail_short_seed_is_advisory_despite_long_push_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_pair_side_profile(Path(tmp), status="WATCH_ONLY", direction="SHORT"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        side=Side.SHORT,
                        method=TradeMethod.RANGE_ROTATION,
                        order_type=OrderType.LIMIT,
                        metadata={
                            "forecast_seed": True,
                            "forecast_direction": "RANGE",
                            "forecast_confidence": 0.78,
                            "geometry_model": "RANGE_RAIL_LIMIT",
                            "range_entry_side": "resistance",
                            "chart_direction_bias": "LONG",
                            "m5_long_bias": 0.75,
                            "m5_short_bias": 0.125,
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "WARN")

    def test_watch_only_range_rotation_wrong_rail_side_still_blocks_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_pair_side_profile(Path(tmp), status="WATCH_ONLY"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.RANGE_ROTATION,
                        order_type=OrderType.LIMIT,
                        metadata={
                            "forecast_seed": True,
                            "forecast_direction": "RANGE",
                            "forecast_confidence": 0.78,
                            "geometry_model": "RANGE_RAIL_LIMIT",
                            "range_entry_side": "resistance",
                            "chart_direction_bias": "LONG",
                            "m5_long_bias": 0.125,
                            "m5_short_bias": 0.75,
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_missing_profile_blocks_live_send_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.STOP_ENTRY,
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_missing_profile_oanda_firepower_vehicle_is_advisory_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_CHF",
                        side=Side.SHORT,
                        method=TradeMethod.RANGE_ROTATION,
                        order_type=OrderType.LIMIT,
                        metadata=_oanda_firepower_metadata(pair="GBP_CHF", side="SHORT"),
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "WARN")

    def test_high_confidence_forecast_seed_missing_profile_is_advisory_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.TREND_CONTINUATION,
                        order_type=OrderType.STOP_ENTRY,
                        metadata={"forecast_seed": True, "forecast_confidence": 0.72},
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "WARN")

    def test_range_rail_forecast_seed_missing_profile_is_advisory_under_sl_free(self) -> None:
        # Current live shape: a pair with no mined profile can still expose a
        # non-market range-rail LIMIT when the forecast is an auditable RANGE
        # and the order waits on the correct support/resistance side. Other
        # forecast, spread, reward/risk, and gateway gates still decide whether
        # the lane becomes LIVE_READY.
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.RANGE_ROTATION,
                        order_type=OrderType.LIMIT,
                        metadata={
                            "forecast_seed": True,
                            "forecast_direction": "RANGE",
                            "forecast_confidence": 0.52,
                            "geometry_model": "RANGE_RAIL_LIMIT",
                            "range_entry_side": "support",
                            "chart_direction_bias": "LONG",
                            "m5_long_bias": 0.125,
                            "m5_short_bias": 0.75,
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "WARN")

    def test_low_confidence_range_rail_missing_profile_still_blocks_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.RANGE_ROTATION,
                        order_type=OrderType.LIMIT,
                        metadata={
                            "forecast_seed": True,
                            "forecast_direction": "RANGE",
                            "forecast_confidence": 0.49,
                            "geometry_model": "RANGE_RAIL_LIMIT",
                            "range_entry_side": "support",
                            "chart_direction_bias": "LONG",
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_range_rail_missing_profile_uses_same_side_projection_support_under_sl_free(self) -> None:
        # The profile gate mirrors the RiskEngine/intents RANGE rail support
        # exception: a passive LIMIT at the correct rail may collect missing
        # profile evidence when the calibrated RANGE forecast is a near miss and
        # a current audited projection supports the same fade side. This is not
        # a MARKET/chase override and does not rescue weak/no-evidence lanes.
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.RANGE_ROTATION,
                        order_type=OrderType.LIMIT,
                        metadata={
                            "forecast_seed": True,
                            "forecast_direction": "RANGE",
                            "forecast_confidence": 0.49,
                            "forecast_horizon_min": 120,
                            "geometry_model": "RANGE_RAIL_LIMIT",
                            "range_entry_side": "support",
                            "range_tp_is_inside_box": True,
                            "range_sl_outside_box": True,
                            "chart_direction_bias": "LONG",
                            "forecast_market_support": {
                                "ok": False,
                                "direction": "RANGE",
                                "unselected_projection_count": 1,
                                "unselected_signals": [
                                    {
                                        "name": "liquidity_sweep_low",
                                        "direction": "UP",
                                        "confidence": 0.65,
                                        "hit_rate": 1.0,
                                        "samples": 40,
                                        "lead_time_min": 15.0,
                                        "target_pips": 6.0,
                                    }
                                ],
                            },
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "WARN")

    def test_range_rail_missing_profile_blocks_opposite_projection_support_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.RANGE_ROTATION,
                        order_type=OrderType.LIMIT,
                        metadata={
                            "forecast_seed": True,
                            "forecast_direction": "RANGE",
                            "forecast_confidence": 0.49,
                            "forecast_horizon_min": 120,
                            "geometry_model": "RANGE_RAIL_LIMIT",
                            "range_entry_side": "support",
                            "range_tp_is_inside_box": True,
                            "range_sl_outside_box": True,
                            "chart_direction_bias": "LONG",
                            "forecast_market_support": {
                                "ok": False,
                                "direction": "RANGE",
                                "unselected_projection_count": 1,
                                "unselected_signals": [
                                    {
                                        "name": "liquidity_sweep_high",
                                        "direction": "DOWN",
                                        "confidence": 0.9,
                                        "hit_rate": 0.92,
                                        "samples": 41,
                                        "lead_time_min": 15.0,
                                    }
                                ],
                            },
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_high_confidence_forecast_seed_missing_profile_blocks_when_chart_opposes_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.TREND_CONTINUATION,
                        order_type=OrderType.STOP_ENTRY,
                        metadata={
                            "forecast_seed": True,
                            "forecast_confidence": 0.72,
                            "chart_direction_bias": "SHORT",
                            "m5_long_bias": 0.2,
                            "m5_short_bias": 0.8,
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_low_confidence_forecast_seed_missing_profile_blocks_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.TREND_CONTINUATION,
                        order_type=OrderType.STOP_ENTRY,
                        metadata={
                            "forecast_seed": True,
                            "forecast_confidence": 0.5349,
                            "forecast_market_support_ok": True,
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_forecast_seed_missing_profile_market_blocks_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.TREND_CONTINUATION,
                        order_type=OrderType.MARKET,
                        metadata={"forecast_seed": True, "forecast_confidence": 0.91},
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_empty_profile_blocks_forecast_seed_live_send_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strategy.json"
            path.write_text(json.dumps({"profiles": []}))
            profile = StrategyProfile.load(path)
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "EUR_USD",
                        method=TradeMethod.BREAKOUT_FAILURE,
                        order_type=OrderType.STOP_ENTRY,
                        metadata={"forecast_seed": True},
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_EMPTY")
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_reversal_recovery_hedge_missing_profile_is_advisory_under_sl_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(_profile(Path(tmp), status="CANDIDATE"))
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _intent(
                        "GBP_USD",
                        method=TradeMethod.TREND_CONTINUATION,
                        order_type=OrderType.MARKET,
                        metadata={
                            "position_intent": "HEDGE",
                            "hedge_recovery": True,
                            "hedge_timing_class": "REVERSAL",
                        },
                    ),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_PROFILE_MISSING")
        self.assertEqual(issues[0].severity, "WARN")

    def test_tp_proof_collection_downgrades_pair_side_block_to_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(
                _pair_side_profile(Path(tmp), status="BLOCK_UNTIL_NEW_EVIDENCE", direction="SHORT")
            )
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(_tp_proof_intent(), for_live_send=True)
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "STRATEGY_NOT_ELIGIBLE")
        self.assertEqual(issues[0].severity, "WARN")

    def test_tp_proof_collection_works_before_positive_rotation_label_is_added(self) -> None:
        metadata = _tp_proof_metadata()
        metadata.pop("positive_rotation_mode")
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(
                _pair_side_profile(Path(tmp), status="BLOCK_UNTIL_NEW_EVIDENCE", direction="SHORT")
            )
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(_tp_proof_intent(metadata=metadata), for_live_send=True)
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "WARN")

    def test_tp_proof_collection_does_not_downgrade_market_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(
                _pair_side_profile(Path(tmp), status="BLOCK_UNTIL_NEW_EVIDENCE", direction="SHORT")
            )
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(
                    _tp_proof_intent(order_type=OrderType.MARKET),
                    for_live_send=True,
                )
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "BLOCK")

    def test_tp_proof_collection_does_not_downgrade_when_tp_losses_exist(self) -> None:
        metadata = _tp_proof_metadata(capture_take_profit_losses=1)
        with tempfile.TemporaryDirectory() as tmp:
            profile = StrategyProfile.load(
                _pair_side_profile(Path(tmp), status="BLOCK_UNTIL_NEW_EVIDENCE", direction="SHORT")
            )
            prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                issues = profile.validate(_tp_proof_intent(metadata=metadata), for_live_send=True)
            finally:
                _restore_env("QR_TRADER_DISABLE_SL_REPAIR", prior)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "BLOCK")


def _profile(root: Path, *, status: str) -> Path:
    path = root / "strategy.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "method": "BREAKOUT_FAILURE",
                        "status": status,
                        "required_fix": "build trigger/pending-entry receipts before live execution",
                    }
                ]
            }
        )
    )
    return path


def _pair_side_profile(root: Path, *, status: str, direction: str = "LONG") -> Path:
    path = root / "strategy.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": direction,
                        "status": status,
                        "required_fix": "insufficient or mixed evidence; can be observed but not promoted to live execution",
                    }
                ]
            }
        )
    )
    return path


def _oanda_firepower_metadata(
    *,
    pair: str = "EUR_USD",
    side: str = "LONG",
    status: str = "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
    vehicle_count: int = 1,
    estimated_return_pct: float = 2.8,
) -> dict:
    return {
        "oanda_campaign_firepower_seed": True,
        "oanda_campaign_vehicle_key": f"{pair}|{side}|range_reversion|tp1_sl1",
        "oanda_campaign_vehicle_count": vehicle_count,
        "oanda_campaign_vehicle_keys": [f"{pair}|{side}|range_reversion|tp1_sl1"],
        "oanda_campaign_firepower_status": status,
        "oanda_campaign_exit_shape": "tp1_sl1",
        "oanda_campaign_exit_shapes": ["tp1_sl1"],
        "oanda_campaign_estimated_return_pct_per_active_day": estimated_return_pct,
        "oanda_campaign_live_permission": False,
        "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
        "tp_target_intent": "HARVEST",
        "opportunity_mode": "HARVEST",
    }


def _intent(
    pair: str,
    *,
    side: Side = Side.LONG,
    method: TradeMethod = TradeMethod.TREND_CONTINUATION,
    order_type: OrderType = OrderType.MARKET,
    metadata: dict | None = None,
) -> OrderIntent:
    return OrderIntent(
        pair=pair,
        side=side,
        order_type=order_type,
        units=1000,
        entry=1.0,
        tp=1.01,
        sl=0.99,
        thesis="test",
        owner=Owner.TRADER,
        market_context=MarketContext(
            regime=f"{method.value} test",
            narrative="test",
            chart_story="test",
            method=method,
            invalidation="test",
        ),
        metadata=metadata or {},
    )


def _tp_proof_metadata(**overrides: object) -> dict:
    metadata = {
        "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
        "attach_take_profit_on_fill": True,
        "tp_target_intent": "HARVEST",
        "opportunity_mode": "HARVEST",
        "capture_take_profit_scope": "PAIR_SIDE_METHOD",
        "capture_take_profit_trades": 6,
        "capture_take_profit_wins": 6,
        "capture_take_profit_losses": 0,
        "capture_take_profit_expectancy_jpy": 992.7,
        "positive_rotation_mode": "TP_PROOF_COLLECTION_HARVEST",
        "positive_rotation_pessimistic_expectancy_jpy": 189.2,
    }
    metadata.update(overrides)
    return metadata


def _tp_proof_intent(
    *,
    metadata: dict | None = None,
    order_type: OrderType = OrderType.LIMIT,
) -> OrderIntent:
    return _intent(
        "EUR_USD",
        side=Side.SHORT,
        method=TradeMethod.BREAKOUT_FAILURE,
        order_type=order_type,
        metadata=metadata or _tp_proof_metadata(),
    )


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _disable_sl_free_for_test() -> str | None:
    prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
    return prior


if __name__ == "__main__":
    unittest.main()

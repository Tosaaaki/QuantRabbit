from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.models import BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.strategy.position_manager import (
    ACTION_BREAK_EVEN_STOP,
    ACTION_HARVEST_TP,
    ACTION_HOLD_PROTECTED,
    ACTION_HOLD_SL_FREE,
    ACTION_PROFIT_PROTECT,
    ACTION_REPAIR_PROTECTION,
    ACTION_REPAIR_TAKE_PROFIT,
    ACTION_REVIEW_EXIT,
    ACTION_TAKE_PROFIT_MARKET,
    PositionManager,
)


class PositionManagerTest(unittest.TestCase):
    def test_holds_protected_position_when_not_contradicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=160, short_score=120)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="1",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=1000,
                    entry_price=1.1729,
                    unrealized_pl_jpy=-50,
                    take_profit=1.1741,
                    stop_loss=1.1721,
                )
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "pm.json",
                report_path=root / "pm.md",
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
            payload = json.loads((root / "pm.json").read_text())
            self.assertEqual(payload["snapshot_fetched_at_utc"], snapshot.fetched_at_utc.isoformat())
            report = (root / "pm.md").read_text()
            self.assertIn(f"Broker snapshot fetched at UTC: `{snapshot.fetched_at_utc.isoformat()}`", report)
            self.assertIn("remaining risk", report)

    def test_missing_stop_requires_protection_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=160, short_score=120)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="1",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=1000,
                    entry_price=1.1729,
                    unrealized_pl_jpy=10,
                    take_profit=1.1741,
                    stop_loss=None,
                )
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "pm.json",
                report_path=root / "pm.md",
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_REPAIR_PROTECTION)
            self.assertIsNotNone(result.positions[0].recommended_stop_loss)

    def test_profit_requires_break_even_after_session_noise_buffer_clears(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=160, short_score=120)
            pair_charts = _pair_charts(root, atr_pips=1.0)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="1",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=1000,
                    entry_price=1.1729,
                    unrealized_pl_jpy=250,
                    take_profit=1.1741,
                    stop_loss=1.1721,
                ),
                bid=1.1738,
                ask=1.1739,
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=pair_charts,
                output_path=root / "pm.json",
                report_path=root / "pm.md",
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_PROFIT_PROTECT)
            self.assertEqual(result.positions[0].recommended_stop_loss, 1.1729)
            report = (root / "pm.md").read_text()
            self.assertIn("session bucket LONDON", report)
            self.assertIn("SL distance", report)

    def test_profit_protection_waits_inside_current_atr_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=160, short_score=120)
            pair_charts = _pair_charts(root, atr_pips=10.0)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="1",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=1000,
                    entry_price=1.1729,
                    unrealized_pl_jpy=90,
                    take_profit=1.1741,
                    stop_loss=1.1721,
                ),
                bid=1.1738,
                ask=1.1739,
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=pair_charts,
                output_path=root / "pm.json",
                report_path=root / "pm.md",
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
            self.assertIn("session noise", (root / "pm.md").read_text())

    def test_sl_free_profitable_short_waits_until_tp_progress_gate_clears(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=120, short_score=160)
                pair_charts = _pair_charts(root, atr_pips=1.0)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="short-be",
                        pair="EUR_USD",
                        side=Side.SHORT,
                        units=22000,
                        entry_price=1.16077,
                        unrealized_pl_jpy=1900,
                        take_profit=1.15946,
                        stop_loss=None,
                    ),
                    bid=1.16012,
                    ask=1.16020,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].action, ACTION_HOLD_SL_FREE)
                self.assertIsNone(result.positions[0].recommended_stop_loss)
                report = (root / "pm.md").read_text()
                self.assertIn("TP-progress gate", report)
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_sl_free_profitable_short_gets_profit_lock_after_noise_and_tp_progress_clear(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=120, short_score=160)
                pair_charts = _pair_charts(root, atr_pips=1.0)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="short-be",
                        pair="EUR_USD",
                        side=Side.SHORT,
                        units=22000,
                        entry_price=1.16077,
                        unrealized_pl_jpy=3000,
                        take_profit=1.15946,
                        stop_loss=None,
                    ),
                    bid=1.15962,
                    ask=1.15970,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_BREAK_EVEN_STOP)
                self.assertEqual(result.positions[0].action, ACTION_BREAK_EVEN_STOP)
                self.assertEqual(result.positions[0].recommended_stop_loss, 1.1598)
                report = (root / "pm.md").read_text()
                self.assertIn("SL-free profit-lock trigger", report)
                self.assertIn("TP progress gate", report)
                self.assertIn("+9.7pip", report)
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_sl_free_break_even_waits_until_executable_profit_clears_micro_noise(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=120, short_score=160)
                pair_charts = _pair_charts(root, atr_pips=5.0)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="short-wait",
                        pair="EUR_USD",
                        side=Side.SHORT,
                        units=22000,
                        entry_price=1.16077,
                        unrealized_pl_jpy=500,
                        take_profit=1.15946,
                        stop_loss=None,
                    ),
                    bid=1.16042,
                    ask=1.16050,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].action, ACTION_HOLD_SL_FREE)
                self.assertIsNone(result.positions[0].recommended_stop_loss)
                self.assertIn("SL-free profit-lock deferred", (root / "pm.md").read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_sl_free_profit_lock_requires_fresh_volatility_source(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=120, short_score=160)
                stale_at = datetime.now(timezone.utc) - timedelta(minutes=20)
                pair_charts = _pair_charts(root, atr_pips=1.0, generated_at=stale_at)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="short-stale-vol",
                        pair="EUR_USD",
                        side=Side.SHORT,
                        units=22000,
                        entry_price=1.16077,
                        unrealized_pl_jpy=1900,
                        take_profit=1.15946,
                        stop_loss=None,
                    ),
                    bid=1.15962,
                    ask=1.15970,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].action, ACTION_HOLD_SL_FREE)
                self.assertIsNone(result.positions[0].recommended_stop_loss)
                self.assertIn("fresh volatility", (root / "pm.md").read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_sl_free_profit_lock_uses_quick_m1_range_over_slower_atr(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                now = datetime.now(timezone.utc)
                recent_m1 = [
                    {
                        "t": (now - timedelta(seconds=120)).isoformat(),
                        "o": 1.16020,
                        "h": 1.16060,
                        "l": 1.16010,
                        "c": 1.16045,
                        "v": 10,
                    },
                    {
                        "t": (now - timedelta(seconds=60)).isoformat(),
                        "o": 1.16045,
                        "h": 1.16055,
                        "l": 1.16018,
                        "c": 1.16040,
                        "v": 10,
                    },
                ]
                decision = _decision(root, long_score=120, short_score=160)
                pair_charts = _pair_charts(root, atr_pips=1.0, generated_at=now, recent_m1_candles=recent_m1)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="short-quick-vol",
                        pair="EUR_USD",
                        side=Side.SHORT,
                        units=22000,
                        entry_price=1.16077,
                        unrealized_pl_jpy=1900,
                        take_profit=1.15946,
                        stop_loss=None,
                    ),
                    bid=1.16032,
                    ask=1.16040,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].action, ACTION_HOLD_SL_FREE)
                self.assertIsNone(result.positions[0].recommended_stop_loss)
                self.assertIn("quick M1 range", (root / "pm.md").read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_bb_upper_rejection_keeps_short_tp_at_lower_rail_and_adds_be(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=120, short_score=160)
                pair_charts = _bb_rail_pair_charts(root)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="short-bb",
                        pair="EUR_USD",
                        side=Side.SHORT,
                        units=22000,
                        entry_price=1.16077,
                        unrealized_pl_jpy=1900,
                        take_profit=1.15950,
                        stop_loss=None,
                    ),
                    bid=1.15962,
                    ask=1.15970,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_BREAK_EVEN_STOP)
                self.assertEqual(result.positions[0].action, ACTION_BREAK_EVEN_STOP)
                self.assertEqual(result.positions[0].recommended_stop_loss, 1.1598)
                self.assertIsNone(result.positions[0].recommended_take_profit)
                report = (root / "pm.md").read_text()
                self.assertIn("BB rail", report)
                self.assertIn("keep existing TP", report)
                self.assertIn("SL-free profit-lock trigger", report)
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_adaptive_harvest_tp_waits_until_near_target_progress_clears(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _adaptive_harvest_pair_charts(root, atr_pips=1.0, harvest_price=1.16510)
                _write_latest_forecast(root, direction="UNCLEAR", confidence=0.24)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="long-near-tp",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=3600,
                        entry_price=1.16492,
                        unrealized_pl_jpy=100,
                        take_profit=1.16568,
                        stop_loss=None,
                    ),
                    bid=1.16502,
                    ask=1.16504,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].action, ACTION_HOLD_SL_FREE)
                self.assertIsNone(result.positions[0].recommended_take_profit)
                report = (root / "pm.md").read_text()
                self.assertIn("adaptive TP-progress gate", report)
                self.assertIn("existing TP 7.6pip is within", report)
                self.assertIn("forecast + technical MFE-risk", report)
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_adaptive_harvest_tp_keeps_reachable_target_when_forecast_supports_runner(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _adaptive_harvest_pair_charts(root, atr_pips=1.0, harvest_price=1.16550)
                _write_latest_forecast(root, direction="UP", confidence=0.80)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="long-supported-runner",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=3600,
                        entry_price=1.16492,
                        unrealized_pl_jpy=400,
                        take_profit=1.16568,
                        stop_loss=None,
                    ),
                    bid=1.16545,
                    ask=1.16547,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_BREAK_EVEN_STOP)
                self.assertEqual(result.positions[0].action, ACTION_BREAK_EVEN_STOP)
                self.assertIsNone(result.positions[0].recommended_take_profit)
                report = (root / "pm.md").read_text()
                self.assertIn("forecast UP conf=0.80 does not weaken LONG runner", report)
                self.assertIn("reachable TP contraction needs market-read MFE risk", report)
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_adaptive_harvest_tp_requires_forecast_technical_and_progress_for_reachable_target(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _adaptive_harvest_pair_charts(root, atr_pips=1.0, harvest_price=1.16550)
                _write_latest_forecast(root, direction="UNCLEAR", confidence=0.24)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="long-market-read-harvest",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=3600,
                        entry_price=1.16492,
                        unrealized_pl_jpy=400,
                        take_profit=1.16568,
                        stop_loss=None,
                    ),
                    bid=1.16545,
                    ask=1.16547,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HARVEST_TP)
                self.assertEqual(result.positions[0].action, ACTION_HARVEST_TP)
                self.assertEqual(result.positions[0].recommended_take_profit, 1.1655)
                report = (root / "pm.md").read_text()
                self.assertIn("forecast + technical MFE-risk", report)
                self.assertIn("harvest TP", report)
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_adaptive_harvest_tp_allows_stale_wide_target_contraction(self) -> None:
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _adaptive_harvest_pair_charts(root, atr_pips=2.0, harvest_price=1.16510)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="long-stale-wide-tp",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=3600,
                        entry_price=1.16492,
                        unrealized_pl_jpy=100,
                        take_profit=1.16802,
                        stop_loss=None,
                    ),
                    bid=1.16502,
                    ask=1.16504,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HARVEST_TP)
                self.assertEqual(result.positions[0].action, ACTION_HARVEST_TP)
                self.assertEqual(result.positions[0].recommended_take_profit, 1.1651)
                report = (root / "pm.md").read_text()
                self.assertIn("stale-wide", report)
                self.assertIn("harvest TP", report)
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_disaster_stop_protected_winner_still_gets_adaptive_harvest_tp(self) -> None:
        # Disaster SLs bound tail risk but should not disable the SL-free
        # profit-harvest TP path. 2026-06-12 USD_CHF had both TP and a
        # catastrophe stop, so the old branch skipped adaptive HARVEST_TP and
        # waited for a later market-close signal after most MFE was gone.
        prior = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _adaptive_harvest_pair_charts(root, atr_pips=2.9, harvest_price=1.16535)
                _write_latest_forecast(root, direction="UP", confidence=0.36)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="long-disaster-stop-winner",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=3600,
                        entry_price=1.16492,
                        unrealized_pl_jpy=100,
                        take_profit=1.16679,
                        stop_loss=1.15800,
                    ),
                    bid=1.16525,
                    ask=1.16527,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HARVEST_TP)
                self.assertEqual(result.positions[0].action, ACTION_HARVEST_TP)
                self.assertEqual(result.positions[0].recommended_take_profit, 1.16535)
                report = (root / "pm.md").read_text()
                self.assertIn("under market-read MFE risk", report)
                self.assertIn("harvest TP", report)
        finally:
            if prior is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior

    def test_usd_quote_position_risk_uses_snapshot_conversion_not_static_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=160, short_score=120)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="1",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=1000,
                    entry_price=1.2000,
                    unrealized_pl_jpy=0,
                    take_profit=1.2020,
                    stop_loss=1.1990,
                ),
                usd_jpy_bid=199.99,
                usd_jpy_ask=200.0,
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "pm.json",
                report_path=root / "pm.md",
            ).run(snapshot)

            self.assertEqual(result.positions[0].remaining_risk_jpy, 200.0)
            self.assertEqual(result.positions[0].remaining_reward_jpy, 400.0)

    def test_usd_quote_position_risk_does_not_use_static_conversion_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=160, short_score=120)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="1",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=1000,
                    entry_price=1.2000,
                    unrealized_pl_jpy=0,
                    take_profit=1.2020,
                    stop_loss=1.1990,
                ),
                include_usd_jpy=False,
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "pm.json",
                report_path=root / "pm.md",
            ).run(snapshot)

            self.assertIsNone(result.positions[0].remaining_risk_jpy)
            self.assertIn("cannot be converted", (root / "pm.md").read_text())

    def test_missing_stop_without_conversion_routes_to_exit_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=160, short_score=120)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="1",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=1000,
                    entry_price=1.2000,
                    unrealized_pl_jpy=-20,
                    take_profit=1.2020,
                    stop_loss=None,
                ),
                include_usd_jpy=False,
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "pm.json",
                report_path=root / "pm.md",
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_REVIEW_EXIT)
            self.assertIsNone(result.positions[0].recommended_stop_loss)
            self.assertIn("needs exit review", (root / "pm.md").read_text())

    def test_auto_close_kill_switch_demotes_exit_review_without_crashing(self) -> None:
        prior = os.environ.get("QR_DISABLE_AUTO_CLOSE")
        os.environ["QR_DISABLE_AUTO_CLOSE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=160, short_score=120)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="1",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.2000,
                        unrealized_pl_jpy=-20,
                        take_profit=1.2020,
                        stop_loss=None,
                    ),
                    include_usd_jpy=False,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=root / "missing_pair_charts.json",
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].close_review_action, ACTION_REVIEW_EXIT)
                payload = json.loads((root / "pm.json").read_text())
                self.assertEqual(payload["positions"][0]["action"], ACTION_HOLD_PROTECTED)
                self.assertEqual(payload["positions"][0]["close_review_action"], ACTION_REVIEW_EXIT)
                self.assertIn("QR_DISABLE_AUTO_CLOSE=1", (root / "pm.md").read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_DISABLE_AUTO_CLOSE", None)
            else:
                os.environ["QR_DISABLE_AUTO_CLOSE"] = prior

    def test_auto_close_kill_switch_holds_legacy_structural_exit(self) -> None:
        prior_close = os.environ.get("QR_DISABLE_AUTO_CLOSE")
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_DISABLE_AUTO_CLOSE"] = "1"
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _structural_reversal_pair_charts(root)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="legacy-1",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.2000,
                        unrealized_pl_jpy=-250,
                        take_profit=1.2020,
                        stop_loss=None,
                    ),
                    bid=1.1980,
                    ask=1.1981,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "data" / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].action, ACTION_HOLD_PROTECTED)
                report = (root / "pm.md").read_text()
                self.assertIn("loss-cut: macro REVERSED", report)
                self.assertIn("QR_ALLOW_STRUCTURAL_AUTO_CLOSE=1", report)
        finally:
            if prior_close is None:
                os.environ.pop("QR_DISABLE_AUTO_CLOSE", None)
            else:
                os.environ["QR_DISABLE_AUTO_CLOSE"] = prior_close
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

    def test_auto_close_kill_switch_holds_next_generation_structural_exit_by_default(self) -> None:
        prior_close = os.environ.get("QR_DISABLE_AUTO_CLOSE")
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_struct = os.environ.get("QR_ALLOW_STRUCTURAL_AUTO_CLOSE")
        os.environ["QR_DISABLE_AUTO_CLOSE"] = "1"
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_ALLOW_STRUCTURAL_AUTO_CLOSE", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / "data"
                data_root.mkdir(parents=True)
                (data_root / "entry_thesis_ledger.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_utc": "2026-05-15T06:00:00Z",
                            "trade_id": "future-1",
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "entry_price": 1.2000,
                            "forecast_direction": "LONG",
                            "forecast_confidence": 0.72,
                            "regime": "TREND_UP",
                            "invalidation_price": 1.1984,
                            "target_price": 1.2020,
                            "key_drivers": ["test"],
                        }
                    )
                    + "\n"
                )
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _structural_reversal_pair_charts(root)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="future-1",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.2000,
                        unrealized_pl_jpy=-250,
                        take_profit=1.2020,
                        stop_loss=None,
                    ),
                    bid=1.1980,
                    ask=1.1981,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=data_root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].action, ACTION_HOLD_PROTECTED)
                report = (root / "pm.md").read_text()
                self.assertIn("loss-cut: macro REVERSED", report)
                self.assertIn("QR_ALLOW_STRUCTURAL_AUTO_CLOSE=1", report)
        finally:
            if prior_close is None:
                os.environ.pop("QR_DISABLE_AUTO_CLOSE", None)
            else:
                os.environ["QR_DISABLE_AUTO_CLOSE"] = prior_close
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl
            if prior_struct is None:
                os.environ.pop("QR_ALLOW_STRUCTURAL_AUTO_CLOSE", None)
            else:
                os.environ["QR_ALLOW_STRUCTURAL_AUTO_CLOSE"] = prior_struct

    def test_auto_close_kill_switch_allows_next_generation_structural_exit_with_explicit_opt_in(self) -> None:
        prior_close = os.environ.get("QR_DISABLE_AUTO_CLOSE")
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_struct = os.environ.get("QR_ALLOW_STRUCTURAL_AUTO_CLOSE")
        os.environ["QR_DISABLE_AUTO_CLOSE"] = "1"
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_ALLOW_STRUCTURAL_AUTO_CLOSE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / "data"
                data_root.mkdir(parents=True)
                (data_root / "entry_thesis_ledger.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_utc": "2026-05-15T06:00:00Z",
                            "trade_id": "future-1",
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "entry_price": 1.2000,
                            "forecast_direction": "LONG",
                            "forecast_confidence": 0.72,
                            "regime": "TREND_UP",
                            "invalidation_price": 1.1984,
                            "target_price": 1.2020,
                            "key_drivers": ["test"],
                        }
                    )
                    + "\n"
                )
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _structural_reversal_pair_charts(root)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="future-1",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.2000,
                        unrealized_pl_jpy=-250,
                        take_profit=1.2020,
                        stop_loss=None,
                    ),
                    bid=1.1980,
                    ask=1.1981,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=data_root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_REVIEW_EXIT)
                self.assertEqual(result.positions[0].action, ACTION_REVIEW_EXIT)
                report = (root / "pm.md").read_text()
                self.assertIn("loss-cut: macro REVERSED", report)
                self.assertIn("next-generation entry thesis ledger present", report)
        finally:
            if prior_close is None:
                os.environ.pop("QR_DISABLE_AUTO_CLOSE", None)
            else:
                os.environ["QR_DISABLE_AUTO_CLOSE"] = prior_close
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl
            if prior_struct is None:
                os.environ.pop("QR_ALLOW_STRUCTURAL_AUTO_CLOSE", None)
            else:
                os.environ["QR_ALLOW_STRUCTURAL_AUTO_CLOSE"] = prior_struct

    def test_entry_thesis_invalidation_hit_routes_sl_free_loss_to_review_exit(self) -> None:
        prior_close = os.environ.get("QR_DISABLE_AUTO_CLOSE")
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_struct = os.environ.get("QR_ALLOW_STRUCTURAL_AUTO_CLOSE")
        os.environ["QR_DISABLE_AUTO_CLOSE"] = "1"
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_ALLOW_STRUCTURAL_AUTO_CLOSE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / "data"
                data_root.mkdir(parents=True)
                (data_root / "entry_thesis_ledger.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_utc": "2026-06-05T11:19:50Z",
                            "trade_id": "472071",
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "entry_price": 1.34702,
                            "forecast_direction": "UP",
                            "forecast_confidence": 0.58,
                            "regime": "RANGE",
                            "invalidation_price": 1.34679,
                            "target_price": 1.34853,
                            "key_drivers": ["failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE"],
                        }
                    )
                    + "\n"
                )
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _entry_invalidation_technical_pair_charts(root)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="472071",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=6000,
                        entry_price=1.34702,
                        unrealized_pl_jpy=-2981.9,
                        take_profit=1.34853,
                        stop_loss=None,
                    ),
                    bid=1.34392,
                    ask=1.34405,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=data_root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].action, ACTION_HOLD_PROTECTED)
                report = (root / "pm.md").read_text()
                self.assertIn("loss-cut: entry thesis invalidation hit", report)
                self.assertIn("technical invalidation confirmed against LONG", report)
                self.assertIn("soft entry-thesis / forecast-collapse evidence must go through gpt_trader Gate A/B", report)
        finally:
            if prior_close is None:
                os.environ.pop("QR_DISABLE_AUTO_CLOSE", None)
            else:
                os.environ["QR_DISABLE_AUTO_CLOSE"] = prior_close
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl
            if prior_struct is None:
                os.environ.pop("QR_ALLOW_STRUCTURAL_AUTO_CLOSE", None)
            else:
                os.environ["QR_ALLOW_STRUCTURAL_AUTO_CLOSE"] = prior_struct

    def test_entry_thesis_confidence_collapse_routes_loss_to_review_exit(self) -> None:
        prior_close = os.environ.get("QR_DISABLE_AUTO_CLOSE")
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_struct = os.environ.get("QR_ALLOW_STRUCTURAL_AUTO_CLOSE")
        os.environ["QR_DISABLE_AUTO_CLOSE"] = "1"
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_ALLOW_STRUCTURAL_AUTO_CLOSE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / "data"
                data_root.mkdir(parents=True)
                (data_root / "entry_thesis_ledger.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_utc": "2026-06-05T11:19:50Z",
                            "trade_id": "forecast-collapse",
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "entry_price": 1.34702,
                            "forecast_direction": "UP",
                            "forecast_confidence": 0.70,
                            "regime": "TREND_UP",
                            "invalidation_price": 1.34000,
                            "target_price": 1.34853,
                            "key_drivers": ["forecast=UP@conf=0.70"],
                        }
                    )
                    + "\n"
                )
                (data_root / "forecast_history.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_utc": "2026-06-05T11:30:00Z",
                            "cycle_id": "position-forecast-refresh:test",
                            "pair": "EUR_USD",
                            "direction": "UP",
                            "confidence": 0.20,
                            "current_price": 1.3460,
                            "invalidation_price": 1.3452,
                            "target_price": 1.3470,
                            "horizon_min": 60,
                        }
                    )
                    + "\n"
                )
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _entry_invalidation_technical_pair_charts(root)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="forecast-collapse",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=6000,
                        entry_price=1.34702,
                        unrealized_pl_jpy=-620.0,
                        take_profit=1.34853,
                        stop_loss=None,
                    ),
                    bid=1.34600,
                    ask=1.34613,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=data_root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].action, ACTION_HOLD_PROTECTED)
                report = (root / "pm.md").read_text()
                self.assertIn("loss-cut: entry thesis confidence collapse", report)
                self.assertIn("technical invalidation confirmed against LONG", report)
                self.assertIn("soft entry-thesis / forecast-collapse evidence must go through gpt_trader Gate A/B", report)
        finally:
            if prior_close is None:
                os.environ.pop("QR_DISABLE_AUTO_CLOSE", None)
            else:
                os.environ["QR_DISABLE_AUTO_CLOSE"] = prior_close
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl
            if prior_struct is None:
                os.environ.pop("QR_ALLOW_STRUCTURAL_AUTO_CLOSE", None)
            else:
                os.environ["QR_ALLOW_STRUCTURAL_AUTO_CLOSE"] = prior_struct

    def test_entry_thesis_data_root_is_independent_from_output_path(self) -> None:
        prior_close = os.environ.get("QR_DISABLE_AUTO_CLOSE")
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_struct = os.environ.get("QR_ALLOW_STRUCTURAL_AUTO_CLOSE")
        os.environ["QR_DISABLE_AUTO_CLOSE"] = "1"
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_ALLOW_STRUCTURAL_AUTO_CLOSE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / "data"
                output_root = root / "diagnostics"
                data_root.mkdir(parents=True)
                output_root.mkdir(parents=True)
                (data_root / "entry_thesis_ledger.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_utc": "2026-06-05T11:19:50Z",
                            "trade_id": "separate-root",
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "entry_price": 1.34702,
                            "forecast_direction": "UP",
                            "forecast_confidence": 0.70,
                            "regime": "TREND_UP",
                            "invalidation_price": 1.34500,
                            "target_price": 1.34853,
                            "key_drivers": ["forecast=UP@conf=0.70"],
                        }
                    )
                    + "\n"
                )
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _entry_invalidation_technical_pair_charts(root)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="separate-root",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=6000,
                        entry_price=1.34702,
                        unrealized_pl_jpy=-620.0,
                        take_profit=1.34853,
                        stop_loss=None,
                    ),
                    bid=1.34470,
                    ask=1.34483,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=output_root / "pm.json",
                    report_path=output_root / "pm.md",
                    data_root=data_root,
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].action, ACTION_HOLD_PROTECTED)
                report = (output_root / "pm.md").read_text()
                self.assertIn("entry thesis invalidation hit", report)
                self.assertIn("soft entry-thesis / forecast-collapse evidence must go through gpt_trader Gate A/B", report)
        finally:
            if prior_close is None:
                os.environ.pop("QR_DISABLE_AUTO_CLOSE", None)
            else:
                os.environ["QR_DISABLE_AUTO_CLOSE"] = prior_close
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl
            if prior_struct is None:
                os.environ.pop("QR_ALLOW_STRUCTURAL_AUTO_CLOSE", None)
            else:
                os.environ["QR_ALLOW_STRUCTURAL_AUTO_CLOSE"] = prior_struct

    def test_sl_free_profitable_macro_reversal_uses_profit_market_take_under_auto_close_kill_switch(self) -> None:
        prior_close = os.environ.get("QR_DISABLE_AUTO_CLOSE")
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_DISABLE_AUTO_CLOSE"] = "1"
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _structural_reversal_pair_charts(root)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="profit-reversal",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.2000,
                        unrealized_pl_jpy=250,
                        take_profit=1.2020,
                        stop_loss=None,
                    ),
                    bid=1.2013,
                    ask=1.2014,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "data" / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_TAKE_PROFIT_MARKET)
                self.assertEqual(result.positions[0].action, ACTION_TAKE_PROFIT_MARKET)
                report = (root / "pm.md").read_text()
                self.assertIn("profit-harvest market close", report)
                self.assertNotIn("legacy/no-ledger close", report)
        finally:
            if prior_close is None:
                os.environ.pop("QR_DISABLE_AUTO_CLOSE", None)
            else:
                os.environ["QR_DISABLE_AUTO_CLOSE"] = prior_close
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

    def test_profitable_macro_reversal_waits_when_executable_profit_is_inside_noise(self) -> None:
        # 2026-06-18 AUD_NZD repair: the matrix path recorded
        # "profit < market noise floor" through the local-top signal, then
        # still closed by the broad REVERSED/DEAD profit-harvest branch for a
        # tiny win. That keeps average wins too small relative to loss closes.
        prior_close = os.environ.get("QR_DISABLE_AUTO_CLOSE")
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_DISABLE_AUTO_CLOSE"] = "1"
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _structural_reversal_pair_charts(root)
                snapshot = _snapshot(
                    BrokerPosition(
                        trade_id="thin-profit-reversal",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.2000,
                        unrealized_pl_jpy=30,
                        take_profit=1.2020,
                        stop_loss=None,
                    ),
                    bid=1.20008,
                    ask=1.20028,
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "data" / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].action, ACTION_HOLD_SL_FREE)
                report = (root / "pm.md").read_text()
                self.assertIn("profit-harvest market close skipped", report)
                self.assertIn("market noise floor", report)
        finally:
            if prior_close is None:
                os.environ.pop("QR_DISABLE_AUTO_CLOSE", None)
            else:
                os.environ["QR_DISABLE_AUTO_CLOSE"] = prior_close
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

    def test_protected_profitable_long_temporary_top_uses_profit_market_take(self) -> None:
        # Regression from 2026-06-12 USD_CAD screenshot: the long swing thesis
        # can still be broadly valid, but a local M1 top followed by lower
        # closes should bank the current MFE and wait for a pullback/retest
        # re-entry instead of HOLD_PROTECTED until the full TP.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=180, short_score=80)
            pair_charts = _temporary_top_pair_charts(root)
            _write_latest_forecast(root, direction="UP", confidence=0.39)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="temporary-top",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=5000,
                    entry_price=1.19980,
                    unrealized_pl_jpy=450,
                    take_profit=1.20075,
                    stop_loss=1.19800,
                ),
                bid=1.20040,
                ask=1.20060,
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=pair_charts,
                output_path=root / "data" / "pm.json",
                report_path=root / "pm.md",
                data_root=root,
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_TAKE_PROFIT_MARKET)
            self.assertEqual(result.positions[0].action, ACTION_TAKE_PROFIT_MARKET)
            report = (root / "pm.md").read_text()
            self.assertIn("temporary top profit-take", report)
            self.assertIn("fresh LIVE_READY pullback/retest lane", report)

    def test_temporary_top_waits_when_attached_tp_progress_is_too_shallow(self) -> None:
        # 2026-06-18 GBP_CHF regression: local-top evidence must not turn an
        # attached-TP runner into a micro-scalp when only a small fraction of
        # the planned TP has printed.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=180, short_score=80)
            pair_charts = _temporary_top_pair_charts(root)
            _write_latest_forecast(root, direction="UP", confidence=0.39)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="temporary-top-shallow-progress",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=5000,
                    entry_price=1.19980,
                    unrealized_pl_jpy=450,
                    take_profit=1.20320,
                    stop_loss=1.19800,
                ),
                bid=1.20040,
                ask=1.20060,
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=pair_charts,
                output_path=root / "data" / "pm.json",
                report_path=root / "pm.md",
                data_root=root,
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
            self.assertEqual(result.positions[0].action, ACTION_HOLD_PROTECTED)
            report = (root / "pm.md").read_text()
            self.assertIn("temporary top profit-take", report)
            self.assertIn("TP progress", report)
            self.assertIn("keep broker TP", report)

    def test_profitable_long_local_swing_top_does_not_wait_for_full_spread_pullback(self) -> None:
        # The live 2026-06-12 USD_CAD sidecar saw the local top logic but
        # skipped because pullback was still smaller than the spread/ATR floor.
        # A confirmed M1 rollover stack should bank profit while still close to
        # the top; waiting for a full floor gives back the very MFE the operator
        # asked the trader to capture.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=180, short_score=80)
            pair_charts = _local_swing_top_without_rail_pair_charts(root)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="local-swing-top",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=5000,
                    entry_price=1.19980,
                    unrealized_pl_jpy=650,
                    take_profit=1.20110,
                    stop_loss=1.19800,
                ),
                bid=1.20064,
                ask=1.20082,
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=pair_charts,
                output_path=root / "data" / "pm.json",
                report_path=root / "pm.md",
                data_root=root,
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_TAKE_PROFIT_MARKET)
            self.assertEqual(result.positions[0].action, ACTION_TAKE_PROFIT_MARKET)
            report = (root / "pm.md").read_text()
            self.assertIn("M1 local swing top", report)
            self.assertIn("close-confirmed M1 rollover stack is already complete", report)
            self.assertIn("temporary top profit-take", report)

    def test_profitable_long_three_signal_local_top_takes_profit_before_full_micro_flip(self) -> None:
        # 2026-06-12 USD_CHF: by the time M1/M5 fully flip against the runner,
        # the local top is already gone. With local-swing context present,
        # rollover + weak runner forecast + latest close away from the extreme
        # is enough to bank the temporary MFE and wait for a fresh re-entry.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=180, short_score=80)
            pair_charts = _three_signal_local_top_pair_charts(root)
            _write_latest_forecast(root, direction="UP", confidence=0.45)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="three-signal-top",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=5000,
                    entry_price=1.19980,
                    unrealized_pl_jpy=650,
                    take_profit=1.20110,
                    stop_loss=1.19800,
                ),
                bid=1.20064,
                ask=1.20082,
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=pair_charts,
                output_path=root / "data" / "pm.json",
                report_path=root / "pm.md",
                data_root=root,
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_TAKE_PROFIT_MARKET)
            report = (root / "pm.md").read_text()
            self.assertIn("temporary top profit-take", report)
            self.assertIn("M1 local swing top", report)
            self.assertIn("forecast UP confidence 0.45 below runner threshold", report)
            self.assertIn("latest M1 close moved away from the local extreme", report)

    def test_profitable_long_delayed_m1_top_uses_full_context_window_for_harvest(self) -> None:
        # Live 2026-06-12 USD_CHF had six current rollover signals, but the top
        # candle had slipped just outside the short evidence window. The
        # extreme context should use the full bounded M1 packet while reversal
        # evidence remains current.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=180, short_score=80)
            pair_charts = _delayed_local_swing_top_pair_charts(root)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="delayed-swing-top",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=5000,
                    entry_price=1.19980,
                    unrealized_pl_jpy=650,
                    take_profit=1.20105,
                    stop_loss=1.19800,
                ),
                bid=1.20058,
                ask=1.20076,
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=pair_charts,
                output_path=root / "data" / "pm.json",
                report_path=root / "pm.md",
                data_root=root,
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_TAKE_PROFIT_MARKET)
            report = (root / "pm.md").read_text()
            self.assertIn("temporary top profit-take", report)
            self.assertIn("M1 local swing top", report)
            self.assertIn("M1 rollover", report)

    def test_profitable_long_mfe_giveback_takes_profit_before_red(self) -> None:
        # Regression from the execution-timing audit: many losing market closes
        # were positive earlier but had no clean rail/distribution context. If
        # recent executable MFE has already given back most of its move and the
        # position is still profitable, bank it instead of waiting for red.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=180, short_score=80)
            pair_charts = _mfe_giveback_pair_charts(root)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="mfe-giveback",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=5000,
                    entry_price=1.19900,
                    unrealized_pl_jpy=320,
                    take_profit=1.19995,
                    stop_loss=1.19800,
                ),
                bid=1.19985,
                ask=1.20000,
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=pair_charts,
                output_path=root / "data" / "pm.json",
                report_path=root / "pm.md",
                data_root=root,
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_TAKE_PROFIT_MARKET)
            self.assertEqual(result.positions[0].action, ACTION_TAKE_PROFIT_MARKET)
            report = (root / "pm.md").read_text()
            self.assertIn("MFE giveback profit-take", report)
            self.assertIn("post-close re-entry discipline", report)

    def test_mfe_giveback_waits_when_attached_tp_progress_is_too_shallow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=180, short_score=80)
            pair_charts = _mfe_giveback_pair_charts(root)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="mfe-giveback-shallow-progress",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=5000,
                    entry_price=1.19900,
                    unrealized_pl_jpy=320,
                    take_profit=1.20320,
                    stop_loss=1.19800,
                ),
                bid=1.19985,
                ask=1.20000,
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=pair_charts,
                output_path=root / "data" / "pm.json",
                report_path=root / "pm.md",
                data_root=root,
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
            self.assertEqual(result.positions[0].action, ACTION_HOLD_PROTECTED)
            report = (root / "pm.md").read_text()
            self.assertIn("MFE giveback profit-take", report)
            self.assertIn("TP progress", report)
            self.assertIn("keep broker TP", report)

    def test_profitable_long_half_mfe_giveback_takes_profit_before_red(self) -> None:
        # The audit lag showed that waiting for a deeper giveback lets the next
        # cycle see the trade only after it has already gone red. Half-giveback
        # plus reversal evidence is the earliest profit-only harvest point.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=180, short_score=80)
            pair_charts = _mfe_giveback_pair_charts(root)
            snapshot = _snapshot(
                BrokerPosition(
                    trade_id="half-mfe-giveback",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=5000,
                    entry_price=1.19900,
                    unrealized_pl_jpy=420,
                    take_profit=1.20020,
                    stop_loss=1.19800,
                ),
                bid=1.19985,
                ask=1.20000,
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=pair_charts,
                output_path=root / "data" / "pm.json",
                report_path=root / "pm.md",
                data_root=root,
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_TAKE_PROFIT_MARKET)
            self.assertEqual(result.positions[0].action, ACTION_TAKE_PROFIT_MARKET)
            self.assertIn(">= 0.50× MFE", (root / "pm.md").read_text())

    def test_operator_manual_position_without_tp_preserves_no_broker_tp_by_default(self) -> None:
        prior = os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _pair_charts(root, atr_pips=10.0)
                snapshot = BrokerSnapshot(
                    fetched_at_utc=datetime.now(timezone.utc),
                    positions=(
                        BrokerPosition(
                            trade_id="manual-1",
                            pair="EUR_USD",
                            side=Side.LONG,
                            units=25000,
                            entry_price=1.1729,
                            owner=Owner.UNKNOWN,
                        ),
                    ),
                    quotes={"EUR_USD": Quote("EUR_USD", 1.1728, 1.1729, timestamp_utc=datetime.now(timezone.utc))},
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
                self.assertEqual(result.positions[0].trade_id, "manual-1")
                self.assertIsNone(result.positions[0].recommended_stop_loss)
                self.assertIsNone(result.positions[0].recommended_take_profit)
                report = (root / "pm.md").read_text()
                self.assertIn("TP-only profit management enabled", report)
                self.assertIn("preserving no-broker-TP runner", report)
        finally:
            if prior is not None:
                os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = prior

    def test_operator_manual_position_gets_take_profit_when_missing_repair_enabled(self) -> None:
        prior = os.environ.get("QR_ENABLE_MISSING_TP_REPAIR")
        os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                decision = _decision(root, long_score=160, short_score=120)
                pair_charts = _pair_charts(root, atr_pips=10.0)
                snapshot = BrokerSnapshot(
                    fetched_at_utc=datetime.now(timezone.utc),
                    positions=(
                        BrokerPosition(
                            trade_id="manual-1",
                            pair="EUR_USD",
                            side=Side.LONG,
                            units=25000,
                            entry_price=1.1729,
                            owner=Owner.UNKNOWN,
                        ),
                    ),
                    quotes={"EUR_USD": Quote("EUR_USD", 1.1728, 1.1729, timestamp_utc=datetime.now(timezone.utc))},
                )

                result = PositionManager(
                    trader_decision_path=decision,
                    pair_charts_path=pair_charts,
                    output_path=root / "pm.json",
                    report_path=root / "pm.md",
                ).run(snapshot)

                self.assertEqual(result.action, ACTION_REPAIR_TAKE_PROFIT)
                self.assertEqual(result.positions[0].trade_id, "manual-1")
                self.assertIsNone(result.positions[0].recommended_stop_loss)
                self.assertIsNotNone(result.positions[0].recommended_take_profit)
                report = (root / "pm.md").read_text()
                self.assertIn("TP-only profit management enabled", report)
                self.assertIn("SL and loss-close management disabled", report)
        finally:
            if prior is None:
                os.environ.pop("QR_ENABLE_MISSING_TP_REPAIR", None)
            else:
                os.environ["QR_ENABLE_MISSING_TP_REPAIR"] = prior

    def test_operator_manual_position_with_existing_tp_keeps_stop_loss_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=160, short_score=120)
            snapshot = BrokerSnapshot(
                fetched_at_utc=datetime.now(timezone.utc),
                positions=(
                    BrokerPosition(
                        trade_id="manual-2",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=25000,
                        entry_price=1.1729,
                        take_profit=1.1800,
                        stop_loss=None,
                        owner=Owner.MANUAL,
                    ),
                ),
                quotes={"EUR_USD": Quote("EUR_USD", 1.1738, 1.1739, timestamp_utc=datetime.now(timezone.utc))},
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "pm.json",
                report_path=root / "pm.md",
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_HOLD_PROTECTED)
            self.assertIsNone(result.positions[0].recommended_stop_loss)
            self.assertIsNone(result.positions[0].recommended_take_profit)
            self.assertIn("stop-loss untouched", (root / "pm.md").read_text())


def _decision(root: Path, *, long_score: float, short_score: float) -> Path:
    path = root / "decision.json"
    path.write_text(
        json.dumps(
            {
                "scores": [
                    {"pair": "EUR_USD", "direction": "LONG", "score": long_score},
                    {"pair": "EUR_USD", "direction": "SHORT", "score": short_score},
                ]
            }
        )
    )
    return path


def _write_latest_forecast(
    root: Path,
    *,
    pair: str = "EUR_USD",
    direction: str,
    confidence: float,
    horizon_min: float = 0.0,
) -> Path:
    path = root / "forecast_history.jsonl"
    path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "cycle_id": "test-cycle",
                "pair": pair,
                "direction": direction,
                "confidence": confidence,
                "horizon_min": horizon_min,
            }
        )
        + "\n"
    )
    return path


def _snapshot(
    position: BrokerPosition,
    *,
    bid: float = 1.1728,
    ask: float = 1.1729,
    usd_jpy_bid: float = 156.99,
    usd_jpy_ask: float = 157.0,
    include_usd_jpy: bool = True,
) -> BrokerSnapshot:
    now = datetime.now(timezone.utc)
    if position.owner == Owner.UNKNOWN:
        position = replace(position, owner=Owner.TRADER)
    quotes = {"EUR_USD": Quote("EUR_USD", bid, ask, timestamp_utc=now)}
    if include_usd_jpy:
        quotes["USD_JPY"] = Quote("USD_JPY", usd_jpy_bid, usd_jpy_ask, timestamp_utc=now)
    return BrokerSnapshot(
        fetched_at_utc=now,
        positions=(position,),
        quotes=quotes,
    )


def _pair_charts(
    root: Path,
    *,
    atr_pips: float,
    generated_at: datetime | None = None,
    recent_m1_candles: list[dict[str, object]] | None = None,
) -> Path:
    path = root / "pair_charts.json"
    now = generated_at or datetime.now(timezone.utc)
    views = []
    if recent_m1_candles is not None:
        views.append(
            {
                "granularity": "M1",
                "indicators": {"atr_pips": atr_pips},
                "recent_candles": recent_m1_candles,
            }
        )
    views.append(
        {
            "granularity": "M5",
            "indicators": {"atr_pips": atr_pips},
        }
    )
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": now.isoformat(),
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "session": {"current_tag": "LONDON_KILLZONE"},
                        "views": views,
                    }
                ]
            }
        )
    )
    return path


def _adaptive_harvest_pair_charts(root: Path, *, atr_pips: float, harvest_price: float) -> Path:
    path = root / "pair_charts_adaptive_harvest.json"
    now = datetime.now(timezone.utc)
    chart_story = (
        "M1(TREND_UP,ADX=19,ST=+,struct=BOS_UP@1.1648) "
        "M5(TREND_UP,ADX=24,ST=+,struct=BOS_UP@1.1649) "
        "H1(TREND_UP,ADX=31,ST=+) "
        "H4(RANGE,ADX=12,ST=+) "
        "D(RANGE,ADX=10,ST=+)"
    )
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": now.isoformat(),
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "chart_story": chart_story,
                        "confluence": {
                            "price_percentile_24h": 0.98,
                            "price_percentile_7d": 0.98,
                            "tf_agreement_score": 0.33,
                            "range_24h_sigma_multiple": 3.1,
                        },
                        "session": {"current_tag": "LONDON_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M5",
                                "regime": "TREND_UP",
                                "indicators": {"atr_pips": atr_pips},
                            },
                            {
                                "granularity": "M15",
                                "regime": "TREND_UP",
                                "indicators": {
                                    "atr_pips": atr_pips * 2.0,
                                    "rsi_14": 74.0,
                                    "stoch_rsi": 0.91,
                                    "williams_r_14": -12.0,
                                    "close": harvest_price,
                                    "bb_upper": harvest_price - 0.00001,
                                    "donchian_high": harvest_price - 0.00001,
                                },
                                "structure": {
                                    "liquidity": [
                                        {
                                            "side": "EQ_HIGH",
                                            "price": harvest_price,
                                            "indices": [1, 2, 3, 4],
                                        }
                                    ]
                                },
                            },
                        ],
                    }
                ],
            }
        )
    )
    return path


def _bb_rail_pair_charts(root: Path) -> Path:
    path = root / "pair_charts_bb_rail.json"
    now = datetime.now(timezone.utc)
    chart_story = (
        "M1(UNCLEAR,ADX=14,ST=+,struct=CHOCH_DOWN@1.1597) "
        "M5(RANGE,ADX=14,ST=+,struct=BOS_DOWN@1.1603) "
        "H1(TREND_DOWN,ADX=38,ST=-,struct=CHOCH_DOWN@1.1629) "
        "H4(TREND_DOWN,ADX=42,ST=-,struct=BOS_DOWN@1.1608) "
        "D(UNCLEAR,ADX=19,ST=+)"
    )
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": now.isoformat(),
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "chart_story": chart_story,
                        "session": {"current_tag": "LONDON_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M1",
                                "indicators": {
                                    "atr_pips": 1.0,
                                    "close": 1.16025,
                                    "bb_lower": 1.15990,
                                    "bb_upper": 1.16030,
                                    "stoch_rsi": 0.95,
                                    "williams_r_14": -6.0,
                                    "mfi_14": 84.0,
                                },
                            },
                            {
                                "granularity": "M5",
                                "indicators": {
                                    "atr_pips": 1.0,
                                    "close": 1.16025,
                                    "bb_lower": 1.15950,
                                    "bb_upper": 1.16030,
                                    "stoch_rsi": 0.94,
                                    "williams_r_14": -12.0,
                                },
                            },
                            {
                                "granularity": "M15",
                                "indicators": {"atr_pips": 4.0},
                            },
                            {
                                "granularity": "H1",
                                "regime": "TREND_DOWN",
                                "indicators": {"atr_pips": 8.0},
                            },
                            {
                                "granularity": "H4",
                                "regime": "TREND_DOWN",
                                "indicators": {"atr_pips": 19.0},
                            },
                        ],
                    }
                ]
            }
        )
    )
    return path


def _structural_reversal_pair_charts(root: Path) -> Path:
    path = root / "pair_charts_structural.json"
    chart_story = (
        "M1(TREND_DOWN,ADX=24,ST=-,struct=BOS_DOWN@1.1980) "
        "M5(TREND_DOWN,ADX=27,ST=-,struct=BOS_DOWN@1.1980) "
        "H1(TREND_DOWN,ADX=30,ST=-,struct=BOS_DOWN@1.1980) "
        "H4(TREND_DOWN,ADX=31,ST=-,struct=BOS_DOWN@1.1980) "
        "D(RANGE,ADX=10,ST=-)"
    )
    path.write_text(
        json.dumps(
            {
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "chart_story": chart_story,
                        "session": {"current_tag": "LONDON_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M1",
                                "indicators": {"atr_pips": 1.0},
                            },
                            {
                                "granularity": "M5",
                                "indicators": {"atr_pips": 1.0},
                            },
                            {
                                "granularity": "H1",
                                "regime": "TREND_DOWN",
                                "indicators": {"atr_pips": 4.0},
                                "structure": {
                                    "structure_events": [
                                        {"kind": "BOS_DOWN", "close_confirmed": True},
                                    ]
                                },
                            },
                            {
                                "granularity": "H4",
                                "regime": "TREND_DOWN",
                                "indicators": {"atr_pips": 8.0},
                                "structure": {
                                    "structure_events": [
                                        {"kind": "BOS_DOWN", "close_confirmed": True},
                                    ]
                                },
                            },
                        ],
                    }
                ]
            }
        )
    )
    return path


def _temporary_top_pair_charts(root: Path) -> Path:
    path = root / "pair_charts_temporary_top.json"
    now = datetime.now(timezone.utc)
    recent = [
        {"t": (now - timedelta(minutes=11)).isoformat(), "o": 1.20020, "h": 1.20042, "l": 1.20010, "c": 1.20036, "complete": True},
        {"t": (now - timedelta(minutes=10)).isoformat(), "o": 1.20036, "h": 1.20068, "l": 1.20030, "c": 1.20062, "complete": True},
        {"t": (now - timedelta(minutes=9)).isoformat(), "o": 1.20062, "h": 1.20118, "l": 1.20058, "c": 1.20105, "complete": True},
        {"t": (now - timedelta(minutes=8)).isoformat(), "o": 1.20105, "h": 1.20122, "l": 1.20094, "c": 1.20102, "complete": True},
        {"t": (now - timedelta(minutes=7)).isoformat(), "o": 1.20102, "h": 1.20108, "l": 1.20082, "c": 1.20084, "complete": True},
        {"t": (now - timedelta(minutes=6)).isoformat(), "o": 1.20084, "h": 1.20088, "l": 1.20066, "c": 1.20068, "complete": True},
        {"t": (now - timedelta(minutes=5)).isoformat(), "o": 1.20068, "h": 1.20070, "l": 1.20050, "c": 1.20055, "complete": True},
        {"t": (now - timedelta(minutes=4)).isoformat(), "o": 1.20055, "h": 1.20060, "l": 1.20036, "c": 1.20042, "complete": True},
    ]
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": now.isoformat(),
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "chart_story": (
                            "M1(TREND_DOWN,ADX=27,ST=-,struct=BOS_UP@1.2010) "
                            "M5(RANGE,ADX=16,ST=-,struct=CHOCH_UP@1.2008:wick) "
                            "H1(TREND_UP,ADX=25,ST=+) "
                            "H4(TREND_UP,ADX=28,ST=+) "
                            "D(TREND_UP,ADX=33,ST=+)"
                        ),
                        "confluence": {
                            "price_percentile_24h": 0.57,
                            "price_percentile_7d": 0.82,
                            "range_24h_sigma_multiple": 3.0,
                            "score_balance": "LONG_LEAN",
                            "score_gap": 0.84,
                            "tf_agreement_score": 0.67,
                        },
                        "session": {"current_tag": "NY_AM_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M1",
                                "regime": "TREND_DOWN",
                                "long_bias": 0.29,
                                "short_bias": 0.71,
                                "recent_candles": recent,
                                "indicators": {
                                    "atr_pips": 1.4,
                                    "bb_upper": 1.20115,
                                    "bb_lower": 1.20030,
                                    "donchian_high": 1.20122,
                                    "donchian_low": 1.20030,
                                    "supertrend_dir": -1,
                                    "psar_dir": -1,
                                },
                                "structure": {
                                    "fair_value_gaps": [
                                        {"direction": "DOWN", "filled": False},
                                    ]
                                },
                            },
                            {
                                "granularity": "M5",
                                "regime": "RANGE",
                                "long_bias": 0.62,
                                "short_bias": 0.20,
                                "indicators": {
                                    "atr_pips": 3.8,
                                    "bb_upper": 1.20110,
                                    "bb_lower": 1.19980,
                                    "donchian_high": 1.20122,
                                    "donchian_low": 1.19980,
                                    "supertrend_dir": -1,
                                },
                            },
                            {
                                "granularity": "H1",
                                "regime": "TREND_UP",
                                "indicators": {"atr_pips": 8.0},
                            },
                            {
                                "granularity": "H4",
                                "regime": "TREND_UP",
                                "indicators": {"atr_pips": 18.0},
                            },
                        ],
                    }
                ],
            }
        )
    )
    return path


def _mfe_giveback_pair_charts(root: Path) -> Path:
    path = root / "pair_charts_mfe_giveback.json"
    now = datetime.now(timezone.utc)
    recent = [
        {"t": (now - timedelta(minutes=7)).isoformat(), "o": 1.20048, "h": 1.20066, "l": 1.20034, "c": 1.20062, "complete": True},
        {"t": (now - timedelta(minutes=6)).isoformat(), "o": 1.20062, "h": 1.20075, "l": 1.20042, "c": 1.20070, "complete": True},
        {"t": (now - timedelta(minutes=5)).isoformat(), "o": 1.20070, "h": 1.20072, "l": 1.20038, "c": 1.20050, "complete": True},
        {"t": (now - timedelta(minutes=4)).isoformat(), "o": 1.20050, "h": 1.20055, "l": 1.20022, "c": 1.20028, "complete": True},
        {"t": (now - timedelta(minutes=3)).isoformat(), "o": 1.20028, "h": 1.20032, "l": 1.20006, "c": 1.20012, "complete": True},
        {"t": (now - timedelta(minutes=2)).isoformat(), "o": 1.20012, "h": 1.20018, "l": 1.19992, "c": 1.20000, "complete": True},
    ]
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": now.isoformat(),
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "chart_story": (
                            "M1(TREND_DOWN,ADX=25,ST=-,struct=CHOCH_DOWN@1.2002) "
                            "M5(RANGE,ADX=18,ST=-,struct=CHOCH_UP@1.2008:wick) "
                            "H1(TREND_UP,ADX=25,ST=+) "
                            "H4(TREND_UP,ADX=28,ST=+) "
                            "D(TREND_UP,ADX=33,ST=+)"
                        ),
                        "confluence": {
                            "price_percentile_24h": 0.50,
                            "price_percentile_7d": 0.52,
                            "score_balance": "LONG_LEAN",
                            "score_gap": 0.54,
                            "tf_agreement_score": 0.67,
                        },
                        "session": {"current_tag": "NY_AM_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M1",
                                "regime": "TREND_DOWN",
                                "long_bias": 0.30,
                                "short_bias": 0.70,
                                "recent_candles": recent,
                                "indicators": {
                                    "atr_pips": 8.0,
                                    "supertrend_dir": -1,
                                    "psar_dir": -1,
                                },
                            },
                            {
                                "granularity": "M5",
                                "regime": "RANGE",
                                "long_bias": 0.50,
                                "short_bias": 0.50,
                                "indicators": {"atr_pips": 4.0},
                            },
                            {"granularity": "H1", "regime": "TREND_UP", "indicators": {"atr_pips": 8.0}},
                            {"granularity": "H4", "regime": "TREND_UP", "indicators": {"atr_pips": 18.0}},
                        ],
                    }
                ],
            }
        )
    )
    return path


def _local_swing_top_without_rail_pair_charts(root: Path) -> Path:
    path = root / "pair_charts_local_swing_top.json"
    now = datetime.now(timezone.utc)
    recent = [
        {"t": (now - timedelta(minutes=9)).isoformat(), "o": 1.19992, "h": 1.20010, "l": 1.19988, "c": 1.20005, "complete": True},
        {"t": (now - timedelta(minutes=8)).isoformat(), "o": 1.20005, "h": 1.20036, "l": 1.20002, "c": 1.20030, "complete": True},
        {"t": (now - timedelta(minutes=7)).isoformat(), "o": 1.20030, "h": 1.20072, "l": 1.20028, "c": 1.20068, "complete": True},
        {"t": (now - timedelta(minutes=6)).isoformat(), "o": 1.20068, "h": 1.20070, "l": 1.20058, "c": 1.20066, "complete": True},
        {"t": (now - timedelta(minutes=5)).isoformat(), "o": 1.20066, "h": 1.20067, "l": 1.20056, "c": 1.20064, "complete": True},
        {"t": (now - timedelta(minutes=4)).isoformat(), "o": 1.20064, "h": 1.20065, "l": 1.20054, "c": 1.20063, "complete": True},
        {"t": (now - timedelta(minutes=3)).isoformat(), "o": 1.20063, "h": 1.20064, "l": 1.20052, "c": 1.20062, "complete": True},
    ]
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": now.isoformat(),
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "chart_story": (
                            "M1(TREND_DOWN,ADX=25,ST=-,struct=CHOCH_DOWN@1.2006) "
                            "M5(TREND_DOWN,ADX=23,ST=-,struct=CHOCH_DOWN@1.2006) "
                            "H1(TREND_UP,ADX=25,ST=+) "
                            "H4(TREND_UP,ADX=28,ST=+) "
                            "D(TREND_UP,ADX=33,ST=+)"
                        ),
                        "confluence": {
                            "price_percentile_24h": 0.51,
                            "price_percentile_7d": 0.54,
                            "range_24h_sigma_multiple": 0.9,
                            "score_balance": "LONG_LEAN",
                            "score_gap": 0.74,
                            "tf_agreement_score": 0.67,
                        },
                        "session": {"current_tag": "NY_AM_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M1",
                                "regime": "TREND_DOWN",
                                "long_bias": 0.31,
                                "short_bias": 0.69,
                                "recent_candles": recent,
                                "indicators": {
                                    "atr_pips": 1.2,
                                    "supertrend_dir": -1,
                                    "psar_dir": -1,
                                },
                            },
                            {
                                "granularity": "M5",
                                "regime": "TREND_DOWN",
                                "long_bias": 0.35,
                                "short_bias": 0.65,
                                "indicators": {
                                    "atr_pips": 3.8,
                                    "supertrend_dir": -1,
                                    "psar_dir": -1,
                                },
                            },
                            {"granularity": "H1", "regime": "TREND_UP", "indicators": {"atr_pips": 8.0}},
                            {"granularity": "H4", "regime": "TREND_UP", "indicators": {"atr_pips": 18.0}},
                        ],
                    }
                ],
            }
        )
    )
    return path


def _three_signal_local_top_pair_charts(root: Path) -> Path:
    path = root / "pair_charts_three_signal_local_top.json"
    now = datetime.now(timezone.utc)
    recent = [
        {"t": (now - timedelta(minutes=6)).isoformat(), "o": 1.19992, "h": 1.20010, "l": 1.19988, "c": 1.20005, "complete": True},
        {"t": (now - timedelta(minutes=5)).isoformat(), "o": 1.20005, "h": 1.20044, "l": 1.20002, "c": 1.20038, "complete": True},
        {"t": (now - timedelta(minutes=4)).isoformat(), "o": 1.20038, "h": 1.20090, "l": 1.20035, "c": 1.20084, "complete": True},
        {"t": (now - timedelta(minutes=3)).isoformat(), "o": 1.20084, "h": 1.20086, "l": 1.20070, "c": 1.20078, "complete": True},
        {"t": (now - timedelta(minutes=2)).isoformat(), "o": 1.20078, "h": 1.20080, "l": 1.20066, "c": 1.20070, "complete": True},
        {"t": (now - timedelta(minutes=1)).isoformat(), "o": 1.20070, "h": 1.20072, "l": 1.20060, "c": 1.20064, "complete": True},
    ]
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": now.isoformat(),
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "chart_story": (
                            "M1(TREND_UP,ADX=25,ST=+,struct=BOS_UP@1.2004) "
                            "M5(TREND_UP,ADX=23,ST=+,struct=BOS_UP@1.2004) "
                            "H1(TREND_UP,ADX=25,ST=+) "
                            "H4(TREND_UP,ADX=28,ST=+) "
                            "D(TREND_UP,ADX=33,ST=+)"
                        ),
                        "confluence": {
                            "price_percentile_24h": 0.52,
                            "price_percentile_7d": 0.56,
                            "range_24h_sigma_multiple": 0.9,
                            "score_balance": "LONG_LEAN",
                            "score_gap": 0.74,
                            "tf_agreement_score": 0.67,
                        },
                        "session": {"current_tag": "NY_AM_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M1",
                                "regime": "TREND_UP",
                                "long_bias": 0.60,
                                "short_bias": 0.40,
                                "recent_candles": recent,
                                "indicators": {
                                    "atr_pips": 1.2,
                                    "supertrend_dir": 1,
                                    "psar_dir": 1,
                                },
                            },
                            {
                                "granularity": "M5",
                                "regime": "TREND_UP",
                                "long_bias": 0.62,
                                "short_bias": 0.38,
                                "indicators": {
                                    "atr_pips": 3.8,
                                    "supertrend_dir": 1,
                                    "psar_dir": 1,
                                },
                            },
                            {"granularity": "H1", "regime": "TREND_UP", "indicators": {"atr_pips": 8.0}},
                            {"granularity": "H4", "regime": "TREND_UP", "indicators": {"atr_pips": 18.0}},
                        ],
                    }
                ],
            }
        )
    )
    return path


def _delayed_local_swing_top_pair_charts(root: Path) -> Path:
    path = root / "pair_charts_delayed_local_swing_top.json"
    now = datetime.now(timezone.utc)
    recent = [
        {"t": (now - timedelta(minutes=15)).isoformat(), "o": 1.20034, "h": 1.20048, "l": 1.20032, "c": 1.20044, "complete": True},
        {"t": (now - timedelta(minutes=14)).isoformat(), "o": 1.20044, "h": 1.20070, "l": 1.20042, "c": 1.20066, "complete": True},
        {"t": (now - timedelta(minutes=13)).isoformat(), "o": 1.20066, "h": 1.20090, "l": 1.20062, "c": 1.20084, "complete": True},
        {"t": (now - timedelta(minutes=12)).isoformat(), "o": 1.20084, "h": 1.20086, "l": 1.20070, "c": 1.20076, "complete": True},
        {"t": (now - timedelta(minutes=11)).isoformat(), "o": 1.20076, "h": 1.20078, "l": 1.20067, "c": 1.20070, "complete": True},
        {"t": (now - timedelta(minutes=10)).isoformat(), "o": 1.20070, "h": 1.20074, "l": 1.20062, "c": 1.20068, "complete": True},
        {"t": (now - timedelta(minutes=9)).isoformat(), "o": 1.20068, "h": 1.20072, "l": 1.20060, "c": 1.20066, "complete": True},
        {"t": (now - timedelta(minutes=8)).isoformat(), "o": 1.20066, "h": 1.20068, "l": 1.20056, "c": 1.20064, "complete": True},
        {"t": (now - timedelta(minutes=7)).isoformat(), "o": 1.20064, "h": 1.20066, "l": 1.20054, "c": 1.20062, "complete": True},
        {"t": (now - timedelta(minutes=6)).isoformat(), "o": 1.20062, "h": 1.20064, "l": 1.20052, "c": 1.20060, "complete": True},
        {"t": (now - timedelta(minutes=5)).isoformat(), "o": 1.20060, "h": 1.20062, "l": 1.20050, "c": 1.20058, "complete": True},
        {"t": (now - timedelta(minutes=4)).isoformat(), "o": 1.20058, "h": 1.20061, "l": 1.20049, "c": 1.20057, "complete": True},
        {"t": (now - timedelta(minutes=3)).isoformat(), "o": 1.20057, "h": 1.20060, "l": 1.20048, "c": 1.20059, "complete": True},
        {"t": (now - timedelta(minutes=2)).isoformat(), "o": 1.20059, "h": 1.20060, "l": 1.20047, "c": 1.20055, "complete": True},
        {"t": (now - timedelta(minutes=1)).isoformat(), "o": 1.20055, "h": 1.20058, "l": 1.20045, "c": 1.20052, "complete": True},
    ]
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": now.isoformat(),
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "chart_story": (
                            "M1(IMPULSE_DOWN,ADX=24,ST=-,struct=CHOCH_DOWN@1.2006) "
                            "M5(TREND_UP,ADX=28,ST=+,struct=BOS_UP@1.2005) "
                            "H1(TREND_UP,ADX=24,ST=+) "
                            "H4(TREND_UP,ADX=26,ST=+) "
                            "D(TREND_UP,ADX=31,ST=+)"
                        ),
                        "confluence": {
                            "price_percentile_24h": 0.52,
                            "price_percentile_7d": 0.55,
                            "range_24h_sigma_multiple": 0.9,
                            "score_balance": "LONG_LEAN",
                            "score_gap": 0.61,
                            "tf_agreement_score": 0.67,
                        },
                        "session": {"current_tag": "NY_AM_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M1",
                                "regime": "IMPULSE_DOWN",
                                "long_bias": 0.22,
                                "short_bias": 0.78,
                                "recent_candles": recent,
                                "indicators": {
                                    "atr_pips": 1.2,
                                    "supertrend_dir": -1,
                                    "psar_dir": -1,
                                    "donchian_high": 1.20090,
                                    "bb_upper": 1.20150,
                                },
                            },
                            {
                                "granularity": "M5",
                                "regime": "TREND_UP",
                                "long_bias": 0.55,
                                "short_bias": 0.45,
                                "indicators": {
                                    "atr_pips": 3.8,
                                    "supertrend_dir": 1,
                                    "psar_dir": -1,
                                },
                            },
                            {"granularity": "H1", "regime": "TREND_UP", "indicators": {"atr_pips": 8.0}},
                            {"granularity": "H4", "regime": "TREND_UP", "indicators": {"atr_pips": 18.0}},
                        ],
                    }
                ],
            }
        )
    )
    return path


def _entry_invalidation_technical_pair_charts(root: Path) -> Path:
    path = root / "pair_charts_entry_invalidation.json"
    chart_story = (
        "M1(TREND_DOWN,ADX=21,ST=-,struct=NONE) "
        "M5(TREND_DOWN,ADX=22,ST=-,struct=NONE) "
        "M15(TREND_DOWN,ADX=23,ST=-,struct=NONE) "
        "M30(TREND_DOWN,ADX=21,ST=-,struct=NONE) "
        "H1(TREND_DOWN,ADX=20,ST=-,struct=NONE) "
        "H4(RANGE,ADX=12,ST=-)"
    )
    adverse_indicators = {
        "atr_pips": 1.0,
        "rsi_14": 39.0,
        "macd_hist": -0.0001,
        "supertrend_dir": -1,
        "ichimoku_cloud_pos": -1,
        "plus_di_14": 13.0,
        "minus_di_14": 24.0,
    }
    path.write_text(
        json.dumps(
            {
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "chart_story": chart_story,
                        "session": {"current_tag": "LONDON_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M5",
                                "regime": "TREND_DOWN",
                                "indicators": dict(adverse_indicators),
                            },
                            {
                                "granularity": "M15",
                                "regime": "TREND_DOWN",
                                "indicators": dict(adverse_indicators),
                            },
                            {
                                "granularity": "M30",
                                "regime": "TREND_DOWN",
                                "indicators": dict(adverse_indicators),
                            },
                            {
                                "granularity": "H1",
                                "regime": "TREND_DOWN",
                                "indicators": dict(adverse_indicators),
                            },
                            {
                                "granularity": "H4",
                                "regime": "RANGE",
                                "indicators": {"atr_pips": 8.0},
                            },
                        ],
                    }
                ]
            }
        )
    )
    return path


if __name__ == "__main__":
    unittest.main()

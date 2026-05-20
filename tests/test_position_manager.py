from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.models import BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.strategy.position_manager import (
    ACTION_BREAK_EVEN_STOP,
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
            self.assertIn("remaining risk", (root / "pm.md").read_text())

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

    def test_sl_free_profitable_short_gets_break_even_after_micro_noise_clears(self) -> None:
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

                self.assertEqual(result.action, ACTION_BREAK_EVEN_STOP)
                self.assertEqual(result.positions[0].action, ACTION_BREAK_EVEN_STOP)
                self.assertEqual(result.positions[0].recommended_stop_loss, 1.16077)
                self.assertIn("SL-free profit BE trigger", (root / "pm.md").read_text())
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
                self.assertIn("SL-free BE deferred", (root / "pm.md").read_text())
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
                self.assertIn("legacy/no-ledger", report)
        finally:
            if prior_close is None:
                os.environ.pop("QR_DISABLE_AUTO_CLOSE", None)
            else:
                os.environ["QR_DISABLE_AUTO_CLOSE"] = prior_close
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

    def test_auto_close_kill_switch_allows_next_generation_structural_exit(self) -> None:
        prior_close = os.environ.get("QR_DISABLE_AUTO_CLOSE")
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_DISABLE_AUTO_CLOSE"] = "1"
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
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
                    bid=1.2010,
                    ask=1.2011,
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


def _pair_charts(root: Path, *, atr_pips: float) -> Path:
    path = root / "pair_charts.json"
    path.write_text(
        json.dumps(
            {
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "session": {"current_tag": "LONDON_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M5",
                                "indicators": {"atr_pips": atr_pips},
                            }
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


if __name__ == "__main__":
    unittest.main()

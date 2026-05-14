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
    ACTION_HOLD_PROTECTED,
    ACTION_PROFIT_PROTECT,
    ACTION_REPAIR_PROTECTION,
    ACTION_REVIEW_EXIT,
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

    def test_operator_manual_position_is_not_managed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = _decision(root, long_score=160, short_score=120)
            snapshot = BrokerSnapshot(
                fetched_at_utc=datetime.now(timezone.utc),
                positions=(
                    BrokerPosition(
                        trade_id="manual-1",
                        pair="USD_JPY",
                        side=Side.LONG,
                        units=25000,
                        entry_price=155.962,
                        owner=Owner.UNKNOWN,
                    ),
                ),
                quotes={"USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=datetime.now(timezone.utc))},
            )

            result = PositionManager(
                trader_decision_path=decision,
                pair_charts_path=root / "missing_pair_charts.json",
                output_path=root / "pm.json",
                report_path=root / "pm.md",
            ).run(snapshot)

            self.assertEqual(result.action, "NO_POSITION")
            self.assertEqual(result.positions, ())
            self.assertIn("manual/tagless positions", (root / "pm.md").read_text())


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


if __name__ == "__main__":
    unittest.main()

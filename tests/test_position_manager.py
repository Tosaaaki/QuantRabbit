from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.models import BrokerPosition, BrokerSnapshot, Quote, Side
from quant_rabbit.strategy.position_manager import ACTION_HOLD_PROTECTED, ACTION_PROFIT_PROTECT, ACTION_REPAIR_PROTECTION, PositionManager


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
                output_path=root / "pm.json",
                report_path=root / "pm.md",
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_REPAIR_PROTECTION)
            self.assertIsNotNone(result.positions[0].recommended_stop_loss)

    def test_profit_requires_break_even_stop_plan(self) -> None:
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
                    unrealized_pl_jpy=90,
                    take_profit=1.1741,
                    stop_loss=1.1721,
                ),
                bid=1.1738,
                ask=1.1739,
            )

            result = PositionManager(
                trader_decision_path=decision,
                output_path=root / "pm.json",
                report_path=root / "pm.md",
            ).run(snapshot)

            self.assertEqual(result.action, ACTION_PROFIT_PROTECT)
            self.assertEqual(result.positions[0].recommended_stop_loss, 1.1729)


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


def _snapshot(position: BrokerPosition, *, bid: float = 1.1728, ask: float = 1.1729) -> BrokerSnapshot:
    now = datetime.now(timezone.utc)
    return BrokerSnapshot(
        fetched_at_utc=now,
        positions=(position,),
        quotes={"EUR_USD": Quote("EUR_USD", bid, ask, timestamp_utc=now)},
    )


if __name__ == "__main__":
    unittest.main()

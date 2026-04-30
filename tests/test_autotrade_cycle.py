from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.automation import AutoTradeCycle
from quant_rabbit.models import BrokerOrder, BrokerSnapshot, Owner, Quote


class AutoTradeCycleTest(unittest.TestCase):
    def test_existing_pending_order_turns_cycle_monitor_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    orders=(
                        BrokerOrder(
                            order_id="1",
                            pair="AUD_JPY",
                            order_type="STOP",
                            price=112.576,
                            state="PENDING",
                            units=1000,
                            owner=Owner.TRADER,
                        ),
                    ),
                    quotes={"AUD_JPY": Quote("AUD_JPY", 112.49, 112.50, timestamp_utc=now)},
                )
            )

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "MONITOR_ONLY_EXPOSURE_OPEN")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders_sent, [])
            self.assertIn("monitor-only", (root / "report.md").read_text())
            self.assertTrue((root / "decision.json").exists())


class FakeCycleClient:
    def __init__(self, snapshot: BrokerSnapshot) -> None:
        self.snapshot_value = snapshot
        self.orders_sent: list[dict[str, Any]] = []
        self.orders_canceled: list[str] = []

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        return self.snapshot_value

    def post_order_json(self, order_request: dict[str, Any]) -> dict[str, Any]:
        self.orders_sent.append(order_request)
        return {"orderCreateTransaction": {"id": "1"}}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        self.orders_canceled.append(order_id)
        return {"orderCancelTransaction": {"id": "2", "orderID": order_id}}


if __name__ == "__main__":
    unittest.main()

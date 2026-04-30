from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.strategy.profile import StrategyProfile
from quant_rabbit.strategy.receipt_promotion import ReceiptPromoter


class ReceiptPromotionTest(unittest.TestCase):
    def test_promotes_risk_repair_and_pending_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = _profile(root)
            intents = _intents(root)
            report = root / "promotion.md"

            summary = ReceiptPromoter(profile_path=profile, intents_path=intents, report_path=report).run()

            self.assertEqual(summary.promoted, 2)
            payload = json.loads(profile.read_text())
            statuses = {(item["pair"], item["direction"]): item["status"] for item in payload["profiles"]}
            self.assertEqual(statuses[("EUR_USD", "LONG")], "CANDIDATE")
            self.assertEqual(statuses[("EUR_JPY", "LONG")], "CANDIDATE")
            self.assertEqual(statuses[("USD_JPY", "SHORT")], "BLOCK_UNTIL_NEW_EVIDENCE")
            loaded = StrategyProfile.load(profile)
            self.assertEqual(loaded.validate(_intent("EUR_USD", "LONG"), for_live_send=True), ())
            self.assertIn("RISK_REPAIR_CANDIDATE -> CANDIDATE", report.read_text())

    def test_does_not_promote_blocking_risk_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = _profile(root)
            intents = root / "intents.json"
            intents.write_text(
                json.dumps(
                    {
                        "snapshot_path": "snapshot.json",
                        "results": [
                            _receipt(
                                "risk:EUR_USD:LONG",
                                "EUR_USD",
                                "LONG",
                                "STOP-ENTRY",
                                risk_issues=[
                                    {
                                        "code": "LOSS_CAP_EXCEEDED",
                                        "message": "too much risk",
                                        "severity": "BLOCK",
                                    }
                                ],
                            )
                        ],
                    }
                )
            )

            summary = ReceiptPromoter(profile_path=profile, intents_path=intents, report_path=root / "report.md").run()

            self.assertEqual(summary.promoted, 0)
            payload = json.loads(profile.read_text())
            statuses = {(item["pair"], item["direction"]): item["status"] for item in payload["profiles"]}
            self.assertEqual(statuses[("EUR_USD", "LONG")], "RISK_REPAIR_CANDIDATE")


def _profile(root: Path) -> Path:
    path = root / "profile.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "RISK_REPAIR_CANDIDATE",
                        "required_fix": "cap risk",
                    },
                    {
                        "pair": "EUR_JPY",
                        "direction": "LONG",
                        "status": "MINE_MISSED_EDGE",
                        "required_fix": "arm pending receipt",
                    },
                    {
                        "pair": "USD_JPY",
                        "direction": "SHORT",
                        "status": "BLOCK_UNTIL_NEW_EVIDENCE",
                        "required_fix": "bad live history",
                    },
                ]
            }
        )
    )
    return path


def _intents(root: Path) -> Path:
    path = root / "intents.json"
    path.write_text(
        json.dumps(
            {
                "snapshot_path": "snapshot.json",
                "results": [
                    _receipt("risk:EUR_USD:LONG", "EUR_USD", "LONG", "STOP-ENTRY"),
                    _receipt("trigger:EUR_JPY:LONG", "EUR_JPY", "LONG", "LIMIT"),
                    _receipt("blocked:USD_JPY:SHORT", "USD_JPY", "SHORT", "STOP-ENTRY"),
                ],
            }
        )
    )
    return path


def _receipt(
    lane_id: str,
    pair: str,
    direction: str,
    order_type: str,
    *,
    risk_issues: list[dict[str, str]] | None = None,
) -> dict:
    return {
        "lane_id": lane_id,
        "status": "DRY_RUN_PASSED",
        "risk_allowed": True,
        "risk_issues": risk_issues or [],
        "strategy_issues": [],
        "live_blockers": [],
        "intent": {
            "pair": pair,
            "side": direction,
            "order_type": order_type,
            "units": 1000,
            "entry": 1.1,
            "tp": 1.2,
            "sl": 1.0,
            "thesis": "test",
            "owner": "trader",
        },
    }


def _intent(pair: str, direction: str):
    from quant_rabbit.models import OrderIntent, OrderType, Owner, Side

    return OrderIntent(
        pair=pair,
        side=Side.parse(direction),
        order_type=OrderType.STOP_ENTRY,
        units=1000,
        entry=1.1,
        tp=1.2,
        sl=1.0,
        thesis="test",
        owner=Owner.TRADER,
    )


if __name__ == "__main__":
    unittest.main()

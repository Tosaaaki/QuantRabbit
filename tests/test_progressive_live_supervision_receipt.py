from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from quant_rabbit.fast_bot import REGIME_CONTRACT, _seal, build_fast_bot_shadow
from quant_rabbit.fast_bot_promotion import _contains_forbidden_order_keys


TOOL = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "seal_progressive_live_supervision_receipt.py"
)
SPEC = importlib.util.spec_from_file_location(
    "seal_progressive_live_supervision_receipt",
    TOOL,
)
assert SPEC is not None and SPEC.loader is not None
supervision_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervision_tool)


class ProgressiveLiveSupervisionReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
        regime = _seal(
            {
                "contract": REGIME_CONTRACT,
                "schema_version": 1,
                "generated_at_utc": self.now.isoformat(),
                "rows": [
                    {
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "method": "RANGE_ROTATION",
                        "state": "GO",
                        "execution_enabled": True,
                        "score": 5.0,
                        "m1_closed_candle_utc": self.now.isoformat(),
                        "m5_atr_pips": 5.0,
                    }
                ],
            }
        )
        shadow = build_fast_bot_shadow(
            regime,
            broker_snapshot={
                "quotes": {
                    "EUR_USD": {
                        "bid": 1.16410,
                        "ask": 1.16418,
                        "timestamp_utc": self.now.isoformat(),
                    }
                }
            },
            now_utc=self.now,
        )
        self.signal = shadow["signals"][0]
        self.manifest = {"software_version_sha256": "a" * 64}
        body = {
            "contract": "QR_PROGRESSIVE_LIVE_MODE_LEDGER_V1",
            "evaluated_at_utc": self.now.isoformat(),
            "software_version_sha256": "a" * 64,
            "release_receipt_sha256": "b" * 64,
            "mode": "THROTTLED_LIVE",
            "signal_receipts": [
                {
                    "signal": self.signal,
                    "mode_receipt": {
                        "mode": "THROTTLED_LIVE",
                        "calculated_units": 1,
                    },
                }
            ],
        }
        self.event = {
            **body,
            "event_id": f"qrplm:{supervision_tool._canonical_sha(body)}",
        }
        self.state = {
            "contract": "QR_PROGRESSIVE_LIVE_PREFLIGHT_V1",
            "last_event_id": self.event["event_id"],
            "mode": "THROTTLED_LIVE",
            "promotion_ready": True,
        }
        self.admission = {"allowed_strategy_ids": [self.signal["strategy_id"]]}
        self.risk = {
            "max_loss_per_order_jpy": 500.0,
            "max_bot_positions": 2,
        }
        self.release = {"release_receipt_sha256": "b" * 64}

    def build(self, **overrides):
        values = {
            "preflight_state": self.state,
            "event": self.event,
            "release_receipt": self.release,
            "software_manifest": self.manifest,
            "expected_packet_sha256": "c" * 64,
            "decision": "ALLOW",
            "regime": "RANGE",
            "allowed_strategy_ids": [f"live-{self.signal['strategy_id']}"],
            "risk_budget_cap_jpy": 500.0,
            "max_positions_cap": 2,
            "expiry_seconds": 300,
            "review_reason": "Current bounded regime review allows this strategy.",
            "now_utc": self.now,
        }
        values.update(overrides)
        with mock.patch.object(
            supervision_tool._PREFLIGHT,
            "verify_release_receipt",
            return_value=(self.admission, self.risk),
        ):
            return supervision_tool.build_progressive_live_supervision_receipt(**values)

    def test_allow_receipt_binds_exact_event_and_has_no_order_fields(self) -> None:
        receipt = self.build()

        self.assertEqual(receipt["decision"], "ALLOW")
        self.assertEqual(receipt["event_id"], self.event["event_id"])
        self.assertEqual(receipt["dedupe_key"], self.event["event_id"])
        self.assertEqual(
            receipt["feature_snapshot_sha256"],
            self.event["event_id"].removeprefix("qrplm:"),
        )
        self.assertEqual(receipt["signal_sha256"], self.signal["signal_sha256"])
        self.assertEqual(receipt["ai_order_authority"], "NONE")
        self.assertFalse(receipt["broker_mutation_allowed"])
        self.assertFalse(_contains_forbidden_order_keys(receipt))
        self.assertIn("receipt_sha256", receipt)

    def test_allow_cannot_exceed_user_risk_contract(self) -> None:
        with self.assertRaisesRegex(
            supervision_tool.SupervisionSealBlocked,
            "RISK_BUDGET_EXCEEDS_USER_CONTRACT",
        ):
            self.build(risk_budget_cap_jpy=500.01)

    def test_freeze_has_zero_entry_capacity_and_no_signal_order_fields(self) -> None:
        receipt = self.build(
            decision="FREEZE_NEW",
            allowed_strategy_ids=[],
            risk_budget_cap_jpy=0.0,
            max_positions_cap=0,
        )

        self.assertEqual(receipt["decision"], "FREEZE_NEW")
        self.assertEqual(receipt["allowed_strategy_ids"], [])
        self.assertEqual(receipt["risk_budget_cap_jpy"], 0.0)
        self.assertFalse(_contains_forbidden_order_keys(receipt))


if __name__ == "__main__":
    unittest.main()

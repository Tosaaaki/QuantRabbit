from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_progressive_live_owner_cycle.py"
SPEC = importlib.util.spec_from_file_location("run_progressive_live_owner_cycle", TOOL)
assert SPEC is not None and SPEC.loader is not None
owner_cycle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(owner_cycle)


class ProgressiveLiveOwnerCycleTest(unittest.TestCase):
    def test_supervision_is_bound_to_actual_preflight_event(self) -> None:
        class RecordingInventory:
            def __init__(self) -> None:
                self.event = None

            def apply_supervision_receipt(self, *, event, receipt, now_utc):
                self.event = event
                return "APPLIED_ALLOW"

        inventory = RecordingInventory()
        result = owner_cycle._apply_supervision_to_inventory(
            inventory,
            supervision={
                "event_id": "receipt-self-asserted-event",
                "dedupe_key": "receipt-self-asserted-event",
            },
            mode_event={"event_id": "qrplm:" + "a" * 64},
            now_utc=owner_cycle.datetime.now(owner_cycle.timezone.utc),
        )

        self.assertEqual(result, "APPLIED_ALLOW")
        self.assertEqual(
            inventory.event,
            {
                "event_id": "qrplm:" + "a" * 64,
                "dedupe_key": "qrplm:" + "a" * 64,
            },
        )

    def test_unaccepted_or_unsafe_preflight_never_constructs_write_client(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            blocked = {
                "mode": "SHADOW_ONLY",
                "transition_reason": "RISK_CONTRACT_UNACCEPTED",
                "needs_user_decision": True,
                "waiting_external_state": False,
                "promotion_ready": False,
            }
            with mock.patch.object(owner_cycle, "run_preflight", return_value=blocked), mock.patch.object(
                owner_cycle.OandaExecutionClient,
                "__init__",
                side_effect=AssertionError("write-capable client must not be constructed"),
            ):
                result = owner_cycle.run_owner_cycle(
                    env_file=root / "env",
                    approval_packet_path=root / "packet.json",
                    expected_packet_sha256="a" * 64,
                    resident_status_path=root / "resident.json",
                    release_receipt_path=None,
                    supervision_receipt_path=None,
                    inventory_state_path=root / "inventory.json",
                    preflight_state_root=root / "preflight",
                    owner_state_root=root / "owner",
                    strategy_profile_path=root / "profile.json",
                    execution_ledger_path=root / "execution.db",
                    target_state_path=root / "target.json",
                )
            self.assertEqual(result["status"], "NO_LIVE_DISPATCH")
            self.assertEqual(result["live_order_gateway_invocation_count"], 0)
            self.assertEqual(result["external_order_attempts"], 0)
            self.assertEqual(result["external_orders"], 0)
            self.assertFalse(result["broker_mutation_performed"])
            self.assertEqual(result["manual_tagless_policy"], "NO_TOUCH")


if __name__ == "__main__":
    unittest.main()

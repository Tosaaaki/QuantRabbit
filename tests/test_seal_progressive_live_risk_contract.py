from __future__ import annotations

import importlib.util
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.fast_bot_promotion import (
    FORWARD_ADMISSION_CONTRACT,
    RISK_CONTRACT,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "seal_progressive_live_risk_contract",
    ROOT / "tools" / "seal_progressive_live_risk_contract.py",
)
assert SPEC and SPEC.loader
sealer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sealer)


def packet() -> dict:
    value = {
        "contract": sealer.PACKET_CONTRACT,
        "status": "NEEDS_USER_DECISION",
        "initial_pairs": ["EUR_USD", "USD_JPY"],
        "allowed_strategy_ids": [
            "trend_continuation",
            "range_rotation",
            "breakout_failure",
        ],
        "candidate_limits": {
            "max_loss_per_order_jpy": 500.0,
            "stop_drawdown_jpy": 5000.0,
            "minimum_margin_buffer_jpy": 50000.0,
            "max_post_entry_current_mcp": 0.85,
            "max_post_entry_stress_mcp": 0.9,
            "max_currency_factor_nav_multiple": 3.0,
            "max_bot_positions": 2,
            "mode_hysteresis_mcp": 0.03,
            "stress_pips": 25.0,
            "max_account_snapshot_age_seconds": 20.0,
        },
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return {**value, "packet_sha256": sealer.canonical_sha(value)}


def resident() -> dict:
    return {
        "run_state": "RUNNING",
        "execution_authority": "NONE",
        "broker_mutation_allowed": False,
        "live_permission": False,
        "promotion_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_positions_policy": "NO_TOUCH",
        "existing_tp_sl_policy": "NO_TOUCH",
        "last_error": None,
        "source_commit": "a" * 40,
        "source_bundle_sha256": "b" * 64,
        "pid": 123,
        "started_at_utc": "2026-08-28T00:00:00+00:00",
        "heartbeat_at_utc": "2026-08-28T00:01:00+00:00",
    }


class SealProgressiveLiveRiskContractTests(unittest.TestCase):
    def test_exact_packet_and_zero_authority_resident_build_release(self) -> None:
        approval = packet()
        release = sealer.build_release_receipt(
            packet=approval,
            expected_packet_sha256=approval["packet_sha256"],
            resident_status=resident(),
            manifest={
                "commit": "c" * 40,
                "git_tree": "d" * 40,
                "files": {"source.py": "e" * 64},
                "software_version_sha256": "f" * 64,
            },
            acceptance_id="explicit-user-risk-approval-20260828",
            accepted_at_utc=datetime(2026, 8, 28, tzinfo=timezone.utc),
            live_campaign_id="live-fb-20260828-cycle-1",
        )
        self.assertEqual(release["status"], "SEALED_AWAITING_FRESH_ACCOUNT_GATE")
        self.assertFalse(release["live_permission"])
        self.assertFalse(release["broker_mutation_allowed"])
        self.assertEqual(
            release["forward_admission"]["contract"],
            FORWARD_ADMISSION_CONTRACT,
        )
        self.assertEqual(release["risk_contract"]["contract"], RISK_CONTRACT)
        self.assertTrue(release["risk_contract"]["accepted_by_user"])
        self.assertEqual(release["risk_contract"]["max_loss_per_order_jpy"], 500.0)
        body = {
            key: value
            for key, value in release.items()
            if key != "release_receipt_sha256"
        }
        self.assertEqual(release["release_receipt_sha256"], sealer.canonical_sha(body))

    def test_modified_packet_is_rejected(self) -> None:
        approval = packet()
        expected = approval["packet_sha256"]
        approval["candidate_limits"]["stop_drawdown_jpy"] = 6000.0
        with self.assertRaisesRegex(sealer.SealBlocked, "SHA256_MISMATCH"):
            sealer.verify_approval_packet(
                approval,
                expected_packet_sha256=expected,
            )

    def test_resident_authority_or_order_attempt_blocks_seal(self) -> None:
        for key, value in (
            ("execution_authority", "LIVE"),
            ("external_order_attempts", 1),
            ("external_orders", 1),
            ("run_state", "STOPPED"),
        ):
            with self.subTest(key=key):
                status = resident()
                status[key] = value
                with self.assertRaisesRegex(
                    sealer.SealBlocked,
                    "RESIDENT_SHADOW_AUTHORITY_OR_STATE_INVALID",
                ):
                    sealer.verify_resident_shadow(status)

    def test_missing_explicit_acceptance_id_blocks_release(self) -> None:
        approval = packet()
        with self.assertRaisesRegex(sealer.SealBlocked, "EXPLICIT_ACCEPTANCE"):
            sealer.build_release_receipt(
                packet=approval,
                expected_packet_sha256=approval["packet_sha256"],
                resident_status=resident(),
                manifest={"software_version_sha256": "f" * 64},
                acceptance_id="",
                accepted_at_utc=datetime(2026, 8, 28, tzinfo=timezone.utc),
                live_campaign_id="live-fb-20260828-cycle-1",
            )


if __name__ == "__main__":
    unittest.main()

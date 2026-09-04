from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.policy_snapshot import (
    POLICY_SNAPSHOT_SCHEMA_VERSION,
    PolicyBinding,
    PolicySnapshotError,
    load_and_verify_policy_snapshot,
    seal_policy_snapshot,
    verify_policy_snapshot,
    write_sealed_policy_snapshot,
)


NOW = datetime(2026, 9, 4, 2, 0, tzinfo=timezone.utc)


def _payload() -> dict[str, object]:
    return {
        "schema_version": POLICY_SNAPSHOT_SCHEMA_VERSION,
        "policy_version": "QR_AI_TRADER_SEALED_POLICY_V1_2026-09-04",
        "project_key": "project.qr-trading",
        "broker_account_id": "paper-account",
        "environment": "practice",
        "issued_at_utc": (NOW - timedelta(minutes=5)).isoformat(),
        "expires_at_utc": (NOW + timedelta(hours=4)).isoformat(),
        "revocation_epoch": 3,
        "source_pages": [
            {"page_id": "project-route", "last_edited_at_utc": NOW.isoformat()},
            {"page_id": "operating-manual", "last_edited_at_utc": NOW.isoformat()},
        ],
        "hot_path": {
            "notion_access_allowed": False,
            "browser_access_allowed": False,
            "ordinary_network_destinations": ["market_data", "broker"],
            "legacy_strategy_authority": "BASELINE_ONLY",
            "manual_positions": "NO_TOUCH",
        },
    }


def _binding() -> PolicyBinding:
    return PolicyBinding(
        project_key="project.qr-trading",
        broker_account_id="paper-account",
        environment="practice",
        revocation_epoch=3,
    )


class PolicySnapshotTests(unittest.TestCase):
    def test_valid_snapshot_round_trip(self) -> None:
        sealed = seal_policy_snapshot(_payload())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "policy.json"
            path.write_text(json.dumps(sealed), encoding="utf-8")
            verified = load_and_verify_policy_snapshot(
                path,
                binding=_binding(),
                now=NOW,
                required_source_pages=("project-route", "operating-manual"),
            )
        self.assertEqual(verified["snapshot_sha256"], sealed["snapshot_sha256"])

    def test_atomic_writer_publishes_verified_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "control" / "policy.json"
            sealed = write_sealed_policy_snapshot(path, _payload())
            verified = load_and_verify_policy_snapshot(path, binding=_binding(), now=NOW)
            self.assertEqual(verified, sealed)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*")), [])

    def test_tamper_fails_closed(self) -> None:
        sealed = seal_policy_snapshot(_payload())
        sealed["environment"] = "live"
        with self.assertRaisesRegex(PolicySnapshotError, "digest") as raised:
            verify_policy_snapshot(sealed, binding=_binding(), now=NOW)
        self.assertEqual(raised.exception.code, "POLICY_SNAPSHOT_TAMPERED")

    def test_expired_fails_closed(self) -> None:
        payload = _payload()
        payload["expires_at_utc"] = (NOW - timedelta(seconds=1)).isoformat()
        with self.assertRaises(PolicySnapshotError) as raised:
            verify_policy_snapshot(seal_policy_snapshot(payload), binding=_binding(), now=NOW)
        self.assertEqual(raised.exception.code, "POLICY_SNAPSHOT_EXPIRED")

    def test_revocation_epoch_mismatch_fails_closed(self) -> None:
        with self.assertRaises(PolicySnapshotError) as raised:
            verify_policy_snapshot(seal_policy_snapshot(_payload()), binding=PolicyBinding(
                project_key="project.qr-trading",
                broker_account_id="paper-account",
                environment="practice",
                revocation_epoch=4,
            ), now=NOW)
        self.assertEqual(raised.exception.code, "POLICY_SNAPSHOT_BINDING_MISMATCH")

    def test_legacy_authority_cannot_be_reenabled(self) -> None:
        payload = _payload()
        payload["hot_path"]["legacy_strategy_authority"] = "LIVE"  # type: ignore[index]
        with self.assertRaises(PolicySnapshotError) as raised:
            verify_policy_snapshot(seal_policy_snapshot(payload), binding=_binding(), now=NOW)
        self.assertEqual(raised.exception.code, "POLICY_SNAPSHOT_RULE_MISMATCH")


if __name__ == "__main__":
    unittest.main()

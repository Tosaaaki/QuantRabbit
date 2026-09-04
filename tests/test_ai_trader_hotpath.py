from __future__ import annotations

import fcntl
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.ai_trading_runtime import PreparedRun
from quant_rabbit.policy_snapshot import POLICY_SNAPSHOT_SCHEMA_VERSION, seal_policy_snapshot
from quant_rabbit.runtime_capacity import CapacityAssessment, CapacityStatus
from tools.ai_trader_hotpath import (
    MAX_OUTPUT_BYTES,
    HotPathOptions,
    _encode_payload,
    run_hotpath,
)


NOW = datetime(2026, 9, 4, 2, 0, tzinfo=timezone.utc)


class AITraderHotPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state_root = self.root / "state"
        self.data = self.root / "data"
        self.data.mkdir()
        (self.data / "market.json").write_text('{"price":1.25}', encoding="utf-8")
        self.config = self.root / "runtime.json"
        self.config.write_text(
            json.dumps(
                {
                    "version": 1,
                    "state_root": str(self.state_root),
                    "profiles": {
                        "intraday": {
                            "kind": "trade",
                            "decision_max_age_seconds": 900,
                            "sink": "paper_ledger",
                            "allowed_actions": ["ENTER", "WAIT", "REQUEST_EVIDENCE"],
                            "workers": {
                                "market": [
                                    {
                                        "path": "data/market.json",
                                        "required": True,
                                        "max_age_seconds": 10**9,
                                    }
                                ]
                            },
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        self.policy = self.root / "policy.json"
        self.policy.write_text(json.dumps(seal_policy_snapshot(_policy_payload())), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def options(self) -> HotPathOptions:
        return HotPathOptions(
            config_path=self.config,
            profile="intraday",
            repo_root=self.root,
            state_root=self.state_root,
            policy_snapshot_path=self.policy,
            project_key="project.qr-trading",
            broker_account_id="paper-account",
            environment="practice",
            revocation_epoch=3,
            required_source_pages=("project-route",),
            lock_path=self.root / "hotpath.lock",
            capacity_filesystem=self.root,
            low_free_bytes=0,
            high_free_bytes=1,
            state_quota_pressure_bytes=10**7,
            state_quota_block_bytes=2 * 10**7,
        )

    def test_overlap_stops_before_policy_or_expensive_work(self) -> None:
        options = self.options()
        descriptor = options.lock_path.open("a+")
        fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with patch("tools.ai_trader_hotpath.load_and_verify_policy_snapshot") as verify:
                code, payload = run_hotpath(options, now=NOW)
        finally:
            descriptor.close()

        self.assertEqual(code, 75)
        self.assertEqual(payload["status"], "LOCKED")
        verify.assert_not_called()

    def test_policy_failure_is_compact_and_never_prepares(self) -> None:
        options = self.options()
        options = HotPathOptions(**{**options.__dict__, "policy_snapshot_path": self.root / "missing.json"})
        with patch("tools.ai_trader_hotpath.prepare_run") as prepare:
            code, payload = run_hotpath(options, now=NOW)

        self.assertNotEqual(code, 0)
        self.assertEqual(payload["status"], "BLOCKED_POLICY")
        self.assertEqual(payload["code"], "POLICY_SNAPSHOT_MISSING")
        prepare.assert_not_called()

    def test_capacity_block_never_prepares(self) -> None:
        assessment = CapacityAssessment(
            status=CapacityStatus.BLOCK,
            filesystem_path=str(self.root),
            total_bytes=100,
            used_bytes=99,
            free_bytes=1,
            low_free_bytes=10,
            high_free_bytes=20,
            roots=(),
            issues=("FILESYSTEM_FREE_BELOW_LOW_WATERMARK",),
        )
        with (
            patch("tools.ai_trader_hotpath.evaluate_capacity", return_value=assessment),
            patch("tools.ai_trader_hotpath.prepare_run") as prepare,
        ):
            code, payload = run_hotpath(self.options(), now=NOW)

        self.assertNotEqual(code, 0)
        self.assertEqual(payload["status"], "BLOCKED_CAPACITY")
        prepare.assert_not_called()

    def test_active_lease_blocks_overlapping_run_without_new_directory(self) -> None:
        first_code, first = run_hotpath(self.options(), now=NOW)
        run_directories = sorted((self.state_root / "runs").iterdir())
        second_code, second = run_hotpath(self.options(), now=NOW + timedelta(minutes=1))

        self.assertEqual(first_code, 0)
        self.assertEqual(first["status"], "READY")
        self.assertEqual(second_code, 75)
        self.assertEqual(second["status"], "LOCKED")
        self.assertEqual(second["code"], "HOTPATH_ACTIVE_LEASE")
        self.assertEqual(sorted((self.state_root / "runs").iterdir()), run_directories)

    def test_output_remains_below_sixteen_kibibytes(self) -> None:
        prepared = PreparedRun(
            manifest_path=Path("/" + "m" * 20_000),
            candidate_path=Path("/" + "c" * 20_000),
            run_id="r" * 20_000,
            profile="intraday",
            kind="trade",
            ready=False,
            blockers=tuple("b" * 2_000 for _ in range(1_000)),
        )
        with patch("tools.ai_trader_hotpath.prepare_run", return_value=prepared):
            code, payload = run_hotpath(self.options(), now=NOW)

        self.assertNotEqual(code, 0)
        self.assertLess(len(_encode_payload(payload)), MAX_OUTPUT_BYTES)
        self.assertNotIn("sources", payload)


def _policy_payload() -> dict[str, object]:
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
        ],
        "hot_path": {
            "notion_access_allowed": False,
            "browser_access_allowed": False,
            "ordinary_network_destinations": ["market_data", "broker"],
            "legacy_strategy_authority": "BASELINE_ONLY",
            "manual_positions": "NO_TOUCH",
        },
    }


if __name__ == "__main__":
    unittest.main()

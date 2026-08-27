from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_causal_min_spread_representative_v26_recovery_once as one_shot


def authorization() -> dict:
    return {
        "cycle_id": "V26",
        "authorized": True,
        "scope": "ONE_TIMESTAMP_ONLY_PAPER_RECOVERY_ATTEMPT",
        "recovery_attempt_limit": 1,
        "one_shot_launcher_sha256": one_shot.sha256_file(Path(one_shot.__file__)),
        "authority": {
            "paper_only": True,
            "live_authority": False,
            "broker_account_access": False,
            "credential_access": False,
            "order_endpoint": False,
            "external_orders": 0,
            "deploy": False,
            "external_config_mutation": False,
        },
    }


class OneShotLauncherTest(unittest.TestCase):
    def test_requires_durable_authorized_intent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization_path = root / "authorization.json"
            state_path = root / "state.json"
            authorization_path.write_text(json.dumps(authorization()), encoding="utf-8")
            state_path.write_text(json.dumps({
                "cycles": {"V26": {
                    "status": "FAILED_OFFICIAL_EXECUTION_NO_RERUN",
                    "official_attempts": 1,
                    "recovery_attempts": 0,
                }},
            }), encoding="utf-8")
            with mock.patch.object(one_shot, "AUTHORIZATION", authorization_path), \
                    mock.patch.object(one_shot, "STATE", state_path), \
                    mock.patch.object(one_shot, "RESULT", root / "result.json"), \
                    mock.patch.object(one_shot, "LEDGER", root / "ledger.jsonl"):
                with self.assertRaisesRegex(RuntimeError, "not durably registered"):
                    one_shot.validate_one_shot_intent()

    def test_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization_path = root / "authorization.json"
            state_path = root / "state.json"
            result_path = root / "result.json"
            authorization_path.write_text(json.dumps(authorization()), encoding="utf-8")
            authorization_hash = one_shot.sha256_file(authorization_path)
            state_path.write_text(json.dumps({
                "cycles": {"V26": {
                    "status": "RECOVERY_ATTEMPT_STARTED",
                    "official_attempts": 1,
                    "recovery_attempts": 1,
                    "recovery_authorization_sha256": authorization_hash,
                }},
            }), encoding="utf-8")
            result_path.write_text("already exists", encoding="utf-8")
            with mock.patch.object(one_shot, "AUTHORIZATION", authorization_path), \
                    mock.patch.object(one_shot, "STATE", state_path), \
                    mock.patch.object(one_shot, "RESULT", result_path), \
                    mock.patch.object(one_shot, "LEDGER", root / "ledger.jsonl"):
                with self.assertRaisesRegex(RuntimeError, "already exist"):
                    one_shot.validate_one_shot_intent()


if __name__ == "__main__":
    unittest.main()

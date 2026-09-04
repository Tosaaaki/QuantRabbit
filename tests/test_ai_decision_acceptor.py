from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.ai_decision_acceptor import monitor_candidate
from quant_rabbit.ai_trading_runtime import AcceptedRun, prepare_run
from tools.ai_trader_hotpath import HotPathOptions, _launch_acceptor


NOW = datetime(2026, 9, 4, 4, 0, tzinfo=timezone.utc)
REPO_ROOT = Path(__file__).resolve().parents[1]


class AIDecisionAcceptorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = self.root / "state"
        data = self.root / "data"
        data.mkdir()
        self.evidence = data / "evidence.json"
        self.evidence.write_text(json.dumps(_evidence_packet()), encoding="utf-8")
        self.config = self.root / "runtime.json"
        self.config.write_text(json.dumps({
            "version": 2,
            "state_root": str(self.state),
            "profiles": {
                "intraday": {
                    "kind": "trade",
                    "decision_max_age_seconds": 900,
                    "candidate_accept_slo_seconds": 2,
                    "sink": "paper_ledger",
                    "allowed_actions": ["ENTER", "WAIT", "REQUEST_EVIDENCE"],
                    "workers": {
                        "evidence": [
                            {"path": "data/evidence.json", "required": True, "max_age_seconds": 10**9}
                        ]
                    },
                }
            },
        }), encoding="utf-8")
        self.prepared = prepare_run(
            config_path=self.config,
            profile="intraday",
            repo_root=self.root,
            now=NOW,
        )
        self.manifest = json.loads(self.prepared.manifest_path.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_accepts_completed_candidate_without_model_task_followup(self) -> None:
        candidate = json.loads(self.prepared.candidate_path.read_text(encoding="utf-8"))
        candidate.update({
            "model": "gpt-5.6-luna",
            "reasoning_effort": "max",
            "decided_at_utc": (NOW + timedelta(seconds=1)).isoformat(),
            "thesis": "No bounded entry has sufficient edge.",
            "evidence_refs": ["evidence:data/evidence.json"],
            "action": "WAIT",
            "confidence": 0.7,
            "orders": [],
            "position_actions": [],
            "requested_evidence": [],
        })
        self.prepared.candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

        outcome = self._monitor(now=NOW + timedelta(seconds=1, milliseconds=200))

        self.assertEqual(outcome["status"], "ACCEPTED")
        self.assertEqual(outcome["accepted_status"], "ACCEPTED_PAPER")
        self.assertEqual(outcome["attempt_count"], 1)
        self.assertTrue(outcome["slo_met"])
        self.assertTrue(Path(outcome["receipt_path"]).is_file())

    def test_template_expiry_is_an_explicit_terminal_outcome(self) -> None:
        outcome = self._monitor(now=NOW + timedelta(seconds=901))

        self.assertEqual(outcome["status"], "EXPIRED_NO_DECISION")
        self.assertEqual(outcome["code"], "CANDIDATE_NOT_AUTHORED_BEFORE_DEADLINE")
        self.assertFalse(outcome["broker_outcome_unknown"])
        self.assertFalse((self.prepared.manifest_path.parent / "receipt.json").exists())

    def test_stable_malformed_candidate_is_rejected_once(self) -> None:
        self.prepared.candidate_path.write_text("{", encoding="utf-8")

        outcome = self._monitor(now=NOW + timedelta(seconds=1))

        self.assertEqual(outcome["status"], "REJECTED")
        self.assertEqual(outcome["code"], "CANDIDATE_JSON_INVALID")
        self.assertEqual(outcome["attempt_count"], 0)
        self.assertFalse(outcome["broker_outcome_unknown"])

    def test_accept_uses_stable_snapshot_if_candidate_path_changes(self) -> None:
        candidate = json.loads(self.prepared.candidate_path.read_text(encoding="utf-8"))
        candidate.update({
            "model": "gpt-5.6-luna",
            "reasoning_effort": "max",
            "decided_at_utc": (NOW + timedelta(seconds=1)).isoformat(),
            "thesis": "Stable candidate snapshot.",
            "evidence_refs": ["evidence:data/evidence.json"],
            "action": "WAIT",
            "confidence": 0.7,
            "orders": [],
            "position_actions": [],
            "requested_evidence": [],
        })
        self.prepared.candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
        receipt_path = self.prepared.manifest_path.parent / "receipt.json"

        def accept_side_effect(**kwargs):
            self.prepared.candidate_path.write_text('{"changed":"after-detection"}', encoding="utf-8")
            self.assertEqual(kwargs["candidate_payload"], candidate)
            return AcceptedRun(
                receipt_path=receipt_path,
                run_id=self.prepared.run_id,
                profile="intraday",
                kind="trade",
                status="ACCEPTED_PAPER",
            )

        with patch("quant_rabbit.ai_decision_acceptor.accept_run", side_effect=accept_side_effect):
            outcome = self._monitor(now=NOW + timedelta(seconds=1, milliseconds=200))

        self.assertEqual(outcome["status"], "ACCEPTED")

    def test_real_detached_process_handshakes_and_consumes_candidate(self) -> None:
        live_clock = datetime.now(timezone.utc)
        self.evidence.write_text(json.dumps(_evidence_packet(live_clock)), encoding="utf-8")
        state = self.root / "detached-state"
        config = self.root / "detached-runtime.json"
        config.write_text(json.dumps({
            "version": 2,
            "state_root": str(state),
            "profiles": {
                "intraday": {
                    "kind": "trade",
                    "decision_max_age_seconds": 30,
                    "candidate_accept_slo_seconds": 2,
                    "sink": "paper_ledger",
                    "allowed_actions": ["ENTER", "WAIT", "REQUEST_EVIDENCE"],
                    "workers": {
                        "evidence": [
                            {"path": str(self.evidence), "required": True, "max_age_seconds": 10**9}
                        ]
                    },
                }
            },
        }), encoding="utf-8")
        prepared = prepare_run(
            config_path=config,
            profile="intraday",
            repo_root=REPO_ROOT,
            state_root=state,
            now=live_clock,
        )
        options = HotPathOptions(
            config_path=config,
            profile="intraday",
            repo_root=REPO_ROOT,
            state_root=state,
            policy_snapshot_path=self.root / "unused-policy.json",
            project_key="project.qr-trading",
            broker_account_id="paper-account",
            environment="practice",
            revocation_epoch=1,
            required_source_pages=("project-route",),
            lock_path=self.root / "unused.lock",
            capacity_filesystem=self.root,
            low_free_bytes=0,
            high_free_bytes=1,
            state_quota_pressure_bytes=10**7,
            state_quota_block_bytes=2 * 10**7,
            auto_accept=True,
            acceptor_poll_seconds=0.02,
        )

        handshake = _launch_acceptor(options, prepared=prepared, state_root=state)
        self.assertEqual(handshake["status"], "WAITING_FOR_CANDIDATE")
        manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
        candidate = json.loads(prepared.candidate_path.read_text(encoding="utf-8"))
        candidate.update({
            "model": "integration-test",
            "reasoning_effort": "low",
            "decided_at_utc": datetime.now(timezone.utc).isoformat(),
            "thesis": "Detached acceptor integration WAIT.",
            "evidence_refs": [f"evidence:{self.evidence}"],
            "action": "WAIT",
            "confidence": 0.5,
            "orders": [],
            "position_actions": [],
            "requested_evidence": [],
        })
        prepared.candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
        status_path = Path(manifest["acceptor_status_path"])
        deadline = time.monotonic() + 5
        status: dict[str, object] = {}
        while time.monotonic() < deadline:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") in {"ACCEPTED", "REJECTED", "FAILED"}:
                break
            time.sleep(0.02)

        self.assertEqual(status.get("status"), "ACCEPTED", status)
        self.assertEqual(status.get("attempt_count"), 1)
        self.assertTrue((prepared.manifest_path.parent / "receipt.json").is_file())

    def _monitor(self, *, now: datetime) -> dict[str, object]:
        return monitor_candidate(
            config_path=self.config,
            manifest_path=self.prepared.manifest_path,
            candidate_path=self.prepared.candidate_path,
            repo_root=self.root,
            state_root=self.state,
            initial_candidate_sha256=self.manifest["candidate_template_sha256"],
            poll_interval_seconds=0.02,
            now_fn=lambda: now,
        )


def _evidence_packet(now: datetime = NOW) -> dict[str, object]:
    body: dict[str, object] = {
        "contract": "QR_AI_EVIDENCE_PACKET_V1",
        "schema_version": 1,
        "status": "READY",
        "evidence_as_of_utc": now.isoformat(),
        "source_set_sha256": "sources-1",
        "broker_epoch": {"as_of_utc": now.isoformat(), "last_transaction_id": "9001"},
    }
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**body, "packet_sha256": hashlib.sha256(raw.encode()).hexdigest()}


if __name__ == "__main__":
    unittest.main()

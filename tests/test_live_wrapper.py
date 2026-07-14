from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.automation import _acquire_autotrade_lock
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS
from quant_rabbit.live_runtime_lock import (
    LiveLockAlreadyHeld,
    acquire_live_lock_owner,
    inspect_live_lock,
    live_lock_generation_guard,
    release_live_lock_owner,
    write_live_lock_owner,
)


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run-autotrade-live.sh"
GUARDIAN_WRAPPER = ROOT / "scripts" / "run-position-guardian-live.sh"


class LiveWrapperTest(unittest.TestCase):
    def test_unset_live_enabled_stays_dry_run_even_with_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture)
            env.pop("QR_LIVE_ENABLED", None)

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertIn("QR_LIVE_ENABLED=0\n", payload)
            self.assertIn("QR_AUTOTRADE_LOCK_HELD=1\n", payload)
            self.assertIn("<--send>", payload)
            self.assertIn("<--use-gpt-trader>", payload)
            self.assertIn("<--reuse-market-artifacts>", payload)
            self.assertIn("<-m><quant_rabbit.cli><cycle-sidecars>", payload)
            self.assertEqual((root / "sync.args").read_text(), "--live-only --skip-tests\n")

    def test_env_file_live_enabled_allows_live_send_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            env.pop("QR_LIVE_ENABLED", None)

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertIn("QR_LIVE_ENABLED=1\n", payload)
            self.assertIn("QR_POSITION_GUARDIAN_ACTIVE=1\n", payload)
            self.assertNotIn("forcing dry-run mode", result.stderr)
            self.assertEqual((root / "sync.args").read_text(), "--live-only --skip-tests\n")

    def test_live_send_marks_inactive_position_guardian_for_gateway_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1", guardian_loaded=False)

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertIn("QR_POSITION_GUARDIAN_ACTIVE=0\n", payload)
            self.assertIn("LiveOrderGateway will block fresh entry sends", result.stderr)

    def test_live_gpt_handoff_adds_missing_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")

            result = subprocess.run(
                [
                    "bash",
                    str(WRAPPER),
                    "--reuse-market-artifacts",
                    "--use-gpt-trader",
                    "--gpt-decision-response",
                    "data/codex_trader_decision_response.json",
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertIn("QR_LIVE_ENABLED=1\n", payload)
            self.assertIn("<--gpt-decision-response><data/codex_trader_decision_response.json>", payload)
            self.assertIn("<--send>", payload)
            self.assertIn("<-m><quant_rabbit.cli><cycle-sidecars>", payload)
            self.assertIn("adding --send to avoid a stage-only live trader cycle", result.stderr)

    def test_stage_only_live_gpt_handoff_requires_explicit_escape_hatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            env["QR_ALLOW_LIVE_STAGE_ONLY"] = "1"

            result = subprocess.run(
                [
                    "bash",
                    str(WRAPPER),
                    "--reuse-market-artifacts",
                    "--use-gpt-trader",
                    "--gpt-decision-response",
                    "data/codex_trader_decision_response.json",
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertNotIn("<--send>", payload)
            self.assertIn("<-m><quant_rabbit.cli><cycle-sidecars>", payload)
            self.assertIn("QR_ALLOW_LIVE_STAGE_ONLY=1; keeping GPT handoff stage-only", result.stderr)

    def test_unconsumed_verified_request_evidence_reaches_cycle_without_recomposition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            data = root / "data"
            data.mkdir()
            response = data / "codex_trader_decision_response.json"
            gpt_decision = data / "gpt_trader_decision.json"
            response.write_text('{"action":"REQUEST_EVIDENCE"}\n')
            gpt_decision.write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "decision": {"action": "REQUEST_EVIDENCE"},
                        "input_packet": {"active_path": {"status": "NO_TRADE_WITH_CAUSE"}},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            os.utime(response, (100.0, 100.0))
            os.utime(gpt_decision, (101.0, 101.0))

            result = subprocess.run(
                [
                    "bash",
                    str(WRAPPER),
                    "--reuse-market-artifacts",
                    "--use-gpt-trader",
                    "--gpt-decision-response",
                    "data/codex_trader_decision_response.json",
                    "--send",
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertEqual(_captured_cli_commands(payload)[:2], ["autotrade-cycle", "cycle-sidecars"])
            self.assertNotIn("trader-draft-decision", payload)
            self.assertNotIn("gpt-trader-decision", payload)
            self.assertIn(
                "preserving unconsumed ACCEPTED REQUEST_EVIDENCE for one gateway-maintenance cycle",
                result.stderr,
            )

    def test_consumed_verified_request_evidence_is_recomposed_before_next_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            data = root / "data"
            data.mkdir()
            docs = root / "docs"
            docs.mkdir()
            response = data / "codex_trader_decision_response.json"
            gpt_decision = data / "gpt_trader_decision.json"
            autotrade_report = docs / "autotrade_cycle_report.md"
            response.write_text('{"action":"REQUEST_EVIDENCE"}\n')
            gpt_decision.write_text(
                json.dumps(
                    {
                        "status": "ACCEPTED",
                        "decision": {"action": "REQUEST_EVIDENCE"},
                        "input_packet": {"active_path": {"status": "NO_TRADE_WITH_CAUSE"}},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            autotrade_report.write_text("# consumed\n")
            os.utime(response, (100.0, 100.0))
            os.utime(gpt_decision, (101.0, 101.0))
            os.utime(autotrade_report, (102.0, 102.0))

            result = subprocess.run(
                [
                    "bash",
                    str(WRAPPER),
                    "--reuse-market-artifacts",
                    "--use-gpt-trader",
                    "--gpt-decision-response",
                    "data/codex_trader_decision_response.json",
                    "--send",
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertEqual(
                _captured_cli_commands(payload)[:3],
                ["trader-draft-decision", "gpt-trader-decision", "autotrade-cycle"],
            )
            self.assertIn("already consumed as ACCEPTED REQUEST_EVIDENCE", result.stderr)

    def test_stale_codex_market_read_receipt_is_never_overwritten_by_auto_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            data = root / "data"
            data.mkdir()
            response = data / "codex_trader_decision_response.json"
            response.write_text(
                json.dumps(
                    {
                        "action": "WAIT",
                        "decision_provenance": {"author_kind": "CODEX_MARKET_READ"},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            broker_snapshot = data / "broker_snapshot.json"
            broker_snapshot.write_text("{}\n")
            os.utime(response, (100.0, 100.0))
            os.utime(broker_snapshot, (200.0, 200.0))

            result = subprocess.run(
                [
                    "bash",
                    str(WRAPPER),
                    "--reuse-market-artifacts",
                    "--use-gpt-trader",
                    "--gpt-decision-response",
                    "data/codex_trader_decision_response.json",
                    "--send",
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertEqual(_captured_cli_commands(payload)[:2], ["autotrade-cycle", "cycle-sidecars"])
            self.assertNotIn("trader-draft-decision", payload)
            self.assertNotIn("gpt-trader-decision", payload)
            self.assertIn("preserving Codex-authored market-read receipt", result.stderr)

    def test_successful_cycle_refreshes_post_gateway_sidecars_under_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = capture.read_text()
            self.assertIn("<-m><quant_rabbit.cli><autotrade-cycle>", payload)
            self.assertIn("<-m><quant_rabbit.cli><cycle-sidecars>", payload)
            self.assertIn("refreshing post-gateway sidecars under live lock", result.stderr)

    def test_failed_cycle_runs_canonical_failure_sidecar_command_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1", python_exit=37)

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 37)
            payload = capture.read_text()
            self.assertIn("<-m><quant_rabbit.cli><autotrade-cycle>", payload)
            self.assertNotIn("<-m><quant_rabbit.cli><cycle-sidecars>", payload)
            self.assertIn("<-m><quant_rabbit.cli><post-autotrade-failure-sidecars>", payload)
            self.assertEqual(
                _captured_cli_commands(payload),
                [
                    "trader-draft-decision",
                    "gpt-trader-decision",
                    "autotrade-cycle",
                    "post-autotrade-failure-sidecars",
                ],
            )
            self.assertIn("refreshing failure-repair sidecars under live lock", result.stderr)

    def test_sync_failure_continues_when_runtime_is_current_with_report_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1", sync_exit=37)
            _init_git(root)
            (root / "docs").mkdir(exist_ok=True)
            (root / "data").mkdir(exist_ok=True)
            (root / "docs" / "cycle_report.md").write_text("tracked report\n")
            (root / "docs" / "guardian_action_review.md").write_text("tracked review\n")
            (root / "data" / "guardian_trigger_contract.json").write_text('{"generated_at_utc":"tracked"}\n')
            (root / "data" / "guardian_receipt_consumption.json").write_text('{"status":"tracked"}\n')
            (root / "data" / "guardian_receipt_operator_review.json").write_text('{"status":"tracked"}\n')
            (root / "data" / "as_proof_pack_queue.json").write_text('{"generated_at":"tracked"}\n')
            (root / "docs" / "as_proof_pack_queue.md").write_text("tracked proof report\n")
            (root / "data" / "active_trader_contract.json").write_text('{"status":"tracked"}\n')
            (root / "docs" / "active_trader_contract.md").write_text("tracked active contract\n")
            _run(["git", "add", "."], cwd=root)
            _run(["git", "commit", "-m", "initial"], cwd=root)
            _run(["git", "branch", "-m", "main"], cwd=root)
            (root / "docs" / "cycle_report.md").write_text("runtime drift\n")
            (root / "docs" / "guardian_action_review.md").write_text("runtime review\n")
            (root / "data" / "guardian_trigger_contract.json").write_text('{"generated_at_utc":"runtime"}\n')
            (root / "data" / "guardian_receipt_consumption.json").write_text('{"status":"runtime"}\n')
            (root / "data" / "guardian_receipt_operator_review.json").write_text('{"status":"runtime"}\n')
            (root / "data" / "as_proof_pack_queue.json").write_text('{"generated_at":"runtime"}\n')
            (root / "docs" / "as_proof_pack_queue.md").write_text("runtime proof report\n")
            (root / "data" / "active_trader_contract.json").write_text('{"status":"runtime"}\n')
            (root / "docs" / "active_trader_contract.md").write_text("runtime active contract\n")
            env["QR_SYNC_DEV_ROOT"] = str(root)
            env["QR_SYNC_MAIN_BRANCH"] = "main"
            env["QR_SYNC_MARKER_PATH"] = str(root / "docs" / "sync_report.md")

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(capture.exists())
            self.assertIn("live sync failed with status=37", result.stderr)
            self.assertIn("proof-evidence drift is present", result.stderr)

    def test_empty_verdict_marker_is_removed_before_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            _init_git(root)
            (root / "EXTEND").write_text("")
            _run(["git", "add", "."], cwd=root)
            _run(["git", "commit", "-m", "initial"], cwd=root)
            _run(["git", "branch", "-m", "main"], cwd=root)
            _run(["git", "rm", "--cached", "EXTEND"], cwd=root)
            _run(["git", "commit", "-m", "untrack marker"], cwd=root)
            (root / "EXTEND").write_text("")
            env["QR_SYNC_DEV_ROOT"] = str(root)
            env["QR_SYNC_MAIN_BRANCH"] = "main"

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "EXTEND").exists())
            self.assertIn("removed empty verdict marker: EXTEND", result.stderr)

    def test_existing_live_lock_blocks_overlapping_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture)
            lock_dir = root / "lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()) + "\n")

            result = subprocess.run(
                ["bash", str(WRAPPER), "--send"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 75)
            self.assertFalse(capture.exists())
            self.assertIn("another autotrade cycle is already running", result.stderr)

    def test_live_send_waits_for_position_guardian_lock_then_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            env["QR_RUN_POST_GATEWAY_SIDECARS"] = "0"
            env["QR_AUTOTRADE_LOCK_WAIT_SECONDS"] = "5"
            env["QR_AUTOTRADE_LOCK_POLL_SECONDS"] = "0.1"
            # Deliberately use a generic OS command: only the persisted owner
            # label identifies this holder as the position guardian.
            proc = subprocess.Popen(["sleep", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                lock_dir = root / "lock"
                lock_dir.mkdir()
                (lock_dir / "pid").write_text(f"{proc.pid}\n")
                # The lock's persisted owner label is authoritative even when
                # `ps` contains no guardian script name.
                (lock_dir / "command").write_text("run-position-guardian-live\n")

                result = subprocess.run(
                    ["bash", str(WRAPPER), "--send"],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            finally:
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait(timeout=2)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(capture.exists())
            self.assertIn("waiting up to 5s", result.stderr)
            # Depending on when the parent observes exit, the holder is either
            # already absent or briefly a zombie. Both are safely reclaimable.
            self.assertRegex(
                result.stderr,
                r"removing (?:defunct lock holder pid=\d+:|stale lock:)",
            )

    def test_live_lock_removes_recycled_pid_owner_before_acquire(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_dir = root / "lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(f"{os.getpid()}\n")
            (lock_dir / "command").write_text("run-autotrade-live\n")
            (lock_dir / "process_started_at").write_text("Mon Jan  1 00:00:00 2001\n")
            env = os.environ.copy()
            env.update(
                {
                    "QR_HELPER": str(ROOT / "scripts" / "qr-live-lock.sh"),
                    "QR_LOCK_DIR": str(lock_dir),
                }
            )

            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        "set -euo pipefail; "
                        "source \"$QR_HELPER\"; "
                        "qr_live_lock_acquire \"$QR_LOCK_DIR\" test-lock 0 '' 0.1; "
                        "qr_live_lock_release"
                    ),
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(lock_dir.exists())
            self.assertIn("removing recycled-pid lock holder", result.stderr)

    def test_live_lock_malformed_birth_metadata_stays_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_dir = root / "lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(f"{os.getpid()}\n")
            (lock_dir / "command").write_text("run-autotrade-live\n")
            (lock_dir / "process_started_at").write_text("malformed-owner-evidence\n")
            env = os.environ.copy()
            env.update(
                {
                    "QR_HELPER": str(ROOT / "scripts" / "qr-live-lock.sh"),
                    "QR_LOCK_DIR": str(lock_dir),
                }
            )

            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        "set -euo pipefail; "
                        "source \"$QR_HELPER\"; "
                        "qr_live_lock_acquire \"$QR_LOCK_DIR\" test-lock 0 '' 0.1"
                    ),
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertTrue(lock_dir.exists())
            self.assertIn("another autotrade cycle is already running", result.stderr)

    def test_live_lock_multiline_or_nul_birth_metadata_stays_fail_closed(self) -> None:
        current_birth = subprocess.run(
            ["ps", "-p", str(os.getpid()), "-o", "lstart="],
            env={**os.environ, "LC_ALL": "C"},
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout.strip()
        payloads = {
            "multiple-lines": f"{current_birth}\njunk\n".encode(),
            "nul-byte": current_birth.encode() + b"\x00\n",
        }
        for label, payload in payloads.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lock_dir = root / "lock"
                lock_dir.mkdir()
                (lock_dir / "pid").write_text(f"{os.getpid()}\n")
                (lock_dir / "command").write_text("run-autotrade-live\n")
                (lock_dir / "process_started_at").write_bytes(payload)
                env = os.environ.copy()
                env.update(
                    {
                        "QR_HELPER": str(ROOT / "scripts" / "qr-live-lock.sh"),
                        "QR_LOCK_DIR": str(lock_dir),
                    }
                )

                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        (
                            "set -euo pipefail; "
                            "source \"$QR_HELPER\"; "
                            "qr_live_lock_acquire \"$QR_LOCK_DIR\" test-lock 0 '' 0.1"
                        ),
                    ],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )

                self.assertEqual(result.returncode, 75, result.stderr)
                self.assertTrue(lock_dir.exists())

    def test_live_lock_preserves_owner_during_pid_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_dir = root / "lock"
            lock_dir.mkdir()
            holder = root / "delayed-lock-owner.sh"
            ready = root / "delayed-lock-owner.ready"
            release = root / "delayed-lock-owner.release"
            contender_stderr = root / "delayed-lock-contender.stderr"
            holder.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'ready\\n' > \"$QR_READY\"\n"
                "while [[ ! -f \"$QR_RELEASE\" ]]; do sleep 0.005; done\n"
                "printf '%s\\n' \"$$\" > \"$QR_LOCK_DIR/pid\"\n"
                "printf '%s\\n' run-position-guardian-live > \"$QR_LOCK_DIR/command\"\n"
                "sleep 0.7\n"
            )
            holder.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "QR_HELPER": str(ROOT / "scripts" / "qr-live-lock.sh"),
                    "QR_LOCK_DIR": str(lock_dir),
                    "QR_READY": str(ready),
                    "QR_RELEASE": str(release),
                    "QR_LIVE_LOCK_INIT_GRACE_SECONDS": "0.5",
                }
            )
            proc = subprocess.Popen(
                [str(holder)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                ready_deadline = time.monotonic() + 1.0
                while not ready.exists() and time.monotonic() < ready_deadline:
                    time.sleep(0.005)
                self.assertTrue(ready.exists(), "delayed lock holder did not start")
                with contender_stderr.open("w") as stderr_handle:
                    contender = subprocess.Popen(
                        [
                            "bash",
                            "-c",
                            (
                                "set -euo pipefail; "
                                "source \"$QR_HELPER\"; "
                                "qr_live_lock_acquire \"$QR_LOCK_DIR\" test-lock 2 "
                                "run-position-guardian-live 0.05; "
                                "qr_live_lock_release"
                            ),
                        ],
                        env=env,
                        text=True,
                        stdout=subprocess.DEVNULL,
                        stderr=stderr_handle,
                    )
                    observed_deadline = time.monotonic() + 2.0
                    while time.monotonic() < observed_deadline:
                        stderr_handle.flush()
                        if (
                            contender_stderr.exists()
                            and "lock owner metadata is initializing"
                            in contender_stderr.read_text()
                        ):
                            break
                        time.sleep(0.005)
                    else:
                        contender.terminate()
                        contender.wait(timeout=2)
                        self.fail("contender did not observe initializing owner metadata")
                    release.write_text("release\n")
                    contender.wait(timeout=4)
            finally:
                release.write_text("release\n")
                proc.wait(timeout=2)

            stderr = contender_stderr.read_text()
            self.assertEqual(contender.returncode, 0, stderr)
            self.assertFalse(lock_dir.exists())
            self.assertIn("lock owner metadata is initializing", stderr)
            self.assertIn("waiting up to 2s", stderr)

    def test_position_guardian_yields_to_initializing_full_trader_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            (root / "lock").mkdir()

            result = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(capture.exists())
            self.assertIn("owner metadata is initializing", result.stderr)
            self.assertIn("skipped guardian cycle", result.stderr)

    def test_two_contenders_cannot_both_reap_and_acquire_stale_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_dir = root / "lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text("99999999\n")
            results = _run_competing_shell_lockers(root, delay_owner_write=False)

            self.assertEqual(sorted(result.returncode for result in results), [0, 75])
            self.assertEqual(len((root / "winners").read_text().splitlines()), 1)

    def test_two_contenders_cannot_delete_initializing_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = _run_competing_shell_lockers(root, delay_owner_write=True)

            self.assertEqual(sorted(result.returncode for result in results), [0, 75])
            self.assertEqual(len((root / "winners").read_text().splitlines()), 1)

    def test_shell_and_python_share_generation_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_dir = Path(tmp) / "lock"
            env = os.environ.copy()
            env.update(
                {
                    "QR_HELPER": str(ROOT / "scripts" / "qr-live-lock.sh"),
                    "QR_LOCK_DIR": str(lock_dir),
                }
            )
            with live_lock_generation_guard(lock_dir):
                proc = subprocess.Popen(
                    [
                        "bash",
                        "-c",
                        (
                            "set -euo pipefail; "
                            "source \"$QR_HELPER\"; "
                            "qr_live_lock_guard_acquire \"$QR_LOCK_DIR\"; "
                            "printf acquired; "
                            "qr_live_lock_guard_release"
                        ),
                    ],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                time.sleep(0.2)
                self.assertIsNone(proc.poll(), "shell bypassed Python's generation guard")

            stdout, stderr = proc.communicate(timeout=2)
            self.assertEqual(proc.returncode, 0, stderr)
            self.assertEqual(stdout, "acquired")

    def test_shell_contender_preserves_active_python_owner_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_dir = Path(tmp) / "lock"
            token = acquire_live_lock_owner(lock_dir, "python-owner")
            env = os.environ.copy()
            env.update(
                {
                    "QR_HELPER": str(ROOT / "scripts" / "qr-live-lock.sh"),
                    "QR_LOCK_DIR": str(lock_dir),
                }
            )
            try:
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        (
                            "set -euo pipefail; "
                            "source \"$QR_HELPER\"; "
                            "qr_live_lock_acquire \"$QR_LOCK_DIR\" shell-contender 0 '' 0.1"
                        ),
                    ],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )

                self.assertEqual(result.returncode, 75, result.stderr)
                self.assertEqual((lock_dir / "token").read_text().strip(), token)
            finally:
                self.assertTrue(release_live_lock_owner(lock_dir, token))

    def test_python_contender_preserves_active_shell_owner_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_dir = root / "lock"
            ready = root / "ready"
            env = os.environ.copy()
            env.update(
                {
                    "QR_HELPER": str(ROOT / "scripts" / "qr-live-lock.sh"),
                    "QR_LOCK_DIR": str(lock_dir),
                    "QR_READY_FILE": str(ready),
                }
            )
            proc = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    (
                        "set -euo pipefail; "
                        "source \"$QR_HELPER\"; "
                        "qr_live_lock_acquire \"$QR_LOCK_DIR\" shell-owner 0 '' 0.1; "
                        "printf ready > \"$QR_READY_FILE\"; "
                        "sleep 0.7; "
                        "qr_live_lock_release"
                    ),
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 2
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists(), "shell owner did not acquire lock")
            try:
                with self.assertRaises(LiveLockAlreadyHeld):
                    acquire_live_lock_owner(
                        lock_dir,
                        "python-contender",
                        init_grace_seconds=0,
                    )
            finally:
                stdout, stderr = proc.communicate(timeout=2)
            self.assertEqual(proc.returncode, 0, f"{stdout}\n{stderr}")

    def test_python_release_preserves_reacquired_generation_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_dir = Path(tmp) / "lock"
            with live_lock_generation_guard(lock_dir):
                lock_dir.mkdir()
                old_token = write_live_lock_owner(lock_dir, "old-owner")
                (lock_dir / "token").write_text("new-owner-token\n")

            self.assertFalse(release_live_lock_owner(lock_dir, old_token))
            self.assertTrue(lock_dir.exists())
            self.assertEqual((lock_dir / "token").read_text(), "new-owner-token\n")

    def test_python_acquire_reaps_zombie_and_recycled_pid_generations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zombie_lock = root / "zombie-lock"
            proc = subprocess.Popen(["sleep", "0.05"])
            time.sleep(0.15)
            try:
                zombie_lock.mkdir()
                (zombie_lock / "pid").write_text(f"{proc.pid}\n")
                zombie_token = acquire_live_lock_owner(
                    zombie_lock,
                    "python-zombie-reaper",
                    init_grace_seconds=0,
                )
                self.assertEqual(inspect_live_lock(zombie_lock).status, "ACTIVE")
                self.assertTrue(release_live_lock_owner(zombie_lock, zombie_token))
            finally:
                proc.wait(timeout=2)

            recycled_lock = root / "recycled-lock"
            recycled_lock.mkdir()
            (recycled_lock / "pid").write_text(f"{os.getpid()}\n")
            (recycled_lock / "process_started_at").write_text(
                "Mon Jan  1 00:00:00 2001\n"
            )
            recycled_token = acquire_live_lock_owner(
                recycled_lock,
                "python-recycled-reaper",
                init_grace_seconds=0,
            )
            self.assertEqual(inspect_live_lock(recycled_lock).status, "ACTIVE")
            self.assertTrue(release_live_lock_owner(recycled_lock, recycled_token))

    def test_python_acquire_keeps_malformed_active_birth_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_dir = Path(tmp) / "lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(f"{os.getpid()}\n")
            (lock_dir / "process_started_at").write_text(
                "Tue Jul 14 00:00:00 2026\njunk\n"
            )

            with self.assertRaises(LiveLockAlreadyHeld):
                acquire_live_lock_owner(lock_dir, "python-contender", init_grace_seconds=0)
            self.assertTrue(lock_dir.exists())
            self.assertEqual(inspect_live_lock(lock_dir).status, "ACTIVE_IDENTITY_UNAVAILABLE")

    def test_python_identity_normalizes_internal_process_birth_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_dir = Path(tmp) / "lock"
            current_birth = subprocess.run(
                ["ps", "-p", str(os.getpid()), "-o", "lstart="],
                env={**os.environ, "LC_ALL": "C"},
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            ).stdout.strip()
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(f"{os.getpid()}\n")
            (lock_dir / "process_started_at").write_text(
                "  " + current_birth.replace(" ", "   ") + "  \n"
            )

            inspection = inspect_live_lock(lock_dir)
            self.assertTrue(inspection.active)
            self.assertEqual(inspection.status, "ACTIVE")

    def test_python_acquire_preserves_delayed_owner_during_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_dir = root / "lock"
            lock_dir.mkdir()
            ready = root / "holder-ready"
            holder = root / "delayed-python-owner.sh"
            holder.write_text(
                "#!/usr/bin/env bash\n"
                "printf ready > \"$QR_READY_FILE\"\n"
                "sleep 0.05\n"
                "printf '%s\\n' \"$$\" > \"$QR_LOCK_DIR/pid\"\n"
                "sleep 0.7\n"
            )
            holder.chmod(0o755)
            proc = subprocess.Popen(
                [str(holder)],
                env={
                    **os.environ,
                    "QR_LOCK_DIR": str(lock_dir),
                    "QR_READY_FILE": str(ready),
                },
            )
            try:
                deadline = time.monotonic() + 2
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists(), "delayed owner did not start")
                with self.assertRaises(LiveLockAlreadyHeld):
                    acquire_live_lock_owner(
                        lock_dir,
                        "python-initialization-contender",
                        init_grace_seconds=0.3,
                    )
                self.assertEqual(inspect_live_lock(lock_dir).pid, proc.pid)
            finally:
                proc.wait(timeout=2)

    def test_position_guardian_skips_when_full_trader_lock_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.json"
            env = _wrapper_env(root, capture, live_enabled="1")
            lock_dir = root / "lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()) + "\n")
            (lock_dir / "command").write_text("run-autotrade-live\n")

            result = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(capture.exists())
            self.assertIn("live runtime lock busy", result.stderr)
            self.assertIn("skipped guardian cycle", result.stderr)

    def test_position_guardian_monitors_every_open_owner_and_bounds_candidate_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = {
                "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                "positions": [
                    {"trade_id": "t1", "pair": "EUR_USD", "owner": "trader"},
                    {"trade_id": "m1", "pair": "USD_JPY", "owner": "operator_manual"},
                    {"trade_id": "u1", "pair": "GBP_USD", "owner": "unknown"},
                ],
                "orders": [
                    {"order_id": "o1", "pair": "AUD_CAD", "owner": "trader", "state": "PENDING"},
                    {"order_id": "o2", "pair": "USD_JPY", "owner": "operator_manual", "state": "PENDING"},
                ],
                "quotes": {},
            }
            triggers = {
                "entries": [
                    {"pair": "AUD_JPY", "lane_id": "lane-a", "status": "DRY_RUN_BLOCKED"},
                    {"pair": "NZD_USD", "lane_id": "lane-b", "status": "DRY_RUN_BLOCKED"},
                    {"pair": "CAD_JPY", "lane_id": "lane-c", "status": "DRY_RUN_BLOCKED"},
                ]
            }
            env, capture = _guardian_wrapper_env(
                root,
                snapshot=snapshot,
                trigger_contract=triggers,
                candidate_limit=2,
            )

            result = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            pair_call = next(call for call in calls if "pair-charts" in call)
            self.assertEqual(
                pair_call[pair_call.index("--pairs") + 1],
                "EUR_USD,GBP_USD,USD_JPY,AUD_CAD,AUD_USD",
            )
            self.assertEqual(pair_call[pair_call.index("--timeframes") + 1], "M1,M5,M15")
            self.assertNotIn("M30", pair_call)
            management_call = next(call for call in calls if "position-management" in call)
            execution_call = next(call for call in calls if "position-execution" in call)
            self.assertEqual(
                management_call[management_call.index("--snapshot") + 1],
                "data/position_guardian_trader_snapshot.json",
            )
            self.assertEqual(
                execution_call[execution_call.index("--snapshot") + 1],
                "data/position_guardian_trader_snapshot.json",
            )
            management_snapshot = json.loads((root / "data" / "position_guardian_trader_snapshot.json").read_text())
            self.assertEqual(
                [(item["trade_id"], item["owner"]) for item in management_snapshot["positions"]],
                [("t1", "trader")],
            )
            self.assertEqual(
                [(item["order_id"], item["owner"]) for item in management_snapshot["orders"]],
                [("o1", "trader")],
            )
            self.assertTrue(management_snapshot["position_guardian_scope"]["non_trader_positions_monitor_only"])
            self.assertTrue(management_snapshot["position_guardian_scope"]["non_trader_orders_monitor_only"])
            charts = json.loads((root / "data" / "position_guardian_pair_charts.json").read_text())
            self.assertEqual(
                charts["guardian_monitor_pairs"],
                ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_CAD", "AUD_USD"],
            )
            scope = charts["guardian_monitor_scope"]
            self.assertEqual(scope["USD_JPY"]["position_owners"], ["operator_manual"])
            self.assertTrue(scope["USD_JPY"]["non_trader_monitor_only"])
            self.assertFalse(scope["USD_JPY"]["management_write_eligible"])
            self.assertIn("active_pending_order", scope["AUD_CAD"]["reasons"])
            freshness = json.loads((root / "data" / "position_guardian_chart_freshness.json").read_text())
            self.assertEqual(freshness["hard_priority_candidate_pairs"], ["AUD_CAD"])
            self.assertNotIn("AUD_JPY", freshness["monitor_pairs"])
            self.assertEqual(freshness["status"], "FRESH")
            self.assertEqual(freshness["timeframes"], ["M1", "M5", "M15"])
            self.assertIn("closed-candle chart refresh FRESH", result.stderr)

    def test_position_guardian_monitor_only_continues_and_reuses_chart_until_next_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = {
                "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                "positions": [
                    {"trade_id": "m1", "pair": "USD_JPY", "owner": "operator_manual"},
                    {"trade_id": "u1", "pair": "GBP_USD", "owner": "unknown"},
                ],
                "orders": [],
                "quotes": {},
            }
            triggers = {"entries": [{"pair": "EUR_USD", "lane_id": "lane-a", "status": "TRIGGER_READY"}]}
            env, capture = _guardian_wrapper_env(
                root,
                snapshot=snapshot,
                trigger_contract=triggers,
                candidate_limit=1,
            )

            results = [
                subprocess.run(
                    ["bash", str(GUARDIAN_WRAPPER)],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                for _ in range(2)
            ]

            for result in results:
                self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            self.assertEqual(sum("pair-charts" in call for call in calls), 1)
            self.assertEqual(sum("guardian-event-router" in call for call in calls), 2)
            self.assertFalse(any("position-management" in call for call in calls))
            self.assertFalse(any("position-execution" in call for call in calls))
            pair_call = next(call for call in calls if "pair-charts" in call)
            self.assertEqual(pair_call[pair_call.index("--pairs") + 1], "GBP_USD,USD_JPY,EUR_USD")
            management = json.loads((root / "data" / "position_guardian_management.json").read_text())
            self.assertEqual(management["action"], "MONITOR_ONLY_NO_TRADER_POSITION")
            self.assertEqual(management["positions"], [])
            heartbeat = json.loads((root / "data" / "position_guardian.json").read_text())
            self.assertEqual(heartbeat["status"], "MONITOR_ONLY_NO_TRADER_POSITION")
            self.assertEqual(heartbeat["monitor_pairs"], ["GBP_USD", "USD_JPY", "EUR_USD"])
            self.assertIn("completed read-only monitor scope", results[-1].stderr)
            self.assertIn("reused data/position_guardian_pair_charts.json", results[-1].stderr)

    def test_position_guardian_success_cursor_covers_configured_g8_without_raising_pair_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quote_pairs = [
                "AUD_CAD",
                "AUD_CHF",
                "AUD_JPY",
                "AUD_NZD",
                "CAD_CHF",
                "CAD_JPY",
                "CHF_JPY",
                "EUR_AUD",
            ]
            env, capture = _guardian_wrapper_env(
                root,
                snapshot={
                    "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                    "positions": [],
                    "orders": [],
                    "quotes": {
                        **{pair: {"bid": 1.0, "ask": 1.0001} for pair in quote_pairs},
                        "XAU_USD": {"bid": 2000.0, "ask": 2000.1},
                    },
                },
                trigger_contract={"entries": []},
                candidate_limit=2,
            )

            cycle_count = (len(DEFAULT_TRADER_PAIRS) + 1) // 2
            for cycle in range(cycle_count):
                result = subprocess.run(
                    ["bash", str(GUARDIAN_WRAPPER)],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                if cycle + 1 < cycle_count:
                    freshness_path = root / "data" / "position_guardian_chart_freshness.json"
                    freshness = json.loads(freshness_path.read_text())
                    freshness["next_refresh_after_utc"] = "2000-01-01T00:00:00+00:00"
                    freshness_path.write_text(json.dumps(freshness))

            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            pair_calls = [call for call in calls if "pair-charts" in call]
            self.assertEqual(len(pair_calls), cycle_count)
            observed: set[str] = set()
            for call in pair_calls:
                selected = call[call.index("--pairs") + 1].split(",")
                self.assertEqual(len(selected), 2)
                observed.update(selected)
            self.assertEqual(observed, set(DEFAULT_TRADER_PAIRS))

            freshness = json.loads((root / "data" / "position_guardian_chart_freshness.json").read_text())
            self.assertEqual(freshness["coverage_policy"], "HARD_PRIORITY_THEN_HOURLY_FRONTIER_THEN_G8_ROUND_ROBIN")
            self.assertEqual(freshness["coverage_cursor"], 0)
            self.assertEqual(
                freshness["coverage_cursor_semantics"],
                "NEXT_CONFIGURED_PAIR_INDEX",
            )
            self.assertEqual(
                freshness["coverage_cursor_advance_policy"],
                "AFTER_SUCCESSFUL_CLOSED_M1_REFRESH_ONLY",
            )
            self.assertEqual(freshness["rotation_period_seconds"], 60)

    def test_position_guardian_scope_change_waits_for_next_closed_m1_grace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, capture = _guardian_wrapper_env(
                root,
                snapshot={
                    "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                    "positions": [],
                    "orders": [],
                    "quotes": {
                        "EUR_USD": {"bid": 1.0, "ask": 1.0001},
                        "USD_JPY": {"bid": 150.0, "ask": 150.01},
                    },
                },
                trigger_contract={"entries": []},
                candidate_limit=1,
            )
            first = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            initial_freshness = json.loads(
                (root / "data" / "position_guardian_chart_freshness.json").read_text()
            )
            self.assertEqual(initial_freshness["coverage_cursor"], 1)

            (root / "data" / "active_opportunity_board.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                        "top_lane": {"pair": "USD_JPY", "lane_id": "new-hourly-lane"},
                    }
                )
            )
            env["QR_POSITION_GUARDIAN_ROTATION_CURSOR"] = "5"
            second = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(second.returncode, 0, second.stderr)

            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            self.assertEqual(sum("pair-charts" in call for call in calls), 1)
            persisted = json.loads(
                (root / "data" / "position_guardian_chart_freshness.json").read_text()
            )
            self.assertEqual(persisted["coverage_cursor"], 1)
            self.assertEqual(
                persisted["next_refresh_after_utc"],
                initial_freshness["next_refresh_after_utc"],
            )
            charts = json.loads(
                (root / "data" / "position_guardian_pair_charts.json").read_text()
            )
            management = json.loads(
                (root / "data" / "position_guardian_management.json").read_text()
            )
            heartbeat = json.loads((root / "data" / "position_guardian.json").read_text())
            self.assertEqual(
                management["monitor_scope"]["candidate_pairs"],
                persisted["candidate_pairs"],
            )
            self.assertEqual(
                management["monitor_scope"]["monitor_pairs"],
                persisted["monitor_pairs"],
            )
            self.assertEqual(heartbeat["monitor_pairs"], persisted["monitor_pairs"])
            self.assertEqual(
                charts["guardian_monitor_pairs"],
                persisted["monitor_pairs"],
            )

    def test_position_guardian_rotation_override_is_bootstrap_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, capture = _guardian_wrapper_env(
                root,
                snapshot={
                    "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                    "positions": [],
                    "orders": [],
                    "quotes": {
                        pair: {"bid": 1.0, "ask": 1.0001}
                        for pair in DEFAULT_TRADER_PAIRS
                    },
                },
                trigger_contract={"entries": []},
                candidate_limit=2,
            )
            env["QR_POSITION_GUARDIAN_ROTATION_CURSOR"] = "5"

            first = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            freshness_path = root / "data" / "position_guardian_chart_freshness.json"
            first_freshness = json.loads(freshness_path.read_text())
            self.assertEqual(first_freshness["coverage_cursor"], 7)
            self.assertEqual(
                first_freshness["coverage_cursor_source"],
                "BOOTSTRAP_EXPLICIT_OVERRIDE",
            )
            first_freshness["next_refresh_after_utc"] = "2000-01-01T00:00:00+00:00"
            freshness_path.write_text(json.dumps(first_freshness))

            second = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(second.returncode, 0, second.stderr)

            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            pair_calls = [call for call in calls if "pair-charts" in call]
            self.assertEqual(len(pair_calls), 2)
            self.assertEqual(
                pair_calls[0][pair_calls[0].index("--pairs") + 1].split(","),
                list(DEFAULT_TRADER_PAIRS[5:7]),
            )
            self.assertEqual(
                pair_calls[1][pair_calls[1].index("--pairs") + 1].split(","),
                list(DEFAULT_TRADER_PAIRS[7:9]),
            )
            final_freshness = json.loads(freshness_path.read_text())
            self.assertEqual(final_freshness["coverage_cursor"], 9)
            self.assertEqual(
                final_freshness["coverage_cursor_source"],
                "NEXT_UNCOVERED_PAIR_AFTER_CLOSED_M1",
            )

    def test_position_guardian_legacy_scope_without_active_cursor_stays_frozen_until_grace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, capture = _guardian_wrapper_env(
                root,
                snapshot={
                    "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                    "positions": [],
                    "orders": [],
                    "quotes": {
                        "EUR_USD": {"bid": 1.0, "ask": 1.0001},
                        "USD_JPY": {"bid": 150.0, "ask": 150.01},
                    },
                },
                trigger_contract={"entries": []},
                candidate_limit=1,
            )
            first = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)

            freshness_path = root / "data" / "position_guardian_chart_freshness.json"
            legacy = json.loads(freshness_path.read_text())
            initial_candidates = list(legacy["candidate_pairs"])
            initial_monitor = list(legacy["monitor_pairs"])
            for key in (
                "coverage_cursor_semantics",
                "active_scope_cursor",
                "proposed_coverage_next_cursor",
                "proposed_coverage_scanned_count",
            ):
                legacy.pop(key, None)
            freshness_path.write_text(json.dumps(legacy))
            (root / "data" / "active_opportunity_board.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                        "top_lane": {"pair": "USD_JPY", "lane_id": "legacy-board-churn"},
                    }
                )
            )
            env["QR_POSITION_GUARDIAN_ROTATION_CURSOR"] = "5"

            second = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(second.returncode, 0, second.stderr)
            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            self.assertEqual(sum("pair-charts" in call for call in calls), 1)
            persisted = json.loads(freshness_path.read_text())
            self.assertEqual(persisted["candidate_pairs"], initial_candidates)
            self.assertEqual(persisted["monitor_pairs"], initial_monitor)
            management = json.loads(
                (root / "data" / "position_guardian_management.json").read_text()
            )
            heartbeat = json.loads((root / "data" / "position_guardian.json").read_text())
            charts = json.loads(
                (root / "data" / "position_guardian_pair_charts.json").read_text()
            )
            self.assertEqual(
                management["monitor_scope"]["candidate_pairs"],
                initial_candidates,
            )
            self.assertEqual(
                management["monitor_scope"]["monitor_pairs"],
                initial_monitor,
            )
            self.assertEqual(heartbeat["monitor_pairs"], initial_monitor)
            self.assertEqual(charts["guardian_monitor_pairs"], initial_monitor)

    def test_position_guardian_retries_partial_or_stale_scope_before_advancing_cursor(self) -> None:
        for failure_mode in ("PARTIAL", "STALE"):
            with self.subTest(failure_mode=failure_mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                env, capture = _guardian_wrapper_env(
                    root,
                    snapshot={
                        "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                        "positions": [],
                        "orders": [],
                        "quotes": {
                            pair: {"bid": 1.0, "ask": 1.0001}
                            for pair in DEFAULT_TRADER_PAIRS
                        },
                    },
                    trigger_contract={"entries": []},
                    candidate_limit=2,
                )

                initial = subprocess.run(
                    ["bash", str(GUARDIAN_WRAPPER)],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(initial.returncode, 0, initial.stderr)
                freshness_path = root / "data" / "position_guardian_chart_freshness.json"
                first_freshness = json.loads(freshness_path.read_text())
                self.assertEqual(first_freshness["coverage_cursor"], 2)
                first_freshness["next_refresh_after_utc"] = "2000-01-01T00:00:00+00:00"
                freshness_path.write_text(json.dumps(first_freshness))

                env["QR_FAKE_GUARDIAN_CHART_MODE"] = failure_mode
                failed = subprocess.run(
                    ["bash", str(GUARDIAN_WRAPPER)],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(failed.returncode, 0, failed.stderr)
                failed_freshness = json.loads(freshness_path.read_text())
                self.assertEqual(failed_freshness["status"], failure_mode)
                self.assertEqual(failed_freshness["coverage_cursor"], 2)
                self.assertEqual(failed_freshness["proposed_coverage_cursor"], 2)
                self.assertEqual(failed_freshness["proposed_coverage_next_cursor"], 4)
                self.assertEqual(failed_freshness["active_scope_cursor"], 2)
                self.assertTrue(failed_freshness["coverage_retry_required"])
                self.assertFalse(failed_freshness["coverage_cursor_advanced"])
                failed_freshness["next_refresh_after_utc"] = "2000-01-01T00:00:00+00:00"
                freshness_path.write_text(json.dumps(failed_freshness))

                (root / "data" / "active_opportunity_board.json").write_text(
                    json.dumps(
                        {
                            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                            "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                            "top_lane": {
                                "pair": "USD_JPY",
                                "lane_id": "changed-before-retry",
                            },
                        }
                    )
                )

                env["QR_FAKE_GUARDIAN_CHART_MODE"] = "FRESH"
                recovered = subprocess.run(
                    ["bash", str(GUARDIAN_WRAPPER)],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(recovered.returncode, 0, recovered.stderr)

                calls = [json.loads(line) for line in capture.read_text().splitlines()]
                pair_calls = [call for call in calls if "pair-charts" in call]
                self.assertEqual(len(pair_calls), 3)
                failed_pairs = pair_calls[1][pair_calls[1].index("--pairs") + 1]
                retried_pairs = pair_calls[2][pair_calls[2].index("--pairs") + 1]
                self.assertEqual(retried_pairs, failed_pairs)
                final_freshness = json.loads(freshness_path.read_text())
                self.assertEqual(final_freshness["status"], "FRESH")
                self.assertEqual(final_freshness["coverage_cursor"], 4)
                self.assertEqual(final_freshness["active_scope_cursor"], 2)
                self.assertFalse(final_freshness["coverage_retry_required"])
                self.assertTrue(final_freshness["coverage_cursor_advanced"])

    def test_position_guardian_retry_expansion_records_new_hard_pair_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_snapshot = {
                "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                "positions": [],
                "orders": [],
                "quotes": {
                    pair: {"bid": 1.0, "ask": 1.0001}
                    for pair in DEFAULT_TRADER_PAIRS
                },
            }
            env, capture = _guardian_wrapper_env(
                root,
                snapshot=base_snapshot,
                trigger_contract={"entries": []},
                candidate_limit=2,
            )
            freshness_path = root / "data" / "position_guardian_chart_freshness.json"
            snapshot_source = Path(env["QR_FAKE_GUARDIAN_SNAPSHOT"])

            initial = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(initial.returncode, 0, initial.stderr)
            initial_freshness = json.loads(freshness_path.read_text())
            initial_freshness["next_refresh_after_utc"] = "2000-01-01T00:00:00+00:00"
            freshness_path.write_text(json.dumps(initial_freshness))

            env["QR_FAKE_GUARDIAN_CHART_MODE"] = "STALE"
            failed = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(failed.returncode, 0, failed.stderr)
            failed_freshness = json.loads(freshness_path.read_text())
            failed_pairs = list(failed_freshness["candidate_pairs"])
            failed_freshness["next_refresh_after_utc"] = "2000-01-01T00:00:00+00:00"
            freshness_path.write_text(json.dumps(failed_freshness))

            hard_pair = DEFAULT_TRADER_PAIRS[10]
            self.assertNotIn(hard_pair, failed_pairs)
            snapshot_source.write_text(
                json.dumps(
                    {
                        **base_snapshot,
                        "orders": [
                            {
                                "order_id": "new-hard-during-retry",
                                "pair": hard_pair,
                                "state": "PENDING",
                            }
                        ],
                    }
                )
            )
            env["QR_FAKE_GUARDIAN_CHART_MODE"] = "FRESH"

            recovered = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(recovered.returncode, 0, recovered.stderr)

            freshness = json.loads(freshness_path.read_text())
            self.assertEqual(freshness["effective_candidate_pair_limit"], 3)
            self.assertEqual(freshness["hard_priority_candidate_pairs"], [hard_pair])
            self.assertTrue(freshness["candidate_pair_limit_expanded"])
            self.assertTrue(
                freshness["candidate_pair_limit_expanded_for_hard_priority"]
            )
            self.assertEqual(
                freshness["candidate_pair_limit_expansion_reasons"],
                ["HARD_PRIORITY_SCOPE", "RETRY_SCOPE_RETENTION"],
            )
            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            pair_calls = [call for call in calls if "pair-charts" in call]
            retried_pairs = set(
                pair_calls[-1][pair_calls[-1].index("--pairs") + 1].split(",")
            )
            self.assertEqual(retried_pairs, {*failed_pairs, hard_pair})

    def test_position_guardian_current_integrity_block_does_not_pin_rotation(self) -> None:
        for chart_mode in (
            "INTEGRITY_BLOCKED",
            "INTEGRITY_BLOCKED_PROVENANCE_INVALID",
        ):
            with self.subTest(chart_mode=chart_mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                env, capture = _guardian_wrapper_env(
                    root,
                    snapshot={
                        "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                        "positions": [],
                        "orders": [],
                        "quotes": {
                            pair: {"bid": 1.0, "ask": 1.0001}
                            for pair in DEFAULT_TRADER_PAIRS
                        },
                    },
                    trigger_contract={"entries": []},
                    candidate_limit=2,
                )
                env["QR_FAKE_GUARDIAN_CHART_MODE"] = chart_mode

                first = subprocess.run(
                    ["bash", str(GUARDIAN_WRAPPER)],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(first.returncode, 0, first.stderr)
                freshness_path = root / "data" / "position_guardian_chart_freshness.json"
                freshness = json.loads(freshness_path.read_text())
                self.assertEqual(freshness["status"], "FRESH")
                self.assertEqual(
                    freshness["blocked_technical_inputs"],
                    [f"{DEFAULT_TRADER_PAIRS[0]}:M5"],
                )
                self.assertEqual(freshness["coverage_cursor"], 2)
                self.assertTrue(freshness["coverage_cursor_advanced"])
                self.assertFalse(freshness["coverage_retry_required"])

                freshness["next_refresh_after_utc"] = "2000-01-01T00:00:00+00:00"
                freshness_path.write_text(json.dumps(freshness))
                env["QR_FAKE_GUARDIAN_CHART_MODE"] = "FRESH"
                second = subprocess.run(
                    ["bash", str(GUARDIAN_WRAPPER)],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(second.returncode, 0, second.stderr)

                calls = [json.loads(line) for line in capture.read_text().splitlines()]
                pair_calls = [call for call in calls if "pair-charts" in call]
                self.assertEqual(len(pair_calls), 2)
                self.assertEqual(
                    pair_calls[0][pair_calls[0].index("--pairs") + 1].split(","),
                    list(DEFAULT_TRADER_PAIRS[:2]),
                )
                self.assertEqual(
                    pair_calls[1][pair_calls[1].index("--pairs") + 1].split(","),
                    list(DEFAULT_TRADER_PAIRS[2:4]),
                )
                final_freshness = json.loads(freshness_path.read_text())
                self.assertEqual(final_freshness["coverage_cursor"], 4)

    def test_position_guardian_integrity_block_without_valid_clock_retries_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, _capture = _guardian_wrapper_env(
                root,
                snapshot={
                    "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                    "positions": [],
                    "orders": [],
                    "quotes": {
                        pair: {"bid": 1.0, "ask": 1.0001}
                        for pair in DEFAULT_TRADER_PAIRS
                    },
                },
                trigger_contract={"entries": []},
                candidate_limit=2,
            )
            env["QR_FAKE_GUARDIAN_CHART_MODE"] = "INTEGRITY_BLOCKED_BAD_CLOCK"

            result = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            freshness = json.loads(
                (root / "data" / "position_guardian_chart_freshness.json").read_text()
            )
            self.assertEqual(freshness["status"], "PARTIAL")
            self.assertIsNone(freshness["coverage_cursor"])
            self.assertFalse(freshness["coverage_cursor_advanced"])
            self.assertTrue(freshness["coverage_retry_required"])
            blocked_row = next(
                row
                for row in freshness["rows"]
                if row["pair"] == DEFAULT_TRADER_PAIRS[0]
                and row["timeframe"] == "M5"
            )
            self.assertEqual(
                blocked_row["status"],
                "INVALID_TECHNICAL_CANDLE_INTEGRITY_RECEIPT",
            )

    def test_position_guardian_invalid_or_incomplete_integrity_retries_scope(self) -> None:
        cases = (
            (
                "INTEGRITY_BLOCKED_INVALID_RECEIPT",
                "INVALID_TECHNICAL_CANDLE_INTEGRITY_RECEIPT",
            ),
            (
                "INTEGRITY_BLOCKED_MALFORMED",
                "INCOMPLETE_TECHNICAL_CANDLE_INTEGRITY_ACQUISITION",
            ),
            (
                "INTEGRITY_BLOCKED_FUTURE_CLOCK",
                "FUTURE_COMPLETE_CANDLE_TIME",
            ),
            ("FUTURE", "FUTURE_COMPLETE_CANDLE_TIME"),
            (
                "NAIVE_CANDLE_TIME",
                "INVALID_TECHNICAL_CANDLE_INTEGRITY_RECEIPT",
            ),
            (
                "INTEGRITY_BLOCKED_WRONG_PAIR",
                "INVALID_TECHNICAL_CANDLE_INTEGRITY_RECEIPT",
            ),
            (
                "INTEGRITY_BLOCKED_WRONG_TIMEFRAME",
                "INVALID_TECHNICAL_CANDLE_INTEGRITY_RECEIPT",
            ),
            (
                "INTEGRITY_BLOCKED_FALSE_FLAG",
                "INVALID_TECHNICAL_CANDLE_INTEGRITY_RECEIPT",
            ),
            ("NAIVE_GENERATED_AT", "INVALID_CHART_GENERATED_AT_UTC"),
            ("FUTURE_GENERATED_AT", "INVALID_CHART_GENERATED_AT_UTC"),
            (
                "MISSING_INTEGRITY_RECEIPT",
                "INVALID_TECHNICAL_CANDLE_INTEGRITY_RECEIPT",
            ),
            (
                "EMPTY_INTEGRITY_RECEIPT",
                "INVALID_TECHNICAL_CANDLE_INTEGRITY_RECEIPT",
            ),
        )
        for chart_mode, expected_row_status in cases:
            with self.subTest(chart_mode=chart_mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                env, _capture = _guardian_wrapper_env(
                    root,
                    snapshot={
                        "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                        "positions": [],
                        "orders": [],
                        "quotes": {
                            pair: {"bid": 1.0, "ask": 1.0001}
                            for pair in DEFAULT_TRADER_PAIRS
                        },
                    },
                    trigger_contract={"entries": []},
                    candidate_limit=2,
                )
                env["QR_FAKE_GUARDIAN_CHART_MODE"] = chart_mode

                result = subprocess.run(
                    ["bash", str(GUARDIAN_WRAPPER)],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                freshness = json.loads(
                    (root / "data" / "position_guardian_chart_freshness.json").read_text()
                )
                self.assertEqual(freshness["status"], "PARTIAL")
                self.assertIsNone(freshness["coverage_cursor"])
                self.assertFalse(freshness["coverage_cursor_advanced"])
                self.assertTrue(freshness["coverage_retry_required"])
                blocked_row = next(
                    row
                    for row in freshness["rows"]
                    if row["pair"] == DEFAULT_TRADER_PAIRS[0]
                    and row["timeframe"] == "M5"
                )
                self.assertEqual(blocked_row["status"], expected_row_status)

    def test_position_guardian_absolute_cursor_covers_all_pairs_under_priority_churn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            priority_pair = DEFAULT_TRADER_PAIRS[-1]
            env, capture = _guardian_wrapper_env(
                root,
                snapshot={
                    "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                    "positions": [],
                    "orders": [],
                    "quotes": {
                        pair: {"bid": 1.0, "ask": 1.0001}
                        for pair in DEFAULT_TRADER_PAIRS
                    },
                },
                trigger_contract={"entries": []},
                candidate_limit=2,
            )
            board_path = root / "data" / "active_opportunity_board.json"
            freshness_path = root / "data" / "position_guardian_chart_freshness.json"

            for cycle in range(20):
                if cycle:
                    freshness = json.loads(freshness_path.read_text())
                    freshness["next_refresh_after_utc"] = "2000-01-01T00:00:00+00:00"
                    freshness_path.write_text(json.dumps(freshness))
                if cycle % 2:
                    board_path.write_text(
                        json.dumps(
                            {
                                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                                "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                                "top_lane": {
                                    "pair": priority_pair,
                                    "lane_id": f"priority-{cycle}",
                                },
                            }
                        )
                    )
                else:
                    board_path.unlink(missing_ok=True)
                result = subprocess.run(
                    ["bash", str(GUARDIAN_WRAPPER)],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            pair_calls = [call for call in calls if "pair-charts" in call]
            self.assertEqual(len(pair_calls), 20)
            observed = {
                pair
                for call in pair_calls
                for pair in call[call.index("--pairs") + 1].split(",")
            }
            self.assertEqual(observed, set(DEFAULT_TRADER_PAIRS))
            freshness = json.loads(freshness_path.read_text())
            self.assertEqual(
                freshness["coverage_cursor_semantics"],
                "NEXT_CONFIGURED_PAIR_INDEX",
            )

    def test_position_guardian_retry_keeps_failed_open_pair_after_position_closes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_snapshot = {
                "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                "positions": [],
                "orders": [],
                "quotes": {
                    pair: {"bid": 1.0, "ask": 1.0001}
                    for pair in DEFAULT_TRADER_PAIRS
                },
            }
            env, capture = _guardian_wrapper_env(
                root,
                snapshot=base_snapshot,
                trigger_contract={"entries": []},
                candidate_limit=2,
            )
            snapshot_source = Path(env["QR_FAKE_GUARDIAN_SNAPSHOT"])
            freshness_path = root / "data" / "position_guardian_chart_freshness.json"

            initial = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(initial.returncode, 0, initial.stderr)
            initial_freshness = json.loads(freshness_path.read_text())
            initial_freshness["next_refresh_after_utc"] = "2000-01-01T00:00:00+00:00"
            freshness_path.write_text(json.dumps(initial_freshness))

            formerly_open_pair = DEFAULT_TRADER_PAIRS[2]
            snapshot_source.write_text(
                json.dumps(
                    {
                        **base_snapshot,
                        "positions": [
                            {
                                "pair": formerly_open_pair,
                                "owner": "operator_manual",
                            }
                        ],
                    }
                )
            )
            env["QR_FAKE_GUARDIAN_CHART_MODE"] = "STALE"
            failed = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(failed.returncode, 0, failed.stderr)
            failed_freshness = json.loads(freshness_path.read_text())
            self.assertEqual(failed_freshness["status"], "STALE")
            self.assertIn(formerly_open_pair, failed_freshness["monitor_pairs"])
            failed_freshness["next_refresh_after_utc"] = "2000-01-01T00:00:00+00:00"
            freshness_path.write_text(json.dumps(failed_freshness))

            snapshot_source.write_text(json.dumps(base_snapshot))
            env["QR_FAKE_GUARDIAN_CHART_MODE"] = "FRESH"
            recovered = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(recovered.returncode, 0, recovered.stderr)

            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            pair_calls = [call for call in calls if "pair-charts" in call]
            self.assertEqual(len(pair_calls), 3)
            failed_pairs = pair_calls[1][pair_calls[1].index("--pairs") + 1]
            retried_pairs = pair_calls[2][pair_calls[2].index("--pairs") + 1]
            self.assertEqual(retried_pairs, failed_pairs)
            self.assertIn(formerly_open_pair, retried_pairs.split(","))
            final_freshness = json.loads(freshness_path.read_text())
            self.assertEqual(final_freshness["coverage_cursor"], 5)

    def test_position_guardian_zero_rotation_budget_still_keeps_all_hard_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, capture = _guardian_wrapper_env(
                root,
                snapshot={
                    "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                    "positions": [],
                    "orders": [
                        {"order_id": "pending-1", "pair": "AUD_CAD", "state": "PENDING"}
                    ],
                    "quotes": {
                        "AUD_CAD": {"bid": 0.9, "ask": 0.9001},
                        "AUD_JPY": {"bid": 100.0, "ask": 100.01},
                    },
                },
                trigger_contract={
                    "entries": [
                        {"pair": "AUD_JPY", "lane_id": "hard-lane", "status": "LIVE_READY"}
                    ]
                },
                candidate_limit=0,
            )

            result = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            pair_call = next(call for call in calls if "pair-charts" in call)
            self.assertEqual(
                set(pair_call[pair_call.index("--pairs") + 1].split(",")),
                {"AUD_CAD", "AUD_JPY"},
            )
            freshness = json.loads(
                (root / "data" / "position_guardian_chart_freshness.json").read_text()
            )
            self.assertEqual(freshness["configured_candidate_pair_limit"], 0)
            self.assertEqual(freshness["effective_candidate_pair_limit"], 2)
            self.assertTrue(freshness["candidate_pair_limit_expanded_for_hard_priority"])
            self.assertTrue(freshness["candidate_pair_limit_applied"])
            self.assertFalse(freshness["coverage_cursor_advanced"])
            self.assertFalse(freshness["coverage_cursor_wrapped"])

    def test_position_guardian_zero_budget_without_hard_pairs_preserves_coverage_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, capture = _guardian_wrapper_env(
                root,
                snapshot={
                    "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                    "positions": [],
                    "orders": [],
                    "quotes": {},
                },
                trigger_contract={"entries": []},
                candidate_limit=2,
            )
            first = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            freshness_path = root / "data" / "position_guardian_chart_freshness.json"
            initial = json.loads(freshness_path.read_text())
            self.assertEqual(initial["coverage_cursor"], 2)
            initial["next_refresh_after_utc"] = "2000-01-01T00:00:00+00:00"
            freshness_path.write_text(json.dumps(initial))
            env["QR_POSITION_GUARDIAN_MAX_CANDIDATE_PAIRS"] = "0"

            paused = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(paused.returncode, 0, paused.stderr)
            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            self.assertEqual(sum("pair-charts" in call for call in calls), 1)
            freshness = json.loads(freshness_path.read_text())
            self.assertEqual(freshness["status"], "NO_MONITOR_SCOPE")
            self.assertEqual(freshness["coverage_cursor"], 2)
            self.assertEqual(
                freshness["coverage_cursor_semantics"],
                "NEXT_CONFIGURED_PAIR_INDEX",
            )
            self.assertFalse(freshness["coverage_cursor_advanced"])
            self.assertEqual(freshness["configured_candidate_pair_limit"], 0)

    def test_position_guardian_empty_scope_resets_legacy_cursor_before_absolute_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, capture = _guardian_wrapper_env(
                root,
                snapshot={
                    "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                    "positions": [],
                    "orders": [],
                    "quotes": {},
                },
                trigger_contract={"entries": []},
                candidate_limit=2,
            )
            first = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)

            freshness_path = root / "data" / "position_guardian_chart_freshness.json"
            legacy = json.loads(freshness_path.read_text())
            for key in (
                "coverage_cursor_semantics",
                "active_scope_cursor",
                "proposed_coverage_next_cursor",
                "proposed_coverage_scanned_count",
            ):
                legacy.pop(key, None)
            legacy["coverage_cursor"] = 13
            legacy["next_refresh_after_utc"] = "2000-01-01T00:00:00+00:00"
            freshness_path.write_text(json.dumps(legacy))
            env["QR_POSITION_GUARDIAN_MAX_CANDIDATE_PAIRS"] = "0"

            empty = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(empty.returncode, 0, empty.stderr)
            migrated = json.loads(freshness_path.read_text())
            self.assertEqual(migrated["status"], "NO_MONITOR_SCOPE")
            self.assertEqual(migrated["coverage_cursor"], 0)
            self.assertEqual(
                migrated["coverage_cursor_semantics"],
                "NEXT_CONFIGURED_PAIR_INDEX",
            )

            env["QR_POSITION_GUARDIAN_MAX_CANDIDATE_PAIRS"] = "1"
            resumed = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)

            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            pair_calls = [call for call in calls if "pair-charts" in call]
            self.assertEqual(len(pair_calls), 2)
            self.assertEqual(
                pair_calls[-1][pair_calls[-1].index("--pairs") + 1],
                DEFAULT_TRADER_PAIRS[0],
            )
            final_freshness = json.loads(freshness_path.read_text())
            self.assertEqual(final_freshness["coverage_cursor"], 1)

    def test_position_guardian_rejects_candidate_limit_over_28_before_broker_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, capture = _guardian_wrapper_env(
                root,
                snapshot={
                    "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                    "positions": [],
                    "orders": [],
                    "quotes": {},
                },
                trigger_contract={"entries": []},
                candidate_limit=29,
            )

            result = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("must be an integer in 0..28", result.stderr)
            self.assertFalse(capture.exists())

    def test_position_guardian_pins_hourly_board_and_non_eurusd_frontier_before_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, capture = _guardian_wrapper_env(
                root,
                snapshot={
                    "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                    "positions": [],
                    "orders": [],
                    "quotes": {
                        pair: {"bid": 1.0, "ask": 1.0001}
                        for pair in ("AUD_CAD", "CAD_JPY", "CHF_JPY", "EUR_AUD")
                    },
                },
                trigger_contract={"entries": []},
                candidate_limit=2,
            )
            (root / "data" / "active_opportunity_board.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                        "top_lane": {"pair": "CAD_JPY", "lane_id": "board-lane"},
                    }
                )
            )
            (root / "data" / "non_eurusd_live_grade_frontier.json").write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "NON_EURUSD_FRONTIER_FOUND",
                        "next_evidence_lane": {"pair": "CHF_JPY", "lane_id": "frontier-lane"},
                    }
                )
            )

            result = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            pair_call = next(call for call in calls if "pair-charts" in call)
            self.assertEqual(pair_call[pair_call.index("--pairs") + 1], "CAD_JPY,CHF_JPY")
            freshness = json.loads((root / "data" / "position_guardian_chart_freshness.json").read_text())
            self.assertEqual(freshness["hourly_priority_candidate_pairs"], ["CAD_JPY", "CHF_JPY"])

    def test_position_guardian_replaces_old_scope_with_configured_rotation_when_idle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, capture = _guardian_wrapper_env(
                root,
                snapshot={
                    "fetched_at_utc": "2026-07-10T00:00:00+00:00",
                    "positions": [],
                    "orders": [],
                    "quotes": {},
                },
                trigger_contract={"entries": []},
                candidate_limit=2,
            )
            stale_charts = root / "data" / "position_guardian_pair_charts.json"
            stale_charts.write_text('{"charts":[{"pair":"EUR_USD"}],"guardian_monitor_pairs":["EUR_USD"]}\n')

            result = subprocess.run(
                ["bash", str(GUARDIAN_WRAPPER)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            pair_call = next(call for call in calls if "pair-charts" in call)
            expected = list(DEFAULT_TRADER_PAIRS[:2])
            self.assertEqual(
                pair_call[pair_call.index("--pairs") + 1].split(","),
                expected,
            )
            self.assertEqual(sum("guardian-event-router" in call for call in calls), 1)
            charts = json.loads(stale_charts.read_text())
            self.assertEqual(charts["guardian_monitor_pairs"], expected)
            self.assertEqual(
                [item["pair"] for item in charts["charts"]],
                expected,
            )

    def test_live_lock_release_preserves_reacquired_lock_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_dir = root / "lock"
            env = os.environ.copy()
            env.update(
                {
                    "QR_HELPER": str(ROOT / "scripts" / "qr-live-lock.sh"),
                    "QR_LOCK_DIR": str(lock_dir),
                }
            )

            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        "set -euo pipefail; "
                        "source \"$QR_HELPER\"; "
                        "qr_live_lock_acquire \"$QR_LOCK_DIR\" test-lock 0 '' 0.1; "
                        "printf '%s\\n' other-token > \"$QR_LOCK_DIR/token\"; "
                        "qr_live_lock_release; "
                        "test -d \"$QR_LOCK_DIR\""
                    ),
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(lock_dir.exists())

    def test_direct_send_lock_blocks_overlapping_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_dir = root / "lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()) + "\n")
            original_lock_dir = os.environ.get("QR_AUTOTRADE_LOCK_DIR")
            original_lock_held = os.environ.get("QR_AUTOTRADE_LOCK_HELD")
            os.environ["QR_AUTOTRADE_LOCK_DIR"] = str(lock_dir)
            os.environ.pop("QR_AUTOTRADE_LOCK_HELD", None)
            try:
                with self.assertRaisesRegex(RuntimeError, "another autotrade cycle is already running"):
                    _acquire_autotrade_lock(send=True)
            finally:
                _restore_env("QR_AUTOTRADE_LOCK_DIR", original_lock_dir)
                _restore_env("QR_AUTOTRADE_LOCK_HELD", original_lock_held)

    def test_forged_held_env_cannot_bypass_another_direct_send_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_dir = Path(tmp) / "lock"
            proc = subprocess.Popen(["sleep", "2"])
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(f"{proc.pid}\n")
            old_lock_dir = os.environ.get("QR_AUTOTRADE_LOCK_DIR")
            old_held = os.environ.get("QR_AUTOTRADE_LOCK_HELD")
            old_owner_token = os.environ.get("QR_AUTOTRADE_LOCK_OWNER_TOKEN")
            os.environ["QR_AUTOTRADE_LOCK_DIR"] = str(lock_dir)
            os.environ["QR_AUTOTRADE_LOCK_HELD"] = "1"
            os.environ.pop("QR_AUTOTRADE_LOCK_OWNER_TOKEN", None)
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "another autotrade cycle is already running",
                ):
                    _acquire_autotrade_lock(send=True)
                self.assertTrue(lock_dir.exists())
            finally:
                proc.terminate()
                proc.wait(timeout=2)
                _restore_env("QR_AUTOTRADE_LOCK_DIR", old_lock_dir)
                _restore_env("QR_AUTOTRADE_LOCK_HELD", old_held)
                _restore_env("QR_AUTOTRADE_LOCK_OWNER_TOKEN", old_owner_token)


def _run_competing_shell_lockers(
    root: Path,
    *,
    delay_owner_write: bool,
) -> list[subprocess.CompletedProcess[str]]:
    ready = root / "ready"
    start = root / "start"
    winners = root / "winners"
    env = os.environ.copy()
    env.update(
        {
            "QR_HELPER": str(ROOT / "scripts" / "qr-live-lock.sh"),
            "QR_LOCK_DIR": str(root / "lock"),
            "QR_READY_FILE": str(ready),
            "QR_START_FILE": str(start),
            "QR_WINNERS_FILE": str(winners),
            "QR_LIVE_LOCK_INIT_GRACE_SECONDS": "0.05",
        }
    )
    delayed_owner = ""
    if delay_owner_write:
        delayed_owner = (
            "eval \"$(declare -f qr_live_lock_write_owner | "
            "sed '1s/qr_live_lock_write_owner/qr_live_lock_write_owner_impl/')\"; "
            "qr_live_lock_write_owner() { sleep 0.3; "
            "qr_live_lock_write_owner_impl \"$@\"; }; "
        )
    command = (
        "set -euo pipefail; "
        "source \"$QR_HELPER\"; "
        + delayed_owner
        + "printf '%s\\n' \"$$\" >> \"$QR_READY_FILE\"; "
        "while [[ ! -f \"$QR_START_FILE\" ]]; do sleep 0.01; done; "
        "qr_live_lock_acquire \"$QR_LOCK_DIR\" contender 0 '' 0.01; "
        "printf '%s\\n' \"$$\" >> \"$QR_WINNERS_FILE\"; "
        "sleep 0.5; "
        "qr_live_lock_release"
    )
    processes = [
        subprocess.Popen(
            ["bash", "-c", command],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if ready.exists() and len(ready.read_text().splitlines()) == 2:
            break
        time.sleep(0.01)
    else:
        for process in processes:
            process.terminate()
        raise AssertionError("competing lock processes did not reach the start barrier")
    start.write_text("go\n")

    results = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=5)
        results.append(
            subprocess.CompletedProcess(
                process.args,
                process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        )
    return results


def _wrapper_env(
    root: Path,
    capture: Path,
    *,
    live_enabled: str | None = None,
    sync_exit: int = 0,
    python_exit: int = 0,
    guardian_loaded: bool = True,
) -> dict[str, str]:
    env_file = root / "oanda.env"
    lines = [
        "QR_OANDA_ACCOUNT_ID=acct-test",
        "QR_OANDA_TOKEN=token-test",
        "QR_OANDA_BASE_URL=https://example.test",
    ]
    if live_enabled is not None:
        lines.append(f"QR_LIVE_ENABLED={live_enabled}")
    env_file.write_text(
        "\n".join(lines)
        + "\n"
    )
    fake_scripts = root / "scripts"
    fake_scripts.mkdir()
    fake_sync = fake_scripts / "sync-live-runtime.sh"
    fake_sync.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "printf '%s\\n' \"$*\" > \"$QR_SYNC_MARKER_PATH\"",
                f"exit {sync_exit}",
            ]
        )
        + "\n"
    )
    fake_sync.chmod(0o755)
    fake_guardian_check = fake_scripts / "install-position-guardian.sh"
    fake_guardian_check.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "if [[ \"${1:-}\" != \"--require-loaded\" ]]; then",
                "  printf 'unexpected guardian check args: %s\\n' \"$*\" >&2",
                "  exit 64",
                "fi",
                "if [[ \"${QR_FAKE_POSITION_GUARDIAN_LOADED:-1}\" == \"1\" ]]; then",
                "  printf '[install-position-guardian] active OK: label=com.quantrabbit.position-guardian\\n'",
                "  exit 0",
                "fi",
                "printf '[install-position-guardian] position guardian launchd label is not loaded: com.quantrabbit.position-guardian\\n' >&2",
                "exit 6",
            ]
        )
        + "\n"
    )
    fake_guardian_check.chmod(0o755)
    fake_bin = root / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'if [[ "${{1:-}}" == "-" ]]; then exec "{sys.executable}" "$@"; fi',
                "{",
                "  printf 'QR_LIVE_ENABLED=%s\\n' \"${QR_LIVE_ENABLED:-}\"",
                "  printf 'QR_REQUIRE_POSITION_GUARDIAN_ACTIVE=%s\\n' \"${QR_REQUIRE_POSITION_GUARDIAN_ACTIVE:-}\"",
                "  printf 'QR_POSITION_GUARDIAN_ACTIVE=%s\\n' \"${QR_POSITION_GUARDIAN_ACTIVE:-}\"",
                "  printf 'QR_AUTOTRADE_LOCK_HELD=%s\\n' \"${QR_AUTOTRADE_LOCK_HELD:-}\"",
                "  printf 'PYTHONPATH=%s\\n' \"${PYTHONPATH:-}\"",
                "  printf 'ARGV='",
                "  for arg in \"$@\"; do printf '<%s>' \"$arg\"; done",
                "  printf '\\n'",
                "} >> \"$QR_CAPTURE_PATH\"",
                "for arg in \"$@\"; do",
                "  if [[ \"$arg\" == \"autotrade-cycle\" ]]; then",
                f"    exit {python_exit}",
                "  fi",
                "done",
                "exit 0",
            ]
        )
        + "\n"
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env.get('PATH', '')}",
            "QR_CAPTURE_PATH": str(capture),
            "QR_PYTHON": str(fake_python),
            "QR_OANDA_ENV_FILE": str(env_file),
            "QR_AUTOTRADE_LOCK_DIR": str(root / "lock"),
            "QR_TRADER_ROOT_DIR": str(root),
            "QR_LIVE_SYNC_ENABLED": "1",
            "QR_SYNC_MARKER_PATH": str(root / "sync.args"),
            "QR_FAKE_POSITION_GUARDIAN_LOADED": "1" if guardian_loaded else "0",
        }
    )
    return env


def _guardian_wrapper_env(
    root: Path,
    *,
    snapshot: dict[str, object],
    trigger_contract: dict[str, object],
    candidate_limit: int,
) -> tuple[dict[str, str], Path]:
    data = root / "data"
    docs = root / "docs"
    data.mkdir(parents=True)
    docs.mkdir(parents=True)
    snapshot_source = root / "snapshot-source.json"
    snapshot_source.write_text(json.dumps(snapshot))
    (data / "guardian_trigger_contract.json").write_text(json.dumps(trigger_contract))
    (data / "order_intents.json").write_text('{"results":[]}\n')
    env_file = root / "oanda.env"
    env_file.write_text("QR_LIVE_ENABLED=1\n")
    capture = root / "guardian-calls.jsonl"
    fake_python = root / "fake-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        + r'''
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.analysis.candles import _technical_candles_from_payload
from quant_rabbit.instruments import NORMAL_SPREAD_PIPS, instrument_pip_factor
from quant_rabbit.risk import RiskPolicy


def value_after(args, name, default=None):
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return default


def integrity_blocked_view(pair, timeframe, now, chart_mode):
    duration = {"M1": 60, "M5": 300, "M15": 900}[timeframe]
    latest_epoch = int(now.timestamp())
    latest_epoch -= latest_epoch % duration
    if chart_mode in {"FUTURE", "INTEGRITY_BLOCKED_FUTURE_CLOCK"}:
        pass
    elif chart_mode == "STALE":
        latest_epoch -= duration * 4
    else:
        latest_epoch -= duration
    latest = datetime.fromtimestamp(latest_epoch, tz=timezone.utc)
    count = 120
    factor = instrument_pip_factor(pair)
    decimals = 3 if factor == 100 else 5
    base = 100.0 if factor == 100 else 1.0
    cap_pips = NORMAL_SPREAD_PIPS[pair] * RiskPolicy().max_spread_multiple
    clean_spread_pips = NORMAL_SPREAD_PIPS[pair]
    if chart_mode == "INTEGRITY_BLOCKED_PROVENANCE_INVALID":
        contaminated_index = count - 6
    elif chart_mode == "INTEGRITY_BLOCKED_MALFORMED":
        contaminated_index = None
    elif chart_mode.startswith("INTEGRITY_BLOCKED"):
        contaminated_index = count - 1
    else:
        contaminated_index = None
    candles = []
    for index in range(count):
        started = latest - timedelta(seconds=duration * (count - 1 - index))
        spread_pips = cap_pips + 0.1 if index == contaminated_index else clean_spread_pips
        half_spread = spread_pips / factor / 2.0
        bid = f"{base - half_spread:.{decimals}f}"
        ask = f"{base + half_spread:.{decimals}f}"
        mid = f"{base:.{decimals}f}"
        candle = {
                "time": started.isoformat().replace("+00:00", "Z"),
                "complete": True,
                "volume": 1,
                "mid": {"o": mid, "h": mid, "l": mid, "c": mid},
                "bid": {"o": bid, "h": bid, "l": bid, "c": bid},
                "ask": {"o": ask, "h": ask, "l": ask, "c": ask},
            }
        if chart_mode == "INTEGRITY_BLOCKED_MALFORMED" and index == count - 6:
            candle.pop("ask")
        candles.append(candle)
    batch = _technical_candles_from_payload(
        {"instrument": pair, "granularity": timeframe, "candles": candles},
        pair=pair,
        granularity=timeframe,
        requested_count=count,
        pip_factor=factor,
        normal_spread_pips=NORMAL_SPREAD_PIPS[pair],
        max_spread_multiple=RiskPolicy().max_spread_multiple,
    )
    integrity = dict(batch.integrity)
    tail_count = integrity["recent_clean_tail_count"]
    published = list(batch.candles[-tail_count:])[-30:] if tail_count else []
    recent = [
        {
            "t": candle.timestamp_utc.isoformat(),
            "complete": candle.complete,
            "o": candle.open,
            "h": candle.high,
            "l": candle.low,
            "c": candle.close,
            "v": candle.volume,
        }
        for candle in published
    ]
    if chart_mode == "NAIVE_CANDLE_TIME" and recent:
        recent[-1]["t"] = recent[-1]["t"].replace("+00:00", "")
    if chart_mode == "INTEGRITY_BLOCKED_BAD_CLOCK":
        integrity["latest_complete_timestamp_utc"] = "2026-07-14T00:00:00"
    elif chart_mode == "INTEGRITY_BLOCKED_INVALID_RECEIPT":
        integrity["raw_entry_count"] += 1
    elif chart_mode == "INTEGRITY_BLOCKED_WRONG_PAIR":
        integrity["pair"] = "GBP_USD" if pair != "GBP_USD" else "EUR_USD"
    elif chart_mode == "INTEGRITY_BLOCKED_WRONG_TIMEFRAME":
        integrity["granularity"] = "M1"
    elif chart_mode == "INTEGRITY_BLOCKED_FALSE_FLAG":
        integrity["forecast_blocking"] = False
    view = {
        "granularity": timeframe,
        "recent_candles": recent,
        # Match the production chart contract: this count records the full
        # contiguous clean tail even though only the newest 30 candles are
        # published in ``recent_candles``.
        "indicators": {"candles_count": tail_count},
        "candle_integrity": integrity,
    }
    if chart_mode == "MISSING_INTEGRITY_RECEIPT":
        view.pop("candle_integrity")
    elif chart_mode == "EMPTY_INTEGRITY_RECEIPT":
        view["candle_integrity"] = {}
    return view


args = sys.argv[1:]
if args and args[0] == "-":
    source = sys.stdin.read()
    sys.argv = ["-", *args[1:]]
    exec(compile(source, "<guardian-helper>", "exec"), {"__name__": "__main__"})
    raise SystemExit(0)

capture = Path(os.environ["QR_FAKE_GUARDIAN_CAPTURE"])
capture.parent.mkdir(parents=True, exist_ok=True)
with capture.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\n")

if len(args) < 3 or args[:2] != ["-m", "quant_rabbit.cli"]:
    raise SystemExit(64)
command = args[2]
if command == "broker-snapshot":
    destination = Path(value_after(args, "--output"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(os.environ["QR_FAKE_GUARDIAN_SNAPSHOT"], destination)
elif command == "pair-charts":
    pairs = [item for item in str(value_after(args, "--pairs", "")).split(",") if item]
    timeframes = [item for item in str(value_after(args, "--timeframes", "")).split(",") if item]
    durations = {"M1": 60, "M5": 300, "M15": 900}
    now = datetime.now(timezone.utc)
    chart_mode = str(os.environ.get("QR_FAKE_GUARDIAN_CHART_MODE") or "FRESH").upper()
    if chart_mode not in {
        "FRESH",
        "PARTIAL",
        "STALE",
        "FUTURE",
        "NAIVE_CANDLE_TIME",
        "NAIVE_GENERATED_AT",
        "FUTURE_GENERATED_AT",
        "MISSING_INTEGRITY_RECEIPT",
        "EMPTY_INTEGRITY_RECEIPT",
        "INTEGRITY_BLOCKED",
        "INTEGRITY_BLOCKED_BAD_CLOCK",
        "INTEGRITY_BLOCKED_PROVENANCE_INVALID",
        "INTEGRITY_BLOCKED_FUTURE_CLOCK",
        "INTEGRITY_BLOCKED_INVALID_RECEIPT",
        "INTEGRITY_BLOCKED_MALFORMED",
        "INTEGRITY_BLOCKED_WRONG_PAIR",
        "INTEGRITY_BLOCKED_WRONG_TIMEFRAME",
        "INTEGRITY_BLOCKED_FALSE_FLAG",
    }:
        raise SystemExit(65)
    charts = []
    for pair in pairs:
        views = []
        for timeframe in timeframes:
            targeted_mode = chart_mode if chart_mode in {
                "FRESH",
                "PARTIAL",
                "STALE",
                "FUTURE",
                "NAIVE_CANDLE_TIME",
            } else "FRESH"
            if pair == pairs[0] and timeframe == "M5" and chart_mode in {
                "INTEGRITY_BLOCKED",
                "INTEGRITY_BLOCKED_BAD_CLOCK",
                "INTEGRITY_BLOCKED_PROVENANCE_INVALID",
                "INTEGRITY_BLOCKED_FUTURE_CLOCK",
                "INTEGRITY_BLOCKED_INVALID_RECEIPT",
                "INTEGRITY_BLOCKED_MALFORMED",
                "INTEGRITY_BLOCKED_WRONG_PAIR",
                "INTEGRITY_BLOCKED_WRONG_TIMEFRAME",
                "INTEGRITY_BLOCKED_FALSE_FLAG",
                "MISSING_INTEGRITY_RECEIPT",
                "EMPTY_INTEGRITY_RECEIPT",
            }:
                targeted_mode = chart_mode
            view = integrity_blocked_view(pair, timeframe, now, targeted_mode)
            views.append(view)
        charts.append({"pair": pair, "views": views})
    if chart_mode == "PARTIAL" and charts:
        charts = charts[:-1]
    pairs_failed = len(pairs) - len(charts)
    payload = {
        "generated_at_utc": (
            now.replace(tzinfo=None).isoformat()
            if chart_mode == "NAIVE_GENERATED_AT"
            else (now + timedelta(minutes=10)).isoformat()
            if chart_mode == "FUTURE_GENERATED_AT"
            else now.isoformat()
        ),
        "timeframes": timeframes,
        "pairs_requested": len(pairs),
        "pairs_succeeded": len(charts),
        "pairs_failed": pairs_failed,
        "partial": chart_mode == "PARTIAL",
        "failures": (
            [{"pair": pairs[-1], "error": "injected partial chart failure"}]
            if pairs_failed
            else []
        ),
        "charts": charts,
    }
    output = Path(value_after(args, "--output"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload) + "\n")
    report = Path(value_after(args, "--report"))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Pair Charts Report\n")
elif command == "position-management":
    output = Path(value_after(args, "--output"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"action":"HOLD","positions":[]}\n')
    report = Path(value_after(args, "--report"))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Position Management Report\n")
elif command == "position-execution":
    output = Path(value_after(args, "--output"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"status":"NO_ACTION","sent":false}\n')
    report = Path(value_after(args, "--report"))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Position Execution Report\n")
elif command == "guardian-event-router":
    output = Path(value_after(args, "--output"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"events":[]}\n')
raise SystemExit(0)
'''.lstrip()
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "QR_PYTHON": str(fake_python),
            "QR_TRADER_ROOT_DIR": str(root),
            "QR_OANDA_ENV_FILE": str(env_file),
            "QR_LIVE_ENABLED": "1",
            "QR_AUTOTRADE_LOCK_DIR": str(root / "lock"),
            "QR_FAKE_GUARDIAN_CAPTURE": str(capture),
            "QR_FAKE_GUARDIAN_SNAPSHOT": str(snapshot_source),
            "QR_POSITION_GUARDIAN_MAX_CANDIDATE_PAIRS": str(candidate_limit),
        }
    )
    return env, capture


def _init_git(root: Path) -> None:
    _run(["git", "init"], cwd=root)
    _run(["git", "config", "user.email", "test@example.invalid"], cwd=root)
    _run(["git", "config", "user.name", "Test User"], cwd=root)


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def _captured_cli_commands(payload: str) -> list[str]:
    marker = "ARGV=<-m><quant_rabbit.cli><"
    commands: list[str] = []
    for line in payload.splitlines():
        if line.startswith(marker):
            commands.append(line[len(marker) :].split(">", 1)[0])
    return commands


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    unittest.main()

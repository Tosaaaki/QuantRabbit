from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.automation import _acquire_autotrade_lock


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
            holder = root / "run-position-guardian-live.sh"
            holder.write_text("#!/usr/bin/env bash\nsleep 1\n")
            holder.chmod(0o755)
            proc = subprocess.Popen([str(holder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                lock_dir = root / "lock"
                lock_dir.mkdir()
                (lock_dir / "pid").write_text(f"{proc.pid}\n")

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
            self.assertIn("removing defunct lock holder", result.stderr)

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
                "EUR_USD,GBP_USD,USD_JPY,AUD_CAD,AUD_JPY",
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
                ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_CAD", "AUD_JPY"],
            )
            scope = charts["guardian_monitor_scope"]
            self.assertEqual(scope["USD_JPY"]["position_owners"], ["operator_manual"])
            self.assertTrue(scope["USD_JPY"]["non_trader_monitor_only"])
            self.assertFalse(scope["USD_JPY"]["management_write_eligible"])
            self.assertIn("active_trigger_candidate", scope["AUD_JPY"]["reasons"])
            self.assertIn("active_pending_order", scope["AUD_CAD"]["reasons"])
            freshness = json.loads((root / "data" / "position_guardian_chart_freshness.json").read_text())
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

    def test_position_guardian_clears_old_chart_scope_when_nothing_is_open_or_active(self) -> None:
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
            self.assertFalse(any("pair-charts" in call for call in calls))
            self.assertEqual(sum("guardian-event-router" in call for call in calls), 1)
            charts = json.loads(stale_charts.read_text())
            self.assertEqual(charts["guardian_monitor_pairs"], [])
            self.assertEqual(charts["charts"], [])

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


def value_after(args, name, default=None):
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return default


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
    charts = []
    for pair in pairs:
        views = []
        for timeframe in timeframes:
            started = now - timedelta(seconds=durations[timeframe])
            views.append(
                {
                    "granularity": timeframe,
                    "recent_candles": [
                        {
                            "t": started.isoformat(),
                            "complete": True,
                            "o": 1.0,
                            "h": 1.1,
                            "l": 0.9,
                            "c": 1.0,
                            "v": 1,
                        }
                    ],
                }
            )
        charts.append({"pair": pair, "views": views})
    payload = {
        "generated_at_utc": now.isoformat(),
        "timeframes": timeframes,
        "pairs_requested": len(pairs),
        "pairs_succeeded": len(pairs),
        "pairs_failed": 0,
        "partial": False,
        "failures": [],
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

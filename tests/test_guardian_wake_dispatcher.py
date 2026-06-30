from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.guardian_wake_dispatcher import DispatcherPaths, run_dispatcher


NOW = datetime(2026, 6, 30, 4, 0, tzinfo=timezone.utc)


class GuardianWakeDispatcherTest(unittest.TestCase):
    def test_gpt_json_only_prompt_contains_required_schema(self) -> None:
        prompt = (Path(__file__).resolve().parents[1] / "docs" / "guardian_wake_prompt.md").read_text()

        self.assertIn("Return JSON only", prompt)
        self.assertIn('"action": "TRADE|ADD|HOLD|HARVEST|REDUCE|CANCEL_PENDING|NO_ACTION"', prompt)
        self.assertIn('"new_information": true', prompt)
        self.assertIn('"thesis_state": "ALIVE|WOUNDED|INVALIDATED|EMERGENCY"', prompt)
        self.assertIn('"ownership": "SYSTEM|OPERATOR_MANUAL|UNKNOWN"', prompt)
        self.assertIn('"gateway_required": true', prompt)
        self.assertIn('"no_direct_oanda": true', prompt)

    def test_wake_false_does_not_start_codex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp), wake=False)
            paths.action_receipt.write_text('{"status":"STALE"}\n')
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            self.assertEqual(result["status"], "NO_WAKE")
            self.assertEqual(calls, [])
            self.assertFalse(paths.action_receipt.exists())
            self.assertTrue(paths.action_review.exists())
            self.assertIn("Receipt exists: `no`", paths.action_review.read_text())

    def test_accepted_hold_receipt_survives_later_no_wake(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="HOLD")),
            )
            paths.escalation.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "wake_gpt": False}))

            second = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=5),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="TRADE")),
            )

            self.assertEqual(first["status"], "RECEIPT_WRITTEN")
            self.assertEqual(second["status"], "NO_WAKE")
            payload = json.loads(paths.action_receipt.read_text())
            self.assertEqual(payload["receipt"]["action"], "HOLD")
            self.assertEqual(payload["receipt_lifecycle"], "ACTIVE")
            review = paths.action_review.read_text()
            self.assertIn("Dispatcher status: `NO_WAKE`", review)
            self.assertIn("Latest Accepted Receipt", review)
            self.assertIn("Action: `HOLD`", review)

    def test_accepted_no_action_receipt_survives_later_suppressed_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="NO_ACTION")),
            )

            result = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=5),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="TRADE")),
            )

            self.assertEqual(result["status"], "SUPPRESSED")
            self.assertEqual(len(calls), 1)
            payload = json.loads(paths.action_receipt.read_text())
            self.assertEqual(payload["receipt"]["action"], "NO_ACTION")
            self.assertEqual(payload["receipt_lifecycle"], "ACTIVE")
            review = paths.action_review.read_text()
            self.assertIn("Dispatcher status: `SUPPRESSED`", review)
            self.assertIn("Action: `NO_ACTION`", review)

    def test_wake_true_starts_codex_exec_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(calls), 1)
            self.assertEqual(Path(calls[0][0]).name, "codex")
            self.assertEqual(calls[0][1:3], ["exec", "-"])
            self.assertTrue(paths.action_receipt.exists())

    def test_active_qr_trader_lock_queues_instead_of_starting_second_codex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            paths.live_lock.mkdir(parents=True)
            (paths.live_lock / "pid").write_text(str(os.getpid()))
            (paths.live_lock / "command").write_text("run-autotrade-live")
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            self.assertEqual(result["status"], "QUEUED_FOR_ACTIVE_TRADER")
            self.assertEqual(calls, [])
            escalation = json.loads(paths.escalation.read_text())
            self.assertTrue(escalation["queued_for_active_trader"])

    def test_duplicate_wake_is_throttled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            result = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=5),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(result["status"], "SUPPRESSED")
            self.assertEqual(len(calls), 1)
            reasons = {item["reason"] for item in result["selection"]["suppressed"]}
            self.assertIn("THROTTLED", reasons)
            self.assertTrue(paths.action_receipt.exists())

    def test_p0_severity_increase_can_bypass_throttle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root, severity="P1", reasons=("NEW_EVENT",))
            calls: list[list[str]] = []
            run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))
            _write_wake(paths, severity="P0", reasons=("SEVERITY_INCREASE",))

            result = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=1),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(event_id="event-P0")),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(calls), 2)

    def test_codex_command_uses_gpt55_and_read_only_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            cmd = calls[0]
            self.assertEqual(cmd[cmd.index("-m") + 1], "gpt-5.5")
            self.assertEqual(cmd[cmd.index("-s") + 1], "read-only")
            self.assertEqual(cmd[cmd.index("-C") + 1], str(paths.live_root))
            self.assertIn('model_reasoning_effort = "high"', (paths.codex_home / "config.toml").read_text())

    def test_dispatcher_never_calls_oanda_or_gateway_cli_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            command_text = " ".join(calls[0]).lower()
            self.assertNotIn("oanda", command_text)
            self.assertNotIn("stage-live-order", command_text)
            self.assertNotIn("autotrade-cycle", command_text)
            self.assertNotIn("position-execution", command_text)

    def test_dispatcher_source_has_no_direct_oanda_order_cancel_close_path(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "tools" / "guardian_wake_dispatcher.py").read_text()

        for forbidden in (
            "requests.",
            "urllib.request",
            "api-fxtrade.oanda.com",
            "OANDA_TOKEN",
            "openTrades",
            "stage-live-order",
            "position-execution",
            "autotrade-cycle --send",
        ):
            self.assertNotIn(forbidden, source)

    def test_invalid_gpt_output_does_not_create_executable_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, "not json"))

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertFalse(paths.action_receipt.exists())
            self.assertTrue(paths.action_review.exists())

    def test_empty_last_message_falls_back_to_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex_stdout(calls, _valid_receipt()))

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertTrue(result["codex"]["stdout_fallback_used"])
            self.assertEqual(len(calls), 1)
            self.assertTrue(paths.action_receipt.exists())

    def test_jsonl_stdout_assistant_message_beats_trailing_status_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            stdout = "\n".join(
                [
                    json.dumps({"type": "session_started", "session_id": "s1"}),
                    json.dumps({"type": "agent_message", "message": _valid_receipt()}),
                    json.dumps({"type": "task_complete", "status": "success"}),
                ]
            )

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex_stdout(calls, stdout))

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertTrue(result["codex"]["stdout_fallback_used"])
            self.assertEqual(result["codex"]["last_message_source"], "stdout-assistant")
            self.assertTrue(paths.action_receipt.exists())

    def test_output_last_message_json_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(result["codex"]["last_message_source"], "output-last-message")
            self.assertTrue(paths.action_receipt.exists())

    def test_explicit_output_file_json_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_explicit(paths, calls, _valid_receipt()),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(result["codex"]["last_message_source"], "explicit-output-file")
            self.assertTrue(paths.action_receipt.exists())

    def test_session_jsonl_assistant_json_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_session(paths, calls, _valid_receipt()),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(result["codex"]["last_message_source"], "session-jsonl-assistant")
            self.assertTrue(result["codex"]["session_jsonl_fallback_used"])
            self.assertTrue(paths.action_receipt.exists())

    def test_empty_output_becomes_codex_empty_last_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, ""))

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertEqual(result["parse"]["error"], "CODEX_EMPTY_LAST_MESSAGE")
            self.assertEqual(len(calls), 2)
            self.assertFalse(paths.action_receipt.exists())

    def test_codex_banner_without_json_becomes_no_json_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, "OpenAI Codex\nworking in read-only mode\n"),
            )

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertEqual(result["parse"]["error"], "CODEX_NO_JSON_RECEIPT")
            self.assertEqual(len(calls), 2)
            self.assertFalse(paths.action_receipt.exists())

    def test_banner_only_output_retries_once_and_can_recover(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_stdout_sequence(
                    calls,
                    ["OpenAI Codex\nworking in read-only mode\n", _valid_receipt()],
                ),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(calls), 2)
            self.assertTrue(paths.action_receipt.exists())

    def test_assistant_text_without_json_becomes_no_json_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_session(paths, calls, "I cannot recommend action."),
            )

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertEqual(result["parse"]["error"], "CODEX_NO_JSON_RECEIPT")

    def test_session_jsonl_without_assistant_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_session_without_assistant(paths, calls),
            )

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertEqual(result["parse"]["error"], "CODEX_NO_ASSISTANT_MESSAGE")

    def test_codex_timeout_is_classified_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            def fake_timeout(cmd, *, input, capture_output, text, timeout, env):
                calls.append(list(cmd))
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="", stderr="still running")

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=fake_timeout)

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertEqual(result["parse"]["error"], "CODEX_TIMEOUT")
            self.assertEqual(len(calls), 1)
            self.assertFalse(paths.action_receipt.exists())

    def test_codex_auth_or_sandbox_failure_is_classified_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_failed(calls, stderr="not authenticated; login required"),
            )

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertEqual(result["parse"]["error"], "CODEX_AUTH_OR_SANDBOX_FAILURE")
            self.assertEqual(len(calls), 1)
            self.assertFalse(paths.action_receipt.exists())

    def test_unsupported_codex_preflight_queues_without_full_wake(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            codex_calls: list[list[str]] = []
            preflight_calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={
                    "QR_GUARDIAN_WAKE_CODEX_PREFLIGHT": "1",
                    "QR_GUARDIAN_WAKE_CODEX_BIN": "/opt/homebrew/bin/codex",
                },
                subprocess_run=_fake_codex(codex_calls, _valid_receipt()),
                codex_preflight_run=_fake_preflight(preflight_calls, stdout="codex-cli 0.25.0\n"),
            )

            self.assertEqual(result["status"], "CODEX_MODEL_UNSUPPORTED")
            self.assertEqual(result["parse"]["error"], "CODEX_MODEL_UNSUPPORTED")
            self.assertEqual(codex_calls, [])
            self.assertEqual(preflight_calls, [["/opt/homebrew/bin/codex", "--version"]])
            self.assertFalse(paths.action_receipt.exists())
            self.assertTrue(paths.action_review.exists())
            self.assertTrue(result["queued_for_active_trader"])
            preflight = result["codex_preflight"]
            self.assertEqual(preflight["codex_binary_path"], "/opt/homebrew/bin/codex")
            self.assertEqual(preflight["codex_version"], "0.25.0")
            self.assertEqual(preflight["requested_model"], "gpt-5.5")
            self.assertIn("QR_GUARDIAN_WAKE_CODEX_BIN", preflight["remediation_hint"])

            state = json.loads(paths.dispatcher_state.read_text())
            self.assertEqual(state["last_status"], "CODEX_MODEL_UNSUPPORTED")
            self.assertEqual(state["last_result"]["codex_preflight"]["codex_version"], "0.25.0")
            escalation = json.loads(paths.escalation.read_text())
            self.assertTrue(escalation["queued_for_active_trader"])

    def test_unsupported_codex_model_error_is_not_schema_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_failed(
                    calls,
                    stderr="gpt-5.5 model requires a newer version of Codex",
                ),
            )

            self.assertEqual(result["status"], "CODEX_MODEL_UNSUPPORTED")
            self.assertEqual(result["parse"]["error"], "CODEX_MODEL_UNSUPPORTED")
            self.assertNotEqual(result["parse"]["error"], "SCHEMA_INVALID")
            self.assertEqual(len(calls), 1)
            self.assertEqual(result["codex"]["requested_model"], "gpt-5.5")
            self.assertIn("newer version", result["codex"]["raw_stderr_excerpt"])
            self.assertIn("QR_GUARDIAN_WAKE_CODEX_BIN", result["codex"]["remediation_hint"])
            self.assertFalse(paths.action_receipt.exists())

    def test_supported_preflight_records_codex_version_and_writes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            codex_calls: list[list[str]] = []
            preflight_calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={
                    "QR_GUARDIAN_WAKE_CODEX_PREFLIGHT": "1",
                    "QR_GUARDIAN_WAKE_CODEX_BIN": "/Applications/Codex.app/Contents/Resources/codex",
                },
                subprocess_run=_fake_codex(codex_calls, _valid_receipt()),
                codex_preflight_run=_fake_preflight(preflight_calls, stdout="codex-cli 0.142.4\n"),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertNotIn(result["status"], {"CODEX_MODEL_UNSUPPORTED", "CODEX_CLI_VERSION_UNSUPPORTED"})
            self.assertEqual(preflight_calls, [["/Applications/Codex.app/Contents/Resources/codex", "--version"]])
            self.assertEqual(len(codex_calls), 1)
            self.assertEqual(codex_calls[0][0], "/Applications/Codex.app/Contents/Resources/codex")
            self.assertEqual(result["codex_preflight"]["codex_version"], "0.142.4")
            self.assertEqual(result["codex"]["codex_version"], "0.142.4")
            self.assertTrue(paths.action_receipt.exists())

    def test_supported_preflight_valid_json_ignores_stale_unsupported_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            codex_calls: list[list[str]] = []
            preflight_calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={
                    "QR_GUARDIAN_WAKE_CODEX_PREFLIGHT": "1",
                    "QR_GUARDIAN_WAKE_CODEX_BIN": "/Applications/Codex.app/Contents/Resources/codex",
                },
                subprocess_run=_fake_codex_with_diagnostics(
                    codex_calls,
                    _valid_receipt(),
                    stderr="old session diagnostic: gpt-5.5 model requires a newer version of Codex",
                    returncode=0,
                ),
                codex_preflight_run=_fake_preflight(preflight_calls, stdout="codex-cli 0.142.4\n"),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(result["codex"]["status"], "OK")
            self.assertTrue(paths.action_receipt.exists())

    def test_old_session_unsupported_text_does_not_poison_new_valid_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            stale_dir = paths.codex_home / "sessions" / "2026" / "06" / "29"
            stale_dir.mkdir(parents=True, exist_ok=True)
            (stale_dir / "rollout-old.jsonl").write_text(
                json.dumps(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "gpt-5.5 model requires a newer version of Codex",
                    }
                )
                + "\n"
            )
            codex_calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(codex_calls, _valid_receipt()))

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(result["codex"]["last_message_source"], "output-last-message")
            self.assertTrue(paths.action_receipt.exists())

    def test_current_cli_version_failure_still_classifies_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_failed(
                    calls,
                    stderr="codex cli version unsupported; too old, please upgrade",
                ),
            )

            self.assertEqual(result["status"], "CODEX_CLI_VERSION_UNSUPPORTED")
            self.assertEqual(result["parse"]["error"], "CODEX_CLI_VERSION_UNSUPPORTED")
            self.assertFalse(paths.action_receipt.exists())

    def test_one_repair_retry_occurs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["not json", "still not json"]),
            )

            self.assertEqual(result["status"], "PARSE_FAILED")
            self.assertTrue(result["repair_attempted"])
            self.assertEqual([attempt["attempt"] for attempt in result["codex_attempts"]], ["initial", "repair"])
            self.assertEqual(len(calls), 2)

    def test_invalid_after_retry_does_not_create_action_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            paths.action_receipt.write_text('{"status":"STALE"}\n')
            calls: list[list[str]] = []

            run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["not json", "still not json"]),
            )

            self.assertFalse(paths.action_receipt.exists())

    def test_parse_failure_for_new_event_keeps_unrelated_accepted_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="HOLD")),
            )
            _write_wake(paths, severity="P0", reasons=("SEVERITY_INCREASE",))

            result = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=1),
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["not json", "still not json"]),
            )

            self.assertEqual(result["status"], "PARSE_FAILED")
            payload = json.loads(paths.action_receipt.read_text())
            self.assertEqual(payload["receipt"]["action"], "HOLD")
            self.assertEqual(payload["receipt_lifecycle"], "ACTIVE")
            archive_dir = paths.action_receipt.parent / "guardian_action_receipts"
            self.assertTrue(any(archive_dir.glob("*.json")))
            review = paths.action_review.read_text()
            self.assertIn("Dispatcher status: `PARSE_FAILED`", review)
            self.assertIn("Action: `HOLD`", review)

    def test_valid_retry_creates_guardian_action_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["not json", _valid_receipt()]),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertTrue(result["repair_attempted"])
            self.assertEqual(len(calls), 2)
            self.assertTrue(paths.action_receipt.exists())

    def test_empty_output_retries_once_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["", _valid_receipt()]),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(calls), 2)
            self.assertTrue(paths.action_receipt.exists())

    def test_stale_broker_snapshot_refreshes_before_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            stale = json.loads(paths.broker_snapshot.read_text())
            stale["fetched_at_utc"] = (NOW - timedelta(minutes=10)).isoformat()
            paths.broker_snapshot.write_text(json.dumps(stale))
            codex_calls: list[list[str]] = []
            refresh_calls: list[list[str]] = []

            def fake_refresh(cmd, *, cwd, capture_output, text, timeout, env):
                refresh_calls.append(list(cmd))
                fresh = dict(stale)
                fresh["fetched_at_utc"] = NOW.isoformat()
                paths.broker_snapshot.write_text(json.dumps(fresh))
                return SimpleNamespace(returncode=0, stdout="refreshed", stderr="")

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(codex_calls, _valid_receipt()),
                snapshot_refresh_run=fake_refresh,
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertEqual(len(refresh_calls), 1)
            self.assertEqual(len(codex_calls), 1)
            self.assertTrue(result["broker_snapshot_freshness"]["fresh"])

    def test_stale_broker_snapshot_queues_without_prompt_when_refresh_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            stale = json.loads(paths.broker_snapshot.read_text())
            stale["fetched_at_utc"] = (NOW - timedelta(minutes=10)).isoformat()
            paths.broker_snapshot.write_text(json.dumps(stale))
            codex_calls: list[list[str]] = []

            def fake_refresh(cmd, *, cwd, capture_output, text, timeout, env):
                return SimpleNamespace(returncode=1, stdout="", stderr="missing read credentials")

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(codex_calls, _valid_receipt()),
                snapshot_refresh_run=fake_refresh,
            )

            self.assertEqual(result["status"], "BROKER_SNAPSHOT_STALE")
            self.assertTrue(result["queued_for_active_trader"])
            self.assertEqual(codex_calls, [])
            self.assertFalse(paths.action_receipt.exists())

    def test_active_lock_after_invalid_output_queues_instead_of_retrying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            def fake(cmd, *, input, capture_output, text, timeout, env):
                calls.append(list(cmd))
                out_path = Path(cmd[cmd.index("--output-last-message") + 1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text("not json")
                paths.live_lock.mkdir(parents=True)
                (paths.live_lock / "pid").write_text(str(os.getpid()))
                (paths.live_lock / "command").write_text("run-autotrade-live")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=fake)

            self.assertEqual(result["status"], "QUEUED_FOR_ACTIVE_TRADER")
            self.assertEqual(len(calls), 1)
            self.assertFalse(paths.action_receipt.exists())

    def test_repeated_parse_failure_is_queued_for_active_trader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []
            first = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["not json", "still not json"]),
            )

            second = run_dispatcher(
                paths=paths,
                now=NOW + timedelta(minutes=1),
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(first["status"], "PARSE_FAILED")
            self.assertEqual(second["status"], "QUEUED_FOR_ACTIVE_TRADER")
            self.assertEqual(len(calls), 2)

    def test_valid_gpt_output_creates_guardian_action_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            payload = json.loads(paths.action_receipt.read_text())
            self.assertEqual(payload["status"], "ACCEPTED")
            self.assertEqual(payload["receipt_status"], "ACCEPTED")
            self.assertEqual(payload["receipt_lifecycle"], "ACTIVE")
            self.assertIn("expires_at_utc", payload)
            self.assertFalse(payload["consumed_by_trader"])
            self.assertEqual(payload["receipt"]["action"], "TRADE")
            self.assertEqual(payload["receipt"]["dedupe_key"], payload["selected_event"]["dedupe_key"])
            self.assertTrue(payload["receipt"]["gateway_required"])
            self.assertTrue(payload["receipt"]["no_direct_oanda"])

    def test_selected_event_aud_jpy_receipt_usd_cad_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            aud = _event(
                severity="P1",
                event_id="event-aud",
                pair="AUD_JPY",
                direction="LONG",
                dedupe_key="AUD_JPY|thesis|FAILED_ACCEPTANCE|TRADE",
            )
            usd = _event(
                severity="P1",
                event_id="event-usd",
                pair="USD_CAD",
                direction="SHORT",
                dedupe_key="USD_CAD|thesis|FAILED_ACCEPTANCE|TRADE",
            )
            paths.events.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "events": [aud, usd]}))
            paths.escalation.write_text(
                json.dumps(
                    {
                        "generated_at_utc": NOW.isoformat(),
                        "wake_gpt": True,
                        "events_to_review": [
                            {**aud, "wake_reason_codes": ["NEW_EVENT"]},
                            {**usd, "wake_reason_codes": ["NEW_EVENT"]},
                        ],
                    }
                )
            )
            paths.action_receipt.write_text('{"status":"STALE"}\n')
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(
                    calls,
                    _valid_receipt(
                        event_id="event-usd",
                        pair="USD_CAD",
                        side="SHORT",
                        dedupe_key="USD_CAD|thesis|FAILED_ACCEPTANCE|TRADE",
                    ),
                ),
            )

            self.assertEqual(result["status"], "RECEIPT_EVENT_MISMATCH")
            self.assertFalse(paths.action_receipt.exists())
            self.assertTrue(result["queued_for_active_trader"])
            self.assertIn("RECEIPT_EVENT_MISMATCH", paths.action_review.read_text())

    def test_selected_event_event_id_mismatch_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(event_id="event-other")),
            )

            self.assertEqual(result["status"], "RECEIPT_EVENT_MISMATCH")
            self.assertFalse(paths.action_receipt.exists())

    def test_selected_event_pair_and_event_id_match_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(event_id="event-P1", pair="EUR_USD", side="LONG")),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertTrue(paths.action_receipt.exists())

    def test_selected_event_side_mismatch_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(event_id="event-P1", pair="EUR_USD", side="SHORT")),
            )

            self.assertEqual(result["status"], "RECEIPT_EVENT_MISMATCH")
            self.assertFalse(paths.action_receipt.exists())

    def test_selected_event_dedupe_mismatch_rejects_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(dedupe_key="WRONG|DEDUPE")),
            )

            self.assertEqual(result["status"], "RECEIPT_EVENT_MISMATCH")
            self.assertFalse(paths.action_receipt.exists())

    def test_accepted_receipt_writes_matching_review_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            payload = json.loads(paths.action_receipt.read_text())
            review = paths.action_review.read_text()
            selected = payload["selected_event"]
            self.assertEqual(payload["dispatcher_status"], result["status"])
            self.assertEqual(payload["selected_event_id"], selected["event_id"])
            self.assertEqual(payload["selected_event_dedupe_key"], selected["dedupe_key"])
            self.assertEqual(payload["receipt"]["event_id"], selected["event_id"])
            self.assertEqual(payload["receipt"]["dedupe_key"], selected["dedupe_key"])
            self.assertIn(f"Generated at: `{payload['generated_at_utc']}`", review)
            self.assertIn(f"Event id: `{selected['event_id']}`", review)
            self.assertIn(f"Dedupe key: `{selected['dedupe_key']}`", review)
            self.assertIn("Receipt exists: `yes`", review)
            self.assertIn("Dispatcher status: `RECEIPT_WRITTEN`", review)

    def test_valid_no_action_output_preserves_guardian_action_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="NO_ACTION")),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            payload = json.loads(paths.action_receipt.read_text())
            receipt = payload["receipt"]
            self.assertEqual(receipt["action"], "NO_ACTION")
            for field in (
                "event_id",
                "pair",
                "side",
                "thesis_state",
                "reason",
                "invalidation_evidence",
                "harvest_trigger",
                "margin_state",
                "ownership",
            ):
                self.assertIn(field, receipt)
                self.assertNotEqual(receipt[field], "")
            self.assertTrue(receipt["gateway_required"])
            self.assertTrue(receipt["no_direct_oanda"])

    def test_valid_hold_output_creates_guardian_action_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex(calls, _valid_receipt(action="HOLD")),
            )

            self.assertEqual(result["status"], "RECEIPT_WRITTEN")
            self.assertTrue(paths.action_receipt.exists())
            payload = json.loads(paths.action_receipt.read_text())
            self.assertEqual(payload["status"], "ACCEPTED")
            self.assertEqual(payload["receipt"]["action"], "HOLD")
            self.assertEqual(result["gateway_handoff"]["status"], "SKIPPED_DEFAULT_OFF")

    def test_default_gateway_handoff_off_prevents_immediate_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            self.assertEqual(result["gateway_handoff"]["status"], "SKIPPED_DEFAULT_OFF")
            self.assertEqual(len(calls), 1)

    def test_enabled_gateway_handoff_requires_action_execute_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={"QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1", "QR_LIVE_ENABLED": "1"},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(result["gateway_handoff"]["required_gateway"], "LiveOrderGateway")
            self.assertEqual(result["gateway_handoff"]["status"], "SKIPPED_ACTION_EXECUTE_DISABLED")
            self.assertEqual(len(calls), 1)

    def test_all_guardian_execute_flags_call_action_cycle_cli_after_accepted_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            codex_calls: list[list[str]] = []
            action_calls: list[list[str]] = []

            def fake_action_cycle(cmd, **kwargs):
                action_calls.append(list(cmd))
                return SimpleNamespace(
                    returncode=0,
                    stdout='{"status":"VERIFIED_NO_SEND","executed":false}\n',
                    stderr="",
                )

            with patch("tools.guardian_wake_dispatcher.subprocess.run", side_effect=fake_action_cycle):
                result = run_dispatcher(
                    paths=paths,
                    now=NOW,
                    env={
                        "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1",
                        "QR_GUARDIAN_ACTION_EXECUTE": "1",
                        "QR_LIVE_ENABLED": "1",
                    },
                    subprocess_run=_fake_codex(codex_calls, _valid_receipt()),
                )

            self.assertEqual(result["gateway_handoff"]["status"], "ACTION_CYCLE_CALLED")
            self.assertEqual(action_calls, [["python3", "-m", "quant_rabbit.cli", "guardian-action-cycle"]])
            self.assertEqual(len(codex_calls), 1)


def _fixture(
    root: Path,
    *,
    wake: bool = True,
    severity: str = "P1",
    reasons: tuple[str, ...] = ("NEW_EVENT",),
) -> DispatcherPaths:
    live_root = root / "live"
    live_root.mkdir(parents=True)
    paths = DispatcherPaths.from_root(root, live_root=live_root)
    for directory in (root / "data", root / "docs", root / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    paths.prompt_template.write_text("Return JSON only.\n")
    paths.daily_target_state.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "mode": "PURSUE_TARGET"}))
    paths.event_report.write_text("# Guardian Event Report\n")
    paths.event_state.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "events": {}}))
    paths.broker_snapshot.write_text(
        json.dumps(
            {
                "fetched_at_utc": NOW.isoformat(),
                "account": {"nav_jpy": 200000, "margin_available_jpy": 150000},
                "positions": [],
                "orders": [],
                "quotes": {"EUR_USD": {"bid": 1.17, "ask": 1.1701}},
            }
        )
    )
    if wake:
        _write_wake(paths, severity=severity, reasons=reasons)
    else:
        paths.events.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "events": []}))
        paths.escalation.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "wake_gpt": False}))
    return paths


def _write_wake(paths: DispatcherPaths, *, severity: str, reasons: tuple[str, ...]) -> None:
    event = _event(severity=severity)
    review_event = {**event, "wake_reason_codes": list(reasons)}
    paths.events.write_text(json.dumps({"generated_at_utc": NOW.isoformat(), "events": [event]}))
    paths.escalation.write_text(
        json.dumps(
            {
                "generated_at_utc": NOW.isoformat(),
                "wake_gpt": True,
                "model_target": "GPT-5.5",
                "wake_reason_codes": list(reasons),
                "events_to_review": [review_event],
            }
        )
    )


def _event(
    *,
    severity: str,
    event_id: str | None = None,
    pair: str = "EUR_USD",
    direction: str = "LONG",
    dedupe_key: str | None = None,
) -> dict:
    return _event_payload(
        severity=severity,
        event_id=event_id,
        pair=pair,
        direction=direction,
        dedupe_key=dedupe_key,
    )


def _event_payload(
    *,
    severity: str,
    event_id: str | None = None,
    pair: str = "EUR_USD",
    direction: str = "LONG",
    dedupe_key: str | None = None,
) -> dict:
    return {
        "event_id": event_id or f"event-{severity}",
        "event_type": "FAILED_ACCEPTANCE",
        "pair": pair,
        "direction": direction,
        "thesis": "major figure rejection",
        "price_zone": f"{pair} 1.1700 rejection",
        "severity": severity,
        "recommended_review_type": "ENTRY_REVIEW",
        "dedupe_key": dedupe_key or f"{pair}|major_figure_rejection|FAILED_ACCEPTANCE|TRADE",
        "action_hint": "TRADE",
        "thesis_state": "ALIVE",
        "detected_at_utc": NOW.isoformat(),
        "details": {},
    }


def _valid_receipt(
    *,
    event_id: str = "event-P1",
    action: str = "TRADE",
    pair: str = "EUR_USD",
    side: str = "LONG",
    dedupe_key: str | None = None,
) -> str:
    payload = {
        "action": action,
        "event_id": event_id,
        "new_information": True,
        "pair": pair,
        "side": side,
        "thesis_state": "ALIVE",
        "reason": "fresh failed acceptance at the major figure",
        "invalidation_evidence": "accepted trade back below failed-acceptance support",
        "invalidation": "accepted trade back below failed-acceptance support",
        "harvest_trigger": "upper range rail",
        "margin_state": "margin available and no active position",
        "ownership": "system",
        "gateway_required": True,
        "no_direct_oanda": True,
    }
    if dedupe_key is not None:
        payload["dedupe_key"] = dedupe_key
    return json.dumps(payload)


def _fake_codex(calls: list[list[str]], output: str):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _fake_codex_stdout(calls: list[list[str]], stdout: str):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    return run


def _fake_codex_explicit(paths: DispatcherPaths, calls: list[list[str]], output: str):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        paths.codex_explicit_output.parent.mkdir(parents=True, exist_ok=True)
        paths.codex_explicit_output.write_text(output)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _fake_codex_session(paths: DispatcherPaths, calls: list[list[str]], assistant_text: str):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        session_dir = paths.codex_home / "sessions" / "2026" / "06" / "30"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_path = session_dir / f"rollout-{len(calls)}.jsonl"
        session_path.write_text(
            json.dumps({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": assistant_text}]})
            + "\n"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _fake_codex_session_without_assistant(paths: DispatcherPaths, calls: list[list[str]]):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        session_dir = paths.codex_home / "sessions" / "2026" / "06" / "30"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_path = session_dir / f"rollout-no-assistant-{len(calls)}.jsonl"
        session_path.write_text(json.dumps({"type": "message", "role": "user", "content": "prompt"}) + "\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _fake_codex_failed(calls: list[list[str]], *, stderr: str):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        return SimpleNamespace(returncode=1, stdout="", stderr=stderr)

    return run


def _fake_codex_with_diagnostics(
    calls: list[list[str]],
    output: str,
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
):
    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return run


def _fake_preflight(calls: list[list[str]], *, stdout: str = "", stderr: str = "", returncode: int = 0):
    def run(cmd, *, capture_output, text, timeout, env):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return run


def _fake_codex_sequence(calls: list[list[str]], outputs: list[str]):
    remaining = list(outputs)

    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        output = remaining.pop(0) if remaining else outputs[-1]
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _fake_codex_stdout_sequence(calls: list[list[str]], outputs: list[str]):
    remaining = list(outputs)

    def run(cmd, *, input, capture_output, text, timeout, env):
        calls.append(list(cmd))
        output = remaining.pop(0) if remaining else outputs[-1]
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    return run


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
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
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            self.assertEqual(result["status"], "NO_WAKE")
            self.assertEqual(calls, [])
            self.assertFalse(paths.action_receipt.exists())

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
            calls: list[list[str]] = []

            run_dispatcher(
                paths=paths,
                now=NOW,
                env={},
                subprocess_run=_fake_codex_sequence(calls, ["not json", "still not json"]),
            )

            self.assertFalse(paths.action_receipt.exists())

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
            self.assertEqual(payload["receipt"]["action"], "TRADE")
            self.assertEqual(payload["receipt"]["dedupe_key"], payload["selected_event"]["dedupe_key"])
            self.assertTrue(payload["receipt"]["gateway_required"])
            self.assertTrue(payload["receipt"]["no_direct_oanda"])

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


def _event(*, severity: str) -> dict:
    return {
        "event_id": f"event-{severity}",
        "event_type": "FAILED_ACCEPTANCE",
        "pair": "EUR_USD",
        "direction": "LONG",
        "thesis": "major figure rejection",
        "price_zone": "EUR_USD 1.1700 rejection",
        "severity": severity,
        "recommended_review_type": "ENTRY_REVIEW",
        "dedupe_key": "EUR_USD|major_figure_rejection|FAILED_ACCEPTANCE|TRADE",
        "action_hint": "TRADE",
        "thesis_state": "ALIVE",
        "detected_at_utc": NOW.isoformat(),
        "details": {},
    }


def _valid_receipt(*, event_id: str = "event-P1") -> str:
    return json.dumps(
        {
            "action": "TRADE",
            "event_id": event_id,
            "new_information": True,
            "pair": "EUR_USD",
            "side": "LONG",
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
    )


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


if __name__ == "__main__":
    unittest.main()

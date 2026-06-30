from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from tools.guardian_wake_dispatcher import DispatcherPaths, run_dispatcher


NOW = datetime(2026, 6, 30, 4, 0, tzinfo=timezone.utc)


class GuardianWakeDispatcherTest(unittest.TestCase):
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

    def test_valid_gpt_output_creates_guardian_action_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            payload = json.loads(paths.action_receipt.read_text())
            self.assertEqual(payload["status"], "ACCEPTED")
            self.assertEqual(payload["receipt"]["action"], "TRADE")
            self.assertTrue(payload["receipt"]["gateway_required"])
            self.assertTrue(payload["receipt"]["no_direct_oanda"])

    def test_default_gateway_handoff_off_prevents_immediate_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _fixture(Path(tmp))
            calls: list[list[str]] = []

            result = run_dispatcher(paths=paths, now=NOW, env={}, subprocess_run=_fake_codex(calls, _valid_receipt()))

            self.assertEqual(result["gateway_handoff"]["status"], "SKIPPED_DEFAULT_OFF")
            self.assertEqual(len(calls), 1)

    def test_enabled_gateway_handoff_still_requires_live_order_gateway(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _fixture(root)
            gateway_file = root / "src" / "quant_rabbit" / "broker" / "execution.py"
            gateway_file.parent.mkdir(parents=True)
            gateway_file.write_text("class LiveOrderGateway: pass\n")
            calls: list[list[str]] = []

            result = run_dispatcher(
                paths=paths,
                now=NOW,
                env={"QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": "1", "QR_LIVE_ENABLED": "1"},
                subprocess_run=_fake_codex(calls, _valid_receipt()),
            )

            self.assertEqual(result["gateway_handoff"]["required_gateway"], "LiveOrderGateway")
            self.assertEqual(result["gateway_handoff"]["status"], "SKIPPED_NO_SAFE_EXISTING_CLI_ROUTE")
            self.assertEqual(len(calls), 1)


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


if __name__ == "__main__":
    unittest.main()

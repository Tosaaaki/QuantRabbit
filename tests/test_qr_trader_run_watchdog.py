from __future__ import annotations

import json
import os
import plistlib
import shutil
import subprocess
import tempfile
import unittest
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from quant_rabbit.qr_trader_run_watchdog import WatchdogPaths, run_watchdog


ROOT = Path(__file__).resolve().parents[1]
PLIST = ROOT / "scripts" / "guardian" / "com.quantrabbit.qr-trader-run-watchdog.plist"
WATCHDOG_SOURCE = ROOT / "src" / "quant_rabbit" / "qr_trader_run_watchdog.py"


class QRTraderRunWatchdogTest(unittest.TestCase):
    def test_active_recent_run_is_ok(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:52:00+00:00"))
            _write_journal(root, _dt("2026-07-01T02:51:00+00:00"))
            _write_memory(automation_dir, _dt("2026-07-01T02:53:00+00:00"))

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "OK")
            self.assertFalse(payload["missed_expected_window"])
            self.assertLess(payload["minutes_since_last_run"], 15)
            self.assertTrue(paths.output_json.exists())
            self.assertTrue(paths.output_report.exists())
            self.assertTrue(paths.output_log.exists())

    def test_active_stale_run_is_stale(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T01:30:00+00:00"))
            _write_journal(root, _dt("2026-07-01T01:29:00+00:00"))
            _write_memory(automation_dir, _dt("2026-07-01T01:31:00+00:00"))

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "STALE")
            self.assertTrue(payload["missed_expected_window"])
            self.assertIn("QR_TRADER_RUN_STALE", _issue_codes(payload))
            self.assertEqual(payload["severity"], "P0")

    def test_wrong_model_cadence_and_cwd_is_broken(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(
                automation_dir,
                model="gpt-5.4-mini",
                rrule="RRULE:FREQ=MINUTELY;INTERVAL=20",
                cwds=["/Users/tossaki/App/QuantRabbit"],
            )
            _write_decision(root, _dt("2026-07-01T02:58:00+00:00"))

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "BROKEN")
            codes = _issue_codes(payload)
            self.assertIn("QR_TRADER_AUTOMATION_WRONG_MODEL", codes)
            self.assertIn("QR_TRADER_AUTOMATION_WRONG_CADENCE", codes)
            self.assertIn("QR_TRADER_AUTOMATION_WRONG_CWD", codes)

    def test_active_guardian_reduce_receipt_not_consumed_before_expiry_is_p1(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T01:00:00+00:00"))
            _write_guardian_receipt(
                root,
                action="REDUCE",
                generated_at=_dt("2026-07-01T01:10:00+00:00"),
                expires_at=_dt("2026-07-01T02:25:00+00:00"),
                consumed=False,
            )

            payload = run_watchdog(paths=paths, now_utc=now)

            guardian_issues = payload["guardian_receipt"]["issues"]
            self.assertEqual(guardian_issues[0]["code"], "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER")
            self.assertEqual(guardian_issues[0]["severity"], "P1")
            self.assertTrue(payload["guardian_receipt"]["high_urgency_action"])

    def test_missing_evidence_is_unknown(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            _, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)

            payload = run_watchdog(paths=paths, now_utc=now)

            self.assertEqual(payload["status"], "UNKNOWN")
            self.assertIn("QR_TRADER_RUN_EVIDENCE_MISSING", _issue_codes(payload))

    def test_watchdog_never_calls_oanda_or_codex_by_default(self) -> None:
        now = _dt("2026-07-01T03:00:00+00:00")
        source = WATCHDOG_SOURCE.read_text()
        self.assertNotIn("Oanda", source)
        self.assertNotIn("requests.", source)
        with tempfile.TemporaryDirectory() as tmp:
            root, automation_dir, paths = _fixture(tmp, now=now)
            _write_automation(automation_dir)
            _write_decision(root, _dt("2026-07-01T02:59:00+00:00"))

            with mock.patch.object(urllib.request, "urlopen") as urlopen, mock.patch.object(
                subprocess, "run"
            ) as subprocess_run:
                payload = run_watchdog(paths=paths, now_utc=now, env={"QR_TRADER_WATCHDOG_CAN_WAKE": "0"})

            urlopen.assert_not_called()
            subprocess_run.assert_not_called()
            self.assertFalse(payload["execution_boundary"]["calls_oanda"])
            self.assertFalse(payload["execution_boundary"]["runs_codex_by_default"])
            self.assertEqual(payload["environment"]["QR_TRADER_WATCHDOG_CAN_WAKE"], "0")

    def test_launchd_plist_has_safe_defaults_and_lints_when_available(self) -> None:
        payload = plistlib.loads(PLIST.read_bytes())

        self.assertEqual(payload["Label"], "com.quantrabbit.qr-trader-run-watchdog")
        self.assertEqual(payload["WorkingDirectory"], "/Users/tossaki/App/QuantRabbit-live")
        self.assertEqual(payload["StartInterval"], 300)
        self.assertEqual(
            payload["StandardOutPath"],
            "/Users/tossaki/App/QuantRabbit-live/logs/qr_trader_run_watchdog.launchd.log",
        )
        self.assertEqual(
            payload["StandardErrorPath"],
            "/Users/tossaki/App/QuantRabbit-live/logs/qr_trader_run_watchdog.launchd.err",
        )
        command = " ".join(payload["ProgramArguments"])
        self.assertIn("tools/qr_trader_run_watchdog.py", command)
        self.assertIn("/Users/tossaki/App/QuantRabbit-live", command)
        env = payload["EnvironmentVariables"]
        self.assertEqual(env["QR_TRADER_WATCHDOG_CAN_WAKE"], "0")
        self.assertEqual(env["CODEX_DISABLE_UPDATE_CHECK"], "1")

        if shutil.which("plutil"):
            result = subprocess.run(["plutil", "-lint", str(PLIST)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


def _fixture(tmp: str, *, now: datetime) -> tuple[Path, Path, WatchdogPaths]:
    root = Path(tmp) / "live"
    automation_dir = Path(tmp) / "codex" / "automations" / "qr-trader"
    for rel in ("data", "docs", "logs"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    automation_dir.mkdir(parents=True, exist_ok=True)
    paths = WatchdogPaths.from_root(
        root,
        automation_dir=automation_dir,
        codex_logs=Path(tmp) / "codex" / "logs_2.sqlite",
    )
    return root, automation_dir, paths


def _write_automation(
    automation_dir: Path,
    *,
    status: str = "ACTIVE",
    model: str = "gpt-5.5",
    reasoning_effort: str = "high",
    rrule: str = "RRULE:FREQ=MINUTELY;INTERVAL=60;BYDAY=MO,TU,WE,TH,FR,SA",
    cwds: list[str] | None = None,
) -> None:
    cwds = cwds or ["/Users/tossaki/App/QuantRabbit-live"]
    automation_dir.joinpath("automation.toml").write_text(
        "\n".join(
            [
                'id = "qr-trader"',
                f'status = "{status}"',
                f'rrule = "{rrule}"',
                f'model = "{model}"',
                f'reasoning_effort = "{reasoning_effort}"',
                "cwds = [" + ", ".join(json.dumps(item) for item in cwds) + "]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_decision(root: Path, generated_at: datetime) -> None:
    _write_json(root / "data" / "codex_trader_decision_response.json", {"generated_at_utc": generated_at.isoformat(), "action": "WAIT"})
    (root / "docs" / "gpt_trader_decision_report.md").write_text(
        f"# GPT Trader Decision Report\n\n- Generated at UTC: `{generated_at.isoformat()}`\n- Status: `ACCEPTED`\n",
        encoding="utf-8",
    )
    (root / "docs" / "autotrade_cycle_report.md").write_text(
        f"# Autotrade Cycle Report\n\n- Generated at UTC: `{generated_at.isoformat()}`\n- Status: `NO_LIVE_READY_INTENT`\n",
        encoding="utf-8",
    )


def _write_journal(root: Path, generated_at: datetime) -> None:
    journal = root / "logs" / "trader_journal.jsonl"
    journal.write_text(json.dumps({"ts": generated_at.isoformat(), "status": "NO_LIVE_READY_INTENT"}) + "\n")


def _write_memory(automation_dir: Path, generated_at: datetime) -> None:
    memory = automation_dir / "memory.md"
    memory.write_text(f"# memory\n\n- {generated_at.isoformat()} trader cycle complete\n", encoding="utf-8")
    os.utime(memory, (generated_at.timestamp(), generated_at.timestamp()))


def _write_guardian_receipt(
    root: Path,
    *,
    action: str,
    generated_at: datetime,
    expires_at: datetime,
    consumed: bool,
) -> None:
    _write_json(
        root / "data" / "guardian_action_receipt.json",
        {
            "receipt_status": "ACCEPTED",
            "receipt_lifecycle": "ACTIVE",
            "generated_at_utc": generated_at.isoformat(),
            "expires_at_utc": expires_at.isoformat(),
            "consumed_by_trader": consumed,
            "receipt": {"action": action},
        },
    )
    (root / "docs" / "guardian_action_review.md").write_text(
        "# Guardian Action Review\n\n- Status: `RECEIPT_WRITTEN`\n", encoding="utf-8"
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _issue_codes(payload: dict[str, object]) -> set[str]:
    return {str(item["code"]) for item in payload["issues"] if isinstance(item, dict)}


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


if __name__ == "__main__":
    unittest.main()

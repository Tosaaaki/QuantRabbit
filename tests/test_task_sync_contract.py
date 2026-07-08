from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import check_task_sync  # noqa: E402


def _valid_qr_trader_prompt() -> str:
    return " ".join(
        (
            *check_task_sync.EXPECTED_QR_TRADER_GUARDIAN_STARTUP_READS,
            *check_task_sync.EXPECTED_QR_TRADER_RUNTIME_DRIFT_PROMPT_PHRASES,
        )
    )


class TaskSyncContractTest(unittest.TestCase):
    def test_qr_trader_expected_runtime_policy_is_hourly_gpt55_high(self) -> None:
        self.assertEqual(
            check_task_sync.EXPECTED_QR_TRADER_RRULE,
            "RRULE:FREQ=MINUTELY;INTERVAL=60;BYDAY=MO,TU,WE,TH,FR,SA",
        )
        self.assertEqual(check_task_sync.EXPECTED_QR_TRADER_MODEL, "gpt-5.5")
        self.assertEqual(check_task_sync.EXPECTED_QR_TRADER_REASONING, "high")

    def test_qr_trader_automation_validator_accepts_hourly_gpt55_high(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            automation = Path(tmp) / "automation.toml"
            prompt = _valid_qr_trader_prompt()
            automation.write_text(
                "\n".join(
                    [
                        "version = 1",
                        'id = "qr-trader"',
                        'kind = "cron"',
                        'name = "QR vNext Trader"',
                        f'prompt = "{prompt}"',
                        'status = "ACTIVE"',
                        'rrule = "RRULE:FREQ=MINUTELY;INTERVAL=60;BYDAY=MO,TU,WE,TH,FR,SA"',
                        'model = "gpt-5.5"',
                        'reasoning_effort = "high"',
                        'execution_environment = "local"',
                        'cwds = ["/Users/tossaki/App/QuantRabbit-live"]',
                    ]
                )
                + "\n"
            )
            original = check_task_sync.QR_TRADER_AUTOMATION_PATH
            check_task_sync.QR_TRADER_AUTOMATION_PATH = automation
            try:
                issues: list[str] = []
                check_task_sync._validate_qr_trader_automation(issues)
            finally:
                check_task_sync.QR_TRADER_AUTOMATION_PATH = original

            self.assertEqual(issues, [])

    def test_qr_trader_automation_validator_accepts_weekend_paused_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            automation = Path(tmp) / "automation.toml"
            weekend_state = Path(tmp) / "weekend-state.json"
            prompt = _valid_qr_trader_prompt()
            automation.write_text(
                "\n".join(
                    [
                        "version = 1",
                        'id = "qr-trader"',
                        'kind = "cron"',
                        'name = "QR vNext Trader"',
                        f'prompt = "{prompt}"',
                        'status = "PAUSED"',
                        'rrule = "RRULE:FREQ=MINUTELY;INTERVAL=60;BYDAY=MO,TU,WE,TH,FR,SA"',
                        'model = "gpt-5.5"',
                        'reasoning_effort = "high"',
                        'execution_environment = "local"',
                        'cwds = ["/Users/tossaki/App/QuantRabbit-live"]',
                    ]
                )
                + "\n"
            )
            weekend_state.write_text(
                json.dumps({"mode": "paused", "managed_task_keys": ["codex:qr-trader"]}) + "\n",
                encoding="utf-8",
            )
            original_automation = check_task_sync.QR_TRADER_AUTOMATION_PATH
            original_weekend = check_task_sync.QR_WEEKEND_TASK_STATE_PATH
            check_task_sync.QR_TRADER_AUTOMATION_PATH = automation
            check_task_sync.QR_WEEKEND_TASK_STATE_PATH = weekend_state
            try:
                issues: list[str] = []
                check_task_sync._validate_qr_trader_automation(
                    issues,
                    now_utc=datetime(2026, 7, 3, 21, 30, tzinfo=timezone.utc),
                )
            finally:
                check_task_sync.QR_TRADER_AUTOMATION_PATH = original_automation
                check_task_sync.QR_WEEKEND_TASK_STATE_PATH = original_weekend

            self.assertEqual(issues, [])

    def test_qr_trader_automation_validator_rejects_midweek_paused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            automation = Path(tmp) / "automation.toml"
            weekend_state = Path(tmp) / "weekend-state.json"
            prompt = _valid_qr_trader_prompt()
            automation.write_text(
                "\n".join(
                    [
                        "version = 1",
                        'id = "qr-trader"',
                        'kind = "cron"',
                        'name = "QR vNext Trader"',
                        f'prompt = "{prompt}"',
                        'status = "PAUSED"',
                        'rrule = "RRULE:FREQ=MINUTELY;INTERVAL=60;BYDAY=MO,TU,WE,TH,FR,SA"',
                        'model = "gpt-5.5"',
                        'reasoning_effort = "high"',
                        'execution_environment = "local"',
                        'cwds = ["/Users/tossaki/App/QuantRabbit-live"]',
                    ]
                )
                + "\n"
            )
            weekend_state.write_text(
                json.dumps({"mode": "paused", "managed_task_keys": ["codex:qr-trader"]}) + "\n",
                encoding="utf-8",
            )
            original_automation = check_task_sync.QR_TRADER_AUTOMATION_PATH
            original_weekend = check_task_sync.QR_WEEKEND_TASK_STATE_PATH
            check_task_sync.QR_TRADER_AUTOMATION_PATH = automation
            check_task_sync.QR_WEEKEND_TASK_STATE_PATH = weekend_state
            try:
                issues: list[str] = []
                check_task_sync._validate_qr_trader_automation(
                    issues,
                    now_utc=datetime(2026, 7, 1, 3, 0, tzinfo=timezone.utc),
                )
            finally:
                check_task_sync.QR_TRADER_AUTOMATION_PATH = original_automation
                check_task_sync.QR_WEEKEND_TASK_STATE_PATH = original_weekend

            self.assertIn("qr-trader automation status expected 'ACTIVE', got 'PAUSED'", issues)

    def test_qr_trader_automation_validator_rejects_stale_runtime_drift_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            automation = Path(tmp) / "automation.toml"
            prompt = (
                " ".join(check_task_sync.EXPECTED_QR_TRADER_GUARDIAN_STARTUP_READS)
                + " docs/*_report.md docs/guardian_action_review.md data/guardian_trigger_contract.json "
                + "runtime drift and **do not** block the run"
            )
            automation.write_text(
                "\n".join(
                    [
                        "version = 1",
                        'id = "qr-trader"',
                        'kind = "cron"',
                        'name = "QR vNext Trader"',
                        f'prompt = "{prompt}"',
                        'status = "ACTIVE"',
                        'rrule = "RRULE:FREQ=MINUTELY;INTERVAL=60;BYDAY=MO,TU,WE,TH,FR,SA"',
                        'model = "gpt-5.5"',
                        'reasoning_effort = "high"',
                        'execution_environment = "local"',
                        'cwds = ["/Users/tossaki/App/QuantRabbit-live"]',
                    ]
                )
                + "\n"
            )
            original = check_task_sync.QR_TRADER_AUTOMATION_PATH
            check_task_sync.QR_TRADER_AUTOMATION_PATH = automation
            try:
                issues: list[str] = []
                check_task_sync._validate_qr_trader_automation(issues)
            finally:
                check_task_sync.QR_TRADER_AUTOMATION_PATH = original

            self.assertTrue(
                any("runtime drift allowance" in issue for issue in issues),
                issues,
            )

    def test_source_dirt_check_requires_explanation_for_guarded_source_files(self) -> None:
        dirty = [
            " M src/quant_rabbit/automation.py",
            " M docs/trader_decision_report.md",
            " M src/quant_rabbit/risk.py",
        ]

        self.assertEqual(
            check_task_sync.unexplained_source_dirt(
                dirty,
                {"src/quant_rabbit/automation.py": "committed prior guardian receipt gate hardening"},
            ),
            ["src/quant_rabbit/risk.py"],
        )
        self.assertEqual(
            check_task_sync.unexplained_source_dirt(
                dirty,
                {
                    "src/quant_rabbit/automation.py": "committed prior guardian receipt gate hardening",
                    "src/quant_rabbit/risk.py": "committed prior guardian receipt gate hardening",
                },
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()

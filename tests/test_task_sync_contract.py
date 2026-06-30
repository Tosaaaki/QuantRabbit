from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import check_task_sync  # noqa: E402


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
            automation.write_text(
                "\n".join(
                    [
                        "version = 1",
                        'id = "qr-trader"',
                        'kind = "cron"',
                        'name = "QR vNext Trader"',
                        'prompt = "test"',
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


if __name__ == "__main__":
    unittest.main()

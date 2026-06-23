from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.cli import main
from quant_rabbit.trader_repair_orchestrator import (
    STATUS_APPROVAL_REQUIRED,
    STATUS_READY,
    TraderRepairOrchestrator,
)
from quant_rabbit.trader_support_bot import (
    REPAIR_AUTOMATION_ALLOWED_ACTIONS,
    REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS,
    REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS,
)


class TraderRepairOrchestratorTest(unittest.TestCase):
    def test_builds_codex_repair_queue_without_live_side_effects(self) -> None:
        now = datetime(2026, 6, 23, 10, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                        priority="P0",
                        suggested_files=[
                            "src/quant_rabbit/profit_capture_bot.py",
                            "tests/test_profit_capture_bot.py",
                        ],
                        verification_commands=[
                            "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --max-events 80"
                        ],
                    ),
                    _request(
                        "RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT",
                        priority="P0",
                        requires_explicit_operator_approval=True,
                    ),
                ],
            )

            summary = TraderRepairOrchestrator(
                support_bot_path=support,
                output_path=output,
                report_path=report,
                trader_request="利確 bot を直して",
                now_utc=now,
            ).run()

            self.assertEqual(summary.status, STATUS_READY)
            payload = json.loads(output.read_text())
            self.assertTrue(payload["read_only"])
            self.assertEqual(payload["live_side_effects"], [])
            self.assertEqual(payload["selected_request"]["code"], "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY")
            self.assertEqual(
                payload["selected_request"]["automation_status"],
                "READY_FOR_CODEX_IMPLEMENTATION",
            )
            self.assertIn(
                "PYTHONPATH=src python3 -m unittest tests.test_profit_capture_bot -v",
                payload["selected_request"]["targeted_test_commands"],
            )
            contract = payload["execution_contract"]
            self.assertEqual(contract["codex_may_execute"], REPAIR_AUTOMATION_ALLOWED_ACTIONS)
            self.assertEqual(
                contract["requires_explicit_operator_approval_for"],
                REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS,
            )
            self.assertEqual(contract["forbidden_direct_actions"], REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS)
            self.assertFalse(contract["quant_rabbit_code_may_call_model_api"])
            self.assertIn("Order send", contract["orders_closes_launchd_policy"])
            self.assertIn("REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY", report.read_text())

    def test_only_approval_required_requests_return_diagnostic_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = root / "support.json"
            output = root / "orchestrator.json"
            report = root / "orchestrator.md"
            _write_support(
                support,
                [
                    _request(
                        "RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT",
                        priority="P0",
                        requires_explicit_operator_approval=True,
                    )
                ],
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "trader-repair-orchestrator",
                        "--trader-support-bot",
                        str(support),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                    ]
                )

            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], STATUS_APPROVAL_REQUIRED)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["actionable_requests"], [])
            self.assertEqual(payload["approval_required_requests"][0]["automation_status"], "WAITING_FOR_OPERATOR_APPROVAL")
            self.assertIn("launchd_load", payload["approval_required_requests"][0]["automation_contract"]["requires_explicit_operator_approval_for"])


def _write_support(path: Path, requests: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "SUPPORT_BLOCKED",
                "repair_requests": requests,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _request(
    code: str,
    *,
    priority: str,
    suggested_files: list[str] | None = None,
    verification_commands: list[str] | None = None,
    requires_explicit_operator_approval: bool = False,
) -> dict[str, object]:
    return {
        "code": code,
        "priority": priority,
        "status": "READY_FOR_CODE_REPAIR",
        "source_findings": [code.replace("REPAIR_", "")],
        "problem": f"{code} problem",
        "why_now": f"{code} why now",
        "clearance_conditions": [f"{code} clears"],
        "verification_commands": verification_commands or ["PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot"],
        "suggested_files": suggested_files or ["tests/test_trader_support_bot.py"],
        "required_tests": ["regression", "positive path"],
        "requires_explicit_operator_approval": requires_explicit_operator_approval,
        "automation_contract": {
            "codex_may_execute": REPAIR_AUTOMATION_ALLOWED_ACTIONS,
            "commit_and_live_sync_required": True,
            "quant_rabbit_code_may_call_model_api": False,
            "live_side_effects_allowed": [],
            "requires_explicit_operator_approval_for": REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS,
            "forbidden_direct_actions": REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS,
            "orders_closes_launchd_policy": "approval required",
        },
        "read_only": True,
        "live_side_effects": [],
    }

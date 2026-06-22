from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.cli import main
from quant_rabbit.profit_capture_bot import (
    STATUS_BLOCKED,
    STATUS_READY,
    STATUS_WATCH,
    ProfitCaptureBot,
)


class ProfitCaptureBotTest(unittest.TestCase):
    def test_bankable_when_attached_tp_progress_clears_noise(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            files = _write_fixture(Path(tmp), now=now, ask=161.660, m1_atr_pips=0.5)

            summary = ProfitCaptureBot(
                broker_snapshot_path=files["broker"],
                pair_charts_path=files["charts"],
                position_management_path=files["position_management"],
                position_guardian_management_path=files["guardian_management"],
                execution_timing_audit_path=files["timing"],
                output_path=files["output"],
                report_path=files["report"],
                now_utc=now,
            ).run()

            payload = json.loads(files["output"].read_text())
            self.assertEqual(summary.status, STATUS_READY)
            self.assertEqual(payload["metrics"]["bankable_positions"], 1)
            self.assertEqual(payload["metrics"]["historical_counterfactual_profit_capture_delta_jpy"], 446.04)
            self.assertEqual(payload["positions"][0]["gate_status"], "BANKABLE_NOW")
            self.assertGreater(payload["positions"][0]["tp_progress"], 0.3)
            self.assertEqual(payload["history"]["top_misses"][0]["counterfactual_jpy"], 105.84)
            self.assertEqual(payload["history"]["top_misses"][0]["counterfactual_delta_jpy"], 446.04)
            self.assertIn("Profit Capture Bot Report", files["report"].read_text())

    def test_watch_reports_trigger_when_position_is_not_profitable(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            files = _write_fixture(
                Path(tmp),
                now=now,
                ask=161.730,
                m1_atr_pips=0.5,
                historical_missed=0,
            )

            summary = ProfitCaptureBot(
                broker_snapshot_path=files["broker"],
                pair_charts_path=files["charts"],
                position_management_path=files["position_management"],
                position_guardian_management_path=files["guardian_management"],
                execution_timing_audit_path=files["timing"],
                output_path=files["output"],
                report_path=files["report"],
                now_utc=now,
            ).run()

            payload = json.loads(files["output"].read_text())
            self.assertEqual(summary.status, STATUS_WATCH)
            position = payload["positions"][0]
            self.assertEqual(position["gate_status"], "WATCH_NOT_PROFITABLE")
            self.assertEqual(position["capture_trigger"]["quote_side"], "ask")
            self.assertEqual(position["capture_trigger"]["comparator"], "<=")
            self.assertLess(position["capture_trigger"]["price"], 161.692)

    def test_blocks_when_capture_inputs_are_missing(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            files = _write_fixture(Path(tmp), now=now, ask=161.660, m1_atr_pips=None)

            summary = ProfitCaptureBot(
                broker_snapshot_path=files["broker"],
                pair_charts_path=files["charts"],
                position_management_path=files["position_management"],
                position_guardian_management_path=files["guardian_management"],
                execution_timing_audit_path=files["timing"],
                output_path=files["output"],
                report_path=files["report"],
                now_utc=now,
            ).run()

            payload = json.loads(files["output"].read_text())
            self.assertEqual(summary.status, STATUS_BLOCKED)
            self.assertIn("M1_ATR_MISSING", payload["positions"][0]["blocker_codes"])
            self.assertIn("PROFIT_CAPTURE_INPUT_MISSING", {item["code"] for item in payload["blockers"]})

    def test_blocks_when_required_artifact_is_missing(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            files = _write_fixture(Path(tmp), now=now, ask=161.660, m1_atr_pips=0.5)
            files["broker"].unlink()

            summary = ProfitCaptureBot(
                broker_snapshot_path=files["broker"],
                pair_charts_path=files["charts"],
                position_management_path=files["position_management"],
                position_guardian_management_path=files["guardian_management"],
                execution_timing_audit_path=files["timing"],
                output_path=files["output"],
                report_path=files["report"],
                now_utc=now,
            ).run()

            payload = json.loads(files["output"].read_text())
            self.assertEqual(summary.status, STATUS_BLOCKED)
            self.assertIn("BROKER_SNAPSHOT_MISSING", {item["code"] for item in payload["blockers"]})

    def test_cli_returns_error_code_distinct_from_blocked_diagnostic(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            files = _write_fixture(Path(tmp), now=now, ask=161.660, m1_atr_pips=None)
            files["broker"].write_text("{not-json", encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "profit-capture-bot",
                        "--broker-snapshot",
                        str(files["broker"]),
                        "--pair-charts",
                        str(files["charts"]),
                        "--position-management",
                        str(files["position_management"]),
                        "--position-guardian-management",
                        str(files["guardian_management"]),
                        "--execution-timing-audit",
                        str(files["timing"]),
                        "--output",
                        str(files["output"]),
                        "--report",
                        str(files["report"]),
                    ]
                )

            self.assertEqual(code, 3)
            self.assertIn("error", json.loads(stdout.getvalue()))


def _write_fixture(
    root: Path,
    *,
    now: datetime,
    ask: float,
    m1_atr_pips: float | None,
    historical_missed: int = 1,
) -> dict[str, Path]:
    data = root / "data"
    docs = root / "docs"
    data.mkdir()
    docs.mkdir()
    files = {
        "broker": data / "broker_snapshot.json",
        "charts": data / "pair_charts.json",
        "position_management": data / "position_management.json",
        "guardian_management": data / "position_guardian_management.json",
        "timing": data / "execution_timing_audit.json",
        "output": data / "profit_capture_bot.json",
        "report": docs / "profit_capture_bot_report.md",
    }
    _write_json(
        files["broker"],
        {
            "fetched_at_utc": now.isoformat(),
            "quotes": {
                "USD_JPY": {
                    "bid": ask - 0.004,
                    "ask": ask,
                    "timestamp_utc": now.isoformat(),
                }
            },
            "positions": [
                {
                    "trade_id": "472792",
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "owner": "trader",
                    "units": 6300,
                    "entry_price": 161.692,
                    "unrealized_pl_jpy": 120.0 if ask < 161.692 else -120.0,
                    "take_profit": 161.636,
                    "stop_loss": 161.745,
                }
            ],
        },
    )
    indicators = {} if m1_atr_pips is None else {"atr_pips": m1_atr_pips}
    _write_json(
        files["charts"],
        {
            "generated_at_utc": now.isoformat(),
            "charts": [{"pair": "USD_JPY", "views": [{"granularity": "M1", "indicators": indicators}]}],
        },
    )
    position_row = {
        "trade_id": "472792",
        "pair": "USD_JPY",
        "side": "SHORT",
        "action": "HOLD_PROTECTED",
    }
    _write_json(files["position_management"], {"generated_at_utc": now.isoformat(), "positions": [position_row]})
    _write_json(files["guardian_management"], {"generated_at_utc": now.isoformat(), "positions": [position_row]})
    _write_json(
        files["timing"],
        {
            "generated_at_utc": now.isoformat(),
            "summary": {
                "loss_closes_profit_capture_missed": historical_missed,
                "stop_loss_closes_profit_capture_missed": historical_missed,
                "loss_close_estimated_capture_gap_jpy": 340.2 if historical_missed else 0.0,
                "loss_close_actual_pl_jpy": -340.2 if historical_missed else 0.0,
                "loss_close_counterfactual_profit_capture_pl_jpy": 105.84 if historical_missed else 0.0,
                "loss_close_counterfactual_profit_capture_delta_jpy": 446.04 if historical_missed else 0.0,
                "loss_close_counterfactual_profit_capture_jpy": 105.84 if historical_missed else 0.0,
            },
            "loss_close_regrets": [
                {
                    "trade_id": "472792",
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "exit_reason": "STOP_LOSS_ORDER",
                    "realized_pl_jpy": -340.2,
                    "profit_capture_missed_before_loss_close": bool(historical_missed),
                    "profit_capture_counterfactual_exit": "TP_PROGRESS_CAPTURE",
                    "profit_capture_counterfactual_pips": 3.0,
                    "profit_capture_counterfactual_jpy": 105.84,
                    "profit_capture_counterfactual_net_improvement_jpy": 446.04,
                }
            ],
        },
    )
    return files


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

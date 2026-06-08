from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from quant_rabbit.cli import (
    _LIVE_RUNTIME_COMMANDS,
    _SL_FREE_RUNTIME_DEFAULTS,
    _auto_refresh_market_evidence_if_required,
    _pre_entry_projection_verification_if_required,
    _refresh_current_forecast_history,
    _snapshot_from_json,
    _snapshot_to_json,
    main,
)
from quant_rabbit.models import BrokerOrder, BrokerSnapshot, Owner, Quote
from quant_rabbit.paths import DEFAULT_MARKET_CONTEXT_MATRIX
from quant_rabbit.strategy.intent_generator import _snapshot_from_json as _intent_snapshot_from_json


class CliHelpTest(unittest.TestCase):
    def _adverse_partial_close_files(self, root: Path) -> tuple[Path, Path]:
        snapshot = root / "snapshot.json"
        snapshot.write_text(json.dumps({
            "positions": [
                {
                    "trade_id": "t-adverse",
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "units": 10000,
                    "entry_price": 1.2,
                    "take_profit": None,
                    "stop_loss": None,
                    "unrealized_pl_jpy": -1000,
                    "owner": "trader",
                }
            ],
            "orders": [],
            "quotes": {"EUR_USD": {"bid": 1.195, "ask": 1.1951}},
        }))
        pair_charts = root / "pair_charts.json"
        pair_charts.write_text(json.dumps({
            "charts": [
                {
                    "pair": "EUR_USD",
                    "chart_story": "EUR_USD RANGE; M5(UNCLEAR struct=CHOCH_DOWN@1.1950)",
                    "confluence": {"h4_atr_pips": 20.0},
                    "views": [],
                }
            ]
        }))
        return snapshot, pair_charts

    def _partial_close_artifact_args(self, root: Path) -> list[str]:
        return [
            "--output",
            str(root / "partial_close.json"),
            "--report",
            str(root / "partial_close.md"),
            "--execution-ledger-db",
            str(root / "execution_ledger.db"),
            "--execution-ledger-report",
            str(root / "execution_ledger.md"),
        ]

    def test_top_level_help_renders_daily_target_percent_text(self) -> None:
        stdout = io.StringIO()

        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("daily-target-state", help_text)
        self.assertIn("10% target", help_text)

    def test_snapshot_json_preserves_pending_order_thesis_raw_fields(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = BrokerSnapshot(
            fetched_at_utc=now,
            orders=(
                BrokerOrder(
                    order_id="472032",
                    pair="GBP_CAD",
                    order_type="STOP",
                    price=1.86796,
                    state="PENDING",
                    units=3000,
                    owner=Owner.TRADER,
                    raw={
                        "accountID": "should-not-be-persisted",
                        "createTime": "2026-06-04T22:45:52.123456789Z",
                        "clientExtensions": {"tag": "trader", "comment": "qr-vnext"},
                        "takeProfitOnFill": {"price": "1.87852"},
                        "stopLossOnFill": {"price": "1.86646"},
                        "timeInForce": "GTC",
                    },
                ),
            ),
            quotes={"GBP_CAD": Quote("GBP_CAD", 1.8672, 1.8673, timestamp_utc=now)},
        )

        payload = json.loads(_snapshot_to_json(snapshot))
        order_raw = payload["orders"][0]["raw"]
        restored = _snapshot_from_json(payload)
        restored_for_reuse = _intent_snapshot_from_json(payload)

        self.assertNotIn("accountID", order_raw)
        self.assertEqual(order_raw["createTime"], "2026-06-04T22:45:52.123456789Z")
        self.assertEqual(restored.orders[0].raw["clientExtensions"]["tag"], "trader")
        self.assertEqual(restored.orders[0].raw["stopLossOnFill"]["price"], "1.86646")
        self.assertEqual(restored_for_reuse.orders[0].raw["takeProfitOnFill"]["price"], "1.87852")

    def test_autotrade_missing_gpt_decision_response_returns_json_error(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = main(
                [
                    "autotrade-cycle",
                    "--use-gpt-trader",
                    "--gpt-decision-response",
                    "/tmp/qr-missing-gpt-decision-response.json",
                ]
            )

        self.assertEqual(code, 2)
        payload = json.loads(stdout.getvalue())
        self.assertIn("qr-missing-gpt-decision-response.json", payload["error"])

    def test_position_management_command_refreshes_sidecar_from_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            pair_charts = root / "pair_charts.json"
            output = root / "position_management.json"
            report = root / "position_management.md"
            fetched_at = datetime(2026, 6, 5, 20, 59, 2, tzinfo=timezone.utc)
            snapshot.write_text(json.dumps({
                "fetched_at_utc": fetched_at.isoformat(),
                "positions": [
                    {
                        "trade_id": "t-position",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "units": 1000,
                        "entry_price": 1.1000,
                        "take_profit": 1.1020,
                        "stop_loss": 1.0980,
                        "unrealized_pl_jpy": 0.0,
                        "owner": "trader",
                    }
                ],
                "orders": [],
                "quotes": {
                    "EUR_USD": {
                        "bid": 1.1004,
                        "ask": 1.1005,
                        "timestamp_utc": fetched_at.isoformat(),
                    }
                },
                "home_conversions": {"USD": 160.0, "JPY": 1.0},
            }))
            pair_charts.write_text(json.dumps({"charts": []}))
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main([
                    "position-management",
                    "--snapshot",
                    str(snapshot),
                    "--pair-charts",
                    str(pair_charts),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                ])

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "OK")
            self.assertEqual(payload["snapshot_fetched_at_utc"], fetched_at.isoformat())
            self.assertEqual(payload["position_count"], 1)
            self.assertTrue(output.exists())
            self.assertTrue(report.exists())
            saved = json.loads(output.read_text())
            self.assertEqual(saved["positions"][0]["trade_id"], "t-position")

    def test_gpt_trader_requires_codex_decision_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            snapshot.write_text(json.dumps({"positions": [], "orders": [], "quotes": {}}))
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(["gpt-trader-decision", "--snapshot", str(snapshot)])

        self.assertEqual(code, 2)
        payload = json.loads(stdout.getvalue())
        self.assertIn("--decision-response", payload["error"])

    def test_thesis_evolution_legacy_alias_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            snapshot.write_text(json.dumps({"positions": [], "orders": [], "quotes": {}}))
            pair_charts = root / "pair_charts.json"
            pair_charts.write_text(json.dumps({"charts": []}))
            output = root / "thesis_evolution.json"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "thesis-evolution",
                        "--snapshot",
                        str(snapshot),
                        "--pair-charts",
                        str(pair_charts),
                        "--output",
                        str(output),
                    ]
                )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["evolution_count"], 0)

    def test_refresh_current_forecast_history_records_snapshot_cycle_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            pair_charts = root / "pair_charts.json"
            pair_charts.write_text(json.dumps({
                "generated_at_utc": "2026-05-30T00:01:00+00:00",
                "charts": [{"pair": "EUR_USD", "views": []}],
            }))
            snapshot_payload = {
                "fetched_at_utc": "2026-05-30T00:00:00+00:00",
                "positions": [],
                "orders": [],
                "quotes": {
                    "EUR_USD": {
                        "bid": 1.1000,
                        "ask": 1.1002,
                        "timestamp_utc": "2026-05-30T00:00:00+00:00",
                    }
                },
            }
            forecast = SimpleNamespace(
                pair="EUR_USD",
                direction="UP",
                confidence=0.72,
                current_price=1.1001,
                invalidation_price=1.0990,
                target_price=1.1020,
                horizon_min=60,
                raw_confidence=0.72,
                calibration_multiplier=1.0,
                up_score=12.0,
                down_score=3.0,
                range_score=0.0,
                drivers_for=("test up",),
                drivers_against=(),
                rationale_summary="UP=12 DOWN=3",
            )

            with mock.patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ), mock.patch(
                "quant_rabbit.strategy.projection_ledger.projection_telemetry_market_open",
                return_value=True,
            ):
                first = _refresh_current_forecast_history(
                    snapshot_payload=snapshot_payload,
                    pair_charts_path=pair_charts,
                    pairs=["EUR_USD"],
                    data_root=data_root,
                    cycle_source="test",
                )
                second = _refresh_current_forecast_history(
                    snapshot_payload=snapshot_payload,
                    pair_charts_path=pair_charts,
                    pairs=["EUR_USD"],
                    data_root=data_root,
                    cycle_source="test",
                )

            self.assertEqual(first["recorded"], 1)
            self.assertEqual(first["projection_recorded"], 1)
            self.assertEqual(first["projection_skipped"], {})
            self.assertEqual(first["forecasts"]["EUR_USD"]["direction"], "UP")
            self.assertEqual(second["recorded"], 0)
            self.assertEqual(second["skipped"]["EUR_USD"], "already_recorded_for_cycle")
            rows = (data_root / "forecast_history.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 1)
            row = json.loads(rows[0])
            self.assertEqual(row["pair"], "EUR_USD")
            self.assertEqual(row["cycle_id"], "test:2026-05-30T00:00:00+00:00:2026-05-30T00:01:00+00:00")
            self.assertEqual(row["timestamp_utc"], "2026-05-30T00:00:00Z")
            self.assertEqual(row["direction"], "UP")
            self.assertTrue((data_root / "projection_ledger.jsonl").exists())

    def test_refresh_current_forecast_history_skips_closed_market_without_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            pair_charts = root / "pair_charts.json"
            pair_charts.write_text(json.dumps({
                "generated_at_utc": "2026-05-30T00:01:00+00:00",
                "charts": [{"pair": "EUR_USD", "views": []}],
            }))
            snapshot_payload = {
                "fetched_at_utc": "2026-05-30T00:00:00+00:00",
                "positions": [],
                "orders": [],
                "quotes": {
                    "EUR_USD": {
                        "bid": 1.1000,
                        "ask": 1.1002,
                        "timestamp_utc": "2026-05-30T00:00:00+00:00",
                    }
                },
            }
            forecast = SimpleNamespace(
                pair="EUR_USD",
                direction="UP",
                confidence=0.72,
                current_price=1.1001,
                invalidation_price=1.0990,
                target_price=1.1020,
                horizon_min=60,
                raw_confidence=0.72,
                calibration_multiplier=1.0,
                up_score=12.0,
                down_score=3.0,
                range_score=0.0,
                drivers_for=("test up",),
                drivers_against=(),
                rationale_summary="UP=12 DOWN=3",
            )

            with mock.patch(
                "quant_rabbit.strategy.intent_generator._forecast_seed_for_pair",
                return_value=forecast,
            ), mock.patch(
                "quant_rabbit.strategy.projection_ledger.projection_telemetry_market_open",
                return_value=False,
            ):
                first = _refresh_current_forecast_history(
                    snapshot_payload=snapshot_payload,
                    pair_charts_path=pair_charts,
                    pairs=["EUR_USD"],
                    data_root=data_root,
                    cycle_source="test",
                )

            self.assertEqual(first["recorded"], 0)
            self.assertEqual(first["projection_recorded"], 0)
            self.assertEqual(first["skipped"]["EUR_USD"], "market_closed_at_forecast_emission")
            self.assertFalse((data_root / "forecast_history.jsonl").exists())
            self.assertFalse((data_root / "projection_ledger.jsonl").exists())

    def test_replay_execution_missing_prices_returns_json_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intents = root / "intents.json"
            intents.write_text(json.dumps({"results": []}))
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "replay-execution",
                        "--intents",
                        str(intents),
                        "--prices",
                        str(root / "missing_prices.json"),
                    ]
                )

        self.assertEqual(code, 2)
        payload = json.loads(stdout.getvalue())
        self.assertIn("missing_prices.json", payload["error"])

    def test_news_snapshot_no_fetch_writes_runtime_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "data" / "news_items.json"
            digest = root / "logs" / "news_digest.md"
            flow = root / "logs" / "news_flow_log.md"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main(
                    [
                        "news-snapshot",
                        "--no-fetch",
                        "--output",
                        str(output),
                        "--digest",
                        str(digest),
                        "--flow-log",
                        str(flow),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["items"], 0)
            self.assertTrue(output.exists())
            self.assertTrue(digest.exists())
            self.assertTrue(flow.exists())

    def test_adverse_partial_close_defaults_to_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"QR_DISABLE_ADVERSE_PARTIAL_CLOSE": ""}):
            root = Path(tmp)
            snapshot, pair_charts = self._adverse_partial_close_files(root)
            stdout = io.StringIO()

            with mock.patch("quant_rabbit.cli.OandaExecutionClient") as client_cls, redirect_stdout(stdout):
                code = main([
                    "adverse-partial-close",
                    "--snapshot",
                    str(snapshot),
                    "--pair-charts",
                    str(pair_charts),
                    *self._partial_close_artifact_args(root),
                ])

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["dry_run"])
        self.assertFalse(payload["send"])
        self.assertEqual(payload["actions_count"], 1)
        self.assertFalse(payload["results"][0]["sent"])
        client_cls.assert_not_called()

    def test_adverse_partial_close_send_requires_confirm_live_and_live_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"QR_DISABLE_ADVERSE_PARTIAL_CLOSE": ""}, clear=False):
            root = Path(tmp)
            snapshot, pair_charts = self._adverse_partial_close_files(root)
            stdout = io.StringIO()

            with mock.patch("quant_rabbit.cli.OandaExecutionClient") as client_cls, redirect_stdout(stdout):
                code = main([
                    "adverse-partial-close",
                    "--snapshot",
                    str(snapshot),
                    "--pair-charts",
                    str(pair_charts),
                    *self._partial_close_artifact_args(root),
                    "--send",
                ])

        self.assertEqual(code, 2)
        self.assertIn("--confirm-live", json.loads(stdout.getvalue())["error"])
        client_cls.assert_not_called()

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {
            "QR_DISABLE_ADVERSE_PARTIAL_CLOSE": "",
            "QR_LIVE_ENABLED": "",
        }, clear=False):
            root = Path(tmp)
            snapshot, pair_charts = self._adverse_partial_close_files(root)
            stdout = io.StringIO()

            with mock.patch("quant_rabbit.cli.OandaExecutionClient") as client_cls, redirect_stdout(stdout):
                code = main([
                    "adverse-partial-close",
                    "--snapshot",
                    str(snapshot),
                    "--pair-charts",
                    str(pair_charts),
                    *self._partial_close_artifact_args(root),
                    "--send",
                    "--confirm-live",
                ])

        self.assertEqual(code, 2)
        self.assertIn("QR_LIVE_ENABLED=1", json.loads(stdout.getvalue())["error"])
        client_cls.assert_not_called()

    def test_adverse_partial_close_send_calls_broker_when_explicitly_live(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {
            "QR_DISABLE_ADVERSE_PARTIAL_CLOSE": "",
            "QR_LIVE_ENABLED": "1",
        }, clear=False):
            root = Path(tmp)
            snapshot, pair_charts = self._adverse_partial_close_files(root)
            stdout = io.StringIO()
            client = mock.Mock()
            client.close_trade.return_value = {"ok": True}

            with mock.patch("quant_rabbit.cli.OandaExecutionClient", return_value=client), redirect_stdout(stdout):
                code = main([
                    "adverse-partial-close",
                    "--snapshot",
                    str(snapshot),
                    "--pair-charts",
                    str(pair_charts),
                    *self._partial_close_artifact_args(root),
                    "--send",
                    "--confirm-live",
                ])

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["dry_run"])
            self.assertTrue(payload["send"])
            self.assertTrue(payload["results"][0]["sent"])
            client.close_trade.assert_called_once_with("t-adverse", units="5000")
            self.assertEqual(payload["execution_ledger"]["status"], "RECORDED")
            with sqlite3.connect(root / "execution_ledger.db") as conn:
                event = conn.execute(
                    """
                    SELECT event_type, exit_reason, trade_id
                    FROM execution_events
                    WHERE event_type = 'GATEWAY_TRADE_CLOSE_SENT'
                    """
                ).fetchone()
            self.assertEqual(event, ("GATEWAY_TRADE_CLOSE_SENT", "ADVERSE_PARTIAL_CLOSE", "t-adverse"))

    def test_generate_intents_refreshes_market_story_when_news_is_newer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            news_dir = root / "logs"
            news_dir.mkdir()
            digest = news_dir / "news_digest.md"
            digest.write_text("EUR_USD risk-on breakout failure pressure\n", encoding="utf-8")
            profile = root / "data" / "market_story_profile.json"
            profile.parent.mkdir()
            profile.write_text("{}", encoding="utf-8")
            report = root / "data" / "market_story_report.md"
            os.utime(profile, (100, 100))
            os.utime(digest, (200, 200))

            summary = type(
                "Summary",
                (),
                {
                    "output_path": root / "data" / "order_intents.json",
                    "report_path": root / "docs" / "order_intents_report.md",
                    "candidates_seen": 0,
                    "generated": 0,
                    "needs_snapshot": False,
                    "dry_run_passed": 0,
                    "live_ready": 0,
                },
            )()
            stdout = io.StringIO()

            with mock.patch("quant_rabbit.cli.MarketStoryMiner") as miner_cls, mock.patch(
                "quant_rabbit.cli.IntentGenerator"
            ) as generator_cls, redirect_stdout(stdout):
                miner_cls.return_value.run.return_value = object()
                generator_cls.return_value.run.return_value = summary
                code = main(
                    [
                        "generate-intents",
                        "--campaign-plan",
                        str(root / "data" / "daily_campaign_plan.json"),
                        "--strategy-profile",
                        str(root / "data" / "strategy_profile.json"),
                        "--snapshot",
                        str(root / "data" / "broker_snapshot.json"),
                        "--output",
                        str(summary.output_path),
                        "--report",
                        str(summary.report_path),
                        "--max-loss-jpy",
                        "100",
                        "--market-story-profile",
                        str(profile),
                        "--market-story-report",
                        str(report),
                        "--market-news-dir",
                        str(news_dir),
                    ]
                )

        self.assertEqual(code, 0)
        miner_cls.assert_called_once_with(report_path=report, profile_path=profile, news_root=news_dir)
        miner_cls.return_value.run.assert_called_once_with()
        generator_cls.return_value.run.assert_called_once_with(
            snapshot_path=root / "data" / "broker_snapshot.json",
            max_candidates=56,
        )

    def test_generate_intents_uses_default_broker_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "data" / "broker_snapshot.json"
            snapshot.parent.mkdir()
            snapshot.write_text(
                json.dumps(
                    {
                        "fetched_at_utc": "2026-06-02T01:00:00+00:00",
                        "quotes": {"EUR_USD": {"bid": 1.1, "ask": 1.1001}},
                    }
                )
            )
            summary = SimpleNamespace(
                output_path=root / "data" / "order_intents.json",
                report_path=root / "docs" / "order_intents_report.md",
                candidates_seen=1,
                generated=1,
                needs_snapshot=False,
                dry_run_passed=1,
                live_ready=0,
            )
            stdout = io.StringIO()

            with mock.patch("quant_rabbit.cli.DEFAULT_BROKER_SNAPSHOT", snapshot), mock.patch(
                "quant_rabbit.cli._auto_refresh_market_evidence_if_required",
                return_value={"status": "SKIPPED", "reason": "test"},
            ), mock.patch("quant_rabbit.cli.IntentGenerator") as generator_cls, redirect_stdout(stdout):
                generator_cls.return_value.run.return_value = summary
                code = main(
                    [
                        "generate-intents",
                        "--campaign-plan",
                        str(root / "data" / "daily_campaign_plan.json"),
                        "--strategy-profile",
                        str(root / "data" / "strategy_profile.json"),
                        "--output",
                        str(summary.output_path),
                        "--report",
                        str(summary.report_path),
                        "--no-refresh-market-story",
                    ]
                )

        self.assertEqual(code, 0)
        generator_cls.return_value.run.assert_called_once_with(snapshot_path=snapshot, max_candidates=56)

    def test_generate_intents_returns_json_error_for_stale_campaign_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stdout = io.StringIO()

            with mock.patch(
                "quant_rabbit.cli._auto_refresh_market_evidence_if_required",
                return_value={"status": "SKIPPED", "reason": "test"},
            ), mock.patch(
                "quant_rabbit.cli._refresh_snapshot_after_market_evidence_if_required",
                return_value={"status": "SKIPPED", "reason": "test"},
            ), mock.patch(
                "quant_rabbit.cli.IntentGenerator"
            ) as generator_cls, redirect_stdout(stdout):
                generator_cls.return_value.run.side_effect = RuntimeError("campaign plan stale while target is open")
                code = main(
                    [
                        "generate-intents",
                        "--campaign-plan",
                        str(root / "data" / "daily_campaign_plan.json"),
                        "--strategy-profile",
                        str(root / "data" / "strategy_profile.json"),
                        "--snapshot",
                        str(root / "data" / "broker_snapshot.json"),
                        "--output",
                        str(root / "data" / "order_intents.json"),
                        "--report",
                        str(root / "docs" / "order_intents_report.md"),
                        "--no-refresh-market-story",
                    ]
                )

        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stdout.getvalue())["error"], "campaign plan stale while target is open")

    def test_plan_campaign_updates_target_before_plan_timestamp(self) -> None:
        calls: list[str] = []
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def target_run(**_: object) -> SimpleNamespace:
                calls.append("target")
                return SimpleNamespace(
                    state_path=root / "data" / "daily_target_state.json",
                    report_path=root / "docs" / "daily_target_report.md",
                    remaining_target_jpy=1000.0,
                )

            def campaign_run(**_: object) -> SimpleNamespace:
                calls.append("campaign")
                return SimpleNamespace(
                    report_path=root / "docs" / "daily_campaign_report.md",
                    plan_path=root / "data" / "daily_campaign_plan.json",
                    target_jpy=1000.0,
                    lanes=1,
                    actionable_lanes=1,
                    rejected_lanes=0,
                )

            with mock.patch("quant_rabbit.cli.DailyTargetLedger") as ledger_cls, mock.patch(
                "quant_rabbit.cli.CampaignPlanner"
            ) as planner_cls, redirect_stdout(stdout):
                ledger_cls.return_value.run.side_effect = target_run
                planner_cls.return_value.run.side_effect = campaign_run
                code = main(
                    [
                        "plan-campaign",
                        "--start-balance",
                        "10000",
                        "--plan",
                        str(root / "data" / "daily_campaign_plan.json"),
                        "--report",
                        str(root / "docs" / "daily_campaign_report.md"),
                    ]
                )

        self.assertEqual(code, 0)
        self.assertEqual(calls, ["target", "campaign"])

    def test_generate_intents_syncs_execution_ledger_before_live_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "data" / "broker_snapshot.json"
            snapshot.parent.mkdir()
            snapshot.write_text(
                json.dumps(
                    {
                        "fetched_at_utc": "2026-06-02T01:00:00+00:00",
                        "quotes": {"EUR_USD": {"bid": 1.1, "ask": 1.1001}},
                    }
                )
            )
            summary = SimpleNamespace(
                output_path=root / "data" / "order_intents.json",
                report_path=root / "docs" / "order_intents_report.md",
                candidates_seen=0,
                generated=0,
                needs_snapshot=False,
                dry_run_passed=0,
                live_ready=0,
            )
            ledger_summary = SimpleNamespace(
                status="SYNCED",
                transactions_seen=3,
                transactions_inserted=2,
                events_inserted=2,
                last_transaction_id="471858",
                baseline_transaction_id=None,
            )
            stdout = io.StringIO()

            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "quant_rabbit.cli._running_under_test_harness", return_value=False
            ), mock.patch(
                "quant_rabbit.cli._auto_refresh_market_evidence_if_required",
                return_value={"status": "SKIPPED", "reason": "test"},
            ) as market_refresh, mock.patch("quant_rabbit.cli.OandaReadOnlyClient") as client_cls, mock.patch(
                "quant_rabbit.cli.ExecutionLedger"
            ) as ledger_cls, mock.patch(
                "quant_rabbit.strategy.projection_ledger.load_ledger",
                return_value=[],
            ), mock.patch(
                "quant_rabbit.cli._refresh_current_forecast_history",
                return_value={"recorded": 0, "skipped": {}, "cycle_id": "cycle"},
            ), mock.patch(
                "quant_rabbit.cli.IntentGenerator"
            ) as generator_cls, redirect_stdout(stdout):
                ledger_cls.return_value.sync_oanda_transactions.return_value = ledger_summary
                generator_cls.return_value.run.return_value = summary
                code = main(
                    [
                        "generate-intents",
                        "--campaign-plan",
                        str(root / "data" / "daily_campaign_plan.json"),
                        "--strategy-profile",
                        str(root / "data" / "strategy_profile.json"),
                        "--snapshot",
                        str(snapshot),
                        "--output",
                        str(summary.output_path),
                        "--report",
                        str(summary.report_path),
                        "--no-refresh-market-story",
                    ]
                )

        self.assertEqual(code, 0)
        market_refresh.assert_called_once_with(
            label="generate-intents",
            reuse_market_artifacts=False,
            market_context_matrix_path=DEFAULT_MARKET_CONTEXT_MATRIX,
        )
        client_cls.assert_called_once_with()
        ledger_cls.return_value.sync_oanda_transactions.assert_called_once_with(client_cls.return_value)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["market_evidence_refresh"]["status"], "SKIPPED")
        self.assertEqual(payload["execution_ledger_sync"]["status"], "SYNCED")
        self.assertEqual(payload["execution_ledger_sync"]["last_transaction_id"], "471858")

    def test_generate_intents_refreshes_snapshot_after_market_evidence_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "data" / "broker_snapshot.json"
            snapshot.parent.mkdir()
            snapshot.write_text(
                json.dumps(
                    {
                        "fetched_at_utc": "2026-06-02T01:00:00+00:00",
                        "quotes": {"EUR_USD": {"bid": 1.1, "ask": 1.1001}},
                    }
                )
            )
            summary = SimpleNamespace(
                output_path=root / "data" / "order_intents.json",
                report_path=root / "docs" / "order_intents_report.md",
                candidates_seen=0,
                generated=0,
                needs_snapshot=False,
                dry_run_passed=0,
                live_ready=0,
            )
            fresh_at = datetime(2026, 6, 2, 1, 5, tzinfo=timezone.utc)
            fresh_snapshot = BrokerSnapshot(
                fetched_at_utc=fresh_at,
                quotes={"EUR_USD": Quote("EUR_USD", 1.1002, 1.1004, fresh_at)},
            )
            stdout = io.StringIO()

            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "quant_rabbit.cli._running_under_test_harness", return_value=False
            ), mock.patch(
                "quant_rabbit.cli._auto_refresh_market_evidence_if_required",
                return_value={"status": "REFRESHED", "pairs": 28},
            ) as market_refresh, mock.patch("quant_rabbit.cli.OandaReadOnlyClient") as client_cls, mock.patch(
                "quant_rabbit.cli._pre_entry_execution_ledger_sync_if_required",
                return_value=None,
            ), mock.patch(
                "quant_rabbit.cli._pre_entry_projection_verification_if_required",
                return_value=None,
            ), mock.patch("quant_rabbit.cli.IntentGenerator") as generator_cls, redirect_stdout(stdout):
                client_cls.return_value.snapshot.return_value = fresh_snapshot
                generator_cls.return_value.run.return_value = summary
                code = main(
                    [
                        "generate-intents",
                        "--campaign-plan",
                        str(root / "data" / "daily_campaign_plan.json"),
                        "--strategy-profile",
                        str(root / "data" / "strategy_profile.json"),
                        "--snapshot",
                        str(snapshot),
                        "--output",
                        str(summary.output_path),
                        "--report",
                        str(summary.report_path),
                        "--no-refresh-market-story",
                    ]
                )
                refreshed = json.loads(snapshot.read_text())

        self.assertEqual(code, 0)
        market_refresh.assert_called_once_with(
            label="generate-intents",
            reuse_market_artifacts=False,
            market_context_matrix_path=DEFAULT_MARKET_CONTEXT_MATRIX,
        )
        client_cls.return_value.snapshot.assert_called_once_with(("EUR_USD",))
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["snapshot_refresh"]["status"], "REFRESHED")
        self.assertEqual(refreshed["fetched_at_utc"], fresh_at.isoformat())
        self.assertEqual(refreshed["quotes"]["EUR_USD"]["bid"], 1.1002)

    def test_generate_intents_delegates_pre_entry_forecast_to_intent_generator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "data" / "broker_snapshot.json"
            snapshot.parent.mkdir()
            snapshot.write_text(
                json.dumps(
                    {
                        "fetched_at_utc": "2026-06-02T01:00:00+00:00",
                        "quotes": {"EUR_USD": {"bid": 1.1, "ask": 1.1001}},
                    }
                )
            )
            summary = SimpleNamespace(
                output_path=root / "data" / "order_intents.json",
                report_path=root / "docs" / "order_intents_report.md",
                candidates_seen=0,
                generated=0,
                needs_snapshot=False,
                dry_run_passed=0,
                live_ready=0,
            )
            stdout = io.StringIO()

            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "quant_rabbit.cli._running_under_test_harness", return_value=False
            ), mock.patch(
                "quant_rabbit.cli._auto_refresh_market_evidence_if_required",
                return_value={"status": "SKIPPED", "reason": "test"},
            ) as market_refresh, mock.patch(
                "quant_rabbit.strategy.projection_ledger.load_ledger",
                return_value=[],
            ), mock.patch(
                "quant_rabbit.cli._pre_entry_execution_ledger_sync_if_required",
                return_value=None,
            ), mock.patch(
                "quant_rabbit.cli._refresh_current_forecast_history"
            ) as refresh_forecast, mock.patch(
                "quant_rabbit.cli.IntentGenerator"
            ) as generator_cls, redirect_stdout(stdout):
                generator_cls.return_value.run.return_value = summary
                code = main(
                    [
                        "generate-intents",
                        "--campaign-plan",
                        str(root / "data" / "daily_campaign_plan.json"),
                        "--strategy-profile",
                        str(root / "data" / "strategy_profile.json"),
                        "--snapshot",
                        str(snapshot),
                        "--output",
                        str(summary.output_path),
                        "--report",
                        str(summary.report_path),
                        "--no-refresh-market-story",
                    ]
                )

        self.assertEqual(code, 0)
        market_refresh.assert_called_once_with(
            label="generate-intents",
            reuse_market_artifacts=False,
            market_context_matrix_path=DEFAULT_MARKET_CONTEXT_MATRIX,
        )
        refresh_forecast.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["market_evidence_refresh"]["status"], "SKIPPED")
        self.assertEqual(payload["forecast_refresh"]["status"], "DELEGATED_TO_INTENT_GENERATOR")

    def test_autotrade_gpt_protect_status_exits_zero(self) -> None:
        summary = type(
            "Summary",
            (),
            {
                "status": "GPT_PROTECT",
                "report_path": Path("docs/autotrade_cycle_report.md"),
                "snapshot_path": Path("data/broker_snapshot.json"),
                "intents_path": Path("data/order_intents.json"),
                "selected_lane_id": None,
                "selected_lane_ids": (),
                "deterministic_lane_id": None,
                "decision_source": "gpt_trader",
                "sent": False,
                "sent_count": 0,
                "positions": 8,
                "orders": 6,
                "live_ready": 0,
                "receipt_promotions": 0,
                "canceled_orders": (),
                "position_management_action": "HOLD_PROTECTED",
                "position_execution_status": "NO_ACTION",
                "position_execution_sent": False,
                "target_status": "PURSUE_TARGET",
                "target_remaining_jpy": 38087.0,
                "target_progress_pct": -90.9,
                "selected_lane_score": None,
                "selected_lane_size_multiple": None,
                "gpt_status": "ACCEPTED",
                "gpt_action": "PROTECT",
                "gpt_allowed": True,
                "gpt_issues": 0,
                "gpt_error": None,
            },
        )()
        stdout = io.StringIO()

        with mock.patch("quant_rabbit.cli.AutoTradeCycle") as cycle_cls, redirect_stdout(stdout):
            cycle_cls.return_value.run.return_value = summary
            code = main(["autotrade-cycle"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "GPT_PROTECT")

    def test_auto_market_evidence_refresh_skips_unittest_harness(self) -> None:
        result = _auto_refresh_market_evidence_if_required(label="test")

        self.assertEqual(result, {"status": "SKIPPED", "reason": "test_harness"})

    def test_autotrade_reuse_market_artifacts_skips_auto_market_refresh(self) -> None:
        summary = type(
            "Summary",
            (),
            {
                "status": "NO_LIVE_READY_INTENT",
                "report_path": Path("docs/autotrade_cycle_report.md"),
                "snapshot_path": Path("data/broker_snapshot.json"),
                "intents_path": Path("data/order_intents.json"),
                "selected_lane_id": None,
                "selected_lane_ids": (),
                "deterministic_lane_id": None,
                "decision_source": None,
                "sent": False,
                "sent_count": 0,
                "positions": 0,
                "orders": 0,
                "live_ready": 0,
                "receipt_promotions": 0,
                "canceled_orders": (),
                "position_management_action": None,
                "position_execution_status": None,
                "position_execution_sent": False,
                "target_status": None,
                "target_remaining_jpy": None,
                "target_progress_pct": None,
                "selected_lane_score": None,
                "selected_lane_size_multiple": None,
                "gpt_status": None,
                "gpt_action": None,
                "gpt_allowed": None,
                "gpt_issues": 0,
                "gpt_error": None,
            },
        )()
        stdout = io.StringIO()

        with (
            mock.patch("quant_rabbit.cli._auto_refresh_market_evidence_if_required") as refresh,
            mock.patch("quant_rabbit.cli.AutoTradeCycle") as cycle_cls,
            redirect_stdout(stdout),
        ):
            refresh.return_value = {"status": "SKIPPED", "reason": "reuse_market_artifacts"}
            cycle_cls.return_value.run.return_value = summary
            code = main(["autotrade-cycle", "--reuse-market-artifacts"])

        self.assertEqual(code, 0)
        refresh.assert_called_once_with(label="autotrade-cycle", reuse_market_artifacts=True)

    def test_reuse_market_artifacts_requires_market_context_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            matrix = Path(tmp) / "missing_market_context_matrix.json"

            with self.assertRaisesRegex(RuntimeError, "missing required market evidence artifact"):
                _auto_refresh_market_evidence_if_required(
                    label="autotrade-cycle",
                    reuse_market_artifacts=True,
                    market_context_matrix_path=matrix,
                )

    def test_reuse_market_artifacts_requires_non_empty_market_context_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            matrix = Path(tmp) / "market_context_matrix.json"
            matrix.write_text(json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "pairs": {}}))

            with self.assertRaisesRegex(RuntimeError, "has no pair/side matrix rows"):
                _auto_refresh_market_evidence_if_required(
                    label="autotrade-cycle",
                    reuse_market_artifacts=True,
                    market_context_matrix_path=matrix,
                )

    def test_reuse_market_artifacts_requires_context_asset_and_broker_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = root / "market_context_matrix.json"
            context_assets = root / "context_asset_charts.json"
            broker_instruments = root / "broker_instruments.json"
            matrix.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "pairs": {"EUR_USD": {"LONG": {"evidence_ref": "matrix:EUR_USD:LONG"}}},
                    }
                )
            )
            context_assets.write_text(json.dumps({"charts": [{"pair": "XAU_USD", "views": []}]}))

            with self.assertRaisesRegex(RuntimeError, "broker_instruments"):
                _auto_refresh_market_evidence_if_required(
                    label="autotrade-cycle",
                    reuse_market_artifacts=True,
                    market_context_matrix_path=matrix,
                    context_asset_charts_path=context_assets,
                    broker_instruments_path=broker_instruments,
                    order_intents_path=root / "missing_order_intents.json",
                )

    def test_reuse_market_artifacts_requires_intents_newer_than_market_context_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = root / "market_context_matrix.json"
            context_assets = root / "context_asset_charts.json"
            broker_instruments = root / "broker_instruments.json"
            order_intents = root / "order_intents.json"
            matrix.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-08T08:10:00+00:00",
                        "pairs": {"EUR_USD": {"LONG": {"evidence_ref": "matrix:EUR_USD:LONG"}}},
                    }
                )
            )
            context_assets.write_text(json.dumps({"charts": [{"pair": "XAU_USD", "views": []}]}))
            broker_instruments.write_text(json.dumps({"tradeable_instruments": ["EUR_USD"]}))
            order_intents.write_text(
                json.dumps({"generated_at_utc": "2026-06-08T08:00:00+00:00", "results": []})
            )

            with self.assertRaisesRegex(RuntimeError, "predates market_context_matrix"):
                _auto_refresh_market_evidence_if_required(
                    label="autotrade-cycle",
                    reuse_market_artifacts=True,
                    market_context_matrix_path=matrix,
                    context_asset_charts_path=context_assets,
                    broker_instruments_path=broker_instruments,
                    order_intents_path=order_intents,
                )

    def test_context_asset_charts_cli_writes_non_fx_technical_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "context_asset_charts.json"
            report = root / "context_asset_charts.md"
            payload = {
                "generated_at_utc": "2026-06-07T00:00:00+00:00",
                "role": "NON_FX_CONTEXT_TECHNICALS_NOT_TRADE_PERMISSION",
                "charts": [{"pair": "XAU_USD", "views": []}],
                "issues": [],
            }
            stdout = io.StringIO()
            client = object()

            with (
                mock.patch("quant_rabbit.cli.OandaReadOnlyClient", return_value=client),
                mock.patch("quant_rabbit.analysis.context_assets.build_context_asset_charts") as build,
                mock.patch("quant_rabbit.analysis.context_assets.write_context_asset_charts_report") as write_report,
                redirect_stdout(stdout),
            ):
                build.return_value = payload
                code = main([
                    "context-asset-charts",
                    "--instruments",
                    "XAU_USD,WTICO_USD",
                    "--timeframes",
                    "M5,H1",
                    "--count",
                    "50",
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                ])

            self.assertEqual(code, 0)
            build.assert_called_once()
            kwargs = build.call_args.kwargs
            self.assertIs(kwargs["client"], client)
            self.assertEqual(kwargs["instruments"], ("XAU_USD", "WTICO_USD"))
            self.assertEqual(kwargs["timeframes"], ("M5", "H1"))
            self.assertEqual(kwargs["count"], 50)
            self.assertEqual(json.loads(output.read_text()), payload)
            write_report.assert_called_once_with(payload, report)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["assets"], 1)
            self.assertEqual(summary["role"], "NON_FX_CONTEXT_TECHNICALS_NOT_TRADE_PERMISSION")


class LiveRuntimeBootstrapTest(unittest.TestCase):
    """Coverage for 2026-05-11 cli bootstrap-gate fix.

    The SKILL flow invokes `gpt-trader-decision` (and friends) directly,
    not through `scripts/run-autotrade-live.sh`, so the routine process
    inherits the user shell env with `QR_LIVE_ENABLED` unset. Previously
    the SL-free knob bootstrap only fired on `QR_LIVE_ENABLED=1`, which
    meant the verifier ran without `QR_TRADER_DISABLE_SL_REPAIR=1` and
    flagged trader-owned TP-only positions as non-layerable. Tests pin
    the gate to fire for every command in `_LIVE_RUNTIME_COMMANDS` even
    without QR_LIVE_ENABLED, while leaving the QR_LIVE_ENABLED=1 path
    intact for the wrapper-driven autotrade flow.
    """

    SL_FREE_KEYS = (
        "QR_TRADER_DISABLE_SL_REPAIR",
        "QR_GEOMETRY_ATR_MULT",
        "QR_GEOMETRY_SPREAD_FLOOR_MULT",
        "QR_MAX_PORTFOLIO_POSITIONS",
        "QR_TRADER_POSITION_NAV_PCT",
        "QR_TRADER_BASE_UNITS",
        "QR_DISABLE_AUTO_CLOSE",
        # Added 2026-05-13 (feedback_broker_sl_noise_hunt.md): broker
        # SL on new entries and trailing SL are BOTH off by default
        # under the SL-free runtime. Routine cycles never silently
        # attach a tight broker SL or trail an existing one.
        "QR_NEW_ENTRY_INITIAL_SL",
        "QR_DISABLE_TRAILING_SL",
        "QR_REQUIRE_FORECAST_FOR_LIVE",
        "QR_REQUIRE_TELEMETRY_FOR_LIVE",
    )

    def setUp(self) -> None:
        self._prior: dict[str, str | None] = {}
        for key in (*self.SL_FREE_KEYS, "QR_LIVE_ENABLED"):
            self._prior[key] = os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_known_live_runtime_commands(self) -> None:
        # Guard against silent drift; widening this set requires explicit
        # review because adding a non-live command to it would leak the
        # SL-free env into unrelated test runs.
        self.assertEqual(
            _LIVE_RUNTIME_COMMANDS,
            frozenset(
                {
                    "autotrade-cycle",
                    "gpt-trader-decision",
                    "stage-live-order",
                    "generate-intents",
                    "trader-prompt-route",
                    # daily-target-state added 2026-05-13: protected flag
                    # computation reads `QR_TRADER_DISABLE_SL_REPAIR` and
                    # was producing protected=False for SL-free positions
                    # when invoked outside the autotrade-cycle wrapper.
                    "daily-target-state",
                    # position-management refreshes the existing-position
                    # sidecar before GPT/memory-health and must classify
                    # SL-free TP-only positions under live defaults.
                    "position-management",
                    # profit-partial-close reads broker truth / pair_charts
                    # and may send risk-reducing profit partial closes.
                    "profit-partial-close",
                    # completion-status audits live exposure and must classify
                    # SL-free trader positions under the same runtime defaults.
                    "completion-status",
                    # memory-health audits live routing memory before
                    # entry/verify routing and must use the same SL-free
                    # defaults when classifying broker snapshot state.
                    "memory-health",
                }
            ),
        )

    def test_gpt_trader_decision_bootstraps_without_qr_live_enabled(self) -> None:
        # In production the cli is invoked by the routine (not pytest), so
        # `_running_under_test_harness()` returns False and the
        # _LIVE_RUNTIME_COMMANDS gate fires. We mock the harness check
        # here so the unit test can verify the production gating without
        # bootstrapping SL-free into the rest of the unittest process.
        stdout = io.StringIO()
        with mock.patch("quant_rabbit.cli._running_under_test_harness", return_value=False):
            with redirect_stdout(stdout):
                code = main(["gpt-trader-decision", "--snapshot", "/dev/null"])
        self.assertEqual(code, 2)
        for key, expected in _SL_FREE_RUNTIME_DEFAULTS.items():
            self.assertEqual(os.environ.get(key), expected, key)

    def test_non_live_command_does_not_bootstrap(self) -> None:
        stdout = io.StringIO()
        with mock.patch("quant_rabbit.cli._running_under_test_harness", return_value=False):
            with self.assertRaises(SystemExit), redirect_stdout(stdout):
                main(["--help"])
        for key in self.SL_FREE_KEYS:
            self.assertIsNone(os.environ.get(key), key)

    def test_test_harness_blocks_bootstrap_without_qr_live_enabled(self) -> None:
        # The very behavior that prevents test pollution: live-runtime
        # commands skip the bootstrap under unittest unless QR_LIVE_ENABLED=1
        # is explicit.
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(["gpt-trader-decision", "--snapshot", "/dev/null"])
        self.assertEqual(code, 2)
        for key in self.SL_FREE_KEYS:
            self.assertIsNone(os.environ.get(key), key)

    def test_qr_live_enabled_forces_bootstrap_even_under_tests(self) -> None:
        # Tests that explicitly want the SL-free path opt in by setting
        # QR_LIVE_ENABLED=1 (mirrors `run-autotrade-live.sh`).
        os.environ["QR_LIVE_ENABLED"] = "1"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(["gpt-trader-decision", "--snapshot", "/dev/null"])
        self.assertEqual(code, 2)
        for key, expected in _SL_FREE_RUNTIME_DEFAULTS.items():
            self.assertEqual(os.environ.get(key), expected, key)

    def test_generate_intents_preflight_verifies_expired_projection_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "quotes": {
                            "EUR_USD": {
                                "bid": 1.1642,
                                "ask": 1.1643,
                            }
                        }
                    }
                )
            )
            expired = SimpleNamespace(
                resolution_status="PENDING",
                pair="EUR_USD",
                timestamp_emitted_utc=(datetime.now(timezone.utc) - timedelta(minutes=90))
                .isoformat()
                .replace("+00:00", "Z"),
                resolution_window_min=60.0,
            )
            os.environ["QR_PROJECTION_VERIFY_M1_COUNT"] = "0"
            os.environ["QR_PROJECTION_VERIFY_M5_COUNT"] = "0"
            try:
                with mock.patch("quant_rabbit.cli._running_under_test_harness", return_value=False):
                    with mock.patch("quant_rabbit.strategy.projection_ledger.load_ledger", return_value=[expired]):
                        with mock.patch(
                            "quant_rabbit.strategy.projection_ledger.verify_pending",
                            return_value={"HIT": 1, "MISS": 0, "TIMEOUT": 0, "PENDING": 0},
                        ) as verify_pending:
                            summary = _pre_entry_projection_verification_if_required(
                                telemetry_required=True,
                                snapshot_path=snapshot,
                                pair_charts_path=root / "missing_pair_charts.json",
                            )
            finally:
                os.environ.pop("QR_PROJECTION_VERIFY_M1_COUNT", None)
                os.environ.pop("QR_PROJECTION_VERIFY_M5_COUNT", None)

        self.assertEqual(summary["status"], "OK")
        self.assertEqual(summary["expired_pending_pairs"], 1)
        verify_pending.assert_called_once()

    def test_generate_intents_preflight_fetches_m5_projection_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "quotes": {
                            "EUR_USD": {
                                "bid": 1.1642,
                                "ask": 1.1643,
                            }
                        }
                    }
                )
            )
            expired = SimpleNamespace(
                resolution_status="PENDING",
                pair="EUR_USD",
                timestamp_emitted_utc=(datetime.now(timezone.utc) - timedelta(minutes=90))
                .isoformat()
                .replace("+00:00", "Z"),
                resolution_window_min=60.0,
            )
            os.environ["QR_PROJECTION_VERIFY_M1_COUNT"] = "0"
            os.environ["QR_PROJECTION_VERIFY_M5_COUNT"] = "2"
            candle = SimpleNamespace(
                timestamp_utc=datetime.now(timezone.utc) - timedelta(minutes=80),
                high=1.1660,
                low=1.1630,
                close=1.1650,
            )
            try:
                with mock.patch("quant_rabbit.cli._running_under_test_harness", return_value=False):
                    with mock.patch("quant_rabbit.strategy.projection_ledger.load_ledger", return_value=[expired]):
                        with mock.patch("quant_rabbit.cli.OandaReadOnlyClient", return_value=object()) as client_cls:
                            with mock.patch(
                                "quant_rabbit.analysis.candles.fetch_candles_via_client",
                                return_value=(candle,),
                            ) as fetch_candles:
                                with mock.patch(
                                    "quant_rabbit.strategy.projection_ledger.verify_pending",
                                    return_value={"HIT": 1, "MISS": 0, "TIMEOUT": 0, "PENDING": 0},
                                ) as verify_pending:
                                    summary = _pre_entry_projection_verification_if_required(
                                        telemetry_required=True,
                                        snapshot_path=snapshot,
                                        pair_charts_path=root / "missing_pair_charts.json",
                                    )
            finally:
                os.environ.pop("QR_PROJECTION_VERIFY_M1_COUNT", None)
                os.environ.pop("QR_PROJECTION_VERIFY_M5_COUNT", None)

        self.assertEqual(summary["status"], "OK")
        self.assertEqual(summary["candle_granularity_counts"], {"EUR_USD": {"M5": 1}})
        client_cls.assert_called_once()
        fetch_candles.assert_called_once_with(mock.ANY, "EUR_USD", "M5", count=2)
        candles_arg = verify_pending.call_args.kwargs["candles_by_pair"]
        self.assertEqual(candles_arg["EUR_USD"]["M5"], [candle])

    def test_generate_intents_preflight_fetches_retryable_truth_timeout_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "quotes": {
                            "EUR_USD": {
                                "bid": 1.1642,
                                "ask": 1.1643,
                            }
                        }
                    }
                )
            )
            timeout = SimpleNamespace(
                resolution_status="TIMEOUT",
                resolution_evidence="no M1 candle truth for projection window",
                pair="EUR_USD",
                timestamp_emitted_utc=(datetime.now(timezone.utc) - timedelta(minutes=90))
                .isoformat()
                .replace("+00:00", "Z"),
                resolution_window_min=60.0,
            )
            os.environ["QR_PROJECTION_VERIFY_M1_COUNT"] = "0"
            os.environ["QR_PROJECTION_VERIFY_M5_COUNT"] = "2"
            candle = SimpleNamespace(
                timestamp_utc=datetime.now(timezone.utc) - timedelta(minutes=80),
                high=1.1660,
                low=1.1630,
                close=1.1650,
            )
            try:
                with mock.patch("quant_rabbit.cli._running_under_test_harness", return_value=False):
                    with mock.patch("quant_rabbit.strategy.projection_ledger.load_ledger", return_value=[timeout]):
                        with mock.patch("quant_rabbit.cli.OandaReadOnlyClient", return_value=object()) as client_cls:
                            with mock.patch(
                                "quant_rabbit.analysis.candles.fetch_candles_via_client",
                                return_value=(candle,),
                            ) as fetch_candles:
                                with mock.patch(
                                    "quant_rabbit.strategy.projection_ledger.verify_pending",
                                    return_value={"HIT": 1, "MISS": 0, "TIMEOUT": 0, "PENDING": 0},
                                ) as verify_pending:
                                    summary = _pre_entry_projection_verification_if_required(
                                        telemetry_required=True,
                                        snapshot_path=snapshot,
                                        pair_charts_path=root / "missing_pair_charts.json",
                                    )
            finally:
                os.environ.pop("QR_PROJECTION_VERIFY_M1_COUNT", None)
                os.environ.pop("QR_PROJECTION_VERIFY_M5_COUNT", None)

        self.assertEqual(summary["status"], "OK")
        self.assertEqual(summary["expired_pending_pairs"], 0)
        self.assertEqual(summary["pending_pairs"], 0)
        self.assertEqual(summary["retryable_timeout_pairs"], 1)
        client_cls.assert_called_once()
        fetch_candles.assert_called_once_with(mock.ANY, "EUR_USD", "M5", count=2)
        candles_arg = verify_pending.call_args.kwargs["candles_by_pair"]
        self.assertEqual(candles_arg["EUR_USD"]["M5"], [candle])


if __name__ == "__main__":
    unittest.main()

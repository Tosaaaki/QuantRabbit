from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from quant_rabbit.cli import (
    _LIVE_RUNTIME_COMMANDS,
    _SL_FREE_RUNTIME_DEFAULTS,
    _refresh_current_forecast_history,
    main,
)


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

    def test_top_level_help_renders_daily_target_percent_text(self) -> None:
        stdout = io.StringIO()

        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("daily-target-state", help_text)
        self.assertIn("10% target", help_text)

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
            self.assertEqual(first["forecasts"]["EUR_USD"]["direction"], "UP")
            self.assertEqual(second["recorded"], 0)
            self.assertEqual(second["skipped"]["EUR_USD"], "already_recorded_for_cycle")
            rows = (data_root / "forecast_history.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 1)
            row = json.loads(rows[0])
            self.assertEqual(row["pair"], "EUR_USD")
            self.assertEqual(row["cycle_id"], "test:2026-05-30T00:00:00+00:00:2026-05-30T00:01:00+00:00")
            self.assertEqual(row["direction"], "UP")

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
                    "--send",
                    "--confirm-live",
                ])

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["dry_run"])
        self.assertTrue(payload["send"])
        self.assertTrue(payload["results"][0]["sent"])
        client.close_trade.assert_called_once_with("t-adverse", units="5000")

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
                    # profit-partial-close reads broker truth / pair_charts
                    # and may send risk-reducing profit partial closes.
                    "profit-partial-close",
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


if __name__ == "__main__":
    unittest.main()

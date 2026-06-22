from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from quant_rabbit.cli import (
    DIRECT_AUTOTRADE_AUDIT_SIDECARS_DIGEST,
    _LIVE_RUNTIME_COMMANDS,
    _SL_FREE_RUNTIME_DEFAULTS,
    _auto_refresh_market_evidence_if_required,
    _direct_autotrade_audit_sidecar_steps,
    _pre_entry_projection_verification_if_required,
    _refresh_current_forecast_history,
    _resolve_audit_execution_ledger_db,
    _resolve_audit_sidecar_path,
    _run_direct_autotrade_audit_sidecars,
    _snapshot_from_json,
    _snapshot_to_json,
    main,
)
from quant_rabbit.models import AccountSummary, BrokerOrder, BrokerSnapshot, Owner, Quote
from quant_rabbit.paths import DEFAULT_CAPTURE_ECONOMICS, DEFAULT_EXECUTION_LEDGER_DB, DEFAULT_MARKET_CONTEXT_MATRIX
from quant_rabbit.profitability_acceptance import (
    ProfitabilityAcceptanceAuditor,
    _execution_ledger_close_findings,
    _order_intent_metrics,
)
from quant_rabbit.strategy.intent_generator import _snapshot_from_json as _intent_snapshot_from_json


class RuntimeLedgerSelectionTest(unittest.TestCase):
    def _write_gateway_ledger(self, path: Path, latest_ts: str | None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                CREATE TABLE execution_events (
                    ts_utc TEXT,
                    source TEXT,
                    event_type TEXT
                )
                """
            )
            if latest_ts is not None:
                conn.execute(
                    """
                    INSERT INTO execution_events (ts_utc, source, event_type)
                    VALUES (?, 'gateway', 'GATEWAY_POSITION_NO_ACTION')
                    """,
                    (latest_ts,),
                )

    def test_default_audit_ledger_uses_fresher_live_gateway_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev_ledger = root / "dev" / "data" / "execution_ledger.db"
            live_root = root / "QuantRabbit-live"
            live_ledger = live_root / "data" / "execution_ledger.db"
            self._write_gateway_ledger(dev_ledger, "2026-06-01T00:00:00+00:00")
            self._write_gateway_ledger(live_ledger, "2026-06-22T00:00:00+00:00")

            resolved = _resolve_audit_execution_ledger_db(
                dev_ledger,
                default_path=dev_ledger,
                live_root=live_root,
            )

        self.assertEqual(resolved, live_ledger)

    def test_default_audit_ledger_keeps_requested_when_live_stream_not_fresher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev_ledger = root / "dev" / "data" / "execution_ledger.db"
            live_root = root / "QuantRabbit-live"
            self._write_gateway_ledger(dev_ledger, "2026-06-22T00:00:00+00:00")
            self._write_gateway_ledger(
                live_root / "data" / "execution_ledger.db",
                "2026-06-01T00:00:00+00:00",
            )

            resolved = _resolve_audit_execution_ledger_db(
                dev_ledger,
                default_path=dev_ledger,
                live_root=live_root,
            )

        self.assertEqual(resolved, dev_ledger)

    def test_explicit_audit_ledger_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_ledger = root / "dev" / "data" / "execution_ledger.db"
            explicit_ledger = root / "custom" / "execution_ledger.db"
            live_root = root / "QuantRabbit-live"
            self._write_gateway_ledger(default_ledger, "2026-06-01T00:00:00+00:00")
            self._write_gateway_ledger(explicit_ledger, "2026-06-01T00:00:00+00:00")
            self._write_gateway_ledger(
                live_root / "data" / "execution_ledger.db",
                "2026-06-22T00:00:00+00:00",
            )

            resolved = _resolve_audit_execution_ledger_db(
                explicit_ledger,
                default_path=default_ledger,
                live_root=live_root,
            )

        self.assertEqual(resolved, explicit_ledger)

    def test_default_audit_sidecar_follows_selected_live_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev_ledger = root / "dev" / "data" / "execution_ledger.db"
            dev_timing = root / "dev" / "data" / "execution_timing_audit.json"
            live_ledger = root / "QuantRabbit-live" / "data" / "execution_ledger.db"
            live_timing = root / "QuantRabbit-live" / "data" / "execution_timing_audit.json"
            dev_timing.parent.mkdir(parents=True, exist_ok=True)
            live_timing.parent.mkdir(parents=True, exist_ok=True)
            dev_timing.write_text("{}", encoding="utf-8")
            live_timing.write_text("{}", encoding="utf-8")

            resolved = _resolve_audit_sidecar_path(
                dev_timing,
                default_path=dev_timing,
                selected_ledger_path=live_ledger,
                default_ledger_path=dev_ledger,
            )

        self.assertEqual(resolved, live_timing)

    def test_explicit_audit_sidecar_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev_ledger = root / "dev" / "data" / "execution_ledger.db"
            dev_timing = root / "dev" / "data" / "execution_timing_audit.json"
            explicit_timing = root / "custom" / "execution_timing_audit.json"
            live_ledger = root / "QuantRabbit-live" / "data" / "execution_ledger.db"
            live_timing = root / "QuantRabbit-live" / "data" / "execution_timing_audit.json"
            explicit_timing.parent.mkdir(parents=True, exist_ok=True)
            live_timing.parent.mkdir(parents=True, exist_ok=True)
            explicit_timing.write_text("{}", encoding="utf-8")
            live_timing.write_text("{}", encoding="utf-8")

            resolved = _resolve_audit_sidecar_path(
                explicit_timing,
                default_path=dev_timing,
                selected_ledger_path=live_ledger,
                default_ledger_path=dev_ledger,
            )

        self.assertEqual(resolved, explicit_timing)

    def test_default_audit_sidecar_keeps_requested_when_live_sidecar_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev_ledger = root / "dev" / "data" / "execution_ledger.db"
            dev_timing = root / "dev" / "data" / "execution_timing_audit.json"
            live_ledger = root / "QuantRabbit-live" / "data" / "execution_ledger.db"
            dev_timing.parent.mkdir(parents=True, exist_ok=True)
            dev_timing.write_text("{}", encoding="utf-8")

            resolved = _resolve_audit_sidecar_path(
                dev_timing,
                default_path=dev_timing,
                selected_ledger_path=live_ledger,
                default_ledger_path=dev_ledger,
            )

        self.assertEqual(resolved, dev_timing)

    def test_profitability_acceptance_default_timing_follows_resolved_live_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev_timing = root / "dev" / "data" / "execution_timing_audit.json"
            live_ledger = root / "QuantRabbit-live" / "data" / "execution_ledger.db"
            live_timing = root / "QuantRabbit-live" / "data" / "execution_timing_audit.json"
            dev_timing.parent.mkdir(parents=True, exist_ok=True)
            live_timing.parent.mkdir(parents=True, exist_ok=True)
            dev_timing.write_text("{}", encoding="utf-8")
            live_timing.write_text("{}", encoding="utf-8")
            summary = SimpleNamespace(
                status="PROFITABILITY_ACCEPTANCE_PASSED",
                output_path=root / "acceptance.json",
                report_path=root / "acceptance.md",
                findings=[],
                blockers=[],
                metrics={},
            )
            auditor = mock.Mock()
            auditor.run.return_value = summary
            stdout = io.StringIO()

            with mock.patch(
                "quant_rabbit.cli.DEFAULT_EXECUTION_TIMING_AUDIT",
                dev_timing,
            ), mock.patch(
                "quant_rabbit.cli._resolve_audit_execution_ledger_db",
                return_value=live_ledger,
            ), mock.patch(
                "quant_rabbit.profitability_acceptance.ProfitabilityAcceptanceAuditor",
                return_value=auditor,
            ), redirect_stdout(stdout):
                code = main(
                    [
                        "profitability-acceptance",
                        "--order-intents",
                        str(root / "intents.json"),
                        "--target-state",
                        str(root / "target.json"),
                        "--self-improvement-audit",
                        str(root / "self_improvement.json"),
                        "--capture-economics",
                        str(root / "capture.json"),
                        "--projection-ledger",
                        str(root / "projection_ledger.jsonl"),
                        "--bidask-rules",
                        str(root / "bidask_rules.json"),
                        "--oanda-rotation-mining",
                        str(root / "oanda_rotation.json"),
                        "--output",
                        str(summary.output_path),
                        "--report",
                        str(summary.report_path),
                    ]
                )

        self.assertEqual(code, 0)
        self.assertEqual(
            auditor.run.call_args.kwargs["execution_timing_audit_path"],
            live_timing,
        )


class CliHelpTest(unittest.TestCase):
    def _write_oanda_firepower_report(
        self,
        path: Path,
        *,
        status: str,
        high_precision_count: int = 1,
        evidence_queue_count: int = 0,
        high_precision_daily_return_pct: float = 5.4,
        evidence_queue_daily_return_pct: float = 0.0,
    ) -> None:
        def section(count: int, daily_return_pct: float, status_label: str) -> dict[str, object]:
            top_vehicles = []
            if count > 0:
                top_vehicles.append(
                    {
                        "vehicle_key": f"EUR_USD|LONG|range_reclaim|tp1_sl1|{status_label}",
                        "evidence_status": status_label,
                        "pair": "EUR_USD",
                        "firepower_side": "LONG",
                        "validation_n": 80,
                        "active_days": 20,
                        "estimated_return_pct_per_active_day_at_observed_frequency": daily_return_pct,
                    }
                )
            return {
                "unique_vehicle_count": count,
                "pair_count": 1 if count else 0,
                "observed_attempts_per_active_day": 4.0 if count else 0.0,
                "weighted_return_pct_per_trade_at_risk_lens": 1.35 if count else 0.0,
                "estimated_return_pct_per_active_day_at_observed_frequency": daily_return_pct,
                "trades_needed_for_minimum_5pct_at_weighted_expectancy": 4 if count else None,
                "trades_needed_for_target_10pct_at_weighted_expectancy": 8 if count else None,
                "top_vehicles": top_vehicles,
            }

        path.write_text(
            json.dumps(
                {
                    "generated_at_utc": "2026-06-21T00:00:00Z",
                    "campaign_firepower": {
                        "contract": "audit-only firepower estimate; live gates still decide",
                        "per_trade_risk_pct_lens": 1.0,
                        "minimum_return_pct": 5.0,
                        "target_return_pct": 10.0,
                        "status": status,
                        "high_precision": section(
                            high_precision_count,
                            high_precision_daily_return_pct,
                            "HIGH_PRECISION_VALIDATED",
                        ),
                        "evidence_queue": section(
                            evidence_queue_count,
                            evidence_queue_daily_return_pct,
                            "EVIDENCE_COLLECTION_ONLY",
                        ),
                    },
                }
            )
        )

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

    def test_verify_projections_reports_economic_precision_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            snapshot.write_text(json.dumps({"quotes": {}}))
            hit_rates = {
                "session_expansion_london": {
                    "GBP_USD:TREND": {
                        "hit_rate": 0.98,
                        "samples": 100,
                        "economic_hit_rate": 0.96,
                        "economic_samples": 100,
                        "timeout_rate": 0.02,
                    },
                    "EUR_USD:TREND": {
                        "hit_rate": 0.98,
                        "samples": 100,
                        "economic_hit_rate": 0.50,
                        "economic_samples": 100,
                        "timeout_rate": 0.48,
                        "timeout_count": 48,
                    },
                },
                "directional_forecast_up": {
                    "GBP_USD:TREND": {
                        "hit_rate": 1.0,
                        "samples": 100,
                        "economic_hit_rate": 1.0,
                        "economic_samples": 100,
                    }
                },
            }
            stdout = io.StringIO()

            with mock.patch(
                "quant_rabbit.strategy.projection_ledger.load_ledger",
                return_value=[],
            ), mock.patch(
                "quant_rabbit.strategy.projection_ledger.retryable_truth_timeout_pairs",
                return_value=[],
            ), mock.patch(
                "quant_rabbit.strategy.projection_ledger.verify_pending",
                return_value={"HIT": 0, "MISS": 0, "TIMEOUT": 0, "PENDING": 0},
            ), mock.patch(
                "quant_rabbit.strategy.projection_ledger.compute_hit_rates",
                return_value=hit_rates,
            ), redirect_stdout(stdout):
                code = main(
                    [
                        "verify-projections",
                        "--snapshot",
                        str(snapshot),
                        "--pair-charts",
                        str(root / "missing_pair_charts.json"),
                        "--m1-count",
                        "0",
                        "--m5-count",
                        "0",
                    ]
                )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(
            payload["economic_precision_edges"][0]["signal_name"],
            "session_expansion_london",
        )
        self.assertEqual(payload["economic_precision_edges"][0]["pair"], "GBP_USD")
        self.assertTrue(payload["economic_precision_edges"][0]["passes_economic_precision"])
        self.assertEqual(payload["economic_precision_gaps"][0]["pair"], "EUR_USD")
        self.assertTrue(
            all(
                item["signal_name"] != "directional_forecast_up"
                for item in payload["economic_precision_edges"]
            )
        )

    def test_profitability_acceptance_blocks_systemic_profit_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.json"
            intents = root / "intents.json"
            self_audit = root / "self_improvement.json"
            capture = root / "capture.json"
            ledger = root / "execution_ledger.db"
            projection = root / "projection_ledger.jsonl"
            bidask = root / "bidask_rules.json"
            oanda_rotation = root / "oanda_rotation.json"
            output = root / "acceptance.json"
            report = root / "acceptance.md"

            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 5000.0}))
            intents.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T00:00:00+00:00",
                        "results": [
                            {
                                "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                                "status": "DRY_RUN_BLOCKED",
                                "risk_allowed": False,
                                "intent": {
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "order_type": "LIMIT",
                                    "market_context": {"method": "RANGE_ROTATION"},
                                    "metadata": {
                                        "capture_economics_trades": 29,
                                        "positive_rotation_mode": "TP_PROOF_COLLECTION_HARVEST",
                                        "self_improvement_p0_repair_live_ready": True,
                                        "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                                    }
                                },
                                "risk_issues": [{"code": "STALE_QUOTE"}, {"code": "SPREAD_TOO_WIDE"}],
                                "live_blocker_codes": [
                                    "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                    "RANGE_ROTATION_BROADER_LOCATION_CHASE",
                                ],
                                "live_blockers": ["SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE"],
                            }
                        ],
                    }
                )
            )
            self_audit.write_text(
                json.dumps(
                    {
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "profitability discipline has failed",
                            }
                        ],
                    }
                )
            )
            capture.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T00:00:10+00:00",
                        "status": "NEGATIVE_EXPECTANCY",
                        "overall": {
                            "trades": 30,
                            "net_jpy": -4056.9,
                            "expectancy_jpy_per_trade": -135.2,
                            "win_rate": 0.6667,
                            "payoff_ratio": 0.392,
                        },
                        "by_exit_reason": {
                            "TAKE_PROFIT_ORDER": {
                                "trades": 20,
                                "net_jpy": 11830.5,
                                "expectancy_jpy_per_trade": 591.5,
                            },
                            "MARKET_ORDER_TRADE_CLOSE": {
                                "trades": 7,
                                "net_jpy": -15091.7,
                                "expectancy_jpy_per_trade": -2156.0,
                            },
                        },
                        "segment_repair_priorities": {
                            "items": [
                                {
                                    "priority_class": "PRESERVE_TP_PROVEN_REPAIR_MARKET_CLOSE_LEAK",
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "method": "BREAKOUT_FAILURE",
                                    "take_profit_trades": 20,
                                    "take_profit_expectancy_jpy": 591.5,
                                    "market_close_net_jpy": -15091.7,
                                    "market_close_expectancy_jpy": -2156.0,
                                }
                            ]
                        },
                    }
                )
            )
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO execution_events (
                        ts_utc, event_type, trade_id, order_id, lane_id, pair, side,
                        realized_pl_jpy, exit_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "2026-06-21T00:00:00+00:00",
                            "GATEWAY_TRADE_CLOSE_RECONCILED",
                            "T-loss",
                            "O-loss",
                            "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                            "EUR_USD",
                            "LONG",
                            None,
                            "BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED",
                        ),
                        (
                            "2026-06-21T00:00:00+00:00",
                            "TRADE_CLOSED",
                            "T-loss",
                            "O-loss",
                            "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                            "EUR_USD",
                            "LONG",
                            -1234.5,
                            "MARKET_ORDER_TRADE_CLOSE",
                        ),
                    ],
                )
            bidask.write_text(
                json.dumps(
                    {
                        "contrarian_edge_rules": [
                            {
                                "name": "AUD_JPY_UP_FADE_TO_DOWN_RANK_ONLY",
                                "pair": "AUD_JPY",
                                "forecast_direction": "UP",
                                "direction": "DOWN",
                                "samples": 40,
                                "active_days": 2,
                                "positive_day_rate": 0.5,
                                "daily_stability_status": "INSUFFICIENT_ACTIVE_DAYS",
                                "optimized_profit_factor": 2.31,
                            }
                        ]
                    }
                )
            )
            self._write_oanda_firepower_report(
                oanda_rotation,
                status="VERIFIED_MINIMUM_5_ROUTE_ESTIMATED",
            )
            projection.write_text("")
            hit_rates = {
                "bb_squeeze_expansion_imminent": {
                    "EUR_USD:TREND": {
                        "hit_rate": 1.0,
                        "samples": 50,
                        "economic_hit_rate": 0.5,
                        "economic_samples": 100,
                        "timeout_rate": 0.5,
                        "timeout_count": 50,
                    }
                },
                "session_expansion_london": {
                    "GBP_USD:TREND": {
                        "hit_rate": 0.98,
                        "samples": 100,
                        "economic_hit_rate": 0.96,
                        "economic_samples": 100,
                    }
                },
            }
            stdout = io.StringIO()

            with mock.patch(
                "quant_rabbit.profitability_acceptance.compute_hit_rates",
                return_value=hit_rates,
            ), redirect_stdout(stdout):
                code = main(
                    [
                        "profitability-acceptance",
                        "--order-intents",
                        str(intents),
                        "--target-state",
                        str(target),
                        "--self-improvement-audit",
                        str(self_audit),
                        "--capture-economics",
                        str(capture),
                        "--execution-ledger-db",
                        str(ledger),
                        "--projection-ledger",
                        str(projection),
                        "--bidask-rules",
                        str(bidask),
                        "--oanda-rotation-mining",
                        str(oanda_rotation),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                    ]
                )
            output_exists = output.exists()
            report_exists = report.exists()

        self.assertEqual(code, 2)
        payload = json.loads(stdout.getvalue())
        codes = {item["code"] for item in payload["findings"]}
        self.assertEqual(payload["status"], "PROFITABILITY_ACCEPTANCE_BLOCKED")
        self.assertIn("SELF_IMPROVEMENT_P0_PRESENT", codes)
        self.assertIn("NEGATIVE_EXPECTANCY_ACTIVE", codes)
        self.assertIn("ORDER_INTENTS_CAPTURE_ECONOMICS_STALE", codes)
        self.assertIn("MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE", codes)
        self.assertIn("RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK", codes)
        self.assertIn("UNVERIFIED_LOSS_SIDE_MARKET_CLOSE_RECONCILED", codes)
        self.assertIn("PROJECTION_HEADLINE_PRECISION_ECONOMIC_GAP", codes)
        self.assertIn("BIDASK_CONTRARIAN_EDGE_NOT_DAILY_STABLE", codes)
        self.assertIn("NO_LIVE_READY_TARGET_COVERAGE", codes)
        self.assertIn("REPAIR_FRONTIER_BLOCKED", codes)
        frontier = payload["metrics"]["order_intents"]["repair_frontier"]
        self.assertEqual(frontier["candidate_count"], 1)
        self.assertEqual(frontier["live_ready_count"], 0)
        self.assertEqual(
            frontier["top_remaining_blockers"][0],
            {"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 1},
        )
        self.assertEqual(
            frontier["examples"][0]["blocker_codes"],
            ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "RANGE_ROTATION_BROADER_LOCATION_CHASE"],
        )
        self.assertEqual(
            payload["metrics"]["execution_ledger_close_leak"]["recent_unverified_loss_closes"],
            1,
        )
        self.assertTrue(output_exists)
        self.assertTrue(report_exists)

    def test_profitability_acceptance_treats_stale_gateway_stream_as_ledger_integrity_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "execution_ledger.db"
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        source TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT,
                        raw_json TEXT
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO execution_events (
                        ts_utc, source, event_type, trade_id, order_id, lane_id, pair, side,
                        realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "2026-06-01T00:00:00+00:00",
                            "gateway",
                            "GATEWAY_POSITION_NO_ACTION",
                            "T-old",
                            None,
                            None,
                            "EUR_USD",
                            None,
                            None,
                            "HOLD_PROTECTED",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:00:00+00:00",
                            "ledger_reconcile",
                            "GATEWAY_TRADE_CLOSE_RECONCILED",
                            "T-loss",
                            "O-loss",
                            "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                            "EUR_USD",
                            "LONG",
                            None,
                            "BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED",
                            json.dumps({"reconcile_reason": "NO_LOCAL_POSITION_EXECUTION_RECEIPT"}),
                        ),
                        (
                            "2026-06-21T00:00:10+00:00",
                            "oanda",
                            "TRADE_CLOSED",
                            "T-loss",
                            "O-loss",
                            "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                            "EUR_USD",
                            "LONG",
                            -1234.5,
                            "MARKET_ORDER_TRADE_CLOSE",
                            "{}",
                        ),
                    ],
                )

            metrics, findings = _execution_ledger_close_findings(ledger)

        codes = {item["code"] for item in findings}
        self.assertIn("RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK", codes)
        self.assertIn("EXECUTION_LEDGER_GATEWAY_RECEIPT_STREAM_STALE", codes)
        self.assertNotIn("UNVERIFIED_LOSS_SIDE_MARKET_CLOSE_RECONCILED", codes)
        self.assertTrue(metrics["gateway_event_stream_stale"])
        self.assertEqual(metrics["recent_unverified_loss_closes"], 1)

    def test_profitability_acceptance_counts_direct_gateway_close_sent_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "execution_ledger.db"
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        source TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT,
                        raw_json TEXT
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO execution_events (
                        ts_utc, source, event_type, trade_id, order_id, lane_id, pair, side,
                        realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "2026-06-21T00:00:00+00:00",
                            "gateway",
                            "GATEWAY_GPT_CLOSE_ACCEPTED",
                            "T-gpt",
                            None,
                            None,
                            "EUR_USD",
                            None,
                            None,
                            "GPT_CLOSE_ACCEPTED",
                            json.dumps({"decision": {"action": "CLOSE", "close_trade_ids": ["T-gpt"]}}),
                        ),
                        (
                            "2026-06-21T00:01:00+00:00",
                            "gateway",
                            "GATEWAY_TRADE_CLOSE_SENT",
                            "T-gpt",
                            "O-gpt",
                            "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                            "EUR_USD",
                            "LONG",
                            None,
                            "GPT_CLOSE",
                            json.dumps({"request": {"type": "CLOSE", "trade_id": "T-gpt"}}),
                        ),
                        (
                            "2026-06-21T00:01:05+00:00",
                            "oanda",
                            "TRADE_CLOSED",
                            "T-gpt",
                            "O-gpt",
                            "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                            "EUR_USD",
                            "LONG",
                            -250.0,
                            "MARKET_ORDER_TRADE_CLOSE",
                            "{}",
                        ),
                    ],
                )

            metrics, findings = _execution_ledger_close_findings(ledger)

        codes = {item["code"] for item in findings}
        self.assertIn("RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK", codes)
        self.assertNotIn("UNVERIFIED_LOSS_SIDE_MARKET_CLOSE_RECONCILED", codes)
        self.assertEqual(metrics["latest_gateway_market_close_ts_utc"], "2026-06-21T00:01:00+00:00")
        self.assertEqual(metrics["recent_loss_closes"], 1)
        self.assertEqual(metrics["recent_unverified_loss_closes"], 0)
        self.assertEqual(
            metrics["recent_loss_examples"][0]["close_provenance"],
            "GATEWAY_TRADE_CLOSE_SENT",
        )

    def test_profitability_acceptance_excludes_timing_contained_loss_from_recent_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            timing = root / "execution_timing_audit.json"
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT,
                        raw_json TEXT
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO execution_events (
                        ts_utc, event_type, trade_id, order_id, lane_id, pair, side,
                        realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "2026-06-21T00:00:00+00:00",
                            "GATEWAY_TRADE_CLOSE_SENT",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            None,
                            "GPT_CLOSE",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:00:05+00:00",
                            "TRADE_CLOSED",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            -1019.78,
                            "MARKET_ORDER_TRADE_CLOSE",
                            "{}",
                        ),
                    ],
                )
            timing.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T01:00:00+00:00",
                        "market_close_counterfactuals": [
                            {
                                "trade_id": "T-contained",
                                "order_id": "O-contained",
                                "post_close_path_label": "LOSS_CLOSE_CONTAINED_RISK",
                            }
                        ],
                    }
                )
            )

            metrics, findings = _execution_ledger_close_findings(
                ledger,
                execution_timing_audit_path=timing,
            )

        codes = {item["code"] for item in findings}
        self.assertNotIn("RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK", codes)
        self.assertNotIn("UNVERIFIED_LOSS_SIDE_MARKET_CLOSE_RECONCILED", codes)
        self.assertEqual(metrics["recent_loss_closes"], 1)
        self.assertEqual(metrics["recent_contained_risk_loss_closes"], 1)
        self.assertEqual(metrics["recent_leak_loss_closes"], 0)
        self.assertEqual(
            metrics["recent_loss_timing_label_counts"],
            {"LOSS_CLOSE_CONTAINED_RISK": 1},
        )

    def test_profitability_acceptance_requires_close_gate_evidence_for_contained_gpt_loss_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            timing = root / "execution_timing_audit.json"
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT,
                        raw_json TEXT
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO execution_events (
                        ts_utc, event_type, trade_id, order_id, lane_id, pair, side,
                        realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "2026-06-21T00:00:00+00:00",
                            "GATEWAY_TRADE_CLOSE_SENT",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            None,
                            "GPT_CLOSE",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:00:05+00:00",
                            "TRADE_CLOSED",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            -1019.78,
                            "MARKET_ORDER_TRADE_CLOSE",
                            "{}",
                        ),
                    ],
                )
            timing.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T01:00:00+00:00",
                        "market_close_counterfactuals": [
                            {
                                "trade_id": "T-contained",
                                "order_id": "O-contained",
                                "post_close_path_label": "LOSS_CLOSE_CONTAINED_RISK",
                            }
                        ],
                    }
                )
            )

            metrics, findings = _execution_ledger_close_findings(
                ledger,
                execution_timing_audit_path=timing,
            )

        codes = {item["code"] for item in findings}
        self.assertNotIn("RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK", codes)
        self.assertIn("LOSS_CLOSE_GATE_EVIDENCE_MISSING", codes)
        self.assertEqual(metrics["recent_contained_risk_loss_closes"], 1)
        self.assertEqual(metrics["recent_close_gate_unverified_loss_closes"], 1)

    def test_profitability_acceptance_accepts_contained_gpt_loss_close_with_close_gate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            timing = root / "execution_timing_audit.json"
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT,
                        raw_json TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE verification_observations (
                        ts_utc TEXT,
                        subject_id TEXT,
                        check_name TEXT,
                        status TEXT
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO execution_events (
                        ts_utc, event_type, trade_id, order_id, lane_id, pair, side,
                        realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "2026-06-21T00:00:00+00:00",
                            "GATEWAY_TRADE_CLOSE_SENT",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            None,
                            "GPT_CLOSE",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:00:05+00:00",
                            "TRADE_CLOSED",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            -1019.78,
                            "MARKET_ORDER_TRADE_CLOSE",
                            "{}",
                        ),
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO verification_observations (
                        ts_utc, subject_id, check_name, status
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        "2026-06-21T00:00:00+00:00",
                        "T-contained",
                        "close_gate_evidence",
                        "PASS",
                    ),
                )
            timing.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T01:00:00+00:00",
                        "market_close_counterfactuals": [
                            {
                                "trade_id": "T-contained",
                                "order_id": "O-contained",
                                "post_close_path_label": "LOSS_CLOSE_CONTAINED_RISK",
                            }
                        ],
                    }
                )
            )

            metrics, findings = _execution_ledger_close_findings(
                ledger,
                execution_timing_audit_path=timing,
            )

        codes = {item["code"] for item in findings}
        self.assertNotIn("RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK", codes)
        self.assertNotIn("LOSS_CLOSE_GATE_EVIDENCE_MISSING", codes)
        self.assertEqual(metrics["recent_contained_risk_loss_closes"], 1)
        self.assertEqual(metrics["recent_close_gate_unverified_loss_closes"], 0)

    def test_profitability_acceptance_matches_close_gate_evidence_at_gpt_accept_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            timing = root / "execution_timing_audit.json"
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT,
                        raw_json TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE verification_observations (
                        ts_utc TEXT,
                        subject_id TEXT,
                        check_name TEXT,
                        status TEXT
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO execution_events (
                        ts_utc, event_type, trade_id, order_id, lane_id, pair, side,
                        realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "2026-06-21T00:00:00+00:00",
                            "GATEWAY_GPT_CLOSE_ACCEPTED",
                            "T-contained",
                            None,
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            None,
                            "GPT_CLOSE_ACCEPTED",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:01:30+00:00",
                            "GATEWAY_TRADE_CLOSE_SENT",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            None,
                            "GPT_CLOSE",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:01:35+00:00",
                            "TRADE_CLOSED",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            -1019.78,
                            "MARKET_ORDER_TRADE_CLOSE",
                            "{}",
                        ),
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO verification_observations (
                        ts_utc, subject_id, check_name, status
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        "2026-06-21T00:00:00+00:00",
                        "T-contained",
                        "close_gate_evidence",
                        "PASS",
                    ),
                )
            timing.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T01:00:00+00:00",
                        "market_close_counterfactuals": [
                            {
                                "trade_id": "T-contained",
                                "order_id": "O-contained",
                                "post_close_path_label": "LOSS_CLOSE_CONTAINED_RISK",
                            }
                        ],
                    }
                )
            )

            metrics, findings = _execution_ledger_close_findings(
                ledger,
                execution_timing_audit_path=timing,
            )

        codes = {item["code"] for item in findings}
        self.assertNotIn("RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK", codes)
        self.assertNotIn("LOSS_CLOSE_GATE_EVIDENCE_MISSING", codes)
        self.assertEqual(metrics["recent_contained_risk_loss_closes"], 1)
        self.assertEqual(metrics["recent_close_gate_unverified_loss_closes"], 0)

    def test_profitability_acceptance_does_not_reuse_stale_close_gate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            timing = root / "execution_timing_audit.json"
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT,
                        raw_json TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE verification_observations (
                        ts_utc TEXT,
                        subject_id TEXT,
                        check_name TEXT,
                        status TEXT
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO execution_events (
                        ts_utc, event_type, trade_id, order_id, lane_id, pair, side,
                        realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "2026-06-21T00:00:00+00:00",
                            "GATEWAY_GPT_CLOSE_ACCEPTED",
                            "T-contained",
                            None,
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            None,
                            "GPT_CLOSE_ACCEPTED",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:01:00+00:00",
                            "GATEWAY_GPT_CLOSE_ACCEPTED",
                            "T-contained",
                            None,
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            None,
                            "GPT_CLOSE_ACCEPTED",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:01:30+00:00",
                            "GATEWAY_TRADE_CLOSE_SENT",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            None,
                            "GPT_CLOSE",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:01:35+00:00",
                            "TRADE_CLOSED",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            -1019.78,
                            "MARKET_ORDER_TRADE_CLOSE",
                            "{}",
                        ),
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO verification_observations (
                        ts_utc, subject_id, check_name, status
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        "2026-06-21T00:00:00+00:00",
                        "T-contained",
                        "close_gate_evidence",
                        "PASS",
                    ),
                )
            timing.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T01:00:00+00:00",
                        "market_close_counterfactuals": [
                            {
                                "trade_id": "T-contained",
                                "order_id": "O-contained",
                                "post_close_path_label": "LOSS_CLOSE_CONTAINED_RISK",
                            }
                        ],
                    }
                )
            )

            metrics, findings = _execution_ledger_close_findings(
                ledger,
                execution_timing_audit_path=timing,
            )

        codes = {item["code"] for item in findings}
        self.assertIn("LOSS_CLOSE_GATE_EVIDENCE_MISSING", codes)
        self.assertEqual(metrics["recent_close_gate_unverified_loss_closes"], 1)

    def test_profitability_acceptance_counts_only_premature_timing_losses_as_recent_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            timing = root / "execution_timing_audit.json"
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT,
                        raw_json TEXT
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO execution_events (
                        ts_utc, event_type, trade_id, order_id, lane_id, pair, side,
                        realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "2026-06-21T00:00:00+00:00",
                            "GATEWAY_TRADE_CLOSE_SENT",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            None,
                            "GPT_CLOSE",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:00:05+00:00",
                            "TRADE_CLOSED",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            -1019.78,
                            "MARKET_ORDER_TRADE_CLOSE",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:05:00+00:00",
                            "GATEWAY_TRADE_CLOSE_SENT",
                            "T-premature",
                            "O-premature",
                            "range_trader:NZD_USD:LONG:RANGE_ROTATION",
                            "NZD_USD",
                            "LONG",
                            None,
                            "GPT_CLOSE",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:05:05+00:00",
                            "TRADE_CLOSED",
                            "T-premature",
                            "O-premature",
                            "range_trader:NZD_USD:LONG:RANGE_ROTATION",
                            "NZD_USD",
                            "LONG",
                            -1380.8,
                            "MARKET_ORDER_TRADE_CLOSE",
                            "{}",
                        ),
                    ],
                )
            timing.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T01:00:00+00:00",
                        "market_close_counterfactuals": [
                            {
                                "trade_id": "T-contained",
                                "order_id": "O-contained",
                                "post_close_path_label": "LOSS_CLOSE_CONTAINED_RISK",
                            },
                            {
                                "trade_id": "T-premature",
                                "order_id": "O-premature",
                                "post_close_path_label": "LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE",
                            },
                        ],
                    }
                )
            )

            metrics, findings = _execution_ledger_close_findings(
                ledger,
                execution_timing_audit_path=timing,
            )

        leak = next(
            finding
            for finding in findings
            if finding["code"] == "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK"
        )
        self.assertEqual(metrics["recent_loss_closes"], 2)
        self.assertEqual(metrics["recent_contained_risk_loss_closes"], 1)
        self.assertEqual(metrics["recent_premature_loss_closes"], 1)
        self.assertEqual(metrics["recent_leak_loss_closes"], 1)
        self.assertEqual(leak["evidence"]["recent_leak_loss_closes"], 1)
        self.assertEqual(
            leak["evidence"]["by_lane"][0]["lane_id"],
            "range_trader:NZD_USD:LONG:RANGE_ROTATION",
        )

    def test_profitability_acceptance_keeps_unverified_p0_for_contained_reconciled_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            timing = root / "execution_timing_audit.json"
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        source TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT,
                        raw_json TEXT
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO execution_events (
                        ts_utc, source, event_type, trade_id, order_id, lane_id, pair, side,
                        realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "2026-06-21T00:00:00+00:00",
                            "gateway",
                            "GATEWAY_POSITION_NO_ACTION",
                            "T-old",
                            None,
                            None,
                            "EUR_USD",
                            None,
                            None,
                            "HOLD_PROTECTED",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:10:00+00:00",
                            "ledger_reconcile",
                            "GATEWAY_TRADE_CLOSE_RECONCILED",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            None,
                            "BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED",
                            json.dumps({"reconcile_reason": "NO_LOCAL_POSITION_EXECUTION_RECEIPT"}),
                        ),
                        (
                            "2026-06-21T00:10:05+00:00",
                            "oanda",
                            "TRADE_CLOSED",
                            "T-contained",
                            "O-contained",
                            "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
                            "EUR_CHF",
                            "LONG",
                            -1019.78,
                            "MARKET_ORDER_TRADE_CLOSE",
                            "{}",
                        ),
                    ],
                )
            timing.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T01:00:00+00:00",
                        "market_close_counterfactuals": [
                            {
                                "trade_id": "T-contained",
                                "order_id": "O-contained",
                                "post_close_path_label": "LOSS_CLOSE_CONTAINED_RISK",
                            }
                        ],
                    }
                )
            )

            metrics, findings = _execution_ledger_close_findings(
                ledger,
                execution_timing_audit_path=timing,
            )

        codes = {item["code"] for item in findings}
        self.assertNotIn("RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK", codes)
        self.assertIn("UNVERIFIED_LOSS_SIDE_MARKET_CLOSE_RECONCILED", codes)
        self.assertEqual(metrics["recent_contained_risk_loss_closes"], 1)
        self.assertEqual(metrics["recent_leak_loss_closes"], 0)
        self.assertEqual(metrics["recent_unverified_loss_closes"], 1)

    def test_profitability_acceptance_backfills_close_leak_lane_from_entry_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "execution_ledger.db"
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT,
                        raw_json TEXT
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO execution_events (
                        ts_utc, event_type, trade_id, order_id, lane_id, pair, side,
                        realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "2026-06-21T00:00:00+00:00",
                            "ORDER_FILLED",
                            "T-loss",
                            "O-entry",
                            "range_trader:NZD_USD:LONG:RANGE_ROTATION",
                            "NZD_USD",
                            "LONG",
                            0.0,
                            "LIMIT_ORDER",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:30:00+00:00",
                            "GATEWAY_TRADE_CLOSE_SENT",
                            "T-loss",
                            "O-close",
                            "",
                            "NZD_USD",
                            "",
                            None,
                            "GPT_CLOSE",
                            "{}",
                        ),
                        (
                            "2026-06-21T00:30:01+00:00",
                            "TRADE_CLOSED",
                            "T-loss",
                            "O-close",
                            "",
                            "NZD_USD",
                            "LONG",
                            -1380.8,
                            "MARKET_ORDER_TRADE_CLOSE",
                            "{}",
                        ),
                    ],
                )

            metrics, findings = _execution_ledger_close_findings(ledger)

        self.assertEqual(metrics["recent_loss_closes"], 1)
        self.assertEqual(
            metrics["recent_loss_examples"][0]["lane_id"],
            "range_trader:NZD_USD:LONG:RANGE_ROTATION",
        )
        self.assertEqual(
            metrics["recent_loss_examples"][0]["close_provenance"],
            "GATEWAY_TRADE_CLOSE_SENT",
        )
        self.assertEqual(
            metrics["recent_loss_by_lane"][0],
            {
                "lane_id": "range_trader:NZD_USD:LONG:RANGE_ROTATION",
                "pair": "NZD_USD",
                "side": "LONG",
                "method": "RANGE_ROTATION",
                "loss_closes": 1,
                "net_jpy": -1380.8,
            },
        )
        leak = next(
            finding
            for finding in findings
            if finding["code"] == "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK"
        )
        self.assertEqual(leak["evidence"]["by_lane"], metrics["recent_loss_by_lane"])

    def test_profitability_acceptance_does_not_call_gpt_reconciled_loss_close_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "execution_ledger.db"
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT,
                        raw_json TEXT
                    )
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO execution_events (
                        ts_utc, event_type, trade_id, order_id, lane_id, pair, side,
                        realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "2026-06-21T00:00:00+00:00",
                            "GATEWAY_GPT_CLOSE_ACCEPTED",
                            "T-gpt",
                            None,
                            None,
                            "EUR_USD",
                            None,
                            None,
                            "GPT_CLOSE_ACCEPTED",
                            json.dumps({"decision": {"action": "CLOSE", "close_trade_ids": ["T-gpt"]}}),
                        ),
                        (
                            "2026-06-21T00:00:10+00:00",
                            "GATEWAY_TRADE_CLOSE_RECONCILED",
                            "T-gpt",
                            "O-gpt",
                            None,
                            "EUR_USD",
                            None,
                            None,
                            "GPT_CLOSE_RECONCILED",
                            json.dumps({"reconciled_from": ["GATEWAY_GPT_CLOSE_ACCEPTED"]}),
                        ),
                        (
                            "2026-06-21T00:00:10+00:00",
                            "TRADE_CLOSED",
                            "T-gpt",
                            "O-gpt",
                            None,
                            "EUR_USD",
                            "LONG",
                            -250.0,
                            "MARKET_ORDER_TRADE_CLOSE",
                            "{}",
                        ),
                    ],
                )

            metrics, findings = _execution_ledger_close_findings(ledger)

        codes = {item["code"] for item in findings}
        self.assertIn("RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK", codes)
        self.assertNotIn("UNVERIFIED_LOSS_SIDE_MARKET_CLOSE_RECONCILED", codes)
        self.assertEqual(metrics["recent_unverified_loss_closes"], 0)
        self.assertEqual(
            metrics["recent_loss_examples"][0]["close_provenance"],
            "GATEWAY_GPT_CLOSE_RECONCILED",
        )

    def test_profitability_acceptance_blocks_evidence_queue_oanda_firepower(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.json"
            intents = root / "intents.json"
            self_audit = root / "self_improvement.json"
            capture = root / "capture.json"
            ledger = root / "execution_ledger.db"
            projection = root / "projection_ledger.jsonl"
            bidask = root / "bidask_rules.json"
            oanda_rotation = root / "oanda_rotation.json"
            output = root / "acceptance.json"
            report = root / "acceptance.md"

            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 5000.0}))
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                                "status": "LIVE_READY",
                                "risk_issues": [],
                                "live_blockers": [],
                            }
                        ]
                    }
                )
            )
            self_audit.write_text(json.dumps({"status": "SELF_IMPROVEMENT_OK", "findings": []}))
            capture.write_text(
                json.dumps(
                    {
                        "status": "POSITIVE_EXPECTANCY",
                        "overall": {
                            "trades": 40,
                            "net_jpy": 12000.0,
                            "expectancy_jpy_per_trade": 300.0,
                            "win_rate": 0.7,
                            "payoff_ratio": 1.2,
                        },
                        "by_exit_reason": {},
                        "segment_repair_priorities": {"items": []},
                    }
                )
            )
            bidask.write_text(json.dumps({"contrarian_edge_rules": []}))
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT
                    )
                    """
                )
            projection.write_text("")
            self._write_oanda_firepower_report(
                oanda_rotation,
                status="EVIDENCE_QUEUE_ONLY_NO_VERIFIED_FIREPOWER",
                high_precision_count=0,
                evidence_queue_count=1,
                high_precision_daily_return_pct=0.0,
                evidence_queue_daily_return_pct=8.0,
            )
            hit_rates = {
                "session_expansion_london": {
                    "GBP_USD:TREND": {
                        "hit_rate": 0.98,
                        "samples": 100,
                        "economic_hit_rate": 0.96,
                        "economic_samples": 100,
                    }
                }
            }
            stdout = io.StringIO()

            with mock.patch(
                "quant_rabbit.profitability_acceptance.compute_hit_rates",
                return_value=hit_rates,
            ), redirect_stdout(stdout):
                code = main(
                    [
                        "profitability-acceptance",
                        "--order-intents",
                        str(intents),
                        "--target-state",
                        str(target),
                        "--self-improvement-audit",
                        str(self_audit),
                        "--capture-economics",
                        str(capture),
                        "--execution-ledger-db",
                        str(ledger),
                        "--projection-ledger",
                        str(projection),
                        "--bidask-rules",
                        str(bidask),
                        "--oanda-rotation-mining",
                        str(oanda_rotation),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                    ]
                )

        self.assertEqual(code, 2)
        payload = json.loads(stdout.getvalue())
        codes = [item["code"] for item in payload["findings"]]
        self.assertEqual(payload["status"], "PROFITABILITY_ACCEPTANCE_BLOCKED")
        self.assertEqual(codes, ["OANDA_CAMPAIGN_FIREPOWER_UNVERIFIED"])
        firepower = payload["metrics"]["oanda_campaign_firepower"]
        self.assertEqual(firepower["status"], "EVIDENCE_QUEUE_ONLY_NO_VERIFIED_FIREPOWER")
        self.assertEqual(firepower["high_precision"]["unique_vehicle_count"], 0)
        self.assertEqual(firepower["evidence_queue"]["unique_vehicle_count"], 1)
        self.assertEqual(
            firepower["evidence_queue"]["estimated_return_pct_per_active_day_at_observed_frequency"],
            8.0,
        )

    def test_profitability_acceptance_uses_packaged_oanda_firepower_when_latest_log_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.json"
            intents = root / "intents.json"
            self_audit = root / "self_improvement.json"
            capture = root / "capture.json"
            ledger = root / "execution_ledger.db"
            projection = root / "projection_ledger.jsonl"
            bidask = root / "bidask_rules.json"
            missing_latest = root / "logs" / "reports" / "forecast_improvement" / "missing_latest.json"
            packaged = root / "src" / "quant_rabbit" / "oanda_universal_rotation_precision_rules.json"
            output = root / "acceptance.json"
            report = root / "acceptance.md"

            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 5000.0}))
            intents.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T00:05:00+00:00",
                        "results": [
                            {
                                "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                                "status": "LIVE_READY",
                                "risk_issues": [],
                                "live_blockers": [],
                            }
                        ],
                    }
                )
            )
            self_audit.write_text(json.dumps({"status": "SELF_IMPROVEMENT_OK", "findings": []}))
            capture.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-21T00:04:00+00:00",
                        "status": "POSITIVE_EXPECTANCY",
                        "overall": {
                            "trades": 40,
                            "net_jpy": 12000.0,
                            "expectancy_jpy_per_trade": 300.0,
                            "win_rate": 0.7,
                            "payoff_ratio": 1.2,
                        },
                        "by_exit_reason": {},
                        "segment_repair_priorities": {"items": []},
                    }
                )
            )
            bidask.write_text(json.dumps({"contrarian_edge_rules": []}))
            packaged.parent.mkdir(parents=True, exist_ok=True)
            self._write_oanda_firepower_report(
                packaged,
                status="VERIFIED_TARGET_10_ROUTE_ESTIMATED",
                high_precision_count=3,
                high_precision_daily_return_pct=11.0,
            )
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT
                    )
                    """
                )
            projection.write_text("")

            with mock.patch(
                "quant_rabbit.profitability_acceptance.DEFAULT_OANDA_UNIVERSAL_ROTATION_MINING",
                missing_latest,
            ), mock.patch(
                "quant_rabbit.profitability_acceptance.DEFAULT_OANDA_UNIVERSAL_ROTATION_PACKAGED_RULES",
                packaged,
            ), mock.patch(
                "quant_rabbit.profitability_acceptance.compute_hit_rates",
                return_value={
                    "session_expansion_london": {
                        "EUR_USD:TREND": {
                            "hit_rate": 0.98,
                            "samples": 100,
                            "economic_hit_rate": 0.96,
                            "economic_samples": 100,
                        }
                    }
                },
            ):
                summary = ProfitabilityAcceptanceAuditor(
                    output_path=output,
                    report_path=report,
                ).run(
                    order_intents_path=intents,
                    target_state_path=target,
                    self_improvement_path=self_audit,
                    capture_economics_path=capture,
                    execution_ledger_path=ledger,
                    projection_ledger_path=projection,
                    bidask_rules_path=bidask,
                    oanda_rotation_mining_path=missing_latest,
                )

            payload = json.loads(output.read_text())
            codes = {item["code"] for item in payload["findings"]}
            self.assertEqual(summary.status, "PROFITABILITY_ACCEPTANCE_PASSED")
            self.assertNotIn("OANDA_CAMPAIGN_FIREPOWER_REPORT_MISSING", codes)
            firepower = payload["metrics"]["oanda_campaign_firepower"]
            self.assertEqual(firepower["path"], str(packaged))
            self.assertEqual(firepower["status"], "VERIFIED_TARGET_10_ROUTE_ESTIMATED")

    def test_profitability_acceptance_order_metrics_prefers_live_blocker_codes(self) -> None:
        metrics = _order_intent_metrics(
            {
                "generated_at_utc": "2026-06-21T12:00:00+00:00",
                "results": [
                    {
                        "status": "DRY_RUN_BLOCKED",
                        "risk_issues": [
                            {
                                "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                "message": "EUR_USD LONG forecast confidence is below live floor",
                                "severity": "BLOCK",
                            }
                        ],
                        "live_blocker_codes": ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"],
                        "live_blockers": [
                            "EUR_USD LONG forecast confidence 0.44 < 0.65 for live send",
                        ],
                    },
                    {
                        "status": "DRY_RUN_BLOCKED",
                        "risk_issues": [
                            {
                                "code": "SPREAD_TOO_WIDE",
                                "message": "spread is wider than session cap",
                                "severity": "BLOCK",
                            }
                        ],
                        "live_blocker_codes": ["SPREAD_TOO_WIDE"],
                        "live_blockers": ["spread too wide: current spread exceeds cap"],
                    },
                    {
                        "status": "DRY_RUN_BLOCKED",
                        "risk_issues": [],
                        "live_blockers": [
                            {"code": "LEGACY_BLOCKER", "message": "legacy blocker detail"},
                        ],
                    },
                    {
                        "status": "DRY_RUN_BLOCKED",
                        "risk_issues": [
                            {"code": "WARN_ONLY_DIAGNOSTIC", "severity": "WARN"},
                            {"code": "LEGACY_RISK_BLOCK", "severity": "BLOCK"},
                        ],
                        "live_blockers": ["legacy risk detail"],
                    },
                ],
            }
        )

        blockers = {item["code"]: item["count"] for item in metrics["top_blockers"]}
        self.assertEqual(blockers["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"], 1)
        self.assertEqual(blockers["SPREAD_TOO_WIDE"], 1)
        self.assertEqual(blockers["LEGACY_BLOCKER"], 1)
        self.assertEqual(blockers["LEGACY_RISK_BLOCK"], 1)
        self.assertNotIn("WARN_ONLY_DIAGNOSTIC", blockers)
        self.assertNotIn("EUR_USD LONG forecast confidence 0.44 < 0.65 for live send", blockers)
        self.assertNotIn("spread too wide", blockers)

    def test_profitability_acceptance_order_metrics_reports_repair_frontier(self) -> None:
        metrics = _order_intent_metrics(
            {
                "generated_at_utc": "2026-06-21T12:00:00+00:00",
                "results": [
                    {
                        "lane_id": "range_trader:USD_JPY:LONG:RANGE_ROTATION",
                        "status": "DRY_RUN_BLOCKED",
                        "risk_allowed": False,
                        "intent": {
                            "pair": "USD_JPY",
                            "side": "LONG",
                            "order_type": "LIMIT",
                            "market_context": {"method": "RANGE_ROTATION"},
                            "metadata": {
                                "positive_rotation_mode": "OANDA_CAMPAIGN_FIREPOWER_HARVEST",
                                "self_improvement_p0_repair_live_ready": True,
                                "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                                "positive_rotation_oanda_campaign_matching_vehicle_key": (
                                    "USD_JPY|LONG|range_reversion|tp1_sl1"
                                ),
                            },
                        },
                        "live_blocker_codes": [
                            "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                            "RANGE_PHASE_NOT_ROTATION",
                        ],
                    },
                    {
                        "lane_id": "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION",
                        "status": "DRY_RUN_BLOCKED",
                        "intent": {
                            "pair": "EUR_USD",
                            "side": "SHORT",
                            "order_type": "MARKET",
                            "market_context": {"method": "TREND_CONTINUATION"},
                            "metadata": {},
                        },
                        "live_blocker_codes": ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"],
                    },
                    {
                        "lane_id": "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT",
                        "status": "LIVE_READY",
                        "risk_allowed": True,
                        "intent": {
                            "pair": "AUD_JPY",
                            "side": "SHORT",
                            "order_type": "LIMIT",
                            "market_context": {"method": "BREAKOUT_FAILURE"},
                            "metadata": {
                                "positive_rotation_mode": "TP_PROOF_COLLECTION_HARVEST",
                                "self_improvement_p0_repair_live_ready": True,
                                "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                            },
                        },
                        "live_blocker_codes": [],
                    },
                ],
            }
        )

        frontier = metrics["repair_frontier"]
        self.assertEqual(frontier["candidate_count"], 2)
        self.assertEqual(frontier["live_ready_count"], 1)
        self.assertEqual(frontier["blocked_count"], 1)
        self.assertEqual(
            frontier["top_remaining_blockers"],
            [
                {"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 1},
                {"code": "RANGE_PHASE_NOT_ROTATION", "count": 1},
            ],
        )
        self.assertEqual(frontier["examples"][0]["pair"], "AUD_JPY")
        self.assertEqual(frontier["examples"][0]["status"], "LIVE_READY")
        self.assertEqual(frontier["examples"][1]["pair"], "USD_JPY")
        self.assertEqual(
            frontier["examples"][1]["blocker_codes"],
            ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "RANGE_PHASE_NOT_ROTATION"],
        )

    def test_profitability_acceptance_passes_when_profit_invariants_are_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.json"
            intents = root / "intents.json"
            self_audit = root / "self_improvement.json"
            capture = root / "capture.json"
            ledger = root / "execution_ledger.db"
            projection = root / "projection_ledger.jsonl"
            bidask = root / "bidask_rules.json"
            oanda_rotation = root / "oanda_rotation.json"
            output = root / "acceptance.json"
            report = root / "acceptance.md"

            target.write_text(json.dumps({"status": "PURSUE_TARGET", "remaining_target_jpy": 5000.0}))
            intents.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "lane_id": "failure_trader:GBP_USD:SHORT:BREAKOUT_FAILURE",
                                "status": "LIVE_READY",
                                "risk_issues": [],
                                "live_blockers": [],
                            }
                        ]
                    }
                )
            )
            self_audit.write_text(json.dumps({"status": "SELF_IMPROVEMENT_OK", "findings": []}))
            capture.write_text(
                json.dumps(
                    {
                        "status": "POSITIVE_EXPECTANCY",
                        "overall": {
                            "trades": 40,
                            "net_jpy": 12000.0,
                            "expectancy_jpy_per_trade": 300.0,
                            "win_rate": 0.7,
                            "payoff_ratio": 1.2,
                        },
                        "by_exit_reason": {},
                        "segment_repair_priorities": {"items": []},
                    }
                )
            )
            bidask.write_text(json.dumps({"contrarian_edge_rules": []}))
            self._write_oanda_firepower_report(
                oanda_rotation,
                status="VERIFIED_MINIMUM_5_ROUTE_ESTIMATED",
            )
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    CREATE TABLE execution_events (
                        ts_utc TEXT,
                        event_type TEXT,
                        trade_id TEXT,
                        order_id TEXT,
                        lane_id TEXT,
                        pair TEXT,
                        side TEXT,
                        realized_pl_jpy REAL,
                        exit_reason TEXT
                    )
                    """
                )
            projection.write_text("")
            hit_rates = {
                "session_expansion_london": {
                    "GBP_USD:TREND": {
                        "hit_rate": 0.98,
                        "samples": 100,
                        "economic_hit_rate": 0.96,
                        "economic_samples": 100,
                    }
                }
            }
            stdout = io.StringIO()

            with mock.patch(
                "quant_rabbit.profitability_acceptance.compute_hit_rates",
                return_value=hit_rates,
            ), redirect_stdout(stdout):
                code = main(
                    [
                        "profitability-acceptance",
                        "--order-intents",
                        str(intents),
                        "--target-state",
                        str(target),
                        "--self-improvement-audit",
                        str(self_audit),
                        "--capture-economics",
                        str(capture),
                        "--execution-ledger-db",
                        str(ledger),
                        "--projection-ledger",
                        str(projection),
                        "--bidask-rules",
                        str(bidask),
                        "--oanda-rotation-mining",
                        str(oanda_rotation),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                    ]
                )
            output_exists = output.exists()
            report_exists = report.exists()

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "PROFITABILITY_ACCEPTANCE_PASSED")
        self.assertEqual(payload["findings"], [])
        self.assertTrue(output_exists)
        self.assertTrue(report_exists)

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

    def test_position_execution_command_stages_position_management_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            management = root / "position_management.json"
            output = root / "position_execution.json"
            report = root / "position_execution.md"
            ledger = root / "execution_ledger.db"
            ledger_report = root / "execution_ledger.md"
            fetched_at = datetime(2026, 6, 12, 13, 45, tzinfo=timezone.utc)
            snapshot.write_text(json.dumps({
                "fetched_at_utc": fetched_at.isoformat(),
                "positions": [
                    {
                        "trade_id": "t-profit",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "units": 1000,
                        "entry_price": 1.1000,
                        "take_profit": 1.1030,
                        "stop_loss": 1.0950,
                        "unrealized_pl_jpy": 600.0,
                        "owner": "trader",
                    }
                ],
                "orders": [],
                "quotes": {
                    "EUR_USD": {
                        "bid": 1.1010,
                        "ask": 1.1011,
                        "timestamp_utc": fetched_at.isoformat(),
                    }
                },
                "home_conversions": {"USD": 160.0, "JPY": 1.0},
            }))
            management.write_text(json.dumps({
                "generated_at_utc": "2026-06-12T13:45:00+00:00",
                "snapshot_fetched_at_utc": fetched_at.isoformat(),
                "action": "TAKE_PROFIT_MARKET",
                "positions": [
                    {
                        "trade_id": "t-profit",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "units": 1000,
                        "action": "TAKE_PROFIT_MARKET",
                        "unrealized_pl_jpy": 600.0,
                        "remaining_risk_jpy": 800.0,
                        "remaining_reward_jpy": 1200.0,
                        "same_direction_score": 100.0,
                        "opposite_direction_score": 80.0,
                        "recommended_stop_loss": None,
                        "recommended_take_profit": None,
                        "reasons": ["temporary top profit-take"],
                        "owner": "trader",
                    }
                ],
            }))
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main([
                    "position-execution",
                    "--snapshot",
                    str(snapshot),
                    "--position-management",
                    str(management),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                    "--execution-ledger-db",
                    str(ledger),
                    "--execution-ledger-report",
                    str(ledger_report),
                ])

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "STAGED")
            self.assertFalse(payload["sent"])
            saved = json.loads(output.read_text())
            self.assertEqual(saved["actions"][0]["request"]["type"], "CLOSE")
            self.assertEqual(saved["execution_ledger"]["status"], "RECORDED")
            with sqlite3.connect(ledger) as conn:
                event = conn.execute(
                    "SELECT event_type, exit_reason, trade_id FROM execution_events"
                ).fetchone()
            self.assertEqual(event, ("GATEWAY_POSITION_ACTION_STAGED", "TAKE_PROFIT_MARKET", "t-profit"))

    def test_position_execution_send_requires_confirm_live_and_live_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            management = root / "position_management.json"
            snapshot.write_text(json.dumps({"positions": [], "orders": [], "quotes": {}}))
            management.write_text(json.dumps({"generated_at_utc": "now", "positions": []}))
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main([
                    "position-execution",
                    "--snapshot",
                    str(snapshot),
                    "--position-management",
                    str(management),
                    "--send",
                ])

            self.assertEqual(code, 2)
            self.assertIn("--confirm-live", json.loads(stdout.getvalue())["error"])

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"QR_LIVE_ENABLED": ""}, clear=False):
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            management = root / "position_management.json"
            snapshot.write_text(json.dumps({"positions": [], "orders": [], "quotes": {}}))
            management.write_text(json.dumps({"generated_at_utc": "now", "positions": []}))
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = main([
                    "position-execution",
                    "--snapshot",
                    str(snapshot),
                    "--position-management",
                    str(management),
                    "--send",
                    "--confirm-live",
                ])

            self.assertEqual(code, 2)
            self.assertIn("QR_LIVE_ENABLED=1", json.loads(stdout.getvalue())["error"])

    def test_position_execution_live_send_closes_through_gateway(self) -> None:
        class CloseClient:
            def __init__(self) -> None:
                self.closed: list[tuple[str, str]] = []

            def close_trade_with_provenance(
                self,
                trade_id: str,
                units: str = "ALL",
                *,
                provenance: str,
            ) -> dict:
                self.closed.append((trade_id, units))
                return {
                    "orderCreateTransaction": {
                        "id": "101",
                        "reason": "TRADE_CLOSE",
                        "tradeClose": {"tradeID": trade_id, "units": units},
                    },
                    "relatedTransactionIDs": ["101"],
                }

            def replace_trade_dependent_orders(self, trade_id: str, order_request: dict) -> dict:
                raise AssertionError("not expected")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"QR_LIVE_ENABLED": "1"}, clear=False):
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            management = root / "position_management.json"
            output = root / "position_execution.json"
            fetched_at = datetime(2026, 6, 12, 13, 45, tzinfo=timezone.utc)
            snapshot.write_text(json.dumps({
                "fetched_at_utc": fetched_at.isoformat(),
                "positions": [
                    {
                        "trade_id": "t-profit",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "units": 1000,
                        "entry_price": 1.1000,
                        "take_profit": 1.1030,
                        "stop_loss": 1.0950,
                        "unrealized_pl_jpy": 600.0,
                        "owner": "trader",
                    }
                ],
                "orders": [],
                "quotes": {
                    "EUR_USD": {
                        "bid": 1.1010,
                        "ask": 1.1011,
                        "timestamp_utc": fetched_at.isoformat(),
                    }
                },
                "home_conversions": {"USD": 160.0, "JPY": 1.0},
            }))
            management.write_text(json.dumps({
                "generated_at_utc": "2026-06-12T13:45:00+00:00",
                "snapshot_fetched_at_utc": fetched_at.isoformat(),
                "action": "TAKE_PROFIT_MARKET",
                "positions": [
                    {
                        "trade_id": "t-profit",
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "units": 1000,
                        "action": "TAKE_PROFIT_MARKET",
                        "unrealized_pl_jpy": 600.0,
                        "remaining_risk_jpy": 800.0,
                        "remaining_reward_jpy": 1200.0,
                        "same_direction_score": 100.0,
                        "opposite_direction_score": 80.0,
                        "recommended_stop_loss": None,
                        "recommended_take_profit": None,
                        "reasons": ["temporary top profit-take"],
                        "owner": "trader",
                    }
                ],
            }))
            client = CloseClient()
            stdout = io.StringIO()

            with mock.patch("quant_rabbit.cli.OandaExecutionClient", return_value=client), redirect_stdout(stdout):
                code = main([
                    "position-execution",
                    "--snapshot",
                    str(snapshot),
                    "--position-management",
                    str(management),
                    "--output",
                    str(output),
                    "--report",
                    str(root / "position_execution.md"),
                    "--execution-ledger-db",
                    str(root / "execution_ledger.db"),
                    "--execution-ledger-report",
                    str(root / "execution_ledger.md"),
                    "--send",
                    "--confirm-live",
                ])

            self.assertEqual(code, 0)
            self.assertEqual(client.closed, [("t-profit", "ALL")])
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "SENT")
            self.assertTrue(payload["sent"])
            saved = json.loads(output.read_text())
            self.assertTrue(saved["sent"])
            self.assertEqual(saved["actions"][0]["response"]["orderCreateTransaction"]["id"], "101")

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

    def test_thesis_evolution_backfills_entry_thesis_before_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            (data / "forecast_history.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp_utc": "2026-06-19T07:50:00Z",
                        "cycle_id": "cycle-before-fill",
                        "pair": "EUR_USD",
                        "direction": "DOWN",
                        "confidence": 0.61,
                        "target_price": 1.1441,
                        "invalidation_price": 1.1518,
                        "horizon_min": 60,
                    }
                )
                + "\n"
            )
            snapshot = data / "broker_snapshot.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "fetched_at_utc": "2026-06-19T08:10:00Z",
                        "positions": [
                            {
                                "trade_id": "472732",
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "owner": "trader",
                                "units": 6300,
                                "entry_price": 1.14486,
                                "take_profit": 1.14414,
                                "stop_loss": 1.15171,
                                "unrealized_pl_jpy": -10.0,
                            }
                        ],
                        "orders": [],
                        "quotes": {
                            "EUR_USD": {
                                "bid": 1.1447,
                                "ask": 1.1448,
                                "timestamp_utc": "2026-06-19T08:10:00Z",
                            }
                        },
                    }
                )
            )
            pair_charts = data / "pair_charts.json"
            pair_charts.write_text(
                json.dumps(
                    {
                        "charts": [
                            {
                                "pair": "EUR_USD",
                                "confluence": {"dominant_regime": "TREND_DOWN"},
                                "views": [],
                            }
                        ]
                    }
                )
            )
            fill = {
                "id": "472732",
                "type": "ORDER_FILL",
                "time": "2026-06-19T08:01:32.903433014Z",
                "orderID": "472730",
                "instrument": "EUR_USD",
                "units": "-6300",
                "price": "1.14486",
                "reason": "LIMIT_ORDER",
                "clientOrderID": "qrv1-EURUSD-S-81b9490de070",
                "tradeOpened": {
                    "tradeID": "472732",
                    "units": "-6300",
                    "price": "1.14486",
                    "clientExtensions": {
                        "id": "qrv1-EURUSD-S-d0dda4b89776",
                        "tag": "trader",
                        "comment": (
                            "qr-vnext lane=failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT "
                            "desk=failure_trader"
                        ),
                    },
                },
            }
            tp = {
                "id": "472733",
                "type": "TAKE_PROFIT_ORDER",
                "batchID": "472732",
                "time": "2026-06-19T08:01:32.903433014Z",
                "tradeID": "472732",
                "price": "1.14414",
            }
            sl = {
                "id": "472734",
                "type": "STOP_LOSS_ORDER",
                "batchID": "472732",
                "time": "2026-06-19T08:01:32.903433014Z",
                "tradeID": "472732",
                "price": "1.15171",
            }
            with sqlite3.connect(data / "execution_ledger.db") as conn:
                conn.executescript(
                    """
                    CREATE TABLE oanda_transactions (
                        transaction_id TEXT PRIMARY KEY,
                        type TEXT NOT NULL,
                        time_utc TEXT,
                        batch_id TEXT,
                        request_id TEXT,
                        raw_json TEXT NOT NULL,
                        inserted_at_utc TEXT NOT NULL
                    );
                    """
                )
                for payload in (fill, tp, sl):
                    conn.execute(
                        """
                        INSERT INTO oanda_transactions(
                            transaction_id, type, time_utc, batch_id, request_id, raw_json, inserted_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            payload["id"],
                            payload["type"],
                            payload.get("time"),
                            payload.get("batchID"),
                            None,
                            json.dumps(payload),
                            "2026-06-19T08:02:00Z",
                        ),
                    )

            old_cwd = Path.cwd()
            stdout = io.StringIO()
            try:
                os.chdir(root)
                with mock.patch(
                    "quant_rabbit.cli._refresh_current_forecast_history",
                    return_value={"recorded": 0},
                ):
                    with redirect_stdout(stdout):
                        code = main(
                            [
                                "thesis-evolution-check",
                                "--snapshot",
                                str(snapshot),
                                "--pair-charts",
                                str(pair_charts),
                                "--output",
                                str(data / "thesis_evolution_report.json"),
                            ]
                        )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            report = json.loads((data / "thesis_evolution_report.json").read_text())

            self.assertEqual(payload["entry_thesis_backfill"]["status"], "BACKFILLED")
            self.assertEqual(payload["by_status"]["UNVERIFIABLE"], 0)
            self.assertEqual(report["entry_thesis_coverage"]["missing"], 0)
            self.assertFalse(report["entry_thesis_coverage"]["blocking"])
            self.assertIn("472732", (data / "entry_thesis_ledger.jsonl").read_text())

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

            class CloseClient:
                def __init__(self) -> None:
                    self.close_calls: list[tuple[str, str, str]] = []

                def close_trade_with_provenance(
                    self,
                    trade_id: str,
                    units: str = "ALL",
                    *,
                    provenance: str,
                ) -> dict:
                    self.close_calls.append((trade_id, units, provenance))
                    return {"ok": True}

            client = CloseClient()

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
            self.assertEqual(client.close_calls, [("t-adverse", "5000", "adverse_partial_close")])
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

    def test_adverse_partial_close_send_syncs_broker_close_outcome(self) -> None:
        class SyncingCloseClient:
            def __init__(self) -> None:
                self.close_calls: list[tuple[str, str]] = []
                self.sync_calls: list[str] = []

            def account_summary(self, *, now_utc=None) -> AccountSummary:
                return AccountSummary(nav_jpy=100000.0, balance_jpy=100000.0, last_transaction_id="100")

            def close_trade_with_provenance(
                self,
                trade_id: str,
                units: str = "ALL",
                *,
                provenance: str,
            ) -> dict:
                self.close_calls.append((trade_id, units))
                return {
                    "orderCreateTransaction": {
                        "id": "101",
                        "reason": "TRADE_CLOSE",
                        "tradeClose": {"tradeID": trade_id, "units": units},
                    },
                    "orderFillTransaction": {
                        "id": "102",
                        "type": "ORDER_FILL",
                        "orderID": "101",
                        "instrument": "EUR_USD",
                        "units": "-5000",
                        "reason": "MARKET_ORDER_TRADE_CLOSE",
                        "tradesClosed": [
                            {
                                "tradeID": trade_id,
                                "units": "5000",
                                "price": "1.19500",
                                "realizedPL": "-500.0",
                            }
                        ],
                    },
                    "relatedTransactionIDs": ["101", "102"],
                }

            def transactions_since_id(self, transaction_id: str) -> dict:
                self.sync_calls.append(transaction_id)
                if transaction_id != "100":
                    return {"transactions": [], "lastTransactionID": transaction_id}
                return {
                    "lastTransactionID": "102",
                    "transactions": [
                        {
                            "id": "101",
                            "type": "MARKET_ORDER",
                            "time": "2026-06-08T00:00:01Z",
                            "reason": "TRADE_CLOSE",
                            "tradeClose": {"tradeID": "t-adverse", "units": "5000"},
                        },
                        {
                            "id": "102",
                            "type": "ORDER_FILL",
                            "time": "2026-06-08T00:00:02Z",
                            "orderID": "101",
                            "instrument": "EUR_USD",
                            "units": "-5000",
                            "reason": "MARKET_ORDER_TRADE_CLOSE",
                            "tradesClosed": [
                                {
                                    "tradeID": "t-adverse",
                                    "units": "5000",
                                    "price": "1.19500",
                                    "realizedPL": "-500.0",
                                }
                            ],
                        },
                    ],
                }

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {
            "QR_DISABLE_ADVERSE_PARTIAL_CLOSE": "",
            "QR_LIVE_ENABLED": "1",
        }, clear=False):
            root = Path(tmp)
            snapshot, pair_charts = self._adverse_partial_close_files(root)
            stdout = io.StringIO()
            client = SyncingCloseClient()

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

            with sqlite3.connect(root / "execution_ledger.db") as conn:
                rows = conn.execute(
                    """
                    SELECT event_type, exit_reason, trade_id, order_id, realized_pl_jpy
                    FROM execution_events
                    WHERE event_type IN ('GATEWAY_TRADE_CLOSE_SENT', 'ORDER_ACCEPTED', 'TRADE_CLOSED')
                    ORDER BY event_type
                    """
                ).fetchall()

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["execution_ledger_pre_sync"]["status"], "BASELINED")
        self.assertEqual(payload["execution_ledger_post_sync"]["status"], "SYNCED")
        self.assertEqual(client.close_calls, [("t-adverse", "5000")])
        self.assertEqual(client.sync_calls, ["100"])
        self.assertIn(("GATEWAY_TRADE_CLOSE_SENT", "ADVERSE_PARTIAL_CLOSE", "t-adverse", "101", None), rows)
        self.assertIn(("ORDER_ACCEPTED", "TRADE_CLOSE", "t-adverse", "101", None), rows)
        self.assertIn(("TRADE_CLOSED", "MARKET_ORDER_TRADE_CLOSE", "t-adverse", "101", -500.0), rows)

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

    def test_generate_intents_can_reuse_market_artifacts_before_replacing_intents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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

            with mock.patch(
                "quant_rabbit.cli._auto_refresh_market_evidence_if_required",
                return_value={"status": "SKIPPED", "reason": "reuse_market_artifacts"},
            ) as market_refresh, mock.patch("quant_rabbit.cli.IntentGenerator") as generator_cls, redirect_stdout(stdout):
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
                        "--market-context-matrix",
                        str(root / "data" / "market_context_matrix.json"),
                        "--reuse-market-artifacts",
                        "--no-refresh-market-story",
                    ]
                )

        self.assertEqual(code, 0)
        market_refresh.assert_called_once_with(
            label="generate-intents",
            reuse_market_artifacts=True,
            market_context_matrix_path=root / "data" / "market_context_matrix.json",
            validate_order_intents_freshness=False,
        )
        generator_cls.return_value.run.assert_called_once_with(
            snapshot_path=root / "data" / "broker_snapshot.json",
            max_candidates=56,
        )

    def test_generate_intents_refreshes_memory_health_after_direct_intent_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            docs_root = root / "docs"
            data_root.mkdir()
            docs_root.mkdir()
            snapshot = data_root / "broker_snapshot.json"
            intents = data_root / "order_intents.json"
            memory = data_root / "memory_health.json"
            memory_report = docs_root / "memory_health_report.md"
            snapshot.write_text(
                json.dumps(
                    {
                        "fetched_at_utc": "2026-06-16T09:46:24+00:00",
                        "account": {"fetched_at_utc": "2026-06-16T09:46:24+00:00"},
                        "quotes": {"EUR_USD": {"bid": 1.1, "ask": 1.1001}},
                    }
                )
            )
            summary = SimpleNamespace(
                output_path=intents,
                report_path=docs_root / "order_intents_report.md",
                candidates_seen=1,
                generated=1,
                needs_snapshot=False,
                dry_run_passed=1,
                live_ready=1,
            )

            def generator_run(**_: object) -> SimpleNamespace:
                intents.write_text(
                    json.dumps({"generated_at_utc": "2026-06-16T09:46:59+00:00", "results": []})
                )
                return summary

            health_summary = SimpleNamespace(
                status="MEMORY_HEALTH_PASS",
                output_path=memory,
                report_path=memory_report,
                blockers=0,
                warnings=0,
            )
            stdout = io.StringIO()

            with mock.patch.dict(os.environ, {"QR_LIVE_ENABLED": "1"}, clear=False), mock.patch(
                "quant_rabbit.cli.DEFAULT_MEMORY_HEALTH",
                memory,
            ), mock.patch(
                "quant_rabbit.cli.DEFAULT_MEMORY_HEALTH_REPORT",
                memory_report,
            ), mock.patch(
                "quant_rabbit.cli._auto_refresh_market_evidence_if_required",
                return_value={"status": "SKIPPED", "reason": "test"},
            ), mock.patch(
                "quant_rabbit.cli._refresh_snapshot_after_market_evidence_if_required",
                return_value=None,
            ), mock.patch(
                "quant_rabbit.cli._pre_entry_execution_ledger_sync_if_required",
                return_value=None,
            ), mock.patch(
                "quant_rabbit.cli._pre_entry_projection_verification_if_required",
                return_value=None,
            ), mock.patch(
                "quant_rabbit.memory_health.MemoryHealthAuditor"
            ) as memory_cls, mock.patch("quant_rabbit.cli.IntentGenerator") as generator_cls, redirect_stdout(stdout):
                generator_cls.return_value.run.side_effect = generator_run
                memory_cls.return_value.run.return_value = health_summary
                code = main(
                    [
                        "generate-intents",
                        "--campaign-plan",
                        str(data_root / "daily_campaign_plan.json"),
                        "--strategy-profile",
                        str(data_root / "strategy_profile.json"),
                        "--snapshot",
                        str(snapshot),
                        "--output",
                        str(intents),
                        "--report",
                        str(summary.report_path),
                        "--no-refresh-market-story",
                    ]
                )

        self.assertEqual(code, 0)
        memory_cls.assert_called_once_with(output_path=memory, report_path=memory_report)
        memory_cls.return_value.run.assert_called_once_with(
            snapshot_path=snapshot,
            target_state_path=mock.ANY,
            order_intents_path=intents,
            strategy_profile_path=mock.ANY,
            forecast_history_path=mock.ANY,
            projection_ledger_path=mock.ANY,
            learning_audit_path=mock.ANY,
            entry_thesis_ledger_path=mock.ANY,
            execution_ledger_db_path=mock.ANY,
        )
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["memory_health_refresh"]["status"], "MEMORY_HEALTH_PASS")
        self.assertEqual(payload["memory_health_refresh"]["blockers"], 0)
        self.assertEqual(payload["memory_health_refresh"]["warnings"], 0)

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

    def test_generate_intents_refreshes_campaign_plan_after_preflight_updates_target_state(self) -> None:
        calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            docs_root = root / "docs"
            old_ts = "2026-06-08T15:00:00+00:00"
            new_ts = "2026-06-08T15:05:00+00:00"
            campaign = data_root / "daily_campaign_plan.json"
            target = data_root / "daily_target_state.json"
            strategy = data_root / "strategy_profile.json"
            story = data_root / "market_story_profile.json"
            snapshot = data_root / "broker_snapshot.json"
            campaign.write_text(
                json.dumps(
                    {
                        "generated_at_utc": old_ts,
                        "start_balance_jpy": 10000.0,
                        "target_jpy": 1000.0,
                        "target_return_pct": 10.0,
                        "lanes": [],
                    }
                )
            )
            strategy.write_text(json.dumps({"generated_at_utc": old_ts, "pairs": []}))
            story.write_text(json.dumps({"generated_at_utc": old_ts, "pair_profiles": []}))
            snapshot.write_text(json.dumps({"fetched_at_utc": new_ts, "quotes": {}}))

            def projection_preflight(**_: object) -> dict[str, str]:
                calls.append("target_update")
                target.write_text(
                    json.dumps(
                        {
                            "generated_at_utc": new_ts,
                            "status": "PURSUE_TARGET",
                            "start_balance_jpy": 10000.0,
                            "target_jpy": 1000.0,
                            "target_return_pct": 10.0,
                            "daily_risk_budget_jpy": 1000.0,
                            "per_trade_risk_budget_jpy": 100.0,
                        }
                    )
                )
                return {"status": "OK"}

            def campaign_run(**_: object) -> SimpleNamespace:
                calls.append("campaign")
                return SimpleNamespace(
                    report_path=campaign.with_suffix(".md"),
                    plan_path=campaign,
                    target_jpy=1000.0,
                    lanes=1,
                    actionable_lanes=1,
                    rejected_lanes=0,
                )

            def generator_run(**_: object) -> SimpleNamespace:
                calls.append("generator")
                return SimpleNamespace(
                    output_path=data_root / "order_intents.json",
                    report_path=docs_root / "order_intents_report.md",
                    candidates_seen=1,
                    generated=1,
                    needs_snapshot=False,
                    dry_run_passed=1,
                    live_ready=1,
                )

            stdout = io.StringIO()
            with mock.patch(
                "quant_rabbit.cli._auto_refresh_market_evidence_if_required",
                return_value={"status": "SKIPPED", "reason": "test"},
            ), mock.patch(
                "quant_rabbit.cli._refresh_snapshot_after_market_evidence_if_required",
                return_value=None,
            ), mock.patch(
                "quant_rabbit.cli._pre_entry_execution_ledger_sync_if_required",
                return_value=None,
            ), mock.patch(
                "quant_rabbit.cli._pre_entry_projection_verification_if_required",
                side_effect=projection_preflight,
            ), mock.patch("quant_rabbit.cli.CampaignPlanner") as planner_cls, mock.patch(
                "quant_rabbit.cli.IntentGenerator"
            ) as generator_cls, redirect_stdout(stdout):
                planner_cls.return_value.run.side_effect = campaign_run
                generator_cls.return_value.run.side_effect = generator_run
                code = main(
                    [
                        "generate-intents",
                        "--campaign-plan",
                        str(campaign),
                        "--strategy-profile",
                        str(strategy),
                        "--snapshot",
                        str(snapshot),
                        "--output",
                        str(data_root / "order_intents.json"),
                        "--report",
                        str(docs_root / "order_intents_report.md"),
                        "--market-story-profile",
                        str(story),
                        "--no-refresh-market-story",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertEqual(calls, ["target_update", "campaign", "generator"])
        planner_cls.assert_called_once_with(
            strategy_profile=strategy,
            market_story_profile=story,
            report_path=campaign.with_suffix(".md"),
            plan_path=campaign,
            oanda_rotation_mining=None,
        )
        planner_cls.return_value.run.assert_called_once_with(start_balance_jpy=10000.0, target_return_pct=10.0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["campaign_refresh"]["status"], "REFRESHED")
        self.assertIn("daily_target_state_newer", payload["campaign_refresh"]["refresh_reasons"])
        self.assertEqual(payload["live_ready"], 1)

    def test_generate_intents_refreshes_default_campaign_when_oanda_firepower_is_newer(self) -> None:
        calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            docs_root = root / "docs"
            docs_root.mkdir()
            old_ts = "2026-06-21T18:25:00+00:00"
            new_ts = "2026-06-21T19:18:00+00:00"
            campaign = data_root / "daily_campaign_plan.json"
            target = data_root / "daily_target_state.json"
            strategy = data_root / "strategy_profile.json"
            story = data_root / "market_story_profile.json"
            snapshot = data_root / "broker_snapshot.json"
            oanda = root / "oanda_universal_rotation_mining_latest.json"
            campaign.write_text(
                json.dumps(
                    {
                        "generated_at_utc": old_ts,
                        "start_balance_jpy": 10000.0,
                        "target_jpy": 1000.0,
                        "target_return_pct": 10.0,
                        "lanes": [
                            {
                                "desk": "range_trader",
                                "pair": "EUR_USD",
                                "direction": "SHORT",
                                "method": "RANGE_ROTATION",
                            }
                        ],
                    }
                )
            )
            target.write_text(
                json.dumps(
                    {
                        "generated_at_utc": old_ts,
                        "status": "PURSUE_TARGET",
                        "start_balance_jpy": 10000.0,
                        "target_jpy": 1000.0,
                        "target_return_pct": 10.0,
                        "daily_risk_budget_jpy": 1000.0,
                        "per_trade_risk_budget_jpy": 100.0,
                    }
                )
            )
            strategy.write_text(json.dumps({"generated_at_utc": old_ts, "pairs": []}))
            story.write_text(json.dumps({"generated_at_utc": old_ts, "pair_profiles": []}))
            snapshot.write_text(json.dumps({"fetched_at_utc": old_ts, "quotes": {}}))
            oanda.write_text(
                json.dumps(
                    {
                        "generated_at_utc": new_ts,
                        "campaign_firepower": {
                            "status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
                            "high_precision": {
                                "unique_vehicle_count": 1,
                                "top_vehicles": [
                                    {
                                        "vehicle_key": "GBP_USD|SHORT|range_reversion|tp1_sl1",
                                        "pair": "GBP_USD",
                                        "shape": "range_reversion",
                                        "firepower_side": "SHORT",
                                        "exit_shape": "tp1_sl1",
                                    }
                                ],
                            },
                        },
                    }
                )
            )
            summary = SimpleNamespace(
                output_path=data_root / "order_intents.json",
                report_path=docs_root / "order_intents_report.md",
                candidates_seen=1,
                generated=1,
                needs_snapshot=False,
                dry_run_passed=1,
                live_ready=0,
            )

            def campaign_run(**_: object) -> SimpleNamespace:
                calls.append("campaign")
                return SimpleNamespace(
                    report_path=docs_root / "daily_campaign_report.md",
                    plan_path=campaign,
                    target_jpy=1000.0,
                    lanes=2,
                    actionable_lanes=2,
                    rejected_lanes=0,
                )

            def generator_run(**_: object) -> SimpleNamespace:
                calls.append("generator")
                return summary

            stdout = io.StringIO()
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "quant_rabbit.cli.DEFAULT_CAMPAIGN_PLAN", campaign
            ), mock.patch("quant_rabbit.cli.DEFAULT_CAMPAIGN_REPORT", docs_root / "daily_campaign_report.md"), mock.patch(
                "quant_rabbit.cli.DEFAULT_OANDA_UNIVERSAL_ROTATION_MINING", oanda
            ), mock.patch(
                "quant_rabbit.cli._auto_refresh_market_evidence_if_required",
                return_value={"status": "SKIPPED", "reason": "test"},
            ), mock.patch(
                "quant_rabbit.cli._refresh_snapshot_after_market_evidence_if_required",
                return_value=None,
            ), mock.patch(
                "quant_rabbit.cli._pre_entry_execution_ledger_sync_if_required",
                return_value=None,
            ), mock.patch(
                "quant_rabbit.cli._pre_entry_projection_verification_if_required",
                return_value=None,
            ), mock.patch("quant_rabbit.cli.CampaignPlanner") as planner_cls, mock.patch(
                "quant_rabbit.cli.IntentGenerator"
            ) as generator_cls, redirect_stdout(stdout):
                planner_cls.return_value.run.side_effect = campaign_run
                generator_cls.return_value.run.side_effect = generator_run
                code = main(
                    [
                        "generate-intents",
                        "--campaign-plan",
                        str(campaign),
                        "--strategy-profile",
                        str(strategy),
                        "--snapshot",
                        str(snapshot),
                        "--output",
                        str(summary.output_path),
                        "--report",
                        str(summary.report_path),
                        "--market-story-profile",
                        str(story),
                        "--no-refresh-market-story",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertEqual(calls, ["campaign", "generator"])
        planner_cls.assert_called_once_with(
            strategy_profile=strategy,
            market_story_profile=story,
            report_path=docs_root / "daily_campaign_report.md",
            plan_path=campaign,
            oanda_rotation_mining=oanda,
        )
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["campaign_refresh"]["status"], "REFRESHED")
        self.assertIn("oanda_rotation_mining_newer", payload["campaign_refresh"]["refresh_reasons"])
        self.assertIn("oanda_rotation_mining_seed_missing", payload["campaign_refresh"]["refresh_reasons"])
        self.assertEqual(payload["campaign_refresh"]["oanda_rotation_mining_path"], str(oanda))

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
            call_order: list[str] = []

            def capture_preflight(**_: object) -> dict[str, object]:
                call_order.append("capture")
                return {
                    "status": "NEGATIVE_EXPECTANCY",
                    "trades": 215,
                    "expectancy_jpy": -168.9,
                }

            def generator_run(*_: object, **__: object) -> SimpleNamespace:
                call_order.append("generator")
                return summary

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
                "quant_rabbit.cli._pre_entry_capture_economics_refresh_if_required",
                side_effect=capture_preflight,
            ), mock.patch(
                "quant_rabbit.cli.IntentGenerator"
            ) as generator_cls, redirect_stdout(stdout):
                ledger_cls.return_value.sync_oanda_transactions.return_value = ledger_summary
                generator_cls.return_value.run.side_effect = generator_run
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
        self.assertEqual(payload["capture_economics_refresh"]["trades"], 215)
        self.assertEqual(call_order, ["capture", "generator"])

    def test_pre_entry_capture_economics_refresh_uses_current_ledger_summary(self) -> None:
        from quant_rabbit.cli import _pre_entry_capture_economics_refresh_if_required

        summary = SimpleNamespace(
            status="NEGATIVE_EXPECTANCY",
            output_path=Path("/tmp/capture.json"),
            report_path=Path("/tmp/capture.md"),
            trades=215,
            win_rate=0.6047,
            payoff_ratio=0.392,
            expectancy_jpy=-168.9,
        )
        with mock.patch("quant_rabbit.cli._running_under_test_harness", return_value=False), mock.patch(
            "quant_rabbit.capture_economics.build_capture_economics",
            return_value=summary,
        ) as build:
            result = _pre_entry_capture_economics_refresh_if_required(
                telemetry_required=True,
                execution_ledger_sync={"status": "SYNCED"},
            )

        build.assert_called_once_with(
            ledger_path=DEFAULT_EXECUTION_LEDGER_DB,
            output_path=DEFAULT_CAPTURE_ECONOMICS,
            report_path=mock.ANY,
        )
        self.assertEqual(result["status"], "NEGATIVE_EXPECTANCY")
        self.assertEqual(result["trades"], 215)
        self.assertEqual(result["expectancy_jpy"], -168.9)
        self.assertEqual(result["execution_ledger_sync_status"], "SYNCED")

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

    def test_generate_intents_refreshes_daily_target_after_snapshot_refresh_before_campaign(self) -> None:
        calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            old_ts = "2026-06-02T01:00:00+00:00"
            fresh_at = datetime(2026, 6, 2, 1, 5, tzinfo=timezone.utc)
            snapshot = data_root / "broker_snapshot.json"
            campaign = data_root / "daily_campaign_plan.json"
            target = data_root / "daily_target_state.json"
            strategy = data_root / "strategy_profile.json"
            story = data_root / "market_story_profile.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "fetched_at_utc": old_ts,
                        "quotes": {"EUR_USD": {"bid": 1.1, "ask": 1.1001}},
                    }
                )
            )
            campaign.write_text(
                json.dumps(
                    {
                        "generated_at_utc": old_ts,
                        "start_balance_jpy": 10000.0,
                        "target_jpy": 1000.0,
                        "target_return_pct": 10.0,
                        "lanes": [],
                    }
                )
            )
            target.write_text(
                json.dumps(
                    {
                        "generated_at_utc": old_ts,
                        "status": "PURSUE_TARGET",
                        "start_balance_jpy": 10000.0,
                        "target_jpy": 1000.0,
                        "target_return_pct": 10.0,
                    }
                )
            )
            strategy.write_text(json.dumps({"generated_at_utc": old_ts, "pairs": []}))
            story.write_text(json.dumps({"generated_at_utc": old_ts, "pair_profiles": []}))
            summary = SimpleNamespace(
                output_path=data_root / "order_intents.json",
                report_path=root / "docs" / "order_intents_report.md",
                candidates_seen=1,
                generated=1,
                needs_snapshot=False,
                dry_run_passed=1,
                live_ready=1,
            )
            fresh_snapshot = BrokerSnapshot(
                fetched_at_utc=fresh_at,
                quotes={"EUR_USD": Quote("EUR_USD", 1.1002, 1.1004, fresh_at)},
            )

            def target_run(**_: object) -> SimpleNamespace:
                calls.append("target")
                payload = json.loads(target.read_text())
                payload["generated_at_utc"] = fresh_at.isoformat()
                target.write_text(json.dumps(payload))
                return SimpleNamespace(
                    state_path=target,
                    report_path=target.with_suffix(".md"),
                    status="PURSUE_TARGET",
                    remaining_target_jpy=1000.0,
                    remaining_risk_budget_jpy=900.0,
                    per_trade_risk_budget_jpy=100.0,
                )

            def campaign_run(**_: object) -> SimpleNamespace:
                calls.append("campaign")
                return SimpleNamespace(
                    report_path=campaign.with_suffix(".md"),
                    plan_path=campaign,
                    target_jpy=1000.0,
                    lanes=1,
                    actionable_lanes=1,
                    rejected_lanes=0,
                )

            def generator_run(**_: object) -> SimpleNamespace:
                calls.append("generator")
                return summary

            stdout = io.StringIO()
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "quant_rabbit.cli._running_under_test_harness", return_value=False
            ), mock.patch(
                "quant_rabbit.cli._auto_refresh_market_evidence_if_required",
                return_value={"status": "REFRESHED", "pairs": 28},
            ), mock.patch("quant_rabbit.cli.OandaReadOnlyClient") as client_cls, mock.patch(
                "quant_rabbit.cli._pre_entry_execution_ledger_sync_if_required",
                return_value=None,
            ), mock.patch(
                "quant_rabbit.cli._pre_entry_projection_verification_if_required",
                return_value=None,
            ), mock.patch("quant_rabbit.cli.DailyTargetLedger") as ledger_cls, mock.patch(
                "quant_rabbit.cli.CampaignPlanner"
            ) as planner_cls, mock.patch("quant_rabbit.cli.IntentGenerator") as generator_cls, redirect_stdout(stdout):
                client_cls.return_value.snapshot.return_value = fresh_snapshot
                ledger_cls.return_value.run.side_effect = target_run
                planner_cls.return_value.run.side_effect = campaign_run
                generator_cls.return_value.run.side_effect = generator_run
                code = main(
                    [
                        "generate-intents",
                        "--campaign-plan",
                        str(campaign),
                        "--strategy-profile",
                        str(strategy),
                        "--snapshot",
                        str(snapshot),
                        "--output",
                        str(summary.output_path),
                        "--report",
                        str(summary.report_path),
                        "--market-story-profile",
                        str(story),
                        "--no-refresh-market-story",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertEqual(calls, ["target", "campaign", "generator"])
        ledger_cls.assert_called_once_with(
            state_path=target,
            report_path=target.with_suffix(".md"),
            pace_backtest_path=mock.ANY,
        )
        planner_cls.return_value.run.assert_called_once_with(start_balance_jpy=10000.0, target_return_pct=10.0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["snapshot_refresh"]["status"], "REFRESHED")
        self.assertEqual(payload["daily_target_refresh"]["status"], "REFRESHED")
        self.assertEqual(payload["campaign_refresh"]["status"], "REFRESHED")
        self.assertIn("daily_target_state_newer", payload["campaign_refresh"]["refresh_reasons"])

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

    def test_reuse_market_artifacts_can_skip_existing_intent_freshness_for_generation(self) -> None:
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

            result = _auto_refresh_market_evidence_if_required(
                label="generate-intents",
                reuse_market_artifacts=True,
                validate_order_intents_freshness=False,
                market_context_matrix_path=matrix,
                context_asset_charts_path=context_assets,
                broker_instruments_path=broker_instruments,
                order_intents_path=order_intents,
            )

        self.assertEqual(result, {"status": "SKIPPED", "reason": "reuse_market_artifacts"})

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
        "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE",
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
                    # cycle-refresh / cycle-sidecars (2026-06-10) run the
                    # SKILL_trader.md step lists in one process; nested
                    # generate-intents / daily-target-state / position
                    # sidecar calls must see identical SL-free defaults.
                    "cycle-refresh",
                    "cycle-sidecars",
                    # memory-health audits live routing memory before
                    # entry/verify routing and must use the same SL-free
                    # defaults when classifying broker snapshot state.
                    "memory-health",
                    # self-improvement-audit is consumed by the verifier and
                    # gateway as the live-facing repair gate.
                    "self-improvement-audit",
                    # trader-support-bot reads the same live support state and
                    # must classify guardian/profit-capture under live defaults.
                    "trader-support-bot",
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
                                "quant_rabbit.projection_truth.fetch_candles_via_client",
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
                                "quant_rabbit.projection_truth.fetch_candles_via_client",
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


class ConsolidatedCycleCommandTest(unittest.TestCase):
    """Mechanics of the cycle-refresh/cycle-sidecars in-process step runner.

    Network-dependent steps are not exercised here; behavior parity with the
    SKILL_trader.md skeleton is asserted at the step-list level.
    """

    def test_run_cycle_steps_isolates_failures_and_aborts_on_required(self) -> None:
        from quant_rabbit.cli import _run_cycle_steps

        steps = [
            # argparse rejects the unknown command with SystemExit(2).
            {"argv": ["definitely-not-a-command"], "required": True},
            {"argv": ["broker-snapshot"], "required": False},
        ]
        results, aborted = _run_cycle_steps(steps)

        self.assertTrue(aborted)
        self.assertEqual(results[0]["status"], "FAILED_REQUIRED")
        self.assertEqual(results[0]["rc"], 2)
        self.assertIn("tail", results[0])
        self.assertEqual(results[1]["status"], "SKIPPED_AFTER_ABORT")

    def test_run_cycle_steps_continues_after_optional_failure(self) -> None:
        from quant_rabbit.cli import _run_cycle_steps

        steps = [
            {"argv": ["definitely-not-a-command"], "required": False},
            {"argv": ["another-bad-command"], "required": False},
        ]
        results, aborted = _run_cycle_steps(steps)

        self.assertFalse(aborted)
        self.assertEqual([r["status"] for r in results], ["FAILED", "FAILED"])

    def test_run_cycle_steps_accepts_declared_detection_exit_codes(self) -> None:
        from quant_rabbit.cli import _run_cycle_steps

        steps = [
            {"argv": ["definitely-not-a-command"], "required": False, "ok_rcs": [0, 2]},
            {"argv": ["another-bad-command"], "required": False},
        ]
        results, aborted = _run_cycle_steps(steps)

        self.assertFalse(aborted)
        self.assertEqual(results[0]["status"], "OK")
        self.assertEqual(results[0]["rc"], 2)
        self.assertNotIn("tail", results[0])
        self.assertEqual(results[1]["status"], "FAILED")

    def test_run_cycle_steps_records_optional_step_timeout(self) -> None:
        from quant_rabbit.cli import _CycleStepTimeout, _run_cycle_steps

        steps = [
            {"argv": ["broker-snapshot"], "required": False},
            {"argv": ["another-bad-command"], "required": False},
        ]

        with mock.patch(
            "quant_rabbit.cli._run_with_cycle_step_timeout",
            side_effect=[_CycleStepTimeout("cycle step exceeded 0.1s timeout"), 0],
        ):
            results, aborted = _run_cycle_steps(steps)

        self.assertFalse(aborted)
        self.assertEqual(results[0]["status"], "TIMED_OUT")
        self.assertEqual(results[0]["rc"], 124)
        self.assertIn("_CycleStepTimeout", results[0]["tail"])
        self.assertEqual(results[1]["status"], "OK")

    def test_run_cycle_steps_aborts_after_required_step_timeout(self) -> None:
        from quant_rabbit.cli import _CycleStepTimeout, _run_cycle_steps

        steps = [
            {"argv": ["broker-snapshot"], "required": True},
            {"argv": ["another-bad-command"], "required": False},
        ]

        with mock.patch(
            "quant_rabbit.cli._run_with_cycle_step_timeout",
            side_effect=_CycleStepTimeout("cycle step exceeded 0.1s timeout"),
        ):
            results, aborted = _run_cycle_steps(steps)

        self.assertTrue(aborted)
        self.assertEqual(results[0]["status"], "TIMED_OUT_REQUIRED")
        self.assertEqual(results[0]["rc"], 124)
        self.assertIn("_CycleStepTimeout", results[0]["tail"])
        self.assertEqual(results[1]["status"], "SKIPPED_AFTER_ABORT")

    def test_cycle_step_timeout_seconds_respects_env_and_overrides(self) -> None:
        from quant_rabbit.cli import DEFAULT_CYCLE_STEP_TIMEOUT_SECONDS, _cycle_step_timeout_seconds

        with mock.patch.dict(os.environ, {"QR_CYCLE_STEP_TIMEOUT_SECONDS": "0"}, clear=False):
            self.assertIsNone(_cycle_step_timeout_seconds({"argv": ["broker-snapshot"]}))
            self.assertEqual(
                _cycle_step_timeout_seconds({"argv": ["broker-snapshot"], "timeout_seconds": "7.5"}),
                7.5,
            )

        with mock.patch.dict(os.environ, {"QR_CYCLE_STEP_TIMEOUT_SECONDS": "not-a-number"}, clear=False):
            self.assertEqual(
                _cycle_step_timeout_seconds({"argv": ["broker-snapshot"]}),
                DEFAULT_CYCLE_STEP_TIMEOUT_SECONDS,
            )

    def test_cycle_refresh_refuses_existing_live_runtime_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_dir = Path(tmp) / "lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(f"{os.getpid()}\n")
            digest = Path(tmp) / "digest.json"
            stderr = io.StringIO()

            with mock.patch.dict(os.environ, {"QR_AUTOTRADE_LOCK_DIR": str(lock_dir)}, clear=False):
                with redirect_stderr(stderr):
                    rc = main(["cycle-refresh", "--digest-output", str(digest)])

            self.assertEqual(rc, 75)
            self.assertFalse(digest.exists())
            self.assertIn("refusing cycle-refresh overlap", stderr.getvalue())

    def test_cycle_runtime_lock_restores_env_and_removes_lock(self) -> None:
        from quant_rabbit.cli import _acquire_cycle_runtime_lock, _release_cycle_runtime_lock

        with tempfile.TemporaryDirectory() as tmp:
            lock_dir = Path(tmp) / "lock"
            with mock.patch.dict(os.environ, {"QR_AUTOTRADE_LOCK_DIR": str(lock_dir)}, clear=False):
                os.environ.pop("QR_AUTOTRADE_LOCK_HELD", None)
                token = _acquire_cycle_runtime_lock("cycle-refresh")
                self.assertIsNotNone(token)
                self.assertTrue(lock_dir.exists())
                self.assertEqual(os.environ.get("QR_AUTOTRADE_LOCK_HELD"), "1")

                _release_cycle_runtime_lock(token)

                self.assertFalse(lock_dir.exists())
                self.assertNotIn("QR_AUTOTRADE_LOCK_HELD", os.environ)

    def test_cycle_refresh_steps_mirror_skill_skeleton(self) -> None:
        from quant_rabbit.cli import _cycle_refresh_steps, _cycle_sidecar_steps

        refresh = [" ".join(s["argv"]) for s in _cycle_refresh_steps("10")]
        # Order-sensitive anchors from docs/SKILL_trader.md section 2.
        self.assertEqual(refresh[0], "broker-snapshot --output data/broker_snapshot.json")
        intent_step = "generate-intents --snapshot data/broker_snapshot.json --reuse-market-artifacts"
        self.assertIn(intent_step, refresh)
        self.assertLess(refresh.index("verify-projections"), refresh.index(intent_step))
        self.assertLess(refresh.index("capture-economics"), refresh.index(intent_step))
        verify_index = refresh.index("verify-projections")
        intent_index = refresh.index(intent_step)
        post_projection_snapshot = [
            index
            for index, step in enumerate(refresh)
            if step == "broker-snapshot --output data/broker_snapshot.json"
            and verify_index < index < intent_index
        ]
        post_projection_target = [
            index
            for index, step in enumerate(refresh)
            if step == "daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10"
            and verify_index < index < intent_index
        ]
        self.assertTrue(post_projection_snapshot)
        self.assertTrue(post_projection_target)
        self.assertLess(post_projection_snapshot[-1], post_projection_target[-1])
        self.assertLess(post_projection_target[-1], intent_index)
        self.assertLess(refresh.index("tp-rebalance"), refresh.index(intent_step))
        self.assertIn("news-snapshot", refresh)
        self.assertIn("news-health --strict", refresh)
        story_step = (
            "mine-market-stories --news-dir logs --profile "
            "data/market_story_profile.json --report data/market_story_report.md"
        )
        self.assertLess(refresh.index("news-snapshot"), refresh.index(story_step))
        self.assertLess(refresh.index("news-snapshot"), refresh.index("news-health --strict"))
        self.assertLess(refresh.index("capture-economics"), refresh.index("operator-precedent-audit"))
        self.assertLess(refresh.index("capture-economics"), refresh.index("manual-market-context-audit"))
        self.assertIn("execution-timing-audit --max-events 80", refresh)
        self.assertFalse(
            any(
                step.startswith("execution-timing-audit --lookback-hours 24")
                for step in refresh
            )
        )
        self.assertLess(refresh.index("execution-timing-audit --max-events 80"), refresh.index("self-improvement-audit"))
        self.assertLess(refresh.index("manual-market-context-audit"), refresh.index("operator-precedent-audit"))
        self.assertLess(refresh.index("operator-precedent-audit"), refresh.index("verification-ledger-audit"))
        self.assertLess(refresh.index("memory-health"), refresh.index("self-improvement-audit"))
        self.assertLess(refresh.index("self-improvement-audit"), refresh.index("profitability-acceptance"))
        self.assertLess(refresh.index("profitability-acceptance"), refresh.index("trader-support-bot"))
        self.assertEqual(refresh[-1], "trader-support-bot")
        refresh_by_step = {" ".join(s["argv"]): s for s in _cycle_refresh_steps("10")}
        self.assertEqual(refresh_by_step["execution-timing-audit --max-events 80"]["timeout_seconds"], 60.0)
        self.assertFalse(refresh_by_step["execution-timing-audit --max-events 80"]["required"])
        self.assertTrue(refresh_by_step["position-management"]["required"])
        self.assertTrue(refresh_by_step["memory-health"]["required"])
        self.assertTrue(refresh_by_step["profitability-acceptance"]["required"])
        self.assertEqual(refresh_by_step["profitability-acceptance"]["ok_rcs"], [0, 2])
        self.assertTrue(refresh_by_step["trader-support-bot"]["required"])
        self.assertEqual(refresh_by_step["trader-support-bot"]["ok_rcs"], [0, 2])

        with mock.patch.dict(os.environ, {"QR_LIVE_ENABLED": ""}, clear=False):
            sidecar_specs = _cycle_sidecar_steps()
            sidecars = [" ".join(s["argv"]) for s in sidecar_specs]
        self.assertIn("profit-partial-close", sidecars)
        self.assertNotIn("profit-partial-close --send --confirm-live", sidecars)
        self.assertIn("position-execution", sidecars)
        self.assertNotIn("position-execution --send --confirm-live", sidecars)
        self.assertLess(sidecars.index("forecast-persistence-check"), sidecars.index("position-management"))
        self.assertLess(sidecars.index("position-management"), sidecars.index("position-execution"))
        self.assertLess(sidecars.index("position-execution"), sidecars.index("memory-health"))
        self.assertLess(sidecars.index("memory-health"), sidecars.index("self-improvement-audit"))
        self.assertLess(sidecars.index("self-improvement-audit"), sidecars.index("profitability-acceptance"))
        self.assertLess(sidecars.index("profitability-acceptance"), sidecars.index("trader-support-bot"))
        self.assertEqual(sidecars[-1], "trader-support-bot")
        sidecars_by_step = {" ".join(s["argv"]): s for s in sidecar_specs}
        self.assertTrue(sidecars_by_step["position-management"]["required"])
        self.assertFalse(sidecars_by_step["position-execution"]["required"])
        self.assertTrue(sidecars_by_step["memory-health"]["required"])
        self.assertTrue(sidecars_by_step["profitability-acceptance"]["required"])
        self.assertEqual(sidecars_by_step["profitability-acceptance"]["ok_rcs"], [0, 2])
        self.assertTrue(sidecars_by_step["trader-support-bot"]["required"])
        self.assertEqual(sidecars_by_step["trader-support-bot"]["ok_rcs"], [0, 2])

        with mock.patch.dict(os.environ, {"QR_LIVE_ENABLED": "1"}, clear=False):
            sidecars_live = [" ".join(s["argv"]) for s in _cycle_sidecar_steps()]
        self.assertIn("profit-partial-close --send --confirm-live", sidecars_live)
        self.assertIn("position-execution --send --confirm-live", sidecars_live)


    def test_direct_autotrade_audit_sidecars_run_without_wrapper_lock(self) -> None:
        digest = {
            "kind": "direct_autotrade_audit_sidecars_digest",
            "aborted": False,
            "steps_failed": [],
        }
        step_results = [{"step": "memory-health", "status": "OK", "rc": 0}]

        with (
            mock.patch.dict(
                os.environ,
                {
                    "QR_RUN_DIRECT_AUTOTRADE_AUDIT_SIDECARS": "1",
                    "QR_AUTOTRADE_LOCK_HELD": "",
                },
                clear=False,
            ),
            mock.patch("quant_rabbit.cli._running_under_test_harness", return_value=False),
            mock.patch("quant_rabbit.cli._run_cycle_steps", return_value=(step_results, False)) as run_steps,
            mock.patch("quant_rabbit.cli._cycle_digest", return_value=digest) as cycle_digest,
            mock.patch("quant_rabbit.cli._write_json") as write_json,
        ):
            result = _run_direct_autotrade_audit_sidecars()

        self.assertEqual(result, digest)
        run_steps.assert_called_once_with(_direct_autotrade_audit_sidecar_steps())
        direct_sidecars = [" ".join(s["argv"]) for s in _direct_autotrade_audit_sidecar_steps()]
        direct_sidecar_specs = {" ".join(s["argv"]): s for s in _direct_autotrade_audit_sidecar_steps()}
        self.assertEqual(direct_sidecars[0], "verify-projections")
        self.assertLess(direct_sidecars.index("verify-projections"), direct_sidecars.index("memory-health"))
        self.assertLess(direct_sidecars.index("verify-projections"), direct_sidecars.index("self-improvement-audit"))
        self.assertLess(direct_sidecars.index("self-improvement-audit"), direct_sidecars.index("profitability-acceptance"))
        self.assertLess(direct_sidecars.index("profitability-acceptance"), direct_sidecars.index("trader-support-bot"))
        self.assertEqual(direct_sidecars[-1], "trader-support-bot")
        self.assertTrue(direct_sidecar_specs["profitability-acceptance"]["required"])
        self.assertEqual(direct_sidecar_specs["profitability-acceptance"]["ok_rcs"], [0, 2])
        self.assertTrue(direct_sidecar_specs["trader-support-bot"]["required"])
        self.assertEqual(direct_sidecar_specs["trader-support-bot"]["ok_rcs"], [0, 2])
        cycle_digest.assert_called_once_with(
            kind="direct_autotrade_audit_sidecars_digest",
            step_results=step_results,
            aborted=False,
        )
        write_json.assert_called_once_with(DIRECT_AUTOTRADE_AUDIT_SIDECARS_DIGEST, digest)

    def test_direct_autotrade_audit_sidecars_skip_under_wrapper_lock(self) -> None:
        with (
            mock.patch.dict(os.environ, {"QR_AUTOTRADE_LOCK_HELD": "1"}, clear=False),
            mock.patch("quant_rabbit.cli._running_under_test_harness", return_value=False),
            mock.patch("quant_rabbit.cli._run_cycle_steps") as run_steps,
        ):
            result = _run_direct_autotrade_audit_sidecars()

        self.assertIsNone(result)
        run_steps.assert_not_called()

    def test_direct_autotrade_audit_sidecars_can_be_disabled(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {"QR_RUN_DIRECT_AUTOTRADE_AUDIT_SIDECARS": "0", "QR_AUTOTRADE_LOCK_HELD": ""},
                clear=False,
            ),
            mock.patch("quant_rabbit.cli._running_under_test_harness", return_value=False),
            mock.patch("quant_rabbit.cli._run_cycle_steps") as run_steps,
        ):
            result = _run_direct_autotrade_audit_sidecars()

        self.assertIsNone(result)
        run_steps.assert_not_called()

    def test_cycle_digest_summarizes_failed_steps(self) -> None:
        from quant_rabbit.cli import _cycle_digest

        digest = _cycle_digest(
            kind="cycle_refresh_digest",
            step_results=[
                {"step": "broker-snapshot", "status": "OK", "rc": 0, "seconds": 0.1},
                {"step": "news-health --strict", "status": "FAILED", "rc": 1, "seconds": 0.1, "tail": "stale"},
            ],
            aborted=False,
        )

        self.assertEqual(digest["kind"], "cycle_refresh_digest")
        self.assertFalse(digest["aborted"])
        self.assertEqual(digest["steps_ok"], ["broker-snapshot"])
        self.assertEqual(len(digest["steps_failed"]), 1)
        self.assertEqual(digest["steps_failed"][0]["step"], "news-health --strict")

    def test_cycle_digest_uses_live_blocker_codes(self) -> None:
        from quant_rabbit.cli import _cycle_digest

        with tempfile.TemporaryDirectory() as tmp:
            intents_path = Path(tmp) / "order_intents.json"
            intents_path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-01-02T00:00:00+00:00",
                        "results": [
                            {
                                "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                                "status": "DRY_RUN_PASSED",
                                "live_blockers": [
                                    "EUR_USD LONG forecast UP confidence 0.48 < 0.65 for live send"
                                ],
                                "live_blocker_codes": ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"],
                                "risk_issues": [
                                    {
                                        "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                                        "message": "EUR_USD LONG forecast UP confidence 0.48 < 0.65 for live send",
                                        "severity": "WARN",
                                    }
                                ],
                            },
                            {
                                "lane_id": "legacy:EUR_USD:SHORT:RANGE_ROTATION",
                                "status": "DRY_RUN_BLOCKED",
                                "live_blockers": [{"code": "LEGACY_BLOCKER"}],
                            }
                        ],
                    }
                )
            )

            with mock.patch("quant_rabbit.cli.DEFAULT_ORDER_INTENTS", intents_path):
                digest = _cycle_digest(
                    kind="cycle_refresh_digest",
                    step_results=[],
                    aborted=False,
                )

        self.assertEqual(
            digest["intents"]["top_blockers"],
            {"FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE": 1, "LEGACY_BLOCKER": 1},
        )

    def test_cycle_digest_summarizes_trader_support_bot(self) -> None:
        from quant_rabbit.cli import _cycle_digest

        with tempfile.TemporaryDirectory() as tmp:
            support_path = Path(tmp) / "trader_support_bot.json"
            support_path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-22T12:15:00+00:00",
                        "status": "SUPPORT_BLOCKED",
                        "metrics": {
                            "send_fresh_entries_allowed": False,
                            "guardian_active": False,
                            "guardian_heartbeat_fresh": False,
                            "profit_capture_missed_loss_closes": 2,
                            "profit_capture_estimated_gap_jpy": 646.489,
                            "live_ready_lanes": 0,
                            "repair_frontier_lanes": 8,
                        },
                        "guardian": {
                            "active_source": "plist_missing",
                            "heartbeat_age_seconds": 1234.0,
                        },
                        "blockers": [
                            {"code": "POSITION_GUARDIAN_INACTIVE"},
                            {"code": "LOSS_CLOSE_PROFIT_CAPTURE_MISSED"},
                        ],
                        "operator_actions": [
                            {"code": "CHECK_POSITION_GUARDIAN_PREFLIGHT"},
                            {"code": "LOAD_POSITION_GUARDIAN_ONLY_IF_APPROVED"},
                        ],
                        "profit_capture": {
                            "top_misses": [{"trade_id": "472792", "pair": "USD_JPY"}],
                        },
                        "entry_readiness": {
                            "repair_frontier": [
                                {
                                    "lane_id": "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                                    "remaining_blocker_codes_after_guardian_and_repair_exemption": [
                                        "FORECAST_CONTEXT_REQUIRED_FOR_LIVE"
                                    ],
                                }
                            ],
                        },
                    }
                )
            )

            with mock.patch("quant_rabbit.cli.DEFAULT_TRADER_SUPPORT_BOT", support_path):
                digest = _cycle_digest(kind="cycle_refresh_digest", step_results=[], aborted=False)

        support = digest["trader_support_bot"]
        self.assertEqual(support["status"], "SUPPORT_BLOCKED")
        self.assertFalse(support["send_fresh_entries_allowed"])
        self.assertFalse(support["guardian_active"])
        self.assertEqual(support["guardian_active_source"], "plist_missing")
        self.assertEqual(support["profit_capture_missed_loss_closes"], 2)
        self.assertEqual(support["repair_frontier_lanes"], 8)
        self.assertEqual(
            support["top_blocker_codes"],
            ["POSITION_GUARDIAN_INACTIVE", "LOSS_CLOSE_PROFIT_CAPTURE_MISSED"],
        )
        self.assertEqual(
            support["operator_action_codes"],
            ["CHECK_POSITION_GUARDIAN_PREFLIGHT", "LOAD_POSITION_GUARDIAN_ONLY_IF_APPROVED"],
        )
        self.assertEqual(support["top_profit_capture_misses"][0]["trade_id"], "472792")
        self.assertEqual(
            support["repair_frontier"][0]["remaining_blocker_codes_after_guardian_and_repair_exemption"],
            ["FORECAST_CONTEXT_REQUIRED_FOR_LIVE"],
        )

    def test_cycle_digest_summarizes_operator_precedent(self) -> None:
        from quant_rabbit.cli import _cycle_digest

        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "operator_precedent_audit.json"
            manual_context_path = Path(tmp) / "manual_market_context_audit.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "status": "OPERATOR_PRECEDENT_WARN",
                        "operator_claim": {"verified": True},
                        "precedent": {
                            "funding_adjusted_performance": {
                                "best_30d": {"return_pct": 319.72}
                            },
                            "winning_shape": {
                                "primary_pair": "USD_JPY",
                                "primary_direction": "LONG",
                                "primary_sessions": ["LONDON_AM", "NY_OVERLAP"],
                                "positive_sessions": ["LONDON_AM", "NY_OVERLAP"],
                            },
                        },
                        "runtime_alignment": {
                            "live_ready_lanes": 1,
                            "aligned_live_ready_lanes": 0,
                            "manual_context_alignment": {
                                "status": "MANUAL_CONTEXT_ALIGNMENT_READY",
                                "compatible_lanes": [{"lane_id": "a"}],
                                "conflicting_lanes": [{"lane_id": "b"}],
                                "conflicting_aligned_lanes": 0,
                            },
                        },
                        "warnings": ["not aligned"],
                        "blockers": [],
                    }
                )
            )
            manual_context_path.write_text(
                json.dumps(
                    {
                        "status": "MANUAL_MARKET_CONTEXT_PASS",
                        "sample": {
                            "pair": "USD_JPY",
                            "analyzed_trades": 411,
                            "coverage_pct": 100.0,
                        },
                        "guidance": {
                            "prefer_when_citing_precedent": {
                                "h1_alignment": "WITH_H1_TREND",
                                "session_jst": "LONDON_AM",
                            },
                            "require_extra_current_reason_when_conflicting": {
                                "h1_alignment": "AGAINST_H1_TREND",
                            },
                        },
                        "position_building_profile": {
                            "bounded_lt_12h_excluding_margin_closeout": {
                                "multi_entry_clusters": 10,
                                "net_jpy": 108343.7,
                            },
                            "adverse_adds": {
                                "clusters": 8,
                                "net_jpy": 102564.0,
                                "avg_adverse_add_pips": 6.45,
                            },
                        },
                        "warnings": [],
                        "blockers": [],
                    }
                )
            )

            with mock.patch("quant_rabbit.cli.DEFAULT_OPERATOR_PRECEDENT_AUDIT", audit_path), mock.patch(
                "quant_rabbit.cli.DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT",
                manual_context_path,
            ):
                digest = _cycle_digest(kind="cycle_refresh_digest", step_results=[], aborted=False)

        precedent = digest["operator_precedent"]
        self.assertEqual(precedent["status"], "OPERATOR_PRECEDENT_WARN")
        self.assertTrue(precedent["claim_verified"])
        self.assertEqual(precedent["best_30d_return_pct"], 319.72)
        self.assertEqual(precedent["primary_pair"], "USD_JPY")
        self.assertEqual(precedent["primary_sessions"], ["LONDON_AM", "NY_OVERLAP"])
        self.assertEqual(precedent["aligned_live_ready_lanes"], 0)
        self.assertEqual(precedent["manual_context_alignment_status"], "MANUAL_CONTEXT_ALIGNMENT_READY")
        self.assertEqual(precedent["manual_context_compatible_lanes"], 1)
        self.assertEqual(precedent["manual_context_conflicting_lanes"], 1)
        self.assertEqual(precedent["manual_context_conflicting_aligned_lanes"], 0)
        manual_context = digest["manual_market_context"]
        self.assertEqual(manual_context["status"], "MANUAL_MARKET_CONTEXT_PASS")
        self.assertEqual(manual_context["prefer_h1_alignment"], "WITH_H1_TREND")
        self.assertEqual(manual_context["conflict_h1_alignment"], "AGAINST_H1_TREND")
        self.assertEqual(manual_context["position_building"]["adverse_add_clusters"], 8)
        self.assertEqual(manual_context["position_building"]["adverse_add_net_jpy"], 102564.0)
        self.assertFalse(manual_context["position_building"]["nanpin_is_live_permission"])


class PairChartsCommandTest(unittest.TestCase):
    class _FakeChart:
        def __init__(self, pair: str, long_score: float = 0.7, short_score: float = 0.3) -> None:
            self.pair = pair
            self.long_score = long_score
            self.short_score = short_score

        def to_dict(self) -> dict[str, object]:
            return {
                "pair": self.pair,
                "long_score": self.long_score,
                "short_score": self.short_score,
                "dominant_regime": "RANGE",
                "chart_story": f"{self.pair} test chart",
                "warnings": [],
                "views": [],
                "session": None,
                "confluence": {},
            }

    def test_pair_charts_writes_partial_success_instead_of_stale_whole_packet(self) -> None:
        def fake_build(pair: str, **_kwargs: object) -> PairChartsCommandTest._FakeChart:
            if pair == "GBP_USD":
                raise RuntimeError("broker candle timeout")
            return self._FakeChart(pair)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pair_charts.json"
            report = Path(tmp) / "pair_charts.md"
            stdout = io.StringIO()
            with (
                mock.patch("quant_rabbit.cli.OandaReadOnlyClient", return_value=object()),
                mock.patch("quant_rabbit.analysis.chart_reader.build_pair_chart", side_effect=fake_build),
                mock.patch("quant_rabbit.analysis.score_momentum.attach_score_momentum"),
                redirect_stdout(stdout),
            ):
                rc = main(
                    [
                        "pair-charts",
                        "--pairs",
                        "EUR_USD,GBP_USD,AUD_USD",
                        "--timeframes",
                        "M5",
                        "--workers",
                        "2",
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                    ]
                )

            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text())
            self.assertTrue(payload["partial"])
            self.assertEqual(payload["pairs_requested"], 3)
            self.assertEqual(payload["pairs_succeeded"], 2)
            self.assertEqual(payload["pairs_failed"], 1)
            self.assertEqual(payload["failures"][0]["pair"], "GBP_USD")
            self.assertEqual({chart["pair"] for chart in payload["charts"]}, {"EUR_USD", "AUD_USD"})
            printed = json.loads(stdout.getvalue())
            self.assertTrue(printed["partial"])
            self.assertEqual(printed["pairs_failed"], 1)
            self.assertIn("GBP_USD", report.read_text())

    def test_pair_charts_returns_failure_when_every_pair_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pair_charts.json"
            stdout = io.StringIO()
            with (
                mock.patch("quant_rabbit.cli.OandaReadOnlyClient", return_value=object()),
                mock.patch(
                    "quant_rabbit.analysis.chart_reader.build_pair_chart",
                    side_effect=RuntimeError("broker down"),
                ),
                redirect_stdout(stdout),
            ):
                rc = main(
                    [
                        "pair-charts",
                        "--pairs",
                        "EUR_USD,GBP_USD",
                        "--timeframes",
                        "M5",
                        "--workers",
                        "2",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(rc, 2)
            self.assertFalse(output.exists())
            printed = json.loads(stdout.getvalue())
            self.assertEqual(printed["pairs_requested"], 2)
            self.assertEqual(len(printed["failures"]), 2)


if __name__ == "__main__":
    unittest.main()

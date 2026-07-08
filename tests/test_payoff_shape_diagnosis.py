from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any

from quant_rabbit.payoff_shape_diagnosis import build_payoff_shape_diagnosis


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _make_ledger(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT,
                source TEXT,
                event_type TEXT,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                client_order_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                price REAL,
                tp REAL,
                sl REAL,
                realized_pl_jpy REAL,
                financing_jpy REAL,
                exit_reason TEXT,
                oanda_transaction_id TEXT,
                related_transaction_ids_json TEXT,
                raw_json TEXT,
                inserted_at_utc TEXT
            )
            """
        )
        rows: list[tuple[Any, ...]] = []
        closes = [
            ("h1", "EUR_USD", "LONG", "BREAKOUT_FAILURE", "TAKE_PROFIT_ORDER", 100.0),
            ("h2", "EUR_USD", "LONG", "BREAKOUT_FAILURE", "TAKE_PROFIT_ORDER", 120.0),
            ("h3", "EUR_USD", "LONG", "BREAKOUT_FAILURE", "MARKET_ORDER_TRADE_CLOSE", -80.0),
            ("n1", "AUD_USD", "SHORT", "RANGE_ROTATION", "STOP_LOSS_ORDER", -200.0),
        ]
        for idx, (trade_id, pair, side, method, exit_reason, pl) in enumerate(closes):
            lane_id = f"failure_trader:{pair}:{side}:{method}"
            rows.append(
                (
                    f"fill-{trade_id}",
                    f"2026-06-0{idx + 1}T00:00:00Z",
                    "test",
                    "ORDER_FILLED",
                    lane_id,
                    f"o-{trade_id}",
                    trade_id,
                    None,
                    pair,
                    side,
                    1000,
                    1.0,
                    1.1,
                    0.9,
                    None,
                    None,
                    None,
                    None,
                    "[]",
                    "{}",
                    "2026-06-01T00:00:00Z",
                )
            )
            rows.append(
                (
                    f"close-{trade_id}",
                    f"2026-06-0{idx + 1}T01:00:00Z",
                    "test",
                    "TRADE_CLOSED",
                    None,
                    f"o-{trade_id}",
                    trade_id,
                    None,
                    pair,
                    side,
                    1000,
                    1.0,
                    None,
                    None,
                    pl,
                    0.0,
                    exit_reason,
                    None,
                    "[]",
                    "{}",
                    "2026-06-01T00:00:00Z",
                )
            )
        conn.executemany(
            """
            INSERT INTO execution_events (
                event_uid, ts_utc, source, event_type, lane_id, order_id, trade_id,
                client_order_id, pair, side, units, price, tp, sl, realized_pl_jpy,
                financing_jpy, exit_reason, oanda_transaction_id,
                related_transaction_ids_json, raw_json, inserted_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )


def _make_target_ledger(path: Path, *, tp_count: int = 17) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
                event_uid TEXT PRIMARY KEY,
                ts_utc TEXT,
                source TEXT,
                event_type TEXT,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                client_order_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                price REAL,
                tp REAL,
                sl REAL,
                realized_pl_jpy REAL,
                financing_jpy REAL,
                exit_reason TEXT,
                oanda_transaction_id TEXT,
                related_transaction_ids_json TEXT,
                raw_json TEXT,
                inserted_at_utc TEXT
            )
            """
        )
        rows: list[tuple[Any, ...]] = []
        for idx in range(tp_count):
            trade_id = f"target-{idx}"
            rows.append(
                (
                    f"fill-{trade_id}",
                    f"2026-06-{idx % 9 + 1:02d}T00:00:00Z",
                    "test",
                    "ORDER_FILLED",
                    "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                    f"o-{trade_id}",
                    trade_id,
                    None,
                    "EUR_USD",
                    "SHORT",
                    1000,
                    1.0,
                    0.999,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "[]",
                    "{}",
                    "2026-06-01T00:00:00Z",
                )
            )
            rows.append(
                (
                    f"close-{trade_id}",
                    f"2026-06-{idx % 9 + 1:02d}T01:00:00Z",
                    "test",
                    "TRADE_CLOSED",
                    None,
                    f"o-{trade_id}",
                    trade_id,
                    None,
                    "EUR_USD",
                    "SHORT",
                    1000,
                    1.0,
                    None,
                    None,
                    100.0,
                    0.0,
                    "TAKE_PROFIT_ORDER",
                    None,
                    "[]",
                    "{}",
                    "2026-06-01T00:00:00Z",
                )
            )
        conn.executemany(
            """
            INSERT INTO execution_events (
                event_uid, ts_utc, source, event_type, lane_id, order_id, trade_id,
                client_order_id, pair, side, units, price, tp, sl, realized_pl_jpy,
                financing_jpy, exit_reason, oanda_transaction_id,
                related_transaction_ids_json, raw_json, inserted_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )


class PayoffShapeDiagnosisTest(unittest.TestCase):
    def test_diagnosis_separates_harvest_partial_and_no_trade_without_live_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "ledger.db"
            _make_ledger(ledger)
            capture = root / "capture.json"
            timing = root / "timing.json"
            intents = root / "order_intents.json"
            replay = root / "replay.json"
            month_scale = root / "month_scale.json"
            output = root / "payoff_shape_diagnosis.json"
            report = root / "payoff_shape_diagnosis_report.md"

            _write_json(
                capture,
                {
                    "status": "NEGATIVE_EXPECTANCY",
                    "repair_summary": {"dominant_loss_exit_reason": "MARKET_ORDER_TRADE_CLOSE"},
                },
            )
            _write_json(
                timing,
                {
                    "summary": {
                        "loss_closes_audited": 1,
                        "market_closes_audited": 1,
                        "profit_market_closes_left_runner_upside": 1,
                    },
                    "market_close_counterfactuals": [
                        {
                            "trade_id": "h3",
                            "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "post_close_path_label": "PROFIT_CLOSE_LEFT_RUNNER_UPSIDE",
                            "post_close_favorable_pips": 18.0,
                            "estimated_post_close_favorable_jpy": 900.0,
                            "tp_touched_after_market_close": True,
                        }
                    ],
                    "loss_close_regrets": [
                        {
                            "trade_id": "n1",
                            "lane_id": "failure_trader:AUD_USD:SHORT:RANGE_ROTATION",
                            "pair": "AUD_USD",
                            "side": "SHORT",
                            "exit_reason": "STOP_LOSS_ORDER",
                            "realized_pl_jpy": -200.0,
                            "had_positive_mfe_before_loss_close": True,
                            "profit_capture_missed_before_loss_close": True,
                            "mfe_pips_before_loss_close": 4.0,
                            "estimated_mfe_jpy_before_loss_close": 60.0,
                        }
                    ],
                },
            )
            _write_json(
                intents,
                {
                    "results": [
                        {
                            "lane_id": "failure_trader:AUD_USD:SHORT:RANGE_ROTATION",
                            "status": "DRY_RUN_BLOCKED",
                            "risk_allowed": False,
                            "live_blocker_codes": ["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                            "risk_metrics": {"reward_risk": 1.1, "risk_jpy": 100.0},
                            "intent": {
                                "pair": "AUD_USD",
                                "side": "SHORT",
                                "market_context": {"method": "RANGE_ROTATION"},
                                "metadata": {
                                    "session_current_tag": "LONDON",
                                    "tp_target_intent": "HARVEST",
                                    "bidask_replay_precision_negative": {
                                        "name": "AUD_USD_SHORT_RANGE_ROTATION",
                                        "avg_mfe_pips": 1.0,
                                        "avg_mae_pips": 5.0,
                                        "avg_final_pips": -2.5,
                                        "samples": 50,
                                        "blocks_live_support": True,
                                    },
                                },
                            },
                        }
                    ]
                },
            )
            _write_json(replay, {"summary": {"total_historical_net_jpy": -100.0}})
            _write_json(
                month_scale,
                {
                    "fresh_entries_must_remain_blocked": True,
                    "blocker": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
                    "pl_summary": {"current_residual_pl_jpy": -500.0},
                    "method_rollups": [
                        {"method": "RANGE_ROTATION", "actual_pl_jpy": -500.0},
                    ],
                    "residual_losing_families": [
                        {
                            "pair": "AUD_USD",
                            "side": "SHORT",
                            "strategy": "RANGE_ROTATION",
                            "actual_pl_jpy": -500.0,
                            "block_reasons": ["NEGATIVE_RESIDUAL"],
                        }
                    ],
                },
            )

            summary = build_payoff_shape_diagnosis(
                ledger_path=ledger,
                capture_economics_path=capture,
                execution_timing_audit_path=timing,
                order_intents_path=intents,
                replay_backtest_path=replay,
                month_scale_residuals_path=month_scale,
                output_path=output,
                report_path=report,
            )

            self.assertEqual(summary.status, "OK")
            payload = json.loads(output.read_text())
            required_keys = {
                "harvest_candidates",
                "runner_candidates",
                "partial_tp_runner_candidates",
                "no_trade_shapes",
                "family_stats",
                "pair_stats",
                "session_stats",
                "mfe_mae_summary",
                "missed_runner_cases",
                "overheld_harvest_cases",
                "payoff_shape_recommendations",
                "next_evidence_actions",
                "live_side_effects",
            }
            self.assertTrue(required_keys.issubset(payload))
            self.assertEqual(payload["live_side_effects"], [])
            self.assertTrue(payload["safety_contract"]["no_live_order"])
            self.assertTrue(payload["safety_contract"]["proof_queue_count_is_not_live_permission"])
            self.assertTrue(payload["safety_contract"]["no_4x_deficit_lot_backsolve"])
            self.assertEqual(payload["overall_payoff_shape_verdict"]["classification"], "MIXED_HARVEST_PRIMARY")
            self.assertFalse(payload["overall_payoff_shape_verdict"]["live_promotion_allowed"])
            self.assertIn("NEGATIVE_EXPECTANCY", payload["overall_payoff_shape_verdict"]["live_promotion_blockers"])
            self.assertIn("MONTH_SCALE_REPLAY_NEGATIVE", payload["overall_payoff_shape_verdict"]["live_promotion_blockers"])
            self.assertEqual(payload["harvest_candidates"][0]["shape_key"], "EUR_USD|LONG|BREAKOUT_FAILURE")
            self.assertFalse(any(row["live_promotion_allowed"] for row in payload["harvest_candidates"]))
            self.assertEqual(payload["partial_tp_runner_candidates"][0]["shape_key"], "EUR_USD|LONG|BREAKOUT_FAILURE")
            self.assertTrue(any(row["shape_key"] == "AUD_USD|SHORT|RANGE_ROTATION" for row in payload["no_trade_shapes"]))
            self.assertIn("LONDON", payload["session_stats"])
            self.assertGreater(payload["mfe_mae_summary"]["execution_timing"]["post_close_runner"]["estimated_followthrough_jpy"], 0)
            self.assertTrue(report.exists())
            report_text = report.read_text()
            self.assertIn("MIXED_HARVEST_PRIMARY", report_text)
            self.assertIn("Live side effects: `[]`", report_text)

    def test_applies_read_only_canonical_proof_floor_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "ledger.db"
            _make_target_ledger(ledger, tp_count=17)
            capture = root / "capture.json"
            timing = root / "timing.json"
            intents = root / "order_intents.json"
            replay = root / "replay.json"
            month_scale = root / "month_scale.json"
            proof_update = root / "proof_floor_update.json"
            output = root / "payoff_shape_diagnosis.json"
            report = root / "payoff_shape_diagnosis_report.md"

            _write_json(capture, {"status": "NEGATIVE_EXPECTANCY"})
            _write_json(timing, {"summary": {"loss_closes_audited": 0}})
            _write_json(intents, {"results": []})
            _write_json(replay, {"summary": {"total_historical_net_jpy": 1.0}})
            _write_json(month_scale, {"fresh_entries_must_remain_blocked": False})
            _write_json(
                proof_update,
                {
                    "target_shape": "EUR_USD|SHORT|BREAKOUT_FAILURE",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "canonical_integration_status": "CANONICAL_PROOF_UPDATE_READY_AS_EVIDENCE_ONLY",
                    "pre_update_tp_proof": {"wins": 17, "losses": 0, "proof_floor": 20, "remaining_samples": 3},
                    "post_update_tp_proof": {
                        "wins": 20,
                        "losses": 0,
                        "proof_floor": 20,
                        "remaining_samples": 0,
                        "proof_floor_reached": True,
                    },
                    "accepted_sample_checks": [
                        {"trade_id": "469278", "realized_pl_jpy": 10.0},
                        {"trade_id": "469427", "realized_pl_jpy": 20.0},
                        {"trade_id": "469898", "realized_pl_jpy": 30.0},
                    ],
                    "required_checks": {
                        "duplicates_checked": {"passed": True},
                        "tp_or_attached_harvest_checked": {"passed": True},
                        "market_close_excluded": {"passed": True},
                        "spread_slippage_fields_present": {"passed": True},
                        "live_permission_not_created": {"passed": True},
                    },
                },
            )

            build_payoff_shape_diagnosis(
                ledger_path=ledger,
                capture_economics_path=capture,
                execution_timing_audit_path=timing,
                order_intents_path=intents,
                replay_backtest_path=replay,
                month_scale_residuals_path=month_scale,
                proof_floor_update_path=proof_update,
                output_path=output,
                report_path=report,
            )

            payload = json.loads(output.read_text())
            self.assertTrue(payload["canonical_proof_reconciliation"]["applied"])
            self.assertEqual(payload["canonical_proof_reconciliation"]["accepted_legacy_sample_trade_ids"], ["469278", "469427", "469898"])
            target = next(row for row in payload["harvest_candidates"] if row["shape_key"] == "EUR_USD|SHORT|BREAKOUT_FAILURE")
            self.assertEqual(target["classification"], "HARVEST_PROOF_FLOOR_REACHED_EVIDENCE_ONLY")
            self.assertEqual(target["take_profit_trades"], 20)
            self.assertEqual(target["take_profit_losses"], 0)
            self.assertEqual(target["proof_gap_trades"], 0)
            self.assertEqual(target["take_profit_net_jpy"], 1760.0)
            self.assertEqual(target["canonical_proof_reconciliation"]["scope"], "broad_take_profit_order_proof_only_not_exact_limit_sample_floor")
            self.assertFalse(payload["overall_payoff_shape_verdict"]["live_promotion_allowed"])


if __name__ == "__main__":
    unittest.main()

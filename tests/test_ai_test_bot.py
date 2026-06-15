from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.ai_test_bot import AITestBotBacktester, TestBotBucket, TestBotTrade, _row_matches_selected_buckets


class AITestBotBacktesterTest(unittest.TestCase):
    def test_defaults_use_recent_pretrade_evidence_with_high_support_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            rows = []
            for index in range(10):
                day = f"2026-04-{1 + (index % 6):02d}"
                rows.append(
                    (
                        "pretrade_outcomes",
                        day,
                        "EUR_USD",
                        "LONG",
                        100.0,
                        json.dumps({"id": f"train-{index}", "pretrade_level": "HIGH"}),
                    )
                )
            rows.extend(
                [
                    (
                        "pretrade_outcomes",
                        "2026-04-07",
                        "EUR_USD",
                        "LONG",
                        150.0,
                        json.dumps({"id": "validation-1", "pretrade_level": "HIGH"}),
                    ),
                    (
                        "pretrade_outcomes",
                        "2026-04-07",
                        "EUR_USD",
                        "LONG",
                        200.0,
                        json.dumps({"id": "validation-2", "pretrade_level": "HIGH"}),
                    ),
                ]
            )
            _seed_db(db, rows)

            summary = AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
            ).run(start_balance_jpy=10_000.0)

            self.assertEqual(summary.validation_days, 1)
            payload = json.loads((root / "ai_backtest.json").read_text())
            self.assertEqual(payload["source_tables"], ["trades", "pretrade_outcomes", "seat_outcomes"])
            self.assertEqual(payload["training_days"], 6)
            self.assertEqual(payload["min_train_trades"], 10)
            self.assertEqual(payload["max_active_buckets"], 6)
            self.assertFalse(payload["target_ceiling"]["prediction_only_target_possible"])
            self.assertEqual(payload["target_ceiling"]["oracle_target_gap_jpy"], 650.0)
            self.assertTrue(any("archive opportunity ceiling misses 10% target" in item for item in payload["action_items"]))
            self.assertIn("All-positive oracle ceiling", (root / "ai_backtest.md").read_text())
            self.assertIn("Source Contributions", (root / "ai_backtest.md").read_text())
            day = payload["days"][0]
            self.assertEqual(
                day["selected_buckets"],
                ["pretrade_outcomes:EUR_USD:LONG:HIGH:UNSPECIFIED"],
            )
            self.assertEqual(day["managed_net_jpy"], 350.0)

    def test_default_backtest_reports_unselected_negative_seat_discovery_drag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_db(
                db,
                [
                    (
                        "pretrade_outcomes",
                        "2026-04-01",
                        "EUR_USD",
                        "LONG",
                        100.0,
                        json.dumps({"id": "train", "pretrade_level": "HIGH"}),
                    ),
                    (
                        "pretrade_outcomes",
                        "2026-04-02",
                        "EUR_USD",
                        "LONG",
                        150.0,
                        json.dumps({"id": "validation", "pretrade_level": "HIGH"}),
                    ),
                    (
                        "seat_outcomes",
                        "2026-04-02",
                        "EUR_USD",
                        "SHORT",
                        -300.0,
                        json.dumps({"source": "s_hunt", "setup_type": "MARKET", "id": "seat-validation"}),
                    ),
                ],
            )

            AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
            ).run(start_balance_jpy=2000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            by_source = {item["source_table"]: item for item in payload["source_contributions"]}
            self.assertEqual(payload["source_tables"], ["trades", "pretrade_outcomes", "seat_outcomes"])
            self.assertEqual(by_source["seat_outcomes"]["validation_universe_managed_net_jpy"], -100.0)
            self.assertEqual(by_source["seat_outcomes"]["selected_trades"], 0)
            self.assertTrue(any("seat_outcomes discovery universe is negative" in item for item in payload["action_items"]))
            report = (root / "ai_backtest.md").read_text()
            self.assertIn("`seat_outcomes` validation_net=`-100`", report)

    def test_reports_target_band_between_floor_and_stretch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            rows = []
            for index in range(10):
                day = f"2026-04-{1 + (index % 6):02d}"
                rows.append(
                    (
                        "pretrade_outcomes",
                        day,
                        "EUR_USD",
                        "LONG",
                        100.0,
                        json.dumps({"id": f"train-{index}", "pretrade_level": "HIGH"}),
                    )
                )
            rows.append(
                (
                    "pretrade_outcomes",
                    "2026-04-07",
                    "EUR_USD",
                    "LONG",
                    550.0,
                    json.dumps({"id": "validation", "pretrade_level": "HIGH"}),
                )
            )
            _seed_db(db, rows)

            AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=1000.0,
            ).run(start_balance_jpy=10_000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            band = payload["target_band"]
            self.assertEqual(band["status"], "SELECTED_POLICY_REACHES_FLOOR_BELOW_STRETCH")
            self.assertEqual(band["selected_attainable_return_pct"], 5.0)
            self.assertEqual(band["selected_best_return_pct"], 5.5)
            self.assertEqual(band["all_positive_oracle_best_return_pct"], 5.5)
            by_pct = {item["return_pct"]: item for item in band["bands"]}
            self.assertEqual(by_pct[5.0]["target_jpy"], 500.0)
            self.assertEqual(by_pct[5.0]["selected_target_hit_days"], 1)
            self.assertEqual(by_pct[6.0]["selected_target_hit_days"], 0)
            self.assertEqual(by_pct[10.0]["required_trades_per_day_at_observed_expectancy"], 2)
            sizing = payload["mechanism_ablation"]["target_sizing"]
            self.assertEqual(sizing["status"], "FLOOR_ALREADY_HIT")
            sizing_by_pct = {item["return_pct"]: item for item in sizing["bands"]}
            self.assertEqual(sizing_by_pct[5.0]["status"], "ALREADY_HIT")
            self.assertEqual(sizing_by_pct[5.0]["scaled_target_hit_days"], 1)
            self.assertEqual(sizing_by_pct[5.0]["scaled_max_drawdown_jpy"], 0.0)
            self.assertAlmostEqual(sizing_by_pct[5.0]["scaled_worst_day_jpy"], 500.005)
            self.assertAlmostEqual(sizing_by_pct[10.0]["required_size_multiplier"], 1.8182)
            self.assertEqual(sizing_by_pct[10.0]["scaled_target_hit_days"], 1)
            self.assertEqual(sizing_by_pct[10.0]["scaled_max_drawdown_jpy"], 0.0)
            self.assertEqual(sizing_by_pct[10.0]["scaled_worst_day_jpy"], 1000.01)
            self.assertEqual(sizing_by_pct[10.0]["status"], "MATERIAL_SIZE_UP_REQUIRED")
            self.assertTrue(any("selected policy currently reaches 5%" in item for item in payload["action_items"]))
            report = (root / "ai_backtest.md").read_text()
            self.assertIn("## Target Band", report)
            self.assertIn("### Target-Aware Sizing Diagnostics", report)
            self.assertIn("Selected-policy best return: `5.50%`", report)
            self.assertIn("`5.0%` target=`500` selected_hits=`1/1`", report)
            self.assertIn("`10.0%` target=`1000` required_size_multiplier=`1.8182`", report)
            self.assertIn("scaled_target_hits=`1` scaled_max_dd=`0`", report)
            self.assertIn("`10.0%` target=`1000` selected_hits=`0/1`", report)

    def test_walk_forward_policy_does_not_select_validation_only_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-04-01", "EUR_USD", "LONG", 100.0),
                    ("trades", "2026-04-02", "EUR_USD", "LONG", 120.0),
                    ("trades", "2026-04-03", "EUR_USD", "LONG", 90.0),
                    ("trades", "2026-04-03", "GBP_USD", "LONG", 5000.0),
                ],
            )

            summary = AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=2,
                min_train_trades=2,
                max_active_buckets=1,
            ).run(start_balance_jpy=500.0)

            self.assertEqual(summary.status, "TARGET_COVERAGE_CERTIFIED")
            self.assertEqual(summary.validation_days, 1)
            self.assertEqual(summary.total_managed_net_jpy, 90.0)
            payload = json.loads((root / "ai_backtest.json").read_text())
            day = payload["days"][0]
            self.assertEqual(day["selected_buckets"], ["trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED"])
            self.assertEqual(day["selected_trades"], 1)
            self.assertEqual(day["managed_net_jpy"], 90.0)
            self.assertNotIn("GBP_USD", " ".join(day["selected_buckets"]))
            self.assertFalse(payload["live_permission"])
            self.assertEqual(payload["firepower"]["best_selected_day_jpy"], 90.0)
            contribution_buckets = {item["bucket"] for item in payload["bucket_contributions"]}
            self.assertIn("trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED", contribution_buckets)
            self.assertEqual(payload["oracle"]["top_n_target_hit_days"], 1)
            self.assertTrue(payload["target_ceiling"]["prediction_only_target_possible"])
            self.assertEqual(payload["action_items"], [])
            self.assertEqual(payload["missed_best_days"][0]["best_bucket"], "trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED")
            self.assertIn("offline research bot", (root / "ai_backtest.md").read_text())

    def test_default_policy_rejects_submajority_training_win_rate_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-04-01", "EUR_USD", "LONG", 300.0, json.dumps({"id": "bad-1"})),
                    ("trades", "2026-04-01", "EUR_USD", "LONG", -50.0, json.dumps({"id": "bad-2"})),
                    ("trades", "2026-04-01", "EUR_USD", "LONG", -50.0, json.dumps({"id": "bad-3"})),
                    ("trades", "2026-04-01", "GBP_USD", "LONG", 40.0, json.dumps({"id": "good-1"})),
                    ("trades", "2026-04-01", "GBP_USD", "LONG", 40.0, json.dumps({"id": "good-2"})),
                    ("trades", "2026-04-01", "GBP_USD", "LONG", 40.0, json.dumps({"id": "good-3"})),
                    ("trades", "2026-04-02", "EUR_USD", "LONG", -500.0, json.dumps({"id": "bad-val"})),
                    ("trades", "2026-04-02", "GBP_USD", "LONG", 80.0, json.dumps({"id": "good-val"})),
                ],
            )

            AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=3,
                max_active_buckets=1,
            ).run(start_balance_jpy=500.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            day = payload["days"][0]
            self.assertEqual(day["selected_buckets"], ["trades:GBP_USD:LONG:UNSPECIFIED:UNSPECIFIED"])
            self.assertEqual(day["managed_net_jpy"], 80.0)
            self.assertEqual(payload["min_train_win_rate_pct"], 55.0)
            self.assertIn("majority capped win rate", (root / "ai_backtest.md").read_text())

    def test_applies_equity_loss_cap_to_validation_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-04-01", "EUR_USD", "SHORT", 200.0),
                    ("trades", "2026-04-02", "EUR_USD", "SHORT", -400.0),
                ],
            )

            AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=75.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            day = payload["days"][0]
            self.assertEqual(day["raw_net_jpy"], -400.0)
            self.assertEqual(day["managed_net_jpy"], -75.0)
            self.assertEqual(payload["status"], "BLOCKED")
            self.assertTrue(any("out-of-sample managed net is not positive" in item for item in payload["blockers"]))
            self.assertEqual(payload["bucket_contributions"][0]["managed_net_jpy"], -75.0)

    def test_dedupes_seat_outcomes_by_overlapping_matched_trades_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            raw_train = json.dumps(
                {
                    "source": "s_hunt",
                    "setup_type": "MARKET",
                    "matched_trade_ids": "seat-1",
                    "updated_at": "2026-04-01 01:00:00",
                }
            )
            raw_train_later = json.dumps(
                {
                    "source": "s_hunt",
                    "setup_type": "MARKET",
                    "matched_trade_ids": "seat-1",
                    "updated_at": "2026-04-01 02:00:00",
                }
            )
            raw_train_overlap = json.dumps(
                {
                    "source": "s_hunt",
                    "setup_type": "MARKET",
                    "matched_trade_ids": "seat-1,seat-extra",
                    "updated_at": "2026-04-01 01:30:00",
                }
            )
            raw_validation = json.dumps(
                {
                    "source": "s_hunt",
                    "setup_type": "MARKET",
                    "matched_trade_ids": "seat-2",
                    "updated_at": "2026-04-02 01:00:00",
                }
            )
            raw_validation_overlap = json.dumps(
                {
                    "source": "s_hunt",
                    "setup_type": "MARKET",
                    "matched_trade_ids": "seat-2,seat-next",
                    "updated_at": "2026-04-02 02:00:00",
                }
            )
            _seed_db(
                db,
                [
                    ("seat_outcomes", "2026-04-01", "EUR_USD", "SHORT", 500.0, raw_train),
                    ("seat_outcomes", "2026-04-01", "EUR_USD", "SHORT", 500.0, raw_train_overlap),
                    ("seat_outcomes", "2026-04-01", "EUR_USD", "SHORT", 500.0, raw_train_later),
                    ("seat_outcomes", "2026-04-02", "EUR_USD", "SHORT", 700.0, raw_validation),
                    ("seat_outcomes", "2026-04-02", "EUR_USD", "SHORT", 700.0, raw_validation_overlap),
                ],
            )

            AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("seat_outcomes",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            self.assertEqual(payload["raw_rows"], 5)
            self.assertEqual(payload["deduped_rows"], 2)
            self.assertEqual(payload["deduped_away_rows"], 3)
            day = payload["days"][0]
            self.assertEqual(day["selected_trades"], 1)
            self.assertEqual(day["managed_net_jpy"], 700.0)
            self.assertEqual(day["selected_buckets"], ["seat_outcomes:EUR_USD:SHORT:MARKET:S_HUNT"])

    def test_dedupes_same_trade_id_but_keeps_pretrade_bucket_as_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            _seed_db(
                db,
                [
                    (
                        "pretrade_outcomes",
                        "2026-04-01",
                        "EUR_USD",
                        "LONG",
                        100.0,
                        json.dumps({"trade_id": "train-1", "pretrade_level": "HIGH", "created_at": "2026-04-01 01:00:00"}),
                    ),
                    (
                        "trades",
                        "2026-04-01",
                        "EUR_USD",
                        "LONG",
                        100.0,
                        json.dumps({"trade_id": "train-1", "created_at": "2026-04-20 01:00:00"}),
                    ),
                    (
                        "pretrade_outcomes",
                        "2026-04-01",
                        "EUR_USD",
                        "LONG",
                        120.0,
                        json.dumps({"trade_id": "train-2", "pretrade_level": "HIGH", "created_at": "2026-04-01 02:00:00"}),
                    ),
                    (
                        "trades",
                        "2026-04-01",
                        "EUR_USD",
                        "LONG",
                        120.0,
                        json.dumps({"trade_id": "train-2", "created_at": "2026-04-20 02:00:00"}),
                    ),
                    (
                        "pretrade_outcomes",
                        "2026-04-02",
                        "EUR_USD",
                        "LONG",
                        150.0,
                        json.dumps({"trade_id": "validation-1", "pretrade_level": "HIGH", "created_at": "2026-04-02 01:00:00"}),
                    ),
                    (
                        "trades",
                        "2026-04-02",
                        "EUR_USD",
                        "LONG",
                        150.0,
                        json.dumps({"trade_id": "validation-1", "created_at": "2026-04-20 03:00:00"}),
                    ),
                ],
            )

            AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=1000.0,
                training_days=1,
                min_train_trades=2,
                max_active_buckets=6,
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            self.assertEqual(payload["raw_rows"], 6)
            self.assertEqual(payload["deduped_rows"], 3)
            self.assertEqual(payload["deduped_away_rows"], 3)
            by_source = {item["source_table"]: item for item in payload["source_contributions"]}
            self.assertEqual(by_source["trades"]["deduped_trades"], 3)
            self.assertNotIn("pretrade_outcomes", by_source)
            day = payload["days"][0]
            self.assertEqual(day["selected_buckets"], ["trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED"])
            self.assertEqual(day["selected_trades"], 1)
            self.assertEqual(day["managed_net_jpy"], 150.0)
            self.assertNotIn(
                "pretrade_outcomes:EUR_USD:LONG:HIGH:UNSPECIFIED",
                {item["bucket"] for item in payload["bucket_contributions"]},
            )
            bucket_labels = {item["bucket"] for item in payload["evidence_bucket_contributions"]}
            self.assertIn("pretrade_outcomes:EUR_USD:LONG:HIGH:UNSPECIFIED", bucket_labels)

    def test_evidence_alias_does_not_become_selection_bucket(self) -> None:
        pretrade_bucket = TestBotBucket(
            source_table="pretrade_outcomes",
            pair="EUR_USD",
            direction="LONG",
            execution_style="HIGH",
            allocation_band="UNSPECIFIED",
        )
        context_bucket = TestBotBucket(
            source_table="trades_theme",
            pair="EUROPE_FX_USD_WEAK",
            direction="LONG",
            execution_style="FX_RISK_THEME",
            allocation_band="ALL",
        )
        row = TestBotTrade(
            source_id="trade-1",
            session_date="2026-04-02",
            source_table="trades",
            pair="EUR_USD",
            direction="LONG",
            execution_style="UNSPECIFIED",
            allocation_band="UNSPECIFIED",
            pl_jpy=150.0,
            opportunity_key="trade-1",
            sort_key="trade-1",
            extra_buckets=(pretrade_bucket, context_bucket),
        )

        self.assertTrue(_row_matches_selected_buckets(row, {row.bucket}))
        self.assertTrue(_row_matches_selected_buckets(row, {context_bucket}))
        self.assertFalse(_row_matches_selected_buckets(row, {pretrade_bucket}))

    def test_execution_ledger_source_uses_original_position_side_without_exit_reason_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            _seed_execution_ledger(ledger)

            AITestBotBacktester(
                db_path=root / "missing_legacy.db",
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("execution_ledger",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            self.assertEqual(payload["source_tables"], ["execution_ledger"])
            self.assertEqual(payload["raw_rows"], 2)
            self.assertEqual(payload["source_contributions"][0]["source_table"], "execution_ledger")
            day = payload["days"][0]
            self.assertEqual(day["selected_buckets"], ["execution_ledger:GBP_USD:LONG:TREND_TRADER:TREND_CONTINUATION"])
            self.assertEqual(day["selected_trades"], 1)
            self.assertEqual(day["managed_net_jpy"], 150.0)
            self.assertNotIn("GBP_USD:SHORT", json.dumps(payload))
            self.assertNotIn("TAKE_PROFIT_ORDER", " ".join(day["selected_buckets"]))
            report = (root / "ai_backtest.md").read_text()
            self.assertIn("sent gateway entry", report)
            self.assertIn("exit reason remains post-trade evidence", report)

    def test_execution_ledger_source_uses_fill_side_when_units_are_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            _seed_execution_ledger(
                ledger,
                [
                    (
                        "fill-train",
                        "2026-06-01T01:00:00Z",
                        "ORDER_FILLED",
                        "short-train",
                        "EUR_USD",
                        "SHORT",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-train",
                        "2026-06-01T03:00:00Z",
                        "TRADE_CLOSED",
                        "short-train",
                        "EUR_USD",
                        "LONG",
                        1000,
                        100.0,
                        "{}",
                    ),
                    (
                        "fill-validation",
                        "2026-06-02T01:00:00Z",
                        "ORDER_FILLED",
                        "short-validation",
                        "EUR_USD",
                        "SHORT",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-validation",
                        "2026-06-02T03:00:00Z",
                        "TRADE_CLOSED",
                        "short-validation",
                        "EUR_USD",
                        "LONG",
                        1000,
                        150.0,
                        "{}",
                    ),
                ],
            )

            AITestBotBacktester(
                db_path=root / "missing_legacy.db",
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("execution_ledger",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            day = payload["days"][0]
            self.assertEqual(day["selected_buckets"], ["execution_ledger:EUR_USD:SHORT:TREND_TRADER:TREND_CONTINUATION"])
            self.assertNotIn("EUR_USD:LONG", json.dumps(payload))

    def test_execution_ledger_source_ignores_unattributed_manual_closes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            _seed_execution_ledger(
                ledger,
                [
                    (
                        "fill-bot-train",
                        "2026-06-01T01:00:00Z",
                        "ORDER_FILLED",
                        "bot-train",
                        "GBP_USD",
                        "LONG",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-bot-train",
                        "2026-06-01T03:00:00Z",
                        "TRADE_CLOSED",
                        "bot-train",
                        "GBP_USD",
                        "SHORT",
                        1000,
                        100.0,
                        "{}",
                    ),
                    (
                        "fill-manual-train",
                        "2026-06-01T04:00:00Z",
                        "ORDER_FILLED",
                        "manual-train",
                        "USD_JPY",
                        "LONG",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-manual-train",
                        "2026-06-01T05:00:00Z",
                        "TRADE_CLOSED",
                        "manual-train",
                        "USD_JPY",
                        "SHORT",
                        1000,
                        5000.0,
                        "{}",
                    ),
                    (
                        "fill-bot-validation",
                        "2026-06-02T01:00:00Z",
                        "ORDER_FILLED",
                        "bot-validation",
                        "GBP_USD",
                        "LONG",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-bot-validation",
                        "2026-06-02T03:00:00Z",
                        "TRADE_CLOSED",
                        "bot-validation",
                        "GBP_USD",
                        "SHORT",
                        1000,
                        150.0,
                        "{}",
                    ),
                    (
                        "fill-manual-validation",
                        "2026-06-02T04:00:00Z",
                        "ORDER_FILLED",
                        "manual-validation",
                        "USD_JPY",
                        "LONG",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-manual-validation",
                        "2026-06-02T05:00:00Z",
                        "TRADE_CLOSED",
                        "manual-validation",
                        "USD_JPY",
                        "SHORT",
                        1000,
                        6000.0,
                        "{}",
                    ),
                ],
                attributed_trade_ids={"bot-train", "bot-validation"},
            )

            AITestBotBacktester(
                db_path=root / "missing_legacy.db",
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("execution_ledger",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            self.assertEqual(payload["raw_rows"], 2)
            self.assertEqual(payload["days"][0]["selected_buckets"], ["execution_ledger:GBP_USD:LONG:TREND_TRADER:TREND_CONTINUATION"])
            self.assertNotIn("USD_JPY", json.dumps(payload))

    def test_execution_ledger_source_rejects_raw_negative_training_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "execution_ledger.db"
            _seed_execution_ledger(
                ledger,
                [
                    (
                        "fill-loss",
                        "2026-06-01T01:00:00Z",
                        "ORDER_FILLED",
                        "trade-loss",
                        "EUR_USD",
                        "LONG",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-loss",
                        "2026-06-01T03:00:00Z",
                        "TRADE_CLOSED",
                        "trade-loss",
                        "EUR_USD",
                        "SHORT",
                        1000,
                        -500.0,
                        "{}",
                    ),
                    (
                        "fill-win",
                        "2026-06-01T04:00:00Z",
                        "ORDER_FILLED",
                        "trade-win",
                        "EUR_USD",
                        "LONG",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-win",
                        "2026-06-01T06:00:00Z",
                        "TRADE_CLOSED",
                        "trade-win",
                        "EUR_USD",
                        "SHORT",
                        1000,
                        200.0,
                        "{}",
                    ),
                    (
                        "fill-validation",
                        "2026-06-02T01:00:00Z",
                        "ORDER_FILLED",
                        "trade-validation",
                        "EUR_USD",
                        "LONG",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-validation",
                        "2026-06-02T03:00:00Z",
                        "TRADE_CLOSED",
                        "trade-validation",
                        "EUR_USD",
                        "SHORT",
                        1000,
                        700.0,
                        "{}",
                    ),
                ],
            )

            AITestBotBacktester(
                db_path=root / "missing_legacy.db",
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=2,
                max_active_buckets=1,
                source_tables=("execution_ledger",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            day = payload["days"][0]
            self.assertEqual(day["selected_buckets"], [])
            self.assertEqual(day["selected_trades"], 0)
            self.assertEqual(day["managed_net_jpy"], 0.0)
            self.assertEqual(payload["oracle"]["best_train_eligible_all_positive_day_jpy"], 0.0)
            self.assertIn("raw-positive training net", (root / "ai_backtest.md").read_text())

    def test_mixed_runtime_keeps_execution_ledger_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            ledger = root / "execution_ledger.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-06-01", "EUR_USD", "LONG", 100.0),
                    ("trades", "2026-06-02", "EUR_USD", "LONG", 120.0),
                ],
            )
            _seed_execution_ledger(
                ledger,
                [
                    (
                        "fill-ledger-train",
                        "2026-06-01T01:00:00Z",
                        "ORDER_FILLED",
                        "ledger-train",
                        "GBP_USD",
                        "LONG",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-ledger-train",
                        "2026-06-01T03:00:00Z",
                        "TRADE_CLOSED",
                        "ledger-train",
                        "GBP_USD",
                        "SHORT",
                        1000,
                        500.0,
                        "{}",
                    ),
                    (
                        "fill-ledger-validation",
                        "2026-06-02T01:00:00Z",
                        "ORDER_FILLED",
                        "ledger-validation",
                        "GBP_USD",
                        "LONG",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-ledger-validation",
                        "2026-06-02T03:00:00Z",
                        "TRADE_CLOSED",
                        "ledger-validation",
                        "GBP_USD",
                        "SHORT",
                        1000,
                        -700.0,
                        "{}",
                    ),
                ],
            )

            AITestBotBacktester(
                db_path=db,
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("trades", "execution_ledger"),
            ).run(start_balance_jpy=10_000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            self.assertEqual(payload["execution_ledger_selection"], "diagnostic_only_mixed_sources")
            self.assertEqual(payload["promotable_source_tables"], ["trades"])
            day = payload["days"][0]
            self.assertEqual(day["selected_buckets"], ["trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED"])
            self.assertEqual(day["selected_trades"], 1)
            self.assertEqual(day["managed_net_jpy"], 120.0)
            by_source = {item["source_table"]: item for item in payload["source_contributions"]}
            self.assertEqual(by_source["execution_ledger"]["validation_universe_trades"], 1)
            self.assertEqual(by_source["execution_ledger"]["selected_trades"], 0)
            self.assertTrue(any("diagnostic-only" in item for item in payload["action_items"]))

    def test_mechanism_ablation_reports_loss_cap_and_close_gate_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            ledger = root / "execution_ledger.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-06-01", "EUR_USD", "LONG", 100.0),
                    ("trades", "2026-06-02", "EUR_USD", "LONG", -500.0),
                ],
            )
            _seed_execution_ledger(
                ledger,
                [
                    (
                        "fill-market-loss",
                        "2026-06-01T01:00:00Z",
                        "ORDER_FILLED",
                        "market-loss",
                        "EUR_USD",
                        "LONG",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-market-loss",
                        "2026-06-01T02:00:00Z",
                        "TRADE_CLOSED",
                        "market-loss",
                        "EUR_USD",
                        "SHORT",
                        1000,
                        -300.0,
                        json.dumps({"reason": "MARKET_ORDER_TRADE_CLOSE"}),
                    ),
                    (
                        "fill-tp",
                        "2026-06-01T03:00:00Z",
                        "ORDER_FILLED",
                        "tp-win",
                        "GBP_USD",
                        "LONG",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-tp",
                        "2026-06-01T04:00:00Z",
                        "TRADE_CLOSED",
                        "tp-win",
                        "GBP_USD",
                        "SHORT",
                        1000,
                        220.0,
                        json.dumps({"reason": "TAKE_PROFIT_ORDER"}),
                    ),
                    (
                        "fill-gated-loss",
                        "2026-06-01T05:00:00Z",
                        "ORDER_FILLED",
                        "gated-loss",
                        "AUD_JPY",
                        "LONG",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "gateway-close-gated-loss",
                        "2026-06-01T05:30:00Z",
                        "GATEWAY_TRADE_CLOSE_SENT",
                        "gated-loss",
                        "AUD_JPY",
                        "LONG",
                        None,
                        None,
                        json.dumps({"management_action": "REVIEW_EXIT"}),
                    ),
                    (
                        "close-gated-loss",
                        "2026-06-01T06:00:00Z",
                        "TRADE_CLOSED",
                        "gated-loss",
                        "AUD_JPY",
                        "SHORT",
                        1000,
                        -180.0,
                        json.dumps({"reason": "MARKET_ORDER_TRADE_CLOSE"}),
                    ),
                ],
            )

            AITestBotBacktester(
                db_path=db,
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("trades",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            risk_cap = payload["mechanism_ablation"]["risk_engine_loss_cap"]
            self.assertEqual(risk_cap["raw_selected_net_jpy"], -500.0)
            self.assertEqual(risk_cap["managed_selected_net_jpy"], -100.0)
            self.assertEqual(risk_cap["managed_net_minus_raw_net_jpy"], 400.0)
            close_gate = payload["mechanism_ablation"]["close_gate_ab"]
            self.assertEqual(close_gate["status"], "MEASURED")
            self.assertEqual(close_gate["close_events"], 3)
            self.assertEqual(close_gate["bot_attributed_close_events"], 3)
            self.assertEqual(close_gate["loss_side_market_close_count"], 2)
            self.assertEqual(close_gate["loss_side_market_close_net_jpy"], -480.0)
            self.assertEqual(close_gate["gateway_loss_side_market_close_count"], 1)
            self.assertEqual(close_gate["gateway_gpt_close_loss_side_market_close_count"], 0)
            self.assertEqual(close_gate["gateway_review_exit_loss_side_market_close_count"], 1)
            self.assertEqual(close_gate["unattributed_loss_side_market_close_count"], 1)
            self.assertEqual(close_gate["take_profit_close_net_jpy"], 220.0)
            source_segments = {item["source"]: item for item in close_gate["close_source_segments"]}
            self.assertEqual(source_segments["NO_CLOSE_ORDER_PROVENANCE"]["net_jpy"], -300.0)
            self.assertEqual(source_segments["GATEWAY:REVIEW_EXIT"]["net_jpy"], -180.0)
            self.assertEqual(source_segments["BROKER:TAKE_PROFIT_ORDER"]["profit_factor"], None)
            self.assertEqual(
                close_gate["loss_side_market_close_daily"],
                [
                    {
                        "day": "2026-06-01",
                        "count": 2,
                        "net_jpy": -480.0,
                        "gateway_close_sent_count": 1,
                        "gateway_close_reconciled_count": 0,
                        "gateway_gpt_close_count": 0,
                        "gateway_gpt_close_accepted_count": 0,
                        "gateway_review_exit_count": 1,
                        "gateway_other_close_count": 0,
                        "stale_gpt_close_satisfied_count": 0,
                        "broker_trade_close_accepted_count": 0,
                        "broker_accepted_without_gateway_count": 0,
                        "broker_accepted_without_gateway_source_counts": {},
                        "broker_accepted_without_gateway_evidence_counts": {},
                        "no_close_order_provenance_count": 1,
                        "bot_attributed_count": 2,
                    }
                ],
            )
            self.assertEqual(close_gate["loss_side_market_close_examples"][0]["trade_id"], "market-loss")
            self.assertEqual(close_gate["loss_side_market_close_examples"][0]["pl_jpy"], -300.0)
            self.assertFalse(close_gate["loss_side_market_close_examples"][0]["gateway_close_sent"])
            self.assertEqual(close_gate["loss_side_market_close_examples"][1]["gateway_close_reasons"], ["REVIEW_EXIT"])
            self.assertTrue(any("RiskEngine loss-cap ablation" in item for item in payload["action_items"]))
            self.assertTrue(any("legacy gateway REVIEW_EXIT" in item for item in payload["action_items"]))
            report = (root / "ai_backtest.md").read_text()
            self.assertIn("## Mechanism Ablations", report)
            self.assertIn("CLOSE Gate A/B Diagnostics", report)
            self.assertIn("Worst loss-side market close examples", report)

    def test_close_gate_diagnostic_reports_direct_or_manual_broker_closes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            ledger = root / "execution_ledger.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-06-01", "EUR_USD", "LONG", 100.0),
                    ("trades", "2026-06-02", "EUR_USD", "LONG", -200.0),
                ],
            )
            _seed_execution_ledger(
                ledger,
                [
                    (
                        "fill-unattributed",
                        "2026-06-01T01:00:00Z",
                        "ORDER_FILLED",
                        "manual-or-missing-receipt",
                        "EUR_USD",
                        "LONG",
                        1000,
                        None,
                        "{}",
                    ),
                    (
                        "close-unattributed",
                        "2026-06-01T02:00:00Z",
                        "TRADE_CLOSED",
                        "manual-or-missing-receipt",
                        "EUR_USD",
                        "SHORT",
                        1000,
                        -250.0,
                        json.dumps({"reason": "MARKET_ORDER_TRADE_CLOSE"}),
                    ),
                ],
                attributed_trade_ids=set(),
            )

            AITestBotBacktester(
                db_path=db,
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("trades",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            close_gate = payload["mechanism_ablation"]["close_gate_ab"]
            self.assertEqual(close_gate["status"], "MEASURED")
            self.assertEqual(close_gate["close_events"], 1)
            self.assertEqual(close_gate["bot_attributed_close_events"], 0)
            self.assertEqual(close_gate["unattributed_loss_side_market_close_count"], 1)
            self.assertTrue(any("zero gateway-attributed entry closes" in item for item in payload["action_items"]))

    def test_close_gate_diagnostic_attributes_broker_client_extension_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            ledger = root / "execution_ledger.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-06-01", "EUR_USD", "LONG", 100.0),
                    ("trades", "2026-06-02", "EUR_USD", "LONG", -200.0),
                ],
            )
            _seed_broker_attributed_ledger(ledger)

            AITestBotBacktester(
                db_path=db,
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("trades",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            close_gate = payload["mechanism_ablation"]["close_gate_ab"]
            self.assertEqual(close_gate["status"], "MEASURED")
            self.assertEqual(close_gate["close_events"], 1)
            self.assertEqual(close_gate["bot_attributed_close_events"], 1)
            self.assertEqual(close_gate["bot_attributed_loss_side_market_close_count"], 1)
            self.assertEqual(close_gate["unattributed_loss_side_market_close_count"], 1)
            self.assertFalse(any("zero gateway-attributed entry closes" in item for item in payload["action_items"]))

    def test_close_gate_diagnostic_counts_broker_accepted_trade_close_without_gateway_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            ledger = root / "execution_ledger.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-06-01", "EUR_USD", "LONG", 100.0),
                    ("trades", "2026-06-02", "EUR_USD", "LONG", -200.0),
                ],
            )
            _seed_broker_accepted_close_ledger(ledger)

            AITestBotBacktester(
                db_path=db,
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("trades",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            close_gate = payload["mechanism_ablation"]["close_gate_ab"]
            self.assertEqual(close_gate["status"], "MEASURED")
            self.assertEqual(close_gate["close_events"], 1)
            self.assertEqual(close_gate["bot_attributed_close_events"], 1)
            self.assertEqual(close_gate["gateway_close_sent_events"], 0)
            self.assertEqual(close_gate["broker_trade_close_accept_events"], 1)
            self.assertEqual(close_gate["broker_trade_close_accept_trade_ids"], 1)
            self.assertEqual(close_gate["broker_trade_close_accept_order_ids"], 1)
            self.assertEqual(
                close_gate["broker_trade_close_accept_source_counts"],
                {"TRADER_ENTRY_LANE_ID": 1},
            )
            self.assertEqual(close_gate["loss_side_market_close_count"], 1)
            self.assertEqual(close_gate["broker_trade_close_loss_side_market_close_count"], 1)
            self.assertEqual(
                close_gate["broker_trade_close_loss_side_market_close_source_counts"],
                {"TRADER_ENTRY_LANE_ID": 1},
            )
            self.assertEqual(close_gate["broker_accepted_without_gateway_loss_side_market_close_count"], 1)
            self.assertEqual(
                close_gate["broker_accepted_without_gateway_loss_side_market_close_source_counts"],
                {"TRADER_ENTRY_LANE_ID": 1},
            )
            self.assertEqual(
                close_gate["broker_accepted_without_gateway_loss_side_market_close_evidence_counts"],
                {"NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 1, "TRADER_ENTRY_LANE_ID": 1},
            )
            self.assertEqual(close_gate["unattributed_loss_side_market_close_count"], 0)
            example = close_gate["loss_side_market_close_examples"][0]
            self.assertFalse(example["gateway_close_sent"])
            self.assertTrue(example["broker_trade_close_accepted"])
            self.assertEqual(example["broker_trade_close_sources"], ["TRADER_ENTRY_LANE_ID"])
            self.assertEqual(
                example["broker_trade_close_evidence"],
                ["NO_LOCAL_GATEWAY_CLOSE_RECEIPT", "TRADER_ENTRY_LANE_ID"],
            )
            self.assertTrue(example["close_order_provenance"])
            self.assertTrue(any("broker accepted TRADE_CLOSE orders exist" in item for item in payload["action_items"]))
            self.assertTrue(any("NO_LOCAL_GATEWAY_CLOSE_RECEIPT" in item for item in payload["action_items"]))
            self.assertTrue(any("TRADER_ENTRY_LANE_ID" in item for item in payload["action_items"]))
            self.assertFalse(any("direct/manual broker TRADE_CLOSE" in item for item in payload["action_items"]))
            self.assertTrue(any("worst close-source segment" in item for item in payload["action_items"]))
            self.assertFalse(any("lack both gateway close receipts" in item for item in payload["action_items"]))
            report = (root / "ai_backtest.md").read_text()
            self.assertIn("TRADER_ENTRY_LANE_ID", report)
            self.assertIn("NO_LOCAL_GATEWAY_CLOSE_RECEIPT", report)
            self.assertIn("Close source segments", report)

    def test_close_gate_diagnostic_uses_entry_lane_for_broker_accepted_close_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            ledger = root / "execution_ledger.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-06-01", "EUR_USD", "LONG", 100.0),
                    ("trades", "2026-06-02", "EUR_USD", "LONG", -200.0),
                ],
            )
            _seed_broker_accepted_close_ledger(ledger)
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    UPDATE execution_events
                    SET lane_id = ?
                    WHERE event_uid = 'fill-entry'
                    """,
                    ("trend_trader:EUR_USD:LONG:TREND_CONTINUATION",),
                )

            AITestBotBacktester(
                db_path=db,
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("trades",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            close_gate = payload["mechanism_ablation"]["close_gate_ab"]
            self.assertEqual(
                close_gate["broker_trade_close_accept_source_counts"],
                {"TRADER_ENTRY_LANE_ID": 1},
            )
            self.assertEqual(
                close_gate["broker_accepted_without_gateway_loss_side_market_close_source_counts"],
                {"TRADER_ENTRY_LANE_ID": 1},
            )
            self.assertEqual(
                close_gate["broker_accepted_without_gateway_loss_side_market_close_evidence_counts"],
                {"NO_LOCAL_GATEWAY_CLOSE_RECEIPT": 1, "TRADER_ENTRY_LANE_ID": 1},
            )
            example = close_gate["loss_side_market_close_examples"][0]
            self.assertEqual(example["broker_trade_close_sources"], ["TRADER_ENTRY_LANE_ID"])
            self.assertEqual(
                example["broker_trade_close_evidence"],
                ["NO_LOCAL_GATEWAY_CLOSE_RECEIPT", "TRADER_ENTRY_LANE_ID"],
            )
            source_segments = {item["source"]: item for item in close_gate["close_source_segments"]}
            self.assertIn("BROKER_ACCEPT:TRADER_ENTRY_LANE_ID", source_segments)
            self.assertNotIn("BROKER_ACCEPT:DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE", source_segments)

    def test_close_gate_diagnostic_classifies_gpt_accepted_close_without_position_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            ledger = root / "execution_ledger.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-06-01", "EUR_USD", "LONG", 100.0),
                    ("trades", "2026-06-02", "EUR_USD", "LONG", -200.0),
                ],
            )
            _seed_broker_accepted_close_ledger(ledger)
            with sqlite3.connect(ledger) as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events (
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id, pair, side, units,
                        realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "gpt-close-accepted",
                        "2026-06-01T01:59:59Z",
                        "GATEWAY_GPT_CLOSE_ACCEPTED",
                        None,
                        None,
                        "broker-close-trade",
                        "EUR_USD",
                        None,
                        None,
                        None,
                        "GPT_CLOSE_ACCEPTED",
                        json.dumps(
                            {
                                "status": "ACCEPTED",
                                "decision": {
                                    "action": "CLOSE",
                                    "close_trade_ids": ["broker-close-trade"],
                                },
                            }
                        ),
                    ),
                )

            AITestBotBacktester(
                db_path=db,
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("trades",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            close_gate = payload["mechanism_ablation"]["close_gate_ab"]
            self.assertEqual(close_gate["gateway_close_sent_events"], 0)
            self.assertEqual(close_gate["gateway_gpt_close_accepted_events"], 1)
            self.assertEqual(
                close_gate["gateway_gpt_close_accepted_without_sent_loss_side_market_close_count"],
                1,
            )
            self.assertEqual(
                close_gate["broker_accepted_without_gateway_loss_side_market_close_source_counts"],
                {"GATEWAY_GPT_CLOSE_ACCEPTED": 1, "TRADER_ENTRY_LANE_ID": 1},
            )
            example = close_gate["loss_side_market_close_examples"][0]
            self.assertTrue(example["gateway_gpt_close_accepted"])
            self.assertEqual(example["close_source"], "GATEWAY:GPT_CLOSE_ACCEPTED_NO_POSITION_RECEIPT")
            self.assertEqual(example["broker_trade_close_sources"], ["GATEWAY_GPT_CLOSE_ACCEPTED", "TRADER_ENTRY_LANE_ID"])
            source_segments = {item["source"]: item for item in close_gate["close_source_segments"]}
            self.assertEqual(
                source_segments["GATEWAY:GPT_CLOSE_ACCEPTED_NO_POSITION_RECEIPT"][
                    "loss_side_market_close_count"
                ],
                1,
            )
            self.assertTrue(any("GPT CLOSE accepted receipts exist" in item for item in payload["action_items"]))
            self.assertFalse(any("direct/manual broker TRADE_CLOSE" in item for item in payload["action_items"]))

    def test_close_gate_diagnostic_counts_reconciled_gpt_close_as_gateway_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            ledger = root / "execution_ledger.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-06-01", "EUR_USD", "LONG", 100.0),
                    ("trades", "2026-06-02", "EUR_USD", "LONG", -200.0),
                ],
            )
            _seed_broker_accepted_close_ledger(ledger)
            with sqlite3.connect(ledger) as conn:
                conn.executemany(
                    """
                    INSERT INTO execution_events (
                        event_uid, ts_utc, event_type, lane_id, order_id, trade_id, pair, side, units,
                        realized_pl_jpy, exit_reason, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "gpt-close-accepted",
                            "2026-06-01T01:59:59Z",
                            "GATEWAY_GPT_CLOSE_ACCEPTED",
                            None,
                            None,
                            "broker-close-trade",
                            "EUR_USD",
                            None,
                            None,
                            None,
                            "GPT_CLOSE_ACCEPTED",
                            json.dumps(
                                {
                                    "status": "ACCEPTED",
                                    "decision": {
                                        "action": "CLOSE",
                                        "close_trade_ids": ["broker-close-trade"],
                                    },
                                }
                            ),
                        ),
                        (
                            "gpt-close-reconciled",
                            "2026-06-01T02:00:01Z",
                            "GATEWAY_TRADE_CLOSE_RECONCILED",
                            None,
                            "order-close",
                            "broker-close-trade",
                            "EUR_USD",
                            None,
                            None,
                            None,
                            "GPT_CLOSE_RECONCILED",
                            json.dumps({"reconciled_from": ["GATEWAY_GPT_CLOSE_ACCEPTED"]}),
                        ),
                    ],
                )

            AITestBotBacktester(
                db_path=db,
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("trades",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            close_gate = payload["mechanism_ablation"]["close_gate_ab"]
            self.assertEqual(close_gate["gateway_close_sent_events"], 0)
            self.assertEqual(close_gate["gateway_close_reconciled_events"], 1)
            self.assertEqual(close_gate["gateway_loss_side_market_close_count"], 1)
            self.assertEqual(close_gate["gateway_gpt_close_loss_side_market_close_count"], 1)
            self.assertEqual(
                close_gate["gateway_gpt_close_accepted_without_sent_loss_side_market_close_count"],
                0,
            )
            self.assertEqual(close_gate["broker_accepted_without_gateway_loss_side_market_close_count"], 0)
            example = close_gate["loss_side_market_close_examples"][0]
            self.assertFalse(example["gateway_close_sent"])
            self.assertTrue(example["gateway_close_reconciled"])
            self.assertEqual(example["gateway_close_reasons"], ["GPT_CLOSE_RECONCILED"])
            self.assertEqual(example["close_source"], "GATEWAY:GPT_CLOSE_RECONCILED")
            source_segments = {item["source"]: item for item in close_gate["close_source_segments"]}
            self.assertEqual(source_segments["GATEWAY:GPT_CLOSE_RECONCILED"]["loss_side_market_close_count"], 1)
            self.assertFalse(any("GPT CLOSE accepted receipts exist" in item for item in payload["action_items"]))
            self.assertFalse(
                any(
                    "broker accepted TRADE_CLOSE orders exist without matching" in item
                    for item in payload["action_items"]
                )
            )

    def test_close_gate_diagnostic_matches_gateway_close_by_order_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            ledger = root / "execution_ledger.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-06-01", "EUR_USD", "LONG", 100.0),
                    ("trades", "2026-06-02", "EUR_USD", "LONG", -200.0),
                ],
            )
            _seed_gateway_close_order_id_ledger(ledger)

            AITestBotBacktester(
                db_path=db,
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("trades",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            close_gate = payload["mechanism_ablation"]["close_gate_ab"]
            self.assertEqual(close_gate["status"], "MEASURED")
            self.assertEqual(close_gate["gateway_close_sent_events"], 1)
            self.assertEqual(close_gate["gateway_loss_side_market_close_count"], 1)
            self.assertEqual(close_gate["gateway_review_exit_loss_side_market_close_count"], 1)
            self.assertEqual(close_gate["broker_accepted_without_gateway_loss_side_market_close_count"], 0)
            self.assertEqual(close_gate["unattributed_loss_side_market_close_count"], 0)
            example = close_gate["loss_side_market_close_examples"][0]
            self.assertTrue(example["gateway_close_sent"])
            self.assertEqual(example["gateway_close_reasons"], ["REVIEW_EXIT"])
            self.assertEqual(example["close_source"], "GATEWAY:REVIEW_EXIT")
            source_segments = {item["source"]: item for item in close_gate["close_source_segments"]}
            self.assertEqual(source_segments["GATEWAY:REVIEW_EXIT"]["count"], 1)
            self.assertEqual(source_segments["GATEWAY:REVIEW_EXIT"]["net_jpy"], -250.0)
            self.assertEqual(source_segments["GATEWAY:REVIEW_EXIT"]["expectancy_jpy"], -250.0)

    def test_close_gate_diagnostic_classifies_stale_gpt_close_satisfied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            ledger = root / "execution_ledger.db"
            _seed_db(
                db,
                [
                    ("trades", "2026-06-01", "EUR_USD", "LONG", 100.0),
                    ("trades", "2026-06-02", "EUR_USD", "LONG", -200.0),
                ],
            )
            _seed_stale_gpt_close_satisfied_ledger(ledger)

            AITestBotBacktester(
                db_path=db,
                execution_ledger_db_path=ledger,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("trades",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            close_gate = payload["mechanism_ablation"]["close_gate_ab"]
            self.assertEqual(close_gate["status"], "MEASURED")
            self.assertEqual(close_gate["gateway_close_sent_events"], 0)
            self.assertEqual(close_gate["stale_gpt_close_satisfied_loss_side_market_close_count"], 1)
            self.assertEqual(close_gate["stale_gpt_close_satisfied_loss_side_market_close_net_jpy"], -250.0)
            self.assertEqual(close_gate["broker_accepted_without_gateway_loss_side_market_close_count"], 1)
            example = close_gate["loss_side_market_close_examples"][0]
            self.assertFalse(example["gateway_close_sent"])
            self.assertTrue(example["stale_gpt_close_satisfied"])
            self.assertEqual(example["close_source"], "GATEWAY:STALE_GPT_CLOSE_SATISFIED")
            source_segments = {item["source"]: item for item in close_gate["close_source_segments"]}
            self.assertEqual(source_segments["GATEWAY:STALE_GPT_CLOSE_SATISFIED"]["net_jpy"], -250.0)
            self.assertNotIn("BROKER_ACCEPT:DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE", source_segments)
            self.assertTrue(any("stale GPT close receipts" in item for item in payload["action_items"]))
            report = (root / "ai_backtest.md").read_text()
            self.assertIn("Stale GPT_CLOSE satisfied", report)

    def test_context_theme_overlay_adds_cross_pair_without_displacing_base_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            rows = [
                (
                    "trades",
                    "2026-04-01",
                    "EUR_JPY",
                    "LONG",
                    10.0,
                    json.dumps({"id": f"risk-train-{index}"}),
                )
                for index in range(20)
            ]
            rows.extend(
                [
                    (
                        "trades",
                        "2026-04-02",
                        "EUR_JPY",
                        "LONG",
                        50.0,
                        json.dumps({"id": "base-validation"}),
                    ),
                    (
                        "trades",
                        "2026-04-02",
                        "GBP_JPY",
                        "LONG",
                        70.0,
                        json.dumps({"id": "theme-validation"}),
                    ),
                ]
            )
            _seed_db(db, rows)

            AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("trades",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            day = payload["days"][0]
            self.assertEqual(
                day["selected_buckets"],
                [
                    "trades:EUR_JPY:LONG:UNSPECIFIED:UNSPECIFIED",
                    "trades_theme:RISK_ON_JPY_CROSS:LONG:FX_RISK_THEME:ALL",
                ],
            )
            self.assertEqual(day["selected_trades"], 2)
            self.assertEqual(day["managed_net_jpy"], 120.0)
            self.assertEqual(payload["context_theme_policy"]["max_active_buckets"], 1)
            self.assertEqual(payload["context_theme_policy"]["min_train_win_rate_pct"], 65.0)
            self.assertEqual(payload["context_feature_coverage"]["rows_with_context_theme_buckets"], 22)
            self.assertIn("Cross-pair context theme buckets", (root / "ai_backtest.md").read_text())

    def test_context_theme_overlay_requires_stronger_win_rate_than_pair_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "legacy.db"
            rows = [
                (
                    "trades",
                    "2026-04-01",
                    "EUR_JPY",
                    "LONG",
                    10.0,
                    json.dumps({"id": f"risk-win-{index}"}),
                )
                for index in range(12)
            ]
            rows.extend(
                (
                    "trades",
                    "2026-04-01",
                    "EUR_JPY",
                    "LONG",
                    -5.0,
                    json.dumps({"id": f"risk-loss-{index}"}),
                )
                for index in range(8)
            )
            rows.extend(
                [
                    (
                        "trades",
                        "2026-04-02",
                        "EUR_JPY",
                        "LONG",
                        50.0,
                        json.dumps({"id": "base-validation"}),
                    ),
                    (
                        "trades",
                        "2026-04-02",
                        "GBP_JPY",
                        "LONG",
                        70.0,
                        json.dumps({"id": "theme-validation"}),
                    ),
                ]
            )
            _seed_db(db, rows)

            AITestBotBacktester(
                db_path=db,
                output_path=root / "ai_backtest.json",
                report_path=root / "ai_backtest.md",
                max_loss_jpy=100.0,
                training_days=1,
                min_train_trades=1,
                max_active_buckets=1,
                source_tables=("trades",),
            ).run(start_balance_jpy=1000.0)

            payload = json.loads((root / "ai_backtest.json").read_text())
            day = payload["days"][0]
            self.assertEqual(day["selected_buckets"], ["trades:EUR_JPY:LONG:UNSPECIFIED:UNSPECIFIED"])
            self.assertEqual(day["selected_trades"], 1)
            self.assertEqual(day["managed_net_jpy"], 50.0)
            self.assertEqual(payload["context_theme_policy"]["min_train_win_rate_pct"], 65.0)


def _seed_db(path: Path, rows: list[tuple[str, str, str, str, float] | tuple[str, str, str, str, float, str]]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE legacy_records (
                source_table TEXT NOT NULL,
                source_id TEXT,
                session_date TEXT,
                pair TEXT,
                direction TEXT,
                pl REAL,
                execution_style TEXT,
                allocation_band TEXT,
                thesis TEXT,
                raw_json TEXT NOT NULL
            )
            """
        )
        expanded = []
        for row in rows:
            source, session_date, pair, direction, pl = row[:5]
            raw_json = row[5] if len(row) == 6 else "{}"
            expanded.append((source, session_date, pair, direction, pl, raw_json))
        conn.executemany(
            """
            INSERT INTO legacy_records(
                source_table, session_date, pair, direction, pl, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            expanded,
        )


def _seed_execution_ledger(
    path: Path,
    rows: list[tuple[str, str, str, str, str, str, int | None, float | None, str]] | None = None,
    attributed_trade_ids: set[str] | None = None,
) -> None:
    if rows is None:
        rows = [
            (
                "fill-1",
                "2026-06-01T01:00:00Z",
                "ORDER_FILLED",
                "trade-1",
                "GBP_USD",
                "LONG",
                1000,
                None,
                "{}",
            ),
            (
                "close-1",
                "2026-06-01T03:00:00Z",
                "TRADE_CLOSED",
                "trade-1",
                "GBP_USD",
                "SHORT",
                1000,
                100.0,
                json.dumps({"reason": "STOP_LOSS_ORDER"}),
            ),
            (
                "fill-2",
                "2026-06-02T01:00:00Z",
                "ORDER_FILLED",
                "trade-2",
                "GBP_USD",
                "LONG",
                1000,
                None,
                "{}",
            ),
            (
                "close-2",
                "2026-06-02T03:00:00Z",
                "TRADE_CLOSED",
                "trade-2",
                "GBP_USD",
                "SHORT",
                1000,
                150.0,
                json.dumps({"reason": "TAKE_PROFIT_ORDER"}),
            ),
        ]
    if attributed_trade_ids is None:
        attributed_trade_ids = {
            trade_id
            for _, _, event_type, trade_id, _, _, _, _, _ in rows
            if event_type == "ORDER_FILLED"
        }
    expanded_rows = []
    for event_uid, ts_utc, event_type, trade_id, pair, side, units, realized_pl_jpy, raw_json in rows:
        order_id = f"order-{trade_id}" if event_type == "ORDER_FILLED" else f"close-order-{trade_id}"
        if event_type == "ORDER_FILLED" and trade_id in attributed_trade_ids:
            lane_id = f"trend_trader:{pair}:{side}:TREND_CONTINUATION"
            expanded_rows.append(
                (
                    f"gateway-{event_uid}",
                    ts_utc,
                    "GATEWAY_ORDER_SENT",
                    lane_id,
                    order_id,
                    trade_id,
                    pair,
                    side,
                    units,
                    None,
                    json.dumps({"lane_id": lane_id}),
                )
            )
        expanded_rows.append(
            (
                event_uid,
                ts_utc,
                event_type,
                None,
                order_id,
                trade_id,
                pair,
                side,
                units,
                realized_pl_jpy,
                raw_json,
            )
        )
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
                event_uid TEXT,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                realized_pl_jpy REAL,
                raw_json TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO execution_events (
                event_uid, ts_utc, event_type, lane_id, order_id, trade_id, pair, side, units,
                realized_pl_jpy, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            expanded_rows,
        )


def _seed_broker_attributed_ledger(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
                event_uid TEXT,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                realized_pl_jpy REAL,
                exit_reason TEXT,
                raw_json TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO execution_events (
                event_uid, ts_utc, event_type, lane_id, order_id, trade_id, pair, side, units,
                realized_pl_jpy, exit_reason, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "accepted-broker",
                    "2026-06-01T01:00:00Z",
                    "ORDER_ACCEPTED",
                    "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                    "order-broker",
                    None,
                    "EUR_USD",
                    "LONG",
                    1000,
                    None,
                    None,
                    "{}",
                ),
                (
                    "fill-broker",
                    "2026-06-01T01:01:00Z",
                    "ORDER_FILLED",
                    None,
                    "order-broker",
                    "broker-trade",
                    "EUR_USD",
                    "LONG",
                    1000,
                    None,
                    None,
                    "{}",
                ),
                (
                    "close-broker",
                    "2026-06-01T02:00:00Z",
                    "TRADE_CLOSED",
                    None,
                    "close-order-broker",
                    "broker-trade",
                    "EUR_USD",
                    "SHORT",
                    1000,
                    -250.0,
                    "MARKET_ORDER_TRADE_CLOSE",
                    json.dumps({"reason": "MARKET_ORDER_TRADE_CLOSE"}),
                ),
            ],
        )


def _seed_broker_accepted_close_ledger(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
                event_uid TEXT,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                realized_pl_jpy REAL,
                exit_reason TEXT,
                raw_json TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO execution_events (
                event_uid, ts_utc, event_type, lane_id, order_id, trade_id, pair, side, units,
                realized_pl_jpy, exit_reason, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "gateway-entry",
                    "2026-06-01T01:00:00Z",
                    "GATEWAY_ORDER_SENT",
                    "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                    "order-entry",
                    None,
                    "EUR_USD",
                    "LONG",
                    1000,
                    None,
                    None,
                    json.dumps({"lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"}),
                ),
                (
                    "fill-entry",
                    "2026-06-01T01:01:00Z",
                    "ORDER_FILLED",
                    None,
                    "order-entry",
                    "broker-close-trade",
                    "EUR_USD",
                    "LONG",
                    1000,
                    None,
                    None,
                    "{}",
                ),
                (
                    "accepted-close",
                    "2026-06-01T02:00:00Z",
                    "ORDER_ACCEPTED",
                    None,
                    "order-close",
                    None,
                    "EUR_USD",
                    "SHORT",
                    None,
                    None,
                    "TRADE_CLOSE",
                    json.dumps({"reason": "TRADE_CLOSE", "tradeClose": {"tradeID": "broker-close-trade"}}),
                ),
                (
                    "closed-trade",
                    "2026-06-01T02:00:01Z",
                    "TRADE_CLOSED",
                    None,
                    "order-close",
                    "broker-close-trade",
                    "EUR_USD",
                    "SHORT",
                    1000,
                    -250.0,
                    "MARKET_ORDER_TRADE_CLOSE",
                    json.dumps({"reason": "MARKET_ORDER_TRADE_CLOSE"}),
                ),
            ],
        )


def _seed_gateway_close_order_id_ledger(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
                event_uid TEXT,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                realized_pl_jpy REAL,
                exit_reason TEXT,
                raw_json TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO execution_events (
                event_uid, ts_utc, event_type, lane_id, order_id, trade_id, pair, side, units,
                realized_pl_jpy, exit_reason, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "gateway-entry",
                    "2026-06-01T01:00:00Z",
                    "GATEWAY_ORDER_SENT",
                    "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                    "order-entry",
                    None,
                    "EUR_USD",
                    "LONG",
                    1000,
                    None,
                    None,
                    json.dumps({"lane_id": "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"}),
                ),
                (
                    "fill-entry",
                    "2026-06-01T01:01:00Z",
                    "ORDER_FILLED",
                    None,
                    "order-entry",
                    "gateway-order-close-trade",
                    "EUR_USD",
                    "LONG",
                    1000,
                    None,
                    None,
                    "{}",
                ),
                (
                    "gateway-close-by-order",
                    "2026-06-01T02:00:00Z",
                    "GATEWAY_TRADE_CLOSE_SENT",
                    None,
                    "close-order-gateway",
                    None,
                    "EUR_USD",
                    None,
                    None,
                    None,
                    "REVIEW_EXIT",
                    json.dumps(
                        {
                            "management_action": "REVIEW_EXIT",
                            "request": {"type": "CLOSE", "trade_id": "gateway-order-close-trade"},
                        }
                    ),
                ),
                (
                    "closed-by-same-order",
                    "2026-06-01T02:00:01Z",
                    "TRADE_CLOSED",
                    None,
                    "close-order-gateway",
                    "gateway-order-close-trade",
                    "EUR_USD",
                    "SHORT",
                    1000,
                    -250.0,
                    "MARKET_ORDER_TRADE_CLOSE",
                    json.dumps({"reason": "MARKET_ORDER_TRADE_CLOSE"}),
                ),
            ],
        )


def _seed_stale_gpt_close_satisfied_ledger(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_events (
                event_uid TEXT,
                ts_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                lane_id TEXT,
                order_id TEXT,
                trade_id TEXT,
                pair TEXT,
                side TEXT,
                units INTEGER,
                realized_pl_jpy REAL,
                exit_reason TEXT,
                raw_json TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO execution_events (
                event_uid, ts_utc, event_type, lane_id, order_id, trade_id, pair, side, units,
                realized_pl_jpy, exit_reason, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "gateway-entry",
                    "2026-06-01T01:00:00Z",
                    "GATEWAY_ORDER_SENT",
                    "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                    "order-entry",
                    None,
                    "EUR_USD",
                    "LONG",
                    1000,
                    None,
                    None,
                    json.dumps({"lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION"}),
                ),
                (
                    "fill-entry",
                    "2026-06-01T01:01:00Z",
                    "ORDER_FILLED",
                    None,
                    "order-entry",
                    "stale-gpt-close-trade",
                    "EUR_USD",
                    "LONG",
                    1000,
                    None,
                    None,
                    "{}",
                ),
                (
                    "accepted-close",
                    "2026-06-01T02:00:00Z",
                    "ORDER_ACCEPTED",
                    None,
                    "close-order",
                    "stale-gpt-close-trade",
                    "EUR_USD",
                    "SHORT",
                    1000,
                    None,
                    "TRADE_CLOSE",
                    json.dumps({"reason": "TRADE_CLOSE", "tradeClose": {"tradeID": "stale-gpt-close-trade"}}),
                ),
                (
                    "closed-trade",
                    "2026-06-01T02:00:01Z",
                    "TRADE_CLOSED",
                    None,
                    "close-order",
                    "stale-gpt-close-trade",
                    "EUR_USD",
                    "LONG",
                    1000,
                    -250.0,
                    "MARKET_ORDER_TRADE_CLOSE",
                    json.dumps({"reason": "MARKET_ORDER_TRADE_CLOSE"}),
                ),
                (
                    "stale-close-satisfied",
                    "2026-06-01T02:02:00Z",
                    "GATEWAY_POSITION_NO_ACTION",
                    None,
                    None,
                    "stale-gpt-close-trade",
                    None,
                    None,
                    None,
                    None,
                    "GPT_CLOSE",
                    json.dumps(
                        {
                            "management_action": "GPT_CLOSE",
                            "request": None,
                            "sent": False,
                            "issues": [
                                {
                                    "severity": "INFO",
                                    "code": "STALE_CLOSE_ALREADY_ABSENT",
                                    "message": "accepted CLOSE receipt named a trade id that is already absent",
                                }
                            ],
                        }
                    ),
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()

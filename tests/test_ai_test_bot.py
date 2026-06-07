from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quant_rabbit.ai_test_bot import AITestBotBacktester


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
            self.assertTrue(any("selected policy currently reaches 5%" in item for item in payload["action_items"]))
            report = (root / "ai_backtest.md").read_text()
            self.assertIn("## Target Band", report)
            self.assertIn("Selected-policy best return: `5.50%`", report)
            self.assertIn("`5.0%` target=`500` selected_hits=`1/1`", report)
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
            self.assertEqual(payload["bucket_contributions"][0]["bucket"], "trades:EUR_USD:LONG:UNSPECIFIED:UNSPECIFIED")
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

    def test_dedupes_same_trade_id_across_pretrade_and_trade_sources(self) -> None:
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
            self.assertNotIn("TAKE_PROFIT_ORDER", json.dumps(payload))
            report = (root / "ai_backtest.md").read_text()
            self.assertIn("sent gateway entry", report)
            self.assertIn("exit reason remains post-trade evidence", report)

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
            self.assertEqual(payload["context_feature_coverage"]["rows_with_context_theme_buckets"], 22)
            self.assertIn("Cross-pair context theme buckets", (root / "ai_backtest.md").read_text())


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


if __name__ == "__main__":
    unittest.main()
